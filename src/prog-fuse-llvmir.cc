/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

// Shared core of the prog-fuse optimization: the N-ary LLVM-IR kernel fusion
// routine command_graph_prog_fuse_llvmir(), used by BOTH the legacy POD pass
// (pod/prog-fuse.cc) and the MLIR pass (mlir/passes/ProgFuse.cpp).
//
// It links the LLVM modules of a chain of programs into a single
// `void __fused_wrapper(void ** args)` that calls each program in order, then
// JIT-compiles it. Two optimizations are applied while building the wrapper:
//
//   1. Argument deduplication. Each input program carries its argument pointers
//      in `command_prog_t::args` (one void* slot per kernel parameter, each
//      the address of the actual value). Parameters with the same identity are
//      merged into a single slot in the fused args buffer; the wrapper routes
//      them all to that slot. Identity is by slot ADDRESS (device/host leaf
//      kernels whose params already alias the same &value storage) or, for
//      programs flagged value_dedup (OpenMP leaf task bodies), by the
//      dereferenced VALUE+type: each task has its own storage for a firstprivate
//      but the values are equal across the fused bodies, so merging them into a
//      single hoisted load lets the fused loops share their trip counts/offsets
//      and fuse. The pass fills the (compacted) fused buffer at fusion time from
//      the originals' slots, so the args are known/stable for every replay.
//
//   2. noalias marking (restrict-like). For a kernel's pointer parameters, if
//      the actual pointer value (deref of the slot) is distinct from the kernel's
//      other pointer parameters, the parameter is marked `noalias`. Combined with
//      inlining the kernels into the wrapper and running an O3 pipeline, this lets
//      LLVM vectorize and (where legal) fuse the kernels' loops. Distinct pointer
//      args are ASSUMED non-overlapping; identical args are routed through one
//      slot (a single SSA load), so genuine cross-kernel dependencies are kept.

# include <cgir/namespace.hpp>
# include <cgir/command.hpp>

# include "prog-fuse-llvmir.hpp"

# if CGIR_SUPPORT_LLVM

# include <llvm/Analysis/InlineCost.h>
# include <llvm/Analysis/ValueTracking.h>
# include <llvm/IR/Function.h>
# include <llvm/IR/IRBuilder.h>
# include <llvm/IR/CallingConv.h>
# include <llvm/IR/Instructions.h>
# include <llvm/IR/IntrinsicInst.h>
# include <llvm/IR/LLVMContext.h>
# include <llvm/IR/MDBuilder.h>
# include <llvm/IR/Module.h>
# include <llvm/IR/Verifier.h>
# include <llvm/IRReader/IRReader.h>
# include <llvm/Linker/Linker.h>
# include <llvm/Transforms/Utils/Cloning.h>
# include <llvm/Bitcode/BitcodeWriter.h>
# include <llvm/MC/TargetRegistry.h>
# include <llvm/Support/FileSystem.h>
# include <llvm/Support/MemoryBuffer.h>
# include <llvm/Support/raw_ostream.h>
# include <llvm/Support/SourceMgr.h>
# include <llvm/Support/TargetSelect.h>
# include <llvm/Target/TargetMachine.h>
# include <llvm/TargetParser/Host.h>
# include <llvm/TargetParser/SubtargetFeature.h>  /* host feature string (JIT retarget) */
# include <llvm/TargetParser/Triple.h>
# include <llvm/Config/llvm-config.h>      /* LLVM_VERSION_STRING (disk cache salt) */

/* Lazy device-bitcode materialization (libdevice/DeviceRTL): parse only the
 * referenced functions instead of the whole (large) library. */
# include <llvm/Bitcode/BitcodeReader.h>

/* Optimization pipeline (inline + loop-fuse + vectorize) run on the fused
 * module before JIT, so noalias/dedup actually translate into fused loops. */
# include <llvm/Passes/PassBuilder.h>
# include <llvm/Passes/OptimizationLevel.h>
# include <llvm/Transforms/Scalar/LoopFuse.h>
# include <llvm/Transforms/Scalar/LoopPassManager.h>
# include <llvm/Transforms/Scalar/LoopRotation.h>
# include <llvm/Transforms/Utils/LoopSimplify.h>
/* Pre-fusion cleanup: promote the per-body reconstruction allocas and remove the
 * inter-loop cruft so LoopFuse sees adjacent loops accessing the shared base
 * pointers directly (see optimize_module). */
# include <llvm/Transforms/Scalar/SROA.h>
# include <llvm/Transforms/Scalar/SimplifyCFG.h>
# include <llvm/Transforms/InstCombine/InstCombine.h>
# include <llvm/Transforms/IPO/OpenMPOpt.h>    /* device SPMD-ization before PTX codegen */
# include <llvm/Transforms/IPO/Internalize.h> /* LTO-style internalize before O3 */

/* In-process JIT (replaces the former Proteus dependency) */
# include <llvm/ExecutionEngine/Orc/LLJIT.h>
# include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
# include <llvm/ExecutionEngine/Orc/Core.h>
# include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
# include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
# include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
# include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
# include <llvm/Support/CodeGen.h>
# include <llvm/Support/Error.h>

/* Device (GPU) code generation: emit PTX for a device target (nvptx64) via the
 * legacy codegen pass manager (addPassesToEmitFile). */
# include <llvm/IR/LegacyPassManager.h>
# include <llvm/ADT/SmallString.h>

# include <algorithm>
# include <atomic>
# include <cassert>
# include <chrono>
# include <cstdint>
# include <cstdio>
# include <cstdlib>
# include <cstring>
# include <memory>
# include <mutex>
# include <optional>
# include <string>
# include <unordered_map>
# include <utility>
# include <vector>

# include <sys/stat.h> /* stat (CGIR_JIT_STATS_CSV header-if-new) */
# include <unistd.h>   /* getpid (atomic temp-file names for the on-disk cache) */

# endif /* CGIR_SUPPORT_LLVM */

CGIR_NAMESPACE_USE;

# if CGIR_SUPPORT_LLVM

/* =============================================================================
 * This file holds seven concerns, in this order. They are marked with `SECTION`
 * banners; each is a seam along which the file is meant to be split once the
 * churn settles (the anonymous namespace below would become a small internal
 * header, and every new LLVM-consuming TU needs the -fno-rtti property that
 * CMakeLists sets on this one):
 *
 *   SECTION 1  profiling / cache-stat instrumentation      -> jit-profile
 *   SECTION 2  environment knobs + host machine identity   -> jit-env
 *   SECTION 3  content hashing, salts, two-level cache     -> jit-cache
 *   SECTION 4  generic LLVM IR utilities                   -> llvm-util
 *   SECTION 5  the prog-fuse pass                          -> stays here
 *   SECTION 6  device (NVPTX) code generation              -> jit-device
 *   SECTION 7  host (CPU) code generation + the JIT entry  -> jit-host
 *
 * Sections 1-3 are the natural first extraction: they are one contiguous
 * anonymous namespace with no dependency on sections 4-7.
 * ========================================================================== */

/* ---------------------------------------------------------------------------
 * SECTION 1 - opt-in JIT profiling / cache-stat instrumentation. Off (and
 * near-free) unless enabled via env, read once at first use:
 *   CGIR_JIT_TIMING=1       accumulate per-phase wall time, print at exit
 *   CGIR_JIT_CACHE_STATS=1  hash each JIT'd body, print distinct-vs-total
 *                           (how many compiles a content cache would skip)
 * A single report is emitted to stderr at static destruction.
 * ------------------------------------------------------------------------- */
namespace {

struct jit_profiler_t
{
    bool timing_on;      // collect per-phase timings
    bool timing_print;   // print the timing table to stderr at exit
    bool stats_on;       // collect cache outcomes
    bool stats_print;    // print cache-stats to stderr at exit
    std::string csv_path;// CGIR_JIT_STATS_CSV: append one machine-readable row at exit
    std::string tag;     // CGIR_STATS_TAG: joins the row to a caller's run

    std::mutex                                mtx;
    std::unordered_map<std::string, double>   phase_secs;    // bucket -> seconds
    std::unordered_map<std::string, uint64_t> phase_calls;   // bucket -> count

    // real cache outcomes, split [0]=host [1]=device
    uint64_t                                  compiled[2] = {0, 0};  // full compiles (misses)
    uint64_t                                  disk_reuse[2] = {0, 0}; // loaded from on-disk cache
    uint64_t                                  mem_reuse[2] = {0, 0};  // reused in-process artifact

    jit_profiler_t()
    {
        const char * t = getenv("CGIR_JIT_TIMING");
        const char * s = getenv("CGIR_JIT_CACHE_STATS");
        const char * c = getenv("CGIR_JIT_STATS_CSV");
        const char * g = getenv("CGIR_STATS_TAG");
        csv_path = (c && c[0] && strcmp(c, "0") != 0) ? c : "";
        tag      = g ? g : "";
        timing_print = (t && t[0] && strcmp(t, "0") != 0);
        stats_print  = (s && s[0] && strcmp(s, "0") != 0);
        // the CSV needs the data, so it implies collection (but not the stderr dump)
        const bool csv_on = !csv_path.empty();
        timing_on = timing_print || csv_on;
        stats_on  = stats_print  || csv_on;
    }

    void add_phase(const char * name, double secs)
    {
        std::lock_guard<std::mutex> lk(mtx);
        phase_secs[name]  += secs;
        phase_calls[name] += 1;
    }

    // outcome: 0 = full compile, 1 = disk reuse, 2 = in-process reuse
    void cache_event(bool device, int outcome)
    {
        std::lock_guard<std::mutex> lk(mtx);
        const int d = device ? 1 : 0;
        if      (outcome == 0) compiled[d]++;
        else if (outcome == 1) disk_reuse[d]++;
        else                   mem_reuse[d]++;
    }

    /* Canonical phase order for the CSV schema (kept in sync with the scoped_phase_t
     * bucket names). A phase absent from a run is written as 0. */
    static const std::vector<const char *> & csv_phases()
    {
        static const std::vector<const char *> P = {
            "jit-total", "jit-parse", "fuse-total",
            "host-optimize", "host-codegen", "host-link", "host-orc-create",
            "dev-emit-total", "dev-spmdize", "dev-link-read", "dev-link-parse",
            "dev-link-linkin", "dev-o3", "dev-ptx-emit",
        };
        return P;
    }

    /* Append one machine-readable row (mirrors CGIR_STATS_CSV): first column the
     * CGIR_STATS_TAG (joins to the caller's run), then per phase <name>_s seconds
     * and <name>_n calls, then split host/device cache counts. Header written once
     * when the file is new/empty. */
    void write_csv_row()
    {
        const auto & phases = csv_phases();
        static std::mutex io_mtx;
        std::lock_guard<std::mutex> lk(io_mtx);

        struct stat st;
        const bool need_header = (stat(csv_path.c_str(), &st) != 0 || st.st_size == 0);
        FILE * f = fopen(csv_path.c_str(), "a");
        if (!f) return ;

        if (need_header)
        {
            fputs("tag", f);
            for (const char * p : phases)
            {
                std::string col(p);
                for (char & ch : col) if (ch == '-') ch = '_';
                fprintf(f, ",%s_s,%s_n", col.c_str(), col.c_str());
            }
            for (const char * cls : { "host", "device" })
                fprintf(f, ",%s_total,%s_compiled,%s_disk_reuse,%s_mem_reuse",
                        cls, cls, cls, cls);
            fputc('\n', f);
        }

        fputs(tag.c_str(), f);
        for (const char * p : phases)
        {
            double        sec   = 0.0;
            uint64_t      calls = 0;
            if (auto it = phase_secs.find(p);  it != phase_secs.end())  sec   = it->second;
            if (auto it = phase_calls.find(p); it != phase_calls.end()) calls = it->second;
            fprintf(f, ",%.6f,%llu", sec, (unsigned long long) calls);
        }
        for (int d = 0 ; d < 2 ; ++d)
            fprintf(f, ",%llu,%llu,%llu,%llu",
                    (unsigned long long) (compiled[d] + disk_reuse[d] + mem_reuse[d]),
                    (unsigned long long) compiled[d],
                    (unsigned long long) disk_reuse[d],
                    (unsigned long long) mem_reuse[d]);
        fputc('\n', f);
        fclose(f);
    }

    ~jit_profiler_t()
    {
        if (timing_print && !phase_secs.empty())
        {
            std::vector<std::pair<std::string, double>> v(phase_secs.begin(), phase_secs.end());
            std::sort(v.begin(), v.end(),
                      [] (const auto & a, const auto & b) { return a.second > b.second; });
            fprintf(stderr, "\n[cgir jit timing] %-24s %10s %14s\n", "phase", "calls", "seconds");
            fprintf(stderr,   "[cgir jit timing] ------------------------------------------------\n");
            for (auto & p : v)
                fprintf(stderr, "[cgir jit timing] %-24s %10llu %14.6f\n",
                        p.first.c_str(), (unsigned long long) phase_calls[p.first], p.second);
            fprintf(stderr, "[cgir jit timing] (buckets nest: e.g. jit-total covers the rest)\n");
        }
        if (stats_print)
        {
            for (int d = 0 ; d < 2 ; ++d)
            {
                const uint64_t total = compiled[d] + disk_reuse[d] + mem_reuse[d];
                if (!total)
                    continue ;
                fprintf(stderr,
                    "[cgir jit cache-stats] %-6s: %llu total, %llu compiled, "
                    "%llu disk-reuse, %llu mem-reuse (%.1f%% reused)\n",
                    d ? "device" : "host",
                    (unsigned long long) total, (unsigned long long) compiled[d],
                    (unsigned long long) disk_reuse[d], (unsigned long long) mem_reuse[d],
                    100.0 * (double) (disk_reuse[d] + mem_reuse[d]) / (double) total);
            }
        }
        // write the CSV row only for runs that actually JIT-compiled something
        if (!csv_path.empty() && !(phase_secs.empty() &&
            compiled[0] == 0 && disk_reuse[0] == 0 && mem_reuse[0] == 0 &&
            compiled[1] == 0 && disk_reuse[1] == 0 && mem_reuse[1] == 0))
            write_csv_row();
    }
};

static jit_profiler_t &
jit_prof(void)
{
    static jit_profiler_t p;
    return p;
}

/* RAII wall-clock timer adding its lifetime to a named bucket. Cost when
 * disabled: one cached bool test (no clock reads). */
struct scoped_phase_t
{
    const char *                          name;
    std::chrono::steady_clock::time_point t0;
    bool                                  on;

    explicit scoped_phase_t(const char * n) : name(n), on(jit_prof().timing_on)
    {
        if (on)
            t0 = std::chrono::steady_clock::now();
    }
    ~scoped_phase_t()
    {
        if (!on)
            return ;
        const auto t1 = std::chrono::steady_clock::now();
        jit_prof().add_phase(name, std::chrono::duration<double>(t1 - t0).count());
    }
};

/* ---------------------------------------------------------------------------
 * SECTION 2 - environment knobs and host machine identity.
 * ------------------------------------------------------------------------- */

/* True iff the env var `var` is set to a non-empty, non-"0" value. */
static bool
env_flag(const char * var)
{
    const char * s = getenv(var);
    return s && s[0] != '\0' && strcmp(s, "0") != 0;
}

/* Value of the env var `var`, or `dflt` when unset/empty. */
static const char *
env_str(const char * var, const char * dflt)
{
    const char * s = getenv(var);
    return (s && s[0]) ? s : dflt;
}

/* The machine this process is running on, as an LLVM (target-cpu, target-features)
 * pair. Stamped onto host modules before optimization (stamp_host_target_attrs)
 * and folded into the host cache key (jit_host_salt) -- one source for both, so a
 * cached object can never disagree with the attributes it was compiled under.
 * The feature list is sorted because StringMap iteration order is unspecified and
 * the string ends up in a hash. Computed once. */
static const std::pair<std::string, std::string> &
host_target_attrs(void)
{
    static const std::pair<std::string, std::string> attrs = [] {
        std::vector<llvm::StringRef> enabled;
        for (const auto & f : llvm::sys::getHostCPUFeatures())
            if (f.second)
                enabled.push_back(f.first());
        std::sort(enabled.begin(), enabled.end());

        llvm::SubtargetFeatures features;
        for (const llvm::StringRef & f : enabled)
            features.AddFeature(f, true);

        return std::make_pair(llvm::sys::getHostCPUName().str(), features.getString());
    }();
    return attrs;
}

/* `CGIR_JIT_DEVICE_NOALIAS` -- see the use site in command_graph_jit_llvmir. It
 * changes the emitted code, so it is also folded into the JIT cache key. */
static bool
device_assume_noalias_params(void)
{
    static const bool value = env_flag("CGIR_JIT_DEVICE_NOALIAS");
    return value;
}

/* ---------------------------------------------------------------------------
 * SECTION 3 - content hashing, toolchain salts and the two-level JIT result
 * cache (in-process + optional on-disk).
 * ------------------------------------------------------------------------- */

/* FNV-1a 64-bit over a byte range, chainable via `h` (seed the first call with
 * jit_fnv1a). Cheap content hash for the cache key and the cache-stat estimate. */
static uint64_t
jit_fnv1a_seed(uint64_t h, const void * data, size_t n)
{
    const unsigned char * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0 ; i < n ; ++i)
    {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t
jit_fnv1a(const void * data, size_t n)
{
    return jit_fnv1a_seed(1469598103934665603ULL, data, n);
}

/* Salt folded into every disk key so an incompatible toolchain/codegen change
 * never reads a stale artifact: LLVM version + this TU's build timestamp (so any
 * rebuild of the cgir JIT codegen auto-invalidates the on-disk cache) + a manual
 * format version. Computed once. (clang task-IR changes alter the hashed IR bytes,
 * so they invalidate on their own.) */
static uint64_t
jit_static_salt(void)
{
    static const int      CGIR_JIT_CACHE_FORMAT = 1;          // bump to force-invalidate
    static const char     build_id[] = __DATE__ " " __TIME__; // cgir codegen rebuild
    static const uint64_t s = [] {
        uint64_t h = jit_fnv1a_seed(
            jit_fnv1a_seed(
                jit_fnv1a(LLVM_VERSION_STRING, strlen(LLVM_VERSION_STRING)),
                build_id, sizeof(build_id)),
            &CGIR_JIT_CACHE_FORMAT, sizeof(CGIR_JIT_CACHE_FORMAT));
        /* codegen-affecting knobs must be part of the key, or the on-disk cache
         * would hand back PTX built under a different assumption */
        const char noalias = device_assume_noalias_params() ? 1 : 0;
        return jit_fnv1a_seed(h, &noalias, sizeof(noalias));
    }();
    return s;
}

/* Host codegen depends on the host CPU; a cached object is only valid on a
 * matching CPU. Computed once. */
static uint64_t
jit_host_salt(void)
{
    static const uint64_t s = [] {
        /* The host CPU *and* its feature set: both are stamped onto the module
         * before optimization (stamp_host_target_attrs), so a cached object is
         * only valid on a machine that reports the same ones. The code model is
         * in here too, being a codegen-affecting knob. */
        const auto & attrs = host_target_attrs();
        uint64_t h = jit_static_salt();
        h = jit_fnv1a_seed(h, attrs.first.data(),  attrs.first.size());
        h = jit_fnv1a_seed(h, attrs.second.data(), attrs.second.size());
        const char * cm = env_str("CGIR_JIT_HOST_CODE_MODEL", "small");
        h = jit_fnv1a_seed(h, cm, strlen(cm) + 1);
        return h;
    }();
    return s;
}

/* Device PTX depends on the linked device libraries (DeviceRTL/libdevice); fold
 * each lib's identity (path + size + mtime) into the key. Stat is cached per path. */
static uint64_t
jit_device_lib_salt(const char * path)
{
    static std::mutex m;
    static std::unordered_map<std::string, uint64_t> cache;
    std::lock_guard<std::mutex> lk(m);
    auto it = cache.find(path);
    if (it != cache.end()) return it->second;
    uint64_t h = jit_fnv1a(path, strlen(path));
    llvm::sys::fs::file_status st;
    if (!llvm::sys::fs::status(path, st))
    {
        const uint64_t sz = st.getSize();
        const auto     mt = st.getLastModificationTime().time_since_epoch().count();
        h = jit_fnv1a_seed(h, &sz, sizeof(sz));
        h = jit_fnv1a_seed(h, &mt, sizeof(mt));
    }
    cache[path] = h;
    return h;
}

/* Content key for the JIT result cache: every byte/attribute a cached artifact
 * depends on. Two task instances of the same construct share the same IR bytes,
 * externs, proto and symbol, so they collide here (the intended reuse). Anything a
 * recompile would fold in (constant-propagated or not) is part of the IR bytes, so
 * a hit is byte-identical to recompiling -- caching never changes generated code.
 * The salt (LLVM/format version, host CPU, device-lib identity) makes disk keys
 * self-invalidating across incompatible toolchains. */
static uint64_t
jit_cache_key(const command_prog_t * prog)
{
    const auto & L = prog->source.content.llvmir;
    uint64_t h = jit_fnv1a(L.raw, L.size);
    h = jit_fnv1a_seed(h, &L.proto, sizeof(L.proto));
    if (L.symbol) h = jit_fnv1a_seed(h, L.symbol, strlen(L.symbol) + 1);
    if (L.triple) h = jit_fnv1a_seed(h, L.triple, strlen(L.triple) + 1);
    if (L.arch)   h = jit_fnv1a_seed(h, L.arch,   strlen(L.arch)   + 1);
    /* Extern *names* only. The address is bound at link time (the emitted object
     * relocates against the name), so it does not change the generated code --
     * and folding it in would make the key differ on every run under ASLR, which
     * silently defeats the on-disk cache for any program with externalized
     * globals. */
    h = jit_fnv1a_seed(h, &L.externs_count, sizeof(L.externs_count));
    for (size_t i = 0 ; L.externs && i < L.externs_count ; ++i)
        if (L.externs[i].name)
            h = jit_fnv1a_seed(h, L.externs[i].name, strlen(L.externs[i].name) + 1);
    if (L.triple)   // device: fold device-lib identity + static salt
    {
        for (size_t i = 0 ; L.device_libs && i < L.device_libs_count ; ++i)
            if (L.device_libs[i] && L.device_libs[i][0])
            {
                const uint64_t d = jit_device_lib_salt(L.device_libs[i]);
                h = jit_fnv1a_seed(h, &d, sizeof(d));
            }
        const uint64_t s = jit_static_salt();
        h = jit_fnv1a_seed(h, &s, sizeof(s));
    }
    else            // host: fold host-CPU salt
    {
        const uint64_t s = jit_host_salt();
        h = jit_fnv1a_seed(h, &s, sizeof(s));
    }
    return h;
}

/* Process-global JIT result cache. In-process (Phase 1): host compiled function
 * pointer + prototype (code kept mapped by the leaked LLJIT), device emitted PTX.
 * On-disk (Phase 2): host relocatable object (.o) and device PTX, so a later run
 * skips optimize/codegen. A hit reuses the artifact and skips the whole pipeline
 * for further instances of the same construct.
 *   CGIR_JIT_CACHE=0        disable all caching (mem + disk)
 *   CGIR_JIT_CACHE_DIR=DIR  enable the on-disk cache at DIR (opt-in; off by
 *                           default so a stale build is never silently reused
 *                           while the toolchain is under development)
 * Guarded by a mutex for a future parallel jit pass. */
struct jit_cache_t
{
    struct host_entry_t { int proto; void * fn; };

    bool                                       enabled;
    bool                                       disk_enabled = false;
    std::string                                dir;
    std::mutex                                 mtx;
    std::unordered_map<uint64_t, host_entry_t> host;        // in-process fn ptr
    std::unordered_map<uint64_t, std::string>  device_ptx;  // in-process PTX

    jit_cache_t()
    {
        const char * e = getenv("CGIR_JIT_CACHE");
        enabled = !(e && e[0] == '0' && e[1] == '\0');
        if (!enabled) return ;

        // on-disk cache is opt-in: only when CGIR_JIT_CACHE_DIR is set
        if (const char * cd = getenv("CGIR_JIT_CACHE_DIR"); cd && cd[0] &&
            !llvm::sys::fs::create_directories(cd))
        {
            dir = cd;
            disk_enabled = true;
        }
    }

    std::string path_for(uint64_t k, const char * ext) const
    {
        char name[40];
        snprintf(name, sizeof(name), "/%016llx.%s", (unsigned long long) k, ext);
        return dir + name;
    }

    static bool read_file(const std::string & p, std::string & out)
    {
        auto buf = llvm::MemoryBuffer::getFile(p, /*IsText*/ false);
        if (!buf) return false;
        out.assign((*buf)->getBufferStart(), (*buf)->getBufferSize());
        return true;
    }

    static void write_file_atomic(const std::string & p, const std::string & data)
    {
        static std::atomic<unsigned> seq{0};
        std::string tmp = p + ".tmp." + std::to_string((long) getpid()) + "." +
                          std::to_string(seq.fetch_add(1));
        {
            std::error_code ec;
            llvm::raw_fd_ostream f(tmp, ec, llvm::sys::fs::OF_None);
            if (ec) return ;
            f.write(data.data(), data.size());
            f.close();
            if (f.has_error()) { llvm::sys::fs::remove(tmp); return ; }
        }
        if (llvm::sys::fs::rename(tmp, p)) llvm::sys::fs::remove(tmp);
    }

    // ---- host: in-process compiled function pointer ----
    bool host_get_fn(uint64_t k, host_entry_t & out)
    {
        if (!enabled) return false;
        std::lock_guard<std::mutex> lk(mtx);
        auto it = host.find(k);
        if (it == host.end()) return false;
        out = it->second;
        return true;
    }
    void host_put_fn(uint64_t k, int proto, void * fn)
    {
        if (!enabled) return ;
        std::lock_guard<std::mutex> lk(mtx);
        host[k] = host_entry_t{proto, fn};
    }

    // ---- host: on-disk relocatable object (.o) ----
    bool host_get_obj(uint64_t k, std::string & obj)
    {
        if (!enabled || !disk_enabled) return false;
        return read_file(path_for(k, "o"), obj);
    }
    void host_put_obj(uint64_t k, const std::string & obj)
    {
        if (!enabled || !disk_enabled) return ;
        write_file_atomic(path_for(k, "o"), obj);
    }

    // ---- device: PTX (in-process, then disk). Returns 2=mem, 1=disk, 0=miss. ----
    int device_get(uint64_t k, std::string & out)
    {
        if (!enabled) return 0;
        {
            std::lock_guard<std::mutex> lk(mtx);
            auto it = device_ptx.find(k);
            if (it != device_ptx.end()) { out = it->second; return 2; }
        }
        if (disk_enabled && read_file(path_for(k, "ptx"), out))
        {
            std::lock_guard<std::mutex> lk(mtx);
            device_ptx[k] = out;
            return 1;
        }
        return 0;
    }
    void device_put(uint64_t k, const std::string & ptx)
    {
        if (!enabled) return ;
        { std::lock_guard<std::mutex> lk(mtx); device_ptx[k] = ptx; }
        if (disk_enabled) write_file_atomic(path_for(k, "ptx"), ptx);
    }
};

static jit_cache_t &
jit_cache(void)
{
    static jit_cache_t c;
    return c;
}

} // namespace

/* ---------------------------------------------------------------------------
 * SECTION 4 - generic LLVM IR utilities shared by the fuse pass and the JIT.
 * ------------------------------------------------------------------------- */

/**
 *  Parse an LLVM IR (textual .ll, NUL-terminated) or LLVM bitcode (binary) blob
 *  into an llvm::Module. Returns nullptr and prints a diagnostic on failure.
 *
 *  Textual IR coming from a C string usually counts a trailing NUL in `size`,
 *  which the LL parser does not want; bitcode is binary and may legitimately end
 *  in 0x00, so we must NOT blindly strip the last byte. We detect bitcode by its
 *  magic and only drop a trailing NUL for textual input.
 */
static std::unique_ptr<llvm::Module>
parse_llvmir(const char * ir, size_t size, llvm::LLVMContext & ctx)
{
    bool is_bitcode = false;
    if (size >= 4)
    {
        const unsigned char * b = reinterpret_cast<const unsigned char *>(ir);
        /* raw bitcode magic 'B' 'C' 0xC0 0xDE, or the bitcode wrapper magic */
        is_bitcode = (b[0] == 0x42 && b[1] == 0x43 && b[2] == 0xC0 && b[3] == 0xDE) ||
                     (b[0] == 0xDE && b[1] == 0xC0 && b[2] == 0x17 && b[3] == 0x0B);
    }

    size_t len = size;
    if (!is_bitcode && len > 0 && ir[len - 1] == '\0')
        len -= 1; /* drop the NUL terminator of textual IR */

    llvm::StringRef sr(ir, len);
    llvm::MemoryBufferRef buf(sr, "prog-fuse-input");
    llvm::SMDiagnostic diag;
    auto mod = llvm::parseIR(buf, diag, ctx);
    if (!mod)
    {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        diag.print("prog-fuse", os);
        fprintf(stderr, "prog-fuse: failed to parse LLVM IR:\n%s\n", msg.c_str());
    }
    return mod;
}

/**
 *  Return the first void-returning, non-declaration function defined in the
 *  module.  This is the "kernel" we want to call.  Returns nullptr if none.
 */
static llvm::Function *
find_kernel(llvm::Module & M)
{
    for (llvm::Function & F : M)
    {
        if (!F.isDeclaration() && F.getReturnType()->isVoidTy())
            return &F;
    }
    return nullptr;
}

/* True iff `fty` is the uniform program-launch shape `void(void**)`: a
 * void-returning function taking a single pointer argument. This distinguishes a
 * "packed" wrapper (the args[0]==tt kernel / the fused `__fused_wrapper`) from an
 * "unpacked" leaf kernel that takes one parameter per captured value. */
static bool
is_void_voidptr(const llvm::FunctionType * fty)
{
    return fty->getReturnType()->isVoidTy() &&
           fty->getNumParams() == 1 &&
           fty->getParamType(0)->isPointerTy();
}

/* True iff `fty` is the packed-buffer launch shape `void(void*, size_t)`: a
 * void-returning function taking a byte-buffer pointer and its size. */
static bool
is_void_ptr_size(const llvm::FunctionType * fty)
{
    return fty->getReturnType()->isVoidTy() &&
           fty->getNumParams() == 2 &&
           fty->getParamType(0)->isPointerTy() &&
           fty->getParamType(1)->isIntegerTy();
}

/* True iff `fty` is the nanos6 outline shape `void(void*, void*, void*)`: a
 * void-returning function taking (args block, device env, translation table).
 * See CGIR_COMMAND_PROG_SOURCE_PROTO_NANOS6_OUTLINE. */
static bool
is_void_ptr_ptr_ptr(const llvm::FunctionType * fty)
{
    return fty->getReturnType()->isVoidTy() &&
           fty->getNumParams() == 3 &&
           fty->getParamType(0)->isPointerTy() &&
           fty->getParamType(1)->isPointerTy() &&
           fty->getParamType(2)->isPointerTy();
}

/**
 *  Return the first externally-linked `void(ptr)` definition in the module.
 *  This is the packed-args entry of a program whose launcher consumes a single
 *  void** args block (e.g. an OpenMP outlined task body,
 *  `void .omp_task_entry.(void ** args)`). The IR closure serialised by the
 *  producer externalises exactly that entry (its callees stay internal), so the
 *  first external void(ptr) definition is the intended wrapper. Returns nullptr
 *  if none.
 */
static llvm::Function *
find_packed_wrapper(llvm::Module & M)
{
    for (llvm::Function & F : M)
    {
        if (F.isDeclaration() || !F.hasExternalLinkage())
            continue;
        if (is_void_voidptr(F.getFunctionType()))
            return &F;
    }
    return nullptr;
}

/**
 *  Return the first externally-linked, non-declaration function in the module.
 *  serializeClosureToBitcode() externalizes exactly the closure's entry (its
 *  callees stay internal), so for a task body serialized in leaf form this is
 *  the leaf kernel `void .omp_task_kernel.(<captured values...>)` — a
 *  void-returning function whose parameters are the individual captured values
 *  (NOT the void** packed form). Returns nullptr if none.
 */
static llvm::Function *
find_external_def(llvm::Module & M)
{
    for (llvm::Function & F : M)
        if (!F.isDeclaration() && F.hasExternalLinkage())
            return &F;
    return nullptr;
}

/* Unpack args[idx] (a void* == &value) to a value of type T:
 *   %slot  = getelementptr ptr, %args_ptr, idx
 *   %voidp = load ptr, %slot
 *   %val   = load T, %voidp
 * Shared by the fused `__fused_wrapper` and the JIT fallback wrapper, both of
 * which consume the uniform void** args buffer. */
static llvm::Value *
emit_load_arg(llvm::IRBuilder<> & b, llvm::Value * args_ptr, unsigned idx,
              llvm::Type * T)
{
    llvm::Type * ptr_ty = llvm::PointerType::getUnqual(b.getContext());
    llvm::Value * slot  = b.CreateGEP(
        ptr_ty, args_ptr, llvm::ConstantInt::get(b.getInt64Ty(), idx), "slot");
    llvm::Value * voidp = b.CreateLoad(ptr_ty, slot, "voidp");
    return b.CreateLoad(T, voidp, "argval");
}

/* One-time LLVM target/codegen initialization, shared by the fuse and jit
 * passes (both build a host TargetMachine / JIT). */
static void
ensure_llvm_initialized(void)
{
    static std::once_flag llvm_init_flag;
    std::call_once(llvm_init_flag, []() {
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();
    });
}

/**
 *  Rename every function definition in the module by prepending 'prefix'.
 *  We only rename definitions (not declarations) to avoid touching external
 *  references that might matter.
 */
static void
prefix_functions(llvm::Module & M, const std::string & prefix)
{
    std::vector<llvm::Function *> defs;
    for (llvm::Function & F : M)
        if (!F.isDeclaration())
            defs.push_back(&F);

    for (llvm::Function * F : defs)
        F->setName(prefix + F->getName().str());
}

/* Attach shared-domain scoped-noalias metadata to the fused wrapper's memory
 * accesses: one alias scope per distinct captured base pointer, with every data
 * access marked `noalias` against all the OTHER bases. This is the restrict-like
 * assumption (distinct captured base pointer => non-overlapping), consistent with
 * prog-fuse's dedup contract (the same array collapses to one slot, hence one base
 * load, so distinct base loads are distinct arrays).
 *
 * A "captured base" is a pointer LOADED from the args buffer -- the deduplicated
 * array base each kernel reads/writes through. We restrict tagging to loads/stores
 * of NON-pointer values (the actual array data) so the args-buffer pointer chasing
 * is skipped, and require the access's underlying object to be a load (the base),
 * so stack temporaries are left alone.
 *
 * MUST run AFTER SROA (see optimize_module): only then does each array access
 * resolve to its base load -- uniformly for the pointers (void**, base = the
 * &value's deref) and packed (void*,size_t, base = the pointer read from the
 * buffer) shapes. With it, DependenceAnalysis can fuse the loops and the loop
 * vectorizer needs no runtime alias check. */
static void
tag_noalias_domains(llvm::Function & F)
{
    llvm::LLVMContext & ctx = F.getContext();
    llvm::MDBuilder mdb(ctx);
    llvm::MDNode * domain = mdb.createAnonymousAliasScopeDomain("cgir.prog-fuse");

    /* Only the deduplicated CAPTURE base loads carry the "cgir.fuse.base" marker
     * (stamped where they are hoisted, see command_graph_prog_fuse_llvmir). We tag
     * exactly those, never arbitrary loaded pointers: the restrict-like "distinct
     * base => no overlap" assumption is only valid for distinct captures (same
     * array collapses to one slot = one marked load), NOT for pointers a body loads
     * from its own data (e.g. pointer-array/linked-structure chasing), which could
     * genuinely alias. Missing the marker just leaves an access untagged
     * (conservative), never mis-tagged. */
    const unsigned base_kind = ctx.getMDKindID("cgir.fuse.base");

    /* the deduplicated captured base of a taggable data access, or nullptr */
    auto data_base = [&] (llvm::Instruction & I) -> llvm::Value *
    {
        llvm::Value * ptr = llvm::getLoadStorePointerOperand(&I);
        if (ptr == nullptr)
            return nullptr;
        llvm::Type * ty = nullptr;
        if (auto * LI = llvm::dyn_cast<llvm::LoadInst>(&I))
            ty = LI->getType();
        else if (auto * SI = llvm::dyn_cast<llvm::StoreInst>(&I))
            ty = SI->getValueOperand()->getType();
        if (ty == nullptr || ty->isPointerTy())
            return nullptr; /* skip args-buffer pointer chasing */
        llvm::Value * base = llvm::getUnderlyingObject(ptr);
        auto * L = llvm::dyn_cast<llvm::LoadInst>(base);
        return (L && L->getMetadata(base_kind)) ? base : nullptr;
    };

    std::unordered_map<llvm::Value *, llvm::MDNode *> base_to_scope;
    std::vector<llvm::Metadata *>                     all_scopes;
    for (llvm::BasicBlock & BB : F)
        for (llvm::Instruction & I : BB)
        {
            llvm::Value * base = data_base(I);
            if (base == nullptr)
                continue;
            if (base_to_scope.find(base) == base_to_scope.end())
            {
                llvm::MDNode * sc = mdb.createAnonymousAliasScope(domain, "cgir.prog-fuse.base");
                base_to_scope[base] = sc;
                all_scopes.push_back(sc);
            }
        }

    /* With <2 distinct bases there is nothing to disambiguate. */
    if (all_scopes.size() < 2)
        return ;

    for (llvm::BasicBlock & BB : F)
        for (llvm::Instruction & I : BB)
        {
            llvm::Value * base = data_base(I);
            if (base == nullptr)
                continue;
            llvm::Metadata * own = base_to_scope[base];
            I.setMetadata(llvm::LLVMContext::MD_alias_scope,
                          llvm::MDNode::get(ctx, { own }));
            std::vector<llvm::Metadata *> others;
            others.reserve(all_scopes.size() - 1);
            for (llvm::Metadata * s : all_scopes)
                if (s != own)
                    others.push_back(s);
            I.setMetadata(llvm::LLVMContext::MD_noalias,
                          llvm::MDNode::get(ctx, others));
        }
}

/* Structural (by-value) equality of two constants. prog-fuse parses every kernel
 * module into ONE LLVMContext, which renames structurally-identical named struct
 * types (ConfigurationEnvironmentTy -> .2, .3, ...); identical launch configs are
 * then DISTINCT uniqued Constants, so a pointer compare is insufficient -- recurse
 * and compare integer field values instead. */
static bool
constant_value_equal(llvm::Constant * a, llvm::Constant * b)
{
    if (a == b)                                 return true;   // same uniqued constant
    if (a == nullptr || b == nullptr)           return false;
    if (a->isNullValue() && b->isNullValue())   return true;   // zero-inits of renamed types
    if (auto * ia = llvm::dyn_cast<llvm::ConstantInt>(a))
    {
        auto * ib = llvm::dyn_cast<llvm::ConstantInt>(b);
        return ib && ia->getValue() == ib->getValue();
    }
    auto * aa = llvm::dyn_cast<llvm::ConstantAggregate>(a);
    auto * ab = llvm::dyn_cast<llvm::ConstantAggregate>(b);
    if (!aa || !ab || aa->getNumOperands() != ab->getNumOperands())
        return false;
    for (unsigned i = 0 ; i < aa->getNumOperands() ; ++i)
        if (!constant_value_equal(aa->getOperand(i), ab->getOperand(i)))
            return false;
    return true;
}

/* Collapse the per-kernel OpenMP-device runtime brackets in a fused device kernel.
 * Inlining N device target-region kernels into one wrapper leaves N
 * `__kmpc_target_init`/`__kmpc_target_deinit` pairs, but a single launch must have
 * exactly ONE. For SPMD kernels (target_init returns -1 => every thread runs the
 * body) with an identical launch configuration, keep the FIRST init and the LAST
 * deinit and drop the inner ones: each removed init's result is replaced by -1 so
 * its "== -1 => body" branch falls through into the body, and each removed deinit
 * is erased. Returns false if the kernels are not the expected SPMD shape or their
 * launch configurations differ (caller must then NOT fuse them). Assumes the
 * inlined bodies appear in launch order (block layout order), which holds since the
 * wrapper calls them in order and we inline in place. */
static bool
collapse_device_kernel_brackets(llvm::Function & F)
{
    std::vector<llvm::CallInst *> inits, deinits;
    for (llvm::BasicBlock & BB : F)
        for (llvm::Instruction & I : BB)
            if (auto * CI = llvm::dyn_cast<llvm::CallInst>(&I))
                if (llvm::Function * cf = CI->getCalledFunction())
                {
                    if (cf->getName() == "__kmpc_target_init")   inits.push_back(CI);
                    else if (cf->getName() == "__kmpc_target_deinit") deinits.push_back(CI);
                }

    /* need N matched pairs (N == number of fused kernels >= 2) */
    if (inits.size() < 2 || inits.size() != deinits.size())
        return false;

    /* All kernels must share the launch configuration: the first field of
     * KernelEnvironmentTy (ConfigurationEnvironmentTy), which holds only integers
     * (exec-mode, thread/team bounds, reduction sizes). Compare it by VALUE
     * (constant_value_equal), not by Constant pointer: parsing the kernels into one
     * context renames the type per kernel (.2/.3), so identical configs are not the
     * same uniqued Constant. */
    auto config_of = [] (llvm::CallInst * init) -> llvm::Constant *
    {
        llvm::Value * env = init->getArgOperand(0)->stripPointerCasts();
        auto * G = llvm::dyn_cast<llvm::GlobalVariable>(env);
        if (!G || !G->hasInitializer())
            return nullptr;
        auto * CS = llvm::dyn_cast<llvm::ConstantStruct>(G->getInitializer());
        return (CS && CS->getNumOperands() >= 1) ? CS->getOperand(0) : nullptr;
    };
    llvm::Constant * cfg0 = config_of(inits[0]);
    if (cfg0 == nullptr)
        return false;
    for (size_t i = 1 ; i < inits.size() ; ++i)
        if (!constant_value_equal(config_of(inits[i]), cfg0))
            return false;

    /* Each init must be the SPMD pattern: its result feeds `icmp eq <res>, -1`. */
    auto is_spmd_init = [] (llvm::CallInst * init) -> bool
    {
        for (llvm::User * u : init->users())
            if (auto * ic = llvm::dyn_cast<llvm::ICmpInst>(u))
                if (ic->getPredicate() == llvm::CmpInst::ICMP_EQ)
                    if (auto * c = llvm::dyn_cast<llvm::ConstantInt>(ic->getOperand(1)))
                        if (c->isMinusOne())
                            return true;
        return false;
    };
    for (llvm::CallInst * init : inits)
        if (!is_spmd_init(init))
            return false;

    /* keep inits[0] and deinits[last]; drop the inner brackets */
    llvm::Type * i32 = llvm::Type::getInt32Ty(F.getContext());
    for (size_t i = 1 ; i < inits.size() ; ++i)
    {
        inits[i]->replaceAllUsesWith(llvm::ConstantInt::getSigned(i32, -1));
        inits[i]->eraseFromParent();
    }
    for (size_t i = 0 ; i + 1 < deinits.size() ; ++i)
        deinits[i]->eraseFromParent();
    return true;
}

/* Write `M` as textual IR to <dir>/<name> (defined below; forward-declared so the
 * optimize pipeline can dump the pre-LoopFuse wrapper for debugging). */
static void dump_module(const std::string & dir, const char * name, llvm::Module & M);

/* Run an O3 module pipeline (inlining + loop-fuse + vectorization) on the
 * merged module, so the inlined kernels' loops can vectorize/fuse. `dump_dir` is
 * the (possibly empty) CGIR_PROG_FUSE_DUMP directory: when set, the wrapper is
 * dumped after the SROA+noalias cleanup and BEFORE loop-fusion, so the exact IR
 * LoopFuse operates on can be inspected.
 *
 * `run_o3`: host chains run the final O3 here; DEVICE chains pass false and keep
 * only the fusion-specific loop work, deferring O3 to emit_device_ptx (which runs
 * after the DeviceRTL is linked, so O3 can inline the runtime). */
static void
optimize_module(llvm::Module & M, llvm::TargetMachine * tm, const std::string & dump_dir,
                bool run_o3 = true)
{
    llvm::PassBuilder PB(tm);

    llvm::LoopAnalysisManager     LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager    CGAM;
    llvm::ModuleAnalysisManager   MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    /* Explicitly canonicalize and fuse the loops BEFORE the O3 pipeline.
     *
     * Injecting LoopFuse through registerScalarOptimizerLateEPCallback did NOT
     * fuse them in practice (the fused module came out with two separate loop
     * nests), even though the very same pre-optimization module fuses cleanly
     * when loop-fusion is run explicitly. So we run it ourselves here: the
     * kernels are already inlined into the wrapper (step 6b) and carry the
     * shared-domain noalias metadata, so loop-simplify + loop-rotate + loop-fuse
     * reliably merges the kernels' loops into one. The subsequent O3 pipeline
     * then vectorizes the single fused loop (e.g. scale+axpy -> one axpby).
     *
     * NOTE on a precondition for fusion to actually fire (LLVM >= 23): LoopFuse
     * relies on DependenceAnalysis to prove the cross-loop dependences legal, and
     * its sibling-loop ("SameSD") reasoning only kicks in when the access address
     * recurrence is provably non-wrapping, i.e. the kernel's array GEPs are
     * `inbounds` (checkSubscript -> hasNoSignedWrap in DependenceAnalysis.cpp).
     * Compiler-emitted kernels (clang/libomptarget) always use `inbounds` GEPs.
     *
     * BUT LoopFuse also requires the loops to be ADJACENT and to access memory it
     * can relate. As inlined by step 6b (before any SROA), each kernel body still
     * rebuilds its captures through LOCAL allocas and its params carry
     * `llvm.experimental.noalias.scope.decl` intrinsics -- so consecutive loops
     * (a) access the shared arrays indirectly through distinct allocas (DA cannot
     * relate scale's y[i] to axpy's y[i]) and (b) are separated by the scope.decl
     * cruft (not adjacent). We therefore CLEAN UP first: SROA promotes the
     * reconstruction allocas so the accesses use the deduplicated base pointers
     * directly, and instcombine + simplify-cfg drop the now-dead scope decls and
     * empty blocks. THEN we attach the shared-domain scoped-noalias metadata
     * (tag_noalias_domains) -- now that each access resolves to its base load --
     * so DependenceAnalysis can prove the cross-loop dependences and the vectorizer
     * needs no runtime alias check; after which loop-simplify/rotate/fuse merge the
     * loops (e.g. scale+axpy -> one axpby) and the O3 pipeline vectorizes it. */
    {
        /* 1. cleanup: promote reconstruction allocas + remove inter-loop cruft */
        llvm::FunctionPassManager FPM1;
        FPM1.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
        FPM1.addPass(llvm::InstCombinePass());
        FPM1.addPass(llvm::SimplifyCFGPass());
        llvm::ModulePassManager MPM1;
        MPM1.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM1)));
        MPM1.run(M, MAM);

        /* 2. attach scoped-noalias now that accesses resolve to their base loads */
        if (llvm::Function * wrapper = M.getFunction("__fused_wrapper"))
            tag_noalias_domains(*wrapper);
        /* the tagging edited IR outside the pass managers; drop cached analyses so
         * the loop passes below see the new metadata. */
        MAM.invalidate(M, llvm::PreservedAnalyses::none());

        /* dump the exact IR loop-fusion will see (post cleanup + noalias, pre-fuse) */
        dump_module(dump_dir, "prefuse.ll", M);

        /* 3. canonicalize + rotate + fuse the loops */
        llvm::FunctionPassManager FPM2;
        FPM2.addPass(llvm::LoopSimplifyPass());
        llvm::LoopPassManager LPM;
        LPM.addPass(llvm::LoopRotatePass());
        FPM2.addPass(llvm::createFunctionToLoopPassAdaptor(std::move(LPM)));
        FPM2.addPass(llvm::LoopFusePass());
        llvm::ModulePassManager MPM2;
        MPM2.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM2)));
        MPM2.run(M, MAM);
    }

    if (run_o3)
    {
        llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
        MPM.run(M, MAM);
    }
}

/* ------------------------------------------------------------------------- *
 * Debug dump options, shared by the fuse and jit passes.                     *
 *                                                                           *
 * Controlled by an environment variable (read on each pass). When set (to    *
 * anything other than "" or "0"), each fusion / JIT writes its IR as textual  *
 * .ll files under <base>/<prefix>-<seq>/, so the transformation can be        *
 * visualized. <base> is ~/.cgir/tmp by default, or the env var's value when  *
 * it is an absolute path (so the output location can be overridden). <seq> is *
 * a per-process counter so concurrent runs do not clobber each other.        *
 *                                                                           *
 *   CGIR_PROG_FUSE_DUMP -> <base>/prog-fuse-<seq>/{input-<i>,merged,fused}.ll *
 *   CGIR_JIT_DUMP       -> <base>/jit-<seq>/{input,final}.ll                  *
 * ------------------------------------------------------------------------- */

/* True iff the dump env var `var` is set to a non-empty, non-"0" value. */
static bool
dump_enabled(const char * var)
{
    return env_flag(var);
}

/* Create (mkdir -p) a fresh dump directory <base>/<prefix>-<seq> and return its
 * path, or an empty string on failure (see the block comment above). */
static std::string
dump_make_dir(const char * var, const char * prefix)
{
    /* base directory: an absolute `var` value, else ~/.cgir/tmp */
    std::string base;
    const char * s = getenv(var);
    if (s && s[0] == '/')
    {
        base = s;
    }
    else
    {
        const char * home = getenv("HOME");
        base  = home ? home : ".";
        base += "/.cgir/tmp";
    }

    static std::atomic<unsigned> seq{0};
    char sub[48];
    snprintf(sub, sizeof(sub), "/%s-%u", prefix, seq.fetch_add(1));
    std::string dir = base + sub;

    if (std::error_code ec = llvm::sys::fs::create_directories(dir))
    {
        fprintf(stderr, "cgir: cannot create dump dir '%s': %s\n",
                dir.c_str(), ec.message().c_str());
        return std::string();
    }
    return dir;
}

/* Write `M` as textual IR to <dir>/<name>. No-op if `dir` is empty. */
static void
dump_module(const std::string & dir, const char * name, llvm::Module & M)
{
    if (dir.empty())
        return ;

    std::string path = dir + "/" + name;
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec)
    {
        fprintf(stderr, "cgir: cannot write '%s': %s\n", path.c_str(), ec.message().c_str());
        return ;
    }
    M.print(os, /* AssemblyAnnotationWriter */ nullptr);
}

# endif /* CGIR_SUPPORT_LLVM */

/* ---------------------------------------------------------------------------
 * SECTION 5 - the prog-fuse pass: merge a chain of programs into one.
 * ------------------------------------------------------------------------- */

void
CGIR_NAMESPACE::command_graph_prog_fuse_llvmir(
    command_prog_t ** progs,
    size_t n,
    command_prog_t * dst
) {
    # if !CGIR_SUPPORT_LLVM
    (void) progs; (void) n; (void) dst;
    fprintf(stderr, "prog-fuse: LLVM support not enabled (rebuild with -DUSE_LLVM=ON)\n");
    abort();
    # else
    assert(n >= 2);

    scoped_phase_t _fuse_total("fuse-total");

    /* ------------------------------------------------------------------ *
     * 0. One-time LLVM global initialisation                             *
     * ------------------------------------------------------------------ */
    ensure_llvm_initialized();

    /* Optional IR dumping for debugging (CGIR_PROG_FUSE_DUMP). When enabled, a
     * fresh per-fusion directory receives the input, merged and fused IR. */
    const bool  dump     = dump_enabled("CGIR_PROG_FUSE_DUMP");
    std::string dump_dir = dump ? dump_make_dir("CGIR_PROG_FUSE_DUMP", "prog-fuse") : std::string();

    /* ------------------------------------------------------------------ *
     * 1. Parse all N IR modules into the SAME LLVMContext                 *
     * ------------------------------------------------------------------ */
    auto ctx = std::make_unique<llvm::LLVMContext>();

    std::vector<std::unique_ptr<llvm::Module>> mods(n);
    for (size_t i = 0 ; i < n ; ++i)
    {
        if (progs[i]->source.type != COMMAND_PROG_SOURCE_TYPE_LLVMIR)
        {
            fprintf(stderr, "prog-fuse: program %zu is not LLVM IR\n", i);
            abort();
        }
        mods[i] = parse_llvmir(
            static_cast<const char *>(progs[i]->source.content.llvmir.raw),
            progs[i]->source.content.llvmir.size,
            *ctx
        );
        if (!mods[i]) { fprintf(stderr, "prog-fuse: failed to parse program %zu\n", i); abort(); }

        /* dump the original input IR (before prefixing/linking) */
        if (dump)
        {
            char name[32];
            snprintf(name, sizeof(name), "input-%zu.ll", i);
            dump_module(dump_dir, name, *mods[i]);
        }
    }

    /* ------------------------------------------------------------------ *
     * 2/3. Per input: identify the entry function and arity, snapshot its *
     *      argument slots (from command_prog_t::args), then prefix-rename  *
     *      its definitions so they do not clash on link.                  *
     *                                                                     *
     *  An entry is the program's kernel (first void definition). A program *
     *  that is itself a previously-fused module instead exposes a          *
     *  `void __fused_wrapper(void**)` entry; its arity comes from the      *
     *  recorded command_prog_t::n_args and it is invoked on a slice.       *
     * ------------------------------------------------------------------ */
    struct fuse_input_t
    {
        std::string          fused_name;  /* entry name after prefixing */
        bool                 is_wrapper;  /* true if entry is void(void**) */
        /* true if entry is a nanos6 outline void(void*, void*, void*): fused at
         * the program level. Each constituent is called with its per-instance
         * (args, dev, translation); no per-slot args are consumed (no dedup). */
        bool                 is_outline;
        /* true if entry is a packed self-contained body void(void*, size_t): its
         * arg slots (&value) come from the params table (one per capture), not the
         * IR parameters (which are just the buffer pointer + size). Fusion
         * reconstructs each such body's own packed buffer from the deduplicated
         * fused buffer (at the params' offsets) and calls it. */
        bool                 is_packed_leaf;
        size_t               buf_size;    /* packed-leaf: this body's buffer size */
        unsigned             arity;       /* number of arg slots consumed */
        std::vector<void *>  slots;       /* the originals' arg slots (&value) */
        /* Deduplicate leaf slots by the DEREFERENCED value (not by slot address)
         * for programs whose arg slots are per-task copies of a shared value
         * (OpenMP leaf task bodies: each task has its own storage for a
         * firstprivate, but the values are equal across the fused bodies). This
         * merges them into one hoisted load so the fused loops share their trip
         * count / offsets and LoopFuse can fire. Safe under the same frozen-
         * taskgraph assumption replay already relies on (the recorded slots'
         * values do not change between replays). Address dedup (false) is kept
         * for device/host leaf kernels whose slots already alias by identity. */
        bool                 value_dedup; /* dedup leaf slots by value, not addr */
        std::vector<llvm::Type *> ptypes; /* leaf param types (for width/type key) */
        /* Per-parameter kind+size from the source (cgir_command_prog_param_t), when
         * the compiler forwarded them. When present, dedup is EXPLICIT: a REFERENCE
         * (shared) slot dedups by its pointer value, a COPY (firstprivate) slot by
         * memcmp over `size` bytes (any size). NULL => fall back to `ptypes`/width. */
        const command_prog_param_t * params;
        size_t                       param_count;
    };
    std::vector<fuse_input_t> inputs(n);

    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Module & M = *mods[i];

        llvm::Function * entry = nullptr;
        bool is_wrapper = false;
        bool is_outline = false;
        bool is_packed_leaf = false;
        size_t buf_size = 0;
        unsigned arity = 0;
        bool value_dedup = false;

        /* A task-spawn body comes in one of three forms, distinguished by the
         * entry signature:
         *  - unpacked leaf (fusable): `void .omp_task_kernel.(<captured values...>)`,
         *    one parameter per captured value, so its loops can be fused. Deduped
         *    by VALUE (each task has its own storage, so slot addresses differ
         *    across bodies even when values are equal; see value_dedup).
         *  - packed leaf (fusable): `void .omp_task_kernel.(void*, size_t)`, a
         *    self-contained body reading its captures from a packed buffer at the
         *    params table's offsets. Its arg slots come from the params table (one
         *    &value per capture); fusion reconstructs each body's buffer from the
         *    deduplicated fused buffer and calls it (see is_packed_leaf).
         *  - packed wrapper: the `void(void**)` kernel consuming the whole args
         *    block (args[0] == kmp_task_t*); fuses only at the PROGRAM level. */
        if (progs[i]->launch_mode == CGIR_COMMAND_PROG_LAUNCH_MODE_TASK_SPAWN)
        {
            if (const char * sym = progs[i]->source.content.llvmir.symbol)
                entry = M.getFunction(sym);
            if (entry == nullptr)
                entry = M.getFunction("__fused_wrapper");
            if (entry == nullptr)
                entry = find_packed_wrapper(M);   /* packed void(void**) kernel */
            if (entry == nullptr)
                entry = find_external_def(M);      /* unpacked/packed .omp_task_kernel. */
            if (!entry || entry->isDeclaration())
            {
                fprintf(stderr, "prog-fuse: no entry found in task-spawn program %zu\n", i);
                abort();
            }

            llvm::FunctionType * fty = entry->getFunctionType();
            const command_prog_param_t * params = progs[i]->source.content.llvmir.params;
            if (progs[i]->source.content.llvmir.proto == CGIR_COMMAND_PROG_SOURCE_PROTO_NANOS6_OUTLINE)
            {
                /* nanos6 outline void(args, dev, translation): fused at the program
                 * level (each constituent called with its per-instance pointers),
                 * so it consumes no per-slot args and needs no value dedup. */
                if (!is_void_ptr_ptr_ptr(fty))
                {
                    fprintf(stderr, "prog-fuse: nanos6 outline program %zu entry '%s' is not "
                                    "void(void*,void*,void*)\n", i, entry->getName().str().c_str());
                    abort();
                }
                is_outline = true;
                arity      = 0;
            }
            else if (is_void_voidptr(fty))
            {
                is_wrapper = true;
                arity      = (unsigned) progs[i]->n_args;
            }
            else if (is_void_ptr_size(fty))
            {
                /* packed self-contained body: slots + layout come from the params
                 * table. A packed body without a params table is a previously-fused
                 * packed program (prog-fuse clears params); re-fusing that is not
                 * supported (it consumes a whole buffer, not per-slot values). */
                if (params == nullptr)
                {
                    fprintf(stderr, "prog-fuse: cannot re-fuse packed fused program %zu "
                                    "(no params table)\n", i);
                    abort();
                }
                is_packed_leaf = true;
                arity          = (unsigned) progs[i]->source.content.llvmir.param_count;
                /* this body's own packed buffer size = max(offset + size) */
                for (unsigned k = 0 ; k < arity ; ++k)
                {
                    const size_t end = params[k].offset + params[k].size;
                    if (end > buf_size) buf_size = end;
                }
                if (buf_size == 0) buf_size = 1;
            }
            else
            {
                /* unpacked leaf: one slot per captured value, deduplicated by value */
                is_wrapper  = false;
                arity       = fty->getNumParams();
                value_dedup = true;
            }
        }

        /* prefer the explicit entry symbol if the source carries one (e.g. a
         * device kernel sub-module that defines several functions) */
        if (entry == nullptr && progs[i]->source.content.llvmir.symbol)
        {
            const char * sym = progs[i]->source.content.llvmir.symbol;
            entry = M.getFunction(sym);
            if (!entry || entry->isDeclaration())
            {
                fprintf(stderr, "prog-fuse: entry symbol '%s' not found in program %zu\n", sym, i);
                abort();
            }
            is_wrapper = false;
            arity      = entry->getFunctionType()->getNumParams();
            /* device (GPU) target-region kernels: deduplicate their kernelParams by
             * VALUE so a shared mapped array / equal scalar is passed once to the
             * fused kernel (their per-launch kernelParams live at distinct host
             * addresses, so address dedup would not merge them). */
            if (progs[i]->source.content.llvmir.triple)
                value_dedup = true;
        }

        if (entry == nullptr)
        {
            if (llvm::Function * w = M.getFunction("__fused_wrapper"))
            {
                if (!w->isDeclaration() && is_void_voidptr(w->getFunctionType()))
                {
                    entry      = w;
                    is_wrapper = true;
                    arity      = (unsigned) progs[i]->n_args;
                }
            }
        }

        if (entry == nullptr)
        {
            entry = find_kernel(M);
            if (!entry) { fprintf(stderr, "prog-fuse: no kernel found in program %zu\n", i); abort(); }
            is_wrapper = false;
            arity      = entry->getFunctionType()->getNumParams();
        }

        /* snapshot the originals' argument slots (each is a void* = &value) */
        void ** av = progs[i]->args;
        if (av == nullptr && arity > 0)
        {
            fprintf(stderr, "prog-fuse: program %zu has no variadic args populated "
                            "(fusible progs must use the variadic launcher)\n", i);
            abort();
        }
        inputs[i].slots.resize(arity);
        for (unsigned k = 0 ; k < arity ; ++k)
            inputs[i].slots[k] = av ? av[k] : nullptr;

        const std::string name = entry->getName().str();
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "__fz%zu_", i);
        prefix_functions(M, prefix);

        inputs[i].fused_name     = std::string(prefix) + name;
        inputs[i].is_wrapper     = is_wrapper;
        inputs[i].is_outline     = is_outline;
        inputs[i].is_packed_leaf = is_packed_leaf;
        inputs[i].buf_size       = buf_size;
        inputs[i].arity          = arity;
        inputs[i].value_dedup    = value_dedup;
        inputs[i].params         = progs[i]->source.content.llvmir.params;
        inputs[i].param_count    = progs[i]->source.content.llvmir.param_count;

        /* for value-deduplicated leaf inputs, snapshot the parameter types so the
         * dedup key can read exactly the value bytes (and distinguish types) */
        if (value_dedup && !is_wrapper)
        {
            llvm::FunctionType * fty = entry->getFunctionType();
            inputs[i].ptypes.resize(arity);
            for (unsigned k = 0 ; k < arity ; ++k)
                inputs[i].ptypes[k] = fty->getParamType(k);
        }
    }

    /* ------------------------------------------------------------------ *
     * 3b. Deduplicate argument slots across all leaf kernels.             *
     *                                                                     *
     *  Two parameters that map to the SAME identity collapse to a single   *
     *  compacted index; `unique_slots` becomes the fused args buffer       *
     *  contents (in first-occurrence order, storing the FIRST slot address *
     *  seen). A wrapper input keeps a contiguous (non-deduplicated) block, *
     *  since it expects a void** slice.                                    *
     *                                                                      *
     *  Identity is either:                                                 *
     *   - by ADDRESS (default): two params sharing the same &value slot     *
     *     are the same variable (device/host leaf kernels).                *
     *   - by VALUE (value_dedup inputs): two params whose dereferenced      *
     *     value (and type) are equal collapse even when their slot          *
     *     addresses differ. OpenMP leaf task bodies need this: each task    *
     *     has its own storage for a firstprivate, but the value is equal    *
     *     across the fused bodies, and merging them into one hoisted load   *
     *     is what lets the fused loops share trip counts/offsets and fuse.  *
     * ------------------------------------------------------------------ */
    /* Packed-leaf fusion is homogeneous: if any input is a packed self-contained
     * body, they all must be (mixing shapes would need per-input ABIs the single
     * fused entry cannot express). Guaranteed by per-TU -fopenmp-task-jit-type. */
    bool any_packed_leaf = false;
    for (size_t i = 0 ; i < n ; ++i)
        any_packed_leaf |= inputs[i].is_packed_leaf;
    if (any_packed_leaf)
        for (size_t i = 0 ; i < n ; ++i)
            if (!inputs[i].is_packed_leaf)
            {
                fprintf(stderr, "prog-fuse: cannot mix packed and non-packed task "
                                "bodies in one fused chain (program %zu)\n", i);
                abort();
            }

    /* nanos6 outline fusion is likewise homogeneous: the fused entry has the
     * 3-array outline ABI, which cannot also express leaf/wrapper inputs. */
    bool any_outline = false;
    for (size_t i = 0 ; i < n ; ++i)
        any_outline |= inputs[i].is_outline;
    if (any_outline)
        for (size_t i = 0 ; i < n ; ++i)
            if (!inputs[i].is_outline)
            {
                fprintf(stderr, "prog-fuse: cannot mix nanos6 outline and non-outline "
                                "task bodies in one fused chain (program %zu)\n", i);
                abort();
            }

    /* Device (GPU) fusion: the chain's PROGs carry a device codegen target
     * (triple/arch). Homogeneous by construction (a device kernel only chains with
     * others on the same device via equal launch params). The fused entry is a
     * single `ptx_kernel` (see below) compiled to PTX by the jit pass. */
    const char * dev_triple = progs[0]->source.content.llvmir.triple;
    const char * dev_arch   = progs[0]->source.content.llvmir.arch;
    const bool   device     = (dev_triple != nullptr);
    if (device)
        for (size_t i = 0 ; i < n ; ++i)
            if (progs[i]->source.content.llvmir.triple == nullptr)
            {
                fprintf(stderr, "prog-fuse: cannot mix device and host programs in one "
                                "fused chain (program %zu)\n", i);
                abort();
            }

    std::vector<void *>                    unique_slots;
    std::vector<size_t>                    unique_slot_size; /* byte size per unique slot */
    std::vector<bool>                      unique_slot_is_ref; /* slot holds a pointer (REFERENCE) */
    std::unordered_map<std::string, unsigned> slot_to_index;
    std::vector<std::vector<unsigned>>     index_map(n);
    std::vector<unsigned>                  wrapper_block_start(n, 0);

    /* byte size of leaf slot (i,j) from its param descriptor (0 if unknown) */
    auto slot_size = [&] (size_t i, unsigned j) -> size_t
    {
        if (inputs[i].params != nullptr && j < inputs[i].param_count)
        {
            const command_prog_param_t & p = inputs[i].params[j];
            return (p.kind == CGIR_COMMAND_PROG_PARAM_COPY) ? p.size : sizeof(void *);
        }
        return 0;
    };

    /* true iff leaf slot (i,j) is a by-reference (pointer) parameter */
    auto slot_is_ref = [&] (size_t i, unsigned j) -> bool
    {
        return inputs[i].params != nullptr && j < inputs[i].param_count &&
               inputs[i].params[j].kind == CGIR_COMMAND_PROG_PARAM_REFERENCE;
    };

    /* Identity key for a leaf slot. When the compiler forwarded per-parameter
     * descriptors (kind+size), dedup is EXPLICIT: a REFERENCE (shared) slot by
     * its pointer value, a COPY (firstprivate) slot by memcmp over `size` bytes
     * (any size). Otherwise fall back to the value_dedup width heuristic ("V" +
     * type + bytes, scalars <=64b) and finally to "A" + slot address. */
    auto slot_key = [&] (size_t i, unsigned j) -> std::string
    {
        void * s = inputs[i].slots[j];
        if (s != nullptr && inputs[i].params != nullptr && j < inputs[i].param_count)
        {
            const command_prog_param_t & p = inputs[i].params[j];
            const size_t K = (p.kind == CGIR_COMMAND_PROG_PARAM_COPY)
                ? p.size : sizeof(void *);
            if (K > 0)
            {
                std::string key;
                key.push_back(p.kind == CGIR_COMMAND_PROG_PARAM_COPY ? 'C' : 'R');
                key.append(reinterpret_cast<const char *>(&K), sizeof(K));
                key.append(reinterpret_cast<const char *>(s), K);
                return key;
            }
        }
        if (inputs[i].value_dedup && s != nullptr && j < inputs[i].ptypes.size())
        {
            llvm::Type * Ty = inputs[i].ptypes[j];
            uint64_t width = (mods[i] && Ty)
                ? mods[i]->getDataLayout().getTypeStoreSize(Ty).getFixedValue()
                : 0;
            if (width > 0 && width <= 64)
            {
                std::string key;
                key.push_back('V');
                void * typ = (void *) Ty; /* types are uniqued in the shared ctx */
                key.append(reinterpret_cast<const char *>(&typ), sizeof(typ));
                key.append(reinterpret_cast<const char *>(s), (size_t) width);
                return key;
            }
        }
        std::string key;
        key.push_back('A');
        key.append(reinterpret_cast<const char *>(&s), sizeof(s));
        return key;
    };

    for (size_t i = 0 ; i < n ; ++i)
    {
        if (inputs[i].is_wrapper)
        {
            wrapper_block_start[i] = (unsigned) unique_slots.size();
            for (unsigned k = 0 ; k < inputs[i].arity ; ++k)
            {
                unique_slots.push_back(inputs[i].slots[k]);
                unique_slot_size.push_back(sizeof(void *));
                unique_slot_is_ref.push_back(true); /* void** slice slots are pointers */
            }
        }
        else
        {
            index_map[i].resize(inputs[i].arity);
            for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
            {
                std::string key = slot_key(i, j);
                auto it = slot_to_index.find(key);
                unsigned idx;
                if (it == slot_to_index.end())
                {
                    idx = (unsigned) unique_slots.size();
                    unique_slots.push_back(inputs[i].slots[j]);
                    unique_slot_size.push_back(slot_size(i, j));
                    unique_slot_is_ref.push_back(slot_is_ref(i, j));
                    slot_to_index[key] = idx;
                }
                else
                {
                    idx = it->second;
                }
                index_map[i][j] = idx;
            }
        }
    }

    const unsigned total_args = (unsigned) unique_slots.size();

    /* ------------------------------------------------------------------ *
     * 4. Link every module into the first one (the merged module).        *
     * ------------------------------------------------------------------ */
    std::unique_ptr<llvm::Module> mod_u = std::move(mods[0]);
    llvm::Linker linker(*mod_u);
    for (size_t i = 1 ; i < n ; ++i)
    {
        if (linker.linkInModule(std::move(mods[i])))
        {
            fprintf(stderr, "prog-fuse: linking program %zu failed\n", i);
            abort();
        }
    }

    /* ------------------------------------------------------------------ *
     * 5. Make the constituent functions inlinable.                        *
     *                                                                     *
     *  noalias is NOT marked on the kernel parameters here: when a callee  *
     *  with noalias params is inlined, the inliner clones its alias scopes *
     *  into a fresh domain per call site, so two kernels' pointers end up  *
     *  in unrelated domains and are NOT known to be mutually non-aliasing. *
     *  That defeats loop fusion (DependenceAnalysis cannot disambiguate    *
     *  e.g. scale's `y` from axpy's `x`). Instead we inline the kernels    *
     *  ourselves (step 6b) and then attach shared-domain scoped-noalias    *
     *  metadata to the inlined accesses, so all distinct base pointers are  *
     *  mutually non-aliasing in ONE domain.                               *
     * ------------------------------------------------------------------ */
    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Function * F = mod_u->getFunction(inputs[i].fused_name);
        if (!F)
        {
            fprintf(stderr, "prog-fuse: symbol '%s' missing after link\n", inputs[i].fused_name.c_str());
            abort();
        }

        /* fold the constituent into __fused_wrapper */
        F->setLinkage(llvm::GlobalValue::InternalLinkage);
        F->addFnAttr(llvm::Attribute::AlwaysInline);
        /* device kernels carry the `ptx_kernel` calling convention; as an inlined
         * internal helper it must be plain-callable, so neutralize the CC (the body
         * is unaffected; the single fused entry re-declares ptx_kernel). */
        if (device)
            F->setCallingConv(llvm::CallingConv::C);
    }

    /* ------------------------------------------------------------------ *
     * 6. Build the fused wrapper: void __fused_wrapper(void** args)        *
     *    Each kernel reads its (deduplicated) arg slots; a wrapper input   *
     *    is handed its contiguous slice.                                   *
     * ------------------------------------------------------------------ */
    llvm::LLVMContext & llvmctx = mod_u->getContext();
    llvm::Type * void_ty = llvm::Type::getVoidTy(llvmctx);
    llvm::Type * ptr_ty  = llvm::PointerType::getUnqual(llvmctx);
    llvm::Type * i8_ty   = llvm::Type::getInt8Ty(llvmctx);
    llvm::Type * i64_ty  = llvm::Type::getInt64Ty(llvmctx);
    const llvm::DataLayout & DL = mod_u->getDataLayout();

    /* Determine the value type of each deduplicated leaf slot from its first
     * use. Every kernel sharing a slot must read it as the same type (the slot
     * holds one &value); with opaque pointers, pointer args are all `ptr`, and
     * shared scalars (e.g. the length `n`) are consistent. Packed leaves carry no
     * per-slot IR types (their IR params are just buffer ptr + size); their slot
     * byte sizes come from unique_slot_size instead, so they are skipped here. */
    std::vector<llvm::Type *> slot_type(total_args, nullptr);
    for (size_t i = 0 ; i < n ; ++i)
    {
        if (inputs[i].is_wrapper || inputs[i].is_packed_leaf)
            continue;
        llvm::FunctionType * fty = mod_u->getFunction(inputs[i].fused_name)->getFunctionType();
        for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
        {
            const unsigned idx = index_map[i][j];
            llvm::Type *   T   = fty->getParamType(j);
            if (slot_type[idx] == nullptr)
                slot_type[idx] = T;
            else
                assert(slot_type[idx] == T && "deduplicated arg slot used with inconsistent types");
        }
    }

    /* Packed output ABI: emit the fused entry as `void(void*, size_t)` over a
     * single packed byte buffer -- each deduplicated slot inlined at a fixed
     * (naturally-aligned) offset -- instead of `void(void**)` over a pointer
     * array. Requested by -fopenmp-task-jit-type=packed (a PACKED_BUFFER source
     * proto), or forced via CGIR_PROG_FUSE_PACKED. Enabled only when every input
     * is a leaf (a nested void** sub-wrapper needs a pointer slice the buffer
     * cannot give). The dedup/inline/noalias logic is identical; only the arg
     * access shape and the recorded args buffer differ. */
    bool packed_output = dump_enabled("CGIR_PROG_FUSE_PACKED");
    for (size_t i = 0 ; i < n ; ++i)
        if (progs[i]->source.content.llvmir.proto ==
            CGIR_COMMAND_PROG_SOURCE_PROTO_PACKED_BUFFER)
            packed_output = true;
    /* packed self-contained bodies (void(void*,size_t)) can only be fused into a
     * packed buffer -- the reconstruction path requires the fused slot offsets. */
    if (any_packed_leaf)
        packed_output = true;
    for (size_t i = 0 ; packed_output && i < n ; ++i)
        if (inputs[i].is_wrapper)
            packed_output = false;
    /* device fusion uses an individual-parameter ptx_kernel (kernelParams ABI),
     * never the host void** packed byte-buffer shapes. */
    if (device)
        packed_output = false;
    /* nanos6 outline fusion has its own 3-array entry (not a packed byte buffer). */
    if (any_outline)
        packed_output = false;

    std::vector<size_t> slot_offset(total_args, 0);
    size_t packed_size = 0;
    if (packed_output)
    {
        for (unsigned k = 0 ; k < total_args ; ++k)
        {
            size_t sz, al;
            if (any_packed_leaf)
            {
                /* packed leaves: slot bytes come from the params table. Reference
                 * (pointer) slots are reconstructed with a typed load/store (to
                 * preserve pointer provenance so DependenceAnalysis can fuse the
                 * loops), so align them naturally; by-value copies go through memcpy
                 * and tolerate any alignment. */
                sz = unique_slot_size[k] ? unique_slot_size[k] : 1;
                al = unique_slot_is_ref[k] ? sizeof(void *) : 1;
            }
            else
            {
                llvm::Type * T = slot_type[k] ? slot_type[k] : ptr_ty;
                sz = (size_t) DL.getTypeStoreSize(T).getFixedValue();
                al = (size_t) DL.getABITypeAlign(T).value();
            }
            packed_size = (packed_size + al - 1) & ~(al - 1);
            slot_offset[k] = packed_size;
            packed_size += sz;
        }
        if (packed_size == 0)
            packed_size = 1;
    }

    if (device)
    {
        /* Device fusion (Level 1, program-level): build a single `ptx_kernel` over
         * the deduplicated kernel parameters, call each region kernel with its
         * subset, inline them, then collapse the per-kernel target_init/deinit
         * brackets into one. The jit pass codegens this to PTX. */
        std::vector<llvm::Type *> ptypes(total_args);
        for (unsigned k = 0 ; k < total_args ; ++k)
            ptypes[k] = slot_type[k] ? slot_type[k] : ptr_ty;
        llvm::FunctionType * wfty = llvm::FunctionType::get(void_ty, ptypes, false);
        llvm::Function * wrapper = llvm::Function::Create(
            wfty, llvm::GlobalValue::ExternalLinkage, "__fused_wrapper", mod_u.get());
        wrapper->setCallingConv(llvm::CallingConv::PTX_Kernel);

        /* carry the device kernel function attributes (kernel, nvvm.maxntid,
         * target-cpu/features) from a constituent; drop alwaysinline (this is the
         * launched entry, not an inline callee). */
        if (llvm::Function * proto = mod_u->getFunction(inputs[0].fused_name))
        {
            llvm::AttrBuilder ab(llvmctx, proto->getAttributes().getFnAttrs());
            ab.removeAttribute(llvm::Attribute::AlwaysInline);
            wrapper->addFnAttrs(ab);
        }
        /* distinct captured pointers are assumed non-overlapping (restrict) */
        for (unsigned k = 0 ; k < total_args ; ++k)
            if (ptypes[k]->isPointerTy())
                wrapper->addParamAttr(k, llvm::Attribute::NoAlias);

        llvm::BasicBlock * bb = llvm::BasicBlock::Create(llvmctx, "entry", wrapper);
        llvm::IRBuilder<> builder(bb);
        std::vector<llvm::CallInst *> kernel_calls;
        kernel_calls.reserve(n);
        for (size_t i = 0 ; i < n ; ++i)
        {
            llvm::Function * fn = mod_u->getFunction(inputs[i].fused_name);
            std::vector<llvm::Value *> call_args;
            call_args.reserve(inputs[i].arity);
            for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
                call_args.push_back(wrapper->getArg(index_map[i][j]));
            llvm::CallInst * ci = builder.CreateCall(fn->getFunctionType(), fn, call_args);
            ci->setCallingConv(fn->getCallingConv());
            kernel_calls.push_back(ci);
        }
        builder.CreateRetVoid();

        for (llvm::CallInst * ci : kernel_calls)
        {
            llvm::InlineFunctionInfo ifi;
            if (!llvm::InlineFunction(*ci, ifi).isSuccess())
            {
                fprintf(stderr, "prog-fuse: failed to inline a device kernel\n");
                abort();
            }
        }

        /* Transitively force-inline the kernels' whole helper chain into the
         * wrapper (ptx_kernel entry -> *_debug__ -> *_omp_outlined -> ...). Only the
         * entry was inlined above; the __kmpc_target_init/deinit brackets and the
         * compute loops live in those (noinline/optnone) callees, so they must be
         * pulled in here for collapse_device_kernel_brackets + loop-fusion to see
         * them. In SPMD mode the parallel region is a direct call, so every level is
         * inlinable; we stop at declarations (the __kmpc_* runtime) and intrinsics.
         * Iterate to a fixpoint, restarting the scan after each inline (which
         * invalidates iterators). */
        for (bool changed = true ; changed ; )
        {
            changed = false;
            for (llvm::BasicBlock & BB : *wrapper)
            {
                for (llvm::Instruction & I : BB)
                {
                    auto * ci = llvm::dyn_cast<llvm::CallInst>(&I);
                    if (ci == nullptr)
                        continue;
                    llvm::Function * callee = ci->getCalledFunction();
                    if (callee == nullptr || callee == wrapper ||
                        callee->isDeclaration() || callee->isIntrinsic())
                        continue;
                    llvm::InlineFunctionInfo ifi;
                    if (llvm::InlineFunction(*ci, ifi).isSuccess())
                    {
                        changed = true;
                        break;   /* iterators invalidated -- rescan from the top */
                    }
                }
                if (changed)
                    break;
            }
        }

        /* the inlined entries/helpers are now unused; drop the entries (the
         * internal helpers are removed by globalDCE in optimize_module). */
        for (size_t i = 0 ; i < n ; ++i)
        {
            llvm::Function * F = mod_u->getFunction(inputs[i].fused_name);
            if (F && F->use_empty())
                F->eraseFromParent();
        }

        /* collapse the N target_init/deinit brackets into one (SPMD + same config;
         * aborts if that precondition does not hold). */
        if (!collapse_device_kernel_brackets(*wrapper))
        {
            fprintf(stderr, "prog-fuse: device kernels are not fusible "
                            "(non-SPMD or differing launch configuration)\n");
            abort();
        }
    }
    else if (any_outline)
    {
        /* nanos6 outline chain (program-level fusion): build
         *   void __fused_wrapper(void** args_v, void** dev_v, void** transl_v)
         * that calls each constituent outline i with its per-instance triple
         *   (args_v[i], dev_v[i], transl_v[i])
         * then inlines them so the O3 pipeline can merge the bodies and fuse
         * loops where dependence analysis proves it legal. No argument dedup:
         * outlines consume whole per-instance pointers, not per-value slots. */
        llvm::FunctionType * wfty = llvm::FunctionType::get(
            void_ty, { ptr_ty, ptr_ty, ptr_ty }, false);
        llvm::Function * wrapper = llvm::Function::Create(
            wfty, llvm::GlobalValue::ExternalLinkage, "__fused_wrapper", mod_u.get());

        /* the three arrays are distinct allocations (from each other and from the
         * task data), so mark them noalias to free the per-instance slot loads. */
        wrapper->addParamAttr(0, llvm::Attribute::NoAlias);
        wrapper->addParamAttr(1, llvm::Attribute::NoAlias);
        wrapper->addParamAttr(2, llvm::Attribute::NoAlias);

        llvm::BasicBlock * bb = llvm::BasicBlock::Create(llvmctx, "entry", wrapper);
        llvm::IRBuilder<> builder(bb);

        auto load_slot = [&] (llvm::Value * base, size_t idx) -> llvm::Value *
        {
            llvm::Value * p = builder.CreateGEP(
                ptr_ty, base, llvm::ConstantInt::get(i64_ty, idx), "slot");
            return builder.CreateLoad(ptr_ty, p, "inst");
        };

        std::vector<llvm::CallInst *> kernel_calls;
        kernel_calls.reserve(n);
        for (size_t i = 0 ; i < n ; ++i)
        {
            llvm::Function * fn = mod_u->getFunction(inputs[i].fused_name);
            llvm::Value * a = load_slot(wrapper->getArg(0), i);
            llvm::Value * d = load_slot(wrapper->getArg(1), i);
            llvm::Value * t = load_slot(wrapper->getArg(2), i);
            kernel_calls.push_back(builder.CreateCall(fn->getFunctionType(), fn, { a, d, t }));
        }
        builder.CreateRetVoid();

        /* inline the outlines (they are AlwaysInline internal defs) so O3 sees one
         * body; O3's inliner then pulls in their single-use internal callees. */
        for (llvm::CallInst * ci : kernel_calls)
        {
            llvm::InlineFunctionInfo ifi;
            llvm::InlineResult ir = llvm::InlineFunction(*ci, ifi);
            if (!ir.isSuccess())
            {
                fprintf(stderr, "prog-fuse: failed to inline a nanos6 outline: %s\n",
                        ir.getFailureReason());
                abort();
            }
        }

        for (size_t i = 0 ; i < n ; ++i)
        {
            llvm::Function * F = mod_u->getFunction(inputs[i].fused_name);
            if (F && F->use_empty())
                F->eraseFromParent();
        }
    }
    else
    {
    llvm::FunctionType * wrapper_fty = packed_output
        ? llvm::FunctionType::get(void_ty, { ptr_ty, i64_ty }, false)
        : llvm::FunctionType::get(void_ty, { ptr_ty }, false);
    llvm::Function * wrapper = llvm::Function::Create(
        wrapper_fty, llvm::GlobalValue::ExternalLinkage, "__fused_wrapper", mod_u.get());

    /* The args buffer is a distinct allocation from the data the kernels read
     * and write, so mark it noalias: this frees the slot-table loads from being
     * clobbered by the kernels' stores, so they can be hoisted/CSE'd. (The data
     * arrays are kept independent of each other by the shared-domain scoped
     * noalias metadata attached in step 6b.) Both help loop-fusion/vectorization. */
    wrapper->addParamAttr(0, llvm::Attribute::NoAlias);

    llvm::BasicBlock * bb = llvm::BasicBlock::Create(llvmctx, "entry", wrapper);
    llvm::IRBuilder<> builder(bb);
    llvm::Value * args_ptr = wrapper->getArg(0);

    /* Mark a hoisted capture-base load so tag_noalias_domains (run post-SROA)
     * disambiguates ONLY genuine captures -- never pointers a body loads from its
     * own data. Only pointer-typed bases matter (scalars are not memory bases). */
    const unsigned base_kind = llvmctx.getMDKindID("cgir.fuse.base");
    auto mark_base = [&] (llvm::Value * v)
    {
        if (auto * L = llvm::dyn_cast<llvm::LoadInst>(v))
            L->setMetadata(base_kind, llvm::MDNode::get(llvmctx, {}));
    };

    /* Load slot `idx` as type T. void** form: args[idx] is a &value (double load).
     * packed form: the value is inline at args_ptr + slot_offset[idx] (one load). */
    auto load_arg = [&] (unsigned idx, llvm::Type * T) -> llvm::Value *
    {
        if (packed_output)
        {
            llvm::Value * p = builder.CreateGEP(
                i8_ty, args_ptr, llvm::ConstantInt::get(i64_ty, slot_offset[idx]), "slot");
            return builder.CreateLoad(T, p, "argval");
        }
        return emit_load_arg(builder, args_ptr, idx, T);
    };

    /* &args[off] : a void** slice, for a sub-wrapper call (void** form only) */
    auto slice_ptr = [&] (unsigned off) -> llvm::Value *
    {
        return builder.CreateGEP(ptr_ty, args_ptr,
                                 llvm::ConstantInt::get(i64_ty, off), "slice");
    };

    /* Load each used leaf slot ONCE, here at the wrapper entry, and reuse the
     * resulting SSA value for every kernel that shares it. Re-loading per kernel
     * (the previous behavior) produced distinct SSA values for shared args (e.g.
     * `y`, `n`), which defeated loop fusion: the inlined loops then had
     * non-identical SCEV trip counts and were separated by un-hoistable arg
     * loads (LoopFuse requires identical trip counts AND adjacent loops). With a
     * single hoisted load, the fused loops share the same trip count and base
     * pointers, and nothing sits between them. */
    std::vector<llvm::Value *> slot_value(total_args, nullptr);
    for (unsigned k = 0 ; k < total_args ; ++k)
        if (slot_type[k] != nullptr)
        {
            slot_value[k] = load_arg(k, slot_type[k]);
            if (slot_type[k]->isPointerTy())
                mark_base(slot_value[k]); /* a captured array/pointer base */
        }

    /* Packed leaves reconstruct their own buffer in a stack slot. Create ALL the
     * allocas up front (entry block, before any call) so they remain static and
     * promotable after we split the entry block by inlining the calls below. */
    std::vector<llvm::Value *> recon_buf(n, nullptr);
    if (any_packed_leaf)
        for (size_t i = 0 ; i < n ; ++i)
        {
            llvm::AllocaInst * a = builder.CreateAlloca(
                i8_ty, llvm::ConstantInt::get(i64_ty, inputs[i].buf_size), "recon");
            a->setAlignment(llvm::Align(16));
            recon_buf[i] = a;
        }

    /* Hoist ONE typed load per register-sized unique slot, shared by every body
     * that uses it (analogous to slot_value above): a reference slot as a typed
     * `ptr` (preserves provenance -- no inttoptr -- so DependenceAnalysis can fuse),
     * a by-value scalar copy (size 1/2/4/8) as an integer of that width. Each body's
     * reconstruction stores this single SSA value into its recon buffer, so after
     * SROA every body reads the SAME deduplicated value with NO per-body reload.
     * This matters for BOTH correctness (the scoped-noalias tagging must see one
     * base per array; two separate loads would look like distinct, wrongly-noalias
     * arrays) AND fusion: a per-body scalar reload (e.g. a firstprivate loop bound
     * `j`) otherwise lands in the block between the two loops and breaks LoopFuse's
     * adjacency check. Aggregate copies (other sizes) fall back to memcpy. */
    std::vector<llvm::Value *> slot_hoist(total_args, nullptr);
    if (any_packed_leaf)
        for (unsigned k = 0 ; k < total_args ; ++k)
        {
            llvm::Type * ht = nullptr;
            if (unique_slot_is_ref[k])
                ht = ptr_ty;
            else if (unique_slot_size[k] == 1 || unique_slot_size[k] == 2 ||
                     unique_slot_size[k] == 4 || unique_slot_size[k] == 8)
                ht = llvm::Type::getIntNTy(llvmctx, (unsigned) unique_slot_size[k] * 8);
            if (ht == nullptr)
                continue; /* aggregate copy -> reconstructed via memcpy below */
            llvm::Value * src = builder.CreateGEP(
                i8_ty, args_ptr, llvm::ConstantInt::get(i64_ty, slot_offset[k]), "fslot");
            slot_hoist[k] = builder.CreateAlignedLoad(
                ht, src, llvm::MaybeAlign(1),
                unique_slot_is_ref[k] ? "refbase" : "copyval");
            if (unique_slot_is_ref[k])
                mark_base(slot_hoist[k]); /* a captured array/pointer base */
        }

    std::vector<llvm::CallInst *> kernel_calls;
    kernel_calls.reserve(n);
    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Function * fn = mod_u->getFunction(inputs[i].fused_name);
        llvm::FunctionType * fty = fn->getFunctionType();

        if (inputs[i].is_wrapper)
        {
            /* a previously-fused sub-wrapper consumes a raw void** slice */
            kernel_calls.push_back(builder.CreateCall(fty, fn, { slice_ptr(wrapper_block_start[i]) }));
        }
        else if (inputs[i].is_packed_leaf)
        {
            /* Reconstruct this body's own packed buffer from the deduplicated
             * fused buffer: for each capture, copy its bytes from the fused slot
             * (args_ptr + slot_offset) to this body's layout (recon + params.offset),
             * then call the self-contained kernel(recon, size). After AlwaysInline
             * + SROA the recon alloca promotes away and the body reads the shared
             * (deduplicated) slot values directly.
             *
             * A register-sized slot (reference pointer or scalar copy) is copied
             * with a TYPED store of its shared hoisted value (slot_hoist), NOT a
             * byte memcpy: memcpy of a pointer is recovered by SROA as an integer
             * load + inttoptr (loses provenance, defeats DependenceAnalysis), and a
             * per-body memcpy of a scalar leaves a per-body reload that can break
             * LoopFuse adjacency. The typed shared value keeps provenance/`inbounds`
             * and gives every body ONE deduplicated SSA value, so scale's store
             * forwards to axpy's load and LoopFuse merges the loops. Aggregate copies
             * (not hoisted) keep memcpy. */
            llvm::Value * recon = recon_buf[i];
            for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
            {
                const command_prog_param_t & p = inputs[i].params[j];
                const unsigned uidx = index_map[i][j];
                llvm::Value * dst = builder.CreateGEP(
                    i8_ty, recon, llvm::ConstantInt::get(i64_ty, p.offset), "rslot");
                if (slot_hoist[uidx] != nullptr)
                {
                    /* store the shared hoisted value (typed => provenance / no reload) */
                    builder.CreateAlignedStore(slot_hoist[uidx], dst, llvm::MaybeAlign(1));
                }
                else
                {
                    /* aggregate copy: byte memcpy from the fused buffer */
                    llvm::Value * src = builder.CreateGEP(
                        i8_ty, args_ptr,
                        llvm::ConstantInt::get(i64_ty, slot_offset[uidx]), "fslot");
                    builder.CreateMemCpy(dst, llvm::MaybeAlign(1), src, llvm::MaybeAlign(1),
                                         (uint64_t) p.size);
                }
            }
            kernel_calls.push_back(builder.CreateCall(
                fty, fn, { recon, llvm::ConstantInt::get(i64_ty, inputs[i].buf_size) }));
        }
        else
        {
            std::vector<llvm::Value *> call_args;
            call_args.reserve(inputs[i].arity);
            for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
                call_args.push_back(slot_value[index_map[i][j]]);
            kernel_calls.push_back(builder.CreateCall(fty, fn, call_args));
        }
    }

    builder.CreateRetVoid();

    /* ------------------------------------------------------------------ *
     * 6b. Inline the kernels into the wrapper ourselves, then attach       *
     *     shared-domain scoped-noalias metadata to the inlined accesses.   *
     *                                                                     *
     *  Doing the inlining here (instead of leaving it to the O3 pipeline)  *
     *  lets us tag the memory accesses AFTER inlining with ONE alias       *
     *  domain: each distinct base pointer value gets its own scope, and    *
     *  every access is marked noalias against all the OTHER bases' scopes. *
     *  Pointers that share a base (e.g. a `y` used by several kernels) end  *
     *  up in the same scope, so their genuine dependence is preserved.     *
     *  This is the restrict-like assumption (distinct base => no overlap)  *
     *  applied across the whole fused body, which is what lets             *
     *  DependenceAnalysis disambiguate the kernels and LoopFuse fuse them. *
     * ------------------------------------------------------------------ */
    for (llvm::CallInst * ci : kernel_calls)
    {
        llvm::InlineFunctionInfo ifi;
        llvm::InlineResult ir = llvm::InlineFunction(*ci, ifi);
        if (!ir.isSuccess())
        {
            fprintf(stderr, "prog-fuse: failed to inline a kernel: %s\n", ir.getFailureReason());
            abort();
        }
    }

    /* Remove the inlined bodies' `llvm.experimental.noalias.scope.decl` intrinsics:
     * inlining copies each kernel's per-parameter scope declarations into the
     * wrapper, where they land BETWEEN the consecutive loops and break LoopFuse's
     * adjacency check. They are redundant here -- cross-kernel disambiguation is
     * provided by the shared-domain scoped-noalias metadata attached below -- so
     * dropping them is safe (the metadata on the loads/stores is untouched). */
    {
        std::vector<llvm::Instruction *> dead;
        for (llvm::BasicBlock & BB : *wrapper)
            for (llvm::Instruction & I : BB)
                if (auto * II = llvm::dyn_cast<llvm::IntrinsicInst>(&I))
                    if (II->getIntrinsicID() == llvm::Intrinsic::experimental_noalias_scope_decl)
                        dead.push_back(II);
        for (llvm::Instruction * I : dead)
            I->eraseFromParent();
    }

    /* drop the now-inlined (dead) constituent functions */
    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Function * F = mod_u->getFunction(inputs[i].fused_name);
        if (F && F->use_empty())
            F->eraseFromParent();
    }
    }   /* end host wrapper build (else !device) */

    /* Shared-domain scoped-noalias tagging (the restrict-like "distinct captured
     * base pointer => no overlap" assumption that lets DependenceAnalysis fuse and
     * the vectorizer skip runtime alias checks) is NOT done here: at this point the
     * kernels reach their captures through un-promoted reconstruction allocas
     * (packed leaves) or param stores, so the accesses' underlying object is not
     * yet the deduplicated base. It is applied in optimize_module() AFTER the SROA
     * cleanup, when every access resolves to its base load -- uniformly for the
     * pointers (void**) and packed (void*,size_t) shapes. See tag_noalias_domains(). */

    /* ------------------------------------------------------------------ *
     * 7. Stamp the host triple + data layout (parsed IR omits them) and    *
     *    create a host TargetMachine (also used by the opt pipeline).      *
     * ------------------------------------------------------------------ */
    std::unique_ptr<llvm::TargetMachine> tm;
    {
        /* Device chains keep the device triple (from the linked device IR) and use
         * the runtime device arch as the target-cpu; host chains use the process
         * triple + host cpu. */
        if (device)
            mod_u->setTargetTriple(llvm::Triple(dev_triple));
        else if (mod_u->getTargetTriple().empty())
            mod_u->setTargetTriple(llvm::Triple(llvm::sys::getProcessTriple()));

        const llvm::Triple & TT = mod_u->getTargetTriple();
        std::string err;
        const llvm::Target * tgt = llvm::TargetRegistry::lookupTarget(TT, err);
        if (!tgt)
        {
            fprintf(stderr, "prog-fuse: cannot find target '%s': %s\n", TT.str().c_str(), err.c_str());
            abort();
        }
        llvm::TargetOptions opts;
        tm.reset(tgt->createTargetMachine(
            TT, device ? (dev_arch ? dev_arch : "") : llvm::sys::getHostCPUName(),
            /* features */ "", opts,
            /* reloc */ std::nullopt, /* code model */ std::nullopt,
            /* match the O3 IR pipeline below for end-to-end aggressive codegen */
            llvm::CodeGenOptLevel::Aggressive));

        if (tm && (device || mod_u->getDataLayout().isDefault()))
            mod_u->setDataLayout(tm->createDataLayout());
    }

    if (llvm::verifyModule(*mod_u, &llvm::errs()))
    {
        fprintf(stderr, "prog-fuse: merged module verification failed\n");
        abort();
    }

    /* dump the merged module (wrapper + constituents) before optimization */
    if (dump)
        dump_module(dump_dir, "merged.ll", *mod_u);

    /* 8. Optimize (inline the kernels into the wrapper, vectorize, fuse). Device
     * chains defer the final O3 to emit_device_ptx (post DeviceRTL link); host
     * chains run the full pipeline here. */
    if (tm)
        optimize_module(*mod_u, tm.get(), dump_dir, /* run_o3 = */ !device);

    /* dump the final fused/optimized module */
    if (dump)
    {
        dump_module(dump_dir, "fused.ll", *mod_u);
        fprintf(stderr, "prog-fuse: dumped %zu input(s) + merged + fused IR to %s\n",
                n, dump_dir.c_str());
    }

    /* ------------------------------------------------------------------ *
     * 9. Serialise the optimized module to bitcode for dst->source.       *
     * ------------------------------------------------------------------ */
    std::string bitcode;
    {
        llvm::raw_string_ostream os(bitcode);
        llvm::WriteBitcodeToFile(*mod_u, os);
    }
    char * bc_buf = static_cast<char *>(malloc(bitcode.size()));
    if (!bc_buf) { fprintf(stderr, "prog-fuse: malloc failed\n"); abort(); }
    memcpy(bc_buf, bitcode.data(), bitcode.size());

    /* Free the previous source buffer iff a prior fusion produced (owns) it.
     * dst may alias progs[0], whose source was already fully parsed in step 1
     * (parseIR materialises the module), so the old buffer is no longer
     * referenced and is safe to release here. */
    if (dst->source.type == COMMAND_PROG_SOURCE_TYPE_LLVMIR &&
        dst->source.content.llvmir._owned &&
        dst->source.content.llvmir.raw)
    {
        free(dst->source.content.llvmir.raw);
    }

    dst->source.type                  = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    dst->source.content.llvmir.raw    = bc_buf;
    dst->source.content.llvmir.size   = bitcode.size();
    dst->source.content.llvmir._owned = true;   /* heap (malloc) — the pass owns it */
    /* The fused entry is `__fused_wrapper`. Host chains resolve it by name/shape,
     * so clear the per-input symbol; device chains must name it so the driver can
     * cuModuleGetFunction the PTX entry. (dst may alias progs[0] whose triple/arch
     * are preserved here, driving the jit pass's PTX codegen.) */
    dst->source.content.llvmir.symbol = (device || any_outline) ? "__fused_wrapper" : nullptr;
    /* the fused entry is void(void**), void(void*,size_t) when packed, or
     * void(void**,void**,void**) for a nanos6 outline chain; it carries no
     * per-parameter table (a re-fusion detects its shape by signature). */
    dst->source.content.llvmir.proto        = any_outline
        ? CGIR_COMMAND_PROG_SOURCE_PROTO_NANOS6_OUTLINE
        : (packed_output
            ? CGIR_COMMAND_PROG_SOURCE_PROTO_PACKED_BUFFER
            : CGIR_COMMAND_PROG_SOURCE_PROTO_VOID_PTRPTR);
    dst->source.content.llvmir.params       = nullptr;
    dst->source.content.llvmir.param_count  = 0;
    dst->source.content.llvmir._params_owned = false;

    /* ------------------------------------------------------------------ *
     * 9b. Merge the inputs' externalized-global resolution tables.        *
     *                                                                     *
     *  Each input externalizes the mutable globals it references; linking  *
     *  the closures (step 4) merges same-named external declarations into  *
     *  one, so the fused module needs the union of the inputs' tables      *
     *  (deduplicated by symbol name) to resolve them at JIT. Entries are    *
     *  shallow-copied: each `name` points to a stable (compile-time)       *
     *  string and `addr` to a stable process object, so the merged buffer  *
     *  outlives freeing any input table. We collect first (dst may alias   *
     *  progs[0], whose table we read) then free dst's previous owned one.  *
     * ------------------------------------------------------------------ */
    std::vector<command_prog_extern_t> merged_externs;
    for (size_t i = 0 ; i < n ; ++i)
    {
        if (progs[i]->source.type != COMMAND_PROG_SOURCE_TYPE_LLVMIR)
            continue ;
        const command_prog_extern_t * ex = progs[i]->source.content.llvmir.externs;
        const size_t cnt = progs[i]->source.content.llvmir.externs_count;
        for (size_t k = 0 ; k < cnt ; ++k)
        {
            if (ex[k].name == nullptr)
                continue ;
            bool dup = false;
            for (const command_prog_extern_t & m : merged_externs)
                if (m.name == ex[k].name || strcmp(m.name, ex[k].name) == 0) { dup = true; break ; }
            if (!dup)
                merged_externs.push_back(ex[k]);
        }
    }

    if (dst->source.content.llvmir._externs_owned && dst->source.content.llvmir.externs)
        free((void *) dst->source.content.llvmir.externs);

    if (merged_externs.empty())
    {
        dst->source.content.llvmir.externs        = nullptr;
        dst->source.content.llvmir.externs_count  = 0;
        dst->source.content.llvmir._externs_owned = false;
    }
    else
    {
        const size_t bytes = merged_externs.size() * sizeof(command_prog_extern_t);
        command_prog_extern_t * ebuf = static_cast<command_prog_extern_t *>(malloc(bytes));
        if (!ebuf) { fprintf(stderr, "prog-fuse: malloc failed\n"); abort(); }
        memcpy(ebuf, merged_externs.data(), bytes);
        dst->source.content.llvmir.externs        = ebuf;
        dst->source.content.llvmir.externs_count  = merged_externs.size();
        dst->source.content.llvmir._externs_owned = true;
    }

    /* The fused module is NOT compiled here: the `jit` pass
     * (command_graph_jit_llvmir) compiles dst->source and installs the function
     * pointer. mod_u/ctx are released at scope exit (the bitcode above is the
     * canonical artifact). */

    /* ------------------------------------------------------------------ *
     * 10. Fill the compacted args buffer with the deduplicated slots and   *
     *     install the variadic launcher (with a NULL fn, set by `jit`).    *
     *     dst may alias progs[0]; we read all originals' slots in step 2,  *
     *     so writing dst now is safe.                                      *
     * ------------------------------------------------------------------ */
    const size_t n_args = (size_t) total_args;
    void ** args_buf = nullptr;
    if (packed_output)
    {
        /* Packed byte buffer: each deduplicated slot's VALUE inlined at its
         * offset (matching the fused kernel's fixed-offset loads). unique_slots[k]
         * points at the value (a &value slot), so we copy its byte size -- from the
         * params table (packed leaves) or the IR slot type (unpacked leaves). */
        char * buf = static_cast<char *>(calloc(packed_size ? packed_size : 1, 1));
        if (!buf) { fprintf(stderr, "prog-fuse: calloc failed\n"); abort(); }
        for (unsigned k = 0 ; k < total_args ; ++k)
        {
            if (unique_slots[k] == nullptr)
                continue;
            size_t sz;
            if (any_packed_leaf)
                sz = unique_slot_size[k];
            else if (slot_type[k] != nullptr)
                sz = (size_t) DL.getTypeStoreSize(slot_type[k]).getFixedValue();
            else
                continue;
            if (sz == 0)
                continue;
            memcpy(buf + slot_offset[k], unique_slots[k], sz);
        }
        args_buf = reinterpret_cast<void **>(buf);
    }
    else
    {
        args_buf = static_cast<void **>(calloc(n_args ? n_args : 1, sizeof(void *)));
        if (!args_buf) { fprintf(stderr, "prog-fuse: calloc failed\n"); abort(); }
        for (unsigned k = 0 ; k < total_args ; ++k)
            args_buf[k] = unique_slots[k];
    }

    /* Free the previous args buffer iff a prior fusion produced (owns) it. dst
     * may alias progs[0], whose arg slot VALUES were copied into unique_slots
     * (and into the packed buffer above) already, so the old buffer is no longer
     * referenced and is safe to release here. */
    if (dst->_args_owned && dst->args)
        free(dst->args);

    /* A fused chain has no single ahead-of-time KMP routine, so it MUST be
     * JIT-compiled (the `jit` pass fills the launcher fn). It is either a uniform
     * void(void**) VARIADIC program or, under CGIR_PROG_FUSE_PACKED, a
     * void(void*, size_t) PACKED program over the packed byte buffer. */
    dst->prototype             = any_outline
        ? CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_NANOS6
        : (packed_output
            ? CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_PACKED
            : CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_VARIADIC);
    dst->launcher.variadic.fn  = nullptr;  /* compiled by the `jit` pass */
    dst->args                  = args_buf;
    /* For a nanos6 outline chain, n_args is the fused-instance count (the runtime
     * passes args/dev/translation arrays of this length); the compacted args
     * buffer is unused by the nanos6 launcher. */
    dst->n_args                = any_outline ? n : n_args;
    dst->args_size             = packed_output ? packed_size : 0;
    dst->_args_owned           = true;  /* heap (calloc/malloc) — the pass owns it */

    /* ------------------------------------------------------------------ *
     * 11b. Propagate the launch parameters to the fused program.          *
     *                                                                     *
     *  Every fused program shares identical grid/block dimensions and     *
     *  launch mode: the prog-fuse passes only chain programs for which    *
     *  command_prog_launch_params_equal() holds. The fused kernel is a    *
     *  single launch over that common geometry and launch mode, so it     *
     *  inherits progs[0]'s. We set this explicitly (rather than relying   *
     *  on dst aliasing progs[0]) so a distinct dst is correct too.        *
     *  progs[0]'s grid/block/launch_mode are never freed/modified above,  *
     *  so reading them here is safe even when dst == progs[0].            *
     *                                                                     *
     *  The occupancy target is NOT uniform across the inputs (it follows  *
     *  from each kernel's per-block resource use), and the fused kernel   *
     *  replaces all of them at once, so it takes the most restrictive     *
     *  non-zero value -- zero meaning "unconstrained" and losing to any   *
     *  real target. Dynamic shared memory is a requirement rather than a  *
     *  limit, so the fused program needs the largest of its inputs'.      *
     * ------------------------------------------------------------------ */
    unsigned int fused_blocks_per_sm = 0;
    unsigned int fused_dyn_shmem     = 0;
    for (size_t i = 0 ; i < n ; ++i)
    {
        const unsigned int b = progs[i]->blocks_per_sm;
        if (b && (fused_blocks_per_sm == 0 || b < fused_blocks_per_sm))
            fused_blocks_per_sm = b;
        if (progs[i]->dyn_shmem > fused_dyn_shmem)
            fused_dyn_shmem = progs[i]->dyn_shmem;
    }

    if (dst != progs[0])
    {
        dst->grid        = progs[0]->grid;
        dst->block       = progs[0]->block;
        dst->launch_mode = progs[0]->launch_mode;
    }
    else
    {
        assert(memcmp(&dst->grid,  &progs[0]->grid,  sizeof(dst->grid))  == 0);
        assert(memcmp(&dst->block, &progs[0]->block, sizeof(dst->block)) == 0);
        assert(dst->launch_mode == progs[0]->launch_mode);
    }
    dst->blocks_per_sm = fused_blocks_per_sm;
    dst->dyn_shmem     = fused_dyn_shmem;

    /* ------------------------------------------------------------------ *
     * 12. Release the consumed inputs' owned heap buffers.                 *
     *                                                                      *
     *  Every prog other than dst has been merged into dst and the caller   *
     *  will contract it out of the graph. If such an input owns heap        *
     *  buffers from an EARLIER fusion (e.g. re-fusing a node this pass      *
     *  produced), free them now to avoid leaking on re-fusion. All reads of *
     *  the inputs completed in steps 1-4 (source parsed/linked, arg slot   *
     *  values copied into unique_slots), so this is safe. We skip any prog  *
     *  aliasing dst (its old buffers were already handled above) and clear  *
     *  each freed slot so an accidentally repeated prog is not double-freed.*
     * ------------------------------------------------------------------ */
    for (size_t i = 0 ; i < n ; ++i)
    {
        if (progs[i] == dst)
            continue ;

        if (progs[i]->source.type == COMMAND_PROG_SOURCE_TYPE_LLVMIR &&
            progs[i]->source.content.llvmir._owned &&
            progs[i]->source.content.llvmir.raw)
        {
            free(progs[i]->source.content.llvmir.raw);
            progs[i]->source.content.llvmir.raw   = nullptr;
            progs[i]->source.content.llvmir.size  = 0;
            progs[i]->source.content.llvmir._owned = false;
        }

        if (progs[i]->_args_owned && progs[i]->args)
        {
            free(progs[i]->args);
            progs[i]->args        = nullptr;
            progs[i]->n_args      = 0;
            progs[i]->_args_owned = false;
        }

        /* release an owned externs table from a prior fusion (its entries were
         * shallow-copied into dst's merged table in step 9b) */
        if (progs[i]->source.type == COMMAND_PROG_SOURCE_TYPE_LLVMIR &&
            progs[i]->source.content.llvmir._externs_owned &&
            progs[i]->source.content.llvmir.externs)
        {
            free((void *) progs[i]->source.content.llvmir.externs);
            progs[i]->source.content.llvmir.externs        = nullptr;
            progs[i]->source.content.llvmir.externs_count  = 0;
            progs[i]->source.content.llvmir._externs_owned = false;
        }
    }

    # endif /* CGIR_SUPPORT_LLVM */
}

# if CGIR_SUPPORT_LLVM
/* Link the device bitcode at `bc_path` into `M` so the externs the kernel
 * references (e.g. __kmpc_target_init, __nv_cbrt) become defined. Without this the
 * CUDA driver cannot JIT the emitted PTX -- ptxas fails on the unresolved externs.
 * Only referenced symbols and their transitive deps are imported (LinkOnlyNeeded),
 * so the rest of the (large) library is not pulled in. Returns false + sets `err`. */
static bool
link_device_bitcode(llvm::Module & M, const char * bc_path, std::string & err)
{
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buf = [&] {
        scoped_phase_t _p("dev-link-read");
        return llvm::MemoryBuffer::getFile(bc_path);
    }();
    if (!buf)
    {
        err = "cannot read device bitcode '" + std::string(bc_path) + "': " +
              buf.getError().message();
        return false;
    }

    /* Parse the library. For bitcode (libdevice/DeviceRTL are .bc), load it LAZILY:
     * function bodies are materialized on demand, so the following LinkOnlyNeeded
     * link only deserializes the referenced functions instead of the whole (large)
     * library. getOwningLazyBitcodeModule takes ownership of the buffer (kept alive
     * for on-demand materialization). Fall back to eager parseIR for textual IR. */
    std::unique_ptr<llvm::Module> lib;
    {
        scoped_phase_t _p("dev-link-parse");
        const unsigned char * s = (const unsigned char *) (*buf)->getBufferStart();
        const unsigned char * e = (const unsigned char *) (*buf)->getBufferEnd();
        if (llvm::isBitcode(s, e))
        {
            auto m = llvm::getOwningLazyBitcodeModule(std::move(*buf), M.getContext(),
                                                      /* ShouldLazyLoadMetadata */ true);
            if (!m)
            {
                err = "lazy-parse device bitcode '" + std::string(bc_path) + "' failed: " +
                      llvm::toString(m.takeError());
                return false;
            }
            lib = std::move(*m);
        }
        else
        {
            llvm::SMDiagnostic diag;
            lib = llvm::parseIR((*buf)->getMemBufferRef(), diag, M.getContext());
            if (!lib)
            {
                err = "parse device bitcode '" + std::string(bc_path) + "' failed: " +
                      diag.getMessage().str();
                return false;
            }
        }
    }

    /* Align triple/DataLayout with the destination so the linker does not refuse
     * on a mismatch (the library is already nvptx64, same as M). */
    lib->setTargetTriple(M.getTargetTriple());
    lib->setDataLayout(M.getDataLayout());

    llvm::Linker linker(M);
    bool link_failed;
    {
        scoped_phase_t _p("dev-link-linkin");
        link_failed = linker.linkInModule(std::move(lib), llvm::Linker::Flags::LinkOnlyNeeded);
    }
    if (link_failed)
    {
        err = "linking device bitcode '" + std::string(bc_path) + "' failed";
        return false;
    }
    return true;
}

/* Ensure OpenMPOpt recognizes the module and its kernels: the "openmp"/
 * "openmp-device" module flags (else it early-exits) and the "kernel" attribute
 * on each kernel-CC entry (else getDeviceKernels() skips it, so the entry is not
 * preserved and its SPMD transforms are skipped). Idempotent; the flag value is
 * tested for presence only. */
static void
prepare_device_module_for_openmp(llvm::Module & M)
{
    if (!M.getModuleFlag("openmp"))
        M.addModuleFlag(llvm::Module::Max, "openmp", 51);
    if (!M.getModuleFlag("openmp-device"))
        M.addModuleFlag(llvm::Module::Max, "openmp-device", 51);
    for (llvm::Function & F : M)
        if (F.hasKernelCallingConv() && !F.isDeclaration() && !F.hasFnAttribute("kernel"))
            F.addFnAttr("kernel");
}

/* Build the standard analysis managers + PassBuilder for `tm` and run `body`,
 * which appends passes to the given ModulePassManager. Factored out so the
 * pre-link (SPMD-ization) and post-link (O3) steps share the setup. */
template <typename BuildFn>
static void
run_module_passes(llvm::Module & M, llvm::TargetMachine * tm, BuildFn && body)
{
    llvm::PassBuilder PB(tm);

    llvm::LoopAnalysisManager     LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager    CGAM;
    llvm::ModuleAnalysisManager   MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::ModulePassManager MPM;
    body(PB, MPM);
    MPM.run(M, MAM);
}

/* ---------------------------------------------------------------------------
 * SECTION 6 - device (NVPTX) code generation: SPMD-ize the recorded snapshot,
 * link the device libraries, optimize, emit PTX.
 * ------------------------------------------------------------------------- */

/* Pre-link SPMD-ization of the raw generic-mode device snapshot. Runs OpenMPOpt
 * in a NON-post-link phase, BEFORE linking the DeviceRTL: SPMD-ization needs the
 * SPMD runtime (__kmpc_get_hardware_thread_id_in_block, __kmpc_barrier_simple_spmd)
 * "available", which post-link means already defined -- but the generic snapshot
 * does not reference them yet. Non-post-link assumes they are linked later, so
 * OpenMPOpt SPMD-izes and emits the SPMD calls that the following LinkOnlyNeeded
 * then resolves. */
static void
spmdize_device_module(llvm::Module & M, llvm::TargetMachine * tm)
{
    prepare_device_module_for_openmp(M);
    run_module_passes(M, tm, [] (llvm::PassBuilder &, llvm::ModulePassManager & MPM) {
        MPM.addPass(llvm::OpenMPOptPass(llvm::ThinOrFullLTOPhase::None));
    });
}

/* Post-link finalize: O3 inlines the just-linked DeviceRTL, folds the (now
 * constant) kernel-environment config, DCEs the dead runtime and vectorizes.
 * Must run AFTER the DeviceRTL is linked into `M`. */
static void
optimize_device_module_o3(llvm::Module & M, llvm::TargetMachine * tm)
{
    prepare_device_module_for_openmp(M);   // idempotent; robust to link side-effects

    /* LTO-style internalization (matches the offload backend): the DeviceRTL is
     * linked weak/hidden, so its config globals (@__omp_rtl_debug_kind = 0, ...)
     * are not constant-foldable and its unused functions are not DCE-able.
     * Internalizing everything but the kernel entries makes them 'internal', so
     * O3 folds the debug/assert machinery away and globalDCE drops the dead
     * runtime. llvm.used members are auto-preserved. */
    llvm::internalizeModule(M, [] (const llvm::GlobalValue & GV) -> bool {
        // keep kernel entries external (resolved by name via cuModuleGetFunction)
        if (const auto * F = llvm::dyn_cast<llvm::Function>(&GV))
            return F->hasKernelCallingConv();
        return false;
    });

    run_module_passes(M, tm, [] (llvm::PassBuilder & PB, llvm::ModulePassManager & MPM) {
        MPM.addPass(PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3));
    });
}

/* ---------------------------------------------------------------------------
 * SECTION 7 - host (CPU) code generation: retarget at the running machine,
 * optimize, emit an object, link it into the process with ORC.
 * ------------------------------------------------------------------------- */

/* Retarget every definition in `M` at the machine we are running on.
 *
 * A TargetMachine's CPU/features are only a default: both the AArch64 and X86
 * backends prefer a function's own `target-cpu`/`target-features` attributes
 * when they are present, and the compiler stamps them on everything with the
 * baseline the application was built for. Overwriting them is what lets the JIT
 * use the instructions the machine actually has. Declarations are left alone
 * (they carry no code), and the vector-width hints are dropped because they were
 * derived from the old feature set. */
static void
stamp_host_target_attrs(llvm::Module & M)
{
    const std::string & cpu      = host_target_attrs().first;
    const std::string & features = host_target_attrs().second;
    if (cpu.empty() && features.empty())
        return ;

    for (llvm::Function & F : M)
    {
        if (F.isDeclaration())
            continue ;
        if (!cpu.empty())
            F.addFnAttr("target-cpu", cpu);
        if (!features.empty())
            F.addFnAttr("target-features", features);
        F.removeFnAttr("min-legal-vector-width");
        F.removeFnAttr("prefer-vector-width");
        F.removeFnAttr("tune-cpu");
    }
}

/* Optimize a JIT'd host task module before codegen (host analogue of
 * optimize_device_module_o3): O3 optimizes the pre-optimization frontend snapshot.
 * Also promote available_externally definitions (inline callees the closure keeps
 * only for inlining, e.g. a `declare target` SQRT) to internal, so codegen emits
 * them instead of dropping them into unresolvable externals. Externalized globals
 * are plain external declarations, so they are left untouched. */
static void
optimize_host_module(llvm::Module & M, llvm::TargetMachine * tm, llvm::StringRef entry_name)
{
    for (llvm::Function & F : M)
        if (F.hasAvailableExternallyLinkage() && !F.isDeclaration())
            F.setLinkage(llvm::GlobalValue::InternalLinkage);
    for (llvm::GlobalVariable & G : M.globals())
        if (G.hasAvailableExternallyLinkage() && G.hasInitializer())
            G.setLinkage(llvm::GlobalValue::InternalLinkage);

    /* Only the entry is looked up after linking, so everything else can be
     * internal -- which is what lets O3 propagate the arguments the wrapper
     * loads into the body, drop the dead parameters, and DCE what is left. Kept
     * external, the task body is an ABI-visible symbol and none of that applies.
     * (The device path internalizes for the same reason, and the fuse pass does
     * not need to because it force-inlines everything into its wrapper.)
     * llvm.used members are preserved automatically. */
    llvm::internalizeModule(M, [&] (const llvm::GlobalValue & GV) -> bool {
        return GV.getName() == entry_name;
    });

    run_module_passes(M, tm, [] (llvm::PassBuilder & PB, llvm::ModulePassManager & MPM) {
        MPM.addPass(PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3));
    });
}

/* The subtarget feature string the compiler stamped on the device functions
 * (e.g. "+ptx92,+sm_90"). The TargetMachine's *default* subtarget is what the
 * NVPTX AsmPrinter uses for the `.version`/`.target` PTX header and what the
 * backend uses for anything not attached to a function, so building it with an
 * empty feature string silently downgrades the emitted PTX ISA version below
 * what the (per-function) subtarget selects instructions for. Prefer the kernel
 * entry's attribute, then any function's. */
static std::string
device_features_of(const llvm::Module & M)
{
    const llvm::Function * fallback = nullptr;
    for (const llvm::Function & F : M)
    {
        if (F.isDeclaration() || !F.hasFnAttribute("target-features"))
            continue ;
        if (F.hasKernelCallingConv())
            return F.getFnAttribute("target-features").getValueAsString().str();
        if (fallback == nullptr)
            fallback = &F;
    }
    return fallback ? fallback->getFnAttribute("target-features").getValueAsString().str()
                    : std::string();
}

/* Emit PTX for a device (GPU) module: build a device TargetMachine for `triple`
 * (e.g. "nvptx64-nvidia-cuda") + `arch` (target-cpu, e.g. "sm_80"), stamp the
 * module's triple/DataLayout, and run codegen to a PTX assembly string. Returns
 * the PTX (empty + `err` set on failure). The CUDA driver JIT-compiles this PTX
 * to SASS at cuModuleLoadData (see the xkrt driver). */
static std::string
emit_device_ptx(llvm::Module & M, const char * triple, const char * arch,
                const char * const * libs, size_t nlibs,
                const std::string & dump_dir, std::string & err)
{
    llvm::Triple TT(triple);
    std::string terr;
    const llvm::Target * T = llvm::TargetRegistry::lookupTarget(TT, terr);
    if (!T) { err = "lookupTarget('" + TT.str() + "'): " + terr; return {}; }

    /* Read the features off the snapshot BEFORE anything is linked/optimized in:
     * at this point the module holds only the compiler-emitted device functions. */
    const std::string features = device_features_of(M);

    llvm::TargetOptions opts;
    std::unique_ptr<llvm::TargetMachine> TM(T->createTargetMachine(
        TT, arch ? arch : "", features, opts,
        std::nullopt, std::nullopt, llvm::CodeGenOptLevel::Aggressive));
    if (!TM) { err = "createTargetMachine failed"; return {}; }

    M.setTargetTriple(TT);
    M.setDataLayout(TM->createDataLayout());

    scoped_phase_t _emit("dev-emit-total");

    /* 1. Pre-link: SPMD-ize the generic-mode snapshot (must precede the link). */
    {
        scoped_phase_t _p("dev-spmdize");
        spmdize_device_module(M, TM.get());
    }

    /* 2. Link the device bitcode libraries the kernel references (in order, so a
     * later library can resolve externs of an earlier one), so ptxas can JIT the
     * PTX at cuModuleLoadData. CGIR_DEVICE_EXTRA_BC adds one more for debugging. */
    for (size_t i = 0 ; i < nlibs ; ++i)
        if (libs[i] && libs[i][0] && !link_device_bitcode(M, libs[i], err))
            return {};
    if (const char * extra = getenv("CGIR_DEVICE_EXTRA_BC"); extra && extra[0])
        if (!link_device_bitcode(M, extra, err))
            return {};

    /* 3. Post-link: O3 to AOT quality (inline runtime, fold config, DCE, vectorize). */
    {
        scoped_phase_t _p("dev-o3");
        optimize_device_module_o3(M, TM.get());
    }

    if (!dump_dir.empty())
        dump_module(dump_dir, "optimized.ll", M);

    llvm::SmallString<0> out;
    llvm::raw_svector_ostream os(out);
    llvm::legacy::PassManager pm;
    if (TM->addPassesToEmitFile(pm, os, /* DwoOut */ nullptr,
                                llvm::CodeGenFileType::AssemblyFile))
    { err = "addPassesToEmitFile: PTX (assembly) emission not supported"; return {}; }
    {
        scoped_phase_t _p("dev-ptx-emit");
        pm.run(M);
    }
    return std::string(out.begin(), out.end());
}

/* Install a compiled host JIT result onto `prog`, shared by the fresh-compile
 * path and the in-process cache-hit path. For a STANDALONE packed leaf it also
 * materializes the per-instance packed byte buffer from the recorded &value slots
 * (params != NULL identifies the standalone case; prog-fuse clears params on a
 * fused packed program). That packing is per-instance -- each prog has its own
 * args -- so it runs on cache hits too; it is idempotent (args_size == 0 guard). */
static void
install_host_jit_result(command_prog_t * prog, void * fn_addr,
                        bool entry_is_nanos6, bool entry_is_packed)
{
    if (entry_is_nanos6)
    {
        prog->prototype          = CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_NANOS6;
        prog->launcher.nanos6.fn = reinterpret_cast<void (*)(void **, void **, void **)>(fn_addr);
    }
    else if (entry_is_packed)
    {
        const cgir_command_prog_param_t * params = prog->source.content.llvmir.params;
        const size_t nparams = prog->source.content.llvmir.param_count;
        if (params != nullptr && nparams > 0 && prog->args_size == 0)
        {
            size_t buf_size = 0;
            for (size_t k = 0 ; k < nparams ; ++k)
            {
                const size_t end = params[k].offset + params[k].size;
                if (end > buf_size) buf_size = end;
            }
            if (buf_size == 0) buf_size = 1;
            char * buf = static_cast<char *>(calloc(buf_size, 1));
            if (!buf) { fprintf(stderr, "jit: calloc failed\n"); abort(); }
            void ** slots = prog->args;
            for (size_t k = 0 ; slots != nullptr && k < nparams ; ++k)
                if (slots[k] != nullptr)
                    memcpy(buf + params[k].offset, slots[k], params[k].size);
            if (prog->_args_owned && prog->args)
                free(prog->args);
            prog->args        = reinterpret_cast<void **>(buf);
            prog->n_args      = nparams;
            prog->args_size   = buf_size;
            prog->_args_owned = true;
        }
        prog->prototype          = CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_PACKED;
        prog->launcher.packed.fn = reinterpret_cast<void (*)(void *, size_t)>(fn_addr);
    }
    else
    {
        prog->prototype            = CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_VARIADIC;
        prog->launcher.variadic.fn = reinterpret_cast<void (*)(void **)>(fn_addr);
    }
}

/* Replace a device prog's source IR with emitted/cached PTX (owned copy), keeping
 * the entry symbol so the driver can cuModuleGetFunction it. Shared by the fresh
 * emit and the cache-hit paths; leaves launcher null (the driver fills it). */
static void
restore_device_ptx(command_prog_t * prog, const std::string & ptx)
{
    if (prog->source.content.llvmir._owned && prog->source.content.llvmir.raw)
        free(prog->source.content.llvmir.raw);
    char * buf = static_cast<char *>(malloc(ptx.size() + 1));
    if (!buf) { fprintf(stderr, "jit(device): malloc failed\n"); abort(); }
    memcpy(buf, ptx.data(), ptx.size());
    buf[ptx.size()] = '\0';
    prog->source.type                  = COMMAND_PROG_SOURCE_TYPE_PTX;
    prog->source.content.llvmir.raw    = buf;
    prog->source.content.llvmir.size   = ptx.size() + 1; /* incl. NUL */
    prog->source.content.llvmir._owned = true;
    prog->prototype            = CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_VARIADIC;
    prog->launcher.variadic.fn = nullptr;
}

/* Resolve the symbols a host program references but does not define: externalized
 * globals as absolute symbols (their real runtime addresses, so compiled code
 * binds to the process's objects) + a generator over the process's dynamic symbols
 * (libc printf, ...). Shared by the fresh-compile and disk-object load paths. */
static void
setup_host_jit_symbols(llvm::orc::LLJIT & jit, const command_prog_t * prog)
{
    llvm::orc::JITDylib & jd = jit.getMainJITDylib();

    const cgir_command_prog_extern_t * externs = prog->source.content.llvmir.externs;
    const size_t n_externs = prog->source.content.llvmir.externs_count;
    if (externs && n_externs)
    {
        llvm::orc::SymbolMap syms;
        for (size_t i = 0 ; i < n_externs ; ++i)
        {
            if (externs[i].name == nullptr)
                continue ;
            syms[jit.mangleAndIntern(externs[i].name)] =
                llvm::orc::ExecutorSymbolDef(
                    llvm::orc::ExecutorAddr::fromPtr(externs[i].addr),
                    llvm::JITSymbolFlags::Exported);
        }
        if (!syms.empty())
            if (auto err = jd.define(llvm::orc::absoluteSymbols(std::move(syms))))
            {
                llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "jit: ");
                fprintf(stderr, "jit: failed to install externalized-global symbols\n");
                abort();
            }
    }

    auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        jit.getDataLayout().getGlobalPrefix());
    if (!gen)
    {
        llvm::logAllUnhandledErrors(gen.takeError(), llvm::errs(), "jit: ");
        fprintf(stderr, "jit: failed to create process symbol generator\n");
        abort();
    }
    jd.addGenerator(std::move(*gen));
}
# endif /* CGIR_SUPPORT_LLVM */

void
CGIR_NAMESPACE::command_graph_jit_llvmir(
    command_prog_t * prog
) {
    # if !CGIR_SUPPORT_LLVM
    (void) prog;
    fprintf(stderr, "jit: LLVM support not enabled (rebuild with -DUSE_LLVM=ON)\n");
    abort();
    # else
    assert(prog);
    assert(prog->source.type == COMMAND_PROG_SOURCE_TYPE_LLVMIR);
    assert(prog->source.content.llvmir.raw != nullptr);

    scoped_phase_t _jit_total("jit-total");

    /* Result-cache key from the source bytes/attributes (no parse needed): a hit
     * reuses a prior instance's artifact, skipping parse + the compile pipeline.
     * The key covers everything a recompile would reproduce, so a hit is
     * byte-identical to recompiling. */
    const uint64_t cache_key = jit_cache_key(prog);
    const bool     is_device = (prog->source.content.llvmir.triple != nullptr);

    /* Fast path (pre-parse): reuse a cached artifact directly. Device PTX (from
     * this process or a prior run on disk); host compiled function pointer (this
     * process only -- disk objects are loaded further down, needing the entry). */
    if (is_device)
    {
        std::string ptx;
        if (const int hit = jit_cache().device_get(cache_key, ptx))
        {
            restore_device_ptx(prog, ptx);
            if (jit_prof().stats_on) jit_prof().cache_event(true, hit == 1 ? 1 : 2);
            return ;
        }
    }
    else
    {
        jit_cache_t::host_entry_t e;
        if (jit_cache().host_get_fn(cache_key, e))
        {
            install_host_jit_result(prog, e.fn,
                e.proto == (int) CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_NANOS6,
                e.proto == (int) CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_PACKED);
            if (jit_prof().stats_on) jit_prof().cache_event(false, 2);
            return ;
        }
    }

    ensure_llvm_initialized();

    /* Optional IR dumping for debugging (CGIR_JIT_DUMP). */
    const bool  dump     = dump_enabled("CGIR_JIT_DUMP");
    std::string dump_dir = dump ? dump_make_dir("CGIR_JIT_DUMP", "jit") : std::string();

    /* parse the program's IR/bitcode into its own context */
    auto ctx = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> mod = [&] {
        scoped_phase_t _p("jit-parse");
        return parse_llvmir(
            static_cast<const char *>(prog->source.content.llvmir.raw),
            prog->source.content.llvmir.size,
            *ctx
        );
    }();
    if (!mod) { fprintf(stderr, "jit: failed to parse program IR\n"); abort(); }

    /* dump the program IR as parsed, before any transform */
    if (dump)
        dump_module(dump_dir, "input.ll", *mod);

    /* Resolve the entry function, each step a fallback for the previous: the
     * explicit symbol (host tasks and device kernels name their closure entry);
     * else a fused program's __fused_wrapper(void**); else the first externally-
     * linked definition (the task/kernel entry externalized for JIT); else the
     * first void definition. Falling through when a named symbol is absent keeps
     * a stale/wrong name from silently skipping the command. */
    llvm::Function * entry = nullptr;
    if (const char * sym = prog->source.content.llvmir.symbol)
        entry = mod->getFunction(sym);
    if (entry == nullptr || entry->isDeclaration())
        if (llvm::Function * w = mod->getFunction("__fused_wrapper"); w && !w->isDeclaration())
            entry = w;
    if (entry == nullptr || entry->isDeclaration())
    {
        entry = nullptr;
        for (llvm::Function & F : *mod)
            if (!F.isDeclaration() && F.hasExternalLinkage())
            {
                entry = &F;
                break ;
            }
        if (entry == nullptr)
            entry = find_kernel(*mod);
    }

    /* Skip (do not abort) commands whose entry is not externally JIT-resolvable
     * - e.g. a not-yet-externalized internal task body, or a fixed-launcher
     * command that carries a source but is not a JIT target. Leaves the
     * launcher untouched. A nanos6 outline is exempt from the local-linkage
     * check: an unfused (single-instance) outline may be internal in its module
     * and is JIT'd by wrapping it in an external __fused_wrapper below. */
    const bool proto_is_nanos6 =
        (prog->source.content.llvmir.proto == CGIR_COMMAND_PROG_SOURCE_PROTO_NANOS6_OUTLINE);
    if (entry == nullptr || entry->isDeclaration() ||
        (entry->hasLocalLinkage() && !proto_is_nanos6))
        return ;

    /* Device (GPU) miss: emit the (fused) device kernel to PTX for the device's
     * triple/arch, cache it (in-process + on-disk), and store it back in the
     * source; the driver JIT-loads it (cuModuleLoadData) and resolves the entry
     * (source.symbol) at launch. The fast path above already handled cache hits. */
    if (is_device)
    {
        /* Opt-in: assume the kernel's pointer parameters do not overlap.
         *
         * This is the one place the device JIT can beat the ahead-of-time
         * toolchain on code quality. NVPTX lowers a load through a kernel
         * parameter to `ld.global.nc` (the read-only data path) only when the
         * parameter is BOTH readonly and noalias -- `readonly` the optimizer
         * infers on its own, `noalias` it cannot, because an `omp target`'s
         * mapped list items are not required to be distinct objects. The
         * compiler must therefore give up on it; a runtime that knows which
         * buffers a task actually touches does not have to. prog-fuse already
         * makes exactly this assumption for the pointers it captures into a
         * fused wrapper (command_graph_prog_fuse_llvmir, "distinct captured
         * pointers are assumed non-overlapping").
         *
         * Off by default because it is an assumption about the program, not a
         * deduction: enable it only when the mapped buffers of a target region
         * are known to be disjoint. */
        if (device_assume_noalias_params())
        {
            unsigned marked = 0;
            for (llvm::Argument & A : entry->args())
                if (A.getType()->isPointerTy() && !A.hasNoAliasAttr())
                {
                    A.addAttr(llvm::Attribute::NoAlias);
                    ++marked;
                }
            if (marked && dump)
                fprintf(stderr, "jit(device): assuming %u pointer parameters of `%s` do not alias\n",
                        marked, entry->getName().str().c_str());
        }

        const char * dtriple = prog->source.content.llvmir.triple;
        std::string perr;
        std::string ptx = emit_device_ptx(*mod, dtriple,
                                          prog->source.content.llvmir.arch,
                                          prog->source.content.llvmir.device_libs,
                                          prog->source.content.llvmir.device_libs_count,
                                          dump ? dump_dir : std::string(), perr);
        if (ptx.empty())
        {
            fprintf(stderr, "jit(device): PTX emission failed: %s\n", perr.c_str());
            abort();
        }
        jit_cache().device_put(cache_key, ptx);
        if (dump)
        {
            std::string path = dump_dir + "/final.ptx";
            std::error_code ec;
            llvm::raw_fd_ostream f(path, ec, llvm::sys::fs::OF_Text);
            if (!ec) f << ptx;
        }
        restore_device_ptx(prog, ptx);
        if (jit_prof().stats_on) jit_prof().cache_event(true, 0);
        return ;
    }

    /* The runtime launches a JIT'd PROG as `void(void**)` (VARIADIC) or, for a
     * packed body, `void(void*, size_t)` (PACKED). If the entry already has one
     * of those shapes, use it directly; otherwise synthesize a void(void**)
     * wrapper that unpacks the argument slots and calls the entry.
     *
     * Which shape it is must come from the declared proto, not from the
     * signature alone: under opaque pointers a leaf taking a single captured
     * pointer -- `void kernel(double * A)`, proto UNPACKED_PARAMS -- has exactly
     * the shape of `void kernel(void ** args)`, and calling it directly would
     * hand it the slot array instead of `*(double **) args[0]`.
     *
     * UNPACKED_PARAMS is also the default enumerator, so it doubles as "the
     * producer said nothing". A parameter table disambiguates the two: when one
     * is present the program really is a leaf and always gets a wrapper; without
     * it we fall back to the signature, which is what producers that fill in
     * nothing have always relied on. */
    llvm::FunctionType * efty = entry->getFunctionType();
    std::string lookup_name;
    const auto   proto        = prog->source.content.llvmir.proto;
    const bool   proto_unset  = (proto == CGIR_COMMAND_PROG_SOURCE_PROTO_UNPACKED_PARAMS &&
                                 prog->source.content.llvmir.param_count == 0);
    const bool entry_is_nanos6 = proto_is_nanos6;
    const bool entry_is_packed = (proto == CGIR_COMMAND_PROG_SOURCE_PROTO_PACKED_BUFFER) ||
                                 (proto_unset && is_void_ptr_size(efty));
    const bool entry_is_ptrptr = (proto == CGIR_COMMAND_PROG_SOURCE_PROTO_VOID_PTRPTR) ||
                                 (proto_unset && is_void_voidptr(efty));
    if (entry_is_nanos6)
    {
        /* nanos6 outline chain: the runtime launches it as
         *   void(void** args_v, void** dev_v, void** transl_v).
         * A fused chain (>= 2) already exposes that 3-array __fused_wrapper; a
         * standalone (unfused, single-instance) outline is the raw
         * void(args, dev, translation) body, so wrap it in a 1-instance
         * __fused_wrapper here for a uniform launch ABI. Both `void(void**,...)`
         * and `void(void*,...)` have the same opaque-pointer type, so we
         * distinguish the fused wrapper by its name. */
        if (entry->getName() == "__fused_wrapper" && is_void_ptr_ptr_ptr(efty))
        {
            lookup_name = entry->getName().str();
        }
        else
        {
            llvm::LLVMContext & C = mod->getContext();
            llvm::Type * void_ty = llvm::Type::getVoidTy(C);
            llvm::Type * ptr_ty  = llvm::PointerType::getUnqual(C);
            llvm::Type * i64_ty  = llvm::Type::getInt64Ty(C);

            llvm::FunctionType * wfty = llvm::FunctionType::get(
                void_ty, { ptr_ty, ptr_ty, ptr_ty }, false);
            llvm::Function * wrapper = llvm::Function::Create(
                wfty, llvm::GlobalValue::ExternalLinkage, "__fused_wrapper", mod.get());
            llvm::BasicBlock * bb = llvm::BasicBlock::Create(C, "entry", wrapper);
            llvm::IRBuilder<> b(bb);
            auto load0 = [&] (llvm::Value * base) -> llvm::Value *
            {
                llvm::Value * p = b.CreateGEP(
                    ptr_ty, base, llvm::ConstantInt::get(i64_ty, 0), "slot");
                return b.CreateLoad(ptr_ty, p, "inst");
            };
            b.CreateCall(efty, entry,
                { load0(wrapper->getArg(0)), load0(wrapper->getArg(1)), load0(wrapper->getArg(2)) });
            b.CreateRetVoid();
            lookup_name = "__fused_wrapper";
        }
    }
    else if (entry_is_ptrptr || entry_is_packed)
    {
        lookup_name = entry->getName().str();
    }
    else
    {
        llvm::LLVMContext & C = mod->getContext();
        llvm::Type * void_ty = llvm::Type::getVoidTy(C);
        llvm::Type * ptr_ty  = llvm::PointerType::getUnqual(C);

        llvm::FunctionType * wfty = llvm::FunctionType::get(void_ty, { ptr_ty }, false);
        llvm::Function * wrapper = llvm::Function::Create(
            wfty, llvm::GlobalValue::ExternalLinkage, "__cgir_jit_wrapper", mod.get());

        /* The slot array is the runtime's own, distinct from every buffer the
         * body touches (same assumption the fuse pass makes of its wrapper). It
         * is what frees the loads below to be hoisted and CSE'd instead of being
         * re-read after every store the body performs. */
        wrapper->addParamAttr(0, llvm::Attribute::NoAlias);
        wrapper->addParamAttr(0, llvm::Attribute::ReadOnly);

        llvm::BasicBlock * bb = llvm::BasicBlock::Create(C, "entry", wrapper);
        llvm::IRBuilder<> b(bb);
        llvm::Value * args_ptr = wrapper->getArg(0);

        std::vector<llvm::Value *> call_args;
        call_args.reserve(efty->getNumParams());
        for (unsigned k = 0 ; k < efty->getNumParams() ; ++k)
            call_args.push_back(emit_load_arg(b, args_ptr, k, efty->getParamType(k)));
        llvm::CallInst * call = b.CreateCall(efty, entry, call_args);
        b.CreateRetVoid();

        /* Inline it here rather than hoping the cost model does: the wrapper
         * exists only to unpack, and leaving the call in place would both keep a
         * real call on every task launch and hide the now-known argument values
         * from the body. The fuse pass inlines its constituents for the same
         * reason. */
        llvm::InlineFunctionInfo ifi;
        llvm::InlineResult ires = llvm::InlineFunction(*call, ifi);
        if (!ires.isSuccess())
            fprintf(stderr, "jit: could not inline `%s` into its argument-unpacking "
                            "wrapper (%s); leaving the call in place\n",
                    entry->getName().str().c_str(), ires.getFailureReason());

        lookup_name = "__cgir_jit_wrapper";
    }

    auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb)
    {
        llvm::logAllUnhandledErrors(jtmb.takeError(), llvm::errs(), "jit: ");
        fprintf(stderr, "jit: failed to detect host JIT target machine\n");
        abort();
    }

    /* Code model. JIT'd code maps in a fresh region that can be far from the
     * process objects this program binds to (externalized globals as absolute
     * symbols, libc, ...), further than a PC-relative branch or page-relative
     * address can reach -- which is why this used to compile everything with the
     * LARGE model. That is a heavy price to pay everywhere for a few far
     * references: on AArch64 the large model materializes *every* global,
     * constant-pool entry and block address with a 4-instruction MOVZ/MOVK chain
     * instead of ADRP+ADD, and calls out of range must go through a register.
     *
     * The distance problem is the linker's to solve, and ORC's JITLink already
     * does: it builds GOT and PLT tables for the references that need them and
     * relaxes the ones that do not, so only genuinely far symbols pay. LLJIT
     * therefore *wants* Small + PIC (see its JITLink auto-configuration) and only
     * refrains because a code model is already set here. So set the one it wants
     * -- on the builder, since the object below is compiled by a TargetMachine
     * made from it, not by LLJIT's own.
     *
     * `CGIR_JIT_HOST_CODE_MODEL=large` restores the old behaviour, for a platform
     * where LLJIT falls back to RuntimeDyld rather than JITLink. */
    if (strcmp(env_str("CGIR_JIT_HOST_CODE_MODEL", "small"), "large") == 0)
        jtmb->setCodeModel(llvm::CodeModel::Large);
    else
    {
        jtmb->setCodeModel(llvm::CodeModel::Small);
        jtmb->setRelocationModel(llvm::Reloc::PIC_);
    }

    /* Codegen at the same level as the IR pipeline (O3), as the device and fuse
     * paths do; JITTargetMachineBuilder otherwise defaults to O2. */
    jtmb->setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);

    /* JITTargetMachineBuilder's constructor turns emulated TLS on. That would
     * lower every `thread_local` access in a JIT'd body to an
     * __emutls_get_address call instead of a thread-pointer-relative access --
     * and, worse, to a *different* storage than the process uses for the same
     * variable. We link against the running process, so use its TLS model. */
    jtmb->getOptions().EmulatedTLS = false;

    /* Give the optimizer and codegen the machine they are actually running on.
     * detectHost() puts the host CPU and feature set on the TargetMachine, but a
     * per-function `target-cpu`/`target-features` attribute wins over it, and the
     * compiler stamped every function with the baseline the application was built
     * for. Without this the JIT re-optimizes at that baseline and the host
     * detection is wasted -- which throws away the one advantage a JIT has here,
     * since it knows the exact machine and the ahead-of-time compiler did not.
     * (Folded into the cache key by jit_host_salt.) */
    stamp_host_target_attrs(*mod);

    /* Parsed IR may omit the triple (the fuse pass stamps it for the same
     * reason): without it TargetLibraryInfo is built for an unknown triple, so
     * the optimizer does not know which libcalls exist. */
    if (mod->getTargetTriple().empty())
        mod->setTargetTriple(jtmb->getTargetTriple());

    /* Obtain the relocatable object for `lookup_name`: reuse it from the on-disk
     * cache if a prior run compiled this exact body (skips optimize + codegen),
     * else optimize + emit it and cache it. ORC's SimpleCompiler emits the object
     * via the same addPassesToEmitFile, so a self-emitted object loads identically
     * through addObjectFile -- externs/libc are bound at link time either way. */
    std::string obj;
    const bool obj_from_disk = jit_cache().host_get_obj(cache_key, obj);
    if (!obj_from_disk)
    {
        auto otm = jtmb->createTargetMachine();
        if (!otm)
        {
            llvm::logAllUnhandledErrors(otm.takeError(), llvm::errs(), "jit: ");
            fprintf(stderr, "jit: failed to create host target machine\n");
            abort();
        }
        /* likewise for the layout: the default one has the wrong ABI alignments
         * and no native integer widths, which both mislays structs/allocas and
         * degrades what InstCombine and the vectorizer will do */
        if (mod->getDataLayout().isDefault())
            mod->setDataLayout(otm->get()->createDataLayout());
        {
            scoped_phase_t _p("host-optimize");
            optimize_host_module(*mod, otm->get(), lookup_name);
        }
        if (dump)
            dump_module(dump_dir, "final.ll", *mod);
        {
            scoped_phase_t _p("host-codegen");
            llvm::SmallString<0> objbuf;
            llvm::raw_svector_ostream os(objbuf);
            llvm::legacy::PassManager pm;
            if (otm->get()->addPassesToEmitFile(pm, os, /* DwoOut */ nullptr,
                                                llvm::CodeGenFileType::ObjectFile))
            { fprintf(stderr, "jit: host object emission not supported\n"); abort(); }
            pm.run(*mod);
            obj.assign(objbuf.begin(), objbuf.end());
        }
        jit_cache().host_put_obj(cache_key, obj);
    }

    /* Load the object into a fresh LLJIT, bind the program's symbols, resolve the
     * entry. addObjectFile skips IR codegen (the object is already compiled). */
    auto jit_exp = [&] {
        scoped_phase_t _p("host-orc-create");
        return llvm::orc::LLJITBuilder()
            .setJITTargetMachineBuilder(std::move(*jtmb))
            .create();
    }();
    if (!jit_exp)
    {
        llvm::logAllUnhandledErrors(jit_exp.takeError(), llvm::errs(), "jit: ");
        fprintf(stderr, "jit: failed to create LLJIT\n");
        abort();
    }
    std::unique_ptr<llvm::orc::LLJIT> jit = std::move(*jit_exp);

    setup_host_jit_symbols(*jit, prog);

    auto sym = [&] {
        scoped_phase_t _p("host-link");
        if (auto err = jit->addObjectFile(llvm::MemoryBuffer::getMemBufferCopy(obj)))
        {
            llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "jit: ");
            fprintf(stderr, "jit: failed to add object to LLJIT\n");
            abort();
        }
        return jit->lookup(lookup_name);
    }();
    if (!sym)
    {
        llvm::logAllUnhandledErrors(sym.takeError(), llvm::errs(), "jit: ");
        fprintf(stderr, "jit: could not resolve '%s' after JIT\n", lookup_name.c_str());
        abort();
    }
    /* keep the JIT (hence the compiled code) alive for the process lifetime */
    void * fn_addr = reinterpret_cast<void *>(static_cast<uintptr_t>(sym->getValue()));
    jit.release();

    /* Cache the compiled function (+ prototype) for further instances of this
     * construct, record the outcome, and install it. A nanos6 outline chain
     * launches as fn(args_v, dev_v, transl_v); a packed entry as fn(args, size);
     * anything else as a uniform void(void**) VARIADIC over prog->args. */
    const int proto = entry_is_nanos6 ? (int) CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_NANOS6
                    : entry_is_packed ? (int) CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_PACKED
                                      : (int) CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_VARIADIC;
    jit_cache().host_put_fn(cache_key, proto, fn_addr);
    if (jit_prof().stats_on) jit_prof().cache_event(false, obj_from_disk ? 1 : 0);
    install_host_jit_result(prog, fn_addr, entry_is_nanos6, entry_is_packed);
    # endif /* CGIR_SUPPORT_LLVM */
}
