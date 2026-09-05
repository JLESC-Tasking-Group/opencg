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

/*
 *  Support machinery for the LLVM-backed passes: environment knobs, opt-in
 *  profiling, and the JIT result cache. See src/jit-support.hpp.
 *
 *  All the process-global state lives here (function-local statics behind
 *  accessors), which is why this is a translation unit and not a header.
 */

# include "jit-support.hpp"

# if CGIR_SUPPORT_LLVM

#  include <llvm/Config/llvm-config.h>      /* LLVM_VERSION_STRING (disk cache salt) */
#  include <llvm/Support/FileSystem.h>
#  include <llvm/Support/MemoryBuffer.h>
#  include <llvm/Support/raw_ostream.h>
#  include <llvm/TargetParser/Host.h>

#  include <algorithm>
#  include <atomic>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#  include <mutex>
#  include <string>
#  include <unordered_map>
#  include <utility>
#  include <vector>

#  include <sys/stat.h>  /* stat (CGIR_JIT_STATS_CSV header-if-new) */
#  include <unistd.h>    /* getpid (atomic temp-file names for the on-disk cache) */

CGIR_NAMESPACE_USE;

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

/* ---------------------------------------------------------------------------
 * Environment knobs and host machine identity.
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

/* Identity of the machine this process is running on: the host CPU name followed
 * by its enabled features, sorted. Only used to salt the host cache key -- what
 * the code is actually compiled for comes from the TargetMachine that
 * JITTargetMachineBuilder::detectHost() configures (see stamp_host_target_attrs)
 * -- but it has to capture the same thing, or a cached object could be reused on
 * a machine it was not compiled for.
 *
 * Sorted because StringMap iteration order is unspecified and this ends up in a
 * hash; copied into std::strings because getHostCPUFeatures() returns the map by
 * value and its keys live in its own storage, which dies with the temporary.
 * Computed once. */
static const std::string &
host_target_id(void)
{
    static const std::string id = [] {
        const auto host = llvm::sys::getHostCPUFeatures();

        std::vector<std::string> enabled;
        enabled.reserve(host.size());
        for (const auto & f : host)
            if (f.second)
                enabled.push_back(f.first().str());
        std::sort(enabled.begin(), enabled.end());

        std::string s = llvm::sys::getHostCPUName().str();
        for (const std::string & f : enabled)
        {
            s += ',';
            s += f;
        }
        return s;
    }();
    return id;
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
 * Content hashing, toolchain salts and the two-level JIT result cache
 * (in-process + optional on-disk).
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
        /* The host CPU *and* its feature set: the module is compiled for whatever
         * the running machine reports (stamp_host_target_attrs), so a cached
         * object is only valid on a machine that reports the same. The code model
         * is in here too, being a codegen-affecting knob. */
        const std::string & id = host_target_id();
        uint64_t h = jit_static_salt();
        h = jit_fnv1a_seed(h, id.data(), id.size());
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

} /* anonymous namespace */

/* ---------------------------------------------------------------------------
 * Public surface (see jit-support.hpp). Thin forwarders: the state above stays
 * private to this translation unit.
 * ------------------------------------------------------------------------- */

bool
CGIR_NAMESPACE::jit::env_flag(const char * var)
{
    return ::env_flag(var);
}

const char *
CGIR_NAMESPACE::jit::env_str(const char * var, const char * dflt)
{
    return ::env_str(var, dflt);
}

bool
CGIR_NAMESPACE::jit::device_assume_noalias_params(void)
{
    return ::device_assume_noalias_params();
}

bool
CGIR_NAMESPACE::jit::profiling_timing_on(void)
{
    return jit_prof().timing_on;
}

bool
CGIR_NAMESPACE::jit::profiling_stats_on(void)
{
    return jit_prof().stats_on;
}

void
CGIR_NAMESPACE::jit::profiling_add_phase(const char * name, double secs)
{
    jit_prof().add_phase(name, secs);
}

void
CGIR_NAMESPACE::jit::profiling_cache_event(bool device, int outcome)
{
    jit_prof().cache_event(device, outcome);
}

uint64_t
CGIR_NAMESPACE::jit::cache_key(const command_prog_t * prog)
{
    return jit_cache_key(prog);
}

bool
CGIR_NAMESPACE::jit::cache_host_get_fn(uint64_t key, int * proto, void ** fn)
{
    jit_cache_t::host_entry_t e;
    if (!jit_cache().host_get_fn(key, e))
        return false;
    if (proto) *proto = e.proto;
    if (fn)    *fn    = e.fn;
    return true;
}

void
CGIR_NAMESPACE::jit::cache_host_put_fn(uint64_t key, int proto, void * fn)
{
    jit_cache().host_put_fn(key, proto, fn);
}

bool
CGIR_NAMESPACE::jit::cache_host_get_obj(uint64_t key, std::string & out)
{
    return jit_cache().host_get_obj(key, out);
}

void
CGIR_NAMESPACE::jit::cache_host_put_obj(uint64_t key, const std::string & obj)
{
    jit_cache().host_put_obj(key, obj);
}

int
CGIR_NAMESPACE::jit::cache_device_get(uint64_t key, std::string & out)
{
    return jit_cache().device_get(key, out);
}

void
CGIR_NAMESPACE::jit::cache_device_put(uint64_t key, const std::string & ptx)
{
    jit_cache().device_put(key, ptx);
}

# endif /* CGIR_SUPPORT_LLVM */
