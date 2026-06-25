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
// (prog-fuse.cc) and the MLIR pass (../mlir/passes/ProgFuse.cpp).
//
// It links the LLVM modules of a chain of programs into a single
// `void __fused_wrapper(void ** args)` that calls each program in order, then
// JIT-compiles it. Two optimizations are applied while building the wrapper:
//
//   1. Argument deduplication. Each input program carries its argument pointers
//      in `launcher.variadic.args` (one void* slot per kernel parameter, each
//      the address of the actual value). Two parameters that receive the SAME
//      slot (same void* address-of-value) are merged into a single slot in the
//      fused args buffer; the wrapper routes both to that slot. The pass itself
//      fills the (compacted) fused buffer at fusion time from the originals'
//      slots, so the args are known/stable for every replay.
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

# include "prog-fuse.hpp"

# if CGIR_SUPPORT_LLVM

# include <llvm/Analysis/InlineCost.h>
# include <llvm/Analysis/ValueTracking.h>
# include <llvm/IR/Function.h>
# include <llvm/IR/IRBuilder.h>
# include <llvm/IR/Instructions.h>
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
# include <llvm/TargetParser/Triple.h>

/* Optimization pipeline (inline + loop-fuse + vectorize) run on the fused
 * module before JIT, so noalias/dedup actually translate into fused loops. */
# include <llvm/Passes/PassBuilder.h>
# include <llvm/Passes/OptimizationLevel.h>
# include <llvm/Transforms/Scalar/LoopFuse.h>
# include <llvm/Transforms/Scalar/LoopPassManager.h>
# include <llvm/Transforms/Scalar/LoopRotation.h>
# include <llvm/Transforms/Utils/LoopSimplify.h>

/* In-process JIT (replaces the former Proteus dependency) */
# include <llvm/ExecutionEngine/Orc/LLJIT.h>
# include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
# include <llvm/Support/Error.h>

# include <atomic>
# include <cassert>
# include <cstdint>
# include <cstdio>
# include <cstdlib>
# include <cstring>
# include <memory>
# include <mutex>
# include <optional>
# include <string>
# include <unordered_map>
# include <vector>

# endif /* CGIR_SUPPORT_LLVM */

CGIR_NAMESPACE_USE;

# if CGIR_SUPPORT_LLVM

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

/* Run an O3 module pipeline (inlining + loop-fuse + vectorization) on the
 * merged module, so the inlined kernels' loops can vectorize/fuse. */
static void
optimize_module(llvm::Module & M, llvm::TargetMachine * tm)
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
     * Compiler-emitted kernels (clang/libomptarget) always use `inbounds` GEPs,
     * so they fuse. We currently RELY on that witness (option (i)): present =>
     * provably safe => fuse; absent => DA stays conservative => fusion is safely
     * skipped. A future option (ii) — for more aggressive coverage of hand-rolled
     * or non-`inbounds` IR — would have this pass stamp `inbounds`/`nsw` onto the
     * inlined kernel GEPs/IVs under cgir's well-formedness (in-bounds, elementwise)
     * contract. That never produces an illegal fusion (LoopFuse still enforces
     * legality) but does assert memory-safety, so it is deferred for now. */
    {
        llvm::FunctionPassManager FPM;
        FPM.addPass(llvm::LoopSimplifyPass());

        llvm::LoopPassManager LPM;
        LPM.addPass(llvm::LoopRotatePass());
        FPM.addPass(llvm::createFunctionToLoopPassAdaptor(std::move(LPM)));

        FPM.addPass(llvm::LoopFusePass());

        llvm::ModulePassManager PreMPM;
        PreMPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
        PreMPM.run(M, MAM);
    }

    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    MPM.run(M, MAM);
}

/* ------------------------------------------------------------------------- *
 * Debug dump option (CGIR_PROG_FUSE_DUMP).                                   *
 *                                                                           *
 * Like CGIR_OPTIMIZER, this is controlled by an environment variable that is *
 * read once and cached. When CGIR_PROG_FUSE_DUMP is set (to anything other  *
 * than "" or "0"), each fusion writes the input IR modules and their fused/  *
 * optimized result as textual .ll files, so the transformation can be        *
 * visualized:                                                                *
 *                                                                           *
 *   <base>/prog-fuse-<seq>/input-<i>.ll   (each original program, as parsed) *
 *   <base>/prog-fuse-<seq>/merged.ll      (linked + wrapper, before O3)      *
 *   <base>/prog-fuse-<seq>/fused.ll       (after the O3 fusion pipeline)     *
 *                                                                           *
 * <base> is ~/.cgir/tmp by default, or the value of CGIR_PROG_FUSE_DUMP when *
 * it is an absolute path (so the output location can be overridden). <seq>   *
 * is a per-process counter so concurrent fusions do not clobber each other.  *
 * ------------------------------------------------------------------------- */
static bool
prog_fuse_dump_enabled(void)
{
    static int enabled = -1;
    if (enabled == -1)
    {
        const char * s = getenv("CGIR_PROG_FUSE_DUMP");
        enabled = (s && s[0] != '\0' && strcmp(s, "0") != 0) ? 1 : 0;
    }
    return (bool) enabled;
}

/* Create (mkdir -p) a fresh per-fusion dump directory and return its path, or
 * an empty string on failure. */
static std::string
prog_fuse_dump_make_dir(void)
{
    /* base directory: an absolute CGIR_PROG_FUSE_DUMP value, else ~/.cgir/tmp */
    std::string base;
    const char * s = getenv("CGIR_PROG_FUSE_DUMP");
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
    char sub[32];
    snprintf(sub, sizeof(sub), "/prog-fuse-%u", seq.fetch_add(1));
    std::string dir = base + sub;

    if (std::error_code ec = llvm::sys::fs::create_directories(dir))
    {
        fprintf(stderr, "prog-fuse: cannot create dump dir '%s': %s\n",
                dir.c_str(), ec.message().c_str());
        return std::string();
    }
    return dir;
}

/* Write `M` as textual IR to <dir>/<name>. No-op if `dir` is empty. */
static void
prog_fuse_dump_module(const std::string & dir, const char * name, llvm::Module & M)
{
    if (dir.empty())
        return ;

    std::string path = dir + "/" + name;
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec)
    {
        fprintf(stderr, "prog-fuse: cannot write '%s': %s\n", path.c_str(), ec.message().c_str());
        return ;
    }
    M.print(os, /* AssemblyAnnotationWriter */ nullptr);
}

# endif /* CGIR_SUPPORT_LLVM */

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

    /* ------------------------------------------------------------------ *
     * 0. One-time LLVM global initialisation                             *
     * ------------------------------------------------------------------ */
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

    /* Optional IR dumping for debugging (CGIR_PROG_FUSE_DUMP). When enabled, a
     * fresh per-fusion directory receives the input, merged and fused IR. */
    const bool  dump     = prog_fuse_dump_enabled();
    std::string dump_dir = dump ? prog_fuse_dump_make_dir() : std::string();

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
            prog_fuse_dump_module(dump_dir, name, *mods[i]);
        }
    }

    /* ------------------------------------------------------------------ *
     * 2/3. Per input: identify the entry function and arity, snapshot its *
     *      argument slots (from launcher.variadic.args), then prefix-rename*
     *      its definitions so they do not clash on link.                  *
     *                                                                     *
     *  An entry is the program's kernel (first void definition). A program *
     *  that is itself a previously-fused module instead exposes a          *
     *  `void __fused_wrapper(void**)` entry; its arity comes from the      *
     *  recorded launcher.variadic.args_size and it is invoked on a slice.  *
     * ------------------------------------------------------------------ */
    struct fuse_input_t
    {
        std::string         fused_name; /* entry name after prefixing */
        bool                is_wrapper; /* true if entry is void(void**) */
        unsigned            arity;      /* number of arg slots consumed */
        std::vector<void *> slots;      /* the originals' arg slots (&value) */
    };
    std::vector<fuse_input_t> inputs(n);

    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Module & M = *mods[i];

        llvm::Function * entry = nullptr;
        bool is_wrapper = false;
        unsigned arity = 0;

        if (llvm::Function * w = M.getFunction("__fused_wrapper"))
        {
            if (!w->isDeclaration() &&
                w->getReturnType()->isVoidTy() &&
                w->getFunctionType()->getNumParams() == 1)
            {
                entry      = w;
                is_wrapper = true;
                arity      = (unsigned) (progs[i]->launcher.variadic.args_size / sizeof(void *));
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
        void ** av = static_cast<void **>(progs[i]->launcher.variadic.args);
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

        inputs[i].fused_name = std::string(prefix) + name;
        inputs[i].is_wrapper = is_wrapper;
        inputs[i].arity      = arity;
    }

    /* ------------------------------------------------------------------ *
     * 3b. Deduplicate argument slots across all leaf kernels.             *
     *                                                                     *
     *  Two parameters that receive the SAME slot (same void* address) map  *
     *  to a single compacted index. `unique_slots` becomes the fused args  *
     *  buffer contents (in first-occurrence order). A wrapper input keeps  *
     *  a contiguous (non-deduplicated) block, since it expects a void**    *
     *  slice.                                                              *
     * ------------------------------------------------------------------ */
    std::vector<void *>                  unique_slots;
    std::unordered_map<void *, unsigned> slot_to_index;
    std::vector<std::vector<unsigned>>   index_map(n);
    std::vector<unsigned>                wrapper_block_start(n, 0);

    for (size_t i = 0 ; i < n ; ++i)
    {
        if (inputs[i].is_wrapper)
        {
            wrapper_block_start[i] = (unsigned) unique_slots.size();
            for (unsigned k = 0 ; k < inputs[i].arity ; ++k)
                unique_slots.push_back(inputs[i].slots[k]);
        }
        else
        {
            index_map[i].resize(inputs[i].arity);
            for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
            {
                void * s = inputs[i].slots[j];
                auto it = slot_to_index.find(s);
                unsigned idx;
                if (it == slot_to_index.end())
                {
                    idx = (unsigned) unique_slots.size();
                    unique_slots.push_back(s);
                    slot_to_index[s] = idx;
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
    }

    /* ------------------------------------------------------------------ *
     * 6. Build the fused wrapper: void __fused_wrapper(void** args)        *
     *    Each kernel reads its (deduplicated) arg slots; a wrapper input   *
     *    is handed its contiguous slice.                                   *
     * ------------------------------------------------------------------ */
    llvm::LLVMContext & llvmctx = mod_u->getContext();
    llvm::Type * void_ty = llvm::Type::getVoidTy(llvmctx);
    llvm::Type * ptr_ty  = llvm::PointerType::getUnqual(llvmctx);
    llvm::Type * i64_ty  = llvm::Type::getInt64Ty(llvmctx);

    llvm::FunctionType * wrapper_fty = llvm::FunctionType::get(void_ty, { ptr_ty }, false);
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

    /* load args[idx] (a void* = &value) and dereference to a value of type T */
    auto load_arg = [&] (unsigned idx, llvm::Type * T) -> llvm::Value *
    {
        llvm::Value * slot  = builder.CreateGEP(ptr_ty, args_ptr,
                                  llvm::ConstantInt::get(i64_ty, idx), "slot");
        llvm::Value * voidp = builder.CreateLoad(ptr_ty, slot, "voidp");
        return builder.CreateLoad(T, voidp, "argval");
    };

    /* &args[off] : a void** slice, for a sub-wrapper call */
    auto slice_ptr = [&] (unsigned off) -> llvm::Value *
    {
        return builder.CreateGEP(ptr_ty, args_ptr,
                                 llvm::ConstantInt::get(i64_ty, off), "slice");
    };

    /* Determine the value type of each deduplicated leaf slot from its first
     * use. Every kernel sharing a slot must read it as the same type (the slot
     * holds one &value); with opaque pointers, pointer args are all `ptr`, and
     * shared scalars (e.g. the length `n`) are consistent. */
    std::vector<llvm::Type *> slot_type(total_args, nullptr);
    for (size_t i = 0 ; i < n ; ++i)
    {
        if (inputs[i].is_wrapper)
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
            slot_value[k] = load_arg(k, slot_type[k]);

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

    /* drop the now-inlined (dead) constituent functions */
    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Function * F = mod_u->getFunction(inputs[i].fused_name);
        if (F && F->use_empty())
            F->eraseFromParent();
    }

    {
        /* One alias-scope domain for the whole fused body; one scope per
         * distinct base pointer value (the deref of each pointer arg slot). */
        llvm::MDBuilder mdb(llvmctx);
        llvm::MDNode *  domain = mdb.createAnonymousAliasScopeDomain("cgir.prog-fuse");

        std::unordered_map<void *, llvm::MDNode *> base_to_scope;
        std::vector<llvm::Metadata *>              all_scopes;
        std::unordered_map<llvm::Value *, void *>  load_base; /* hoisted ptr load -> base */

        for (unsigned k = 0 ; k < total_args ; ++k)
        {
            if (slot_value[k] == nullptr)                                continue; /* wrapper-block slot */
            if (slot_type[k] == nullptr || !slot_type[k]->isPointerTy()) continue; /* not a pointer    */
            if (unique_slots[k] == nullptr)                              continue; /* no &value         */

            void * base = *static_cast<void **>(unique_slots[k]);
            if (base == nullptr)
                continue;

            load_base[slot_value[k]] = base;
            if (base_to_scope.find(base) == base_to_scope.end())
            {
                llvm::MDNode * sc = mdb.createAnonymousAliasScope(domain, "cgir.prog-fuse.ptr");
                base_to_scope[base] = sc;
                all_scopes.push_back(sc);
            }
        }

        /* With <2 distinct bases there is nothing to disambiguate. */
        if (all_scopes.size() >= 2)
        {
            for (llvm::BasicBlock & BB : *wrapper)
            {
                for (llvm::Instruction & I : BB)
                {
                    if (!I.mayReadOrWriteMemory())
                        continue;
                    llvm::Value * ptr = llvm::getLoadStorePointerOperand(&I);
                    if (ptr == nullptr)
                        continue;

                    auto it = load_base.find(llvm::getUnderlyingObject(ptr));
                    if (it == load_base.end())
                        continue;

                    llvm::Metadata * own = base_to_scope[it->second];
                    I.setMetadata(llvm::LLVMContext::MD_alias_scope,
                                  llvm::MDNode::get(llvmctx, { own }));

                    std::vector<llvm::Metadata *> others;
                    others.reserve(all_scopes.size() - 1);
                    for (llvm::Metadata * s : all_scopes)
                        if (s != own)
                            others.push_back(s);
                    I.setMetadata(llvm::LLVMContext::MD_noalias,
                                  llvm::MDNode::get(llvmctx, others));
                }
            }
        }
    }

    /* ------------------------------------------------------------------ *
     * 7. Stamp the host triple + data layout (parsed IR omits them) and    *
     *    create a host TargetMachine (also used by the opt pipeline).      *
     * ------------------------------------------------------------------ */
    std::unique_ptr<llvm::TargetMachine> tm;
    {
        if (mod_u->getTargetTriple().empty())
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
            TT, llvm::sys::getHostCPUName(), /* features */ "", opts,
            /* reloc */ std::nullopt, /* code model */ std::nullopt,
            /* match the O3 IR pipeline below for end-to-end aggressive codegen */
            llvm::CodeGenOptLevel::Aggressive));

        if (tm && mod_u->getDataLayout().isDefault())
            mod_u->setDataLayout(tm->createDataLayout());
    }

    if (llvm::verifyModule(*mod_u, &llvm::errs()))
    {
        fprintf(stderr, "prog-fuse: merged module verification failed\n");
        abort();
    }

    /* dump the merged module (wrapper + constituents) before optimization */
    if (dump)
        prog_fuse_dump_module(dump_dir, "merged.ll", *mod_u);

    /* ------------------------------------------------------------------ *
     * 8. Optimize (inline the kernels into the wrapper, vectorize, fuse).  *
     * ------------------------------------------------------------------ */
    if (tm)
        optimize_module(*mod_u, tm.get());

    /* dump the final fused/optimized module */
    if (dump)
    {
        prog_fuse_dump_module(dump_dir, "fused.ll", *mod_u);
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

    /* ------------------------------------------------------------------ *
     * 10. JIT the optimized module and resolve the wrapper.               *
     * ------------------------------------------------------------------ */
    auto jit_exp = llvm::orc::LLJITBuilder().create();
    if (!jit_exp)
    {
        llvm::logAllUnhandledErrors(jit_exp.takeError(), llvm::errs(), "prog-fuse: ");
        fprintf(stderr, "prog-fuse: failed to create LLJIT\n");
        abort();
    }
    std::unique_ptr<llvm::orc::LLJIT> jit = std::move(*jit_exp);

    llvm::orc::ThreadSafeModule tsm(std::move(mod_u), std::move(ctx));
    if (auto err = jit->addIRModule(std::move(tsm)))
    {
        llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "prog-fuse: ");
        fprintf(stderr, "prog-fuse: failed to add IR module to LLJIT\n");
        abort();
    }

    auto sym = jit->lookup("__fused_wrapper");
    if (!sym)
    {
        llvm::logAllUnhandledErrors(sym.takeError(), llvm::errs(), "prog-fuse: ");
        fprintf(stderr, "prog-fuse: could not resolve '__fused_wrapper' after JIT\n");
        abort();
    }
    void * fn_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(sym->getValue()));

    /* keep the JIT (hence the compiled code) alive for the process lifetime */
    jit.release();

    /* ------------------------------------------------------------------ *
     * 11. Fill the compacted args buffer with the deduplicated slots and   *
     *     install the variadic launcher. dst may alias progs[0]; we read   *
     *     all originals' slots in step 2, so writing dst now is safe.      *
     * ------------------------------------------------------------------ */
    const size_t n_args = (size_t) total_args;
    void ** args_buf = static_cast<void **>(calloc(n_args ? n_args : 1, sizeof(void *)));
    if (!args_buf) { fprintf(stderr, "prog-fuse: calloc failed\n"); abort(); }
    for (unsigned k = 0 ; k < total_args ; ++k)
        args_buf[k] = unique_slots[k];

    /* Free the previous args buffer iff a prior fusion produced (owns) it. dst
     * may alias progs[0], whose arg slot VALUES were copied into unique_slots
     * in step 2/3b, so the old buffer is no longer referenced and is safe to
     * release here. */
    if (dst->launcher.variadic._args_owned && dst->launcher.variadic.args)
        free(dst->launcher.variadic.args);

    dst->launcher.variadic.fn          = fn_ptr;
    dst->launcher.variadic.args        = args_buf;
    dst->launcher.variadic.args_size   = n_args * sizeof(void *);
    dst->launcher.variadic._args_owned = true;  /* heap (calloc) — the pass owns it */

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

        if (progs[i]->launcher.variadic._args_owned && progs[i]->launcher.variadic.args)
        {
            free(progs[i]->launcher.variadic.args);
            progs[i]->launcher.variadic.args        = nullptr;
            progs[i]->launcher.variadic.args_size   = 0;
            progs[i]->launcher.variadic._args_owned = false;
        }
    }

    # endif /* CGIR_SUPPORT_LLVM */
}
