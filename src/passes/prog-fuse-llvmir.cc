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

# include <llvm/IR/Function.h>
# include <llvm/IR/IRBuilder.h>
# include <llvm/IR/LLVMContext.h>
# include <llvm/IR/Module.h>
# include <llvm/IR/Verifier.h>
# include <llvm/IRReader/IRReader.h>
# include <llvm/Linker/Linker.h>
# include <llvm/Bitcode/BitcodeWriter.h>
# include <llvm/MC/TargetRegistry.h>
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

/* In-process JIT (replaces the former Proteus dependency) */
# include <llvm/ExecutionEngine/Orc/LLJIT.h>
# include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
# include <llvm/Support/Error.h>

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

    /* Insert loop fusion into the default pipeline, after the scalar loop
     * optimizations and before vectorization. LoopFuse is experimental and a
     * no-op when fusion is illegal, so this is a best-effort enabler. */
    PB.registerScalarOptimizerLateEPCallback(
        [] (llvm::FunctionPassManager & FPM, llvm::OptimizationLevel)
        {
            FPM.addPass(llvm::LoopFusePass());
        });

    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    MPM.run(M, MAM);
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
     * 5. Make the constituent functions inlinable, and mark noalias on    *
     *    pointer parameters whose pointer value is distinct from the       *
     *    kernel's other pointer parameters (restrict-like).               *
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

        if (inputs[i].is_wrapper)
            continue;

        llvm::FunctionType * fty = F->getFunctionType();
        for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
        {
            if (!fty->getParamType(j)->isPointerTy())
                continue;
            if (inputs[i].slots[j] == nullptr)
                continue;

            /* the slot holds &ptr; dereference once to get the actual base */
            void * base_j = *static_cast<void **>(inputs[i].slots[j]);

            bool distinct = true;
            for (unsigned k = 0 ; k < inputs[i].arity ; ++k)
            {
                if (k == j) continue;
                if (!fty->getParamType(k)->isPointerTy()) continue;
                if (inputs[i].slots[k] == nullptr) continue;
                if (*static_cast<void **>(inputs[i].slots[k]) == base_j)
                {
                    distinct = false;
                    break;
                }
            }

            if (distinct)
                F->addParamAttr(j, llvm::Attribute::NoAlias);
        }
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

    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Function * fn = mod_u->getFunction(inputs[i].fused_name);
        llvm::FunctionType * fty = fn->getFunctionType();

        if (inputs[i].is_wrapper)
        {
            builder.CreateCall(fty, fn, { slice_ptr(wrapper_block_start[i]) });
        }
        else
        {
            std::vector<llvm::Value *> call_args;
            call_args.reserve(inputs[i].arity);
            for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
                call_args.push_back(load_arg(index_map[i][j], fty->getParamType(j)));
            builder.CreateCall(fty, fn, call_args);
        }
    }

    builder.CreateRetVoid();

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

    /* ------------------------------------------------------------------ *
     * 8. Optimize (inline the kernels into the wrapper, vectorize, fuse).  *
     * ------------------------------------------------------------------ */
    if (tm)
        optimize_module(*mod_u, tm.get());

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
        dst->source.content.llvmir.owned &&
        dst->source.content.llvmir.raw)
    {
        free(dst->source.content.llvmir.raw);
    }

    dst->source.type                 = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    dst->source.content.llvmir.raw   = bc_buf;
    dst->source.content.llvmir.size  = bitcode.size();
    dst->source.content.llvmir.owned = true;   /* heap (malloc) — the pass owns it */

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
    if (dst->launcher.variadic.args_owned && dst->launcher.variadic.args)
        free(dst->launcher.variadic.args);

    dst->launcher.variadic.fn         = fn_ptr;
    dst->launcher.variadic.args       = args_buf;
    dst->launcher.variadic.args_size  = n_args * sizeof(void *);
    dst->launcher.variadic.args_owned = true;  /* heap (calloc) — the pass owns it */

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
            progs[i]->source.content.llvmir.owned &&
            progs[i]->source.content.llvmir.raw)
        {
            free(progs[i]->source.content.llvmir.raw);
            progs[i]->source.content.llvmir.raw   = nullptr;
            progs[i]->source.content.llvmir.size  = 0;
            progs[i]->source.content.llvmir.owned = false;
        }

        if (progs[i]->launcher.variadic.args_owned && progs[i]->launcher.variadic.args)
        {
            free(progs[i]->launcher.variadic.args);
            progs[i]->launcher.variadic.args       = nullptr;
            progs[i]->launcher.variadic.args_size  = 0;
            progs[i]->launcher.variadic.args_owned = false;
        }
    }

    # endif /* CGIR_SUPPORT_LLVM */
}
