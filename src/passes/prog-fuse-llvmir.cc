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

/* In-process JIT (replaces the former Proteus dependency) */
# include <llvm/ExecutionEngine/Orc/LLJIT.h>
# include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
# include <llvm/Support/Error.h>

# include <cstdint>
# include <cstdio>
# include <cstdlib>
# include <cstring>
# include <memory>
# include <mutex>
# include <optional>
# include <string>
# include <vector>

# endif /* CGIR_SUPPORT_LLVM */

CGIR_NAMESPACE_USE;

# if CGIR_SUPPORT_LLVM

/**
 *  Fused launcher state: holds the compiled JIT module and the function
 *  pointer for the sequential wrapper, plus a flat args buffer concatenating
 *  every fused program's argument layout.
 *
 *  The variadic launcher stores:
 *    fn        -> address of the compiled __fused_wrapper symbol
 *    args      -> heap-allocated array of (sum of all arities) void* pointers
 *    args_size -> (sum of all arities) * sizeof(void *)
 *
 *  The wrapper has the signature:
 *    void __fused_wrapper(void** args)
 *
 *  It unpacks the void* array and calls each program's entry in order. Fusion
 *  is composable: a program that is itself a fused wrapper is invoked on its
 *  args slice, so chains (a -> b -> c -> ...) fuse correctly.
 *
 *  The JIT must stay alive for as long as the node is reachable, so we leak it
 *  intentionally. (A proper implementation would tie its lifetime to the
 *  command_prog_t.)
 */

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
 *
 *  We explicitly require a void return type to avoid accidentally selecting
 *  a non-void helper that happens to appear before the actual kernel in the
 *  module's function list.
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
 *  We only rename definitions (not declarations) to avoid touching
 *  external references that might matter.
 */
static void
prefix_functions(llvm::Module & M, const std::string & prefix)
{
    /* Collect first to avoid iterator invalidation during rename */
    std::vector<llvm::Function *> defs;
    for (llvm::Function & F : M)
        if (!F.isDeclaration())
            defs.push_back(&F);

    for (llvm::Function * F : defs)
        F->setName(prefix + F->getName().str());
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
     *                                                                     *
     *  LLVM's ManagedStatic infrastructure (statistics, target registry, *
     *  command-line option tables, …) must be bootstrapped before ANY    *
     *  other LLVM API is called, including parseIR().  Calling these     *
     *  before the targets are registered triggers StringMap assertion    *
     *  failures on recent LLVM builds (>= 21).                           *
     *                                                                     *
     *  Registering all targets covers both the host (needed by the LLJIT  *
     *  in step 7) and any cross-target the fused IR might reference. We    *
     *  guard with std::call_once so registration is idempotent across     *
     *  multiple fusion calls.                                             *
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
     *                                                                     *
     *  LLVM types and IR values are context-specific: mixing types from  *
     *  different LLVMContext objects causes undefined behaviour. We       *
     *  therefore parse every IR blob into a single shared context so that *
     *  llvm::Linker and IRBuilder can freely reference types from any of  *
     *  them.                                                               *
     *                                                                     *
     *  Ownership of the context is transferred to the JIT at step 7 (we   *
     *  intentionally leak it to keep the compiled code alive).            *
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
     * 2/3. Identify each program's entry function and arity, then rename   *
     *      its definitions with a unique prefix to avoid clashes on link.  *
     *                                                                     *
     *  An entry is normally the program's kernel (first void definition). *
     *  If a program is itself a previously-fused module it instead exposes *
     *  a `void __fused_wrapper(void**)` entry; we then call it with a      *
     *  pointer into the args slice and take its arity from the recorded    *
     *  launcher.variadic.args_size. This makes fusion composable (chains). *
     * ------------------------------------------------------------------ */

    struct fuse_input_t
    {
        std::string fused_name; /* entry name after prefixing */
        bool        is_wrapper; /* true if entry is void(void**) */
        unsigned    arity;      /* number of arg slots consumed */
    };
    std::vector<fuse_input_t> inputs(n);

    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Module & M = *mods[i];

        llvm::Function * entry = nullptr;
        bool is_wrapper = false;
        unsigned arity = 0;

        /* a previously-fused program exposes void __fused_wrapper(void**) */
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

        /* otherwise it is an ordinary kernel */
        if (entry == nullptr)
        {
            entry = find_kernel(M);
            if (!entry) { fprintf(stderr, "prog-fuse: no kernel found in program %zu\n", i); abort(); }
            is_wrapper = false;
            arity      = entry->getFunctionType()->getNumParams();
        }

        const std::string name = entry->getName().str();

        char prefix[32];
        snprintf(prefix, sizeof(prefix), "__fz%zu_", i);
        prefix_functions(M, prefix);

        inputs[i].fused_name = std::string(prefix) + name;
        inputs[i].is_wrapper = is_wrapper;
        inputs[i].arity      = arity;
    }

    /* ------------------------------------------------------------------ *
     * 4. Link every module into the first one (the merged module).        *
     *    All share the same LLVMContext, so no cross-context cloning.      *
     *    linkInModule takes ownership of each linked-in module.            *
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
     * 5. Build the fused wrapper function inside the merged module        *
     *                                                                     *
     *  The wrapper has the signature:                                     *
     *    void __fused_wrapper(void** args)                                *
     *                                                                     *
     *  Layout of the args array (each element is a void* pointing to     *
     *  the actual value; matches the standard CUDA/variadic convention): *
     *  the concatenation of each program's argument pointers, in order.  *
     *                                                                     *
     *  For each argument we:                                              *
     *    1. GEP into the args array to get args[i]  (a void*)            *
     *    2. Load the void* stored there             (the arg pointer)    *
     *    3. Load the typed value through that pointer                     *
     *                                                                     *
     *  This double-dereference matches how the caller is expected to     *
     *  populate the args buffer:                                          *
     *    double s = 2.0;  args[0] = &s;                                  *
     *    double *y = ...; args[1] = &y;                                  *
     *    int64_t n = 8;   args[2] = &n;                                  *
     * ------------------------------------------------------------------ */

    llvm::LLVMContext & llvmctx = mod_u->getContext();
    llvm::Type * void_ty   = llvm::Type::getVoidTy(llvmctx);
    llvm::Type * ptr_ty    = llvm::PointerType::getUnqual(llvmctx);  /* opaque ptr (void*) */
    llvm::Type * i64_ty    = llvm::Type::getInt64Ty(llvmctx);

    /* Create:  void __fused_wrapper(void** args) */
    llvm::FunctionType * wrapper_fty = llvm::FunctionType::get(void_ty, { ptr_ty }, false);
    llvm::Function * wrapper = llvm::Function::Create(
        wrapper_fty,
        llvm::GlobalValue::ExternalLinkage,
        "__fused_wrapper",
        mod_u.get()
    );

    llvm::BasicBlock * bb = llvm::BasicBlock::Create(llvmctx, "entry", wrapper);
    llvm::IRBuilder<> builder(bb);

    llvm::Value * args_ptr = wrapper->getArg(0);  /* void** args */

    /* Helper: load args[idx] and dereference to obtain a value of type T.
     *
     *   slot  = &args[idx]              (GEP into the outer void** array)
     *   voidp = *slot                   (load the void* stored at args[idx])
     *   value = *(T*)voidp              (load the actual typed value)
     */
    auto load_arg = [&](unsigned idx, llvm::Type * T) -> llvm::Value *
    {
        llvm::Value * slot  = builder.CreateGEP(ptr_ty, args_ptr,
                                  llvm::ConstantInt::get(i64_ty, idx), "slot");
        llvm::Value * voidp = builder.CreateLoad(ptr_ty, slot, "voidp");
        return builder.CreateLoad(T, voidp, "argval");
    };

    /* Helper: pointer to args[off] (a void** slice), for sub-wrapper calls. */
    auto slice_ptr = [&](unsigned off) -> llvm::Value *
    {
        return builder.CreateGEP(ptr_ty, args_ptr,
                                 llvm::ConstantInt::get(i64_ty, off), "slice");
    };

    /* Emit, in order, a call to each program's entry, consuming a contiguous
     * slice of the args array. The fused args layout is the concatenation of
     * every program's argument pointers:
     *   [ prog0 args | prog1 args | ... | progN-1 args ]
     * An ordinary kernel's slice is unpacked into typed values; a fused
     * sub-wrapper is handed &args[off] directly (it unpacks its own slice). */
    unsigned off = 0;
    for (size_t i = 0 ; i < n ; ++i)
    {
        llvm::Function * fn = mod_u->getFunction(inputs[i].fused_name);
        if (!fn)
        {
            fprintf(stderr, "prog-fuse: symbol '%s' missing after link\n", inputs[i].fused_name.c_str());
            abort();
        }
        llvm::FunctionType * fty = fn->getFunctionType();

        if (inputs[i].is_wrapper)
        {
            builder.CreateCall(fty, fn, { slice_ptr(off) });
        }
        else
        {
            std::vector<llvm::Value *> call_args;
            for (unsigned j = 0 ; j < inputs[i].arity ; ++j)
                call_args.push_back(load_arg(off + j, fty->getParamType(j)));
            builder.CreateCall(fty, fn, call_args);
        }

        off += inputs[i].arity;
    }

    builder.CreateRetVoid();

    const unsigned total_args = off;

    /* ------------------------------------------------------------------ *
     * 6. Stamp the host target triple and data layout onto the merged    *
     *    module, then verify it.                                          *
     *                                                                     *
     *  Parsed IR strings have no target triple or data layout (the .ll   *
     *  inputs omit them). The JIT needs a concrete host data layout for   *
     *  correct code generation and symbol mangling, and serialisation     *
     *  walks the DataLayout fields. We set the process triple, then       *
     *  create a TargetMachine for the host and use its DataLayout.        *
     * ------------------------------------------------------------------ */
    {
        /* Set the target triple to the host process triple if not set. */
        if (mod_u->getTargetTriple().empty())
            mod_u->setTargetTriple(llvm::Triple(llvm::sys::getProcessTriple()));

        /* Create a TargetMachine for the host to get the correct DataLayout. */
        if (mod_u->getDataLayout().isDefault())
        {
            /* Both lookupTarget() and createTargetMachine() take Triple on
             * LLVM >= 22.  Use getTargetTriple() which returns const Triple&. */
            const llvm::Triple & TT = mod_u->getTargetTriple();
            std::string err;
            const llvm::Target * tgt = llvm::TargetRegistry::lookupTarget(TT, err);
            if (!tgt)
            {
                fprintf(stderr, "prog-fuse: cannot find target '%s': %s\n",
                        TT.str().c_str(), err.c_str());
                abort();
            }
            llvm::TargetOptions opts;
            std::unique_ptr<llvm::TargetMachine> tm(
                tgt->createTargetMachine(
                    TT,
                    llvm::sys::getHostCPUName(),
                    /* features */ "",
                    opts,
                    /* reloc      */ std::nullopt,
                    /* code model */ std::nullopt,
                    llvm::CodeGenOptLevel::Default
                )
            );
            if (tm)
                mod_u->setDataLayout(tm->createDataLayout());
        }
    }

    if (llvm::verifyModule(*mod_u, &llvm::errs()))
    {
        fprintf(stderr, "prog-fuse: merged module verification failed\n");
        abort();
    }

    /* ------------------------------------------------------------------ *
     * 6b. Serialise to LLVM bitcode and store it as the fused node's      *
     *     source blob (the in-memory module itself is consumed by the JIT *
     *     in step 7).                                                      *
     *                                                                     *
     *  We use the bitcode writer rather than the text printer because the *
     *  text printer (Module::print / AssemblyWriter::printFunction) reads *
     *  function attributes and argument metadata that may contain         *
     *  uninitialised padding when the input IR used legacy typed-pointer  *
     *  syntax (double*, i64*) auto-promoted to opaque pointers. The       *
     *  bitcode writer serialises the in-memory IR binary fields directly  *
     *  and never exercises the attribute-printing path, producing a       *
     *  clean, well-defined byte stream.                                   *
     * ------------------------------------------------------------------ */
    std::string bitcode;
    {
        llvm::raw_string_ostream os(bitcode);
        llvm::WriteBitcodeToFile(*mod_u, os);
    }

    /* Store the bitcode blob as the fused node's source.
     * dst may alias progs[0] (in-place fusion); we update the pointer last.
     * The old buffer (user-supplied static literal or prior malloc) is
     * intentionally not freed — ownership tracking is absent in
     * command_prog_t.  TODO: add an ownership flag for proper cleanup. */
    char * bc_buf = static_cast<char *>(malloc(bitcode.size()));
    if (!bc_buf) { fprintf(stderr, "prog-fuse: malloc failed\n"); abort(); }
    memcpy(bc_buf, bitcode.data(), bitcode.size());

    dst->source.type                  = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    dst->source.content.llvmir.raw    = bc_buf;
    dst->source.content.llvmir.size   = bitcode.size();

    /* ------------------------------------------------------------------ *
     * 7. JIT-compile the fused module in-process (LLVM ORC LLJIT) and     *
     *    set up the variadic launcher so the fused function is directly   *
     *    callable.                                                         *
     *                                                                     *
     *  launcher.variadic layout:                                          *
     *    fn        -> void __fused_wrapper(void** args)                  *
     *    args      -> void*[total_args]  (runtime fills arg pointers here)*
     *    args_size -> total_args * sizeof(void*)                          *
     * ------------------------------------------------------------------ */

    /* Create an LLJIT. It must outlive the JIT'd code, so we intentionally
     * leak it (its lifetime is conceptually tied to the fused command). */
    auto jit_exp = llvm::orc::LLJITBuilder().create();
    if (!jit_exp)
    {
        llvm::logAllUnhandledErrors(jit_exp.takeError(), llvm::errs(), "prog-fuse: ");
        fprintf(stderr, "prog-fuse: failed to create LLJIT\n");
        abort();
    }
    std::unique_ptr<llvm::orc::LLJIT> jit = std::move(*jit_exp);

    /* hand the merged module (and its owning context) to the JIT */
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

    /* Allocate the args buffer that the runtime will fill before calling fn.
     * Each slot holds a void* pointing to an actual argument value.
     * The buffer is zeroed; the runtime is responsible for populating it. */
    const size_t n_args = (size_t) total_args;
    void ** args_buf = static_cast<void **>(calloc(n_args, sizeof(void *)));
    if (!args_buf && n_args > 0)
    {
        fprintf(stderr, "prog-fuse: calloc failed\n");
        abort();
    }

    dst->launcher.variadic.fn        = fn_ptr;
    dst->launcher.variadic.args      = args_buf;
    dst->launcher.variadic.args_size = n_args * sizeof(void *);

    # endif /* CGIR_SUPPORT_LLVM */
}
