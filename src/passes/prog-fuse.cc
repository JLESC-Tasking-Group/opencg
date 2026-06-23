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

# include <opencg/namespace.hpp>
# include <opencg/command.hpp>
# include <opencg/command-graph.hpp>

# if OPENCG_SUPPORT_LLVM

# include <llvm/IR/Function.h>
# include <llvm/IR/IRBuilder.h>
# include <llvm/IR/LLVMContext.h>
# include <llvm/IR/Module.h>
# include <llvm/IR/Verifier.h>
# include <llvm/IRReader/IRReader.h>
# include <llvm/Linker/Linker.h>
# include <llvm/Bitcode/BitcodeWriter.h>
# include <llvm/MC/TargetRegistry.h>
# include <llvm/Support/InitLLVM.h>
# include <llvm/Support/MemoryBuffer.h>
# include <llvm/Support/raw_ostream.h>
# include <llvm/Support/SourceMgr.h>
# include <llvm/Support/TargetSelect.h>
# include <llvm/Target/TargetMachine.h>
# include <llvm/TargetParser/Host.h>

/* In-process JIT (replaces the former Proteus dependency) */
# include <llvm/ExecutionEngine/Orc/LLJIT.h>
# include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
# include <llvm/Support/Error.h>

# include <cstring>
# include <memory>
# include <mutex>
# include <string>
# include <vector>

# endif /* OPENCG_SUPPORT_LLVM */

OCG_NAMESPACE_USE;

/* pass local storage */
struct pls_t
{
    bool contracted;

    pls_t(void) : contracted(false) {}
    ~pls_t(void) {}
};

using node_t = command_graph_t::node_iterator_t<pls_t>;

# if OPENCG_SUPPORT_LLVM

/**
 *  Fused launcher state: holds the compiled JIT module and the function
 *  pointer for the sequential wrapper, plus a flat args buffer combining
 *  u's and v's argument layouts.
 *
 *  The variadic launcher stores:
 *    fn        -> address of the compiled __fused_wrapper symbol
 *    args      -> heap-allocated array of (nu + nv) void* pointers
 *    args_size -> (nu + nv) * sizeof(void *)
 *
 *  The wrapper has the signature:
 *    void __fused_wrapper(void** args)
 *
 *  It unpacks the void* array, casts each element back to the correct
 *  type, and calls u_func followed by v_func.
 *
 *  The LLVMIRJitModule must stay alive for as long as the node is
 *  reachable, so we heap-allocate it and leak intentionally here.
 *  (A proper implementation would tie its lifetime to the command_prog_t.)
 */

/**
 *  Parse a null-terminated LLVM IR string into an llvm::Module.
 *  Returns nullptr and prints a diagnostic on failure.
 */
static std::unique_ptr<llvm::Module>
parse_llvmir(const char * ir, size_t size, llvm::LLVMContext & ctx)
{
    /* MemoryBuffer does not need the string to be null-terminated when the
     * size is passed explicitly, but the string from the test IS null-
     * terminated, so both forms work.  Use the size-aware variant to be safe
     * with binary bitcode inputs. */
    llvm::StringRef sr(ir, size > 0 ? size - 1 : 0); /* strip trailing NUL if any */
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

# endif /* OPENCG_SUPPORT_LLVM */

static inline void
command_graph_pass_prog_fuse_llvmir(
    command_graph_t * cg,
    command_prog_t * pu,
    command_prog_t * pv,
    command_prog_t * puv
) {
    # if !OPENCG_SUPPORT_LLVM
    fprintf(stderr, "prog-fuse: LLVM support not enabled (rebuild with -DUSE_LLVM=ON)\n");
    abort();
    # else

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
     * 1. Parse both IR modules into the SAME LLVMContext                 *
     *                                                                     *
     *  LLVM types and IR values are context-specific: mixing types from  *
     *  two different LLVMContext objects causes undefined behaviour.      *
     *  We therefore parse both IR strings into a single shared context   *
     *  so that llvm::Linker and IRBuilder can freely reference types     *
     *  from either module.                                                *
     *                                                                     *
     *  Ownership of the context is transferred to the heap-allocated     *
     *  LLVMIRJitModule at step 7 (we intentionally leak both to keep     *
     *  the compiled code alive for the lifetime of the node).            *
     * ------------------------------------------------------------------ */

    auto ctx = std::make_unique<llvm::LLVMContext>();

    auto mod_u = parse_llvmir(
        static_cast<const char *>(pu->source.content.llvmir.raw),
        pu->source.content.llvmir.size,
        *ctx
    );
    if (!mod_u) { fprintf(stderr, "prog-fuse: failed to parse u IR\n"); abort(); }

    auto mod_v = parse_llvmir(
        static_cast<const char *>(pv->source.content.llvmir.raw),
        pv->source.content.llvmir.size,
        *ctx
    );
    if (!mod_v) { fprintf(stderr, "prog-fuse: failed to parse v IR\n"); abort(); }

    /* ------------------------------------------------------------------ *
     * 2. Identify the kernel function in each module                      *
     * ------------------------------------------------------------------ */

    llvm::Function * fu = find_kernel(*mod_u);
    if (!fu) { fprintf(stderr, "prog-fuse: no kernel found in u IR\n"); abort(); }

    llvm::Function * fv = find_kernel(*mod_v);
    if (!fv) { fprintf(stderr, "prog-fuse: no kernel found in v IR\n"); abort(); }

    /* Remember original names before renaming */
    std::string name_u = fu->getName().str();
    std::string name_v = fv->getName().str();

    /* Collect the arity of each kernel (parameter count).
     * We must snapshot these counts now because the Function objects
     * will be consumed/moved when mod_v is linked into mod_u. */
    const unsigned nu = fu->getFunctionType()->getNumParams();
    const unsigned nv = fv->getFunctionType()->getNumParams();

    /* ------------------------------------------------------------------ *
     * 3. Rename functions to avoid symbol clashes after linking           *
     * ------------------------------------------------------------------ */

    const std::string prefix_u = "__fused_u_";
    const std::string prefix_v = "__fused_v_";

    prefix_functions(*mod_u, prefix_u);
    prefix_functions(*mod_v, prefix_v);

    const std::string fused_name_u = prefix_u + name_u;
    const std::string fused_name_v = prefix_v + name_v;

    /* ------------------------------------------------------------------ *
     * 4. Link mod_v into mod_u using LLVM's IR linker                    *
     *                                                                     *
     *  Both modules share the same LLVMContext so no cross-context       *
     *  cloning is required.  linkInModule takes ownership of mod_v.      *
     * ------------------------------------------------------------------ */

    llvm::Linker linker(*mod_u);
    /* linkInModule takes ownership; returns true on failure */
    if (linker.linkInModule(std::move(mod_v)))
    {
        fprintf(stderr, "prog-fuse: IR linking failed\n");
        abort();
    }

    /* ------------------------------------------------------------------ *
     * 5. Build the fused wrapper function inside the merged module        *
     *                                                                     *
     *  The wrapper has the signature:                                     *
     *    void __fused_wrapper(void** args)                                *
     *                                                                     *
     *  Layout of the args array (each element is a void* pointing to     *
     *  the actual value; matches the standard CUDA/variadic convention): *
     *    args[0 .. nu-1]      -> argument pointers for u's kernel        *
     *    args[nu .. nu+nv-1]  -> argument pointers for v's kernel        *
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

    /* Look up the renamed functions in the merged module.
     * After linking, both __fused_u_<name> and __fused_v_<name> are
     * defined in mod_u.  Retrieve their types from the merged module so
     * that all LLVM type objects belong to the same LLVMContext. */
    llvm::Function * call_u = mod_u->getFunction(fused_name_u);
    llvm::Function * call_v = mod_u->getFunction(fused_name_v);

    if (!call_u)
    {
        fprintf(stderr, "prog-fuse: symbol '%s' missing after link\n", fused_name_u.c_str());
        abort();
    }
    if (!call_v)
    {
        fprintf(stderr, "prog-fuse: symbol '%s' missing after link\n", fused_name_v.c_str());
        abort();
    }

    /* Build call to u's kernel: call_u(args[0], ..., args[nu-1])
     * Parameter types come from the merged module's function signature. */
    {
        llvm::FunctionType * fty = call_u->getFunctionType();
        std::vector<llvm::Value *> call_args;
        for (unsigned i = 0; i < nu; ++i)
            call_args.push_back(load_arg(i, fty->getParamType(i)));
        builder.CreateCall(fty, call_u, call_args);
    }

    /* Build call to v's kernel: call_v(args[nu], ..., args[nu+nv-1]) */
    {
        llvm::FunctionType * fty = call_v->getFunctionType();
        std::vector<llvm::Value *> call_args;
        for (unsigned i = 0; i < nv; ++i)
            call_args.push_back(load_arg(nu + i, fty->getParamType(i)));
        builder.CreateCall(fty, call_v, call_args);
    }

    builder.CreateRetVoid();

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
     * puv may alias pu (in-place fusion); we update the pointer last.
     * The old buffer (user-supplied static literal or prior malloc) is
     * intentionally not freed — ownership tracking is absent in
     * command_prog_t.  TODO: add an ownership flag for proper cleanup. */
    char * bc_buf = static_cast<char *>(malloc(bitcode.size()));
    if (!bc_buf) { fprintf(stderr, "prog-fuse: malloc failed\n"); abort(); }
    memcpy(bc_buf, bitcode.data(), bitcode.size());

    puv->source.type                  = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    puv->source.content.llvmir.raw    = bc_buf;
    puv->source.content.llvmir.size   = bitcode.size();

    /* ------------------------------------------------------------------ *
     * 7. JIT-compile the fused module in-process (LLVM ORC LLJIT) and     *
     *    set up the variadic launcher so the fused function is directly   *
     *    callable.                                                         *
     *                                                                     *
     *  launcher.variadic layout:                                          *
     *    fn        -> void __fused_wrapper(void** args)                  *
     *    args      -> void*[nu + nv]  (runtime fills arg pointers here)  *
     *    args_size -> (nu + nv) * sizeof(void*)                          *
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
    const size_t n_args = static_cast<size_t>(nu) + static_cast<size_t>(nv);
    void ** args_buf = static_cast<void **>(calloc(n_args, sizeof(void *)));
    if (!args_buf && n_args > 0)
    {
        fprintf(stderr, "prog-fuse: calloc failed\n");
        abort();
    }

    puv->launcher.variadic.fn        = fn_ptr;
    puv->launcher.variadic.args      = args_buf;
    puv->launcher.variadic.args_size = n_args * sizeof(void *);

    # endif /* OPENCG_SUPPORT_LLVM */
}

static inline void
command_graph_pass_prog_fuse(
    command_graph_t * cg,
    command_graph_node_t * u,
    command_graph_node_t * v,
    std::vector<node_t> & nodes
) {
    /* assert tests */
    assert(u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(u->command);
    assert(u->command->type == COMMAND_TYPE_PROG);

    assert(v->command);
    assert(v->type == COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(v->command->type == COMMAND_TYPE_PROG);

    assert(std::find(u->successors.begin(),   u->successors.end(),   v) != u->successors.end());
    assert(std::find(v->predecessors.begin(), v->predecessors.end(), u) != v->predecessors.end());

    /* only llvm ir supported currently */
    if (u->command->prog.source.type != COMMAND_PROG_SOURCE_TYPE_LLVMIR ||
        v->command->prog.source.type != COMMAND_PROG_SOURCE_TYPE_LLVMIR)
    {
        fprintf(stderr, "Only supporting LLVMIR source type for kernel fusion\n");
        abort();
    }

    /* fuse programs to the node 'u' */
    command_graph_pass_prog_fuse_llvmir(
        cg,
        &u->command->prog,
        &v->command->prog,
        &u->command->prog
    );

    /* remove node 'v' */
    cg->contract<COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE | COMMAND_GRAPH_CONTRACTION_HINT_INPLACE>(u, v);

    /* mark 'v' as contracted to skip it from future lookups */
    nodes[v->iterator_index].data.contracted = true;
}

void
command_graph_t::pass_prog_fuse(void)
{
    /* Iterate through all nodes, and contract until we tried all nodes.
     * New node can be safely pushed-back: they will be iterated on. */
    constexpr bool include_entry_exit = false;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();

    /* iterate through each original nodes */
    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        node_t & node = nodes[i];
        command_graph_node_t * u = node.node;
        assert(u);

        /* if the node was already contracted, ignore it */
        if (node.data.contracted)
            continue ;
        assert(!node.data.contracted);

        /* Detect u->v sequence */
        if (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && u->command->type == COMMAND_TYPE_PROG)
        {
retry_node:
            for (command_graph_node_t * v : u->successors)
            {
                assert(u != v);
                if (v->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && v->command->type == COMMAND_TYPE_PROG)
                {
                    if (this->are_sequence(u, v))
                    {
                        command_graph_pass_prog_fuse(this, u, v, nodes);
                        goto retry_node;
                    }
                }
            }
        }

        // TODO: it this needed ?
        # if 0
        /* 2. detect v->u sequence */
        for (command_graph_node_t * v : u->predecessors)
        {
            assert(u != v);
            if (command_graph_pass_batch_can_batch(u, v))
            {
                if (this->are_sequence(v, u))
                {
                    command_graph_pass_batch_contract<COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE>(this, u, v, nodes);
                    goto retry_node;
                }
            }
        }
        # endif
    }
}
