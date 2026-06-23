/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** OpenCG MLIR optimization entry points.
**
**  - load_dialect / create_pass : small pieces of the public ocg::cg API.
**  - command_graph_optimize_mlir : the (MLIR-free signature) hook called by the
**    out-of-line dispatcher in src/command-graph.cc when the MLIR optimizer is
**    selected. It imports the POD command graph into the `cg` dialect, runs the
**    requested pass, then exports the result back into the POD graph.
**
** Passes not yet ported to MLIR (e.g. prog-fuse) have a factory returning
** nullptr and transparently fall back to the legacy POD implementations.
**
** This software is governed by the CeCILL-C license. See the LICENSE file.
**/

# include <opencg/mlir/opencg-mlir.hpp>

# include <mlir/IR/MLIRContext.h>
# include <mlir/IR/Verifier.h>
# include <mlir/Pass/PassManager.h>

# include <opencg/command-graph.hpp>
# include <opencg/command-graph-pass.hpp>

# include <cstdio>
# include <cstdlib>
# include <memory>

OCG_NAMESPACE_BEGIN
namespace cg {

void
load_dialect(mlir::MLIRContext & ctx)
{
    ctx.getOrLoadDialect<CGDialect>();
}

/* prog-fuse is intentionally not ported to MLIR (it relies on LLVM IR linking +
 * JIT). Returning nullptr makes the dispatcher fall back to the legacy POD
 * implementation. */
std::unique_ptr<mlir::Pass>
create_prog_fuse_pass(void)
{
    return nullptr;
}

std::unique_ptr<mlir::Pass>
create_pass(command_graph_pass_t pass)
{
    switch (pass)
    {
        # define DEF(ENUM, FUNC, NAME, MK) case (ENUM): return MK();
        OCG_FORALL_COMMAND_GRAPH_PASS(DEF);
        # undef DEF
        default:
            return nullptr;
    }
}

} /* namespace cg */
OCG_NAMESPACE_END

OCG_NAMESPACE_BEGIN

void
command_graph_optimize_mlir(
    command_graph_t * cgraph,
    command_graph_pass_t pass
) {
    /* select the MLIR pass; passes not ported to MLIR fall back to legacy */
    std::unique_ptr<mlir::Pass> p = cg::create_pass(pass);
    if (p == nullptr)
    {
        cgraph->optimize_legacy(pass);
        return ;
    }

    /* set up the context and load the cg dialect */
    mlir::MLIRContext context;
    cg::load_dialect(context);

    /* import POD -> MLIR */
    mlir::OwningOpRef<mlir::ModuleOp> module = cg::import_command_graph(context, cgraph);
    cg::GraphOp graph = cg::get_graph(module.get());
    if (!graph)
    {
        fprintf(stderr, "opencg/mlir: no cg.graph produced by import\n");
        abort();
    }

    if (mlir::failed(mlir::verify(graph.getOperation())))
    {
        fprintf(stderr, "opencg/mlir: imported cg.graph failed verification\n");
        abort();
    }

    /* run the pass */
    mlir::PassManager pm(&context, cg::GraphOp::getOperationName());
    pm.addPass(std::move(p));
    if (mlir::failed(pm.run(graph.getOperation())))
    {
        fprintf(stderr, "opencg/mlir: pass '%s' failed\n", command_graph_pass_to_str(pass));
        abort();
    }

    /* export MLIR -> POD */
    cg::export_command_graph(graph, cgraph);
}

OCG_NAMESPACE_END
