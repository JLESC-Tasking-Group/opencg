/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** CGIR MLIR optimization entry points.
**
**  - load_dialect / create_pass : small pieces of the public cgir::cg API.
**  - command_graph_optimize_mlir : the (MLIR-free signature) hook called by the
**    out-of-line dispatcher in src/command-graph.cc when the MLIR optimizer is
**    selected. It imports the POD command graph into the `cg` dialect, runs the
**    requested pass, then exports the result back into the POD graph.
**
** All passes are available as MLIR passes. As a safety net, if a factory ever
** returns nullptr (pass not implemented in MLIR), the dispatcher transparently
** falls back to the legacy POD implementation.
**
** This software is governed by the CeCILL-C license. See the LICENSE file.
**/

# include <cgir/mlir/cgir-mlir.hpp>

# include <mlir/IR/MLIRContext.h>
# include <mlir/IR/Verifier.h>
# include <mlir/Pass/Pass.h>
# include <mlir/Pass/PassManager.h>

# include <cgir/command-graph.hpp>
# include <cgir/command-graph-pass.hpp>

# include <cstdio>
# include <cstdlib>
# include <memory>

CGIR_NAMESPACE_BEGIN
namespace cg {

void
load_dialect(mlir::MLIRContext & ctx)
{
    ctx.getOrLoadDialect<CGDialect>();
}

/* The `jit` pass is not an MLIR graph transformation (it only installs function
 * pointers on PROG commands); returning nullptr makes the dispatcher fall back
 * to the legacy POD pass_jit, which operates directly on the POD graph. */
std::unique_ptr<mlir::Pass>
create_jit_pass(void)
{
    return nullptr;
}

/* The `sequence` pass is only implemented as a legacy POD pass; returning
 * nullptr makes the dispatcher fall back to command_graph_t::pass_sequence. */
std::unique_ptr<mlir::Pass>
create_sequence_pass(void)
{
    return nullptr;
}

std::unique_ptr<mlir::Pass>
create_pass(command_graph_pass_t pass)
{
    switch (pass)
    {
        # define DEF(ENUM, FUNC, NAME, MK) case (ENUM): return MK();
        CGIR_FORALL_COMMAND_GRAPH_PASS(DEF);
        # undef DEF
        default:
            return nullptr;
    }
}

} /* namespace cg */
CGIR_NAMESPACE_END

CGIR_NAMESPACE_BEGIN

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
        fprintf(stderr, "cgir/mlir: no cg.graph produced by import\n");
        abort();
    }

    if (mlir::failed(mlir::verify(graph.getOperation())))
    {
        fprintf(stderr, "cgir/mlir: imported cg.graph failed verification\n");
        abort();
    }

    /* run the pass */
    mlir::PassManager pm(&context, cg::GraphOp::getOperationName());
    pm.addPass(std::move(p));
    if (mlir::failed(pm.run(graph.getOperation())))
    {
        fprintf(stderr, "cgir/mlir: pass '%s' failed\n", command_graph_pass_to_str(pass));
        abort();
    }

    /* export MLIR -> POD */
    cg::export_command_graph(graph, cgraph);
}

CGIR_NAMESPACE_END
