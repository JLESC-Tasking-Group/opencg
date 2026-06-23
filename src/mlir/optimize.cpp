/*
** OpenCG MLIR optimization entry point.
**
** command_graph_optimize_mlir() is the (MLIR-free signature) hook called by the
** out-of-line dispatcher in src/command-graph.cc when the MLIR optimizer is
** selected. It imports the POD command graph into the `cg` dialect, runs the
** requested pass, then exports the result back into the POD graph.
**
** Passes not yet ported to MLIR transparently fall back to the legacy C++
** implementations.
*/

#include "bridge.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include <opencg/command-graph.hpp>
#include <opencg/command-graph-pass.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>

using namespace mlir;
using namespace ocg::cg;

namespace ocg {

void
command_graph_optimize_mlir(
    command_graph_t * cg,
    command_graph_pass_t pass
) {
    /* Select an MLIR pass for the requested optimization. Passes not yet ported
     * to MLIR fall back to the legacy C++ implementation (no MLIR round trip). */
    std::unique_ptr<Pass> p;
    switch (pass)
    {
        case COMMAND_GRAPH_PASS_REDUCE_NODE: p = createReduceNodePass(); break;
        case COMMAND_GRAPH_PASS_REDUCE_EDGE: p = createReduceEdgePass(); break;
        default:
            cg->optimize_legacy(pass);
            return;
    }

    /* set up the context and load the cg dialect */
    MLIRContext context;
    context.getOrLoadDialect<CGDialect>();

    /* import POD -> MLIR */
    NodeMap map;
    OwningOpRef<ModuleOp> module = import_command_graph(context, cg, map);
    GraphOp graph = get_graph(module.get());
    if (!graph)
    {
        fprintf(stderr, "opencg/mlir: no cg.graph produced by import\n");
        abort();
    }

    if (failed(verify(graph.getOperation())))
    {
        fprintf(stderr, "opencg/mlir: imported cg.graph failed verification\n");
        abort();
    }

    /* run the pass */
    PassManager pm(&context, GraphOp::getOperationName());
    pm.addPass(std::move(p));
    if (failed(pm.run(graph.getOperation())))
    {
        fprintf(stderr, "opencg/mlir: pass '%s' failed\n", command_graph_pass_to_str(pass));
        abort();
    }

    /* export MLIR -> POD */
    export_command_graph(graph, cg, map);
}

} // namespace ocg
