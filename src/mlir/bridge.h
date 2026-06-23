/*
** OpenCG <-> MLIR bridge.
**
** Declares the import (POD command graph -> cg dialect) and export (cg dialect
** -> POD command graph) routines, plus the MLIR pass factories. Internal to
** libopencg.
*/

#ifndef OPENCG_MLIR_BRIDGE_H
#define OPENCG_MLIR_BRIDGE_H

#include "CG/CGDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/DenseMap.h"

#include <opencg/command.hpp>
#include <opencg/command-graph.hpp>

#include <memory>

namespace mlir { class Pass; }

namespace ocg {
namespace cg {

/* Maps each cg op back to the POD command-graph node it was imported from.
 * Ops created by passes (no originating node) are simply absent from the map. */
using NodeMap = ::llvm::DenseMap<::mlir::Operation *, ::ocg::command_graph_node_t *>;

/* Import a POD command graph into a fresh module containing a single cg.graph.
 * Populates `map` with op -> source-node associations. */
::mlir::OwningOpRef<::mlir::ModuleOp>
import_command_graph(
    ::mlir::MLIRContext & ctx,
    ::ocg::command_graph_t * cg,
    NodeMap & map
);

/* Export an (optimized) cg.graph back into the POD command graph `cg`.
 * Reuses original POD nodes/commands for ops still present in `map` (preserving
 * device, payload, flags and any runtime/replay state), and allocates new nodes
 * via the graph's allocator callbacks for ops created by passes. The POD graph's
 * entry/exit pointers and node adjacency are rebuilt to match the MLIR graph. */
void
export_command_graph(
    GraphOp graph,
    ::ocg::command_graph_t * cg,
    NodeMap & map
);

/* Return the single cg.graph op nested directly in `module` (null if none). */
GraphOp get_graph(::mlir::ModuleOp module);

/* Pass factories. */
std::unique_ptr<::mlir::Pass> createReduceNodePass();
std::unique_ptr<::mlir::Pass> createReduceEdgePass();

} // namespace cg
} // namespace ocg

#endif /* OPENCG_MLIR_BRIDGE_H */
