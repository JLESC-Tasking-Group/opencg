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

#include <opencg/command.hpp>
#include <opencg/command-graph.hpp>

#include <memory>

namespace mlir { class Pass; }

namespace ocg {
namespace cg {

/* Discardable attribute name carrying the originating POD node pointer
 * (as an i64) on each imported op. It survives op replacement as long as a
 * pass copies it onto any op it substitutes, letting the exporter reuse the
 * original node/command (preserving device, payload, command flags and runtime
 * replay state). Ops created by a pass without this attribute are treated as new. */
static constexpr const char * kSrcNodeAttr = "cg.src_node";

/* Read/write the originating POD node carried by `cg.src_node`. */
::ocg::command_graph_node_t * get_src_node(::mlir::Operation * op);
void set_src_node(::mlir::Operation * op, ::ocg::command_graph_node_t * node);

/* Import a POD command graph into a fresh module containing a single cg.graph.
 * Each created op carries a `cg.src_node` attribute back to its source node. */
::mlir::OwningOpRef<::mlir::ModuleOp>
import_command_graph(
    ::mlir::MLIRContext & ctx,
    ::ocg::command_graph_t * cg
);

/* Export an (optimized) cg.graph back into the POD command graph `cg`.
 * Reuses original POD nodes/commands for ops still carrying `cg.src_node`
 * (preserving device, payload, flags and runtime/replay state), and allocates
 * new nodes via the graph's allocator callbacks otherwise. The POD graph's
 * entry/exit pointers and node adjacency are rebuilt to match the MLIR graph. */
void
export_command_graph(
    GraphOp graph,
    ::ocg::command_graph_t * cg
);

/* Return the single cg.graph op nested directly in `module` (null if none). */
GraphOp get_graph(::mlir::ModuleOp module);

/* Pass factories. */
std::unique_ptr<::mlir::Pass> createReduceNodePass();
std::unique_ptr<::mlir::Pass> createReduceEdgePass();
std::unique_ptr<::mlir::Pass> createCopyNormalizePass();
std::unique_ptr<::mlir::Pass> createCopyFusePass();
std::unique_ptr<::mlir::Pass> createBatchPass();

} // namespace cg
} // namespace ocg

#endif /* OPENCG_MLIR_BRIDGE_H */
