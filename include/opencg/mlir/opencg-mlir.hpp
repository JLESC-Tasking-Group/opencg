/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** Public C++ API of the OpenCG `cg` MLIR dialect.
**
** This header (and the `cg` dialect it exposes) lets a runtime system other
** than XKRT interoperate with OpenCG at the MLIR level: it may build `cg` ops
** directly, run the optimization passes, and/or convert to and from the POD
** command graph. Using this header requires MLIR; the POD command graph headers
** (opencg/command*.hpp) do NOT, so a runtime that only wants the POD path never
** needs MLIR.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software. See the LICENSE file.
**/

#ifndef __OPENCG_MLIR_HPP__
# define __OPENCG_MLIR_HPP__

# include <opencg/mlir/CG/CGDialect.h>

# include <opencg/command.hpp>
# include <opencg/command-graph.hpp>
# include <opencg/command-graph-pass.hpp>

# include <mlir/IR/BuiltinOps.h>
# include <mlir/IR/MLIRContext.h>
# include <mlir/IR/OwningOpRef.h>

# include <memory>

namespace mlir { class Pass; }

OCG_NAMESPACE_BEGIN
namespace cg {

/**
 *  Discardable attribute carrying the originating POD node pointer (as an i64)
 *  on each imported op. It survives op replacement as long as a pass copies it
 *  onto any op it substitutes, letting the exporter reuse the original
 *  node/command (preserving device, payload, command flags and runtime replay
 *  state). Ops created by a pass without this attribute are treated as new.
 */
static constexpr const char * kSrcNodeAttr = "cg.src_node";

/* Load (register) the cg dialect into a context. */
void
load_dialect(::mlir::MLIRContext & ctx);

/* Read/write the originating POD node carried by `cg.src_node`. */
command_graph_node_t *
get_src_node(::mlir::Operation * op);

void
set_src_node(::mlir::Operation * op, command_graph_node_t * node);

/* Return the single cg.graph op nested directly in `module` (null if none). */
GraphOp
get_graph(::mlir::ModuleOp module);

/**
 *  Import a POD command graph into a fresh module containing a single cg.graph.
 *  Each created op carries a `cg.src_node` attribute back to its source node.
 */
::mlir::OwningOpRef<::mlir::ModuleOp>
import_command_graph(::mlir::MLIRContext & ctx, command_graph_t * cg);

/**
 *  Export an (optimized) cg.graph back into the POD command graph `cg`.
 *  Reuses original POD nodes/commands for ops still carrying `cg.src_node`
 *  (preserving device, payload, flags and runtime replay state), and allocates
 *  new nodes via the graph's allocator callbacks otherwise. The POD graph's
 *  entry/exit pointers and node adjacency are rebuilt to match the MLIR graph.
 */
void
export_command_graph(GraphOp graph, command_graph_t * cg);

/* Pass factories. The factory matching a not-yet-ported pass returns nullptr. */
# define DEF(ENUM, FUNC, NAME, MK) std::unique_ptr<::mlir::Pass> MK(void);
OCG_FORALL_COMMAND_GRAPH_PASS(DEF)
# undef DEF

/* Create the MLIR pass implementing `pass`, or nullptr if not ported to MLIR. */
std::unique_ptr<::mlir::Pass>
create_pass(command_graph_pass_t pass);

} /* namespace cg */
OCG_NAMESPACE_END

#endif /* __OPENCG_MLIR_HPP__ */
