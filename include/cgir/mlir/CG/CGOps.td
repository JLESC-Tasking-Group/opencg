//===- CGOps.td - CGIR command-graph dialect ops ---------*- tablegen -*-===//
#ifndef CGIR_CG_OPS_TD
#define CGIR_CG_OPS_TD

include "CGDialect.td"
include "CGTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// cg.graph : the command-graph container
//===----------------------------------------------------------------------===//

def GraphOp : CG_Op<"graph", [
    IsolatedFromAbove,
    NoTerminator,
    SingleBlock,
    DeclareOpInterfaceMethods<RegionKindInterface>
  ]> {
  let summary = "A command graph: a graph region of command nodes.";
  let description = [{
    Holds a single graph region whose ops are the command-graph nodes. Edges are
    encoded as `!cg.token` operands. The region has no SSA dominance so that
    edges may reference nodes appearing later in the block.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let extraClassDeclaration = [{
    ::mlir::Block & getBodyBlock() { return getBody().front(); }
  }];
}

//===----------------------------------------------------------------------===//
// Node ops
//
// Every node op:
//   - consumes a variadic list of `!cg.token` predecessors
//   - produces a single `!cg.token`
//   - carries an `duid` (device unique id) i32 attribute
//===----------------------------------------------------------------------===//

def EmptyOp : CG_Op<"empty"> {
  let summary = "A control (empty) node, used to reduce graph complexity.";
  let arguments = (ins
    Variadic<CG_TokenType>:$predecessors,
    I32Attr:$duid,
    UnitAttr:$is_entry,
    UnitAttr:$is_exit
  );
  let results = (outs CG_TokenType:$token);
}

def Copy1DOp : CG_Op<"copy_1d"> {
  let summary = "A 1-D memory copy command (mirrors command_copy_1D_t).";
  let arguments = (ins
    Variadic<CG_TokenType>:$predecessors,
    I32Attr:$duid,
    I32Attr:$kind,        // cgir::command_type_t (which H2H/H2D/D2H/D2D variant)
    I32Attr:$src_duid,
    I32Attr:$dst_duid,
    I64Attr:$src_addr,
    I64Attr:$dst_addr,
    I64Attr:$size
  );
  let results = (outs CG_TokenType:$token);
}

def Copy2DOp : CG_Op<"copy_2d"> {
  let summary = "A 2-D memory copy command (mirrors command_copy_2D_t).";
  let arguments = (ins
    Variadic<CG_TokenType>:$predecessors,
    I32Attr:$duid,
    I32Attr:$kind,        // cgir::command_type_t (which H2H/H2D/D2H/D2D variant)
    I32Attr:$src_duid,
    I32Attr:$dst_duid,
    I64Attr:$src_addr,
    I64Attr:$src_ld,
    I64Attr:$dst_addr,
    I64Attr:$dst_ld,
    I64Attr:$m,
    I64Attr:$n,
    I64Attr:$sizeof_type
  );
  let results = (outs CG_TokenType:$token);
}

def GenericOp : CG_Op<"generic"> {
  let summary = "An opaque command node carrying raw cgir::command_t bytes.";
  let description = [{
    A carrier for command kinds not yet modeled with typed ops (PROG, FD_*,
    BATCH, CTRL_*). The full `cgir::command_t` payload is stored as a byte blob so
    that import/export round-trips losslessly. Topology passes treat it as an
    opaque command node.
  }];
  let arguments = (ins
    Variadic<CG_TokenType>:$predecessors,
    I32Attr:$duid,
    I32Attr:$kind,        // cgir::command_type_t
    DenseI8ArrayAttr:$blob
  );
  let results = (outs CG_TokenType:$token);
}

//===----------------------------------------------------------------------===//
// cg.batch : a contracted monochromatic sub-graph
//
// Produced by the batch pass. The body region holds the batched member node ops
// connected by their internal !cg.token edges; member ops only reference values
// defined within the region (IsolatedFromAbove). External precedence enters via
// the op's $predecessors operands and leaves via its $token result. On export
// this becomes a COMMAND_TYPE_BATCH command whose batch.cg is a sub command
// graph (consumed by drivers, e.g. mapped to a CUgraph).
//===----------------------------------------------------------------------===//

def BatchOp : CG_Op<"batch", [
    IsolatedFromAbove,
    NoTerminator,
    SingleBlock,
    DeclareOpInterfaceMethods<RegionKindInterface>
  ]> {
  let summary = "A batched sub-graph of same-device commands.";
  let arguments = (ins
    Variadic<CG_TokenType>:$predecessors,
    I32Attr:$duid
  );
  let results = (outs CG_TokenType:$token);
  let regions = (region SizedRegion<1>:$body);
  let extraClassDeclaration = [{
    ::mlir::Block & getBodyBlock() { return getBody().front(); }
  }];
}

#endif // CGIR_CG_OPS_TD
