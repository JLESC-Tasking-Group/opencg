//===- CGDialect.td - CGIR command-graph dialect ---------*- tablegen -*-===//
//
// Defines the `cg` MLIR dialect: the canonical optimization-time IR for CGIR
// command graphs. This dialect is an *implementation detail* of libcgir and
// is never exposed to CGIR consumers (e.g. XKRT).
//
//===----------------------------------------------------------------------===//

#ifndef CGIR_CG_DIALECT_TD
#define CGIR_CG_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def CG_Dialect : Dialect {
  let name = "cg";
  let summary = "CGIR command-graph dialect.";
  let description = [{
    The `cg` dialect models an CGIR command graph as a region of command
    nodes. Precedence constraints (edges) are modeled as SSA values of type
    `!cg.token`: each node op consumes the tokens of its predecessors and
    produces a single token consumed by its successors. The enclosing
    `cg.graph` region is a graph region (no SSA dominance), which lets edges
    reference any node regardless of textual order.
  }];
  let cppNamespace = "::cgir::cg";

  // Enable default parser/printer for the dialect's types (!cg.token).
  let useDefaultTypePrinterParser = 1;

  // NB: ODS attributes are stored as op *properties* (the framework default
  // since LLVM 18). The import/export bridge still builds ops from a
  // hand-assembled OperationState via addAttribute(...): inherent attributes are
  // automatically migrated from the OperationState dictionary into properties on
  // Operation::create, and read back transparently via getAttrOfType<...>() and
  // the generated ODS getters.
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class CG_Op<string mnemonic, list<Trait> traits = []>
    : Op<CG_Dialect, mnemonic, traits>;

class CG_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<CG_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // CGIR_CG_DIALECT_TD
