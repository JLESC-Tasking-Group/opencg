//===- CGTypes.td - CGIR command-graph dialect types -----*- tablegen -*-===//
#ifndef CGIR_CG_TYPES_TD
#define CGIR_CG_TYPES_TD

include "CGDialect.td"

//===----------------------------------------------------------------------===//
// cg.token
//===----------------------------------------------------------------------===//

def CG_TokenType : CG_Type<"Token", "token"> {
  let summary = "A precedence edge / completion token.";
  let description = [{
    An SSA value of type `!cg.token` models a precedence constraint (an edge) in
    the command graph. A node op consumes the tokens of all its predecessors and
    produces a single token, consumed by all its successors.
  }];
}

#endif // CGIR_CG_TYPES_TD
