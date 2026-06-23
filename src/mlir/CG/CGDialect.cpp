/*
** OpenCG `cg` MLIR dialect - definitions.
*/

#include "CG/CGDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ocg::cg;

//===----------------------------------------------------------------------===//
// Types
//
// Included first: with useDefaultTypePrinterParser, the generated dialect
// parseType/printType (in CGDialect.cpp.inc below) reference the type
// parser/printer emitted here.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "CGTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "CGDialect.cpp.inc"

void CGDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "CGTypes.cpp.inc"
    >();

    addOperations<
#define GET_OP_LIST
#include "CGOps.cpp.inc"
    >();
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "CGOps.cpp.inc"

//===----------------------------------------------------------------------===//
// RegionKindInterface : cg.graph holds a *graph* region (no SSA dominance).
//===----------------------------------------------------------------------===//

RegionKind GraphOp::getRegionKind(unsigned /*index*/)
{
    return RegionKind::Graph;
}
