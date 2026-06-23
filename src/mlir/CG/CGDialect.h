/*
** OpenCG `cg` MLIR dialect - C++ surface.
**
** Pulls together the TableGen-generated declarations for the dialect, its
** `!cg.token` type, and its ops. Internal to libopencg (never installed).
*/

#ifndef OPENCG_CG_CGDIALECT_H
#define OPENCG_CG_CGDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/RegionKindInterface.h"

// Dialect declaration (CGDialect).
#include "CGDialect.h.inc"

// Type declarations (TokenType).
#define GET_TYPEDEF_CLASSES
#include "CGTypes.h.inc"

// Op declarations (GraphOp, EmptyOp, Copy1DOp, Copy2DOp, GenericOp).
#define GET_OP_CLASSES
#include "CGOps.h.inc"

#endif // OPENCG_CG_CGDIALECT_H
