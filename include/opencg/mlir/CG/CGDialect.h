/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** OpenCG `cg` MLIR dialect - C++ surface.
**
** Pulls together the TableGen-generated declarations for the dialect, its
** `!cg.token` type, and its ops. This is a public header: using it requires
** MLIR (the POD command graph headers do not).
**
** This software is governed by the CeCILL-C license. See the LICENSE file.
**/

#ifndef __OPENCG_CG_CGDIALECT_H__
# define __OPENCG_CG_CGDIALECT_H__

# include <mlir/IR/Builders.h>
# include <mlir/IR/BuiltinTypes.h>
# include <mlir/IR/Dialect.h>
# include <mlir/IR/OpDefinition.h>
# include <mlir/IR/OpImplementation.h>
# include <mlir/Interfaces/SideEffectInterfaces.h>
# include <mlir/IR/RegionKindInterface.h>

/* Dialect declaration (CGDialect). */
# include "CGDialect.h.inc"

/* Type declarations (TokenType). */
# define GET_TYPEDEF_CLASSES
# include "CGTypes.h.inc"

/* Op declarations (GraphOp, EmptyOp, Copy1DOp, Copy2DOp, GenericOp, BatchOp). */
# define GET_OP_CLASSES
# include "CGOps.h.inc"

#endif /* __OPENCG_CG_CGDIALECT_H__ */
