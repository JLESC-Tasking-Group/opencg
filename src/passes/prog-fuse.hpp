/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** Shared core of the prog-fuse optimization: fuses two LLVM-IR programs into
** one. Used by both the legacy POD pass (src/passes/prog-fuse.cc) and the MLIR
** pass (src/mlir/passes/ProgFuse.cpp). Its signature exposes no LLVM type, so
** callers (e.g. the MLIR pass) need not include LLVM headers; the symbol is
** resolved within libopencg, where the LLVM machinery lives.
**
** This software is governed by the CeCILL-C license. See the LICENSE file.
**/

#ifndef __OPENCG_PROG_FUSE_HPP__
# define __OPENCG_PROG_FUSE_HPP__

# include <opencg/command.hpp>
# include <opencg/namespace.hpp>

OCG_NAMESPACE_BEGIN

/**
 *  Fuse two LLVM-IR PROG commands into one.
 *      - pu  : the first program (executed first)
 *      - pv  : the second program (executed after pu)
 *      - puv : the destination program (may alias pu for in-place fusion)
 *
 *  Links the two LLVM IR modules, builds a sequential wrapper, JIT-compiles it
 *  in-process and sets puv's variadic launcher to the fused function. Both pu
 *  and pv must have a COMMAND_PROG_SOURCE_TYPE_LLVMIR source.
 *
 *  Requires OPENCG_SUPPORT_LLVM; aborts otherwise.
 */
void
command_graph_prog_fuse_llvmir(
    command_prog_t * pu,
    command_prog_t * pv,
    command_prog_t * puv
);

OCG_NAMESPACE_END

#endif /* __OPENCG_PROG_FUSE_HPP__ */
