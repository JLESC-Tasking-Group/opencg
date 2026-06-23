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
 *  Fuse a chain of N >= 2 LLVM-IR PROG commands into one.
 *      - progs : the programs to fuse, in execution order (progs[0] first)
 *      - n     : number of programs (>= 2)
 *      - dst   : the destination program (may alias progs[0] for in-place fusion)
 *
 *  Links all N LLVM IR modules, builds a single sequential wrapper that calls
 *  each program's entry in order, JIT-compiles it in-process and sets dst's
 *  variadic launcher to the fused function. Each program must have a
 *  COMMAND_PROG_SOURCE_TYPE_LLVMIR source. Fusion is composable: a program that
 *  is itself the result of a previous fusion is handled correctly, so arbitrary
 *  chains fuse into one wrapper.
 *
 *  Requires OPENCG_SUPPORT_LLVM; aborts otherwise.
 */
void
command_graph_prog_fuse_llvmir(
    command_prog_t ** progs,
    size_t n,
    command_prog_t * dst
);

OCG_NAMESPACE_END

#endif /* __OPENCG_PROG_FUSE_HPP__ */
