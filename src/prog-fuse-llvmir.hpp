/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** Shared core of the prog-fuse optimization: fuses two LLVM-IR programs into
** one. Used by both the legacy POD pass (src/pod/prog-fuse.cc) and the MLIR
** pass (src/mlir/passes/ProgFuse.cpp). Its signature exposes no LLVM type, so
** callers (e.g. the MLIR pass) need not include LLVM headers; the symbol is
** resolved within libcgir, where the LLVM machinery lives.
**
** This software is governed by the CeCILL-C license. See the LICENSE file.
**/

#ifndef __CGIR_PROG_FUSE_LLVMIR_HPP__
# define __CGIR_PROG_FUSE_LLVMIR_HPP__

# include <cgir/command.hpp>
# include <cgir/namespace.hpp>

CGIR_NAMESPACE_BEGIN

/**
 *  Fuse a chain of N >= 2 LLVM-IR PROG commands into one.
 *      - progs : the programs to fuse, in execution order (progs[0] first)
 *      - n     : number of programs (>= 2)
 *      - dst   : the destination program (may alias progs[0] for in-place fusion)
 *
 *  Links all N LLVM IR modules, builds a single sequential `__fused_wrapper`
 *  that calls each program's entry in order, and stores it as dst's LLVM-IR
 *  source together with the deduplicated args buffer. The wrapper is NOT
 *  compiled here: dst->launcher.variadic.fn is left NULL and a subsequent
 *  `jit` pass (command_graph_jit_llvmir) must compile it before execution.
 *  Each program must have a COMMAND_PROG_SOURCE_TYPE_LLVMIR source. Fusion is
 *  composable: a program that is itself the result of a previous fusion is
 *  handled correctly, so arbitrary chains fuse into one wrapper.
 *
 *  RETURNS true iff the chain was fused. Refusing a chain is a normal outcome,
 *  not an error: a chain may be unfusable because a program is not in a form
 *  this routine handles, or -- for device chains -- because fusing it would not
 *  preserve the program's meaning (see device_chain_is_fusible). On false, `dst`
 *  is left EXACTLY as it was and the caller must keep the chain's nodes
 *  separate; a caller that contracts the chain regardless would execute only
 *  progs[0]. Only an out-of-memory condition still aborts.
 *
 *  Requires CGIR_SUPPORT_LLVM; returns false otherwise.
 */
bool
command_graph_prog_fuse_llvmir(
    command_prog_t ** progs,
    size_t n,
    command_prog_t * dst
);

/**
 *  JIT-compile a PROG's LLVM-IR source in-process and install the resulting
 *  function pointer into prog->launcher.variadic.fn, overwriting any previous
 *  value. The entry is `__fused_wrapper` if the module defines it (a fused
 *  program), otherwise the module's first void-returning definition.
 *
 *  prog->source must be a COMMAND_PROG_SOURCE_TYPE_LLVMIR with a non-NULL raw
 *  buffer. Requires CGIR_SUPPORT_LLVM; aborts otherwise.
 */
void
command_graph_jit_llvmir(
    command_prog_t * prog
);

/**
 *  True iff two programs have rigorously identical launch parameters, i.e. the
 *  same grid and block dimensions AND the same launch mode. The prog-fuse passes
 *  only fuse programs whose launch geometry is identical: the fused program is a
 *  single kernel launched over one grid/block, so fusing kernels launched over
 *  different iteration spaces would be unsound. Likewise, the launch mode
 *  (see command_prog_launch_mode_t) must match: a fused program is replayed with
 *  a single launch mode, so e.g. a directly-launched program cannot be fused
 *  with one that must be re-spawned as a task.
 *
 *  TODO: support fusing programs with non-identical grids in the future, e.g. by
 *  launching over the bounding grid and predicating/padding each original
 *  program so out-of-range iterations become no-ops. This would be worthwhile
 *  when the grids differ only slightly (a few percent).
 */
static inline bool
command_prog_launch_params_equal(
    const command_prog_t * a,
    const command_prog_t * b
) {
    /* NOTE: blocks_per_sm is deliberately NOT compared. It is the occupancy each
     * program happened to be recorded with, which follows from its register
     * footprint and so differs from kernel to kernel; requiring equality would
     * refuse almost every fusion. The fused program takes the most restrictive
     * of its inputs instead (see the launch-parameter propagation in
     * command_graph_prog_fuse_llvmir). */
    return a->launch_mode == b->launch_mode
        && a->grid.x  == b->grid.x  && a->grid.y  == b->grid.y  && a->grid.z  == b->grid.z
        && a->block.x == b->block.x && a->block.y == b->block.y && a->block.z == b->block.z;
}

CGIR_NAMESPACE_END

#endif /* __CGIR_PROG_FUSE_LLVMIR_HPP__ */
