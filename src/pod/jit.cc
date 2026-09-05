/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

// Legacy (POD) jit pass: JIT-compiles every PROG node carrying a non-null
// LLVM-IR source (e.g. a program produced by prog-fuse) and installs the
// resulting function pointer, via the shared core command_graph_jit_llvmir()
// (see ../prog-fuse-llvmir.cc).

# include <cgir/namespace.hpp>
# include <cgir/command.hpp>
# include <cgir/command-graph.hpp>

# include "prog-fuse-llvmir.hpp"

# include <stdlib.h>
# include <string.h>

CGIR_NAMESPACE_USE;

/* true iff `u` is a command node holding a non-null LLVM-IR program source */
static inline bool
node_has_llvmir_source(const command_graph_node_t * u)
{
    return u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND
        && u->command != nullptr
        && u->command->type == COMMAND_TYPE_PROG
        && u->command->prog.source.type == COMMAND_PROG_SOURCE_TYPE_LLVMIR
        && u->command->prog.source.content.llvmir.raw != nullptr;
}

/* `CGIR_JIT_AOT_DEVICE=0` makes the pass leave alone any device program that
 * still holds its ahead-of-time compiled kernel, i.e. recompile only what
 * prog-fuse synthesized. On by default: recompiling is safe and pays (measured
 * on Krylov CG / GH200: 1.95ms ahead-of-time vs 1.82ms recompiled).
 *
 * It was NOT safe before the runtime learned to preserve occupancy across a code
 * substitution, and the failure is worth remembering, because nothing about it
 * is visible in the generated code. On a CSR SpMV `omp target teams distribute
 * parallel for`, the recompiled inner loop is instruction-for-instruction
 * identical to the ahead-of-time one; but ahead-of-time the kernel keeps a
 * runtime `IsSPMDMode` dispatch and a dead duplicate of the loop nest for
 * generic mode, and with it 9 named barriers -- and the barrier count was what
 * limited the kernel to 7 blocks/SM. This pass runs OpenMPOpt, which proves the
 * kernel SPMD-only and deletes that path, leaving 1 barrier: the device then
 * co-schedules 16 blocks/SM, they compete for the same cache, and DRAM traffic
 * goes from 168MB to 625MB for a 6x slower kernel. Better code, three times
 * slower. See command_prog_t::blocks_per_sm and the driver-side enforcement
 * (xkrt's cu_prog_prepare) for how that is now held in place -- a driver without
 * that support should set this to 0. */
static inline bool
jit_recompiles_aot_device(void)
{
    static const bool value = [] {
        const char * e = getenv("CGIR_JIT_AOT_DEVICE");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return value;
}

/* true iff the pass must leave `prog` alone: a device program that already holds
 * the kernel handle of its ahead-of-time compiled image, with recompilation
 * disabled (see jit_recompiles_aot_device).
 *
 * Programs synthesized by prog-fuse have no ahead-of-time counterpart -- the
 * pass clears `launcher.variadic.fn` precisely to mean "must be JIT'd" -- so
 * they are never skipped. Host programs are never skipped either: for them the
 * JIT is a real specialization (externalized globals bound to their runtime
 * addresses, task body inlined into a direct call, KMP dispatch dropped). */
static inline bool
jit_skips_aot_device(const command_prog_t * prog)
{
    return prog->source.content.llvmir.triple != nullptr   /* device program */
        && prog->launcher.variadic.fn != nullptr           /* AOT kernel available */
        && !jit_recompiles_aot_device();
}

void
command_graph_t::pass_jit(void)
{
    this->walk([&] (command_graph_node_t * node)
    {
        if (node_has_llvmir_source(node) && !jit_skips_aot_device(&node->command->prog))
            command_graph_jit_llvmir(&node->command->prog);
    });
}
