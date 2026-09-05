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

// Legacy (POD) jit pass: JIT-compiles the PROG nodes carrying a non-null
// LLVM-IR source (e.g. a program produced by prog-fuse) and installs the
// resulting function pointer, via the shared core command_graph_jit_llvmir()
// (see ../prog-fuse-llvmir.cc).
//
// Device programs that still hold their ahead-of-time compiled kernel are
// currently left alone; this is an interim guard, not a statement about JIT
// code quality (see jit_skips_aot_device below).

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

/* `CGIR_JIT_AOT_DEVICE=1` re-enables JIT-compiling device programs that already
 * carry their ahead-of-time compiled kernel (see jit_skips_aot_device()). Off by
 * default; useful to measure JIT-emitted code against the AOT toolchain. */
static inline bool
jit_recompiles_aot_device(void)
{
    static const bool value = [] {
        const char * e = getenv("CGIR_JIT_AOT_DEVICE");
        return e && e[0] && strcmp(e, "0");
    }();
    return value;
}

/* true iff `prog` is a device program that already holds the kernel handle of
 * its ahead-of-time compiled image, in which case the jit pass leaves it alone.
 *
 * INTERIM GUARD -- this is not a claim that JIT-emitted device code is worse.
 * It is not: on the measured case (a CSR SpMV `omp target teams distribute
 * parallel for`) the JIT-emitted inner loop is instruction-for-instruction
 * identical to the ahead-of-time one. What differs is a *launch* property that
 * the JIT changes implicitly. Ahead-of-time, the kernel keeps a runtime
 * `IsSPMDMode` dispatch and a dead duplicate of the loop nest for generic mode,
 * which costs registers (42/thread). The jit pass runs OpenMPOpt, which proves
 * the kernel SPMD-only and deletes that dead path, so the leaner kernel needs
 * fewer registers (32/thread) -- and ptxas sizes registers to an occupancy
 * target, so the device co-schedules more blocks per SM (42% -> 95% achieved
 * occupancy). For a kernel whose gather relies on cache reuse that is a loss:
 * the resident blocks' streaming footprint evicts the reused window (L2 hit
 * 66% -> 43%, DRAM reads 174MB -> 625MB, 6x slower).
 *
 * The proper fix is for the runtime to preserve (and then deliberately choose)
 * the occupancy of a program whose code it replaces, which the device driver
 * does via command_prog_t::blocks_per_sm. Once that is wired for every driver
 * this skip becomes unnecessary and `CGIR_JIT_AOT_DEVICE` should default on.
 *
 * Programs synthesized by prog-fuse have no AOT counterpart, and the pass
 * clears `launcher.variadic.fn` precisely to mean "must be JIT'd", so they are
 * never skipped here (they carry the same occupancy exposure, which is why the
 * driver-side guard is the real fix).
 *
 * Host programs are never skipped: for them the JIT is a real specialization
 * (externalized globals bound to their runtime addresses, task body inlined
 * into a direct call, KMP dispatch dropped). */
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
