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

void
command_graph_t::pass_jit(void)
{
    this->walk([&] (command_graph_node_t * node)
    {
        if (node_has_llvmir_source(node))
            command_graph_jit_llvmir(&node->command->prog);
    });
}
