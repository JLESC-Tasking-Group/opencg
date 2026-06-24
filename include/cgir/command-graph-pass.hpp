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

#ifndef __CGIR_COMMAND_GRAPH_PASS_HPP__
# define __CGIR_COMMAND_GRAPH_PASS_HPP__

# include <cgir/namespace.hpp>

CGIR_NAMESPACE_BEGIN

/* For each command graph optimization pass:
 *   - ENUM : the command_graph_pass_t enumerator
 *   - FUNC : the legacy (POD) member function on command_graph_t
 *   - NAME : a human readable name
 *   - MK   : the MLIR pass factory in namespace cgir::cg (see cgir/mlir).
 *            A not-yet-ported pass uses a factory that returns nullptr, so the
 *            dispatcher transparently falls back to the legacy POD pass.
 * The MK column is only referenced by the (MLIR-only) src/mlir translation
 * units; POD consumers expand this macro ignoring it. */
# define CGIR_FORALL_COMMAND_GRAPH_PASS(F)                                                                   \
    F(COMMAND_GRAPH_PASS_BATCH,             pass_batch,             "batch",            create_batch_pass)          \
    F(COMMAND_GRAPH_PASS_COPY_FUSE,         pass_copy_fuse,         "copy-fuse",        create_copy_fuse_pass)      \
    F(COMMAND_GRAPH_PASS_COPY_NORMALIZE,    pass_copy_normalize,    "copy-normalize",   create_copy_normalize_pass) \
    F(COMMAND_GRAPH_PASS_PROG_FUSE,         pass_prog_fuse,         "prog-fuse",        create_prog_fuse_pass)      \
    F(COMMAND_GRAPH_PASS_REDUCE_NODE,       pass_reduce_node,       "reduce-node",      create_reduce_node_pass)    \
    F(COMMAND_GRAPH_PASS_REDUCE_EDGE,       pass_reduce_edge,       "reduce-edge",      create_reduce_edge_pass)

enum command_graph_pass_t
{
    # define DEF(ENUM, FUNC, NAME, MK) ENUM,
    CGIR_FORALL_COMMAND_GRAPH_PASS(DEF)
    # undef DEF
    COMMAND_GRAPH_PASS_MAX
};

static inline const char *
command_graph_pass_to_str(command_graph_pass_t opt)
{
    switch (opt)
    {
        # define DEF(ENUM, FUNC, NAME, MK) case(ENUM): return NAME;
        CGIR_FORALL_COMMAND_GRAPH_PASS(DEF);
        # undef DEF
        default:
            return "(null)";
    }
}

CGIR_NAMESPACE_END

#endif /* __CGIR_COMMAND_GRAPH_PASS_HPP__ */
