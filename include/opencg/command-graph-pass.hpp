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

#ifndef __OPENCG_COMMAND_GRAPH_PASS_HPP__
# define __OPENCG_COMMAND_GRAPH_PASS_HPP__

# include <opencg/namespace.hpp>

OCG_NAMESPACE_BEGIN

# define OCG_FORALL_COMMAND_GRAPH_PASS(F)                                               \
    F(COMMAND_GRAPH_PASS_BATCH,             pass_batch,             "batch")            \
    F(COMMAND_GRAPH_PASS_COPY_FUSE,         pass_copy_fuse,         "copy-fuse")        \
    F(COMMAND_GRAPH_PASS_COPY_NORMALIZE,    pass_copy_normalize,    "copy-normalize")   \
    F(COMMAND_GRAPH_PASS_PROG_FUSE,         pass_prog_fuse,         "prog-fuse")        \
    F(COMMAND_GRAPH_PASS_REDUCE_NODE,       pass_reduce_node,       "reduce-node")      \
    F(COMMAND_GRAPH_PASS_REDUCE_EDGE,       pass_reduce_edge,       "reduce-edge")

enum command_graph_pass_t
{
    # define DEF(ENUM, FUNC, NAME) ENUM,
    OCG_FORALL_COMMAND_GRAPH_PASS(DEF)
    # undef DEF
    COMMAND_GRAPH_PASS_MAX
};

static inline const char *
command_graph_pass_to_str(command_graph_pass_t opt)
{
    switch (opt)
    {
        # define DEF(ENUM, FUNC, NAME) case(ENUM): return NAME;
        OCG_FORALL_COMMAND_GRAPH_PASS(DEF);
        # undef DEF
        default:
            return "(null)";
    }
}

OCG_NAMESPACE_END

#endif /* __OPENCG_COMMAND_GRAPH_PASS_HPP__ */
