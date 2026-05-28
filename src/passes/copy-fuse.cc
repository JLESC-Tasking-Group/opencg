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

# include <opencg/namespace.hpp>
# include <opencg/command.hpp>
# include <opencg/command-graph.hpp>
# include <opencg/min-max.h>

# include <queue>
# include <stack>

OCG_NAMESPACE_USE;

/* pass local storage */
struct pls_t
{
    bool contracted;

    pls_t(void) : contracted(false) {}
    ~pls_t(void) {}
};

using node_t = command_graph_t::node_iterator_t<pls_t>;

static inline bool
command_graph_pass_batch_try_fuse_copy(
    command_graph_t * cg,
    command_graph_node_t * u,
    command_graph_node_t * v
) {
    # define MUST(T) if (!(T)) return false;
    MUST(u != v);
    MUST(cg->are_false_twins(u, v));
    MUST(u->command);
    MUST(v->command);

    /* Try merge */
    switch (u->command->type)
    {
        case (COMMAND_TYPE_COPY_H2H_1D):
        case (COMMAND_TYPE_COPY_H2D_1D):
        case (COMMAND_TYPE_COPY_D2H_1D):
        case (COMMAND_TYPE_COPY_D2D_1D):
        {
            switch (v->command->type)
            {
                case (COMMAND_TYPE_COPY_H2H_1D):
                case (COMMAND_TYPE_COPY_H2D_1D):
                case (COMMAND_TYPE_COPY_D2H_1D):
                case (COMMAND_TYPE_COPY_D2D_1D):
                {
                    // must have same src/dst device
                    MUST(u->command->copy_1D.src_device_unique_id == v->command->copy_1D.src_device_unique_id &&
                         u->command->copy_1D.dst_device_unique_id == v->command->copy_1D.dst_device_unique_id);

                    //  u's src
                    //  [.........................]
                    //                 v's src
                    //                 [.........................]
                    //
                    //          u's dst
                    //          [.........................]
                    //                         v's dst
                    //                         [.........................]
                    //

                    // memory mapping src/dst must be contiguous
                    MUST(u->command->copy_1D.src_device_addr - u->command->copy_1D.dst_device_addr == v->command->copy_1D.src_device_addr - v->command->copy_1D.dst_device_addr);

                    // compute end of src segments
                    const uintptr_t u_src_bgn = u->command->copy_1D.src_device_addr;
                    const uintptr_t v_src_bgn = v->command->copy_1D.src_device_addr;
                    const uintptr_t u_src_end = u->command->copy_1D.src_device_addr + u->command->copy_1D.size;
                    const uintptr_t v_src_end = v->command->copy_1D.src_device_addr + v->command->copy_1D.size;

                    // Test if u and v segments overlap or are adjacents
                    if (u_src_bgn <= v_src_end && v_src_bgn <= u_src_end)
                    {
                        const uintptr_t src_bgn = MIN(u_src_bgn, v_src_bgn);
                        const uintptr_t src_end = MAX(u_src_end, v_src_end);
                        const uintptr_t src_dst_bias = u->command->copy_1D.src_device_addr - u->command->copy_1D.dst_device_addr;

                        cg->contract<COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS | COMMAND_GRAPH_CONTRACTION_HINT_INPLACE>(u, v);
                        u->command->copy_1D.src_device_addr = src_bgn;
                        u->command->copy_1D.dst_device_addr = src_bgn - src_dst_bias;
                        u->command->copy_1D.size            = src_end - src_bgn;

                        return true;
                    }

                    return false;
                }

                case (COMMAND_TYPE_COPY_H2H_2D):
                case (COMMAND_TYPE_COPY_H2D_2D):
                case (COMMAND_TYPE_COPY_D2H_2D):
                case (COMMAND_TYPE_COPY_D2D_2D):
                {
                    // LOGGER_FATAL("TODO: merge 1d and 2d");
                    abort();
                    return false;
                }

                default:
                    return false;

            }   /* switch v->command->type */

            return false;
        }

        // 2D copies (col major)
        case (COMMAND_TYPE_COPY_H2H_2D):
        case (COMMAND_TYPE_COPY_H2D_2D):
        case (COMMAND_TYPE_COPY_D2H_2D):
        case (COMMAND_TYPE_COPY_D2D_2D):
        {
            switch (v->command->type)
            {
                case (COMMAND_TYPE_COPY_H2H_1D):
                case (COMMAND_TYPE_COPY_H2D_1D):
                case (COMMAND_TYPE_COPY_D2H_1D):
                case (COMMAND_TYPE_COPY_D2D_1D):
                {
                    // LOGGER_FATAL("TODO: merge 2d and 1d");
                    abort();
                    return false;
                }

                case (COMMAND_TYPE_COPY_H2H_2D):
                case (COMMAND_TYPE_COPY_H2D_2D):
                case (COMMAND_TYPE_COPY_D2H_2D):
                case (COMMAND_TYPE_COPY_D2D_2D):
                {
                    // must be on the same device
                    MUST(u->command->copy_2D.src_device_unique_id == v->command->copy_2D.src_device_unique_id &&
                         u->command->copy_2D.dst_device_unique_id == v->command->copy_2D.dst_device_unique_id);

                    // byte-unit quantities
                    const size_t u_src_stride = u->command->copy_2D.sizeof_type * u->command->copy_2D.src_ld;
                    const size_t u_dst_stride = u->command->copy_2D.sizeof_type * u->command->copy_2D.dst_ld;
                    const size_t u_col_bytes  = u->command->copy_2D.sizeof_type * u->command->copy_2D.m;

                    const size_t v_src_stride = v->command->copy_2D.sizeof_type * v->command->copy_2D.src_ld;
                    const size_t v_dst_stride = v->command->copy_2D.sizeof_type * v->command->copy_2D.dst_ld;
                    const size_t v_col_bytes  = v->command->copy_2D.sizeof_type * v->command->copy_2D.m;

                    // Requires identical col length length, and row striding.
                    MUST(u_col_bytes == v_col_bytes && u_src_stride == v_src_stride && u_dst_stride == v_dst_stride);

                    // memory mapping src/dst must be contiguous
                    MUST(u->command->copy_2D.src_addr - u->command->copy_2D.dst_addr == v->command->copy_2D.src_addr - v->command->copy_2D.dst_addr);

                    //
                    //  Fusion possible case:
                    //
                    //      u:  [.     .     .]    .     .     .    [.     .     .]    .     .     .    [.     .     .]    .     .     .
                    //      v:   .     .     .    [.     .]    .     .     .     .    [.     .]    .     .     .     .    [.     .]    .
                    //
                    //      u:  [.     .     .]    .     .     .    [.     .     .]    .     .     .    [.     .     .]    .     .     .
                    //      v:   .    [.     .     .     .]    .     .    [.     .     .     .]    .     .    [.     .     .     .]    .
                    //
                    //      u:  [.     .     .]    .     .     .    [.     .     .]    .     .     .    [.     .     .]    .     .     .
                    //      v:  [.     .     .     .     .]    .    [.     .     .     .     .]    .    [.     .     .     .     .]    .
                    //
                    //      u:  [.     .     .]    .     .     .    [.     .     .]    .     .     .    [.     .     .]    .     .     .
                    //      v:  [.     .     .     .     .     .     .     .     .     .]    .    .     [.     .     .     .     .     .
                    //
                    //      u:  [.     .     .]    .     .     .    [.     .     .]    .     .     .    [.     .     .]    .     .     .
                    //      v:   .    [.     .     .     .     .     .     .]    .     .     .     .     .    [.     .     .     .     .
                    //
                    //  TODO: what general conditions to match all these?
                    //
                    // LOGGER_FATAL("TODO");
                    abort();
                    # if 0
                    if (...)
                    {
                        return true;
                    }
                    # endif

                    return false;
                }

                default:
                    return false;

            } /* switch v->command->type */

            return false;
        }
        default:
            return false;
    }

    # undef MUST
}

//  OPENCG can generate multiple copies with same source, same destination, on contiguous memory.
//  In such case, these nodes would be false-twins --- in such case, this pass merges them to a single copy.
//  TODO: does other runtimes would have similar merge needs but with different nodes relationships? (e.g., u->v sequence)
//
//  This pass also merge discontiguous 1D copies to a single 2D when possible.
//  TODO: merge multiple 2D to 3D
void
command_graph_t::pass_copy_fuse(void)
{
    /* Iterate through all nodes, and fuse contiguous copies occuring in sibling nodes */
    constexpr bool include_entry_exit = false;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();

    /* iterate through each original nodes */
    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        node_t & node = nodes[i];
        command_graph_node_t * u = node.node;
        assert(u);

        /* if the node was already contracted, ignore it */
        if (node.data.contracted)
        {
            // LOGGER_DEBUG("Skipping %zu: already contracted", u->index);
            continue ;
        }
        assert(!node.data.contracted);

retry_node:

        /* 1. detect false twins */
        for (command_graph_node_t * pred : u->predecessors)
            for (command_graph_node_t * v : pred->successors)
                if (command_graph_pass_batch_try_fuse_copy(this, u, v))
                    goto retry_node;
    }
}
