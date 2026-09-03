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

# include <cgir/namespace.hpp>
# include <cgir/command.hpp>
# include <cgir/command-graph.hpp>
# include <cgir/min-max.h>

# include <queue>
# include <stack>

CGIR_NAMESPACE_USE;

/* pass local storage */
struct copy_fuse_pls_t
{
    bool contracted;

    copy_fuse_pls_t(void) : contracted(false) {}
    ~copy_fuse_pls_t(void) {}
};

static inline bool
command_graph_pass_try_fuse_copy(
    command_graph_t * cg,
    command_graph_node_t * u,
    command_graph_node_t * v
) {
    assert(u != v);
    assert(u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(v->type == COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(cg->are_false_twins(u, v) || cg->are_serial(u, v));
    assert(u->command);
    assert(v->command);

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
                    if (u->command->copy_1D.src_device_unique_id != v->command->copy_1D.src_device_unique_id ||
                        u->command->copy_1D.dst_device_unique_id != v->command->copy_1D.dst_device_unique_id)
                        return false;

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
                    if (u->command->copy_1D.src_device_addr - u->command->copy_1D.dst_device_addr != v->command->copy_1D.src_device_addr - v->command->copy_1D.dst_device_addr)
                        return false;

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
                    if (u->command->copy_2D.src_device_unique_id != v->command->copy_2D.src_device_unique_id ||
                        u->command->copy_2D.dst_device_unique_id != v->command->copy_2D.dst_device_unique_id)
                        return false;

                    // byte-unit quantities
                    const size_t u_src_stride = u->command->copy_2D.sizeof_type * u->command->copy_2D.src_ld;
                    const size_t u_dst_stride = u->command->copy_2D.sizeof_type * u->command->copy_2D.dst_ld;
                    const size_t u_col_bytes  = u->command->copy_2D.sizeof_type * u->command->copy_2D.m;

                    const size_t v_src_stride = v->command->copy_2D.sizeof_type * v->command->copy_2D.src_ld;
                    const size_t v_dst_stride = v->command->copy_2D.sizeof_type * v->command->copy_2D.dst_ld;
                    const size_t v_col_bytes  = v->command->copy_2D.sizeof_type * v->command->copy_2D.m;

                    // Requires identical col length length, and row striding.
                    if (u_col_bytes != v_col_bytes || u_src_stride != v_src_stride || u_dst_stride != v_dst_stride)
                        return false;

                    // memory mapping src/dst must be contiguous
                    if (u->command->copy_2D.src_addr - u->command->copy_2D.dst_addr != v->command->copy_2D.src_addr - v->command->copy_2D.dst_addr)
                        return false;

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
}

static inline void
command_graph_pass_copy_fuse_normalize(command_t * command)
{
    assert(command);

    /**
     * if it is a 2D copy,
     *  - normalize it to a 1D copy if possible
     *  - set sizeof_type to 1 and scale dimensions
     */

    switch (command->type)
    {
        # define HANDLE_CASE(X)                                                         \
            case (COMMAND_TYPE_COPY_##X##_2D):                                          \
            {                                                                           \
                if (command->copy_2D.m == command->copy_2D.src_ld &&                    \
                    command->copy_2D.m == command->copy_2D.dst_ld)                      \
                {                                                                       \
                    command->type = COMMAND_TYPE_COPY_##X##_1D;                         \
                                                                                        \
                    const size_t src_addr = command->copy_2D.src_addr;                  \
                    const size_t dst_addr = command->copy_2D.dst_addr;                  \
                    const size_t m        = command->copy_2D.m;                         \
                    const size_t n        = command->copy_2D.n;                         \
                    const size_t s        = command->copy_2D.sizeof_type;               \
                                                                                        \
                    command->copy_1D.src_device_addr = src_addr;                        \
                    command->copy_1D.dst_device_addr = dst_addr;                        \
                    command->copy_1D.size            = m * n * s;                       \
                                                                                        \
                    break ;                                                             \
                }                                                                       \
                else                                                                    \
                {                                                                       \
                    /* normalize to byte units: scale the column extent AND both   */   \
                    /* leading dimensions by the element size, otherwise the row   */   \
                    /* strides would lose the 'sizeof_type' factor.                */   \
                    const size_t s = command->copy_2D.sizeof_type;                      \
                    command->copy_2D.sizeof_type = 1;                                   \
                    command->copy_2D.m      = command->copy_2D.m      * s;              \
                    command->copy_2D.src_ld = command->copy_2D.src_ld * s;              \
                    command->copy_2D.dst_ld = command->copy_2D.dst_ld * s;              \
                    break ;                                                             \
                }                                                                       \
            }

        HANDLE_CASE(H2H);
        HANDLE_CASE(H2D);
        HANDLE_CASE(D2H);
        HANDLE_CASE(D2D);

        # undef HANDLE_CASE

        default:
            break ;
    }
}

//  This pass also merge discontiguous 1D copies to a single 2D when possible.
//  TODO: merge multiple 2D to 3D
void
command_graph_t::pass_copy_fuse(void)
{
    /* Iterate through all nodes, and fuse contiguous copies occuring in sibling nodes */
    constexpr bool include_entry_exit = false;
    auto nodes = this->create_node_iterators<include_entry_exit, copy_fuse_pls_t>();

    /* 1. normalize copies */
    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        command_graph_node_t * u = nodes[i].node;
        assert(u);

        /* for each command */
        if (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
        {
            assert(u->command);
            command_graph_pass_copy_fuse_normalize(u->command);
        }
    }

    /* 2. try fusing */
    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        auto & it = nodes[i];
        command_graph_node_t * u = it.node;
        assert(u);

        /* if the node was already contracted, ignore it */
        if (it.data.contracted)
        {
            // LOGGER_DEBUG("Skipping %zu: already contracted", u->index);
            continue ;
        }
        assert(!it.data.contracted);

retry_node:

        for (command_graph_node_t * pred : u->predecessors)
        {
            for (command_graph_node_t * v : pred->successors)
            {
                if (u != v &&
                    u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND &&
                    v->type != COMMAND_GRAPH_NODE_TYPE_COMMAND &&
                    (this->are_false_twins(u, v) || this->are_serial(u, v))
                ) {
                    if (command_graph_pass_try_fuse_copy(this, u, v))
                        goto retry_node;
                }
            }
        }
    }
}
