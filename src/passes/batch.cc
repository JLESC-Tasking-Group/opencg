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

# define COMMAND_IS_ATOMIC(CMD) (CMD->type != COMMAND_TYPE_BATCH || CMD->batch.cg == NULL)

/**
 *  Init a batch command
 */
template <command_graph_contraction_hint_t hint>
inline void
command_batch_init(
    command_graph_t * original_cg,
    command_graph_node_t * u,
    command_graph_node_t * v
) {
    assert(original_cg);
    assert(u);
    assert(v);

    /* retrieve original u/v commands */
    command_t * cmd_u = u->command;
    command_t * cmd_v = v->command;
    assert(cmd_u);
    assert(cmd_v);

    /* create a new batch command, and replace u's command (since 'v' just got
     * contracted into 'u', we are building a new 'batch' command to 'u' here */
    assert(original_cg->command_new);
    command_t * cmd = original_cg->command_new(original_cg, COMMAND_TYPE_BATCH);
    cmd->batch.driver_handle = NULL;
    assert(original_cg->command_graph_new);
    cmd->batch.cg = original_cg->command_graph_new(original_cg);
    assert(cmd->batch.cg);
    u->command = cmd;

    command_graph_node_t * entry = cmd->batch.cg->node_get_entry();
    command_graph_node_t * exit  = cmd->batch.cg->node_get_exit();
    assert(entry);
    assert(exit);
    entry->successors.clear();
    exit->predecessors.clear();
    assert(entry->predecessors.size() == 0);
    assert(exit->successors.size() == 0);

    /* create new nodes corresponding to u and v in the new batch command graph */
    assert(cmd->batch.cg->command_graph_node_new);
    command_graph_node_t * uu = cmd->batch.cg->command_graph_node_new(cmd->batch.cg, cmd_u, u->device_unique_id);
    command_graph_node_t * vv = cmd->batch.cg->command_graph_node_new(cmd->batch.cg, cmd_v, v->device_unique_id);
    assert(uu);
    assert(vv);

    if constexpr (hint == COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS)
    {
        entry->precedes(uu);
        entry->precedes(vv);
        uu->precedes(exit);
        vv->precedes(exit);
    }
    else if constexpr (hint == COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE)
    {
        entry->precedes(uu);
        uu->precedes(vv);
        vv->precedes(exit);
    }
    else if constexpr (hint == COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE)
    {
        entry->precedes(uu);
        vv->precedes(uu);
        uu->precedes(exit);
    }
    else
    {
        abort();
    }
}

/**
 *  Add the node 'v' to the graph 'u_graph'.
 *  The node 'v' may be allocated within another command graph
 */
template <command_graph_contraction_hint_t hint>
static inline void
command_graph_pass_batch_contract_batch_single_node(
    command_graph_t * u_graph,
    command_graph_node_t * v
) {
    assert(u_graph);
    assert(v);
    assert(v->command);

    command_graph_node_t * u_entry = u_graph->node_get_entry();
    command_graph_node_t * u_exit  = u_graph->node_get_exit();
    assert(u_entry);
    assert(u_exit);

    /* clear original edges in 'v' */
    v->predecessors.clear();
    v->successors.clear();

    /**
     *      <>
     *    /    \
     *   Gu     v
     *    \    /
     *      <>
     */
    if constexpr (hint & COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS)
    {
        u_entry->precedes(v);
        v->precedes(u_exit);
    }
    /**
     *      < >
     *       |
     *      Gu
     *       |
     *       v
     *       |
     *      < >
     */
    else if constexpr (hint & COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE)
    {
        // connect u's exit predecessors to 'v'
        for (command_graph_node_t * pred : u_exit->predecessors)
        {
            pred->successors.erase(std::find(pred->successors.begin(), pred->successors.end(), u_exit));
            pred->precedes(v);
        }
        u_exit->predecessors.clear();

        // connect 'v' to u's exit
        v->precedes(u_exit);
    }
    /**
     *      < >
     *       |
     *       v
     *       |
     *      Gu
     *       |
     *      < >
     */
    else if constexpr (hint & COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE)
    {
        // connect u's entry successors to 'v'
        for (command_graph_node_t * succ : u_entry->successors)
        {
            succ->predecessors.erase(std::find(succ->predecessors.begin(), succ->predecessors.end(), u_entry));
            v->precedes(succ);
        }
        u_entry->successors.clear();

        // connect 'v' to u's entry
        u_entry->precedes(v);
    }
    else
    {
        // LOGGER_FATAL("Not implemented");
        abort();
    }
}

/**
 *  Merge the graph 'v' to 'u'
 */
template <command_graph_contraction_hint_t hint>
static inline void
command_graph_pass_batch_contract_batch_merge(
    command_graph_node_t * u,
    command_graph_node_t * v
) {
    assert(u);
    assert(v);
    assert(u->device_unique_id == v->device_unique_id);
    assert(u->command);
    assert(v->command);
    assert(u->command->batch.cg);
    assert(v->command->batch.cg);

    # if 0
    command_graph_pass_batch_contract_batch_single_node<hint>(&u->command->batch.cg, v);
    # else
    command_graph_t * u_cg = u->command->batch.cg;
    command_graph_t * v_cg = v->command->batch.cg;

    command_graph_node_t * u_entry = u_cg->node_get_entry();
    command_graph_node_t * u_exit  = u_cg->node_get_exit();
    assert(u_entry);
    assert(u_exit);

    command_graph_node_t * v_entry = v_cg->node_get_entry();
    command_graph_node_t * v_exit  = v_cg->node_get_exit();
    assert(v_entry);
    assert(v_exit);

    if constexpr (hint & COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS)
    {
        // move all nodes from v to u
        for (command_graph_node_t * succ : v_entry->successors)
        {
            assert(succ != v_exit);
            succ->predecessors.erase(std::find(succ->predecessors.begin(), succ->predecessors.end(), v_entry));
            u_entry->precedes(succ);
        }

        for (command_graph_node_t * pred : v_exit->predecessors)
        {
            assert(pred != v_entry);
            pred->successors.erase(std::find(pred->successors.begin(), pred->successors.end(), v_exit));
            pred->precedes(u_exit);
        }

        # if 0 // not needed
        // reconnect only entry->exit in v
        n_entry->successors.clear();
        n_exit->predecessors.clear();
        v_entry->precedes(v_exit);
        # endif
    }
    else
    {
        if constexpr (hint & COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE)
        {
            // TODO: this create u->v sequence of control nodes... maybe merge v_exit and u_exit here instead
            u_cg->node_set_exit(v_exit);
            u_exit->precedes(v_entry);
        }
        else if constexpr (hint & COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE)
        {
            // TODO: this create u->v sequence of control nodes... maybe merge v_exit and u_entry here instead
            u_cg->node_set_entry(v_entry);
            v_exit->precedes(u_entry);
        }
        else
        {
            // LOGGER_FATAL("Not implemented");
            abort();
        }
    }
    # endif
}

/* Contract u and v, whether being twins or with u->v sequence.
 * One of the two node is removed from the graph, the other contracts both.
 * The contracted one is returned. */
template <command_graph_contraction_hint_t hint>
static inline command_graph_node_t *
command_graph_pass_batch_contract(
    command_graph_t * cg,
    command_graph_node_t * u,
    command_graph_node_t * v,
    std::vector<node_t> & nodes
) {
    assert(u->device_unique_id == v->device_unique_id);

    /* update commands */
    if (u->command && v->command)
    {
        if (COMMAND_IS_ATOMIC(u->command) && !COMMAND_IS_ATOMIC(v->command))
        {
swap_u_v:
            if constexpr(hint & COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE)
                return command_graph_pass_batch_contract<COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE>(cg, v, u, nodes);
            else if constexpr(hint & COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE)
                return command_graph_pass_batch_contract<COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE>(cg, v, u, nodes);
            else
            {
                static_assert(hint & COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS);
                std::swap(u, v);
            }
        }
        else
        {
            // nothing to do
        }
    }
    else if (u->command && !v->command)
    {
        // nothing to do
    }
    else if (!u->command && v->command)
    {
        goto swap_u_v;
    }
    else
    {
        assert(!u->command && !v->command);
        // nothing to do
    }

    /* always contract in place */
    cg->contract<hint | COMMAND_GRAPH_CONTRACTION_HINT_INPLACE>(u, v);

    /**
     *  A command is 'atomic' if it is either:
     *      - a non-COMMAND_TYPE_BATCH command
     *      - a non-COMMAND_TYPE_BATCH command with no cg (i.e., cg=NULL)
     *
     *  Once we reached that point, we:
     *      - contracted v to u
     *      - either have:
     *          (a) u and v are controls            -> nothing to do
     *          (b) u is a command, v is a control  -> nothing to do
     *          (c) u and v are commands
     *              (c1) u is non-atomic
     *                  (c1a) v is non-atomic       -> merge two graphs (TODO: shall we fallbacks to c1b instead?)
     *                  (c1b) v is atomic           -> append 'v' to 'u'
     *              (c2) u is atomic, v is atomic   -> convert 'u' to a non-atomic COMMAND_TYPE_BATCH appended with 'v'
     */


    /* mark 'v' as contracted to skip it from future contractions */
    nodes[v->iterator_index].data.contracted = true;

    /* Initialize the command of 'w' */
    if (!u->command)
    {
        // nothing to do - (a)
    }
    else
    {
        if (!v->command)
        {
            // nothing to do - (b)
        }
        else
        {
            // (c)
            assert(u->command);
            assert(v->command);

            if (!COMMAND_IS_ATOMIC(u->command))
            {
                // (c1)
                if (!COMMAND_IS_ATOMIC(v->command))
                {
                    // (c1a)
                    command_graph_pass_batch_contract_batch_merge<hint>(u, v);
                }
                else
                {
                    // (c1b)
                    assert(u->command->batch.cg);
                    command_graph_pass_batch_contract_batch_single_node<hint>(u->command->batch.cg, v);
                }
            }
            else
            {
                // (c2)

                assert(COMMAND_IS_ATOMIC(u->command));
                assert(COMMAND_IS_ATOMIC(v->command));

                command_batch_init<hint>(cg, u, v);
            }
        }
    }

    return u;
}

static inline bool
command_graph_pass_batch_can_batch(
    const command_graph_node_t * u,
    const command_graph_node_t * v
) {
    return u != v && u->device_unique_id == v->device_unique_id;
}

void
command_graph_t::pass_batch(void)
{
    /* Iterate through all nodes, and contract until we tried all nodes.
     * New node can be safely pushed-back: they will be iterated on. */
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
        {
            for (command_graph_node_t * v : pred->successors)
            {
                if (command_graph_pass_batch_can_batch(u, v))
                {
                    if (this->are_false_twins(u, v))
                    {
                        command_graph_pass_batch_contract<COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS>(this, u, v, nodes);
                        goto retry_node;
                    }
                }
            }
        }

        /* 2. detect u->v sequence */
        for (command_graph_node_t * v : u->successors)
        {
            assert(u != v);
            if (command_graph_pass_batch_can_batch(u, v))
            {
                if (this->are_sequence(u, v))
                {
                    command_graph_pass_batch_contract<COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE>(this, u, v, nodes);
                    goto retry_node;
                }
            }
        }

        /* 3. detect v->u sequence */
        for (command_graph_node_t * v : u->predecessors)
        {
            assert(u != v);
            if (command_graph_pass_batch_can_batch(u, v))
            {
                if (this->are_sequence(v, u))
                {
                    command_graph_pass_batch_contract<COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE>(this, u, v, nodes);
                    goto retry_node;
                }
            }
        }
    }
}
