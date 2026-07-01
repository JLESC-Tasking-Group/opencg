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

# include <queue>
# include <stack>

CGIR_NAMESPACE_USE;

/* pass local storage */
struct pls_t
{
    bool contracted;

    pls_t(void) : contracted(false) {}
    ~pls_t(void) {}
};

using node_t = command_graph_t::node_iterator_t<pls_t>;

# define NODE_IS_ATOMIC(N) (N->type == COMMAND_GRAPH_NODE_TYPE_EMPTY || (N->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && N->command && (N->command->type != COMMAND_TYPE_BATCH || N->command->batch.cg == NULL)))

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
    assert(NODE_IS_ATOMIC(u));
    assert(NODE_IS_ATOMIC(v));

    /* retrieve original u/v commands */
    command_t * cmd_u = u->command; // can be null if COMMAND_GRAPH_NODE_TYPE_EMPTY
    command_t * cmd_v = v->command; // can be null if COMMAND_GRAPH_NODE_TYPE_EMPTY

    /* create a new batch command */
    assert(original_cg->command_new);
    command_t * cmd = original_cg->command_new(original_cg, COMMAND_TYPE_BATCH);
    assert(cmd);
    cmd->batch.driver_handle = NULL;

    /* replace u's command (since 'v' just got contracted into 'u'),
     * we are building a new 'batch' command to the node 'u' here */
    assert(original_cg->command_graph_new);
    cmd->batch.cg = original_cg->command_graph_new(original_cg);
    assert(cmd->batch.cg);
    u->type = COMMAND_GRAPH_NODE_TYPE_COMMAND;
    u->command = cmd;

    /* remove entry->exit edge in the new cg */
    command_graph_node_t * entry = cmd->batch.cg->node_get_entry();
    command_graph_node_t * exit  = cmd->batch.cg->node_get_exit();
    assert(entry);
    assert(exit);
    entry->successors.clear();
    exit->predecessors.clear();
    assert(entry->predecessors.size() == 0);
    assert(exit->successors.size() == 0);

    /* create new nodes corresponding to u and v in the new batch cg */
    assert(cmd->batch.cg->command_graph_node_new);
    command_graph_node_t * uu = cmd->batch.cg->command_graph_node_new(cmd->batch.cg, u->device_unique_id, u->type);
    command_graph_node_t * vv = cmd->batch.cg->command_graph_node_new(cmd->batch.cg, v->device_unique_id, v->type);
    assert(uu);
    assert(vv);
    uu->command = cmd_u;
    vv->command = cmd_v;

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
    assert(NODE_IS_ATOMIC(v));

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
    assert(!NODE_IS_ATOMIC(u));
    assert(!NODE_IS_ATOMIC(v));

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
    assert(u->type != COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH);
    assert(v->type != COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH);

    if (NODE_IS_ATOMIC(u))
    {
        if (NODE_IS_ATOMIC(v))
        {
            // nothing to do
        }
        else
        {
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
    }
    else
    {
        // u is non-atomic, so it must be a batch
        assert(u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && u->command && u->command->type == COMMAND_TYPE_BATCH && u->command->batch.cg);
        // nothing to do
    }

    /* always contract in place */
    cg->contract<hint | COMMAND_GRAPH_CONTRACTION_HINT_INPLACE>(u, v);

    /* mark 'v' as contracted to skip it from future contractions */
    nodes[v->iterator_index].data.contracted = true;

    /**
     *  A node is atomic if it is empty, or a non-batch command, or a batch command with no CGIR attached
     *
     *  At this point, either
     *      (a) u is atomic, v must be too      -> batch 'u' and 'v' to the new batch 'u'
     *      (b) u is not atomic
     *          (b.1) v is atomic               -> add 'v' to the graph of 'u'
     *          (b.2) v is not atomic           -> merge the two graph 'u' and 'v'
     */

    if (NODE_IS_ATOMIC(u))
    {
        // (a)
        assert(NODE_IS_ATOMIC(v));
        command_batch_init<hint>(cg, u, v);
    }
    else
    {
        if (NODE_IS_ATOMIC(v))
            command_graph_pass_batch_contract_batch_single_node<hint>(u->command->batch.cg, v);
        else
            command_graph_pass_batch_contract_batch_merge<hint>(u, v);
    }

    return u;
}

static inline bool
command_graph_pass_batch_can_batch(
    const command_graph_node_t * u,
    const command_graph_node_t * v
) {
    return u != v && u->device_unique_id == v->device_unique_id && u->type != COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH && v->type != COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH;
}

void
command_graph_t::pass_batch(void)
{
    /* Iterate through all nodes, and contract until we tried all nodes.
     * New node can be safely pushed-back: they will be iterated on. */
    constexpr bool include_entry_exit = false;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();

    command_graph_node_t * exit = this->node_get_exit();
    assert(exit);

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
            if (v == exit)
                continue ;

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

        # if 0
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
        # endif
    }
}
