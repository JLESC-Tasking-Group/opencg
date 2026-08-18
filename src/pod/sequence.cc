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

# include <algorithm>
# include <utility>
# include <vector>

# include <assert.h>

CGIR_NAMESPACE_USE;

/*
 * SEQUENCE PASS - group maximal linear chains (u -> v -> ... -> w) of
 * same-device task nodes into COMMAND_TYPE_BATCH sub-graphs flagged
 * `is_sequence`. Such a batch is a plain sequence of OpenMP task bodies, which
 * the runtime replays as a single "super" task instead of one task per command
 * (see command_graph_t::is_sequence and xkrt's command_graph_replay_sequence).
 */

/* per-node pass storage */
struct sequence_pls_t
{
    bool contracted;
    sequence_pls_t(void) : contracted(false) {}
    ~sequence_pls_t(void) {}
};

/* A node is "sequence-able" (may belong to an OpenMP-task sequence) if it is a
 * control node, or a PROG command whose launch mode is TASK_SPAWN (a recorded
 * OpenMP task body). */
# define NODE_IS_SEQABLE(N)                                             \
    ((N)->type == COMMAND_GRAPH_NODE_TYPE_EMPTY ||                      \
     ((N)->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && (N)->command &&   \
      (N)->command->type == COMMAND_TYPE_PROG &&                        \
      (N)->command->prog.launch_mode == CGIR_COMMAND_PROG_LAUNCH_MODE_TASK_SPAWN))

/* Contract the chain [head .. tail] into the fresh BATCH node 'B' whose
 * sub-graph is 'sub'. The chain's internal edges are left untouched; only the
 * two boundaries move onto 'B', and the sub-graph entry/exit wrap head/tail. */
/* Move the chain's external boundary onto the batch node 'B': the head's
 * predecessors and the tail's successors (all external). Leaves the head with no
 * predecessor and the tail with no successor, so they can be reused as the sub
 * command-graph's entry/exit. The internal chain edges are untouched. */
static inline void
command_graph_pass_sequence_detach_boundary(
    command_graph_node_t * B,
    command_graph_node_t * head,
    command_graph_node_t * tail
) {
    /* head's predecessors (all external) now precede the batch node */
    for (command_graph_node_t * p : head->predecessors)
    {
        auto it = std::find(p->successors.begin(), p->successors.end(), head);
        assert(it != p->successors.end());
        if (std::find(p->successors.begin(), p->successors.end(), B) == p->successors.end())
            *it = B;
        else
            p->successors.erase(it);
    }
    B->predecessors = std::move(head->predecessors);
    head->predecessors.clear();

    /* tail's successors (all external) now succeed the batch node */
    for (command_graph_node_t * s : tail->successors)
    {
        auto it = std::find(s->predecessors.begin(), s->predecessors.end(), tail);
        assert(it != s->predecessors.end());
        if (std::find(s->predecessors.begin(), s->predecessors.end(), B) == s->predecessors.end())
            *it = B;
        else
            s->predecessors.erase(it);
    }
    B->successors = std::move(tail->successors);
    tail->successors.clear();
}

void
command_graph_t::pass_sequence(void)
{
    /* forward DFS pre-order from entry: a chain's head precedes its interior */
    constexpr bool include_entry_exit = true;
    auto nodes = this->create_node_iterators<include_entry_exit, sequence_pls_t>();
    const int n = (int) nodes.size();
    if (n <= 2)   /* only entry/exit: nothing to sequence */
        return ;

    command_graph_node_t * g_entry = this->node_get_entry();
    command_graph_node_t * g_exit  = this->node_get_exit();

    auto is_cand = [&] (command_graph_node_t * x) {
        return x != g_entry && x != g_exit && NODE_IS_SEQABLE(x);
    };

    for (int i = 0 ; i < n ; ++i)
    {
        command_graph_node_t * u = nodes[i].node;

        if (nodes[i].data.contracted || !is_cand(u) || u->successors.size() != 1)
            continue ;

        command_graph_node_t * v = u->successors.front();
        if (!is_cand(v) || v->device_unique_id != u->device_unique_id || v->predecessors.size() != 1)
            continue ;
        assert(v->predecessors.front() == u);   /* v's single predecessor is u */

        /* walk the maximal chain once: mark members contracted, count COMMANDs,
         * and remember the tail. All members share u's device (the walk requires
         * nxt->device == cur->device at every step), so the batch device is
         * simply u->device_unique_id. */
        nodes[u->iterator_index].data.contracted = true;
        int ncmd = (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND) ? 1 : 0;

        command_graph_node_t * tail = u;
        command_graph_node_t * cur  = u;
        while (cur->successors.size() == 1)
        {
            command_graph_node_t * nxt = cur->successors.front();
            if (!is_cand(nxt) ||
                nxt->device_unique_id != cur->device_unique_id ||
                nxt->predecessors.size() != 1)
                break ;
            nodes[nxt->iterator_index].data.contracted = true;
            if (nxt->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
                ++ncmd;
            tail = nxt;
            cur  = nxt;
        }

        if (ncmd < 2)
            continue ;
        /* fresh BATCH command + batch node */
        assert(this->command_new && this->command_graph_new && this->command_graph_node_new);
        command_t * cmd = this->command_new(this, COMMAND_TYPE_BATCH);
        assert(cmd);

        command_graph_node_t * B =
            this->command_graph_node_new(this, u->device_unique_id, COMMAND_GRAPH_NODE_TYPE_COMMAND);
        assert(B);
        B->command = cmd;

        /* move the chain's external boundary onto B, leaving the head (u) with no
         * predecessor and the tail with no successor */
        command_graph_pass_sequence_detach_boundary(B, u, tail);

        /* the sub command-graph reuses the chain head/tail as entry/exit (no
         * extra empty control nodes); the internal chain edges are kept as-is */
        command_graph_t * sub = this->command_graph_new(this, u, tail);
        assert(sub);
        cmd->batch.cg    = sub;
        sub->is_sequence = true;
    }

# ifndef NDEBUG
    this->coherence_checks();
# endif
}
