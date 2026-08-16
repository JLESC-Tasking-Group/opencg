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
# include <map>
# include <set>
# include <tuple>
# include <unordered_map>
# include <vector>

# include <assert.h>

CGIR_NAMESPACE_USE;

/*
 * BATCH PASS - group same-device "islands" into COMMAND_TYPE_BATCH sub-graphs so
 * a driver can capture each island as a single vendor graph (CUDA/HIP graph,
 * Level-Zero command list, ...).
 *
 * A command graph is just a DAG whose nodes carry a device id. An island is a
 * connected region of same-device nodes: it is found by a flood-fill
 * (union-find) that unites same-device nodes joined by an edge, then folds in
 * same-device false twins (identical neighborhood, e.g. two independent ops with
 * the same dependencies). Each island with >= 2 COMMAND nodes is materialized
 * IN PLACE into a fresh BATCH node whose sub-graph reuses the very same node
 * objects and their internal edges; only the island boundary is rewired.
 */

/* per-node pass storage (the device stays on the node: node->device_unique_id) */
struct pls_t
{
    bool                   cand;    /* batching candidate (not entry/exit, not an opaque batch/graph node) */
    int                    parent;  /* union-find parent index (island detection) */
    command_graph_node_t * rep;     /* representative top-level node after batching */
};
using node_t = command_graph_t::node_iterator_t<pls_t>;

void
command_graph_t::pass_batch(void)
{
    /* Index every top-level node; node->iterator_index is its index in [0, n). */
    constexpr bool include_entry_exit = true;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();
    const int n = (int) nodes.size();
    if (n <= 2)   /* only entry/exit: nothing to batch */
        return ;

    command_graph_node_t * g_entry = this->node_get_entry();
    command_graph_node_t * g_exit  = this->node_get_exit();

    /* initialize per-node storage: candidate flag (entry/exit and pre-existing
     * batches / nested graphs are opaque boundaries, never grouped), union-find
     * parent (self), and post-batch representative (self). */
    for (int i = 0 ; i < n ; ++i)
    {
        command_graph_node_t * u = nodes[i].node;
        const bool boundary =
            (u == g_entry) || (u == g_exit) ||
            (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH) ||
            (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && u->command &&
             u->command->type == COMMAND_TYPE_BATCH);
        nodes[i].data.cand   = !boundary;
        nodes[i].data.parent = i;
        nodes[i].data.rep    = u;
    }

    /* -------------------------------------------------------------------- *
     * Detect islands: union-find flood-fill.                               *
     * -------------------------------------------------------------------- */

    auto find = [&] (int x)
    {
        while (nodes[x].data.parent != x)
        {
            nodes[x].data.parent = nodes[nodes[x].data.parent].data.parent;
            x = nodes[x].data.parent;
        }
        return x;
    };

    auto unite = [&] (int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a != b)
            nodes[a].data.parent = b;
    };

    /* 1. flood-fill along same-device edges */
    for (int i = 0 ; i < n ; ++i)
    {
        if (!nodes[i].data.cand)
            continue ;
        const device_unique_id_t di = nodes[i].node->device_unique_id;
        for (command_graph_node_t * s : nodes[i].node->successors)
        {
            const int j = (int) s->iterator_index;
            if (nodes[j].data.cand && di == s->device_unique_id)
                unite(i, j);
        }
    }

    /* 2. fold same-device false twins: two islands with the same device and the
     * same set of neighbor-islands (predecessors and successors) are twins. This
     * is re-evaluated on the quotient until stable (so twins that only appear
     * after an edge-merge, e.g. a sibling of a merged chain, are caught too). */
    bool changed = true;
    while (changed)
    {
        changed = false;

        std::unordered_map<int, std::set<int>> rpred, rsucc;
        for (int i = 0 ; i < n ; ++i)
        {
            const int ri = find(i);
            for (command_graph_node_t * s : nodes[i].node->successors)
            {
                const int rj = find((int) s->iterator_index);
                if (ri != rj)
                {
                    rsucc[ri].insert(rj);
                    rpred[rj].insert(ri);
                }
            }
        }

        std::map<std::tuple<device_unique_id_t, std::set<int>, std::set<int>>, int> seen;
        for (int i = 0 ; i < n ; ++i)
        {
            if (!nodes[i].data.cand || find(i) != i)   /* candidate island roots only */
                continue ;
            auto key = std::make_tuple(nodes[i].node->device_unique_id, rpred[i], rsucc[i]);
            auto it  = seen.find(key);
            if (it == seen.end())
                seen.emplace(std::move(key), i);
            else
            {
                unite(i, it->second);
                changed = true;
            }
        }
    }

    /* group members by island root (candidates only) */
    std::unordered_map<int, std::vector<command_graph_node_t *>> island;
    for (int i = 0 ; i < n ; ++i)
        if (nodes[i].data.cand)
            island[find(i)].push_back(nodes[i].node);

    /* -------------------------------------------------------------------- *
     * Materialize each island with >= 2 COMMAND members.                   *
     * -------------------------------------------------------------------- */

    /* nodes[i].data.rep already holds each node's post-batch representative
     * (itself); materialization overwrites it with the BATCH node for members. */
    struct island_mat_t
    {
        command_graph_t *      sub;
        command_graph_node_t * s_entry;
        command_graph_node_t * s_exit;
    };
    std::unordered_map<int, island_mat_t> mat;

    for (auto & kv : island)
    {
        std::vector<command_graph_node_t *> & members = kv.second;

        int                ncmd = 0;
        device_unique_id_t gdev = nodes[kv.first].node->device_unique_id;
        for (command_graph_node_t * m : members)
            if (m->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
            {
                if (ncmd == 0)
                    gdev = m->device_unique_id;
                ++ncmd;
            }
        if (ncmd < 2)
            continue ;

        /* fresh BATCH command + its (initialized) sub command-graph */
        assert(this->command_new && this->command_graph_new && this->command_graph_node_new);
        command_t * cmd = this->command_new(this, COMMAND_TYPE_BATCH);
        assert(cmd);
        cmd->batch.driver_handle = NULL;

        command_graph_t * sub = this->command_graph_new(this, nullptr, nullptr);
        assert(sub);
        cmd->batch.cg = sub;

        command_graph_node_t * s_entry = sub->node_get_entry();
        command_graph_node_t * s_exit  = sub->node_get_exit();
        s_entry->successors.clear();     /* drop the default entry->exit edge */
        s_exit->predecessors.clear();

        /* fresh top-level BATCH node; every member maps to it */
        command_graph_node_t * B = this->command_graph_node_new(this, gdev, COMMAND_GRAPH_NODE_TYPE_COMMAND);
        assert(B);
        B->command = cmd;
        for (command_graph_node_t * m : members)
            nodes[m->iterator_index].data.rep = B;

        mat[kv.first] = island_mat_t{ sub, s_entry, s_exit };
    }

    /* Rewire boundaries only: every original edge whose endpoints map to
     * different representatives is moved onto the batch node(s); internal edges
     * (both endpoints in the same island) are left physically untouched. */
    std::vector<std::pair<command_graph_node_t *, command_graph_node_t *>> edges;
    for (int i = 0 ; i < n ; ++i)
        for (command_graph_node_t * s : nodes[i].node->successors)
            edges.emplace_back(nodes[i].node, s);

    for (auto & e : edges)
    {
        command_graph_node_t * u  = e.first;
        command_graph_node_t * v  = e.second;
        command_graph_node_t * ru = nodes[u->iterator_index].data.rep;
        command_graph_node_t * rv = nodes[v->iterator_index].data.rep;

        if (ru == rv)
            continue ;              /* internal to one island: keep as-is */
        if (ru == u && rv == v)
            continue ;              /* neither endpoint batched: keep as-is */

        /* boundary edge: detach the original, add the mapped one (deduplicated) */
        u->successors.erase(std::find(u->successors.begin(), u->successors.end(), v));
        v->predecessors.erase(std::find(v->predecessors.begin(), v->predecessors.end(), u));
        if (std::find(ru->successors.begin(), ru->successors.end(), rv) == ru->successors.end())
            ru->precedes(rv);
    }

    /* Connect each sub-graph's entry/exit to the island's sources/sinks. After
     * boundary removal a member's remaining edges are all internal, so a source
     * has no predecessor and a sink no successor. (is_sequence stays false here;
     * linear task chains are handled by the dedicated 'sequence' pass.) */
    for (auto & kv : mat)
    {
        island_mat_t & M       = kv.second;
        auto &         members = island[kv.first];

        for (command_graph_node_t * m : members)
        {
            if (m->predecessors.empty()) M.s_entry->precedes(m);
            if (m->successors.empty())   m->precedes(M.s_exit);
        }
    }

# ifndef NDEBUG
    this->coherence_checks();
# endif
}
