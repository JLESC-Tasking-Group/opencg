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

# include <cgir/bitset2d.hpp>
# include <cgir/namespace.hpp>
# include <cgir/command.hpp>
# include <cgir/command-graph.hpp>

# include <algorithm>
# include <set>
# include <unordered_map>
# include <utility>
# include <vector>

# include <assert.h>

CGIR_NAMESPACE_USE;

/*
 * BATCH PASS - group "islands" of same-device nodes into COMMAND_TYPE_BATCH
 * sub-graphs, so a driver can capture each island as a single vendor graph
 * (CUDA/HIP graph, Level-Zero command list, ...).
 *
 * An island is a MAXIMAL, CONVEX set of same-device nodes. Convexity is what
 * makes an island executable as one atomic sub-graph: no node outside the
 * island lies on a path between two of its members (otherwise that external
 * node would have to run "in the middle" of the batch -> deadlock). Islands are
 * grown by contracting, on a scratch quotient graph, two kinds of same-device
 * relations:
 *
 *   - safe edge  u -> v : the direct edge is the ONLY u..v path (no alternate
 *                         path through another successor). Contracting it never
 *                         creates a cycle nor traps an external node, so the
 *                         result stays a convex DAG. This generalizes a plain
 *                         sequence (single-succ u / single-pred v) to any
 *                         non-bypassed edge, and subsumes v->u (every directed
 *                         edge is examined from its tail).
 *   - false twins u,v   : identical predecessor and successor sets (parallel,
 *                         unconnected) -> merging them is convex.
 *
 * Detection runs on scratch adjacency (Phase 1, no graph mutation); only then
 * is each island with >= 2 real COMMAND members materialized (Phase 2) into a
 * fresh top-level BATCH node whose sub-graph is the island's induced sub-graph.
 */

/* pass local storage (unused per-node data; islands are tracked out-of-band) */
struct pls_t {};
using node_t = command_graph_t::node_iterator_t<pls_t>;

/* A node is "sequence-able" (may belong to an OpenMP-task sequence) if it is a
 * control node, or a PROG command whose launch mode is TASK_SPAWN (a recorded
 * OpenMP task body). Only a linear chain of such nodes yields is_sequence. */
# define NODE_IS_SEQABLE(N)                                             \
    ((N)->type == COMMAND_GRAPH_NODE_TYPE_EMPTY ||                      \
     ((N)->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && (N)->command &&   \
      (N)->command->type == COMMAND_TYPE_PROG &&                        \
      (N)->command->prog.launch_mode == CGIR_COMMAND_PROG_LAUNCH_MODE_TASK_SPAWN))

void
command_graph_t::pass_batch(void)
{
    /* Index every top-level node (entry/exit included) so the scratch graph can
     * reason about the true neighborhood, including whether two nodes share the
     * entry/exit as their sole predecessor/successor (i.e. are false twins). The
     * scratch index of a node is its iterator_index in [0, n). */
    constexpr bool include_entry_exit = true;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();
    const int n = (int) nodes.size();
    if (n == 0)
        return ;

    command_graph_node_t * g_entry = this->node_get_entry();
    command_graph_node_t * g_exit  = this->node_get_exit();

    std::unordered_map<command_graph_node_t *, int> node2idx;
    node2idx.reserve((size_t) n * 2);
    for (int i = 0 ; i < n ; ++i)
        node2idx[nodes[i].node] = i;

    /* scratch quotient graph */
    std::vector<device_unique_id_t>                   dev(n);
    std::vector<char>                                 alive(n, 1);
    std::vector<char>                                 cand(n, 0);
    std::vector<std::set<int>>                        preds(n), succs(n);
    std::vector<std::vector<command_graph_node_t *>>  members(n);

    for (int i = 0 ; i < n ; ++i)
    {
        command_graph_node_t * u = nodes[i].node;
        dev[i] = u->device_unique_id;
        members[i].push_back(u);

        /* candidates: everything but entry/exit and pre-existing batches /
         * nested command-graphs (treated as opaque boundaries). */
        const bool is_boundary =
            (u == g_entry) || (u == g_exit) ||
            (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH) ||
            (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && u->command &&
             u->command->type == COMMAND_TYPE_BATCH);
        cand[i] = !is_boundary;
    }

    /* scratch adjacency (deduplicated via std::set) */
    for (int i = 0 ; i < n ; ++i)
        for (command_graph_node_t * s : nodes[i].node->successors)
        {
            auto it = node2idx.find(s);
            assert(it != node2idx.end()); /* every top-level node is indexed */
            succs[i].insert(it->second);
            preds[it->second].insert(i);
        }

    /* -------------------------------------------------------------------- *
     * Phase 1 : grow islands to a fixpoint on the scratch graph.           *
     * -------------------------------------------------------------------- */

    auto can_batch = [&] (int a, int b) {
        return a != b && alive[a] && alive[b] && cand[a] && cand[b] && dev[a] == dev[b];
    };

    auto false_twins = [&] (int a, int b) {
        return preds[a] == preds[b] && succs[a] == succs[b];
    };

    /* reachability over the current scratch graph: reach.test(i,j) == "j is
     * reachable from i (inclusive of i)". Recomputed after each contraction so
     * the convexity test below is always exact on the quotient graph. */
    bitset2d_t<uint64_t, int> reach(n);
    std::vector<int> stack;
    auto compute_reach = [&] (void) {
        reach.reset_all();
        for (int i = 0 ; i < n ; ++i)
        {
            if (!alive[i])
                continue ;
            reach.set(i, i);
            stack.clear();
            stack.push_back(i);
            while (!stack.empty())
            {
                const int x = stack.back();
                stack.pop_back();
                for (int y : succs[x])
                    if (!reach.test(i, y))
                    {
                        reach.set(i, y);
                        stack.push_back(y);
                    }
            }
        }
    };

    /* an edge u->v is convex-safe to contract iff it is the only u..v path,
     * i.e. no OTHER successor w of u can still reach v. */
    auto safe_edge = [&] (int u, int v) {
        if (succs[u].find(v) == succs[u].end())
            return false ;
        for (int w : succs[u])
        {
            if (w == v)
                continue ;
            if (reach.test(w, v))
                return false ;
        }
        return true ;
    };

    /* absorb island 'b' into island 'a' */
    auto merge = [&] (int a, int b) {
        for (int x : preds[b])
        {
            if (x == a)
                continue ;
            succs[x].erase(b);
            succs[x].insert(a);
            preds[a].insert(x);
        }
        for (int y : succs[b])
        {
            if (y == a)
                continue ;
            preds[y].erase(b);
            preds[y].insert(a);
            succs[a].insert(y);
        }
        /* drop the (now self-) edges between a and b */
        preds[a].erase(b);
        succs[a].erase(b);
        preds[a].erase(a);
        succs[a].erase(a);

        for (command_graph_node_t * m : members[b])
            members[a].push_back(m);

        alive[b] = 0;
        preds[b].clear();
        succs[b].clear();
        members[b].clear();
    };

    bool changed = true;
    while (changed)
    {
        changed = false;
        compute_reach();

        for (int a = 0 ; a < n && !changed ; ++a)
        {
            if (!alive[a] || !cand[a])
                continue ;

            /* 1. false twins (share a predecessor -> compare full neighborhood) */
            for (int p : preds[a])
            {
                for (int b : succs[p])
                    if (can_batch(a, b) && false_twins(a, b))
                    {
                        merge(a, b);
                        changed = true;
                        break ;
                    }
                if (changed)
                    break ;
            }
            if (changed)
                break ;

            /* 2. convex-safe edges a->b (covers sequences and, over all a, v->u) */
            for (int b : succs[a])
                if (can_batch(a, b) && safe_edge(a, b))
                {
                    merge(a, b);
                    changed = true;
                    break ;
                }
        }
    }

    /* -------------------------------------------------------------------- *
     * Phase 2 : materialize each island with >= 2 COMMAND members.         *
     * -------------------------------------------------------------------- */

    /* island root of every node index (its surviving representative) */
    std::vector<int> root(n, -1);
    for (int a = 0 ; a < n ; ++a)
        if (alive[a])
            for (command_graph_node_t * m : members[a])
                root[m->iterator_index] = a;

    /* rep[i] : the top-level node that replaces node i after batching
     * (itself for entry/exit/singletons, the new BATCH node for members). */
    std::vector<command_graph_node_t *> rep(n);
    for (int i = 0 ; i < n ; ++i)
        rep[i] = nodes[i].node;

    for (int a = 0 ; a < n ; ++a)
    {
        if (!alive[a] || !cand[a])
            continue ;

        /* count real commands and pick a real (non-255) device */
        int ncmd = 0;
        device_unique_id_t gdev = dev[a];
        for (command_graph_node_t * m : members[a])
            if (m->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
            {
                if (ncmd == 0)
                    gdev = m->device_unique_id;
                ++ncmd;
            }
        if (ncmd < 2)
            continue ;

        /* new BATCH command + its (initialized) sub command-graph */
        assert(this->command_new);
        command_t * cmd = this->command_new(this, COMMAND_TYPE_BATCH);
        assert(cmd);
        cmd->batch.driver_handle = NULL;

        assert(this->command_graph_new);
        command_graph_t * sub = this->command_graph_new(this);
        assert(sub);
        cmd->batch.cg = sub;

        command_graph_node_t * s_entry = sub->node_get_entry();
        command_graph_node_t * s_exit  = sub->node_get_exit();
        assert(s_entry && s_exit);
        s_entry->successors.clear();     /* drop the default entry->exit edge */
        s_exit->predecessors.clear();

        /* clone every member into the sub-graph */
        std::unordered_map<command_graph_node_t *, command_graph_node_t *> map;
        map.reserve(members[a].size() * 2);
        for (command_graph_node_t * m : members[a])
        {
            assert(sub->command_graph_node_new);
            command_graph_node_t * mm = sub->command_graph_node_new(sub, m->device_unique_id, m->type);
            assert(mm);
            if (m->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
                mm->command = m->command;
            map[m] = mm;
        }

        /* rebuild the induced sub-graph, wire the boundary to entry/exit, and
         * decide is_sequence (a linear chain of seqable nodes). */
        bool seq     = true;
        int  sources = 0;
        int  sinks   = 0;
        for (command_graph_node_t * m : members[a])
        {
            int iin  = 0;
            int iout = 0;

            for (command_graph_node_t * s : m->successors)
            {
                auto it = node2idx.find(s);
                if (it != node2idx.end() && root[it->second] == a)
                {
                    map[m]->precedes(map[s]);   /* internal edge (added once, from its tail) */
                    ++iout;
                }
            }
            for (command_graph_node_t * p : m->predecessors)
            {
                auto it = node2idx.find(p);
                if (it != node2idx.end() && root[it->second] == a)
                    ++iin;
            }

            if (iin == 0)  { s_entry->precedes(map[m]); ++sources; }
            if (iout == 0) { map[m]->precedes(s_exit);  ++sinks;   }
            if (iin > 1 || iout > 1)
                seq = false;
            if (!NODE_IS_SEQABLE(m))
                seq = false;
        }
        sub->is_sequence = seq && (sources == 1) && (sinks == 1);

        /* fresh top-level BATCH node; every member now maps to it */
        assert(this->command_graph_node_new);
        command_graph_node_t * B = this->command_graph_node_new(this, gdev, COMMAND_GRAPH_NODE_TYPE_COMMAND);
        assert(B);
        B->command = cmd;
        for (command_graph_node_t * m : members[a])
            rep[m->iterator_index] = B;
    }

    /* -------------------------------------------------------------------- *
     * Rewire the top level: map every original edge through rep(), dropping *
     * island-internal edges and deduplicating. Cross-island edges become   *
     * B_i -> B_j automatically; member nodes end up detached (orphaned).    *
     * -------------------------------------------------------------------- */
    std::vector<std::pair<command_graph_node_t *, command_graph_node_t *>> edges;
    for (int i = 0 ; i < n ; ++i)
        for (command_graph_node_t * s : nodes[i].node->successors)
            edges.emplace_back(nodes[i].node, s);

    for (int i = 0 ; i < n ; ++i)
    {
        nodes[i].node->successors.clear();
        nodes[i].node->predecessors.clear();
    }

    for (auto & e : edges)
    {
        command_graph_node_t * a2 = rep[e.first->iterator_index];
        command_graph_node_t * b2 = rep[e.second->iterator_index];
        if (a2 == b2)
            continue ;   /* island-internal edge */
        if (std::find(a2->successors.begin(), a2->successors.end(), b2) == a2->successors.end())
            a2->precedes(b2);
    }

# ifndef NDEBUG
    this->coherence_checks();
# endif
}
