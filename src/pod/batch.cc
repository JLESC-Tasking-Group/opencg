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
 * A set S may only be packed when it is *admissible*: monochromatic, and
 *      forall u in N-(S), forall v in S : u < v      and
 *      forall v in S, forall w in N+(S) : v < w
 * with '<' the precedence (transitive closure of E). This is parallelism
 * preservation: the packed node inherits N-(S) as its predecessors and N+(S) as
 * its successors, so unless every boundary node already precedes (resp. follows)
 * every member, packing adds an ordering that G did not have -- which serializes
 * commands, and on a multi-device graph serializes whole devices. Admissible
 * also implies convex, so it rules out the cycle a non-convex packing creates.
 */

/* per-node pass storage (the device stays on the node: node->device_unique_id) */
struct batch_pls_t
{
    bool candidate;                 /* batching candidate (not entry/exit, not an opaque batch/graph node) */
    int cls;                        /* refinement class index, or -1 for a non-candidate */
    command_graph_node_t * rep;     /* representative top-level node after batching */
};

void
command_graph_t::pass_batch(void)
{
    /* Index every top-level node; node->iterator_index is its index in [0, n). */
    constexpr bool include_entry_exit = true;
    auto nodes = this->create_node_iterators<include_entry_exit, batch_pls_t>();
    const int n = (int) nodes.size();
    if (n <= 2)   /* only entry/exit: nothing to batch */
        return ;

    command_graph_node_t * g_entry = this->node_get_entry();
    command_graph_node_t * g_exit  = this->node_get_exit();

    /* initialize per-node storage: candidate flag (entry/exit and pre-existing
     * batches / nested graphs are opaque boundaries, never grouped), refinement
     * class (none yet), and post-batch representative (self). */
    for (int i = 0 ; i < n ; ++i)
    {
        command_graph_node_t * u = nodes[i].node;
        const bool boundary =
            (u == g_entry) || (u == g_exit) ||
            (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH) ||
            (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && u->command && u->command->type == COMMAND_TYPE_BATCH);
        nodes[i].data.candidate = !boundary;
        nodes[i].data.cls       = -1;
        nodes[i].data.rep       = u;
    }

    /* -------------------------------------------------------------------- *
     * Partition into maximal admissible sets, by refinement.               *
     *                                                                      *
     * Start from the partition of the candidates by device, then split a   *
     * class C by any node x outside it that C's members do not all see the *
     * same way: the members x precedes, those that precede x, and those    *
     * concurrent with x, become three classes.  Iterate to a fixed point.  *
     *                                                                      *
     * Correctness: refinement only splits, so every class stays            *
     * monochromatic; the stopping condition is exactly admissibility; and  *
     * an admissible T inside a class C is never cut, because a splitter of *
     * C lies outside C, hence outside T, hence sees all of T the same way. *
     * Each split adds a class, so at most |V| - 1 of them happen.          *
     * -------------------------------------------------------------------- */

    /* initial classes: one per device, over the candidates only */
    int ncls = 0;
    {
        std::unordered_map<device_unique_id_t, int> by_device;
        for (int i = 0 ; i < n ; ++i)
        {
            if (!nodes[i].data.candidate)
                continue ;
            auto it = by_device.find(nodes[i].node->device_unique_id);
            if (it == by_device.end())
                it = by_device.emplace(nodes[i].node->device_unique_id, ncls++).first;
            nodes[i].data.cls = it->second;
        }
    }

    /* A splitter is a node outside the class it splits. With a single class,
     * the only nodes outside it are the entry, the exit, and the opaque nodes
     * a previous pass left behind. The entry precedes and the exit follows
     * every command, so neither can split anything: if there is no opaque node
     * either, the initial partition is already the answer and the |V|^2-bit
     * precedence matrix is never built. This is the single-device case. */
    bool may_split = (ncls > 1);
    if (!may_split)
        for (int i = 0 ; i < n ; ++i)
            if (!nodes[i].data.candidate && nodes[i].node != g_entry && nodes[i].node != g_exit)
            {
                may_split = true;
                break ;
            }

    if (may_split)
    {
        /* reachability of the *original* graph: r.test(a, b) iff a < b (the
         * diagonal is set, but a splitter is never a member of the class it
         * splits, so the pairs tested below always have a != b) */
        bitset2d_t<uint64_t, command_graph_node_index_t> r(n);
        this->walk<COMMAND_GRAPH_WALK_SEARCH_DFS, COMMAND_GRAPH_WALK_ORDER_POST>(
            [&] (command_graph_node_t * node)
            {
                r.set(node->iterator_index, node->iterator_index);
                for (command_graph_node_t * succ : node->successors)
                    r.or_rows(node->iterator_index, succ->iterator_index);
            }
        );

        /* (class, x precedes v, v precedes x) -> class of v after the split */
        std::map<std::tuple<int, bool, bool>, int> bucket;
        std::set<int> touched;      /* classes that already kept their index */

        bool changed = true;
        while (changed)
        {
            changed = false;

            for (int x = 0 ; x < n ; ++x)
            {
                const int  xcls = nodes[x].data.cls;
                const auto xi   = (command_graph_node_index_t) x;
                bucket.clear();
                touched.clear();

                for (int v = 0 ; v < n ; ++v)
                {
                    const int c = nodes[v].data.cls;
                    if (c < 0 || c == xcls)
                        continue ;      /* not a candidate, or x is inside v's class */

                    const auto vi = (command_graph_node_index_t) v;
                    const auto key = std::make_tuple(c, r.test(xi, vi), r.test(vi, xi));

                    auto it = bucket.find(key);
                    if (it == bucket.end())
                    {
                        /* the first key seen for a class keeps the class index,
                         * the next ones open a new class */
                        const bool first = touched.insert(c).second;
                        const int  id    = first ? c : ncls++;
                        it = bucket.emplace(key, id).first;
                        if (!first)
                            changed = true;
                    }
                    nodes[v].data.cls = it->second;
                }
            }
        }
    }

    /* group the candidates by class; the key is the index of the first member,
     * so that nodes[key] is a node of the island */
    std::unordered_map<int, std::vector<command_graph_node_t *>> island;
    {
        std::unordered_map<int, int> root_of;    /* class -> first member index */
        for (int i = 0 ; i < n ; ++i)
        {
            const int c = nodes[i].data.cls;
            if (c < 0)
                continue ;
            auto it = root_of.find(c);
            if (it == root_of.end())
                it = root_of.emplace(c, i).first;
            island[it->second].push_back(nodes[i].node);
        }
    }

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
        {
            if (m->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
            {
                if (ncmd == 0)
                    gdev = m->device_unique_id;
                ++ncmd;
            }
        }
        if (ncmd < 2)
            continue ;

        /* fresh BATCH command + its (initialized) sub command-graph */
        assert(this->command_new && this->command_graph_new && this->command_graph_node_new);
        command_t * cmd = this->command_new(this, COMMAND_TYPE_BATCH);
        assert(cmd);

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
    for (int i = 0 ; i < n ; ++i)
    {
        command_graph_node_t * u  = nodes[i].node;
        assert(u);

        /* Iterate a *copy* of the successors. When u is not itself batched we
         * have ru == u, so the ru->precedes(rv) below appends the fresh BATCH
         * node to the very list being walked; a std::list iterator stays valid
         * and would reach it. That node has no entry in `nodes` -- its
         * iterator_index is still the constructor's 0, which aliases the graph
         * entry -- so mapping it through nodes[] would silently reroute the
         * edge onto the entry. */
        const std::vector<command_graph_node_t *> out(u->successors.begin(), u->successors.end());

        /* for each edge u->v */
        for (command_graph_node_t * v : out)
        {
            assert(v);
            assert(v->iterator_index < (command_graph_node_index_t) n &&
                   nodes[v->iterator_index].node == v);

            command_graph_node_t * ru = nodes[u->iterator_index].data.rep;
            command_graph_node_t * rv = nodes[v->iterator_index].data.rep;
            assert(ru);
            assert(rv);

            if (ru == rv)           continue; /* internal to one island: keep as-is */
            if (ru == u && rv == v) continue; /* neither endpoint batched: keep as-is */

            /* boundary edge: detach the original, add the mapped one (deduplicated) */
            u->successors.erase(std::find(u->successors.begin(), u->successors.end(), v));
            v->predecessors.erase(std::find(v->predecessors.begin(), v->predecessors.end(), u));
            if (std::find(ru->successors.begin(), ru->successors.end(), rv) == ru->successors.end())
                ru->precedes(rv);
        }
    }

    /* Connect each sub-graph's entry/exit to the island's sources/sinks. After
     * boundary removal a member's remaining edges are all internal, so a source
     * has no predecessor and a sink no successor. (is_serial stays false here;
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
