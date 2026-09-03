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
 *
 * Four stages:
 *      1. seed     - same-device connected components
 *      2. merge    - fold same-device twin islands, to a fixed point
 *      3. restrict - drop members until every island is admissible
 *      4. extend   - add unpacked nodes back while admissibility is preserved
 *
 * Stages 1-2 deliberately ignore admissibility, and stage 3 is not cleanup: it
 * is the projection back onto the feasible sets. The order cannot be changed:
 *
 *   - Re-running stage 1 after stage 3 does not terminate. It would re-absorb
 *     exactly the node stage 3 removed (they are joined by a same-device edge),
 *     so the two would oscillate with period two.
 *
 *   - Growing only through admissible states -- never producing a broken island
 *     in the first place -- cannot reach the answer, because admissibility is
 *     not closed under union. In tests/batch.cc test_batch_island, of the 4095
 *     non-trivial subsets only 19 are admissible, with sizes {1,2,3,4,12}, and
 *     no bipartition of the 12 nodes into two admissible halves exists. The
 *     correct 12-node packing is admissible but unreachable by any merge
 *     sequence that keeps every intermediate admissible; such a scheme stops at
 *     size 4.
 *
 * Stage 4 exists so that saturation is a postcondition rather than a hope: on
 * exit, no unpacked node can be added to any island. It cannot loop with stage
 * 3, because every addition it makes preserves admissibility.
 */

/* per-node pass storage (the device stays on the node: node->device_unique_id) */
struct batch_pls_t
{
    bool candidate;                 /* batching candidate (not entry/exit, not an opaque batch/graph node) */
    int parent;                     /* union-find parent index (island detection) */
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
     * batches / nested graphs are opaque boundaries, never grouped), union-find
     * parent (self), and post-batch representative (self). */
    for (int i = 0 ; i < n ; ++i)
    {
        command_graph_node_t * u = nodes[i].node;
        const bool boundary =
            (u == g_entry) || (u == g_exit) ||
            (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH) ||
            (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && u->command && u->command->type == COMMAND_TYPE_BATCH);
        nodes[i].data.candidate   = !boundary;
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

    /* Stage 1 needs no explicit series rule, and needs no re-run after stage 2:
     *   - a series edge is a same-device edge, so the flood-fill below already
     *     subsumes it;
     *   - twins are never edge-connected (an edge a->b with a in A, b in B would
     *     put b in N+(A) = N+(B), which cannot hold since N+(B) excludes B), so
     *     folding twins never creates a same-device edge-adjacency that the
     *     flood-fill had not already collapsed. */

    /* 1. seed: flood-fill along same-device edges */
    for (int i = 0 ; i < n ; ++i)
    {
        if (!nodes[i].data.candidate)
            continue ;
        const device_unique_id_t di = nodes[i].node->device_unique_id;
        for (command_graph_node_t * s : nodes[i].node->successors)
        {
            const int j = (int) s->iterator_index;
            if (nodes[j].data.candidate && di == s->device_unique_id)
                unite(i, j);
        }
    }

    /* 2. merge: fold same-device false twins. Two islands with the same device
     * and the same set of neighbor-islands (predecessors and successors). This
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
            if (!nodes[i].data.candidate || find(i) != i)   /* candidate island roots only */
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
        if (nodes[i].data.candidate)
            island[find(i)].push_back(nodes[i].node);

    /* -------------------------------------------------------------------- *
     * 3. restrict / 4. extend                                               *
     *                                                                       *
     * restrict is the projection back onto the admissible sets. Operator R: *
     * pick a non-admissible island S and a witness,                         *
     *   - u in N-(S) with u not< v : X = the successors of u that are in S  *
     *   - w in N+(S) with v not< w : X = the predecessors of w that are in S*
     * and move X out of S. X is never empty (u has an edge into S, by       *
     * definition of N-(S)) and never all of S (if u had an edge to every    *
     * member then u < v for all v, contradicting the witness), so S stays   *
     * non-empty and strictly shrinks.                                       *
     *   Termination: PHI = sum over islands of (|S| - 1) is a non-negative  *
     * integer that each step decreases by |X| >= 1, so R reaches a fixed    *
     * point in at most |V| - 1 steps, where every island is admissible.     *
     *                                                                       *
     * extend then adds unpacked nodes back while admissibility holds. Each  *
     * addition moves one node from unpacked to packed, so it terminates in  *
     * at most |V| steps; and since every addition preserves admissibility,  *
     * restrict never has to run again. On exit no unpacked node can be      *
     * added to any island (saturation).                                     *
     *                                                                       *
     * Both need the precedence relation, which costs |V|^2 bits. The scan   *
     * below skips it when it is provably useless. On a single-device graph  *
     * that is always the case: two same-device islands can never be edge-   *
     * adjacent (stage 1 would have merged them), so every island carries the*
     * key (dev, {s}, {t}) and stage 2 folds them all into one island        *
     * holding every candidate, whose boundary is exactly {s, t}.            *
     * -------------------------------------------------------------------- */

    {
        /* island root of each node, or -1 for a node no island holds */
        std::vector<int> iid((size_t) n, -1);
        for (auto & kv : island)
            if (kv.second.size() >= 2)
                for (command_graph_node_t * m : kv.second)
                    iid[m->iterator_index] = kv.first;

        /* An island whose boundary is contained in {s, t} is admissible for
         * free: s precedes every node and every node precedes t. */
        bool need_restrict = false;
        bool has_unpacked  = false;
        for (int i = 0 ; i < n ; ++i)
            if (nodes[i].data.candidate && iid[i] < 0)
                has_unpacked = true;
        for (auto & kv : island)
        {
            if (kv.second.size() < 2)
                continue ;
            for (command_graph_node_t * v : kv.second)
            {
                for (command_graph_node_t * u : v->predecessors)
                    if (iid[u->iterator_index] != kv.first && u != g_entry && u != g_exit)
                        need_restrict = true;
                for (command_graph_node_t * w : v->successors)
                    if (iid[w->iterator_index] != kv.first && w != g_entry && w != g_exit)
                        need_restrict = true;
            }
        }
        const bool need_extend = has_unpacked && !island.empty();

        if (need_restrict || need_extend)
        {
            /* reachability of the *original* graph: r.test(a, b) iff a < b (the
             * diagonal is set, but the pairs tested below always have a != b,
             * since a boundary node is never a member of the island tested) */
            bitset2d_t<uint64_t, command_graph_node_index_t> r(n);
            this->walk<COMMAND_GRAPH_WALK_SEARCH_DFS, COMMAND_GRAPH_WALK_ORDER_POST>(
                [&] (command_graph_node_t * node)
                {
                    r.set(node->iterator_index, node->iterator_index);
                    for (command_graph_node_t * succ : node->successors)
                        r.or_rows(node->iterator_index, succ->iterator_index);
                }
            );

            /* ---------------- 3. restrict ---------------- */
            bool restricted = true;
            while (restricted)
            {
                restricted = false;

                for (auto & kv : island)
                {
                    const int root = kv.first;
                    std::vector<command_graph_node_t *> & S = kv.second;
                    if (S.size() < 2)
                        continue ;

                    std::vector<command_graph_node_t *> X;

                    /* in-side: every external predecessor precedes every member */
                    for (command_graph_node_t * v : S)
                    {
                        for (command_graph_node_t * u : v->predecessors)
                        {
                            if (iid[u->iterator_index] == root)
                                continue ;                  /* internal edge */

                            bool bad = false;
                            for (command_graph_node_t * m : S)
                                if (!r.test(u->iterator_index, m->iterator_index))
                                {
                                    bad = true;
                                    break ;
                                }
                            if (!bad)
                                continue ;

                            for (command_graph_node_t * x : u->successors)
                                if (iid[x->iterator_index] == root)
                                    X.push_back(x);
                            break ;
                        }
                        if (!X.empty())
                            break ;
                    }

                    /* out-side: every member precedes every external successor */
                    if (X.empty())
                    {
                        for (command_graph_node_t * v : S)
                        {
                            for (command_graph_node_t * w : v->successors)
                            {
                                if (iid[w->iterator_index] == root)
                                    continue ;              /* internal edge */

                                bool bad = false;
                                for (command_graph_node_t * m : S)
                                    if (!r.test(m->iterator_index, w->iterator_index))
                                    {
                                        bad = true;
                                        break ;
                                    }
                                if (!bad)
                                    continue ;

                                for (command_graph_node_t * x : w->predecessors)
                                    if (iid[x->iterator_index] == root)
                                        X.push_back(x);
                                break ;
                            }
                            if (!X.empty())
                                break ;
                        }
                    }

                    if (X.empty())
                        continue ;                          /* S is admissible */

                    const size_t before = S.size();
                    for (command_graph_node_t * x : X)
                        iid[x->iterator_index] = -1;
                    S.erase(
                        std::remove_if(S.begin(), S.end(),
                            [&] (command_graph_node_t * m) { return iid[m->iterator_index] != root; }),
                        S.end()
                    );
                    assert(!S.empty());
                    assert(S.size() < before);
                    (void) before;
                    restricted = true;
                    break ;                                 /* restart the scan */
                }
            }

            /* ---------------- 4. extend ---------------- */

            /* an island restrict shrank below two members is not a batch any
             * more: release it so its node can be re-homed below */
            for (auto & kv : island)
                if (kv.second.size() < 2)
                    for (command_graph_node_t * m : kv.second)
                        iid[m->iterator_index] = -1;

            /* true iff S + {v} is admissible; S is assumed admissible already */
            auto admissible_with = [&] (
                const std::vector<command_graph_node_t *> & S,
                const int root,
                command_graph_node_t * v
            ) {
                /* every external predecessor of S + {v} precedes every member */
                auto ok_in = [&] (command_graph_node_t * u)
                {
                    if (u == v || iid[u->iterator_index] == root)
                        return true;                        /* internal */
                    if (!r.test(u->iterator_index, v->iterator_index))
                        return false;
                    for (command_graph_node_t * m : S)
                        if (!r.test(u->iterator_index, m->iterator_index))
                            return false;
                    return true;
                };
                auto ok_out = [&] (command_graph_node_t * w)
                {
                    if (w == v || iid[w->iterator_index] == root)
                        return true;                        /* internal */
                    if (!r.test(v->iterator_index, w->iterator_index))
                        return false;
                    for (command_graph_node_t * m : S)
                        if (!r.test(m->iterator_index, w->iterator_index))
                            return false;
                    return true;
                };
                for (command_graph_node_t * u : v->predecessors)
                    if (!ok_in(u)) return false;
                for (command_graph_node_t * w : v->successors)
                    if (!ok_out(w)) return false;
                for (command_graph_node_t * m : S)
                {
                    for (command_graph_node_t * u : m->predecessors)
                        if (!ok_in(u)) return false;
                    for (command_graph_node_t * w : m->successors)
                        if (!ok_out(w)) return false;
                }
                return true;
            };

            bool extended = true;
            while (extended)
            {
                extended = false;

                for (auto & kv : island)
                {
                    const int root = kv.first;
                    std::vector<command_graph_node_t *> & S = kv.second;
                    if (S.size() < 2)
                        continue ;
                    const device_unique_id_t dev = S.front()->device_unique_id;

                    for (int i = 0 ; i < n ; ++i)
                    {
                        command_graph_node_t * v = nodes[i].node;
                        if (!nodes[i].data.candidate || iid[i] >= 0)
                            continue ;                      /* already packed */
                        if (v->device_unique_id != dev)
                            continue ;
                        if (!admissible_with(S, root, v))
                            continue ;

                        iid[i] = root;
                        S.push_back(v);
                        extended = true;
                        break ;
                    }
                    if (extended)
                        break ;
                }
            }
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
