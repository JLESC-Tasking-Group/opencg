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

/*
 * BATCH PASS - randomized invariant harness.
 *
 * Generates random command graphs, runs the batch (packing) pass on each, and
 * checks the invariants the pass must satisfy on every input. See the header of
 * src/pod/batch.cc for the algorithm; the properties checked here are:
 *
 *   monochromatic  every packed island holds commands of a single device;
 *   admissible     every external predecessor of an island precedes all of its
 *                  members, and every member precedes all external successors
 *                  (i.e. the packing is parallelism-preserving);
 *   saturated      no unpacked command could be added to an island;
 *   acyclic        the packed top-level graph is still a DAG. Note that
 *                  coherence_checks() does *not* test this, and a packing that
 *                  is not parallelism-preserving is exactly how a cycle appears;
 *   conserved      every original command ends up in exactly one island, or
 *                  unpacked -- never lost, never duplicated.
 *
 * Two further metrics -- whether an island is inclusion-maximal, and how the
 * packing compares to an exhaustive optimum -- are exponential, so they only
 * run on small graphs. They measure heuristic *quality*, not correctness, so
 * they are reported and never asserted.
 *
 * Usage: batch-random [ngraphs] [min_nodes] [max_nodes] [seed]
 */

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>

# include <algorithm>
# include <set>
# include <unordered_map>
# include <vector>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

/* Exhaustive checks are 2^k; only run them below this many same-device nodes. */
# define EXHAUSTIVE_MAX_NODES   12

/* ------------------------------------------------------------------------- *
 * Deterministic PRNG. std::rand and <random> give implementation-defined
 * sequences, so a failing seed would not reproduce elsewhere.
 * ------------------------------------------------------------------------- */

struct rng_t
{
    uint64_t s;
    rng_t(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}

    uint64_t next(void)
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    uint32_t below(uint32_t n)      { return (uint32_t) (next() % n); }
    bool     chance(double p)       { return (double) (next() >> 11) / 9007199254740992.0 < p; }
};

/* ------------------------------------------------------------------------- *
 * Graph construction
 * ------------------------------------------------------------------------- */

static command_graph_node_t *
make_command_node(command_graph_t * cg, const device_unique_id_t device)
{
    command_t * cmd = command_new(cg, COMMAND_TYPE_COPY_D2D_1D);
    assert(cmd);
    cmd->copy_1D.src_device_unique_id = device;
    cmd->copy_1D.dst_device_unique_id = device;
    cmd->copy_1D.src_device_addr      = 0;
    cmd->copy_1D.dst_device_addr      = 0;
    cmd->copy_1D.size                 = 0;

    command_graph_node_t * node =
        command_graph_node_new(cg, device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(node);
    node->command = cmd;
    return node;
}

static command_graph_t *
make_graph(command_graph_node_t ** entry, command_graph_node_t ** exit)
{
    command_graph_t * cg = command_graph_new();
    assert(cg);
    cg->init(command_new, command_graph_node_new, command_graph_new);
    *entry = cg->node_get_entry();
    *exit  = cg->node_get_exit();
    (*entry)->successors.clear();
    (*exit)->predecessors.clear();
    return cg;
}

/* A random DAG on `nv` commands: an edge i -> j may only go forward in the
 * generation order, which makes acyclicity structural. Commands with no
 * predecessor are attached to the entry, those with no successor to the exit,
 * so the result is a well-formed command graph. */
static command_graph_t *
make_random_graph(rng_t & rng, const int nv, const int ndev, const double p,
                  std::vector<command_graph_node_t *> & cmds)
{
    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    cmds.clear();
    for (int i = 0 ; i < nv ; ++i)
        cmds.push_back(make_command_node(cg, (device_unique_id_t) rng.below((uint32_t) ndev)));

    for (int i = 0 ; i < nv ; ++i)
        for (int j = i + 1 ; j < nv ; ++j)
            if (rng.chance(p))
                cmds[i]->precedes(cmds[j]);

    for (int i = 0 ; i < nv ; ++i)
    {
        if (cmds[i]->predecessors.empty()) entry->precedes(cmds[i]);
        if (cmds[i]->successors.empty())   cmds[i]->precedes(exit);
    }
    return cg;
}

/* ------------------------------------------------------------------------- *
 * Snapshot of the graph, taken *before* the pass. The pass rewires edges and
 * overwrites iterator_index, so every check below refers to this snapshot.
 * ------------------------------------------------------------------------- */

struct snapshot_t
{
    std::vector<command_graph_node_t *>                  node;
    std::unordered_map<command_graph_node_t *, int>      index;
    std::vector<device_unique_id_t>                      dev;
    std::vector<std::vector<int>>                        pred, succ;
    bitset2d_t<uint64_t, command_graph_node_index_t> *   reach;

    snapshot_t(command_graph_t * cg) : reach(NULL)
    {
        constexpr bool include_entry_exit = true;
        auto it = cg->create_node_iterators<include_entry_exit>();
        const int n = (int) it.size();

        node.resize(n); dev.resize(n); pred.resize(n); succ.resize(n);
        for (int i = 0 ; i < n ; ++i)
        {
            node[i] = it[i].node;
            index[it[i].node] = i;
            dev[i] = it[i].node->device_unique_id;
        }
        for (int i = 0 ; i < n ; ++i)
        {
            for (command_graph_node_t * u : node[i]->predecessors) pred[i].push_back(index[u]);
            for (command_graph_node_t * w : node[i]->successors)   succ[i].push_back(index[w]);
        }

        reach = new bitset2d_t<uint64_t, command_graph_node_index_t>(n);
        cg->walk<COMMAND_GRAPH_WALK_SEARCH_DFS, COMMAND_GRAPH_WALK_ORDER_POST>(
            [&] (command_graph_node_t * u)
            {
                const int i = index[u];
                reach->set(i, i);
                for (command_graph_node_t * s : u->successors)
                    reach->or_rows(i, index[s]);
            }
        );
    }
    ~snapshot_t(void) { delete reach; }

    /* a < b in the original graph */
    bool precedes(int a, int b) const { return reach->test(a, b); }

    /* Is `S` admissible? `in_s` must be a membership predicate over indices. */
    template<typename PRED>
    bool admissible(const std::vector<int> & S, PRED in_s) const
    {
        if (S.empty()) return true;
        for (int v : S)
        {
            for (int u : pred[v])
                if (!in_s(u))
                    for (int m : S)
                        if (!precedes(u, m)) return false;
            for (int w : succ[v])
                if (!in_s(w))
                    for (int m : S)
                        if (!precedes(m, w)) return false;
        }
        return true;
    }
};

/* ------------------------------------------------------------------------- *
 * Result of one pass run, expressed in snapshot indices
 * ------------------------------------------------------------------------- */

struct result_t
{
    std::vector<std::vector<int>> island;   /* packed islands (>= 2 commands) */
    std::vector<int>              unpacked; /* commands left at the top level */
    std::vector<int>              owner;    /* index -> island, or -1         */
    bool                          acyclic;
};

/* Collect the members of a packed sub-graph. Materialization reuses the very
 * same node objects, so a member is any inner node present in the snapshot;
 * the sub-graph entry/exit are freshly allocated and therefore absent. */
static void
collect_members(command_graph_t * sub, const snapshot_t & snap, std::vector<int> & out)
{
    sub->walk([&](command_graph_node_t * u)
    {
        auto it = snap.index.find(u);
        if (it != snap.index.end())
            out.push_back(it->second);
    });
}

/* Kahn on the *packed* top-level graph. coherence_checks() does not test
 * acyclicity, and it is precisely what a non-parallelism-preserving packing
 * breaks, so it is checked explicitly. */
static bool
top_level_acyclic(command_graph_t * cg)
{
    constexpr bool include_entry_exit = true;
    auto it = cg->create_node_iterators<include_entry_exit>();
    const int n = (int) it.size();

    std::vector<int> indeg((size_t) n, 0);
    for (int i = 0 ; i < n ; ++i)
        indeg[i] = (int) it[i].node->predecessors.size();

    std::vector<int> stack;
    for (int i = 0 ; i < n ; ++i)
        if (indeg[i] == 0) stack.push_back(i);

    int seen = 0;
    while (!stack.empty())
    {
        const int i = stack.back(); stack.pop_back();
        ++seen;
        for (command_graph_node_t * s : it[i].node->successors)
        {
            const int j = (int) s->iterator_index;
            if (--indeg[j] == 0) stack.push_back(j);
        }
    }
    return seen == n;
}

static void
collect_result(command_graph_t * cg, const snapshot_t & snap, result_t & r)
{
    r.owner.assign(snap.node.size(), -1);
    r.acyclic = top_level_acyclic(cg);

    cg->walk([&](command_graph_node_t * u)
    {
        if (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND &&
            u->command && u->command->type == COMMAND_TYPE_BATCH)
        {
            assert(u->command->batch.cg);
            std::vector<int> members;
            collect_members(u->command->batch.cg, snap, members);
            const int id = (int) r.island.size();
            for (int m : members) r.owner[m] = id;
            r.island.push_back(members);
        }
        else
        {
            auto it = snap.index.find(u);
            if (it != snap.index.end() && u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
                r.unpacked.push_back(it->second);
        }
    });
}

/* ------------------------------------------------------------------------- *
 * Invariants
 * ------------------------------------------------------------------------- */

struct stats_t
{
    long graphs = 0, islands = 0, commands = 0, packed = 0;
    long skippable = 0;                 /* graphs where reachability is not needed */
    long exh_checked = 0, exh_maximal = 0;
    long opt_alg = 0, opt_best = 0, opt_graphs = 0;
    long biggest = 0;
};

static int
check_one(command_graph_t * cg, const snapshot_t & snap,
          const std::vector<command_graph_node_t *> & cmds,
          uint64_t seed, stats_t & st)
{
    cg->coherence_checks();

    result_t r;
    collect_result(cg, snap, r);

    /* (1) the packed graph is still a DAG */
    if (!r.acyclic)
    {
        fprintf(stderr, "FAIL [seed %llu]: packed graph has a cycle\n",
                (unsigned long long) seed);
        return 1;
    }

    /* (2) conservation: every command exactly once */
    {
        std::vector<int> seen(snap.node.size(), 0);
        for (auto & S : r.island) for (int m : S) ++seen[m];
        for (int m : r.unpacked)                  ++seen[m];
        for (command_graph_node_t * c : cmds)
        {
            const int i = snap.index.at(c);
            if (seen[i] != 1)
            {
                fprintf(stderr, "FAIL [seed %llu]: command %d appears %d times after the pass\n",
                        (unsigned long long) seed, i, seen[i]);
                return 1;
            }
        }
    }

    for (size_t id = 0 ; id < r.island.size() ; ++id)
    {
        const std::vector<int> & S = r.island[id];

        /* (3) monochromatic */
        for (int m : S)
            if (snap.dev[m] != snap.dev[S[0]])
            {
                fprintf(stderr, "FAIL [seed %llu]: island %zu mixes devices %u and %u\n",
                        (unsigned long long) seed, id, snap.dev[S[0]], snap.dev[m]);
                return 1;
            }

        /* (4) admissible, i.e. the packing is parallelism-preserving */
        const int iid = (int) id;
        if (!snap.admissible(S, [&](int x) { return r.owner[x] == iid; }))
        {
            fprintf(stderr, "FAIL [seed %llu]: island %zu is not parallelism-preserving\n",
                    (unsigned long long) seed, id);
            return 1;
        }

        /* (5) saturated: no unpacked command could have been added */
        for (int v : r.unpacked)
        {
            if (snap.dev[v] != snap.dev[S[0]]) continue;
            std::vector<int> S2 = S; S2.push_back(v);
            if (snap.admissible(S2, [&](int x) { return r.owner[x] == iid || x == v; }))
            {
                fprintf(stderr, "FAIL [seed %llu]: island %zu is not saturated, command %d fits\n",
                        (unsigned long long) seed, id, v);
                return 1;
            }
        }

        st.islands += 1;
        st.packed  += (long) S.size();
        if ((long) S.size() > st.biggest) st.biggest = (long) S.size();
    }

    st.graphs   += 1;
    st.commands += (long) cmds.size();
    return 0;
}

/* ------------------------------------------------------------------------- *
 * Quality metrics (exponential; small graphs only; reported, never asserted)
 * ------------------------------------------------------------------------- */

/* Is some strict superset of `S` (drawn from the same device) admissible? If so
 * `S` is not inclusion-maximal. Enumerates subsets of the other same-device
 * commands, hence the size guard. */
static bool
has_admissible_superset(const snapshot_t & snap, const std::vector<int> & S,
                        const std::vector<int> & others)
{
    const size_t k = others.size();
    for (uint32_t mask = 1 ; mask < (1u << k) ; ++mask)
    {
        std::vector<int> T = S;
        for (size_t b = 0 ; b < k ; ++b)
            if (mask & (1u << b)) T.push_back(others[b]);

        std::set<int> in(T.begin(), T.end());
        if (snap.admissible(T, [&](int x) { return in.count(x) != 0; }))
            return true;
    }
    return false;
}

/* Largest number of commands an exhaustive search can pack: enumerate every
 * admissible same-device subset, then take the best disjoint family. */
static long
exhaustive_optimum(const snapshot_t & snap, const std::vector<int> & cmds)
{
    std::vector<std::vector<int>> adm;
    const size_t k = cmds.size();
    for (uint32_t mask = 0 ; mask < (1u << k) ; ++mask)
    {
        std::vector<int> S;
        for (size_t b = 0 ; b < k ; ++b)
            if (mask & (1u << b)) S.push_back(cmds[b]);
        if (S.size() < 2) continue;

        bool mono = true;
        for (int m : S) if (snap.dev[m] != snap.dev[S[0]]) { mono = false; break; }
        if (!mono) continue;

        std::set<int> in(S.begin(), S.end());
        if (snap.admissible(S, [&](int x) { return in.count(x) != 0; }))
            adm.push_back(S);
    }

    long best = 0;
    std::vector<char> used(snap.node.size(), 0);
    /* depth-first over the admissible sets, keeping the disjoint ones */
    struct rec_t
    {
        const std::vector<std::vector<int>> & adm;
        std::vector<char> & used;
        long & best;
        void go(size_t i, long cov)
        {
            if (cov > best) best = cov;
            for (size_t j = i ; j < adm.size() ; ++j)
            {
                bool free_ = true;
                for (int m : adm[j]) if (used[m]) { free_ = false; break; }
                if (!free_) continue;
                for (int m : adm[j]) used[m] = 1;
                go(j + 1, cov + (long) adm[j].size());
                for (int m : adm[j]) used[m] = 0;
            }
        }
    } rec { adm, used, best };
    rec.go(0, 0);
    return best;
}

/* ------------------------------------------------------------------------- *
 * Driver
 * ------------------------------------------------------------------------- */

int
main(int argc, char ** argv)
{
    const long   ngraphs   = (argc > 1) ? strtol(argv[1], NULL, 10) : 1000;
    const int    min_nodes = (argc > 2) ? (int) strtol(argv[2], NULL, 10) : 10;
    const int    max_nodes = (argc > 3) ? (int) strtol(argv[3], NULL, 10) : 100;
    const uint64_t base    = (argc > 4) ? strtoull(argv[4], NULL, 10) : 20260101ull;

    if (min_nodes < 1 || max_nodes < min_nodes)
    {
        fprintf(stderr, "usage: %s [ngraphs] [min_nodes] [max_nodes] [seed]\n", argv[0]);
        return 2;
    }

    fprintf(stdout, "batch-random: %ld graphs, %d-%d commands, base seed %llu\n",
            ngraphs, min_nodes, max_nodes, (unsigned long long) base);

    stats_t st;
    int rc = 0;

    for (long g = 0 ; g < ngraphs && rc == 0 ; ++g)
    {
        const uint64_t seed = base + (uint64_t) g;
        rng_t rng(seed);

        const int    nv   = min_nodes + (int) rng.below((uint32_t) (max_nodes - min_nodes + 1));
        const int    ndev = 1 + (int) rng.below(4);
        const double p    = 0.05 + 0.45 * ((double) rng.below(1000) / 1000.0);

        std::vector<command_graph_node_t *> cmds;
        command_graph_t * cg = make_random_graph(rng, nv, ndev, p, cmds);

        snapshot_t snap(cg);

        /* does this graph even need the precedence relation? mirrors the
         * pre-scan in the pass: it is needed only when some island has a
         * boundary node other than the entry/exit, or a command stays unpacked */
        const bool single_device = (ndev == 1);
        if (single_device) st.skippable += 1;

        std::vector<int> cmd_idx;
        for (command_graph_node_t * c : cmds) cmd_idx.push_back(snap.index.at(c));

        cg->optimize(COMMAND_GRAPH_PASS_BATCH);

        rc |= check_one(cg, snap, cmds, seed, st);
        if (rc) break;

        /* quality metrics, small graphs only */
        if (nv <= EXHAUSTIVE_MAX_NODES)
        {
            result_t r;
            collect_result(cg, snap, r);

            for (size_t id = 0 ; id < r.island.size() ; ++id)
            {
                std::vector<int> others;
                for (int v : cmd_idx)
                    if (r.owner[v] != (int) id && snap.dev[v] == snap.dev[r.island[id][0]])
                        others.push_back(v);
                if (others.size() > 20) continue;   /* keep 2^k sane */
                st.exh_checked += 1;
                if (!has_admissible_superset(snap, r.island[id], others))
                    st.exh_maximal += 1;
            }

            long alg = 0;
            for (auto & S : r.island) alg += (long) S.size();
            st.opt_alg    += alg;
            st.opt_best   += exhaustive_optimum(snap, cmd_idx);
            st.opt_graphs += 1;
        }

        /* the test allocators are raw `new` and nothing owns the nodes, so the
         * graph is intentionally leaked, as in the other tests */
    }

    if (rc)
    {
        fprintf(stderr, "batch-random: FAILED\n");
        return rc;
    }

    fprintf(stdout, "  graphs                : %ld\n", st.graphs);
    fprintf(stdout, "  commands              : %ld\n", st.commands);
    fprintf(stdout, "  islands               : %ld (largest %ld)\n", st.islands, st.biggest);
    fprintf(stdout, "  commands packed       : %ld (%.1f%%)\n",
            st.packed, st.commands ? 100.0 * (double) st.packed / (double) st.commands : 0.0);
    fprintf(stdout, "  single-device graphs  : %ld (%.1f%%, precedence relation not needed)\n",
            st.skippable, st.graphs ? 100.0 * (double) st.skippable / (double) st.graphs : 0.0);

    if (st.exh_checked)
        fprintf(stdout, "  inclusion-maximal     : %ld/%ld (%.1f%%)\n",
                st.exh_maximal, st.exh_checked,
                100.0 * (double) st.exh_maximal / (double) st.exh_checked);
    else
        fprintf(stdout, "  inclusion-maximal     : skipped (graphs above %d commands)\n",
                EXHAUSTIVE_MAX_NODES);

    if (st.opt_graphs && st.opt_best)
        fprintf(stdout, "  vs exhaustive optimum : %ld/%ld (%.1f%%) over %ld graphs\n",
                st.opt_alg, st.opt_best,
                100.0 * (double) st.opt_alg / (double) st.opt_best, st.opt_graphs);
    else
        fprintf(stdout, "  vs exhaustive optimum : skipped (graphs above %d commands)\n",
                EXHAUSTIVE_MAX_NODES);

    fprintf(stdout, "PASS [batch-random]: all invariants hold on %ld graphs\n", st.graphs);
    return 0;
}
