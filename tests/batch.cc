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

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

/* Allocate a fresh command + command node on `batch_device`. The concrete
 * command type is irrelevant to the batch pass; a cheap 1D copy keeps the node
 * well-formed (e.g. for dumping). */
static command_graph_node_t *
make_command_node(command_graph_t * cg, const device_unique_id_t batch_device)
{
    command_t * cmd = command_new(cg, COMMAND_TYPE_COPY_D2D_1D);
    assert(cmd);
    cmd->copy_1D.src_device_unique_id = batch_device;
    cmd->copy_1D.dst_device_unique_id = batch_device;
    cmd->copy_1D.src_device_addr      = 0;
    cmd->copy_1D.dst_device_addr      = 0;
    cmd->copy_1D.size                 = 0;

    command_graph_node_t * node =
        command_graph_node_new(cg, batch_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(node);
    node->command = cmd;
    return node;
}

/* Allocate a graph and detach the default entry -> exit edge so a topology can
 * be built explicitly. */
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

/* Count the nodes of `cg`, excluding its entry/exit control nodes. */
static size_t
count_nodes(command_graph_t * cg)
{
    size_t n = 0;
    cg->walk([&](command_graph_node_t * node) {
        ++n;
    });
    return n;
}

/* Return the unique non-entry/exit node of `cg`, or NULL if there is not exactly
 * one. */
static command_graph_node_t *
single_node(command_graph_t * cg)
{
    command_graph_node_t * found = NULL;
    size_t n = 0;
    cg->walk([&](command_graph_node_t * node) {
        if (node != cg->node_get_entry() && node != cg->node_get_exit())
        {
            found = node;
            ++n;
        }
    });
    return (n == 1) ? found : NULL;
}

/* Verify that `cg` collapsed to `expected_top_level` BATCH nodes whose inner graph
 * holds exactly each `expected_inner` commands. Returns 0 on success, 1 on failure. */
static int
check_batch(
    command_graph_t * cg,
    const char * name,
    size_t expected_top_level,
    size_t expected_inner,
    const device_unique_id_t batch_device
) {
    cg->coherence_checks();

    size_t top = count_nodes(cg);
    if (top != expected_top_level)
    {
        fprintf(stderr, "FAIL [%s]: expected %zu top-level node after batch, got %zu\n", name, expected_top_level, top);
        return 1;
    }

    command_graph_node_t * node = single_node(cg);
    assert(node);

    if (node->type != COMMAND_GRAPH_NODE_TYPE_COMMAND ||
        node->command == NULL ||
        node->command->type != COMMAND_TYPE_BATCH)
    {
        fprintf(stderr, "FAIL [%s]: remaining node is not a BATCH command\n", name);
        return 1;
    }

    if (node->device_unique_id != batch_device)
    {
        fprintf(stderr, "FAIL [%s]: batch node has device %u, expected %u\n",
                name, node->device_unique_id, batch_device);
        return 1;
    }

    command_graph_t * inner = node->command->batch.cg;
    if (inner == NULL)
    {
        fprintf(stderr, "FAIL [%s]: BATCH command has no inner command graph\n", name);
        return 1;
    }

    size_t inner_count = count_nodes(inner);
    if (inner_count != expected_inner)
    {
        fprintf(stderr, "FAIL [%s]: expected %zu commands inside the batch, got %zu\n",
                name, expected_inner, inner_count);
        return 1;
    }

    fprintf(stdout, "PASS [%s]: collapsed to 1 BATCH node containing %zu commands\n",
            name, inner_count);
    return 0;
}

/* Verify that `cg` collapsed to exactly two BATCH nodes, one per device, that
 * each holds `inner` nodes (commands plus the sub-graph entry/exit), and that
 * the two are *concurrent*: neither is a predecessor of the other, and both
 * share the same single predecessor and the same single successor.
 *
 * The last check is the point of the test: a batch that swallows a node whose
 * other predecessors live on the other device makes the two batches sequential,
 * which silently removes the concurrency between the two devices. */
static int
check_two_parallel_batches(
    command_graph_t * cg,
    const char * name,
    size_t expected_top_level,
    const device_unique_id_t dev_a,
    const device_unique_id_t dev_b,
    size_t expected_inner
) {
    cg->coherence_checks();

    size_t top = count_nodes(cg);
    if (top != expected_top_level)
    {
        fprintf(stderr, "FAIL [%s]: expected %zu top-level nodes, got %zu\n", name, expected_top_level, top);
        return 1;
    }

    command_graph_node_t * ba = NULL;
    command_graph_node_t * bb = NULL;
    size_t nbatch = 0;
    cg->walk([&](command_graph_node_t * node) {
        if (node->type == COMMAND_GRAPH_NODE_TYPE_COMMAND &&
            node->command && node->command->type == COMMAND_TYPE_BATCH)
        {
            ++nbatch;
            if (node->device_unique_id == dev_a) ba = node;
            if (node->device_unique_id == dev_b) bb = node;
        }
    });

    if (nbatch != 2 || ba == NULL || bb == NULL)
    {
        fprintf(stderr, "FAIL [%s]: expected 2 BATCH nodes on devices %u and %u, got %zu\n",
                name, dev_a, dev_b, nbatch);
        return 1;
    }

    for (command_graph_node_t * b : { ba, bb })
    {
        if (b->command->batch.cg == NULL)
        {
            fprintf(stderr, "FAIL [%s]: BATCH on device %u has no inner graph\n", name, b->device_unique_id);
            return 1;
        }
        size_t inner = count_nodes(b->command->batch.cg);
        if (inner != expected_inner)
        {
            fprintf(stderr, "FAIL [%s]: BATCH on device %u holds %zu nodes, expected %zu\n",
                    name, b->device_unique_id, inner, expected_inner);
            return 1;
        }
    }

    /* the two batches must not be ordered with respect to each other */
    for (command_graph_node_t * s : ba->successors)
        if (s == bb)
        {
            fprintf(stderr, "FAIL [%s]: batch(dev %u) precedes batch(dev %u): the two devices are serialized\n",
                    name, dev_a, dev_b);
            return 1;
        }
    for (command_graph_node_t * s : bb->successors)
        if (s == ba)
        {
            fprintf(stderr, "FAIL [%s]: batch(dev %u) precedes batch(dev %u): the two devices are serialized\n",
                    name, dev_b, dev_a);
            return 1;
        }

    /* and they must share their boundary: same single predecessor, same single
     * successor. That pins them as two parallel branches of the graph. */
    if (ba->predecessors.size() != 1 || bb->predecessors.size() != 1 ||
        ba->successors.size()   != 1 || bb->successors.size()   != 1 ||
        ba->predecessors.front() != bb->predecessors.front() ||
        ba->successors.front()   != bb->successors.front())
    {
        fprintf(stderr, "FAIL [%s]: the two batches do not share the same boundary\n", name);
        return 1;
    }

    fprintf(stdout, "PASS [%s]: 2 concurrent BATCH nodes (devices %u and %u), %zu nodes each\n",
            name, dev_a, dev_b, expected_inner);
    return 0;
}

/*
 *  Sequence batching: entry -> u -> v -> exit, u and v on the same device.
 *  u->v is a sequence, so the pass groups them into one batch.
 */
static int
test_batch_sequence(void)
{
    constexpr device_unique_id_t device_unique_id = 1;

    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * u = make_command_node(cg, device_unique_id);
    command_graph_node_t * v = make_command_node(cg, device_unique_id);

    /* entry -> u -> v -> exit */
    entry->precedes(u);
    u->precedes(v);
    v->precedes(exit);

    cg->dump("cg-pre-batch-sequence.dot");
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);
    cg->dump("cg-post-batch-sequence.dot");

    constexpr const char * const name     = "sequence";
    constexpr size_t expected_top_level   = 3;
    constexpr size_t expected_inner_level = 4;
    return check_batch(cg, name, expected_top_level, expected_inner_level, device_unique_id);
}

/*
 *  False-twin batching: entry -> u -> exit and entry -> v -> exit, with u and v
 *  on the same device. u and v are independent but share the same neighborhood
 *  (false twins), so the pass groups them into one batch.
 */
static int
test_batch_false_twins(void)
{
    constexpr device_unique_id_t device_unique_id = 1;

    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * u = make_command_node(cg, device_unique_id);
    command_graph_node_t * v = make_command_node(cg, device_unique_id);

    /* entry -> u -> exit and entry -> v -> exit (u, v independent) */
    entry->precedes(u);
    u->precedes(exit);
    entry->precedes(v);
    v->precedes(exit);

    cg->dump("cg-pre-batch-false-twins.dot");
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);
    cg->dump("cg-post-batch-false-twins.dot");

    constexpr const char * const name     = "false-twins";
    constexpr size_t expected_top_level   = 3;
    constexpr size_t expected_inner_level = 4;
    return check_batch(cg, name, expected_top_level, expected_inner_level, device_unique_id);
}

/*
 *  Mixed batching: entry -> u -> v -> exit (a u->v sequence) together with
 *  entry -> w -> exit (w independent from u and v), all on the same device.
 *  The pass first batches the u->v sequence, then folds w into the resulting
 *  batch as a false twin, collapsing everything into one batch of 3 commands.
 */
static int
test_batch_mixed(void)
{
    constexpr device_unique_id_t device_unique_id = 1;

    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * u = make_command_node(cg, device_unique_id);
    command_graph_node_t * v = make_command_node(cg, device_unique_id);
    command_graph_node_t * w = make_command_node(cg, device_unique_id);

    /* entry -> u -> v -> exit  (sequence) */
    entry->precedes(u);
    u->precedes(v);
    v->precedes(exit);

    /* entry -> w -> exit  (independent from u, v) */
    entry->precedes(w);
    w->precedes(exit);

    cg->dump("cg-pre-batch-mixed.dot");
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);
    cg->dump("cg-post-batch-mixed.dot");

    constexpr const char * const name     = "mixed";
    constexpr size_t expected_top_level   = 3;
    constexpr size_t expected_inner_level = 5;
    return check_batch(cg, name, expected_top_level, expected_inner_level, device_unique_id);
}

/*
 *  Island (flood-fill): a branchy, same-device DAG that is neither a plain
 *  sequence nor a set of false twins. Every c* node is edge-connected to the
 *  rest on the same device, so the flood-fill puts them all in one island and
 *  the pass must collapse ALL 12 commands into a single BATCH.
 *
 *  Topology (all c* on batch_device):
 *      entry -> c1, c7, c13
 *      c1 -> c2 ; c2 -> c3 ; c3 -> c4, c6 ; c4 -> exit ; c6 -> exit
 *      c7 -> c8 ; c8 -> c9 ; c9 -> c10, c12
 *      c10 -> c11 ; c11 -> c2 ; c12 -> c6 ; c13 -> c8
 */
static int
test_batch_island(void)
{
    constexpr device_unique_id_t device_unique_id = 1;

    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * c1  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c2  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c3  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c4  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c6  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c7  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c8  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c9  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c10 = make_command_node(cg, device_unique_id);
    command_graph_node_t * c11 = make_command_node(cg, device_unique_id);
    command_graph_node_t * c12 = make_command_node(cg, device_unique_id);
    command_graph_node_t * c13 = make_command_node(cg, device_unique_id);

    entry->precedes(c1);
    entry->precedes(c7);
    entry->precedes(c13);

    c1->precedes(c2);
    c2->precedes(c3);
    c3->precedes(c4);
    c3->precedes(c6);
    c4->precedes(exit);
    c6->precedes(exit);

    c7->precedes(c8);
    c8->precedes(c9);
    c9->precedes(c10);
    c9->precedes(c12);
    c10->precedes(c11);
    c11->precedes(c2);
    c12->precedes(c6);
    c13->precedes(c8);

    cg->dump("cg-pre-batch-island.dot");
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);
    cg->dump("cg-post-batch-island.dot");

    constexpr const char * const name     = "island";
    constexpr size_t expected_top_level   = 3;
    constexpr size_t expected_inner_level = 14;
    return check_batch(cg, name, expected_top_level, expected_inner_level, device_unique_id);
}

/**
 *         s
 *        / \
 *       0   0
 *        \ /
 *         0
 *        / \
 *       0   0
 *        \ /
 *         t
 *
 * should become
 *
 *         s
 *         |
 *       00000
 *         |
 *         t
 */
static int
test_batch_false_twins_sequence(void)
{
    constexpr device_unique_id_t device_unique_id = 1;

    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * c1  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c2  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c3  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c4  = make_command_node(cg, device_unique_id);
    command_graph_node_t * c5  = make_command_node(cg, device_unique_id);

    entry->precedes(c1);
    entry->precedes(c2);

    c1->precedes(c3);
    c2->precedes(c3);

    c3->precedes(c4);
    c3->precedes(c5);

    c4->precedes(exit);
    c5->precedes(exit);

    cg->dump("cg-pre-batch-false-twins-sequence.dot");
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);
    cg->dump("cg-post-batch-false-twins-sequence.dot");

    constexpr const char * const name     = "false-twins-sequence";
    constexpr size_t expected_top_level   = 3;
    constexpr size_t expected_inner_level = 7;
    return check_batch(cg, name, expected_top_level, expected_inner_level, device_unique_id);
}

/**
 *         s
 *       / | \
 *      0  0  1
 *       \ |  | \
 *         0  1 1
 *         | /
 *         0
 *         |
 *         t
 *
 * should become
 *
 *         s
 *        / \
 *      000 111
 *       | /
 *       0
 *       |
 *       t
 */
static int
test_batch_multi_dev(void)
{
    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * c1  = make_command_node(cg, 0);
    command_graph_node_t * c2  = make_command_node(cg, 0);
    command_graph_node_t * c3  = make_command_node(cg, 1);
    command_graph_node_t * c4  = make_command_node(cg, 0);
    command_graph_node_t * c5  = make_command_node(cg, 1);
    command_graph_node_t * c6  = make_command_node(cg, 1);
    command_graph_node_t * c7  = make_command_node(cg, 0);

    entry->precedes(c1);
    entry->precedes(c2);
    entry->precedes(c3);

    c1->precedes(c4);
    c2->precedes(c4);
    c3->precedes(c5);
    c3->precedes(c6);

    c4->precedes(c7);
    c5->precedes(c7);
    c6->precedes(c7);

    c7->precedes(exit);

    cg->dump("cg-pre-batch-multi-dev.dot");
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);
    cg->dump("cg-post-batch-multi-dev.dot");

    /* c7 depends on c5 and c6, which live on device 1: pulling it into device
     * 0's batch would make that batch wait for device 1's, serializing the two.
     * It must therefore stay a top-level command, leaving
     *   entry -> {c1,c2,c4} -> c7 -> exit  and  entry -> {c3,c5,c6} -> c7
     * with the two batches concurrent. */
    constexpr const char * const name   = "multi-dev";
    constexpr size_t expected_top_level = 5;   /* entry, 2 batches, c7, exit */
    constexpr size_t expected_inner     = 5;   /* 3 commands + sub-graph entry/exit */
    return check_two_parallel_batches(cg, name, expected_top_level, 0, 1, expected_inner);
}

int
main(void)
{
    int rc = 0;
    rc |= test_batch_sequence();
    rc |= test_batch_false_twins();
    rc |= test_batch_mixed();
    rc |= test_batch_island();
    rc |= test_batch_false_twins_sequence();
    rc |= test_batch_multi_dev();
    return rc;
}
