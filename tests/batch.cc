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
 *  Tests for the `batch` pass (command_graph_t::pass_batch).
 *
 *  The pass groups nodes that share the SAME device into a single
 *  COMMAND_TYPE_BATCH command whose `batch.cg` sub-graph holds the grouped
 *  commands. Two contraction shapes are exercised (plus their combination):
 *
 *    1. sequence    : entry -> u -> v -> exit               (u->v batched)
 *    2. false-twins : entry -> {u, v} -> exit               (u,v independent)
 *    3. mixed       : entry -> u -> v -> exit, entry -> w -> exit
 *                     (u->v sequence, then w joins as a false twin)
 *
 *  Each case must collapse to exactly ONE top-level BATCH node whose inner
 *  graph contains all the original commands.
 */

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

/* Device shared by every batchable leaf command. The batch pass groups by node
 * device id, and entry/exit use CGIR_UNSPECIFIED_DEVICE_UNIQUE_ID, so giving the
 * leaves a real (identical) device id is what makes them eligible to batch. */
static constexpr device_unique_id_t batch_device = 1;

/* Allocate a fresh command + command node on `batch_device`. The concrete
 * command type is irrelevant to the batch pass; a cheap 1D copy keeps the node
 * well-formed (e.g. for dumping). */
static command_graph_node_t *
make_command_node(command_graph_t * cg)
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
        if (node != cg->node_get_entry() && node != cg->node_get_exit())
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

/* Verify that `cg` collapsed to a single top-level BATCH node whose inner graph
 * holds exactly `expected_inner` commands. Returns 0 on success, 1 on failure. */
static int
check_batch(command_graph_t * cg, const char * name, size_t expected_inner)
{
    cg->coherence_checks();

    size_t top = count_nodes(cg);
    if (top != 1)
    {
        fprintf(stderr, "FAIL [%s]: expected 1 top-level node after batch, got %zu\n", name, top);
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

/*
 *  Sequence batching: entry -> u -> v -> exit, u and v on the same device.
 *  u->v is a sequence, so the pass groups them into one batch.
 */
static int
test_batch_sequence(void)
{
    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * u = make_command_node(cg);
    command_graph_node_t * v = make_command_node(cg);

    /* entry -> u -> v -> exit */
    entry->precedes(u);
    u->precedes(v);
    v->precedes(exit);

    cg->dump("cg-pre-batch-sequence.dot");
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);
    cg->dump("cg-post-batch-sequence.dot");

    return check_batch(cg, "sequence", 2);
}

/*
 *  False-twin batching: entry -> u -> exit and entry -> v -> exit, with u and v
 *  on the same device. u and v are independent but share the same neighborhood
 *  (false twins), so the pass groups them into one batch.
 */
static int
test_batch_false_twins(void)
{
    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * u = make_command_node(cg);
    command_graph_node_t * v = make_command_node(cg);

    /* entry -> u -> exit and entry -> v -> exit (u, v independent) */
    entry->precedes(u);
    u->precedes(exit);
    entry->precedes(v);
    v->precedes(exit);

    cg->dump("cg-pre-batch-false-twins.dot");
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);
    cg->dump("cg-post-batch-false-twins.dot");

    return check_batch(cg, "false-twins", 2);
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
    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * u = make_command_node(cg);
    command_graph_node_t * v = make_command_node(cg);
    command_graph_node_t * w = make_command_node(cg);

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

    return check_batch(cg, "mixed", 3);
}

int
main(void)
{
    int rc = 0;
    rc |= test_batch_sequence();
    rc |= test_batch_false_twins();
    rc |= test_batch_mixed();
    return rc;
}
