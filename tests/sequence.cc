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
 *  Tests for the `sequence` pass (command_graph_t::pass_sequence).
 *
 *  The pass collapses a maximal linear chain (u -> v -> ... -> w) of same-device
 *  TASK_SPAWN PROG commands into a single COMMAND_TYPE_BATCH whose sub-graph is
 *  flagged `is_sequence`. Chains are broken by branches (a node with != 1
 *  successor / predecessor) and by device changes.
 */

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

/* Device shared by the task nodes of a chain. */
static constexpr device_unique_id_t task_device = 1;

/* Allocate a TASK_SPAWN PROG node on `dev` (the only nodes the sequence pass
 * considers sequence-able). */
static command_graph_node_t *
make_task_node(command_graph_t * cg, device_unique_id_t dev = task_device)
{
    command_t * cmd = command_new(cg, COMMAND_TYPE_PROG);
    assert(cmd);
    cmd->prog.launch_mode = CGIR_COMMAND_PROG_LAUNCH_MODE_TASK_SPAWN;

    command_graph_node_t * node =
        command_graph_node_new(cg, dev, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(node);
    node->command = cmd;
    return node;
}

/* Allocate a graph and detach the default entry -> exit edge. */
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

/* Count the COMMAND nodes of `cg` (entry/exit included). A sequence sub-graph
 * reuses the chain head/tail as entry/exit, so its commands are NOT all
 * "interior" -- count by type instead of excluding entry/exit. */
static size_t
count_command_nodes(command_graph_t * cg)
{
    size_t n = 0;
    cg->walk([&](command_graph_node_t * node) {
        if (node->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
            ++n;
    });
    return n;
}

/* Count the top-level BATCH nodes of `cg` (and, via `inner`, the size of the
 * unique one's is_sequence sub-graph when there is exactly one). */
static size_t
count_batches(command_graph_t * cg, command_graph_node_t ** the_batch)
{
    command_graph_node_t * found = NULL;
    size_t nbatch = 0;
    cg->walk([&](command_graph_node_t * node) {
        if (node == cg->node_get_entry() || node == cg->node_get_exit())
            return ;
        if (node->type == COMMAND_GRAPH_NODE_TYPE_COMMAND &&
            node->command && node->command->type == COMMAND_TYPE_BATCH)
        {
            found = node;
            ++nbatch;
        }
    });
    if (the_batch)
        *the_batch = (nbatch == 1) ? found : NULL;
    return nbatch;
}

/*
 *  A same-device chain of TASK_SPAWN progs collapses into ONE is_sequence batch.
 *      entry -> a -> b -> c -> exit
 */
static int
test_sequence_chain(void)
{
    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * a = make_task_node(cg);
    command_graph_node_t * b = make_task_node(cg);
    command_graph_node_t * c = make_task_node(cg);

    entry->precedes(a);
    a->precedes(b);
    b->precedes(c);
    c->precedes(exit);

    cg->dump("cg-pre-sequence-chain.dot");
    cg->optimize(COMMAND_GRAPH_PASS_SEQUENCE);
    cg->dump("cg-post-sequence-chain.dot");

    cg->coherence_checks();

    command_graph_node_t * batch = NULL;
    if (count_batches(cg, &batch) != 1 || count_nodes(cg) != 1)
    {
        fprintf(stderr, "FAIL [chain]: expected exactly 1 top-level BATCH\n");
        return 1;
    }
    if (batch->command->batch.cg == NULL || !batch->command->batch.cg->is_sequence)
    {
        fprintf(stderr, "FAIL [chain]: batch sub-graph is not flagged is_sequence\n");
        return 1;
    }
    if (batch->device_unique_id != task_device)
    {
        fprintf(stderr, "FAIL [chain]: batch device %u != %u\n",
                batch->device_unique_id, task_device);
        return 1;
    }
    size_t inner = count_command_nodes(batch->command->batch.cg);
    if (inner != 3)
    {
        fprintf(stderr, "FAIL [chain]: expected 3 commands in the sequence, got %zu\n", inner);
        return 1;
    }

    fprintf(stdout, "PASS [chain]: collapsed to 1 is_sequence BATCH of %zu\n", inner);
    return 0;
}

/*
 *  A branch is NOT a chain: `a` has two successors, so no sequence edge exists
 *  and nothing is batched.
 *      entry -> a -> {b, c} -> exit
 */
static int
test_sequence_not_a_chain(void)
{
    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * a = make_task_node(cg);
    command_graph_node_t * b = make_task_node(cg);
    command_graph_node_t * c = make_task_node(cg);

    entry->precedes(a);
    a->precedes(b);
    a->precedes(c);
    b->precedes(exit);
    c->precedes(exit);

    cg->optimize(COMMAND_GRAPH_PASS_SEQUENCE);
    cg->coherence_checks();

    if (count_batches(cg, NULL) != 0 || count_nodes(cg) != 3)
    {
        fprintf(stderr, "FAIL [not-a-chain]: expected no batch and 3 top-level nodes\n");
        return 1;
    }

    fprintf(stdout, "PASS [not-a-chain]: nothing batched\n");
    return 0;
}

/*
 *  A chain stops at a device change: a(dev1) -> b(dev1) -> c(dev2). Only {a,b}
 *  form a same-device sequence; c stays a separate top-level node.
 */
static int
test_sequence_device_split(void)
{
    command_graph_node_t * entry;
    command_graph_node_t * exit;
    command_graph_t * cg = make_graph(&entry, &exit);

    command_graph_node_t * a = make_task_node(cg, 1);
    command_graph_node_t * b = make_task_node(cg, 1);
    command_graph_node_t * c = make_task_node(cg, 2);

    entry->precedes(a);
    a->precedes(b);
    b->precedes(c);
    c->precedes(exit);

    cg->optimize(COMMAND_GRAPH_PASS_SEQUENCE);
    cg->coherence_checks();

    command_graph_node_t * batch = NULL;
    /* top level: BATCH{a,b} and c -> 2 nodes, 1 batch */
    if (count_batches(cg, &batch) != 1 || count_nodes(cg) != 2)
    {
        fprintf(stderr, "FAIL [device-split]: expected 1 BATCH + 1 node\n");
        return 1;
    }
    if (batch->command->batch.cg == NULL ||
        !batch->command->batch.cg->is_sequence ||
        count_command_nodes(batch->command->batch.cg) != 2)
    {
        fprintf(stderr, "FAIL [device-split]: batch is not an is_sequence of 2\n");
        return 1;
    }

    fprintf(stdout, "PASS [device-split]: chain split at device boundary\n");
    return 0;
}

int
main(void)
{
    int rc = 0;
    rc |= test_sequence_chain();
    rc |= test_sequence_not_a_chain();
    rc |= test_sequence_device_split();
    return rc;
}
