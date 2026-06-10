# Command Graphs {#command_graphs}

A command graph (`ocg::command_graph_t`) is a directed acyclic graph (DAG) where nodes represent commands and edges represent precedence constraints.

## Graph Structure

Every command graph has:
- An **entry** node (no predecessors)
- An **exit** node (no successors)
- Interior nodes, each with at least one predecessor and one successor

The entry and exit nodes are control nodes (their `command` pointer is `NULL`), automatically created during `ocg::command_graph_t::init()`.

## Nodes

A node (`ocg::command_graph_node_t`) holds:
- A pointer to a `ocg::command_t` (or `NULL` for control nodes)
- A `device_unique_id_t` identifying which device executes the command
- Lists of **predecessors** and **successors** (as `std::list<command_graph_node_t *>`)

### Adding Edges

```cpp
// Make node A precede node B (A -> B)
nodeA->precedes(nodeB);

// Equivalently, make node B succeed node A
nodeB->succeed(nodeA);
```

## Allocators

The graph uses three user-provided allocators, set via `ocg::command_graph_t::init()`:
- `command_allocator_t` -- allocates `ocg::command_t`
- `command_graph_node_allocator_t` -- allocates `ocg::command_graph_node_t`
- `command_graph_allocator_t` -- allocates `ocg::command_graph_t` (for sub-graphs in batches)

This design lets the embedding runtime (e.g. XKRT) manage memory using its own allocators.

## Graph Traversal

The graph supports DFS and BFS walks in both forward and backward directions:

```cpp
// Forward DFS walk (default)
cg.walk([](command_graph_node_t * node) {
    // process node
});

// Backward BFS walk
cg.walk<COMMAND_GRAPH_WALK_DIRECTION_BACKWARD,
        COMMAND_GRAPH_WALK_SEARCH_BFS>(
    [](command_graph_node_t * node) {
        // process node
    }
);
```

Walk IDs are used to track visited nodes, avoiding the need to reset visit flags between walks.

## Node Iterators

For passes that need random access to nodes, use `ocg::command_graph_t::create_node_iterators()`:

```cpp
auto iters = cg.create_node_iterators<MyData>();
for (auto & it : iters) {
    it.node;  // the command_graph_node_t*
    it.data;  // per-node storage of type MyData
}
```

## Node Contraction

The `ocg::command_graph_t::contract()` method merges two nodes into one, reconnecting edges.
Contraction hints (`ocg::command_graph_contraction_hint_t`) accelerate the operation for known graph patterns:
- `COMMAND_GRAPH_CONTRACTION_HINT_FALSE_TWINS` -- u and v share the same predecessors and successors
- `COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE` -- u -> v where u has one successor and v has one predecessor
- `COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE` -- v -> u (reverse)
- `COMMAND_GRAPH_CONTRACTION_HINT_INPLACE` -- reuse u's node instead of allocating a new one

## Graph Utilities

- `ocg::command_graph_t::are_false_twins(u, v)` -- checks if two nodes share the same neighborhood
- `ocg::command_graph_t::are_sequence(u, v)` -- checks if u -> v is a simple sequence
- `ocg::command_graph_t::remove(u)` -- removes a node and reconnects its predecessors to its successors
- `ocg::command_graph_t::coherence_checks()` -- validates graph invariants
- `ocg::command_graph_t::dump(file)` -- outputs the graph in Graphviz DOT format
