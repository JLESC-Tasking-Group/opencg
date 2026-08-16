# Optimization Passes {#optimization_passes}

CGIR includes several optimization passes that transform command graphs to reduce overhead and improve execution efficiency.

All passes are enumerated by `CGIR_FORALL_COMMAND_GRAPH_PASS` and can be invoked via:

```cpp
cg.optimize(COMMAND_GRAPH_PASS_BATCH);
// or call the method directly:
cg.pass_batch();
```

## Available Passes

### Transitive Edge Reduction (reduce-edge)

Removes redundant edges from the graph. If node `v` is reachable from `u` through an intermediate path via `w`, the direct edge `u -> v` is redundant and can be removed.

The implementation builds a reachability matrix using `cgir::bitset2d_t` via backward BFS, then prunes edges where the reachability test succeeds through alternative paths. This minimizes the graph complexity while preserving the execution order.

### Node Reduction (reduce-node)

Removes control nodes (nodes without commands) when doing so reduces graph complexity. A control node with `m` predecessors and `n` successors is removed if:

```
m * n < 1 + m + n
```

This heuristic ensures that removing the node (and connecting all predecessors to all successors) does not increase the total number of edges.

### Batching (batch)

Groups **islands** of same-device nodes into `COMMAND_TYPE_BATCH` nodes. An island
is a connected region of nodes on one device, found by a **flood-fill**
(union-find):
- **Edges**: two same-device nodes joined by an edge are in the same island.
- **False twins**: two same-device islands with the same neighborhood (identical
  predecessor and successor islands) are folded together — this catches
  independent same-device ops that share their dependencies.

Each island with at least two `COMMAND` members is materialized **in place** into a
single fresh top-level BATCH node whose `cgir::command_batch_t` sub-graph reuses the
very same node objects and their internal edges. Only the island boundary is rewired:
external edges are moved to/from the batch node (deduplicated; cross-island edges
become batch-to-batch edges), and the sub-graph's entry/exit are connected to the
island's sources/sinks. The sub-graph's `is_sequence` flag is set when the island is a
linear chain of task-spawn programs.

This enables mapping to vendor-specific batching mechanisms (e.g. CUDA Graphs, HIP Graphs, Level Zero command lists).

### Copy Fusion (copy-fuse)

Merges contiguous or overlapping 1D memory copies between the same pair of devices.
False-twin copy nodes whose address ranges are adjacent or overlapping are fused into a single, larger copy command. This reduces the number of individual copy operations submitted to the driver.

2D copy fusion is planned but not yet implemented.

### Copy Normalization (copy-normalize)

Normalizes copy commands:
- Converts 2D copies to 1D when `src_ld == m` and `dst_ld == m` (i.e., the matrix is contiguous)
- Normalizes remaining 2D copies to `sizeof_type = 1` (byte-level addressing)

### Program Fusion (prog-fuse)

Planned but not yet implemented. Will merge compatible kernel launches.

## Running Multiple Passes

Multiple passes are applied in a single call by combining their `COMMAND_GRAPH_PASS_*_BIT` constants into a `command_graph_pass_set_t`:

```cpp
cg.optimize(
      COMMAND_GRAPH_PASS_REDUCE_NODE_BIT
    | COMMAND_GRAPH_PASS_REDUCE_EDGE_BIT
    | COMMAND_GRAPH_PASS_BATCH_BIT);
```

Enabled passes always run in a fixed *canonical order* — the order in which they are declared in `CGIR_FORALL_COMMAND_GRAPH_PASS` (`copy-normalize`, `copy-fuse`, `reduce-node`, `reduce-edge`, `prog-fuse`, `batch`) — regardless of the order of the bits. This guarantees the reduction passes run before batching, so the graph is simplified before it is contracted.

A single pass can still be run on its own with the `command_graph_pass_t` overload:

```cpp
cg.optimize(COMMAND_GRAPH_PASS_REDUCE_NODE);
```
