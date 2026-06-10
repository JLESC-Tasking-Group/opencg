# Optimization Passes {#optimization_passes}

OpenCG includes several optimization passes that transform command graphs to reduce overhead and improve execution efficiency.

All passes are enumerated by `OCG_FORALL_COMMAND_GRAPH_PASS` and can be invoked via:

```cpp
cg.optimize(COMMAND_GRAPH_PASS_BATCH);
// or call the method directly:
cg.pass_batch();
```

## Available Passes

### Transitive Edge Reduction (`reduce-edge`)

Removes redundant edges from the graph. If node `v` is reachable from `u` through an intermediate path via `w`, the direct edge `u -> v` is redundant and can be removed.

The implementation builds a reachability matrix using `ocg::bitset2d_t` via backward BFS, then prunes edges where the reachability test succeeds through alternative paths. This minimizes the graph complexity while preserving the execution order.

### Node Reduction (`reduce-node`)

Removes control nodes (nodes without commands) when doing so reduces graph complexity. A control node with `m` predecessors and `n` successors is removed if:

```
m * n < 1 + m + n
```

This heuristic ensures that removing the node (and connecting all predecessors to all successors) does not increase the total number of edges.

### Batching (`batch`)

Contracts nodes that execute on the same device into `COMMAND_TYPE_BATCH` nodes.
The pass detects two patterns:
- **False twins**: nodes with identical predecessor and successor sets
- **Sequential pairs**: `u -> v` where `u` has a single successor and `v` has a single predecessor

Contracted nodes become a `ocg::command_batch_t` containing a sub-command-graph.
This enables mapping to vendor-specific batching mechanisms (e.g. CUDA Graphs, HIP Graphs, Level Zero command lists).

### Copy Fusion (`copy-fuse`)

Merges contiguous or overlapping 1D memory copies between the same pair of devices.
False-twin copy nodes whose address ranges are adjacent or overlapping are fused into a single, larger copy command. This reduces the number of individual copy operations submitted to the driver.

2D copy fusion is planned but not yet implemented.

### Copy Normalization (`copy-normalize`)

Normalizes copy commands:
- Converts 2D copies to 1D when `src_ld == m` and `dst_ld == m` (i.e., the matrix is contiguous)
- Normalizes remaining 2D copies to `sizeof_type = 1` (byte-level addressing)

### Program Fusion (`prog-fuse`)

Planned but not yet implemented. Will merge compatible kernel launches.

## Running Multiple Passes

Passes can be composed by calling them sequentially:

```cpp
cg.optimize(COMMAND_GRAPH_PASS_REDUCE_NODE);
cg.optimize(COMMAND_GRAPH_PASS_REDUCE_EDGE);
cg.optimize(COMMAND_GRAPH_PASS_BATCH);
```

The order matters: typically, reduction passes should run before batching to simplify the graph first.
