# Integration {#integration}

CGIR is designed as a building block for runtime systems that manage heterogeneous computing on multi-device architectures.

## XKRT

CGIR is integrated into the [XKRT](https://github.com/anlsys/xkrt) runtime system, where it serves as the command graph abstraction for record-and-replay execution:

- The [record/replay executioner](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L418) records device commands into an CGIR command graph
- The [CUDA driver](https://github.com/anlsys/xkrt/blob/master/src/driver/driver_cu.cc#L618) executes optimized command graphs on NVIDIA GPUs

## XKOMP

The [XKOMP](https://github.com/anlsys/xkomp) compiler provides support for the `taskgraph` construct, which eventually maps to CGIR command graphs for execution via XKRT.

## Custom Integration

To integrate CGIR into your own runtime:

1. **Provide allocators**: implement `command_allocator_t`, `command_graph_node_allocator_t`, and `command_graph_allocator_t` for your memory management scheme
2. **Build the graph**: create nodes with commands and connect them with precedence edges
3. **Optimize**: run the desired optimization passes
4. **Execute**: traverse the optimized graph and dispatch commands to the target device APIs

### Example

See `tests/h2d-distribute.cc` for a complete example that:
1. Creates a command graph distributing a host buffer to 4 devices
2. Runs `reduce-node`, `reduce-edge`, and `batch` passes
3. Dumps the graph at each stage in Graphviz DOT format

## CMake Integration

CGIR installs CMake package configuration files. In your project:

```cmake
find_package(CGIR REQUIRED)
target_link_libraries(my_target CGIR::cgir)
```
