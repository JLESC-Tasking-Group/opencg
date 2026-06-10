# OpenCG - Open Command Graphs

OpenCG is a library to manipulate directed acyclic graph **command graphs** (**CG**) that is **Vendor-agnostic**, supports **Multi-devices** and includes **Optimization passes**

In a command graph
- Nodes hold a **device unique identifier** and are either
  - a **command** (e.g., kernel launch, H2D copy, etc.)
  - a **command graph** (i.e., the structure is recursive)
  - a **condition** (demuxer or loop, to conditionally execute associated commands)
- Edges represent **precedence constraints** for execution

A few examples of **commands**:
- 1-dimensional copy: copy(src, dst, ptr, size)
- 2-dimensional copy(m, n, src, src_ld, dst, dst_ld)
- etc. See [include/opencg/command.hpp](include/opencg/command.hpp) for available commands  

A few examples of **optimization passes**:
- Transitive reduction, to minimize the graph complexity
- Batching, typically to minimize overheads by mapping to a vendor-specific batching structure (CUDA/HIP graphs, Level Zero command queues, etc.)
- etc. See [include/opencg/command-graph-pass.hpp](include/opencg/command-graph-pass.hpp) for available passes

# Installation
Using simple `cmake`
```
mkdir build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make
```

# Example
OpenCG is integrated into the [XKRT](https://github.com/anlsys/xkrt) runtime system.
It serves as its abstraction for representing commands, notably to record and replay.
The [XKOMP](https://github.com/anlsys/xkomp) support for the `taskgraph` construct eventually fallbacks to OpenCG.
See the glue here:
- The [record/replay executionner](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L418)
- The [CUDA driver](https://github.com/anlsys/xkrt/blob/master/src/driver/driver_cu.cc#L618) to execute command graphs
