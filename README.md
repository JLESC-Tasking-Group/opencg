[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://JLESC-Tasking-Group.github.io/cgir/)

# CGIR - Open Command Graphs

CGIR is a library that defines **commands graph** (**CG**) - a **Vendor-agnostic** and **Multi-devices** Intermediate Representation (IR) for programming [**Command Processor**](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/command-processor.html
) available on modern GPUs. CGIR provides **Optimization passes** of its IR.

In a command graph
- Nodes hold a **device unique identifier** and are either
  - a **command** (e.g., kernel launch, H2D copy, etc.)
  - a **command graph** (i.e., the structure is recursive)
  - a **condition** (demuxer or loop, to conditionally execute associated commands)
- Edges represent **precedence constraints** for execution

A few examples of **commands**:
- 1-dimensional copy: copy(src, dst, ptr, size)
- 2-dimensional copy(m, n, src, src_ld, dst, dst_ld)
- etc. See [include/cgir/command.hpp](include/cgir/command.hpp) for available commands  

A few examples of **optimization passes**:
- Transitive reduction, to minimize the graph complexity
- Batching, typically to minimize overheads by mapping to a vendor-specific batching structure (CUDA/HIP graphs, Level Zero command queues, etc.)
- etc. See [include/cgir/command-graph-pass.hpp](include/cgir/command-graph-pass.hpp) for available passes

# Installation
Using simple `cmake`
```
mkdir build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make
```

# Runtime Configuration
You may set the following environment variables
- `CGIR_OPTIMIZER` as `mlir` or `pod` to switch the optimizer representation.
- `CGIR_JIT_DUMP` to dump the LLVM IR before and after JIT passes.
- `CGIR_PROG_FUSE_DUMP` to dump the LLVM IR of each prog-fusion for debugging:
  the input programs (`input-<i>.ll`), the merged module before optimization
  (`merged.ll`) and the fused/optimized result (`fused.ll`). Set it to any
  non-empty value other than `0` to write under `~/.cgir/tmp/prog-fuse-<seq>/`,
  or to an absolute path to override the output directory.
- `CGIR_OPTIMIZE_DUMP` to dump the command graph as Graphviz `.dot` before and
  after every optimization pass (works for both the legacy and the `mlir`
  optimizer). Each pass writes `optimize-<seq>-<pass>-before.dot` and
  `optimize-<seq>-<pass>-after.dot`, where `<seq>` increments per pass so the
  files form the timeline of an `optimize()` call. Set it to any non-empty value
  other than `0` to write under `~/.cgir/tmp/`, or to an absolute path to
  override the output directory. Render a dump with e.g.
  `dot -Tpng optimize-0-copy-fuse-after.dot -o after.png`.
- `CGIR_STATS_CSV` to append one machine-readable CSV row per optimization pass,
  recording the pass wall-clock time and the command-graph node/edge and
  per-command-type counts *before* and *after* the pass (nodes, edges, empty
  control nodes, commands, PROG, 1D/2D copies, batches). Set it to a file path;
  rows accumulate across passes and processes. This is the data source for the
  evaluation harness (`bench/`). Independent of `CGIR_OPTIMIZE_DUMP`.
- `CGIR_STATS_TAG` an arbitrary string written verbatim in the `tag` column of
  every `CGIR_STATS_CSV` row, so a caller (e.g. the benchmark runner) can join
  the per-pass stats back to a specific run.


# Example
CGIR is integrated into the [XKRT](https://github.com/anlsys/xkrt) runtime system.
It serves as its abstraction for representing commands, notably to record and replay.
The [XKOMP](https://github.com/anlsys/xkomp) support for the `taskgraph` construct eventually fallbacks to CGIR.
See the glue here:
- The [graph instanciation](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L97-L101) for a task dependence graph IR.
- The [replay executionner](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L418)
- The [CUDA driver](https://github.com/anlsys/xkrt/blob/master/src/driver/driver_cu.cc#L618) to execute command graphs
