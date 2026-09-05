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

JIT compilation caching and profiling (the `jit` pass compiles each program's
LLVM IR to a host function or device PTX):
- `CGIR_JIT_DEVICE_NOALIAS` set to any non-empty value other than `0` makes the
  `jit` pass assume the pointer parameters of a device kernel do not overlap.
  This is the one place the device JIT can beat the ahead-of-time toolchain:
  NVPTX only emits `ld.global.nc` (the read-only data path, which matters for an
  indirect gather) for a kernel parameter that is *both* `readonly` and
  `noalias`, and the compiler can never infer the latter because an `omp target`
  does not require its mapped list items to be distinct objects -- whereas a
  runtime that knows which buffers a task touches can. Off by default: it is an
  assumption about the program, not a deduction. (`prog-fuse` already makes the
  same assumption for the pointers it captures into a fused wrapper.)
- `CGIR_JIT_AOT_DEVICE` set to `0` makes the `jit` pass leave alone any device
  program that still carries its ahead-of-time compiled kernel, i.e. recompile
  only what `prog-fuse` synthesized. **On by default** — recompiling is safe and
  pays (Krylov CG on GH200: 1.95 ms ahead-of-time vs 1.82 ms recompiled).
  Set it to `0` on a runtime whose device driver cannot preserve a program's
  occupancy across the substitution (see `command_prog_t::blocks_per_sm`):
  recompiling changes the per-block resources a kernel uses, hence how many
  blocks the hardware co-schedules, and that alone has been measured to cost 6x
  on an unchanged instruction stream.
- `CGIR_JIT_CACHE` gates the JIT **result cache**, which is **on by default**.
  Set it to `0` to disable all caching (in-process and on-disk). The in-process
  cache is content-addressed: task instances of the same construct (identical IR,
  externs, prototype and target) reuse the already-compiled host function pointer
  or emitted device PTX instead of recompiling. A hit is byte-identical to
  recompiling, so it never changes generated code.
- `CGIR_JIT_CACHE_DIR` enables the **persistent on-disk cache** at the given
  directory (opt-in; disabled unless set). It stores compiled host objects
  (`<hash>.o`) and device PTX (`<hash>.ptx`) so a later run skips
  optimization/codegen. Keys fold a toolchain salt (LLVM version, the cgir build
  timestamp, the host CPU, and the device libraries' identity), so entries
  self-invalidate on a cgir rebuild, an LLVM change, or a different CPU/target;
  host objects rebind their external symbols per run (robust to ASLR).
- `CGIR_JIT_TIMING` set to any non-empty value other than `0` to print, at
  process exit, a per-phase wall-clock breakdown of JIT compilation accumulated
  across all compiles (`jit-parse`, `host-optimize`/`host-codegen`/`host-link`,
  device `dev-spmdize`/`dev-link-parse`/`dev-o3`/`dev-ptx-emit`, ...). The buckets
  nest under `jit-total`.
- `CGIR_JIT_CACHE_STATS` set to any non-empty value other than `0` to print, at
  process exit, the JIT result-cache outcomes split host/device: total programs,
  `compiled` (full compiles), `disk-reuse`, `mem-reuse`, and the overall reuse %.
- `CGIR_JIT_STATS_CSV` set to a file path to append, at process exit, one
  machine-readable row with the full JIT breakdown (mirrors `CGIR_STATS_CSV`):
  the `tag` (`CGIR_STATS_TAG`, first column, to join back to a caller's run),
  every timing phase as `<phase>_s`/`<phase>_n` (seconds/calls), and split
  host/device cache counts (`{host,device}_{total,compiled,disk_reuse,mem_reuse}`).
  The header is written when the file is new; rows accumulate across processes.
  Enabling it also turns on the timing/cache collection (without the stderr dump);
  a row is written only for runs that actually JIT-compiled.


# Example
CGIR is integrated into the [XKRT](https://github.com/anlsys/xkrt) runtime system.
It serves as its abstraction for representing commands, notably to record and replay.
The [XKOMP](https://github.com/anlsys/xkomp) support for the `taskgraph` construct eventually fallbacks to CGIR.
See the glue here:
- The [graph instanciation](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L97-L101) for a task dependence graph IR.
- The [replay executionner](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L418)
- The [CUDA driver](https://github.com/anlsys/xkrt/blob/master/src/driver/driver_cu.cc#L618) to execute command graphs
