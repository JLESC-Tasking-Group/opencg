# Commands {#commands}

Commands represent the individual operations within a command graph.
Each command is stored in the `cgir::command_t` union, tagged by a `cgir::command_type_t` enum value.

## Command Types

All command types are enumerated via the `CGIR_FORALL_COMMAND_TYPE` X-macro.
Use `cgir::command_type_to_str()` to get a human-readable name for any command type.

### Memory Copies

CGIR supports 1D and 2D memory copies between host (H) and device (D) memories:

| Type                           | Struct                       | Description                     |
|--------------------------------|------------------------------|---------------------------------|
| `COMMAND_TYPE_COPY_H2H_1D`    | `cgir::command_copy_1D_t`    | Host-to-Host 1D copy            |
| `COMMAND_TYPE_COPY_H2D_1D`    | `cgir::command_copy_1D_t`    | Host-to-Device 1D copy          |
| `COMMAND_TYPE_COPY_D2H_1D`    | `cgir::command_copy_1D_t`    | Device-to-Host 1D copy          |
| `COMMAND_TYPE_COPY_D2D_1D`    | `cgir::command_copy_1D_t`    | Device-to-Device 1D copy        |
| `COMMAND_TYPE_COPY_H2H_2D`    | `cgir::command_copy_2D_t`    | Host-to-Host 2D copy            |
| `COMMAND_TYPE_COPY_H2D_2D`    | `cgir::command_copy_2D_t`    | Host-to-Device 2D copy          |
| `COMMAND_TYPE_COPY_D2H_2D`    | `cgir::command_copy_2D_t`    | Device-to-Host 2D copy          |
| `COMMAND_TYPE_COPY_D2D_2D`    | `cgir::command_copy_2D_t`    | Device-to-Device 2D copy        |

**1D copies** (`cgir::command_copy_1D_t`) specify source/destination device IDs, addresses, and byte size.

**2D copies** (`cgir::command_copy_2D_t`) additionally specify leading dimensions (`src_ld`, `dst_ld`), a column count `m`, a row count `n`, and `sizeof_type`.

### Kernel Launch

| Type                  | Struct                    | Description           |
|-----------------------|---------------------------|-----------------------|
| `COMMAND_TYPE_PROG`   | `cgir::command_prog_t`    | Program/kernel launch |

A program command holds:
- A **launcher** (either fixed-arity or variadic function pointer + arguments)
- A **source** pointer and **source type** (`LLVMIR`, `MLIR`, `PTX`, `CL`, or `SPIRV`)
- **Grid** and **block** dimensions (3D)

### File I/O

| Type                    | Struct                   | Description          |
|-------------------------|--------------------------|----------------------|
| `COMMAND_TYPE_FD_READ`  | `cgir::command_file_t`   | Read from file       |
| `COMMAND_TYPE_FD_WRITE` | `cgir::command_file_t`   | Write to file        |

### Batch

| Type                  | Struct                    | Description                           |
|-----------------------|---------------------------|---------------------------------------|
| `COMMAND_TYPE_BATCH`  | `cgir::command_batch_t`   | A batch of commands (sub-graph)       |

A batch node contains a pointer to a sub `cgir::command_graph_t` and a driver-specific handle.
Batches are typically created by the [batching optimization pass](@ref optimization_passes).

### Control Flow

| Type                     | Struct                         | Description                      |
|--------------------------|--------------------------------|----------------------------------|
| `COMMAND_TYPE_CTRL_LOOP` | `cgir::command_ctrl_loop_t`    | Loop back to a previous node     |
| `COMMAND_TYPE_CTRL_DEMUX`| `cgir::command_ctrl_demux_t`   | Conditionally execute commands   |

The **demuxer** uses a bitset to select which commands (up to `CGIR_COMMAND_CTRL_DEMUX_SIZE_MAX`) to execute from a list.

## The command_t Union

All command data is stored in `cgir::command_t`, which is a tagged union:

```cpp
struct command_t {
    command_type_t type;
    union {
        command_prog_t       prog;
        command_copy_1D_t    copy_1D;
        command_copy_2D_t    copy_2D;
        command_file_t       file;
        command_batch_t      batch;
        command_ctrl_loop_t  loop;
        command_ctrl_demux_t demux;
    };
};
```
