# Commands {#commands}

Commands represent the individual operations within a command graph.
Each command is stored in the `ocg::command_t` union, tagged by a `ocg::command_type_t` enum value.

## Command Types

All command types are enumerated via the `OCG_FORALL_COMMAND_TYPE` X-macro.
Use `ocg::command_type_to_str()` to get a human-readable name for any command type.

### Memory Copies

OpenCG supports 1D and 2D memory copies between host (H) and device (D) memories:

| Type                           | Struct                       | Description                     |
|--------------------------------|------------------------------|---------------------------------|
| `COMMAND_TYPE_COPY_H2H_1D`    | `ocg::command_copy_1D_t`    | Host-to-Host 1D copy            |
| `COMMAND_TYPE_COPY_H2D_1D`    | `ocg::command_copy_1D_t`    | Host-to-Device 1D copy          |
| `COMMAND_TYPE_COPY_D2H_1D`    | `ocg::command_copy_1D_t`    | Device-to-Host 1D copy          |
| `COMMAND_TYPE_COPY_D2D_1D`    | `ocg::command_copy_1D_t`    | Device-to-Device 1D copy        |
| `COMMAND_TYPE_COPY_H2H_2D`    | `ocg::command_copy_2D_t`    | Host-to-Host 2D copy            |
| `COMMAND_TYPE_COPY_H2D_2D`    | `ocg::command_copy_2D_t`    | Host-to-Device 2D copy          |
| `COMMAND_TYPE_COPY_D2H_2D`    | `ocg::command_copy_2D_t`    | Device-to-Host 2D copy          |
| `COMMAND_TYPE_COPY_D2D_2D`    | `ocg::command_copy_2D_t`    | Device-to-Device 2D copy        |

**1D copies** (`ocg::command_copy_1D_t`) specify source/destination device IDs, addresses, and byte size.

**2D copies** (`ocg::command_copy_2D_t`) additionally specify leading dimensions (`src_ld`, `dst_ld`), a column count `m`, a row count `n`, and `sizeof_type`.

### Kernel Launch

| Type                  | Struct                    | Description           |
|-----------------------|---------------------------|-----------------------|
| `COMMAND_TYPE_PROG`   | `ocg::command_prog_t`    | Program/kernel launch |

A program command holds:
- A **launcher** (either fixed-arity or variadic function pointer + arguments)
- A **source** pointer and **source type** (`LLVMIR`, `MLIR`, `PTX`, `CL`, or `SPIRV`)
- **Grid** and **block** dimensions (3D)

### File I/O

| Type                    | Struct                   | Description          |
|-------------------------|--------------------------|----------------------|
| `COMMAND_TYPE_FD_READ`  | `ocg::command_file_t`   | Read from file       |
| `COMMAND_TYPE_FD_WRITE` | `ocg::command_file_t`   | Write to file        |

### Batch

| Type                  | Struct                    | Description                           |
|-----------------------|---------------------------|---------------------------------------|
| `COMMAND_TYPE_BATCH`  | `ocg::command_batch_t`   | A batch of commands (sub-graph)       |

A batch node contains a pointer to a sub `ocg::command_graph_t` and a driver-specific handle.
Batches are typically created by the [batching optimization pass](@ref optimization_passes).

### Control Flow

| Type                     | Struct                         | Description                      |
|--------------------------|--------------------------------|----------------------------------|
| `COMMAND_TYPE_CTRL_LOOP` | `ocg::command_ctrl_loop_t`    | Loop back to a previous node     |
| `COMMAND_TYPE_CTRL_DEMUX`| `ocg::command_ctrl_demux_t`   | Conditionally execute commands   |

The **demuxer** uses a bitset to select which commands (up to `OCG_COMMAND_CTRL_DEMUX_SIZE_MAX`) to execute from a list.

## The `command_t` Union

All command data is stored in `ocg::command_t`, which is a tagged union:

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
