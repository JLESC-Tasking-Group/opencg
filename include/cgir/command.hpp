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

#ifndef __CGIR_COMMAND_HPP__
# define __CGIR_COMMAND_HPP__

# include <cgir/cgir.hpp>
# include <cgir/device-type.hpp>
# include <cgir/namespace.hpp>
# include <cgir/prog-source.h>

# include <stddef.h>

CGIR_NAMESPACE_BEGIN

/* Alias the shared C source types (cgir/prog-source.h) into the cgir:: namespace. */
typedef ::cgir_command_prog_source_t        command_prog_source_t;
typedef ::cgir_command_prog_source_type_t   command_prog_source_type_t;
typedef ::cgir_command_prog_extern_t        command_prog_extern_t;
typedef ::cgir_command_prog_source_proto_t  command_prog_source_proto_t;
typedef ::cgir_command_prog_param_t         command_prog_param_t;
typedef ::cgir_command_prog_param_kind_t    command_prog_param_kind_t;

constexpr command_prog_source_type_t COMMAND_PROG_SOURCE_TYPE_LLVMIR = ::CGIR_COMMAND_PROG_SOURCE_TYPE_LLVMIR;
constexpr command_prog_source_type_t COMMAND_PROG_SOURCE_TYPE_MLIR   = ::CGIR_COMMAND_PROG_SOURCE_TYPE_MLIR;
constexpr command_prog_source_type_t COMMAND_PROG_SOURCE_TYPE_PTX    = ::CGIR_COMMAND_PROG_SOURCE_TYPE_PTX;
constexpr command_prog_source_type_t COMMAND_PROG_SOURCE_TYPE_CL     = ::CGIR_COMMAND_PROG_SOURCE_TYPE_CL;
constexpr command_prog_source_type_t COMMAND_PROG_SOURCE_TYPE_SPIRV  = ::CGIR_COMMAND_PROG_SOURCE_TYPE_SPIRV;

/* How the runtime must invoke a PROG's launcher on execution/replay. This is an
 * attribute of the command (not of its source code), decoupling the *way* a
 * program is launched from the launcher function itself. Keeping it separate is
 * what lets prog-fuse merge programs that must be launched identically (e.g. a
 * chain of OpenMP outlined task bodies): the launcher stays a plain
 * `void(void**)` that fusion can link/inline, while the runtime replays the
 * fused program with the shared launch mode. */
typedef enum   command_prog_launch_mode_t
{
    /* The launcher is invoked directly by the executing thread; the command
     * completes when the launcher returns. */
    CGIR_COMMAND_PROG_LAUNCH_MODE_DIRECT = 0,

    /* The runtime must spawn a task (in the current thread's team) that runs the
     * launcher; the command completes when that task completes. Used for OpenMP
     * outlined task bodies, whose replay must re-spawn a task. */
    CGIR_COMMAND_PROG_LAUNCH_MODE_TASK_SPAWN

}              command_prog_launch_mode_t;

/* Move data between devices */
struct command_copy_1D_t
{
    device_unique_id_t src_device_unique_id;
    device_unique_id_t dst_device_unique_id;
    uintptr_t src_device_addr;
    uintptr_t dst_device_addr;
    size_t size;
};

struct command_copy_2D_t
{
    device_unique_id_t src_device_unique_id;
    device_unique_id_t dst_device_unique_id;
    uintptr_t src_addr;
    size_t src_ld;
    uintptr_t dst_addr;
    size_t dst_ld;
    size_t m;
    size_t n;
    size_t sizeof_type;
};

/* Which launcher variant of a PROG command is active, i.e. how the runtime must
 * call its function pointer. Selects the `launcher` union member of
 * command_prog_t. A recorded OpenMP task starts as KMP (its ahead-of-time
 * routine) and the `jit` pass flips it to VARIADIC once it compiles the body. */
typedef enum   command_prog_function_prototype_t
{
    /* launcher.fixed.fn(args[CGIR_CALLBACK_ARGS_MAX]) — a host callback with a
     * fixed-size inline argument array (the C-API launcher path). */
    CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_FIXED = 0,

    /* launcher.variadic.fn(args) — the uniform program entry `void(void**)` used
     * by JIT'd/fused host programs (the `__fused_wrapper`) and by device kernels
     * (`args` is the CUDA/HIP kernelParams). The argument buffer lives in
     * command_prog_t::args / ::n_args (not in the union). */
    CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_VARIADIC,

    /* launcher.kmp.fn(gtid, task) — an ahead-of-time-compiled OpenMP task body on
     * the standard libomp routine ABI (kmp_int32 (*)(kmp_int32 gtid,
     * kmp_task_t *)). Used to run a recorded task body directly when it has not
     * been JIT-compiled into a VARIADIC program. */
    CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_KMP,

    /* launcher.packed.fn(args, args_size) — a program over a single packed byte
     * buffer (leading by-reference pointers, then inline by-value copies). The
     * buffer is command_prog_t::args and its size command_prog_t::args_size. Used
     * by JIT'd/fused programs compiled in the packed ABI and by device kernels
     * launched via the CUDA/HIP parameter-buffer form. */
    CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_PACKED

}              command_prog_function_prototype_t;

/* a prog to be submitted via (cuKernelLaunch, ...) */
struct command_prog_t
{
    /* which `launcher` member is active / how the runtime must call it */
    command_prog_function_prototype_t prototype;

    /* Program function pointer, in one of the prototypes above. */
    union {

        /* Fixed argument sizes */
        struct {
            # define CGIR_CALLBACK_ARGS_MAX 4
            void (*fn)(void * [CGIR_CALLBACK_ARGS_MAX]);
            void * args[CGIR_CALLBACK_ARGS_MAX];
        } fixed;

        /* Uniform program entry `void(void**)`. Its argument array lives in
         * command_prog_t::args (below), not here. */
        struct {
            void (*fn)(void ** args);   /* program entry: fn(args) */
        } variadic;

        /* Ahead-of-time OpenMP task body on the standard libomp routine ABI:
         * fn(gtid, task). `int` mirrors kmp_int32 and `task` the kmp_task_t*;
         * kept generic so CGIR does not depend on the OpenMP headers. */
        struct {
            int   (*fn)(int gtid, void * task);
            void  * task;
        } kmp;

        /* Program over a packed byte buffer: fn(args, args_size). The buffer is
         * command_prog_t::args (reinterpreted as bytes) and its length
         * command_prog_t::args_size. */
        struct {
            void (*fn)(void * args, size_t args_size);
        } packed;

    } launcher;

    /* Argument array for the VARIADIC launcher: `n_args` pointers, one per
     * program parameter, each pointing to the parameter's value (args[k]==&value).
     * This is the uniform launch buffer for host/JIT'd programs (the fused
     * `__fused_wrapper`) and device kernels (the CUDA/HIP kernelParams). It is
     * kept outside the launcher union so a recorded OpenMP task can hold both
     * these args (for a later JIT and for prog-fuse value-dedup) and its KMP
     * ahead-of-time routine at the same time. */
    void ** args;               /* array of pointers, one per parameter */
    size_t  n_args;             /* number of pointers in `args` */

    /* For the PACKED launcher: byte size of the `args` buffer (which is then a
     * single packed byte block, not a pointer array). Unused by VARIADIC. */
    size_t  args_size;

    /* true iff `args` is a heap buffer owned by this prog (e.g. the compacted
     * argument buffer produced by the prog-fuse pass). The owner frees `args`
     * before overwriting/discarding it. Defaults to false (see command_t's
     * constructor) so a caller-owned static or stack buffer is never freed on the
     * first fusion. */
    bool _args_owned;

    // TODO: shouldnt this bellow be user-defined instead ?

    /* source of the prog - shared (C-compatible) structure, see
     * cgir/prog-source.h */
    command_prog_source_t source;

    /* how the runtime must launch this program (see command_prog_launch_mode_t) */
    command_prog_launch_mode_t launch_mode;

    /* grid parameters */
    struct {
        unsigned int x, y, z;
    } grid;

    /* block dimension */
    struct {
        unsigned int x, y, z;
    } block;
};

/* read/write files */
struct command_file_t
{
    int fd;
    void * buffer;
    size_t size;
    size_t offset;
};

struct command_graph_t;

/* a batch of multiple dependent commands, contracted by a driver into a single
 * opaque executable (e.g. CUgraphExec on CUDA) */
struct command_batch_t
{
    /* the command graph of that batch */
    command_graph_t * cg;

    /* driver specific handle */
    void * driver_handle;

    /* true iff the batch's sub-graph is a linear chain (A -> B -> ... -> Z) of
     * PROG commands whose launch mode is TASK_SPAWN (i.e. a sequence of OpenMP
     * tasks). When set, the runtime may replay the whole batch as a single
     * "super" task instead of one task per command. Set by the batch pass. */
    bool is_sequence;
};

struct command_graph_node_t;

/* a loop in the command graph */
struct command_ctrl_loop_t
{
    /* where to restart */
    command_graph_node_t * to;

    /* condition either to restart or pursue */
    ;   // TODO
};

/* demuxer bitset */
typedef int command_ctrl_demux_bitset_t;

/* Maximum number of commands in a demuxer */
# define CGIR_COMMAND_CTRL_DEMUX_SIZE_MAX (8 * sizeof(command_ctrl_demux_bitset_t))

/* demuxer sizes */
typedef uint8_t command_ctrl_demux_size_t;
static_assert((1LU << (8 * sizeof(command_ctrl_demux_size_t))) - 1 >= CGIR_COMMAND_CTRL_DEMUX_SIZE_MAX);

/* a loop in the command graph */
struct command_ctrl_demux_t
{
    /* list of commands */
    command_t * commands;
    command_ctrl_demux_size_t n;

    /* bitflag of commands to executes */
    command_ctrl_demux_bitset_t bitset;
};

/* commands */
struct command_t
{
    /* the command type */
    command_type_t type;

    /* type-specific info */
    union
    {
        command_prog_t          prog;
        command_copy_1D_t       copy_1D;
        command_copy_2D_t       copy_2D;
        command_file_t          file;
        command_batch_t         batch;
        command_ctrl_loop_t     loop;
        command_ctrl_demux_t    demux;
    };

    command_t(command_type_t type) : type(type)
    {
        /* A program may own heap buffers filled by the prog-fuse pass (the
         * compacted args buffer and the serialized bitcode). Clear the
         * ownership flags so the pass never frees a caller-owned (static or
         * stack) buffer when it overwrites the node on the first fusion; the
         * pass sets them true once it installs its own heap buffers. */
        if (type == COMMAND_TYPE_PROG)
        {
            prog.source.content.llvmir._owned         = false;
            prog.source.content.llvmir.symbol         = nullptr;
            prog.source.content.llvmir.externs        = nullptr;
            prog.source.content.llvmir.externs_count  = 0;
            prog.source.content.llvmir._externs_owned = false;
            prog.source.content.llvmir.triple         = nullptr;
            prog.source.content.llvmir.arch           = nullptr;
            prog.source.content.llvmir.runtime_bc     = nullptr;
            prog.source.content.llvmir.proto          = CGIR_COMMAND_PROG_SOURCE_PROTO_UNPACKED_PARAMS;
            prog.source.content.llvmir.params         = nullptr;
            prog.source.content.llvmir.param_count    = 0;
            prog.source.content.llvmir._params_owned  = false;
            prog._args_owned = false;

            /* Default function prototype: the uniform `void(void**)` variadic
             * launcher. Producers override it (XKOMP's task recorder sets KMP,
             * the C-API launcher path sets FIXED). */
            prog.prototype = CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_VARIADIC;
            prog.args_size = 0;

            /* Default launch mode: the launcher is invoked directly. Producers
             * (e.g. XKOMP's task recorder) overwrite this with TASK_SPAWN when
             * the program must be replayed as a spawned task. */
            prog.launch_mode = CGIR_COMMAND_PROG_LAUNCH_MODE_DIRECT;

            /* Default launch parameters. Producers overwrite these; zeroing them
             * makes the value deterministic so the prog-fuse launch-parameter
             * legality check (command_prog_launch_params_equal) is well-defined
             * even when a producer leaves them unset. */
            prog.grid.x  = prog.grid.y  = prog.grid.z  = 0;
            prog.block.x = prog.block.y = prog.block.z = 0;
        }
        /* A batch defaults to an unset sub-graph and is not a sequence until the
         * batch pass builds it and decides otherwise. Keeps the flag defined for
         * every batch command however it is constructed. */
        else if (type == COMMAND_TYPE_BATCH)
        {
            batch.cg            = nullptr;
            batch.driver_handle = nullptr;
            batch.is_sequence   = false;
        }
    }
};

CGIR_NAMESPACE_END

#endif /* __CGIR_COMMAND_HPP__ */
