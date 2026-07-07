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

/* a prog to be submitted via (cuKernelLaunch, ...) */
struct command_prog_t
{
    /* Program function and arguments */
    union {

        /* Fixed argument sizes */
        struct {
            # define CGIR_CALLBACK_ARGS_MAX 4
            void (*fn)(void * [CGIR_CALLBACK_ARGS_MAX]);
            void * args[CGIR_CALLBACK_ARGS_MAX];
        } fixed;

        /* Program launched over an argument array.  `args` is an array of
         * `n_args` pointers, one per program parameter, each pointing to the
         * parameter's value (args[k] == &value) This is the uniform launch
         * form for host/JIT'd programs (the fused `__fused_wrapper`) and for
         * device kernels (the pointer array is the CUDA/HIP `kernelParams`).
         * */
        struct {
            void (*fn)(void ** args);   /* program entry: fn(args) */
            void ** args;               /* array of pointers, one per parameter */
            size_t  n_args;             /* number of pointers in `args` */

            /* true iff `args` is a heap buffer owned by this prog (e.g. the
             * compacted argument buffer produced by the prog-fuse pass). The
             * owner frees `args` before overwriting/discarding it. Defaults to
             * false (see command_t's constructor) so a caller-owned static or
             * stack buffer is never freed on the first fusion. */
            bool _args_owned;
        } variadic;

    } launcher;

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
            prog.launcher.variadic._args_owned = false;

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
    }
};

CGIR_NAMESPACE_END

#endif /* __CGIR_COMMAND_HPP__ */
