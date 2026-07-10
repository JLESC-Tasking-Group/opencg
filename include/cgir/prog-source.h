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

/**
 * @file prog-source.h
 * @brief The source code attached to a program (PROG) command.
 *
 * C-compatible (no namespace, POD) so it can be shared between CGIR's
 * `command_prog_t` and external runtimes (e.g. XKRT's task formats) whose
 * headers must remain includable from C. CGIR's C++ code aliases these types
 * into the `cgir::` namespace (see `cgir/command.hpp`).
 */

#ifndef __CGIR_PROG_SOURCE_H__
# define __CGIR_PROG_SOURCE_H__

# include <stddef.h>
# include <stdbool.h>

/* format of a prog source code */
typedef enum    cgir_command_prog_source_type_t
{
    /* LLVM IR */
    CGIR_COMMAND_PROG_SOURCE_TYPE_LLVMIR,

    /* MLIR */
    CGIR_COMMAND_PROG_SOURCE_TYPE_MLIR,

    /* PTX */
    CGIR_COMMAND_PROG_SOURCE_TYPE_PTX,

    /* CL (OpenCL prog language) */
    CGIR_COMMAND_PROG_SOURCE_TYPE_CL,

    /* SPIRV */
    CGIR_COMMAND_PROG_SOURCE_TYPE_SPIRV

}               cgir_command_prog_source_type_t;

/* Prototype (ABI) of a program's entry function within `raw`, so the fuse/jit
 * passes know how it consumes its arguments. */
typedef enum    cgir_command_prog_source_proto_t
{
    /* void kernel(v0, v1, ...): one value parameter per capture (the fusion
     * unit; each parameter is described by cgir_command_prog_param_t below). */
    CGIR_COMMAND_PROG_SOURCE_PROTO_UNPACKED_PARAMS = 0,

    /* void kernel(void ** args): a single args block (args[0] == the task, or a
     * previously-fused __fused_wrapper). */
    CGIR_COMMAND_PROG_SOURCE_PROTO_VOID_PTRPTR,

    /* void kernel(void * args, size_t args_size): a packed byte buffer (leading
     * by-reference pointers, then inline by-value copies). */
    CGIR_COMMAND_PROG_SOURCE_PROTO_PACKED_BUFFER

}               cgir_command_prog_source_proto_t;

/* Whether an entry parameter is passed by reference or by value/copy. */
typedef enum    cgir_command_prog_param_kind_t
{
    CGIR_COMMAND_PROG_PARAM_REFERENCE = 0,  /* shared: the arg slot holds a pointer */
    CGIR_COMMAND_PROG_PARAM_COPY            /* firstprivate: the arg slot holds a copy */
}               cgir_command_prog_param_kind_t;

/* Descriptor of one entry parameter: its passing kind, byte size and byte
 * offset within the packed buffer (PACKED_BUFFER proto). Lets the fuse pass
 * deduplicate references by pointer and copies by memcmp over `size` bytes, and
 * lets the fuse/jit passes pack/reconstruct the per-task buffer at `offset`.
 * `offset` is the producer's (compiler's) natural-aligned layout of THIS task's
 * buffer; it is unused for the UNPACKED_PARAMS proto. C-compatible POD. */
typedef struct  cgir_command_prog_param_t
{
    cgir_command_prog_param_kind_t kind;
    size_t                         size;   /* byte size of the value */
    size_t                         offset; /* byte offset within the packed buffer */
}               cgir_command_prog_param_t;

/* One entry of a program's externalized-global resolution table: a symbol that
 * appears as an *external declaration* in the program's IR (`raw`) together with
 * the real runtime address it must bind to. Producers (e.g. a compiler emitting
 * a task body) externalize the globals a program reads/writes and record their
 * process addresses here; the JIT installs them as absolute symbols so the
 * compiled program operates on the real objects instead of module-local copies.
 * C-compatible POD; addresses are resolved at load time by the producer. */
typedef struct  cgir_command_prog_extern_t
{
    const char * name;   /* symbol name as it appears in the program IR */
    void       * addr;   /* real runtime address to bind that symbol to */
}               cgir_command_prog_extern_t;

/* source code of a prog */
typedef struct  cgir_command_prog_source_t
{
    /* format of the prog */
    cgir_command_prog_source_type_t type;

    /* the source code itself */
    union {
        struct {
            void * raw;
            size_t size;

            /* true iff `raw` is a heap buffer owned by this source (e.g. the
             * bitcode produced by the prog-fuse pass). The owner frees `raw`
             * before overwriting/discarding it. Defaults to false (see
             * command_t's constructor). */
            bool _owned;

            /* optional name of the entry function within `raw` (non-owning). If
             * NULL, the consumer falls back to a heuristic (`__fused_wrapper`,
             * else the first void-returning definition). Needed when `raw`
             * holds several definitions (e.g. a device kernel plus its outlined
             * callees). */
            const char * symbol;

            /* externalized-global resolution table. For each global that `raw`
             * references as an external declaration, its name and the real
             * runtime address to bind it to (see cgir_command_prog_extern_t).
             * The JIT installs these as absolute symbols so the program binds to
             * the process's real objects rather than duplicating module-local
             * globals. NULL/0 when the program is self-contained. */
            const cgir_command_prog_extern_t * externs;
            size_t externs_count;

            /* true iff `externs` is a heap buffer owned by this source (e.g. the
             * merged table produced by the prog-fuse pass). Compiler-provided
             * tables are non-owning (compile-time constants); defaults to false
             * (see command_t's constructor). */
            bool _externs_owned;

            /* prototype (ABI) of the entry function within `raw` (see enum). */
            cgir_command_prog_source_proto_t proto;

            /* per-parameter descriptors of the entry function, in order (see
             * cgir_command_prog_param_t): whether each argument is by reference
             * (shared) or by copy (firstprivate) and its byte size. NULL/0 when
             * unknown (the fuse pass then falls back to a size heuristic). */
            const cgir_command_prog_param_t * params;
            size_t param_count;
            bool   _params_owned;   /* true iff `params` is a heap buffer this source owns */
        } llvmir;
    } content;

}               cgir_command_prog_source_t;

#endif /* __CGIR_PROG_SOURCE_H__ */
