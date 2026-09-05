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

/*
 *  How the `jit` pass decides the shape of a program's entry, i.e. whether the
 *  compiled function can be launched as-is or needs an argument-unpacking
 *  wrapper synthesized around it.
 *
 *  The interesting case is a leaf whose only capture is a pointer. Under opaque
 *  pointers its LLVM signature is `void(ptr)`, which is EXACTLY the signature of
 *  the uniform `void kernel(void ** args)` launch shape -- so the signature
 *  alone cannot tell them apart, and guessing wrong is silent: the kernel gets
 *  handed the slot array instead of `*(double **) args[0]` and quietly reads and
 *  writes the wrong memory. `cgir_command_prog_source_proto_t` is what
 *  disambiguates, and this test pins that it is honoured.
 *
 *  Three programs, all host, all JIT'd unfused:
 *    1. UNPACKED_PARAMS with one pointer capture -> must get a wrapper
 *    2. VOID_PTRPTR                              -> must be launched directly
 *    3. UNPACKED_PARAMS with several captures    -> must get a wrapper
 *
 *  Each kernel doubles y[0], so a misdispatch is caught by y[0] being left
 *  untouched (the kernel having scribbled on the slot array instead).
 */

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <math.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

/*  void one_ptr(double * y) { y[0] *= 2; }
 *
 *  One parameter, and it is a pointer: `void(ptr)`, indistinguishable by
 *  signature from the `void(void**)` launch shape. */
static const char one_ptr_llvm_ir[] =
    "define void @one_ptr(double* %y) {\n"
    "entry:\n"
    "  %p = getelementptr inbounds double, double* %y, i64 0\n"
    "  %v = load double, double* %p\n"
    "  %r = fmul double %v, 2.000000e+00\n"
    "  store double %r, double* %p\n"
    "  ret void\n"
    "}\n";

/*  void via_args(void ** args) { *(double *) args[0] *= 2; }
 *
 *  Genuinely the uniform launch shape: it takes the slot array and dereferences
 *  it itself. Same signature as one_ptr; only the proto tells them apart. */
static const char via_args_llvm_ir[] =
    "define void @via_args(ptr %args) {\n"
    "entry:\n"
    "  %slot = getelementptr inbounds ptr, ptr %args, i64 0\n"
    "  %y = load ptr, ptr %slot\n"
    "  %v = load double, ptr %y\n"
    "  %r = fmul double %v, 2.000000e+00\n"
    "  store double %r, ptr %y\n"
    "  ret void\n"
    "}\n";

/*  void two_args(double * y, double k) { y[0] *= k; }
 *
 *  A leaf with more than one capture: unambiguous by signature, and the case
 *  that already worked. Here to catch a fix that over-corrects and starts
 *  wrapping (or not wrapping) everything. */
static const char two_args_llvm_ir[] =
    "define void @two_args(double* %y, double %k) {\n"
    "entry:\n"
    "  %v = load double, double* %y\n"
    "  %r = fmul double %v, %k\n"
    "  store double %r, double* %y\n"
    "  ret void\n"
    "}\n";

/* Build a one-node graph around `prog`, run the jit pass, and launch it. */
static int
jit_and_launch(const char * what,
               const char * ir, size_t ir_size, const char * symbol,
               cgir_command_prog_source_proto_t proto,
               const cgir_command_prog_param_t * params, size_t param_count,
               void ** args, size_t n_args)
{
    command_graph_t * cg = command_graph_new();
    assert(cg);
    cg->init(command_new, command_graph_node_new, command_graph_new);

    command_graph_node_t * entry = cg->node_get_entry();
    command_graph_node_t * exit  = cg->node_get_exit();
    entry->successors.clear();
    exit->predecessors.clear();

    command_t * cmd = command_new(cg, COMMAND_TYPE_PROG);
    assert(cmd);
    cmd->prog.source.type                       = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    cmd->prog.source.content.llvmir.raw         = (void *) ir;
    cmd->prog.source.content.llvmir.size        = ir_size;
    cmd->prog.source.content.llvmir.symbol      = symbol;
    cmd->prog.source.content.llvmir.proto       = proto;
    cmd->prog.source.content.llvmir.params      = params;
    cmd->prog.source.content.llvmir.param_count = param_count;
    /* host program: no triple/arch, so the jit pass compiles it for the CPU */
    cmd->prog.launcher.variadic.fn = nullptr;
    cmd->prog.args                 = args;
    cmd->prog.n_args               = n_args;

    constexpr device_unique_id_t host_device = 0;
    command_graph_node_t * u = command_graph_node_new(cg, host_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(u);
    u->command = cmd;

    entry->precedes(u);
    u->precedes(exit);

    cg->optimize(COMMAND_GRAPH_PASS_JIT);

    if (cmd->prog.prototype != CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_VARIADIC)
    {
        fprintf(stderr, "FAIL[%s]: jit did not install a variadic launcher (prototype=%d)\n",
                what, (int) cmd->prog.prototype);
        return 1;
    }
    if (cmd->prog.launcher.variadic.fn == nullptr)
    {
        fprintf(stderr, "FAIL[%s]: jit left the launcher NULL\n", what);
        return 1;
    }

    cmd->prog.launcher.variadic.fn(cmd->prog.args);
    return 0;
}

int
main(void)
{
    int rc = 0;

    /* ---------------------------------------------------------------- *
     *  1. one pointer capture, UNPACKED_PARAMS.                         *
     *                                                                    *
     *  The regression this test exists for: dispatching on the signature *
     *  alone classifies `void(ptr)` as the uniform launch shape, calls   *
     *  one_ptr(args) and doubles the *slot* rather than y[0].            *
     * ---------------------------------------------------------------- */
    {
        double   y  = 21.0;
        double * yp = &y;
        void *   args[1] = { &yp };
        const cgir_command_prog_param_t params[1] = {
            { CGIR_COMMAND_PROG_PARAM_REFERENCE, sizeof(double *), 0 },
        };

        if (jit_and_launch("one-pointer capture",
                           one_ptr_llvm_ir, sizeof(one_ptr_llvm_ir), "one_ptr",
                           CGIR_COMMAND_PROG_SOURCE_PROTO_UNPACKED_PARAMS,
                           params, 1, args, 1))
            return 1;

        if (fabs(y - 42.0) > 1e-12)
        {
            fprintf(stderr,
                    "FAIL: one-pointer leaf was launched with the slot array instead of "
                    "its capture: y = %f (expected 42.0)\n", y);
            rc = 1;
        }
        else
            fprintf(stdout, "PASS: one-pointer leaf got an unpacking wrapper (y = %.1f)\n", y);
    }

    /* ---------------------------------------------------------------- *
     *  2. the same signature, but genuinely the uniform launch shape.   *
     *  Must be launched directly -- wrapping it would double-deref.     *
     * ---------------------------------------------------------------- */
    {
        double   y  = 21.0;
        double * yp = &y;
        void *   args[1] = { &yp };

        if (jit_and_launch("void(void**) entry",
                           via_args_llvm_ir, sizeof(via_args_llvm_ir), "via_args",
                           CGIR_COMMAND_PROG_SOURCE_PROTO_VOID_PTRPTR,
                           nullptr, 0, args, 1))
            return 1;

        if (fabs(y - 42.0) > 1e-12)
        {
            fprintf(stderr, "FAIL: void(void**) entry was not launched directly: "
                            "y = %f (expected 42.0)\n", y);
            rc = 1;
        }
        else
            fprintf(stdout, "PASS: void(void**) entry launched directly (y = %.1f)\n", y);
    }

    /* ---------------------------------------------------------------- *
     *  3. several captures: the unambiguous case, which must keep       *
     *  working (a fix that wraps nothing, or wraps everything, fails    *
     *  one of these three).                                             *
     * ---------------------------------------------------------------- */
    {
        double   y  = 21.0;
        double   k  = 2.0;
        double * yp = &y;
        void *   args[2] = { &yp, &k };
        const cgir_command_prog_param_t params[2] = {
            { CGIR_COMMAND_PROG_PARAM_REFERENCE, sizeof(double *), 0 },
            { CGIR_COMMAND_PROG_PARAM_COPY,      sizeof(double),   8 },
        };

        if (jit_and_launch("multi-capture leaf",
                           two_args_llvm_ir, sizeof(two_args_llvm_ir), "two_args",
                           CGIR_COMMAND_PROG_SOURCE_PROTO_UNPACKED_PARAMS,
                           params, 2, args, 2))
            return 1;

        if (fabs(y - 42.0) > 1e-12)
        {
            fprintf(stderr, "FAIL: multi-capture leaf mis-launched: y = %f (expected 42.0)\n", y);
            rc = 1;
        }
        else
            fprintf(stdout, "PASS: multi-capture leaf got an unpacking wrapper (y = %.1f)\n", y);
    }

    if (rc == 0)
        fprintf(stdout, "ALL TESTS PASSED\n");
    return rc;
}
