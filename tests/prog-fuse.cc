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

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <math.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

/**
 *  LLVM-IR source for scale: y[i] = s * y[i]
 *
 *  define void @scale(double %s, double* %y, i64 %n) {
 *  entry:
 *    br label %loop
 *  loop:
 *    %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
 *    %ptr = getelementptr double, double* %y, i64 %i
 *    %val = load double, double* %ptr
 *    %res = fmul double %s, %val
 *    store double %res, double* %ptr
 *    %i.next = add i64 %i, 1
 *    %cond = icmp slt i64 %i.next, %n
 *    br i1 %cond, label %loop, label %exit
 *  exit:
 *    ret void
 *  }
 */
static const char scale_llvm_ir[] =
    "define void @scale(double %s, double* %y, i64 %n) {\n"
    "entry:\n"
    "  br label %loop\n"
    "loop:\n"
    "  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]\n"
    "  %ptr = getelementptr double, double* %y, i64 %i\n"
    "  %val = load double, double* %ptr\n"
    "  %res = fmul double %s, %val\n"
    "  store double %res, double* %ptr\n"
    "  %i.next = add i64 %i, 1\n"
    "  %cond = icmp slt i64 %i.next, %n\n"
    "  br i1 %cond, label %loop, label %exit\n"
    "exit:\n"
    "  ret void\n"
    "}\n";

/**
 *  LLVM-IR source for axpy: y[i] = a * x[i] + y[i]
 *
 *  define void @axpy(double %a, double* %x, double* %y, i64 %n) {
 *  entry:
 *    br label %loop
 *  loop:
 *    %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
 *    %xptr = getelementptr double, double* %x, i64 %i
 *    %yptr = getelementptr double, double* %y, i64 %i
 *    %xval = load double, double* %xptr
 *    %yval = load double, double* %yptr
 *    %prod = fmul double %a, %xval
 *    %sum  = fadd double %prod, %yval
 *    store double %sum, double* %yptr
 *    %i.next = add i64 %i, 1
 *    %cond = icmp slt i64 %i.next, %n
 *    br i1 %cond, label %loop, label %exit
 *  exit:
 *    ret void
 *  }
 */
static const char axpy_llvm_ir[] =
    "define void @axpy(double %a, double* %x, double* %y, i64 %n) {\n"
    "entry:\n"
    "  br label %loop\n"
    "loop:\n"
    "  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]\n"
    "  %xptr = getelementptr double, double* %x, i64 %i\n"
    "  %yptr = getelementptr double, double* %y, i64 %i\n"
    "  %xval = load double, double* %xptr\n"
    "  %yval = load double, double* %yptr\n"
    "  %prod = fmul double %a, %xval\n"
    "  %sum  = fadd double %prod, %yval\n"
    "  store double %sum, double* %yptr\n"
    "  %i.next = add i64 %i, 1\n"
    "  %cond = icmp slt i64 %i.next, %n\n"
    "  br i1 %cond, label %loop, label %exit\n"
    "exit:\n"
    "  ret void\n"
    "}\n";

int
main(void)
{
    constexpr size_t n = 8;
    constexpr double s = 2.0;
    constexpr double a = 3.0;

    /* Allocate vectors */
    double x[n], y[n], y_expected[n];

    for (size_t i = 0 ; i < n ; ++i)
    {
        x[i] = (double) (i + 1);       /* x = {1, 2, 3, ..., 8} */
        y[i] = (double) (n - i);        /* y = {8, 7, 6, ..., 1} */
    }

    /* Compute expected result: y = a * x + s * y */
    for (size_t i = 0 ; i < n ; ++i)
        y_expected[i] = a * x[i] + s * y[i];

    /* Typed argument values; their addresses are the launcher arg slots.
     * y and n are shared by both kernels (scale's y == axpy's y, same n), so
     * argument deduplication should compact 7 slots -> 5. These must outlive the
     * fused-wrapper call (their addresses are stored in the fused args buffer). */
    double   s_val = s;
    double   a_val = a;
    double * xp    = x;
    double * yp    = y;
    int64_t  nn    = static_cast<int64_t>(n);

    /* Per-program argument slot arrays (each slot = &value), populated BEFORE
     * fusion so the pass can read them, deduplicate, and fill the fused buffer. */
    void * scale_args[3] = { &s_val, &yp, &nn };       /* scale(s, y, n)   */
    void * axpy_args[4]  = { &a_val, &xp, &yp, &nn };  /* axpy(a, x, y, n) */

    /* Create a command graph */
    command_graph_t * cg = command_graph_new();
    assert(cg);
    cg->init(command_new, command_graph_node_new, command_graph_new);

    command_graph_node_t * entry = cg->node_get_entry();
    command_graph_node_t * exit  = cg->node_get_exit();

    /* Remove the default entry -> exit edge */
    entry->successors.clear();
    exit->predecessors.clear();

    /* Node u: scale(s, y, n)  =>  y := s * y */
    command_t * cmd_u = command_new(cg, COMMAND_TYPE_PROG);
    assert(cmd_u);
    cmd_u->prog.source.type                 = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    cmd_u->prog.source.content.llvmir.raw   = (void *) scale_llvm_ir;
    cmd_u->prog.source.content.llvmir.size  = sizeof(scale_llvm_ir);
    cmd_u->prog.launcher.variadic.fn        = nullptr;
    cmd_u->prog.launcher.variadic.args      = scale_args;
    cmd_u->prog.launcher.variadic.args_size = sizeof(scale_args);

    constexpr device_unique_id_t host_device = 0;
    command_graph_node_t * u = command_graph_node_new(cg, host_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(u);
    u->command = cmd_u;

    /* Node v: axpy(a, x, y, n)  =>  y := a * x + y */
    command_t * cmd_v = command_new(cg, COMMAND_TYPE_PROG);
    assert(cmd_v);
    cmd_v->prog.source.type                 = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    cmd_v->prog.source.content.llvmir.raw   = (void *) axpy_llvm_ir;
    cmd_v->prog.source.content.llvmir.size  = sizeof(axpy_llvm_ir);
    cmd_v->prog.launcher.variadic.fn        = nullptr;
    cmd_v->prog.launcher.variadic.args      = axpy_args;
    cmd_v->prog.launcher.variadic.args_size = sizeof(axpy_args);

    command_graph_node_t * v = command_graph_node_new(cg, host_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(v);
    v->command = cmd_v;

    /* Build graph: entry -> u -> v -> exit */
    entry->precedes(u);
    u->precedes(v);
    v->precedes(exit);

    /* Dump graph before optimization */
    cg->dump("cg-pre-prog-fuse.dot");

    /* Run the prog-fuse pass */
    cg->optimize(COMMAND_GRAPH_PASS_PROG_FUSE);

    /* Dump graph after optimization */
    cg->dump("cg-post-prog-fuse.dot");

    /* Verify: after fusion, there should be a single (non entry/exit) node */
    size_t node_count = 0;
    cg->walk([&](command_graph_node_t * node) {
        if (node != cg->node_get_entry() && node != cg->node_get_exit())
            node_count++;
    });

    if (node_count != 1)
    {
        fprintf(stderr, "FAIL: expected 1 node after prog-fuse, got %zu\n", node_count);
        return 1;
    }

    /* Verify the fused node is a PROG with LLVM-IR source */
    command_graph_node_t * fused = nullptr;
    cg->walk([&](command_graph_node_t * node) {
        if (node != cg->node_get_entry() && node != cg->node_get_exit())
            fused = node;
    });

    assert(fused);
    assert(fused->command);

    if (fused->command->type != COMMAND_TYPE_PROG)
    {
        fprintf(stderr, "FAIL: fused node is not a PROG (type=%d)\n", fused->command->type);
        return 1;
    }

    if (fused->command->prog.source.type != COMMAND_PROG_SOURCE_TYPE_LLVMIR)
    {
        fprintf(stderr, "FAIL: fused node source is not LLVM-IR (type=%d)\n",
                fused->command->prog.source.type);
        return 1;
    }

    if (fused->command->prog.source.content.llvmir.raw == nullptr ||
        fused->command->prog.source.content.llvmir.size == 0)
    {
        fprintf(stderr, "FAIL: fused node has NULL source\n");
        return 1;
    }

    fprintf(stdout, "PASS: prog-fuse produced 1 fused PROG node with LLVM-IR source\n");

    /* ------------------------------------------------------------------ *
     *  Numerical correctness test                                         *
     *                                                                     *
     *  The fused wrapper has signature:                                   *
     *    void __fused_wrapper(void ** args)                               *
     *                                                                     *
     *  The pass read the originals' args (scale_args / axpy_args), merged  *
     *  the shared slots (y and n), and filled the fused buffer with the    *
     *  5 deduplicated slots:                                               *
     *    args[0] = &s_val   (scale s)                                      *
     *    args[1] = &yp      (y*, shared by scale and axpy)                 *
     *    args[2] = &nn      (n,  shared by scale and axpy)                 *
     *    args[3] = &a_val   (axpy a)                                       *
     *    args[4] = &xp      (axpy x*)                                      *
     *  Each slot holds a void* that points to the actual value, matching   *
     *  the double-dereference in load_arg() inside the wrapper.            *
     * ------------------------------------------------------------------ */

    /* The variadic launcher stores fn, the args buffer (already filled by the
     * pass), and its byte size.  args_size / sizeof(void*) gives the slots.  */
    void *   fn        = fused->command->prog.launcher.variadic.fn;
    void **  args_buf  = static_cast<void **>(fused->command->prog.launcher.variadic.args);
    size_t   args_size = fused->command->prog.launcher.variadic.args_size;

    if (!fn)
    {
        fprintf(stderr, "FAIL: fused launcher fn is NULL\n");
        return 1;
    }

    size_t n_args = args_size / sizeof(void *);
    if (n_args != 5)
    {
        fprintf(stderr, "FAIL: expected 5 deduplicated arg slots "
                        "(scale{s,y,n} + axpy{a,x,y,n} with y and n shared), got %zu\n", n_args);
        return 1;
    }

    /* The pass already filled args_buf with the deduplicated slots (pointing at
     * s_val / yp / nn / a_val / xp declared above), so just invoke the wrapper. */
    typedef void (*fused_fn_t)(void **);
    reinterpret_cast<fused_fn_t>(fn)(args_buf);

    /* Check results: the fused kernel performed scale then axpy, so:
     *   y[i]  = a * x[i] + s * y_init[i]
     * which matches y_expected[] computed above.                         */
    bool all_correct = true;
    for (size_t i = 0; i < n; ++i)
    {
        double diff = fabs(y[i] - y_expected[i]);
        if (diff > 1e-10)
        {
            fprintf(stderr,
                    "FAIL: y[%zu] = %.6f, expected %.6f (diff = %.2e)\n",
                    i, y[i], y_expected[i], diff);
            all_correct = false;
        }
    }

    if (!all_correct)
        return 1;

    fprintf(stdout, "PASS: fused kernel produced numerically correct results\n");
    return 0;
}
