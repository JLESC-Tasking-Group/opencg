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

/*
 *  IMPORTANT — why these kernels use `getelementptr inbounds` and `add nsw`:
 *
 *  For the prog-fuse pass to actually FUSE the two loops (scale ; axpy) into a
 *  single one, LLVM's LoopFuse must prove the cross-loop memory dependences are
 *  legal. On LLVM >= 23 its DependenceAnalysis only reasons about a sibling-loop
 *  ("SameSD") subscript when the address recurrence is known not to wrap, i.e.
 *  the access GEP is `inbounds` (checkSubscript -> hasNoSignedWrap in
 *  DependenceAnalysis.cpp). Without `inbounds`, the subscript is classified as
 *  NonLinear, the SameSD level is revoked, and fusion is (conservatively)
 *  rejected — even though it is legal. Real compiler-emitted kernels (e.g.
 *  clang/libomptarget) always emit `inbounds` array GEPs, so they fuse; this
 *  hand-written IR must do the same to be representative.
 *
 *  cgir therefore RELIES on `inbounds` as the no-wrap/legality witness (present
 *  => provably safe => fuse; absent => DA stays conservative => skip). This is
 *  option (i): sound and zero-risk. A future option (ii) could have the pass
 *  defensively stamp `inbounds`/`nsw` onto kernel GEPs/IVs under cgir's
 *  well-formedness (in-bounds, elementwise) contract to fuse un-annotated IR
 *  more aggressively; it never causes an illegal fusion (LoopFuse still checks
 *  legality) but does assert memory-safety, so it is left for later.
 */

/**
 *  LLVM-IR source for scale: y[i] = s * y[i]
 *
 *  define void @scale(double %s, double* %y, i64 %n) {
 *  entry:
 *    br label %loop
 *  loop:
 *    %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
 *    %ptr = getelementptr inbounds double, double* %y, i64 %i
 *    %val = load double, double* %ptr
 *    %res = fmul double %s, %val
 *    store double %res, double* %ptr
 *    %i.next = add nsw i64 %i, 1
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
    "  %ptr = getelementptr inbounds double, double* %y, i64 %i\n"
    "  %val = load double, double* %ptr\n"
    "  %res = fmul double %s, %val\n"
    "  store double %res, double* %ptr\n"
    "  %i.next = add nsw i64 %i, 1\n"
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
 *    %xptr = getelementptr inbounds double, double* %x, i64 %i
 *    %yptr = getelementptr inbounds double, double* %y, i64 %i
 *    %xval = load double, double* %xptr
 *    %yval = load double, double* %yptr
 *    %prod = fmul double %a, %xval
 *    %sum  = fadd double %prod, %yval
 *    store double %sum, double* %yptr
 *    %i.next = add nsw i64 %i, 1
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
    "  %xptr = getelementptr inbounds double, double* %x, i64 %i\n"
    "  %yptr = getelementptr inbounds double, double* %y, i64 %i\n"
    "  %xval = load double, double* %xptr\n"
    "  %yval = load double, double* %yptr\n"
    "  %prod = fmul double %a, %xval\n"
    "  %sum  = fadd double %prod, %yval\n"
    "  store double %sum, double* %yptr\n"
    "  %i.next = add nsw i64 %i, 1\n"
    "  %cond = icmp slt i64 %i.next, %n\n"
    "  br i1 %cond, label %loop, label %exit\n"
    "exit:\n"
    "  ret void\n"
    "}\n";

/*
 *  Negative test: two otherwise-fusable LLVM-IR programs in series, but with
 *  DIFFERENT launch parameters (here, distinct grid.x while everything else is
 *  identical), must NOT be fused. The prog-fuse legality gate
 *  (command_prog_launch_params_equal) should reject the chain, leaving both
 *  nodes in the graph. Returns 0 on success, 1 on failure.
 */
static int
test_distinct_grid_not_fused(void)
{
    /* Argument slots: only needed so the nodes are well-formed PROG commands.
     * The pass never reads them here because the chain is rejected before any
     * fusion is attempted. */
    double   s_val = 2.0;
    double   a_val = 3.0;
    double   xbuf  = 1.0;
    double   ybuf  = 1.0;
    double * xp    = &xbuf;
    double * yp    = &ybuf;
    int64_t  nn    = 8;

    void * scale_args[3] = { &s_val, &yp, &nn };       /* scale(s, y, n)   */
    void * axpy_args[4]  = { &a_val, &xp, &yp, &nn };  /* axpy(a, x, y, n) */

    command_graph_t * cg = command_graph_new();
    assert(cg);
    cg->init(command_new, command_graph_node_new, command_graph_new);

    command_graph_node_t * entry = cg->node_get_entry();
    command_graph_node_t * exit  = cg->node_get_exit();
    entry->successors.clear();
    exit->predecessors.clear();

    constexpr device_unique_id_t host_device = 0;

    /* Node u: scale, grid = (1,1,1), block = (1,1,1) */
    command_t * cmd_u = command_new(cg, COMMAND_TYPE_PROG);
    assert(cmd_u);
    cmd_u->prog.source.type                 = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    cmd_u->prog.source.content.llvmir.raw   = (void *) scale_llvm_ir;
    cmd_u->prog.source.content.llvmir.size  = sizeof(scale_llvm_ir);
    cmd_u->prog.launcher.variadic.fn        = nullptr;
    cmd_u->prog.args      = scale_args;
    cmd_u->prog.n_args    = sizeof(scale_args) / sizeof(void *);
    cmd_u->prog.grid.x  = 1; cmd_u->prog.grid.y  = 1; cmd_u->prog.grid.z  = 1;
    cmd_u->prog.block.x = 1; cmd_u->prog.block.y = 1; cmd_u->prog.block.z = 1;

    command_graph_node_t * u = command_graph_node_new(cg, host_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(u);
    u->command = cmd_u;

    /* Node v: axpy, same block but grid.x = 2 (DIFFERENT from u) */
    command_t * cmd_v = command_new(cg, COMMAND_TYPE_PROG);
    assert(cmd_v);
    cmd_v->prog.source.type                 = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
    cmd_v->prog.source.content.llvmir.raw   = (void *) axpy_llvm_ir;
    cmd_v->prog.source.content.llvmir.size  = sizeof(axpy_llvm_ir);
    cmd_v->prog.launcher.variadic.fn        = nullptr;
    cmd_v->prog.args      = axpy_args;
    cmd_v->prog.n_args    = sizeof(axpy_args)  / sizeof(void *);
    cmd_v->prog.grid.x  = 2; cmd_v->prog.grid.y  = 1; cmd_v->prog.grid.z  = 1;
    cmd_v->prog.block.x = 1; cmd_v->prog.block.y = 1; cmd_v->prog.block.z = 1;

    command_graph_node_t * v = command_graph_node_new(cg, host_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(v);
    v->command = cmd_v;

    /* Build graph: entry -> u -> v -> exit */
    entry->precedes(u);
    u->precedes(v);
    v->precedes(exit);

    cg->dump("cg-pre-prog-fuse-distinct-grid.dot");

    /* Run the prog-fuse pass: must be a no-op because the launch params differ */
    cg->optimize(COMMAND_GRAPH_PASS_PROG_FUSE);

    cg->dump("cg-post-prog-fuse-distinct-grid.dot");

    /* Verify: both nodes remain (NOT fused) */
    size_t node_count = 0;
    cg->walk([&](command_graph_node_t * node) {
        if (node != cg->node_get_entry() && node != cg->node_get_exit())
            node_count++;
    });

    if (node_count != 2)
    {
        fprintf(stderr, "FAIL: expected 2 nodes (no fusion across distinct grids), got %zu\n", node_count);
        return 1;
    }

    fprintf(stdout, "PASS: prog-fuse left programs with distinct grids unfused (2 nodes)\n");
    return 0;
}

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
    cmd_u->prog.args      = scale_args;
    cmd_u->prog.n_args    = sizeof(scale_args) / sizeof(void *);
    /* Identical launch params on both progs (required for fusion); the fused
     * node must inherit these exact grid/block dimensions. */
    cmd_u->prog.grid.x  = 4;  cmd_u->prog.grid.y  = 2; cmd_u->prog.grid.z  = 1;
    cmd_u->prog.block.x = 32; cmd_u->prog.block.y = 1; cmd_u->prog.block.z = 1;

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
    cmd_v->prog.args      = axpy_args;
    cmd_v->prog.n_args    = sizeof(axpy_args)  / sizeof(void *);
    /* same launch params as cmd_u */
    cmd_v->prog.grid.x  = 4;  cmd_v->prog.grid.y  = 2; cmd_v->prog.grid.z  = 1;
    cmd_v->prog.block.x = 32; cmd_v->prog.block.y = 1; cmd_v->prog.block.z = 1;

    command_graph_node_t * v = command_graph_node_new(cg, host_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(v);
    v->command = cmd_v;

    /* Build graph: entry -> u -> v -> exit */
    entry->precedes(u);
    u->precedes(v);
    v->precedes(exit);

    /* Dump graph before optimization */
    cg->dump("cg-pre-prog-fuse.dot");

    /* Run the prog-fuse pass (fuses IR, leaves the launcher fn NULL) followed by
     * the jit pass (compiles the fused program and installs the fn) */
    cg->optimize(COMMAND_GRAPH_PASS_PROG_FUSE);
    cg->optimize(COMMAND_GRAPH_PASS_JIT);

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

    /* The fused node must inherit the (identical) launch parameters of the
     * programs it fused (grid=(4,2,1), block=(32,1,1) set above). */
    if (fused->command->prog.grid.x  != 4  || fused->command->prog.grid.y  != 2 || fused->command->prog.grid.z  != 1 ||
        fused->command->prog.block.x != 32 || fused->command->prog.block.y != 1 || fused->command->prog.block.z != 1)
    {
        fprintf(stderr,
                "FAIL: fused node launch params not propagated: grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                fused->command->prog.grid.x, fused->command->prog.grid.y, fused->command->prog.grid.z,
                fused->command->prog.block.x, fused->command->prog.block.y, fused->command->prog.block.z);
        return 1;
    }

    fprintf(stdout, "PASS: prog-fuse produced 1 fused PROG node with LLVM-IR source\n");
    fprintf(stdout, "PASS: fused node inherited the programs' launch parameters\n");

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
     * pass), and its slot count n_args.  */
    auto     fn       = fused->command->prog.launcher.variadic.fn;
    void **  args_buf = fused->command->prog.args;
    size_t   n_args   = fused->command->prog.n_args;

    if (!fn)
    {
        fprintf(stderr, "FAIL: fused launcher fn is NULL\n");
        return 1;
    }

    if (n_args != 5)
    {
        fprintf(stderr, "FAIL: expected 5 deduplicated arg slots "
                        "(scale{s,y,n} + axpy{a,x,y,n} with y and n shared), got %zu\n", n_args);
        return 1;
    }

    /* The pass already filled args_buf with the deduplicated slots (pointing at
     * s_val / yp / nn / a_val / xp declared above), so just invoke the wrapper. */
    fn(args_buf);

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

    /* Negative test: programs with distinct launch parameters must not fuse. */
    if (test_distinct_grid_not_fused() != 0)
        return 1;

    return 0;
}
