/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software. See the LICENSE file.
**/

/*
 *  Regression test for the device-chain fusion legality gate.
 *
 *  Two `omp target` regions run as two kernel launches, and the launch boundary
 *  is a DEVICE-WIDE barrier. Fusing them into one launch removes it: nothing in
 *  a single SPMD launch synchronizes across thread blocks. The fusion is
 *  therefore only meaning-preserving when no thread of the second kernel reads a
 *  location a DIFFERENT thread of the first one wrote.
 *
 *  Before the gate existed, prog-fuse collapsed the brackets unconditionally.
 *  On the Krylov solvers that silently corrupted the answer -- CG's relative
 *  residual went from 4e-15 to 2.6e-03 -- because `dot(p, Ap)` is a reduction
 *  whose value is only complete at kernel exit, and the next kernel divides by
 *  it. The failure was nondeterministic (a race), which is exactly why it needs
 *  a deterministic test.
 *
 *  Each case below builds a two-node device chain that prog-fuse MUST refuse,
 *  and checks that both nodes survive the pass. A refusal leaves the chain as
 *  two separate launches: slower, but correct.
 *
 *  Not asserted here: that a legal (element-wise) DEVICE chain still fuses.
 *  Reaching the fusion itself needs an NVPTX-capable LLVM (the pass builds a
 *  device TargetMachine after the gate), which is not a property of every build
 *  that can run this test. The positive direction is covered by the host chain
 *  in prog-fuse.cc, which exercises the same merge/dedup/loop-fusion path.
 */

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

/* The device kernels below are the minimal shape collapse_device_kernel_brackets
 * recognizes: a `__kmpc_target_init` whose result feeds `icmp eq -1` (the SPMD
 * pattern), the body, then `__kmpc_target_deinit`. The kernel environment is a
 * constant struct whose first field is the configuration the gate compares
 * across kernels -- identical here, so the chain is rejected on its MEMORY
 * behaviour and not on a launch-configuration mismatch. */
# define DEVICE_PREAMBLE                                                       \
    "%struct.ConfigEnv = type { i8, i8, i32 }\n"                               \
    "%struct.KernelEnv = type { %struct.ConfigEnv, i8*, i8* }\n"               \
    "declare i32 @__kmpc_target_init(i8*)\n"                                   \
    "declare void @__kmpc_target_deinit()\n"

# define KERNEL_ENV(NAME)                                                      \
    "@" NAME " = weak constant %struct.KernelEnv "                             \
    "{ %struct.ConfigEnv { i8 1, i8 1, i32 1 }, i8* null, i8* null }\n"

/* dscale: y[i] = s * y[i] -- element-wise, legal on its own. */
static const char dscale_ir[] =
    DEVICE_PREAMBLE
    KERNEL_ENV("kenv_scale")
    "define void @dscale(double %s, double* %y, i64 %n) {\n"
    "entry:\n"
    "  %tid = call i32 @__kmpc_target_init(i8* bitcast (%struct.KernelEnv* @kenv_scale to i8*))\n"
    "  %spmd = icmp eq i32 %tid, -1\n"
    "  br i1 %spmd, label %loop, label %done\n"
    "loop:\n"
    "  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]\n"
    "  %ptr = getelementptr inbounds double, double* %y, i64 %i\n"
    "  %val = load double, double* %ptr\n"
    "  %res = fmul double %s, %val\n"
    "  store double %res, double* %ptr\n"
    "  %i.next = add nsw i64 %i, 1\n"
    "  %cond = icmp slt i64 %i.next, %n\n"
    "  br i1 %cond, label %loop, label %fini\n"
    "fini:\n"
    "  call void @__kmpc_target_deinit()\n"
    "  br label %done\n"
    "done:\n"
    "  ret void\n"
    "}\n";

/* dshift: z[i] = y[i + 1] -- thread i reads the element thread i+1 of the
 * PREVIOUS kernel wrote. Legal only because the launch boundary separates them;
 * fusing must be refused. */
static const char dshift_ir[] =
    DEVICE_PREAMBLE
    KERNEL_ENV("kenv_shift")
    "define void @dshift(double* %y, double* %z, i64 %n) {\n"
    "entry:\n"
    "  %tid = call i32 @__kmpc_target_init(i8* bitcast (%struct.KernelEnv* @kenv_shift to i8*))\n"
    "  %spmd = icmp eq i32 %tid, -1\n"
    "  br i1 %spmd, label %loop, label %done\n"
    "loop:\n"
    "  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]\n"
    "  %j = add nsw i64 %i, 1\n"
    "  %yptr = getelementptr inbounds double, double* %y, i64 %j\n"
    "  %zptr = getelementptr inbounds double, double* %z, i64 %i\n"
    "  %val = load double, double* %yptr\n"
    "  store double %val, double* %zptr\n"
    "  %i.next = add nsw i64 %i, 1\n"
    "  %cond = icmp slt i64 %i.next, %n\n"
    "  br i1 %cond, label %loop, label %fini\n"
    "fini:\n"
    "  call void @__kmpc_target_deinit()\n"
    "  br label %done\n"
    "done:\n"
    "  ret void\n"
    "}\n";

/* dsum: acc[0] = sum(y[i]) through the OpenMP device reduction runtime. Its
 * value only exists once every thread has contributed, i.e. at kernel exit.
 * This is the Krylov `dot()` shape that produced the corruption. */
static const char dsum_ir[] =
    DEVICE_PREAMBLE
    KERNEL_ENV("kenv_sum")
    "declare i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(i8*, i32, i8*, i8*)\n"
    "define void @dsum(double* %y, double* %acc, i64 %n) {\n"
    "entry:\n"
    "  %tid = call i32 @__kmpc_target_init(i8* bitcast (%struct.KernelEnv* @kenv_sum to i8*))\n"
    "  %spmd = icmp eq i32 %tid, -1\n"
    "  br i1 %spmd, label %loop, label %done\n"
    "loop:\n"
    "  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]\n"
    "  %acc0 = phi double [ 0.0, %entry ], [ %acc1, %loop ]\n"
    "  %ptr = getelementptr inbounds double, double* %y, i64 %i\n"
    "  %val = load double, double* %ptr\n"
    "  %acc1 = fadd double %acc0, %val\n"
    "  %i.next = add nsw i64 %i, 1\n"
    "  %cond = icmp slt i64 %i.next, %n\n"
    "  br i1 %cond, label %loop, label %reduce\n"
    "reduce:\n"
    "  %accb = bitcast double* %acc to i8*\n"
    "  %r = call i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(i8* null, i32 1, i8* %accb, i8* null)\n"
    "  store double %acc1, double* %acc\n"
    "  call void @__kmpc_target_deinit()\n"
    "  br label %done\n"
    "done:\n"
    "  ret void\n"
    "}\n";

/* Build `entry -> u -> v -> exit` with two device PROG commands, run prog-fuse,
 * and return the number of surviving command nodes. */
static size_t
run_device_chain(const char * ir_u, size_t ir_u_size, void ** args_u, size_t n_args_u,
                 const char * ir_v, size_t ir_v_size, void ** args_v, size_t n_args_v,
                 const char * dotfile)
{
    command_graph_t * cg = command_graph_new();
    assert(cg);
    cg->init(command_new, command_graph_node_new, command_graph_new);

    command_graph_node_t * entry = cg->node_get_entry();
    command_graph_node_t * exit  = cg->node_get_exit();
    entry->successors.clear();
    exit->predecessors.clear();

    constexpr device_unique_id_t gpu_device = 1;

    auto make = [&] (const char * ir, size_t ir_size, void ** args, size_t n_args)
    {
        command_t * cmd = command_new(cg, COMMAND_TYPE_PROG);
        assert(cmd);
        cmd->prog.source.type                = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
        cmd->prog.source.content.llvmir.raw  = (void *) ir;
        cmd->prog.source.content.llvmir.size = ir_size;
        /* a non-null triple is what makes this a DEVICE program */
        cmd->prog.source.content.llvmir.triple = "nvptx64-nvidia-cuda";
        cmd->prog.launcher.variadic.fn       = nullptr;
        cmd->prog.args   = args;
        cmd->prog.n_args = n_args;
        cmd->prog.grid.x  = 1; cmd->prog.grid.y  = 1; cmd->prog.grid.z  = 1;
        cmd->prog.block.x = 1; cmd->prog.block.y = 1; cmd->prog.block.z = 1;

        command_graph_node_t * node =
            command_graph_node_new(cg, gpu_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
        assert(node);
        node->command = cmd;
        return node;
    };

    command_graph_node_t * u = make(ir_u, ir_u_size, args_u, n_args_u);
    command_graph_node_t * v = make(ir_v, ir_v_size, args_v, n_args_v);

    entry->precedes(u);
    u->precedes(v);
    v->precedes(exit);

    cg->optimize(COMMAND_GRAPH_PASS_PROG_FUSE);
    cg->dump(dotfile);

    size_t nodes = 0;
    cg->walk([&](command_graph_node_t * node) {
        if (node != cg->node_get_entry() && node != cg->node_get_exit())
            ++nodes;
    });
    return nodes;
}

int
main(void)
{
    int failures = 0;

    double   s_val = 2.0;
    double   ybuf[4] = { 1.0, 2.0, 3.0, 4.0 };
    double   zbuf[4] = { 0.0, 0.0, 0.0, 0.0 };
    double   accbuf  = 0.0;
    double * yp   = ybuf;
    double * zp   = zbuf;
    double * accp = &accbuf;
    int64_t  nn   = 4;

    void * scale_args[3] = { &s_val, &yp, &nn };   /* dscale(s, y, n)  */
    void * shift_args[3] = { &yp, &zp, &nn };      /* dshift(y, z, n)  */
    void * sum_args[3]   = { &yp, &accp, &nn };    /* dsum(y, acc, n)  */

    /* Case 1: a neighbour read. dshift's thread i reads y[i+1], which dscale's
     * thread i+1 wrote -- a cross-thread dependence that only the launch
     * boundary orders. */
    {
        size_t nodes = run_device_chain(
            dscale_ir, sizeof(dscale_ir), scale_args, 3,
            dshift_ir, sizeof(dshift_ir), shift_args, 3,
            "cg-prog-fuse-device-neighbour.dot");
        if (nodes != 2)
        {
            fprintf(stderr, "FAIL: a cross-thread (neighbour) device chain was fused "
                            "into %zu node(s); it must stay 2\n", nodes);
            ++failures;
        }
        else
            fprintf(stdout, "PASS: neighbour-read device chain left unfused\n");
    }

    /* Case 2: a reduction. Its result is only complete at kernel exit, so a
     * later kernel in the same launch may read a partial sum. */
    {
        size_t nodes = run_device_chain(
            dsum_ir,   sizeof(dsum_ir),   sum_args,   3,
            dscale_ir, sizeof(dscale_ir), scale_args, 3,
            "cg-prog-fuse-device-reduction.dot");
        if (nodes != 2)
        {
            fprintf(stderr, "FAIL: a device chain containing a reduction was fused "
                            "into %zu node(s); it must stay 2\n", nodes);
            ++failures;
        }
        else
            fprintf(stdout, "PASS: reduction device chain left unfused\n");
    }

    if (failures)
        fprintf(stderr, "%d device-legality case(s) failed\n", failures);
    return failures ? 1 : 0;
}
