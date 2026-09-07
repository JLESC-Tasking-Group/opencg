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
 *  and checks BOTH that the nodes survive AND that the pass refused them for the
 *  expected reason. Checking the node count alone is not enough: a chain is also
 *  left unfused when the IR does not parse, when NVPTX is unavailable, or when
 *  the kernels are not in the shape the pass recognizes. An earlier version of
 *  this test asserted only the count, passed, and gave false confidence while
 *  the gate it was meant to guard was in fact approving everything it saw.
 *
 *  The reason is read from the pass's own stderr diagnostic, which is the only
 *  channel it has. This is why that diagnostic must stay stable.
 *
 *  Not asserted here: that a legal (element-wise) DEVICE chain still fuses.
 *  Reaching the fusion itself needs an NVPTX-capable LLVM (the pass builds a
 *  device TargetMachine after the gate), which is not a property of every build
 *  that can run this test. The positive direction is covered by the host chain
 *  in prog-fuse.cc, which exercises the same merge/dedup/loop-fusion path.
 */

# include <unistd.h>

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <initializer_list>
# include <string>

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
    "define ptx_kernel void @dscale(double %s, double* %y, i64 %n) {\n"
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
    "define ptx_kernel void @dshift(double* %y, double* %z, i64 %n) {\n"
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
    "define ptx_kernel void @dsum(double* %y, double* %acc, i64 %n) {\n"
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

/* dopaque: the generic-mode shape. The kernel does nothing itself; it hands the
 * loop, as a function POINTER, to a runtime declaration that will run it. This is
 * exactly how clang emits `omp target teams distribute parallel for` before
 * SPMD-ization, and it is what defeated the first version of the gate: the fused
 * wrapper then contains no load or store at all, so a pairwise dependence test
 * over its accesses examines nothing and approves everything. The pass must
 * notice that it cannot see the memory effects and refuse. */
static const char dopaque_ir[] =
    DEVICE_PREAMBLE
    KERNEL_ENV("kenv_opaque")
    "declare void @__kmpc_parallel_51(i8*, i32, i32, i32, i32, i8*, i8*, i8**, i64)\n"
    "define internal void @outlined_body(double %s, double* %y, i64 %n) {\n"
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
    "}\n"
    "define ptx_kernel void @dopaque(double %s, double* %y, i64 %n) {\n"
    "entry:\n"
    "  %tid = call i32 @__kmpc_target_init(i8* bitcast (%struct.KernelEnv* @kenv_opaque to i8*))\n"
    "  %spmd = icmp eq i32 %tid, -1\n"
    "  br i1 %spmd, label %par, label %done\n"
    "par:\n"
    "  call void @__kmpc_parallel_51(i8* null, i32 %tid, i32 1, i32 -1, i32 -1,\n"
    "                                i8* bitcast (void (double, double*, i64)* @outlined_body to i8*),\n"
    "                                i8* null, i8** null, i64 0)\n"
    "  call void @__kmpc_target_deinit()\n"
    "  br label %done\n"
    "done:\n"
    "  ret void\n"
    "}\n";

/* Redirect stderr to a temporary file for the duration of a scope, so the test
 * can read the pass's diagnostic. The pass has no other way to report WHY it
 * refused a chain, and the reason is the thing under test. */
struct captured_stderr {
    int saved;
    FILE *tmp;

    captured_stderr() : saved(dup(fileno(stderr))), tmp(tmpfile())
    {
        fflush(stderr);
        if (tmp)
            dup2(fileno(tmp), fileno(stderr));
    }

    std::string release()
    {
        fflush(stderr);
        dup2(saved, fileno(stderr));
        close(saved);
        saved = -1;
        std::string out;
        if (tmp) {
            rewind(tmp);
            char buf[4096];
            size_t k;
            while ((k = fread(buf, 1, sizeof(buf), tmp)) > 0)
                out.append(buf, k);
            fclose(tmp);
            tmp = nullptr;
        }
        /* echo it, so a failing run still shows what the pass said */
        fputs(out.c_str(), stderr);
        return out;
    }

    ~captured_stderr()
    {
        if (saved >= 0) { dup2(saved, fileno(stderr)); close(saved); }
        if (tmp) fclose(tmp);
    }
};

/* Build `entry -> u -> v -> exit` with two device PROG commands, run prog-fuse,
 * and return the number of surviving command nodes. `diag` receives whatever the
 * pass wrote to stderr. */
static size_t
run_device_chain(const char * ir_u, size_t ir_u_size, void ** args_u, size_t n_args_u,
                 const char * ir_v, size_t ir_v_size, void ** args_v, size_t n_args_v,
                 const char * dotfile, std::string & diag)
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

    {
        captured_stderr cap;
        cg->optimize(COMMAND_GRAPH_PASS_PROG_FUSE);
        diag = cap.release();
    }
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

    /* A case is only satisfied when the chain survived AND the pass said it
     * refused it for `expect`. Without the second half the test passes when the
     * chain is left unfused for an unrelated reason -- a missing NVPTX target,
     * unparsable IR, an unrecognized kernel shape -- and so cannot tell a working
     * gate from an absent one. That is not a hypothetical failure mode: it is how
     * this test previously reported success while the gate approved an SpMV
     * gather fused with the kernel that produced its input. */
    auto check = [&] (const char * what, size_t nodes, const std::string & diag,
                      const char * expect)
    {
        if (nodes != 2)
        {
            fprintf(stderr, "FAIL: %s was fused into %zu node(s); it must stay 2\n",
                    what, nodes);
            ++failures;
            return;
        }
        if (diag.find("left unfused") == std::string::npos)
        {
            fprintf(stderr, "FAIL: %s stayed 2 nodes but the pass never reported "
                            "refusing it -- it was not even considered for fusion, "
                            "so this case proves nothing\n", what);
            ++failures;
            return;
        }
        if (diag.find(expect) == std::string::npos)
        {
            fprintf(stderr, "FAIL: %s was refused, but not for the expected reason "
                            "(wanted a diagnostic containing '%s')\n", what, expect);
            ++failures;
            return;
        }
        fprintf(stdout, "PASS: %s left unfused (%s)\n", what, expect);
    };

    /* Same, when more than one diagnostic legitimately satisfies the case. */
    auto check_any = [&] (const char * what, size_t nodes, const std::string & diag,
                          std::initializer_list<const char *> expect)
    {
        for (const char * e : expect)
            if (diag.find(e) != std::string::npos)
            {
                check(what, nodes, diag, e);
                return;
            }
        check(what, nodes, diag, *expect.begin());   /* reports the mismatch */
    };

    /* Case 1: a neighbour read. dshift's thread i reads y[i+1], which dscale's
     * thread i+1 wrote -- a cross-thread dependence that only the launch
     * boundary orders. */
    {
        std::string diag;
        size_t nodes = run_device_chain(
            dscale_ir, sizeof(dscale_ir), scale_args, 3,
            dshift_ir, sizeof(dshift_ir), shift_args, 3,
            "cg-prog-fuse-device-neighbour.dot", diag);
        check("a neighbour-read device chain", nodes, diag, "different indices");
    }

    /* Case 2: a reduction. Its result is only complete at kernel exit, so a
     * later kernel in the same launch may read a partial sum. */
    {
        std::string diag;
        size_t nodes = run_device_chain(
            dsum_ir,   sizeof(dsum_ir),   sum_args,   3,
            dscale_ir, sizeof(dscale_ir), scale_args, 3,
            "cg-prog-fuse-device-reduction.dot", diag);
        check("a reduction device chain", nodes, diag, "across threads");
    }

    /* Case 3: an opaque call. A kernel that hands a function pointer to a runtime
     * declaration hides its memory effects -- this is the shape a generic-mode
     * `teams distribute parallel for` has, and approving it is what let the real
     * corruption through. */
    {
        std::string diag;
        size_t nodes = run_device_chain(
            dopaque_ir, sizeof(dopaque_ir), scale_args, 3,
            dscale_ir,  sizeof(dscale_ir),  scale_args, 3,
            "cg-prog-fuse-device-opaque.dot", diag);
        /* Two diagnostics satisfy this case, and both are the gate saying it
         * cannot see: the call that hides the region ("not visible here"), or the
         * consequence when that call is the only thing in the body ("no visible
         * memory access"). Which one fires depends on what the device passes make
         * of this synthetic kernel; the property under test is that a hidden
         * region is never APPROVED. */
        check_any("a device chain with a hidden parallel region", nodes, diag,
                  { "not visible here", "no visible memory access" });
    }

    if (failures)
        fprintf(stderr, "%d device-legality case(s) failed\n", failures);
    return failures ? 1 : 0;
}
