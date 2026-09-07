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
 *  Device program fusion: the residency precondition.
 *
 *  Two device programs run as two kernel launches, and the boundary between them
 *  is a device-wide barrier -- every thread of the first finishes before any
 *  thread of the second starts. Fusing them into one launch removes it, so the
 *  fused kernel carries a grid-wide barrier instead (emit_grid_barrier).
 *
 *  That barrier only completes if every block of the grid is resident: a block
 *  the hardware has not scheduled never arrives at it, and the ones that have
 *  spin forever. cgir therefore fuses device programs only when the runtime has
 *  told it how many blocks fit at once (command_prog_t::max_coresident_blocks)
 *  and the grid is within that.
 *
 *  Getting this wrong hangs the GPU rather than returning a wrong answer, so the
 *  two directions are worth pinning down:
 *
 *    - a grid that fits          -> fused into one node
 *    - a grid that does not fit  -> left as two nodes
 *    - no residency figure at all-> left as two nodes
 *
 *  The fused case additionally checks that the surviving program is marked as
 *  needing a co-resident grid, because that flag is what makes the CUDA driver
 *  use a cooperative launch; a fused kernel launched normally would hang.
 */

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

/* A minimal device kernel in the shape cgir recognizes: the SPMD bracket
 * (__kmpc_target_init returning -1 for every thread) around an element-wise
 * loop. Two of these are chained; what varies between the cases below is only
 * the launch geometry and the residency figure. */
# define DEVICE_PREAMBLE                                                       \
    "%struct.ConfigEnv = type { i8, i8, i32 }\n"                               \
    "%struct.KernelEnv = type { %struct.ConfigEnv, i8*, i8* }\n"               \
    "declare i32 @__kmpc_target_init(i8*)\n"                                   \
    "declare void @__kmpc_target_deinit()\n"

# define DEVICE_KERNEL(NAME, ENV)                                              \
    DEVICE_PREAMBLE                                                            \
    "@" ENV " = weak constant %struct.KernelEnv "                              \
    "{ %struct.ConfigEnv { i8 1, i8 1, i32 1 }, i8* null, i8* null }\n"        \
    "define ptx_kernel void @" NAME "(double %s, double* %y, i64 %n) {\n"      \
    "entry:\n"                                                                 \
    "  %tid = call i32 @__kmpc_target_init(i8* bitcast (%struct.KernelEnv* @" ENV " to i8*))\n" \
    "  %spmd = icmp eq i32 %tid, -1\n"                                         \
    "  br i1 %spmd, label %loop, label %done\n"                                \
    "loop:\n"                                                                  \
    "  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]\n"                       \
    "  %ptr = getelementptr inbounds double, double* %y, i64 %i\n"             \
    "  %val = load double, double* %ptr\n"                                     \
    "  %res = fmul double %s, %val\n"                                          \
    "  store double %res, double* %ptr\n"                                      \
    "  %i.next = add nsw i64 %i, 1\n"                                          \
    "  %cond = icmp slt i64 %i.next, %n\n"                                     \
    "  br i1 %cond, label %loop, label %fini\n"                                \
    "fini:\n"                                                                  \
    "  call void @__kmpc_target_deinit()\n"                                    \
    "  br label %done\n"                                                       \
    "done:\n"                                                                  \
    "  ret void\n"                                                             \
    "}\n"

static const char dscale_ir[] = DEVICE_KERNEL("dscale", "kenv_scale");
static const char dshift_ir[] = DEVICE_KERNEL("dshift", "kenv_shift");

/* Build `entry -> u -> v -> exit` with two device PROG commands sharing a grid
 * of `grid_x` blocks and a residency budget of `coresident` blocks, run
 * prog-fuse, and report how many command nodes survive. `fused_needs_grid`
 * receives the surviving program's requires_coresident_grid flag. */
static size_t
run_device_chain(unsigned grid_x, unsigned coresident, bool & fused_needs_grid)
{
    static double   s_val = 2.0;
    static double   ybuf[4] = { 1.0, 2.0, 3.0, 4.0 };
    static double * yp = ybuf;
    static int64_t  nn = 4;
    static void *   args[3] = { &s_val, &yp, &nn };

    command_graph_t * cg = command_graph_new();
    assert(cg);
    cg->init(command_new, command_graph_node_new, command_graph_new);

    command_graph_node_t * entry = cg->node_get_entry();
    command_graph_node_t * exit  = cg->node_get_exit();
    entry->successors.clear();
    exit->predecessors.clear();

    constexpr device_unique_id_t gpu_device = 1;

    auto make = [&] (const char * ir, size_t ir_size)
    {
        command_t * cmd = command_new(cg, COMMAND_TYPE_PROG);
        assert(cmd);
        cmd->prog.source.type                  = COMMAND_PROG_SOURCE_TYPE_LLVMIR;
        cmd->prog.source.content.llvmir.raw    = (void *) ir;
        cmd->prog.source.content.llvmir.size   = ir_size;
        /* a non-null triple is what makes this a DEVICE program */
        cmd->prog.source.content.llvmir.triple = "nvptx64-nvidia-cuda";
        cmd->prog.launcher.variadic.fn         = nullptr;
        cmd->prog.args   = args;
        cmd->prog.n_args = 3;
        cmd->prog.grid.x  = grid_x; cmd->prog.grid.y  = 1; cmd->prog.grid.z  = 1;
        cmd->prog.block.x = 32;     cmd->prog.block.y = 1; cmd->prog.block.z = 1;
        cmd->prog.max_coresident_blocks = coresident;

        command_graph_node_t * node =
            command_graph_node_new(cg, gpu_device, COMMAND_GRAPH_NODE_TYPE_COMMAND);
        assert(node);
        node->command = cmd;
        return node;
    };

    command_graph_node_t * u = make(dscale_ir, sizeof(dscale_ir));
    command_graph_node_t * v = make(dshift_ir, sizeof(dshift_ir));

    entry->precedes(u);
    u->precedes(v);
    v->precedes(exit);

    cg->optimize(COMMAND_GRAPH_PASS_PROG_FUSE);

    size_t nodes = 0;
    fused_needs_grid = false;
    cg->walk([&](command_graph_node_t * node) {
        if (node == cg->node_get_entry() || node == cg->node_get_exit())
            return;
        ++nodes;
        if (node->command && node->command->type == COMMAND_TYPE_PROG)
            fused_needs_grid = node->command->prog.requires_coresident_grid;
    });
    return nodes;
}

int
main(void)
{
    int failures = 0;
    bool needs_grid = false;

    /* A grid that fits: fusing is safe, so the two programs become one launch,
     * and that launch must be flagged as needing every block resident. */
    {
        size_t nodes = run_device_chain(/* grid */ 64, /* coresident */ 128, needs_grid);
        if (nodes != 1)
        {
            fprintf(stderr, "FAIL: a device chain whose grid fits (64 of 128 blocks) "
                            "produced %zu node(s); it should fuse into 1\n", nodes);
            ++failures;
        }
        else if (!needs_grid)
        {
            fprintf(stderr, "FAIL: the fused device program is not marked as needing a "
                            "co-resident grid, so a driver would launch it normally and "
                            "its grid-wide barrier would hang\n");
            ++failures;
        }
        else
            fprintf(stdout, "PASS: a device chain whose grid fits fused into one "
                            "cooperative launch\n");
    }

    /* A grid that does not fit: the barrier could not complete, so the chain must
     * stay two launches. */
    {
        size_t nodes = run_device_chain(/* grid */ 4096, /* coresident */ 128, needs_grid);
        if (nodes != 2)
        {
            fprintf(stderr, "FAIL: a device chain whose grid does NOT fit (4096 of 128 "
                            "blocks) produced %zu node(s); it must stay 2, or its "
                            "grid-wide barrier hangs the device\n", nodes);
            ++failures;
        }
        else
            fprintf(stdout, "PASS: a device chain too large to be co-resident left "
                            "unfused\n");
    }

    /* No residency figure: nothing is known, so nothing may be assumed. */
    {
        size_t nodes = run_device_chain(/* grid */ 64, /* coresident */ 0, needs_grid);
        if (nodes != 2)
        {
            fprintf(stderr, "FAIL: a device chain with no residency figure produced %zu "
                            "node(s); it must stay 2\n", nodes);
            ++failures;
        }
        else
            fprintf(stdout, "PASS: a device chain with no residency figure left "
                            "unfused\n");
    }

    if (failures)
        fprintf(stderr, "%d device-residency case(s) failed\n", failures);
    return failures ? 1 : 0;
}
