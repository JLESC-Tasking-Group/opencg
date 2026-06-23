/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** cg-prog-fuse : fuse adjacent LLVM-IR programs.
** Mirrors the legacy command_graph_t::pass_prog_fuse.
**
** A u -> v series of PROG commands (u with a single successor v, v with a
** single predecessor u), both with an LLVM-IR source, is fused: the two LLVM
** modules are linked and JIT-compiled into one (delegated to the shared
** command_graph_prog_fuse_llvmir core, which also backs the legacy pass), then
** v is contracted away.
**
** PROG commands are carried in the `cg` dialect as cg.generic ops (an opaque
** command blob). The actual fusion operates on the originating POD command
** (reached via the cg.src_node attribute), so this pass requires ops imported
** from a POD command graph; cg.generic PROG ops without a source node are
** skipped. Non-LLVM-IR program pairs are also skipped (left unfused).
**
** This software is governed by the CeCILL-C license. See the LICENSE file.
**/

# include <opencg/mlir/opencg-mlir.hpp>

# include "passes/prog-fuse.hpp"

# include <mlir/Pass/Pass.h>
# include <llvm/ADT/SmallPtrSet.h>
# include <llvm/ADT/SmallVector.h>

using namespace mlir;

namespace {

using ::ocg::cg::GenericOp;
using ::ocg::cg::get_src_node;

/* Return the originating POD node iff `op` is an imported PROG command. */
static ::ocg::command_graph_node_t *
prog_src_node(GenericOp op)
{
    ::ocg::command_graph_node_t * node = get_src_node(op.getOperation());
    if (node == nullptr)
        return nullptr;
    if (node->type != ::ocg::COMMAND_GRAPH_NODE_TYPE_COMMAND)
        return nullptr;
    if (node->command == nullptr)
        return nullptr;
    if (node->command->type != ::ocg::COMMAND_TYPE_PROG)
        return nullptr;
    return node;
}

/* u -> v series: u has a single successor (v), v has a single predecessor (u). */
static bool
is_series(Operation * u, Operation * v)
{
    Value ut = u->getResult(0);
    if (!ut.hasOneUse())
        return false;
    if (*ut.user_begin() != v)
        return false;
    if (v->getNumOperands() != 1 || v->getOperand(0) != ut)
        return false;
    return true;
}

struct ProgFusePass
    : public PassWrapper<ProgFusePass, OperationPass<::ocg::cg::GraphOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProgFusePass)

    StringRef getArgument() const final { return "cg-prog-fuse"; }
    StringRef getDescription() const final
    {
        return "Fuse adjacent LLVM-IR program commands";
    }

    void runOnOperation() override
    {
        using namespace ::ocg::cg;

        GraphOp graph = getOperation();
        Block & body = graph.getBodyBlock();

        const uint32_t prog_kind = (uint32_t) ::ocg::COMMAND_TYPE_PROG;

        /* snapshot PROG ops; v's are erased lazily and tracked in `dead` */
        llvm::SmallVector<Operation *> work;
        for (Operation & o : body)
            if (GenericOp g = dyn_cast<GenericOp>(&o))
                if (g.getKind() == prog_kind)
                    work.push_back(&o);

        llvm::SmallPtrSet<Operation *, 16> dead;

        for (Operation * uop : work)
        {
            if (dead.count(uop))
                continue;

            GenericOp u = cast<GenericOp>(uop);
            ::ocg::command_graph_node_t * un = prog_src_node(u);
            if (un == nullptr)
                continue;

            /* greedily fuse the chain u -> v -> ... starting at u */
            bool changed = true;
            while (changed)
            {
                changed = false;

                Value ut = u.getToken();
                if (!ut.hasOneUse())
                    break;

                Operation * user = *ut.user_begin();
                if (dead.count(user))
                    break;

                GenericOp v = dyn_cast<GenericOp>(user);
                if (!v || v.getKind() != prog_kind)
                    break;

                ::ocg::command_graph_node_t * vn = prog_src_node(v);
                if (vn == nullptr)
                    break;

                if (!is_series(u.getOperation(), v.getOperation()))
                    break;

                /* only LLVM-IR programs can be fused */
                if (un->command->prog.source.type != ::ocg::COMMAND_PROG_SOURCE_TYPE_LLVMIR ||
                    vn->command->prog.source.type != ::ocg::COMMAND_PROG_SOURCE_TYPE_LLVMIR)
                    break;

                /* fuse v into u (in place on u's command) */
                ::ocg::command_graph_prog_fuse_llvmir(
                    &un->command->prog,
                    &vn->command->prog,
                    &un->command->prog
                );

                /* contract the series: v's successors now depend on u */
                dead.insert(v.getOperation());
                v.getToken().replaceAllUsesWith(u.getToken());
                v->erase();

                changed = true;
            }
        }
    }
};

} /* anonymous namespace */

std::unique_ptr<Pass>
ocg::cg::create_prog_fuse_pass(void)
{
    return std::make_unique<ProgFusePass>();
}
