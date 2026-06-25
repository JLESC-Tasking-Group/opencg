/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** cg-prog-fuse : fuse chains of adjacent LLVM-IR programs.
** Mirrors the legacy command_graph_t::pass_prog_fuse.
**
** A maximal chain u -> v -> w -> ... of PROG commands in series (each link: a
** single successor / single predecessor), all with an LLVM-IR source, is fused:
** all the LLVM modules are linked and JIT-compiled into ONE wrapper (delegated
** to the shared N-ary command_graph_prog_fuse_llvmir core, which also backs the
** legacy pass), then the rest of the chain is contracted away into u.
**
** PROG commands are carried in the `cg` dialect as cg.generic ops (an opaque
** command blob). The actual fusion operates on the originating POD command
** (reached via the cg.src_node attribute), so this pass requires ops imported
** from a POD command graph; cg.generic PROG ops without a source node are
** skipped. Non-LLVM-IR programs are also skipped (left unfused).
**
** This software is governed by the CeCILL-C license. See the LICENSE file.
**/

# include <cgir/mlir/cgir-mlir.hpp>

# include "prog-fuse-llvmir.hpp"

# include <mlir/Pass/Pass.h>
# include <llvm/ADT/SmallPtrSet.h>
# include <llvm/ADT/SmallVector.h>

using namespace mlir;

namespace {

using ::cgir::cg::GenericOp;
using ::cgir::cg::get_src_node;

/* Return the originating POD node iff `op` is an imported PROG command. */
static ::cgir::command_graph_node_t *
prog_src_node(GenericOp op)
{
    ::cgir::command_graph_node_t * node = get_src_node(op.getOperation());
    if (node == nullptr)
        return nullptr;
    if (node->type != ::cgir::COMMAND_GRAPH_NODE_TYPE_COMMAND)
        return nullptr;
    if (node->command == nullptr)
        return nullptr;
    if (node->command->type != ::cgir::COMMAND_TYPE_PROG)
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
    : public PassWrapper<ProgFusePass, OperationPass<::cgir::cg::GraphOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProgFusePass)

    StringRef getArgument() const final { return "cg-prog-fuse"; }
    StringRef getDescription() const final
    {
        return "Fuse adjacent LLVM-IR program commands";
    }

    void runOnOperation() override
    {
        using namespace ::cgir::cg;

        GraphOp graph = getOperation();
        Block & body = graph.getBodyBlock();

        const uint32_t prog_kind = (uint32_t) ::cgir::COMMAND_TYPE_PROG;
        const auto LLVMIR = ::cgir::COMMAND_PROG_SOURCE_TYPE_LLVMIR;

        /* snapshot PROG ops; chain members are erased lazily, tracked in `dead` */
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
            ::cgir::command_graph_node_t * un = prog_src_node(u);
            if (un == nullptr || un->command->prog.source.type != LLVMIR)
                continue;

            /* collect the maximal chain u -> v -> w -> ... of fusable
             * LLVM-IR programs in series */
            llvm::SmallVector<Operation *> chain_ops;
            llvm::SmallVector<::cgir::command_graph_node_t *> chain_nodes;
            chain_ops.push_back(uop);
            chain_nodes.push_back(un);

            Operation * cur = uop;
            ::cgir::command_graph_node_t * cur_node = un;
            for (;;)
            {
                Value ct = cur->getResult(0);
                if (!ct.hasOneUse())
                    break;

                Operation * w = *ct.user_begin();
                if (dead.count(w))
                    break;

                GenericOp wg = dyn_cast<GenericOp>(w);
                if (!wg || wg.getKind() != prog_kind)
                    break;

                ::cgir::command_graph_node_t * wn = prog_src_node(wg);
                if (wn == nullptr)
                    break;
                if (!is_series(cur, w))
                    break;
                if (cur_node->command->prog.source.type != LLVMIR ||
                    wn->command->prog.source.type       != LLVMIR)
                    break;

                chain_ops.push_back(w);
                chain_nodes.push_back(wn);
                cur = w;
                cur_node = wn;
            }

            if (chain_ops.size() < 2)
                continue;

            /* fuse the whole chain into u's command (N-ary, single wrapper) */
            llvm::SmallVector<::cgir::command_prog_t *> progs;
            progs.reserve(chain_nodes.size());
            for (::cgir::command_graph_node_t * cn : chain_nodes)
                progs.push_back(&cn->command->prog);

            ::cgir::command_graph_prog_fuse_llvmir(progs.data(), progs.size(), &un->command->prog);

            /* contract the chain into u: u's token takes over the chain tail's
             * successors, and the rest of the chain is erased (back to front) */
            Value u_tok = u.getToken();
            chain_ops.back()->getResult(0).replaceAllUsesWith(u_tok);
            for (size_t k = chain_ops.size(); k-- > 1; )
            {
                dead.insert(chain_ops[k]);
                chain_ops[k]->erase();
            }
        }
    }
};

} /* anonymous namespace */

std::unique_ptr<Pass>
cgir::cg::create_prog_fuse_pass(void)
{
    return std::make_unique<ProgFusePass>();
}
