/*
** cg-reduce-node : remove control (empty) nodes that do not reduce graph
** complexity. Mirrors the legacy command_graph_t::pass_reduce_node.
**
** For an empty node with m predecessors and n successors, the node is removed
** (its predecessors connected directly to its successors) when keeping it does
** not pay off, i.e. when  m*n < 1 + m + n. Entry/exit nodes are never removed.
*/

#include "bridge.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct ReduceNodePass
    : public PassWrapper<ReduceNodePass, OperationPass<::ocg::cg::GraphOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceNodePass)

    StringRef getArgument() const final { return "cg-reduce-node"; }
    StringRef getDescription() const final
    {
        return "Remove control nodes that do not reduce graph complexity";
    }

    void runOnOperation() override
    {
        using namespace ::ocg::cg;

        GraphOp graph = getOperation();
        Block & body = graph.getBodyBlock();

        for (Operation & opref : llvm::make_early_inc_range(body))
        {
            EmptyOp e = dyn_cast<EmptyOp>(&opref);
            if (!e)
                continue;
            if (e.getIsEntry() || e.getIsExit())
                continue;

            Value tok = e.getToken();

            /* m = predecessor edges (operands), n = successor edges (uses) */
            const size_t m = e->getNumOperands();
            size_t n = 0;
            for (Operation * u : tok.getUsers())
            {
                (void) u;
                ++n;
            }

            if (!(m * n < 1 + m + n))
                continue;

            /* snapshot the empty node's predecessor tokens */
            llvm::SmallVector<Value> preds(e->getOperands().begin(),
                                           e->getOperands().end());

            /* snapshot the unique successor ops */
            llvm::SmallPtrSet<Operation *, 8> seen;
            llvm::SmallVector<Operation *> users;
            for (Operation * u : tok.getUsers())
                if (seen.insert(u).second)
                    users.push_back(u);

            /* for each successor, drop the edge to e and connect e's preds */
            for (Operation * user : users)
            {
                llvm::SetVector<Value> newops;
                for (Value v : user->getOperands())
                    if (v != tok)
                        newops.insert(v);
                for (Value p : preds)
                    newops.insert(p);
                user->setOperands(newops.takeVector());
            }

            e->erase();
        }
    }
};

} // anonymous namespace

std::unique_ptr<Pass>
ocg::cg::createReduceNodePass()
{
    return std::make_unique<ReduceNodePass>();
}
