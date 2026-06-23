/*
** cg-reduce-edge : transitive reduction of the command graph.
** Mirrors the legacy command_graph_t::pass_reduce_edge.
**
** An edge u -> v is removed when v is reachable from another successor w of u.
** Reachability is computed once over the original graph (reverse-topological
** accumulation), then redundant edges are dropped.
*/

# include <opencg/mlir/opencg-mlir.hpp>

# include "mlir/Pass/Pass.h"
# include "llvm/ADT/DenseMap.h"
# include "llvm/ADT/SmallPtrSet.h"
# include "llvm/ADT/SmallVector.h"

# include <opencg/bitset2d.hpp>

# include <vector>

using namespace mlir;

namespace {

struct ReduceEdgePass
    : public PassWrapper<ReduceEdgePass, OperationPass<::ocg::cg::GraphOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceEdgePass)

    StringRef getArgument() const final { return "cg-reduce-edge"; }
    StringRef getDescription() const final
    {
        return "Transitive reduction of the command graph";
    }

    void runOnOperation() override
    {
        using namespace ::ocg::cg;

        GraphOp graph = getOperation();
        Block & body = graph.getBodyBlock();

        /* index ops */
        llvm::SmallVector<Operation *> ops;
        llvm::DenseMap<Operation *, int> idx;
        for (Operation & o : body)
        {
            idx[&o] = (int) ops.size();
            ops.push_back(&o);
        }
        const int n = (int) ops.size();
        if (n == 0)
            return;

        /* successor adjacency (deduplicated) + predecessor in-degree */
        std::vector<llvm::SmallVector<int>> succ(n);
        std::vector<int> indeg(n, 0);

        for (int i = 0; i < n; ++i)
        {
            Operation * u = ops[i];
            llvm::SmallPtrSet<Operation *, 8> seen;
            for (Operation * v : u->getResult(0).getUsers())
            {
                if (!seen.insert(v).second)
                    continue;
                auto it = idx.find(v);
                if (it == idx.end())
                    continue;
                succ[i].push_back(it->second);
            }
        }
        for (int j = 0; j < n; ++j)
        {
            Operation * v = ops[j];
            llvm::SmallPtrSet<Operation *, 8> seen;
            for (Value pv : v->getOperands())
            {
                Operation * d = pv.getDefiningOp();
                if (!d)
                    continue;
                if (!seen.insert(d).second)
                    continue;
                if (idx.find(d) != idx.end())
                    ++indeg[j];
            }
        }

        /* Kahn topological order */
        std::vector<int> topo;
        topo.reserve(n);
        {
            std::vector<int> deg = indeg;
            llvm::SmallVector<int> queue;
            for (int i = 0; i < n; ++i)
                if (deg[i] == 0)
                    queue.push_back(i);
            while (!queue.empty())
            {
                int u = queue.back();
                queue.pop_back();
                topo.push_back(u);
                for (int w : succ[u])
                    if (--deg[w] == 0)
                        queue.push_back(w);
            }
        }

        /* reachability r[i][j] = (j reachable from i), inclusive of i.
         * Accumulate in reverse-topological order so successors are done first. */
        bitset2d_t<uint64_t, int> r(n);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            const int i = *it;
            r.set(i, i);
            for (int s : succ[i])
                r.or_rows(i, s);
        }

        /* drop redundant edges u -> v */
        for (int i = 0; i < n; ++i)
        {
            Operation * u = ops[i];
            Value tok = u->getResult(0);

            for (int vi : succ[i])
            {
                bool redundant = false;
                for (int wi : succ[i])
                {
                    if (wi == vi)
                        continue;
                    if (r.test(wi, vi))
                    {
                        redundant = true;
                        break;
                    }
                }
                if (!redundant)
                    continue;

                /* remove first occurrence of u's token in v's operands */
                Operation * v = ops[vi];
                for (unsigned k = 0; k < v->getNumOperands(); ++k)
                {
                    if (v->getOperand(k) == tok)
                    {
                        v->eraseOperand(k);
                        break;
                    }
                }
            }
        }
    }
};

} // anonymous namespace

std::unique_ptr<Pass>
ocg::cg::create_reduce_edge_pass(void)
{
    return std::make_unique<ReduceEdgePass>();
}
