/*
** cg-batch : greedy monochromatic contraction into cg.batch sub-graphs.
** Mirrors the legacy command_graph_t::pass_batch.
**
** Two passes:
**   1. Greedy grouping on a scratch graph: starting from every candidate node
**      (non entry/exit), repeatedly identify a same-device neighbor that is a
**      false-twin or forms a series (u->v with u single-succ / v single-pred,
**      or symmetrically), contracting them. This yields groups of >= 2 ops.
**   2. Materialization: each group is contracted into a cg.batch op whose region
**      holds the member ops (with their internal token edges); external
**      precedence enters via the batch op operands and leaves via its token.
**
** Deviations from legacy (documented): entry/exit are never merged; 2-D/1-D
** specifics are irrelevant here (batching is type-agnostic, device-based).
*/

# include <opencg/mlir/opencg-mlir.hpp>

# include "mlir/IR/Builders.h"
# include "mlir/IR/BuiltinAttributes.h"
# include "mlir/Pass/Pass.h"
# include "llvm/ADT/DenseMap.h"
# include "llvm/ADT/STLExtras.h"
# include "llvm/ADT/SetVector.h"
# include "llvm/ADT/SmallPtrSet.h"
# include "llvm/ADT/SmallVector.h"

# include <cstdint>
# include <deque>
# include <set>
# include <vector>

using namespace mlir;

namespace {

using ::ocg::cg::BatchOp;
using ::ocg::cg::EmptyOp;

static inline uint32_t
get_duid(Operation * op)
{
    return (uint32_t) op->getAttrOfType<IntegerAttr>("duid").getInt();
}

static inline bool
is_entry_exit(Operation * op)
{
    if (auto e = dyn_cast<EmptyOp>(op))
        return e.getIsEntry() || e.getIsExit();
    return false;
}

/* Materialize one group (>= 2 member ops, all same device) into a cg.batch op. */
static void
materialize_batch(
    Block & body,
    llvm::ArrayRef<Operation *> members,
    uint32_t device
) {
    MLIRContext * ctx = body.getParentOp()->getContext();
    OpBuilder b(ctx);
    Location loc = members.front()->getLoc();
    Type tok = members.front()->getResult(0).getType();

    llvm::SmallPtrSet<Operation *, 16> group(members.begin(), members.end());

    /* external predecessor tokens (defined outside the group), deduplicated */
    llvm::SetVector<Value> ext_preds;
    for (Operation * m : members)
        for (Value t : m->getOperands())
        {
            Operation * d = t.getDefiningOp();
            if (!d || !group.count(d))
                ext_preds.insert(t);
        }

    /* create the batch op at the end of the parent block */
    b.setInsertionPointToEnd(&body);
    OperationState st(loc, BatchOp::getOperationName());
    st.addTypes({tok});
    st.addOperands(ext_preds.getArrayRef());
    st.addAttribute("duid", b.getI32IntegerAttr((int32_t) device));
    Region * region = st.addRegion();
    region->emplaceBlock();
    Operation * batch_op = b.create(st);
    Value batch_token = batch_op->getResult(0);

    /* external successors of members now depend on the batch token */
    for (Operation * m : members)
    {
        Value mt = m->getResult(0);
        for (OpOperand & use : llvm::make_early_inc_range(mt.getUses()))
        {
            Operation * owner = use.getOwner();
            if (owner != batch_op && !group.count(owner))
                use.set(batch_token);
        }
    }

    /* move members into the batch region */
    Block & batch_block = batch_op->getRegion(0).front();
    for (Operation * m : members)
        m->moveBefore(&batch_block, batch_block.end());

    /* strip external predecessor operands from members (carried by the batch) */
    for (Operation * m : members)
    {
        llvm::SmallVector<Value> keep;
        for (Value t : m->getOperands())
        {
            Operation * d = t.getDefiningOp();
            if (d && group.count(d))
                keep.push_back(t);
        }
        m->setOperands(keep);
    }
}

struct BatchPass
    : public PassWrapper<BatchPass, OperationPass<::ocg::cg::GraphOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchPass)

    StringRef getArgument() const final { return "cg-batch"; }
    StringRef getDescription() const final
    {
        return "Contract same-device nodes into cg.batch sub-graphs";
    }

    void runOnOperation() override
    {
        using namespace ::ocg::cg;

        GraphOp graph = getOperation();
        Block & body = graph.getBodyBlock();

        /* ---- index ops ---- */
        std::vector<Operation *> ops;
        llvm::DenseMap<Operation *, int> id;
        for (Operation & o : body)
        {
            id[&o] = (int) ops.size();
            ops.push_back(&o);
        }
        const int n = (int) ops.size();
        if (n == 0)
            return;

        std::vector<uint32_t> dev(n);
        std::vector<char> cand(n), alive(n, 1);
        for (int i = 0; i < n; ++i)
        {
            dev[i]  = get_duid(ops[i]);
            cand[i] = !is_entry_exit(ops[i]);
        }

        /* ---- scratch adjacency (sets, deduplicated) ---- */
        std::vector<std::set<int>> preds(n), succs(n);
        for (int i = 0; i < n; ++i)
            for (Value t : ops[i]->getOperands())
            {
                Operation * d = t.getDefiningOp();
                if (!d)
                    continue;
                auto it = id.find(d);
                if (it == id.end())
                    continue;
                preds[i].insert(it->second);
                succs[it->second].insert(i);
            }

        std::vector<std::vector<Operation *>> members(n);
        for (int i = 0; i < n; ++i)
            members[i].push_back(ops[i]);

        auto can_batch = [&] (int a, int b) {
            return a != b && alive[a] && alive[b] && cand[a] && cand[b] && dev[a] == dev[b];
        };
        auto false_twins = [&] (int a, int b) {
            return preds[a] == preds[b] && succs[a] == succs[b];
        };
        auto are_seq = [&] (int a, int b) {
            return succs[a].size() == 1 && *succs[a].begin() == b &&
                   preds[b].size() == 1 && *preds[b].begin() == a;
        };
        auto merge = [&] (int a, int b) {
            /* absorb b into a */
            for (int x : preds[b])
            {
                if (x == a) continue;
                succs[x].erase(b);
                succs[x].insert(a);
                preds[a].insert(x);
            }
            for (int y : succs[b])
            {
                if (y == a) continue;
                preds[y].erase(b);
                preds[y].insert(a);
                succs[a].insert(y);
            }
            /* drop any edge between a and b (self-loop after contraction) */
            preds[a].erase(b);
            succs[a].erase(b);
            preds[a].erase(a);
            succs[a].erase(a);
            for (Operation * m : members[b])
                members[a].push_back(m);
            alive[b] = 0;
            preds[b].clear();
            succs[b].clear();
        };

        /* ---- greedy contraction ---- */
        std::deque<int> q;
        for (int i = 0; i < n; ++i)
            if (cand[i])
                q.push_back(i);

        while (!q.empty())
        {
            int i = q.front();
            q.pop_front();
            if (!alive[i] || !cand[i])
                continue;

            bool found = false;

            /* 1. false twins among siblings */
            for (int p : preds[i])
            {
                for (int v : succs[p])
                    if (can_batch(i, v) && false_twins(i, v))
                    {
                        merge(i, v);
                        found = true;
                        break;
                    }
                if (found)
                    break;
            }
            if (found) { q.push_back(i); continue; }

            /* 2. i -> v series */
            for (int v : succs[i])
                if (can_batch(i, v) && are_seq(i, v))
                {
                    merge(i, v);
                    found = true;
                    break;
                }
            if (found) { q.push_back(i); continue; }

            /* 3. v -> i series */
            for (int v : preds[i])
                if (can_batch(i, v) && are_seq(v, i))
                {
                    merge(i, v);
                    found = true;
                    break;
                }
            if (found) { q.push_back(i); continue; }
        }

        /* ---- collect groups: only batch groups with >= 2 COMMAND members ----
         * Mirrors legacy batch.cc, which forms a BATCH command only when
         * contracting two actual commands (cases a/b with control nodes do no
         * batching). Control/empty nodes only *bridge* commands inside a group
         * and are never batched on their own. This also guarantees the batch
         * node device is a real command device, never OCG_UNSPECIFIED (255),
         * avoiding XKRT's replay assertion on COMMAND-node devices. */
        std::vector<std::pair<std::vector<Operation *>, uint32_t>> groups;
        for (int i = 0; i < n; ++i)
        {
            if (!(alive[i] && cand[i] && members[i].size() >= 2))
                continue;

            int ncmd = 0;
            uint32_t gdev = dev[i];
            for (Operation * m : members[i])
                if (!isa<EmptyOp>(m))
                {
                    if (ncmd == 0)
                        gdev = get_duid(m);   /* a real command device */
                    ++ncmd;
                }

            if (ncmd >= 2)
                groups.emplace_back(members[i], gdev);
        }

        /* ---- materialize ---- */
        for (auto & g : groups)
            materialize_batch(body, g.first, g.second);
    }
};

} // anonymous namespace

std::unique_ptr<Pass>
ocg::cg::create_batch_pass(void)
{
    return std::make_unique<BatchPass>();
}
