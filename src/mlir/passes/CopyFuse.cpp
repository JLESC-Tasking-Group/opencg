/*
** cg-copy-fuse : fuse false-twin contiguous 1-D copies.
** Mirrors the legacy command_graph_t::pass_copy_fuse (1-D case).
**
** Two false-twin 1-D copies (same predecessors and successors) with the same
** src/dst device, the same src->dst address bias, and overlapping/adjacent
** source segments are merged into a single 1-D copy spanning their union.
**
** NB: the legacy pass aborts on 2-D twin fusion (a TODO); here we simply skip
** non-1-D fusions rather than abort.
*/

# include <cgir/mlir/cgir-mlir.hpp>

# include "mlir/Pass/Pass.h"
# include "llvm/ADT/STLExtras.h"
# include "llvm/ADT/SmallPtrSet.h"
# include "llvm/ADT/SmallVector.h"

# include <algorithm>
# include <cstdint>

using namespace mlir;

namespace {

using ::cgir::cg::Copy1DOp;

/* false twins: identical predecessor set (operands) and successor set (users) */
static bool
false_twins(Operation * u, Operation * v)
{
    llvm::SmallPtrSet<Value, 8> up(u->getOperands().begin(), u->getOperands().end());
    llvm::SmallPtrSet<Value, 8> vp(v->getOperands().begin(), v->getOperands().end());
    if (up.size() != vp.size())
        return false;
    for (Value x : up)
        if (!vp.count(x))
            return false;

    llvm::SmallPtrSet<Operation *, 8> us(u->getResult(0).user_begin(), u->getResult(0).user_end());
    llvm::SmallPtrSet<Operation *, 8> vs(v->getResult(0).user_begin(), v->getResult(0).user_end());
    if (us.size() != vs.size())
        return false;
    for (Operation * x : us)
        if (!vs.count(x))
            return false;

    return true;
}

/* try to fuse v into u (u is modified in place); v's token is disconnected from
 * its successors but v is NOT erased here (the caller defers erasure). */
static bool
try_fuse(Copy1DOp u, Copy1DOp v)
{
    if (u.getOperation() == v.getOperation())
        return false;
    if (!false_twins(u.getOperation(), v.getOperation()))
        return false;
    if (u.getSrcDuid() != v.getSrcDuid() || u.getDstDuid() != v.getDstDuid())
        return false;

    const uint64_t ubias = u.getSrcAddr() - u.getDstAddr();
    const uint64_t vbias = v.getSrcAddr() - v.getDstAddr();
    if (ubias != vbias)
        return false;

    const uint64_t ub = u.getSrcAddr();
    const uint64_t ue = u.getSrcAddr() + u.getSize();
    const uint64_t vb = v.getSrcAddr();
    const uint64_t ve = v.getSrcAddr() + v.getSize();

    /* require overlap or adjacency */
    if (!(ub <= ve && vb <= ue))
        return false;

    const uint64_t sb = std::min(ub, vb);
    const uint64_t se = std::max(ue, ve);

    u.setSrcAddr(sb);
    u.setDstAddr(sb - ubias);
    u.setSize(se - sb);

    /* disconnect v's token from its (shared) successors; they keep u's token */
    Value vt = v.getToken();
    llvm::SmallVector<Operation *> vusers(vt.getUsers().begin(), vt.getUsers().end());
    for (Operation * user : vusers)
    {
        for (int k = (int) user->getNumOperands() - 1; k >= 0; --k)
            if (user->getOperand((unsigned) k) == vt)
                user->eraseOperand((unsigned) k);
    }

    return true;
}

struct CopyFusePass
    : public PassWrapper<CopyFusePass, OperationPass<::cgir::cg::GraphOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyFusePass)

    StringRef getArgument() const final { return "cg-copy-fuse"; }
    StringRef getDescription() const final
    {
        return "Fuse false-twin contiguous 1-D copies";
    }

    void runOnOperation() override
    {
        using namespace ::cgir::cg;

        GraphOp graph = getOperation();
        Block & body = graph.getBodyBlock();

        /* snapshot candidate ops; defer all erasure to the end */
        llvm::SmallVector<Operation *> work;
        for (Operation & o : body)
            if (isa<Copy1DOp>(&o))
                work.push_back(&o);

        llvm::SmallPtrSet<Operation *, 16> dead;

        for (Operation * uop : work)
        {
            if (dead.count(uop))
                continue;
            Copy1DOp u = cast<Copy1DOp>(uop);

            bool changed = true;
            while (changed)
            {
                changed = false;
                if (u->getNumOperands() == 0)
                    break;

                /* siblings share u's first predecessor */
                Value p0 = u->getOperand(0);
                llvm::SmallVector<Operation *> sibs(p0.getUsers().begin(), p0.getUsers().end());
                for (Operation * sop : sibs)
                {
                    if (sop == uop || dead.count(sop))
                        continue;
                    Copy1DOp v = dyn_cast<Copy1DOp>(sop);
                    if (!v)
                        continue;
                    if (try_fuse(u, v))
                    {
                        dead.insert(sop);
                        changed = true;
                        break;
                    }
                }
            }
        }

        for (Operation * d : dead)
            d->erase();
    }
};

} // anonymous namespace

std::unique_ptr<Pass>
cgir::cg::create_copy_fuse_pass(void)
{
    return std::make_unique<CopyFusePass>();
}
