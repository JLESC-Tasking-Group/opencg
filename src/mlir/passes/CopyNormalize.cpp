/*
** cg-copy-normalize : normalize 2-D copies.
** Mirrors the legacy command_graph_t::pass_copy_normalize.
**
**   - If a 2-D copy is contiguous (m == src_ld == dst_ld), rewrite it to a 1-D
**     copy of size m*n*sizeof_type.
**   - Otherwise normalize it to sizeof_type == 1 (scaling m by the old size).
*/

# include <cgir/mlir/cgir-mlir.hpp>

# include "mlir/IR/Builders.h"
# include "mlir/Pass/Pass.h"
# include "llvm/ADT/STLExtras.h"

# include <cgir/command-type.hpp>

# include <cstdint>

using namespace mlir;

namespace {

/* map a 2-D copy command_type_t to its 1-D counterpart */
static int32_t
copy_2d_to_1d_kind(int32_t k)
{
    switch (k)
    {
        case ::cgir::COMMAND_TYPE_COPY_H2H_2D: return ::cgir::COMMAND_TYPE_COPY_H2H_1D;
        case ::cgir::COMMAND_TYPE_COPY_H2D_2D: return ::cgir::COMMAND_TYPE_COPY_H2D_1D;
        case ::cgir::COMMAND_TYPE_COPY_D2H_2D: return ::cgir::COMMAND_TYPE_COPY_D2H_1D;
        case ::cgir::COMMAND_TYPE_COPY_D2D_2D: return ::cgir::COMMAND_TYPE_COPY_D2D_1D;
        default:                              return k;
    }
}

struct CopyNormalizePass
    : public PassWrapper<CopyNormalizePass, OperationPass<::cgir::cg::GraphOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyNormalizePass)

    StringRef getArgument() const final { return "cg-copy-normalize"; }
    StringRef getDescription() const final
    {
        return "Normalize 2-D copies (contiguous -> 1-D, else sizeof_type=1)";
    }

    void runOnOperation() override
    {
        using namespace ::cgir::cg;

        GraphOp graph = getOperation();
        Block & body = graph.getBodyBlock();

        for (Operation & opref : llvm::make_early_inc_range(body))
        {
            Copy2DOp c = dyn_cast<Copy2DOp>(&opref);
            if (!c)
                continue;

            const uint64_t m = c.getM();
            const uint64_t n = c.getN();
            const uint64_t s = c.getSizeofType();

            if (m == c.getSrcLd() && m == c.getDstLd())
            {
                /* contiguous: convert to a 1-D copy */
                OpBuilder b(c);
                OperationState st(c.getLoc(), Copy1DOp::getOperationName());
                st.addTypes({c.getToken().getType()});
                st.addOperands(c->getOperands());
                st.addAttribute("duid",     b.getI32IntegerAttr((int32_t) c.getDuid()));
                st.addAttribute("kind",     b.getI32IntegerAttr(copy_2d_to_1d_kind((int32_t) c.getKind())));
                st.addAttribute("src_duid", b.getI32IntegerAttr((int32_t) c.getSrcDuid()));
                st.addAttribute("dst_duid", b.getI32IntegerAttr((int32_t) c.getDstDuid()));
                st.addAttribute("src_addr", b.getI64IntegerAttr((int64_t) c.getSrcAddr()));
                st.addAttribute("dst_addr", b.getI64IntegerAttr((int64_t) c.getDstAddr()));
                st.addAttribute("size",     b.getI64IntegerAttr((int64_t) (m * n * s)));
                Operation * node_1d = b.create(st);

                /* preserve the originating POD node, then replace */
                if (::cgir::command_graph_node_t * sn = get_src_node(c))
                    set_src_node(node_1d, sn);

                c.getToken().replaceAllUsesWith(node_1d->getResult(0));
                c->erase();
            }
            else
            {
                /* normalize to byte units: scale the column extent AND both
                 * leading dimensions by the element size, otherwise the row
                 * strides would lose the 'sizeof_type' factor. */
                c.setSrcLd(c.getSrcLd() * s);
                c.setDstLd(c.getDstLd() * s);
                c.setM(m * s);
                c.setSizeofType(1);
            }
        }
    }
};

} // anonymous namespace

std::unique_ptr<Pass>
cgir::cg::create_copy_normalize_pass(void)
{
    return std::make_unique<CopyNormalizePass>();
}
