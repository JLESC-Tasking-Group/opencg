/*
** OpenCG -> MLIR import.
**
** Builds a `cg` dialect module that is structurally isomorphic to a POD
** command graph: one op per node, edges encoded as !cg.token operands.
*/

# include <opencg/mlir/opencg-mlir.hpp>

# include "mlir/IR/Builders.h"
# include "mlir/IR/BuiltinAttributes.h"
# include "mlir/IR/BuiltinOps.h"

# include <opencg/command.hpp>
# include <opencg/command-graph.hpp>
# include <opencg/command-type.hpp>

# include <cstdint>
# include <cstdio>
# include <cstdlib>
# include <vector>

using namespace mlir;

namespace ocg {
namespace cg {

namespace {

/* Create the cg op mirroring a single POD node. Edges are wired in a later
 * pass; here the op is created with no operands. */
static Operation *
create_node_op(
    OpBuilder & b,
    Location loc,
    Type tok,
    command_graph_node_t * node,
    bool is_entry,
    bool is_exit
) {
    const int32_t duid = (int32_t) (uint32_t) node->device_unique_id;

    /* ---- control (empty) node ---- */
    if (node->type == COMMAND_GRAPH_NODE_TYPE_EMPTY)
    {
        OperationState st(loc, EmptyOp::getOperationName());
        st.addTypes({tok});
        st.addAttribute("duid", b.getI32IntegerAttr(duid));
        if (is_entry) st.addAttribute("is_entry", b.getUnitAttr());
        if (is_exit)  st.addAttribute("is_exit",  b.getUnitAttr());
        return b.create(st);
    }

    /* ---- command node ---- */
    if (node->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
    {
        command_t * cmd = node->command;
        assert(cmd);

        switch (cmd->type)
        {
            case COMMAND_TYPE_COPY_H2H_1D:
            case COMMAND_TYPE_COPY_H2D_1D:
            case COMMAND_TYPE_COPY_D2H_1D:
            case COMMAND_TYPE_COPY_D2D_1D:
            {
                OperationState st(loc, Copy1DOp::getOperationName());
                st.addTypes({tok});
                st.addAttribute("duid",     b.getI32IntegerAttr(duid));
                st.addAttribute("kind",     b.getI32IntegerAttr((int32_t) cmd->type));
                st.addAttribute("src_duid", b.getI32IntegerAttr((int32_t) (uint32_t) cmd->copy_1D.src_device_unique_id));
                st.addAttribute("dst_duid", b.getI32IntegerAttr((int32_t) (uint32_t) cmd->copy_1D.dst_device_unique_id));
                st.addAttribute("src_addr", b.getI64IntegerAttr((int64_t) cmd->copy_1D.src_device_addr));
                st.addAttribute("dst_addr", b.getI64IntegerAttr((int64_t) cmd->copy_1D.dst_device_addr));
                st.addAttribute("size",     b.getI64IntegerAttr((int64_t) cmd->copy_1D.size));
                return b.create(st);
            }

            case COMMAND_TYPE_COPY_H2H_2D:
            case COMMAND_TYPE_COPY_H2D_2D:
            case COMMAND_TYPE_COPY_D2H_2D:
            case COMMAND_TYPE_COPY_D2D_2D:
            {
                OperationState st(loc, Copy2DOp::getOperationName());
                st.addTypes({tok});
                st.addAttribute("duid",        b.getI32IntegerAttr(duid));
                st.addAttribute("kind",        b.getI32IntegerAttr((int32_t) cmd->type));
                st.addAttribute("src_duid",    b.getI32IntegerAttr((int32_t) (uint32_t) cmd->copy_2D.src_device_unique_id));
                st.addAttribute("dst_duid",    b.getI32IntegerAttr((int32_t) (uint32_t) cmd->copy_2D.dst_device_unique_id));
                st.addAttribute("src_addr",    b.getI64IntegerAttr((int64_t) cmd->copy_2D.src_addr));
                st.addAttribute("src_ld",      b.getI64IntegerAttr((int64_t) cmd->copy_2D.src_ld));
                st.addAttribute("dst_addr",    b.getI64IntegerAttr((int64_t) cmd->copy_2D.dst_addr));
                st.addAttribute("dst_ld",      b.getI64IntegerAttr((int64_t) cmd->copy_2D.dst_ld));
                st.addAttribute("m",           b.getI64IntegerAttr((int64_t) cmd->copy_2D.m));
                st.addAttribute("n",           b.getI64IntegerAttr((int64_t) cmd->copy_2D.n));
                st.addAttribute("sizeof_type", b.getI64IntegerAttr((int64_t) cmd->copy_2D.sizeof_type));
                return b.create(st);
            }

            default:
                break;
        }

        /* opaque carrier: store the raw command bytes (base subobject) */
        OperationState st(loc, GenericOp::getOperationName());
        st.addTypes({tok});
        st.addAttribute("duid", b.getI32IntegerAttr(duid));
        st.addAttribute("kind", b.getI32IntegerAttr((int32_t) cmd->type));
        llvm::ArrayRef<int8_t> bytes(
            reinterpret_cast<const int8_t *>(cmd), sizeof(::ocg::command_t));
        st.addAttribute("blob", DenseI8ArrayAttr::get(b.getContext(), bytes));
        return b.create(st);
    }

    /* COMMAND_GRAPH / CONDITION nodes are not modeled yet (Phase 2+). */
    fprintf(stderr, "opencg/mlir: unsupported node type %d during import\n", (int) node->type);
    abort();
}

} // anonymous namespace

GraphOp
get_graph(ModuleOp module)
{
    for (Operation & op : module.getBody()->getOperations())
        if (auto g = dyn_cast<GraphOp>(&op))
            return g;
    return GraphOp();
}

command_graph_node_t *
get_src_node(Operation * op)
{
    if (auto a = op->getAttrOfType<IntegerAttr>(kSrcNodeAttr))
        return reinterpret_cast<command_graph_node_t *>((uintptr_t) a.getInt());
    return nullptr;
}

void
set_src_node(Operation * op, command_graph_node_t * node)
{
    OpBuilder b(op->getContext());
    op->setAttr(kSrcNodeAttr, b.getI64IntegerAttr((int64_t) (uintptr_t) node));
}

OwningOpRef<ModuleOp>
import_command_graph(
    MLIRContext & ctx,
    command_graph_t * cg
) {
    OpBuilder b(&ctx);
    Location loc = b.getUnknownLoc();

    OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
    b.setInsertionPointToStart(module->getBody());

    /* create the graph container with a single (graph) block */
    OperationState gstate(loc, GraphOp::getOperationName());
    Region * region = gstate.addRegion();
    region->emplaceBlock();
    Operation * graph_op = b.create(gstate);
    Block & body = graph_op->getRegion(0).front();
    b.setInsertionPointToStart(&body);

    Type tok = TokenType::get(&ctx);

    /* enumerate POD nodes (entry/exit included), assigning iterator_index */
    struct pls_t {};
    std::vector<command_graph_t::node_iterator_t<pls_t>> nodes =
        cg->create_node_iterators<pls_t, /*include_entry_exit*/ true>();
    const size_t n = nodes.size();

    command_graph_node_t * entry = cg->node_get_entry();
    command_graph_node_t * exit  = cg->node_get_exit();

    std::vector<Operation *> ops(n, nullptr);

    /* pass 1: one op per node */
    for (size_t i = 0; i < n; ++i)
    {
        command_graph_node_t * node = nodes[i].node;
        Operation * op = create_node_op(b, loc, tok, node, node == entry, node == exit);
        assert(node->iterator_index < n);
        ops[node->iterator_index] = op;
        set_src_node(op, node);
    }

    /* pass 2: wire predecessor tokens (order-independent: graph region) */
    for (size_t i = 0; i < n; ++i)
    {
        command_graph_node_t * node = nodes[i].node;
        Operation * op = ops[node->iterator_index];

        llvm::SmallVector<Value> preds;
        for (command_graph_node_t * p : node->predecessors)
        {
            assert(p->iterator_index < n);
            Operation * pop = ops[p->iterator_index];
            assert(pop);
            preds.push_back(pop->getResult(0));
        }
        op->setOperands(preds);
    }

    return module;
}

} // namespace cg
} // namespace ocg
