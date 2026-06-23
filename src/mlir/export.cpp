/*
** MLIR -> OpenCG export.
**
** Rebuilds the POD command graph topology from an (optimized) cg.graph,
** reusing original POD nodes/commands when an op still maps to one (preserving
** device, payload, command flags and runtime/replay state) and allocating new
** nodes/commands via the graph's allocator callbacks otherwise.
*/

#include "bridge.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <opencg/command.hpp>
#include <opencg/command-graph.hpp>
#include <opencg/command-type.hpp>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

using namespace mlir;

namespace ocg {
namespace cg {

namespace {

static inline uint8_t
get_duid(Operation * op)
{
    IntegerAttr a = op->getAttrOfType<IntegerAttr>("duid");
    assert(a);
    return (uint8_t) (uint32_t) a.getInt();
}

static inline bool
op_is_entry(Operation * op)
{
    if (auto e = dyn_cast<EmptyOp>(op))
        return e.getIsEntry();
    return false;
}

static inline bool
op_is_exit(Operation * op)
{
    if (auto e = dyn_cast<EmptyOp>(op))
        return e.getIsExit();
    return false;
}

/* Write op attributes back into a (reused) POD command, in case a pass changed
 * fields (e.g. copy-fuse rewriting addr/size). Topology is handled separately. */
static void
update_node_from_op(command_graph_node_t * node, Operation * op)
{
    node->device_unique_id = get_duid(op);

    if (auto c = dyn_cast<Copy1DOp>(op))
    {
        command_t * cmd = node->command;
        if (cmd)
        {
            cmd->type                          = (command_type_t) c.getKind();
            cmd->copy_1D.src_device_unique_id  = (device_unique_id_t) (uint32_t) c.getSrcDuid();
            cmd->copy_1D.dst_device_unique_id  = (device_unique_id_t) (uint32_t) c.getDstDuid();
            cmd->copy_1D.src_device_addr       = (uintptr_t) c.getSrcAddr();
            cmd->copy_1D.dst_device_addr       = (uintptr_t) c.getDstAddr();
            cmd->copy_1D.size                  = (size_t) c.getSize();
        }
    }
    else if (auto c = dyn_cast<Copy2DOp>(op))
    {
        command_t * cmd = node->command;
        if (cmd)
        {
            cmd->type                          = (command_type_t) c.getKind();
            cmd->copy_2D.src_device_unique_id  = (device_unique_id_t) (uint32_t) c.getSrcDuid();
            cmd->copy_2D.dst_device_unique_id  = (device_unique_id_t) (uint32_t) c.getDstDuid();
            cmd->copy_2D.src_addr              = (uintptr_t) c.getSrcAddr();
            cmd->copy_2D.src_ld                = (size_t) c.getSrcLd();
            cmd->copy_2D.dst_addr              = (uintptr_t) c.getDstAddr();
            cmd->copy_2D.dst_ld                = (size_t) c.getDstLd();
            cmd->copy_2D.m                     = (size_t) c.getM();
            cmd->copy_2D.n                     = (size_t) c.getN();
            cmd->copy_2D.sizeof_type           = (size_t) c.getSizeofType();
        }
    }
    /* EmptyOp / GenericOp: nothing besides duid to update */
}

/* Allocate a fresh POD node+command for an op created by a pass (no source). */
static command_graph_node_t *
create_pod_node_from_op(command_graph_t * cg, Operation * op)
{
    const uint8_t duid = get_duid(op);

    if (isa<EmptyOp>(op))
        return cg->command_graph_node_new(cg, duid, COMMAND_GRAPH_NODE_TYPE_EMPTY);

    if (auto c = dyn_cast<Copy1DOp>(op))
    {
        command_t * cmd = cg->command_new(cg, (command_type_t) c.getKind());
        cmd->copy_1D.src_device_unique_id = (device_unique_id_t) (uint32_t) c.getSrcDuid();
        cmd->copy_1D.dst_device_unique_id = (device_unique_id_t) (uint32_t) c.getDstDuid();
        cmd->copy_1D.src_device_addr      = (uintptr_t) c.getSrcAddr();
        cmd->copy_1D.dst_device_addr      = (uintptr_t) c.getDstAddr();
        cmd->copy_1D.size                 = (size_t) c.getSize();
        command_graph_node_t * node = cg->command_graph_node_new(cg, duid, COMMAND_GRAPH_NODE_TYPE_COMMAND);
        node->command = cmd;
        return node;
    }

    if (auto c = dyn_cast<Copy2DOp>(op))
    {
        command_t * cmd = cg->command_new(cg, (command_type_t) c.getKind());
        cmd->copy_2D.src_device_unique_id = (device_unique_id_t) (uint32_t) c.getSrcDuid();
        cmd->copy_2D.dst_device_unique_id = (device_unique_id_t) (uint32_t) c.getDstDuid();
        cmd->copy_2D.src_addr             = (uintptr_t) c.getSrcAddr();
        cmd->copy_2D.src_ld               = (size_t) c.getSrcLd();
        cmd->copy_2D.dst_addr             = (uintptr_t) c.getDstAddr();
        cmd->copy_2D.dst_ld               = (size_t) c.getDstLd();
        cmd->copy_2D.m                    = (size_t) c.getM();
        cmd->copy_2D.n                    = (size_t) c.getN();
        cmd->copy_2D.sizeof_type          = (size_t) c.getSizeofType();
        command_graph_node_t * node = cg->command_graph_node_new(cg, duid, COMMAND_GRAPH_NODE_TYPE_COMMAND);
        node->command = cmd;
        return node;
    }

    if (auto g = dyn_cast<GenericOp>(op))
    {
        command_t * cmd = cg->command_new(cg, (command_type_t) g.getKind());
        llvm::ArrayRef<int8_t> blob = g.getBlob();
        assert(blob.size() == sizeof(::ocg::command_t));
        /* overwrite the base command subobject with the saved bytes */
        memcpy((void *) static_cast<::ocg::command_t *>(cmd), blob.data(), blob.size());
        command_graph_node_t * node = cg->command_graph_node_new(cg, duid, COMMAND_GRAPH_NODE_TYPE_COMMAND);
        node->command = cmd;
        return node;
    }

    fprintf(stderr, "opencg/mlir: unsupported op '%s' during export\n",
            op->getName().getStringRef().str().c_str());
    abort();
}

} // anonymous namespace

void
export_command_graph(
    GraphOp graph,
    command_graph_t * cg,
    NodeMap & map
) {
    Block & body = graph.getBodyBlock();

    llvm::DenseMap<Operation *, command_graph_node_t *> out;
    command_graph_node_t * entryNode = nullptr;
    command_graph_node_t * exitNode  = nullptr;

    /* pass 1: resolve a POD node for every op and clear its adjacency */
    for (Operation & opref : body)
    {
        Operation * op = &opref;

        command_graph_node_t * node;
        auto it = map.find(op);
        if (it != map.end())
        {
            node = it->second;            /* reuse: keeps command/flags/replay state */
            update_node_from_op(node, op);
        }
        else
        {
            node = create_pod_node_from_op(cg, op);
        }

        out[op] = node;
        node->predecessors.clear();
        node->successors.clear();

        if (op_is_entry(op)) entryNode = node;
        if (op_is_exit(op))  exitNode  = node;
    }

    /* pass 2: relink edges from token operands */
    for (Operation & opref : body)
    {
        Operation * op = &opref;
        command_graph_node_t * node = out[op];
        for (Value pred : op->getOperands())
        {
            Operation * defop = pred.getDefiningOp();
            assert(defop);
            command_graph_node_t * pnode = out[defop];
            assert(pnode);
            pnode->precedes(node);
        }
    }

    /* rewire entry/exit */
    assert(entryNode && "exported graph has no entry node");
    assert(exitNode  && "exported graph has no exit node");
    assert(entryNode->predecessors.size() == 0);
    assert(exitNode->successors.size()    == 0);
    cg->entry = entryNode;
    cg->exit  = exitNode;
}

} // namespace cg
} // namespace ocg
