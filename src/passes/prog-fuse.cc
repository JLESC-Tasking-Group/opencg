/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

# include <opencg/namespace.hpp>
# include <opencg/command.hpp>
# include <opencg/command-graph.hpp>

OCG_NAMESPACE_USE;

/* pass local storage */
struct pls_t
{
    bool contracted;

    pls_t(void) : contracted(false) {}
    ~pls_t(void) {}
};

using node_t = command_graph_t::node_iterator_t<pls_t>;

static inline void
command_graph_pass_prog_fuse_llvmir(
    command_graph_t * cg,
    command_prog_t * pu,
    command_prog_t * pv,
    command_prog_t * puv
) {
    # if !OPENCG_SUPPORT_PROTEUS
    fprintf(stderr, "Not supported\n");
    abort();
    # else


    // TODO





    # endif
}

static inline void
command_graph_pass_prog_fuse(
    command_graph_t * cg,
    command_graph_node_t * u,
    command_graph_node_t * v,
    std::vector<node_t> & nodes
) {
    /* assert tests */
    assert(u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(u->command);
    assert(u->command->type == COMMAND_TYPE_PROG);

    assert(v->command);
    assert(v->type == COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(v->command->type == COMMAND_TYPE_PROG);

    assert(std::find(u->successors.begin(),   u->successors.end(),   v) != u->successors.end());
    assert(std::find(v->predecessors.begin(), v->predecessors.end(), u) != v->predecessors.end());

    /* only llvm ir supported currently */
    if (u->command->prog.source.type != COMMAND_PROG_SOURCE_TYPE_LLVMIR ||
        v->command->prog.source.type != COMMAND_PROG_SOURCE_TYPE_LLVMIR)
    {
        fprintf(stderr, "Only supporting LLVMIR source type for kernel fusion\n");
        abort();
    }

    /* fuse programs to the node 'u' */
    command_graph_pass_prog_fuse_llvmir(
        cg,
        &u->command->prog,
        &v->command->prog,
        &u->command->prog
    );

    /* remove node 'v' */
    cg->contract<COMMAND_GRAPH_CONTRACTION_HINT_U_V_SEQUENCE | COMMAND_GRAPH_CONTRACTION_HINT_INPLACE>(u, v);

    /* mark 'v' as contracted to skip it from future lookups */
    nodes[v->iterator_index].data.contracted = true;
}

void
command_graph_t::pass_prog_fuse(void)
{
    /* Iterate through all nodes, and contract until we tried all nodes.
     * New node can be safely pushed-back: they will be iterated on. */
    constexpr bool include_entry_exit = false;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();

    /* iterate through each original nodes */
    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        node_t & node = nodes[i];
        command_graph_node_t * u = node.node;
        assert(u);

        /* if the node was already contracted, ignore it */
        if (node.data.contracted)
            continue ;
        assert(!node.data.contracted);

        /* Detect u->v sequence */
        if (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && u->command->type == COMMAND_TYPE_PROG)
        {
retry_node:
            for (command_graph_node_t * v : u->successors)
            {
                assert(u != v);
                if (v->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && v->command->type == COMMAND_TYPE_PROG)
                {
                    if (this->are_sequence(u, v))
                    {
                        command_graph_pass_prog_fuse(this, u, v, nodes);
                        goto retry_node;
                    }
                }
            }
        }

        // TODO: it this needed ?
        # if 0
        /* 2. detect v->u sequence */
        for (command_graph_node_t * v : u->predecessors)
        {
            assert(u != v);
            if (command_graph_pass_batch_can_batch(u, v))
            {
                if (this->are_sequence(v, u))
                {
                    command_graph_pass_batch_contract<COMMAND_GRAPH_CONTRACTION_HINT_V_U_SEQUENCE>(this, u, v, nodes);
                    goto retry_node;
                }
            }
        }
        # endif
    }
}
