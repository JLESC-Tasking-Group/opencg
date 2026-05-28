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

# include <opencg/command.hpp>
# include <opencg/command-graph.hpp>

OCG_NAMESPACE_USE;

///////////
//  DUMP //
///////////

static inline void
command_graph_dump_interior(
    command_graph_t * cg,
    FILE * f
) {
    // Check that there is a unique source and sink
    struct pls_t {};
    using node_t = command_graph_t::node_iterator_t<pls_t>;
    constexpr bool include_entry_exit = true;
    std::vector<node_t> nodes = cg->create_node_iterators<pls_t, include_entry_exit>();
    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        command_graph_node_t * node = nodes[i].node;
        assert(node);
        /* print the node */
        if (node->command == NULL)
        {
            fprintf(f, "  \"%p\" [label=\"node %lu\\ndev=%u\"] ;\n", node, node->iterator_index, node->device_unique_id);
        }
        else
        {
            assert(node->command);
            if (node->command->type == COMMAND_TYPE_BATCH && node->command->batch.cg)
            {
                fprintf(f, "  subgraph cluster_%p {\n", node);
                fprintf(f, "    \"%p\" [style=invis, width=0, height=0, label=\"\"] ;\n", node);
                fprintf(f, "    label=\"node %lu\\ndev=%u\\ncmd=%s\" ;\n",
                        node->iterator_index, node->device_unique_id, command_type_to_str(node->command->type));
                command_graph_dump_interior(node->command->batch.cg, f);
                fprintf(f, "  }\n");
            }
            else
            {
                fprintf(f, "  \"%p\" [label=\"node %lu\\ndev=%u\\ncmd=%s\"] ;\n",
                        node, node->iterator_index, node->device_unique_id, command_type_to_str(node->command->type));
            }
        }

        /* print edges */
        node->foreach_successor([&] (command_graph_node_t * succ)
        {
            fprintf(f, "  \"%p\" -> \"%p\"", node, succ);
            if (node->command && node->command->type == COMMAND_TYPE_BATCH)
                fprintf(f, " [ltail=cluster_%p]", node);
            if (succ->command && succ->command->type == COMMAND_TYPE_BATCH)
                fprintf(f, " [lhead=cluster_%p]", succ);
            fprintf(f, " ;\n");
        });
    }
}

void
command_graph_t::dump(FILE * f)
{
    fprintf(f, "digraph G {\n");
    fprintf(f, "  compound = true ;\n");
    command_graph_dump_interior(this, f);
    fprintf(f, "}\n");
}

void
command_graph_t::coherence_checks(void)
{
    // Check that entry/exit are correct
    command_graph_node_t * entry = this->node_get_entry();
    command_graph_node_t * exit  = this->node_get_exit();
    assert(entry->predecessors.size() == 0);
    assert( exit->successors.size()   == 0);

    // Check that there is a unique source and sink
    struct pls_t {};
    using node_t = command_graph_t::node_iterator_t<pls_t>;
    constexpr bool include_entry_exit = true;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();
    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        node_t & node = nodes[i];
        command_graph_node_t * u = node.node;
        assert(u);
        assert(u == this->node_get_entry() || u->predecessors.size());
        assert(u == this->node_get_exit()  || u->successors.size());
    }
}
