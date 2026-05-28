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
struct pls_t {};
using node_t = command_graph_t::node_iterator_t<pls_t>;

void
command_graph_t::pass_reduce_node(void)
{
    /* Iterate through all nodes */
    constexpr bool include_entry_exit = false;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();

    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        node_t & node = nodes[i];
        command_graph_node_t * u = node.node;
        assert(u);

        # if 0
        if (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
        {
            if (u->command)
            {
                if (u->command->type == COMMAND_TYPE_BATCH && u->command->batch)
                {
                    if (u->command->batch->has_cg)
                    {
                        u->command->batch->cg.pass_reduction_node();
                    }
                }
            }
        }
        # endif

        /* remove control nodes that do not simplifies the graph complexity */
        if (!u->command)
        {
            command_graph_node_index_t m = u->predecessors.size();
            command_graph_node_index_t n = u->successors.size();

            /**
             *  Complexity with the control node is: node + predecessors edges + successor edges =
             *                                      1  +         m          +        n
             *
             *  Complexity without the control node is: m x n
             */

            if (m * n < 1 + m + n)
            {
                // LOGGER_DEBUG("Removing %zu", u->index);
                this->remove(u);
            }
        }
    }
}
