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

# include <cgir/namespace.hpp>
# include <cgir/command.hpp>
# include <cgir/command-graph.hpp>

CGIR_NAMESPACE_USE;

/* reachability matrix */
typedef bitset2d_t<uint64_t, command_graph_node_index_t> command_graph_reachability_t;

void
command_graph_t::pass_transitive_reduction(void)
{
    /* Iterate through all nodes */
    constexpr bool include_entry_exit = true;
    auto nodes = this->create_node_iterators<include_entry_exit>();
    const int n = nodes.size();

    /* allocate reachability */
    command_graph_reachability_t r(n);

    /* compute reachability */
    this->walk<COMMAND_GRAPH_WALK_SEARCH_DFS, COMMAND_GRAPH_WALK_ORDER_POST>(
        [&] (command_graph_node_t * node)
        {
            r.set(node->iterator_index, node->iterator_index);
            for (command_graph_node_t * succ : node->successors)
                r.or_rows(node->iterator_index, succ->iterator_index);
        }
    );

    for (command_graph_node_index_t i = 0 ; i < n ; ++i)
    {
        command_graph_node_t * u = nodes[i].node;
        assert(u);

        # if 0
        if (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND)
        {
            assert(u->command);
            if (u->command->type == COMMAND_TYPE_BATCH && u->command->batch)
            {
                if (u->command->batch->has_cg)
                {
                    u->command->batch->cg.pass_reduction_edge();
                }
            }
        }
        # endif

        /* for each successor 'v' of 'u' */
        for (auto itv = u->successors.begin(); itv != u->successors.end(); )
        {
            command_graph_node_t * v = *itv;
            assert(u != v);

            bool is_redundant = false;

            /* test if 'v' is reachable from any other successor 'w' of 'u' */
            for (command_graph_node_t * w : u->successors)
            {
                if (v == w) continue;

                if (r.test(w->iterator_index, v->iterator_index))
                {
                    is_redundant = true;
                    break; /* Stop searching 'w' as soon as reachability is proven */
                }
            }

            if (is_redundant)
            {
                /* remove 'v' from 'u' successors (erase returns the next valid iterator) */
                itv = u->successors.erase(itv);

                /* remove 'u' from 'v' predecessors */
                auto it_pred = std::find(v->predecessors.begin(), v->predecessors.end(), u);
                if (it_pred != v->predecessors.end())
                    v->predecessors.erase(it_pred);
            }
            else
            {
                ++itv;
            }
        }
    }
}
