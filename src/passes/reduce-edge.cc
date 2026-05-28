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

/* reachability matrix */
typedef bitset2d_t<uint64_t, command_graph_node_index_t> command_graph_reachability_t;

void
command_graph_t::pass_reduce_edge(void)
{
    /* Iterate through all nodes */
    constexpr bool include_entry_exit = true;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();
    const int n = nodes.size();

    /* allocate reachability */
    command_graph_reachability_t r(n);

    /* compute reachability */
    command_graph_node_t * exit = this->node_get_exit();
    this->walk<COMMAND_GRAPH_WALK_DIRECTION_BACKWARD, COMMAND_GRAPH_WALK_SEARCH_BFS>(
        exit,
        [&] (command_graph_node_t * node)
        {
            r.set(node->iterator_index, node->iterator_index);
            for (command_graph_node_t * succ : node->successors)
                r.or_rows(node->iterator_index, succ->iterator_index);
        }
    );

    for (command_graph_node_index_t i = 0 ; i < n ; ++i)
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
                        u->command->batch->cg.pass_reduction_edge();
                    }
                }
            }
        }
        # endif

        /* for each successor 'v' of 'u' */
        for (auto itv = u->successors.begin(); itv != u->successors.end(); )
        {
            command_graph_node_t * v = *itv;
            assert(u != v);

            /* for each successor 'w' of 'u' */
            for (command_graph_node_t * w : u->successors)
            {
                assert(u != w);
                if (v == w)
                    continue ;

                /* test if 'v' is reachable from any other successors 'w' of 'u' */
                if (r.test(w->iterator_index, v->iterator_index))
                {
                    /* remove 'v' from 'u' successors list, because we can reach it from 'w' */
                    itv = u->successors.erase(itv);

                    /* also remove 'u' from 'v' predecessors list */
                    assert(std::count(v->predecessors.begin(), v->predecessors.end(), u) == 1);
                    v->predecessors.erase(std::find(v->predecessors.begin(), v->predecessors.end(), u));
                    goto skip_increment;
                }
            }

            /* process next successors */
            ++itv;
skip_increment:
            ;
        }
    }
}
