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

/* pass local storage */
struct pls_t {};
using node_t = command_graph_t::node_iterator_t<pls_t>;

//  This pass convert 2D copies to 1D when applicable,
//  and convert 2D copies to a normalized form (i.e., with sizeof_type = 1)
void
command_graph_t::pass_copy_normalize(void)
{
    /* Iterate through all nodes, and fuse contiguous copies occuring in sibling nodes */
    constexpr bool include_entry_exit = false;
    std::vector<node_t> nodes = this->create_node_iterators<pls_t, include_entry_exit>();

    /* iterate through each node */
    for (command_graph_node_index_t i = 0 ; i < nodes.size() ; ++i)
    {
        node_t & node = nodes[i];
        command_graph_node_t * u = node.node;
        assert(u);

        /* for each command */
        if (u->command)
        {
            /**
             * if it is a 2D copy,
             *  - normalize it to a 1D copy if possible
             *  - set sizeof_type to 1 and scale dimensions
             */

            switch (u->command->type)
            {
                # define HANDLE_CASE(X)                                                             \
                    case (COMMAND_TYPE_COPY_##X##_2D):                                              \
                    {                                                                               \
                        if (u->command->copy_2D.m == u->command->copy_2D.src_ld &&                  \
                            u->command->copy_2D.m == u->command->copy_2D.dst_ld)                    \
                        {                                                                           \
                            u->command->type = COMMAND_TYPE_COPY_##X##_1D;                          \
                                                                                                    \
                            const size_t src_addr = u->command->copy_2D.src_addr;                   \
                            const size_t dst_addr = u->command->copy_2D.dst_addr;                   \
                            const size_t m        = u->command->copy_2D.m;                          \
                            const size_t n        = u->command->copy_2D.n;                          \
                            const size_t s        = u->command->copy_2D.sizeof_type;                \
                                                                                                    \
                            u->command->copy_1D.src_device_addr = src_addr;                         \
                            u->command->copy_1D.dst_device_addr = dst_addr;                         \
                            u->command->copy_1D.size            = m * n * s;                        \
                                                                                                    \
                            break ;                                                                 \
                        }                                                                           \
                        else                                                                        \
                        {                                                                           \
                            const size_t s = u->command->copy_2D.sizeof_type;                       \
                            u->command->copy_2D.sizeof_type = 1;                                    \
                            u->command->copy_2D.m = u->command->copy_2D.m * s;                      \
                            break ;                                                                 \
                        }                                                                           \
                    }

                HANDLE_CASE(H2H);
                HANDLE_CASE(H2D);
                HANDLE_CASE(D2H);
                HANDLE_CASE(D2D);

                # undef HANDLE_CASE

                default:
                    break ;
            }
        }
    }
}
