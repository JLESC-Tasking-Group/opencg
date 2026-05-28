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

# include <stdlib.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "opencg-tests.cc"

int
main(void)
{
    constexpr size_t size = 1024;
    uintptr_t src = (uintptr_t) malloc(size);
    assert(src);

    const device_unique_id_t ndevices = 4;

    assert(size % ndevices == 0);
    const size_t bs = size / ndevices;

    /* create a CG */
    command_graph_t * cg = command_graph_new();

    /* entry/exit nodes */
    command_graph_node_t * entry = cg->node_get_entry();
    command_graph_node_t * exit  = cg->node_get_exit();

    // for each device
    for (device_unique_id_t i = 0 ; i < ndevices ; ++i)
    {
        device_unique_id_t device_unique_id = (device_unique_id_t) (i + 1);

        // allocate device memory
        void * dst = malloc(bs);

        // create a command and initialize it
        constexpr command_type_t type  = COMMAND_TYPE_COPY_H2D_1D;
        command_t * command = (command_t *) malloc(sizeof(command_t));
        assert(command);
        new (command) command_t(type);
        command->copy_1D.src_device_unique_id   = OCG_UNSPECIFIED_DEVICE_UNIQUE_ID;
        command->copy_1D.dst_device_unique_id   = device_unique_id;
        command->copy_1D.src_device_addr        = src + i * bs;
        command->copy_1D.dst_device_addr        = (uintptr_t) dst;
        command->copy_1D.size                   = bs;

        // add node to CG
        command_graph_node_t * node = command_graph_node_new(cg, command, device_unique_id);
        assert(node);
        entry->precedes(node);
        node->precedes(exit);
    }

    /* optimize the cg->*/
    cg->optimize(COMMAND_GRAPH_PASS_REDUCE_NODE);
    cg->optimize(COMMAND_GRAPH_PASS_REDUCE_EDGE);
    cg->optimize(COMMAND_GRAPH_PASS_BATCH);

    return 0;
}
