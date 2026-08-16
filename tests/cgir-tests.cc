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

# include <cgir/cgir.hpp>

CGIR_NAMESPACE_USE;

static command_t *
command_new(command_graph_t * cg, const command_type_t type)
{
    return new command_t(type);
}

static command_graph_node_t *
command_graph_node_new(
    command_graph_t * cg,
    const device_unique_id_t device_unique_id,
    const command_graph_node_type_t type
) {
    return new command_graph_node_t(device_unique_id, type);
}

static command_graph_t *
command_graph_new(command_graph_t * original_cg, command_graph_node_t * entry, command_graph_node_t * exit)
{
    command_graph_t * cg = new command_graph_t();
    /* When invoked as the allocator callback for a *sub*-graph (e.g. a batch or a
     * sequence), initialize it so it has entry/exit nodes and allocator
     * callbacks. If entry/exit are provided (e.g. a sequence's head/tail), reuse
     * them; otherwise fresh EMPTY entry/exit are allocated. The top-level graph
     * is created with original_cg == NULL and is initialized explicitly by the
     * caller via cg->init(...). */
    if (original_cg)
        cg->init(command_new, command_graph_node_new, command_graph_new, entry, exit);
    return cg;
}

static command_graph_t *
command_graph_new(void)
{
    return command_graph_new(NULL, nullptr, nullptr);
}
