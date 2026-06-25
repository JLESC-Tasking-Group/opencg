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

// Common code shared by the POD (src/pod/) and MLIR (src/mlir/) layers: the
// optimization dispatcher that selects, at run-time, between the legacy POD
// passes and the MLIR `cg` pipeline.

# include <cgir/command.hpp>
# include <cgir/command-graph.hpp>

# include <stdlib.h>
# include <string.h>

CGIR_NAMESPACE_USE;

////////////////////////////
// OPTIMIZATION DISPATCHER //
////////////////////////////

# if CGIR_USE_MLIR
/* Defined in src/mlir/optimize.cpp. MLIR-free signature so this translation
 * unit (and any consumer) never needs MLIR headers. */
namespace CGIR_NAMESPACE { void command_graph_optimize_mlir(command_graph_t * cg, command_graph_pass_t pass); }

/* Returns true if the MLIR optimizer should be used.
 * Controlled by the CGIR_OPTIMIZER environment variable:
 *   - "mlir"   -> use the MLIR `cg` pipeline
 *   - anything else (or unset) -> use the legacy POD passes (default)
 * The legacy POD passes remain the default so existing behavior is unchanged;
 * setting CGIR_OPTIMIZER=mlir enables A/B comparison within a single binary. */
static inline bool
command_graph_use_mlir_optimizer(void)
{
    static int use_mlir_optimizer = -1;
    if (use_mlir_optimizer == -1)
    {
        const char * s = getenv("CGIR_OPTIMIZER");
        use_mlir_optimizer = (s && strcmp(s, "mlir") == 0) ? 1 : 0;
    }
    return (bool) use_mlir_optimizer;
}
# endif /* CGIR_USE_MLIR */

void
CGIR_NAMESPACE::command_graph_optimize(
    command_graph_t * cg,
    command_graph_pass_t pass
) {
    # if CGIR_USE_MLIR
    if (command_graph_use_mlir_optimizer())
    {
        command_graph_optimize_mlir(cg, pass);
        return ;
    }
    # endif /* CGIR_USE_MLIR */

    /* legacy C++ passes */
    cg->optimize_legacy(pass);
}
