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
# include <cgir/command-type.hpp>

# include <atomic>
# include <chrono>
# include <mutex>
# include <string>

# include <errno.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <sys/stat.h>
# include <sys/types.h>

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

/* ------------------------------------------------------------------------- *
 * Debug dump option (CGIR_OPTIMIZE_DUMP).                                    *
 *                                                                           *
 * Like CGIR_PROG_FUSE_DUMP (which dumps the LLVM IR of each prog-fusion),    *
 * this is controlled by an environment variable read once and cached. When   *
 * CGIR_OPTIMIZE_DUMP is set (to anything other than "" or "0"), every pass   *
 * run through the dispatcher writes the command graph as a Graphviz .dot     *
 * file both BEFORE and AFTER the transformation, so each pass can be         *
 * visualized:                                                               *
 *                                                                           *
 *   <base>/optimize-<seq>-<pass>-before.dot   (graph before the pass)        *
 *   <base>/optimize-<seq>-<pass>-after.dot    (graph after  the pass)        *
 *                                                                           *
 * <base> is ~/.cgir/tmp by default, or the value of CGIR_OPTIMIZE_DUMP when  *
 * it is an absolute path (so the output location can be overridden). <seq>   *
 * is a per-process counter (one per pass invocation) so concurrent / chained *
 * optimizations do not clobber each other; because the dispatcher is invoked *
 * once per pass in pipeline order, the <seq> values form the timeline of an  *
 * optimize() call (the "after" of one pass is the "before" of the next).     *
 * This works for both the legacy POD passes and the MLIR pipeline since both *
 * route through command_graph_optimize() below.                             *
 * ------------------------------------------------------------------------- */
static bool
optimize_dump_enabled(void)
{
    static int enabled = -1;
    if (enabled == -1)
    {
        const char * s = getenv("CGIR_OPTIMIZE_DUMP");
        enabled = (s && s[0] != '\0' && strcmp(s, "0") != 0) ? 1 : 0;
    }
    return (bool) enabled;
}

/* Per-process counter handing out a fresh sequence number for each pass
 * invocation, so concurrent / chained optimizations never share a filename. */
static unsigned
optimize_dump_make_seq(void)
{
    static std::atomic<unsigned> seq{0};
    return seq.fetch_add(1);
}

/* Recursively create `path` (like `mkdir -p`). Returns true on success or if it
 * already exists. */
static bool
optimize_dump_mkdir_p(const std::string & path)
{
    std::string partial;
    size_t i = 0;
    if (!path.empty() && path[0] == '/')
    {
        partial = "/";
        i = 1;
    }
    while (i < path.size())
    {
        size_t slash = path.find('/', i);
        if (slash == std::string::npos)
            slash = path.size();
        partial += path.substr(i, slash - i);
        if (!partial.empty() && mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
            return false;
        partial += "/";
        i = slash + 1;
    }
    return true;
}

/* Resolve (and create once) the base dump directory: an absolute
 * CGIR_OPTIMIZE_DUMP value, else ~/.cgir/tmp. Returns an empty string if the
 * directory could not be created. Computed once and cached (thread-safe). */
static const std::string &
optimize_dump_base_dir(void)
{
    static const std::string base = []() -> std::string
    {
        std::string b;
        const char * s = getenv("CGIR_OPTIMIZE_DUMP");
        if (s && s[0] == '/')
        {
            b = s;
        }
        else
        {
            const char * home = getenv("HOME");
            b  = home ? home : ".";
            b += "/.cgir/tmp";
        }

        if (!optimize_dump_mkdir_p(b))
        {
            fprintf(stderr, "cgir: cannot create dump dir '%s': %s\n",
                    b.c_str(), strerror(errno));
            return std::string();
        }
        return b;
    }();
    return base;
}

/* Write `cg` as a Graphviz .dot file for the given pass and phase ("before" or
 * "after"). No-op if the base dump directory is unavailable. */
static void
optimize_dump_graph(
    command_graph_t * cg,
    unsigned seq,
    command_graph_pass_t pass,
    const char * phase
) {
    const std::string & base = optimize_dump_base_dir();
    if (base.empty())
        return ;

    std::string path = base + "/optimize-" + std::to_string(seq) + "-"
                     + command_graph_pass_to_str(pass) + "-" + phase + ".dot";

    FILE * f = fopen(path.c_str(), "w");
    if (!f)
    {
        fprintf(stderr, "cgir: cannot write '%s': %s\n", path.c_str(), strerror(errno));
        return ;
    }
    cg->dump(f);
    fclose(f);
}

/* ------------------------------------------------------------------------- *
 * Machine-readable per-pass statistics (CGIR_STATS_CSV).                     *
 *                                                                           *
 * When CGIR_STATS_CSV is set (to a path other than "" or "0"), every pass    *
 * run through the dispatcher appends one CSV row measuring the pass's effect  *
 * on the command graph: its wall-clock time and the graph's node/edge and    *
 * per-command-type counts BEFORE and AFTER the transformation. This is the    *
 * data source for the paper's evaluation (graph reduction, copy/prog fusion,  *
 * batching). It is independent of CGIR_OPTIMIZE_DUMP (which writes .dot).     *
 *                                                                           *
 * Rows are tagged with CGIR_STATS_TAG (a caller-provided run identifier, set  *
 * per run by the benchmark harness) so a single appended file can hold every  *
 * run of a sweep and be joined back to the harness' runs.csv on that tag.     *
 *                                                                           *
 * Counts are taken over the top-level graph (a forward walk from entry): the  *
 * batch pass collapses a device's commands into a single BATCH node whose     *
 * sub-graph is not descended into, so after batching nodes_after drops and    *
 * batch_after rises -- exactly the signal we want to report.                  *
 * ------------------------------------------------------------------------- */
static const char *
stats_csv_path(void)
{
    static const char * path   = nullptr;
    static int          inited = 0;
    if (!inited)
    {
        const char * s = getenv("CGIR_STATS_CSV");
        path   = (s && s[0] != '\0' && strcmp(s, "0") != 0) ? s : nullptr;
        inited = 1;
    }
    return path;
}

/* Node-type and command-type counts of a command graph. */
struct cg_metrics_t
{
    size_t nodes, edges;                    /* totals */
    size_t empty, command, graph, condition;/* by node type */
    size_t prog, copy1d, copy2d, batch;     /* by command type (COMMAND nodes) */
};

/* Count nodes, edges and command types over the top-level graph. */
static cg_metrics_t
stats_compute_metrics(command_graph_t * cg)
{
    cg_metrics_t m;
    memset(&m, 0, sizeof(m));

    cg->walk([&] (command_graph_node_t * node)
    {
        m.nodes += 1;
        m.edges += node->successors.size();

        switch (node->type)
        {
            case COMMAND_GRAPH_NODE_TYPE_EMPTY:
                m.empty += 1;
                break;

            case COMMAND_GRAPH_NODE_TYPE_COMMAND:
                m.command += 1;
                if (node->command)
                {
                    switch (node->command->type)
                    {
                        case COMMAND_TYPE_PROG:
                            m.prog += 1;
                            break;
                        case COMMAND_TYPE_COPY_H2H_1D:
                        case COMMAND_TYPE_COPY_H2D_1D:
                        case COMMAND_TYPE_COPY_D2H_1D:
                        case COMMAND_TYPE_COPY_D2D_1D:
                            m.copy1d += 1;
                            break;
                        case COMMAND_TYPE_COPY_H2H_2D:
                        case COMMAND_TYPE_COPY_H2D_2D:
                        case COMMAND_TYPE_COPY_D2H_2D:
                        case COMMAND_TYPE_COPY_D2D_2D:
                            m.copy2d += 1;
                            break;
                        case COMMAND_TYPE_BATCH:
                            m.batch += 1;
                            break;
                        default:
                            break;
                    }
                }
                break;

            case COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH:
                m.graph += 1;
                break;

            case COMMAND_GRAPH_NODE_TYPE_CONDITION:
                m.condition += 1;
                break;
        }
    });

    return m;
}

/* Append one row for a completed pass. Serialized so appends stay atomic even
 * if two graphs are optimized concurrently. */
static void
stats_emit(
    command_graph_pass_t pass,
    unsigned seq,
    double pass_ms,
    const cg_metrics_t & b,
    const cg_metrics_t & a
) {
    const char * path = stats_csv_path();
    if (!path)
        return ;

    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    struct stat st;
    const bool need_header = (stat(path, &st) != 0 || st.st_size == 0);

    FILE * f = fopen(path, "a");
    if (!f)
    {
        fprintf(stderr, "cgir: cannot write '%s': %s\n", path, strerror(errno));
        return ;
    }

    if (need_header)
        fprintf(f,
            "tag,seq,pass,pass_ms,"
            "nodes_before,nodes_after,edges_before,edges_after,"
            "empty_before,empty_after,command_before,command_after,"
            "graph_before,graph_after,"
            "prog_before,prog_after,copy1d_before,copy1d_after,"
            "copy2d_before,copy2d_after,batch_before,batch_after\n");

    const char * tag = getenv("CGIR_STATS_TAG");
    fprintf(f,
        "%s,%u,%s,%.6f,"
        "%zu,%zu,%zu,%zu,"
        "%zu,%zu,%zu,%zu,"
        "%zu,%zu,"
        "%zu,%zu,%zu,%zu,"
        "%zu,%zu,%zu,%zu\n",
        tag ? tag : "", seq, command_graph_pass_to_str(pass), pass_ms,
        b.nodes, a.nodes, b.edges, a.edges,
        b.empty, a.empty, b.command, a.command,
        b.graph, a.graph,
        b.prog, a.prog, b.copy1d, a.copy1d,
        b.copy2d, a.copy2d, b.batch, a.batch);

    fclose(f);
}

void
CGIR_NAMESPACE::command_graph_optimize(
    command_graph_t * cg,
    command_graph_pass_t pass
) {
    /* Optional command-graph dumping for debugging (CGIR_OPTIMIZE_DUMP). When
     * enabled, the graph is written as .dot before and after the pass. A single
     * per-pass sequence number ties the before/after pair together. */
    const bool     dump  = optimize_dump_enabled();
    const bool     stats = (stats_csv_path() != nullptr);
    const unsigned seq   = (dump || stats) ? optimize_dump_make_seq() : 0u;
    if (dump)
        optimize_dump_graph(cg, seq, pass, "before");

    /* Snapshot graph metrics before the pass (stats only). */
    cg_metrics_t before;
    if (stats)
        before = stats_compute_metrics(cg);

    const auto t0 = std::chrono::steady_clock::now();

    # if CGIR_USE_MLIR
    if (command_graph_use_mlir_optimizer())
        command_graph_optimize_mlir(cg, pass);
    else
    # endif /* CGIR_USE_MLIR */
        /* legacy C++ passes */
        cg->optimize_legacy(pass);

    const auto t1 = std::chrono::steady_clock::now();

    if (stats)
    {
        const cg_metrics_t after   = stats_compute_metrics(cg);
        const double       pass_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_emit(pass, seq, pass_ms, before, after);
    }

    if (dump)
        optimize_dump_graph(cg, seq, pass, "after");
}
