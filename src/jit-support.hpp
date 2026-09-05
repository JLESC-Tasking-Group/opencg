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

/*
 *  Support machinery shared by the LLVM-backed passes (prog-fuse and jit):
 *  environment knobs, opt-in profiling, and the JIT result cache. None of it
 *  transforms IR -- it only decides what to do, measures how long it took, and
 *  avoids doing it twice.
 *
 *  Internal to src/; not installed. Everything here is process-global state
 *  behind accessor functions, so it must live in one translation unit rather
 *  than being header-only.
 */

# ifndef __CGIR_JIT_SUPPORT_HPP__
# define __CGIR_JIT_SUPPORT_HPP__

# include <cgir/namespace.hpp>
# include <cgir/command.hpp>

# if CGIR_SUPPORT_LLVM

#  include <chrono>
#  include <cstdint>
#  include <string>

CGIR_NAMESPACE_BEGIN
namespace jit {

/* ---------------------------------------------------------------------------
 * Environment knobs.
 * ------------------------------------------------------------------------- */

/* True iff the env var `var` is set to a non-empty, non-"0" value. */
bool env_flag(const char * var);

/* Value of the env var `var`, or `dflt` when unset/empty. */
const char * env_str(const char * var, const char * dflt);

/* `CGIR_JIT_DEVICE_NOALIAS`: assume a device kernel's pointer parameters do not
 * overlap. Read once; folded into the cache key, since it changes the code. */
bool device_assume_noalias_params(void);

/* ---------------------------------------------------------------------------
 * Opt-in profiling. Off (and near-free) unless enabled via env, read once at
 * first use:
 *   CGIR_JIT_TIMING=1       accumulate per-phase wall time, print at exit
 *   CGIR_JIT_CACHE_STATS=1  count cache outcomes, print at exit
 *   CGIR_JIT_STATS_CSV=path append one machine-readable row at exit
 *   CGIR_STATS_TAG=s        join that row to a caller's run
 * A single report is emitted to stderr at static destruction.
 * ------------------------------------------------------------------------- */

/* Whether timings are being collected (checked before reading any clock). */
bool profiling_timing_on(void);

/* Whether cache outcomes are being collected. */
bool profiling_stats_on(void);

/* Add `secs` to the named bucket. */
void profiling_add_phase(const char * name, double secs);

/* Record one cache outcome: 0 = full compile, 1 = disk reuse, 2 = in-process
 * reuse, for the host (device=false) or device (device=true) compiler. */
void profiling_cache_event(bool device, int outcome);

/* RAII wall-clock timer adding its lifetime to a named bucket. Cost when
 * disabled: one cached bool test (no clock reads). */
struct scoped_phase_t
{
    const char *                          name;
    std::chrono::steady_clock::time_point t0;
    bool                                  on;

    explicit scoped_phase_t(const char * n) : name(n), on(profiling_timing_on())
    {
        if (on)
            t0 = std::chrono::steady_clock::now();
    }

    ~scoped_phase_t()
    {
        if (!on)
            return ;
        const auto t1 = std::chrono::steady_clock::now();
        profiling_add_phase(name, std::chrono::duration<double>(t1 - t0).count());
    }
};

/* ---------------------------------------------------------------------------
 * JIT result cache: content-addressed, in-process, optionally backed by disk.
 * ------------------------------------------------------------------------- */

/* Content key for `prog`: every byte and attribute a cached artifact depends
 * on. Two instances of the same construct collide here, which is the intended
 * reuse -- a hit is byte-identical to recompiling. */
uint64_t cache_key(const command_prog_t * prog);

/* Host: compiled function pointer, in-process only. Returns true on a hit and
 * writes the entry's prototype and address. */
bool cache_host_get_fn(uint64_t key, int * proto, void ** fn);
void cache_host_put_fn(uint64_t key, int proto, void * fn);

/* Host: relocatable object, on disk (opt-in via CGIR_JIT_CACHE_DIR). */
bool cache_host_get_obj(uint64_t key, std::string & out);
void cache_host_put_obj(uint64_t key, const std::string & obj);

/* Device: emitted PTX. Returns 0 = miss, 1 = from disk, 2 = in-process. */
int  cache_device_get(uint64_t key, std::string & out);
void cache_device_put(uint64_t key, const std::string & ptx);

} /* namespace jit */
CGIR_NAMESPACE_END

# endif /* CGIR_SUPPORT_LLVM */

# endif /* __CGIR_JIT_SUPPORT_HPP__ */
