/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
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

#ifndef __OPENCG_COMMAND_TYPE_HPP__
# define __OPENCG_COMMAND_TYPE_HPP__

# include <opencg/namespace.hpp>

# include <cassert>

OCG_NAMESPACE_BEGIN

/* Macro to iterate through all command type */
# define OCG_FORALL_COMMAND_TYPE(F)                                                 \
    F(COMMAND_TYPE_PROG,            command_prog_t,             "PROG")             \
    F(COMMAND_TYPE_COPY_H2H_1D,     command_copy_1D_t,          "COPY-1D-H2H")      \
    F(COMMAND_TYPE_COPY_H2D_1D,     command_copy_1D_t,          "COPY-1D-H2D")      \
    F(COMMAND_TYPE_COPY_D2H_1D,     command_copy_1D_t,          "COPY-1D-D2H")      \
    F(COMMAND_TYPE_COPY_D2D_1D,     command_copy_1D_t,          "COPY-1D-D2D")      \
    F(COMMAND_TYPE_COPY_H2H_2D,     command_copy_2D_t,          "COPY-2D-H2H")      \
    F(COMMAND_TYPE_COPY_H2D_2D,     command_copy_2D_t,          "COPY-2D-H2D")      \
    F(COMMAND_TYPE_COPY_D2H_2D,     command_copy_2D_t,          "COPY-2D-D2H")      \
    F(COMMAND_TYPE_COPY_D2D_2D,     command_copy_2D_t,          "COPY-2D-D2D")      \
    F(COMMAND_TYPE_FD_READ,         command_file_t,             "FD-READ")          \
    F(COMMAND_TYPE_FD_WRITE,        command_file_t,             "FD-WRITE")         \
    F(COMMAND_TYPE_BATCH,           command_batch_t,            "BATCH")            \
    F(COMMAND_TYPE_CTRL_LOOP,       command_ctrl_loop_t,        "CTRL-LOOP")        \
    F(COMMAND_TYPE_CTRL_DEMUX,      command_ctrl_demux_t,       "CTRL-DEMUX")

enum command_type_t
{
    # define DEF(ENUM, TYPE, NAME) ENUM,
    OCG_FORALL_COMMAND_TYPE(DEF)
    # undef DEF
    COMMAND_TYPE_MAX
};

static inline const char *
command_type_to_str(command_type_t type)
{
    switch (type)
    {
        # define DEF(ENUM, TYPE, NAME) case(ENUM): return NAME;
        OCG_FORALL_COMMAND_TYPE(DEF);
        # undef DEF
        default:
            return "(null)";
    }
}

OCG_NAMESPACE_END

#endif /* __OPENCG_COMMAND_TYPE_HPP__ */
