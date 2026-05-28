# Try to find OpenCG headers and libraries.
#
# Usage of this module as follows:
#
#     find_package(OpenCG)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  OpenCG_PREFIX         Set this variable to the root installation of
#                      libpapi if the module has problems finding the
#                      proper installation path.
#
# Variables defined by this module:
#
#  OpenCG_FOUND              System has OpenCG libraries and headers
#  OpenCG_LIBRARIES          The OpenCG library
#  OpenCG_INCLUDE_DIRS       The location of OpenCG headers

find_library(OpenCG_LIBRARIES NAMES libopencg.so)
find_path(OpenCG_INCLUDE_DIRS NAMES opencg/opencg.hpp)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCG DEFAULT_MSG
    OpenCG_LIBRARIES
    OpenCG_INCLUDE_DIRS
)

mark_as_advanced(
    OpenCG_LIBRARIES
    OpenCG_INCLUDE_DIRS
)
