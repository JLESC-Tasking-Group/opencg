# Try to find CGIR headers and libraries.
#
# Usage of this module as follows:
#
#     find_package(CGIR)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  CGIR_PREFIX         Set this variable to the root installation of
#                      libpapi if the module has problems finding the
#                      proper installation path.
#
# Variables defined by this module:
#
#  CGIR_FOUND              System has CGIR libraries and headers
#  CGIR_LIBRARIES          The CGIR library
#  CGIR_INCLUDE_DIRS       The location of CGIR headers

find_library(CGIR_LIBRARIES NAMES libcgir.so)
find_path(CGIR_INCLUDE_DIRS NAMES cgir/cgir.hpp)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CGIR DEFAULT_MSG
    CGIR_LIBRARIES
    CGIR_INCLUDE_DIRS
)

mark_as_advanced(
    CGIR_LIBRARIES
    CGIR_INCLUDE_DIRS
)
