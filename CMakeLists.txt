# CMake Minimum version
cmake_minimum_required(VERSION 3.17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

SET(OpenCG opencg)

# Project name
project(${OpenCG})

#########
# RPATH #
#########
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

###############
# C++ version #
###############

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Optional: disables compiler-specific extensions

###########
# OPTIONS #
###########


##########
# OpenCG #
##########

SET(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/passes/batch.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/passes/copy-fuse.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/passes/copy-normalize.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/passes/prog-fuse.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/passes/reduce-edge.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/passes/reduce-node.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/command-graph.cc
)

# OpenCG Lib
add_library(${OpenCG} SHARED ${SOURCE_FILES})
set_target_properties(${OpenCG} PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set_target_properties(${OpenCG} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

# targets to install (lib)
set(OpenCG_TARGETS ${OpenCG})

################
# INSTALLATION #
################

# Installation - lib and include files
install(TARGETS ${OpenCG_TARGETS} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Installation - cmake files
target_include_directories(${OpenCG} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
message(STATUS "Targets are ${OpenCG_TARGETS}")
install(TARGETS ${OpenCG_TARGETS}
        EXPORT OpenCGTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT OpenCGTargets
        FILE OpenCGTargets.cmake
        NAMESPACE OpenCG::
        DESTINATION lib/cmake/OpenCG)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/OpenCGConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCGConfig.cmake"
    INSTALL_DESTINATION lib/cmake/OpenCG
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCGConfig.cmake"
    DESTINATION lib/cmake/OpenCG
)

################
# Sub projects #
################

add_subdirectory(tests/)
enable_testing()
