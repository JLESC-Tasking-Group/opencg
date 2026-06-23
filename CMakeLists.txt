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

include(BooleanUtils)
ocgoption(BUILD_C_API "Build the C API." ON)
ocgoption(USE_LLVM    "Use LLVM for IR manipulation (required by prog-fuse)." OFF)
ocgoption(USE_PROTEUS "Use Proteus for JIT (implies USE_LLVM)." OFF)
ocgoption(USE_MLIR    "Use MLIR for the command-graph IR and optimization passes." OFF)

# Proteus requires LLVM; force USE_LLVM ON if USE_PROTEUS is requested.
if (USE_PROTEUS)
    set(USE_LLVM 1)
endif()

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

if (BUILD_C_API)
    list(APPEND SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/api/c/src/opencg.cc)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api/c/src)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api/c/include)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/api/c/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")
endif()

###################################
# MLIR `cg` dialect + IR pipeline #
###################################
#
# All MLIR usage is confined to src/mlir/ and compiled into an OBJECT library
# `opencgmlir`, whose objects are folded into libopencg.so. This keeps MLIR an
# *implementation detail* of OpenCG: no installed/public header includes MLIR,
# so consumers (e.g. XKRT) never need MLIR to build against OpenCG.
#
# Requires an MLIR install matching your LLVM. Point CMake at it with:
#   -DUSE_MLIR=ON -DMLIR_DIR=/path/to/llvm-project/<build>/lib/cmake/mlir
#
if (USE_MLIR)
    find_package(MLIR REQUIRED CONFIG)
    message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION} in: ${MLIR_DIR}")
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (for MLIR) in: ${LLVM_DIR}")

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)

    set(OPENCG_MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/mlir)

    # Generate the `cg` dialect from TableGen. The *.inc files land in
    # ${CMAKE_CURRENT_BINARY_DIR} and are pulled in by the dialect headers.
    set(LLVM_TARGET_DEFINITIONS ${OPENCG_MLIR_SRC}/CG/CGOps.td)
    mlir_tablegen(CGDialect.h.inc   -gen-dialect-decls -dialect=cg)
    mlir_tablegen(CGDialect.cpp.inc -gen-dialect-defs  -dialect=cg)
    mlir_tablegen(CGTypes.h.inc     -gen-typedef-decls -typedefs-dialect=cg)
    mlir_tablegen(CGTypes.cpp.inc   -gen-typedef-defs  -typedefs-dialect=cg)
    mlir_tablegen(CGOps.h.inc       -gen-op-decls)
    mlir_tablegen(CGOps.cpp.inc     -gen-op-defs)
    add_public_tablegen_target(OpenCGMLIRIncGen)

    # The MLIR translation units, isolated in an OBJECT library so MLIR/LLVM
    # build flags (notably -fno-rtti) do not leak onto the rest of OpenCG.
    add_library(opencgmlir OBJECT
        ${OPENCG_MLIR_SRC}/CG/CGDialect.cpp
        ${OPENCG_MLIR_SRC}/import.cpp
        ${OPENCG_MLIR_SRC}/export.cpp
        ${OPENCG_MLIR_SRC}/optimize.cpp
        ${OPENCG_MLIR_SRC}/passes/ReduceNode.cpp
        ${OPENCG_MLIR_SRC}/passes/ReduceEdge.cpp
    )
    add_dependencies(opencgmlir OpenCGMLIRIncGen)
    set_target_properties(opencgmlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(opencgmlir PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}                 # generated CG*.inc
        ${OPENCG_MLIR_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )
    target_compile_definitions(opencgmlir PRIVATE ${LLVM_DEFINITIONS})
    if (NOT LLVM_ENABLE_RTTI)
        target_compile_options(opencgmlir PRIVATE -fno-rtti)
    endif()

    # MLIR component libraries needed by the dialect, bridge and passes.
    set(OPENCG_MLIR_LIBS
        MLIRIR
        MLIRSupport
        MLIRParser
        MLIRPass
        MLIRTransformUtils
    )
    target_link_libraries(opencgmlir PRIVATE ${OPENCG_MLIR_LIBS})

    # Fold the MLIR objects into libopencg and resolve MLIR symbols there.
    list(APPEND SOURCE_FILES $<TARGET_OBJECTS:opencgmlir>)
endif()

# OpenCG Lib
add_library(${OpenCG} SHARED ${SOURCE_FILES})

if (USE_MLIR)
    add_dependencies(${OpenCG} OpenCGMLIRIncGen)
    target_link_libraries(${OpenCG} PRIVATE ${OPENCG_MLIR_LIBS})
endif()
set_target_properties(${OpenCG} PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set_target_properties(${OpenCG} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

# targets to install (lib)
set(OpenCG_TARGETS ${OpenCG})

###############
# DEPENDENCES #
###############

if (USE_PROTEUS)
    # Find Proteus first so that its find_package(LLVM) call populates
    # LLVM_DIR, LLVM_INCLUDE_DIRS, LLVM_DEFINITIONS, LLVM_LINK_LLVM_DYLIB,
    # etc. before our own find_package(LLVM) runs.  This guarantees that both
    # OpenCG and Proteus resolve LLVM from the exact same installation.
    find_package(proteus CONFIG REQUIRED)
    find_package(Threads REQUIRED)
    target_link_libraries(opencg PRIVATE proteusFrontend)
endif()

if (USE_LLVM)
    # Allow the user to point at a specific LLVM installation via
    # -DLLVM_DIR=<path/to/lib/cmake/llvm> or the legacy
    # -DLLVM_INSTALL_DIR=<prefix>.  When USE_PROTEUS is also set,
    # find_package(proteus) above has already set LLVM_DIR to the
    # installation Proteus was built against, so both use the same LLVM.
    if (DEFINED LLVM_INSTALL_DIR AND NOT DEFINED LLVM_DIR)
        set(LLVM_DIR "${LLVM_INSTALL_DIR}/lib/cmake/llvm")
    endif()

    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in: ${LLVM_DIR}")

    # Propagate LLVM preprocessor definitions (e.g. -D__STDC_CONSTANT_MACROS).
    target_compile_definitions(${OpenCG} PRIVATE ${LLVM_DEFINITIONS})

    # Add LLVM's own include directories (source tree + build tree headers).
    target_include_directories(${OpenCG} PRIVATE ${LLVM_INCLUDE_DIRS})

    # Link LLVM.
    #
    # When USE_PROTEUS is also enabled, Proteus already links LLVM (either the
    # monolithic libLLVM.so dylib or individual static archives).  Linking a
    # *second* set of LLVM libraries — even from the same installation — causes
    # two copies of LLVM's global state (StringMap, LLVMContext registry, …) to
    # be initialised in the same process, which triggers assertion failures such
    # as the StringMap "NumItems + NumTombstones <= NumBuckets" abort.
    #
    # The safe strategy is:
    #
    #   USE_LLVM only (no Proteus):
    #     Link per-component static libraries normally; they are the only LLVM
    #     copy in the process so there is no duplication.
    #
    #   USE_LLVM + USE_PROTEUS + LLVM_LINK_LLVM_DYLIB:
    #     Both OpenCG and Proteus link the monolithic libLLVM.so.  The dynamic
    #     linker loads it once; a single global state is shared.
    #
    #   USE_LLVM + USE_PROTEUS + static LLVM (no dylib):
    #     Proteus hides its LLVM archives via --exclude-libs on its INTERFACE
    #     link flags, so the symbols are not exported from libproteus into
    #     libopencg.so.  OpenCG must link the same archives directly.
    #     Because find_package(proteus) ran first and fixed LLVM_DIR, both pick
    #     the same .a files, so the static linker sees only one copy of each
    #     translation unit and there is no duplicate global state at runtime.

    if (USE_PROTEUS AND LLVM_LINK_LLVM_DYLIB)
        # Single libLLVM.so — one global state, no duplication.
        message(STATUS "LLVM: linking monolithic dylib (LLVM_LINK_LLVM_DYLIB=ON)")
        target_link_libraries(${OpenCG} PRIVATE LLVM)
    else()
        # Static component libs (standalone or Proteus-static path).
        llvm_map_components_to_libnames(LLVM_LIBS
            core            # LLVMCore      – IRBuilder, Module, Function, …
            irreader        # LLVMIRReader  – parseIR()
            bitwriter       # LLVMBitWriter – WriteBitcodeToFile()
            linker          # LLVMLinker    – Linker::linkInModule()
            support         # LLVMSupport   – raw_ostream, MemoryBuffer, SourceMgr
            analysis        # LLVMAnalysis  – verifyModule()
            mc              # LLVMMC        – TargetRegistry::lookupTarget()
            target          # LLVMTarget    – TargetMachine, InitializeAllTargets()
            ${LLVM_TARGETS_TO_BUILD}  # architecture-specific codegen/MC/asm libs
        )
        message(STATUS "LLVM: linking component libs: ${LLVM_LIBS}")
        target_link_libraries(${OpenCG} PRIVATE ${LLVM_LIBS})
    endif()
endif()

################
# INSTALLATION #
################

# Header file generated
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/opencg")
configure_file(
    ${CMAKE_SOURCE_DIR}/include/opencg/support.h.in
    ${CMAKE_BINARY_DIR}/include/opencg/support.h
)
install(
    FILES "${CMAKE_BINARY_DIR}/include/opencg/support.h"
    DESTINATION include/opencg
)

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
