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
ocgoption(USE_LLVM    "Use LLVM for IR manipulation and JIT (required by prog-fuse)." OFF)
ocgoption(USE_MLIR    "Use MLIR for the command-graph IR and optimization passes." OFF)

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
# The MLIR implementation lives in src/mlir/ and is folded into libopencg.so.
# The `cg` dialect headers (include/opencg/mlir/) ARE public: another runtime
# can build cg ops, run the passes, and convert to/from the POD command graph.
# Using them requires MLIR, but the POD headers (opencg/command*.hpp) do not,
# so a runtime that only wants the POD path (e.g. XKRT) never needs MLIR.
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
    set(OPENCG_MLIR_TD  ${CMAKE_CURRENT_SOURCE_DIR}/include/opencg/mlir/CG)

    # TableGen needs the dialect .td directory and MLIR's own .td on its
    # include path (for `include "CGDialect.td"` and `include "mlir/IR/..."`).
    include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${OPENCG_MLIR_TD})

    # Generate the `cg` dialect from TableGen. The *.inc files land in
    # ${CMAKE_CURRENT_BINARY_DIR}; the public dialect headers include them with
    # quote-includes, so on install we place the decl *.inc next to the headers.
    set(LLVM_TARGET_DEFINITIONS ${OPENCG_MLIR_TD}/CGOps.td)
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
        ${OPENCG_MLIR_SRC}/passes/CopyNormalize.cpp
        ${OPENCG_MLIR_SRC}/passes/CopyFuse.cpp
        ${OPENCG_MLIR_SRC}/passes/Batch.cpp
        ${OPENCG_MLIR_SRC}/passes/ProgFuse.cpp
    )
    add_dependencies(opencgmlir OpenCGMLIRIncGen)
    set_target_properties(opencgmlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(opencgmlir PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}                 # generated CG*.inc (quote-include)
        ${CMAKE_CURRENT_SOURCE_DIR}/include         # public opencg/ + opencg/mlir/ headers
        ${CMAKE_CURRENT_BINARY_DIR}/include         # generated opencg/support.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src             # internal shared headers (passes/prog-fuse.hpp)
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

if (USE_LLVM)
    # Allow the user to point at a specific LLVM installation via
    # -DLLVM_DIR=<path/to/lib/cmake/llvm> or the legacy -DLLVM_INSTALL_DIR=<prefix>.
    if (DEFINED LLVM_INSTALL_DIR AND NOT DEFINED LLVM_DIR)
        set(LLVM_DIR "${LLVM_INSTALL_DIR}/lib/cmake/llvm")
    endif()

    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in: ${LLVM_DIR}")

    # Propagate LLVM preprocessor definitions (e.g. -D__STDC_CONSTANT_MACROS).
    target_compile_definitions(${OpenCG} PRIVATE ${LLVM_DEFINITIONS})

    # Add LLVM's own include directories (source tree + build tree headers).
    target_include_directories(${OpenCG} PRIVATE ${LLVM_INCLUDE_DIRS})

    # Link the LLVM component libraries used by prog-fuse: IR manipulation /
    # linking / bitcode plus the in-process ORC JIT (LLJIT) that compiles the
    # fused module (this replaces the former Proteus dependency).
    llvm_map_components_to_libnames(LLVM_LIBS
        core            # LLVMCore         - IRBuilder, Module, Function, ...
        irreader        # LLVMIRReader     - parseIR()
        bitwriter       # LLVMBitWriter    - WriteBitcodeToFile()
        linker          # LLVMLinker       - Linker::linkInModule()
        support         # LLVMSupport      - raw_ostream, MemoryBuffer, SourceMgr
        analysis        # LLVMAnalysis     - verifyModule()
        mc              # LLVMMC           - TargetRegistry::lookupTarget()
        target          # LLVMTarget       - TargetMachine, InitializeAllTargets()
        orcjit          # LLVMOrcJIT       - LLJIT
        executionengine # LLVMExecutionEngine
        jitlink         # LLVMJITLink      - LLJIT object linking
        native          # host target codegen/asm (CodeGen, AsmPrinter, ...)
        nativecodegen   # host target codegen registration
        ${LLVM_TARGETS_TO_BUILD}  # architecture-specific codegen/MC/asm libs
    )
    message(STATUS "LLVM: linking component libs: ${LLVM_LIBS}")
    target_link_libraries(${OpenCG} PRIVATE ${LLVM_LIBS})
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

# Installation - public MLIR `cg` dialect: its TableGen sources and the generated
# declaration .inc files (placed next to the dialect headers so their
# quote-includes resolve against the install tree).
if (USE_MLIR)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/opencg/mlir/
            DESTINATION include/opencg/mlir
            FILES_MATCHING PATTERN "*.td")
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CGDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/CGTypes.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/CGOps.h.inc
        DESTINATION include/opencg/mlir/CG
    )
endif()

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
