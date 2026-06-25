# CMake Minimum version
cmake_minimum_required(VERSION 3.17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

SET(CGIR cgir)

# Project name
project(${CGIR})

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
cgiroption(BUILD_C_API "Build the C API." ON)

# USE_LLVM / USE_MLIR default OFF, but auto-enable when the caller points cmake
# at an install — the trailing variable names are checked by cgiroption (see
# cmake/BooleanUtils.cmake): if any is defined, the option turns ON so the
# variable is consumed (no "Manually-specified variables were not used" warning).
cgiroption(USE_LLVM "Use LLVM for IR manipulation and JIT (required by prog-fuse)." OFF LLVM_DIR LLVM_INSTALL_DIR)
cgiroption(USE_MLIR "Use MLIR for the command-graph IR and optimization passes." OFF MLIR_DIR)

##########
# CGIR #
##########

SET(SOURCE_FILES
    # common (POD + MLIR)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cgir.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/prog-fuse-llvmir.cc
    # POD interface implementation
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pod/command-graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pod/batch.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pod/copy-fuse.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pod/copy-normalize.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pod/prog-fuse.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pod/reduce-edge.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pod/reduce-node.cc
)

if (BUILD_C_API)
    list(APPEND SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/api/c/src/cgir.cc)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api/c/src)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api/c/include)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/api/c/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")
endif()

###################################
# MLIR `cg` dialect + IR pipeline #
###################################
#
# The MLIR implementation lives in src/mlir/ and is folded into libcgir.so.
# The `cg` dialect headers (include/cgir/mlir/) ARE public: another runtime
# can build cg ops, run the passes, and convert to/from the POD command graph.
# Using them requires MLIR, but the POD headers (cgir/command*.hpp) do not,
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

    set(CGIR_MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/mlir)
    set(CGIR_MLIR_TD  ${CMAKE_CURRENT_SOURCE_DIR}/include/cgir/mlir/CG)

    # TableGen needs the dialect .td directory and MLIR's own .td on its
    # include path (for `include "CGDialect.td"` and `include "mlir/IR/..."`).
    include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${CGIR_MLIR_TD})

    # Generate the `cg` dialect from TableGen. The *.inc files land in
    # ${CMAKE_CURRENT_BINARY_DIR}; the public dialect headers include them with
    # quote-includes, so on install we place the decl *.inc next to the headers.
    set(LLVM_TARGET_DEFINITIONS ${CGIR_MLIR_TD}/CGOps.td)
    mlir_tablegen(CGDialect.h.inc   -gen-dialect-decls -dialect=cg)
    mlir_tablegen(CGDialect.cpp.inc -gen-dialect-defs  -dialect=cg)
    mlir_tablegen(CGTypes.h.inc     -gen-typedef-decls -typedefs-dialect=cg)
    mlir_tablegen(CGTypes.cpp.inc   -gen-typedef-defs  -typedefs-dialect=cg)
    mlir_tablegen(CGOps.h.inc       -gen-op-decls)
    mlir_tablegen(CGOps.cpp.inc     -gen-op-defs)
    add_public_tablegen_target(CGIRMLIRIncGen)

    # The MLIR translation units, isolated in an OBJECT library so MLIR/LLVM
    # build flags (notably -fno-rtti) do not leak onto the rest of CGIR.
    add_library(cgirmlir OBJECT
        ${CGIR_MLIR_SRC}/CG/CGDialect.cpp
        ${CGIR_MLIR_SRC}/import.cpp
        ${CGIR_MLIR_SRC}/export.cpp
        ${CGIR_MLIR_SRC}/optimize.cpp
        ${CGIR_MLIR_SRC}/passes/ReduceNode.cpp
        ${CGIR_MLIR_SRC}/passes/ReduceEdge.cpp
        ${CGIR_MLIR_SRC}/passes/CopyNormalize.cpp
        ${CGIR_MLIR_SRC}/passes/CopyFuse.cpp
        ${CGIR_MLIR_SRC}/passes/Batch.cpp
        ${CGIR_MLIR_SRC}/passes/ProgFuse.cpp
    )
    add_dependencies(cgirmlir CGIRMLIRIncGen)
    set_target_properties(cgirmlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(cgirmlir PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}                 # generated CG*.inc (quote-include)
        ${CMAKE_CURRENT_SOURCE_DIR}/include         # public cgir/ + cgir/mlir/ headers
        ${CMAKE_CURRENT_BINARY_DIR}/include         # generated cgir/support.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src             # internal shared headers (prog-fuse-llvmir.hpp)
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )
    target_compile_definitions(cgirmlir PRIVATE ${LLVM_DEFINITIONS})
    if (NOT LLVM_ENABLE_RTTI)
        target_compile_options(cgirmlir PRIVATE -fno-rtti)
    endif()

    # MLIR component libraries needed by the dialect, bridge and passes.
    set(CGIR_MLIR_LIBS
        MLIRIR
        MLIRSupport
        MLIRParser
        MLIRPass
        MLIRTransformUtils
    )
    target_link_libraries(cgirmlir PRIVATE ${CGIR_MLIR_LIBS})

    # Fold the MLIR objects into libcgir and resolve MLIR symbols there.
    list(APPEND SOURCE_FILES $<TARGET_OBJECTS:cgirmlir>)
endif()

# CGIR Lib
add_library(${CGIR} SHARED ${SOURCE_FILES})

# Internal include path so POD/shared translation units can include the shared
# core header (src/prog-fuse-llvmir.hpp) regardless of their subdirectory.
target_include_directories(${CGIR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (USE_MLIR)
    add_dependencies(${CGIR} CGIRMLIRIncGen)
    target_link_libraries(${CGIR} PRIVATE ${CGIR_MLIR_LIBS})
endif()
set_target_properties(${CGIR} PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set_target_properties(${CGIR} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

# targets to install (lib)
set(CGIR_TARGETS ${CGIR})

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
    target_compile_definitions(${CGIR} PRIVATE ${LLVM_DEFINITIONS})

    # Add LLVM's own include directories (source tree + build tree headers).
    target_include_directories(${CGIR} PRIVATE ${LLVM_INCLUDE_DIRS})

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
        passes          # LLVMPasses       - PassBuilder O3 pipeline, LoopFuse, vectorizers
        native          # host target codegen/asm (CodeGen, AsmPrinter, ...)
        nativecodegen   # host target codegen registration
        ${LLVM_TARGETS_TO_BUILD}  # architecture-specific codegen/MC/asm libs
    )
    message(STATUS "LLVM: linking component libs: ${LLVM_LIBS}")
    target_link_libraries(${CGIR} PRIVATE ${LLVM_LIBS})
endif()

################
# INSTALLATION #
################

# Header file generated
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/cgir")
configure_file(
    ${CMAKE_SOURCE_DIR}/include/cgir/support.h.in
    ${CMAKE_BINARY_DIR}/include/cgir/support.h
)
install(
    FILES "${CMAKE_BINARY_DIR}/include/cgir/support.h"
    DESTINATION include/cgir
)

# Installation - lib and include files
install(TARGETS ${CGIR_TARGETS} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Installation - public MLIR `cg` dialect: its TableGen sources and the generated
# declaration .inc files (placed next to the dialect headers so their
# quote-includes resolve against the install tree).
if (USE_MLIR)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cgir/mlir/
            DESTINATION include/cgir/mlir
            FILES_MATCHING PATTERN "*.td")
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CGDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/CGTypes.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/CGOps.h.inc
        DESTINATION include/cgir/mlir/CG
    )
endif()

# Installation - cmake files
target_include_directories(${CGIR} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
message(STATUS "Targets are ${CGIR_TARGETS}")
install(TARGETS ${CGIR_TARGETS}
        EXPORT CGIRTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT CGIRTargets
        FILE CGIRTargets.cmake
        NAMESPACE CGIR::
        DESTINATION lib/cmake/CGIR)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/CGIRConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/CGIRConfig.cmake"
    INSTALL_DESTINATION lib/cmake/CGIR
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CGIRConfig.cmake"
    DESTINATION lib/cmake/CGIR
)

##################
# UTILITY BINARY #
##################

add_executable(cgir_info ${CMAKE_SOURCE_DIR}/src/cgir_info.cc)
target_link_libraries(cgir_info PRIVATE ${CGIR})
install(TARGETS cgir_info RUNTIME DESTINATION bin)

################
# Sub projects #
################

add_subdirectory(tests/)
enable_testing()
