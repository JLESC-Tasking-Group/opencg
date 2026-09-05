# CMake Minimum version
cmake_minimum_required(VERSION 3.10)
include(CTest)

# Project name
project(tests)

# list of tests
set(TEST_SOURCES
    batch.cc
    batch-random.cc
    sequence.cc
    h2d-distribute.cc
)

# Tests that compile and run a program, so they need the LLVM-backed passes
# (prog-fuse / jit); without USE_LLVM those abort by design.
if (USE_LLVM)
    list(APPEND TEST_SOURCES
        prog-fuse.cc
        jit-entry-shape.cc
    )
endif()

# Loop over each test source file
foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE ${CGIR})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
