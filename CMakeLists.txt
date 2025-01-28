cmake_minimum_required(VERSION 3.16)

project(BonsaiProject)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

# Find LLVM
set(LLVM_ROOT "")
if (EXISTS "${CMAKE_SOURCE_DIR}/deps/llvm-install")
    set(LLVM_ROOT "${CMAKE_SOURCE_DIR}/deps/llvm-install")
elseif(DEFINED ENV{LLVM_ROOT} AND NOT "$ENV{LLVM_ROOT}" STREQUAL "")
    set(LLVM_ROOT "$ENV{LLVM_ROOT}")
else()
    message(FATAL_ERROR "LLVM must be installed. Either use the submodule at deps/llvm-install or set the LLVM_ROOT environment variable to an installation of LLVM 19.1.6")
endif()

list(APPEND CMAKE_PREFIX_PATH ${LLVM_ROOT}/lib/cmake/llvm)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Add LLVM library directory to the linker search paths
link_directories(${LLVM_ROOT}/lib)

# Specify LLVM components
llvm_map_components_to_libnames(LLVM_LIBS 
    core 
    irreader 
    support 
    analysis 
    target
    passes
    native
    # irbuilder
    orcjit
    executionengine # Required for LLJIT
    runtimeDyld # Required for JIT execution
)

# Set RPATH for runtime library search on macOS
set(CMAKE_INSTALL_RPATH "${LLVM_ROOT}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Gather source files for the library
file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.cpp)

add_compile_options(-Wall)

# Create object files from source files
add_library(bonsai_objects OBJECT ${SRC_FILES})
target_include_directories(bonsai_objects PRIVATE ${INCLUDE_DIR})
target_link_libraries(bonsai_objects ${LLVM_LIBS})

# Add the executable target
add_executable(compiler compiler.cpp $<TARGET_OBJECTS:bonsai_objects>)
target_include_directories(compiler PRIVATE ${INCLUDE_DIR})
target_link_libraries(compiler ${LLVM_LIBS})

# Tests
enable_testing()
file(GLOB_RECURSE TEST_FILES ${TESTS_DIR}/*.cpp)

foreach(TEST_FILE IN LISTS TEST_FILES)
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} $<TARGET_OBJECTS:bonsai_objects>)
    target_include_directories(${TEST_NAME} PRIVATE ${INCLUDE_DIR})
    target_link_libraries(${TEST_NAME} ${LLVM_LIBS})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# Custom target for running all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_NAME}
)
