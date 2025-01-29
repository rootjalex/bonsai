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
