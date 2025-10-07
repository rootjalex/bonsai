cmake_minimum_required(VERSION 3.20)
project(pbrt_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ---------------------------------------------------------------------------
# OpenMP setup (Apple Silicon / macOS)
# ---------------------------------------------------------------------------
if(APPLE)
    # Prefer Homebrew LLVM for proper OpenMP support
    if(EXISTS "/opt/homebrew/opt/llvm/bin/clang++")
        set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
        set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
        message(STATUS "Using Homebrew LLVM for OpenMP support")
    endif()
endif()

# Now find OpenMP
find_package(OpenMP)

# ---------------------------------------------------------------------------
# Fetch PBRT-v4 (build externally)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    pbrt
    GIT_REPOSITORY https://github.com/mmp/pbrt-v4.git
    GIT_TAG        master
)
FetchContent_GetProperties(pbrt)

if(NOT pbrt_POPULATED)
    FetchContent_Populate(pbrt)

    message(STATUS "Configuring PBRT-v4 in ${pbrt_BINARY_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${pbrt_SOURCE_DIR} -B ${pbrt_BINARY_DIR} 
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DPBRT_USE_SANITIZERS=OFF
        RESULT_VARIABLE PBRT_CONFIG_RESULT
    )
    
    if(NOT PBRT_CONFIG_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to configure PBRT")
    endif()
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${pbrt_BINARY_DIR} --target pbrt_lib -j
        RESULT_VARIABLE PBRT_BUILD_RESULT
    )
    
    if(NOT PBRT_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build PBRT")
    endif()
    
    # Store the dependencies that PBRT found
    set(PBRT_DEPS_DIR "${pbrt_BINARY_DIR}/_deps")
endif()

# ---------------------------------------------------------------------------
# Your executable (using main.cpp as the source)
# ---------------------------------------------------------------------------
add_executable(pbrt_runner main.cpp)

target_include_directories(pbrt_runner PRIVATE
    ${pbrt_SOURCE_DIR}/src
    ${pbrt_BINARY_DIR}/src
)

# Link PBRT static library and its dependencies
target_link_libraries(pbrt_runner PRIVATE
    ${pbrt_BINARY_DIR}/libpbrt_lib.a
)

# Required system libraries for PBRT
find_library(DOUBLE_CONVERSION_LIB NAMES double-conversion 
    PATHS /opt/homebrew/lib /usr/local/lib 
    REQUIRED)
find_library(UTF8PROC_LIB NAMES utf8proc 
    PATHS /opt/homebrew/lib /usr/local/lib 
    REQUIRED)
find_library(LIBDEFLATE_LIB NAMES deflate 
    PATHS /opt/homebrew/lib /usr/local/lib 
    REQUIRED)

message(STATUS "Found double-conversion: ${DOUBLE_CONVERSION_LIB}")
message(STATUS "Found utf8proc: ${UTF8PROC_LIB}")
message(STATUS "Found libdeflate: ${LIBDEFLATE_LIB}")

target_link_libraries(pbrt_runner PRIVATE
    ${DOUBLE_CONVERSION_LIB}
    ${UTF8PROC_LIB}
    ${LIBDEFLATE_LIB}
)

# Find OpenEXR and Imath (also typically needed)
find_package(OpenEXR QUIET)
if(OpenEXR_FOUND)
    target_link_libraries(pbrt_runner PRIVATE OpenEXR::OpenEXR)
else()
    # Try to find manually
    find_library(OPENEXR_LIB NAMES OpenEXR PATHS /opt/homebrew/lib /usr/local/lib)
    find_library(IMATH_LIB NAMES Imath PATHS /opt/homebrew/lib /usr/local/lib)
    if(OPENEXR_LIB)
        target_link_libraries(pbrt_runner PRIVATE ${OPENEXR_LIB})
    endif()
    if(IMATH_LIB)
        target_link_libraries(pbrt_runner PRIVATE ${IMATH_LIB})
    endif()
endif()

find_package(Imath QUIET)
if(Imath_FOUND)
    target_link_libraries(pbrt_runner PRIVATE Imath::Imath)
endif()

# ---------------------------------------------------------------------------
# OpenMP linking
# ---------------------------------------------------------------------------
if(OpenMP_CXX_FOUND)
    target_link_libraries(pbrt_runner PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(pbrt_runner PRIVATE _OPENMP)
elseif(APPLE)
    # Manual fallback for macOS
    target_compile_options(pbrt_runner PRIVATE -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include)
    target_link_options(pbrt_runner PRIVATE -L/opt/homebrew/opt/libomp/lib -lomp)
    target_compile_definitions(pbrt_runner PRIVATE _OPENMP)
endif()

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(pbrt_runner PRIVATE -Wall -Wextra -Wno-unused-parameter)
elseif (MSVC)
    target_compile_options(pbrt_runner PRIVATE /W4 /permissive-)
endif()
