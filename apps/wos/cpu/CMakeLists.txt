cmake_minimum_required(VERSION 3.14)
project(closest_point_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Layout and application name (passed from bash script)
set(LAYOUT "" CACHE STRING "Layout name for Bonsai build")
set(APPLICATION "" CACHE STRING "Application name")

# Option for FCPW vectorization
option(FCPW_USE_ENOKI "Enable FCPW vectorization with Enoki" OFF)

# Fetch FCPW from GitHub
include(FetchContent)

# Fetch polyscope for mesh loading (same as FCPW demo)
FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG master
)
FetchContent_MakeAvailable(polyscope)

FetchContent_Declare(
    fcpw
    GIT_REPOSITORY https://github.com/rohan-sawhney/fcpw.git
    GIT_TAG e36bc9b34af6088fb78ddbb6a93e26686779678a
)

# Configure FCPW options before making it available
set(FCPW_USE_ENOKI ${FCPW_USE_ENOKI} CACHE BOOL "Enable Enoki vectorization" FORCE)
set(FCPW_BUILD_TESTS OFF CACHE BOOL "Build FCPW tests" FORCE)
set(FCPW_BUILD_EXAMPLES OFF CACHE BOOL "Build FCPW examples" FORCE)

FetchContent_MakeAvailable(fcpw)

# Check if generated files exist
set(BONSAI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${APPLICATION}.h")
set(BONSAI_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${APPLICATION}.cpp")

if(NOT EXISTS "${BONSAI_HEADER}")
    message(FATAL_ERROR "Generated header not found: ${BONSAI_HEADER}")
endif()

if(NOT EXISTS "${BONSAI_SOURCE}")
    message(FATAL_ERROR "Generated source not found: ${BONSAI_SOURCE}")
endif()

# Main executable with Bonsai generated code
add_executable(${APPLICATION}_${LAYOUT}.out
    main.cpp
    ${BONSAI_SOURCE}
)

target_include_directories(${APPLICATION}_${LAYOUT}.out
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${APPLICATION}_${LAYOUT}.out
    PRIVATE
    fcpw
    polyscope
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${APPLICATION}_${LAYOUT}.out PRIVATE -Wall -Wextra -O3)
endif()

message(STATUS "FCPW vectorization (Enoki): ${FCPW_USE_ENOKI}")