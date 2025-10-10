cmake_minimum_required(VERSION 3.14)
project(closest_point_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Find FCL (assuming it's already installed)
# find_package(fcl REQUIRED)

# Main executable
add_executable(closest_point_benchmark main.cpp)

# Link libraries
target_link_libraries(closest_point_benchmark
    PRIVATE
    fcpw
    polyscope
)

# Include directories
# target_include_directories(closest_point_benchmark
#     PRIVATE
#     ${FCL_INCLUDE_DIRS}
# )

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(closest_point_benchmark PRIVATE -Wall -Wextra -O3)
endif()

# Add your bonsai library (adjust path as needed)
# target_link_libraries(closest_point_benchmark PRIVATE bonsai)
# target_include_directories(closest_point_benchmark PRIVATE ${BONSAI_INCLUDE_DIR})

message(STATUS "FCPW vectorization (Enoki): ${FCPW_USE_ENOKI}")