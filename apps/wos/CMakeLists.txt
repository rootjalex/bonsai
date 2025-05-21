cmake_minimum_required(VERSION 3.15...3.27)
project(wos-test-bonsai LANGUAGES CXX CUDA)

set(EIGEN3_INCLUDE_DIR "/scratch/ajroot/zombie-mirror/deps/fcpw/deps/eigen-git-mirror")

# Enable LTO globally (if supported)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO/LTO not supported: ${ipo_error}")
endif()

# Set high optimization level
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_ARCHITECTURES 89)

# Ensure you're building in Release mode (or set manually when invoking CMake)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ZOMBIE_DIR /scratch/ajroot//zombie-mirror)
option(ZOMBIE_BUILD_DEMO OFF)
option(ZOMBIE_BUILD_TESTS OFF)

# Add zombie as a subdirectory, so the "zombie" target is available
add_subdirectory(${ZOMBIE_DIR} ${CMAKE_BINARY_DIR}/zombie)

# Also add its dependencies
if(NOT TARGET geometry-central)
  add_subdirectory(${ZOMBIE_DIR}/test/deps/geometry-central ${CMAKE_BINARY_DIR}/geometry-central)
endif()

if(NOT TARGET polyscope)
  add_subdirectory(${ZOMBIE_DIR}/test/deps/polyscope ${CMAKE_BINARY_DIR}/polyscope)
endif()

# Include path
set(ZOMBIE_TEST_LIBIGL_INCLUDES ${ZOMBIE_DIR}/test/deps/libigl/include)
set(ZOMBIE_TEST_ARGS_INCLUDES ${ZOMBIE_DIR}/test/deps/args)
set(ZOMBIE_TEST_PCG32_INCLUDES ${ZOMBIE_DIR}/deps/pcg32)
set(ZOMBIE_TEST_NANOFLANN_INCLUDES ${ZOMBIE_DIR}/deps/nanoflann/include)
set(ZOMBIE_EIGEN_INCLUDES ${ZOMBIE_DIR}/deps/fcpw/deps/eigen-git-mirror)

# Build test_wos
add_executable(test_wos_bonsai solve.cpp solve_bonsai.cu)

target_compile_options(test_wos_bonsai PRIVATE -O3 -march=native)
#target_link_options(test_wos_bonsai PRIVATE -flto)

set_target_properties(test_wos_bonsai PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
)

# Link libraries
target_link_libraries(test_wos_bonsai
    zombie
    geometry-central
    polyscope
)

# Add include directories
target_include_directories(test_wos_bonsai PRIVATE
    ${ZOMBIE_TEST_LIBIGL_INCLUDES}
    ${ZOMBIE_TEST_ARGS_INCLUDES}
    ${ZOMBIE_TEST_PCG32_INCLUDES}
    ${ZOMBIE_TEST_NANOFLANN_INCLUDES}
    ${ZOMBIE_DIR}/include
    ${ZOMBIE_EIGEN_INCLUDES}
    /scratch/ajroot/_bonsai/runtime/CUDA
    /usr/local/cuda/include
)
