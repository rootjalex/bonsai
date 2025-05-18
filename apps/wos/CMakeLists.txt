cmake_minimum_required(VERSION 3.15...3.27)
project(wos-test-bonsai)

set(ZOMBIE_DIR /Users/ajroot/projects/zombie-mirror)
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

# Include paths
set(ZOMBIE_TEST_LIBIGL_INCLUDES ${ZOMBIE_DIR}/test/deps/libigl/include)
set(ZOMBIE_TEST_ARGS_INCLUDES ${ZOMBIE_DIR}/test/deps/args)
set(ZOMBIE_TEST_PCG32_INCLUDES ${ZOMBIE_DIR}/deps/pcg32)
set(ZOMBIE_TEST_NANOFLANN_INCLUDES ${ZOMBIE_DIR}/deps/nanoflann/include)

# Build test_wos
add_executable(test_wos_bonsai solve.cpp solve_bonsai.o)

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
)
