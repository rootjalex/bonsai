cmake_minimum_required(VERSION 3.30)
project(collision-detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3 -march=native")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

option(USE_PROFILING "Use the profiled builds" OFF)

if (USE_PROFILING)
  message(STATUS "Using profiled build")
  set(FCL_PATH "/Users/ajroot/projects/pldi-bonsai/deps/fcl/build_profiled/install")
  set(BONSAI_SRC "cd_profiled.cpp")
else()
  message(STATUS "Using unprofiled build")
  set(FCL_PATH "/scratch/ajroot/pldi-bonsai/deps/fcl/build/install")
  set(BONSAI_SRC "cd_gen.cpp")
endif()

if(NOT EXISTS ${FCL_PATH})
  message(FATAL_ERROR "FCL_PATH does not exist: ${FCL_PATH}")
endif()

find_package(fcl CONFIG REQUIRED
  PATHS
   ${FCL_PATH}
  NO_DEFAULT_PATH
)

message(STATUS "FCL include dirs: ${fcl_INCLUDE_DIRS}")

# If using imported targets (modern CMake, which fcl usually does)
get_target_property(FCL_LIB_LOCATION fcl LOCATION)
message(STATUS "FCL library location: ${FCL_LIB_LOCATION}")

get_target_property(FCL_LIB_INCLUDE_DIRS fcl INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "FCL include directories (from target): ${FCL_LIB_INCLUDE_DIRS}")

add_executable(main
  main.cpp
  # The bonsai object file after compilation.
  ${BONSAI_SRC}
)

# Add your include path (relative to the current CMakeLists.txt)
target_include_directories(main
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(main
  PRIVATE
    Eigen3::Eigen # https://eigen.tuxfamily.org/index.php?title=Main_Page
    fcl           # https://github.com/flexible-collision-library/fcl
)

if (USE_PROFILING)
  target_compile_definitions(main PRIVATE AJR_PROFILING=1)
endif()
