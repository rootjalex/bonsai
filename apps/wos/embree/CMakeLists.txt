cmake_minimum_required(VERSION 3.14)
project(embree_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set conda paths BEFORE any find_package calls
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    # Set as system include directories (highest priority)
    include_directories(SYSTEM ${CONDA_PREFIX}/include)
    link_directories(${CONDA_PREFIX}/lib)
    list(APPEND CMAKE_PREFIX_PATH ${CONDA_PREFIX})
    
    # Also set C flags to ensure Embree's dependencies find headers
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CONDA_PREFIX}/include")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CONDA_PREFIX}/include")
endif()

# Find Embree package
find_package(embree 4.0 QUIET)

if(NOT embree_FOUND)
    message(STATUS "")
    message(STATUS "============================================")
    message(STATUS "Embree 4 not found!")
    message(STATUS "============================================")
    message(STATUS "Please install Embree using one of these methods:")
    message(STATUS "")
    message(STATUS "1. Conda (recommended):")
    message(STATUS "   conda install -c conda-forge embree")
    message(STATUS "")
    message(STATUS "2. Homebrew:")
    message(STATUS "   brew install embree")
    message(STATUS "")
    message(STATUS "3. Specify embree_DIR manually:")
    message(STATUS "   cmake .. -Dembree_DIR=/path/to/embree/lib/cmake/embree-X.X.X")
    message(STATUS "")
    message(STATUS "4. Download from:")
    message(STATUS "   https://github.com/RenderKit/embree/releases")
    message(STATUS "")
    message(FATAL_ERROR "Cannot continue without Embree")
endif()

message(STATUS "Embree version: ${embree_VERSION}")
message(STATUS "Embree includes: ${EMBREE_INCLUDE_DIRS}")

find_package(Eigen3 REQUIRED)

# Create object library for main file
add_library(main_${BVH_SUFFIX}_obj OBJECT main_${BVH_SUFFIX}.cpp)
target_include_directories(main_${BVH_SUFFIX}_obj
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../../..
    ${CMAKE_SOURCE_DIR}/../../../..
    ${CMAKE_SOURCE_DIR}/../../../runtime/CPP
)
target_link_libraries(main_${BVH_SUFFIX}_obj
    PRIVATE
    embree
    Eigen3::Eigen
)

# Get APPLICATION and LAYOUT from cache
set(LAYOUT "" CACHE STRING "Layout name for Bonsai build")
set(APPLICATION "" CACHE STRING "Application name")

set(BONSAI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${APPLICATION}.h")
set(BONSAI_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${APPLICATION}.cpp")

if(NOT EXISTS "${BONSAI_HEADER}")
    message(FATAL_ERROR "Generated header not found: ${BONSAI_HEADER}")
endif()

if(NOT EXISTS "${BONSAI_SOURCE}")
    message(FATAL_ERROR "Generated source not found: ${BONSAI_SOURCE}")
endif()

# Create final executable
add_executable(${APPLICATION}_${LAYOUT}.out
    ${BONSAI_SOURCE}
    $<TARGET_OBJECTS:main_${BVH_SUFFIX}_obj>
)

target_include_directories(${APPLICATION}_${LAYOUT}.out
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../../..
    ${CMAKE_SOURCE_DIR}/../../../..
    ${CMAKE_SOURCE_DIR}/../../../runtime/CPP
)

target_link_libraries(${APPLICATION}_${LAYOUT}.out
    PRIVATE
    embree
    Eigen3::Eigen
)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${APPLICATION}_${LAYOUT}.out PRIVATE -O3 -march=native)
    target_compile_options(main_${BVH_SUFFIX}_obj PRIVATE -O3 -march=native)
endif()