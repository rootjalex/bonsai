cmake_minimum_required(VERSION 3.14)
project(closest_point_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set conda paths BEFORE any FetchContent calls
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    # Set as system include directories (highest priority)
    include_directories(SYSTEM ${CONDA_PREFIX}/include)
    link_directories(${CONDA_PREFIX}/lib)
    list(APPEND CMAKE_PREFIX_PATH ${CONDA_PREFIX})
    
    # Also set C flags to ensure GLFW's C code finds the headers
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CONDA_PREFIX}/include")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CONDA_PREFIX}/include")
endif()

include(FetchContent)

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
set(FCPW_USE_ENOKI ${FCPW_USE_ENOKI} CACHE BOOL "Enable Enoki vectorization" FORCE)
set(FCPW_BUILD_TESTS OFF CACHE BOOL "Build FCPW tests" FORCE)
set(FCPW_BUILD_EXAMPLES OFF CACHE BOOL "Build FCPW examples" FORCE)
FetchContent_MakeAvailable(fcpw)

find_package(Eigen3 REQUIRED)

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
    fcpw
    polyscope
    Eigen3::Eigen
)

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
    fcpw
    polyscope
    Eigen3::Eigen
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${APPLICATION}_${LAYOUT}.out PRIVATE -O3)
endif()