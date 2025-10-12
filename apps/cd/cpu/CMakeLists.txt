cmake_minimum_required(VERSION 3.10)
project(collision-detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Try to find packages locally first (conda or system)
find_package(Eigen3 3.0.5 QUIET CONFIG)
find_package(fcl QUIET CONFIG)

# If Eigen3 not found, fetch from GitHub
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally, fetching from GitHub...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# If FCL not found, fetch dependencies and FCL from GitHub
if(NOT fcl_FOUND)
    message(STATUS "FCL not found locally, fetching from GitHub...")
    
    # FCL requires libccd - fetch it first
    FetchContent_Declare(
        ccd
        GIT_REPOSITORY https://github.com/danfis/libccd.git
        GIT_TAG v2.1
        GIT_SHALLOW TRUE
    )
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ccd)
    
    # Now fetch FCL
    FetchContent_Declare(
        fcl
        GIT_REPOSITORY https://github.com/flexible-collision-library/fcl.git
        GIT_TAG 0.7.0
        GIT_SHALLOW TRUE
    )
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(FCL_WITH_OCTOMAP OFF CACHE BOOL "" FORCE)  # Disable optional octomap dependency
    set(FCL_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fcl)
endif()

set(APPLICATION "cd")
message(STATUS "building with LAYOUT: ${LAYOUT}")
message(STATUS "building with APPLICATION: ${APPLICATION}")

add_library(main_library main.cpp)
target_link_libraries(main_library PUBLIC Eigen3::Eigen fcl)
target_include_directories(main_library PUBLIC         
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../../..
    ${CMAKE_SOURCE_DIR}/../../../..
    ${CMAKE_SOURCE_DIR}/../../../runtime/CPP    
)
target_compile_options(main_library PUBLIC -O3 -march=native)

add_executable(${APPLICATION}_${LAYOUT}
    main.cpp
    ${CMAKE_SOURCE_DIR}/${APPLICATION}.cpp
)

target_compile_options(${APPLICATION}_${LAYOUT} PRIVATE -O3 -march=native)
target_include_directories(${APPLICATION}_${LAYOUT} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../../..
    ${CMAKE_SOURCE_DIR}/../../../..
    ${CMAKE_SOURCE_DIR}/../../../apps/${APPLICATION}
    ${CMAKE_SOURCE_DIR}/../../../runtime/CPP        
)

target_link_libraries(${APPLICATION}_${LAYOUT} PRIVATE Eigen3::Eigen fcl)

set_target_properties(${APPLICATION}_${LAYOUT} PROPERTIES
    OUTPUT_NAME "${APPLICATION}_${LAYOUT}.out"
)