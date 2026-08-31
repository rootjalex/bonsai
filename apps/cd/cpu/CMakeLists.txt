cmake_minimum_required(VERSION 3.10)
project(collision-detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(APPLICATION "cd")
message(STATUS "building with LAYOUT: ${LAYOUT}")
message(STATUS "building with APPLICATION: ${APPLICATION}")

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

set_target_properties(${APPLICATION}_${LAYOUT} PROPERTIES
    OUTPUT_NAME "${APPLICATION}_${LAYOUT}.out"
)

# On conda environments, link against conda's libstdc++ to avoid version mismatch
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    target_link_options(${APPLICATION}_${LAYOUT} PRIVATE "-Wl,-rpath,${CONDA_PREFIX}/lib")
endif()
