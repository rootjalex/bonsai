cmake_minimum_required(VERSION 3.30)
project(collision-detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fcl REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
target_compile_options(main_library PUBLIC)
target_link_options(main_library PUBLIC)



add_executable(${APPLICATION}_${LAYOUT}
    main.cpp
    ${CMAKE_SOURCE_DIR}/${APPLICATION}.cpp
)

target_compile_options(${APPLICATION}_${LAYOUT} PRIVATE -O3 -march=native -g)
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
