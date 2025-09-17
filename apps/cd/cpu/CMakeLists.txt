cmake_minimum_required(VERSION 3.30)
project(collision-detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fcl REQUIRED)

set(APPLICATION "cd")
message(STATUS "building with LAYOUT: ${LAYOUT}")
message(STATUS "building with APPLICATION: ${APPLICATION}")

add_library(main_library main.cpp)
target_link_libraries(main_library PUBLIC Eigen3::Eigen fcl)
target_include_directories(main_library PUBLIC 
    ${CMAKE_SOURCE_DIR}
    apps/${APPLICATION}
    runtime/CPP
)

add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/${APPLICATION}.cpp
    COMMAND ${CMAKE_SOURCE_DIR}/build/compiler 
            -i ${CMAKE_SOURCE_DIR}/kernel/main.bonsai 
            -l ${CMAKE_BINARY_DIR}/${LAYOUT}.bonsai 
            -b cppx 
            -o ${CMAKE_SOURCE_DIR}/${APPLICATION}
    DEPENDS ${CMAKE_SOURCE_DIR}/kernel/main.bonsai
            ${CMAKE_BINARY_DIR}/${LAYOUT}.bonsai
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating C++ code with layout: ${LAYOUT}"
)

add_executable(${APPLICATION}_${LAYOUT}
    main.cpp
    ${CMAKE_SOURCE_DIR}/${APPLICATION}.cpp
)

target_compile_options(${APPLICATION}_${LAYOUT} PRIVATE -O3 -g)
target_include_directories(main_library PUBLIC
    .                                    # -I. (current source directory)
    ../../../apps/${APPLICATION}         # -Iapps/${APPLICATION} 
    ../../../runtime/CPP                 # -Iruntime/CPP (where bonsai runtime lives)
)


target_link_libraries(${APPLICATION}_${LAYOUT} PRIVATE Eigen3::Eigen fcl)

set_target_properties(${APPLICATION}_${LAYOUT} PROPERTIES
    OUTPUT_NAME "${APPLICATION}_${LAYOUT}.out"
)
