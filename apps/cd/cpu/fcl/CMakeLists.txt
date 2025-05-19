cmake_minimum_required(VERSION 3.10)
project(fcl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fcl REQUIRED)

set(BONSAI_O "${..CMAKE_BINARY_DIR}main.o")

add_executable(main 
  main.cpp
  ${BONSAI_O}
)

target_link_libraries(main
  PRIVATE
    Eigen3::Eigen
    fcl
)