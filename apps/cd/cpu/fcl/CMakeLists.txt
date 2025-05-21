cmake_minimum_required(VERSION 3.30)
project(collision-detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fcl REQUIRED)

# The bonsai object file after compilation.
set(BONSAI_O "${..CMAKE_BINARY_DIR}main.o")

add_executable(main 
  main.cpp
  ${BONSAI_O}
)

target_link_libraries(main
  PRIVATE
    Eigen3::Eigen # https://eigen.tuxfamily.org/index.php?title=Main_Page
    fcl           # https://github.com/flexible-collision-library/fcl
)
