cmake_minimum_required(VERSION 3.10)
project(EmbreeMinimal)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP (macOS specific handling)
if(APPLE)
    # On macOS, try to find libomp from Homebrew
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    
    if(LIBOMP_PREFIX)
        message(STATUS "Found libomp via Homebrew at: ${LIBOMP_PREFIX}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        set(OPENMP_FOUND TRUE)
    else()
        message(STATUS "libomp not found via Homebrew")
        message(STATUS "Install with: brew install libomp")
        set(OPENMP_FOUND FALSE)
    endif()
else()
    find_package(OpenMP)
    set(OPENMP_FOUND ${OpenMP_CXX_FOUND})
endif()

if(OPENMP_FOUND)
    message(STATUS "OpenMP will be used for parallel execution")
else()
    message(STATUS "OpenMP not available - will compile with single-threaded fallback")
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
  message(STATUS "1. Homebrew (easiest):")
  message(STATUS "   brew install embree")
  message(STATUS "")
  message(STATUS "2. Specify embree_DIR manually:")
  message(STATUS "   cmake .. -Dembree_DIR=/path/to/embree/lib/cmake/embree-X.X.X")
  message(STATUS "")
  message(STATUS "3. Download from:")
  message(STATUS "   https://github.com/RenderKit/embree/releases")
  message(STATUS "")
  message(FATAL_ERROR "Cannot continue without Embree")
endif()

# Create executable
add_executable(main_trace main_trace.cpp)

# Include directories (needed regardless of OpenMP)
target_include_directories(main_trace PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}          # apps/rt/embree
    ${CMAKE_CURRENT_SOURCE_DIR}/..       # apps/rt
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/CPP  # runtime/CPP
)
target_compile_options(main_trace PRIVATE -O3 -march=native)

# Link Embree library
target_link_libraries(main_trace PRIVATE embree)

# Link OpenMP if available
if(OPENMP_FOUND)
    if(APPLE)
        target_compile_options(main_trace PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(main_trace PRIVATE ${OpenMP_omp_LIBRARY})
        target_include_directories(main_trace PRIVATE ${LIBOMP_PREFIX}/include)
    else()
        target_link_libraries(main_trace PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_definitions(main_trace PRIVATE HAVE_OPENMP)
endif()

# On conda environments, link against conda's libstdc++ to avoid version mismatch
if(DEFINED ENV{CONDA_PREFIX})
    target_link_options(main_trace PRIVATE "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
endif()

# Print information
message(STATUS "Embree version: ${embree_VERSION}")
message(STATUS "Embree includes: ${EMBREE_INCLUDE_DIRS}")
message(STATUS "Embree libraries: ${EMBREE_LIBRARIES}")