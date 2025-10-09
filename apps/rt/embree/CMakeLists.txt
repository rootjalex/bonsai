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

# Function to install Embree automatically
function(install_embree)
    message(STATUS "")
    message(STATUS "============================================")
    message(STATUS "Embree 4 not found - attempting installation")
    message(STATUS "============================================")
    
    if(APPLE)
        # Check if Homebrew is available
        execute_process(COMMAND which brew
                       OUTPUT_VARIABLE BREW_EXECUTABLE
                       OUTPUT_STRIP_TRAILING_WHITESPACE
                       ERROR_QUIET)
        
        if(BREW_EXECUTABLE)
            message(STATUS "Installing Embree via Homebrew...")
            execute_process(COMMAND brew install embree
                           RESULT_VARIABLE BREW_RESULT
                           OUTPUT_VARIABLE BREW_OUTPUT
                           ERROR_VARIABLE BREW_ERROR)
            
            if(BREW_RESULT EQUAL 0)
                message(STATUS "✓ Embree installed successfully via Homebrew")
                # Embree should now be in standard location
                execute_process(COMMAND brew --prefix embree
                               OUTPUT_VARIABLE EMBREE_PREFIX
                               OUTPUT_STRIP_TRAILING_WHITESPACE)
                set(embree_DIR "${EMBREE_PREFIX}/lib/cmake/embree-4.3.3" PARENT_SCOPE)
            else()
                message(FATAL_ERROR "Failed to install Embree via Homebrew: ${BREW_ERROR}")
            endif()
        else()
            message(FATAL_ERROR "Homebrew not found. Please install Homebrew first:\n/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        endif()
        
    elseif(UNIX)
        # Linux - try different package managers
        message(STATUS "Attempting to install Embree on Linux...")
        
        # Try apt-get (Debian/Ubuntu)
        execute_process(COMMAND which apt-get
                       OUTPUT_VARIABLE APT_EXECUTABLE
                       OUTPUT_STRIP_TRAILING_WHITESPACE
                       ERROR_QUIET)
        
        if(APT_EXECUTABLE)
            message(STATUS "Using apt-get to install Embree...")
            execute_process(COMMAND sudo apt-get update
                           RESULT_VARIABLE APT_UPDATE_RESULT)
            execute_process(COMMAND sudo apt-get install -y libembree-dev libembree4
                           RESULT_VARIABLE APT_RESULT
                           OUTPUT_VARIABLE APT_OUTPUT
                           ERROR_VARIABLE APT_ERROR)
            
            if(APT_RESULT EQUAL 0)
                message(STATUS "✓ Embree installed successfully via apt-get")
            else()
                message(WARNING "apt-get installation failed, trying manual download...")
                install_embree_from_source()
            endif()
        else()
            # Try yum (CentOS/RHEL/Fedora)
            execute_process(COMMAND which yum
                           OUTPUT_VARIABLE YUM_EXECUTABLE
                           OUTPUT_STRIP_TRAILING_WHITESPACE
                           ERROR_QUIET)
            
            if(YUM_EXECUTABLE)
                message(STATUS "Using yum to install Embree...")
                execute_process(COMMAND sudo yum install -y embree-devel
                               RESULT_VARIABLE YUM_RESULT
                               OUTPUT_VARIABLE YUM_OUTPUT
                               ERROR_VARIABLE YUM_ERROR)
                
                if(YUM_RESULT EQUAL 0)
                    message(STATUS "✓ Embree installed successfully via yum")
                else()
                    message(WARNING "yum installation failed, trying manual download...")
                    install_embree_from_source()
                endif()
            else()
                # No package manager found, download from GitHub
                install_embree_from_source()
            endif()
        endif()
    else()
        message(FATAL_ERROR "Unsupported platform for automatic Embree installation")
    endif()
endfunction()

# Function to download and install Embree from GitHub releases
function(install_embree_from_source)
    message(STATUS "Downloading Embree from GitHub releases...")
    
    set(EMBREE_VERSION "4.3.3")
    set(INSTALL_DIR "${CMAKE_BINARY_DIR}/embree-install")
    
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            set(EMBREE_ARCHIVE "embree-${EMBREE_VERSION}.arm64.macosx.tar.gz")
        else()
            set(EMBREE_ARCHIVE "embree-${EMBREE_VERSION}.x86_64.macosx.tar.gz")
        endif()
    elseif(UNIX)
        set(EMBREE_ARCHIVE "embree-${EMBREE_VERSION}.x86_64.linux.tar.gz")
    endif()
    
    set(EMBREE_URL "https://github.com/RenderKit/embree/releases/download/v${EMBREE_VERSION}/${EMBREE_ARCHIVE}")
    set(DOWNLOAD_PATH "${CMAKE_BINARY_DIR}/${EMBREE_ARCHIVE}")
    
    message(STATUS "Downloading from: ${EMBREE_URL}")
    
    file(DOWNLOAD "${EMBREE_URL}" "${DOWNLOAD_PATH}"
         STATUS DOWNLOAD_STATUS
         SHOW_PROGRESS)
    
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_ERROR)
    if(NOT DOWNLOAD_ERROR EQUAL 0)
        list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MSG)
        message(FATAL_ERROR "Failed to download Embree: ${DOWNLOAD_ERROR_MSG}")
    endif()
    
    message(STATUS "Extracting Embree...")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_PATH}"
                   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                   RESULT_VARIABLE EXTRACT_RESULT)
    
    if(NOT EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract Embree archive")
    endif()
    
    # Find the extracted directory
    file(GLOB EMBREE_EXTRACTED_DIR "${CMAKE_BINARY_DIR}/embree-${EMBREE_VERSION}.*")
    list(GET EMBREE_EXTRACTED_DIR 0 EMBREE_DIR_PATH)
    
    message(STATUS "✓ Embree downloaded and extracted to: ${EMBREE_DIR_PATH}")
    
    # Set embree_DIR for find_package
    set(embree_DIR "${EMBREE_DIR_PATH}/lib/cmake/embree-${EMBREE_VERSION}" PARENT_SCOPE)
endfunction()

# Find Embree package
find_package(embree 4.0 QUIET)

if(NOT embree_FOUND)
    # Attempt automatic installation
    install_embree()
    
    # Try to find it again after installation
    find_package(embree 4.0 QUIET)
    
    if(NOT embree_FOUND)
        message(FATAL_ERROR "Embree installation failed. Please install manually.")
    endif()
endif()

message(STATUS "✓ Embree found")

# Create executable
add_executable(main_trace main_trace.cpp)

# Link Embree library
target_link_libraries(main_trace PRIVATE embree)

# Link OpenMP if available
if(OPENMP_FOUND)
    if(APPLE)
        target_compile_options(main_trace PRIVATE ${OpenMP_CXX_FLAGS} -O3 -march=native)
        target_link_libraries(main_trace PRIVATE ${OpenMP_omp_LIBRARY})
        target_include_directories(main_trace PRIVATE
            ${LIBOMP_PREFIX}/include
            ${CMAKE_CURRENT_SOURCE_DIR}          # apps/rt/embree
            ${CMAKE_CURRENT_SOURCE_DIR}/..       # apps/rt
            ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/CPP  # runtime/CPP
        )
    else()
        target_link_libraries(main_trace PRIVATE OpenMP::OpenMP_CXX)
        target_include_directories(main_trace PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}          # apps/rt/embree
            ${CMAKE_CURRENT_SOURCE_DIR}/..       # apps/rt
            ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/CPP  # runtime/CPP
        )
        target_compile_options(main_trace PRIVATE -O3 -march=native)
    endif()
    target_compile_definitions(main_trace PRIVATE HAVE_OPENMP)
endif()

# Print information
message(STATUS "Embree version: ${embree_VERSION}")
message(STATUS "Embree includes: ${EMBREE_INCLUDE_DIRS}")
message(STATUS "Embree libraries: ${EMBREE_LIBRARIES}")