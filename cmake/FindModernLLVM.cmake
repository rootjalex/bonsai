# This file wraps the upstream package config modules for LLVM to fix
# pathological issues in its implementation.

set(REASON_FAILURE_MESSAGE "")

# Fallback configurations for weirdly built LLVMs
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL MinSizeRel Release RelWithDebInfo "")
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo Release MinSizeRel "")
set(CMAKE_MAP_IMPORTED_CONFIG_RELEASE Release MinSizeRel RelWithDebInfo "")

if (ModernLLVM_FIND_VERSION_EXACT)
    set(_EXACT EXACT)
else ()
    set(_EXACT "")
endif ()
find_package(
    LLVM ${ModernLLVM_FIND_VERSION} ${_EXACT}
    PATHS
    # macOS paths
    "/opt/homebrew/opt/llvm@${ModernLLVM_FIND_VERSION_MAJOR}"
    "/opt/homebrew/opt/llvm"
    # Linux paths
    "/usr/lib/llvm-${ModernLLVM_FIND_VERSION_MAJOR}"
)

if (NOT DEFINED ModernLLVM_SHARED_LIBS)
    if (LLVM_FOUND AND LLVM_LINK_LLVM_DYLIB)
        set(ModernLLVM_SHARED_LIBS YES)
    else ()
        set(ModernLLVM_SHARED_LIBS NO)
    endif ()
endif ()

option(ModernLLVM_SHARED_LIBS "Enable to link to shared libLLVM" "${ModernLLVM_SHARED_LIBS}")

set(ModernLLVM_LIBS "")
if (LLVM_FOUND)
    foreach (comp IN LISTS ModernLLVM_FIND_COMPONENTS)
        llvm_map_components_to_libnames(libs ${comp})
        list(APPEND ModernLLVM_LIBS ${libs})

        set(ModernLLVM_${comp}_FOUND 1)
        foreach (lib IN LISTS libs)
            if (NOT TARGET ${lib})
                set(ModernLLVM_${comp}_FOUND 0)
            endif ()
        endforeach ()
    endforeach ()

    set(ModernLLVM_SHARED_LIBRARY "LLVM")
    if (ModernLLVM_SHARED_LIBS)
        if (TARGET "${ModernLLVM_SHARED_LIBRARY}")
            set(ModernLLVM_LIBS LLVM ${CMAKE_DL_LIBS})
        else ()
            string(APPEND ModernLLVM_SHARED_LIBRARY "-NOTFOUND")
            string(APPEND REASON_FAILURE_MESSAGE
                   "ModernLLVM_SHARED_LIBS=${ModernLLVM_SHARED_LIBS} but the shared LLVM target does not exist.\n")
        endif ()
    endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ModernLLVM
    REQUIRED_VARS LLVM_CONFIG ModernLLVM_SHARED_LIBRARY
    VERSION_VAR LLVM_PACKAGE_VERSION
    REASON_FAILURE_MESSAGE "${REASON_FAILURE_MESSAGE}"
    HANDLE_COMPONENTS
    HANDLE_VERSION_RANGE
)

if (LLVM_FOUND AND NOT TARGET ModernLLVM::LLVM)
    add_library(ModernLLVM::LLVM INTERFACE IMPORTED)

    # Definitions
    separate_arguments(LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
    list(REMOVE_ITEM LLVM_DEFINITIONS "-D_GLIBCXX_ASSERTIONS") # work around https://reviews.llvm.org/D142279
    target_compile_definitions(ModernLLVM::LLVM INTERFACE ${LLVM_DEFINITIONS})

    # Include paths
    target_include_directories(ModernLLVM::LLVM INTERFACE "${LLVM_INCLUDE_DIRS}")

    # Component libraries
    if (ModernLLVM_SHARED_LIBS)
        target_link_libraries(ModernLLVM::LLVM INTERFACE LLVM ${CMAKE_DL_LIBS})
    else ()
        target_link_libraries(ModernLLVM::LLVM INTERFACE ${ModernLLVM_LIBS})
    endif ()
endif ()
