cmake_minimum_required(VERSION 3.30)
project(bonsai VERSION 0.1.0)

include(CMakeDependentOption)

if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif ()

##
# Custom CMake modules

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

##
# Project options

cmake_dependent_option(
    bonsai_BUILD_TESTING
    "Compile Bonsai with common compiler warnings"
    "${PROJECT_IS_TOP_LEVEL}"
    BUILD_TESTING
    OFF
)

option(
    bonsai_ENABLE_WARNINGS
    "Compile Bonsai with common compiler warnings"
    "${PROJECT_IS_TOP_LEVEL}"
)

option(
    bonsai_SKIP_INSTALL_RULES
    "Exclude bonsai's install rules from the build"
    OFF
)

##
# Process options
if (bonsai_SKIP_INSTALL_RULES)
    set(CMAKE_SKIP_INSTALL_RULES ON)
endif ()

##
# Dependencies

find_package(
    ModernLLVM 19.1.6
    REQUIRED
    COMPONENTS
    core
    irreader
    support
    analysis
    target
    passes
    native
    # irbuilder
    orcjit
    executionengine # Required for LLJIT
    runtimeDyld # Required for JIT execution
)

##
# Bonsai compiler library

add_library(bonsai_library STATIC)
add_library(bonsai::library ALIAS bonsai_library)

target_compile_features(bonsai_library PUBLIC cxx_std_20)
target_link_libraries(bonsai_library PUBLIC ModernLLVM::LLVM)

target_sources(
    bonsai_library
    PRIVATE
    src/CLI/CLI.cpp
    src/CodeGen/ASM.cpp
    src/CodeGen/CodeGen_CUDA.cpp
    src/CodeGen/CodeGen_LLVM.cpp
    src/CodeGen/CPP.cpp
    src/CodeGen/JIT.cpp
    src/IR/Analysis.cpp
    src/IR/Equality.cpp
    src/IR/Expr.cpp
    src/IR/Float16.cpp
    src/IR/Interface.cpp
    src/IR/Layout.cpp
    src/IR/Mutator.cpp
    src/IR/Operators.cpp
    src/IR/Printer.cpp
    src/IR/Stmt.cpp
    src/IR/Type.cpp
    src/IR/TypeEnforcement.cpp
    src/IR/ValidateLayout.cpp
    src/IR/Visitor.cpp
    src/IR/WriteLoc.cpp
    src/Lower/Canonicalize.cpp
    src/Lower/CompilerOptions.cpp
    src/Lower/Defers.cpp
    src/Lower/DynamicArrays.cpp
    src/Lower/DynamicSets.cpp
    src/Lower/Externs.cpp
    src/Lower/ForEachs.cpp
    src/Lower/Generics.cpp
    src/Lower/Geometrics.cpp
    src/Lower/Intrinsics.cpp
    src/Lower/Lambdas.cpp
    src/Lower/Layouts.cpp
    src/Lower/LogicalOperations.cpp
    src/Lower/LoopTransforms.cpp
    src/Lower/Lower.cpp
    src/Lower/Maps.cpp
    src/Lower/Mutability.cpp
    src/Lower/Options.cpp
    src/Lower/Pass.cpp
    src/Lower/PassManager.cpp
    src/Lower/PredicateAnalysis.cpp
    src/Lower/Random.cpp
    src/Lower/RecLoops.cpp
    src/Lower/RenamePointerToExpr.cpp
    src/Lower/ReturnToOutParameter.cpp
    src/Lower/Sorts.cpp
    src/Lower/Trees.cpp
    src/Lower/Tuples.cpp
    src/Lower/TopologicalOrder.cpp
    src/Lower/TypeInference.cpp
    src/Lower/VerifyLayouts.cpp
    src/Lower/VerifyOptions.cpp
    src/Lower/Yields.cpp
    src/Opt/DCE.cpp
    src/Opt/Fusion.cpp
    src/Opt/Inline.cpp
    src/Opt/Parallelize.cpp
    src/Opt/Simplify.cpp
    src/Opt/Unswitch.cpp
    src/Parser/Lexer.cpp
    src/Parser/Parser.cpp
    src/Parser/Token.cpp
    src/Utils.cpp
)

target_sources(
    bonsai_library
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
    include/Bonsai.h
    include/CLI/CLI.h
    include/CodeGen/ASM.h
    include/CodeGen/CodeGen_CUDA.h
    include/CodeGen/CodeGen_LLVM.h
    include/CodeGen/CPP.h
    include/CodeGen/JIT.h
    include/CompilerOptions.h
    include/Error.h
    include/IR/Analysis.h
    include/IR/Equality.h
    include/IR/Expr.h
    include/IR/Float16.h
    include/IR/Frame.h
    include/IR/Function.h
    include/IR/IRFwdDecl.h
    include/IR/IRHandle.h
    include/IR/IRNode.h
    include/IR/Interface.h
    include/IR/Layout.h
    include/IR/Mutator.h
    include/IR/Operators.h
    include/IR/Printer.h
    include/IR/Program.h
    include/IR/Schedule.h
    include/IR/Stmt.h
    include/IR/Target.h
    include/IR/Type.h
    include/IR/TypeEnforcement.h
    include/IR/ValidateLayout.h
    include/IR/Visitor.h
    include/IR/WriteLoc.h
    include/IntrusivePtr.h
    include/LLVMIncl.h
    include/Lower/Canonicalize.h
    include/Lower/Defers.h
    include/Lower/DynamicArrays.h
    include/Lower/DynamicSets.h
    include/Lower/Externs.h
    include/Lower/ForEachs.h
    include/Lower/Generics.h
    include/Lower/Geometrics.h
    include/Lower/Intrinsics.h
    include/Lower/Lambdas.h
    include/Lower/Layouts.h
    include/Lower/LogicalOperations.h
    include/Lower/LoopTransforms.h
    include/Lower/Lower.h
    include/Lower/Maps.h
    include/Lower/Mutability.h
    include/Lower/Options.h
    include/Lower/Pass.h
    include/Lower/PassManager.h
    include/Lower/PredicateAnalysis.h
    include/Lower/Random.h
    include/Lower/RecLoops.h
    include/Lower/RenamePointerToExpr.h
    include/Lower/ReturnToOutParameter.h
    include/Lower/Sorts.h
    include/Lower/Trees.h
    include/Lower/Tuples.h
    include/Lower/TopologicalOrder.h
    include/Lower/TypeInference.h
    include/Lower/VerifyLayouts.h
    include/Lower/VerifyOptions.h
    include/Lower/Yields.h
    include/Opt/DCE.h
    include/Opt/Fusion.h
    include/Opt/Inline.h
    include/Opt/Parallelize.h
    include/Opt/Simplify.h
    include/Opt/Unswitch.h
    include/Parser/Lexer.h
    include/Parser/Parser.h
    include/Parser/Token.h
    include/RefCount.h
    include/Scope.h
    include/Utils.h
)

if (bonsai_ENABLE_WARNINGS)
    target_compile_options(
        bonsai_library
        PUBLIC
        "$<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-Wall>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W3>"
    )
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fno-omit-frame-pointer")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=undefined -fno-omit-frame-pointer")

##
# Bonsai compiler

add_executable(bonsai_compiler compiler.cpp)
add_executable(bonsai::compiler ALIAS bonsai_compiler)
set_target_properties(
    bonsai_compiler
    PROPERTIES
    EXPORT_NAME compiler
    OUTPUT_NAME compiler
)

target_link_libraries(bonsai_compiler PRIVATE bonsai::library)

##
# Testing

if (bonsai_BUILD_TESTING)
    add_subdirectory(tests)
endif ()

##
# Installation rules

if (NOT CMAKE_SKIP_INSTALL_RULES)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(bonsai_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/cmake/bonsai"
        CACHE STRING "Install destination for bonsai CMake package files")

    install(
        TARGETS bonsai_compiler
        EXPORT bonsai-targets
        COMPONENT bonsai-runtime
    )

    install(
        EXPORT bonsai-targets
        NAMESPACE bonsai::
        DESTINATION "${bonsai_INSTALL_CMAKEDIR}"
        COMPONENT bonsai-development
    )

    configure_package_config_file(
        cmake/bonsai-config.cmake.in
        cmake/bonsai-config.cmake
        INSTALL_DESTINATION "${bonsai_INSTALL_CMAKEDIR}"
    )

    write_basic_package_version_file(
        cmake/bonsai-config-version.cmake
        COMPATIBILITY SameMajorVersion
    )

    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cmake/"
        DESTINATION "${bonsai_INSTALL_CMAKEDIR}"
        COMPONENT bonsai-development
    )
endif ()
