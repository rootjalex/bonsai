##
# Test runner executable
add_executable(bonsai_test_runner runner.cpp)
add_executable(bonsai::test_runner ALIAS bonsai_test_runner)
target_link_libraries(bonsai_test_runner PRIVATE bonsai::library)
target_compile_features(bonsai_test_runner PRIVATE cxx_std_20)

##
# Add tests subdirectory
add_subdirectory(bonsai)
