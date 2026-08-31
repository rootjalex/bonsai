##
# CTest implementation

# FLAGS are passed to the compiler for every test in the call, after the
# flags the test file itself declares. Use them for settings a whole directory
# shares -- notably pinning the LLVM target for tests that diff generated
# code, so their output does not depend on the machine running them.

# A handful of the backends goldens record what LLVM's optimizer produced, and
# that legitimately differs between target architectures: LLVM 19 SLP
# vectorizes a pair of i32 operations on AArch64 but not on x86-64, and
# SimplifyCFG names the same select %spec.select on one and %common.ret.op on
# the other. Where that happens, an optional <name>.<arch>.expect overrides the
# base golden on that architecture, and the base file stays whatever the
# machine CI runs on blesses.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" BONSAI_TEST_ARCH)

function(add_bonsai_tests)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "" "FILES;LABELS;FLAGS")
    foreach (file IN LISTS ARG_FILES)
        cmake_path(SET input "${file}")
        cmake_path(REPLACE_EXTENSION input "expect" OUTPUT_VARIABLE expect)
        cmake_path(REPLACE_EXTENSION input "actual" OUTPUT_VARIABLE actual)
        cmake_path(REPLACE_EXTENSION input "${BONSAI_TEST_ARCH}.expect"
                   OUTPUT_VARIABLE expect_arch)

        cmake_path(ABSOLUTE_PATH input)
        cmake_path(ABSOLUTE_PATH expect)
        cmake_path(ABSOLUTE_PATH expect_arch)
        cmake_path(ABSOLUTE_PATH actual BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

        cmake_path(REMOVE_FILENAME actual OUTPUT_VARIABLE actual_dir)
        file(MAKE_DIRECTORY "${actual_dir}")

        cmake_path(RELATIVE_PATH input
                   BASE_DIRECTORY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}"
                   OUTPUT_VARIABLE test_name)

        set(run_test "${test_name} :: execute")
        set(check_test "${test_name} :: validate")

        cmake_path(RELATIVE_PATH input
                   BASE_DIRECTORY "${PROJECT_SOURCE_DIR}"
                   OUTPUT_VARIABLE input_file_arg)

        add_test(NAME "${run_test}"
                 COMMAND bonsai_test_runner "${input_file_arg}" "${actual}"
                         ${ARG_FLAGS}
                 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        set_tests_properties(
            "${run_test}" PROPERTIES FIXTURES_SETUP "${actual}"
        )

        add_test(
            NAME "${check_test}"
            COMMAND
            "${CMAKE_COMMAND}" "-DACTUAL=${actual}" "-DEXPECT=${expect}"
            "-DEXPECT_ARCH=${expect_arch}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/check_output.cmake"
        )
        set_tests_properties(
            "${check_test}" PROPERTIES FIXTURES_REQUIRED "${actual}"
        )

        set_tests_properties(
            "${run_test}" "${check_test}" PROPERTIES LABELS "${ARG_LABELS}"
        )
    endforeach ()
endfunction()

##
# Add tests directories

add_subdirectory(backends)
add_subdirectory(correctness)
add_subdirectory(error)
add_subdirectory(lower)
add_subdirectory(opt)
add_subdirectory(parsing)
add_subdirectory(ssa)
