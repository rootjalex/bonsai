##
# CTest implementation

function(add_bonsai_tests)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "" "FILES;LABELS")
    foreach (file IN LISTS ARG_FILES)
        cmake_path(SET input "${file}")
        cmake_path(REPLACE_EXTENSION input "expect" OUTPUT_VARIABLE expect)
        cmake_path(REPLACE_EXTENSION input "actual" OUTPUT_VARIABLE actual)

        cmake_path(ABSOLUTE_PATH input)
        cmake_path(ABSOLUTE_PATH expect)
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
                 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        set_tests_properties(
            "${run_test}" PROPERTIES FIXTURES_SETUP "${actual}"
        )

        add_test(
            NAME "${check_test}"
            COMMAND
            "${CMAKE_COMMAND}" "-DACTUAL=${actual}" "-DEXPECT=${expect}"
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
