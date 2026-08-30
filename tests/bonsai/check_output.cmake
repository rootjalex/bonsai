cmake_minimum_required(VERSION 3.30)

get_property(role GLOBAL PROPERTY CMAKE_ROLE)
if (NOT role STREQUAL "SCRIPT")
    message(FATAL_ERROR "must be run in script mode (-P)")
endif ()

if (NOT EXISTS "${ACTUAL}")
    message(FATAL_ERROR "Missing actual file: ${ACTUAL}")
endif ()

# An <name>.<arch>.expect next to the base golden overrides it on that
# architecture; see the comment in CMakeLists.txt. Swapping EXPECT here means
# BONSAI_UPDATE_EXPECT re-blesses the override where one exists and the base
# golden where one does not, so updating on a host that has an override can
# never silently clobber the golden the other architecture relies on.
if (EXPECT_ARCH AND EXISTS "${EXPECT_ARCH}")
    set(EXPECT "${EXPECT_ARCH}")
endif ()

if ("$ENV{BONSAI_UPDATE_EXPECT}")
    file(COPY_FILE "${ACTUAL}" "${EXPECT}"
         RESULT error
         ONLY_IF_DIFFERENT
         INPUT_MAY_BE_RECENT)
    if (error)
        message(FATAL_ERROR "${error}")
    endif ()
else ()
    if (NOT EXISTS "${EXPECT}")
        message(FATAL_ERROR "Missing expect file: ${EXPECT}")
    endif ()

    execute_process(
        COMMAND diff "${ACTUAL}" "${EXPECT}"
        COMMAND_ERROR_IS_FATAL ANY
        COMMAND_ECHO STDOUT
    )
endif ()
