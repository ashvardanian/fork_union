# Shared logic for setting target properties for tests and examples
function (set_target_properties_for_fork_union_script target_name)
    target_link_libraries(${TGT} PRIVATE fork_union)

    # In release mode, enable optimizations
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Enabling optimizations for ${TGT}")
        target_compile_options(${TGT} PRIVATE -O3)
    endif ()

    # if we’re on GCC and not in Release, turn on TSAN here too
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Enabling ThreadSanitizer for ${TGT}")
        target_compile_options(${TGT} PRIVATE -fsanitize=thread -g -O1)
        target_link_options(${TGT} PRIVATE -fsanitize=thread)
    endif ()
endfunction ()

# Test targets for C++11, C++14, C++17, and C++20
set(TEST_SOURCES test.cpp)
set(CXX_STANDARDS 11 14 17 20)
foreach (STD IN LISTS CXX_STANDARDS)
    # Derive a unique target name
    set(TGT fork_union_test_cpp${STD})

    # Create the executable
    add_executable(${TGT} ${TEST_SOURCES})
    set_target_properties(
        ${target_name}
        PROPERTIES CXX_STANDARD ${STD}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
    )

    # register it as a CTest test
    add_test(NAME ${TGT} COMMAND ${TGT})
    set_target_properties_for_fork_union_script(${TGT})

endforeach ()

# Include the N-body benchmark
add_executable(fork_union_nbody nbody.cpp)
target_link_libraries(fork_union_nbody PRIVATE fork_union)
set_target_properties(
    fork_union_nbody
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Enabling OpenMP for fork_union_nbody via imported target")
    target_compile_options(fork_union_nbody PRIVATE -fopenmp)
    target_link_options(fork_union_nbody PRIVATE -fopenmp)
endif ()
set_target_properties_for_fork_union_script(fork_union_nbody)
