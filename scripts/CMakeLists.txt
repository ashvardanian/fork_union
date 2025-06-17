# Shared logic for setting target properties for tests and examples
function (set_target_properties_for_fork_union_script target_name)
    target_link_libraries(${TGT} PRIVATE fork_union)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # This warning annoys us by reminding, that before GCC 4.6 the ABI for passing objects aligned to ≥32-bytes was
        # different... but that GCC version was released in 2014, over a decade ago.
        target_compile_options(${TGT} PRIVATE -Wno-psabi)
    endif ()

    # In release mode, enable optimizations
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Enabling optimizations for ${TGT}")
        target_compile_options(${TGT} PRIVATE -O3)
    endif ()

    # if we're not in Release mode
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        # if we're on GCC/Clang-compatible compiler and not in Release, turn on TSAN here too
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
            OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
            OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
        )
            message(STATUS "Enabling ThreadSanitizer for ${TGT}")
            target_compile_options(${TGT} PRIVATE -g -O1)
            target_compile_options(${TGT} PRIVATE -fsanitize=undefined,thread -fno-omit-frame-pointer)
            target_link_options(${TGT} PRIVATE -fsanitize=undefined,thread)
        endif ()
    endif ()
endfunction ()

# Test targets for C++14, C++17, and C++20
set(TEST_SOURCES test.cpp)
set(CXX_STANDARDS 14 17 20)
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
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Enabling OpenMP for fork_union_nbody on GCC")
    target_compile_options(fork_union_nbody PRIVATE -fopenmp)
    target_link_options(fork_union_nbody PRIVATE -fopenmp)
    target_compile_options(fork_union_nbody PRIVATE -flto)
    target_compile_options(fork_union_nbody PRIVATE -ffast-math)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Enabling OpenMP for fork_union_nbody on Clang")
    target_compile_options(fork_union_nbody PRIVATE -Xpreprocessor -fopenmp)
    target_link_options(fork_union_nbody PRIVATE -lomp)
endif ()
set_target_properties_for_fork_union_script(fork_union_nbody)
