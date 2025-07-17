cmake_minimum_required(VERSION 3.14)

project(
    fork_union
    VERSION 2.1.2
    DESCRIPTION "Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library"
    LANGUAGES CXX
)

# Enforce C++17 as the minimum standard for the project.
set(PROJECT_CXX_STANDARD 17)
set(PROJECT_CXX_EXTENSIONS OFF)
set(PROJECT_CXX_STANDARD_REQUIRED ON)

# Header-only interface library
add_library(fork_union INTERFACE)
target_include_directories(
    fork_union INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)

# Strict compilation flags
target_compile_options(
    fork_union INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wcast-qual -Wcast-align -Wunused -Wno-unused-parameter -Wno-unknown-pragmas>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
)

# Pre-compiled libraries built from `c/lib.cpp`
add_library(fork_union_dynamic SHARED c/lib.cpp)
add_library(fork_union_static STATIC c/lib.cpp)

# Prefer C++20 for library builds
set_target_properties(
    fork_union_dynamic fork_union_static
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
)

# Re-use the public interface of the header-only target
target_link_libraries(fork_union_dynamic PUBLIC fork_union)
target_link_libraries(fork_union_static PUBLIC fork_union)

# Set the output directory for all executables - on Windows requires more boilerplate:
# https://stackoverflow.com/a/25328001
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})

# Static analysis tools
find_program(CPPCHECK_EXECUTABLE cppcheck)
if(CPPCHECK_EXECUTABLE)
    add_custom_target(
        cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            --enable=all
            --std=c++17
            --verbose
            --quiet
            --error-exitcode=1
            --suppress=missingIncludeSystem
            --suppress=unusedFunction
            --suppress=unmatchedSuppression
            --suppress=ConfigurationNotChecked
            --suppress=knownConditionTrueFalse
            --suppress=shadowFunction
            --suppress=shadowVariable
            --suppress=useStlAlgorithm
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -DFU_ENABLE_NUMA=1
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/c
        COMMENT "Running cppcheck static analysis"
    )
endif()

find_program(CLANG_TIDY_EXECUTABLE clang-tidy)
if(CLANG_TIDY_EXECUTABLE)
    add_custom_target(
        clang-tidy
        COMMAND ${CLANG_TIDY_EXECUTABLE} 
            --config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy
            --quiet
            ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/c/*.cpp
            --
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -std=c++17
        COMMENT "Running clang-tidy static analysis"
    )
endif()

# Tests & benchmarking scripts
include(CTest)
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(scripts)
endif ()

# Install rules
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    TARGETS fork_union
    EXPORT fork_unionTargets
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    TARGETS fork_union_dynamic fork_union_static
    EXPORT fork_unionTargets # same export set
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so / .dylib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # .a
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # .dll on Windows
)

# Export config files for find_package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fork_unionConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    cmake/fork_unionConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/fork_unionConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fork_union
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/fork_unionConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/fork_unionConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fork_union
)

install(
    EXPORT fork_unionTargets
    NAMESPACE fork_union::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fork_union
)

# On Linux, let's check if `libnuma` and `pthreads` are available.
if (UNIX AND NOT APPLE) # "True Linux" in CMake-speak

    # Find POSIX threads library
    find_package(Threads)

    # Find libNUMA
    find_library(NUMA_LIBRARY NAMES numa)
    set(NUMA_FOUND FALSE)
    if (NUMA_LIBRARY)
        set(NUMA_FOUND TRUE)
        message(STATUS "libNUMA found: ${NUMA_LIBRARY}")
    else ()
        message(STATUS "libNUMA not found – building without NUMA support")
    endif ()

    # Link to PThreads, libNUMA
    if (Threads_FOUND AND NUMA_FOUND)
        target_link_libraries(fork_union INTERFACE ${NUMA_LIBRARY} Threads::Threads)
        target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=1)
    else ()
        target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=0)
    endif ()

else () # Non-Linux platforms
    target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=0)
endif ()
