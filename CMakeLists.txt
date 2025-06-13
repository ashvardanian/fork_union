cmake_minimum_required(VERSION 3.14)

project(
    fork_union
    VERSION 1.0.4
    DESCRIPTION "OpenMP-style cross-platform fine-grained parallelism library"
    LANGUAGES CXX
)

# Enforce C++11
set(PROJECT_CXX_STANDARD 11)
set(PROJECT_CXX_EXTENSIONS OFF)
set(PROJECT_CXX_STANDARD_REQUIRED ON)

# Header-only interface library
add_library(fork_union INTERFACE)
target_include_directories(
    fork_union INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)

# Enable tests if requested
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
    find_package(Threads) # Creates Threads::Threads if found
    if (Threads_FOUND)
        target_link_libraries(fork_union INTERFACE Threads::Threads)
    endif ()

    find_library(NUMA_LIBRARY NAMES numa)
    set(NUMA_FOUND FALSE)
    if (NUMA_LIBRARY)
        set(NUMA_FOUND TRUE)
        message(STATUS "libnuma found: ${NUMA_LIBRARY}")
    else ()
        message(STATUS "libnuma not found – building without NUMA support")
    endif ()
    if (NUMA_FOUND)
        target_link_libraries(fork_union INTERFACE ${NUMA_LIBRARY})
        target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=1)
    else ()
        target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=0)
    endif ()
else () # Non-Linux platforms
    target_compile_definitions(fork_union INTERFACE FU_ENABLE_NUMA=0)
endif ()
