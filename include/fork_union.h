/**
 *  @brief  OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   fork_union.h
 *  @author Ash Vardanian
 *  @date   June 17, 2025
 *
 *  Fork Union provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
 *  avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
 *  The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
 *  execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
 *  to use even with the maximal `size_t` values. It's compatible with C 99 and later.
 *
 *  @code{.c}
 *  #include <stdio.h> // `printf`
 *  #include <stdlib.h> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.h> // `fu_pool_t`
 *
 *  struct print_args_context_t {
 *      size_t argc; // ? Number of arguments
 *      char **argv; // ? Array of arguments
 *  };
 *
 *  void print_arg(void *context_punned, size_t task_index, size_t thread_index, size_t colocation_index) {
 *      print_args_context_t *context = (print_args_context_t *)context_punned;
 *      printf(
 *          "Printing argument # %zu from thread # %zu at colocation # %zu: %s\n",
 *          task_index, context->argc, thread_index, colocation_index, context->argv[task_index]);
 *  }
 *
 *  int main(int argc, char *argv[]) {
 *      char const *caps = fu_capabilities_string();
 *      if (!caps) return EXIT_FAILURE; // ! Thread pool is not supported
 *      printf("Fork Union capabilities: %s\n", caps);
 *
 *      fu_pool_t *pool = fu_pool_new();
 *      if (!pool) return EXIT_FAILURE; // ! Failed to create a thread pool
 *
 *      size_t threads = fu_count_logical_cores();
 *      if (!fu_pool_spawn(pool, threads, fu_caller_inclusive_k)) return EXIT_FAILURE; // ! Can't spawn
 *
 *      print_args_context_t context = {argc, argv};
 *      fu_pool_for_n(pool, argc, &print_arg, &context);
 *      fu_pool_delete(pool);
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  Unlike the C++ version, the C header wraps the best-fit pre-compiled platform-specific instantiation
 *  of C++ templates. It also uses a singleton state to store the NUMA topology and other OS/machine specs.
 *  Under the hood, the `fu_pool_t` maps to a `basic_pool` or `linux_distributed_pool`.
 *  For advanced usage, prefer the core C++ library.
 */
#pragma once
#include <stddef.h> // `size_t`, `bool`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region - Types

typedef int fu_bool_t;             // ? A simple boolean type, 0 for false, 1 for true
typedef void *fu_pool_t;           // ? A simple cross-platform opaque wrapper
typedef void *fu_lambda_context_t; // ? Type-punned pointer to the user-defined context

typedef void (*fu_for_threads_t)(fu_lambda_context_t, size_t thread, size_t colocation);
typedef void (*fu_for_prongs_t)(fu_lambda_context_t, size_t task, size_t thread, size_t colocation);
typedef void (*fu_for_slices_t)(fu_lambda_context_t, size_t first, size_t count, size_t thread, size_t colocation);

typedef enum fu_caller_exclusivity_t {
    fu_caller_inclusive_k,
    fu_caller_exclusive_k,
} fu_caller_exclusivity_t;

#pragma endregion - Types

#pragma region - Metadata

/**
 *  @brief Describes available OS+CPU capabilities used by the thread pools.
 *
 *  @retval `nullptr`, if the thread pool is not supported on the current platform.
 *  @retval "serial" for the default C++ STL-powered thread pool.
 *  @retval "numa" for the NUMA-aware thread pool on Linux-based systems.
 *  @retval "numa+x86tpause" for the NUMA-aware pool with `tpause` instruction with "waitpkg" CPU feature.
 *  @retval "numa+aarch64wfet" for ...
 *  @retval "numa+x86pause" for ...
 *  @retval "numa+risc5pause" for ...
 *  @retval "numa+aarch64yield" for ...
 */
char const *fu_capabilities_string();

/**
 *  @brief Describes the number of physical CPU cores available on the system.
 *  @retval 0 if the thread pool is not supported on the current platform.
 *
 *  On x86, if hyper-threading is enabled, will be 2x of the number of physical cores.
 */
size_t fu_count_logical_cores();

/**
 *  @brief Describes the maximum number of individually addressable thread groups.
 *  @retval 0 if the thread pool is not supported on the current platform.
 *  @retval 1 on most desktop or IoT platforms.
 *  @retval 4-32 is a typical range on high-end cloud servers.
 *
 *  May be as big as the product of the number of NUMA nodes and QoS levels.
 *  @sa `fu_count_numa_nodes`, `fu_count_quality_levels`, `fu_count_huge_pages`.
 */
size_t fu_count_colocations();
size_t fu_count_numa_nodes();
size_t fu_count_quality_levels();
size_t fu_count_huge_pages();

#pragma endregion - Metadata

#pragma region - Memory

/**
 *  @brief Allocates memory on a specific NUMA node.
 *  @param numa_node_index The index of the NUMA node to allocate memory on.
 *  @param minimum_bytes Minimum number of bytes to allocate.
 *  @param allocated_pointer Pointer to store the address of the allocated memory.
 *  @param allocated_bytes Pointer to store the actual number of bytes allocated.
 *  @retval 1 if the allocation was successful.
 *  @retval 0 if the allocation failed.
 */
fu_bool_t fu_allocate_at_least(                   //
    size_t numa_node_index, size_t minimum_bytes, //
    void **allocated_pointer, size_t *allocated_bytes);

/**
 *  @brief Releases memory allocated on a specific NUMA node.
 *  @param numa_node_index The index of the NUMA node where the memory was allocated.
 *  @param pointer Pointer to the memory to be released.
 *  @param bytes Number of bytes to release.
 */
void fu_free(size_t numa_node_index, void *pointer, size_t bytes);

#pragma endregion - Memory

#pragma region - Lifetime

fu_pool_t *fu_pool_new();
void fu_pool_delete(fu_pool_t *pool);
fu_bool_t fu_pool_spawn(fu_pool_t *pool, size_t threads, fu_caller_exclusivity_t exclusivity);
void fu_pool_sleep(fu_pool_t *pool, size_t micros);
void fu_pool_terminate(fu_pool_t *pool);

size_t fu_pool_count_colocations(fu_pool_t *pool);
size_t fu_pool_count_threads(fu_pool_t *pool);
size_t fu_pool_count_threads_in_colocation(fu_pool_t *pool, size_t colocation_index);

#pragma endregion - Lifetime

#pragma region - Primary API

void fu_pool_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context);
void fu_pool_for_n(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);
void fu_pool_for_n_dynamic(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);
void fu_pool_for_slices(fu_pool_t *pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context);

#pragma endregion - Primary API

#pragma region - Flexible API

void fu_pool_unsafe_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context);
void fu_pool_unsafe_for_n(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);
void fu_pool_unsafe_for_n_dynamic(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context);
void fu_pool_unsafe_for_slices(fu_pool_t *pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context);
void fu_pool_unsafe_join(fu_pool_t *pool);

#pragma endregion - Flexible API

#ifdef __cplusplus
} // extern "C"
#endif