#pragma once
#include <stddef.h> // `size_t`, `bool`

#ifdef __cplusplus
extern "C" {
#endif

typedef void *fu_thread_pool_t; /// ? A simple cross-platform
typedef void *fu_numa_pools_t;
typedef void *fu_lambda_context_t;

typedef void (*fu_broadcast_lambda_t)(void *context, size_t thread, size_t node);
typedef void (*fu_for_n_lambda_t)(void *context, size_t task, size_t thread, size_t node);
typedef void (*fu_for_slices_lambda_t)(void *context, size_t begin, size_t count, size_t thread, size_t node);

/**
 *  @brief Describes available OS+CPU capabilities used by the thread pools.
 *
 *  If "numa" isn't part of the string, then the `fu_numa_pools_new` would fail.
 *
 *  @retval "serial" for the default C++ STL-powered thread pool.
 *  @retval "numa" for the NUMA-aware thread pool on Linux-based systems.
 *  @retval "numa+tpause" for the NUMA-aware pool with `tpause` instruction with "waitpkg" CPU feature.
 *  @retval "numa+wfet" for ...
 *  @retval "numa+x86pause" for ...
 *  @retval "numa+risc5pause" for ...
 *  @retval "numa+aarch64yield" for ...
 */
char *fu_capabilities_string();

fu_thread_pool_t *fu_thread_pool_new();
void fu_thread_pool_delete(fu_thread_pool_t *pool);
bool fu_thread_pool_try_spawn(fu_thread_pool_t *pool, size_t threads);
void fu_thread_pool_sleep(fu_thread_pool_t *pool, size_t micros);
void fu_thread_pool_terminate(fu_thread_pool_t *pool);

void fu_thread_pool_broadcast(fu_thread_pool_t *pool, fu_broadcast_lambda_t, fu_lambda_context_t);
void fu_thread_pool_for_n(fu_thread_pool_t *pool, size_t n, fu_for_n_lambda_t, fu_lambda_context_t);
void fu_thread_pool_for_n_dynamic(fu_thread_pool_t *pool, size_t n, fu_for_n_lambda_t, fu_lambda_context_t);
void fu_thread_pool_for_slices(fu_thread_pool_t *pool, size_t n, fu_for_slices_lambda_t, fu_lambda_context_t);

fu_numa_pools_t *fu_numa_pools_new();
void fu_numa_pools_delete(fu_numa_pools_t *pool);
bool fu_numa_pools_try_spawn(fu_numa_pools_t *pool, size_t threads, size_t scratch_space);
void fu_numa_pools_sleep(fu_numa_pools_t *pool, size_t micros);
void fu_numa_pools_terminate(fu_numa_pools_t *pool);

void fu_numa_pools_broadcast(fu_numa_pools_t *pool, fu_broadcast_lambda_t, fu_lambda_context_t);
void fu_numa_pools_for_n(fu_numa_pools_t *pool, size_t n, fu_for_n_lambda_t, fu_lambda_context_t);
void fu_numa_pools_for_n_dynamic(fu_numa_pools_t *pool, size_t n, fu_for_n_lambda_t, fu_lambda_context_t);
void fu_numa_pools_for_slices(fu_numa_pools_t *pool, size_t n, fu_for_slices_lambda_t, fu_lambda_context_t);

size_t fu_count_cores();
size_t fu_count_numa_nodes();
void *fu_numa_alloc(size_t size, size_t node);
void fu_numa_dealloc(void *begin, size_t size, size_t node);
void *fu_numa_scratch_space(size_t node);

#ifdef __cplusplus
} // extern "C"
#endif