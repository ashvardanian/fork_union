#include <cstdio>    // `std::printf`, `std::fprintf`
#include <cstdlib>   // `EXIT_FAILURE`, `EXIT_SUCCESS`
#include <vector>    // `std::vector`
#include <algorithm> // `std::sort`

#include <fork_union.hpp>

namespace fu = ashvardanian::fork_union;

constexpr std::size_t default_parts = 10000; // 10K

static bool test_try_spawn_success() noexcept {
    fu::thread_pool_t pool;
    std::size_t const count_threads = std::thread::hardware_concurrency();
    if (!pool.try_spawn(count_threads)) return false;
    return true;
}

static bool test_try_spawn_zero() noexcept {
    fu::thread_pool_t pool;
    return !pool.try_spawn(0u);
}

/** @brief Make sure that `broadcast` is called from each thread. */
static bool test_broadcast() noexcept {
    std::size_t const count_threads = std::thread::hardware_concurrency();
    std::vector<std::atomic<bool>> visited(count_threads);
    {
        fu::thread_pool_t pool;
        if (!pool.try_spawn(count_threads)) return false;
        pool.broadcast([&](std::size_t const thread_index) noexcept {
            visited[thread_index].store(true, std::memory_order_relaxed);
        });
    }
    for (std::size_t i = 0; i < count_threads; ++i)
        if (!visited[i]) return false;
    return true;
}

/** @brief Shows how to control multiple thread-pools from the same main thread. */
static bool test_exclusivity() noexcept {
    constexpr std::size_t count_threads = 4;
    std::vector<std::atomic<bool>> visited(count_threads);
    {
        fu::thread_pool_t first_pool, second_pool;
        if (!first_pool.try_spawn(2, fu::caller_inclusive_k)) return false;
        if (!second_pool.try_spawn(2, fu::caller_exclusive_k)) return false;
        auto join_second = second_pool.broadcast([&](std::size_t const thread_index) noexcept {
            visited[2 + thread_index].store(true, std::memory_order_relaxed);
        });
        first_pool.broadcast([&](std::size_t const thread_index) noexcept {
            visited[thread_index].store(true, std::memory_order_relaxed);
        });
        join_second.wait();
    }
    for (std::size_t i = 0; i < count_threads; ++i)
        if (!visited[i]) return false;
    return true;
}

/** @brief Make sure that `for_n` is called from each thread. */
static bool test_uncomfortable_input_size() noexcept {
    std::size_t const count_threads = std::thread::hardware_concurrency();

    fu::thread_pool_t pool;
    if (!pool.try_spawn(count_threads)) return false;

    for (std::size_t input_size = 0; input_size < count_threads; ++input_size) {
        std::atomic<bool> out_of_bounds(false);
        for_n(pool, input_size, [&](std::size_t const task_index) noexcept {
            if (task_index >= count_threads) out_of_bounds.store(true, std::memory_order_relaxed);
        });
        if (out_of_bounds.load(std::memory_order_relaxed)) return false;
    }

    return true;
}

/** @brief Convenience structure to ensure we output match locations to independent cache lines. */
struct alignas(fu::default_alignment_k) aligned_visit_t {
    std::size_t task_index = 0;
    bool operator<(aligned_visit_t const &other) const noexcept { return task_index < other.task_index; }
    bool operator==(aligned_visit_t const &other) const noexcept { return task_index == other.task_index; }
    bool operator!=(std::size_t other_index) const noexcept { return task_index != other_index; }
    bool operator==(std::size_t other_index) const noexcept { return task_index == other_index; }
};

bool contains_iota(std::vector<aligned_visit_t> &visited) noexcept {
    std::sort(visited.begin(), visited.end());
    std::size_t visited_progress = 0;
    for (; visited_progress < visited.size(); ++visited_progress)
        if (visited[visited_progress] != visited_progress) break;
    if (visited_progress != visited.size()) {
        return false; // ! Put on a separate line for a breakpoint
    }
    return true;
}

/** @brief Make sure that `for_n` is called the right number of times with the right prong IDs. */
static bool test_for_n() noexcept {

    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(default_parts);

    fu::thread_pool_t pool;
    std::size_t const count_threads = std::thread::hardware_concurrency();
    if (!pool.try_spawn(count_threads)) return false;
    for_n(pool, default_parts, [&](std::size_t const task_index) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = task_index;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parts`).
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure repeated calls to `for_n` work
    counter = 0;
    for_n(pool, default_parts, [&](std::size_t const task_index) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = task_index;
    });

    return counter.load() == default_parts && contains_iota(visited);
}

/** @brief Make sure that `for_n_dynamic` is called the right number of times with the right prong IDs. */
static bool test_for_n_dynamic() noexcept {

    fu::thread_pool_t pool;
    std::size_t const count_threads = std::thread::hardware_concurrency();
    if (!pool.try_spawn(count_threads)) return false;
    std::vector<aligned_visit_t> visited(default_parts);
    std::atomic<std::size_t> counter(0);
    for_n_dynamic(pool, default_parts, [&](std::size_t const task_index) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = task_index;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parts`).
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure repeated calls to `for_n` work
    counter = 0;
    for_n_dynamic(pool, default_parts, [&](std::size_t const task_index) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = task_index;
    });

    return counter.load() == default_parts && contains_iota(visited);
}

/** @brief Stress-tests the implementation by oversubscribing the number of threads. */
static bool test_oversubscribed_unbalanced_threads() noexcept {
    constexpr std::size_t oversubscription = 3;

    fu::thread_pool_t pool;
    std::size_t const count_threads = std::thread::hardware_concurrency() * oversubscription;
    if (!pool.try_spawn(count_threads)) return false;
    std::vector<aligned_visit_t> visited(default_parts);
    std::atomic<std::size_t> counter(0);
    thread_local volatile std::size_t some_local_work = 0;
    for_n_dynamic(pool, default_parts, [&](std::size_t const task_index) noexcept {
        // Perform some weird amount of work, that is not very different between consecutive tasks.
        for (std::size_t i = 0; i != task_index % oversubscription; ++i) some_local_work = some_local_work + i * i;

        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = task_index;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parts`).
    return counter.load() == default_parts && contains_iota(visited);
}

/** @brief Make sure that that we can combine static and dynamic workloads over the same pool with & w/out resetting. */
template <bool should_restart_>
static bool test_mixed_restart() noexcept {

    fu::thread_pool_t pool;
    std::size_t const count_threads = std::thread::hardware_concurrency();
    if (!pool.try_spawn(count_threads)) return false;
    std::vector<aligned_visit_t> visited(default_parts);
    std::atomic<std::size_t> counter(0);

    fu::for_n(pool, default_parts, [&](std::size_t const task_index) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = task_index;
    });
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure that the pool can be reset and reused
    if (should_restart_) {
        pool.terminate();
        if (!pool.try_spawn(count_threads)) return false;
    }

    // Make sure repeated calls to `for_n` work
    counter = 0;
    fu::for_n_dynamic(pool, default_parts, [&](std::size_t const task_index) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = task_index;
    });

    return counter.load() == default_parts && contains_iota(visited);
}

/** @brief Hard complex example, involving launching multiple tasks, including static and dynamic ones,
 *         stopping them half-way, resetting & reinitializing, and raising exceptions.
 */
template <typename pool_type_>
static bool stress_test_composite(std::size_t const threads_count, std::size_t const default_parts) noexcept {

    using pool_t = pool_type_;
    using index_t = typename pool_t::index_t;
    using prong_t = fu::prong<index_t>;

    pool_t pool;
    if (!pool.try_spawn(threads_count)) return false;

    // Make sure that no overflow happens in the static scheduling
    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(default_parts);
    fu::for_n(pool, default_parts, [&](prong_t prong) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = prong.task_index;
    });
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure that no overflow happens in the dynamic scheduling
    counter = 0;
    fu::for_n_dynamic(pool, default_parts, [&](prong_t prong) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task_index = prong.task_index;
    });
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure the operations can be interrupted from inside the prong
    return true;
}

void log_numa_topology(void) {
#if FU_ENABLE_NUMA
    fu::numa_topology topo;
    if (!topo.try_harvest()) {
        std::fprintf(stderr, "Failed to harvest NUMA topology\n");
        return;
    }

    std::printf("Harvested NUMA topology:\n");
    std::printf("- %zu nodes, %zu threads\n", topo.nodes_count(), topo.threads_count());
    for (std::size_t i = 0; i < topo.nodes_count(); ++i) {
        auto const n = topo.node(i);
        std::printf("- node %d : %zu MiB, %zu cores: %d...%d\n",  //
                    n.node_id, n.memory_size >> 20, n.core_count, //
                    n.first_core_id[0], n.first_core_id[n.core_count - 1]);
    }
#endif
}

int main(void) {

    std::printf("Welcome to the Fork Union library test suite!\n");
    log_numa_topology();

    std::printf("Starting unit tests...\n");
    using test_func_t = bool() /* noexcept */;
    struct {
        char const *name;
        test_func_t *function;
    } const unit_tests[] = {
        {"`try_spawn` normal", test_try_spawn_success},                                     //
        {"`try_spawn` zero threads", test_try_spawn_zero},                                  //
        {"`broadcast` dispatch", test_broadcast},                                           //
        {"`caller_exclusive_k` calls", test_exclusivity},                                   //
        {"`for_n` for uncomfortable input size", test_uncomfortable_input_size},            //
        {"`for_n` static scheduling", test_for_n},                                          //
        {"`for_n_dynamic` dynamic scheduling", test_for_n_dynamic},                         //
        {"`for_n_dynamic` oversubscribed threads", test_oversubscribed_unbalanced_threads}, //
        {"`terminate` avoided", test_mixed_restart<false>},                                 //
        {"`terminate` and re-spawn", test_mixed_restart<true>},                             //
    };

    std::size_t const total_unit_tests = sizeof(unit_tests) / sizeof(unit_tests[0]);
    std::size_t failed_unit_tests = 0;
    for (std::size_t i = 0; i < total_unit_tests; ++i) {
        std::printf("Running %s... ", unit_tests[i].name);
        bool const ok = unit_tests[i].function();
        if (ok) { std::printf("PASS\n"); }
        else { std::printf("FAIL\n"); }
        failed_unit_tests += !ok;
    }

    if (failed_unit_tests > 0) {
        std::fprintf(stderr, "%zu/%zu unit tests failed\n", failed_unit_tests, total_unit_tests);
        return EXIT_FAILURE;
    }
    std::printf("All %zu unit tests passed\n", total_unit_tests);

    // Start stress-testing the implementation
    std::printf("Starting stress tests...\n");
    std::size_t const max_cores = std::thread::hardware_concurrency();
    using fu32_t = fu::thread_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint32_t>;
    using fu16_t = fu::thread_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint16_t>;
    using fu8_t = fu::thread_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint8_t>;
    using stress_test_func_t = bool(std::size_t, std::size_t) /* noexcept */;
    struct {
        char const *name;
        stress_test_func_t *function;
        std::size_t count_threads;
        std::size_t count_tasks;
    } const stress_tests[] = {
        {"`fu8` with 3 threads & 3 inputs", &stress_test_composite<fu8_t>, 3, 3},
        {"`fu8` with 3 threads & 2 inputs", &stress_test_composite<fu8_t>, 3, 2},
        {"`fu8` with 3 threads & 4 inputs", &stress_test_composite<fu8_t>, 3, 4},
        {"`fu8` with 3 threads & 5 inputs", &stress_test_composite<fu8_t>, 3, 5},
        {"`fu8` with 7 threads & 255 inputs", &stress_test_composite<fu8_t>, 7, 255},
        {"`fu8` with 255 threads & 7 inputs", &stress_test_composite<fu8_t>, 255, 7},
        {"`fu8` with 253 threads & 254 inputs", &stress_test_composite<fu8_t>, 253, 254},
        {"`fu8` with 253 threads & 255 inputs", &stress_test_composite<fu8_t>, 253, 255},
        {"`fu8` with 255 threads & 255 inputs", &stress_test_composite<fu8_t>, 255, 255},
        {"`fu16` with thread/core & 65K inputs", &stress_test_composite<fu16_t>, max_cores, UINT16_MAX},
        {"`fu16` with 333 threads & 65K inputs", &stress_test_composite<fu16_t>, 333, UINT16_MAX},
    };

    std::size_t const total_stress_tests = sizeof(stress_tests) / sizeof(stress_tests[0]);
    std::size_t failed_stress_tests = 0;
    for (std::size_t i = 0; i < total_stress_tests; ++i) {
        std::printf("Running %s... ", stress_tests[i].name);
        bool const ok = stress_tests[i].function(stress_tests[i].count_threads, stress_tests[i].count_tasks);
        if (ok) { std::printf("PASS\n"); }
        else { std::printf("FAIL\n"); }
        failed_stress_tests += !ok;
    }

    if (failed_stress_tests > 0) {
        std::fprintf(stderr, "%zu/%zu stress tests failed\n", failed_stress_tests, total_stress_tests);
        return EXIT_FAILURE;
    }
    std::printf("All %zu stress tests passed\n", total_stress_tests);

    return EXIT_SUCCESS;
}
