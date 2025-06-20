#include <cstdio>    // `std::printf`, `std::fprintf`
#include <cstdlib>   // `EXIT_FAILURE`, `EXIT_SUCCESS`
#include <vector>    // `std::vector`
#include <algorithm> // `std::sort`

#include <fork_union.hpp>

/* Namespaces, constants, and explicit type instantiations. */
namespace fu = ashvardanian::fork_union;

using fu32_t = fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint32_t>;
using fu16_t = fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint16_t>;
using fu8_t = fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint8_t>;

template class fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint32_t>;
template class fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint16_t>;
template class fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t, std::uint8_t>;
template class fu::basic_pool<>;

#if FU_ENABLE_NUMA
template class fu::linux_colocated_pool<>;
template class fu::linux_distributed_pool<>;
#endif

constexpr std::size_t default_parts = 10000; // 10K

struct make_pool_t {
    fu::basic_pool_t construct() const noexcept { return fu::basic_pool_t(); }
    std::size_t scope(std::size_t oversubscription = 1) const noexcept {
        return std::thread::hardware_concurrency() * oversubscription;
    }
};

#if FU_ENABLE_NUMA
static fu::ram_page_settings_t ram_settings;
static fu::numa_topology_t numa_topology;
struct make_linux_colocated_pool_t {
    fu::linux_colocated_pool_t construct() const noexcept { return fu::linux_colocated_pool_t("fork_union"); }
    fu::numa_node_t scope(std::size_t = 0) const noexcept { return numa_topology.node(0); }
};
struct make_linux_distributed_pool_t {
    fu::linux_distributed_pool_t construct() const noexcept { return fu::linux_distributed_pool_t("fork_union"); }
    std::size_t scope(std::size_t = 0) const noexcept { return numa_topology.threads_count(); }
};
#endif

static bool test_try_spawn_zero() noexcept {
    fu::basic_pool_t pool;
    return !pool.try_spawn(0u);
}

template <typename make_pool_type_ = make_pool_t>
static bool test_try_spawn_success() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;
    return true;
}

/** @brief Make sure that `broadcast` is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static bool test_broadcast() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    pool.for_threads([&](std::size_t const thread_index) noexcept { //
        visited[thread_index].store(true, std::memory_order_relaxed);
    });

    for (std::size_t i = 0; i < pool.threads_count(); ++i)
        if (!visited[i]) return false;
    return true;
}

/** @brief Shows how to control multiple thread-pools from the same main thread. */
template <typename make_pool_type_ = make_pool_t>
static bool test_exclusivity() noexcept {

    auto maker = make_pool_type_ {};

    // First try with externally defined lambdas with a clearly long lifetime:
    {
        auto first_pool = maker.construct();
        auto second_pool = maker.construct();
        if (!first_pool.try_spawn(maker.scope(), fu::caller_inclusive_k)) return false;
        if (!second_pool.try_spawn(maker.scope(), fu::caller_exclusive_k)) return false;

        std::size_t const first_size = first_pool.threads_count();
        std::size_t const second_size = second_pool.threads_count();
        std::size_t const total_size = first_size + second_size;
        std::vector<std::atomic<bool>> visited(total_size);

        auto do_second = [&](std::size_t const thread_index) noexcept {
            visited[first_size + thread_index].store(true, std::memory_order_relaxed);
        };
        auto do_first = [&](std::size_t const thread_index) noexcept {
            visited[thread_index].store(true, std::memory_order_relaxed);
        };

        // Repeat the same logic a few times and check for correctness:
        for (std::size_t iteration = 0; iteration < 3; ++iteration) {
            auto join_second = second_pool.for_threads(do_second);
            first_pool.for_threads(do_first);
            join_second.join();

            // Validate:
            for (std::size_t i = 0; i < total_size; ++i)
                if (!visited[i]) return false;
        }
    }

    // Now do the same with inline lambdas, where they should be re-packaged into returned objects:
    {
        auto first_pool = maker.construct();
        auto second_pool = maker.construct();
        if (!first_pool.try_spawn(maker.scope(), fu::caller_inclusive_k)) return false;
        if (!second_pool.try_spawn(maker.scope(), fu::caller_exclusive_k)) return false;

        std::size_t const first_size = first_pool.threads_count();
        std::size_t const second_size = second_pool.threads_count();
        std::size_t const total_size = first_size + second_size;
        std::vector<std::atomic<bool>> visited(total_size);

        auto join_second = second_pool.for_threads([&](std::size_t const thread_index) noexcept {
            visited[first_size + thread_index].store(true, std::memory_order_relaxed);
        });
        first_pool.for_threads([&](std::size_t const thread_index) noexcept {
            visited[thread_index].store(true, std::memory_order_relaxed);
        });
        join_second.join();

        // Validate:
        for (std::size_t i = 0; i < total_size; ++i)
            if (!visited[i]) return false;
    }
    return true;
}

/** @brief Make sure that `for_n` is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static bool test_uncomfortable_input_size() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::size_t const max_input_size = pool.threads_count() * 3; // Arbitrary size, larger than the number of threads
    for (std::size_t input_size = 0; input_size <= max_input_size; ++input_size) {
        std::atomic<bool> out_of_bounds(false);
        pool.for_n(input_size, [&](std::size_t const task) noexcept {
            if (task >= input_size) out_of_bounds.store(true, std::memory_order_relaxed);
        });
        if (out_of_bounds.load(std::memory_order_relaxed)) return false;
    }

    return true;
}

/** @brief Convenience structure to ensure we output match locations to independent cache lines. */
struct alignas(fu::default_alignment_k) aligned_visit_t {
    std::size_t task = 0;
    bool operator<(aligned_visit_t const &other) const noexcept { return task < other.task; }
    bool operator==(aligned_visit_t const &other) const noexcept { return task == other.task; }
    bool operator!=(std::size_t other_index) const noexcept { return task != other_index; }
    bool operator==(std::size_t other_index) const noexcept { return task == other_index; }
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
template <typename make_pool_type_ = make_pool_t>
static bool test_for_n() noexcept {

    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(default_parts);

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    pool.for_n(default_parts, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parts`).
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n(default_parts, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    return counter.load() == default_parts && contains_iota(visited);
}

/** @brief Make sure that `for_n_dynamic` is called the right number of times with the right prong IDs. */
template <typename make_pool_type_ = make_pool_t>
static bool test_for_n_dynamic() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::vector<aligned_visit_t> visited(default_parts);
    std::atomic<std::size_t> counter(0);
    pool.for_n_dynamic(default_parts, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parts`).
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n_dynamic(default_parts, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    return counter.load() == default_parts && contains_iota(visited);
}

/** @brief Stress-tests the implementation by oversubscribing the number of threads. */
template <typename make_pool_type_ = make_pool_t>
static bool test_oversubscribed_threads() noexcept {
    constexpr std::size_t oversubscription = 3;

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope(oversubscription))) return false;

    std::vector<aligned_visit_t> visited(default_parts);
    std::atomic<std::size_t> counter(0);
    thread_local volatile std::size_t some_local_work = 0;
    pool.for_n_dynamic(default_parts, [&](std::size_t const task) noexcept {
        // Perform some weird amount of work, that is not very different between consecutive tasks.
        for (std::size_t i = 0; i != task % oversubscription; ++i) some_local_work = some_local_work + i * i;

        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all prong IDs are unique and form the full range of [0, `default_parts`).
    return counter.load() == default_parts && contains_iota(visited);
}

/** @brief Make sure that that we can combine static and dynamic workloads over the same pool with & w/out resetting. */
template <bool should_restart_, typename make_pool_type_ = make_pool_t>
static bool test_mixed_restart() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!pool.try_spawn(maker.scope())) return false;

    std::vector<aligned_visit_t> visited(default_parts);
    std::atomic<std::size_t> counter(0);

    pool.for_n(default_parts, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure that the pool can be reset and reused
    if (should_restart_) {
        pool.terminate();
        if (!pool.try_spawn(maker.scope())) return false;
    }

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n_dynamic(default_parts, [&](std::size_t const task) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
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
    pool.for_n(default_parts, [&](prong_t prong) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = prong.task;
    });
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure that no overflow happens in the dynamic scheduling
    counter = 0;
    pool.for_n_dynamic(default_parts, [&](prong_t prong) noexcept {
        // ? Relax the memory order, as we don't care about the order of the results, will sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = prong.task;
    });
    if (counter.load() != default_parts) return false;
    if (!contains_iota(visited)) return false;

    // Make sure the operations can be interrupted from inside the prong
    return true;
}

void log_numa_topology(void) {
#if FU_ENABLE_NUMA
    if (!ram_settings.try_harvest()) {
        std::fprintf(stderr, "Failed to harvest RAM page settings\n");
        exit(EXIT_FAILURE);
    }
    if (!numa_topology.try_harvest()) {
        std::fprintf(stderr, "Failed to harvest NUMA topology\n");
        exit(EXIT_FAILURE);
    }

    std::printf("Harvested topology:\n");
    std::printf("- %zu huge page sizes:\n", ram_settings.size());
    for (auto const &setting : ram_settings) {
        if (setting.size_bytes > 1024 * 1024 * 1024) { std::printf("  - %zu GiB\n", setting.size_bytes >> 30); }
        else if (setting.size_bytes > 1024 * 1024) { std::printf("  - %zu MiB\n", setting.size_bytes >> 20); }
        else if (setting.size_bytes > 1024) { std::printf("  - %zu KiB\n", setting.size_bytes >> 10); }
        else { std::printf("  - %zu B\n", setting.size_bytes); }
    }
    std::printf("- %zu NUMA nodes, %zu threads\n", numa_topology.nodes_count(), numa_topology.threads_count());
    for (std::size_t i = 0; i < numa_topology.nodes_count(); ++i) {
        auto const n = numa_topology.node(i);
        std::printf("- node %d : %zu MiB, %zu cores: %d...%d\n",  //
                    n.node_id, n.memory_size >> 20, n.core_count, //
                    n.first_core_id[0], n.first_core_id[n.core_count - 1]);
    }
    std::printf("Current process ID: %d\n", ::getpid());
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
        {"`try_spawn` zero threads", test_try_spawn_zero},                       //
        {"`try_spawn` normal", test_try_spawn_success},                          //
        {"`broadcast` dispatch", test_broadcast},                                //
        {"`caller_exclusive_k` calls", test_exclusivity},                        //
        {"`for_n` for uncomfortable input size", test_uncomfortable_input_size}, //
        {"`for_n` static scheduling", test_for_n},                               //
        {"`for_n_dynamic` dynamic scheduling", test_for_n_dynamic},              //
        {"`for_n_dynamic` oversubscribed threads", test_oversubscribed_threads}, //
        {"`terminate` avoided", test_mixed_restart<false>},                      //
        {"`terminate` and re-spawn", test_mixed_restart<true>},                  //
#if FU_ENABLE_NUMA
        {"NUMA `try_spawn` normal", test_try_spawn_success<make_linux_colocated_pool_t>},
        {"NUMA `broadcast` dispatch", test_broadcast<make_linux_colocated_pool_t>},
        {"NUMA `caller_exclusive_k` calls", test_exclusivity<make_linux_colocated_pool_t>},
        {"NUMA `for_n` for uncomfortable input size", test_uncomfortable_input_size<make_linux_colocated_pool_t>},
        {"NUMA `for_n` static scheduling", test_for_n<make_linux_colocated_pool_t>},
        {"NUMA `for_n_dynamic` dynamic scheduling", test_for_n_dynamic<make_linux_colocated_pool_t>},
        {"NUMA `for_n_dynamic` oversubscribed threads", test_oversubscribed_threads<make_linux_colocated_pool_t>},
        {"NUMA `terminate` avoided", test_mixed_restart<false, make_linux_colocated_pool_t>},
        {"NUMA `terminate` and re-spawn", test_mixed_restart<true, make_linux_colocated_pool_t>},
#endif // FU_ENABLE_NUMA
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
