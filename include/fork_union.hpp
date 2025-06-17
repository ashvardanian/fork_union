/**
 *  @brief  OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   fork_union.hpp
 *  @author Ash Vardanian
 *  @date   May 2, 2025
 *
 *  Fork Union provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
 *  avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
 *  The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
 *  execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
 *  to use even with the maximal `std::size_t` values. It's compatible with C++11 and later.
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp> // `fu::basic_pool_t`
 *
 *  using fu = ashvardanian::fork_union;
 *  int main(int argc, char *argv[]) {
 *
 *      fu::basic_pool_t pool;
 *      if (!pool.try_spawn(std::thread::hardware_concurrency()))
 *          return EXIT_FAILURE;
 *
 *      pool.for_n(argc, [=](auto prong) noexcept {
 *          auto [task_index, thread_index, colocation_index] = prong;
 *          std::printf(
 *              "Printing argument # %zu (of %zu) from thread # %zu at colocation # %zu: %s\n",
 *              task_index, argc, thread_index, colocation_index, argv[task_index]);
 *      });
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  The next layer of logic is for basic index-addressable tasks. It includes basic parallel loops:
 *
 *  - `for_n` - for iterating over a range of similar duration tasks, addressable by an index.
 *  - `for_n_dynamic` - for unevenly distributed tasks, where each task may take a different time.
 *  - `for_slices` - for iterating over a range of similar duration tasks, addressable by a slice.
 *
 *  ------------------------------------------------------------------------------------------------
 *
 *  On Linux, when libNUMA and libpthread are available, the library can also leverage NUMA-aware
 *  memory allocations and pin threads to specific physical cores to increase memory locality.
 *  It should reduce memory access latency by around 35% on average, compared to remote accesses.
 *  @sa `numa_topology_t`, `linux_colocated_pool_t`, `linux_pool_t`.
 *
 *  On x86, Arm, and RISC-V architectures, depending on the CPU features available, the library also
 *  exposes cheaper "busy waiting" mechanisms, such as `tpause`, `wfet`, and `yield` instructions.
 *  @sa `aarch64_yield_t`, `aarch64_wfet_t`, `x86_yield_t`, `x86_tpause_1us_t`, `riscv_yield_t`.
 *
 *  Minimum version of C++ 14 is needed to allow an `auto` placeholder type for return values.
 *  This significantly reduces code bloat needed to infer the return type of lambdas.
 *  @see https://en.cppreference.com/w/cpp/language/auto.html
 */
#pragma once
#include <memory>  // `std::allocator`
#include <thread>  // `std::thread`
#include <atomic>  // `std::atomic`
#include <cstddef> // `std::max_align_t`
#include <cassert> // `assert`
#include <cstring> // `std::strlen`
#include <utility> // `std::exchange`, `std::addressof`
#include <new>     // `std::hardware_destructive_interference_size`

#define FORK_UNION_VERSION_MAJOR 1
#define FORK_UNION_VERSION_MINOR 0
#define FORK_UNION_VERSION_PATCH 4

#if !defined(FU_ALLOW_UNSAFE)
#define FU_ALLOW_UNSAFE 0
#endif

#if !defined(FU_ENABLE_NUMA)
#if defined(__linux__) && defined(__GLIBC__) && __GLIBC_PREREQ(2, 30)
#define FU_ENABLE_NUMA 1
#else
#define FU_ENABLE_NUMA 0
#endif
#endif

#if FU_ALLOW_UNSAFE
#include <exception> // `std::exception_ptr`
#endif

#if FU_ENABLE_NUMA
#include <numa.h>    // `numa_available`, `numa_node_of_cpu`, `numa_alloc_onnode`
#include <pthread.h> // `pthread_getaffinity_np`
#include <unistd.h>  // `gettid`
#endif

/**
 *  On C++17 and later we can detect misuse of lambdas that are not properly annotated.
 *  On C++20 and later we can use concepts for cleaner compile-time checks.
 */
#define _FU_DETECT_CPP_20 (__cplusplus >= 202002L)
#define _FU_DETECT_CPP_17 (__cplusplus >= 201703L)

#if _FU_DETECT_CPP_17
#include <type_traits> // `std::is_nothrow_invocable_r`
#endif

#if _FU_DETECT_CPP_20
#include <concepts> // `std::same_as`, `std::invocable`
#endif

#if _FU_DETECT_CPP_17
#define _FU_MAYBE_UNUSED [[maybe_unused]]
#else
#if defined(__GNUC__) || defined(__clang__)
#define _FU_MAYBE_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
#define _FU_MAYBE_UNUSED __pragma(warning(suppress : 4100))
#else
#define _FU_MAYBE_UNUSED
#endif
#endif

#if _FU_DETECT_CPP_20
#define _FU_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define _FU_UNLIKELY(x) (x)
#endif

namespace ashvardanian {
namespace fork_union {

#pragma region - Helpers and Constants

/**
 *  @brief Defines variable alignment to avoid false sharing.
 *  @see https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_size
 *  @see https://docs.rs/crossbeam-utils/latest/crossbeam_utils/struct.CachePadded.html
 *
 *  The C++ STL way to do it is to use `std::hardware_destructive_interference_size` if available:
 *
 *  @code{.cpp}
 *  #if defined(__cpp_lib_hardware_interference_size)
 *  static constexpr std::size_t default_alignment_k = std::hardware_destructive_interference_size;
 *  #else
 *  static constexpr std::size_t default_alignment_k = alignof(std::max_align_t);
 *  #endif
 *  @endcode
 *
 *  That however results into all kinds of ABI warnings with GCC, and suboptimal alignment choice,
 *  unless you hard-code `--param hardware_destructive_interference_size=64` or disable the warning
 *  with `-Wno-interference-size`.
 */
static constexpr std::size_t default_alignment_k = 128;

/**
 *  @brief Defines saturated addition for a given unsigned integer type.
 *  @see https://en.cppreference.com/w/cpp/numeric/add_sat
 */
template <typename scalar_type_>
inline scalar_type_ add_sat(scalar_type_ a, scalar_type_ b) noexcept {
    static_assert(std::is_unsigned<scalar_type_>::value, "Scalar type must be an unsigned integer");
#if defined(__cpp_lib_saturation_arithmetic)
    return std::add_sat(a, b); // In C++26
#else
    return (std::numeric_limits<scalar_type_>::max() - a < b) ? std::numeric_limits<scalar_type_>::max() : a + b;
#endif
}

/** @brief Checks if the @p x is a power of two. */
constexpr bool is_power_of_two(std::size_t x) noexcept { return x && ((x & (x - 1)) == 0); }

/**
 *  @brief Defines the in- and exclusivity of the calling thread in for the executing task.
 *  @sa `caller_inclusive_k` and `caller_exclusive_k`
 *
 *  This enum affects how the join is performed. If the caller is inclusive, 1/Nth of the call
 *  will be executed by the calling thread (as opposed to workers) and the join will happen
 *  inside of the calling scope.
 */
enum caller_exclusivity_t {
    caller_inclusive_k,
    caller_exclusive_k,
};

/**
 *  @brief Defines the mood of the thread-pool, whether it is busy or about to die.
 *  @sa `mood_t::grind_k`, `mood_t::chill_k`, `mood_t::die_k`
 */
enum class mood_t {
    grind_k = 0, // ? That's our default ;)
    chill_k,     // ? Sleepy and tired, but just a wake-up call away
    die_k,       // ? The thread is about to die, we must exit the loop peacefully
};

struct standard_yield_t {
    inline void operator()() const noexcept { std::this_thread::yield(); }
};

/**
 *  @brief A synchronization point that waits for all threads to finish the last broadcasted call.
 *
 *  You don't have to explicitly handle the return value and wait on it.
 *  According to the  C++ standard, the destructor of the `broadcast_join_t` will be called
 *  in the end of the `for_threads`-calling expression.
 */
template <typename basic_pool_type_, typename function_type_>
struct broadcast_join {

    using basic_pool_t = basic_pool_type_;
    using function_t = function_type_;

  private:
    basic_pool_t &pool_;
    function_t function_; // ? We need this to extend the lifetime of the lambda

  public:
    explicit broadcast_join(basic_pool_t &pool, function_t func) noexcept : pool_(pool), function_(func) {}

    broadcast_join(broadcast_join &&) = default;
    broadcast_join(broadcast_join const &) = delete;
    broadcast_join &operator=(broadcast_join &&) = default;
    broadcast_join &operator=(broadcast_join const &) = delete;
    ~broadcast_join() noexcept { wait(); }
    void wait() const noexcept { pool_.unsafe_join(); }
    function_t const &function() const noexcept { return function_; }
};

/**
 *  @brief A "prong" - is a tip of a "fork" - pinning "task" to a "thread" and "memory" location.
 *  @note On heterogeneous chips, different QoS can be differentiated via a "colocation" identifier.
 */
template <typename index_type_ = std::size_t>
struct prong {
    using index_t = index_type_;
    using task_index_t = index_t;       // ? A.k.a. "task index" in [0, prongs_count)
    using thread_index_t = index_t;     // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using memory_index_t = index_t;     // ? A.k.a. NUMA "node ID" in [0, numa_nodes_count)
    using colocation_index_t = index_t; // ? A.k.a. NUMA or QoS-specific "colocation ID"

    task_index_t task {0};
    thread_index_t thread {0};
    memory_index_t memory {0};
    colocation_index_t colocation {0};

    constexpr prong() = default;
    constexpr prong(prong const &) = default;
    constexpr prong(prong &&) = default;
    constexpr prong &operator=(prong const &) = default;
    constexpr prong &operator=(prong &&) = default;

    inline prong(task_index_t task, thread_index_t thread = 0, memory_index_t memory = 0,
                 colocation_index_t colocation = 0) noexcept
        : task(task), thread(thread), memory(memory), colocation(colocation) {}

    inline operator task_index_t() const noexcept { return task_index; }
};

using prong_t = prong<>; // ? Default prong type with `std::size_t` indices

/**
 *  @brief Placeholder type for Parallel Algorithms.
 */
struct dummy_lambda_t {};

/**
 *  @brief A trivial minimalistic lock-free "mutex" implementation using `std::atomic_flag`.
 *  @tparam micro_yield_type_ The type of the yield function to be used for busy-waiting.
 *  @tparam alignment_ The alignment of the mutex. Defaults to `default_alignment_k`.
 *
 *  The C++ standard would recommend using `std::hardware_destructive_interference_size`
 *  alignment, as well as `std::atomic_flag::notify_one` and `std::this_thread::yield` APIs,
 *  but our solution is better despite being more primitive.
 *
 *  @see Compatible with STL unique locks: https://en.cppreference.com/w/cpp/thread/unique_lock.html
 */
#if _FU_DETECT_CPP_20

template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
class spin_mutex {
    using micro_yield_t = micro_yield_type_;
    static constexpr std::size_t alignment_k = alignment_;
    alignas(alignment_k) std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

  public:
    void lock() noexcept {
        micro_yield_t micro_yield;
        while (flag_.test_and_set(std::memory_order_acquire)) micro_yield();
    }
    bool try_lock() noexcept { return !flag_.test_and_set(std::memory_order_acquire); }
    void unlock() noexcept { flag_.clear(std::memory_order_release); }
};

#else // _FU_DETECT_CPP_20

template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
class spin_mutex {
    using micro_yield_t = micro_yield_type_;
    static constexpr std::size_t alignment_k = alignment_;

    /**
     *  Theoretically, the choice of `std::atomic<bool>` is suboptimal in the presence of `std::atomic_flag`.
     *  The latter is guaranteed to be lock-free, while the former is not. But until C++20, the flag doesn't
     *  have a non-modifying load operation - the `std::atomic_flag::test` was added in C++20.
     *  @see https://en.cppreference.com/w/cpp/atomic/atomic_flag.html
     */
    std::atomic<bool> flag_ {false};

  public:
    void lock() noexcept {
        micro_yield_t micro_yield;
        while (flag_.exchange(true, std::memory_order_acquire)) micro_yield();
    }
    bool try_lock() noexcept { return !flag_.exchange(true, std::memory_order_acquire); }
    void unlock() noexcept { flag_.store(false, std::memory_order_release); }
};

#endif // _FU_DETECT_CPP_20

using spin_mutex_t = spin_mutex<>;

template <typename index_type_ = std::size_t>
struct indexed_range {
    using index_t = index_type_;

    index_t first {0};
    index_t count {0};
};

using indexed_range_t = indexed_range<>;

/**
 *  @brief Splits a range of tasks into fair-sized chunks for each thread.
 *  @see https://lemire.me/blog/2025/05/22/dividing-an-array-into-fair-sized-chunks/
 *
 *  The first `(tasks % threads)` chunks have size `ceil(tasks / threads)`.
 *  The remaining `tasks - (tasks % threads)` chunks have size `floor(tasks / threads)`
 */
template <typename index_type_ = std::size_t>
struct indexed_split {
    using index_t = index_type_;
    using indexed_range_t = indexed_range<index_t>;

    index_t quotient {0};
    index_t remainder {0};

    inline indexed_split() = default;
    inline indexed_split(index_t const tasks_count, index_t const threads_count) noexcept
        : quotient(tasks_count / threads_count), remainder(tasks_count % threads_count) {}

    inline indexed_range_t operator[](index_t const i) const noexcept {
        index_t const begin = quotient * i + (i < remainder ? i : remainder);
        index_t const count = quotient + (i < remainder ? 1 : 0);
        return {begin, count};
    }
};

using indexed_split_t = indexed_split<>;

#pragma endregion - Helpers and Constants

#pragma region - Basic Pool

/**
 *  @brief Minimalistic STL-based non-resizable thread-pool for simultaneous blocking tasks.
 *
 *  This thread-pool @b can't:
 *  - dynamically @b resize: all threads must be stopped and re-initialized to grow/shrink.
 *  - @b re-enter: it can't be used recursively and will deadlock if you try to do so.
 *  - @b copy/move: the threads depend on the address of the parent structure.
 *  - handle @b exceptions: you must `try-catch` them yourself and return `void`.
 *  - @b stop early: assuming the user can do it better, knowing the task granularity.
 *  - @b overflow: as all APIs are aggressively tested with smaller index types.
 *
 *  This allows this thread-pool to be extremely lightweight and fast, @b without heap allocations
 *  and no expensive abstractions. It only uses `std::thread` and `std::atomic`, but avoids
 *  `std::function`, `std::future`, `std::promise`, `std::condition_variable`, that bring
 *  unnecessary overhead.
 *  @see https://ashvardanian.com/posts/beyond-openmp-in-cpp-rust/#four-horsemen-of-performance
 *
 *  Repeated operations are performed with a @b "weak" memory model, to leverage in-hardware
 *  support for atomic fence-less operations on Arm and IBM Power architectures. Most atomic
 *  counters use the "acquire-release" model, and some going further to "relaxed" model.
 *  @see https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
 *  @see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
 *
 *  ------------------------------------------------------------------------------------------------
 *
 *  A minimal example, similar to `#pragma omp parallel` in OpenMP:
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp> // `basic_pool_t`
 *
 *  using fu = ashvardanian::fork_union;
 *  int main() {
 *      fu::basic_pool_t pool; // ? Alias to `fu::basic_pool<>` template
 *      if (!pool.try_spawn(std::thread::hardware_concurrency())) return EXIT_FAILURE;
 *      pool.for_threads([](std::size_t i) noexcept { std::printf("Hi from thread %zu\n", i); });
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  Unlike OpenMP, however, separate thread-pools can be created isolating work and resources.
 *  This is handy when when some logic has to be split between "performance" & "efficiency" cores,
 *  between different NUMA nodes, between GUI and background tasks, etc. It may look like this:
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp> // `basic_pool_t`
 *
 *  using fu = ashvardanian::fork_union;
 *  int main() {
 *      fu::basic_pool_t first_pool, second_pool;
 *      if (!first_pool.try_spawn(2) || !second_pool.try_spawn(2, fu::caller_exclusive_k)) return EXIT_FAILURE;
 *      auto join = second_pool.for_threads([](std::size_t i) noexcept { poll_ssd(i); });
 *      first_pool.for_threads([](std::size_t i) noexcept { poll_nic(i); });
 *      join.wait(); // ! Wait for the second pool to finish
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  ------------------------------------------------------------------------------------------------
 *
 *  @tparam allocator_type_ The type of the allocator to be used for the thread pool.
 *  @tparam micro_yield_type_ The type of the yield function to be used for busy-waiting.
 *  @tparam index_type_ Use `std::size_t`, but or a smaller type for debugging.
 *  @tparam alignment_ The alignment of the thread pool. Defaults to `default_alignment_k`.
 */
template <                                                  //
    typename allocator_type_ = std::allocator<std::thread>, //
    typename micro_yield_type_ = standard_yield_t,          //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k            //
    >
class basic_pool {

  public:
    using allocator_t = allocator_type_;
    using micro_yield_t = micro_yield_type_;
    static_assert(std::is_nothrow_invocable_r<void, micro_yield_t>::value,
                  "Yield must be callable w/out arguments & return void");
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(is_power_of_two(alignment_k), "Alignment must be a power of 2");

    using index_t = index_type_;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using epoch_index_t = index_t;      // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using thread_index_t = index_t;     // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using colocation_index_t = index_t; // ? A.k.a. "NUMA node ID" in [0, numa_nodes_count)
    using indexed_split_t = indexed_split<index_t>;

    using punned_fork_context_t = void const *;                           // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, thread_index_t); // ? Wraps lambda's `operator()`

  private:
    // Thread-pool-specific variables:
    allocator_t allocator_ {};
    std::thread *workers_ {nullptr};
    thread_index_t threads_count_ {0};
    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count
    std::size_t sleep_length_micros_ {0}; // ? How long to sleep in microseconds when waiting for tasks
    alignas(alignment_) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    punned_fork_context_t fork_state_ {nullptr}; // ? Pointer to the users lambda
    trampoline_t fork_trampoline_ {nullptr};     // ? Calls the lambda
    alignas(alignment_) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_) std::atomic<epoch_index_t> epoch_ {0};

  public:
    basic_pool(basic_pool &&) = delete;
    basic_pool(basic_pool const &) = delete;
    basic_pool &operator=(basic_pool &&) = delete;
    basic_pool &operator=(basic_pool const &) = delete;

    basic_pool(allocator_t const &alloc = {}) noexcept : allocator_(alloc) {}
    ~basic_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept { return sizeof(basic_pool) + threads_count() * sizeof(std::thread); }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return mood_.is_lock_free() && threads_to_sync_.is_lock_free(); }

#pragma region Core API

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized.
     */
    thread_index_t threads_count() const noexcept { return threads_count_; }

    /**
     *  @brief Reports if the current calling thread will be used for broadcasts.
     *  @note This API is @b not synchronized.
     */
    caller_exclusivity_t caller_exclusivity() const noexcept { return exclusivity_; }

    /**
     *  @brief Creates a thread-pool with the given number of threads.
     *  @param[in] threads The number of threads to be used.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @retval false if the number of threads is zero or the "workers" allocation failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     */
    bool try_spawn(                   //
        thread_index_t const threads, //
        caller_exclusivity_t const exclusivity = caller_inclusive_k) noexcept {

        if (threads == 0) return false;        // ! Can't have zero threads working on something
        if (threads_count_ != 0) return false; // ! Already initialized

        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (threads == 1 && use_caller_thread) {
            threads_count_ = 1;
            return true; // ! The current thread will always be used
        }

        // Allocate the thread pool
        thread_index_t const worker_threads = threads - use_caller_thread;
        std::thread *const workers = allocator_.allocate(worker_threads);
        if (!workers) return false; // ! Allocation failed

        // Before we start the threads, make sure we set some of the shared
        // state variables that will be used in the `_worker_loop` function.
        workers_ = workers;
        threads_count_ = threads;
        exclusivity_ = exclusivity;
        mood_.store(mood_t::grind_k, std::memory_order_release);
        auto reset_on_failure = [&]() noexcept {
            allocator_.deallocate(workers, threads);
            workers_ = nullptr;
            threads_count_ = 0;
        };

        // Initializing the thread pool can fail for all kinds of reasons,
        // that the `std::thread` documentation describes as "implementation-defined".
        // https://en.cppreference.com/w/cpp/thread/thread/thread
        for (thread_index_t i = 0; i < worker_threads; ++i) {
            try {
                thread_index_t const i_with_caller = i + use_caller_thread;
                new (&workers[i]) std::thread([this, i_with_caller] { _worker_loop(i_with_caller); });
            }
            catch (...) {
                mood_.store(mood_t::die_k, std::memory_order_release);
                for (thread_index_t j = 0; j < i; ++j) {
                    workers[j].join(); // ? Wait for the thread to exit
                    workers[j].~thread();
                }
                reset_on_failure();
                return false;
            }
        }

        return true;
    }

    /**
     *  @brief Executes a @p function in parallel on all threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     *  @return A `broadcast_join` synchronization point that waits in the destructor.
     *  @note Even in the `caller_exclusive_k` mode, can be called from just one thread!
     *  @sa For advanced resource management, consider `unsafe_broadcast` and `unsafe_join`.
     */
    template <typename function_type_>
    broadcast_join<basic_pool, function_type_> for_threads(function_type_ &&function) noexcept {
        broadcast_join<basic_pool, function_type_> joiner {*this, std::forward<function_type_>(function)};
        unsafe_for_threads(joiner.function());
        return joiner;
    }

    /**
     *  @brief Executes a @p function in parallel on all threads, not waiting for the result.
     *  @param[in] function The callback, receiving the thread index as an argument.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename function_type_>
    void unsafe_for_threads(function_type_ const &function) noexcept {

        thread_index_t const threads = threads_count();
        assert(threads != 0 && "Thread pool not initialized");
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (threads == 1 && use_caller_thread) return function(static_cast<thread_index_t>(0));

        // Optional check: even in exclusive mode, only one thread can call this function.
        assert((use_caller_thread || threads_to_sync_.load(std::memory_order_acquire) == 0) &&
               "The broadcast function can be called only from one thread at a time");

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        // ? we live to the higher-level algorithms.
        static_assert(std::is_nothrow_invocable_r<void, function_type_, thread_index_t>::value,
                      "The callback must be invocable with a `thread_index_t` argument");
#endif

        // Configure "fork" details
        fork_state_ = std::addressof(function);
        fork_trampoline_ = &_call_as_lambda<function_type_>;
        threads_to_sync_.store(threads - use_caller_thread, std::memory_order_relaxed);

        // We are most likely already "grinding", but in the unlikely case we are not,
        // let's wake up from the "chilling" state with relaxed semantics. Assuming the sleeping
        // logic for the workers also checks the epoch counter, no synchronization is needed and
        // no immediate wake-up is required.
        mood_t may_be_chilling = mood_t::chill_k;
        mood_.compare_exchange_weak(          //
            may_be_chilling, mood_t::grind_k, //
            std::memory_order_relaxed, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // Execute on the current "main" thread
        if (use_caller_thread) function(static_cast<thread_index_t>(0));
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        micro_yield_t micro_yield;
        while (threads_to_sync_.load(std::memory_order_acquire)) micro_yield();
    }

#pragma endregion Core API

#pragma region Control Flow

    /**
     *  @brief Stops all threads and deallocates the thread-pool after the last call finishes.
     *  @note Can be called from @b any thread at any time.
     *  @note Must `try_spawn` again to re-use the pool.
     *
     *  When and how @b NOT to use this function:
     *  - as a synchronization point between concurrent tasks.
     *
     *  When and how to use this function:
     *  - as a de-facto @b destructor, to stop all threads and deallocate the pool.
     *  - when you want to @b restart with a different number of threads.
     */
    void terminate() noexcept {
        if (threads_count_ == 0) return; // ? Uninitialized

        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (threads_count_ == 1 && use_caller_thread) {
            threads_count_ = 0;
            return; // ? No worker threads to join
        }
        assert(threads_to_sync_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running

        // Notify all worker threads...
        mood_.store(mood_t::die_k, std::memory_order_release);

        // ... and wait for them to finish
        thread_index_t const worker_threads = threads_count_ - use_caller_thread;
        for (thread_index_t i = 0; i != worker_threads; ++i) {
            workers_[i].join();    // ? Wait for the thread to finish
            workers_[i].~thread(); // ? Call destructor
        }

        // Deallocate the thread pool
        allocator_.deallocate(workers_, worker_threads);

        // Prepare for future spawns
        threads_count_ = 0;
        workers_ = nullptr;
        _reset_fork();
        mood_.store(mood_t::grind_k, std::memory_order_relaxed);
        epoch_.store(0, std::memory_order_relaxed);
    }

    /**
     *  @brief Transitions "workers" to a sleeping state, waiting for a wake-up call.
     *  @param[in] wake_up_periodicity_micros How often to check for new work in microseconds.
     *  @note Can only be called @b between the tasks for a single thread. No synchronization is performed.
     *
     *  This function may be used in some batch-processing operations when we clearly understand
     *  that the next task won't be arriving for a while and power can be saved without major
     *  latency penalties.
     *
     *  It may also be used in a high-level Python or JavaScript library offloading some parallel
     *  operations to an underlying C++ engine, where latency is irrelevant.
     */
    void sleep(std::size_t wake_up_periodicity_micros) noexcept {
        assert(wake_up_periodicity_micros > 0 && "Sleep length must be positive");
        sleep_length_micros_ = wake_up_periodicity_micros;
        mood_.store(mood_t::chill_k, std::memory_order_release);
    }

#pragma endregion Control Flow

#pragma region Indexed Task Scheduling

    /**
     *  @brief Distributes @p (n) similar duration calls between threads in slices, as opposed to individual indices.
     *  @param[in] n The total length of the range to split between threads.
     *  @param[in] function The callback, receiving @b `prong_t` or an unsigned integer and the slice length.
     */
    template <typename function_type_ = dummy_lambda_t>
#if _FU_DETECT_CPP_20
        requires( // ? The callback must be invocable with a `prong_t` or a `index_t` argument and an unsigned counter
            std::is_nothrow_invocable_r_v<void, function_type_, prong_t, index_t> ||
            std::is_nothrow_invocable_r_v<void, function_type_, index_t, index_t>)
#endif
    auto for_slices(index_t const n, function_type_ &&function) noexcept {

        thread_index_t const threads = threads_count();
        indexed_split_t const splits {n, threads};
        return for_threads([splits, function](thread_index_t const thread_index) noexcept {
            auto const range = splits[thread_index];
            if (range.count == 0) return; // ? No work for this thread
            function(prong_t {range.first, thread_index}, range.count);
        });
    }

    /**
     *  @brief Distributes @p (n) similar duration calls between threads.
     *  @param[in] n The number of times to call the @p function.
     *  @param[in] function The callback, receiving @b `prong_t` or a call index as an argument.
     *
     *  Is designed for a "balanced" workload, where all threads have roughly the same amount of work.
     *  @sa `for_n_dynamic` for a more dynamic workload.
     *  The @p function is called @p (n) times, and each thread receives a slice of consecutive tasks.
     *  @sa `for_slices` if you prefer to receive workload slices over individual indices.
     */
    template <typename function_type_ = dummy_lambda_t>
    auto for_n(index_t const n, function_type_ &&function) noexcept {
        return for_slices(pool, n, [function](prong_t const start_prong, index_t const count_prongs) noexcept {
            for (index_t i = 0; i < count_prongs; ++i)
                function(prong_t {static_cast<index_t>(start_prong.task + i), start_prong.thread});
        });
    }

    /**
     *  @brief Executes uneven tasks on all threads, greedying for work.
     *  @param[in] n The number of times to call the @p function.
     *  @param[in] function The callback, receiving the `prong_t` or the task index as an argument.
     *  @sa `for_n` for a more "balanced" evenly-splittable workload.
     */
    template <typename function_type_ = dummy_lambda_t>
#if _FU_DETECT_CPP_20
        requires( // ? The callback must be invocable with a `prong_t` or a `index_t` argument
            std::is_nothrow_invocable_r_v<void, function_type_, prong_t> ||
            std::is_nothrow_invocable_r_v<void, function_type_, index_t>)
#endif
    auto for_n_dynamic(index_t const n, function_type_ &&function) noexcept {

        // If there are fewer tasks than threads, each thread gets at most 1 task
        // and that's easier to schedule statically!
        thread_index_t const threads = threads_count();
        index_t const n_dynamic = n > threads ? n - threads : 0;
        assert((n_dynamic + threads) >= n_dynamic && "Overflow detected");
        alignas(alignment_k) std::atomic<index_t> prongs_progress {0};

        // If we run this loop at 1 Billion times per second on a 64-bit machine, then every 585 years
        // of computational time we will wrap around the `std::size_t` capacity for the `new_prong_index`.
        // In case we `n + thread >= std::size_t(-1)`, a simple condition won't be enough.
        // Alternatively, we can make sure, that each thread can do at least one increment of `prongs_progress`
        // without worrying about the overflow. The way to achieve that is to preprocess the trailing `threads`
        // of elements externally, before entering this loop!
        return for_threads([n, n_dynamic, function,
                            prongs_progress = std::move(prongs_progress)](thread_index_t const thread) noexcept {
            // Run (up to) one static prong on the current thread
            index_t const one_static_prong_index = static_cast<index_t>(n_dynamic + thread);
            prong_t prong(thread, one_static_prong_index);
            if (one_static_prong_index < n) function(prong);

            // The rest can be synchronized with a trivial atomic counter
            while (true) {
                prong.task = prongs_progress.fetch_add(1, std::memory_order_relaxed);
                bool const beyond_last_prong = prong.task >= n_dynamic;
                if (beyond_last_prong) break;
                function(prong);
            }
        });
    }

#pragma endregion Indexed Task Scheduling

#pragma region Colocations Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     *  @retval 1 constant for compatibility.
     */
    constexpr index_t colocations_count() const noexcept { return 1; }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b colocation.
     *  @return Same value as `threads_count()`, as we only support one colocation.
     */
    thread_index_t threads_count(index_t colocation_index) const noexcept {
        assert(colocation_index == 0 && "Only one colocation is supported");
        return count_threads();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b colocation.
     *  @return Same value as `global_thread_index`, as we only support one colocation.
     */
    constexpr thread_index_t thread_local_index(thread_index_t global_thread_index,
                                                index_t colocation_index) const noexcept {
        assert(colocation_index == 0 && "Only one colocation is supported");
        return global_thread_index;
    }

#pragma endregion Colocations Compatibility

  private:
    void _reset_fork() noexcept {
        fork_state_ = nullptr;
        fork_trampoline_ = nullptr;
    }

    /**
     *  @brief A trampoline function that is used to call the user-defined lambda.
     *  @param[in] punned_lambda_pointer The pointer to the user-defined lambda.
     *  @param[in] prong The index of the thread & task index packed together.
     */
    template <typename function_type_>
    static void _call_as_lambda(punned_fork_context_t punned_lambda_pointer, thread_index_t thread_index) noexcept {
        function_type_ const &lambda_object = *static_cast<function_type_ const *>(punned_lambda_pointer);
        lambda_object(thread_index);
    }

    /**
     *  @brief The worker thread loop that is called by each of `this->workers_`.
     *  @param[in] thread_index The index of the thread that is executing this function.
     */
    void _worker_loop(thread_index_t const thread_index) noexcept {
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (use_caller_thread) assert(thread_index != 0 && "The zero index is for the main thread, not worker!");

        epoch_index_t last_epoch = 0;
        while (true) {
            // Wait for either: a new ticket or a stop flag
            epoch_index_t new_epoch;
            mood_t mood;
            micro_yield_t micro_yield;
            while ((new_epoch = epoch_.load(std::memory_order_acquire)) == last_epoch &&
                   (mood = mood_.load(std::memory_order_acquire)) == mood_t::grind_k)
                micro_yield();

            if (_FU_UNLIKELY(mood == mood_t::die_k)) break;
            if (_FU_UNLIKELY(mood == mood_t::chill_k) && (new_epoch == last_epoch)) {
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_length_micros_));
                continue;
            }

            fork_trampoline_(fork_state_, thread_index);
            last_epoch = new_epoch;

            // ! The decrement must come after the task is executed
            _FU_MAYBE_UNUSED thread_index_t const before_decrement =
                threads_to_sync_.fetch_sub(1, std::memory_order_release);
            assert(before_decrement > 0 && "We can't be here if there are no worker threads");
        }
    }
};

using basic_pool_t = basic_pool<>;

#pragma region Concepts
#if _FU_DETECT_CPP_20

struct broadcasted_noop_t {
    template <typename index_type_>
    void operator()(index_type_) const noexcept
        requires(std::unsigned_integral<index_type_> && std::convertible_to<index_type_, std::size_t>)
    {}
};

template <typename pool_type_>
concept is_pool = //
    std::unsigned_integral<decltype(std::declval<pool_type_ const &>().threads_count())> &&
    std::convertible_to<decltype(std::declval<pool_type_ const &>().threads_count()), std::size_t> &&
    requires(pool_type_ &p) {
        { p.for_threads(broadcasted_noop_t {}) };
    };

template <typename pool_type_>
concept is_unsafe_pool =   //
    is_pool<pool_type_> && //
    requires(pool_type_ &p) {
        { p.unsafe_for_threads(broadcasted_noop_t {}) } -> std::same_as<void>;
    } && //
    requires(pool_type_ &p) {
        { p.unsafe_join() } -> std::same_as<void>;
    };

#endif
#pragma endregion Concepts
#pragma endregion - Basic Pool

#pragma region - Hardware Friendly Yield

#if defined(__GNUC__) || defined(__clang__) // We need inline assembly support

#if defined(__aarch64__)

struct aarch64_yield_t {
    inline void operator()() const noexcept { __asm__ __volatile__("yield"); }
};

/**
 *  @brief On AArch64 uses the `WFET` instruction to "Wait For Event (Timed)".
 *
 *  Places the core into light sleep mode, waiting for an event to wake it up,
 *  or the timeout to expire.
 */
struct aarch64_wfet_t {
    inline void operator()() const noexcept {}
};
#endif

#if defined(__x86_64__) || defined(__i386)
struct x86_yield_t {
    inline void operator()() const noexcept { __asm__ __volatile__("pause"); }
};

#pragma GCC push_options
#pragma GCC target("waitpkg")
#pragma clang attribute push(__waitpkg__)

/**
 *  @brief On x86 uses the `TPAUSE` instruction to yield for 1 microsecond if `WAITPKG` is supported.
 *
 *  There are several newer ways to yield on x86, but they may require different privileges:
 *  - `MONITOR` and `MWAIT` in SSE - used for power management, require RING 0 privilege.
 *  - `UMONITOR` and `UMWAIT` in `WAITPKG` - are the user-space variants.
 *  - `MWAITX` in `MONITORX` ISA on AMD - used for power management, requires RING 0 privilege.
 *  - `TPAUSE` in `WAITPKG` - time-based pause instruction, available in RING 3.
 */
struct x86_tpause_1us_t {
    inline void operator()() const noexcept {
        __asm__ volatile( //
            "xor %%eax,%%eax\n\t"
            "mov $1000, %%ebx\n\t"
            ".byte 0x66,0x0F,0xAE,0xFA" ::
                : "eax", "ebx", "memory");
    }
};

#pragma GCC pop_options
#pragma clang attribute pop

#endif

#if defined(__riscv)
struct riscv_yield_t {
    inline void operator()() const noexcept { __asm__ __volatile__("pause"); }
};
#endif

#endif

#pragma endregion - Hardware Friendly Yield

#pragma region - NUMA Pools

using numa_node_id_t = int; // ? A.k.a. NUMA node ID, in [0, numa_max_node())
using numa_core_id_t = int; // ? A.k.a. CPU core ID, in [0, threads_count)
using qos_level_t = int;    // ? Quality of Service, like: "performance", "efficiency", "low-power"

#if FU_ENABLE_NUMA

enum numa_pin_granularity_t {
    numa_pin_to_core_k = 0,
    numa_pin_to_node_k,
};

struct ram_page_setting_t {
    std::size_t size_bytes {0};      // ? Huge page size in bytes, e.g. 4 KB, 2 MB, or 1 GB
    std::size_t available_pages {0}; // ? Number of pages available for this size, 0 if not available
};

/**
 *  @brief Describes the configured & supported (by OS & CPU) memory pages sizes.
 *
 *  @section Huge Pages & Transparent Huge Pages
 *
 *  Virtual Address Space (VAS) is divided into pages, typically 4 KB in size.
 *  Converting a virtual address to a physical address requires a page table lookup.
 *  Think of it as a hash table... and as everyone knows, hash table lookups and updates
 *  aren't free, so most chips have a "Translation Lookaside Buffer" @b (TLB) cache
 *  as part of the "Memory Management Unit" @b (MMU) to speed up the process.
 *
 *  To keep it fast, in Big Data applications, one would like to use larger pages,
 *  to reduce the number of distinct entries in the TLB cache. Going from 4 KB to
 *  2 MB or 1 GB "Huge Pages" @b (HPs), reduces the table size by 512 or 262K times,
 *  respectively.
 *
 *  To benefit from those, some applications rely on "Transparent Huge Pages" @b (THP),
 *  which are automatically allocated by the kernel. Such implicit behaviour isn't
 *  great for performance-oriented applications, so the `linux_numa_allocator` provides
 *  a @b `fetch_max_huge_size` API
 *
 *  @see https://docs.kernel.org/admin-guide/mm/hugetlbpage.html
 */
template <std::size_t max_page_sizes_ = 32>
struct ram_page_settings {
    static constexpr std::size_t max_page_sizes_k = max_page_sizes_;
    std::array<ram_page_setting_t, max_page_sizes_k> sizes {0}; // ? Huge page sizes in bytes
    std::size_t count_sizes {0};                                // ? Number of supported huge page sizes

    /**
     *  @brief Fetches the maximum supported huge page size on the current system.
     *  @retval `numa_pagesize()` if huge pages are not supported or not available.
     *  @retval 2 MB is the most common huge page size on Linux systems.
     *
     *  Being supported by the kernel, doesn't mean that pages of that size have
     *  a valid mount point. That can be checked with @b `hugetlbfs_find_path_for_size`.
     */
    bool try_harvest() noexcept {
        // glibc ≥ 2.10 supports `gethugepagesizes`
        long count = ::gethugepagesizes(nullptr, 0);
        if (count <= 0) return false;

        std::array<long, 32> buf {};
        if (count > static_cast<long>(buf.size())) count = buf.size();
        ::gethugepagesizes(buf.data(), count);

        // Export the settings into the `sizes` array
        for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i) {
            if (buf[i] <= 0) continue; // ? Skip invalid sizes
            sizes[i].size_bytes = static_cast<std::size_t>(buf[i]);
        }
        count_sizes = static_cast<std::size_t>(count);
        return true;
    }

    std::size_t size() const noexcept { return count_sizes; }
    ram_page_setting_t const *begin() const noexcept { return sizes.data(); }
    ram_page_setting_t const *end() const noexcept { return sizes.data() + count_sizes; }
    ram_page_setting_t const &operator[](std::size_t const index) const noexcept {
        assert(index < count_sizes && "Index is out of bounds");
        return sizes[index];
    }
};

using ram_page_settings_t = ram_page_settings<>;

/**
 *  @brief Describes a NUMA node, containing its ID, memory size, and core IDs.
 *  @sa Views different slices of the `numa_topology` structure.
 */
struct numa_node_t {
    numa_node_id_t node_id {-1};                   // ? Unique NUMA node ID, in [0, numa_max_node())
    std::size_t memory_size {0};                   // ? RAM volume in bytes
    numa_core_id_t const *first_core_id {nullptr}; // ? Pointer to the first core ID in the `core_ids` array
    std::size_t core_count {0};                    // ? Number of items in `core_ids` array
};

/**
 *  @brief Used inside `linux_colocated_pool` to describe a pinned thread.
 *
 *  On Linux, we can advise the scheduler on the importance of certain execution threads.
 *  For that we need to know the thread IDs - `pid_t`, which is not the same as `pthread_t`,
 *  and not a process ID, but a thread ID... counter-intuitive, I know.
 *  @see https://man7.org/linux/man-pages/man2/gettid.2.html
 *
 *  That `pid_t` can only be retrieved from inside the thread via `gettid` system call,
 *  so we need some shared memory to make those IDs visible to other threads. Moreover,
 *  we need to safeguard the reads/writes with atomics to avoid race conditions.
 *  @see https://stackoverflow.com/a/558815
 */
struct alignas(default_alignment_k) numa_pthread_t {
    std::atomic<pthread_t> handle;
    std::atomic<pid_t> id;
    numa_core_id_t core_id;
    qos_level_t qos_level;
};

/**
 *  @brief NUMA topology descriptor: describing memory pools and core counts next to them.
 *
 *  Uses dynamic memory to store the NUMA nodes and their cores. Assuming we may soon have
 *  Intel "Sierra Forest"-like CPUs with 288 cores with up to 8 sockets per node, this structure
 *  can easily grow to 10 KB.
 */
template <typename allocator_type_ = std::allocator<char>>
struct numa_topology {

    using allocator_t = allocator_type_;
    using cores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<int>;
    using nodes_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<numa_node_t>;

  private:
    static constexpr std::size_t max_huge_page_sizes_k = 32;

    numa_node_t *nodes_ {nullptr};
    numa_core_id_t *node_core_ids_ {nullptr}; // ? Unsigned integers in [0, threads_count), grouped by NUMA node
    std::size_t nodes_count_ {0};             // ? Number of NUMA nodes
    std::size_t cores_count_ {0};             // ? Total number of cores in all nodes
    allocator_t allocator_ {};
    std::array<std::size_t, max_huge_page_sizes_k> huge_page_sizes_ {0}; // ? Huge page sizes in bytes

  public:
    constexpr numa_topology() noexcept = default;
    numa_topology(numa_topology &&o) noexcept
        : allocator_(std::move(o.allocator_)), nodes_(o.nodes_), node_core_ids_(o.node_core_ids_),
          nodes_count_(o.nodes_count_), cores_count_(o.cores_count_) {
        o.nodes_ = nullptr;
        o.node_core_ids_ = nullptr;
        o.nodes_count_ = 0;
        o.cores_count_ = 0;
    }

    numa_topology(numa_topology const &) = delete;
    numa_topology &operator=(numa_topology const &) = delete;
    numa_topology &operator=(numa_topology &&) = delete;

    ~numa_topology() noexcept { reset(); }

    void reset() noexcept {
        cores_allocator_t cores_alloc {allocator_};
        nodes_allocator_t nodes_alloc {allocator_};

        if (node_core_ids_) cores_alloc.deallocate(node_core_ids_, cores_count_);
        if (nodes_) nodes_alloc.deallocate(nodes_, nodes_count_);

        nodes_ = nullptr;
        node_core_ids_ = nullptr;
        nodes_count_ = cores_count_ = 0;
    }

    std::size_t nodes_count() const noexcept { return nodes_count_; }
    std::size_t threads_count() const noexcept { return cores_count_; }
    numa_node_t node(std::size_t const node_id) const noexcept {
        assert(node_id < nodes_count_ && "Node ID is out of bounds");
        return nodes_[node_id];
    }

    /**
     *  @brief Harvests CPU-memory topology.
     *  @retval false if the kernel lacks NUMA support or the harvest failed.
     *  @retval true if the harvest was successful and the topology is ready to use.
     */
    bool try_harvest() noexcept {
        struct bitmask *numa_mask = nullptr;
        numa_node_t *nodes_ptr = nullptr;
        numa_core_id_t *core_ids_ptr = nullptr;
        numa_node_id_t max_numa_node_id = -1;

        // Allocators must be visible to the cleanup path
        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};

        // These counters are reused in the failure handler
        std::size_t fetched_nodes = 0, fetched_cores = 0;

        if (::numa_available() < 0) goto failed_harvest; // ! Linux kernel lacks NUMA support
        ::numa_node_to_cpu_update();                     // ? Reset the outdated stale state

        numa_mask = ::numa_allocate_cpumask();
        if (!numa_mask) goto failed_harvest; // ! Allocation failed

        // First pass – measure
        max_numa_node_id = ::numa_max_node();
        for (numa_node_id_t node_id = 0; node_id <= max_numa_node_id; ++node_id) {
            long long dummy;
            if (::numa_node_size64(node_id, &dummy) < 0) continue; // ! Offline node
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue; // ! Invalid CPU map
            std::size_t const node_cores = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));
            assert(node_cores > 0 && "Node must have at least one core");
            fetched_nodes += 1;
            fetched_cores += node_cores;
        }
        if (fetched_nodes == 0) goto failed_harvest; // ! Zero nodes is not a valid state

        // Second pass – allocate
        nodes_ptr = nodes_alloc.allocate(fetched_nodes);
        core_ids_ptr = cores_alloc.allocate(fetched_cores);
        if (!nodes_ptr || !core_ids_ptr) goto failed_harvest; // ! Allocation failed

        // Populate
        for (numa_node_id_t node_id = 0, core_index = 0, node_index = 0; node_id <= max_numa_node_id; ++node_id) {
            long long memory_size;
            if (::numa_node_size64(node_id, &memory_size) < 0) continue;
            ::numa_bitmask_clearall(numa_mask);
            if (::numa_node_to_cpus(node_id, numa_mask) < 0) continue;

            numa_node_t &node = nodes_ptr[node_index];
            node.node_id = node_id;
            node.memory_size = static_cast<std::size_t>(memory_size);
            node.first_core_id = core_ids_ptr + core_index;
            node.core_count = static_cast<std::size_t>(::numa_bitmask_weight(numa_mask));
            assert(node.core_count > 0 && "Node is known to have at least one core");

            // Most likely, this will fill `core_ids_ptr` with `std::iota`-like values
            for (std::size_t bit_offset = 0; bit_offset < numa_mask->size; ++bit_offset)
                if (::numa_bitmask_isbitset(numa_mask, bit_offset))
                    core_ids_ptr[core_index++] = static_cast<numa_core_id_t>(bit_offset);

            node_index++;
        }

        // Commit
        nodes_ = nodes_ptr;
        node_core_ids_ = core_ids_ptr;
        nodes_count_ = fetched_nodes;
        cores_count_ = fetched_cores;

        ::numa_free_cpumask(numa_mask); // ? Clean up
        return true;

    failed_harvest:
        if (nodes_ptr) nodes_alloc.deallocate(nodes_ptr, fetched_nodes);
        if (core_ids_ptr) cores_alloc.deallocate(core_ids_ptr, fetched_cores);
        if (numa_mask) ::numa_free_cpumask(numa_mask);
        return false;
    }
};

using numa_topology_t = numa_topology<>;

/**
 *  @brief Back-ports the C++ 23 `std::allocation_result`.
 *  @see https://en.cppreference.com/w/cpp/memory/allocator/allocate_at_least
 */
template <typename pointer_type_ = char, typename size_type_ = std::size_t>
struct allocation_result {
    using pointer_type = pointer_type_;
    using size_type = size_type_;

    pointer_type ptr {nullptr}; // ? Pointer to the allocated memory, or nullptr if allocation failed
    size_type count {0};        // ? Number of elements allocated, or 0 if allocation failed

    constexpr allocation_result() noexcept = default;
    constexpr allocation_result(pointer_type p, size_type s) noexcept : ptr(p), count(s) {}
    explicit constexpr operator bool() const noexcept { return ptr != nullptr && count > 0; }

#if defined(__cpp_lib_allocate_at_least)
    operator std::allocation_result<pointer_type, size_type>() const noexcept {
        return std::allocation_result<pointer_type, size_type>(ptr, count);
    }
#endif
};

/**
 *  @brief STL-compatible allocator pinned to a NUMA node, prioritizing Huge Pages.
 *
 *  A light-weight, but high-latency BLOB allocator, tied to a specific NUMA node ID.
 *  Every allocation is a system call to `mmap` and subsequent `mbind`, aligned to at
 *  least 4 KB page size.
 *
 *  @section C++ 23 Functionality
 *
 *  Whenever possible, the newer `allocate_at_least` API should be used to reduce the
 *  number of reallocations.
 */
template <typename value_type_ = char>
struct linux_numa_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using propagate_on_container_move_assignment = std::true_type;

    numa_node_id_t node_id {-1};  // ? Unique NUMA node ID, in [0, numa_max_node())
    size_type bytes_per_page {0}; // ? NUMA or Huge Page size in bytes, typically 4 KB, 2 MB, or 1 GB

    constexpr linux_numa_allocator() noexcept = default;
    explicit constexpr linux_numa_allocator(numa_node_id_t id, size_type paging = 0) noexcept
        : node_id(id), bytes_per_page(paging) {}

    template <typename other_type_>
    constexpr linux_numa_allocator(linux_numa_allocator<other_type_> const &o) noexcept
        : node_id(o.node_id), bytes_per_page(o.bytes_per_page) {}

    /**
     *  @brief Allocates memory for at least `size` elements of `value_type`.
     *  @param[in] size The number of elements to allocate.
     *  @return allocation_result with a pointer to the allocated memory and the number of elements allocated.
     *  @retval empty object if the allocation failed or the size is not a multiple of `sizeof(value_type)`.
     */
    allocation_result<value_type *, size_type> allocate_at_least(size_type size) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        size_type const page_size_bytes = bytes_per_page == 0 ? ::numa_pagesize() : bytes_per_page;
        size_type const aligned_size_bytes = (size_bytes + page_size_bytes - 1) / page_size_bytes * page_size_bytes;

        // Check if the new size is actually perfectly divisible by the `sizeof(value_type)`
        if (aligned_size_bytes % sizeof(value_type)) return {}; // ! Not a size multiple

        size_type const aligned_size = aligned_size_bytes / sizeof(value_type);
        value_type *result_ptr = allocate(aligned_size);
        if (!result_ptr) return {}; // ! Allocation failed
        return {result_ptr, aligned_size};
    }

    value_type *allocate(size_type n) noexcept {
        size_type const size_bytes = size * sizeof(value_type);
        size_type const page_size_bytes = bytes_per_page == 0 ? ::numa_pagesize() : bytes_per_page;
        size_type const aligned_size_bytes = (size_bytes + page_size_bytes - 1) / page_size_bytes * page_size_bytes;

        // In simple cases, just redirect to `numa_alloc_onnode`
        if (page_size_bytes == ::numa_pagesize())
            return static_cast<value_type *>(::numa_alloc_onnode(aligned_size_bytes, node_id));

        // Make sure the page size makes sense for Linux
        int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (page_size_bytes == 4u * 1024u) { mmap_flags |= MAP_HUGETLB; }
        else if (page_size_bytes == 2u * 1024u * 1024u) { mmap_flags |= MAP_HUGETLB | MAP_HUGE_2MB; }
        else if (page_size_bytes == 1u * 1024u * 1024u * 1024u) { mmap_flags |= MAP_HUGETLB | MAP_HUGE_1GB; }
        else { return nullptr; } // ! Unsupported page size

        // Under the hood, `numa_alloc_onnode` uses `mmap` and `mbind` to allocate memory
        void *result_ptr = ::mmap(nullptr, aligned_size_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
        if (result_ptr == MAP_FAILED) return nullptr; // ! Allocation failed

        // Pin the memory - that may require an extra allocation for `node_mask` on some systems
        ::nodemask_t node_mask;
        ::bitmask node_mask_as_bitset;
        node_mask_as_bitset.size = sizeof(node_mask) * 8;
        node_mask_as_bitset.maskp = &node_mask.n[0];
        numa_bitmask_setbit(&node_mask_as_bitset, node_id);
        long binding_status = ::mbind(      //
            result_ptr, aligned_size_bytes, //
            MPOL_BIND, &node_mask, sizeof(node_mask) * 8 - 1, MPOL_F_STATIC_NODES);
        if (binding_status < 0) return nullptr; // ! Binding failed

        return static_cast<value_type *>(result_ptr);
    }

    void deallocate(value_type *p, size_type n) noexcept { numa_free(p, n * sizeof(value_type)); }

    template <typename other_type_>
    bool operator==(linux_numa_allocator<other_type_> const &o) const noexcept {
        return node_id == o.node_id && bytes_per_page == o.bytes_per_page;
    }

    template <typename other_type_>
    bool operator!=(linux_numa_allocator<other_type_> const &o) const noexcept {
        return node_id != o.node_id || bytes_per_page != o.bytes_per_page;
    }
};

using linux_numa_allocator_t = linux_numa_allocator<>;

#pragma region - Linux Colocated Pool

/**
 *  @brief A Linux-only thread-pool pinned to one NUMA node and same QoS level physical cores.
 *
 *  Differs from the `basic_pool` template in the following ways:
 *  - constructor API: receives a name for the threads.
 *  - implementation & API of `try_spawn`: uses POSIX APIs to allocate, name, & pin threads.
 *  - worker loop: using Linux-specific napping mechanism to reduce power consumption.
 *  - implementation `sleep`: informing the scheduler to move the thread to IDLE state.
 *  - availability of `terminate`: which can be called mid-air to shred the pool.
 *
 *  When not to use this thread-pool?
 *  - don't use outside of Linux or in UMA (Uniform Memory Access) systems.
 *  - don't use if you just need to pin everything to a single NUMA node,
 *    for that: `numactl --cpunodebind=2 --membind=2 your_program`
 *
 *  How to best leverage this thread-pool?
 *  - use in conjunction with @b `linux_numa_allocator` to pin memory to the same NUMA node.
 *  - make sure the Linux kernel is built with @b `CONFIG_SCHED_IDLE` support.
 *  - avoid recreating the @b `numa_topology`, as it's expensive to harvest.
 */
template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
struct linux_colocated_pool {

  public:
    using allocator_t = linux_numa_allocator_t;
    using micro_yield_t = micro_yield_type_;
    static_assert(std::is_nothrow_invocable_r<void, micro_yield_t>::value,
                  "Yield must be callable w/out arguments & return void");
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(alignment_k > 0 && (alignment_k & (alignment_k - 1)) == 0, "Alignment must be a power of 2");

    using index_t = std::size_t;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using epoch_index_t = index_t;  // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using thread_index_t = index_t; // ? A.k.a. "core index" or "thread ID" in [0, threads_count)

    using punned_fork_context_t = void const *;                           // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, thread_index_t); // ? Wraps lambda's `operator()`

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using numa_pthread_allocator_t = typename allocator_traits_t::template rebind_alloc<numa_pthread_t>;

    // Thread-pool-specific variables:
    allocator_t allocator_ {};

    /**
     *  Differs from STL `workers_` in base in type and size, as it may contain the `pthread_self`
     *  at the first position. If the @b `numa_pin_to_core_k` granularity is used, the `numa_pthread_t::core_id`
     *  will be set to the individual core IDs.
     */
    numa_pthread_t *pthreads_ {nullptr};
    thread_index_t pthreads_count_ {0};

    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count
    std::size_t sleep_length_micros_ {0}; // ? How long to sleep in microseconds when waiting for tasks

    using char16_name_t = char[16];    // ? Fixed-size thread name buffer, for POSIX thread naming
    char16_name_t name_ {};            // ? Thread name buffer, for POSIX thread naming
    numa_node_id_t numa_node_id_ {-1}; // ? Unique NUMA node ID, in [0, numa_max_node())
    numa_pin_granularity_t pin_granularity_ {numa_pin_to_core_k};

    alignas(alignment_) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    punned_fork_context_t fork_state_ {nullptr}; // ? Pointer to the users lambda
    trampoline_t fork_trampoline_ {nullptr};     // ? Calls the lambda
    alignas(alignment_) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_) std::atomic<epoch_index_t> epoch_ {0};

  public:
    linux_colocated_pool(linux_colocated_pool &&) = delete;
    linux_colocated_pool(linux_colocated_pool const &) = delete;
    linux_colocated_pool &operator=(linux_colocated_pool &&) = delete;
    linux_colocated_pool &operator=(linux_colocated_pool const &) = delete;

    explicit linux_colocated_pool(char const *name = "fork_union") noexcept {
        assert(name && "Thread name must not be null");
        if (std::strlen(name_) == 0) { std::strncpy(name_, "fork_union", sizeof(name_) - 1); } // ? Default name
        else { std::strncpy(name_, name, sizeof(name_) - 1), name_[sizeof(name_) - 1] = '\0'; }
    }

    ~linux_colocated_pool() noexcept { terminate(); }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept {
        return sizeof(linux_colocated_pool) + threads_count() * sizeof(numa_pthread_t);
    }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return mood_.is_lock_free() && threads_to_sync_.is_lock_free(); }

#pragma region Core API

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized.
     */
    thread_index_t threads_count() const noexcept { return pthreads_count_; }

    /**
     *  @brief Reports if the current calling thread will be used for broadcasts.
     *  @note This API is @b not synchronized.
     */
    caller_exclusivity_t caller_exclusivity() const noexcept { return exclusivity_; }

    /**
     *  @brief Creates a thread-pool addressing all cores on the given NUMA @p node.
     *  @param[in] node Describes the NUMA node to use, with its ID, memory size, and core IDs.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     *  @sa Other overloads of `try_spawn` that allow to specify the number of threads.
     */
    bool try_spawn(numa_node_t const node, caller_exclusivity_t const exclusivity = caller_inclusive_k) noexcept {
        return try_spawn(node, node.core_count, exclusivity);
    }

    /**
     *  @brief Creates a thread-pool with the given number of @p threads on the given NUMA @p node.
     *  @param[in] node Describes the NUMA node to use, with its ID, memory size, and core IDs.
     *  @param[in] threads The number of threads to be used.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @param[in] pin_granularity How to pin the threads to the NUMA node?
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     *
     *  @section Over- and Under-subscribing Cores and Pinning
     *
     *  We may accept @p threads different from the @p node.core_count, which allows us to:
     *  - over-subscribe the cores, i.e. use more threads than cores available on the NUMA node.
     *  - under-subscribe the cores, i.e. use fewer threads than cores available on the NUMA node.
     *
     *  If you only have one thread-pool active at any part of your application, that's meaningless.
     *  You'd be better off using exactly the number of cores available on the NUMA node and pinning
     *  them to individual cores with @b `numa_pin_to_core_k` granularity.
     */
    bool try_spawn(numa_node_t const node, thread_index_t const threads,
                   caller_exclusivity_t const exclusivity = caller_inclusive_k,
                   numa_pin_granularity_t const pin_granularity = numa_pin_to_core_k) noexcept {

        if (threads == 0) return false;         // ! Can't have zero threads working on something
        if (pthreads_count_ != 0) return false; // ! Already initialized

        // Allocate the thread pool of `numa_pthread_t` objects
        allocator_ = linux_numa_allocator_t {node.node_id};
        numa_pthread_allocator_t pthread_allocator {allocator_};
        numa_pthread_t *const pthreads = pthread_allocator.allocate(threads);
        if (!pthreads) {
            pthread_allocator.deallocate(pthreads, threads);
            return false; // ! Allocation failed
        }

        // Allocate the `cpu_set_t` structure, assuming we may be on a machine
        // with a ridiculously large number of cores.
        int const max_possible_cores = ::numa_num_possible_cpus();
        cpu_set_t *cpu_set_ptr = CPU_ALLOC(max_possible_cores);

        // Before we start the threads, make sure we set some of the shared
        // state variables that will be used in the `_posix_worker_loop` function.
        pthreads_ = pthreads;
        pthreads_count_ = threads;
        exclusivity_ = exclusivity;
        numa_node_id_ = node.node_id;
        pin_granularity_ = pin_granularity;
        auto reset_on_failure = [&]() noexcept {
            pthread_allocator.deallocate(pthreads, threads);
            pthreads_ = nullptr;
            pthreads_count_ = 0;
            numa_node_id_ = -1;
            pin_granularity_ = numa_pin_to_core_k;
        };

        // Include the main thread into the list of handles
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (use_caller_thread) {
            pthreads_[0].handle.store(::pthread_self(), std::memory_order_release);
            pthreads_[0].id.store(::gettid(), std::memory_order_release);
        }

        // The startup sequence for the POSIX threads differs from the `basic_pool`,
        // where at start up there is a race condition to read the `pthreads_`.
        // So we mark the threads as "chilling" until the
        mood_.store(mood_t::chill_k, std::memory_order_release);

        // Initializing the thread pool can fail for all kinds of reasons, like:
        // - `EAGAIN` if we reach the `RLIMIT_NPROC` soft resource limit.
        // - `EINVAL` if an invalid attribute was specified.
        // - `EPERM` if we don't have the right permissions.
        for (thread_index_t i = use_caller_thread; i < threads; ++i) {

            pthread_t pthread_handle;
            int creation_result = ::pthread_create(&pthread_handle, NULL, &_posix_worker_loop, this);
            pthreads_[i].handle.store(pthread_handle, std::memory_order_relaxed);
            pthreads_[i].id.store(-1, std::memory_order_relaxed);
            pthreads_[i].core_id = -1; // ? Not pinned yet

            if (creation_result != 0) {
                mood_.store(mood_t::die_k, std::memory_order_release);
                for (thread_index_t j = use_caller_thread; j < i; ++j) {
                    pthread_t pthread_handle = pthreads_[j].handle.load(std::memory_order_relaxed);
                    int cancel_result = ::pthread_cancel(pthread_handle);
                    assert(cancel_result == 0 && "Failed to cancel a thread");
                }
                reset_on_failure();
                CPU_FREE(cpu_set_ptr);
                return false; // ! Thread creation failed
            }
        }

        // Name all of the threads
        char16_name_t name;
        for (thread_index_t i = 0; i < pthreads_count_; ++i) {
            fill_thread_name(                                    //
                name, name_,                                     //
                static_cast<std::size_t>(node.first_core_id[i]), //
                static_cast<std::size_t>(max_possible_cores));
            pthread_t pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
            int naming_result = ::pthread_setname_np(pthread_handle, name);
            assert(naming_result == 0 && "Failed to name a thread");
        }

        // Pin all of the threads
        std::size_t const cpu_set_size = CPU_ALLOC_SIZE(max_possible_cores);
        if (pin_granularity == numa_pin_to_core_k) {
            // Configure a mask for each thread, pinning it to a specific core
            for (thread_index_t i = 0; i < pthreads_count_; ++i) {
                // Assign to a core in a round-robin fashion
                numa_core_id_t cpu = node.first_core_id[i % node.core_count];
                assert(cpu >= 0 && "Invalid CPU core ID");
                CPU_ZERO(cpu_set_ptr);
                CPU_SET(cpu, cpu_set_ptr);

                // Assign the mask to the thread
                pthread_t pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
                int pin_result = ::pthread_setaffinity_np(pthread_handle, cpu_set_size, cpu_set_ptr);
                assert(pin_result == 0 && "Failed to pin a thread to a NUMA node");
                pthreads_[i].core_id = cpu;
            }
        }
        else {
            // Configure one mask that will be shared by all threads
            CPU_ZERO(cpu_set_ptr);
            for (std::size_t i = 0; i < node.core_count; ++i) {
                numa_core_id_t cpu = node.first_core_id[i];
                assert(cpu >= 0 && "Invalid CPU core ID");
                CPU_SET(cpu, cpu_set_ptr);
            }
            assert(static_cast<std::size_t>(CPU_COUNT(cpu_set_ptr)) == node.core_count &&
                   "The CPU set must match the number of cores in the NUMA node");

            // Assign the same mask to all threads
            for (thread_index_t i = 0; i < pthreads_count_; ++i) {
                pthread_t pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
                int pin_result = ::pthread_setaffinity_np(pthread_handle, cpu_set_size, cpu_set_ptr);
                assert(pin_result == 0 && "Failed to pin a thread to a NUMA node");
            }
        }

        // If all went well, we can store the thread-pool and start using it
        CPU_FREE(cpu_set_ptr); // ? Clean up the CPU set
        mood_.store(mood_t::grind_k, std::memory_order_release);
        return true;
    }

    /**
     *  @brief Executes a @p function in parallel on all threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     *  @return A `broadcast_join` synchronization point that waits in the destructor.
     *  @note Even in the `caller_exclusive_k` mode, can be called from just one thread!
     *  @sa For advanced resource management, consider `unsafe_broadcast` and `unsafe_join`.
     */
    template <typename function_type_>
    broadcast_join<linux_colocated_pool, function_type_> for_threads(function_type_ &&function) noexcept {
        broadcast_join<linux_colocated_pool, function_type_> joiner {*this, std::forward<function_type_>(function)};
        unsafe_for_threads(joiner.function());
        return joiner;
    }

    /**
     *  @brief Executes a @p function in parallel on all threads, not waiting for the result.
     *  @param[in] function The callback, receiving the thread index as an argument.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename function_type_>
    void unsafe_for_threads(function_type_ const &function) noexcept {

        thread_index_t const threads = threads_count();
        assert(threads != 0 && "Thread pool not initialized");
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (threads == 1 && use_caller_thread) return function(static_cast<thread_index_t>(0));

        // Optional check: even in exclusive mode, only one thread can call this function.
        assert((use_caller_thread || threads_to_sync_.load(std::memory_order_acquire) == 0) &&
               "The broadcast function can be called only from one thread at a time");

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        // ? we live to the higher-level algorithms.
        static_assert(std::is_nothrow_invocable_r<void, function_type_, thread_index_t>::value,
                      "The callback must be invocable with a `thread_index_t` argument");
#endif

        // Configure "fork" details
        fork_state_ = std::addressof(function);
        fork_trampoline_ = &_call_as_lambda<function_type_>;
        threads_to_sync_.store(threads - use_caller_thread, std::memory_order_relaxed);

        // We are most likely already "grinding", but in the unlikely case we are not,
        // let's wake up from the "chilling" state with relaxed semantics. Assuming the sleeping
        // logic for the workers also checks the epoch counter, no synchronization is needed and
        // no immediate wake-up is required.
        mood_t may_be_chilling = mood_t::chill_k;
        bool const was_chilling = mood_.compare_exchange_weak( //
            may_be_chilling, mood_t::grind_k,                  //
            std::memory_order_relaxed, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // If the workers were indeed "chilling", we can inform the scheduler to wake them up.
        if (was_chilling) {
            for (std::size_t i = use_caller_thread; i < pthreads_count_; ++i) {
                pid_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
                if (pthread_id < 0) continue; // ? Not set yet
                sched_param param {};
                ::sched_setscheduler(pthread_id, SCHED_FIFO | SCHED_RR, &param);
            }
        }

        // Execute on the current "main" thread
        if (use_caller_thread) function(static_cast<thread_index_t>(0));
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        micro_yield_t micro_yield;
        while (threads_to_sync_.load(std::memory_order_acquire)) micro_yield();
    }

#pragma endregion Core API

#pragma region Control Flow

    /**
     *  @brief Stops all threads and deallocates the thread-pool after the last call finishes.
     *  @note Can be called from @b any thread at any time.
     *  @note Must `try_spawn` again to re-use the pool.
     *
     *  When and how @b NOT to use this function:
     *  - as a synchronization point between concurrent tasks.
     *
     *  When and how to use this function:
     *  - as a de-facto @b destructor, to stop all threads and deallocate the pool.
     *  - when you want to @b restart with a different number of threads.
     */
    void terminate() noexcept {
        assert(threads_to_sync_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running
        if (pthreads_count_ == 0) return;                              // ? Uninitialized

        numa_pthread_allocator_t pthread_allocator {allocator_};

        // Stop all threads and wait for them to finish
        mood_.store(mood_t::die_k, std::memory_order_release);

        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        thread_index_t const threads = pthreads_count_;
        for (thread_index_t i = use_caller_thread; i != threads; ++i) {
            void *returned_value = nullptr;
            pthread_t const pthread_handle = pthreads_[i].handle.load(std::memory_order_relaxed);
            int const join_result = ::pthread_join(pthread_handle, &returned_value);
            assert(join_result == 0 && "Thread join failed");
        }

        // Deallocate the handles and IDs
        pthread_allocator.deallocate(pthreads_, threads);

        // Prepare for future spawns
        pthreads_count_ = 0;
        pthreads_ = nullptr;

        // Unpin the caller thread if it was part of this pool and was pinned to the NUMA node.
        if (use_caller_thread) _reset_affinity();
        _reset_fork();

        mood_.store(mood_t::grind_k, std::memory_order_relaxed);
        epoch_.store(0, std::memory_order_relaxed);
    }

    /**
     *  @brief Transitions "workers" to a sleeping state, waiting for a wake-up call.
     *  @param[in] wake_up_periodicity_micros How often to check for new work in microseconds.
     *  @note Can only be called @b between the tasks for a single thread. No synchronization is performed.
     *
     *  This function may be used in some batch-processing operations when we clearly understand
     *  that the next task won't be arriving for a while and power can be saved without major
     *  latency penalties.
     *
     *  It may also be used in a high-level Python or JavaScript library offloading some parallel
     *  operations to an underlying C++ engine, where latency is irrelevant.
     */
    void sleep(std::size_t wake_up_periodicity_micros) noexcept {
        assert(wake_up_periodicity_micros > 0 && "Sleep length must be positive");
        sleep_length_micros_ = wake_up_periodicity_micros;
        mood_.store(mood_t::chill_k, std::memory_order_release);

        // On Linux we can update the thread's scheduling class to IDLE,
        // which will reduce the power consumption:
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        for (std::size_t i = use_caller_thread; i < pthreads_count_; ++i) {
            pid_t const pthread_id = pthreads_[i].id.load(std::memory_order_acquire);
            if (pthread_id < 0) continue; // ? Not set yet
            sched_param param {};
            ::sched_setscheduler(pthread_id, SCHED_IDLE, &param);
        }
    }

#pragma endregion Control Flow

#pragma region Indexed Task Scheduling

#pragma endregion Indexed Task Scheduling

#pragma region Colocations Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     *  @retval 1 constant for compatibility.
     */
    constexpr index_t colocations_count() const noexcept { return 1; }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b colocation.
     *  @return Same value as `threads_count()`, as we only support one colocation.
     */
    thread_index_t threads_count(index_t colocation_index) const noexcept {
        assert(colocation_index == 0 && "Only one colocation is supported");
        return count_threads();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b colocation.
     *  @return Same value as `global_thread_index`, as we only support one colocation.
     */
    constexpr thread_index_t thread_local_index(thread_index_t global_thread_index,
                                                index_t colocation_index) const noexcept {
        assert(colocation_index == 0 && "Only one colocation is supported");
        return global_thread_index;
    }

#pragma endregion Colocations Compatibility

  private:
    void _reset_fork() noexcept {
        fork_state_ = nullptr;
        fork_trampoline_ = nullptr;
    }

    void _reset_affinity() noexcept {
        int const max_possible_cores = ::numa_num_possible_cpus();
        cpu_set_t *cpu_set_ptr = CPU_ALLOC(max_possible_cores);
        if (!cpu_set_ptr) return;
        CPU_ZERO(cpu_set_ptr);
        for (int cpu = 0; cpu < max_possible_cores; ++cpu) CPU_SET(cpu, cpu_set_ptr);
        int pin_result = ::pthread_setaffinity_np(::pthread_self(), CPU_ALLOC_SIZE(max_possible_cores), cpu_set_ptr);
        assert(pin_result == 0 && "Failed to reset the caller thread's affinity");
        CPU_FREE(cpu_set_ptr);
        int spread_result = ::numa_run_on_node(-1); // !? Shouldn't it be `numa_all_nodes`
        assert(spread_result == 0 && "Failed to reset the caller thread's NUMA node affinity");
    }

    /**
     *  @brief A trampoline function that is used to call the user-defined lambda.
     *  @param[in] punned_lambda_pointer The pointer to the user-defined lambda.
     *  @param[in] prong The index of the thread & task index packed together.
     */
    template <typename function_type_>
    static void _call_as_lambda(punned_fork_context_t punned_lambda_pointer, thread_index_t thread_index) noexcept {
        function_type_ const &lambda_object = *static_cast<function_type_ const *>(punned_lambda_pointer);
        lambda_object(thread_index);
    }

    static void *_posix_worker_loop(void *arg) noexcept {
        linux_colocated_pool *pool = static_cast<linux_colocated_pool *>(arg);

        // Following section untile the main `while` loop may introduce race conditions,
        // so spin-loop for a bit until the pool is ready.
        mood_t mood;
        micro_yield_t micro_yield;
        while ((mood = pool->mood_.load(std::memory_order_acquire)) == mood_t::chill_k) micro_yield();

        // If we are ready to start grinding, export this threads metadata to make it externally
        // observable and controllable.
        thread_index_t thread_index = -1;
        if (mood == mood_t::grind_k) {
            // We locate the thread index by enumerating the `pthreads_` array
            numa_pthread_t *const numa_pthreads = pool->pthreads_;
            thread_index_t const numa_pthreads_count = pool->pthreads_count_;
            pthread_t const thread_handle = ::pthread_self();
            for (thread_index = 0; thread_index < numa_pthreads_count; ++thread_index)
                if (::pthread_equal(numa_pthreads[thread_index].handle.load(std::memory_order_relaxed), thread_handle))
                    break;
            assert(thread_index < numa_pthreads_count && "Thread index must be in [0, threads_count)");

            // Assign the pthread ID to the shared memory
            pid_t const pthread_id = ::gettid();
            numa_pthreads[thread_index].id.store(pthread_id, std::memory_order_release);

            // Ensure this function isn't used by the main caller
            caller_exclusivity_t const exclusivity = pool->caller_exclusivity();
            bool const use_caller_thread = exclusivity == caller_inclusive_k;
            if (use_caller_thread) assert(thread_index != 0 && "The zero index is for the main thread, not worker!");
        }

        // Run the infinite loop, using Linux-specific napping mechanism
        epoch_index_t last_epoch = 0;
        epoch_index_t new_epoch;
        while (true) {
            // Wait for either: a new ticket or a stop flag
            while ((new_epoch = pool->epoch_.load(std::memory_order_acquire)) == last_epoch &&
                   (mood = pool->mood_.load(std::memory_order_acquire)) == mood_t::grind_k)
                micro_yield();

            if (_FU_UNLIKELY(mood == mood_t::die_k)) break;
            if (_FU_UNLIKELY(mood == mood_t::chill_k) && (new_epoch == last_epoch)) {
                struct timespec ts {0, static_cast<long>(pool->sleep_length_micros_ * 1000)};
                ::clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
                continue;
            }

            pool->fork_trampoline_(pool->fork_state_, thread_index);
            last_epoch = new_epoch;

            // ! The decrement must come after the task is executed
            _FU_MAYBE_UNUSED thread_index_t const before_decrement =
                pool->threads_to_sync_.fetch_sub(1, std::memory_order_release);
            assert(before_decrement > 0 && "We can't be here if there are no worker threads");
        }

        return nullptr;
    }

    static void fill_thread_name(                          //
        char16_name_t &output_name, char const *base_name, //
        std::size_t const index, std::size_t const max_possible_cores) noexcept {

        constexpr std::size_t max_visible_chars = sizeof(char16_name_t) - 1; // room left after the terminator
        int const digits = max_possible_cores < 10      ? 1
                           : max_possible_cores < 100   ? 2
                           : max_possible_cores < 1000  ? 3
                           : max_possible_cores < 10000 ? 4
                                                        : 0; // fall‑through – let snprintf clip

        if (digits == 0) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
            //  "%s:%zu" - worst‑case  (base up to 11 chars) + ":" + up‑to‑2‑digit index
            std::snprintf(&output_name[0], sizeof(char16_name_t), "%s:%zu", base_name, index + 1);
#pragma GCC diagnostic pop
        }
        else {
            int const base_len = static_cast<int>(max_visible_chars - digits - 1); // -1 for ':'
            // "%.*s" - truncates base_name to base_len
            // "%0*zu" - prints zero‑padded index using exactly 'digits' characters
            std::snprintf(&output_name[0], sizeof(char16_name_t), "%.*s:%0*zu", base_len, base_name, digits, index + 1);
        }
    }
};

#pragma endregion - Linux Colocated Pool

#pragma region - Linux Pool

/**
 *  @brief A Linux-only thread-pool addressing all colocated "thread islands", NUMA nodes, and QoS levels.
 *
 *  Differs from the `basic_pool` template in the following ways:
 *  - constructor API: receives the NUMA nodes topology, & a name for threads.
 *  - implementation of `try_spawn`: redirects to individual `linux_colocated_pool` instances.
 *
 *  Many of the parallel ops benefit from having some minimal amount of @b "scratch-space" that
 *  can be used as an output buffer for partial results, before they can be aggregated from the
 *  calling thread. Reductions are a great example, and allocating a new buffer for each thread
 *  on each call is quite wasteful, so we always keep some around.
 *
 *  This thread-pool doesn't (yet) provide "reductions" or other reach operations, but uses a
 *  small pool of NUMA-local memory to dampen the cost of `for_n_dynamic` scheduling.
 */
template <typename micro_yield_type_ = standard_yield_t, std::size_t alignment_ = default_alignment_k>
struct linux_pool {

    using linux_colocated_pool_t = linux_colocated_pool<micro_yield_type_, alignment_>;
    using numa_topology_t = numa_topology<>;

    using allocator_t = linux_numa_allocator_t;
    using micro_yield_t = typename linux_colocated_pool_t::micro_yield_t;
    using index_t = typename linux_colocated_pool_t::index_t;
    using epoch_index_t = typename linux_colocated_pool_t::epoch_index_t;
    using thread_index_t = typename linux_colocated_pool_t::thread_index_t;
    static constexpr std::size_t alignment_k = linux_colocated_pool_t::alignment_k;

  private:
    numa_topology_t topology_ {};
    char name_[16] {}; // ? Thread name buffer, for POSIX thread naming
    thread_index_t threads_count_ {0};
    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count

    struct local_pool_t {
        alignas(alignment_k) linux_colocated_pool_t pool {};
        thread_index_t first_thread_index {0};                         // ? The first thread index in this local pool
        alignas(alignment_k) std::atomic<index_t> dynamic_counter {0}; // ? Used for `for_n_dynamic` optimization
    };

    /**
     *  A heap allocated array of individual thread pools. Each one of them is allocated on its own NUMA node.
     *  They are sorted/partitioned in such a way that the first one @b always contains the current thread.
     *  Moreover, assuming NUMA allocators operate at @b page-granularity, each "local pool" comes with its
     *  own "scratch space" arena, that can be used for temporary allocations.
     */
    local_pool_t **local_pools_ {nullptr}; // ? Array of thread pools for each NUMA node
    std::size_t local_pools_count_ {0};    // ? Number of NUMA nodes in the topology

    using local_pool_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<local_pool_t>;
    using local_pools_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<local_pool_t *>;

  public:
    linux_pool(linux_pool &&) = delete;
    linux_pool(linux_pool const &) = delete;
    linux_pool &operator=(linux_pool &&) = delete;
    linux_pool &operator=(linux_pool const &) = delete;

    linux_pool(numa_topology_t topo) noexcept : linux_pool("fork_union", topo) {}
    linux_pool(char const *name, numa_topology_t topo) noexcept : topology_(topo) {
        assert(name && "Thread name must not be null");
        if (std::strlen(name_) == 0) { std::strncpy(name_, "fork_union", sizeof(name_) - 1); } // ? Default name
        else { std::strncpy(name_, name, sizeof(name_) - 1), name_[sizeof(name_) - 1] = '\0'; }
    }

    ~linux_pool() noexcept { terminate(); }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return local_pools_ && local_pools_[0] && local_pools_[0]->is_lock_free(); }

    /**
     *  @brief Returns the NUMA topology used by this thread-pool.
     *  @note This API is @b not synchronized.
     */
    numa_topology_t const &topology() const noexcept { return topology_; }

    /**
     *  @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept {
        std::size_t total_bytes = sizeof(linux_pool);
        for (std::size_t i = 0; i < local_pools_count_; ++i) {
            linux_colocated_pool_t *pool = local_pools_[i];
            assert(pool && "NUMA thread pool must not be null");
            total_bytes += pool->memory_usage();
        }
        return total_bytes;
    }

#pragma region Core API

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized.
     */
    thread_index_t threads_count() const noexcept { return threads_count_; }

    /**
     *  @brief Reports if the current calling thread will be used for broadcasts.
     *  @note This API is @b not synchronized.
     */
    caller_exclusivity_t caller_exclusivity() const noexcept { return exclusivity_; }

    /**
     *  @brief Creates a thread-pool addressing all cores across all NUMA nodes.
     *  @param[in] threads The number of threads to be used.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @retval false if the number of threads is zero or if spawning has failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     */
    bool try_spawn(                   //
        thread_index_t const threads, //
        caller_exclusivity_t const exclusivity = caller_inclusive_k) noexcept {

        if (threads == 0) return false;        // ! Can't have zero threads working on something
        if (threads_count_ != 0) return false; // ! Already initialized

        // We are going to place the control structures on the first NUMA node,
        // and pin the caller thread to it as well.
        numa_node_t const &first_node = topology_.node(0);
        numa_node_index_t const first_node_id = first_node.node_id; // ? Typically zero
        linux_numa_allocator_t allocator {first_node_id};
        local_pools_allocator_t local_pools_allocator {allocator};
        thread_index_t const nodes_count = std::min(topology_.nodes_count(), threads);

        local_pool_t **local_pools = local_pools_allocator.allocate(nodes_count);
        if (!local_pools) return false; // ! Allocation failed

        // Now allocate each "local pool" on its own NUMA node
        std::fill_n(local_pools, local_pools + nodes_count, nullptr);
        for (numa_node_index_t numa_index = 0; numa_index < nodes_count; ++numa_index) {
            numa_node_t const &node = topology_.node(numa_index);
            numa_node_id_t const node_id = node.node_id;
            local_pool_allocator_t allocator {node_id};
            local_pool_t *pool = allocator.allocate(1);
            local_pools[numa_index] = pool;
        }

        // If any one of the allocations failed, we need to clean up
        auto reset_on_failure = [&]() noexcept {
            for (numa_node_index_t numa_index = 0; numa_index < nodes_count; ++numa_index) {
                local_pool_t *pool = local_pools[numa_index];
                if (!pool) continue;
                pool->terminate(); // ? Stop the pool if it was started
                numa_node_t const &node = topology_.node(numa_index);
                numa_node_id_t const node_id = node.node_id;
                local_pool_allocator_t allocator {node_id};
                allocator.deallocate(pool, 1); // ? Reclaim the memory
            }
            local_pools_allocator.deallocate(local_pools, nodes_count);
        };
        if (std::any_of(local_pools, local_pools + nodes_count, [](local_pool_t *pool) { return !pool; })) {
            reset_on_failure();
            return false; // ! Allocation failed
        }

        // Every NUMA pool is allocated separately
        // - the first one may be "inclusive".
        // - others are always "exclusive" to the caller thread.
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        thread_index_t const threads_per_node = (threads - use_caller_thread) / nodes_count;
        if (!local_pools[0]->pool.try_spawn(first_node, threads_per_node, exclusivity)) {
            reset_on_failure();
            return false; // ! Spawning failed
        }

        for (numa_node_index_t numa_index = 1; numa_index < nodes_count; ++numa_index) {
            numa_node_t const &node = topology_.node(numa_index);
            local_pool_t *local_pool = local_pools[numa_index];
            if (!local_pool->pool.try_spawn(node, threads_per_node, caller_exclusive_k)) {
                reset_on_failure();
                return false; // ! Spawning failed
            }
        }

        local_pools_ = local_pools;
        local_pools_count_ = nodes_count;
        threads_count_ = threads;
        exclusivity_ = exclusivity;
        return true;
    }

    /**
     *  @brief Executes a @p function in parallel on all threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     *  @return A `broadcast_join` synchronization point that waits in the destructor.
     *  @note Even in the `caller_exclusive_k` mode, can be called from just one thread!
     *  @sa For advanced resource management, consider `unsafe_broadcast` and `unsafe_join`.
     */
    template <typename function_type_>
    broadcast_join<linux_pool, function_type_> for_threads(function_type_ &&function) noexcept {
        broadcast_join<linux_pool, function_type_> joiner {*this, std::forward<function_type_>(function)};
        unsafe_for_threads(joiner.function());
        return joiner;
    }

    /**
     *  @brief Executes a @p function in parallel on all threads, not waiting for the result.
     *  @param[in] function The callback, receiving the thread index as an argument.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename function_type_>
    void unsafe_for_threads(function_type_ const &function) noexcept {
        assert(local_pools_ && "Thread pools must be initialized before broadcasting");

        // Submit to every thread pool
        for (std::size_t i = 1; i < local_pools_count_; ++i) {
            linux_colocated_pool_t *pool = local_pools_[i];
            assert(pool && "NUMA thread pool must not be null");
            pool->unsafe_for_threads(function);
        }
        local_pools_[0]->unsafe_for_threads(function);
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    void unsafe_join() noexcept {
        assert(local_pools_ && "Thread pools must be initialized before broadcasting");

        // Wait for everyone to finish
        for (std::size_t i = 1; i < local_pools_count_; ++i) {
            linux_colocated_pool_t *pool = local_pools_[i];
            assert(pool && "NUMA thread pool must not be null");
            pool->unsafe_join();
        }
        local_pools_[0]->unsafe_join();
    }

#pragma endregion Core API

#pragma region Control Flow

    /**
     *  @brief Stops all threads and deallocates the thread-pool after the last call finishes.
     *  @note Can be called from @b any thread at any time.
     *  @note Must `try_spawn` again to re-use the pool.
     *
     *  When and how @b NOT to use this function:
     *  - as a synchronization point between concurrent tasks.
     *
     *  When and how to use this function:
     *  - as a de-facto @b destructor, to stop all threads and deallocate the pool.
     *  - when you want to @b restart with a different number of threads.
     */
    void terminate() noexcept {
        if (local_pools_ == nullptr) return; // ? Uninitialized
        for (std::size_t i = 0; i < local_pools_count_; ++i) {
            linux_colocated_pool_t *pool = local_pools_[i];
            assert(pool && "NUMA thread pool must not be null");
            pool->terminate();
        }
    }

    /**
     *  @brief Transitions "workers" to a sleeping state, waiting for a wake-up call.
     *  @param[in] wake_up_periodicity_micros How often to check for new work in microseconds.
     *  @note Can only be called @b between the tasks for a single thread. No synchronization is performed.
     *
     *  This function may be used in some batch-processing operations when we clearly understand
     *  that the next task won't be arriving for a while and power can be saved without major
     *  latency penalties.
     *
     *  It may also be used in a high-level Python or JavaScript library offloading some parallel
     *  operations to an underlying C++ engine, where latency is irrelevant.
     */
    void sleep(std::size_t wake_up_periodicity_micros) noexcept {
        assert(wake_up_periodicity_micros > 0 && "Sleep length must be positive");
        for (std::size_t i = 0; i < local_pools_count_; ++i) {
            linux_colocated_pool_t *pool = local_pools_[i];
            assert(pool && "NUMA thread pool must not be null");
            pool->sleep(wake_up_periodicity_micros);
        }
    }

#pragma endregion Control Flow

#pragma region Indexed Task Scheduling

    /**
     *  @brief Distributes @p (n) similar duration calls between threads in slices, as opposed to individual indices.
     *  @param[in] n The total length of the range to split between threads.
     *  @param[in] function The callback, receiving @b `prong_t` or an unsigned integer and the slice length.
     */
    template <typename function_type_ = dummy_lambda_t>
#if _FU_DETECT_CPP_20
        requires( // ? The callback must be invocable with a `prong_t` or a `index_t` argument and an unsigned counter
            std::is_nothrow_invocable_r_v<void, function_type_, prong_t, index_t> ||
            std::is_nothrow_invocable_r_v<void, function_type_, index_t, index_t>)
#endif
    auto for_slices(index_t const n, function_type_ &&function) noexcept {

        // We we will end up with 2 levels of splitting,
        // - one explicit level for colocations,
        // - one implicit level for their inner colocated threads.
        thread_index_t const threads = threads_count();
        indexed_split_t const splits {n, threads};
        return for_threads([splits, function](thread_index_t const thread_index) noexcept {
            auto const range = splits[thread_index];
            if (range.count == 0) return; // ? No work for this thread
            function(prong_t {range.first, thread_index}, range.count);
        });
    }

    /**
     *  @brief Distributes @p (n) similar duration calls between threads.
     *  @param[in] n The number of times to call the @p function.
     *  @param[in] function The callback, receiving @b `prong_t` or a call index as an argument.
     *
     *  Is designed for a "balanced" workload, where all threads have roughly the same amount of work.
     *  @sa `for_n_dynamic` for a more dynamic workload.
     *  The @p function is called @p (n) times, and each thread receives a slice of consecutive tasks.
     *  @sa `for_slices` if you prefer to receive workload slices over individual indices.
     */
    template <typename function_type_ = dummy_lambda_t>
    auto for_n(index_t const n, function_type_ &&function) noexcept {
        return for_slices(pool, n, [function](prong_t const start_prong, index_t const count_prongs) noexcept {
            for (index_t i = 0; i < count_prongs; ++i)
                function(prong_t {static_cast<index_t>(start_prong.task + i), start_prong.thread});
        });
    }

#pragma endregion Indexed Task Scheduling

#pragma region Colocations Compatibility

    /**
     *  @brief Number of individual sub-pool with the same NUMA-locality and QoS.
     */
    index_t colocations_count() const noexcept { return local_pools_count_; }

    /**
     *  @brief Returns the number of threads in one NUMA-specific local @b colocation.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized and doesn't check for out-of-bounds access.
     */
    thread_index_t threads_count(index_t colocation) const noexcept {
        assert(local_pools_ && "Local pools must be initialized");
        assert(colocation < local_pools_count_ && "Local pool index out of bounds");
        return local_pools_[colocation]->pool.threads_count();
    }

    /**
     *  @brief Converts a @p `global_thread_index` to a local thread index within a @b colocation.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is @b not synchronized and doesn't check for out-of-bounds access.
     */
    thread_index_t thread_local_index(thread_index_t global_thread_index, index_t colocation) const noexcept {
        assert(local_pools_ && "Local pools must be initialized");
        assert(colocation < local_pools_count_ && "Local pool index out of bounds");
        return global_thread_index - local_pools_[colocation]->first_thread_index;
    }

#pragma endregion Colocations Compatibility
};

using linux_colocated_pool_t = linux_colocated_pool<>;
using linux_pool_t = linux_pool<>;

static_assert(is_unsafe_pool<basic_pool_t> && is_unsafe_pool<linux_colocated_pool_t>,
              "These thread pools must be flexible and support unsafe operations");
static_assert(is_pool<basic_pool_t> && is_pool<linux_colocated_pool_t> && is_pool<linux_pool_t>,
              "These thread pools must be fully compatible with the high-level APIs");

#endif // FU_ENABLE_NUMA
#pragma endregion - NUMA Pools

} // namespace fork_union
} // namespace ashvardanian
