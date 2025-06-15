/**
 *  @brief  OpenMP-style cross-platform fine-grained parallelism library.
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
 *  #include <fork_union.hpp> // `fu::thread_pool_t`
 *
 *  using fu = ashvardanian::fork_union;
 *  int main(int argc, char *argv[]) {
 *
 *      fu::thread_pool_t pool;
 *      if (!pool.try_spawn(std::thread::hardware_concurrency()))
 *          return EXIT_FAILURE;
 *
 *      fu::for_n(pool, argc, [](auto prong) noexcept {
 *          auto [thread_index, task_index] = prong;
 *          std::printf(
 *              "Printing argument %zu from thread %zu: %s\n",
 *              task_index, thread_index, argv[task_index]);
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
 */
#pragma once
#include <memory>  // `std::allocator`
#include <thread>  // `std::thread`
#include <atomic>  // `std::atomic`
#include <cstddef> // `std::max_align_t`
#include <cassert> // `assert`
#include <new>     // `std::hardware_destructive_interference_size`

#if !defined(FU_ALLOW_UNSAFE)
#define FU_ALLOW_UNSAFE 0
#endif

#if FU_ALLOW_UNSAFE
#include <exception> // `std::exception_ptr`
#endif

#define FORK_UNION_VERSION_MAJOR 1
#define FORK_UNION_VERSION_MINOR 0
#define FORK_UNION_VERSION_PATCH 6

/**
 *  On C++17 and later we can detect misuse of lambdas that are not properly annotated.
 *  On C++20 and later we can use concepts for cleaner compile-time checks.
 */
#define _FU_DETECT_CPP_20 (__cplusplus >= 202002L)
#define _FU_DETECT_CPP_17 (__cplusplus >= 201703L)
#if _FU_DETECT_CPP_17
#include <type_traits> // `std::is_nothrow_invocable_r`
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

namespace ashvardanian {
namespace fork_union {

/**
 *  @brief Defines variable alignment to avoid false sharing.
 *  @see https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_size
 *  @see https://docs.rs/crossbeam-utils/latest/crossbeam_utils/struct.CachePadded.html
 */
#if defined(__cpp_lib_hardware_interference_size)
static constexpr std::size_t default_alignment_k = std::hardware_destructive_interference_size;
#else
static constexpr std::size_t default_alignment_k = alignof(std::max_align_t);
#endif

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

#pragma region - Thread Pool

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
 *  This allows this thread-pool to be extremely lightweight and fast, with no heap allocations
 *  and no expensive abstractions. It only uses `std::thread` and `std::atomic`, but avoids
 *  `std::function`, `std::future`, `std::promise`, `std::condition_variable`, that bring
 *  unnecessary overhead.
 *  @see https://ashvardanian.com/posts/beyond-openmp-in-cpp-rust/#four-horsemen-of-performance
 *
 *  Repeated operations are performed with a "weak" memory model, to be able to leverage in-hardware
 *  support for atomic fence-less operations on Arm and IBM Power architectures. Most atomic counters
 *  use the "acquire-release" model, and some going further to "relaxed" model.
 *  @see https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
 *  @see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
 *
 *  A minimal example, similar to `#pragma omp parallel` in OpenMP:
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp>
 *
 *  using fu = ashvardanian::fork_union;
 *  int main() {
 *      fu::thread_pool<> pool; // ? Or `fu::thread_pool_t` alias
 *      if (!pool.try_spawn(std::thread::hardware_concurrency())) return EXIT_FAILURE;
 *      pool.broadcast([](std::size_t index) { std::printf("Hello from thread %zu\n", index); });
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  @param allocator_type_ The type of the allocator to be used for the thread pool.
 *  @param index_type_ Defaults to `std::size_t`, but can be changed to a smaller type for debugging.
 *  @param alignment_ The alignment of the thread pool. Defaults to `default_alignment_k`.
 */
template <                                                  //
    typename allocator_type_ = std::allocator<std::thread>, //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k            //
    >
class thread_pool {

  public:
    using allocator_t = allocator_type_;
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(alignment_k > 0 && (alignment_k & (alignment_k - 1)) == 0, "Alignment must be a power of 2");

    using index_t = index_type_;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using generation_index_t = index_t; // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using thread_index_t = index_t;     // ? A.k.a. "core index" or "thread ID" in [0, threads_count)

    using punned_fork_context_t = void const *;                           // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, thread_index_t); // ? Wraps lambda's `operator()`

  private:
    // Thread-pool-specific variables:
    allocator_t allocator_ {};
    std::thread *workers_ {nullptr};
    thread_index_t threads_count_ {0};

    /**
     *  Theoretically, the choice of `std::atomic<bool>` is suboptimal in the presence of `std::atomic_flag`.
     *  The latter is guaranteed to be lock-free, while the former is not. But until C++20, the flag doesn't
     *  have a non-modifying load operation - the `std::atomic_flag::test` was added in C++20.
     *  @see https://en.cppreference.com/w/cpp/atomic/atomic_flag.html
     */
    alignas(alignment_) std::atomic<bool> stop_ {false};

    // Task-specific variables:
    punned_fork_context_t fork_lambda_pointer_ {nullptr}; // ? Pointer to the users lambda
    trampoline_t fork_trampoline_pointer_ {nullptr};      // ? Calls the lambda
    alignas(alignment_) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_) std::atomic<generation_index_t> fork_generation_ {0};

  public:
    thread_pool(thread_pool &&) = delete;
    thread_pool(thread_pool const &) = delete;
    thread_pool &operator=(thread_pool &&) = delete;
    thread_pool &operator=(thread_pool const &) = delete;

    thread_pool(allocator_t const &alloc = {}) noexcept : allocator_(alloc) {}
    ~thread_pool() noexcept { stop_and_reset(); }

    /**
     *  @brief Returns the number of threads in the thread-pool, including the main thread.
     *  @retval 0 if the thread-pool is not initialized, 1 if only the main thread is used.
     *  @note This API is not synchronized and is expected to be called only from the main thread.
     */
    thread_index_t threads() const noexcept { return threads_count_; }

    /** @brief Estimates the amount of memory managed by this pool handle and internal structures. */
    std::size_t memory_usage() const noexcept { return sizeof(thread_pool) + threads() * sizeof(std::thread); }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept {
        return stop_.is_lock_free() && threads_to_sync_.is_lock_free() && fork_generation_.is_lock_free();
    }

    /**
     *  @brief Creates a thread-pool with the given number of threads.
     *  @note This is the de-facto @b constructor, and can only be called once.
     *  @param[in] planned_threads The number of threads to be used. Should be larger than one.
     *  @retval false if the number of threads is zero or the "workers" allocation failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     */
    bool try_spawn(thread_index_t const planned_threads) noexcept {
        if (planned_threads == 0) return false; // ! Can't have zero threads working on something
        if (threads_count_ != 0) return false;  // ! Already initialized
        if (planned_threads == 1) {
            threads_count_ = 1;
            return true; // ! The current thread will always be used
        }

        // Allocate the thread pool
        thread_index_t const worker_threads = planned_threads - 1;
        std::thread *const workers = allocator_.allocate(worker_threads);
        if (!workers) return false; // ! Allocation failed

        // Initialize the thread pool can fail for all kinds of reasons,
        // that the `std::thread` documentation describes as "implementation-defined".
        // https://en.cppreference.com/w/cpp/thread/thread/thread
        for (thread_index_t i = 0; i < worker_threads; ++i) {
            try {
                new (&workers[i]) std::thread([this, i] { _worker_loop(i + 1); });
            }
            catch (...) {
                for (thread_index_t j = 0; j < i; ++j) workers[j].~thread();
                allocator_.deallocate(workers, worker_threads);
                return false; // ! Thread creation failed
            }
        }

        // If all went well, we can store the thread-pool and start using it
        workers_ = workers;
        threads_count_ = planned_threads;
        return true;
    }

    /**
     *  @brief Stops all threads and deallocates the thread-pool. Must `try_spawn` again to re-use.
     *  @note Can't be called while any tasks are running.
     */
    void stop_and_reset() noexcept {
        if (threads_count_ == 0) return; // ? Uninitialized
        if (threads_count_ == 1) {
            threads_count_ = 0;
            return; // ? No worker threads to join
        }
        assert(threads_to_sync_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running

        // Stop all threads and wait for them to finish
        _reset_fork_description();
        stop_.store(true, std::memory_order_release);

        thread_index_t const worker_threads = threads_count_ - 1;
        for (thread_index_t i = 0; i != worker_threads; ++i) {
            workers_[i].join();    // ? Wait for the thread to finish
            workers_[i].~thread(); // ? Call destructor
        }

        // Deallocate the thread pool
        allocator_.deallocate(workers_, worker_threads);

        // Prepare for future spawns
        threads_count_ = 0;
        workers_ = nullptr;
        stop_.store(false, std::memory_order_relaxed);
        fork_generation_.store(0, std::memory_order_relaxed);
    }

    /**
     *  @brief Executes a function in parallel on the current and all worker threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     */
    template <typename function_type_>
    void broadcast(function_type_ const &function) noexcept {
        thread_index_t const threads_count = threads();
        assert(threads_count != 0 && "Thread pool not initialized");
        if (threads_count == 1) return function(static_cast<thread_index_t>(0));

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        // ? we live to the higher-level algorithms.
        static_assert(std::is_nothrow_invocable_r<void, function_type_, thread_index_t>::value,
                      "The callback must be invocable with a `thread_index_t` argument");
#endif

        // Configure "fork" details
        fork_lambda_pointer_ = std::addressof(function);
        fork_trampoline_pointer_ = &_call_as_lambda<function_type_>;
        threads_to_sync_.store(threads_count - 1, std::memory_order_relaxed);
        fork_generation_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // Execute on the current "main" thread
        function(static_cast<thread_index_t>(0));

        // Wait until all threads are done
        while (threads_to_sync_.load(std::memory_order_acquire)) std::this_thread::yield();

        // An optional reset of the task variables for debuggability
        _reset_fork_description();
    }

  private:
    void _reset_fork_description() noexcept {
        fork_lambda_pointer_ = nullptr;
        fork_trampoline_pointer_ = nullptr;
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
        generation_index_t last_fork_generation = 0;
        assert(thread_index != 0 && "The zero index is for the main thread, not worker!");

        while (true) {
            // Wait for either: a new ticket or a stop flag
            generation_index_t new_fork_generation;
            bool wants_to_stop;
            while ((new_fork_generation = fork_generation_.load(std::memory_order_acquire)) == last_fork_generation &&
                   (wants_to_stop = stop_.load(std::memory_order_acquire)) == false)
                std::this_thread::yield();

#if _FU_DETECT_CPP_20
            if (wants_to_stop) [[unlikely]] // Attributes require C++20
                return;
#else
            if (wants_to_stop) return;
#endif

            fork_trampoline_pointer_(fork_lambda_pointer_, thread_index);
            last_fork_generation = new_fork_generation;

            // ! The decrement must come after the task is executed
            _FU_MAYBE_UNUSED thread_index_t const before_decrement =
                threads_to_sync_.fetch_sub(1, std::memory_order_release);
            assert(before_decrement > 0 && "We can't be here if there are no worker threads");
        }
    }
};

using thread_pool_t = thread_pool<std::allocator<std::thread>>;

#pragma endregion - Thread Pool

#pragma region - Indexed Tasks

/**
 *  @brief A "prong" - is a tip of a "fork" - describing a "thread" pinning a "task".
 */
template <typename index_type_ = std::size_t>
struct prong {
    using thread_index_t = index_type_; // ? A.k.a. "core index" or "thread ID" in [0, threads_count)
    using task_index_t = index_type_;   // ? A.k.a. "task index" in [0, prongs_count)

    thread_index_t thread_index {0};
    task_index_t task_index {0};

    inline prong() = default;
    inline prong(prong const &) = default;
    inline prong(prong &&) = default;
    inline prong &operator=(prong const &) = default;
    inline prong &operator=(prong &&) = default;

    inline prong(thread_index_t thread, task_index_t task) noexcept : thread_index(thread), task_index(task) {}
    inline operator task_index_t() const noexcept { return task_index; }
};

using prong_t = prong<>; // ? Default prong type with `std::size_t` indices

/**
 *  @brief Placeholder type for Parallel Algorithms.
 */
struct dummy_lambda_t {};

/**
 *  @brief Distributes @p (n) similar duration calls between threads in slices, as opposed to individual indices.
 *
 *  @param[in] pool The thread pool to use for parallel execution
 *  @param[in] n The total length of the range to split between threads.
 *  @param[in] function The callback, receiving @b `prong_t` or an unsigned integer and the slice length.
 */
template <                                                  //
    typename allocator_type_ = std::allocator<std::thread>, //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k,           //
    typename function_type_ = dummy_lambda_t                //
    >
void for_slices(                                                 //
    thread_pool<allocator_type_, index_type_, alignment_> &pool, //
    std::size_t const n, function_type_ const &function) noexcept {

    using pool_t = thread_pool<allocator_type_, index_type_, alignment_>;
    using thread_index_t = typename pool_t::thread_index_t;
    using index_t = index_type_;
    using prong_t = prong<index_t>; // ? A.k.a. "task" = (thread_index, task_index)

#if _FU_DETECT_CPP_17 // ? Having static asserts on C++17 this helps with saner compilation errors
    // ? Exception handling and aggregating return values drastically increases code complexity
    static_assert((std::is_nothrow_invocable_r<void, function_type_, prong_t, index_t>::value ||
                   std::is_nothrow_invocable_r<void, function_type_, index_t, index_t>::value),
                  "The callback must be invocable with a `prong_t` or a `index_t` argument and an unsigned counter");
#endif

    assert(n <= static_cast<std::size_t>(std::numeric_limits<index_t>::max()) && "Will overflow");
    index_t const prongs_count = static_cast<index_t>(n);
    if (prongs_count == 0) return;

    // No need to slice the workload if we only have one thread
    thread_index_t const threads_count = pool.threads();
    if (threads_count == 1 || prongs_count == 1) return function(prong_t {0, 0}, prongs_count);

    // The first (N % M) chunks have size ceil(N/M)
    // The remaining N - (N % M) chunks have size floor(N/M)
    //     where N = prongs_count, M = threads_count
    // See https://lemire.me/blog/2025/05/22/dividing-an-array-into-fair-sized-chunks/
    index_t const quotient = prongs_count / threads_count;
    index_t const remainder = prongs_count % threads_count;

    pool.broadcast([quotient, remainder, function](thread_index_t const thread_index) noexcept {
        index_t const begin = quotient * thread_index + (thread_index < remainder ? thread_index : remainder);
        index_t const count = quotient + (thread_index < remainder ?  1 : 0);
        function(prong_t {thread_index, begin}, count);
    });
}

/**
 *  @brief Distributes @p (n) similar duration calls between threads.
 *
 *  @param[in] pool The thread pool to use for parallel execution.
 *  @param[in] n The number of times to call the @p function.
 *  @param[in] function The callback, receiving @b `prong_t` or a call index as an argument.
 *
 *  Is designed for a "balanced" workload, where all threads have roughly the same amount of work.
 *  @sa `for_n_dynamic` for a more dynamic workload.
 *  The @p function is called @p (n) times, and each thread receives a slice of consecutive tasks.
 *  @sa `for_slices` if you prefer to receive workload slices over individual indices.
 */
template <                                                  //
    typename allocator_type_ = std::allocator<std::thread>, //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k,           //
    typename function_type_ = dummy_lambda_t                //
    >
void for_n(                                                      //
    thread_pool<allocator_type_, index_type_, alignment_> &pool, //
    std::size_t const n, function_type_ const &function) noexcept {

    using index_t = index_type_;
    using prong_t = prong<index_t>; // ? A.k.a. "task" = (thread_index, task_index)

    for_slices(pool, n, [function](prong_t const start_prong, index_t const count_prongs) noexcept {
        for (index_t prong_offset = 0; prong_offset < count_prongs; ++prong_offset)
            function(prong_t {start_prong.thread_index, static_cast<index_t>(start_prong.task_index + prong_offset)});
    });
}

/**
 *  @brief Executes uneven tasks on all threads, greedying for work.
 *  @param[in] n The number of times to call the @p function.
 *  @param[in] function The callback, receiving the `prong_t` or the task index as an argument.
 *  @sa `for_n` for a more "balanced" evenly-splittable workload.
 */
template <                                                  //
    typename allocator_type_ = std::allocator<std::thread>, //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k,           //
    typename function_type_ = dummy_lambda_t                //
    >
void for_n_dynamic(                                              //
    thread_pool<allocator_type_, index_type_, alignment_> &pool, //
    std::size_t const n, function_type_ const &function) noexcept {

    using pool_t = thread_pool<allocator_type_, index_type_, alignment_>;
    using thread_index_t = typename pool_t::thread_index_t;
    using index_t = index_type_;
    using prong_t = prong<index_t>; // ? A.k.a. "task" = (thread_index, prong_index)
    static constexpr std::size_t alignment_k = pool_t::alignment_k;

#if _FU_DETECT_CPP_17 // ? Having static asserts on C++17 this helps with saner compilation errors
    // ? Exception handling and aggregating return values drastically increases code complexity
    static_assert((std::is_nothrow_invocable_r<void, function_type_, prong_t>::value ||
                   std::is_nothrow_invocable_r<void, function_type_, index_t>::value),
                  "The callback must be invocable with a `prong_t` or a `index_t` argument");
#endif

    // No need to slice the work if there is just one task
    assert(n <= static_cast<std::size_t>(std::numeric_limits<index_t>::max()) && "Will overflow");
    index_t const prongs_count = static_cast<index_t>(n);
    if (prongs_count == 0) return;
    if (prongs_count == 1) return function(prong_t {0, 0});

    // If there is just one thread, all work is done on the current thread
    thread_index_t const threads_count = pool.threads();
    if (threads_count == 1) {
        for (index_t i = 0; i < prongs_count; ++i) function(prong_t {0, i});
        return;
    }

    // If there are fewer tasks than threads, each thread gets at most 1 task
    // and that's easier to schedule statically!
    index_t const prongs_static = threads_count;
    if (prongs_count <= prongs_static) return for_n(pool, prongs_count, function);

    // Configure "fork" details
    alignas(alignment_k) std::atomic<index_t> prongs_progress {0};
    index_t const prongs_dynamic = prongs_count - prongs_static;
    assert((prongs_dynamic + threads_count) >= prongs_dynamic && "Overflow detected");

    // If we run this loop at 1 Billion times per second on a 64-bit machine, then every 585 years
    // of computational time we will wrap around the `std::size_t` capacity for the `new_prong_index`.
    // In case we `prongs_count + thread_index >= std::size_t(-1)`, a simple condition won't be enough.
    // Alternatively, we can make sure, that each thread can do at least one increment of `prongs_progress`
    // without worrying about the overflow. The way to achieve that is to preprocess the trailing `threads_count`
    // of elements externally, before entering this loop!
    pool.broadcast(
        [prongs_count, prongs_dynamic, function, &prongs_progress](thread_index_t const thread_index) noexcept {
            // Run one static prong on the current thread
            index_t const one_static_prong_index = static_cast<index_t>(prongs_dynamic + thread_index);
            prong_t prong(thread_index, one_static_prong_index);
            function(prong);

            // The rest can be synchronized with a trivial atomic counter
            while (true) {
                prong.task_index = prongs_progress.fetch_add(1, std::memory_order_relaxed);
                bool const beyond_last_prong = prong.task_index >= prongs_dynamic;
                if (beyond_last_prong) break;
                function(prong);
            }
        });
}

#pragma endregion - Indexed Tasks

} // namespace fork_union
} // namespace ashvardanian
