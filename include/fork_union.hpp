/**
 *  @file   fork_union.hpp
 *  @brief  Minimalistic C++ thread-pool designed for SIMT-style 'Fork-Join' parallelism.
 *  @author Ash Vardanian
 *  @date   May 2, 2025
 */
#pragma once
#include <memory>  // `std::allocator`
#include <thread>  // `std::thread`
#include <atomic>  // `std::atomic`
#include <cstddef> // `std::max_align_t`
#include <cassert> // `assert`
#include <new>     // `std::hardware_destructive_interference_size`

#define FORK_UNION_VERSION_MAJOR 0
#define FORK_UNION_VERSION_MINOR 3
#define FORK_UNION_VERSION_PATCH 3

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

/**
 *  @brief  Minimalistic STL-based non-resizable thread-pool for simultaneous blocking tasks.
 *          "Fork" many threads, enumerate the "prongs" of the "fork", and "join" them back :)
 *
 *  Fork Union supports 2 modes of operation: static and dynamic. Static mode is designed
 *  for balanced workloads, taking roughly the same amount of time to execute on each thread.
 *  Dynamic mode is designed for uneven workloads, with threads "greedying" for work.
 *
 *  This thread-pool @b can't:
 *  - dynamically @b resize: all threads must be stopped and re-initialized to grow/shrink.
 *  - @b re-enter: it can't be used recursively and will deadlock if you try to do so.
 *  - @b copy/move: the threads depend on the address of the parent structure.
 *  - handle @b exceptions: you must `try-catch` them yourself and return `void`.
 *  - @b stop early: assuming the user can do it better, knowing the task granularity.
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
 *
 *  @section Usage
 *
 *  A minimal example, similar to `#pragma omp parallel` in OpenMP:
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp>
 *
 *  using fun = ashvardanian::fork_union;
 *  int main() {
 *      fun::fork_union<> pool;
 *      if (!pool.try_spawn(std::thread::hardware_concurrency())) return EXIT_FAILURE;
 *      pool.for_each_thread([](std::size_t index) { std::printf("Hello from thread %zu\n", index); });
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  User callbacks in a simple case receive just a single integer identifying the part
 *  of the task to be executed. For more advanced use-cases, when the user needs to know
 *  the thread ID (likely to use some thread-local storage), the callbacks should receive
 *  a `prong_t` argument, and unpack the thread ID from it.
 *
 *  @code{.cpp}
 *  #include <cstdio> // `std::printf`
 *  #include <cstdlib> // `EXIT_FAILURE`, `EXIT_SUCCESS`
 *  #include <fork_union.hpp>
 *
 *  using fun = ashvardanian::fork_union;
 *  int main() {
 *      fun::fork_union<> pool;
 *      if (!pool.try_spawn(std::thread::hardware_concurrency()))
 *          return EXIT_FAILURE;
 *
 *      pool.for_each_slice([](auto prong) {
 *          auto const [thread_index, task_index] = prong;
 *          std::printf("Executing task %zu on core %zu\n", task_index, thread_index);
 *      });
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
class fork_union {
  public:
    using allocator_t = allocator_type_;
    using index_t = index_type_;
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    static_assert(alignment_k > 0 && (alignment_k & (alignment_k - 1)) == 0, "Alignment must be a power of 2");

    using thread_index_t = index_t;
    using prong_index_t = index_t;
    using generation_index_t = index_t;

    /**
     *  @brief A POD-type describing a certain part of a parallel task.
     */
    struct prong_t {
        thread_index_t thread_index {0};
        prong_index_t prong_index {0};

        inline prong_t() = default;
        inline prong_t(prong_t const &) = default;
        inline prong_t(prong_t &&) = default;
        inline prong_t &operator=(prong_t const &) = default;
        inline prong_t &operator=(prong_t &&) = default;

        inline prong_t(thread_index_t const thread_index, prong_index_t const prong_index) noexcept
            : thread_index(thread_index), prong_index(prong_index) {}

        inline operator prong_index_t() const noexcept { return prong_index; }
    };

    using punned_fork_context_t = void const *;
    using trampoline_pointer_t = void (*)(punned_fork_context_t, prong_t);

    struct c_callback_t {
        trampoline_pointer_t callable {nullptr};
        punned_fork_context_t context {nullptr};

        inline c_callback_t() = default;
        inline c_callback_t(c_callback_t const &) = default;
        inline c_callback_t(c_callback_t &&) = default;
        inline c_callback_t &operator=(c_callback_t const &) = default;
        inline c_callback_t &operator=(c_callback_t &&) = default;

        inline c_callback_t(trampoline_pointer_t const callable, punned_fork_context_t const context) noexcept
            : callable(callable), context(context) {}

        inline void operator()(prong_t prong) const noexcept { callable(context, prong); }
    };

  private:
    // Thread-pool-specific variables:
    allocator_t allocator_ {};
    std::thread *workers_ {nullptr};
    thread_index_t threads_count_ {0};
    alignas(alignment_) std::atomic<bool> stop_ {false};

    // Task-specific variables:
    punned_fork_context_t fork_lambda_pointer_ {nullptr};    // ? Pointer to the users lambda
    trampoline_pointer_t fork_trampoline_pointer_ {nullptr}; // ? Calls the lambda
    prong_index_t prongs_count_ {0};
    alignas(alignment_) std::atomic<prong_index_t> prongs_remaining_ {0};
    alignas(alignment_) std::atomic<prong_index_t> prongs_passed_ {0}; // ? Only used in dynamic mode
    alignas(alignment_) std::atomic<generation_index_t> fork_generation_ {0};

  public:
    fork_union(fork_union &&) = delete;
    fork_union(fork_union const &) = delete;
    fork_union &operator=(fork_union &&) = delete;
    fork_union &operator=(fork_union const &) = delete;

    fork_union(allocator_t const &alloc = {}) noexcept : allocator_(alloc) {}
    ~fork_union() noexcept { stop_and_reset(); }
    thread_index_t threads_count() const noexcept { return threads_count_; }

    /**
     *  @brief Creates a thread-pool with the given number of threads.
     *  @note This is the de-facto @b constructor, and can only be called once.
     *  @param[in] planned_threads The number of threads to be used. Should be larger than one.
     *  @retval false if the number of threads is zero or the "workers" allocation failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     */
    bool try_spawn(thread_index_t const planned_threads) noexcept {
        if (planned_threads == 0) return false; // ! Can't have zero threads working on something
        if (planned_threads == 1) return true;  // ! The current thread will always be used
        if (threads_count_ != 0) return false;  // ! Already initialized

        // Allocate the thread pool
        thread_index_t const worker_threads = planned_threads - 1;
        std::thread *const workers = allocator_.allocate(worker_threads);
        if (workers == nullptr) return false; // ! Allocation failed

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
        if (threads_count_ == 0) return;                                // ? Uninitialized
        if (threads_count_ == 1) return;                                // ? No worker threads to join
        assert(prongs_remaining_.load(std::memory_order_seq_cst) == 0); // ! No tasks must be running

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
     *  @brief Distributes @p (n) similar duration calls between threads.
     *  @param[in] n The number of times to call the @p function.
     *  @param[in] function The callback, receiving @b `prong_t` or a call index as an argument.
     *
     *  Is designed for a "balanced" workload, where all threads have roughly the same amount of work.
     *  @sa `for_each_dynamic` for a more dynamic workload.
     *  The @p function is called @p (n) times, and each thread receives a slice of consecutive tasks.
     *  @sa `for_each_slice` if you prefer to receive workload slices over individual indices.
     */
    template <typename function_type_>
    void for_each_static(prong_index_t const n, function_type_ const &function) noexcept {

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        static_assert((std::is_nothrow_invocable_r<void, function_type_, prong_t>::value ||
                       std::is_nothrow_invocable_r<void, function_type_, prong_index_t>::value),
                      "The callback must be invocable with a `prong_t` or a `prong_index_t` argument");
#endif

        for_each_slice(n, [function](prong_t start_task, prong_index_t count) noexcept {
            for (prong_index_t i = 0; i < count; ++i)
                function(prong_t {start_task.thread_index, static_cast<prong_index_t>(start_task.prong_index + i)});
        });
    }

    /**
     *  @brief Distributes @p (n) similar duration calls between threads in slices, as opposed to individual indices.
     *  @param[in] n The total length of the range to split between threads.
     *  @param[in] function The callback, receiving @b `prong_t` or an unsigned integer and the slice length.
     */
    template <typename function_type_>
    void for_each_slice(prong_index_t const n, function_type_ const &function) noexcept {

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        static_assert(
            (std::is_nothrow_invocable_r<void, function_type_, prong_t, prong_index_t>::value ||
             std::is_nothrow_invocable_r<void, function_type_, prong_index_t, prong_index_t>::value),
            "The callback must be invocable with a `prong_t` or a `prong_index_t` argument and an unsigned counter");
#endif

        // No need to slice the workload if we only have one thread
        assert(threads_count_ != 0 && "Thread pool not initialized");
        if (threads_count_ == 1 || n == 1) return function(prong_t {0, 0}, n);
        if (n == 0) return;

        // Divide and round-up the workload size per thread - assuming some fuzzer may
        // pass an absurdly large `n` as an input, the addition may overflow,
        // so `(n + threads_count_ - 1) / threads_count_` is not the safest option.
        // Instead, we can do: `n / threads_count_ + (n % threads_count_ != 0)`,
        // but avoiding the cost of the second integer division, replacing it with multiplication.
        prong_index_t const n_per_thread_lower_bound = n / threads_count_;
        prong_index_t const n_per_thread = n_per_thread_lower_bound + ((n_per_thread_lower_bound * threads_count_) < n);
        for_each_thread([n, n_per_thread, n_per_thread_lower_bound, function](thread_index_t thread_index) noexcept {
            // Multiplying `thread_index` by `n_per_thread` may overflow. For an 8-bit `prong_index_t` type:
            // - 254 threads,
            // - 255 tasks,
            // - each thread gets 1 or 2 tasks
            // In that case, both `begin` and `begin_lower_bound` will overflow, but we can use
            // their relative values to determine the real slice length for the thread.
            prong_index_t const begin = thread_index * n_per_thread;                         // ? Handled overflow
            prong_index_t const begin_lower_bound = n_per_thread_lower_bound * thread_index; // ? Handled overflow
            bool const begin_overflows = begin_lower_bound > begin;
            bool const begin_exceeds_n = begin >= n;
            if (begin_overflows || begin_exceeds_n) return;
            prong_index_t const count = (std::min<prong_index_t>)(add_sat(begin, n_per_thread), n) - begin;
            function(prong_t {thread_index, begin}, count);
        });
    }

    /**
     *  @brief Executes a function in parallel on the current and all worker threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     */
    template <typename function_type_>
    void for_each_thread(function_type_ const &function) noexcept {
        if (threads_count_ == 1) return function(0);

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        static_assert(std::is_nothrow_invocable_r<void, function_type_, prong_index_t>::value,
                      "The callback must be invocable with a `prong_index_t` argument");
#endif

        // Configure "fork" details
        fork_lambda_pointer_ = std::addressof(function);
        fork_trampoline_pointer_ = &_call_lambda_from_thread<function_type_>;
        prongs_count_ = threads_count_;
        prongs_remaining_.store(threads_count_ - 1, std::memory_order_relaxed);
        fork_generation_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // Execute on the current "main" thread
        function(0);

        // Wait until all threads are done
        while (prongs_remaining_.load(std::memory_order_acquire)) std::this_thread::yield();

        // An optional reset of the task variables for debuggability
        _reset_fork_description();
    }

    /**
     *  @brief Executes uneven tasks on all threads, greedying for work.
     *  @param[in] n The number of times to call the @p function.
     *  @param[in] function The callback, receiving the `prong_t` or the task index as an argument.
     *  @sa `for_each_static` for a more "balanced" evenly-splittable workload.
     */
    template <typename function_type_>
    void for_each_dynamic(prong_index_t const n, function_type_ const &function) noexcept {

#if _FU_DETECT_CPP_17
        // ? Exception handling and aggregating return values drastically increases code complexity
        static_assert((std::is_nothrow_invocable_r<void, function_type_, prong_t>::value ||
                       std::is_nothrow_invocable_r<void, function_type_, prong_index_t>::value),
                      "The callback must be invocable with a `prong_t` or a `prong_index_t` argument");
#endif

        // If there is just one thread, all work is done on the current thread
        if (threads_count_ == 1) {
            for (prong_index_t i = 0; i < n; ++i) function(prong_t {0, i});
            return;
        }

        // No need to slice the work if there is just one task
        if (n == 0) return;
        if (n == 1) return function(prong_t {0, 0});

        // If there are fewer tasks than threads, each thread gets at most 1 task
        // and that's easier to schedule statically!
        prong_index_t const min_static_suffix_size = threads_count_;
        if (n <= min_static_suffix_size) return for_each_static(n, function);

        // Configure "fork" details
        prong_index_t const n_without_suffix = n - min_static_suffix_size;
        fork_lambda_pointer_ = std::addressof(function);
        fork_trampoline_pointer_ = &_call_lambda_from_thread<function_type_>;
        prongs_count_ = n;
        prongs_passed_.store(0, std::memory_order_relaxed);
        prongs_remaining_.store(n, std::memory_order_relaxed);
        fork_generation_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // Execute on the current "main" thread, assuming the last `threads_count_` will
        // be handled separately to avoid overflow in the dynamic worker loop.
        fork_trampoline_pointer_(fork_lambda_pointer_, {0, n_without_suffix});
        prongs_remaining_.fetch_sub(1, std::memory_order_release);
        _worker_loop_for_dynamic_tasks(0, n_without_suffix);

        // We may be in the in-flight position, where the current thread is already receiving
        // tasks beyond the `prongs_count_` index, but the worker threads are still executing
        // tasks and haven't decremented the `prongs_remaining_` variable yet.
        while (prongs_remaining_.load(std::memory_order_acquire)) std::this_thread::yield();

        // Optional reset of the task variables for debuggability
        _reset_fork_description();
    }

  private:
    void _reset_fork_description() noexcept {
        prongs_count_ = 0;
        fork_lambda_pointer_ = nullptr;
        fork_trampoline_pointer_ = nullptr;
    }

    /**
     *  @brief A trampoline function that is used to call the user-defined lambda.
     *  @param[in] punned_lambda_pointer The pointer to the user-defined lambda.
     *  @param[in] prong The index of the thread & task index packed together.
     */
    template <typename function_type_>
    static void _call_lambda_from_thread(punned_fork_context_t punned_lambda_pointer, prong_t prong) noexcept {
        auto const &lambda_object = *static_cast<function_type_ const *>(punned_lambda_pointer);
        lambda_object(prong);
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

            if (wants_to_stop) return;

            // Check if we are operating in the "dynamic" eager mode or a balanced "static" mode
            prong_index_t const prongs_count = prongs_count_;
            bool const is_static = prongs_count == threads_count_;
            if (is_static) {
                fork_trampoline_pointer_(fork_lambda_pointer_, {thread_index, thread_index});
                // ! The decrement must come after the task is executed
                _FU_MAYBE_UNUSED prong_index_t const before_decrement =
                    prongs_remaining_.fetch_sub(1, std::memory_order_release);
                assert(before_decrement > 0 && "We can't be here if there are no worker threads");
            }
            else {
                // Execute one task per thread separately to avoid possible overflow in the dynamic worker loop
                prong_index_t const n_without_suffix = prongs_count - threads_count_;
                fork_trampoline_pointer_(fork_lambda_pointer_,
                                         {thread_index, static_cast<prong_index_t>(n_without_suffix + thread_index)});
                prongs_remaining_.fetch_sub(1, std::memory_order_release);
                _worker_loop_for_dynamic_tasks(thread_index, n_without_suffix);
            }
            last_fork_generation = new_fork_generation;
        }
    }

    void _worker_loop_for_dynamic_tasks(thread_index_t const thread_index, prong_index_t const prongs_count) noexcept {

        // If we run this loop at 1 Billion times per second on a 64-bit machine, then every 585 years
        // of computational time we will wrap around the `std::size_t` capacity for the `new_prong_index`.
        // In case we `prongs_count + thread_index >= std::size_t(-1)`, a simple condition won't be enough.
        // Alternatively, we can make sure, that each thread can do at least one increment of `prongs_passed_`
        // without worrying about the overflow. The way to achieve that is to preprocess the trailing `threads_count_`
        // of elements externally, before entering this loop!
        assert((prongs_count + threads_count_) >= prongs_count); // ? Detects overflow

        // Unlike the thread-balanced mode, we need to keep track of the number of passed tasks.
        // The traditional way to achieve that is to use the same single atomic `prongs_remaining_`
        // variable, but using `compare_exchange_weak` interfaces. It's much more expensive on modern
        // CPUs, so we use an additional atomic variable and have 2x atomic increments of 2x atomic variables,
        // instead of 1x atomic load and 1x atomic CAS on 1x atomic variable (in optimistic low-contention case).
        while (true) {

            // The relative order of executed tasks doesn't matter here, so we can use relaxed memory order.
            prong_index_t const new_prong_index = prongs_passed_.fetch_add(1, std::memory_order_relaxed);
            bool const beyond_last_prong = new_prong_index >= prongs_count;
            if (beyond_last_prong) break;

            fork_trampoline_pointer_(fork_lambda_pointer_, {thread_index, new_prong_index});
            // ! The decrement must come after the task is executed, as our main thread is waiting
            // ! for that counter to reach zero, to exit the current scope.
            _FU_MAYBE_UNUSED prong_index_t const before_decrement =
                prongs_remaining_.fetch_sub(1, std::memory_order_release);
            assert(before_decrement > 0 && "We can't be here if there are no tasks left");
        }
    }
};

using fork_union_t = fork_union<std::allocator<std::thread>>;

} // namespace fork_union
} // namespace ashvardanian
