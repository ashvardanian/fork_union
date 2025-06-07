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
 *  On Linux, when libNUMA and libpthread are available, the library can also leverage NUMA-aware memory allocations
 *  and pin threads to specific physical cores to increase memory locality. It should reduce memory access latency by
 *  around 35% on average, compared to remote accesses.
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

#define FORK_UNION_VERSION_MAJOR 1
#define FORK_UNION_VERSION_MINOR 0
#define FORK_UNION_VERSION_PATCH 2

#if !defined(FU_ALLOW_UNSAFE)
#define FU_ALLOW_UNSAFE 0
#endif

#if !defined(FU_ENABLE_NUMA)
#if defined(__linux__) && defined(__GLIBC__)
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

/**
 *  @brief Defines the in- and exclusivity of the calling thread in for the executing task.
 *  @sa caller_inclusive_k and caller_exclusive_k
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
 *  @sa mood_t::grind_k, mood_t::chill_k, mood_t::die_k
 */
enum class mood_t {
    grind_k = 0, // ? That's our default ;)
    chill_k,     // ? Sleepy and tired, but just a wake-up call away
    die_k,       // ? The thread is about to die, we must exit the loop peacefully
};

struct standard_yield_t {
    inline void operator()() const noexcept { std::this_thread::yield(); }
};

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
 *      pool.broadcast([](std::size_t index) noexcept { std::printf("Hello from thread %zu\n", index); });
 *      return EXIT_SUCCESS;
 *  }
 *  @endcode
 *
 *  Unlike OpenMP, however, separate thread-pools can be created isolating work and resources.
 *  This is handy when when some logic has to be split between "performance" and "efficiency" cores,
 *  between different NUMA nodes, between GUI and background tasks, etc.
 *
 *
 *
 *  @tparam allocator_type_ The type of the allocator to be used for the thread pool.
 *  @tparam micro_yield_type_ The type of the yield function to be used for busy-waiting.
 *  @tparam index_type_ Defaults to `std::size_t`, but can be changed to a smaller type for debugging.
 *  @tparam alignment_ The alignment of the thread pool. Defaults to `default_alignment_k`.
 */
template <                                                  //
    typename allocator_type_ = std::allocator<std::thread>, //
    typename micro_yield_type_ = standard_yield_t,          //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k            //
    >
class thread_pool {

  public:
    using allocator_t = allocator_type_;
    using micro_yield_t = micro_yield_type_;
    static_assert(std::is_nothrow_invocable_r<void, micro_yield_t>::value,
                  "Yield must be callable w/out arguments & return void");
    static constexpr std::size_t alignment_k = alignment_;
    static_assert(alignment_k > 0 && (alignment_k & (alignment_k - 1)) == 0, "Alignment must be a power of 2");

    using index_t = index_type_;
    static_assert(std::is_unsigned<index_t>::value, "Index type must be an unsigned integer");
    using epoch_index_t = index_t;  // ? A.k.a. number of previous API calls in [0, UINT_MAX)
    using thread_index_t = index_t; // ? A.k.a. "core index" or "thread ID" in [0, threads_count)

    using punned_fork_context_t = void const *;                           // ? Pointer to the on-stack lambda
    using trampoline_t = void (*)(punned_fork_context_t, thread_index_t); // ? Wraps lambda's `operator()`

  private:
    // Thread-pool-specific variables:
    allocator_t allocator_ {};
    std::thread *workers_ {nullptr};
    thread_index_t threads_count_ {0};
    caller_exclusivity_t exclusivity_ {caller_inclusive_k}; // ? Whether the caller thread is included in the count
    std::size_t sleep_length_micros_ {0}; // ? How long to sleep in microseconds when waiting for tasks

    /**
     *  Theoretically, the choice of `std::atomic<bool>` is suboptimal in the presence of `std::atomic_flag`.
     *  The latter is guaranteed to be lock-free, while the former is not. But until C++20, the flag doesn't
     *  have a non-modifying load operation - the `std::atomic_flag::test` was added in C++20.
     *  @see https://en.cppreference.com/w/cpp/atomic/atomic_flag.html
     */
    alignas(alignment_) std::atomic<mood_t> mood_ {mood_t::grind_k};

    // Task-specific variables:
    punned_fork_context_t fork_state_ {nullptr}; // ? Pointer to the users lambda
    trampoline_t fork_trampoline_ {nullptr};     // ? Calls the lambda
    alignas(alignment_) std::atomic<thread_index_t> threads_to_sync_ {0};
    alignas(alignment_) std::atomic<epoch_index_t> epoch_ {0};

  public:
    thread_pool(thread_pool &&) = delete;
    thread_pool(thread_pool const &) = delete;
    thread_pool &operator=(thread_pool &&) = delete;
    thread_pool &operator=(thread_pool const &) = delete;

    thread_pool(allocator_t const &alloc = {}) noexcept : allocator_(alloc) {}
    ~thread_pool() noexcept { terminate(); }

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

    /** @brief Estimates the amount of memory managed by this pool handle and internal structures.
     *  @note This API is @b not synchronized.
     */
    std::size_t memory_usage() const noexcept { return sizeof(thread_pool) + threads_count() * sizeof(std::thread); }

    /** @brief Checks if the thread-pool's core synchronization points are lock-free. */
    bool is_lock_free() const noexcept { return mood_.is_lock_free() && threads_to_sync_.is_lock_free(); }

    /**
     *  @brief Creates a thread-pool with the given number of threads.
     *  @param[in] threads The number of threads to be used. Should be larger than one.
     *  @param[in] exclusivity Should we count the calling thread as one of the threads?
     *  @retval false if the number of threads is zero or the "workers" allocation failed.
     *  @retval true if the thread-pool was created successfully, started, and is ready to use.
     *  @note This is the de-facto @b constructor - you only call it again after `terminate`.
     */
    bool try_spawn(thread_index_t const threads, caller_exclusivity_t const exclusivity = caller_inclusive_k) noexcept {

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

        // Initialize the thread pool can fail for all kinds of reasons,
        // that the `std::thread` documentation describes as "implementation-defined".
        // https://en.cppreference.com/w/cpp/thread/thread/thread
        exclusivity_ = exclusivity;
        for (thread_index_t i = 0; i < worker_threads; ++i) {
            try {
                thread_index_t const i_with_caller = i + use_caller_thread;
                new (&workers[i]) std::thread([this, i_with_caller] { _worker_loop(i_with_caller); });
            }
            catch (...) {
                for (thread_index_t j = 0; j < i; ++j) workers[j].~thread();
                allocator_.deallocate(workers, worker_threads);
                return false; // ! Thread creation failed
            }
        }

        // If all went well, we can store the thread-pool and start using it
        workers_ = workers;
        threads_count_ = threads;
        return true;
    }

    /**
     *  @brief A synchronization point that waits for all threads to finish the last broadcasted call.
     *
     *  You don't have to explicitly handle the return value and wait on it.
     *  According to the  C++ standard, the destructor of the `broadcast_join_t` will be called
     *  in the end of the `broadcast`-calling expression.
     */
    class broadcast_join_t {
        std::atomic<thread_index_t> &threads_to_sync_;

      public:
        explicit broadcast_join_t(std::atomic<thread_index_t> &sync) noexcept : threads_to_sync_(sync) {}
        broadcast_join_t(broadcast_join_t &&) = default;
        broadcast_join_t(broadcast_join_t const &) = delete;
        broadcast_join_t &operator=(broadcast_join_t &&) = default;
        broadcast_join_t &operator=(broadcast_join_t const &) = delete;
        ~broadcast_join_t() noexcept { wait(); }
        void wait() const noexcept {
            micro_yield_t micro_yield;
            while (threads_to_sync_.load(std::memory_order_acquire)) micro_yield();
        }
    };

    /**
     *  @brief Executes a @p function in parallel on all threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     *  @return A `broadcast_join_t` synchronization point that waits in the destructor.
     *  @note Even in the `caller_exclusive_k` mode, can be called from just one thread!
     *  @sa For advanced resource management, consider `unsafe_broadcast` and `unsafe_join`.
     */
    template <typename function_type_>
    broadcast_join_t broadcast(function_type_ const &function) noexcept {
        unsafe_broadcast(function);
        return unsafe_join();
    }

    /**
     *  @brief Executes a @p function in parallel on all threads, not waiting for the result.
     *  @param[in] function The callback, receiving the thread index as an argument.
     *  @sa Use in conjunction with `unsafe_join`.
     */
    template <typename function_type_>
    void unsafe_broadcast(function_type_ const &function) noexcept {

        thread_index_t const threads = threads_count();
        assert(threads != 0 && "Thread pool not initialized");
        caller_exclusivity_t const exclusivity = caller_exclusivity();
        bool const use_caller_thread = exclusivity == caller_inclusive_k;
        if (threads == 1 && use_caller_thread) return function(static_cast<thread_index_t>(0));

        // Optional check: even in exclusive mode, only one thread can call this function.
        assert((use_caller_thread || threads_to_sync_.load(std::memory_order_acq_rel) == 0) &&
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
        epoch_.fetch_add(1, std::memory_order_release); // ? Wake up sleepers

        // Execute on the current "main" thread
        if (use_caller_thread) function(static_cast<thread_index_t>(0));
    }

    /** @brief Blocks the calling thread until the currently broadcasted task finishes. */
    broadcast_join_t unsafe_join() noexcept { return broadcast_join_t(threads_to_sync_); }

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

        // Stop all threads and wait for them to finish
        _reset_fork();
        mood_.store(mood_t::die_k, std::memory_order_release);

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
        mood_.store(mood_t::grind_k, std::memory_order_relaxed);
        epoch_.store(0, std::memory_order_relaxed);
    }

    /**
     *  @brief Transitions "workers" to a sleeping state, waiting for a wake-up call.
     *  @param[in] wake_up_periodicity_micros How often to check for new work in microseconds.
     *  @note Can only be called between the tasks for a single thread. No synchronization is performed.
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
    }

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

            if (_FU_UNLIKELY(mood == mood_t::die_k)) { return; }
            if (_FU_UNLIKELY(mood == mood_t::chill_k)) {
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

using thread_pool_t = thread_pool<>;

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
        { p.broadcast(broadcasted_noop_t {}) };
    };

template <typename pool_type_>
concept is_unsafe_pool =   //
    is_pool<pool_type_> && //
    requires(pool_type_ &p) {
        { p.unsafe_broadcast(broadcasted_noop_t {}) } -> std::same_as<void>;
    } && //
    requires(pool_type_ &p) {
        { p.unsafe_join() } -> std::same_as<typename pool_type_::broadcast_join_t>;
    };

#endif
#pragma endregion Concepts
#pragma endregion - Thread Pool

#pragma region - Hardware Friendly Yield

#pragma endregion - Hardware Friendly Yield

#pragma region - NUMA Pools
#if FU_ENABLE_NUMA

using numa_core_id_t = int; // ? A.k.a. CPU core ID, in [0, threads_count)
using numa_node_id_t = int; // ? A.k.a. NUMA node ID, in [0, numa_max_node())

/**
 *  @brief NUMA topology descriptor: describing memory pools and core counts next to them.
 *
 *  Uses dynamic memory to store the NUMA nodes and their cores. Assuming we may soon have
 *  Intel "Sierra Forest"-like CPUs with 288 cores with up to 8 sockets per node, this structure
 *  can easily grow to 10 KB.
 */
template <typename allocator_type_ = std::allocator<char>>
struct numa_topology {

    struct numa_node_t {
        numa_node_id_t node_id {-1};                   // ? Unique NUMA node ID, in [0, numa_max_node())
        std::size_t memory_size {0};                   // ? RAM volume in bytes
        numa_core_id_t const *first_core_id {nullptr}; // ? Pointer to the first core ID in the `core_ids` array
        std::size_t core_count {0};                    // ? Number of items in `core_ids` array
    };

    using allocator_t = allocator_type_;
    using cores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<int>;
    using nodes_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<numa_node_t>;

  private:
    numa_node_t *nodes_ {nullptr};
    numa_core_id_t *grouped_core_ids_ {nullptr}; // ? Unsigned integers in [0, threads_count), grouped by NUMA node
    std::size_t nodes_count_ {0};                // ? Number of NUMA nodes
    std::size_t cores_count_ {0};                // ? Total number of cores in all nodes
    allocator_t allocator_ {};

  public:
    constexpr numa_topology() noexcept = default;
    numa_topology(numa_topology &&o) noexcept
        : allocator_(std::move(o.allocator_)), nodes_(o.nodes_), grouped_core_ids_(o.grouped_core_ids_),
          nodes_count_(o.nodes_count_), cores_count_(o.cores_count_) {
        o.nodes_ = nullptr;
        o.grouped_core_ids_ = nullptr;
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

        if (grouped_core_ids_) cores_alloc.deallocate(grouped_core_ids_, cores_count_);
        if (nodes_) nodes_alloc.deallocate(nodes_, nodes_count_);

        nodes_ = nullptr;
        grouped_core_ids_ = nullptr;
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
        numa_core_id_t *cores_ptr = nullptr;
        numa_node_id_t max_numa_node_id = -1;

        // Allocators must be visible to the cleanup path
        nodes_allocator_t nodes_alloc {allocator_};
        cores_allocator_t cores_alloc {allocator_};

        // These counters are reused in the failure handler
        std::size_t fetched_nodes = 0, fetched_cores = 0;

        if (numa_available() < 0) goto failed_harvest; // ! Linux kernel lacks NUMA support

        numa_mask = numa_allocate_cpumask();
        if (!numa_mask) goto failed_harvest; // ! Allocation failed

        // First pass – measure
        max_numa_node_id = numa_max_node();
        for (numa_node_id_t node_id = 0; node_id <= max_numa_node_id; ++node_id) {
            long long dummy;
            if (numa_node_size64(node_id, &dummy) < 0) continue;     // ! Offline node
            if (numa_node_to_cpus(node_id, numa_mask) < 0) continue; // ! Invalid CPU map
            fetched_nodes += 1;
            fetched_cores += static_cast<std::size_t>(numa_bitmask_weight(numa_mask));
        }
        if (fetched_nodes == 0) goto failed_harvest; // ! Zero nodes is not a valid state

        // Second pass – allocate
        nodes_ptr = nodes_alloc.allocate(fetched_nodes);
        cores_ptr = cores_alloc.allocate(fetched_cores);
        if (!nodes_ptr || !cores_ptr) goto failed_harvest; // ! Allocation failed

        // Populate
        for (numa_node_id_t node_id = 0, core_id = 0; node_id <= max_numa_node_id; ++node_id) {
            long long memory_size;
            if (numa_node_size64(node_id, &memory_size) < 0) continue;
            if (numa_node_to_cpus(node_id, numa_mask) < 0) continue;

            numa_node_t &node = nodes_ptr[node_id];
            node.node_id = node_id;
            node.memory_size = static_cast<std::size_t>(memory_size);
            node.first_core_id = cores_ptr + core_id;
            node.core_count = static_cast<std::size_t>(numa_bitmask_weight(numa_mask));

            // Most likely, this will fill `cores_ptr` with `std::iota`-like values
            for (std::size_t bit_offset = 0; bit_offset < numa_mask->size; ++bit_offset)
                if (numa_bitmask_isbitset(numa_mask, bit_offset))
                    cores_ptr[core_id++] = static_cast<numa_core_id_t>(bit_offset);
        }

        // Commit
        nodes_ = nodes_ptr;
        grouped_core_ids_ = cores_ptr;
        nodes_count_ = fetched_nodes;
        cores_count_ = fetched_cores;

        numa_free_cpumask(numa_mask); // ? Clean up
        return true;

    failed_harvest:
        if (nodes_ptr) nodes_alloc.deallocate(nodes_ptr, fetched_nodes);
        if (cores_ptr) cores_alloc.deallocate(cores_ptr, fetched_cores);
        if (numa_mask) numa_free_cpumask(numa_mask);
        return false;
    }
};

/** @brief STL-compatible allocator that puts memory on a fixed NUMA node. */
template <typename value_type_>
struct numa_allocator {
    using value_type = value_type_;
    numa_node_id_t node_id {-1};

    explicit numa_allocator(numa_node_id_t id = -1) noexcept : node_id(id) {}
    template <typename other_type_>
    constexpr numa_allocator(numa_allocator<other_type_> const &o) noexcept : node_id(o.node_id) {}

    value_type *allocate(std::size_t n) noexcept {
        return static_cast<value_type *>(numa_alloc_onnode(n * sizeof(value_type), node_id));
    }
    void deallocate(value_type *p, std::size_t n) noexcept { numa_free(p, n * sizeof(value_type)); }

    template <typename other_type_>
    bool operator==(numa_allocator<other_type_> const &o) const noexcept {
        return node_id == o.node_id;
    }

    template <typename other_type_>
    bool operator!=(numa_allocator<other_type_> const &o) const noexcept {
        return node_id != o.node_id;
    }
};

/**
 *  @brief A thread-pool addressing all of the threads on a certain NUMA node.
 */
template <                                                  //
    typename allocator_type_ = numa_allocator<std::thread>, //
    typename micro_yield_type_ = standard_yield_t,          //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k            //
    >
struct numa_thread_pool {};

template <                                                  //
    typename allocator_type_ = numa_allocator<std::thread>, //
    typename micro_yield_type_ = standard_yield_t,          //
    typename index_type_ = std::size_t,                     //
    std::size_t alignment_ = default_alignment_k            //
    >
struct numa_thread_pools {};

using numa_thread_pool_t = numa_thread_pool<>;
using numa_thread_pools_t = numa_thread_pools<>;

// static_assert(is_unsafe_pool<thread_pool_t> && is_unsafe_pool<numa_thread_pool_t>,
//               "These thread pools must be flexible and support unsafe operations");
// static_assert(is_pool<thread_pool_t> && is_pool<numa_thread_pool_t> && is_pool<numa_thread_pools_t>,
//               "These thread pools must be fully compatible with the high-level APIs");

#endif // FU_ENABLE_NUMA
#pragma endregion - NUMA Pools

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
template <typename pool_type_ = thread_pool_t, typename function_type_ = dummy_lambda_t>
#if _FU_DETECT_CPP_20
    requires(                    //
        (is_pool<pool_type_>) && //
        // ? The callback must be invocable with a `prong_t` or a `index_t` argument and an unsigned counter
        (std::is_nothrow_invocable_r_v<void, function_type_, prong<typename pool_type_::index_t>,
                                       typename pool_type_::index_t> ||
         std::is_nothrow_invocable_r_v<void, function_type_, typename pool_type_::index_t,
                                       typename pool_type_::index_t>))
#endif
void for_slices(pool_type_ &pool, std::size_t const n, function_type_ const &function) noexcept {

    using pool_t = pool_type_;
    using thread_index_t = typename pool_t::thread_index_t;
    using index_t = typename pool_t::index_t; // ? A.k.a. "task index" in [0, prongs_count)
    using prong_t = prong<index_t>;           // ? A.k.a. "task" = {thread_index, task_index}

    assert(n <= static_cast<std::size_t>(std::numeric_limits<index_t>::max()) && "Will overflow");
    index_t const prongs_count = static_cast<index_t>(n);
    if (prongs_count == 0) return;

    // No need to slice the workload if we only have one thread
    thread_index_t const threads_count = pool.threads_count();
    if (threads_count == 1 || prongs_count == 1) return function(prong_t {0, 0}, prongs_count);

    // Divide and round-up the workload size per thread - assuming some fuzzer may
    // pass an absurdly large `prongs_count` as an input, the addition may overflow,
    // so `(prongs_count + threads_count - 1) / threads_count` is not the safest option.
    // Instead, we can do: `prongs_count / threads_count + (prongs_count % threads_count != 0)`,
    // but avoiding the cost of the second integer division, replacing it with multiplication.
    index_t const tasks_per_thread_lower_bound = prongs_count / threads_count;
    index_t const tasks_per_thread =
        tasks_per_thread_lower_bound + ((tasks_per_thread_lower_bound * threads_count) < prongs_count);

    pool.broadcast([prongs_count, tasks_per_thread, tasks_per_thread_lower_bound,
                    function](thread_index_t const thread_index) noexcept {
        // Multiplying `thread_index` by `tasks_per_thread` may overflow. For an 8-bit `index_t` type:
        // - 254 threads,
        // - 255 tasks,
        // - each thread gets 1 or 2 tasks
        // In that case, both `begin` and `begin_lower_bound` will overflow, but we can use
        // their relative values to determine the real slice length for the thread.
        index_t const begin = thread_index * tasks_per_thread;                         // ? Handled overflow
        index_t const begin_lower_bound = tasks_per_thread_lower_bound * thread_index; // ? Handled overflow
        bool const begin_overflows = begin_lower_bound > begin;
        bool const begin_exceeds_n = begin >= prongs_count;
        if (begin_overflows || begin_exceeds_n) return;
        index_t const count = (std::min<index_t>)(add_sat(begin, tasks_per_thread), prongs_count) - begin;
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
template <typename pool_type_ = thread_pool_t, typename function_type_ = dummy_lambda_t>
void for_n(pool_type_ &pool, std::size_t const n, function_type_ const &function) noexcept {

    using pool_t = pool_type_;
    using index_t = typename pool_t::index_t; // ? A.k.a. "task index" in [0, prongs_count)
    using prong_t = prong<index_t>;           // ? A.k.a. "task" = {thread_index, task_index}

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
template <typename pool_type_ = thread_pool_t, typename function_type_ = dummy_lambda_t>
#if _FU_DETECT_CPP_20
    requires(                    //
        (is_pool<pool_type_>) && //
        // ? The callback must be invocable with a `prong_t` or a `index_t` argument
        (std::is_nothrow_invocable_r_v<void, function_type_, prong<typename pool_type_::index_t>> ||
         std::is_nothrow_invocable_r_v<void, function_type_, typename pool_type_::index_t>))
#endif
void for_n_dynamic(pool_type_ &pool, std::size_t const n, function_type_ const &function) noexcept {

    using pool_t = pool_type_;
    using thread_index_t = typename pool_t::thread_index_t;
    using index_t = typename pool_t::index_t; // ? A.k.a. "task index" in [0, prongs_count)
    using prong_t = prong<index_t>;           // ? A.k.a. "task" = {thread_index, prong_index}
    static constexpr std::size_t alignment_k = pool_t::alignment_k;

    // No need to slice the work if there is just one task
    assert(n <= static_cast<std::size_t>(std::numeric_limits<index_t>::max()) && "Will overflow");
    index_t const prongs_count = static_cast<index_t>(n);
    if (prongs_count == 0) return;
    if (prongs_count == 1) return function(prong_t {0, 0});

    // If there is just one thread, all work is done on the current thread
    thread_index_t const threads_count = pool.threads_count();
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
