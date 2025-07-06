//! OpenMP-style cross-platform fine-grained parallelism library.
//!
//! Fork Union provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
//! avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
//! The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
//! execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
//! to use even with the maximal `usize` values.
//!
//! This Rust wrapper provides a safe interface around the precompiled C library, maintaining zero-allocation
//! principles while leveraging NUMA-aware optimizations and CPU-specific busy-waiting instructions.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::ffi::CStr;

use core::cell::UnsafeCell;
use core::ffi::{c_char, c_int, c_void};
use core::ptr::NonNull;
use core::slice;
use core::sync::atomic::{AtomicBool, Ordering};

/// A generic spin mutex that uses CPU-specific pause instructions for efficient busy-waiting.
///
/// This is a low-level synchronization primitive that spins on a busy loop rather than
/// blocking the thread. It's most appropriate for very short critical sections where
/// the cost of context switching would be higher than busy-waiting.
///
/// The generic parameter `P` allows customization of the pause behavior:
/// - `true` enables CPU-specific pause instructions (recommended for most use cases)
/// - `false` disables pause instructions (may be useful in some specialized scenarios)
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// // Create a spin mutex with pause instructions enabled
/// let mutex = BasicSpinMutex::<i32, true>::new(42);
///
/// // Lock, access data, and unlock
/// {
///     let mut guard = mutex.lock();
///     *guard = 100;
/// } // Lock is automatically released when guard goes out of scope
///
/// // Verify the value was changed
/// assert_eq!(*mutex.lock(), 100);
/// ```
///
/// # Performance Characteristics
///
/// - **Very fast for short critical sections** - no syscalls or context switches
/// - **CPU-efficient busy-waiting** - uses pause instructions when `P = true`
/// - **Memory efficient** - only requires a single atomic bool plus the protected data
/// - **Can cause high CPU usage** - spins continuously until lock is acquired
/// - **Not fair** - no guarantee of acquisition order
///
/// # When to Use
///
/// Use `BasicSpinMutex` when:
/// - Critical sections are very short (microseconds)
/// - Lock contention is low
/// - You need the absolute minimum latency
/// - You're in a no_std environment
///
/// Avoid `BasicSpinMutex` when:
/// - Critical sections are long (milliseconds or more)
/// - Lock contention is high
/// - You need fairness guarantees
/// - Power consumption is a concern
pub struct BasicSpinMutex<T, const PAUSE: bool> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

impl<T, const PAUSE: bool> BasicSpinMutex<T, PAUSE> {
    /// Creates a new spin mutex in the unlocked state.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to be protected by the mutex
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(0);
    /// ```
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquires the lock, returning a guard that provides access to the protected data.
    ///
    /// This method will spin until the lock is acquired. If the lock is already held,
    /// it will busy-wait using CPU-specific pause instructions (if `PAUSE = true`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(0);
    /// let mut guard = mutex.lock();
    /// *guard = 42;
    /// ```
    pub fn lock(&self) -> BasicSpinMutexGuard<T, PAUSE> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Busy-wait with pause instructions if enabled
            if PAUSE {
                core::hint::spin_loop();
            }
        }
        BasicSpinMutexGuard { mutex: self }
    }

    /// Attempts to acquire the lock without blocking.
    ///
    /// Returns `Some(guard)` if the lock was successfully acquired, or `None` if
    /// the lock is currently held by another thread.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(0);
    ///
    /// if let Some(mut guard) = mutex.try_lock() {
    ///     *guard = 42;
    ///     println!("Lock acquired and value set");
    /// } else {
    ///     println!("Lock is currently held by another thread");
    /// };
    /// ```
    pub fn try_lock(&self) -> Option<BasicSpinMutexGuard<T, PAUSE>> {
        if self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(BasicSpinMutexGuard { mutex: self })
        } else {
            None
        }
    }

    /// Checks if the mutex is currently locked.
    ///
    /// This method provides a non-blocking way to check the lock state, but should
    /// be used carefully as the state can change immediately after this call returns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(0);
    /// assert!(!mutex.is_locked());
    ///
    /// {
    ///     let _guard = mutex.lock();
    ///     assert!(mutex.is_locked());
    /// }
    ///
    /// assert!(!mutex.is_locked());
    /// ```
    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Acquire)
    }

    /// Consumes the mutex and returns the protected data.
    ///
    /// This method bypasses the locking mechanism entirely since we have exclusive
    /// ownership of the mutex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(42);
    /// let data = mutex.into_inner();
    /// assert_eq!(data, 42);
    /// ```
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Gets a mutable reference to the protected data.
    ///
    /// Since this requires a mutable reference to the mutex, no locking is needed
    /// as we have exclusive access.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut mutex = BasicSpinMutex::<i32, true>::new(0);
    /// *mutex.get_mut() = 42;
    /// assert_eq!(*mutex.lock(), 42);
    /// ```
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }
}

// Safety: BasicSpinMutex can be sent between threads if T can be sent
unsafe impl<T: Send, const PAUSE: bool> Send for BasicSpinMutex<T, PAUSE> {}
// Safety: BasicSpinMutex can be shared between threads if T can be sent
unsafe impl<T: Send, const PAUSE: bool> Sync for BasicSpinMutex<T, PAUSE> {}

/// A guard providing access to the data protected by a `BasicSpinMutex`.
///
/// The lock is automatically released when this guard is dropped.
pub struct BasicSpinMutexGuard<'a, T, const PAUSE: bool> {
    mutex: &'a BasicSpinMutex<T, PAUSE>,
}

impl<'a, T, const PAUSE: bool> BasicSpinMutexGuard<'a, T, PAUSE> {
    /// Returns a reference to the protected data.
    ///
    /// This method is rarely needed since the guard implements `Deref`.
    pub fn get(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }

    /// Returns a mutable reference to the protected data.
    ///
    /// This method is rarely needed since the guard implements `DerefMut`.
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<'a, T, const PAUSE: bool> core::ops::Deref for BasicSpinMutexGuard<'a, T, PAUSE> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T, const PAUSE: bool> core::ops::DerefMut for BasicSpinMutexGuard<'a, T, PAUSE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<'a, T, const PAUSE: bool> Drop for BasicSpinMutexGuard<'a, T, PAUSE> {
    fn drop(&mut self) {
        self.mutex.locked.store(false, Ordering::Release);
    }
}

/// A type alias for the most commonly used spin mutex configuration.
///
/// This is equivalent to `BasicSpinMutex<T, true>`, which enables CPU-specific
/// pause instructions for efficient busy-waiting.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// let mutex = SpinMutex::new(42);
/// let mut guard = mutex.lock();
/// *guard = 100;
/// ```
pub type SpinMutex<T> = BasicSpinMutex<T, true>;

/// A "prong" - the tip of a "fork" - pinning a "task" to a "thread" and "memory" location.
///
/// A `Prong` represents a single unit of work that connects:
/// - A **task** (what work to do) - identified by `task_index`  
/// - A **thread** (which CPU thread is executing it) - identified by `thread_index`
/// - A **colocation** (which NUMA node/QoS level it's running on) - identified by `colocation_index`
///
/// This metadata is essential for NUMA-aware algorithms, debugging parallel execution,
/// and understanding load distribution across the thread pool.
#[derive(Copy, Clone, Debug)]
pub struct Prong {
    /// The logical index of the task being processed (0-based)
    pub task_index: usize,
    /// The physical thread executing this task (0-based)  
    pub thread_index: usize,
    /// The colocation group this thread belongs to (NUMA node + QoS level)
    pub colocation_index: usize,
}

/// Error types that can occur during thread pool operations.
#[derive(Debug)]
pub enum Error {
    /// Thread pool creation failed
    CreationFailed,
    /// Thread spawning failed
    SpawnFailed,
    /// Invalid parameter provided
    InvalidParameter,
    /// Platform not supported
    UnsupportedPlatform,
}

#[cfg(feature = "std")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreationFailed => write!(f, "failed to create thread pool"),
            Self::SpawnFailed => write!(f, "failed to spawn worker threads"),
            Self::InvalidParameter => write!(f, "invalid parameter provided"),
            Self::UnsupportedPlatform => write!(f, "platform not supported"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

// C FFI declarations
extern "C" {

    // Library metadata
    fn fu_version_major() -> usize;
    fn fu_version_minor() -> usize;
    fn fu_version_patch() -> usize;
    fn fu_enabled_numa() -> c_int;
    fn fu_capabilities_string() -> *const c_char;

    // Systems metadata
    fn fu_count_logical_cores() -> usize;
    fn fu_count_colocations() -> usize;
    fn fu_count_numa_nodes() -> usize;
    fn fu_count_quality_levels() -> usize;

    // Core thread pool operations
    fn fu_pool_new(name: *const c_char) -> *mut c_void;
    fn fu_pool_delete(pool: *mut c_void);
    fn fu_pool_spawn(pool: *mut c_void, threads: usize, exclusivity: c_int) -> c_int;
    fn fu_pool_terminate(pool: *mut c_void);
    fn fu_pool_count_threads(pool: *mut c_void) -> usize;
    fn fu_pool_count_colocations(pool: *mut c_void) -> usize;
    fn fu_pool_count_threads_in(pool: *mut c_void, colocation_index: usize) -> usize;
    fn fu_pool_locate_thread_in(pool: *mut c_void, global_thread_index: usize, colocation_index: usize) -> usize;

    #[allow(dead_code)]
    fn fu_pool_for_threads(
        pool: *mut c_void,
        callback: extern "C" fn(*mut c_void, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_for_n(
        pool: *mut c_void,
        n: usize,
        callback: extern "C" fn(*mut c_void, usize, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_for_n_dynamic(
        pool: *mut c_void,
        n: usize,
        callback: extern "C" fn(*mut c_void, usize, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_for_slices(
        pool: *mut c_void,
        n: usize,
        callback: extern "C" fn(*mut c_void, usize, usize, usize, usize),
        context: *mut c_void,
    );

    fn fu_pool_unsafe_for_threads(
        pool: *mut c_void,
        callback: extern "C" fn(*mut c_void, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_unsafe_join(pool: *mut c_void);
    fn fu_pool_sleep(pool: *mut c_void, micros: usize);

    fn fu_allocate_at_least(
        numa_node_index: usize,
        minimum_bytes: usize,
        allocated_bytes: *mut usize,
        bytes_per_page: *mut usize,
    ) -> *mut c_void;
    fn fu_allocate(numa_node_index: usize, bytes: usize) -> *mut c_void;
    fn fu_free(numa_node_index: usize, pointer: *mut c_void, bytes: usize);
    fn fu_volume_huge_pages(numa_node_index: usize) -> usize;
    fn fu_volume_any_pages(numa_node_index: usize) -> usize;

}

/// Returns a string describing available platform capabilities.
#[cfg(feature = "std")]
pub fn capabilities_string() -> Option<&'static str> {
    unsafe {
        let ptr = fu_capabilities_string();
        if ptr.is_null() {
            None
        } else {
            CStr::from_ptr(ptr).to_str().ok()
        }
    }
}

/// Returns a raw pointer to the capabilities string for no_std environments.
pub fn capabilities_string_ptr() -> *const c_char {
    unsafe { fu_capabilities_string() }
}

/// Returns the volume of any pages (huge or regular) available on the specified NUMA node.
pub fn volume_any_pages(numa_node_index: usize) -> usize {
    unsafe { fu_volume_any_pages(numa_node_index) }
}

/// Returns the number of logical CPU cores available on the system.
pub fn count_logical_cores() -> usize {
    unsafe { fu_count_logical_cores() }
}

/// Returns the number of NUMA nodes available on the system.
pub fn count_numa_nodes() -> usize {
    unsafe { fu_count_numa_nodes() }
}

/// Returns the number of distinct thread colocations available.
///
/// A "colocation" represents a group of threads that share the same:
/// - **NUMA memory domain** - threads with fast local memory access
/// - **Quality-of-Service level** - P-cores vs E-cores on heterogeneous CPUs  
/// - **Cache hierarchy** - threads sharing L3 cache
///
/// # Typical Values
///
/// - `1` on most desktop, laptop, or IoT platforms with unified memory
/// - `2-8` on typical dual-socket servers or heterogeneous mobile chips
/// - `4-32` on high-end cloud servers with multiple sockets
pub fn count_colocations() -> usize {
    unsafe { fu_count_colocations() }
}

/// Defines whether the calling thread participates in task execution.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CallerExclusivity {
    /// The calling thread participates in the workload (spawns N-1 workers)
    Inclusive = 0,
    /// The calling thread only coordinates, doesn't execute tasks (spawns N workers)
    Exclusive = 1,
}

/// Returns the number of distinct Quality-of-Service levels.
pub fn count_quality_levels() -> usize {
    unsafe { fu_count_quality_levels() }
}

/// Returns true if NUMA support was compiled into the library.
pub fn numa_enabled() -> bool {
    unsafe { fu_enabled_numa() != 0 }
}

/// Returns the major version number of the Fork Union library.
pub fn version_major() -> usize {
    unsafe { fu_version_major() }
}

/// Returns the minor version number of the Fork Union library.
pub fn version_minor() -> usize {
    unsafe { fu_version_minor() }
}

/// Returns the patch version number of the Fork Union library.
pub fn version_patch() -> usize {
    unsafe { fu_version_patch() }
}

/// Returns the library version as a tuple of (major, minor, patch).
pub fn version() -> (usize, usize, usize) {
    (version_major(), version_minor(), version_patch())
}

/// Minimalistic, fixed‑size thread‑pool for blocking scoped parallelism.
///
/// This is a safe Rust wrapper around the precompiled C thread pool implementation.
/// The current thread **participates** in the work, so for `N`‑way parallelism the
/// implementation actually spawns **N − 1** background workers and runs the last
/// slice on the caller thread.
///
/// # Thread Safety
///
/// `ThreadPool` is `Send + Sync` and can be safely shared between threads, though
/// operations require a mutable reference to ensure exclusive access during execution.
///
/// # Performance Characteristics
///
/// - Zero dynamic allocations during task execution
/// - Leverages weak memory model for optimal ARM and PowerPC performance  
/// - NUMA-aware thread placement when available
/// - Uses CPU-specific busy-waiting instructions for minimal latency
///
/// # Examples
///
/// Basic usage with simple computations:
///
/// ```rust
/// use fork_union::*;
///
/// // Create a thread pool with 4 threads
/// let mut pool = spawn(4);
///
/// // Execute work on each thread
/// pool.for_threads(|thread_index, colocation_index| {
///     println!("Thread {} on colocation {}", thread_index, colocation_index);
/// });
///
/// // Distribute 1000 tasks across threads
/// pool.for_n(1000, |prong| {
///     // Each task gets a unique index via prong.task_index
///     let result = prong.task_index * prong.task_index;
///     std::hint::black_box(result); // Prevent optimization
/// });
/// ```
///
/// For working with data, use the helper functions:
///
/// ```rust
/// use fork_union::*;
///
/// let mut pool = spawn(4);
/// let mut data = vec![0u32; 1000];
///
/// // Process each element in parallel
/// for_each_prong_mut(&mut pool, &mut data, |item, prong| {
///     *item = prong.task_index as u32 * 2;
/// });
///
/// // Verify results
/// for (i, &value) in data.iter().enumerate() {
///     assert_eq!(value, i as u32 * 2);
/// }
/// ```
pub struct ThreadPool {
    inner: *mut c_void,
}

unsafe impl Send for ThreadPool {}
unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn try_spawn_with_exclusivity(
        threads: usize,
        exclusivity: CallerExclusivity,
    ) -> Result<Self, Error> {
        Self::try_named_spawn_with_exclusivity(None, threads, exclusivity)
    }

    pub fn try_named_spawn_with_exclusivity(
        name: Option<&str>,
        threads: usize,
        exclusivity: CallerExclusivity,
    ) -> Result<Self, Error> {
        if threads == 0 {
            return Err(Error::InvalidParameter);
        }

        unsafe {
            let name_ptr = if let Some(name_str) = name {
                let mut name_buffer = [0u8; 16];
                let name_bytes = name_str.as_bytes();
                let copy_len = core::cmp::min(name_bytes.len(), 15); // Leave space for null terminator
                name_buffer[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
                // name_buffer[copy_len] is already 0 from initialization
                name_buffer.as_ptr() as *const c_char
            } else {
                core::ptr::null()
            };

            let inner = fu_pool_new(name_ptr);
            if inner.is_null() {
                return Err(Error::CreationFailed);
            }

            let success = fu_pool_spawn(inner, threads, exclusivity as c_int);
            if success == 0 {
                fu_pool_delete(inner);
                return Err(Error::SpawnFailed);
            }

            Ok(Self { inner })
        }
    }
    /// Creates a new thread pool with the specified number of threads.
    ///
    /// By default, uses `CallerExclusivity::Inclusive`, meaning the calling thread
    /// participates in work execution. For `N` threads, this spawns `N-1` background
    /// workers plus uses the caller thread.
    ///
    /// # Arguments
    ///
    /// * `threads` - Total number of threads including the caller thread
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// // Create a pool that uses 4 threads total (3 spawned + caller)
    /// let pool = ThreadPool::try_spawn(4).expect("Failed to create thread pool");
    /// assert_eq!(pool.threads(), 4);
    /// ```
    pub fn try_spawn(threads: usize) -> Result<Self, Error> {
        Self::try_spawn_with_exclusivity(threads, CallerExclusivity::Inclusive)
    }

    /// Creates a new named thread pool with the specified number of threads.
    ///
    /// The thread pool name can be useful for debugging, profiling, and system monitoring.
    /// On supported platforms, the name may be visible in system tools and thread listings.
    /// Names are truncated to 15 characters (plus null terminator) to fit platform limits.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the thread pool (up to 15 characters)
    /// * `threads` - Total number of threads including the caller thread
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let pool = ThreadPool::try_named_spawn("worker_pool", 4).expect("Failed to create thread pool");
    /// assert_eq!(pool.threads(), 4);
    /// ```
    pub fn try_named_spawn(name: &str, threads: usize) -> Result<Self, Error> {
        Self::try_named_spawn_with_exclusivity(Some(name), threads, CallerExclusivity::Inclusive)
    }

    /// Returns the number of threads in the pool.
    pub fn threads(&self) -> usize {
        unsafe { fu_pool_count_threads(self.inner) }
    }

    /// Returns the number of thread colocations in the pool.
    ///
    /// Colocations group threads by NUMA domain, QoS level, and cache hierarchy.
    /// This information is useful for NUMA-aware load balancing and memory allocation.
    pub fn colocations(&self) -> usize {
        unsafe { fu_pool_count_colocations(self.inner) }
    }

    /// Returns the number of threads in a specific colocation.
    ///
    /// This method is useful for NUMA-aware load balancing, allowing you to understand
    /// how many threads are available in each colocation group.
    ///
    /// # Arguments
    ///
    /// * `colocation_index` - The colocation to query (0-based)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let pool = spawn(8);
    /// let total_colocations = pool.colocations();
    ///
    /// for colocation_index in 0..total_colocations {
    ///     let thread_count = pool.count_threads_in(colocation_index);
    ///     println!("Colocation {} has {} threads", colocation_index, thread_count);
    /// }
    /// ```
    pub fn count_threads_in(&self, colocation_index: usize) -> usize {
        unsafe { fu_pool_count_threads_in(self.inner, colocation_index) }
    }

    /// Converts a global thread index to a local thread index within a colocation.
    ///
    /// This is useful for distributed thread pools where threads are grouped into
    /// colocations (NUMA nodes or QoS levels). The local index can be used for
    /// per-colocation data structures or algorithms.
    ///
    /// # Arguments
    ///
    /// * `global_thread_index` - The global thread index to convert
    /// * `colocation_index` - The colocation to get the local index for
    ///
    /// # Returns
    ///
    /// The local thread index within the specified colocation.
    pub fn locate_thread_in(&self, global_thread_index: usize, colocation_index: usize) -> usize {
        unsafe { fu_pool_locate_thread_in(self.inner, global_thread_index, colocation_index) }
    }

    /// Transitions worker threads to a power-saving sleep state.
    ///
    /// This function places worker threads into a low-power sleep state when no work
    /// is available for extended periods. Threads will periodically check for new work
    /// at the specified interval.
    ///
    /// # Arguments
    ///
    /// * `micros` - Wake-up check interval in microseconds, must be > 0
    ///
    /// # Safety
    ///
    /// This function is **not thread-safe** and should only be called between task batches
    /// when no parallel operations are in progress.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// // Process a batch of work
    /// pool.for_n(1000, |prong| {
    ///     // Do some work...
    ///     std::hint::black_box(prong.task_index * 2);
    /// });
    ///
    /// // Put threads to sleep between batches to save power
    /// // Check for new work every 10 milliseconds
    /// pool.sleep(10_000); // 10,000 microseconds = 10ms
    ///
    /// // Process another batch
    /// pool.for_n(500, |prong| {
    ///     std::hint::black_box(prong.task_index * 3);
    /// });
    /// ```
    pub fn sleep(&mut self, micros: usize) {
        unsafe {
            fu_pool_sleep(self.inner, micros);
        }
    }

    /// Executes a function on each thread of the pool, returning a closure object.
    ///
    /// This operation provides explicit control over broadcast and join phases,
    /// allowing you to start work on threads and then wait for completion separately.
    ///
    /// # Arguments
    ///
    /// * `function` - Closure to execute on each thread, receiving (thread_index, colocation_index)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// {
    ///     let _op = pool.for_threads(|thread_index, colocation_index| {
    ///         println!("Thread {} on colocation {}", thread_index, colocation_index);
    ///         // Simulate some work
    ///         for i in 0..1000 {
    ///             std::hint::black_box(i * thread_index);
    ///         }
    ///     });
    ///     // Work executes when _op is dropped
    /// }
    /// ```
    pub fn for_threads<F>(&mut self, function: F) -> ForThreadsOperation<F>
    where
        F: Fn(usize, usize) + Sync,
    {
        ForThreadsOperation::new(self, function)
    }

    /// Distributes `n` similar duration calls between threads by individual indices.
    ///
    /// Uses static load balancing where each thread gets a predetermined set of tasks.
    /// This is optimal when all tasks have similar execution time.
    ///
    /// # Arguments
    ///
    /// * `n` - Total number of tasks to distribute
    /// * `function` - Closure executed for each task, receiving a `Prong` with task metadata
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// pool.for_n(1000, |prong| {
    ///     // Simulate computation based on task index
    ///     let result = prong.task_index * prong.task_index;
    ///     std::hint::black_box(result); // Prevent optimization
    /// });
    /// ```
    pub fn for_n<F>(&mut self, n: usize, function: F) -> ForNOperation<F>
    where
        F: Fn(Prong) + Sync,
    {
        ForNOperation {
            pool: self,
            n,
            function,
        }
    }

    /// Executes `n` uneven tasks on all threads, greedily stealing work.
    ///
    /// Uses dynamic load balancing with work-stealing. Threads that finish their
    /// assigned tasks early will steal work from busy threads. This is optimal
    /// when task execution times vary significantly.
    ///
    /// # Arguments
    ///
    /// * `n` - Total number of tasks to distribute
    /// * `function` - Closure executed for each task, receiving a `Prong` with task metadata
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// pool.for_n_dynamic(100, |prong| {
    ///     // Simulate variable work duration - some tasks take longer
    ///     let iterations = if prong.task_index % 10 == 0 { 10000 } else { 100 };
    ///     for i in 0..iterations {
    ///         std::hint::black_box(prong.task_index * i);
    ///     }
    /// });
    /// ```
    pub fn for_n_dynamic<F>(&mut self, n: usize, function: F) -> ForNDynamicOperation<F>
    where
        F: Fn(Prong) + Sync,
    {
        ForNDynamicOperation {
            pool: self,
            n,
            function,
        }
    }

    /// Distributes `n` similar duration calls between threads in slices.
    ///
    /// Instead of individual task assignment, this method groups tasks into
    /// contiguous slices and assigns each slice to a thread. This reduces
    /// per-task overhead and improves cache locality.
    ///
    /// # Arguments
    ///
    /// * `n` - Total number of tasks to distribute
    /// * `function` - Closure executed for each slice, receiving a `Prong` (with first task index) and slice size
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// pool.for_slices(1000, |prong, count| {
    ///     let start_index = prong.task_index;
    ///     
    ///     // Process the slice - each thread gets a contiguous range
    ///     for i in 0..count {
    ///         let global_index = start_index + i;
    ///         let result = global_index * global_index;
    ///         std::hint::black_box(result);
    ///     }
    ///     
    ///     println!("Thread {} processed slice [{}, {})",
    ///              prong.thread_index, start_index, start_index + count);
    /// });
    /// ```
    pub fn for_slices<F>(&mut self, n: usize, function: F) -> ForSlicesOperation<F>
    where
        F: Fn(Prong, usize) + Sync,
    {
        ForSlicesOperation {
            pool: self,
            n,
            function,
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        unsafe {
            fu_pool_terminate(self.inner);
            fu_pool_delete(self.inner);
        }
    }
}

/// Result of a NUMA allocation containing both the allocated pointer and metadata.
#[derive(Debug)]
pub struct AllocationResult {
    ptr: NonNull<u8>,
    allocated_bytes: usize,
    bytes_per_page: usize,
    numa_node: usize,
}

impl AllocationResult {
    /// Returns the allocated memory as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.allocated_bytes) }
    }

    /// Returns the allocated memory as an immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.allocated_bytes) }
    }

    /// Returns the raw pointer to the allocated memory.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the number of bytes actually allocated (may be larger than requested).
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Returns the page size used for this allocation.
    pub fn bytes_per_page(&self) -> usize {
        self.bytes_per_page
    }

    /// Returns the NUMA node this memory was allocated on.
    pub fn numa_node(&self) -> usize {
        self.numa_node
    }

    /// Converts a typed slice into the allocation's memory space.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `T` has the correct alignment for the allocated memory
    /// - The allocation is large enough to hold the requested number of `T` elements
    /// - The memory is properly initialized before use
    pub unsafe fn as_mut_slice_of<T>(&mut self) -> &mut [T] {
        let element_size = core::mem::size_of::<T>();
        let element_count = self.allocated_bytes / element_size;
        slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, element_count)
    }

    /// Converts a typed slice into the allocation's memory space (immutable).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `T` has the correct alignment for the allocated memory
    /// - The allocation contains valid data of type `T`
    pub unsafe fn as_slice_of<T>(&self) -> &[T] {
        let element_size = core::mem::size_of::<T>();
        let element_count = self.allocated_bytes / element_size;
        slice::from_raw_parts(self.ptr.as_ptr() as *const T, element_count)
    }
}

impl Drop for AllocationResult {
    fn drop(&mut self) {
        unsafe {
            fu_free(
                self.numa_node,
                self.ptr.as_ptr() as *mut c_void,
                self.allocated_bytes,
            );
        }
    }
}

// Safety: AllocationResult can be sent between threads since it owns its memory
unsafe impl Send for AllocationResult {}
// Safety: AllocationResult can be shared between threads with proper synchronization
unsafe impl Sync for AllocationResult {}

/// NUMA-aware memory allocator pinned to a specific NUMA node.
///
/// This allocator provides efficient memory allocation on a specific NUMA node,
/// which is beneficial for performance in multi-socket systems where memory access
/// latency varies based on the physical location of memory relative to the CPU.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
/// let allocator = PinnedAllocator::new(0).expect("Failed to create allocator for NUMA node 0");
/// let allocation = allocator.allocate(1024).expect("Failed to allocate 1024 bytes");
///
/// // Access the allocated memory
/// let memory_slice = allocation.as_slice();
/// assert_eq!(memory_slice.len(), 1024);
/// println!("Allocated {} bytes on NUMA node {}",
///          allocation.allocated_bytes(), allocation.numa_node());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PinnedAllocator {
    numa_node: usize,
}

impl PinnedAllocator {
    /// Creates a new allocator pinned to the specified NUMA node.
    ///
    /// # Arguments
    ///
    /// * `numa_node` - The NUMA node index (0-based) to pin allocations to
    ///
    /// # Errors
    ///
    /// Returns `None` if the NUMA node index is invalid (>= available NUMA nodes).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// // Create allocator for the first NUMA node
    /// let allocator = PinnedAllocator::new(0).expect("NUMA node 0 should be available");
    ///
    /// // Check if a specific NUMA node exists
    /// let numa_count = count_numa_nodes();
    /// if numa_count > 1 {
    ///     let allocator2 = PinnedAllocator::new(1).expect("NUMA node 1 should be available");
    ///     println!("Created allocator for NUMA node: {}", allocator2.numa_node());
    /// }
    /// ```
    pub fn new(numa_node: usize) -> Option<Self> {
        if numa_node >= count_numa_nodes() {
            return None;
        }

        Some(Self { numa_node })
    }

    /// Returns the NUMA node this allocator is pinned to.
    pub fn numa_node(&self) -> usize {
        self.numa_node
    }

    /// Returns the volume of huge pages available on this allocator's NUMA node.
    ///
    /// This method queries the operating system for the amount of huge page memory
    /// available for allocation on this NUMA node, which is useful for planning
    /// large memory allocations.
    ///
    /// # Returns
    ///
    /// The number of bytes of huge page memory available, or 0 if huge pages
    /// are not available on this NUMA node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let huge_page_volume = allocator.volume_huge_pages();
    ///
    /// if huge_page_volume > 0 {
    ///     println!("NUMA node 0 has {} bytes of huge pages available", huge_page_volume);
    /// } else {
    ///     println!("No huge pages available on NUMA node 0");
    /// }
    /// ```
    pub fn volume_huge_pages(&self) -> usize {
        unsafe { fu_volume_huge_pages(self.numa_node) }
    }

    /// Returns the volume of any pages (huge or regular) available on this allocator's NUMA node.
    pub fn volume_any_pages(&self) -> usize {
        unsafe { fu_volume_any_pages(self.numa_node) }
    }

    /// Allocates memory with at least the requested size on this allocator's NUMA node.
    ///
    /// Returns both the actual allocated size and page size information, which can be
    /// useful for optimizing memory access patterns.
    ///
    /// # Arguments
    ///
    /// * `minimum_bytes` - The minimum number of bytes to allocate
    ///
    /// # Errors
    ///
    /// Returns `None` if allocation fails or if `minimum_bytes` is 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let allocation = allocator.allocate_at_least(1024).expect("Failed to allocate memory");
    ///
    /// println!("Requested 1024 bytes, got {} bytes on {} byte pages",
    ///          allocation.allocated_bytes(), allocation.bytes_per_page());
    ///
    /// // The allocation might be larger than requested due to page alignment
    /// assert!(allocation.allocated_bytes() >= 1024);
    ///
    /// // Access the memory as a byte slice
    /// let memory = allocation.as_slice();
    /// println!("Can access {} bytes of memory", memory.len());
    /// ```
    pub fn allocate_at_least(&self, minimum_bytes: usize) -> Option<AllocationResult> {
        if minimum_bytes == 0 {
            return None;
        }

        let mut allocated_bytes = 0usize;
        let mut bytes_per_page = 0usize;

        unsafe {
            let ptr = fu_allocate_at_least(
                self.numa_node,
                minimum_bytes,
                &mut allocated_bytes as *mut usize,
                &mut bytes_per_page as *mut usize,
            );

            if ptr.is_null() || allocated_bytes == 0 {
                return None;
            }

            Some(AllocationResult {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                allocated_bytes,
                bytes_per_page,
                numa_node: self.numa_node,
            })
        }
    }

    /// Allocates exactly the requested number of bytes on this allocator's NUMA node.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The exact number of bytes to allocate
    ///
    /// # Errors
    ///
    /// Returns `None` if allocation fails or if `bytes` is 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let allocation = allocator.allocate(1024).expect("Failed to allocate memory");
    /// assert_eq!(allocation.allocated_bytes(), 1024);
    ///
    /// // Write some data to the allocated memory
    /// let mut allocation = allocation; // Make mutable
    /// let memory = allocation.as_mut_slice();
    /// memory[0] = 42;
    /// memory[1023] = 255;
    ///
    /// // Verify the data was written
    /// assert_eq!(memory[0], 42);
    /// assert_eq!(memory[1023], 255);
    /// ```
    pub fn allocate(&self, bytes: usize) -> Option<AllocationResult> {
        if bytes == 0 {
            return None;
        }

        unsafe {
            let ptr = fu_allocate(self.numa_node, bytes);

            if ptr.is_null() {
                return None;
            }

            Some(AllocationResult {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                allocated_bytes: bytes,
                bytes_per_page: 0, // Not provided by fu_allocate
                numa_node: self.numa_node,
            })
        }
    }

    /// Allocates memory for a specific number of elements of type T.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of elements to allocate space for
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let mut allocation = allocator.allocate_for::<u64>(100).expect("Failed to allocate");
    ///
    /// // Verify the allocation size first
    /// assert_eq!(allocation.allocated_bytes(), 100 * std::mem::size_of::<u64>());
    ///
    /// // Access as typed slice
    /// let slice = unsafe { allocation.as_mut_slice_of::<u64>() };
    /// slice[0] = 42;
    /// slice[99] = 12345;
    ///
    /// // Read back the values
    /// assert_eq!(slice[0], 42);
    /// assert_eq!(slice[99], 12345);
    /// ```
    pub fn allocate_for<T>(&self, count: usize) -> Option<AllocationResult> {
        let bytes = count.checked_mul(core::mem::size_of::<T>())?;
        self.allocate(bytes)
    }

    /// Allocates memory for at least the specified number of elements of type T.
    ///
    /// This function may allocate more elements than requested for optimal page alignment.
    ///
    /// # Arguments
    ///
    /// * `min_count` - The minimum number of elements to allocate space for
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let mut allocation = allocator.allocate_for_at_least::<u32>(1000).expect("Failed to allocate");
    /// let actual_count = allocation.allocated_bytes() / std::mem::size_of::<u32>();
    /// println!("Requested {} u32s, got space for {} u32s", 1000, actual_count);
    ///
    /// // The allocation provides at least the requested number of elements
    /// assert!(actual_count >= 1000);
    ///
    /// // Initialize the allocated memory
    /// let slice = unsafe { allocation.as_mut_slice_of::<u32>() };
    /// for i in 0..1000 {
    ///     slice[i] = i as u32;
    /// }
    ///
    /// // Verify initialization
    /// assert_eq!(slice[0], 0);
    /// assert_eq!(slice[999], 999);
    /// ```
    pub fn allocate_for_at_least<T>(&self, min_count: usize) -> Option<AllocationResult> {
        let min_bytes = min_count.checked_mul(core::mem::size_of::<T>())?;
        self.allocate_at_least(min_bytes)
    }
}

/// Creates an allocator for the first available NUMA node (typically node 0).
///
/// This is a convenience function for systems where NUMA awareness is desired
/// but the specific node doesn't matter.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// let allocator = default_numa_allocator().expect("No NUMA nodes available");
/// let allocation = allocator.allocate(1024).expect("Failed to allocate");
///
/// // The default allocator uses NUMA node 0
/// assert_eq!(allocation.numa_node(), 0);
///
/// // For more control, create specific NUMA allocators
/// let numa_count = count_numa_nodes();
/// println!("System has {} NUMA nodes available", numa_count);
///
/// if numa_count > 1 {
///     let allocator_node1 = PinnedAllocator::new(1).expect("NUMA node 1 available");
///     let allocation2 = allocator_node1.allocate(2048).expect("Failed to allocate on node 1");
///     assert_eq!(allocation2.numa_node(), 1);
/// }
/// ```
pub fn default_numa_allocator() -> Option<PinnedAllocator> {
    PinnedAllocator::new(0)
}

/// A Vec-like container that uses NUMA-aware pinned memory allocation.
///
/// `PinnedVec<T>` provides a dynamic array similar to `std::vec::Vec<T>` but
/// allocates memory on a specific NUMA node for optimal performance in
/// multi-socket systems. The memory is allocated using the `PinnedAllocator`
/// and automatically manages growth and shrinkage.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// // Create a vector on NUMA node 0
/// let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
/// let mut vec = PinnedVec::<u64>::new_in(allocator);
/// 
/// // Add elements
/// vec.push(42).expect("Failed to push");
/// vec.push(100).expect("Failed to push");
/// 
/// // Access elements
/// assert_eq!(vec.len(), 2);
/// assert_eq!(vec[0], 42);
/// assert_eq!(vec[1], 100);
/// 
/// // Iterate over elements
/// for (i, &value) in vec.iter().enumerate() {
///     println!("Element {}: {}", i, value);
/// }
/// ```
pub struct PinnedVec<T> {
    allocator: PinnedAllocator,
    allocation: Option<AllocationResult>,
    len: usize,
    capacity: usize,
    _phantom: core::marker::PhantomData<T>,
}

impl<T> PinnedVec<T> {
    /// Creates a new empty `PinnedVec` using the specified allocator.
    ///
    /// # Arguments
    ///
    /// * `allocator` - The `PinnedAllocator` to use for memory allocation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    /// 
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
    /// let vec = PinnedVec::<i32>::new_in(allocator);
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 0);
    /// ```
    pub fn new_in(allocator: PinnedAllocator) -> Self {
        Self {
            allocator,
            allocation: None,
            len: 0,
            capacity: 0,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Creates a new `PinnedVec` with the specified capacity using the given allocator.
    ///
    /// # Arguments
    ///
    /// * `allocator` - The `PinnedAllocator` to use for memory allocation
    /// * `capacity` - The initial capacity to allocate
    ///
    /// # Errors
    ///
    /// Returns `None` if allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    /// 
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
    /// let vec = PinnedVec::<i32>::with_capacity_in(allocator, 100).expect("Failed to create PinnedVec");
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 100);
    /// ```
    pub fn with_capacity_in(allocator: PinnedAllocator, capacity: usize) -> Option<Self> {
        let mut vec = Self {
            allocator,
            allocation: None,
            len: 0,
            capacity: 0,
            _phantom: core::marker::PhantomData,
        };
        
        if capacity > 0 {
            vec.reserve(capacity).ok()?;
        }
        
        Some(vec)
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the NUMA node this vector's memory is allocated on.
    pub fn numa_node(&self) -> usize {
        self.allocator.numa_node()
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// # Arguments
    ///
    /// * `additional` - The number of additional elements to reserve space for
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    /// 
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
    /// let mut vec = PinnedVec::<i32>::new_in(allocator);
    /// vec.reserve(10).expect("Failed to reserve");
    /// assert!(vec.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, additional: usize) -> Result<(), &'static str> {
        let needed_capacity = self.len.checked_add(additional).ok_or("Capacity overflow")?;
        if needed_capacity <= self.capacity {
            return Ok(());
        }

        let new_capacity = needed_capacity.max(self.capacity * 2).max(4);
        self.grow_to(new_capacity)
    }

    /// Grows the vector to the specified capacity.
    fn grow_to(&mut self, new_capacity: usize) -> Result<(), &'static str> {
        if new_capacity <= self.capacity {
            return Ok(());
        }

        let new_allocation = self.allocator
            .allocate_for::<T>(new_capacity)
            .ok_or("Failed to allocate memory")?;

        if let Some(old_allocation) = self.allocation.take() {
            // Copy existing elements to new allocation
            unsafe {
                let old_ptr = old_allocation.as_ptr() as *const T;
                let new_ptr = new_allocation.as_ptr() as *mut T;
                core::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len);
            }
        }

        self.allocation = Some(new_allocation);
        self.capacity = new_capacity;
        Ok(())
    }

    /// Appends an element to the back of the vector.
    ///
    /// # Arguments
    ///
    /// * `value` - The element to append
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails when growing the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    /// 
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
    /// let mut vec = PinnedVec::<i32>::new_in(allocator);
    /// vec.push(42).expect("Failed to push");
    /// assert_eq!(vec.len(), 1);
    /// assert_eq!(vec[0], 42);
    /// ```
    pub fn push(&mut self, value: T) -> Result<(), &'static str> {
        if self.len >= self.capacity {
            self.reserve(1)?;
        }

        unsafe {
            let ptr = self.as_mut_ptr().add(self.len);
            core::ptr::write(ptr, value);
        }
        self.len += 1;
        Ok(())
    }

    /// Removes the last element from the vector and returns it.
    ///
    /// # Returns
    ///
    /// The last element, or `None` if the vector is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    /// 
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
    /// let mut vec = PinnedVec::<i32>::new_in(allocator);
    /// vec.push(42).expect("Failed to push");
    /// assert_eq!(vec.pop(), Some(42));
    /// assert_eq!(vec.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        unsafe {
            let ptr = self.as_mut_ptr().add(self.len);
            Some(core::ptr::read(ptr))
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    /// 
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
    /// let mut vec = PinnedVec::<i32>::new_in(allocator);
    /// vec.push(42).expect("Failed to push");
    /// vec.clear();
    /// assert_eq!(vec.len(), 0);
    /// ```
    pub fn clear(&mut self) {
        unsafe {
            let ptr = self.as_mut_ptr();
            for i in 0..self.len {
                core::ptr::drop_in_place(ptr.add(i));
            }
        }
        self.len = 0;
    }

    /// Returns a raw pointer to the vector's buffer.
    pub fn as_ptr(&self) -> *const T {
        match &self.allocation {
            Some(alloc) => alloc.as_ptr() as *const T,
            None => core::ptr::NonNull::dangling().as_ptr(),
        }
    }

    /// Returns a mutable raw pointer to the vector's buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match &self.allocation {
            Some(alloc) => alloc.as_ptr() as *mut T,
            None => core::ptr::NonNull::dangling().as_ptr(),
        }
    }

    /// Returns a slice containing the entire vector.
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Returns a mutable slice containing the entire vector.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    /// Returns an iterator over the vector.
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the vector.
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Inserts an element at position `index`, shifting all elements after it to the right.
    ///
    /// # Arguments
    ///
    /// * `index` - The position to insert at
    /// * `element` - The element to insert
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails when growing the vector.
    pub fn insert(&mut self, index: usize, element: T) -> Result<(), &'static str> {
        if index > self.len {
            panic!("insertion index (is {}) should be <= len (is {})", index, self.len);
        }

        if self.len >= self.capacity {
            self.reserve(1)?;
        }

        unsafe {
            let ptr = self.as_mut_ptr();
            core::ptr::copy(ptr.add(index), ptr.add(index + 1), self.len - index);
            core::ptr::write(ptr.add(index), element);
        }
        self.len += 1;
        Ok(())
    }

    /// Removes and returns the element at position `index`, shifting all elements after it to the left.
    ///
    /// # Arguments
    ///
    /// * `index` - The position to remove from
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    pub fn remove(&mut self, index: usize) -> T {
        if index >= self.len {
            panic!("removal index (is {}) should be < len (is {})", index, self.len);
        }

        unsafe {
            let ptr = self.as_mut_ptr();
            let result = core::ptr::read(ptr.add(index));
            core::ptr::copy(ptr.add(index + 1), ptr.add(index), self.len - index - 1);
            self.len -= 1;
            result
        }
    }

    /// Extend the vector by cloning elements from a slice.
    ///
    /// # Arguments
    ///
    /// * `other` - The slice to copy elements from
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails when growing the vector.
    pub fn extend_from_slice(&mut self, other: &[T]) -> Result<(), &'static str> 
    where 
        T: Clone,
    {
        self.reserve(other.len())?;
        for item in other {
            self.push(item.clone())?;
        }
        Ok(())
    }
}

impl<T> core::ops::Index<usize> for PinnedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> core::ops::IndexMut<usize> for PinnedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<T> Drop for PinnedVec<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

unsafe impl<T: Send> Send for PinnedVec<T> {}
unsafe impl<T: Sync> Sync for PinnedVec<T> {}

/// A thread-safe wrapper around raw pointers for sharing read-only data across threads.
///
/// This type is designed for scenarios where you need to share immutable data
/// across async tasks or threads, particularly when the standard borrowing rules
/// would prevent such sharing. The caller is responsible for ensuring that:
/// - The pointed-to data remains valid for the lifetime of use
/// - The data is not modified while being accessed through `SyncConstPtr`
///
/// # Safety
///
/// This type is marked as `Send + Sync` but requires careful usage:
/// - Only use with data that won't be modified during the lifetime of the pointer
/// - Ensure the pointed-to data outlives all uses of the `SyncConstPtr`
/// - The `get` method is unsafe and requires the caller to ensure bounds checking
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
/// 
/// let data = vec![1, 2, 3, 4, 5];
/// let sync_ptr = SyncConstPtr::new(data.as_ptr());
/// 
/// // Safe to use in async contexts
/// let value = unsafe { sync_ptr.get(0) };
/// assert_eq!(*value, 1);
/// ```
#[derive(Copy, Clone, Debug)]
pub struct SyncConstPtr<T> {
    ptr: *const T,
}

impl<T> SyncConstPtr<T> {
    /// Creates a new `SyncConstPtr` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The pointer is valid for the intended usage duration
    /// - The pointed-to data will not be modified during use
    /// - The pointer is properly aligned for type `T`
    pub fn new(ptr: *const T) -> Self {
        Self { ptr }
    }

    /// Gets a reference to the element at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The index is within bounds of the allocated data
    /// - The data at the index is properly initialized
    /// - The data remains valid for the lifetime of the returned reference
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to access
    ///
    /// # Returns
    ///
    /// A reference to the element at the given index.
    pub unsafe fn get(&self, index: usize) -> &T {
        &*self.ptr.add(index)
    }

    /// Returns the raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

unsafe impl<T> Send for SyncConstPtr<T> {}
unsafe impl<T> Sync for SyncConstPtr<T> {}

/// Operation object for parallel thread execution with explicit broadcast/join control.
pub struct ForThreadsOperation<'a, F>
where
    F: Fn(usize, usize) + Sync,
{
    pool: &'a mut ThreadPool,
    function: F,
    did_broadcast: bool,
    did_join: bool,
}

impl<'a, F> ForThreadsOperation<'a, F>
where
    F: Fn(usize, usize) + Sync,
{
    /// Create a new ForThreadsOperation (internal use by ThreadPool)
    pub(crate) fn new(pool: &'a mut ThreadPool, function: F) -> Self {
        Self {
            pool,
            function,
            did_broadcast: false,
            did_join: false,
        }
    }

    /// Broadcast the work to all threads without waiting for completion.
    /// This is safe to call multiple times - subsequent calls are no-ops.
    pub fn broadcast(&mut self) {
        if self.did_broadcast {
            return; // No need to broadcast again
        }

        extern "C" fn trampoline<F>(ctx: *mut c_void, thread_index: usize, colocation_index: usize)
        where
            F: Fn(usize, usize) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(thread_index, colocation_index);
        }

        unsafe {
            let ctx = &self.function as *const F as *mut c_void;
            fu_pool_unsafe_for_threads(self.pool.inner, trampoline::<F>, ctx);
            self.did_broadcast = true;
        }
    }

    /// Wait for all threads to complete their work.
    /// If broadcast() hasn't been called yet, this will call it first.
    pub fn join(&mut self) {
        if !self.did_broadcast {
            self.broadcast();
        }
        if self.did_join {
            return; // No need to join again
        }
        unsafe {
            fu_pool_unsafe_join(self.pool.inner);
            self.did_join = true;
        }
    }
}

impl<'a, F> Drop for ForThreadsOperation<'a, F>
where
    F: Fn(usize, usize) + Sync,
{
    fn drop(&mut self) {
        self.join();
    }
}

/// Operation object for parallel task execution with static load balancing.
pub struct ForNOperation<'a, F>
where
    F: Fn(Prong) + Sync,
{
    pool: &'a mut ThreadPool,
    n: usize,
    function: F,
}

impl<'a, F> Drop for ForNOperation<'a, F>
where
    F: Fn(Prong) + Sync,
{
    fn drop(&mut self) {
        extern "C" fn trampoline<F>(
            ctx: *mut c_void,
            task_index: usize,
            thread_index: usize,
            colocation_index: usize,
        ) where
            F: Fn(Prong) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(Prong {
                task_index,
                thread_index,
                colocation_index,
            });
        }

        unsafe {
            let ctx = &self.function as *const F as *mut c_void;
            fu_pool_for_n(self.pool.inner, self.n, trampoline::<F>, ctx);
        }
    }
}

/// Operation object for parallel task execution with dynamic work-stealing.
pub struct ForNDynamicOperation<'a, F>
where
    F: Fn(Prong) + Sync,
{
    pool: &'a mut ThreadPool,
    n: usize,
    function: F,
}

impl<'a, F> Drop for ForNDynamicOperation<'a, F>
where
    F: Fn(Prong) + Sync,
{
    fn drop(&mut self) {
        extern "C" fn trampoline<F>(
            ctx: *mut c_void,
            task_index: usize,
            thread_index: usize,
            colocation_index: usize,
        ) where
            F: Fn(Prong) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(Prong {
                task_index,
                thread_index,
                colocation_index,
            });
        }

        unsafe {
            let ctx = &self.function as *const F as *mut c_void;
            fu_pool_for_n_dynamic(self.pool.inner, self.n, trampoline::<F>, ctx);
        }
    }
}

/// Operation object for parallel slice execution.
pub struct ForSlicesOperation<'a, F>
where
    F: Fn(Prong, usize) + Sync,
{
    pool: &'a mut ThreadPool,
    n: usize,
    function: F,
}

impl<'a, F> Drop for ForSlicesOperation<'a, F>
where
    F: Fn(Prong, usize) + Sync,
{
    fn drop(&mut self) {
        extern "C" fn trampoline<F>(
            ctx: *mut c_void,
            first_index: usize,
            count: usize,
            thread_index: usize,
            colocation_index: usize,
        ) where
            F: Fn(Prong, usize) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(
                Prong {
                    task_index: first_index,
                    thread_index,
                    colocation_index,
                },
                count,
            );
        }

        unsafe {
            let ctx = &self.function as *const F as *mut c_void;
            fu_pool_for_slices(self.pool.inner, self.n, trampoline::<F>, ctx);
        }
    }
}

/// Spawns a pool with the specified number of threads.
pub fn spawn(threads: usize) -> ThreadPool {
    ThreadPool::try_spawn(threads).expect("Failed to spawn ThreadPool")
}

/// Spawns a named pool with the specified number of threads.
pub fn named_spawn(name: &str, threads: usize) -> ThreadPool {
    ThreadPool::try_named_spawn(name, threads).expect("Failed to spawn named ThreadPool")
}

/// Standalone function to distribute `n` similar duration calls between threads.
pub fn for_n<F>(pool: &mut ThreadPool, n: usize, function: F)
where
    F: Fn(Prong) + Sync,
{
    let _operation = pool.for_n(n, function);
    // Operation executes and joins in its destructor
}

/// Standalone function to execute `n` uneven tasks on all threads.
pub fn for_n_dynamic<F>(pool: &mut ThreadPool, n: usize, function: F)
where
    F: Fn(Prong) + Sync,
{
    let _operation = pool.for_n_dynamic(n, function);
    // Operation executes and joins in its destructor
}

/// Standalone function to distribute `n` tasks in slices.
pub fn for_slices<F>(pool: &mut ThreadPool, n: usize, function: F)
where
    F: Fn(Prong, usize) + Sync,
{
    let _operation = pool.for_slices(n, function);
    // Operation executes and joins in its destructor
}

/// Helper function to visit every element exactly once with mutable access.
pub fn for_each_prong_mut<T, F>(pool: &mut ThreadPool, data: &mut [T], function: F)
where
    T: Send + Sync,
    F: Fn(&mut T, Prong) + Sync + Send,
{
    let base_ptr = data.as_mut_ptr() as usize;
    let n = data.len();

    let _operation = pool.for_n(n, move |prong| {
        let item = unsafe { &mut *(base_ptr as *mut T).add(prong.task_index) };
        function(item, prong);
    });
}

/// Helper function to visit every element exactly once with dynamic work-stealing.
pub fn for_each_prong_mut_dynamic<T, F>(pool: &mut ThreadPool, data: &mut [T], function: F)
where
    T: Send + Sync,
    F: Fn(&mut T, Prong) + Sync + Send,
{
    let base_ptr = data.as_mut_ptr() as usize;
    let n = data.len();

    let _operation = pool.for_n_dynamic(n, move |prong| {
        let item = unsafe { &mut *(base_ptr as *mut T).add(prong.task_index) };
        function(item, prong);
    });
}

/// Splits a range of tasks into fair-sized chunks for parallel distribution.
///
/// The first `(tasks % threads)` chunks have size `ceil(tasks / threads)`.
/// The remaining chunks have size `floor(tasks / threads)`.
///
/// This ensures optimal load balancing across threads with minimal size variance.
/// See: https://lemire.me/blog/2025/05/22/dividing-an-array-into-fair-sized-chunks/
#[derive(Debug, Clone)]
pub struct IndexedSplit {
    quotient: usize,
    remainder: usize,
}

impl IndexedSplit {
    /// Creates a new indexed split for distributing tasks across threads.
    ///
    /// # Arguments
    ///
    /// * `tasks_count` - Total number of tasks to distribute
    /// * `threads_count` - Number of threads to distribute across (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `threads_count` is zero.
    pub fn new(tasks_count: usize, threads_count: usize) -> Self {
        assert!(threads_count > 0, "Threads count must be greater than zero");
        Self {
            quotient: tasks_count / threads_count,
            remainder: tasks_count % threads_count,
        }
    }

    /// Returns the range for a specific thread index.
    pub fn get(&self, thread_index: usize) -> core::ops::Range<usize> {
        let begin = self.quotient * thread_index + thread_index.min(self.remainder);
        let count = self.quotient + if thread_index < self.remainder { 1 } else { 0 };
        begin..(begin + count)
    }
}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use std::vec::Vec;

    #[inline]
    fn hw_threads() -> usize {
        count_logical_cores().max(1)
    }

    #[test]
    fn test_capabilities() {
        let caps = capabilities_string();
        std::println!("Capabilities: {:?}", caps);
        assert!(caps.is_some());
    }

    #[test]
    fn test_system_info() {
        let cores = count_logical_cores();
        let numa = count_numa_nodes();
        let colocations = count_colocations();
        let qos = count_quality_levels();

        std::println!(
            "Cores: {}, NUMA: {}, Colocations: {}, QoS: {}",
            cores,
            numa,
            colocations,
            qos
        );
        assert!(cores > 0);
    }

    #[test]
    fn test_spawn_and_basic_info() {
        let pool = spawn(2);
        assert_eq!(pool.threads(), 2);
        assert!(pool.colocations() > 0);
    }

    #[test]
    fn test_for_threads_dispatch() {
        let count_threads = hw_threads();
        let mut pool = spawn(count_threads);

        let visited: Arc<Vec<AtomicBool>> =
            Arc::new((0..count_threads).map(|_| AtomicBool::new(false)).collect());
        let visited_ref = Arc::clone(&visited);

        {
            let _op = pool.for_threads(move |thread_index, _colocation| {
                if thread_index < visited_ref.len() {
                    visited_ref[thread_index].store(true, Ordering::Relaxed);
                }
            });
        } // Operation executes in destructor

        for (i, flag) in visited.iter().enumerate() {
            assert!(
                flag.load(Ordering::Relaxed),
                "thread {} never reached the callback",
                i
            );
        }
    }

    #[test]
    fn test_for_n_static_scheduling() {
        const EXPECTED_PARTS: usize = 1_000;
        let mut pool = spawn(hw_threads());

        let visited: Arc<Vec<AtomicBool>> = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        for_n(&mut pool, EXPECTED_PARTS, move |prong| {
            let task_index = prong.task_index;
            if visited_ref[task_index].swap(true, Ordering::Relaxed) {
                duplicate_ref.store(true, Ordering::Relaxed);
            }
        });

        assert!(
            !duplicate.load(Ordering::Relaxed),
            "static scheduling produced duplicate task IDs"
        );
        for flag in visited.iter() {
            assert!(flag.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn test_for_n_dynamic_scheduling() {
        const EXPECTED_PARTS: usize = 1_000;
        let mut pool = spawn(hw_threads());

        let visited: Arc<Vec<AtomicBool>> = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        for_n_dynamic(&mut pool, EXPECTED_PARTS, move |prong| {
            let task_index = prong.task_index;
            if visited_ref[task_index].swap(true, Ordering::Relaxed) {
                duplicate_ref.store(true, Ordering::Relaxed);
            }
        });

        assert!(
            !duplicate.load(Ordering::Relaxed),
            "dynamic scheduling produced duplicate task IDs"
        );
        for flag in visited.iter() {
            assert!(flag.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn test_for_each_mut() {
        const ELEMENTS: usize = 1000;
        let mut pool = spawn(hw_threads());
        let mut data = std::vec![0u64; ELEMENTS];

        for_each_prong_mut(&mut pool, &mut data, |x, prong| {
            *x = prong.task_index as u64 * 2;
        });

        for (i, &value) in data.iter().enumerate() {
            assert_eq!(value, i as u64 * 2);
        }
    }

    #[test]
    fn test_closure_objects() {
        let mut pool = spawn(hw_threads());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ref = Arc::clone(&counter);

        // Test that the operation object properly executes on drop
        {
            let _op = pool.for_n(1000, move |_prong| {
                counter_ref.fetch_add(1, Ordering::Relaxed);
            });
        } // Operation executes here in the destructor

        // Now the operation should have completed
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_explicit_broadcast_join() {
        let mut pool = spawn(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ref = Arc::clone(&counter);

        let mut operation = pool.for_threads(move |_thread_index, _colocation| {
            counter_ref.fetch_add(1, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(10)); // Simulate work
        });

        // Broadcast work to threads but don't wait yet
        operation.broadcast();

        // Do some other work while threads are running
        thread::sleep(Duration::from_millis(5));

        // Now wait for completion
        operation.join();
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_join_without_explicit_broadcast() {
        let mut pool = spawn(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ref = Arc::clone(&counter);

        let mut operation = pool.for_threads(move |_thread_index, _colocation| {
            counter_ref.fetch_add(1, Ordering::Relaxed);
        });

        // Join without calling broadcast first - should work
        operation.join();
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_pinned_allocator_creation() {
        let numa_count = count_numa_nodes();
        assert!(numa_count > 0, "System should have at least one NUMA node");

        // Test valid NUMA node
        let allocator = PinnedAllocator::new(0).expect("NUMA node 0 should be available");
        assert_eq!(allocator.numa_node(), 0);

        // Test invalid NUMA node
        let invalid_allocator = PinnedAllocator::new(numa_count + 10);
        assert!(
            invalid_allocator.is_none(),
            "Invalid NUMA node should return None"
        );
    }

    #[test]
    fn test_basic_allocation() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let allocation = allocator
            .allocate(1024)
            .expect("Failed to allocate 1024 bytes");

        assert_eq!(allocation.allocated_bytes(), 1024);
        assert_eq!(allocation.numa_node(), 0);

        // Test that we can write to the memory
        let slice = allocation.as_slice();
        assert_eq!(slice.len(), 1024);
    }

    #[test]
    fn test_allocate_zero_bytes() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let allocation = allocator.allocate(0);
        assert!(
            allocation.is_none(),
            "Allocating 0 bytes should return None"
        );
    }

    #[test]
    fn test_allocate_at_least() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let allocation = allocator
            .allocate_at_least(1000)
            .expect("Failed to allocate at least 1000 bytes");

        assert!(allocation.allocated_bytes() >= 1000);
        assert_eq!(allocation.numa_node(), 0);

        // bytes_per_page should be set to something reasonable
        if allocation.bytes_per_page() > 0 {
            assert!(allocation.bytes_per_page() >= 512); // Reasonable minimum page size
        }
    }

    #[test]
    fn test_pinned_vec_creation() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let vec = PinnedVec::<i32>::new_in(allocator);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert_eq!(vec.numa_node(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_pinned_vec_with_capacity() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let vec = PinnedVec::<i32>::with_capacity_in(allocator, 10).expect("Failed to create PinnedVec");
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 10);
        assert_eq!(vec.numa_node(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_pinned_vec_push_pop() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        
        // Test push
        vec.push(42).expect("Failed to push");
        assert_eq!(vec.len(), 1);
        assert!(!vec.is_empty());
        assert_eq!(vec[0], 42);

        vec.push(100).expect("Failed to push");
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[1], 100);

        // Test pop
        assert_eq!(vec.pop(), Some(100));
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.pop(), Some(42));
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.pop(), None);
    }

    #[test]
    fn test_pinned_vec_indexing() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        vec.push(10).expect("Failed to push");
        vec.push(20).expect("Failed to push");
        vec.push(30).expect("Failed to push");

        // Test indexing
        assert_eq!(vec[0], 10);
        assert_eq!(vec[1], 20);
        assert_eq!(vec[2], 30);

        // Test mutable indexing
        vec[1] = 25;
        assert_eq!(vec[1], 25);
    }

    #[test]
    fn test_pinned_vec_clear() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        vec.push(1).expect("Failed to push");
        vec.push(2).expect("Failed to push");
        vec.push(3).expect("Failed to push");

        assert_eq!(vec.len(), 3);
        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_pinned_vec_insert_remove() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        vec.push(1).expect("Failed to push");
        vec.push(3).expect("Failed to push");

        // Insert in the middle
        vec.insert(1, 2).expect("Failed to insert");
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);

        // Remove from the middle
        let removed = vec.remove(1);
        assert_eq!(removed, 2);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 3);
    }

    #[test]
    fn test_pinned_vec_reserve() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        assert_eq!(vec.capacity(), 0);

        vec.reserve(10).expect("Failed to reserve");
        assert!(vec.capacity() >= 10);
        assert_eq!(vec.len(), 0);

        // Adding elements shouldn't require new allocation
        for i in 0..10 {
            vec.push(i).expect("Failed to push");
        }
        assert_eq!(vec.len(), 10);
    }

    #[test]
    fn test_pinned_vec_extend_from_slice() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        let data = [1, 2, 3, 4, 5];
        
        vec.extend_from_slice(&data).expect("Failed to extend");
        assert_eq!(vec.len(), 5);
        for (i, &value) in data.iter().enumerate() {
            assert_eq!(vec[i], value);
        }
    }

    #[test]
    fn test_pinned_vec_iterators() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        for i in 0..5 {
            vec.push(i).expect("Failed to push");
        }

        // Test immutable iterator
        let collected: Vec<i32> = vec.iter().copied().collect();
        let expected = Vec::from([0, 1, 2, 3, 4]);
        assert_eq!(collected, expected);

        // Test mutable iterator
        for value in vec.iter_mut() {
            *value *= 2;
        }
        assert_eq!(vec[0], 0);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 4);
        assert_eq!(vec[3], 6);
        assert_eq!(vec[4], 8);
    }

    #[test]
    fn test_pinned_vec_slices() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        for i in 0..5 {
            vec.push(i).expect("Failed to push");
        }

        // Test as_slice
        let slice = vec.as_slice();
        assert_eq!(slice.len(), 5);
        assert_eq!(slice[2], 2);

        // Test as_mut_slice
        let mut_slice = vec.as_mut_slice();
        mut_slice[2] = 99;
        assert_eq!(vec[2], 99);
    }

    #[test]
    fn test_pinned_vec_growth() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create allocator");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        
        // Push many elements to test growth
        for i in 0..100 {
            vec.push(i).expect("Failed to push");
        }
        
        assert_eq!(vec.len(), 100);
        for i in 0..100 {
            assert_eq!(vec[i], i as i32);
        }
    }

    #[test]
    fn test_pinned_vec_invalid_numa_node() {
        let numa_count = count_numa_nodes();
        let allocator = PinnedAllocator::new(numa_count + 1);
        assert!(allocator.is_none());
    }

    #[test]
    fn test_sync_const_ptr() {
        let data = Vec::from([1, 2, 3, 4, 5]);
        let sync_ptr = SyncConstPtr::new(data.as_ptr());
        
        unsafe {
            assert_eq!(*sync_ptr.get(0), 1);
            assert_eq!(*sync_ptr.get(2), 3);
            assert_eq!(*sync_ptr.get(4), 5);
        }
        
        assert_eq!(sync_ptr.as_ptr(), data.as_ptr());
    }

    #[test]
    fn test_sync_const_ptr_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        
        assert_send::<SyncConstPtr<i32>>();
        assert_sync::<SyncConstPtr<i32>>();
    }

    #[test]
    fn test_pinned_vec_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        
        assert_send::<PinnedVec<i32>>();
        assert_sync::<PinnedVec<i32>>();
    }
}
