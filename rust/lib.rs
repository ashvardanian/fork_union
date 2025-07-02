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

use core::ffi::{c_char, c_int, c_void};
use core::ptr::NonNull;
use core::slice;

/// Describes a portion of work executed on a specific thread.
#[derive(Copy, Clone, Debug)]
pub struct Prong {
    pub task_index: usize,
    pub thread_index: usize,
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
    fn fu_capabilities_string() -> *const c_char;

    // Systems metadata
    fn fu_count_logical_cores() -> usize;
    fn fu_count_colocations() -> usize;
    fn fu_count_numa_nodes() -> usize;
    fn fu_count_quality_levels() -> usize;

    // Core thread pool operations
    fn fu_pool_new() -> *mut c_void;
    fn fu_pool_delete(pool: *mut c_void);
    fn fu_pool_spawn(pool: *mut c_void, threads: usize, exclusivity: c_int) -> c_int;
    fn fu_pool_terminate(pool: *mut c_void);
    fn fu_pool_count_threads(pool: *mut c_void) -> usize;
    fn fu_pool_count_colocations(pool: *mut c_void) -> usize;
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

    fn fu_allocate_at_least(
        numa_node_index: usize,
        minimum_bytes: usize,
        allocated_bytes: *mut usize,
        bytes_per_page: *mut usize,
    ) -> *mut c_void;
    fn fu_allocate(numa_node_index: usize, bytes: usize) -> *mut c_void;
    fn fu_free(numa_node_index: usize, pointer: *mut c_void, bytes: usize);

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

/// Returns the number of logical CPU cores available on the system.
pub fn count_logical_cores() -> usize {
    unsafe { fu_count_logical_cores() }
}

/// Returns the number of NUMA nodes available on the system.
pub fn count_numa_nodes() -> usize {
    unsafe { fu_count_numa_nodes() }
}

/// Returns the number of distinct thread colocations available.
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
/// pool.for_threads(|thread_idx, colocation_idx| {
///     println!("Thread {} on colocation {}", thread_idx, colocation_idx);
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
        if threads == 0 {
            return Err(Error::InvalidParameter);
        }

        unsafe {
            let inner = fu_pool_new();
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

    /// Returns the number of threads in the pool.
    pub fn threads(&self) -> usize {
        unsafe { fu_pool_count_threads(self.inner) }
    }

    /// Returns the number of thread colocations in the pool.
    pub fn colocations(&self) -> usize {
        unsafe { fu_pool_count_colocations(self.inner) }
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
    ///     let _op = pool.for_threads(|thread_idx, colocation_idx| {
    ///         println!("Thread {} on colocation {}", thread_idx, colocation_idx);
    ///         // Simulate some work
    ///         for i in 0..1000 {
    ///             std::hint::black_box(i * thread_idx);
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
    ///     let start_idx = prong.task_index;
    ///     
    ///     // Process the slice - each thread gets a contiguous range
    ///     for i in 0..count {
    ///         let global_idx = start_idx + i;
    ///         let result = global_idx * global_idx;
    ///         std::hint::black_box(result);
    ///     }
    ///     
    ///     println!("Thread {} processed slice [{}, {})",
    ///              prong.thread_index, start_idx, start_idx + count);
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
}
