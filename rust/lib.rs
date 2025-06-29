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

use core::ptr;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::ffi::CStr;

/// Describes a portion of work executed on a specific thread.
#[derive(Copy, Clone, Debug)]
pub struct Prong {
    pub thread_index: usize,
    pub task_index: usize,
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
    fn fu_capabilities_string() -> *const core::ffi::c_char;
    fn fu_count_logical_cores() -> usize;
    fn fu_count_colocations() -> usize;
    fn fu_count_numa_nodes() -> usize;
    fn fu_count_quality_levels() -> usize;

    fn fu_pool_new() -> *mut core::ffi::c_void;
    fn fu_pool_delete(pool: *mut core::ffi::c_void);
    fn fu_pool_spawn(
        pool: *mut core::ffi::c_void,
        threads: usize,
        exclusivity: core::ffi::c_int,
    ) -> core::ffi::c_int;
    fn fu_pool_terminate(pool: *mut core::ffi::c_void);
    fn fu_pool_count_threads(pool: *mut core::ffi::c_void) -> usize;
    fn fu_pool_count_colocations(pool: *mut core::ffi::c_void) -> usize;

    fn fu_pool_unsafe_for_threads(
        pool: *mut core::ffi::c_void,
        callback: extern "C" fn(*mut core::ffi::c_void, usize, usize),
        context: *mut core::ffi::c_void,
    );
    fn fu_pool_unsafe_for_n(
        pool: *mut core::ffi::c_void,
        n: usize,
        callback: extern "C" fn(*mut core::ffi::c_void, usize, usize, usize),
        context: *mut core::ffi::c_void,
    );
    fn fu_pool_unsafe_for_n_dynamic(
        pool: *mut core::ffi::c_void,
        n: usize,
        callback: extern "C" fn(*mut core::ffi::c_void, usize, usize, usize),
        context: *mut core::ffi::c_void,
    );
    fn fu_pool_unsafe_for_slices(
        pool: *mut core::ffi::c_void,
        n: usize,
        callback: extern "C" fn(*mut core::ffi::c_void, usize, usize, usize, usize),
        context: *mut core::ffi::c_void,
    );
    fn fu_pool_unsafe_join(pool: *mut core::ffi::c_void);
}

const FU_CALLER_INCLUSIVE: core::ffi::c_int = 0;
const FU_CALLER_EXCLUSIVE: core::ffi::c_int = 1;

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
#[cfg(not(feature = "std"))]
pub fn capabilities_string_ptr() -> *const core::ffi::c_char {
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
    Inclusive,
    /// The calling thread only coordinates, doesn't execute tasks (spawns N workers)
    Exclusive,
}

impl CallerExclusivity {
    fn to_c_int(self) -> core::ffi::c_int {
        match self {
            CallerExclusivity::Inclusive => FU_CALLER_INCLUSIVE,
            CallerExclusivity::Exclusive => FU_CALLER_EXCLUSIVE,
        }
    }
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
pub struct ThreadPool {
    inner: *mut core::ffi::c_void,
}

unsafe impl Send for ThreadPool {}
unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    /// Creates a new thread pool with the specified number of threads.
    pub fn try_spawn(planned_threads: usize) -> Result<Self, Error> {
        if planned_threads == 0 {
            return Err(Error::InvalidParameter);
        }

        unsafe {
            let inner = fu_pool_new();
            if inner.is_null() {
                return Err(Error::CreationFailed);
            }

            let success = fu_pool_spawn(inner, planned_threads, FU_CALLER_INCLUSIVE);
            if success == 0 {
                fu_pool_delete(inner);
                return Err(Error::SpawnFailed);
            }

            Ok(Self { inner })
        }
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
    pub fn for_threads<F>(&mut self, function: F) -> ForThreadsOperation<F>
    where
        F: Fn(usize, usize) + Sync,
    {
        ForThreadsOperation {
            pool: self,
            function,
        }
    }

    /// Distributes `n` similar duration calls between threads by individual indices.
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

/// Operation object for parallel thread execution.
pub struct ForThreadsOperation<'a, F> {
    pool: &'a mut ThreadPool,
    function: F,
}

impl<'a, F> Drop for ForThreadsOperation<'a, F>
where
    F: Fn(usize, usize) + Sync,
{
    fn drop(&mut self) {
        extern "C" fn trampoline<F>(
            ctx: *mut core::ffi::c_void,
            thread_index: usize,
            colocation_index: usize,
        ) where
            F: Fn(usize, usize) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(thread_index, colocation_index);
        }

        unsafe {
            let ctx = &self.function as *const F as *mut core::ffi::c_void;
            fu_pool_unsafe_for_threads(self.pool.inner, trampoline::<F>, ctx);
            fu_pool_unsafe_join(self.pool.inner);
        }
    }
}

/// Operation object for parallel task execution with static load balancing.
pub struct ForNOperation<'a, F> {
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
            ctx: *mut core::ffi::c_void,
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
            let ctx = &self.function as *const F as *mut core::ffi::c_void;
            fu_pool_unsafe_for_n(self.pool.inner, self.n, trampoline::<F>, ctx);
            fu_pool_unsafe_join(self.pool.inner);
        }
    }
}

/// Operation object for parallel task execution with dynamic work-stealing.
pub struct ForNDynamicOperation<'a, F> {
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
            ctx: *mut core::ffi::c_void,
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
            let ctx = &self.function as *const F as *mut core::ffi::c_void;
            fu_pool_unsafe_for_n_dynamic(self.pool.inner, self.n, trampoline::<F>, ctx);
            fu_pool_unsafe_join(self.pool.inner);
        }
    }
}

/// Operation object for parallel slice execution.
pub struct ForSlicesOperation<'a, F> {
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
            ctx: *mut core::ffi::c_void,
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
            let ctx = &self.function as *const F as *mut core::ffi::c_void;
            fu_pool_unsafe_for_slices(self.pool.inner, self.n, trampoline::<F>, ctx);
            fu_pool_unsafe_join(self.pool.inner);
        }
    }
}

/// Spawns a pool with the specified number of threads.
pub fn spawn(planned_threads: usize) -> ThreadPool {
    ThreadPool::try_spawn(planned_threads).expect("Failed to spawn ThreadPool")
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
    T: Send,
    F: Fn(&mut T, Prong) + Sync,
{
    let base_ptr = data.as_mut_ptr();
    let n = data.len();

    let _operation = pool.for_n(n, move |prong| unsafe {
        let item = &mut *base_ptr.add(prong.task_index);
        function(item, prong);
    });
}

/// Helper function to visit every element exactly once with dynamic work-stealing.
pub fn for_each_prong_mut_dynamic<T, F>(pool: &mut ThreadPool, data: &mut [T], function: F)
where
    T: Send,
    F: Fn(&mut T, Prong) + Sync,
{
    let base_ptr = data.as_mut_ptr();
    let n = data.len();

    let _operation = pool.for_n_dynamic(n, move |prong| unsafe {
        let item = &mut *base_ptr.add(prong.task_index);
        function(item, prong);
    });
}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    #[inline]
    fn hw_threads() -> usize {
        count_logical_cores().max(1)
    }

    #[test]
    fn test_capabilities() {
        let caps = capabilities_string();
        println!("Capabilities: {:?}", caps);
        assert!(caps.is_some());
    }

    #[test]
    fn test_system_info() {
        let cores = count_logical_cores();
        let numa = count_numa_nodes();
        let colocations = count_colocations();
        let qos = count_quality_levels();

        println!(
            "Cores: {}, NUMA: {}, Colocations: {}, QoS: {}",
            cores, numa, colocations, qos
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

        let visited = Arc::new(
            (0..count_threads)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
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
        const EXPECTED_PARTS: usize = 10_000;
        let mut pool = spawn(hw_threads());

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
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
        const EXPECTED_PARTS: usize = 10_000;
        let mut pool = spawn(hw_threads());

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
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
            // At this point, operation hasn't executed yet
            assert_eq!(counter.load(Ordering::Relaxed), 0);
        } // Operation executes here in the destructor

        // Now the operation should have completed
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }
}
