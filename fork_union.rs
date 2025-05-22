#![feature(allocator_api)]
use core::fmt::Write as _;
use std::alloc::{AllocError, Allocator, Global};
use std::collections::TryReserveError;
use std::fmt;
use std::io::Error as IoError;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

use crossbeam_utils::CachePadded;

#[derive(Debug)]
pub enum ForkUnionError {
    Alloc(AllocError),
    Reserve(TryReserveError),
    Spawn(IoError),
}

impl fmt::Display for ForkUnionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "allocation failure"),
            Self::Reserve(e) => write!(f, "reservation failure: {e}"),
            Self::Spawn(e) => write!(f, "thread-spawn failure: {e}"),
        }
    }
}

impl std::error::Error for ForkUnionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // `AllocError` doesn't implement `Error`, so no source here.
            Self::Alloc(_) => None,
            // These *do* implement `Error`, forward them.
            Self::Reserve(e) => Some(e),
            Self::Spawn(e) => Some(e),
        }
    }
}

/// Describes a portion of work executed on a specific thread.
#[derive(Copy, Clone)]
pub struct Task {
    pub thread_index: usize,
    pub task_index: usize,
}

type Trampoline = unsafe fn(*const (), Task);

/// Dummy trampoline function as opposed to the real `worker_loop`.
unsafe fn dummy_trampoline(_ctx: *const (), _task: Task) {
    unreachable!("dummy_trampoline should not be called")
}

/// The shared state of the thread pool, used by all threads.
/// It intentionally pads all of independently mutable regions to avoid false sharing.
/// The `task_trampoline` function receives the `task_context` state pointers and
/// some ethereal `Task` index similar to C-style thread pools.
#[repr(align(64))]
struct Inner {
    pub total_threads: usize,
    pub task_context: *const (),
    pub task_trampoline: Trampoline,
    pub task_parts_count: usize,

    pub stop: CachePadded<AtomicBool>,
    pub task_parts_remaining: CachePadded<AtomicUsize>,
    pub task_parts_passed: CachePadded<AtomicUsize>,
    pub task_generation: CachePadded<AtomicUsize>,
}

unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

impl Inner {
    pub fn new(threads: usize) -> Self {
        Self {
            stop: CachePadded::new(AtomicBool::new(false)),
            total_threads: threads,
            task_context: ptr::null(),
            task_trampoline: dummy_trampoline,
            task_parts_count: 0,

            task_parts_remaining: CachePadded::new(AtomicUsize::new(0)),
            task_parts_passed: CachePadded::new(AtomicUsize::new(0)),
            task_generation: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    fn reset_task(&self) {
        unsafe {
            let mutable_self = self as *const Self as *mut Self;
            (*mutable_self).task_parts_count = 0;
            (*mutable_self).task_context = ptr::null();
            (*mutable_self).task_trampoline = dummy_trampoline;
        }
    }

    fn trampoline(&self) -> Trampoline {
        self.task_trampoline
    }

    fn context(&self) -> *const () {
        self.task_context
    }
}

/// Minimalistic, fixed‑size thread‑pool for blocking scoped parallelism.
///
/// - You create the pool once with **N** logical threads (`try_spawn[_in]`).
/// - You submit a *single* blocking kernel (`for_each_*`) that might touch millions of tasks.
/// - The pool is torn down (or reused for the next kernel) when the call returns.
///
/// The current thread **participates** in the work, so for `N`‑way parallelism the
/// implementation actually spawns **N − 1** background workers and runs the last
/// slice on the caller thread. Thread count is *runtime constant* – there is no grow/shrink API.
///
/// ### Why another pool?
///
/// - Zero external deps – built only on `std::thread` and `std::sync::atomic`;
///   no channels, no `Mutex`, no `Condvar`, no `async`, no `crossbeam`, no allocation per task.
/// - Weak memory model – atomics use *acquire‑release* or even *relaxed* orderings
///   where safe, better matching Arm / PowerPC memory semantics.
/// - Friendly to custom allocators – the backing storage for the workers
///   vector can have a runtime-defined size and use any memory allocator.
///
/// ### Task‑dispatch flavours
///
/// | Method             | Scheduling          | Suitable for        |
/// |--------------------|---------------------|---------------------|
/// | `for_each_thread`  | one call per worker | thread‑local state  |
/// | `for_each_static`  | static slices       | evenly sized tasks  |
/// | `for_each_dynamic` | work‑stealing       | unpredictable tasks |
///
/// ### Example
///
/// The simplest example of using the pool:
///
/// ```no_run
/// use fork_union::spawn;
///
/// fn heavy_math(_: usize) {}
///
/// let pool = spawn(4); // ! Unsafe shortcut, see below
/// pool.for_each_static(400, |i| {
///     heavy_math(i); // Will be called 400 times, 100 per thread.
/// });
/// ```
///
/// The recommended way, however, is to use the Allocator API and the safer `try_spawn_in` method:
///
/// ```no_run
/// #![feature(allocator_api)]
/// use std::thread;
/// use std::error::Error;
/// use std::alloc::Global;
/// use fork_union::ForkUnion;
///
/// fn heavy_math(_: usize) {}
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let pool = ForkUnion::try_spawn_in(4, Global)?;
///     pool.for_each_dynamic(400, |i| {
///         heavy_math(i); // More expensive to synchronize, but better for uneven workloads.
///     });
///     Ok(())
/// }
/// ```
///
pub struct ForkUnion<A: Allocator + Clone = Global> {
    inner: Box<Inner, A>,
    workers: Vec<JoinHandle<()>, A>,
}

impl<A: Allocator + Clone> ForkUnion<A> {
    /// Creates the pool with the desired number of threads using a custom allocator.
    pub fn try_named_spawn_in(
        name: &str,
        planned_threads: usize,
        alloc: A,
    ) -> Result<Self, ForkUnionError> {
        if planned_threads == 0 {
            return Err(ForkUnionError::Spawn(IoError::new(
                std::io::ErrorKind::InvalidInput,
                "Thread count must be > 0",
            )));
        }
        if planned_threads == 1 {
            return Ok(Self {
                inner: Box::new_in(Inner::new(1), alloc.clone()),
                workers: Vec::new_in(alloc.clone()),
            });
        }

        // With the `inner` object allocated on the heap it we will be able to move the owning object around.
        let inner = Box::try_new_in(Inner::new(planned_threads), alloc.clone())
            .map_err(ForkUnionError::Alloc)?;
        let inner_ptr: &'static Inner = unsafe { &*(inner.as_ref() as *const Inner) };

        let workers_cap = planned_threads.saturating_sub(1);
        let mut workers = Vec::try_with_capacity_in(workers_cap, alloc.clone())
            .map_err(ForkUnionError::Reserve)?;
        for i in 0..workers_cap {
            // We need to carefully fill the workers name
            let mut worker_name = String::new();
            worker_name
                .try_reserve_exact(name.len() + 3)
                .map_err(ForkUnionError::Reserve)?;
            worker_name.push_str(name);
            write!(&mut worker_name, "{:03}", i + 1)
                .expect("writing into a reserved String never fails");

            // We are using the `spawn_unchecked` as the thread may easily outlive the caller.
            unsafe {
                let worker = thread::Builder::new()
                    .name(worker_name)
                    .spawn_unchecked(move || worker_loop(inner_ptr, i + 1))
                    .map_err(ForkUnionError::Spawn)?;
                workers.push(worker);
            }
        }

        Ok(Self { inner, workers })
    }

    /// Creates the pool with the desired number of threads using a custom allocator.
    pub fn try_spawn_in(planned_threads: usize, alloc: A) -> Result<Self, ForkUnionError> {
        Self::try_named_spawn_in("ForkUnion", planned_threads, alloc)
    }

    /// Returns the number of threads in the pool.
    pub fn thread_count(&self) -> usize {
        self.inner.total_threads
    }

    /// Stops all threads and waits for them to finish.
    pub fn stop_and_reset(&mut self) {
        if self.inner.total_threads <= 1 {
            return;
        }
        assert!(self.inner.task_parts_remaining.load(Ordering::SeqCst) == 0);
        self.inner.reset_task();
        self.inner.stop.store(true, Ordering::Release);
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
        self.inner.stop.store(false, Ordering::Relaxed);
    }

    /// Executes a function on each thread of the pool.
    pub fn for_each_thread<F>(&self, function: F)
    where
        F: Fn(usize) + Sync,
    {
        if self.inner.total_threads == 1 {
            function(0);
            return;
        }
        let ctx = &function as *const F as *const ();
        unsafe {
            let inner_ptr = self.inner.as_ref() as *const Inner as *mut Inner;
            (*inner_ptr).task_context = ctx;
            (*inner_ptr).task_trampoline = call_thread::<F>;
            (*inner_ptr).task_parts_count = self.inner.total_threads;
        }
        self.inner
            .task_parts_remaining
            .store(self.inner.total_threads - 1, Ordering::Relaxed);
        self.inner
            .task_generation
            .fetch_add(1, Ordering::Release);
        function(0);
        while self
            .inner
            .task_parts_remaining
            .load(Ordering::Acquire)
            != 0
        {
            thread::yield_now();
        }
        self.inner.reset_task();
    }

    /// Executes evenly sized tasks on each thread.
    pub fn for_each_slice<F>(&self, n: usize, function: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        assert!(self.inner.total_threads != 0, "Thread pool not initialized");
        if self.inner.total_threads == 1 || n == 1 {
            function(0, n);
            return;
        }
        if n == 0 {
            return;
        }
        let per_thread = n.div_ceil(self.inner.total_threads);
        self.for_each_thread(|thread_index| {
            let begin = std::cmp::min(thread_index * per_thread, n);
            let count = std::cmp::min(begin + per_thread, n) - begin;
            function(begin, count);
        });
    }

    /// Executes `n` balanced tasks in parallel.
    pub fn for_each_static<F>(&self, n: usize, function: F)
    where
        F: Fn(usize) + Sync,
    {
        self.for_each_slice(n, |start, count| {
            for i in 0..count {
                function(start + i);
            }
        });
    }

    /// Executes `n` uneven tasks, greedily stealing work.
    pub fn for_each_dynamic<F>(&self, n: usize, function: F)
    where
        F: Fn(usize) + Sync,
    {
        if self.inner.total_threads == 1 {
            for i in 0..n {
                function(i);
            }
            return;
        }
        if n == 0 {
            return;
        }
        if n == 1 {
            function(0);
            return;
        }
        let ctx = &function as *const F as *const ();
        unsafe {
            let inner_ptr = self.inner.as_ref() as *const Inner as *mut Inner;
            (*inner_ptr).task_context = ctx;
            (*inner_ptr).task_trampoline = call_index::<F>;
            (*inner_ptr).task_parts_count = n;
        }
        self.inner
            .task_parts_passed
            .store(0, Ordering::Relaxed);
        self.inner
            .task_parts_remaining
            .store(n, Ordering::Relaxed);
        self.inner
            .task_generation
            .fetch_add(1, Ordering::Release);
        // SAFETY: `self.inner` lives as long as the pool
        let inner_ref: &'static Inner = unsafe { &*(self.inner.as_ref() as *const Inner) };
        dynamic_loop(inner_ref, 0);
        while self
            .inner
            .task_parts_remaining
            .load(Ordering::Acquire)
            != 0
        {
            thread::yield_now();
        }
        self.inner.reset_task();
    }
}

impl<A: Allocator + Clone> Drop for ForkUnion<A> {
    fn drop(&mut self) {
        self.stop_and_reset();
    }
}

impl<A> ForkUnion<A>
where
    A: Allocator + Clone + Default,
{
    pub fn try_spawn(planned_threads: usize) -> Result<Self, ForkUnionError> {
        Self::try_named_spawn_in("ForkUnion", planned_threads, A::default())
    }

    pub fn try_named_spawn(name: &str, planned_threads: usize) -> Result<Self, ForkUnionError> {
        Self::try_named_spawn_in(name, planned_threads, A::default())
    }
}

unsafe fn call_thread<F: Fn(usize)>(ctx: *const (), task: Task) {
    let f = &*(ctx as *const F);
    f(task.thread_index);
}

unsafe fn call_index<F: Fn(usize)>(ctx: *const (), task: Task) {
    let f = &*(ctx as *const F);
    f(task.task_index);
}

fn worker_loop(inner: &'static Inner, thread_index: usize) {
    let mut last_generation = 0usize;
    loop {
        let mut new_generation;
        while {
            new_generation = inner.task_generation.load(Ordering::Acquire);
            new_generation == last_generation && !inner.stop.load(Ordering::Acquire)
        } {
            thread::yield_now();
        }
        if inner.stop.load(Ordering::Acquire) {
            return;
        }
        let is_static = inner.task_parts_count == inner.total_threads;
        if is_static && inner.task_parts_count > 0 {
            let trampoline = inner.trampoline();
            let context = inner.context();
            unsafe {
                trampoline(
                    context,
                    Task {
                        thread_index,
                        task_index: thread_index,
                    },
                );
            }
            inner
                .task_parts_remaining
                .fetch_sub(1, Ordering::AcqRel);
        } else {
            dynamic_loop(inner, thread_index);
        }
        last_generation = new_generation;
    }
}

fn dynamic_loop(inner: &'static Inner, thread_index: usize) {
    let trampoline = inner.trampoline();
    let context = inner.context();
    loop {
        let idx = inner
            .task_parts_passed
            .fetch_add(1, Ordering::Relaxed);
        if idx >= inner.task_parts_count {
            break;
        }
        unsafe {
            trampoline(
                context,
                Task {
                    thread_index,
                    task_index: idx,
                },
            );
        }
        inner
            .task_parts_remaining
            .fetch_sub(1, Ordering::AcqRel);
    }
}

/// Spawns a pool with the default allocator.
pub fn spawn(planned_threads: usize) -> ForkUnion<Global> {
    ForkUnion::<Global>::try_spawn_in(planned_threads, Global)
        .expect("Failed to spawn ForkUnion thread pool")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{Global, Layout};
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };

    #[inline]
    fn hw_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    #[test]
    fn spawn_with_global() {
        let pool = spawn(2);
        assert_eq!(pool.thread_count(), 2);
    }

    #[test]
    fn spawn_with_allocator() {
        let pool = ForkUnion::try_spawn_in(2, Global).expect("spawn");
        assert_eq!(pool.thread_count(), 2);
    }

    #[test]
    fn for_each_thread_dispatch() {
        let count_threads = hw_threads();
        let pool = spawn(count_threads);

        let visited = Arc::new(
            (0..count_threads)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let visited_ref = Arc::clone(&visited);

        pool.for_each_thread(move |thread_index| {
            visited_ref[thread_index].store(true, Ordering::Relaxed);
        });

        for flag in visited.iter() {
            assert!(
                flag.load(Ordering::Relaxed),
                "thread never reached the callback"
            );
        }
    }

    #[test]
    fn for_each_static_uncomfortable_input_size() {
        let count_threads = hw_threads();
        let pool = spawn(count_threads);

        for input_size in 0..count_threads {
            let out_of_bounds = AtomicBool::new(false);
            pool.for_each_static(input_size, |task_index| {
                if task_index >= count_threads {
                    out_of_bounds.store(true, Ordering::Relaxed);
                }
            });
            assert!(
                !out_of_bounds.load(Ordering::Relaxed),
                "task_index exceeded thread_count at n = {input_size}"
            );
        }
    }

    #[test]
    fn for_each_static_static_scheduling() {
        const EXPECTED_PARTS: usize = 10_000_000;
        let pool = spawn(hw_threads());

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        pool.for_each_static(EXPECTED_PARTS, move |task_index| {
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
    fn for_each_dynamic_dynamic_scheduling() {
        const EXPECTED_PARTS: usize = 10_000_000;
        let pool = spawn(hw_threads());

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        pool.for_each_dynamic(EXPECTED_PARTS, move |task_index| {
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
    fn oversubscribed_unbalanced_threads() {
        const EXPECTED_PARTS: usize = 10_000_000;
        const OVERSUBSCRIPTION: usize = 7;
        let threads = hw_threads() * OVERSUBSCRIPTION;
        let pool = spawn(threads);

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        thread_local! { static LOCAL_WORK: std::cell::Cell<usize> = std::cell::Cell::new(0); }

        pool.for_each_dynamic(EXPECTED_PARTS, move |task_index| {
            // Mildly unbalanced CPU burn
            LOCAL_WORK.with(|cell| {
                let mut acc = cell.get();
                for i in 0..task_index % OVERSUBSCRIPTION {
                    acc = acc.wrapping_add(i * i);
                }
                cell.set(acc);
            });

            if visited_ref[task_index].swap(true, Ordering::Relaxed) {
                duplicate_ref.store(true, Ordering::Relaxed);
            }
        });

        assert!(
            !duplicate.load(Ordering::Relaxed),
            "oversubscribed run produced duplicate task IDs"
        );
        for flag in visited.iter() {
            assert!(flag.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn c_function_pointer_compatibility() {
        static TASK_COUNTER: AtomicUsize = AtomicUsize::new(0);
        const EXPECTED_PARTS: usize = 10_000_000;

        fn tally(_: usize) {
            TASK_COUNTER.fetch_add(1, Ordering::Relaxed);
        }

        TASK_COUNTER.store(0, Ordering::Relaxed);
        let pool = spawn(hw_threads());
        pool.for_each_dynamic(EXPECTED_PARTS, tally as fn(usize));

        assert_eq!(
            TASK_COUNTER.load(Ordering::Relaxed),
            EXPECTED_PARTS,
            "function-pointer callback executed the wrong number of times"
        );
    }

    #[test]
    fn concurrent_histogram_array() {
        const HIST_SIZE: usize = 16;
        const ELEMENTS: usize = 1_000_000;
        let pool = spawn(hw_threads());

        let values: Vec<usize> = (0..ELEMENTS).map(|i| i % HIST_SIZE).collect();
        let histogram = Arc::new(
            (0..HIST_SIZE)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>(),
        );
        let hist_ref = Arc::clone(&histogram);

        pool.for_each_dynamic(ELEMENTS, |task_index| {
            let value = values[task_index];
            hist_ref[value].fetch_add(1, Ordering::Relaxed);
        });

        for (i, counter) in histogram.iter().enumerate() {
            assert_eq!(
                counter.load(Ordering::Relaxed),
                ELEMENTS / HIST_SIZE,
                "histogram bin {i} has incorrect count",
            );
        }
    }

    fn increment_all(pool: &ForkUnion, data: &[AtomicUsize]) {
        pool.for_each_static(data.len(), |i| {
            data[i].fetch_add(1, Ordering::Relaxed);
        });
    }

    #[test]
    fn pass_pool_and_reuse() {
        const ELEMENTS: usize = 128;
        let pool = spawn(hw_threads());

        let data = (0..ELEMENTS)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>();

        increment_all(&pool, &data);
        increment_all(&pool, &data);

        for counter in data.iter() {
            assert_eq!(counter.load(Ordering::Relaxed), 2);
        }
    }

    #[test]
    fn manual_stop_and_reset() {
        let mut pool = spawn(hw_threads());
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        pool.for_each_static(1000, |_| {
            COUNTER.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(COUNTER.load(Ordering::Relaxed), 1000);
        pool.stop_and_reset();
        pool.stop_and_reset();
    }

    #[derive(Clone)]
    struct CountingAllocator {
        used: Arc<AtomicUsize>,
        limit: Option<usize>,
    }

    impl CountingAllocator {
        fn new(limit: Option<usize>) -> Self {
            Self {
                used: Arc::new(AtomicUsize::new(0)),
                limit,
            }
        }
    }

    unsafe impl Allocator for CountingAllocator {
        fn allocate(&self, layout: Layout) -> Result<std::ptr::NonNull<[u8]>, AllocError> {
            if let Some(limit) = self.limit {
                let new = self.used.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
                if new > limit {
                    self.used.fetch_sub(layout.size(), Ordering::SeqCst);
                    return Err(AllocError);
                }
            }
            Global.allocate(layout)
        }

        fn allocate_zeroed(&self, layout: Layout) -> Result<std::ptr::NonNull<[u8]>, AllocError> {
            if let Some(limit) = self.limit {
                let new = self.used.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
                if new > limit {
                    self.used.fetch_sub(layout.size(), Ordering::SeqCst);
                    return Err(AllocError);
                }
            }
            Global.allocate_zeroed(layout)
        }

        unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: Layout) {
            if self.limit.is_some() {
                self.used.fetch_sub(layout.size(), Ordering::SeqCst);
            }
            Global.deallocate(ptr, layout)
        }

        unsafe fn grow(
            &self,
            ptr: std::ptr::NonNull<u8>,
            old: Layout,
            new: Layout,
        ) -> Result<std::ptr::NonNull<[u8]>, AllocError> {
            if let Some(limit) = self.limit {
                let extra = new.size().saturating_sub(old.size());
                let new_used = self.used.fetch_add(extra, Ordering::SeqCst) + extra;
                if new_used > limit {
                    self.used.fetch_sub(extra, Ordering::SeqCst);
                    return Err(AllocError);
                }
            }
            Global.grow(ptr, old, new)
        }

        unsafe fn grow_zeroed(
            &self,
            ptr: std::ptr::NonNull<u8>,
            old: Layout,
            new: Layout,
        ) -> Result<std::ptr::NonNull<[u8]>, AllocError> {
            if let Some(limit) = self.limit {
                let extra = new.size().saturating_sub(old.size());
                let new_used = self.used.fetch_add(extra, Ordering::SeqCst) + extra;
                if new_used > limit {
                    self.used.fetch_sub(extra, Ordering::SeqCst);
                    return Err(AllocError);
                }
            }
            Global.grow_zeroed(ptr, old, new)
        }

        unsafe fn shrink(
            &self,
            ptr: std::ptr::NonNull<u8>,
            old: Layout,
            new: Layout,
        ) -> Result<std::ptr::NonNull<[u8]>, AllocError> {
            if self.limit.is_some() {
                let delta = old.size().saturating_sub(new.size());
                self.used.fetch_sub(delta, Ordering::SeqCst);
            }
            Global.shrink(ptr, old, new)
        }
    }

    #[test]
    fn for_each_dynamic_dynamic_scheduling_with_counting_alloc() {
        const EXPECTED_PARTS: usize = 10_000_000;
        const MINIMAL_NEEDED_SIZE: usize = size_of::<Inner>();

        let small_allocator = CountingAllocator::new(Some(MINIMAL_NEEDED_SIZE - 1));
        assert!(
            ForkUnion::try_spawn_in(hw_threads(), small_allocator.clone()).is_err(),
            "We should not be able to spawn a pool with a small allocator"
        );

        let large_allocator = CountingAllocator::new(Some(1024 * 1024));
        let pool = ForkUnion::try_spawn_in(hw_threads(), large_allocator.clone())
            .expect("We should have enough memory for this!");

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        pool.for_each_dynamic(EXPECTED_PARTS, move |task_index| {
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
}
