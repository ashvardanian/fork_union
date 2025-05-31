//! OpenMP-style cross-platform fine-grained parallelism library.
//!
//! Fork Union provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
//! avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
//! The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
//! execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
//! to use even with the maximal `usize` values. It's compatible with the Nightly toolchain and requires
//! the `allocator_api` feature to be enabled.
#![feature(allocator_api)]
use core::fmt::Write as _;
use core::marker::PhantomData;
use std::alloc::{AllocError, Allocator, Global};
use std::collections::TryReserveError;
use std::fmt;
use std::io::Error as IoError;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

/// Pads the wrapped value to 128 bytes to avoid false sharing.
#[repr(align(128))]
struct Padded<T>(T);

impl<T> Padded<T> {
    #[inline]
    fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T> core::ops::Deref for Padded<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for Padded<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug)]
pub enum Error {
    Alloc(AllocError),
    Reserve(TryReserveError),
    Spawn(IoError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "allocation failure"),
            Self::Reserve(e) => write!(f, "reservation failure: {e}"),
            Self::Spawn(e) => write!(f, "thread-spawn failure: {e}"),
        }
    }
}

impl std::error::Error for Error {
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

type Trampoline = unsafe fn(*const (), usize, &mut bool);

/// Dummy trampoline function as opposed to the real `worker_loop`.
unsafe fn dummy_trampoline(_ctx: *const (), _index: usize, _stop: &mut bool) {
    unreachable!("dummy_trampoline should not be called")
}

unsafe fn stop_trampoline(_ctx: *const (), _index: usize, stop: &mut bool) {
    *stop = true;
}

/// The shared state of the thread pool, used by all threads.
/// It intentionally pads all of independently mutable regions to avoid false sharing.
/// The `task_trampoline` function receives the `task_context` state pointers and
/// some ethereal thread index similar to C-style thread pools.
#[repr(align(128))]
struct Inner {
    pub total_threads: usize,

    pub fork_context: *const (),
    pub fork_trampoline: Trampoline,
    pub threads_to_sync: Padded<AtomicUsize>,
    pub fork_generation: Padded<AtomicUsize>,
}

unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

impl Inner {
    pub fn new(threads: usize) -> Self {
        Self {
            total_threads: threads,
            fork_context: ptr::null(),
            fork_trampoline: dummy_trampoline,
            threads_to_sync: Padded::new(AtomicUsize::new(0)),
            fork_generation: Padded::new(AtomicUsize::new(0)),
        }
    }

    fn reset_fork(&self) {
        unsafe {
            let this = self as *const Self as *mut Self;
            (*this).fork_context = ptr::null();
            (*this).fork_trampoline = dummy_trampoline;
        }
    }

    fn trampoline(&self) -> Trampoline {
        self.fork_trampoline
    }

    fn context(&self) -> *const () {
        self.fork_context
    }
}

/// Minimalistic, fixed‑size thread‑pool for blocking scoped parallelism.
///
/// - You create the pool once with **N** logical threads (`try_spawn[_in]`).
/// - You submit a *single* blocking kernel (`broadcast`) on all running threads.
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
/// | `broadcast`        | one call per worker | thread‑local state  |
/// | `for_n`            | static slices       | evenly sized tasks  |
/// | `for_n_dynamic`    | work‑stealing       | unpredictable tasks |
///
/// ### Example
///
/// The simplest example of using the pool to greet from all threads:
///
/// ```no_run
/// use fork_union as fu;
/// let pool = fu::spawn(4); // ! Unsafe shortcut, see below
/// pool.broadcast(|thread_index| {
///     println!("Hello from thread {thread_index}!");
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
/// use fork_union as fu;
///
/// fn heavy_math(_: usize) {}
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let pool = fu::ThreadPool::try_spawn_in(4, Global)?;
///     fu::for_n_dynamic(&pool, 400, |prong| {
///         heavy_math(prong.task_index);
///     });
///     Ok(())
/// }
/// ```
///
pub struct ThreadPool<A: Allocator + Clone = Global> {
    inner: Box<Inner, A>,
    workers: Vec<JoinHandle<()>, A>,
}

impl<A: Allocator + Clone> ThreadPool<A> {
    /// Creates the pool with the desired number of threads using a custom allocator.
    pub fn try_named_spawn_in(name: &str, planned_threads: usize, alloc: A) -> Result<Self, Error> {
        if planned_threads == 0 {
            return Err(Error::Spawn(IoError::new(
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
        let inner =
            Box::try_new_in(Inner::new(planned_threads), alloc.clone()).map_err(Error::Alloc)?;
        let inner_ptr: &'static Inner = unsafe { &*(inner.as_ref() as *const Inner) };

        let workers_cap = planned_threads.saturating_sub(1);
        let mut workers =
            Vec::try_with_capacity_in(workers_cap, alloc.clone()).map_err(Error::Reserve)?;
        for i in 0..workers_cap {
            // We need to carefully fill the workers name
            let mut worker_name = String::new();
            worker_name
                .try_reserve_exact(name.len() + 3)
                .map_err(Error::Reserve)?;
            worker_name.push_str(name);
            write!(&mut worker_name, "{:03}", i + 1)
                .expect("writing into a reserved String never fails");

            // We are using the `spawn_unchecked` as the thread may easily outlive the caller.
            unsafe {
                let worker = thread::Builder::new()
                    .name(worker_name)
                    .spawn_unchecked(move || worker_loop(inner_ptr, i + 1))
                    .map_err(Error::Spawn)?;
                workers.push(worker);
            }
        }

        Ok(Self { inner, workers })
    }

    /// Creates the pool with the desired number of threads using a custom allocator.
    pub fn try_spawn_in(planned_threads: usize, alloc: A) -> Result<Self, Error> {
        Self::try_named_spawn_in("ThreadPool", planned_threads, alloc)
    }

    /// Returns the number of threads in the pool.
    pub fn threads(&self) -> usize {
        self.inner.total_threads
    }

    /// Stops all threads and waits for them to finish.
    pub fn stop_and_reset(&mut self) {
        if self.inner.total_threads <= 1 {
            return;
        }
        assert!(self.inner.threads_to_sync.load(Ordering::SeqCst) == 0);
        self.inner.fork_context = ptr::null();
        self.inner.fork_trampoline = stop_trampoline;
        self.inner.fork_generation.fetch_add(1, Ordering::Release);
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }

    /// Executes a function on each thread of the pool.
    pub fn broadcast<F>(&self, function: F)
    where
        F: Fn(usize) + Sync,
    {
        let threads = self.threads();
        assert!(threads != 0, "Thread pool not initialized");
        if threads == 1 {
            function(0);
            return;
        }

        let ctx = &function as *const F as *const ();
        unsafe {
            let inner_ptr = self.inner.as_ref() as *const Inner as *mut Inner;
            (*inner_ptr).fork_context = ctx;
            (*inner_ptr).fork_trampoline = call_lambda::<F>;
        }
        self.inner
            .threads_to_sync
            .store(threads - 1, Ordering::Relaxed);
        self.inner.fork_generation.fetch_add(1, Ordering::Release);

        function(0);

        while self.inner.threads_to_sync.load(Ordering::Acquire) != 0 {
            thread::yield_now();
        }
        self.inner.reset_fork();
    }
}

impl<A: Allocator + Clone> Drop for ThreadPool<A> {
    fn drop(&mut self) {
        self.stop_and_reset();
    }
}

impl<A> ThreadPool<A>
where
    A: Allocator + Clone + Default,
{
    pub fn try_spawn(planned_threads: usize) -> Result<Self, Error> {
        Self::try_named_spawn_in("ThreadPool", planned_threads, A::default())
    }

    pub fn try_named_spawn(name: &str, planned_threads: usize) -> Result<Self, Error> {
        Self::try_named_spawn_in(name, planned_threads, A::default())
    }
}

unsafe fn call_lambda<F: Fn(usize)>(ctx: *const (), index: usize, _stop: &mut bool) {
    let f = &*(ctx as *const F);
    f(index);
}

fn worker_loop(inner: &'static Inner, thread_index: usize) {
    let mut last_generation = 0usize;
    let mut stop = false;
    assert!(thread_index != 0);
    loop {
        let mut new_generation;
        while {
            new_generation = inner.fork_generation.load(Ordering::Acquire);
            new_generation == last_generation
        } {
            thread::yield_now();
        }

        let trampoline = inner.trampoline();
        let context = inner.context();
        unsafe {
            trampoline(context, thread_index, &mut stop);
        }

        if stop {
            return;
        }

        last_generation = new_generation;

        let before = inner.threads_to_sync.fetch_sub(1, Ordering::Release);
        debug_assert!(before > 0);
    }
}

/// Spawns a pool with the default allocator.
pub fn spawn(planned_threads: usize) -> ThreadPool<Global> {
    ThreadPool::<Global>::try_spawn_in(planned_threads, Global)
        .expect("Failed to spawn ThreadPool thread pool")
}

/// Describes a portion of work executed on a specific thread.
#[derive(Copy, Clone)]
pub struct Prong {
    pub thread_index: usize,
    pub task_index: usize,
}

/// Distributes `n` similar duration calls between threads in slices.
pub fn for_slices<A, F>(pool: &ThreadPool<A>, n: usize, function: F)
where
    A: Allocator + Clone,
    F: Fn(Prong, usize) + Sync,
{
    assert!(pool.threads() != 0, "Thread pool not initialized");
    if n == 0 {
        return;
    }
    let threads = pool.threads();
    if threads == 1 || n == 1 {
        function(
            Prong {
                thread_index: 0,
                task_index: 0,
            },
            n,
        );
        return;
    }

    let tasks_per_thread_lower_bound = n / threads;
    let tasks_per_thread =
        tasks_per_thread_lower_bound + ((tasks_per_thread_lower_bound * threads) < n) as usize;

    pool.broadcast(|thread_index| {
        let begin = thread_index * tasks_per_thread;
        let begin_lower_bound = tasks_per_thread_lower_bound * thread_index;
        let begin_overflows = begin_lower_bound > begin;
        let begin_exceeds = begin >= n;
        if begin_overflows || begin_exceeds {
            return;
        }
        let count = std::cmp::min(begin.saturating_add(tasks_per_thread), n) - begin;
        function(
            Prong {
                thread_index,
                task_index: begin,
            },
            count,
        );
    });
}

/// Distributes `n` similar duration calls between threads by individual indices.
pub fn for_n<A, F>(pool: &ThreadPool<A>, n: usize, function: F)
where
    A: Allocator + Clone,
    F: Fn(Prong) + Sync,
{
    for_slices(pool, n, |start_prong, count| {
        for offset in 0..count {
            function(Prong {
                thread_index: start_prong.thread_index,
                task_index: start_prong.task_index + offset,
            });
        }
    });
}

/// Executes `n` uneven tasks on all threads, greedily stealing work.
pub fn for_n_dynamic<A, F>(pool: &ThreadPool<A>, n: usize, function: F)
where
    A: Allocator + Clone,
    F: Fn(Prong) + Sync,
{
    let prongs_count = n;
    if prongs_count == 0 {
        return;
    }
    if prongs_count == 1 {
        function(Prong {
            thread_index: 0,
            task_index: 0,
        });
        return;
    }
    let threads = pool.threads();
    if threads == 1 {
        for i in 0..prongs_count {
            function(Prong {
                thread_index: 0,
                task_index: i,
            });
        }
        return;
    }

    let prongs_static = threads;
    if prongs_count <= prongs_static {
        return for_n(pool, prongs_count, function);
    }

    let prongs_dynamic = prongs_count - prongs_static;
    let progress = Padded::new(AtomicUsize::new(0));
    pool.broadcast(|thread_index| {
        let static_index = prongs_dynamic + thread_index;
        let mut prong = Prong {
            thread_index,
            task_index: static_index,
        };
        function(prong);
        loop {
            let idx = progress.fetch_add(1, Ordering::Relaxed);
            if idx >= prongs_dynamic {
                break;
            }
            prong.task_index = idx;
            function(prong);
        }
    });
}

/// Raw mutable pointer that can cross threads.
///
/// # Safety
/// You must uphold the usual aliasing rules manually.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SyncMutPtr<T>(*mut T, PhantomData<*mut T>);

/// Raw const pointer that can cross threads.
///
/// # Safety
/// You must ensure the closure environment is thread-safe.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SyncConstPtr<F>(*const F, PhantomData<*const F>);

unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}
unsafe impl<F> Send for SyncConstPtr<F> {}
unsafe impl<F> Sync for SyncConstPtr<F> {}

impl<T> SyncMutPtr<T> {
    #[inline]
    pub fn new(ptr: *mut T) -> Self {
        Self(ptr, PhantomData)
    }

    /// # Safety
    /// Caller must ensure `index` is in-bounds **and** that aliasing rules are respected.
    #[inline(always)]
    pub unsafe fn get_mut(&self, index: usize) -> &mut T {
        &mut *self.0.add(index)
    }
}

impl<F> SyncConstPtr<F> {
    #[inline]
    pub fn new(ptr: *const F) -> Self {
        Self(ptr, PhantomData)
    }

    /// # Safety
    /// Same safety requirements as the original closure plus `item`’s aliasing.
    #[inline]
    pub unsafe fn call<T>(&self, item: &mut T, prong: Prong)
    where
        F: Fn(&mut T, Prong),
    {
        (&*self.0)(item, prong)
    }

    /// # Safety
    /// Caller must ensure `index` is in-bounds **and** that aliasing rules are respected.
    #[inline(always)]
    pub unsafe fn get(&self, index: usize) -> &F {
        &*self.0.add(index)
    }
}

/// Visit every element **exactly once**, passing both the `Prong`
/// (so you know `thread_index` and `task_index`) **and** a mutable
/// reference to that element.
///
/// The work distribution is the same _static_ slicing that `for_n`
/// already uses, so the overhead is one extra pointer and a bounds
/// calculation — nothing more.
///
/// ```no_run
/// use fork_union as fu;
/// let pool = fu::spawn(1);
/// let mut data = vec![0u64; 1_000_000];
/// fu::for_each_prong_mut(&pool, &mut data, |x, prong| {
///     *x = prong.task_index as u64 * 2;
/// });
/// ```
///
/// There is a lot of ugly `unsafe` boilerplate needed to mutate a
/// set of elements in parallel, so this API serves as a shortcut.
///
/// Similar to Rayon's `par_chunks_mut`.
pub fn for_each_prong_mut<A, T, F>(pool: &ThreadPool<A>, data: &mut [T], function: F)
where
    A: Allocator + Clone,
    T: Send,
    F: Fn(&mut T, Prong) + Sync,
{
    let base_ptr = SyncMutPtr::new(data.as_mut_ptr());
    let fun_ptr = SyncConstPtr::new(&function as *const F);
    let n = data.len();
    for_n(pool, n, move |prong| unsafe {
        let item = base_ptr.get_mut(prong.task_index);
        fun_ptr.call(item, prong);
    });
}

/// Visit every element **exactly once**, passing both the `Prong`
/// (so you know `thread_index` and `task_index`) **and** a mutable
/// reference to that element.
///
/// The work distribution is _dynamic_ stealing, so the threads
/// will compete for the elements, and the order of execution is
/// not guaranteed to be sequential.
///
/// ```no_run
/// use fork_union as fu;
/// let pool = fu::spawn(1);
/// let mut strings = vec![String::new(); 1_000];
/// fu::for_each_prong_mut_dynamic(&pool, &mut strings, |s, prong| {
///     s.push_str(&format!("hello from thread {}", prong.thread_index));
/// });
/// ```
///
/// There is a lot of ugly `unsafe` boilerplate needed to mutate a
/// set of elements in parallel, so this API serves as a shortcut.
///
/// Similar to Rayon's `par_iter_mut`.
pub fn for_each_prong_mut_dynamic<A, T, F>(pool: &ThreadPool<A>, data: &mut [T], function: F)
where
    A: Allocator + Clone,
    T: Send,
    F: Fn(&mut T, Prong) + Sync,
{
    let base_ptr = SyncMutPtr::new(data.as_mut_ptr());
    let fun_ptr = SyncConstPtr::new(&function as *const F);
    let n = data.len();
    for_n_dynamic(pool, n, move |prong| unsafe {
        let item = base_ptr.get_mut(prong.task_index);
        fun_ptr.call(item, prong);
    });
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
        assert_eq!(pool.threads(), 2);
    }

    #[test]
    fn spawn_with_allocator() {
        let pool = ThreadPool::try_spawn_in(2, Global).expect("spawn");
        assert_eq!(pool.threads(), 2);
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

        pool.broadcast(move |thread_index| {
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
            for_n(&pool, input_size, |prong| {
                let task_index = prong.task_index;
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

        for_n(&pool, EXPECTED_PARTS, move |prong| {
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

        for_n_dynamic(&pool, EXPECTED_PARTS, move |prong| {
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

        for_n_dynamic(&pool, EXPECTED_PARTS, move |prong| {
            let task_index = prong.task_index;
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
        for_n_dynamic(&pool, EXPECTED_PARTS, |prong| tally(prong.task_index));

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

        for_n_dynamic(&pool, ELEMENTS, |prong| {
            let task_index = prong.task_index;
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

    fn increment_all(pool: &ThreadPool, data: &[AtomicUsize]) {
        for_n(pool, data.len(), |prong| {
            data[prong.task_index].fetch_add(1, Ordering::Relaxed);
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

        for_n(&pool, 1000, |_| {
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
            ThreadPool::try_spawn_in(hw_threads(), small_allocator.clone()).is_err(),
            "We should not be able to spawn a pool with a small allocator"
        );

        let large_allocator = CountingAllocator::new(Some(1024 * 1024));
        let pool = ThreadPool::try_spawn_in(hw_threads(), large_allocator.clone())
            .expect("We should have enough memory for this!");

        let visited = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        for_n_dynamic(&pool, EXPECTED_PARTS, move |prong| {
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
}
