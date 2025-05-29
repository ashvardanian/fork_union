//! Demo app: N-Body simulation with Fork Union and Rayon.
//!
//! To control the script, several environment variables are used:
//!
//! - `NBODY_COUNT` - number of bodies in the simulation (default: number of threads).
//! - `NBODY_ITERATIONS` - number of iterations to run the simulation (default: 1000).
//! - `NBODY_BACKEND` - backend to use for the simulation (default: `fork_union_static`).
//! - `NBODY_THREADS` - number of threads to use for the simulation (default: number of hardware threads).
//!
//! The backends include: `fork_union_static`, `fork_union_dynamic`, `rayon_static`, and `rayon_dynamic`.
//! To compile and run:
//!
//! ```sh
//! cargo +nightly run --example nbody --release
//! ```
//!
//! The default profiling scheme is to 1M iterations for 128 particles on each backend:
//!
//! ```sh
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=rayon_static cargo +nightly run --example nbody --release
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=rayon_dynamic cargo +nightly run --example nbody --release
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=fork_union_static cargo +nightly run --example nbody --release
//! time NBODY_COUNT=128 NBODY_THREADS=$(nproc) NBODY_ITERATIONS=1000000 \
//!     NBODY_BACKEND=fork_union_dynamic cargo +nightly run --example nbody --release
//! ```
use rand::{rng, Rng};
use std::env;
use std::error::Error;

use fork_union as fu;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};

/// Physical constants.
const G_CONST: f32 = 6.674e-11;
const DT_CONST: f32 = 0.01;
const SOFTEN_CONST: f32 = 1.0e-9;

/// Simple 3-vector used everywhere.
#[derive(Clone, Copy, Default)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

impl std::ops::AddAssign for Vector3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

#[derive(Copy, Clone)]
struct Body {
    position: Vector3,
    velocity: Vector3,
    mass: f32,
}

/// Fast reciprocal square-root (one Newton step of the classic Quake hack).
#[inline]
fn fast_rsqrt(x: f32) -> f32 {
    let i = 0x5f37_59dfu32.wrapping_sub(x.to_bits() >> 1);
    let mut y = f32::from_bits(i);
    let x2 = 0.5 * x;
    y *= 1.5 - x2 * y * y;
    y
}

#[inline]
fn gravitational_force(bi: &Body, bj: &Body) -> Vector3 {
    let dx = bj.position.x - bi.position.x;
    let dy = bj.position.y - bi.position.y;
    let dz = bj.position.z - bi.position.z;
    let l2 = dx * dx + dy * dy + dz * dz + SOFTEN_CONST;
    let inv = fast_rsqrt(l2);
    let inv3 = inv * inv * inv;
    let mag = G_CONST * bi.mass * bj.mass * inv3;
    Vector3 {
        x: mag * dx,
        y: mag * dy,
        z: mag * dz,
    }
}

#[inline]
fn apply_force(b: &mut Body, f: &Vector3) {
    b.velocity.x += f.x / b.mass * DT_CONST;
    b.velocity.y += f.y / b.mass * DT_CONST;
    b.velocity.z += f.z / b.mass * DT_CONST;

    b.position.x += b.velocity.x * DT_CONST;
    b.position.y += b.velocity.y * DT_CONST;
    b.position.z += b.velocity.z * DT_CONST;
}

/// Return the number of logical CPUs visible to this process.
#[inline]
fn hw_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// ────────────────────────────────────────────────────────────────────────────
// Fork-Union kernels
// ────────────────────────────────────────────────────────────────────────────
fn iteration_fu_static(pool: &fu::ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    let bodies_ptr = fu::SyncConstPtr::new(bodies.as_ptr());

    fu::for_each_prong_mut(pool, forces, move |force, prong| unsafe {
        let bi = bodies_ptr.get(prong.task_index);
        let mut acc = Vector3::default();

        for j in 0..n {
            let bj = bodies_ptr.get(j);
            acc += gravitational_force(bi, bj);
        }
        *force = acc;
    });
    fu::for_each_prong_mut(pool, bodies, move |body, prong| {
        apply_force(body, &forces[prong.task_index]);
    });
}

fn iteration_fu_dynamic(pool: &fu::ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    let bodies_ptr = fu::SyncConstPtr::new(bodies.as_ptr());

    fu::for_each_prong_mut_dynamic(pool, forces, move |force, prong| unsafe {
        let bi = bodies_ptr.get(prong.task_index);
        let mut acc = Vector3::default();

        for j in 0..n {
            let bj = bodies_ptr.get(j);
            acc += gravitational_force(bi, bj);
        }
        *force = acc;
    });
    fu::for_each_prong_mut_dynamic(pool, bodies, move |body, prong| {
        apply_force(body, &forces[prong.task_index]);
    });
}

// ────────────────────────────────────────────────────────────────────────────
// Rayon kernels
// ────────────────────────────────────────────────────────────────────────────
fn iteration_rayon_dynamic(pool: &ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();

    pool.install(|| {
        forces.par_iter_mut().enumerate().for_each(|(i, force)| {
            let mut acc = Vector3::default();
            for j in 0..n {
                acc += gravitational_force(&bodies[i], &bodies[j]);
            }
            *force = acc;
        });
    });

    pool.install(|| {
        bodies
            .par_iter_mut()
            .zip(forces.par_iter())
            .for_each(|(b, f)| apply_force(b, f));
    });
}

// "Static" scheduling: one *contiguous* stripe per thread, no stealing.
fn iteration_rayon_static(pool: &ThreadPool, bodies: &mut [Body], forces: &mut [Vector3]) {
    let n = bodies.len();
    let workers = rayon::current_num_threads();
    let stride = (n + workers - 1) / workers;

    pool.install(|| {
        forces
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(chunk_idx, f_chunk)| {
                let start = chunk_idx * stride;

                for (local, force) in f_chunk.iter_mut().enumerate() {
                    let i = start + local;
                    let mut acc = Vector3::default();
                    for j in 0..n {
                        acc += gravitational_force(&bodies[i], &bodies[j]);
                    }
                    *force = acc;
                }
            });
    });

    pool.install(|| {
        bodies
            .par_chunks_mut(stride)
            .zip(forces.par_chunks(stride))
            .for_each(|(b_chunk, f_chunk)| {
                for (b, f) in b_chunk.iter_mut().zip(f_chunk.iter()) {
                    apply_force(b, f);
                }
            });
    });
}

fn main() -> Result<(), Box<dyn Error>> {
    let n = env::var("NBODY_COUNT").ok().and_then(|v| v.parse().ok());
    let iters = env::var("NBODY_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000);
    let backend = env::var("NBODY_BACKEND").unwrap_or_else(|_| "fork_union_static".into());
    let threads = env::var("NBODY_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| hw_threads().into());

    let bodies_n = n.unwrap_or(threads);

    // Allocate & initialize bodies
    let mut bodies = vec![
        Body {
            position: Vector3::default(),
            velocity: Vector3::default(),
            mass: 0.0
        };
        bodies_n
    ];
    let mut forces = vec![Vector3::default(); bodies_n];

    let mut generator = rng();
    bodies.iter_mut().for_each(|b| {
        // positions & velocities in [0, 1)
        b.position = Vector3 {
            x: generator.random(),
            y: generator.random(),
            z: generator.random(),
        };
        b.velocity = Vector3 {
            x: generator.random(),
            y: generator.random(),
            z: generator.random(),
        };

        // mass in [1 e20, 1 e25)
        b.mass = generator.random_range(1.0e20..1.0e25);
    });

    // Run the chosen backend
    match backend.as_str() {
        "fork_union_static" => {
            let pool = fu::ThreadPool::try_spawn(threads)
                .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}"));
            for _ in 0..iters {
                iteration_fu_static(&pool, &mut bodies, &mut forces);
            }
        }
        "fork_union_dynamic" => {
            let pool = fu::ThreadPool::try_spawn(threads)
                .unwrap_or_else(|e| panic!("Failed to start Fork-Union pool: {e}"));
            for _ in 0..iters {
                iteration_fu_dynamic(&pool, &mut bodies, &mut forces);
            }
        }
        "rayon_static" => {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            for _ in 0..iters {
                iteration_rayon_static(&pool, &mut bodies, &mut forces);
            }
        }
        "rayon_dynamic" => {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            for _ in 0..iters {
                iteration_rayon_dynamic(&pool, &mut bodies, &mut forces);
            }
        }
        _ => panic!("Unsupported backend: '{backend}'"),
    }

    Ok(())
}
