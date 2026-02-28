//! High-performance Rust implementations of DDM simulators
//!
//! This module provides parallel DDM simulators using Rayon for work-stealing
//! parallelism and PyO3 for zero-copy Python interoperability.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;  // Same algorithm as NumPy's PCG64, very fast
use rand_distr::{Distribution, StandardNormal, Uniform};

/// Get information about the Rust backend
#[pyfunction]
fn get_rust_info() -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("available", true)?;
        dict.set_item("rayon_threads", rayon::current_num_threads())?;
        dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
        Ok(dict.unbind())
    })
}

/// Thread-safe RNG initialization using PCG64 (same as NumPy)
fn create_rng(seed: u64, thread_id: usize, sample_id: usize, trial_id: usize) -> Pcg64Mcg {
    // Mix seed with indices to create unique per-sample seed
    let combined_seed = seed
        .wrapping_add(thread_id as u64 * 0x9E3779B97F4A7C15)
        .wrapping_add(sample_id as u64 * 0xBF58476D1CE4E5B9)
        .wrapping_add(trial_id as u64 * 0x94D049BB133111EB);

    Pcg64Mcg::seed_from_u64(combined_seed)
}

/// Simulate a single DDM sample
#[inline]
fn ddm_single(
    v: f32,
    a: f32,
    z: f32,
    t_ndt: f32,
    deadline: f32,
    s: f32,
    delta_t: f32,
    max_t: f32,
    rng: &mut Pcg64Mcg,
    smooth_unif: bool,
) -> (f32, i32) {
    let sqrt_dt = delta_t.sqrt();
    let sqrt_st = sqrt_dt * s;
    let deadline_tmp = max_t.min(deadline - t_ndt);

    let uniform = Uniform::new(0.0f32, 1.0f32);

    // Initialize particle
    let mut y = z * a;
    let mut t_particle = 0.0f32;

    // Random walk until boundary or deadline
    // Using StandardNormal (Ziggurat algorithm) for speed
    while y > 0.0 && y < a && t_particle <= deadline_tmp {
        let noise: f32 = { let x: f64 = StandardNormal.sample(rng); x as f32 };
        y += v * delta_t + sqrt_st * noise;
        t_particle += delta_t;
    }

    // Apply smoothing if requested
    let smooth_u = if smooth_unif {
        let u: f32 = uniform.sample(rng);
        if t_particle == 0.0 {
            u * 0.5 * delta_t
        } else if t_particle < deadline_tmp {
            (0.5 - u) * delta_t
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Compute RT
    let mut rt = t_particle + t_ndt + smooth_u;

    // Determine choice
    let choice = if y > 0.0 { 1i32 } else { -1i32 };

    // Enforce deadline
    if rt >= deadline || deadline <= 0.0 {
        rt = -999.0;
    }

    (rt, choice)
}

/// Sequential DDM simulation (single-threaded, no Rayon overhead)
#[pyfunction]
#[pyo3(signature = (v, a, z, t, deadline, s, delta_t, max_t, n_samples, n_trials, seed, smooth_unif))]
fn ddm_rust_sequential<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f32>,
    a: PyReadonlyArray1<'py, f32>,
    z: PyReadonlyArray1<'py, f32>,
    t: PyReadonlyArray1<'py, f32>,
    deadline: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    delta_t: f32,
    max_t: f32,
    n_samples: usize,
    n_trials: usize,
    seed: u64,
    smooth_unif: bool,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let v = v.as_slice()?;
    let a = a.as_slice()?;
    let z = z.as_slice()?;
    let t = t.as_slice()?;
    let deadline = deadline.as_slice()?;
    let s = s.as_slice()?;

    let total_sims = n_samples * n_trials;
    let mut rts = vec![0.0f32; total_sims];
    let mut choices = vec![0i32; total_sims];

    // Sequential simulation (no Rayon)
    for k in 0..n_trials {
        for n in 0..n_samples {
            let idx = n * n_trials + k;
            let mut rng = create_rng(seed, 0, n, k);

            let (rt, choice) = ddm_single(
                v[k], a[k], z[k], t[k], deadline[k], s[k],
                delta_t, max_t, &mut rng, smooth_unif
            );

            rts[idx] = rt;
            choices[idx] = choice;
        }
    }

    let rts_array = rts.to_pyarray_bound(py);
    let choices_array = choices.to_pyarray_bound(py);

    Ok((rts_array, choices_array))
}

/// Parallel DDM simulation
#[pyfunction]
#[pyo3(signature = (v, a, z, t, deadline, s, delta_t, max_t, n_samples, n_trials, n_threads, seed, smooth_unif))]
fn ddm_rust_parallel<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f32>,
    a: PyReadonlyArray1<'py, f32>,
    z: PyReadonlyArray1<'py, f32>,
    t: PyReadonlyArray1<'py, f32>,
    deadline: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    delta_t: f32,
    max_t: f32,
    n_samples: usize,
    n_trials: usize,
    n_threads: usize,
    seed: u64,
    smooth_unif: bool,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    // Convert to Rust slices
    let v = v.as_slice()?;
    let a = a.as_slice()?;
    let z = z.as_slice()?;
    let t = t.as_slice()?;
    let deadline = deadline.as_slice()?;
    let s = s.as_slice()?;

    // Configure thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Total number of simulations
    let total_sims = n_samples * n_trials;

    // Allocate output vectors
    let mut rts = vec![0.0f32; total_sims];
    let mut choices = vec![0i32; total_sims];

    // Run parallel simulation
    pool.install(|| {
        rts.par_iter_mut()
            .zip(choices.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (rt_out, choice_out))| {
                let n = idx / n_trials;  // sample index
                let k = idx % n_trials;  // trial index

                let mut rng = create_rng(seed, rayon::current_thread_index().unwrap_or(0), n, k);

                let (rt, choice) = ddm_single(
                    v[k], a[k], z[k], t[k], deadline[k], s[k],
                    delta_t, max_t, &mut rng, smooth_unif
                );

                *rt_out = rt;
                *choice_out = choice;
            });
    });

    // Convert to numpy arrays
    let rts_array = rts.to_pyarray_bound(py);
    let choices_array = choices.to_pyarray_bound(py);

    Ok((rts_array, choices_array))
}

/// Simulate a single DDM sample with flexible boundary
#[inline]
fn ddm_flexbound_single(
    v: f32,
    z: f32,
    t_ndt: f32,
    deadline: f32,
    s: f32,
    boundary: &[f32],
    delta_t: f32,
    max_t: f32,
    num_steps: usize,
    rng: &mut Pcg64Mcg,
    smooth_unif: bool,
) -> (f32, i32) {
    let sqrt_dt = delta_t.sqrt();
    let sqrt_st = sqrt_dt * s;
    let deadline_tmp = max_t.min(deadline - t_ndt);

    let uniform = Uniform::new(0.0f32, 1.0f32);

    // Initialize particle with flexible boundary
    let mut y = -boundary[0] + (z * 2.0 * boundary[0]);
    let mut t_particle = 0.0f32;
    let mut ix = 0usize;

    // Random walk
    while y >= -boundary[ix] && y <= boundary[ix] && t_particle <= deadline_tmp {
        let noise: f32 = { let x: f64 = StandardNormal.sample(rng); x as f32 };
        y += v * delta_t + sqrt_st * noise;
        t_particle += delta_t;
        ix += 1;
        if ix >= num_steps {
            ix = num_steps - 1;
        }
    }

    // Apply smoothing
    let smooth_u = if smooth_unif {
        let u: f32 = uniform.sample(rng);
        if t_particle == 0.0 {
            u * 0.5 * delta_t
        } else if t_particle < deadline_tmp {
            (0.5 - u) * delta_t
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut rt = t_particle + t_ndt + smooth_u;
    let choice = if y > 0.0 { 1i32 } else { -1i32 };

    if rt >= deadline || deadline <= 0.0 {
        rt = -999.0;
    }

    (rt, choice)
}

/// Parallel DDM simulation with flexible boundary
#[pyfunction]
#[pyo3(signature = (v, z, t, deadline, s, boundary, delta_t, max_t, n_samples, n_trials, num_steps, n_threads, seed, smooth_unif))]
fn ddm_flexbound_rust_parallel<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f32>,
    z: PyReadonlyArray1<'py, f32>,
    t: PyReadonlyArray1<'py, f32>,
    deadline: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    boundary: PyReadonlyArray2<'py, f32>,
    delta_t: f32,
    max_t: f32,
    n_samples: usize,
    n_trials: usize,
    num_steps: usize,
    n_threads: usize,
    seed: u64,
    smooth_unif: bool,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let v = v.as_slice()?;
    let z = z.as_slice()?;
    let t = t.as_slice()?;
    let deadline = deadline.as_slice()?;
    let s = s.as_slice()?;
    let boundary = boundary.as_array();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let total_sims = n_samples * n_trials;
    let mut rts = vec![0.0f32; total_sims];
    let mut choices = vec![0i32; total_sims];

    pool.install(|| {
        rts.par_iter_mut()
            .zip(choices.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (rt_out, choice_out))| {
                let n = idx / n_trials;
                let k = idx % n_trials;

                let mut rng = create_rng(seed, rayon::current_thread_index().unwrap_or(0), n, k);

                // Get boundary for this trial
                let boundary_k: Vec<f32> = (0..num_steps)
                    .map(|i| boundary[[k, i]])
                    .collect();

                let (rt, choice) = ddm_flexbound_single(
                    v[k], z[k], t[k], deadline[k], s[k],
                    &boundary_k, delta_t, max_t, num_steps,
                    &mut rng, smooth_unif
                );

                *rt_out = rt;
                *choice_out = choice;
            });
    });

    let rts_array = rts.to_pyarray_bound(py);
    let choices_array = choices.to_pyarray_bound(py);

    Ok((rts_array, choices_array))
}

/// Simulate a single full DDM sample with inter-trial variability
#[inline]
fn full_ddm_single(
    v: f32,
    z: f32,
    t_ndt: f32,
    sz: f32,
    sv: f32,
    st: f32,
    deadline: f32,
    s: f32,
    boundary: &[f32],
    delta_t: f32,
    max_t: f32,
    num_steps: usize,
    rng: &mut Pcg64Mcg,
    smooth_unif: bool,
) -> (f32, i32) {
    let sqrt_dt = delta_t.sqrt();
    let sqrt_st = sqrt_dt * s;
    let deadline_tmp = max_t.min(deadline - t_ndt);

    let uniform = Uniform::new(0.0f32, 1.0f32);

    // Apply inter-trial variability
    let z_var: f32 = uniform.sample(rng);
    let v_var: f32 = { let x: f64 = StandardNormal.sample(rng); x as f32 };
    let t_var: f32 = uniform.sample(rng);

    let z_effective = z + sz * (z_var - 0.5) * 2.0;
    let drift_increment = (v + sv * v_var) * delta_t;
    let t_effective = t_ndt + st * (t_var - 0.5) * 2.0;

    // Initialize particle
    let mut y = -boundary[0] + (z_effective * 2.0 * boundary[0]);
    let mut t_particle = 0.0f32;
    let mut ix = 0usize;

    // Random walk
    while y >= -boundary[ix] && y <= boundary[ix] && t_particle <= deadline_tmp {
        let noise: f32 = { let x: f64 = StandardNormal.sample(rng); x as f32 };
        y += drift_increment + sqrt_st * noise;
        t_particle += delta_t;
        ix += 1;
        if ix >= num_steps {
            ix = num_steps - 1;
        }
    }

    // Apply smoothing
    let smooth_u = if smooth_unif {
        let u: f32 = uniform.sample(rng);
        if t_particle == 0.0 {
            u * 0.5 * delta_t
        } else if t_particle < deadline_tmp {
            (0.5 - u) * delta_t
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut rt = t_particle + t_effective + smooth_u;
    let choice = if y > 0.0 { 1i32 } else { -1i32 };

    if rt >= deadline || deadline <= 0.0 {
        rt = -999.0;
    }

    (rt, choice)
}

/// Parallel full DDM simulation with inter-trial variability
#[pyfunction]
#[pyo3(signature = (v, z, t, sz, sv, st, deadline, s, boundary, delta_t, max_t, n_samples, n_trials, num_steps, n_threads, seed, smooth_unif))]
fn full_ddm_rust_parallel<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f32>,
    z: PyReadonlyArray1<'py, f32>,
    t: PyReadonlyArray1<'py, f32>,
    sz: PyReadonlyArray1<'py, f32>,
    sv: PyReadonlyArray1<'py, f32>,
    st: PyReadonlyArray1<'py, f32>,
    deadline: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    boundary: PyReadonlyArray2<'py, f32>,
    delta_t: f32,
    max_t: f32,
    n_samples: usize,
    n_trials: usize,
    num_steps: usize,
    n_threads: usize,
    seed: u64,
    smooth_unif: bool,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
    let v = v.as_slice()?;
    let z = z.as_slice()?;
    let t = t.as_slice()?;
    let sz = sz.as_slice()?;
    let sv = sv.as_slice()?;
    let st = st.as_slice()?;
    let deadline = deadline.as_slice()?;
    let s = s.as_slice()?;
    let boundary = boundary.as_array();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let total_sims = n_samples * n_trials;
    let mut rts = vec![0.0f32; total_sims];
    let mut choices = vec![0i32; total_sims];

    pool.install(|| {
        rts.par_iter_mut()
            .zip(choices.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (rt_out, choice_out))| {
                let n = idx / n_trials;
                let k = idx % n_trials;

                let mut rng = create_rng(seed, rayon::current_thread_index().unwrap_or(0), n, k);

                let boundary_k: Vec<f32> = (0..num_steps)
                    .map(|i| boundary[[k, i]])
                    .collect();

                let (rt, choice) = full_ddm_single(
                    v[k], z[k], t[k], sz[k], sv[k], st[k],
                    deadline[k], s[k], &boundary_k,
                    delta_t, max_t, num_steps,
                    &mut rng, smooth_unif
                );

                *rt_out = rt;
                *choice_out = choice;
            });
    });

    let rts_array = rts.to_pyarray_bound(py);
    let choices_array = choices.to_pyarray_bound(py);

    Ok((rts_array, choices_array))
}

/// Python module definition
#[pymodule]
fn ssms_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_rust_info, m)?)?;
    m.add_function(wrap_pyfunction!(ddm_rust_sequential, m)?)?;
    m.add_function(wrap_pyfunction!(ddm_rust_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(ddm_flexbound_rust_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(full_ddm_rust_parallel, m)?)?;
    Ok(())
}
