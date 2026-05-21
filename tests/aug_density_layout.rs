//! Layout self-test for the production gemm path used by `rayleigh_ritz` to
//! produce `bp_dev = β_g^* · ψ` and `compute_aug_density_gpu`'s reinterpret
//! `Array2::from_shape_vec((n_e, n_bands).f(), ...)`.
//!
//! Two assertions:
//!
//! 1. **Production path is internally consistent**: gemm output (col-major
//!    ne × n_bands) reinterpreted with `.f()` recovers the logical bp[n, b]
//!    that brute-force CPU computation produces.
//!
//! 2. **Test data path is inconsistent**: H2D of `arr.iter()` on a row-major
//!    `Array2` (as `tests/ca_scf_convergence.rs::aug_density_gpu_matches_cpu_cu111_co`
//!    does) reinterpreted with `.f()` does NOT recover the logical values.
//!    This pinpoints the existing test as not exercising production semantics.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use cudarc::driver::CudaContext;
use ndarray::{Array2, ShapeBuilder};
use num_complex::Complex64;

use chemrust_scf::device::blas::{op, BlasHandle, ZgemmConfig};
use chemrust_scf::device::{complex_to_cuda, cuda_to_complex, CudaComplex};

fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

/// Build a deterministic row-major (m × n) complex matrix.
fn deterministic_matrix(m: usize, n: usize, salt: f64) -> Array2<Complex64> {
    let mut a = Array2::<Complex64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let re = (i as f64 + 0.5) + salt;
            let im = (j as f64 + 0.25) - salt;
            a[[i, j]] = Complex64::new(re, im);
        }
    }
    a
}

/// CPU brute-force: bp_truth[n, b] = Σ_g conj(beta[n, g]) · psi[g, b].
fn bp_truth(beta: &Array2<Complex64>, psi: &Array2<Complex64>) -> Array2<Complex64> {
    let ne = beta.shape()[0];
    let n_pw = beta.shape()[1];
    let n_bands = psi.shape()[1];
    assert_eq!(n_pw, psi.shape()[0]);
    let mut bp = Array2::<Complex64>::zeros((ne, n_bands));
    for n in 0..ne {
        for b in 0..n_bands {
            let mut acc = Complex64::ZERO;
            for g in 0..n_pw {
                acc += beta[[n, g]].conj() * psi[[g, b]];
            }
            bp[[n, b]] = acc;
        }
    }
    bp
}

#[test]
#[ignore = "requires GPU"]
fn production_gemm_path_recovers_bp_truth() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    // Tiny dimensions
    let ne: usize = 3;
    let n_pw: usize = 8;
    let n_bands: usize = 4;

    // β_g: row-major Array2 (n_e, n_pw) — matches `compute_beta_g` output.
    let beta = deterministic_matrix(ne, n_pw, 0.7);
    // ψ: row-major Array2 (n_pw, n_bands) for CPU compute, then we flatten
    //    col-major to match production memory layout.
    let psi = deterministic_matrix(n_pw, n_bands, -0.3);

    // CPU brute-force truth.
    let bp_truth_arr = bp_truth(&beta, &psi);

    // Build the GPU inputs the same way production does.
    let ctx = Arc::new(CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(Arc::clone(&stream)).expect("BlasHandle");

    // β_g upload: `beta.iter()` on a row-major Array2 produces
    // flat[i*n_pw + j] = beta[i, j]. This is what `vnl_data.rs:177` does.
    let beta_flat: Vec<CudaComplex> = beta.iter().map(|&c| complex_to_cuda(c)).collect();
    assert_eq!(beta_flat.len(), ne * n_pw);
    let beta_dev = stream.clone_htod(&beta_flat).expect("H2D beta");

    // ψ upload: production ψ is col-major (n_pw, n_bands) coming out of the
    // step-5a gemm. We construct an equivalent col-major flat directly so
    // there is no second layout to confuse the test.
    let mut psi_col_flat: Vec<CudaComplex> = vec![CudaComplex { x: 0.0, y: 0.0 }; n_pw * n_bands];
    for g in 0..n_pw {
        for b in 0..n_bands {
            psi_col_flat[g + b * n_pw] = complex_to_cuda(psi[[g, b]]);
        }
    }
    let psi_dev = stream.clone_htod(&psi_col_flat).expect("H2D psi");

    // Production gemm (rayleigh_ritz.rs:284-302).
    let mut bp_dev: cudarc::driver::CudaSlice<CudaComplex> =
        stream.alloc_zeros(ne * n_bands).expect("alloc bp");
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,
                transb: op::N,
                m: ne as i32,
                n: n_bands as i32,
                k: n_pw as i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw as i32,
                ldb: n_pw as i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: ne as i32,
            },
            &beta_dev,
            &psi_dev,
            &mut bp_dev,
        )
        .expect("gemm");
    }
    stream.synchronize().expect("sync");

    // Production D2H + F-order reinterpret (density.rs:487-499).
    let bp_host: Vec<CudaComplex> = stream.clone_dtoh(&bp_dev).expect("D2H bp");
    let bp_complex: Vec<Complex64> = bp_host.iter().map(|&c| cuda_to_complex(c)).collect();
    let bp_gemm = Array2::from_shape_vec((ne, n_bands).f(), bp_complex)
        .expect("from_shape_vec");

    // Element-wise comparison.
    let mut max_diff = 0.0_f64;
    for n in 0..ne {
        for b in 0..n_bands {
            let diff = (bp_gemm[[n, b]] - bp_truth_arr[[n, b]]).norm();
            if diff > max_diff { max_diff = diff; }
            if diff > 1e-10 {
                eprintln!(
                    "[n={n},b={b}] gemm={:?} truth={:?}  diff={:.3e}",
                    bp_gemm[[n, b]], bp_truth_arr[[n, b]], diff,
                );
            }
        }
    }
    eprintln!("max |bp_gemm - bp_truth| = {:.3e}", max_diff);
    // Discriminator: brute-force truth values are O(10-100) in magnitude;
    // diff > 1.0 would be ≥ 1% relative error. Place threshold at 1e-10
    // (well within numerical noise of the gemm).
    assert!(
        max_diff < 1e-10,
        "production gemm + F-order reinterpret does NOT match brute-force \
         bp_truth (max diff = {max_diff:.3e}). Layout bug confirmed.",
    );
}

#[test]
#[ignore = "requires GPU"]
fn test_path_arr_iter_h2d_is_layout_inconsistent() {
    // This test demonstrates that the existing pattern in
    // `tests/ca_scf_convergence.rs::aug_density_gpu_matches_cpu_cu111_co`
    // does NOT produce a buffer that `compute_aug_density_gpu` can correctly
    // reinterpret with `.f()`. Used purely as a diagnostic — if it stops
    // failing, the issue has changed.
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let ne: usize = 3;
    let n_bands: usize = 4;

    let bp_logical = deterministic_matrix(ne, n_bands, 0.0);

    let ctx = Arc::new(CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();

    // Test-style H2D: `arr.iter()` on row-major Array2.
    let bp_row_flat: Vec<CudaComplex> = bp_logical.iter().map(|&c| complex_to_cuda(c)).collect();
    let bp_dev_test = stream.clone_htod(&bp_row_flat).expect("H2D");

    // Production-style D2H + F-order reinterpret.
    let bp_host: Vec<CudaComplex> = stream.clone_dtoh(&bp_dev_test).expect("D2H");
    let bp_complex: Vec<Complex64> = bp_host.iter().map(|&c| cuda_to_complex(c)).collect();
    let bp_interpreted = Array2::from_shape_vec((ne, n_bands).f(), bp_complex)
        .expect("from_shape_vec");

    // Find the worst layout mismatch (off-diagonals will differ when ne != n_b).
    let mut max_diff = 0.0_f64;
    let mut n_disagree = 0usize;
    for n in 0..ne {
        for b in 0..n_bands {
            let diff = (bp_interpreted[[n, b]] - bp_logical[[n, b]]).norm();
            if diff > 1e-10 { n_disagree += 1; }
            if diff > max_diff { max_diff = diff; }
        }
    }
    eprintln!("test-path: {n_disagree}/{} elements disagree, max diff = {max_diff:.3e}",
              ne * n_bands);

    // Assert the layout mismatch is real and large (discriminator: > 0.1).
    assert!(
        max_diff > 0.1,
        "expected the row-major-arr.iter() H2D + F-order reinterpret to be \
         layout-inconsistent for ne={ne} != n_bands={n_bands}, but max diff \
         was only {max_diff:.3e}. Either the test is wrong or the layout \
         conventions have changed.",
    );
}
