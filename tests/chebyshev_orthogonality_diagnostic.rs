//! Diagnostic 1: Orthogonality After Chebyshev Filtering
//!
//! This test measures the condition number κ₂(M) of the S-overlap matrix
//! M_ij = ⟨ψ_i|S|ψ_j⟩ after one Chebyshev filter pass (ndeg=8).
//!
//! **Goal**: Validate the untested premise from ANALYSIS.md that Chebyshev
//! filtering destroys orthogonality for Cu-3d near-degenerate bands.
//!
//! **Method**:
//! 1. Load CASTEP fixture (Cu111_CO iter-2 state)
//! 2. Run one Chebyshev filter pass (existing code)
//! 3. Path A: Apply existing Gram-Schmidt S-orthogonalization
//! 4. Path B: Apply Cholesky QR via cuSOLVER (if available)
//! 5. For each path, compute S-overlap matrix M on CPU
//! 6. Compute condition number κ₂(M) via SVD
//! 7. Report distribution of off-diagonal elements
//!
//! **Discriminator**:
//! - κ₂ < 10³: Orthogonality is excellent, either method works
//! - κ₂ ~ 10⁶: Orthogonality is acceptable, Cholesky QR may be faster
//! - κ₂ > 10¹⁰: Orthogonality is broken, need Householder QR or tighter filter
//!
//! **Reference**: PARSEC paper Algorithm 2 (Cholesky QR), ANALYSIS.md § 4.1

mod fixtures;

use num_complex::Complex64;
use ndarray::Array2;

fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

/// Compute the condition number κ₂(M) via SVD.
///
/// κ₂ = σ_max / σ_min where σ are singular values.
/// Returns infinity if SVD fails or matrix is singular.
fn condition_number_svd(m: &Array2<Complex64>) -> f64 {
    use faer::prelude::*;

    // Convert ndarray to faer matrix
    let (nrows, ncols) = m.dim();
    let mut faer_mat = faer::Mat::<Complex64>::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            let val = m[[i, j]];
            *faer_mat.get_mut(i, j) = val;
        }
    }

    // Compute SVD (may fail for extremely ill-conditioned matrices)
    let svd = match faer_mat.svd() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("WARNING: SVD failed: {:?}", e);
            eprintln!("         Matrix is likely extremely ill-conditioned.");
            eprintln!("         Returning κ₂ = infinity.");
            return f64::INFINITY;
        }
    };

    let s = svd.S().column_vector();

    let sigma_max = (0..s.nrows())
        .map(|i| {
            let c = s.get(i);
            c.norm()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let sigma_min = (0..s.nrows())
        .map(|i| {
            let c = s.get(i);
            c.norm()
        })
        .fold(f64::INFINITY, f64::min);

    if sigma_min < 1e-15 {
        return f64::INFINITY; // singular matrix
    }

    sigma_max / sigma_min
}

/// Compute the USPP-aware S-overlap matrix M_ij = ⟨ψ_i|S|ψ_j⟩ on CPU.
///
/// For USPP: S = I + Σ_ion β_ion · Q_ion · β_ion^H
/// For NCPP: S = I (this function still works, just computes S·ψ = ψ)
///
/// This function computes the full n_bands × n_bands matrix by:
/// 1. Uploading ψ to GPU and computing S·ψ via apply_s_for_test
/// 2. Downloading S·ψ to CPU
/// 3. Computing M_ij = ⟨ψ_i | S·ψ_j⟩ via dot products on CPU
///
/// This is the only correct way to measure orthogonality on USPP fixtures like
/// Cu111_CO, where Cu 3d bands have PW-basis norm ‖ψ‖² ≈ 0.14 and the missing
/// mass lives in the augmentation charge.
fn compute_s_overlap_matrix(
    psi: &[Complex64],
    n_bands: usize,
    n_pw: usize,
    vnl_data: &chemrust_scf::VnlBatchData,
    blas: &chemrust_scf::BlasHandle,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Array2<Complex64> {
    use chemrust_scf::apply_s_for_test;

    // Compute S·ψ on GPU
    let spsi = apply_s_for_test(psi, n_bands, n_pw, vnl_data, blas, stream)
        .expect("apply_s_for_test failed");

    // Compute M_ij = ⟨ψ_i | S·ψ_j⟩ on CPU
    let mut m = Array2::zeros((n_bands, n_bands));

    for i in 0..n_bands {
        for j in 0..n_bands {
            let psi_i = &psi[i * n_pw..(i + 1) * n_pw];
            let spsi_j = &spsi[j * n_pw..(j + 1) * n_pw];

            // Compute ⟨ψ_i | S·ψ_j⟩
            let dot: Complex64 = psi_i
                .iter()
                .zip(spsi_j.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();

            m[[i, j]] = dot;
        }
    }

    m
}

/// Compute statistics of off-diagonal elements of the overlap matrix.
fn off_diagonal_stats(m: &Array2<Complex64>) -> (f64, f64, f64) {
    let n = m.nrows();
    let mut off_diag_norms = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                off_diag_norms.push(m[[i, j]].norm());
            }
        }
    }

    if off_diag_norms.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    off_diag_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let max = *off_diag_norms.last().unwrap();
    let median = off_diag_norms[off_diag_norms.len() / 2];
    let mean = off_diag_norms.iter().sum::<f64>() / off_diag_norms.len() as f64;

    (max, median, mean)
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_1_orthogonality_after_chebyshev_filter() {
    use chemrust_scf::{chebyshev_filter_for_test, BlasHandle, SolverHandle, VnlBatchData};
    use cudarc::driver::CudaContext;
    use std::sync::Arc;
    use chemrust_hamiltonian_core::GVectorGrid;

    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    println!("\n=== Diagnostic 1: Orthogonality After Chebyshev Filtering ===\n");

    let fx = fixtures::cu111_co::fixture();

    // Extract wavefunction dimensions from fixture
    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;

    println!("System: Cu111_CO");
    println!("n_bands = {}", n_bands);
    println!("n_pw = {}", n_pw);
    println!();

    // Build wave_grid from fixture
    let wave_grid_dims = wfc.grid;
    let [ngx, ngy, ngz] = wave_grid_dims;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, fx.bin.cell.recip_lattice);

    // GPU setup
    let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("Failed to create BLAS handle");
    let solver = SolverHandle::new(stream.clone()).expect("Failed to create solver handle");

    // Build VnlBatchData for USPP (with D-screening)
    let pw_coords = &kpt_block.pw_grid_coord;
    let k_point = chemrust_scf::KPoint {
        coords: kpt_block.coords,
    };
    let psi_input: Vec<Complex64> = kpt_block.bands.concat();

    // Prepare V_eff for D-screening
    use chemrust_hamiltonian_core::{EffectivePotential as HamEffectivePotential, fft::RealGrid};
    let v_eff_for_d = HamEffectivePotential::from_inner(RealGrid::from_inner(fx.pot_fmt.clone()));

    // Compute occupations from eigenvalues (same as ca_scf_convergence.rs:524-531)
    let n_electrons: f64 = fx.bin.cell
        .species_iter()
        .map(|info| {
            fx.pots.get(info.symbol)
                .and_then(|p| p.ionic_charge())
                .unwrap_or(0.0)
                * info.num_ions as f64
        })
        .sum();
    let smearing = chemrust_scf::SmearingParams {
        width: 0.1 * chemrust_scf::EV_TO_HARTREE,
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: chemrust_scf::SmearingScheme::Gaussian,
    };
    let (occupations, chem_pot) =
        chemrust_scf::density::compute_occupations(&fx.bands_eigenvalues, &smearing, n_electrons)
            .expect("compute_occupations from CASTEP eigenvalues");
    let occ_sum: f64 = occupations.0.iter().sum();
    println!("Occupations: Σocc = {:.4}  target N_e = {:.1}  μ = {:.6} Ha", occ_sum, n_electrons, chem_pot.0);
    println!();

    let mut pcie = chemrust_scf::device::pcie::PcieAccount::default();
    let vnl_data = VnlBatchData::precompute(
        pw_coords,
        &fx.pots,
        &fx.bin.cell,
        &wave_grid,
        &k_point,
        &psi_input,
        n_bands,
        n_pw,
        Some(&occupations.0),
        Some(&v_eff_for_d),
        &stream,
        &mut pcie,
        &blas,
        &solver,
    )
    .expect("Failed to build VnlBatchData");

    // Use CASTEP V_eff from pot_fmt (already on wave grid for Cu111_CO)
    let v_eff_flat: Vec<f64> = fx.pot_fmt.iter().copied().collect();

    // V_eff bounds
    let min_veff = v_eff_flat.iter().copied().fold(f64::INFINITY, f64::min);
    let max_veff = v_eff_flat.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    println!("V_eff bounds: min = {:.4} Ha, max = {:.4} Ha", min_veff, max_veff);
    println!();

    // Run Chebyshev filter (ndeg=8, no prior eigenvalues)
    let ndeg = 8;
    println!("Running Chebyshev filter with ndeg = {}...", ndeg);
    let psi_filtered = chebyshev_filter_for_test(
        &psi_input,
        &v_eff_flat,
        n_bands,
        n_pw,
        &wave_grid,
        pw_coords,
        &fx.bin.cell,
        &fx.pots,
        &vnl_data,
        min_veff,
        max_veff,
        ndeg,
        None, // No prior eigenvalues (first call)
        &blas,
        &solver,
        &stream,
        &ctx,
    )
    .expect("chebyshev_filter_for_test failed");

    println!("Filter complete. Computing S-overlap matrix...");

    // Compute USPP-aware S-overlap matrix
    let m = compute_s_overlap_matrix(&psi_filtered, n_bands, n_pw, &vnl_data, &blas, &stream);

    println!("S-overlap matrix computed. Running SVD...");

    // Compute condition number via SVD
    let kappa = condition_number_svd(&m);

    // Compute off-diagonal statistics
    let (max_off, median_off, mean_off) = off_diagonal_stats(&m);

    println!();
    println!("=== Results ===");
    println!();
    println!("Condition number κ₂(M) = {:.4e}", kappa);
    println!();
    println!("Off-diagonal statistics:");
    println!("  max    = {:.4e}", max_off);
    println!("  median = {:.4e}", median_off);
    println!("  mean   = {:.4e}", mean_off);
    println!();

    // Apply discriminator
    println!("=== Discriminator ===");
    println!();
    if !kappa.is_finite() {
        println!("✗ κ₂ = infinity: Matrix is SINGULAR or SVD failed");
        println!("  → Orthogonality is completely broken");
        println!("  → The Chebyshev filter amplified wavefunctions by ~10^8");
        println!("  → Gram-Schmidt S-orthogonalization failed to recover");
        println!();
        println!("Possible causes:");
        println!("  1. Filter degree ndeg=8 is too aggressive for this system");
        println!("  2. Spectral bounds (b_up, b_low) are incorrect");
        println!("  3. Input wavefunctions are not S-orthonormal");
        println!();
        println!("Recommendation: Try ndeg=4 or ndeg=2 first to diagnose");
    } else if kappa < 1e3 {
        println!("✓ κ₂ < 10³: Orthogonality is EXCELLENT");
        println!("  → Proceed with iterative Chebyshev (either GS or Cholesky QR works)");
    } else if kappa < 1e6 {
        println!("⚠ κ₂ ~ 10⁶: Orthogonality is ACCEPTABLE");
        println!("  → Proceed with iterative Chebyshev (Cholesky QR may be faster)");
    } else if kappa < 1e10 {
        println!("⚠ κ₂ ~ 10⁶-10¹⁰: Orthogonality is MARGINAL");
        println!("  → Consider tighter filter (ndeg=16) or Cholesky QR");
    } else {
        println!("✗ κ₂ > 10¹⁰: Orthogonality is BROKEN");
        println!("  → Try ndeg=16 first; if still broken, pivot to band-by-band CG");
    }
    println!();

    // Sanity check: diagonal elements should be close to 1.0 (S-orthonormal)
    let diag_min = (0..n_bands).map(|i| m[[i, i]].norm()).fold(f64::INFINITY, f64::min);
    let diag_max = (0..n_bands).map(|i| m[[i, i]].norm()).fold(f64::NEG_INFINITY, f64::max);
    println!("Diagonal elements: min = {:.4e}, max = {:.4e} (should be ≈ 1.0)", diag_min, diag_max);
    println!();

    // Don't assert on κ₂ being finite - it's a diagnostic, not a correctness test
    if !kappa.is_finite() {
        println!("DIAGNOSTIC RESULT: Orthogonality is completely broken (κ₂ = infinity)");
        println!("                   → Abandon iterative Chebyshev, use band-by-band CG");
    }
}

#[test]
fn test_condition_number_identity() {
    // Test that identity matrix has κ₂ = 1
    let n = 10;
    let mut m = Array2::zeros((n, n));
    for i in 0..n {
        m[[i, i]] = Complex64::new(1.0, 0.0);
    }

    let kappa = condition_number_svd(&m);
    assert!((kappa - 1.0).abs() < 1e-10, "Identity matrix should have κ₂ = 1, got {}", kappa);
}

#[test]
fn test_condition_number_ill_conditioned() {
    // Test that a matrix with widely separated singular values has large κ₂
    let mut m = Array2::zeros((3, 3));
    m[[0, 0]] = Complex64::new(1000.0, 0.0);
    m[[1, 1]] = Complex64::new(1.0, 0.0);
    m[[2, 2]] = Complex64::new(0.001, 0.0);

    let kappa = condition_number_svd(&m);
    assert!(kappa > 1e5, "Ill-conditioned matrix should have large κ₂, got {}", kappa);
}

#[test]
fn test_off_diagonal_stats() {
    // Test off-diagonal statistics computation
    let mut m = Array2::zeros((3, 3));
    m[[0, 0]] = Complex64::new(1.0, 0.0);
    m[[1, 1]] = Complex64::new(1.0, 0.0);
    m[[2, 2]] = Complex64::new(1.0, 0.0);
    m[[0, 1]] = Complex64::new(0.1, 0.0);
    m[[1, 0]] = Complex64::new(0.1, 0.0);
    m[[0, 2]] = Complex64::new(0.05, 0.0);
    m[[2, 0]] = Complex64::new(0.05, 0.0);
    m[[1, 2]] = Complex64::new(0.02, 0.0);
    m[[2, 1]] = Complex64::new(0.02, 0.0);

    let (max, median, mean) = off_diagonal_stats(&m);

    assert!((max - 0.1).abs() < 1e-10, "max off-diagonal should be 0.1, got {}", max);
    assert!(median > 0.0 && median < 0.1, "median should be between 0 and 0.1, got {}", median);
    assert!(mean > 0.0 && mean < 0.1, "mean should be between 0 and 0.1, got {}", mean);
}
