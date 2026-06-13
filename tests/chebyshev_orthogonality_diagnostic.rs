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

#![cfg(feature = "chebyshev")]

mod fixtures;

use ndarray::Array2;
use num_complex::Complex64;

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

/// Helper function to run orthogonality diagnostic with a specific filter mode
fn run_orthogonality_diagnostic(filter_mode: chemrust_scf::FilterMode, mode_name: &str) {
    use chemrust_hamiltonian_core::GVectorGrid;
    use chemrust_scf::{BlasHandle, SolverHandle, VnlBatchData};
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    println!(
        "\n=== Diagnostic 1: Orthogonality After Chebyshev Filtering ({}) ===\n",
        mode_name
    );

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
    let kernels = chemrust_scf::CudaKernelSet::new(&ctx).expect("CudaKernelSet");

    // Build VnlBatchData for USPP (with D-screening)
    let pw_coords = &kpt_block.pw_grid_coord;
    let k_point = chemrust_scf::KPoint {
        coords: kpt_block.coords,
        weight: 1.0,
    };
    let psi_input: Vec<Complex64> = kpt_block.bands.concat();

    // Prepare V_eff for D-screening
    use chemrust_hamiltonian_core::{EffectivePotential as HamEffectivePotential, fft::RealGrid};
    let v_eff_for_d = HamEffectivePotential::from_inner(RealGrid::from_inner(fx.pot_fmt.clone()));

    // Compute occupations from eigenvalues (same as ca_scf_convergence.rs:524-531)
    let n_electrons: f64 = fx
        .bin
        .cell
        .species_iter()
        .map(|info| {
            fx.pots
                .get(info.symbol)
                .and_then(|p| p.ionic_charge())
                .unwrap_or(0.0)
                * info.num_ions as f64
        })
        .sum();
    let smearing = chemrust_scf::SmearingParams {
        width: 0.1 * chemrust_scf::EV_TO_HARTREE,
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: chemrust_scf::SmearingScheme::Gaussian,
        spin_fix: 10,
    };
    let (occupations, chem_pot) =
        chemrust_scf::density::compute_occupations(&fx.bands_eigenvalues, &smearing, n_electrons, 1.0)
            .expect("compute_occupations from CASTEP eigenvalues");
    let occ_sum: f64 = occupations.0.iter().sum();
    println!(
        "Occupations: Σocc = {:.4}  target N_e = {:.1}  μ = {:.6} Ha",
        occ_sum, n_electrons, chem_pot.0
    );
    println!();

    let mut pcie = chemrust_scf::PcieAccount::default();
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
        &kernels,
    )
    .expect("Failed to build VnlBatchData");

    // Use CASTEP V_eff from pot_fmt (already on wave grid for Cu111_CO)
    let v_eff_flat: Vec<f64> = fx.pot_fmt.iter().copied().collect();

    // V_eff bounds
    let min_veff = v_eff_flat.iter().copied().fold(f64::INFINITY, f64::min);
    let max_veff = v_eff_flat.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "V_eff bounds: min = {:.4} Ha, max = {:.4} Ha",
        min_veff, max_veff
    );
    println!();

    // Run Chebyshev filter (ndeg=8, no prior eigenvalues)
    let ndeg = 8;
    println!(
        "Running Chebyshev filter with ndeg = {}, mode = {}...",
        ndeg, mode_name
    );
    let psi_filtered = chemrust_scf::chebyshev_filter_for_test(
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
        filter_mode,
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
    let diag_min = (0..n_bands)
        .map(|i| m[[i, i]].norm())
        .fold(f64::INFINITY, f64::min);
    let diag_max = (0..n_bands)
        .map(|i| m[[i, i]].norm())
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Diagonal elements: min = {:.4e}, max = {:.4e} (should be ≈ 1.0)",
        diag_min, diag_max
    );
    println!();

    // Don't assert on κ₂ being finite - it's a diagnostic, not a correctness test
    if !kappa.is_finite() {
        println!("DIAGNOSTIC RESULT: Orthogonality is completely broken (κ₂ = infinity)");
        println!("                   → Abandon iterative Chebyshev, use band-by-band CG");
    }
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_1_orthogonality_after_chebyshev_filter() {
    // Original test: BareH mode (physically incorrect for USPP, but baseline)
    run_orthogonality_diagnostic(chemrust_scf::FilterMode::BareH, "BareH");
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_1b_orthogonality_sinvh_keep_h_eig() {
    // CRITICAL: Test the production filter mode (SinvHKeepHEig)
    // This is the mode actually used in the SCF loop for USPP systems.
    // If κ₂ > 10¹⁰ here, the entire PARSEC Algorithm 4 approach may not work.
    run_orthogonality_diagnostic(chemrust_scf::FilterMode::SinvHKeepHEig, "SinvHKeepHEig");
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_1c_orthogonality_sinvh_full_das() {
    // Optional: Test the full Das Algorithm 3 mode
    run_orthogonality_diagnostic(chemrust_scf::FilterMode::SinvHFullDas, "SinvHFullDas");
}

// =========================================================================
// Diagnostic 2: Per-band residual norms after Chebyshev filter + RR
// =========================================================================
//
// Measures per-band S^{-1}-weighted and unweighted L2 residuals after one
// Chebyshev filter pass (ndeg=8, SinvHKeepHEig) followed by standard
// Rayleigh-Ritz diagonalization.
//
// This establishes a baseline: well-separated bands should have small
// residuals, while the Cu 3d degenerate cluster (~-0.50 to -0.47 Ha) is
// expected to show larger residuals due to RR subspace mixing.
//
// **Flow** (all GPU-resident except final scalar norms):
//   1. chebyshev_filter_for_test_gpu() → (psi_row_gpu, hpsi_row_gpu, kernels)
//   2. rayleigh_ritz_with_matrices()  → (psi_new_gpu, eigenvalues, S_sub, H_sub, X)
//   3. compute_residual_norms_for_test() → (sinv_norms, l2_norms)
//   4. classify bands, report, assert against external anchors
// =========================================================================

/// Band class for grouped reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandClass {
    DeepCore,
    Cu3d,
    Valence,
    NearFermi,
    Conduction,
}

/// Classify bands by eigenvalue range (Ha) relative to the Fermi level.
///
/// Classification rules (calibrated on Cu111_CO fixture):
/// - DeepCore:  band 0 only (−1.055 Ha)
/// - Cu3d:      bands 1-14 (−0.50 to −0.47 Ha)
/// - Valence:   occupied region, well below Fermi
/// - NearFermi: within 0.05 Ha of Fermi level (metallic band gap ~0.0001 Ha)
/// - Conduction: above Fermi
fn classify_bands(eigenvalues: &[f64], fermi_energy: f64) -> Vec<BandClass> {
    let mut classes = Vec::with_capacity(eigenvalues.len());
    for (b, &eig) in eigenvalues.iter().enumerate() {
        let cls = match b {
            0 => BandClass::DeepCore,
            1..=14 => BandClass::Cu3d,
            _ => {
                if eig > fermi_energy + 0.05 {
                    BandClass::Conduction
                } else if eig > fermi_energy - 0.05 {
                    BandClass::NearFermi
                } else {
                    BandClass::Valence
                }
            }
        };
        classes.push(cls);
    }
    classes
}

/// Per-group statistics: (max, mean, median, count)
fn group_stats(norms: &[f64], classes: &[BandClass], target: BandClass) -> (f64, f64, f64, usize) {
    let vals: Vec<f64> = norms
        .iter()
        .zip(classes.iter())
        .filter(|&(_, &c)| c == target)
        .map(|(&n, _)| n)
        .collect();
    if vals.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, 0);
    }
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let mut sorted = vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    (max, mean, median, vals.len())
}

/// Compute MAE between two sorted eigenvalue slices.
fn mean_abs_err(rr_eigs: &[f64], ref_eigs: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return f64::NAN;
    }
    indices
        .iter()
        .map(|&b| (rr_eigs[b] - ref_eigs[b]).abs())
        .sum::<f64>()
        / indices.len() as f64
}

/// Diagnostic 2: measure per-band residual norms after one Chebyshev filter + RR pass.
///
/// Observational claims (printed, not asserted):
///   - Cu 3d cluster MAE > well-separated MAE  (RR mixing)
///   - Band 0 has among the lowest residuals (most isolated band)
///
/// Hard assertions (failure = bug):
///   - All residual norms finite and < 1.0 Ha
///   - Eigenvalues are sorted ascending
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_2_residual_norms_after_chebyshev_filter() {
    use chemrust_scf::{BlasHandle, Cpu, SolverHandle, VnlBatchData};
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    println!("\n=== Diagnostic 2: Per-Band Residual Norms After Chebyshev Filter ===\n");

    let fx = fixtures::cu111_co::fixture();

    // Extract wavefunction dimensions
    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;

    println!("System: Cu111_CO  n_bands = {}  n_pw = {}\n", n_bands, n_pw);

    // Build wave_grid
    use chemrust_hamiltonian_core::GVectorGrid;
    let [ngx, ngy, ngz] = wfc.grid;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, fx.bin.cell.recip_lattice);

    // GPU setup
    let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("Failed to create BLAS handle");
    let solver = SolverHandle::new(stream.clone()).expect("Failed to create solver handle");
    let kernels = chemrust_scf::CudaKernelSet::new(&ctx).expect("CudaKernelSet");

    // Build VnlBatchData with D-screening (same as Diagnostic 1)
    let pw_coords = &kpt_block.pw_grid_coord;
    let k_point = chemrust_scf::KPoint {
        coords: kpt_block.coords,
        weight: 1.0,
    };
    let psi_input: Vec<num_complex::Complex64> = kpt_block.bands.concat();

    use chemrust_hamiltonian_core::{EffectivePotential as HamEffectivePotential, fft::RealGrid};
    let v_eff_for_d = HamEffectivePotential::from_inner(RealGrid::from_inner(fx.pot_fmt.clone()));

    // Compute occupations
    let n_electrons: f64 = fx
        .bin
        .cell
        .species_iter()
        .map(|info| {
            fx.pots
                .get(info.symbol)
                .and_then(|p| p.ionic_charge())
                .unwrap_or(0.0)
                * info.num_ions as f64
        })
        .sum();
    let smearing = chemrust_scf::SmearingParams {
        width: 0.1 * chemrust_scf::EV_TO_HARTREE,
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: chemrust_scf::SmearingScheme::Gaussian,
        spin_fix: 10,
    };
    let (occupations, chem_pot) =
        chemrust_scf::density::compute_occupations(&fx.bands_eigenvalues, &smearing, n_electrons, 1.0)
            .expect("compute_occupations");
    let occ_sum: f64 = occupations.0.iter().sum();
    println!(
        "Occupations: Σocc = {:.4}  target N_e = {:.1}  μ = {:.6} Ha\n",
        occ_sum, n_electrons, chem_pot.0
    );

    // Determine number of occupied bands (CASTEP logic: ceiling(n_electrons / 2) for spin-unpolarized)
    let n_occ = (n_electrons / 2.0).ceil() as usize;
    println!(
        "Number of occupied bands: n_occ = {} (from N_e = {})\n",
        n_occ, n_electrons
    );

    let mut pcie = chemrust_scf::PcieAccount::default();
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
        &kernels,
    )
    .expect("VnlBatchData::precompute");

    // CASTEP V_eff from pot_fmt (already on wave grid for Cu111_CO)
    let v_eff_flat: Vec<f64> = fx.pot_fmt.iter().copied().collect();
    let min_veff = v_eff_flat.iter().copied().fold(f64::INFINITY, f64::min);
    let max_veff = v_eff_flat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "V_eff bounds: min = {:.4} Ha, max = {:.4} Ha\n",
        min_veff, max_veff
    );

    // ------------------------------------------------------------------
    // Step 1: Run Chebyshev filter → GPU-resident (psi_row, hpsi_row, kernels)
    // ------------------------------------------------------------------
    let ndeg = 8;
    let eigenvalues = fx.bands_eigenvalues.clone();
    println!("Running Chebyshev filter (ndeg={}, SinvHKeepHEig)...", ndeg);
    let (psi_row_gpu, hpsi_row_gpu, kernels) = chemrust_scf::chebyshev_filter_for_test_gpu(
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
        Some(&eigenvalues),
        chemrust_scf::FilterMode::SinvHKeepHEig,
        &blas,
        &solver,
        &stream,
        &ctx,
    )
    .expect("chebyshev_filter_for_test_gpu failed");
    println!("Filter complete.\n");

    // Verify post-GS S-orthonormality: GS should produce ψ with norm²_S = 1.0
    {
        use chemrust_scf::device::cuda_vec_to_complex;
        let psi_gs_full: Vec<Complex64> = cuda_vec_to_complex(
            stream.clone_dtoh(psi_row_gpu.as_device_slice())
                .expect("clone_dtoh psi_row_gpu"),
        );
        let psi_band0 = &psi_gs_full[0..n_pw];
        let spsi_band0 = chemrust_scf::apply_s_for_test(
            psi_band0, 1, n_pw, &vnl_data, &blas, &stream,
        )
        .expect("apply_s_for_test post-GS");
        let norm_sq_s_post_gs: f64 = psi_band0
            .iter()
            .zip(spsi_band0.iter())
            .map(|(p, sp)| (p.conj() * sp).re)
            .sum();
        println!(
            "[S-norm post-GS] band-0 norm²_S = {:.6e}  (≈1.0 for S-orthonormal)",
            norm_sq_s_post_gs
        );
    }

    // ------------------------------------------------------------------
    // Step 2: Rayleigh-Ritz with matrices (returns subspace matrices for diagnostics)
    // ------------------------------------------------------------------
    println!("Running Rayleigh-Ritz (ZHEGVD)...");
    let rr = chemrust_scf::rayleigh_ritz_with_matrices(
        &psi_row_gpu,
        &hpsi_row_gpu,
        &vnl_data,
        n_bands,
        n_pw,
        &kernels,
        &mut pcie,
        &solver,
        &blas,
        &stream,
        &ctx,
        None, // no pinning (Procrustes would contaminate residual)
        None, // no pinning config
    )
    .expect("rayleigh_ritz_with_matrices failed");
    let psi_new_gpu = rr.0;
    let Cpu(rr_eigenvalues): Cpu<Vec<f64>> = rr.1;
    // X is returned as Cpu<Vec<CudaComplex>> — convert to num_complex::Complex64
    let x = chemrust_scf::device::cuda_vec_to_complex(rr.5.0);
    println!("RR complete.\n");

    // Verify post-RR S-orthonormality: ZHEGVD should produce ψ with norm²_S = 1.0
    {
        use chemrust_scf::device::cuda_vec_to_complex;
        let psi_post_rr_full: Vec<Complex64> = cuda_vec_to_complex(
            stream.clone_dtoh(psi_new_gpu.as_device_slice())
                .expect("clone_dtoh psi_new_gpu"),
        );
        let psi_band0 = &psi_post_rr_full[0..n_pw];
        let spsi_band0 = chemrust_scf::apply_s_for_test(
            psi_band0, 1, n_pw, &vnl_data, &blas, &stream,
        )
        .expect("apply_s_for_test post-RR");
        let norm_sq_s_post_rr: f64 = psi_band0
            .iter()
            .zip(spsi_band0.iter())
            .map(|(p, sp)| (p.conj() * sp).re)
            .sum();
        println!(
            "[S-norm post-RR] band-0 norm²_S = {:.6e}  (≈1.0 for S-orthonormal)",
            norm_sq_s_post_rr
        );
    }

    // ------------------------------------------------------------------
    // Step 3: Compute GPU-resident per-band residual norms
    // ------------------------------------------------------------------
    println!("Computing per-band residual norms (GPU)...");
    let (sinv_norms, l2_norms) = chemrust_scf::compute_residual_norms_for_test(
        &psi_new_gpu,
        &hpsi_row_gpu,
        &rr_eigenvalues,
        &x,
        n_bands,
        n_pw,
        &vnl_data,
        &blas,
        &solver,
        &stream,
    )
    .expect("compute_residual_norms_for_test failed");

    // ------------------------------------------------------------------
    // Step 4: Report + Assert
    // ------------------------------------------------------------------
    let fermi = fx.check.eigenvalues.fermi_energy;
    let classes = classify_bands(&fx.bands_eigenvalues, fermi);

    // ----- Report: per-band table -----
    println!("\n--- Per-Band Results ---\n");
    println!(
        "{0:>4} | {1:>12} | {2:>12} | {3:>12} | {4:>12} | {5:>10}",
        "band", "RR_eig (Ha)", "ref_eig (Ha)", "sinv_resid", "l2_resid", "class"
    );
    println!(
        "{:-<4}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<10}",
        "", "", "", "", "", ""
    );

    for b in 0..n_bands {
        let cls = match classes[b] {
            BandClass::DeepCore => "core",
            BandClass::Cu3d => "cu3d",
            BandClass::Valence => "val",
            BandClass::NearFermi => "nFermi",
            BandClass::Conduction => "cond",
        };
        println!("{b:4} | {rr:>12.6} | {ref:>12.6} | {sr:>12.3e} | {lr:>12.3e} | {cls:>10}",
                 b = b,
                 rr = rr_eigenvalues[b],
                 ref = fx.bands_eigenvalues[b],
                 sr = sinv_norms[b],
                 lr = l2_norms[b],
                 cls = cls);
    }

    // ----- Per-group statistics -----
    println!("\n--- Per-Group Statistics ---\n");
    for cls in &[
        BandClass::DeepCore,
        BandClass::Cu3d,
        BandClass::Valence,
        BandClass::NearFermi,
        BandClass::Conduction,
    ] {
        let (max_s, mean_s, med_s, cnt) = group_stats(&sinv_norms, &classes, *cls);
        let (max_l, mean_l, med_l, _) = group_stats(&l2_norms, &classes, *cls);
        let label = format!("{:?}", cls);
        println!(
            "{label:12} (n={cnt:3}): S⁻¹ norm   max={max_s:.3e}  mean={mean_s:.3e}  median={med_s:.3e}",
            cnt = cnt,
            max_s = max_s,
            mean_s = mean_s,
            med_s = med_s
        );
        println!(
            "{label:12}          L₂ norm     max={max_l:.3e}  mean={mean_l:.3e}  median={med_l:.3e}\n",
            max_l = max_l,
            mean_l = mean_l,
            med_l = med_l
        );
    }

    // ----- Eigenvalue accuracy vs CASTEP reference -----
    let well_sep: Vec<usize> = (0..n_bands)
        .filter(|&b| {
            let gap = if b == 0 {
                fx.bands_eigenvalues[1] - fx.bands_eigenvalues[0]
            } else if b == n_bands - 1 {
                fx.bands_eigenvalues[b] - fx.bands_eigenvalues[b - 1]
            } else {
                (fx.bands_eigenvalues[b + 1] - fx.bands_eigenvalues[b - 1]) / 2.0
            };
            classes[b] != BandClass::Cu3d && gap > 0.01
        })
        .collect();
    let cu3d: Vec<usize> = (0..n_bands)
        .filter(|&b| classes[b] == BandClass::Cu3d)
        .collect();

    let mae_sep = mean_abs_err(&rr_eigenvalues, &fx.bands_eigenvalues, &well_sep);
    let mae_cu3d = mean_abs_err(&rr_eigenvalues, &fx.bands_eigenvalues, &cu3d);
    println!("--- Eigenvalue Accuracy (MAE vs CASTEP) ---");
    println!(
        "  Well-separated bands (n={}): MAE = {:.3e} Ha",
        well_sep.len(),
        mae_sep
    );
    println!(
        "  Cu 3d cluster      (n={}): MAE = {:.3e} Ha",
        cu3d.len(),
        mae_cu3d
    );
    println!("  Ratio (cu3d / sep): {:.2}\n", mae_cu3d / mae_sep);

    // ----- Overall metrics -----
    let max_sinv = sinv_norms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_l2 = l2_norms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_sinv_band = sinv_norms
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(b, _)| b)
        .unwrap_or(0);

    println!("--- Summary ---");
    println!(
        "  Max S⁻¹ residual: {:.3e} Ha (band {})",
        max_sinv, max_sinv_band
    );
    println!("  Max L2 residual:  {:.3e} Ha", max_l2);
    println!("  Band 0 S⁻¹ residual: {:.3e} Ha", sinv_norms[0]);

    // ------------------------------------------------------------------
    // Assertions
    // ------------------------------------------------------------------

    // Anchor 4: eigenvalues are sorted
    for b in 1..n_bands {
        assert!(
            rr_eigenvalues[b] >= rr_eigenvalues[b - 1] - 1e-10,
            "Eigenvalues not sorted at band {}: {} < {}",
            b,
            rr_eigenvalues[b],
            rr_eigenvalues[b - 1]
        );
    }
    println!("✓ Eigenvalues are sorted ascending");

    // Anchor 3: sanity ceiling — no residual > 1.0 Ha
    assert!(
        max_sinv < 1.0,
        "Max S⁻¹ residual {:.3e} Ha exceeds 1.0 Ha sanity ceiling (band {})",
        max_sinv,
        max_sinv_band
    );
    println!(
        "✓ Max S⁻¹ residual {:.3e} Ha < 1.0 Ha sanity ceiling",
        max_sinv
    );

    // Anchor 1: well-separated eigenvalue MAE (reported, not fatal)
    if mae_sep > 0.01 {
        eprintln!(
            "NOTE: well-separated eigenvalue MAE = {:.3e} Ha > 0.01 Ha",
            mae_sep
        );
        eprintln!("      May indicate filter spectral bounds or convergence issues.");
    } else {
        println!(
            "✓ Well-separated eigenvalue MAE = {:.3e} Ha < 0.01 Ha",
            mae_sep
        );
    }

    // Anchor 2: Cu 3d MAE ratio (observational discriminator)
    let ratio = mae_cu3d / mae_sep;
    println!(
        "  Cu 3d / well-sep MAE ratio = {:.2} (>{:.0} = RR mixing expected)",
        ratio, 1.0
    );

    // Anchor 5: band 0 (deep core) should be among 5 lowest residuals
    let mut sorted_idx: Vec<usize> = (0..n_bands).collect();
    sorted_idx.sort_by(|&a, &b| sinv_norms[a].partial_cmp(&sinv_norms[b]).unwrap());
    let rank0 = sorted_idx
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(usize::MAX);
    if rank0 < 5 {
        println!("✓ Band 0 is among the 5 lowest residuals (rank {})", rank0);
    } else {
        eprintln!("NOTE: Band 0 residual rank = {} (expected in top 5)", rank0);
    }

    println!("\n=== Diagnostic 2 Complete ===\n");
}

// =========================================================================
// Diagnostic 3: Outer Loop Convergence Test
//
// Tests if iterating Chebyshev filter → Rayleigh-Ritz → residual check
// converges monotonically over 10 iterations.
//
// Flow:
//   1. Setup: upload V_eff, FFT indices, compile kernels (one-time)
//   2. Upload initial ψ from .check (iter 0)
//   3. For each iteration:
//      a. chebyshev_filter_iteration_gpu (reuses pre-built state)
//      b. rayleigh_ritz_with_matrices
//      c. compute_residual_norms_for_test
//      d. track history, psi_gpu = psi_new (move)
//   4. Verify SC-1 through SC-5
// =========================================================================

/// Upload wavefunction host data to GPU as ColumnDistributed.
fn upload_psi_to_gpu_column(
    psi_host: &[Complex64],
    n_bands: usize,
    n_pw: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    pcie: &mut chemrust_scf::PcieAccount,
) -> chemrust_scf::Gpu<chemrust_scf::WavefunctionSet<chemrust_scf::ColumnDistributed>> {
    use chemrust_scf::{ColumnDistributed, Gpu, WavefunctionSet};
    let wfc = WavefunctionSet::<ColumnDistributed>::new(psi_host.to_vec(), n_bands, n_pw);
    Gpu::from_host_with(&wfc, stream, pcie).expect("Failed to upload psi to GPU")
}

/// Compute mean residual over a set of band indices.
fn compute_mean_residual(norms: &[f64], band_indices: &[usize]) -> f64 {
    if band_indices.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = band_indices.iter().map(|&b| norms[b]).sum();
    sum / band_indices.len() as f64
}

/// Count bands in a range with residual below a threshold.
fn count_converged_bands(norms: &[f64], band_range: &[usize], threshold: f64) -> usize {
    band_range.iter().filter(|&&b| norms[b] < threshold).count()
}

/// Compute the maximum absolute eigenvalue drift between two iterations.
fn compute_max_eigenvalue_drift(eigs_a: &[f64], eigs_b: &[f64]) -> f64 {
    eigs_a
        .iter()
        .zip(eigs_b.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f64, f64::max)
}

/// Verify SC-1: residual monotonicity per group.
///
/// Criteria:
/// - residual[N] ≤ residual[N-1] × 1.05 for most iterations (≤2 non-consecutive violations)
/// - At least one of {iter-5, iter-10} shows residual[N] < residual[1] × 0.8
/// - Groups in `skip_reduction_groups` are exempt from the 20% reduction check
///   (they already start near convergence from the .check restart)
fn verify_residual_monotonicity(
    history: &[Vec<f64>],
    groups: &[(&str, &[usize])],
    skip_reduction_groups: &[&str],
) {
    println!("\n--- SC-1: Residual Monotonicity ---");
    for (name, band_indices) in groups {
        let means: Vec<f64> = history
            .iter()
            .map(|norms| compute_mean_residual(norms, band_indices))
            .collect();

        // Check monotonicity: residual[N] ≤ residual[N-1] × 1.05
        let mut violations = Vec::new();
        for n in 1..means.len() {
            if means[n] > means[n - 1] * 1.05 {
                violations.push(n);
            }
        }

        // Check 20% reduction by iter-5 or iter-10
        let has_reduction = if means.len() > 5 {
            means[4] < means[0] * 0.8 || means[means.len() - 1] < means[0] * 0.8
        } else {
            false
        };

        let violations_ok = violations.len() <= 2;
        let non_consec = violations.windows(2).all(|w| w[1] - w[0] > 1);

        println!(
            "  {:>8}: start={:.3e} iter5={:.3e} iter10={:.3e} violations={}{} reduction={}",
            name,
            means[0],
            if means.len() > 4 { means[4] } else { f64::NAN },
            *means.last().unwrap_or(&f64::NAN),
            if violations_ok { "✓" } else { "✗" },
            if !violations.is_empty() {
                format!(" ({})", violations.len())
            } else {
                String::new()
            },
            if has_reduction { "✓" } else { "✗" },
        );

        assert!(
            violations_ok && non_consec,
            "SC-1 failed for '{}': {} violations (max 2, non-consecutive)",
            name,
            violations.len()
        );

        // Skip 20% reduction check for groups that start near convergence
        // (e.g., core band from .check restart)
        let skip_reduction = skip_reduction_groups.contains(&name);
        if !skip_reduction {
            assert!(
                has_reduction,
                "SC-1 failed for '{}': no 20%% reduction by iter-5 or iter-10",
                name
            );
        }
    }
}

/// Verify SC-5: band 0 eigenvalue stays within [lo, hi] across all iterations.
fn verify_band0_stability(history: &[Vec<f64>], lo: f64, hi: f64) {
    println!("\n--- SC-5: Band 0 Stability ---");
    for (iter, eigs) in history.iter().enumerate() {
        let val = eigs[0];
        let ok = val >= lo && val <= hi;
        println!(
            "  iter={}: band0={:.6} Ha  {}",
            iter + 1,
            val,
            if ok { "✓" } else { "✗" }
        );
        assert!(
            ok,
            "SC-5 failed at iter {}: band 0 eigenvalue {:.6} Ha outside [{}, {}]",
            iter + 1,
            val,
            lo,
            hi
        );
    }
}

/// Validate S-orthonormality baseline of CASTEP .check wavefunctions.
///
/// CASTEP stores wavefunctions in grid-sum convention where
/// Σ ψ*[i]·(S·ψ)[i] = 1 for an S-orthonormal wavefunction (no Ω/N_grid factor).
/// This test verifies that our S-operator convention matches and that the
/// norm²_S ≈ 1.0 expectation is correct, ruling out a false positive in the
/// Gram-Schmidt diagnostic output.
///
/// Fast setup: GPU + vnl_data + 1-band apply_s_for_test → ~10 seconds.
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_s_norm_baseline() {
    use chemrust_scf::{BlasHandle, SolverHandle, VnlBatchData};
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    println!("\n=== S-Baseline: S-orthonormality Convention Validation ===\n");

    let fx = fixtures::cu111_co::fixture();
    let kpt_block = &fx.check.wavefunction.as_ref()
        .expect(".check must have wavefunction").kpt_data[0];
    let n_pw = kpt_block.nplw;
    let psi_input: Vec<Complex64> = kpt_block.bands.concat();

    let ctx = Arc::new(CudaContext::new(0).expect("CudaContext"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("BlasHandle");
    let solver = SolverHandle::new(stream.clone()).expect("SolverHandle");
    let kernels = chemrust_scf::CudaKernelSet::new(&ctx).expect("CudaKernelSet");

    // Compute occupations (needed for D-screening in VnlBatchData)
    let n_electrons: f64 = fx.bin.cell.species_iter()
        .map(|info| fx.pots.get(info.symbol).and_then(|p| p.ionic_charge()).unwrap_or(0.0) * info.num_ions as f64)
        .sum();
    let (occupations, _) = chemrust_scf::density::compute_occupations(
        &fx.bands_eigenvalues,
        &chemrust_scf::SmearingParams {
            width: 0.1 * chemrust_scf::EV_TO_HARTREE,
            electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
            scheme: chemrust_scf::SmearingScheme::Gaussian,
            spin_fix: 10,
        },
        n_electrons,
        1.0,
    ).expect("compute_occupations");

    let wave_grid = chemrust_hamiltonian_core::GVectorGrid::new(
        fx.check.wavefunction.as_ref().unwrap().grid[0],
        fx.check.wavefunction.as_ref().unwrap().grid[1],
        fx.check.wavefunction.as_ref().unwrap().grid[2],
        fx.bin.cell.recip_lattice,
    );

    let mut pcie = chemrust_scf::PcieAccount::default();
    let vnl_data = VnlBatchData::precompute(
        &kpt_block.pw_grid_coord,
        &fx.pots,
        &fx.bin.cell,
        &wave_grid,
        &chemrust_scf::KPoint { coords: kpt_block.coords, weight: 1.0 },
        &psi_input,
        kpt_block.bands.len(),
        n_pw,
        Some(&occupations.0),
        Some(&chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(fx.pot_fmt.clone()),
        )),
        &stream,
        &mut pcie,
        &blas,
        &kernels,
    ).expect("VnlBatchData::precompute");

    // Compute norm²_S for band 0 of the CASTEP .check wavefunction
    let psi_band0_host = &psi_input[0..n_pw];
    let spsi_band0 = chemrust_scf::apply_s_for_test(
        psi_band0_host, 1, n_pw, &vnl_data, &blas, &stream,
    ).expect("apply_s_for_test");
    let norm_sq_s_baseline: f64 = psi_band0_host.iter()
        .zip(spsi_band0.iter())
        .map(|(p, sp)| (p.conj() * sp).re)
        .sum();
    println!(
        "[S-norm baseline] CASTEP ψ₀: norm²_S = {:.6e}  (grid-sum convention; ≈1.0 expected)",
        norm_sq_s_baseline
    );
    let deviation = (norm_sq_s_baseline - 1.0).abs();
    if deviation < 0.01 {
        println!("  ✓ Convention confirmed (deviation = {:.2e})\n", deviation);
    } else {
        println!("  ⚠ norm²_S ≠ 1.0 by {:.2e} — convention may differ\n", deviation);
    }
}

/// Validate V_eff decomposition using `VEffBuilder`.
///
/// Uses `VEffBuilder::assemble_on_fine_grid` to reconstruct V_eff from the
/// wave-grid density (which includes NLCC via `reconstruct_rho_core`) and
/// compares against the reference V_eff from `.pot_fmt`.
///
/// If max|residual| ≈ 0, the Hamiltonian potential matches our reconstruction,
/// narrowing the H mismatch to kinetic energy, D_screened, or FFT conventions.
/// If max|residual| ≫ 0, the V_eff construction itself is the root cause.
///
/// CPU-only (no GPU needed).
#[test]
fn diagnostic_veff_decomposition() {
    use chemrust_hamiltonian_core::{
        GVectorGrid,
        band_structure::VEffBuilder,
        NonSpin,
    };

    println!("\n=== V_eff Decomposition: VEffBuilder vs pot_fmt ===\n");

    let fx = fixtures::cu111_co::fixture();

    let wave_grid = GVectorGrid::new(
        fx.bin.density.grid[0],
        fx.bin.density.grid[1],
        fx.bin.density.grid[2],
        fx.bin.cell.recip_lattice,
    );
    let fg_dims = fx.check.fine_grid.expect(".check must have fine_grid");
    let fine_grid = GVectorGrid::new(fg_dims[0], fg_dims[1], fg_dims[2], fx.bin.cell.recip_lattice);

    // Build V_eff via VEffBuilder (handles V_ion, V_H, NLCC → V_xc automatically)
    let builder = VEffBuilder::<NonSpin>::new(&fx.bin.cell, &fx.pots, &wave_grid);
    let veff_reconstructed = builder
        .assemble_on_fine_grid(&fx.bin.density.charge, &wave_grid, &fine_grid)
        .expect("VEffBuilder::assemble_on_fine_grid failed");
    let veff_rec_arr = veff_reconstructed.as_real_grid().as_real_array();

    // Reference V_eff from .pot_fmt
    let veff_ref_arr = &fx.pot_fmt;

    // Residual: reconstructed vs reference
    let residual: ndarray::Array3<f64> = veff_rec_arr - veff_ref_arr;

    fn mi(x: &ndarray::Array3<f64>) -> f64 { x.iter().cloned().fold(f64::INFINITY, f64::min) }
    fn ma(x: &ndarray::Array3<f64>) -> f64 { x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    fn me(x: &ndarray::Array3<f64>) -> f64 { x.iter().sum::<f64>() / x.len() as f64 }

    println!("               min          max          mean");
    println!("V_rec  : [{:>10.4e}, {:>10.4e}, {:>10.4e}]", mi(veff_rec_arr), ma(veff_rec_arr), me(veff_rec_arr));
    println!("V_ref  : [{:>10.4e}, {:>10.4e}, {:>10.4e}]", mi(veff_ref_arr), ma(veff_ref_arr), me(veff_ref_arr));
    let mr = residual.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    println!("residual: [{:>10.4e}, {:>10.4e}]  max|r|={:.4e}", mi(&residual), ma(&residual), mr);

    if mr < 1e-10 {
        println!("  ✓ V_eff reconstruction matches reference to machine precision");
    } else if mr < 1e-4 {
        println!("  ~ V_eff reconstruction: max|r| = {:.2e} (likely fine-grid upsample rounding)", mr);
    } else {
        println!("  ⚠ V_eff mismatch: max|r| = {:.4e} — VEffBuilder != pot_fmt", mr);
    }
    println!();
}

/// Compute per-band kinetic energy expectation T_b = Σ|ψ_b(G)|² × ½|G|².
///
/// Compares against CASTEP eigenvalues to isolate the non-local (D_screened)
/// contribution: ⟨V_nl⟩_b ≈ ε_b − T_b. Since V_eff is already validated,
/// the residual ε_b − T_b − ⟨V_eff⟩_b depends only on D_screened correctness.
///
/// CPU-only (no GPU needed).
#[test]
fn diagnostic_kinetic_energy() {
    println!("\n=== Kinetic Energy: T_b = Σ|ψ_b(G)|² × ½|G|² ===\n");

    let fx = fixtures::cu111_co::fixture();
    let kpt_block = &fx.check.wavefunction.as_ref()
        .expect(".check must have wavefunction").kpt_data[0];
    let pw_coords = &kpt_block.pw_grid_coord;
    let r = fx.bin.cell.recip_lattice.as_array();

    // ½|G|² per PW component
    let kinetic: Vec<f64> = pw_coords.iter()
        .map(|&[h, k, l]| {
            let (hf, kf, lf) = (h as f64, k as f64, l as f64);
            let gx = hf * r[0][0] + kf * r[1][0] + lf * r[2][0];
            let gy = hf * r[0][1] + kf * r[1][1] + lf * r[2][1];
            let gz = hf * r[0][2] + kf * r[1][2] + lf * r[2][2];
            0.5 * (gx * gx + gy * gy + gz * gz)
        })
        .collect();

    let kinetic_max = kinetic.iter().cloned().fold(0.0_f64, f64::max);
    let kinetic_min = kinetic.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("Kinetic energy per PW: min={:.4e} Ha, max={:.4e} Ha", kinetic_min, kinetic_max);
    println!();

    // T_b per band
    println!("{:>6} {:>14} {:>14} {:>14} {:>14}", "band", "T_b (Ha)", "ε_b (Ha)", "ε_b−T_b", "label");
    for (b, psi_band) in kpt_block.bands.iter().enumerate() {
        let t_band: f64 = psi_band.iter()
            .zip(kinetic.iter())
            .map(|(&c, &ke)| c.norm_sqr() * ke)
            .sum();
        let eps_b = fx.bands_eigenvalues[b];
        let label = if b < 1 { "core" }
            else if b < 15 { "cu3d" }
            else if b < 82 { "val" }
            else if b < 93 { "nFermi" }
            else if b < 97 { "near-F" }
            else { "cond" };
        println!("{:>6} {:>14.6e} {:>14.6e} {:>14.6e} {:>6}", b, t_band, eps_b, eps_b - t_band, label);
    }
    println!();
}

/// Differential H·ψ diagnostic: compare ⟨ψ|H|ψ⟩/⟨ψ|S|ψ⟩ against CASTEP ε.
///
/// Starts from CASTEP .check wavefunctions (S-orthonormal, exact eigenfunctions
/// of CASTEP's H). Applies our full Hamiltonian H·ψ = (T+V_eff+V_nl)·ψ,
/// then computes per-band Rayleigh quotients RQ_b = ⟨ψ_b|H|ψ_b⟩/⟨ψ_b|S|ψ_b⟩.
///
/// Δ_b = RQ_b − ε_b(CASTEP) — if non-zero, our H differs from CASTEP's H.
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_h_psi_differential() {
    use chemrust_scf::{
        BlasHandle, CudaKernelSet, EffectivePotential, FineGridArray, Gpu,
        SolverHandle, VnlBatchData, WaveGridArray, apply_h_components_for_test,
        apply_s_for_test, HComponentsForTest,
    };
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    println!("\n=== H·ψ Differential: ⟨ψ|H|ψ⟩/⟨ψ|S|ψ⟩ vs CASTEP ε ===\n");

    let fx = fixtures::cu111_co::fixture();
    let wfc = fx.check.wavefunction.as_ref()
        .expect(".check must have wavefunction");
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;
    let psi_input: Vec<num_complex::Complex64> = kpt_block.bands.concat();

    // GPU setup
    let ctx = Arc::new(CudaContext::new(0).expect("CudaContext"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("BlasHandle");
    let solver = SolverHandle::new(stream.clone()).expect("SolverHandle");
    let kernels = CudaKernelSet::new(&ctx).expect("CudaKernelSet");
    let wave_grid = chemrust_hamiltonian_core::GVectorGrid::new(
        wfc.grid[0], wfc.grid[1], wfc.grid[2],
        fx.bin.cell.recip_lattice,
    );

    let mut pcie = chemrust_scf::PcieAccount::default();
    let pw_coords = &kpt_block.pw_grid_coord;
    let k_point = chemrust_scf::KPoint { coords: kpt_block.coords, weight: 1.0 };

    let n_electrons: f64 = fx.bin.cell.species_iter()
        .map(|info| fx.pots.get(info.symbol).and_then(|p| p.ionic_charge()).unwrap_or(0.0) * info.num_ions as f64)
        .sum();
    let (occupations, _) = chemrust_scf::density::compute_occupations(
        &fx.bands_eigenvalues,
        &chemrust_scf::SmearingParams {
            width: 0.1 * chemrust_scf::EV_TO_HARTREE,
            electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
            scheme: chemrust_scf::SmearingScheme::Gaussian,
            spin_fix: 10,
        }, n_electrons, 1.0,
    ).expect("compute_occupations");

    let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
        chemrust_hamiltonian_core::fft::RealGrid::from_inner(fx.pot_fmt.clone()));
    let vnl_data = VnlBatchData::precompute(
        pw_coords, &fx.pots, &fx.bin.cell, &wave_grid, &k_point,
        &psi_input, n_bands, n_pw, Some(&occupations.0), Some(&v_eff_for_d),
        &stream, &mut pcie, &blas, &kernels,
    ).expect("VnlBatchData");

    // Upload V_eff to GPU
    let v_eff_arr = ndarray::Array3::from_shape_vec(
        (wave_grid.grid()[2], wave_grid.grid()[1], wave_grid.grid()[0]),
        fx.pot_fmt.iter().copied().collect(),
    ).unwrap();
    let v_eff_wave = WaveGridArray::from_inner(v_eff_arr);
    let v_eff_fine = FineGridArray::from_inner(v_eff_wave.into_inner());
    let v_eff_gpu = Gpu::from_host_with(
        &chemrust_scf::EffectivePotential::from_inner(v_eff_fine),
        &stream, &mut pcie,
    ).expect("V_eff upload");

    // Upload psi to GPU
    let psi_gpu = upload_psi_to_gpu_column(&psi_input, n_bands, n_pw, &stream, &mut pcie);

    // FFT index map
    let fft_idx: Vec<i32> = chemrust_scf::pw_coords_to_fft_indices(pw_coords, &wave_grid);
    let fft_idx_dev = stream.clone_htod(&fft_idx).expect("FFT indices");

    // Apply full Hamiltonian: H·ψ (returns CPU Vec<Complex64> directly)
    let h_components: HComponentsForTest = apply_h_components_for_test(
        &psi_gpu, &v_eff_gpu, &wave_grid, pw_coords, &vnl_data,
        &fft_idx_dev, &kernels, &blas, &stream,
    ).expect("apply_h_components_for_test");

    // Download ψ from GPU to CPU for dot products
    let psi_raw: Vec<num_complex::Complex64> = stream.clone_dtoh(psi_gpu.as_device_slice())
        .expect("psi D2H")
        .into_iter()
        .map(|c| num_complex::Complex64::new(c.x, c.y))
        .collect();

    // Apply S·ψ (returns CPU Vec<Complex64> directly)
    let spsi = apply_s_for_test(&psi_input, n_bands, n_pw, &vnl_data, &blas, &stream)
        .expect("apply_s_for_test");

    // Per-band diagnostics
    struct BandDiag { b: usize, rq: f64, eps: f64, delta: f64,
        t_expect: f64, tv_expect: f64, s_norm: f64, label: String }
    let mut diags: Vec<BandDiag> = Vec::with_capacity(n_bands);

    for b in 0..n_bands {
        let off = b * n_pw;
        let psi_b = &psi_raw[off..off + n_pw];
        let h_b = &h_components.hpsi_full[off..off + n_pw];
        let tv_b = &h_components.hpsi_tv[off..off + n_pw];
        let t_b = &h_components.hpsi_t[off..off + n_pw];
        let s_b = &spsi[off..off + n_pw];

        let h_dot: f64 = psi_b.iter().zip(h_b).map(|(p, h)| (p.conj() * h).re).sum();
        let tv_dot: f64 = psi_b.iter().zip(tv_b).map(|(p, h)| (p.conj() * h).re).sum();
        let t_dot: f64 = psi_b.iter().zip(t_b).map(|(p, h)| (p.conj() * h).re).sum();
        let s_dot: f64 = psi_b.iter().zip(s_b).map(|(p, s)| (p.conj() * s).re).sum();

        let rq = if s_dot.abs() > 1e-30 { h_dot / s_dot } else { 0.0 };
        let eps = fx.bands_eigenvalues[b];
        let delta = rq - eps;

        let label = if b < 1 { "core" } else if b < 15 { "cu3d" }
            else if b < 82 { "val" } else if b < 93 { "nFermi" }
            else { "cond" };

        diags.push(BandDiag { b, rq, eps, delta, t_expect: t_dot, tv_expect: tv_dot, s_norm: s_dot, label: label.into() });
    }

    // Print summary: max|Δ| per group
    println!("Group       bands  max|Δ| (Ha)  max|Δ|/|ε|  band@max  ⟨S⟩ range");
    for (gname, gbands) in [("core", 0..1), ("cu3d", 1..15), ("val", 15..82), ("nFermi", 82..93), ("cond", 93..160)] {
        let mut max_delta = 0.0_f64;
        let mut max_b = 0usize;
        let mut s_min = f64::INFINITY;
        let mut s_max = 0.0_f64;
        for d in diags.iter().filter(|d| gbands.contains(&d.b)) {
            let ad = d.delta.abs();
            if ad > max_delta { max_delta = ad; max_b = d.b; }
            s_min = s_min.min(d.s_norm);
            s_max = s_max.max(d.s_norm);
        }
        let eps_at_max = diags[max_b].eps;
        let rel = if eps_at_max.abs() > 1e-10 { max_delta / eps_at_max.abs() } else { max_delta };
        println!("{:<10} {b:>3}..{e:<3} {max_delta:>10.3e}  {rel:>10.3e}  {max_b:>5}  [{s_min:.4}, {s_max:.4}]",
            gname, b=gbands.start, e=gbands.end-1, max_delta=max_delta, rel=rel, max_b=max_b, s_min=s_min, s_max=s_max);
    }

    // Top offenders
    diags.sort_by(|a, b| b.delta.abs().partial_cmp(&a.delta.abs()).unwrap());
    println!("\nTop 10 bands by |Δ|:");
    println!("  band |  Δ (Ha)  | RQ (Ha)  | ε (Ha)  | label | ⟨S⟩");
    for d in diags.iter().take(10) {
        println!("  {:>4} | {:>8.2e} | {:>8.4} | {:>8.4} | {:<6} | {:.4}",
            d.b, d.delta, d.rq, d.eps, d.label, d.s_norm);
    }

    // Component breakdown for band 0
    let b0 = diags.iter().find(|d| d.b == 0).unwrap();
    println!("\nComponent breakdown (band 0):");
    println!("  ⟨T⟩      = {:.6e} Ha", b0.t_expect);
    println!("  ⟨T+V_eff⟩= {:.6e} Ha", b0.tv_expect);
    println!("  ⟨V_nl⟩   = {:.6e} Ha  (H_full − H_tv)", b0.rq * b0.s_norm - b0.tv_expect);
    println!("  ⟨S⟩      = {:.6e}", b0.s_norm);
    println!("  RQ       = {:.6e} Ha  (⟨H⟩/⟨S⟩)", b0.rq);
    println!("  ε_ref    = {:.6e} Ha", b0.eps);
    println!("  Δ        = {:.6e} Ha", b0.delta);
    println!();
}

/// Diagnostic 3: outer loop convergence test.
///
/// Tests whether 10 iterations of Chebyshev filter → Rayleigh-Ritz → residual
/// check produce monotonically decreasing residuals (SC-1), conduction band
/// convergence within 5 iterations (SC-2), ≥5× occupied band reduction (SC-3),
/// eigenvalue stability (SC-4), and no cascade divergence (SC-5).
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_3_outer_loop_convergence() {
    use chemrust_scf::{
        BlasHandle, Cpu, EffectivePotential, FilterMode, FineGridArray, Gpu, SolverHandle,
        VnlBatchData, WaveGridArray,
    };
    use cudarc::driver::{CudaContext, CudaSlice};
    use ndarray::Array3;
    use std::sync::Arc;

    use chemrust_hamiltonian_core::{
        EffectivePotential as HamEffectivePotential, GVectorGrid, fft::RealGrid,
    };

    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    println!("\n=== Diagnostic 3: Outer Loop Convergence Test (10 iterations) ===\n");

    let fx = fixtures::cu111_co::fixture();

    // Extract wavefunction dimensions
    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;
    let [ngx, ngy, ngz] = wfc.grid;

    println!("System: Cu111_CO  n_bands = {}  n_pw = {}\n", n_bands, n_pw);

    // Build wave_grid
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, fx.bin.cell.recip_lattice);

    // GPU setup
    let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("Failed to create BLAS handle");
    let solver = SolverHandle::new(stream.clone()).expect("Failed to create solver handle");
    let kernels = chemrust_scf::CudaKernelSet::new(&ctx).expect("CudaKernelSet");

    // Build VnlBatchData with D-screening
    let pw_coords = &kpt_block.pw_grid_coord;
    let k_point = chemrust_scf::KPoint {
        coords: kpt_block.coords,
        weight: 1.0,
    };
    let psi_input: Vec<Complex64> = kpt_block.bands.concat();

    let v_eff_for_d = HamEffectivePotential::from_inner(RealGrid::from_inner(fx.pot_fmt.clone()));

    // Compute occupations
    let n_electrons: f64 = fx
        .bin
        .cell
        .species_iter()
        .map(|info| {
            fx.pots
                .get(info.symbol)
                .and_then(|p| p.ionic_charge())
                .unwrap_or(0.0)
                * info.num_ions as f64
        })
        .sum();
    let smearing = chemrust_scf::SmearingParams {
        width: 0.1 * chemrust_scf::EV_TO_HARTREE,
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: chemrust_scf::SmearingScheme::Gaussian,
        spin_fix: 10,
    };
    let (occupations, chem_pot) =
        chemrust_scf::density::compute_occupations(&fx.bands_eigenvalues, &smearing, n_electrons, 1.0)
            .expect("compute_occupations");
    let occ_sum: f64 = occupations.0.iter().sum();
    println!(
        "Occupations: Σocc = {:.4}  target N_e = {:.1}  μ = {:.6} Ha\n",
        occ_sum, n_electrons, chem_pot.0
    );

    // Determine number of occupied bands (CASTEP logic: ceiling(n_electrons / 2) for spin-unpolarized)
    let n_occ = (n_electrons / 2.0).ceil() as usize;
    println!(
        "Number of occupied bands: n_occ = {} (from N_e = {})\n",
        n_occ, n_electrons
    );

    let mut pcie = chemrust_scf::PcieAccount::default();
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
        &kernels,
    )
    .expect("VnlBatchData::precompute");

    // CASTEP V_eff from pot_fmt
    let v_eff_flat: Vec<f64> = fx.pot_fmt.iter().copied().collect();
    let min_veff = v_eff_flat.iter().copied().fold(f64::INFINITY, f64::min);
    let max_veff = v_eff_flat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "V_eff bounds: min = {:.4} Ha, max = {:.4} Ha\n",
        min_veff, max_veff
    );

    let ndeg = 8;

    // ==================================================================
    // One-time GPU setup (reused across all iterations)
    // ==================================================================

    // Upload V_eff to GPU
    let v_eff_arr = Array3::from_shape_vec((ngx, ngy, ngz), v_eff_flat).unwrap();
    let v_eff_wave = WaveGridArray::from_inner(v_eff_arr);
    let v_eff_fine = FineGridArray::from_inner(v_eff_wave.into_inner());
    let v_eff_inner = EffectivePotential::from_inner(v_eff_fine);
    let v_eff_gpu = Gpu::from_host_with(&v_eff_inner, &stream, &mut pcie)
        .expect("Failed to upload V_eff to GPU");

    // Upload PW-to-FFT index map
    let fft_idx: Vec<i32> = chemrust_scf::pw_coords_to_fft_indices(pw_coords, &wave_grid);
    let fft_idx_dev: CudaSlice<i32> = stream
        .clone_htod(&fft_idx)
        .map_err(|e| format!("Failed to upload FFT indices: {:?}", e))
        .unwrap();

    // Compile kernels (expensive, done once)
    let kernels = chemrust_scf::CudaKernelSet::new(&ctx).expect("Failed to compile CUDA kernels");

    // Upload initial psi from .check file
    let mut psi_gpu = upload_psi_to_gpu_column(&psi_input, n_bands, n_pw, &stream, &mut pcie);

    // Band group definitions
    let core: Vec<usize> = (0..1).collect();
    let cu3d: Vec<usize> = (1..15).collect();
    let val: Vec<usize> = (15..82).collect();
    let nfermi: Vec<usize> = (82..97).collect();
    let cond: Vec<usize> = (97..160).collect();
    let groups: [(&str, &[usize]); 5] = [
        ("core", &core),
        ("cu3d", &cu3d),
        ("val", &val),
        ("nFermi", &nfermi),
        ("cond", &cond),
    ];

    // ==================================================================
    // Outer loop
    // ==================================================================
    let n_outer_iters = 10;
    let ritz_tol = 1e-4; // Zhou 2014 Alg 5.1 step 10: Ritz convergence threshold (Ha)
    let mut residual_history: Vec<Vec<f64>> = Vec::with_capacity(n_outer_iters);
    let mut eigenvalue_history: Vec<Vec<f64>> = Vec::with_capacity(n_outer_iters);

    // Initialize eigenvalues from CASTEP for first iteration (continuation scenario)
    let mut eigenvalues_for_filter: Option<Vec<f64>> = Some(fx.bands_eigenvalues.clone());

    println!("Starting outer loop ({} iterations)...\n", n_outer_iters);
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12}",
        "iter", "core", "cu3d", "val", "nFermi", "cond", "band0_eig"
    );
    println!(
        "{:-<6}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<12}",
        "", "", "", "", "", "", ""
    );

    for iter in 0..n_outer_iters {
        // Step 1: Chebyshev filter with eigenvalue feedback from previous iteration
        let (psi_row_gpu, hpsi_row_gpu) = chemrust_scf::chebyshev_filter_iteration_gpu(
            &psi_gpu,
            &v_eff_gpu,
            &fft_idx_dev,
            &kernels,
            &wave_grid,
            pw_coords,
            &fx.bin.cell,
            &fx.pots,
            &vnl_data,
            min_veff,
            max_veff,
            ndeg,
            eigenvalues_for_filter.as_deref(), // Use previous iteration's eigenvalues
            FilterMode::SinvHKeepHEig,
            &blas,
            &solver,
            &stream,
            &ctx,
        )
        .expect("Chebyshev filter iteration failed");

        // Step 2: Rayleigh-Ritz
        let rr = chemrust_scf::rayleigh_ritz_with_matrices(
            &psi_row_gpu,
            &hpsi_row_gpu,
            &vnl_data,
            n_bands,
            n_pw,
            &kernels,
            &mut pcie,
            &solver,
            &blas,
            &stream,
            &ctx,
            None,
            None,
        )
        .expect("Rayleigh-Ritz failed");
        let psi_new_gpu = rr.0;
        let Cpu(eigenvalues): Cpu<Vec<f64>> = rr.1;
        let x = chemrust_scf::device::cuda_vec_to_complex(rr.5.0);

        // Zhou 2014 Alg 5.1 step 10: exit if Ritz values stop changing
        if let Some(prev_ritz) = eigenvalue_history.last() {
            let ritz_max_delta = eigenvalues
                .iter()
                .zip(prev_ritz.iter())
                .map(|(r_new, r_old)| (r_new - r_old).abs())
                .fold(0.0_f64, f64::max);
            println!(
                "  [Ritz check] max|ΔR| = {:.2e} Ha",
                ritz_max_delta
            );
            if ritz_max_delta < ritz_tol {
                println!(
                    "  ✓ Ritz values converged at iter={} (max|ΔR| = {:.2e} < {:.2e})",
                    iter + 1,
                    ritz_max_delta,
                    ritz_tol
                );
                // Store final eigenvalues and residual history before breaking
                eigenvalue_history.push(eigenvalues.clone());
                // Compute residuals one last time for the convergence record
                let (sinv_final, _l2_final) = chemrust_scf::compute_residual_norms_for_test(
                    &psi_new_gpu,
                    &hpsi_row_gpu,
                    &eigenvalues,
                    &x,
                    n_bands,
                    n_pw,
                    &vnl_data,
                    &blas,
                    &solver,
                    &stream,
                )
                .expect("Final residual computation failed");
                residual_history.push(sinv_final.clone());

                // Print final iteration summary
                let core_final = compute_mean_residual(&sinv_final, &core);
                let cu3d_final = compute_mean_residual(&sinv_final, &cu3d);
                let val_final = compute_mean_residual(&sinv_final, &val);
                let nfermi_final = compute_mean_residual(&sinv_final, &nfermi);
                let cond_final = compute_mean_residual(&sinv_final, &cond);
                println!(
                    "iter={:2} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>12.6}",
                    iter + 1,
                    core_final,
                    cu3d_final,
                    val_final,
                    nfermi_final,
                    cond_final,
                    eigenvalues[0],
                );
                psi_gpu = psi_new_gpu;
                break;
            }
        }

        // Step 3: Compute per-band S⁻¹-weighted residuals
        let (sinv_norms, _l2_norms) = chemrust_scf::compute_residual_norms_for_test(
            &psi_new_gpu,
            &hpsi_row_gpu,
            &eigenvalues,
            &x,
            n_bands,
            n_pw,
            &vnl_data,
            &blas,
            &solver,
            &stream,
        )
        .expect("Residual computation failed");

        // Track history
        residual_history.push(sinv_norms.clone());
        eigenvalue_history.push(eigenvalues.clone());

        // Store eigenvalues for next iteration's filter
        eigenvalues_for_filter = Some(eigenvalues.clone());

        // Print per-iteration summary
        let core_mean = compute_mean_residual(&sinv_norms, &core);
        let cu3d_mean = compute_mean_residual(&sinv_norms, &cu3d);
        let val_mean = compute_mean_residual(&sinv_norms, &val);
        let nfermi_mean = compute_mean_residual(&sinv_norms, &nfermi);
        let cond_mean = compute_mean_residual(&sinv_norms, &cond);
        println!(
            "iter={:2} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>12.6}",
            iter + 1,
            core_mean,
            cu3d_mean,
            val_mean,
            nfermi_mean,
            cond_mean,
            eigenvalues[0],
        );

        // Update psi for next iteration (move — drops old Gpu, frees its device memory)
        psi_gpu = psi_new_gpu;
    }

    // ==================================================================
    // Verify Success Criteria
    // ==================================================================

    println!("\n=== Verification ===\n");

    // SC-1: Residual Monotonicity
    verify_residual_monotonicity(&residual_history, &groups, &["core"]);

    // SC-2: Conduction Band Early Convergence
    println!("\n--- SC-2: Conduction Band Early Convergence ---");
    let n_converged_at_iter5 = count_converged_bands(&residual_history[4], &cond, 0.01);
    println!(
        "  conduction bands converged at iter-5: {}/63 (need ≥50)",
        n_converged_at_iter5
    );
    assert!(
        n_converged_at_iter5 >= 50,
        "SC-2 failed: only {}/63 conduction bands converged at iter-5 (expected ≥50)",
        n_converged_at_iter5
    );

    // SC-3: Occupied Band Residual Reduction
    println!("\n--- SC-3: Occupied Band Residual Reduction ---");
    let occupied_all: Vec<usize> = (0..82).collect();
    let occupied_mean_iter1 = compute_mean_residual(&residual_history[0], &occupied_all);
    let occupied_mean_iter10 = compute_mean_residual(&residual_history[9], &occupied_all);
    let reduction_factor = occupied_mean_iter1 / occupied_mean_iter10;
    println!("  occupied mean iter-1:  {:.3e} Ha", occupied_mean_iter1);
    println!("  occupied mean iter-10: {:.3e} Ha", occupied_mean_iter10);
    println!("  reduction: {:.2}× (need ≥5×)", reduction_factor);
    assert!(
        reduction_factor >= 5.0,
        "SC-3 failed: occupied band reduction = {:.2}× (expected ≥5×)",
        reduction_factor
    );

    // SC-4: Eigenvalue Stability
    println!("\n--- SC-4: Eigenvalue Stability ---");
    for iter in 4..n_outer_iters {
        let max_drift =
            compute_max_eigenvalue_drift(&eigenvalue_history[iter - 1], &eigenvalue_history[iter]);
        println!(
            "  iter {}-{}: max drift = {:.3e} Ha (need < 0.1 Ha)",
            iter,
            iter + 1,
            max_drift
        );
        assert!(
            max_drift < 0.1,
            "SC-4 failed: max eigenvalue drift at iter-{} = {:.3e} Ha (expected < 0.1 Ha)",
            iter + 1,
            max_drift
        );
    }

    // SC-5: No Cascade (band 0 eigenvalue stability)
    verify_band0_stability(&eigenvalue_history, -1.10, -1.01);

    println!("\n=== Diagnostic 3 Complete: All Success Criteria Passed ===\n");
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
    assert!(
        (kappa - 1.0).abs() < 1e-10,
        "Identity matrix should have κ₂ = 1, got {}",
        kappa
    );
}

#[test]
fn test_condition_number_ill_conditioned() {
    // Test that a matrix with widely separated singular values has large κ₂
    let mut m = Array2::zeros((3, 3));
    m[[0, 0]] = Complex64::new(1000.0, 0.0);
    m[[1, 1]] = Complex64::new(1.0, 0.0);
    m[[2, 2]] = Complex64::new(0.001, 0.0);

    let kappa = condition_number_svd(&m);
    assert!(
        kappa > 1e5,
        "Ill-conditioned matrix should have large κ₂, got {}",
        kappa
    );
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

    assert!(
        (max - 0.1).abs() < 1e-10,
        "max off-diagonal should be 0.1, got {}",
        max
    );
    assert!(
        median > 0.0 && median < 0.1,
        "median should be between 0 and 0.1, got {}",
        median
    );
    assert!(
        mean > 0.0 && mean < 0.1,
        "mean should be between 0 and 0.1, got {}",
        mean
    );
}

/// Diagnostic: compute GPU H·psi per-band expectations using the REFERENCE
/// V_eff (pot_fmt) and compare against CASTEP eigenvalues.
///
/// The key difference from test_5: test_5 uses a RECONSTRUCTED V_eff from
/// VEffBuilder (via build_v_eff()). This test uses the REFERENCE pot_fmt V_eff.
/// If the residuals are smaller here, V_eff reconstruction is the problem.
/// If residuals are the same size, the error is in V_NL.
///
/// Also decomposes into kinetic (CPU-computed, exact) and V_NL (via subtraction)
/// to show which component contributes the gap.
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_h_psi_component_breakdown() {
    use chemrust_scf::{
        BlasHandle, CudaKernelSet, EffectivePotential, FineGridArray,
        Gpu, SolverHandle, VnlBatchData, WaveGridArray, apply_h_components_for_test,
        apply_s_for_test,
    };
    use cudarc::driver::{CudaContext, CudaSlice};
    use std::sync::Arc;

    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    println!("\n=== H·ψ component breakdown: REFERENCE V_eff vs CASTEP ε ===\n");

    let fx = fixtures::cu111_co::fixture();
    let wfc = fx.check.wavefunction.as_ref()
        .expect(".check must have wavefunction");
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;
    let psi_input: Vec<Complex64> = kpt_block.bands.concat();

    // GPU setup
    let ctx = Arc::new(CudaContext::new(0).expect("CudaContext"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("BlasHandle");
    let solver = SolverHandle::new(stream.clone()).expect("SolverHandle");
    let kernels = CudaKernelSet::new(&ctx).expect("CudaKernelSet");
    let wave_grid = chemrust_hamiltonian_core::GVectorGrid::new(
        wfc.grid[0], wfc.grid[1], wfc.grid[2],
        fx.bin.cell.recip_lattice,
    );
    let mut pcie = chemrust_scf::PcieAccount::default();
    let pw_coords = &kpt_block.pw_grid_coord;
    let k_point = chemrust_scf::KPoint { coords: kpt_block.coords, weight: 1.0 };
    let cell = &fx.bin.cell;

    // Occupations from CASTEP eigenvalues
    let n_electrons: f64 = cell.species_iter()
        .map(|info| fx.pots.get(info.symbol).and_then(|p| p.ionic_charge()).unwrap_or(0.0) * info.num_ions as f64)
        .sum();
    let (occupations, _) = chemrust_scf::density::compute_occupations(
        &fx.bands_eigenvalues,
        &chemrust_scf::SmearingParams {
            width: 0.1 * chemrust_scf::EV_TO_HARTREE,
            electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
            scheme: chemrust_scf::SmearingScheme::Gaussian,
            spin_fix: 10,
        }, n_electrons, 1.0,
    ).expect("compute_occupations");

    // Build VnlBatchData using REFERENCE V_eff (pot_fmt), not reconstructed
    let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
        chemrust_hamiltonian_core::fft::RealGrid::from_inner(fx.pot_fmt.clone()));
    let vnl_data = VnlBatchData::precompute(
        pw_coords, &fx.pots, cell, &wave_grid, &k_point,
        &psi_input, n_bands, n_pw, Some(&occupations.0), Some(&v_eff_for_d),
        &stream, &mut pcie, &blas, &kernels,
    ).expect("VnlBatchData");

    // Upload reference V_eff to GPU
    let v_eff_arr = ndarray::Array3::from_shape_vec(
        (wave_grid.grid()[2], wave_grid.grid()[1], wave_grid.grid()[0]),
        fx.pot_fmt.iter().copied().collect(),
    ).unwrap();
    let v_eff_wave = WaveGridArray::from_inner(v_eff_arr);
    let v_eff_fine = FineGridArray::from_inner(v_eff_wave.into_inner());
    let v_eff_gpu = Gpu::from_host_with(
        &chemrust_scf::EffectivePotential::from_inner(v_eff_fine),
        &stream, &mut pcie,
    ).expect("V_eff upload");

    // Upload psi to GPU using the existing helper
    let psi_gpu = upload_psi_to_gpu_column(&psi_input, n_bands, n_pw, &stream, &mut pcie);

    // FFT index map
    let fft_idx: Vec<i32> = chemrust_scf::pw_coords_to_fft_indices(pw_coords, &wave_grid);
    let fft_idx_dev = stream.clone_htod(&fft_idx).expect("FFT indices");

    // GPU: get H·psi components
    let h_components = apply_h_components_for_test(
        &psi_gpu, &v_eff_gpu, &wave_grid, pw_coords, &vnl_data,
        &fft_idx_dev, &kernels, &blas, &stream,
    ).expect("apply_h_components_for_test");

    // Download psi + spsi for per-band dot products
    let psi_raw: Vec<Complex64> = stream.clone_dtoh(psi_gpu.as_device_slice())
        .expect("psi D2H")
        .into_iter()
        .map(|c| Complex64::new(c.x, c.y))
        .collect();
    let spsi = apply_s_for_test(&psi_input, n_bands, n_pw, &vnl_data, &blas, &stream)
        .expect("apply_s_for_test");

    // CPU kinetic expectation (purely geometric, should match GPU exactly)
    let r = cell.recip_lattice.as_array();
    let kinetic_g2: Vec<f64> = pw_coords.iter()
        .map(|&[h, k, l]| {
            let gx = h as f64 * r[0][0] + k as f64 * r[1][0] + l as f64 * r[2][0];
            let gy = h as f64 * r[0][1] + k as f64 * r[1][1] + l as f64 * r[2][1];
            let gz = h as f64 * r[0][2] + k as f64 * r[1][2] + l as f64 * r[2][2];
            0.5 * (gx * gx + gy * gy + gz * gz)
        })
        .collect();
    let mut t_cpu = vec![0.0_f64; n_bands];
    for b in 0..n_bands {
        let off = b * n_pw;
        t_cpu[b] = psi_input[off..off + n_pw].iter()
            .zip(kinetic_g2.iter())
            .map(|(&c, &ke)| c.norm_sqr() * ke)
            .sum();
    }

    // Per-band diagnostics
    struct BandDiag { b: usize, t_gpu: f64, tv_gpu: f64, h_gpu: f64,
        s_norm: f64, rq_gpu: f64, t_cpu: f64, eps: f64, delta: f64, label: String }
    let mut diags: Vec<BandDiag> = Vec::with_capacity(n_bands);

    for b in 0..n_bands {
        let off = b * n_pw;
        let psi_b = &psi_raw[off..off + n_pw];
        let h_b = &h_components.hpsi_full[off..off + n_pw];
        let tv_b = &h_components.hpsi_tv[off..off + n_pw];
        let t_b = &h_components.hpsi_t[off..off + n_pw];
        let s_b = &spsi[off..off + n_pw];

        let h_dot: f64 = psi_b.iter().zip(h_b).map(|(p, h)| (p.conj() * h).re).sum();
        let tv_dot: f64 = psi_b.iter().zip(tv_b).map(|(p, h)| (p.conj() * h).re).sum();
        let t_dot: f64 = psi_b.iter().zip(t_b).map(|(p, h)| (p.conj() * h).re).sum();
        let s_dot: f64 = psi_b.iter().zip(s_b).map(|(p, s)| (p.conj() * s).re).sum();

        let rq = if s_dot.abs() > 1e-30 { h_dot / s_dot } else { 0.0 };
        let eps = fx.bands_eigenvalues[b];
        let delta = rq - eps;
        let label = if b < 1 { "core" } else if b < 15 { "cu3d" }
            else if b < 82 { "val" } else if b < 93 { "nFermi" } else { "cond" };

        diags.push(BandDiag { b, t_gpu: t_dot, tv_gpu: tv_dot, h_gpu: h_dot,
            s_norm: s_dot, rq_gpu: rq, t_cpu: t_cpu[b], eps, delta, label: label.into() });
    }

    // ---- Report ----
    println!("Group       bands  max|Δ|_refV  mean|Δ|_refV  worst_b");
    println!("  (refV_eff means using REFERENCE pot_fmt V_eff)");
    for (gname, gb) in [("core", 0..1), ("cu3d", 1..15), ("val", 15..82),
                         ("nFermi", 82..93), ("cond", 93..160)] {
        let mut max_ad = 0.0_f64;
        let mut sum_ad = 0.0_f64;
        let mut cnt = 0usize;
        let mut wb = 0usize;
        for d in diags.iter().filter(|d| gb.contains(&d.b)) {
            let ad = d.delta.abs();
            if ad > max_ad { max_ad = ad; wb = d.b; }
            sum_ad += ad;
            cnt += 1;
        }
        let mean_ad = sum_ad / cnt.max(1) as f64;
        println!("{:<10} {:>3}..{:<3}  {:>10.3e}  {:>10.3e}  {:>7}",
            gname, gb.start, gb.end-1, max_ad, mean_ad, wb);
    }
    println!();

    // Top 10 by |Δ|
    diags.sort_by(|a, b| b.delta.abs().partial_cmp(&a.delta.abs()).unwrap());
    println!("Top 10 bands by |Δ| (ref V_eff RQ vs CASTEP ε):");
    println!("  band |  Δ (Ha)    | RQ (Ha)   | ε (Ha)    | label  | ⟨S⟩    | K gap");
    for d in diags.iter().take(10) {
        let k_gap = (d.t_gpu - d.t_cpu).abs();
        println!("  {:>4} | {:>+9.2e} | {:>9.4} | {:>9.4} | {:<6} | {:>7.4} | {:>6.1e}",
            d.b, d.delta, d.rq_gpu, d.eps, d.label, d.s_norm, k_gap);
    }
    println!();

    // Band 0 detailed breakdown
    let b0 = diags.iter().find(|d| d.b == 0).unwrap();
    println!("Band 0 detailed breakdown:");
    println!("  T_gpu       = {:.10e} Ha", b0.t_gpu);
    println!("  T_cpu       = {:.10e} Ha  |Δ| = {:.2e}", b0.t_cpu, (b0.t_gpu - b0.t_cpu).abs());
    println!("  ⟨T+V_eff⟩   = {:.10e} Ha", b0.tv_gpu);
    println!("  ⟨V_eff⟩     = {:.10e} Ha", b0.tv_gpu - b0.t_gpu);
    println!("  ⟨V_nl⟩_gpu  = {:.10e} Ha  (h_full − h_tv)", b0.h_gpu - b0.tv_gpu);
    println!("  ⟨S⟩         = {:.10e}", b0.s_norm);
    println!("  RQ_gpu      = {:.10e} Ha", b0.rq_gpu);
    println!("  ε_CASTEP    = {:.10e} Ha", b0.eps);
    println!("  Δ           = {:.10e} Ha", b0.delta);
    println!();

    // Comparison with test_5 results (reconstructed V_eff):
    // test_5 band 0: |Δ| = 0.013097 Ha  (reconstructed V_eff)
    // This test band 0 Δ = ?
    // If |Δ|_test < |Δ|_test5 → V_eff reconstruction adds error
    // If |Δ|_test ≈ |Δ|_test5 → V_eff not the problem, V_NL is
    // If |Δ|_test > |Δ|_test5 → somehow reconstructed V_eff is better (unlikely)
    // Download c_proj from GPU and compare against CPU beta_phi for first Cu ion
    let first_cu_ion = cell.ion_species.iter().position(|&s| {
        cell.species_symbols.get(s).map_or(false, |sym| *sym == "Cu")
    }).unwrap_or(0);

    if let Some(entry) = vnl_data.entries.get(first_cu_ion) {
        let ne = entry.n_expanded as usize;
        let n_pw_i32 = n_pw as i32;
        let n_bands_i32 = n_bands as i32;
        let one = chemrust_scf::device::CudaComplex { x: 1.0, y: 0.0 };
        let zero = chemrust_scf::device::CudaComplex { x: 0.0, y: 0.0 };

        // Compute c_proj on GPU: beta_g^H · psi
        let mut c_proj_dev: CudaSlice<chemrust_scf::device::CudaComplex> = stream
            .alloc_zeros(ne * n_bands).expect("alloc c_proj");
        unsafe {
            blas.gemm_c64(
                chemrust_scf::device::blas::ZgemmConfig {
                    transa: chemrust_scf::device::blas::op::C,
                    transb: chemrust_scf::device::blas::op::N,
                    m: ne as i32,
                    n: n_bands_i32,
                    k: n_pw_i32,
                    alpha: one,
                    lda: n_pw_i32,
                    ldb: n_pw_i32,
                    beta: zero,
                    ldc: ne as i32,
                },
                &entry.beta_g,
                psi_gpu.as_device_slice(),
                &mut c_proj_dev,
            ).expect("gemm c_proj");
        }
        let c_proj_host: Vec<chemrust_scf::device::CudaComplex> =
            stream.clone_dtoh(&c_proj_dev).expect("c_proj D2H");

        // Compute beta_phi on CPU using the Gamma-point-aware formula
        use chemrust_hamiltonian_core::augment::beta_phi::compute_beta_phi;
        use chemrust_hamiltonian_core::types::KptWaveBlock;
        let cpu_block = KptWaveBlock {
            coords: kpt_block.coords,
            nplw: n_pw,
            pw_grid_coord: pw_coords.clone(),
            bands: psi_input.chunks(n_pw).map(|chunk| chunk.to_vec()).collect(),
        };
        let sym = &cell.species_symbols[cell.ion_species[first_cu_ion]];
        let pot = fx.pots.get(sym).unwrap();
        let k_cart = [0.0; 3];
        if let chemrust_hamiltonian_core::Pseudopotential::Usp(aug) = pot {
            let beta_phi_cpu = compute_beta_phi(
                &cpu_block, aug, cell, first_cu_ion, &wave_grid,
                pot.gmax(), k_cart,
            ).expect("compute_beta_phi");

            // Compare GPU c_proj vs CPU beta_phi for band 0
            let gpu_b0: Complex64 = Complex64::new(c_proj_host[0].x, c_proj_host[0].y);
            let cpu_b0 = beta_phi_cpu[[0, 0]];
            let ratio = if cpu_b0.norm_sqr() > 1e-30 { gpu_b0 / cpu_b0 } else { Complex64::ZERO };
            println!("\nGamma convention check (Cu ion {}, band 0, proj 0):", first_cu_ion);
            println!("  GPU c_proj (β^H·ψ, complex gemm): ({:+.6e}, {:+.6e})", gpu_b0.re, gpu_b0.im);
            println!("  CPU beta_phi (Gamma formula):     ({:+.6e}, {:+.6e})", cpu_b0.re, cpu_b0.im);
            println!("  Ratio GPU/CPU: {:+.6e}", ratio);

            let gpu_b0_1 = Complex64::new(c_proj_host[1].x, c_proj_host[1].y);
            let cpu_b0_1 = beta_phi_cpu[[1, 0]];
            let ratio_1 = if cpu_b0_1.norm_sqr() > 1e-30 { gpu_b0_1 / cpu_b0_1 } else { Complex64::ZERO };
            println!("  GPU c_proj (proj 1): ({:+.6e}, {:+.6e})", gpu_b0_1.re, gpu_b0_1.im);
            println!("  CPU beta_phi (proj 1): ({:+.6e}, {:+.6e})", cpu_b0_1.re, cpu_b0_1.im);
            println!("  Ratio GPU/CPU: {:+.6e}", ratio_1);

            if (ratio - Complex64::ONE).norm() > 1e-10 {
                println!("  ⚠ GPU and CPU c_proj DIFFER — Gamma-point convention mismatch!");
            } else {
                println!("  ✓ GPU and CPU c_proj match — Gamma-point convention is correct.");
            }
        }
    }

    println!("\nComparison with test_5 (reconstructed V_eff):");
    println!("  test_5 band 0: |Δ| = 0.013097 Ha (using RECONSTRUCTED V_eff)");
    println!("  this test     : |Δ| = {:.6e} Ha (using REFERENCE pot_fmt V_eff)", b0.delta.abs());
    let ref_better = b0.delta.abs() < 0.013;
    println!("  -> Reference V_eff gives {} residuals than reconstructed V_eff",
        if ref_better { "SMALLER" } else { "SIMILAR or LARGER" });
    if !ref_better {
        println!("  -> V_eff reconstruction is NOT the dominant error source.");
        println!("  -> V_NL (D screening or assembly formula) is the likely culprit.");
    }
}
