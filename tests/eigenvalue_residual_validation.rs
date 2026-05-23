//! Eigenvalue residual validation test.
//!
//! For each eigenvalue λ_i and corresponding eigenvector ψ_i, the residual is:
//!   r_i = ||H ψ_i - λ_i ψ_i|| / ||ψ_i||
//!
//! For a correct diagonalization, residuals should be small (< 1e-4 Ha typically).
//! Large residuals indicate:
//! - Incorrect eigenvalues from Lanczos/Rayleigh-Ritz
//! - Non-Hermitian H (produces garbage eigenvalues)
//! - Numerical issues in the eigensolver
//!
//! This test isolates eigenvalue quality independent of SCF convergence.
//! It validates that when we apply H to the CASTEP reference wavefunctions
//! and compute the Rayleigh quotient, the residuals are small.

mod fixtures;

use num_complex::Complex64;

fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

/// Compute the L2 norm of a complex vector.
fn l2_norm(v: &[Complex64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Compute the residual for a single eigenvalue-eigenvector pair.
///
/// residual = ||H ψ - λ ψ|| / ||ψ||
fn compute_residual(
    h_psi: &[Complex64],
    eigenvalue: f64,
    psi: &[Complex64],
) -> f64 {
    let n_pw = psi.len();
    assert_eq!(h_psi.len(), n_pw, "H ψ and ψ must have same length");

    let psi_norm = l2_norm(psi);
    if psi_norm < 1e-15 {
        return f64::INFINITY; // degenerate or zero eigenvector
    }

    // Compute H ψ - λ ψ
    let mut residual_vec = vec![Complex64::ZERO; n_pw];
    for i in 0..n_pw {
        residual_vec[i] = h_psi[i] - eigenvalue * psi[i];
    }

    let residual_norm = l2_norm(&residual_vec);
    residual_norm / psi_norm
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn eigenvalue_residuals_castep_psi_with_computed_veff() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixtures::cu111_co::fixture();
    let scf_state = fixtures::cu111_co::build_scf_state(&fx);

    // Build V_eff with energy components
    let scf_state = scf_state
        .build_v_eff_with_energy()
        .expect("Failed to build V_eff");

    // Extract reference eigenvalues from fixture
    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;

    // Compute occupations from reference eigenvalues
    let n_electrons = 186.0;
    let width = 0.01; // Fermi-Dirac smearing width
    let reference_eigs = &fx.bands_eigenvalues;
    let mut lo = -100.0;
    let mut hi = 100.0;
    for _ in 0..80 {
        let mu = 0.5 * (lo + hi);
        let n: f64 = reference_eigs
            .iter()
            .map(|&e| libm::erfc((e - mu) / width))
            .sum();
        if n > n_electrons {
            hi = mu;
        } else {
            lo = mu;
        }
    }
    let mu = 0.5 * (lo + hi);
    let occupations: Vec<f64> = reference_eigs
        .iter()
        .map(|&e| 0.5 * libm::erfc((e - mu) / width))
        .collect();

    // Apply H to the CASTEP reference wavefunctions (on VEffBuilt state)
    let h_components = scf_state
        .apply_h_components_for_test(Some(&occupations))
        .expect("Failed to compute H components");

    assert_eq!(h_components.n_bands, n_bands);
    assert_eq!(h_components.n_pw, n_pw);

    // Extract CASTEP wavefunctions (from fixture)
    let psi_flat: Vec<Complex64> = kpt_block.bands.concat();
    assert_eq!(psi_flat.len(), n_bands * n_pw);

    // Compute residuals for each band using CASTEP eigenvalues
    let mut max_residual = 0.0;
    let mut max_residual_band = 0;
    let mut residuals_above_threshold = Vec::new();

    for band in 0..n_bands {
        let psi_start = band * n_pw;
        let psi_end = psi_start + n_pw;
        let psi = &psi_flat[psi_start..psi_end];

        let h_psi_start = band * n_pw;
        let h_psi_end = h_psi_start + n_pw;
        let h_psi = &h_components.hpsi_full[h_psi_start..h_psi_end];

        let eigenvalue = reference_eigs[band];
        let residual = compute_residual(h_psi, eigenvalue, psi);

        if residual > max_residual {
            max_residual = residual;
            max_residual_band = band;
        }

        // Track bands with residuals above threshold
        if residual > 1e-4 {
            residuals_above_threshold.push((band, residual));
        }
    }

    // Report findings
    eprintln!(
        "[Eigenvalue Residuals] max_residual = {:.3e} Ha (band {})",
        max_residual, max_residual_band
    );
    eprintln!(
        "[Eigenvalue Residuals] {} bands with residual > 1e-4 Ha",
        residuals_above_threshold.len()
    );
    if !residuals_above_threshold.is_empty() {
        for (band, residual) in residuals_above_threshold.iter().take(10) {
            eprintln!("  band {}: {:.3e} Ha", band, residual);
        }
    }

    // Assertion: max residual should be small
    // With CASTEP reference ψ and computed V_eff, residuals should be < 1e-3 Ha
    assert!(
        max_residual < 1e-3,
        "[Eigenvalue Residuals] max residual = {:.3e} Ha (band {}) exceeds 1e-3 Ha threshold. \
         This indicates H|ψ⟩ computation or eigenvalue quality issue.",
        max_residual, max_residual_band
    );
}
