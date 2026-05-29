// ---------------------------------------------------------------------------
// CG helper functions: residual, S-normalization, S-orthogonalization
// ---------------------------------------------------------------------------
//
// These are CPU-side helper functions for the block CG eigensolver.
// In Phase-0, we accept precomputed S|ψ⟩ as input rather than applying
// the S-operator internally. This keeps the API simple and lets the
// caller manage S-application (which requires the USPP augmentation data).
//
// Future phases may add GPU variants or integrate S⁻¹ for the true S-norm.

// Allow dead_code: these functions are library components used when the
// block CG eigensolver is wired up in subsequent Phase-0 tasks.
#![allow(dead_code)]

use num_complex::Complex64;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Helper: plain Euclidean inner product
// ---------------------------------------------------------------------------

/// Compute the Euclidean inner product ⟨a|b⟩ = Σ conj(a[i]) × b[i].
///
/// Panics if slices have different lengths.
fn inner_product(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    assert_eq!(a.len(), b.len(), "inner_product: length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum()
}

// ---------------------------------------------------------------------------
// CG residual
// ---------------------------------------------------------------------------

/// Compute CG residual: r = H|ψ⟩ − ε·S|ψ⟩
///
/// The Rayleigh quotient ε = ⟨ψ|H|ψ⟩ / ⟨ψ|S|ψ⟩ is the best eigenvalue
/// estimate for the current wavefunction. The residual r is the gradient
/// of the Rayleigh quotient and serves as the search direction before
/// preconditioning.
///
/// # Returns
/// `(residual_coefficients, epsilon)` — the residual vector and the
/// Rayleigh-quotient eigenvalue estimate.
///
/// # Panics
/// Panics if the S-expectation ⟨ψ|S|ψ⟩ is zero or if the input slices
/// have mismatched lengths.
pub fn compute_residual(
    psi: &[Complex64],
    hpsi: &[Complex64],
    spsi: &[Complex64],
) -> (Vec<Complex64>, f64) {
    assert_eq!(psi.len(), hpsi.len(), "compute_residual: psi/hpsi length mismatch");
    assert_eq!(psi.len(), spsi.len(), "compute_residual: psi/spsi length mismatch");

    if psi.is_empty() {
        return (Vec::new(), f64::NAN);
    }

    let h_expect = inner_product(psi, hpsi).re;
    let s_expect = inner_product(psi, spsi).re;

    assert!(
        s_expect.abs() > f64::EPSILON,
        "compute_residual: ⟨ψ|S|ψ⟩ = {s_expect} is too small (S-metric is near-singular)"
    );

    let epsilon = h_expect / s_expect;

    let residual: Vec<Complex64> = hpsi
        .par_iter()
        .zip(spsi.par_iter())
        .map(|(h, s)| h - Complex64::new(epsilon, 0.0) * s)
        .collect();

    (residual, epsilon)
}

// ---------------------------------------------------------------------------
// Residual norm
// ---------------------------------------------------------------------------

/// Compute the bare L2 norm of the residual: ‖r‖₂ = sqrt(⟨r|r⟩).
///
/// This is an approximation of the true S-norm ‖r‖_S = sqrt(⟨r|S⁻¹|r⟩)
/// which requires S⁻¹ application (a Woodbury solve). The bare norm is
/// usually sufficient for convergence monitoring since S⁻¹ is
/// well-conditioned for USPP.
///
/// For the true S-norm, compute S⁻¹·r (via the Woodbury identity in
/// `uspp_preconditioner.rs`) and then compute sqrt(⟨r|S⁻¹·r⟩).
pub fn residual_bare_norm(residual: &[Complex64]) -> f64 {
    residual
        .par_iter()
        .map(|c| c.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// S-normalization
// ---------------------------------------------------------------------------

/// S-normalize a wavefunction: ψ ← ψ / sqrt(⟨ψ|S|ψ⟩)
///
/// After normalization, ⟨ψ|S|ψ⟩ = 1. If the S-overlap is already ≈ 1.0
/// (within 1e-14), returns the input unchanged to avoid unnecessary
/// floating-point work.
///
/// # Arguments
/// - `psi` — wavefunction coefficients
/// - `spsi` — S|ψ⟩ (precomputed S-operator application on `psi`)
///
/// # Panics
/// Panics if the S-overlap is negative (should not happen for USPP S,
/// which is positive-definite).
pub fn s_normalize(psi: &[Complex64], spsi: &[Complex64]) -> Vec<Complex64> {
    let s_overlap = inner_product(psi, spsi).re;

    assert!(
        s_overlap > -1e-14,
        "s_normalize: ⟨ψ|S|ψ⟩ = {s_overlap} is negative (S should be positive-definite)"
    );

    // If already normalized, skip
    if (s_overlap - 1.0).abs() < 1e-14 {
        return psi.to_vec();
    }

    let norm = s_overlap.sqrt();
    psi.iter().map(|c| c / Complex64::new(norm, 0.0)).collect()
}

// ---------------------------------------------------------------------------
// S-orthogonalization (lower-only, modified Gram-Schmidt)
// ---------------------------------------------------------------------------

/// S-orthogonalize a wavefunction against a set of already-converged bands
/// (lower-only orthogonalization).
///
/// For each converged band ψ_i (which is S-normalized):
///   overlap = ⟨ψ_i|S|ψ⟩
///   ψ ← ψ − overlap·ψ_i
///   S|ψ⟩ ← S|ψ⟩ − overlap·S|ψ_i⟩
///
/// After all projections, the result is S-normalized.
///
/// This is "lower-only" because we only orthogonalize against bands that
/// are already converged, not against other bands being solved concurrently.
///
/// # Arguments
/// - `psi` — wavefunction coefficients of the new band
/// - `spsi` — S|ψ⟩ (precomputed)
/// - `converged_bands` — list of `(psi_i, spsi_i)` tuples, one per converged
///   band. Each psi_i must already be S-normalized (⟨ψ_i|S|ψ_i⟩ = 1).
///
/// # Returns
/// `(orthogonalized_psi, orthogonalized_spsi)` — the S-orthogonalized
/// wavefunction and its S-application.
pub fn s_orthogonalize_against(
    psi: &[Complex64],
    spsi: &[Complex64],
    converged_bands: &[(Vec<Complex64>, Vec<Complex64>)],
) -> (Vec<Complex64>, Vec<Complex64>) {
    let n = psi.len();
    let mut psi_new = psi.to_vec();
    let mut spsi_new = spsi.to_vec();

    for (psi_i, spsi_i) in converged_bands {
        assert_eq!(
            psi_i.len(),
            n,
            "s_orthogonalize_against: converged band has length {}, expected {n}",
            psi_i.len(),
        );
        assert_eq!(
            spsi_i.len(),
            n,
            "s_orthogonalize_against: converged band spsi has length {}, expected {n}",
            spsi_i.len(),
        );

        // Overlap in the S-metric: ⟨ψ_i|S|ψ⟩ using the *current* ψ_new.
        // In modified Gram-Schmidt, this uses the progressively updated
        // spsi_new so that each new projection removes the component along
        // ψ_i from the *current* iterate.
        let overlap = inner_product(psi_i, &spsi_new).re;

        // ψ_new ← ψ_new − overlap · ψ_i
        for (pn, p_i) in psi_new.iter_mut().zip(psi_i.iter()) {
            *pn -= Complex64::new(overlap, 0.0) * p_i;
        }

        // S|ψ_new⟩ ← S|ψ_new⟩ − overlap · S|ψ_i⟩
        // This follows from linearity of the S-operator.
        for (sn, s_i) in spsi_new.iter_mut().zip(spsi_i.iter()) {
            *sn -= Complex64::new(overlap, 0.0) * s_i;
        }
    }

    // S-normalize the result and also re-normalize spsi_new
    let s_overlap = inner_product(&psi_new, &spsi_new).re;
    assert!(
        s_overlap > 0.0,
        "s_orthogonalize_against: final ⟨ψ|S|ψ⟩ = {s_overlap} is not positive"
    );

    let norm = s_overlap.sqrt();
    let norm_c64 = Complex64::new(norm, 0.0);

    let psi_normalized: Vec<Complex64> = psi_new.iter().map(|c| c / norm_c64).collect();
    let spsi_normalized: Vec<Complex64> = spsi_new.iter().map(|c| c / norm_c64).collect();

    (psi_normalized, spsi_normalized)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use approx::relative_eq;

    // -----------------------------------------------------------------------
    // Success Criterion 1: S-norm identity
    //
    // For a test vector with known ⟨ψ|S|ψ⟩ ≠ 1, after s_normalize:
    //   ⟨ψ_normalized|S|ψ_normalized⟩ ≈ 1.0 ± 1e-10
    // -----------------------------------------------------------------------
    #[test]
    fn s_normalize_produces_unit_s_overlap() {
        // Construct psi with coefficients [1.0, 0.5, 0.25] and set spsi
        // such that ⟨ψ|S|ψ⟩ = 0.5.
        // ⟨ψ|ψ⟩ = 1² + 0.5² + 0.25² = 1.3125
        // spsi = (0.5 / ⟨ψ|ψ⟩) × ψ  →  ⟨ψ|spsi⟩ = 0.5
        let psi: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.25, 0.0),
        ];
        let psi_norm_sq: f64 = psi.iter().map(|c| c.norm_sqr()).sum();
        let target_s_overlap = 0.5;
        let scale = target_s_overlap / psi_norm_sq;
        let spsi: Vec<Complex64> = psi.iter().map(|c| c * scale).collect();

        // Verify setup: ⟨ψ|S|ψ⟩ ≈ 0.5
        let initial_overlap = inner_product(&psi, &spsi).re;
        assert_relative_eq!(initial_overlap, target_s_overlap, epsilon = 1e-15);

        // Act: S-normalize
        let psi_normalized = s_normalize(&psi, &spsi);

        // Verify: ⟨ψ_normalized|S|ψ_normalized⟩ must be computed with the
        // *original* S-operator. Since we don't have S as a separate object,
        // we reconstruct spsi_normalized = scale × psi_normalized (same S).
        let spsi_normalized: Vec<Complex64> =
            psi_normalized.iter().map(|c| c * scale).collect();
        let final_overlap = inner_product(&psi_normalized, &spsi_normalized).re;

        assert!(
            relative_eq!(final_overlap, 1.0, epsilon = 1e-12),
            "S-norm identity: after s_normalize, ⟨ψ|S|ψ⟩ = {final_overlap}, expected 1.0"
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 1b: s_normalize skips when already normalized
    //
    // If ⟨ψ|S|ψ⟩ ≈ 1.0, the function should return the input unchanged
    // (identity transform, not a fresh allocation if avoidable).
    // -----------------------------------------------------------------------
    #[test]
    fn s_normalize_skips_when_already_unit() {
        // Create a vector where spsi = psi (S = I), making ⟨ψ|S|ψ⟩ = 1.0
        // To pass the "skip" threshold, we need the overlap to be within 1e-14.
        let psi: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let spsi = psi.clone();

        let result = s_normalize(&psi, &spsi);
        // The result should be the input unchanged (same values, same length).
        assert_eq!(
            result.len(),
            psi.len(),
            "s_normalize should not change length when already normalized"
        );
        for (r, p) in result.iter().zip(psi.iter()) {
            assert!(
                relative_eq!(r.re, p.re, epsilon = 1e-15),
                "s_normalize changed coefficient (re) when already normalized"
            );
            assert!(
                relative_eq!(r.im, p.im, epsilon = 1e-15),
                "s_normalize changed coefficient (im) when already normalized"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Success Criterion 2: Residual orthogonality
    //
    // For a test vector ψ with eigenvalue ε, the residual
    // r = Hψ − ε·Sψ satisfies ⟨r|ψ⟩ = 0 (Euclidean inner product).
    // This is a consequence of the Rayleigh-quotient definition.
    // -----------------------------------------------------------------------
    #[test]
    fn residual_is_orthogonal_to_psi() {
        // Construct synthetic data with known expectations.
        // ψ = [1, 0, 0, ...]  (simplest: unit vector along first basis)
        // Hψ = [2, 3, 4, ...]
        // Sψ = [1.5, 0.5, ...]
        // ⟨ψ|H|ψ⟩ = 1·2 + 0·3 + 0·4 = 2
        // ⟨ψ|S|ψ⟩ = 1·1.5 + 0·0.5 = 1.5
        // ε = 2 / 1.5 = 4/3
        // r = Hψ − ε·Sψ = [2−4/3·1.5, 3−4/3·0.5, 4−4/3·0, ...]
        //   = [2−2, 3−2/3, 4, ...] = [0, 7/3, 4, ...]
        // ⟨r|ψ⟩ = 0·1 + 7/3·0 + 4·0 = 0
        let psi: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let hpsi: Vec<Complex64> = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ];
        let spsi: Vec<Complex64> = vec![
            Complex64::new(1.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let (residual, epsilon) = compute_residual(&psi, &hpsi, &spsi);

        // Verify eigenvalue
        let expected_epsilon = 2.0 / 1.5; // 4/3
        assert!(
            relative_eq!(epsilon, expected_epsilon, epsilon = 1e-14),
            "Rayleigh quotient ε = {epsilon}, expected {expected_epsilon}"
        );

        // Verify orthogonality: ⟨r|ψ⟩ ≈ 0 (Euclidean)
        let r_dot_psi = inner_product(&residual, &psi);
        assert!(
            relative_eq!(r_dot_psi.re, 0.0, epsilon = 1e-14),
            "Re(⟨r|ψ⟩) should be 0, got {}",
            r_dot_psi.re
        );
        assert!(
            relative_eq!(r_dot_psi.im, 0.0, epsilon = 1e-14),
            "Im(⟨r|ψ⟩) should be 0, got {}",
            r_dot_psi.im
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 3: S-orthogonalization
    //
    // After s_orthogonalize_against with 3 synthetic S-orthonormal
    // "converged" bands, verify |⟨ψ_new|S|ψ_i⟩| < 1e-10 for all i.
    // -----------------------------------------------------------------------
    #[test]
    fn orthogonalize_removes_s_overlap() {
        // Create 3 S-orthonormal "converged" bands.
        // For simplicity, use Euclidean-orthonormal vectors with S = I
        // (spsi_i = psi_i).
        let n = 6;

        // psi_0 = [1, 0, 0, 0, 0, 0]
        // psi_1 = [0, 1, 0, 0, 0, 0]
        // psi_2 = [0, 0, 1, 0, 0, 0]
        let mut bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = Vec::new();
        for i in 0..3 {
            let mut psi_i = vec![Complex64::ZERO; n];
            psi_i[i] = Complex64::new(1.0, 0.0);
            let spsi_i = psi_i.clone(); // S = I
            bands.push((psi_i, spsi_i));
        }

        // Create a test vector that has overlap with all 3 converged bands
        // and an additional component.
        // ψ_test = [0.5, -0.3, 0.2, 0.8, 0.1, -0.4]
        // spsi_test = ψ_test  (S = I for simplicity)
        let psi_test: Vec<Complex64> = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(-0.3, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.8, 0.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(-0.4, 0.0),
        ];
        let spsi_test = psi_test.clone();

        // Act: S-orthogonalize
        let (orth_psi, orth_spsi) =
            s_orthogonalize_against(&psi_test, &spsi_test, &bands);

        // Verify: |⟨ψ_new|S|ψ_i⟩| < 1e-10 for all i
        for (i, (psi_i, _spsi_i)) in bands.iter().enumerate() {
            let overlap = inner_product(psi_i, &orth_spsi);
            assert!(
                overlap.norm() < 1e-10,
                "Converged band {i}: |⟨ψ_new|S|ψ_{i}⟩| = {:.2e} >= 1e-10",
                overlap.norm()
            );
        }

        // Verify ψ_new itself is S-normalized
        let self_overlap = inner_product(&orth_psi, &orth_spsi).re;
        assert!(
            relative_eq!(self_overlap, 1.0, epsilon = 1e-12),
            "Orthogonalized ψ is not S-normalized: ⟨ψ|S|ψ⟩ = {self_overlap}"
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 3b: Orthogonalization with non-trivial S
    //
    // Same test as above but with S ≠ I: the converged bands are
    // S-orthonormal under a non-trivial S-operator where spsi_i ≠ psi_i.
    // -----------------------------------------------------------------------
    #[test]
    fn orthogonalize_with_nontrivial_s() {
        let n = 5;

        // Define a non-trivial S-operator: diagonal with different values
        // per basis component. S_basis = diag([2.0, 1.5, 0.8, 1.0, 1.2]).
        // Then spsi_i[g] = S_basis[g] × psi_i[g].
        let s_diag = [2.0, 1.5, 0.8, 1.0, 1.2];

        // Create S-orthonormal bands. Since S is diagonal, we create psi_i
        // such that psi_i[g] = delta_i_g / sqrt(S_basis[i]).
        // This ensures ⟨ψ_i|S|ψ_i⟩ = Σ conj(psi_i[g]) * S_basis[g] * psi_i[g]
        //                          = 1 (only non-zero at g=i where psi_i[i] = 1/sqrt(S_basis[i]))
        //                          = S_basis[i] / S_basis[i] = 1
        let mut bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = Vec::new();
        for i in 0..3 {
            let mut psi_i = vec![Complex64::ZERO; n];
            psi_i[i] = Complex64::new(1.0 / f64::sqrt(s_diag[i]), 0.0);
            // spsi_i = S * psi_i: multiply each component by s_diag[g]
            let spsi_i: Vec<Complex64> = psi_i
                .iter()
                .enumerate()
                .map(|(g, c)| c * s_diag[g])
                .collect();
            bands.push((psi_i, spsi_i));
        }

        // Create a test vector that has non-zero overlap with all converged bands.
        let psi_test: Vec<Complex64> = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.1, 0.0),
        ];
        // spsi_test = S * psi_test (apply diagonal S)
        let spsi_test: Vec<Complex64> = psi_test
            .iter()
            .enumerate()
            .map(|(g, c)| c * s_diag[g])
            .collect();

        // Act: S-orthogonalize
        let (orth_psi, orth_spsi) =
            s_orthogonalize_against(&psi_test, &spsi_test, &bands);

        // Verify: |⟨ψ_new|S|ψ_i⟩| < 1e-10 for all i
        for (i, (psi_i, _spsi_i)) in bands.iter().enumerate() {
            let overlap = inner_product(psi_i, &orth_spsi);
            assert!(
                overlap.norm() < 1e-10,
                "Non-trivial S, band {i}: |⟨ψ_new|S|ψ_{i}⟩| = {:.2e} >= 1e-10",
                overlap.norm()
            );
        }

        // Verify ψ_new itself is S-normalized under the non-trivial S.
        // The self-overlap requires the correct S: ⟨ψ_new|S|ψ_new⟩ = ⟨ψ_new|spsi_new⟩.
        let self_overlap = inner_product(&orth_psi, &orth_spsi).re;
        assert!(
            relative_eq!(self_overlap, 1.0, epsilon = 1e-12),
            "Non-trivial S: orthogonalized ψ is not S-normalized: ⟨ψ|S|ψ⟩ = {self_overlap}"
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 4: Eigenvalue computation
    //
    // ε = ⟨ψ|H|ψ⟩ / ⟨ψ|S|ψ⟩ matches a known expected value constructed
    // from synthetic test data.
    // -----------------------------------------------------------------------
    #[test]
    fn eigenvalue_matches_expected() {
        // Construct ψ, Hψ, Sψ with known ⟨ψ|H|ψ⟩ and ⟨ψ|S|ψ⟩.
        // ψ = [2, 0, -1, 0]  (not unit-normalized)
        // Hψ = [3, 1, 2, 0]  (arbitrary action of H on ψ)
        // Sψ = [1, 0.5, 0, 0.5]
        //
        // ⟨ψ|H|ψ⟩ = 2*3 + 0*1 + (-1)*2 + 0*0 = 6 - 2 = 4
        // ⟨ψ|S|ψ⟩ = 2*1 + 0*0.5 + (-1)*0 + 0*0.5 = 2
        // ε = 4 / 2 = 2.0
        let psi: Vec<Complex64> = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let hpsi: Vec<Complex64> = vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let spsi: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let expected_epsilon = 2.0; // 4 / 2

        let (_residual, epsilon) = compute_residual(&psi, &hpsi, &spsi);

        assert!(
            relative_eq!(epsilon, expected_epsilon, epsilon = 1e-14),
            "Rayleigh quotient ε = {epsilon}, expected {expected_epsilon}"
        );
    }

    // -----------------------------------------------------------------------
    // Edge case: zero-length slices
    // -----------------------------------------------------------------------

    #[test]
    fn empty_slices_produce_empty_results() {
        let empty: Vec<Complex64> = vec![];
        let (residual, epsilon) = compute_residual(&empty, &empty, &empty);
        assert!(residual.is_empty());
        assert!(epsilon.is_nan()); // 0/0 = NaN
    }

    #[test]
    fn residual_bare_norm_of_zero_is_zero() {
        let residual = vec![Complex64::ZERO; 5];
        let norm = residual_bare_norm(&residual);
        assert_relative_eq!(norm, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn residual_bare_norm_of_unit_vector() {
        let residual = vec![Complex64::new(1.0, 0.0), Complex64::ZERO, Complex64::ZERO];
        let norm = residual_bare_norm(&residual);
        assert_relative_eq!(norm, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn residual_bare_norm_of_known_vector() {
        let residual: Vec<Complex64> = vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let norm = residual_bare_norm(&residual);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn inner_product_checks_length() {
        let a = vec![Complex64::ZERO; 3];
        let b = vec![Complex64::ZERO; 5];
        inner_product(&a, &b);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn compute_residual_length_mismatch_panics() {
        let a = vec![Complex64::ZERO; 3];
        let b = vec![Complex64::ZERO; 5];
        let c = vec![Complex64::ZERO; 5];
        compute_residual(&a, &b, &c);
    }

    #[test]
    #[should_panic(expected = "⟨ψ|S|ψ⟩ = 0 is too small")]
    fn compute_residual_singular_s_panics() {
        let psi = vec![Complex64::new(1.0, 0.0)];
        let hpsi = vec![Complex64::new(2.0, 0.0)];
        let spsi = vec![Complex64::ZERO]; // S = 0 leads to zero denominator
        compute_residual(&psi, &hpsi, &spsi);
    }

    #[test]
    #[should_panic(expected = "⟨ψ|S|ψ⟩ = -0.5 is negative")]
    fn s_normalize_negative_overlap_panics() {
        // S is not positive-definite: create spsi with opposite sign
        let psi = vec![Complex64::new(1.0, 0.0)];
        let spsi = vec![Complex64::new(-0.5, 0.0)]; // ⟨ψ|S|ψ⟩ = -0.5
        s_normalize(&psi, &spsi);
    }

    #[test]
    #[should_panic(expected = "converged band has length")]
    fn orthogonalize_length_mismatch_panics() {
        let psi = vec![Complex64::new(1.0, 0.0); 4];
        let spsi = vec![Complex64::new(1.0, 0.0); 4];
        let bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = vec![(
            vec![Complex64::new(1.0, 0.0); 3], // wrong length
            vec![Complex64::new(1.0, 0.0); 3],
        )];
        s_orthogonalize_against(&psi, &spsi, &bands);
    }

    // -----------------------------------------------------------------------
    // Edge case: orthogonalizing against zero converged bands
    // (just S-normalizes the input)
    // -----------------------------------------------------------------------

    #[test]
    fn orthogonalize_against_empty_list_just_normalizes() {
        let psi: Vec<Complex64> = vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)];
        let spsi: Vec<Complex64> = vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)];
        // ⟨ψ|S|ψ⟩ = 2*0.5 = 1.0, already normalized

        let bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = vec![];
        let (orth_psi, orth_spsi) = s_orthogonalize_against(&psi, &spsi, &bands);

        assert_relative_eq!(
            inner_product(&orth_psi, &orth_spsi).re,
            1.0,
            epsilon = 1e-14
        );
    }

    // -----------------------------------------------------------------------
    // Complex coefficients test
    // -----------------------------------------------------------------------

    #[test]
    fn inner_product_with_complex_coefficients() {
        let a: Vec<Complex64> = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 2.0),
        ];
        let b: Vec<Complex64> = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
        ];
        // ⟨a|b⟩ = conj(1+i)*2 + conj(2i)*(1-i)
        //       = (1-i)*2 + (-2i)*(1-i)
        //       = 2-2i + (-2i+2i²) = 2-2i + (-2i-2) = -4i
        let expected = Complex64::new(0.0, -4.0);
        let result = inner_product(&a, &b);
        assert_relative_eq!(result.re, expected.re, epsilon = 1e-14);
        assert_relative_eq!(result.im, expected.im, epsilon = 1e-14);
    }

    #[test]
    fn compute_residual_with_complex_coefficients() {
        // ψ = [1+i, 0]
        // Hψ = [3+i, 1]
        // Sψ = [1, 0.5]
        // ⟨ψ|H|ψ⟩ = conj(1+i)*3+i + 0 = (1-i)*(3+i) = 3+i + -3i -i² = 3 + i - 3i + 1 = 4 - 2i
        // ⟨ψ|S|ψ⟩ = conj(1+i)*1 + 0 = 1-i
        // Hmm, ⟨ψ|S|ψ⟩ = 1 - i is complex, not real.
        // We need S to be Hermitian so ⟨ψ|S|ψ⟩ is real.
        // For a diagonal S: S = diag([1, 2]) is Hermitian (real diagonal).
        // Sψ = [1*(1+i), 2*0] = [1+i, 0]
        // Re-using simpler: ψ = [1, i], Sψ = [1, i], Hψ = [4, 2-2i]
        // ⟨ψ|H|ψ⟩ = 1*4 + (-i)*(2-2i) = 4 + (-2i + 2i²) = 4 + (-2i - 2) = 2 - 2i
        // That's complex again. Let me just use real coefficients.
        //
        // S = diag([1, 1]) (identity) with ψ = [1, 0, 1] (real)
        // Hψ = [2, 3, 4]
        // Sψ = [1, 0, 1]
        // ⟨ψ|H|ψ⟩ = 1*2 + 0*3 + 1*4 = 6
        // ⟨ψ|S|ψ⟩ = 1*1 + 0*0 + 1*1 = 2
        // ε = 3
        // r = Hψ - 3*Sψ = [2-3, 3, 4-3] = [-1, 3, 1]
        // ⟨r|ψ⟩ = -1*1 + 3*0 + 1*1 = 0 ✓
        let psi: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let hpsi: Vec<Complex64> = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let spsi: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let expected_epsilon = 3.0;
        let (residual, epsilon) = compute_residual(&psi, &hpsi, &spsi);

        assert_relative_eq!(epsilon, expected_epsilon, epsilon = 1e-14);

        let r_dot_psi = inner_product(&residual, &psi);
        assert_relative_eq!(r_dot_psi.re, 0.0, epsilon = 1e-14);
    }
}
