//! Synthetic locked-input construction helpers for Gate 3' Davidson
//! identity-preservation tests.
//!
//! Provides `compute_s_block_sum` (S-weighted subspace overlap) and
//! `construct_synthetic_locked_input` (build a wavefunction whose locked
//! bands are bitwise-identical to a reference, with random noise on all
//! other bands followed by S-orthogonalization against the locked subspace).

use chemrust_hamiltonian_core::NonSpin;
use chemrust_scf::{MixingOff, ScfIteration, VEffBuilt};
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Compute the S-weighted block sum: Σ_{a,b ∈ band_range} |⟨ψ_a | S·ψ_ref_b⟩|²
///
/// `s_psi_ref` is S·psi_ref precomputed via `apply_s_for_test`.
/// Both `psi` and `s_psi_ref` are flat column-major arrays (n_bands × n_pw).
pub fn compute_s_block_sum(
    psi: &[Complex64],
    s_psi_ref: &[Complex64],
    n_pw: usize,
    band_range: std::ops::Range<usize>,
) -> f64 {
    let mut total = 0.0_f64;
    for a in band_range.clone() {
        let band_a = &psi[a * n_pw..(a + 1) * n_pw];
        for b in band_range.clone() {
            let s_ref_b = &s_psi_ref[b * n_pw..(b + 1) * n_pw];
            let dot: Complex64 = band_a
                .iter()
                .zip(s_ref_b.iter())
                .map(|(x, y)| x.conj() * y)
                .sum();
            total += dot.norm_sqr();
        }
    }
    total
}

/// Construct a synthetic wavefunction where bands in `locked_range` are
/// bitwise identical to `psi_ref` and all other bands have random noise,
/// then are S-orthogonalized against the locked subspace.
///
/// # Algorithm
///
/// 1. Clone `psi_ref` → `psi_perturbed`.
/// 2. For `b ∉ locked_range`: add random complex noise (L2-norm ≈ epsilon)
///    drawn from `StdRng(seed)`.
/// 3. Compute `S·psi_perturbed` via `state_for_s.apply_s_for_test`.
/// 4. For each non-locked band `b`:
///    a. Save the pre-projection band vector.
///    b. Compute `dot_j = ⟨psi_ref[j] | S·psi_perturbed[b]⟩` for each
///       locked `j`.
///    c. `psi_perturbed[b] -= Σ_j dot_j · psi_ref[j]`.
///    d. `norm_initial² = Re⟨pre-projection-b | S·psi_perturbed[b]⟩`.
///    e. `norm_new² = norm_initial² - Σ_j |dot_j|²` (holds because the
///       locked subspace is S-orthonormal).
///    f. `psi_perturbed[b] /= √norm_new²`.
pub fn construct_synthetic_locked_input(
    psi_ref: &[Complex64],
    n_pw: usize,
    n_bands: usize,
    locked_range: std::ops::Range<usize>,
    epsilon: f64,
    seed: u64,
    state_for_s: &ScfIteration<NonSpin, VEffBuilt, MixingOff>,
) -> Vec<Complex64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut psi_perturbed = psi_ref.to_vec();

    // Step 1: Add noise to non-locked bands
    for b in 0..n_bands {
        if locked_range.contains(&b) {
            continue;
        }
        let offset = b * n_pw;
        for g in 0..n_pw {
            let re: f64 = rng.gen_range(-1.0..1.0);
            let im: f64 = rng.gen_range(-1.0..1.0);
            psi_perturbed[offset + g] = Complex64::new(
                psi_perturbed[offset + g].re + epsilon * re,
                psi_perturbed[offset + g].im + epsilon * im,
            );
        }
    }

    // Step 2: Compute S·psi_perturbed (one GPU call)
    let s_perturbed = state_for_s
        .apply_s_for_test(&psi_perturbed, n_bands)
        .expect("apply_s_for_test in construct_synthetic_locked_input");

    // Step 3: S-orthogonalize non-locked bands against locked bands
    let locked_indices: Vec<usize> = locked_range.clone().collect();

    for b in 0..n_bands {
        if locked_range.contains(&b) {
            continue;
        }

        let psi_b_init = psi_perturbed[b * n_pw..(b + 1) * n_pw].to_vec();
        let s_pert_b = &s_perturbed[b * n_pw..(b + 1) * n_pw];

        // norm_initial² = Re⟨psi_b_init | S·psi_b_init⟩
        let s_self: Complex64 = psi_b_init
            .iter()
            .zip(s_pert_b.iter())
            .map(|(p, s)| p.conj() * s)
            .sum();
        let norm_sq_initial = s_self.re;

        // Compute dots and project
        let mut dots_sq_sum = 0.0_f64;
        for &j in &locked_indices {
            let psi_ref_j = &psi_ref[j * n_pw..(j + 1) * n_pw];
            let dot: Complex64 = psi_ref_j
                .iter()
                .zip(s_pert_b.iter())
                .map(|(p, s)| p.conj() * s)
                .sum();
            dots_sq_sum += dot.norm_sqr();

            // psi_perturbed[b] -= dot · psi_ref[j]
            for g in 0..n_pw {
                psi_perturbed[b * n_pw + g].re -=
                    dot.re * psi_ref_j[g].re - dot.im * psi_ref_j[g].im;
                psi_perturbed[b * n_pw + g].im -=
                    dot.re * psi_ref_j[g].im + dot.im * psi_ref_j[g].re;
            }
        }

        let norm_sq_new = norm_sq_initial - dots_sq_sum;
        let inv_norm = 1.0 / norm_sq_new.max(1e-30).sqrt();
        for g in 0..n_pw {
            psi_perturbed[b * n_pw + g].re *= inv_norm;
            psi_perturbed[b * n_pw + g].im *= inv_norm;
        }
    }

    psi_perturbed
}
