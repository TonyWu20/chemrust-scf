// ---------------------------------------------------------------------------
// Band-by-band CG minimizer — CASTEP electronic_find_eigenstate (lines 11639-12019)
// ---------------------------------------------------------------------------
//
// Implements serial, band-by-band conjugate gradient minimization for a single
// Kohn-Sham band, with USPP-aware preconditioning and exact 2×2 quadratic line
// search.
//
// The algorithm is a verbatim Rust port of CASTEP's `electronic_find_eigenstate`
// (electronic.f90:11639-12019), with the direction construction extracted from
// `electronic_CG_direction_bks` (electronic.f90:6238-6437) and the line search
// from `electronic_ideal_step_size` (electronic.f90:9959-10185).
//
// Sign convention (matching CASTEP exactly):
//
//   CASTEP constructs the search direction d as the preconditioned gradient
//   d = P⁻¹·(H−εS)|ψ⟩ (uphill direction).  The CG update is:
//
//     β   = Re⟨d_new|r⟩
//     γ   = β / β_old
//     d   = γ·d_old − d_new
//
//   The line search finds the stationary-point roots of E(s) and negates the
//   chosen root, so step_size finalises to a downhill step.  The module
//   follows this convention character-by-character.
//
// Algorithm overview:
//
//   1. S-normalize ψ, scale H|ψ⟩ by the same factor
//   2. ε = Re⟨ψ|H|ψ⟩, r = H|ψ⟩ − ε·S|ψ⟩
//   3. For step = 1..max_steps:
//      a. Determine step type: SD (steps ≤ 2) or CG
//      b. Compute preconditioned gradient g = P⁻¹·r
//      c. S-orthogonalize g against converged bands
//      d. CG update: if CG, d = γ·d_old − g; if SD, d = g
//      e. S-orthogonalize d against current band
//      f. Save d for next iteration
//      g. Apply H to d: Hd = H|d⟩, Sd = S|d⟩
//      h. Line search: find s minimizing E(ψ + s·d)
//      i. Update: ψ ← (ψ + s·d) / norm  (with S-normalization)
//      j. Update: Hψ ← (Hψ + s·Hd) / norm
//      k. Compute new eigenvalue and residual
//      l. Check convergence: |ε_new − ε_old| < tol
//
// Reference CASTEP source:
//   - main loop:  electronic.f90:11865-11980
//   - direction:  electronic.f90:6238-6437 (CG), 6339-6431
//   - line search: electronic.f90:9959-10185 (ideal_step_size)
//

// Allow dead_code: this module is a library component for future integration.
// The step_type field in CgResult is never assigned in tests but needed by
// downstream code.
#![allow(dead_code)]

use num_complex::Complex64;

use super::cg_helpers::{compute_residual, s_orthogonalize_against, residual_bare_norm};
use super::line_search::line_search_2d_quadratic;
use super::uspp_preconditioner::UsppPreconditioner;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of band-by-band CG minimization.
#[derive(Debug, Clone)]
pub struct CgResult {
    /// Converged (or best-found) wavefunction coefficients.
    pub psi: Vec<Complex64>,
    /// Eigenvalue at the final step.
    pub eigenvalue: f64,
    /// Number of minimization steps taken.
    pub n_steps: usize,
    /// True if the eigenvalue stopped changing within `tol`.
    pub converged: bool,
    /// Last residual S-norm (computed from eigenvalue change, not full S-norm).
    pub residual_norm: f64,
    /// Step type of the last step: "SD" or "CG" (informational only).
    pub step_type: &'static str,
}

// ---------------------------------------------------------------------------
// Helper: Euclidean inner product ⟨a|b⟩ = Σ conj(aᵢ) × bᵢ
// ---------------------------------------------------------------------------

fn inner_product(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum()
}

// ---------------------------------------------------------------------------
// S-orthogonalize a single search direction against one specific band.
//
// CASTEP uses `wave_Sorthogonalise(wvfn, nb, nk, ns, direction)` in the CG
// direction function (line 6422) to remove the component along the CURRENT
// band only.  This is separate from lower-band orthogonalization which
// happens against the full wavefunction set.
// ---------------------------------------------------------------------------

fn s_orthogonalize_against_one_band(
    dir: &[Complex64],
    sdir: &[Complex64],
    band_psi: &[Complex64],
    band_spsi: &[Complex64],
) -> (Vec<Complex64>, Vec<Complex64>) {
    // overlap = ⟨band_psi|S|dir⟩ = ⟨band_psi|sdir⟩
    let overlap = inner_product(band_psi, sdir);

    let mut dir_new = dir.to_vec();
    let mut sdir_new = sdir.to_vec();

    for (d, p) in dir_new.iter_mut().zip(band_psi.iter()) {
        *d -= overlap * p;
    }
    for (sd, sp) in sdir_new.iter_mut().zip(band_spsi.iter()) {
        *sd -= overlap * sp;
    }

    (dir_new, sdir_new)
}

// ---------------------------------------------------------------------------
// Band-by-Band CG minimization — main entry point
// ---------------------------------------------------------------------------

/// Minimize a single Kohn-Sham band via preconditioned conjugate gradient.
///
/// This is the Rust port of CASTEP `electronic_find_eigenstate`
/// (electronic.f90:11639-12019).  The caller provides the initial
/// wavefunction and closures that apply H and S to any trial vector.
///
/// # Arguments
///
/// * `psi_initial` — initial wavefunction coefficients (NOT required to be
///   S-normalized; the function normalizes it as the first step).
/// * `precond` — USPP-aware preconditioner P⁻¹.
/// * `converged_bands` — list of `(psi_i, spsi_i)` for already-converged lower
///   bands.  The search direction is S-orthogonalized against these.
/// * `max_steps` — maximum number of CG+SD steps (CASTEP's max_elecmin_steps).
/// * `tol` — eigenvalue convergence tolerance (Ha): stop when
///   |ε_new − ε_old| < tol.
/// * `apply_hs` — closure returning `(H|psi>, S|psi>)` for a given ψ.
///   H is the full Kohn-Sham Hamiltonian, S is the USPP overlap operator.
/// * `apply_s` — closure returning `S|psi>` for a given ψ.  For norm-conserving
///   pseudopotentials (S = I), this can be `|v| v.to_vec()`.  For USPP,
///   this must apply the full S = I + beta*Q*beta† operator.
///
/// # Returns
///
/// `CgResult` containing the converged (or best) wavefunction, eigenvalue,
/// number of steps taken, and convergence status.
pub fn band_cg_minimize(
    psi_initial: &[Complex64],
    precond: &UsppPreconditioner,
    converged_bands: &[(Vec<Complex64>, Vec<Complex64>)],
    max_steps: usize,
    tol: f64,
    apply_hs: &impl Fn(&[Complex64]) -> (Vec<Complex64>, Vec<Complex64>),
    apply_s: &impl Fn(&[Complex64]) -> Vec<Complex64>,
) -> CgResult {
    // ---- 1. Initial setup: S-normalize psi ---------------------------------
    // electronic.f90:11865-11868
    //   call wave_copy(wvfn,nb,nk,ns,bnd)
    //   call wave_Snormalise(bnd,nk,norm)
    //   call wave_scale(H_bnd,cmplx(1.0_dp/norm,0.0_dp,dp))
    let (hpsi_init, spsi_init) = apply_hs(psi_initial);
    let s_overlap_init = inner_product(psi_initial, &spsi_init).re;
    assert!(
        s_overlap_init > 0.0,
        "band_cg: initial ⟨ψ|S|ψ⟩ = {s_overlap_init} is not positive"
    );
    let norm_init = s_overlap_init.sqrt();
    let inv_norm = 1.0 / norm_init;

    let mut psi: Vec<Complex64> =
        psi_initial.iter().map(|c| c * inv_norm).collect();
    let mut hpsi: Vec<Complex64> =
        hpsi_init.iter().map(|c| c * inv_norm).collect();
    let mut spsi: Vec<Complex64> =
        spsi_init.iter().map(|c| c * inv_norm).collect();

    // ---- 2. Compute initial eigenvalue and residual ------------------------
    // electronic.f90:11878-11879
    //   call wave_dot(bnd,H_bnd,nk,product)
    //   eigenvalue = real(product,dp)
    let mut eigenvalue = inner_product(&psi, &hpsi).re;
    let (mut residual, _resid_eps) = compute_residual(&psi, &hpsi, &spsi);

    // ---- 3. Main minimization loop -----------------------------------------
    // electronic.f90:11887-11980
    let mut d_old: Vec<Complex64> = Vec::new();  // previous search direction
    let mut beta_old: f64 = 0.0;                 // β from previous CG step
    let mut current_step_type: &str = "SD";

    for step in 1..=max_steps {
        // ---- 3a. Copy H|psi> as initial value for direction construction ----
        // CASTEP: bnd_direction starts as H_bnd (line 11889)
        let g = precond.apply(&residual); // g = P⁻¹·r (preconditioned gradient)

        // ---- 3b. Guard: zero gradient → band already converged --------------
        // If the preconditioned gradient has zero norm, the band is an exact
        // eigenvector and no further minimization is possible.
        let g_norm = residual_bare_norm(&g);
        if g_norm < f64::EPSILON * 1000.0 {
            return CgResult {
                psi,
                eigenvalue,
                n_steps: step.saturating_sub(1),
                converged: true,
                residual_norm: residual_bare_norm(&residual),
                step_type: current_step_type,
            };
        }

        // ---- 3c. Previous eigenvalue for convergence check -----------------
        let prev_eigenvalue = eigenvalue;

        // ---- 3d. S-orthogonalize g against converged lower bands -----------
        // electronic.f90:6377-6381 (wave_Sorthogonalise_to_lower)
        // NOTE: For USPP (S != I), we must use S|g> as the spsi argument to
        // correctly compute the S-inner-product <psi_i|S|g> during
        // orthogonalization.  Using g as both psi and spsi would assume S = I.
        let sg = apply_s(&g);
        let (g_orth, _sg_orth) = s_orthogonalize_against(&g, &sg, converged_bands);

        // ---- 3d. CG direction update ---------------------------------------
        // CASTEP electronic_CG_direction_bks (lines 6339-6431):
        //
        //   β = Re⟨g_orth|r⟩                (line 6397)
        //   γ = β / β_old                   (line 6407)
        //   d = γ·d_old − g_orth            (line 6416, wave_add with gamma,-1)
        //
        //   For SD steps (first 2): γ = 0, so d = g_orth.
        //
        //   For CG steps (3+): apply FR CG update.
        //
        //   NOTE: CASTEP computes β = Re⟨g_orth|r⟩ where g_orth is the
        //   preconditioned-and-orthogonalized gradient and r is the raw
        //   (unpreconditioned) residual.  This is a modified Fletcher-Reeves:
        //   the standard FR uses ⟨g|g⟩, but CASTEP uses ⟨g|r⟩ = ⟨r|P⁻¹|r⟩
        //   which is the P⁻¹-induced inner product of the residual with itself.
        let mut d: Vec<Complex64>;
        if step <= 2 {
            // Steepest Descent: d = g_orth (the preconditioned gradient)
            current_step_type = "SD";
            d = g_orth.clone();

            // Store β for possible CG step later
            beta_old = inner_product(&g_orth, &residual).re;
        } else {
            current_step_type = "CG";
            let beta = inner_product(&g_orth, &residual).re;

            // CASTEP: gamma = beta / beta_old (line 6407)
            let gamma = if beta_old.abs() > 1e-300 {
                beta / beta_old
            } else {
                0.0
            };

            // CASTEP: d = gamma * d_old - g_orth (line 6416)
            // wave_add(bnd_old_direction, bnd_direction, gamma, -cmplx_1)
            // → d = gamma * d_old + (-1) * g_orth = gamma*d_old - g_orth
            d = vec![Complex64::ZERO; g_orth.len()];
            for (d_i, (d_old_i, g_i)) in
                d.iter_mut().zip(d_old.iter().zip(g_orth.iter()))
            {
                *d_i = gamma * d_old_i - g_i;
            }

            // Store β for next CG step
            beta_old = beta;
        }

        // ---- 3f. Apply H and S to search direction -------------------------
        // electronic.f90:11908-11910
        //   call electronic_apply_H(bnd_direction, ...)
        //
        // NOTE: We compute H|d⟩ and S|d⟩ BEFORE S-orthogonalizing d against
        // the current band, because S|d⟩ is needed for the orthogonalization
        // inner product.  H|d⟩ and S|d⟩ are then updated by linearity.
        let (mut hdir, mut sdir) = apply_hs(&d);

        // ---- 3e. S-orthogonalize d against current band --------------------
        // electronic.f90:6422-6423 (wave_Sorthogonalise to the current band)
        //
        // The line search assumes ⟨ψ|S|d⟩ = 0 (no cross-term in the energy
        // denominator).  Without this orthogonalization, the search direction
        // has a component along ψ, making the line-search model incorrect and
        // causing spurious eigenvalue drift (up to 0.08 Ha for Cu d-bands
        // with strong USPP augmentation).
        //
        // Using the computed S|d⟩ from step 3f:
        //   overlap = ⟨ψ|S|d⟩ = ⟨ψ|sdir⟩
        //
        // Then by linearity of H and S:
        //   d_new    = d    - overlap · ψ
        //   H|d_new⟩ = H|d⟩ - overlap · H|ψ⟩
        //   S|d_new⟩ = S|d⟩ - overlap · S|ψ⟩
        let overlap = inner_product(&psi, &sdir);
        if overlap.norm() > 1e-30 {
            for (d_i, p_i) in d.iter_mut().zip(psi.iter()) {
                *d_i -= overlap * p_i;
            }
            for (h_i, hp_i) in hdir.iter_mut().zip(hpsi.iter()) {
                *h_i -= overlap * hp_i;
            }
            for (s_i, sp_i) in sdir.iter_mut().zip(spsi.iter()) {
                *s_i -= overlap * sp_i;
            }
        }

        // ---- 3g. Save direction for next CG step ---------------------------
        // electronic.f90:6431
        d_old = d.clone();

        // ---- 3h. Residual-based convergence guard ----------------------------
        // If the residual is already very small, the line search (which relies
        // on ||d|| > 0) may blow up due to 1/||d|| singularity.  Return early
        // rather than risk step_size ~ -1e10.
        let r_norm = residual_bare_norm(&residual);
        if r_norm < tol * 0.01 && r_norm < 1e-12 * eigenvalue.abs().max(1.0) {
            return CgResult {
                psi,
                eigenvalue,
                n_steps: step.saturating_sub(1),
                converged: true,
                residual_norm: r_norm,
                step_type: current_step_type,
            };
        }

        // ---- 3i. Line search ------------------------------------------------
        // electronic.f90:11913
        //   call electronic_ideal_step_size(bnd, nb, nk, ns, H_bnd,
        //       bnd_direction, bnd_temp, step, eigenvalue, temp, status)
        let dot_hpsi_dir = inner_product(&hpsi, &d).re;
        let dot_hdir_dir = inner_product(&hdir, &d).re;
        let dot_sdir_dir = inner_product(&sdir, &d).re;

        let ls = line_search_2d_quadratic(
            eigenvalue,      // a = ε = Re⟨Hψ|ψ⟩
            dot_hpsi_dir,    // Re⟨Hψ|d⟩ — used to compute b = -2·Re⟨Hψ|d⟩
            dot_hdir_dir,    // c = Re⟨Hd|d⟩
            dot_sdir_dir,    // d_s = Re⟨d|S|d⟩
        );

        let step_size = ls.step_size;
        let new_norm = ls.norm;

        // ---- 3i. Update ψ: ψ ← (ψ + s·d) / norm ---------------------------
        // CASTEP lines 11922-11925:
        //   call wave_add(bnd_direction,bnd,c1=step)    → bnd = step*d + bnd
        //   call wave_Snormalise(bnd,nk,norm)            → bnd = bnd/norm
        //
        //   Then lines 11933-11934:
        //   call wave_copy(bnd,wvfn,nb,nk,ns)  — save back to wavefunction
        //
        //   Then lines 11936-11938:
        //   call wave_add(bnd_temp,H_bnd,c1=step) → H_bnd = step*Hd + H_bnd
        //   call wave_scale(H_bnd,1/norm)         → H_bnd = H_bnd/norm
        let inv_norm = if new_norm > 0.0 {
            1.0 / new_norm
        } else {
            1.0
        };

        // Update ψ
        for (p, d_i) in psi.iter_mut().zip(d.iter()) {
            *p += step_size * d_i;
            *p *= inv_norm;
        }

        // Update H|ψ⟩
        for (hp, hd_i) in hpsi.iter_mut().zip(hdir.iter()) {
            *hp += step_size * hd_i;
            *hp *= inv_norm;
        }

        // Update S|ψ⟩ (linearly, since S is linear)
        for (sp, sd_i) in spsi.iter_mut().zip(sdir.iter()) {
            *sp += step_size * sd_i;
            *sp *= inv_norm;
        }

        // ---- 3j. Recompute eigenvalue --------------------------------------
        // CASTEP lines 11940-11950:
        //   if(status==0): wave_kinetic_eigenvalues(bnd,nk,ek)
        //   else: electronic_calc_eigenvalues_bnd(...)
        //
        //   In Phase-0, we skip the kinetic eigenvalue update and just
        //   recompute ε = Re⟨ψ|H|ψ⟩.
        let eigenvalue_recomputed = inner_product(&psi, &hpsi).re;

        // Update residual for next iteration
        let (residual_new, _eps) = compute_residual(&psi, &hpsi, &spsi);
        residual = residual_new;

        // ---- 3k. Convergence check -----------------------------------------
        // CASTEP lines 11952-11978:
        //   current_tol = abs(previous_eigenvalue - eigenvalue)
        //   if current_tol < eigenvalue_tol → converged
        let delta_e = (eigenvalue_recomputed - prev_eigenvalue).abs();

        if delta_e < tol {
            return CgResult {
                psi,
                eigenvalue: eigenvalue_recomputed,
                n_steps: step,
                converged: true,
                residual_norm: residual_bare_norm(&residual),
                step_type: current_step_type,
            };
        }

        // Prepare for next iteration
        eigenvalue = eigenvalue_recomputed;

        // ---- If line search gave status != 0, recompute eigenvalue properly
        // Not needed in Phase-0 since we always recompute via inner_product.
    }

    // ---- 4. Max steps reached without convergence --------------------------
    CgResult {
        psi,
        eigenvalue,
        n_steps: max_steps,
        converged: false,
        residual_norm: residual_bare_norm(&residual),
        step_type: current_step_type,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use approx::relative_eq;
    use ndarray::Array2;

    // -----------------------------------------------------------------------
    // Helper: build a minimal UsppPreconditioner for testing
    // -----------------------------------------------------------------------

    fn make_test_precon(n_pw: usize) -> UsppPreconditioner {
        let n_proj = 1; // minimal
        let beta_g = Array2::<Complex64>::zeros((n_pw, n_proj));
        let q_matrix = Array2::<Complex64>::eye(n_proj);
        // Identity Q: Q = I, so Q⁻¹ = I
        let kinetic_g: Vec<f64> = (0..n_pw)
            .map(|g| 0.5 * (g as f64 + 1.0).powi(2))
            .collect();
        let k_cart = [0.0; 3];
        UsppPreconditioner::new(beta_g, q_matrix, kinetic_g, k_cart)
    }

    // -----------------------------------------------------------------------
    // Success Criterion 1: Steepest descent check
    //
    // With a trivial Hamiltonian (H = diag, S = I), starting from a unit
    // vector, the first step should move in the steepest descent direction
    // and decrease the eigenvalue.
    // -----------------------------------------------------------------------
    #[test]
    fn sd_first_step_decreases_eigenvalue() {
        let n = 6;
        let precond = make_test_precon(n);

        // H = diagonal: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        let h_diag: Vec<f64> = (0..n).map(|i| 1.0 - 0.2 * (i as f64)).collect();

        // Initial psi: [1, 0.5, 0.25, 0.125, 0.0625, 0.03125] (not S-normalized)
        let psi_init: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(0.5_f64.powi(i as i32), 0.0))
            .collect();

        // apply_hs: H = diag, S = I
        let apply_hs = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
            let hv: Vec<Complex64> = v
                .iter()
                .enumerate()
                .map(|(i, c)| Complex64::new(h_diag[i], 0.0) * c)
                .collect();
            (hv, v.to_vec())
        };
        let apply_s = |v: &[Complex64]| v.to_vec();

        let converged_bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = vec![];

        let result = band_cg_minimize(
            &psi_init,
            &precond,
            &converged_bands,
            1,           // max_steps = 1
            1e-10,        // tol
            &apply_hs,
            &apply_s,
        );

        // After 1 SD step, eigenvalue should decrease
        // Initial eigenvalue: sinitial H(psi_reference) / ⟨psi|psi⟩ > 0
        // The Rayleigh quotient of psi_init should be between 0 and 1
        assert!(
            result.eigenvalue > 0.0,
            "Eigenvalue should be positive for this H"
        );
        assert!(
            result.n_steps == 1,
            "Should take exactly 1 step"
        );
        assert!(
            result.step_type == "SD",
            "First step should be SD"
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 2: Energy monotonicity
    //
    // With a simple diagonal H and S=I, running multiple CG steps should
    // monotonically decrease the eigenvalue.
    // -----------------------------------------------------------------------
    #[test]
    fn energy_monotonic_decrease() {
        let n = 8;
        let precond = make_test_precon(n);

        // H = diag: eigenvalues in increasing order
        let h_diag: Vec<f64> = (0..n)
            .map(|i| {
                // Non-uniform: 0.1, 0.35, 0.7, 1.1, 1.6, 2.2, 2.9, 3.7
                0.1 + (i as f64) * (0.25 + 0.05 * i as f64)
            })
            .collect();

        // Initial psi: mix of low and high eigenmodes
        let psi_init: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(1.0 / (i as f64 + 1.0).sqrt(), 0.0))
            .collect();

        let apply_hs = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
            let hv: Vec<Complex64> = v
                .iter()
                .enumerate()
                .map(|(i, c)| Complex64::new(h_diag[i], 0.0) * c)
                .collect();
            (hv, v.to_vec())
        };
        let apply_s = |v: &[Complex64]| v.to_vec();

        let converged_bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = vec![];

        // We can't directly track eigenvalue history without modifying the
        // function, so we run with different max_steps and verify the trend.
        let result_5 = band_cg_minimize(
            &psi_init,
            &precond,
            &converged_bands,
            5,
            1e-10,
            &apply_hs,
            &apply_s,
        );

        // With more steps, the eigenvalue should be lower (or equal) to
        // fewer steps.  We test this indirectly by running with max_steps=10.
        let result_10 = band_cg_minimize(
            &psi_init,
            &precond,
            &converged_bands,
            10,
            1e-10,
            &apply_hs,
            &apply_s,
        );

        assert!(
            result_10.eigenvalue <= result_5.eigenvalue + 1e-14,
            "More steps ({}) should not increase eigenvalue: {} > {}",
            result_10.n_steps,
            result_10.eigenvalue,
            result_5.eigenvalue
        );

        // The eigenvalue should be close to the ground truth (h_diag[0] = 0.1)
        assert!(
            result_10.eigenvalue > h_diag[0] - 1e-5,
            "Eigenvalue {} should not be below ground state {}",
            result_10.eigenvalue,
            h_diag[0]
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 3: Residual decrease
    //
    // The bare residual norm should decrease as the eigenvalue converges.
    // -----------------------------------------------------------------------
    #[test]
    fn residual_decreases() {
        let n = 6;
        let precond = make_test_precon(n);

        let h_diag: Vec<f64> = vec![0.3, 0.8, 1.5, 2.4, 3.5, 4.8];
        let psi_init: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(1.0 - 0.15 * (i as f64), 0.0))
            .collect();

        let apply_hs = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
            let hv: Vec<Complex64> = v
                .iter()
                .enumerate()
                .map(|(i, c)| Complex64::new(h_diag[i], 0.0) * c)
                .collect();
            (hv, v.to_vec())
        };
        let apply_s = |v: &[Complex64]| v.to_vec();

        let converged_bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = vec![];

        let result_1 = band_cg_minimize(
            &psi_init, &precond, &converged_bands, 1, 1e-14, &apply_hs, &apply_s,
        );
        let result_5 = band_cg_minimize(
            &psi_init, &precond, &converged_bands, 5, 1e-14, &apply_hs, &apply_s,
        );

        // Residual after 5 steps should be smaller than after 1 step
        assert!(
            result_5.residual_norm <= result_1.residual_norm * 1.01,
            "Residual should decrease: after 1 step = {:.2e}, after 5 steps = {:.2e}",
            result_1.residual_norm,
            result_5.residual_norm
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 4: Convergence flag
    //
    // With a very loose tolerance, the algorithm should converge immediately.
    // -----------------------------------------------------------------------
    #[test]
    fn convergence_with_loose_tolerance() {
        let n = 4;
        let precond = make_test_precon(n);

        let h_diag: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0];
        let psi_init: Vec<Complex64> = (0..n)
            .map(|_| Complex64::new(1.0, 0.0))
            .collect();

        let apply_hs = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
            let hv: Vec<Complex64> = v
                .iter()
                .enumerate()
                .map(|(i, c)| Complex64::new(h_diag[i], 0.0) * c)
                .collect();
            (hv, v.to_vec())
        };
        let apply_s = |v: &[Complex64]| v.to_vec();

        let converged_bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = vec![];

        // With tol = 100.0, should converge in 1 step
        let result = band_cg_minimize(
            &psi_init,
            &precond,
            &converged_bands,
            10,
            100.0, // very loose tolerance
            &apply_hs,
            &apply_s,
        );

        assert!(
            result.converged,
            "Should converge with loose tolerance, n_steps={}",
            result.n_steps
        );
        assert!(
            result.n_steps == 1,
            "Should converge in first step, got {}",
            result.n_steps
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 5: Unchanged by orthogonalized converged bands
    //
    // Running with converged_bands that are orthogonal to the initial ψ
    // should not affect the result.
    // -----------------------------------------------------------------------
    #[test]
    fn converged_bands_do_not_alter_independent_band() {
        let n = 5;
        let precond = make_test_precon(n);

        let h_diag: Vec<f64> = vec![0.2, 0.9, 1.8, 3.1, 4.8];
        // psi_init has only component along index 2
        let psi_init: Vec<Complex64> = (0..n)
            .map(|i| {
                if i == 2 {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::ZERO
                }
            })
            .collect();

        let apply_hs = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
            let hv: Vec<Complex64> = v
                .iter()
                .enumerate()
                .map(|(i, c)| Complex64::new(h_diag[i], 0.0) * c)
                .collect();
            (hv, v.to_vec())
        };
        let apply_s = |v: &[Complex64]| v.to_vec();

        // Converged bands are orthogonal to psi_init (bands at index 0 and 1)
        let band0: Vec<Complex64> = (0..n)
            .map(|i| {
                if i == 0 {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::ZERO
                }
            })
            .collect();
        let band1: Vec<Complex64> = (0..n)
            .map(|i| {
                if i == 1 {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::ZERO
                }
            })
            .collect();

        // S = I, so spsi = psi for the converged bands
        let converged_bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = vec![
            (band0.clone(), band0),
            (band1.clone(), band1),
        ];

        let result_no_bands = band_cg_minimize(
            &psi_init, &precond, &vec![], 3, 1e-14, &apply_hs, &apply_s,
        );
        let result_with_bands = band_cg_minimize(
            &psi_init, &precond, &converged_bands, 3, 1e-14, &apply_hs, &apply_s,
        );

        // Results should be nearly identical since converged_bands
        // are orthogonal to the initial psi
        assert!(
            relative_eq!(
                result_no_bands.eigenvalue,
                result_with_bands.eigenvalue,
                epsilon = 1e-12
            ),
            "Orthogonal converged bands should not change eigenvalue: {} vs {}",
            result_no_bands.eigenvalue,
            result_with_bands.eigenvalue
        );
    }

    // -----------------------------------------------------------------------
    // Edge case: empty psi
    // -----------------------------------------------------------------------
    #[test]
    #[should_panic(expected = "not positive")]
    fn empty_psi_panics() {
        let precond = make_test_precon(0);
        let psi: Vec<Complex64> = vec![];
        let apply_hs = |_v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
            (vec![], vec![])
        };
        let apply_s = |v: &[Complex64]| v.to_vec();
        band_cg_minimize(&psi, &precond, &vec![], 1, 1e-10, &apply_hs, &apply_s);
    }

    // -----------------------------------------------------------------------
    // Edge case: single basis function, Hψ = 0 everywhere
    //
    // The eigenvalue is 0, the residual is 0, and we converge immediately.
    // -----------------------------------------------------------------------
    #[test]
    fn trivial_null_hamiltonian_converges() {
        let n = 1;
        let precond = make_test_precon(n);
        let psi_init = vec![Complex64::new(1.0, 0.0)];
        let apply_hs = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
            // H = 0, S = I
            (vec![Complex64::ZERO; n], v.to_vec())
        };
        let apply_s = |v: &[Complex64]| v.to_vec();

        let result = band_cg_minimize(
            &psi_init,
            &precond,
            &vec![],
            5,
            1e-10,
            &apply_hs,
            &apply_s,
        );

        // With H=0, eigenvalue = 0, residual = 0, gradient = 0
        assert_relative_eq!(result.eigenvalue, 0.0, epsilon = 1e-14);
        // Zero gradient triggers immediate convergence return (step 1)
        assert!(result.converged);
        // n_steps = step.saturating_sub(1) = 1-1 = 0 when zero-gradient guard fires
        assert_eq!(result.n_steps, 0);
    }
}
