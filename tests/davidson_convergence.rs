// ---------------------------------------------------------------------------
// Integration tests for inner-loop Davidson convergence check
// ---------------------------------------------------------------------------
//
// Tests for `check_inner_convergence` which implements CASTEP's
// hamiltonian.f90:1178-1228 convergence logic including:
//   (a) Absolute tolerance with EPS guard for large eigenvalues
//   (b) Relative stagnation detection (0.3 heuristic when tol_rel ≤ 0)
//   (c) Uphill eigenvalue override (numerical noise near convergence)
//
// All assertions are anchored to the CASTEP reference implementation,
// not to circular round-trip or vacuous properties.

use chemrust_scf::{check_inner_convergence, BandConvStatus};

// -----------------------------------------------------------------------
// (a) Absolute tolerance
// -----------------------------------------------------------------------

#[test]
fn test_absolute_tolerance_convergence() {
    // |ΔE| < tol_abs AND new < prev (downhill) → band converges
    // Source: hamiltonian.f90:1187-1192
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        1.0,          // prev_eig
        1.0 - 5e-9,   // new_eig (|ΔE| = 5e-9, downhill direction)
        1e-8,         // tol_abs (|ΔE| < tol_abs)
        0.0,          // tol_rel (not set)
        &mut break_cond_tol,
        true,         // is_first_step
        0,            // outer_iter
        10,           // max_outer_iter
    );
    assert!(
        result.converged,
        "|ΔE|=5e-9 < tol_abs=1e-8 (downhill) should converge"
    );
    assert!(
        !result.opt_stopped,
        "First step should not trigger opt_stopped"
    );
}

#[test]
fn test_absolute_tolerance_not_converged() {
    // |ΔE| > tol_abs → band NOT converged
    // Source: hamiltonian.f90:1187-1192
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        1.0,          // prev_eig
        1.0 + 3e-8,   // new_eig (|ΔE| = 3e-8)
        1e-8,         // tol_abs (|ΔE| > tol_abs)
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(
        !result.converged,
        "|ΔE|=3e-8 > tol_abs=1e-8 should NOT converge"
    );
}

#[test]
fn test_absolute_tolerance_exact_zero_diff() {
    // Zero change → always converged
    // Source: hamiltonian.f90:1187
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        1.0,
        1.0,
        1e-8,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(result.converged, "Zero diff should converge");
}

// -----------------------------------------------------------------------
// EPS guard for large eigenvalues
// -----------------------------------------------------------------------

#[test]
fn test_eps_guard_large_eigenvalue() {
    // For a large eigenvalue, the EPS guard dominates tol_abs.
    //
    // eps_guard = 2 * |100| * EPS ≈ 2 * 100 * 2.22e-16 ≈ 4.44e-14
    // With tol_abs = 1e-15 (smaller than eps_guard):
    //   threshold = max(1e-15, 4.44e-14) = 4.44e-14
    //
    // diff = 3e-14 < 4.44e-14 → band converges via EPS guard
    // diff = 3e-14 > 1e-15 → would NOT converge with tol_abs alone
    //
    // Source: hamiltonian.f90:1187-1192, eps guard formula 2*|eig|*eps
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        100.0,             // prev_eig (large eigenvalue)
        100.0 + 3e-14,     // new_eig
        1e-15,             // tol_abs (smaller than eps_guard)
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(
        result.converged,
        "|ΔE|=3e-14 < eps_guard≈4.44e-14 should converge via EPS guard \
         (diff={} would NOT converge with tol_abs={} alone)",
        3e-14, 1e-15
    );
}

#[test]
fn test_eps_guard_large_eigenvalue_not_converged() {
    // diff above eps_guard → not converged
    // Source: hamiltonian.f90:1187-1192
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        100.0,
        100.0 + 1e-13,  // |ΔE| = 1e-13 > eps_guard ≈ 4.44e-14
        1e-15,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(
        !result.converged,
        "|ΔE|=1e-13 > eps_guard≈4.44e-14 should NOT converge"
    );
}

#[test]
fn test_eps_guard_tol_abs_dominates_for_small_eigenvalues() {
    // For typical eigenvalues (|eig| ≈ 1), tol_abs dominates:
    //   threshold = max(1e-8, 2*|1|*EPS ≈ 4.4e-16) = 1e-8
    // diff = 5e-9 < 1e-8 → converged (downhill direction)
    // Source: hamiltonian.f90:1187-1192, tol_abs > eps_guard for small eig
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        1.0,
        1.0 - 5e-9,  // downhill direction
        1e-8,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(result.converged, "tol_abs dominates for small eigenvalues (downhill)");

    let mut break_cond_tol = 0.0;
    let result2 = check_inner_convergence(
        1.0,
        1.0 - 2e-8,  // |ΔE| > tol_abs (downhill)
        1e-8,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(!result2.converged, "diff > tol_abs should not converge");
}

// -----------------------------------------------------------------------
// (b) Relative stagnation — first step sets baseline
// -----------------------------------------------------------------------

#[test]
fn test_relative_stagnation_first_step() {
    // First step: break_cond_tol is set to |ΔE|, no opt_stop triggered.
    // Source: hamiltonian.f90:1195-1216, Step 1
    let mut break_cond_tol = 0.0;
    let delta_e = 1e-6;
    let result = check_inner_convergence(
        1.0,
        1.0 + delta_e,  // |ΔE| = 1e-6
        1e-8,           // tol_abs smaller than |ΔE|
        0.0,
        &mut break_cond_tol,
        true,            // is_first_step
        0,
        10,
    );
    assert!(!result.converged, "First step with |ΔE| > tol_abs should not converge");
    assert!(!result.opt_stopped, "First step should not trigger opt_stopped");
    assert!(
        (break_cond_tol - delta_e).abs() < 1e-15,
        "break_cond_tol should be set to |ΔE|={}, got {}",
        delta_e, break_cond_tol,
    );
}

// -----------------------------------------------------------------------
// (b) Relative stagnation — second step with 0.3 heuristic
// -----------------------------------------------------------------------

#[test]
fn test_relative_stagnation_second_step() {
    // Second step: if |ΔE| < break_cond_tol * 0.3 → opt_stopped = true.
    //
    // Simulate: first step had |ΔE| = 1e-6 → break_cond_tol = 1e-6
    //           second step |ΔE| = 2e-7 < 1e-6 * 0.3 = 3e-7
    //
    // Source: hamiltonian.f90:1195-1216, Step 2 with tol_rel ≤ 0.
    // The 0.3 heuristic is CASTEP's stagnation detection.
    let mut break_cond_tol = 1e-6;  // set by first step
    let result = check_inner_convergence(
        1.0,
        1.0 + 2e-7,  // |ΔE| = 2e-7
        1e-8,         // tol_abs (|ΔE| > tol_abs)
        0.0,          // tol_rel ≤ 0 → use 0.3 heuristic
        &mut break_cond_tol,
        false,   // not first step
        0,       // outer_iter < max_outer_iter
        10,
    );
    assert!(
        !result.converged,
        "|ΔE|=2e-7 > tol_abs=1e-8, not converged by abs tol"
    );
    assert!(
        result.opt_stopped,
        "|ΔE|=2e-7 < break_cond_tol*0.3=3e-7 should trigger opt_stopped"
    );
}

#[test]
fn test_relative_stagnation_no_opt_stop_when_delta_large() {
    // Second step, |ΔE| NOT small enough → no opt_stop
    // Source: hamiltonian.f90:1195-1216
    let mut break_cond_tol = 1e-6;
    let result = check_inner_convergence(
        1.0,
        1.0 + 5e-7,  // |ΔE| = 5e-7 > 3e-7 = break_cond_tol * 0.3
        1e-8,
        0.0,
        &mut break_cond_tol,
        false,
        0,
        10,
    );
    assert!(!result.converged, "|ΔE|=5e-7 > tol_abs=1e-8");
    assert!(
        !result.opt_stopped,
        "|ΔE|=5e-7 not < break_cond_tol*0.3=3e-7, should NOT opt_stop"
    );
}

#[test]
fn test_relative_stagnation_opt_stop_with_absolute_convergence() {
    // Band converges BOTH by abs tol AND stagnation in the same step.
    // |ΔE| < tol_abs AND |ΔE| < break_cond_tol * 0.3
    //
    // First step: |ΔE| = 1e-6 → break_cond_tol = 1e-6
    // Second step: |ΔE| = 5e-9 < 1e-8 (tol_abs) AND < 3e-7 (break_cond_tol * 0.3)
    // Use downhill direction (new < prev) to avoid uphill override.
    // Result: converged=true, opt_stopped=true
    // Source: hamiltonian.f90:1187-1192 and 1195-1216
    let mut break_cond_tol = 1e-6;
    let result = check_inner_convergence(
        1.0,
        1.0 - 5e-9,  // |ΔE| = 5e-9, downhill
        1e-8,         // tol_abs
        0.0,
        &mut break_cond_tol,
        false,
        0,
        10,
    );
    assert!(
        result.converged,
        "|ΔE|=5e-9 < tol_abs=1e-8 (downhill) should converge"
    );
    assert!(
        result.opt_stopped,
        "|ΔE|=5e-9 < break_cond_tol*0.3=3e-7 should opt_stop"
    );
}

// -----------------------------------------------------------------------
// (b) tol_rel > 0 path
// -----------------------------------------------------------------------

#[test]
fn test_relative_stagnation_with_tol_rel() {
    // When tol_rel > 0, the user-specified ratio is used instead of 0.3.
    // First step: |ΔE| = 0.1 → break_cond_tol = 0.1
    // Second step: |ΔE| = 0.001 < 0.1 * 0.02 = 0.002 → converged & opt_stopped
    // Use downhill direction (new < prev) to avoid uphill override.
    //
    // (tol_rel=0.02 means: <2% of first-step improvement → stop)
    // Source: hamiltonian.f90:1195-1216, tol_rel > 0 path
    let mut break_cond_tol = 0.1;
    let result = check_inner_convergence(
        1.0,
        1.0 - 1e-3,  // |ΔE| = 0.001, downhill
        1e-8,
        0.02,          // tol_rel = 2%
        &mut break_cond_tol,
        false,
        0,
        10,
    );
    assert!(
        result.converged,
        "|ΔE|=0.001 < break_cond_tol*0.02=0.002 should converge with tol_rel"
    );
    assert!(
        result.opt_stopped,
        "tol_rel > 0 should also set opt_stopped"
    );
}

#[test]
fn test_relative_stagnation_tol_rel_not_met() {
    // When tol_rel > 0 but |ΔE| > break_cond_tol * tol_rel
    // Source: hamiltonian.f90:1195-1216
    let mut break_cond_tol = 0.1;
    let result = check_inner_convergence(
        1.0,
        1.0 + 1e-2,  // |ΔE| = 0.01 > 0.1*0.02=0.002
        1e-8,
        0.02,
        &mut break_cond_tol,
        false,
        0,
        10,
    );
    assert!(!result.converged, "|ΔE| > break_cond_tol * tol_rel");
    assert!(!result.opt_stopped, "no stagnation");
}

// -----------------------------------------------------------------------
// (c) Uphill eigenvalue detection
// -----------------------------------------------------------------------

#[test]
fn test_uphill_not_converged() {
    // Eigenvalue went UP: prev - new < -100 * max(EPS, EPS * |prev|).
    //
    // For prev=1.0: uphill_threshold = -100 * max(EPS, EPS*1.0) = -2.22e-14
    //   new = 1.0 + 1e-10 → prev - new = -1e-10 < -2.22e-14 → uphill
    //   |ΔE| = 1e-10 < tol_abs = 1e-8 → would be converged by abs tol
    //   BUT uphill overrides: converged = false
    //
    // Source: hamiltonian.f90:1218-1224
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        1.0,            // prev_eig
        1.0 + 1e-10,    // new_eig (larger than prev)
        1e-8,           // tol_abs
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(
        !result.converged,
        "Uphill: eigenvalue increased by 1e-10 should NOT be converged \
         even though |ΔE| < tol_abs"
    );
}

#[test]
fn test_uphill_downhill_normal_convergence() {
    // Eigenvalue went DOWN (normal convergence): should converge normally.
    // prev - new = 1.0 - (1.0 - 9e-7) = 9e-7 > -2.22e-14 → no uphill
    // |ΔE| = 9e-7 < 1e-6 → converges normally
    // Source: hamiltonian.f90:1218-1224
    // (Uphill condition must NOT fire for downhill convergence)
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        1.0,
        1.0 - 9e-7,  // new_eig < prev_eig (normal)
        1e-6,         // tol_abs
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(
        result.converged,
        "Normal downhill convergence should be marked converged"
    );
}

#[test]
fn test_uphill_zero_prev_eig() {
    // Edge case: prev_eig = 0.0
    // uphill_threshold = -100 * max(EPS, EPS*|0.0|) = -100*EPS = -2.22e-14
    // new = 0.0 + 1e-12 → prev - new = -1e-12 < -2.22e-14 → uphill
    // |ΔE| = 1e-12 < 1e-8 → would be converged by abs tol
    // BUT uphill overrides
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        0.0,            // prev_eig = 0.0
        1e-12,          // new_eig > prev_eig
        1e-8,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(
        !result.converged,
        "Uphill with prev_eig=0 should prevent convergence"
    );
}

#[test]
fn test_uphill_no_override_when_not_converged_by_abs_tol() {
    // Uphill triggers, but band was already NOT converged by abs tol.
    // converged=false before uphill check, stays false after.
    //
    // prev=1.0, new=1.0+5e-7 (uphill, but |ΔE|=5e-7 > tol_abs=1e-8)
    // Already not converged by (a). Uphill check: converged=false (no-op).
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        1.0,
        1.0 + 5e-7,  // uphill AND |ΔE| > tol_abs
        1e-8,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(
        !result.converged,
        "Both uphill and |ΔE| > tol_abs → not converged"
    );
}

// -----------------------------------------------------------------------
// Negative eigenvalues
// -----------------------------------------------------------------------

#[test]
fn test_negative_eigenvalues_converge() {
    // Negative eigenvalues should be handled correctly.
    // |ΔE| < tol_abs AND new < prev (more negative = lower) → converged
    // Source: hamiltonian.f90:1187-1192
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        -10.0,
        -10.0 - 1e-9,  // |ΔE| = 1e-9, new more negative (downhill)
        1e-8,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(result.converged, "Negative eig: |ΔE|=1e-9 < 1e-8 (downhill)");

    let mut break_cond_tol = 0.0;
    let result2 = check_inner_convergence(
        -10.0,
        -11.0,  // |ΔE| = 1.0
        1e-8,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(!result2.converged, "Negative eig: |ΔE|=1.0 > 1e-8");
}

// -----------------------------------------------------------------------
// edge: break_cond_tol = 0 (no prior baseline)
// -----------------------------------------------------------------------

#[test]
fn test_break_cond_tol_zero_not_first_step() {
    // If break_cond_tol is 0.0 and is_first_step = false (shouldn't normally
    // happen, but check robustness): the 0.3 heuristic check should not
    // crash or produce wrong results.
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        1.0,
        1.0 + 1e-6,
        1e-8,
        0.0,
        &mut break_cond_tol,
        false,
        0,
        10,
    );
    // delta_e = 1e-6, break_cond_tol * 0.3 = 0.0
    // delta_e < 0.0 is false → no opt_stop
    assert!(!result.converged, "|ΔE|=1e-6 > tol_abs=1e-8");
    assert!(!result.opt_stopped, "break_cond_tol=0 prevents opt_stop");
}

// -----------------------------------------------------------------------
// Test on last outer iteration: 0.3 heuristic disabled
// -----------------------------------------------------------------------

#[test]
fn test_relative_stagnation_last_outer_iter_disabled() {
    // On the last outer iteration (outer_iter == max_outer_iter - 1,
    // i.e. iteration 9 with max_outer_iter=10), the 0.3 heuristic
    // should NOT trigger opt_stopped.
    // This ensures the solver doesn't exit prematurely on the final
    // iteration.
    //
    // With outer_iter + 1 < max_outer_iter: 9 + 1 = 10 < 10 → false.
    // So 0.3 heuristic is disabled.
    //
    // Use downhill direction to isolate the opt_stop behavior.
    // Source: hamiltonian.f90:1195-1216, "If tol_rel ≤ 0 AND outer_iter < max"
    // with 1-based Fortran indexing: outer_iter < max means
    // outer_iter = 1..max-1, not the last iteration.
    let mut break_cond_tol = 1e-6;
    let result = check_inner_convergence(
        1.0,
        1.0 - 2e-7,  // |ΔE| < break_cond_tol * 0.3, downhill
        1e-8,
        0.0,
        &mut break_cond_tol,
        false,
        9,    // outer_iter = 9, max_outer_iter = 10 (last iter)
        10,
    );
    assert!(!result.converged, "|ΔE| > tol_abs");
    assert!(
        !result.opt_stopped,
        "0.3 heuristic disabled on last outer iteration"
    );
}

// -----------------------------------------------------------------------
// tol_rel with tol_rel <= 0 and last iteration
// -----------------------------------------------------------------------

#[test]
fn test_relative_stagnation_non_last_uses_heuristic() {
    // Non-last iteration: 0.3 heuristic IS active.
    // Same as last-iter test but outer_iter=8 (not last).
    let mut break_cond_tol = 1e-6;
    let result = check_inner_convergence(
        1.0,
        1.0 + 2e-7,
        1e-8,
        0.0,
        &mut break_cond_tol,
        false,
        8,    // not last
        10,
    );
    assert!(
        result.opt_stopped,
        "0.3 heuristic active on non-last iteration"
    );
}

// -----------------------------------------------------------------------
// Discriminator: wrong convergence without EPS guard
// -----------------------------------------------------------------------

#[test]
fn test_discriminator_eps_guard_vs_naive() {
    // A naive implementation comparing only |ΔE| < tol_abs would
    // FALSE-NEGATIVE on large eigenvalues where machine precision
    // limits convergence.
    //
    // Here: tol_abs = 1e-15, |ΔE| = 3e-14 > tol_abs.
    // Naive says: not converged.
    // Correct (with EPS guard): A large eigenvalue (100 Ha) raises
    // threshold to eps_guard ≈ 4.44e-14 > 3e-14 → converged.
    //
    // Source: hamiltonian.f90:1187-1192, "2*|eig|*eps" guard
    let mut break_cond_tol = 0.0;
    let result = check_inner_convergence(
        100.0,
        100.0 + 3e-14,
        1e-15,
        0.0,
        &mut break_cond_tol,
        true,
        0,
        10,
    );
    assert!(
        result.converged,
        "Discriminator: EPS guard must catch this case where naively |ΔE| > tol_abs"
    );
}

// -----------------------------------------------------------------------
// Smoke test: BandConvStatus Debug and Clone
// -----------------------------------------------------------------------

#[test]
fn test_band_conv_status_debug_clone() {
    let status = BandConvStatus {
        converged: true,
        opt_stopped: false,
    };
    let _ = format!("{:?}", status);
    let cloned = status.clone();
    assert_eq!(cloned.converged, status.converged);
    assert_eq!(cloned.opt_stopped, status.opt_stopped);
}
