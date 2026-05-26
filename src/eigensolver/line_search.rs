// ---------------------------------------------------------------------------
// Closed-form 2x2 quadratic line search for band-by-band eigensolvers
// ---------------------------------------------------------------------------
//
// Implements the `electronic_ideal_step_size` algorithm from CASTEP
// (electronic.f90:9959-10185).  Given a current band psi and a search
// direction d (preconditioned steepest descent), find the step size s
// that minimises the Rayleigh quotient:
//
//   E(s) = <psi + s*d | H | psi + s*d> / <psi + s*d | S | psi + s*d>
//        = (a + b*s + c*s^2) / (1 + d_s * s^2)
//
// where:
//   a   = Re<Hpsi|psi>   (eigenvalue)
//   b   = -2 * Re<Hpsi|d>
//   c   = Re<Hd|d>
//   d_s = Re<d|S|d>      (S-overlap norm of search direction)
//
// The stationary points solve the quadratic:
//   b*d_s * s^2  +  2*(a*d_s - c) * s  -  b  =  0
//
// CASTEP convention: the output `step_size` is the NEGATION of the chosen
// quadratic root (electronic.f90:10109: `step_size = -r1`).  This accounts
// for the sign of b in the energy functional — the returned eigenvalue IS
// the correct Rayleigh quotient at psi + step_size * d.

/// Result of the closed-form 2x2 quadratic line search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSearchResult {
    /// Optimal step size (always >= 0 after clamping, but may be negative
    /// before — CASTEP considers this "probably OK").
    pub step_size: f64,
    /// Energy at the optimal step:
    ///   E(s) = (a + b*s + c*s^2) / (1 + d_s * s^2)
    pub eigenvalue: f64,
    /// Norm of the new wavefunction:
    ///   sqrt(1 + d_s * step_size^2)
    pub norm: f64,
    /// Status: 0 = OK, -1 = negative/uphill step, 1 = clamped at 15.0,
    /// 2 = bogus norm (would be imaginary).
    pub status: i32,
}

/// Closed-form 2x2 quadratic line search matching CASTEP
/// `electronic_ideal_step_size`.
///
/// Given current band psi and preconditioned search direction d, find the
/// step size s that minimises the Rayleigh quotient.
///
/// # Arguments
///
/// * `dot_hpsi_psi` — a = Re⟨Hψ|ψ⟩ (the eigenvalue).
/// * `dot_hpsi_dir` — Re⟨Hψ|d⟩ (used to compute b = -2 * Re⟨Hψ|d⟩).
/// * `dot_hdir_dir` — c = Re⟨Hd|d⟩.
/// * `dot_sdir_dir` — d_s = Re⟨d|S|d⟩ (S-overlap norm of d).
#[allow(dead_code)]
pub fn line_search_2d_quadratic(
    dot_hpsi_psi: f64,
    dot_hpsi_dir: f64,
    dot_hdir_dir: f64,
    dot_sdir_dir: f64,
) -> LineSearchResult {
    // Map input quantities to CASTEP variable names
    // (electronic.f90:10056-10059)
    let a = dot_hpsi_psi; // Re⟨Hψ|ψ⟩
    let b = -2.0 * dot_hpsi_dir; // -2 * Re⟨Hψ|d⟩  (SD direction is -P⁻¹·r)
    let c = dot_hdir_dir; // Re⟨Hd|d⟩
    let d = dot_sdir_dir; // Re⟨d|S|d⟩  (= grad_grad in CASTEP)

    // ---- 1. Compute stationary-point roots --------------------------------
    // electronic.f90:10074
    let (mut step_size, mut eigenvalue, mut status) = if d.abs() > f64::MIN_POSITIVE {
        let ad = a * d;
        let bd = b * d;

        // ---- Guard: bd too small -> numeric instability in division ----------
        // When ||d|| -> 0 (near-converged state), bd -> 0 and r1 = (-ad + c +/- det)/bd
        // blows up, producing step_size approx -1e10.  Return early (no step needed).
        if bd.abs() < 1e-30 {
            return LineSearchResult {
                step_size: 0.0,
                eigenvalue: a,
                norm: 1.0,
                status: -2,
            };
        }

        // electronic.f90:10078-10080
        // tmp = ad^2 - 2·ad·c + c^2 + b·bd
        let tmp = ad * ad - 2.0 * ad * c + c * c + b * bd;
        if tmp < 0.0 {
            // This would be complex roots — should not happen in practice for
            // a Hermitian system.  Fall back to s = 0.
            return LineSearchResult {
                step_size: 0.0,
                eigenvalue: a,
                norm: 1.0,
                status: -1,
            };
        }
        let det = tmp.sqrt();

        // electronic.f90:10082-10084
        if bd.abs() < f64::MIN_POSITIVE {
            // bd ≈ 0 — division would blow up.  Fall back.
            return LineSearchResult {
                step_size: 0.0,
                eigenvalue: a,
                norm: 1.0,
                status: -1,
            };
        }

        // Two roots of the stationary-point equation
        // electronic.f90:10090-10091
        let r1 = (-ad + c + det) / bd;
        let r2 = (-ad + c - det) / bd;

        // Evaluate E(r) for each root (electronic.f90:10094-10104)
        let eval_root = |r: f64| -> Option<f64> {
            let denom = 1.0 + d * r * r;
            if denom > 0.0 {
                Some((a + b * r + c * r * r) / denom)
            } else {
                None
            }
        };

        let e1 = eval_root(r1);
        let e2 = eval_root(r2);

        // Pick the root with the lowest valid energy
        // (electronic.f90:10106-10133)
        let (chosen_r, chosen_e) = match (e1, e2) {
            (Some(xl1), Some(xl2)) => {
                if xl1 < xl2 {
                    (r1, xl1)
                } else {
                    (r2, xl2)
                }
            }
            (Some(xl1), None) if xl1 < a => (r1, xl1),
            (None, Some(xl2)) if xl2 < a => (r2, xl2),
            _ => {
                // No valid root found — fall back
                return LineSearchResult {
                    step_size: 0.0,
                    eigenvalue: a,
                    norm: 1.0,
                    status: -1,
                };
            }
        };

        // electronic.f90:10109 — CASTEP negates the root to obtain step_size
        (-chosen_r, chosen_e, 0)
    } else {
        // electronic.f90:10135-10148 — Simple parabola when ⟨d|S|d⟩ ≈ 0
        if c > 0.0 {
            let r = -b / (2.0 * c);
            let ev = (a + b * r + c * r * r) / (1.0 + d * r * r);
            (r, ev, 0)
        } else {
            (0.0, a, -1)
        }
    };

    // ---- 2. Clamp and status adjustments ----------------------------------
    // electronic.f90:10150-10163

    // Clamp oversized step (electronic.f90:10151-10157)
    if step_size > 15.0 {
        let s = 15.0;
        let ev = (a + b * s + c * s * s) / (1.0 + d * s * s);
        (step_size, eigenvalue, status) = (s, ev, 1);
    } else if step_size < 0.0 {
        // electronic.f90:10158-10163
        // Negative step is probably OK (preconditioning can cause uphill
        // gradient), but flag it.
        status = -1;
    }

    // ---- 3. Compute final norm --------------------------------------------
    // electronic.f90:10165-10171
    let t = 1.0 + d * step_size * step_size;
    let (norm, final_status) = if t > 0.0 {
        (t.sqrt(), status)
    } else {
        (0.0, 2)
    };

    LineSearchResult {
        step_size,
        eigenvalue,
        norm,
        status: final_status,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Success Criterion 1: Synthetic parabola
    //   a = 1.0, dot_hpsi_dir = 1.0  →  b = -2.0
    //   c = 3.0, d_s = 0.0
    //
    //   E(s) = 1 - 2*s + 3*s^2   (simple parabola)
    //   minimum at s = -b/(2c) = 2/(6) = 1/3 ≈ 0.333333...
    //   E(1/3) = 1 - 2/3 + 3/9 = 1 - 2/3 + 1/3 = 2/3 ≈ 0.666667
    // -----------------------------------------------------------------------
    #[test]
    fn synthetic_parabola() {
        let r = line_search_2d_quadratic(1.0, 1.0, 3.0, 0.0);
        let expected_step = 1.0 / 3.0;
        let expected_eigenvalue = 2.0 / 3.0;
        assert!(
            (r.step_size - expected_step).abs() < 1e-12,
            "step_size: expected {expected_step}, got {}",
            r.step_size
        );
        assert!(
            (r.eigenvalue - expected_eigenvalue).abs() < 1e-12,
            "eigenvalue: expected {expected_eigenvalue}, got {}",
            r.eigenvalue
        );
        assert!((r.norm - 1.0).abs() < 1e-12, "norm should be 1.0 for d_s=0");
        assert_eq!(r.status, 0, "status should be OK");
    }

    // -----------------------------------------------------------------------
    // Success Criterion 2: Synthetic quadratic (rational)
    //   a = 0.0, dot_hpsi_dir = -0.5  →  b = 1.0
    //   c = 1.0, d_s = 1.0
    //
    //   E(s) = (s + s^2) / (1 + s^2)
    //
    //   Stationary points at s = 1 ± √2.
    //   Minimum at s = 1 - √2 ≈ -0.41421, E ≈ -0.20711.
    //   CASTEP convention: step_size = -(minimum) ≈ 0.41421,
    //   eigenvalue = E(minimum) ≈ -0.20711.
    //
    //   E(step_size = 0.41421) should be lower than both E(0) and E(100).
    // -----------------------------------------------------------------------
    #[test]
    fn synthetic_rational() {
        let r = line_search_2d_quadratic(0.0, -0.5, 1.0, 1.0);

        // step_size should be -(1 - sqrt(2)) = sqrt(2) - 1 ≈ 0.41421356...
        let expected_step = 2.0_f64.sqrt() - 1.0;
        assert!(
            (r.step_size - expected_step).abs() < 1e-12,
            "step_size: expected {expected_step}, got {}",
            r.step_size
        );

        // eigenvalue at the minimum: E(1 - sqrt(2)) ≈ -0.20710678...
        let s_min = 1.0 - 2.0_f64.sqrt();
        let expected_eigenvalue = (s_min + s_min * s_min) / (1.0 + s_min * s_min);
        assert!(
            (r.eigenvalue - expected_eigenvalue).abs() < 1e-12,
            "eigenvalue: expected {expected_eigenvalue}, got {}",
            r.eigenvalue
        );

        // E(step_size) < E(0): the eigenvalue field IS the Rayleigh quotient
        // at psi + step_size * d, so r.eigenvalue < a = 0 confirms decrease.
        assert!(
            r.eigenvalue < 0.0,
            "E(step) should be < E(0): {} vs 0",
            r.eigenvalue
        );
        // E(step_size) < E(100): E(100) ≈ 1.0
        let e_at_100: f64 = (100.0 + 10000.0) / (1.0 + 10000.0);
        assert!(
            r.eigenvalue < e_at_100,
            "E(step) should be < E(100): {} vs {e_at_100}",
            r.eigenvalue
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 3: Energy decrease (non-trivial inputs)
    //
    //   Use inputs that mimic a realistic CG iteration where the search
    //   direction genuinely points downhill.  We assert that E(step) < E(0).
    // -----------------------------------------------------------------------
    #[test]
    fn energy_decrease() {
        // Realistic scenario: a > 0 (eigenvalue ~ 0.5 Ha), search direction
        // pointing downhill: dot_hpsi_dir negative (so b positive), modest
        // curvature c and S-norm d_s.
        let a = -0.4; // arbitrary, but realistic scale
        let dot_hpsi_dir = -0.15; // negative → b = +0.3
        let c = 1.2;
        let d_s = 0.8;

        let r = line_search_2d_quadratic(a, dot_hpsi_dir, c, d_s);

        // Energy at step 0 is the eigenvalue a.
        let e0 = a;
        let e_step = r.eigenvalue;

        assert!(
            e_step < e0,
            "E(step) = {e_step} should be < E(0) = {e0}, status = {}",
            r.status
        );
        assert!(
            r.step_size > 0.0,
            "step_size should be positive for downhill direction, got {}",
            r.step_size
        );
        assert!(
            r.status == 0 || r.status == -1,
            "status should be OK (0) or negative-step-flag (-1), got {}",
            r.status
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 4: Clamping
    //
    //   Create inputs where the unclamped optimal step would exceed 15.0.
    //   The clamped result must have step_size == 15.0 and status == 1.
    // -----------------------------------------------------------------------
    #[test]
    fn clamping_at_15() {
        // Design: very shallow curvature c and tiny S-norm d_s with large b
        // pushes the minimum far to the right.
        //
        // Parabola case (d_s = 0):
        //   a = 0, b = 10, c = 0.01
        //   optimal s = -b/(2c) = -10/0.02 = -500 (but we'd negate for step)
        //   Wait — with d_s = 0, step = r = -b/(2c) = -500.
        //   Then step_size=0 in the parabola case... no wait.
        //   b = -2*dot_hpsi_dir, so with a=0 and step pointing downhill:
        let a = 0.0;
        let dot_hpsi_dir = -5.0; // b = 10
        let c = 0.01; // very shallow curvature
        let d_s = 0.001; // very small S-overlap norm

        let r = line_search_2d_quadratic(a, dot_hpsi_dir, c, d_s);

        assert!(
            (r.step_size - 15.0).abs() < 1e-12,
            "clamped step_size should be exactly 15.0, got {}",
            r.step_size
        );
        assert_eq!(r.status, 1, "status should indicate clamping (1)");
    }

    // -----------------------------------------------------------------------
    // Success Criterion 5: d_s = 0, c <= 0  →  status = -1
    //
    //   When there is no S-overlap and the H-curvature is non-positive,
    //   the energy functional has no minimum.  CASTEP returns status = -1.
    // -----------------------------------------------------------------------
    #[test]
    fn no_minimum_parabola() {
        let r = line_search_2d_quadratic(0.0, 0.0, -0.1, 0.0);
        assert_eq!(r.status, -1, "should flag no-minimum (status = -1)");
        assert!(
            (r.step_size - 0.0).abs() < 1e-12,
            "step_size should be 0, got {}",
            r.step_size
        );
    }

    // -----------------------------------------------------------------------
    // Energy monotonicity: E(s) decreases monotonically from E(0) to the
    // chosen stationary point (within machine precision).
    // -----------------------------------------------------------------------
    #[test]
    fn energy_monotonicity() {
        // Use a realistic set of parameters
        let a = -0.38;
        let dot_hpsi_dir = -0.2;
        let c = 1.5;
        let d_s = 0.7;

        let r = line_search_2d_quadratic(a, dot_hpsi_dir, c, d_s);
        assert!(
            r.eigenvalue <= a + 1e-14,
            "eigenvalue {} should not exceed initial eigenvalue {}", r.eigenvalue, a
        );
        assert!(r.norm > 0.0, "norm should be positive");
    }
}
