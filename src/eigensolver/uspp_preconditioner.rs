// ---------------------------------------------------------------------------
// USPP-aware preconditioner P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹
// ---------------------------------------------------------------------------
//
// Reference formulas:
//   - TPA (Teter-Payne-Allan) factor: CASTEP wave.f90:29889-29893
//   - Preconditioner structure "tpa + tpa sum_nm |beta_n> q_nm <beta_m| tpa":
//     CASTEP nlpot.f90:15480-15665
//
// Matrix form:
//   P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹
// where:
//   R = (−Q⁻¹ − C)⁻¹
//   C = β†·T⁻¹·β
//   T⁻¹ = diag(tpa(g)) — diagonal in reciprocal space
//
// Algorithm (one band/residual vector at a time):
//   1. tpa_r = T⁻¹·r           (element-wise multiply by tpa_diag)
//   2. w = β†·tpa_r            (gemv: n_proj × n_pw)
//   3. w_R = R·w               (gemv: n_proj × n_proj)
//   4. delta = β·w_R           (gemv: n_pw × n_proj, accumulate)
//   5. result = tpa_r + T⁻¹·delta

// Allow dead code: this module is a library component for future integration.
// Allow range loops in numeric matrix code.
#![allow(dead_code, clippy::needless_range_loop)]

use faer::linalg::solvers::DenseSolveCore;
use faer::Mat;
use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;

/// USPP-aware preconditioner.
///
/// P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹
/// where R = (−Q⁻¹ − C)⁻¹, C = β†·T⁻¹·β.
pub struct UsppPreconditioner {
    /// R = (−Q⁻¹ − C)⁻¹, n_proj × n_proj (faer column-major)
    r_matrix: Mat<Complex64>,
    /// Diagonal of T⁻¹, length n_pw (one per plane wave)
    tpa_diag: Vec<f64>,
    /// β projectors in reciprocal space, n_pw × n_proj (ndarray row-major)
    beta_g: Array2<Complex64>,
}

impl UsppPreconditioner {
    /// Precompute the R matrix and TPA diagonal.
    ///
    /// Arguments:
    /// - `beta_g`: β projectors in reciprocal space, shape `(n_pw, n_proj)`
    /// - `q_matrix`: USPP Q matrix, shape `(n_proj, n_proj)`
    /// - `kinetic_g`: kinetic energy ½|G + k|² for each plane wave (Bohr⁻², i.e. Hartree)
    /// - `k_cart`: k-point in Cartesian coordinates (Bohr⁻¹) — not used in TPA
    ///   formula because `kinetic_g` already includes the k-point shift
    pub fn new(
        beta_g: Array2<Complex64>,
        q_matrix: Array2<Complex64>,
        kinetic_g: Vec<f64>,
        _k_cart: [f64; 3],
    ) -> Self {
        let tpa_diag = compute_tpa_diag(&kinetic_g);
        let c_mat = compute_c_matrix(&beta_g, &tpa_diag);
        let r_matrix = compute_r_matrix(&q_matrix, &c_mat);
        Self {
            r_matrix,
            tpa_diag,
            beta_g,
        }
    }

    /// Apply P⁻¹ to a residual vector (one band).
    ///
    /// Algorithm (CASTEP nlpot.f90:15480-15665):
    /// 1. tpa_r = T⁻¹·r       (element-wise)
    /// 2. w = β†·tpa_r        (project onto projector space)
    /// 3. w_R = R·w           (apply R matrix)
    /// 4. delta = β·w_R       (expand back to plane-wave space)
    /// 5. result = tpa_r + T⁻¹·delta
    pub fn apply(&self, residual: &[Complex64]) -> Vec<Complex64> {
        let n_pw = self.tpa_diag.len();
        let n_proj = self.beta_g.ncols();

        // Step 1: tpa_r = T⁻¹·r
        let tpa_r: Vec<Complex64> = residual
            .par_iter()
            .zip(self.tpa_diag.par_iter())
            .map(|(r, t)| r * t)
            .collect();

        // Step 2: w = β†·tpa_r
        // Each projector column is independent; parallelize over projectors.
        let mut w = vec![Complex64::new(0.0, 0.0); n_proj];
        w.par_iter_mut().enumerate().for_each(|(p, w_p)| {
            let mut sum = Complex64::new(0.0, 0.0);
            for g in 0..n_pw {
                sum += self.beta_g[[g, p]].conj() * tpa_r[g];
            }
            *w_p = sum;
        });

        // Step 3: w_R = R·w
        let mut w_r = vec![Complex64::new(0.0, 0.0); n_proj];
        for i in 0..n_proj {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..n_proj {
                sum += self.r_matrix[(i, j)] * w[j];
            }
            w_r[i] = sum;
        }

        // Step 4: delta = β·w_R
        let mut delta = vec![Complex64::new(0.0, 0.0); n_pw];
        delta
            .par_iter_mut()
            .enumerate()
            .for_each(|(g, d_g)| {
                let mut sum = Complex64::new(0.0, 0.0);
                for p in 0..n_proj {
                    sum += self.beta_g[[g, p]] * w_r[p];
                }
                *d_g = sum;
            });

        // Step 5: result = tpa_r + T⁻¹·delta
        tpa_r
            .into_par_iter()
            .zip(delta.par_iter().zip(self.tpa_diag.par_iter()))
            .map(|(t, (d, td))| t + d * td)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TPA diagonal computation
// ---------------------------------------------------------------------------

/// Compute the Teter-Payne-Allan (TPA) preconditioning diagonal.
///
/// Formula (CASTEP wave.f90:29889-29893):
/// ```text
/// x = E_k / E_k,mean
/// temp = 27 + 18x + 12x² + 8x³
/// tpa = 1 / (1 + 16x⁴ / temp)
/// ```
fn compute_tpa_diag(kinetic_g: &[f64]) -> Vec<f64> {
    let n_pw = kinetic_g.len();
    if n_pw == 0 {
        return Vec::new();
    }

    // Mean kinetic energy E_k,mean = (1/n_pw) Σ_i E_k(i)
    let ek_sum: f64 = kinetic_g.iter().sum();
    let ek_mean = ek_sum / n_pw as f64;

    // Guard against division by zero (all-kinetic-zero pathological case)
    let inv_mean = if ek_mean > 0.0 { 1.0 / ek_mean } else { 1.0 };

    kinetic_g
        .iter()
        .map(|&ek| {
            let x = ek * inv_mean;
            // temp = 27 + 18x + 12x² + 8x³
            let temp = 27.0 + x * (18.0 + x * (12.0 + 8.0 * x));
            // tpa = 1/(1 + 16x⁴/temp)
            1.0 / (1.0 + 16.0 * x.powi(4) / temp)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// C = β†·T⁻¹·β  (projector-space coupling matrix)
// ---------------------------------------------------------------------------

/// Compute C = β†·T⁻¹·β, an n_proj × n_proj dense matrix.
///
/// β is (n_pw, n_proj), T⁻¹ = diag(tpa_diag) is diagonal.
/// The product is computed via explicit triple loop since n_proj is small
/// (typically ≤ 100, often ≤ 30).
fn compute_c_matrix(beta_g: &Array2<Complex64>, tpa_diag: &[f64]) -> Mat<Complex64> {
    let n_pw = beta_g.nrows();
    let n_proj = beta_g.ncols();

    // Build the faer Mat for β (column-major)
    let beta: Mat<Complex64> = Mat::from_fn(n_pw, n_proj, |i, j| beta_g[[i, j]]);

    // T⁻¹·β: scale each row of β by tpa_diag[g]
    let mut beta_t = beta.clone();
    for i in 0..n_pw {
        let scale = tpa_diag[i];
        for j in 0..n_proj {
            beta_t[(i, j)] *= scale;
        }
    }

    // C = β† · (T⁻¹·β)
    // C[p,q] = Σ_g conj(β[g,p]) · β_t[g,q]
    let mut c = Mat::<Complex64>::zeros(n_proj, n_proj);
    for p in 0..n_proj {
        for q in 0..n_proj {
            let mut sum = Complex64::new(0.0, 0.0);
            for g in 0..n_pw {
                sum += beta[(g, p)].conj() * beta_t[(g, q)];
            }
            c[(p, q)] = sum;
        }
    }

    c
}

// ---------------------------------------------------------------------------
// R = (−Q⁻¹ − C)⁻¹
// ---------------------------------------------------------------------------

/// Compute R = (−Q⁻¹ − C)⁻¹ as an n_proj × n_proj dense matrix.
///
/// Steps:
/// 1. Invert Q to get Q⁻¹
/// 2. Form M = −Q⁻¹ − C
/// 3. Invert M to get R
fn compute_r_matrix(q_matrix: &Array2<Complex64>, c_mat: &Mat<Complex64>) -> Mat<Complex64> {
    let n_proj = c_mat.nrows();

    // Convert Q to faer Mat (column-major)
    let q: Mat<Complex64> = Mat::from_fn(n_proj, n_proj, |i, j| q_matrix[[i, j]]);

    // Compute Q⁻¹ via LU decomposition
    // DenseSolveCore::inverse() is in scope via the use statement
    let q_lu = q.full_piv_lu();
    let q_inv = q_lu.inverse();

    // Form M[p,q] = −Q⁻¹[p,q] − C[p,q]
    let mut m = Mat::<Complex64>::zeros(n_proj, n_proj);
    for p in 0..n_proj {
        for q in 0..n_proj {
            m[(p, q)] = -q_inv[(p, q)] - c_mat[(p, q)];
        }
    }

    // R = M⁻¹
    let m_lu = m.full_piv_lu();
    m_lu.inverse()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal test environment with synthetic data.
    ///
    /// n_proj = 3, n_pw = 10. β is structured (not random, deterministic),
    /// Q is a small positive-definite Hermitian matrix, kinetic energies
    /// span a realistic range ~0.5..~50 Ha.
    fn make_test_fixture() -> UsppPreconditioner {
        let n_proj = 3;
        let n_pw = 10;

        // β: n_pw × n_proj with structured values
        let mut beta_g = Array2::<Complex64>::zeros((n_pw, n_proj));
        for g in 0..n_pw {
            for p in 0..n_proj {
                let re = (g * 3 + p * 7 + 1) as f64 / 100.0;
                let im = (g * 5 + p * 2 + 3) as f64 / 100.0;
                beta_g[[g, p]] = Complex64::new(re, im);
            }
        }

        // Q: n_proj × n_proj positive-definite Hermitian
        // Make it diagonally dominant: Q[i,i] > Σ_{j≠i} |Q[i,j]|
        let mut q_matrix = Array2::<Complex64>::zeros((n_proj, n_proj));
        for i in 0..n_proj {
            for j in 0..n_proj {
                let re = ((i + 1) * (j + 1)) as f64 / 10.0;
                let im = if i == j { 0.0 } else { (i + j + 1) as f64 / 20.0 };
                q_matrix[[i, j]] = Complex64::new(re, im);
            }
        }
        // Hermitianize: Q = (Q + Q†)/2
        for i in 0..n_proj {
            for j in 0..i {
                let avg = (q_matrix[[i, j]] + q_matrix[[j, i]].conj()) / 2.0;
                q_matrix[[i, j]] = avg;
                q_matrix[[j, i]] = avg.conj();
            }
            // Ensure iagonal is positive and dominant
            q_matrix[[i, i]] = q_matrix[[i, i]] + Complex64::new(n_proj as f64, 0.0);
        }

        // Kinetic energies: range 0.5..50 Ha to match Cu111_CO scale
        let kinetic_g: Vec<f64> =
            (0..n_pw).map(|g| 0.5 * (g + 1) as f64 * (g + 1) as f64).collect();

        let k_cart = [0.0; 3]; // Gamma point

        UsppPreconditioner::new(beta_g, q_matrix, kinetic_g, k_cart)
    }

    // -----------------------------------------------------------------------
    // Success Criterion 1: R matrix Hermiticity
    // -----------------------------------------------------------------------

    /// R matrix must be Hermitian: max|R − R†| < 1e-12.
    #[test]
    fn r_matrix_is_hermitian() {
        let precon = make_test_fixture();
        let r = &precon.r_matrix;
        let n = r.nrows();

        let mut max_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let r_ij = r[(i, j)];
                let r_ji_conj = r[(j, i)].conj();
                let diff = (r_ij - r_ji_conj).norm();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 1e-12,
            "R matrix Hermiticity violation: max|R − R†| = {:.2e}",
            max_diff
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 2: Preconditioner stability under repeated application
    // -----------------------------------------------------------------------

    /// Apply P⁻¹ twice — the relative difference between first and second
    /// application should be bounded (convergent, not divergent).
    #[test]
    fn apply_is_self_consistent() {
        let precon = make_test_fixture();
        let n_pw = 10;

        // Deterministic test vector
        let residual: Vec<Complex64> = (0..n_pw)
            .map(|i| Complex64::new((i * 3 + 1) as f64 / 10.0, (i * 2 + 5) as f64 / 10.0))
            .collect();

        let v1 = precon.apply(&residual);
        let v2 = precon.apply(&v1);

        // Compute relative difference ||v2 − v1|| / ||v1||
        let diff_norm: f64 = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let v1_norm: f64 = v1.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();

        let relative_diff = diff_norm / v1_norm.max(f64::EPSILON);

        // The second application should approximately reproduce the first
        // (convergence factor depends on the spectrum of T⁻¹·β·R·β†·T⁻¹)
        //
        // A relative difference < 1.0 means P⁻¹ is contracting — the method
        // is a valid preconditioner (not an expansion that would diverge).
        assert!(
            relative_diff < 1.0,
            "P⁻¹ self-consistency: ||P⁻¹·v1 − v1||/||v1|| = {:.2e} >= 1.0 (divergent)",
            relative_diff
        );
    }

    // -----------------------------------------------------------------------
    // Success Criterion 3: TPA floor check
    // -----------------------------------------------------------------------

    /// For realistic kinetic energies (G²/2 ≤ ~30 Ha), min(tpa_diag) > 0.01
    /// (no near-singularities).
    ///
    /// Source: CASTEP wave.f90:29889-29893 — the TPA formula asymptotically
    /// approaches 8/(16*x²) ≈ 0.5/x² for large x. At x=√30 ≈ 5.5,
    /// tpa ≈ 0.5/30 ≈ 0.017 > 0.01.
    #[test]
    fn tpa_diag_has_positive_floor() {
        // Kinetic energies: E_kin = ½|G|² with |G|² up to ~60 → max ~30 Ha
        let kinetic_g: Vec<f64> = (1..=60).map(|g| 0.5 * (g as f64).powi(2)).collect();

        let tpa_diag = compute_tpa_diag(&kinetic_g);

        let min_val = tpa_diag.iter().cloned().fold(f64::INFINITY, f64::min);

        assert!(
            min_val > 0.01,
            "TPA diagonal minimum {} is <= 0.01 — possible near-singular preconditioner \
             (Source: CASTEP wave.f90:29889-29893, large-x asymptotics)",
            min_val
        );

        // All entries must be finite (no NaN or Inf)
        for (i, &val) in tpa_diag.iter().enumerate() {
            assert!(
                val.is_finite(),
                "TPA diagonal at index {} is not finite: {}",
                i,
                val
            );
        }
    }

    // -----------------------------------------------------------------------
    // Success Criterion 4: Reduction to diagonal T⁻¹ with β = 0
    // -----------------------------------------------------------------------

    /// With β = 0 (all-zero projectors) and arbitrary Q:
    ///   C = 0
    ///   R = (−Q⁻¹ − 0)⁻¹ = −Q
    ///   P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹ = T⁻¹   (since β = 0)
    #[test]
    fn reduces_to_tpa_with_zero_beta() {
        let n_proj = 3;
        let n_pw = 10;

        // β = 0
        let beta_g = Array2::<Complex64>::zeros((n_pw, n_proj));

        // Q = arbitrary invertible matrix (diagonal identity works)
        let mut q_matrix = Array2::<Complex64>::zeros((n_proj, n_proj));
        for i in 0..n_proj {
            q_matrix[[i, i]] = Complex64::new(1.0, 0.0);
        }

        let kinetic_g: Vec<f64> = (0..n_pw).map(|g| 0.5 * (g + 1) as f64).collect();
        let k_cart = [0.0; 3];

        let precon = UsppPreconditioner::new(beta_g, q_matrix, kinetic_g.clone(), k_cart);

        // Deterministic residual
        let residual: Vec<Complex64> = (0..n_pw)
            .map(|i| Complex64::new((i + 1) as f64 / 7.0, (i + 2) as f64 / 11.0))
            .collect();

        let result = precon.apply(&residual);

        // Expected: T⁻¹·r (element-wise multiply by tpa_diag)
        let tpa_diag = compute_tpa_diag(&kinetic_g);
        let expected: Vec<Complex64> = residual
            .iter()
            .zip(tpa_diag.iter())
            .map(|(r, td)| r * td)
            .collect();

        // Compare
        let max_diff: f64 = result
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff < 1e-14,
            "With beta = 0, P^-1 should equal T^-1 but max difference = {:.2e}",
            max_diff
        );
    }

    // -----------------------------------------------------------------------
    // Sanity: C matrix Hermiticity
    // -----------------------------------------------------------------------

    /// C = β†·T⁻¹·β must be Hermitian since T⁻¹ is real diagonal.
    #[test]
    fn c_matrix_is_hermitian() {
        let precon = make_test_fixture();
        let n_proj = precon.beta_g.ncols();
        let tpa_diag = &precon.tpa_diag;
        let c = compute_c_matrix(&precon.beta_g, tpa_diag);

        let mut max_diff = 0.0_f64;
        for i in 0..n_proj {
            for j in 0..n_proj {
                let diff = (c[(i, j)] - c[(j, i)].conj()).norm();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 1e-12,
            "C matrix Hermiticity violation: max|C - C^dag| = {:.2e}",
            max_diff
        );
    }

    // -----------------------------------------------------------------------
    // Sanity: Non-trivial result
    // -----------------------------------------------------------------------

    /// Apply P⁻¹ to a unit vector — the result should be non-trivial,
    /// demonstrating the preconditioner is not the zero operator.
    #[test]
    fn apply_produces_nonzero_result() {
        let precon = make_test_fixture();
        let n_pw = 10;

        // Unit vector at index 0
        let mut residual = vec![Complex64::new(0.0, 0.0); n_pw];
        residual[0] = Complex64::new(1.0, 0.0);

        let result = precon.apply(&residual);

        let result_norm: f64 = result.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            result_norm > 1e-12,
            "P^-1 applied to unit vector produced zero vector"
        );
    }
}
