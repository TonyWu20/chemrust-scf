// ---------------------------------------------------------------------------
// Integration tests for TPA preconditioner vector R(G) construction
// ---------------------------------------------------------------------------
//
// Tests outcome C1-C6 from Group C of the phase-6 plan:
//
//   C1: tpa(0.0) == 1.0                  (no preconditioning at zero kinetic energy)
//   C2: tpa(1.0) ≈ 0.80247               (source: formula evaluation)
//   C3: tpa(10.0) ≈ 0.0555               (source: formula evaluation)
//   C4: R(G)_max = 1.0 for the lowest kinetic energy PW
//   C5: R(G) length = n_pw
//   C6: All R(G) values in [0, 1]
//
// The discriminator: the OLD wrong formula `1/(T−λ)` gives values >1 and can
// diverge to infinity when T≈λ. The correct tpa() always gives values in
// [0, 1]. Any test that asserts `tpa(x) ∈ [0, 1]` for all x ≥ 0 will catch
// the wrong formula.
// ---------------------------------------------------------------------------

use chemrust_scf::tpa;
use chemrust_scf::compute_c_matrix;
use chemrust_scf::invert_q_matrix;
use chemrust_scf::assemble_r_beta;
use chemrust_scf::assemble_q_rcq;
use cudarc::cufft::sys::double2 as CudaComplex;
use ndarray::Array2;
use num_complex::Complex64;

mod davidson_precon_build {
    use super::{
        tpa, compute_c_matrix, invert_q_matrix, assemble_r_beta, assemble_q_rcq, CudaComplex,
        Array2, Complex64,
    };

    // -----------------------------------------------------------------------
    // C1–C3: tpa() formula-value checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_tpa_formula_values() {
        // C1: tpa(0) == 1 — no preconditioning at zero kinetic energy
        let tpa_0 = tpa(0.0);
        assert!(
            (tpa_0 - 1.0).abs() < 1e-15,
            "C1 FAIL: tpa(0) should be exactly 1.0, got {}",
            tpa_0,
        );

        // C2: tpa(1.0) ≈ 0.80247
        //     Formula: denominator = 27 + 1*(18 + 1*(12 + 8*1)) = 27 + 18 + 12 + 8 = 65
        //              numerator = 16*1^4 = 16
        //              tpa = 1 / (1 + 16/65) = 65/81 ≈ 0.8024691358...
        let tpa_1 = tpa(1.0);
        let expected_1 = 65.0 / 81.0; // exactly 65/81
        assert!(
            (tpa_1 - expected_1).abs() < 1e-14,
            "C2 FAIL: tpa(1) should be 65/81 ≈ {:.10}, got {:.10}",
            expected_1,
            tpa_1,
        );

        // C3: tpa(10.0) is small ≈ 0.0555
        //     Formula: denominator = 27 + 10*(18 + 10*(12 + 8*10))
        //                          = 27 + 10*(18 + 10*(12 + 80))
        //                          = 27 + 10*(18 + 920)
        //                          = 27 + 9380 = 9407
        //              numerator = 16*10^4 = 160000
        //              tpa = 1 / (1 + 160000/9407) ≈ 0.05552...
        let tpa_10 = tpa(10.0);
        let expected_10 = 1.0 / (1.0 + 160_000.0 / 9407.0);
        assert!(
            (tpa_10 - expected_10).abs() < 1e-14,
            "C3 FAIL: tpa(10) should be {:.10}, got {:.10}",
            expected_10,
            tpa_10,
        );
    }

    #[test]
    fn test_tpa_monotonic_decreasing() {
        // tpa(x) should be monotonically decreasing for x >= 0
        let xs = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1e6];
        for i in 1..xs.len() {
            let prev = tpa(xs[i - 1]);
            let curr = tpa(xs[i]);
            assert!(
                curr <= prev,
                "tpa is not monotonic decreasing at x={}: prev={}, curr={}",
                xs[i],
                prev,
                curr,
            );
        }
    }

    #[test]
    fn test_tpa_bounded_by_zero_and_one() {
        // For any x >= 0, tpa(x) must be in [0, 1]
        for x in [0.0, 0.001, 0.1, 1.0, 10.0, 100.0, 1e6, 1e12] {
            let val = tpa(x);
            assert!(
                val >= 0.0 && val <= 1.0,
                "tpa({}) = {} is not in [0, 1]",
                x,
                val,
            );
        }
    }

    #[test]
    fn test_tpa_asymptotic_zero() {
        // tpa(x) → 0 as x → ∞
        let tpa_large = tpa(1e12);
        assert!(
            tpa_large < 1e-10,
            "tpa(1e12) should approach 0, got {}",
            tpa_large,
        );
        // Numerical: x = 1e12, x^4 = 1e48, numerator = 16e48,
        // denominator ≈ 8*x^2 = 8e24, ratio ≈ 2e24, tpa ≈ 5e-25
        // So < 1e-10 is very generous.
    }

    // -----------------------------------------------------------------------
    // C4–C6: R(G) vector property checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_r_vector_properties() {
        // Create synthetic kinetic energies (Hartree) spanning a realistic range
        let kinetic_energies: Vec<f64> = vec![
            0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0,
        ];
        let n_pw = kinetic_energies.len();
        let mean_ek: f64 = kinetic_energies.iter().sum::<f64>() / n_pw as f64;

        // Compute R(G) vector: R(G) = tpa(pw_ek / mean_ek)
        let r_values: Vec<f64> = kinetic_energies
            .iter()
            .map(|&ek| tpa(ek / mean_ek))
            .collect();

        // C5: R(G) length = n_pw
        assert_eq!(
            r_values.len(),
            n_pw,
            "C5 FAIL: R(G) length {} does not match n_pw {}",
            r_values.len(),
            n_pw,
        );

        // C6: All R(G) values in [0, 1]
        for (i, &r) in r_values.iter().enumerate() {
            assert!(
                r >= 0.0 && r <= 1.0,
                "C6 FAIL: R(G)[{}] = {} is not in [0, 1]",
                i,
                r,
            );
        }

        // C4: R(G)_max = 1.0
        //     tpa(0) = 1.0 at the lowest kinetic energy PW (index 0).
        //     Since tpa(x) ≤ 1 for all x ≥ 0, the global maximum is 1.0.
        let min_ek = kinetic_energies[0];
        assert_eq!(
            min_ek, 0.0,
            "Expected first kinetic energy to be 0.0 for C4 check",
        );

        assert!(
            (r_values[0] - 1.0).abs() < 1e-15,
            "C4 FAIL: R(G) at minimum kinetic energy should be 1.0, got {}",
            r_values[0],
        );

        let max_r = r_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_r - 1.0).abs() < 1e-15,
            "C4 FAIL: max R(G) should be 1.0, got {}",
            max_r,
        );

        // Discriminator: verify ALL R(G) values are ≤ 1 (the old `1/(T-λ)` formula
        // would produce values > 1 for small kinetic energies).
        assert!(
            r_values.iter().all(|&r| r <= 1.0),
            "C4/C6 discriminator FAIL: found R(G) > 1.0 — old wrong formula detected",
        );
    }

    // -----------------------------------------------------------------------
    // C = β^H · diag(R) · β — Hermitian property and cross-path verification
    // -----------------------------------------------------------------------

    /// C1: C[n,m] is Hermitian: max|C[n,m] − conj(C[m,n])| < 1×10⁻¹²
    ///
    /// Anchored to the mathematical property β^H·diag(R)·β = (β^H·diag(R)·β)^H.
    /// A buggy implementation (wrong indexing, missing conj, sign error) would
    /// produce a non-Hermitian result.
    #[test]
    fn test_c_matrix_hermitian() {
        let ne = 3;
        let n_pw = 10;

        // Synthetic beta_g: (ne × n_pw) row-major with structured values
        // so that random cancellation does not mask indexing bugs.
        let beta_g: Vec<CudaComplex> = (0..ne * n_pw)
            .map(|i| CudaComplex {
                x: (i as f64) * 0.5,
                y: (i as f64) * 0.3 + 1.0,
            })
            .collect();

        // R(G) monotonically decreasing — physical range
        let r_vector: Vec<f64> = (0..n_pw).map(|g| 1.0 / (g as f64 + 1.0)).collect();

        let c = compute_c_matrix(&beta_g, &r_vector, ne, n_pw);

        // Verify Hermitian: C[n,m] ≈ conj(C[m,n]) for all n,m
        for n in 0..ne {
            for m in 0..ne {
                let forward = c[[n, m]];
                let backward = c[[m, n]].conj();
                let diff = (forward - backward).norm();
                assert!(
                    diff < 1e-12,
                    "C[{n},{m}] = ({re:.6e},{im:.6e}) not conj(C[{m},{n}]) = \
                     ({re2:.6e},{im2:.6e}), diff {diff:.6e}",
                    n = n,
                    m = m,
                    re = forward.re,
                    im = forward.im,
                    re2 = backward.re,
                    im2 = backward.im,
                );
            }
        }
    }

    /// C2: C = β^H·diag(R)·β matches explicit triple-loop reference.
    ///
    /// Cross-path verification: the production path uses the Gram-matrix form
    /// `α·α^H` with α[n,G] = conj(β[n,G])·sqrt(R[G]) via ndarray `.dot()`.
    /// The reference path uses the explicit triple sum
    /// Σ_G conj(β[n,G])·β[m,G]·R[G] with plain for-loops.
    ///
    /// These are structurally independent: different indexing, different
    /// intermediate data layout, different arithmetic sequence.
    #[test]
    fn test_c_matrix_against_explicit() {
        let ne = 5;
        let n_pw = 12;

        // Synthetic beta with varied complex values to exercise both real and
        // imaginary paths.
        let beta_g: Vec<CudaComplex> = (0..ne * n_pw)
            .map(|i| {
                let n = i / n_pw;
                let g = i % n_pw;
                CudaComplex {
                    x: (n as f64) * 0.7 - (g as f64) * 0.2,
                    y: (n as f64) * 0.1 + (g as f64) * 0.5,
                }
            })
            .collect();

        let r_vector: Vec<f64> = (0..n_pw).map(|g| 0.5_f64.powi(g as i32)).collect();

        // Production path (Gram matrix via ndarray)
        let c = compute_c_matrix(&beta_g, &r_vector, ne, n_pw);

        // Reference path: explicit triple-loop
        let mut c_ref = Array2::<Complex64>::zeros((ne, ne));
        for n in 0..ne {
            for m in 0..ne {
                let mut sum = Complex64::new(0.0, 0.0);
                for g in 0..n_pw {
                    let bn = Complex64::new(
                        beta_g[n * n_pw + g].x,
                        beta_g[n * n_pw + g].y,
                    );
                    let bm = Complex64::new(
                        beta_g[m * n_pw + g].x,
                        beta_g[m * n_pw + g].y,
                    );
                    sum += bn.conj() * bm * r_vector[g];
                }
                c_ref[[n, m]] = sum;
            }
        }

        // Assert agreement to machine precision
        for n in 0..ne {
            for m in 0..ne {
                let diff = (c[[n, m]] - c_ref[[n, m]]).norm();
                assert!(
                    diff < 1e-12,
                    "C[{n},{m}] mismatch: optimized = ({re:.10e},{im:.10e}), \
                     explicit = ({re2:.10e},{im2:.10e}), diff = {diff:.6e}",
                    re = c[[n, m]].re,
                    im = c[[n, m]].im,
                    re2 = c_ref[[n, m]].re,
                    im2 = c_ref[[n, m]].im,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // C1-C3: Q^-1 inversion properties
    // -----------------------------------------------------------------------

    /// C1: Q^-1 . Q - I < 1x10^-12 (matrix inverse identity)
    ///
    /// Anchored to the mathematical identity Q . Q^-1 = I. Uses a strictly
    /// diagonally dominant symmetric positive-definite matrix so that
    /// Cholesky is guaranteed to succeed.
    #[test]
    fn test_q_inverse_identity() {
        let n = 5;
        // SPD matrix: strictly diagonally dominant with positive diagonal
        let q: Vec<f64> = vec![
            4.0, 1.0, 0.0, 0.5, 0.2, //
            1.0, 5.0, 2.0, 0.0, 0.3, //
            0.0, 2.0, 6.0, 1.0, 0.0, //
            0.5, 0.0, 1.0, 4.0, 0.5, //
            0.2, 0.3, 0.0, 0.5, 3.0, //
        ];

        let q_inv = invert_q_matrix(&q, n);

        // Compute Q^-1 . Q and measure Frobenius-like deviation from I
        let mut max_diff = 0.0;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q_inv[i * n + k] * q[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (dot - expected).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 1e-12,
            "C1 FAIL: max|Q^-1 . Q - I| = {:.6e}, expected < 1e-12",
            max_diff,
        );
    }

    /// C2: Q^-1 is symmetric: max|Q^-1[n,m] - Q^-1[m,n]| < 1x10^-12
    ///
    /// The inverse of a real symmetric matrix is symmetric. A buggy
    /// implementation (wrong indexing, incorrect factorization) would
    /// produce an asymmetric result.
    #[test]
    fn test_q_inverse_symmetric() {
        let n = 5;
        // Same SPD matrix as C1
        let q: Vec<f64> = vec![
            4.0, 1.0, 0.0, 0.5, 0.2, //
            1.0, 5.0, 2.0, 0.0, 0.3, //
            0.0, 2.0, 6.0, 1.0, 0.0, //
            0.5, 0.0, 1.0, 4.0, 0.5, //
            0.2, 0.3, 0.0, 0.5, 3.0, //
        ];

        let q_inv = invert_q_matrix(&q, n);

        let mut max_asym = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                let diff = (q_inv[i * n + j] - q_inv[j * n + i]).abs();
                if diff > max_asym {
                    max_asym = diff;
                }
            }
        }

        assert!(
            max_asym < 1e-12,
            "C2 FAIL: max|Q^-1[n,m] - Q^-1[m,n]| = {:.6e}, expected < 1e-12",
            max_asym,
        );
    }

    /// C3: Q^-1 with a zero diagonal (singular projector channel) has that
    /// row/col zeroed, and the remaining non-singular sub-block is correct.
    ///
    /// Source: CASTEP nlpot.f90:13894-13959 - abs(ps_q(m,m,nsp1)) > tiny
    /// check skips non-augmenting projector channels.
    #[test]
    fn test_q_inverse_singular_diagonal() {
        let n = 4;
        // Q has a zero diagonal at index 2, simulating a non-augmenting
        // projector channel that CASTEP would skip.
        let q: Vec<f64> = vec![
            4.0, 1.0, 0.0, 0.5, //
            1.0, 5.0, 0.0, 0.3, //
            0.0, 0.0, 0.0, 0.0, //
            0.5, 0.3, 0.0, 3.0, //
        ];

        let q_inv = invert_q_matrix(&q, n);

        // Row and col 2 should be all zeros (singular channel)
        for j in 0..n {
            assert!(
                q_inv[2 * n + j].abs() < 1e-16,
                "C3 FAIL: Q^-1[2,{j}] should be zero, got {:.6e}",
                q_inv[2 * n + j],
            );
            assert!(
                q_inv[j * n + 2].abs() < 1e-16,
                "C3 FAIL: Q^-1[{j},2] should be zero, got {:.6e}",
                q_inv[j * n + 2],
            );
        }

        // Non-singular sub-block (indices 0, 1, 3) should still satisfy
        // the inverse identity
        let sub_indices = [0usize, 1, 3];
        let mut max_diff = 0.0;
        for &i in &sub_indices {
            for &j in &sub_indices {
                let mut dot = 0.0;
                for &k in &sub_indices {
                    dot += q_inv[i * n + k] * q[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (dot - expected).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 1e-12,
            "C3 FAIL: non-singular sub-block max|Q^-1 . Q - I| = {:.6e}, expected < 1e-12",
            max_diff,
        );
    }

    // -----------------------------------------------------------------------
    // C1-C3: R_beta = (−Q⁻¹ − C)⁻¹ — assembly, symmetry, inverse identity
    // -----------------------------------------------------------------------

    /// C1: R_beta[n,m] is symmetric: max|R[n,m] − R[m,n]| < 1×10⁻¹²
    ///
    /// Anchored to the mathematical property that the inverse of a real
    /// symmetric matrix is symmetric. Test creates synthetic Q⁻¹ (5×5 SPD)
    /// and C (5×5 Hermitian = 0.1*I), computes R_beta = (−C − Q⁻¹)⁻¹, and
    /// checks the symmetry of the result.
    #[test]
    fn test_r_beta_symmetric() {
        let ne = 5;

        // Q⁻¹: 5×5 SPD matrix (same as in test_q_inverse_identity)
        let q_inv: Vec<f64> = vec![
            4.0, 1.0, 0.0, 0.5, 0.2, //
            1.0, 5.0, 2.0, 0.0, 0.3, //
            0.0, 2.0, 6.0, 1.0, 0.0, //
            0.5, 0.0, 1.0, 4.0, 0.5, //
            0.2, 0.3, 0.0, 0.5, 3.0, //
        ];

        // C: 0.1 × I (real diagonal, Hermitian)
        let mut c_matrix = Array2::<Complex64>::zeros((ne, ne));
        for i in 0..ne {
            c_matrix[[i, i]] = Complex64::new(0.1, 0.0);
        }

        let r_beta = assemble_r_beta(&q_inv, &c_matrix, ne, 1.0);

        let mut max_asym = 0.0;
        for i in 0..ne {
            for j in i + 1..ne {
                let diff = (r_beta[[i, j]] - r_beta[[j, i]]).norm();
                if diff > max_asym {
                    max_asym = diff;
                }
            }
        }

        assert!(
            max_asym < 1e-12,
            "C1 FAIL: max|R[n,m] - R[m,n]| = {:.6e}, expected < 1e-12",
            max_asym,
        );
    }

    /// C2: R_beta = (−Q⁻¹ − C)⁻¹ — inverse identity verification
    ///
    /// Computes M = −C_re − Q⁻¹ (real part), then checks
    /// ‖M · R_beta − I‖_max < 1×10⁻¹⁰.
    ///
    /// Anchored to the matrix inverse definition M · M⁻¹ = I. Uses
    /// synthetic data where M is negative definite (guaranteed invertible).
    #[test]
    fn test_r_beta_inverse_identity() {
        let ne = 5;

        // Q⁻¹: 5×5 SPD matrix
        let q_inv: Vec<f64> = vec![
            4.0, 1.0, 0.0, 0.5, 0.2, //
            1.0, 5.0, 2.0, 0.0, 0.3, //
            0.0, 2.0, 6.0, 1.0, 0.0, //
            0.5, 0.0, 1.0, 4.0, 0.5, //
            0.2, 0.3, 0.0, 0.5, 3.0, //
        ];

        // C: 0.1 × I (real diagonal, Hermitian)
        let mut c_matrix = Array2::<Complex64>::zeros((ne, ne));
        for i in 0..ne {
            c_matrix[[i, i]] = Complex64::new(0.1, 0.0);
        }

        let r_beta = assemble_r_beta(&q_inv, &c_matrix, ne, 1.0);

        // Compute M = −C_re − Q⁻¹
        let mut m_mat = Array2::<f64>::zeros((ne, ne));
        for i in 0..ne {
            for j in 0..ne {
                m_mat[[i, j]] = -c_matrix[[i, j]].re - q_inv[i * ne + j];
            }
        }

        // Compute M · R_beta (real-real product since R has zero imag)
        let mut max_diff = 0.0;
        for i in 0..ne {
            for j in 0..ne {
                let mut dot = 0.0;
                for k in 0..ne {
                    dot += m_mat[[i, k]] * r_beta[[k, j]].re;
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (dot - expected).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 1e-10,
            "C2 FAIL: max|M·R − I| = {:.6e}, expected < 1e-10",
            max_diff,
        );
    }

    /// C3: R_beta with mixture_weight = 2.0 — verifies M = −C − Q⁻¹/2
    /// and the inverse identity still holds.
    ///
    /// Same synthetic data as C1/C2, but mixture_weight = 2.0 changes
    /// the matrix being inverted. The inverse identity should still hold.
    #[test]
    fn test_r_beta_mixture_weight() {
        let ne = 5;
        let mixture_weight = 2.0;

        // Q⁻¹: 5×5 SPD matrix
        let q_inv: Vec<f64> = vec![
            4.0, 1.0, 0.0, 0.5, 0.2, //
            1.0, 5.0, 2.0, 0.0, 0.3, //
            0.0, 2.0, 6.0, 1.0, 0.0, //
            0.5, 0.0, 1.0, 4.0, 0.5, //
            0.2, 0.3, 0.0, 0.5, 3.0, //
        ];

        // C: 0.1 × I (real diagonal, Hermitian)
        let mut c_matrix = Array2::<Complex64>::zeros((ne, ne));
        for i in 0..ne {
            c_matrix[[i, i]] = Complex64::new(0.1, 0.0);
        }

        let r_beta = assemble_r_beta(&q_inv, &c_matrix, ne, mixture_weight);

        // Compute M = −C_re − Q⁻¹/w
        let mut m_mat = Array2::<f64>::zeros((ne, ne));
        for i in 0..ne {
            for j in 0..ne {
                m_mat[[i, j]] = -c_matrix[[i, j]].re - q_inv[i * ne + j] / mixture_weight;
            }
        }

        // Compute M · R_beta
        let mut max_diff = 0.0;
        for i in 0..ne {
            for j in 0..ne {
                let mut dot = 0.0;
                for k in 0..ne {
                    dot += m_mat[[i, k]] * r_beta[[k, j]].re;
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (dot - expected).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 1e-10,
            "C3 FAIL: max|M·R − I| (w={}) = {:.6e}, expected < 1e-10",
            mixture_weight,
            max_diff,
        );
    }

    // -----------------------------------------------------------------------
    // Q_RCQ assembly — dimensions, formula verification, block structure
    // -----------------------------------------------------------------------

    /// C1: Q_RCQ dimensions match (n_total_proj, n_total_proj).
    ///
    /// Creates 2-ion synthetic data with ne=3 and ne=4 projectors,
    /// so n_total_proj = 7. Q_RCQ should be 7×7.
    #[test]
    fn test_q_rcq_dimensions() {
        let ne0 = 3;
        let ne1 = 4;
        let n_total_proj = ne0 + ne1;

        // Per-ion Q matrices (real symmetric)
        let q0 = vec![
            4.0, 1.0, 0.5,
            1.0, 5.0, 0.3,
            0.5, 0.3, 6.0,
        ];
        let q1 = vec![
            3.0, 0.2, 0.0, 0.1,
            0.2, 4.0, 0.5, 0.0,
            0.0, 0.5, 5.0, 0.3,
            0.1, 0.0, 0.3, 6.0,
        ];

        // Per-ion C matrices (complex Hermitian)
        let c0 = Array2::<Complex64>::from_shape_vec((ne0, ne0), vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.2, -0.1), Complex64::new(0.1, 0.3),
            Complex64::new(0.2, 0.1), Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.5),
            Complex64::new(0.1, -0.3), Complex64::new(0.0, -0.5), Complex64::new(3.0, 0.0),
        ]).unwrap();

        let c1 = Array2::<Complex64>::from_shape_vec((ne1, ne1), vec![
            Complex64::new(1.5, 0.0), Complex64::new(0.1, 0.2), Complex64::new(0.0, -0.1), Complex64::new(0.3, 0.0),
            Complex64::new(0.1, -0.2), Complex64::new(2.5, 0.0), Complex64::new(0.2, 0.1), Complex64::new(0.0, -0.3),
            Complex64::new(0.0, 0.1), Complex64::new(0.2, -0.1), Complex64::new(3.5, 0.0), Complex64::new(0.1, 0.2),
            Complex64::new(0.3, 0.0), Complex64::new(0.0, 0.3), Complex64::new(0.1, -0.2), Complex64::new(4.5, 0.0),
        ]).unwrap();

        // Per-ion R_beta matrices (complex, real-valued)
        let rb0 = Array2::<Complex64>::zeros((ne0, ne0));
        let rb1 = Array2::<Complex64>::zeros((ne1, ne1));

        let ion_offsets = vec![0, ne0, n_total_proj];
        let mixture_weights = vec![1.0, 1.0];

        // Build block-diagonal c_global from per-ion C matrices
        let mut c_global = Array2::<Complex64>::zeros((n_total_proj, n_total_proj));
        for m in 0..ne0 {
            for n in 0..ne0 {
                c_global[[m, n]] = c0[[m, n]];
            }
        }
        for m in 0..ne1 {
            for n in 0..ne1 {
                c_global[[ne0 + m, ne0 + n]] = c1[[m, n]];
            }
        }

        let q_rcq = assemble_q_rcq(
            &[q0, q1], &c_global, &[rb0, rb1],
            &ion_offsets, &mixture_weights,
        );

        assert_eq!(
            q_rcq.shape(),
            &[n_total_proj, n_total_proj],
            "C1 FAIL: Q_RCQ should be {n_total_proj}×{n_total_proj}, got {:?}",
            q_rcq.shape(),
        );
    }

    /// C2: Q_RCQ against explicit manual computation.
    ///
    /// For a single ion with ne=3, computes Q_RCQ via the production function
    /// AND via an explicit per-element computation of -Q + C·Q - R_beta·(C·Q).
    /// Must agree to 1e-12 (cross-path verification).
    ///
    /// The manual path uses plain ndarray dot products (structurally
    /// independent from the assembly function's per-loops).
    #[test]
    fn test_q_rcq_against_manual() {
        let ne = 3;
        let n_total_proj = ne;
        let mixture_weight = 1.0;

        // Q matrix (real symmetric, SPD)
        let q: Vec<f64> = vec![
            4.0, 1.0, 0.5,
            1.0, 5.0, 0.3,
            0.5, 0.3, 6.0,
        ];

        // Build C as Array2<Complex64> (Hermitian)
        let mut c = Array2::<Complex64>::zeros((ne, ne));
        c[[0, 0]] = Complex64::new(1.0, 0.0);
        c[[0, 1]] = Complex64::new(0.2, -0.1);
        c[[0, 2]] = Complex64::new(0.1, 0.3);
        c[[1, 0]] = Complex64::new(0.2, 0.1); // conj of [0,1]
        c[[1, 1]] = Complex64::new(2.0, 0.0);
        c[[1, 2]] = Complex64::new(0.0, 0.5);
        c[[2, 0]] = Complex64::new(0.1, -0.3); // conj of [0,2]
        c[[2, 1]] = Complex64::new(0.0, -0.5); // conj of [1,2]
        c[[2, 2]] = Complex64::new(3.0, 0.0);

        // R_beta (3×3 complex, using non-trivial real-valued matrix)
        let mut r_beta = Array2::<Complex64>::zeros((ne, ne));
        r_beta[[0, 0]] = Complex64::new(0.5, 0.0);
        r_beta[[0, 1]] = Complex64::new(0.1, 0.0);
        r_beta[[1, 0]] = Complex64::new(0.1, 0.0);
        r_beta[[0, 2]] = Complex64::new(0.05, 0.0);
        r_beta[[2, 0]] = Complex64::new(0.05, 0.0);
        r_beta[[1, 1]] = Complex64::new(0.4, 0.0);
        r_beta[[1, 2]] = Complex64::new(0.08, 0.0);
        r_beta[[2, 1]] = Complex64::new(0.08, 0.0);
        r_beta[[2, 2]] = Complex64::new(0.3, 0.0);

        let ion_offsets = vec![0, n_total_proj];
        let mixture_weights = vec![mixture_weight];

        // Production path (single ion: c_global is just c)
        let q_rcq = assemble_q_rcq(
            &[q.clone()], &c, &[r_beta.clone()], &ion_offsets, &mixture_weights,
        );

        // Manual path: explicit -Q + C·Q - R_beta·(C·Q)
        // Step 2: -Q * w
        let q_complex = Array2::from_shape_vec((ne, ne),
            q.iter().map(|&v| Complex64::new(-v * mixture_weight, 0.0)).collect()
        ).unwrap();
        let mut manual = q_complex.clone();

        // Step 3: + C·Q
        let q_w = Array2::from_shape_vec((ne, ne),
            q.iter().map(|&v| Complex64::new(v * mixture_weight, 0.0)).collect()
        ).unwrap();
        let cq = c.dot(&q_w);
        manual = manual + &cq;

        // Step 4: - R_beta·(C·Q)
        let r_cq = r_beta.dot(&cq);
        manual = manual - &r_cq;

        // Cross-path verification
        let mut max_diff = 0.0;
        for i in 0..ne {
            for j in 0..ne {
                let diff = (q_rcq[[i, j]] - manual[[i, j]]).norm();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 1e-12,
            "C2 FAIL: max|Q_RCQ - manual| = {:.6e}, expected < 1e-12",
            max_diff,
        );
    }

    /// C3: Off-diagonal blocks (ion i × ion j, i ≠ j) have no −Q term.
    ///
    /// With block-diagonal C, the off-diagonal contribution is
    /// C·Q − R_beta·(C·Q) where C is block-diagonal. Since the per-ion C
    /// contribution at off-diagonal is zero, the result should be zero
    /// to numerical precision.
    ///
    /// The test constructs two ions (ne=2 each), computes Q_RCQ, and verifies
    /// that off-diagonal blocks are zero (no −Q initialization for off-diagonal).
    #[test]
    fn test_q_rcq_two_ions_block_structure() {
        let ne0 = 2;
        let ne1 = 2;

        // Ion 0: 2×2 Q, C, R_beta
        let q0: Vec<f64> = vec![3.0, 0.5, 0.5, 4.0];
        let mut c0 = Array2::<Complex64>::zeros((ne0, ne0));
        c0[[0, 0]] = Complex64::new(1.0, 0.0);
        c0[[0, 1]] = Complex64::new(0.3, -0.2);
        c0[[1, 0]] = Complex64::new(0.3, 0.2);
        c0[[1, 1]] = Complex64::new(2.0, 0.0);
        let mut rb0 = Array2::<Complex64>::zeros((ne0, ne0));
        rb0[[0, 0]] = Complex64::new(0.4, 0.0);
        rb0[[0, 1]] = Complex64::new(0.1, 0.0);
        rb0[[1, 0]] = Complex64::new(0.1, 0.0);
        rb0[[1, 1]] = Complex64::new(0.3, 0.0);

        // Ion 1: 2×2 Q, C, R_beta
        let q1: Vec<f64> = vec![5.0, 0.2, 0.2, 6.0];
        let mut c1 = Array2::<Complex64>::zeros((ne1, ne1));
        c1[[0, 0]] = Complex64::new(1.5, 0.0);
        c1[[0, 1]] = Complex64::new(0.1, 0.1);
        c1[[1, 0]] = Complex64::new(0.1, -0.1);
        c1[[1, 1]] = Complex64::new(2.5, 0.0);
        let mut rb1 = Array2::<Complex64>::zeros((ne1, ne1));
        rb1[[0, 0]] = Complex64::new(0.3, 0.0);
        rb1[[0, 1]] = Complex64::new(0.05, 0.0);
        rb1[[1, 0]] = Complex64::new(0.05, 0.0);
        rb1[[1, 1]] = Complex64::new(0.2, 0.0);

        let n_total_proj = ne0 + ne1;
        let ion_offsets = vec![0, ne0, n_total_proj];
        let mixture_weights = vec![1.0, 1.0];

        // Build block-diagonal c_global from per-ion C matrices
        let mut c_global = Array2::<Complex64>::zeros((n_total_proj, n_total_proj));
        for m in 0..ne0 {
            for n in 0..ne0 {
                c_global[[m, n]] = c0[[m, n]];
            }
        }
        for m in 0..ne1 {
            for n in 0..ne1 {
                c_global[[ne0 + m, ne0 + n]] = c1[[m, n]];
            }
        }

        let q_rcq = assemble_q_rcq(
            &[q0, q1], &c_global, &[rb0.clone(), rb1.clone()],
            &ion_offsets, &mixture_weights,
        );

        // Compute expected diagonal blocks manually
        // Ion 0: -Q + C·Q - R_beta·(C·Q)
        let q0_arr = Array2::from_shape_vec((ne0, ne0), vec![
            Complex64::new(3.0, 0.0), Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let q0_scaled = q0_arr.mapv(|v| Complex64::new(v.re * (-1.0), 0.0));
        let cq0 = c0.dot(&q0_arr.mapv(|v| Complex64::new(v.re, 0.0)));
        let r_cq0 = rb0.dot(&cq0);
        let expected_block0 = q0_scaled + &cq0 - &r_cq0;

        // Ion 1: -Q + C·Q - R_beta·(C·Q)
        let q1_arr = Array2::from_shape_vec((ne1, ne1), vec![
            Complex64::new(5.0, 0.0), Complex64::new(0.2, 0.0),
            Complex64::new(0.2, 0.0), Complex64::new(6.0, 0.0),
        ]).unwrap();
        let q1_scaled = q1_arr.mapv(|v| Complex64::new(v.re * (-1.0), 0.0));
        let cq1 = c1.dot(&q1_arr.mapv(|v| Complex64::new(v.re, 0.0)));
        let r_cq1 = rb1.dot(&cq1);
        let expected_block1 = q1_scaled + &cq1 - &r_cq1;

        // Verify diagonal blocks
        for m in 0..ne0 {
            for n in 0..ne0 {
                let diff = (q_rcq[[m, n]] - expected_block0[[m, n]]).norm();
                assert!(
                    diff < 1e-12,
                    "C3 FAIL: diag block0[{m},{n}] mismatch: diff = {:.6e}",
                    diff,
                );
            }
        }
        for m in 0..ne1 {
            for n in 0..ne1 {
                let diff = (q_rcq[[ne0 + m, ne0 + n]] - expected_block1[[m, n]]).norm();
                assert!(
                    diff < 1e-12,
                    "C3 FAIL: diag block1[{m},{n}] mismatch: diff = {:.6e}",
                    diff,
                );
            }
        }

        // Verify off-diagonal blocks are zero (no −Q term for off-diagonal,
        // and block-diagonal C makes C·Q = R_beta·(C·Q) = 0 for off-diagonal)
        // Off-diagonal block (0,1): rows 0..ne0, cols ne0..ne0+ne1
        let mut max_off_a = 0.0_f64;
        for m in 0..ne0 {
            for n in 0..ne1 {
                let val = q_rcq[[m, ne0 + n]];
                let abs_val = val.norm();
                if abs_val > max_off_a {
                    max_off_a = abs_val;
                }
            }
        }
        assert!(
            max_off_a < 1e-14,
            "C3 FAIL: off-diagonal block (0,1) should be zero, max |entry| = {:.6e}",
            max_off_a,
        );

        // Off-diagonal block (1,0): rows ne0..ne0+ne1, cols 0..ne0
        let mut max_off_b = 0.0_f64;
        for m in 0..ne1 {
            for n in 0..ne0 {
                let val = q_rcq[[ne0 + m, n]];
                let abs_val = val.norm();
                if abs_val > max_off_b {
                    max_off_b = abs_val;
                }
            }
        }
        assert!(
            max_off_b < 1e-14,
            "C3 FAIL: off-diagonal block (1,0) should be zero, max |entry| = {:.6e}",
            max_off_b,
        );
    }
}
