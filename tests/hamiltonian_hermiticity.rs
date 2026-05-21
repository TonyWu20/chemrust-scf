//! Hermiticity self-test for the production Hamiltonian apply path.
//!
//! For a Hermitian operator H, ⟨ψ_a|H|ψ_b⟩ = ⟨ψ_b|H|ψ_a⟩^* must hold for
//! every pair of vectors. The diagonal ⟨ψ|H|ψ⟩ must be REAL.
//!
//! If H is non-Hermitian, Lanczos eigenvalue estimation (used to compute
//! the Chebyshev filter b_up) returns garbage. This test isolates the
//! Hermiticity question independent of any iteration dynamics.
//!
//! Per the open-followups #7 and the iter-2 b_up jump (22.8 → 107.5 Ha
//! despite ~unchanged V_eff range), a non-Hermitian H is one of the
//! remaining hypotheses for the SCF divergence.

mod fixtures;

use num_complex::Complex64;

fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

/// Compute `H_sub[a, b] = ⟨ψ_a | h_part | ψ_b⟩ = Σ_g ψ_a(g)̄ · h_part_b(g)`
/// for a subset of band indices, returning a flat HashMap.
///
/// `psi` layout: row-major (n_bands × n_pw) — `psi[b * n_pw + g]`.
/// `hpsi` layout: same.
fn compute_h_sub_for_pairs(
    psi: &[Complex64],
    hpsi: &[Complex64],
    n_pw: usize,
    pairs: &[(usize, usize)],
) -> Vec<((usize, usize), Complex64)> {
    pairs
        .iter()
        .map(|&(a, b)| {
            let mut acc = Complex64::ZERO;
            for g in 0..n_pw {
                acc += psi[a * n_pw + g].conj() * hpsi[b * n_pw + g];
            }
            ((a, b), acc)
        })
        .collect()
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn hamiltonian_apply_is_hermitian_on_fixture_state() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);

    // Build V_eff once.
    let v_built = state.build_v_eff_with_energy().expect("build_v_eff");
    let n_bands = 160; // fixture has 160 bands

    // Apply H to the full 160-band ψ. Returns hpsi_t (kinetic only),
    // hpsi_tv (kinetic + V_loc), hpsi_full (kinetic + V_loc + V_NL),
    // all as flat Vec<Complex64> in row-major (n_bands × n_pw) layout.
    let components = v_built
        .apply_h_components_for_test(None)
        .expect("apply_h_components_for_test");
    let hpsi_t = components.hpsi_t;
    let hpsi_tv = components.hpsi_tv;
    let hpsi_full = components.hpsi_full;
    let out_n_bands = components.n_bands;
    let n_pw = components.n_pw;

    assert_eq!(out_n_bands, n_bands);
    eprintln!("n_bands={n_bands} n_pw={n_pw}");

    // We need the input ψ to compute ⟨ψ_a | hpsi_b⟩. The apply uses
    // self.psi internally; reproduce by re-reading the fixture's wavefunctions.
    let kpt = fx.check.wavefunction.as_ref().unwrap();
    let psi_flat: Vec<Complex64> = kpt.kpt_data[0].bands.concat();
    assert_eq!(psi_flat.len(), n_bands * n_pw);

    // Sample a manageable subset of band pairs (covers low, mid, high).
    let probe_bands = [0usize, 1, 5, 50, 80, 100, 130, 159];
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for &a in &probe_bands {
        for &b in &probe_bands {
            pairs.push((a, b));
        }
    }

    // Three Hermiticity checks: T, T+V_loc, T+V_loc+V_NL.
    for (label, hpsi) in [
        ("T (kinetic only)", &hpsi_t),
        ("T + V_loc (FFT path)", &hpsi_tv),
        ("T + V_loc + V_NL (full H)", &hpsi_full),
    ] {
        let hsub = compute_h_sub_for_pairs(&psi_flat, hpsi, n_pw, &pairs);
        let hsub_map: std::collections::HashMap<(usize, usize), Complex64> =
            hsub.into_iter().collect();

        // Diagonal: imag part must be ~0 (real expectation values).
        let mut max_diag_imag = 0.0_f64;
        for &a in &probe_bands {
            let v = hsub_map[&(a, a)];
            let imag = v.im.abs();
            if imag > max_diag_imag {
                max_diag_imag = imag;
            }
            eprintln!(
                "[{label}] H_sub[{a}, {a}] = {:.6e} + {:.3e}i  Ha",
                v.re, v.im
            );
        }
        eprintln!("[{label}] max |Im(H_sub[a, a])| = {max_diag_imag:.3e}");

        // Off-diagonal Hermiticity: H_sub[a, b] == H_sub[b, a]^*.
        let mut max_offdiag_violation = 0.0_f64;
        let mut worst_pair = (0usize, 0usize);
        for &a in &probe_bands {
            for &b in &probe_bands {
                if a >= b {
                    continue;
                }
                let h_ab = hsub_map[&(a, b)];
                let h_ba = hsub_map[&(b, a)];
                let violation = (h_ab - h_ba.conj()).norm();
                if violation > max_offdiag_violation {
                    max_offdiag_violation = violation;
                    worst_pair = (a, b);
                }
            }
        }
        let (a, b) = worst_pair;
        let h_ab = hsub_map[&(a, b)];
        let h_ba = hsub_map[&(b, a)];
        eprintln!(
            "[{label}] worst off-diag pair (a={a}, b={b}): \
             H_sub[a,b]={:.4e}+{:.4e}i  H_sub[b,a]^*={:.4e}+{:.4e}i  \
             |Δ|={max_offdiag_violation:.3e}",
            h_ab.re, h_ab.im, h_ba.conj().re, h_ba.conj().im
        );

        // Discriminator: typical diagonal H_sub values are O(0.1-1) Ha;
        // a non-Hermitian bug would give Im(diag) of similar magnitude.
        // We threshold at 1e-6 Ha which is well above ZGEMM noise.
        // Off-diagonal threshold: 1e-6 absolute (Hermitian: typical
        // off-diag values are 0 for orthonormal eigenstates, but
        // numerical noise of inner-products is ~1e-12).
        assert!(
            max_diag_imag < 1e-6,
            "[{label}] H is non-Hermitian: max diagonal imag = {max_diag_imag:.3e} Ha. \
             Lanczos eigenvalue estimation will fail."
        );
        assert!(
            max_offdiag_violation < 1e-6,
            "[{label}] H is non-Hermitian: max off-diag |H[a,b] - H[b,a]^*| = \
             {max_offdiag_violation:.3e} Ha (pair a={a}, b={b}). \
             Lanczos eigenvalue estimation will fail."
        );
    }
}

// ---------------------------------------------------------------------------
// Iter-2 Hermiticity: drive one SCF step, then check H_iter2
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data, runs ≥1 full SCF iter"]
fn hamiltonian_apply_is_hermitian_on_iter2_state() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);

    // Iter-1 SCF cycle.
    let iter1_v = state.build_v_eff_with_energy().expect("iter-1 build_v_eff");
    let iter1_diag = iter1_v.diagonalize(8, None).expect("iter-1 diagonalize");
    let iter1_dens = iter1_diag.construct_density_off().expect("iter-1 construct_density");
    let iter1_mixed = iter1_dens.mix();
    let iter2_init = match iter1_mixed.check(1e-8).expect("iter-1 check") {
        chemrust_scf::CheckOutcome::Converged(_) => panic!(
            "iter-1 unexpectedly converged — iter-2 Hermiticity test cannot run"
        ),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    // Iter-2 V_eff from iter-1 output density.
    let iter2_v = iter2_init.build_v_eff_with_energy().expect("iter-2 build_v_eff");
    eprintln!("[iter-2] V_eff built from iter-1 output density");

    let components = iter2_v
        .apply_h_components_for_test(None)
        .expect("apply_h_components_for_test iter-2");
    let hpsi_t = components.hpsi_t;
    let hpsi_tv = components.hpsi_tv;
    let hpsi_full = components.hpsi_full;
    let n_bands = components.n_bands;
    let n_pw = components.n_pw;
    eprintln!("[iter-2] n_bands={n_bands} n_pw={n_pw}");

    // CRITICAL: H is applied to iter2_v.psi (iter-1's RR output, rotated
    // from CASTEP ψ). We must compute ⟨ψ|H|ψ⟩ with the SAME ψ that H was
    // applied to, otherwise the inner product is meaningless. Pull the
    // actual ψ from the state via the debug accessor.
    let psi_flat: Vec<Complex64> = iter2_v.psi_data().to_vec();
    assert_eq!(psi_flat.len(), n_bands * n_pw);

    let probe_bands = [0usize, 1, 5, 50, 80, 100, 130, 159];
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for &a in &probe_bands {
        for &b in &probe_bands {
            pairs.push((a, b));
        }
    }

    for (label, hpsi) in [
        ("iter-2 T (kinetic only)", &hpsi_t),
        ("iter-2 T + V_loc (FFT path)", &hpsi_tv),
        ("iter-2 T + V_loc + V_NL (full H)", &hpsi_full),
    ] {
        let hsub = compute_h_sub_for_pairs(&psi_flat, hpsi, n_pw, &pairs);
        let hsub_map: std::collections::HashMap<(usize, usize), Complex64> =
            hsub.into_iter().collect();

        let mut max_diag_imag = 0.0_f64;
        for &a in &probe_bands {
            let v = hsub_map[&(a, a)];
            if v.im.abs() > max_diag_imag {
                max_diag_imag = v.im.abs();
            }
            eprintln!(
                "[{label}] H_sub[{a}, {a}] = {:.6e} + {:.3e}i  Ha",
                v.re, v.im
            );
        }
        eprintln!("[{label}] max |Im(H_sub[a, a])| = {max_diag_imag:.3e}");

        let mut max_offdiag_violation = 0.0_f64;
        let mut worst_pair = (0usize, 0usize);
        for &a in &probe_bands {
            for &b in &probe_bands {
                if a >= b {
                    continue;
                }
                let h_ab = hsub_map[&(a, b)];
                let h_ba = hsub_map[&(b, a)];
                let violation = (h_ab - h_ba.conj()).norm();
                if violation > max_offdiag_violation {
                    max_offdiag_violation = violation;
                    worst_pair = (a, b);
                }
            }
        }
        let (a, b) = worst_pair;
        let h_ab = hsub_map[&(a, b)];
        let h_ba = hsub_map[&(b, a)];
        eprintln!(
            "[{label}] worst off-diag pair (a={a}, b={b}): \
             H_sub[a,b]={:.4e}+{:.4e}i  H_sub[b,a]^*={:.4e}+{:.4e}i  \
             |Δ|={max_offdiag_violation:.3e}",
            h_ab.re, h_ab.im, h_ba.conj().re, h_ba.conj().im
        );

        assert!(
            max_diag_imag < 1e-6,
            "[{label}] H is non-Hermitian: max diagonal imag = {max_diag_imag:.3e} Ha."
        );
        assert!(
            max_offdiag_violation < 1e-6,
            "[{label}] H is non-Hermitian: max off-diag |H[a,b] - H[b,a]^*| = \
             {max_offdiag_violation:.3e} Ha (pair a={a}, b={b})."
        );
    }
}
