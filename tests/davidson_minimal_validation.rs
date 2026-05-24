// ---------------------------------------------------------------------------
// Minimal Davidson single-sweep validation (Phase 0 Gate 3)
// ---------------------------------------------------------------------------
//
// Self-consistency test: run the Davidson-based diagonalize with CASTEP's
// V_eff pinned and compare the output ψ against the CASTEP reference.
//
// All tests require GPU and CASTEP fixture data — marked `#[ignore]`.
// Run with: cargo test --release --test davidson_minimal_validation -- --ignored

use num_complex::Complex64;

mod fixtures;

#[cfg(feature = "scf_diag")]
mod tests {
    use chemrust_scf::*;
    use num_complex::Complex64;

    /// Self-consistency test: with CASTEP's V_eff pinned, the Davidson
    /// single-sweep should preserve the CASTEP-converged ψ band-by-band
    /// to within machine precision.
    #[test]
    #[ignore = "requires GPU and CASTEP fixture data"]
    fn davidson_minimal_self_consistency_with_pinned_castep_veff() {
        // Set env var for dispatch (reserved — Davidson dispatch not yet wired)
        unsafe {
            std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson");
        }

        let fx = fixtures::cu111_co::fixture();
        let veff_state = fixtures::cu111_co::build_state_with_castep_veff(fx);
        let diag = veff_state.diagonalize(0, None).expect("davidson diag");
        let psi_out = diag.psi_data();

        let psi_castep = fixtures::cu111_co::castep_psi_first_kpoint(fx);
        let n_pw = fixtures::cu111_co::n_pw_first_kpoint(fx);

        // Compare psi_out vs psi_castep band-by-band
        for b in 0..160.min(psi_castep.len() / n_pw) {
            let band_out = &psi_out[b * n_pw..(b + 1) * n_pw];
            let band_in = &psi_castep[b * n_pw..(b + 1) * n_pw];
            let diff_norm: f64 = band_in
                .iter()
                .zip(band_out.iter())
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(
                diff_norm < 1e-12,
                "band {b}: ‖ψ_out − ψ_castep‖₂ = {diff_norm:.3e}, want ≤ 1e-12"
            );
        }

        // Cleanup
        unsafe {
            std::env::remove_var("CHEMRUST_EIGENSOLVER");
        }
    }
}
