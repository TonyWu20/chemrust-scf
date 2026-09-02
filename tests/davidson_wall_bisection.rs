// ---------------------------------------------------------------------------
// Davidson wall bisection harness: NiO non-spin SCF loop
// ---------------------------------------------------------------------------
//
// Verifies the D1/D2 S-norm gating in `davidson_diagonalise`
// (docs/load-bearing-diagnostic-overhead.md). The D1/D2 blocks used to
// run in every build and acted as accidental host-delay walls masking the
// hidden sync dependency. They are now gated behind `scf_diag`.
//
// This harness runs the full NiO SCF loop on the BLOCK DAVIDSON path
// (the default eigensolver) and asserts convergence. It must converge in
// both feature sets:
//
//   cargo test --release -- --ignored davidson_wall_bisection
//       (default features: D1/D2 ON, wall present)
//   cargo test --release --features scf_diag -- --ignored davidson_wall_bisection
//       (scf_diag: D1/D2 OFF, wall removed, zdotc reads synced)
//
// If the scf_diag build diverges while the default build converges, a
// wall is still load-bearing and the gating is premature. The relaxed
// gate catches eigenvalue explosion (the documented divergence mode)
// without comparing against the CASTEP reference energy.

mod fixtures;

#[test]
#[ignore = "requires GPU and NiO fixture data"]
fn nio_scf_loop_converges() {
    use std::sync::Arc;

    // Per-iteration energy table (CASTEP column format) is emitted via
    // tracing::info! inside run_scf_with_energy_gated. Install a
    // subscriber so the test log shows every SCF step: energy, Fermi
    // level, energy gain, and timer. A diverging SCF shows itself in
    // the first few rows.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .try_init()
        .ok();

    let fx = fixtures::nio_no_spin::fixture();
    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();

    // Full 14-k-point system (CASTEP runs all 14 k-points; the
    // single-k-point reduced builder is not a valid SCF system).
    // CASTEP scheme: Pulay (Kerker first pass, then DIIS), matching
    // the NiO .param.  CASTEP converged the same loop in 91 cycles
    // (job 2965); the pure loop converges in ~31 cycles.
    let state = fixtures::nio_no_spin::build_scf_state_all_kpts(fx, &stream);
    let gate = chemrust_scf::ScfDivergenceGate {
        // Relaxed thresholds: this harness detects divergence (eigenvalue
        // explosion / V_eff runaway), not CASTEP energy agreement.
        max_last_band_ha: 30.0,
        min_band0_ha: -30.0,
        max_veff_range_factor: 5.0,
        // CASTEP took 91 cycles from the iter-2 state; the pure loop
        // converges in ~31.  Allow the long tail.
        max_iter: 40,
        electron_count_tolerance: 0.05,
        soft_fraction_tolerance: 0.20,
        #[cfg(feature = "scf_diag")]
        check_raw_sections: None,
    };
    let result = chemrust_scf::run_scf_with_energy_gated(state, 8, 1e-8, Some(gate));
    match result {
        Ok(final_state) => {
            let ev = final_state.total_energy * chemrust_scf::HARTREE_TO_EV;
            eprintln!("[wall_bisection] NiO SCF converged: E_total = {ev:.8} eV");
        }
        Err(e) => {
            eprintln!("[wall_bisection] NiO SCF FAILED: {e}");
            panic!("SCF loop must converge: {e}");
        }
    }
}
