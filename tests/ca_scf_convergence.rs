//! SCF convergence tests against CASTEP reference data.
//!
//! Two sub-tests, both `#[ignore]` (require GPU):
//! 1. **Fixed-point stability** — start from the converged state, verify the
//!    SCF loop reproduces the reference energy without drifting.
//! 2. **Perturbation recovery** — add 5% noise to the converged density, verify
//!    the SCF converges back to the reference energy.
//!
//! The `.check` caveat: `.check` stores the converged final state.  We cannot
//! test SCF convergence from CASTEP's actual starting density (pseudoatomic SCF +
//! atomic superposition is not implemented in chemrust-hamiltonian).  The
//! perturbation-recovery test is the practical workaround.

mod fixtures;

use rand::Rng;

/// Returns `true` if a CUDA-capable GPU is available at device 0.
fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

/// Range (max − min) of a V_eff field on its fine grid.
fn v_eff_range(v: &chemrust_hamiltonian_core::EffectivePotential) -> f64 {
    let arr = v.as_real_grid().as_real_array();
    let min = arr.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max - min
}

// ---------------------------------------------------------------------------
// Test 3a: Fixed-point stability
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn fixed_point_matches_castep_energy() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);
    let result = chemrust_scf::run_scf_with_energy(state, 8, 1e-8).expect("SCF converged");

    let computed_ev = result.total_energy * chemrust_scf::HARTREE_TO_EV;
    let diff_ev = (computed_ev - fixtures::cu111_co::REFERENCE_ENERGY_EV).abs();

    println!("Computed total energy: {:.8} eV", computed_ev);
    println!("Reference total energy: {:.8} eV", fixtures::cu111_co::REFERENCE_ENERGY_EV);
    println!("Absolute difference:    {:.8} eV", diff_ev);

    assert!(
        diff_ev < fixtures::cu111_co::TOLERANCE_EV,
        "Total energy differs by {:.8} eV, exceeds tolerance {:.8} eV",
        diff_ev,
        fixtures::cu111_co::TOLERANCE_EV,
    );
}

// ---------------------------------------------------------------------------
// Test 3b: Perturbation recovery
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data, long-running"]
fn perturbation_recovers_castep_energy() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixtures::cu111_co::fixture();
    let mut state = fixtures::cu111_co::build_scf_state(fx);

    // Apply 5% multiplicative noise, renormalise to preserve total charge
    let mut rng = rand::thread_rng();
    let mut noisy_arr = state.density_mut().as_wave_array().clone().into_owned();
    let total: f64 = noisy_arr.iter().sum();

    for v in noisy_arr.iter_mut() {
        *v *= 1.0 + 0.05 * (rng.r#gen::<f64>() * 2.0 - 1.0);
    }

    // Renormalise to preserve total charge
    let new_total: f64 = noisy_arr.iter().sum();
    let scale = total / new_total;
    for v in noisy_arr.iter_mut() {
        *v *= scale;
    }

    *state.density_mut() = chemrust_scf::Density::from_inner(chemrust_scf::WaveGridArray::from_inner(
        noisy_arr,
    ));

    let result = chemrust_scf::run_scf_with_energy(state, 8, 1e-8).expect("SCF converged after perturbation");

    let computed_ev = result.total_energy * chemrust_scf::HARTREE_TO_EV;
    let diff_ev = (computed_ev - fixtures::cu111_co::REFERENCE_ENERGY_EV).abs();

    println!("Computed total energy (after perturbation): {:.8} eV", computed_ev);
    println!("Reference total energy: {:.8} eV", fixtures::cu111_co::REFERENCE_ENERGY_EV);
    println!("Absolute difference:    {:.8} eV", diff_ev);

    assert!(
        diff_ev < fixtures::cu111_co::TOLERANCE_EV,
        "Total energy after perturbation differs by {:.8} eV, exceeds tolerance {:.8} eV",
        diff_ev,
        fixtures::cu111_co::TOLERANCE_EV,
    );
}

// ---------------------------------------------------------------------------
// Issue #8 discriminator: USPP augmentation density wired into iter-2 V_eff
// ---------------------------------------------------------------------------
//
// Pre-fix (smooth-only ρ): iter-1 V_eff range ≈ 8.69 Ha (fixture density,
// CASTEP convention) but iter-2 jumps to ≈ 39.95 Ha because the rebuilt
// density misses the augmentation contribution → V_H, V_xc collapse →
// V_eff ≈ V_loc deep wells.
//
// Post-fix: iter-2 V_eff range stays within ±1 Ha of iter-1's, since
// ρ_total = ρ_PW + ρ_aug now feeds Poisson + XC consistently.
//
// Discriminator ratio: |ΔV| ≈ 31 Ha pre-fix vs ≈ 1 Ha post-fix → 30× margin.

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn iter2_v_eff_range_within_one_ha_of_iter1() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);

    // ---- Iter-1: V_eff from fixture density (CASTEP-augmented) ----
    let iter1_veff_built = state.build_v_eff_with_energy().expect("iter-1 build_v_eff");
    let iter1_range = v_eff_range(
        iter1_veff_built
            .v_eff()
            .as_ref()
            .expect("iter-1 V_eff present after build"),
    );

    // Drive iter-1 through the rest of the SCF cycle so iter-2 starts from a
    // chemrust-built density (the failure surface of Issue #8).
    let iter1_diag = iter1_veff_built
        .diagonalize(8, None)
        .expect("iter-1 diagonalize");
    let iter1_dens = iter1_diag
        .construct_density_off()
        .expect("iter-1 construct_density");
    let iter1_mixed = iter1_dens.mix();
    let iter2_init = match iter1_mixed
        .check(1e-8)
        .expect("iter-1 check")
    {
        chemrust_scf::CheckOutcome::Converged(_) => panic!(
            "iter-1 unexpectedly converged — iter-2 V_eff cannot be measured",
        ),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    // ---- Iter-2: V_eff from chemrust-built density (the bug surface) ----
    let iter2_veff_built = iter2_init
        .build_v_eff_with_energy()
        .expect("iter-2 build_v_eff");
    let iter2_range = v_eff_range(
        iter2_veff_built
            .v_eff()
            .as_ref()
            .expect("iter-2 V_eff present after build"),
    );

    println!("iter-1 V_eff range: {iter1_range:.4} Ha");
    println!("iter-2 V_eff range: {iter2_range:.4} Ha");
    println!("|Δrange|:           {:.4} Ha", (iter2_range - iter1_range).abs());

    // Anchor: pre-fix iter-1 ≈ 8.69 Ha, iter-2 ≈ 39.95 Ha.
    // Post-fix iter-2 should land within ±1 Ha of iter-1.
    assert!(
        (iter2_range - iter1_range).abs() < 1.0,
        "iter-2 V_eff range {iter2_range:.4} Ha differs from iter-1 {iter1_range:.4} Ha \
         by more than 1 Ha — augmentation density likely missing",
    );
}
