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

// ---------------------------------------------------------------------------
// Test: GPU aug density matches CPU aug density on Cu111_CO fixture
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn aug_density_gpu_matches_cpu_cu111_co() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    use chemrust_hamiltonian_core::{
        GVectorGrid, Pseudopotential,
        augment::beta_phi::compute_beta_phi,
        pseudopotential::HasAugmentationData,
    };
    use chemrust_scf::density::test_api::{
        build_q_sf_cache, compute_aug_density_fine, compute_aug_density_gpu,
        save_q_sf_cache_to_disk, load_q_sf_cache_from_disk,
    };
    use chemrust_scf::device::pcie::PcieAccount;
    use ndarray::Array2;
    use num_complex::Complex64;
    use std::sync::Arc;

    let fx = fixtures::cu111_co::fixture();
    let cell = &fx.bin.cell;
    let pots = &fx.pots;

    let wfc = fx.check.wavefunction.as_ref().expect(".check must have wavefunction");
    let [ngx, ngy, ngz] = wfc.grid;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);

    let [fgx, fgy, fgz] = fx.check.fine_grid.expect(".check must have fine_grid");
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    let kpt_block = &wfc.kpt_data[0];
    let k_cart = [0.0f64; 3]; // Gamma point

    // Build occupations from .check eigenvalues (Gaussian smearing, 0.1 eV)
    let n_bands = kpt_block.bands.len();
    let occupations: Vec<f64> = (0..n_bands).map(|_| 1.0).collect(); // unit occupations for test

    // Build beta_psi_per_ion: ⟨β_I|ψ_b⟩ for all ions
    let beta_psi_per_ion: Vec<Array2<Complex64>> = (0..cell.num_ions)
        .map(|ion_idx| {
            let species_idx = cell.ion_species[ion_idx];
            let symbol = &cell.species_symbols[species_idx];
            let pot = pots.get(symbol).expect("pot must exist");
            let aug: &dyn HasAugmentationData = match pot {
                Pseudopotential::Usp(d) => d,
                Pseudopotential::Recpot(_) => {
                    // No augmentation: return zero matrix
                    let n_exp = 0;
                    return Array2::zeros((n_exp, n_bands));
                }
            };
            let gmax_pp = pot.gmax();
            compute_beta_phi(kpt_block, aug, cell, ion_idx, &wave_grid, gmax_pp, k_cart)
                .expect("compute_beta_phi")
        })
        .collect();

    // CPU path — load from disk if available, compute and save otherwise.
    let cpu_cache_path = std::path::Path::new("/tmp/cu111_co_rho_aug_cpu.bin");
    let rho_aug_cpu = if cpu_cache_path.exists() {
        eprintln!("[CPU aug] loading from disk: {}", cpu_cache_path.display());
        let bytes = std::fs::read(cpu_cache_path).expect("read cpu cache");
        let n = bytes.len() / 8;
        let vals: Vec<f64> = (0..n).map(|i| {
            f64::from_le_bytes(bytes[i*8..(i+1)*8].try_into().unwrap())
        }).collect();
        // Shape is (ngx, ngy, ngz) C-order — same as fft_inverse_3d output.
        let [fgx, fgy, fgz] = [fgx, fgy, fgz];
        chemrust_hamiltonian_core::fft::RealGrid::from_inner(
            ndarray::Array3::from_shape_vec((fgx, fgy, fgz), vals).expect("shape")
        )
    } else {
        eprintln!("[CPU aug] computing (first run, will save to disk)...");
        let r = compute_aug_density_fine(
            &beta_psi_per_ion,
            &occupations,
            pots,
            cell,
            &fine_grid,
        ).expect("CPU aug density");
        let bytes: Vec<u8> = r.as_real_array().iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        std::fs::write(cpu_cache_path, &bytes).expect("write cpu cache");
        eprintln!("[CPU aug] saved to {}", cpu_cache_path.display());
        r
    };

    // GPU path — load cache from disk if available, build and save otherwise.
    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();
    let cache_path = std::path::Path::new("/tmp/cu111_co_q_sf_cache.bin");

    let cache = if cache_path.exists() {
        eprintln!("[QSfCache] loading from disk: {}", cache_path.display());
        let mut pcie = PcieAccount::default();
        let c = load_q_sf_cache_from_disk(cache_path, &stream, &mut pcie)
            .expect("load_q_sf_cache_from_disk");
        eprintln!("[QSfCache] loaded, H2D {} bytes", pcie.h2d_bytes);
        c
    } else {
        eprintln!("[QSfCache] building (first run, will save to disk)...");
        let mut pcie = PcieAccount::default();
        let c = build_q_sf_cache(pots, cell, &fine_grid, &stream, &mut pcie)
            .expect("build_q_sf_cache");
        eprintln!("[QSfCache] built, H2D {} bytes — saving to {}", pcie.h2d_bytes, cache_path.display());
        save_q_sf_cache_to_disk(&c, &stream, cache_path).expect("save_q_sf_cache_to_disk");
        c
    };

    let mut pcie2 = PcieAccount::default();
    let rho_aug_gpu = compute_aug_density_gpu(
        &cache,
        &beta_psi_per_ion,
        &occupations,
        &stream,
        &mut pcie2,
    ).expect("GPU aug density");

    // ---------------------------------------------------------------------------
    // Diagnostics: isolate where GPU and CPU diverge
    // ---------------------------------------------------------------------------
    {
        use chemrust_hamiltonian_core::{
            augment::q_apply::{apply_q_and_sf, compute_q_nm_flat},
        };
        use ndarray::Array2;

        // Pick ion 0 (first Cu ion)
        let ion_idx = 0;
        let species_idx = cache.ion_species[ion_idx];
        let bp = &beta_psi_per_ion[ion_idx];
        let ne = bp.shape()[0];
        let n_bands = occupations.len();

        // --- Diag 1: ω matrix ---
        let mut omega_cpu = Array2::<Complex64>::zeros((ne, ne));
        for n in 0..ne {
            for m in 0..ne {
                let mut acc = Complex64::ZERO;
                for b in 0..n_bands {
                    acc += occupations[b] * bp[[n, b]].conj() * bp[[m, b]];
                }
                omega_cpu[[n, m]] = acc;
            }
        }
        let omega_norm: f64 = omega_cpu.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        eprintln!("[Diag] ion={ion_idx} species={species_idx} ne={ne} ‖ω‖_∞ = {omega_norm:.4e}");

        // --- Diag 2: Q_{nm}(G) from compute_q_nm_flat vs apply_q_and_sf ---
        // Use pair (0,0) as a spot check.
        let pot = pots.get(&cell.species_symbols[species_idx]).unwrap();
        let aug = match pot {
            chemrust_hamiltonian_core::Pseudopotential::Usp(d) => d as &dyn chemrust_hamiltonian_core::pseudopotential::HasAugmentationData,
            _ => panic!("expected USP"),
        };

        // CPU reference: apply_q_and_sf with rho_nm = identity (pair 0,0 only)
        let mut rho_unit = Array2::<Complex64>::zeros((ne, ne));
        rho_unit[[0, 0]] = Complex64::new(1.0, 0.0);
        let q_ref = apply_q_and_sf(&rho_unit, aug, cell, ion_idx, &fine_grid, pot.gmax())
            .expect("apply_q_and_sf");
        let q_ref_sum: f64 = q_ref.iter().map(|c| c.norm()).sum();
        eprintln!("[Diag] apply_q_and_sf pair(0,0) Σ|Q| = {q_ref_sum:.4e}");

        // compute_q_nm_flat with origin cell (no SF)
        let mut cell_origin = cell.clone();
        for mut row in cell_origin.ionic_positions.rows_mut() { row.fill(0.0); }
        let (flat, _) = compute_q_nm_flat(aug, &cell_origin, ion_idx, &fine_grid)
            .expect("compute_q_nm_flat");
        let n_fine = fine_grid.grid().iter().product::<usize>();
        // pair (0,0) is at flat[0..n_fine]
        let q_flat_sum: f64 = flat[..n_fine].iter().map(|c| c.norm()).sum();
        eprintln!("[Diag] compute_q_nm_flat pair(0,0) Σ|Q| = {q_flat_sum:.4e}");

        // --- Diag 3: SF for ion 0 at G=0 ---
        let sf_host: Vec<chemrust_scf::device::CudaComplex> = stream.clone_dtoh(&cache.ion_sf[ion_idx].sf).unwrap();
        let sf_g0 = sf_host[0];
        eprintln!("[Diag] SF[ion=0, G=0] = ({:.6}, {:.6}i)  |SF| = {:.6}",
            sf_g0.x, sf_g0.y, (sf_g0.x*sf_g0.x + sf_g0.y*sf_g0.y).sqrt());

        // --- Diag 4: Q_{nm} GPU slice for species 0, pair (0,0), first 3 elements ---
        if let Some(se) = cache.species_entries[species_idx].as_ref() {
            let q_gpu_host: Vec<chemrust_scf::device::CudaComplex> = stream.clone_dtoh(&se.q_nm).unwrap();
            eprintln!("[Diag] Q_gpu[species={species_idx}] first 3 elements: {:?}",
                &q_gpu_host[..3.min(q_gpu_host.len())]);
            eprintln!("[Diag] Q_flat[pair(0,0)] first 3 elements: {:?}",
                flat[..3.min(flat.len())].iter().map(|c| (c.re, c.im)).collect::<Vec<_>>());
        }
    }

    // Compare
    let cpu_arr = rho_aug_cpu.as_real_array();
    let gpu_arr = rho_aug_gpu.as_real_array();

    assert_eq!(cpu_arr.shape(), gpu_arr.shape(), "shape mismatch");

    let max_diff = cpu_arr.iter().zip(gpu_arr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let cpu_sum: f64 = cpu_arr.iter().sum();
    let gpu_sum: f64 = gpu_arr.iter().sum();
    let sum_diff = (cpu_sum - gpu_sum).abs();

    println!("‖ρ_aug_gpu − ρ_aug_cpu‖_∞ = {:.4e}", max_diff);
    println!("∫ρ_aug_cpu = {:.6e}  ∫ρ_aug_gpu = {:.6e}  |Δ| = {:.4e}", cpu_sum, gpu_sum, sum_diff);

    assert!(
        max_diff < 1e-6,
        "‖ρ_aug_gpu − ρ_aug_cpu‖_∞ = {:.4e}, expected < 1e-6",
        max_diff,
    );
    assert!(
        sum_diff < 1e-4 * cpu_sum.abs().max(1.0),
        "∫ρ_aug sum diff = {:.4e}, expected < 1e-4 × |∫ρ_aug_cpu|",
        sum_diff,
    );
}
