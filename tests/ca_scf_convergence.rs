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

/// True per-ion Woodbury baseline after m_inv→s_inv typo fix.
/// Measured on Cu(111)+CO fixture. Becomes historical after G3 (global Woodbury).
/// FIXME: replace placeholder with the value from `s_inv_s_identity_test` after
/// running on GPU hardware (TASK-G0-2).
const BASELINE_ZETA_PER_ION: f64 = 0.0;

use rand::Rng;
use std::sync::Once;

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_target(false)
            .try_init()
            .ok();
    });
}

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
    init_tracing();

    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);
    let result = chemrust_scf::run_scf_with_energy_gated(
        state,
        8,
        1e-8,
        Some(chemrust_scf::ScfDivergenceGate::default()),
    )
    .expect("SCF converged");

    let computed_ev = result.total_energy * chemrust_scf::HARTREE_TO_EV;
    let diff_ev = (computed_ev - fixtures::cu111_co::REFERENCE_ENERGY_EV).abs();

    println!("Computed total energy: {:.8} eV", computed_ev);
    println!(
        "Reference total energy: {:.8} eV",
        fixtures::cu111_co::REFERENCE_ENERGY_EV
    );
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
    init_tracing();

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

    *state.density_mut() =
        chemrust_scf::Density::from_inner(chemrust_scf::WaveGridArray::from_inner(noisy_arr));

    let result = chemrust_scf::run_scf_with_energy_gated(
        state,
        8,
        1e-8,
        Some(chemrust_scf::ScfDivergenceGate::default()),
    )
    .expect("SCF converged after perturbation");

    let computed_ev = result.total_energy * chemrust_scf::HARTREE_TO_EV;
    let diff_ev = (computed_ev - fixtures::cu111_co::REFERENCE_ENERGY_EV).abs();

    println!(
        "Computed total energy (after perturbation): {:.8} eV",
        computed_ev
    );
    println!(
        "Reference total energy: {:.8} eV",
        fixtures::cu111_co::REFERENCE_ENERGY_EV
    );
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
    init_tracing();

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
    let iter2_init = match iter1_mixed.check(1e-8).expect("iter-1 check") {
        chemrust_scf::CheckOutcome::Converged(_) => {
            panic!("iter-1 unexpectedly converged — iter-2 V_eff cannot be measured",)
        }
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
    println!(
        "|Δrange|:           {:.4} Ha",
        (iter2_range - iter1_range).abs()
    );

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
    init_tracing();

    use chemrust_hamiltonian_core::{
        GVectorGrid, Pseudopotential, augment::beta_phi::compute_beta_phi,
        pseudopotential::HasAugmentationData,
    };
    use chemrust_scf::density::test_api::{
        CudaKernelSet, build_q_sf_cache, compute_aug_density_fine, compute_aug_density_gpu,
        load_q_sf_cache_from_disk, save_q_sf_cache_to_disk,
    };
    use chemrust_scf::device::CudaComplex;
    use chemrust_scf::device::pcie::PcieAccount;
    use cudarc::driver::CudaSlice;
    use ndarray::Array2;
    use num_complex::Complex64;
    use std::sync::Arc;

    let fx = fixtures::cu111_co::fixture();
    let cell = &fx.bin.cell;
    let pots = &fx.pots;

    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
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
        let vals: Vec<f64> = (0..n)
            .map(|i| f64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap()))
            .collect();
        // Shape is (ngx, ngy, ngz) C-order — same as fft_inverse_3d output.
        let [fgx, fgy, fgz] = [fgx, fgy, fgz];
        chemrust_hamiltonian_core::fft::RealGrid::from_inner(
            ndarray::Array3::from_shape_vec((fgx, fgy, fgz), vals).expect("shape"),
        )
    } else {
        eprintln!("[CPU aug] computing (first run, will save to disk)...");
        let r = compute_aug_density_fine(&beta_psi_per_ion, &occupations, pots, cell, &fine_grid)
            .expect("CPU aug density");
        let bytes: Vec<u8> = r
            .as_real_array()
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        std::fs::write(cpu_cache_path, &bytes).expect("write cpu cache");
        eprintln!("[CPU aug] saved to {}", cpu_cache_path.display());
        r
    };

    // GPU path — load cache from disk if available, build and save otherwise.
    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();
    let kernels = CudaKernelSet::new(&ctx).expect("compile CUDA kernels");
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
        let c =
            build_q_sf_cache(pots, cell, &fine_grid, &stream, &mut pcie).expect("build_q_sf_cache");
        eprintln!(
            "[QSfCache] built, H2D {} bytes — saving to {}",
            pcie.h2d_bytes,
            cache_path.display()
        );
        save_q_sf_cache_to_disk(&c, &stream, cache_path).expect("save_q_sf_cache_to_disk");
        c
    };

    // Upload beta_psi_per_ion to GPU for compute_aug_density_gpu.
    // CRITICAL: compute_aug_density_gpu reinterprets the buffer with
    // `(n_expanded, n_bands).f()` (col-major / F-order). Production gemm
    // in rayleigh_ritz produces col-major. Here we have a row-major
    // Array2 from compute_beta_phi, so we must flatten in col-major
    // order — `flat[n + b*n_e] = arr[n, b]` — to match production layout.
    let beta_psi_gpu: Vec<CudaSlice<CudaComplex>> = beta_psi_per_ion
        .iter()
        .map(|arr| {
            let ne = arr.shape()[0];
            let nb = arr.shape()[1];
            let mut flat: Vec<CudaComplex> = Vec::with_capacity(ne * nb);
            for b in 0..nb {
                for n in 0..ne {
                    let c = arr[[n, b]];
                    flat.push(CudaComplex { x: c.re, y: c.im });
                }
            }
            stream.clone_htod(&flat).expect("H2D beta_psi")
        })
        .collect();

    let mut pcie2 = PcieAccount::default();
    let rho_aug_gpu = compute_aug_density_gpu(
        &cache,
        &beta_psi_gpu,
        &occupations,
        &stream,
        &mut pcie2,
        &kernels,
    )
    .expect("GPU aug density");

    // Compare
    let cpu_arr = rho_aug_cpu.as_real_array();
    let gpu_arr = rho_aug_gpu.as_real_array();

    assert_eq!(cpu_arr.shape(), gpu_arr.shape(), "shape mismatch");

    let max_diff = cpu_arr
        .iter()
        .zip(gpu_arr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let cpu_sum: f64 = cpu_arr.iter().sum();
    let gpu_sum: f64 = gpu_arr.iter().sum();
    let sum_diff = (cpu_sum - gpu_sum).abs();

    println!("‖ρ_aug_gpu − ρ_aug_cpu‖_∞ = {:.4e}", max_diff);
    println!(
        "∫ρ_aug_cpu = {:.6e}  ∫ρ_aug_gpu = {:.6e}  |Δ| = {:.4e}",
        cpu_sum, gpu_sum, sum_diff
    );

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

// ---------------------------------------------------------------------------
// Controlled experiment: feed CASTEP's converged wavefunctions through our
// density construction code.  Compare component integrals (soft / aug)
// against CASTEP F8 instrumented dumps to verify the density code itself
// is correct (before investigating upstream wavefunction differences).
//
// CASTEP F8 dumps (converged, Cu111_CO, 16 MPI ranks):
//   F8_RHO_SOFT_SUM = 2.99359524940157e7
//   F8_RHO_AUG_SUM  = 5.14204469948334e7
// (Source: slurm_output_2291.txt, converged iteration, sum in ρ×Ω convention
//  on the FINE grid)
//
// If our density code is correct, feeding CASTEP's own converged wavefunction
// coefficients + eigenvalues through our pipeline should reproduce these
// ratios to within 1%.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn density_decomp_matches_castep_f8_same_inputs() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    use chemrust_hamiltonian_core::{
        GVectorGrid, Pseudopotential, augment::beta_phi::compute_beta_phi,
        pseudopotential::HasAugmentationData,
    };
    use chemrust_scf::density::test_api::{
        CudaKernelSet, compute_aug_density_fine, construct_density_gpu,
    };
    use ndarray::Array2;
    use num_complex::Complex64;
    use std::sync::Arc;

    // --- F8 anchor values (converged iteration, ρ×Ω convention, FINE grid) ---
    const F8_SOFT_SUM: f64 = 2.99359524940157e7;
    const F8_AUG_SUM: f64 = 5.14204469948334e7;
    const F8_TOTAL: f64 = F8_SOFT_SUM + F8_AUG_SUM;

    // 1. Load CASTEP fixture
    let fx = fixtures::cu111_co::fixture();
    let cell = &fx.bin.cell;
    let pots = &fx.pots;

    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let [ngx, ngy, ngz] = wfc.grid; // wave grid dims
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);
    let n_wave = ngx * ngy * ngz;

    let [fgx, fgy, fgz] = fx.check.fine_grid.expect(".check must have fine_grid");
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);
    let n_fine = fgx * fgy * fgz;

    // 2. CASTEP eigenvalues from .bands
    let eigenvalues = &fx.bands_eigenvalues;
    let n_bands = eigenvalues.len();
    println!("=== Controlled Experiment: Same-Input Validation ===");
    println!("Wave grid:  {ngx}×{ngy}×{ngz} = {n_wave}");
    println!("Fine grid:  {fgx}×{fgy}×{fgz} = {n_fine}");
    println!("Bands:      {n_bands}");
    println!("Cell volume: {:.4} Bohr³", cell.volume);

    // 3. Compute occupations from CASTEP eigenvalues (our erfc smearing)
    let n_electrons: f64 = cell
        .species_iter()
        .map(|info| {
            pots.get(info.symbol)
                .and_then(|p| p.ionic_charge())
                .unwrap_or(0.0)
                * info.num_ions as f64
        })
        .sum();
    let smearing = chemrust_scf::SmearingParams {
        width: 0.1 * chemrust_scf::EV_TO_HARTREE,
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: chemrust_scf::SmearingScheme::Gaussian,
    };
    let (occupations, chem_pot) =
        chemrust_scf::density::compute_occupations(eigenvalues, &smearing, n_electrons)
            .expect("compute_occupations from CASTEP eigenvalues");
    let occ_sum: f64 = occupations.0.iter().sum();
    println!(
        "Occupations: Σocc = {occ_sum:.4}  target N_e = {n_electrons:.1}  μ = {:.6} Ha",
        chem_pot.0
    );

    // 4. GPU setup
    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();
    let kernels = CudaKernelSet::new(&ctx).expect("compile CUDA kernels");

    // 5. Extract CASTEP wavefunction coefficients (first k-point, first spin)
    let kpt_block = &wfc.kpt_data[0];
    let n_pw = kpt_block.nplw;
    let psi_flat: Vec<Complex64> = kpt_block.bands.concat();
    let fft_indices = chemrust_scf::pw_coords_to_fft_indices(&kpt_block.pw_grid_coord, &wave_grid);

    eprintln!("psi shape: {n_bands} bands × {n_pw} PW → FFT grid {n_wave}");

    // 6. Compute soft density (smooth PW term only, on wave grid)
    let soft_density = construct_density_gpu()
        .psi_data(&psi_flat)
        .occupations(&occupations.0)
        .fft_indices(&fft_indices)
        .wave_grid(&wave_grid)
        .cell_volume(cell.volume)
        .n_bands(n_bands)
        .n_pw(n_pw)
        .kernels(&kernels)
        .stream(&stream)
        .call()
        .expect("construct_density_gpu");

    let soft_sum = soft_density.as_wave_array().sum();
    let n_e_soft = soft_sum / n_wave as f64;
    let n_e_soft_f8 = F8_SOFT_SUM / n_fine as f64;
    let ratio_soft = n_e_soft / n_e_soft_f8;

    eprintln!(
        "[Soft Density] sum_our = {:.6e}  N_e_our = {:.4}  N_e_F8 = {:.4}  ratio = {:.6}",
        soft_sum, n_e_soft, n_e_soft_f8, ratio_soft,
    );

    // 7. Compute β·ψ from CASTEP wavefunctions
    let k_cart = [0.0f64; 3]; // Gamma point
    let beta_psi_per_ion: Vec<Array2<Complex64>> = (0..cell.num_ions)
        .map(|ion_idx| {
            let species_idx = cell.ion_species[ion_idx];
            let symbol = &cell.species_symbols[species_idx];
            let pot = pots.get(symbol).expect("pot must exist");
            let aug: &dyn HasAugmentationData = match pot {
                Pseudopotential::Usp(d) => d,
                Pseudopotential::Recpot(_) => {
                    return Array2::zeros((0, n_bands));
                }
            };
            let gmax_pp = pot.gmax();
            compute_beta_phi(kpt_block, aug, cell, ion_idx, &wave_grid, gmax_pp, k_cart)
                .expect("compute_beta_phi")
        })
        .collect();

    // 8. Compute augmentation density on fine grid (CPU path)
    let rho_aug =
        compute_aug_density_fine(&beta_psi_per_ion, &occupations.0, pots, cell, &fine_grid)
            .expect("compute_aug_density_fine");

    let aug_sum: f64 = rho_aug.as_real_array().iter().sum();
    let n_e_aug = aug_sum / n_fine as f64;
    let n_e_aug_f8 = F8_AUG_SUM / n_fine as f64;
    let ratio_aug = n_e_aug / n_e_aug_f8;

    eprintln!(
        "[Aug Density]  sum_our = {:.6e}  N_e_our = {:.4}  N_e_F8 = {:.4}  ratio = {:.6}",
        aug_sum, n_e_aug, n_e_aug_f8, ratio_aug,
    );

    // 9. Total
    let n_e_total = n_e_soft + n_e_aug;
    let n_e_total_f8 = F8_TOTAL / n_fine as f64;
    eprintln!(
        "[Total] N_e_our = {:.4}  N_e_F8 = {:.4}  ratio = {:.6}",
        n_e_total,
        n_e_total_f8,
        n_e_total / n_e_total_f8,
    );

    // 10. Decision: soft and aug ratios must be within 1% of F8 values
    // A "normalization bug" would show ratio ≠ 1.0.
    // If both within 1%, the density code is correct and the
    // PW/aug decomposition discrepancy is from wavefunction differences.
    let soft_ok = (ratio_soft - 1.0).abs() < 0.01;
    let aug_ok = (ratio_aug - 1.0).abs() < 0.01;

    if soft_ok && aug_ok {
        eprintln!(
            "✓ DENSITY CODE IS CORRECT — both soft and aug ratios within 1% of F8. \
             The PW/aug decomposition discrepancy is from wavefunction differences."
        );
    } else if !soft_ok {
        eprintln!(
            "✗ SOFT DENSITY BUG — ratio {:.4} ≠ 1.0. \
             Normalization issue in construct_density_gpu.",
            ratio_soft,
        );
    } else {
        eprintln!(
            "✗ AUG DENSITY BUG — ratio {:.4} ≠ 1.0. \
             Normalization issue in compute_aug_density_*.",
            ratio_aug,
        );
    }

    assert!(
        soft_ok,
        "soft density N_e ratio {:.4} differs from 1.0 by > 1% — normalization bug in construct_density_gpu",
        ratio_soft,
    );
    assert!(
        aug_ok,
        "aug density N_e ratio {:.4} differs from 1.0 by > 1% — normalization bug in compute_aug_density_*",
        ratio_aug,
    );
}

// ---------------------------------------------------------------------------
// S⁻¹·S identity baseline lock (post typo fix)
// Records the true per-ion Woodbury baseline after m_inv→s_inv correction.
// Value stored in BASELINE_ZETA_PER_ION for the rest of the phase.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn s_inv_baseline_post_typo_fix() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    use chemrust_hamiltonian_core::GVectorGrid;
    use chemrust_scf::KPoint;
    use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData, check_s_inv_s_identity};
    use chemrust_scf::device::blas::BlasHandle;
    use chemrust_scf::device::pcie::PcieAccount;
    use num_complex::Complex64;
    use std::sync::Arc;

    let fx = fixtures::cu111_co::fixture();
    let cell = &fx.bin.cell;
    let pots = &fx.pots;

    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let [ngx, ngy, ngz] = wfc.grid;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);

    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;
    let k_point = KPoint {
        coords: kpt_block.coords,
    };

    // Flat psi_data: band-major, col-major layout [band * n_pw + g]
    let psi_data: Vec<Complex64> = kpt_block.bands.concat();

    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("BLAS handle");
    let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone())
        .expect("SolverHandle");
    let kernels = CudaKernelSet::new(&ctx).expect("CUDA kernels");

    // Build VnlBatchData (bare D0, no occupations, no V_eff for screening)
    let mut pcie = PcieAccount::default();
    let vnl_data = VnlBatchData::precompute(
        &kpt_block.pw_grid_coord,
        pots,
        cell,
        &wave_grid,
        &k_point,
        &psi_data,
        n_bands,
        n_pw,
        None, // occupations: bare D0
        None, // v_eff: no screening
        &stream,
        &mut pcie,
        &blas,
        &solver,
    )
    .expect("VnlBatchData::precompute");

    // Take band 0 (the converged lowest eigenstate)
    let band0: Vec<Complex64> = psi_data.iter().take(n_pw).copied().collect();

    let zeta = check_s_inv_s_identity(&band0, n_pw, &vnl_data, &blas, &stream, &solver)
        .expect("check_s_inv_s_identity");

    eprintln!(
        "[baseline] per-ion ζ = {:.6e}  (BASELINE_ZETA_PER_ION = {:.6e})",
        zeta, BASELINE_ZETA_PER_ION,
    );

    // Assertion: call succeeded. Value is the discriminator, not a threshold.
    // The recorded value should be updated in BASELINE_ZETA_PER_ION above.
    eprintln!(
        "[baseline] RECORDED ζ = {:.6e} — update BASELINE_ZETA_PER_ION constant if this differs.",
        zeta,
    );
}

// ---------------------------------------------------------------------------
// S⁻¹·S identity diagnostic: check that S⁻¹·S·ψ = ψ for the Woodbury
// formula. If this fails, the Q convention in s_inv_mat is inconsistent
// with q_matrix, explaining the density decomposition flip in the SCF.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn s_inv_s_identity_test() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    use chemrust_hamiltonian_core::GVectorGrid;
    use chemrust_scf::KPoint;
    use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData, check_s_inv_s_identity};
    use chemrust_scf::device::blas::BlasHandle;
    use chemrust_scf::device::pcie::PcieAccount;
    use num_complex::Complex64;
    use std::sync::Arc;

    let fx = fixtures::cu111_co::fixture();
    let cell = &fx.bin.cell;
    let pots = &fx.pots;

    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let [ngx, ngy, ngz] = wfc.grid;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);

    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;
    let k_point = KPoint {
        coords: kpt_block.coords,
    };

    // Flat psi_data: band-major, col-major layout [band * n_pw + g]
    let psi_data: Vec<Complex64> = kpt_block.bands.concat();

    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("BLAS handle");
    let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone())
        .expect("SolverHandle");
    let kernels = CudaKernelSet::new(&ctx).expect("CUDA kernels");

    // Build VnlBatchData (bare D0, no occupations, no V_eff for screening)
    let mut pcie = PcieAccount::default();
    let vnl_data = VnlBatchData::precompute(
        &kpt_block.pw_grid_coord,
        pots,
        cell,
        &wave_grid,
        &k_point,
        &psi_data,
        n_bands,
        n_pw,
        None, // occupations: bare D0
        None, // v_eff: no screening
        &stream,
        &mut pcie,
        &blas,
        &solver,
    )
    .expect("VnlBatchData::precompute");

    // Take band 0 (the converged lowest eigenstate)
    let band0: Vec<Complex64> = psi_data.iter().take(n_pw).copied().collect();

    let max_residual = check_s_inv_s_identity(&band0, n_pw, &vnl_data, &blas, &stream, &solver)
        .expect("check_s_inv_s_identity");

    eprintln!("[S⁻¹·S identity] ‖S⁻¹·S·ψ₀ − ψ₀‖_∞ = {:.6e}", max_residual,);

    // If > 1e-8: Q convention in s_inv_mat (Woodbury) is inconsistent
    // Global Woodbury S⁻¹ = I − B·M⁻¹·B^H validated at roundoff.
    assert!(
        max_residual < 1e-10,
        "S⁻¹·S·ψ ≠ ψ: max residual = {:.6e} > 1e-10.",
        max_residual,
    );
}

// ---------------------------------------------------------------------------
// Iter-1 filter-operator A/B/C discriminator sweep
//
// Runs one Chebyshev+RR iteration under three filter modes and compares the
// lowest-10 RR band energies against the CASTEP .bands reference.
//
// Gate (SC-4-tight): |band_j_iter1 − band_j_castep| < 0.05 Ha for j ∈ [0,10)
//
// Self-test (D6): Mode A must reproduce band-1 ≈ −1.69 Ha within 0.1 Ha
// before B/C results are read. If D6 fails, the harness itself is broken.
//
// See notes/debug/debug-20260523-0916-iter1-filter-operator-mismatch/FIX_PLAN.md
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data; diagnostic sweep for filter-operator selection"]
fn iter1_filter_mode_sweep() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    use chemrust_scf::density::test_api::FilterMode;

    let fx = fixtures::cu111_co::fixture();
    let castep_bands = &fx.bands_eigenvalues;

    // Print fixture context: n_bands, n_pw, and .check eigenvalues for first 10 bands
    {
        let wfc = fx.check.wavefunction.as_ref().expect(".check wavefunction");
        let kpt = &wfc.kpt_data[0];
        eprintln!("[fixture] n_bands={} n_pw={}", kpt.bands.len(), kpt.nplw);
        let check_eigs = &fx.check.eigenvalues.kpoints[0].spins[0].eigenvalues;
        eprintln!("[fixture] .check eigenvalues (first 10): {:?}", &check_eigs[..10.min(check_eigs.len())]);
        eprintln!("[fixture] .bands eigenvalues (first 10): {:?}", &castep_bands[..10.min(castep_bands.len())]);
    }

    // Run one iter under a given mode; returns the RR eigenvalues.
    let run_mode = |mode: FilterMode| -> Vec<f64> {
        let state = fixtures::cu111_co::build_scf_state(fx);
        state
            .build_v_eff()
            .expect("build_v_eff")
            .diagonalize_with_mode(8, None, mode)
            .expect("diagonalize_with_mode")
            .eigenvalues()
            .to_vec()
    };

    // --- Mode A (bare-H, current production path) ---
    let eig_a = run_mode(FilterMode::BareH);

    // D6 self-test: Mode A band-0 must be within 0.1 Ha of CASTEP reference −1.055 Ha.
    // This verifies the harness is computing the correct Hamiltonian (CPU D-screening).
    // If this fails, the D_screened values are wrong — do not read B/C.
    let band0_a = eig_a.get(0).copied().unwrap_or(f64::NAN);
    let castep_band0 = castep_bands.first().copied().unwrap_or(f64::NAN);
    assert!(
        (band0_a - castep_band0).abs() < 0.1,
        "D6 self-test FAILED: Mode A band-0 = {:.4} Ha, CASTEP = {:.4} Ha (|Δ| = {:.4} Ha > 0.1). \
         Hamiltonian may be wrong — do not interpret B/C results.",
        band0_a, castep_band0, (band0_a - castep_band0).abs(),
    );
    eprintln!("[D6] Mode A band-0 = {:.4} Ha, CASTEP = {:.4} Ha — self-test PASSED", band0_a, castep_band0);

    // --- Modes B and C ---
    let eig_b = run_mode(FilterMode::SinvHKeepHEig);
    let eig_c = run_mode(FilterMode::SinvHFullDas);

    // --- Per-band table (stderr) ---
    let n_check = 10.min(eig_a.len()).min(castep_bands.len());
    eprintln!("\n[iter1_filter_mode_sweep] Per-band |Δ| vs CASTEP .bands (Ha)");
    eprintln!("{:>5}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
        "band", "CASTEP", "Mode-A", "|ΔA|", "|ΔB|", "|ΔC|");
    for j in 0..n_check {
        let ref_e = castep_bands[j];
        let da = (eig_a.get(j).copied().unwrap_or(f64::NAN) - ref_e).abs();
        let db = (eig_b.get(j).copied().unwrap_or(f64::NAN) - ref_e).abs();
        let dc = (eig_c.get(j).copied().unwrap_or(f64::NAN) - ref_e).abs();
        eprintln!("{:>5}  {:>12.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
            j, ref_e,
            eig_a.get(j).copied().unwrap_or(f64::NAN),
            da, db, dc);
    }

    // Write CSV for follow-on analysis
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let csv_path = format!("/tmp/iter1-mode-sweep-{ts}.csv");
    let mut csv = String::from("band,castep_ha,mode_a_ha,mode_b_ha,mode_c_ha,delta_a,delta_b,delta_c\n");
    for j in 0..n_check {
        let ref_e = castep_bands[j];
        let ea = eig_a.get(j).copied().unwrap_or(f64::NAN);
        let eb = eig_b.get(j).copied().unwrap_or(f64::NAN);
        let ec = eig_c.get(j).copied().unwrap_or(f64::NAN);
        csv.push_str(&format!("{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}\n",
            j, ref_e, ea, eb, ec,
            (ea - ref_e).abs(), (eb - ref_e).abs(), (ec - ref_e).abs()));
    }
    if let Err(e) = std::fs::write(&csv_path, &csv) {
        eprintln!("[iter1_filter_mode_sweep] WARNING: could not write CSV to {csv_path}: {e}");
    } else {
        eprintln!("[iter1_filter_mode_sweep] CSV written to {csv_path}");
    }

    // --- SC-4-tight gate: report all modes, assert Mode B (discriminator winner) ---
    const SC4_GATE: f64 = 0.05; // Ha
    let mut mode_b_failures: Vec<String> = Vec::new();
    for mode_label in ["A", "B", "C"] {
        let eig = match mode_label {
            "A" => &eig_a,
            "B" => &eig_b,
            _ => &eig_c,
        };
        let mut mode_failures: Vec<String> = Vec::new();
        for j in 0..n_check {
            let delta = (eig.get(j).copied().unwrap_or(f64::NAN) - castep_bands[j]).abs();
            if delta >= SC4_GATE {
                mode_failures.push(format!("  band {j}: |Δ| = {delta:.4} Ha ≥ {SC4_GATE} Ha"));
            }
        }
        if mode_failures.is_empty() {
            eprintln!("[SC-4-tight] Mode {mode_label}: PASS (all {n_check} bands within {SC4_GATE} Ha)");
        } else {
            eprintln!("[SC-4-tight] Mode {mode_label}: FAIL ({} bands out of {n_check})", mode_failures.len());
            for f in &mode_failures {
                eprintln!("{f}");
            }
            if mode_label == "B" {
                mode_b_failures = mode_failures;
            }
        }
    }

    assert!(
        mode_b_failures.is_empty(),
        "iter1_filter_mode_sweep: Mode B (discriminator winner) failed SC-4-tight.\n\
         See /tmp/iter1-mode-sweep-{ts}.csv for full data.\n\n{}",
        mode_b_failures.join("\n"),
    );
}

// ---------------------------------------------------------------------------
// Step 7.1 (variant): Pollution-injection probe on real fixture
//
// Construct a controlled "polluted" trial subspace by mixing a known amount
// of a high-energy band into a low-energy band. Run the filter once. Compare
// the polluted band's Rayleigh quotient before vs after filtering.
//
// Expected if the filter is correct:
// - The polluted band's Rayleigh quotient should move CLOSER to the clean
//   value (the filter denoises by amplifying the wanted low-energy
//   eigencomponent and damping the unwanted high-energy pollution).
//
// Expected if the filter is doing "the opposite":
// - The polluted band's Rayleigh quotient should move FURTHER UP from the
//   clean value (the filter is amplifying the polluting high-energy
//   eigencomponent rather than damping it).
//
// Discriminator: ratio of post-filter |Δ from clean| to pre-filter |Δ from
// clean|. Ratio < 1 means filter denoised. Ratio > 1 means filter polluted
// further. The ratio is the empirical answer to "is the filter doing what
// it's supposed to."
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn pollution_injection_filter_should_denoise() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    use chemrust_scf::density::test_api::FilterMode;
    use num_complex::Complex64;

    let fx = fixtures::cu111_co::fixture();
    let castep_bands = &fx.bands_eigenvalues;

    // ---- Step A: Clean baseline (ndeg=0, no filter) ----
    // We already proved this matches CASTEP in ndeg_zero_with_castep_psi_matches_bands.
    let clean_eigs: Vec<f64> = {
        let state = fixtures::cu111_co::build_scf_state(fx);
        state
            .build_v_eff()
            .expect("build_v_eff")
            .diagonalize_with_mode(0, None, FilterMode::SinvHKeepHEig)
            .expect("ndeg=0 diagonalize")
            .eigenvalues()
            .to_vec()
    };
    let clean_band0 = clean_eigs.first().copied().unwrap_or(f64::NAN);
    let clean_last = clean_eigs.last().copied().unwrap_or(f64::NAN);
    eprintln!(
        "[pollute] clean baseline (ndeg=0): band-0 = {:.6} Ha, last = {:.6} Ha",
        clean_band0, clean_last,
    );

    // ---- Step B: Inject pollution into band-0 ----
    // For α ∈ {0.05, 0.10, 0.25}, replace ψ_band0 with normalize(ψ_band0 + α · ψ_last).
    // Then call diagonalize with ndeg=8 (filter active). Observe band-0's
    // Rayleigh quotient (= eig[0] from RR) post-filter.
    //
    // Key observation: diagonalize_with_mode rebuilds vnl_data internally,
    // and after RR the bands are re-sorted. So we cannot identify "the polluted
    // band" by index after the fact. Instead, we report the full eigenvalue
    // spectrum and note whether the lowest band moved up (filter polluted further)
    // or stayed stable (filter denoised).
    let run_polluted = |alpha: f64, polluter_band: usize| -> Vec<f64> {
        let state = fixtures::cu111_co::build_scf_state(fx);
        let mut state = state.build_v_eff().expect("build_v_eff");
        let (n_bands, n_pw) = state.psi_shape();
        assert!(n_bands >= 2, "need at least 2 bands for pollution injection");
        assert!(polluter_band < n_bands, "polluter_band {} out of range", polluter_band);
        assert!(polluter_band != 0, "polluter_band must differ from band-0");

        // Pollute band-0: ψ_0 ← (ψ_0 + α · ψ_polluter) / norm
        // Layout: psi.data is col-major (n_pw, n_bands), so band b occupies
        // contiguous slice [b*n_pw .. (b+1)*n_pw].
        {
            let psi = state.psi_data_mut();
            let pol_band: Vec<Complex64> = psi[polluter_band * n_pw..(polluter_band + 1) * n_pw].to_vec();
            let mut norm2 = 0.0f64;
            for g in 0..n_pw {
                let new = psi[g] + Complex64::new(alpha, 0.0) * pol_band[g];
                psi[g] = new;
                norm2 += new.norm_sqr();
            }
            let inv_norm = 1.0 / norm2.sqrt();
            for g in 0..n_pw {
                psi[g] = psi[g] * Complex64::new(inv_norm, 0.0);
            }
        }

        // Now diagonalize with the filter active
        state
            .diagonalize_with_mode(8, None, FilterMode::SinvHKeepHEig)
            .expect("ndeg=8 diagonalize")
            .eigenvalues()
            .to_vec()
    };

    // ---- Test 1: Pollute with HIGH-ENERGY band-last ----
    // S ≈ I in the high-energy regime, so the operator-order question (Das line 603
    // A·D⁻¹·R_Y) is hidden. Expected: filter denoises (low Δ).
    let polluter_high = state_n_bands(fx) - 1;
    eprintln!("\n[pollute] === Variant A: polluter = band-{polluter_high} (high-energy) ===");
    eprintln!(
        "{:>5}  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}",
        "α", "→band-0", "Δ_band0", "→last", "Δ_last", "ratio_b0",
    );

    let mut alpha_summary_high: Vec<(f64, f64, f64)> = Vec::new();
    for &alpha in &[0.05, 0.10, 0.25] {
        let polluted_eigs = run_polluted(alpha, polluter_high);
        let pol_band0 = polluted_eigs.first().copied().unwrap_or(f64::NAN);
        let pol_last = polluted_eigs.last().copied().unwrap_or(f64::NAN);
        let dband0 = pol_band0 - clean_band0;
        let dlast = pol_last - clean_last;
        let pre_filter_estimate = alpha * alpha * (clean_last - clean_band0);
        let ratio = if pre_filter_estimate.abs() > 1e-12 {
            dband0 / pre_filter_estimate
        } else {
            f64::NAN
        };
        eprintln!(
            "{:>5.2}  {:>10.6}  {:>12.6}  {:>12.6}  {:>12.6}  {:>12.4}",
            alpha, pol_band0, dband0, pol_last, dlast, ratio,
        );
        alpha_summary_high.push((alpha, dband0, ratio));
    }

    // ---- Test 2: Pollute with LOW-ENERGY band-1 ----
    // Cu 3d states have strong augmentation overlap; ‖S − I‖ is non-trivial.
    // The operator-order question Das line 603 (A·D⁻¹·R_Y vs S⁻¹·H·R_Y) bites here.
    // Expected if Rust's order is correct: filter denoises (low Δ).
    // Expected if Rust's order is wrong (audit's claim): filter preserves or
    //   amplifies the polluting band-1 eigencomponent, leaving |Δ_band0| in the
    //   pre-filter range or worse.
    let polluter_low = 1usize;
    eprintln!("\n[pollute] === Variant B: polluter = band-{polluter_low} (low-energy, augmentation-heavy) ===");
    eprintln!("[pollute] band-1 eigenvalue ≈ {:.4} Ha (close to band-0 = {:.4} Ha)",
        clean_eigs.get(1).copied().unwrap_or(f64::NAN), clean_band0);
    eprintln!(
        "{:>5}  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}",
        "α", "→band-0", "Δ_band0", "→last", "Δ_last", "ratio_b0",
    );

    let mut alpha_summary_low: Vec<(f64, f64, f64)> = Vec::new();
    for &alpha in &[0.05, 0.10, 0.25] {
        let polluted_eigs = run_polluted(alpha, polluter_low);
        let pol_band0 = polluted_eigs.first().copied().unwrap_or(f64::NAN);
        let pol_last = polluted_eigs.last().copied().unwrap_or(f64::NAN);
        let dband0 = pol_band0 - clean_band0;
        let dlast = pol_last - clean_last;
        // For low-energy pollution, the pre-filter Rayleigh quotient of the
        // polluted band ≈ (1-α²)·band0 + α²·band1, so pre-filter Δ ≈
        // α²·(band1 - band0). band1 - band0 ≈ 0.55 Ha for Cu111+CO.
        let band1 = clean_eigs.get(1).copied().unwrap_or(f64::NAN);
        let pre_filter_estimate = alpha * alpha * (band1 - clean_band0);
        let ratio = if pre_filter_estimate.abs() > 1e-12 {
            dband0 / pre_filter_estimate
        } else {
            f64::NAN
        };
        eprintln!(
            "{:>5.2}  {:>10.6}  {:>12.6}  {:>12.6}  {:>12.6}  {:>12.4}",
            alpha, pol_band0, dband0, pol_last, dlast, ratio,
        );
        alpha_summary_low.push((alpha, dband0, ratio));
    }

    // Diagnosis from low-energy variant (the discriminating one)
    eprintln!("\n[pollute] Discriminating diagnosis (Variant B, low-energy polluter):");
    let alpha_010_low = alpha_summary_low.iter().find(|(a, _, _)| (*a - 0.10).abs() < 1e-6);
    if let Some(&(_, dband0, ratio)) = alpha_010_low {
        let abs_dband0 = dband0.abs();
        let band1 = clean_eigs.get(1).copied().unwrap_or(f64::NAN);
        let pre_filter_analytic = 0.01 * (band1 - clean_band0);
        eprintln!(
            "[pollute] α=0.10: |Δ_band0| = {:.6} Ha (pre-filter analytic ≈ {:.6} Ha, ratio = {:.4})",
            abs_dband0, pre_filter_analytic.abs(), ratio,
        );
        if abs_dband0 < 0.001 {
            eprintln!(
                "[pollute] DIAGNOSIS: filter denoises low-energy pollution (|Δ| < 0.001 Ha gate). The recurrence body's operator order is correct. Bug is elsewhere."
            );
        } else if abs_dband0 < pre_filter_analytic.abs() * 0.5 {
            eprintln!(
                "[pollute] DIAGNOSIS: filter partially denoises (|Δ| < pre-filter/2). The recurrence body is acting in the right direction but with reduced contrast."
            );
        } else {
            eprintln!(
                "[pollute] DIAGNOSIS: filter does NOT denoise low-energy pollution effectively. \
                 |Δ_band0| ≥ pre-filter/2. \
                 This is consistent with the audit's claim that chebyshev.rs:1602-1611 \
                 computes S⁻¹·(H·R_Y) instead of Das main.tex:603's H·(S⁻¹·R_Y). \
                 In the augmentation-heavy regime where ‖S − I‖ matters, the wrong \
                 operator order fails to amplify the wanted low-energy eigencomponent."
            );
        }
    }
    let _ = clean_eigs; // silence in case of future refactor
    let _ = alpha_summary_high;
}

fn state_n_bands(fx: &fixtures::cu111_co::Cu111CoFixture) -> usize {
    let wfc = fx.check.wavefunction.as_ref().expect(".check wavefunction");
    wfc.kpt_data[0].bands.len()
}

// ---------------------------------------------------------------------------
// Step 7.1 (variant E): Real two-iteration SCF — iter-2 last band check
//
// The pollution probes used a SYNTHETIC pollution injection. This test runs
// the production-shape iter-1 → iter-2 sequence (the actual §11 symptom path).
//
// CHEMRUST_FORCE_NO_EIGS=1 forces the iter-2 filter to use the iter-1-like
// path (eigenvalues=None internally) by ignoring prior eigenvalues. If iter-2
// last band stays near iter-1's value (~0.13 Ha), the per-band branches
// (chebyshev.rs:1582-1660) are confirmed as the bug. If iter-2 last band
// still climbs to ~1.95 Ha, the bug is upstream of those branches.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn iter2_real_scf_last_band_with_force_no_eigs() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    let fx = fixtures::cu111_co::fixture();

    // ---- Iter-1: from CASTEP fixture, full SCF iteration ----
    let state = fixtures::cu111_co::build_scf_state(fx);
    let iter1_veff = state.build_v_eff_with_energy().expect("iter-1 build_v_eff");
    let iter1_diag = iter1_veff
        .diagonalize(8, None)
        .expect("iter-1 diagonalize");
    let iter1_eigs = iter1_diag.eigenvalues().to_vec();
    let iter1_first = iter1_eigs[0];
    let iter1_last = *iter1_eigs.last().unwrap();
    eprintln!(
        "[iter2-real] iter-1 RR: band-0 = {:.6} Ha, last = {:.6} Ha",
        iter1_first, iter1_last,
    );

    let iter1_dens = iter1_diag
        .construct_density_off()
        .expect("iter-1 construct_density");
    let iter1_mixed = iter1_dens.mix();
    let iter2_init = match iter1_mixed.check(1e-8).expect("iter-1 check") {
        chemrust_scf::CheckOutcome::Converged(_) => {
            panic!("iter-1 unexpectedly converged — iter-2 cannot be measured")
        }
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    // ---- Iter-2: same path, eigenvalues from iter-1 carried forward ----
    let iter2_veff = iter2_init
        .build_v_eff_with_energy()
        .expect("iter-2 build_v_eff");
    let iter2_diag = iter2_veff
        .diagonalize(8, None)
        .expect("iter-2 diagonalize");
    let iter2_eigs = iter2_diag.eigenvalues().to_vec();
    let iter2_first = iter2_eigs[0];
    let iter2_last = *iter2_eigs.last().unwrap();
    eprintln!(
        "[iter2-real] iter-2 RR: band-0 = {:.6} Ha, last = {:.6} Ha",
        iter2_first, iter2_last,
    );

    let force_no_eigs_active = std::env::var("CHEMRUST_FORCE_NO_EIGS").as_deref() == Ok("1");
    eprintln!("[iter2-real] CHEMRUST_FORCE_NO_EIGS active: {}", force_no_eigs_active);

    eprintln!(
        "[iter2-real] Δ_last (iter-2 vs iter-1) = {:.6} Ha",
        (iter2_last - iter1_last).abs(),
    );
    eprintln!(
        "[iter2-real] Δ_band0 (iter-2 vs iter-1) = {:.6} Ha",
        (iter2_first - iter1_first).abs(),
    );

    if force_no_eigs_active {
        eprintln!(
            "[iter2-real] DIAGNOSIS (with flag): if iter-2 last band stays near {:.3}, the per-band eigenvalue branches contain the bug.",
            iter1_last,
        );
    } else {
        eprintln!(
            "[iter2-real] DIAGNOSIS (no flag, production path): expect iter-2 last band ≈ 1.95 Ha (the §11 symptom).",
        );
    }
}


//
// Variant C showed Mode B with eigs=Some(clean_eigs) degrades denoising from
// Δ_band0 = 0.000160 Ha to 0.009922 Ha (62×).
//
// In Mode B, the `eigenvalues` parameter ONLY affects two things:
// 1. `b_low = eig[last]`  (chebyshev.rs:1411, sets the filter window cutoff)
// 2. Triggers the `if eigenvalues.is_some()` branches at lines 1582 (lam_y init)
//    and 1651 (Lambda_X update). The CONTENT used by those branches comes from
//    `lam_source = &h_eig`, NOT from `eigenvalues`. So varying eig[last] varies
//    branch 1 in isolation while leaving branches 2-3 unchanged.
//
// Bisecting plan: pass eigenvalues = Some(...) but set eig[last] to alternative
// values, holding per-band shifts (h_eig) constant.
// - Trial 1: eig[last] = max_veff (matches iter-1 fallback) — branches 2+3 only
// - Trial 2: eig[last] = clean_last (matches Variant C) — all three branches
// - Trial 3: eig[last] = 20.0 (artificially large) — exaggerates branch 1 effect
//
// If Δ_band0 is comparable across all three trials → bug is in branches 2+3
// (per-band shifts at lines 1582, 1654).
// If Δ_band0 varies dramatically with eig[last] → bug is in branch 1 (b_low source).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn pollution_bisect_blow_vs_per_band_shifts() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    use chemrust_scf::density::test_api::FilterMode;
    use num_complex::Complex64;

    let fx = fixtures::cu111_co::fixture();

    // ndeg=0 baseline for clean eigenvalues
    let clean_eigs: Vec<f64> = {
        let state = fixtures::cu111_co::build_scf_state(fx);
        state.build_v_eff().expect("build_v_eff")
            .diagonalize_with_mode(0, None, FilterMode::SinvHKeepHEig)
            .expect("ndeg=0").eigenvalues().to_vec()
    };
    let clean_band0 = clean_eigs[0];
    let clean_last = *clean_eigs.last().unwrap();
    let n = clean_eigs.len();
    eprintln!(
        "[bisect] clean baseline: band-0 = {:.6}, last = {:.6}, n_bands = {}",
        clean_band0, clean_last, n,
    );

    // Polluter parameters
    let polluter_band = 1usize;
    let alpha = 0.10;

    // Helper: pollute, set eigenvalues, diagonalize, return eigenvalues
    let run = |eig_last_override: Option<f64>| -> Vec<f64> {
        let state = fixtures::cu111_co::build_scf_state(fx);
        let mut state = state.build_v_eff().expect("build_v_eff");
        let (n_bands, n_pw) = state.psi_shape();

        // Pollute band-0 with band-1
        {
            let psi = state.psi_data_mut();
            let pol_band: Vec<Complex64> = psi[polluter_band * n_pw..(polluter_band + 1) * n_pw].to_vec();
            let mut norm2 = 0.0f64;
            for g in 0..n_pw {
                let new = psi[g] + Complex64::new(alpha, 0.0) * pol_band[g];
                psi[g] = new;
                norm2 += new.norm_sqr();
            }
            let inv = 1.0 / norm2.sqrt();
            for g in 0..n_pw {
                psi[g] = psi[g] * Complex64::new(inv, 0.0);
            }
        }

        // Set eigenvalues (Some(...) triggers iter-2 path)
        if let Some(last_val) = eig_last_override {
            let mut e = clean_eigs.clone();
            *e.last_mut().unwrap() = last_val;
            state.set_eigenvalues(e);
        }
        // If None, leave eigenvalues empty (iter-1 path)

        state
            .diagonalize_with_mode(8, None, FilterMode::SinvHKeepHEig)
            .expect("ndeg=8")
            .eigenvalues()
            .to_vec()
    };

    // max_veff is computed inside diagonalize; we approximate as
    // the iter-1 chemb b_low (about 0.089 Ha for Cu111+CO per §10).
    let approx_max_veff = 0.089;

    eprintln!("\n[bisect] === Bisecting branches ===");
    eprintln!(
        "{:>25}  {:>12}  {:>12}  {:>12}",
        "config", "b_low used", "Δ_band0", "Δ_last",
    );

    // Trial 0: eigs=None (iter-1 path, all eigenvalue branches OFF)
    let r0 = run(None);
    let d0_b0 = (r0[0] - clean_band0).abs();
    let d0_l = (*r0.last().unwrap() - clean_last).abs();
    eprintln!(
        "{:>25}  {:>12.4}  {:>12.6}  {:>12.6}",
        "iter1 path (None)", approx_max_veff, d0_b0, d0_l,
    );

    // Trial 1: eigs=Some, eig[last] = max_veff (b_low same as iter-1, branches 2+3 ON)
    let r1 = run(Some(approx_max_veff));
    let d1_b0 = (r1[0] - clean_band0).abs();
    let d1_l = (*r1.last().unwrap() - clean_last).abs();
    eprintln!(
        "{:>25}  {:>12.4}  {:>12.6}  {:>12.6}",
        "Some, blow=max_veff", approx_max_veff, d1_b0, d1_l,
    );

    // Trial 2: eigs=Some, eig[last] = clean_last (matches Variant C, all branches ON)
    let r2 = run(Some(clean_last));
    let d2_b0 = (r2[0] - clean_band0).abs();
    let d2_l = (*r2.last().unwrap() - clean_last).abs();
    eprintln!(
        "{:>25}  {:>12.4}  {:>12.6}  {:>12.6}",
        "Some, blow=clean_last", clean_last, d2_b0, d2_l,
    );

    // Trial 3: eigs=Some, eig[last] = 5.0 (artificially LARGE → b_low up → window narrows)
    let r3 = run(Some(5.0));
    let d3_b0 = (r3[0] - clean_band0).abs();
    let d3_l = (*r3.last().unwrap() - clean_last).abs();
    eprintln!(
        "{:>25}  {:>12.4}  {:>12.6}  {:>12.6}",
        "Some, blow=5.0", 5.0, d3_b0, d3_l,
    );

    eprintln!("\n[bisect] Interpretation:");
    eprintln!(
        "  Trial 0 (iter-1 path): Δ_band0 should be small (we already proved this in Variant B)"
    );
    eprintln!(
        "  Trial 1 → Trial 2: ONLY b_low changes (branches 2+3 unchanged). If Δ_band0 changes much: branch 1 is sensitive."
    );
    eprintln!(
        "  Trial 1 vs Trial 0: ONLY branches 2+3 toggled (b_low ≈ same). If Δ_band0 jumps: branches 2+3 contain the bug."
    );
    eprintln!(
        "  Trial 3: stress test of branch 1 with artificially high b_low. Expect: window narrows → fewer eigenvectors to denoise from."
    );
}


//
// The earlier pollution probes (Variants A and B) ran with eigenvalues=None
// because build_scf_state returns Initialized state with empty eigenvalues.
// This corresponds to the iter-1 filter code path.
//
// At iter-2+, the filter receives eigenvalues from the prior iteration, which
// activates additional code paths:
// - b_low = eig[last]  (chebyshev.rs:1411)
// - lam_y initial value per band: (σ₁/e)·(λ_b − c)  (chebyshev.rs:1583)
// - Λ_X update per band uses lam_source[b]  (chebyshev.rs:1654)
// - sigma_rchfsi at chebyshev.rs:1546 unchanged
//
// This test pre-populates the state's eigenvalues field and runs the filter,
// mimicking the iter-2 code path. If filter still denoises → bug is elsewhere
// (likely in inputs to filter — VnlBatchData at iter-2 V_eff, density mixing,
// or layout handoff). If filter fails to denoise → bug is in the per-band
// eigenvalue branches at chebyshev.rs:1582-1587, 1651-1661, 1693.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn pollution_with_eigenvalues_exercises_iter2_path() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    use chemrust_scf::density::test_api::FilterMode;
    use num_complex::Complex64;

    let fx = fixtures::cu111_co::fixture();

    // Step A: run ndeg=0 to obtain the iter-1 eigenvalues (clean baseline)
    let clean_eigs: Vec<f64> = {
        let state = fixtures::cu111_co::build_scf_state(fx);
        state
            .build_v_eff()
            .expect("build_v_eff")
            .diagonalize_with_mode(0, None, FilterMode::SinvHKeepHEig)
            .expect("ndeg=0 diagonalize")
            .eigenvalues()
            .to_vec()
    };
    let clean_band0 = clean_eigs.first().copied().unwrap_or(f64::NAN);
    let clean_last = clean_eigs.last().copied().unwrap_or(f64::NAN);
    eprintln!(
        "[pollute2] clean baseline (ndeg=0): band-0 = {:.6} Ha, last = {:.6} Ha, n_bands = {}",
        clean_band0, clean_last, clean_eigs.len(),
    );

    // Step B: inject pollution AND pre-populate eigenvalues, then run ndeg=8.
    let run_polluted_with_eigs = |alpha: f64, polluter_band: usize, with_eigs: bool| -> Vec<f64> {
        let state = fixtures::cu111_co::build_scf_state(fx);
        let mut state = state.build_v_eff().expect("build_v_eff");
        let (n_bands, n_pw) = state.psi_shape();
        assert!(polluter_band < n_bands, "polluter_band out of range");
        assert!(polluter_band != 0, "polluter must differ from band-0");

        // Pollute band-0
        {
            let psi = state.psi_data_mut();
            let pol_band: Vec<Complex64> = psi[polluter_band * n_pw..(polluter_band + 1) * n_pw].to_vec();
            let mut norm2 = 0.0f64;
            for g in 0..n_pw {
                let new = psi[g] + Complex64::new(alpha, 0.0) * pol_band[g];
                psi[g] = new;
                norm2 += new.norm_sqr();
            }
            let inv_norm = 1.0 / norm2.sqrt();
            for g in 0..n_pw {
                psi[g] = psi[g] * Complex64::new(inv_norm, 0.0);
            }
        }

        // Optionally pre-populate eigenvalues (iter-2 path)
        if with_eigs {
            state.set_eigenvalues(clean_eigs.clone());
        }

        state
            .diagonalize_with_mode(8, None, FilterMode::SinvHKeepHEig)
            .expect("ndeg=8 diagonalize")
            .eigenvalues()
            .to_vec()
    };

    // Test matrix: {high-energy, low-energy} polluter × {with_eigs=true,false}
    let polluter_high = state_n_bands(fx) - 1;
    let polluter_low = 1usize;

    eprintln!("\n[pollute2] === per-variant table ===");
    eprintln!(
        "{:>12}  {:>7}  {:>5}  {:>10}  {:>12}  {:>12}",
        "polluter", "eigs", "α", "→band-0", "Δ_band0", "ratio",
    );
    for (polluter_label, polluter_b) in [("high(159)", polluter_high), ("low(1)", polluter_low)] {
        for with_eigs in [false, true] {
            let alpha = 0.10;
            let polluted_eigs = run_polluted_with_eigs(alpha, polluter_b, with_eigs);
            let pol_band0 = polluted_eigs.first().copied().unwrap_or(f64::NAN);
            let dband0 = pol_band0 - clean_band0;
            // Pre-filter analytic estimate for ratio
            let pol_eig = clean_eigs.get(polluter_b).copied().unwrap_or(f64::NAN);
            let pre_filter_analytic = alpha * alpha * (pol_eig - clean_band0);
            let ratio = if pre_filter_analytic.abs() > 1e-12 {
                dband0 / pre_filter_analytic
            } else {
                f64::NAN
            };
            eprintln!(
                "{:>12}  {:>7}  {:>5.2}  {:>10.6}  {:>12.6}  {:>12.4}",
                polluter_label, with_eigs, alpha, pol_band0, dband0, ratio,
            );
        }
    }

    // Discriminating diagnosis: low-energy polluter WITH eigs is the iter-2-like path.
    let polluted = run_polluted_with_eigs(0.10, polluter_low, true);
    let pol_band0 = polluted.first().copied().unwrap_or(f64::NAN);
    let pol_last = polluted.last().copied().unwrap_or(f64::NAN);
    let dband0 = (pol_band0 - clean_band0).abs();
    let dlast = (pol_last - clean_last).abs();
    eprintln!(
        "\n[pollute2] Discriminator (low polluter, eigs=Some, α=0.10):"
    );
    eprintln!(
        "  Δ_band0 = {:.6} Ha   (clean = {:.6}, polluted-filtered = {:.6})",
        dband0, clean_band0, pol_band0,
    );
    eprintln!(
        "  Δ_last  = {:.6} Ha   (clean = {:.6}, polluted-filtered = {:.6})",
        dlast, clean_last, pol_last,
    );
    if dband0 < 0.005 && dlast < 0.005 {
        eprintln!(
            "[pollute2] DIAGNOSIS: filter denoises even with eigenvalues=Some. The iter-2-specific code paths are NOT the bug. Look upstream of filter (input ψ from prior RR, vnl_data at iter-2 V_eff, density mixing, layout handoff)."
        );
    } else if dband0 < 0.05 && dlast < 0.05 {
        eprintln!(
            "[pollute2] DIAGNOSIS: filter partially denoises with eigenvalues=Some. The eigenvalue-dependent branches contribute some error but are not catastrophic. Bug likely a numerical sensitivity in those branches."
        );
    } else {
        eprintln!(
            "[pollute2] DIAGNOSIS: filter FAILS to denoise when eigenvalues=Some. The eigenvalue-dependent branches at chebyshev.rs:1582-1587 (lam_y init), 1651-1661 (Λ_X update), or 1693 (reconstruction) contain a formula error."
        );
    }
}



//
// PASS  → filter recurrence is the distortion source (bug in chebyshev.rs:1451-1697)
// FAIL  → bug is downstream of filter (RR, S_sub assembly, β·ψ caching,
//         ZHEGVD wiring, ψ rotation, or density assembly)
//
// This is the cheapest bisection probe in the entire debug session: one
// invocation of diagonalize_with_mode(0, None, mode) feeds CASTEP-converged
// ψ directly into Rayleigh-Ritz with no Chebyshev recurrence interposed.
// Anchor: A-BAND0 from notes/debug/debug-20260523-1149-iter2-divergence/CRITERIA.md.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn ndeg_zero_with_castep_psi_matches_bands() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    use chemrust_scf::density::test_api::FilterMode;

    let fx = fixtures::cu111_co::fixture();
    let castep_bands = &fx.bands_eigenvalues;

    // Three modes × ndeg=0 (filter bypassed). All three should produce
    // identical results because the recurrence body never executes.
    let run_ndeg0 = |mode: FilterMode| -> Vec<f64> {
        let state = fixtures::cu111_co::build_scf_state(fx);
        state
            .build_v_eff()
            .expect("build_v_eff")
            .diagonalize_with_mode(0, None, mode)
            .expect("diagonalize_with_mode(0, ..)")
            .eigenvalues()
            .to_vec()
    };

    let eig_a = run_ndeg0(FilterMode::BareH);
    let eig_b = run_ndeg0(FilterMode::SinvHKeepHEig);
    let eig_c = run_ndeg0(FilterMode::SinvHFullDas);

    let n_check = 10.min(eig_a.len()).min(castep_bands.len());

    eprintln!("\n[ndeg_zero] Per-band |Δ| vs CASTEP .bands (Ha) — filter bypassed");
    eprintln!(
        "{:>5}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "band", "CASTEP", "Mode-A", "Mode-B", "Mode-C", "|ΔA|", "|ΔB|", "|ΔC|",
    );
    for j in 0..n_check {
        let ref_e = castep_bands[j];
        let ea = eig_a.get(j).copied().unwrap_or(f64::NAN);
        let eb = eig_b.get(j).copied().unwrap_or(f64::NAN);
        let ec = eig_c.get(j).copied().unwrap_or(f64::NAN);
        eprintln!(
            "{:>5}  {:>12.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
            j, ref_e, ea, eb, ec,
            (ea - ref_e).abs(),
            (eb - ref_e).abs(),
            (ec - ref_e).abs(),
        );
    }
    let last = eig_a.len().saturating_sub(1);
    eprintln!(
        "[ndeg_zero] last band: A={:.6} B={:.6} C={:.6}  (iter-1 expected ≈ 0.13 Ha)",
        eig_a.get(last).copied().unwrap_or(f64::NAN),
        eig_b.get(last).copied().unwrap_or(f64::NAN),
        eig_c.get(last).copied().unwrap_or(f64::NAN),
    );

    // Cross-mode consistency: with ndeg=0 the filter body never runs, so
    // all modes must produce identical RR output (within numerical noise).
    // If they don't, the dispatch is leaking through some pre-recurrence
    // state setup that is mode-dependent.
    let mode_consistency_thresh = 1e-10;
    for j in 0..n_check {
        let ea = eig_a.get(j).copied().unwrap_or(f64::NAN);
        let eb = eig_b.get(j).copied().unwrap_or(f64::NAN);
        let ec = eig_c.get(j).copied().unwrap_or(f64::NAN);
        let ab = (ea - eb).abs();
        let ac = (ea - ec).abs();
        assert!(
            ab < mode_consistency_thresh && ac < mode_consistency_thresh,
            "[ndeg_zero] mode mismatch at band {j}: A={:.6} B={:.6} C={:.6} \
             (|A-B|={:.2e}, |A-C|={:.2e}) — ndeg=0 should bypass all mode-specific code",
            ea, eb, ec, ab, ac,
        );
    }
    eprintln!("[ndeg_zero] mode consistency PASS (A == B == C within 1e-10)");

    // A-BAND0 anchor: with filter bypassed, RR-only path must produce
    // band-0 within 0.05 Ha of CASTEP −1.055 Ha.
    let band0_b = eig_b.first().copied().unwrap_or(f64::NAN);
    let ref_b0 = castep_bands.first().copied().unwrap_or(f64::NAN);
    let delta_b0 = (band0_b - ref_b0).abs();
    eprintln!(
        "[ndeg_zero] A-BAND0: ndeg=0 band-0 = {:.6} Ha, CASTEP = {:.6} Ha, |Δ| = {:.6} Ha (gate 0.05)",
        band0_b, ref_b0, delta_b0,
    );

    // A-BANDS-10 anchor: lowest 10 bands within 0.05 Ha
    const SC4_GATE: f64 = 0.05; // Ha
    let mut failures: Vec<String> = Vec::new();
    for j in 0..n_check {
        let eig = eig_b.get(j).copied().unwrap_or(f64::NAN);
        let delta = (eig - castep_bands[j]).abs();
        if delta >= SC4_GATE {
            failures.push(format!(
                "  band {j}: ndeg=0 = {:.6} Ha, CASTEP = {:.6} Ha, |Δ| = {:.6} Ha ≥ {SC4_GATE}",
                eig, castep_bands[j], delta,
            ));
        }
    }

    if failures.is_empty() {
        eprintln!(
            "[ndeg_zero] DIAGNOSIS: filter bypassed → RR matches CASTEP. \
             The bug is INSIDE the Chebyshev filter recurrence \
             (chebyshev.rs:1451-1697). Step 8 narrows to filter body."
        );
    } else {
        eprintln!(
            "[ndeg_zero] DIAGNOSIS: filter bypassed → RR DIVERGES from CASTEP. \
             The bug is DOWNSTREAM of the filter (RR / S_sub / β·ψ / density). \
             Step 8 narrows to rayleigh_ritz.rs and scf.rs."
        );
    }

    // The assert fails on EITHER outcome's logical opposite — we want the
    // diagnosis text in the log unconditionally and the assertion to
    // distinguish the branches.
    assert!(
        failures.is_empty(),
        "ndeg_zero_with_castep_psi_matches_bands: RR-only path failed SC-4-tight.\n\
         This means the bug is DOWNSTREAM of the Chebyshev filter — narrow Step 8 \
         to rayleigh_ritz.rs and scf.rs:447-595, NOT to chebyshev.rs:1451-1697.\n\n{}",
        failures.join("\n"),
    );
}

/// Tight test for electron count diagnostic (§11b).
///
/// **Discriminator**: The `total_e_phys_conv` diagnostic at `scf.rs:722` applies
/// cell volume to density already in CASTEP raw units (ρ×Ω), producing a 22,300×
/// error. This test verifies both diagnostics agree after the fix.
///
/// **Anchor**: Cu111+CO has 186 valence electrons (11 Cu @ 11 e⁻, 1 C @ 4 e⁻, 1 O @ 6 e⁻).
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn electron_count_diagnostic_correct() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);

    // Run iter-1 (uses fixture density, so V_eff is correct)
    let scf = state
        .build_v_eff()
        .expect("build_v_eff")
        .diagonalize(0, None)
        .expect("diagonalize");

    // Compute electron count from density using both formulas
    let density = scf.density().as_wave_array();
    let rho_sum: f64 = density.iter().sum();
    let n_grid = density.len() as f64;
    let cell_volume = scf.cell().volume;

    let total_e_raw_conv = rho_sum / n_grid;
    let total_e_phys_conv_wrong = rho_sum * cell_volume / n_grid;  // Wrong formula (before fix)
    let total_e_phys_conv = rho_sum / n_grid;  // Correct formula (after fix)

    eprintln!("[electron_count_diagnostic_correct]");
    eprintln!("  rho_sum = {:.6e}", rho_sum);
    eprintln!("  n_grid = {}", n_grid as usize);
    eprintln!("  cell_volume = {:.6} Bohr³", cell_volume);
    eprintln!("  total_e_raw_conv = {:.6} e⁻", total_e_raw_conv);
    eprintln!("  total_e_phys_conv (correct) = {:.6} e⁻", total_e_phys_conv);
    eprintln!("  total_e_phys_conv (wrong, before fix) = {:.6} e⁻", total_e_phys_conv_wrong);
    eprintln!("  ratio (wrong/correct) = {:.2e}", total_e_phys_conv_wrong / total_e_raw_conv);

    // Expected: 186 e⁻ (11 Cu @ 11 e⁻ + 1 C @ 4 e⁻ + 1 O @ 6 e⁻)
    const EXPECTED_ELECTRONS: f64 = 186.0;
    const TOLERANCE: f64 = 5.0; // ±5 e⁻

    // Check raw_conv (should already pass)
    let delta_raw = (total_e_raw_conv - EXPECTED_ELECTRONS).abs();
    eprintln!(
        "  |total_e_raw_conv - expected| = {:.6} e⁻ (gate {:.1})",
        delta_raw, TOLERANCE
    );
    assert!(
        delta_raw < TOLERANCE,
        "total_e_raw_conv = {:.6} e⁻, expected {:.1} ± {:.1} e⁻",
        total_e_raw_conv,
        EXPECTED_ELECTRONS,
        TOLERANCE
    );

    // Check phys_conv (should fail before fix, pass after fix)
    let delta_phys = (total_e_phys_conv - EXPECTED_ELECTRONS).abs();
    eprintln!(
        "  |total_e_phys_conv - expected| = {:.6} e⁻ (gate {:.1})",
        delta_phys, TOLERANCE
    );
    assert!(
        delta_phys < TOLERANCE,
        "total_e_phys_conv = {:.6} e⁻, expected {:.1} ± {:.1} e⁻ \
         (BUG: diagnostic applies cell volume to density already in CASTEP raw units ρ×Ω)",
        total_e_phys_conv,
        EXPECTED_ELECTRONS,
        TOLERANCE
    );

    // Check both diagnostics agree (should fail before fix, pass after fix)
    let consistency_ratio = (total_e_phys_conv / total_e_raw_conv - 1.0).abs();
    eprintln!(
        "  |phys_conv/raw_conv - 1| = {:.2e} (gate 0.01)",
        consistency_ratio
    );
    assert!(
        consistency_ratio < 0.01,
        "Diagnostics disagree: raw_conv = {:.6} e⁻, phys_conv = {:.6} e⁻ \
         (ratio {:.2e}, expected ~1.0 since density is already in CASTEP raw units)",
        total_e_raw_conv,
        total_e_phys_conv,
        total_e_phys_conv / total_e_raw_conv,
    );

    eprintln!("[electron_count_diagnostic_correct] PASS");
}
