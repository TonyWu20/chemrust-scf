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
    let result = chemrust_scf::run_scf_with_energy(state, 8, 1e-8).expect("SCF converged");

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

    let result = chemrust_scf::run_scf_with_energy(state, 8, 1e-8)
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
        &kernels,
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
        &kernels,
        &solver,
    )
    .expect("VnlBatchData::precompute");

    // Take band 0 (the converged lowest eigenstate)
    let band0: Vec<Complex64> = psi_data.iter().take(n_pw).copied().collect();

    let max_residual = check_s_inv_s_identity(&band0, n_pw, &vnl_data, &blas, &stream, &solver)
        .expect("check_s_inv_s_identity");

    eprintln!("[S⁻¹·S identity] ‖S⁻¹·S·ψ₀ − ψ₀‖_∞ = {:.6e}", max_residual,);

    // If > 1e-8: Q convention in s_inv_mat (Woodbury) is inconsistent
    // with q_matrix (S operator). The S⁻¹·H filter will over-subtract.
    assert!(
        max_residual < 1e-6,
        "S⁻¹·S·ψ ≠ ψ: max residual = {:.6e} > 1e-6. \
         Q convention mismatch between s_inv_mat and q_matrix.",
        max_residual,
    );
}
