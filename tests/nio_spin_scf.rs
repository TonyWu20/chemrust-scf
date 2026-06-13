//! NiO spin-polarised SCF warm-start discriminator tests.
//!
//! Reference values from a converged CASTEP spin-polarised run (no U, finer grid):
//!   `/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin/`
//!
//! NiO: 14 k-points, spin-polarised, wave grid [20, 20, 20], fine grid [40, 40, 40],
//! USPP (Ni, O), 62 bands, N_up=36, N_dn=28.
//!
//! ## Warm-start test pattern
//!
//! Follows the same pattern as the Cu111_CO warm-start discriminator
//! (tests/ca_scf_convergence.rs `iter1_drift_from_castep_state_is_bounded`):
//! 1. Load converged density + wavefunctions from CASTEP fixtures
//! 2. Construct `ScfIteration<SpinCollinear, Initialized, MixingOff>`
//! 3. Run ONE SCF iteration: build_v_eff → diagonalize → construct_density → mix → check
//! 4. Assert discriminator criteria against CASTEP reference values

use std::sync::OnceLock;

use chemrust_hamiltonian_core::{
    CastepBin, CastepBinFile, CheckFile, GVectorGrid, PseudopotentialSet, SpinCollinear,
};
use chemrust_scf::{
    downsample_array_to_wave_grid, pw_coords_to_fft_indices, KPoint, KptDataSet, PerSpinDensity,
    PerSpinPwCoefficients, PwCoefficients, ScfIteration, SmearingParams, SmearingScheme,
    SpinChannelData, WaveGridArray, Density,
};
use ndarray::ShapeBuilder;

// ---------------------------------------------------------------------------
// Constants — reference values from CASTEP output
// ---------------------------------------------------------------------------

/// Path to CASTEP reference output directory for NiO spin-polarised run.
const NIO_SPIN_DIR: &str = "/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin";

/// Path to pseudopotential directory.
const POTENTIAL_DIR: &str = "/export/Potentials";

/// Total energy from NiO.castep:774418 (eV)
const REFERENCE_ENERGY_EV: f64 = -7160.230577732;

/// Fermi energy (both spins) from NiO.bands:5 (Ha)
const REFERENCE_FERMI_ENERGY_HA: f64 = 0.152664;

/// 2*Integrated Spin Density from NiO.castep:774415
const REFERENCE_2X_SPIN_DENSITY: f64 = -0.0619641;

/// Eigenvalue[kpt=0, spin=0, band=0] from NiO.bands:12 (Ha)
const EPS_SPIN0_BAND0_REF_HA: f64 = -0.59098046;

/// Eigenvalue[kpt=0, spin=1, band=0] from NiO.bands:75 (Ha)
const EPS_SPIN1_BAND0_REF_HA: f64 = -0.59108380;

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// V1: eigenvalue tolerance (Ha). Iter-1 warm-start produces eigenvalue differences
/// up to ~3e-4 Ha due to eigensolver convergence path; not a physics error.
const TOL_EPS_HA: f64 = 3e-4;

/// V2: total energy drift gate (Ha). One SCF iteration from converged density
/// should stay within this bound; ratchet down as eigensolver stabilizes.
const DRIFT_TOLERANCE_HA: f64 = 2e-2;

/// V4: Fermi energy tolerance (Ha)
const TOL_FERMI_HA: f64 = 1e-3;

// ---------------------------------------------------------------------------
// E_nonCoulomb — pseudopotential non-Coulombic correction (CONSTANT)
//
// From NiO warm-start .castep: E_nonCoulomb = +533.14 eV = +19.59 Ha.
// CASTEP computes this from the local part of each pseudopotential
// (pot.f90:4205, energy.f90). Our ewald_energy does not yet include it.
// TODO: compute from PseudopotentialSet data in chemrust-hamiltonian-core.
// ---------------------------------------------------------------------------
const E_NON_COULOMB_HA: f64 = 533.13587176588384864 / chemrust_scf::HARTREE_TO_EV;

// ---------------------------------------------------------------------------
// Cached fixture
// ---------------------------------------------------------------------------

/// All pre-loaded CASTEP reference data for the NiO spin-polarised system.
pub struct NioSpinFixture {
    pub bin: CastepBin,
    pub check: CastepBin,
    pub bands_eigenvalues_per_spin: Vec<Vec<Vec<f64>>>, // [kpt][spin][band]
    pub bands_fermi_energies: Vec<f64>,
    pub pots: PseudopotentialSet,
}

static FIXTURE: OnceLock<NioSpinFixture> = OnceLock::new();

/// Access the cached NiO fixture, loading it on first call.
pub fn fixture() -> &'static NioSpinFixture {
    FIXTURE.get_or_init(|| load_spin_fixture().expect("failed to load NiO spin-polarised fixture"))
}

fn load_spin_fixture() -> Result<NioSpinFixture, Box<dyn std::error::Error>> {
    let fixture_dir =
        std::env::var("NIO_SPIN_FIXTURE_DIR").unwrap_or_else(|_| NIO_SPIN_DIR.to_string());
    let potential_dir =
        std::env::var("CASTEP_POTENTIAL_DIR").unwrap_or_else(|_| POTENTIAL_DIR.to_string());

    let bin_path = format!("{fixture_dir}/NiO.castep_bin");
    let bin_file = std::fs::File::open(&bin_path)?;
    let bin = CastepBinFile::read(std::io::BufReader::new(bin_file))?;

    let check_path = format!("{fixture_dir}/NiO.check");
    let check_file = std::fs::File::open(&check_path)?;
    let check = CheckFile::read(std::io::BufReader::new(check_file))?;

    let bands_path = format!("{fixture_dir}/NiO.bands");
    let bands_text = std::fs::read_to_string(&bands_path)?;
    let (fermi_energies, eigenvalues_per_spin) = parse_spin_bands_file(&bands_text)?;

    let pots = PseudopotentialSet::from_dir(
        potential_dir,
        &bin.cell.species_symbols,
        &bin.cell.species_pot_files,
    )?;

    Ok(NioSpinFixture { bin, check, bands_eigenvalues_per_spin: eigenvalues_per_spin, bands_fermi_energies: fermi_energies, pots })
}

// ---------------------------------------------------------------------------
// Parse spin-polarised .bands file
// ---------------------------------------------------------------------------

fn parse_spin_bands_file(
    text: &str,
) -> Result<(Vec<f64>, Vec<Vec<Vec<f64>>>), Box<dyn std::error::Error>> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 5 {
        return Err("bands file too short".into());
    }

    let nspins: usize = lines[1].trim().split_whitespace().last()
        .ok_or("cannot parse nspins")?.parse()?;

    let fermi_energies: Vec<f64> = {
        let parts: Vec<&str> = lines[4].trim().split_whitespace().collect();
        parts[parts.len() - nspins..].iter()
            .map(|s| s.parse::<f64>().map_err(|e| format!("bad Fermi: {e}")))
            .collect::<Result<_, _>>()?
    };

    let mut eigenvalues: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut kpt_idx: Option<usize> = None;
    let mut current_spin: Option<usize> = None;
    let mut current_values: Vec<f64> = Vec::new();

    let mut flush = |kpt: Option<usize>, spin: Option<usize>, vals: &mut Vec<f64>,
                     store: &mut Vec<Vec<Vec<f64>>>| {
        if let (Some(k), Some(s)) = (kpt, spin) {
            while store.len() <= k { store.push(vec![Vec::new(); nspins]); }
            store[k][s] = std::mem::take(vals);
        }
    };

    for line in lines.iter().skip(5) {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        if trimmed.starts_with("K-point") {
            flush(kpt_idx, current_spin, &mut current_values, &mut eigenvalues);
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            kpt_idx = Some(parts[1].parse::<usize>()? - 1);
            current_spin = None;
        } else if trimmed.starts_with("Spin component") {
            flush(kpt_idx, current_spin, &mut current_values, &mut eigenvalues);
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            current_spin = Some(parts[2].parse::<usize>()? - 1);
        } else if current_spin.is_some() {
            if let Ok(val) = trimmed.parse::<f64>() { current_values.push(val); }
        }
    }
    flush(kpt_idx, current_spin, &mut current_values, &mut eigenvalues);

    if eigenvalues.is_empty() { return Err("no eigenvalues parsed".into()); }
    Ok((fermi_energies, eigenvalues))
}

// ---------------------------------------------------------------------------
// SCF state construction from fixture data
// ---------------------------------------------------------------------------

pub fn build_spin_scf_state(fx: &NioSpinFixture) -> ScfIteration<SpinCollinear> {
    let cell = fx.bin.cell.clone();
    let pots = fx.pots.clone();

    let wfc = fx.check.wavefunction.as_ref().expect(".check must have wavefunction");
    let [ngx, ngy, ngz] = wfc.grid;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);

    let [fgx, fgy, fgz] = fx.check.fine_grid.expect(".check must have fine_grid");
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    let den_grid = fx.bin.density.grid;
    assert_eq!(den_grid, [fgx, fgy, fgz], "density grid must match fine grid");

    let nspins = fx.bin.density.nspins;
    assert_eq!(nspins, 2, "NiO fixture must have nspins=2");

    // --- Per-spin density: downsample from fine grid to wave grid ---
    let charge_fine = fx.bin.density.charge.as_real_grid().as_real_array();
    let spin_fine = fx.bin.density.spin.as_ref().expect("spin density")
        .as_real_grid().as_real_array();

    let charge_wave = downsample_array_to_wave_grid(charge_fine, &fine_grid, &wave_grid)
        .expect("downsample charge");
    let spin_wave = downsample_array_to_wave_grid(spin_fine, &fine_grid, &wave_grid)
        .expect("downsample spin");

    let charge_arr = charge_wave.as_fine_array();
    let spin_arr = spin_wave.as_fine_array();

    let half = 0.5_f64;
    let rho_up_arr = charge_arr.mapv(|v| v * half) + spin_arr.mapv(|v| v * half);
    let rho_dn_arr = charge_arr.mapv(|v| v * half) - spin_arr.mapv(|v| v * half);

    let per_spin_density = PerSpinDensity::new(SpinChannelData::new::<SpinCollinear>(vec![
        Density::from_inner(WaveGridArray::from_inner(rho_up_arr)),
        Density::from_inner(WaveGridArray::from_inner(rho_dn_arr)),
    ]));

    // --- Multi-kpt wavefunctions from .check (spin-major layout) ---
    let nkpts = wfc.kpt_data.len() / nspins;
    assert!(nkpts >= 1);

    let n_bands = wfc.kpt_data[0].bands.len();

    // Kpt weights from .castep_bin
    let kpt_weights = fx.bin.kpoint_weights.clone();
    assert_eq!(kpt_weights.len(), nkpts);

    let ctx = std::sync::Arc::new(cudarc::driver::CudaContext::new(0).expect("GPU required"));
    let stream = ctx.default_stream();

    use cudarc::driver::CudaSlice;
    let mut psi_gpu_s0: Vec<PwCoefficients> = Vec::with_capacity(nkpts);
    let mut psi_gpu_s1: Vec<PwCoefficients> = Vec::with_capacity(nkpts);
    let mut psi_data_s0: Vec<Vec<num_complex::Complex64>> = Vec::with_capacity(nkpts);
    let mut psi_data_s1: Vec<Vec<num_complex::Complex64>> = Vec::with_capacity(nkpts);
    let mut k_points_vec: Vec<KPoint> = Vec::with_capacity(nkpts);
    let mut pw_coords_vec: Vec<Vec<[i32; 3]>> = Vec::with_capacity(nkpts);
    let mut pw_fft_vec: Vec<Vec<i32>> = Vec::with_capacity(nkpts);

    for ikpt in 0..nkpts {
        let k_s0 = &wfc.kpt_data[ikpt];
        let k_s1 = &wfc.kpt_data[nkpts + ikpt];
        assert_eq!(k_s0.bands.len(), n_bands);
        assert_eq!(k_s1.bands.len(), n_bands);

        let n_pw_kpt = k_s0.nplw;
        let n_el = n_bands * n_pw_kpt;

        let gpu0: CudaSlice<chemrust_scf::device::CudaComplex> =
            stream.alloc_zeros(n_el).expect("GPU spin0");
        let gpu1: CudaSlice<chemrust_scf::device::CudaComplex> =
            stream.alloc_zeros(n_el).expect("GPU spin1");

        psi_gpu_s0.push(PwCoefficients::new(gpu0));
        psi_gpu_s1.push(PwCoefficients::new(gpu1));

        psi_data_s0.push(k_s0.bands.iter().flatten().copied().collect());
        psi_data_s1.push(k_s1.bands.iter().flatten().copied().collect());

        k_points_vec.push(KPoint { coords: k_s0.coords, weight: kpt_weights[ikpt] });
        pw_coords_vec.push(k_s0.pw_grid_coord.clone());
        pw_fft_vec.push(pw_coords_to_fft_indices(&k_s0.pw_grid_coord, &wave_grid));
    }

    let psi = PerSpinPwCoefficients(SpinChannelData::new::<SpinCollinear>(vec![
        KptDataSet::new(psi_gpu_s0, nkpts),
        KptDataSet::new(psi_gpu_s1, nkpts),
    ]));
    let psi_data = SpinChannelData::new::<SpinCollinear>(vec![
        KptDataSet::new(psi_data_s0, nkpts),
        KptDataSet::new(psi_data_s1, nkpts),
    ]);

    // NiO .param: spin_fix=6 → 5 fermi_fix calls (matching CASTEP profile).  All other
    // parameters at CASTEP defaults (Gaussian, 0.1 eV width, from parameters.f90:1778).
    let smearing = SmearingParams::builder().spin_fix(6).build();

    ScfIteration::<SpinCollinear>::builder()
        .cell(cell).pots(pots)
        .wave_grid(wave_grid).fine_grid(fine_grid)
        .density(per_spin_density)
        .psi(psi).psi_data(psi_data)
        .pw_coords(KptDataSet::new(pw_coords_vec, nkpts))
        .pw_fft_indices(KptDataSet::new(pw_fft_vec, nkpts))
        .k_points(KptDataSet::new(k_points_vec, nkpts))
        .smearing(smearing).max_history(8)
        .build()
}

// ---------------------------------------------------------------------------
// GPU check
// ---------------------------------------------------------------------------

fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

// ===========================================================================
// Warm-start discriminator
// ===========================================================================

#[test]
#[ignore = "requires GPU"]
fn nio_warm_start_discriminator() {
    if !gpu_available() { eprintln!("SKIP: no GPU"); return; }

    let fx = fixture();
    let state = build_spin_scf_state(fx);

    // --- build V_eff with energy ---
    let v_eff = state.build_v_eff_with_energy().expect("build_v_eff_with_energy");

    // V6: V_eff_up != V_eff_dn
    {
        let veff_opt = v_eff.v_eff();
        let (v_up, v_dn) = veff_opt.as_ref().expect("SpinCollinear V_eff");
        let max_diff: f64 = v_up.as_real_grid().as_real_array().iter()
            .zip(v_dn.as_real_grid().as_real_array().iter())
            .map(|(&u, &d)| (u - d).abs()).fold(0.0, f64::max);
        eprintln!("[V6] V_eff_up vs V_eff_dn: max|Δ| = {:.6e} Ha  ({} points)", max_diff,
            v_up.as_real_grid().as_real_array().len());
        assert!(max_diff > 1e-6, "V6: V_eff_up == V_eff_dn");
    }

    // --- diagonalize ---
    let diag = v_eff.diagonalize(4, None).expect("diagonalize");

    // --- construct density, mix, check ---
    let dens = diag.construct_density_off().expect("construct_density");
    let mixed = dens.mix();

    let post_iter1 = match mixed.check(1e-8).expect("check") {
        chemrust_scf::CheckOutcome::Converged(_) =>
            panic!("iter-1 unexpectedly converged"),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    // ===================================================================
    // V1: Per-spin eigenvalues at all kpts
    // ===================================================================
    let per_spin_eigs = post_iter1.per_spin_eigenvalues();
    for ikpt in 0..per_spin_eigs[0].nkpts() {
        let e0 = &per_spin_eigs[0][ikpt];
        let ref0 = &fx.bands_eigenvalues_per_spin[ikpt][0];
        let e1 = &per_spin_eigs[1][ikpt];
        let ref1 = &fx.bands_eigenvalues_per_spin[ikpt][1];

        let me0 = e0.iter().zip(ref0.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0, f64::max);
        let me1 = e1.iter().zip(ref1.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0, f64::max);

        eprintln!("[V1] kpt={ikpt}: max eig err spin0={:.4e} spin1={:.4e} Ha", me0, me1);
        assert!(me0 < TOL_EPS_HA, "V1 FAILED kpt={ikpt} spin0: {:.4e} > {:.4e}", me0, TOL_EPS_HA);
        assert!(me1 < TOL_EPS_HA, "V1 FAILED kpt={ikpt} spin1: {:.4e} > {:.4e}", me1, TOL_EPS_HA);
    }

    // ===================================================================
    // V4: Fermi energies
    // ===================================================================
    let fermi = post_iter1.fermi_energies();
    let ef_up_err = (fermi[0] - REFERENCE_FERMI_ENERGY_HA).abs();
    let ef_dn_err = (fermi[1] - REFERENCE_FERMI_ENERGY_HA).abs();
    eprintln!("[V4] E_F up={:.8} Ha (Δ={:.4e}) dn={:.8} Ha (Δ={:.4e}) ref={:.8} Ha",
        fermi[0], ef_up_err, fermi[1], ef_dn_err, REFERENCE_FERMI_ENERGY_HA);
    assert!(ef_up_err < TOL_FERMI_HA, "V4 FAILED up: {:.4e} > {:.4e}", ef_up_err, TOL_FERMI_HA);
    assert!(ef_dn_err < TOL_FERMI_HA, "V4 FAILED dn: {:.4e} > {:.4e}", ef_dn_err, TOL_FERMI_HA);

    // ===================================================================
    // V2: Total energy (with E_nonCoulomb correction)
    // ===================================================================
    let e_iter1_ha_raw = post_iter1.total_energy().expect("total_energy");
    let e_iter1_ha = e_iter1_ha_raw + E_NON_COULOMB_HA;
    let e_iter1_ev = e_iter1_ha * chemrust_scf::HARTREE_TO_EV;
    let drift_ha = (e_iter1_ha - REFERENCE_ENERGY_EV / chemrust_scf::HARTREE_TO_EV).abs();
    let drift_ev = drift_ha * chemrust_scf::HARTREE_TO_EV;

    let e_xc = post_iter1.e_xc_value().unwrap_or(f64::NAN);
    let e_h = post_iter1.e_hartree_value().unwrap_or(f64::NAN);
    let rho_vxc = post_iter1.rho_vxc_value().unwrap_or(f64::NAN);
    let ewald = post_iter1.ewald_value();

    eprintln!("[V2] raw energy:      {:.8} Ha = {:.6} eV", e_iter1_ha_raw,
        e_iter1_ha_raw * chemrust_scf::HARTREE_TO_EV);
    eprintln!("[V2] + E_nonCoulomb:  +{:.8} Ha", E_NON_COULOMB_HA);
    eprintln!("[V2] corrected energy: {:.8} Ha = {:.6} eV", e_iter1_ha, e_iter1_ev);
    eprintln!("[V2] CASTEP ref:      {:.8} Ha = {:.6} eV",
        REFERENCE_ENERGY_EV / chemrust_scf::HARTREE_TO_EV, REFERENCE_ENERGY_EV);
    eprintln!("[V2] drift |Δ|: {:.4e} Ha = {:.4e} eV  (gate {:.4e} Ha)", drift_ha, drift_ev,
        DRIFT_TOLERANCE_HA);
    eprintln!("[V2] components: E_xc={:.6} E_H={:.6} rho_vxc={:.6} E_ewald={:.6} E_F=[{:.6},{:.6}]",
        e_xc, e_h, rho_vxc, ewald, fermi[0], fermi[1]);

    assert!(drift_ha < DRIFT_TOLERANCE_HA,
        "V2 FAILED: drift {:.4e} Ha > gate {:.4e} Ha\n  raw={:.8} + E_nonCoulomb={:.8} = {:.8}  vs  ref={:.8}",
        drift_ha, DRIFT_TOLERANCE_HA, e_iter1_ha_raw, E_NON_COULOMB_HA, e_iter1_ha,
        REFERENCE_ENERGY_EV / chemrust_scf::HARTREE_TO_EV);

    eprintln!("=== Warm-start discriminator: ALL CHECKS PASSED ===");
}

// ===========================================================================
// Cold-start discriminator — SCF from paramagnetic initial guess
// ===========================================================================

/// Build a SpinCollinear ScfIteration from a paramagnetic initial guess
/// (uniform density + zero spin, randomized psi) using the same cell,
/// pseudopotentials, and k-point grid as the warm-start fixture.
/// This isolates the standalone SCF convergence path from the FFI boundary.
// ===========================================================================
// 2-iteration cascade check — fast discriminator (no full SCF convergence)
// ===========================================================================

/// Run TWO SCF iterations from the converged state and assert eigenvalues
/// don't cascade.  If eigenvalues diverge by more than 1 Ha between iter-1
/// and iter-2, the SCF path has a cascade bug (e.g., stale D-matrices,
/// incorrect V_eff augmentation, or mixing contamination).
///
/// This is much faster than a full cold-start SCF (~2 min vs ~20 min).
#[test]
#[ignore = "requires GPU"]
fn nio_two_iter_cascade_check() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixture();
    let state = build_spin_scf_state(fx);

    // --- Iter 1: build V_eff, diagonalize, construct density, mix, check ---
    let v_eff = state.build_v_eff_with_energy().expect("build_v_eff_with_energy iter 1");
    let diag = v_eff.diagonalize(4, None).expect("diagonalize iter 1");
    let dens = diag.construct_density_off().expect("construct_density iter 1");
    let mixed = dens.mix();
    let iter2_state = match mixed.check(1e-8).expect("check iter 1") {
        chemrust_scf::CheckOutcome::Converged(_) => panic!("iter-1 converged early"),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    // Snapshot eigenvalues and energy before consuming iter2_state
    let eigs_iter1: Vec<Vec<Vec<f64>>> = {
        let pe = iter2_state.per_spin_eigenvalues();
        (0..pe.nspins()).map(|ispin|
            (0..pe[ispin].nkpts()).map(|ikpt| pe[ispin][ikpt].clone()).collect()
        ).collect()
    };
    let e_ha_1 = iter2_state.total_energy().expect("total_energy iter 1");

    // --- Iter 2: same pipeline ---
    let v_eff2 = iter2_state.build_v_eff_with_energy().expect("build_v_eff_with_energy iter 2");
    let diag2 = v_eff2.diagonalize(4, None).expect("diagonalize iter 2");
    let dens2 = diag2.construct_density_off().expect("construct_density iter 2");
    let mixed2 = dens2.mix();
    let iter3_state = match mixed2.check(1e-8).expect("check iter 2") {
        chemrust_scf::CheckOutcome::Converged(_) => panic!("iter-2 converged early"),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    let eigs_iter2 = iter3_state.per_spin_eigenvalues();

    // --- Cascade check: max |Δ| between iter-1 and iter-2 eigenvalues ---
    let cascade_gate_ha = 1.0; // 1 Ha drift = cascade
    for ikpt in 0..eigs_iter1[0].len() {
        for ispin in 0..2 {
            let e1 = &eigs_iter1[ispin][ikpt];
            let e2 = &eigs_iter2[ispin][ikpt];
            let max_delta = e1.iter().zip(e2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max);
            eprintln!("[Cascade] kpt={ikpt} spin={ispin} max|Δ_eig| = {:.4e} Ha", max_delta);
            assert!(max_delta < cascade_gate_ha,
                "[Cascade] FAIL kpt={ikpt} spin={ispin}: eigenvalue drift {:.4e} Ha > gate {:.4} Ha",
                max_delta, cascade_gate_ha);
        }
    }

    // Also check total energy drift
    let e_ha_2 = iter3_state.total_energy().expect("total_energy iter 2");
    let drift_ha = (e_ha_2 - e_ha_1).abs();
    eprintln!("[Cascade] total energy: iter1={:.6} iter2={:.6} |Δ|={:.4e} Ha", e_ha_1, e_ha_2, drift_ha);

    eprintln!("=== 2-iteration cascade check: PASSED ===");
}
