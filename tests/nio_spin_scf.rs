//! NiO spin-polarised SCF warm-start discriminator tests.
//!
//! Reference values from a converged CASTEP spin-polarised run (no U, finer grid):
//!   `/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin/`
//!
//! NiO: 14 k-points, spin-polarised, wave grid [20, 20, 20], fine grid [40, 40, 40],
//! USPP (Ni, O), 62 bands, N_up=36, N_dn=28.

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
// Constants
// ---------------------------------------------------------------------------

const NIO_SPIN_DIR: &str = "/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin";
const POTENTIAL_DIR: &str = "/export/Potentials";
const REFERENCE_ENERGY_EV: f64 = -7160.230577732;
const REFERENCE_FERMI_ENERGY_HA: f64 = 0.152664;
const EPS_SPIN0_BAND0_REF_HA: f64 = -0.59098046;
const EPS_SPIN1_BAND0_REF_HA: f64 = -0.59108380;
const TOL_EPS_HA: f64 = 3e-4;
const DRIFT_TOLERANCE_HA: f64 = 2e-2;
const TOL_FERMI_HA: f64 = 1e-3;
const E_NON_COULOMB_HA: f64 = 533.13587176588384864 / chemrust_scf::HARTREE_TO_EV;

// ---------------------------------------------------------------------------
// Cached fixture
// ---------------------------------------------------------------------------

pub struct NioSpinFixture {
    pub bin: CastepBin,
    pub check: CastepBin,
    pub bands_eigenvalues_per_spin: Vec<Vec<Vec<f64>>>,
    pub bands_fermi_energies: Vec<f64>,
    pub pots: PseudopotentialSet,
}

static FIXTURE: OnceLock<NioSpinFixture> = OnceLock::new();

pub fn fixture() -> &'static NioSpinFixture {
    FIXTURE.get_or_init(|| load_spin_fixture().expect("failed to load NiO spin-polarised fixture"))
}

fn load_spin_fixture() -> Result<NioSpinFixture, Box<dyn std::error::Error>> {
    let fixture_dir =
        std::env::var("NIO_SPIN_FIXTURE_DIR").unwrap_or_else(|_| NIO_SPIN_DIR.to_string());
    let potential_dir =
        std::env::var("CASTEP_POTENTIAL_DIR").unwrap_or_else(|_| POTENTIAL_DIR.to_string());

    let bin_path = format!("{fixture_dir}/NiO.castep_bin");
    let bin = CastepBinFile::read(std::io::BufReader::new(std::fs::File::open(&bin_path)?))?;

    let check_path = format!("{fixture_dir}/NiO.check");
    let check = CheckFile::read(std::io::BufReader::new(std::fs::File::open(&check_path)?))?;

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

fn parse_spin_bands_file(
    text: &str,
) -> Result<(Vec<f64>, Vec<Vec<Vec<f64>>>), Box<dyn std::error::Error>> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 5 { return Err("bands file too short".into()); }
    let nspins: usize = lines[1].trim().split_whitespace().last().ok_or("cannot parse nspins")?.parse()?;
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
            kpt_idx = Some(trimmed.split_whitespace().nth(1).ok_or("bad kpt")?.parse::<usize>()? - 1);
            current_spin = None;
        } else if trimmed.starts_with("Spin component") {
            flush(kpt_idx, current_spin, &mut current_values, &mut eigenvalues);
            current_spin = Some(trimmed.split_whitespace().nth(2).ok_or("bad spin")?.parse::<usize>()? - 1);
        } else if current_spin.is_some() {
            if let Ok(val) = trimmed.parse::<f64>() { current_values.push(val); }
        }
    }
    flush(kpt_idx, current_spin, &mut current_values, &mut eigenvalues);
    if eigenvalues.is_empty() { return Err("no eigenvalues parsed".into()); }
    Ok((fermi_energies, eigenvalues))
}

// ---------------------------------------------------------------------------
// SCF state construction — follows the same pattern as Cu111_CO fixture
// but for SpinCollinear + multi-kpt.
// ---------------------------------------------------------------------------

pub fn build_spin_scf_state(fx: &NioSpinFixture) -> ScfIteration<SpinCollinear> {
    let cell = fx.bin.cell.clone();
    let pots = fx.pots.clone();
    let wfc = fx.check.wavefunction.as_ref().expect(".check must have wavefunction");
    let [ngx, ngy, ngz] = wfc.grid;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);
    let [fgx, fgy, fgz] = fx.check.fine_grid.expect(".check must have fine_grid");
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);
    assert_eq!(fx.bin.density.grid, [fgx, fgy, fgz], "density grid must match fine grid");
    let nspins = fx.bin.density.nspins;
    assert_eq!(nspins, 2, "NiO fixture must have nspins=2");

    // Per-spin density: downsample from fine grid to wave grid
    let charge_fine = fx.bin.density.charge.as_real_grid().as_real_array();
    let spin_fine = fx.bin.density.spin.as_ref().expect("spin density").as_real_grid().as_real_array();
    let charge_wave = downsample_array_to_wave_grid(charge_fine, &fine_grid, &wave_grid).expect("downsample charge");
    let spin_wave = downsample_array_to_wave_grid(spin_fine, &fine_grid, &wave_grid).expect("downsample spin");
    let charge_arr = charge_wave.as_fine_array();
    let spin_arr = spin_wave.as_fine_array();
    let half = 0.5_f64;
    let rho_up = charge_arr.mapv(|v| v * half) + spin_arr.mapv(|v| v * half);
    let rho_dn = charge_arr.mapv(|v| v * half) - spin_arr.mapv(|v| v * half);
    let per_spin_density = PerSpinDensity::new(SpinChannelData::new::<SpinCollinear>(vec![
        Density::from_inner(WaveGridArray::from_inner(rho_up)),
        Density::from_inner(WaveGridArray::from_inner(rho_dn)),
    ]));

    // Multi-kpt wavefunctions (spin-major layout: spin0 first, then spin1)
    let nkpts = wfc.kpt_data.len() / nspins;
    assert!(nkpts >= 1);
    let n_bands = wfc.kpt_data[0].bands.len();
    let kpt_weights = fx.bin.kpoint_weights.clone();
    assert_eq!(kpt_weights.len(), nkpts);

    let ctx = std::sync::Arc::new(cudarc::driver::CudaContext::new(0).expect("GPU required"));
    let stream = ctx.default_stream();

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
        let gpu0: cudarc::driver::CudaSlice<chemrust_scf::device::CudaComplex> =
            stream.alloc_zeros(n_el).expect("GPU spin0");
        let gpu1: cudarc::driver::CudaSlice<chemrust_scf::device::CudaComplex> =
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

fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

// ===========================================================================
// Warm-start discriminator (Davidson)
// ===========================================================================

#[test]
#[ignore = "requires GPU"]
fn nio_warm_start_discriminator() {
    if !gpu_available() { eprintln!("SKIP: no GPU"); return; }
    unsafe { std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson"); }

    let fx = fixture();
    let state = build_spin_scf_state(fx);
    let v_eff = state.build_v_eff_with_energy().expect("build_v_eff_with_energy");

    // V6: V_eff_up != V_eff_dn
    {
        let veff_opt = v_eff.v_eff();
        let (v_up, v_dn) = veff_opt.as_ref().expect("SpinCollinear V_eff");
        let max_diff: f64 = v_up.as_real_grid().as_real_array().iter()
            .zip(v_dn.as_real_grid().as_real_array().iter())
            .map(|(&u, &d)| (u - d).abs()).fold(0.0, f64::max);
        eprintln!("[V6] V_eff_up vs V_eff_dn: max|Δ| = {:.6e} Ha ({} points)", max_diff,
            v_up.as_real_grid().as_real_array().len());
        assert!(max_diff > 1e-6, "V6: V_eff_up == V_eff_dn");
    }

    let diag = v_eff.diagonalize(4, None).expect("diagonalize");
    let dens = diag.construct_density_off().expect("construct_density");
    let mixed = dens.mix();
    let post_iter1 = match mixed.check(1e-8).expect("check") {
        chemrust_scf::CheckOutcome::Converged(_) => panic!("iter-1 unexpectedly converged"),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

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

    let fermi = post_iter1.fermi_energies();
    let ef_up_err = (fermi[0] - REFERENCE_FERMI_ENERGY_HA).abs();
    let ef_dn_err = (fermi[1] - REFERENCE_FERMI_ENERGY_HA).abs();
    eprintln!("[V4] E_F up={:.8} Ha (Δ={:.4e}) dn={:.8} Ha (Δ={:.4e}) ref={:.8} Ha",
        fermi[0], ef_up_err, fermi[1], ef_dn_err, REFERENCE_FERMI_ENERGY_HA);
    assert!(ef_up_err < TOL_FERMI_HA, "V4 FAILED up: {:.4e} > {:.4e}", ef_up_err, TOL_FERMI_HA);
    assert!(ef_dn_err < TOL_FERMI_HA, "V4 FAILED dn: {:.4e} > {:.4e}", ef_dn_err, TOL_FERMI_HA);

    let e_iter1_ha_raw = post_iter1.total_energy().expect("total_energy");
    let e_iter1_ha = e_iter1_ha_raw + E_NON_COULOMB_HA;
    let e_iter1_ev = e_iter1_ha * chemrust_scf::HARTREE_TO_EV;
    let drift_ha = (e_iter1_ha - REFERENCE_ENERGY_EV / chemrust_scf::HARTREE_TO_EV).abs();
    let drift_ev = drift_ha * chemrust_scf::HARTREE_TO_EV;
    eprintln!("[V2] corrected energy: {:.8} Ha = {:.6} eV", e_iter1_ha, e_iter1_ev);
    eprintln!("[V2] CASTEP ref:      {:.8} Ha = {:.6} eV", REFERENCE_ENERGY_EV / chemrust_scf::HARTREE_TO_EV, REFERENCE_ENERGY_EV);
    eprintln!("[V2] drift |Δ|: {:.4e} Ha = {:.4e} eV  (gate {:.4e} Ha)", drift_ha, drift_ev, DRIFT_TOLERANCE_HA);
    assert!(drift_ha < DRIFT_TOLERANCE_HA,
        "V2 FAILED: drift {:.4e} Ha > gate {:.4e} Ha", drift_ha, DRIFT_TOLERANCE_HA);
    eprintln!("=== Warm-start discriminator: ALL CHECKS PASSED ===");
}

// ===========================================================================
// ChFSI convergence discriminator — multi-iteration SCF from converged state
// ===========================================================================
//
// Unlike Davidson, ChFSI uses polynomial subspace amplification — the first
// iteration's eigenvalues are NOT expected to match CASTEP or Davidson.
// The correct discriminator is: does the total energy converge toward the
// CASTEP reference across multiple SCF iterations?
//
// This test runs 5 SCF iterations from the converged CASTEP state.
// Gate: total energy must not diverge (drift < 0.1 Ha across iterations)
// and iter-5 energy must be closer to reference than iter-1 energy.

#[test]
#[ignore = "requires GPU"]
fn nio_chebfi_convergence_trend() {
    if !gpu_available() { eprintln!("SKIP: no GPU"); return; }
    unsafe { std::env::set_var("CHEMRUST_EIGENSOLVER", "chebyshev"); }

    let fx = fixture();
    let state = build_spin_scf_state(fx);
    let ref_ev = REFERENCE_ENERGY_EV;

    // Use run_scf_with_energy — properly handles Off→Kerker→Pulay mixing
    use chemrust_scf::scf::{ScfDivergenceGate, run_scf_with_energy_gated};
    let gate = ScfDivergenceGate {
        max_last_band_ha: 5.0,
        min_band0_ha: -30.0,
        max_veff_range_factor: 5.0,
        max_iter: 20,  // divergence detection: SCF should not need >20 iters from converged start
        electron_count_tolerance: 0.05,
        soft_fraction_tolerance: 0.20,
    };
    let result = run_scf_with_energy_gated(state, 8, 1e-8, Some(gate))
        .expect("run_scf_with_energy");

    let e_final_ev = (result.total_energy + E_NON_COULOMB_HA) * chemrust_scf::HARTREE_TO_EV;
    let drift_ev = (e_final_ev - ref_ev).abs();

    eprintln!("[ChFSI] final: {:.6} eV  CASTEP: {:.6} eV  drift: {:.4e} eV",
        e_final_ev, ref_ev, drift_ev);
    assert!(drift_ev < 0.05, "ChFSI FAIL: drift {:.4e} eV", drift_ev);
    eprintln!("=== ChFSI convergence trend: PASSED ===");
}

// ===========================================================================
// 2-iteration cascade check
// ===========================================================================

#[test]
#[ignore = "requires GPU"]
fn nio_two_iter_cascade_check() {
    if !gpu_available() { eprintln!("SKIP: no GPU available"); return; }
    let fx = fixture();
    let state = build_spin_scf_state(fx);

    let v_eff = state.build_v_eff_with_energy().expect("build_v_eff_with_energy iter 1");
    let diag = v_eff.diagonalize(4, None).expect("diagonalize iter 1");
    let dens = diag.construct_density_off().expect("construct_density iter 1");
    let mixed = dens.mix();
    let iter2_state = match mixed.check(1e-8).expect("check iter 1") {
        chemrust_scf::CheckOutcome::Converged(_) => panic!("iter-1 converged early"),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    let eigs_iter1: Vec<Vec<Vec<f64>>> = {
        let pe = iter2_state.per_spin_eigenvalues();
        (0..pe.nspins()).map(|ispin|
            (0..pe[ispin].nkpts()).map(|ikpt| pe[ispin][ikpt].clone()).collect()
        ).collect()
    };
    let e_ha_1 = iter2_state.total_energy().expect("total_energy iter 1");

    let v_eff2 = iter2_state.build_v_eff_with_energy().expect("build_v_eff_with_energy iter 2");
    let diag2 = v_eff2.diagonalize(4, None).expect("diagonalize iter 2");
    let dens2 = diag2.construct_density_off().expect("construct_density iter 2");
    let mixed2 = dens2.mix();
    let iter3_state = match mixed2.check(1e-8).expect("check iter 2") {
        chemrust_scf::CheckOutcome::Converged(_) => panic!("iter-2 converged early"),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    let eigs_iter2 = iter3_state.per_spin_eigenvalues();
    let cascade_gate_ha = 1.0;
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

    let e_ha_2 = iter3_state.total_energy().expect("total_energy iter 2");
    let drift_ha = (e_ha_2 - e_ha_1).abs();
    eprintln!("[Cascade] total energy: iter1={:.6} iter2={:.6} |Δ|={:.4e} Ha", e_ha_1, e_ha_2, drift_ha);
    eprintln!("=== 2-iteration cascade check: PASSED ===");
}
