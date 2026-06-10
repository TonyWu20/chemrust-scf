//! NiO spin-polarised SCF discriminator tests.
//!
//! Reference values from a converged CASTEP spin-polarised run (no U, finer grid):
//!   `/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin/`
//!
//! NiO: 14 k-points, spin-polarised, wave grid [32, 32, 32], fine grid [40, 40, 40],
//! USPP (Ni, O), 62 bands, N_up=36, N_dn=28.
//!
//! ## Discriminator assertions (V1–V7)
//!
//! | ID | Assertion                                    | Tolerance       | Source              |
//! |----|----------------------------------------------|-----------------|---------------------|
//! | V1 | max|eps - eps_ref| per spin                | < 1e-4 Ha       | NiO.bands:12,75     |
//! | V2 | |E_total - E_ref|                           | < 2.7e-5 eV     | NiO.castep:774418   |
//! | V3 | |spin_density - (-0.0619641)|               | rel < 1e-4      | NiO.castep:774415   |
//! | V4 | |E_Fermi - 0.152664|                        | < 1e-4 Ha       | NiO.bands:5         |
//! | V5 | max occupancy difference                     | < 1e-3          | (self-consistent)   |
//! | V6 | V_eff_up != V_eff_dn (qualitative)           | —               | (self-consistent)   |
//! | V7 | N_up ≈ 36.00, N_dn ≈ 28.00                  | ±0.01           | NiO.bands:3         |
//!
//! ## Sanity
//! - SUM occ = 64.00 (charge neutrality)
//! - ρ_spin(r) != 0 identically (non-trivial spin polarisation)

use std::io::BufReader;

use chemrust_hamiltonian_core::{
    CastepBin, CastepBinFile, CheckFile, GVectorGrid, SpinCollinear,
};
use chemrust_scf::{
    ColumnDistributed, Density, KPoint, PerSpinDensity, SmearingParams,
    SmearingScheme, SpinChannelData, WaveGridArray, WavefunctionSet, pw_coords_to_fft_indices,
};
use num_complex::Complex64;

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
const REFERENCE_SPIN_DENSITY: f64 = -0.0619641;

/// Number of up-spin electrons from NiO.bands:3
const N_UP_REF: f64 = 36.00;

/// Number of down-spin electrons from NiO.bands:3
const N_DN_REF: f64 = 28.00;

/// Eigenvalue[kpt=0, spin=0, band=0] from NiO.bands:12 (Ha)
#[allow(unused)]
const EPS_SPIN0_BAND0_REF_HA: f64 = -0.59098046;

/// Eigenvalue[kpt=0, spin=1, band=0] from NiO.bands:75 (Ha)
#[allow(unused)]
const EPS_SPIN1_BAND0_REF_HA: f64 = -0.59108380;

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// V1: Per-spin eigenvalue tolerance (Ha)
const TOL_EPS_HA: f64 = 1e-4;

/// V2: Total energy tolerance (eV) — 2.7e-5 eV ≈ 1e-6 Ha
const TOL_ENERGY_EV: f64 = 2.7e-5;

/// V3: Relative tolerance for integrated spin density
const TOL_SPIN_DENSITY_REL: f64 = 1e-4;

/// V4: Fermi energy tolerance (Ha)
const TOL_FERMI_HA: f64 = 1e-4;

/// V5: Maximum per-band occupancy difference
const TOL_OCC: f64 = 1e-3;

/// V7: Electron count tolerance
const TOL_COUNTS: f64 = 0.01;

// ---------------------------------------------------------------------------
// Helper: parse spin-polarised .bands file
// ---------------------------------------------------------------------------

/// Parse a spin-polarised CASTEP `.bands` file.
///
/// Returns `(fermi_energies, per_spin_eigenvalues)` where `per_spin_eigenvalues[i]`
/// is the flat list of eigenvalue vectors per k-point for spin channel `i`.
///
/// ## Format (spin-polarised)
/// ```text
/// Number of k-points    14
/// Number of spin components 2
/// Number of electrons  36.00     28.00
/// Number of eigenvalues     62    62
/// Fermi energies (in atomic units)     0.152664    0.152664
/// ...
/// K-point     1  0.33333333  0.33333333  0.33333333  0.07407407
/// Spin component 1
///    -0.59098046
///    ...
/// Spin component 2
///    -0.59108380
///    ...
/// K-point     2 ...
/// ```
#[allow(unused)]
fn parse_spin_bands_file(text: &str) -> (Vec<f64>, Vec<Vec<Vec<f64>>>) {
    let lines: Vec<&str> = text.lines().collect();

    // Parse header
    let nspins = {
        let header = lines[1].trim();
        let parts: Vec<&str> = header.split_whitespace().collect();
        // "Number of spin components 2"
        parts[parts.len() - 1].parse::<usize>().expect("nspins")
    };

    let fermi_energies = {
        let header = lines[4].trim();
        // "Fermi energies (in atomic units)     0.152664    0.152664"
        let parts: Vec<&str> = header.split_whitespace().collect();
        parts[5..]
            .iter()
            .map(|s| s.parse::<f64>().expect("fermi energy"))
            .collect::<Vec<f64>>()
    };

    // Collect eigenvalues per (kpoint, spin)
    // Each spin section has 62 eigenvalues (n_bands), one per line
    let mut eigenvalues: Vec<Vec<Vec<f64>>> = Vec::new(); // [kpt][spin][band]
    let mut kpt_idx: Option<usize> = None;
    let mut current_spin: Option<usize> = None;
    let mut current_values: Vec<f64> = Vec::new();

    for line in lines.iter().skip(5) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with("K-point") {
            // Flush previous spin block
            if let (Some(kpt), Some(spin)) = (kpt_idx, current_spin) {
                while eigenvalues.len() <= kpt {
                    eigenvalues.push(vec![Vec::new(); nspins]);
                }
                eigenvalues[kpt][spin] = current_values.clone();
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            let kpt: usize = parts[1].parse().expect("k-point index");
            kpt_idx = Some(kpt - 1); // CASTEP 1-based
            current_spin = None;
            current_values = Vec::new();
        } else if trimmed.starts_with("Spin component") {
            // Flush previous spin block for same kpt
            if let (Some(kpt), Some(spin)) = (kpt_idx, current_spin) {
                while eigenvalues.len() <= kpt {
                    eigenvalues.push(vec![Vec::new(); nspins]);
                }
                eigenvalues[kpt][spin] = current_values.clone();
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            let spin: usize = parts[2].parse().expect("spin index");
            current_spin = Some(spin - 1); // CASTEP 1-based
            current_values = Vec::new();
        } else if let Some(_spin) = current_spin {
            // Try parsing as eigenvalue
            if let Ok(val) = trimmed.parse::<f64>() {
                current_values.push(val);
            }
        }
    }

    // Flush last block
    if let (Some(kpt), Some(spin)) = (kpt_idx, current_spin) {
        while eigenvalues.len() <= kpt {
            eigenvalues.push(vec![Vec::new(); nspins]);
        }
        eigenvalues[kpt][spin] = current_values;
    }

    (fermi_energies, eigenvalues)
}

// ---------------------------------------------------------------------------
// Fixture loader
// ---------------------------------------------------------------------------

/// Pre-loaded CASTEP reference data for NiO spin-polarised run.
#[allow(unused)]
pub struct NiOSpinFixture {
    pub bin: CastepBin,
    pub check: CastepBin,
    pub bands_eigenvalues_per_spin: Vec<Vec<Vec<f64>>>, // [kpt][spin][band]
    pub bands_fermi_energies: Vec<f64>,
}

/// Load the NiO spin-polarised fixture from disk.
///
/// Requires the fixture directory to be present (set `NIO_SPIN_FIXTURE_DIR` env
/// variable to override the default path).  Falls back to
/// `/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin/`.
#[allow(unused)]
fn load_spin_fixture() -> NiOSpinFixture {
    let fixture_dir =
        std::env::var("NIO_SPIN_FIXTURE_DIR").unwrap_or_else(|_| NIO_SPIN_DIR.to_string());

    // 1. Load .castep_bin (cell, density, eigenvalues)
    let bin_path = format!("{fixture_dir}/NiO.castep_bin");
    let bin_file = std::fs::File::open(&bin_path)
        .unwrap_or_else(|e| panic!("cannot open {bin_path}: {e}"));
    let bin_reader = BufReader::new(bin_file);
    let bin = CastepBinFile::read(bin_reader)
        .unwrap_or_else(|e| panic!("failed to parse {bin_path}: {e}"));

    // 2. Load .check (wavefunction + fine grid)
    let check_path = format!("{fixture_dir}/NiO.check");
    let check_file = std::fs::File::open(&check_path)
        .unwrap_or_else(|e| panic!("cannot open {check_path}: {e}"));
    let check_reader = BufReader::new(check_file);
    let check = CheckFile::read(check_reader)
        .unwrap_or_else(|e| panic!("failed to parse {check_path}: {e}"));

    // 3. Load .bands (eigenvalues + Fermi energies per spin)
    let bands_path = format!("{fixture_dir}/NiO.bands");
    let bands_text = std::fs::read_to_string(&bands_path)
        .unwrap_or_else(|e| panic!("cannot read {bands_path}: {e}"));
    let (fermi_energies, eigenvalues_per_spin) = parse_spin_bands_file(&bands_text);

    NiOSpinFixture {
        bin,
        check,
        bands_eigenvalues_per_spin: eigenvalues_per_spin,
        bands_fermi_energies: fermi_energies,
    }
}

// ---------------------------------------------------------------------------
// SCF state builder (spin-polarised)
// ---------------------------------------------------------------------------

/// Build a fully-initialised `ScfIteration<SpinCollinear>` from the NiO fixture.
///
/// Extracts per-spin wavefunctions from the `.check` file (spin-major layout:
/// `kpt_data[0..nkpts]` = spin0, `kpt_data[nkpts..]` = spin1) and constructs
/// two-density `PerSpinDensity` from the `.castep_bin`.
///
/// # Panics
///
/// Panics if the .check file does not have the expected spin-polarised layout.
#[allow(unused)]
fn build_spin_scf_state(fx: &NiOSpinFixture) -> ScfIterationSnapshot {
    let cell = fx.bin.cell.clone();
    let nspins = fx.bin.eigenvalues.nspins.max(1);
    assert_eq!(nspins, 2, "NiO fixture must be spin-polarised (nspins=2)");

    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction section");
    let wave_grid_dims = wfc.grid; // [ngx, ngy, ngz]
    let [ngx, ngy, ngz] = wave_grid_dims;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);

    let fine_grid_dims = fx.check.fine_grid.expect(".check must have fine_grid");
    let [fgx, fgy, fgz] = fine_grid_dims;
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    // kpt_data layout: spin-major, kpoint-inner
    // kpt_data[0..nkpts] = spin0 for each kpoint
    // kpt_data[nkpts..]  = spin1 for each kpoint
    let nkpts = wfc.kpt_data.len() / nspins;
    let kpt0_spin0 = &wfc.kpt_data[0];
    let kpt0_spin1 = &wfc.kpt_data[nkpts]; // first kpt, second spin

    // First k-point (k_frac ≈ [1/3, 1/3, 1/3]) for single-kpoint tests
    let n_pw = kpt0_spin0.nplw;
    assert_eq!(
        kpt0_spin1.nplw, n_pw,
        "nplw must match across spins for same k-point"
    );
    let n_bands = kpt0_spin0.bands.len();
    assert_eq!(
        kpt0_spin1.bands.len(),
        n_bands,
        "n_bands must match across spins"
    );

    // Per-spin wavefunction data as flat vecs
    let psi_data_spin0: Vec<Complex64> = kpt0_spin0.bands.iter().flatten().copied().collect();
    let psi_data_spin1: Vec<Complex64> = kpt0_spin1.bands.iter().flatten().copied().collect();

    // Construct PerSpinPwCoefficients (GPU-backed; placeholder CPU vecs)
    let psi_data = vec![psi_data_spin0, psi_data_spin1];
    let n_bands_total = psi_data[0].len() / n_pw;
    let psi_sets: Vec<WavefunctionSet<ColumnDistributed>> = psi_data
        .iter()
        .map(|data| WavefunctionSet::<ColumnDistributed>::new(data.clone(), n_bands_total, n_pw))
        .collect();

    // TODO: GPU allocation for PerSpinPwCoefficients requires CudaContext.
    // The GPU-side psi will be constructed inside `run_scf`.
    // For now this is a placeholder that demonstrates the shape.

    // Per-spin density from .castep_bin
    // bin.density.charge is total density ρ_up + ρ_down (nspins=1 broadcast)
    // bin.density.spin is spin density ρ_up - ρ_down (Option)
    let dens_charge = fx.bin.density.charge.as_real_grid().as_real_array().clone();
    let dens_spin = fx
        .bin
        .density
        .spin
        .as_ref()
        .map(|s| s.as_real_grid().as_real_array().clone());

    // Reconstruct ρ_up, ρ_down from total and spin densities:
    // ρ_up   = (ρ_total + ρ_spin) / 2
    // ρ_down = (ρ_total - ρ_spin) / 2
    let (density_up, density_dn) = if let Some(spin_arr) = dens_spin.as_ref() {
        let half = 0.5_f64;
        let up = dens_charge.mapv(|v| v * half) + spin_arr.mapv(|v| v * half);
        let dn = dens_charge.mapv(|v| v * half) - spin_arr.mapv(|v| v * half);
        (up, dn)
    } else {
        // If no spin density available, approximate: charge is close to double the up
        let half = 0.5_f64;
        let up = dens_charge.mapv(|v| v * half);
        let dn = up.clone();
        (up, dn)
    };

    let per_spin_density = PerSpinDensity::new(SpinChannelData::new::<SpinCollinear>(vec![
        Density::from_inner(WaveGridArray::from_inner(density_up)),
        Density::from_inner(WaveGridArray::from_inner(density_dn)),
    ]));

    let pw_coords = kpt0_spin0.pw_grid_coord.clone();
    let pw_fft_indices = pw_coords_to_fft_indices(&pw_coords, &wave_grid);

    let k_point = KPoint {
        coords: kpt0_spin0.coords,
    };

    // Smearing: Gaussian, 0.1 eV (CASTEP default). NiO cell no smearing width specified.
    let smearing = SmearingParams {
        width: 0.1 * chemrust_scf::EV_TO_HARTREE,
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: SmearingScheme::Gaussian,
    };

    ScfIterationSnapshot {
        density: per_spin_density,
        psi_data,
        psi_sets,
        pw_coords,
        pw_fft_indices,
        k_point,
        smearing,
        cell,
        wave_grid,
        fine_grid,
        n_bands,
        n_pw,
    }
}

/// Snapshot of the SCF state components needed for discriminator tests.
///
/// This avoids constructing the full type-state `ScfIteration<SpinCollinear>`
/// which requires GPU device handles and complex phase transitions.  Tests
/// that need the full `ScfIteration` can construct it manually.
#[allow(unused)]
pub struct ScfIterationSnapshot {
    pub density: PerSpinDensity,
    pub psi_data: Vec<Vec<Complex64>>,
    pub psi_sets: Vec<WavefunctionSet<ColumnDistributed>>,
    pub pw_coords: Vec<[i32; 3]>,
    pub pw_fft_indices: Vec<i32>,
    pub k_point: KPoint,
    pub smearing: SmearingParams,
    pub cell: chemrust_hamiltonian_core::CellGeometry,
    pub wave_grid: GVectorGrid,
    pub fine_grid: GVectorGrid,
    pub n_bands: usize,
    pub n_pw: usize,
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Check if CUDA-capable GPU is available at device 0.
#[allow(unused)]
fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

// ===========================================================================
// V1: Per-spin eigenvalue comparison
// ===========================================================================

/// V1: Maximum eigenvalue difference < 1e-4 Ha per spin channel.
///
/// Compares the Rust SCF-calculated eigenvalues against CASTEP reference from
/// NiO.bands for the first k-point.  Validates that the eigensolver produces
/// correct per-spin eigenvalues at the first k-point.
#[test]
#[ignore = "requires GPU and full spin-polarised SCF pipeline"]
fn v1_eigenvalues_match_per_spin() {
    // TODO: implement when the spin-polarised SCF pipeline is operational
    // 1. Load fixture (spin-polarised .check + .castep_bin)
    // 2. Build ScfIteration<SpinCollinear, Initialized, MixingOff>
    // 3. Run a single-shot diagonalize (or 2 SCF iterations)
    // 4. Extract per-spin eigenvalues from the result
    // 5. Compare against bands_eigenvalues_per_spin[0][spin]
    //    (kpt=0, spin=0 and spin=1)
    // 6. Assert max|eps - eps_ref| < TOL_EPS_HA for each spin channel
    //
    // Reference:
    //   EPS_SPIN0_BAND0_REF_HA = -0.59098046  (NiO.bands:12)
    //   EPS_SPIN1_BAND0_REF_HA = -0.59108380  (NiO.bands:75)
    todo!("V1: eigenvalue comparison -- needs spin-polarised SCF pipeline");
}

// ===========================================================================
// V2: Total energy comparison
// ===========================================================================

/// V2: Total energy difference < 2.7e-5 eV (≈ 1e-6 Ha).
///
/// Validates that the Rust SCF's converged total energy matches CASTEP's
/// reference to within chemical accuracy.  This is the most stringent
/// discriminator: the total energy integrates all sources of error
/// (eigensolver, density construction, V_eff assembly, augmentation).
#[test]
#[ignore = "requires GPU and full spin-polarised SCF pipeline"]
fn v2_total_energy_matches_reference() {
    // TODO: implement when the spin-polarised SCF pipeline is operational
    // 1. Build ScfIteration<SpinCollinear> from fixture
    // 2. Run SCF to convergence (or a fixed number of iterations)
    // 3. Extract total_energy from the converged state
    // 4. Assert |E_total * HARTREE_TO_EV - REFERENCE_ENERGY_EV| < TOL_ENERGY_EV
    //
    // Reference:
    //   REFERENCE_ENERGY_EV = -7160.230577732  (NiO.castep:774418)
    //   TOL_ENERGY_EV = 2.7e-5  (≈ 1e-6 Ha)
    todo!("V2: total energy comparison -- requires full SCF convergence");
}

// ===========================================================================
// V3: Integrated spin density
// ===========================================================================

/// V3: Integrated spin density relative error < 1e-4.
///
/// Compares 2*∫ρ_spin(r) dr (the integrated spin density) against CASTEP's
/// reference.  This directly validates the quality of the spin-polarised
/// density: if ρ_up and ρ_down are both correct, their difference should
/// match CASTEP's reference.
///
/// The spin density is a more sensitive discriminator than total energy
/// alone — it can reveal cancellation errors in the total energy.
#[test]
#[ignore = "requires GPU and full spin-polarised SCF pipeline"]
fn v3_spin_density_matches_reference() {
    // TODO: implement when the spin-polarised density is available
    // 1. Build PerSpinDensity from SCF cycle or fixture
    // 2. Compute spin() -> ρ_up - ρ_down
    // 3. Integrate over the fine grid to get 2*∫ρ_spin = sum(rho_spin)
    // 4. Assert |integrated - REFERENCE_SPIN_DENSITY| / |REFERENCE_SPIN_DENSITY| < TOL_SPIN_DENSITY_REL
    //
    // Reference:
    //   REFERENCE_SPIN_DENSITY = -0.0619641  (NiO.castep:774415)
    //   TOL_SPIN_DENSITY_REL = 1e-4
    //
    // Sanity check: ρ_spin(r) must not be identically zero everywhere.
    todo!("V3: integrated spin density comparison -- needs spin-polarised density");
}

// ===========================================================================
// V4: Fermi energy
// ===========================================================================

/// V4: Fermi energy difference < 1e-4 Ha per spin.
///
/// The Fermi energy is determined by the occupation search (filling
/// eigenvalues up to charge neutrality).  A correct eigensolver + density
/// should produce Fermi energies matching CASTEP's reference.
#[test]
#[ignore = "requires GPU and full spin-polarised SCF pipeline"]
fn v4_fermi_energy_matches_reference() {
    // TODO: implement when spin-polarised occupation search is operational
    // 1. Extract Fermi energies from converged SCF state
    // 2. Assert |E_F[ispin] - REFERENCE_FERMI_ENERGY_HA| < TOL_FERMI_HA
    //
    // Reference:
    //   REFERENCE_FERMI_ENERGY_HA = 0.152664  (NiO.bands:5, both spins)
    //   TOL_FERMI_HA = 1e-4
    todo!("V4: Fermi energy comparison -- needs occupation search");
}

// ===========================================================================
// V5: Occupancy comparison
// ===========================================================================

/// V5: Maximum per-band occupancy difference < 1e-3.
///
/// Occupancies are a sensitive discriminator: they depend on both the
/// eigenvalue spectrum and the Fermi energy.  A small eigenvalue error
/// near the Fermi surface can cause large occupancy changes.
#[test]
#[ignore = "requires GPU and full spin-polarised SCF pipeline"]
fn v5_occupancies_match_reference() {
    // TODO: implement when spin-polarised occupancy calculation is operational
    // 1. Extract per-spin occupancies from converged SCF state
    // 2. Compare against CASTEP reference (from .check eigenvalues)
    // 3. Assert max|occ - occ_ref| < TOL_OCC
    //
    // Reference tolerance:
    //   TOL_OCC = 1e-3
    todo!("V5: occupancy comparison -- needs SCF convergence");
}

// ===========================================================================
// V6: V_eff qualitative spin-polarisation check
// ===========================================================================

/// V6: V_eff_up != V_eff_dn qualitatively.
///
/// A trivial but important sanity: for a spin-polarised system, the
/// effective potential must differ between spin channels.  If V_eff_up ==
/// V_eff_down to machine precision, the spin-polarised potential assembly
/// has failed (e.g., the exchange-correlation functional returned a
/// non-spin-polarised result).
#[test]
#[ignore = "requires GPU and full spin-polarised SCF pipeline"]
fn v6_veff_spin_polarised() {
    // TODO: implement when VEffBuilder<SpinCollinear> is operational
    // 1. Build V_eff for both spins (via BuildVEffWithEnergy or directly)
    // 2. Compute max|V_eff_up - V_eff_dn| over the fine grid
    // 3. Assert the difference is non-trivial (> 1e-6 Ha threshold)
    //
    // This is a qualitative check — any non-trivial difference indicates
    // the spin-polarised V_eff assembly is active.
    todo!("V6: V_eff spin-polarisation check -- needs VEffBuilder<SpinCollinear>");
}

// ===========================================================================
// V7: Electron count per spin
// ===========================================================================

/// V7: N_up ≈ 36.00, N_dn ≈ 28.00 (±0.01).
///
/// The total number of electrons per spin channel must match the
/// spin-polarised occupation sum.  This validates that the occupation
/// search correctly distributes the 64 valence electrons across the two
/// spin channels.
#[test]
#[ignore = "requires GPU and full spin-polarised SCF pipeline"]
fn v7_electron_counts_per_spin() {
    // TODO: implement when spin-polarised occupation search is operational
    // 1. Sum per-spin occupancies
    // 2. Assert |N_up - 36.00| < TOL_COUNTS
    // 3. Assert |N_dn - 28.00| < TOL_COUNTS
    // 4. Assert N_up + N_dn ≈ 64.00 (charge neutrality)
    //
    // Reference:
    //   N_UP_REF = 36.00  (NiO.bands:3)
    //   N_DN_REF = 28.00  (NiO.bands:3)
    //   TOL_COUNTS = 0.01
    todo!("V7: electron count per spin -- needs occupation search");
}

// ===========================================================================
// Sanity: charge neutrality and non-trivial spin density
// ===========================================================================

/// Sanity check: total occupancy sums to 64.00 (charge neutrality).
///
/// This must hold even before the full SCF pipeline is operational: the
/// charge density from .castep_bin integrated over the grid must give
/// the correct N_electrons.
#[test]
#[ignore = "requires spin-polarised fixture loading — enable after .check/.castep_bin parsing is verified for nspins=2"]
fn sanity_charge_neutrality() {
    // TODO: verify that the sum of occupancies (from .check eigenvalues or
    // .bands) equals 64.00.  This is a basic sanity that the reference
    // data is internally consistent.
    //
    // SUM occ = 64.00 (from NiO.castep:131 "number of electrons : 64.00")
    todo!("sanity: verify total charge = 64.00 electrons");
}

/// Sanity check: ρ_spin(r) is not identically zero.
///
/// For NiO (spin-polarised), the spin density ρ_up - ρ_down must have
/// non-zero values.  An all-zero spin density indicates a bug in the
/// density handling (e.g., both spin channels collapsed to the same value).
#[test]
#[ignore = "requires spin-polarised density integration"]
fn sanity_spin_density_nonzero() {
    // TODO: compute max|ρ_spin(r)| over the fine grid from the loaded
    // fixture.  Assert max|ρ_spin| > 1e-10 (any non-trivial value).
    //
    // The reference gives 2*∫ρ_spin = -0.0619641, so the pointwise
    // spin density is certainly non-zero.
    todo!("sanity: verify spin density is not identically zero");
}
