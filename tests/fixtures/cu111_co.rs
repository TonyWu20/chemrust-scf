use std::collections::HashMap;
use std::sync::OnceLock;

use chemrust_hamiltonian_core::{
    CastepBin, CastepBinFile, CheckFile, EffectivePotential, ElectronDensity, GVectorGrid,
    NonSpin, PseudopotentialSet, formatted,
};
use chemrust_scf::{
    ColumnDistributed, Density, EffectivePotential as ScfEffectivePotential,
    FineGridArray, KPoint, MixingOff, ScfIteration, SmearingParams, SmearingScheme,
    SmearingWidth, VEffBuilt,
    WaveGridArray, WavefunctionSet, pw_coords_to_fft_indices,
};
use chemrust_scf::spin_types::{KptDataSet, PerSpinDensity, PerSpinPwCoefficients, SpinChannelData};
use chemrust_scf::PwCoefficients;
use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Path to CASTEP reference output directory.
/// Override via `CASTEP_FIXTURE_DIR` environment variable.
pub const FIXTURE_DIR: &str = "/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8";

/// Path to pseudopotential directory.
/// Override via `CASTEP_POTENTIAL_DIR` environment variable.
pub const POTENTIAL_DIR: &str = "/export/Potentials";

/// CASTEP reference total energy in eV (Cu111_CO.castep line 326).
pub const REFERENCE_ENERGY_EV: f64 = -24110.96665069;

/// Legacy fixed-point tolerance (eV), used by the deprecated
/// `fixed_point_matches_castep_energy` test. Empirically chosen before SCF
/// behavior was characterized; it conflates algorithm-fidelity (does CASTEP's
/// ψ stay fixed under our operator?) with convergence (does our SCF reach
/// 1e-5 eV?). See `notes/failure-patterns.md` § tolerance-conflation.
pub const TOLERANCE_EV: f64 = 2e-4;

/// Q1 (algorithm-fidelity probe) tolerance: per-iteration energy drift after
/// **one** SCF iteration starting from CASTEP's converged state. Calibrated
/// at 2× the empirical iter-1 noise floor (~9.8 mHa from T3, REVIEW_PROMPT.md
/// `cascade_with_castep_veff_substitution`) per ODD discriminator rule.
/// Acts as a regression bar; ratchet down as eigensolver rotation is reduced.
pub const DRIFT_TOLERANCE_HA: f64 = 2e-2;

/// Q2 (drop-in fidelity ship-gate) tolerance: total-energy convergence to
/// CASTEP's `ELEC_ENERGY_TOL` (`Cu111_CO.param:39`). This is the actual
/// drop-in-replacement quality target: the chemrust-scf SCF, used as a
/// CASTEP backend via C bindings, must deliver this tolerance for downstream
/// CASTEP code (forces, stress, properties).
pub const CASTEP_TOLERANCE_EV: f64 = 1e-5;

// ---------------------------------------------------------------------------
// Cached fixture
// ---------------------------------------------------------------------------

/// All pre-loaded CASTEP reference data for the Cu111_CO system.
///
/// Loaded once (lazily) and cached for the lifetime of the test process.
pub struct Cu111CoFixture {
    pub bin: CastepBin,
    pub check: CastepBin,
    pub pot_fmt: ndarray::Array3<f64>,
    pub den_fmt: ElectronDensity,
    pub bands_eigenvalues: Vec<f64>,
    pub pots: PseudopotentialSet,
}

static FIXTURE: OnceLock<Cu111CoFixture> = OnceLock::new();

/// Access the cached Cu111_CO fixture, loading it on first call.
pub fn fixture() -> &'static Cu111CoFixture {
    FIXTURE.get_or_init(|| load_fixture().expect("failed to load Cu111_CO fixture"))
}

fn load_fixture() -> Result<Cu111CoFixture, Box<dyn std::error::Error>> {
    let fixture_dir =
        std::env::var("CASTEP_FIXTURE_DIR").unwrap_or_else(|_| FIXTURE_DIR.to_string());
    let potential_dir =
        std::env::var("CASTEP_POTENTIAL_DIR").unwrap_or_else(|_| POTENTIAL_DIR.to_string());

    // 1. Load .castep_bin (cell, density on wave grid, eigenvalues)
    let bin_path = format!("{fixture_dir}/Cu111_CO.castep_bin");
    let bin_file = std::fs::File::open(&bin_path)?;
    let bin_reader = std::io::BufReader::new(bin_file);
    let bin = CastepBinFile::read(bin_reader)?;

    // 2. Load .check (wavefunction + fine_grid, 155 MB — cached once)
    let check_path = format!("{fixture_dir}/Cu111_CO.check");
    let check_file = std::fs::File::open(&check_path)?;
    let check_reader = std::io::BufReader::new(check_file);
    let check = CheckFile::read(check_reader)?;

    // 3. Load .pot_fmt (reference V_eff on fine grid)
    let pot_path = format!("{fixture_dir}/Cu111_CO.pot_fmt");
    let pot_text = std::fs::read_to_string(&pot_path)?;
    let (_pot_grid, pot_arr) = formatted::parse_pot_fmt(&pot_text)?;

    // 4. Load .den_fmt (reference density on fine grid)
    let den_path = format!("{fixture_dir}/Cu111_CO.den_fmt");
    let den_text = std::fs::read_to_string(&den_path)?;
    let den_fmt = formatted::parse_den_fmt(&den_text)?;

    // 5. Load .bands (eigenvalues in Hartree)
    let bands_path = format!("{fixture_dir}/Cu111_CO.bands");
    let bands_text = std::fs::read_to_string(&bands_path)?;
    let bands_eigenvalues = parse_bands_file(&bands_text)?;

    // 6. Load pseudopotentials for all species using exact filenames from .castep_bin
    let pots = PseudopotentialSet::from_dir(
        potential_dir,
        &bin.cell.species_symbols,
        &bin.cell.species_pot_files,
    )?;

    Ok(Cu111CoFixture {
        bin,
        check,
        pot_fmt: pot_arr,
        den_fmt,
        bands_eigenvalues,
        pots,
    })
}

/// Parse eigenvalues from a CASTEP `.bands` file.
///
/// Format (Gamma-point, non-spin-polarised):
///   Number of k-points     1
///   Number of spin components 1
///   Number of electrons  186.0
///   Number of eigenvalues    160
///   Fermi energy (in atomic units)    -0.122443
///   Unit cell vectors
///      ...
///   K-point     1 -0.25000000  0.00000000  0.00000000  1.00000000
///   Spin component 1
///      -1.05502287
///      -0.49721145
///      ...
fn parse_bands_file(text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let eigenvalues: Vec<f64> = text
        .lines()
        .skip_while(|line| !line.trim().starts_with("Spin component"))
        .skip(1) // skip "Spin component N" header
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect();
    if eigenvalues.is_empty() {
        return Err("no eigenvalues parsed from .bands file".into());
    }
    Ok(eigenvalues)
}

// ---------------------------------------------------------------------------
// SCF state construction from fixture data
// ---------------------------------------------------------------------------

/// Build a fully-initialised `ScfIteration` from the Cu111_CO fixture.
///
/// Loads density from `.castep_bin` (wave grid) and wavefunctions from `.check`
/// (gamma-point only — Cu111_CO uses a single k-point).
pub fn build_scf_state(
    fx: &Cu111CoFixture,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> ScfIteration {
    let cell = fx.bin.cell.clone();
    let pots = fx.pots.clone();

    // Wavefunction grid from .check (field_meta.grid is [0,0,0] in .castep_bin)
    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let wave_grid_dims = wfc.grid; // CASTEP [ngx, ngy, ngz]
    let [ngx, ngy, ngz] = wave_grid_dims;

    // GVectorGrid expects positional (ngx, ngy, ngz)
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);

    // Fine grid from .check file
    let fine_grid_dims = fx.check.fine_grid.expect(".check must have fine_grid");
    let [fgx, fgy, fgz] = fine_grid_dims;
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    // Verify density grid matches wavefunction grid
    let den_grid = fx.bin.density.grid; // [ngx, ngy, ngz]
    assert_eq!(
        den_grid, wave_grid_dims,
        "density grid {:?} does not match wavefunction grid {:?}",
        den_grid, wave_grid_dims,
    );

    // Density from .castep_bin — try raw values (no volume normalization)
    let density = Density::from_inner(WaveGridArray::from_inner(
        fx.bin.density.charge.as_real_grid().as_real_array().clone(),
    ));

    // Wavefunctions from .check (first k-point, first spin)
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;

    let flat_bands: Vec<num_complex::Complex64> = kpt_block.bands.concat();
    let flat_bands_clone = flat_bands.clone();
    let psi = WavefunctionSet::<ColumnDistributed>::new(flat_bands, n_bands, n_pw);

    let pw_coords = KptDataSet::new(vec![kpt_block.pw_grid_coord.clone()], 1);
    let pw_fft_indices_data = pw_coords_to_fft_indices(&kpt_block.pw_grid_coord, &wave_grid);
    let pw_fft_indices = KptDataSet::new(vec![pw_fft_indices_data], 1);

    let k_point = KPoint {
        coords: kpt_block.coords,
        weight: 1.0,
    };

    // Smearing: Gaussian, 0.1 eV (CASTEP default, Cu111_CO.castep line 154)
    let smearing = SmearingParams {
        width: chemrust_scf::SmearingWidth::ev(0.1),
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: SmearingScheme::Gaussian,
        spin_fix: 10,
        mixing_scheme: chemrust_scf::MixingScheme::Off,
        net_spin: 0.0,
    };

    // CPU-side psi: SpinChannelData<KptDataSet<Vec<Complex64>>>
    let psi_cpu_data: SpinChannelData<KptDataSet<Vec<Complex64>>> =
        SpinChannelData::new::<NonSpin>(vec![
            KptDataSet::new(vec![flat_bands_clone], 1),
        ]);
    // GPU-side psi: placeholder (diagonalize_inner uploads from psi_cpu).
    let psi_gpu_placeholder = PwCoefficients::new(
        stream.alloc_zeros::<chemrust_scf::device::CudaComplex>(0)
            .expect("dummy psi alloc"),
    );
    let psi_gpu = PerSpinPwCoefficients::new(
        SpinChannelData::new::<NonSpin>(vec![
            KptDataSet::new(vec![psi_gpu_placeholder], 1),
        ]),
    );
    let density_ps = PerSpinDensity::new(
        SpinChannelData::new::<NonSpin>(vec![density]),
    );
    let k_points_ps = KptDataSet::new(vec![k_point], 1);

    ScfIteration::builder()
        .cell(cell)
        .pots(pots)
        .wave_grid(wave_grid)
        .fine_grid(fine_grid)
        .density(density_ps)
        .psi(psi_gpu)
        .psi_data(psi_cpu_data)
        .pw_coords(pw_coords)
        .pw_fft_indices(pw_fft_indices)
        .k_points(k_points_ps)
        .smearing(smearing)
        .max_history(8)
        .build()
}

// ---------------------------------------------------------------------------
// T0: D_band_debug.dat parser (D-screening EXTERNAL anchor)
// ---------------------------------------------------------------------------

/// Parse `D_band_debug.dat` and return the **converged** (last) D_screened
/// matrix per `(species_idx, ion_idx_in_species)`.
///
/// File format (CASTEP `nlpot.f90:531-544`, append mode):
/// ```text
///   nsp  num_ps_projectors(nsp)
///   dn  dm  nl_d(dm,dn,ni,nsp,ns)
///   ...
/// ```
/// where `nsp` is the 1-based species index, `num_ps_projectors` is the
/// number of β projectors for that species, and the following
/// `num_proj·(num_proj+1)/2` lines are upper-triangular (dn ≤ dm) elements.
///
/// The file accumulates over SCF iterations in append mode. Taking the
/// **last** block per `(species, ion)` yields the converged-iteration value.
///
/// `mixture_weight` (from VCA) is assumed 1.0 for Cu111+CO; the parser
/// divides by `mixture_weight` for each ion if a non-1.0 value is provided.
pub fn load_castep_d_screened(
    d_dump_path: &str,
    mixture_weight: f64,
) -> Result<HashMap<(usize, usize), ndarray::Array2<f64>>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(d_dump_path)?;
    let lines: Vec<&str> = text.lines().collect();

    let mut result: HashMap<(usize, usize), ndarray::Array2<f64>> = HashMap::new();
    let mut i = 0usize;

    while i < lines.len() {
        let header = lines[i].trim();
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 2 {
            i += 1;
            continue;
        }
        let species_idx: usize = parts[0].parse::<usize>()? - 1; // CASTEP 1-based → 0-based
        let num_proj: usize = parts[1].parse()?;
        let n_pairs = num_proj * (num_proj + 1) / 2;
        i += 1;

        // Read upper-triangular pairs
        let mut mat = ndarray::Array2::<f64>::zeros((num_proj, num_proj));
        for _ in 0..n_pairs {
            if i >= lines.len() {
                return Err("unexpected EOF in D_band_debug.dat data block".into());
            }
            let data_parts: Vec<&str> = lines[i].trim().split_whitespace().collect();
            if data_parts.len() != 3 {
                i += 1;
                continue;
            }
            let dn: usize = data_parts[0].parse::<usize>()? - 1; // 0-based
            let dm: usize = data_parts[1].parse::<usize>()? - 1;
            let raw: f64 = data_parts[2].parse()?;
            let val = raw / mixture_weight;
            mat[[dn, dm]] = val;
            mat[[dm, dn]] = val; // symmetric (nlpot.f90:526)
            i += 1;
        }

        // Track ion index within species: count how many blocks of this species
        // we've seen so far (before overwriting)
        let ion_in_species = result
            .keys()
            .filter(|(s, _)| *s == species_idx)
            .count()
            + 1;

        // Last-write-wins = converged iteration
        result.insert((species_idx, ion_in_species), mat);
    }

    Ok(result)
}

/// Convert the D-dump `(species_idx, ion_in_species)` key to a per-global-ion
/// `Vec`, using the cell's `ion_species` mapping to count ions per species.
pub fn d_screened_by_global_ion(
    d_map: &HashMap<(usize, usize), ndarray::Array2<f64>>,
    num_ions: usize,
    ion_species: &[usize],
) -> Vec<Option<ndarray::Array2<f64>>> {
    // Count how many ions of each species have been seen so far
    let mut species_counters = vec![0usize; ion_species.iter().max().copied().unwrap_or(0) + 1];
    let mut result: Vec<Option<ndarray::Array2<f64>>> = Vec::with_capacity(num_ions);

    for global_ion in 0..num_ions {
        let sp = ion_species[global_ion];
        species_counters[sp] += 1;
        let ion_in_sp = species_counters[sp];
        result.push(d_map.get(&(sp, ion_in_sp)).cloned());
    }

    result
}

/// Wrap the fixture's `.pot_fmt` array as an `EffectivePotential`.
pub fn castep_veff_as_effective(fx: &Cu111CoFixture) -> EffectivePotential {
    use chemrust_hamiltonian_core::fft::RealGrid;
    EffectivePotential::from_inner(RealGrid::from_inner(fx.pot_fmt.clone()))
}

/// Extract the converged wavefunction (first k-point) from the .check fixture
/// as a flat `Vec<Complex64>` in column-major (n_bands × n_pw) layout.
pub fn castep_psi_first_kpoint(fx: &Cu111CoFixture) -> Vec<num_complex::Complex64> {
    fx.check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction")
        .kpt_data[0]
        .bands
        .concat()
}

/// Number of plane-waves at the first k-point in the .check fixture.
pub fn n_pw_first_kpoint(fx: &Cu111CoFixture) -> usize {
    fx.check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction")
        .kpt_data[0]
        .nplw
}

/// Build an SCF state with CASTEP's V_eff pinned in place.
///
/// Constructs a VEffBuilt state via `build_scf_state` + `build_v_eff_with_energy`,
/// then replaces the assembled V_eff with the CASTEP reference from `.pot_fmt`.
/// The returned state has `MixingOff` — suitable for a single-shot diagonalize.
pub fn build_state_with_castep_veff(
    fx: &Cu111CoFixture,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> ScfIteration<NonSpin, VEffBuilt, MixingOff> {
    let mut state = build_scf_state(fx, stream)
        .build_v_eff_with_energy()
        .expect("build_v_eff_with_energy failed");

    // CASTEP V_eff from pot_fmt: already chemrust_hamiltonian_core::EffectivePotential,
    // which is exactly NonSpin::VEff — the type set_v_eff expects.
    let castep_veff = castep_veff_as_effective(fx);
    state.set_v_eff(castep_veff);
    state
}

/// Build an SCF state with CASTEP V_eff pinned AND custom psi injected.
///
/// Calls `build_state_with_castep_veff`, then overwrites the internal
/// wavefunction with the provided `psi_in`.
pub fn build_state_with_castep_veff_and_psi(
    fx: &Cu111CoFixture,
    psi_in: &[Complex64],
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> ScfIteration<NonSpin, VEffBuilt, MixingOff> {
    let mut state = build_state_with_castep_veff(fx, stream);
    state.psi_data_mut().copy_from_slice(psi_in);
    state
}
