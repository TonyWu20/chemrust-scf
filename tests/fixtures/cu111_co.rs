use std::sync::OnceLock;

use chemrust_hamiltonian_core::{
    formatted, CastepBin, CastepBinFile, CheckFile, ElectronDensity, GVectorGrid,
    PseudopotentialSet,
};
use chemrust_scf::{
    pw_coords_to_fft_indices, ColumnDistributed, Density, KPoint, ScfIteration, SmearingParams,
    SmearingScheme, WaveGridArray, WavefunctionSet,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Path to CASTEP reference output directory.
/// Override via `CASTEP_FIXTURE_DIR` environment variable.
pub const FIXTURE_DIR: &str = "/export/public_castep_jobs/tony/Cu111_CO_SinglePoint";

/// Path to pseudopotential directory.
/// Override via `CASTEP_POTENTIAL_DIR` environment variable.
pub const POTENTIAL_DIR: &str = "/export/Potentials";

/// CASTEP reference total energy in eV (Cu111_CO.castep line 326).
pub const REFERENCE_ENERGY_EV: f64 = -24110.96665069;

/// Validation tolerance for total energy (eV).
pub const TOLERANCE_EV: f64 = 2e-4;

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
    let fixture_dir = std::env::var("CASTEP_FIXTURE_DIR").unwrap_or_else(|_| FIXTURE_DIR.to_string());
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
pub fn build_scf_state(fx: &Cu111CoFixture) -> ScfIteration {
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
    let fine_grid_dims = fx
        .check
        .fine_grid
        .expect(".check must have fine_grid");
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
    let psi = WavefunctionSet::<ColumnDistributed>::new(flat_bands, n_bands, n_pw);

    let pw_coords = kpt_block.pw_grid_coord.clone();
    let pw_fft_indices = pw_coords_to_fft_indices(&pw_coords, &wave_grid);

    let k_point = KPoint {
        coords: kpt_block.coords,
    };

    // Smearing: Gaussian, 0.1 eV (CASTEP default, Cu111_CO.castep line 154)
    let smearing = SmearingParams {
        width: 0.1 * chemrust_scf::EV_TO_HARTREE,
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: SmearingScheme::Gaussian,
    };

    ScfIteration::builder()
        .cell(cell)
        .pots(pots)
        .wave_grid(wave_grid)
        .fine_grid(fine_grid)
        .density(density)
        .psi(psi)
        .pw_coords(pw_coords)
        .pw_fft_indices(pw_fft_indices)
        .k_point(k_point)
        .smearing(smearing)
        .max_history(8)
        .build()
}
