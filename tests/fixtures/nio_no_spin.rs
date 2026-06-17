use std::sync::OnceLock;

use chemrust_hamiltonian_core::{
    CastepBin, CastepBinFile, CheckFile, GVectorGrid, NonSpin, PseudopotentialSet,
};
use chemrust_scf::{
    ColumnDistributed, Density, KPoint, ScfIteration, SmearingParams,
    SmearingScheme, SmearingWidth, WaveGridArray, WavefunctionSet,
    pw_coords_to_fft_indices,
};
use chemrust_scf::spin_types::{KptDataSet, PerSpinDensity, PerSpinPwCoefficients, SpinChannelData};
use chemrust_scf::PwCoefficients;
use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Path to CASTEP reference output directory (non-spin-polarised NiO,
/// CPU-only clean reference run).
pub const FIXTURE_DIR: &str =
    "/export/public_castep_jobs/tony/NiO_no_u_finer_grid_no_spin_cpu_reference";

/// Path to pseudopotential directory.
pub const POTENTIAL_DIR: &str = "/export/Potentials";

/// CASTEP reference total energy in eV (NiO.castep, final run).
pub const REFERENCE_ENERGY_EV: f64 = -7160.229765582;

/// Q2 gate tolerance (eV).
pub const TOLERANCE_EV: f64 = 2e-4;

// ---------------------------------------------------------------------------
// Cached fixture
// ---------------------------------------------------------------------------

pub struct NiONoSpinFixture {
    pub bin: CastepBin,
    pub check: CastepBin,
    pub bands_eigenvalues: Vec<f64>,
    pub pots: PseudopotentialSet,
}

static FIXTURE: OnceLock<NiONoSpinFixture> = OnceLock::new();

pub fn fixture() -> &'static NiONoSpinFixture {
    FIXTURE.get_or_init(|| load_fixture().expect("failed to load NiO non-spin fixture"))
}

fn load_fixture() -> Result<NiONoSpinFixture, Box<dyn std::error::Error>> {
    let fixture_dir =
        std::env::var("CASTEP_FIXTURE_DIR").unwrap_or_else(|_| FIXTURE_DIR.to_string());
    let potential_dir =
        std::env::var("CASTEP_POTENTIAL_DIR").unwrap_or_else(|_| POTENTIAL_DIR.to_string());

    // 1. Load .castep_bin (cell, density, eigenvalues)
    let bin_path = format!("{fixture_dir}/NiO.castep_bin");
    let bin_file = std::fs::File::open(&bin_path)?;
    let bin_reader = std::io::BufReader::new(bin_file);
    let bin = CastepBinFile::read(bin_reader)?;

    // 2. Load .check (wavefunction + fine_grid)
    let check_path = format!("{fixture_dir}/NiO.check");
    let check_file = std::fs::File::open(&check_path)?;
    let check_reader = std::io::BufReader::new(check_file);
    let check = CheckFile::read(check_reader)?;

    // 3. Load .bands (eigenvalues in Hartree) — first spin component only
    let bands_path = format!("{fixture_dir}/NiO.bands");
    let bands_text = std::fs::read_to_string(&bands_path)?;
    let bands_eigenvalues = parse_bands_first_spin(&bands_text)?;

    // 4. Load pseudopotentials
    let pots = PseudopotentialSet::from_dir(
        potential_dir,
        &bin.cell.species_symbols,
        &bin.cell.species_pot_files,
    )?;

    Ok(NiONoSpinFixture {
        bin,
        check,
        bands_eigenvalues,
        pots,
    })
}

// ---------------------------------------------------------------------------
// SCF state construction (analogous to cu111_co::build_scf_state)
// ---------------------------------------------------------------------------

/// Build an `ScfIteration` from the NiO non-spin fixture using the first
/// k-point.
pub fn build_scf_state(
    fx: &NiONoSpinFixture,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> ScfIteration {
    let cell = fx.bin.cell.clone();
    let pots = fx.pots.clone();

    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let wave_grid_dims = wfc.grid;
    let [ngx, ngy, ngz] = wave_grid_dims;

    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);

    let fine_grid_dims = fx.check.fine_grid.expect(".check must have fine_grid");
    let [fgx, fgy, fgz] = fine_grid_dims;
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    // Density from .castep_bin
    let density = Density::from_inner(WaveGridArray::from_inner(
        fx.bin.density.charge.as_real_grid().as_real_array().clone(),
    ));

    // First k-point wavefunctions
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;

    let flat_bands: Vec<Complex64> = kpt_block.bands.concat();
    let flat_bands_clone = flat_bands.clone();
    let psi = WavefunctionSet::<ColumnDistributed>::new(flat_bands, n_bands, n_pw);

    let pw_coords = KptDataSet::new(vec![kpt_block.pw_grid_coord.clone()], 1);
    let pw_fft_indices_data = pw_coords_to_fft_indices(&kpt_block.pw_grid_coord, &wave_grid);
    let pw_fft_indices = KptDataSet::new(vec![pw_fft_indices_data], 1);

    let k_point = KPoint {
        coords: kpt_block.coords,
        weight: 1.0,
    };

    let smearing = SmearingParams {
        width: SmearingWidth::ev(0.1),
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: SmearingScheme::Gaussian,
        spin_fix: 6,
    };

    let psi_cpu_data: SpinChannelData<KptDataSet<Vec<Complex64>>> =
        SpinChannelData::new::<NonSpin>(vec![KptDataSet::new(vec![flat_bands_clone], 1)]);

    let psi_gpu_placeholder = PwCoefficients::new(
        stream
            .alloc_zeros::<chemrust_scf::device::CudaComplex>(0)
            .expect("dummy psi alloc"),
    );
    let psi_gpu = PerSpinPwCoefficients::new(SpinChannelData::new::<NonSpin>(vec![
        KptDataSet::new(vec![psi_gpu_placeholder], 1),
    ]));

    let density_ps = PerSpinDensity::new(SpinChannelData::new::<NonSpin>(vec![density]));
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
// Bands file parsing (non-spin: single spin component)
// ---------------------------------------------------------------------------

fn parse_bands_first_spin(text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let eigenvalues: Vec<f64> = text
        .lines()
        .skip_while(|line| !line.trim().starts_with("Spin component"))
        .skip(1) // skip "Spin component N" header
        .take_while(|line| !line.trim().starts_with("Spin component"))
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect();
    if eigenvalues.is_empty() {
        return Err("no eigenvalues parsed from .bands file".into());
    }
    Ok(eigenvalues)
}
