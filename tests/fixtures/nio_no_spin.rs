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
/// k-point with the CASTEP parameter-set defaults (Pulay mixing,
/// Gaussian smearing 0.1 eV).
pub fn build_scf_state(
    fx: &NiONoSpinFixture,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> ScfIteration {
    build_scf_state_with_scheme(fx, stream, chemrust_scf::MixingScheme::Pulay)
}

/// Build the SCF state with an explicit density-mixing scheme.
///
/// `scheme = MixingScheme::Kerker` pins the loop to Kerker mixing
/// (no DIIS/Pulay); `Pulay` follows the CASTEP parameter-set default
/// (Kerker first, then Pulay). Used to A/B-isolate the DIIS/Pulay
/// mixer in the full loop.
pub fn build_scf_state_with_scheme(
    fx: &NiONoSpinFixture,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    scheme: chemrust_scf::MixingScheme,
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

    // Density from .castep_bin. The NiO .bin stores the charge on the
    // FINE grid. `ScfIteration::new` expects a wave-grid density and
    // upsamples it itself, so downsample fine → wave first.
    // Band-limited truncation (FFT to G-space, keep wave-grid G-vectors,
    // inverse FFT) matches CASTEP's fine → normal grid density transfer.
    let charge_fine =
        fx.bin.density.charge.as_real_grid().as_real_array();
    let charge_wave = chemrust_scf::downsample_array_to_wave_grid(
        charge_fine,
        &fine_grid,
        &wave_grid,
    )
    .expect("downsample fine-grid charge to wave grid");
    let density = Density::from_inner(WaveGridArray::from_inner(
        charge_wave.into_inner().as_array().clone(),
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

    // CASTEP NiO input: mixing_scheme = PULAY (history 20, amplitude 0.5,
    // G-cutoff 1.5 1/A), Gaussian smearing 0.1 eV. Non-spin run: CASTEP
    // gates the spin_fix transition on spin_polarised (electronic.f90:315,
    // 621), so it never applies here — spin_fix = -1 disables it.
    let smearing = SmearingParams {
        width: SmearingWidth::ev(0.1),
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: SmearingScheme::Gaussian,
        spin_fix: -1,
        mixing_scheme: scheme,
        net_spin: 0.0,
    };

    // CASTEP .param: mix_charge_gmax = 1.5 1/ang (a₀⁻¹) — Kerker kernel
    // scale. 1 /Å = 1.88972612545 a₀⁻¹.
    let mix_gmax = 1.5 * 1.88972612545;

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
        .mix_gmax(mix_gmax)
        .build()
}

/// Build a full 14-k-point SCF state from the NiO non-spin fixture.
///
/// The fixture `.check` file stores wavefunctions for all 14 k-points
/// (the `build_scf_state` single-k-point builder is a reduced system and
/// must not be used for SCF-loop diagnostics). k-point weights come
/// from `fx.bin.kpoint_weights`.
pub fn build_scf_state_all_kpts(
    fx: &NiONoSpinFixture,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> ScfIteration {
    build_scf_state_all_kpts_with_scheme(fx, stream, chemrust_scf::MixingScheme::Pulay)
}

/// Full 14-k-point SCF state with an explicit density-mixing scheme.
pub fn build_scf_state_all_kpts_with_scheme(
    fx: &NiONoSpinFixture,
    _stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    scheme: chemrust_scf::MixingScheme,
) -> ScfIteration {
    use chemrust_scf::spin_types::KptDataSet;

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

    // CASTEP total density (all k-points) on the fine grid → wave grid.
    let charge_fine =
        fx.bin.density.charge.as_real_grid().as_real_array();
    let charge_wave = chemrust_scf::downsample_array_to_wave_grid(
        charge_fine,
        &fine_grid,
        &wave_grid,
    )
    .expect("downsample fine-grid charge to wave grid");
    let density = Density::from_inner(WaveGridArray::from_inner(
        charge_wave.into_inner().as_array().clone(),
    ));

    let nkpts = wfc.kpt_data.len();
    let kpt_weights = fx.bin.kpoint_weights.clone();
    assert_eq!(kpt_weights.len(), nkpts, "k-point weight count mismatch");

    let n_bands = wfc.kpt_data[0].bands.len();

    let ctx = std::sync::Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA GPU"));
    let stream_gpu = ctx.default_stream();

    let mut psi_gpu: Vec<PwCoefficients> = Vec::with_capacity(nkpts);
    let mut psi_data: Vec<Vec<num_complex::Complex64>> = Vec::with_capacity(nkpts);
    let mut k_points_vec: Vec<KPoint> = Vec::with_capacity(nkpts);
    let mut pw_coords_vec: Vec<Vec<[i32; 3]>> = Vec::with_capacity(nkpts);
    let mut pw_fft_vec: Vec<Vec<i32>> = Vec::with_capacity(nkpts);

    for ikpt in 0..nkpts {
        let block = &wfc.kpt_data[ikpt];
        assert_eq!(block.bands.len(), n_bands, "band count mismatch at kpt {ikpt}");
        let n_pw_kpt = block.nplw;
        let n_el = n_bands * n_pw_kpt;
        let slice: cudarc::driver::CudaSlice<chemrust_scf::device::CudaComplex> =
            stream_gpu.alloc_zeros(n_el).expect("GPU kpt bands");
        psi_gpu.push(PwCoefficients::new(slice));
        psi_data.push(block.bands.iter().flatten().copied().collect());
        k_points_vec.push(KPoint { coords: block.coords, weight: kpt_weights[ikpt] });
        pw_coords_vec.push(block.pw_grid_coord.clone());
        pw_fft_vec.push(chemrust_scf::pw_coords_to_fft_indices(
            &block.pw_grid_coord,
            &wave_grid,
        ));
    }

    let psi = PerSpinPwCoefficients::new(SpinChannelData::new::<NonSpin>(vec![
        KptDataSet::new(psi_gpu, nkpts),
    ]));
    let psi_cpu = SpinChannelData::new::<NonSpin>(vec![KptDataSet::new(psi_data, nkpts)]);
    let density_ps = PerSpinDensity::new(SpinChannelData::new::<NonSpin>(vec![density]));
    let k_points_ps = KptDataSet::new(k_points_vec, nkpts);

    let smearing = SmearingParams {
        width: SmearingWidth::ev(0.1),
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: SmearingScheme::Gaussian,
        spin_fix: -1,
        mixing_scheme: scheme,
        net_spin: 0.0,
    };

    // CASTEP .param: mix_charge_gmax = 1.5 1/ang; 1 /Å = 1.88972612545 a₀⁻¹.
    let mix_gmax = 1.5 * 1.88972612545;

    ScfIteration::builder()
        .cell(cell)
        .pots(pots)
        .wave_grid(wave_grid)
        .fine_grid(fine_grid)
        .density(density_ps)
        .psi(psi)
        .psi_data(psi_cpu)
        .pw_coords(KptDataSet::new(pw_coords_vec, nkpts))
        .pw_fft_indices(KptDataSet::new(pw_fft_vec, nkpts))
        .k_points(k_points_ps)
        .smearing(smearing)
        .max_history(8)
        .mix_gmax(mix_gmax)
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
