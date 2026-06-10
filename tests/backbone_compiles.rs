use std::sync::Arc;

use cudarc::driver::CudaContext;
use ndarray::Array3;
use num_complex::Complex64;

use chemrust_hamiltonian_core::{
    CellGeometry, GVectorGrid, NonSpin, PseudopotentialSet, RealLattice, RecipLattice,
};
use chemrust_scf::{
    Density, KPoint, KptDataSet, PerSpinDensity, PerSpinPwCoefficients, PwCoefficients,
    ScfIteration, SmearingParams, SmearingScheme, SpinChannelData, WaveGridArray,
    WavefunctionSet, ColumnDistributed, device::CudaComplex, run_scf,
};

/// A minimal 1-atom cubic cell at (0,0,0).
fn dummy_cell() -> CellGeometry {
    use ndarray::Array2;
    CellGeometry {
        real_lattice: RealLattice::from_inner([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]),
        recip_lattice: RecipLattice::from_inner([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]),
        volume: 125.0,
        num_species: 1,
        num_ions: 1,
        ionic_positions: Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap(),
        species_symbols: vec!["Cu".into()],
        species_pot_files: vec!["Cu_00.usp".into()],
        num_ions_in_species: vec![1],
        ion_species: vec![0],
        max_ions_in_species: 1,
        species_lcao_states: vec![],
    }
}

/// A 4×4×6 wavefunction FFT grid (non-cubic, Fortran layout [ngz, ngy, ngx]).
fn dummy_wave_grid() -> GVectorGrid {
    GVectorGrid::new(
        4, 4, 6,
        RecipLattice::from_inner([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]),
    )
}

/// An 8×8×12 fine FFT grid (2× upsampled).
fn dummy_fine_grid() -> GVectorGrid {
    GVectorGrid::new(
        8, 8, 12,
        RecipLattice::from_inner([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]),
    )
}

/// Zero-initialized density on the 6×4×4 wave grid.
fn dummy_density() -> Density {
    Density::from_inner(WaveGridArray::from_inner(Array3::<f64>::zeros((6, 4, 4))))
}

/// Minimal wavefunction set: 4 bands, 27 plane waves (3³).
fn dummy_wavefunctions() -> WavefunctionSet<ColumnDistributed> {
    WavefunctionSet::new(vec![Complex64::ZERO; 4 * 27], 4, 27)
}

fn dummy_per_spin_density() -> PerSpinDensity {
    PerSpinDensity(SpinChannelData::new::<NonSpin>(vec![dummy_density()]))
}

fn dummy_per_spin_psi() -> (PerSpinPwCoefficients, SpinChannelData<KptDataSet<Vec<Complex64>>>) {
    let wfn = dummy_wavefunctions();
    let pw = PwCoefficients::new(
        Arc::new(CudaContext::new(0).unwrap()).default_stream()
            .alloc_zeros::<CudaComplex>(wfn.data.len()).unwrap()
    );
    let psi_gpu = PerSpinPwCoefficients(SpinChannelData::new::<NonSpin>(
        vec![KptDataSet::new(vec![pw], 1)],
    ));
    let psi_data = SpinChannelData::new::<NonSpin>(vec![
        KptDataSet::new(vec![wfn.data], 1),
    ]);
    (psi_gpu, psi_data)
}

#[test]
#[ignore = "requires GPU; full validation is Group F"]
fn backbone_compiles() {
    let (psi, psi_data) = dummy_per_spin_psi();
    let pw_coords: Vec<[i32; 3]> = (0..27)
        .map(|i| {
            let iz = i / 9 - 1;
            let iy = (i / 3) % 3 - 1;
            let ix = i % 3 - 1;
            [ix, iy, iz]
        })
        .collect();
    let pw_fft_indices: Vec<i32> = {
        let ngz = 6; let ngy = 4;
        (0..27)
            .map(|i| {
                let ix = i % 3;
                let iy = (i / 3) % 3;
                let iz = i / 9;
                (iz + ngz * (iy + ngy * ix)) as i32
            })
            .collect()
    };
    let state: ScfIteration = ScfIteration::builder()
        .cell(dummy_cell())
        .pots(PseudopotentialSet::new())
        .wave_grid(dummy_wave_grid())
        .fine_grid(dummy_fine_grid())
        .density(dummy_per_spin_density())
        .psi(psi)
        .psi_data(psi_data)
        .pw_coords(KptDataSet::new(vec![pw_coords], 1))
        .pw_fft_indices(KptDataSet::new(vec![pw_fft_indices], 1))
        .k_points(KptDataSet::new(vec![KPoint::default()], 1))
        .smearing(SmearingParams {
            width: 0.01,
            electron_temperature: 0.01,
            scheme: SmearingScheme::Gaussian,
        })
        .max_history(4)
        .build();

    // build_v_eff() is wired to VEffBuilder, then GPU density follows.
    // Ignored: requires GPU (Group F handles full validation).
    let _ = run_scf(state, 4, 1e-8);
}
