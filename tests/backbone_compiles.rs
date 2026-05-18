use ndarray::Array3;
use num_complex::Complex64;

use chemrust_hamiltonian_core::{
    CellGeometry, GVectorGrid, PseudopotentialSet, RealLattice, RecipLattice,
};
use chemrust_scf::{
    Density, KPoint, ScfIteration, SmearingParams, WaveGridArray, WavefunctionSet,
    ColumnDistributed, run_scf,
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
    }
}

/// A 4³ wavefunction FFT grid (Fortran layout [ngz, ngy, ngx]).
fn dummy_wave_grid() -> GVectorGrid {
    GVectorGrid::new(
        [4, 4, 4],
        RecipLattice::from_inner([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]),
    )
}

/// An 8³ fine FFT grid (2× upsampled).
fn dummy_fine_grid() -> GVectorGrid {
    GVectorGrid::new(
        [8, 8, 8],
        RecipLattice::from_inner([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]),
    )
}

/// Zero-initialized density on the 4³ wave grid.
fn dummy_density() -> Density {
    Density::from_inner(WaveGridArray::from_inner(Array3::<f64>::zeros((4, 4, 4))))
}

/// Minimal wavefunction set: 4 bands, 27 plane waves (3³).
fn dummy_wavefunctions() -> WavefunctionSet<ColumnDistributed> {
    WavefunctionSet::new(vec![Complex64::ZERO; 4 * 27], 4, 27)
}

#[test]
#[should_panic(expected = "not yet implemented")]
fn backbone_compiles() {
    let state: ScfIteration = ScfIteration::builder()
        .cell(dummy_cell())
        .pots(PseudopotentialSet::new())
        .wave_grid(dummy_wave_grid())
        .fine_grid(dummy_fine_grid())
        .density(dummy_density())
        .psi(dummy_wavefunctions())
        .k_point(KPoint::default())
        .smearing(SmearingParams {
            width: 0.01,
            electron_temperature: 0.01,
        })
        .max_history(4)
        .build();

    // The first todo!() inside build_v_eff() should fire here.
    let _ = run_scf(state, 4, 1e-8);
}
