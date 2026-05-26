//! Capture SCF iteration state as a CASTEP-compatible binary representation.
//!
//! This module implements the iter-2 SCF cascade discriminator: writes
//! chemrust's SCF state as a CASTEP `.check` file, then lets CASTEP
//! continue the SCF from that checkpoint.
//!
//! If CASTEP also cascades → iter-2 wavefunctions are corrupted.
//! If CASTEP converges → bug is in chemrust's V_eff/D_screened assembly.
//!
//! Feature-gated behind `#[cfg(any(test, feature = "scf_diag"))]`.

#![cfg(any(test, feature = "scf_diag"))]

use chemrust_hamiltonian_core::fft::{upsample_density_to_fine_grid, RealGrid};
use chemrust_hamiltonian_core::types::{
    BandEigenvalues, KPointData, KptWaveBlock, SpinChannel, WavefunctionCoeffs,
};
use chemrust_hamiltonian_core::{
    CastepBin, CellGeometry, Density as CoreDensity, ElectronDensity, FieldMetadata,
    SpinPolicy, Version,
};
use num_complex::Complex64;

use crate::density::compute_occupations;
use crate::scf::{Initialized, ScfIteration};
use crate::MixingOff;

/// Capture the current SCF state as a CASTEP-compatible binary
/// representation suitable for writing as a `.check` file.
///
/// Returns `Some(CastepBin)` with all fields populated from the SCF state,
/// or `None` if a required field is unavailable (e.g. wavefunctions not yet
/// computed, or density augmentation not available).
pub fn capture_as_castep_bin<S: SpinPolicy>(
    state: &ScfIteration<S, Initialized, MixingOff>,
    n_electrons: f64,
) -> Option<CastepBin> {
    let nspins = S::nspins();
    let n_bands = state.psi.n_bands;
    let n_pw = state.psi.n_pw;

    // --- Version and metadata ---
    let version = Version { major: 6, minor: 110 };
    let wave_grid_dims = {
        let [ngz, ngy, ngx] = state.wave_grid.grid();
        [ngx, ngy, ngz]
    };
    let field_meta = FieldMetadata {
        version,
        grid: wave_grid_dims,
        nspins,
        max_number_of_bands: n_bands,
    };

    // --- Cell geometry (clone directly) ---
    let cell = state.cell.clone();

    // --- Default orig_cell (no cell optimization in SCF) ---
    let orig_cell = CellGeometry {
        real_lattice: chemrust_hamiltonian_core::RealLattice::from_inner([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        recip_lattice: chemrust_hamiltonian_core::RecipLattice::from_inner([[0.0; 3]; 3]),
        volume: 1.0,
        num_species: 0,
        num_ions: 0,
        ionic_positions: ndarray::Array2::from_shape_vec((0, 3), vec![])
            .expect("invariant: (0,3) shape succeeds"),
        species_symbols: Vec::new(),
        species_pot_files: Vec::new(),
        num_ions_in_species: Vec::new(),
        ion_species: Vec::new(),
        max_ions_in_species: 0,
        species_lcao_states: vec![],
    };

    // --- Total energy ---
    let total_energy = state.total_energy.unwrap_or(0.0);

    // --- Occupations (recomputed — not stored on ScfIteration) ---
    let (occupations, _chem_pot) = compute_occupations(&state.eigenvalues, &state.smearing, n_electrons)
        .ok()?;

    // --- Band eigenvalues ---
    let eigenvalues = BandEigenvalues {
        kpoints: vec![KPointData {
            coords: state.k_point.coords,
            spins: vec![SpinChannel {
                eigenvalues: state.eigenvalues.clone(),
                occupancies: occupations.0,
            }],
            kpoint_weight: 1.0,
        }],
        nbands_max: n_bands,
        nspins,
        fermi_energy: state.fermi_energy.unwrap_or(0.0),
    };

    // --- Fine grid dimensions ---
    let fine_grid_dims = {
        let [ngz, ngy, ngx] = state.fine_grid.grid();
        [ngx, ngy, ngz]
    };

    // --- Total density on fine grid ---
    // 1. Upsample smooth density from wave grid to fine grid
    let rho_wave = state.density.as_wave_array();
    let rho_wave_padded = {
        let mut arr = ndarray::Array3::<f64>::zeros(wave_grid_dims);
        let shape = rho_wave.shape();
        for ix in 0..shape[0].min(wave_grid_dims[0]) {
            for iy in 0..shape[1].min(wave_grid_dims[1]) {
                for iz in 0..shape[2].min(wave_grid_dims[2]) {
                    arr[[ix, iy, iz]] = rho_wave[[ix, iy, iz]];
                }
            }
        }
        arr
    };
    let rho_wave_real = RealGrid::from_inner(rho_wave_padded);
    let rho_fine = upsample_density_to_fine_grid(
        &rho_wave_real,
        &state.wave_grid,
        &state.fine_grid,
    )
    .ok()?;

    // 2. Add augmentation density (if present)
    let rho_total = match &state.density_aug_fine {
        Some(aug) => {
            let summed = rho_fine.as_real_array().to_owned() + aug.as_real_array();
            summed
        }
        None => rho_fine.as_real_array().to_owned(),
    };

    // --- Wavefunction coefficients ---
    // psi.data is band-major: `data[b * n_pw + g]` is coefficient g of band b.
    let bands: Vec<Vec<Complex64>> = (0..n_bands)
        .map(|b| {
            let start = b * n_pw;
            state.psi.data[start..start + n_pw].to_vec()
        })
        .collect();

    let have_gamma = state.k_point.coords.iter().all(|&c| c.abs() < 1e-12);

    let wavefunction = WavefunctionCoeffs {
        have_gamma,
        grid: wave_grid_dims,
        nspins,
        kpt_data: vec![KptWaveBlock {
            coords: state.k_point.coords,
            nplw: n_pw,
            pw_grid_coord: state.pw_coords.clone(),
            bands,
        }],
    };

    // --- Electron density ---
    let density = ElectronDensity {
        nspins,
        grid: fine_grid_dims,
        charge: CoreDensity::from_inner(RealGrid::from_inner(rho_total)),
        spin: None,
    };

    Some(CastepBin {
        version,
        field_meta,
        cell,
        orig_cell,
        total_energy,
        eigenvalues,
        density,
        parameters_raw: vec![],
        cell_raw: vec![],
        orig_cell_raw: vec![],
        kpoint_weights: vec![1.0], // Single Gamma-point with weight 1.0
        fine_grid: Some(fine_grid_dims),
        wavefunction: Some(wavefunction),
        forces: None,
        stress: None,
        strain: None,
        hubbard_u: None,
        ldauoccm: None,
    })
}
