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
    let n_bands = state.n_bands;
    let nkpts = state.nkpts;
    let n_pw = if nkpts > 0 { state.pw_coords[0].len() } else { 0 };

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
    // Use kpt-0 eigenvalues for occupation computation (single-kpt path).
    let (occupations, _chem_pot) = if nkpts > 0 {
        compute_occupations(&state.eigenvalues[0][0], &state.smearing, n_electrons, 1.0)
            .ok()?
    } else {
        return None;
    };

    // --- Band eigenvalues ---
    // Build per-kpt eigenvalue data from the nested structure
    let kpoints: Vec<KPointData> = (0..nkpts)
        .map(|ikpt| {
            let kpt_coords = state.k_points[ikpt].coords;
            let kpt_weight = state.k_points[ikpt].weight;
            let eigs = &state.eigenvalues[0][ikpt];
            let (occ, _) = compute_occupations(eigs, &state.smearing, n_electrons, 1.0).ok()
                .unwrap_or_else(|| {
                    let occ = vec![0.0; eigs.len()];
                    (crate::types::Occupations(occ), crate::types::ChemicalPotential(0.0))
                });
            KPointData {
                coords: kpt_coords,
                spins: vec![SpinChannel {
                    eigenvalues: eigs.clone(),
                    occupancies: occ.0,
                }],
                kpoint_weight: kpt_weight,
            }
        })
        .collect();
    let eigenvalues = BandEigenvalues {
        kpoints,
        nbands_max: n_bands,
        nspins,
        fermi_energy: state.fermi_energy[0],
    };

    // --- Fine grid dimensions ---
    let fine_grid_dims = {
        let [ngz, ngy, ngx] = state.fine_grid.grid();
        [ngx, ngy, ngz]
    };

    // --- Total density on fine grid ---
    // 1. Upsample smooth density from wave grid to fine grid
    let total_density = state.density.total();
    let rho_wave = total_density.as_wave_array();
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

    // 2. Add augmentation density (if present) — spin-0 channel
    let rho_total = match state.density_aug_fine[0].as_ref() {
        Some(aug) => {
            rho_fine.as_real_array().to_owned() + aug.as_real_array()
        }
        None => rho_fine.as_real_array().to_owned(),
    };

    // --- Wavefunction coefficients ---
    // psi_cpu is band-major: `data[b * n_pw + g]` is coefficient g of band b.
    // Use kpt-0 for capture (gamma-point path).
    let psi_kpt0 = &state.psi_cpu[0][0];
    let bands: Vec<Vec<Complex64>> = (0..n_bands)
        .map(|b| {
            let start = b * n_pw;
            psi_kpt0[start..start + n_pw].to_vec()
        })
        .collect();

    let have_gamma = if nkpts > 0 {
        state.k_points[0].coords.iter().all(|&c| c.abs() < 1e-12)
    } else {
        true
    };

    let kpt_data: Vec<KptWaveBlock> = (0..nkpts)
        .map(|ikpt| {
            let n_pw_kpt = state.pw_coords[ikpt].len();
            let psi_kpt = &state.psi_cpu[0][ikpt];
            let bands_kpt: Vec<Vec<Complex64>> = (0..n_bands)
                .map(|b| {
                    let start = b * n_pw_kpt;
                    psi_kpt[start..start + n_pw_kpt].to_vec()
                })
                .collect();
            KptWaveBlock {
                coords: state.k_points[ikpt].coords,
                nplw: n_pw_kpt,
                pw_grid_coord: state.pw_coords[ikpt].clone(),
                bands: bands_kpt,
            }
        })
        .collect();

    let wavefunction = WavefunctionCoeffs {
        have_gamma,
        grid: wave_grid_dims,
        nspins,
        kpt_data,
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
        kpoint_weights: (0..nkpts).map(|ikpt| state.k_points[ikpt].weight).collect(),
        fine_grid: Some(fine_grid_dims),
        wavefunction: Some(wavefunction),
        forces: None,
        stress: None,
        strain: None,
        hubbard_u: None,
        ldauoccm: None,
    })
}
