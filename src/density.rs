// ---------------------------------------------------------------------------
// Density construction from wavefunctions (GPU)
// ---------------------------------------------------------------------------
//
// 1. Compute Gaussian-smearing occupation numbers on CPU
// 2. Scatter sparse PW → full FFT grid on GPU
// 3. Batched C2C IFFT → ψ(r) for all bands
// 4. ρ(r) = (1/Ω) · Σ_b occ_b · |ψ_b(r)|²
// 5. D2H → Density

use std::sync::Arc;

use bon::builder;
use chemrust_hamiltonian_core::{
    assemble_aug_density_fine, CellGeometry, GVectorGrid, PseudopotentialSet,
    augment::beta_phi::expanded_projector_count,
    augment::q_apply::compute_q_nm_flat,
    pseudopotential::{Pseudopotential, HasAugmentationData},
    fft::RealGrid,
};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ndarray::{Array2, Array3, ShapeBuilder};
use num_complex::Complex64;

use crate::device::blas::{BlasHandle, op};
use crate::device::fft::{BatchedFftPlan3d, FftPlan3d};
use crate::device::{complex_slice_to_cuda, CudaComplex};
use crate::device::pcie::PcieAccount;
use crate::eigensolver::kernels::CudaKernelSet;
use crate::types::{ChemicalPotential, Density, Error, Occupations, SmearingParams, SmearingScheme, WaveGridArray};

// ---------------------------------------------------------------------------
// Occupation numbers (Gaussian smearing, CASTEP default)
// ---------------------------------------------------------------------------

/// Compute occupation numbers via Gaussian smearing.
///
/// occ_b = occ_factor · erfc((ε_b - μ) / w)
///
/// The chemical potential μ satisfies Σ_b occ_b = N_electrons.
///
/// `occ_factor` accounts for spin degeneracy:
/// - NonSpin (nspins=1): occ_factor = 1.0, range [0, 2], matches erfc directly.
/// - SpinCollinear (nspins=2): occ_factor = 0.5, range [0, 1], matching CASTEP's
///   `algor_integrated_broadening * real(2/nspins,dp)` formula.
///
/// CASTEP reference: algor.F90:2979 (algor_integrated_broadening = 0.5*erf(x)+0.5)
/// electronic.f90:9421 (occ = algor_integrated_broadening * real(2/nspins)).
pub fn compute_occupations(
    eigenvalues: &[f64],
    smearing: &SmearingParams,
    n_electrons: f64,
    occ_factor: f64,
) -> Result<(Occupations, ChemicalPotential), Error> {
    match smearing.scheme {
        SmearingScheme::Gaussian => {
            let mu = find_chemical_potential(eigenvalues, smearing.width.to_ha(), n_electrons, occ_factor)?;
            let occ = Occupations(
                eigenvalues
                    .iter()
                    .map(|&e| occ_factor * libm::erfc((e - mu) / smearing.width.to_ha()))
                    .collect(),
            );
            Ok((occ, ChemicalPotential(mu)))
        }
    }
}

/// Bisection search for μ such that Σ occ_factor · erfc((ε_b - μ) / w) = N_electrons.
fn find_chemical_potential(
    eigenvalues: &[f64],
    width: f64,
    n_electrons: f64,
    occ_factor: f64,
) -> Result<f64, Error> {
    let emin = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax = eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if eigenvalues.is_empty() || emax < emin {
        return Err(Error::NotImplemented);
    }
    let mut lo = emin - 10.0 * width;
    let mut hi = emax + 10.0 * width;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let sum: f64 = eigenvalues
            .iter()
            .map(|&e| occ_factor * libm::erfc((e - mid) / width))
            .sum();
        if sum > n_electrons {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Multi-kpt weighted occupation search: find μ such that
/// Σ_k w_k Σ_b occ_factor · erfc((ε_{bk} - μ) / w) = N_electrons_per_spin.
///
/// `occ_factor` accounts for spin degeneracy:
/// - NonSpin (nspins=1): occ_factor = 1.0, effective range [0, 2]
/// - SpinCollinear (nspins=2): occ_factor = 0.5, effective range [0, 1]
///
/// CASTEP reference: algor.F90:2979, electronic.f90:9421.
///
/// `per_kpt_eigenvalues` is a slice where each element is the eigenvalue list
/// for one k-point (length = n_bands each). `kpt_weights` must be the same
/// length and sum to 1.0 (or to N_kpts normalization).
///
/// For nkpts=1 with weight=1.0, this is equivalent to `compute_occupations`.
///
/// Returns `(Vec<Vec<f64>>, f64)` where the outer Vec is per-kpt occupations
/// and the inner Vecs are per-band occupations (length = n_bands each).
pub fn compute_occupations_weighted(
    per_kpt_eigenvalues: &[Vec<f64>],
    kpt_weights: &[f64],
    smearing: &SmearingParams,
    n_electrons_per_spin: f64,
    occ_factor: f64,
) -> Result<(Vec<Vec<f64>>, ChemicalPotential), Error> {
    match smearing.scheme {
        SmearingScheme::Gaussian => {
            // Flatten eigenvalues with weights for chemical potential search.
            // For the bisection, we need Σ_k w_k Σ_b occ_factor · erfc((ε_{bk} - μ) / w).
            let mu = find_chemical_potential_weighted(
                per_kpt_eigenvalues,
                kpt_weights,
                smearing.width.to_ha(),
                n_electrons_per_spin,
                occ_factor,
            )?;

            // Compute per-kpt occupations at the found μ
            let per_kpt_occs: Vec<Vec<f64>> = per_kpt_eigenvalues
                .iter()
                .map(|eigs| {
                    eigs.iter()
                        .map(|&e| occ_factor * libm::erfc((e - mu) / smearing.width.to_ha()))
                        .collect()
                })
                .collect();

            Ok((per_kpt_occs, ChemicalPotential(mu)))
        }
    }
}

/// Bisection search for μ such that
/// Σ_k w_k Σ_b occ_factor · erfc((ε_{bk} - μ) / w) = N.
fn find_chemical_potential_weighted(
    per_kpt_eigenvalues: &[Vec<f64>],
    kpt_weights: &[f64],
    width: f64,
    n_electrons: f64,
    occ_factor: f64,
) -> Result<f64, Error> {
    assert_eq!(
        per_kpt_eigenvalues.len(),
        kpt_weights.len(),
        "kpt_eigenvalues and kpt_weights must have same length"
    );

    // Find global eigenvalue range across all kpts
    let mut emin = f64::INFINITY;
    let mut emax = f64::NEG_INFINITY;
    for eigs in per_kpt_eigenvalues {
        for &e in eigs {
            if e < emin { emin = e; }
            if e > emax { emax = e; }
        }
    }
    if per_kpt_eigenvalues.is_empty() || emax < emin {
        return Err(Error::NotImplemented);
    }

    let mut lo = emin - 10.0 * width;
    let mut hi = emax + 10.0 * width;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let sum: f64 = per_kpt_eigenvalues
            .iter()
            .zip(kpt_weights.iter())
            .flat_map(|(eigs, &w)| {
                eigs.iter().map(move |&e| w * occ_factor * libm::erfc((e - mid) / width))
            })
            .sum();
        if sum > n_electrons {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// CASTEP lower-bound bisection for FIXED-spin occupation search.
///
/// Finds the Fermi energy for a single spin channel such that:
///   Σ_b occ_factor · erfc((ε_b - μ) / w) = n_spin_electrons
///
/// The returned `fermi_energy` is a **lower bound** (CASTEP convention).
/// After convergence, occupations are computed at the found fermi level.
///
/// # Algorithm (CASTEP electronic.f90:8602-8880)
/// 1. `lo = min(eigenvalues) - 4·width`, `hi = max(eigenvalues) + 4·width`
/// 2. `delta_E = hi - lo`, `fermi = lo`
/// 3. For 80 steps or until `delta_E <= 1e-12`:
///    - `delta_E /= 2`, `trial = fermi + delta_E`
///    - If `Σ occ_factor · erfc((e - trial)/width) <= n_spin_electrons`: `fermi = trial` (accept)
/// 4. Compute occupations at `fermi` and return.
pub fn find_fermi_fix(
    eigenvalues: &[f64],
    smearing: &SmearingParams,
    n_spin_electrons: f64,
    occ_factor: f64,
) -> Result<(f64, Vec<f64>), Error> {
    let width = smearing.width.to_ha();
    let emin = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax = eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if eigenvalues.is_empty() || emax < emin {
        return Err(Error::NotImplemented);
    }

    // Edge case: no electrons for this spin channel
    if n_spin_electrons <= 0.0 {
        let occupations = vec![0.0; eigenvalues.len()];
        return Ok((f64::NEG_INFINITY, occupations));
    }

    // Edge case: all bands fully occupied
    // (each band can contribute up to 2 * occ_factor — 2 for NonSpin, 1 for SpinCollinear)
    let max_possible = 2.0 * occ_factor * eigenvalues.len() as f64;
    if n_spin_electrons >= max_possible {
        let occupations = vec![2.0 * occ_factor; eigenvalues.len()];
        return Ok((f64::INFINITY, occupations));
    }

    let lo = emin - 4.0 * width;
    let hi = emax + 4.0 * width;
    let mut delta_e = hi - lo;
    let mut fermi = lo;

    for _ in 0..80 {
        delta_e *= 0.5;
        let trial = fermi + delta_e;
        let total_occ: f64 = eigenvalues
            .iter()
            .map(|&e| occ_factor * libm::erfc((e - trial) / width))
            .sum();
        if total_occ <= n_spin_electrons {
            fermi = trial;
        }
        if delta_e <= 1e-12 {
            break;
        }
    }

    let occupations: Vec<f64> = eigenvalues
        .iter()
        .map(|&e| occ_factor * libm::erfc((e - fermi) / width))
        .collect();

    Ok((fermi, occupations))
}

/// CASTEP lower-bound bisection for FREE-spin (shared Fermi energy).
///
/// Finds a SINGLE Fermi energy shared by both spin channels such that:
///   Σ_b occ_factor · erfc((ε↑_b - μ) / w) + Σ_b occ_factor · erfc((ε↓_b - μ) / w) = n_electrons
///
/// Returns (fermi_energy, occ_up, occ_dn, net_spin).
///
/// # Algorithm (CASTEP electronic.f90:8910-9209)
/// 1. `lo = min(ev_up ∪ ev_dn) - 4·width`, `hi = max(ev_up ∪ ev_dn) + 4·width`
/// 2. Same lower-bound bisection as `find_fermi_fix`, but summing over BOTH spins.
/// 3. After E_F found: compute occ_up, occ_dn at the shared fermi.
/// 4. `net_spin = Σ occ_up - Σ occ_dn`
/// 5. `fermi` is a lower bound. Both spin channels share the same `fermi` value.
pub fn find_fermi_free(
    ev_up: &[f64],
    ev_dn: &[f64],
    smearing: &SmearingParams,
    n_electrons: f64,
    occ_factor: f64,
) -> Result<(f64, Vec<f64>, Vec<f64>, f64), Error> {
    let width = smearing.width.to_ha();

    if ev_up.is_empty() || ev_dn.is_empty() {
        return Err(Error::NotImplemented);
    }

    let emin_up = ev_up.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax_up = ev_up.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let emin_dn = ev_dn.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax_dn = ev_dn.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let emin = emin_up.min(emin_dn);
    let emax = emax_up.max(emax_dn);

    // Edge case: no electrons
    if n_electrons <= 0.0 {
        let occ_up = vec![0.0; ev_up.len()];
        let occ_dn = vec![0.0; ev_dn.len()];
        return Ok((f64::NEG_INFINITY, occ_up, occ_dn, 0.0));
    }

    // Edge case: all bands fully occupied
    let max_possible = 2.0 * occ_factor * (ev_up.len() + ev_dn.len()) as f64;
    if n_electrons >= max_possible {
        let occ_up = vec![2.0 * occ_factor; ev_up.len()];
        let occ_dn = vec![2.0 * occ_factor; ev_dn.len()];
        return Ok((f64::INFINITY, occ_up, occ_dn, 0.0));
    }

    let lo = emin - 4.0 * width;
    let hi = emax + 4.0 * width;
    let mut delta_e = hi - lo;
    let mut fermi = lo;

    for _ in 0..80 {
        delta_e *= 0.5;
        let trial = fermi + delta_e;

        let total_occ_up: f64 = ev_up
            .iter()
            .map(|&e| occ_factor * libm::erfc((e - trial) / width))
            .sum();
        let total_occ_dn: f64 = ev_dn
            .iter()
            .map(|&e| occ_factor * libm::erfc((e - trial) / width))
            .sum();
        let total_occ = total_occ_up + total_occ_dn;

        if total_occ <= n_electrons {
            fermi = trial;
        }
        if delta_e <= 1e-12 {
            break;
        }
    }

    // Compute occupations at the shared fermi energy
    let occ_up: Vec<f64> = ev_up
        .iter()
        .map(|&e| occ_factor * libm::erfc((e - fermi) / width))
        .collect();
    let occ_dn: Vec<f64> = ev_dn
        .iter()
        .map(|&e| occ_factor * libm::erfc((e - fermi) / width))
        .collect();

    let net_spin: f64 = occ_up.iter().sum::<f64>() - occ_dn.iter().sum::<f64>();

    Ok((fermi, occ_up, occ_dn, net_spin))
}

/// K-point-weighted version of [`find_fermi_free`] for the CASTEP
/// `electronic_find_fermi_free` constraint (electronic.f90:9329-9600).
///
/// A single shared chemical potential μ is found by bisection on the
/// kpt-weighted total electron count:
///
/// ```text
/// total_occ(μ) = (2/nspins) · Σ_ns Σ_k w_k Σ_b I((μ − ε_bk)/w)  =  N
/// ```
///
/// where `I(x) = 0.5·erf(x) + 0.5 = 0.5·erfc((ε−μ)/w)` is the
/// integrated Gaussian broadening (CASTEP `algor_integrated_broadening`,
/// smearing_scheme GAUSSIAN).  The k-point weights `w_k` sum to 1
/// over the k-point list.
///
/// # Arguments
/// * `per_kpt_up` / `per_kpt_dn` — `[kpt][band]` eigenvalues in Hartree
/// * `kpt_weights` — k-point weights (`Σ w_k = 1`)
/// * `smearing` — smearing parameters (Gaussian width)
/// * `n_electrons` — total electron count `N` (both channels combined)
/// * `nspins` — number of spin channels (2 for SpinCollinear)
///
/// # Returns
/// `(fermi, occ_up_per_kpt, occ_dn_per_kpt)` with per-band occupations
/// `I((μ − ε)/w)` in `[0, 1]` per channel.
pub fn find_fermi_free_weighted(
    per_kpt_up: &[Vec<f64>],
    per_kpt_dn: &[Vec<f64>],
    kpt_weights: &[f64],
    smearing: &SmearingParams,
    n_electrons: f64,
    nspins: usize,
) -> Result<(f64, Vec<Vec<f64>>, Vec<Vec<f64>>), Error> {
    let width = smearing.width.to_ha();
    let scale = 2.0 / nspins.max(1) as f64;

    if per_kpt_up.is_empty() || per_kpt_dn.is_empty() {
        return Err(Error::NotImplemented);
    }
    if per_kpt_up.len() != per_kpt_dn.len() || per_kpt_up.len() != kpt_weights.len() {
        return Err(Error::NotImplemented);
    }

    let mut emin = f64::INFINITY;
    let mut emax = f64::NEG_INFINITY;
    for k in 0..per_kpt_up.len() {
        for &e in per_kpt_up[k].iter().chain(per_kpt_dn[k].iter()) {
            emin = emin.min(e);
            emax = emax.max(e);
        }
    }
    if n_electrons <= 0.0 {
        let occ_up: Vec<Vec<f64>> = per_kpt_up
            .iter()
            .map(|eigs| vec![0.0; eigs.len()])
            .collect();
        let occ_dn: Vec<Vec<f64>> = per_kpt_dn
            .iter()
            .map(|eigs| vec![0.0; eigs.len()])
            .collect();
        return Ok((f64::NEG_INFINITY, occ_up, occ_dn));
    }

    let total_occ = |mu: f64| -> f64 {
        let mut sum = 0.0;
        for k in 0..per_kpt_up.len() {
            let wk = kpt_weights[k];
            let bands: f64 = per_kpt_up[k]
                .iter()
                .map(|&e| 0.5 * libm::erfc((e - mu) / width))
                .sum::<f64>()
                + per_kpt_dn[k].iter().map(|&e| 0.5 * libm::erfc((e - mu) / width)).sum::<f64>();
            sum += wk * bands;
        }
        scale * sum
    };

    let lo = emin - 4.0 * width;
    let hi = emax + 4.0 * width;
    let mut delta_e = hi - lo;
    let mut fermi = lo;

    for _ in 0..80 {
        delta_e *= 0.5;
        let trial = fermi + delta_e;
        if total_occ(trial) <= n_electrons {
            fermi = trial;
        }
        if delta_e <= 1e-12 {
            break;
        }
    }

    let occ_up: Vec<Vec<f64>> = per_kpt_up
        .iter()
        .map(|eigs| {
            eigs.iter()
                .map(|&e| 0.5 * libm::erfc((e - fermi) / width))
                .collect()
        })
        .collect();
    let occ_dn: Vec<Vec<f64>> = per_kpt_dn
        .iter()
        .map(|eigs| {
            eigs.iter()
                .map(|&e| 0.5 * libm::erfc((e - fermi) / width))
                .collect()
        })
        .collect();

    Ok((fermi, occ_up, occ_dn))
}

// ---------------------------------------------------------------------------
// Electronic entropy -TS (Mermin free energy correction)
// ---------------------------------------------------------------------------

/// Compute the electronic entropy contribution TS for the Mermin free energy.
///
/// For Gaussian smearing, CASTEP's formula (electronic.f90:9768-9784) gives:
///   TS_spin = Σ_k w_k Σ_b exp(-((μ - ε_bk) / w)^2)
///   TS = Σ_ns TS_spin · w / (nspins · √π)
///
/// The total energy is then: E_total = E_band - E_H + E_xc - ρV_xc + E_ewald - TS.
///
/// `per_kpt_eigenvalues` is a slice where each element is the eigenvalue list
/// for one k-point (same format as `compute_occupations_weighted`).
pub fn compute_entropy_ts(
    per_kpt_eigenvalues: &[Vec<f64>],
    kpt_weights: &[f64],
    fermi_energy: f64,
    width: f64,
) -> f64 {
    let mut spin_ts = 0.0;
    for (ikpt, eigs) in per_kpt_eigenvalues.iter().enumerate() {
        let wk = kpt_weights[ikpt];
        let mut band_sum = 0.0;
        for &e in eigs {
            let x = (fermi_energy - e) / width;
            band_sum += (-x * x).exp();
        }
        spin_ts += wk * band_sum;
    }
    spin_ts
}

// ---------------------------------------------------------------------------
// QSfCache: GPU-resident Q augmentation function cache
// ---------------------------------------------------------------------------

/// Per-species Q augmentation function cache on GPU.
///
/// Stores `Q_{nm}(G)` (no structure factor) for all (n_exp, m_exp) pairs of
/// one species. The flat GPU slice is indexed as `[pair_idx * n_fine_grid + g_idx]`.
/// The structure factor `exp(-iG·R_I)` is applied per-ion at contraction time.
pub struct QSfSpeciesEntry {
    /// Flat GPU slice: [n_pairs × n_fine_grid] CudaComplex.
    /// `Q_{nm}(G)` without structure factor.
    pub q_nm: CudaSlice<CudaComplex>,
    /// Number of expanded projectors for this species.
    pub n_expanded: usize,
    /// n_pairs = n_expanded²
    pub n_pairs: usize,
}

/// Per-ion structure factor cache on GPU.
///
/// Stores `exp(-iG·R_I)` for one ion as a flat [n_fine_grid] GPU slice.
/// Geometry-static: built once per cell, reused every SCF iteration.
pub struct IonSfEntry {
    /// Flat GPU slice: [n_fine_grid] CudaComplex. `exp(-iG·R_I)`.
    pub sf: CudaSlice<CudaComplex>,
}

/// GPU cache for USPP augmentation density computation.
///
/// Species-shared layout: `Q_{nm}(G)` stored once per species (no structure
/// factor). Structure factors `exp(-iG·R_I)` stored per ion. At contraction
/// time, `Q_{nm}(G) · exp(-iG·R_I)` is formed on-the-fly via element-wise
/// multiply into a temporary, then contracted with `ω^I_{nm}` via gemv.
///
/// Memory: n_species × n_pairs × n_fine_grid × 16 bytes
///       + n_ions × n_fine_grid × 16 bytes
/// For Cu111_CO: 1 × 324 × 437k × 16 ≈ 2.3 GB  +  18 × 437k × 16 ≈ 126 MB
pub struct QSfCache {
    /// Per-species Q function slices, keyed by species index.
    pub species_entries: Vec<Option<QSfSpeciesEntry>>,
    /// Per-ion structure factor slices.
    pub ion_sf: Vec<IonSfEntry>,
    /// ion_species[ion_idx] = species_idx — mirrors CellGeometry.ion_species.
    pub ion_species: Vec<usize>,
    /// Fine grid dimensions [ngz, ngy, ngx].
    pub fine_grid: [usize; 3],
}

/// Build the QSfCache using the species-shared layout.
///
/// Computes `Q_{nm}(G)` once per species (no structure factor) and
/// `exp(-iG·R_I)` once per ion. Total VRAM: O(n_species × n_pairs × n_fine_grid).
pub fn build_q_sf_cache(
    pots: &PseudopotentialSet,
    cell: &CellGeometry,
    fine_grid: &GVectorGrid,
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<QSfCache, Error> {
    let [ngz, ngy, ngx] = fine_grid.grid();
    let n_fine_grid = ngz * ngy * ngx;
    let tau = 2.0 * std::f64::consts::PI;

    // --- Per-species Q_{nm}(G) (no structure factor) ---
    // Use a dummy ion_idx=0 position of (0,0,0) so exp(-iG·R)=1 and
    // compute_q_nm_per_pair returns pure Q_{nm}(G).
    let n_species = cell.num_species;
    let mut species_entries: Vec<Option<QSfSpeciesEntry>> = Vec::with_capacity(n_species);

    // Build a temporary cell with all ions at the origin to strip the SF.
    let mut cell_origin = cell.clone();
    for mut row in cell_origin.ionic_positions.rows_mut() {
        row.fill(0.0);
    }

    for species_idx in 0..n_species {
        let symbol = &cell.species_symbols[species_idx];
        let Some(pot) = pots.get(symbol) else {
            species_entries.push(None);
            continue;
        };
        let aug = match pot {
            Pseudopotential::Usp(d) => d,
            Pseudopotential::Recpot(_) => {
                species_entries.push(None);
                continue;
            }
        };

        let projectors = aug.projectors();
        let n_expanded = expanded_projector_count(projectors);
        let n_pairs = n_expanded * n_expanded;

        // Find the first ion of this species to use as the representative.
        let rep_ion = cell.ion_species.iter().position(|&s| s == species_idx)
            .unwrap_or(0);

        // Single-pass flat buffer: [n_pairs × n_fine_grid], pair-major grid-minor.
        let (q_flat_complex, _) = compute_q_nm_flat(aug, &cell_origin, rep_ion, fine_grid)
            .map_err(|_| Error::NotImplemented)?;

        let q_flat: Vec<CudaComplex> = q_flat_complex.iter()
            .map(|&c| CudaComplex { x: c.re, y: c.im })
            .collect();

        let q_gpu = stream.clone_htod(&q_flat).map_err(Error::Cuda)?;
        pcie.record_h2d(&q_gpu);

        species_entries.push(Some(QSfSpeciesEntry { q_nm: q_gpu, n_expanded, n_pairs }));
    }

    // --- Per-ion structure factors exp(-iG·R_I) ---
    let mut ion_sf: Vec<IonSfEntry> = Vec::with_capacity(cell.num_ions);

    for ion_idx in 0..cell.num_ions {
        let pos = cell.ionic_positions.row(ion_idx);
        let (rx, ry, rz) = (pos[0], pos[1], pos[2]);

        // Fortran order: iz fastest — matches cuFFT scatter formula iz + ngz*(iy + ngy*ix)
        // and the Q_{nm} flat buffer from compute_q_nm_per_pair (Array3 F-order iteration).
        let sf_host: Vec<CudaComplex> = {
            let mut v = Vec::with_capacity(n_fine_grid);
            for ix in 0..ngx {
                for iy in 0..ngy {
                    for iz in 0..ngz {
                        let gf = fine_grid.gvecs()[[iz, iy, ix]];
                        let phase = -tau * (gf[0] * rx + gf[1] * ry + gf[2] * rz);
                        let (s, c) = phase.sin_cos();
                        v.push(CudaComplex { x: c, y: s });
                    }
                }
            }
            v
        };

        let sf_gpu = stream.clone_htod(&sf_host).map_err(Error::Cuda)?;
        pcie.record_h2d(&sf_gpu);
        ion_sf.push(IonSfEntry { sf: sf_gpu });
    }

    Ok(QSfCache { species_entries, ion_sf, ion_species: cell.ion_species.clone(), fine_grid: [ngz, ngy, ngx] })
}


// ---------------------------------------------------------------------------
// GPU density construction (builder API via bon)
// ---------------------------------------------------------------------------

/// Build electron density from wavefunctions on GPU.
///
/// Pipeline:
/// 1. H2D psi, fft_indices, occupations
/// 2. Scatter sparse PW → full FFT grid (reuse `scatter_pw_to_grid` kernel)
/// 3. Batched C2C IFFT (reuse `BatchedFftPlan3d`)
/// 4. ρ[r] = Σ_b occ_b |ψ_b[r]|² (`accumulate_density` kernel)
/// 5. D2H → Density(WaveGridArray)
///
/// **Unit convention**: the output is in CASTEP raw units (ρ_phys × V_cell),
/// matching `.castep_bin` density storage and `solve_poisson`/`compute_pbe_xc`
/// expectations downstream. The `accumulate_density` kernel multiplies by
/// `inv_omega = 1.0`, i.e. no Ω division (left as a parameter for potential
/// future Ha/Bohr³ callers, but always 1.0 in this SCF pipeline).
#[doc(hidden)]
#[builder]
pub fn construct_density_gpu(
    psi_data: &[Complex64],
    occupations: &[f64],
    fft_indices: &[i32],
    wave_grid: &GVectorGrid,
    cell_volume: f64,
    n_bands: usize,
    n_pw: usize,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<Density, Error> {
    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = (ngz * ngy * ngx) as i32;
    // CASTEP raw density convention: ρ stored as ρ_phys × V_cell (electrons
    // per grid point × N_grid). solve_poisson + compute_pbe_xc downstream
    // expect this convention. We keep `inv_omega` as a kernel parameter to
    // preserve the existing call site, but pass 1.0 to skip the Ω division.
    let _ = cell_volume;
    let inv_omega = 1.0_f64;

    let n_bands_i = n_bands as i32;
    let n_pw_i = n_pw as i32;

    // 1. H2D: psi, fft_indices, occupations
    let psi_slice: Vec<CudaComplex> = complex_slice_to_cuda(psi_data);
    let psi_dev: CudaSlice<CudaComplex> = stream
        .clone_htod(&psi_slice)
        .map_err(Error::Cuda)?;
    let fft_idx_dev: CudaSlice<i32> = stream
        .clone_htod(fft_indices)
        .map_err(Error::Cuda)?;
    let occ_dev: CudaSlice<f64> = stream
        .clone_htod(occupations)
        .map_err(Error::Cuda)?;

    // 2. Allocate + zero the full FFT grid: n_bands × grid_size complex
    let mut grid_dev: CudaSlice<CudaComplex> = {
        let g: CudaSlice<CudaComplex> =
            stream.alloc_zeros(n_bands * grid_size as usize).map_err(Error::Cuda)?;
        g
    };

    // 3. Scatter sparse PW → full FFT grid
    unsafe {
        stream
            .launch_builder(&kernels.scatter_pw_to_grid)
            .arg(&psi_dev)
            .arg(&fft_idx_dev)
            .arg(&mut grid_dev)
            .arg(&n_pw_i)
            .arg(&n_bands_i)
            .arg(&grid_size)
            .launch(LaunchConfig::for_num_elems(
                (n_bands_i * n_pw_i) as u32,
            ))
    }
    .map_err(Error::Cuda)?;

    // 4. Batched C2C IFFT (in-place on grid_dev)
    // Fortran data layout (ngz, ngy, ngx) with ngz innermost (stride-1).
    // cuFFT n[0] is innermost, so plan dims = (ngz, ngy, ngx).
    // Verified by cufft_dim_ordering_isolated_diagnostic.
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngz as i32, ngy as i32, ngx as i32,
        n_bands as i32, Arc::clone(stream),
    )?;
    // In-place IFFT: same buffer for input and output via raw pointer
    unsafe {
        let ptr = &mut grid_dev as *mut CudaSlice<CudaComplex>;
        fft_plan.c2c_inverse(&mut *ptr, &mut *ptr)?
    };

    // DIAGNOSTIC: probe Σ|grid[r]|² for the first band to pin down the
    // missing factor of ~4 in the density normalization.
    // - If Σ|grid|² == N (=ngx·ngy·ngz): IFFT is unnormalized, Σ_G|c|²=1 holds
    //   → factor-of-4 lives in `accumulate_density` kernel or in `inv_omega`
    // - If Σ|grid|² == N/4: IFFT or the PW coef convention carries the factor
    // - If Σ|grid|² ≈ 1: IFFT divides by N (fully normalized)
    {
        let probe: Vec<CudaComplex> = stream
            .clone_dtoh(&grid_dev)
            .map_err(Error::Cuda)?;
        let s_b0: f64 = probe
            .iter()
            .take(grid_size as usize)
            .map(|c| c.x.powi(2) + c.y.powi(2))
            .sum();
        let psi_pw_norm_b0: f64 = psi_data
            .iter()
            .take(n_pw)
            .map(|c| c.re * c.re + c.im * c.im)
            .sum();
        eprintln!(
            "[ConstructDensity] band-0 Σ|grid[r]|² = {:.6e}  N=ngx·ngy·ngz={}  Σ_G|c_G|² = {:.6e}  ratio Σ|grid|² / (N · Σ|c|²) = {:.6e}",
            s_b0,
            grid_size,
            psi_pw_norm_b0,
            s_b0 / (grid_size as f64 * psi_pw_norm_b0),
        );
    }

    // 5. Accumulate density: ρ[r] = inv_omega × Σ_b occ[b] × |ψ_b[r]|²
    let mut rho_dev: CudaSlice<f64> = {
        let r: CudaSlice<f64> =
            stream.alloc_zeros(grid_size as usize).map_err(Error::Cuda)?;
        r
    };
    unsafe {
        stream
            .launch_builder(&kernels.accumulate_density)
            .arg(&grid_dev)
            .arg(&occ_dev)
            .arg(&mut rho_dev)
            .arg(&n_bands_i)
            .arg(&grid_size)
            .arg(&inv_omega)
            .launch(LaunchConfig::for_num_elems(grid_size as u32))
    }
    .map_err(Error::Cuda)?;

    // 6. D2H density
    let rho_host: Vec<f64> = stream
        .clone_dtoh(&rho_dev)
        .map_err(Error::Cuda)?;
    let array = Array3::from_shape_vec((ngx, ngy, ngz), rho_host)
        .map_err(|_| Error::NotImplemented)?;

    Ok(Density::from_inner(WaveGridArray::from_inner(array)))
}

// ---------------------------------------------------------------------------
// USPP augmentation density on the fine grid
// ---------------------------------------------------------------------------

/// Build the USPP augmentation density `ρ_aug(r)` on the fine grid from
/// cached `⟨β|ψ⟩` projections and band occupations.
///
/// Pipeline (CPU-only):
/// 1. For each ion `I`, compute `ω^I_{nm} = Σ_b occ_b · conj(βψ_I)_{n,b} · (βψ_I)_{m,b}`.
///    `ω^I` is Hermitian by construction.
/// 2. Hand the per-ion `ω` slice to `chemrust_hamiltonian_core::assemble_aug_density_fine`,
///    which sums `Σ_I ω^I · Q^I(G) · exp(-iG·R_I)` and inverse-FFTs to real
///    space on the fine grid.
///
/// `beta_psi_per_ion` must have one entry per ion in `cell.ionic_positions`,
/// each shape `(n_expanded × n_bands)`. Ions whose pseudopotential lacks
/// augmentation (Recpot) contribute nothing and may carry any value (the
/// upstream wrapper skips them).
pub fn compute_aug_density_fine(
    beta_psi_per_ion: &[Array2<Complex64>],
    occupations: &[f64],
    pots: &PseudopotentialSet,
    cell: &CellGeometry,
    fine_grid: &GVectorGrid,
) -> Result<chemrust_hamiltonian_core::fft::RealGrid<f64>, Error> {
    debug_assert_eq!(
        beta_psi_per_ion.len(),
        cell.num_ions,
        "beta_psi_per_ion length {} must equal cell.num_ions {}",
        beta_psi_per_ion.len(),
        cell.num_ions,
    );

    let rho_nm_per_ion: Vec<Array2<Complex64>> = beta_psi_per_ion
        .iter()
        .map(|bp| {
            let (ne, n_bands) = (bp.shape()[0], bp.shape()[1]);
            debug_assert_eq!(
                n_bands,
                occupations.len(),
                "beta_psi n_bands ({}) must match occupations len ({})",
                n_bands,
                occupations.len(),
            );
            // occ[b] ∈ [0,2] via erfc smearing — spin degeneracy already encoded.
            // CASTEP ion.f90:7114 multiplies by 2.0 because its occ ∈ [0,1].
            // No extra spin_deg factor here.
            let mut rho_nm = Array2::<Complex64>::zeros((ne, ne));
            for n in 0..ne {
                for m in 0..ne {
                    let mut acc = Complex64::ZERO;
                    for b in 0..n_bands {
                        acc += occupations[b] * bp[[n, b]].conj() * bp[[m, b]];
                    }
                    rho_nm[[n, m]] = acc;
                }
            }
            rho_nm
        })
        .collect();

    assemble_aug_density_fine(&rho_nm_per_ion, pots, cell, fine_grid)
        .map_err(|_| Error::NotImplemented)
}

// ---------------------------------------------------------------------------
// GPU augmentation density: compute_aug_density_gpu
// ---------------------------------------------------------------------------

/// Compute ρ_aug(r) on the fine grid using GPU-resident QSfCache.
///
/// Algorithm per ion I:
/// 1. Compute ω^I_{nm} on CPU (n_expanded ~18, cheap)
/// 2. H2D ω^I
/// 3. Allocate tmp[n_fine_grid]: tmp[g] = Σ_{nm} ω_{nm} · Q_{nm}(g) via gemv
///    (uses species-shared Q_{nm}(G) from cache)
/// 4. Element-wise multiply tmp[g] *= exp(-iG·R_I) (from ion_sf cache)
/// 5. Accumulate: ρ_aug(G) += tmp
///
/// After all ions: C2C inverse FFT, D2H, normalize.
pub fn compute_aug_density_gpu(
    q_sf_cache: &QSfCache,
    beta_psi_per_ion: &[CudaSlice<CudaComplex>],
    occupations: &[f64],
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
    kernels: &CudaKernelSet,
) -> Result<RealGrid<f64>, Error> {
    let [ngz, ngy, ngx] = q_sf_cache.fine_grid;
    let n_fine_grid = ngz * ngy * ngx;
    let n_bands = occupations.len();

    let blas = BlasHandle::new(Arc::clone(stream)).map_err(Error::Blas)?;

    let mut rho_aug_g: CudaSlice<CudaComplex> = stream
        .alloc_zeros(n_fine_grid)
        .map_err(Error::Cuda)?;

    for (ion_idx, bp_dev) in beta_psi_per_ion.iter().enumerate() {
        let species_idx = q_sf_cache.ion_species[ion_idx];

        let species_entry = match q_sf_cache.species_entries.get(species_idx).and_then(|e| e.as_ref()) {
            Some(e) => e,
            None => continue,
        };

        let n_expanded = species_entry.n_expanded;
        let n_pairs = species_entry.n_pairs;

        // D2H the GPU-resident βψ_I for ω computation on CPU
        let bp_host: Vec<CudaComplex> = stream.clone_dtoh(bp_dev).map_err(Error::Cuda)?;
        pcie.d2h_bytes += bp_host.len() * std::mem::size_of::<CudaComplex>();

        // Convert to ndarray Array2 for indexing (col-major layout)
        let bp_complex: Vec<Complex64> = bp_host
            .iter()
            .map(|c| Complex64::new(c.x, c.y))
            .collect();
        let bp = Array2::from_shape_vec(
            (n_expanded, n_bands).f(),
            bp_complex,
        )
        .map_err(|_| Error::NotImplemented)?;

        // ω^I_{nm} on CPU (n_expanded ~18, O(ne² × n_bands) ≈ 18² × 160 = 52k ops)
        // occ[b] ∈ [0,2] via erfc smearing — spin degeneracy already encoded.
        // No extra spin_deg factor (CASTEP ion.f90:7114 multiplies by 2 because its occ ∈ [0,1]).
        let mut omega_host: Vec<CudaComplex> = vec![CudaComplex { x: 0.0, y: 0.0 }; n_pairs];
        for n in 0..n_expanded {
            for m in 0..n_expanded {
                let mut acc = Complex64::ZERO;
                for b in 0..n_bands {
                    acc += occupations[b] * bp[[n, b]].conj() * bp[[m, b]];
                }
                // row-major pair order: pair_idx = n * n_expanded + m
                omega_host[n * n_expanded + m] = CudaComplex { x: acc.re, y: acc.im };
            }
        }
        let omega_dev: CudaSlice<CudaComplex> = stream.clone_htod(&omega_host).map_err(Error::Cuda)?;

        // tmp[g] = Σ_{nm} ω_{nm} · Q_{nm}(g)
        // Q flat buffer layout: flat[pair_idx * n_fine_grid + g_idx] (pair-major, grid-minor).
        // Interpreted as col-major matrix: element [g, p] = flat[g + n_fine_grid * p]
        //   → this IS col-major (n_fine_grid × n_pairs) with lda=n_fine_grid.
        // gemv: tmp(n_fine_grid) = A(n_fine_grid × n_pairs) · ω(n_pairs)
        //   → trans=N, m=n_fine_grid, n=n_pairs, lda=n_fine_grid
        let mut tmp: CudaSlice<CudaComplex> = stream.alloc_zeros(n_fine_grid).map_err(Error::Cuda)?;
        let one  = CudaComplex { x: 1.0, y: 0.0 };
        let zero = CudaComplex { x: 0.0, y: 0.0 };
        unsafe {
            blas.gemv_c64(
                op::N,
                n_fine_grid as i32,
                n_pairs as i32,
                one,
                &species_entry.q_nm,
                n_fine_grid as i32,
                &omega_dev,
                1,
                zero,
                &mut tmp,
                1,
            ).map_err(Error::Blas)?;
        }

        // tmp[g] *= exp(-iG·R_I)  (element-wise, using ion_sf cache on GPU)
        unsafe {
            stream
                .launch_builder(&kernels.cpx_mul_inplace)
                .arg(&mut tmp)
                .arg(&q_sf_cache.ion_sf[ion_idx].sf)
                .arg(&(n_fine_grid as i32))
                .launch(LaunchConfig::for_num_elems(n_fine_grid as u32))
        }
        .map_err(Error::Cuda)?;

        // ρ_aug(G) += tmp
        blas.axpy_c64(n_fine_grid as i32, one, &tmp, 1, &mut rho_aug_g, 1)
            .map_err(Error::Blas)?;
    }

    // C2C inverse FFT ρ_aug(G) → ρ_aug(r)
    let fft_plan = FftPlan3d::plan_c2c(ngx as i32, ngy as i32, ngz as i32, Arc::clone(stream))?;
    unsafe {
        let ptr = &mut rho_aug_g as *mut CudaSlice<CudaComplex>;
        fft_plan.c2c_inverse(&mut *ptr, &mut *ptr)?;
    }

    let rho_aug_host: Vec<CudaComplex> = stream.clone_dtoh(&rho_aug_g).map_err(Error::Cuda)?;
    pcie.record_d2h(&rho_aug_g);

    // cuFFT inverse is unnormalized (same as CPU fft_inverse_3d). No 1/N factor.
    // cuFFT plan (ngx, ngy, ngz) with iz innermost → flat index iz + ngz*(iy + ngy*ix).
    // CPU fft_inverse_3d returns (ngx, ngy, ngz) C-order. Match that shape.
    let rho_arr = Array3::from_shape_fn((ngx, ngy, ngz), |(ix, iy, iz)| {
        let idx = iz + ngz * (iy + ngy * ix);
        rho_aug_host[idx].x
    });

    Ok(RealGrid::from_inner(rho_arr))
}

/// Re-exports for integration tests that need to compare CPU and GPU
/// augmentation density paths directly.
pub mod test_api {
    pub use super::{
        QSfCache, QSfSpeciesEntry, IonSfEntry,
        build_q_sf_cache,
        compute_aug_density_fine,
        compute_aug_density_gpu,
        construct_density_gpu,
        save_q_sf_cache_to_disk,
        load_q_sf_cache_from_disk,
    };
    pub use crate::eigensolver::kernels::CudaKernelSet;
    #[cfg(feature = "chebyshev")]
    pub use crate::eigensolver::chebyshev::FilterMode;
    #[cfg(feature = "chebyshev")]
    pub use crate::eigensolver::hamiltonian::check_s_inv_s_identity;
    pub use crate::eigensolver::vnl_data::{VnlBatchData, VnlIonData};
    #[cfg(all(any(test, feature = "scf_diag"), any(test, feature = "chebyshev")))]
    pub use crate::eigensolver::rayleigh_ritz::rayleigh_ritz_with_matrices;
}

// ---------------------------------------------------------------------------
// QSfCache disk serialization (for test caching)
// ---------------------------------------------------------------------------

/// Serialized form of QSfCache for disk storage.
/// Layout: a simple binary format — no external dependencies.
///
/// Format:
///   [u64: n_species]
///   for each species:
///     [u8: present (1) or absent (0)]
///     if present:
///       [u64: n_expanded] [u64: n_pairs] [u64: n_fine_grid]
///       [n_pairs * n_fine_grid * 16 bytes: q_nm flat f64 pairs]
///   [u64: n_ions]
///   for each ion:
///     [u64: n_fine_grid] [n_fine_grid * 16 bytes: sf flat f64 pairs]
///   [u64: ngz] [u64: ngy] [u64: ngx]
///   [u64: n_ions for ion_species]
///   [n_ions * 8 bytes: ion_species u64 values]
pub fn save_q_sf_cache_to_disk(
    cache: &QSfCache,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    path: &std::path::Path,
) -> Result<(), Error> {
    let mut buf: Vec<u8> = Vec::new();

    let write_u64 = |buf: &mut Vec<u8>, v: u64| buf.extend_from_slice(&v.to_le_bytes());
    let write_f64 = |buf: &mut Vec<u8>, v: f64| buf.extend_from_slice(&v.to_le_bytes());

    write_u64(&mut buf, cache.species_entries.len() as u64);
    for entry in &cache.species_entries {
        match entry {
            None => { buf.push(0); }
            Some(e) => {
                buf.push(1);
                write_u64(&mut buf, e.n_expanded as u64);
                write_u64(&mut buf, e.n_pairs as u64);
                let n_fine = e.q_nm.len() / e.n_pairs;
                write_u64(&mut buf, n_fine as u64);
                let host: Vec<CudaComplex> = stream.clone_dtoh(&e.q_nm).map_err(Error::Cuda)?;
                for c in &host { write_f64(&mut buf, c.x); write_f64(&mut buf, c.y); }
            }
        }
    }

    write_u64(&mut buf, cache.ion_sf.len() as u64);
    let [ngz, ngy, ngx] = cache.fine_grid;
    let n_fine = ngz * ngy * ngx;
    for ion in &cache.ion_sf {
        write_u64(&mut buf, n_fine as u64);
        let host: Vec<CudaComplex> = stream.clone_dtoh(&ion.sf).map_err(Error::Cuda)?;
        for c in &host { write_f64(&mut buf, c.x); write_f64(&mut buf, c.y); }
    }

    write_u64(&mut buf, ngz as u64);
    write_u64(&mut buf, ngy as u64);
    write_u64(&mut buf, ngx as u64);

    write_u64(&mut buf, cache.ion_species.len() as u64);
    for &s in &cache.ion_species { write_u64(&mut buf, s as u64); }

    std::fs::write(path, &buf).map_err(|e| Error::Io(e.to_string()))?;
    Ok(())
}

/// Load a QSfCache from disk (written by `save_q_sf_cache_to_disk`).
pub fn load_q_sf_cache_from_disk(
    path: &std::path::Path,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<QSfCache, Error> {
    let bytes = std::fs::read(path).map_err(|e| Error::Io(e.to_string()))?;
    let mut pos = 0usize;

    let read_u64 = |bytes: &[u8], pos: &mut usize| -> u64 {
        let v = u64::from_le_bytes(bytes[*pos..*pos+8].try_into().unwrap());
        *pos += 8;
        v
    };
    let read_f64 = |bytes: &[u8], pos: &mut usize| -> f64 {
        let v = f64::from_le_bytes(bytes[*pos..*pos+8].try_into().unwrap());
        *pos += 8;
        v
    };

    let n_species = read_u64(&bytes, &mut pos) as usize;
    let mut species_entries: Vec<Option<QSfSpeciesEntry>> = Vec::with_capacity(n_species);
    for _ in 0..n_species {
        let present = bytes[pos]; pos += 1;
        if present == 0 {
            species_entries.push(None);
        } else {
            let n_expanded = read_u64(&bytes, &mut pos) as usize;
            let n_pairs    = read_u64(&bytes, &mut pos) as usize;
            let n_fine     = read_u64(&bytes, &mut pos) as usize;
            let mut host: Vec<CudaComplex> = Vec::with_capacity(n_pairs * n_fine);
            for _ in 0..n_pairs * n_fine {
                let x = read_f64(&bytes, &mut pos);
                let y = read_f64(&bytes, &mut pos);
                host.push(CudaComplex { x, y });
            }
            let gpu = stream.clone_htod(&host).map_err(Error::Cuda)?;
            pcie.record_h2d(&gpu);
            species_entries.push(Some(QSfSpeciesEntry { q_nm: gpu, n_expanded, n_pairs }));
        }
    }

    let n_ions = read_u64(&bytes, &mut pos) as usize;
    let mut ion_sf: Vec<IonSfEntry> = Vec::with_capacity(n_ions);
    for _ in 0..n_ions {
        let n_fine = read_u64(&bytes, &mut pos) as usize;
        let mut host: Vec<CudaComplex> = Vec::with_capacity(n_fine);
        for _ in 0..n_fine {
            let x = read_f64(&bytes, &mut pos);
            let y = read_f64(&bytes, &mut pos);
            host.push(CudaComplex { x, y });
        }
        let gpu = stream.clone_htod(&host).map_err(Error::Cuda)?;
        pcie.record_h2d(&gpu);
        ion_sf.push(IonSfEntry { sf: gpu });
    }

    let ngz = read_u64(&bytes, &mut pos) as usize;
    let ngy = read_u64(&bytes, &mut pos) as usize;
    let ngx = read_u64(&bytes, &mut pos) as usize;

    let n_ion_species = read_u64(&bytes, &mut pos) as usize;
    let mut ion_species: Vec<usize> = Vec::with_capacity(n_ion_species);
    for _ in 0..n_ion_species {
        ion_species.push(read_u64(&bytes, &mut pos) as usize);
    }

    Ok(QSfCache { species_entries, ion_sf, ion_species, fine_grid: [ngz, ngy, ngx] })
}
