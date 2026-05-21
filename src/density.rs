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
    augment::q_apply::apply_q_and_sf,
    pseudopotential::{Pseudopotential, HasAugmentationData},
};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ndarray::{Array2, Array3};
use num_complex::Complex64;

use crate::device::fft::BatchedFftPlan3d;
use crate::device::{complex_slice_to_cuda, CudaComplex};
use crate::device::pcie::PcieAccount;
use crate::eigensolver::chebyshev::CudaKernelSet;
use crate::types::{ChemicalPotential, Density, Error, Occupations, SmearingParams, SmearingScheme, WaveGridArray};

// ---------------------------------------------------------------------------
// Occupation numbers (Gaussian smearing, CASTEP default)
// ---------------------------------------------------------------------------

/// Compute occupation numbers via Gaussian smearing.
///
/// occ_b = erfc((ε_b - μ) / w)
///
/// The chemical potential μ satisfies Σ_b occ_b = N_electrons.
pub(crate) fn compute_occupations(
    eigenvalues: &[f64],
    smearing: &SmearingParams,
    n_electrons: f64,
) -> Result<(Occupations, ChemicalPotential), Error> {
    match smearing.scheme {
        SmearingScheme::Gaussian => {
            let mu = find_chemical_potential(eigenvalues, smearing.width, n_electrons)?;
            let occ = Occupations(
                eigenvalues
                    .iter()
                    .map(|&e| libm::erfc((e - mu) / smearing.width))
                    .collect(),
            );
            Ok((occ, ChemicalPotential(mu)))
        }
    }
}

/// Bisection search for μ such that Σ erfc((ε_b - μ) / w) = N_electrons.
fn find_chemical_potential(
    eigenvalues: &[f64],
    width: f64,
    n_electrons: f64,
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
            .map(|&e| libm::erfc((e - mid) / width))
            .sum();
        if sum > n_electrons {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

// ---------------------------------------------------------------------------
// QSfCache: GPU-resident Q augmentation function cache
// ---------------------------------------------------------------------------

/// Per-ion Q augmentation function cache on GPU.
///
/// Stores `Q^I_{nm}(G) · exp(-iG·R_I)` for all non-zero (n_exp, m_exp) pairs
/// of a single ion. The flat GPU slice is indexed as `[pair_idx * n_fine_grid + grid_idx]`.
pub(crate) struct QSfIonEntry {
    /// Flat GPU slice: [n_pairs × n_fine_grid] CudaComplex.
    /// `Q^I_{nm}(G) · exp(-iG·R_I)` for all non-zero (n_exp, m_exp) pairs.
    pub q_sf: CudaSlice<CudaComplex>,
    /// Expanded projector pair indices: Vec<(n_exp, m_exp)>.
    pub pairs: Vec<(usize, usize)>,
    /// Number of expanded projectors for this ion.
    pub n_expanded: usize,
}

/// GPU cache for USPP augmentation density computation.
///
/// Caches `Q^I_{nm}(G) · exp(-iG·R_I)` per ion on GPU. Geometry-static:
/// built once at SCF init, invariant under SCF iterations.
///
/// Memory layout: species-shared approach (no per-ion structure factor).
/// Store `Q_{nm}(G)` without structure factor; apply `exp(-iG·R_I)` per-iteration
/// on GPU (cheap element-wise multiply).
pub(crate) struct QSfCache {
    /// Per-ion GPU slices. Each slice is flat [n_pairs × n_fine_grid] Complex128.
    pub entries: Vec<QSfIonEntry>,
    /// Fine grid dimensions [ngz, ngy, ngx].
    pub fine_grid: [usize; 3],
}

/// Build the QSfCache by computing Q augmentation functions for all ions.
///
/// For each ion, calls `apply_q_and_sf` to compute `Q^I_{nm}(G) · exp(-iG·R_I)`
/// on the fine grid, then uploads to GPU.
///
/// # Arguments
/// - `pots` — pseudopotential set (species-keyed)
/// - `cell` — cell geometry
/// - `fine_grid` — fine FFT grid
/// - `stream` — CUDA stream for H2D transfers
/// - `pcie` — PCIe accounting for memory tracking
///
/// # Returns
/// `QSfCache` with all ions' Q functions cached on GPU.
pub(crate) fn build_q_sf_cache(
    pots: &PseudopotentialSet,
    cell: &CellGeometry,
    fine_grid: &GVectorGrid,
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<QSfCache, Error> {
    let [ngz, ngy, ngx] = fine_grid.grid();
    let n_fine_grid = ngz * ngy * ngx;

    let mut entries = Vec::with_capacity(cell.num_ions);

    for ion_idx in 0..cell.num_ions {
        let species_idx = cell.ion_species[ion_idx];
        let symbol = &cell.species_symbols[species_idx];

        let Some(pot) = pots.get(symbol) else {
            // Species not in pseudopotential set: empty entry
            entries.push(QSfIonEntry {
                q_sf: stream.alloc_zeros(0).map_err(Error::Cuda)?,
                pairs: Vec::new(),
                n_expanded: 0,
            });
            continue;
        };

        // Check if this species has augmentation data (USP vs Recpot)
        let aug = match pot {
            Pseudopotential::Usp(usp_data) => usp_data,
            Pseudopotential::Recpot(_) => {
                // No augmentation: empty entry
                entries.push(QSfIonEntry {
                    q_sf: stream.alloc_zeros(0).map_err(Error::Cuda)?,
                    pairs: Vec::new(),
                    n_expanded: 0,
                });
                continue;
            }
        };

        let projectors = aug.projectors();
        let n_expanded = expanded_projector_count(projectors);
        let gmax_pp = pot.gmax();

        // Call apply_q_and_sf once per (n,m) pair with a unit matrix that has
        // only rho_nm[n,m] = 1.0. This extracts Q_{nm}(G)·exp(-iG·R_I) for
        // each pair individually, so compute_aug_density_gpu can contract with
        // the full ω^I_{nm} at runtime.
        let mut q_sf_flat: Vec<CudaComplex> = Vec::with_capacity(n_expanded * n_expanded * n_fine_grid);
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n_expanded * n_expanded);

        for n in 0..n_expanded {
            for m in 0..n_expanded {
                let mut rho_nm = Array2::<Complex64>::zeros((n_expanded, n_expanded));
                rho_nm[[n, m]] = Complex64::new(1.0, 0.0);

                let q_nm_g = apply_q_and_sf(&rho_nm, aug, cell, ion_idx, fine_grid, gmax_pp)
                    .map_err(|_| Error::NotImplemented)?;

                // q_nm_g is Array3<Complex64> of shape (ngz, ngy, ngx) in Fortran layout.
                // Append in row-major order (iter() follows memory order for F-layout).
                q_sf_flat.extend(q_nm_g.iter().map(|&c| CudaComplex { x: c.re, y: c.im }));
                pairs.push((n, m));
            }
        }

        // Upload flat [n_pairs × n_fine_grid] to GPU
        let q_sf_gpu = stream
            .clone_htod(&q_sf_flat)
            .map_err(Error::Cuda)?;

        pcie.record_h2d(&q_sf_gpu);

        entries.push(QSfIonEntry {
            q_sf: q_sf_gpu,
            pairs,
            n_expanded,
        });
    }

    Ok(QSfCache {
        entries,
        fine_grid: [ngz, ngy, ngx],
    })
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
#[builder]
pub(crate) fn construct_density_gpu(
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
    // cuFFT: n[0] outermost, n[rank-1] innermost. Our scatter formula makes
    // iz innermost, so plan dims = (ngx, ngy, ngz). See chebyshev.rs:844.
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32,
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
            .map(|c| (c.x as f64).powi(2) + (c.y as f64).powi(2))
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
pub(crate) fn compute_aug_density_fine(
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
