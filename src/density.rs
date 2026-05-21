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
    augment::q_apply::compute_q_nm_per_pair,
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
pub struct QSfIonEntry {
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
pub struct QSfCache {
    /// Per-ion GPU slices. Each slice is flat [n_pairs × n_fine_grid] Complex128.
    pub entries: Vec<QSfIonEntry>,
    /// Fine grid dimensions [ngz, ngy, ngx].
    pub fine_grid: [usize; 3],
}

/// Build the QSfCache by computing Q augmentation functions for all ions.
///
/// For each ion, calls `compute_q_nm_per_pair` to compute `Q_{nm}(G) · exp(-iG·R_I)`
/// for all (n_exp, m_exp) pairs in a single pass (radial Bessel transforms computed
/// once, reused across all pairs), then uploads the flat result to GPU.
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
pub fn build_q_sf_cache(
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

        // Compute Q_{nm}(G)·exp(-iG·R_I) for all (n_exp, m_exp) pairs in one
        // pass — qlnm_g (radial Bessel transforms) is computed once and reused
        // across all pairs, avoiding the O(n_pairs) redundant recomputation that
        // the old per-pair apply_q_and_sf loop incurred.
        let per_pair = compute_q_nm_per_pair(aug, cell, ion_idx, fine_grid)
            .map_err(|_| Error::NotImplemented)?;

        let mut q_sf_flat: Vec<CudaComplex> = Vec::with_capacity(n_expanded * n_expanded * n_fine_grid);
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n_expanded * n_expanded);

        for ((n, m), q_arr) in per_pair {
            q_sf_flat.extend(q_arr.iter().map(|&c| CudaComplex { x: c.re, y: c.im }));
            pairs.push((n, m));
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
/// Algorithm:
/// 1. For each ion I:
///    a. H2D beta_psi_I (n_expanded × n_bands)
///    b. gemm: ω^I = conj(βψ) · diag(occ) · βψ^T  (n_expanded × n_expanded)
///    c. gemv: ρ_aug(G) += Q_cache[I] · ω_flat  (n_fine_grid accumulation)
/// 2. C2C inverse FFT ρ_aug(G) → ρ_aug(r)
/// 3. D2H, normalize by 1/N_grid (cuFFT unnormalized), return RealGrid<f64>
pub fn compute_aug_density_gpu(
    q_sf_cache: &QSfCache,
    beta_psi_per_ion: &[Array2<Complex64>],
    occupations: &[f64],
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<RealGrid<f64>, Error> {
    let [ngz, ngy, ngx] = q_sf_cache.fine_grid;
    let n_fine_grid = ngz * ngy * ngx;
    let n_bands = occupations.len();

    let blas = BlasHandle::new(Arc::clone(stream)).map_err(Error::Blas)?;

    // Accumulator: ρ_aug(G) on GPU, shape [n_fine_grid], initialized to zero.
    let mut rho_aug_g: CudaSlice<CudaComplex> = stream
        .alloc_zeros(n_fine_grid)
        .map_err(Error::Cuda)?;

    // H2D occupations once
    let occ_host: Vec<CudaComplex> = occupations
        .iter()
        .map(|&o| CudaComplex { x: o, y: 0.0 })
        .collect();
    let occ_dev: CudaSlice<CudaComplex> = stream.clone_htod(&occ_host).map_err(Error::Cuda)?;

    for (ion_idx, entry) in q_sf_cache.entries.iter().enumerate() {
        let n_pairs = entry.pairs.len();
        if n_pairs == 0 || entry.q_sf.len() == 0 {
            continue;
        }
        let n_expanded = entry.n_expanded;
        if n_expanded == 0 {
            continue;
        }

        let bp = &beta_psi_per_ion[ion_idx];
        debug_assert_eq!(bp.shape()[0], n_expanded);
        debug_assert_eq!(bp.shape()[1], n_bands);

        // H2D beta_psi_I: shape (n_expanded × n_bands), col-major
        let bp_host: Vec<CudaComplex> = bp
            .iter()
            .map(|&c| CudaComplex { x: c.re, y: c.im })
            .collect();
        let bp_dev: CudaSlice<CudaComplex> = stream.clone_htod(&bp_host).map_err(Error::Cuda)?;
        pcie.record_h2d(&bp_dev);

        // Step 1b: ω^I_{nm} = Σ_b occ_b · conj(βψ_I)_{n,b} · (βψ_I)_{m,b}
        // = (βψ · diag(occ))^H · βψ  — but simpler: scale each column of βψ by sqrt(occ_b),
        // then ω = scaled_βψ · scaled_βψ^H.
        // We use: ω = conj(βψ) · diag(occ) · βψ^T
        // In cuBLAS col-major: ω = βψ^H · diag(occ) · βψ
        // Implemented as two steps: first scale βψ columns by occ, then gemm.
        //
        // Simpler: ω_{nm} = Σ_b occ_b · bp[n,b]* · bp[m,b]
        // = (bp^H · diag(occ) · bp) where bp is (n_expanded × n_bands) col-major.
        // cuBLAS ZGEMM: C = α·A^H·B + β·C
        //   A = bp (n_expanded × n_bands), A^H = (n_bands × n_expanded)
        //   B = bp (n_expanded × n_bands)
        //   C = ω (n_expanded × n_expanded)
        // But we need to weight by occ first. Scale bp columns by occ on GPU via axpy is complex.
        // Simpler: compute ω on CPU (n_expanded is small, ~18 for Cu).
        let ne = n_expanded;
        let mut omega_host: Vec<CudaComplex> = vec![CudaComplex { x: 0.0, y: 0.0 }; ne * ne];
        for n in 0..ne {
            for m in 0..ne {
                let mut acc = Complex64::ZERO;
                for b in 0..n_bands {
                    acc += occupations[b] * bp[[n, b]].conj() * bp[[m, b]];
                }
                // col-major: omega[m * ne + n]
                omega_host[m * ne + n] = CudaComplex { x: acc.re, y: acc.im };
            }
        }
        let omega_dev: CudaSlice<CudaComplex> = stream.clone_htod(&omega_host).map_err(Error::Cuda)?;

        // Step 1c: ρ_aug(G) += Σ_{nm} ω_{nm} · Q_{nm}(G)
        // Q cache: flat [n_pairs × n_fine_grid], row-major (pair_idx * n_fine_grid + g_idx)
        // ω_flat: [n_pairs] = [n_expanded × n_expanded] in same (n,m) order as pairs
        // This is: ρ_aug += Q^T · ω_flat  where Q is (n_pairs × n_fine_grid)
        // = gemv: y = α·A·x + β·y  with A=(n_fine_grid × n_pairs), x=ω_flat, y=ρ_aug
        // In cuBLAS col-major: A stored as (n_pairs × n_fine_grid) row-major
        //   = (n_fine_grid × n_pairs) col-major → use CUBLAS_OP_T
        // gemv: y(n_fine_grid) = α · A^T(n_fine_grid × n_pairs) · x(n_pairs) + β·y
        //   where A is stored col-major as (n_pairs × n_fine_grid)
        //   → transa=T, m=n_pairs, n=n_fine_grid → result has n_fine_grid elements
        let alpha = CudaComplex { x: 1.0, y: 0.0 };
        let beta  = CudaComplex { x: 1.0, y: 0.0 };
        // Q cache: [n_pairs × n_fine_grid] row-major = [n_fine_grid × n_pairs] col-major.
        // gemv: ρ_aug(n_fine_grid) += Q^T(n_fine_grid × n_pairs) · ω_flat(n_pairs)
        // cublasZgemv: y = α·op(A)·x + β·y
        //   A stored col-major as (n_pairs × n_fine_grid), trans=T → op(A) is (n_fine_grid × n_pairs)
        //   m=n_pairs (rows of A), n=n_fine_grid (cols of A), result length = n_fine_grid
        unsafe {
            blas.gemv_c64(
                op::T,
                n_pairs as i32,
                n_fine_grid as i32,
                alpha,
                &entry.q_sf,
                n_pairs as i32,
                &omega_dev,
                1,
                beta,
                &mut rho_aug_g,
                1,
            ).map_err(Error::Blas)?;
        }
    }

    // Step 2: C2C inverse FFT ρ_aug(G) → ρ_aug(r)
    // cuFFT plan dims: (ngx, ngy, ngz) — innermost first, matching scatter formula.
    let fft_plan = FftPlan3d::plan_c2c(ngx as i32, ngy as i32, ngz as i32, Arc::clone(stream))?;
    unsafe {
        let ptr = &mut rho_aug_g as *mut CudaSlice<CudaComplex>;
        fft_plan.c2c_inverse(&mut *ptr, &mut *ptr)?;
    }

    // Step 3: D2H, normalize by 1/N_grid (cuFFT is unnormalized), extract real part
    let rho_aug_host: Vec<CudaComplex> = stream.clone_dtoh(&rho_aug_g).map_err(Error::Cuda)?;
    pcie.record_d2h(&rho_aug_g);

    let inv_n = 1.0 / n_fine_grid as f64;
    // RealGrid stores data in Fortran layout (ngz, ngy, ngx).f()
    let rho_arr = Array3::from_shape_fn((ngz, ngy, ngx).f(), |(iz, iy, ix)| {
        let idx = iz + ngz * (iy + ngy * ix);
        rho_aug_host[idx].x as f64 * inv_n
    });

    Ok(RealGrid::from_inner(rho_arr))
}

/// Re-exports for integration tests that need to compare CPU and GPU
/// augmentation density paths directly.
pub mod test_api {
    pub use super::{
        QSfCache, QSfIonEntry,
        build_q_sf_cache,
        compute_aug_density_fine,
        compute_aug_density_gpu,
    };
}
