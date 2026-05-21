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

        // compute_q_nm_per_pair with origin cell → pure Q_{nm}(G), no SF.
        let per_pair = compute_q_nm_per_pair(aug, &cell_origin, rep_ion, fine_grid)
            .map_err(|_| Error::NotImplemented)?;

        let mut q_flat: Vec<CudaComplex> = Vec::with_capacity(n_pairs * n_fine_grid);
        for (_, q_arr) in per_pair {
            q_flat.extend(q_arr.iter().map(|&c| CudaComplex { x: c.re, y: c.im }));
        }

        let q_gpu = stream.clone_htod(&q_flat).map_err(Error::Cuda)?;
        pcie.record_h2d(&q_gpu);

        species_entries.push(Some(QSfSpeciesEntry { q_nm: q_gpu, n_expanded, n_pairs }));
    }

    // --- Per-ion structure factors exp(-iG·R_I) ---
    let mut ion_sf: Vec<IonSfEntry> = Vec::with_capacity(cell.num_ions);

    for ion_idx in 0..cell.num_ions {
        let pos = cell.ionic_positions.row(ion_idx);
        let (rx, ry, rz) = (pos[0], pos[1], pos[2]);

        // Fortran layout: iz fastest, then iy, then ix — matches Array3 F-order iteration.
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
/// Algorithm per ion I:
/// 1. Compute ω^I_{nm} on CPU (n_expanded ~18, cheap)
/// 2. H2D ω^I
/// 3. Allocate tmp[n_fine_grid]: tmp[g] = Σ_{nm} ω_{nm} · Q_{nm}(g) via gemv
///    (uses species-shared Q_{nm}(G) from cache)
/// 4. Element-wise multiply tmp[g] *= exp(-iG·R_I) (from ion_sf cache)
/// 5. Accumulate: ρ_aug(G) += tmp
/// After all ions: C2C inverse FFT, D2H, normalize.
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

    let mut rho_aug_g: CudaSlice<CudaComplex> = stream
        .alloc_zeros(n_fine_grid)
        .map_err(Error::Cuda)?;

    for ion_idx in 0..q_sf_cache.ion_sf.len() {
        let species_idx = q_sf_cache.ion_species[ion_idx];

        let species_entry = match q_sf_cache.species_entries.get(species_idx).and_then(|e| e.as_ref()) {
            Some(e) => e,
            None => continue,
        };

        let n_expanded = species_entry.n_expanded;
        let n_pairs = species_entry.n_pairs;
        let bp = &beta_psi_per_ion[ion_idx];

        // ω^I_{nm} on CPU (n_expanded ~18, O(ne² × n_bands) ≈ 18² × 160 = 52k ops)
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
        // Q stored as [n_pairs × n_fine_grid] row-major = [n_fine_grid × n_pairs] col-major.
        // gemv: tmp(n_fine_grid) = Q^T · ω  where Q is (n_pairs × n_fine_grid) col-major
        //   → trans=T, m=n_pairs, n=n_fine_grid
        let mut tmp: CudaSlice<CudaComplex> = stream.alloc_zeros(n_fine_grid).map_err(Error::Cuda)?;
        let one  = CudaComplex { x: 1.0, y: 0.0 };
        let zero = CudaComplex { x: 0.0, y: 0.0 };
        unsafe {
            blas.gemv_c64(
                op::T,
                n_pairs as i32,
                n_fine_grid as i32,
                one,
                &species_entry.q_nm,
                n_pairs as i32,
                &omega_dev,
                1,
                zero,
                &mut tmp,
                1,
            ).map_err(Error::Blas)?;
        }

        // tmp[g] *= exp(-iG·R_I)  (element-wise, using ion_sf cache)
        // Implemented as axpy-style: no dedicated kernel, use the multiply kernel
        // via a custom CUDA kernel or do it on CPU. Since we don't have a
        // pointwise-multiply kernel yet, do it via D2H → multiply → H2D.
        // This is a temporary fallback — a proper GPU kernel would avoid the roundtrip.
        let tmp_host: Vec<CudaComplex> = stream.clone_dtoh(&tmp).map_err(Error::Cuda)?;
        let sf_host: Vec<CudaComplex> = stream.clone_dtoh(&q_sf_cache.ion_sf[ion_idx].sf).map_err(Error::Cuda)?;
        let multiplied: Vec<CudaComplex> = tmp_host.iter().zip(sf_host.iter()).map(|(t, s)| {
            // (a + ib)(c + id) = (ac - bd) + i(ad + bc)
            CudaComplex {
                x: t.x * s.x - t.y * s.y,
                y: t.x * s.y + t.y * s.x,
            }
        }).collect();
        let mut tmp_sf: CudaSlice<CudaComplex> = stream.clone_htod(&multiplied).map_err(Error::Cuda)?;

        // ρ_aug(G) += tmp_sf
        blas.axpy_c64(n_fine_grid as i32, one, &tmp_sf, 1, &mut rho_aug_g, 1)
            .map_err(Error::Blas)?;
        drop(tmp_sf);
    }

    // C2C inverse FFT ρ_aug(G) → ρ_aug(r)
    let fft_plan = FftPlan3d::plan_c2c(ngx as i32, ngy as i32, ngz as i32, Arc::clone(stream))?;
    unsafe {
        let ptr = &mut rho_aug_g as *mut CudaSlice<CudaComplex>;
        fft_plan.c2c_inverse(&mut *ptr, &mut *ptr)?;
    }

    let rho_aug_host: Vec<CudaComplex> = stream.clone_dtoh(&rho_aug_g).map_err(Error::Cuda)?;
    pcie.record_d2h(&rho_aug_g);

    let inv_n = 1.0 / n_fine_grid as f64;
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
        QSfCache, QSfSpeciesEntry, IonSfEntry,
        build_q_sf_cache,
        compute_aug_density_fine,
        compute_aug_density_gpu,
    };
}
