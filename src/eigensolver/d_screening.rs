// ---------------------------------------------------------------------------
// GPU D-matrix screening for USPP V_NL.
//
// D_nm = D0_nm + (1/N) · Re{ Σ_G V_eff_fft(G) · exp(+iG·R) · conj(Q_nm(G)) }
//
// Algorithm:
//   1. w = V_eff_fft · conj(ion_sf)     via cpx_conj_mul kernel (grid-parallel)
//   2. tmp = Q^H · w                     via cuBLAS gemv (single launch)
//   3. D2H tmp (few KB), finalize D on CPU → upload D matrix
//
// The Q matrix is stored per species: Q_flat[p * n_wave + g] for p = 0..n_lower_pairs.
// Interpreted as col-major (n_wave × n_lower_pairs) with lda = n_wave.

use std::sync::Arc;

use chemrust_hamiltonian_core::{
    CellGeometry, GVectorGrid, PseudopotentialSet,
    nlpot::precompute_q_on_grid,
    pseudopotential::{Pseudopotential, HasAugmentationData},
};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array2;

use crate::device::{blas::{BlasHandle, op}, CudaComplex, pcie::PcieAccount};
use crate::eigensolver::kernels::CudaKernelSet;
use crate::types::Error;

// ---------------------------------------------------------------------------
// Per-species wave-grid Q cache
// ---------------------------------------------------------------------------

/// Q_nm(G) for one species on the wave grid, lower-triangle pairs only.
pub struct WaveQSpeciesEntry {
    /// Flattened Q_nm(G): [n_lower_pairs × n_wave_grid], pair-major, grid-minor.
    pub q_nm: CudaSlice<CudaComplex>,
    /// Number of expanded projectors (Σ(2l+1)) for this species.
    pub n_expanded: usize,
    /// Number of lower-triangle (n,m) pairs stored.
    pub n_lower_pairs: usize,
    /// Map pair index → (n, m) expanded indices.
    pub pair_indices: Vec<(usize, usize)>,
}

/// Structure factor exp(-iG·R_I) for one ion on the wave grid.
pub struct WaveSfEntry {
    pub sf: CudaSlice<CudaComplex>,
}

/// GPU cache for D-matrix screening on the wave grid.
///
/// Built fresh inside VnlBatchData::precompute, replacing the old
/// HashMap<String, Option<QOnGrid>> local cache with a GPU-resident
/// data structure.
pub struct WaveScreeningCache {
    pub species_entries: Vec<Option<WaveQSpeciesEntry>>,
    pub ion_sf: Vec<WaveSfEntry>,
    pub ion_species: Vec<usize>,
    pub wave_grid: [usize; 3],
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build the WaveScreeningCache for GPU D-matrix screening.
///
/// Precomputes Q_nm(G) once per species on CPU (via `precompute_q_on_grid`),
/// flattens to pair-major layout, and uploads to GPU. Structure factors
/// are computed per ion.
#[allow(clippy::too_many_arguments)]
pub fn build_wave_screening_cache(
    pots: &PseudopotentialSet,
    cell: &CellGeometry,
    wave_grid: &GVectorGrid,
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<WaveScreeningCache, Error> {
    let [ngz, ngy, ngx] = wave_grid.grid();
    let n_wave_grid = ngz * ngy * ngx;
    let tau = 2.0 * std::f64::consts::PI;
    let n_species = cell.num_species;

    // --- Per-species Q on wave grid ---
    let mut species_entries: Vec<Option<WaveQSpeciesEntry>> = Vec::with_capacity(n_species);

    for species_idx in 0..n_species {
        let symbol = &cell.species_symbols[species_idx];
        let Some(pot) = pots.get(symbol) else {
            species_entries.push(None);
            continue;
        };
        let aug: &dyn HasAugmentationData = match pot {
            Pseudopotential::Usp(d) => d,
            _ => {
                species_entries.push(None);
                continue;
            }
        };

        // CPU precompute Q on wave grid (once per species, uses existing code)
        let q_on_grid = match precompute_q_on_grid(aug, wave_grid) {
            Ok(q) => q,
            Err(_) => {
                species_entries.push(None);
                continue;
            }
        };

        let n_lower_pairs = q_on_grid.pairs.len();
        let pair_indices: Vec<(usize, usize)> = q_on_grid.pairs
            .iter()
            .map(|((n, m), _)| (*n, *m))
            .collect();

        let n_expanded = pair_indices
            .iter()
            .flat_map(|&(n, m)| [n, m])
            .max()
            .map(|max_idx| max_idx + 1)
            .unwrap_or(0);

        // Flatten Q arrays into GPU buffer: pair-major, grid-minor.
        // .t() transposes so that .iter() traverses iz-fastest (Fortran
        // order), matching the structure factor and V_eff_fft convention.
        let mut q_flat: Vec<CudaComplex> = Vec::with_capacity(n_lower_pairs * n_wave_grid);
        for ((_n, _m), q_arr) in &q_on_grid.pairs {
            for &c in q_arr.t().iter() {
                q_flat.push(CudaComplex { x: c.re, y: c.im });
            }
        }

        let q_gpu = stream.clone_htod(&q_flat).map_err(Error::Cuda)?;
        pcie.record_h2d(&q_gpu);

        species_entries.push(Some(WaveQSpeciesEntry {
            q_nm: q_gpu,
            n_expanded,
            n_lower_pairs,
            pair_indices,
        }));
    }

    // --- Per-ion structure factors on wave grid ---
    let mut ion_sf: Vec<WaveSfEntry> = Vec::with_capacity(cell.num_ions);
    for ion_idx in 0..cell.num_ions {
        let pos = cell.ionic_positions.row(ion_idx);
        let (rx, ry, rz) = (pos[0], pos[1], pos[2]);

        let sf_host: Vec<CudaComplex> = {
            let n_wave = n_wave_grid;
            let mut v = Vec::with_capacity(n_wave);
            // Fortran order: iz fastest, matching GVectorGrid layout.
            for ix in 0..ngx {
                for iy in 0..ngy {
                    for iz in 0..ngz {
                        let gf = wave_grid.gvecs()[[iz, iy, ix]];
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
        ion_sf.push(WaveSfEntry { sf: sf_gpu });
    }

    Ok(WaveScreeningCache {
        species_entries,
        ion_sf,
        ion_species: cell.ion_species.clone(),
        wave_grid: [ngz, ngy, ngx],
    })
}

// ---------------------------------------------------------------------------
// GPU D-screening
// ---------------------------------------------------------------------------

/// Screen the D matrix for one ion using GPU kernels and cuBLAS gemv.
///
/// Returns the screened D matrix as an (n_expanded × n_expanded) real array,
/// ready for upload as VnlIonData.d_matrix.
#[allow(clippy::too_many_arguments)]
pub fn screen_d_gpu(
    cache: &WaveScreeningCache,
    v_eff_fft_dev: &CudaSlice<CudaComplex>,
    ion_idx: usize,
    species_idx: usize,
    d0_expanded: &[f64],
    n_wave_grid: usize,
    kernels: &CudaKernelSet,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<Array2<f64>, Error> {
    let species_entry = match cache.species_entries.get(species_idx).and_then(|e| e.as_ref()) {
        Some(e) => e,
        None => {
            // No Q data for this species — return D0 flat array as 2D.
            let ne = (d0_expanded.len() as f64).sqrt() as usize;
            let mut d = Array2::<f64>::zeros((ne, ne));
            for n in 0..ne {
                for m in 0..ne {
                    d[[n, m]] = d0_expanded[n * ne + m];
                }
            }
            return Ok(d);
        }
    };

    let n_expanded = species_entry.n_expanded;
    let n_lower_pairs = species_entry.n_lower_pairs;

    // Step 1: w[g] = V_eff_fft[g] * conj(ion_sf[g])
    // Uses the cpx_conj_mul kernel: dst = a * conj(b).
    let mut w: CudaSlice<CudaComplex> = stream.alloc_zeros(n_wave_grid).map_err(Error::Cuda)?;
    unsafe {
        stream
            .launch_builder(&kernels.cpx_conj_mul)
            .arg(&mut w)
            .arg(v_eff_fft_dev)
            .arg(&cache.ion_sf[ion_idx].sf)
            .arg(&(n_wave_grid as i32))
            .launch(LaunchConfig::for_num_elems(n_wave_grid as u32))
    }
    .map_err(Error::Cuda)?;

    // Step 2: tmp[p] = Σ_g conj(Q[p, g]) * w[g]
    // Q is stored pair-major: Q_flat[p * n_wave + g].
    // Interpreted as col-major matrix (n_wave × n_lower_pairs), lda = n_wave.
    // gemv with trans=C computes: y[p] = Σ_g conj(A[g,p]) * x[g]
    let mut tmp: CudaSlice<CudaComplex> = stream.alloc_zeros(n_lower_pairs).map_err(Error::Cuda)?;
    let one  = CudaComplex { x: 1.0, y: 0.0 };
    let zero = CudaComplex { x: 0.0, y: 0.0 };
    unsafe {
        blas.gemv_c64(
            op::C,
            n_wave_grid as i32,
            n_lower_pairs as i32,
            one,
            &species_entry.q_nm,
            n_wave_grid as i32,
            &w,
            1,
            zero,
            &mut tmp,
            1,
        )
        .map_err(Error::Blas)?;
    }

    // Step 3: D2H tmp (few KB), finalize D on CPU.
    let tmp_host: Vec<CudaComplex> = stream.clone_dtoh(&tmp).map_err(Error::Cuda)?;

    let inv_n = 1.0 / (n_wave_grid as f64);
    let mut d_screen = Array2::<f64>::zeros((n_expanded, n_expanded));

    for (p, &(n, m)) in species_entry.pair_indices.iter().enumerate() {
        let screening = tmp_host[p].x * inv_n;  // Re(tmp[p]) / N
        let val = d0_expanded[n * n_expanded + m] + screening;
        d_screen[[n, m]] = val;
        d_screen[[m, n]] = val;  // D is real symmetric
    }

    Ok(d_screen)
}

/// Debug variant of [`screen_d_gpu`]: returns intermediate `w` and `tmp` buffers
/// alongside the final D matrix for diagnostic comparison.
#[cfg(any(test, feature = "scf_diag"))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn screen_d_gpu_debug(
    cache: &WaveScreeningCache,
    v_eff_fft_dev: &CudaSlice<CudaComplex>,
    ion_idx: usize,
    species_idx: usize,
    d0_expanded: &[f64],
    n_wave_grid: usize,
    kernels: &CudaKernelSet,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(Array2<f64>, Vec<CudaComplex>, Vec<CudaComplex>), Error> {
    let species_entry = match cache.species_entries.get(species_idx).and_then(|e| e.as_ref()) {
        Some(e) => e,
        None => {
            let ne = (d0_expanded.len() as f64).sqrt() as usize;
            let mut d = Array2::<f64>::zeros((ne, ne));
            for n in 0..ne {
                for m in 0..ne {
                    d[[n, m]] = d0_expanded[n * ne + m];
                }
            }
            return Ok((d, vec![], vec![]));
        }
    };

    let n_expanded = species_entry.n_expanded;
    let n_lower_pairs = species_entry.n_lower_pairs;

    let mut w: CudaSlice<CudaComplex> = stream.alloc_zeros(n_wave_grid).map_err(Error::Cuda)?;
    unsafe {
        stream
            .launch_builder(&kernels.cpx_conj_mul)
            .arg(&mut w)
            .arg(v_eff_fft_dev)
            .arg(&cache.ion_sf[ion_idx].sf)
            .arg(&(n_wave_grid as i32))
            .launch(LaunchConfig::for_num_elems(n_wave_grid as u32))
    }
    .map_err(Error::Cuda)?;

    // D2H w for diagnostic
    let w_host: Vec<CudaComplex> = stream.clone_dtoh(&w).map_err(Error::Cuda)?;

    let mut tmp: CudaSlice<CudaComplex> = stream.alloc_zeros(n_lower_pairs).map_err(Error::Cuda)?;
    let one = CudaComplex { x: 1.0, y: 0.0 };
    let zero = CudaComplex { x: 0.0, y: 0.0 };
    unsafe {
        blas.gemv_c64(
            op::C,
            n_wave_grid as i32,
            n_lower_pairs as i32,
            one,
            &species_entry.q_nm,
            n_wave_grid as i32,
            &w,
            1,
            zero,
            &mut tmp,
            1,
        )
        .map_err(Error::Blas)?;
    }

    // D2H tmp for diagnostic
    let tmp_host: Vec<CudaComplex> = stream.clone_dtoh(&tmp).map_err(Error::Cuda)?;

    let inv_n = 1.0 / (n_wave_grid as f64);
    let mut d_screen = Array2::<f64>::zeros((n_expanded, n_expanded));

    for (p, &(n, m)) in species_entry.pair_indices.iter().enumerate() {
        let screening = tmp_host[p].x * inv_n;
        let val = d0_expanded[n * n_expanded + m] + screening;
        d_screen[[n, m]] = val;
        d_screen[[m, n]] = val;
    }

    Ok((d_screen, w_host, tmp_host))
}

/// Re-exports for integration tests that need GPU D-screening items.
#[cfg(any(test, feature = "scf_diag"))]
pub mod test_api {
    pub use super::{
        build_wave_screening_cache,
        screen_d_gpu,
        screen_d_gpu_debug,
        WaveQSpeciesEntry,
        WaveScreeningCache,
        WaveSfEntry,
    };
}
