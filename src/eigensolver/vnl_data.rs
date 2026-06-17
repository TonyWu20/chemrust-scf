use std::sync::Arc;

use chemrust_hamiltonian_core::augment::beta_phi::{
    compute_beta_g, expanded_projector_count, expanded_projector_lm,
};
use chemrust_hamiltonian_core::nlpot::build_d0_expanded;
use crate::eigensolver::d_screening::{build_wave_screening_cache, screen_d_gpu, WaveScreeningCache};
use crate::eigensolver::kernels::CudaKernelSet;
use chemrust_hamiltonian_core::pseudopotential::HasAugmentationData;
use chemrust_hamiltonian_core::Pseudopotential;
use cudarc::driver::{CudaSlice, CudaStream};
use num_complex::Complex64;

use crate::device::pcie::PcieAccount;
use crate::device::CudaComplex;
use crate::types::{Error, KPoint};

// ---------------------------------------------------------------------------
// HandleSharedVnl — kpt-independent data shared across ALL kpts and spins
// ---------------------------------------------------------------------------

/// K-point-INDEPENDENT V_NL data shared across ALL k-points and ALL spin
/// channels.  Built once at init / first SCF iteration.
///
/// Screening caches (Q(G) + structure factors) depend only on pseudopotentials
/// and grid geometry — not on k-point or V_eff.  Per-ion Q and D0 matrices
/// similarly depend only on the pseudopotential.  Lifting these to the handle
/// level avoids duplicating them per k-point (e.g. 14× for 14-kpt NiO).
pub struct HandleSharedVnl {
    pub screening_cache: WaveScreeningCache,
    pub screening_cache_fine: Option<WaveScreeningCache>,
    /// Per-ion USPP Q augmentation matrices on GPU (n_expanded × n_expanded).
    pub per_ion_q: Vec<CudaSlice<CudaComplex>>,
    /// Per-ion unscreened D0 matrices (CPU).
    pub per_ion_d0_expanded: Vec<Vec<f64>>,
    /// Per-ion expanded projector count.
    pub per_ion_n_expanded: Vec<i32>,
    /// H2D bytes from the fresh build.
    pub screening_h2d_bytes: usize,
}

// ---------------------------------------------------------------------------
// KptSharedVnl — per-kpt data shared across spin channels
// ---------------------------------------------------------------------------

/// Per-KPOINT V_NL data shared across spin channels.
///
/// Owns per-ion β(G+k) projector arrays (kpt-dependent via G+k) and holds an
/// Arc to the handle-level kpt-independent state.
pub struct KptSharedVnl {
    /// Per-ion β(G+k) projector arrays on GPU. Index `[ion_idx]`.
    pub per_ion_beta_g: Vec<CudaSlice<CudaComplex>>,
    /// Kpt-independent shared state (screening caches, Q, D0).
    pub handle: Arc<HandleSharedVnl>,
}

#[doc(hidden)]
pub struct VnlIonData {
    pub beta_g: CudaSlice<CudaComplex>,
    pub d_matrix: CudaSlice<CudaComplex>,
    /// Raw D0 (unscreened) used for re-screening each SCF step.
    pub d0_expanded: Vec<f64>,
    /// Expanded USPP Q augmentation matrix (n_expanded × n_expanded).
    pub q_matrix: CudaSlice<CudaComplex>,
    pub n_expanded: i32,
}

/// GPU-resident batch V_NL data with shared spin-independent state via Arc.
pub struct VnlBatchData {
    pub entries: Vec<VnlIonData>,
    /// Spin-independent shared state (screening caches, beta_g, Q, D0).
    pub shared: Arc<KptSharedVnl>,
    /// Concatenated beta-projectors for Woodbury S^{-1} (n_pw × n_total_expanded).
    #[cfg(feature = "chebyshev")]
    pub b_concat: CudaSlice<CudaComplex>,
    /// LU-factorized M matrix for Woodbury S^{-1} (nte × nte complex).
    #[cfg(feature = "chebyshev")]
    pub lu_m: CudaSlice<CudaComplex>,
    /// LU pivot indices for Woodbury S^{-1} (nte × i32).
    #[cfg(feature = "chebyshev")]
    pub lu_ipiv: CudaSlice<i32>,
    /// Total expanded projector count sum(per_ion_n_expanded).
    #[cfg(feature = "chebyshev")]
    pub n_total_expanded: i32,
    /// Per-ion Q^{-1} matrices (flat row-major, n_expanded × n_expanded per ion).
    /// Used for exact residual computation in stationary Woodbury iteration.
    #[cfg(feature = "chebyshev")]
    pub q_inv_per_ion: Vec<Vec<f64>>,
    /// Per-ion Q matrices (flat row-major, n_expanded × n_expanded per ion).
    /// Used as block-diagonal preconditioner Q in stationary iteration
    /// (ABINIT's D⁻¹ = Q). Missing from the earlier LU-based approach.
    #[cfg(feature = "chebyshev")]
    pub q_per_ion: Vec<Vec<f64>>,
    /// B^H·B Gram matrix (nte × nte, real-symmetric, flat row-major f64).
    /// Used for exact residual computation in stationary Woodbury iteration.
    #[cfg(feature = "chebyshev")]
    pub bhb_cpu: Vec<f64>,
}

/// Build the expanded USPP Q augmentation matrix (n_expanded × n_expanded).
///
/// Mirrors [`build_d0_expanded`] but reads from `aug.q_aug()` instead of
/// `aug.d_zero()`. The Q matrix is used by Rayleigh-Ritz to build the
/// correct S-overlap for the generalized eigenvalue problem.
pub fn build_q_expanded(aug: &dyn HasAugmentationData) -> Vec<f64> {
    let projs = aug.projectors();
    let n_exp = expanded_projector_count(projs);
    if n_exp == 0 || projs.is_empty() {
        return vec![];
    }
    let q_rows = aug.q_aug();

    // 1-based within-l-channel count for each radial index
    let within_l: Vec<usize> = projs
        .iter()
        .enumerate()
        .map(|(i, _)| projs[..=i].iter().filter(|p| p.l == projs[i].l).count())
        .collect();

    let q_lookup = |n_rad: usize, m_rad: usize| -> f64 {
        if n_rad >= projs.len() || m_rad >= projs.len() {
            return 0.0;
        }
        if projs[n_rad].l != projs[m_rad].l {
            return 0.0;
        }
        let cnt_n = within_l[n_rad];
        let cnt_m = within_l[m_rad];
        let (canon, smaller_cnt) = if cnt_n >= cnt_m {
            (n_rad, cnt_m)
        } else {
            (m_rad, cnt_n)
        };
        q_rows
            .0
            .get(canon)
            .and_then(|r| r.0.get(smaller_cnt.saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
    };

    let mut q = vec![0.0_f64; n_exp * n_exp];
    for n_e in 0..n_exp {
        let pn = expanded_projector_lm(projs, n_e);
        for m_e in 0..n_exp {
            let pm = expanded_projector_lm(projs, m_e);
            if pn.l == pm.l && pn.m == pm.m {
                q[n_e * n_exp + m_e] = q_lookup(pn.rad_idx, pm.rad_idx);
            }
        }
    }
    q
}

/// Build kpt-independent shared VNL state (screening caches + per-ion Q/D0).
///
/// Call once per run (not per kpt).  The returned Arc is shared across all
/// k-points and spin channels via `HandleSharedVnl`.
pub fn build_handle_shared_vnl(
    pots: &chemrust_hamiltonian_core::PseudopotentialSet,
    cell: &chemrust_hamiltonian_core::CellGeometry,
    wave_grid: &chemrust_hamiltonian_core::GVectorGrid,
    fine_grid: Option<&chemrust_hamiltonian_core::GVectorGrid>,
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<Arc<HandleSharedVnl>, Error> {
    let sc = build_wave_screening_cache(pots, cell, wave_grid, stream, pcie)?;
    let sc_fine = match fine_grid {
        Some(fg) => Some(build_wave_screening_cache(pots, cell, fg, stream, pcie)?),
        None => None,
    };
    let mut per_ion_q = Vec::new();
    let mut per_ion_d0 = Vec::new();
    let mut per_ion_ne = Vec::new();
    for ion_idx in 0..cell.num_ions {
        let species_idx = cell.ion_species[ion_idx];
        let symbol = &cell.species_symbols[species_idx];
        if let Some(pot) = pots.get(symbol) {
            if let Pseudopotential::Usp(aug) = pot {
                let n_exp = expanded_projector_count(aug.projectors()) as i32;
                let q_cpu = build_q_expanded(aug);
                let q_flat: Vec<CudaComplex> = q_cpu.iter()
                    .map(|&q| CudaComplex { x: q, y: 0.0 }).collect();
                let q_dev = stream.clone_htod(&q_flat).map_err(Error::Cuda)?;
                pcie.h2d_bytes += q_flat.len() * std::mem::size_of::<CudaComplex>();
                let d0 = build_d0_expanded(aug);
                let d0_flat: Vec<f64> = d0.iter().cloned().collect();
                per_ion_q.push(q_dev);
                per_ion_d0.push(d0_flat);
                per_ion_ne.push(n_exp);
            }
        }
    }
    Ok(Arc::new(HandleSharedVnl {
        screening_cache: sc,
        screening_cache_fine: sc_fine,
        per_ion_q,
        per_ion_d0_expanded: per_ion_d0,
        per_ion_n_expanded: per_ion_ne,
        screening_h2d_bytes: pcie.h2d_bytes,
    }))
}

impl VnlBatchData {
    #[allow(clippy::too_many_arguments)]
    pub fn precompute(
        pw_coords: &[[i32; 3]],
        pots: &chemrust_hamiltonian_core::PseudopotentialSet,
        cell: &chemrust_hamiltonian_core::CellGeometry,
        wave_grid: &chemrust_hamiltonian_core::GVectorGrid,
        k_point: &KPoint,
        psi_data: &[Complex64],
        n_bands: usize,
        n_pw: usize,
        _occupations: Option<&[f64]>,
        v_eff_wave: Option<&chemrust_hamiltonian_core::EffectivePotential>,
        stream: &Arc<CudaStream>,
        pcie: &mut PcieAccount,
        blas: &crate::device::blas::BlasHandle,
        kernels: &CudaKernelSet,
    ) -> Result<Self, Error> {
        Self::precompute_with_d_override(
            pw_coords, pots, cell, wave_grid, None, k_point,
            psi_data, n_bands, n_pw, _occupations, v_eff_wave,
            None,      // d_override
            None,      // shared
            None,      // handle_shared — test path, builds fresh
            None,      // solver_thunk — test path, no Woodbury
            stream, pcie, blas, kernels,
        )
    }

    /// Test-only / scf_diag variant of [`precompute`] that allows injecting
    /// externally-supplied per-ion D matrices in place of the screened D
    /// computed from V_eff. Used by the T-prime discriminator test
    /// (`iter2_band0_with_castep_d_injection`) to determine whether the
    /// SCF cascade is D-driven or eigensolver-rotation-driven.
    ///
    /// `d_override`: optional slice of length `cell.num_ions`. Entry `i`
    /// contains an `Option<Vec<f64>>` — when `Some`, the inner Vec is a
    /// flat row-major (n_expanded × n_expanded) D matrix that replaces
    /// our computed screened D for ion `i`; when `None`, the per-ion D
    /// is computed normally from V_eff. Passing `None` for the outer
    /// `Option` is equivalent to calling [`precompute`].
    ///
    /// `shared`: optional `Arc<KptSharedVnl>` to reuse spin-independent data
    /// across spin channels. When `Some`, screening caches, beta_g, and Q
    /// matrices are cloned from the shared Arc (shallow CudaSlice clone, no
    /// VRAM cost). When `None`, everything is built fresh and a new
    /// `KptSharedVnl` is stored in the returned `VnlBatchData.shared`.
    #[allow(clippy::too_many_arguments)]
    pub fn precompute_with_d_override(
        pw_coords: &[[i32; 3]],
        pots: &chemrust_hamiltonian_core::PseudopotentialSet,
        cell: &chemrust_hamiltonian_core::CellGeometry,
        wave_grid: &chemrust_hamiltonian_core::GVectorGrid,
        fine_grid: Option<&chemrust_hamiltonian_core::GVectorGrid>,
        k_point: &KPoint,
        psi_data: &[Complex64],
        n_bands: usize,
        n_pw: usize,
        _occupations: Option<&[f64]>,
        v_eff_wave: Option<&chemrust_hamiltonian_core::EffectivePotential>,
        d_override: Option<&[Option<Vec<f64>>]>,
        shared: Option<Arc<KptSharedVnl>>,
        // When `Some`, screening caches and per-ion Q/D0 are taken from the
        // handle instead of being built fresh.  Must be provided when `shared`
        // is `None` (first kpt build); ignored when `shared` is `Some`.
        handle_shared: Option<Arc<HandleSharedVnl>>,
        solver_thunk: Option<&crate::device::solver::SolverHandle>,
        stream: &Arc<CudaStream>,
        pcie: &mut PcieAccount,
        blas: &crate::device::blas::BlasHandle,
        kernels: &CudaKernelSet,
    ) -> Result<Self, Error> {
        let kf = k_point.coords;
        let recip = cell.recip_lattice.as_array();
        let mut k_cart = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                k_cart[j] += kf[i] * recip[i][j];
            }
        }

        // Build bands as Vec<Vec<Complex64>> for KptWaveBlock
        let bands: Vec<Vec<Complex64>> = (0..n_bands)
            .map(|b| psi_data[b * n_pw..(b + 1) * n_pw].to_vec())
            .collect();

        let wave_block = chemrust_hamiltonian_core::types::KptWaveBlock {
            coords: kf,
            nplw: n_pw,
            pw_grid_coord: pw_coords.to_vec(),
            bands,
        };

        let mut entries = Vec::new();

        // FFT V_eff once (CPU) → upload to GPU for D-matrix screening.
        let v_eff_fft = v_eff_wave.and_then(|v_eff| {
            chemrust_hamiltonian_core::fft_forward_3d(v_eff.as_real_grid()).ok()
        });

        let [ngz, ngy, ngx] = wave_grid.grid();
        let n_wave_grid = ngz * ngy * ngx;

        // -----------------------------------------------------------------------
        // Build or reuse the shared spin-independent state
        // -----------------------------------------------------------------------
        // Screening caches are kpt-independent — prefer handle, fall back to
        // fresh build (backward compat for callers that don't have a handle).
        let (screening_cache, screening_cache_fine, screening_h2d_bytes) = match shared {
            Some(ref shared_arc) => {
                // Reuse from shared KptSharedVnl (which delegates to HandleSharedVnl).
                (shared_arc.handle.screening_cache.clone(),
                 shared_arc.handle.screening_cache_fine.clone(), 0)
            }
            None => match handle_shared {
                Some(ref h) => {
                    (h.screening_cache.clone(), h.screening_cache_fine.clone(), 0)
                }
                None => {
                    let pcie_before_screening = pcie.h2d_bytes;
                    let sc = build_wave_screening_cache(pots, cell, wave_grid, stream, pcie)?;
                    let sc_fine = match fine_grid {
                        Some(fg) => Some(build_wave_screening_cache(pots, cell, fg, stream, pcie)?),
                        None => None,
                };
                let h2d = pcie.h2d_bytes - pcie_before_screening;

                (sc, sc_fine, h2d)
                }
            }
        };

        // Upload V_eff_fft to GPU (Fortran order via .t().iter()).
        let v_eff_fft_dev: Option<CudaSlice<CudaComplex>> = match &v_eff_fft {
            Some(fft) => {
                let v_eff_flat: Vec<CudaComplex> = fft.as_recip_array()
                    .t()
                    .iter()
                    .map(|c| CudaComplex { x: c.re, y: c.im })
                    .collect();
                let dev = stream.clone_htod(&v_eff_flat).map_err(Error::Cuda)?;
                pcie.record_h2d(&dev);
                Some(dev)
            }
            None => None,
        };

        // Collectors for the shared Arc (populated only in fresh-build path).
        let mut per_ion_beta_g: Vec<CudaSlice<CudaComplex>> = Vec::new();
        let mut per_ion_q: Vec<CudaSlice<CudaComplex>> = Vec::new();
        let mut per_ion_d0_expanded: Vec<Vec<f64>> = Vec::new();
        let mut per_ion_n_expanded: Vec<i32> = Vec::new();

        for ion_idx in 0..cell.num_ions {
            let species_idx = cell.ion_species[ion_idx];
            let symbol = &cell.species_symbols[species_idx];
            let pot = pots.get(symbol).ok_or_else(|| {
                Error::Nvrtc(format!("missing pseudopotential for species {symbol}"))
            })?;
            let aug: &dyn HasAugmentationData = match pot {
                Pseudopotential::Usp(d) => d,
                _ => continue,
            };

            // Spin-independent per-ion data: beta_g, q_matrix, d0_expanded, n_expanded.
            // Either built fresh or cloned from the shared Arc.
            // Spin-independent per-ion data: beta_g (kpt-dependent), q/d0 (kpt-independent).
            // When `shared` is Some, reuse from existing KptSharedVnl.
            // When `shared` is None but `handle_shared` is Some, use handle for q/d0
            // and build fresh beta_g.
            let (beta_dev, q_dev, d0_expanded, n_expanded) = match shared {
                Some(ref shared_arc) => {
                    debug_assert!(
                        entries.len() < shared_arc.per_ion_beta_g.len(),
                        "shared per_ion vectors out of sync with entries"
                    );
                    let handle = &shared_arc.handle;
                    let bg = shared_arc.per_ion_beta_g[entries.len()].clone();
                    let qm = handle.per_ion_q[entries.len()].clone();
                    let d0 = handle.per_ion_d0_expanded[entries.len()].clone();
                    let ne = handle.per_ion_n_expanded[entries.len()];
                    (bg, qm, d0, ne)
                }
                None => {
                    let beta_g = compute_beta_g(
                        &wave_block, aug, cell, ion_idx, wave_grid, pot.gmax(), k_cart,
                    ).map_err(|_| Error::Nvrtc(
                        format!("compute_beta_g failed for ion {ion_idx}")
                    ))?;
                    let n_expanded = beta_g.shape()[0] as i32;
                    let d0_expanded = build_d0_expanded(aug);

                    // Upload beta_g to GPU (always kpt-dependent).
                    let beta_flat: Vec<CudaComplex> =
                        beta_g.iter().map(|&c| crate::device::complex_to_cuda(c)).collect();
                    let beta_dev = stream.clone_htod(&beta_flat).map_err(Error::Cuda)?;
                    pcie.h2d_bytes += beta_flat.len() * std::mem::size_of::<CudaComplex>();

                    // Q and D0 are kpt-independent — take from handle if available.
                    let (q_dev, d0_flat) = match handle_shared {
                        Some(ref h) => {
                            let qm = h.per_ion_q[entries.len()].clone();
                            let d0 = h.per_ion_d0_expanded[entries.len()].clone();
                            (qm, d0)
                        }
                        None => {
                            let q_cpu = build_q_expanded(aug);
                            let q_flat: Vec<CudaComplex> =
                                q_cpu.iter().map(|&q| CudaComplex { x: q, y: 0.0 }).collect();
                            let q_dev = stream.clone_htod(&q_flat).map_err(Error::Cuda)?;
                            pcie.h2d_bytes += q_flat.len() * std::mem::size_of::<CudaComplex>();
                            let d0_flat: Vec<f64> = d0_expanded.iter().cloned().collect();
                            (q_dev, d0_flat)
                        }
                    };

                    // Collect for KptSharedVnl / HandleSharedVnl.
                    per_ion_beta_g.push(beta_dev.clone());
                    if handle_shared.is_none() {
                        // Fresh build: collect Q and D0 for HandleSharedVnl construction.
                        per_ion_q.push(q_dev.clone());
                        per_ion_d0_expanded.push(d0_flat.clone());
                        per_ion_n_expanded.push(n_expanded);
                    }

                    (beta_dev, q_dev, d0_flat, n_expanded)
                }
            };

            // Compute screened D matrix: D = D0 + (1/N)·Re(Σ_G V_eff(G)·exp(+iG·R)·conj(Q_nm(G)))
            // GPU path via screen_d_gpu; CPU reference is compute_screened_d_from_fft.
            //
            // T-prime override path: when `d_override[ion_idx]` is `Some`, replace
            // our computed D_screened with externally-supplied values (e.g. parsed
            // from CASTEP's `D_band_debug.dat`) to discriminate whether the SCF
            // cascade is D-driven or eigensolver-rotation-driven.
            //
            // Note: d0_expanded is Vec<f64> in both paths (shared clone or fresh
            // after flattening build_d0_expanded's Array2). We reconstruct Array2
            // for the fallback arm to match screen_d_gpu's return type.
            let d_screened = match d_override.and_then(|o| o.get(ion_idx).and_then(|d| d.as_ref())) {
                Some(d_inj) => {
                    let n_exp = n_expanded as usize;
                    if d_inj.len() != n_exp * n_exp {
                        return Err(Error::Nvrtc(format!(
                            "d_override for ion {ion_idx} has length {} but expected {n_exp}×{n_exp} = {}",
                            d_inj.len(), n_exp * n_exp,
                        )));
                    }
                    ndarray::Array2::from_shape_vec((n_exp, n_exp), d_inj.clone())
                        .map_err(|e| Error::Nvrtc(format!(
                            "d_override reshape failed for ion {ion_idx}: {e}"
                        )))?
                }
                None => match v_eff_fft_dev.as_ref() {
                    Some(fft_dev) => screen_d_gpu(
                        &screening_cache, fft_dev, ion_idx, species_idx,
                        &d0_expanded, n_wave_grid,
                        kernels, blas, stream,
                    )?,
                    None => {
                        // No V_eff available: construct Array2 from flat Vec<f64>
                        let ne = (d0_expanded.len() as f64).sqrt() as usize;
                        ndarray::Array2::from_shape_vec((ne, ne), d0_expanded.clone())
                            .map_err(|e| Error::Nvrtc(format!(
                                "d0 reshape failed for ion {ion_idx}: {e}"
                            )))?
                    }
                },
            };

            // Diagnostic: report D magnitudes per ion to catch screening explosions.
            #[cfg(feature = "scf_diag")]
            {
                let d_min = d_screened.iter().cloned().fold(f64::INFINITY, f64::min);
                let d_max = d_screened.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let d_amax = d_screened.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
                let d0_amax = d0_expanded.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
                eprintln!(
                    "[D_screened] ion={ion_idx:2} sym={symbol} ne={n_expanded:2}  \
                     d0_amax={d0_amax:.4e}  d_screen_min={d_min:.4e} d_screen_max={d_max:.4e} d_screen_amax={d_amax:.4e}  \
                     (init, pre-rescreen)"
                );
            }

            // Diag: β-projector L2 norms per ion/projector
            #[cfg(feature = "scf_diag")]
            {
                // Beta norms diagnostic — skipped in shared path since we
                // don't have the CPU-side beta_g matrix available.
                if shared.is_none() {
                    let n_g = n_pw; // approximate
                    let np = n_expanded as usize;
                    eprintln!(
                        "[Diag-beta] ion={ion_idx:2} sym={symbol} ne={np}:  \
                         n_wave_grid={n_g}",
                    );
                }
            }

            let d_flat: Vec<CudaComplex> =
                d_screened.iter().map(|&d| CudaComplex { x: d, y: 0.0 }).collect();
            let d_dev = stream.clone_htod(&d_flat).map_err(Error::Cuda)?;

            entries.push(VnlIonData {
                beta_g: beta_dev,
                d_matrix: d_dev,
                d0_expanded,
                q_matrix: q_dev,
                n_expanded,
            });
        }

        // -----------------------------------------------------------------------
        // Construct the shared Arc (or clone the existing one)
        // Build or reuse KptSharedVnl.
        let shared_arc = match shared {
            Some(ref arc) => arc.clone(),
            None => {
                let handle = match handle_shared {
                    Some(ref h) => h.clone(),
                    None => {
                        // Backward compat: build HandleSharedVnl from fresh data.
                        Arc::new(HandleSharedVnl {
                            screening_cache,
                            screening_cache_fine,
                            per_ion_q,
                            per_ion_d0_expanded,
                            per_ion_n_expanded,
                            screening_h2d_bytes,
                        })
                    }
                };
                Arc::new(KptSharedVnl {
                    per_ion_beta_g,
                    handle,
                })
            }
        };

        // -----------------------------------------------------------------------
        // Woodbury S^{-1} assembly (Chebyshev feature only)
        // -----------------------------------------------------------------------
        // Builds the concatenated B matrix and LU-factored M = Q^{-1} + B^H·B + eps·I
        // used by apply_s_inverse for the global USPP S^{-1} operator.
        #[cfg(feature = "chebyshev")]
        let (b_concat, lu_m, lu_ipiv, n_total_expanded, q_inv_per_ion, q_per_ion, bhb_cpu) = if let Some(solver) = solver_thunk {
            let n_total_expanded: i32 = entries.iter().map(|e| e.n_expanded).sum();
            let nte = n_total_expanded as usize;

            if nte == 0 {
                // No USPP ions: Woodbury = identity
                let empty = stream.alloc_zeros::<CudaComplex>(0).map_err(Error::Cuda)?;
                let empty_ipiv = stream.alloc_zeros::<i32>(0).map_err(Error::Cuda)?;
                (empty.clone(), empty, empty_ipiv, 0, Vec::new(), Vec::new(), Vec::new())
            } else {
                // Step 1: Concatenate per-ion beta_g into b_concat (n_pw × nte)
                let mut b_concat_data: Vec<CudaComplex> = Vec::with_capacity(n_pw * nte);
                for entry in &entries {
                    let _ne = entry.n_expanded as usize;
                    let beta_host: Vec<CudaComplex> = stream
                        .clone_dtoh(&entry.beta_g)
                        .map_err(Error::Cuda)?;
                    // beta_g is (n_expanded × n_pw) in row-major, but we need
                    // (n_pw × n_total_expanded) with each ion's block contiguous.
                    // b_concat layout: [ion0_col0(n_pw), ion0_col1(n_pw), ...,
                    //                    ion1_col0(n_pw), ...]
                    // beta_g layout: [ne][n_pw] row-major = same as b_concat blocks.
                    b_concat_data.extend(beta_host.iter());
                }
                let b_concat = stream.clone_htod(&b_concat_data).map_err(Error::Cuda)?;

                // Step 2: B^H·B on GPU (nte × nte)
                // Technically this is B^H·B where B is n_pw × nte,
                // so result = B^H·B is nte × nte.
                let mut bhb: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(nte * nte).map_err(Error::Cuda)?;
                unsafe {
                    blas.gemm_c64(
                        crate::device::blas::ZgemmConfig {
                            transa: crate::device::blas::op::C,
                            transb: crate::device::blas::op::N,
                            m: nte as i32,
                            n: nte as i32,
                            k: n_pw as i32,
                            alpha: CudaComplex { x: 1.0, y: 0.0 },
                            lda: n_pw as i32,
                            ldb: n_pw as i32,
                            beta: CudaComplex { x: 0.0, y: 0.0 },
                            ldc: nte as i32,
                        },
                        &b_concat,
                        &b_concat,
                        &mut bhb,
                    )?;
                }

                // Step 3: Build M = Q^{-1} + B^H·B + eps·I on CPU
                // First, rebuild per-ion Q matrices CPU side and invert them.
                // Q matrices are real-symmetric, n_expanded × n_expanded.
                const EPS: f64 = 1e-15;
                let mut m_cpu: Vec<f64> = vec![0.0; nte * nte];

                // Compute Q^{-1} per ion and assemble block-diagonal Q_inv into m_cpu.
                // Also add B^H·B (downloaded from GPU) and eps·I.
                // Q matrices are real-valued; build_q_expanded returns Vec<f64>.
                // We rebuild from the augmentation data stored in each ion's species.
                let mut offset = 0;
                let mut q_inv_per_ion: Vec<Vec<f64>> = Vec::new();
                let mut q_per_ion: Vec<Vec<f64>> = Vec::new();
                for ion_idx in 0..cell.num_ions {
                    let species_idx = cell.ion_species[ion_idx];
                    let symbol = &cell.species_symbols[species_idx];
                    if let Some(pot) = pots.get(symbol) {
                        if let Pseudopotential::Usp(aug) = pot {
                            let q_cpu = build_q_expanded(aug);
                            let ne = expanded_projector_count(aug.projectors());
                            if ne == 0 { continue; }
                            // Save Q and Q⁻¹ for stationary Woodbury iteration.
                            // Q = block-diagonal preconditioner (ABINIT's D⁻¹)
                            // Q⁻¹ = exact operator for residual computation
                            q_per_ion.push(q_cpu.clone());
                            // Invert Q (small matrix, Gaussian elimination with partial pivot)
                            let q_inv = invert_small_real_matrix(&q_cpu, ne);
                            q_inv_per_ion.push(q_inv.clone());
                            for i in 0..ne {
                                for j in 0..ne {
                                    m_cpu[(offset + i) * nte + (offset + j)] += q_inv[i * ne + j];
                                }
                            }
                            offset += ne;
                        }
                    }
                }

                // Download B^H·B and add to m_cpu
                let bhb_complex: Vec<CudaComplex> = stream
                    .clone_dtoh(&bhb)
                    .map_err(Error::Cuda)?;
                // Save real part of bhb for iterative refinement
                let bhb_cpu_real: Vec<f64> = bhb_complex.iter().map(|c| c.x).collect();
                for i in 0..nte {
                    for j in 0..nte {
                        m_cpu[i * nte + j] += bhb_complex[i * nte + j].x;
                    }
                }

                // Add eps·I on diagonal
                for i in 0..nte {
                    m_cpu[i * nte + i] += EPS;
                }

                // Step 4: Upload M to GPU, call zgetrf for LU factorization
                let m_flat: Vec<CudaComplex> = m_cpu.iter()
                    .map(|&v| CudaComplex { x: v, y: 0.0 })
                    .collect();
                let mut lu_m = stream.clone_htod(&m_flat).map_err(Error::Cuda)?;
                let mut lu_ipiv = stream.alloc_zeros::<i32>(nte).map_err(Error::Cuda)?;
                let mut info_dev = stream.alloc_zeros::<i32>(1).map_err(Error::Cuda)?;
                solver.zgetrf(
                    nte as i32,
                    nte as i32,
                    &mut lu_m,
                    &mut lu_ipiv,
                    &mut info_dev,
                )?;

                (b_concat, lu_m, lu_ipiv, n_total_expanded, q_inv_per_ion, q_per_ion, bhb_cpu_real)
            }
        } else {
            // Davidson / test path: no Woodbury, empty dummy fields
            let empty = stream.alloc_zeros::<CudaComplex>(0).map_err(Error::Cuda)?;
            let empty_ipiv = stream.alloc_zeros::<i32>(0).map_err(Error::Cuda)?;
            (empty.clone(), empty, empty_ipiv, 0, Vec::new(), Vec::new(), Vec::new())
        };

        Ok(VnlBatchData {
            entries,
            shared: shared_arc,
            #[cfg(feature = "chebyshev")]
            b_concat,
            #[cfg(feature = "chebyshev")]
            lu_m,
            #[cfg(feature = "chebyshev")]
            lu_ipiv,
            #[cfg(feature = "chebyshev")]
            n_total_expanded,
            #[cfg(feature = "chebyshev")]
            q_inv_per_ion,
            #[cfg(feature = "chebyshev")]
            q_per_ion,
            #[cfg(feature = "chebyshev")]
            bhb_cpu,
        })
    }

    /// Re-screen D matrices using the current V_eff (called each SCF step).
    ///
    /// FFTs V_eff to reciprocal space on CPU, uploads to GPU, and calls
    /// `screen_d_gpu` for each ion. Updates `d_matrix` on GPU in-place.
    pub fn rescreen_d(
        &mut self,
        v_eff_real: &ndarray::Array3<f64>,
        stream: &Arc<CudaStream>,
        kernels: &CudaKernelSet,
        blas: &crate::device::blas::BlasHandle,
    ) -> Result<(), Error> {
        // Determine cache and grid: use fine-grid cache when available
        // (rescreen_d is called with fine-grid V_eff from CASTEP/FFI path),
        // otherwise fall back to wave-grid cache (pure Rust SCF path).
        let n_grid;
        let use_cache;
        if let Some(ref fine_cache) = self.shared.handle.screening_cache_fine {
            let [ngz_f, ngy_f, ngx_f] = fine_cache.wave_grid;
            n_grid = ngz_f * ngy_f * ngx_f;
            use_cache = fine_cache;
        } else {
            let [ngz, ngy, ngx] = self.shared.handle.screening_cache.wave_grid;
            n_grid = ngz * ngy * ngx;
            use_cache = &self.shared.handle.screening_cache;
        };

        // FFT V_eff real → reciprocal on CPU.
        let real_grid = chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_real.clone());
        let v_eff_fft = chemrust_hamiltonian_core::fft_forward_3d(&real_grid)
            .map_err(|e| Error::Nvrtc(format!("V_eff FFT failed: {e}")))?;

        // Upload to GPU (Fortran order).
        let v_eff_flat: Vec<CudaComplex> = v_eff_fft.as_recip_array()
            .t()
            .iter()
            .map(|c| CudaComplex { x: c.re, y: c.im })
            .collect();
        let v_eff_fft_dev = stream.clone_htod(&v_eff_flat).map_err(Error::Cuda)?;

        // Re-screen each ion.
        for (ion_idx, entry) in self.entries.iter_mut().enumerate() {
            let species_idx = use_cache.ion_species[ion_idx];
            let d_screened = screen_d_gpu(
                use_cache,
                &v_eff_fft_dev,
                ion_idx,
                species_idx,
                &entry.d0_expanded,
                n_grid,
                kernels,
                blas,
                stream,
            )?;
            // Upload new D matrix to GPU, replacing old one.
            let d_flat: Vec<CudaComplex> =
                d_screened.iter().map(|&d| CudaComplex { x: d, y: 0.0 }).collect();
            let d_dev = stream.clone_htod(&d_flat).map_err(Error::Cuda)?;
            entry.d_matrix = d_dev;
        }

        Ok(())
    }
}

/// Invert a small real matrix via Gaussian elimination with partial pivoting.
///
/// `mat` is a flat row-major `n × n` matrix. Returns `n × n` flat row-major inverse.
/// This is only used for Q matrices (n_expanded × n_expanded, typically 2-6),
/// so performance is not a concern.
fn invert_small_real_matrix(mat: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    // Augmented matrix [A | I]
    let mut aug = vec![0.0_f64; n * n * 2];
    for i in 0..n {
        for j in 0..n {
            aug[i * (2 * n) + j] = mat[i * n + j];
        }
        aug[i * (2 * n) + n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (2 * n) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (2 * n) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            // Near-singular: return identity (Q should be SPD, so this shouldn't happen)
            let mut identity = vec![0.0; n * n];
            for i in 0..n {
                identity[i * n + i] = 1.0;
            }
            return identity;
        }
        if max_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * (2 * n) + j, max_row * (2 * n) + j);
            }
        }
        let pivot = aug[col * (2 * n) + col];
        // Normalize pivot row
        for j in 0..(2 * n) {
            aug[col * (2 * n) + j] /= pivot;
        }
        // Eliminate other rows
        for row in 0..n {
            if row != col {
                let factor = aug[row * (2 * n) + col];
                for j in 0..(2 * n) {
                    aug[row * (2 * n) + j] -= factor * aug[col * (2 * n) + j];
                }
            }
        }
    }

    // Extract inverse (right half of augmented matrix)
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * (2 * n) + n + j];
        }
    }
    inv
}
