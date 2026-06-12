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
use faer::linalg::solvers::{DenseSolveCore, Llt};
use faer::mat::Mat;
use faer::Side;
use num_complex::Complex64;

use crate::device::blas::{self, ZgemmConfig};
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::types::{Error, KPoint};

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

/// GPU-resident batch V_NL data with metadata for PCI-E tracking.
pub struct VnlBatchData {
    pub entries: Vec<VnlIonData>,
    /// H2D bytes uploaded for GPU D-matrix screening (V_eff FFT + Q cache + SF).
    /// Used by the PCI-E accounting assertion in the hot path.
    pub screening_h2d_bytes: usize,
    /// GPU screening cache (Q matrices + structure factors), built once at init
    /// and reused for D-matrix re-screening every SCF step with the current V_eff.
    pub screening_cache: Option<WaveScreeningCache>,
    /// GPU screening cache on the fine grid, built once at init and used for
    /// D-matrix re-screening with fine-grid V_eff (via rescreen_d).
    pub screening_cache_fine: Option<WaveScreeningCache>,
    /// Concatenated β-projectors: n_pw × n_total_expanded (col-major).
    pub b_concat: CudaSlice<CudaComplex>,
    /// LU factor (P·L·U) of M = Q⁻¹ + B^H·B (n_total_expanded × n_total_expanded).
    pub lu_m: CudaSlice<CudaComplex>,
    /// Pivot indices from LU factorisation (Fortran 1-based).
    pub lu_ipiv: CudaSlice<i32>,
    /// Sum of all per-ion n_expanded values.
    pub n_total_expanded: i32,
}

/// Build the expanded USPP Q augmentation matrix (n_expanded × n_expanded).
///
/// Mirrors [`build_d0_expanded`] but reads from `aug.q_aug()` instead of
/// `aug.d_zero()`. The Q matrix is used by Rayleigh-Ritz to build the
/// correct S-overlap for the generalized eigenvalue problem.
fn build_q_expanded(aug: &dyn HasAugmentationData) -> Vec<f64> {
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
        solver: &SolverHandle,
    ) -> Result<Self, Error> {
        Self::precompute_with_d_override(
            pw_coords, pots, cell, wave_grid, None, k_point,
            psi_data, n_bands, n_pw, _occupations, v_eff_wave,
            None,
            stream, pcie, blas, kernels, solver,
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
        stream: &Arc<CudaStream>,
        pcie: &mut PcieAccount,
        blas: &crate::device::blas::BlasHandle,
        kernels: &CudaKernelSet,
        solver: &SolverHandle,
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
        // Collectors for the global Woodbury assembly (after the per-ion loop).
        let mut per_ion_q: Vec<Vec<f64>> = Vec::new();
        let mut per_ion_beta_flat: Vec<Vec<CudaComplex>> = Vec::new();
        let mut per_ion_ne: Vec<usize> = Vec::new();

        // FFT V_eff once (CPU) → upload to GPU for D-matrix screening.
        let v_eff_fft = v_eff_wave.and_then(|v_eff| {
            chemrust_hamiltonian_core::fft_forward_3d(v_eff.as_real_grid()).ok()
        });

        let [ngz, ngy, ngx] = wave_grid.grid();
        let n_wave_grid = ngz * ngy * ngx;

        // Snapshot pcie before screening-related H2D so we can report the budget.
        let pcie_before_screening = pcie.h2d_bytes;

        // Build GPU screening cache (Q arrays + structure factors). Always built
        // so it is available for D-matrix re-screening every SCF step.
        let screening_cache: Option<WaveScreeningCache> =
            Some(build_wave_screening_cache(pots, cell, wave_grid, stream, pcie)?);

        // Build fine-grid screening cache when a fine grid is provided (FFI path).
        // rescreen_d uses this to avoid grid-mismatch truncation when processing
        // fine-grid V_eff that has more FFT points than the wave grid.
        let screening_cache_fine: Option<WaveScreeningCache> = match fine_grid {
            Some(fg) => Some(build_wave_screening_cache(pots, cell, fg, stream, pcie)?),
            None => None,
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

        let screening_h2d_bytes: usize = pcie.h2d_bytes - pcie_before_screening;

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
            let gmax_pp = pot.gmax();
            let beta_g = compute_beta_g(&wave_block, aug, cell, ion_idx, wave_grid, gmax_pp, k_cart)
                .map_err(|_| Error::Nvrtc(format!("compute_beta_g failed for ion {ion_idx}")))?;
            let d0_expanded = build_d0_expanded(aug);
            let n_expanded = beta_g.shape()[0] as i32;

            // Compute screened D matrix: D = D0 + (1/N)·Re(Σ_G V_eff(G)·exp(+iG·R)·conj(Q_nm(G)))
            // GPU path via screen_d_gpu; CPU reference is compute_screened_d_from_fft.
            //
            // T-prime override path: when `d_override[ion_idx]` is `Some`, replace
            // our computed D_screened with externally-supplied values (e.g. parsed
            // from CASTEP's `D_band_debug.dat`) to discriminate whether the SCF
            // cascade is D-driven or eigensolver-rotation-driven.
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
                None => match (&screening_cache, &v_eff_fft_dev) {
                    (Some(cache), Some(fft_dev)) => {
                        let d0_flat: Vec<f64> = d0_expanded.iter().cloned().collect();
                        screen_d_gpu(
                            cache, fft_dev, ion_idx, species_idx,
                            &d0_flat, n_wave_grid,
                            kernels, blas, stream,
                        )?
                    }
                    _ => d0_expanded.clone(),
                },
            };

            // Diagnostic: report D magnitudes per ion to catch screening explosions.
            // NOTE: when V_eff is unavailable at init time (cold-start), this prints
            // bare unscreened d0 values — the d_screen_amax == d0_amax you see here
            // is from the fallback d0_expanded.clone().  The actual screened D is
            // computed later in rescreen_d() when V_eff is available, and the values
            // here are NOT what the Davidson solver uses.
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
                let n_g = beta_g.nrows();
                let np = beta_g.ncols();
                let mut beta_norms: Vec<f64> = Vec::with_capacity(np);
                for n in 0..np {
                    let mut nrm2 = 0.0f64;
                    for g in 0..n_g {
                        let v = beta_g[[g, n]];
                        nrm2 += v.re * v.re + v.im * v.im;
                    }
                    beta_norms.push(nrm2.sqrt());
                }
                let min_bn = beta_norms.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_bn = beta_norms.iter().cloned().fold(0.0f64, f64::max);
                eprintln!(
                    "[Diag-beta] ion={ion_idx:2} sym={symbol} ne={np}:  \
                     |beta| min={min_bn:.4e} max={max_bn:.4e}  n_wave_grid={n_g}",
                );
            }

            let beta_flat: Vec<CudaComplex> =
                beta_g.iter().map(|&c| crate::device::complex_to_cuda(c)).collect();
            let d_flat: Vec<CudaComplex> =
                d_screened.iter().map(|&d| CudaComplex { x: d, y: 0.0 }).collect();

            let beta_dev = stream.clone_htod(&beta_flat).map_err(Error::Cuda)?;
            let d_dev = stream.clone_htod(&d_flat).map_err(Error::Cuda)?;
            pcie.h2d_bytes += (beta_flat.len() + d_flat.len()) * std::mem::size_of::<CudaComplex>();

            // Build and upload the expanded Q augmentation matrix (same indexing
            // convention as D0, read from aug.q_aug()).
            let q_cpu = build_q_expanded(aug);
            let q_flat: Vec<CudaComplex> =
                q_cpu.iter().map(|&q| CudaComplex { x: q, y: 0.0 }).collect();
            let q_dev = stream.clone_htod(&q_flat).map_err(Error::Cuda)?;
            pcie.h2d_bytes += q_flat.len() * std::mem::size_of::<CudaComplex>();

            // Precompute Woodbury components for global S^{-1} assembly.
            // Q (real, block-diagonal) saved for global Cholesky inversion;
            // beta saved for B^H·B Gram (ZGEMM on GPU, preserves cross-ion imag).
            let ne = n_expanded as usize;

            // Save per-ion Q matrix for global Woodbury assembly.
            // (Q is block-diagonal across ions; we build the full global
            // block-diagonal Q below and invert it in one Cholesky pass,
            // matching CASTEP nlpot_prepare_Sinv_full lines 11166-11193.)
            per_ion_q.push(q_cpu);
            let beta_this_ion = beta_flat.clone();
            per_ion_beta_flat.push(beta_this_ion);
            per_ion_ne.push(ne);

            let d0_flat: Vec<f64> = d0_expanded.iter().cloned().collect();
            entries.push(VnlIonData {
                beta_g: beta_dev,
                d_matrix: d_dev,
                d0_expanded: d0_flat,
                q_matrix: q_dev,
                n_expanded,
            });
        }
        // -----------------------------------------------------------------------
        // Global Woodbury assembly (after per-ion loop)
        // -----------------------------------------------------------------------
        let n_total_expanded: i32 = per_ion_ne.iter().map(|&ne| ne as i32).sum();

        // 1. Build global block-diagonal Q matrix (real, n_expanded per ion).
        //    CASTEP nlpot_prepare_Sinv_full lines 11166-11193: Q is block-diagonal
        //    with `ps_q(m,n,nsp)` entries multiplied by `mixture_weight` (≈1.0).
        //    We assemble the full matrix and invert it globally in one Cholesky pass.
        let nte = n_total_expanded as usize;
        let mut q_global = vec![0.0_f64; nte * nte];
        {
            let mut offset = 0;
            for (ion_idx, &ne) in per_ion_ne.iter().enumerate() {
                let q = &per_ion_q[ion_idx];
                for i in 0..ne {
                    for j in 0..ne {
                        q_global[(offset + i) * nte + (offset + j)] = q[i * ne + j];
                    }
                }
                offset += ne;
            }
        }

        // 2. Invert Q globally via Cholesky (faer), matching CASTEP's
        //    `invert_q_matrix` from preconditioner.rs with TINY = 1e-14.
        //    Rows/cols with |diag| ≤ TINY are zeroed (pseudo-inverse for
        //    singular Q channels; cf. nlpot.f90:13920-13932).
        const Q_TINY: f64 = 1e-14;
        let nonsingular: Vec<usize> =
            (0..nte).filter(|&i| q_global[i * nte + i].abs() > Q_TINY).collect();
        let mut q_inv_global = vec![0.0_f64; nte * nte];
        let m = nonsingular.len();
        if m > 0 {
            let mut sub_q = Mat::<f64>::zeros(m, m);
            for (ki, &i) in nonsingular.iter().enumerate() {
                for (kj, &j) in nonsingular.iter().enumerate() {
                    sub_q[(ki, kj)] = q_global[i * nte + j];
                }
            }
            if let Ok(llt) = Llt::new(sub_q.as_ref(), Side::Lower) {
                let sub_inv = llt.inverse();
                for (ki, &i) in nonsingular.iter().enumerate() {
                    for (kj, &j) in nonsingular.iter().enumerate() {
                        q_inv_global[i * nte + j] = sub_inv[(ki, kj)];
                    }
                }
            }
            // If Cholesky fails (not SPD), leave singular rows/cols zeroed
            // (safe fallback — the LU below handles the resulting M).
        }

        // 3. Concatenate B: shape n_pw × n_total_expanded (col-major).
        let mut b_concat_cpu: Vec<CudaComplex> = Vec::with_capacity(n_pw * nte);
        for (ion_idx, _ne) in per_ion_ne.iter().enumerate() {
            let beta_flat = &per_ion_beta_flat[ion_idx];
            b_concat_cpu.extend_from_slice(beta_flat);
        }
        let b_concat = stream.clone_htod(&b_concat_cpu).map_err(Error::Cuda)?;
        pcie.h2d_bytes += b_concat_cpu.len() * std::mem::size_of::<CudaComplex>();

        // 4. Compute B^H·B on GPU, keep full complex (cross-ion blocks have
        //    non-zero imaginary parts from structure-factor phase differences).
        let mut bh_b_dev: CudaSlice<CudaComplex> =
            stream.alloc_zeros(nte * nte).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::C,
                    transb: blas::op::N,
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
                &mut bh_b_dev,
            )?;
        }
        // D2H: copy full complex B^H·B to CPU for M assembly.
        let mut m_cpu: Vec<CudaComplex> = stream.clone_dtoh(&bh_b_dev).map_err(Error::Cuda)?;

        // 5. M = Q⁻¹ + B^H·B + ε·I  (complex, in-place on m_cpu).
        //    Q⁻¹ is real; imaginary parts come solely from B^H·B cross-ion phases.
        for i in 0..nte {
            for j in 0..nte {
                m_cpu[i * nte + j].x += q_inv_global[i * nte + j];
            }
            // Regularisation prevents exact-zero pivot from zgetrf.
            m_cpu[i * nte + i].x += 1e-12;
        }

        // 6. LU factor M = P·L·U.
        let mut lu_m_dev = stream.clone_htod(&m_cpu).map_err(Error::Cuda)?;
        pcie.h2d_bytes += m_cpu.len() * std::mem::size_of::<CudaComplex>();
        let mut lu_ipiv_dev = stream.alloc_zeros::<i32>(nte).map_err(Error::Cuda)?;
        let mut lu_info = stream.alloc_zeros::<i32>(1).map_err(Error::Cuda)?;
        solver.zgetrf(
            nte as i32,
            nte as i32,
            &mut lu_m_dev,
            &mut lu_ipiv_dev,
            &mut lu_info,
        )?;
        let lu_info_cpu: Vec<i32> = stream.clone_dtoh(&lu_info).map_err(Error::Cuda)?;
        if lu_info_cpu[0] != 0 {
            return Err(Error::Nvrtc(format!(
                "global Woodbury M singular: zgetrf info = {} (zero pivot at row {})",
                lu_info_cpu[0], lu_info_cpu[0],
            )));
        }

        Ok(VnlBatchData {
            entries,
            screening_h2d_bytes,
            screening_cache,
            screening_cache_fine,
            b_concat,
            lu_m: lu_m_dev,
            lu_ipiv: lu_ipiv_dev,
            n_total_expanded,
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
        if let Some(ref fine_cache) = self.screening_cache_fine {
            let [ngz_f, ngy_f, ngx_f] = fine_cache.wave_grid;
            n_grid = ngz_f * ngy_f * ngx_f;
            use_cache = fine_cache;
        } else if let Some(ref c) = self.screening_cache {
            let [ngz, ngy, ngx] = c.wave_grid;
            n_grid = ngz * ngy * ngx;
            use_cache = c;
        } else {
            return Ok(());
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
