use std::sync::Arc;

use chemrust_hamiltonian_core::augment::beta_phi::{
    compute_beta_g, expanded_projector_count, expanded_projector_lm,
};
use chemrust_hamiltonian_core::nlpot::build_d0_expanded;
use chemrust_hamiltonian_core::pseudopotential::HasAugmentationData;
use chemrust_hamiltonian_core::Pseudopotential;
use cudarc::driver::{CudaSlice, CudaStream};
use ndarray::Array2;
use num_complex::Complex64;

use crate::device::blas::{self, ZgemmConfig};
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::d_screening::WaveScreeningCache;
use crate::types::{Error, KPoint};

#[doc(hidden)]
pub struct VnlIonData {
    pub beta_g: CudaSlice<CudaComplex>,
    pub d_matrix: CudaSlice<CudaComplex>,
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
        kernels: &crate::eigensolver::chebyshev::CudaKernelSet,
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
        let mut per_ion_q_inv: Vec<Vec<f64>> = Vec::new();
        let mut per_ion_beta_flat: Vec<Vec<CudaComplex>> = Vec::new();
        let mut per_ion_ne: Vec<usize> = Vec::new();

        // Precompute V_eff FFT on CPU and upload to GPU for D-matrix screening.
        // The FFT stays on CPU (chemrust-hamiltonian) for now — only the result
        // is uploaded. Build the wave-grid screening cache (Q per species, SF per ion).
        let screening_h2d_start = pcie.h2d_bytes;
        let (v_eff_fft_dev, mut screening_cache): (Option<CudaSlice<CudaComplex>>, Option<WaveScreeningCache>) =
            v_eff_wave.and_then(|v_eff| {
                let fft = chemrust_hamiltonian_core::fft_forward_3d(v_eff.as_real_grid()).ok()?;
                let v_flat: Vec<CudaComplex> = fft.as_recip_array().iter()
                    .map(|&c| CudaComplex { x: c.re, y: c.im })
                    .collect();
                let v_dev = stream.clone_htod(&v_flat).ok()?;
                pcie.record_h2d(&v_dev);

                let cache = crate::eigensolver::d_screening::build_wave_screening_cache(
                    pots, cell, wave_grid, stream, pcie,
                ).ok()?;

                Some((v_dev, cache))
            }).unzip();
        let screening_h2d_bytes = pcie.h2d_bytes - screening_h2d_start;

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
            let n_wave = wave_grid.grid().iter().product::<usize>();

            // Compute screened D matrix: D = D0 + (1/N) · Re{Σ V_eff_fft · exp(+iG·R) · conj(Q)}
            let d_screened: Array2<f64> = match (&v_eff_fft_dev, screening_cache.as_mut()) {
                (Some(v_dev), Some(cache)) => {
                    crate::eigensolver::d_screening::screen_d_gpu(
                        cache, v_dev, ion_idx, species_idx,
                        d0_expanded.as_slice().expect("d0_expanded must be contiguous"),
                        n_wave, kernels, blas, stream,
                    ).unwrap_or(d0_expanded.clone())
                }
                _ => d0_expanded.clone(),
            };

            // Diagnostic: report D magnitudes per ion to catch screening explosions.
            let d_min = d_screened.iter().cloned().fold(f64::INFINITY, f64::min);
            let d_max = d_screened.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let d_amax = d_screened.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
            let d0_amax = d0_expanded.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
            eprintln!(
                "[D_screened] ion={ion_idx:2} sym={symbol} ne={n_expanded:2}  \
                 d0_amax={d0_amax:.4e}  d_screen_min={d_min:.4e} d_screen_max={d_max:.4e} d_screen_amax={d_amax:.4e}"
            );

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

            // Precompute S^{-1} Woodbury matrix: M^{-1} = (Q^{-1} + G)^{-1}
            // where G = beta_g^H · beta_g is the projector Gram matrix.
            let ne = n_expanded as usize;
            let beta_arr = beta_g; // &Array2<Complex64>, shape (ne, n_pw)
            let n_pw_local = beta_arr.shape()[1];

            // G[i,j] = Σ_g beta_i^*(g) · beta_j(g)  → real symmetric
            let mut gram = vec![0.0_f64; ne * ne];
            for i in 0..ne {
                for j in i..ne {
                    let mut s = 0.0_f64;
                    for g in 0..n_pw_local {
                        s += (beta_arr[[i, g]].conj() * beta_arr[[j, g]]).re;
                    }
                    gram[i * ne + j] = s;
                    gram[j * ne + i] = s;
                }
            }

            // M = Q^{-1} + G.  Compute Q^{-1} by direct inversion (ne ≤ 18).
            let eps_reg = 1e-12_f64;

            // First invert Q → q_inv using Gauss-Jordan on a copy.
            let mut q_inv = q_cpu.clone();
            // Augment with identity in-place using row operations
            let mut inv = vec![0.0_f64; ne * ne];
            for i in 0..ne { inv[i * ne + i] = 1.0; }
            for col in 0..ne {
                let mut pivot = col;
                for row in col..ne {
                    if q_inv[row * ne + col].abs() > q_inv[pivot * ne + col].abs() {
                        pivot = row;
                    }
                }
                if q_inv[pivot * ne + col].abs() < eps_reg {
                    // Singular column — Q has no contribution for this projector.
                    // Leave q_inv row as zero (effectively no 1/Q term for this channel).
                    // Zero out the corresponding row of inv.
                    for c in 0..ne { inv[col * ne + c] = 0.0; }
                    continue;
                }
                for c in 0..ne {
                    q_inv.swap(col * ne + c, pivot * ne + c);
                    inv.swap(col * ne + c, pivot * ne + c);
                }
                let piv_val = q_inv[col * ne + col];
                for c in 0..ne {
                    q_inv[col * ne + c] /= piv_val;
                    inv[col * ne + c] /= piv_val;
                }
                for row in 0..ne {
                    if row == col { continue; }
                    let factor = q_inv[row * ne + col];
                    if factor.abs() < eps_reg { continue; }
                    for c in 0..ne {
                        q_inv[row * ne + c] -= factor * q_inv[col * ne + c];
                        inv[row * ne + c] -= factor * inv[col * ne + c];
                    }
                }
            }
            // q_inv no longer needed; inv now holds Q^{-1} (or pseudo-inverse for
            // singular rows).

            // M = Q^{-1} + G
            let mut m_mat = gram.clone();
            for i in 0..ne {
                for j in 0..ne {
                    m_mat[i * ne + j] += inv[i * ne + j];
                }
            }

            // Save per-ion Q⁻¹ and beta for global Woodbury assembly.
            per_ion_q_inv.push(inv);
            let beta_this_ion = beta_flat.clone();
            per_ion_beta_flat.push(beta_this_ion);
            per_ion_ne.push(ne);

            entries.push(VnlIonData {
                beta_g: beta_dev,
                d_matrix: d_dev,
                q_matrix: q_dev,
                n_expanded,
            });
        }
        // -----------------------------------------------------------------------
        // Global Woodbury assembly (after per-ion loop)
        // -----------------------------------------------------------------------
        let n_total_expanded: i32 = per_ion_ne.iter().map(|&ne| ne as i32).sum();

        // 1. Block-diagonal Q⁻¹ as complex (Q is real, so y = 0.0).
        let nte = n_total_expanded as usize;
        let mut q_inv_blkdiag = vec![CudaComplex { x: 0.0, y: 0.0 }; nte * nte];
        let mut offset = 0;
        for (ion_idx, &ne) in per_ion_ne.iter().enumerate() {
            let inv = &per_ion_q_inv[ion_idx];
            for i in 0..ne {
                for j in 0..ne {
                    q_inv_blkdiag[(offset + i) * nte + (offset + j)].x = inv[i * ne + j];
                }
            }
            offset += ne;
        }

        // 2. Concatenate B: shape n_pw × n_total_expanded (col-major).
        let mut b_concat_cpu: Vec<CudaComplex> = Vec::with_capacity(n_pw * nte);
        for (ion_idx, _ne) in per_ion_ne.iter().enumerate() {
            let beta_flat = &per_ion_beta_flat[ion_idx];
            // beta_flat is ne × n_pw in row-major = n_pw × ne in col-major (lda = n_pw).
            // Concatenate along the column axis → append all ne*n_pw elements.
            b_concat_cpu.extend_from_slice(beta_flat);
        }
        let b_concat = stream.clone_htod(&b_concat_cpu).map_err(Error::Cuda)?;
        pcie.h2d_bytes += b_concat_cpu.len() * std::mem::size_of::<CudaComplex>();

        // 3. Compute B^H·B on GPU, keep full complex (cross-ion blocks have
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

        // 4. M = Q⁻¹ + B^H·B + ε·I  (complex, in-place on m_cpu).
        //    Q⁻¹ is real (im part stays 0 from step 1).
        for i in 0..nte {
            for j in 0..nte {
                m_cpu[i * nte + j].x += q_inv_blkdiag[i * nte + j].x;
                // imag stays from B^H·B (no Q⁻¹ imag contribution)
            }
            // Vestigial regularisation prevents exact-zero pivot from zgetrf.
            m_cpu[i * nte + i].x += 1e-12;
        }

        // 5. LU factor M = P·L·U.
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
            b_concat,
            lu_m: lu_m_dev,
            lu_ipiv: lu_ipiv_dev,
            n_total_expanded,
        })
    }
}
