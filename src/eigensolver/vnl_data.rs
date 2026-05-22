use std::collections::HashMap;
use std::sync::Arc;

use chemrust_hamiltonian_core::augment::beta_phi::{
    compute_beta_g, expanded_projector_count, expanded_projector_lm,
};
use chemrust_hamiltonian_core::nlpot::build_d0_expanded;
use chemrust_hamiltonian_core::pseudopotential::HasAugmentationData;
use chemrust_hamiltonian_core::Pseudopotential;
use cudarc::driver::{CudaSlice, CudaStream};
use num_complex::Complex64;

use crate::device::pcie::PcieAccount;
use crate::device::CudaComplex;
use crate::types::{Error, KPoint};

#[doc(hidden)]
pub struct VnlIonData {
    pub beta_g: CudaSlice<CudaComplex>,
    pub d_matrix: CudaSlice<CudaComplex>,
    /// Expanded USPP Q augmentation matrix (n_expanded × n_expanded).
    pub q_matrix: CudaSlice<CudaComplex>,
    /// `(Q^{-1} + G)^{-1}` where `G = beta_g^H · beta_g` is the projector
    /// Gram matrix for this ion.  Precomputed on CPU and uploaded to GPU.
    /// Used to apply `S^{-1}` via the Woodbury formula in the Chebyshev filter:
    ///   `S^{-1} ψ = ψ - beta_g · s_inv_mat · (beta_g^H · ψ)`
    pub s_inv_mat: CudaSlice<CudaComplex>,
    pub n_expanded: i32,
}

#[doc(hidden)]
pub struct VnlBatchData {
    pub entries: Vec<VnlIonData>,
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

        // Precompute Q-on-grid once per species and FFT V_eff once total.
        // For Cu111_CO: 18 Cu ions × 171 Q-pairs × 437k grid points = ~1.3B ops
        // if done per-ion. Caching reduces Q to one call per species and FFT to one call.
        let v_eff_fft = v_eff_wave.and_then(|v_eff| {
            chemrust_hamiltonian_core::fft_forward_3d(v_eff.as_real_grid()).ok()
        });

        let q_on_grid_cache: HashMap<String, Option<chemrust_hamiltonian_core::QOnGrid>> =
            if v_eff_fft.is_some() {
                cell.species_symbols.iter().filter_map(|symbol| {
                    let pot = pots.get(symbol)?;
                    let aug: &dyn HasAugmentationData = match pot {
                        Pseudopotential::Usp(d) => d,
                        _ => return None,
                    };
                    let q = chemrust_hamiltonian_core::precompute_q_on_grid(aug, wave_grid).ok();
                    Some((symbol.clone(), q))
                }).collect()
            } else {
                HashMap::new()
            };

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

            // Compute screened D matrix: D = D0 + ∫ Q(r)·V_eff(r) dr
            let d_screened = match (&v_eff_fft, q_on_grid_cache.get(symbol).and_then(|o| o.as_ref())) {
                (Some(fft), Some(q_on_grid)) => {
                    chemrust_hamiltonian_core::compute_screened_d_from_fft(
                        q_on_grid, fft, cell, ion_idx, wave_grid, &d0_expanded,
                    )
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

            // Invert M → s_inv_mat using the same Gauss-Jordan.
            let mut m_inv = m_mat.clone();
            // Augment with identity
            let mut s_inv = vec![0.0_f64; ne * ne];
            for i in 0..ne { s_inv[i * ne + i] = 1.0; }
            for col in 0..ne {
                let mut pivot = col;
                for row in col..ne {
                    if m_inv[row * ne + col].abs() > m_inv[pivot * ne + col].abs() {
                        pivot = row;
                    }
                }
                if m_inv[pivot * ne + col].abs() < eps_reg {
                    continue;
                }
                for c in 0..ne {
                    m_inv.swap(col * ne + c, pivot * ne + c);
                    s_inv.swap(col * ne + c, pivot * ne + c);
                }
                let piv_val = m_inv[col * ne + col];
                for c in 0..ne {
                    m_inv[col * ne + c] /= piv_val;
                    s_inv[col * ne + c] /= piv_val;
                }
                for row in 0..ne {
                    if row == col { continue; }
                    let factor = m_inv[row * ne + col];
                    if factor.abs() < eps_reg { continue; }
                    for c in 0..ne {
                        m_inv[row * ne + c] -= factor * m_inv[col * ne + c];
                        s_inv[row * ne + c] -= factor * s_inv[col * ne + c];
                    }
                }
            }
            // s_inv now holds M^{-1}

            let s_inv_flat: Vec<CudaComplex> = m_inv
                .iter()
                .map(|&x| CudaComplex { x, y: 0.0 })
                .collect();
            let s_inv_dev = stream.clone_htod(&s_inv_flat).map_err(Error::Cuda)?;
            pcie.h2d_bytes += s_inv_flat.len() * std::mem::size_of::<CudaComplex>();

            entries.push(VnlIonData {
                beta_g: beta_dev,
                d_matrix: d_dev,
                q_matrix: q_dev,
                s_inv_mat: s_inv_dev,
                n_expanded,
            });
        }
        Ok(VnlBatchData { entries })
    }
}
