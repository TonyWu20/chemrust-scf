use std::sync::Arc;

use chemrust_hamiltonian_core::augment::beta_phi::compute_beta_g;
use chemrust_hamiltonian_core::nlpot::build_d0_expanded;
use chemrust_hamiltonian_core::pseudopotential::HasAugmentationData;
use chemrust_hamiltonian_core::Pseudopotential;
use cudarc::driver::{CudaSlice, CudaStream};
use num_complex::Complex64;

use crate::device::pcie::PcieAccount;
use crate::device::CudaComplex;
use crate::types::{Error, KPoint};

pub(crate) struct VnlIonData {
    pub beta_g: CudaSlice<CudaComplex>,
    pub d_matrix: CudaSlice<CudaComplex>,
    pub n_expanded: i32,
}

pub(crate) struct VnlBatchData {
    pub entries: Vec<VnlIonData>,
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
        occupations: Option<&[f64]>,
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
            let d_mat = build_d0_expanded(aug);
            let n_expanded = beta_g.shape()[0] as i32;

            // Compute screened D matrix if occupations are provided.
            // D_screened = D0 + Σ_b occ[b] · (β^H · ψ_b) · (β^H · ψ_b)^H
            // where (β^H · ψ_b)[i] = Σ_g conj(beta_g[i][g]) · ψ_b[g]
            let d_screened: Vec<f64> = if let Some(occ) = occupations {
                let mut screened = d_mat.clone();
                // beta_g is (n_expanded, n_pw) row-major
                let beta_slice = beta_g.as_slice().unwrap();
                for b in 0..n_bands {
                    let occ_b = occ[b];
                    if occ_b.abs() < 1e-15 {
                        continue;
                    }
                    // C_proj[i] = Σ_g conj(beta_g[i][g]) · psi_data[b * n_pw + g]
                    let mut c_proj = vec![Complex64::new(0.0, 0.0); n_expanded as usize];
                    for i in 0..n_expanded as usize {
                        let mut sum = Complex64::new(0.0, 0.0);
                        let row_offset = i * n_pw;
                        for g in 0..n_pw {
                            sum += beta_slice[row_offset + g].conj()
                                * psi_data[b * n_pw + g];
                        }
                        c_proj[i] = sum;
                    }
                    // D_screen[i][j] += occ_b · c_proj[i] · conj(c_proj[j])
                    for i in 0..n_expanded as usize {
                        for j in 0..n_expanded as usize {
                            screened[[i, j]] += occ_b * (c_proj[i] * c_proj[j].conj()).re;
                        }
                    }
                }
                screened.iter().copied().collect()
            } else {
                d_mat.iter().copied().collect()
            };

            let beta_flat: Vec<CudaComplex> =
                beta_g.iter().map(|&c| crate::device::complex_to_cuda(c)).collect();
            let d_flat: Vec<CudaComplex> =
                d_screened.iter().map(|&d| CudaComplex { x: d, y: 0.0 }).collect();

            let beta_dev = stream.clone_htod(&beta_flat).map_err(Error::Cuda)?;
            let d_dev = stream.clone_htod(&d_flat).map_err(Error::Cuda)?;
            pcie.h2d_bytes += (beta_flat.len() + d_flat.len()) * std::mem::size_of::<CudaComplex>();

            entries.push(VnlIonData {
                beta_g: beta_dev,
                d_matrix: d_dev,
                n_expanded,
            });
        }
        Ok(VnlBatchData { entries })
    }
}
