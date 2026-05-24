// ---------------------------------------------------------------------------
// Rayleigh-Ritz subspace diagonalization for SCF eigensolver
// ---------------------------------------------------------------------------
//
// Given filtered wavefunctions psi (RowDistributed) and H|psi> (RowDistributed),
// solve the generalized eigenvalue problem in the subspace:
//   H_sub * X = lambda * S_sub * X
// where H_sub = psi^dag * H|psi> and S_sub = psi^dag * psi.
//
// Then rotate psi to the new eigenbasis and extract eigenvalues.

use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::cusolver::sys::{cublasFillMode_t, cusolverEigMode_t};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::device::blas::{op, ZgemmConfig};
use crate::device::blas::BlasHandle;
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::device::{CudaComplex, Gpu};
use crate::eigensolver::chebyshev::CudaKernelSet;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::layout::{ColumnDistributed, Cpu, RowDistributed, WavefunctionSet};
use crate::types::Error;

// ---------------------------------------------------------------------------
// Procrustes pin configuration (eigenvector rotation stabilization)
// ---------------------------------------------------------------------------

/// Pin mode for Procrustes-based eigenvector rotation stabilization.
///
/// The pin corrects for arbitrary in-block unitary rotations that ZHEGVD
/// produces in near-degenerate eigenvalue clusters. Two variants are
/// implemented and can be selected via environment variable `CHEMRUST_PIN_MODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinMode {
    /// No pinning; standard RR behavior (baseline for A/B testing).
    Off,
    /// Pre-RR pin: rotate ψ_row before assembling H_sub/S_sub, then re-run ZHEGVD.
    PreRr,
    /// Post-RR pin: apply rotation to X after ZHEGVD, before final ψ_new = ψ_row · X.
    PostRr,
}

/// Configuration for Procrustes pinning in Rayleigh-Ritz.
#[derive(Debug, Clone)]
pub struct RrPinConfig {
    /// Eigenvalue spacing threshold (Ha) for detecting degenerate blocks.
    /// Default 0.05 Ha (catches Cu-3d Δε ≈ 7 mHa and Fermi cluster Δε ≈ 0.3 mHa,
    /// and is loose enough to merge fragmented near-degenerate clusters whose
    /// boundaries jitter under Chebyshev filter pollution — empirically validated
    /// against `pin_preserves_castep_basis_at_iter1_postrr` where 0.01 Ha
    /// produced 1e-6 outliers from over-fragmentation).
    pub eps_degen: f64,
    /// Pin mode (Off, PreRr, or PostRr).
    pub mode: PinMode,
}

impl RrPinConfig {
    /// Create a new pin configuration from environment variables.
    ///
    /// `CHEMRUST_PIN_MODE` ∈ {`off`, `prerr`, `postrr`}; defaults to `off`.
    /// `CHEMRUST_PIN_EPS_DEGEN` is the eigenvalue spacing threshold in Ha; defaults to 0.05.
    pub fn from_env() -> Self {
        let mode_str = std::env::var("CHEMRUST_PIN_MODE").unwrap_or_else(|_| "off".to_string());
        let mode = match mode_str.to_lowercase().as_str() {
            "off" => PinMode::Off,
            "prerr" => PinMode::PreRr,
            "postrr" => PinMode::PostRr,
            _ => {
                eprintln!(
                    "CHEMRUST_PIN_MODE='{}' not recognized; defaulting to 'off'",
                    mode_str
                );
                PinMode::Off
            }
        };

        let eps_degen_str = std::env::var("CHEMRUST_PIN_EPS_DEGEN").unwrap_or_else(|_| "0.05".to_string());
        let eps_degen = eps_degen_str.parse::<f64>().unwrap_or(0.05);

        RrPinConfig { eps_degen, mode }
    }
}


// ---------------------------------------------------------------------------
// Type alias for the complex Rayleigh-Ritz return type
// ---------------------------------------------------------------------------

type RayleighRitzResult = Result<
    (
        Gpu<WavefunctionSet<ColumnDistributed>>,
        Cpu<Vec<f64>>,
        Vec<CudaSlice<CudaComplex>>,
    ),
    Error,
>;

// ---------------------------------------------------------------------------
// Helper functions for Procrustes pinning
// ---------------------------------------------------------------------------

/// Detect degenerate eigenvalue blocks based on eigenvalue spacing.
///
/// Returns a Vec of (lo, hi) pairs where each pair represents a contiguous
/// block of eigenvalues with consecutive spacing < eps_degen and block size ≥ 2.
///
/// # Arguments
/// - `eigenvalues`: sorted eigenvalues (ascending order)
/// - `eps_degen`: eigenvalue spacing threshold (Ha)
///
/// # Returns
/// Vec of (lo, hi) pairs where hi is exclusive (standard Rust range notation).
fn detect_degenerate_blocks(eigenvalues: &[f64], eps_degen: f64) -> Vec<(usize, usize)> {
    if eigenvalues.len() < 2 {
        return vec![];
    }

    let mut blocks = vec![];
    let mut block_start = 0;

    for i in 0..eigenvalues.len() - 1 {
        let spacing = (eigenvalues[i + 1] - eigenvalues[i]).abs();
        if spacing >= eps_degen {
            // End of a potential block
            if i > block_start {
                // Block has size >= 2
                blocks.push((block_start, i + 1));
            }
            block_start = i + 1;
        }
    }

    // Check the final block
    if eigenvalues.len() - 1 > block_start {
        blocks.push((block_start, eigenvalues.len()));
    }

    blocks
}

/// Solve the Rayleigh-Ritz generalized eigenvalue problem in the subspace.

///
/// Input:
/// - `psi_row`: filtered wavefunctions in RowDistributed layout (n_pw x n_bands)
/// - `hpsi_row`: H|psi> in RowDistributed layout (n_pw x n_bands)
/// - `vnl_data`: precomputed V_NL data (beta_g, D, Q matrices per ion)
/// - `prev_psi_dev`: (optional) previous iteration's ψ for Procrustes pinning
/// - `pin_cfg`: (optional) Procrustes pin configuration
///
/// The overlap matrix S_sub includes the USPP augmentation:
///   S_sub = psi^dag·psi  +  Σ_ion C_proj^dag · q · C_proj
/// where C_proj = beta_g^H · psi and q is the expanded Q augmentation matrix.
///
/// Output:
/// - `psi_col`: rotated wavefunctions in ColumnDistributed layout (n_bands x n_pw)
/// - `Cpu(eigenvalues)`: converged eigenvalues as a Vec<f64>
#[allow(clippy::too_many_arguments)]
pub(crate) fn rayleigh_ritz(
    psi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    hpsi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    vnl_data: &VnlBatchData,
    n_bands: usize,
    n_pw: usize,
    _kernels: &CudaKernelSet,
    pcie: &mut PcieAccount,
    solver: &SolverHandle,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
    prev_psi_dev: Option<&CudaSlice<CudaComplex>>,
    pin_cfg: Option<&RrPinConfig>,
) -> RayleighRitzResult {
    let n = n_bands as i32;
    let k = n_pw as i32;

    // ---- Step 1: H_sub = psi^dag * hpsi  (n_bands x n_bands) ----
    // psi_row is (n_pw x n_bands) col-major, so psi^dag is (n_bands x n_pw)
    // Using gemm with transa=C (conj-transpose psi), transb=N (hpsi as-is)
    // C(m,n) = alpha * op(A)(m,k) * op(B)(k,n) + beta * C(m,n)
    // H_sub(n_bands, n_bands) = conj(psi_row^T)(n_bands, n_pw) * hpsi_row(n_pw, n_bands)
    //                        = psi_dag * hpsi
    // psi_row: lda = n_pw (col-major, n_pw rows)
    // hpsi_row: ldb = n_pw
    // H_sub: ldc = n_bands
    let mut h_sub_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_bands).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,   // conj(psi^T)
                transb: op::N,   // hpsi
                m: n,            // rows = n_bands
                n,               // cols = n_bands
                k,               // inner dim = n_pw
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,          // lda of psi (n_pw rows, col-major)
                ldb: k,          // ldb of hpsi (n_pw rows, col-major)
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n,          // ldc of H_sub (n_bands rows, col-major)
            },
            psi_row.as_device_slice(),
            hpsi_row.as_device_slice(),
            &mut h_sub_dev,
        )?;
    }

    // ---- Step 2: S_sub = psi^dag * psi  (n_bands x n_bands) ----
    // Bare plane-wave overlap (missing USPP augmentation — added below).
    let mut s_sub_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_bands).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,   // conj(psi^T)
                transb: op::N,   // psi
                m: n,
                n,
                k,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,
                ldb: k,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n,
            },
            psi_row.as_device_slice(),
            psi_row.as_device_slice(),
            &mut s_sub_dev,
        )?;
    }

    // ---- Step 2b: Add USPP S-augmentation for each ion ----
    // S_sub += Σ_ion C_proj^dag · q · C_proj
    // where C_proj = beta_g^H · psi_row is the projector-wavefunction overlap.
    let psi_slice = psi_row.as_device_slice();
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // C_proj = beta_g^H · psi_row  (n_expanded × n_bands)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n,
                    k,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: k,   // beta_g is (ne, n_pw) row-major → col-major (n_pw, ne)
                    ldb: k,   // psi_row is (n_pw, n_bands)
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_slice,
                &mut c_proj,
            )?;
        }

        // temp = q · C_proj  (n_expanded × n_bands)
        let mut temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::N,
                    transb: op::N,
                    m: ne,
                    n,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.q_matrix,
                &c_proj,
                &mut temp,
            )?;
        }

        // S_sub += C_proj^dag · temp  (n_bands × n_bands, accumulated)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: n,
                    n,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 }, // accumulate into S_sub
                    ldc: n,
                },
                &c_proj,
                &temp,
                &mut s_sub_dev,
            )?;
        }
    }

    // ---- Step 3: Solve generalized eigenvalue problem via ZHEGVD ----
    // A * X = lambda * B * X  where A = H_sub, B = S_sub
    // On return: h_sub_dev contains eigenvectors X (column-major, n x n)
    let mut eigenvalues_dev: CudaSlice<f64> =
        stream.alloc_zeros(n_bands).map_err(Error::Cuda)?;
    let mut info_dev: CudaSlice<i32> = stream.alloc_zeros(1).map_err(Error::Cuda)?;

    solver.zhegvd(
        cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        n,
        &mut h_sub_dev,
        &mut s_sub_dev,
        &mut eigenvalues_dev,
        &mut info_dev,
    )?;

    // Check solver info
    stream.synchronize().map_err(Error::Cuda)?;
    let info: Vec<i32> = stream.clone_dtoh(&info_dev).map_err(Error::Cuda)?;
    if info[0] != 0 {
        return Err(Error::RayleighRitzFailed { info: info[0] });
    }

    // ---- Step 3b: D2H eigenvalues (moved here for PostRr pin) ----
    let eigenvalues_host: Vec<f64> = stream
        .clone_dtoh(&eigenvalues_dev)
        .map_err(Error::Cuda)?;
    pcie.d2h_bytes += eigenvalues_host.len() * 8;

    // ---- Step 3c: Procrustes pin (PostRr variant) ----
    // Apply pin AFTER ZHEGVD, on the eigenvector matrix X (in h_sub_dev).
    // This corrects for arbitrary in-block unitary rotations in near-degenerate clusters.
    if let (Some(prev_psi), Some(cfg)) = (prev_psi_dev, pin_cfg) {
        if cfg.mode == PinMode::PostRr {
            // Detect degenerate blocks
            let blocks = detect_degenerate_blocks(&eigenvalues_host, cfg.eps_degen);

            if !blocks.is_empty() {
                // Compute T = ψ_prev^H · S · ψ_row (full n×n, USPP-augmented)
                let mut t_dev: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_bands * n_bands).map_err(Error::Cuda)?;

                // Bare plane-wave: T = prev_psi^H · psi_row
                unsafe {
                    blas.gemm_c64(
                        ZgemmConfig {
                            transa: op::C,
                            transb: op::N,
                            m: n,
                            n,
                            k,
                            alpha: CudaComplex { x: 1.0, y: 0.0 },
                            lda: k,
                            ldb: k,
                            beta: CudaComplex { x: 0.0, y: 0.0 },
                            ldc: n,
                        },
                        prev_psi,
                        psi_row.as_device_slice(),
                        &mut t_dev,
                    )?;
                }

                // Add USPP augmentation: T += Σ_ion C_prev^H · q · C_row
                for entry in &vnl_data.entries {
                    let ne = entry.n_expanded;

                    // C_prev = beta_g^H · prev_psi
                    let mut c_prev: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
                    unsafe {
                        blas.gemm_c64(
                            ZgemmConfig {
                                transa: op::C,
                                transb: op::N,
                                m: ne,
                                n,
                                k,
                                alpha: CudaComplex { x: 1.0, y: 0.0 },
                                lda: k,
                                ldb: k,
                                beta: CudaComplex { x: 0.0, y: 0.0 },
                                ldc: ne,
                            },
                            &entry.beta_g,
                            prev_psi,
                            &mut c_prev,
                        )?;
                    }

                    // C_row = beta_g^H · psi_row
                    let mut c_row: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
                    unsafe {
                        blas.gemm_c64(
                            ZgemmConfig {
                                transa: op::C,
                                transb: op::N,
                                m: ne,
                                n,
                                k,
                                alpha: CudaComplex { x: 1.0, y: 0.0 },
                                lda: k,
                                ldb: k,
                                beta: CudaComplex { x: 0.0, y: 0.0 },
                                ldc: ne,
                            },
                            &entry.beta_g,
                            psi_row.as_device_slice(),
                            &mut c_row,
                        )?;
                    }

                    // temp = q · C_row
                    let mut temp: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
                    unsafe {
                        blas.gemm_c64(
                            ZgemmConfig {
                                transa: op::N,
                                transb: op::N,
                                m: ne,
                                n,
                                k: ne,
                                alpha: CudaComplex { x: 1.0, y: 0.0 },
                                lda: ne,
                                ldb: ne,
                                beta: CudaComplex { x: 0.0, y: 0.0 },
                                ldc: ne,
                            },
                            &entry.q_matrix,
                            &c_row,
                            &mut temp,
                        )?;
                    }

                    // T += C_prev^H · temp
                    unsafe {
                        blas.gemm_c64(
                            ZgemmConfig {
                                transa: op::C,
                                transb: op::N,
                                m: n,
                                n,
                                k: ne,
                                alpha: CudaComplex { x: 1.0, y: 0.0 },
                                lda: ne,
                                ldb: ne,
                                beta: CudaComplex { x: 1.0, y: 0.0 }, // accumulate
                                ldc: n,
                            },
                            &c_prev,
                            &temp,
                            &mut t_dev,
                        )?;
                    }
                }

                // Compute M = T · X (full n×n gemm)
                let mut m_dev: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_bands * n_bands).map_err(Error::Cuda)?;
                unsafe {
                    blas.gemm_c64(
                        ZgemmConfig {
                            transa: op::N,
                            transb: op::N,
                            m: n,
                            n,
                            k: n,
                            alpha: CudaComplex { x: 1.0, y: 0.0 },
                            lda: n,
                            ldb: n,
                            beta: CudaComplex { x: 0.0, y: 0.0 },
                            ldc: n,
                        },
                        &t_dev,
                        &h_sub_dev,
                        &mut m_dev,
                    )?;
                }

                // D2H M
                let m_host: Vec<CudaComplex> = stream
                    .clone_dtoh(&m_dev)
                    .map_err(Error::Cuda)?;
                pcie.d2h_bytes += n_bands * n_bands * 16;

                // Per-block pin loop with typed faer Mat API
                use faer::Mat;
                use num_complex::Complex64;

                // One typed conversion from the col-major device snapshot.
                // faer's Mat<T> is column-major by default — matches CUDA's layout.
                let m_full: Mat<Complex64> = Mat::from_fn(n_bands, n_bands, |r, c| {
                    let cu = m_host[c * n_bands + r];
                    Complex64::new(cu.x, cu.y)
                });

                // D2H X once, wrap as a typed col-major Mat.
                let x_host: Vec<CudaComplex> = stream
                    .clone_dtoh(&h_sub_dev)
                    .map_err(Error::Cuda)?;
                pcie.d2h_bytes += n_bands * n_bands * 16;

                let mut x_full: Mat<Complex64> = Mat::from_fn(n_bands, n_bands, |r, c| {
                    let cu = x_host[c * n_bands + r];
                    Complex64::new(cu.x, cu.y)
                });

                for &(lo, hi) in &blocks {
                    let k_block = hi - lo;

                    // Typed block extraction. submatrix(start_row, start_col, nrows, ncols) → MatRef.
                    let m_block = m_full.submatrix(lo, lo, k_block, k_block).to_owned();

                    let svd = match m_block.svd() {
                        Ok(s) => s,
                        Err(_) => {
                            #[cfg(feature = "scf_diag")]
                            eprintln!("[PostRr] SVD failed for block [{}, {})", lo, hi);
                            continue;
                        }
                    };

                    let u = svd.U();           // Mat<Complex64> (k × k)
                    let v = svd.V();           // Mat<Complex64> (k × k)  — NOT V^H
                    let r: Mat<Complex64> = u * v.adjoint();   // Procrustes-optimal unitary R = U · V^H

                    // Sanity: ‖R^H · R − I‖_F < 1e-10
                    {
                        let rh_r: Mat<Complex64> = r.adjoint() * &r;
                        let mut frob_err_sq = 0.0_f64;
                        for i in 0..k_block {
                            for j in 0..k_block {
                                let target = if i == j { Complex64::new(1.0, 0.0) } else { Complex64::ZERO };
                                frob_err_sq += (rh_r[(i, j)] - target).norm_sqr();
                            }
                        }
                        if frob_err_sq.sqrt() > 1e-10 {
                            #[cfg(feature = "scf_diag")]
                            eprintln!(
                                "[PostRr] R not unitary for block [{}, {}): ‖R^H·R − I‖_F = {:.3e} — skipping",
                                lo, hi, frob_err_sq.sqrt()
                            );
                            continue;
                        }
                    }

                    // Apply X[:, lo..hi] ← X[:, lo..hi] · R^H using typed mat-mul.
                    let x_slab = x_full.submatrix(0, lo, n_bands, k_block).to_owned();
                    let x_slab_new: Mat<Complex64> = x_slab * r.adjoint();

                    // Copy the rotated slab back into x_full's column band.
                    // faer offers no `assign_to_submatrix`, so we walk the typed Mat directly.
                    for c in 0..k_block {
                        for row in 0..n_bands {
                            x_full[(row, lo + c)] = x_slab_new[(row, c)];
                        }
                    }

                    #[cfg(feature = "scf_diag")]
                    eprintln!("[PostRr] pinned block [{}, {}), k={}", lo, hi, k_block);
                }

                // Single typed conversion back to flat Vec<CudaComplex> for H2D.
                let mut x_host_out: Vec<CudaComplex> = Vec::with_capacity(n_bands * n_bands);
                for c in 0..n_bands {
                    for r in 0..n_bands {
                        let z = x_full[(r, c)];
                        x_host_out.push(CudaComplex { x: z.re, y: z.im });
                    }
                }
                let h_sub_dev_new = stream.clone_htod(&x_host_out).map_err(Error::Cuda)?;
                pcie.h2d_bytes += n_bands * n_bands * 16;
                h_sub_dev = h_sub_dev_new;
            }
        }
    }


    // ---- Step 4-5: Rotate psi_new = psi_row · X ----
    //
    // psi_row memory is col-major (n_pw, n_bands): each col is a band ψ_b.
    // X (in h_sub_dev) is col-major (n_bands, n_bands): eigenvectors of (H,S).
    // The rotated wavefunctions ψ'_k = Σ_b X[b, k] · ψ_b correspond to
    // (psi_row · X)[g, k] in col-major (n_pw, n_bands).
    //
    // The result is then in the same layout as ColumnDistributed
    // (which is also col-major (n_pw, n_bands) — see comment in
    // `chebyshev.rs` after the apply_full_hamiltonian transpose-skip).
    //
    // gemm: C(m,n) = α A(m,k) B(k,n)
    //   m = n_pw, n = n_bands, k = n_bands (inner)
    //   A = psi_row, lda = n_pw      (col-major (n_pw, n_bands))
    //   B = X,       ldb = n_bands   (col-major (n_bands, n_bands))
    //   C = psi_new, ldc = n_pw      (col-major (n_pw, n_bands))
    let mut psi_new_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_pw).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::N,
                transb: op::N,
                m: k,            // n_pw
                n,               // n_bands
                k: n,            // n_bands (inner)
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,          // psi_row leading dim = n_pw
                ldb: n,          // X leading dim = n_bands
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: k,          // psi_new leading dim = n_pw
            },
            psi_row.as_device_slice(),
            &h_sub_dev,
            &mut psi_new_dev,
        )?;
    }

    // ---- Step 5b: Project β_g^H · ψ_new per ion (USPP augmentation density) ----
    //
    // For each ion I, compute βψ_I = β_g^H · ψ_new with shape (n_expanded × n_bands).
    // These projections feed `compute_aug_density_gpu` which accumulates the
    // occupancy matrix ω^I_{nm} = Σ_b occ_b · conj(βψ_I)_{n,b} · (βψ_I)_{m,b}.
    //
    // Layout: β_g is col-major (n_pw, n_expanded), ψ_new is col-major
    // (n_pw, n_bands). The gemm with transa=C gives (n_expanded × n_bands)
    // col-major. Keep βψ_I GPU-resident as CudaSlice<CudaComplex> to avoid
    // unnecessary D2H roundtrip.
    let mut beta_psi_per_ion: Vec<CudaSlice<CudaComplex>> =
        Vec::with_capacity(vnl_data.entries.len());
    let psi_new_slice: &CudaSlice<CudaComplex> = &psi_new_dev;
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;
        let mut bp_dev: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n,
                    k,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: k,
                    ldb: k,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_new_slice,
                &mut bp_dev,
            )?;
        }
        beta_psi_per_ion.push(bp_dev);
    }

    // ---- Step 6: D2H eigenvalues (already done in Step 3b) ----
    // eigenvalues_host was already D2H'd for the pin logic
    let eigenvalues = eigenvalues_host;

    stream.synchronize().map_err(Error::Cuda)?;

    // Wrap psi_new into Gpu<WavefunctionSet<ColumnDistributed>>
    let psi_new_gpu = Gpu::<WavefunctionSet<ColumnDistributed>> {
        slice: psi_new_dev,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };

    Ok((psi_new_gpu, Cpu(eigenvalues), beta_psi_per_ion))
}

/// Test-only variant that also returns H_sub, S_sub, and X for mathematical validation.
///
/// Identical to `rayleigh_ritz` but captures the n×n subspace matrices before ZHEGVD
/// consumes them, and the eigenvector matrix X after ZHEGVD writes it into `h_sub_dev`.
///
/// Return tuple: (psi_new, eigenvalues, beta_psi_per_ion, H_sub, S_sub, X)
/// All matrices are col-major (n_bands × n_bands) on the host.
#[cfg(any(test, feature = "scf_diag"))]
#[allow(clippy::too_many_arguments)]
pub fn rayleigh_ritz_with_matrices(
    psi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    hpsi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    vnl_data: &VnlBatchData,
    n_bands: usize,
    n_pw: usize,
    _kernels: &CudaKernelSet,
    pcie: &mut PcieAccount,
    solver: &SolverHandle,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
    prev_psi_dev: Option<&CudaSlice<CudaComplex>>,
    pin_cfg: Option<&RrPinConfig>,
) -> Result<
    (
        Gpu<WavefunctionSet<ColumnDistributed>>,
        Cpu<Vec<f64>>,
        Vec<CudaSlice<CudaComplex>>,
        Cpu<Vec<CudaComplex>>,
        Cpu<Vec<CudaComplex>>,
        Cpu<Vec<CudaComplex>>,
    ),
    Error,
> {
    let n = n_bands as i32;
    let k = n_pw as i32;

    // ---- Step 1: H_sub = psi^dag * hpsi ----
    let mut h_sub_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_bands).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,
                transb: op::N,
                m: n,
                n,
                k,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,
                ldb: k,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n,
            },
            psi_row.as_device_slice(),
            hpsi_row.as_device_slice(),
            &mut h_sub_dev,
        )?;
    }

    // ---- Step 2: S_sub = psi^dag * psi ----
    let mut s_sub_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_bands).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,
                transb: op::N,
                m: n,
                n,
                k,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,
                ldb: k,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n,
            },
            psi_row.as_device_slice(),
            psi_row.as_device_slice(),
            &mut s_sub_dev,
        )?;
    }

    // ---- Step 2b: Add USPP S-augmentation ----
    let psi_slice = psi_row.as_device_slice();
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n,
                    k,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: k,
                    ldb: k,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_slice,
                &mut c_proj,
            )?;
        }

        let mut temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::N,
                    transb: op::N,
                    m: ne,
                    n,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.q_matrix,
                &c_proj,
                &mut temp,
            )?;
        }

        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: n,
                    n,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 },
                    ldc: n,
                },
                &c_proj,
                &temp,
                &mut s_sub_dev,
            )?;
        }
    }

    // ---- Save H_sub and S_sub before ZHEGVD overwrites them ----
    stream.synchronize().map_err(Error::Cuda)?;
    let h_sub_host_pre = stream.clone_dtoh(&h_sub_dev).map_err(Error::Cuda)?;
    let s_sub_host_pre = stream.clone_dtoh(&s_sub_dev).map_err(Error::Cuda)?;

    // ---- Step 3: ZHEGVD ----
    let mut eigenvalues_dev: CudaSlice<f64> =
        stream.alloc_zeros(n_bands).map_err(Error::Cuda)?;
    let mut info_dev: CudaSlice<i32> = stream.alloc_zeros(1).map_err(Error::Cuda)?;

    solver.zhegvd(
        cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        n,
        &mut h_sub_dev,
        &mut s_sub_dev,
        &mut eigenvalues_dev,
        &mut info_dev,
    )?;

    stream.synchronize().map_err(Error::Cuda)?;
    let info: Vec<i32> = stream.clone_dtoh(&info_dev).map_err(Error::Cuda)?;
    if info[0] != 0 {
        return Err(Error::RayleighRitzFailed { info: info[0] });
    }

    // ---- Save X (h_sub_dev now contains eigenvectors) ----
    let x_host = stream.clone_dtoh(&h_sub_dev).map_err(Error::Cuda)?;

    // ---- Step 4-5: Rotate psi_new = psi_row · X ----
    let mut psi_new_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_pw).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::N,
                transb: op::N,
                m: k,
                n,
                k: n,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,
                ldb: n,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: k,
            },
            psi_row.as_device_slice(),
            &h_sub_dev,
            &mut psi_new_dev,
        )?;
    }

    // ---- Step 5b: β_g^H · ψ_new per ion ----
    let mut beta_psi_per_ion: Vec<CudaSlice<CudaComplex>> =
        Vec::with_capacity(vnl_data.entries.len());
    let psi_new_slice: &CudaSlice<CudaComplex> = &psi_new_dev;
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;
        let mut bp_dev: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n,
                    k,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: k,
                    ldb: k,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_new_slice,
                &mut bp_dev,
            )?;
        }
        beta_psi_per_ion.push(bp_dev);
    }

    // ---- Step 6: D2H eigenvalues ----
    let eigenvalues: Vec<f64> = stream
        .clone_dtoh(&eigenvalues_dev)
        .map_err(Error::Cuda)?;
    pcie.d2h_bytes += eigenvalues.len() * 8;

    stream.synchronize().map_err(Error::Cuda)?;

    let psi_new_gpu = Gpu::<WavefunctionSet<ColumnDistributed>> {
        slice: psi_new_dev,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };

    Ok((
        psi_new_gpu,
        Cpu(eigenvalues),
        beta_psi_per_ion,
        Cpu(h_sub_host_pre),
        Cpu(s_sub_host_pre),
        Cpu(x_host),
    ))
}

/// Test-only: expose `rayleigh_ritz_with_matrices` through the density test_api path.
#[cfg(test)]
pub(crate) mod test_api {
    pub use super::rayleigh_ritz_with_matrices;
}
