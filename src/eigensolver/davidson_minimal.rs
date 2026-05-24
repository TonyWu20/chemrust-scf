// ---------------------------------------------------------------------------
// Minimal single-sweep Davidson eigensolver (Phase 0 Gate 3)
// ---------------------------------------------------------------------------
//
// Implements a single-sweep Davidson algorithm for per-band locking:
//   1. Compute H|ψ⟩ and S|ψ⟩ for all bands
//   2. Per-band Rayleigh quotient λ_b = Re⟨ψ_b|H|ψ_b⟩ / Re⟨ψ_b|S|ψ_b⟩
//   3. Residual r_b = H|ψ_b⟩ − λ_b·S|ψ_b⟩
//   4. Lock bands with ‖r_b‖₂ < lock_tol
//   5. Sub-block ZHEGVD on unconverged band subspace (k×k)
//   6. Rotate unconverged bands into new eigenbasis
//   7. S-orthogonalize against locked bands
//   8. Concatenate locked + unconverged into output
//
// Reference: Zhou (2014), Davidson eigenvector locking pattern.

use std::sync::Arc;

use cudarc::cublas::sys::{
    cublasDznrm2_v2, cublasZaxpy_v2, cublasZcopy_v2, cublasZdotc_v2,
};
use cudarc::cusolver::sys::{cublasFillMode_t, cusolverEigMode_t};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::chebyshev::{apply_full_hamiltonian, apply_s_times, CudaKernelSet};
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;

// ---------------------------------------------------------------------------
// Result types and diagnostics
// ---------------------------------------------------------------------------

/// Result of a single-sweep Davidson diagonalization.
#[allow(dead_code)]
pub struct DavidsonResult {
    pub psi_out: CudaSlice<CudaComplex>,
    pub eigenvalues: Vec<f64>,
    pub n_locked: usize,
    pub n_unconverged: usize,
    pub residual_norms: Vec<f64>,
}

/// Snapshot of Davidson diagnostics after the most recent solve.
#[derive(Clone)]
pub struct DavidsonDiagnostic {
    pub n_locked: usize,
    pub n_unconverged: usize,
    pub locked_indices: Vec<usize>,
    pub unconv_indices: Vec<usize>,
    pub residual_norms: Vec<f64>,
    pub max_residual: f64,
    pub mean_residual: f64,
    pub lock_tol: f64,
}

/// Most recent Davidson diagnostic, accessible via re-export for tests.
pub static DAVIDSON_LAST_DIAG: std::sync::Mutex<Option<DavidsonDiagnostic>> =
    std::sync::Mutex::new(None);

// ---------------------------------------------------------------------------
// Main single-sweep driver
// ---------------------------------------------------------------------------

/// Run a single-sweep Davidson correction + locking.
///
/// # Safety
///
/// All device pointers must be valid and of sufficient size. `psi_in` must
/// be an S-orthonormalized wavefunction set (column-major, n_bands × n_pw).
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn, dead_code)]
pub(crate) unsafe fn davidson_minimal_single_sweep(
    psi_in: &CudaSlice<CudaComplex>,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &CudaSlice<f64>,
    fft_idx_dev: &CudaSlice<i32>,
    vnl_data: &VnlBatchData,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    lock_tol: f64,
    blas: &BlasHandle,
    solver: &SolverHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
    _ctx: &Arc<CudaContext>,
) -> Result<DavidsonResult, Error> {
    let n_elem = n_bands * n_pw;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;
    let handle = blas.raw_handle();

    // ------------------------------------------------------------------
    // Step 1: Allocate workspaces
    // ------------------------------------------------------------------
    let mut hpsi_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut spsi_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let grid_alloc = n_bands * grid_size;
    let mut grid_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // ------------------------------------------------------------------
    // Step 2: Hψ = apply_full_hamiltonian(psi_in)
    // ------------------------------------------------------------------
    unsafe {
        apply_full_hamiltonian(
            psi_in,
            v_eff_dev,
            kinetic_dev,
            fft_idx_dev,
            n_pw,
            n_bands,
            grid_size,
            inv_ntotal,
            fft_plan,
            &mut hpsi_dev,
            &mut grid_dev,
            vnl_data,
            blas,
            kernels,
            stream,
        )?;
    }

    // ------------------------------------------------------------------
    // Step 3: Sψ — pre-copy psi_in into spsi_dev, then apply_s_times
    // ------------------------------------------------------------------
    stream.memcpy_dtod(psi_in, &mut spsi_dev).map_err(Error::Cuda)?;
    unsafe {
        apply_s_times(
            psi_in,
            &mut spsi_dev,
            vnl_data,
            n_bands_i32,
            n_pw_i32,
            blas,
            stream,
        )?;
    }

    // ------------------------------------------------------------------
    // Step 4: Per-band Rayleigh quotient λ_b = Re⟨ψ_b|Hψ_b⟩ / Re⟨ψ_b|Sψ_b⟩
    // ------------------------------------------------------------------
    let mut lambdas = vec![0.0_f64; n_bands];
    {
        let (psi_ptr, _) = psi_in.device_ptr(stream);
        let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
        let (spsi_ptr, _) = spsi_dev.device_ptr(stream);

        for (b, lambda) in lambdas.iter_mut().enumerate() {
            let psi_b = (psi_ptr as *const CudaComplex).add(b * n_pw);
            let hpsi_b = (hpsi_ptr as *const CudaComplex).add(b * n_pw);
            let spsi_b = (spsi_ptr as *const CudaComplex).add(b * n_pw);

            let mut dot_h = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle,
                n_pw_i32,
                psi_b as *const _,
                1,
                hpsi_b as *const _,
                1,
                &mut dot_h as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;

            let mut dot_s = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle,
                n_pw_i32,
                psi_b as *const _,
                1,
                spsi_b as *const _,
                1,
                &mut dot_s as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;

            *lambda = dot_h.x / dot_s.x;
        }
    }

    // ------------------------------------------------------------------
    // Step 5: Per-band residual r_b = Hψ_b − λ_b · Sψ_b
    // ------------------------------------------------------------------
    let mut residual_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    stream
        .memcpy_dtod(&hpsi_dev, &mut residual_dev)
        .map_err(Error::Cuda)?;

    {
        let (spsi_ptr, _) = spsi_dev.device_ptr(stream);
        let (residual_ptr_mut, _) = residual_dev.device_ptr_mut(stream);

        for (b, lambda) in lambdas.iter().enumerate() {
            let spsi_b = (spsi_ptr as *const CudaComplex).add(b * n_pw);
            let r_b = (residual_ptr_mut as *mut CudaComplex).add(b * n_pw);
            let alpha = CudaComplex {
                x: -lambda,
                y: 0.0,
            };
            cublasZaxpy_v2(
                handle,
                n_pw_i32,
                &alpha as *const _ as *const _,
                spsi_b as *const _,
                1,
                r_b as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // ------------------------------------------------------------------
    // Step 6: Per-band L2 norm of residual
    // ------------------------------------------------------------------
    let mut residual_norms = vec![0.0_f64; n_bands];
    {
        let (residual_ptr, _) = residual_dev.device_ptr(stream);

        for (b, rn) in residual_norms.iter_mut().enumerate() {
            let r_b = (residual_ptr as *const CudaComplex).add(b * n_pw);
            cublasDznrm2_v2(
                handle,
                n_pw_i32,
                r_b as *const _,
                1,
                rn as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // ------------------------------------------------------------------
    // Step 7: Lock list — bands below lock_tol
    // ------------------------------------------------------------------
    let mut locked = vec![false; n_bands];
    let mut unconv_idx: Vec<usize> = Vec::new();
    for (b, rn) in residual_norms.iter().enumerate() {
        if *rn < lock_tol {
            locked[b] = true;
        } else {
            unconv_idx.push(b);
        }
    }
    let n_locked = locked.iter().filter(|&&l| l).count();
    let k = unconv_idx.len();
    let locked_indices: Vec<usize> = locked
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l)
        .map(|(i, _)| i)
        .collect();

    // ------------------------------------------------------------------
    // Step 8: Early return if all bands are locked
    // ------------------------------------------------------------------
    if k == 0 {
        let mut psi_out: CudaSlice<CudaComplex> =
            stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
        stream
            .memcpy_dtod(psi_in, &mut psi_out)
            .map_err(Error::Cuda)?;

        let eigenvalues = lambdas;

        // Update diagnostics
        let max_res = residual_norms
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        let mean_res = residual_norms.iter().sum::<f64>() / n_bands as f64;
        *DAVIDSON_LAST_DIAG.lock().unwrap() = Some(DavidsonDiagnostic {
            n_locked,
            n_unconverged: k,
            locked_indices: locked_indices.clone(),
            unconv_indices: unconv_idx.clone(),
            residual_norms: residual_norms.clone(),
            max_residual: max_res,
            mean_residual: mean_res,
            lock_tol,
        });

        return Ok(DavidsonResult {
            psi_out,
            eigenvalues,
            n_locked,
            n_unconverged: k,
            residual_norms,
        });
    }

    // ------------------------------------------------------------------
    // Step 9: Sub-block ZHEGVD on unconverged (k × k)
    // ------------------------------------------------------------------
    let k_i32 = k as i32;

    // 9a. Gather unconverged columns into contiguous buffers
    let mut psi_unconv_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?;
    let mut hpsi_unconv_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?;

    {
        let (psi_ptr, _) = psi_in.device_ptr(stream);
        let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
        let (psi_unconv_mut, _) = psi_unconv_dev.device_ptr_mut(stream);
        let (hpsi_unconv_mut, _) = hpsi_unconv_dev.device_ptr_mut(stream);

        for (u, &b) in unconv_idx.iter().enumerate() {
            // Copy psi_in column b → psi_unconv column u
            cublasZcopy_v2(
                handle,
                n_pw_i32,
                (psi_ptr as *const CudaComplex).add(b * n_pw) as *const _,
                1,
                (psi_unconv_mut as *mut CudaComplex).add(u * n_pw) as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;

            // Copy hpsi_dev column b → hpsi_unconv column u
            cublasZcopy_v2(
                handle,
                n_pw_i32,
                (hpsi_ptr as *const CudaComplex).add(b * n_pw) as *const _,
                1,
                (hpsi_unconv_mut as *mut CudaComplex).add(u * n_pw) as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // 9b. Build H_sub = ψ_unconv^H · hpsi_unconv  (k×k)
    let mut h_sub_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(k * k).map_err(Error::Cuda)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,
                transb: op::N,
                m: k_i32,
                n: k_i32,
                k: n_pw_i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw_i32,
                ldb: n_pw_i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: k_i32,
            },
            &psi_unconv_dev,
            &hpsi_unconv_dev,
            &mut h_sub_dev,
        )?;
    }

    // S_sub = ψ_unconv^H · ψ_unconv  (bare PW overlap)  (k×k)
    let mut s_sub_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(k * k).map_err(Error::Cuda)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,
                transb: op::N,
                m: k_i32,
                n: k_i32,
                k: n_pw_i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw_i32,
                ldb: n_pw_i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: k_i32,
            },
            &psi_unconv_dev,
            &psi_unconv_dev,
            &mut s_sub_dev,
        )?;
    }

    // Add USPP overlap contribution: S += c_proj^H · q_matrix · c_proj
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // c_proj = beta_g^H · psi_unconv  (ne × k)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * k).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n: k_i32,
                    k: n_pw_i32,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw_i32,
                    ldb: n_pw_i32,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                &psi_unconv_dev,
                &mut c_proj,
            )?;
        }

        // temp = q_matrix · c_proj  (ne × k)
        let mut temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * k).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::N,
                    transb: op::N,
                    m: ne,
                    n: k_i32,
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

        // S_sub += c_proj^H · temp  (k×k)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: k_i32,
                    n: k_i32,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 }, // accumulate
                    ldc: k_i32,
                },
                &c_proj,
                &temp,
                &mut s_sub_dev,
            )?;
        }
    }

    // 9c. Solve generalized eigenvalue problem via ZHEGVD:
    //      H_sub · X = Λ · S_sub · X
    //     h_sub_dev is overwritten with eigenvectors (column-major)
    let mut eig_dev: CudaSlice<f64> = stream.alloc_zeros(k).map_err(Error::Cuda)?;
    let mut info_dev: CudaSlice<i32> = stream.alloc_zeros(1).map_err(Error::Cuda)?;
    solver.zhegvd(
        cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        k_i32,
        &mut h_sub_dev, // overwritten → eigenvectors
        &mut s_sub_dev, // overwritten (scratch)
        &mut eig_dev,
        &mut info_dev,
    )?;

    // 9d. Check ZHEGVD status, D2H eigenvalues
    let info_cpu: Vec<i32> = stream.clone_dtoh(&info_dev).map_err(Error::Cuda)?;
    if info_cpu[0] != 0 {
        return Err(Error::RayleighRitzFailed {
            info: info_cpu[0],
        });
    }
    let eigenvalues_k: Vec<f64> = stream.clone_dtoh(&eig_dev).map_err(Error::Cuda)?;

    // 9e. h_sub_dev now holds eigenvectors (k×k). Use directly for rotation.
    //     (No separate H2D needed — already on GPU.)

    // ------------------------------------------------------------------
    // Step 10: Rotate unconverged: ψ_new = ψ_unconv · X  (n_pw × k)
    // ------------------------------------------------------------------
    let mut psi_unconv_new: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::N,
                transb: op::N,
                m: n_pw_i32,
                n: k_i32,
                k: k_i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw_i32,
                ldb: k_i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n_pw_i32,
            },
            &psi_unconv_dev,
            &h_sub_dev, // eigenvectors in column-major
            &mut psi_unconv_new,
        )?;
    }

    // ------------------------------------------------------------------
    // Step 11: Single-pass S-orthogonalize unconverged against locked
    // ------------------------------------------------------------------
    if n_locked > 0 {
        let (psi_in_ptr, _) = psi_in.device_ptr(stream);
        let (psi_unconv_new_mut, _) = psi_unconv_new.device_ptr_mut(stream);

        for u in 0..k {
            let psi_new_u = (psi_unconv_new_mut as *mut CudaComplex).add(u * n_pw);

            // Copy psi_unconv_new[:, u] to a single-band buffer then compute S·ψ_u
            let mut psi_u_dev: CudaSlice<CudaComplex> =
                stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
            let mut s_u_dev: CudaSlice<CudaComplex> =
                stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;

            // psi_u_dev = psi_unconv_new[:, u]
            {
                let (psi_u_mut, _) = psi_u_dev.device_ptr_mut(stream);
                cublasZcopy_v2(
                    handle,
                    n_pw_i32,
                    psi_new_u as *const _,
                    1,
                    psi_u_mut as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }

            // s_u_dev = psi_u_dev  (pre-copy — apply_s_times accumulates)
            stream
                .memcpy_dtod(&psi_u_dev, &mut s_u_dev)
                .map_err(Error::Cuda)?;

            // s_u_dev = S · psi_u_dev
            unsafe {
                apply_s_times(
                    &psi_u_dev,
                    &mut s_u_dev,
                    vnl_data,
                    1, // n_bands = 1
                    n_pw_i32,
                    blas,
                    stream,
                )?;
            }

            // For each locked band: dot = ⟨ψ_j|S|ψ_u⟩; ψ_u -= dot · ψ_j
            let (s_u_ptr, _) = s_u_dev.device_ptr(stream);

            for (j, is_locked) in locked.iter().enumerate() {
                if !is_locked {
                    continue;
                }
                let psi_in_j = (psi_in_ptr as *const CudaComplex).add(j * n_pw);

                let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                cublasZdotc_v2(
                    handle,
                    n_pw_i32,
                    psi_in_j as *const _,
                    1,
                    s_u_ptr as *const _,
                    1,
                    &mut dot as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;

                let neg_dot = CudaComplex {
                    x: -dot.x,
                    y: -dot.y,
                };
                cublasZaxpy_v2(
                    handle,
                    n_pw_i32,
                    &neg_dot as *const _ as *const _,
                    psi_in_j as *const _,
                    1,
                    psi_new_u as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 12: Concatenate locked + unconverged into output buffer
    // ------------------------------------------------------------------
    let mut psi_out: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    {
        let (psi_out_mut, _) = psi_out.device_ptr_mut(stream);
        let (psi_in_ptr, _) = psi_in.device_ptr(stream);
        let (psi_unconv_new_ptr, _) = psi_unconv_new.device_ptr(stream);

        for (b, is_locked) in locked.iter().enumerate() {
            let dst = (psi_out_mut as *mut CudaComplex).add(b * n_pw);
            if *is_locked {
                let src = (psi_in_ptr as *const CudaComplex).add(b * n_pw);
                cublasZcopy_v2(handle, n_pw_i32, src as *const _, 1, dst as *mut _, 1)
                    .result()
                    .map_err(Error::Blas)?;
            } else {
                let pos = unconv_idx.iter().position(|&x| x == b).unwrap();
                let src = (psi_unconv_new_ptr as *const CudaComplex).add(pos * n_pw);
                cublasZcopy_v2(handle, n_pw_i32, src as *const _, 1, dst as *mut _, 1)
                    .result()
                    .map_err(Error::Blas)?;
            }
        }
    }

    // Build eigenvalue output: locked from Rayleigh quotient, unconv from ZHEGVD
    let mut eigenvalues = lambdas; // copy
    for (pos, &b) in unconv_idx.iter().enumerate() {
        eigenvalues[b] = eigenvalues_k[pos];
    }

    // ------------------------------------------------------------------
    // Step 13: Write diagnostics and return
    // ------------------------------------------------------------------
    let max_res = residual_norms
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    let mean_res = residual_norms.iter().sum::<f64>() / n_bands as f64;
    *DAVIDSON_LAST_DIAG.lock().unwrap() = Some(DavidsonDiagnostic {
        n_locked,
        n_unconverged: k,
        locked_indices: locked_indices.clone(),
        unconv_indices: unconv_idx.clone(),
        residual_norms: residual_norms.clone(),
        max_residual: max_res,
        mean_residual: mean_res,
        lock_tol,
    });

    Ok(DavidsonResult {
        psi_out,
        eigenvalues,
        n_locked,
        n_unconverged: k,
        residual_norms,
    })
}
