// ---------------------------------------------------------------------------
// Phase A: T + V_loc Chebyshev filter (simplified, no V_NL, no S-operator)
// ---------------------------------------------------------------------------
//
// The Chebyshev recurrence is the standard unscaled form:
//   σ(H) = (H - c·I) / e
//   ψ_1 = σ(H)·ψ_0
//   ψ_k = 2·σ(H)·ψ_{k-1} - ψ_{k-2}

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::driver::{CudaSlice, CudaStream};

use crate::device::blas::{BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::{KineticPreconditioner, PwCoefficients};
use crate::eigensolver::hamiltonian::apply_v_loc_hamiltonian;
use crate::eigensolver::kernels::CudaKernelSet;
use crate::types::Error;

// ---------------------------------------------------------------------------
// T + V_loc application
// ---------------------------------------------------------------------------

pub(crate) unsafe fn apply_h_tv(
    psi_dev: &PwCoefficients,
    hpsi_dev: &mut PwCoefficients,
    grid_dev: &mut CudaSlice<CudaComplex>,
    kinetic_dev: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    v_eff_dev: &CudaSlice<f64>,
    n_pw: i32,
    n_bands: i32,
    grid_size: i32,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        apply_v_loc_hamiltonian(
            psi_dev,
            hpsi_dev,
            grid_dev,
            kinetic_dev,
            fft_idx_dev,
            v_eff_dev,
            n_pw,
            n_bands,
            grid_size,
            inv_ntotal,
            fft_plan,
            kernels,
            stream,
        )
    }
}

// ---------------------------------------------------------------------------
// Scaled Hamiltonian: hpsi = (hpsi - c·psi) / e
// ---------------------------------------------------------------------------

pub(crate) unsafe fn apply_scaled_hamiltonian_inplace(
    hpsi_dev: &mut PwCoefficients,
    psi_dev: &PwCoefficients,
    n_pw: i32,
    n_bands: i32,
    center: f64,
    half_width: f64,
    blas: &BlasHandle,
) -> Result<(), Error> {
    let n = n_pw * n_bands;
    let inv_e = CudaComplex { x: 1.0 / half_width, y: 0.0 };
    let neg_c_over_e = CudaComplex { x: -center / half_width, y: 0.0 };

    unsafe {
        cudarc::cublas::sys::cublasZscal_v2(
            blas.raw_handle(),
            n,
            &inv_e as *const _ as *const _,
            hpsi_dev.device_ptr_mut(blas.stream()).0 as *mut _,
            1,
        )
        .result()?;
        cudarc::cublas::sys::cublasZaxpy_v2(
            blas.raw_handle(),
            n,
            &neg_c_over_e as *const _ as *const _,
            psi_dev.device_ptr(blas.stream()).0 as *const _,
            1,
            hpsi_dev.device_ptr_mut(blas.stream()).0 as *mut _,
            1,
        )
        .result()?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Chebyshev combination: psi_next = 2*hpsi - psi_prev
// ---------------------------------------------------------------------------

pub(crate) unsafe fn chebyshev_combine(
    psi_next: &mut PwCoefficients,
    hpsi: &PwCoefficients,
    psi_prev: &PwCoefficients,
    n_pw: usize,
    n_bands: usize,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    let n = (n_pw * n_bands) as u32;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;
    let total = (n_pw * n_bands) as i32;

    let mut scale_dev: CudaSlice<f64> = stream.alloc_zeros(n_bands).map_err(Error::Cuda)?;
    stream
        .memcpy_htod(&vec![2.0f64; n_bands], &mut scale_dev)
        .map_err(Error::Cuda)?;

    // Zero psi_next
    unsafe {
        stream
            .launch_builder(&kernels.zero_buffer)
            .arg(&mut **psi_next)
            .arg(&total)
            .launch(LaunchConfig::for_num_elems(n))
    }
    .map_err(Error::Cuda)?;

    // psi_next = 2.0 * hpsi via band_scale_axpy(dst, hpsi, scale=2, alpha=1)
    unsafe {
        stream
            .launch_builder(&kernels.band_scale_axpy)
            .arg(&mut **psi_next)
            .arg(&**hpsi)
            .arg(&scale_dev)
            .arg(&1.0f64)
            .arg(&n_pw_i32)
            .arg(&n_bands_i32)
            .launch(LaunchConfig::for_num_elems(n))
    }
    .map_err(Error::Cuda)?;

    // Update scale: all 1.0
    stream
        .memcpy_htod(&vec![1.0f64; n_bands], &mut scale_dev)
        .map_err(Error::Cuda)?;

    // psi_next += (-1.0) * psi_prev  => psi_next = 2*hpsi - psi_prev
    unsafe {
        stream
            .launch_builder(&kernels.band_scale_axpy)
            .arg(&mut **psi_next)
            .arg(&**psi_prev)
            .arg(&scale_dev)
            .arg(&-1.0f64)
            .arg(&n_pw_i32)
            .arg(&n_bands_i32)
            .launch(LaunchConfig::for_num_elems(n))
    }
    .map_err(Error::Cuda)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// L2 Gram-Schmidt (modified Gram-Schmidt, no S-operator)
// ---------------------------------------------------------------------------

pub(crate) unsafe fn gram_schmidt(
    psi: &mut PwCoefficients,
    n_pw: usize,
    n_bands: usize,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    let n_pw_i32 = n_pw as i32;
    let band_stride = (n_pw * std::mem::size_of::<CudaComplex>()) as u64;

    // Get base device pointers (u64 byte addresses) upfront.
    let (r_base, _) = psi.device_ptr(stream);
    let (w_base, _) = psi.device_ptr_mut(stream);

    for b in 0..n_bands {
        let b_off = (b as u64) * band_stride;

        // 1. Subtract projections of previous bands
        for p in 0..b {
            let p_off = (p as u64) * band_stride;

            let mut dot_h: CudaComplex = CudaComplex { x: 0.0, y: 0.0 };
            unsafe {
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(),
                    n_pw_i32,
                    (r_base + p_off) as *const _,
                    1,
                    (r_base + b_off) as *const _,
                    1,
                    &mut dot_h as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;
            }

            if dot_h.x == 0.0 && dot_h.y == 0.0 {
                continue;
            }

            // psi[b] -= dot * psi[p]
            let alpha = CudaComplex { x: -dot_h.x, y: -dot_h.y };
            unsafe {
                cudarc::cublas::sys::cublasZaxpy_v2(
                    blas.raw_handle(),
                    n_pw_i32,
                    &alpha as *const _ as *const _,
                    (r_base + p_off) as *const _,
                    1,
                    (w_base + b_off) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }

        // 2. Normalise: norm = sqrt(<psi[b]|psi[b]>)
        let mut norm: f64 = 0.0;
        unsafe {
            cudarc::cublas::sys::cublasDznrm2_v2(
                blas.raw_handle(),
                n_pw_i32,
                (r_base + b_off) as *const _,
                1,
                &mut norm as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;
        }

        if norm > 1e-30 {
            let inv_norm = CudaComplex { x: 1.0 / norm, y: 0.0 };
            unsafe {
                cudarc::cublas::sys::cublasZscal_v2(
                    blas.raw_handle(),
                    n_pw_i32,
                    &inv_norm as *const _ as *const _,
                    (w_base + b_off) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Subspace matrix assembly: H_sub = psi^H · hpsi, S_sub = psi^H · psi
// ---------------------------------------------------------------------------

pub(crate) unsafe fn build_subspace_matrices(
    psi: &PwCoefficients,
    hpsi: &PwCoefficients,
    h_sub: &mut CudaSlice<CudaComplex>,
    s_sub: &mut CudaSlice<CudaComplex>,
    n_pw: i32,
    n_bands: i32,
    blas: &BlasHandle,
) -> Result<(), Error> {
    let alpha = CudaComplex { x: 1.0, y: 0.0 };
    let beta = CudaComplex { x: 0.0, y: 0.0 };

    // H_sub = psi^H · hpsi
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_C,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_bands,
                n: n_bands,
                k: n_pw,
                alpha,
                lda: n_pw,
                ldb: n_pw,
                beta,
                ldc: n_bands,
            },
            psi,
            hpsi,
            h_sub,
        )
    }
    .map_err(Error::Blas)?;

    // S_sub = psi^H · psi
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_C,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_bands,
                n: n_bands,
                k: n_pw,
                alpha,
                lda: n_pw,
                ldb: n_pw,
                beta,
                ldc: n_bands,
            },
            psi,
            psi,
            s_sub,
        )
    }
    .map_err(Error::Blas)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Generalized eigenvalue solve: H_sub · X = lambda · S_sub · X
// ---------------------------------------------------------------------------

pub(crate) unsafe fn solve_generalized(
    x: &mut CudaSlice<CudaComplex>,
    s: &mut CudaSlice<CudaComplex>,
    eigenvalues: &mut CudaSlice<f64>,
    info: &mut CudaSlice<i32>,
    n_bands: i32,
    solver: &SolverHandle,
    _stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    use cudarc::cusolver::sys::{
        cublasFillMode_t, cusolverEigMode_t,
    };
    solver.zhegvd(
        cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        n_bands,
        x,
        s,
        eigenvalues,
        info,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Basis rotation: psi_out = psi · X, hpsi_out = hpsi · X
// ---------------------------------------------------------------------------

pub(crate) unsafe fn rotate_basis(
    psi: &PwCoefficients,
    hpsi: &PwCoefficients,
    x: &CudaSlice<CudaComplex>,
    psi_out: &mut PwCoefficients,
    hpsi_out: &mut PwCoefficients,
    n_pw: i32,
    n_bands: i32,
    blas: &BlasHandle,
) -> Result<(), Error> {
    let alpha = CudaComplex { x: 1.0, y: 0.0 };
    let beta = CudaComplex { x: 0.0, y: 0.0 };

    // psi_out = psi · X
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_pw,
                n: n_bands,
                k: n_bands,
                alpha,
                lda: n_pw,
                ldb: n_bands,
                beta,
                ldc: n_pw,
            },
            psi,
            x,
            psi_out,
        )
    }
    .map_err(Error::Blas)?;

    // hpsi_out = hpsi · X
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_pw,
                n: n_bands,
                k: n_bands,
                alpha,
                lda: n_pw,
                ldb: n_bands,
                beta,
                ldc: n_pw,
            },
            hpsi,
            x,
            hpsi_out,
        )
    }
    .map_err(Error::Blas)?;

    Ok(())
}
