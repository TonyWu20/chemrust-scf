// ---------------------------------------------------------------------------
// Hamiltonian and USPP overlap operators (GPU)
// ---------------------------------------------------------------------------
//
// Implements:
//   1. c2c_inverse_inplace / c2c_forward_inplace — C2C FFT wrappers
//   2. apply_v_loc_hamiltonian — T + V_loc via FFT round-trip
//   3. apply_v_nl_hamiltonian — V_NL via cuBLAS gemm with β-projectors
//   4. apply_full_hamiltonian — composes V_loc + V_NL
//   5. apply_s_times — S·ψ = ψ + β·Q·β^H·ψ (USPP overlap)

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::device::blas::{self, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::{
    KineticPreconditioner, PwCoefficients,
};
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;
use bon::builder;

// ---------------------------------------------------------------------------
// Helper: call cuFFT C2C in-place (same buffer for input and output)
// ---------------------------------------------------------------------------

/// Call `c2c_inverse` in-place. cuFFT natively supports in-place transforms,
/// so passing the same `&mut` twice via raw pointer is correct.
unsafe fn c2c_inverse_inplace(
    plan: &BatchedFftPlan3d,
    buf: &mut CudaSlice<CudaComplex>,
) -> Result<(), cudarc::cufft::result::CufftError> {
    let ptr = buf as *mut CudaSlice<CudaComplex>;
    unsafe { plan.c2c_inverse(&mut *ptr, &mut *ptr) }
}

/// Call `c2c_forward` in-place.
unsafe fn c2c_forward_inplace(
    plan: &BatchedFftPlan3d,
    buf: &mut CudaSlice<CudaComplex>,
) -> Result<(), cudarc::cufft::result::CufftError> {
    let ptr = buf as *mut CudaSlice<CudaComplex>;
    unsafe { plan.c2c_forward(&mut *ptr, &mut *ptr) }
}

// ---------------------------------------------------------------------------
// Full Hamiltonian application (T + V_loc on GPU)
// ---------------------------------------------------------------------------

/// Compute (T + V_loc)|psi> on GPU using FFT-based approach.
///
/// Steps:
/// 1. hpsi = T|psi>  (init_kinetic kernel)
/// 2. Scatter psi coefficients to FFT grid
/// 3. Batched C2C IFFT
/// 4. V_eff multiply (pointwise)
/// 5. Batched C2C FFT
/// 6. Gather + add to hpsi: hpsi += grid / N_total
#[builder]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_v_loc_hamiltonian(
    psi_dev: &PwCoefficients,
    hpsi_dev: &mut PwCoefficients,
    grid_dev: &mut CudaSlice<CudaComplex>,
    kinetic_dev: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    v_eff_dev: &CudaSlice<f64>,
    n_pw: i32,
    n_bands: i32,
    grid_size: i32,
    ngx: i32,
    ngy: i32,
    ngz: i32,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
    _blas: Option<&BlasHandle>,
) -> Result<(), Error> {
    // 1. hpsi = kinetic * psi  (T|psi>)
    unsafe {
        stream
            .launch_builder(&kernels.init_kinetic)
            .arg(&mut **hpsi_dev)
            .arg(&**psi_dev)
            .arg(&**kinetic_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;

    // Diag: |hpsi|² for last band after kinetic
    #[cfg(feature = "scf_diag")]
    if let Some(bh) = _blas {
        let lb = n_bands as usize - 1;
        let (h_ptr, _) = (&*hpsi_dev).device_ptr(stream);
        let col = (h_ptr as *const CudaComplex).add(lb * n_pw as usize);
        let mut nrm2 = CudaComplex { x: 0.0, y: 0.0 };
        unsafe { cublasZdotc_v2(bh.raw_handle(), n_pw, col as *const _, 1, col as *const _, 1, &mut nrm2 as *mut _ as *mut _); }
        eprintln!("[Diag-hpsi-step] after kinetic: |hpsi[{}]|²={:.6e}", lb, nrm2.x);
    }

    // 2. Zero grid, then scatter psi to FFT grid positions
    unsafe {
        stream
            .launch_builder(&kernels.zero_buffer)
            .arg(&mut *grid_dev)
            .arg(&(n_bands * grid_size))
            .launch(LaunchConfig::for_num_elems((n_bands * grid_size) as u32))
    }
    .map_err(Error::Cuda)?;

    // Nyquist: -1 if odd-sized (no Nyquist plane), N/2 if even.
    let nyq_x = if ngx % 2 == 0 { ngx / 2 } else { -1 };
    let nyq_y = if ngy % 2 == 0 { ngy / 2 } else { -1 };
    let nyq_z = if ngz % 2 == 0 { ngz / 2 } else { -1 };

    unsafe {
        stream
            .launch_builder(&kernels.scatter_pw_to_grid_nyq)
            .arg(&**psi_dev)
            .arg(fft_idx_dev)
            .arg(&mut *grid_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .arg(&grid_size)
            .arg(&ngy)
            .arg(&ngz)
            .arg(&nyq_x)
            .arg(&nyq_y)
            .arg(&nyq_z)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;

    // 3. Batched C2C IFFT (in-place)
    unsafe { c2c_inverse_inplace(fft_plan, grid_dev)?; }

    // 4. V_eff multiply: grid *= V_eff
    unsafe {
        stream
            .launch_builder(&kernels.veff_multiply)
            .arg(&mut *grid_dev)
            .arg(v_eff_dev)
            .arg(&grid_size)
            .arg(&n_bands)
            .launch(LaunchConfig::for_num_elems((n_bands * grid_size) as u32))
    }
    .map_err(Error::Cuda)?;

    // 5. Batched C2C FFT (in-place)
    unsafe { c2c_forward_inplace(fft_plan, grid_dev)?; }

    // 6. Gather: hpsi += grid / N_total
    // grid is const (read-only), hpsi is mutable (read-write for accumulation)
    unsafe {
        stream
            .launch_builder(&kernels.gather_add_kinetic)
            .arg(&*grid_dev)
            .arg(fft_idx_dev)
            .arg(&mut **hpsi_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .arg(&grid_size)
            .arg(&inv_ntotal)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    // Diag: |hpsi|² for last band after kinetic+Vloc
    #[cfg(feature = "scf_diag")]
    if let Some(bh) = _blas {
        let lb = n_bands as usize - 1;
        let (h_ptr, _) = (&*hpsi_dev).device_ptr(stream);
        let col = (h_ptr as *const CudaComplex).add(lb * n_pw as usize);
        let mut nrm2 = CudaComplex { x: 0.0, y: 0.0 };
        unsafe { cublasZdotc_v2(bh.raw_handle(), n_pw, col as *const _, 1, col as *const _, 1, &mut nrm2 as *mut _ as *mut _); }
        eprintln!("[Diag-hpsi-step] after kinetic+Vloc: |hpsi[{}]|²={:.6e}", lb, nrm2.x);
    }
    Ok(())
}

/// Apply the full Hamiltonian H|psi>. For Phase 2 this includes T + V_loc
/// Apply the full Hamiltonian H|psi>. Includes T + V_loc (FFT-based)
/// and V_NL (non-local pseudopotential via cuBLAS gemm).
#[builder]
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_full_hamiltonian(
    psi_dev: &PwCoefficients,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    hpsi_dev: &mut PwCoefficients,
    grid_dev: &mut CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        apply_v_loc_hamiltonian()
            .psi_dev(psi_dev)
            .hpsi_dev(hpsi_dev)
            .grid_dev(grid_dev)
            .kinetic_dev(kinetic_dev)
            .fft_idx_dev(fft_idx_dev)
            .v_eff_dev(v_eff_dev)
            .n_pw(n_pw as i32)
            .n_bands(n_bands as i32)
            .grid_size(grid_size as i32)
            .ngx(fft_plan.nx())
            .ngy(fft_plan.ny())
            .ngz(fft_plan.nz())
            .inv_ntotal(inv_ntotal)
            .fft_plan(fft_plan)
            .kernels(kernels)
            .stream(stream)
            .maybe_blas(Some(blas))
            .call()?;

        apply_v_nl_hamiltonian()
            .psi_dev(psi_dev)
            .hpsi_dev(hpsi_dev)
            .vnl_data(vnl_data)
            .n_bands(n_bands as i32)
            .n_pw(n_pw as i32)
            .blas(blas)
            .stream(stream)
            .call()?;

        // Diag: |hpsi|² for last band after full H (kinetic+Vloc+VNL)
        #[cfg(feature = "scf_diag")]
        {
            let lb = n_bands - 1;
            let (h_ptr, _) = (&*hpsi_dev).device_ptr(stream);
            let col = (h_ptr as *const CudaComplex).add(lb * n_pw);
            let mut nrm2 = CudaComplex { x: 0.0, y: 0.0 };
            unsafe { cublasZdotc_v2(blas.raw_handle(), n_pw as i32, col as *const _, 1, col as *const _, 1, &mut nrm2 as *mut _ as *mut _); }
            eprintln!("[Diag-hpsi-step] after kinetic+Vloc+VNL: |hpsi[{}]|²={:.6e}", lb, nrm2.x);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// V_NL (non-local pseudopotential) via cuBLAS gemm
// ---------------------------------------------------------------------------

/// Apply V_NL|psi> and accumulate into hpsi for one batch of ion projectors.
///
/// For each ion's (beta_g, d_matrix, n_expanded):
///   C_proj = beta^H . psi     (n_expanded x n_bands)
///   C_proj = D . C_proj       (n_expanded x n_bands)
///   hpsi   += beta . C_proj   (n_pw x n_bands, accumulated)
#[builder]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_v_nl_hamiltonian(
    psi_dev: &PwCoefficients,
    hpsi_dev: &mut PwCoefficients,
    vnl_data: &VnlBatchData,
    n_bands: i32,
    n_pw: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // C_proj = beta^H . psi  (n_expanded x n_bands)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros((ne * n_bands) as usize).map_err(Error::Cuda)?;

        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::C, // conj(beta^T)
                    transb: blas::op::N,
                    m: ne,
                    n: n_bands,
                    k: n_pw,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw, // beta_g is (ne, n_pw) row-major = col-major (n_pw, ne)
                    ldb: n_pw, // psi is (n_pw, n_bands) col-major
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_dev,
                &mut c_proj,
            )?;
        }

        // C_proj = D . C_proj  (n_expanded x n_bands)
        // Use a temp buffer since in-place gemm is not supported.
        let mut c_temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros((ne * n_bands) as usize).map_err(Error::Cuda)?;

        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::N,
                    transb: blas::op::N,
                    m: ne,
                    n: n_bands,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.d_matrix,
                &c_proj,
                &mut c_temp,
            )?;
        }
        std::mem::swap(&mut c_proj, &mut c_temp);

        // V_NL += beta . C_proj  (n_pw x n_bands, accumulated into hpsi)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::N,
                    transb: blas::op::N,
                    m: n_pw,
                    n: n_bands,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 }, // accumulate into hpsi
                    ldc: n_pw,
                },
                &entry.beta_g,
                &c_proj,
                hpsi_dev,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// S operator (USPP overlap)
// ---------------------------------------------------------------------------

/// Apply the USPP overlap matrix `S` to each band of `psi`.
///
/// S = I + Σ_I β_I · Q_I · β_I^H
///
/// For each ion with projectors `beta_g` and `q_matrix`:
///   p = beta_g^H · psi          (project, ne × n_bands)
///   q = q_matrix · p            (expand, ne × n_bands)
///   spsi += beta_g · q          (accumulate, n_pw × n_bands, α = +1)
///
/// The caller is responsible for copying `psi_dev` into `spsi_dev` first
/// (the identity term) before calling this to accumulate the β·Q·β^H·ψ correction.
#[builder]
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_s_times(
    psi_dev: &PwCoefficients,     // input ψ (n_pw × n_bands, col-major)
    spsi_dev: &mut PwCoefficients, // output S·ψ (caller pre-copies psi into this)
    vnl_data: &VnlBatchData,
    n_bands: i32,
    n_pw: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // p = beta_g^H · psi  (n_expanded × n_bands)
        let mut p: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::C,
                    transb: blas::op::N,
                    m: ne,
                    n: n_bands,
                    k: n_pw,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw,
                    ldb: n_pw,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_dev,
                &mut p,
            )?;
        }

        // q = q_matrix · p  (n_expanded × n_bands)
        let mut q: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::N,
                    transb: blas::op::N,
                    m: ne,
                    n: n_bands,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.q_matrix,
                &p,
                &mut q,
            )?;
        }

        // spsi += beta_g · q  (accumulate with α = +1)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::N,
                    transb: blas::op::N,
                    m: n_pw,
                    n: n_bands,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 },
                    ldc: n_pw,
                },
                &entry.beta_g,
                &q,
                spsi_dev,
            )?;
        }
    }
    Ok(())
}

