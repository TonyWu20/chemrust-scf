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
//   6. apply_s_inverse — S⁻¹ via Woodbury
//   7. check_s_inv_s_identity — diagnostic: verifies S⁻¹·S ≈ I

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::cusolver::sys::cublasOperation_t;

use crate::device::blas::{self, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;

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
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_v_loc_hamiltonian(
    psi_dev: &CudaSlice<CudaComplex>,
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    grid_dev: &mut CudaSlice<CudaComplex>,
    kinetic_dev: &CudaSlice<f64>,
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
    // 1. hpsi = kinetic * psi  (T|psi>)
    unsafe {
        stream
            .launch_builder(&kernels.init_kinetic)
            .arg(&mut *hpsi_dev)
            .arg(psi_dev)
            .arg(kinetic_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;

    // TEMPORARY: skip V_loc (FFT round-trip) to isolate kinetic+V_NL.
    // If V(pot) in Diag-HOp is unchanged, V_loc was never contributing.
    // If V(pot) changes (gets smaller/closer to 0), V_loc was contributing.

    // 2. Zero grid, then scatter psi to FFT grid positions
    unsafe {
        stream
            .launch_builder(&kernels.zero_buffer)
            .arg(&mut *grid_dev)
            .arg(&(n_bands * grid_size))
            .launch(LaunchConfig::for_num_elems((n_bands * grid_size) as u32))
    }
    .map_err(Error::Cuda)?;

    unsafe {
        stream
            .launch_builder(&kernels.scatter_pw_to_grid)
            .arg(psi_dev)
            .arg(fft_idx_dev)
            .arg(&mut *grid_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .arg(&grid_size)
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
            .arg(&mut *hpsi_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .arg(&grid_size)
            .arg(&inv_ntotal)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    Ok(())
}

/// Apply the full Hamiltonian H|psi>. For Phase 2 this includes T + V_loc
/// Apply the full Hamiltonian H|psi>. Includes T + V_loc (FFT-based)
/// and V_NL (non-local pseudopotential via cuBLAS gemm).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_full_hamiltonian(
    psi_dev: &CudaSlice<CudaComplex>,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &CudaSlice<f64>,
    fft_idx_dev: &CudaSlice<i32>,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    grid_dev: &mut CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        apply_v_loc_hamiltonian(
            psi_dev, hpsi_dev, grid_dev,
            kinetic_dev, fft_idx_dev, v_eff_dev,
            n_pw as i32, n_bands as i32, grid_size as i32, inv_ntotal,
            fft_plan, kernels, stream,
        )?;

        apply_v_nl_hamiltonian(
            psi_dev, hpsi_dev, vnl_data,
            n_bands as i32, n_pw as i32,
            blas, stream,
        )?;
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
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_v_nl_hamiltonian(
    psi_dev: &CudaSlice<CudaComplex>,
    hpsi_dev: &mut CudaSlice<CudaComplex>,
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
// S⁻¹ (USPP overlap inverse via Woodbury)
// ---------------------------------------------------------------------------

/// Apply the USPP overlap inverse to each band of `hpsi`.
///
/// Uses the global Woodbury formula (PHASE_PLAN.md):
///   S⁻¹·v = v − B · M⁻¹ · (B^H · v)
///   where M = Q⁻¹ + B^H·B (Cholesky-factored in VnlBatchData::precompute).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_s_inverse(
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    n_bands: i32,
    n_pw: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    solver: &SolverHandle,
) -> Result<(), Error> {
    let nte = vnl_data.n_total_expanded;

    // 1. temp = B^H · hpsi  (nte × n_bands)
    let mut temp: CudaSlice<CudaComplex> =
        stream.alloc_zeros(nte as usize * n_bands as usize).map_err(Error::Cuda)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: blas::op::C,
                transb: blas::op::N,
                m: nte,
                n: n_bands,
                k: n_pw,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw,
                ldb: n_pw,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: nte,
            },
            &vnl_data.b_concat,
            hpsi_dev,
            &mut temp,
        )?;
    }

    // 2. Solve M·x = temp via LU factor (zgetrs, in-place overwrites temp).
    let mut info_dev = stream.alloc_zeros::<i32>(1).map_err(Error::Cuda)?;
    solver.zgetrs(
        cublasOperation_t::CUBLAS_OP_N,
        nte,
        n_bands,
        &vnl_data.lu_m,
        &vnl_data.lu_ipiv,
        &mut temp,
        &mut info_dev,
    )?;

    // 3. hpsi −= B · x  (accumulate with α = −1)
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: blas::op::N,
                transb: blas::op::N,
                m: n_pw,
                n: n_bands,
                k: nte,
                alpha: CudaComplex { x: -1.0, y: 0.0 },
                lda: n_pw,
                ldb: nte,
                beta: CudaComplex { x: 1.0, y: 0.0 },
                ldc: n_pw,
            },
            &vnl_data.b_concat,
            &temp,
            hpsi_dev,
        )?;
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
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_s_times(
    psi_dev: &CudaSlice<CudaComplex>,     // input ψ (n_pw × n_bands, col-major)
    spsi_dev: &mut CudaSlice<CudaComplex>, // output S·ψ (caller pre-copies psi into this)
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

// ---------------------------------------------------------------------------
// S⁻¹·S identity diagnostic
// ---------------------------------------------------------------------------

#[doc(hidden)]
pub fn check_s_inv_s_identity(
    psi_host: &[num_complex::Complex64],
    n_pw: usize,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    solver: &SolverHandle,
) -> Result<f64, Error> {
    use crate::device::blas::op;
    let n = n_pw as i32;
    let psi_cuda: Vec<CudaComplex> = psi_host
        .iter()
        .map(|&c| CudaComplex { x: c.re, y: c.im })
        .collect();
    let psi_dev: CudaSlice<CudaComplex> = stream
        .clone_htod(&psi_cuda).map_err(Error::Cuda)?;

    // 1. Build S·psi = psi + Σ β_g · q_matrix · (β_g^H · psi)
    let mut spsi_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    stream.memcpy_dtod(&psi_dev, &mut spsi_dev).map_err(Error::Cuda)?;

    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // c_proj = β_g^H · psi  (ne × 1)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemv_c64(
                op::C, n, ne,
                CudaComplex { x: 1.0, y: 0.0 },
                &entry.beta_g, n,
                &psi_dev, 1,
                CudaComplex { x: 0.0, y: 0.0 },
                &mut c_proj, 1,
            ).map_err(Error::Blas)?;
        }

        // temp = q_matrix · c_proj  (ne × 1)
        let mut temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemv_c64(
                op::N, ne, ne,
                CudaComplex { x: 1.0, y: 0.0 },
                &entry.q_matrix, ne,
                &c_proj, 1,
                CudaComplex { x: 0.0, y: 0.0 },
                &mut temp, 1,
            ).map_err(Error::Blas)?;
        }

        // spsi += β_g · temp  (n_pw × 1)
        unsafe {
            blas.gemv_c64(
                op::N, n, ne,
                CudaComplex { x: 1.0, y: 0.0 },
                &entry.beta_g, n,
                &temp, 1,
                CudaComplex { x: 1.0, y: 0.0 },
                &mut spsi_dev, 1,
            ).map_err(Error::Blas)?;
        }
    }

    // 2. Apply S⁻¹ to spsi
    unsafe {
        apply_s_inverse(
            &mut spsi_dev, vnl_data, 1, n, blas, stream, solver,
        )?;
    }

    // 3. D2H and compute max residual ‖spsi − psi‖_∞
    let result: Vec<CudaComplex> = stream.clone_dtoh(&spsi_dev).map_err(Error::Cuda)?;
    let max_residual = psi_host.iter().zip(result.iter())
        .map(|(&p, &r)| {
            let dr = r.x - p.re;
            let di = r.y - p.im;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0_f64, f64::max);

    Ok(max_residual)
}
