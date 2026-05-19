// ---------------------------------------------------------------------------
// Chebyshev polynomial filtering for DFT SCF diagonalization
// ---------------------------------------------------------------------------
//
// Implements:
//   1. SpectralBounds estimation (lambda_max, eps_cut, center, half_width)
//   2. CUDA kernel compilation (NVRTC) for H|psi> operations
//   3. apply_full_hamiltonian() — T + V_loc (FFT-based) on GPU
//   4. apply_scaled_hamiltonian() — sigma(H).psi
//   5. chebyshev_filter() — main driver: recurrence + norm check

use std::marker::PhantomData;
use std::sync::Arc;

use chemrust_hamiltonian_core::{CellGeometry, GVectorGrid, PseudopotentialSet};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::device::blas::{self, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::{CudaComplex, Gpu};
use crate::eigensolver::vnl_data::VnlBatchData;

// ---------------------------------------------------------------------------
// Helper: call cuFFT C2C in-place (same buffer for input and output)
// ---------------------------------------------------------------------------

/// Call `c2c_inverse` in-place. cuFFT natively supports in-place transforms,
/// so passing the same `&mut` twice via raw pointer is correct.
pub(crate) unsafe fn c2c_inverse_inplace(
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
use crate::layout::{ColumnDistributed, RowDistributed, WavefunctionSet};
use crate::types::{Error, KPoint};

// ---------------------------------------------------------------------------
// Type alias for the complex Chebyshev return type
// ---------------------------------------------------------------------------

/// Returns (psi_row, hpsi_row) — both in RowDistributed layout.
type ChebyshevResult = Result<
    (
        Gpu<WavefunctionSet<RowDistributed>>,
        Gpu<WavefunctionSet<RowDistributed>>,
    ),
    Error,
>;

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled via NVRTC at startup)
// ---------------------------------------------------------------------------

const CUDA_KERNEL_SRC: &str = "
extern \"C\" __global__ void zero_buffer(double2* buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) { buf[i].x = 0.0; buf[i].y = 0.0; }
}

extern \"C\" __global__ void zero_buffer_real(double* buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) { buf[i] = 0.0; }
}

extern \"C\" __global__ void init_kinetic(
    double2* hpsi, const double2* psi, const double* kinetic,
    int n_pw, int n_bands
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int g = tid % n_pw;
        double k = kinetic[g];
        hpsi[tid].x = psi[tid].x * k;
        hpsi[tid].y = psi[tid].y * k;
        tid += stride;
    }
}

extern \"C\" __global__ void scatter_pw_to_grid(
    const double2* psi, const int* fft_idx,
    double2* grid, int n_pw, int n_bands, int grid_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int b = tid / n_pw;
        int g = tid % n_pw;
        grid[b * grid_size + fft_idx[g]] = psi[b * n_pw + g];
        tid += stride;
    }
}

extern \"C\" __global__ void veff_multiply(
    double2* grid, const double* veff,
    int grid_size, int n_bands
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * grid_size) {
        int r = tid % grid_size;
        double v = veff[r];
        grid[tid].x *= v;
        grid[tid].y *= v;
        tid += stride;
    }
}

extern \"C\" __global__ void gather_add_kinetic(
    const double2* grid, const int* fft_idx,
    double2* result, int n_pw, int n_bands, int grid_size, double inv_ntotal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int b = tid / n_pw;
        int g = tid % n_pw;
        double2 v = grid[b * grid_size + fft_idx[g]];
        v.x *= inv_ntotal;
        v.y *= inv_ntotal;
        result[tid].x += v.x;
        result[tid].y += v.y;
        tid += stride;
    }
}

extern \"C\" __global__ void transpose_col_to_row(
    const double2* col, double2* row, int n_bands, int n_pw
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int b = tid / n_pw;
        int g = tid % n_pw;
        row[g * n_bands + b] = col[b * n_pw + g];
        tid += stride;
    }
}

extern \"C\" __global__ void accumulate_density(
    const double2* psi_r, const double* occ,
    double* rho, int n_bands, int grid_size, double inv_omega
) {
    extern __shared__ double sdata[];
    int r = blockIdx.x;
    if (r >= grid_size) return;
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int b = tid; b < n_bands; b += blockDim.x) {
        double2 psi = psi_r[b * grid_size + r];
        sum += occ[b] * (psi.x * psi.x + psi.y * psi.y);
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) rho[r] = sdata[0] * inv_omega;
}

extern \"C\" __global__ void transpose_row_to_col(
    const double2* row, double2* col, int n_bands, int n_pw
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int b = tid / n_pw;
        int g = tid % n_pw;
        col[b * n_pw + g] = row[g * n_bands + b];
        tid += stride;
    }
}
";

// ---------------------------------------------------------------------------
// Compiled kernels handle
// ---------------------------------------------------------------------------

/// Handles to all compiled CUDA kernels used in the Chebyshev filter.
pub(crate) struct CudaKernelSet {
    pub(crate) zero_buffer: CudaFunction,
    #[allow(dead_code)]
    pub(crate) zero_buffer_real: CudaFunction, // reserved for future real-buffer clearing
    pub(crate) init_kinetic: CudaFunction,
    pub(crate) scatter_pw_to_grid: CudaFunction,
    pub(crate) accumulate_density: CudaFunction,
    pub(crate) veff_multiply: CudaFunction,
    pub(crate) gather_add_kinetic: CudaFunction,
    pub(crate) transpose_col_to_row: CudaFunction,
    pub(crate) transpose_row_to_col: CudaFunction,
}

impl CudaKernelSet {
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self, Error> {
        let ptx = compile_ptx(CUDA_KERNEL_SRC).map_err(|e| Error::Nvrtc(e.to_string()))?;
        let module: Arc<CudaModule> = ctx.load_module(ptx).map_err(Error::Cuda)?;
        let load = |name: &str| -> Result<CudaFunction, Error> {
            module.load_function(name).map_err(Error::Cuda)
        };
        Ok(Self {
            zero_buffer: load("zero_buffer")?,
            zero_buffer_real: load("zero_buffer_real")?,
            init_kinetic: load("init_kinetic")?,
            scatter_pw_to_grid: load("scatter_pw_to_grid")?,
            accumulate_density: load("accumulate_density")?,
            veff_multiply: load("veff_multiply")?,
            gather_add_kinetic: load("gather_add_kinetic")?,
            transpose_col_to_row: load("transpose_col_to_row")?,
            transpose_row_to_col: load("transpose_row_to_col")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Spectral Bounds
// ---------------------------------------------------------------------------

/// Estimated spectral bounds for the Hamiltonian at this SCF iteration.
#[allow(dead_code)]
pub(crate) struct SpectralBounds {
    pub lambda_max: f64,
    pub eps_cut: f64,
    pub center: f64,
    pub half_width: f64,
}

/// Estimate spectral bounds for the Hamiltonian.
///
/// * First call (no eigenvalues yet): rough estimate from kinetic + potential.
/// * Subsequent calls: use eigenvalue spectrum with guard band.
pub(crate) fn compute_spectral_bounds(
    eigenvalues: Option<&[f64]>,
    wave_grid: &GVectorGrid,
    min_veff: f64,
    max_veff: f64,
) -> Result<SpectralBounds, Error> {
    let gmax = wave_grid.gmax();
    let kinetic_max = 0.5 * gmax * gmax;

    let (lambda_max, eps_cut) = match eigenvalues {
        None | Some([]) => {
            let lm = kinetic_max + (max_veff - min_veff);
            (lm, lm / 3.0)
        }
        Some(eig) => {
            let e0 = eig[0];
            let e_last = eig[eig.len() - 1];
            let lm = kinetic_max + (max_veff - min_veff);
            let raw_eps = e_last + 0.2 * (e_last - e0);
            (lm, raw_eps.min(lm * 0.95))
        }
    };

    Ok(SpectralBounds {
        lambda_max,
        eps_cut,
        center: (lambda_max + eps_cut) / 2.0,
        half_width: (lambda_max - eps_cut) / 2.0,
    })
}

// ---------------------------------------------------------------------------
// Precomputed FFT metadata (uploaded to GPU)
// ---------------------------------------------------------------------------

/// Kinetic energy ½|G|² for each grid point (Hartree atomic units).
fn compute_kinetic_energies(wave_grid: &GVectorGrid) -> Vec<f64> {
    wave_grid.g2().iter().map(|g2| 0.5 * g2).collect()
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
unsafe fn apply_v_loc_hamiltonian(
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
unsafe fn apply_full_hamiltonian(
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
unsafe fn apply_v_nl_hamiltonian(
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
// Scaled Hamiltonian
// ---------------------------------------------------------------------------

/// Compute sigma(H).psi = (H.psi - c*psi) / e in-place on hpsi_dev.
fn apply_scaled_hamiltonian_inplace(
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    psi_dev: &CudaSlice<CudaComplex>,
    n: i32,
    center: f64,
    half_width: f64,
    blas: &BlasHandle,
) -> Result<(), Error> {
    let inv_e = 1.0 / half_width;
    let neg_c_over_e = -center / half_width;

    // Step 1: hpsi *= (1/e)  via zscal
    let alpha_e = CudaComplex { x: inv_e, y: 0.0 };
    unsafe {
        let (ptr, _) = hpsi_dev.device_ptr_mut(blas.stream());
        cudarc::cublas::sys::cublasZscal_v2(
            blas.raw_handle(),
            n,
            &alpha_e as *const _ as *const _,
            ptr as *mut _,
            1,
        )
        .result()
        .map_err(Error::Blas)?;
    }

    // Step 2: hpsi += (-c/e) * psi  via axpy
    let alpha_c = CudaComplex { x: neg_c_over_e, y: 0.0 };
    blas.axpy_c64(n, alpha_c, psi_dev, 1, hpsi_dev, 1)
        .map_err(Error::Blas)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Norm helper
// ---------------------------------------------------------------------------

/// Compute Frobenius norm via cuBLAS dot product: sqrt(conj(x).x)
fn compute_frobenius_norm(buf: &CudaSlice<CudaComplex>, n: i32, blas: &BlasHandle) -> Result<f64, Error> {
    let dot = blas.dotc_c64(n, buf, 1, buf, 1).map_err(Error::Blas)?;
    Ok(dot.x.sqrt())
}

/// Check that norm growth between consecutive iterations does not exceed 10x.
fn check_norm_stability(norm_curr: f64, norm_prev: f64, iteration: usize) -> Result<(), Error> {
    let growth_factor = if norm_prev > 0.0 { norm_curr / norm_prev } else { 1.0 };
    if growth_factor > 10.0 {
        return Err(Error::ChebyshevDiverged {
            iteration,
            norm_previous: norm_prev,
            norm_current: norm_curr,
            growth_factor,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Transpose helper
// ---------------------------------------------------------------------------

/// Transpose ColumnDistributed layout -> RowDistributed layout on GPU.
unsafe fn transpose_col_to_row_on_gpu(
    col_dev: &CudaSlice<CudaComplex>,
    row_dev: &mut CudaSlice<CudaComplex>,
    n_bands: i32,
    n_pw: i32,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        stream
            .launch_builder(&kernels.transpose_col_to_row)
            .arg(col_dev)
            .arg(&mut *row_dev)
            .arg(&n_bands)
            .arg(&n_pw)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    Ok(())
}

/// Transpose RowDistributed layout -> ColumnDistributed layout on GPU.
pub(crate) unsafe fn transpose_row_to_col_on_gpu(
    row_dev: &CudaSlice<CudaComplex>,
    col_dev: &mut CudaSlice<CudaComplex>,
    n_bands: i32,
    n_pw: i32,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        stream
            .launch_builder(&kernels.transpose_row_to_col)
            .arg(row_dev)
            .arg(&mut *col_dev)
            .arg(&n_bands)
            .arg(&n_pw)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Chebyshev filter — main entry point
// ---------------------------------------------------------------------------

/// Apply Chebyshev polynomial filtering to the wavefunctions.
///
/// Returns (psi_row, hpsi_row) where:
/// - `psi_row` is the filtered wavefunction in RowDistributed layout
/// - `hpsi_row` is H|psi> in RowDistributed layout (needed by Rayleigh-Ritz)
///
/// `ndeg` is the Chebyshev polynomial degree.
/// `eigenvalues` is `None` on the first SCF iteration, `Some(&[...])` thereafter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chebyshev_filter(
    psi_gpu: &Gpu<WavefunctionSet<ColumnDistributed>>,
    v_eff_gpu: &Gpu<crate::types::EffectivePotential>,
    _pots: &PseudopotentialSet,
    wave_grid: &GVectorGrid,
    _k_point: &KPoint,
    _cell: &CellGeometry,
    vnl_data: &VnlBatchData,
    fft_idx_dev: &CudaSlice<i32>,      // PW-to-FFT-grid index map (length = n_pw)
    min_veff: f64,
    max_veff: f64,
    kernels: &CudaKernelSet,            // pre-compiled GPU kernels (shared)
    eigenvalues: Option<&[f64]>,
    ndeg: usize,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> ChebyshevResult {
    // ---- Dimensions ----
    let n_bands = psi_gpu.shape()[0];
    let n_pw = psi_gpu.shape()[1];
    let n_elem = n_bands * n_pw;
    let n_elem_i32 = n_elem as i32;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;

    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = ngx * ngy * ngz;
    let inv_ntotal = 1.0 / (grid_size as f64);
    let grid_alloc = n_bands * grid_size;

    // ---- Precompute & upload kinetic energy ----
    let kinetic_cpu = compute_kinetic_energies(wave_grid);
    let kinetic_dev: CudaSlice<f64> = stream.clone_htod(&kinetic_cpu).map_err(Error::Cuda)?;

    // ---- FFT plan (batched C2C) ----
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32, n_bands_i32, stream.clone(),
    )?;

    // ---- GPU workspace buffers ----
    let v_eff_dev = v_eff_gpu.as_device_slice();
    let psi_input = psi_gpu.as_device_slice().clone();

    // Three wavefunction buffers for the recurrence
    let mut buf_a: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut buf_b: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut buf_c: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;

    // Hamiltonian workspace
    let mut hpsi_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut grid_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // Output RowDistributed buffers
    let mut psi_row_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut hpsi_row_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;

    // ---- Spectral bounds ----
    let bounds = compute_spectral_bounds(eigenvalues, wave_grid, min_veff, max_veff)?;

    // ---- Chebyshev recurrence ----
    //
    // psi_0 = psi_input (ColumnDistributed)
    // psi_1 = sigma(H) . psi_0
    // For k = 2..ndeg:
    //   psi_k = 2 * sigma(H) . psi_{k-1} - psi_{k-2}
    //
    // Buffers: buf_a = psi_k-2, buf_b = psi_k-1, buf_c = psi_k

    // Copy psi_input into buf_a as psi_0, buf_b will be psi_1
    stream.memcpy_dtod(&psi_input, &mut buf_a).map_err(Error::Cuda)?;

    if ndeg >= 1 {
        // hpsi = H.psi_0
        unsafe {
            apply_full_hamiltonian(
                &buf_a, v_eff_dev, &kinetic_dev, fft_idx_dev,
                n_pw, n_bands, grid_size, inv_ntotal,
                &fft_plan, &mut hpsi_dev, &mut grid_dev, vnl_data, blas, kernels, stream,
            )?;
        }

        // sigma(H).psi_0 = (H.psi_0 - c*psi_0) / e
        apply_scaled_hamiltonian_inplace(
            &mut hpsi_dev, &buf_a, n_elem_i32,
            bounds.center, bounds.half_width, blas,
        )?;

        // Copy to buf_b = psi_1
        stream.memcpy_dtod(&hpsi_dev, &mut buf_b).map_err(Error::Cuda)?;
    }

    // Higher iterations
    for k in 2..=ndeg {
        // hpsi = H.psi_{k-1} (psi_{k-1} is in buf_b)
        unsafe {
            apply_full_hamiltonian(
                &buf_b, v_eff_dev, &kinetic_dev, fft_idx_dev,
                n_pw, n_bands, grid_size, inv_ntotal,
                &fft_plan, &mut hpsi_dev, &mut grid_dev, vnl_data, blas, kernels, stream,
            )?;
        }

        // sigma(H).psi_{k-1}
        apply_scaled_hamiltonian_inplace(
            &mut hpsi_dev, &buf_b, n_elem_i32,
            bounds.center, bounds.half_width, blas,
        )?;

        // psi_k = 2 * sigma(H).psi_{k-1} - psi_{k-2}
        // hpsi_dev now = sigma(H).psi_{k-1}
        // buf_a = psi_{k-2}
        // Write result into buf_c

        // buf_c = 2 * hpsi_dev (copy hpsi_dev to buf_c, then scal by 2)
        stream.memcpy_dtod(&hpsi_dev, &mut buf_c).map_err(Error::Cuda)?;

        let two = CudaComplex { x: 2.0, y: 0.0 };
        unsafe {
            let (ptr, _) = buf_c.device_ptr_mut(stream);
            cudarc::cublas::sys::cublasZscal_v2(
                blas.raw_handle(),
                n_elem_i32,
                &two as *const _ as *const _,
                ptr as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;
        }

        // buf_c = buf_c - psi_{k-2} = 2*sigma(H).psi_{k-1} - psi_{k-2}
        let neg_one = CudaComplex { x: -1.0, y: 0.0 };
        blas.axpy_c64(n_elem_i32, neg_one, &buf_a, 1, &mut buf_c, 1)
            .map_err(Error::Blas)?;

        // Norm stability check
        let norm_prev = compute_frobenius_norm(&buf_b, n_elem_i32, blas)?;
        let norm_curr = compute_frobenius_norm(&buf_c, n_elem_i32, blas)?;
        check_norm_stability(norm_curr, norm_prev, k)?;

        // Rotate: buf_a = psi_{k-2} → becomes buf_b for next iter? No:
        // psi_{k-2} → buf_a (old), psi_{k-1} → buf_b (old), psi_k → buf_c (new)
        // Next iteration needs: psi_{k-1} → buf_a, psi_k → buf_b
        std::mem::swap(&mut buf_a, &mut buf_b);
        std::mem::swap(&mut buf_b, &mut buf_c);
    }

    // After loop:
    // If ndeg >= 1: psi_ndeg is in buf_b
    // If ndeg == 0: psi_0 is in buf_a (or psi_input)
    // Normalize: if ndeg >= 1, final is buf_b; else final is buf_a
    let final_psi: &CudaSlice<CudaComplex> = if ndeg >= 1 { &buf_b } else { &buf_a };

    // Compute final H|psi> for Rayleigh-Ritz
    unsafe {
        apply_full_hamiltonian(
            final_psi, v_eff_dev, &kinetic_dev, fft_idx_dev,
            n_pw, n_bands, grid_size, inv_ntotal,
            &fft_plan, &mut hpsi_dev, &mut grid_dev, vnl_data, blas, kernels, stream,
        )?;
    }

    // Transpose both to RowDistributed
    unsafe {
        transpose_col_to_row_on_gpu(
            final_psi, &mut psi_row_dev, n_bands_i32, n_pw_i32, kernels, stream,
        )?;
        transpose_col_to_row_on_gpu(
            &hpsi_dev, &mut hpsi_row_dev, n_bands_i32, n_pw_i32, kernels, stream,
        )?;
    }

    // Wrap into Gpu<WavefunctionSet<L>>
    let psi_row = Gpu::<WavefunctionSet<RowDistributed>> {
        slice: psi_row_dev,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };
    let hpsi_row = Gpu::<WavefunctionSet<RowDistributed>> {
        slice: hpsi_row_dev,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };

    Ok((psi_row, hpsi_row))
}
