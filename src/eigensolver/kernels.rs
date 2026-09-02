// ---------------------------------------------------------------------------
// CUDA kernel compilation for DFT SCF diagonalization (NVRTC kernels)
// ---------------------------------------------------------------------------

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::compile_ptx_with_opts;
use cudarc::nvrtc::CompileOptions;

use crate::types::Error;

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled via NVRTC at startup)
// ---------------------------------------------------------------------------

const CUDA_KERNEL_SRC: &str = "
extern \"C\" __global__ void zero_buffer(double2* buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) { buf[i].x = 0.0; buf[i].y = 0.0; }
}

extern \"C\" __global__ void copy_buffer(
    double2* dst, const double2* src, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
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

extern \"C\" __global__ void scatter_pw_to_grid_nyq(
    const double2* psi, const int* fft_idx,
    double2* grid, int n_pw, int n_bands, int grid_size,
    int ngy, int ngz, int nyq_x, int nyq_y, int nyq_z
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
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (r < grid_size) {
        double sum = 0.0;
        for (int b = 0; b < n_bands; b++) {
            double2 psi = psi_r[b * grid_size + r];
            sum += occ[b] * (psi.x * psi.x + psi.y * psi.y);
        }
        rho[r] = sum * inv_omega;
        r += stride;
    }
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

// dst[b*n_pw + g] += alpha * src[b*n_pw + g] * scale[b]
// Used for: Y·Λ_Y term (Step 3), S·X·Λ subtraction (Step 1), X·Λ_Y reconstruction (Step 4)
extern \"C\" __global__ void band_scale_axpy(
    double2* dst,
    const double2* src,
    const double* scale,
    double alpha,
    int n_pw, int n_bands
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_pw * n_bands;
    if (idx >= total) return;
    int b = idx / n_pw;
    double s = alpha * scale[b];
    dst[idx].x += s * src[idx].x;
    dst[idx].y += s * src[idx].y;
}

// a[i] *= b[i]  (element-wise complex multiply, in-place)
extern \"C\" __global__ void cpx_mul_inplace(
    double2* a, const double2* b, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        double ax = a[i].x, ay = a[i].y;
        double bx = b[i].x, by = b[i].y;
        a[i].x = ax * bx - ay * by;
        a[i].y = ax * by + ay * bx;
    }
}

// dst[i] = a[i] * conj(b[i])  (element-wise complex multiply with conjugate)
extern \"C\" __global__ void cpx_conj_mul(
    double2* dst, const double2* a, const double2* b, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        double ax = a[i].x, ay = a[i].y;
        double bx = b[i].x, by = b[i].y;
        dst[i].x = ax * bx + ay * by;   // Re(a * conj(b))
        dst[i].y = ay * bx - ax * by;   // Im(a * conj(b))
    }
}

// Column scaling: out[r + c*nrow] *= eig[c]
// Used in USPP preconditioner PASS 2 to scale beta_phi_psi by eigenvalues.
// Matches CASTEP nlpot.f90:16100-16104 (E_beta = eigenvalue * beta_phi)
extern \"C\" __global__ void scale_cols_by_eig(
    double2* data,
    const double* eig,
    int nrow,
    int ncol
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = nrow * ncol;
    for (int i = tid; i < total; i += stride) {
        int c = i / nrow;
        double e = eig[c];
        data[i].x *= e;
        data[i].y *= e;
    }
}

// Fused Chebyshev three-term recurrence step (grid-stride):
//   dst[i] = a * (h[i] - c * cur[i]) - prev[i]
// Replaces the per-step sequence {D2D copy h->dst; D2D copy cur->tmp;
// Zscal(tmp,-c); Zaxpy(dst+=tmp); Zscal(dst,a); Zaxpy(dst+=-prev)}
// (7 full memory passes) with a single fused pass.
extern \"C\" __global__ void cheb_step_combine(
    double2* dst, const double2* h, const double2* cur, const double2* prev,
    double c, double a, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        double hx = h[i].x, hy = h[i].y;
        double cx = cur[i].x, cy = cur[i].y;
        double u0 = hx - c * cx;
        double u1 = hy - c * cy;
        dst[i].x = a * u0 - prev[i].x;
        dst[i].y = a * u1 - prev[i].y;
    }
}

// Fused Chebyshev T_1 initialization step (grid-stride):
//   dst[i] = a * (src[i] - c * ref[i])
// Replaces {Zscal(dst,-c*ref); Zaxpy(dst+=src); Zscal(dst,a)}.
extern \"C\" __global__ void cheb_shift_scale(
    double2* dst, const double2* src, const double2* ref,
    double c, double a, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        double u0 = src[i].x - c * ref[i].x;
        double u1 = src[i].y - c * ref[i].y;
        dst[i].x = a * u0;
        dst[i].y = a * u1;
    }
}

// Per-band squared-norm partial reduction for Woodbury S^{-1} refinement
// convergence check.  input is nte x n_bands column-major (stride nte).
// out_r2[b] = sum_i |r[i + b*nte]|^2,  out_p2[b] = sum_i |p[i + b*nte]|^2
// (both in one launch; either output may be the same buffer as the other
// call site wants — pass the two device pointers explicitly).
// Launch with gridDim = n_bands, blockDim = 128; shared-memory reduce.
extern \"C\" __global__ void band_sqnorms(
    const double2* r, const double2* p, double* out_r2, double* out_p2,
    int nte, int n_bands
) {
    extern __shared__ double sh[];  // 2 * blockDim.x
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    double r2 = 0.0, p2 = 0.0;
    for (int i = tid; i < nte; i += nthreads) {
        double2 rv = r[i + b * nte];
        double2 pv = p[i + b * nte];
        r2 += rv.x * rv.x + rv.y * rv.y;
        p2 += pv.x * pv.x + pv.y * pv.y;
    }
    // Block-reduce r2 into sh[0..nthreads), p2 into sh[nthreads..2*nthreads)
    sh[tid] = r2;
    sh[nthreads + tid] = p2;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] += sh[tid + s];
            sh[nthreads + tid] += sh[nthreads + tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_r2[b] = sh[0];
        out_p2[b] = sh[nthreads];
    }
}
";

// ---------------------------------------------------------------------------
// Compiled kernels handle
// ---------------------------------------------------------------------------

/// Handles to all compiled CUDA kernels used in the Chebyshev filter.
#[doc(hidden)]
pub struct CudaKernelSet {
    pub(crate) zero_buffer: CudaFunction,
    #[allow(dead_code)]
    pub(crate) zero_buffer_real: CudaFunction,
    pub(crate) init_kinetic: CudaFunction,
    pub(crate) scatter_pw_to_grid: CudaFunction,
    pub(crate) scatter_pw_to_grid_nyq: CudaFunction,
    pub(crate) accumulate_density: CudaFunction,
    pub(crate) veff_multiply: CudaFunction,
    pub(crate) gather_add_kinetic: CudaFunction,
    #[allow(dead_code)]
    pub(crate) transpose_col_to_row: CudaFunction,
    #[allow(dead_code)]
    pub(crate) transpose_row_to_col: CudaFunction,
    pub(crate) cpx_mul_inplace: CudaFunction,
    pub(crate) cpx_conj_mul: CudaFunction,
    pub(crate) scale_cols_by_eig: CudaFunction,
    pub(crate) band_scale_axpy: CudaFunction,
    pub(crate) copy_buffer: CudaFunction,
    pub(crate) cheb_step_combine: CudaFunction,
    pub(crate) cheb_shift_scale: CudaFunction,
    pub(crate) band_sqnorms: CudaFunction,
}

impl CudaKernelSet {
    #[doc(hidden)]
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Error> {
        // NVRTC (CUDA 12.9) has no -O flag; optimization is fixed. The
        // arch flag selects the code target:
        //   "sm_120"  -> native Blackwell SASS, no JIT (RTX PRO 5000).
        //                Use "compute_61" (PTX) to keep Pascal GPUs
        //                runnable; omit it entirely for max portability
        //                (NVRTC default emits PTX targeting sm_52).
        const NVRTC_GPU_ARCH: &str = "sm_120";
        let opts = CompileOptions {
            arch: Some(NVRTC_GPU_ARCH),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(CUDA_KERNEL_SRC, opts).map_err(|e| Error::Nvrtc(e.to_string()))?;
        let module: Arc<CudaModule> = ctx.load_module(ptx).map_err(Error::Cuda)?;
        let load = |name: &str| -> Result<CudaFunction, Error> {
            module.load_function(name).map_err(Error::Cuda)
        };
        Ok(Self {
            zero_buffer: load("zero_buffer")?,
            zero_buffer_real: load("zero_buffer_real")?,
            init_kinetic: load("init_kinetic")?,
            scatter_pw_to_grid: load("scatter_pw_to_grid")?,
            scatter_pw_to_grid_nyq: load("scatter_pw_to_grid_nyq")?,
            accumulate_density: load("accumulate_density")?,
            veff_multiply: load("veff_multiply")?,
            gather_add_kinetic: load("gather_add_kinetic")?,
            transpose_col_to_row: load("transpose_col_to_row")?,
            transpose_row_to_col: load("transpose_row_to_col")?,
            cpx_mul_inplace: load("cpx_mul_inplace")?,
            cpx_conj_mul: load("cpx_conj_mul")?,
            scale_cols_by_eig: load("scale_cols_by_eig")?,
            band_scale_axpy: load("band_scale_axpy")?,
            copy_buffer: load("copy_buffer")?,
            cheb_step_combine: load("cheb_step_combine")?,
            cheb_shift_scale: load("cheb_shift_scale")?,
            band_sqnorms: load("band_sqnorms")?,
        })
    }
}
