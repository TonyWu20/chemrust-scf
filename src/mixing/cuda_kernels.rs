// ---------------------------------------------------------------------------
// CUDA kernels for mixing element-wise operations (compiled via NVRTC)
// ---------------------------------------------------------------------------

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream,
};
use cudarc::nvrtc::compile_ptx_with_opts;
use cudarc::nvrtc::CompileOptions;

use crate::device::blas::BlasHandle;
use crate::types::Error;

/// CUDA kernel source for mixing element-wise complex operations.
const CUDA_KERNEL_SRC: &str = "
extern \"C\" __global__ void cpx_sub(
    double2* dst, const double2* a, const double2* b, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i].x = a[i].x - b[i].x;
        dst[i].y = a[i].y - b[i].y;
    }
}

extern \"C\" __global__ void cpx_full_update(
    double2* dst,
    const double2* n_in,
    const double* kerker,
    const double2* r_curr,
    const double2* sum_delta_r,
    const double2* sum_delta_n,
    const double2* n_out,
    const double* mask,
    int n,
    double amp_k,
    double amp_n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        if (mask[i] > 0.5) {
            // Mixing basis component (G within the CASTEP mix cutoff, incl. G=0).
            // CASTEP dm_sub_mix.f90 dm_mix_density_pulay exact form:
            //   new = n_in + sum_dn + K*(R + sum_dr)
            // K carries mix_charge_amp (scaled by amp_k here; the kerker
            // slice holds the PURE G2/(G2+q2) kernel). The DIIS density
            // part (sum_dn) is unscaled (amp_n = 1.0).
            double r_plus_x = r_curr[i].x + sum_delta_r[i].x;
            double r_plus_y = r_curr[i].y + sum_delta_r[i].y;
            double k = kerker[i];
            dst[i].x = n_in[i].x + amp_n * sum_delta_n[i].x + amp_k * k * r_plus_x;
            dst[i].y = n_in[i].y + amp_n * sum_delta_n[i].y + amp_k * k * r_plus_y;
        } else {
            // High-G content: CASTEP dm_mix_density_to_density carries the
            // above-cutoff components from the FRESH output density of this
            // cycle (dencut keeps the high-frequency FFT components of the
            // current wavefunction density). Use n_out, not n_in.
            dst[i].x = n_out[i].x;
            dst[i].y = n_out[i].y;
        }
    }
}

extern \"C\" __global__ void cpx_zero(double2* buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) { buf[i].x = 0.0; buf[i].y = 0.0; }
}

extern \"C\" __global__ void cpx_mask(
    double2* dst, const double2* a, const double* mask, int n
) {
    // Band-limit a complex array to the CASTEP mix basis (mask = 1.0
    // inside the cutoff incl. G=0, 0.0 above).
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i].x = a[i].x * mask[i];
        dst[i].y = a[i].y * mask[i];
    }
}
";

/// Compiled GPU kernels for mixing element-wise operations.
///
/// Created lazily when first needed (transition to Kerker phase).
pub(crate) struct MixingCudaKernels {
    pub stream: Arc<CudaStream>,
    pub blas: BlasHandle,
    pub cpx_sub: CudaFunction,
    pub cpx_full_update: CudaFunction,
    pub cpx_mask: CudaFunction,
    #[allow(dead_code)]
    pub cpx_zero: CudaFunction,
}

impl MixingCudaKernels {
    /// Compile all mixing CUDA kernels and create a cuBLAS handle.
    #[allow(dead_code)]
    pub fn new() -> Result<Self, Error> {
        let ctx = Arc::new(CudaContext::new(0).map_err(Error::Cuda)?);
        let stream = ctx.default_stream();
        Self::new_from_stream(&stream)
    }

    /// Compile mixing CUDA kernels on an existing stream (sharing its context).
    ///
    /// Use this when a CUDA context already exists (e.g., alongside a
    /// `KerkerPreconditioner`) to avoid context isolation (device pointers
    /// are not valid across different CUDA contexts).
    pub fn new_from_stream(stream: &Arc<CudaStream>) -> Result<Self, Error> {
        let ctx = stream.context();
        let blas = BlasHandle::new(stream.clone())?;

        let opts = CompileOptions { arch: Some("sm_120"), ..Default::default() };
        let ptx = compile_ptx_with_opts(CUDA_KERNEL_SRC, opts).map_err(|e| Error::Nvrtc(e.to_string()))?;
        let module: Arc<CudaModule> = ctx.load_module(ptx).map_err(Error::Cuda)?;

        let load = |name: &str| -> Result<CudaFunction, Error> {
            module.load_function(name).map_err(Error::Cuda)
        };

        Ok(Self {
            stream: stream.clone(),
            blas,
            cpx_sub: load("cpx_sub")?,
            cpx_full_update: load("cpx_full_update")?,
            cpx_mask: load("cpx_mask")?,
            cpx_zero: load("cpx_zero")?,
        })
    }
}
