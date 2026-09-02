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
    double2* dst_c,
    double2* dst_s,
    const double2* n_in_c,
    const double2* n_in_s,
    const double* kerker_c,
    const double* kerker_s,
    const double2* r_c,
    const double2* r_s,
    const double2* sum_dr_c,
    const double2* sum_dr_s,
    const double2* sum_dn_c,
    const double2* sum_dn_s,
    const double2* n_out_c,
    const double2* n_out_s,
    const double* mask,
    int n,
    double amp_c,
    double amp_s,
    double amp_n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        if (mask[i] > 0.5) {
            // Mixing-basis component (G within the CASTEP mix cutoff, incl. G=0).
            // CASTEP dm_sub_mix.f90 dm_mix_density_pulay exact form, applied
            // per part (charge c, spin s); dm_apply_kerker scales each part
            // with its own kerker_matrix column (dm_sub_kerker.f90:51-64):
            //   new_c = n_in_c + sum_dn_c + Kc*(r_c + sum_dr_c)
            //   new_s = n_in_s + sum_dn_s + Ks*(r_s + sum_dr_s)
            // The DIIS density part (sum_dn) is unscaled (amp_n = 1.0, charge
            // and spin parts alike).  The Kerker part carries amp_c / amp_s;
            // the kerker slices hold the PURE G2/(G2+q2) kernels, and the
            // spin slice holds 1.0 at G=0 so amp_s * 1.0 = CASTEP's
            // Ks(0) = mix_spin_amp (dm_sub_base.f90:681-682).
            double rcx = r_c[i].x + sum_dr_c[i].x;
            double rcy = r_c[i].y + sum_dr_c[i].y;
            double rsx = r_s[i].x + sum_dr_s[i].x;
            double rsy = r_s[i].y + sum_dr_s[i].y;
            double kc = kerker_c[i];
            double ks = kerker_s[i];
            dst_c[i].x = n_in_c[i].x + amp_n * sum_dn_c[i].x + amp_c * kc * rcx;
            dst_c[i].y = n_in_c[i].y + amp_n * sum_dn_c[i].y + amp_c * kc * rcy;
            dst_s[i].x = n_in_s[i].x + amp_n * sum_dn_s[i].x + amp_s * ks * rsx;
            dst_s[i].y = n_in_s[i].y + amp_n * sum_dn_s[i].y + amp_s * ks * rsy;
        } else {
            // High-G content: CASTEP dm_mix_density_to_density carries the
            // above-cutoff components from the FRESH output density of this
            // cycle (dencut keeps the high-frequency FFT components of the
            // current wavefunction density). Use n_out, not n_in.
            dst_c[i].x = n_out_c[i].x;
            dst_c[i].y = n_out_c[i].y;
            dst_s[i].x = n_out_s[i].x;
            dst_s[i].y = n_out_s[i].y;
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
