// ---------------------------------------------------------------------------
// Teter-Payne-Allan (TPA) diagonal preconditioner for Davidson
// ---------------------------------------------------------------------------
// NOTE: dead_code allowed because Group C (Davidson) will be the consumer.
#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::eigensolver::davidson_types::{KineticPreconditioner, PwCoefficients};
use crate::types::Error;

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled via NVRTC at startup)
// ---------------------------------------------------------------------------

const TPA_PRECOND_KERNEL: &str = r#"
extern "C" __global__ void tpa_precondition(
    double2* precond,
    const double2* __restrict__ residual,
    const double* __restrict__ kinetic,
    double lambda,
    double clamp_eps,
    int n_pw
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = g; i < n_pw; i += stride) {
        double denom = kinetic[i] - lambda;
        if (fabs(denom) < clamp_eps) {
            denom = copysign(clamp_eps, denom);
        }
        double inv = 1.0 / denom;
        precond[i].x = residual[i].x * inv;
        precond[i].y = residual[i].y * inv;
    }
}
"#;

// ---------------------------------------------------------------------------
// TPA Preconditioner
// ---------------------------------------------------------------------------

/// Teter-Payne-Allan diagonal preconditioner for Davidson block eigensolver.
///
/// Applies P⁻¹ · residual where the diagonal of P is `T(g) − λ`, with `T(g)`
/// the kinetic energy of plane-wave component `g` and `λ` the eigenvalue shift.
/// Division-by-zero is prevented by clamping `|T(g) − λ| ≥ `clamp_eps`.
pub(crate) struct TpaPreconditioner {
    kernel: CudaFunction,
    clamp_eps: f64,
}

#[bon::bon]
impl TpaPreconditioner {
    /// Compile the TPA preconditioner CUDA kernel via NVRTC.
    ///
    /// `clamp_eps` is the minimum value of `|T(g) − λ|` (default `1e-12`).
    pub fn new(ctx: &Arc<CudaContext>, clamp_eps: f64) -> Result<Self, Error> {
        let ptx = compile_ptx(TPA_PRECOND_KERNEL).map_err(|e| Error::Nvrtc(e.to_string()))?;
        let module: Arc<CudaModule> = ctx.load_module(ptx).map_err(Error::Cuda)?;
        let kernel = module
            .load_function("tpa_precondition")
            .map_err(Error::Cuda)?;
        Ok(Self { kernel, clamp_eps })
    }

    /// Apply P⁻¹ · residual → precond (may alias residual for in-place).
    ///
    /// `precond` and `residual` may point to the same `CudaSlice` — the kernel
    /// reads each residual element before writing the corresponding precond
    /// element, so in-place operation is safe.
    ///
    /// # Safety
    ///
    /// - `precond` and `residual` must have length ≥ `n_pw`.
    /// - `kinetic_dev` must have length ≥ `n_pw`.
    /// - No other kernel on the same stream may read/write `precond` or
    ///   `residual` concurrently.
    #[builder]
    pub(crate) unsafe fn apply(
        &self,
        precond: &mut PwCoefficients,
        residual: &PwCoefficients,
        kinetic_dev: &KineticPreconditioner,
        lambda: f64,
        n_pw: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Error> {
        let n_pw_i32 = n_pw as i32;
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&mut **precond)
                .arg(&**residual)
                .arg(&**kinetic_dev)
                .arg(&lambda)
                .arg(&self.clamp_eps)
                .arg(&n_pw_i32)
                .launch(LaunchConfig::for_num_elems(n_pw as u32))
                .map(|_| ())
        }
        .map_err(Error::Cuda)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the TPA clamping prevents infinities when `T(g) ≈ λ`.
    ///
    /// Creates a synthetic kinetic array where one element equals `λ`, applies
    /// the preconditioner formula on the CPU, and checks the result is finite
    /// (not Inf/NaN).
    #[test]
    fn test_tpa_clamps_near_zero() {
        let kinetic: Vec<f64> = vec![10.0, 5.0, 3.0, 2.0, 0.5];
        let lambda: f64 = 0.5;
        let clamp_eps: f64 = 1e-12;
        let residual = vec![
            CudaComplex { x: 1.0, y: 2.0 },
            CudaComplex { x: 3.0, y: 4.0 },
            CudaComplex { x: 0.0, y: 1.0 },
            CudaComplex { x: -1.0, y: 0.0 },
            CudaComplex { x: 5.0, y: -3.0 },
        ];
        let n_pw = kinetic.len();

        let mut precond = vec![CudaComplex { x: 0.0, y: 0.0 }; n_pw];

        for i in 0..n_pw {
            let mut denom = kinetic[i] - lambda;
            if denom.abs() < clamp_eps {
                denom = clamp_eps.copysign(denom);
            }
            let inv = 1.0 / denom;
            precond[i].x = residual[i].x * inv;
            precond[i].y = residual[i].y * inv;
        }

        // The element with kinetic[4] == lambda should be clamped
        let clamped = &precond[4];
        assert!(
            clamped.x.is_finite() && clamped.y.is_finite(),
            "Expected finite result for clamped element, got ({}, {})",
            clamped.x,
            clamped.y,
        );
        // Verify identity: denom = clamp_eps (positive since denom == 0)
        let expected_x = residual[4].x / clamp_eps;
        let expected_y = residual[4].y / clamp_eps;
        assert!(
            (clamped.x - expected_x).abs() < 1e-20,
            "Clamped x mismatch: expected {:.6e}, got {:.6e}",
            expected_x,
            clamped.x,
        );
        assert!(
            (clamped.y - expected_y).abs() < 1e-20,
            "Clamped y mismatch: expected {:.6e}, got {:.6e}",
            expected_y,
            clamped.y,
        );
    }

    /// Verify the TPA formula `precond = residual / (T(g) − λ)` to high
    /// precision when `|T(g) − λ|` is large (no clamping active).
    #[test]
    fn test_tpa_identity_far_from_lambda() {
        let kinetic: Vec<f64> = vec![100.0, 50.0, 30.0, 20.0, 10.0];
        let lambda: f64 = 0.5;
        let clamp_eps: f64 = 1e-12;
        let residual = vec![
            CudaComplex { x: 0.5, y: 1.0 },
            CudaComplex { x: -2.0, y: 3.0 },
            CudaComplex { x: 1.5, y: -0.5 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 7.0, y: -4.0 },
        ];
        let n_pw = kinetic.len();

        let mut precond = vec![CudaComplex { x: 0.0, y: 0.0 }; n_pw];

        for i in 0..n_pw {
            let mut denom = kinetic[i] - lambda;
            if denom.abs() < clamp_eps {
                denom = clamp_eps.copysign(denom);
            }
            let inv = 1.0 / denom;
            precond[i].x = residual[i].x * inv;
            precond[i].y = residual[i].y * inv;
        }

        for i in 0..n_pw {
            let expected_x = residual[i].x / (kinetic[i] - lambda);
            let expected_y = residual[i].y / (kinetic[i] - lambda);
            assert!(
                (precond[i].x - expected_x).abs() < 1e-12,
                "Element {i} x mismatch: expected {expected_x:.6e}, got {precond:.6e}",
                precond = precond[i].x,
            );
            assert!(
                (precond[i].y - expected_y).abs() < 1e-12,
                "Element {i} y mismatch: expected {expected_y:.6e}, got {precond:.6e}",
                precond = precond[i].y,
            );
        }
    }
}
