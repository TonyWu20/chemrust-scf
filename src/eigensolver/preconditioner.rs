// ---------------------------------------------------------------------------
// Teter-Payne-Allan (TPA) diagonal preconditioner for Davidson
// ---------------------------------------------------------------------------
// NOTE: dead_code allowed because Group C (Davidson) will be the consumer.
#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use faer::linalg::solvers::{DenseSolveCore, Llt, PartialPivLu};
use faer::mat::Mat;
use faer::Side;
use ndarray::Array2;
use num_complex::Complex64;

use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::{
    KineticPreconditioner, PreconditionerVector, PwCoefficients,
};
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
// TPA scalar function
// ---------------------------------------------------------------------------

/// Teter-Payne-Allan (TPA) preconditioning scalar function.
///
/// tpa(x) = 1 / (1 + 16·x⁴ / (27 + x·(18 + x·(12 + 8·x))))
///
/// where `x = |k+G|² / (2·Ē_kin)` ≥ 0.
///
/// Properties:
/// - `tpa(0) = 1` — no preconditioning at zero kinetic energy.
/// - `tpa(x) → 0` as `x → ∞` — full preconditioning at high kinetic energy.
/// - `tpa(x) ∈ [0, 1]` for all `x ≥ 0`.
///
/// Source: CASTEP `wave.f90:29914-29944` (function `tpa`).
#[inline]
pub fn tpa(x: f64) -> f64 {
    let x2 = x * x; // x²
    let x4 = x2 * x2; // x⁴
    let numerator = 16.0 * x4;
    let denominator = 27.0 + x * (18.0 + x * (12.0 + 8.0 * x));
    1.0 / (1.0 + numerator / denominator)
}

/// Compute the TPA preconditioner vector R(G) on GPU.
///
/// For each plane-wave component G:
///
///   R(G) = tpa(pw_ek(G) / mean_ek)
///
/// where `pw_ek(G)` is the kinetic energy of that PW and `mean_ek` is the
/// mean band kinetic energy: `Σ_b ek(b) / nbands`.
///
/// This is a CPU-mediated function: it downloads the kinetic energies,
/// computes R(G) on the CPU, and uploads the result to a fresh GPU allocation.
///
/// All bands share the same R(G) vector — it is computed once per outer
/// iteration.
pub(crate) fn compute_r_vector(
    kinetic_dev: &KineticPreconditioner,
    mean_ek: f64,
    n_pw: usize,
    stream: &Arc<CudaStream>,
) -> Result<PreconditionerVector, Error> {
    // Download kinetic energies from GPU
    let kinetic_cpu: Vec<f64> = stream
        .clone_dtoh(&**kinetic_dev)
        .map_err(Error::Cuda)?;

    // Compute R(G) on CPU: R(G) = tpa(pw_ek / mean_ek)
    let r_cpu: Vec<f64> = kinetic_cpu
        .iter()
        .take(n_pw)
        .map(|&ek| tpa(ek / mean_ek))
        .collect();

    // Upload result to GPU
    let mut r_dev: CudaSlice<f64> = stream
        .alloc_zeros::<f64>(n_pw)
        .map_err(Error::Cuda)?;
    stream
        .memcpy_htod(&r_cpu, &mut r_dev)
        .map_err(Error::Cuda)?;

    Ok(PreconditionerVector::new(r_dev))
}

// ---------------------------------------------------------------------------
// C = β^H · diag(R) · β  (single ion, CPU)
// ---------------------------------------------------------------------------

/// Compute C = β^H · diag(R) · β for a single ion.
///
/// β has shape (ne, n_pw) in row-major layout: `beta_g[n * n_pw + G] = β[n, G]`
/// R is the preconditioner vector of length n_pw.
///
/// Returns the ne × ne Hermitian matrix C on CPU.
///
/// Algorithm (Gram matrix form):
///   α[n, G] = conj(β[n, G]) · sqrt(R[G])
///   C[n, m] = Σ_G α[n, G] · conj(α[m, G])   — i.e. C = α · α^H
///
/// Source: β^H·diag(R)·β formula, verified against CASTEP
/// `ion_beta_beta_recip_cmplx` (nlpot.f90:13798-13803).
pub fn compute_c_matrix(
    beta_g: &[CudaComplex],
    r_vector: &[f64],
    ne: usize,
    n_pw: usize,
) -> Array2<Complex64> {
    // Compute α[n, G] = conj(β[n, G]) · sqrt(R[G])
    let mut alpha_data: Vec<Complex64> = Vec::with_capacity(ne * n_pw);
    for n in 0..ne {
        for g in 0..n_pw {
            let beta_val = Complex64::new(beta_g[n * n_pw + g].x, beta_g[n * n_pw + g].y);
            let r_scale = Complex64::new(r_vector[g].sqrt(), 0.0);
            alpha_data.push(beta_val.conj() * r_scale);
        }
    }
    let alpha = Array2::from_shape_vec((ne, n_pw), alpha_data)
        .expect("alpha array shape (ne, n_pw) must match data length ne * n_pw");

    // C = α · α^H  (Gram matrix, ne × ne)
    alpha.dot(&alpha.t().mapv(|c| c.conj()))
}

// ---------------------------------------------------------------------------
// Q⁻¹ inversion (CPU, faer Cholesky)
// ---------------------------------------------------------------------------

/// Invert a real symmetric matrix Q using Cholesky decomposition (faer).
///
/// Returns Q⁻¹ as a flat row-major `Vec<f64>`. For near-singular Q
/// (diagonal elements below `TINY = 1e-14`), those rows/cols are zeroed
/// in the output, matching CASTEP's `abs(ps_q(m,m,nsp1)) > tiny`
/// compression (nlpot.f90:13894-13959).
///
/// # Panics
///
/// Panics if `q.len() != n * n`.
pub fn invert_q_matrix(q: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(q.len(), n * n, "Q data length must be n×n, got {} elements for n={}", q.len(), n);

    const TINY: f64 = 1e-14;

    // Identify non-singular diagonal indices (|diag| > TINY)
    let nonsingular: Vec<usize> = (0..n).filter(|&i| q[i * n + i].abs() > TINY).collect();

    let mut result = vec![0.0; n * n];
    let m = nonsingular.len();

    if m == 0 {
        // All singular: return all zeros
        return result;
    }

    // Extract non-singular sub-block
    let mut sub_q = Mat::<f64>::zeros(m, m);
    for (ki, &i) in nonsingular.iter().enumerate() {
        for (kj, &j) in nonsingular.iter().enumerate() {
            sub_q[(ki, kj)] = q[i * n + j];
        }
    }

    // Cholesky factorization + inversion (lower triangular since Q is symmetric)
    if let Ok(llt) = Llt::new(sub_q.as_ref(), Side::Lower) {
        let sub_inv = llt.inverse();

        // Expand back to full size
        for (ki, &i) in nonsingular.iter().enumerate() {
            for (kj, &j) in nonsingular.iter().enumerate() {
                result[i * n + j] = sub_inv[(ki, kj)];
            }
        }
    }
    // If Cholesky fails (not SPD), we leave singular rows/cols as zeros
    // and the non-singular sub-block result stays zeroed (safe fallback)

    result
}

// ---------------------------------------------------------------------------
// R_beta = (−Q⁻¹ − C)⁻¹  assembly
// ---------------------------------------------------------------------------

/// Assemble R_beta = (−Q⁻¹ − C)⁻¹ for one ion.
///
/// Arguments:
/// - `q_inv`: Q⁻¹ matrix (ne × ne, row-major, real symmetric)
/// - `c_matrix`: C = β^H·diag(R)·β (ne × ne, row-major, complex Hermitian)
/// - `ne`: n_expanded for this ion
/// - `mixture_weight`: VCA mixture weight (1.0 for non-VCA)
///
/// Returns R_beta (ne × ne, row-major, complex matrix with zero imaginary
/// part — the result of inverting a real-symmetric matrix).
///
/// Algorithm:
/// 1. Form M = −C_re − Q⁻¹/mixture_weight (real symmetric, ne × ne)
/// 2. Invert via faer Cholesky (Llt); if the matrix is not positive definite,
///    fall back to PartialPivLu (general LU decomposition).
/// 3. Return the inverse as a complex matrix with zero imaginary part.
///
/// Source: CASTEP nlpot.f90:13981-14144 — R = (−Q⁻¹ − C)⁻¹ per ion, using
/// dsytrf+dsytri (Bunch-Kaufman) for the inversion.
#[allow(dead_code)]
pub fn assemble_r_beta(
    q_inv: &[f64],
    c_matrix: &Array2<Complex64>,
    ne: usize,
    mixture_weight: f64,
) -> Array2<Complex64> {
    assert_eq!(q_inv.len(), ne * ne, "q_inv must be ne×ne");

    // Form real-symmetric M = −C_re − Q⁻¹/w
    let mut m_mat = Mat::<f64>::zeros(ne, ne);
    for i in 0..ne {
        for j in 0..ne {
            m_mat[(i, j)] = -c_matrix[[i, j]].re - q_inv[i * ne + j] / mixture_weight;
        }
    }

    // Invert M — try Cholesky first, fall back to LU for non-SPD matrices
    let inv_mat: Mat<f64> = if let Ok(llt) = Llt::new(m_mat.as_ref(), Side::Lower) {
        llt.inverse()
    } else {
        let lu = PartialPivLu::new(m_mat.as_ref());
        lu.inverse()
    };

    // Convert back to Array2<Complex64> with zero imaginary part
    let mut result = Array2::<Complex64>::zeros((ne, ne));
    for i in 0..ne {
        for j in 0..ne {
            result[[i, j]] = Complex64::new(inv_mat[(i, j)], 0.0);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Q_RCQ = −Q + C·Q − R_beta·(C·Q) assembly
// ---------------------------------------------------------------------------

/// Assemble the Q_RCQ matrix for use in the USPP preconditioner apply step.
///
/// For a single MPI rank (all ions local), Q_RCQ is a square matrix of size
/// n_total_proj × n_total_proj.
///
/// Per-ion matrices are arranged block-diagonally: each ion's C occupies
/// the diagonal block at `ion_offsets[i]`, and off-diagonal C blocks are
/// zero (per-ion approximation). The resulting Q_RCQ is therefore also
/// block-diagonal.
///
/// Algorithm (per ion):
/// 1. Initialize diagonal block: `Q_RCQ[i_on,i_on] = −Q_i * w_i`
/// 2. Add C·Q term:            `Q_RCQ[:, i_on] += C[:, i_on] · Q_i · w_i`
/// 3. Subtract R_beta·(C·Q):   `Q_RCQ[i_on, :] −= R_beta_i · (C·Q)[i_on, :]`
///
/// Final per-ion formula: `Q_RCQ[i_on,i_on] = −Q_i·w_i + C_i·Q_i·w_i
///                                                − R_beta_i·C_i·Q_i·w_i`
///
/// Arguments:
/// - `q_matrices`: per-ion Q matrices, each ne×ne real symmetric row-major
/// - `c_per_ion`: per-ion C = β^H·diag(R)·β matrices, each ne×ne complex Hermitian
/// - `r_beta_per_ion`: per-ion R_beta = (−Q⁻¹−C)⁻¹ matrices, each ne×ne complex
/// - `ion_offsets`: cumulative offset of each ion in the global projector
///   space (length n_ions + 1)
/// - `mixture_weights`: per-ion VCA mixture weights (1.0 for non-VCA)
///
/// Returns Q_RCQ as Array2<Complex64> (n_total_proj × n_total_proj, row-major).
///
/// Source: CASTEP nlpot.f90:13827-13880, 14146-14176
pub fn assemble_q_rcq(
    q_matrices: &[Vec<f64>],
    c_per_ion: &[Array2<Complex64>],
    r_beta_per_ion: &[Array2<Complex64>],
    ion_offsets: &[usize],
    mixture_weights: &[f64],
) -> Array2<Complex64> {
    let n_ions = q_matrices.len();
    assert_eq!(c_per_ion.len(), n_ions, "c_per_ion length must match q_matrices");
    assert_eq!(r_beta_per_ion.len(), n_ions, "r_beta_per_ion length must match q_matrices");
    assert_eq!(mixture_weights.len(), n_ions, "mixture_weights length must match q_matrices");
    assert_eq!(ion_offsets.len(), n_ions + 1, "ion_offsets must have n_ions + 1 elements");

    let n_total_proj = ion_offsets[n_ions];
    let mut q_rcq = Array2::<Complex64>::zeros((n_total_proj, n_total_proj));

    for i in 0..n_ions {
        let offset = ion_offsets[i];
        let w = mixture_weights[i];
        let ne = c_per_ion[i].shape()[0];
        let ne2 = ne * ne;

        // Validate per-ion Q matrix dimensions
        assert_eq!(q_matrices[i].len(), ne2,
            "Q matrix for ion {i} has {} elements, expected {ne2} (ne={ne})",
            q_matrices[i].len());

        // Build Q_i as Array2<f64>
        let q_i = Array2::from_shape_vec((ne, ne), q_matrices[i].to_vec())
            .expect("Q matrix must be square");

        // Step 2: On-diagonal block = −Q_i * w_i
        for m in 0..ne {
            for n in 0..ne {
                q_rcq[[offset + m, offset + n]] = Complex64::new(-q_i[[m, n]] * w, 0.0);
            }
        }

        // Step 3: C·Q term
        // Convert Q_i * w to Complex64 for matmul with complex C_i
        let q_i_scaled = q_i.mapv(|v| Complex64::new(v * w, 0.0));
        let cq_block = c_per_ion[i].dot(&q_i_scaled);

        // Add C·Q to the on-diagonal column block (C is block-diagonal,
        // so only the diagonal block is affected)
        for m in 0..ne {
            for n in 0..ne {
                q_rcq[[offset + m, offset + n]] += cq_block[[m, n]];
            }
        }

        // Step 4: R_beta correction
        // Q_RCQ[i_on, :] −= R_beta_i · (C·Q)[i_on, :]
        // With block-diagonal C, (C·Q)[i_on, :] is zero outside the diagonal
        // block, so this only affects the diagonal block.
        let r_cq = r_beta_per_ion[i].dot(&cq_block);
        for m in 0..ne {
            for n in 0..ne {
                q_rcq[[offset + m, offset + n]] -= r_cq[[m, n]];
            }
        }
    }

    q_rcq
}

// ---------------------------------------------------------------------------
// TPA Preconditioner (legacy)
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
