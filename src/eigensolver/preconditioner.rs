// ---------------------------------------------------------------------------
// Teter-Payne-Allan (TPA) diagonal preconditioner for Davidson
// ---------------------------------------------------------------------------
// NOTE: dead_code allowed because Group C (Davidson) will be the consumer.
#![allow(dead_code)]

#[cfg(feature = "scf_diag")]
macro_rules! precon_diag {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}
#[cfg(not(feature = "scf_diag"))]
macro_rules! precon_diag {
    ($($arg:tt)*) => {};
}

use std::sync::Arc;

use cudarc::cublas::sys::{cublasHandle_t, cublasZcopy_v2, cublasZdotc_v2};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use faer::linalg::solvers::{DenseSolveCore, Llt, PartialPivLu};
use faer::mat::Mat;
use faer::Side;
use ndarray::{Array2, ShapeBuilder};
use num_complex::Complex64;

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::{
    KineticPreconditioner, PreconditionerVector, PwCoefficients,
};
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled via NVRTC at startup)
// ---------------------------------------------------------------------------

const TPA_APPLY_KERNEL: &str = r#"
// Kernel 1: Fused residual + TPA scaling for each band
// out[G,b] = (hpsi[G,b] - e[b]*psi[G,b]) * r[G]
extern "C" __global__ void tpa_apply_residual(
    double2* out,
    const double2* psi,
    const double2* hpsi,
    const double* eigenvalues,
    const double* r_vector,
    int n_pw,
    int n_bands
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = n_pw * n_bands;
    for (int i = tid; i < total; i += stride) {
        int g = i % n_pw;
        int b = i / n_pw;
        double e = eigenvalues[b];
        double r = r_vector[g];
        double hx = hpsi[i].x;
        double hy = hpsi[i].y;
        double px = psi[i].x;
        double py = psi[i].y;
        out[i].x = (hx - e * px) * r;
        out[i].y = (hy - e * py) * r;
    }
}

// Kernel 2: Add NL correction with TPA factor
// out[G,b] += correction[G,b] * r[G]
extern "C" __global__ void tpa_apply_add(
    double2* out,
    const double2* correction,
    const double* r_vector,
    int n_pw,
    int n_bands
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = n_pw * n_bands;
    for (int i = tid; i < total; i += stride) {
        int g = i % n_pw;
        double r = r_vector[g];
        out[i].x += correction[i].x * r;
        out[i].y += correction[i].y * r;
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
///
/// Reference: CASTEP `nlpot.f90:13525-13593` (nlpot_prepare_precon R(G) computation).
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
/// Reference: CASTEP `nlpot.f90:13600-13670` (C = β^H·R·β assembly).
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
// C_global = β^H · diag(R) · β  (cross-ion blocks, CPU)
// ---------------------------------------------------------------------------

/// Compute the global C = β^H · diag(R) · β matrix spanning all ions.
///
/// Unlike compute_c_matrix which computes per-ion C_i, this function computes
/// the full global matrix with cross-ion blocks:
///
///   C[off_i+m, off_j+n] = Σ_G conj(β_i[G,m]) · R(G) · β_j[G,n]
///
/// # Shape
///
/// Returns (total_ne, total_ne) where total_ne = ion_offsets[n_ions].
/// Diagonal blocks at [off_i..off_i+ne_i, off_j..off_j+ne_j] are the per-ion
/// C_i matrices. Off-diagonal blocks capture cross-ion projector overlap
/// weighted by the TPA preconditioner vector.
///
/// Reference: CASTEP nlpot.f90:13805-13818 (ion_beta_beta_recip).
pub fn compute_c_global(
    beta_g_per_ion: &[Array2<Complex64>],
    r_vector: &[f64],
    ion_offsets: &[usize],
) -> Array2<Complex64> {
    let n_ions = beta_g_per_ion.len();
    let total_ne = ion_offsets[n_ions];
    let n_pw = r_vector.len();

    // Build alpha_global: alpha[n, G] = conj(beta[G, n]) * sqrt(R[G])
    // alpha_global has shape (total_ne, n_pw), row-major.
    let mut alpha_data: Vec<Complex64> = Vec::with_capacity(total_ne * n_pw);
    for i in 0..n_ions {
        let ne_i = beta_g_per_ion[i].shape()[1];
        for n in 0..ne_i {
            for g in 0..n_pw {
                let beta_val = beta_g_per_ion[i][[g, n]];
                let r_scale = Complex64::new(r_vector[g].sqrt(), 0.0);
                alpha_data.push(beta_val.conj() * r_scale);
            }
        }
    }
    let alpha = Array2::from_shape_vec((total_ne, n_pw), alpha_data)
        .expect("alpha shape (total_ne, n_pw) must match");

    // C = alpha * alpha^H  — single matrix multiply, matches CASTEP's
    // ion_beta_beta_recip which computes C = beta^H * diag(R) * beta globally.
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
/// Reference: CASTEP `nlpot.f90:13880-13980` (Q⁻¹ via Decomposition).
///
/// # Panics
///
/// Panics if `q.len() != n * n`.
pub fn invert_q_matrix(q: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(q.len(), n * n, "Q data length must be n×n, got {} elements for n={}", q.len(), n);

    const TINY: f64 = f64::MIN_POSITIVE; // ≈2.22e-308, matches CASTEP's tiny(1.0_dp)

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
    } else {
        // CASTEP nlpot.f90:13948-13968 uses dsytrf+dsytri (Bunch-Kaufman).
        // Cholesky fails for non-SPD matrices.  Fall back to LU decomposition
        // (matches assemble_r_beta's fallback pattern, line 338-343).
        let lu = PartialPivLu::new(sub_q.as_ref());
        let sub_inv = lu.inverse();

        for (ki, &i) in nonsingular.iter().enumerate() {
            for (kj, &j) in nonsingular.iter().enumerate() {
                result[i * n + j] = sub_inv[(ki, kj)];
            }
        }
    }

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
// Q_RCQ = −Q − R_beta·C·Q assembly (global C, full off-diagonal blocks)
// ---------------------------------------------------------------------------

/// Assemble the Q_RCQ matrix for use in the USPP preconditioner apply step.
///
/// For a single MPI rank (all ions local), Q_RCQ is a square matrix of size
/// total_ne × total_ne.  Unlike the per-ion approximation, this function uses
/// the FULL global C matrix (including cross-ion blocks), so the resulting
/// Q_RCQ has non-zero off-diagonal blocks matching CASTEP.
///
/// Algorithm:
/// 1. Set diagonal blocks: `Q_RCQ[off_i..off_i+ne_i, off_i..off_i+ne_i] = −Q_i * w_i`
/// 2. Compute CQ = C_global * Q_global (Q_global is block-diagonal Q_i * w_i)
/// 3. For each ion i: `Q_RCQ[off_i..off_i+ne_i, :] −= R_beta_i * CQ[off_i..off_i+ne_i, :]`
///
/// Step 3 populates BOTH diagonal and off-diagonal blocks because CQ is a
/// full matrix (cross-ion C × block-diagonal Q).
///
/// Final formula: `Q_RCQ = −Q − R_beta · C · Q`  (CASTEP nlpot.f90:14162-14180)
///
/// Arguments:
/// - `q_matrices`: per-ion Q matrices, each ne×ne real symmetric row-major
/// - `c_global`: global C = β^H·diag(R)·β, shape (total_ne, total_ne)
/// - `r_beta_per_ion`: per-ion R_beta = (−Q⁻¹−C)⁻¹ matrices, each ne×ne complex
/// - `ion_offsets`: cumulative offset of each ion in the global projector
///   space (length n_ions + 1)
/// - `mixture_weights`: per-ion VCA mixture weights (1.0 for non-VCA)
///
/// Returns Q_RCQ as Array2<Complex64> (total_ne × total_ne, row-major).
///
/// Source: CASTEP nlpot.f90:13841-13893, 14162-14180
pub fn assemble_q_rcq(
    q_matrices: &[Vec<f64>],
    c_global: &Array2<Complex64>,
    r_beta_per_ion: &[Array2<Complex64>],
    ion_offsets: &[usize],
    mixture_weights: &[f64],
) -> Array2<Complex64> {
    let n_ions = q_matrices.len();
    assert_eq!(r_beta_per_ion.len(), n_ions, "r_beta_per_ion length must match q_matrices");
    assert_eq!(mixture_weights.len(), n_ions, "mixture_weights length must match q_matrices");
    assert_eq!(ion_offsets.len(), n_ions + 1, "ion_offsets must have n_ions + 1 elements");

    let total_ne = ion_offsets[n_ions];
    let mut q_rcq = Array2::<Complex64>::zeros((total_ne, total_ne));

    // Step 1: Build Q_global (block-diagonal Q_i * w_i) and set Q_RCQ diagonal = −Q_i * w_i
    let mut q_global = Array2::<Complex64>::zeros((total_ne, total_ne));
    for i in 0..n_ions {
        let offset = ion_offsets[i];
        let w = mixture_weights[i];
        let ne = r_beta_per_ion[i].shape()[0];
        let ne2 = ne * ne;

        assert_eq!(q_matrices[i].len(), ne2,
            "Q matrix for ion {i} has {} elements, expected {ne2} (ne={ne})",
            q_matrices[i].len());

        let q_i = Array2::from_shape_vec((ne, ne), q_matrices[i].to_vec())
            .expect("Q matrix must be square");

        for m in 0..ne {
            for n in 0..ne {
                q_rcq[[offset + m, offset + n]] = Complex64::new(-q_i[[m, n]] * w, 0.0);
                q_global[[offset + m, offset + n]] = Complex64::new(q_i[[m, n]] * w, 0.0);
            }
        }
    }

    // Step 2: CQ = C_global * Q_global  (full matrix product)
    let cq = c_global.dot(&q_global);

    // Step 3: Q_RCQ[ion_block, :] −= R_beta * CQ[ion_block, :]
    // CASTEP nlpot.f90:14166-14180
    for i in 0..n_ions {
        let offset = ion_offsets[i];
        let ne = r_beta_per_ion[i].shape()[0];
        let cq_block = cq.slice(ndarray::s![offset..offset + ne, ..]);
        let contrib = r_beta_per_ion[i].dot(&cq_block);
        for m in 0..ne {
            for k in 0..total_ne {
                q_rcq[[offset + m, k]] -= contrib[[m, k]];
            }
        }
    }

    q_rcq
}

// ---------------------------------------------------------------------------
// TPA Preconditioner (correct TPA-apply kernels)
// ---------------------------------------------------------------------------

/// Teter-Payne-Allan preconditioner for Davidson block eigensolver.
///
/// Applies the TPA preconditioner to residual vectors. This replaces the old
/// `1/(T−λ)` formula with the correct CASTEP formula:
///
///   preconditioned(b,G) = (Hψ(b,G) − e(b)·ψ(b,G)) · R(G)
///
/// where `R(G) = tpa(pw_ek(G) / mean_ek)` is the TPA preconditioner vector
/// (the same for all bands).
///
/// Two fused kernels are provided:
/// - `apply_residual`: forms the preconditioned residual directly from ψ, Hψ,
///   eigenvalues, and the R vector.
/// - `apply_add`: adds a correction term (e.g. NL contribution) with the TPA
///   R-vector scaling.
///
/// Reference: CASTEP `nlpot.f90:15879-16274` (nlpot_apply_precon_ES_slice).
pub struct TpaPreconditioner {
    kernel_residual: CudaFunction,
    kernel_add: CudaFunction,
}

impl TpaPreconditioner {
    /// Compile the TPA-apply CUDA kernels via NVRTC.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Error> {
        let ptx = compile_ptx(TPA_APPLY_KERNEL).map_err(|e| Error::Nvrtc(e.to_string()))?;
        let module: Arc<CudaModule> = ctx.load_module(ptx).map_err(Error::Cuda)?;
        let kernel_residual = module
            .load_function("tpa_apply_residual")
            .map_err(Error::Cuda)?;
        let kernel_add = module
            .load_function("tpa_apply_add")
            .map_err(Error::Cuda)?;
        Ok(Self { kernel_residual, kernel_add })
    }

    /// Apply the TPA-preconditioned residual.
    ///
    /// Computes: `out[G,b] = (hpsi[G,b] − e[b]·psi[G,b]) · R(G)`
    ///
    /// CASTEP nlpot.f90:15970 — USPP correction via NL weights,
    /// not by replacing ψ with Sψ in the kernel.
    /// by replacing ψ with Sψ in the kernel.
    ///
    /// # Safety
    ///
    /// - All slices must have sufficient length:
    ///   `out`, `psi`, `hpsi` ≥ `n_pw * n_bands`
    ///   `eigenvalues_dev` ≥ `n_bands`
    ///   `r_vector` ≥ `n_pw`
    /// - No other kernel on the same stream may read/write these buffers
    ///   concurrently.
    pub(crate) unsafe fn apply_residual(
        &self,
        out: &mut PwCoefficients,
        psi: &PwCoefficients,
        hpsi: &PwCoefficients,
        eigenvalues_dev: &CudaSlice<f64>,
        r_vector: &PreconditionerVector,
        n_pw: usize,
        n_bands: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Error> {
        let total = n_pw * n_bands;
        let n_pw_i32 = n_pw as i32;
        let n_bands_i32 = n_bands as i32;
        unsafe {
            stream
                .launch_builder(&self.kernel_residual)
                .arg(&mut **out)
                .arg(&**psi)
                .arg(&**hpsi)
                .arg(eigenvalues_dev)
                .arg(&**r_vector)
                .arg(&n_pw_i32)
                .arg(&n_bands_i32)
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map(|_| ())
        }
        .map_err(Error::Cuda)
    }

    /// Add the NL correction term with TPA factor.
    ///
    /// Computes: `out[G,b] += correction[G,b] · R(G)`
    ///
    /// # Safety
    ///
    /// - All slices must have sufficient length:
    ///   `out`, `correction` ≥ `n_pw * n_bands`
    ///   `r_vector` ≥ `n_pw`
    /// - No other kernel on the same stream may read/write these buffers
    ///   concurrently.
    pub(crate) unsafe fn apply_add(
        &self,
        out: &mut PwCoefficients,
        correction: &PwCoefficients,
        r_vector: &PreconditionerVector,
        n_pw: usize,
        n_bands: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Error> {
        let total = n_pw * n_bands;
        let n_pw_i32 = n_pw as i32;
        let n_bands_i32 = n_bands as i32;
        unsafe {
            stream
                .launch_builder(&self.kernel_add)
                .arg(&mut **out)
                .arg(&**correction)
                .arg(&**r_vector)
                .arg(&n_pw_i32)
                .arg(&n_bands_i32)
                .launch(LaunchConfig::for_num_elems(total as u32))
                .map(|_| ())
        }
        .map_err(Error::Cuda)
    }
}

// ---------------------------------------------------------------------------
// apply_preconditioner — TPA preconditioner entry point
// ---------------------------------------------------------------------------

/// Apply the TPA preconditioner to the residual vector.
///
/// NCPP path (always applied):
///
///   `precon[G,b] = (hpsi[G,b] - e[b] * psi[G,b]) * R(G)`
///
/// where `R(G) = tpa(pw_ek(G) / mean_ek)` is the TPA preconditioner vector
/// (computed once per outer iteration by [`compute_r_vector`]).
///
/// USPP NL correction (applied when all optional USPP parameters are provided):
///
///   For each ion i with ne projectors:
///     1. βψ[n,b] = Σ_G conj(β_n(G)) · ψ(G,b)           (beta_phi for psi)
///     2. βψ_precon[n,b] = Σ_G conj(β_n(G)) · precon(G,b) (beta_phi for precon)
///     3. E_beta[b] = eigenvalue[b] · βψ[:,b]           (scale by eigenvalues)
///     4. weight = Q_RCQ_block · E_beta + R_beta_block · βψ_precon
///     5. temp[G,b] = Σ_n β_n(G) · weight[n,b]          (NL correction)
///     6. precon[G,b] += temp[G,b] · R(G)               (TPA-scaled accumulation)
///
/// Reference: CASTEP `nlpot_apply_precon_ES_slice` (nlpot.f90:15879-16231).
///
/// # Safety
///
/// - All input slices must have sufficient length:
///   `psi`, `hpsi` >= `n_pw * n_bands`
///   `eigenvalues` >= `n_bands`
///   `r_vector` >= `n_pw`
/// - No other kernel on the same stream may read/write these buffers
///   concurrently.
/// - When USPP params are provided, `vnl_data` and `blas` must be valid.
#[bon::builder]
/// CASTEP nlpot.f90:15970 — USPP correction via NL weights,
/// The kernel computes `(hpsi - ε·psi) * R(G)`, which is correct
/// for USPP (S≠I ⇒ spsi≠psi).
pub unsafe fn apply_preconditioner(
    psi: &PwCoefficients,
    hpsi: &PwCoefficients,
    eigenvalues: &CudaSlice<f64>,
    r_vector: &PreconditionerVector,
    tpa_preconditioner: &TpaPreconditioner,
    n_bands: usize,
    n_pw: usize,
    stream: &Arc<CudaStream>,
    // Optional USPP NL correction parameters
    vnl_data: Option<&VnlBatchData>,
    r_beta_per_ion: Option<&[Array2<Complex64>]>,
    q_rcq: Option<&Array2<Complex64>>,
    blas: Option<&BlasHandle>,
) -> Result<PwCoefficients, Error> {
    let total = n_pw * n_bands;
    let out_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(total).map_err(Error::Cuda)?;
    let mut precon = PwCoefficients::new(out_dev);

    // Step 1: TPA step — always applied (same for NCPP and USPP)
    unsafe {
        tpa_preconditioner.apply_residual(
            &mut precon,
            psi,
            hpsi,
            eigenvalues,
            r_vector,
            n_pw,
            n_bands,
            stream,
        )
    }?;

    // Diagnostic: check eigenvalues only (lightweight, no per-band L2² download)
    // Full precon download is ~25 MB and impacts test runtime — only done on anomaly.
    if let Ok(eig_cpu) = stream.clone_dtoh(eigenvalues) {
        let has_anomaly = eig_cpu.iter().any(|e| e.abs() > 1e10 || e.is_nan());
        if has_anomaly {
            let precon_cpu = stream.clone_dtoh(&*precon).unwrap_or_default();
            let mut max_l2sq = 0.0f64;
            let mut max_b = 0usize;
            for b in 0..n_bands {
                let mut l2sq = 0.0f64;
                let base = b * n_pw;
                for g in 0..n_pw {
                    let c = precon_cpu[base + g];
                    l2sq += c.x * c.x + c.y * c.y;
                }
                if l2sq > max_l2sq { max_l2sq = l2sq; max_b = b; }
            }
            let _max_band_eig = if max_b < eig_cpu.len() { eig_cpu[max_b] } else { f64::NAN };
            precon_diag!("[Diag-Precon-TPA] ANOMALY: L2² min=N/A max={:.4e} (band {}/{}, eig={:.6e}) across {} bands",
                max_l2sq, max_b, n_bands, _max_band_eig, n_bands);
            precon_diag!("[Diag-Precon-TPA] ANOMALY: all band eigenvalues:");
            for _b in 0..n_bands.min(eig_cpu.len()) {
                precon_diag!("  band {}: eig={:.6e}", _b, eig_cpu[_b]);
            }
        }
    }

    // Step 2: USPP NL correction (only if all USPP params are provided)
    if let (Some(vnl_data), Some(r_beta_per_ion), Some(q_rcq), Some(blas)) =
        (vnl_data, r_beta_per_ion, q_rcq, blas)
    {
        // Early return if no species have augmentation (all NCPP)
        if vnl_data.entries.iter().all(|e| e.n_expanded == 0) {
            return Ok(precon);
        }

        // Download eigenvalues to CPU (needed for weight scaling in step 4)
        let eigenvalues_cpu: Vec<f64> = stream
            .clone_dtoh(eigenvalues)
            .map_err(Error::Cuda)?;

        // Build ion_offsets from cumulative n_expanded sums
        let mut ion_offsets = Vec::with_capacity(vnl_data.entries.len() + 1);
        let mut cum = 0usize;
        for entry in &vnl_data.entries {
            ion_offsets.push(cum);
            cum += entry.n_expanded as usize;
        }
        ion_offsets.push(cum);
        let total_ne = cum;

        // Clone the TPA-preconditioned buffer BEFORE any USPP NL modification.
        // CASTEP nlpot.f90:15995 calls wave_beta_phi(precon_slice) on the
        // FRESHLY TPA-preconditioned residual BEFORE any in-place correction.
        // If we used `precon` directly in the per-ion beta^H*precon GEMM below,
        // each ion would read an already-corrected buffer, creating a feedback
        // loop that causes exponential growth of the NL weights.
        let precon_tpa_dev: CudaSlice<CudaComplex> =
            stream.alloc_zeros(n_pw * n_bands).map_err(Error::Cuda)?;
        let handle: cublasHandle_t = blas.raw_handle();
        unsafe {
            let (precon_ptr, _) = precon.device_ptr(stream);
            let (tpa_ptr, _) = precon_tpa_dev.device_ptr(stream);
            cublasZcopy_v2(
                handle,
                (n_pw * n_bands) as i32,
                precon_ptr as *const _,
                1,
                tpa_ptr as *mut _,
                1,
            );
        }
        let precon_tpa = PwCoefficients::new(precon_tpa_dev);

        // -----------------------------------------------------------------------
        // PASS 1: Gather global arrays (per-ion GEMM, accumulate into global)
        // -----------------------------------------------------------------------
        let mut global_beta_phi_psi = Array2::<Complex64>::zeros((total_ne, n_bands));
        let mut global_beta_phi_precon = Array2::<Complex64>::zeros((total_ne, n_bands));
        let mut global_r_beta = Array2::<Complex64>::zeros((total_ne, total_ne));

        for (i, entry) in vnl_data.entries.iter().enumerate() {
            let ne = entry.n_expanded as usize;
            if ne == 0 {
                continue;
            }
            let offset = ion_offsets[i];

            // ---------------------------------------------------------------
            // Step 3a: beta_phi for psi — beta_g^H · psi (ne × n_bands)
            // ---------------------------------------------------------------
            let mut beta_phi_psi_dev: CudaSlice<CudaComplex> = stream
                .alloc_zeros(ne * n_bands)
                .map_err(Error::Cuda)?;
            unsafe {
                blas.gemm_c64(
                    ZgemmConfig {
                        transa: op::C,
                        transb: op::N,
                        m: ne as i32,
                        n: n_bands as i32,
                        k: n_pw as i32,
                        alpha: CudaComplex { x: 1.0, y: 0.0 },
                        lda: n_pw as i32,
                        ldb: n_pw as i32,
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                        ldc: ne as i32,
                    },
                    &entry.beta_g,
                    psi,
                    &mut beta_phi_psi_dev,
                )?;
            }

            // ── GEMM CROSS-CHECK (ion 0, band 0 only) ────────────────────
            // Manually compute ⟨beta_g[0]|psi[0]⟩ via cublasZdotc and
            // compare with GEMM[0,0].  If they disagree the GEMM is reading
            // wrong memory (likely buffer aliasing or stale pointer).
            // Also verify the Cauchy-Schwarz bound:
            //   |⟨beta|psi⟩| ≤ sqrt(L2²(beta) × L2²(psi))
            if i == 0 {
                let (beta_ptr, _) = entry.beta_g.device_ptr(stream);
                let (psi_ptr, _) = psi.device_ptr(stream);
                let n_pw_i32 = n_pw as i32;
                let handle: cublasHandle_t = blas.raw_handle();

                // dot1 = ⟨beta_g[0]|psi[0]⟩  (manual dot product)
                let mut dot1 = CudaComplex { x: 0.0, y: 0.0 };
                unsafe {
                    cublasZdotc_v2(
                        handle,
                        n_pw_i32,
                        beta_ptr as *const _,
                        1,
                        psi_ptr as *const _,
                        1,
                        &mut dot1 as *mut _ as *mut _,
                    );
                }

                // dot2 = ⟨beta_g[0]|beta_g[0]⟩  (L2² of projector)
                let mut dot2 = CudaComplex { x: 0.0, y: 0.0 };
                unsafe {
                    cublasZdotc_v2(
                        handle,
                        n_pw_i32,
                        beta_ptr as *const _,
                        1,
                        beta_ptr as *const _,
                        1,
                        &mut dot2 as *mut _ as *mut _,
                    );
                }

                // dot3 = ⟨psi[0]|psi[0]⟩  (L2² of psi column 0)
                let mut dot3 = CudaComplex { x: 0.0, y: 0.0 };
                unsafe {
                    cublasZdotc_v2(
                        handle,
                        n_pw_i32,
                        psi_ptr as *const _,
                        1,
                        psi_ptr as *const _,
                        1,
                        &mut dot3 as *mut _ as *mut _,
                    );
                }

                // Cauchy-Schwarz bound
                let bound = (dot2.x * dot3.x).sqrt();
                let _exceeded = if dot1.x.abs() > bound * 1.01 {
                    "CS-VIOLATION"
                } else {
                    "OK"
                };

                precon_diag!(
                    "[Diag-GEMM-xcheck] \
                     dotc(beta[0],psi[0])=({:.6e},{:.6e}) \
                     |beta|²={:.6e} |psi|²={:.6e} \
                     CS_bound={:.6e} CS={}",
                    dot1.x, dot1.y,
                    dot2.x, dot3.x,
                    bound, _exceeded,
                );

                // Now download GEMM[0,0] and compare with dot1
                let gemm00 = stream.alloc_zeros::<CudaComplex>(1).map_err(Error::Cuda)?;
                unsafe {
                    // GEMM result is column-major: element (proj=0, band=0) is at offset 0
                    let (gemm_ptr, _) = beta_phi_psi_dev.device_ptr(stream);
                    let (g00_ptr, _) = gemm00.device_ptr(stream);
                    cublasZcopy_v2(handle, 1,
                        gemm_ptr as *const _, 1,
                        g00_ptr as *mut _, 1,
                    );
                }
                let gemm00_cpu: Vec<CudaComplex> = stream.clone_dtoh(&gemm00).map_err(Error::Cuda)?;
                let _reldiff = if dot1.x.abs() > 1e-30 {
                    ((gemm00_cpu[0].x - dot1.x) / dot1.x).abs()
                } else {
                    0.0
                };
                precon_diag!(
                    "[Diag-GEMM-xcheck] GEMM[0,0]=({:.6e},{:.6e}) \
                     dotc[0,0]=({:.6e},{:.6e}) reldiff={:.6e}",
                    gemm00_cpu[0].x, gemm00_cpu[0].y,
                    dot1.x, dot1.y,
                    _reldiff,
                );

                // Dump first 5 elements of psi[0] and beta_g[0] directly
                let psi_5 = stream.alloc_zeros::<CudaComplex>(5).map_err(Error::Cuda)?;
                let beta_5 = stream.alloc_zeros::<CudaComplex>(5).map_err(Error::Cuda)?;
                unsafe {
                    let (p5_ptr, _) = psi_5.device_ptr(stream);
                    cublasZcopy_v2(handle, 5,
                        psi_ptr as *const _, 1,
                        p5_ptr as *mut _, 1,
                    );
                    let (b5_ptr, _) = beta_5.device_ptr(stream);
                    cublasZcopy_v2(handle, 5,
                        beta_ptr as *const _, 1,
                        b5_ptr as *mut _, 1,
                    );
                }
                let _psi_5_cpu: Vec<CudaComplex> = stream.clone_dtoh(&psi_5).map_err(Error::Cuda)?;
                let _beta_5_cpu: Vec<CudaComplex> = stream.clone_dtoh(&beta_5).map_err(Error::Cuda)?;
                precon_diag!(
                    "[Diag-GEMM-xcheck] psi[0..4]: {:?}",
                    _psi_5_cpu.iter().map(|c| (c.x, c.y)).collect::<Vec<_>>()
                );
                precon_diag!(
                    "[Diag-GEMM-xcheck] beta_g[0..4]: {:?}",
                    _beta_5_cpu.iter().map(|c| (c.x, c.y)).collect::<Vec<_>>()
                );
            }

            // D2H: copy beta_phi_psi to CPU for global accumulation
            let beta_phi_psi_cpu: Vec<CudaComplex> = stream
                .clone_dtoh(&beta_phi_psi_dev)
                .map_err(Error::Cuda)?;

            // ---------------------------------------------------------------
            // Step 3b: beta_phi for precon — beta_g^H · precon_tpa (ne × n_bands)
            // CRITICAL: Use precon_tpa (clone of the original TPA-preconditioned
            // residual) rather than `precon` (which gets modified in-place by
            // previous ions' corrections).  This matches CASTEP nlpot.f90:15995
            // where wave_beta_phi(precon_slice) is called BEFORE any USPP
            // modification.
            let mut beta_phi_precon_dev: CudaSlice<CudaComplex> = stream
                .alloc_zeros(ne * n_bands)
                .map_err(Error::Cuda)?;
            unsafe {
                blas.gemm_c64(
                    ZgemmConfig {
                        transa: op::C,
                        transb: op::N,
                        m: ne as i32,
                        n: n_bands as i32,
                        k: n_pw as i32,
                        alpha: CudaComplex { x: 1.0, y: 0.0 },
                        lda: n_pw as i32,
                        ldb: n_pw as i32,
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                        ldc: ne as i32,
                    },
                    &entry.beta_g,
                    &precon_tpa,
                    &mut beta_phi_precon_dev,
                )?;
            }

            // D2H: copy beta_phi_precon to CPU
            let beta_phi_precon_cpu: Vec<CudaComplex> = stream
                .clone_dtoh(&beta_phi_precon_dev)
                .map_err(Error::Cuda)?;

            // ---------------------------------------------------------------
            // Accumulate per-ion data into global arrays
            // ---------------------------------------------------------------
            // Convert to Array2<Complex64> for ndarray arithmetic.
            // CRITICAL: GEMM output is column-major (Fortran order).
            // Use .f() to tell ndarray the memory layout, otherwise
            // element [n,b] reads from the wrong position.
            let beta_phi_psi_arr = Array2::from_shape_vec((ne, n_bands).f(),
                beta_phi_psi_cpu.iter().map(|c| Complex64::new(c.x, c.y)).collect()
            ).expect("beta_phi_psi shape (ne, n_bands) must match data");

            let beta_phi_precon_arr = Array2::from_shape_vec((ne, n_bands).f(),
                beta_phi_precon_cpu.iter().map(|c| Complex64::new(c.x, c.y)).collect()
            ).expect("beta_phi_precon shape (ne, n_bands) must match data");

            // Copy beta_phi_psi into global_beta_phi_psi[offset..offset+ne, :]
            for n in 0..ne {
                for b in 0..n_bands {
                    global_beta_phi_psi[[offset + n, b]] = beta_phi_psi_arr[[n, b]];
                }
            }

            // Copy beta_phi_precon into global_beta_phi_precon[offset..offset+ne, :]
            for n in 0..ne {
                for b in 0..n_bands {
                    global_beta_phi_precon[[offset + n, b]] = beta_phi_precon_arr[[n, b]];
                }
            }

            // Copy R_beta into global_r_beta (diagonal block)
            for m in 0..ne {
                for n in 0..ne {
                    global_r_beta[[offset + m, offset + n]] = r_beta_per_ion[i][[m, n]];
                }
            }

            // ── PRECON CROSS-CHECK (ion 0 only) ──────────────────────────
            // Independently verify that beta_phi_precon from GPU GEMM
            // and the CPU weight computation match, using:
            //  (A) cublasZdotc for <beta_g[0]|precon[0]> vs GEMM
            //  (C) [n,b] double-loop max-norm vs .iter()
            if i == 0 {
                let handle: cublasHandle_t = blas.raw_handle();
                let (beta_ptr, _) = entry.beta_g.device_ptr(stream);
                let (precon_ptr, _) = precon_tpa.device_ptr(stream);
                let n_pw_i32 = n_pw as i32;

                // --- A: beta_phi_precon[0,0] via cublasZdotc ---
                let mut dot_precon = CudaComplex { x: 0.0, y: 0.0 };
                unsafe {
                    cublasZdotc_v2(
                        handle,
                        n_pw_i32,
                        beta_ptr as *const _,
                        1,
                        precon_ptr as *const _,
                        1,
                        &mut dot_precon as *mut _ as *mut _,
                    );
                }
                let gemm_bpp00 = beta_phi_precon_arr[[0, 0]];
                let norm_dot = (dot_precon.x.powi(2) + dot_precon.y.powi(2)).sqrt();
                let diff_bpp = ((gemm_bpp00.re - dot_precon.x).powi(2)
                              + (gemm_bpp00.im - dot_precon.y).powi(2)).sqrt();
                let _reldiff_bpp = if norm_dot > 1e-30 { diff_bpp / norm_dot } else { 0.0 };
                precon_diag!(
                    "[Diag-Precon-xcheck] beta_phi_precon[0,0]: \
                     GEMM=({:.6e},{:.6e}) dotc=({:.6e},{:.6e}) reldiff={:.6e}",
                    gemm_bpp00.re, gemm_bpp00.im,
                    dot_precon.x, dot_precon.y,
                    _reldiff_bpp,
                );

                // --- C: max norms — .iter() vs [n,b] double-loop ---
                let mut max_bpp_iter = 0.0f64;
                for v in beta_phi_precon_arr.iter() { let a = v.norm(); if a > max_bpp_iter { max_bpp_iter = a; } }
                let _max_bps_iter = beta_phi_psi_arr.iter().map(|c| c.norm()).fold(0.0f64, f64::max);
                let mut max_bpp_idx = 0.0f64;
                let mut max_bps_idx = 0.0f64;
                for b in 0..n_bands {
                    for n in 0..ne {
                        let bpp = beta_phi_precon_arr[[n, b]].norm();
                        if bpp > max_bpp_idx { max_bpp_idx = bpp; }
                        let bps = beta_phi_psi_arr[[n, b]].norm();
                        if bps > max_bps_idx { max_bps_idx = bps; }
                    }
                }
                precon_diag!(
                    "[Diag-Precon-xcheck] max via .iter():    |bphi_precon|={:.6e} |bphi_psi|={:.6e}",
                    max_bpp_iter, max_bps_iter,
                );
                precon_diag!(
                    "[Diag-Precon-xcheck] max via [n,b]:     |bphi_precon|={:.6e} |bphi_psi|={:.6e}",
                    max_bpp_idx, max_bps_idx,
                );
            }
        }

        // -----------------------------------------------------------------------
        // PASS 2: Global weight computation (CPU, ndarray)
        // -----------------------------------------------------------------------
        // CASTEP nlpot.f90:16077-16119: weight = Q_RCQ * (E * beta_phi) + R_beta * beta_phi_precon
        //
        // scaled[n,b] = global_beta_phi_psi[n,b] * eigenvalues[b]
        let mut scaled = Array2::<Complex64>::zeros((total_ne, n_bands));
        for b in 0..n_bands {
            let e = eigenvalues_cpu[b];
            for n in 0..total_ne {
                scaled[[n, b]] = global_beta_phi_psi[[n, b]] * e;
            }
        }

        // global_weight = Q_RCQ · scaled + global_r_beta · global_beta_phi_precon
        let global_weight = q_rcq.dot(&scaled) + global_r_beta.dot(&global_beta_phi_precon);

        // Diagnostic: report global weight max |entry|
        {
            let mut max_w = 0.0f64;
            for v in global_weight.iter() {
                let a = v.norm();
                if a > max_w { max_w = a; }
            }
            precon_diag!(
                "[Diag-Precon] global weight max|entry|={:.6e} (Q_RCQ shape {}x{}, total_ne={}, n_bands={})",
                max_w, q_rcq.shape()[0], q_rcq.shape()[1], total_ne, n_bands,
            );
        }

        // -----------------------------------------------------------------------
        // PASS 3: Apply per-ion corrections (extract weight slice, upload, GEMM)
        // -----------------------------------------------------------------------
        for (i, entry) in vnl_data.entries.iter().enumerate() {
            let ne = entry.n_expanded as usize;
            if ne == 0 {
                continue;
            }
            let offset = ion_offsets[i];

            // Extract weight_i = global_weight[offset..offset+ne, :]
            let weight_i = global_weight.slice(ndarray::s![offset..offset + ne, ..]).to_owned();

            // Diag: per-ion weight magnitude
            {
                let mut max_w = 0.0f64;
                for v in weight_i.iter() { let a = v.norm(); if a > max_w { max_w = a; } }
                precon_diag!(
                    "[Diag-Precon] PASS3 ion={}: max|weight|={:.6e} (from global weight slice)",
                    i, max_w,
                );
            }

            // Upload weight_i to GPU (ne × n_bands) in column-major order.
            let mut weight_flat: Vec<CudaComplex> = Vec::with_capacity(ne * n_bands);
            for b in 0..n_bands {
                for n in 0..ne {
                    let c = weight_i[[n, b]];
                    weight_flat.push(CudaComplex { x: c.re, y: c.im });
                }
            }
            let weight_dev = stream
                .clone_htod(&weight_flat)
                .map_err(Error::Cuda)?;

            // ---------------------------------------------------------------
            // Step 5: Apply NL correction with TPA scaling
            // ---------------------------------------------------------------
            // temp = beta_g · weight  (n_pw × n_bands)
            let mut temp_dev: CudaSlice<CudaComplex> = stream
                .alloc_zeros(n_pw * n_bands)
                .map_err(Error::Cuda)?;
            unsafe {
                blas.gemm_c64(
                    ZgemmConfig {
                        transa: op::N,
                        transb: op::N,
                        m: n_pw as i32,
                        n: n_bands as i32,
                        k: ne as i32,
                        alpha: CudaComplex { x: 1.0, y: 0.0 },
                        lda: n_pw as i32,
                        ldb: ne as i32,
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                        ldc: n_pw as i32,
                    },
                    &entry.beta_g,
                    &weight_dev,
                    &mut temp_dev,
                )?;
            }

            // Apply TPA scaling: precon += temp · R(G)
            let temp_pw = PwCoefficients::new(temp_dev);
            unsafe {
                tpa_preconditioner.apply_add(
                    &mut precon,
                    &temp_pw,
                    r_vector,
                    n_pw,
                    n_bands,
                    stream,
                )?;
            }
        }
    }

    Ok(precon)
}

// ---------------------------------------------------------------------------
// Preconditioner preparation builder (USPP assembly pipeline)
// ---------------------------------------------------------------------------

/// Result of `prepare_preconditioner`.
pub struct PreconditionerPrepResult {
    /// TPA preconditioner vector R(G) on CPU.
    pub r_vector: Vec<f64>,
    /// Per-ion R_beta = (−Q⁻¹ − C)⁻¹ matrices.
    pub r_beta_per_ion: Vec<Array2<Complex64>>,
    /// Global Q_RCQ = −Q + C·Q − R_beta·(C·Q) matrix.
    pub q_rcq: Array2<Complex64>,
}

/// Prepare the USPP preconditioner matrices.
///
/// This orchestrates the preconditioner setup for a Davidson outer iteration:
/// 1. Compute R(G) TPA preconditioner vector from kinetic energies
/// 2. Compute C = β^H·diag(R)·β per ion
/// 3. Invert Q⁻¹ per ion (SPD Cholesky inverse)
/// 4. Assemble R_beta = (−Q⁻¹ − C)⁻¹ per ion
/// 5. Assemble Q_RCQ = −Q + C·Q − R_beta·(C·Q) global matrix
///
/// All computation happens on CPU. The caller is responsible for uploading
/// results to GPU as needed.
///
/// Reference: CASTEP nlpot.f90:13525-14736 (nlpot_prepare_precon)
#[bon::builder]
pub fn prepare_preconditioner(
    pw_ek: &[f64],
    mean_ek: f64,
    n_pw: usize,
    beta_g_per_ion: &[Array2<Complex64>],
    q_matrices: &[Vec<f64>],
    ion_n_expanded: &[usize],
    mixture_weights: &[f64],
) -> Result<PreconditionerPrepResult, Error> {
    // 1. Compute R(G) on CPU: R(G) = tpa(pw_ek / mean_ek)
    let r_vector: Vec<f64> = pw_ek
        .iter()
        .take(n_pw)
        .map(|&ek| tpa(ek / mean_ek))
        .collect();

    // 2-4. Per-ion: C matrix, Q⁻¹, R_beta
    let n_ions = beta_g_per_ion.len();
    let mut c_per_ion = Vec::with_capacity(n_ions);
    let mut q_inv_per_ion = Vec::with_capacity(n_ions);
    let mut r_beta_per_ion_vec = Vec::with_capacity(n_ions);
    let mut ion_offsets = Vec::with_capacity(n_ions + 1);
    let mut cum = 0usize;
    ion_offsets.push(0);

    for i in 0..n_ions {
        let ne = ion_n_expanded[i];
        cum += ne;
        ion_offsets.push(cum);

        // Convert Array2<Complex64> from (n_pw, ne) to flat (ne, n_pw) layout
        // for compute_c_matrix
        let beta_shape = beta_g_per_ion[i].shape();
        assert_eq!(
            beta_shape[0], n_pw,
            "beta_g_per_ion[{i}] has {} rows, expected {n_pw}",
            beta_shape[0]
        );
        assert_eq!(
            beta_shape[1], ne,
            "beta_g_per_ion[{i}] has {} cols, expected {ne}",
            beta_shape[1]
        );

        let mut beta_flat: Vec<CudaComplex> = Vec::with_capacity(ne * n_pw);
        for n in 0..ne {
            for g in 0..n_pw {
                let val = beta_g_per_ion[i][[g, n]];
                beta_flat.push(CudaComplex {
                    x: val.re,
                    y: val.im,
                });
            }
        }

        // 2. C = β^H · diag(R) · β
        let c = compute_c_matrix(&beta_flat, &r_vector, ne, n_pw);
        c_per_ion.push(c);

        // 3. Q⁻¹
        let q_inv = invert_q_matrix(&q_matrices[i], ne);
        q_inv_per_ion.push(q_inv);

        // 4. R_beta = (−Q⁻¹ − C)⁻¹
        let r_beta = assemble_r_beta(&q_inv_per_ion[i], &c_per_ion[i], ne, mixture_weights[i]);
        r_beta_per_ion_vec.push(r_beta);
    }

    // 2b. Compute global C with cross-ion blocks
    let c_global = compute_c_global(
        beta_g_per_ion,
        &r_vector,
        &ion_offsets,
    );

    // 5. Q_RCQ = −Q − R_beta·C·Q  (global C, full off-diagonal blocks)
    let q_rcq = assemble_q_rcq(
        q_matrices,
        &c_global,
        &r_beta_per_ion_vec,
        &ion_offsets,
        mixture_weights,
    );

    // Diag: print max |entry| of r_beta and q_rcq, plus per-ion details
    {
        let mut max_r_beta = 0.0f64;
        let mut _max_rb_ion = 0usize;
        for (ion, rb) in r_beta_per_ion_vec.iter().enumerate() {
            for v in rb.iter() {
                let a = Complex64::new(v.re, v.im).norm();
                if a > max_r_beta { max_r_beta = a; _max_rb_ion = ion; }
            }
        }
        let mut max_q_rcq = 0.0f64;
        for v in q_rcq.iter() {
            let a = Complex64::new(v.re, v.im).norm();
            if a > max_q_rcq { max_q_rcq = a; }
        }
        precon_diag!(
            "[Diag-Precon] R_beta max|entry|={:.6e} (ion={})  Q_RCQ max|entry|={:.6e}",
            max_r_beta, _max_rb_ion, max_q_rcq
        );
        // Per-ion R_beta and C diagnostics
        for (ion, rb) in r_beta_per_ion_vec.iter().enumerate() {
            let ne = rb.shape()[0];
            if ne == 0 { continue; }
            let mut max_c = 0.0f64;
            for v in c_per_ion[ion].iter() {
                let a = Complex64::new(v.re, v.im).norm();
                if a > max_c { max_c = a; }
            }
            let mut max_rb = 0.0f64;
            for v in rb.iter() {
                let a = Complex64::new(v.re, v.im).norm();
                if a > max_rb { max_rb = a; }
            }
            if max_rb > 10.0 {
                precon_diag!(
                    "[Diag-Precon] ion={ion} ne={ne}: max|C|={max_c:.4e} max|R_beta|={max_rb:.4e}  \
                     C[0,0]=({c00_re:.4e},{c00_im:.4e})  R_beta[0,0]=({rb00_re:.4e},{rb00_im:.4e})",
                    c00_re=c_per_ion[ion][[0,0]].re, c00_im=c_per_ion[ion][[0,0]].im,
                    rb00_re=rb[[0,0]].re, rb00_im=rb[[0,0]].im,
                );
            }
        }
        // Dump R(G) stats
        let mut r_min = f64::MAX;
        let mut r_max = 0.0f64;
        let mut _r_mean = 0.0f64;
        for &r in &r_vector {
            if r < r_min { r_min = r; }
            if r > r_max { r_max = r; }
            _r_mean += r;
        }
        _r_mean /= r_vector.len() as f64;
        precon_diag!(
            "[Diag-Precon] R(G) tpa: min={:.4e} max={:.6e} mean={:.4e}",
            r_min, r_max, _r_mean
        );
    }

    Ok(PreconditionerPrepResult {
        r_vector,
        r_beta_per_ion: r_beta_per_ion_vec,
        q_rcq,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the correct TPA-apply residual formula on CPU.
    ///
    /// Formula: `out[G,b] = (hpsi[G,b] − e[b]·psi[G,b]) · R(G)`
    ///
    /// This tests the element-wise fused operation that replaces the old
    /// incorrect `1/(T−λ)` preconditioner. The R-vector is the TPA scalar
    /// applied per PW (same for all bands).
    #[test]
    fn test_tpa_apply_residual_formula() {
        let n_pw = 4;
        let n_bands = 3;
        let eigenvalues = vec![0.5, 1.0, 1.5];
        let r_vector = vec![0.8, 0.6, 0.4, 0.2];

        // Band 0: psi=(1,0), hpsi=(2,0.5)
        // Band 1: psi=(0,1), hpsi=(-1,2)
        // Band 2: psi=(0.5,-0.5), hpsi=(0,0)
        let psi: Vec<CudaComplex> = vec![
            CudaComplex { x: 1.0, y: 0.0 }, CudaComplex { x: 0.0, y: 1.0 }, CudaComplex { x: 0.5, y: -0.5 }, CudaComplex { x: -1.0, y: 2.0 }, // PW 0 across bands
            CudaComplex { x: 0.0, y: -1.0 }, CudaComplex { x: 1.0, y: 0.0 }, CudaComplex { x: 0.0, y: 0.5 }, CudaComplex { x: 2.0, y: -1.0 }, // PW 1
            CudaComplex { x: 0.5, y: 0.5 }, CudaComplex { x: -1.0, y: 1.0 }, CudaComplex { x: 1.0, y: 0.0 }, CudaComplex { x: 0.0, y: -2.0 }, // PW 2
            CudaComplex { x: -0.5, y: 1.0 }, CudaComplex { x: 0.5, y: -0.5 }, CudaComplex { x: -1.0, y: 0.0 }, CudaComplex { x: 1.0, y: 0.5 }, // PW 3
        ];
        let hpsi: Vec<CudaComplex> = vec![
            CudaComplex { x: 2.0, y: 0.5 }, CudaComplex { x: -1.0, y: 2.0 }, CudaComplex { x: 0.0, y: 0.0 }, CudaComplex { x: 3.0, y: 1.0 },
            CudaComplex { x: 1.0, y: -0.5 }, CudaComplex { x: 0.5, y: 1.5 }, CudaComplex { x: 2.0, y: -1.0 }, CudaComplex { x: -1.0, y: 0.0 },
            CudaComplex { x: -1.0, y: 1.0 }, CudaComplex { x: 1.0, y: -2.0 }, CudaComplex { x: 0.5, y: 0.5 }, CudaComplex { x: 2.0, y: 1.5 },
            CudaComplex { x: 0.0, y: -1.0 }, CudaComplex { x: 2.0, y: 1.0 }, CudaComplex { x: 1.5, y: 0.0 }, CudaComplex { x: -2.0, y: 1.0 },
        ];

        // Path A (fused): out[G,b] = (hpsi[G,b] - e[b]*psi[G,b]) * R(G)
        let mut fused = vec![CudaComplex { x: 0.0, y: 0.0 }; n_pw * n_bands];
        for b in 0..n_bands {
            let e = eigenvalues[b];
            for g in 0..n_pw {
                let i = b * n_pw + g;
                let r = r_vector[g];
                let hx = hpsi[i].x;
                let hy = hpsi[i].y;
                let px = psi[i].x;
                let py = psi[i].y;
                fused[i].x = (hx - e * px) * r;
                fused[i].y = (hy - e * py) * r;
            }
        }

        // Path B (decomposed): residual first, then scale by R
        // Step B-a: residual[G,b] = hpsi[G,b] - e[b]*spsi[G,b]
        let mut residual = vec![CudaComplex { x: 0.0, y: 0.0 }; n_pw * n_bands];
        for b in 0..n_bands {
            let e = eigenvalues[b];
            for g in 0..n_pw {
                let i = b * n_pw + g;
                residual[i].x = hpsi[i].x - e * psi[i].x;
                residual[i].y = hpsi[i].y - e * psi[i].y;
            }
        }
        // Step B-b: result[G,b] = residual[G,b] * R(G)
        let mut result = vec![CudaComplex { x: 0.0, y: 0.0 }; n_pw * n_bands];
        for i in 0..(n_pw * n_bands) {
            let g = i % n_pw;
            let r = r_vector[g];
            result[i].x = residual[i].x * r;
            result[i].y = residual[i].y * r;
        }

        // Cross-path verification: fused == result
        for i in 0..(n_pw * n_bands) {
            let diff_x = (fused[i].x - result[i].x).abs();
            let diff_y = (fused[i].y - result[i].y).abs();
            assert!(
                diff_x < 1e-15,
                "Element {i} x mismatch: fused={:.6e}, decomposed={:.6e}, diff={:.2e}",
                fused[i].x, result[i].x, diff_x,
            );
            assert!(
                diff_y < 1e-15,
                "Element {i} y mismatch: fused={:.6e}, decomposed={:.6e}, diff={:.2e}",
                fused[i].y, result[i].y, diff_y,
            );
        }
    }

    /// Verify the TPA-apply add formula on CPU.
    ///
    /// Formula: `out[G,b] += correction[G,b] · R(G)`
    #[test]
    fn test_tpa_apply_add_formula() {
        let n_pw = 3;
        let n_bands = 2;
        let r_vector = vec![0.9, 0.5, 0.1];
        let total = n_pw * n_bands;

        // Initial out values
        let out: Vec<CudaComplex> = (0..total)
            .map(|i| CudaComplex {
                x: (i as f64) * 1.0,
                y: (i as f64) * 2.0,
            })
            .collect();

        let correction: Vec<CudaComplex> = (0..total)
            .map(|i| CudaComplex {
                x: (i as f64) * 0.5,
                y: (i as f64) * 0.25,
            })
            .collect();

        // Path A (fused): out[G,b] += correction[G,b] * R(G)
        let start_out = out.clone();
        let mut fused = out;
        for i in 0..total {
            let g = i % n_pw;
            let r = r_vector[g];
            fused[i].x += correction[i].x * r;
            fused[i].y += correction[i].y * r;
        }

        // Path B (decomposed): scale correction first, then add
        // Step B-a: scaled_correction[G,b] = correction[G,b] * R(G)
        let mut scaled_correction = vec![CudaComplex { x: 0.0, y: 0.0 }; total];
        for i in 0..total {
            let g = i % n_pw;
            let r = r_vector[g];
            scaled_correction[i].x = correction[i].x * r;
            scaled_correction[i].y = correction[i].y * r;
        }
        // Step B-b: result[G,b] = initial_out[G,b] + scaled_correction[G,b]
        let mut result = start_out;
        for i in 0..total {
            result[i].x += scaled_correction[i].x;
            result[i].y += scaled_correction[i].y;
        }

        // Cross-path verification: fused == result
        for i in 0..total {
            let diff_x = (fused[i].x - result[i].x).abs();
            let diff_y = (fused[i].y - result[i].y).abs();
            assert!(
                diff_x < 1e-15,
                "Element {i} x mismatch: fused={:.6e}, decomposed={:.6e}, diff={:.2e}",
                fused[i].x, result[i].x, diff_x,
            );
            assert!(
                diff_y < 1e-15,
                "Element {i} y mismatch: fused={:.6e}, decomposed={:.6e}, diff={:.2e}",
                fused[i].y, result[i].y, diff_y,
            );
        }
    }

    /// Verify the TPA scalar function `tpa(x)` matches known values.
    #[test]
    fn test_tpa_scalar() {
        // tpa(0) should be exactly 1.0
        assert!((tpa(0.0) - 1.0).abs() < 1e-15, "tpa(0) = {}, expected 1.0", tpa(0.0));
        // tpa(x) should be in (0, 1] for finite x >= 0
        assert!(tpa(1.0) > 0.0 && tpa(1.0) <= 1.0, "tpa(1) = {} out of range", tpa(1.0));
        assert!(tpa(10.0) > 0.0 && tpa(10.0) <= 1.0, "tpa(10) = {} out of range", tpa(10.0));
        assert!(tpa(100.0) > 0.0 && tpa(100.0) <= 1.0, "tpa(100) = {} out of range", tpa(100.0));
        // tpa(x) should be monotonically decreasing for x >= 0
        assert!(tpa(0.0) > tpa(1.0), "tpa not decreasing at 0->1");
        assert!(tpa(1.0) > tpa(10.0), "tpa not decreasing at 1->10");
        assert!(tpa(10.0) > tpa(100.0), "tpa not decreasing at 10->100");

        // Anchored value: tpa(1.0) = 65/81
        //   numerator = 16*1^4 = 16
        //   denominator = 27 + 1*(18 + 1*(12 + 8*1)) = 27 + 18 + 12 + 8 = 65
        //   tpa(1.0) = 1 / (1 + 16/65) = 1 / (81/65) = 65/81
        let expected_tpa_1 = 65.0 / 81.0;
        assert!(
            (tpa(1.0) - expected_tpa_1).abs() < 1e-15,
            "tpa(1.0) = {:.15e}, expected 65/81 = {:.15e}",
            tpa(1.0),
            expected_tpa_1
        );
    }

    /// GPU vs CPU: TPA preconditioner apply_residual at realistic scale.
    ///
    /// The kernel formula is:  out[b,G] = (Hψ[b,G] - ε[b]·ψ[b,G]) · r_vector[G]
    ///
    /// This is a fused multiply-subtract-scale — tests whether GPU FMA
    /// (fused multiply-add) produces different results from separate CPU
    /// operations for the preconditioned residual.
    #[test]
    fn tpa_apply_residual_realistic_scale() {
        use crate::device::CudaComplex;
        use cudarc::driver::{CudaContext, CudaStream};
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();
        let ctx_arc = Arc::new(ctx.clone());

        // Realistic Davidson block size
        let n_pw = 60067usize;
        let n_bands = 25;
        let total = n_pw * n_bands;

        // Generate deterministic pseudo-random data
        let psi: Vec<CudaComplex> = (0..total)
            .map(|idx| {
                let phase = (idx as f64 * 0.987654321).sin() * 1000.0;
                CudaComplex {
                    x: (phase * 1.3).sin() * 1e-3,
                    y: (phase * 1.7).cos() * 1e-3,
                }
            }).collect();
        let hpsi: Vec<CudaComplex> = (0..total)
            .map(|idx| {
                let phase = ((idx + 500) as f64 * 0.987654321).sin() * 1000.0;
                CudaComplex {
                    x: (phase * 1.9).cos() * 1e-3,
                    y: (phase * 1.1).sin() * 1e-3,
                }
            }).collect();
        // Realistic eigenvalues: -1.0 to +0.1 Ha
        let eigenvalues: Vec<f64> = (0..n_bands)
            .map(|b| -1.0 + (b as f64) * 1.1 / (n_bands - 1) as f64)
            .collect();
        // TPA r_vector: real kinetic energies on G-grid
        let r_vector: Vec<f64> = (0..n_pw)
            .map(|g| {
                let x = (g as f64) / (n_pw as f64) * 10.0;
                tpa(x) // use the real TPA function
            })
            .collect();

        // CPU reference: out[b,G] = (Hψ[b,G] - ε[b]·ψ[b,G]) · r_vector[G]
        let mut cpu_out = vec![CudaComplex { x: 0.0, y: 0.0 }; total];
        for b in 0..n_bands {
            let e = eigenvalues[b];
            for g in 0..n_pw {
                let i = b * n_pw + g;
                let r = r_vector[g];
                cpu_out[i].x = (hpsi[i].x - e * psi[i].x) * r;
                cpu_out[i].y = (hpsi[i].y - e * psi[i].y) * r;
            }
        }

        // GPU: upload data and run kernel
        use crate::eigensolver::davidson_types::{PwCoefficients, PreconditionerVector};
        let stream_arc = Arc::new(stream.clone());

        let mut psi_dev = PwCoefficients::new(
            stream_arc.alloc_zeros::<CudaComplex>(total).expect("alloc psi"));
        let mut hpsi_dev = PwCoefficients::new(
            stream_arc.alloc_zeros::<CudaComplex>(total).expect("alloc hpsi"));
        let mut out_dev = PwCoefficients::new(
            stream_arc.alloc_zeros::<CudaComplex>(total).expect("alloc out"));
        let mut eig_dev = stream_arc.alloc_zeros::<f64>(n_bands).expect("alloc eig");
        let mut rvec_dev = PreconditionerVector::new(
            stream_arc.alloc_zeros::<f64>(n_pw).expect("alloc rvec"));
        stream_arc.memcpy_htod(&psi, &mut psi_dev.0).expect("H2D psi");
        stream_arc.memcpy_htod(&hpsi, &mut hpsi_dev.0).expect("H2D hpsi");
        stream_arc.memcpy_htod(&eigenvalues, &mut eig_dev).expect("H2D eig");
        stream_arc.memcpy_htod(&r_vector, &mut rvec_dev.0).expect("H2D rvec");

        let tpa = super::TpaPreconditioner::new(&ctx_arc).expect("TpaPreconditioner");
        unsafe {
            tpa.apply_residual(
                &mut out_dev, &psi_dev, &hpsi_dev, &eig_dev, &rvec_dev,
                n_pw, n_bands, &stream_arc,
            ).expect("GPU apply_residual");
        }
        stream_arc.synchronize().expect("sync");
        let gpu_out: Vec<CudaComplex> = stream_arc.clone_dtoh(&out_dev.0).expect("D2H");
        stream.synchronize().expect("sync");
        let gpu_out: Vec<CudaComplex> = stream_arc.clone_dtoh(&out_dev.0).expect("D2H");

        // Compare element-wise
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut sum_signed = 0.0f64;
        for i in 0..total {
            let cpu = cpu_out[i];
            let gpu = gpu_out[i];
            let abs_diff = ((cpu.x - gpu.x).powi(2) + (cpu.y - gpu.y).powi(2)).sqrt();
            let cpu_norm = (cpu.x.powi(2) + cpu.y.powi(2)).sqrt();
            let rel = if cpu_norm > 1e-30 { abs_diff / cpu_norm } else { abs_diff };
            if abs_diff > max_abs { max_abs = abs_diff; }
            if rel > max_rel { max_rel = rel; }
            sum_signed += gpu.x - cpu.x;
        }
        let mean_signed = sum_signed / total as f64;

        let status = if max_rel > 1e-6 { "HIGH" }
            else if max_rel > 1e-9 { "WARN" }
            else { "ok" };
        eprintln!(
            "TPA apply_residual (n_pw={n_pw}, n_bands={n_bands}): \
             max|Δ|={:.3e} max rel={:.3e} mean_signed={:+.3e} [{status}]",
            max_abs, max_rel, mean_signed,
        );

        assert!(
            max_rel < 1e-9,
            "TPA apply_residual: max rel error {:.3e} exceeds 1e-9.\n\
             GPU FMA in fused multiply-subtract-scale produces different\n\
             per-element results vs CPU separate operations.",
            max_rel,
        );
        assert!(
            mean_signed.abs() < 1e-15,
            "TPA apply_residual: systematic bias {:.3e} detected.\n\
             GPU FMA is systematically biasing the preconditioned residual.",
            mean_signed,
        );
    }
}
