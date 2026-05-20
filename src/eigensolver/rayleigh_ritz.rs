// ---------------------------------------------------------------------------
// Rayleigh-Ritz subspace diagonalization for SCF eigensolver
// ---------------------------------------------------------------------------
//
// Given filtered wavefunctions psi (RowDistributed) and H|psi> (RowDistributed),
// solve the generalized eigenvalue problem in the subspace:
//   H_sub * X = lambda * S_sub * X
// where H_sub = psi^dag * H|psi> and S_sub = psi^dag * psi.
//
// Then rotate psi to the new eigenbasis and extract eigenvalues.

use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::cusolver::sys::{cublasFillMode_t, cusolverEigMode_t};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use ndarray::ShapeBuilder;

use crate::device::blas::{op, ZgemmConfig};
use crate::device::blas::BlasHandle;
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::device::{CudaComplex, Gpu};
use crate::eigensolver::chebyshev::CudaKernelSet;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::layout::{ColumnDistributed, Cpu, RowDistributed, WavefunctionSet};
use crate::types::Error;

// ---------------------------------------------------------------------------
// Type alias for the complex Rayleigh-Ritz return type
// ---------------------------------------------------------------------------

type RayleighRitzResult = Result<
    (
        Gpu<WavefunctionSet<ColumnDistributed>>,
        Cpu<Vec<f64>>,
        Cpu<Vec<ndarray::Array2<num_complex::Complex64>>>,
    ),
    Error,
>;

/// Solve the Rayleigh-Ritz generalized eigenvalue problem in the subspace.
///
/// Input:
/// - `psi_row`: filtered wavefunctions in RowDistributed layout (n_pw x n_bands)
/// - `hpsi_row`: H|psi> in RowDistributed layout (n_pw x n_bands)
/// - `vnl_data`: precomputed V_NL data (beta_g, D, Q matrices per ion)
///
/// The overlap matrix S_sub includes the USPP augmentation:
///   S_sub = psi^dag·psi  +  Σ_ion C_proj^dag · q · C_proj
/// where C_proj = beta_g^H · psi and q is the expanded Q augmentation matrix.
///
/// Output:
/// - `psi_col`: rotated wavefunctions in ColumnDistributed layout (n_bands x n_pw)
/// - `Cpu(eigenvalues)`: converged eigenvalues as a Vec<f64>
#[allow(clippy::too_many_arguments)]
pub(crate) fn rayleigh_ritz(
    psi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    hpsi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    vnl_data: &VnlBatchData,
    n_bands: usize,
    n_pw: usize,
    kernels: &CudaKernelSet,
    pcie: &mut PcieAccount,
    solver: &SolverHandle,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> RayleighRitzResult {
    let n = n_bands as i32;
    let k = n_pw as i32;

    // ---- Step 1: H_sub = psi^dag * hpsi  (n_bands x n_bands) ----
    // psi_row is (n_pw x n_bands) col-major, so psi^dag is (n_bands x n_pw)
    // Using gemm with transa=C (conj-transpose psi), transb=N (hpsi as-is)
    // C(m,n) = alpha * op(A)(m,k) * op(B)(k,n) + beta * C(m,n)
    // H_sub(n_bands, n_bands) = conj(psi_row^T)(n_bands, n_pw) * hpsi_row(n_pw, n_bands)
    //                        = psi_dag * hpsi
    // psi_row: lda = n_pw (col-major, n_pw rows)
    // hpsi_row: ldb = n_pw
    // H_sub: ldc = n_bands
    let mut h_sub_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_bands).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,   // conj(psi^T)
                transb: op::N,   // hpsi
                m: n,            // rows = n_bands
                n,               // cols = n_bands
                k,               // inner dim = n_pw
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,          // lda of psi (n_pw rows, col-major)
                ldb: k,          // ldb of hpsi (n_pw rows, col-major)
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n,          // ldc of H_sub (n_bands rows, col-major)
            },
            psi_row.as_device_slice(),
            hpsi_row.as_device_slice(),
            &mut h_sub_dev,
        )?;
    }

    // ---- Step 2: S_sub = psi^dag * psi  (n_bands x n_bands) ----
    // Bare plane-wave overlap (missing USPP augmentation — added below).
    let mut s_sub_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_bands).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,   // conj(psi^T)
                transb: op::N,   // psi
                m: n,
                n,
                k,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,
                ldb: k,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n,
            },
            psi_row.as_device_slice(),
            psi_row.as_device_slice(),
            &mut s_sub_dev,
        )?;
    }

    // ---- Step 2b: Add USPP S-augmentation for each ion ----
    // S_sub += Σ_ion C_proj^dag · q · C_proj
    // where C_proj = beta_g^H · psi_row is the projector-wavefunction overlap.
    let psi_slice = psi_row.as_device_slice();
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // C_proj = beta_g^H · psi_row  (n_expanded × n_bands)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n,
                    k,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: k,   // beta_g is (ne, n_pw) row-major → col-major (n_pw, ne)
                    ldb: k,   // psi_row is (n_pw, n_bands)
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_slice,
                &mut c_proj,
            )?;
        }

        // temp = q · C_proj  (n_expanded × n_bands)
        let mut temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::N,
                    transb: op::N,
                    m: ne,
                    n,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.q_matrix,
                &c_proj,
                &mut temp,
            )?;
        }

        // S_sub += C_proj^dag · temp  (n_bands × n_bands, accumulated)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: n,
                    n,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 }, // accumulate into S_sub
                    ldc: n,
                },
                &c_proj,
                &temp,
                &mut s_sub_dev,
            )?;
        }
    }

    // ---- Step 3: Solve generalized eigenvalue problem via ZHEGVD ----
    // A * X = lambda * B * X  where A = H_sub, B = S_sub
    // On return: h_sub_dev contains eigenvectors X (column-major, n x n)
    let mut eigenvalues_dev: CudaSlice<f64> =
        stream.alloc_zeros(n_bands).map_err(Error::Cuda)?;
    let mut info_dev: CudaSlice<i32> = stream.alloc_zeros(1).map_err(Error::Cuda)?;

    solver.zhegvd(
        cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        n,
        &mut h_sub_dev,
        &mut s_sub_dev,
        &mut eigenvalues_dev,
        &mut info_dev,
    )?;

    // Check solver info
    stream.synchronize().map_err(Error::Cuda)?;
    let info: Vec<i32> = stream.clone_dtoh(&info_dev).map_err(Error::Cuda)?;
    if info[0] != 0 {
        return Err(Error::RayleighRitzFailed { info: info[0] });
    }

    // ---- Step 4-5: Rotate psi_new = psi_row · X ----
    //
    // psi_row memory is col-major (n_pw, n_bands): each col is a band ψ_b.
    // X (in h_sub_dev) is col-major (n_bands, n_bands): eigenvectors of (H,S).
    // The rotated wavefunctions ψ'_k = Σ_b X[b, k] · ψ_b correspond to
    // (psi_row · X)[g, k] in col-major (n_pw, n_bands).
    //
    // The result is then in the same layout as ColumnDistributed
    // (which is also col-major (n_pw, n_bands) — see comment in
    // `chebyshev.rs` after the apply_full_hamiltonian transpose-skip).
    //
    // gemm: C(m,n) = α A(m,k) B(k,n)
    //   m = n_pw, n = n_bands, k = n_bands (inner)
    //   A = psi_row, lda = n_pw      (col-major (n_pw, n_bands))
    //   B = X,       ldb = n_bands   (col-major (n_bands, n_bands))
    //   C = psi_new, ldc = n_pw      (col-major (n_pw, n_bands))
    let mut psi_new_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_pw).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::N,
                transb: op::N,
                m: k,            // n_pw
                n,               // n_bands
                k: n,            // n_bands (inner)
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,          // psi_row leading dim = n_pw
                ldb: n,          // X leading dim = n_bands
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: k,          // psi_new leading dim = n_pw
            },
            psi_row.as_device_slice(),
            &h_sub_dev,
            &mut psi_new_dev,
        )?;
    }

    // ---- Step 5b: Project β_g^H · ψ_new per ion (USPP augmentation density) ----
    //
    // For each ion I, compute βψ_I = β_g^H · ψ_new with shape (n_expanded × n_bands).
    // These projections feed `compute_aug_density_fine` which accumulates the
    // occupancy matrix ω^I_{nm} = Σ_b occ_b · conj(βψ_I)_{n,b} · (βψ_I)_{m,b}.
    //
    // Layout: β_g is col-major (n_pw, n_expanded), ψ_new is col-major
    // (n_pw, n_bands). The gemm with transa=C gives (n_expanded × n_bands)
    // col-major, which maps to an ndarray Array2 with shape (n_expanded,
    // n_bands) in column-major order. We copy to host as a flat Vec and
    // construct Array2 via `from_shape_vec` with `.f()` layout to preserve
    // the column-major memory order.
    let mut beta_psi_per_ion: Vec<ndarray::Array2<num_complex::Complex64>> =
        Vec::with_capacity(vnl_data.entries.len());
    let psi_new_slice: &CudaSlice<CudaComplex> = &psi_new_dev;
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;
        let mut bp_dev: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n,
                    k,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: k,
                    ldb: k,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_new_slice,
                &mut bp_dev,
            )?;
        }
        let bp_host: Vec<CudaComplex> = stream.clone_dtoh(&bp_dev).map_err(Error::Cuda)?;
        pcie.d2h_bytes += bp_host.len() * std::mem::size_of::<CudaComplex>();
        let bp_complex: Vec<num_complex::Complex64> = bp_host
            .iter()
            .map(|c| num_complex::Complex64::new(c.x, c.y))
            .collect();
        let arr = ndarray::Array2::from_shape_vec(
            (ne as usize, n_bands).f(),
            bp_complex,
        )
        .map_err(|_| Error::NotImplemented)?;
        beta_psi_per_ion.push(arr);
    }

    // ---- Step 6: D2H eigenvalues ----
    let eigenvalues: Vec<f64> = stream
        .clone_dtoh(&eigenvalues_dev)
        .map_err(Error::Cuda)?;
    pcie.d2h_bytes += eigenvalues.len() * 8;

    stream.synchronize().map_err(Error::Cuda)?;

    // Wrap psi_new into Gpu<WavefunctionSet<ColumnDistributed>>
    let psi_new_gpu = Gpu::<WavefunctionSet<ColumnDistributed>> {
        slice: psi_new_dev,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };

    Ok((psi_new_gpu, Cpu(eigenvalues), Cpu(beta_psi_per_ion)))
}
