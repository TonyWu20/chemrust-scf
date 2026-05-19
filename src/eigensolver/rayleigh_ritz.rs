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

use crate::device::blas::{op, ZgemmConfig};
use crate::device::blas::BlasHandle;
use crate::device::solver::SolverHandle;
use crate::device::{CudaComplex, Gpu};
use crate::layout::{ColumnDistributed, Cpu, RowDistributed, WavefunctionSet};
use crate::types::Error;

// ---------------------------------------------------------------------------
// Type alias for the complex Rayleigh-Ritz return type
// ---------------------------------------------------------------------------

type RayleighRitzResult = Result<
    (
        Gpu<WavefunctionSet<ColumnDistributed>>,
        Cpu<Vec<f64>>,
    ),
    Error,
>;

/// Solve the Rayleigh-Ritz generalized eigenvalue problem in the subspace.
///
/// Input:
/// - `psi_row`: filtered wavefunctions in RowDistributed layout (n_pw x n_bands)
/// - `hpsi_row`: H|psi> in RowDistributed layout (n_pw x n_bands)
///
/// Output:
/// - `psi_col`: rotated wavefunctions in ColumnDistributed layout (n_bands x n_pw)
/// - `Cpu(eigenvalues)`: converged eigenvalues as a Vec<f64>
#[allow(clippy::too_many_arguments)]
pub(crate) fn rayleigh_ritz(
    psi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    hpsi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    n_bands: usize,
    n_pw: usize,
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

    // ---- Step 4: Transpose psi from RowDistributed to ColumnDistributed ----
    // This is a re-interpretation: we need psi in (n_bands x n_pw) col-major.
    // Currently psi_row has data as col-major (n_pw x n_bands).
    // The same memory reinterpreted as (n_bands x n_pw) requires a transpose.
    //
    // For now: copy and transpose using a new buffer.

    // We already have the RowDistributed data. We need ColumnDistributed.
    // ColumnDistributed has data laid out as col-major (n_bands, n_pw), lda = n_bands.
    // We get this by transposing the (n_pw, n_bands) matrix.
    //
    // psi_col_data[b * n_pw + g] = psi_row_data[g * n_bands + b]

    // For simplicity, do this with a D2H + H2D roundtrip (fine for moderate sizes).
    // A GPU kernel would be better for large systems; this avoids complexity for Phase 2.
    let psi_row_host: Vec<CudaComplex> = stream
        .clone_dtoh(psi_row.as_device_slice())
        .map_err(Error::Cuda)?;

    let mut psi_col_host = vec![
        CudaComplex { x: 0.0, y: 0.0 };
        n_bands * n_pw
    ];

    // Transpose: col-major (n_pw x n_bands) → col-major (n_bands x n_pw)
    for b in 0..n_bands {
        for g in 0..n_pw {
            psi_col_host[b * n_pw + g] = psi_row_host[g * n_bands + b];
        }
    }

    let psi_col_dev: CudaSlice<CudaComplex> = stream
        .clone_htod(&psi_col_host)
        .map_err(Error::Cuda)?;

    // ---- Step 5: Rotate psi_new = X * psi_col ----
    // psi_col is (n_bands x n_pw), X = eigenvectors is (n_bands x n_bands) in h_sub_dev
    // psi_new(n_bands, n_pw) = X(n_bands, n_bands) * psi_col(n_bands, n_pw)
    //
    // gemm: C(m,n) = alpha * A(m,k) * B(k,n) + beta * C(m,n)
    // psi_new(n_bands, n_pw) = X(n_bands, n_bands) * psi_col(n_bands, n_pw)
    // m=n_bands, k=n_bands, n=n_pw
    // X is col-major (n x n), lda = n
    // psi_col is col-major (n x k/k=n_pw), ldb = n
    // psi_new is col-major (n x n_pw), ldc = n
    let mut psi_new_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_bands * n_pw).map_err(Error::Cuda)?;

    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::N,   // X as-is
                transb: op::N,   // psi_col as-is
                m: n,            // n_bands
                n: k,            // n_pw
                k: n,            // n_bands (inner dim)
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n,          // X is n x n, col-major
                ldb: n,          // psi_col is n x n_pw, col-major
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n,          // psi_new is n x n_pw, col-major
            },
            &h_sub_dev,   // contains eigenvector matrix X
            &psi_col_dev,
            &mut psi_new_dev,
        )?;
    }

    // ---- Step 6: D2H eigenvalues ----
    let eigenvalues: Vec<f64> = stream
        .clone_dtoh(&eigenvalues_dev)
        .map_err(Error::Cuda)?;

    stream.synchronize().map_err(Error::Cuda)?;

    // Wrap psi_new into Gpu<WavefunctionSet<ColumnDistributed>>
    let psi_new_gpu = Gpu::<WavefunctionSet<ColumnDistributed>> {
        slice: psi_new_dev,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };

    Ok((psi_new_gpu, Cpu(eigenvalues)))
}
