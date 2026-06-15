// ---------------------------------------------------------------------------
// cuSOLVER wrapper — dense eigenvalue solvers (ZHEEVD, ZPOTRF, ZPOTRS)
//
// ZHEGVD is gated behind `chebyshev` (suspended path). The Davidson solver
// uses ZHEEVD (standard EVP) matching CASTEP's algor_diagonalise.
// ---------------------------------------------------------------------------

use std::sync::Arc;

use cudarc::cusolver::result::CusolverError;
use cudarc::cusolver::safe::DnHandle;
use cudarc::cusolver::sys::{
    self as sys, cublasFillMode_t, cublasOperation_t, cusolverEigMode_t,
};
#[cfg(any(test, feature = "chebyshev"))]
use cudarc::cusolver::sys::cusolverEigType_t;
use cudarc::driver::{result::DriverError, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use thiserror::Error;

use super::CudaComplex;

#[derive(Error, Debug)]
pub enum SolverError {
    #[error("cuSOLVER error: {0}")]
    Cusolver(#[from] CusolverError),
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
}

/// Wrapper around the cuSOLVER dense handle for our SCF eigenvalue problems.
pub struct SolverHandle {
    inner: DnHandle,
    stream: Arc<CudaStream>,
}

impl SolverHandle {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, SolverError> {
        let inner = DnHandle::new(stream.clone())?;
        Ok(Self { inner, stream })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Solve `A·X = λ·B·X` via ZHEGVD (generalized EVP).
    ///
    /// Only used by the suspended Chebyshev path (`rayleigh_ritz`).
    /// Davidson solver uses `zheevd` (standard EVP) matching CASTEP.
    ///
    /// - `a`, `b` are overwritten. On return `a` contains eigenvectors.
    /// - `eigenvalues[0..n-1]` filled on return.
    /// - `info[0]` must be 0 for success.
    #[allow(clippy::too_many_arguments)]
    #[cfg(any(test, feature = "chebyshev"))]
    pub fn zhegvd(
        &self,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: &mut CudaSlice<CudaComplex>,
        b: &mut CudaSlice<CudaComplex>,
        eigenvalues: &mut CudaSlice<f64>,
        info: &mut CudaSlice<i32>,
    ) -> Result<(), SolverError> {
        let handle = self.inner.cu();
        let itype = cusolverEigType_t::CUSOLVER_EIG_TYPE_1;

        unsafe {
            // Cast CudaComplex <-> cuDoubleComplex (same layout, different Rust types)
            let a_raw = a.device_ptr_mut(&self.stream).0 as *mut sys::cuDoubleComplex;
            let b_raw = b.device_ptr_mut(&self.stream).0 as *mut sys::cuDoubleComplex;
            let w_raw = eigenvalues.device_ptr_mut(&self.stream).0 as *mut f64;
            let info_raw = info.device_ptr_mut(&self.stream).0 as *mut i32;

            // Query workspace size
            let mut lwork: i32 = 0;
            sys::cusolverDnZhegvd_bufferSize(
                handle, itype, jobz, uplo, n,
                a_raw as *const _, n,
                b_raw as *const _, n,
                w_raw as *const _,
                &mut lwork as *mut _,
            )
            .result()?;

            // Allocate workspace
            let workspace = self.stream.alloc_zeros::<CudaComplex>(lwork as usize)?;
            let work_raw = workspace.device_ptr(&self.stream).0 as *const sys::cuDoubleComplex;

            // Solve
            sys::cusolverDnZhegvd(
                handle, itype, jobz, uplo, n,
                a_raw, n,
                b_raw, n,
                w_raw,
                work_raw as *mut _, lwork,
                info_raw,
            )
            .result()?;

            Ok(())
        }
    }

    /// Solve `A·X = X·Λ` via ZHEEVD (standard Hermitian EVP).
    ///
    /// CASTEP `algor_diagonalise` solves the STANDARD eigenvalue problem on
    /// S-orthonormalized superspace vectors (hamiltonian.f90:476-480).  We
    /// match this by using ZHEEVD on H_sub directly — no overlap matrix.
    /// This is more robust than ZHEGVD(B=I) when H_sub has extreme eigenvalue
    /// spread (cold-start), because ZHEEVD uses QR iteration rather than
    /// Divide-and-Conquer, avoiding info=N convergence failures.
    ///
    /// - `a` overwritten with eigenvectors on return.
    /// - `eigenvalues[0..n-1]` filled on return.
    /// - `info[0]` must be 0 for success.
    pub fn zheevd(
        &self,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: &mut CudaSlice<CudaComplex>,
        eigenvalues: &mut CudaSlice<f64>,
        info: &mut CudaSlice<i32>,
    ) -> Result<(), SolverError> {
        let handle = self.inner.cu();

        unsafe {
            let a_raw = a.device_ptr_mut(&self.stream).0 as *mut sys::cuDoubleComplex;
            let w_raw = eigenvalues.device_ptr_mut(&self.stream).0 as *mut f64;
            let info_raw = info.device_ptr_mut(&self.stream).0 as *mut i32;

            // Query workspace size
            let mut lwork: i32 = 0;
            sys::cusolverDnZheevd_bufferSize(
                handle, jobz, uplo, n,
                a_raw as *const _, n,
                w_raw as *const _,
                &mut lwork as *mut _,
            )
            .result()?;

            // Allocate workspace
            let workspace = self.stream.alloc_zeros::<CudaComplex>(lwork as usize)?;
            let work_raw = workspace.device_ptr(&self.stream).0 as *const sys::cuDoubleComplex;

            // Solve
            sys::cusolverDnZheevd(
                handle, jobz, uplo, n,
                a_raw, n,
                w_raw,
                work_raw as *mut _, lwork,
                info_raw,
            )
            .result()?;

            Ok(())
        }
    }

    /// Cholesky factorization `A = L·L^H` (or `U^H·U`) in-place.
    ///
    /// - `uplo`: `CUBLAS_FILL_MODE_LOWER` → L·L^H, `CUBLAS_FILL_MODE_UPPER` → U^H·U.
    /// - `a` is overwritten with the triangular factor.
    /// - `info[0]` must be 0 for success; `info[0] = k` means the k-th leading
    ///   minor is not positive definite.
    pub fn zpotrf(
        &self,
        uplo: cublasFillMode_t,
        n: i32,
        a: &mut CudaSlice<CudaComplex>,
        info: &mut CudaSlice<i32>,
    ) -> Result<(), SolverError> {
        let handle = self.inner.cu();

        unsafe {
            let a_raw = a.device_ptr_mut(&self.stream).0 as *mut sys::cuDoubleComplex;
            let info_raw = info.device_ptr_mut(&self.stream).0 as *mut i32;

            // Query workspace size
            let mut lwork: i32 = 0;
            sys::cusolverDnZpotrf_bufferSize(handle, uplo, n, a_raw, n, &mut lwork as *mut _)
                .result()?;

            // Allocate workspace
            let workspace = self.stream.alloc_zeros::<CudaComplex>(lwork as usize)?;
            let work_raw = workspace.device_ptr(&self.stream).0 as *mut sys::cuDoubleComplex;

            // Factorise
            sys::cusolverDnZpotrf(handle, uplo, n, a_raw, n, work_raw, lwork, info_raw)
                .result()?;

            Ok(())
        }
    }

    /// Solve `A·X = B` after `A` has been Cholesky-factored by `zpotrf`.
    ///
    /// - `uplo` must match the `uplo` used in `zpotrf`.
    /// - `a` is the triangular factor from `zpotrf` (read-only).
    /// - `b` is the RHS on input, overwritten with the solution X on output.
    /// - `info[0]` must be 0 for success.
    pub fn zpotrs(
        &self,
        uplo: cublasFillMode_t,
        n: i32,
        nrhs: i32,
        a: &CudaSlice<CudaComplex>,
        b: &mut CudaSlice<CudaComplex>,
        info: &mut CudaSlice<i32>,
    ) -> Result<(), SolverError> {
        let handle = self.inner.cu();

        unsafe {
            let a_raw = a.device_ptr(&self.stream).0 as *const sys::cuDoubleComplex;
            let b_raw = b.device_ptr_mut(&self.stream).0 as *mut sys::cuDoubleComplex;
            let info_raw = info.device_ptr_mut(&self.stream).0 as *mut i32;

            sys::cusolverDnZpotrs(handle, uplo, n, nrhs, a_raw, n, b_raw, n, info_raw)
                .result()?;

            Ok(())
        }
    }

    /// LU factorisation `P·A = L·U` in-place with partial pivoting.
    ///
    /// - `m`, `n`: matrix dimensions (square for our use, but cuSOLVER supports
    ///   rectangular via `cusolverDnZgetrf`).
    /// - `a` is overwritten: lower part (unit L) and upper part (U).
    /// - `ipiv` receives pivot indices (1-based, Fortran convention).
    /// - `info[0]` must be 0 for success; `info[0] = k` means U[k,k] = 0.
    pub fn zgetrf(
        &self,
        m: i32,
        n: i32,
        a: &mut CudaSlice<CudaComplex>,
        ipiv: &mut CudaSlice<i32>,
        info: &mut CudaSlice<i32>,
    ) -> Result<(), SolverError> {
        let handle = self.inner.cu();

        unsafe {
            let a_raw = a.device_ptr_mut(&self.stream).0 as *mut sys::cuDoubleComplex;
            let ipiv_raw = ipiv.device_ptr_mut(&self.stream).0 as *mut i32;
            let info_raw = info.device_ptr_mut(&self.stream).0 as *mut i32;

            // Query workspace size
            let mut lwork: i32 = 0;
            sys::cusolverDnZgetrf_bufferSize(handle, m, n, a_raw, m, &mut lwork as *mut _)
                .result()?;

            // Allocate workspace
            let workspace = self.stream.alloc_zeros::<CudaComplex>(lwork as usize)?;
            let work_raw = workspace.device_ptr(&self.stream).0 as *mut sys::cuDoubleComplex;

            // Factorise
            sys::cusolverDnZgetrf(handle, m, n, a_raw, m, work_raw, ipiv_raw, info_raw)
                .result()?;

            Ok(())
        }
    }

    /// Solve `A·X = B` after `A` has been LU-factored by `zgetrf`.
    ///
    /// - `trans`: `CUBLAS_OP_N` for `A·X = B`, `CUBLAS_OP_T`/`CUBLAS_OP_C` for transposed.
    /// - `a` is the LU factor from `zgetrf` (read-only).
    /// - `ipiv` is the pivot array from `zgetrf`.
    /// - `b` is the RHS on input, overwritten with the solution X on output.
    #[allow(clippy::too_many_arguments)]
    pub fn zgetrs(
        &self,
        trans: cublasOperation_t,
        n: i32,
        nrhs: i32,
        a: &CudaSlice<CudaComplex>,
        ipiv: &CudaSlice<i32>,
        b: &mut CudaSlice<CudaComplex>,
        info: &mut CudaSlice<i32>,
    ) -> Result<(), SolverError> {
        let handle = self.inner.cu();

        unsafe {
            let a_raw = a.device_ptr(&self.stream).0 as *const sys::cuDoubleComplex;
            let ipiv_raw = ipiv.device_ptr(&self.stream).0 as *const i32;
            let b_raw = b.device_ptr_mut(&self.stream).0 as *mut sys::cuDoubleComplex;
            let info_raw = info.device_ptr_mut(&self.stream).0 as *mut i32;

            sys::cusolverDnZgetrs(handle, trans, n, nrhs, a_raw, n, ipiv_raw, b_raw, n, info_raw)
                .result()?;

            Ok(())
        }
    }
}

impl std::fmt::Debug for SolverHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverHandle").finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    #[test]
    #[cfg(feature = "chebyshev")]
    fn test_zhegvd_4x4_diagonal() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let solver = SolverHandle::new(stream.clone()).unwrap();

        let n = 4i32;

        // A = diag(1,2,3,4), B = I → eigenvalues 1,2,3,4
        let mut h_a = vec![CudaComplex { x: 0.0, y: 0.0 }; (n * n) as usize];
        for i in 0..n {
            h_a[(i * n + i) as usize] = CudaComplex { x: (i + 1) as f64, y: 0.0 };
        }

        let mut h_b = vec![CudaComplex { x: 0.0, y: 0.0 }; (n * n) as usize];
        for i in 0..n {
            h_b[(i * n + i) as usize] = CudaComplex { x: 1.0, y: 0.0 };
        }

        let mut d_a = stream.clone_htod(&h_a).unwrap();
        let mut d_b = stream.clone_htod(&h_b).unwrap();
        let mut d_eigenvalues = stream.alloc_zeros::<f64>(n as usize).unwrap();
        let mut d_info = stream.alloc_zeros::<i32>(1).unwrap();

        solver
            .zhegvd(
                cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
                cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                n,
                &mut d_a,
                &mut d_b,
                &mut d_eigenvalues,
                &mut d_info,
            )
            .unwrap();

        let eigenvalues: Vec<f64> = stream.clone_dtoh(&d_eigenvalues).unwrap();
        let info: Vec<i32> = stream.clone_dtoh(&d_info).unwrap();

        assert_eq!(info[0], 0, "cuSOLVER info != 0");

        let expected = [1.0, 2.0, 3.0, 4.0];
        for i in 0..n as usize {
            assert!(
                (eigenvalues[i] - expected[i]).abs() < 1e-8,
                "Eigenvalue {i}: got {}, expected {}",
                eigenvalues[i],
                expected[i],
            );
        }
    }

    #[test]
    fn zpotrs_round_trip() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let solver = SolverHandle::new(stream.clone()).unwrap();

        let n = 4i32;

        // Build HPD matrix A = M^H·M where M is a simple 4×4 lower-triangular matrix
        let m: Vec<f64> = vec![
            2.0, 0.0, 0.0, 0.0,
            0.5, 1.5, 0.0, 0.0,
            0.3, 0.4, 1.2, 0.0,
            0.1, 0.2, 0.3, 1.1,
        ];
        // A = M·M^T (symmetric positive-definite)
        let mut h_a = vec![CudaComplex { x: 0.0, y: 0.0 }; (n * n) as usize];
        for i in 0..n as usize {
            for j in 0..n as usize {
                let mut sum = 0.0;
                for k in 0..n as usize {
                    sum += m[i * n as usize + k] * m[j * n as usize + k];
                }
                h_a[i * n as usize + j] = CudaComplex { x: sum, y: 0.0 };
            }
        }

        // x_exact = [1, 2, 3, 4]^T, b = A·x_exact
        let x_exact: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut h_b = vec![CudaComplex { x: 0.0, y: 0.0 }; n as usize];
        for i in 0..n as usize {
            let mut sum = 0.0;
            for j in 0..n as usize {
                sum += h_a[i * n as usize + j].x * x_exact[j];
            }
            h_b[i] = CudaComplex { x: sum, y: 0.0 };
        }

        // Upload A and B
        let mut d_a = stream.clone_htod(&h_a).unwrap();
        let mut d_b = stream.clone_htod(&h_b).unwrap();
        let mut d_info = stream.alloc_zeros::<i32>(1).unwrap();

        // Factor A = L·L^H
        solver
            .zpotrf(cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, n, &mut d_a, &mut d_info)
            .unwrap();
        let info: Vec<i32> = stream.clone_dtoh(&d_info).unwrap();
        assert_eq!(info[0], 0, "zpotrf info != 0");

        // Solve A·x = b using the Cholesky factor
        let mut d_info2 = stream.alloc_zeros::<i32>(1).unwrap();
        solver
            .zpotrs(
                cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                n, 1,
                &d_a, &mut d_b, &mut d_info2,
            )
            .unwrap();
        let info2: Vec<i32> = stream.clone_dtoh(&d_info2).unwrap();
        assert_eq!(info2[0], 0, "zpotrs info != 0");

        let x_solved: Vec<CudaComplex> = stream.clone_dtoh(&d_b).unwrap();
        let max_err = x_solved.iter().zip(x_exact.iter())
            .map(|(s, e)| (s.x - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "zpotrs round-trip max error = {:.2e} >= 1e-12",
            max_err,
        );
    }

    #[test]
    fn zpotrs_multi_rhs() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let solver = SolverHandle::new(stream.clone()).unwrap();

        let n = 16i32;
        let nrhs = 8i32;

        // Build HPD matrix A = M·M^T via a random lower-triangular M
        // (guarantees positive definiteness)
        let mut m = vec![0.0_f64; (n * n) as usize];
        for i in 0..n as usize {
            for j in 0..n as usize {
                m[i * n as usize + j] = if i >= j {
                    (i * n as usize + j) as f64 * 0.1 + 1.0
                } else {
                    0.0
                };
            }
        }
        let mut h_a = vec![CudaComplex { x: 0.0, y: 0.0 }; (n * n) as usize];
        for i in 0..n as usize {
            for j in 0..n as usize {
                let mut sum = 0.0;
                for k in 0..n as usize {
                    sum += m[i * n as usize + k] * m[j * n as usize + k];
                }
                h_a[i * n as usize + j] = CudaComplex { x: sum, y: 0.0 };
            }
        }

        // x_exact: n×nrhs matrix, each column = [1,2,...,n] * col_factor
        let mut h_x_exact = vec![0.0_f64; (n * nrhs) as usize];
        for col in 0..nrhs as usize {
            let col_factor = (col + 1) as f64;
            for row in 0..n as usize {
                h_x_exact[row * nrhs as usize + col] = (row + 1) as f64 * col_factor;
            }
        }
        // b = A · x_exact (col-major: n×nrhs)
        let mut h_b = vec![CudaComplex { x: 0.0, y: 0.0 }; (n * nrhs) as usize];
        for col in 0..nrhs as usize {
            for i in 0..n as usize {
                let mut sum = 0.0;
                for j in 0..n as usize {
                    sum += h_a[i * n as usize + j].x * h_x_exact[j * nrhs as usize + col];
                }
                h_b[i * nrhs as usize + col] = CudaComplex { x: sum, y: 0.0 };
            }
        }

        let mut d_a = stream.clone_htod(&h_a).unwrap();
        let mut d_b = stream.clone_htod(&h_b).unwrap();
        let mut d_info = stream.alloc_zeros::<i32>(1).unwrap();

        // Factor A = L·L^H
        solver
            .zpotrf(cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, n, &mut d_a, &mut d_info)
            .unwrap();
        let info: Vec<i32> = stream.clone_dtoh(&d_info).unwrap();
        assert_eq!(info[0], 0, "zpotrf info != 0");

        // Solve A·X = B
        let mut d_info2 = stream.alloc_zeros::<i32>(1).unwrap();
        solver
            .zpotrs(
                cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                n, nrhs,
                &d_a, &mut d_b, &mut d_info2,
            )
            .unwrap();
        let info2: Vec<i32> = stream.clone_dtoh(&d_info2).unwrap();
        assert_eq!(info2[0], 0, "zpotrs info != 0");

        let x_solved: Vec<CudaComplex> = stream.clone_dtoh(&d_b).unwrap();
        let max_err = x_solved.iter().zip(h_x_exact.iter())
            .map(|(s, e)| (s.x - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "zpotrs multi-rhs max error = {:.2e} >= 1e-12",
            max_err,
        );
    }

    #[test]
    fn zgetrs_round_trip() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let solver = SolverHandle::new(stream.clone()).unwrap();

        let n = 4i32;

        // A simple full-rank 4×4 matrix
        let h_a: Vec<CudaComplex> = vec![
            CudaComplex { x: 2.0, y: 0.0 }, CudaComplex { x: 1.0, y: 0.0 }, CudaComplex { x: 0.0, y: 0.0 }, CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 1.0, y: 0.0 }, CudaComplex { x: 3.0, y: 0.0 }, CudaComplex { x: 1.0, y: 0.0 }, CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 }, CudaComplex { x: 1.0, y: 0.0 }, CudaComplex { x: 4.0, y: 0.0 }, CudaComplex { x: 2.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 }, CudaComplex { x: 0.0, y: 0.0 }, CudaComplex { x: 2.0, y: 0.0 }, CudaComplex { x: 5.0, y: 0.0 },
        ];

        // x_exact = [1, 2, 3, 4]^T, b = A·x_exact
        let x_exact: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut h_b = vec![CudaComplex { x: 0.0, y: 0.0 }; n as usize];
        for i in 0..n as usize {
            let mut sum = 0.0;
            for j in 0..n as usize {
                sum += h_a[i * n as usize + j].x * x_exact[j];
            }
            h_b[i] = CudaComplex { x: sum, y: 0.0 };
        }

        let mut d_a = stream.clone_htod(&h_a).unwrap();
        let mut d_b = stream.clone_htod(&h_b).unwrap();
        let mut d_ipiv = stream.alloc_zeros::<i32>(n as usize).unwrap();
        let mut d_info = stream.alloc_zeros::<i32>(1).unwrap();

        // Factor A = P·L·U
        solver
            .zgetrf(n, n, &mut d_a, &mut d_ipiv, &mut d_info)
            .unwrap();
        let info: Vec<i32> = stream.clone_dtoh(&d_info).unwrap();
        assert_eq!(info[0], 0, "zgetrf info != 0");

        // Solve A·x = b using the LU factor
        let mut d_info2 = stream.alloc_zeros::<i32>(1).unwrap();
        solver
            .zgetrs(
                cublasOperation_t::CUBLAS_OP_N,
                n, 1,
                &d_a, &d_ipiv, &mut d_b, &mut d_info2,
            )
            .unwrap();
        let info2: Vec<i32> = stream.clone_dtoh(&d_info2).unwrap();
        assert_eq!(info2[0], 0, "zgetrs info != 0");

        let x_solved: Vec<CudaComplex> = stream.clone_dtoh(&d_b).unwrap();
        let max_err = x_solved.iter().zip(x_exact.iter())
            .map(|(s, e)| (s.x - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "zgetrs round-trip max error = {:.2e} >= 1e-12",
            max_err,
        );
    }

    #[test]
    fn zgetrs_multi_rhs() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let solver = SolverHandle::new(stream.clone()).unwrap();

        let n = 16i32;
        let nrhs = 8i32;

        // Full-rank real matrix: A[i,j] = 1/(i+j+1) + δ_ij·n  (diagonally dominant)
        let mut h_a = vec![CudaComplex { x: 0.0, y: 0.0 }; (n * n) as usize];
        for i in 0..n as usize {
            for j in 0..n as usize {
                let val = 1.0 / ((i + j + 1) as f64);
                h_a[i * n as usize + j] = CudaComplex {
                    x: val + if i == j { n as f64 } else { 0.0 },
                    y: 0.0,
                };
            }
        }

        // x_exact: n×nrhs, each column = varying amplitudes
        let mut h_x_exact = vec![0.0_f64; (n * nrhs) as usize];
        for col in 0..nrhs as usize {
            let col_factor = (col + 1) as f64 * 0.5;
            for row in 0..n as usize {
                h_x_exact[row * nrhs as usize + col] = (row + 1) as f64 * col_factor;
            }
        }
        // b = A · x_exact
        let mut h_b = vec![CudaComplex { x: 0.0, y: 0.0 }; (n * nrhs) as usize];
        for col in 0..nrhs as usize {
            for i in 0..n as usize {
                let mut sum = 0.0;
                for j in 0..n as usize {
                    sum += h_a[i * n as usize + j].x * h_x_exact[j * nrhs as usize + col];
                }
                h_b[i * nrhs as usize + col] = CudaComplex { x: sum, y: 0.0 };
            }
        }

        let mut d_a = stream.clone_htod(&h_a).unwrap();
        let mut d_b = stream.clone_htod(&h_b).unwrap();
        let mut d_ipiv = stream.alloc_zeros::<i32>(n as usize).unwrap();
        let mut d_info = stream.alloc_zeros::<i32>(1).unwrap();

        solver
            .zgetrf(n, n, &mut d_a, &mut d_ipiv, &mut d_info)
            .unwrap();
        let info: Vec<i32> = stream.clone_dtoh(&d_info).unwrap();
        assert_eq!(info[0], 0, "zgetrf info != 0");

        let mut d_info2 = stream.alloc_zeros::<i32>(1).unwrap();
        solver
            .zgetrs(
                cublasOperation_t::CUBLAS_OP_N,
                n, nrhs,
                &d_a, &d_ipiv, &mut d_b, &mut d_info2,
            )
            .unwrap();
        let info2: Vec<i32> = stream.clone_dtoh(&d_info2).unwrap();
        assert_eq!(info2[0], 0, "zgetrs info != 0");

        let x_solved: Vec<CudaComplex> = stream.clone_dtoh(&d_b).unwrap();
        let max_err = x_solved.iter().zip(h_x_exact.iter())
            .map(|(s, e)| (s.x - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "zgetrs multi-rhs max error = {:.2e} >= 1e-12",
            max_err,
        );
    }

    /// Synthesis test: cuSOLVER ZPOTRF vs CPU Cholesky on near-singular matrices.
    ///
    /// Generates complex Hermitian positive-definite matrices with controlled
    /// condition numbers and compares whether cuSOLVER ZPOTRF and an independent
    /// CPU Cholesky both succeed or fail.  This tests the claim that cuSOLVER's
    /// ZPOTRF is less robust than LAPACK's for near-singular matrices.
    ///
    /// The test builds A = Q · diag(λ_i) · Q^H where eigenvalues λ_i are:
    ///   - First n_big eigenvalues: 1.0
    ///   - Remaining eigenvalues: ε (controlled → 1/condition_number)
    /// The condition number is κ = 1/ε.
    #[test]
    fn zpotrf_near_singular_cpu_comparison() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();
        let solver = SolverHandle::new(stream.clone()).expect("SolverHandle");

        // Matrix sizes to test — 52, 78, 130, 156 (typical superspace sizes)
        let sizes = [52, 78, 130, 156];
        // Condition numbers: κ = 1/ε
        let conds = [1e8_f64, 1e10, 1e12, 1e14, 1e15, 1e16];
        // How many "big" eigenvalues (the rest are ε)
        let n_big_ratios = [0.2_f64, 0.5, 0.8]; // 20%, 50%, 80% big eigenvalues

        let mut summary = Vec::new();

        for &n in &sizes {
            for &frac in &n_big_ratios {
                let n_big = (n as f64 * frac).round() as usize;
                for &cond in &conds {
                    let eps = 1.0 / cond;
                    // Build eigenvalues
                    let mut evals = vec![eps; n];
                    for i in 0..n_big.min(n) {
                        evals[i] = 1.0;
                    }

                    // Build random orthonormal basis Q via QR on random matrix
                    let mut rng_q: Vec<f64> = (0..(n * n * 2))
                        .map(|i| {
                            let x = (i as f64 * 1.23456789).sin() * 1000.0;
                            x - x.floor()
                        })
                        .collect();
                    // Simple Gram-Schmidt to get Q
                    let mut q_re = vec![0.0_f64; n * n];
                    let mut q_im = vec![0.0_f64; n * n];
                    for col in 0..n {
                        // Start with random vector
                        for row in 0..n {
                            let idx = (row * n + col) * 2;
                            q_re[row * n + col] = rng_q[idx];
                            q_im[row * n + col] = rng_q[idx + 1];
                        }
                        // Orthogonalize against previous columns
                        for prev in 0..col {
                            let mut dot_re = 0.0_f64;
                            let mut dot_im = 0.0_f64;
                            for row in 0..n {
                                let a_re = q_re[row * n + prev];
                                let a_im = q_im[row * n + prev];
                                let b_re = q_re[row * n + col];
                                let b_im = q_im[row * n + col];
                                // conj(a) * b = (a_re - i*a_im) * (b_re + i*b_im)
                                dot_re += a_re * b_re + a_im * b_im;
                                dot_im += a_re * b_im - a_im * b_re;
                            }
                            for row in 0..n {
                                q_re[row * n + col] -= dot_re * q_re[row * n + prev] - dot_im * q_im[row * n + prev];
                                q_im[row * n + col] -= dot_re * q_im[row * n + prev] + dot_im * q_re[row * n + prev];
                            }
                        }
                        // Normalize
                        let mut norm = 0.0_f64;
                        for row in 0..n {
                            norm += q_re[row * n + col].powi(2) + q_im[row * n + col].powi(2);
                        }
                        let inv_norm = 1.0 / norm.sqrt();
                        for row in 0..n {
                            q_re[row * n + col] *= inv_norm;
                            q_im[row * n + col] *= inv_norm;
                        }
                    }

                    // Build A = Q · diag(λ) · Q^H, stored column-major (Fortran order)
                    let mut a_cpu = vec![num_complex::Complex64::new(0.0, 0.0); n * n];
                    for i in 0..n {
                        for j in 0..n {
                            // A[i,j] = Σ_k Q[i,k] * λ_k * conj(Q[j,k])
                            let mut sum_re = 0.0_f64;
                            let mut sum_im = 0.0_f64;
                            for k in 0..n {
                                let q_ik_re = q_re[i * n + k];
                                let q_ik_im = q_im[i * n + k];
                                let q_jk_re = q_re[j * n + k];
                                let q_jk_im = q_im[j * n + k];
                                // q_ik * λ_k * conj(q_jk) = λ_k * (q_ik_re + i*q_ik_im) * (q_jk_re - i*q_jk_im)
                                let l = evals[k];
                                sum_re += l * (q_ik_re * q_jk_re + q_ik_im * q_jk_im);
                                sum_im += l * (q_ik_im * q_jk_re - q_ik_re * q_jk_im);
                            }
                            a_cpu[i + j * n] = num_complex::Complex64::new(sum_re, sum_im);
                        }
                    }

                    // --- CPU Cholesky (independent reference) ---
                    let mut a_cpu_chol = a_cpu.clone();
                    let cpu_ok = cpu_cholesky_upper(&mut a_cpu_chol, n);

                    // --- GPU cuSOLVER ZPOTRF ---
                    let mut h_a_gpu: Vec<CudaComplex> = a_cpu.iter()
                        .map(|c| CudaComplex { x: c.re, y: c.im })
                        .collect();
                    let mut d_a = stream.alloc_zeros::<CudaComplex>(n * n).unwrap();
                    stream.memcpy_htod(&h_a_gpu, &mut d_a).unwrap();
                    let mut d_info = stream.alloc_zeros::<i32>(1).unwrap();

                    let result = solver.zpotrf(
                        cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                        n as i32,
                        &mut d_a,
                        &mut d_info,
                    );
                    let gpu_ok = if result.is_ok() {
                        let info: Vec<i32> = stream.clone_dtoh(&d_info).unwrap();
                        info[0] == 0
                    } else {
                        false
                    };

                    summary.push(format!(
                        "n={:3} n_big={:3}/{:3} κ={:.0e}  CPU={} GPU={}",
                        n, n_big, n, cond,
                        if cpu_ok { "OK " } else { "FAIL" },
                        if gpu_ok { "OK " } else { "FAIL" },
                    ));

                    // Assert: for well-conditioned matrices (κ ≤ 1e10), GPU must match CPU
                    if cond <= 1e10 {
                        assert_eq!(
                            gpu_ok, cpu_ok,
                            "GPU/CPU mismatch at n={n} n_big={n_big} κ={cond:.0e}"
                        );
                    }
                }
            }
        }

        eprintln!("=== ZPOTRF near-singular synthesis test ===");
        for line in &summary {
            eprintln!("  {line}");
        }

        // Count GPU failures for κ ≥ 1e14
        let gpu_fail_high = summary.iter()
            .filter(|s| s.contains("GPU=FAIL") && (s.contains("κ=1e14") || s.contains("κ=1e15") || s.contains("κ=1e16")))
            .count();
        let cpu_fail_high = summary.iter()
            .filter(|s| s.contains("CPU=FAIL") && (s.contains("κ=1e14") || s.contains("κ=1e15") || s.contains("κ=1e16")))
            .count();
        eprintln!("  GPU failures at κ≥1e14: {gpu_fail_high}");
        eprintln!("  CPU failures at κ≥1e14: {cpu_fail_high}");

        // If GPU fails more than CPU at high κ, cuSOLVER is less robust
        if gpu_fail_high > cpu_fail_high {
            eprintln!("  RESULT: cuSOLVER ZPOTRF is LESS ROBUST than CPU Cholesky at high κ");
        } else if gpu_fail_high < cpu_fail_high {
            eprintln!("  RESULT: cuSOLVER ZPOTRF is MORE ROBUST than CPU Cholesky at high κ");
        } else {
            eprintln!("  RESULT: cuSOLVER ZPOTRF and CPU Cholesky have SIMILAR robustness");
        }
    }

    /// Independent CPU-side Cholesky factorization (upper triangle).
    /// Returns false if factorization fails (diagonal ≤ 0).
    fn cpu_cholesky_upper(a: &mut [num_complex::Complex64], n: usize) -> bool {
        for k in 0..n {
            // U[k,k] = sqrt(A[k,k] - Σ_{p<k} |U[p,k]|²)
            let mut sum_sq = 0.0_f64;
            for p in 0..k {
                let upk = a[p + k * n];
                sum_sq += upk.re.powi(2) + upk.im.powi(2);
            }
            let akk = a[k + k * n].re - sum_sq;
            if akk <= 0.0 {
                return false;
            }
            let ukk = akk.sqrt();
            a[k + k * n] = num_complex::Complex64::new(ukk, 0.0);

            // U[k,j] = (A[k,j] - Σ_{p<k} conj(U[p,k]) * U[p,j]) / U[k,k]
            for j in (k + 1)..n {
                let mut sum = num_complex::Complex64::new(0.0, 0.0);
                for p in 0..k {
                    let upk = a[p + k * n];
                    let upj = a[p + j * n];
                    sum += upk.conj() * upj;
                }
                let akj = a[k + j * n] - sum;
                a[k + j * n] = akj / ukk;
            }
        }
        true
    }

    /// Test cuSOLVER ZHEEVD accuracy on near-singular Hermitian matrices.
    ///
    /// Simulates a Davidson superspace H_sub where duplicate search columns
    /// create near-zero eigenvalues: H = Q · diag(λ) · Q^H.
    /// Some λ_i = ε ≪ 1 simulate duplicate columns → rank deficiency.
    /// Measures eigenvalue residual ‖H·v_j - λ_j·v_j‖ for the smallest
    /// eigenvalues and checks that accuracy does not catastrophically degrade.
    #[test]
    fn zheevd_near_singular_accuracy() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();
        let solver = SolverHandle::new(stream.clone()).expect("SolverHandle");

        // Simulate a superspace with near-duplicate search columns.
        // Small ε = 10⁻⁴, 10⁻⁸, 10⁻¹² (condition number κ = 1/ε).
        // Larger n_big simulates more "real" columns vs duplicate columns.
        let cases = [
            (52,  47, 1e-4_f64,   5), // n=52, 5 near-zero eigenvalues at 1e-4
            (52,  47, 1e-8_f64,   5),
            (52,  47, 1e-12_f64,  5),
            (78,  68, 1e-8_f64,  10),
            (78,  68, 1e-12_f64, 10),
            (130, 110, 1e-8_f64, 20),
            (130, 110, 1e-12_f64,20),
            (156, 126, 1e-8_f64, 30),
            (156, 126, 1e-12_f64,30),
        ];

        let mut max_residual = 0.0_f64;
        for &(n, n_big, eps, n_small) in &cases {
            let n_small = n_small.min(n - n_big);
            // Generate eigenvalues: n_big at 1.0, n_small at eps
            let mut evals: Vec<f64> = vec![1.0; n_big];
            evals.extend(std::iter::repeat(eps).take(n_small));
            // Fill remaining with interpolated values
            while evals.len() < n { evals.push(eps * 10.0); }
            evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Build random unitary Q via QR on random complex matrix
            let mut rng: Vec<f64> = (0..(n*n*2)).map(|i| {
                let x = (i as f64 * 0.987654321).sin() * 1000.0;
                x - x.floor()
            }).collect();
            let mut q_cpu: Vec<CudaComplex> = Vec::with_capacity(n*n);
            for i in 0..n {
                for j in 0..n {
                    q_cpu.push(CudaComplex {
                        x: rng[2*(i*n+j)] - 0.5,
                        y: rng[2*(i*n+j)+1] - 0.5,
                    });
                }
            }
            // Gram-Schmidt orthogonalize columns of Q
            for j in 0..n {
                // Normalize column j
                let mut norm2 = 0.0_f64;
                for i in 0..n { let c = q_cpu[i + j*n]; norm2 += c.x*c.x + c.y*c.y; }
                let inv_norm = 1.0 / norm2.sqrt();
                for i in 0..n { q_cpu[i + j*n].x *= inv_norm; q_cpu[i + j*n].y *= inv_norm; }
                // Project out previous columns
                for k in 0..j {
                    let mut dot_re = 0.0_f64; let mut dot_im = 0.0_f64;
                    for i in 0..n {
                        let a = q_cpu[i + j*n]; let b = q_cpu[i + k*n];
                        dot_re += a.x * b.x + a.y * b.y;
                        dot_im += a.x * b.y - a.y * b.x;
                    }
                    for i in 0..n {
                        let b = q_cpu[i + k*n];
                        q_cpu[i + j*n].x -= dot_re * b.x - dot_im * b.y;
                        q_cpu[i + j*n].y -= dot_re * b.y + dot_im * b.x;
                    }
                    // Re-normalize
                    norm2 = 0.0;
                    for i in 0..n { let c = q_cpu[i + j*n]; norm2 += c.x*c.x + c.y*c.y; }
                    let inv_n = 1.0 / norm2.sqrt();
                    for i in 0..n { q_cpu[i + j*n].x *= inv_n; q_cpu[i + j*n].y *= inv_n; }
                }
            }

            // Build H = Q · diag(λ) · Q^H (column-major)
            let mut h_cpu = vec![CudaComplex{x:0.0,y:0.0}; n*n];
            for i in 0..n {
                for j in 0..n {
                    let mut sum_re = 0.0; let mut sum_im = 0.0;
                    for k in 0..n {
                        let q_ik = q_cpu[i + k*n];
                        let q_jk = q_cpu[j + k*n];
                        let lam = evals[k];
                        sum_re += lam * (q_ik.x * q_jk.x + q_ik.y * q_jk.y);
                        sum_im += lam * (q_ik.x * q_jk.y - q_ik.y * q_jk.x);
                    }
                    h_cpu[i + j*n] = CudaComplex{x: sum_re, y: sum_im};
                }
            }

            // Upload to GPU
            let mut h_gpu = stream.alloc_zeros::<CudaComplex>(n*n).expect("alloc H");
            stream.memcpy_htod(&h_cpu, &mut h_gpu).expect("H2D H");
            let mut eig_gpu = stream.alloc_zeros::<f64>(n).expect("alloc eig");
            let mut info_gpu = stream.alloc_zeros::<i32>(1).expect("alloc info");

            solver.zheevd(
                cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
                cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                n as i32, &mut h_gpu, &mut eig_gpu, &mut info_gpu,
            ).expect("ZHEEVD");

            let info: Vec<i32> = stream.clone_dtoh(&info_gpu).expect("D2H info");
            assert_eq!(info[0], 0, "ZHEEVD info={} at n={} n_small={} eps={:e}", info[0], n, n_small, eps);

            let ev_gpu: Vec<f64> = stream.clone_dtoh(&eig_gpu).expect("D2H eig");
            let v_gpu: Vec<CudaComplex> = stream.clone_dtoh(&h_gpu).expect("D2H V"); // overwritten with eigenvectors

            // Check residual ‖H·v_j - λ_j·v_j‖ for small eigenvalues
            for j in 0..n_small {
                let lam = ev_gpu[j];
                // Compute H·v_j on CPU
                let mut hv_re = vec![0.0_f64; n]; let mut hv_im = vec![0.0_f64; n];
                for i in 0..n {
                    for k in 0..n {
                        let h_ik = h_cpu[i + k*n];
                        let v_kj = v_gpu[k + j*n];
                        hv_re[i] += h_ik.x * v_kj.x - h_ik.y * v_kj.y;
                        hv_im[i] += h_ik.x * v_kj.y + h_ik.y * v_kj.x;
                    }
                }
                // Residual = H·v - λ·v
                let mut res_norm2 = 0.0_f64;
                for i in 0..n {
                    let v_ij = v_gpu[i + j*n];
                    let dr = hv_re[i] - lam * v_ij.x;
                    let di = hv_im[i] - lam * v_ij.y;
                    res_norm2 += dr*dr + di*di;
                }
                let res_norm = res_norm2.sqrt();
                if res_norm > max_residual { max_residual = res_norm; }
                let status = if res_norm > 1e-6 { "HIGH" } else if res_norm > 1e-9 { "WARN" } else { "ok" };
                eprintln!("  ZHEEVD res n={} n_small={} eps={:e} j={} λ={:.6e} ‖Hv-λv‖={:.3e} {}",
                    n, n_small, eps, j, lam, res_norm, status);
            }

            stream.synchronize().expect("sync");
        }
        eprintln!("  max residual across all cases: {:.3e}", max_residual);
        assert!(max_residual < 1e-3, "ZHEEVD residual {:.3e} exceeds 1e-3 for near-singular matrix", max_residual);
    }
}
