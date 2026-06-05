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
}
