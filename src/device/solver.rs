// ---------------------------------------------------------------------------
// cuSOLVER wrapper — dense eigenvalue solver (ZHEGVD)
// ---------------------------------------------------------------------------

use std::sync::Arc;

use cudarc::cusolver::result::CusolverError;
use cudarc::cusolver::safe::DnHandle;
use cudarc::cusolver::sys::{
    self as sys, cublasFillMode_t, cusolverEigMode_t, cusolverEigType_t,
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

    /// Solve `A·X = λ·B·X` via ZHEGVD.
    ///
    /// - `a`, `b` are overwritten. On return `a` contains eigenvectors.
    /// - `eigenvalues[0..n-1]` filled on return.
    /// - `info[0]` must be 0 for success.
    #[allow(clippy::too_many_arguments)]
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
}
