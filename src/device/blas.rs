// ---------------------------------------------------------------------------
// cuBLAS wrapper — gemm/gemv via config-struct API, axpy/dot via sys FFI
// ---------------------------------------------------------------------------

use std::sync::Arc;

use cudarc::cublas::safe::{GemmConfig, GemvConfig};
use cudarc::cublas::{CudaBlas, Gemm, Gemv};
use cudarc::cublas::result::CublasError;
use cudarc::cublas::sys::{self as blas_sys, cublasHandle_t, cublasOperation_t};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use super::CudaComplex;

/// Convenience constants for matrix layout.
pub mod op {
    use cudarc::cublas::sys::cublasOperation_t;
    pub const N: cublasOperation_t = cublasOperation_t::CUBLAS_OP_N;
    pub const T: cublasOperation_t = cublasOperation_t::CUBLAS_OP_T;
    pub const C: cublasOperation_t = cublasOperation_t::CUBLAS_OP_C;
}

/// Config for complex ZGEMM — same fields as GemmConfig<CudaComplex>.
pub struct ZgemmConfig {
    pub transa: cublasOperation_t,
    pub transb: cublasOperation_t,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub alpha: CudaComplex,
    pub lda: i32,
    pub ldb: i32,
    pub beta: CudaComplex,
    pub ldc: i32,
}

/// Wrapper around a cuBLAS handle for our SCF operations.
pub struct BlasHandle {
    inner: CudaBlas,
    stream: Arc<CudaStream>,
}

impl BlasHandle {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CublasError> {
        let inner = CudaBlas::new(stream.clone())?;
        Ok(Self { inner, stream })
    }

    pub fn raw_handle(&self) -> cublasHandle_t {
        *self.inner.handle()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    // ── GEMM (f64) ──

    /// C = α·A·B + β·C  (f64).
    ///
    /// # Safety
    /// A, B, C must point to valid device memory of sufficient dimensions.
    pub unsafe fn gemm_f64(
        &self,
        cfg: GemmConfig<f64>,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        c: &mut CudaSlice<f64>,
    ) -> Result<(), CublasError> {
        unsafe { self.inner.gemm(cfg, a, b, c) }
    }

    // ── GEMM (complex) ──

    /// C = α·A·B + β·C  (complex, ZGEMM).
    ///
    /// # Safety
    /// A, B, C must point to valid device memory of sufficient dimensions.
    pub unsafe fn gemm_c64(
        &self,
        cfg: ZgemmConfig,
        a: &CudaSlice<CudaComplex>,
        b: &CudaSlice<CudaComplex>,
        c: &mut CudaSlice<CudaComplex>,
    ) -> Result<(), CublasError> {
        unsafe {
            let (a_ptr, _) = a.device_ptr(&self.stream);
            let (b_ptr, _) = b.device_ptr(&self.stream);
            let (c_ptr, _) = c.device_ptr_mut(&self.stream);
            blas_sys::cublasZgemm_v2(
                self.raw_handle(),
                cfg.transa, cfg.transb,
                cfg.m, cfg.n, cfg.k,
                &cfg.alpha as *const _ as *const _,
                a_ptr as *const _, cfg.lda,
                b_ptr as *const _, cfg.ldb,
                &cfg.beta as *const _ as *const _,
                c_ptr as *mut _, cfg.ldc,
            )
            .result()
        }
    }

    // ── GEMV (f64) ──

    /// y = α·A·x + β·y  (f64).
    ///
    /// # Safety
    /// A, x, y must point to valid device memory.
    pub unsafe fn gemv_f64(
        &self,
        cfg: GemvConfig<f64>,
        a: &CudaSlice<f64>,
        x: &CudaSlice<f64>,
        y: &mut CudaSlice<f64>,
    ) -> Result<(), CublasError> {
        unsafe { self.inner.gemv(cfg, a, x, y) }
    }

    /// y = α·A·x + β·y  (complex, ZGEMV).
    ///
    /// `trans` controls whether A is used as-is (N), transposed (T), or conjugate-transposed (C).
    /// `m` = rows of A, `n` = cols of A (before transpose).
    ///
    /// # Safety
    /// A, x, y must point to valid device memory of sufficient dimensions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemv_c64(
        &self,
        trans: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: CudaComplex,
        a: &CudaSlice<CudaComplex>,
        lda: i32,
        x: &CudaSlice<CudaComplex>,
        incx: i32,
        beta: CudaComplex,
        y: &mut CudaSlice<CudaComplex>,
        incy: i32,
    ) -> Result<(), CublasError> {
        unsafe {
            let (a_ptr, _) = a.device_ptr(&self.stream);
            let (x_ptr, _) = x.device_ptr(&self.stream);
            let (y_ptr, _) = y.device_ptr_mut(&self.stream);
            blas_sys::cublasZgemv_v2(
                self.raw_handle(),
                trans,
                m, n,
                &alpha as *const _ as *const _,
                a_ptr as *const _, lda,
                x_ptr as *const _, incx,
                &beta as *const _ as *const _,
                y_ptr as *mut _, incy,
            )
            .result()
        }
    }

    // ── AXPY (f64) ──

    /// y = α·x + y  (f64)
    pub fn axpy_f64(
        &self,
        n: i32,
        alpha: f64,
        x: &CudaSlice<f64>,
        incx: i32,
        y: &mut CudaSlice<f64>,
        incy: i32,
    ) -> Result<(), CublasError> {
        unsafe {
            let (x_ptr, _) = x.device_ptr(&self.stream);
            let (y_ptr, _) = y.device_ptr_mut(&self.stream);
            blas_sys::cublasDaxpy_v2(
                self.raw_handle(),
                n,
                &alpha as *const _,
                x_ptr as *const _, incx,
                y_ptr as *mut _, incy,
            )
            .result()
        }
    }

    /// y = α·x + y  (complex)
    pub fn axpy_c64(
        &self,
        n: i32,
        alpha: CudaComplex,
        x: &CudaSlice<CudaComplex>,
        incx: i32,
        y: &mut CudaSlice<CudaComplex>,
        incy: i32,
    ) -> Result<(), CublasError> {
        unsafe {
            let (x_ptr, _) = x.device_ptr(&self.stream);
            let (y_ptr, _) = y.device_ptr_mut(&self.stream);
            blas_sys::cublasZaxpy_v2(
                self.raw_handle(),
                n,
                &alpha as *const _ as *const _,
                x_ptr as *const _, incx,
                y_ptr as *mut _, incy,
            )
            .result()
        }
    }

    /// x = α·x  (complex, in-place)
    pub fn scal_c64(
        &self,
        n: i32,
        alpha: CudaComplex,
        x: &mut CudaSlice<CudaComplex>,
    ) -> Result<(), CublasError> {
        unsafe {
            let (x_ptr, _) = x.device_ptr_mut(&self.stream);
            let alpha_ptr = &alpha as *const _;
            blas_sys::cublasZscal_v2(
                self.raw_handle(),
                n,
                alpha_ptr as *const _,
                x_ptr as *mut _,
                1,
            )
            .result()
        }
    }

    // ── DOT (f64) ──

    /// inner = x·y  (f64)
    pub fn dot_f64(
        &self,
        n: i32,
        x: &CudaSlice<f64>,
        incx: i32,
        y: &CudaSlice<f64>,
        incy: i32,
    ) -> Result<f64, CublasError> {
        unsafe {
            let (x_ptr, _) = x.device_ptr(&self.stream);
            let (y_ptr, _) = y.device_ptr(&self.stream);
            let mut result: f64 = 0.0;
            blas_sys::cublasDdot_v2(
                self.raw_handle(), n,
                x_ptr as *const _, incx,
                y_ptr as *const _, incy,
                &mut result as *mut _,
            )
            .result()?;
            Ok(result)
        }
    }

    /// inner = conj(x)·y  (complex)
    pub fn dotc_c64(
        &self,
        n: i32,
        x: &CudaSlice<CudaComplex>,
        incx: i32,
        y: &CudaSlice<CudaComplex>,
        incy: i32,
    ) -> Result<CudaComplex, CublasError> {
        unsafe {
            let (x_ptr, _) = x.device_ptr(&self.stream);
            let (y_ptr, _) = y.device_ptr(&self.stream);
            let mut result = CudaComplex { x: 0.0, y: 0.0 };
            blas_sys::cublasZdotc_v2(
                self.raw_handle(), n,
                x_ptr as *const _, incx,
                y_ptr as *const _, incy,
                &mut result as *mut _ as *mut _,
            )
            .result()?;
            Ok(result)
        }
    }

    // ── IDAMAX ──

    /// Index (1-based) of element with maximum absolute value.
    pub fn iamax_f64(
        &self,
        n: i32,
        x: &CudaSlice<f64>,
        incx: i32,
    ) -> Result<i32, CublasError> {
        unsafe {
            let (x_ptr, _) = x.device_ptr(&self.stream);
            let mut result: i32 = 0;
            blas_sys::cublasIdamax_v2(
                self.raw_handle(), n,
                x_ptr as *const _, incx,
                &mut result as *mut _,
            )
            .result()?;
            Ok(result)
        }
    }
}

impl std::fmt::Debug for BlasHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlasHandle").finish_non_exhaustive()
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
    fn test_zgemm_small() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).unwrap();

        // A = [[1+i, 0], [0, 1-i]]  (2×2 complex, column-major)
        let a_data: Vec<CudaComplex> = vec![
            CudaComplex { x: 1.0, y: 1.0 }, CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 }, CudaComplex { x: 1.0, y: -1.0 },
        ];
        // B = [[1+0i, 2+0i], [3+0i, 4+0i]]  (2×2, column-major)
        let b_data: Vec<CudaComplex> = vec![
            CudaComplex { x: 1.0, y: 0.0 }, CudaComplex { x: 3.0, y: 0.0 },
            CudaComplex { x: 2.0, y: 0.0 }, CudaComplex { x: 4.0, y: 0.0 },
        ];

        let a_dev = stream.clone_htod(&a_data).unwrap();
        let b_dev = stream.clone_htod(&b_data).unwrap();
        let mut c_dev = stream.alloc_zeros::<CudaComplex>(4).unwrap();

        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::N,
                    transb: op::N,
                    m: 2, n: 2, k: 2,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: 2, ldb: 2, beta: CudaComplex { x: 0.0, y: 0.0 }, ldc: 2,
                },
                &a_dev, &b_dev, &mut c_dev,
            )
        }
        .unwrap();

        let c: Vec<CudaComplex> = stream.clone_dtoh(&c_dev).unwrap();
        // C = A·B = [[(1+i)*1 + 0*3, (1+i)*2 + 0*4],
        //            [0*1 + (1-i)*3, 0*2 + (1-i)*4]]
        //   = [[1+i, 2+2i], [3-3i, 4-4i]]
        let expected = [
            (1.0, 1.0), (3.0, -3.0),
            (2.0, 2.0), (4.0, -4.0),
        ];
        for i in 0..4 {
            let diff_x = (c[i].x - expected[i].0).abs();
            let diff_y = (c[i].y - expected[i].1).abs();
            assert!(
                diff_x < 1e-10 && diff_y < 1e-10,
                "Mismatch at {i}: got ({},{}), expected ({},{})",
                c[i].x, c[i].y, expected[i].0, expected[i].1,
            );
        }
    }

    #[test]
    fn test_dgemm() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).unwrap();

        let m = 3i32;
        let k = 2i32;
        let n = 4i32;

        let a_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f64> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];

        let a_dev = stream.clone_htod(&a_data).unwrap();
        let b_dev = stream.clone_htod(&b_data).unwrap();
        let mut c_dev = stream.alloc_zeros::<f64>((m * n) as usize).unwrap();

        unsafe {
            blas.gemm_f64(
                GemmConfig {
                    transa: op::N,
                    transb: op::N,
                    m, n, k,
                    alpha: 1.0,
                    lda: m,
                    ldb: k,
                    beta: 0.0,
                    ldc: m,
                },
                &a_dev, &b_dev, &mut c_dev,
            )
        }
        .unwrap();

        let c: Vec<f64> = stream.clone_dtoh(&c_dev).unwrap();
        assert!((c[0] - 39.0).abs() < 1e-10);
        assert!((c[1] - 54.0).abs() < 1e-10);
        assert!((c[4] - 68.0).abs() < 1e-10);
        assert!((c[11] - 123.0).abs() < 1e-10);
    }

    #[test]
    fn test_daxpy() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x_dev = stream.clone_htod(&x).unwrap();
        let mut y_dev = stream.alloc_zeros::<f64>(4).unwrap();

        blas.axpy_f64(4, 2.0, &x_dev, 1, &mut y_dev, 1).unwrap();
        let y: Vec<f64> = stream.clone_dtoh(&y_dev).unwrap();

        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[1] - 4.0).abs() < 1e-10);
        assert!((y[2] - 6.0).abs() < 1e-10);
        assert!((y[3] - 8.0).abs() < 1e-10);
    }
}
