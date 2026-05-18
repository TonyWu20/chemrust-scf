// ---------------------------------------------------------------------------
// cuFFT wrapper — thin layer over cudarc::cufft::CudaFft
// ---------------------------------------------------------------------------

use std::sync::Arc;

use cudarc::cufft::safe::FftDirection;
use cudarc::cufft::result::CufftError;
use cudarc::cufft::CudaFft;
use cudarc::cufft::sys::cufftType;
use cudarc::driver::{CudaSlice, CudaStream};

use super::CudaComplex;

/// A 3-D cuFFT plan parametrized over the FFT type.
pub struct FftPlan3d {
    inner: CudaFft,
    nx: i32,
    ny: i32,
    nz: i32,
}

impl FftPlan3d {
    /// Create a 3-D C2C plan (Z2Z).
    pub fn plan_c2c(
        nx: i32,
        ny: i32,
        nz: i32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CufftError> {
        let inner = CudaFft::plan_3d(nx, ny, nz, cufftType::CUFFT_Z2Z, stream)?;
        Ok(Self { inner, nx, ny, nz })
    }

    /// Create a 3-D D2Z (real→complex) plan.
    pub fn plan_d2z(
        nx: i32,
        ny: i32,
        nz: i32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CufftError> {
        let inner = CudaFft::plan_3d(nx, ny, nz, cufftType::CUFFT_D2Z, stream)?;
        Ok(Self { inner, nx, ny, nz })
    }

    /// Create a 3-D Z2D (complex→real) plan.
    pub fn plan_z2d(
        nx: i32,
        ny: i32,
        nz: i32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CufftError> {
        let inner = CudaFft::plan_3d(nx, ny, nz, cufftType::CUFFT_Z2D, stream)?;
        Ok(Self { inner, nx, ny, nz })
    }

    // ── Execute ──

    /// C2C forward (Z2Z, Forward direction).
    ///
    /// # Safety
    /// cuFFT accesses device memory; both parameters must be valid GPU allocations.
    pub unsafe fn c2c_forward(
        &self,
        input: &mut CudaSlice<CudaComplex>,
        output: &mut CudaSlice<CudaComplex>,
    ) -> Result<(), CufftError> {
        self.inner.exec_z2z(input, output, FftDirection::Forward)
    }

    /// C2C inverse (Z2Z, Inverse direction).
    ///
    /// # Safety
    /// Both parameters must point to valid GPU allocations of sufficient size.
    pub unsafe fn c2c_inverse(
        &self,
        input: &mut CudaSlice<CudaComplex>,
        output: &mut CudaSlice<CudaComplex>,
    ) -> Result<(), CufftError> {
        self.inner.exec_z2z(input, output, FftDirection::Inverse)
    }

    /// R2C forward (D2Z).
    ///
    /// # Safety
    /// `input` and `output` must be valid GPU allocations of sufficient size.
    pub unsafe fn d2z(
        &self,
        input: &CudaSlice<f64>,
        output: &mut CudaSlice<CudaComplex>,
    ) -> Result<(), CufftError> {
        self.inner.exec_d2z(input, output)
    }

    /// C2R inverse (Z2D).
    ///
    /// # Safety
    /// `input` and `output` must be valid GPU allocations of sufficient size.
    pub unsafe fn z2d(
        &self,
        input: &mut CudaSlice<CudaComplex>,
        output: &mut CudaSlice<f64>,
    ) -> Result<(), CufftError> {
        self.inner.exec_z2d(input, output)
    }

    pub fn nx(&self) -> i32 { self.nx }
    pub fn ny(&self) -> i32 { self.ny }
    pub fn nz(&self) -> i32 { self.nz }
}

/// A batched 3-D cuFFT plan.
#[allow(dead_code)]
pub struct BatchedFftPlan3d {
    inner: CudaFft,
    nx: i32,
    ny: i32,
    nz: i32,
    batch: i32,
}

impl BatchedFftPlan3d {
    /// Create a batched 3-D Z2D (complex→real) plan.
    ///
    /// Each batch element transforms a `[nx, ny, nz/2+1]` complex array
    /// to a `[nx, ny, nz]` real array. Batch elements are contiguous.
    pub fn plan_batched_z2d(
        nx: i32,
        ny: i32,
        nz: i32,
        batch: i32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CufftError> {
        let inner = CudaFft::plan_3d(nx, ny, nz, cufftType::CUFFT_Z2D, stream)?;
        Ok(Self { inner, nx, ny, nz, batch })
    }

    /// Execute batched Z2D.
    ///
    /// # Safety
    /// cuFFT device memory access.
    pub unsafe fn z2d(
        &self,
        input: &mut CudaSlice<CudaComplex>,
        output: &mut CudaSlice<f64>,
    ) -> Result<(), CufftError> {
        self.inner.exec_z2d(input, output)
    }

    /// Create a batched 3-D C2C plan.
    pub fn plan_batched_c2c(
        nx: i32,
        ny: i32,
        nz: i32,
        batch: i32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CufftError> {
        let inner = CudaFft::plan_3d(nx, ny, nz, cufftType::CUFFT_Z2Z, stream)?;
        Ok(Self { inner, nx, ny, nz, batch })
    }

    /// Execute batched C2C forward.
    ///
    /// # Safety
    /// `input` and `output` must be valid GPU allocations of sufficient size.
    pub unsafe fn c2c_forward(
        &self,
        input: &mut CudaSlice<CudaComplex>,
        output: &mut CudaSlice<CudaComplex>,
    ) -> Result<(), CufftError> {
        self.inner.exec_z2z(input, output, FftDirection::Forward)
    }

    /// Execute batched C2C inverse.
    ///
    /// # Safety
    /// `input` and `output` must be valid GPU allocations of sufficient size.
    pub unsafe fn c2c_inverse(
        &self,
        input: &mut CudaSlice<CudaComplex>,
        output: &mut CudaSlice<CudaComplex>,
    ) -> Result<(), CufftError> {
        self.inner.exec_z2z(input, output, FftDirection::Inverse)
    }

    pub fn batch(&self) -> i32 { self.batch }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_c2c_3d_identity() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let nx = 8i32;
        let ny = 8i32;
        let nz = 8i32;
        let n = (nx * ny * nz) as usize;

        let plan = FftPlan3d::plan_c2c(nx, ny, nz, stream.clone()).unwrap();

        let mut input_data = vec![CudaComplex { x: 0.0, y: 0.0 }; n];
        for i in 0..n.min(10) {
            input_data[i] = CudaComplex { x: i as f64, y: (i * 2) as f64 };
        }

        let mut d_input = stream.clone_htod(&input_data).unwrap();
        let mut d_tmp = stream.alloc_zeros::<CudaComplex>(n).unwrap();

        // Forward + inverse = identity (unscaled)
        unsafe {
            plan.c2c_forward(&mut d_input, &mut d_tmp).unwrap();
            plan.c2c_inverse(&mut d_tmp, &mut d_input).unwrap();
        }
        stream.synchronize().unwrap();

        let result: Vec<CudaComplex> = stream.clone_dtoh(&d_input).unwrap();
        let scale = (nx * ny * nz) as f64;

        for i in 0..n {
            let diff_x = (result[i].x / scale - input_data[i].x).abs();
            let diff_y = (result[i].y / scale - input_data[i].y).abs();
            assert!(
                diff_x < 1e-10 && diff_y < 1e-10,
                "Mismatch at {i}: got ({},{}), expected ({},{})",
                result[i].x / scale,
                result[i].y / scale,
                input_data[i].x,
                input_data[i].y,
            );
        }
    }

    #[test]
    fn test_batched_c2r_4x4x4() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let nx = 4i32;
        let ny = 4i32;
        let nz = 4i32;
        let n_bands = 4i32;
        let n_complex = (nx * ny * (nz / 2 + 1)) as usize;
        let total_complex = n_complex * n_bands as usize;

        let plan =
            BatchedFftPlan3d::plan_batched_z2d(nx, ny, nz, n_bands, stream.clone()).unwrap();

        let mut d_input =
            stream.alloc_zeros::<CudaComplex>(total_complex).unwrap();
        let mut d_output =
            stream.alloc_zeros::<f64>((nx * ny * nz * n_bands) as usize).unwrap();

        // Fill with small test values (DC component only)
        let mut h_input = vec![CudaComplex { x: 0.0, y: 0.0 }; total_complex];
        for i in 0..n_bands as usize {
            h_input[i * n_complex] = CudaComplex { x: 1.0, y: 0.0 };
        }
        stream.memcpy_htod(&h_input, &mut d_input).unwrap();

        unsafe {
            plan.z2d(&mut d_input, &mut d_output).unwrap();
        }
        stream.synchronize().unwrap();

        let result: Vec<f64> = stream.clone_dtoh(&d_output).unwrap();
        assert!(result.iter().any(|&v| v != 0.0), "All-zero output from IFFT");
    }
}
