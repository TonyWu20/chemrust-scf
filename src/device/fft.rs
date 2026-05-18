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

/// A batched 3-D cuFFT plan backed by `cufftPlanMany`.
///
/// Each batch element is an independent 3-D transform. Data for each batch
/// element is contiguous and separated by `idist`/`odist` elements.
pub struct BatchedFftPlan3d {
    inner: CudaFft,
    nx: i32,
    ny: i32,
    nz: i32,
    batch: i32,
    /// Complex elements per input batch: (nx/2+1) × ny × nz
    idist: usize,
    /// Real elements per output batch: nx × ny × nz
    odist: usize,
}

impl BatchedFftPlan3d {
    /// Create a batched 3-D Z2D (complex→real) plan.
    ///
    /// cuFFT stores data in row-major order with `n[0]` innermost (fastest).
    /// The non-redundant half for C2R is `n[rank-1]` (the outermost/slowest
    /// dimension). With `n = [nx, ny, nz]`, the half dimension is `nz`, so
    /// the complex input per batch is `nx × ny × (nz/2 + 1)` elements.
    ///
    /// Each batch element transforms a `[nx, ny, nz/2+1]` complex array
    /// to a `[nx, ny, nz]` real array. Batch elements are contiguous:
    /// input stride = `nx * ny * (nz/2 + 1)`, output stride = `nx * ny * nz`.
    pub fn plan_batched_z2d(
        nx: i32,
        ny: i32,
        nz: i32,
        batch: i32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CufftError> {
        let nz_half = nz / 2 + 1;
        let idist = (nx * ny * nz_half) as usize;
        let odist = (nx * ny * nz) as usize;
        let inner = CudaFft::plan_many(
            &[nx, ny, nz],
            Some(&[nx, ny, nz_half]),
            1,
            idist as i32,
            Some(&[nx, ny, nz]),
            1,
            odist as i32,
            cufftType::CUFFT_Z2D,
            batch,
            stream,
        )?;
        Ok(Self { inner, nx, ny, nz, batch, idist, odist })
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
        let odist = (nx * ny * nz) as usize;
        let inner = CudaFft::plan_many(
            &[nx, ny, nz],
            Some(&[nx, ny, nz]),
            1,
            odist as i32,
            Some(&[nx, ny, nz]),
            1,
            odist as i32,
            cufftType::CUFFT_Z2Z,
            batch,
            stream,
        )?;
        Ok(Self { inner, nx, ny, nz, batch, idist: odist, odist })
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
    pub fn idist(&self) -> usize { self.idist }
    pub fn odist(&self) -> usize { self.odist }
    pub fn grid_dims(&self) -> (i32, i32, i32) { (self.nx, self.ny, self.nz) }
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
        let nz_half = nz / 2 + 1;                 // = 3 (half dim is n[rank-1])
        let n_complex = (nx * ny * nz_half) as usize;  // 4×4×3 = 48 per batch
        let n_real = (nx * ny * nz) as usize;           // 4×4×4 = 64 per batch
        let total_complex = n_complex * n_bands as usize;
        let total_real = n_real * n_bands as usize;

        let plan =
            BatchedFftPlan3d::plan_batched_z2d(nx, ny, nz, n_bands, stream.clone()).unwrap();

        assert_eq!(plan.idist(), n_complex);
        assert_eq!(plan.odist(), n_real);

        let mut d_input =
            stream.alloc_zeros::<CudaComplex>(total_complex).unwrap();
        let mut d_output =
            stream.alloc_zeros::<f64>(total_real).unwrap();

        // Band 0: DC component = 1.0+0i, all other freq zero
        // Band 1-3: all zero
        let mut h_input = vec![CudaComplex { x: 0.0, y: 0.0 }; total_complex];
        h_input[0] = CudaComplex { x: 1.0, y: 0.0 };
        stream.memcpy_htod(&h_input, &mut d_input).unwrap();

        unsafe {
            plan.z2d(&mut d_input, &mut d_output).unwrap();
        }
        stream.synchronize().unwrap();

        let result: Vec<f64> = stream.clone_dtoh(&d_output).unwrap();

        // Band 0: DC-only IFFT (unscaled) → uniform 1.0 at every grid point
        for idx in 0..n_real {
            let diff = (result[idx] - 1.0).abs();
            assert!(
                diff < 1e-10,
                "Band 0, element {idx}: expected 1.0, got {}",
                result[idx],
            );
        }

        // Bands 1-3: zero input → zero output (inter-band independence)
        for band in 1..n_bands as usize {
            let offset = band * n_real;
            for idx in 0..n_real {
                let diff = result[offset + idx].abs();
                assert!(
                    diff < 1e-10,
                    "Band {band}, element {idx}: expected 0.0, got {}",
                    result[offset + idx],
                );
            }
        }
    }

    #[test]
    fn test_batched_c2r_nonuniform() {
        // Two batches with different DC values at their natural plan offsets.
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let nx = 4i32;
        let ny = 4i32;
        let nz = 4i32;
        let n_bands = 2i32;
        let nz_half = nz / 2 + 1;
        let n_complex = (nx * ny * nz_half) as usize;
        let n_real = (nx * ny * nz) as usize;
        let total_complex = n_complex * n_bands as usize;
        let total_real = n_real * n_bands as usize;

        let plan =
            BatchedFftPlan3d::plan_batched_z2d(nx, ny, nz, n_bands, stream.clone()).unwrap();

        let mut d_input =
            stream.alloc_zeros::<CudaComplex>(total_complex).unwrap();
        let mut d_output =
            stream.alloc_zeros::<f64>(total_real).unwrap();

        // Band 0: DC=1.0+0i (all other freq zero), at natural offset 0
        // Band 1: DC=2.0+0i, at natural offset idist
        let mut h_input = vec![CudaComplex { x: 0.0, y: 0.0 }; total_complex];
        h_input[0] = CudaComplex { x: 1.0, y: 0.0 };
        h_input[n_complex] = CudaComplex { x: 2.0, y: 0.0 };
        stream.memcpy_htod(&h_input, &mut d_input).unwrap();

        unsafe {
            plan.z2d(&mut d_input, &mut d_output).unwrap();
        }
        stream.synchronize().unwrap();

        let result: Vec<f64> = stream.clone_dtoh(&d_output).unwrap();

        // Band 0: DC-only → uniform 1.0
        for idx in 0..n_real {
            assert!(
                (result[idx] - 1.0).abs() < 1e-10,
                "Band 0, element {idx}: expected 1.0, got {}",
                result[idx],
            );
        }

        // Band 1: DC-only with DC=2.0 → uniform 2.0
        for idx in 0..n_real {
            assert!(
                (result[n_real + idx] - 2.0).abs() < 1e-10,
                "Band 1, element {idx}: expected 2.0, got {}",
                result[n_real + idx],
            );
        }
    }
}
