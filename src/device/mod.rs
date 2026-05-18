use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::{
    result::DriverError, CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits,
};
use ndarray::Array3;
use num_complex::Complex64;

use crate::layout::{Cpu, Layout, WavefunctionSet};
use crate::types::{
    Density, DensityUpsampled, EffectivePotential, FineGridArray, WaveGridArray,
};

// ---------------------------------------------------------------------------
// CUDA-compatible complex wrapper
// ---------------------------------------------------------------------------

/// GPU-compatible complex f64 type. Same repr as CUDA's `double2` /
/// `cuDoubleComplex` and `num_complex::Complex64`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaF64x2(pub f64, pub f64);

impl From<Complex64> for CudaF64x2 {
    fn from(c: Complex64) -> Self {
        Self(c.re, c.im)
    }
}

impl From<CudaF64x2> for Complex64 {
    fn from(c: CudaF64x2) -> Self {
        Self::new(c.0, c.1)
    }
}

impl From<&Complex64> for CudaF64x2 {
    fn from(c: &Complex64) -> Self {
        Self(c.re, c.im)
    }
}

// Safety: CudaF64x2 is #[repr(C)] with two f64, matching CUDA's double2 layout.
unsafe impl DeviceRepr for CudaF64x2 {}
// Safety: Zeroed bytes represent a valid CudaF64x2(0.0, 0.0).
unsafe impl ValidAsZeroBits for CudaF64x2 {}

// ---------------------------------------------------------------------------
// DeviceMapped trait — element-type safe
// ---------------------------------------------------------------------------

/// Trait for types that can be transferred between CPU and GPU.
///
/// The `Elem` associated type preserves the scalar type: `f64` for
/// real-valued quantities, `CudaF64x2` for complex-valued.
pub trait DeviceMapped: Sized {
    /// CUDA-compatible element type (`f64` for real, `CudaF64x2` for complex).
    type Elem: DeviceRepr + ValidAsZeroBits;

    /// Number of scalar elements when flattened.
    fn num_elems(&self) -> usize;

    /// Shape metadata for reconstruction (e.g. `[nx, ny, nz]` for 3-D grids).
    fn shape_metadata(&self) -> Vec<usize>;

    /// Flatten to host vector for H2D transfer.
    fn flatten_host(&self) -> Vec<Self::Elem>;

    /// Reconstruct from host data with given shape metadata.
    fn unflatten_host(data: Vec<Self::Elem>, shape: &[usize]) -> Self;
}

// ---------------------------------------------------------------------------
// f64-array helpers
// ---------------------------------------------------------------------------

fn shape3(arr: &Array3<f64>) -> Vec<usize> {
    vec![arr.shape()[0], arr.shape()[1], arr.shape()[2]]
}

fn flatten_f64(arr: &Array3<f64>) -> Vec<f64> {
    arr.iter().copied().collect()
}

fn unflatten_f64(data: Vec<f64>, shape: &[usize]) -> Array3<f64> {
    Array3::from_shape_vec(ndarray::Ix3(shape[0], shape[1], shape[2]), data)
        .expect("DeviceMapped: valid 3-D shape")
}

fn flatten_c64(data: &[Complex64]) -> Vec<CudaF64x2> {
    data.iter().map(|&c| CudaF64x2::from(c)).collect()
}

fn unflatten_c64(data: Vec<CudaF64x2>) -> Vec<Complex64> {
    data.into_iter().map(Complex64::from).collect()
}

// ---------------------------------------------------------------------------
// DeviceMapped impls — real-valued (Elem = f64)
// ---------------------------------------------------------------------------

macro_rules! impl_device_mapped_real {
    ($ty:ty) => {
        impl DeviceMapped for $ty {
            type Elem = f64;

            fn num_elems(&self) -> usize {
                self.0.len()
            }

            fn shape_metadata(&self) -> Vec<usize> {
                shape3(&self.0)
            }

            fn flatten_host(&self) -> Vec<f64> {
                flatten_f64(&self.0)
            }

            fn unflatten_host(data: Vec<f64>, shape: &[usize]) -> Self {
                Self(unflatten_f64(data, shape))
            }
        }
    };
}

impl_device_mapped_real!(WaveGridArray);
impl_device_mapped_real!(FineGridArray);

impl DeviceMapped for Density {
    type Elem = f64;

    fn num_elems(&self) -> usize {
        self.0.num_elems()
    }

    fn shape_metadata(&self) -> Vec<usize> {
        shape3(self.0.as_array())
    }

    fn flatten_host(&self) -> Vec<f64> {
        self.0.flatten_host()
    }

    fn unflatten_host(data: Vec<f64>, shape: &[usize]) -> Self {
        Self(WaveGridArray::unflatten_host(data, shape))
    }
}

impl DeviceMapped for EffectivePotential {
    type Elem = f64;

    fn num_elems(&self) -> usize {
        self.0.num_elems()
    }

    fn shape_metadata(&self) -> Vec<usize> {
        shape3(self.0.as_array())
    }

    fn flatten_host(&self) -> Vec<f64> {
        self.0.flatten_host()
    }

    fn unflatten_host(data: Vec<f64>, shape: &[usize]) -> Self {
        Self(FineGridArray::unflatten_host(data, shape))
    }
}

impl DeviceMapped for DensityUpsampled {
    type Elem = f64;

    fn num_elems(&self) -> usize {
        self.0.num_elems()
    }

    fn shape_metadata(&self) -> Vec<usize> {
        shape3(self.0.as_array())
    }

    fn flatten_host(&self) -> Vec<f64> {
        self.0.flatten_host()
    }

    fn unflatten_host(data: Vec<f64>, shape: &[usize]) -> Self {
        Self(FineGridArray::unflatten_host(data, shape))
    }
}

// ---------------------------------------------------------------------------
// DeviceMapped impls — complex-valued (Elem = CudaF64x2)
// ---------------------------------------------------------------------------

impl<L: Layout> DeviceMapped for WavefunctionSet<L> {
    type Elem = CudaF64x2;

    fn num_elems(&self) -> usize {
        self.data.len()
    }

    fn shape_metadata(&self) -> Vec<usize> {
        vec![self.n_bands, self.n_pw]
    }

    fn flatten_host(&self) -> Vec<CudaF64x2> {
        flatten_c64(&self.data)
    }

    fn unflatten_host(data: Vec<CudaF64x2>, shape: &[usize]) -> Self {
        let n_bands = shape[0];
        let n_pw = shape.get(1).copied().unwrap_or(1);
        let complex_data = unflatten_c64(data);
        Self::new(complex_data, n_bands, n_pw)
    }
}

// ---------------------------------------------------------------------------
// Gpu<T> — device-resident data, explicit sync only
// ---------------------------------------------------------------------------

/// GPU-resident data.
///
/// `Gpu<Density>` stores `CudaSlice<f64>`, `Gpu<WavefunctionSet>` stores
/// `CudaSlice<CudaF64x2>` — the element type matches the domain quantity.
///
/// No `Deref<Target=T>` — prevents accidental CPU reads of GPU data.
#[derive(Debug)]
pub struct Gpu<T: DeviceMapped> {
    /// Device buffer, typed by the domain quantity's scalar type.
    slice: CudaSlice<T::Elem>,
    /// Shape metadata for reconstruction.
    shape: Vec<usize>,
    /// CUDA context (keeps device alive).
    ctx: Arc<CudaContext>,
    _marker: PhantomData<T>,
}

// Safety: CudaSlice and Arc<CudaContext> are Send+Sync.
unsafe impl<T: DeviceMapped> Send for Gpu<T> {}
unsafe impl<T: DeviceMapped> Sync for Gpu<T> {}

impl<T: DeviceMapped> Gpu<T> {
    /// H2D: construct from a host value.
    pub fn from_host(value: &T, stream: &Arc<CudaStream>) -> Result<Self, DriverError> {
        let shape = value.shape_metadata();
        let host_data = value.flatten_host();
        let n = host_data.len();
        let mut slice: CudaSlice<T::Elem> = stream.alloc_zeros::<T::Elem>(n)?;
        stream.memcpy_htod(&host_data, &mut slice)?;
        let ctx = stream.context();
        Ok(Self {
            slice,
            shape,
            ctx: ctx.clone(),
            _marker: PhantomData,
        })
    }

    /// H2D from a `Cpu<T>` wrapper.
    pub fn from_cpu(value: &Cpu<T>, stream: &Arc<CudaStream>) -> Result<Self, DriverError> {
        Self::from_host(&value.0, stream)
    }

    /// D2H: transfer to host, returning `Cpu<T>`.
    pub fn sync_to_host(&self, stream: &Arc<CudaStream>) -> Result<Cpu<T>, DriverError> {
        let data: Vec<T::Elem> = stream.clone_dtoh(&self.slice)?;
        let value = T::unflatten_host(data, &self.shape);
        Ok(Cpu(value))
    }

    /// Read-only access to the device buffer (for cuBLAS/cuFFT ops).
    pub fn as_device_slice(&self) -> &CudaSlice<T::Elem> {
        &self.slice
    }

    /// Mutable access to the device buffer.
    pub fn as_device_slice_mut(&mut self) -> &mut CudaSlice<T::Elem> {
        &mut self.slice
    }

    /// Shape metadata.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements on device.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Context handle.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
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
    fn test_gpu_density_roundtrip() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let arr = Array3::<f64>::from_shape_vec(
            ndarray::Ix3(4, 4, 4),
            (0..64).map(|i| i as f64).collect(),
        )
        .unwrap();
        let density = Density(WaveGridArray(arr.clone()));

        let gpu = Gpu::from_host(&density, &stream).unwrap();
        let host_back: Cpu<Density> = gpu.sync_to_host(&stream).unwrap();

        assert_eq!(host_back.0.as_wave_array(), &arr);
    }

    #[test]
    fn test_gpu_wavefunction_roundtrip() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let data: Vec<Complex64> = (0..12)
            .map(|i| Complex64::new(i as f64, (i + 1) as f64))
            .collect();
        let wfn = WavefunctionSet::<crate::layout::ColumnDistributed>::new(data.clone(), 3, 4);

        let gpu = Gpu::from_host(&wfn, &stream).unwrap();
        let host_back: Cpu<WavefunctionSet<crate::layout::ColumnDistributed>> =
            gpu.sync_to_host(&stream).unwrap();

        assert_eq!(host_back.0.data, data);
    }

    #[test]
    fn test_gpu_element_type_distinction() {
        // Gpu<Density>::Elem = f64, Gpu<WavefunctionSet>::Elem = CudaF64x2
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let density = Density(WaveGridArray(Array3::<f64>::zeros(ndarray::Ix3(2, 2, 2))));
        let gpu_d: Gpu<Density> = Gpu::from_host(&density, &stream).unwrap();
        let _slice: &CudaSlice<f64> = gpu_d.as_device_slice();

        let wfn = WavefunctionSet::<crate::layout::ColumnDistributed>::new(
            vec![Complex64::new(1.0, 0.0); 4],
            2,
            2,
        );
        let gpu_w: Gpu<WavefunctionSet<crate::layout::ColumnDistributed>> =
            Gpu::from_host(&wfn, &stream).unwrap();
        let _c64_slice: &CudaSlice<CudaF64x2> = gpu_w.as_device_slice();
    }
}
