pub mod fft;
pub mod blas;
pub mod solver;
pub mod pcie;

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
// CUDA complex type (re-export from cudarc, same layout as double2 / Complex64)
// ---------------------------------------------------------------------------

/// CUDA `double2` — `#[repr(C)] { x: f64, y: f64 }`, same layout as
/// `num_complex::Complex64`. Re-exported so FFT/cuBLAS wrappers can use
/// the same type as the underlying CUDA libraries.
pub use cudarc::cufft::sys::double2 as CudaComplex;

pub fn complex_to_cuda(c: Complex64) -> CudaComplex {
    CudaComplex { x: c.re, y: c.im }
}

pub fn cuda_to_complex(c: CudaComplex) -> Complex64 {
    Complex64::new(c.x, c.y)
}

pub fn complex_slice_to_cuda(data: &[Complex64]) -> Vec<CudaComplex> {
    data.iter().map(|&c| complex_to_cuda(c)).collect()
}

pub fn cuda_vec_to_complex(data: Vec<CudaComplex>) -> Vec<Complex64> {
    data.into_iter().map(cuda_to_complex).collect()
}

// ---------------------------------------------------------------------------
// DeviceMapped trait — element-type safe
// ---------------------------------------------------------------------------

/// Trait for types that can be transferred between CPU and GPU.
pub trait DeviceMapped: Sized {
    /// CUDA-compatible element type (`f64` for real, `CudaComplex` for complex).
    type Elem: DeviceRepr + ValidAsZeroBits;

    fn num_elems(&self) -> usize;
    fn shape_metadata(&self) -> Vec<usize>;
    fn flatten_host(&self) -> Vec<Self::Elem>;
    fn unflatten_host(data: Vec<Self::Elem>, shape: &[usize]) -> Self;
}

// ---------------------------------------------------------------------------
// f64-array helpers
// ---------------------------------------------------------------------------

fn shape3(arr: &Array3<f64>) -> Vec<usize> {
    vec![arr.shape()[0], arr.shape()[1], arr.shape()[2]]
}

pub(crate) fn flatten_f64(arr: &Array3<f64>) -> Vec<f64> {
    // Flatten in C order (x fastest → z slowest), matching CASTEP's
    // Fortran convention where the first dimension (x) varies fastest.
    // The GPU scatter formula iz + ngz*(iy + ngy*ix) uses a different
    // convention (z fastest); this is handled by transposing V_eff
    // before GPU upload in the diagonalize path.
    let (nx, ny, nz) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let mut out = Vec::with_capacity(nx * ny * nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                out.push(arr[[x, y, z]]);
            }
        }
    }
    out
}

pub(crate) fn unflatten_f64(data: Vec<f64>, shape: &[usize]) -> Array3<f64> {
    // Inverse of flatten_f64 — interprets flat data as C order (x fastest).
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let mut arr = Array3::zeros((nx, ny, nz));
    let mut idx = 0;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                arr[[x, y, z]] = data[idx];
                idx += 1;
            }
        }
    }
    arr
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
// DeviceMapped impls — complex-valued (Elem = CudaComplex / double2)
// ---------------------------------------------------------------------------

impl<L: Layout> DeviceMapped for WavefunctionSet<L> {
    type Elem = CudaComplex;

    fn num_elems(&self) -> usize {
        self.data.len()
    }

    fn shape_metadata(&self) -> Vec<usize> {
        vec![self.n_bands, self.n_pw]
    }

    fn flatten_host(&self) -> Vec<CudaComplex> {
        complex_slice_to_cuda(&self.data)
    }

    fn unflatten_host(data: Vec<CudaComplex>, shape: &[usize]) -> Self {
        let n_bands = shape[0];
        let n_pw = shape.get(1).copied().unwrap_or(1);
        let complex_data = cuda_vec_to_complex(data);
        Self::new(complex_data, n_bands, n_pw)
    }
}

// ---------------------------------------------------------------------------
// Gpu<T> — device-resident data, explicit sync only
// ---------------------------------------------------------------------------

/// GPU-resident data.
///
/// `Gpu<Density>` stores `CudaSlice<f64>`, `Gpu<WavefunctionSet>` stores
/// `CudaSlice<CudaComplex>` — the element type matches the domain quantity.
///
/// No `Deref<Target=T>` — prevents accidental CPU reads of GPU data.
#[derive(Debug)]
pub struct Gpu<T: DeviceMapped> {
    pub(crate) slice: CudaSlice<T::Elem>,
    pub(crate) shape: Vec<usize>,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) _marker: PhantomData<T>,
}

unsafe impl<T: DeviceMapped> Send for Gpu<T> {}
unsafe impl<T: DeviceMapped> Sync for Gpu<T> {}

impl<T: DeviceMapped> Gpu<T> {
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

    pub fn from_cpu(value: &Cpu<T>, stream: &Arc<CudaStream>) -> Result<Self, DriverError> {
        Self::from_host(&value.0, stream)
    }

    pub fn sync_to_host(&self, stream: &Arc<CudaStream>) -> Result<Cpu<T>, DriverError> {
        let data: Vec<T::Elem> = stream.clone_dtoh(&self.slice)?;
        let value = T::unflatten_host(data, &self.shape);
        Ok(Cpu(value))
    }

    pub fn as_device_slice(&self) -> &CudaSlice<T::Elem> {
        &self.slice
    }

    pub fn as_device_slice_mut(&mut self) -> &mut CudaSlice<T::Elem> {
        &mut self.slice
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

// ---------------------------------------------------------------------------
// Pinned host staging for async H2D
// ---------------------------------------------------------------------------

/// Pinned host staging buffer for asynchronous H2D copies.
///
/// A plain `Vec` dropped right after `memcpy_htod` can be reallocated and
/// overwritten while the DMA is still in flight. A 130 KB block crosses
/// the 128 KB mmap threshold, so the freed pages can be remapped and
/// rewritten before the delayed DMA completes. The GPU then reads mixed
/// values and the drift is silent.
///
/// `PinnedHost` holds a `cuMemHostAlloc` region for the whole scope that
/// enqueues the H2D. The dedicated pinned allocation is never reallocated
/// by the host allocator, so the async DMA is safe at any pacing. No extra
/// stream sync is needed. `Drop` calls `cuMemFreeHost`, which waits for
/// pending copies using this memory. Drop it only after the stream work
/// that reads it is enqueued, at the FFI boundary.
pub struct PinnedHost {
    ptr: *mut u8,
    size: usize,
}

impl PinnedHost {
    /// Allocate `size` bytes of pinned host memory.
    pub fn alloc(size: usize) -> Result<Self, crate::types::Error> {
        if size == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                size: 0,
            });
        }
        let p =
            unsafe { cudarc::driver::result::malloc_host(size, 0u32) }.map_err(crate::types::Error::Cuda)?;
        Ok(Self {
            ptr: p as *mut u8,
            size,
        })
    }

    /// View the buffer as a mutable `T` slice.
    /// `size` must be a multiple of `size_of::<T>()`.
    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        assert!(self.size % std::mem::size_of::<T>() == 0);
        let n = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, n) }
    }

    /// View the buffer as an immutable `T` slice.
    /// `size` must be a multiple of `size_of::<T>()`.
    pub fn as_slice<T>(&self) -> &[T] {
        assert!(self.size % std::mem::size_of::<T>() == 0);
        let n = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, n) }
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudarc::driver::result::free_host(self.ptr as *mut _) };
        }
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
        let _c64_slice: &CudaSlice<CudaComplex> = gpu_w.as_device_slice();
    }
}
