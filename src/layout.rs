use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::{result::DriverError, CudaStream};
use num_complex::Complex64;

use crate::device::{DeviceMapped, Gpu};

// ---------------------------------------------------------------------------
// Sealed layout-marker trait
// ---------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
}

/// Sealed trait for MPI distribution layout.
pub trait Layout: sealed::Sealed {
    fn name() -> &'static str;
}

/// Row-distributed layout: bands split across MPI processes.
pub struct RowDistributed;

/// Column-distributed layout: plane-wave coefficients split across MPI processes.
pub struct ColumnDistributed;

impl sealed::Sealed for RowDistributed {}
impl sealed::Sealed for ColumnDistributed {}

impl Layout for RowDistributed {
    fn name() -> &'static str {
        "row-distributed"
    }
}
impl Layout for ColumnDistributed {
    fn name() -> &'static str {
        "column-distributed"
    }
}

// ---------------------------------------------------------------------------
// WavefunctionSet<Layout>
// ---------------------------------------------------------------------------

/// Complex plane-wave coefficients for all bands, typed by distribution layout.
#[derive(Debug, Clone)]
pub struct WavefunctionSet<L: Layout> {
    pub data: Vec<Complex64>,
    pub n_bands: usize,
    pub n_pw: usize,
    _layout: PhantomData<L>,
}

impl<L: Layout> WavefunctionSet<L> {
    pub fn new(data: Vec<Complex64>, n_bands: usize, n_pw: usize) -> Self {
        debug_assert_eq!(
            data.len(),
            n_bands * n_pw,
            "WavefunctionSet data length {} != n_bands({}) * n_pw({})",
            data.len(),
            n_bands,
            n_pw
        );
        Self {
            data,
            n_bands,
            n_pw,
            _layout: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-resident wrapper
// ---------------------------------------------------------------------------

/// CPU-resident data. Always accessible via `Deref<Target=T>`.
#[derive(Debug, Clone, Default)]
pub struct Cpu<T>(pub T);

impl<T> Cpu<T> {
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: DeviceMapped> Cpu<T> {
    /// Transfer data from host to device.
    pub fn sync_to_device(&self, stream: &Arc<CudaStream>) -> Result<Gpu<T>, DriverError> {
        Gpu::from_host(&self.0, stream)
    }
}

impl<T> std::ops::Deref for Cpu<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Cpu<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
