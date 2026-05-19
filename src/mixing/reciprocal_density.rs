// ---------------------------------------------------------------------------
// Reciprocal-space density on the wave FFT grid (GPU-resident)
// ---------------------------------------------------------------------------

use crate::device::CudaComplex;
use cudarc::driver::CudaSlice;

/// Reciprocal-space density on the wave FFT grid, stored as GPU-resident
/// complex numbers.
///
/// The shape is `[ngz, ngy, ngx]` matching the cuFFT `plan_3d(ngx, ngy, ngz)`
/// layout where `ngx` is the fastest-varying dimension.  For an R2C transform
/// the non-redundant half would be in the `ngx` dimension, but the mixing
/// pipeline uses C2C transforms for simplicity, so the full complex grid is
/// always stored.
pub struct ReciprocalDensity {
    pub(crate) data: CudaSlice<CudaComplex>,
    pub(crate) shape: [usize; 3],  // [ngz, ngy, ngx]
}

impl ReciprocalDensity {
    /// Wrap an existing GPU allocation into a `ReciprocalDensity`.
    pub fn new(data: CudaSlice<CudaComplex>, shape: [usize; 3]) -> Self {
        Self { data, shape }
    }

    /// Access the underlying GPU buffer.
    pub fn as_device_slice(&self) -> &CudaSlice<CudaComplex> {
        &self.data
    }

    /// The grid shape `[ngz, ngy, ngx]`.
    pub fn shape(&self) -> &[usize; 3] {
        &self.shape
    }

    /// Number of complex elements on GPU.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
