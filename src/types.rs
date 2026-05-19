use ndarray::Array3;
use std::ops::{Add, Mul, Sub};

// ---------------------------------------------------------------------------
// Grid-level newtypes
// ---------------------------------------------------------------------------

/// 3D array on the wavefunction-cutoff FFT grid.
#[derive(Debug, Clone)]
pub struct WaveGridArray(pub(crate) Array3<f64>);

impl WaveGridArray {
    pub fn as_array(&self) -> &Array3<f64> {
        &self.0
    }
    pub fn as_array_mut(&mut self) -> &mut Array3<f64> {
        &mut self.0
    }
    pub fn into_inner(self) -> Array3<f64> {
        self.0
    }
    pub fn from_inner(arr: Array3<f64>) -> Self {
        Self(arr)
    }
    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }
}

// ── Arithmetic ops: delegate to inner Array3<f64> ──

impl Add for WaveGridArray {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}
impl Sub for WaveGridArray {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}
impl Mul<f64> for WaveGridArray {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self(self.0 * rhs) }
}

/// 3D array on the 2x upsampled fine FFT grid (used for V_eff assembly).
#[derive(Debug, Clone)]
pub struct FineGridArray(pub(crate) Array3<f64>);

impl FineGridArray {
    pub fn as_array(&self) -> &Array3<f64> {
        &self.0
    }
    pub fn as_array_mut(&mut self) -> &mut Array3<f64> {
        &mut self.0
    }
    pub fn into_inner(self) -> Array3<f64> {
        self.0
    }
    pub fn from_inner(arr: Array3<f64>) -> Self {
        Self(arr)
    }
    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }
}

// ── Arithmetic ops ──

impl Add for FineGridArray {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}
impl Sub for FineGridArray {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}
impl Mul<f64> for FineGridArray {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self(self.0 * rhs) }
}

// ---------------------------------------------------------------------------
// Grid-aware field types
// ---------------------------------------------------------------------------

/// Electron density ρ(r) on the wave grid.
#[derive(Debug, Clone)]
pub struct Density(pub(crate) WaveGridArray);

impl Density {
    pub fn as_wave_array(&self) -> &Array3<f64> {
        self.0.as_array()
    }
    pub fn as_wave_array_mut(&mut self) -> &mut Array3<f64> {
        self.0.as_array_mut()
    }
    pub fn into_inner(self) -> WaveGridArray {
        self.0
    }
    pub fn from_inner(arr: WaveGridArray) -> Self {
        Self(arr)
    }
}

// ── Arithmetic ops ──

impl Add for Density {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}
impl Sub for Density {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}
impl Mul<f64> for Density {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self(self.0 * rhs) }
}

/// Effective potential V_eff[ρ] assembled on the fine grid.
#[derive(Debug, Clone)]
pub struct EffectivePotential(pub(crate) FineGridArray);

impl EffectivePotential {
    pub fn as_fine_array(&self) -> &Array3<f64> {
        self.0.as_array()
    }
    pub fn as_fine_array_mut(&mut self) -> &mut Array3<f64> {
        self.0.as_array_mut()
    }
    pub fn into_inner(self) -> FineGridArray {
        self.0
    }
    pub fn from_inner(arr: FineGridArray) -> Self {
        Self(arr)
    }
}

// ── Arithmetic ops ──

impl Add for EffectivePotential {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}
impl Sub for EffectivePotential {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}
impl Mul<f64> for EffectivePotential {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self(self.0 * rhs) }
}

/// Intermediate: density upsampled to the fine grid (before V_eff assembly).
#[derive(Debug, Clone)]
pub struct DensityUpsampled(pub(crate) FineGridArray);

impl DensityUpsampled {
    pub fn as_fine_array(&self) -> &Array3<f64> {
        self.0.as_array()
    }
    pub fn as_fine_array_mut(&mut self) -> &mut Array3<f64> {
        self.0.as_array_mut()
    }
    pub fn into_inner(self) -> FineGridArray {
        self.0
    }
    pub fn from_inner(arr: FineGridArray) -> Self {
        Self(arr)
    }
}

// ── Arithmetic ops ──

impl Add for DensityUpsampled {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}
impl Sub for DensityUpsampled {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}
impl Mul<f64> for DensityUpsampled {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self(self.0 * rhs) }
}

// ---------------------------------------------------------------------------
// Stub domain types
// ---------------------------------------------------------------------------

/// K-point in fractional reciprocal-lattice coordinates. Defaults to Gamma.
#[derive(Debug, Clone, Copy)]
pub struct KPoint {
    pub coords: [f64; 3],
}

impl Default for KPoint {
    fn default() -> Self {
        Self { coords: [0.0, 0.0, 0.0] }
    }
}

/// Fermi-Dirac smearing parameters.
#[derive(Debug, Clone, Copy)]
pub struct SmearingParams {
    /// Smearing width in Hartree.
    pub width: f64,
    /// Electron temperature kT in Hartree.
    pub electron_temperature: f64,
}

/// Output of a converged SCF calculation.
#[derive(Debug, Clone)]
pub struct FinalResult {
    pub density: Density,
    pub eigenvalues: Vec<f64>,
    pub total_energy: f64,
}

// ---------------------------------------------------------------------------
// Crate-level error type
// ---------------------------------------------------------------------------

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("not yet implemented")]
    NotImplemented,
    #[error("Chebyshev filtering diverged: norm grew {growth_factor:.2e}x at iteration {iteration}")]
    ChebyshevDiverged {
        iteration: usize,
        norm_previous: f64,
        norm_current: f64,
        growth_factor: f64,
    },
    #[error("Rayleigh-Ritz ZHEGVD failed: info={info}")]
    RayleighRitzFailed { info: i32 },
    #[error("cuSOLVER error: {0}")]
    Solver(#[from] crate::device::solver::SolverError),
    #[error("cuBLAS error: {0}")]
    Blas(#[from] cudarc::cublas::result::CublasError),
    #[error("cuFFT error: {0}")]
    Fft(#[from] cudarc::cufft::result::CufftError),
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::result::DriverError),
    #[error("NVRTC compilation error: {0}")]
    Nvrtc(String),
}
