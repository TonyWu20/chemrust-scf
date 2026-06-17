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

impl Density {
    /// Reinterpret as a [`FineDensity`].  The inner array is consumed and
    /// re-wrapped — the caller must ensure the data is actually on the fine
    /// grid (i.e. has fine-grid dimensions).
    pub fn into_fine(self) -> FineDensity {
        FineDensity(FineGridArray(self.0 .0))
    }
}

/// Electron density ρ(r) on the fine grid.
///
/// Unlike [`Density`] (wave grid), this type is compile-time evidence that the
/// density is already on the fine grid — no upsampling is needed before V_eff
/// assembly.  Created by [`combine_soft_aug_on_fine`] and consumed by
/// [`build_v_eff_with_energy`].
///
/// CASTEP stores its mixed density on the fine grid (density.f90:597-616,
/// electronic.f90:597-616).
#[derive(Debug, Clone)]
pub struct FineDensity(pub(crate) FineGridArray);

impl FineDensity {
    /// View the fine-grid data as a raw `Array3<f64>`.
    pub fn as_fine_array(&self) -> &Array3<f64> {
        self.0.as_array()
    }

    /// Mutable view of the fine-grid data.
    pub fn as_fine_array_mut(&mut self) -> &mut Array3<f64> {
        self.0.as_array_mut()
    }

    /// Consume and return the inner [`FineGridArray`].
    pub fn into_inner(self) -> FineGridArray {
        self.0
    }

    /// Wrap a [`FineGridArray`].
    pub fn from_inner(arr: FineGridArray) -> Self {
        Self(arr)
    }

    /// Flatten to a `Vec<f64>` in row-major order (for GPU upload).
    pub fn flatten_host(&self) -> Vec<f64> {
        self.0.as_array().iter().cloned().collect()
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.0.as_array().len()
    }
}

// ── Arithmetic ops ──

impl Add for FineDensity {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}
impl Sub for FineDensity {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}
impl Mul<f64> for FineDensity {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Self(self.0 * rhs) }
}

impl FineDensity {
    /// Re-wrap as a [`Density`] for mixing.rs compatibility.
    /// The inner array retains fine-grid dimensions.
    pub fn to_density(&self) -> Density {
        Density(WaveGridArray(self.0.as_array().clone()))
    }
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
    /// K-point weight (sum of weights = 1 for Monkhorst-Pack, 1 for Gamma-point).
    pub weight: f64,
}

impl Default for KPoint {
    fn default() -> Self {
        Self { coords: [0.0, 0.0, 0.0], weight: 1.0 }
    }
}

/// Smearing scheme for occupation-number smearing.
#[derive(Debug, Clone, Copy, Default)]
pub enum SmearingScheme {
    /// Gaussian smearing (CASTEP default): occ = erfc((μ - ε) / width).
    #[default]
    Gaussian,
    // Future: FermiDirac, MethfesselPaxton, MarzariVanderbilt.
}

/// Smearing width with explicit unit, following the same pattern as
/// `castep_cell_io::CutOffEnergy`.  Carries an `Option<EnergyUnit>` so the
/// unit is preserved through parsing and serialization.
///
/// Internally, all computations use Hartree.  Call `.to_ha()` to convert.
///
/// Example:
/// ```ignore
/// let w = SmearingWidth::ev(0.1);          // 0.1 eV
/// let w = SmearingWidth::ha(0.003675);     // same in Hartree
/// assert!((w.to_ha() - 0.003675).abs() < 1e-8);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SmearingWidth {
    /// Value in the units specified by `unit`.
    pub value: f64,
    /// Energy unit (None = Hartree, default for internal use).
    pub unit: Option<castep_cell_io::units::EnergyUnit>,
}

impl SmearingWidth {
    /// CASTEP default smearing width: 0.1 eV.
    pub const fn ev(value: f64) -> Self {
        Self { value, unit: Some(castep_cell_io::units::EnergyUnit::ElectronVolt) }
    }

    /// Smearing width in Hartree (no unit marker).
    pub const fn ha(value: f64) -> Self {
        Self { value, unit: Some(castep_cell_io::units::EnergyUnit::Hartree) }
    }

    /// Convert to Hartree for internal computation.
    pub fn to_ha(self) -> f64 {
        match self.unit {
            None | Some(castep_cell_io::units::EnergyUnit::Hartree) => self.value,
            Some(castep_cell_io::units::EnergyUnit::ElectronVolt) => {
                self.value * crate::energy::EV_TO_HARTREE
            }
            Some(other) => {
                // Fallback: assume the value is already in Hartree and warn.
                tracing::warn!(
                    "SmearingWidth with unhandled unit {other:?}; treating as Hartree. \
                     Only Hartree and ElectronVolt are supported for smearing width."
                );
                self.value
            }
        }
    }
}

impl Default for SmearingWidth {
    /// CASTEP default: 0.1 eV (parameters.f90:1778).
    fn default() -> Self {
        Self::ev(0.1)
    }
}

/// Smearing parameters for occupation-number smearing.
///
/// CASTEP uses Gaussian smearing by default. The smearing width is
/// `SMEARING_WIDTH` in the `.param` file (default 0.1 eV for CASTEP).
/// Convert eV to Hartree: divide by 27.2114.
///
/// Builder: `SmearingParams::builder()` starts from CASTEP defaults.
/// Override per-fixture: `SmearingParams::builder().spin_fix(6).build()`.
#[derive(Debug, Clone, Copy, bon::Builder)]
pub struct SmearingParams {
    /// Smearing width.  CASTEP default: 0.1 eV.
    #[builder(default)]
    pub width: SmearingWidth,
    /// Electron temperature kT in Hartree (internal unit, no unit marker).
    #[builder(default = 0.1 * crate::energy::EV_TO_HARTREE)]
    pub electron_temperature: f64,
    /// Smearing scheme (CASTEP default: Gaussian).
    #[builder(default)]
    pub scheme: SmearingScheme,
    /// CASTEP spin_fix: SCF cycle (1-based) at which spin is freed.
    /// CASTEP default: 10 (parameters.f90:1778).
    /// CASTEP frees at scf_cycle == spin_fix (1-based); our 0-based
    /// scf_iter equivalent is scf_iter >= spin_fix - 1.
    /// Override per-fixture, e.g. NiO .param has `spin_fix : 6`.
    #[builder(default = 10i32)]
    pub spin_fix: i32,
}

/// Output of a converged SCF calculation.
#[derive(Debug, Clone)]
pub struct FinalResult {
    pub density: FineDensity,
    pub eigenvalues: crate::spin_types::PerSpinEigenvalues,
    pub total_energy: f64,
}

// ---------------------------------------------------------------------------
// Occupation numbers and chemical potential
// ---------------------------------------------------------------------------

/// Kinetic energies ½|G|² per plane-wave (Hartree), indexed by PW index.
#[derive(Debug, Clone)]
pub struct KineticEnergies(pub Vec<f64>);

/// Occupation numbers per band (dimensionless, 0–2 per band for spin-degenerate).
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct Occupations(pub Vec<f64>);

/// Chemical potential μ from smearing occupation search (Hartree).
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct ChemicalPotential(pub f64);

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
    #[error("I/O error: {0}")]
    Io(String),
}
