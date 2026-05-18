use ndarray::Array3;

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
}
