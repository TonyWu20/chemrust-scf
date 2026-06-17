//! Spin-polarised wrapper types for SCF state quantities.
//!
//! Each quantity that varies per spin channel is wrapped in a `SpinChannelData<T>`
//! newtype that validates the number of channels matches the chosen [`SpinPolicy`]
//! (1 for [`NonSpin`], 2 for [`SpinCollinear`]).
//!
//! CASTEP reference: `electronic.f90:488-495` (spin loop), `density.f90:30-37` (charge/spin storage).

use std::ops::{Deref, DerefMut, Index, IndexMut};

use cudarc::driver::CudaSlice;

#[allow(unused_imports)]
use chemrust_hamiltonian_core::{NonSpin, SpinCollinear, SpinPolicy};
use chemrust_hamiltonian_core::fft::RealGrid;

use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::PwCoefficients;
use crate::types::{Density, FineDensity};

// ---------------------------------------------------------------------------
// SpinChannelData<T> — generic container with length validation
// ---------------------------------------------------------------------------

/// A fixed-length container whose length is determined by the chosen [`SpinPolicy`].
///
/// - [`NonSpin`] → length 1
/// - [`SpinCollinear`] → length 2
///
/// Construction with `::new::<S>(vec)` asserts that `vec.len() == S::nspins()`.
#[derive(Debug, Clone)]
pub struct SpinChannelData<T>(Vec<T>);

impl<T> SpinChannelData<T> {
    /// Create a new `SpinChannelData`, asserting the vector length matches `S::nspins()`.
    ///
    /// # Panics
    ///
    /// Panics if `vec.len() != S::nspins()`.
    pub fn new<S: SpinPolicy>(vec: Vec<T>) -> Self {
        let expected = S::nspins();
        assert_eq!(
            vec.len(),
            expected,
            "SpinChannelData length mismatch: got {}, expected {} for {}",
            vec.len(),
            expected,
            std::any::type_name::<S>(),
        );
        Self(vec)
    }

    /// Access the element for spin channel `ispin` (0-based).
    pub fn for_spin(&self, ispin: usize) -> &T {
        &self.0[ispin]
    }

    /// Mutably access the element for spin channel `ispin` (0-based).
    pub fn for_spin_mut(&mut self, ispin: usize) -> &mut T {
        &mut self.0[ispin]
    }

    /// Number of spin channels (1 for NonSpin, 2 for SpinCollinear).
    pub fn nspins(&self) -> usize {
        self.0.len()
    }

    /// Consume self and return the inner `Vec<T>`.
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }

    /// Borrow the inner `Vec<T>`.
    pub fn as_vec(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T> Deref for SpinChannelData<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> DerefMut for SpinChannelData<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> Index<usize> for SpinChannelData<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for SpinChannelData<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.0[index]
    }
}

// ---------------------------------------------------------------------------
// KptDataSet<T> — generic container with length validation
// ---------------------------------------------------------------------------

/// A fixed-length container whose length is the number of k-points.
///
/// Construction with `::new(vec, expected_nkpts)` asserts that `vec.len() == expected_nkpts`.
/// Use `KptDataSet::new(vec![...], 1)` for gamma-point (single-kpt) wrapping.
///
/// When nested inside `SpinChannelData`, the full nesting is:
/// `SpinChannelData<KptDataSet<T>>` — spin-outer, kpt-inner.
///
/// Pattern follows `SpinChannelData<T>` exactly.
#[derive(Debug, Clone)]
pub struct KptDataSet<T>(Vec<T>);

impl<T> KptDataSet<T> {
    /// Create a new `KptDataSet`, asserting the vector length matches `expected_nkpts`.
    ///
    /// # Panics
    ///
    /// Panics if `vec.len() != expected_nkpts`.
    pub fn new(vec: Vec<T>, expected_nkpts: usize) -> Self {
        assert_eq!(
            vec.len(),
            expected_nkpts,
            "KptDataSet length mismatch: got {}, expected {} k-points",
            vec.len(),
            expected_nkpts,
        );
        Self(vec)
    }

    /// Number of k-points.
    pub fn nkpts(&self) -> usize {
        self.0.len()
    }

    /// Consume self and return the inner `Vec<T>`.
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }

    /// Borrow the inner `Vec<T>`.
    pub fn as_vec(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T> Deref for KptDataSet<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> DerefMut for KptDataSet<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> Index<usize> for KptDataSet<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for KptDataSet<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.0[index]
    }
}

// ---------------------------------------------------------------------------
// Per-spin eigenvalue and occupation newtypes
// ---------------------------------------------------------------------------

/// Eigenvalues (Hartree) for each spin channel, each being a `KptDataSet<Vec<f64>>`
/// (per-kpt eigenvalues of length = nbands).
#[derive(Debug, Clone)]
pub struct PerSpinEigenvalues(pub SpinChannelData<KptDataSet<Vec<f64>>>);

impl PerSpinEigenvalues {
    pub fn new(value: SpinChannelData<KptDataSet<Vec<f64>>>) -> Self {
        Self(value)
    }
}

impl Deref for PerSpinEigenvalues {
    type Target = SpinChannelData<KptDataSet<Vec<f64>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PerSpinEigenvalues {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Charge density ρ(r) for each spin channel.
///
/// - `NonSpin`: single density ρ
/// - `SpinCollinear`: two densities ρ↑, ρ↓
#[derive(Debug, Clone)]
pub struct PerSpinDensity(pub SpinChannelData<Density>);

impl PerSpinDensity {
    pub fn new(value: SpinChannelData<Density>) -> Self {
        Self(value)
    }

    /// Total charge density: `ρ↑ + ρ↓` (or ρ for NonSpin).
    pub fn total(&self) -> Density {
        if self.nspins() == 1 {
            self.0[0].clone()
        } else {
            self.0[0].clone() + self.0[1].clone()
        }
    }

    /// Spin density: `ρ↑ - ρ↓` (zero for NonSpin).
    pub fn spin(&self) -> SpinDensity {
        if self.nspins() == 1 {
            SpinDensity(self.0[0].clone() - self.0[0].clone())
        } else {
            SpinDensity(self.0[0].clone() - self.0[1].clone())
        }
    }
}

impl Deref for PerSpinDensity {
    type Target = SpinChannelData<Density>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PerSpinDensity {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Spin density `ρ_spin = ρ↑ - ρ↓` as a semantic marker newtype.
///
/// For NonSpin this is a zero field.
#[derive(Debug, Clone)]
pub struct SpinDensity(pub Density);

impl SpinDensity {
    pub fn into_inner(self) -> Density {
        self.0
    }
}

impl Deref for SpinDensity {
    type Target = Density;
    fn deref(&self) -> &Density {
        &self.0
    }
}

impl DerefMut for SpinDensity {
    fn deref_mut(&mut self) -> &mut Density {
        &mut self.0
    }
}

/// Occupation numbers (`f_nk`) for each spin channel.
#[derive(Debug, Clone)]
pub struct OccupationSet(pub SpinChannelData<Vec<f64>>);

impl OccupationSet {
    pub fn new(value: SpinChannelData<Vec<f64>>) -> Self {
        Self(value)
    }
}

impl Deref for OccupationSet {
    type Target = SpinChannelData<Vec<f64>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OccupationSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Fermi energy (Hartree) per spin channel.
///
/// Length-nspins vector. NonSpin → length 1, SpinCollinear → length 2.
#[derive(Debug, Clone)]
pub struct FermiEnergies(pub Vec<f64>);

impl FermiEnergies {
    pub fn new(value: Vec<f64>) -> Self {
        Self(value)
    }
}

impl Deref for FermiEnergies {
    type Target = Vec<f64>;
    fn deref(&self) -> &Vec<f64> {
        &self.0
    }
}

impl DerefMut for FermiEnergies {
    fn deref_mut(&mut self) -> &mut Vec<f64> {
        &mut self.0
    }
}

/// Electron counts `(n_up, n_dn)`. For NonSpin, `n_dn` is 0.0.
#[derive(Debug, Clone, Copy)]
pub struct ElectronCounts(pub [f64; 2]);

impl ElectronCounts {
    pub fn new(value: [f64; 2]) -> Self {
        Self(value)
    }
}

impl Deref for ElectronCounts {
    type Target = [f64; 2];
    fn deref(&self) -> &[f64; 2] {
        &self.0
    }
}

impl DerefMut for ElectronCounts {
    fn deref_mut(&mut self) -> &mut [f64; 2] {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// Per-spin wavefunction / projector newtypes
// ---------------------------------------------------------------------------

/// Plane-wave coefficients `C_{n𝐤}(𝐆)` for each spin channel and k-point.
///
/// Nested as `SpinChannelData<KptDataSet<PwCoefficients>>` — spin-outer, kpt-inner.
#[derive(Debug, Clone)]
pub struct PerSpinPwCoefficients(pub SpinChannelData<KptDataSet<PwCoefficients>>);

impl PerSpinPwCoefficients {
    pub fn new(value: SpinChannelData<KptDataSet<PwCoefficients>>) -> Self {
        Self(value)
    }
}

impl Deref for PerSpinPwCoefficients {
    type Target = SpinChannelData<KptDataSet<PwCoefficients>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PerSpinPwCoefficients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Non-local pseudopotential beta-projector data per spin channel and k-point, stored as
/// `Option` (None if projectors not yet initialised, Some if they are).
///
/// Each element is a `Vec` of beta-projector waves, one per atom, each a
/// `CudaSlice<CudaComplex>` on the GPU.
///
/// Nested as `SpinChannelData<KptDataSet<Option<Vec<CudaSlice<CudaComplex>>>>>` —
/// spin-outer, kpt-inner.
#[derive(Debug, Clone)]
pub struct PerSpinBetaProjections(
    pub SpinChannelData<KptDataSet<Option<Vec<CudaSlice<CudaComplex>>>>>,
);

impl PerSpinBetaProjections {
    pub fn new(value: SpinChannelData<KptDataSet<Option<Vec<CudaSlice<CudaComplex>>>>>) -> Self {
        Self(value)
    }
}

impl Deref for PerSpinBetaProjections {
    type Target = SpinChannelData<KptDataSet<Option<Vec<CudaSlice<CudaComplex>>>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PerSpinBetaProjections {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Augmentation density on the real-space fine grid per spin channel, stored as
/// `Option` (None if not yet initialised, Some if it is).
#[derive(Debug, Clone)]
pub struct PerSpinAugDensity(pub SpinChannelData<Option<RealGrid<f64>>>);

impl PerSpinAugDensity {
    pub fn new(value: SpinChannelData<Option<RealGrid<f64>>>) -> Self {
        Self(value)
    }
}

impl Deref for PerSpinAugDensity {
    type Target = SpinChannelData<Option<RealGrid<f64>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PerSpinAugDensity {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// PerSpinFineDensity — compile-time evidence of fine-grid density
// ---------------------------------------------------------------------------

/// Charge density ρ(r) on the **fine grid** for each spin channel.
///
/// Unlike [`PerSpinDensity`] (wave grid), this type guarantees at compile time
/// that the density lives on the fine grid — no upsampling is needed before
/// V_eff assembly.  Created after mixing and consumed by
/// `build_v_eff_with_energy`.
///
/// CASTEP stores its mixed density on the fine grid.
#[derive(Debug, Clone)]
pub struct PerSpinFineDensity(pub SpinChannelData<FineDensity>);

impl PerSpinFineDensity {
    pub fn new(value: SpinChannelData<FineDensity>) -> Self {
        Self(value)
    }

    /// Total charge density: `ρ↑ + ρ↓` (or ρ for NonSpin).
    pub fn total(&self) -> FineDensity {
        if self.nspins() == 1 {
            self[0].clone()
        } else {
            self[0].clone() + self[1].clone()
        }
    }

    /// Spin density: `ρ↑ - ρ↓` (zero for NonSpin).
    pub fn spin(&self) -> FineSpinDensity {
        if self.nspins() == 1 {
            FineSpinDensity(self[0].clone() - self[0].clone())
        } else {
            FineSpinDensity(self[0].clone() - self[1].clone())
        }
    }
}

impl Deref for PerSpinFineDensity {
    type Target = SpinChannelData<FineDensity>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PerSpinFineDensity {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Spin density `ρ↑ - ρ↓` on the fine grid.
#[derive(Debug, Clone)]
pub struct FineSpinDensity(pub FineDensity);

impl FineSpinDensity {
    pub fn into_inner(self) -> FineDensity {
        self.0
    }
}
