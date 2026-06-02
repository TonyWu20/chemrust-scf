// Allow dead code for newtypes not yet used by current tasks
// but declared for forward-looking type-safety refactoring.
#![allow(dead_code)]

use std::ops::{Deref, DerefMut};

use cudarc::driver::CudaSlice;
use ndarray::Array2;
use num_complex::Complex64;

use crate::device::CudaComplex;

// ---------------------------------------------------------------------------
// Newtype structs for the Chebyshev–Davidson solver
// ---------------------------------------------------------------------------
//
// Each newtype wraps its inner type to disambiguate quantities that share the
// same raw type (e.g. multiple `CudaSlice<CudaComplex>` slices with different
// physical meanings, or multiple `Array2<Complex64>` matrices on the CPU).
//
// All newtypes:
//   - Derive `Debug`, `Clone`
//   - Implement `Deref<Target = Inner>` and `DerefMut`
//   - Construct via `::new(value: Inner) -> Self`
//   - Keep their inner field private
// ---------------------------------------------------------------------------

/// Plane-wave coefficients in PW × band layout on GPU.
#[derive(Debug, Clone)]
pub struct PwCoefficients(pub(crate) CudaSlice<CudaComplex>);

impl PwCoefficients {
    pub fn new(value: CudaSlice<CudaComplex>) -> Self {
        Self(value)
    }
}

impl Deref for PwCoefficients {
    type Target = CudaSlice<CudaComplex>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PwCoefficients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Ion × band beta-projections on GPU.
#[derive(Debug, Clone)]
pub struct BetaProjections(pub(crate) CudaSlice<CudaComplex>);

impl BetaProjections {
    pub fn new(value: CudaSlice<CudaComplex>) -> Self {
        Self(value)
    }
}

impl Deref for BetaProjections {
    type Target = CudaSlice<CudaComplex>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BetaProjections {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Subspace Hamiltonian matrix (CPU) used for mixing with the rotation matrix.
#[derive(Debug, Clone)]
pub struct SubspaceHamiltonian(pub(crate) Array2<Complex64>);

impl SubspaceHamiltonian {
    pub fn new(value: Array2<Complex64>) -> Self {
        Self(value)
    }
}

impl Deref for SubspaceHamiltonian {
    type Target = Array2<Complex64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SubspaceHamiltonian {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Rotation matrix (CPU) used for applying the correct subspace transform.
#[derive(Debug, Clone)]
pub struct RotationMatrix(pub(crate) Array2<Complex64>);

impl RotationMatrix {
    pub fn new(value: Array2<Complex64>) -> Self {
        Self(value)
    }
}

impl Deref for RotationMatrix {
    type Target = Array2<Complex64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RotationMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Preconditioner vector on GPU.
#[derive(Debug, Clone)]
pub struct PreconditionerVector(pub(crate) CudaSlice<f64>);

impl PreconditionerVector {
    pub fn new(value: CudaSlice<f64>) -> Self {
        Self(value)
    }
}

impl Deref for PreconditionerVector {
    type Target = CudaSlice<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PreconditionerVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Kinetic preconditioner T(G) on GPU.
#[derive(Debug, Clone)]
pub struct KineticPreconditioner(pub(crate) CudaSlice<f64>);

impl KineticPreconditioner {
    pub fn new(value: CudaSlice<f64>) -> Self {
        Self(value)
    }
}

impl Deref for KineticPreconditioner {
    type Target = CudaSlice<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for KineticPreconditioner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Residual S-inverse norms (plain L2 norms on CPU).
#[derive(Debug, Clone)]
pub struct ResidualSInvNorm(pub(crate) Vec<f64>);

impl ResidualSInvNorm {
    pub fn new(value: Vec<f64>) -> Self {
        Self(value)
    }
}

impl Deref for ResidualSInvNorm {
    type Target = Vec<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ResidualSInvNorm {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Locked-band mask (CPU).
#[derive(Debug, Clone)]
pub struct LockedMask(pub(crate) Vec<bool>);

impl LockedMask {
    pub fn new(value: Vec<bool>) -> Self {
        Self(value)
    }
}

impl Deref for LockedMask {
    type Target = Vec<bool>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LockedMask {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Block index (CPU scalar).
#[derive(Debug, Clone)]
pub struct BlockIndex(pub(crate) usize);

impl BlockIndex {
    pub fn new(value: usize) -> Self {
        Self(value)
    }
}

impl Deref for BlockIndex {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BlockIndex {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
