use std::marker::PhantomData;

use bon::bon;
use chemrust_hamiltonian_core::{CellGeometry, GVectorGrid, PseudopotentialSet, SpinPolicy, NonSpin};

use crate::layout::{ColumnDistributed, WavefunctionSet};
use crate::mixing::DensityHistory;
use crate::types::{Density, Error, FinalResult, KPoint, SmearingParams};

// ---------------------------------------------------------------------------
// Sealed phase trait and markers
// ---------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
}

/// Phase marker for the SCF state machine. Only 6 implementors exist.
pub trait ScfPhase: sealed::Sealed {
    fn name() -> &'static str;
}

macro_rules! define_phase {
    ($name:ident) => {
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl ScfPhase for $name {
            fn name() -> &'static str {
                stringify!($name)
            }
        }
    };
}

define_phase!(Initialized);
define_phase!(VEffBuilt);
define_phase!(WavefunctionsUpdated);
define_phase!(DensityUpdated);
define_phase!(Mixed);
define_phase!(Converged);

// ---------------------------------------------------------------------------
// ScfIteration — the type-state SCF engine
// ---------------------------------------------------------------------------

/// Central SCF state machine. Generic over spin policy `S` and current phase
/// `State`. Defaults: `S = NonSpin`, `State = Initialized`.
///
/// Each transition consumes `self` and returns a new `ScfIteration` with a
/// different phase marker. The compiler prevents calling a transition in the
/// wrong phase.
pub struct ScfIteration<
    S: SpinPolicy = NonSpin,
    State: ScfPhase = Initialized,
> {
    // --- Immutable across all SCF iterations ---
    pub cell: CellGeometry,
    pub pots: PseudopotentialSet,
    pub wave_grid: GVectorGrid,
    pub fine_grid: GVectorGrid,
    pub k_point: KPoint,
    pub smearing: SmearingParams,

    // --- Mutable state, governed by phase ---
    pub density: Density,
    pub psi: WavefunctionSet<ColumnDistributed>,
    pub eigenvalues: Vec<f64>,
    pub v_eff: Option<S::VEff>,
    pub history: DensityHistory,
    pub previous_density: Density,

    _phase: PhantomData<State>,
}

// ---------------------------------------------------------------------------
// Constructor (Initialized phase only)
// ---------------------------------------------------------------------------

#[bon]
impl<S: SpinPolicy> ScfIteration<S, Initialized> {
    /// Construct an SCF iteration state in the `Initialized` phase.
    #[builder]
    pub fn new(
        cell: CellGeometry,
        pots: PseudopotentialSet,
        wave_grid: GVectorGrid,
        fine_grid: GVectorGrid,
        density: Density,
        psi: WavefunctionSet<ColumnDistributed>,
        k_point: KPoint,
        smearing: SmearingParams,
        max_history: usize,
    ) -> Self {
        let previous_density = density.clone();
        Self {
            cell,
            pots,
            wave_grid,
            fine_grid,
            density,
            psi,
            k_point,
            smearing,
            eigenvalues: Vec::new(),
            v_eff: None,
            history: DensityHistory::new(max_history),
            previous_density,
            _phase: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Transition 1: Initialized → VEffBuilt
// ---------------------------------------------------------------------------

impl<S: SpinPolicy> ScfIteration<S, Initialized> {
    /// Assemble V_eff[ρ] from the current density.
    /// Consumes `self`, returns a state in the `VEffBuilt` phase.
    pub fn build_v_eff(self) -> Result<ScfIteration<S, VEffBuilt>, Error> {
        todo!()
    }
}

// ---------------------------------------------------------------------------
// Transition 2: VEffBuilt → WavefunctionsUpdated
// ---------------------------------------------------------------------------

impl<S: SpinPolicy> ScfIteration<S, VEffBuilt> {
    /// Chebyshev filter + Rayleigh-Ritz diagonalization of H[V_eff].
    /// `ndeg` is the Chebyshev polynomial degree.
    pub fn diagonalize(
        self,
        ndeg: usize,
    ) -> Result<ScfIteration<S, WavefunctionsUpdated>, Error> {
        let _ = ndeg;
        todo!()
    }
}

// ---------------------------------------------------------------------------
// Transition 3: WavefunctionsUpdated → DensityUpdated
// ---------------------------------------------------------------------------

impl<S: SpinPolicy> ScfIteration<S, WavefunctionsUpdated> {
    /// Construct new electron density from updated wavefunctions:
    /// ρ(r) = Σ_i occ_i |ψ_i(r)|².
    pub fn construct_density(
        self,
    ) -> Result<ScfIteration<S, DensityUpdated>, Error> {
        todo!()
    }
}

// ---------------------------------------------------------------------------
// Transition 4: DensityUpdated → Mixed (infallible)
// ---------------------------------------------------------------------------

impl<S: SpinPolicy> ScfIteration<S, DensityUpdated> {
    /// Mix new density with history (Pulay / DIIS in later phases).
    /// Infallible: mixing can always proceed, even if it degrades gracefully.
    pub fn mix(self) -> ScfIteration<S, Mixed> {
        todo!()
    }
}

// ---------------------------------------------------------------------------
// Transition 5: Mixed → Converged | Initialized
// ---------------------------------------------------------------------------

/// Outcome of a convergence check: converged, restart, or error.
pub enum CheckOutcome<S: SpinPolicy> {
    Converged(ScfIteration<S, Converged>),
    NotConverged(ScfIteration<S, Initialized>),
}

impl<S: SpinPolicy> ScfIteration<S, Mixed> {
    /// Check SCF convergence.
    ///
    /// Returns:
    /// - `Ok(Converged(...))` — converged; call `.finalize()` for output.
    /// - `Ok(NotConverged(...))` — not converged; restart loop with `v_eff: None`.
    /// - `Err(e)` — computation error (NaN density, etc.).
    pub fn check(
        self,
        tol: f64,
    ) -> Result<CheckOutcome<S>, Error> {
        let _ = tol;
        todo!()
    }
}

// ---------------------------------------------------------------------------
// Terminal: Converged → FinalResult
// ---------------------------------------------------------------------------

impl<S: SpinPolicy> ScfIteration<S, Converged> {
    /// Package converged results.
    pub fn finalize(self) -> FinalResult {
        todo!()
    }
}

// ---------------------------------------------------------------------------
// run_scf — orchestration loop
// ---------------------------------------------------------------------------

/// Run the full SCF cycle to convergence.
///
/// # Arguments
/// * `state` — Freshly initialized `ScfIteration` (Initialized phase).
/// * `ndeg` — Chebyshev polynomial degree for diagonalization.
/// * `tol` — Convergence tolerance (RMS density change in e⁻/Bohr³).
pub fn run_scf<S: SpinPolicy>(
    state: ScfIteration<S, Initialized>,
    ndeg: usize,
    tol: f64,
) -> Result<FinalResult, Error> {
    let mut state = state;
    loop {
        state = {
            let v_eff = state.build_v_eff()?;
            let wfn = v_eff.diagonalize(ndeg)?;
            let dens = wfn.construct_density()?;
            let mixed = dens.mix();
            match mixed.check(tol)? {
                CheckOutcome::Converged(converged) => return Ok(converged.finalize()),
                CheckOutcome::NotConverged(next) => next,
            }
        };
    }
}
