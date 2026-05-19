use std::marker::PhantomData;
use std::sync::Arc;

use bon::bon;
use chemrust_hamiltonian_core::{CellGeometry, GVectorGrid, PseudopotentialSet, SpinPolicy, NonSpin};
use cudarc::driver::CudaContext;
use ndarray::{Array3, ShapeBuilder};
use num_complex::Complex64;

use crate::device::blas::BlasHandle;
use crate::device::solver::SolverHandle;
use crate::device::Gpu;
use crate::eigensolver::chebyshev::chebyshev_filter;
use crate::eigensolver::rayleigh_ritz::rayleigh_ritz;
use crate::layout::{ColumnDistributed, Cpu, WavefunctionSet};
use crate::mixing::DensityHistory;
use crate::types::{Density, EffectivePotential, Error, FinalResult, FineGridArray, KPoint, SmearingParams};

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
    pub(crate) cell: CellGeometry,
    pub(crate) pots: PseudopotentialSet,
    pub(crate) wave_grid: GVectorGrid,
    pub(crate) fine_grid: GVectorGrid,
    pub(crate) k_point: KPoint,
    pub(crate) smearing: SmearingParams,

    // --- Mutable state, governed by phase ---
    pub(crate) density: Density,
    pub(crate) psi: WavefunctionSet<ColumnDistributed>,
    pub(crate) eigenvalues: Vec<f64>,
    pub(crate) v_eff: Option<S::VEff>,
    pub(crate) history: DensityHistory,
    /// Will be read by `check()` in Phase 2 Goal 5. Suppressed until then.
    #[allow(dead_code)]
    pub(crate) previous_density: Density,

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
        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone())?;
        let solver = SolverHandle::new(stream.clone())?;

        // Extract V_eff as raw Array3<f64> via SpinPolicy::v_eff_for_spin
        let v_eff_ref = self.v_eff.as_ref().expect("VEffBuilt phase guarantees v_eff is Some");
        let v_eff_spin = S::v_eff_for_spin(v_eff_ref, 0);
        let v_eff_arr = v_eff_spin.as_array();

        // Downsample V_eff from fine grid to wave grid
        let v_eff_wave = downsample_array_to_wave_grid(v_eff_arr, &self.fine_grid, &self.wave_grid)?;
        let v_eff_gpu = Gpu::from_host(&v_eff_wave, &stream)?;

        // H2D psi
        let psi_gpu = Gpu::from_cpu(&Cpu::new(self.psi), &stream)?;
        let n_bands = psi_gpu.shape()[0];
        let n_pw = psi_gpu.shape()[1];

        // Clone eigenvalues before moving self
        let eig_clone = self.eigenvalues.clone();
        let eig = if eig_clone.is_empty() { None } else { Some(eig_clone.as_slice()) };

        // Chebyshev filter
        let (psi_filtered_row, hpsi_row) = chebyshev_filter(
            &psi_gpu, &v_eff_gpu, &self.pots,
            &self.wave_grid, &self.k_point, &self.cell,
            eig, ndeg, &blas, &stream, &ctx,
        )?;

        // Rayleigh-Ritz
        let (psi_new_gpu, eigenvalues_cpu) = rayleigh_ritz(
            &psi_filtered_row, &hpsi_row,
            n_bands, n_pw,
            &solver, &blas, &stream, &ctx,
        )?;

        stream.synchronize()?;
        let Cpu(psi_new) = psi_new_gpu.sync_to_host(&stream)?;
        let eigenvalues = eigenvalues_cpu.into_inner();

        Ok(ScfIteration {
            psi: psi_new,
            eigenvalues,
            cell: self.cell, pots: self.pots,
            wave_grid: self.wave_grid, fine_grid: self.fine_grid,
            k_point: self.k_point, smearing: self.smearing,
            density: self.density, v_eff: self.v_eff,
            history: self.history, previous_density: self.previous_density,
            _phase: PhantomData,
        })
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
    pub fn mix(mut self) -> ScfIteration<S, Mixed> {
        let (mixed, prev) = self.history.mix(self.density);
        ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_point: self.k_point,
            smearing: self.smearing,
            density: mixed,
            psi: self.psi,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: self.history,
            previous_density: prev,
            _phase: PhantomData,
        }
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

// ---------------------------------------------------------------------------
// V_eff downsampling (fine grid → wave grid)
// ---------------------------------------------------------------------------

/// Downsample a fine-grid real-space array to the wave (normal) FFT grid
/// via G-space truncation.
///
/// Forward FFT on the fine grid, copy only the wave-grid G-vector subset,
/// inverse FFT on the wave grid. Returns a `crate::types::EffectivePotential`
/// on the wave grid suitable for GPU upload.
pub(crate) fn downsample_array_to_wave_grid(
    fine_arr: &Array3<f64>,
    fine_grid: &GVectorGrid,
    wave_grid: &GVectorGrid,
) -> Result<EffectivePotential, Error> {
    use chemrust_hamiltonian_core::fft::{fft_forward_3d, fft_inverse_3d};

    let [ngz, ngy, ngx] = wave_grid.grid();
    let [ngz_f, ngy_f, ngx_f] = fine_grid.grid();

    // Forward FFT fine-grid V_eff → G-space
    let fine_g = fft_forward_3d(fine_arr).map_err(|_| Error::NotImplemented)?;

    // Truncate: copy only wave-grid G-vectors to a new G-space array
    let mut wave_g = Array3::<Complex64>::zeros((ngz, ngy, ngx).f());
    let gvecs_wave = wave_grid.gvecs();
    let f2ix = |f: i32, n: usize| -> usize {
        if f >= 0 { f as usize } else { (n as i32 + f) as usize }
    };
    ndarray::Zip::from(gvecs_wave)
        .and(&mut wave_g)
        .for_each(|&gf, coeff| {
            let fx = gf[0] as i32;
            let fy = gf[1] as i32;
            let fz = gf[2] as i32;
            let ix_f = f2ix(fx, ngx_f);
            let iy_f = f2ix(fy, ngy_f);
            let iz_f = f2ix(fz, ngz_f);
            *coeff = fine_g[[iz_f, iy_f, ix_f]];
        });

    // Inverse FFT back to real space on wave grid
    let n_total_fine = (ngx_f * ngy_f * ngz_f) as f64;
    let rho_wave = fft_inverse_3d(&wave_g).map_err(|_| Error::NotImplemented)?;
    let result = rho_wave.mapv(|x| x / n_total_fine);

    Ok(EffectivePotential::from_inner(FineGridArray::from_inner(result)))
}
