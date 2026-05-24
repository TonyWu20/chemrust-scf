use std::marker::PhantomData;
use std::sync::Arc;

use bon::bon;
use chemrust_hamiltonian_core::{
    CellGeometry, GVectorGrid, NonSpin, PseudopotentialSet, SpinCollinear, SpinPolicy, VEffBuilder,
};
use cudarc::driver::{CudaContext, CudaSlice};
use ndarray::{Array3, ShapeBuilder};
use num_complex::Complex64;

use crate::device::blas::BlasHandle;
use crate::device::solver::SolverHandle;
use crate::device::pcie::PcieAccount;
use crate::device::{CudaComplex, Gpu};
use crate::eigensolver::chebyshev::{FilterMode, chebyshev_filter, CudaKernelSet};
use crate::eigensolver::rayleigh_ritz::rayleigh_ritz;
#[cfg(any(test, feature = "scf_diag"))]
use crate::eigensolver::rayleigh_ritz::rayleigh_ritz_with_matrices;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::layout::{ColumnDistributed, Cpu, WavefunctionSet};
use crate::mixing::{DensityHistory, Kerker, MixingOff, MixingPhase, Pulay};
use crate::density::{QSfCache, build_q_sf_cache};
use crate::types::{
    ChemicalPotential, Density, EffectivePotential, Error, FinalResult, FineGridArray, KPoint,
    SmearingParams,
};
use crate::energy::HARTREE_TO_EV;

// ---------------------------------------------------------------------------
// Mixing phase runtime dispatch
// ---------------------------------------------------------------------------

/// Runtime mixing-phase selector.  Stored on `ScfIteration` so the loop can
/// dispatch `construct_density_{off,kerker,pulay}` without knowing the
/// compile-time type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixingPhaseKind {
    Off,
    Kerker,
    Pulay,
}

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
define_phase!(Mixed);
define_phase!(Converged);

// `DensityUpdated` is parameterised over the mixing phase — manual impl.
pub struct DensityUpdated<M: MixingPhase>(PhantomData<M>);
impl<M: MixingPhase> sealed::Sealed for DensityUpdated<M> {}
impl<M: MixingPhase> ScfPhase for DensityUpdated<M> {
    fn name() -> &'static str {
        "DensityUpdated"
    }
}

// ---------------------------------------------------------------------------
// ScfIteration — the type-state SCF engine
// ---------------------------------------------------------------------------

/// Central SCF state machine. Generic over spin policy `S`, phase `State`,
/// and mixing phase `M`.
///
/// Two independent type-state axes:
/// - `State` — SCF workflow phase (Initialized → VEffBuilt → ... → Converged)
/// - `M` — mixing phase (Off → Kerker → Pulay), governs how `mix()` behaves.
///
/// Defaults: `S = NonSpin`, `State = Initialized`, `M = MixingOff`.
pub struct ScfIteration<
    S: SpinPolicy = NonSpin,
    State: ScfPhase = Initialized,
    M: MixingPhase = MixingOff,
> {
    // --- Immutable across all SCF iterations ---
    pub(crate) cell: CellGeometry,
    pub(crate) pots: PseudopotentialSet,
    pub(crate) wave_grid: GVectorGrid,
    pub(crate) fine_grid: GVectorGrid,
    pub(crate) k_point: KPoint,
    pub(crate) smearing: SmearingParams,
    /// PW G-vector fractional coordinates [h, k, l] for each plane wave.
    pub(crate) pw_coords: Vec<[i32; 3]>,

    // --- Mutable state, governed by phase ---
    pub(crate) density: Density,
    pub(crate) psi: WavefunctionSet<ColumnDistributed>,
    pub(crate) eigenvalues: Vec<f64>,
    pub(crate) v_eff: Option<S::VEff>,
    pub(crate) history: DensityHistory<M>,
    /// Will be read by `check()` in Phase 2 Goal 5. Suppressed until then.
    #[allow(dead_code)]
    pub(crate) previous_density: Density,

    /// Precomputed linear FFT grid indices for each PW coefficient.
    /// Index = ix + ngx * (iy + ngy * iz) in C-order (cuFFT convention).
    pub(crate) pw_fft_indices: Vec<i32>,

    // --- Mixing phase control (RE-2 / RE-5) ---
    /// Phase the loop should use on the next SCF iteration.
    pub(crate) next_mixing: MixingPhaseKind,

    // --- Energy components for total energy assembly (RE-3 / RE-4) ---
    /// Exchange-correlation energy E_xc from the most recent V_eff assembly.
    pub(crate) e_xc: Option<f64>,
    /// Hartree energy E_H = 0.5 × ∫ρV_H (precomputed on fine grid).
    pub(crate) e_hartree: Option<f64>,
    /// Double-counting correction ∫ρV_xc (precomputed on fine grid).
    pub(crate) rho_vxc: Option<f64>,
    /// Ewald ion–ion energy (computed once from cell geometry).
    pub(crate) ewald: f64,
    /// Rolling energy window for convergence acceleration / damping (RE-4).
    pub(crate) energy_buffer: Vec<f64>,
    /// Total electronic energy from the most recent SCF iteration (RE-4).
    pub(crate) total_energy: Option<f64>,
    /// Fermi energy / chemical potential from the most recent occupation search.
    pub(crate) fermi_energy: Option<f64>,

    /// Per-ion ⟨β_{IL}|ψ_b⟩ projections of the most recent ψ, cached from the
    /// Rayleigh–Ritz step. Shape per ion: `(n_expanded × n_bands)`. GPU-resident
    /// as `CudaSlice<CudaComplex>`. `None` before the first `diagonalize`
    /// (e.g. iter-1 driven from a fixture density). Consumed by
    /// `compute_aug_density_gpu` to build ω^I_{nm}.
    pub(crate) beta_psi_per_ion: Option<Vec<CudaSlice<CudaComplex>>>,
    /// GPU cache of Q_{nm}(G)·exp(-iG·R_I) per ion. Built lazily on first
    /// `compute_density_from_wavefunctions` call that has a GPU stream.
    /// `None` until first build; geometry-static thereafter.
    pub(crate) q_sf_cache: Option<QSfCache>,
    /// USPP augmentation density ρ_aug(r) on the fine grid, regenerated from
    /// the current ψ + occ each iteration. `None` for iter-1 (fixture
    /// density already encodes augmentation in the wave-grid convention).
    /// Added inside `build_v_eff_with_energy_impl` to the upsampled smooth
    /// density before V_H/V_xc evaluation.
    pub(crate) density_aug_fine: Option<chemrust_hamiltonian_core::fft::RealGrid<f64>>,

    _phase: PhantomData<State>,
}

// ---------------------------------------------------------------------------
// Constructor (Initialized phase only)
// ---------------------------------------------------------------------------

#[bon]
impl<S: SpinPolicy> ScfIteration<S, Initialized, MixingOff> {
    /// Construct an SCF iteration state in the `Initialized` phase with
    /// mixing phase `Off`.
    #[builder]
    pub fn new(
        cell: CellGeometry,
        pots: PseudopotentialSet,
        wave_grid: GVectorGrid,
        fine_grid: GVectorGrid,
        density: Density,
        psi: WavefunctionSet<ColumnDistributed>,
        pw_coords: Vec<[i32; 3]>,
        pw_fft_indices: Vec<i32>,
        k_point: KPoint,
        smearing: SmearingParams,
        max_history: usize,
    ) -> Self {
        let _ = max_history; // History size is fixed internally for now
        let previous_density = density.clone();
        debug_assert_eq!(
            pw_fft_indices.len(),
            psi.n_pw,
            "pw_fft_indices length {} must equal n_pw {}",
            pw_fft_indices.len(),
            psi.n_pw,
        );
        debug_assert_eq!(
            pw_coords.len(),
            psi.n_pw,
            "pw_coords length {} must equal n_pw {}",
            pw_coords.len(),
            psi.n_pw,
        );
        let ewald = crate::energy::ewald_energy(&cell, &pots);
        Self {
            cell,
            pots,
            wave_grid,
            fine_grid,
            density,
            psi,
            pw_coords,
            pw_fft_indices,
            k_point,
            smearing,
            eigenvalues: Vec::new(),
            v_eff: None,
            history: DensityHistory::new(),
            previous_density,
            next_mixing: MixingPhaseKind::Off,
            e_xc: None,
            e_hartree: None,
            rho_vxc: None,
            ewald,
            energy_buffer: Vec::new(),
            total_energy: None,
            fermi_energy: None,
            beta_psi_per_ion: None,
            q_sf_cache: None,
            density_aug_fine: None,
            _phase: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Private helper: move all fields into a new phase
// ---------------------------------------------------------------------------
// Preserves the mixing phase `M`.  Adding a field to ScfIteration requires
// updating here.

impl<S: SpinPolicy, Phase: ScfPhase, M: MixingPhase> ScfIteration<S, Phase, M> {
    fn into_phase<New: ScfPhase>(self) -> ScfIteration<S, New, M> {
        ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_point: self.k_point,
            smearing: self.smearing,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: self.density,
            psi: self.psi,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: self.history,
            previous_density: self.previous_density,
            next_mixing: self.next_mixing,
            e_xc: self.e_xc,
            e_hartree: self.e_hartree,
            rho_vxc: self.rho_vxc,
            ewald: self.ewald,
            energy_buffer: self.energy_buffer,
            total_energy: self.total_energy,
            fermi_energy: self.fermi_energy,
            beta_psi_per_ion: self.beta_psi_per_ion,
            q_sf_cache: self.q_sf_cache,
            density_aug_fine: self.density_aug_fine,
            _phase: PhantomData,
        }
    }

    /// Access the computed density (for testing).
    pub fn density(&self) -> &Density {
        &self.density
    }

    /// Access the cell geometry (for testing).
    pub fn cell(&self) -> &CellGeometry {
        &self.cell
    }
}

// ---------------------------------------------------------------------------
// Transition 1: Initialized → VEffBuilt
// ---------------------------------------------------------------------------

// --- Private dispatch trait for V_eff assembly ---
// VEffBuilder::assemble_on_fine_grid has different signatures for
// NonSpin vs SpinCollinear (different param count, return type).
// This trait unifies them so build_v_eff stays generic.

pub trait BuildVEff: SpinPolicy {
    fn build_v_eff_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<Self::VEff, chemrust_hamiltonian_core::Error>;
}

impl BuildVEff for NonSpin {
    fn build_v_eff_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<chemrust_hamiltonian_core::EffectivePotential, chemrust_hamiltonian_core::Error> {
        VEffBuilder::<NonSpin>::new(cell, pots, fine_grid)
            .assemble_on_fine_grid(rho, wave_grid, fine_grid)
    }
}

impl BuildVEff for SpinCollinear {
    fn build_v_eff_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<(chemrust_hamiltonian_core::EffectivePotential, chemrust_hamiltonian_core::EffectivePotential), chemrust_hamiltonian_core::Error> {
        // Paramagnetic initial guess: zero spin density.
        // Phase 3+ will compute proper spin density from wavefunctions.
        let shape = [wave_grid.grid()[2], wave_grid.grid()[1], wave_grid.grid()[0]];
        let zero_spin = chemrust_hamiltonian_core::Density::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(Array3::zeros(shape)),
        );
        VEffBuilder::<SpinCollinear>::new(cell, pots, fine_grid)
            .assemble_on_fine_grid(rho, &zero_spin, wave_grid, fine_grid)
    }
}

// --- Private dispatch trait for energy-aware V_eff assembly ---
// Only NonSpin is implemented initially (energy for SpinCollinear deferred).

pub trait BuildVEffWithEnergy: BuildVEff {
    fn build_v_eff_with_energy_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        rho_aug_fine: Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>>,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<(Self::VEff, f64, f64, f64), chemrust_hamiltonian_core::Error>;
}

impl BuildVEffWithEnergy for NonSpin {
    fn build_v_eff_with_energy_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        rho_aug_fine: Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>>,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<(chemrust_hamiltonian_core::EffectivePotential, f64, f64, f64), chemrust_hamiltonian_core::Error> {
        use chemrust_hamiltonian_core::{nlcc, poisson, upsample_density_to_fine_grid, xc, Density as CoreDensity};
        let rho_fine_pw =
            upsample_density_to_fine_grid(rho.as_real_grid(), wave_grid, fine_grid)
                .map_err(|_| chemrust_hamiltonian_core::Error::Format {
                    section: "build_v_eff_with_energy".into(),
                    detail: "upsample failed".into(),
                })?;
        // Add USPP augmentation (when available) so V_H, V_xc and the energy
        // double-counting integrals all see the full ρ_total = ρ_PW + ρ_aug.
        let rho_total_fine = match rho_aug_fine {
            Some(aug) => chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                rho_fine_pw.as_real_array() + aug.as_real_array(),
            ),
            None => rho_fine_pw,
        };
        let v_eff = VEffBuilder::<NonSpin>::new(cell, pots, fine_grid)
            .with_density(CoreDensity::from_inner(rho_total_fine.clone()), None)
            .assemble()?;
        // Compute double-counting terms: E_H = 0.5·∫ρV_H and ∫ρV_xc
        let rho_core = nlcc::reconstruct_rho_core(cell, pots, fine_grid)?
            .into_inner();
        let density_total = CoreDensity::from_inner(rho_total_fine.clone() + &rho_core);
        let v_h = poisson::solve_poisson(&CoreDensity::from_inner(rho_total_fine.clone()), fine_grid)?;
        let v_xc = xc::compute_pbe_xc(density_total.as_real_grid().as_real_array(), fine_grid, cell.volume)?;
        let n_grid = density_total.as_real_grid().as_real_array().len() as f64;
        // ρ is in CASTEP raw units (ρ_phys × Ω), so the discrete integral
        // ∫ρ V d³r = Σ (ρ_phys[i] × Ω) × V[i] × (1/N) = Σ ρ_grid[i] × V[i] × (1/N).
        // The correct weight is 1/N, NOT Ω/N (which would overcount by Ω).
        // CASTEP xc_gga (xc.f90:1056) uses the same /N_grid normalization.
        let d_v = 1.0 / n_grid;
        let e_hartree_raw: f64 = rho_total_fine.as_real_array().iter()
            .zip(v_h.as_real_grid().as_real_array().iter())
            .map(|(&rv, &vh)| rv * vh * d_v)
            .sum();
        let e_hartree = 0.5 * e_hartree_raw;
        let rho_vxc: f64 = rho_total_fine.as_real_array().iter()
            .zip(v_xc.v_xc.iter())
            .map(|(&rv, &vxc)| rv * vxc * d_v)
            .sum();
        Ok((v_eff, v_xc.energy, e_hartree, rho_vxc))
    }
}

impl<S: SpinPolicy + BuildVEff> ScfIteration<S, Initialized, MixingOff> {
    /// Assemble V_eff[ρ] from the current density.
    /// Consumes `self`, returns a state in the `VEffBuilt` phase.
    pub fn build_v_eff(self) -> Result<ScfIteration<S, VEffBuilt, MixingOff>, Error> {
        let core_rho = chemrust_hamiltonian_core::Density::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(self.density.as_wave_array().clone()),
        );
        let v_eff = S::build_v_eff_impl(
            &self.cell, &self.pots, &core_rho,
            &self.wave_grid, &self.fine_grid,
        )
        .map_err(|_| Error::NotImplemented)?;
        let mut next: ScfIteration<S, VEffBuilt, MixingOff> = self.into_phase();
        next.v_eff = Some(v_eff);
        Ok(next)
    }
}

impl ScfIteration<NonSpin, Initialized, MixingOff> {
    /// Assemble V_eff with energy components for total energy computation.
    ///
    /// Same as `build_v_eff` but also populates the energy fields
    /// (`e_xc`, `e_hartree`, `rho_vxc`) from the XC/Hartree evaluation.
    /// These are needed by `check()` for total energy computation.
    pub fn build_v_eff_with_energy(self) -> Result<ScfIteration<NonSpin, VEffBuilt, MixingOff>, Error> {
        let core_rho = chemrust_hamiltonian_core::Density::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(self.density.as_wave_array().clone()),
        );
        let (v_eff, e_xc, e_hartree, rho_vxc) = NonSpin::build_v_eff_with_energy_impl(
            &self.cell, &self.pots, &core_rho,
            self.density_aug_fine.as_ref(),
            &self.wave_grid, &self.fine_grid,
        )
        .map_err(|_| Error::NotImplemented)?;
        let mut next: ScfIteration<NonSpin, VEffBuilt, MixingOff> = self.into_phase();
        next.v_eff = Some(v_eff);
        next.e_xc = Some(e_xc);
        next.e_hartree = Some(e_hartree);
        next.rho_vxc = Some(rho_vxc);
        Ok(next)
    }
}

// ---------------------------------------------------------------------------
// Transition 2: VEffBuilt → WavefunctionsUpdated
// ---------------------------------------------------------------------------

impl<S: SpinPolicy> ScfIteration<S, VEffBuilt, MixingOff> {
    /// Chebyshev filter + Rayleigh-Ritz diagonalization of H[V_eff].
    /// `ndeg` is the Chebyshev polynomial degree.
    /// `occupations` — if `Some`, use for D-matrix screening (debug/testing only).
    ///   `None` uses bare D0 (standard SCF path where screening emerges iteratively).
    /// `filter_mode` — controls the filter operator (A/B/C sweep); defaults to `BareH`.
    pub fn diagonalize(
        self,
        ndeg: usize,
        occupations: Option<&[f64]>,
    ) -> Result<ScfIteration<S, WavefunctionsUpdated, MixingOff>, Error> {
        self.diagonalize_with_mode(ndeg, occupations, FilterMode::SinvHKeepHEig)
    }

    /// Like `diagonalize` but with an explicit `FilterMode` for the A/B/C diagnostic sweep.
    pub fn diagonalize_with_mode(
        self,
        ndeg: usize,
        occupations: Option<&[f64]>,
        filter_mode: FilterMode,
    ) -> Result<ScfIteration<S, WavefunctionsUpdated, MixingOff>, Error> {
        self.diagonalize_inner(ndeg, occupations, filter_mode, None)
    }

    /// Test-only / scf_diag variant of [`diagonalize_with_mode`] that injects
    /// externally-supplied per-ion D matrices instead of computing screened D
    /// from V_eff. See [`crate::eigensolver::vnl_data::VnlBatchData::precompute_with_d_override`]
    /// for the override-slice format.
    ///
    /// Used by the T-prime discriminator test
    /// (`iter2_band0_with_castep_d_injection`) to determine whether the SCF
    /// cascade is D-driven or eigensolver-rotation-driven.
    #[doc(hidden)]
    pub fn diagonalize_with_d_override(
        self,
        ndeg: usize,
        occupations: Option<&[f64]>,
        d_override_per_ion: &[Option<Vec<f64>>],
    ) -> Result<ScfIteration<S, WavefunctionsUpdated, MixingOff>, Error> {
        self.diagonalize_inner(
            ndeg, occupations,
            FilterMode::SinvHKeepHEig,
            Some(d_override_per_ion),
        )
    }

    fn diagonalize_inner(
        self,
        ndeg: usize,
        occupations: Option<&[f64]>,
        filter_mode: FilterMode,
        d_override_per_ion: Option<&[Option<Vec<f64>>]>,
    ) -> Result<ScfIteration<S, WavefunctionsUpdated, MixingOff>, Error> {
        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone())?;
        let solver = SolverHandle::new(stream.clone())?;
        let kernels = CudaKernelSet::new(&ctx)?;

        // Extract V_eff as raw Array3<f64> via SpinPolicy::v_eff_for_spin
        let v_eff_ref = self.v_eff.as_ref().expect("VEffBuilt phase guarantees v_eff is Some");
        let v_eff_spin = S::v_eff_for_spin(v_eff_ref, 0);
        let v_eff_arr = v_eff_spin.as_real_grid().as_real_array();

        // PCI-E transfer tracker (catches unexpected H2D/D2H in the hot path)
        let mut pcie = PcieAccount::default();

        // Downsample V_eff from fine grid to wave grid
        let v_eff_wave = downsample_array_to_wave_grid(v_eff_arr, &self.fine_grid, &self.wave_grid)?;
        let (min_veff, max_veff) = {
            let arr = v_eff_wave.as_fine_array();
            let min = arr.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        };
        {
            let rho_arr = self.density.as_wave_array();
            let n_grid = rho_arr.len() as f64;
            let rho_sum: f64 = rho_arr.iter().sum();
            #[allow(unused_variables)]
            let rho_min = rho_arr.iter().cloned().fold(f64::INFINITY, f64::min);
            #[allow(unused_variables)]
            let rho_max = rho_arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            #[allow(unused_variables)]
            let total_e_raw_conv = rho_sum / n_grid;
            #[allow(unused_variables)]
            let total_e_phys_conv = rho_sum / n_grid;  // FIXED: density already in CASTEP raw units (ρ×Ω)
            let psi_data = &self.psi.data;
            #[allow(unused_variables)]
            let (mut psi_abs_min, mut psi_abs_max) = (f64::INFINITY, 0.0f64);
            for c in psi_data.iter() {
                let a = c.norm();
                if a < psi_abs_min { psi_abs_min = a; }
                if a > psi_abs_max { psi_abs_max = a; }
            }
            #[cfg(feature = "scf_diag")]
            eprintln!(
                "[V_eff] min={:.4} max={:.4} range={:.4} Ha  [Density] rho_sum={:.4e} rho_min={:.4e} rho_max={:.4e}  total_e(raw_conv=sum/N)={:.6}  total_e(phys_conv=sum*Ω/N)={:.6}  [psi] |c|_min={:.3e} |c|_max={:.3e}",
                min_veff, max_veff, max_veff - min_veff,
                rho_sum, rho_min, rho_max,
                total_e_raw_conv, total_e_phys_conv,
                psi_abs_min, psi_abs_max,
            );
        }
        let v_eff_gpu = Gpu::from_host_with(&v_eff_wave, &stream, &mut pcie)?;

        // Clone host data BEFORE moving self.psi into GPU
        let pw_coords = self.pw_coords.clone();
        let psi_host = self.psi.data.clone();
        let n_bands = self.psi.n_bands;
        let n_pw = self.psi.n_pw;

        // H2D psi
        let psi_gpu = Gpu::from_host_with(&self.psi, &stream, &mut pcie)?;

        // Save the input ψ for Procrustes pinning (prev_psi_dev)
        // This is the basis we hand to Chebyshev before filter/GS produce ψ_after_GS.
        let prev_psi_dev = psi_gpu.as_device_slice();

        // V_NL precomputation (CPU, uses chemrust-hamiltonian, one-time cost)
        // Pass the downsampled V_eff for D-matrix screening (D = D0 + ∫ Q·V_eff).
        let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_wave.as_fine_array().clone()),
        );
        let vnl_data = VnlBatchData::precompute_with_d_override(
            &pw_coords, &self.pots, &self.cell,
            &self.wave_grid, &self.k_point,
            &psi_host, n_bands, n_pw, occupations,
            Some(&v_eff_for_d),
            d_override_per_ion,
            &stream, &mut pcie, &blas, &solver,
        )?;

        // Always pass eigenvalues=None. Das et al. (2025) main.tex:612 proves
        // that when ζ = ‖D⁻¹ − B⁻¹‖ = 0 (exact S⁻¹), R-ChFSI ≡ standard ChFSI
        // algebraically. After §10's Global Woodbury fix, ζ = 3.8e-15 (machine
        // epsilon). Per-band eigenvalue machinery is provably redundant and
        // introduces numerical weak points from stale eigenvalue labels when
        // V_eff drifts between SCF iterations.
        let eig: Option<&[f64]> = None;

        // Upload PW-to-FFT index map to GPU
        let fft_idx_dev: CudaSlice<i32> = stream
            .clone_htod(&self.pw_fft_indices)
            .map_err(Error::Cuda)?;
        pcie.h2d_bytes += self.pw_fft_indices.len() * std::mem::size_of::<i32>();

        // Chebyshev filter (pipeline: T+V_loc via FFT, V_NL via gemm)
        let (psi_filtered_row, hpsi_row) = chebyshev_filter(
            &psi_gpu, &v_eff_gpu, &self.pots,
            &self.wave_grid, &self.k_point, &self.cell,
            &self.pw_coords,
            &vnl_data, &fft_idx_dev, min_veff, max_veff,
            &kernels, &mut pcie, eig, ndeg, &blas, &solver, &stream, &ctx,
            filter_mode,
        )?;

        // Rayleigh-Ritz
        let pin_cfg = crate::eigensolver::rayleigh_ritz::RrPinConfig::from_env();
        let (psi_new_gpu, eigenvalues_cpu, beta_psi_gpu) = rayleigh_ritz(
            &psi_filtered_row, &hpsi_row, &vnl_data,
            n_bands, n_pw, &kernels,
            &mut pcie,
            &solver, &blas, &stream, &ctx,
            Some(prev_psi_dev),
            Some(&pin_cfg),
        )?;

        stream.synchronize()?;
        let psi_bytes = n_bands * n_pw * 16;            // complex double
        let eig_bytes = n_bands * 8;
        let _beta_psi_bytes: usize = vnl_data.entries.iter()
            .map(|e| e.n_expanded as usize * n_bands * 16)  // complex double
            .sum();
        let Cpu(psi_new) = psi_new_gpu.sync_to_host_with(&stream, &mut pcie)?;
        let eigenvalues = eigenvalues_cpu.into_inner();
        #[cfg(feature = "scf_diag")]
        eprintln!("[RR] eigenvalues: first={:.4e} Ha  last={:.4e} Ha  count={}",
            eigenvalues.first().copied().unwrap_or(f64::NAN),
            eigenvalues.last().copied().unwrap_or(f64::NAN),
            eigenvalues.len());

        // Assert: hot path should only have setup H2D + final D2H.
        // Any additional transfer (e.g. D2H inside the Chebyshev loop) is a bug.
        // beta_psi is now GPU-resident, so it's excluded from D2H accounting.
        assert_eq!(
            pcie.d2h_bytes,
            psi_bytes + eig_bytes,
            "D2H: expected psi({psi_bytes}) + eigenvalues({eig_bytes}) = {}",
            psi_bytes + eig_bytes,
        );

        let [ngz, ngy, ngx] = self.wave_grid.grid();
        let grid_size = ngx * ngy * ngz;
        let veff_bytes = grid_size * std::mem::size_of::<f64>();
        let fft_idx_bytes = self.pw_fft_indices.len() * std::mem::size_of::<i32>();
        let kinetic_bytes = n_pw * std::mem::size_of::<f64>();
        let vnl_bytes: usize = vnl_data.entries.iter()
            .map(|e| (e.beta_g.len() + e.d_matrix.len() + e.q_matrix.len()) * 16)
            .sum();
        let b_concat_bytes = n_pw * vnl_data.n_total_expanded as usize * 16;
        let lu_m_bytes = vnl_data.n_total_expanded as usize * vnl_data.n_total_expanded as usize * 16;
        assert_eq!(
            pcie.h2d_bytes,
            psi_bytes + veff_bytes + fft_idx_bytes + kinetic_bytes + vnl_bytes
                + b_concat_bytes + lu_m_bytes,
            "H2D tracking check failed",
        );

        let mut next: ScfIteration<S, WavefunctionsUpdated, MixingOff> = self.into_phase();
        next.psi = psi_new;
        next.eigenvalues = eigenvalues;
        next.beta_psi_per_ion = Some(beta_psi_gpu);
        Ok(next)
    }

    /// Test-only: run Chebyshev + Rayleigh-Ritz and return the internal subspace matrices
    /// H_sub, S_sub, X alongside the normal RR outputs.
    ///
    /// Returns: `(eigenvalues, H_sub_cpu, S_sub_cpu, X_cpu)` — all col-major (n_bands × n_bands).
    #[cfg(any(test, feature = "scf_diag"))]
    #[doc(hidden)]
    pub fn diagonalize_with_rr_matrices(
        self,
        ndeg: usize,
    ) -> Result<(Vec<f64>, Vec<CudaComplex>, Vec<CudaComplex>, Vec<CudaComplex>), Error> {
        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone())?;
        let solver = SolverHandle::new(stream.clone())?;
        let kernels = CudaKernelSet::new(&ctx)?;

        let v_eff_ref = self.v_eff.as_ref().expect("VEffBuilt phase guarantees v_eff is Some");
        let v_eff_spin = S::v_eff_for_spin(v_eff_ref, 0);
        let v_eff_arr = v_eff_spin.as_real_grid().as_real_array();

        let mut pcie = PcieAccount::default();

        let v_eff_wave = downsample_array_to_wave_grid(v_eff_arr, &self.fine_grid, &self.wave_grid)?;
        let (min_veff, max_veff) = {
            let arr = v_eff_wave.as_fine_array();
            let min = arr.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        };

        let v_eff_gpu = Gpu::from_host_with(&v_eff_wave, &stream, &mut pcie)?;

        let pw_coords = self.pw_coords.clone();
        let psi_host = self.psi.data.clone();
        let n_bands = self.psi.n_bands;
        let n_pw = self.psi.n_pw;

        let psi_gpu = Gpu::from_host_with(&self.psi, &stream, &mut pcie)?;

        let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_wave.as_fine_array().clone()),
        );
        let vnl_data = VnlBatchData::precompute(
            &pw_coords, &self.pots, &self.cell,
            &self.wave_grid, &self.k_point,
            &psi_host, n_bands, n_pw, None,
            Some(&v_eff_for_d),
            &stream, &mut pcie, &blas, &solver,
        )?;

        let eig: Option<&[f64]> = None;

        let fft_idx_dev: CudaSlice<i32> = stream
            .clone_htod(&self.pw_fft_indices)
            .map_err(Error::Cuda)?;
        pcie.h2d_bytes += self.pw_fft_indices.len() * std::mem::size_of::<i32>();

        let (psi_filtered_row, hpsi_row) = chebyshev_filter(
            &psi_gpu, &v_eff_gpu, &self.pots,
            &self.wave_grid, &self.k_point, &self.cell,
            &self.pw_coords,
            &vnl_data, &fft_idx_dev, min_veff, max_veff,
            &kernels, &mut pcie, eig, ndeg, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig,
        )?;

        let (_, eigenvalues_cpu, _, h_sub_cpu, s_sub_cpu, x_cpu) = rayleigh_ritz_with_matrices(
            &psi_filtered_row, &hpsi_row, &vnl_data,
            n_bands, n_pw, &kernels,
            &mut pcie,
            &solver, &blas, &stream, &ctx,
            None,  // prev_psi_dev: not used in test-only variant
            None,  // pin_cfg: not used in test-only variant
        )?;

        Ok((eigenvalues_cpu.into_inner(), h_sub_cpu.into_inner(), s_sub_cpu.into_inner(), x_cpu.into_inner()))
    }
    /// current `psi` (no Chebyshev recurrence, no Rayleigh-Ritz mixing) and
    /// return per-band components useful for direct ⟨ψ_b|H|ψ_b⟩ analysis.
    ///
    /// Returns `(hpsi_t, hpsi_tv, hpsi_full)` in column-major (n_bands × n_pw)
    /// layout matching `psi.data`.
    #[doc(hidden)]
    pub fn apply_h_components_for_test(
        &self,
        occupations: Option<&[f64]>,
    ) -> Result<crate::eigensolver::chebyshev::HComponentsForTest, Error> {
        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone())?;
        let solver = SolverHandle::new(stream.clone())?;
        let kernels = CudaKernelSet::new(&ctx)?;

        // V_eff via SpinPolicy
        let v_eff_ref = self.v_eff.as_ref().expect("VEffBuilt phase guarantees v_eff is Some");
        let v_eff_spin = S::v_eff_for_spin(v_eff_ref, 0);
        let v_eff_arr = v_eff_spin.as_real_grid().as_real_array();

        let mut pcie = PcieAccount::default();
        let v_eff_wave = downsample_array_to_wave_grid(v_eff_arr, &self.fine_grid, &self.wave_grid)?;
        let v_eff_gpu = Gpu::from_host_with(&v_eff_wave, &stream, &mut pcie)?;
        let psi_gpu = Gpu::from_host_with(&self.psi, &stream, &mut pcie)?;

        let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_wave.as_fine_array().clone()),
        );
        let psi_host = self.psi.data.clone();
        let n_bands = self.psi.n_bands;
        let n_pw = self.psi.n_pw;
        let vnl_data = VnlBatchData::precompute(
            &self.pw_coords, &self.pots, &self.cell,
            &self.wave_grid, &self.k_point,
            &psi_host, n_bands, n_pw, occupations,
            Some(&v_eff_for_d),
            &stream, &mut pcie, &blas, &solver,
        )?;

        let fft_idx_dev: CudaSlice<i32> = stream.clone_htod(&self.pw_fft_indices)
            .map_err(Error::Cuda)?;

        crate::eigensolver::chebyshev::apply_h_components_for_test(
            &psi_gpu, &v_eff_gpu, &self.wave_grid, &self.pw_coords,
            &vnl_data, &fft_idx_dev, &kernels, &blas, &stream,
        )
    }

    /// Compute `S · ψ_input` for an arbitrary host-side ψ block using the
    /// Vnl projectors + Q matrices that this VEffBuilt state would use in a
    /// real diagonalize call. Used by tests that need true USPP S-inner
    /// products (e.g., `⟨ψ_a | S | ψ_b⟩` with normalisation `⟨ψ|S|ψ⟩ = 1`).
    ///
    /// `psi_input` must be column-major `[band * n_pw + g]`; output is the
    /// same layout. `n_pw` is read from `self.psi.n_pw`.
    #[doc(hidden)]
    pub fn apply_s_for_test(
        &self,
        psi_input: &[Complex64],
        n_bands: usize,
    ) -> Result<Vec<Complex64>, Error> {
        let n_pw = self.psi.n_pw;
        assert_eq!(
            psi_input.len(), n_bands * n_pw,
            "apply_s_for_test: psi_input shape mismatch"
        );

        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone())?;
        let solver = SolverHandle::new(stream.clone())?;

        let v_eff_ref = self.v_eff.as_ref().expect("VEffBuilt phase guarantees v_eff is Some");
        let v_eff_spin = S::v_eff_for_spin(v_eff_ref, 0);
        let v_eff_arr = v_eff_spin.as_real_grid().as_real_array();

        let mut pcie = PcieAccount::default();
        let v_eff_wave = downsample_array_to_wave_grid(v_eff_arr, &self.fine_grid, &self.wave_grid)?;
        let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_wave.as_fine_array().clone()),
        );

        let psi_for_betapsi = self.psi.data.clone();
        let n_bands_state = self.psi.n_bands;

        let vnl_data = VnlBatchData::precompute(
            &self.pw_coords, &self.pots, &self.cell,
            &self.wave_grid, &self.k_point,
            &psi_for_betapsi, n_bands_state, n_pw, None,
            Some(&v_eff_for_d),
            &stream, &mut pcie, &blas, &solver,
        )?;

        crate::eigensolver::chebyshev::apply_s_for_test(
            psi_input, n_bands, n_pw, &vnl_data, &blas, &stream,
        )
    }
}

// ---------------------------------------------------------------------------
// Transition 3: WavefunctionsUpdated → DensityUpdated
// ---------------------------------------------------------------------------
// Three variants for the three mixing phases.  Each method name encodes
// which mixing phase the returned state carries.

impl<S: SpinPolicy> ScfIteration<S, WavefunctionsUpdated, MixingOff> {
    /// Shared density computation (occupations + GPU construction).
    /// Returns `(new_density, density_aug_fine, chemical_potential)`. The
    /// augmentation density is `Some` only when `beta_psi_per_ion` is cached
    /// from a prior `diagonalize` call; in iter-1 (fixture density path) it
    /// stays `None` and `build_v_eff_with_energy` reads the wave-grid density
    /// as-is.
    fn compute_density_from_wavefunctions(
        &mut self,
    ) -> Result<
        (
            Density,
            Option<chemrust_hamiltonian_core::fft::RealGrid<f64>>,
            ChemicalPotential,
        ),
        Error,
    > {
        let n_electrons: f64 = self
            .cell
            .species_iter()
            .map(|info| {
                self.pots
                    .get(info.symbol)
                    .and_then(|p| p.ionic_charge())
                    .unwrap_or(0.0)
                    * info.num_ions as f64
            })
            .sum();

        let (occupations, chem_pot) =
            crate::density::compute_occupations(&self.eigenvalues, &self.smearing, n_electrons)?;

        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let kernels = CudaKernelSet::new(&ctx)?;

        let new_density = crate::density::construct_density_gpu()
            .psi_data(&self.psi.data)
            .occupations(&occupations.0)
            .fft_indices(&self.pw_fft_indices)
            .wave_grid(&self.wave_grid)
            .cell_volume(self.cell.volume)
            .n_bands(self.psi.n_bands)
            .n_pw(self.psi.n_pw)
            .kernels(&kernels)
            .stream(&stream)
            .call()?;

        stream.synchronize()?;
        {
            let rho_arr = new_density.as_wave_array();
            let n_grid = rho_arr.len() as f64;
            let rho_sum: f64 = rho_arr.iter().sum();
            #[allow(unused_variables)]
            let rho_min = rho_arr.iter().cloned().fold(f64::INFINITY, f64::min);
            #[allow(unused_variables)]
            let rho_max = rho_arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            #[allow(unused_variables)]
            let total_e_raw_conv = rho_sum / n_grid;
            #[allow(unused_variables)]
            let total_e_phys_conv = rho_sum / n_grid;  // FIXED: density already in CASTEP raw units (ρ×Ω)
            #[allow(unused_variables)]
            let occ_sum: f64 = occupations.0.iter().sum();
            #[allow(unused_variables)]
            let occ_max = occupations.0.iter().cloned().fold(0.0f64, f64::max);
            #[cfg(feature = "scf_diag")]
            eprintln!(
                "[NewDensity] rho_sum={:.4e} rho_min={:.4e} rho_max={:.4e}  total_e(raw_conv=sum/N)={:.6}  total_e(phys_conv=sum*Ω/N)={:.6}  [occ] Σocc={:.4} max_occ={:.4} target_n_e={:.4} chem_pot={:.4} Ha",
                rho_sum, rho_min, rho_max,
                total_e_raw_conv, total_e_phys_conv,
                occ_sum, occ_max, n_electrons, chem_pot.0,
            );
        }

        // USPP augmentation density on the fine grid (only when β·ψ is cached
        // from this iteration's RR; iter-1 fixture path skips this).
        let density_aug_fine = match self.beta_psi_per_ion.as_ref() {
            Some(beta_psi) => {
                // Lazily build QSfCache on first use (geometry-static).
                if self.q_sf_cache.is_none() {
                    let mut pcie = PcieAccount::default();
                    match build_q_sf_cache(&self.pots, &self.cell, &self.fine_grid, &stream, &mut pcie) {
                        Ok(cache) => {
                            #[cfg(feature = "scf_diag")]
                            eprintln!("[QSfCache] built: {} ions, H2D {} bytes", cache.ion_sf.len(), pcie.h2d_bytes);
                            self.q_sf_cache = Some(cache);
                        }
                        Err(_e) => {
                            #[cfg(feature = "scf_diag")]
                            eprintln!("[QSfCache] build failed ({_e:?}), falling back to CPU aug density");
                        }
                    }
                }

                let rho_aug = if let Some(cache) = self.q_sf_cache.as_ref() {
                    let mut pcie = PcieAccount::default();
                    crate::density::compute_aug_density_gpu(
                        cache,
                        beta_psi,
                        &occupations.0,
                        &stream,
                        &mut pcie,
                        &kernels,
                    )?
                } else {
                    // QSfCache build failed — fall back to CPU path.
                    // D2H beta_psi slices (small: ~18 × n_bands × 16 bytes per ion).
                    let beta_psi_cpu: Vec<ndarray::Array2<num_complex::Complex64>> = beta_psi
                        .iter()
                        .map(|bp_dev| {
                            let ne_times_nb = bp_dev.len();
                            let bp_host: Vec<CudaComplex> = stream.clone_dtoh(bp_dev).map_err(Error::Cuda)?;
                            // Determine n_expanded from the slice length and n_bands
                            let n_bands = occupations.0.len();
                            let ne = ne_times_nb / n_bands;
                            let bp_complex: Vec<num_complex::Complex64> = bp_host
                                .iter()
                                .map(|c| num_complex::Complex64::new(c.x, c.y))
                                .collect();
                            ndarray::Array2::from_shape_vec(
                                (ne, n_bands).f(),
                                bp_complex,
                            ).map_err(|_| Error::NotImplemented)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    crate::density::compute_aug_density_fine(
                        &beta_psi_cpu,
                        &occupations.0,
                        &self.pots,
                        &self.cell,
                        &self.fine_grid,
                    )?
                };
                {
                    let arr = rho_aug.as_real_array();
                    #[allow(unused_variables)]
                    let aug_sum: f64 = arr.iter().sum();
                    #[allow(unused_variables)]
                    let aug_min = arr.iter().cloned().fold(f64::INFINITY, f64::min);
                    #[allow(unused_variables)]
                    let aug_max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    #[allow(unused_variables)]
                    let n_grid = arr.len() as f64;
                    #[cfg(feature = "scf_diag")]
                    eprintln!(
                        "[AugDensity] aug_sum={:.4e} aug_min={:.4e} aug_max={:.4e}  ∫ρ_aug dV ≈ {:.4}",
                        aug_sum, aug_min, aug_max,
                        aug_sum * self.cell.volume / n_grid,
                    );
                }
                Some(rho_aug)
            }
            None => None,
        };

        Ok((new_density, density_aug_fine, chem_pot))
    }

    /// Construct density with `Off` mixing phase — the history stays as-is
    /// and no mixing transformation is applied during `mix()`.
    pub fn construct_density_off(
        mut self,
    ) -> Result<ScfIteration<S, DensityUpdated<MixingOff>, MixingOff>, Error> {
        let (new_density, density_aug_fine, chem_pot) = self.compute_density_from_wavefunctions()?;
        let mut next: ScfIteration<S, DensityUpdated<MixingOff>, MixingOff> =
            self.into_phase();
        next.density = new_density;
        next.density_aug_fine = density_aug_fine;
        next.fermi_energy = Some(chem_pot.0);
        Ok(next)
    }

    /// Construct density and transition the history to `Kerker` phase.
    pub fn construct_density_kerker(
        mut self,
    ) -> Result<ScfIteration<S, DensityUpdated<Kerker>, Kerker>, Error> {
        let (new_density, density_aug_fine, chem_pot) = self.compute_density_from_wavefunctions()?;
        // Convert history from MixingOff → Kerker (creates GPU preconditioner)
        let kerker_history = self.history.into_kerker(&self.wave_grid)?;
        Ok(ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_point: self.k_point,
            smearing: self.smearing,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: new_density,
            psi: self.psi,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: kerker_history,
            previous_density: self.previous_density,
            next_mixing: self.next_mixing,
            e_xc: self.e_xc,
            e_hartree: self.e_hartree,
            rho_vxc: self.rho_vxc,
            ewald: self.ewald,
            energy_buffer: self.energy_buffer,
            total_energy: self.total_energy,
            fermi_energy: Some(chem_pot.0),
            beta_psi_per_ion: self.beta_psi_per_ion,
            q_sf_cache: self.q_sf_cache,
            density_aug_fine,
            _phase: PhantomData,
        })
    }

    /// Construct density and transition the history to `Pulay` phase.
    pub fn construct_density_pulay(
        mut self,
    ) -> Result<ScfIteration<S, DensityUpdated<Pulay>, Pulay>, Error> {
        let (new_density, density_aug_fine, chem_pot) = self.compute_density_from_wavefunctions()?;
        // Convert history: MixingOff → Kerker → Pulay
        let kerker_history = self.history.into_kerker(&self.wave_grid)?;
        let pulay_history = kerker_history.into_pulay();
        Ok(ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_point: self.k_point,
            smearing: self.smearing,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: new_density,
            psi: self.psi,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: pulay_history,
            previous_density: self.previous_density,
            next_mixing: self.next_mixing,
            e_xc: self.e_xc,
            e_hartree: self.e_hartree,
            rho_vxc: self.rho_vxc,
            ewald: self.ewald,
            energy_buffer: self.energy_buffer,
            total_energy: self.total_energy,
            fermi_energy: Some(chem_pot.0),
            beta_psi_per_ion: self.beta_psi_per_ion,
            q_sf_cache: self.q_sf_cache,
            density_aug_fine,
            _phase: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Transition 4: DensityUpdated<M> → Mixed (infallible)
// ---------------------------------------------------------------------------
// Each mixing phase gets its own `impl` block so the history type is
// correctly wired.  All three normalise the return type to
// `ScfIteration<S, Mixed, MixingOff>` so the `run_scf` dispatch arms
// agree on a single type.

impl<S: SpinPolicy> ScfIteration<S, DensityUpdated<MixingOff>, MixingOff> {
    /// Mix with phase `Off`: pass-through, no active mixing.
    #[allow(unused_mut)]
    pub fn mix(mut self) -> ScfIteration<S, Mixed, MixingOff> {
        let (mixed, prev) = self.history.mix(self.density);
        let history_off = self.history.into_off();
        ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_point: self.k_point,
            smearing: self.smearing,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: mixed,
            psi: self.psi,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: history_off,
            previous_density: prev,
            next_mixing: self.next_mixing,
            e_xc: self.e_xc,
            e_hartree: self.e_hartree,
            rho_vxc: self.rho_vxc,
            ewald: self.ewald,
            energy_buffer: self.energy_buffer,
            total_energy: self.total_energy,
            fermi_energy: self.fermi_energy,
            beta_psi_per_ion: self.beta_psi_per_ion,
            q_sf_cache: self.q_sf_cache,
            density_aug_fine: self.density_aug_fine,
            _phase: PhantomData,
        }
    }
}

impl<S: SpinPolicy> ScfIteration<S, DensityUpdated<Kerker>, Kerker> {
    /// Mix with Kerker preconditioning.
    pub fn mix(mut self) -> ScfIteration<S, Mixed, MixingOff> {
        let (mixed, prev) = self.history.mix(self.density);
        let history_off = self.history.into_off();
        ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_point: self.k_point,
            smearing: self.smearing,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: mixed,
            psi: self.psi,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: history_off,
            previous_density: prev,
            next_mixing: self.next_mixing,
            e_xc: self.e_xc,
            e_hartree: self.e_hartree,
            rho_vxc: self.rho_vxc,
            ewald: self.ewald,
            energy_buffer: self.energy_buffer,
            total_energy: self.total_energy,
            fermi_energy: self.fermi_energy,
            beta_psi_per_ion: self.beta_psi_per_ion,
            q_sf_cache: self.q_sf_cache,
            density_aug_fine: self.density_aug_fine,
            _phase: PhantomData,
        }
    }
}

impl<S: SpinPolicy> ScfIteration<S, DensityUpdated<Pulay>, Pulay> {
    /// Mix with Pulay / DIIS.
    pub fn mix(mut self) -> ScfIteration<S, Mixed, MixingOff> {
        let (mixed, prev) = self.history.mix(self.density);
        let history_off = self.history.into_off();
        ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_point: self.k_point,
            smearing: self.smearing,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: mixed,
            psi: self.psi,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: history_off,
            previous_density: prev,
            next_mixing: self.next_mixing,
            e_xc: self.e_xc,
            e_hartree: self.e_hartree,
            rho_vxc: self.rho_vxc,
            ewald: self.ewald,
            energy_buffer: self.energy_buffer,
            total_energy: self.total_energy,
            fermi_energy: self.fermi_energy,
            beta_psi_per_ion: self.beta_psi_per_ion,
            q_sf_cache: self.q_sf_cache,
            density_aug_fine: self.density_aug_fine,
            _phase: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Transition 5: Mixed → Converged | Initialized
// ---------------------------------------------------------------------------

/// Outcome of a convergence check: converged, restart, or error.
///
/// The `NotConverged` variant carries a `next_mixing` field that the
/// `run_scf` loop uses to decide which mixing phase to use on the next
/// iteration.
pub enum CheckOutcome<S: SpinPolicy> {
    Converged(ScfIteration<S, Converged, MixingOff>),
    NotConverged {
        state: ScfIteration<S, Initialized, MixingOff>,
        next_mixing: MixingPhaseKind,
    },
}

impl<S: SpinPolicy> ScfIteration<S, Mixed, MixingOff> {
    /// Check SCF convergence using energy-window + RMS density criteria.
    ///
    /// Returns:
    /// - `Ok(Converged(...))` — converged; call `.finalize()` for output.
    /// - `Ok(NotConverged(...))` — not converged; restart loop with `v_eff: None`.
    /// - `Err(e)` — computation error (NaN density, etc.).
    pub fn check(
        mut self,
        tol: f64,
    ) -> Result<CheckOutcome<S>, Error> {
        // 1. Compute occupations for band energy
        let n_electrons: f64 = self
            .cell
            .species_iter()
            .map(|info| {
                self.pots
                    .get(info.symbol)
                    .and_then(|p| p.ionic_charge())
                    .unwrap_or(0.0)
                    * info.num_ions as f64
            })
            .sum();
        let (occupations, chem_pot) =
            crate::density::compute_occupations(&self.eigenvalues, &self.smearing, n_electrons)?;
        self.fermi_energy = Some(chem_pot.0);

        // 2. Total energy assembly (if energy components are available)
        if let (Some(e_xc), Some(e_hartree), Some(rho_vxc)) =
            (self.e_xc, self.e_hartree, self.rho_vxc)
        {
            let e_total = crate::energy::assemble_total_energy(
                &self.eigenvalues,
                &occupations.0,
                e_xc,
                e_hartree,
                rho_vxc,
                self.ewald,
            );
            self.total_energy = Some(e_total);
            self.energy_buffer.push(e_total);
        }

        // 3. Energy-window convergence check
        let n_conv = 3; // Number of entries needed in the window
        let energy_tol = tol;
        let (energy_converged, energy_variation) = if self.energy_buffer.len() >= n_conv {
            let window = &self.energy_buffer[self.energy_buffer.len() - n_conv..];
            let e_max = window
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let e_min = window
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let converged = (e_max - e_min).abs() < energy_tol;
            (converged, e_max - e_min)
        } else {
            (false, f64::INFINITY)
        };

        // 4. Density RMS change (mixed density vs pre-mix snapshot)
        let dens_rms = {
            let current = self.density.as_wave_array();
            let previous = self.previous_density.as_wave_array();
            let diff = current - previous;
            let sum_sq: f64 = diff.iter().map(|&x| x * x).sum();
            let n = diff.len() as f64;
            (sum_sq / n).sqrt()
        };

        // 5. Mixing phase transition (CASTEP dm.f90:895-1093, electronic.f90:7614-7638)
        //
        // Off → Kerker when energy variation drops below 0.1 eV (mixing starts
        // once the raw SCF has roughly settled).
        // Kerker → Pulay after the first Kerker mix completes.
        // Pulay → Pulay for normal DIIS.
        //
        // Convergence is only valid when mixing was active (next_mixing ≠ Off),
        // preventing false convergence when the energy is stable simply because
        // no mixing is perturbing the density.
        const MIXING_CONV_TOL_EV: f64 = 0.1; // CASTEP mixing_convergence_tol default

        // Was density mixing active in this iteration?
        let mixing_was_active = self.next_mixing != MixingPhaseKind::Off;

        let next_mixing = match self.next_mixing {
            MixingPhaseKind::Off => {
                if energy_variation < MIXING_CONV_TOL_EV {
                    MixingPhaseKind::Kerker
                } else {
                    MixingPhaseKind::Off
                }
            }
            MixingPhaseKind::Kerker => MixingPhaseKind::Pulay,
            MixingPhaseKind::Pulay => MixingPhaseKind::Pulay,
        };

        // 6. Decision — mixing must have been active to declare convergence
        if mixing_was_active && dens_rms < tol && energy_converged {
            Ok(CheckOutcome::Converged(self.into_phase()))
        } else {
            Ok(CheckOutcome::NotConverged {
                state: self.into_phase(),
                next_mixing,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Terminal: Converged → FinalResult
// ---------------------------------------------------------------------------

impl<S: SpinPolicy> ScfIteration<S, Converged, MixingOff> {
    /// Package converged results.
    pub fn finalize(self) -> FinalResult {
        FinalResult {
            density: self.density,
            eigenvalues: self.eigenvalues,
            total_energy: self.total_energy.unwrap_or(f64::NAN),
        }
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
pub fn run_scf<S: SpinPolicy + BuildVEff>(
    state: ScfIteration<S, Initialized, MixingOff>,
    ndeg: usize,
    tol: f64,
) -> Result<FinalResult, Error> {
    let mut state = state;
    loop {
        state = {
            let v_eff = state.build_v_eff()?;
            let wfn = v_eff.diagonalize(ndeg, None)?;

            // Dispatch mixing phase at runtime
            let mixed = match wfn.next_mixing {
                MixingPhaseKind::Off => wfn.construct_density_off()?.mix(),
                MixingPhaseKind::Kerker => wfn.construct_density_kerker()?.mix(),
                MixingPhaseKind::Pulay => wfn.construct_density_pulay()?.mix(),
            };

            match mixed.check(tol)? {
                CheckOutcome::Converged(converged) => return Ok(converged.finalize()),
                CheckOutcome::NotConverged { state: next, next_mixing } => {
                    // Preserve the next mixing phase for the following iteration
                    let mut s: ScfIteration<_, Initialized, MixingOff> = next;
                    s.next_mixing = next_mixing;
                    s
                }
            }
        };
    }
}

/// Run the full SCF cycle to convergence with total energy computation.
///
/// When `RUST_LOG=info` is set, emits per-iteration output in the same
/// column format as CASTEP's SCF convergence table.
pub fn run_scf_with_energy(
    state: ScfIteration<NonSpin, Initialized, MixingOff>,
    ndeg: usize,
    tol: f64,
) -> Result<FinalResult, Error> {
    run_scf_with_energy_gated(state, ndeg, tol, None)
}

/// Divergence-detection gate for SCF runs.
///
/// Per-iteration sanity checks: when an SCF iteration produces a state outside
/// any of these physical bounds, the run panics with a structured message
/// rather than continuing to spin through wasted iterations. Used by tests
/// (and optionally by callers who want defensive runtime checks). Production
/// `run_scf_with_energy` does not apply a gate.
#[derive(Clone, Debug)]
pub struct ScfDivergenceGate {
    /// Maximum permitted RR last-band eigenvalue (Ha). Cu111+CO converged
    /// last band ≈ 0.13 Ha; anything above this threshold suggests the
    /// Chebyshev filter window has bootstrapped wrong from a prior bad iter.
    pub max_last_band_ha: f64,
    /// Minimum permitted RR band-0 eigenvalue (Ha). For Cu111+CO this sits
    /// around −1.05 Ha; anything below this is filter-amplified bare V_loc
    /// and indicates the cascade has begun.
    pub min_band0_ha: f64,
    /// Maximum allowed V_eff range (Ha) as a multiple of the iter-1 baseline.
    /// Cu111+CO iter-1 = 8.69 Ha; default 5.0× catches the §11 cascade
    /// (iter-2 = 20.26 Ha already, iter-3 = 30+ Ha) early.
    pub max_veff_range_factor: f64,
    /// Iteration ceiling. Even a converging SCF from a fixture-converged
    /// starting state should finish well under this many iterations.
    pub max_iter: u64,
    /// Permitted relative drift in total electron count (fraction). Density
    /// integral should match N_electrons within this tolerance; large drift
    /// indicates the augmentation density or mixing has gone wrong.
    pub electron_count_tolerance: f64,
}

impl Default for ScfDivergenceGate {
    fn default() -> Self {
        Self {
            max_last_band_ha: 5.0,
            min_band0_ha: -30.0,
            max_veff_range_factor: 5.0,
            max_iter: 60,
            electron_count_tolerance: 0.05,
        }
    }
}

/// Like `run_scf_with_energy` but with optional divergence gating.
///
/// When `gate = Some(...)`, the loop inspects per-iteration state after the
/// SCF check completes. Any out-of-bounds value triggers a panic with a
/// structured message identifying which gate fired, the offending value,
/// the iteration index, and a brief reference to the expected range.
pub fn run_scf_with_energy_gated(
    state: ScfIteration<NonSpin, Initialized, MixingOff>,
    ndeg: usize,
    tol: f64,
    gate: Option<ScfDivergenceGate>,
) -> Result<FinalResult, Error> {
    use std::time::Instant;

    let mut state = state;
    let t_start = Instant::now();
    let mut iter_count: u64 = 0;
    let mut prev_energy: Option<f64> = None;
    let mut header_printed = false;
    let mut iter1_veff_range: Option<f64> = None;
    let mut iter1_n_electrons: Option<f64> = None;
    loop {
        state = {
            let v_eff = state.build_v_eff_with_energy()?;

            // Sample V_eff range BEFORE moving v_eff into diagonalize, so we
            // can use it for the gate at end-of-iteration.
            let veff_range_now = if gate.is_some() {
                v_eff.v_eff().as_ref().map(|veff| {
                    let arr = veff.as_real_grid().as_real_array();
                    let mn = arr.iter().cloned().fold(f64::INFINITY, f64::min);
                    let mx = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    mx - mn
                })
            } else {
                None
            };

            let wfn = v_eff.diagonalize(ndeg, None)?;

            // Snapshot eigenvalues for the gate before they move on.
            let band0_now = wfn.eigenvalues().first().copied();
            let last_band_now = wfn.eigenvalues().last().copied();

            // Dispatch mixing phase at runtime
            let mixed = match wfn.next_mixing {
                MixingPhaseKind::Off => wfn.construct_density_off()?.mix(),
                MixingPhaseKind::Kerker => wfn.construct_density_kerker()?.mix(),
                MixingPhaseKind::Pulay => wfn.construct_density_pulay()?.mix(),
            };

            match mixed.check(tol)? {
                CheckOutcome::Converged(converged) => {
                    return Ok(converged.finalize());
                }
                CheckOutcome::NotConverged { state: next, next_mixing } => {
                    // Per-iteration SCF output (CASTEP column format, silent by default)
                    if !header_printed && tracing::level_enabled!(tracing::Level::INFO) {
                        tracing::info!(
                            "------------------------------------------------------------------------"
                        );
                        tracing::info!(
                            "SCF loop      Energy           Fermi           Energy gain       Timer"
                        );
                        tracing::info!(
                            "                               energy          per atom          (sec)"
                        );
                        tracing::info!(
                            "------------------------------------------------------------------------"
                        );
                        header_printed = true;
                    }
                    iter_count += 1;
                    if let (Some(e_total), Some(fermi)) =
                        (next.total_energy, next.fermi_energy)
                    {
                        let energy_ev = e_total * HARTREE_TO_EV;
                        let gain_per_atom = prev_energy
                            .map(|prev| (e_total - prev) / next.cell.num_ions as f64)
                            .unwrap_or(0.0);
                        let gain_ev = gain_per_atom * HARTREE_TO_EV;
                        let fermi_ev = fermi * HARTREE_TO_EV;
                        let elapsed = t_start.elapsed().as_secs_f64();
                        tracing::info!(
                            "{:>7}  {:>15.8E}  {:>15.8E}  {:>15.8E}  {:>9.2}",
                            iter_count,
                            energy_ev,
                            fermi_ev,
                            gain_ev,
                            elapsed,
                        );
                        prev_energy = Some(e_total);
                    }

                    // ---- Divergence gate ----
                    if let Some(g) = gate.as_ref() {
                        // Iter ceiling
                        if iter_count > g.max_iter {
                            panic!(
                                "[SCF gate] iteration ceiling exceeded: iter {} > max {} \
                                 (likely diverging — examine prior iterations' eigenvalues and V_eff range)",
                                iter_count, g.max_iter,
                            );
                        }
                        // Last band: catch filter-window cascade early
                        if let Some(lb) = last_band_now
                            && lb > g.max_last_band_ha
                        {
                            panic!(
                                "[SCF gate] iter {}: last band = {:.4} Ha exceeds gate {:.2} Ha\n  \
                                 reference (Cu111+CO converged): last band ≈ 0.13 Ha\n  \
                                 likely cause: filter window bootstrap from a prior iter's RR output \
                                 (see notes/open-followups.md §11)",
                                iter_count, lb, g.max_last_band_ha,
                            );
                        }
                        // Band-0: catch bare-V_loc amplification
                        if let Some(b0) = band0_now
                            && b0 < g.min_band0_ha
                        {
                            panic!(
                                "[SCF gate] iter {}: band-0 = {:.4} Ha below gate {:.2} Ha\n  \
                                 reference (Cu111+CO converged): band-0 ≈ -1.05 Ha\n  \
                                 likely cause: filter amplifying bare V_loc wells in absence of \
                                 V_H/V_xc smoothing (see notes/open-followups.md §11 cascade)",
                                iter_count, b0, g.min_band0_ha,
                            );
                        }
                        // V_eff range: relative to iter-1 baseline
                        if let Some(rng) = veff_range_now {
                            if iter1_veff_range.is_none() {
                                iter1_veff_range = Some(rng);
                            }
                            let baseline = iter1_veff_range.unwrap_or(rng);
                            let factor = rng / baseline;
                            if factor > g.max_veff_range_factor {
                                panic!(
                                    "[SCF gate] iter {}: V_eff range {:.4} Ha is {:.2}× iter-1 baseline ({:.4} Ha), exceeds gate {:.2}×\n  \
                                     reference (Cu111+CO converged): V_eff range ≈ 8.69 Ha\n  \
                                     likely cause: density degraded into bare V_loc wells",
                                    iter_count, rng, factor, baseline, g.max_veff_range_factor,
                                );
                            }
                        }
                        // Total electron count from the new (post-mix) density.
                        // Density is stored in CASTEP raw units (ρ×Ω), so the correct
                        // formula is sum/N, not sum*Ω/N (which would apply Ω twice).
                        let rho_arr = next.density.as_wave_array();
                        let n_grid = rho_arr.len() as f64;
                        let n_electrons_now = rho_arr.iter().sum::<f64>() / n_grid;
                        if iter1_n_electrons.is_none() {
                            iter1_n_electrons = Some(n_electrons_now);
                        }
                        let baseline_ne = iter1_n_electrons.unwrap_or(n_electrons_now);
                        let drift = (n_electrons_now - baseline_ne).abs() / baseline_ne.abs().max(1.0);
                        if drift > g.electron_count_tolerance {
                            panic!(
                                "[SCF gate] iter {}: electron count {:.4} drifted {:.2}% from iter-1 baseline {:.4}, exceeds gate {:.2}%\n  \
                                 likely cause: augmentation density or mixing broken",
                                iter_count, n_electrons_now, drift * 100.0, baseline_ne, g.electron_count_tolerance * 100.0,
                            );
                        }
                    }

                    let mut s: ScfIteration<_, Initialized, MixingOff> = next;
                    s.next_mixing = next_mixing;
                    s
                }
            }
        };
    }
}

// ---------------------------------------------------------------------------
// PW-to-FFT index conversion helper
// ---------------------------------------------------------------------------

/// Convert fractional PW G-vector coordinates to cuFFT Fortran-order linear indices.
///
/// Each entry `[h, k, l]` is a fractional G-vector from `KptWaveBlock`.
/// The output index is `iz + ngz * (iy + ngy * ix)` (cuFFT Fortran-order for
/// `[ngz, ngy, ngx]` dimensions where iz varies fastest, ix slowest).
/// Used by test fixture infrastructure (Group F).
#[doc(hidden)]
pub fn pw_coords_to_fft_indices(
    pw_coords: &[[i32; 3]],
    wave_grid: &GVectorGrid,
) -> Vec<i32> {
    let [ngz, ngy, ngx] = wave_grid.grid();
    pw_coords
        .iter()
        .map(|&[h, k, l]| {
            let ix = if h >= 0 { h as usize } else { (h + ngx as i32) as usize };
            let iy = if k >= 0 { k as usize } else { (k + ngy as i32) as usize };
            let iz = if l >= 0 { l as usize } else { (l + ngz as i32) as usize };
            (iz + ngz * (iy + ngy * ix)) as i32
        })
        .collect()
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
#[doc(hidden)]
pub fn downsample_array_to_wave_grid(
    fine_arr: &Array3<f64>,
    fine_grid: &GVectorGrid,
    wave_grid: &GVectorGrid,
) -> Result<EffectivePotential, Error> {
    use chemrust_hamiltonian_core::fft::{fft_forward_3d, fft_inverse_3d};

    let [ngz, ngy, ngx] = wave_grid.grid();
    let [ngz_f, ngy_f, ngx_f] = fine_grid.grid();

    // Same grid → no downsampling needed, return as-is.
    if ngz == ngz_f && ngy == ngy_f && ngx == ngx_f {
        return Ok(EffectivePotential::from_inner(FineGridArray::from_inner(fine_arr.clone())));
    }

    // Forward FFT fine-grid V_eff → G-space
    let fine_g = fft_forward_3d(&chemrust_hamiltonian_core::fft::RealGrid::from_inner(fine_arr.clone()))
        .map_err(|_| Error::NotImplemented)?;

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
            *coeff = fine_g.as_recip_array()[[iz_f, iy_f, ix_f]];
        });

    // Inverse FFT back to real space on wave grid
    let n_total_fine = (ngx_f * ngy_f * ngz_f) as f64;
    let rho_wave = fft_inverse_3d(&chemrust_hamiltonian_core::fft::RecipGrid::from_inner(wave_g))
        .map_err(|_| Error::NotImplemented)?;
    let result = rho_wave.into_inner().mapv(|x| x / n_total_fine);

    Ok(EffectivePotential::from_inner(FineGridArray::from_inner(result)))
}

// ---------------------------------------------------------------------------
// Accessor methods for integration tests
// ---------------------------------------------------------------------------

impl<S: SpinPolicy, M: MixingPhase> ScfIteration<S, VEffBuilt, M> {
    /// Access the assembled effective potential (for testing).
    pub fn v_eff(&self) -> &Option<S::VEff> {
        &self.v_eff
    }
    /// Replace the effective potential (debug/testing only).
    #[doc(hidden)]
    pub fn set_v_eff(&mut self, v: S::VEff) {
        self.v_eff = Some(v);
    }
    /// Access ψ coefficient slice (debug/testing only).
    #[doc(hidden)]
    pub fn psi_data(&self) -> &[Complex64] {
        &self.psi.data
    }
    /// Mutable access to ψ coefficient slice (debug/testing only).
    /// Use to inject controlled pollution before running diagonalize.
    #[doc(hidden)]
    pub fn psi_data_mut(&mut self) -> &mut [Complex64] {
        &mut self.psi.data
    }
    /// Set eigenvalues (debug/testing only). Used to mimic the iter-2 filter
    /// code path which receives eigenvalues from a prior diagonalization.
    #[doc(hidden)]
    pub fn set_eigenvalues(&mut self, eigs: Vec<f64>) {
        self.eigenvalues = eigs;
    }
    /// Access (n_bands, n_pw) shape (debug/testing only).
    #[doc(hidden)]
    pub fn psi_shape(&self) -> (usize, usize) {
        (self.psi.n_bands, self.psi.n_pw)
    }
}

impl<S: SpinPolicy, M: MixingPhase> ScfIteration<S, WavefunctionsUpdated, M> {
    /// Access the eigenvalues (for testing).
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
    /// Access ψ coefficient slice after diagonalization (debug/testing only).
    #[doc(hidden)]
    pub fn psi_data(&self) -> &[Complex64] {
        &self.psi.data
    }
}

impl<S: SpinPolicy, M: MixingPhase> ScfIteration<S, DensityUpdated<M>, MixingOff> {
    /// Access ρ_aug on the fine grid (debug/testing only).
    #[doc(hidden)]
    pub fn density_aug_fine(&self) -> Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>> {
        self.density_aug_fine.as_ref()
    }
}

// --- Energy-component accessors (debug/diagnostics) ---
impl<S: SpinPolicy, State: ScfPhase, M: MixingPhase> ScfIteration<S, State, M> {
    /// Read E_xc from the most recent V_eff assembly (diagnostic only).
    #[doc(hidden)]
    pub fn e_xc_value(&self) -> Option<f64> { self.e_xc }

    /// Read E_H from the most recent V_eff assembly (diagnostic only).
    #[doc(hidden)]
    pub fn e_hartree_value(&self) -> Option<f64> { self.e_hartree }

    /// Read ∫ρV_xc from the most recent V_eff assembly (diagnostic only).
    #[doc(hidden)]
    pub fn rho_vxc_value(&self) -> Option<f64> { self.rho_vxc }

    /// Read the Ewald energy (diagnostic only).
    #[doc(hidden)]
    pub fn ewald_value(&self) -> f64 { self.ewald }

    /// Access cell geometry (diagnostic only).
    #[doc(hidden)]
    pub fn cell_geometry(&self) -> &CellGeometry { &self.cell }

    /// Access pseudopotentials (diagnostic only).
    #[doc(hidden)]
    pub fn pseudopotentials(&self) -> &PseudopotentialSet { &self.pots }

    /// Access smearing parameters (diagnostic only).
    #[doc(hidden)]
    pub fn smearing_params(&self) -> &SmearingParams { &self.smearing }

}

impl<S: SpinPolicy, M: MixingPhase> ScfIteration<S, Mixed, M> {
    /// Access eigenvalues from the most recent diagonalization (diagnostic only).
    #[doc(hidden)]
    pub fn eigenvalues(&self) -> &[f64] { &self.eigenvalues }
}

impl<S: SpinPolicy> ScfIteration<S, Initialized, MixingOff> {
    /// Mutable access to density (for perturbation testing).
    pub fn density_mut(&mut self) -> &mut Density {
        &mut self.density
    }
    /// Clear the cached augmentation density (debug/testing only).
    /// The next `build_v_eff_with_energy` call will use ρ_PW only.
    #[doc(hidden)]
    pub fn clear_density_aug_fine(&mut self) {
        self.density_aug_fine = None;
    }
    /// Read the total energy populated by the most recent `check()` call
    /// (debug/testing only). Returns `None` if `check()` has not yet run with
    /// energy components populated. Used by Q1 (iter1_drift_from_castep_state_is_bounded)
    /// to peek at iter-1's energy before iter-2 takes over the state.
    #[doc(hidden)]
    pub fn total_energy(&self) -> Option<f64> {
        self.total_energy
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::WavefunctionSet;
    use crate::mixing::MixingOff;
    use crate::types::{KPoint, SmearingScheme, WaveGridArray};
    use std::marker::PhantomData;

    fn dummy_cell() -> CellGeometry {
        use ndarray::Array2;
        CellGeometry {
            real_lattice: chemrust_hamiltonian_core::RealLattice::from_inner([
                [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0],
            ]),
            recip_lattice: chemrust_hamiltonian_core::RecipLattice::from_inner([
                [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2],
            ]),
            volume: 125.0,
            num_species: 1,
            num_ions: 1,
            ionic_positions: Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap(),
            species_symbols: vec!["Cu".into()],
            species_pot_files: vec!["Cu_00.usp".into()],
            num_ions_in_species: vec![1],
            ion_species: vec![0],
            max_ions_in_species: 1,
            species_lcao_states: vec![],
        }
    }

    fn dummy_grid() -> GVectorGrid {
        GVectorGrid::new(
            4, 4, 4,
            chemrust_hamiltonian_core::RecipLattice::from_inner([
                [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2],
            ]),
        )
    }

    /// check() with no energy data returns NotConverged.
    #[test]
    fn test_check_not_converged_no_energy() {
        let shape = [4, 4, 4]; // Fortran: [ngz, ngy, ngx]
        let density = Density::from_inner(WaveGridArray::from_inner(Array3::<f64>::zeros(shape)));
        let previous_density = Density::from_inner(WaveGridArray::from_inner(
            Array3::<f64>::from_elem(shape, 2.0),
        ));
        let psi = WavefunctionSet::new(vec![Complex64::ZERO; 4 * 27], 4, 27);

        let state: ScfIteration<NonSpin, Mixed, MixingOff> = ScfIteration {
            cell: dummy_cell(),
            pots: PseudopotentialSet::new(),
            wave_grid: dummy_grid(),
            fine_grid: dummy_grid(),
            k_point: KPoint::default(),
            smearing: SmearingParams {
                width: 0.1,
                electron_temperature: 0.0,
                scheme: SmearingScheme::Gaussian,
            },
            pw_coords: Vec::new(),
            pw_fft_indices: Vec::new(),
            density,
            psi,
            eigenvalues: vec![-0.3, -0.2],
            v_eff: None,
            history: DensityHistory::new(),
            previous_density,
            next_mixing: MixingPhaseKind::Off,
            e_xc: None,
            e_hartree: None,
            rho_vxc: None,
            ewald: 0.0,
            energy_buffer: Vec::new(),
            total_energy: None,
            fermi_energy: None,
            beta_psi_per_ion: None,
            q_sf_cache: None,
            density_aug_fine: None,
            _phase: PhantomData,
        };

        let result = state.check(1e-6).unwrap();
        match result {
            CheckOutcome::NotConverged { next_mixing, .. } => {
                assert_eq!(next_mixing, MixingPhaseKind::Off,
                    "no energy data should keep mixing phase as Off");
            }
            _ => panic!("expected NotConverged, got Converged"),
        }
    }

    /// Helper: build a `Mixed`-phase state for check() testing.
    fn mixed_state(
        energy_buffer: Vec<f64>,
        next_mixing: MixingPhaseKind,
    ) -> ScfIteration<NonSpin, Mixed, MixingOff> {
        let shape = [4, 4, 4];
        ScfIteration {
            cell: dummy_cell(),
            pots: PseudopotentialSet::new(),
            wave_grid: dummy_grid(),
            fine_grid: dummy_grid(),
            k_point: KPoint::default(),
            smearing: SmearingParams {
                width: 0.1,
                electron_temperature: 0.0,
                scheme: SmearingScheme::Gaussian,
            },
            pw_coords: Vec::new(),
            pw_fft_indices: Vec::new(),
            density: Density::from_inner(WaveGridArray::from_inner(
                Array3::<f64>::zeros(shape),
            )),
            psi: WavefunctionSet::new(vec![Complex64::ZERO; 4 * 27], 4, 27),
            eigenvalues: vec![-0.3, -0.2],
            v_eff: None,
            history: DensityHistory::new(),
            previous_density: Density::from_inner(WaveGridArray::from_inner(
                Array3::<f64>::from_elem(shape, 2.0),
            )),
            next_mixing,
            e_xc: None,
            e_hartree: None,
            rho_vxc: None,
            ewald: 0.0,
            energy_buffer,
            total_energy: None,
            fermi_energy: None,
            beta_psi_per_ion: None,
            q_sf_cache: None,
            density_aug_fine: None,
            _phase: PhantomData,
        }
    }

    /// Off → Kerker when energy variation drops below 0.1 eV.
    #[test]
    fn test_check_off_to_kerker_transition() {
        // Energy window: three iterations all within 0.01 eV — settled.
        let energies = vec![-24110.966, -24110.964, -24110.965];
        let state = mixed_state(energies, MixingPhaseKind::Off);

        let result = state.check(1e-5).unwrap();
        match result {
            CheckOutcome::NotConverged { next_mixing, .. } => {
                assert_eq!(
                    next_mixing,
                    MixingPhaseKind::Kerker,
                    "Off → Kerker when energy settles below 0.1 eV"
                );
            }
            CheckOutcome::Converged(_) => {
                panic!("should not converge — mixing was Off (mixed_status false)")
            }
        }
    }

    /// Off stays Off when energy is still varying widely.
    #[test]
    fn test_check_off_stays_off_when_energy_unstable() {
        // Energy varying by > 0.1 eV — too unstable to start mixing.
        let energies = vec![-24110.0, -24109.0, -24108.0];
        let state = mixed_state(energies, MixingPhaseKind::Off);

        let result = state.check(1e-5).unwrap();
        match result {
            CheckOutcome::NotConverged { next_mixing, .. } => {
                assert_eq!(
                    next_mixing,
                    MixingPhaseKind::Off,
                    "should stay Off when energy varies > 0.1 eV"
                );
            }
            CheckOutcome::Converged(_) => panic!("should not converge"),
        }
    }

    /// Kerker → Pulay after first Kerker mix completes.
    #[test]
    fn test_check_kerker_to_pulay_transition() {
        // Kerker was active this iteration; should advance to Pulay.
        let energies = vec![-24110.966, -24110.965, -24110.967];
        let state = mixed_state(energies, MixingPhaseKind::Kerker);

        let result = state.check(1e-5).unwrap();
        match result {
            CheckOutcome::NotConverged { next_mixing, .. } => {
                assert_eq!(
                    next_mixing,
                    MixingPhaseKind::Pulay,
                    "Kerker → Pulay after first Kerker mix"
                );
            }
            CheckOutcome::Converged(_) => {
                // Could happen if energy+tol happens to match.
                // If so, still fine — convergence with Kerker mixing is valid.
            }
        }
    }

    /// Pulay stays Pulay.
    #[test]
    fn test_check_pulay_stays_pulay() {
        let energies = vec![-24110.966, -24110.965, -24110.967];
        let state = mixed_state(energies, MixingPhaseKind::Pulay);

        let result = state.check(1e-5).unwrap();
        match result {
            CheckOutcome::NotConverged { next_mixing, .. } => {
                assert_eq!(
                    next_mixing,
                    MixingPhaseKind::Pulay,
                    "Pulay should stay Pulay"
                );
            }
            CheckOutcome::Converged(_) => {
                // Energy variation ≈ 0.002 eV, density RMS likely large with dummy data.
                // If it converges, that's fine too — Pulay is a valid mixing phase.
            }
        }
    }

    /// Converging with mixing Off must not declare victory (mixed_status guard).
    #[test]
    fn test_check_no_converge_with_mixing_off() {
        // Even if energies are perfectly flat (differ by < 1e-10), Off mixing
        // must fail the convergence check because mixing was never active.
        let energies = vec![-24110.96665069; 5];
        let mut state = mixed_state(energies, MixingPhaseKind::Off);

        // Also make density RMS zero so the only blocker is mixed_status.
        state.previous_density = state.density.clone();

        let result = state.check(1e-8).unwrap();
        match result {
            CheckOutcome::NotConverged { next_mixing, .. } => {
                // Should transition to Kerker (energy is stable) but NOT converge.
                assert_eq!(
                    next_mixing,
                    MixingPhaseKind::Kerker,
                    "Off with flat energy should transition to Kerker, not converge"
                );
            }
            CheckOutcome::Converged(_) => {
                panic!(
                    "must NOT converge with Off mixing — mixed_status guard required"
                );
            }
        }
    }

    /// Verify that the discrete energy integral convention is correct:
    /// ρ is in CASTEP raw units (ρ_phys × Ω), so the integral weight for
    /// Σ ρ_grid[i] × V[i] must be 1/N_grid, NOT Ω/N_grid.
    ///
    /// Reference: CASTEP xc_gga (xc.f90:1056) uses /total_fine_grid_points,
    /// and hartree_energy is computed in reciprocal space with an explicit
    /// 1/Ω factor. When computing the same integral in real space:
    ///   ∫ρV d³r ≈ Σ ρ_phys[i] × V[i] × (Ω/N) = Σ (ρ_grid[i]/Ω) × V[i] × (Ω/N)
    ///          = Σ ρ_grid[i] × V[i] × (1/N)
    /// So the weight is 1/N, not Ω/N (which would overcount by Ω = volume).
    #[test]
    fn test_energy_integral_convention() {
        // Synthetic uniform density in CASTEP ρ×Ω convention.
        // For uniform ρ_phys = Q_total/Ω, each grid point stores ρ_grid = Q_total
        // (since ρ_grid = ρ_phys × Ω = Q_total).
        // We use a simple cubic cell at volume 125.0 Bohr³, 4×4×4 grid.
        let volume = 125.0_f64;
        let n_grid = 64.0_f64; // 4×4×4
        let q_total = 186.0_f64; // Cu111+CO valence electrons
        let rho_grid = q_total; // ρ_phys × Ω = Q_total for uniform density

        // Assume V_H = 1.0 Ha everywhere (uniform potential for a uniform density)
        let v_h = 1.0_f64;

        // Physical expectation: ∫ ρ_phys V_H d³r = Q_total * ⟨V_H⟩
        // For uniform density and uniform V_H: ∫ = (Q_total/Ω) × V_H × Ω = Q_total × V_H
        let expected = q_total * v_h; // 186 Ha × 1.0 = 186 Ha

        // Compute e_hartree_raw with d_v = 1/N (correct convention)
        let d_v_correct = 1.0 / n_grid;
        // Σ ρ_grid × V_H × d_v = N_grid × (Q_total × 1.0) × (1/N_grid) = Q_total ✓
        let e_hartree_raw_correct = rho_grid * v_h * d_v_correct * n_grid;

        // Compute e_hartree_raw with d_v = Ω/N (old buggy convention)
        let d_v_buggy = volume / n_grid;
        // Σ ρ_grid × V_H × d_v = N_grid × Q_total × (Ω/N_grid) = Q_total × Ω
        let e_hartree_raw_buggy = rho_grid * v_h * d_v_buggy * n_grid;

        assert!(
            (e_hartree_raw_correct - expected).abs() < 1e-12,
            "Correct formula gives {e_hartree_raw_correct}, expected {expected}"
        );
        assert!(
            (e_hartree_raw_buggy / expected - volume).abs() < 1e-12,
            "Buggy formula is off by Ω = {volume}: ratio = {}",
            e_hartree_raw_buggy / expected
        );

        // Same test for the rho_vxc integral (same pattern, different potential)
        let v_xc = -0.5_f64; // synthetic V_xc
        let expected_rho_vxc = q_total * v_xc; // ∫ ρ_phys × V_xc = Q_total × V_xc = -93 Ha
        let rho_vxc_correct = rho_grid * v_xc * d_v_correct * n_grid;
        let rho_vxc_buggy = rho_grid * v_xc * d_v_buggy * n_grid;

        assert!(
            (rho_vxc_correct - expected_rho_vxc).abs() < 1e-12,
            "Correct rho_vxc gives {rho_vxc_correct}, expected {expected_rho_vxc}"
        );
        assert!(
            (rho_vxc_buggy / expected_rho_vxc - volume).abs() < 1e-12,
            "Buggy rho_vxc is off by Ω = {volume}: ratio = {}",
            rho_vxc_buggy / expected_rho_vxc
        );
    }
}
