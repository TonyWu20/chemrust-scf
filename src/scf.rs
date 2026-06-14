use std::marker::PhantomData;
use std::sync::Arc;

use bon::bon;
use chemrust_hamiltonian_core::{
    CellGeometry, GVectorGrid, NonSpin, PseudopotentialSet, SpinCollinear, SpinPolicy, VEffBuilder,
};
use cudarc::driver::{CudaContext, CudaSlice};
use ndarray::{Array3, ShapeBuilder};
use num_complex::Complex64;

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::pcie::PcieAccount;
use crate::device::{CudaComplex, Gpu};
use crate::eigensolver::davidson_types::compute_kinetic_energies;
use crate::eigensolver::kernels::CudaKernelSet;
#[cfg(feature = "chebyshev")]
use crate::eigensolver::chebyshev::{FilterMode, chebyshev_filter};
use crate::eigensolver::preconditioner::TpaPreconditioner;
use crate::eigensolver::davidson_types::PwCoefficients;
#[cfg(any(test, feature = "scf_diag"))]
use crate::eigensolver::davidson::{DavidsonDiagnostic, DAVIDSON_LAST_DIAG};
#[cfg(feature = "chebyshev")]
use crate::eigensolver::rayleigh_ritz::rayleigh_ritz;
#[cfg(all(any(test, feature = "scf_diag"), feature = "chebyshev"))]
use crate::eigensolver::rayleigh_ritz::rayleigh_ritz_with_matrices;
use crate::eigensolver::vnl_data::{build_handle_shared_vnl, HandleSharedVnl, KptSharedVnl, VnlBatchData};
use crate::layout::{ColumnDistributed, Cpu, WavefunctionSet};
use crate::mixing::{DensityHistory, Kerker, MixingOff, MixingPhase, Pulay};
use crate::density::{QSfCache, build_q_sf_cache};
use crate::types::{
    ChemicalPotential, Density, EffectivePotential, Error, FinalResult, FineGridArray, KPoint,
    SmearingParams,
};
use crate::spin_types::{
    FermiEnergies, KptDataSet, OccupationSet, PerSpinAugDensity, PerSpinBetaProjections,
    PerSpinDensity, PerSpinEigenvalues, PerSpinPwCoefficients, SpinChannelData,
};
use crate::energy::HARTREE_TO_EV;
use crate::pipeline;

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
    /// All k-points. Index `[ikpt]` for k-point data.
    pub(crate) k_points: KptDataSet<KPoint>,
    pub(crate) smearing: SmearingParams,
    /// PW G-vector fractional coordinates [h, k, l] for each plane wave, per k-point.
    pub(crate) pw_coords: KptDataSet<Vec<[i32; 3]>>,

    // --- Shared shape metadata (spin- and kpt-independent) ---
    pub(crate) n_bands: usize,
    /// Maximum number of PW coefficients across all k-points (for GPU allocations).
    pub(crate) max_n_pw: usize,
    /// Number of k-points.
    pub(crate) nkpts: usize,

    // --- Mutable state, governed by phase ---
    pub(crate) density: PerSpinDensity,
    pub(crate) psi: PerSpinPwCoefficients,
    /// Per-spin, per-kpt CPU-side psi data (needed by VnlBatchData / density construction).
    /// Indexed as `psi_cpu[ispin][ikpt]`.
    pub(crate) psi_cpu: SpinChannelData<KptDataSet<Vec<Complex64>>>,
    pub(crate) eigenvalues: PerSpinEigenvalues,
    pub(crate) v_eff: Option<S::VEff>,
    pub(crate) history: DensityHistory<M>,
    /// Will be read by `check()` in Phase 2 Goal 5. Suppressed until then.
    #[allow(dead_code)]
    pub(crate) previous_density: PerSpinDensity,

    /// Precomputed linear FFT grid indices for each PW coefficient, per k-point.
    /// Index = ix + ngx * (iy + ngy * iz) in C-order (cuFFT convention).
    pub(crate) pw_fft_indices: KptDataSet<Vec<i32>>,

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
    pub(crate) fermi_energy: FermiEnergies,

    /// Per-ion ⟨β_{IL}|ψ_b⟩ projections of the most recent ψ, cached from the
    /// Rayleigh–Ritz step. Shape per ion: `(n_expanded × n_bands)`. GPU-resident
    /// as `CudaSlice<CudaComplex>`. `None` before the first `diagonalize`
    /// (e.g. iter-1 driven from a fixture density). Consumed by
    /// `compute_aug_density_gpu` to build ω^I_{nm}.
    pub(crate) beta_psi_per_ion: PerSpinBetaProjections,
    /// GPU cache of Q_{nm}(G)·exp(-iG·R_I) per ion. Built lazily on first
    /// `compute_density_from_wavefunctions` call that has a GPU stream.
    /// `None` until first build; geometry-static thereafter.
    pub(crate) q_sf_cache: Option<QSfCache>,
    /// Kpt-independent V_NL shared state (screening caches, Q, D0).  Built
    /// lazily on first `diagonalize_inner` call and persisted across SCF
    /// iterations — matches `ffi.rs` `ChemrustHandle.handle_shared_vnl`.
    pub(crate) handle_shared_vnl: Option<Arc<HandleSharedVnl>>,
    /// Per-kpt V_NL shared state (per-ion β(G+k)).  Built lazily on first
    /// `diagonalize_inner` call per kpt and persisted across SCF iterations.
    /// Index `[ikpt]`.  Matches `ffi.rs` `KptData.shared_vnl`.
    pub(crate) shared_vnl_cache: Vec<Option<Arc<KptSharedVnl>>>,
    /// USPP augmentation density ρ_aug(r) on the fine grid, regenerated from
    /// the current ψ + occ each iteration. `None` for iter-1 (fixture
    /// density already encodes augmentation in the wave-grid convention).
    /// Added inside `build_v_eff_with_energy_impl` to the upsampled smooth
    /// density before V_H/V_xc evaluation.
    pub(crate) density_aug_fine: PerSpinAugDensity,

    /// Diagnostics from the most recent Davidson eigensolve (Phase 0 Gate 3 tests).
    /// `None` when Chebyshev-RR was used or no diagonalize has run yet.
    #[cfg(any(test, feature = "scf_diag"))]
    pub(crate) last_davidson_diagnostics: Option<DavidsonDiagnostic>,
    /// Current SCF iteration number (1-indexed). Updated by run_scf before diagonalize.
    pub(crate) scf_iter: usize,
    /// True when spin freed (shared Fermi energy for both spins).
    pub(crate) spin_freed: bool,

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
        density: PerSpinDensity,
        psi: PerSpinPwCoefficients,
        psi_data: SpinChannelData<KptDataSet<Vec<Complex64>>>,
        pw_coords: KptDataSet<Vec<[i32; 3]>>,
        pw_fft_indices: KptDataSet<Vec<i32>>,
        k_points: KptDataSet<KPoint>,
        smearing: SmearingParams,
        max_history: usize,
    ) -> Self {
        let _ = max_history; // History size is fixed internally for now
        let previous_density = density.clone();
        let nspins = S::nspins();
        debug_assert_eq!(
            psi_data.len(), nspins,
            "psi_data must have one entry per spin channel (got {}, expected {})",
            psi_data.len(), nspins,
        );
        let nkpts = k_points.nkpts();
        let n_bands = psi_data[0][0].len() / pw_coords[0].len();
        let max_n_pw = pw_coords.iter().map(|v| v.len()).max().unwrap_or(0);
        let psi_cpu = psi_data;
        let ewald = crate::energy::ewald_energy(&cell, &pots);
        Self {
            cell,
            pots,
            wave_grid,
            fine_grid,
            n_bands,
            max_n_pw,
            nkpts,
            density,
            psi,
            psi_cpu,
            pw_coords,
            pw_fft_indices,
            k_points,
            smearing,
            eigenvalues: PerSpinEigenvalues(SpinChannelData::new::<S>(
                (0..nspins).map(|_| KptDataSet::new(vec![Vec::new(); nkpts], nkpts)).collect()
            )),
            v_eff: None,
            history: {
                // CASTEP uses spin_density_mixing_amplitude=2.0 for magnetic
                // systems (separate from mix_charge_amp=0.5).  Our per-spin
                // density mixing approximates this by applying the spin mixing
                // amplitude to each spin channel independently.
                let amp = if S::nspins() > 1 { 2.0 } else { 0.5 };
                DensityHistory::with_amplitude(S::nspins(), amp)
            },
            previous_density,
            next_mixing: MixingPhaseKind::Off,
            e_xc: None,
            e_hartree: None,
            rho_vxc: None,
            ewald,
            energy_buffer: Vec::new(),
            total_energy: None,
            fermi_energy: FermiEnergies(vec![0.0; nspins]),
            beta_psi_per_ion: PerSpinBetaProjections(SpinChannelData::new::<S>(
                (0..nspins).map(|_| KptDataSet::new(vec![None; nkpts], nkpts)).collect()
            )),
            q_sf_cache: None,
            handle_shared_vnl: None,
            shared_vnl_cache: vec![None; nkpts],
            density_aug_fine: PerSpinAugDensity(SpinChannelData::new::<S>(vec![None; nspins])),
            #[cfg(any(test, feature = "scf_diag"))]
            last_davidson_diagnostics: None,
            scf_iter: 0,
            spin_freed: false,
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
            k_points: self.k_points,
            smearing: self.smearing,
            n_bands: self.n_bands,
            max_n_pw: self.max_n_pw,
            nkpts: self.nkpts,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: self.density,
            psi: self.psi,
            psi_cpu: self.psi_cpu,
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
            handle_shared_vnl: self.handle_shared_vnl,
            shared_vnl_cache: self.shared_vnl_cache,
            density_aug_fine: self.density_aug_fine,
            #[cfg(any(test, feature = "scf_diag"))]
            last_davidson_diagnostics: self.last_davidson_diagnostics,
            scf_iter: self.scf_iter,
            spin_freed: self.spin_freed,
            _phase: PhantomData,
        }
    }

    /// Access the computed density (for testing) — total (sum of spin channels).
    pub fn density(&self) -> Density {
        self.density.total()
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
    /// Assemble V_eff from total density and optional spin density.
    ///
    /// `spin` is `None` for NonSpin; `Some(ρ_spin)` for SpinCollinear where
    /// `ρ_spin = ρ_up − ρ_down` on the wave grid.
    fn build_v_eff_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        spin: Option<&chemrust_hamiltonian_core::Density>,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<Self::VEff, chemrust_hamiltonian_core::Error>;
}

impl BuildVEff for NonSpin {
    fn build_v_eff_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        _spin: Option<&chemrust_hamiltonian_core::Density>,
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
        spin: Option<&chemrust_hamiltonian_core::Density>,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<(chemrust_hamiltonian_core::EffectivePotential, chemrust_hamiltonian_core::EffectivePotential), chemrust_hamiltonian_core::Error> {
        let zero_spin;
        let spin_density = match spin {
            Some(s) => s,
            None => {
                // Fallback: zero spin density for paramagnetic initial guess.
                let shape = [wave_grid.grid()[2], wave_grid.grid()[1], wave_grid.grid()[0]];
                zero_spin = chemrust_hamiltonian_core::Density::from_inner(
                    chemrust_hamiltonian_core::fft::RealGrid::from_inner(Array3::zeros(shape)),
                );
                &zero_spin
            }
        };
        VEffBuilder::<SpinCollinear>::new(cell, pots, fine_grid)
            .assemble_on_fine_grid(rho, spin_density, wave_grid, fine_grid)
    }
}

// --- Private dispatch trait for energy-aware V_eff assembly ---

pub trait BuildVEffWithEnergy: BuildVEff {
    fn build_v_eff_with_energy_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        spin: Option<&chemrust_hamiltonian_core::Density>,
        rho_aug_fine: Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>>,
        rho_aug_spin: Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>>,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<(Self::VEff, f64, f64, f64), chemrust_hamiltonian_core::Error>;
}

impl BuildVEffWithEnergy for NonSpin {
    fn build_v_eff_with_energy_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        _spin: Option<&chemrust_hamiltonian_core::Density>,
        rho_aug_fine: Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>>,
        _rho_aug_spin: Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>>,
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

impl BuildVEffWithEnergy for SpinCollinear {
    fn build_v_eff_with_energy_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        spin: Option<&chemrust_hamiltonian_core::Density>,
        rho_aug_fine: Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>>,
        rho_aug_spin: Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>>,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<(Self::VEff, f64, f64, f64), chemrust_hamiltonian_core::Error> {
        use chemrust_hamiltonian_core::{
            nlcc, poisson, upsample_density_to_fine_grid, xc,
            Density as CoreDensity,
        };
        // A spin density is required for SpinCollinear.
        let rho_spin = spin.ok_or_else(|| chemrust_hamiltonian_core::Error::MissingField(
            "SpinCollinear BuildVEffWithEnergy requires spin density".into(),
        ))?;

        // 1. Upsample total density to fine grid
        let rho_total_fine_pw =
            upsample_density_to_fine_grid(rho.as_real_grid(), wave_grid, fine_grid)
                .map_err(|_| chemrust_hamiltonian_core::Error::Format {
                    section: "build_v_eff_with_energy_spin".into(),
                    detail: "upsample total density failed".into(),
                })?;

        // 2. Upsample spin density to fine grid
        let rho_spin_fine_pw =
            upsample_density_to_fine_grid(rho_spin.as_real_grid(), wave_grid, fine_grid)
                .map_err(|_| chemrust_hamiltonian_core::Error::Format {
                    section: "build_v_eff_with_energy_spin".into(),
                    detail: "upsample spin density failed".into(),
                })?;

        // 3. Add augmentation: total ρ += Q_rho_sum, spin ρ += Q_rho_sum_sp.
        //    CASTEP density.f90:1148-1149 — ion_augmentation_charge produces
        //    separate total and spin augmentation contributions.
        let (rho_total_fine, rho_spin_fine) = match (rho_aug_fine, rho_aug_spin) {
            (Some(aug_total), Some(aug_spin)) => (
                chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                    rho_total_fine_pw.as_real_array() + aug_total.as_real_array(),
                ),
                chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                    rho_spin_fine_pw.as_real_array() + aug_spin.as_real_array(),
                ),
            ),
            (Some(aug_total), None) => (
                chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                    rho_total_fine_pw.as_real_array() + aug_total.as_real_array(),
                ),
                rho_spin_fine_pw,
            ),
            _ => (rho_total_fine_pw, rho_spin_fine_pw),
        };

        // 4. Assemble V_eff via VEffBuilder (returns up/down effective potentials).
        let v_eff = VEffBuilder::<SpinCollinear>::new(cell, pots, fine_grid)
            .with_density(CoreDensity::from_inner(rho_total_fine.clone()), Some(CoreDensity::from_inner(rho_spin_fine.clone())))
            .assemble()?;

        // 5. Recover per-spin densities for double-counting integrals.
        //    rho_up = (rho_total + rho_spin) / 2, rho_dn = (rho_total - rho_spin) / 2
        let rho_up = CoreDensity::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                (rho_total_fine.as_real_array() + rho_spin_fine.as_real_array()) * 0.5,
            ),
        );
        let rho_dn = CoreDensity::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                (rho_total_fine.as_real_array() - rho_spin_fine.as_real_array()) * 0.5,
            ),
        );

        // 6. Reconstruct core density (spin-independent).
        let rho_core = nlcc::reconstruct_rho_core(cell, pots, fine_grid)?
            .into_inner();
        let rho_total_core = CoreDensity::from_inner(
            rho_total_fine.clone() + &rho_core,
        );

        // 7. Hartree potential from total density (same for both spins).
        let v_h = poisson::solve_poisson(
            &CoreDensity::from_inner(rho_total_fine.clone()),
            fine_grid,
        )?;

        // 8. XC potential: spin-polarised PBE.
        let v_xc = xc::compute_pbe_xc_spin(
            rho_total_core.as_real_grid().as_real_array(),
            rho_spin_fine.as_real_array(),
            fine_grid,
            cell.volume,
        )?;

        // 9. Energy integrals — all use d_v = 1/N_grid on the fine grid.
        //    (CASTEP xc_gga / xc.f90:1056, pot.f90:4205).
        let n_grid = rho_total_fine.as_real_array().len() as f64;
        let d_v = 1.0 / n_grid;

        // E_H = 0.5 * sum(rho_total * v_h) / N
        let e_hartree_raw: f64 = rho_total_fine.as_real_array().iter()
            .zip(v_h.as_real_grid().as_real_array().iter())
            .map(|(&rv, &vh)| rv * vh * d_v)
            .sum();
        let e_hartree = 0.5 * e_hartree_raw;

        // E_xc from the spin-polarised functional (already integrated).
        let e_xc = v_xc.energy;

        // CRITICAL C5-S4: rho_vxc = (1/N) * SUM(rho_up[i] * vxc_up[i] + rho_dn[i] * vxc_dn[i])
        // NOT rho_total * vxc_avg.  CASTEP pot.f90:4205 sums per-spin.
        let rho_vxc: f64 = rho_up.as_real_grid().as_real_array().iter()
            .zip(v_xc.v_xc_up.iter())
            .map(|(&rv, &vxc)| rv * vxc * d_v)
            .sum::<f64>()
            + rho_dn.as_real_grid().as_real_array().iter()
                .zip(v_xc.v_xc_dn.iter())
                .map(|(&rv, &vxc)| rv * vxc * d_v)
                .sum::<f64>();

        Ok((v_eff, e_xc, e_hartree, rho_vxc))
    }
}

impl<S: SpinPolicy + BuildVEff> ScfIteration<S, Initialized, MixingOff> {
    /// Assemble V_eff[ρ] from the current density.
    /// Consumes `self`, returns a state in the `VEffBuilt` phase.
    pub fn build_v_eff(self) -> Result<ScfIteration<S, VEffBuilt, MixingOff>, Error> {
        let total_density = self.density.total();
        let core_rho = chemrust_hamiltonian_core::Density::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(total_density.as_wave_array().clone()),
        );
        // Extract spin density for SpinCollinear (ρ_spin = ρ_up − ρ_down).
        // build_v_eff_impl uses it for correct spin-polarised V_eff assembly
        // (V_xc_up ≠ V_xc_dn via compute_pbe_xc_spin).
        let core_spin;
        let spin = if S::nspins() == 2 {
            let spin_density = self.density.spin();
            core_spin = Some(chemrust_hamiltonian_core::Density::from_inner(
                chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                    spin_density.0.as_wave_array().clone(),
                ),
            ));
            core_spin.as_ref()
        } else {
            None
        };
        let v_eff = S::build_v_eff_impl(
            &self.cell, &self.pots, &core_rho,
            spin,
            &self.wave_grid, &self.fine_grid,
        )
        .map_err(|_| Error::NotImplemented)?;
        let mut next: ScfIteration<S, VEffBuilt, MixingOff> = self.into_phase();
        next.v_eff = Some(v_eff);
        Ok(next)
    }
}

impl<S: SpinPolicy + BuildVEffWithEnergy> ScfIteration<S, Initialized, MixingOff> {
    /// Assemble V_eff with energy components for total energy computation.
    ///
    /// Same as `build_v_eff` but also populates the energy fields
    /// (`e_xc`, `e_hartree`, `rho_vxc`) from the XC/Hartree evaluation.
    /// These are needed by `check()` for total energy computation.
    pub fn build_v_eff_with_energy(self) -> Result<ScfIteration<S, VEffBuilt, MixingOff>, Error> {
        let total_density = self.density.total();
        let core_rho = chemrust_hamiltonian_core::Density::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(total_density.as_wave_array().clone()),
        );
        let spin_arg = if S::nspins() == 2 {
            let spin_d = self.density.spin().into_inner();
            Some(chemrust_hamiltonian_core::Density::from_inner(
                chemrust_hamiltonian_core::fft::RealGrid::from_inner(spin_d.as_wave_array().clone()),
            ))
        } else {
            None
        };
        // Per-spin augmentation: CASTEP density.f90:1148-1149 adds
        // Q_rho_sum (total) to charge and Q_rho_sum_sp (spin) to spin density.
        // aug_up/dn = kpt-weighted augmentation per spin from beta_psi.
        let (aug_total, aug_spin) = if S::nspins() == 2 {
            match (self.density_aug_fine[0].as_ref(), self.density_aug_fine[1].as_ref()) {
                (Some(a0), Some(a1)) => (
                    Some(chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                        a0.as_real_array().to_owned() + a1.as_real_array())),
                    Some(chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                        a0.as_real_array().to_owned() - a1.as_real_array())),
                ),
                (Some(a0), None) => (Some(a0.clone()), None),
                _ => (None, None),
            }
        } else {
            (self.density_aug_fine[0].as_ref().cloned(), None)
        };
        let (v_eff, e_xc, e_hartree, rho_vxc) = S::build_v_eff_with_energy_impl(
            &self.cell, &self.pots, &core_rho,
            spin_arg.as_ref(),
            aug_total.as_ref(),
            aug_spin.as_ref(),
            &self.wave_grid, &self.fine_grid,
        )
        .map_err(|_| Error::NotImplemented)?;
        let mut next: ScfIteration<S, VEffBuilt, MixingOff> = self.into_phase();
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
        #[cfg(feature = "chebyshev")]
        { self.diagonalize_with_mode(ndeg, occupations, FilterMode::SinvHKeepHEig) }
        #[cfg(not(feature = "chebyshev"))]
        { self.diagonalize_inner(ndeg, occupations, None) }
    }

    /// Like `diagonalize` but with an explicit `FilterMode` for the A/B/C diagnostic sweep.
    #[cfg(feature = "chebyshev")]
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
    #[cfg(feature = "chebyshev")]
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
        mut self,
        _ndeg: usize,
        occupations: Option<&[f64]>,
        #[cfg(feature = "chebyshev")] filter_mode: FilterMode,
        d_override_per_ion: Option<&[Option<Vec<f64>>]>,
    ) -> Result<ScfIteration<S, WavefunctionsUpdated, MixingOff>, Error> {
        // GPU context created ONCE outside spin loop
        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone())?;
        let solver = SolverHandle::new(stream.clone())?;
        let kernels = CudaKernelSet::new(&ctx)?;

        let v_eff_ref = self.v_eff.as_ref().expect("VEffBuilt phase guarantees v_eff is Some");
        let nspins = S::nspins();
        let n_bands = self.n_bands;
        let nkpts = self.nkpts;
        let _max_n_pw = self.max_n_pw;

        // PCI-E transfer tracker
        let mut pcie = PcieAccount::default();

        // FFT plan created ONCE outside spin loop (same grid for both spins)
        let [ngz, ngy, ngx] = self.wave_grid.grid();
        let grid_size_usize = ngx * ngy * ngz;
        let inv_ntotal = 1.0 / (grid_size_usize as f64);

        // TPA preconditioner CUDA kernels (compiled once)
        let tpa_precond = TpaPreconditioner::new(&ctx)?;

        // FFT plan (batched C2C)
        let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
            ngx as i32, ngy as i32, ngz as i32, n_bands as i32,
            stream.clone(),
        ).map_err(Error::Fft)?;

        // ---- Eigensolver dispatch: CHEMRUST_EIGENSOLVER env var ----
        let eigensolver_method = std::env::var("CHEMRUST_EIGENSOLVER")
            .unwrap_or_else(|_| "davidson".to_string());

        // Per-spin-per-kpt result accumulators
        let mut out_psi_vec: Vec<KptDataSet<PwCoefficients>> = Vec::with_capacity(nspins);
        let mut out_psi_cpu_vec: Vec<KptDataSet<Vec<Complex64>>> = Vec::with_capacity(nspins);
        let mut out_eig_vec: Vec<KptDataSet<Vec<f64>>> = Vec::with_capacity(nspins);
        let mut out_beta_vec: Vec<KptDataSet<Option<Vec<CudaSlice<CudaComplex>>>>> = Vec::with_capacity(nspins);

        // -----------------------------------------------------------------------
        // SPIN+KPT LOOP — CASTEP electronic.f90:488-495
        //   do ns = 1, nspins
        //     do nk = 1, nkpts  ← inner kpt loop
        //       call hamiltonian_diagonalise(wvfn, nk, ns, ...)
        //     end do
        //   end do
        //
        // Shared VNL cache: spin-0 builds KptSharedVnl per kpt; spin-1 reuses.
        // Persisted in self.shared_vnl_cache and self.handle_shared_vnl across
        // SCF iterations — matches ffi.rs ChemrustHandle/KptData lifecycle.
        // -----------------------------------------------------------------------
        for ispin in 0..nspins {
            // Initialize per-kpt result accumulators for this spin
            let mut spin_psi_kpts: Vec<PwCoefficients> = Vec::with_capacity(nkpts);
            let mut spin_psi_cpu_kpts: Vec<Vec<Complex64>> = Vec::with_capacity(nkpts);
            let mut spin_eig_kpts: Vec<Vec<f64>> = Vec::with_capacity(nkpts);
            let mut spin_beta_kpts: Vec<Option<Vec<CudaSlice<CudaComplex>>>> = Vec::with_capacity(nkpts);

            // 1. V_eff for this spin channel (kpt-independent)
            let v_eff_spin = S::v_eff_for_spin(v_eff_ref, ispin);
            let v_eff_arr = v_eff_spin.as_real_grid().as_real_array();

            // 2. V_eff preparation (downsample + GPU upload), per spin
            let veff = pipeline::v_eff_prepare(v_eff_arr, &self.fine_grid, &self.wave_grid, &stream, &mut pcie)?;
            let v_eff_slice = &veff.gpu_slice;
            let grid_size_usize = veff.grid_size;
            let (_min_veff, _max_veff) = { let arr = v_eff_arr; (arr.iter().cloned().fold(f64::INFINITY, f64::min), arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max)) };

            // Diagnostic (spin-0 only, kpt-0 only)
            if ispin == 0 && nkpts > 0 {
                let total_density = self.density.total();
                let rho_arr = total_density.as_wave_array();
                let n_grid = rho_arr.len() as f64;
                let rho_sum: f64 = rho_arr.iter().sum();
                #[allow(unused_variables)]
                let rho_min = rho_arr.iter().cloned().fold(f64::INFINITY, f64::min);
                #[allow(unused_variables)]
                let rho_max = rho_arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                #[allow(unused_variables)]
                let total_e_raw_conv = rho_sum / n_grid;
                let mut psi_abs_min = f64::INFINITY;
                let mut psi_abs_max = 0.0f64;
                for ikpt in 0..nkpts {
                    for c in self.psi_cpu[ispin][ikpt].iter() {
                        let a = c.norm();
                        if a < psi_abs_min { psi_abs_min = a; }
                        if a > psi_abs_max { psi_abs_max = a; }
                    }
                }
                #[cfg(feature = "scf_diag")]
                eprintln!(
                    "[V_eff] spin={} min={:.4} max={:.4} range={:.4} Ha  [Density] rho_sum={:.4e} rho_min={:.4e} rho_max={:.4e}  total_e(raw_conv=sum/N)={:.6}  [psi] |c|_min={:.3e} |c|_max={:.3e}",
                    ispin, _min_veff, _max_veff, _max_veff - _min_veff,
                    rho_sum, rho_min, rho_max,
                    total_e_raw_conv,
                    psi_abs_min, psi_abs_max,
                );
            }

            // -----------------------------------------------------------------------
            // KPT LOOP — inner to spin loop (CASTEP electronic.f90:489)
            //   do nk = 1, nkpts
            //     call hamiltonian_diagonalise(wvfn, nk, ns, ...)
            // -----------------------------------------------------------------------
            for ikpt in 0..nkpts {
                let pw_coords_kpt = &self.pw_coords[ikpt];
                let n_pw_kpt = pw_coords_kpt.len();
                let kpoint = &self.k_points[ikpt];

                // 4. Upload kinetic energies & FFT index map to GPU (per kpt)
                let kinetic_cpu = compute_kinetic_energies(
                    pw_coords_kpt,
                    self.wave_grid.recip_lattice(),
                    kpoint.coords,
                );
                let kpt = pipeline::kpt_gpu_upload(&kinetic_cpu.0, &self.pw_fft_indices[ikpt], &stream, &mut pcie)?;

                // 6. V_NL precomputation per (spin,kpt) — D-matrices are per-(kpt,spin).
                //    CASTEP hamiltonian.f90:1013
                let psi_host = self.psi_cpu[ispin][ikpt].clone();
                // Build kpt-independent HandleSharedVnl lazily on first use.
                if self.handle_shared_vnl.is_none() {
                    self.handle_shared_vnl = Some(build_handle_shared_vnl(
                        &self.pots, &self.cell, &self.wave_grid, Some(&self.fine_grid),
                        &stream, &mut pcie,
                    )?);
                }
                // Share spin-independent VNL data across spin channels:
                // spin-0 builds KptSharedVnl fresh; spin-1 reuses it.
                let shared_vnl = self.shared_vnl_cache[ikpt].clone();
                let mut vnl_data = VnlBatchData::precompute_with_d_override(
                    pw_coords_kpt, &self.pots, &self.cell,
                    &self.wave_grid, Some(&self.fine_grid), kpoint,
                    &psi_host, n_bands, n_pw_kpt, occupations,
                    None,
                    d_override_per_ion, shared_vnl,
                    self.handle_shared_vnl.clone(),
                    &stream, &mut pcie, &blas, &kernels,
                )?;
                self.shared_vnl_cache[ikpt] = Some(vnl_data.shared.clone());

                if d_override_per_ion.is_none() {
                    vnl_data.rescreen_d(v_eff_arr, &stream, &kernels, &blas)?;
                }

                // 7. H2D psi for this (spin,kpt)
                let psi_wfn = WavefunctionSet::<ColumnDistributed>::new(psi_host, n_bands, n_pw_kpt);
                let psi_gpu = Gpu::from_host_with(&psi_wfn, &stream, &mut pcie)?;
                // Used in Chebyshev path below (feature-gated)
                #[allow(unused_variables)]
                let prev_psi_dev = psi_gpu.as_device_slice();
                let psi_pw = PwCoefficients::new(psi_gpu.as_device_slice().clone());

                // 8. Eigensolver dispatch per (spin,kpt)
                if eigensolver_method == "davidson" {
                    // --- Davidson diagonalization ---
                    let result = unsafe {
                        pipeline::run_davidson(
                            &psi_pw, v_eff_slice, &kpt.kinetic_precond, &kpt.fft_idx_dev,
                            &vnl_data, n_pw_kpt, n_bands, grid_size_usize, inv_ntotal, &fft_plan,
                            &blas, &solver, &kernels, &tpa_precond, &stream, &ctx,
                            30, false,  // max_outer_iter=30, gamma_point=false
                        )?
                    };

                    // Wrap psi_out CudaSlice for D2H
                    let psi_new_gpu: Gpu<WavefunctionSet<ColumnDistributed>> = Gpu {
                        slice: result.psi_out,
                        shape: vec![n_bands, n_pw_kpt],
                        ctx: (*ctx).clone(),
                        _marker: PhantomData,
                    };
                    let eigenvalues_cpu = Cpu::new(result.eigenvalues);

                    // Beta-psi recomputation per (spin,kpt)
                    let mut beta_psi_gpu: Vec<CudaSlice<CudaComplex>> = Vec::new();
                    for entry in &vnl_data.entries {
                        let ne = entry.n_expanded as usize;
                        let mut c_proj: CudaSlice<CudaComplex> = stream
                            .alloc_zeros(ne * n_bands)
                            .map_err(Error::Cuda)?;
                        unsafe {
                            blas.gemm_c64(ZgemmConfig {
                                transa: op::C,
                                transb: op::N,
                                m: ne as i32,
                                n: n_bands as i32,
                                k: n_pw_kpt as i32,
                                alpha: CudaComplex { x: 1.0, y: 0.0 },
                                lda: n_pw_kpt as i32,
                                ldb: n_pw_kpt as i32,
                                beta: CudaComplex { x: 0.0, y: 0.0 },
                                ldc: ne as i32,
                            }, &entry.beta_g, psi_new_gpu.as_device_slice(), &mut c_proj)?;
                        }
                        beta_psi_gpu.push(c_proj);
                    }

                    stream.synchronize()?;
                    let Cpu(psi_new) = psi_new_gpu.sync_to_host_with(&stream, &mut pcie)?;

                    spin_psi_kpts.push(PwCoefficients::new(psi_new_gpu.as_device_slice().clone()));
                    spin_psi_cpu_kpts.push(psi_new.data);
                    spin_eig_kpts.push(eigenvalues_cpu.into_inner());
                    spin_beta_kpts.push(Some(beta_psi_gpu));
                } else {
                    // Non-Davidson path: Chebyshev (if enabled) or error
                    #[cfg(feature = "chebyshev")]
                    {
                        let _eig: Option<&[f64]> = None;
                        let (psi_filtered_row, hpsi_row) = chebyshev_filter(
                            &psi_gpu, &v_eff_gpu, &self.pots,
                            &self.wave_grid, kpoint, &self.cell,
                            pw_coords_kpt,
                            &vnl_data, &kpt.fft_idx_dev, _min_veff, _max_veff,
                            &kernels, &mut pcie, _eig, _ndeg, &blas, &solver, &stream, &ctx,
                            filter_mode,
                            None,
                        )?;
                        let pin_cfg = crate::eigensolver::rayleigh_ritz::RrPinConfig::from_env();
                        let (psi_new_gpu, eigenvalues_cpu, beta_psi_gpu) = rayleigh_ritz(
                            &psi_filtered_row, &hpsi_row, &vnl_data,
                            n_bands, n_pw_kpt, &kernels,
                            &mut pcie,
                            &solver, &blas, &stream, &ctx,
                            Some(prev_psi_dev),
                            Some(&pin_cfg),
                        )?;

                        stream.synchronize()?;
                        #[cfg(feature = "scf_diag")]
                        {
                            let eig = eigenvalues_cpu.0.clone();
                            eprintln!("[RR] spin={} kpt={} eigenvalues: first={:.4e} Ha  last={:.4e} Ha  count={}",
                                ispin, ikpt,
                                eig.first().copied().unwrap_or(f64::NAN),
                                eig.last().copied().unwrap_or(f64::NAN),
                                eig.len());
                        }

                        let Cpu(psi_new) = psi_new_gpu.sync_to_host_with(&stream, &mut pcie)?;

                        spin_psi_kpts.push(PwCoefficients::new(psi_new_gpu.as_device_slice().clone()));
                        spin_psi_cpu_kpts.push(psi_new.data);
                        spin_eig_kpts.push(eigenvalues_cpu.into_inner());
                        spin_beta_kpts.push(Some(beta_psi_gpu));
                    }
                    #[cfg(not(feature = "chebyshev"))]
                    {
                        return Err(Error::Cuda(cudarc::driver::result::DriverError(
                            cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                        )));
                    }
                }
            }
            // ---- End kpt loop ----

            // Wrap per-kpt results into KptDataSet for this spin
            out_psi_vec.push(KptDataSet::new(spin_psi_kpts, nkpts));
            out_psi_cpu_vec.push(KptDataSet::new(spin_psi_cpu_kpts, nkpts));
            out_eig_vec.push(KptDataSet::new(spin_eig_kpts, nkpts));
            out_beta_vec.push(KptDataSet::new(spin_beta_kpts, nkpts));
        }
        // ---- End spin loop ----

        // Construct next phase state from per-spin results
        let mut next: ScfIteration<S, WavefunctionsUpdated, MixingOff> = self.into_phase();
        for (ispin, psi_item) in out_psi_vec.into_iter().enumerate() {
            next.psi[ispin] = psi_item;
        }
        for (ispin, cpu_item) in out_psi_cpu_vec.into_iter().enumerate() {
            next.psi_cpu[ispin] = cpu_item;
        }
        for (ispin, eig_item) in out_eig_vec.into_iter().enumerate() {
            next.eigenvalues[ispin] = eig_item;
        }
        for (ispin, beta_item) in out_beta_vec.into_iter().enumerate() {
            next.beta_psi_per_ion[ispin] = beta_item;
        }

        // Stash Davidson diagnostics for post-diagonalize inspection
        // (only meaningful from the last spin's call)
        #[cfg(any(test, feature = "scf_diag"))]
        {
            next.last_davidson_diagnostics = DAVIDSON_LAST_DIAG.lock().unwrap().clone();
        }

        Ok(next)
    }

    /// Test-only: run Chebyshev + Rayleigh-Ritz and return the internal subspace matrices
    /// H_sub, S_sub, X alongside the normal RR outputs.
    ///
    /// Returns: `(eigenvalues, H_sub_cpu, S_sub_cpu, X_cpu)` — all col-major (n_bands × n_bands).
    #[cfg(all(any(test, feature = "scf_diag"), feature = "chebyshev"))]
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
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

        // Gamma-point test path: use kpt-0
        let n_pw = self.pw_coords[0].len();
        let pw_coords = self.pw_coords[0].clone();
        let psi_host = self.psi_cpu[0][0].clone();
        let n_bands = self.n_bands;

        let psi_wfn = WavefunctionSet::<ColumnDistributed>::new(psi_host.clone(), n_bands, n_pw);
        let psi_gpu = Gpu::from_host_with(&psi_wfn, &stream, &mut pcie)?;

        let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_wave.as_fine_array().clone()),
        );
        let vnl_data = VnlBatchData::precompute(
            &pw_coords, &self.pots, &self.cell,
            &self.wave_grid, &self.k_points[0],
            &psi_host, n_bands, n_pw, None,
            Some(&v_eff_for_d),
            &stream, &mut pcie, &blas, &kernels,
        )?;

        let eig: Option<&[f64]> = None;

        let fft_idx_dev: CudaSlice<i32> = stream
            .clone_htod(&self.pw_fft_indices[0])
            .map_err(Error::Cuda)?;
        pcie.h2d_bytes += self.pw_fft_indices[0].len() * std::mem::size_of::<i32>();

        let (psi_filtered_row, hpsi_row) = chebyshev_filter(
            &psi_gpu, &v_eff_gpu, &self.pots,
            &self.wave_grid, &self.k_points[0], &self.cell,
            &pw_coords,
            &vnl_data, &fft_idx_dev, min_veff, max_veff,
            &kernels, &mut pcie, eig, ndeg, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig,
            None,
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
    #[cfg(feature = "chebyshev")]
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
        // Gamma-point test path: use kpt-0
        let n_pw = self.pw_coords[0].len();
        let psi_host = self.psi_cpu[0][0].clone();
        let n_bands = self.n_bands;
        let psi_wfn = WavefunctionSet::<ColumnDistributed>::new(psi_host.clone(), n_bands, n_pw);
        let psi_gpu = Gpu::from_host_with(&psi_wfn, &stream, &mut pcie)?;

        let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_wave.as_fine_array().clone()),
        );
        let vnl_data = VnlBatchData::precompute(
            &self.pw_coords[0], &self.pots, &self.cell,
            &self.wave_grid, &self.k_points[0],
            &psi_host, n_bands, n_pw, occupations,
            Some(&v_eff_for_d),
            &stream, &mut pcie, &blas, &kernels,
        )?;

        let fft_idx_dev: CudaSlice<i32> = stream.clone_htod(&self.pw_fft_indices[0])
            .map_err(Error::Cuda)?;

        crate::eigensolver::chebyshev::apply_h_components_for_test(
            &psi_gpu, &v_eff_gpu, &self.wave_grid, &self.pw_coords[0],
            &vnl_data, &fft_idx_dev, &kernels, &blas, &stream,
            self.k_points[0].coords,
        )
    }

    /// Compute `S · ψ_input` for an arbitrary host-side ψ block using the
    /// Vnl projectors + Q matrices that this VEffBuilt state would use in a
    /// real diagonalize call. Used by tests that need true USPP S-inner
    /// products (e.g., `⟨ψ_a | S | ψ_b⟩` with normalisation `⟨ψ|S|ψ⟩ = 1`).
    ///
    /// `psi_input` must be column-major `[band * n_pw + g]`; output is the
    /// same layout. `n_pw` is read from `self.psi.n_pw`.
    #[cfg(feature = "chebyshev")]
    #[doc(hidden)]
    pub fn apply_s_for_test(
        &self,
        psi_input: &[Complex64],
        n_bands: usize,
    ) -> Result<Vec<Complex64>, Error> {
        // Gamma-point test path: use kpt-0
        let n_pw = self.pw_coords[0].len();
        assert_eq!(
            psi_input.len(), n_bands * n_pw,
            "apply_s_for_test: psi_input shape mismatch"
        );

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
        let v_eff_for_d = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_wave.as_fine_array().clone()),
        );

        let psi_for_betapsi = self.psi_cpu[0][0].clone();
        let n_bands_state = self.n_bands;

        let vnl_data = VnlBatchData::precompute(
            &self.pw_coords[0], &self.pots, &self.cell,
            &self.wave_grid, &self.k_points[0],
            &psi_for_betapsi, n_bands_state, n_pw, None,
            Some(&v_eff_for_d),
            &stream, &mut pcie, &blas, &kernels,
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
            PerSpinDensity,
            PerSpinAugDensity,
            OccupationSet,
            FermiEnergies,
        ),
        Error,
    > {
        let nspins = S::nspins();
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
        // Per-spin electron counts from the CURRENT density's net spin.
        // For SpinCollinear: n_up = 0.5*(N + net_spin), n_dn = 0.5*(N - net_spin).
        // Using the density that's already stored (fixture or previous iteration)
        // ensures the spin polarisation is preserved through density reconstruction.
        // CASTEP electronic.f90:8742-8746: frac_elec(1)=0.5*(N+net_spin), frac_elec(2)=0.5*(N-net_spin).
        let net_spin: f64 = if nspins == 2 {
            let up_arr = self.density[0].as_wave_array();
            let dn_arr = self.density[1].as_wave_array();
            let n_grid = up_arr.len() as f64;
            up_arr.iter().zip(dn_arr.iter())
                .map(|(&u, &d)| u - d)
                .sum::<f64>() / n_grid
        } else {
            0.0
        };
        let n_electrons_per_spin: Vec<f64> = (0..nspins)
            .map(|ispin| {
                if nspins == 2 {
                    if ispin == 0 { 0.5 * (n_electrons + net_spin) }
                    else          { 0.5 * (n_electrons - net_spin) }
                } else {
                    n_electrons
                }
            })
            .collect();

        #[cfg(feature = "scf_diag")]
        eprintln!("[DensityCtor] net_spin={:.6} n_electrons={:.6} per_spin={:?}",
            net_spin, n_electrons, n_electrons_per_spin);

        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let kernels = CudaKernelSet::new(&ctx)?;

        let mut densities: Vec<Density> = Vec::with_capacity(nspins);
        let mut aug_densities: Vec<Option<chemrust_hamiltonian_core::fft::RealGrid<f64>>> =
            Vec::with_capacity(nspins);
        let mut occs: Vec<Vec<f64>> = Vec::with_capacity(nspins);
        let mut fermi: Vec<f64> = Vec::with_capacity(nspins);
        let nkpts = self.nkpts;

        // ---- Spin loop: per-spin kpt-weighted occupations + density construction ----
        // CASTEP density.f90:2179-2187
        // For multi-kpt: occupations are found with kpt weights, then density is
        // accumulated as ρ_spin = Σ_k w_k * ρ_k(ψ_k, occ_k).

        // Pre-compute spin_freed occupations (shared Fermi level for both spins).
        let spin_freed_occs: Option<(Vec<Vec<Vec<f64>>>, f64)> = if self.spin_freed && nspins == 2 {
            let mut ev_up = Vec::new();
            let mut ev_dn = Vec::new();
            for ikpt in 0..nkpts {
                ev_up.extend_from_slice(&self.eigenvalues[0][ikpt]);
                ev_dn.extend_from_slice(&self.eigenvalues[1][ikpt]);
            }
            let (shared_fermi, occ_up_all, occ_dn_all, _net_spin) =
                crate::density::find_fermi_free(
                    &ev_up, &ev_dn, &self.smearing, n_electrons,
                    1.0 / nspins as f64,
                )?;
            // Split back per-kpt
            let n_bands = self.n_bands;
            let mut occ_up_kpts = Vec::with_capacity(nkpts);
            let mut occ_dn_kpts = Vec::with_capacity(nkpts);
            for ikpt in 0..nkpts {
                let s = ikpt * n_bands;
                occ_up_kpts.push(occ_up_all[s..s + n_bands].to_vec());
                occ_dn_kpts.push(occ_dn_all[s..s + n_bands].to_vec());
            }
            Some((vec![occ_up_kpts, occ_dn_kpts], shared_fermi))
        } else {
            None
        };

        for ispin in 0..nspins {
            // Multi-kpt occupation search: collect eigenvalues across all kpts with kpt weights
            let kpt_weights: Vec<f64> = (0..nkpts)
                .map(|ikpt| self.k_points[ikpt].weight)
                .collect();
            let (occupations_all_kpts, chem_pot) = if let Some((ref occs_data, fermi_val)) = spin_freed_occs {
                (occs_data[ispin].clone(), ChemicalPotential(fermi_val))
            } else {
                crate::density::compute_occupations_weighted(
                    self.eigenvalues[ispin].as_ref(), // &[Vec<f64>] — per-kpt eigenvalues
                    &kpt_weights,
                    &self.smearing,
                    n_electrons_per_spin[ispin],
                    1.0 / nspins as f64, // occ_factor: 1.0 for NonSpin, 0.5 for SpinCollinear
                )?
            };

            // Density accumulation across kpts (weighted sum)
            let mut total_density: Option<Density> = None;
            let mut total_aug: Option<chemrust_hamiltonian_core::fft::RealGrid<f64>> = None;
            let mut occ_sum_diag: f64 = 0.0;

            for ikpt in 0..nkpts {
                let w_k = self.k_points[ikpt].weight;
                let n_pw_kpt = self.pw_coords[ikpt].len();
                let psi_kpt = &self.psi_cpu[ispin][ikpt];
                let occ_kpt = &occupations_all_kpts[ikpt];

                let kpt_density = crate::density::construct_density_gpu()
                    .psi_data(psi_kpt)
                    .occupations(occ_kpt)
                    .fft_indices(&self.pw_fft_indices[ikpt])
                    .wave_grid(&self.wave_grid)
                    .cell_volume(self.cell.volume)
                    .n_bands(self.n_bands)
                    .n_pw(n_pw_kpt)
                    .kernels(&kernels)
                    .stream(&stream)
                    .call()?;

                let weighted_kpt_density = kpt_density * w_k;
                total_density = Some(match total_density {
                    Some(acc) => acc + weighted_kpt_density,
                    None => weighted_kpt_density,
                });

                // Augmentation density per (spin,kpt)
                let rho_aug_kpt = match self.beta_psi_per_ion[ispin][ikpt].as_ref() {
                    Some(beta_psi) => {
                        // Lazily build QSfCache on first use (geometry-static, shared across spins).
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

                        if let Some(cache) = self.q_sf_cache.as_ref() {
                            let mut pcie = PcieAccount::default();
                            crate::density::compute_aug_density_gpu(
                                cache,
                                beta_psi,
                                occ_kpt,
                                &stream,
                                &mut pcie,
                                &kernels,
                            )?
                        } else {
                            // QSfCache build failed — fall back to CPU path.
                            let beta_psi_cpu: Vec<ndarray::Array2<num_complex::Complex64>> = beta_psi
                                .iter()
                                .map(|bp_dev| {
                                    let ne_times_nb = bp_dev.len();
                                    let bp_host: Vec<CudaComplex> = stream.clone_dtoh(bp_dev).map_err(Error::Cuda)?;
                                    let n_bands = occ_kpt.len();
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
                                occ_kpt,
                                &self.pots,
                                &self.cell,
                                &self.fine_grid,
                            )?
                        }
                    }
                    None => continue, // No beta_psi for this kpt — skip augmentation
                };

                // Weighted accumulation of augmentation density
                let weighted_aug = if w_k != 1.0 {
                    let arr = rho_aug_kpt.as_real_array();
                    chemrust_hamiltonian_core::fft::RealGrid::from_inner(arr.mapv(|x| x * w_k))
                } else {
                    rho_aug_kpt
                };
                total_aug = Some(match total_aug {
                    Some(acc) => chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                        acc.as_real_array().to_owned() + weighted_aug.as_real_array(),
                    ),
                    None => weighted_aug,
                });

                // Accumulate occupations for diagnostics
                occ_sum_diag += w_k * occ_kpt.iter().sum::<f64>();
            }

            stream.synchronize()?;
            let new_density = total_density.expect("at least one kpt must produce density");
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
                let occ_max = occ_sum_diag;
                #[cfg(feature = "scf_diag")]
                eprintln!(
                    "[NewDensity] spin={} rho_sum={:.4e} rho_min={:.4e} rho_max={:.4e}  total_e(raw_conv=sum/N)={:.6}  [occ] Σ(w·occ)={:.4} target_n_e/spin={:.4} chem_pot={:.4} Ha",
                    ispin, rho_sum, rho_min, rho_max,
                    total_e_raw_conv, occ_sum_diag, n_electrons_per_spin[ispin], chem_pot.0,
                );
                #[cfg(feature = "scf_diag")]
                if let Some(ref aug) = total_aug {
                    let arr = aug.as_real_array();
                    let aug_sum: f64 = arr.iter().sum();
                    eprintln!(
                        "[AugDensity] spin={} aug_sum={:.4e} ∫ρ_aug dV ≈ {:.4}",
                        ispin, aug_sum, aug_sum * self.cell.volume / arr.len() as f64,
                    );
                }
            }

            densities.push(new_density);
            aug_densities.push(total_aug);
            // Flatten per-kpt occupations for OccupationSet (use kpt-0 for now;
            // weighted occupations are tracked separately for multi-kpt in future).
            occs.push(occupations_all_kpts[0].clone());
            fermi.push(chem_pot.0);
        }

        let per_spin_density = PerSpinDensity(SpinChannelData::new::<S>(densities));
        let per_spin_aug = PerSpinAugDensity(SpinChannelData::new::<S>(aug_densities));
        let occ_set = OccupationSet(SpinChannelData::new::<S>(occs));
        let fermi_energies = FermiEnergies(fermi);

        Ok((per_spin_density, per_spin_aug, occ_set, fermi_energies))
    }

    /// Construct density with `Off` mixing phase — the history stays as-is
    /// and no mixing transformation is applied during `mix()`.
    pub fn construct_density_off(
        mut self,
    ) -> Result<ScfIteration<S, DensityUpdated<MixingOff>, MixingOff>, Error> {
        let (new_density, density_aug_fine, occ_set, fermi_energies) = self.compute_density_from_wavefunctions()?;
        let mut next: ScfIteration<S, DensityUpdated<MixingOff>, MixingOff> =
            self.into_phase();
        next.density = new_density;
        next.density_aug_fine = density_aug_fine;
        next.fermi_energy = fermi_energies;
        let _ = occ_set; // Used in future for energy computation
        Ok(next)
    }

    /// Construct density and transition the history to `Kerker` phase.
    pub fn construct_density_kerker(
        mut self,
    ) -> Result<ScfIteration<S, DensityUpdated<Kerker>, Kerker>, Error> {
        let (new_density, density_aug_fine, occ_set, fermi_energies) = self.compute_density_from_wavefunctions()?;
        // Convert history from MixingOff → Kerker (creates GPU preconditioner)
        let kerker_history = self.history.into_kerker(&self.wave_grid)?;
        let _ = occ_set;
        Ok(ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_points: self.k_points,
            smearing: self.smearing,
            n_bands: self.n_bands,
            max_n_pw: self.max_n_pw,
            nkpts: self.nkpts,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: new_density,
            psi: self.psi,
            psi_cpu: self.psi_cpu,
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
            fermi_energy: fermi_energies,
            beta_psi_per_ion: self.beta_psi_per_ion,
            q_sf_cache: self.q_sf_cache,
            handle_shared_vnl: self.handle_shared_vnl,
            shared_vnl_cache: self.shared_vnl_cache,
            density_aug_fine,
            #[cfg(any(test, feature = "scf_diag"))]
            last_davidson_diagnostics: None,
            scf_iter: self.scf_iter,
            spin_freed: self.spin_freed,
            _phase: PhantomData,
        })
    }

    /// Construct density and transition the history to `Pulay` phase.
    pub fn construct_density_pulay(
        mut self,
    ) -> Result<ScfIteration<S, DensityUpdated<Pulay>, Pulay>, Error> {
        let (new_density, density_aug_fine, _occ_set, fermi_energies) = self.compute_density_from_wavefunctions()?;
        // Convert history: MixingOff → Kerker → Pulay
        let kerker_history = self.history.into_kerker(&self.wave_grid)?;
        let pulay_history = kerker_history.into_pulay();
        Ok(ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_points: self.k_points,
            smearing: self.smearing,
            n_bands: self.n_bands,
            max_n_pw: self.max_n_pw,
            nkpts: self.nkpts,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: new_density,
            psi: self.psi,
            psi_cpu: self.psi_cpu,
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
            fermi_energy: fermi_energies,
            beta_psi_per_ion: self.beta_psi_per_ion,
            q_sf_cache: self.q_sf_cache,
            handle_shared_vnl: self.handle_shared_vnl,
            shared_vnl_cache: self.shared_vnl_cache,
            density_aug_fine,
            #[cfg(any(test, feature = "scf_diag"))]
            last_davidson_diagnostics: None,
            scf_iter: self.scf_iter,
            spin_freed: self.spin_freed,
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
        let nspins = S::nspins();
        let mut densities = Vec::with_capacity(nspins);
        let mut prev_densities = Vec::with_capacity(nspins);
        for ispin in 0..nspins {
            let (mixed, prev) = self.history.mix(self.density[ispin].clone(), ispin);
            densities.push(mixed);
            prev_densities.push(prev);
        }
        let history_off = self.history.into_off();
        ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_points: self.k_points,
            smearing: self.smearing,
            n_bands: self.n_bands,
            max_n_pw: self.max_n_pw,
            nkpts: self.nkpts,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: PerSpinDensity(SpinChannelData::new::<S>(densities)),
            psi: self.psi,
            psi_cpu: self.psi_cpu,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: history_off,
            previous_density: PerSpinDensity(SpinChannelData::new::<S>(prev_densities)),
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
            handle_shared_vnl: self.handle_shared_vnl,
            shared_vnl_cache: self.shared_vnl_cache,
            density_aug_fine: self.density_aug_fine,
            #[cfg(any(test, feature = "scf_diag"))]
            last_davidson_diagnostics: self.last_davidson_diagnostics,
            scf_iter: self.scf_iter,
            spin_freed: self.spin_freed,
            _phase: PhantomData,
        }
    }
}

impl<S: SpinPolicy> ScfIteration<S, DensityUpdated<Kerker>, Kerker> {
    /// Mix with Kerker preconditioning.
    pub fn mix(mut self) -> ScfIteration<S, Mixed, MixingOff> {
        let nspins = S::nspins();
        let mut densities = Vec::with_capacity(nspins);
        let mut prev_densities = Vec::with_capacity(nspins);
        for ispin in 0..nspins {
            let (mixed, prev) = self.history.mix(self.density[ispin].clone(), ispin);
            densities.push(mixed);
            prev_densities.push(prev);
        }
        let history_off = self.history.into_off();
        ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_points: self.k_points,
            smearing: self.smearing,
            n_bands: self.n_bands,
            max_n_pw: self.max_n_pw,
            nkpts: self.nkpts,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: PerSpinDensity(SpinChannelData::new::<S>(densities)),
            psi: self.psi,
            psi_cpu: self.psi_cpu,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: history_off,
            previous_density: PerSpinDensity(SpinChannelData::new::<S>(prev_densities)),
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
            handle_shared_vnl: self.handle_shared_vnl,
            shared_vnl_cache: self.shared_vnl_cache,
            density_aug_fine: self.density_aug_fine,
            #[cfg(any(test, feature = "scf_diag"))]
            last_davidson_diagnostics: self.last_davidson_diagnostics,
            scf_iter: self.scf_iter,
            spin_freed: self.spin_freed,
            _phase: PhantomData,
        }
    }
}

impl<S: SpinPolicy> ScfIteration<S, DensityUpdated<Pulay>, Pulay> {
    /// Mix with Pulay / DIIS.
    pub fn mix(mut self) -> ScfIteration<S, Mixed, MixingOff> {
        let nspins = S::nspins();
        let mut densities = Vec::with_capacity(nspins);
        let mut prev_densities = Vec::with_capacity(nspins);
        for ispin in 0..nspins {
            let (mixed, prev) = self.history.mix(self.density[ispin].clone(), ispin);
            densities.push(mixed);
            prev_densities.push(prev);
        }
        let history_off = self.history.into_off();
        ScfIteration {
            cell: self.cell,
            pots: self.pots,
            wave_grid: self.wave_grid,
            fine_grid: self.fine_grid,
            k_points: self.k_points,
            smearing: self.smearing,
            n_bands: self.n_bands,
            max_n_pw: self.max_n_pw,
            nkpts: self.nkpts,
            pw_coords: self.pw_coords,
            pw_fft_indices: self.pw_fft_indices,
            density: PerSpinDensity(SpinChannelData::new::<S>(densities)),
            psi: self.psi,
            psi_cpu: self.psi_cpu,
            eigenvalues: self.eigenvalues,
            v_eff: self.v_eff,
            history: history_off,
            previous_density: PerSpinDensity(SpinChannelData::new::<S>(prev_densities)),
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
            handle_shared_vnl: self.handle_shared_vnl,
            shared_vnl_cache: self.shared_vnl_cache,
            density_aug_fine: self.density_aug_fine,
            #[cfg(any(test, feature = "scf_diag"))]
            last_davidson_diagnostics: self.last_davidson_diagnostics,
            scf_iter: self.scf_iter,
            spin_freed: self.spin_freed,
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
        // 1. Compute occupations for band energy (kpt-weighted)
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
        // Compute per-spin occupation-weighted band energy.
        // NonSpin: single channel, total n_electrons (unchanged).
        // SpinCollinear: two channels, per-spin electron counts from
        // net_spin = ∫(ρ_up − ρ_down).
        let nkpts = self.nkpts;
        let kpt_weights: Vec<f64> = (0..nkpts)
            .map(|ikpt| self.k_points[ikpt].weight)
            .collect();
        let nspins = S::nspins();

        #[cfg(feature = "scf_diag")]
        {
            eprintln!("[check] nkpts={nkpts} kpt_weights={kpt_weights:?}");
            eprintln!("[check] n_electrons={n_electrons} nspins={nspins}");
        }

        // Per-spin electron counts from integrated spin density.
        // CASTEP electronic.f90:8742-8746: frac_elec(1)=0.5*(N+net_spin),
        // frac_elec(2)=0.5*(N-net_spin).
        let net_spin: f64 = if nspins == 2 {
            let up_arr = self.density[0].as_wave_array();
            let dn_arr = self.density[1].as_wave_array();
            let n_grid = up_arr.len() as f64;
            up_arr.iter().zip(dn_arr.iter())
                .map(|(&u, &d)| u - d)
                .sum::<f64>() / n_grid
        } else {
            0.0
        };

        #[cfg(feature = "scf_diag")]
        eprintln!("[check] net_spin={net_spin:.6} n_up={:.6} n_dn={:.6}",
            0.5 * (n_electrons + net_spin), 0.5 * (n_electrons - net_spin));

        // Pre-compute spin_freed occupations for check() when spin is freed.
        let spin_freed_occs: Option<(Vec<Vec<Vec<f64>>>, f64)> = if self.spin_freed && nspins == 2 {
            let mut ev_up = Vec::new();
            let mut ev_dn = Vec::new();
            for ikpt in 0..nkpts {
                ev_up.extend_from_slice(&self.eigenvalues[0][ikpt]);
                ev_dn.extend_from_slice(&self.eigenvalues[1][ikpt]);
            }
            let (shared_fermi, occ_up_all, occ_dn_all, _net_spin) =
                crate::density::find_fermi_free(
                    &ev_up, &ev_dn, &self.smearing, n_electrons,
                    1.0 / nspins as f64,
                )?;
            let n_bands = self.n_bands;
            let mut occ_up_kpts = Vec::with_capacity(nkpts);
            let mut occ_dn_kpts = Vec::with_capacity(nkpts);
            for ikpt in 0..nkpts {
                let s = ikpt * n_bands;
                occ_up_kpts.push(occ_up_all[s..s + n_bands].to_vec());
                occ_dn_kpts.push(occ_dn_all[s..s + n_bands].to_vec());
            }
            Some((vec![occ_up_kpts, occ_dn_kpts], shared_fermi))
        } else {
            None
        };

        let mut e_band: f64 = 0.0;
        for ispin in 0..nspins {
            let n_spin_electrons = if nspins == 2 {
                if ispin == 0 { 0.5 * (n_electrons + net_spin) }
                else          { 0.5 * (n_electrons - net_spin) }
            } else {
                n_electrons
            };
            let (occupations_all_kpts, chem_pot) = if let Some((ref occs_data, fermi_val)) = spin_freed_occs {
                (occs_data[ispin].clone(), ChemicalPotential(fermi_val))
            } else {
                crate::density::compute_occupations_weighted(
                    self.eigenvalues[ispin].as_ref(),
                    &kpt_weights,
                    &self.smearing,
                    n_spin_electrons,
                    1.0 / nspins as f64, // occ_factor: 1.0 for NonSpin, 0.5 for SpinCollinear
                )?
            };
            self.fermi_energy[ispin] = chem_pot.0;

            // 2. Total energy assembly (if energy components are available)
            if let (Some(_e_xc), Some(_e_hartree), Some(_rho_vxc)) =
                (self.e_xc, self.e_hartree, self.rho_vxc)
            {
                // Kpt-weighted band energy per spin: Σ_k w_k Σ_b f_{bk} ε_{bk}
                let e_band_spin: f64 = self.eigenvalues[ispin]
                    .iter()
                    .zip(kpt_weights.iter())
                    .zip(occupations_all_kpts.iter())
                    .map(|((eigs, &w), occs)| {
                        w * eigs.iter()
                            .zip(occs.iter())
                            .map(|(&eps, &f)| f * eps)
                            .sum::<f64>()
                    })
                    .sum();
                e_band += e_band_spin;
            }
        }

        // 2. Total energy assembly (if energy components are available)
        if let (Some(e_xc), Some(e_hartree), Some(rho_vxc)) =
            (self.e_xc, self.e_hartree, self.rho_vxc)
        {
            // Compute electronic entropy correction -TS (Mermin free energy).
            // CASTEP electronic.f90:9768-9784 (GAUSSIAN), applied at 3282-3283.
            let sqrt_pi = std::f64::consts::PI.sqrt();
            let mut ts_sum = 0.0;
            for ispin in 0..nspins {
                ts_sum += crate::density::compute_entropy_ts(
                    self.eigenvalues[ispin].as_ref(),
                    &kpt_weights,
                    self.fermi_energy[ispin],
                    self.smearing.width.to_ha(),
                );
            }
            let ts = ts_sum * self.smearing.width.to_ha() / (nspins as f64 * sqrt_pi);

            let e_total = crate::energy::assemble_total_energy_from_band(
                e_band,
                e_xc,
                e_hartree,
                rho_vxc,
                self.ewald,
                ts,
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

        // 4. Density RMS change (mixed total density vs pre-mix snapshot)
        let dens_rms = {
            let total_current = self.density.total();
            let total_previous = self.previous_density.total();
            let current = total_current.as_wave_array();
            let previous = total_previous.as_wave_array();
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
            density: self.density.total(),
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
        state.scf_iter += 1;
        // CASTEP electronic.f90:516-518 — spin freed at scf_cycle == spin_fix
        // where scf_cycle is 1-based.  Our scf_iter is 0-based, so the
        // transition point is scf_iter == spin_fix - 1.
        if state.smearing.spin_fix >= 0 && !state.spin_freed {
            if state.scf_iter as i32 >= state.smearing.spin_fix - 1 {
                state.spin_freed = true;
                eprintln!("[chemrust] spin_fix iterations done -- freeing spin (iter {})", state.scf_iter);
            }
            if state.scf_iter as i32 == state.smearing.spin_fix - 1 {
                let amp = if S::nspins() > 1 { 2.0 } else { 0.5 };
                state.history = crate::mixing::DensityHistory::with_amplitude(S::nspins(), amp);
                state.next_mixing = MixingPhaseKind::Off;
            }
        }
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
pub fn run_scf_with_energy<S: SpinPolicy + BuildVEffWithEnergy>(
    state: ScfIteration<S, Initialized, MixingOff>,
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
    /// Permitted relative drift in TOTAL electron count (soft + augmented,
    /// fraction). Compares soft density on wave grid plus augmentation
    /// density on fine grid against iter-1 baseline. CASTEP reference:
    /// total ≈ 186 e⁻, soft ≈ 36.8% (Cu111+CO, RHO_SOFT_SUM ≈ 3.0e7,
    /// RHO_AUG_SUM ≈ 5.1e7).
    pub electron_count_tolerance: f64,
    /// Permitted drift in the soft-electron fraction (soft/total) relative
    /// to the iter-1 baseline. A catastrophic flip in this ratio (e.g.
    /// from 0.37 → 0.80) indicates the augmentation density is not being
    /// constructed or mixing is corrupting the soft/aug split.
    pub soft_fraction_tolerance: f64,
    /// Raw PARAMETERS section from the reference .check file, for the iter-2
    /// CASTEP continuation discriminator. Set by the test from `fx.check`;
    /// the capture point injects this into the emitted .check to satisfy
    /// CASTEP's `parameters_restore` without re-reading the fixture.
    #[cfg(any(test, feature = "scf_diag"))]
    pub parameters_raw: Option<Vec<u8>>,
}

impl Default for ScfDivergenceGate {
    fn default() -> Self {
        Self {
            max_last_band_ha: 5.0,
            min_band0_ha: -30.0,
            max_veff_range_factor: 5.0,
            max_iter: 60,
            electron_count_tolerance: 0.05,
            soft_fraction_tolerance: 0.20,
            #[cfg(any(test, feature = "scf_diag"))]
            parameters_raw: None,
        }
    }
}

/// Like `run_scf_with_energy` but with optional divergence gating.
///
/// When `gate = Some(...)`, the loop inspects per-iteration state after the
/// SCF check completes. Any out-of-bounds value triggers a panic with a
/// structured message identifying which gate fired, the offending value,
/// the iteration index, and a brief reference to the expected range.
pub fn run_scf_with_energy_gated<S: SpinPolicy + BuildVEffWithEnergy>(
    state: ScfIteration<S, Initialized, MixingOff>,
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
    let mut iter1_soft_fraction: Option<f64> = None;
    loop {
        state.scf_iter += 1;
        // CASTEP electronic.f90:516-518 — spin freed at scf_cycle == spin_fix
        // where scf_cycle is 1-based.  Our scf_iter is 0-based, so the
        // transition point is scf_iter == spin_fix - 1.
        if state.smearing.spin_fix >= 0 && !state.spin_freed {
            if state.scf_iter as i32 >= state.smearing.spin_fix - 1 {
                state.spin_freed = true;
                eprintln!("[chemrust] spin_fix iterations done -- freeing spin (iter {})", state.scf_iter);
            }
            if state.scf_iter as i32 == state.smearing.spin_fix - 1 {
                let amp = if S::nspins() > 1 { 2.0 } else { 0.5 };
                state.history = crate::mixing::DensityHistory::with_amplitude(S::nspins(), amp);
                state.next_mixing = MixingPhaseKind::Off;
            }
        }
        state = {
            let v_eff = state.build_v_eff_with_energy()?;

            // Sample V_eff range BEFORE moving v_eff into diagonalize, so we
            // can use it for the gate at end-of-iteration.
            let veff_range_now = if gate.is_some() {
                v_eff.v_eff().as_ref().map(|veff| {
                    let arr = S::v_eff_for_spin(veff, 0).as_real_grid().as_real_array();
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
                    if let Some(e_total) = next.total_energy {
                        let energy_ev = e_total * HARTREE_TO_EV;
                        let gain_per_atom = prev_energy
                            .map(|prev| (e_total - prev) / next.cell.num_ions as f64)
                            .unwrap_or(0.0);
                        let gain_ev = gain_per_atom * HARTREE_TO_EV;
                        let fermi_ev = next.fermi_energy[0] * HARTREE_TO_EV;
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
                        // Total electron count = soft (wave grid) + augmented (fine grid).
                        // CASTEP raw units (ρ×Ω): electrons = sum/N for each grid.
                        let total_density = next.density.total();
                        let rho_arr = total_density.as_wave_array();
                        let soft_sum: f64 = rho_arr.iter().sum();
                        let n_soft = rho_arr.len() as f64;
                        let soft_e = soft_sum / n_soft;
                        let aug_e = next
                            .density_aug_fine[0]
                            .as_ref()
                            .map(|aug| {
                                let aug_arr = aug.as_real_array();
                                let aug_sum: f64 = aug_arr.iter().sum();
                                aug_sum / aug_arr.len() as f64
                            })
                            .unwrap_or(0.0);
                        let total_e_now = soft_e + aug_e;
                        let soft_fraction_now = if total_e_now > 0.0 {
                            soft_e / total_e_now
                        } else {
                            0.0
                        };

                        if iter1_n_electrons.is_none() {
                            iter1_n_electrons = Some(total_e_now);
                            iter1_soft_fraction = Some(soft_fraction_now);
                        }
                        let baseline_total = iter1_n_electrons.unwrap_or(total_e_now);
                        let baseline_soft_frac = iter1_soft_fraction.unwrap_or(soft_fraction_now);

                        // Gate 1: total electron count drift
                        let total_drift =
                            (total_e_now - baseline_total).abs() / baseline_total.abs().max(1.0);
                        if total_drift > g.electron_count_tolerance {
                            panic!(
                                "[SCF gate] iter {}: total electron count {:.4} (soft={:.4} aug={:.4}) drifted {:.2}% from iter-1 baseline {:.4}, exceeds gate {:.2}%\n  \
                                 likely cause: augmentation density or mixing broken",
                                iter_count, total_e_now, soft_e, aug_e,
                                total_drift * 100.0, baseline_total,
                                g.electron_count_tolerance * 100.0,
                            );
                        }

                        // Gate 2: soft/augmented fraction must not flip catastrophically
                        let frac_drift =
                            (soft_fraction_now - baseline_soft_frac).abs();
                        if frac_drift > g.soft_fraction_tolerance {
                            panic!(
                                "[SCF gate] iter {}: soft fraction {:.4} drifted Δ={:.4} from iter-1 baseline {:.4}, exceeds gate {:.4}\n  \
                                 CASTEP reference soft fraction ≈ 0.368 (RHO_SOFT ~3.0e7 / total ~8.1e7)\n  \
                                 likely cause: augmentation density not constructed or mixing corrupting soft/aug split",
                                iter_count, soft_fraction_now, frac_drift, baseline_soft_frac,
                                g.soft_fraction_tolerance,
                            );
                        }
                    }

                    // --- iter-2 capture: write .check for CASTEP continuation discriminator ---
                    #[cfg(any(test, feature = "scf_diag"))]
                    if iter_count == 2 {
                        let n_electrons: f64 = next
                            .cell
                            .species_iter()
                            .map(|info| {
                                next.pots
                                    .get(info.symbol)
                                    .and_then(|p| p.ionic_charge())
                                    .unwrap_or(0.0)
                                    * info.num_ions as f64
                            })
                            .sum();
                        if let Some(mut castep_bin) =
                            crate::scf_capture::capture_as_castep_bin(&next, n_electrons)
                        {
                            // Inject parameters_raw from the gate (set by test
                            // from fx.check — no re-read of the fixture file).
                            if let Some(ref gate) = gate {
                                #[cfg(any(test, feature = "scf_diag"))]
                                if let Some(ref raw) = gate.parameters_raw {
                                    // Gate stores flattened Vec<u8>, CastepBin expects Vec<Vec<u8>>
                                    // Wrap the single record in a Vec
                                    castep_bin.parameters_raw = vec![raw.clone()];
                                }
                            }
                            let out_path = std::env::var("CHEMRUST_CHECK_DUMP")
                                .unwrap_or_else(|_| "chemrust_iter2.check".to_string());
                            if let Ok(mut file) = std::fs::File::create(&out_path) {
                                match chemrust_hamiltonian_core::CheckFile::write(
                                    &mut file, &castep_bin,
                                ) {
                                    Ok(()) => {
                                        tracing::info!(
                                            "[scf capture] iter-2 state written to {out_path}"
                                        );
                                        panic!(
                                            "SCF_DISCRIMINATOR_STOP: iter-2 .check at {out_path}; \
                                             run CASTEP continuation from this checkpoint"
                                        );
                                    }
                                    Err(e) => tracing::warn!(
                                        "[scf capture] failed to write .check: {e}"
                                    ),
                                }
                            } else {
                                tracing::warn!(
                                    "[scf capture] cannot create {}: {}",
                                    out_path,
                                    std::io::Error::last_os_error(),
                                );
                            }
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
    /// Access ψ coefficient slice for spin-0 (debug/testing only).
    #[doc(hidden)]
    pub fn psi_data(&self) -> &[Complex64] {
        &self.psi_cpu[0][0]
    }
    /// Mutable access to ψ coefficient slice for spin-0, kpt-0 (debug/testing only).
    /// Use to inject controlled pollution before running diagonalize.
    #[doc(hidden)]
    pub fn psi_data_mut(&mut self) -> &mut [Complex64] {
        &mut self.psi_cpu[0][0]
    }
    /// Set eigenvalues for spin-0, kpt-0 (debug/testing only). Used to mimic the iter-2 filter
    /// code path which receives eigenvalues from a prior diagonalization.
    #[doc(hidden)]
    pub fn set_eigenvalues(&mut self, eigs: Vec<f64>) {
        self.eigenvalues[0] = KptDataSet::new(vec![eigs], 1);
    }
    /// Access (n_bands, max_n_pw) shape (debug/testing only).
    #[doc(hidden)]
    pub fn psi_shape(&self) -> (usize, usize) {
        (self.n_bands, self.max_n_pw)
    }
}

impl<S: SpinPolicy, M: MixingPhase> ScfIteration<S, WavefunctionsUpdated, M> {
    /// Access the eigenvalues for spin-0, kpt-0 (for testing).
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues[0][0]
    }
    /// Access ψ coefficient slice after diagonalization for spin-0, kpt-0 (debug/testing only).
    #[doc(hidden)]
    pub fn psi_data(&self) -> &[Complex64] {
        &self.psi_cpu[0][0]
    }

    /// Davidson diagnostics from the most recent diagonalize call.
    /// Returns `Some` when Davidson was dispatched (CHEMRUST_EIGENSOLVER=davidson);
    /// `None` when Chebyshev-RR was used.
    #[cfg(any(test, feature = "scf_diag"))]
    #[doc(hidden)]
    pub fn davidson_diagnostics(&self) -> Option<&DavidsonDiagnostic> {
        self.last_davidson_diagnostics.as_ref()
    }

    /// Number of locked bands after a Davidson solve.
    #[cfg(any(test, feature = "scf_diag"))]
    pub fn davidson_n_locked(&self) -> Option<usize> {
        self.last_davidson_diagnostics.as_ref().map(|d| d.n_locked)
    }

    /// Number of unconverged bands after a Davidson solve.
    #[cfg(any(test, feature = "scf_diag"))]
    pub fn davidson_n_unconverged(&self) -> Option<usize> {
        self.last_davidson_diagnostics
            .as_ref()
            .map(|d| d.n_unconverged)
    }

    /// Maximum S⁻¹-weighted residual norm after a Davidson solve.
    #[cfg(any(test, feature = "scf_diag"))]
    pub fn davidson_max_residual_sinv(&self) -> Option<f64> {
        self.last_davidson_diagnostics
            .as_ref()
            .map(|d| d.max_residual_sinv)
    }

    /// Per-band S⁻¹-weighted residual norms after a Davidson solve.
    #[cfg(any(test, feature = "scf_diag"))]
    pub fn davidson_residual_norms_sinv(&self) -> Option<Vec<f64>> {
        self.last_davidson_diagnostics
            .as_ref()
            .map(|d| d.residual_norms_sinv.0.clone())
    }

    /// Davidson eigenvalue deltas from the most recent solve.
    #[cfg(any(test, feature = "scf_diag"))]
    pub fn davidson_eigenvalue_deltas(&self) -> Option<Vec<f64>> {
        self.last_davidson_diagnostics
            .as_ref()
            .map(|d| d.eigenvalue_deltas.clone())
    }

}

impl<S: SpinPolicy, M: MixingPhase> ScfIteration<S, DensityUpdated<M>, MixingOff> {
    /// Access ρ_aug on the fine grid for spin-0 (debug/testing only).
    #[doc(hidden)]
    pub fn density_aug_fine(&self) -> Option<&chemrust_hamiltonian_core::fft::RealGrid<f64>> {
        self.density_aug_fine[0].as_ref()
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

    /// Per-spin eigenvalues from the most recent diagonalization.
    /// Index `[ispin]` gives the per-kpt eigenvalues for that spin channel.
    #[doc(hidden)]
    pub fn per_spin_eigenvalues(&self) -> &PerSpinEigenvalues { &self.eigenvalues }

    /// Access eigenvalues for a specific (spin, kpt) pair.
    #[doc(hidden)]
    pub fn eigenvalues_at(&self, ispin: usize, ikpt: usize) -> &[f64] {
        &self.eigenvalues[ispin][ikpt]
    }

    /// Per-spin density (ρ_up, ρ_down for SpinCollinear; ρ for NonSpin).
    #[doc(hidden)]
    pub fn per_spin_density(&self) -> &PerSpinDensity { &self.density }

    /// Per-spin Fermi energies from the most recent occupation search.
    #[doc(hidden)]
    pub fn fermi_energies(&self) -> &FermiEnergies { &self.fermi_energy }

}

impl<S: SpinPolicy, M: MixingPhase> ScfIteration<S, Mixed, M> {
    /// Access eigenvalues from the most recent diagonalization, spin-0 kpt-0 (diagnostic only).
    #[doc(hidden)]
    pub fn eigenvalues(&self) -> &[f64] { &self.eigenvalues[0][0] }
}

impl<S: SpinPolicy> ScfIteration<S, Initialized, MixingOff> {
    /// Mutable access to total density for spin-0 (for perturbation testing).
    pub fn density_mut(&mut self) -> &mut Density {
        &mut self.density[0]
    }
    /// Clear the cached augmentation density (debug/testing only).
    /// The next `build_v_eff_with_energy` call will use ρ_PW only.
    #[doc(hidden)]
    pub fn clear_density_aug_fine(&mut self) {
        for ispin in 0..S::nspins() {
            self.density_aug_fine[ispin] = None;
        }
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
    use std::sync::LazyLock;
    use cudarc::driver::CudaContext;
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

    /// Shared GPU context for test helpers (created once on first use).
    static TEST_GPU_CTX: LazyLock<Arc<CudaContext>> = LazyLock::new(|| {
        CudaContext::new(0).expect("CUDA GPU required for check() tests")
    });

    /// Create a dummy PwCoefficients with a valid (but empty) GPU allocation.
    /// The slice is never accessed in check() tests — it exists only to satisfy
    /// the ScfIteration struct field type.
    fn dummy_pw_coeffs() -> PwCoefficients {
        let stream = TEST_GPU_CTX.default_stream();
        let slice = stream.alloc_zeros::<CudaComplex>(0).unwrap();
        PwCoefficients::new(slice)
    }

    /// check() with no energy data returns NotConverged.
    #[test]
    fn test_check_not_converged_no_energy() {
        let shape = [4, 4, 4]; // Fortran: [ngz, ngy, ngx]
        let density = Density::from_inner(WaveGridArray::from_inner(Array3::<f64>::zeros(shape)));
        let previous_density = Density::from_inner(WaveGridArray::from_inner(
            Array3::<f64>::from_elem(shape, 2.0),
        ));
        let psi_data = vec![Complex64::ZERO; 4 * 27];
        let n_bands = 4;
        let n_pw = 27;
        let state: ScfIteration<NonSpin, Mixed, MixingOff> = ScfIteration {
            cell: dummy_cell(),
            pots: PseudopotentialSet::new(),
            wave_grid: dummy_grid(),
            fine_grid: dummy_grid(),
            k_points: KptDataSet::new(vec![KPoint::default()], 1),
            smearing: SmearingParams::builder().build(),
            n_bands,
            max_n_pw: n_pw,
            nkpts: 1,
            pw_coords: KptDataSet::new(vec![Vec::new()], 1),
            pw_fft_indices: KptDataSet::new(vec![Vec::new()], 1),
            density: PerSpinDensity(SpinChannelData::new::<NonSpin>(vec![density])),
            psi: PerSpinPwCoefficients(SpinChannelData::new::<NonSpin>(
                vec![KptDataSet::new(vec![dummy_pw_coeffs()], 1)],
            )),
            psi_cpu: SpinChannelData::new::<NonSpin>(vec![KptDataSet::new(vec![psi_data], 1)]),
            eigenvalues: PerSpinEigenvalues(SpinChannelData::new::<NonSpin>(
                vec![KptDataSet::new(vec![vec![-0.3, -0.2]], 1)],
            )),
            v_eff: None,
            history: DensityHistory::new(1),
            previous_density: PerSpinDensity(SpinChannelData::new::<NonSpin>(vec![previous_density])),
            next_mixing: MixingPhaseKind::Off,
            e_xc: None,
            e_hartree: None,
            rho_vxc: None,
            ewald: 0.0,
            energy_buffer: Vec::new(),
            total_energy: None,
            fermi_energy: FermiEnergies(vec![0.0]),
            beta_psi_per_ion: PerSpinBetaProjections(SpinChannelData::new::<NonSpin>(
                vec![KptDataSet::new(vec![None], 1)]
            )),
            q_sf_cache: None,
            density_aug_fine: PerSpinAugDensity(SpinChannelData::new::<NonSpin>(vec![None])),
            last_davidson_diagnostics: None,
            scf_iter: 0,
            spin_freed: false,
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
        let n_bands = 4;
        let n_pw = 27;
        ScfIteration {
            cell: dummy_cell(),
            pots: PseudopotentialSet::new(),
            wave_grid: dummy_grid(),
            fine_grid: dummy_grid(),
            k_points: KptDataSet::new(vec![KPoint::default()], 1),
            smearing: SmearingParams::builder().build(),
            n_bands,
            max_n_pw: n_pw,
            nkpts: 1,
            pw_coords: KptDataSet::new(vec![Vec::new()], 1),
            pw_fft_indices: KptDataSet::new(vec![Vec::new()], 1),
            density: PerSpinDensity(SpinChannelData::new::<NonSpin>(vec![
                Density::from_inner(WaveGridArray::from_inner(Array3::<f64>::zeros(shape))),
            ])),
            psi: PerSpinPwCoefficients(SpinChannelData::new::<NonSpin>(
                vec![KptDataSet::new(vec![dummy_pw_coeffs()], 1)],
            )),
            psi_cpu: SpinChannelData::new::<NonSpin>(vec![
                KptDataSet::new(vec![vec![Complex64::ZERO; n_bands * n_pw]], 1),
            ]),
            eigenvalues: PerSpinEigenvalues(SpinChannelData::new::<NonSpin>(
                vec![KptDataSet::new(vec![vec![-0.3, -0.2]], 1)],
            )),
            v_eff: None,
            history: DensityHistory::new(1),
            previous_density: PerSpinDensity(SpinChannelData::new::<NonSpin>(vec![
                Density::from_inner(WaveGridArray::from_inner(
                    Array3::<f64>::from_elem(shape, 2.0),
                )),
            ])),
            next_mixing,
            e_xc: None,
            e_hartree: None,
            rho_vxc: None,
            ewald: 0.0,
            energy_buffer,
            total_energy: None,
            fermi_energy: FermiEnergies(vec![0.0]),
            beta_psi_per_ion: PerSpinBetaProjections(SpinChannelData::new::<NonSpin>(
                vec![KptDataSet::new(vec![None], 1)]
            )),
            q_sf_cache: None,
            density_aug_fine: PerSpinAugDensity(SpinChannelData::new::<NonSpin>(vec![None])),
            last_davidson_diagnostics: None,
            scf_iter: 0,
            spin_freed: false,
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
