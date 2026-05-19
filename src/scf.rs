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
use crate::device::Gpu;
use crate::eigensolver::chebyshev::{chebyshev_filter, CudaKernelSet};
use crate::eigensolver::rayleigh_ritz::rayleigh_ritz;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::layout::{ColumnDistributed, Cpu, WavefunctionSet};
use crate::mixing::{DensityHistory, Kerker, MixingOff, MixingPhase, Pulay};
use crate::types::{
    Density, EffectivePotential, Error, FinalResult, FineGridArray, KPoint, SmearingParams,
};

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
            _phase: PhantomData,
        }
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
            Array3::zeros(shape),
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
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<(Self::VEff, f64, f64, f64), chemrust_hamiltonian_core::Error>;
}

impl BuildVEffWithEnergy for NonSpin {
    fn build_v_eff_with_energy_impl(
        cell: &CellGeometry, pots: &PseudopotentialSet,
        rho: &chemrust_hamiltonian_core::Density,
        wave_grid: &GVectorGrid, fine_grid: &GVectorGrid,
    ) -> Result<(chemrust_hamiltonian_core::EffectivePotential, f64, f64, f64), chemrust_hamiltonian_core::Error> {
        let result = VEffBuilder::<NonSpin>::new(cell, pots, fine_grid)
            .assemble_on_fine_grid_with_energy(rho, wave_grid, fine_grid)?;
        let v_eff = result.v_eff;
        Ok((v_eff, result.e_xc, result.e_hartree, result.rho_vxc))
    }
}

impl<S: SpinPolicy + BuildVEff> ScfIteration<S, Initialized, MixingOff> {
    /// Assemble V_eff[ρ] from the current density.
    /// Consumes `self`, returns a state in the `VEffBuilt` phase.
    pub fn build_v_eff(self) -> Result<ScfIteration<S, VEffBuilt, MixingOff>, Error> {
        let core_rho = chemrust_hamiltonian_core::Density::from_inner(
            self.density.as_wave_array().clone(),
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
            self.density.as_wave_array().clone(),
        );
        let (v_eff, e_xc, e_hartree, rho_vxc) = NonSpin::build_v_eff_with_energy_impl(
            &self.cell, &self.pots, &core_rho,
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
    pub fn diagonalize(
        self,
        ndeg: usize,
    ) -> Result<ScfIteration<S, WavefunctionsUpdated, MixingOff>, Error> {
        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone())?;
        let solver = SolverHandle::new(stream.clone())?;
        let kernels = CudaKernelSet::new(&ctx)?;

        // Extract V_eff as raw Array3<f64> via SpinPolicy::v_eff_for_spin
        let v_eff_ref = self.v_eff.as_ref().expect("VEffBuilt phase guarantees v_eff is Some");
        let v_eff_spin = S::v_eff_for_spin(v_eff_ref, 0);
        let v_eff_arr = v_eff_spin.as_array();

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
        let v_eff_gpu = Gpu::from_host_with(&v_eff_wave, &stream, &mut pcie)?;

        // Clone host data BEFORE moving self.psi into GPU
        let pw_coords = self.pw_coords.clone();
        let psi_host = self.psi.data.clone();
        let n_bands = self.psi.n_bands;
        let n_pw = self.psi.n_pw;

        // H2D psi
        let psi_gpu = Gpu::from_host_with(&self.psi, &stream, &mut pcie)?;

        // V_NL precomputation (CPU, uses chemrust-hamiltonian, one-time cost)
        let vnl_data = VnlBatchData::precompute(
            &pw_coords, &self.pots, &self.cell,
            &self.wave_grid, &self.k_point,
            &psi_host, n_bands, n_pw,
            &stream, &mut pcie,
        )?;

        // Clone eigenvalues before moving self
        let eig_clone = self.eigenvalues.clone();
        let eig = if eig_clone.is_empty() { None } else { Some(eig_clone.as_slice()) };

        // Upload PW-to-FFT index map to GPU
        let fft_idx_dev: CudaSlice<i32> = stream
            .clone_htod(&self.pw_fft_indices)
            .map_err(Error::Cuda)?;
        pcie.h2d_bytes += self.pw_fft_indices.len() * std::mem::size_of::<i32>();

        // Chebyshev filter (pipeline: T+V_loc via FFT, V_NL via gemm)
        let (psi_filtered_row, hpsi_row) = chebyshev_filter(
            &psi_gpu, &v_eff_gpu, &self.pots,
            &self.wave_grid, &self.k_point, &self.cell,
            &vnl_data, &fft_idx_dev, min_veff, max_veff,
            &kernels, &mut pcie, eig, ndeg, &blas, &stream, &ctx,
        )?;

        // Rayleigh-Ritz
        let (psi_new_gpu, eigenvalues_cpu) = rayleigh_ritz(
            &psi_filtered_row, &hpsi_row,
            n_bands, n_pw, &kernels,
            &mut pcie,
            &solver, &blas, &stream, &ctx,
        )?;

        stream.synchronize()?;
        let psi_bytes = n_bands * n_pw * 16;            // complex double
        let eig_bytes = n_bands * 8;
        let Cpu(psi_new) = psi_new_gpu.sync_to_host_with(&stream, &mut pcie)?;
        let eigenvalues = eigenvalues_cpu.into_inner();

        // Assert: hot path should only have setup H2D + final D2H.
        // Any additional transfer (e.g. D2H inside the Chebyshev loop) is a bug.
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
        let kinetic_bytes = grid_size * std::mem::size_of::<f64>();
        let vnl_bytes: usize = vnl_data.entries.iter()
            .map(|e| (e.beta_g.len() + e.d_matrix.len()) * 16)
            .sum();
        assert_eq!(
            pcie.h2d_bytes,
            psi_bytes + veff_bytes + fft_idx_bytes + kinetic_bytes + vnl_bytes,
            "H2D tracking check failed",
        );

        let mut next: ScfIteration<S, WavefunctionsUpdated, MixingOff> = self.into_phase();
        next.psi = psi_new;
        next.eigenvalues = eigenvalues;
        Ok(next)
    }
}

// ---------------------------------------------------------------------------
// Transition 3: WavefunctionsUpdated → DensityUpdated
// ---------------------------------------------------------------------------
// Three variants for the three mixing phases.  Each method name encodes
// which mixing phase the returned state carries.

impl<S: SpinPolicy> ScfIteration<S, WavefunctionsUpdated, MixingOff> {
    /// Shared density computation (occupations + GPU construction).
    fn compute_density_from_wavefunctions(&self) -> Result<Density, Error> {
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

        let occupations =
            crate::density::compute_occupations(&self.eigenvalues, &self.smearing, n_electrons)?;

        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream();
        let kernels = CudaKernelSet::new(&ctx)?;

        let new_density = crate::density::construct_density_gpu()
            .psi_data(&self.psi.data)
            .occupations(&occupations)
            .fft_indices(&self.pw_fft_indices)
            .wave_grid(&self.wave_grid)
            .cell_volume(self.cell.volume)
            .n_bands(self.psi.n_bands)
            .n_pw(self.psi.n_pw)
            .kernels(&kernels)
            .stream(&stream)
            .call()?;

        stream.synchronize()?;
        Ok(new_density)
    }

    /// Construct density with `Off` mixing phase — the history stays as-is
    /// and no mixing transformation is applied during `mix()`.
    pub fn construct_density_off(
        self,
    ) -> Result<ScfIteration<S, DensityUpdated<MixingOff>, MixingOff>, Error> {
        let new_density = self.compute_density_from_wavefunctions()?;
        let mut next: ScfIteration<S, DensityUpdated<MixingOff>, MixingOff> =
            self.into_phase();
        next.density = new_density;
        Ok(next)
    }

    /// Construct density and transition the history to `Kerker` phase.
    pub fn construct_density_kerker(
        self,
    ) -> Result<ScfIteration<S, DensityUpdated<Kerker>, Kerker>, Error> {
        let new_density = self.compute_density_from_wavefunctions()?;
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
            _phase: PhantomData,
        })
    }

    /// Construct density and transition the history to `Pulay` phase.
    pub fn construct_density_pulay(
        self,
    ) -> Result<ScfIteration<S, DensityUpdated<Pulay>, Pulay>, Error> {
        let new_density = self.compute_density_from_wavefunctions()?;
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
        let occupations =
            crate::density::compute_occupations(&self.eigenvalues, &self.smearing, n_electrons)?;

        // 2. Total energy assembly (if energy components are available)
        if let (Some(e_xc), Some(e_hartree), Some(rho_vxc)) =
            (self.e_xc, self.e_hartree, self.rho_vxc)
        {
            let e_total = crate::energy::assemble_total_energy(
                &self.eigenvalues,
                &occupations,
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
        let energy_converged = if self.energy_buffer.len() >= n_conv {
            let window = &self.energy_buffer[self.energy_buffer.len() - n_conv..];
            window
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .all(|diff| diff < energy_tol)
        } else {
            false
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

        // 5. Decision
        let next_mixing = self.next_mixing;
        if dens_rms < tol && energy_converged {
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
            let wfn = v_eff.diagonalize(ndeg)?;

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
pub fn run_scf_with_energy(
    state: ScfIteration<NonSpin, Initialized, MixingOff>,
    ndeg: usize,
    tol: f64,
) -> Result<FinalResult, Error> {
    let mut state = state;
    loop {
        state = {
            let v_eff = state.build_v_eff_with_energy()?;
            let wfn = v_eff.diagonalize(ndeg)?;

            // Dispatch mixing phase at runtime
            let mixed = match wfn.next_mixing {
                MixingPhaseKind::Off => wfn.construct_density_off()?.mix(),
                MixingPhaseKind::Kerker => wfn.construct_density_kerker()?.mix(),
                MixingPhaseKind::Pulay => wfn.construct_density_pulay()?.mix(),
            };

            match mixed.check(tol)? {
                CheckOutcome::Converged(converged) => return Ok(converged.finalize()),
                CheckOutcome::NotConverged { state: next, next_mixing } => {
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

/// Convert fractional PW G-vector coordinates to cuFFT C-order linear indices.
///
/// Each entry `[h, k, l]` is a fractional G-vector from `KptWaveBlock`.
/// The output index is `ix + ngx * (iy + ngy * iz)` (cuFFT C-order, nx fastest).
#[allow(dead_code)]  // Used by test fixture infrastructure (Group F)
pub(crate) fn pw_coords_to_fft_indices(
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
            (ix + ngx * (iy + ngy * iz)) as i32
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
        }
    }

    fn dummy_grid() -> GVectorGrid {
        GVectorGrid::new(
            [4, 4, 4],
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
}
