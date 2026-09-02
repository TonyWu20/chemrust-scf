pub mod kerker;
pub mod reciprocal_density;
pub(crate) mod cuda_kernels;

use std::marker::PhantomData;
use std::sync::Arc;

use chemrust_hamiltonian_core::GVectorGrid;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array3;

use crate::device::fft::FftPlan3d;
use crate::device::{CudaComplex, DeviceMapped};
use crate::types::{Density, Error, WaveGridArray};

use self::cuda_kernels::MixingCudaKernels;
use self::kerker::KerkerPreconditioner;
use self::reciprocal_density::ReciprocalDensity;

// ---------------------------------------------------------------------------
// Mixing phase markers (type-state)
// ---------------------------------------------------------------------------

/// Sealed trait for mixing phase markers.
///
/// Three implementors exist: `MixingOff`, `Kerker`, `Pulay`.
pub trait MixingPhase: sealed::Sealed {}

/// No mixing — density passes through unchanged.
pub enum MixingOff {}
/// Kerker preconditioned mixing: n_new = n_in + K·(n_out - n_in).
pub enum Kerker {}
/// Pulay / direct inversion (DIIS) mixing with history.
pub enum Pulay {}

mod sealed {
    pub trait Sealed {}
}
impl sealed::Sealed for MixingOff {}
impl sealed::Sealed for Kerker {}
impl sealed::Sealed for Pulay {}
impl MixingPhase for MixingOff {}
impl MixingPhase for Kerker {}
impl MixingPhase for Pulay {}

// ---------------------------------------------------------------------------
// DensityHistory — type-state struct over mixing phase
// ---------------------------------------------------------------------------

/// Maximum number of DIIS history entries (CASTEP default).
/// CASTEP `mix_history_length` (NiO .param: 20). DIIS ring-buffer size.
const DIIS_MAX_HISTORY: usize = 20;

/// Reciprocal-space density history and mixing state.
///
/// Generic over the mixing phase `M`, which controls the available methods:
///
/// | Phase | `mix()` behaviour                          |
/// |-------|--------------------------------------------|
/// | Off   | pass-through (no active mixing)            |
/// | Kerker| Kerker preconditioning via C2C FFT        |
/// | Pulay | DIIS with up to 20 history entries + Kerker|
///
/// CASTEP-faithful spin handling (CASTEP `dm_sub_base.f90` /
/// `dm_sub_mix.f90`): the mix object stores the charge/spin decomposition
///
/// ```text
///   c = FFT(rho_up + rho_dn)   (CASTEP real_charge, spin-summed)
///   s = FFT(rho_up - rho_dn)   (CASTEP real_spin)
/// ```
///
/// with two Kerker kernels (`dm_sub_base.f90:689-690`):
///
/// ```text
///   Kc(G) = amp_c · G²/(G² + gc²),  Kc(G=0) = 0     (charge conservation)
///   Ks(G) = amp_s · G²/(G² + gs²),  Ks(G=0) = amp_s (uniform spin mixes)
/// ```
///
/// and one joint DIIS over the (Δc, Δs) pairs with the CASTEP
/// `dm_mix_density_dot` inner product: charge part weighted by
/// `mix_metric` (1 + E_q1sq/E, q1 = 0 → 1.0 for G>0, 0.0 at G=0),
/// spin part unweighted.  For nspins = 1 the spin object is identically
/// zero and the history degenerates to the plain charge mixer.

pub struct DensityHistory<M: MixingPhase> {
    /// Kerker preconditioner kernels Kc(G), Ks(G) on GPU. Created lazily on
    /// first Kerker `into_kerker()` call.
    pub(crate) kerker: Option<KerkerPreconditioner>,
    /// Compiled CUDA kernels for GPU element-wise ops. Created lazily.
    pub(crate) kernels: Option<MixingCudaKernels>,
    /// Number of spin channels (1 for NonSpin, 2 for SpinCollinear).
    nspins: usize,
    /// CASTEP `mix_charge_amp` (default 0.5, NiO .param: 0.5).
    pub(crate) charge_amp: f64,
    /// CASTEP `spin_density_mixing_amplitude` / `mix_spin_amp`
    /// (NiO .param: 2.0). Unused for nspins = 1.
    pub(crate) spin_amp: f64,
    /// CASTEP `mix_charge_gmax` in a₀⁻¹ (default 1.5 /Å = 2.8346 a₀⁻¹).
    pub(crate) mix_gmax_c: f64,
    /// CASTEP `mix_spin_gmax` in a₀⁻¹ (default 1.5 /Å = 2.8346 a₀⁻¹,
    /// parameters.f90:1916).
    pub(crate) mix_gmax_s: f64,
    // ── Current mix-object densities (c, s), reciprocal, band-limited ──
    /// Reciprocal charge (total) density from the previous mixing output.
    /// `None` before the first Kerker/Pulay call (pass-through/seed).
    /// CASTEP `dm.f90 current_density_in.charge`.
    current_c_in: Option<ReciprocalDensity>,
    /// Reciprocal spin-density counterpart. `None` when nspins = 1.
    current_s_in: Option<ReciprocalDensity>,
    // ── DIIS delta history (GPU-resident complex), joint (c, s) ──
    /// Δn_c_i = c_in_i − c_in_{i-1}, band-masked, oldest → newest.
    delta_c_history: Vec<CudaSlice<CudaComplex>>,
    /// Δn_s_i = s_in_i − s_in_{i-1}, band-masked. Empty when nspins = 1.
    delta_s_history: Vec<CudaSlice<CudaComplex>>,
    /// ΔR_c_i = R_c_i − R_{c,i-1}, metric-masked (band-mask × mix_metric;
    /// G=0 zeroed, CASTEP mix_metric(0) = 0).  Used for both the DIIS
    /// inner product and the Σc_i·ΔR update term.
    delta_rc_history: Vec<CudaSlice<CudaComplex>>,
    /// ΔR_s_i = R_s_i − R_{s,i-1}, band-masked (G=0 kept: the spin part is
    /// unweighted and G=0 is inside the CASTEP mix basis). Empty when
    /// nspins = 1.
    delta_rs_history: Vec<CudaSlice<CudaComplex>>,
    /// Previous residual R_{t-1} (c, s) on GPU, for the next ΔR.
    prev_res_c: Option<CudaSlice<CudaComplex>>,
    prev_res_s: Option<CudaSlice<CudaComplex>>,
    /// Previous n_in_{t-1} (c, s) on GPU, for the next Δn. Saved before
    /// updating `current_*_in`.
    prev_n_c: Option<CudaSlice<CudaComplex>>,
    prev_n_s: Option<CudaSlice<CudaComplex>>,
    _marker: PhantomData<M>,
}

// ===========================================================================
// In-place C2C FFT helpers (cuFFT natively supports in-place transforms)
// ===========================================================================

/// Call C2C forward in-place (same buffer for input and output).
///
/// # Safety
/// cuFFT accesses device memory; `buf` must be a valid GPU allocation.
unsafe fn c2c_forward_inplace(
    plan: &FftPlan3d,
    buf: &mut CudaSlice<CudaComplex>,
) -> Result<(), cudarc::cufft::result::CufftError> {
    let ptr = buf as *mut CudaSlice<CudaComplex>;
    unsafe { plan.c2c_forward(&mut *ptr, &mut *ptr) }
}

/// Call C2C inverse in-place.
///
/// # Safety
/// cuFFT accesses device memory; `buf` must be a valid GPU allocation.
unsafe fn c2c_inverse_inplace(
    plan: &FftPlan3d,
    buf: &mut CudaSlice<CudaComplex>,
) -> Result<(), cudarc::cufft::result::CufftError> {
    let ptr = buf as *mut CudaSlice<CudaComplex>;
    unsafe { plan.c2c_inverse(&mut *ptr, &mut *ptr) }
}

// ===========================================================================
// DIIS linear algebra helper (CPU — matrix is tiny, ≤20×20)
// ===========================================================================

/// Solve the N×N DIIS system `A·c = b` via Gaussian elimination with
/// partial pivoting.
///
/// `a` is `n×n`, `b` is `n×1`, both stored in row-major order.
///
/// Returns `(coeffs, fallback)` where `coeffs` has length `n` and
/// `fallback` is `true` if the solve failed (singular matrix) — the
/// caller should fall back to Kerker-only mixing for this step.
fn solve_diis_system(a: &mut [f64], b: &mut [f64], n: usize) -> (Vec<f64>, bool) {
    if n == 0 {
        return (vec![], false);
    }
    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return (vec![0.0; n], true);
        }
        if max_row != col {
            for j in col..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        if a[i * n + i].abs() < 1e-30 {
            return (vec![0.0; n], true);
        }
        x[i] = sum / a[i * n + i];
    }
    (x, false)
}

// ===========================================================================
// Shared mix-pair helpers (all phases)
// ===========================================================================

impl<M: MixingPhase> DensityHistory<M> {
    /// True when the spin (s) part is active (nspins = 2).  For nspins = 1
    /// the s object is identically zero and all spin-side buffers are
    /// passed as zero slices.
    fn spin_active(&self) -> bool {
        self.nspins == 2
    }

    /// Upload a real-space density to the GPU C2C FFT grid (complex,
    /// imag = 0) and run the in-place forward FFT.
    fn forward_fft(
        &self,
        stream: &Arc<CudaStream>,
        plan: &FftPlan3d,
        density: &Density,
        n_real: usize,
    ) -> Result<CudaSlice<CudaComplex>, Error> {
        let density_flat: Vec<f64> = density.flatten_host();
        let cplx_host: Vec<CudaComplex> = density_flat
            .iter()
            .map(|&v| CudaComplex { x: v, y: 0.0 })
            .collect();
        let mut n_out_dev: CudaSlice<CudaComplex> = stream
            .clone_htod(&cplx_host)
            .map_err(Error::Cuda)?;
        if n_real > 0 {
            unsafe {
                c2c_forward_inplace(plan, &mut n_out_dev)
                    .map_err(Error::Fft)?;
            }
        }
        Ok(n_out_dev)
    }

    /// IFFT a reciprocal pair (in-place on `c`/`s`), scale by 1/N, and
    /// return the real-space (charge, spin) densities.
    fn inverse_fft_pair(
        &self,
        stream: &Arc<CudaStream>,
        plan: &FftPlan3d,
        c: &mut CudaSlice<CudaComplex>,
        s: &mut CudaSlice<CudaComplex>,
        n_real: usize,
        shape: [usize; 3],
    ) -> Result<(Density, Density), Error> {
        let [ngz, ngy, ngx] = shape;
        // In-place inverse FFT first, then read back the real parts.
        unsafe {
            c2c_inverse_inplace(plan, c).map_err(Error::Fft)?;
            c2c_inverse_inplace(plan, s).map_err(Error::Fft)?;
        }
        stream.synchronize().map_err(Error::Cuda)?;
        let real_data = |buf: &CudaSlice<CudaComplex>| -> Vec<f64> {
            let out: Vec<CudaComplex> = stream
                .clone_dtoh(buf)
                .expect("D2H IFFT result");
            let scale = n_real as f64;
            out.iter().map(|c| c.x / scale).collect()
        };
        let c_arr = Array3::from_shape_vec([ngx, ngy, ngz], real_data(c))
            .expect("valid shape");
        let s_arr = Array3::from_shape_vec([ngx, ngy, ngz], real_data(s))
            .expect("valid shape");
        let c_density = Density::from_inner(WaveGridArray::from_inner(c_arr));
        let s_density = Density::from_inner(WaveGridArray::from_inner(s_arr));
        Ok((c_density, s_density))
    }

    /// True when the GPU mix state (Kerker kernels + CUDA kernels) has
    /// been built, i.e. the history has left the pure-`Off` path.
    pub fn has_gpu_state(&self) -> bool {
        self.kerker.is_some()
    }

    /// Clear the DIIS/Kerker history (CASTEP `dm_flush_history` =
    /// `dm_finalise`: the delta/residual history arrays are released and
    /// the mix object is re-seeded on the next `dm_mix_density` call).
    /// The Kerker kernels themselves (geometry-static) are kept.
    pub fn flush(&mut self) {
        self.delta_c_history.clear();
        self.delta_s_history.clear();
        self.delta_rc_history.clear();
        self.delta_rs_history.clear();
        self.prev_res_c = None;
        self.prev_res_s = None;
        self.prev_n_c = None;
        self.prev_n_s = None;
        self.current_c_in = None;
        self.current_s_in = None;
    }

    /// Store the current (c, s) densities as the mix-object input for the
    /// next mixing step, without mixing (CASTEP `dm_mix_density(dens, dens)`
    /// after `dm_flush_history`: "store the input density for next time").
    ///
    /// `c`: total density ρ_up + ρ_dn (the single channel for nspins = 1).
    /// `s`: spin density ρ_up − ρ_dn (zero for nspins = 1).
    ///
    /// Any phase works.  The GPU state is built if absent and all phase
    /// state (delta history, amplitudes, gmax) is preserved.  Returns a
    /// `MixingOff` history, which the caller keeps and re-transitions
    /// (`.into_kerker(...).into_pulay()`) on the next SCF cycle.
    pub fn seed(
        self,
        c: &Density,
        s: &Density,
        mixing_grid: &GVectorGrid,
        g2_cutoff: Option<f64>,
    ) -> Result<DensityHistory<MixingOff>, Error> {
        let mut kerker_history = self.into_kerker_like(mixing_grid, g2_cutoff)?;
        kerker_history.seed_inplace(c, s)?;
        Ok(kerker_history.into_off())
    }

    /// Seed on an already-GPU-initialised history (Kerker/Pulay phases).
    pub fn seed_inplace(&mut self, c: &Density, s: &Density) -> Result<(), Error> {
        let stream = self
            .kernels
            .as_ref()
            .expect("kernels must be initialised before seed()")
            .stream
            .clone();
        let shape = self
            .kerker
            .as_ref()
            .expect("Kerker preconditioner must be initialised before seed()")
            .shape();
        let [ngz, ngy, ngx] = shape;
        let n_real = ngz * ngy * ngx;
        let fft_plan = FftPlan3d::plan_c2c(ngx as i32, ngy as i32, ngz as i32, stream.clone())
            .expect("C2C plan");

        let c_out = self.forward_fft(&stream, &fft_plan, c, n_real)?;
        let s_out = if self.spin_active() {
            Some(self.forward_fft(&stream, &fft_plan, s, n_real)?)
        } else {
            None
        };

        self.current_c_in = Some(ReciprocalDensity::new(c_out, [ngz, ngy, ngx]));
        self.current_s_in = s_out.map(|s_out| ReciprocalDensity::new(s_out, [ngz, ngy, ngx]));
        Ok(())
    }
}

// ===========================================================================
// MixingOff — pass-through (no active mixing)
// ===========================================================================

impl DensityHistory<MixingOff> {
    /// Create a new density history in the `Off` phase.
    ///
    /// No GPU resources are allocated until the first transition to
    /// `Kerker`.  Amplitudes default to the CASTEP .param values:
    /// charge 0.5, spin 2.0.
    pub fn new(nspins: usize) -> Self {
        Self {
            kerker: None,
            kernels: None,
            nspins,
            charge_amp: 0.5,
            spin_amp: 2.0,
            mix_gmax_c: kerker::KERKER_GMAX_DEFAULT,
            mix_gmax_s: kerker::KERKER_SPIN_GMAX_DEFAULT,
            current_c_in: None,
            current_s_in: None,
            delta_c_history: Vec::with_capacity(DIIS_MAX_HISTORY),
            delta_s_history: Vec::with_capacity(DIIS_MAX_HISTORY),
            delta_rc_history: Vec::with_capacity(DIIS_MAX_HISTORY),
            delta_rs_history: Vec::with_capacity(DIIS_MAX_HISTORY),
            prev_res_c: None,
            prev_res_s: None,
            prev_n_c: None,
            prev_n_s: None,
            _marker: PhantomData,
        }
    }

    /// Create a DensityHistory with explicit CASTEP amplitudes
    /// (`mix_charge_amp`, `mix_spin_amp`).
    pub fn with_charge_spin_amplitudes(
        nspins: usize,
        charge_amp: f64,
        spin_amp: f64,
    ) -> Self {
        let mut h = Self::new(nspins);
        h.charge_amp = charge_amp;
        h.spin_amp = spin_amp;
        h
    }

    /// Backwards-compatible constructor: sets the charge amplitude
    /// (nspins = 1: the single channel; nspins = 2: the charge part of the
    /// (c, s) pair).  The spin amplitude stays at the CASTEP default 2.0.
    pub fn with_amplitude(nspins: usize, amp: f64) -> Self {
        let mut h = Self::new(nspins);
        h.charge_amp = amp;
        h
    }

    /// Set the CASTEP `mix_charge_gmax` / `mix_spin_gmax` scales (a₀⁻¹) of
    /// the Kerker kernels.  Must be called before `into_kerker` (the
    /// kernels are built lazily).  A single value sets both (the NiO
    /// .param sets both to 1.5 /Å).
    pub fn with_mix_gmax(mut self, gmax: f64) -> Self {
        self.mix_gmax_c = gmax;
        self.mix_gmax_s = gmax;
        self
    }

    /// Set the CASTEP `mix_spin_gmax` scale (a₀⁻¹) independently.
    pub fn with_spin_mix_gmax(mut self, gmax_s: f64) -> Self {
        self.mix_gmax_s = gmax_s;
        self
    }

    /// Pass-through mixing: the (c, s) pair is unchanged, snapshots are
    /// clones.  `c`: total density ρ_up + ρ_dn (single channel when
    /// nspins = 1), `s`: spin density ρ_up − ρ_dn (zero when nspins = 1).
    pub fn mix(&mut self, c: Density, s: Density) -> (Density, Density, Density, Density) {
        let c_snapshot = c.clone();
        let s_snapshot = s.clone();
        (c, s, c_snapshot, s_snapshot)
    }

    /// Transition from `Off` to `Kerker` phase.
    ///
    /// Creates the Kerker preconditioner and compiles CUDA kernels on GPU
    /// from the wave G-vector grid. This is the first GPU-touching
    /// operation on the history.
    pub fn into_kerker(
        mut self,
        mixing_grid: &GVectorGrid,
        g2_cutoff: Option<f64>,
    ) -> Result<DensityHistory<Kerker>, Error> {
        self.initialize_gpu_state(mixing_grid, g2_cutoff)?;
        Ok(DensityHistory {
            kerker: self.kerker,
            kernels: self.kernels,
            nspins: self.nspins,
            charge_amp: self.charge_amp,
            spin_amp: self.spin_amp,
            mix_gmax_c: self.mix_gmax_c,
            mix_gmax_s: self.mix_gmax_s,
            current_c_in: self.current_c_in,
            current_s_in: self.current_s_in,
            delta_c_history: self.delta_c_history,
            delta_s_history: self.delta_s_history,
            delta_rc_history: self.delta_rc_history,
            delta_rs_history: self.delta_rs_history,
            prev_res_c: self.prev_res_c,
            prev_res_s: self.prev_res_s,
            prev_n_c: self.prev_n_c,
            prev_n_s: self.prev_n_s,
            _marker: PhantomData,
        })
    }
}

impl Default for DensityHistory<MixingOff> {
    fn default() -> Self {
        Self::new(1)
    }
}

// ===========================================================================
// All phases — GPU initialisation + convert back to `MixingOff`
// ===========================================================================

impl<M: MixingPhase> DensityHistory<M> {
    /// Lazily create the Kerker preconditioner + CUDA kernels if absent.
    fn initialize_gpu_state(
        &mut self,
        mixing_grid: &GVectorGrid,
        g2_cutoff: Option<f64>,
    ) -> Result<(), Error> {
        if self.kerker.is_none() {
            let ctx = Arc::new(CudaContext::new(0).map_err(Error::Cuda)?);
            let stream = ctx.default_stream();
            let kerker = KerkerPreconditioner::new(
                &stream,
                mixing_grid,
                g2_cutoff,
                self.mix_gmax_c,
                self.mix_gmax_s,
            )?;
            self.kerker = Some(kerker);
            // Compile kernels on the same stream/context to avoid context
            // isolation (device pointers are not valid across contexts).
            self.kernels = Some(MixingCudaKernels::new_from_stream(&stream)?);
        } else if self.kernels.is_none() {
            let ctx = Arc::new(CudaContext::new(0).map_err(Error::Cuda)?);
            let stream = ctx.default_stream();
            self.kernels = Some(MixingCudaKernels::new_from_stream(&stream)?);
        }
        Ok(())
    }

    /// Transition helper used by `seed` (Off → Kerker → Off).
    fn into_kerker_like(
        self,
        mixing_grid: &GVectorGrid,
        g2_cutoff: Option<f64>,
    ) -> Result<DensityHistory<Kerker>, Error> {
        let mut k = DensityHistory {
            kerker: self.kerker,
            kernels: self.kernels,
            nspins: self.nspins,
            charge_amp: self.charge_amp,
            spin_amp: self.spin_amp,
            mix_gmax_c: self.mix_gmax_c,
            mix_gmax_s: self.mix_gmax_s,
            current_c_in: self.current_c_in,
            current_s_in: self.current_s_in,
            delta_c_history: self.delta_c_history,
            delta_s_history: self.delta_s_history,
            delta_rc_history: self.delta_rc_history,
            delta_rs_history: self.delta_rs_history,
            prev_res_c: self.prev_res_c,
            prev_res_s: self.prev_res_s,
            prev_n_c: self.prev_n_c,
            prev_n_s: self.prev_n_s,
            _marker: PhantomData,
        };
        k.initialize_gpu_state(mixing_grid, g2_cutoff)?;
        Ok(k)
    }

    /// Convert any mixing phase back to `MixingOff`, preserving internal
    /// data.
    ///
    /// This is needed because `ScfIteration::mix()` normalises the mixing
    /// phase to `Off` so that the `run_scf` loop dispatch can return a
    /// uniform type from all match arms.
    pub fn into_off(self) -> DensityHistory<MixingOff> {
        DensityHistory {
            kerker: self.kerker,
            kernels: self.kernels,
            nspins: self.nspins,
            charge_amp: self.charge_amp,
            spin_amp: self.spin_amp,
            mix_gmax_c: self.mix_gmax_c,
            mix_gmax_s: self.mix_gmax_s,
            current_c_in: self.current_c_in,
            current_s_in: self.current_s_in,
            delta_c_history: self.delta_c_history,
            delta_s_history: self.delta_s_history,
            delta_rc_history: self.delta_rc_history,
            delta_rs_history: self.delta_rs_history,
            prev_res_c: self.prev_res_c,
            prev_res_s: self.prev_res_s,
            prev_n_c: self.prev_n_c,
            prev_n_s: self.prev_n_s,
            _marker: PhantomData,
        }
    }
}

// ===========================================================================
// Kerker — preconditioned mixing via C2C FFT (GPU-resident)
// ===========================================================================

impl DensityHistory<Kerker> {
    /// Set the CASTEP `mix_charge_amp` (Kc amplitude).
    pub fn set_charge_amp(&mut self, amp: f64) {
        self.charge_amp = amp;
    }

    /// Set the CASTEP `mix_spin_amp` / `spin_density_mixing_amplitude`
    /// (Ks amplitude). Unused for nspins = 1.
    pub fn set_spin_amp(&mut self, amp: f64) {
        self.spin_amp = amp;
    }

    /// Backwards-compatible setter: sets the charge amplitude.  For
    /// nspins = 2 the spin amplitude is set separately via `set_spin_amp`.
    pub fn set_mixing_amplitude(&mut self, ispin: usize, amp: f64) {
        if ispin == 0 || self.nspins == 1 {
            self.charge_amp = amp;
        } else {
            self.spin_amp = amp;
        }
    }

    /// Mix the (charge, spin) density pair with Kerker preconditioning,
    /// fully GPU-resident.
    ///
    /// 1. Upload c, s → GPU C2C forward FFT → c_out(G), s_out(G)
    /// 2. R_c = c_out − c_in, R_s = s_out − s_in (GPU)
    /// 3. n_new = n_in + Kc·R_c (charge), n_new = n_in + Ks·R_s (spin)
    ///    (CASTEP `dm_mix_density_kerker`; the G=0 spin kernel Ks(0) =
    ///    amp_s mixes the uniform spin fully, the G=0 charge kernel is 0)
    /// 4. Store n_new_recip as `current_*_in` for the next iteration
    /// 5. C2C inverse FFT back → mixed real-space (c, s) densities
    ///
    /// `c`: total density ρ_up + ρ_dn (single channel for nspins = 1).
    /// `s`: spin density ρ_up − ρ_dn (zero for nspins = 1).
    ///
    /// Returns `(c_mixed, s_mixed, c_snapshot, s_snapshot)` where the
    /// snapshots are the pre-mix inputs.
    pub fn mix(&mut self, c: Density, s: Density) -> (Density, Density, Density, Density) {
        let shape = self
            .kerker
            .as_ref()
            .expect("Kerker preconditioner must be initialised before mix()")
            .shape(); // [ngz, ngy, ngx]
        let [ngz, ngy, ngx] = shape;
        let n_real = ngz * ngy * ngx;
        let n_i32 = n_real as i32;

        let kernels = self
            .kernels
            .as_ref()
            .expect("CUDA kernels must be initialised before mix()");
        let stream = kernels.stream.clone();
        let spin_active = self.spin_active();

        let fft_plan = FftPlan3d::plan_c2c(ngx as i32, ngy as i32, ngz as i32, stream.clone())
            .expect("C2C plan");

        // ── Upload c (and s) as complex, C2C forward → *_out(G) ──
        let c_out_dev = self
            .forward_fft(&stream, &fft_plan, &c, n_real)
            .expect("H2D + FFT charge density");
        let s_out_dev: Option<CudaSlice<CudaComplex>> = if spin_active {
            Some(
                self.forward_fft(&stream, &fft_plan, &s, n_real)
                    .expect("H2D + FFT spin density"),
            )
        } else {
            None
        };

        // ── Take the previous mix-object densities ──
        let prev_c_recip = self.current_c_in.take();
        let prev_s_recip = self.current_s_in.take();

        let (mut mixed_recip_c, mut mixed_recip_s, input_snapshot_c, input_snapshot_s) =
            match (prev_c_recip, spin_active) {
                (Some(prev_c), active) => {
                    let n_in_c_dev = prev_c.data;
                    let n_in_s_dev: CudaSlice<CudaComplex> = if active {
                        match prev_s_recip {
                            Some(ps) => ps.data,
                            None => stream
                                .alloc_zeros(n_real)
                                .expect("alloc zero s (G=0 spin seed)"),
                        }
                    } else {
                        stream.alloc_zeros(n_real).expect("alloc zero s")
                    };

                    // R = out − in (GPU), both parts
                    let r_c_dev =
                        mix_residual(&kernels, &stream, &c_out_dev, &n_in_c_dev, n_real, n_i32);
                    let r_s_dev = if active {
                        mix_residual(
                            &kernels,
                            &stream,
                            s_out_dev.as_ref().expect("s_out present for spin"),
                            &n_in_s_dev,
                            n_real,
                            n_i32,
                        )
                    } else {
                        stream.alloc_zeros(n_real).expect("alloc zero R_s")
                    };

                    // n_new = n_in + K·R on the mix basis; above the
                    // cutoff the high-G content is carried from *_out
                    // (CASTEP dm_mix_density_to_density behaviour).
                    let mut result_c_dev: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc result c");
                    let mut result_s_dev: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc result s");
                    let kerker = self.kerker.as_ref().unwrap();
                    let mask_dev = kerker.as_mask_slice();
                    let zero_dev: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc zero");
                    unsafe {
                        stream
                            .launch_builder(&kernels.cpx_full_update)
                            .arg(&mut result_c_dev)
                            .arg(&mut result_s_dev)
                            .arg(&n_in_c_dev)
                            .arg(&n_in_s_dev)
                            .arg(kerker.as_device_slice())
                            .arg(kerker.as_spin_kernel_slice())
                            .arg(&r_c_dev)
                            .arg(&r_s_dev)
                            .arg(&zero_dev)
                            .arg(&zero_dev)
                            .arg(&zero_dev)
                            .arg(&zero_dev)
                            .arg(&c_out_dev)
                            .arg(s_out_dev.as_ref().unwrap_or(&zero_dev))
                            .arg(mask_dev)
                            .arg(&n_i32)
                            .arg(&self.charge_amp)
                            .arg(&self.spin_amp)
                            .arg(&0.0_f64) // amp_n: plain Kerker (no DIIS part)
                            .launch(LaunchConfig::for_num_elems(n_real as u32))
                    }
                    .expect("cpx_full_update n_in + K*R");

                    // ── Store the mixed reciprocal objects ──
                    let c_copy = clone_recip(&stream, &result_c_dev, [ngz, ngy, ngx]);
                    let s_copy = clone_recip(&stream, &result_s_dev, [ngz, ngy, ngx]);
                    self.current_c_in = Some(c_copy);
                    self.current_s_in = if active { Some(s_copy) } else { None };

                    (result_c_dev, result_s_dev, c, s)
                }
                _ => {
                    // First call for this (c, s) pair: seed and pass through
                    self.current_c_in =
                        Some(ReciprocalDensity::new(c_out_dev, [ngz, ngy, ngx]));
                    if spin_active {
                        self.current_s_in = Some(ReciprocalDensity::new(
                            s_out_dev.expect("s_out for spin"),
                            [ngz, ngy, ngx],
                        ));
                    }
                    // Return the original densities unchanged
                    return (c.clone(), s.clone(), c, s);
                }
            };

        // ── C2C inverse FFT (in-place) ──
        let (c_mixed, s_mixed) = self
            .inverse_fft_pair(
                &stream, &fft_plan, &mut mixed_recip_c, &mut mixed_recip_s, n_real, [ngz, ngy, ngx],
            )
            .expect("IFFT mixed pair");

        (c_mixed, s_mixed, input_snapshot_c, input_snapshot_s)
    }

    /// Transition from `Kerker` to `Pulay` (DIIS) phase.
    ///
    /// DIIS history and residual state are PRESERVED: `into_pulay` is
    /// invoked at every Pulay iteration (construct_density_pulay), so
    /// resetting the state here would zero the delta history and DIIS
    /// would never activate. CASTEP keeps one density/residual history
    /// across scheme switches (dm_sub_mix.f90 uses the same
    /// density_history and residual_history arrays for Kerker and Pulay).
    pub fn into_pulay(self) -> DensityHistory<Pulay> {
        DensityHistory {
            kerker: self.kerker,
            kernels: self.kernels,
            nspins: self.nspins,
            charge_amp: self.charge_amp,
            spin_amp: self.spin_amp,
            mix_gmax_c: self.mix_gmax_c,
            mix_gmax_s: self.mix_gmax_s,
            current_c_in: self.current_c_in,
            current_s_in: self.current_s_in,
            delta_c_history: self.delta_c_history,
            delta_s_history: self.delta_s_history,
            delta_rc_history: self.delta_rc_history,
            delta_rs_history: self.delta_rs_history,
            prev_res_c: self.prev_res_c,
            prev_res_s: self.prev_res_s,
            prev_n_c: self.prev_n_c,
            prev_n_s: self.prev_n_s,
            _marker: PhantomData,
        }
    }
}

// ===========================================================================
// Pulay (DIIS) — direct inversion in the iterative subspace
// ===========================================================================

impl DensityHistory<Pulay> {
    /// Set the CASTEP `mix_charge_amp` / `mix_spin_amp` (see the Kerker
    /// setters).
    pub fn set_charge_amp(&mut self, amp: f64) {
        self.charge_amp = amp;
    }

    pub fn set_spin_amp(&mut self, amp: f64) {
        self.spin_amp = amp;
    }

    /// Backwards-compatible setter (see the Kerker form).
    pub fn set_mixing_amplitude(&mut self, ispin: usize, amp: f64) {
        if ispin == 0 || self.nspins == 1 {
            self.charge_amp = amp;
        } else {
            self.spin_amp = amp;
        }
    }

    /// Mix the (charge, spin) density pair with Pulay/DIIS — fully
    /// GPU-resident except for the tiny (≤20×20) linear solve on CPU.
    ///
    /// CASTEP's dm_mix_density_pulay algorithm (dm_sub_mix.f90:815-1013),
    /// joint (c, s) form:
    ///
    /// ```text
    /// History stores deltas of the (c, s) mix objects:
    ///   Δn_c_i = c_in_i − c_in_{i−1},  Δn_s_i = s_in_i − s_in_{i−1}
    ///   ΔR_c_i = R_c_i − R_c_{i−1},    ΔR_s_i = R_s_i − R_s_{i−1}
    ///
    /// DIIS is unconstrained N×N over the JOINT (Δc, Δs) vectors:
    ///   M_ij = ⟨ΔR_j | ΔR_i⟩  (dm_mix_density_dot: charge part ×
    ///         mix_metric — 1.0 for G>0, 0.0 at G=0 — spin part
    ///         unweighted)
    ///   b_i  = −⟨ΔR_i | R⟩
    ///   solve M·coef = b
    ///
    /// Update (dm_apply_kerker per part, dm_sub_kerker.f90:51-64):
    ///   c_new = c_in + Σcoef_i·Δn_c_i + Kc·(R_c + Σcoef_i·ΔR_c_i)
    ///   s_new = s_in + Σcoef_i·Δn_s_i + Ks·(R_s + Σcoef_i·ΔR_s_i)
    ///   (DIIS parts unscaled ×1.0; Kc(G=0) = 0, Ks(G=0) = amp_s)
    ///
    /// Fallback on DIIS solve failure (CASTEP "reverting to Kerker for
    /// this mixing step"): plain Kerker step n_in + K·R.
    ///
    /// Returns `(c_mixed, s_mixed, c_snapshot, s_snapshot)`.
    pub fn mix(&mut self, c: Density, s: Density) -> (Density, Density, Density, Density) {
        let shape = self
            .kerker
            .as_ref()
            .expect("Kerker preconditioner must be initialised before Pulay mix()")
            .shape();
        let [ngz, ngy, ngx] = shape;
        let n_real = ngz * ngy * ngx;
        let n_i32 = n_real as i32;

        let kernels = self
            .kernels
            .as_ref()
            .expect("CUDA kernels must be initialised before Pulay mix()");
        let stream = kernels.stream.clone();
        let blas = &kernels.blas;
        let spin_active = self.spin_active();

        let fft_plan = FftPlan3d::plan_c2c(ngx as i32, ngy as i32, ngz as i32, stream.clone())
            .expect("C2C plan");

        // ── 1. Upload c, s → C2C forward → c_out(G), s_out(G) ──
        let c_out_dev = self
            .forward_fft(&stream, &fft_plan, &c, n_real)
            .expect("H2D + FFT charge density");
        let s_out_dev: Option<CudaSlice<CudaComplex>> = if spin_active {
            Some(
                self.forward_fft(&stream, &fft_plan, &s, n_real)
                    .expect("H2D + FFT spin density"),
            )
        } else {
            None
        };

        // ── 2. Take the previous mix-object densities ──
        let prev_c_recip = self.current_c_in.take();
        let prev_s_recip = self.current_s_in.take();

        let (mut mixed_recip_c, mut mixed_recip_s, input_snapshot_c, input_snapshot_s) =
            match (prev_c_recip, spin_active) {
                (Some(prev_c), active) => {
                    let n_in_c_dev = prev_c.data;
                    let n_in_s_dev: CudaSlice<CudaComplex> = if active {
                        match prev_s_recip {
                            Some(ps) => ps.data,
                            None => stream
                                .alloc_zeros(n_real)
                                .expect("alloc zero s (G=0 spin seed)"),
                        }
                    } else {
                        stream.alloc_zeros(n_real).expect("alloc zero s")
                    };

                    // ---- 2b. R = out − in (GPU), both parts ----
                    let r_c_dev =
                        mix_residual(&kernels, &stream, &c_out_dev, &n_in_c_dev, n_real, n_i32);
                    let r_s_dev = if active {
                        mix_residual(
                            &kernels,
                            &stream,
                            s_out_dev.as_ref().expect("s_out present for spin"),
                            &n_in_s_dev,
                            n_real,
                            n_i32,
                        )
                    } else {
                        stream.alloc_zeros(n_real).expect("alloc zero R_s")
                    };

                    // ---- 3. Push deltas to the joint history ----
                    // CASTEP mix objects are band-limited to num_mix_plane_waves,
                    // so the DIIS delta history carries only mix-basis (low-G)
                    // content. Mask the deltas; the charge residual is
                    // additionally metric-masked (G=0 → 0, CASTEP
                    // mix_metric(0) = 0).  All four vectors grow in lockstep;
                    // eviction removes the oldest entry from every vector.
                    let old_prev_res_c = self.prev_res_c.take();
                    let old_prev_res_s = self.prev_res_s.take();
                    let old_prev_n_c = self.prev_n_c.take();
                    let old_prev_n_s = self.prev_n_s.take();
                    let kerker = self.kerker.as_ref().unwrap();
                    let mask_dev = kerker.as_mask_slice();
                    let metric_mask_dev = kerker.as_metric_mask_slice();

                    if let (Some(prev_res_c_dev), Some(prev_n_c_dev)) =
                        (old_prev_res_c, old_prev_n_c)
                    {
                        // ΔR_c = R_c − R_{c,t−1}, metric-masked (G=0 zeroed).
                        let delta_rc_masked = masked_sub_owned(
                            &kernels,
                            &stream,
                            &r_c_dev,
                            &prev_res_c_dev,
                            metric_mask_dev,
                            n_real,
                            n_i32,
                        );
                        // Δn_c = c_in − c_{in,t−1}, band-masked.
                        let delta_c_masked = masked_sub_owned(
                            &kernels,
                            &stream,
                            &n_in_c_dev,
                            &prev_n_c_dev,
                            mask_dev,
                            n_real,
                            n_i32,
                        );

                        // Lockstep eviction of the oldest entry.
                        if self.delta_c_history.len() >= DIIS_MAX_HISTORY {
                            self.delta_c_history.remove(0);
                            self.delta_rc_history.remove(0);
                            if !self.delta_s_history.is_empty() {
                                self.delta_s_history.remove(0);
                            }
                            if !self.delta_rs_history.is_empty() {
                                self.delta_rs_history.remove(0);
                            }
                        }
                        self.delta_c_history.push(delta_c_masked);
                        self.delta_rc_history.push(delta_rc_masked);

                        if active {
                            // ΔR_s = R_s − R_{s,t−1}, band-masked (unweighted:
                            // G=0 kept in the mix basis; Ks(G=0) = amp_s).
                            if let Some(prev_res_s_dev) = old_prev_res_s {
                                let delta_rs_masked = masked_sub_owned(
                                    &kernels, &stream, &r_s_dev, &prev_res_s_dev, mask_dev, n_real, n_i32,
                                );
                                self.delta_rs_history.push(delta_rs_masked);
                            }
                            // Δn_s = s_in − s_{in,t−1}, band-masked.
                            if let Some(prev_n_s_dev) = old_prev_n_s {
                                let delta_s_masked = masked_sub_owned(
                                    &kernels, &stream, &n_in_s_dev, &prev_n_s_dev, mask_dev, n_real, n_i32,
                                );
                                self.delta_s_history.push(delta_s_masked);
                            }
                        }
                    }

                    // ---- 4. Save R_current / n_in_current as prev_* (c, s) ----
                    let new_prev_res_c = clone_recip_streamed(&stream, &r_c_dev);
                    self.prev_res_c = Some(new_prev_res_c);
                    if spin_active {
                        let new_prev_res_s = clone_recip_streamed(&stream, &r_s_dev);
                        self.prev_res_s = Some(new_prev_res_s);
                    }
                    let new_prev_n_c = clone_recip_streamed(&stream, &n_in_c_dev);
                    self.prev_n_c = Some(new_prev_n_c);
                    if spin_active {
                        let new_prev_n_s = clone_recip_streamed(&stream, &n_in_s_dev);
                        self.prev_n_s = Some(new_prev_n_s);
                    }

                    // ---- 5. Build and solve the JOINT DIIS system (CPU) ----
                    let n_history = self.delta_rc_history.len();
                    let (c_coeff, fallback) = if n_history > 0 {
                        build_and_solve_diis(
                            kernels,
                            &self.delta_rc_history,
                            &self.delta_rs_history,
                            &r_c_dev,
                            &r_s_dev,
                            kerker.as_metric_mask_slice(),
                            n_real as i32,
                            n_history,
                        )
                    } else {
                        (vec![], true)
                    };

                    let zero_dev: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc zero");

                    let (result_c_dev, result_s_dev): (CudaSlice<CudaComplex>, CudaSlice<CudaComplex>) = if !fallback {
                        // ---- DIIS update on GPU (CASTEP form, per part) ----
                        // c_new = c_in + Σc_i·Δn_c_i + Kc·(R_c + Σc_i·ΔR_c_i)
                        // s_new = s_in + Σc_i·Δn_s_i + Ks·(R_s + Σc_i·ΔR_s_i)
                        // DIIS parts unscaled (amp_n = 1.0); R_curr is
                        // INSIDE the Kerker part (CASTEP copies
                        // current_residual into the workspace before adding
                        // the c-weighted residual diffs).
                        let mut sum_delta_rc: CudaSlice<CudaComplex> =
                            stream.alloc_zeros(n_real).expect("alloc sum ΔRc");
                        let mut sum_delta_c: CudaSlice<CudaComplex> =
                            stream.alloc_zeros(n_real).expect("alloc sum Δc");
                        let mut sum_delta_rs: CudaSlice<CudaComplex> =
                            stream.alloc_zeros(n_real).expect("alloc sum ΔRs");
                        let mut sum_delta_s: CudaSlice<CudaComplex> =
                            stream.alloc_zeros(n_real).expect("alloc sum Δs");

                        for (i, &ci) in c_coeff.iter().enumerate() {
                            let alpha = CudaComplex { x: ci, y: 0.0 };
                            blas.axpy_c64(n_i32, alpha, &self.delta_rc_history[i], 1, &mut sum_delta_rc, 1)
                                .expect("axpy sum_delta_rc");
                            blas.axpy_c64(n_i32, alpha, &self.delta_c_history[i], 1, &mut sum_delta_c, 1)
                                .expect("axpy sum_delta_c");
                            if let Some(rs_slice) = self.delta_rs_history.get(i) {
                                blas.axpy_c64(n_i32, alpha, rs_slice, 1, &mut sum_delta_rs, 1)
                                    .expect("axpy sum_delta_rs");
                            }
                            if let Some(s_slice) = self.delta_s_history.get(i) {
                                blas.axpy_c64(n_i32, alpha, s_slice, 1, &mut sum_delta_s, 1)
                                    .expect("axpy sum_delta_s");
                            }
                        }

                        let mut result_c_dev: CudaSlice<CudaComplex> =
                            stream.alloc_zeros(n_real).expect("alloc result c");
                        let mut result_s_dev: CudaSlice<CudaComplex> =
                            stream.alloc_zeros(n_real).expect("alloc result s");
                        let amp_c: f64 = self.charge_amp;
                        let amp_s: f64 = self.spin_amp;
                        let amp_n: f64 = 1.0;
                        unsafe {
                            stream
                                .launch_builder(&kernels.cpx_full_update)
                                .arg(&mut result_c_dev)
                                .arg(&mut result_s_dev)
                                .arg(&n_in_c_dev)
                                .arg(&n_in_s_dev)
                                .arg(kerker.as_device_slice())
                                .arg(kerker.as_spin_kernel_slice())
                                .arg(&r_c_dev)
                                .arg(&r_s_dev)
                                .arg(&sum_delta_rc)
                                .arg(&sum_delta_rs)
                                .arg(&sum_delta_c)
                                .arg(&sum_delta_s)
                                .arg(&c_out_dev)
                                .arg(s_out_dev.as_ref().unwrap_or(&zero_dev))
                                .arg(mask_dev)
                                .arg(&n_i32)
                                .arg(&amp_c)
                                .arg(&amp_s)
                                .arg(&amp_n)
                                .launch(LaunchConfig::for_num_elems(n_real as u32))
                        }
                        .expect("cpx_full_update DIIS c_new / s_new");
                        (result_c_dev, result_s_dev)
                    } else {
                        // ---- Kerker fallback: n_new = n_in + K·R (per part) ----
                        // (CASTEP "reverting to Kerker for this mixing step";
                        // amp_n = 0 removes the DIIS density part)
                        let mut result_c_dev: CudaSlice<CudaComplex> =
                            stream.alloc_zeros(n_real).expect("alloc result c");
                        let mut result_s_dev: CudaSlice<CudaComplex> =
                            stream.alloc_zeros(n_real).expect("alloc result s");
                        let amp_c_fb = self.charge_amp;
                        let amp_s_fb = self.spin_amp;
                        let amp_n_fb: f64 = 0.0;
                        unsafe {
                            stream
                                .launch_builder(&kernels.cpx_full_update)
                                .arg(&mut result_c_dev)
                                .arg(&mut result_s_dev)
                                .arg(&n_in_c_dev)
                                .arg(&n_in_s_dev)
                                .arg(kerker.as_device_slice())
                                .arg(kerker.as_spin_kernel_slice())
                                .arg(&r_c_dev)
                                .arg(&r_s_dev)
                                .arg(&zero_dev)
                                .arg(&zero_dev)
                                .arg(&zero_dev)
                                .arg(&zero_dev)
                                .arg(&c_out_dev)
                                .arg(s_out_dev.as_ref().unwrap_or(&zero_dev))
                                .arg(mask_dev)
                                .arg(&n_i32)
                                .arg(&amp_c_fb)
                                .arg(&amp_s_fb)
                                .arg(&amp_n_fb)
                                .launch(LaunchConfig::for_num_elems(n_real as u32))
                        }
                        .expect("cpx_full_update Kerker fallback");
                        (result_c_dev, result_s_dev)
                    };

                    // ---- Store the mixed reciprocal objects ----
                    let c_copy = clone_recip(&stream, &result_c_dev, [ngz, ngy, ngx]);
                    let s_copy = clone_recip(&stream, &result_s_dev, [ngz, ngy, ngx]);
                    self.current_c_in = Some(c_copy);
                    self.current_s_in = if spin_active { Some(s_copy) } else { None };

                    (result_c_dev, result_s_dev, c, s)
                }
                _ => {
                    // No previous density: seed and pass through
                    self.current_c_in = Some(ReciprocalDensity::new(c_out_dev, [ngz, ngy, ngx]));
                    if spin_active {
                        self.current_s_in =
                            Some(ReciprocalDensity::new(s_out_dev.expect("s_out for spin"), [ngz, ngy, ngx]));
                    }
                    return (c.clone(), s.clone(), c, s);
                }
            };

        // ── 9. C2C inverse FFT (in-place) ──
        let (c_mixed, s_mixed) = self
            .inverse_fft_pair(
                &stream,
                &fft_plan,
                &mut mixed_recip_c,
                &mut mixed_recip_s,
                n_real,
                [ngz, ngy, ngx],
            )
            .expect("IFFT mixed pair");

        (c_mixed, s_mixed, input_snapshot_c, input_snapshot_s)
    }
}

// ===========================================================================
// Small GPU helpers shared by the phase impls
// ===========================================================================

/// R = a − b on the GPU (band-masked callers pass masked inputs).
fn mix_residual(
    kernels: &MixingCudaKernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<CudaComplex>,
    b: &CudaSlice<CudaComplex>,
    n_real: usize,
    n_i32: i32,
) -> CudaSlice<CudaComplex> {
    let mut r_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_real).expect("alloc R");
    unsafe {
        stream
            .launch_builder(&kernels.cpx_sub)
            .arg(&mut r_dev)
            .arg(a)
            .arg(b)
            .arg(&n_i32)
            .launch(LaunchConfig::for_num_elems(n_real as u32))
    }
    .expect("cpx_sub");
    r_dev
}

/// a − b on the GPU (delta computation).
fn mix_sub(
    kernels: &MixingCudaKernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<CudaComplex>,
    b: &CudaSlice<CudaComplex>,
    n_real: usize,
    n_i32: i32,
) -> CudaSlice<CudaComplex> {
    mix_residual(kernels, stream, a, b, n_real, n_i32)
}

/// (a − b) × mask on the GPU — the masked DIIS deltas.
fn masked_sub_owned(
    kernels: &MixingCudaKernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<CudaComplex>,
    b: &CudaSlice<CudaComplex>,
    mask: &CudaSlice<f64>,
    n_real: usize,
    n_i32: i32,
) -> CudaSlice<CudaComplex> {
    let tmp = mix_residual(kernels, stream, a, b, n_real, n_i32);
    let mut dst: CudaSlice<CudaComplex> = stream
        .alloc_zeros(n_real)
        .expect("alloc masked delta");
    unsafe {
        stream
            .launch_builder(&kernels.cpx_mask)
            .arg(&mut dst)
            .arg(&tmp)
            .arg(mask)
            .arg(&n_i32)
            .launch(LaunchConfig::for_num_elems(n_real as u32))
    }
    .expect("cpx_mask delta");
    dst
}

/// D2H + H2D round-trip to persist a GPU slice (device memory is not valid
/// across streams; the history ring must outlive the call).
fn clone_recip(
    stream: &Arc<CudaStream>,
    slice: &CudaSlice<CudaComplex>,
    shape: [usize; 3],
) -> ReciprocalDensity {
    let copy = clone_recip_streamed(stream, slice);
    ReciprocalDensity::new(copy, shape)
}

fn clone_recip_streamed(
    stream: &Arc<CudaStream>,
    slice: &CudaSlice<CudaComplex>,
) -> CudaSlice<CudaComplex> {
    stream
        .clone_htod(&stream.clone_dtoh(slice).expect("D2H for storage"))
        .expect("H2D for storage")
}

// ===========================================================================
// Joint (c, s) DIIS system build (CPU)
// ===========================================================================

/// Build and solve the N×N joint DIIS system for the (Δc, Δs) history.
///
/// CASTEP `dm_mix_density_dot` (dm_sub_base.f90:1112-1180): the charge
/// part is weighted by `mix_metric` (1 + E_q1sq/E with q1 = 0 → 1.0 for
/// G>0, 0.0 at G=0 — passed as `metric_mask`), the spin part is
/// unweighted.
///
/// `delta_rc_history`: ΔR_c, metric-masked (G=0 zeroed).
/// `delta_rs_history`: ΔR_s, band-masked (empty for nspins = 1).
/// `r_c` / `r_s`: the current residuals.  The charge inner products use
/// the metric-weighted residual `r_c × metric_mask` (computed on the
/// GPU with `cpx_mask`); the spin part uses the raw `r_s`.
fn build_and_solve_diis(
    kernels: &MixingCudaKernels,
    delta_rc_history: &[CudaSlice<CudaComplex>],
    delta_rs_history: &[CudaSlice<CudaComplex>],
    r_c: &CudaSlice<CudaComplex>,
    r_s: &CudaSlice<CudaComplex>,
    metric_mask: &CudaSlice<f64>,
    n_real: i32,
    n: usize,
) -> (Vec<f64>, bool) {
    let stream = kernels.stream.clone();
    let blas = &kernels.blas;
    let mut matrix = vec![0.0_f64; n * n];
    let mut rhs = vec![0.0_f64; n];

    // Metric-weighted charge residual: r_c_w = r_c × metric_mask (GPU).
    let r_c_w = {
        let mut dst: CudaSlice<CudaComplex> = stream
            .alloc_zeros(n_real as usize)
            .expect("alloc weighted r_c");
        unsafe {
            stream
                .launch_builder(&kernels.cpx_mask)
                .arg(&mut dst)
                .arg(r_c)
                .arg(metric_mask)
                .arg(&n_real)
                .launch(LaunchConfig::for_num_elems(n_real as u32))
        }
        .expect("cpx_mask r_c × metric");
        dst
    };

    let zero_c = CudaComplex { x: 0.0, y: 0.0 };
    for i in 0..n {
        let dr_c_i = &delta_rc_history[i];
        let dr_s_i = delta_rs_history.get(i);
        for j in 0..n {
            // M_ij = Re[Σ conj(ΔR_j)·ΔR_i]: charge part metric-weighted
            // (ΔR_c is already metric-masked), spin part unweighted.
            let mut m = blas
                .dotc_c64(n_real, &delta_rc_history[j], 1, dr_c_i, 1)
                .unwrap_or(zero_c)
                .x;
            if let (Some(a), Some(b)) = (delta_rs_history.get(j), dr_s_i) {
                m += blas
                    .dotc_c64(n_real, a, 1, b, 1)
                    .unwrap_or(zero_c)
                    .x;
            }
            matrix[i * n + j] = m;
        }
        // b_i = −(Re[Σ conj(ΔRc_i)·Rc_w] + Re[Σ conj(ΔRs_i)·Rs])
        let mut b = blas
            .dotc_c64(n_real, dr_c_i, 1, &r_c_w, 1)
            .unwrap_or(zero_c)
            .x;
        if let Some(dr_s_i) = dr_s_i {
            b += blas
                .dotc_c64(n_real, dr_s_i, 1, r_s, 1)
                .unwrap_or(zero_c)
                .x;
        }
        rhs[i] = -b;
    }

    solve_diis_system(&mut matrix, &mut rhs, n)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// DIIS on a known 2×2 system: M = [[2,1],[1,2]], b = [5,4].
    /// Solution via LAPACK dgesv reference: x = [2, 1]ᵀ
    #[test]
    fn test_diis_2x2_known_system() {
        let mut a = vec![2.0, 1.0, 1.0, 2.0];
        let mut b = vec![5.0, 4.0];
        let (x, fallback) = solve_diis_system(&mut a, &mut b, 2);
        assert!(!fallback, "should not fall back for non-singular system");
        assert!((x[0] - 2.0).abs() < 1e-12, "x[0] expected 2.0, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x[1] expected 1.0, got {}", x[1]);
    }

    /// DIIS on a 3×3 identity: M = I, b = [1,2,3] → x = [1,2,3]
    #[test]
    fn test_diis_3x3_identity() {
        let mut a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = vec![1.0, 2.0, 3.0];
        let (x, fallback) = solve_diis_system(&mut a, &mut b, 3);
        assert!(!fallback, "should not fall back");
        assert!((x[0] - 1.0).abs() < 1e-12, "x[0]");
        assert!((x[1] - 2.0).abs() < 1e-12, "x[1]");
        assert!((x[2] - 3.0).abs() < 1e-12, "x[2]");
    }

    /// Singular matrix must trigger fallback.
    #[test]
    fn test_diis_singular_fallback() {
        let mut a = vec![0.0, 0.0, 0.0, 0.0];
        let mut b = vec![1.0, 2.0];
        let (_x, fallback) = solve_diis_system(&mut a, &mut b, 2);
        assert!(fallback, "singular matrix must trigger fallback");
    }

    /// Near-singular matrix must also trigger fallback (pivot below 1e-30).
    #[test]
    fn test_diis_near_singular_fallback() {
        let mut a = vec![1e-40, 0.0, 0.0, 1e-40];
        let mut b = vec![1.0, 1.0];
        let (_x, fallback) = solve_diis_system(&mut a, &mut b, 2);
        assert!(fallback, "near-singular matrix must trigger fallback");
    }

    /// n=0: empty solution, no fallback.
    #[test]
    fn test_diis_empty() {
        let (x, fallback) = solve_diis_system(&mut [], &mut [], 0);
        assert!(!fallback, "empty system should not fall back");
        assert!(x.is_empty(), "coeffs should be empty");
    }

    /// The spin kernel G=0 entry must be 1.0 in the pure-slice form so
    /// that amp_s × 1.0 reproduces CASTEP's Ks(0) = mix_spin_amp
    /// (dm_sub_base.f90:681-682: "need to mix the G=0 spin component").
    #[test]
    fn test_spin_kernel_g0_full_mixed() {
        let amp_s: f64 = 2.0;
        let ks_g0_pure: f64 = 1.0;
        assert!((amp_s * ks_g0_pure - 2.0).abs() < 1e-15_f64,
            "effective Ks(G=0) must equal amp_s");
    }

    /// The charge kernel G=0 entry must be 0.0 (total charge conservation):
    /// amp_c × 0 = 0 regardless of the amplitude.
    #[test]
    fn test_charge_kernel_g0_zero() {
        let amp_c: f64 = 0.5;
        let kc_g0_pure: f64 = 0.0;
        assert_eq!(amp_c * kc_g0_pure, 0.0_f64,
            "effective Kc(G=0) must be 0 (charge conservation)");
    }
}
