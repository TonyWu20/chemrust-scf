pub mod kerker;
pub mod reciprocal_density;
pub(crate) mod cuda_kernels;

use std::marker::PhantomData;
use std::sync::Arc;

use chemrust_hamiltonian_core::GVectorGrid;
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
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
/// | Pulay | DIIS with up to 7 history entries + Kerker |
pub struct DensityHistory<M: MixingPhase> {
    /// Kerker preconditioner kernel K(G) on GPU. Created lazily on first
    /// Kerker `into_kerker()` call.
    pub(crate) kerker: Option<KerkerPreconditioner>,
    /// Compiled CUDA kernels for GPU element-wise ops. Created lazily.
    pub(crate) kernels: Option<MixingCudaKernels>,
    /// Number of spin channels (1 for NonSpin, 2 for SpinCollinear).
    nspins: usize,
    /// Mixing amplitude per spin channel.
    /// CASTEP uses separate amplitudes for charge (default 0.5) and spin
    /// (default 2.0, per NiO .param: spin_density_mixing_amplitude=2.0).
    /// Since our code mixes per-spin densities (ρ_up, ρ_dn) independently
    /// rather than charge+spin, the per-spin amplitude approximates
    /// (amp_charge + amp_spin)/2 for each channel.
    /// Reference: CASTEP dm_sub_mix.f90:434 (amp_c, amp_s).
    pub(crate) mixing_amplitude: Vec<f64>,
    /// CASTEP `mix_charge_gmax` in a₀⁻¹ (default 1.5 /Å = 2.8346 a₀⁻¹).
    /// Sets the Kerker kernel scale: K(G) = G²/(G² + gmax²).
    /// CASTEP dm_sub_base.f90:613 `energy_ch_q0sq = 0.5·mix_charge_gmax²`.
    pub(crate) mix_gmax: f64,
    /// Reciprocal-space density from the previous iteration's mixing output,
    /// per spin channel.  `None` on the first Kerker/Pulay call (pass-through).
    /// CASTEP dm.f90 stores per-spin density history independently.
    pub(crate) current_density_in: Vec<Option<ReciprocalDensity>>,
    // ── DIIS delta history (GPU-resident complex), per spin ──
    /// Δn_i = n_in_i - n_in_{i-1}, ordered oldest to newest, per spin.
    delta_n_history: Vec<Vec<CudaSlice<CudaComplex>>>,
    /// ΔR_i = R_i - R_{i-1}, ordered oldest to newest, per spin.
    delta_r_history: Vec<Vec<CudaSlice<CudaComplex>>>,
    /// Previous residual R_{t-1} (GPU), for computing next ΔR, per spin.
    prev_res: Vec<Option<CudaSlice<CudaComplex>>>,
    /// Previous n_in_{t-1} (GPU), for computing next Δn, per spin.
    /// Saved before updating `current_density_in`.
    prev_n_in: Vec<Option<CudaSlice<CudaComplex>>>,
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
// DIIS linear algebra helper (CPU — matrix is tiny, ≤7×7)
// ===========================================================================

/// Solve the N×N DIIS system `A·c = b` via Gaussian elimination with partial
/// pivoting.
///
/// `a` is `n×n`, `b` is `n×1`, both stored in row-major order.
///
/// Returns `(coeffs, fallback)` where `coeffs` has length `n` and `fallback`
/// is `true` if the solve failed (singular matrix) — caller should fall back
/// to Kerker-only mixing for this step.
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
}

/// Build DIIS matrix M (N×N) and RHS b (N×1) on CPU using cuBLAS dot
/// products on GPU, then solve via `solve_diis_system`.
///
/// M_ij = Re[Σ conj(ΔR_j)·ΔR_i]   (i.e., real part of cuBLAS zdotc)
/// b_i  = -Re[Σ conj(ΔR_i)·R_current]
fn build_and_solve_diis(
    delta_r_history: &[CudaSlice<CudaComplex>],
    r_curr: &CudaSlice<CudaComplex>,
    n_real: i32,
    blas: &crate::device::blas::BlasHandle,
    n: usize,
) -> (Vec<f64>, bool) {
    let mut matrix = vec![0.0_f64; n * n];
    let mut rhs = vec![0.0_f64; n];

    for i in 0..n {
        let delta_r_i = &delta_r_history[i];
        for j in 0..n {
            let delta_r_j = &delta_r_history[j];
            // M_ij = Re[zdotc(delta_r_j, delta_r_i)]
            let dot = blas
                .dotc_c64(n_real, delta_r_j, 1, delta_r_i, 1)
                .unwrap_or(CudaComplex { x: 0.0, y: 0.0 });
            matrix[i * n + j] = dot.x; // real part
        }
        // b_i = -Re[zdotc(delta_r_i, r_curr)]
        let dot = blas
            .dotc_c64(n_real, delta_r_i, 1, r_curr, 1)
            .unwrap_or(CudaComplex { x: 0.0, y: 0.0 });
        rhs[i] = -dot.x;
    }

    solve_diis_system(&mut matrix, &mut rhs, n)
}

// ===========================================================================
// MixingOff — pass-through (no active mixing)
// ===========================================================================

impl DensityHistory<MixingOff> {
    /// Create a new density history in the `Off` phase.
    ///
    /// No GPU resources are allocated until the first transition to `Kerker`.
    ///
    /// `nspins`: number of spin channels (1 for NonSpin, 2 for SpinCollinear).
    /// Per-spin state (current_density_in, DIIS history) is allocated so that
    /// mixing each spin channel independently does not cross-contaminate
    /// (CASTEP dm.f90 stores per-spin density history independently).
    pub fn new(nspins: usize) -> Self {
        // Default: charge mixing amplitude 0.5 for all spin channels.
        // Override with set_mixing_amplitude() for spin-polarised systems
        // where CASTEP uses spin_density_mixing_amplitude=2.0.
        Self::with_amplitude(nspins, 0.5)
    }

    /// Create a DensityHistory with a specific per-spin mixing amplitude.
    ///
    /// CASTEP uses `mix_charge_amp=0.5` for charge density and
    /// `spin_density_mixing_amplitude=2.0` for spin density.
    /// Since our code mixes per-spin densities (ρ_up, ρ_down) independently,
    /// pass `spin_density_mixing_amplitude` from CASTEP .param here.
    pub fn with_amplitude(nspins: usize, amp: f64) -> Self {
        let per_spin = || Vec::with_capacity(DIIS_MAX_HISTORY);
        Self {
            kerker: None,
            kernels: None,
            nspins,
            mixing_amplitude: vec![amp; nspins],
            mix_gmax: crate::mixing::kerker::KERKER_GMAX_DEFAULT,
            current_density_in: (0..nspins).map(|_| None).collect(),
            delta_n_history: (0..nspins).map(|_| per_spin()).collect(),
            delta_r_history: (0..nspins).map(|_| per_spin()).collect(),
            prev_res: (0..nspins).map(|_| None).collect(),
            prev_n_in: (0..nspins).map(|_| None).collect(),
            _marker: PhantomData,
        }
    }

    /// Set the CASTEP `mix_charge_gmax` scale (a₀⁻¹) of the Kerker kernel.
    /// Must be called before `into_kerker` (the kernel is built lazily).
    pub fn with_mix_gmax(mut self, gmax: f64) -> Self {
        self.mix_gmax = gmax;
        self
    }

    /// Pass-through mixing: density unchanged, snapshot = clone.
    ///
    /// `ispin`: spin channel index (0-based).  Not used by MixingOff (pass-through),
    /// but accepted for API uniformity with Kerker/Pulay.
    pub fn mix(&mut self, density: Density, _ispin: usize) -> (Density, Density) {
        let snapshot = density.clone();
        (density, snapshot)
    }

    /// Transition from `Off` to `Kerker` phase.
    ///
    /// Creates the Kerker preconditioner and compiles CUDA kernels on GPU
    /// from the wave G-vector grid. This is the first GPU-touching operation
    /// on the history.
    pub fn into_kerker(
        mut self,
        mixing_grid: &GVectorGrid,
        g2_cutoff: Option<f64>,
    ) -> Result<DensityHistory<Kerker>, Error> {
        if self.kerker.is_none() {
            let ctx = Arc::new(CudaContext::new(0).map_err(Error::Cuda)?);
            let stream = ctx.default_stream();
            let kerker = KerkerPreconditioner::new(&stream, mixing_grid, g2_cutoff, self.mix_gmax)?;
            self.kerker = Some(kerker);
            // Compile kernels on the same stream/context to avoid context
            // isolation (device pointers are not valid across contexts).
            self.kernels = Some(MixingCudaKernels::new_from_stream(&stream)?);
        } else if self.kernels.is_none() {
            // Kerker exists from a prior transition but kernels do not.
            let ctx = Arc::new(CudaContext::new(0).map_err(Error::Cuda)?);
            let stream = ctx.default_stream();
            self.kernels = Some(MixingCudaKernels::new_from_stream(&stream)?);
        }
        Ok(DensityHistory {
            kerker: self.kerker,
            kernels: self.kernels,
            nspins: self.nspins,
            mixing_amplitude: self.mixing_amplitude.clone(),
            mix_gmax: self.mix_gmax,
            current_density_in: self.current_density_in,
            delta_n_history: self.delta_n_history,
            delta_r_history: self.delta_r_history,
            prev_res: self.prev_res,
            prev_n_in: self.prev_n_in,
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
// All phases — convert back to `MixingOff` (for uniform mix() return type)
// ===========================================================================

impl<M: MixingPhase> DensityHistory<M> {
    /// Convert any mixing phase back to `MixingOff`, preserving internal data.
    ///
    /// This is needed because `ScfIteration::mix()` normalises the mixing
    /// phase to `Off` so that the `run_scf` loop dispatch can return a
    /// uniform type from all match arms.
    pub fn into_off(self) -> DensityHistory<MixingOff> {
        DensityHistory {
            kerker: self.kerker,
            kernels: self.kernels,
            nspins: self.nspins,
            mixing_amplitude: self.mixing_amplitude.clone(),
            mix_gmax: self.mix_gmax,
            current_density_in: self.current_density_in,
            delta_n_history: self.delta_n_history,
            delta_r_history: self.delta_r_history,
            prev_res: self.prev_res,
            prev_n_in: self.prev_n_in,
            _marker: PhantomData,
        }
    }
}

// ===========================================================================
// Kerker — preconditioned mixing via C2C FFT (GPU-resident)
// ===========================================================================

impl DensityHistory<Kerker> {
    /// Mix density using Kerker preconditioning, fully GPU-resident.
    ///
    /// 1. Upload density → GPU C2C forward FFT → n_out(G)
    /// 2. Compute R = n_out - n_in on GPU (cpx_sub kernel)
    /// 3. Apply Kerker kernel on GPU: n_new = n_in + K(G)·R(G)
    /// 4. Store n_new_recip as `current_density_in` for next iteration
    /// 5. C2C inverse FFT back → mixed real-space density
    ///
    /// Returns `(mixed_density, input_snapshot)`.
    /// `ispin`: spin channel index (0-based).  Per-spin state is stored and
    /// retrieved independently — mixing spin 0 does not affect spin 1's history.
    pub fn mix(&mut self, density: Density, ispin: usize) -> (Density, Density) {
        debug_assert!(ispin < self.nspins, "ispin {ispin} >= nspins {}", self.nspins);
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
        let stream = &kernels.stream;

        // C2C FFT plan (full complex grid, matching Kerker kernel layout)
        let fft_plan = FftPlan3d::plan_c2c(ngx as i32, ngy as i32, ngz as i32, stream.clone())
            .expect("C2C plan");

        // ── Upload density as complex (real part = density, imag = 0) ──
        let density_flat: Vec<f64> = density.flatten_host();
        let cplx_host: Vec<CudaComplex> = density_flat
            .iter()
            .map(|&v| CudaComplex { x: v, y: 0.0 })
            .collect();
        let mut n_out_dev: CudaSlice<CudaComplex> =
            stream.clone_htod(&cplx_host).expect("H2D density to complex");

        // ── C2C forward FFT (in-place) → n_out(G) ──
        unsafe {
            c2c_forward_inplace(&fft_plan, &mut n_out_dev).expect("C2C forward");
        }

        // ── Take previous density for THIS SPIN, if any ──
        let prev_recip = self.current_density_in[ispin].take();

        let (mut mixed_recip_dev, input_snapshot) = match prev_recip {
            Some(prev) => {
                let n_in_dev = prev.data;

                // R = n_out - n_in  (GPU)
                let mut r_dev: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_real).expect("alloc R");
                unsafe {
                    stream
                        .launch_builder(&kernels.cpx_sub)
                        .arg(&mut r_dev)
                        .arg(&n_out_dev)
                        .arg(&n_in_dev)
                        .arg(&n_i32)
                        .launch(LaunchConfig::for_num_elems(n_real as u32))
                }
                .expect("cpx_sub R = n_out - n_in");

                // result = n_in + amp*K·R on the mix basis; above the
                // cutoff the high-G content is carried from n_out
                // (CASTEP dm_mix_density_to_density behaviour).
                let mut result_dev: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_real).expect("alloc result");
                let kerker_dev = self.kerker.as_ref().unwrap().as_device_slice();
                let mask_dev = self.kerker.as_ref().unwrap().as_mask_slice();
                let amp = self.mixing_amplitude[ispin];
                let amp_n_kerker: f64 = 0.0;

                // Zero buffer for the unused sum_delta_r / sum_delta_n terms
                let zero_dev: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_real).expect("alloc zero");

                unsafe {
                    stream
                        .launch_builder(&kernels.cpx_full_update)
                        .arg(&mut result_dev)
                        .arg(&n_in_dev)
                        .arg(kerker_dev)
                        .arg(&r_dev)
                        .arg(&zero_dev)
                        .arg(&zero_dev)
                        .arg(&n_out_dev)
                        .arg(mask_dev)
                        .arg(&n_i32)
                        .arg(&amp)
                        .arg(&amp_n_kerker)
                        .launch(LaunchConfig::for_num_elems(n_real as u32))
                }
                .expect("cpx_full_update n_in + amp*K*R");

                // ── Store mixed reciprocal as current_density_in[ispin] ──
                let recip_copy = stream
                    .clone_htod(
                        &stream
                            .clone_dtoh(&result_dev)
                            .expect("D2H result for storage"),
                    )
                    .expect("H2D result for storage");
                self.current_density_in[ispin] =
                    Some(ReciprocalDensity::new(recip_copy, [ngz, ngy, ngx]));

                (result_dev, density)
            }
            None => {
                // First call for this spin: store n_out, pass through
                self.current_density_in[ispin] =
                    Some(ReciprocalDensity::new(n_out_dev, [ngz, ngy, ngx]));
                // Return the original density unchanged
                return (density.clone(), density);
            }
        };

        // ── C2C inverse FFT (in-place) ──
        unsafe {
            c2c_inverse_inplace(&fft_plan, &mut mixed_recip_dev).expect("C2C inverse");
        }
        stream.synchronize().expect("stream sync");

        // ── D2H, extract real parts, scale by 1/N ──
        let result_cplx: Vec<CudaComplex> = stream
            .clone_dtoh(&mixed_recip_dev)
            .expect("D2H IFFT result");
        let scale = n_real as f64;
        let real_data: Vec<f64> = result_cplx.iter().map(|c| c.x / scale).collect();

        let arr = Array3::from_shape_vec((ngx, ngy, ngz), real_data).expect("valid shape");
        let mixed_density = Density::from_inner(WaveGridArray::from_inner(arr));

        (mixed_density, input_snapshot)
    }

    /// Transition from `Kerker` to `Pulay` (DIIS) phase.
    ///
    /// DIIS history and residual state are PRESERVED: `into_pulay` is
    /// invoked at every Pulay iteration (construct_density_pulay), so
    /// resetting prev_res/prev_n_in/here would zero the delta history and
    /// DIIS would never activate. CASTEP keeps one density/residual
    /// history across scheme switches (dm_sub_mix.f90 uses the same
    /// density_history and residual_history arrays for Kerker and Pulay).
    pub fn into_pulay(self) -> DensityHistory<Pulay> {
        DensityHistory {
            kerker: self.kerker,
            kernels: self.kernels,
            nspins: self.nspins,
            mixing_amplitude: self.mixing_amplitude.clone(),
            mix_gmax: self.mix_gmax,
            current_density_in: self.current_density_in,
            delta_n_history: self.delta_n_history,
            delta_r_history: self.delta_r_history,
            prev_res: self.prev_res,
            prev_n_in: self.prev_n_in,
            _marker: PhantomData,
        }
    }
}

// ===========================================================================
// Pulay (DIIS) — direct inversion in the iterative subspace
// ===========================================================================

impl DensityHistory<Pulay> {
    /// Set the mixing amplitude for a specific spin channel.
    ///
    /// CASTEP uses separate amplitudes: charge mixing amplitude (default 0.5)
    /// and spin density mixing amplitude (e.g., 2.0 for NiO, per
    /// `spin_density_mixing_amplitude` in .param).  Since our code mixes
    /// per-spin densities independently rather than charge+spin, the per-spin
    /// amplitude should approximate (amp_charge + amp_spin)/2 ≈ 1.25 for
    /// magnetic systems.
    ///
    /// Reference: CASTEP NiO .castep lines "charge density mixing amplitude:
    /// 0.5000" and "spin density mixing amplitude: 2.000".
    pub fn set_mixing_amplitude(&mut self, ispin: usize, amp: f64) {
        assert!(ispin < self.nspins, "ispin {ispin} >= nspins {}", self.nspins);
        self.mixing_amplitude[ispin] = amp;
    }

    /// Mix density using Pulay/DIIS with up to 7 history entries — fully
    /// GPU-resident except for the tiny (≤7×7) linear solve on CPU.
    ///
    /// CASTEP's dm_mix_density_pulay algorithm (dm.f90:1009-1063):
    ///
    ///   History stores **deltas**:
    ///     Δn_i = n_in_i - n_in_{i-1}
    ///     ΔR_i = R_i - R_{i-1}
    ///
    ///   DIIS is **unconstrained** N×N:
    ///     M_ij = ⟨ΔR_j | ΔR_i⟩
    ///     b_i  = -⟨ΔR_i | R_current⟩
    ///     Solve M·c = b  (no Lagrange multiplier, no Σc_i = 1)
    ///
    ///   Update:
    ///     n_new = n_in_current + Σc_i·Δn_i
    ///           + K(G)·[R_current + Σc_i·ΔR_i]
    ///
    /// Returns `(mixed_density, input_snapshot)`.
    /// `ispin`: spin channel index (0-based).  Per-spin DIIS history is stored
    /// and retrieved independently — mixing spin 0 does not affect spin 1's
    /// delta vectors or residual history.
    pub fn mix(&mut self, density: Density, ispin: usize) -> (Density, Density) {
        debug_assert!(ispin < self.nspins, "ispin {ispin} >= nspins {}", self.nspins);
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
        let stream = &kernels.stream;
        let blas = &kernels.blas;

        let fft_plan = FftPlan3d::plan_c2c(ngx as i32, ngy as i32, ngz as i32, stream.clone())
            .expect("C2C plan");

        // ── 1. Upload density as complex, C2C forward → n_out(G) ──
        let density_flat: Vec<f64> = density.flatten_host();
        let cplx_host: Vec<CudaComplex> = density_flat
            .iter()
            .map(|&v| CudaComplex { x: v, y: 0.0 })
            .collect();
        let mut n_out_dev: CudaSlice<CudaComplex> =
            stream.clone_htod(&cplx_host).expect("H2D density");
        unsafe {
            c2c_forward_inplace(&fft_plan, &mut n_out_dev).expect("C2C forward");
        }

        // ── 2. Take previous density for THIS SPIN ──
        let prev_recip = self.current_density_in[ispin].take();

        let (mut mixed_recip_dev, input_snapshot) = match prev_recip {
            Some(prev) => {
                let n_in_dev = prev.data;

                // ---- 2b. R = n_out - n_in  (GPU) ----
                let mut r_dev: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_real).expect("alloc R");
                unsafe {
                    stream
                        .launch_builder(&kernels.cpx_sub)
                        .arg(&mut r_dev)
                        .arg(&n_out_dev)
                        .arg(&n_in_dev)
                        .arg(&n_i32)
                        .launch(LaunchConfig::for_num_elems(n_real as u32))
                }
                .expect("cpx_sub R = n_out - n_in");

                // ---- 3. Push deltas to history per-spin ----
                // CASTEP mix density objects are band-limited to
                // num_mix_plane_waves, so the DIIS delta history carries
                // only mix-basis (low-G) content. Mask here.
                let old_prev_res = self.prev_res[ispin].take();
                let old_prev_n_in = self.prev_n_in[ispin].take();
                let mask_dev = self.kerker.as_ref().unwrap().as_mask_slice();

                if let Some(prev_res_dev) = old_prev_res {
                    // ΔR = R_current - R_{t-1}  (GPU)
                    let mut delta_r_tmp: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc ΔR");
                    unsafe {
                        stream
                            .launch_builder(&kernels.cpx_sub)
                            .arg(&mut delta_r_tmp)
                            .arg(&r_dev)
                            .arg(&prev_res_dev)
                            .arg(&n_i32)
                            .launch(LaunchConfig::for_num_elems(n_real as u32))
                    }
                    .expect("cpx_sub ΔR = R - prev_res");

                    // Δn = n_in_current - n_in_{t-1}  (GPU)
                    let mut delta_n_tmp: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc Δn");
                    if let Some(prev_n_in_dev) = old_prev_n_in {
                        unsafe {
                            stream
                                .launch_builder(&kernels.cpx_sub)
                                .arg(&mut delta_n_tmp)
                                .arg(&n_in_dev)
                                .arg(&prev_n_in_dev)
                                .arg(&n_i32)
                                .launch(LaunchConfig::for_num_elems(n_real as u32))
                        }
                        .expect("cpx_sub Δn = n_in - prev_n_in");
                    }
                    // If old_prev_n_in is None (should not happen), Δn stays zero

                    // Band-limit both deltas to the CASTEP mix basis
                    let mut delta_r_masked: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc ΔR masked");
                    let mut delta_n_masked: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc Δn masked");
                    unsafe {
                        stream
                            .launch_builder(&kernels.cpx_mask)
                            .arg(&mut delta_r_masked)
                            .arg(&delta_r_tmp)
                            .arg(mask_dev)
                            .arg(&n_i32)
                            .launch(LaunchConfig::for_num_elems(n_real as u32))
                        .expect("cpx_mask ΔR");
                        stream
                            .launch_builder(&kernels.cpx_mask)
                            .arg(&mut delta_n_masked)
                            .arg(&delta_n_tmp)
                            .arg(mask_dev)
                            .arg(&n_i32)
                            .launch(LaunchConfig::for_num_elems(n_real as u32))
                        .expect("cpx_mask Δn");
                    }

                    // Push (masked) deltas to history (Vec with automatic eviction), per-spin
                    if self.delta_n_history[ispin].len() >= DIIS_MAX_HISTORY {
                        self.delta_n_history[ispin].remove(0);
                        self.delta_r_history[ispin].remove(0);
                    }
                    self.delta_n_history[ispin].push(delta_n_masked);
                    self.delta_r_history[ispin].push(delta_r_masked);
                }

                // ---- 4. Save R_current and n_in_current as prev_{res,n_in} per-spin ----
                let new_prev_res: CudaSlice<CudaComplex> = stream
                    .clone_htod(
                        &stream
                            .clone_dtoh(&r_dev)
                            .expect("D2H prev_res for storage"),
                    )
                    .expect("H2D prev_res");
                self.prev_res[ispin] = Some(new_prev_res);

                let new_prev_n_in: CudaSlice<CudaComplex> = stream
                    .clone_htod(
                        &stream
                            .clone_dtoh(&n_in_dev)
                            .expect("D2H prev_n_in for storage"),
                    )
                    .expect("H2D prev_n_in");
                self.prev_n_in[ispin] = Some(new_prev_n_in);

                // ---- 5. Build and solve DIIS system (CPU) per-spin ----
                let n_history = self.delta_r_history[ispin].len();
                let (c_coeff, fallback) = if n_history > 0 {
                    build_and_solve_diis(
                        &self.delta_r_history[ispin],
                        &r_dev,
                        n_real as i32,
                        blas,
                        n_history,
                    )
                } else {
                    (vec![], true)
                };

                let kerker_dev = self.kerker.as_ref().unwrap().as_device_slice();
                let mask_dev = self.kerker.as_ref().unwrap().as_mask_slice();

                let zero_dev: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_real).expect("alloc zero");

                let result_dev: CudaSlice<CudaComplex> = if !fallback {
                    // ---- 8. DIIS update on GPU (CASTEP dm_sub_mix.f90 form) ----
                    // n_new = n_in + Σc_i·Δn_i + K·(Σc_i·ΔR_i)
                    //   DIIS part unscaled (amp = 1.0); R_curr NOT in the
                    //   Kerker part; high-G content carried from n_out.

                    // Allocate accumulator buffers
                    let mut sum_delta_r: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc sum ΔR");
                    let mut sum_delta_n: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc sum Δn");

                    // Σc_i·ΔR_i  and  Σc_i·Δn_i  via cuBLAS Zaxpy, per-spin
                    for (i, &ci) in c_coeff.iter().enumerate() {
                        let alpha = CudaComplex { x: ci, y: 0.0 };
                        blas.axpy_c64(n_i32, alpha, &self.delta_r_history[ispin][i], 1, &mut sum_delta_r, 1)
                            .expect("axpy sum_delta_r");
                        blas.axpy_c64(n_i32, alpha, &self.delta_n_history[ispin][i], 1, &mut sum_delta_n, 1)
                            .expect("axpy sum_delta_n");
                    }

                    // CASTEP dm_sub_mix.f90 dm_mix_density_pulay exact form:
                    //   new = n_in + sum_dn + K*(R + sum_dr)
                    // The kerker slice holds the PURE kernel G2/(G2+q2);
                    // amp_k carries mix_charge_amp; the DIIS density part
                    // is unscaled (amp_n = 1.0); R_curr is INSIDE the
                    // Kerker part (CASTEP copies current_residual into the
                    // workspace before adding the c-weighted residual diffs).
                    let charge_amp = self.mixing_amplitude[ispin];
                    let mut result_dev: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc result");
                    let amp_k: f64 = charge_amp;
                    let amp_n: f64 = 1.0;
                    unsafe {
                        stream
                            .launch_builder(&kernels.cpx_full_update)
                            .arg(&mut result_dev)
                            .arg(&n_in_dev)
                            .arg(kerker_dev)
                            .arg(&r_dev)
                            .arg(&sum_delta_r)
                            .arg(&sum_delta_n)
                            .arg(&n_out_dev)
                            .arg(mask_dev)
                            .arg(&n_i32)
                            .arg(&amp_k)
                            .arg(&amp_n)
                            .launch(LaunchConfig::for_num_elems(n_real as u32))
                    }
                    .expect("cpx_full_update DIIS n_new");
                    result_dev
                } else {
                    // ---- Kerker fallback: n_new = n_in + amp_k * K·R ----
                    // (CASTEP dm_mix_density_kerker, taken when the DIIS
                    // solve fails; amp_n = 0)
                    let mut result_dev: CudaSlice<CudaComplex> =
                        stream.alloc_zeros(n_real).expect("alloc result");
                    let amp_k_fb = self.mixing_amplitude[ispin];
                    let amp_n_fb: f64 = 0.0;
                    unsafe {
                        stream
                            .launch_builder(&kernels.cpx_full_update)
                            .arg(&mut result_dev)
                            .arg(&n_in_dev)
                            .arg(kerker_dev)
                            .arg(&r_dev)
                            .arg(&zero_dev)
                            .arg(&zero_dev)
                            .arg(&n_out_dev)
                            .arg(mask_dev)
                            .arg(&n_i32)
                            .arg(&amp_k_fb)
                            .arg(&amp_n_fb)
                            .launch(LaunchConfig::for_num_elems(n_real as u32))
                    }
                    .expect("cpx_full_update Kerker fallback n_in + amp*K*R");

                    // If no history exists yet, also store R_current as prev_res
                    // (already done above)
                    result_dev
                };

                // ---- Store mixed reciprocal as current_density_in[ispin] ----
                let recip_copy = stream
                    .clone_htod(
                        &stream
                            .clone_dtoh(&result_dev)
                            .expect("D2H result for storage"),
                    )
                    .expect("H2D result for storage");
                self.current_density_in[ispin] =
                    Some(ReciprocalDensity::new(recip_copy, [ngz, ngy, ngx]));

                (result_dev, density)
            }
            None => {
                // No previous density for this spin: pass through
                self.current_density_in[ispin] =
                    Some(ReciprocalDensity::new(n_out_dev, [ngz, ngy, ngx]));
                return (density.clone(), density);
            }
        };

        // ── 9. C2C inverse FFT (in-place) ──
        unsafe {
            c2c_inverse_inplace(&fft_plan, &mut mixed_recip_dev).expect("C2C inverse");
        }
        stream.synchronize().expect("stream sync");

        let result_cplx: Vec<CudaComplex> = stream
            .clone_dtoh(&mixed_recip_dev)
            .expect("D2H IFFT result");
        let scale = n_real as f64;
        let real_data: Vec<f64> = result_cplx.iter().map(|c| c.x / scale).collect();

        let arr = Array3::from_shape_vec((ngx, ngy, ngz), real_data).expect("valid shape");
        let mixed_density = Density::from_inner(WaveGridArray::from_inner(arr));

        (mixed_density, input_snapshot)
    }
}
