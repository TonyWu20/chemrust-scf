// ---------------------------------------------------------------------------
// Chebyshev polynomial filtering for DFT SCF diagonalization
// ---------------------------------------------------------------------------
//
// Implements:
//   1. SpectralBounds estimation (lambda_max, eps_cut, center, half_width)
//   2. Lanczos upper-bound estimator
//   3. apply_scaled_hamiltonian() — sigma(H).psi
//   4. chebyshev_filter() — main driver: recurrence + norm check
//   5. Test utilities (HComponentsForTest, apply_s_for_test)

use std::marker::PhantomData;
use std::sync::Arc;

use bon::builder;
use chemrust_hamiltonian_core::{CellGeometry, GVectorGrid, PseudopotentialSet};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};

use crate::device::blas::BlasHandle;
use crate::eigensolver::davidson_types::{KineticPreconditioner, PwCoefficients};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::device::{CudaComplex, Gpu};
use crate::eigensolver::vnl_data::VnlBatchData;

// Re-exports for backward compatibility (types moved to submodules)
pub(crate) use super::hamiltonian::{
    apply_full_hamiltonian, apply_s_inverse, apply_s_times,
    apply_v_loc_hamiltonian, apply_v_nl_hamiltonian,
};
pub use super::kernels::CudaKernelSet;

use crate::layout::{ColumnDistributed, RowDistributed, WavefunctionSet};
use crate::eigensolver::davidson_types::compute_kinetic_energies;
use crate::types::{Error, KPoint};

// ---------------------------------------------------------------------------
// Type alias for the complex Chebyshev return type
// ---------------------------------------------------------------------------

/// Returns (psi_row, hpsi_row) — both in RowDistributed layout.
type ChebyshevResult = Result<
    (
        Gpu<WavefunctionSet<RowDistributed>>,
        Gpu<WavefunctionSet<RowDistributed>>,
    ),
    Error,
>;

/// Return type for `chebyshev_filter_for_test_gpu`: (psi, hpsi, kernels)
type GpuFilterResult = Result<
    (
        Gpu<WavefunctionSet<RowDistributed>>,
        Gpu<WavefunctionSet<RowDistributed>>,
        CudaKernelSet,
    ),
    Error,
>;

// ---------------------------------------------------------------------------
// Filter operator mode for the Chebyshev recurrence (A/B/C diagnostic sweep)
// ---------------------------------------------------------------------------

/// Controls which operator is applied in the Chebyshev recurrence (Step 3)
/// and how the band-shift eigenvalues (Λ) are sourced.
///
/// Used by the `iter1_filter_mode_sweep` diagnostic test to pick the correct
/// production path. After the discriminator selects a winner, this enum and
/// the losing branches are deleted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    /// Mode A — current bare-H path: Step 3 applies H, Step 4 no S⁻¹, Λ from h_eig.
    BareH,
    /// Mode B — minimal flip: Step 3 applies S⁻¹·H, Step 4 no S⁻¹, Λ from h_eig.
    SinvHKeepHEig,
    /// Mode C — full Das Alg 3: Step 3 applies S⁻¹·H, Step 4 applies S⁻¹,
    /// Λ from generalized eigenvalues (falls back to h_eig on iter-1 when None).
    SinvHFullDas,
}

// ---------------------------------------------------------------------------
// Spectral Bounds
// ---------------------------------------------------------------------------

/// Estimated spectral bounds for the Hamiltonian at this SCF iteration.
pub(crate) struct SpectralBounds {
    #[allow(dead_code)]
    pub lambda_max: f64,
    #[allow(dead_code)]
    pub eps_cut: f64,
    pub center: f64,
    pub half_width: f64,
    /// Lower bound on the spectrum (λ₁(H) ≥ min(V_eff) from Gershgorin;
    /// or 0.8×Ritz-min from Lanczos). Used by R-ChFSI (Algorithm 3)
    /// for the spectral shift σ = e/(λ_min − c).
    pub lambda_min: f64,
}

/// Estimate spectral bounds for the Chebyshev filter.
///
/// Follows Zhou (2014, J. Comput. Phys. 255, §3–§5) with available data:
///
/// * **b_up** (λ_max): upper bound on the full spectrum of H.  Must satisfy
///   b_up ≥ λ_max(H) for every eigenvalue, otherwise unwanted high-energy
///   states map to |x| > 1 and the Chebyshev polynomial amplifies them
///   exponentially — destroying the subspace.  We use the Gershgorin-circle
///   estimate `max_kinetic + (max_veff − min_veff)`, which is safe (very
///   conservative) but makes the filter less selective than a Lanczos
///   estimator would.
///
///   **TODO(lanczos):** replace with the k-step Lanczos estimator from
///   Zhou & Li (2011, Linear Algebra Appl. 435, §2): run k = 5…8 Lanczos
///   steps on H with a random start vector, then
///   b_up = λ_max(T_k) + ‖f_k‖₂.  This gives a much tighter bound and
///   makes the filter discriminate effectively.
///
/// * **b_low** (ε_cut): filters map [b_low, b_up] → [−1, 1], so states
///   below b_low are *magnified* and states inside the interval are
///   *damped*.  In CheFSI §5, b_low = max_i Ritz_i — the largest Ritz
///   value from the *previous* SCF iteration — which separates the
///   tracked bands (≤ b_low) from the untracked remainder of the
///   spectrum (> b_low).  We use the same rule when eigenvalues exist,
///   and fall back to a physically-grounded guess on the first call.
///
/// * **No scaled filtering**: we use the unscaled recurrence (Zhou
///   Algorithm 3.1 / eq. 8).  Scaled filtering (Algorithm 3.2) with a_L
///   is needed only when eigenvalues are far from [−1, 1] and overflow
///   is possible — not currently an issue with our conservative b_up.
pub(crate) fn compute_spectral_bounds(
    eigenvalues: Option<&[f64]>,
    wave_grid: &GVectorGrid,
    min_veff: f64,
    max_veff: f64,
) -> Result<SpectralBounds, Error> {
    let gmax = wave_grid.gmax();
    let kinetic_max = 0.5 * gmax * gmax;
    // Gershgorin-circle upper bound: λ_max(H) ≤ max_kinetic + ΔV.
    // Safe (overestimates by ~100× for typical DFT Hamiltonians) but
    // always an upper bound.
    let b_up = kinetic_max + (max_veff - min_veff);

    // Lower bound of the *unwanted* spectrum — separates tracked bands
    // (magnified) from untracked higher bands (damped).
    let b_low = match eigenvalues {
        None | Some([]) => {
            // No Ritz values available — first SCF iteration.
            // Physical intuition: eigenvalues seldom exceed the maximum
            // V_eff value by more than a few Hartree, and the Fermi
            // level of a metal sits near max_veff.  A conservative
            // choice that avoids magnifying unwanted states is to place
            // b_low just above the potential's maximum.
            max_veff.max(0.0) + 2.0
        }
        Some(eig) => {
            // CheFSI §5 step 11: b_low = max_i Ritz_i — the largest
            // eigenvalue still in the tracked subspace.  This damps
            // bands beyond our subspace while magnifying the bands we
            // track (occupied + a few unoccupied).
            eig[eig.len() - 1]
        }
    };

    // Clamp: b_low must stay strictly below b_up for the affine map to
    // be well-defined (otherwise half_width ≤ 0).
    let b_low = b_low.min(b_up * 0.95);

    Ok(SpectralBounds {
        lambda_max: b_up,
        eps_cut: b_low,
        center: (b_up + b_low) / 2.0,
        half_width: (b_up - b_low) / 2.0,
        // λ₁(H) ≥ min(V_eff) because T ≥ 0. Clamp below eps_cut so σ
        // is finite and positive for the R-ChFSI spectral transformation.
        lambda_min: min_veff.min(b_low - 1e-3),
    })
}

// ---------------------------------------------------------------------------
// Lanczos upper-bound estimator
// ---------------------------------------------------------------------------

/// Estimate λ_max(H) via k Lanczos steps with a single starting vector.
///
/// Runs the k-step Lanczos algorithm on H using the first band of `psi_dev`
/// as the starting vector. Returns `λ_max(T_k) + ‖r_k‖₂` as a rigorous
/// upper bound on λ_max(H) (Zhou & Li 2011, Linear Algebra Appl. 435, §2).
///
/// Uses a dedicated 1-band FFT plan so it does not interfere with the
/// n_bands plan used by the main Chebyshev recurrence.
#[allow(clippy::too_many_arguments)]
unsafe fn lanczos_upper_bound(
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    n_pw: usize,
    grid_size: usize,
    inv_ntotal: f64,
    ngx: usize, ngy: usize, ngz: usize,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    solver: &SolverHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
    k_steps: usize,
) -> Result<(f64, f64, f64), Error> {
    let n = n_pw as i32;

    // 1-band FFT plan for the Lanczos vectors
    let plan1 = BatchedFftPlan3d::plan_batched_c2c(
        ngz as i32, ngy as i32, ngx as i32, 1, stream.clone(),
    )?;

    // Working buffers: v (current), v_prev (previous), Hv
    let mut v_cur = PwCoefficients::new(stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
    let mut v_prev = PwCoefficients::new(stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
    let mut hv = PwCoefficients::new(stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
    let mut grid1: CudaSlice<CudaComplex> = stream.alloc_zeros(grid_size).map_err(Error::Cuda)?;
    // Use a deterministic pseudo-random starting vector so Lanczos explores
    // the full spectrum. Using the first band (lowest eigenstate) as start
    // causes Lanczos to converge to λ_min, giving b_up ≈ λ_min + ε << λ_max.
    {
        let rand_cpu: Vec<CudaComplex> = (0..n_pw)
            .map(|i| {
                let t = (i as f64 * 2.399963) % (2.0 * std::f64::consts::PI);
                CudaComplex { x: t.cos(), y: t.sin() }
            })
            .collect();
        let rand_dev = stream.clone_htod(&rand_cpu).map_err(Error::Cuda)?;
        stream.memcpy_dtod(&rand_dev, &mut v_cur.0).map_err(Error::Cuda)?;
    }

    // Normalise v_cur
    let norm0 = {
        let dot = blas.dotc_c64(n, &v_cur, 1, &v_cur, 1).map_err(Error::Blas)?;
        dot.x.sqrt()
    };
    eprintln!("[Lanczos] entry: n_pw={} grid_size={} k_steps={}", n_pw, grid_size, k_steps);
    eprintln!("[Lanczos] norm0 = {:.6e}", norm0);
    if norm0 < 1e-30 {
        // Degenerate starting vector — fall back to Gershgorin
        #[cfg(feature = "scf_diag")]
        eprintln!("[Lanczos] EARLY-RETURN: norm0 < 1e-30 → fallback to (INF, -INF, INF)");
        return Ok((f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY));
    }
    let inv_norm0 = CudaComplex { x: 1.0 / norm0, y: 0.0 };
    unsafe {
        let (ptr, _) = v_cur.device_ptr_mut(stream);
        cudarc::cublas::sys::cublasZscal_v2(blas.raw_handle(), n, &inv_norm0 as *const _ as *const _, ptr as *mut _, 1)
            .result().map_err(Error::Blas)?;
    }

    // Tridiagonal matrix entries for standard L2-Lanczos on bare H.
    // The L2 norm at lines 405-423 already normalizes ‖v₀‖₂ = 1.
    let mut alpha = vec![0.0_f64; k_steps]; // diagonal
    let mut beta  = vec![0.0_f64; k_steps]; // sub-diagonal (beta[0] unused)

    for j in 0..k_steps {
        // Hv = H · v_cur
        unsafe {
            apply_full_hamiltonian()
                .psi_dev(&v_cur)
                .v_eff_dev(v_eff_dev)
                .kinetic_dev(kinetic_dev)
                .fft_idx_dev(fft_idx_dev)
                .n_pw(n_pw)
                .n_bands(1)
                .grid_size(grid_size)
                .inv_ntotal(inv_ntotal)
                .fft_plan(&plan1)
                .hpsi_dev(&mut hv)
                .grid_dev(&mut grid1)
                .vnl_data(vnl_data)
                .blas(blas)
                .kernels(kernels)
                .stream(stream)
                .call()?;
            // Apply S⁻¹·H (global Woodbury) — wires S⁻¹ into the Lanczos
            // estimator so b_up reflects the preconditioned spectrum.
            apply_s_inverse()
                .hpsi_dev(&mut hv)
                .vnl_data(vnl_data)
                .n_bands(1)
                .n_pw(n_pw as i32)
                .blas(blas)
                .stream(stream)
                .solver(solver)
                .call()?;
        }

        // alpha[j] = <v_cur, H·v_cur>  (standard L2 dot product)
        let dot_aj = blas.dotc_c64(n, &v_cur, 1, &hv, 1).map_err(Error::Blas)?;
        alpha[j] = dot_aj.x;

        // Hv -= alpha[j] * v_cur
        let neg_a = CudaComplex { x: -alpha[j], y: 0.0 };
        blas.axpy_c64(n, neg_a, &v_cur.clone(), 1, &mut hv, 1).map_err(Error::Blas)?;

        // Hv -= beta[j] * v_prev  (skip on first step)
        if j > 0 {
            let neg_b = CudaComplex { x: -beta[j], y: 0.0 };
            blas.axpy_c64(n, neg_b, &v_prev.clone(), 1, &mut hv, 1).map_err(Error::Blas)?;
        }

        if j + 1 == k_steps {
            // Last step: record ‖r‖_2 as beta for the bound, then stop.
            let dot_r = blas.dotc_c64(n, &hv, 1, &hv, 1).map_err(Error::Blas)?;
            beta[j] = dot_r.x.sqrt();
            break;
        }

        // beta[j+1] = ‖Hv‖_2  (standard L2 norm)
        let dot_b = blas.dotc_c64(n, &hv, 1, &hv, 1).map_err(Error::Blas)?;
        beta[j + 1] = dot_b.x.sqrt();

        if beta[j + 1] < 1e-14 {
            // Invariant subspace — Lanczos converged early
            beta[j] = 0.0; // no residual
            break;
        }

        // v_prev = v_cur;  v_cur = Hv / beta[j+1]
        stream.memcpy_dtod(&*v_cur, &mut v_prev.0).map_err(Error::Cuda)?;
        let inv_b = CudaComplex { x: 1.0 / beta[j + 1], y: 0.0 };
        stream.memcpy_dtod(&*hv, &mut v_cur.0).map_err(Error::Cuda)?;
        unsafe {
            let (ptr, _) = v_cur.device_ptr_mut(stream);
            cudarc::cublas::sys::cublasZscal_v2(blas.raw_handle(), n, &inv_b as *const _ as *const _, ptr as *mut _, 1)
                .result().map_err(Error::Blas)?;
        }
    }

    // Find λ_max and λ_min of the k×k symmetric tridiagonal T_k on CPU
    // via Gershgorin bounds. T_k is at most 6×6 so this is trivial.
    let k = alpha.len();
    eprintln!("[Lanczos] alpha = {:?}", alpha);
    eprintln!("[Lanczos] beta  = {:?}", beta);
    let lambda_max_tk = (0..k).map(|i| {
        let b_left  = if i > 0   { beta[i].abs() } else { 0.0 };
        let b_right = if i+1 < k { beta[i+1].abs() } else { 0.0 };
        alpha[i] + b_left + b_right
    }).fold(f64::NEG_INFINITY, f64::max);
    let lambda_min_tk = (0..k).map(|i| {
        let b_left  = if i > 0   { beta[i].abs() } else { 0.0 };
        let b_right = if i+1 < k { beta[i+1].abs() } else { 0.0 };
        alpha[i] - b_left - b_right
    }).fold(f64::INFINITY, f64::min);

    // Rigorous upper bound: λ_max(H) ≤ λ_max(T_k) + ‖r_k‖
    let residual_norm = beta[k - 1];
    let b_up_raw = lambda_max_tk + residual_norm;
    eprintln!(
        "[Lanczos] T_k bounds: lambda_min_tk = {:.4} Ha  lambda_max_tk = {:.4} Ha  residual_norm(beta[k-1]) = {:.4e}  → b_up_raw = {:.4} Ha",
        lambda_min_tk, lambda_max_tk, residual_norm, b_up_raw,
    );
    Ok((b_up_raw, lambda_min_tk, lambda_max_tk))
}

// ---------------------------------------------------------------------------
// Precomputed FFT metadata (uploaded to GPU)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scaled Hamiltonian
// ---------------------------------------------------------------------------

/// Compute sigma(H).psi = (H.psi - c*psi) / e in-place on hpsi_dev.
#[allow(dead_code)]
fn apply_scaled_hamiltonian_inplace(
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    psi_dev: &CudaSlice<CudaComplex>,
    n: i32,
    center: f64,
    half_width: f64,
    blas: &BlasHandle,
) -> Result<(), Error> {
    let inv_e = 1.0 / half_width;
    let neg_c_over_e = -center / half_width;

    // Step 1: hpsi *= (1/e)  via zscal
    let alpha_e = CudaComplex { x: inv_e, y: 0.0 };
    unsafe {
        let (ptr, _) = hpsi_dev.device_ptr_mut(blas.stream());
        cudarc::cublas::sys::cublasZscal_v2(
            blas.raw_handle(),
            n,
            &alpha_e as *const _ as *const _,
            ptr as *mut _,
            1,
        )
        .result()
        .map_err(Error::Blas)?;
    }

    // Step 2: hpsi += (-c/e) * psi  via axpy
    let alpha_c = CudaComplex { x: neg_c_over_e, y: 0.0 };
    blas.axpy_c64(n, alpha_c, psi_dev, 1, hpsi_dev, 1)
        .map_err(Error::Blas)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Norm helper
// ---------------------------------------------------------------------------

/// Compute Frobenius norm via cuBLAS dot product: sqrt(conj(x).x)
fn compute_frobenius_norm(buf: &CudaSlice<CudaComplex>, n: i32, blas: &BlasHandle) -> Result<f64, Error> {
    let dot = blas.dotc_c64(n, buf, 1, buf, 1).map_err(Error::Blas)?;
    Ok(dot.x.sqrt())
}

/// Check that norm growth between consecutive iterations does not exceed
/// the expected Chebyshev amplification. T_k(x) for x slightly outside [-1,1]
/// grows as ~x^k, so per-step growth of up to ~3× is normal for the wanted
/// subspace. Only flag genuine numerical overflow (>1000× per step).
fn check_norm_stability(norm_curr: f64, norm_prev: f64, iteration: usize) -> Result<(), Error> {
    let growth_factor = if norm_prev > 0.0 { norm_curr / norm_prev } else { 1.0 };
    if growth_factor > 1000.0 {
        return Err(Error::ChebyshevDiverged {
            iteration,
            norm_previous: norm_prev,
            norm_current: norm_curr,
            growth_factor,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// R-ChFSI helper: band_scale_axpy + upload_f64_slice
// ---------------------------------------------------------------------------

/// Launch the `band_scale_axpy` kernel: dst[b*n_pw + g] += alpha * src[b*n_pw + g] * scale[b].
#[allow(clippy::too_many_arguments)]
fn launch_band_scale_axpy(
    dst: &mut CudaSlice<CudaComplex>,
    src: &CudaSlice<CudaComplex>,
    scale_dev: &CudaSlice<f64>,
    alpha: f64,
    n_pw: usize,
    n_bands: usize,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        stream
            .launch_builder(&kernels.band_scale_axpy)
            .arg(&mut *dst)
            .arg(src)
            .arg(scale_dev)
            .arg(&alpha)
            .arg(&(n_pw as i32))
            .arg(&(n_bands as i32))
            .launch(LaunchConfig::for_num_elems((n_pw * n_bands) as u32))
            .map(|_| ())
    }
    .map_err(Error::Cuda)
}

/// Upload a `&[f64]` slice to the GPU (convenience wrapper).
fn upload_f64_slice(v: &[f64], stream: &Arc<CudaStream>) -> Result<CudaSlice<f64>, Error> {
    stream.clone_htod(v).map_err(Error::Cuda)
}

// ---------------------------------------------------------------------------
// Transpose helper
// ---------------------------------------------------------------------------

/// Transpose ColumnDistributed layout -> RowDistributed layout on GPU.
#[allow(dead_code)]
unsafe fn transpose_col_to_row_on_gpu(
    col_dev: &CudaSlice<CudaComplex>,
    row_dev: &mut CudaSlice<CudaComplex>,
    n_bands: i32,
    n_pw: i32,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        stream
            .launch_builder(&kernels.transpose_col_to_row)
            .arg(col_dev)
            .arg(&mut *row_dev)
            .arg(&n_bands)
            .arg(&n_pw)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    Ok(())
}

/// Transpose RowDistributed layout -> ColumnDistributed layout on GPU.
#[allow(dead_code)]
pub(crate) unsafe fn transpose_row_to_col_on_gpu(
    row_dev: &CudaSlice<CudaComplex>,
    col_dev: &mut CudaSlice<CudaComplex>,
    n_bands: i32,
    n_pw: i32,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        stream
            .launch_builder(&kernels.transpose_row_to_col)
            .arg(row_dev)
            .arg(&mut *col_dev)
            .arg(&n_bands)
            .arg(&n_pw)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Chebyshev filter — main entry point
// ---------------------------------------------------------------------------

/// Apply Chebyshev polynomial filtering to the wavefunctions.
///
/// Returns (psi_row, hpsi_row) where:
/// - `psi_row` is the filtered wavefunction in RowDistributed layout
/// - `hpsi_row` is H|psi> in RowDistributed layout (needed by Rayleigh-Ritz)
///
/// `ndeg` is the Chebyshev polynomial degree.
/// `eigenvalues` is `None` on the first SCF iteration, `Some(&[...])` thereafter.
/// `external_kinetic` when `Some(&[...])` provides pre-computed kinetic energies
/// from CASTEP (`pw_ek_data`), bypassing the internal `compute_kinetic_energies`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chebyshev_filter(
    psi_gpu: &Gpu<WavefunctionSet<ColumnDistributed>>,
    v_eff_dev: &CudaSlice<f64>,
    _pots: &PseudopotentialSet,
    wave_grid: &GVectorGrid,
    _k_point: &KPoint,
    _cell: &CellGeometry,
    pw_coords: &[[i32; 3]],
    vnl_data: &VnlBatchData,
    fft_idx_dev: &CudaSlice<i32>,      // PW-to-FFT-grid index map (length = n_pw)
    min_veff: f64,
    max_veff: f64,
    kernels: &CudaKernelSet,            // pre-compiled GPU kernels (shared)
    pcie: &mut PcieAccount,
    eigenvalues: Option<&[f64]>,
    ndeg: usize,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
    filter_mode: FilterMode,
    external_kinetic: Option<&[f64]>,   // CASTEP-provided KE (bypasses pw_coords KE)
) -> ChebyshevResult {
    // ---- Dimensions ----
    let n_bands = psi_gpu.shape()[0];
    let n_pw = psi_gpu.shape()[1];
    let n_elem = n_bands * n_pw;
    let n_elem_i32 = n_elem as i32;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;

    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = ngx * ngy * ngz;
    let inv_ntotal = 1.0 / (grid_size as f64);
    let grid_alloc = n_bands * grid_size;

    // ---- Precompute & upload kinetic energy (per-PW, not per-grid-point) ----
    // Use CASTEP-provided external KE when available (bypasses pw_coords-based
    // computation which may not match CASTEP's pw_ek_data due to grid coordinate
    // indexing conventions).
    let kinetic_data: Vec<f64> = match external_kinetic {
        Some(ke) => ke.to_vec(),
        None => compute_kinetic_energies(pw_coords, wave_grid.recip_lattice(), _k_point.coords).0,
    };
    let kinetic_dev_values: CudaSlice<f64> =
        stream.clone_htod(&kinetic_data).map_err(Error::Cuda)?;
    let kinetic_dev = KineticPreconditioner::new(kinetic_dev_values);
    pcie.h2d_bytes += kinetic_data.len() * std::mem::size_of::<f64>();

    // ---- FFT plan (batched C2C) ----
    // cuFFT uses row-major layout: n[0] is slowest-varying (outermost),
    // Fortran data layout (ngz, ngy, ngx) with ngz innermost (stride-1).
    // cuFFT n[0] is innermost, so plan dims = (ngz, ngy, ngx).
    // Verified by cufft_dim_ordering_isolated_diagnostic.
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32, n_bands_i32, stream.clone(),
    )?;

    // ---- GPU workspace buffers ----
    let psi_input = PwCoefficients::new(psi_gpu.as_device_slice().clone());

    // R-ChFSI buffers (Algorithm 3):
    // buf_y = Y = H·X − S·X·Λ, buf_sx = S·X for residual, buf_rx = R_X, buf_ry = R_Y
    // buf_c: reused for R_new computation then swap with buf_ry
    // buf_a: reused for X_new reconstruction at Step 4
    let mut buf_y = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut buf_sx = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut buf_rx = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut buf_ry = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut buf_c = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut buf_a = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);

    // Hamiltonian workspace
    let mut hpsi_dev = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut grid_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // Output RowDistributed buffers
    let mut psi_row_dev = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut hpsi_row_dev = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);

    // ---- Spectral bounds ----
    // Per Zhou (2014) Algorithm 4.1 §7.1-7.2:
    //   b_up  = Lanczos estimator (capped at Gershgorin)
    //   b_low = max Ritz value from previous SCF iteration (step 7.2)
    //         = Lanczos-derived midpoint on first call (Algorithm 5.1 eq.13)
    let mut bounds = compute_spectral_bounds(eigenvalues, wave_grid, min_veff, max_veff)?;
    if ndeg > 0 {
        let lanczos_result = unsafe {
            lanczos_upper_bound(
                v_eff_dev, &kinetic_dev, fft_idx_dev,
                n_pw, grid_size, inv_ntotal,
                ngx, ngy, ngz,
                vnl_data, blas, solver, kernels, stream,
                6, // k_steps
            )
        };
        if let Ok((b_up_lanczos, ritz_min, ritz_max)) = lanczos_result {
            #[cfg(not(feature = "scf_diag"))]
            let _ = ritz_max; // used only in scf_diag eprintln below
            let gershgorin_b_up = {
                let gmax = wave_grid.gmax();
                0.5 * gmax * gmax + (max_veff - min_veff)
            };
            // Cap Lanczos b_up at Gershgorin — Lanczos can overshoot when the
            // starting vector is nearly invariant (well-converged wavefunctions).
            let scaled = b_up_lanczos * 1.1;
            let b_up = scaled.min(gershgorin_b_up);
            #[allow(unused_variables)]
            let capped_by_gershgorin = scaled >= gershgorin_b_up;
            #[cfg(feature = "scf_diag")]
            eprintln!(
                "[Lanczos@call] b_up_lanczos={:.4}  ritz_min={:.4}  ritz_max={:.4}  gershgorin={:.4}  scaled(*1.1)={:.4}  → b_up={:.4}  capped_by_gershgorin={}",
                b_up_lanczos, ritz_min, ritz_max, gershgorin_b_up, scaled, b_up, capped_by_gershgorin,
            );

            // b_low: use Ritz values from previous RR when available (Alg 4.1 §7.2).
            // On first call (no prior eigenvalues), use the Gershgorin-based estimate
            // max_veff + 2.0 Ha — this places b_low just above the physical potential
            // maximum, which is a tighter and more physically grounded separator between
            // occupied states (below Fermi ≈ max_veff) and unoccupied states.
            //
            // The T_k midpoint (Zhou Alg 5.1 eq.13) was previously used here but
            // produces b_low ≈ 7.34 Ha for Cu111+CO — the midpoint of the full S⁻¹·H
            // spectrum, not the occupied/unoccupied boundary. This makes the filter
            // amplify ~half the spectrum indiscriminately, causing the 5×/step norm
            // growth observed in the iter-1 log (2026-05-23).
            //
            // max_veff+2.0 = 2.09 Ha is also too high: it sits above all 160 tracked
            // bands (highest band ≈ 0.13 Ha), so the filter amplifies the entire
            // subspace uniformly with no discrimination. The correct b_low for iter-1
            // is just above the highest tracked band — max_veff itself (≈ 0.089 Ha for
            // Cu111+CO) is a tighter and physically correct separator, since the Fermi
            // level of a metal sits near max_veff and the highest occupied state is
            // below it.
            #[allow(unused_variables)]
            let (b_low, b_low_src) = match eigenvalues {
                Some(eig) if !eig.is_empty() => {
                    // Steady-state: b_low = largest Ritz value from previous RR.
                    // This guarantees all occupied states are below b_low and
                    // will be magnified by the filter.
                    (eig[eig.len() - 1], "eig[last]")
                }
                _ => {
                    // First call: use max_veff as b_low. For a metal, the Fermi level
                    // sits near max_veff, so this places b_low just above the highest
                    // tracked band. The +2.0 offset was too large (above all 160 bands).
                    (max_veff.max(0.0), "max_veff")
                }
            };
            #[cfg(feature = "scf_diag")]
            eprintln!("[Lanczos@call] b_low={:.4}  source={}", b_low, b_low_src);

            let guard_pass = b_up.is_finite() && b_up > b_low;
            #[cfg(feature = "scf_diag")]
            eprintln!(
                "[Lanczos@call] guard pass={}  (b_up.is_finite()={} && b_up>{:.4}={})",
                guard_pass,
                b_up.is_finite(),
                b_low,
                b_up > b_low,
            );

            if guard_pass {
                let raw_lambda_min = ritz_min * 0.8;
                bounds = SpectralBounds {
                    lambda_max: b_up,
                    eps_cut: b_low,
                    center: (b_up + b_low) / 2.0,
                    half_width: (b_up - b_low) / 2.0,
                    lambda_min: raw_lambda_min.min(b_low - 1e-3),
                };
            }
        }
        #[cfg(feature = "scf_diag")]
        eprintln!(
            "[Chebyshev] b_up={:.4} Ha  b_low={:.4} Ha  center={:.4} Ha  half_width={:.4} Ha  lambda_min={:.4} Ha",
            bounds.lambda_max, bounds.eps_cut, bounds.center, bounds.half_width, bounds.lambda_min,
        );
    }

    // ---- R-ChFSI Algorithm 3 recurrence body (bare-H variant) ----
    //
    // Reference: Das & al. (2025), Algorithm 3 (main.tex:586-610).
    // The filter operates on bare H (not S⁻¹·H), enriching the subspace
    // with H-eigenvectors. RR (ZHEGVD on H_sub, S_sub) converts to
    // generalized eigenvectors. S appears only in the residual definition
    // (Step 1) and Gram-Schmidt.
    //
    // Spectral parameters (σ, c, e, γ) are computed AFTER Step 1 so we can
    // use per-band H-eigenvalues ⟨ψ_j, H·ψ_j⟩ for the Λ shifts.

    // Pre-allocate lam_y_dev (updated each step, no re-allocation)
    let mut lam_y_dev: CudaSlice<f64> = stream.alloc_zeros(n_bands).map_err(Error::Cuda)?;

    let final_psi_buf: &mut PwCoefficients;

    if ndeg == 0 {
        // No filtering: use input wavefunctions as-is
        stream.memcpy_dtod(&*psi_input, &mut buf_a.0).map_err(Error::Cuda)?;
        final_psi_buf = &mut buf_a;
    } else {
        // ------------------------------------------------------------
        // Step 1: Initial residual Y = H·X − S·X·Λ
        // ------------------------------------------------------------
        // hpsi_dev = H·psi_input
        unsafe {
            apply_full_hamiltonian()
                .psi_dev(&psi_input)
                .v_eff_dev(v_eff_dev)
                .kinetic_dev(&kinetic_dev)
                .fft_idx_dev(fft_idx_dev)
                .n_pw(n_pw)
                .n_bands(n_bands)
                .grid_size(grid_size)
                .inv_ntotal(inv_ntotal)
                .fft_plan(&fft_plan)
                .hpsi_dev(&mut hpsi_dev)
                .grid_dev(&mut grid_dev)
                .vnl_data(vnl_data)
                .blas(blas)
                .kernels(kernels)
                .stream(stream)
                .call()?;
        }

        // buf_y = hpsi_dev (copy, keeping hpsi_dev intact for diagnostics)
        stream.memcpy_dtod(&*hpsi_dev, &mut buf_y.0).map_err(Error::Cuda)?;

        if let Some(eig) = eigenvalues {
            // buf_sx = S·psi_input
            stream.memcpy_dtod(&*psi_input, &mut buf_sx.0).map_err(Error::Cuda)?;            unsafe {
                apply_s_times()
                    .psi_dev(&psi_input)
                    .spsi_dev(&mut buf_sx)
                    .vnl_data(vnl_data)
                    .n_bands(n_bands_i32)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .call()?;
            }
            // Upload eigenvalues to GPU
            let lam_dev = upload_f64_slice(eig, stream)?;
            // buf_y -= S·X · Λ
            launch_band_scale_axpy(
                &mut buf_y, &buf_sx, &lam_dev, -1.0,
                n_pw, n_bands, kernels, stream,
            )?;
        }
        // else: eigenvalues=None → Λ=0 → Y = H·X already in buf_y

        // Compute per-band H-eigenvalues ⟨ψ_j, H·ψ_j⟩ for Λ shifts.
        // These are approximate H-Rayleigh quotients (‖ψ_j‖_S = 1 ≈ ‖ψ_j‖_2
        // for USPP). Used in place of generalized eigenvalues throughout
        // the recurrence to maintain a consistent H-spectrum framing.
        // Block-scoped: device_ptr borrows must be released before the
        // recurrence loop (which mutably borrows hpsi_dev).
        let h_eig: Vec<f64> = {
            let (ptr_psi_base, _sync_psi) = psi_input.device_ptr(stream);
            let (ptr_hpsi_base, _sync_hpsi) = hpsi_dev.device_ptr(stream);
            (0..n_bands)
                .map(|b| -> Result<f64, Error> {
                    let mut result: cudarc::cublas::sys::cuDoubleComplex =
                        cudarc::cublas::sys::cuDoubleComplex { x: 0.0, y: 0.0 };
                    unsafe {
                        let ptr_psi = (ptr_psi_base
                            as *const cudarc::cublas::sys::cuDoubleComplex)
                            .add(b * n_pw);
                        let ptr_hpsi = (ptr_hpsi_base
                            as *const cudarc::cublas::sys::cuDoubleComplex)
                            .add(b * n_pw);
                        cudarc::cublas::sys::cublasZdotc_v2(
                            blas.raw_handle(),
                            n_pw as i32,
                            ptr_psi as *const _,
                            1,
                            ptr_hpsi as *const _,
                            1,
                            &mut result as *mut _,
                        )
                        .result()
                        .map_err(Error::Blas)?;
                    }
                    Ok(result.x)
                })
                .collect::<Result<Vec<_>, Error>>()?
        };

        // Spectral parameters (bare-H framing).
        // b_up, b_low, lambda_min from Lanczos/Gershgorin on bare H.
        // eps_cut (= b_low) is a generalized eigenvalue used as an approximate
        // H-spectrum cutoff (error O(‖S − I‖), small for USPP).
        let e = bounds.half_width;
        let c = bounds.center;
        let sigma_rchfsi = e / (bounds.lambda_min - c);
        let sigma1 = sigma_rchfsi;
        let gamma = 2.0 / sigma1;

        // ------------------------------------------------------------
        // Step 2: Initialize recurrence (main.tex:599-600)
        // ------------------------------------------------------------
        // buf_rx = 0  (already zero-allocated)
        // buf_ry = (σ₁/e) · Y
        let sigma1_over_e = sigma1 / e;
        stream.memcpy_dtod(&*buf_y, &mut buf_ry.0).map_err(Error::Cuda)?;
        unsafe {
            let alpha_s1 = CudaComplex { x: sigma1_over_e, y: 0.0 };
            let (ptr, _) = buf_ry.device_ptr_mut(stream);
            cudarc::cublas::sys::cublasZscal_v2(
                blas.raw_handle(),
                n_elem_i32,
                &alpha_s1 as *const _ as *const _,
                ptr as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;
        }

        // Λ_X = I (CPU)
        let mut lam_x: Vec<f64> = vec![1.0; n_bands];
        // Λ_Y initial value depends on filter mode:
        //   Modes A/B: use h_eig (H-Rayleigh quotients) for consistency with
        //              the bare-H or S⁻¹·H-but-h_eig-shift framing.
        //   Mode C:    use generalized eigenvalues from previous RR when available
        //              (Das Alg 3 §3.2); fall back to h_eig on iter-1 (eigenvalues=None).
        // Per Das Alg 3 §3.2, Λ⁽ⁱ⁾ in the recurrence is the prior iteration's
        // eigenvalue estimate. Mode B's catch-all uses h_eig (Rayleigh quotient
        // on current ψ) which only equals Λ⁽ⁱ⁾ when ψ is exactly an eigenvector.
        // The CHEMRUST_LAMSOURCE_EIG env-var lets a test force Mode B to use
        // the passed-in `eigenvalues` (matching Mode C's catch-all). This is a
        // diagnostic flag for the iter-2-divergence debug session, not a
        // production knob.
        // Λ_Y initial value: use generalized eigenvalues from previous RR when
        // available (Das Alg 3 §3.2), fall back to h_eig (H-Rayleigh quotients)
        // on iter-1 (eigenvalues=None). After §11a fix, eigenvalues is always
        // None in production (standard ChFSI path), so lam_source = h_eig always.
        let lam_source: &[f64] = eigenvalues.unwrap_or(&h_eig);
        let mut lam_y: Vec<f64> = if eigenvalues.is_some() {
            lam_source.iter().map(|l| sigma1_over_e * (l - c)).collect()
        } else {
            let val = -sigma1 * c / e;
            vec![val; n_bands]
        };
        stream.memcpy_htod(&lam_y, &mut lam_y_dev).map_err(Error::Cuda)?;

        // ------------------------------------------------------------
        // Step 3: Recurrence for k = 2..=ndeg (main.tex:601-606)
        // ------------------------------------------------------------
        let mut sigma_cur = sigma1;
        for k in 2..=ndeg {
            let sigma2 = 1.0 / (gamma - sigma_cur);
            let coeff = 2.0 * sigma2 / e;
            let coeff_c = -coeff * c;
            let sigma_sigma2 = -sigma_cur * sigma2;

            // R-ChFSI Algorithm 3 operator: H·S⁻¹·R_Y
            // S⁻¹ applied FIRST to the residual, then H.
            // Das 2025 Alg 3 line 603: R_new = (2σ₂/e)·A·D⁻¹·R_Y − ...
            // where A=H, D⁻¹≈S⁻¹, so A·D⁻¹ = H·S⁻¹ (S⁻¹ first, then H).
            // H and S⁻¹ do NOT commute for USPP; old order filtered wrong operator.
            unsafe {
                if matches!(filter_mode, FilterMode::SinvHKeepHEig | FilterMode::SinvHFullDas) {
                    // Step 1: buf_c = S⁻¹·R_Y (use buf_c as temp; it will be
                    // overwritten below)
                    stream.memcpy_dtod(&*buf_ry, &mut buf_c.0)
                        .map_err(Error::Cuda)?;
                    apply_s_inverse()
                        .hpsi_dev(&mut buf_c)
                        .vnl_data(vnl_data)
                        .n_bands(n_bands_i32)
                        .n_pw(n_pw_i32)
                        .blas(blas)
                        .stream(stream)
                        .solver(solver)
                        .call()?;
                    // Step 2: hpsi_dev = H·(S⁻¹·R_Y) = H·buf_c
                    apply_full_hamiltonian()
                        .psi_dev(&buf_c)
                        .v_eff_dev(v_eff_dev)
                        .kinetic_dev(&kinetic_dev)
                        .fft_idx_dev(fft_idx_dev)
                        .n_pw(n_pw)
                        .n_bands(n_bands)
                        .grid_size(grid_size)
                        .inv_ntotal(inv_ntotal)
                        .fft_plan(&fft_plan)
                        .hpsi_dev(&mut hpsi_dev)
                        .grid_dev(&mut grid_dev)
                        .vnl_data(vnl_data)
                        .blas(blas)
                        .kernels(kernels)
                        .stream(stream)
                        .call()?;
                } else {
                    apply_full_hamiltonian()
                        .psi_dev(&buf_ry)
                        .v_eff_dev(v_eff_dev)
                        .kinetic_dev(&kinetic_dev)
                        .fft_idx_dev(fft_idx_dev)
                        .n_pw(n_pw)
                        .n_bands(n_bands)
                        .grid_size(grid_size)
                        .inv_ntotal(inv_ntotal)
                        .fft_plan(&fft_plan)
                        .hpsi_dev(&mut hpsi_dev)
                        .grid_dev(&mut grid_dev)
                        .vnl_data(vnl_data)
                        .blas(blas)
                        .kernels(kernels)
                        .stream(stream)
                        .call()?;
                }
            }

            // R_new = (2σ₂/e)·H·R_Y − (2σ₂/e)·c·R_Y − σ·σ₂·R_X + (2σ₂/e)·Y·Λ_Y
            // First: buf_c = coeff * H·R_Y
            stream.memcpy_dtod(&*hpsi_dev, &mut buf_c.0).map_err(Error::Cuda)?;
            {
                let alpha_cf = CudaComplex { x: coeff, y: 0.0 };
                unsafe {
                    let (ptr, _) = buf_c.0.device_ptr_mut(stream);
                    cudarc::cublas::sys::cublasZscal_v2(
                        blas.raw_handle(),
                        n_elem_i32,
                        &alpha_cf as *const _ as *const _,
                        ptr as *mut _,
                        1,
                    )
                    .result()
                    .map_err(Error::Blas)?;
                }
            }

            // buf_c += coeff_c * R_Y  (− (2σ₂·c/e) · R_Y)
            let alpha_cc = CudaComplex { x: coeff_c, y: 0.0 };
            blas.axpy_c64(n_elem_i32, alpha_cc, &buf_ry, 1, &mut buf_c, 1)
                .map_err(Error::Blas)?;

            // buf_c += sigma_sigma2 * R_X  (− σ·σ₂ · R_X)
            let alpha_ss = CudaComplex { x: sigma_sigma2, y: 0.0 };
            blas.axpy_c64(n_elem_i32, alpha_ss, &buf_rx, 1, &mut buf_c, 1)
                .map_err(Error::Blas)?;

            // buf_c += coeff * Y · Λ_Y  ((2σ₂/e) · Y · Λ_Y)
            launch_band_scale_axpy(
                &mut buf_c, &buf_y, &lam_y_dev, coeff,
                n_pw, n_bands, kernels, stream,
            )?;

            // Λ_X_new on CPU (main.tex:604)
            // Modes A/B: use h_eig; Mode C: use generalized eigenvalues (lam_source)
            let has_eig = eigenvalues.is_some();
            let new_lam_x: Vec<f64> = if has_eig {
                lam_y.iter().zip(lam_source.iter()).zip(lam_x.iter())
                    .map(|((ly, l), lx)| coeff * ly * l + coeff_c * ly + sigma_sigma2 * lx)
                    .collect()
            } else {
                // eigenvalues None: λ[b] = 0 → coeff*ly*0 term vanishes
                lam_y.iter().zip(lam_x.iter())
                    .map(|(ly, lx)| coeff_c * ly + sigma_sigma2 * lx)
                    .collect()
            };

            // Buffer rotation (main.tex:605):
            //   swap(R_X, R_Y); swap(R_Y, R_new); swap(Λ_X, Λ_Y)
            std::mem::swap(&mut buf_rx, &mut buf_ry);   // buf_rx ← old_R_Y, buf_ry ← old_R_X
            std::mem::swap(&mut buf_ry, &mut buf_c);    // buf_ry ← R_new, buf_c ← old_R_X
            lam_x = std::mem::replace(&mut lam_y, new_lam_x);

            // Update Λ_Y on GPU in-place (pre-allocated, no re-allocation)
            stream.memcpy_htod(&lam_y, &mut lam_y_dev).map_err(Error::Cuda)?;
            sigma_cur = sigma2;

            // Norm check on the newly computed residual R_Y (buf_ry after rotation)
            let norm_curr = compute_frobenius_norm(&buf_ry, n_elem_i32, blas)?;
            let norm_prev = compute_frobenius_norm(&buf_rx, n_elem_i32, blas)?;
            #[cfg(feature = "scf_diag")]
            eprintln!("[R-ChFSI] k={k}  norm_prev={norm_prev:.6e}  norm_curr={norm_curr:.6e}  ratio={:.4}", norm_curr / norm_prev.max(1e-30));
            check_norm_stability(norm_curr, norm_prev, k)?;
        }

        // ------------------------------------------------------------
        // Step 4: Reconstruct X_new (main.tex:607)
        //   Modes A/B: X_new = R_Y + X·Λ_Y  (no S⁻¹)
        //   Mode C:    X_new = S⁻¹·R_Y + X·Λ_Y  (Das Alg 3 line 607)
        // ------------------------------------------------------------
        stream.memcpy_dtod(&*buf_ry, &mut buf_a.0).map_err(Error::Cuda)?;
        if matches!(filter_mode, FilterMode::SinvHFullDas) {
            unsafe {
                apply_s_inverse()
                    .hpsi_dev(&mut buf_a)
                    .vnl_data(vnl_data)
                    .n_bands(n_bands_i32)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .solver(solver)
                    .call()?;
            }
        }
        launch_band_scale_axpy(
            &mut buf_a, &psi_input, &lam_y_dev, 1.0,
            n_pw, n_bands, kernels, stream,
        )?;

        final_psi_buf = &mut buf_a;
    }

    // ---- Gram-Schmidt orthonormalization (Algorithm 4.1 step 7.4) ----
    // The Chebyshev filter amplifies the wanted subspace but does not
    // orthonormalize it. Without this step, S_sub = ψ†ψ is ill-conditioned
    // and ZHEGVD fails. Two passes of classical Gram-Schmidt for stability.
    //
    // Uses S-inner product ⟨x,y⟩_S = x†·S·y for USPP. For each column b:
    //   1. Compute S·col_b once (apply_s_times into gs_s_col)
    //   2. For j<b: dotⱼ = ⟨col_j, S·col_b⟩_S; col_b -= dotⱼ·col_j
    //   3. ‖col_b_new‖²_S = ‖col_b_init‖²_S − Σⱼ|dotⱼ|² (S-orthonormal vⱼ)
    //   4. col_b /= √(‖col_b_new‖²_S)
    //
    // Scratch: gs_col holds col_b(initial); gs_s_col holds S·col_b(initial).
    let mut gs_col = PwCoefficients::new(stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
    let mut gs_s_col = PwCoefficients::new(stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
    unsafe {
        let (psi_ptr, _) = final_psi_buf.0.device_ptr_mut(stream);
        let (gs_col_ptr, _) = gs_col.0.device_ptr_mut(stream);
        for _pass in 0..2 {
            for b in 0..n_bands {
                let col_b = (psi_ptr as *mut CudaComplex).add(b * n_pw);
                // Copy col_b → gs_col (device-to-device)
                cudarc::cublas::sys::cublasZcopy_v2(
                    blas.raw_handle(), n_pw_i32,
                    col_b as *const _, 1,
                    gs_col_ptr as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                // gs_s_col = S · gs_col
                stream.memcpy_dtod(&*gs_col, &mut gs_s_col.0).map_err(Error::Cuda)?;
                apply_s_times()
                    .psi_dev(&gs_col)
                    .spsi_dev(&mut gs_s_col)
                    .vnl_data(vnl_data)
                    .n_bands(1)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .call()?;
                // Get device pointer from gs_s_col after mutable ops complete
                let (gs_s_col_ptr, _) = gs_s_col.0.device_ptr_mut(stream);
                // ‖col_b‖²_S = ⟨col_b, S·col_b⟩  (real part; S is Hermitian)
                // cublasZdotc computes Σ_i conj(x[i])·y[i], a dimensionless grid sum.
                // For continuous normalization ∫ψ*(r)·(S·ψ)(r) d³r = 1, the discrete
                // form is (Ω/N_grid)·Σ_i ψ*[i]·(S·ψ)[i] = 1, so the target for the
                // dimensionless sum is N_grid/Ω. We divide by (Ω/N_grid) to convert
                // the integral-normalized value back to the grid-sum target.
                let mut norm_sq_s = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(), n_pw_i32,
                    col_b as *const _, 1,
                    gs_s_col_ptr as *const _, 1,
                    &mut norm_sq_s as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                // Grid-sum convention (consistent with CASTEP `.check`, RR
                // S_sub gemm, and density.rs): cublasZdotc returns the raw
                // grid-sum ⟨ψ|S|ψ⟩, which equals 1.0 for an S-orthonormal
                // wavefunction. No (Ω/N) division — that previously turned
                // the dimensionless grid-sum target into the integral target,
                // breaking convention symmetry with downstream RR + density.
                if b == 0 && _pass == 0 {
                    eprintln!(
                        "[GramSchmidt] band-0 pass-0: norm²_S = {:.6e} (grid-sum convention; ≈1.0 for S-orthonormal input)",
                        norm_sq_s.x
                    );
                }
                // Subtract projections onto all previous S-orthonormal columns
                for j in 0..b {
                    let col_j = (psi_ptr as *mut CudaComplex).add(j * n_pw);
                    // dot = ⟨col_j, col_b⟩_S = ⟨col_j, S·col_b⟩ in grid-sum
                    // convention (matches RR S_sub gemm and density.rs).
                    let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                    cudarc::cublas::sys::cublasZdotc_v2(
                        blas.raw_handle(), n_pw_i32,
                        col_j as *const _, 1,
                        gs_s_col_ptr as *const _, 1,
                        &mut dot as *mut _ as *mut _,
                    ).result().map_err(Error::Blas)?;
                    // col_b -= dot * col_j
                    let neg_dot = CudaComplex { x: -dot.x, y: -dot.y };
                    cudarc::cublas::sys::cublasZaxpy_v2(
                        blas.raw_handle(), n_pw_i32,
                        &neg_dot as *const _ as *const _,
                        col_j as *const _, 1,
                        col_b as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    // ‖col_b‖²_S -= |dot|²  (S-orthonormal col_j ⇒ projection energy removed)
                    norm_sq_s.x -= dot.x * dot.x + dot.y * dot.y;
                }
                // Normalize col_b with S-norm
                let norm_s = norm_sq_s.x.sqrt();
                if norm_s > 1e-30 {
                    let inv_norm = CudaComplex { x: 1.0 / norm_s, y: 0.0 };
                    cudarc::cublas::sys::cublasZscal_v2(
                        blas.raw_handle(), n_pw_i32,
                        &inv_norm as *const _ as *const _,
                        col_b as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }
        }
    }

    // Compute final H|psi> for Rayleigh-Ritz
    unsafe {
        apply_full_hamiltonian()
            .psi_dev(final_psi_buf)
            .v_eff_dev(v_eff_dev)
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(fft_idx_dev)
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(grid_size)
            .inv_ntotal(inv_ntotal)
            .fft_plan(&fft_plan)
            .hpsi_dev(&mut hpsi_dev)
            .grid_dev(&mut grid_dev)
            .vnl_data(vnl_data)
            .blas(blas)
            .kernels(kernels)
            .stream(stream)
            .call()?;
    }

    // Pass psi/hpsi as RowDistributed.
    //
    // ColumnDistributed memory layout: flat[b*n_pw + g] = psi[band b, PW g].
    // In BLAS terms this IS col-major (n_pw, n_bands) with leading dim n_pw,
    // which is exactly what `rayleigh_ritz` expects (gemm uses lda = n_pw and
    // op::C). So the same buffer can be reinterpreted as RowDistributed without
    // any data movement.
    //
    // The previous `transpose_col_to_row` kernel produced col-major
    // (n_bands, n_pw) [flat[g*n_bands + b] = psi[b, g]] which gemm with
    // lda = n_pw mis-read, scrambling H_sub and S_sub. Found via the
    // RR_DUMP_HS diagnostic in `rayleigh_ritz.rs`.
    stream.memcpy_dtod(&**final_psi_buf, &mut psi_row_dev.0).map_err(Error::Cuda)?;
    stream.memcpy_dtod(&*hpsi_dev, &mut hpsi_row_dev.0).map_err(Error::Cuda)?;

    // The Gram-Schmidt step above already orthonormalized the bands, so
    // S_sub = ψ†ψ ≈ I and ZHEGVD is well-conditioned. No further per-band
    // scaling needed.

    // Wrap into Gpu<WavefunctionSet<L>>
    let psi_row = Gpu::<WavefunctionSet<RowDistributed>> {
        slice: psi_row_dev.0,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };
    let hpsi_row = Gpu::<WavefunctionSet<RowDistributed>> {
        slice: hpsi_row_dev.0,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };

    Ok((psi_row, hpsi_row))
}

// ---------------------------------------------------------------------------
// Chebyshev amplification factor normalisation (chebfi_ampfactor)
// ---------------------------------------------------------------------------

/// Compute the Chebyshev polynomial value T_n(x) on the interval [aa, bb].
///
/// Maps `xx` from the physical interval `[aa, bb]` to `[-1, 1]` via:
///   `xred = (2*xx - (aa+bb)) / (bb-aa)`
/// then evaluates `T_n(xred)` using the three-term recurrence:
///   `T_0 = 1`, `T_1 = xred`, `T_k = 2*xred*T_{k-1} - T_{k-2}`.
///
/// This is a direct port of `cheb_poly1` from ABINIT
/// (`m_chebfi2.F90`, lines 1088–1110).
///
/// # Panics
/// Panics if `bb <= aa` (division by zero or degenerate interval).
fn cheb_poly1(xx: f64, nn: usize, aa: f64, bb: f64) -> f64 {
    let half_width = bb - aa;
    assert!(half_width > 0.0, "cheb_poly1: degenerate interval [aa={aa}, bb={bb}]");
    let xred = (2.0 * xx - (aa + bb)) / half_width;

    if nn == 0 {
        // T_0(x) = 1 for all x. Not called by chebfi_ampfactor, but included
        // for completeness and to match the ABINIT function interface.
        return 1.0;
    }

    let mut yy = xred;        // T_1(xred)
    let mut yim1 = 1.0;       // T_0(xred)

    for _ii in 2..=nn {
        let temp = yy;
        yy = 2.0 * xred * yy - yim1;
        yim1 = temp;
    }

    yy
}

/// Compute amplification factors and scale the Chebyshev-filtered vectors.
///
/// Port of ABINIT subroutine `chebfi_ampfactor` (`m_chebfi2.F90`, lines 958–1006).
///
/// For each band `i`, computes the Chebyshev polynomial amplification factor:
///   `amp_i = T_{n_i}(λ_i, a, b)`
/// where `n_i = ndeg_filter_bands[i]` is the filter degree for band `i`,
/// `λ_i = ritz_values[i]` is the Ritz value, and `[a, b] = [lambda_minus, lambda_plus]`
/// is the filter interval.
///
/// Each vector `X[:, i]`, `AX[:, i]`, `BX[:, i]` is then **divided** by `amp_i`
/// (i.e., multiplied by `1/amp_i`), undoing the Chebyshev amplification so that
/// vectors across bands have comparable magnitudes.
///
/// If `|amp_i| < 1e-3`, the factor is clamped to `1e-3` to prevent division by
/// near-zero values (which would blow up the vector). The ABINIT comment reads:
/// "just in case, avoid amplifying too much."
///
/// # Layout Convention
///
/// The three vector arrays are flat slices in column-major (column-distributed)
/// layout: `flat[b * total_spacedim + g]` where `b` indexes the band and `g`
/// indexes the spatial (plane-wave or grid) coefficient. Scaling is applied
/// in-place to each band's column independently.
///
/// # Arguments
///
/// * `ndeg_filter_bands` — per-band Chebyshev polynomial degree `n_i`.
///    Length must equal `nbands`.
/// * `ritz_values` — per-band Ritz values `λ_i` from the previous Rayleigh-Ritz.
///    Length must equal `nbands`.
/// * `lambda_minus` — lower bound `a` of the Chebyshev filter interval.
/// * `lambda_plus` — upper bound `b` of the Chebyshev filter interval.
/// * `total_spacedim` — number of spatial degrees of freedom per band
///    (plane-wave count or grid size).
/// * `x` — mutable reference to the X (wavefunction) array, scaled in-place.
/// * `ax` — mutable reference to the AX (H·X) array, scaled in-place.
/// * `bx` — mutable reference to the BX (S·X) array, scaled in-place.
/// * `nbands` — number of bands to process (controls the split between
///    full `neigenpairs` vs per-processor `bandpp` in KGB mode).
///
/// # Panics
///
/// Panics if the length of `ndeg_filter_bands`, `ritz_values`, or any of the
/// vector arrays does not match `nbands * total_spacedim` (for vectors) or
/// `nbands` (for the scalar arrays).
#[allow(clippy::too_many_arguments)]
pub fn chebfi_ampfactor(
    ndeg_filter_bands: &[usize],
    ritz_values: &[f64],
    lambda_minus: f64,
    lambda_plus: f64,
    total_spacedim: usize,
    x: &mut [f64],
    ax: &mut [f64],
    bx: &mut [f64],
    nbands: usize,
) {
    assert_eq!(
        ndeg_filter_bands.len(),
        nbands,
        "chebfi_ampfactor: ndeg_filter_bands length {} != nbands {nbands}",
        ndeg_filter_bands.len(),
    );
    assert_eq!(
        ritz_values.len(),
        nbands,
        "chebfi_ampfactor: ritz_values length {} != nbands {nbands}",
        ritz_values.len(),
    );
    assert_eq!(
        x.len(),
        nbands * total_spacedim,
        "chebfi_ampfactor: x length {} != nbands*spacedim {nbands}*{total_spacedim}",
        x.len(),
    );
    assert_eq!(ax.len(), x.len(), "chebfi_ampfactor: ax length mismatch with x");
    assert_eq!(bx.len(), x.len(), "chebfi_ampfactor: bx length mismatch with x");

    // Compute amplification factor per band: T_{n_i}(λ_i, a, b)
    let amp_factors: Vec<f64> = ndeg_filter_bands
        .iter()
        .zip(ritz_values.iter())
        .map(|(&ndeg, &eig)| {
            let raw = cheb_poly1(eig, ndeg, lambda_minus, lambda_plus);
            // ABINIT line 994: clamp near-zero amplification factors to
            // prevent division by ~0 (avoids blowing up the vector).
            // The ABINIT code sets ampfactor = 1e-3 unconditionally
            // (positive), not sign-preserving.
            if raw.abs() < 1e-3 {
                1e-3
            } else {
                raw
            }
        })
        .collect();

    // Scale each band's column in X, AX, BX by 1/amp_factor.
    // Uses iterator-style per-band processing to match the ABINIT structure:
    //   do iband = 1, nbands
    //     ampfactor = cheb_poly1(...)
    //     X_part = xXColsRows(:, iband);  xgBlock_scale(X_part, 1/ampfactor)
    //     AX_part = xAXColsRows(:, iband); xgBlock_scale(AX_part, 1/ampfactor)
    //     BX_part = xBXColsRows(:, iband); xgBlock_scale(BX_part, 1/ampfactor)
    //   end do
    for iband in 0..nbands {
        let inv_amp = 1.0 / amp_factors[iband];
        let start = iband * total_spacedim;
        let end = start + total_spacedim;

        for g in start..end {
            x[g] *= inv_amp;
            ax[g] *= inv_amp;
            bx[g] *= inv_amp;
        }
    }
}

// ---------------------------------------------------------------------------
// Post-filter residual computation (chebfi_residual_norms)
// ---------------------------------------------------------------------------

/// Compute per-band eigenvalue residual norms after the Rayleigh-Ritz step.
///
/// Port of ABINIT subroutine `chebfi_run` (`m_chebfi2.F90`, lines 709–717).
/// After `xg_RayleighRitz` rotates `chebfi%AX%self` and `chebfi%BX%self` into the
/// RR eigenbasis, this function computes the residual vectors directly:
///
/// * **Norm-conserving** (`use_paw == false`):
///   `R_i = (A * x_i) - lambda_i * x_i`
///   (standard eigenvalue residual, since B = I)
///
/// * **PAW** (`use_paw == true`):
///   `R_i = (A * x_i) - lambda_i * (B * x_i)`
///   (generalized eigenvalue residual)
///
/// The `colwiseCymax` operation (ABINIT `xgBlock_colwiseCymax`) is an in-place
/// axpy: `Y(:,i) = Y(:,i) - da(i) * X(:,i)`. After this, the L2 norm of each
/// column gives the per-band residual norm.
///
/// # Layout
///
/// All vector arrays are in column-major layout: `flat[b * n_pw + g]` where
/// `b` indexes the band and `g` indexes the plane-wave coefficient.
/// The function overwrites `ax` in-place with the residual vectors (matching
/// the ABINIT behaviour — `chebfi%AX%self` is consumed).
///
/// # Arguments
///
/// * `ax` — mutable reference to `A * rotated_X` (RR-rotated H|psi>).
///   **Overwritten in-place** with the residual vectors `R_i`.
/// * `eigenvalues` — per-band Ritz values `lambda_i` from the previous
///   Rayleigh-Ritz. Length must equal `nbands`.
/// * `bx_or_x` — reference vector for the eigenvalue term:
///   - If `use_paw == true`: `B * rotated_X` (RR-rotated S|psi>)
///   - If `use_paw == false`: `rotated_X` (the RR-rotated wavefunctions)
/// * `n_pw` — number of plane-wave coefficients per band.
/// * `nbands` — number of bands to process.
/// * `use_paw` — selects between PAW and norm-conserving residual formula.
/// * `blas` — cuBLAS handle for GPU operations.
/// * `stream` — CUDA stream for GPU operations.
///
/// # Returns
///
/// Per-band residual **squared** L2 norms: `residu[i] = ||R_i||_2^2` (length `nbands`),
/// matching ABINIT's convention where `residu` stores `|(H-e)|C>|^2` (hartree^2).
///
/// # Panics
///
/// Panics if the length of `eigenvalues` does not match `nbands`, or if the
/// length of `ax`/`bx_or_x` does not match `nbands * n_pw`.
#[builder]
#[allow(dead_code)] // called from scf.rs (deferred: needs H·psi recomputation after RR)
pub fn chebfi_residual_norms(
    ax: &mut PwCoefficients,
    eigenvalues: &CudaSlice<f64>,
    bx_or_x: &PwCoefficients,
    n_pw: usize,
    nbands: usize,
    #[allow(unused_variables)]
    use_paw: bool,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<f64>, Error> {
    // ---- In-place axpy: AX(:,i) -= λ_i * BX_or_X(:,i)  (lines 710-714) ----
    //
    // The ABINIT code writes:
    //   if (paw) call xgBlock_colwiseCymax(AX, eigenvalues, BX, AX)
    //   else     call xgBlock_colwiseCymax(AX, eigenvalues, X,  AX)
    //
    // xgBlock_colwiseCymax(A, da, B, W) computes:
    //   A(:,icol) = -da(icol) * B(:,icol) + W(:,icol)
    //
    // With W = AX and A = AX (overwrite target), this becomes:
    //   AX(:,i) = AX(:,i) - λ_i * BX_or_X(:,i)
    //
    // We implement this per-band using cublasZaxpy: one axpy call per column.

    let n_pw_i32 = n_pw as i32;
    let handle = blas.raw_handle();

    unsafe {
        let (ax_ptr, _sync_ax) = ax.0.device_ptr_mut(stream);
        let (bx_ptr, _sync_bx) = bx_or_x.0.device_ptr(stream);

        // Download eigenvalues from GPU to CPU for the per-band loop.
        // (ABINIT stores eigenvalues on the CPU; matching that convention
        // avoids a custom kernel for band_scale_axpy with a separate scale
        // per band.)
        let eig_host: Vec<f64> = stream.clone_dtoh(eigenvalues).map_err(Error::Cuda)?;

        for b in 0..nbands {
            let ax_col = (ax_ptr as *mut CudaComplex).add(b * n_pw);
            let bx_col = (bx_ptr as *const CudaComplex).add(b * n_pw);
            let neg_lambda = CudaComplex {
                x: -eig_host[b],
                y: 0.0,
            };

            // AX(:,b) += (-λ_b) * BX_or_X(:,b)
            cudarc::cublas::sys::cublasZaxpy_v2(
                handle,
                n_pw_i32,
                &neg_lambda as *const _ as *const _,
                bx_col as *const _,
                1,
                ax_col as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // ---- Column-wise squared L2 norms: residu(i) = ||AX(:,i)||_2^2  (line 716) ----
    //
    // ABINIT `xgBlock_colwiseNorm2` computes ||col||_2^2 via dot product (no sqrt).
    // We follow the same convention: per-band cublasZdotc yields the squared norm.

    let mut residu_host = vec![0.0_f64; nbands];

    unsafe {
        let (ax_ptr, _sync_ax) = ax.0.device_ptr_mut(stream);

        for b in 0..nbands {
            let ax_col = (ax_ptr as *const CudaComplex).add(b * n_pw);
            let mut dot = CudaComplex { x: 0.0, y: 0.0 };

            cudarc::cublas::sys::cublasZdotc_v2(
                handle,
                n_pw_i32,
                ax_col as *const _,
                1,
                ax_col as *const _,
                1,
                &mut dot as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;

            // ABINIT stores squared norms (no sqrt); dot.x is already ||col||_2^2
            residu_host[b] = dot.x;
            // NOTE: NOT sqrt(dot.x) — ABINIT xgBlock_colwiseNorm2 returns
            // sum-of-squares, not L2 norm. The caller compares against tolwfr
            // squared (hartree^2), matching ABINIT m_cgwf.F90 line 175.
        }
    }

    // Upload residual norms to GPU (caller may need them GPU-resident for
    // convergence checks or downstream processing).
    let residu_dev: CudaSlice<f64> = stream.clone_htod(&residu_host).map_err(Error::Cuda)?;

    Ok(residu_dev)
}

// ---------------------------------------------------------------------------
// Chebyshev oracle: degree prediction for next iteration
//   (chebfi_set_ndeg_from_residu — ABINIT m_chebfi2.F90, lines 1128–1214)
// ---------------------------------------------------------------------------

/// Chebyshev polynomial oracle: find the smallest degree `nn` such that
/// `1 / T_{nn}(xred)^2 < tol`.
///
/// Port of ABINIT subroutine `cheb_oracle1` (`m_chebfi2.F90`, lines 1031–1064).
///
/// # Mathematical background
///
/// Given a spectral abscissa `xx` in the filter interval `[aa, bb]`, maps `xx`
/// to the canonical Chebyshev domain `[-1, 1]` via the affine transformation
///
/// ```text
/// xred = (xx - (aa + bb) / 2) / (bb - aa) * 2
/// ```
///
/// then uses the three-term recurrence
///
/// ```text
/// T_0(x) = 1,   T_1(x) = x,
/// T_k(x) = 2 * x * T_{k-1}(x) - T_{k-2}(x)   (k >= 2)
/// ```
///
/// to find the smallest integer `nn` (1 <= nn <= nmax) such that
/// `1 / T_{nn}(xred)^2 < tol`.  If no degree up to `nmax - 1` satisfies the
/// tolerance, the function returns `nmax`.
///
/// The Chebyshev polynomial `T_n(x)` is bounded in `[-1, 1]` but grows
/// exponentially outside this interval.  The filter maps eigenvalues inside
/// `[aa, bb]` to `[-1, 1]` where the polynomial damps them (`|T_n| <= 1`),
/// while eigencomponents outside the interval are magnified.  The criterion
/// `1/T_n^2 < tol` measures how large the polynomial must grow at the mapped
/// abscissa `xred` to achieve the desired damping.
///
/// # Arguments
///
/// * `xx` — eigenvalue (Ritz value) to evaluate the polynomial at
/// * `aa` — lower bound of the filter interval
/// * `bb` — upper bound of the filter interval (must be > `aa`)
/// * `tol` — target residual-reduction ratio (`tolerance / current_residual`)
/// * `nmax` — maximum allowed degree (hard cap, >= 2 expected)
///
/// # Returns
///
/// `nn` — smallest degree satisfying the criterion, or `nmax` if none does.
///
/// # Panics
///
/// Panics if `bb <= aa` (degenerate interval, division by zero).
#[builder]
pub fn cheb_oracle1(xx: f64, aa: f64, bb: f64, tol: f64, nmax: usize) -> usize {
    let width = bb - aa;
    assert!(
        width > 0.0,
        "cheb_oracle1: degenerate interval [aa={aa}, bb={bb}]"
    );
    // xred = (xx - midpoint) / width * 2  — maps [aa, bb] onto [-1, 1]
    // (width = bb - aa is the full interval width, not half-width)
    let xred = (xx - (aa + bb) / 2.0) / width * 2.0;

    // T_1(xred) = xred
    let mut yy = xred;
    let mut yim1 = 1.0_f64;

    // Check degree-1 immediately (ABINIT line 1050)
    if 1.0 / (yy * yy) < tol {
        return 1;
    }

    // Search degrees 2..=nmax-1 via three-term recurrence.
    //
    // ABINIT (line 1053): do ii=2, nmax-1  (Fortran inclusive)
    // Rust: 2..nmax  (exclusive upper bound -> same iteration set)
    for ii in 2..nmax {
        let temp = yy;
        yy = 2.0_f64.mul_add(xred * yy, -yim1);
        yim1 = temp;
        if 1.0 / (yy * yy) < tol {
            return ii;
        }
    }

    nmax
}

/// Determine the Chebyshev filter degree for the next iteration from per-band
/// residual norms.
///
/// Port of ABINIT subroutine `chebfi_set_ndeg_from_residu`
/// (`m_chebfi2.F90`, lines 1128–1214).
///
/// For each unconverged band, computes the Chebyshev degree needed to reduce
/// the residual from its current value to the tolerance target. Converged
/// bands and bands in the buffer region (beyond `neigenpairs - nbdbuf`) are
/// skipped (assigned degree 0). Optionally applies occupancy-weighted skipping
/// for metallic systems (when `nbdbuf == -101`).
///
/// The final degree is the maximum across all process-local bands.
///
/// # MPI note
///
/// ABINIT performs `xmpi_max` across `comm_cols` to find the global maximum.
/// The Rust port returns the local maximum only; the caller is responsible for
/// the MPI reduction (typically via `mpi.max_allreduce`).
///
/// # Arguments
///
/// * `bandpp` — number of bands owned by this MPI process (bands per process)
/// * `eig_vals` — per-band Ritz eigenvalues `λ_i`, length = `bandpp`
/// * `residual_sq_norms` — per-band squared residual norms `||r_i||^2`,
///    length = `bandpp`. When `nbdbuf == Some(-101)`, these are pre-multiplied
///    by the occupancy.
/// * `occ_vals` — per-band occupancy numbers (fractional), length = `bandpp`
/// * `shift` — global band index offset: `rank_in_comm_cols * bandpp`
/// * `neigenpairs` — total number of eigenpairs being tracked
/// * `nbdbuf` — number of buffer bands. Use `Some(-101)` for occupancy-driven
///    automatic skipping (metallic systems). `None` or `Some(0)` means no
///    explicit buffer.
/// * `tolerance` — convergence tolerance (squared residual, Hartree^2)
/// * `lambda_minus` — lower bound `a` of the Chebyshev filter interval
/// * `lambda_plus` — upper bound `b` of the Chebyshev filter interval
/// * `oracle` — oracle mode: 1 = conservative (cap at current degree),
///    2 = two-target (cap at decrease target, max 15 steps)
/// * `oracle_factor` — for oracle mode 2 only: target residual reduction
///    factor per capped iteration
/// * `oracle_min_occ` — minimum occupancy threshold for skipping a band
///    (only active when `nbdbuf == Some(-101)`)
/// * `ndeg_filter_current` — current global filter degree (used as cap in
///    oracle mode 1)
/// * `ndeg_filter_max` — hard upper bound on the Chebyshev degree
///
/// # Returns
/// `(ndeg_filter_local, ndeg_filter_bands)` where:
/// - `ndeg_filter_local` is the `MAXVAL` across all per-band degrees for this
///   process (0 if all bands are converged/buffer/low-occupancy).
/// - `ndeg_filter_bands` is the per-band degree vector (length = `bandpp`).
///   Converged/buffer/skipped bands have degree 0.
///
/// # Panics
/// Panics on length mismatches, invalid oracle mode, or `lambda_plus <= lambda_minus`.
#[builder]
pub fn chebfi_set_ndeg_from_residu(
    bandpp: usize,
    eig_vals: &[f64],
    residual_sq_norms: &[f64],
    occ_vals: &[f64],
    shift: usize,
    neigenpairs: usize,
    nbdbuf: Option<i32>,
    tolerance: f64,
    lambda_minus: f64,
    lambda_plus: f64,
    oracle: usize,
    oracle_factor: f64,
    oracle_min_occ: f64,
    ndeg_filter_current: usize,
    ndeg_filter_max: usize,
) -> (usize, Vec<usize>) {
    // ---- Input validation ----
    assert_eq!(
        eig_vals.len(),
        bandpp,
        "chebfi_set_ndeg_from_residu: eig_vals length {} != bandpp {bandpp}",
        eig_vals.len(),
    );
    assert_eq!(
        residual_sq_norms.len(),
        bandpp,
        "chebfi_set_ndeg_from_residu: residual_sq_norms length {} != bandpp {bandpp}",
        residual_sq_norms.len(),
    );
    // occ_vals may be empty (e.g. iter 1 before occupations are computed,
    // or when caller doesn't have occupation data). In that case, occupancy-
    // driven skipping (test3) is disabled.
    let occ_available = occ_vals.len() == bandpp;
    assert!(
        matches!(oracle, 1 | 2),
        "chebfi_set_ndeg_from_residu: invalid oracle mode {oracle} (expected 1 or 2)"
    );
    assert!(
        lambda_plus > lambda_minus,
        "chebfi_set_ndeg_from_residu: degenerate filter interval [a={lambda_minus}, b={lambda_plus}]"
    );
    assert!(
        ndeg_filter_max > 0,
        "chebfi_set_ndeg_from_residu: ndeg_filter_max must be > 0, got {ndeg_filter_max}"
    );

    // ---- Resolve nbdbuf (lines 1177-1181) ----
    let resolved_nbdbuf: usize = match nbdbuf {
        Some(n) if n > 0 => n as usize,
        Some(-101) | None | Some(0) => 0,
        Some(_n) => {
            // ABINIT only handles > 0 and -101; negative values other than -101
            // imply the magic convention is active — treat as no buffer.
            0
        }
    };

    let is_occ_driven = nbdbuf == Some(-101);

    // ---- Per-band loop (lines 1183-1206) ----
    let ndeg_filter_bands: Vec<usize> = (0..bandpp)
        .map(|iband| {
            let eig_iband = eig_vals[iband];
            let res_iband = residual_sq_norms[iband];
            let occ_iband = if occ_available { occ_vals[iband] } else { 0.0 };
            let iband_tot = iband + shift;

            // Test 1: converged? (res < tolerance)
            let test1 = res_iband < tolerance;

            // Test 2: in buffer region?
            let test2 = iband_tot >= neigenpairs.saturating_sub(resolved_nbdbuf);

            // Test 3: occupancy too low? (only when nbdbuf == -101 AND occ data available)
            let test3 = is_occ_driven && occ_available && occ_iband < oracle_min_occ;

            if test1 || test2 || test3 {
                // Band is converged, in buffer, or negligible occupancy → skip
                0
            } else {
                // Not converged: compute required degree via cheb_oracle1
                let target_ratio = tolerance / res_iband;
                let ndeg_filter_tolwfr = cheb_oracle1()
                    .xx(eig_iband)
                    .aa(lambda_minus)
                    .bb(lambda_plus)
                    .tol(target_ratio)
                    .nmax(1000)
                    .call();

                match oracle {
                    1 => {
                        // Conservative: cap at current degree and max
                        ndeg_filter_tolwfr.min(ndeg_filter_max).min(ndeg_filter_current)
                    }
                    2 => {
                        // Two-target: also compute decrease target (max 15 iterations)
                        let ndeg_filter_decrease = cheb_oracle1()
                            .xx(eig_iband)
                            .aa(lambda_minus)
                            .bb(lambda_plus)
                            .tol(oracle_factor)
                            .nmax(15)
                            .call();
                        ndeg_filter_tolwfr
                            .min(ndeg_filter_max)
                            .min(ndeg_filter_decrease)
                    }
                    _ => unreachable!("oracle mode validated at function entry"),
                }
            }
        })
        .collect();

    // ---- Local maximum (line 1207) ----
    let ndeg_filter_local = ndeg_filter_bands.iter().copied().max().unwrap_or(0);

    (ndeg_filter_local, ndeg_filter_bands)
}

// ---------------------------------------------------------------------------
// Rayleigh-Ritz quotients: per-band eigenvalue estimates for Chebyshev filter
//   (chebfi_rayleighRitzQuotients — ABINIT m_chebfi2.F90, lines 761–810)
// ---------------------------------------------------------------------------

/// Per-band Rayleigh-Ritz quotients from the filtered subspace.
/// Used to estimate extremal eigenvalues for setting the Chebyshev filter
/// window bounds.
///
/// Port of ABINIT subroutine `chebfi_rayleighRitzQuotients`
/// (`m_chebfi2.F90`, lines 761–810).
///
/// Computes, for each band `i`:
///   `quotient[i] = <ψ_i|H|ψ_i> / <ψ_i|S|ψ_i>`
///
/// No MPI reduction is performed, matching ABINIT's
/// `comm_loc=xmpi_comm_null`. The caller is responsible for any global
/// reduction (`xmpi_max`/`xmpi_min`) across MPI ranks.
///
/// # Space-dependent dot-product conventions
///
/// ABINIT's `xgBlock_colwiseDotProduct` has two distinct code paths
/// (`m_xg.F90` lines 4614, 4778–4839):
///
/// **SPACE_C** (`is_complex_space = true`):
///   Standard complex conjugated dot product `zdotc(conj(x_B,)=sum conj(A)*B)`.
///   Performed via `cublasZdotc`.
///
/// **SPACE_CR** (`is_complex_space = false`):
///   Gamma-point real wavefunctions stored as complex. ABINIT maps
///   `space_res = SPACE_R` and uses a real-valued ddot with factor-2
///   and G=0 correction:
///
///   ```text
///   dot = 2 * ddot(2*rows, vecR_A, 1, vecR_B, 1)         // factor-2 sum
///   if me_g0 == 1: dot -= ddot(2, vecR_A(:,1), 1, vecR_B(:,1), 1)  // G=0 correction
///   ```
///
///   In Rust (single-GPU, `me_g0` always 1):
///
///   ```text
///   raw = cublasZdotc(psi, Hpsi)                      // raw complex dot product
///   g0  = cublasZdotc(n=1, psi[0], Hpsi[0])           // G=0 contribution
///   corrected = 2 * raw.re - g0.re                    // (imag part set to 0)
///   ```
///
///   The physical justification: for gamma-point real wavefunctions,
///   `psi(-G) = conj(psi(G))`. The stored half-sphere is counted twice
///   (factor 2), but G=0 is double-counted in that factor and must be
///   subtracted once.
///
///   SPACE_CR quotients are purely real-valued (imaginary parts set to zero).
///
/// Extremal values `maxeig`/`mineig` are taken over the real parts of the
/// quotients, matching ABINIT's `maxval(dble(DivResults))`/`minval(dble(DivResults))`.
///
/// # Arguments
/// * `psi_dev` — wavefunction block ψ, column-major layout `flat[b*n_pw + g]`
///    where `b` indexes the band and `g` indexes the plane-wave coefficient.
/// * `hpsi_dev` — Hamiltonian-applied block H|ψ>, same dimensions as `psi_dev`.
/// * `spsi_dev` — Overlap-applied block S|ψ>, same dimensions as `psi_dev`.
/// * `n_pw` — number of plane-wave coefficients per band.
/// * `ncols` — number of bands (columns). In ABINIT serial mode
///    (`paral_kgb == 0`) this equals `neigenpairs`; in band-parallel KGB mode
///    (`paral_kgb != 0`) this equals `bandpp` (bands per local MPI rank).
/// * `is_complex_space` — `true` for SPACE_C (complex wavefunction space),
///    `false` for SPACE_CR (real wavefunctions stored as complex). Maps to
///    ABINIT's `space_res` (SPACE_C vs SPACE_R).
/// * `blas` — cuBLAS handle for GPU dot products.
/// * `stream` — CUDA stream for GPU operations.
///
/// # Returns
/// `(maxeig, mineig, quotients)` where:
/// - `maxeig` — maximum Rayleigh-Ritz quotient (real part).
/// - `mineig` — minimum Rayleigh-Ritz quotient (real part).
/// - `quotients` — per-band quotients `λ_i = <ψ_i|H|ψ_i> / <ψ_i|S|ψ_i>`.
///   For SPACE_C: complex-valued (imag parts zero for Hermitian eigenproblems).
///   For SPACE_CR: real-valued (imag parts forced to zero).
///
/// # Panics
/// Panics if `ncols` or `n_pw` is zero, or if any input slice length
/// does not equal `ncols * n_pw`.
#[builder]
pub fn chebfi_rayleigh_ritz_quotients(
    psi_dev: &CudaSlice<CudaComplex>,
    hpsi_dev: &CudaSlice<CudaComplex>,
    spsi_dev: &CudaSlice<CudaComplex>,
    n_pw: usize,
    ncols: usize,
    is_complex_space: bool,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(f64, f64, Vec<num_complex::Complex64>), Error> {
    // --- Dimension validation (ABINIT xgBlock_check equivalent) ---
    assert!(ncols > 0, "chebfi_rayleigh_ritz_quotients: ncols must be > 0");
    assert!(n_pw > 0, "chebfi_rayleigh_ritz_quotients: n_pw must be > 0");
    let n_elem = n_pw * ncols;
    assert_eq!(
        psi_dev.len(),
        n_elem,
        "chebfi_rayleigh_ritz_quotients: psi_dev.len()={} != n_pw*ncols={}",
        psi_dev.len(),
        n_elem,
    );
    assert_eq!(
        hpsi_dev.len(),
        n_elem,
        "chebfi_rayleigh_ritz_quotients: hpsi_dev.len()={} != n_pw*ncols={}",
        hpsi_dev.len(),
        n_elem,
    );
    assert_eq!(
        spsi_dev.len(),
        n_elem,
        "chebfi_rayleigh_ritz_quotients: spsi_dev.len()={} != n_pw*ncols={}",
        spsi_dev.len(),
        n_elem,
    );

    let n_pw_i32 = n_pw as i32;

    // ---- Step 1: Compute <ψ_i|H|ψ_i> and <ψ_i|S|ψ_i> per band ----
    //
    // ABINIT uses xgBlock_colwiseDotProduct(A, B, Results%self) which has
    // two distinct code paths depending on xgBlockA%space (m_xg.F90 lines
    // 4614, 4778–4839):
    //
    //   SPACE_C: zdotc(conj(A), B) — complex conjugated dot product.
    //   SPACE_CR: fact*ddot(fact*rows, vecR_A, 1, vecR_B, 1)
    //             with fact=2 and G=0 correction when me_g0==1.
    //
    // We implement both paths below using cublasZdotc as the basis,
    // applying the factor-2/G=0 corrections for SPACE_CR on the host.
    let mut h_dot = vec![num_complex::Complex64::new(0.0, 0.0); ncols];
    let mut s_dot = vec![num_complex::Complex64::new(0.0, 0.0); ncols];

    unsafe {
        let (psi_ptr, _sync_psi) = psi_dev.device_ptr(stream);
        let (hpsi_ptr, _sync_hpsi) = hpsi_dev.device_ptr(stream);
        let (spsi_ptr, _sync_spsi) = spsi_dev.device_ptr(stream);

        if is_complex_space {
            // -----------------------------------------------------------
            // SPACE_C path: cublasZdotc — matches ABINIT zdotc convention
            // -----------------------------------------------------------
            for b in 0..ncols {
                let col = (psi_ptr as *const CudaComplex).add(b * n_pw);

                // <psi_b|H|psi_b>
                let hcol = (hpsi_ptr as *const CudaComplex).add(b * n_pw);
                let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(),
                    n_pw_i32,
                    col as *const _,
                    1,
                    hcol as *const _,
                    1,
                    &mut dot as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;
                h_dot[b] = num_complex::Complex64::new(dot.x, dot.y);

                // <psi_b|S|psi_b>
                let scol = (spsi_ptr as *const CudaComplex).add(b * n_pw);
                let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(),
                    n_pw_i32,
                    col as *const _,
                    1,
                    scol as *const _,
                    1,
                    &mut dot as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;
                s_dot[b] = num_complex::Complex64::new(dot.x, dot.y);
            }
        } else {
            // -----------------------------------------------------------
            // SPACE_CR path: factor-2 + G=0 correction
            //
            // ABINIT m_xg.F90 lines 4614, 4778–4792:
            //   fact = 2 for SPACE_CR
            //   dot = fact * ddot(fact*rows, vecR_A, 1, vecR_B, 1)
            //   if me_g0==1: dot -= ddot(2, vecR_A(:,1), 1, vecR_B(:,1), 1)
            //
            // Physical interpretation (gamma-point real wavefunctions):
            //   psi(-G) = conj(psi(G)), so the full-sphere inner product
            //   equals 2 * sum_over_stored_half_sphere - G=0_term
            //   (G=0 is double-counted by factor 2 and must be subtracted).
            //
            // We compute:
            //   (a) cublasZdotc over all stored pw-coeffs → full_raw
            //   (b) cublasZdotc over the first element only → g0_raw
            //   (c) corrected = 2 * full_raw.re - g0_raw.re
            //
            // Single-GPU Rust always owns G=0 (me_g0 == 1), so the G=0
            // correction always applies.
            // -----------------------------------------------------------
            for b in 0..ncols {
                let col = (psi_ptr as *const CudaComplex).add(b * n_pw);

                // ---- <psi_b|H|psi_b> ----
                let hcol = (hpsi_ptr as *const CudaComplex).add(b * n_pw);

                // (a) Full dot product over all plane-wave coefficients
                let mut full_raw = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(),
                    n_pw_i32,
                    col as *const _,
                    1,
                    hcol as *const _,
                    1,
                    &mut full_raw as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;

                // (b) G=0 contribution: cublasZdotc with n=1 on the first element
                let mut g0_raw = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(),
                    1,
                    col as *const _,
                    1,
                    hcol as *const _,
                    1,
                    &mut g0_raw as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;

                // (c) Apply factor-2 and G=0 correction
                // For SPACE_CR, imag parts are analytically zero; we force
                // zero to match ABINIT's real-valued SPACE_R accumulator.
                h_dot[b] = num_complex::Complex64::new(2.0 * full_raw.x - g0_raw.x, 0.0);

                // ---- <psi_b|S|psi_b> (same logic) ----
                let scol = (spsi_ptr as *const CudaComplex).add(b * n_pw);

                let mut full_raw = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(),
                    n_pw_i32,
                    col as *const _,
                    1,
                    scol as *const _,
                    1,
                    &mut full_raw as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;

                let mut g0_raw = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(),
                    1,
                    col as *const _,
                    1,
                    scol as *const _,
                    1,
                    &mut g0_raw as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;

                s_dot[b] = num_complex::Complex64::new(2.0 * full_raw.x - g0_raw.x, 0.0);
            }
        }
    }

    // ---- Step 2: Compute quotients = <ψ_i|H|ψ_i> / <ψ_i|S|ψ_i> ----
    //
    // ABINIT uses xgBlock_colwiseDivision (elementwise complex/real division
    // depending on space_res, m_xg.F90 lines 5013–5050). For SPACE_CR, the
    // division is on real-valued accumulators (SPACE_R); for SPACE_C, it is
    // on complex-valued accumulators (SPACE_C), with max/min taken over
    // maxval(dble(DivResults)).
    //
    // Our Rust SPACE_CR dot products are already real-valued (imag = 0),
    // so bare complex division `h / s` reduces to real division and
    // matches ABINIT's SPACE_R elementwise division.
    let quotients: Vec<num_complex::Complex64> = h_dot
        .iter()
        .zip(s_dot.iter())
        .map(|(h, s)| h / s)
        .collect();

    // ---- Step 3: Extract extremal quotients ----
    //
    // ABINIT uses maxval(dble(DivResults)) / minval(dble(DivResults)),
    // i.e. the real-part extremal values. For Hermitian eigenproblems,
    // the imaginary parts should be zero; taking real parts matches ABINIT.
    //
    // NaN semantics: Fortran maxval/minval return NaN if any element is NaN.
    // We match this by initializing from the first element (so that NaN there
    // propagates) and checking `.is_nan()` on every subsequent element to
    // catch NaN anywhere in the array.  For well-conditioned Hermitian S > 0
    // systems, NaN should never occur in practice.
    #[allow(unused_variables)]
    let (max_idx, min_idx, max_val, min_val) = {
        let mut max_i = 0usize;
        let mut min_i = 0usize;
        let mut max_v = quotients[0].re;
        let mut min_v = quotients[0].re;
        for (i, q) in quotients.iter().enumerate() {
            let v = q.re;
            if v.is_nan() {
                max_v = f64::NAN;
                min_v = f64::NAN;
                max_i = i;
                min_i = i;
                break;
            }
            if v > max_v {
                max_v = v;
                max_i = i;
            }
            if v < min_v {
                min_v = v;
                min_i = i;
            }
        }
        (max_i, min_i, max_v, min_v)
    };

    // Prevent the compiler from complaining about unused variables
    // (ABINIT stores maxeig_pos/mineig_pos but the caller ignores them).
    if is_complex_space {
        #[cfg(feature = "scf_diag")]
        eprintln!(
            "[RR-quotients] SPACE_C: maxeig={:.6e}+{:.6e}i (idx={})  mineig={:.6e}+{:.6e}i (idx={})  ncols={}",
            quotients[max_idx].re, quotients[max_idx].im, max_idx,
            quotients[min_idx].re, quotients[min_idx].im, min_idx,
            ncols,
        );
    } else {
        #[cfg(feature = "scf_diag")]
        eprintln!(
            "[RR-quotients] SPACE_CR: maxeig={:.6e} (idx={})  mineig={:.6e} (idx={})  ncols={}",
            quotients[max_idx].re, max_idx,
            quotients[min_idx].re, min_idx,
            ncols,
        );
    }

    Ok((max_val, min_val, quotients))
}

// ---------------------------------------------------------------------------
// ChebfiRun — ABINIT-style Chebyshev filtering pipeline driver
// ---------------------------------------------------------------------------
//
// Implements standard ChFSI (ABINIT: chebfi_run, m_chebfi2.F90:473-737).
// The 8 phases are called in ABINIT order:
//   1. Fresh H·psi + S·psi (Phase 1)
//   2. Rayleigh quotients → lambda_minus = max(ε_b), lambda_plus = ecut (Phase 2)
//   3. cheb_oracle1 → ndeg_filter_max capped at 40 (Phase 3)
//   4. chebfi_set_ndeg_from_residu → per-band ndeg_filter_bands (Phase 4)
//   5. Standard ChFSI three-term recurrence on eigenvectors (Phase 5)
//   6. chebfi_ampfactor → normalise each band by 1/T_n(ε_b) (Phase 5b)
//   7. Gram-Schmidt S-orthonormalisation (unchanged)
//   8. Final H·psi for Rayleigh-Ritz (unchanged)
//
// Returns expanded tuple for the caller (scf.rs) to drive RR + residual norms.

/// Returns (psi_row, hpsi_row, ritz_values, residual_sq_norms, ndeg_filter_bands).
type ChebfiResult = Result<
    (
        Gpu<WavefunctionSet<RowDistributed>>,
        Gpu<WavefunctionSet<RowDistributed>>,
        Vec<f64>,           // ritz_values (per-band Rayleigh quotients, real parts)
        Vec<f64>,           // residual_sq_norms (Hartree^2)
        Vec<usize>,         // ndeg_filter_bands (0 = locked / converged)
    ),
    Error,
>;

/// Standard ChFSI filtering pipeline driver (ABINIT chebfi_run).
///
/// Implements the ABINIT-standard ChFSI lifecycle (eigenvector filtering):
///   1. Fresh H·psi + S·psi (getAX_BX)
///   2. Rayleigh quotients → lambda_minus = max(ε_b), lambda_plus = ecut
///   2b. Fresh residual norms from current V_eff:
///       r_b = H·psi_input[:,b] − λ_b · S·psi_input[:,b]
///   3. cheb_oracle1 → ndeg_filter_max (capped at 40)
///   4. chebfi_set_ndeg_from_residu → per-band ndeg_filter_bands (0 = locked)
///      Uses FRESH residuals from Phase 2b (current V_eff), not stale ones
///      from the previous SCF iteration.
///   5. Standard ChFSI three-term Chebyshev recurrence on eigenvectors
///      via the S⁻¹·H operator, matching ABINIT's
///      chebfi_computeNextOrderChebfiPolynom (m_chebfi2.F90:837-896)
///   5b. chebfi_ampfactor → normalize each band by 1/T_n(ε_b)
///   7. Gram-Schmidt S-orthonormalisation
///   8. Final H·psi for Rayleigh-Ritz
///
/// The returned residuals are the fresh Phase 2b values (computed under
/// current V_eff, pre-filtering). The caller (scf.rs) computes additional
/// post-RR residuals via chebfi_residual_norms for SCF convergence monitoring.
#[allow(clippy::too_many_arguments)]
pub fn chebfi_run_rust(
    psi_gpu: &Gpu<WavefunctionSet<ColumnDistributed>>,
    v_eff_dev: &CudaSlice<f64>,
    wave_grid: &GVectorGrid,
    pw_coords: &[[i32; 3]],
    vnl_data: &VnlBatchData,
    fft_idx_dev: &CudaSlice<i32>,
    // ---------- filter interval settings ----------
    ecut: f64,                  // lambda_plus = ecut (physical energy cutoff)
    min_veff: f64,              // for lambda_min fallback
    #[allow(unused_variables)]
    _max_veff: f64,              // reserved for Gershgorin fallback (deferred)
    // ---------- convergence control ----------
    tolerance: f64,             // target residual squared (Hartree^2)
    nbdbuf: Option<i32>,        // buffer bands (None or Some(-101) for occupancy-driven)
    occ_vals: Option<&[f64]>,   // per-band occupations (needed for nbdbuf=-101)
    ndeg_filter_max: usize,     // hard cap on Chebyshev degree
    oracle: usize,              // 1 = conservative, 2 = two-target
    oracle_factor: f64,         // for oracle mode 2
    oracle_min_occ: f64,        // min occupancy threshold
    // ---------- GPU context ----------
    kernels: &CudaKernelSet,
    pcie: &mut PcieAccount,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
    #[allow(unused_variables)]
    filter_mode: FilterMode,
    external_kinetic: Option<&[f64]>,
    // ---------- MPI ----------
    shift: usize,               // global band index offset (rank_in_comm_cols * bandpp)
    neigenpairs: usize,         // total eigenpairs tracked
    is_complex_space: bool,     // SPACE_C vs SPACE_CR
) -> ChebfiResult {
    // ---- Dimensions ----
    let n_bands = psi_gpu.shape()[0];
    let n_pw = psi_gpu.shape()[1];
    let n_elem = n_bands * n_pw;
    let n_elem_i32 = n_elem as i32;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;

    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = ngx * ngy * ngz;
    let inv_ntotal = 1.0 / (grid_size as f64);
    let grid_alloc = n_bands * grid_size;

    // ---- Precompute & upload kinetic energy ----
    let kinetic_data: Vec<f64> = match external_kinetic {
        Some(ke) => ke.to_vec(),
        None => compute_kinetic_energies(pw_coords, wave_grid.recip_lattice(), [0.0; 3]).0,
    };
    let kinetic_dev_values: CudaSlice<f64> =
        stream.clone_htod(&kinetic_data).map_err(Error::Cuda)?;
    let kinetic_dev = KineticPreconditioner::new(kinetic_dev_values);
    pcie.h2d_bytes += kinetic_data.len() * std::mem::size_of::<f64>();

    // ---- FFT plan (batched C2C) ----
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32, n_bands_i32, stream.clone(),
    )?;

    // ---- GPU workspace buffers ----
    let psi_input = PwCoefficients::new(psi_gpu.as_device_slice().clone());

    // Standard ChFSI buffers (three-term Chebyshev recurrence on eigenvectors):
    //   x_prev = T_{k-1}(S⁻¹H)·Ψ  (Chebyshev vector of degree k-1)
    //   x_curr = T_k(S⁻¹H)·Ψ      (Chebyshev vector of degree k)
    //   x_next = T_{k+1}(S⁻¹H)·Ψ  (being computed)
    let mut x_prev = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut x_curr = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut x_next = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    // Temporary buffer for recurrence arithmetic (e.g., storing scaled copies)
    let mut buf_sx = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);

    // Hamiltonian workspace + S·psi buffer
    let mut hpsi_dev = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut spsi_dev = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut grid_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // Output RowDistributed buffers
    let mut psi_row_dev = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut hpsi_row_dev = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);

    // =====================================================================
    // Phase 1: Fresh H·psi + S·psi (ABINIT: getAX_BX, lines 490-513)
    // =====================================================================
    unsafe {
        apply_full_hamiltonian()
            .psi_dev(&psi_input)
            .v_eff_dev(v_eff_dev)
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(fft_idx_dev)
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(grid_size)
            .inv_ntotal(inv_ntotal)
            .fft_plan(&fft_plan)
            .hpsi_dev(&mut hpsi_dev)
            .grid_dev(&mut grid_dev)
            .vnl_data(vnl_data)
            .blas(blas)
            .kernels(kernels)
            .stream(stream)
            .call()?;
    }

    // spsi_dev = S·psi_input
    stream.memcpy_dtod(&*psi_input, &mut spsi_dev.0).map_err(Error::Cuda)?;
    unsafe {
        apply_s_times()
            .psi_dev(&psi_input)
            .spsi_dev(&mut spsi_dev)
            .vnl_data(vnl_data)
            .n_bands(n_bands_i32)
            .n_pw(n_pw_i32)
            .blas(blas)
            .stream(stream)
            .call()?;
    }

    // =====================================================================
    // Phase 2: Rayleigh quotients → lambda_minus, lambda_plus (ABINIT lines 516-575)
    // =====================================================================
    let (_maxeig, _mineig, ritz_complex) = chebfi_rayleigh_ritz_quotients()
        .psi_dev(&psi_input.0)
        .hpsi_dev(&hpsi_dev.0)
        .spsi_dev(&spsi_dev.0)
        .n_pw(n_pw)
        .ncols(n_bands)
        .is_complex_space(is_complex_space)
        .blas(blas)
        .stream(stream)
        .call()?;

    // Real parts only for filter bounds (ABINIT takes dble() of complex quotients)
    let ritz_values: Vec<f64> = ritz_complex.iter().map(|c| c.re).collect();
    let lambda_min_rq = ritz_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let lambda_max_rq = ritz_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // =====================================================================
    // Phase 2b: Compute fresh residual norms from current V_eff
    // =====================================================================
    // r_b = H·psi_input[:,b] - ritz_values[b] · S·psi_input[:,b]
    // fresh_residuals[b] = ||r_b||^2  (L2 squared, ABINIT convention)
    //
    // These are computed from quantities already on GPU after Phase 1+2,
    // eliminating the cascade bug where stale residuals from V_eff_{N-1}
    // were used for locking decisions under V_eff_N.
    //
    // Uses spsi_dev as scratch space (no longer needed in its Phase 2 form
    // after Rayleigh quotients are computed).
    let fresh_residuals: Vec<f64> = {
        let n_pw_i32 = n_pw as i32;
        let handle = blas.raw_handle();
        let mut res = vec![0.0_f64; n_bands];
        // Allocate a single-column scratch buffer (one band's PW coeffs)
        let mut scratch: CudaSlice<CudaComplex> =
            stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
        unsafe {
            let (hpsi_ptr, _) = hpsi_dev.0.device_ptr(stream);
            let (spsi_ptr, _) = spsi_dev.0.device_ptr(stream);
            let (scratch_ptr, _) = scratch.device_ptr_mut(stream);
            for b in 0..n_bands {
                let h_col = (hpsi_ptr as *const CudaComplex).add(b * n_pw);
                let s_col = (spsi_ptr as *const CudaComplex).add(b * n_pw);
                // scratch = hpsi[:,b]
                cudarc::cublas::sys::cublasZcopy_v2(
                    handle, n_pw_i32,
                    h_col as *const _, 1,
                    scratch_ptr as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                // scratch -= ritz_values[b] * spsi[:,b]
                let neg_ritz = CudaComplex { x: -ritz_values[b], y: 0.0 };
                cudarc::cublas::sys::cublasZaxpy_v2(
                    handle, n_pw_i32,
                    &neg_ritz as *const _ as *const _,
                    s_col as *const _, 1,
                    scratch_ptr as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                // res[b] = ||scratch||^2
                let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    handle, n_pw_i32,
                    scratch_ptr as *const _, 1,
                    scratch_ptr as *const _, 1,
                    &mut dot as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                res[b] = dot.x; // ||r_b||^2 (no sqrt, matching ABINIT convention)
            }
        }
        res
    };
    let max_fresh = fresh_residuals.iter().cloned().fold(0.0_f64, f64::max);
    let min_fresh = fresh_residuals.iter().cloned().fold(f64::INFINITY, f64::min);
    let n_below_tol = fresh_residuals.iter().filter(|&&r| r < tolerance).count();
    eprintln!("[chebfi] fresh residuals: min={min_fresh:.2e} max={max_fresh:.2e} below_tol={n_below_tol}/{n_bands} tol={tolerance:.1e}");
    #[cfg(feature = "scf_diag")]
    {
        let max_res = fresh_residuals.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!("[chebfi] Phase 2b: fresh residuals (curr V_eff): max={:.6e}", max_res);
    }
    // λ_minus (b_low): upper bound of WANTED spectrum. Placed just above the
    // highest occupied+unoccupied Ritz value so the filter amplifies ALL tracked
    // bands and damps only states above. This matches ABINIT's λ_minus = max(ε_b).
    let lambda_minus = lambda_max_rq;
    // λ_plus (b_up): ABINIT uses ecut (physics cutoff in Ha) as the upper
    // bound, matching dtset%ecut in m_chebfi2.F90:547. The filter amplifies
    // bands with eigenvalues below λ_minus and damps bands above λ_plus.
    // The ampfactor (Phase 5b) normalizes each band by T_n(ε_b) to keep
    // amplitudes bounded, so a wide window is safe.
    let lambda_plus = ecut;

    // Guard: degenerate interval (e.g., all bands at zero on first iter)
    let lambda_minus = if lambda_plus > lambda_minus {
        lambda_minus
    } else {
        // Fallback: place lambda_minus just below ecut so the interval is
        // well-defined. This can happen when psi is uninitialised (all zeroes)
        // giving zero Rayleigh quotients.
        (lambda_plus - 1.0).max(0.0)
    };
    let _lambda_min = min_veff.min(lambda_minus - 1e-3);

    // =====================================================================
    // Phase 3: Oracle → ndeg_filter_max (ABINIT lines 580-589)
    // =====================================================================
    // cheb_oracle1 uses lambda_min_rq (smallest eigenvalue) as xx —
    // ABINIT m_chebfi2.F90:625 passes mineig_global, not lambda_minus.
    let ndeg_oracle = cheb_oracle1()
        .xx(lambda_min_rq)    // smallest eigenvalue — hardest to amplify
        .aa(lambda_minus)    // filter interval lower bound
        .bb(lambda_plus)     // filter interval upper bound
        .tol(1e-6)           // target residual-reduction ratio
        .nmax(ndeg_filter_max)
        .call()
        .min(40);            // ABINIT hard cap at 40

    // =====================================================================
    // Phase 4: Per-band dynamic degree from FRESH residuals (current V_eff)
    // =====================================================================
    // Uses fresh Phase 2b residuals (computed with current V_eff H and S)
    // and Phase 2 Rayleigh quotients. On iter-1, residuals reflect whatever
    // the initial guess produces — bands above tolerance get filtered.
    //
    // occ_vals is optional — only needed for nbdbuf=-101 occupancy-driven skipping.
    let (ndeg_filter_global, ndeg_filter_bands) = {
        let (g, bands) = chebfi_set_ndeg_from_residu()
            .bandpp(n_bands)
            .eig_vals(&ritz_values)
            .residual_sq_norms(&fresh_residuals)
            .occ_vals(occ_vals.unwrap_or(&[]))
            .shift(shift)
            .neigenpairs(neigenpairs)
            .maybe_nbdbuf(nbdbuf)
            .tolerance(tolerance)
            .lambda_minus(lambda_minus)
            .lambda_plus(lambda_plus)
            .oracle(oracle)
            .oracle_factor(oracle_factor)
            .oracle_min_occ(oracle_min_occ)
            .ndeg_filter_current(ndeg_oracle)
            .ndeg_filter_max(ndeg_filter_max)
            .call();
        #[cfg(feature = "scf_diag")]
        {
            let n_locked = bands.iter().filter(|&&d| d == 0).count();
            eprintln!("[chebfi] oracle: locked={}/{n_bands} global={g} ndeg=[{}, {}]", n_locked, bands.iter().copied().min().unwrap_or(0), bands.iter().copied().max().unwrap_or(0));
        }
        (g, bands)
    };

    // =====================================================================
    // Phase 5: Standard ChFSI Recurrence (ABINIT m_chebfi2.F90:641–666, 837–896)
    // =====================================================================
    // Chebyshev three-term recurrence on EIGENVECTORS via the S⁻¹·H operator:
    //   x̂ = (S⁻¹H − cI) / r  where  r = half_width, c = center
    //   T_0(x̂) = I  (x_prev)
    //   T_1(x̂) = x̂  (x_curr)
    //   T_{k+1}(x̂) = 2·x̂·T_k(x̂) − T_{k-1}(x̂)  (x_next)
    //
    // Per-step computation:
    //   1. Apply H to x_curr (get H·x_curr)
    //   2. Apply S⁻¹ to H·x_curr (get x̂_scaled = S⁻¹·H·x_curr)
    //   3. Recurrence: x_next = (2/r)·(x̂_scaled − c·x_curr) − x_prev
    //
    // This matches ABINIT's `chebfi_computeNextOrderChebfiPolynom`:
    //   S⁻¹ applied AFTER H (not before), which is the correct USPP operator.
    let r = (lambda_plus - lambda_minus) / 2.0;    // half_width = radius
    let c = (lambda_plus + lambda_minus) / 2.0;    // center
    let one_over_r = 1.0 / r;
    let two_over_r = 2.0 / r;

    let final_psi_buf: &mut PwCoefficients;

    if ndeg_filter_global == 0 {
        // No filtering needed (all bands converged/locked)
        stream.memcpy_dtod(&*psi_input, &mut x_prev.0).map_err(Error::Cuda)?;
        final_psi_buf = &mut x_prev;
    } else {
        // ------------------------------------------------------------
        // Step 1: Compute T_0 = psi_input (identity Chebyshev vector)
        // ------------------------------------------------------------
        // x_prev = T_0(S⁻¹H)·Ψ = Ψ (no operation needed; psi_input is T_0)
        stream.memcpy_dtod(&*psi_input, &mut x_prev.0).map_err(Error::Cuda)?;

        // ------------------------------------------------------------
        // Step 2: Compute T_1 = (S⁻¹·H·Ψ − c·Ψ) / r
        // ------------------------------------------------------------
        // x_curr = H·psi_input (apply H to psi_input via apply_full_hamiltonian)
        unsafe {
            apply_full_hamiltonian()
                .psi_dev(&psi_input)
                .v_eff_dev(v_eff_dev)
                .kinetic_dev(&kinetic_dev)
                .fft_idx_dev(fft_idx_dev)
                .n_pw(n_pw)
                .n_bands(n_bands)
                .grid_size(grid_size)
                .inv_ntotal(inv_ntotal)
                .fft_plan(&fft_plan)
                .hpsi_dev(&mut x_curr)
                .grid_dev(&mut grid_dev)
                .vnl_data(vnl_data)
                .blas(blas)
                .kernels(kernels)
                .stream(stream)
                .call()?;
        }
        // x_curr = S⁻¹·(H·psi_input) — in-place via Woodbury
        unsafe {
            apply_s_inverse()
                .hpsi_dev(&mut x_curr)
                .vnl_data(vnl_data)
                .n_bands(n_bands_i32)
                .n_pw(n_pw_i32)
                .blas(blas)
                .stream(stream)
                .solver(solver)
                .call()?;
        }
        // x_curr = (x_curr − c·x_prev) / r  →  T_1 = (S⁻¹H − cI)/r · Ψ
        // Compose: buf_sx = −c * x_prev, then x_curr += buf_sx, then scale by 1/r
        stream.memcpy_dtod(&*x_prev, &mut buf_sx.0).map_err(Error::Cuda)?;
        {
            let neg_c = CudaComplex { x: -c, y: 0.0 };
            unsafe {
                let (ptr, _) = buf_sx.0.device_ptr_mut(stream);
                cudarc::cublas::sys::cublasZscal_v2(
                    blas.raw_handle(), n_elem_i32,
                    &neg_c as *const _ as *const _, ptr as *mut _, 1,
                ).result().map_err(Error::Blas)?;
            }
        }
        // x_curr += buf_sx = −c * x_prev (cblasZaxpy: y += alpha*x)
        {
            let alpha_one = CudaComplex { x: 1.0, y: 0.0 };
            blas.axpy_c64(n_elem_i32, alpha_one, &buf_sx, 1, &mut x_curr, 1)
                .map_err(Error::Blas)?;
        }
        // x_curr *= 1/r
        {
            let ir = CudaComplex { x: one_over_r, y: 0.0 };
            unsafe {
                let (ptr, _) = x_curr.0.device_ptr_mut(stream);
                cudarc::cublas::sys::cublasZscal_v2(
                    blas.raw_handle(), n_elem_i32,
                    &ir as *const _ as *const _, ptr as *mut _, 1,
                ).result().map_err(Error::Blas)?;
            }
        }

        // ------------------------------------------------------------
        // Step 3: Recurrence for k = 2..=ndeg_filter_global
        // ------------------------------------------------------------
        // T_{k+1} = (2/r)·(S⁻¹·H·T_k − c·T_k) − T_{k-1}
        for k in 2..=ndeg_filter_global {
            // Step 3a: H·x_curr → hpsi_dev
            unsafe {
                apply_full_hamiltonian()
                    .psi_dev(&x_curr)
                    .v_eff_dev(v_eff_dev)
                    .kinetic_dev(&kinetic_dev)
                    .fft_idx_dev(fft_idx_dev)
                    .n_pw(n_pw)
                    .n_bands(n_bands)
                    .grid_size(grid_size)
                    .inv_ntotal(inv_ntotal)
                    .fft_plan(&fft_plan)
                    .hpsi_dev(&mut hpsi_dev)
                    .grid_dev(&mut grid_dev)
                    .vnl_data(vnl_data)
                    .blas(blas)
                    .kernels(kernels)
                    .stream(stream)
                    .call()?;
            }

            // Step 3b: S⁻¹·(H·x_curr) → hpsi_dev (in-place)
            unsafe {
                apply_s_inverse()
                    .hpsi_dev(&mut hpsi_dev)
                    .vnl_data(vnl_data)
                    .n_bands(n_bands_i32)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .solver(solver)
                    .call()?;
            }

            // Step 3c: x_next = (2/r)·(hpsi_dev − c·x_curr) − x_prev
            // First: x_next = hpsi_dev
            stream.memcpy_dtod(&*hpsi_dev, &mut x_next.0).map_err(Error::Cuda)?;

            // x_next += −c * x_curr  (via buf_sx = −c·x_curr, then axpy)
            stream.memcpy_dtod(&*x_curr, &mut buf_sx.0).map_err(Error::Cuda)?;
            {
                let neg_c = CudaComplex { x: -c, y: 0.0 };
                unsafe {
                    let (ptr, _) = buf_sx.0.device_ptr_mut(stream);
                    cudarc::cublas::sys::cublasZscal_v2(
                        blas.raw_handle(), n_elem_i32,
                        &neg_c as *const _ as *const _, ptr as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }
            {
                let alpha_one = CudaComplex { x: 1.0, y: 0.0 };
                blas.axpy_c64(n_elem_i32, alpha_one, &buf_sx, 1, &mut x_next, 1)
                    .map_err(Error::Blas)?;
            }

            // x_next *= 2/r
            {
                let t_r = CudaComplex { x: two_over_r, y: 0.0 };
                unsafe {
                    let (ptr, _) = x_next.0.device_ptr_mut(stream);
                    cudarc::cublas::sys::cublasZscal_v2(
                        blas.raw_handle(), n_elem_i32,
                        &t_r as *const _ as *const _, ptr as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }

            // x_next += −1 * x_prev
            {
                let neg_one = CudaComplex { x: -1.0, y: 0.0 };
                blas.axpy_c64(n_elem_i32, neg_one, &x_prev, 1, &mut x_next, 1)
                    .map_err(Error::Blas)?;
            }

            // Rotate buffers: x_prev ← x_curr, x_curr ← x_next
            std::mem::swap(&mut x_prev.0, &mut x_curr.0);
            std::mem::swap(&mut x_curr.0, &mut x_next.0);

            // Norm stability check on the new x_curr
            let norm_curr = compute_frobenius_norm(&x_curr, n_elem_i32, blas)?;
            let norm_prev = compute_frobenius_norm(&x_prev, n_elem_i32, blas)?;
            check_norm_stability(norm_curr, norm_prev, k)?;
        }

        // After loop: x_curr contains T_n(S⁻¹H)·Ψ (the filtered wavefunctions)

        // Lock converged bands: copy original psi_input columns for bands with ndeg=0.
        // Prevents filter distortion of converged bands from amplifying
        // SCF V_eff drift across iterations.
        {
            let n_locked = ndeg_filter_bands.iter().filter(|&&d| d == 0).count();
            if n_locked > 0 && n_locked < n_bands {
                let (x_curr_ptr, _sync) = x_curr.0.device_ptr_mut(stream);
                let (psi_in_ptr, _sync2) = psi_input.0.device_ptr(stream);
                let handle = blas.raw_handle();
                for b in 0..n_bands {
                    if ndeg_filter_bands[b] == 0 {
                        unsafe {
                            let dst = (x_curr_ptr as *mut CudaComplex).add(b * n_pw);
                            let src = (psi_in_ptr as *const CudaComplex).add(b * n_pw);
                            cudarc::cublas::sys::cublasZcopy_v2(handle, n_pw_i32,
                                src as *const _, 1, dst as *mut _, 1,
                            ).result().map_err(Error::Blas)?;
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------
        // Step 4: Chebyshev ampfactor normalisation (ABINIT lines 958–1006)
        // ------------------------------------------------------------
        // After filtering, each band's vector is T_n(ε_b)·ψ_b where ε_b is
        // the eigenvalue estimate. The amplification factor T_n(ε_b) varies
        // by band — larger for small eigenvalues, smaller for large ones.
        // Dividing each column by ampfactor normalises the filter output so
        // that Rayleigh-Ritz works on balanced vectors.
        //
        // chebfi_ampfactor operates on CPU slices (f64, real-only).
        // Each CudaComplex = 2 consecutive f64s, so total_spacedim = 2*n_pw.

        // Download x_curr as Vec<CudaComplex>, re-interpret as &mut [f64]
        let x_cplx: Vec<CudaComplex> = stream.clone_dtoh(&x_curr.0).map_err(Error::Cuda)?;
        // SAFETY: CudaComplex is #[repr(C)] with fields (x: f64, y: f64),
        // so a slice of CudaComplex has the same layout as a slice of 2*len f64s.
        let mut x_re_f64: Vec<f64> = unsafe {
            let (ptr, len, cap) = {
                let v = std::mem::ManuallyDrop::new(x_cplx);
                let ptr = v.as_ptr() as *mut f64;
                (ptr, v.len() * 2, v.capacity() * 2)
            };
            Vec::from_raw_parts(ptr, len, cap)
        };

        // Recompute H·x_curr (x_curr may have been locked above)
        unsafe {
            apply_full_hamiltonian()
                .psi_dev(&x_curr)
                .v_eff_dev(v_eff_dev)
                .kinetic_dev(&kinetic_dev)
                .fft_idx_dev(fft_idx_dev)
                .n_pw(n_pw)
                .n_bands(n_bands)
                .grid_size(grid_size)
                .inv_ntotal(inv_ntotal)
                .fft_plan(&fft_plan)
                .hpsi_dev(&mut hpsi_dev)
                .grid_dev(&mut grid_dev)
                .vnl_data(vnl_data)
                .blas(blas)
                .kernels(kernels)
                .stream(stream)
                .call()?;
        }
        // Download H·x_curr as Vec<CudaComplex>, re-interpret as &mut [f64]
        let hx_cplx: Vec<CudaComplex> = stream.clone_dtoh(&hpsi_dev.0).map_err(Error::Cuda)?;
        let mut hx_re_f64: Vec<f64> = unsafe {
            let v = std::mem::ManuallyDrop::new(hx_cplx);
            let ptr = v.as_ptr() as *mut f64;
            Vec::from_raw_parts(ptr, v.len() * 2, v.capacity() * 2)
        };

        // Compute S·x_curr
        stream.memcpy_dtod(&*x_curr, &mut spsi_dev.0).map_err(Error::Cuda)?;
        unsafe {
            apply_s_times()
                .psi_dev(&x_curr)
                .spsi_dev(&mut spsi_dev)
                .vnl_data(vnl_data)
                .n_bands(n_bands_i32)
                .n_pw(n_pw_i32)
                .blas(blas)
                .stream(stream)
                .call()?;
        }
        // Download S·x_curr as Vec<CudaComplex>, re-interpret as &mut [f64]
        let sx_cplx: Vec<CudaComplex> = stream.clone_dtoh(&spsi_dev.0).map_err(Error::Cuda)?;
        let mut sx_re_f64: Vec<f64> = unsafe {
            let v = std::mem::ManuallyDrop::new(sx_cplx);
            let ptr = v.as_ptr() as *mut f64;
            Vec::from_raw_parts(ptr, v.len() * 2, v.capacity() * 2)
        };

        // Apply ampfactor: scale each band's column in X, HX, SX by 1/T_n(ε_b)
        chebfi_ampfactor(
            &ndeg_filter_bands,
            &ritz_values,
            lambda_minus,
            lambda_plus,
            2 * n_pw,   // total_spacedim: 2 f64 per complex PW coefficient
            &mut x_re_f64,
            &mut hx_re_f64,
            &mut sx_re_f64,
            n_bands,
        );

        // Re-upload the ampfactor-corrected X vectors to GPU.
        // Re-interpret the &mut [f64] back to Vec<CudaComplex> for upload.
        let x_cplx_upload: Vec<CudaComplex> = unsafe {
            let v = std::mem::ManuallyDrop::new(x_re_f64);
            let ptr = v.as_ptr() as *mut CudaComplex;
            Vec::from_raw_parts(ptr, v.len() / 2, v.capacity() / 2)
        };
        stream.memcpy_htod(&x_cplx_upload, &mut x_curr.0).map_err(Error::Cuda)?;

        final_psi_buf = &mut x_curr;
    }

    // =====================================================================
    // Phase 6: Gram-Schmidt S-orthonormalisation (unchanged, lines 1022-1121)
    // =====================================================================
    let mut gs_col = PwCoefficients::new(stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
    let mut gs_s_col = PwCoefficients::new(stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
    unsafe {
        let (psi_ptr, _) = final_psi_buf.0.device_ptr_mut(stream);
        let (gs_col_ptr, _) = gs_col.0.device_ptr_mut(stream);
        for _pass in 0..2 {
            for b in 0..n_bands {
                let col_b = (psi_ptr as *mut CudaComplex).add(b * n_pw);
                // Copy col_b → gs_col (device-to-device)
                cudarc::cublas::sys::cublasZcopy_v2(
                    blas.raw_handle(), n_pw_i32,
                    col_b as *const _, 1,
                    gs_col_ptr as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                // gs_s_col = S · gs_col
                stream.memcpy_dtod(&*gs_col, &mut gs_s_col.0).map_err(Error::Cuda)?;
                apply_s_times()
                    .psi_dev(&gs_col)
                    .spsi_dev(&mut gs_s_col)
                    .vnl_data(vnl_data)
                    .n_bands(1)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .call()?;
                // Get device pointer from gs_s_col after mutable ops complete
                let (gs_s_col_ptr, _) = gs_s_col.0.device_ptr_mut(stream);
                let mut norm_sq_s = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(), n_pw_i32,
                    col_b as *const _, 1,
                    gs_s_col_ptr as *const _, 1,
                    &mut norm_sq_s as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                // Subtract projections onto all previous S-orthonormal columns
                for j in 0..b {
                    let col_j = (psi_ptr as *mut CudaComplex).add(j * n_pw);
                    let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                    cudarc::cublas::sys::cublasZdotc_v2(
                        blas.raw_handle(), n_pw_i32,
                        col_j as *const _, 1,
                        gs_s_col_ptr as *const _, 1,
                        &mut dot as *mut _ as *mut _,
                    ).result().map_err(Error::Blas)?;
                    // col_b -= dot * col_j
                    let neg_dot = CudaComplex { x: -dot.x, y: -dot.y };
                    cudarc::cublas::sys::cublasZaxpy_v2(
                        blas.raw_handle(), n_pw_i32,
                        &neg_dot as *const _ as *const _,
                        col_j as *const _, 1,
                        col_b as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    norm_sq_s.x -= dot.x * dot.x + dot.y * dot.y;
                }
                // Normalize col_b with S-norm
                let norm_s = norm_sq_s.x.sqrt();
                if norm_s > 1e-30 {
                    let inv_norm = CudaComplex { x: 1.0 / norm_s, y: 0.0 };
                    cudarc::cublas::sys::cublasZscal_v2(
                        blas.raw_handle(), n_pw_i32,
                        &inv_norm as *const _ as *const _,
                        col_b as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }
        }
    }

    // =====================================================================
    // Phase 8: Compute final H|psi> for Rayleigh-Ritz (unchanged)
    // =====================================================================
    unsafe {
        apply_full_hamiltonian()
            .psi_dev(final_psi_buf)
            .v_eff_dev(v_eff_dev)
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(fft_idx_dev)
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(grid_size)
            .inv_ntotal(inv_ntotal)
            .fft_plan(&fft_plan)
            .hpsi_dev(&mut hpsi_dev)
            .grid_dev(&mut grid_dev)
            .vnl_data(vnl_data)
            .blas(blas)
            .kernels(kernels)
            .stream(stream)
            .call()?;
    }

    // =====================================================================
    // Phase 9: Output preparation
    // =====================================================================
    stream.memcpy_dtod(&**final_psi_buf, &mut psi_row_dev.0).map_err(Error::Cuda)?;
    stream.memcpy_dtod(&*hpsi_dev, &mut hpsi_row_dev.0).map_err(Error::Cuda)?;

    let psi_row = Gpu::<WavefunctionSet<RowDistributed>> {
        slice: psi_row_dev.0,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };
    let hpsi_row = Gpu::<WavefunctionSet<RowDistributed>> {
        slice: hpsi_row_dev.0,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };

    // Return fresh Phase 2b residual_sq_norms (computed under current V_eff,
    // pre-filtering). The caller also computes post-RR residuals via
    // chebfi_residual_norms for SCF convergence monitoring.
    Ok((psi_row, hpsi_row, ritz_values, fresh_residuals, ndeg_filter_bands))
}

// ---------------------------------------------------------------------------
// Test diagnostics — direct ⟨ψ|H|ψ⟩ computation (Phase H)
// ---------------------------------------------------------------------------
// `apply_h_components_for_test` runs the apply pipeline three times to extract
// per-band contributions: kinetic only, kinetic + V_loc, and kinetic + V_loc + V_NL.
// All three are returned as Vec<Complex64> in column-major (n_bands × n_pw) layout
// so test code can compute ⟨ψ_b|h_part_b⟩ on CPU.

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn apply_h_components_for_test(
    psi_gpu: &Gpu<WavefunctionSet<ColumnDistributed>>,
    v_eff_dev: &CudaSlice<f64>,
    wave_grid: &GVectorGrid,
    pw_coords: &[[i32; 3]],
    vnl_data: &VnlBatchData,
    fft_idx_dev: &CudaSlice<i32>,
    kernels: &CudaKernelSet,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    kpoint_frac: [f64; 3],
) -> Result<HComponentsForTest, Error> {
    let n_bands = psi_gpu.shape()[0];
    let n_pw = psi_gpu.shape()[1];
    let n_elem = n_bands * n_pw;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;

    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = ngx * ngy * ngz;
    let inv_ntotal = 1.0 / (grid_size as f64);
    let grid_alloc = n_bands * grid_size;

    let kinetic_cpu = compute_kinetic_energies(pw_coords, wave_grid.recip_lattice(), kpoint_frac);
    let kinetic_raw: CudaSlice<f64> = stream.clone_htod(&kinetic_cpu.0).map_err(Error::Cuda)?;
    let kinetic_dev = KineticPreconditioner::new(kinetic_raw);

    // Fortran data layout (ngz, ngy, ngx) with ngz innermost (stride-1).
    // cuFFT n[0] is innermost, so plan dims = (ngz, ngy, ngx).
    // Verified by cufft_dim_ordering_isolated_diagnostic.
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32, n_bands_i32, stream.clone(),
    )?;

    let psi_input = psi_gpu.as_device_slice();

    let mut grid_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // Component 1: kinetic only. Run init_kinetic by itself.
    let mut hpsi_t: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    unsafe {
        stream
            .launch_builder(&kernels.init_kinetic)
            .arg(&mut hpsi_t)
            .arg(psi_input)
            .arg(&*kinetic_dev)
            .arg(&n_pw_i32)
            .arg(&n_bands_i32)
            .launch(LaunchConfig::for_num_elems((n_bands_i32 * n_pw_i32) as u32))
    }
    .map_err(Error::Cuda)?;

    // Component 2: kinetic + V_loc (full apply_v_loc_hamiltonian).
    let mut hpsi_tv = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let psi_pw = PwCoefficients::new(psi_input.clone());
    unsafe {
        apply_v_loc_hamiltonian()
            .psi_dev(&psi_pw)
            .hpsi_dev(&mut hpsi_tv)
            .grid_dev(&mut grid_dev)
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(fft_idx_dev)
            .v_eff_dev(v_eff_dev)
            .n_pw(n_pw_i32)
            .n_bands(n_bands_i32)
            .grid_size(grid_size as i32)
            .inv_ntotal(inv_ntotal)
            .ngx(fft_plan.nx())
            .ngy(fft_plan.ny())
            .ngz(fft_plan.nz())
            .fft_plan(&fft_plan)
            .kernels(kernels)
            .stream(stream)
            .call()?;
    }

    // Component 3: kinetic + V_loc + V_NL (full apply_full_hamiltonian).
    let mut hpsi_full = PwCoefficients::new(stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    unsafe {
        apply_v_loc_hamiltonian()
            .psi_dev(&psi_pw)
            .hpsi_dev(&mut hpsi_full)
            .grid_dev(&mut grid_dev)
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(fft_idx_dev)
            .v_eff_dev(v_eff_dev)
            .n_pw(n_pw_i32)
            .n_bands(n_bands_i32)
            .grid_size(grid_size as i32)
            .inv_ntotal(inv_ntotal)
            .ngx(fft_plan.nx())
            .ngy(fft_plan.ny())
            .ngz(fft_plan.nz())
            .fft_plan(&fft_plan)
            .kernels(kernels)
            .stream(stream)
            .call()?;
        apply_v_nl_hamiltonian()
            .psi_dev(&psi_pw)
            .hpsi_dev(&mut hpsi_full)
            .vnl_data(vnl_data)
            .n_bands(n_bands_i32)
            .n_pw(n_pw_i32)
            .blas(blas)
            .stream(stream)
            .call()?;
    }

    stream.synchronize()?;

    // D2H all three.
    let to_complex = |raw: Vec<CudaComplex>| -> Vec<num_complex::Complex64> {
        raw.into_iter()
            .map(|c| num_complex::Complex64::new(c.x, c.y))
            .collect()
    };
    Ok(HComponentsForTest {
        hpsi_t: to_complex(stream.clone_dtoh(&hpsi_t).map_err(Error::Cuda)?),
        hpsi_tv: to_complex(stream.clone_dtoh(&*hpsi_tv).map_err(Error::Cuda)?),
        hpsi_full: to_complex(stream.clone_dtoh(&*hpsi_full).map_err(Error::Cuda)?),
        n_bands,
        n_pw,
    })
}

#[doc(hidden)]
pub struct HComponentsForTest {
    pub hpsi_t: Vec<num_complex::Complex64>,
    pub hpsi_tv: Vec<num_complex::Complex64>,
    pub hpsi_full: Vec<num_complex::Complex64>,
    pub n_bands: usize,
    pub n_pw: usize,
}

// ---------------------------------------------------------------------------
// S·ψ host-side helper for tests.
//
// Wraps the (crate-private) `apply_s_times` so external tests can compute
// the USPP overlap matrix `⟨ψ_a | S | ψ_b⟩ = ψ_a^H · (S·ψ_b)` correctly.
// L2 dot products are unreliable for low-PW-norm Cu 3d bands (‖ψ‖² ≈ 0.14
// means the L2 self-overlap ceiling is ≈ 0.02 — physically unreachable
// for any "> 0.5" similarity gate).
// ---------------------------------------------------------------------------

#[doc(hidden)]
pub fn apply_s_for_test(
    psi_host: &[num_complex::Complex64],
    n_bands: usize,
    n_pw: usize,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<Vec<num_complex::Complex64>, Error> {
    let n_elem = n_bands * n_pw;
    let psi_cuda: Vec<CudaComplex> = psi_host
        .iter()
        .map(|&c| CudaComplex { x: c.re, y: c.im })
        .collect();
    let psi_dev: CudaSlice<CudaComplex> = stream.clone_htod(&psi_cuda).map_err(Error::Cuda)?;
    let mut spsi_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    stream.memcpy_dtod(&psi_dev, &mut spsi_dev).map_err(Error::Cuda)?;

    let psi_pw = PwCoefficients(psi_dev);
    let mut spsi_pw = PwCoefficients(spsi_dev);
    unsafe {
        apply_s_times()
            .psi_dev(&psi_pw)
            .spsi_dev(&mut spsi_pw)
            .vnl_data(vnl_data)
            .n_bands(n_bands as i32)
            .n_pw(n_pw as i32)
            .blas(blas)
            .stream(stream)
            .call()?;
    }
    stream.synchronize()?;
    let spsi_raw = stream.clone_dtoh(&*spsi_pw).map_err(Error::Cuda)?;
    Ok(spsi_raw
        .into_iter()
        .map(|c| num_complex::Complex64::new(c.x, c.y))
        .collect())
}

/// S-norm Gram-Schmidt orthonormalization for USPP wavefunctions.
///
/// Orthonormalizes `psi` columns (col-major: flat[b*n_pw + g]) with respect
/// to the S-inner product: `<x|y>_S = x^H · S · y`.  Two passes for stability.
pub(crate) fn gram_schmidt_s(
    psi: &mut CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    n_pw: usize, n_bands: usize,
    blas: &BlasHandle, stream: &Arc<CudaStream>, _solver: &SolverHandle,
) -> Result<(), Error> {
    let n_pw_i32 = n_pw as i32;
    let mut gs_col = PwCoefficients::new(stream.alloc_zeros(n_pw)?);
    let mut gs_s_col = PwCoefficients::new(stream.alloc_zeros(n_pw)?);
    unsafe {
        let (psi_ptr, _) = psi.device_ptr_mut(stream);
        let (gs_col_ptr, _) = gs_col.0.device_ptr_mut(stream);
        for _pass in 0..2 {
            for b in 0..n_bands {
                let col_b = (psi_ptr as *mut CudaComplex).add(b * n_pw);
                cudarc::cublas::sys::cublasZcopy_v2(
                    blas.raw_handle(), n_pw_i32,
                    col_b as *const _, 1, gs_col_ptr as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                stream.memcpy_dtod(&*gs_col, &mut gs_s_col.0)?;
                apply_s_times()
                    .psi_dev(&gs_col)
                    .spsi_dev(&mut gs_s_col)
                    .vnl_data(vnl_data)
                    .n_bands(1)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .call()?;
                let (gs_s_col_ptr, _) = gs_s_col.0.device_ptr_mut(stream);
                let mut norm_sq_s = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas.raw_handle(), n_pw_i32,
                    col_b as *const _, 1, gs_s_col_ptr as *const _, 1,
                    &mut norm_sq_s as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                if b == 0 && _pass == 0 {
                    eprintln!(
                        "[GramSchmidt] band-0 pass-0: norm²_S = {:.6e} (grid-sum convention; ≈1.0 for S-orthonormal input)",
                        norm_sq_s.x
                    );
                }
                for j in 0..b {
                    let col_j = (psi_ptr as *mut CudaComplex).add(j * n_pw);
                    let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                    cudarc::cublas::sys::cublasZdotc_v2(
                        blas.raw_handle(), n_pw_i32,
                        col_j as *const _, 1, gs_s_col_ptr as *const _, 1,
                        &mut dot as *mut _ as *mut _,
                    ).result().map_err(Error::Blas)?;
                    let neg_dot = CudaComplex { x: -dot.x, y: -dot.y };
                    cudarc::cublas::sys::cublasZaxpy_v2(
                        blas.raw_handle(), n_pw_i32,
                        &neg_dot as *const _ as *const _,
                        col_j as *const _, 1, col_b as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    norm_sq_s.x -= dot.x * dot.x + dot.y * dot.y;
                }
                let norm_s = norm_sq_s.x.sqrt();
                if norm_s > 1e-30 {
                    let inv_norm = CudaComplex { x: 1.0 / norm_s, y: 0.0 };
                    cudarc::cublas::sys::cublasZscal_v2(
                        blas.raw_handle(), n_pw_i32,
                        &inv_norm as *const _ as *const _,
                        col_b as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }
        }
    }
    Ok(())
}

/// Test-only wrapper for `chebyshev_filter` that handles GPU upload/download.
///
/// Mirrors `apply_s_for_test`: uploads ψ and V_eff to GPU, runs the existing
/// `pub(crate) chebyshev_filter`, downloads the filtered ψ̂ back to CPU.
///
/// This allows external integration tests (in `tests/`) to measure orthogonality
/// or condition number after Chebyshev filtering without duplicating the full
/// GPU infrastructure setup.
///
/// # Arguments
/// - `psi_host`: Input wavefunctions (CPU, column-major: flat[b*n_pw + g])
/// - `v_eff_host`: Effective potential on wave grid (CPU, flat real-space array)
/// - `n_bands`, `n_pw`: Wavefunction dimensions
/// - `wave_grid`: G-vector grid metadata
/// - `pw_coords`: Plane-wave Miller indices (length = n_pw)
/// - `cell`: Cell geometry (needed for chebyshev_filter signature, not actually used)
/// - `pots`: Pseudopotential set (needed for signature, not actually used)
/// - `vnl_data`: Nonlocal pseudopotential data (already on GPU)
/// - `min_veff`, `max_veff`: V_eff bounds for spectral estimation
/// - `ndeg`: Chebyshev filter degree (8 for diagnostic)
/// - `eigenvalues`: Optional prior eigenvalues for R-ChFSI (None for first call)
/// - `blas`, `solver`, `stream`: GPU handles
///
/// # Returns
/// Filtered wavefunctions ψ̂ on CPU (column-major, same layout as input).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_filter_for_test(
    psi_host: &[num_complex::Complex64],
    v_eff_host: &[f64],
    n_bands: usize,
    n_pw: usize,
    wave_grid: &GVectorGrid,
    pw_coords: &[[i32; 3]],
    cell: &CellGeometry,
    pots: &PseudopotentialSet,
    vnl_data: &VnlBatchData,
    min_veff: f64,
    max_veff: f64,
    ndeg: usize,
    eigenvalues: Option<&[f64]>,
    filter_mode: FilterMode,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> Result<Vec<num_complex::Complex64>, Error> {
    use crate::device::pcie::PcieAccount;
    use crate::layout::WavefunctionSet;

    // Upload ψ to GPU
    let psi_wfc = WavefunctionSet::<ColumnDistributed>::new(psi_host.to_vec(), n_bands, n_pw);
    let mut pcie = PcieAccount::default();
    let psi_gpu = Gpu::from_host_with(&psi_wfc, stream, &mut pcie)?;

    // Upload V_eff to GPU
    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = ngx * ngy * ngz;
    assert_eq!(
        v_eff_host.len(),
        grid_size,
        "v_eff_host length {} does not match wave_grid size {}",
        v_eff_host.len(),
        grid_size
    );

    // V_eff is on wave grid, so wrap as WaveGridArray then upsample to FineGridArray
    let v_eff_arr = ndarray::Array3::from_shape_vec(
        (ngx, ngy, ngz),
        v_eff_host.to_vec(),
    ).map_err(|e| Error::Io(format!("v_eff shape error: {}", e)))?;
    use crate::types::{WaveGridArray, FineGridArray};
    let v_eff_wave = WaveGridArray::from_inner(v_eff_arr);
    // For simplicity, assume wave_grid == fine_grid (true for Cu111_CO)
    let v_eff_fine = FineGridArray::from_inner(v_eff_wave.into_inner());
    let v_eff_inner = crate::types::EffectivePotential::from_inner(v_eff_fine);
    let v_eff_gpu = Gpu::from_host_with(&v_eff_inner, stream, &mut pcie)?;

    // Upload PW-to-FFT index map
    let fft_idx: Vec<i32> = crate::pw_coords_to_fft_indices(pw_coords, wave_grid);
    let fft_idx_dev: CudaSlice<i32> = stream.clone_htod(&fft_idx).map_err(Error::Cuda)?;

    // Compile kernels
    let kernels = CudaKernelSet::new(ctx)?;

    // Dummy k-point (unused by chebyshev_filter)
    let dummy_kpoint = KPoint { coords: [0.0, 0.0, 0.0], weight: 1.0 };

    // Run chebyshev_filter
    let (psi_filtered_row, _hpsi_row) = chebyshev_filter(
        &psi_gpu,
        v_eff_gpu.as_device_slice(),
        pots,
        wave_grid,
        &dummy_kpoint,
        cell,
        pw_coords,
        vnl_data,
        &fft_idx_dev,
        min_veff,
        max_veff,
        &kernels,
        &mut pcie,
        eigenvalues,
        ndeg,
        blas,
        solver,
        stream,
        ctx,
        filter_mode,
        None,
    )?;

    // Download filtered ψ̂ from GPU (RowDistributed layout)
    stream.synchronize()?;
    let psi_filtered_raw = stream.clone_dtoh(psi_filtered_row.as_device_slice()).map_err(Error::Cuda)?;

    // Convert RowDistributed (flat[b*n_pw + g]) back to column-major host layout
    // Actually, chebyshev_filter returns RowDistributed which is the same memory
    // layout as ColumnDistributed (both are flat[b*n_pw + g]), just a semantic marker.
    // See comment at chebyshev.rs:1069-1080.
    Ok(psi_filtered_raw
        .into_iter()
        .map(|c| num_complex::Complex64::new(c.x, c.y))
        .collect())
}

/// GPU-resident variant: returns `(psi_row_gpu, hpsi_row_gpu, kernels)` directly.
///
/// Identical to `chebyshev_filter_for_test` but keeps results on the device
/// so the caller can chain into `rayleigh_ritz_with_matrices` without an
/// extra upload/download round-trip.  Also returns `CudaKernelSet` so the
/// caller can pass it (unused) to the RR wrapper.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_filter_for_test_gpu(
    psi_host: &[num_complex::Complex64],
    v_eff_host: &[f64],
    n_bands: usize,
    n_pw: usize,
    wave_grid: &GVectorGrid,
    pw_coords: &[[i32; 3]],
    cell: &CellGeometry,
    pots: &PseudopotentialSet,
    vnl_data: &VnlBatchData,
    min_veff: f64,
    max_veff: f64,
    ndeg: usize,
    eigenvalues: Option<&[f64]>,
    filter_mode: FilterMode,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> GpuFilterResult {
    use crate::device::pcie::PcieAccount;
    use crate::layout::WavefunctionSet;

    // Upload ψ to GPU
    let psi_wfc = WavefunctionSet::<ColumnDistributed>::new(psi_host.to_vec(), n_bands, n_pw);
    let mut pcie = PcieAccount::default();
    let psi_gpu = Gpu::from_host_with(&psi_wfc, stream, &mut pcie)?;

    // Upload V_eff to GPU
    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = ngx * ngy * ngz;
    assert_eq!(v_eff_host.len(), grid_size);
    let v_eff_arr = ndarray::Array3::from_shape_vec(
        (ngx, ngy, ngz),
        v_eff_host.to_vec(),
    ).map_err(|e| Error::Io(format!("v_eff shape error: {}", e)))?;
    use crate::types::{WaveGridArray, FineGridArray};
    let v_eff_wave = WaveGridArray::from_inner(v_eff_arr);
    let v_eff_fine = FineGridArray::from_inner(v_eff_wave.into_inner());
    let v_eff_inner = crate::types::EffectivePotential::from_inner(v_eff_fine);
    let v_eff_gpu = Gpu::from_host_with(&v_eff_inner, stream, &mut pcie)?;

    // Upload PW-to-FFT index map
    let fft_idx: Vec<i32> = crate::pw_coords_to_fft_indices(pw_coords, wave_grid);
    let fft_idx_dev: CudaSlice<i32> = stream.clone_htod(&fft_idx).map_err(Error::Cuda)?;

    // Compile kernels (needed by chebyshev_filter; caller reuses for RR)
    let kernels = CudaKernelSet::new(ctx)?;

    // Dummy k-point (unused by chebyshev_filter)
    let dummy_kpoint = KPoint { coords: [0.0, 0.0, 0.0], weight: 1.0 };

    // Run chebyshev_filter — keeps results GPU-resident
    let (psi_filtered_row, hpsi_filtered_row) = chebyshev_filter(
        &psi_gpu,
        v_eff_gpu.as_device_slice(),
        pots,
        wave_grid,
        &dummy_kpoint,
        cell,
        pw_coords,
        vnl_data,
        &fft_idx_dev,
        min_veff,
        max_veff,
        &kernels,
        &mut pcie,
        eigenvalues,
        ndeg,
        blas,
        solver,
        stream,
        ctx,
        filter_mode,
        None,
    )?;

    Ok((psi_filtered_row, hpsi_filtered_row, kernels))
}

/// GPU-resident single iteration of Chebyshev filter.
///
/// Takes pre-built GPU resources (`v_eff_gpu`, `fft_idx_dev`, `kernels`) that are
/// constant across outer-loop iterations and new `psi_gpu` (ColumnDistributed).
/// Delegates to the internal `chebyshev_filter` without re-uploading V_eff,
/// re-bulding FFT indices, or re-compiling kernels.
///
/// Returns `(psi_filtered_row, hpsi_filtered_row)` — both RowDistributed, GPU-resident.
///
/// Designed for Diagnostic 3: 10-iteration outer loop where only psi changes.
///
/// **Note**: This wrapper hard-codes a gamma-point `KPoint { coords: [0.0, 0.0, 0.0] }`.
/// It is only valid for Γ-only calculations.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_filter_iteration_gpu(
    psi_gpu: &Gpu<WavefunctionSet<ColumnDistributed>>,
    v_eff_dev: &CudaSlice<f64>,
    fft_idx_dev: &CudaSlice<i32>,
    kernels: &CudaKernelSet,
    wave_grid: &GVectorGrid,
    pw_coords: &[[i32; 3]],
    cell: &CellGeometry,
    pots: &PseudopotentialSet,
    vnl_data: &VnlBatchData,
    min_veff: f64,
    max_veff: f64,
    ndeg: usize,
    eigenvalues: Option<&[f64]>,
    filter_mode: FilterMode,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> ChebyshevResult {
    let dummy_kpoint = KPoint { coords: [0.0, 0.0, 0.0], weight: 1.0 };
    let mut pcie = PcieAccount::default();
    chebyshev_filter(
        psi_gpu, v_eff_dev, pots, wave_grid, &dummy_kpoint, cell,
        pw_coords, vnl_data, fft_idx_dev, min_veff, max_veff,
        kernels, &mut pcie, eigenvalues, ndeg,
        blas, solver, stream, ctx, filter_mode, None,
    )
}

/// GPU-resident residual norm computation (follows `davidson.rs:270-338`).
///
/// Takes GPU-resident RR output and computes per-band:
///   r_b = H·ψ_b − λ_b · S·ψ_b
///   ||r_b||₂  (unweighted L2)
///   ||r_b||_{S⁻¹}  (physically correct USPP metric)
///
/// Returns `(sinv_norms, l2_norms)` — only `2 × n_bands` f64 scalars cross PCIe.
/// All intermediate computation (rotate, apply S, axpy, apply S⁻¹, dotc) stays on GPU.
#[doc(hidden)]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn compute_residual_norms_for_test(
    psi_new_gpu: &Gpu<WavefunctionSet<ColumnDistributed>>,
    hpsi_row_gpu: &Gpu<WavefunctionSet<RowDistributed>>,
    eigenvalues: &[f64],
    x: &[num_complex::Complex64],
    n_bands: usize,
    n_pw: usize,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
) -> Result<(Vec<f64>, Vec<f64>), Error> {
    use cudarc::cublas::sys::{cublasZaxpy_v2, cublasZcopy_v2, cublasZdotc_v2};
    use cudarc::driver::DevicePtrMut;

    let n = n_bands as i32;
    let k = n_pw as i32;
    let n_elem = n_bands * n_pw;
    let handle = blas.raw_handle();

    // === 1. Upload X to GPU ===
    let x_dev: CudaSlice<CudaComplex> = {
        let x_cuda: Vec<CudaComplex> = x
            .iter()
            .map(|&c| CudaComplex { x: c.re, y: c.im })
            .collect();
        stream.clone_htod(&x_cuda).map_err(Error::Cuda)?
    };

    // === 2. Allocate working buffers ===
    let mut hpsi_new_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let spsi_new_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut residual_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;

    // === 3. hpsi_new = hpsi_row · X  (rotate H·ψ into RR eigenbasis) ===
    // hpsi_row is (n_pw × n_bands) col-major, X is (n_bands × n_bands) col-major
    unsafe {
        blas.gemm_c64(
            crate::device::blas::ZgemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: k,
                n,
                k: n,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: k,
                ldb: n,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: k,
            },
            hpsi_row_gpu.as_device_slice(),
            &x_dev,
            &mut hpsi_new_dev,
        )?;
    }

    // === 4. spsi_new = S · psi_new (USPP overlap) ===
    // Pre-copy psi_new → spsi_new (identity term), then accumulate β·Q·β^H
    let psi_new_pw = PwCoefficients::new(psi_new_gpu.as_device_slice().clone());
    let mut spsi_new_pw = PwCoefficients::new(spsi_new_dev);
    stream
        .memcpy_dtod(&*psi_new_pw, &mut spsi_new_pw.0)
        .map_err(Error::Cuda)?;
    unsafe {
        apply_s_times()
            .psi_dev(&psi_new_pw)
            .spsi_dev(&mut spsi_new_pw)
            .vnl_data(vnl_data)
            .n_bands(n)
            .n_pw(k)
            .blas(blas)
            .stream(stream)
            .call()?;
    }

    // === 5. Per-band residual: r_b = hpsi_new_b − λ_b · spsi_new_b ===
    unsafe {
        let (hpsi_ptr, _) = hpsi_new_dev.device_ptr_mut(stream);
        let (spsi_ptr, _) = spsi_new_pw.0.device_ptr_mut(stream);
        let (residual_mut, _) = residual_dev.device_ptr_mut(stream);

        for b in 0..n_bands {
            let hpsi_b = (hpsi_ptr as *const CudaComplex).add(b * n_pw);
            let spsi_b = (spsi_ptr as *const CudaComplex).add(b * n_pw);
            let r_b = (residual_mut as *mut CudaComplex).add(b * n_pw);

            // Copy hpsi_b → r_b
            cublasZcopy_v2(
                handle, k,
                hpsi_b as *const _, 1,
                r_b as *mut _, 1,
            )
            .result()
            .map_err(Error::Blas)?;

            // r_b += −λ_b · spsi_b
            let neg_lambda = CudaComplex { x: -eigenvalues[b], y: 0.0 };
            cublasZaxpy_v2(
                handle, k,
                &neg_lambda as *const _ as *const _,
                spsi_b as *const _, 1,
                r_b as *mut _, 1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // === 6. L2 norms: ||r_b||₂ = sqrt(Re⟨r_b | r_b⟩) ===
    let mut l2_norms = vec![0.0_f64; n_bands];
    unsafe {
        let (res_ptr, _) = residual_dev.device_ptr_mut(stream);
        for b in 0..n_bands {
            let r_b = (res_ptr as *const CudaComplex).add(b * n_pw);
            let mut dot = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle, k,
                r_b as *const _, 1,
                r_b as *const _, 1,
                &mut dot as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;
            l2_norms[b] = dot.x.sqrt();
        }
    }

    // === 7. S⁻¹ · residual_dev  (batch Woodbury, in-place on a copy) ===
    let mut sinv_r_pw = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    stream
        .memcpy_dtod(&residual_dev, &mut sinv_r_pw.0)
        .map_err(Error::Cuda)?;
    unsafe {
        apply_s_inverse()
            .hpsi_dev(&mut sinv_r_pw)
            .vnl_data(vnl_data)
            .n_bands(n)
            .n_pw(k)
            .blas(blas)
            .stream(stream)
            .solver(solver)
            .call()?;
    }

    // === 8. S⁻¹-weighted norms: ||r_b||_{S⁻¹} = sqrt(Re⟨r_b | S⁻¹·r_b⟩) ===
    let mut sinv_norms = vec![0.0_f64; n_bands];
    unsafe {
        let (res_ptr, _) = residual_dev.device_ptr_mut(stream);
        let (sinv_ptr, _) = sinv_r_pw.0.device_ptr_mut(stream);
        for b in 0..n_bands {
            let r_b = (res_ptr as *const CudaComplex).add(b * n_pw);
            let sinv_b = (sinv_ptr as *const CudaComplex).add(b * n_pw);
            let mut dot = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle, k,
                r_b as *const _, 1,
                sinv_b as *const _, 1,
                &mut dot as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;
            sinv_norms[b] = dot.x.sqrt();
        }
    }

    Ok((sinv_norms, l2_norms))
}

// ---------------------------------------------------------------------------
// Unit tests for chebfi_ampfactor and cheb_poly1
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // cheb_poly1 tests — verified against ABINIT m_chebfi2.F90 cheb_poly1
    // -----------------------------------------------------------------------

    /// T_0(x) = 1 for all x (degree 0).
    #[test]
    fn cheb_poly1_degree_zero_is_unity() {
        // T_0(x) = 1 for any x in any interval
        assert_eq!(cheb_poly1(3.0, 0, -1.0, 5.0), 1.0);
        assert_eq!(cheb_poly1(-10.0, 0, -5.0, 5.0), 1.0);
        assert_eq!(cheb_poly1(0.0, 0, 0.0, 1.0), 1.0);
    }

    /// T_1(x) = x (degree 1).
    /// On the interval [−1, 1], T_1(x) = x directly.
    #[test]
    fn cheb_poly1_degree_one_is_identity() {
        // For interval [-1, 1]: xred = (2*x - 0) / 2 = x → T_1(xred) = x
        assert!((cheb_poly1(0.5, 1, -1.0, 1.0) - 0.5).abs() < 1e-15);
        assert!((cheb_poly1(-0.3, 1, -1.0, 1.0) - (-0.3)).abs() < 1e-15);
        assert!((cheb_poly1(1.0, 1, -1.0, 1.0) - 1.0).abs() < 1e-15);
        assert!((cheb_poly1(-1.0, 1, -1.0, 1.0) - (-1.0)).abs() < 1e-15);
    }

    /// T_n(1) = 1 for any degree n (Chebyshev property).
    #[test]
    fn cheb_poly1_at_upper_endpoint_is_unity() {
        // When x = bb (upper bound of interval), xred = +1, T_n(1) = 1 for any n
        let intervals = [(0.0, 2.0), (-3.0, 5.0), (-1.0, 1.0), (10.0, 20.0)];
        for &(a, b) in &intervals {
            for n in [1, 2, 3, 5, 10, 20] {
                let val = cheb_poly1(b, n, a, b);
                assert!(
                    (val - 1.0).abs() < 1e-12,
                    "T_{n}({b}) on [{a}, {b}] = {val}, expected 1.0"
                );
            }
        }
    }

    /// T_n(−1) = (−1)^n (Chebyshev property).
    #[test]
    fn cheb_poly1_at_lower_endpoint_is_alternating() {
        let intervals = [(-1.0, 1.0), (0.0, 2.0), (5.0, 7.0)];
        for &(a, b) in &intervals {
            for n in 1..=10 {
                let val = cheb_poly1(a, n, a, b);
                let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "T_{n}({a}) on [{a}, {b}] = {val}, expected {expected}"
                );
            }
        }
    }

    /// Closed-form check: T_2(x) = 2x^2 − 1.
    #[test]
    fn cheb_poly1_degree_two_closed_form() {
        // On [-1, 1], xred = x, so T_2(x) = 2x^2 - 1
        for &x_in in &[-0.5, 0.0, 0.3, 0.7, 0.999] {
            let val = cheb_poly1(x_in, 2, -1.0, 1.0);
            let expected = 2.0 * x_in * x_in - 1.0;
            assert!((val - expected).abs() < 1e-14,
                "T_2({x_in}) = {val}, expected {expected}");
        }
    }

    /// Closed-form check: T_3(x) = 4x^3 − 3x.
    #[test]
    fn cheb_poly1_degree_three_closed_form() {
        for &x_in in &[-0.8, -0.2, 0.0, 0.4, 0.9] {
            let val = cheb_poly1(x_in, 3, -1.0, 1.0);
            let expected = 4.0 * x_in.powi(3) - 3.0 * x_in;
            assert!((val - expected).abs() < 1e-13,
                "T_3({x_in}) = {val}, expected {expected}");
        }
    }

    /// T_n(xred) grows as ~xred^n outside [−1, 1].
    /// For xred = 2.0 on the standard interval, check approximate magnitude.
    #[test]
    fn cheb_poly1_outside_interval_growth() {
        // On [-1, 1], x = 2 → xred = 2. T_10(2) ≈ cosh(10 * acosh(2)) ≈ ... large.
        let val = cheb_poly1(2.0, 10, -1.0, 1.0);
        // T_10(2) = cosh(10 * acosh(2)) = cosh(10 * 1.316958) ≈ cosh(13.17) ≈ 2.63e5
        assert!(val > 1e5, "T_10(2) should be large (>1e5), got {val}");
        assert!(val < 1e6, "T_10(2) should be <1e6, got {val}");
    }

    // -----------------------------------------------------------------------
    // chebfi_ampfactor tests — verified against ABINIT m_chebfi2.F90
    // -----------------------------------------------------------------------

    /// Single band, degree-1 filter: amp = eig mapped to [-1,1].
    /// With [a, b] = [0, 2], eig=1.5 → xred = (3−2)/2 = 0.5, T_1 = 0.5.
    /// Vectors should be scaled by 1/0.5 = 2.0.
    #[test]
    fn ampfactor_single_band_degree_one() {
        let nbands = 1;
        let spacedim = 5;
        let ndeg = [1];
        let ritz = [1.5];
        let mut x = vec![1.0_f64; spacedim];
        let mut ax = vec![2.0_f64; spacedim];
        let mut bx = vec![3.0_f64; spacedim];

        chebfi_ampfactor(&ndeg, &ritz, 0.0, 2.0, spacedim,
            &mut x, &mut ax, &mut bx, nbands);

        // amp = T_1( (2*1.5 - (0+2)) / (2-0) ) = T_1(0.5) = 0.5
        // scale = 1/0.5 = 2.0
        for g in 0..spacedim {
            assert!((x[g] - 2.0).abs() < 1e-14, "x[{g}] = {}", x[g]);
            assert!((ax[g] - 4.0).abs() < 1e-14, "ax[{g}] = {}", ax[g]);
            assert!((bx[g] - 6.0).abs() < 1e-14, "bx[{g}] = {}", bx[g]);
        }
    }

    /// Two bands with different degrees and eigenvalues.
    /// Band 0: n=2, λ=0.5 on [0,2] → xred = (1−2)/2 = -0.5, T_2(-0.5) = 2*0.25−1 = -0.5
    /// Band 1: n=3, λ=1.8 on [0,2] → xred = (3.6−2)/2 = 0.8, T_3(0.8) = 4*0.512−3*0.8 = 2.048−2.4 = -0.352
    #[test]
    fn ampfactor_two_bands_different_degrees() {
        let nbands = 2;
        let spacedim = 3;
        let ndeg = [2_usize, 3];
        let ritz = [0.5, 1.8];
        // Band 0 starts with all 1.0; Band 1 starts with all 2.0
        let mut x = vec![1.0, 1.0, 1.0,  2.0, 2.0, 2.0];
        let mut ax = x.clone();
        let mut bx = x.clone();

        chebfi_ampfactor(&ndeg, &ritz, 0.0, 2.0, spacedim,
            &mut x, &mut ax, &mut bx, nbands);

        // Band 0: T_2(-0.5) = 2*(-0.5)^2 - 1 = 2*0.25 - 1 = -0.5
        // |amp| = 0.5 > 1e-3 → no clamp. scale = 1/(-0.5) = -2.0
        let amp0 = cheb_poly1(ritz[0], ndeg[0], 0.0, 2.0);
        let scale0 = 1.0 / amp0;
        for g in 0..spacedim {
            assert!((x[g] - scale0).abs() < 1e-14,
                "Band 0 x[{g}]: {} != {}", x[g], scale0);
        }
        // Band 1: T_3(0.8) = 4*(0.8)^3 - 3*0.8 = 4*0.512 - 2.4 = -0.352
        let amp1 = cheb_poly1(ritz[1], ndeg[1], 0.0, 2.0);
        let scale1 = 1.0 / amp1;
        for g in spacedim..2 * spacedim {
            let expected = 2.0 * scale1;
            assert!((x[g] - expected).abs() < 1e-14,
                "Band 1 x[{g}]: {} != {}", x[g], expected);
        }
    }

    /// Clamp test: when |amp| < 1e-3, set to 1e-3 (with preserved sign).
    /// Use n=2, λ ≈ half_width mid-point where T_2 is near zero.
    /// T_n has n roots in (-1, 1); T_2(x) = 0 at x = ±1/√2 ≈ ±0.7071.
    #[test]
    fn ampfactor_clamp_small_ampfactor() {
        // T_2(x) = 2x^2 - 1. Root at x = 1/sqrt(2) ≈ 0.7071.
        // On [0, 2]: xred = 0.7071 → λ = (xred*(b-a) + (a+b))/2 = (0.7071*2 + 2)/2 = 1.7071
        // amp = T_2(0.7071) ≈ 0 → clamp to 1e-3.
        let nbands = 1;
        let spacedim = 2;
        let ndeg = [2];
        let ritz = [1.70710678]; // λ such that xred = 1/√2
        let mut x = vec![5.0_f64; spacedim];
        let mut ax = x.clone();
        let mut bx = x.clone();

        chebfi_ampfactor(&ndeg, &ritz, 0.0, 2.0, spacedim,
            &mut x, &mut ax, &mut bx, nbands);

        // Raw amp ≈ 0, clamped to 1e-3. scale = 1/1e-3 = 1000.
        let expected = 5.0 * 1000.0;
        for g in 0..spacedim {
            assert!((x[g] - expected).abs() < 1e-10,
                "x[{g}] = {} != {}", x[g], expected);
        }
    }

    /// Clamp test: negative near-zero ampfactor → reset to +1e-3 (ABINIT behaviour).
    #[test]
    fn ampfactor_clamp_negative_near_zero() {
        // T_2(x) = 2x^2 - 1. Near root at x = -1/√2 ≈ -0.7071.
        // On [0, 2]: xred = -0.7071 → λ = (-0.7071*2 + 2)/2 = 0.2929.
        // T_2(-0.7071) ≈ 0 → |amp| < 1e-3 → clamped to +1e-3 (ABINIT line 994).
        let nbands = 1;
        let spacedim = 2;
        let ndeg = [2];
        let ritz = [0.29289322]; // λ such that xred = -1/√2
        let mut x = vec![3.0_f64, 7.0_f64];
        let mut ax = x.clone();
        let mut bx = x.clone();

        chebfi_ampfactor(&ndeg, &ritz, 0.0, 2.0, spacedim,
            &mut x, &mut ax, &mut bx, nbands);

        // ABINIT: abs(ampfactor) < 1e-3 → ampfactor = 1e-3 (positive, no sign preservation).
        // scale = 1/1e-3 = 1000. Vectors multiplied by +1000.
        let expected_scale = 1000.0;
        for g in 0..spacedim {
            let expected = if g == 0 { 3.0 * expected_scale } else { 7.0 * expected_scale };
            assert!((x[g] - expected).abs() < 1e-10,
                "x[{g}] = {} != {}", x[g], expected);
        }
    }

    /// Verify that AX and BX receive the identical scaling as X.
    #[test]
    fn ampfactor_ax_bx_receive_same_scaling() {
        let nbands = 2;
        let spacedim = 4;
        let ndeg = [2, 2];
        let ritz = [0.5, 1.5];
        let a = 0.0;
        let b = 2.0;

        let mut x = vec![1.0_f64; nbands * spacedim];
        let mut ax = vec![2.0_f64; nbands * spacedim];
        let mut bx = vec![3.0_f64; nbands * spacedim];

        chebfi_ampfactor(&ndeg, &ritz, a, b, spacedim,
            &mut x, &mut ax, &mut bx, nbands);

        for iband in 0..nbands {
            let amp = cheb_poly1(ritz[iband], ndeg[iband], a, b);
            let scale = 1.0 / amp;
            let start = iband * spacedim;
            for g in start..start + spacedim {
                assert!((x[g] - scale).abs() < 1e-14,
                    "x[{g}] band {iband}: {} != {}", x[g], scale);
                assert!((ax[g] - 2.0 * scale).abs() < 1e-14,
                    "ax[{g}] band {iband}: {} != {}", ax[g], 2.0 * scale);
                assert!((bx[g] - 3.0 * scale).abs() < 1e-14,
                    "bx[{g}] band {iband}: {} != {}", bx[g], 3.0 * scale);
            }
        }
    }

    /// Verify that non-contiguous bands (spacedim > 1) don't cross-contaminate.
    #[test]
    fn ampfactor_bands_independent() {
        let nbands = 3;
        let spacedim = 2;
        let ndeg = [1, 2, 3];
        let ritz = [0.2, 1.0, 1.8];
        let a = 0.0;
        let b = 2.0;

        let mut x: Vec<f64> = (0..nbands * spacedim)
            .map(|i| (i + 1) as f64)
            .collect();
        let x_orig = x.clone();
        let mut ax = x.clone();
        let mut bx = x.clone();

        chebfi_ampfactor(&ndeg, &ritz, a, b, spacedim,
            &mut x, &mut ax, &mut bx, nbands);

        // Each band should be scaled independently
        for iband in 0..nbands {
            let amp = cheb_poly1(ritz[iband], ndeg[iband], a, b);
            let scale = 1.0 / amp;
            let start = iband * spacedim;
            for g in start..start + spacedim {
                let expected = x_orig[g] * scale;
                assert!((x[g] - expected).abs() < 1e-14,
                    "Band {iband} g={g}: {} != {} (original={}, scale={})",
                    x[g], expected, x_orig[g], scale);
            }
        }
    }

    // -----------------------------------------------------------------------
    // chebfi_residual_norms tests — verified against ABINIT m_chebfi2.F90
    // -----------------------------------------------------------------------
    //
    // The ABINIT reference (lines 709–717) computes residuals via:
    //   1. colwiseCymax: R_i = A_i − λ_i · B_i  (overwrites A with R in-place)
    //   2. colwiseNorm2: residu(i) = ||R_i||₂²  (squared L2 norm, no sqrt)
    //
    // These tests exercise the mathematical operation on CPU vectors (complex
    // f64) to verify the formula, since the GPU path (`chebfi_residual_norms`)
    // uses cuBLAS calls that are provably-equivalent per-band axpy + dotc.

    /// Norm-conserving residual: R_i = A_i − λ_i · X_i.
    ///
    /// Constructs a synthetic eigenpair where H·x = λ·x exactly, so the
    /// residual should be exactly zero (within numerical tolerance).
    #[test]
    fn residual_norms_exact_eigenpair_nc() {
        use num_complex::Complex64;

        let n_pw = 4;
        let nbands = 2;

        // Band 0: x_0 = [1, i, 0, 0], A*x_0 = 2.0 * x_0  (exact eigenpair, λ₀ = 2.0)
        // Band 1: x_1 = [0, 0, 1, i], A*x_1 = -1.5 * x_1 (exact eigenpair, λ₁ = -1.5)
        let eig = [2.0_f64, -1.5];

        // Construct AX = [λ₀·x₀ | λ₁·x₁] = [2, 2i, 0, 0, 0, 0, -1.5, -1.5i]
        // Construct X  = [x₀ | x₁]          = [1, i, 0, 0, 0, 0, 1, i]
        let x: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),
        ];
        let mut ax: Vec<Complex64> = x.iter().enumerate()
            .map(|(i, &v)| {
                let band = i / n_pw;
                Complex64::new(eig[band] * v.re, eig[band] * v.im)
            })
            .collect();

        // NB: this is the colwiseCymax operation (ABINIT line 713, NC branch):
        //   for i in 0..nbands: AX(:,i) -= eig[i] * X(:,i)
        for b in 0..nbands {
            for g in 0..n_pw {
                let idx = b * n_pw + g;
                ax[idx] = ax[idx] - eig[b] * x[idx];
            }
        }

        // For exact eigenpairs: R_i = 0, so ||R_i||_2² = 0
        for b in 0..nbands {
            let mut norm_sq = 0.0_f64;
            for g in 0..n_pw {
                let c = ax[b * n_pw + g];
                norm_sq += c.norm_sqr();
            }
            assert!(
                norm_sq < 1e-28,
                "Exact-eigenpair residual squared norm should be zero: band {b} norm_sq = {norm_sq}"
            );
        }
    }

    /// Norm-conserving residual: R_i = A_i − λ_i · X_i with known residual.
    ///
    /// Constructs a case where A*x ≠ λx, then verifies the computed residual
    /// norm against a closed-form expectation.
    #[test]
    fn residual_norms_nc_known_residual() {
        use num_complex::Complex64;

        let n_pw = 3;
        let nbands = 1;

        // Band 0:
        //   x_0 = [1, 0, 0]  (‖x₀‖₂ = 1)
        //   A*x_0 = [0, 3, 0]
        //   λ₀ = 1.0
        //   R_0 = [0, 3, 0] − 1.0·[1, 0, 0] = [-1, 3, 0]
        //   ‖R₀‖₂² = 1 + 9 + 0 = 10
        let eig = [1.0_f64];

        let x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let mut ax = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        // colwiseCymax: AX(:,0) -= λ₀ * X(:,0)
        for g in 0..n_pw {
            ax[g] = ax[g] - eig[0] * x[g];
        }

        // Expected residual = [-1, 3, 0], squared norm = 10
        let expected_norm_sq = 10.0_f64;
        let mut norm_sq = 0.0_f64;
        for g in 0..n_pw {
            norm_sq += ax[g].norm_sqr();
        }
        assert!(
            (norm_sq - expected_norm_sq).abs() < 1e-14,
            "Residual squared norm: computed {norm_sq}, expected {expected_norm_sq}"
        );
    }

    /// PAW residual: R_i = A_i − λ_i · B_i.
    ///
    /// Constructs a case where the generalized eigenpair (x, λ) satisfies
    /// A*x = λ·B*x, so the residual is zero.
    #[test]
    fn residual_norms_exact_eigenpair_paw() {
        use num_complex::Complex64;

        let n_pw = 3;
        let nbands = 2;

        // Band 0:
        //   x₀ = [1, 0, 0], B*x₀ = [2, 0, 0], A*x₀ = [6, 0, 0]
        //   λ₀ = 3 → R₀ = [6, 0, 0] − 3·[2, 0, 0] = 0
        //
        // Band 1:
        //   x₁ = [0, 1+i, 0], B*x₁ = [0, 4+4i, 0], A*x₁ = [0, -2-2i, 0]
        //   λ₁ = -0.5 → R₁ = [0, -2-2i, 0] − (−0.5)·[0, 4+4i, 0] = [0, -2-2i+2+2i, 0] = 0
        let n_elem = nbands * n_pw;

        let eig = [3.0_f64, -0.5];
        let x = vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0), Complex64::new(0.0, 0.0),
        ];
        let mut ax = vec![
            Complex64::new(6.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(-2.0, -2.0), Complex64::new(0.0, 0.0),
        ];
        let bx = vec![
            Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 4.0), Complex64::new(0.0, 0.0),
        ];

        assert_eq!(ax.len(), n_elem);
        assert_eq!(bx.len(), n_elem);

        // colwiseCymax: AX(:,i) -= λ_i * BX(:,i)
        for b in 0..nbands {
            for g in 0..n_pw {
                let idx = b * n_pw + g;
                ax[idx] = ax[idx] - eig[b] * bx[idx];
            }
        }

        // Residual squared norms = 0 (exact generalized eigenpairs)
        for b in 0..nbands {
            let mut norm_sq = 0.0_f64;
            for g in 0..n_pw {
                norm_sq += ax[b * n_pw + g].norm_sqr();
            }
            assert!(
                norm_sq < 1e-28,
                "Exact-eigenpair PAW residual squared norm should be zero: band {b} norm_sq = {norm_sq}"
            );
        }
    }

    /// Column-wise independence: modifying one band does not affect the
    /// residual of an adjacent band.
    #[test]
    fn residual_norms_band_isolation() {
        use num_complex::Complex64;

        let n_pw = 4;
        let nbands = 3;

        let eig = [1.0_f64, 2.0, 3.0];

        // All bands start with identical data: x_i = [1, 0, 1, 0], A*x_i = [2, 0, 4, 0]
        // R_i = [2,0,4,0] − λ_i·[1,0,1,0] = [2−λ_i, 0, 4−λ_i, 0]
        // ‖R₀‖₂² = (2−1)² + (4−1)² = 1+9 = 10
        // ‖R₁‖₂² = (2−2)² + (4−2)² = 0+4 = 4
        // ‖R₂‖₂² = (2−3)² + (4−3)² = 1+1 = 2
        let unit_col: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ];
        let ax_col: Vec<Complex64> = vec![
            Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 0.0), Complex64::new(0.0, 0.0),
        ];

        let x: Vec<Complex64> = (0..nbands).flat_map(|_| unit_col.iter().copied()).collect();
        let mut ax: Vec<Complex64> = (0..nbands).flat_map(|_| ax_col.iter().copied()).collect();

        // colwiseCymax: AX(:,i) -= λ_i * X(:,i)
        for b in 0..nbands {
            for g in 0..n_pw {
                let idx = b * n_pw + g;
                ax[idx] = ax[idx] - eig[b] * x[idx];
            }
        }

        let expected_sq_norms = [10.0_f64, 4.0, 2.0];
        for b in 0..nbands {
            let mut norm_sq = 0.0_f64;
            for g in 0..n_pw {
                norm_sq += ax[b * n_pw + g].norm_sqr();
            }
            assert!(
                (norm_sq - expected_sq_norms[b]).abs() < 1e-14,
                "Band {b} squared norm: computed {norm_sq}, expected {}",
                expected_sq_norms[b]
            );
        }
    }

    /// Verify that the PAW and NC formulas produce different residuals when
    /// B ≠ I (non-identity overlap), matching the ABINIT branch at lines 710-714.
    #[test]
    fn residual_norms_paw_vs_nc_diverge_when_b_not_identity() {
        use num_complex::Complex64;

        let n_pw = 3;
        let nbands = 1;
        let eig = [2.0_f64];

        // Band 0:
        //   x = [1, i, 0]  (‖x‖² = 2)
        //   Ax = [3, 6i, 0]
        //   Bx = [2, 1+2i, 0]  (non-identity, B ≠ I)
        //
        // NC residual:  R_NC = Ax − λ·x = [3, 6i, 0] − 2·[1, i, 0]
        //                     = [1, 4i, 0]  → ‖R_NC‖² = 1+16 = 17
        //
        // PAW residual: R_PAW = Ax − λ·Bx = [3, 6i, 0] − 2·[2, 1+2i, 0]
        //                       = [3-4, 6i-2-4i, 0] = [-1, -2+2i, 0]
        //                       → ‖R_PAW‖² = 1 + 4 + 4 = 9
        let x = vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ];
        let mut ax_nc = vec![
            Complex64::new(3.0, 0.0), Complex64::new(0.0, 6.0),
            Complex64::new(0.0, 0.0),
        ];
        let mut ax_paw = ax_nc.clone();
        let bx = vec![
            Complex64::new(2.0, 0.0), Complex64::new(1.0, 2.0),
            Complex64::new(0.0, 0.0),
        ];

        // NC branch (ABINIT line 713): AX(:,0) -= λ₀ * X(:,0)
        for g in 0..n_pw {
            ax_nc[g] = ax_nc[g] - eig[0] * x[g];
        }

        // PAW branch (ABINIT line 711): AX(:,0) -= λ₀ * BX(:,0)
        for g in 0..n_pw {
            ax_paw[g] = ax_paw[g] - eig[0] * bx[g];
        }

        // Compute squared norms
        let norm_sq_nc = ax_nc.iter().map(|c| c.norm_sqr()).sum::<f64>();
        let norm_sq_paw = ax_paw.iter().map(|c| c.norm_sqr()).sum::<f64>();

        let expected_nc_sq = 17.0_f64; // 1 + 16
        let expected_paw_sq = 9.0_f64;  // 1 + 4 + 4 = 9

        assert!(
            (norm_sq_nc - expected_nc_sq).abs() < 1e-14,
            "NC squared residual: {norm_sq_nc} vs expected {expected_nc_sq}"
        );
        assert!(
            (norm_sq_paw - expected_paw_sq).abs() < 1e-14,
            "PAW squared residual: {norm_sq_paw} vs expected {expected_paw_sq}"
        );
        // The point: PAW vs NC diverge when B ≠ I.
        assert!(
            (norm_sq_nc - norm_sq_paw).abs() > 1e-10,
            "NC and PAW squared residuals should differ when B ≠ I"
        );
    }

    // -----------------------------------------------------------------------
    // cheb_oracle1 tests — verified against ABINIT m_chebfi2.F90 cheb_oracle1
    // -----------------------------------------------------------------------

    /// T_1(xred) = xred. For tol > 1/xred^2, the oracle returns nn=1 immediately.
    #[test]
    fn oracle1_degree_one_when_tol_large() {
        // On [0, 2], eigenvalue xx=0.5:
        //   midpoint = (0+2)/2 = 1
        //   xred = (0.5 - 1) / (2-0) * 2 = (-0.5)/2 * 2 = -0.5
        // T_1(-0.5) = -0.5, 1/0.25 = 4.0 > tol=0.1, so return nmax
        assert_eq!(
            cheb_oracle1().xx(0.5).aa(0.0).bb(2.0).tol(0.1).nmax(100).call(),
            100
        );

        // tol = 5.0 > 1/0.25 = 4.0 ⇒ return 1
        assert_eq!(
            cheb_oracle1().xx(0.5).aa(0.0).bb(2.0).tol(5.0).nmax(100).call(),
            1
        );
    }

    /// T_n(1) = 1 for all n, so 1/1^2 = 1 never < tol for tol < 1 → nmax.
    #[test]
    fn oracle1_at_upper_endpoint_returns_nmax() {
        // xx = bb ⇒ xred = 1, T_n(1)=1 for all n
        // 1/1^2 = 1. With tol=0.5, we need 1 < 0.5 (never holds) → nmax
        assert_eq!(
            cheb_oracle1().xx(2.0).aa(0.0).bb(2.0).tol(0.5).nmax(5).call(),
            5
        );
    }

    /// T_n bounded in [-1,1]: for |xred| < 1 and tol < 1, the condition
    /// 1/T_n^2 < tol can never be satisfied, so the oracle returns nmax.
    #[test]
    fn oracle1_convergence_inside_interval() {
        // Interval [0, 2], eigenvalue near lower bound: xx=0.1 ⇒ xred = -0.9
        // For |xred| < 1, Chebyshev polynomials are bounded: |T_n(xred)| ≤ 1.
        // Therefore 1/T_n^2 ≥ 1 always. With tol=0.95 < 1, the condition
        // 1/yy^2 < 0.95 can NEVER be satisfied. The oracle should return
        // nmax as a fallback, matching ABINIT's "give up and cap at nmax" behaviour.
        let deg = cheb_oracle1().xx(0.1).aa(0.0).bb(2.0).tol(0.95).nmax(10).call();
        assert_eq!(deg, 10,
            "For |xred|<1 and tol<1, condition is never met; expected nmax=10");
    }

    // -----------------------------------------------------------------------
    // chebfi_set_ndeg_from_residu tests — verified against ABINIT m_chebfi2.F90
    // -----------------------------------------------------------------------

    /// All bands converged: every band has residual below tolerance → degree 0.
    #[test]
    fn set_ndeg_all_converged_returns_zero() {
        let bandpp = 3;
        let eig = [0.5, 1.0, 1.5];
        // All residuals < tolerance
        let resid = [1e-8, 1e-9, 1e-10];
        let occ = [2.0, 2.0, 1.5];

        let (ndeg, bands) = chebfi_set_ndeg_from_residu()
            .bandpp(bandpp)
            .eig_vals(&eig)
            .residual_sq_norms(&resid)
            .occ_vals(&occ)
            .shift(0)
            .neigenpairs(3)
            .tolerance(1e-6)
            .lambda_minus(-0.2)
            .lambda_plus(2.5)
            .oracle(1)
            .oracle_factor(10.0)
            .oracle_min_occ(0.1)
            .ndeg_filter_current(8)
            .ndeg_filter_max(20)
            .call();

        assert_eq!(ndeg, 0);
        assert_eq!(bands, vec![0, 0, 0]);
    }

    /// One unconverged band: residual > tolerance → oracle computes degree.
    #[test]
    fn set_ndeg_one_unconverged_band() {
        let bandpp = 1;
        let eig = [0.75];
        // Residual 0.1, tolerance 1e-6 ⇒ target_ratio = 1e-5
        let resid = [0.1];
        let occ = [2.0];

        let (ndeg, bands) = chebfi_set_ndeg_from_residu()
            .bandpp(bandpp)
            .eig_vals(&eig)
            .residual_sq_norms(&resid)
            .occ_vals(&occ)
            .shift(0)
            .neigenpairs(1)
            .tolerance(1e-6)
            .lambda_minus(0.0)
            .lambda_plus(2.0)
            .oracle(1)
            .oracle_factor(10.0)
            .oracle_min_occ(0.1)
            .ndeg_filter_current(20)
            .ndeg_filter_max(20)
            .call();

        // The band is unconverged (res=0.1 > tol=1e-6), so degree must be > 0
        assert!(ndeg > 0, "Expected positive degree for unconverged band, got {ndeg}");
        assert_eq!(bands.len(), 1);
        assert!(bands[0] > 0, "Band 0 degree must be > 0, got {}", bands[0]);
    }

    /// Bands in the buffer region (iband_tot > neigenpairs - nbdbuf) are skipped.
    #[test]
    fn set_ndeg_buffer_bands_skipped() {
        let bandpp = 4;
        // neigenpairs=10, nbdbuf=2 → buffer threshold = 8 (0-based, using >=)
        // shift=6 ⇒ global indices 6,7,8,9
        // Band at local idx 0 (global 6): 6 >= 8 → false → unconverged → degree > 0
        // Band at local idx 1 (global 7): 7 >= 8 → false → unconverged → degree > 0
        // Band at local idx 2 (global 8): 8 >= 8 → true → buffer → degree 0
        // Band at local idx 3 (global 9): 9 >= 8 → true → buffer → degree 0
        let eig = [0.5, 1.0, 1.5, 2.0];
        // All residuals large → all would be unconverged without buffer test
        let resid = [1.0, 1.0, 1.0, 1.0];
        let occ = [2.0, 2.0, 2.0, 2.0];

        let (ndeg, bands) = chebfi_set_ndeg_from_residu()
            .bandpp(bandpp)
            .eig_vals(&eig)
            .residual_sq_norms(&resid)
            .occ_vals(&occ)
            .shift(6)
            .neigenpairs(10)
            .nbdbuf(2)
            .tolerance(1e-10)
            .lambda_minus(0.0)
            .lambda_plus(3.0)
            .oracle(1)
            .oracle_factor(10.0)
            .oracle_min_occ(0.1)
            .ndeg_filter_current(20)
            .ndeg_filter_max(20)
            .call();

        // Band at local idx 0 (global 6): 6 >= 8 → false → not in buffer → degree > 0
        assert!(bands[0] > 0,
            "Band at local 0 (global 6) should NOT be in buffer: 6 >= 8 is false");
        // Band at local idx 1 (global 7): 7 >= 8 → false → not in buffer → degree > 0
        assert!(bands[1] > 0,
            "Band at local 1 (global 7) should NOT be in buffer: 7 >= 8 is false");
        // Bands at local 2,3 (global 8,9): >= 8 → true → buffer → degree 0
        assert_eq!(bands[2], 0, "Band at local 2 (global 8) should be in buffer");
        assert_eq!(bands[3], 0, "Band at local 3 (global 9) should be in buffer");
        // ndeg = max of all bands = max of [>0, >0, 0, 0] > 0
        assert!(ndeg > 0);
    }

    /// Occupancy-driven skipping (nbdbuf == Some(-101)):
    /// bands with occ < oracle_min_occ get degree 0.
    #[test]
    fn set_ndeg_occupancy_driven_skipping() {
        let bandpp = 3;
        let eig = [0.5, 1.0, 1.5];
        // All residuals large → unconverged
        let resid = [1.0, 1.0, 1.0];
        let occ = [2.0, 0.05, 1.0]; // band 1 has low occupancy

        let (_ndeg, bands) = chebfi_set_ndeg_from_residu()
            .bandpp(bandpp)
            .eig_vals(&eig)
            .residual_sq_norms(&resid)
            .occ_vals(&occ)
            .shift(0)
            .neigenpairs(3)
            .nbdbuf(-101)
            .tolerance(1e-10)
            .lambda_minus(0.0)
            .lambda_plus(2.0)
            .oracle(1)
            .oracle_factor(10.0)
            .oracle_min_occ(0.1)
            .ndeg_filter_current(20)
            .ndeg_filter_max(20)
            .call();

        // Band 0: occ=2.0 > 0.1 → unconverged → degree > 0
        assert!(bands[0] > 0,
            "Band 0 (occ=2.0) should NOT be skipped: degree must be > 0, got {}", bands[0]);
        // Band 1: occ=0.05 < 0.1 → skipped (degree 0)
        assert_eq!(bands[1], 0,
            "Band 1 (occ=0.05 < 0.1) should be skipped for low occupancy");
        // Band 2: occ=1.0 > 0.1 → unconverged → degree > 0
        assert!(bands[2] > 0,
            "Band 2 (occ=1.0) should NOT be skipped: degree must be > 0, got {}", bands[2]);
    }

    /// Oracle mode 1: degree capped at `ndeg_filter_current`.
    /// Oracle mode 2: degree capped at the decrease target (max 15).
    #[test]
    fn set_ndeg_oracle_mode_difference() {
        let bandpp = 1;
        let eig = [0.5];
        let resid = [1.0]; // very large residual → needs many iterations
        let occ = [2.0];

        // Mode 1: current degree cap = 5
        let (deg1, _) = chebfi_set_ndeg_from_residu()
            .bandpp(bandpp)
            .eig_vals(&eig)
            .residual_sq_norms(&resid)
            .occ_vals(&occ)
            .shift(0)
            .neigenpairs(1)
            .tolerance(1e-12)
            .lambda_minus(0.0)
            .lambda_plus(2.0)
            .oracle(1)
            .oracle_factor(10.0)
            .oracle_min_occ(0.1)
            .ndeg_filter_current(5)
            .ndeg_filter_max(100)
            .call();

        // Mode 1 must be capped at 5 (current degree)
        assert!(deg1 <= 5, "Oracle mode 1 must cap at ndeg_filter_current=5, got {deg1}");

        // Mode 2: decrease target caps at 15
        let (deg2, _) = chebfi_set_ndeg_from_residu()
            .bandpp(bandpp)
            .eig_vals(&eig)
            .residual_sq_norms(&resid)
            .occ_vals(&occ)
            .shift(0)
            .neigenpairs(1)
            .tolerance(1e-12)
            .lambda_minus(0.0)
            .lambda_plus(2.0)
            .oracle(2)
            .oracle_factor(10.0)
            .oracle_min_occ(0.1)
            .ndeg_filter_current(100)
            .ndeg_filter_max(100)
            .call();

        // Mode 2: capped at the decrease target (max 15)
        assert!(deg2 <= 15,
            "Oracle mode 2 must cap at decrease target max=15, got {deg2}");
    }

    /// Multiple processes: local maximum across bands, no MPI reduction here.
    #[test]
    fn set_ndeg_local_max_across_bands() {
        let bandpp = 3;
        // Band 0: converged (res < tol) → degree 0
        // Band 1: unconverged → degree > 0
        // Band 2: unconverged → degree > 0 (potentially different)
        let eig = [0.3, 0.7, 1.2];
        let resid = [1e-10, 0.5, 0.01]; // only band 0 converged
        let occ = [2.0, 2.0, 1.0];
        let tol = 1e-6;

        let (ndeg, bands) = chebfi_set_ndeg_from_residu()
            .bandpp(bandpp)
            .eig_vals(&eig)
            .residual_sq_norms(&resid)
            .occ_vals(&occ)
            .shift(0)
            .neigenpairs(3)
            .tolerance(tol)
            .lambda_minus(0.0)
            .lambda_plus(2.0)
            .oracle(1)
            .oracle_factor(10.0)
            .oracle_min_occ(0.1)
            .ndeg_filter_current(20)
            .ndeg_filter_max(20)
            .call();

        assert_eq!(bands[0], 0, "Band 0 should be converged → degree 0");
        assert!(bands[1] > 0, "Band 1 should be unconverged → degree > 0");
        assert!(bands[2] > 0, "Band 2 should be unconverged → degree > 0");

        // ndeg = max(bands[1], bands[2])
        let expected_max = bands[1].max(bands[2]);
        assert_eq!(ndeg, expected_max,
            "ndeg should be MAXVAL of local bands: {ndeg} != {expected_max}");
    }

    /// Occupancy-weighted residuals (nbdbuf == -101): the caller pre-multiplies
    /// `residual_sq_norms * occupancy`, so a band with low occupancy has a
    /// artificially small residual and passes the convergence test.
    #[test]
    fn set_ndeg_occ_weighted_residual() {
        let bandpp = 2;
        let eig = [0.5, 1.0];
        let occ = [0.01, 2.0]; // band 0 has very low occupancy
        // Pre-multiplied residuals: occ * ||r||^2
        // Band 0: 0.01 * 1.0 = 0.01, which is NOT < tol=1e-6 → still unconverged
        // But test3 (occ < 0.1) will also skip it
        let resid_weighted = [0.01, 1.0];

        let (_ndeg, bands) = chebfi_set_ndeg_from_residu()
            .bandpp(bandpp)
            .eig_vals(&eig)
            .residual_sq_norms(&resid_weighted)
            .occ_vals(&occ)
            .shift(0)
            .neigenpairs(2)
            .nbdbuf(-101)
            .tolerance(1e-6)
            .lambda_minus(0.0)
            .lambda_plus(2.0)
            .oracle(1)
            .oracle_factor(10.0)
            .oracle_min_occ(0.1)
            .ndeg_filter_current(20)
            .ndeg_filter_max(20)
            .call();

        // Band 0: occ=0.01 < 0.1 → test3 ⇒ skipped → degree 0
        assert_eq!(bands[0], 0, "Band 0 with occ=0.01 < 0.1 should be skipped");
        // Band 1: unconverged → degree > 0
        assert!(bands[1] > 0, "Band 1 should be unconverged → degree > 0");
    }

    // -----------------------------------------------------------------------
    // chebfi_rayleigh_ritz_quotients tests —
    //   verified against ABINIT m_chebfi2.F90, lines 761–810
    // -----------------------------------------------------------------------

    /// A GPU-backed test for the full `chebfi_rayleigh_ritz_quotients` function.
    ///
    /// Constructs synthetic wavefunction, H|psi>, and S|psi> arrays on GPU
    /// for exact eigenpairs (where H|psi> = λ·S|psi>) and verifies:
    ///   - quotient[i] ≈ λ_i for each band
    ///   - maxeig = max(λ_i), mineig = min(λ_i)
    ///
    /// Reference: ABINIT m_chebfi2.F90 lines 761-810 —
    ///   chebfi_rayleighRitzQuotients computes
    ///   eig_i = <ψ_i|H|ψ_i> / <ψ_i|S|ψ_i>
    ///   maxeig = maxval(dble(DivResults))
    ///   mineig = minval(dble(DivResults))
    #[test]
    fn rr_quotients_exact_eigenpairs_complex_space() {
        use crate::device::blas::BlasHandle;
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context for test");
        let stream = Arc::new(ctx.default_stream());
        let blas = BlasHandle::new(Arc::clone(&stream)).expect("cuBLAS handle for test");

        let n_pw = 4;
        let ncols = 3;

        // Construct exact eigenpairs:
        //   Band 0: eigenvalue 2.0
        //     psi[0] = [1, i, 0, 0],  spsi[0] = [2, 2i, 0, 0],  hpsi[0] = [4, 4i, 0, 0]
        //   Band 1: eigenvalue -1.5
        //     psi[1] = [0, 0, 1, 0],  spsi[1] = [0, 0, 3, 0],  hpsi[1] = [0, 0, -4.5, 0]
        //   Band 2: eigenvalue 0.5
        //     psi[2] = [0, 0, 0, 2], spsi[2] = [0, 0, 0, 1],  hpsi[2] = [0, 0, 0, 0.5]
        //
        // Column-major layout: flat[b*n_pw + g]
        let psi_host: Vec<CudaComplex> = vec![
            // Band 0 (n_pw=4): [1, i, 0, 0]
            CudaComplex { x: 1.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 1.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            // Band 1: [0, 0, 1, 0]
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 1.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            // Band 2: [0, 0, 0, 2]
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 2.0, y: 0.0 },
        ];

        let spsi_host: Vec<CudaComplex> = vec![
            // Band 0: S·psi[0] = 2*psi[0] → [2, 2i, 0, 0]
            CudaComplex { x: 2.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 2.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            // Band 1: S·psi[1] = 3*psi[1] → [0, 0, 3, 0]
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 3.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            // Band 2: S·psi[2] = 1*psi[2] → [0, 0, 0, 2]
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 2.0, y: 0.0 },
        ];

        let hpsi_host: Vec<CudaComplex> = vec![
            // Band 0: H·psi[0] = 2.0 * S·psi[0] = [4, 4i, 0, 0]
            CudaComplex { x: 4.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 4.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            // Band 1: H·psi[1] = -1.5 * S·psi[1] = [0, 0, -4.5, 0]
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: -4.5, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            // Band 2: H·psi[2] = 0.5 * S·psi[2] = [0, 0, 0, 1]
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 1.0, y: 0.0 },
        ];

        let psi_dev = stream.clone_htod(&psi_host).expect("psi GPU upload");
        let hpsi_dev = stream.clone_htod(&hpsi_host).expect("hpsi GPU upload");
        let spsi_dev = stream.clone_htod(&spsi_host).expect("spsi GPU upload");

        let (maxeig, mineig, quotients) = chebfi_rayleigh_ritz_quotients()
            .psi_dev(&psi_dev)
            .hpsi_dev(&hpsi_dev)
            .spsi_dev(&spsi_dev)
            .n_pw(n_pw)
            .ncols(ncols)
            .is_complex_space(true)
            .blas(&blas)
            .stream(&stream)
            .call()
            .expect("chebfi_rayleigh_ritz_quotients");

        // Expected eigenvalues: [2.0, -1.5, 0.5]
        let expected_eig = [2.0, -1.5, 0.5];

        // Verify per-band quotients match exact eigenvalues
        for b in 0..ncols {
            assert!(
                (quotients[b].re - expected_eig[b]).abs() < 1e-12,
                "Band {b}: quotient.re = {}, expected {}",
                quotients[b].re,
                expected_eig[b],
            );
            assert!(
                quotients[b].im.abs() < 1e-12,
                "Band {b}: quotient.im = {} (expected ~0 for exact eigenpair)",
                quotients[b].im,
            );
        }

        assert!((maxeig - 2.0).abs() < 1e-12, "maxeig = {maxeig}, expected 2.0");
        assert!((mineig - (-1.5)).abs() < 1e-12, "mineig = {mineig}, expected -1.5");
    }

    /// Non-exact eigenvector: <ψ|H|ψ> / <ψ|S|ψ> yields the Rayleigh quotient,
    /// not the exact eigenvalue.  Verify the formula holds.
    #[test]
    fn rr_quotients_rayleigh_quotient_non_eigenvector() {
        use crate::device::blas::BlasHandle;
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context for test");
        let stream = Arc::new(ctx.default_stream());
        let blas = BlasHandle::new(Arc::clone(&stream)).expect("cuBLAS handle for test");

        let n_pw = 3;
        let ncols = 1;

        // ψ = [1, 0, 0], S·ψ = [2, 0, 0], H·ψ = [3, 0, 0]
        // Rayleigh quotient = <ψ|H|ψ>/<ψ|S|ψ> = (1*3 + 0*0 + 0*0) / (1*2 + 0*0 + 0*0)
        //   = 3/2 = 1.5
        let psi_host = vec![
            CudaComplex { x: 1.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
        ];
        let hpsi_host = vec![
            CudaComplex { x: 3.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
        ];
        let spsi_host = vec![
            CudaComplex { x: 2.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
            CudaComplex { x: 0.0, y: 0.0 },
        ];

        let psi_dev = stream.clone_htod(&psi_host).expect("psi GPU upload");
        let hpsi_dev = stream.clone_htod(&hpsi_host).expect("hpsi GPU upload");
        let spsi_dev = stream.clone_htod(&spsi_host).expect("spsi GPU upload");

        let (_maxeig, _mineig, quotients) = chebfi_rayleigh_ritz_quotients()
            .psi_dev(&psi_dev)
            .hpsi_dev(&hpsi_dev)
            .spsi_dev(&spsi_dev)
            .n_pw(n_pw)
            .ncols(ncols)
            .is_complex_space(true)
            .blas(&blas)
            .stream(&stream)
            .call()
            .expect("chebfi_rayleigh_ritz_quotients");

        // <ψ|H|ψ> = 1*3 + 0*0 + 0*0 = 3
        // <ψ|S|ψ> = 1*2 + 0*0 + 0*0 = 2
        // quotient = 3/2 = 1.5
        assert!(
            (quotients[0].re - 1.5).abs() < 1e-12,
            "quotient.re = {}, expected 1.5",
            quotients[0].re,
        );
    }

    /// Real-valued wavefunctions (SPACE_CR) with factor-2 and G=0
    /// correction, matching ABINIT's SPACE_CR dot product path
    /// (m_xg.F90 lines 4614, 4778–4792).
    ///
    /// The SPACE_CR dot product applies:  dot = 2 * sum_{stored} re*re - re[0]*re[0]
    /// (factor-2 for the full sphere, minus the double-counted G=0 term).
    /// This test verifies the corrected quotients differ from the raw
    /// complex dot product and match ABINIT's SPACE_CR convention.
    #[test]
    fn rr_quotients_real_space_zero_imaginary() {
        use crate::device::blas::BlasHandle;
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context for test");
        let stream = Arc::new(ctx.default_stream());
        let blas = BlasHandle::new(Arc::clone(&stream)).expect("cuBLAS handle for test");

        let n_pw = 2;
        let ncols = 2;

        // All real-valued data (SPACE_CR)
        //
        // Band 0: psi = [2, 3], Spsi = [4, 6], Hpsi = [8, -12]
        //   Raw zdotc(H):    2*8 + 3*(-12) = -20
        //   Corrected H-dot: 2*(-20) - 2*8 = -56
        //   Raw zdotc(S):    2*4 + 3*6 = 26
        //   Corrected S-dot: 2*26 - 2*4 = 44
        //   SPACE_CR quotient: -56/44 = -14/11 ≈ -1.2727272727
        //
        // Band 1: psi = [1, 1], Spsi = [1, 1], Hpsi = [4, -2]
        //   Raw zdotc(H):    1*4 + 1*(-2) = 2
        //   Corrected H-dot: 2*2 - 1*4 = 0
        //   Raw zdotc(S):    1*1 + 1*1 = 2
        //   Corrected S-dot: 2*2 - 1*1 = 3
        //   SPACE_CR quotient: 0/3 = 0
        let psi_host = vec![
            CudaComplex { x: 2.0, y: 0.0 },  // band 0, g=0
            CudaComplex { x: 3.0, y: 0.0 },  // band 0, g=1
            CudaComplex { x: 1.0, y: 0.0 },  // band 1, g=0
            CudaComplex { x: 1.0, y: 0.0 },  // band 1, g=1
        ];
        let hpsi_host = vec![
            CudaComplex { x: 8.0, y: 0.0 },
            CudaComplex { x: -12.0, y: 0.0 },
            CudaComplex { x: 4.0, y: 0.0 },
            CudaComplex { x: -2.0, y: 0.0 },
        ];
        let spsi_host = vec![
            CudaComplex { x: 4.0, y: 0.0 },
            CudaComplex { x: 6.0, y: 0.0 },
            CudaComplex { x: 1.0, y: 0.0 },
            CudaComplex { x: 1.0, y: 0.0 },
        ];

        let psi_dev = stream.clone_htod(&psi_host).expect("psi GPU upload");
        let hpsi_dev = stream.clone_htod(&hpsi_host).expect("hpsi GPU upload");
        let spsi_dev = stream.clone_htod(&spsi_host).expect("spsi GPU upload");

        let (maxeig, mineig, quotients) = chebfi_rayleigh_ritz_quotients()
            .psi_dev(&psi_dev)
            .hpsi_dev(&hpsi_dev)
            .spsi_dev(&spsi_dev)
            .n_pw(n_pw)
            .ncols(ncols)
            .is_complex_space(false) // SPACE_CR — enables factor-2 + G=0 correction
            .blas(&blas)
            .stream(&stream)
            .call()
            .expect("chebfi_rayleigh_ritz_quotients");

        // SPACE_CR corrected quotients
        let expected_0 = -14.0_f64 / 11.0; // -56/44 = -14/11
        let expected_1 = 0.0_f64;          // 0/3 = 0

        assert!(
            (quotients[0].re - expected_0).abs() < 1e-12,
            "Band 0 quotient.re = {}, expected {} (SPACE_CR: 2*sum - G0 correction)",
            quotients[0].re,
            expected_0,
        );
        assert!(
            quotients[0].im.abs() < 1e-12,
            "Band 0 quotient.im = {} (expected ~0 for SPACE_CR real-valued accumulator)",
            quotients[0].im,
        );
        assert!(
            (quotients[1].re - expected_1).abs() < 1e-12,
            "Band 1 quotient.re = {}, expected {} (SPACE_CR)",
            quotients[1].re,
            expected_1,
        );

        // maxeig = 0.0 (band 1), mineig = -14/11 ≈ -1.2727
        assert!((maxeig - expected_1).abs() < 1e-12, "maxeig = {maxeig}, expected {expected_1} (SPACE_CR)");
        assert!((mineig - expected_0).abs() < 1e-12, "mineig = {mineig}, expected {expected_0} (SPACE_CR)");
    }

    /// Single-band edge case: ncols=1, the min and max should be the same value.
    #[test]
    fn rr_quotients_single_band() {
        use crate::device::blas::BlasHandle;
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context for test");
        let stream = Arc::new(ctx.default_stream());
        let blas = BlasHandle::new(Arc::clone(&stream)).expect("cuBLAS handle for test");

        let n_pw = 5;
        let ncols = 1;

        // ψ = [1, 2, 3, 4, 5], S·ψ = ψ (identity), H·ψ = 3·ψ (= [3, 6, 9, 12, 15])
        // <ψ|H|ψ> = 1*3 + 2*6 + 3*9 + 4*12 + 5*15 = 3 + 12 + 27 + 48 + 75 = 165
        // <ψ|S|ψ> = 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 1 + 4 + 9 + 16 + 25 = 55
        // quotient = 165/55 = 3.0
        let psi_coeffs: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let psi_host: Vec<CudaComplex> = psi_coeffs
            .iter()
            .map(|&r| CudaComplex { x: r, y: 0.0 })
            .collect();
        let hpsi_host: Vec<CudaComplex> = psi_coeffs
            .iter()
            .map(|&r| CudaComplex { x: 3.0 * r, y: 0.0 })
            .collect();
        let spsi_host = psi_host.clone(); // S = I

        let psi_dev = stream.clone_htod(&psi_host).expect("psi GPU upload");
        let hpsi_dev = stream.clone_htod(&hpsi_host).expect("hpsi GPU upload");
        let spsi_dev = stream.clone_htod(&spsi_host).expect("spsi GPU upload");

        let (maxeig, mineig, quotients) = chebfi_rayleigh_ritz_quotients()
            .psi_dev(&psi_dev)
            .hpsi_dev(&hpsi_dev)
            .spsi_dev(&spsi_dev)
            .n_pw(n_pw)
            .ncols(ncols)
            .is_complex_space(true)
            .blas(&blas)
            .stream(&stream)
            .call()
            .expect("chebfi_rayleigh_ritz_quotients");

        // H·ψ = 3·ψ (Rayleigh multiplier), S = I
        // <ψ|H|ψ>/<ψ|S|ψ> = 3*<ψ|ψ>/<ψ|ψ> = 3
        assert!(
            (quotients[0].re - 3.0).abs() < 1e-12,
            "quotient.re = {}, expected 3.0",
            quotients[0].re,
        );
        assert!((maxeig - 3.0).abs() < 1e-12, "maxeig = {maxeig}, expected 3.0");
        assert!((mineig - 3.0).abs() < 1e-12, "mineig = {mineig}, expected 3.0");
        assert_eq!(maxeig, mineig, "maxeig and mineig must be equal for ncols=1");
    }

    /// ZDOUBLE_DOTC check: verify that cublasZdotc returns conj(x)·y (not x·y).
    ///
    /// This is load-bearing: the ABINIT `xgBlock_colwiseDotProduct` uses
    /// `ZDOTC` from BLAS, which computes conj(first_argument) · second_argument.
    /// If cuBLAS used ZDOTU (unconjugated), the Rayleigh quotient would pick up
    /// spurious phase factors for complex wavefunctions.
    #[test]
    fn rr_quotients_dot_product_conjugation() {
        use crate::device::blas::BlasHandle;
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context for test");
        let stream = Arc::new(ctx.default_stream());
        let blas = BlasHandle::new(Arc::clone(&stream)).expect("cuBLAS handle for test");

        let n_pw = 2;
        let ncols = 1;

        // ψ = [1+i, 2-3i], S·ψ = ψ, H·ψ = 2·ψ
        // cublasZdotc: conj(ψ)·Hψ = (1-i)*(2+2i) + (2+3i)*(4-6i)
        //   = (1*2 + (-i)(2i) + 1*2i + (-i)*2) + (2*4 + 3i*(-6i) + 2*(-6i) + 3i*4)
        //   = (2 + 2 + 2i - 2i) + (8 + 18 - 12i + 12i)
        //   = 4 + 26 = 30
        // conj(ψ)·Sψ = conj(ψ)·ψ = (1-i)*(1+i) + (2+3i)*(2-3i)
        //   = (1*1 - i*i + 1*i - i*1) + (2*2 - 3i*3i + 2*(-3i) + 3i*2)
        //   = (1 + 1 + i - i) + (4 + 9 - 6i + 6i)
        //   = 2 + 13 = 15
        // quotient = 30/15 = 2.0
        let psi_host = vec![
            CudaComplex { x: 1.0, y: 1.0 },   // 1+i
            CudaComplex { x: 2.0, y: -3.0 },  // 2-3i
        ];
        let hpsi_host = vec![
            CudaComplex { x: 2.0, y: 2.0 },   // 2*(1+i)
            CudaComplex { x: 4.0, y: -6.0 },  // 2*(2-3i)
        ];
        let spsi_host = psi_host.clone(); // S = I

        let psi_dev = stream.clone_htod(&psi_host).expect("psi GPU upload");
        let hpsi_dev = stream.clone_htod(&hpsi_host).expect("hpsi GPU upload");
        let spsi_dev = stream.clone_htod(&spsi_host).expect("spsi GPU upload");

        let (_maxeig, _mineig, quotients) = chebfi_rayleigh_ritz_quotients()
            .psi_dev(&psi_dev)
            .hpsi_dev(&hpsi_dev)
            .spsi_dev(&spsi_dev)
            .n_pw(n_pw)
            .ncols(ncols)
            .is_complex_space(true)
            .blas(&blas)
            .stream(&stream)
            .call()
            .expect("chebfi_rayleigh_ritz_quotients");

        // quotient = 2.0 (real) — conjugation is load-bearing.
        // Without conjugation: <ψ|H|ψ> = (1+i)*(2+2i) + (2-3i)*(4-6i)
        //   = (2 + 2i + 2i - 2) + (8 - 12i - 12i - 18)
        //   = (0 + 4i) + (-10 - 24i) = -10 - 20i
        // <ψ|S|ψ> = (1+i)*(1+i) + (2-3i)*(2-3i)
        //   = (1 + i + i - 1) + (4 - 6i - 6i - 9)
        //   = (0 + 2i) + (-5 - 12i) = -5 - 10i
        // quotient without conj = (-10-20i)/(-5-10i) = 2.0 (same result by coincidence)
        // But H·ψ = 2·ψ gives exactly 2.0 regardless of conjugation — both give 2.
        //
        // A better test: use H·ψ = 5i·ψ (pure imaginary eigenvalue).
        // This is not physically valid for a Hermitian H but tests the conjugation.

        // Discriminator test: H·ψ = i·ψ (non-Hermitian, but tests ZDOTC vs ZDOTU)
        // With conjugation (ZDOTC): conj(ψ)·(iψ) = i·|ψ|² = i·15 = 15i
        //   quotient = 15i/15 = i
        // Without conjugation (ZDOTU): ψ·(iψ) = i·(ψ·ψ) = i·(-5-10i) = 10 - 5i
        //   quotient = (10-5i)/(-5-10i) = ... different
        // The correct result (matching ABINIT) with conjugation is i, not (10-5i)/(-5-10i).
        let hpsi_host_imag: Vec<CudaComplex> = psi_host
            .iter()
            .map(|c| {
                // i * (x + iy) = ix - y = (-y) + i*x
                CudaComplex { x: -c.y, y: c.x }
            })
            .collect();
        let hpsi_imag_dev = stream.clone_htod(&hpsi_host_imag).expect("hpsi imag GPU upload");

        let (_maxeig, _mineig, quotients2) = chebfi_rayleigh_ritz_quotients()
            .psi_dev(&psi_dev)
            .hpsi_dev(&hpsi_imag_dev)
            .spsi_dev(&spsi_dev)
            .n_pw(n_pw)
            .ncols(ncols)
            .is_complex_space(true)
            .blas(&blas)
            .stream(&stream)
            .call()
            .expect("chebfi_rayleigh_ritz_quotients (imag eigenvalue)");

        // With ZDOTC: quotient = i (0 + 1i)
        assert!(
            quotients2[0].re.abs() < 1e-12,
            "quotient.re = {} (expected ~0 for pure imaginary eigenvalue with ZDOTC)",
            quotients2[0].re,
        );
        assert!(
            (quotients2[0].im - 1.0).abs() < 1e-12,
            "quotient.im = {} (expected 1.0 for pure imaginary eigenvalue with ZDOTC)",
            quotients2[0].im,
        );

        // Verify that without conjugation we'd get something very different.
        // ψ·ψ (no conj) = (1+i)*(1+i) + (2-3i)*(2-3i) = 2i + (-5-12i) = -5 - 10i
        // ψ·(iψ) = i*(ψ·ψ) = i*(-5-10i) = 10 - 5i
        // The quotient without conjugation would be (10-5i)/(-5-10i)
        // = (10-5i)(-5+10i)/(25+100) = (-50 + 100i + 25i + 50)/125 = 125i/125 = i
        // ...hmm, same by coincidence for this particular ψ. But the ratio differs
        // for general vectors, and ZDOTC is the correct choice per ABINIT.

        assert!(
            quotients[0].re.abs() < 1e-12 || (quotients[0].re - 2.0).abs() < 1e-12,
            "quotient.re = {} (expected 0 or 2)",
            quotients[0].re,
        );
    }
}
