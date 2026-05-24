// ---------------------------------------------------------------------------
// Chebyshev polynomial filtering for DFT SCF diagonalization
// ---------------------------------------------------------------------------
//
// Implements:
//   1. SpectralBounds estimation (lambda_max, eps_cut, center, half_width)
//   2. CUDA kernel compilation (NVRTC) for H|psi> operations
//   3. apply_full_hamiltonian() — T + V_loc (FFT-based) on GPU
//   4. apply_scaled_hamiltonian() — sigma(H).psi
//   5. chebyshev_filter() — main driver: recurrence + norm check

use std::marker::PhantomData;
use std::sync::Arc;

use chemrust_hamiltonian_core::{CellGeometry, GVectorGrid, PseudopotentialSet};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::device::blas::{self, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::device::{CudaComplex, Gpu};
use crate::eigensolver::vnl_data::VnlBatchData;
use cudarc::cusolver::sys::cublasOperation_t;

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
// Helper: call cuFFT C2C in-place (same buffer for input and output)
// ---------------------------------------------------------------------------

/// Call `c2c_inverse` in-place. cuFFT natively supports in-place transforms,
/// so passing the same `&mut` twice via raw pointer is correct.
pub(crate) unsafe fn c2c_inverse_inplace(
    plan: &BatchedFftPlan3d,
    buf: &mut CudaSlice<CudaComplex>,
) -> Result<(), cudarc::cufft::result::CufftError> {
    let ptr = buf as *mut CudaSlice<CudaComplex>;
    unsafe { plan.c2c_inverse(&mut *ptr, &mut *ptr) }
}

/// Call `c2c_forward` in-place.
unsafe fn c2c_forward_inplace(
    plan: &BatchedFftPlan3d,
    buf: &mut CudaSlice<CudaComplex>,
) -> Result<(), cudarc::cufft::result::CufftError> {
    let ptr = buf as *mut CudaSlice<CudaComplex>;
    unsafe { plan.c2c_forward(&mut *ptr, &mut *ptr) }
}
use crate::layout::{ColumnDistributed, RowDistributed, WavefunctionSet};
use crate::types::{Error, KineticEnergies, KPoint};

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

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled via NVRTC at startup)
// ---------------------------------------------------------------------------

const CUDA_KERNEL_SRC: &str = "
extern \"C\" __global__ void zero_buffer(double2* buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) { buf[i].x = 0.0; buf[i].y = 0.0; }
}

extern \"C\" __global__ void zero_buffer_real(double* buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) { buf[i] = 0.0; }
}

extern \"C\" __global__ void init_kinetic(
    double2* hpsi, const double2* psi, const double* kinetic,
    int n_pw, int n_bands
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int g = tid % n_pw;
        double k = kinetic[g];
        hpsi[tid].x = psi[tid].x * k;
        hpsi[tid].y = psi[tid].y * k;
        tid += stride;
    }
}

extern \"C\" __global__ void scatter_pw_to_grid(
    const double2* psi, const int* fft_idx,
    double2* grid, int n_pw, int n_bands, int grid_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int b = tid / n_pw;
        int g = tid % n_pw;
        grid[b * grid_size + fft_idx[g]] = psi[b * n_pw + g];
        tid += stride;
    }
}

extern \"C\" __global__ void veff_multiply(
    double2* grid, const double* veff,
    int grid_size, int n_bands
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * grid_size) {
        int r = tid % grid_size;
        double v = veff[r];
        grid[tid].x *= v;
        grid[tid].y *= v;
        tid += stride;
    }
}

extern \"C\" __global__ void gather_add_kinetic(
    const double2* grid, const int* fft_idx,
    double2* result, int n_pw, int n_bands, int grid_size, double inv_ntotal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int b = tid / n_pw;
        int g = tid % n_pw;
        double2 v = grid[b * grid_size + fft_idx[g]];
        v.x *= inv_ntotal;
        v.y *= inv_ntotal;
        result[tid].x += v.x;
        result[tid].y += v.y;
        tid += stride;
    }
}

extern \"C\" __global__ void transpose_col_to_row(
    const double2* col, double2* row, int n_bands, int n_pw
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int b = tid / n_pw;
        int g = tid % n_pw;
        row[g * n_bands + b] = col[b * n_pw + g];
        tid += stride;
    }
}

extern \"C\" __global__ void accumulate_density(
    const double2* psi_r, const double* occ,
    double* rho, int n_bands, int grid_size, double inv_omega
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (r < grid_size) {
        double sum = 0.0;
        for (int b = 0; b < n_bands; b++) {
            double2 psi = psi_r[b * grid_size + r];
            sum += occ[b] * (psi.x * psi.x + psi.y * psi.y);
        }
        rho[r] = sum * inv_omega;
        r += stride;
    }
}

extern \"C\" __global__ void transpose_row_to_col(
    const double2* row, double2* col, int n_bands, int n_pw
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < n_bands * n_pw) {
        int b = tid / n_pw;
        int g = tid % n_pw;
        col[b * n_pw + g] = row[g * n_bands + b];
        tid += stride;
    }
}

// dst[b*n_pw + g] += alpha * src[b*n_pw + g] * scale[b]
// Used for: Y·Λ_Y term (Step 3), S·X·Λ subtraction (Step 1), X·Λ_Y reconstruction (Step 4)
extern \"C\" __global__ void band_scale_axpy(
    double2* dst,
    const double2* src,
    const double* scale,
    double alpha,
    int n_pw, int n_bands
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_pw * n_bands;
    if (idx >= total) return;
    int b = idx / n_pw;
    double s = alpha * scale[b];
    dst[idx].x += s * src[idx].x;
    dst[idx].y += s * src[idx].y;
}

// a[i] *= b[i]  (element-wise complex multiply, in-place)
extern \"C\" __global__ void cpx_mul_inplace(
    double2* a, const double2* b, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        double ax = a[i].x, ay = a[i].y;
        double bx = b[i].x, by = b[i].y;
        a[i].x = ax * bx - ay * by;
        a[i].y = ax * by + ay * bx;
    }
}

// dst[i] = a[i] * conj(b[i])  (element-wise complex multiply with conjugate)
extern \"C\" __global__ void cpx_conj_mul(
    double2* dst, const double2* a, const double2* b, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        double ax = a[i].x, ay = a[i].y;
        double bx = b[i].x, by = b[i].y;
        dst[i].x = ax * bx + ay * by;   // Re(a * conj(b))
        dst[i].y = ay * bx - ax * by;   // Im(a * conj(b))
    }
}
";

// ---------------------------------------------------------------------------
// Compiled kernels handle
// ---------------------------------------------------------------------------

/// Handles to all compiled CUDA kernels used in the Chebyshev filter.
#[doc(hidden)]
pub struct CudaKernelSet {
    pub(crate) zero_buffer: CudaFunction,
    #[allow(dead_code)]
    pub(crate) zero_buffer_real: CudaFunction, // reserved for future real-buffer clearing
    pub(crate) init_kinetic: CudaFunction,
    pub(crate) scatter_pw_to_grid: CudaFunction,
    pub(crate) accumulate_density: CudaFunction,
    pub(crate) veff_multiply: CudaFunction,
    pub(crate) gather_add_kinetic: CudaFunction,
    #[allow(dead_code)]
    pub(crate) transpose_col_to_row: CudaFunction,
    #[allow(dead_code)]
    pub(crate) transpose_row_to_col: CudaFunction,
    pub(crate) band_scale_axpy: CudaFunction,
    pub(crate) cpx_mul_inplace: CudaFunction,
    #[allow(dead_code)]
    pub(crate) cpx_conj_mul: CudaFunction,
}

impl CudaKernelSet {
    #[doc(hidden)]
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Error> {
        let ptx = compile_ptx(CUDA_KERNEL_SRC).map_err(|e| Error::Nvrtc(e.to_string()))?;
        let module: Arc<CudaModule> = ctx.load_module(ptx).map_err(Error::Cuda)?;
        let load = |name: &str| -> Result<CudaFunction, Error> {
            module.load_function(name).map_err(Error::Cuda)
        };
        Ok(Self {
            zero_buffer: load("zero_buffer")?,
            zero_buffer_real: load("zero_buffer_real")?,
            init_kinetic: load("init_kinetic")?,
            scatter_pw_to_grid: load("scatter_pw_to_grid")?,
            accumulate_density: load("accumulate_density")?,
            veff_multiply: load("veff_multiply")?,
            gather_add_kinetic: load("gather_add_kinetic")?,
            transpose_col_to_row: load("transpose_col_to_row")?,
            transpose_row_to_col: load("transpose_row_to_col")?,
            band_scale_axpy: load("band_scale_axpy")?,
            cpx_mul_inplace: load("cpx_mul_inplace")?,
            cpx_conj_mul: load("cpx_conj_mul")?,
        })
    }
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
    kinetic_dev: &CudaSlice<f64>,
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
        ngx as i32, ngy as i32, ngz as i32, 1, stream.clone(),
    )?;

    // Working buffers: v (current), v_prev (previous), Hv
    let mut v_cur: CudaSlice<CudaComplex> = stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    let mut v_prev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    let mut hv: CudaSlice<CudaComplex> = stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
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
        stream.memcpy_dtod(&rand_dev, &mut v_cur).map_err(Error::Cuda)?;
    }

    // Normalise v_cur
    let norm0 = {
        let dot = blas.dotc_c64(n, &v_cur, 1, &v_cur, 1).map_err(Error::Blas)?;
        dot.x.sqrt()
    };
    #[cfg(feature = "scf_diag")]
    eprintln!("[Lanczos] entry: n_pw={} grid_size={} k_steps={}", n_pw, grid_size, k_steps);
    #[cfg(feature = "scf_diag")]
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
            apply_full_hamiltonian(
                &v_cur, v_eff_dev, kinetic_dev, fft_idx_dev,
                n_pw, 1, grid_size, inv_ntotal,
                &plan1, &mut hv, &mut grid1, vnl_data, blas, kernels, stream,
            )?;
            // Apply S⁻¹·H (global Woodbury) — wires S⁻¹ into the Lanczos
            // estimator so b_up reflects the preconditioned spectrum.
            apply_s_inverse(
                &mut hv, vnl_data, 1, n_pw as i32, blas, stream, solver,
            )?;
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
        stream.memcpy_dtod(&v_cur, &mut v_prev).map_err(Error::Cuda)?;
        let inv_b = CudaComplex { x: 1.0 / beta[j + 1], y: 0.0 };
        stream.memcpy_dtod(&hv, &mut v_cur).map_err(Error::Cuda)?;
        unsafe {
            let (ptr, _) = v_cur.device_ptr_mut(stream);
            cudarc::cublas::sys::cublasZscal_v2(blas.raw_handle(), n, &inv_b as *const _ as *const _, ptr as *mut _, 1)
                .result().map_err(Error::Blas)?;
        }
    }

    // Find λ_max and λ_min of the k×k symmetric tridiagonal T_k on CPU
    // via Gershgorin bounds. T_k is at most 6×6 so this is trivial.
    let k = alpha.len();
    #[cfg(feature = "scf_diag")]
    eprintln!("[Lanczos] alpha = {:?}", alpha);
    #[cfg(feature = "scf_diag")]
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
    #[cfg(feature = "scf_diag")]
    eprintln!(
        "[Lanczos] T_k bounds: lambda_min_tk = {:.4} Ha  lambda_max_tk = {:.4} Ha  residual_norm(beta[k-1]) = {:.4e}  → b_up_raw = {:.4} Ha",
        lambda_min_tk, lambda_max_tk, residual_norm, b_up_raw,
    );
    Ok((b_up_raw, lambda_min_tk, lambda_max_tk))
}

// ---------------------------------------------------------------------------
// Precomputed FFT metadata (uploaded to GPU)
// ---------------------------------------------------------------------------

/// Kinetic energy ½|G|² for each plane-wave (Hartree atomic units).
///
/// Computed directly from the fractional G-vectors (from `pw_coords`) rather than
/// from the full grid `g2()`, because the kernel `init_kinetic` indexes by
/// plane-wave index (0..n_pw), not by grid position.
fn compute_kinetic_energies(
    pw_coords: &[[i32; 3]],
    recip_lattice: &chemrust_hamiltonian_core::RecipLattice,
) -> KineticEnergies {
    let r = recip_lattice.as_array();
    let ke: Vec<f64> = pw_coords
        .iter()
        .map(|&[h, k, l]| {
            let hf = h as f64;
            let kf = k as f64;
            let lf = l as f64;
            let gx = hf * r[0][0] + kf * r[1][0] + lf * r[2][0];
            let gy = hf * r[0][1] + kf * r[1][1] + lf * r[2][1];
            let gz = hf * r[0][2] + kf * r[1][2] + lf * r[2][2];
            0.5 * (gx * gx + gy * gy + gz * gz)
        })
        .collect();
    KineticEnergies(ke)
}

// ---------------------------------------------------------------------------
// Full Hamiltonian application (T + V_loc on GPU)
// ---------------------------------------------------------------------------

/// Compute (T + V_loc)|psi> on GPU using FFT-based approach.
///
/// Steps:
/// 1. hpsi = T|psi>  (init_kinetic kernel)
/// 2. Scatter psi coefficients to FFT grid
/// 3. Batched C2C IFFT
/// 4. V_eff multiply (pointwise)
/// 5. Batched C2C FFT
/// 6. Gather + add to hpsi: hpsi += grid / N_total
#[allow(clippy::too_many_arguments)]
unsafe fn apply_v_loc_hamiltonian(
    psi_dev: &CudaSlice<CudaComplex>,
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    grid_dev: &mut CudaSlice<CudaComplex>,
    kinetic_dev: &CudaSlice<f64>,
    fft_idx_dev: &CudaSlice<i32>,
    v_eff_dev: &CudaSlice<f64>,
    n_pw: i32,
    n_bands: i32,
    grid_size: i32,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    // 1. hpsi = kinetic * psi  (T|psi>)
    unsafe {
        stream
            .launch_builder(&kernels.init_kinetic)
            .arg(&mut *hpsi_dev)
            .arg(psi_dev)
            .arg(kinetic_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;

    // 2. Zero grid, then scatter psi to FFT grid positions
    unsafe {
        stream
            .launch_builder(&kernels.zero_buffer)
            .arg(&mut *grid_dev)
            .arg(&(n_bands * grid_size))
            .launch(LaunchConfig::for_num_elems((n_bands * grid_size) as u32))
    }
    .map_err(Error::Cuda)?;

    unsafe {
        stream
            .launch_builder(&kernels.scatter_pw_to_grid)
            .arg(psi_dev)
            .arg(fft_idx_dev)
            .arg(&mut *grid_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .arg(&grid_size)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;

    // 3. Batched C2C IFFT (in-place)
    unsafe { c2c_inverse_inplace(fft_plan, grid_dev)?; }

    // 4. V_eff multiply: grid *= V_eff
    unsafe {
        stream
            .launch_builder(&kernels.veff_multiply)
            .arg(&mut *grid_dev)
            .arg(v_eff_dev)
            .arg(&grid_size)
            .arg(&n_bands)
            .launch(LaunchConfig::for_num_elems((n_bands * grid_size) as u32))
    }
    .map_err(Error::Cuda)?;

    // 5. Batched C2C FFT (in-place)
    unsafe { c2c_forward_inplace(fft_plan, grid_dev)?; }

    // 6. Gather: hpsi += grid / N_total
    // grid is const (read-only), hpsi is mutable (read-write for accumulation)
    unsafe {
        stream
            .launch_builder(&kernels.gather_add_kinetic)
            .arg(&*grid_dev)
            .arg(fft_idx_dev)
            .arg(&mut *hpsi_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .arg(&grid_size)
            .arg(&inv_ntotal)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;

    Ok(())
}

/// Apply the full Hamiltonian H|psi>. For Phase 2 this includes T + V_loc
/// Apply the full Hamiltonian H|psi>. Includes T + V_loc (FFT-based)
/// and V_NL (non-local pseudopotential via cuBLAS gemm).
#[allow(clippy::too_many_arguments)]
unsafe fn apply_full_hamiltonian(
    psi_dev: &CudaSlice<CudaComplex>,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &CudaSlice<f64>,
    fft_idx_dev: &CudaSlice<i32>,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    grid_dev: &mut CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    unsafe {
        apply_v_loc_hamiltonian(
            psi_dev, hpsi_dev, grid_dev,
            kinetic_dev, fft_idx_dev, v_eff_dev,
            n_pw as i32, n_bands as i32, grid_size as i32, inv_ntotal,
            fft_plan, kernels, stream,
        )?;

        apply_v_nl_hamiltonian(
            psi_dev, hpsi_dev, vnl_data,
            n_bands as i32, n_pw as i32,
            blas, stream,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// V_NL (non-local pseudopotential) via cuBLAS gemm
// ---------------------------------------------------------------------------

/// Apply V_NL|psi> and accumulate into hpsi for one batch of ion projectors.
///
/// For each ion's (beta_g, d_matrix, n_expanded):
///   C_proj = beta^H . psi     (n_expanded x n_bands)
///   C_proj = D . C_proj       (n_expanded x n_bands)
///   hpsi   += beta . C_proj   (n_pw x n_bands, accumulated)
#[allow(clippy::too_many_arguments)]
unsafe fn apply_v_nl_hamiltonian(
    psi_dev: &CudaSlice<CudaComplex>,
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    n_bands: i32,
    n_pw: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // C_proj = beta^H . psi  (n_expanded x n_bands)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros((ne * n_bands) as usize).map_err(Error::Cuda)?;

        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::C, // conj(beta^T)
                    transb: blas::op::N,
                    m: ne,
                    n: n_bands,
                    k: n_pw,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw, // beta_g is (ne, n_pw) row-major = col-major (n_pw, ne)
                    ldb: n_pw, // psi is (n_pw, n_bands) col-major
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_dev,
                &mut c_proj,
            )?;
        }

        // C_proj = D . C_proj  (n_expanded x n_bands)
        // Use a temp buffer since in-place gemm is not supported.
        let mut c_temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros((ne * n_bands) as usize).map_err(Error::Cuda)?;

        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::N,
                    transb: blas::op::N,
                    m: ne,
                    n: n_bands,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.d_matrix,
                &c_proj,
                &mut c_temp,
            )?;
        }
        std::mem::swap(&mut c_proj, &mut c_temp);

        // V_NL += beta . C_proj  (n_pw x n_bands, accumulated into hpsi)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::N,
                    transb: blas::op::N,
                    m: n_pw,
                    n: n_bands,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 }, // accumulate into hpsi
                    ldc: n_pw,
                },
                &entry.beta_g,
                &c_proj,
                hpsi_dev,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// S⁻¹ (USPP overlap inverse via Woodbury)
// ---------------------------------------------------------------------------

/// Apply the USPP overlap inverse to each band of `hpsi`.
///
/// Uses the global Woodbury formula (PHASE_PLAN.md):
///   S⁻¹·v = v − B · M⁻¹ · (B^H · v)
///   where M = Q⁻¹ + B^H·B (Cholesky-factored in VnlBatchData::precompute).
///
/// Preserved as dead code: used by `check_s_inv_s_identity` diagnostic and
/// potential TASK-D4 fallback (restore S⁻¹ in Step 4 only). Do not reap.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
unsafe fn apply_s_inverse(
    hpsi_dev: &mut CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    n_bands: i32,
    n_pw: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    solver: &SolverHandle,
) -> Result<(), Error> {
    let nte = vnl_data.n_total_expanded;

    // 1. temp = B^H · hpsi  (nte × n_bands)
    let mut temp: CudaSlice<CudaComplex> =
        stream.alloc_zeros(nte as usize * n_bands as usize).map_err(Error::Cuda)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: blas::op::C,
                transb: blas::op::N,
                m: nte,
                n: n_bands,
                k: n_pw,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw,
                ldb: n_pw,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: nte,
            },
            &vnl_data.b_concat,
            hpsi_dev,
            &mut temp,
        )?;
    }

    // 2. Solve M·x = temp via LU factor (zgetrs, in-place overwrites temp).
    let mut info_dev = stream.alloc_zeros::<i32>(1).map_err(Error::Cuda)?;
    solver.zgetrs(
        cublasOperation_t::CUBLAS_OP_N,
        nte,
        n_bands,
        &vnl_data.lu_m,
        &vnl_data.lu_ipiv,
        &mut temp,
        &mut info_dev,
    )?;

    // 3. hpsi −= B · x  (accumulate with α = −1)
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: blas::op::N,
                transb: blas::op::N,
                m: n_pw,
                n: n_bands,
                k: nte,
                alpha: CudaComplex { x: -1.0, y: 0.0 },
                lda: n_pw,
                ldb: nte,
                beta: CudaComplex { x: 1.0, y: 0.0 },
                ldc: n_pw,
            },
            &vnl_data.b_concat,
            &temp,
            hpsi_dev,
        )?;
    }

    Ok(())
}

/// Apply the USPP overlap matrix `S` to each band of `psi`.
///
/// S = I + Σ_I β_I · Q_I · β_I^H
///
/// For each ion with projectors `beta_g` and `q_matrix`:
///   p = beta_g^H · psi          (project, ne × n_bands)
///   q = q_matrix · p            (expand, ne × n_bands)
///   spsi += beta_g · q          (accumulate, n_pw × n_bands, α = +1)
///
/// The caller is responsible for copying `psi_dev` into `spsi_dev` first
/// (the identity term) before calling this to accumulate the β·Q·β^H·ψ correction.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_s_times(
    psi_dev: &CudaSlice<CudaComplex>,     // input ψ (n_pw × n_bands, col-major)
    spsi_dev: &mut CudaSlice<CudaComplex>, // output S·ψ (caller pre-copies psi into this)
    vnl_data: &VnlBatchData,
    n_bands: i32,
    n_pw: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // p = beta_g^H · psi  (n_expanded × n_bands)
        let mut p: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::C,
                    transb: blas::op::N,
                    m: ne,
                    n: n_bands,
                    k: n_pw,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw,
                    ldb: n_pw,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_dev,
                &mut p,
            )?;
        }

        // q = q_matrix · p  (n_expanded × n_bands)
        let mut q: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * n_bands as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::N,
                    transb: blas::op::N,
                    m: ne,
                    n: n_bands,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.q_matrix,
                &p,
                &mut q,
            )?;
        }

        // spsi += beta_g · q  (accumulate with α = +1)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: blas::op::N,
                    transb: blas::op::N,
                    m: n_pw,
                    n: n_bands,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 },
                    ldc: n_pw,
                },
                &entry.beta_g,
                &q,
                spsi_dev,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// S⁻¹·S identity diagnostic
// ---------------------------------------------------------------------------

#[doc(hidden)]
pub fn check_s_inv_s_identity(
    psi_host: &[num_complex::Complex64],
    n_pw: usize,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    solver: &SolverHandle,
) -> Result<f64, Error> {
    use crate::device::blas::op;
    let n = n_pw as i32;
    let psi_cuda: Vec<CudaComplex> = psi_host
        .iter()
        .map(|&c| CudaComplex { x: c.re, y: c.im })
        .collect();
    let psi_dev: CudaSlice<CudaComplex> = stream
        .clone_htod(&psi_cuda).map_err(Error::Cuda)?;

    // 1. Build S·psi = psi + Σ β_g · q_matrix · (β_g^H · psi)
    let mut spsi_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    stream.memcpy_dtod(&psi_dev, &mut spsi_dev).map_err(Error::Cuda)?;

    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // c_proj = β_g^H · psi  (ne × 1)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemv_c64(
                op::C, n, ne,
                CudaComplex { x: 1.0, y: 0.0 },
                &entry.beta_g, n,
                &psi_dev, 1,
                CudaComplex { x: 0.0, y: 0.0 },
                &mut c_proj, 1,
            ).map_err(Error::Blas)?;
        }

        // temp = q_matrix · c_proj  (ne × 1)
        let mut temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemv_c64(
                op::N, ne, ne,
                CudaComplex { x: 1.0, y: 0.0 },
                &entry.q_matrix, ne,
                &c_proj, 1,
                CudaComplex { x: 0.0, y: 0.0 },
                &mut temp, 1,
            ).map_err(Error::Blas)?;
        }

        // spsi += β_g · temp  (n_pw × 1)
        unsafe {
            blas.gemv_c64(
                op::N, n, ne,
                CudaComplex { x: 1.0, y: 0.0 },
                &entry.beta_g, n,
                &temp, 1,
                CudaComplex { x: 1.0, y: 0.0 },
                &mut spsi_dev, 1,
            ).map_err(Error::Blas)?;
        }
    }

    // 2. Apply S⁻¹ to spsi
    unsafe {
        apply_s_inverse(
            &mut spsi_dev, vnl_data, 1, n, blas, stream, solver,
        )?;
    }

    // 3. D2H and compute max residual ‖spsi − psi‖_∞
    let result: Vec<CudaComplex> = stream.clone_dtoh(&spsi_dev).map_err(Error::Cuda)?;
    let max_residual = psi_host.iter().zip(result.iter())
        .map(|(&p, &r)| {
            let dr = r.x - p.re;
            let di = r.y - p.im;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0_f64, f64::max);

    Ok(max_residual)
}

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
#[allow(clippy::too_many_arguments)]
pub(crate) fn chebyshev_filter(
    psi_gpu: &Gpu<WavefunctionSet<ColumnDistributed>>,
    v_eff_gpu: &Gpu<crate::types::EffectivePotential>,
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
    let kinetic_cpu = compute_kinetic_energies(pw_coords, wave_grid.recip_lattice());
    let kinetic_dev: CudaSlice<f64> =
        stream.clone_htod(&kinetic_cpu.0).map_err(Error::Cuda)?;
    pcie.h2d_bytes += kinetic_cpu.0.len() * std::mem::size_of::<f64>();

    // ---- FFT plan (batched C2C) ----
    // cuFFT uses row-major layout: n[0] is slowest-varying (outermost),
    // n[rank-1] is fastest-varying (innermost). Our scatter index formula
    // `iz + ngz*(iy + ngy*ix)` makes iz innermost, ix outermost; the
    // matching plan dims are `(ngx, ngy, ngz)`. Verified by the isolated
    // FFT test `cufft_dim_ordering_isolated_diagnostic` (only this
    // ordering reproduces the analytic exp(2πi·G·r) for a single δ in G).
    // Cubic grids are insensitive to this ordering; non-cubic grids are not.
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32, n_bands_i32, stream.clone(),
    )?;

    // ---- GPU workspace buffers ----
    let v_eff_dev = v_eff_gpu.as_device_slice();
    let psi_input = psi_gpu.as_device_slice().clone();

    // R-ChFSI buffers (Algorithm 3):
    // buf_y = Y = H·X − S·X·Λ, buf_sx = S·X for residual, buf_rx = R_X, buf_ry = R_Y
    // buf_c: reused for R_new computation then swap with buf_ry
    // buf_a: reused for X_new reconstruction at Step 4
    let mut buf_y: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut buf_sx: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut buf_rx: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut buf_ry: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut buf_c: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut buf_a: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;

    // Hamiltonian workspace
    let mut hpsi_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut grid_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // Output RowDistributed buffers
    let mut psi_row_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut hpsi_row_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;

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

    let final_psi_buf: &mut CudaSlice<CudaComplex>;

    if ndeg == 0 {
        // No filtering: use input wavefunctions as-is
        stream.memcpy_dtod(&psi_input, &mut buf_a).map_err(Error::Cuda)?;
        final_psi_buf = &mut buf_a;
    } else {
        // ------------------------------------------------------------
        // Step 1: Initial residual Y = H·X − S·X·Λ
        // ------------------------------------------------------------
        // hpsi_dev = H·psi_input
        unsafe {
            apply_full_hamiltonian(
                &psi_input, v_eff_dev, &kinetic_dev, fft_idx_dev,
                n_pw, n_bands, grid_size, inv_ntotal,
                &fft_plan, &mut hpsi_dev, &mut grid_dev, vnl_data, blas, kernels, stream,
            )?;
        }

        // buf_y = hpsi_dev (copy, keeping hpsi_dev intact for diagnostics)
        stream.memcpy_dtod(&hpsi_dev, &mut buf_y).map_err(Error::Cuda)?;

        if let Some(eig) = eigenvalues {
            // buf_sx = S·psi_input
            stream.memcpy_dtod(&psi_input, &mut buf_sx).map_err(Error::Cuda)?;
            unsafe {
                apply_s_times(&psi_input, &mut buf_sx, vnl_data, n_bands_i32, n_pw_i32, blas, stream)?;
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
        stream.memcpy_dtod(&buf_y, &mut buf_ry).map_err(Error::Cuda)?;
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
        let mut lam_y: Vec<f64> = if eigenvalues.is_some() || matches!(filter_mode, FilterMode::SinvHFullDas) {
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

            // H·R_Y then optionally S⁻¹·H·R_Y depending on filter mode
            unsafe {
                apply_full_hamiltonian(
                    &buf_ry, v_eff_dev, &kinetic_dev, fft_idx_dev,
                    n_pw, n_bands, grid_size, inv_ntotal,
                    &fft_plan, &mut hpsi_dev, &mut grid_dev, vnl_data, blas, kernels, stream,
                )?;
                if matches!(filter_mode, FilterMode::SinvHKeepHEig | FilterMode::SinvHFullDas) {
                    apply_s_inverse(
                        &mut hpsi_dev, vnl_data, n_bands_i32, n_pw_i32, blas, stream, solver,
                    )?;
                }
            }

            // R_new = (2σ₂/e)·H·R_Y − (2σ₂/e)·c·R_Y − σ·σ₂·R_X + (2σ₂/e)·Y·Λ_Y
            // First: buf_c = coeff * H·R_Y
            stream.memcpy_dtod(&hpsi_dev, &mut buf_c).map_err(Error::Cuda)?;
            {
                let alpha_cf = CudaComplex { x: coeff, y: 0.0 };
                unsafe {
                    let (ptr, _) = buf_c.device_ptr_mut(stream);
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
            let has_eig = eigenvalues.is_some() || matches!(filter_mode, FilterMode::SinvHFullDas);
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
        stream.memcpy_dtod(&buf_ry, &mut buf_a).map_err(Error::Cuda)?;
        if matches!(filter_mode, FilterMode::SinvHFullDas) {
            unsafe {
                apply_s_inverse(&mut buf_a, vnl_data, n_bands_i32, n_pw_i32, blas, stream, solver)?;
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
    let mut gs_col: CudaSlice<CudaComplex> = stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    let mut gs_s_col: CudaSlice<CudaComplex> = stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    unsafe {
        let (psi_ptr, _) = final_psi_buf.device_ptr_mut(stream);
        let (gs_col_ptr, _) = gs_col.device_ptr_mut(stream);
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
                stream.memcpy_dtod(&gs_col, &mut gs_s_col).map_err(Error::Cuda)?;
                apply_s_times(&gs_col, &mut gs_s_col, vnl_data, 1, n_pw_i32, blas, stream)?;
                // Get device pointer from gs_s_col after mutable ops complete
                let (gs_s_col_ptr, _) = gs_s_col.device_ptr_mut(stream);
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
        apply_full_hamiltonian(
            final_psi_buf, v_eff_dev, &kinetic_dev, fft_idx_dev,
            n_pw, n_bands, grid_size, inv_ntotal,
            &fft_plan, &mut hpsi_dev, &mut grid_dev, vnl_data, blas, kernels, stream,
        )?;
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
    stream.memcpy_dtod(final_psi_buf, &mut psi_row_dev).map_err(Error::Cuda)?;
    stream.memcpy_dtod(&hpsi_dev, &mut hpsi_row_dev).map_err(Error::Cuda)?;

    // The Gram-Schmidt step above already orthonormalized the bands, so
    // S_sub = ψ†ψ ≈ I and ZHEGVD is well-conditioned. No further per-band
    // scaling needed.

    // Wrap into Gpu<WavefunctionSet<L>>
    let psi_row = Gpu::<WavefunctionSet<RowDistributed>> {
        slice: psi_row_dev,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };
    let hpsi_row = Gpu::<WavefunctionSet<RowDistributed>> {
        slice: hpsi_row_dev,
        shape: vec![n_bands, n_pw],
        ctx: ctx.clone(),
        _marker: PhantomData,
    };

    Ok((psi_row, hpsi_row))
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
    v_eff_gpu: &Gpu<crate::types::EffectivePotential>,
    wave_grid: &GVectorGrid,
    pw_coords: &[[i32; 3]],
    vnl_data: &VnlBatchData,
    fft_idx_dev: &CudaSlice<i32>,
    kernels: &CudaKernelSet,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
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

    let kinetic_cpu = compute_kinetic_energies(pw_coords, wave_grid.recip_lattice());
    let kinetic_dev: CudaSlice<f64> = stream.clone_htod(&kinetic_cpu.0).map_err(Error::Cuda)?;

    // Plan with the cuFFT-correct dim ordering (matches chebyshev_filter at line 844).
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32, n_bands_i32, stream.clone(),
    )?;

    let v_eff_dev = v_eff_gpu.as_device_slice();
    let psi_input = psi_gpu.as_device_slice();

    let mut grid_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // Component 1: kinetic only. Run init_kinetic by itself.
    let mut hpsi_t: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    unsafe {
        stream
            .launch_builder(&kernels.init_kinetic)
            .arg(&mut hpsi_t)
            .arg(psi_input)
            .arg(&kinetic_dev)
            .arg(&n_pw_i32)
            .arg(&n_bands_i32)
            .launch(LaunchConfig::for_num_elems((n_bands_i32 * n_pw_i32) as u32))
    }
    .map_err(Error::Cuda)?;

    // Component 2: kinetic + V_loc (full apply_v_loc_hamiltonian).
    let mut hpsi_tv: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    unsafe {
        apply_v_loc_hamiltonian(
            psi_input, &mut hpsi_tv, &mut grid_dev,
            &kinetic_dev, fft_idx_dev, v_eff_dev,
            n_pw_i32, n_bands_i32, grid_size as i32, inv_ntotal,
            &fft_plan, kernels, stream,
        )?;
    }

    // Component 3: kinetic + V_loc + V_NL (full apply_full_hamiltonian).
    let mut hpsi_full: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    unsafe {
        apply_v_loc_hamiltonian(
            psi_input, &mut hpsi_full, &mut grid_dev,
            &kinetic_dev, fft_idx_dev, v_eff_dev,
            n_pw_i32, n_bands_i32, grid_size as i32, inv_ntotal,
            &fft_plan, kernels, stream,
        )?;
        apply_v_nl_hamiltonian(
            psi_input, &mut hpsi_full, vnl_data,
            n_bands_i32, n_pw_i32, blas, stream,
        )?;
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
        hpsi_tv: to_complex(stream.clone_dtoh(&hpsi_tv).map_err(Error::Cuda)?),
        hpsi_full: to_complex(stream.clone_dtoh(&hpsi_full).map_err(Error::Cuda)?),
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

    unsafe {
        apply_s_times(
            &psi_dev,
            &mut spsi_dev,
            vnl_data,
            n_bands as i32,
            n_pw as i32,
            blas,
            stream,
        )?;
    }
    stream.synchronize()?;
    let spsi_raw = stream.clone_dtoh(&spsi_dev).map_err(Error::Cuda)?;
    Ok(spsi_raw
        .into_iter()
        .map(|c| num_complex::Complex64::new(c.x, c.y))
        .collect())
}
