// ---------------------------------------------------------------------------
// Hamiltonian and USPP overlap operators (GPU)
// ---------------------------------------------------------------------------
//
// Implements:
//   1. c2c_inverse_inplace / c2c_forward_inplace — C2C FFT wrappers
//   2. apply_v_loc_hamiltonian — T + V_loc via FFT round-trip
//   3. apply_v_nl_hamiltonian — V_NL via cuBLAS gemm with β-projectors
//   4. apply_full_hamiltonian — composes V_loc + V_NL
//   5. apply_s_times — S·ψ = ψ + β·Q·β^H·ψ (USPP overlap)

use std::sync::Arc;

#[cfg(feature = "scf_diag")]
use std::time::Instant;

use cudarc::cusolver::sys::cublasOperation_t;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::device::blas::{self, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::{
    KineticPreconditioner, PwCoefficients,
};
use crate::eigensolver::beta_phi_cache::BetaPhiCache;
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;
use bon::builder;

// ---------------------------------------------------------------------------
// Helper: call cuFFT C2C in-place (same buffer for input and output)
// ---------------------------------------------------------------------------

/// Call `c2c_inverse` in-place. cuFFT natively supports in-place transforms,
/// so passing the same `&mut` twice via raw pointer is correct.
unsafe fn c2c_inverse_inplace(
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

// ---------------------------------------------------------------------------
// Full Hamiltonian application (T + V_loc on GPU)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Profiling support (scf_diag feature)
// ---------------------------------------------------------------------------
// Per-operation GPU timings for apply_v_loc_hamiltonian, accumulated into a
// thread-local so that apply_full_hamiltonian can print a combined summary.
#[cfg(feature = "scf_diag")]
mod profile {
    use std::cell::RefCell;

    /// Per-operation timing record from one apply_v_loc_hamiltonian call.
    pub(super) struct VLocProfile {
        pub times: [(&'static str, f64); 7], // (label, seconds)
        pub total_s: f64,
        pub n_pw: usize,
        pub n_bands: usize,
        pub ngx: usize,
        pub ngy: usize,
        pub ngz: usize,
    }

    thread_local! {
        pub(super) static V_LOC_PROFILE: RefCell<Option<VLocProfile>> = const { RefCell::new(None) };
    }
}
#[cfg(feature = "scf_diag")]
use profile::V_LOC_PROFILE;

/// Compute (T + V_loc)|psi> on GPU using FFT-based approach.
///
/// Steps:
/// 1. hpsi = T|psi>  (init_kinetic kernel)
/// 2. Scatter psi coefficients to FFT grid
/// 3. Batched C2C IFFT
/// 4. V_eff multiply (pointwise)
/// 5. Batched C2C FFT
/// 6. Gather + add to hpsi: hpsi += grid / N_total
#[builder]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_v_loc_hamiltonian(
    psi_dev: &PwCoefficients,
    hpsi_dev: &mut PwCoefficients,
    grid_dev: &mut CudaSlice<CudaComplex>,
    kinetic_dev: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    v_eff_dev: &CudaSlice<f64>,
    n_pw: i32,
    n_bands: i32,
    grid_size: i32,
    ngx: i32,
    ngy: i32,
    ngz: i32,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
    _blas: Option<&BlasHandle>,
) -> Result<(), Error> {
    // ── Profiling: per-operation GPU timings (scf_diag feature) ──────
    // Each operation is measured with stream sync before/after to capture
    // true GPU execution time.  Sync overhead (~50 µs each) is acceptable
    // for profiling; the feature is off in production builds.
    #[cfg(feature = "scf_diag")]
    let mut _p_times: [(&str, f64); 7] = [("", 0.0); 7];
    #[cfg(feature = "scf_diag")]
    let _p_total_begin = Instant::now();

    // 1. hpsi = kinetic * psi  (T|psi>)
    #[cfg(feature = "scf_diag")]
    { stream.synchronize().map_err(Error::Cuda)?; }
    #[cfg(feature = "scf_diag")]
    let _t1 = Instant::now();
    unsafe {
        stream
            .launch_builder(&kernels.init_kinetic)
            .arg(&mut **hpsi_dev)
            .arg(&**psi_dev)
            .arg(&**kinetic_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    #[cfg(feature = "scf_diag")]
    { stream.synchronize().map_err(Error::Cuda)?; _p_times[0] = ("init_kinetic", _t1.elapsed().as_secs_f64()); }

    // 2. Zero grid, then scatter psi to FFT grid positions
    #[cfg(feature = "scf_diag")]
    let _t2a = Instant::now();
    unsafe {
        stream
            .launch_builder(&kernels.zero_buffer)
            .arg(&mut *grid_dev)
            .arg(&(n_bands * grid_size))
            .launch(LaunchConfig::for_num_elems((n_bands * grid_size) as u32))
    }
    .map_err(Error::Cuda)?;
    #[cfg(feature = "scf_diag")]
    { stream.synchronize().map_err(Error::Cuda)?; _p_times[1] = ("zero_buffer", _t2a.elapsed().as_secs_f64()); }

    // Nyquist: -1 if odd-sized (no Nyquist plane), N/2 if even.
    let nyq_x = if ngx % 2 == 0 { ngx / 2 } else { -1 };
    let nyq_y = if ngy % 2 == 0 { ngy / 2 } else { -1 };
    let nyq_z = if ngz % 2 == 0 { ngz / 2 } else { -1 };

    #[cfg(feature = "scf_diag")]
    let _t2b = Instant::now();
    unsafe {
        stream
            .launch_builder(&kernels.scatter_pw_to_grid_nyq)
            .arg(&**psi_dev)
            .arg(fft_idx_dev)
            .arg(&mut *grid_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .arg(&grid_size)
            .arg(&ngy)
            .arg(&ngz)
            .arg(&nyq_x)
            .arg(&nyq_y)
            .arg(&nyq_z)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    #[cfg(feature = "scf_diag")]
    { stream.synchronize().map_err(Error::Cuda)?; _p_times[2] = ("scatter_pw", _t2b.elapsed().as_secs_f64()); }

    // 3. Batched C2C IFFT (in-place)
    #[cfg(feature = "scf_diag")]
    let _t3 = Instant::now();
    unsafe { c2c_inverse_inplace(fft_plan, grid_dev)?; }
    #[cfg(feature = "scf_diag")]
    { stream.synchronize().map_err(Error::Cuda)?; _p_times[3] = ("cuFFT IFFT", _t3.elapsed().as_secs_f64()); }

    // 4. V_eff multiply: grid *= V_eff
    #[cfg(feature = "scf_diag")]
    let _t4 = Instant::now();
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
    #[cfg(feature = "scf_diag")]
    { stream.synchronize().map_err(Error::Cuda)?; _p_times[4] = ("veff_multiply", _t4.elapsed().as_secs_f64()); }

    // 5. Batched C2C FFT (in-place)
    #[cfg(feature = "scf_diag")]
    let _t5 = Instant::now();
    unsafe { c2c_forward_inplace(fft_plan, grid_dev)?; }
    #[cfg(feature = "scf_diag")]
    { stream.synchronize().map_err(Error::Cuda)?; _p_times[5] = ("cuFFT FFT", _t5.elapsed().as_secs_f64()); }

    // 6. Gather: hpsi += grid / N_total
    // grid is const (read-only), hpsi is mutable (read-write for accumulation)
    #[cfg(feature = "scf_diag")]
    let _t6 = Instant::now();
    unsafe {
        stream
            .launch_builder(&kernels.gather_add_kinetic)
            .arg(&*grid_dev)
            .arg(fft_idx_dev)
            .arg(&mut **hpsi_dev)
            .arg(&n_pw)
            .arg(&n_bands)
            .arg(&grid_size)
            .arg(&inv_ntotal)
            .launch(LaunchConfig::for_num_elems((n_bands * n_pw) as u32))
    }
    .map_err(Error::Cuda)?;
    #[cfg(feature = "scf_diag")]
    {
        stream.synchronize().map_err(Error::Cuda)?;
        _p_times[6] = ("gather_add", _t6.elapsed().as_secs_f64());
        let vloc_total = _p_total_begin.elapsed().as_secs_f64();
        // Store timings in thread-local for apply_full_hamiltonian to print
        V_LOC_PROFILE.with(|cell| {
            cell.replace(Some(profile::VLocProfile {
                times: _p_times,
                total_s: vloc_total,
                n_pw: n_pw as usize,
                n_bands: n_bands as usize,
                ngx: ngx as usize,
                ngy: ngy as usize,
                ngz: ngz as usize,
            }));
        });
    }
    Ok(())
}

/// Apply the full Hamiltonian H|psi>. For Phase 2 this includes T + V_loc
/// Apply the full Hamiltonian H|psi>. Includes T + V_loc (FFT-based)
/// and V_NL (non-local pseudopotential via cuBLAS gemm).
#[builder]
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_full_hamiltonian(
    psi_dev: &PwCoefficients,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    hpsi_dev: &mut PwCoefficients,
    grid_dev: &mut CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
    mut maybe_beta_phi_cache: Option<&mut BetaPhiCache>,
) -> Result<(), Error> {
    #[cfg(feature = "scf_diag")]
    let _p_h_total_begin = Instant::now();

    unsafe {
        apply_v_loc_hamiltonian()
            .psi_dev(psi_dev)
            .hpsi_dev(hpsi_dev)
            .grid_dev(grid_dev)
            .kinetic_dev(kinetic_dev)
            .fft_idx_dev(fft_idx_dev)
            .v_eff_dev(v_eff_dev)
            .n_pw(n_pw as i32)
            .n_bands(n_bands as i32)
            .grid_size(grid_size as i32)
            .ngx(fft_plan.nz()) // plan created as (ngz, ngy, ngx): nz=ngx
            .ngy(fft_plan.ny())
            .ngz(fft_plan.nx()) // plan created as (ngz, ngy, ngx): nx=ngz
            .inv_ntotal(inv_ntotal)
            .fft_plan(fft_plan)
            .kernels(kernels)
            .stream(stream)
            .maybe_blas(Some(blas))
            .call()?;

    #[cfg(feature = "scf_diag")]
    let _p_vnl_begin = Instant::now();
    {
        // bon::builder unwraps Option<T> — the setter takes T, not Option<T>.
        // Conditionally attach the cache so the default (None) is used when
        // no cache is provided (e.g. for search-direction H applications).
        let vnl_builder = apply_v_nl_hamiltonian()
            .psi_dev(psi_dev)
            .hpsi_dev(hpsi_dev)
            .vnl_data(vnl_data)
            .n_bands(n_bands as i32)
            .n_pw(n_pw as i32)
            .blas(blas)
            .stream(stream)
            .maybe_kernels(kernels);
        if let Some(ref mut cache) = maybe_beta_phi_cache {
            vnl_builder.maybe_beta_phi_cache(cache).call()?;
        } else {
            vnl_builder.call()?;
        }
    }

    #[cfg(feature = "scf_diag")]
    {
        stream.synchronize().map_err(Error::Cuda)?;
        let vnl_elapsed = _p_vnl_begin.elapsed().as_secs_f64();
        let h_total = _p_h_total_begin.elapsed().as_secs_f64();

        // Print combined V_loc + V_NL profile
        V_LOC_PROFILE.with(|cell| {
            if let Some(ref p) = *cell.borrow() {
                eprintln!(
                    "[profile-Hpsi] ===== H·psi (npw={} nbands={} grid={}x{}x{}) =====",
                    p.n_pw, p.n_bands, p.ngx, p.ngy, p.ngz,
                );
                let mut vloc_sum = 0.0f64;
                for &(label, secs) in &p.times {
                    let ms = secs * 1000.0;
                    let pct = if p.total_s > 0.0 { 100.0 * secs / p.total_s } else { 0.0 };
                    eprintln!("[profile-Hpsi]   {:<20} {:>8.1} ms  ({:>5.1}%)",
                        label, ms, pct);
                    vloc_sum += secs;
                }
                let vloc_ms = vloc_sum * 1000.0;
                let vnl_ms = vnl_elapsed * 1000.0;
                let total_ms = h_total * 1000.0;
                eprintln!("[profile-Hpsi]   {:-<20} {:->8.1} ms", "", vloc_ms);
                eprintln!("[profile-Hpsi]   {:<20} {:>8.1} ms  (V_NL cuBLAS)", "V_NL", vnl_ms);
                eprintln!("[profile-Hpsi]   {:=<20} {:=>8.1} ms  total", "", total_ms);
                // FFT fraction — key metric for grid-decomposition decision
                let fft_frac = if h_total > 0.0 {
                    (p.times[3].1 + p.times[5].1) / h_total
                } else {
                    0.0
                };
                eprintln!(
                    "[profile-Hpsi]   FFT fraction: {:.1}% {}",
                    fft_frac * 100.0,
                    if fft_frac > 0.6 { "→ grid decomposition justified" }
                    else if fft_frac > 0.4 { "→ FFT + V_NL both significant" }
                    else { "→ V_NL dominates; optimize cuBLAS path" },
                );
            }
        });
    }

    // Diag: |hpsi|² for last band after full H — (scf_diag: API needs DevicePtr + cublas)
    }
    Ok(())
}
// V_NL (non-local pseudopotential) via cuBLAS gemm
// ---------------------------------------------------------------------------

/// Apply V_NL|psi> and accumulate into hpsi for one batch of ion projectors.
///
/// For each ion's (beta_g, d_matrix, n_expanded):
///   C_proj = beta^H . psi     (n_expanded x n_bands)
///   C_proj = D . C_proj       (n_expanded x n_bands)
///   hpsi   += beta . C_proj   (n_pw x n_bands, accumulated)
#[builder]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_v_nl_hamiltonian(
    psi_dev: &PwCoefficients,
    hpsi_dev: &mut PwCoefficients,
    vnl_data: &VnlBatchData,
    n_bands: i32,
    n_pw: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    maybe_beta_phi_cache: Option<&mut BetaPhiCache>,
    maybe_kernels: Option<&CudaKernelSet>,
) -> Result<(), Error> {
    // ── BetaPhiCache read path ──────────────────────────────────────────
    // Skips the β^H·ψ ZGEMM (~33% of V_NL time) when cached projections
    // from the previous outer iteration's compute_all() are still valid.
    //
    // Lifecycle (per Davidson outer iteration):
    //   Start:  cache fully valid from previous iter's compute_all()
    //   H·psi:  copy from cache → skip β^H·ψ ZGEMM (this function)
    //   A1:     invalidate_all() — full subspace rotation
    //   A3:     invalidate_bands(&modified) — block copy-back
    //   End:    compute_all() — populate for next iteration
    //
    // Only consumed when ALL bands valid.  Partial validity after A3 block
    // invalidation falls through to fresh compute.  This is correct because
    // the only H·psi call receiving the cache is at the START of each outer
    // iteration, where the cache is either fully valid (iter ≥ 1) or fully
    // invalid (iter 0, freshly allocated).
    //
    // CASTEP correspondence:  wave_beta_phi populates have_beta_phi(:,:);
    // subsequent V_NL calls skip β^H·ψ recomputation when flag is .true.
    // Our cache mirrors this with per-band bools + are_all_valid() gate.
    // ── BetaPhiCache read path — diagnostic run ────────────────────────
    //
    // Run 2653 proved cache data is bitwise-identical to fresh ZGEMM at
    // {:.3e} precision.  Stream ordering test PASSES.  Yet cache consumption
    // causes NiO SCF divergence.  This run uses {:.15e} precision to check
    // for sub-ULP differences that might accumulate across SCF iterations.
    let cache_all_valid = maybe_beta_phi_cache
        .as_ref()
        .map(|c| c.are_all_valid())
        .unwrap_or(false);

    #[cfg(feature = "scf_diag")]
    if cache_all_valid {
        eprintln!("[profile-Hpsi]   BetaPhiCache HIT — skipping β^H·ψ ZGEMM");
    }

    for (ion_idx, entry) in vnl_data.entries.iter().enumerate() {
        let ne = entry.n_expanded;

        // C_proj = beta^H . psi  (n_expanded × n_bands)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros((ne * n_bands) as usize).map_err(Error::Cuda)?;

        if cache_all_valid {
            // Copy cached β^H·ψ → c_proj using compute-engine kernel.
            // cudaMemcpyAsync (copy engine) has a cache-coherence gap with
            // subsequent cuBLAS ZGEMMs (compute engine) on Pascal GPUs.
            // A compute-engine copy kernel guarantees L1/L2 coherence.
            let cache = maybe_beta_phi_cache.as_ref().unwrap();
            let (cached, cached_ne) = cache
                .ion_projections(ion_idx)
                .expect("BetaPhiCache: ion index out of range");
            assert_eq!(
                cached_ne, ne,
                "BetaPhiCache: n_expanded mismatch ion {ion_idx}: cache={cached_ne} vnl={ne}"
            );
            let copy_n = (ne * n_bands) as usize;
            if let Some(kernels) = maybe_kernels {
                unsafe {
                    stream
                        .launch_builder(&kernels.copy_buffer)
                        .arg(&mut c_proj)
                        .arg(cached)
                        .arg(&(copy_n as i32))
                        .launch(LaunchConfig::for_num_elems(copy_n as u32))
                }
                .map_err(Error::Cuda)?;
            } else {
                stream
                    .memcpy_dtod(cached, &mut c_proj)
                    .map_err(Error::Cuda)?;
            }
        } else {
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
    } // end for ion_idx

    Ok(())
}

// ---------------------------------------------------------------------------
// S operator (USPP overlap)
// ---------------------------------------------------------------------------

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
#[builder]
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_s_times(
    psi_dev: &PwCoefficients,     // input ψ (n_pw × n_bands, col-major)
    spsi_dev: &mut PwCoefficients, // output S·ψ (caller pre-copies psi into this)
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
// S^{-1} operator (USPP preconditioned S-inverse via global Woodbury)
// ---------------------------------------------------------------------------
//
// Applies S^{-1} via the global Woodbury formula:
//   S^{-1} = I - B · M^{-1} · B^H
//
// where B = b_concat (concatenated beta-projectors, n_pw × n_total_expanded),
// and M = Q^{-1} + B^H·B + eps·I (LU-factored, nte × nte).
//
// Only available under the `chebyshev` feature flag.
#[cfg(feature = "chebyshev")]
#[builder]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_s_inverse(
    hpsi_dev: &mut PwCoefficients,
    vnl_data: &VnlBatchData,
    n_bands: i32,
    n_pw: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    solver: &SolverHandle,
) -> Result<(), Error> {
    let nte = vnl_data.n_total_expanded;

    if nte == 0 {
        return Ok(());  // No USPP ions, S = I, S^{-1} = I
    }

    // 1. temp = B^H . hpsi  (nte × n_bands)
    // n_pw rows of B^H, n_bands columns of hpsi
    let mut temp: CudaSlice<CudaComplex> =
        stream.alloc_zeros(nte as usize * n_bands as usize).map_err(Error::Cuda)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: blas::op::C,  // B^H (conjugate transpose of B)
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
            &**hpsi_dev,
            &mut temp,
        )?;
    }

    // 2. Solve M·y = temp via LU factor (zgetrs, in-place).
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

    // 3. Save original hpsi as h_in for iterative refinement
    let mut h_in: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw as usize * n_bands as usize).map_err(Error::Cuda)?;
    stream.memcpy_dtod(&**hpsi_dev, &mut h_in).map_err(Error::Cuda)?;

    // 4. Iterative refinement of y (3 iterations).
    //    Block-diagonal Q preconditioner stationary iteration:
    //      h_out = h_in - B·y
    //      bt_h  = B^H · h_out
    //      q_bt  = Q · bt_h       (per-ion block-diagonal)
    //      y_corr = y - q_bt
    //      r_h   = B · y_corr
    //      r_y   = B^H · r_h
    //      dy    = M^{-1} · r_y   (zgetrs)
    //      y    += dy
    //
    //    Equivalent to ABINIT's m_invovl.F90:1102-1140.
    {
        // Scratch buffers reused across iterations
        let mut bt_h: CudaSlice<CudaComplex> =
            stream.alloc_zeros(nte as usize * n_bands as usize).map_err(Error::Cuda)?;
        let mut h_out: CudaSlice<CudaComplex> =
            stream.alloc_zeros(n_pw as usize * n_bands as usize).map_err(Error::Cuda)?;

        let per_ion_ne = &vnl_data.shared.handle.per_ion_n_expanded;

        for _iter in 0..3 {
            // 4a. h_out = h_in - B·y   (n_pw × n_bands, GEMM with alpha=-1, beta=0)
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
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                        ldc: n_pw,
                    },
                    &vnl_data.b_concat,
                    &temp,
                    &mut h_out,
                )?;
            }
            // h_out += h_in  (AXPY)
            blas.axpy_c64(
                n_pw * n_bands,
                CudaComplex { x: 1.0, y: 0.0 },
                &h_in,
                1,
                &mut h_out,
                1,
            )?;

            // 4b. bt_h = B^H · h_out  (nte × n_bands)
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
                    &h_out,
                    &mut bt_h,
                )?;
            }

            // 4c. q_bt = Q · bt_h  (block-diagonal per-ion Q × bt_h slice)
            //     Allocate bt_q (nte × n_bands, initialized to zero).
            let mut bt_q: CudaSlice<CudaComplex> =
                stream.alloc_zeros(nte as usize * n_bands as usize).map_err(Error::Cuda)?;

            let mut offset: usize = 0;
            for (ion_idx, ion_ne) in per_ion_ne.iter().enumerate() {
                let ne = *ion_ne;
                if ne == 0 {
                    continue;
                }
                let ne_bands = ne as usize * n_bands as usize;

                // Allocate temp buffer for this ion's bt_h slice (ne × n_bands)
                let mut ion_bt_h: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(ne_bands).map_err(Error::Cuda)?;
                // Copy bt_h[offset..offset+ne_bands] into temp via CudaView-as-source
                {
                    let src_view = bt_h.slice(offset..offset + ne_bands);
                    stream.memcpy_dtod(&src_view, &mut ion_bt_h).map_err(Error::Cuda)?;
                }

                // Allocate result buffer
                let mut ion_q_bt: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(ne_bands).map_err(Error::Cuda)?;

                // Q_ion (ne×ne) · ion_bt_h (ne×n_bands) = ion_q_bt (ne×n_bands)
                // ldb = ne (row-major ne×ne Q), ldb = ne (bt_h block contiguously stored)
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
                        &vnl_data.shared.handle.per_ion_q[ion_idx],
                        &ion_bt_h,
                        &mut ion_q_bt,
                    )?;
                }

                // Write back into bt_q at offset
                {
                    let mut dst_view = bt_q.slice_mut(offset..offset + ne_bands);
                    stream.memcpy_dtod(&ion_q_bt, &mut dst_view).map_err(Error::Cuda)?;
                }
                offset += ne_bands;
            }

            // 4d. y_corr = y - q_bt   (copy y to y_corr, then AXPY -q_bt)
            let mut y_corr: CudaSlice<CudaComplex> =
                stream.alloc_zeros(nte as usize * n_bands as usize).map_err(Error::Cuda)?;
            stream.memcpy_dtod(&temp, &mut y_corr).map_err(Error::Cuda)?;
            blas.axpy_c64(
                nte * n_bands,
                CudaComplex { x: -1.0, y: 0.0 },
                &bt_q,
                1,
                &mut y_corr,
                1,
            )?;

            // 4e. r_h = B · y_corr  (n_pw × n_bands)
            let mut r_h: CudaSlice<CudaComplex> =
                stream.alloc_zeros(n_pw as usize * n_bands as usize).map_err(Error::Cuda)?;
            unsafe {
                blas.gemm_c64(
                    ZgemmConfig {
                        transa: blas::op::N,
                        transb: blas::op::N,
                        m: n_pw,
                        n: n_bands,
                        k: nte,
                        alpha: CudaComplex { x: 1.0, y: 0.0 },
                        lda: n_pw,
                        ldb: nte,
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                        ldc: n_pw,
                    },
                    &vnl_data.b_concat,
                    &y_corr,
                    &mut r_h,
                )?;
            }

            // 4f. r_y = B^H · r_h  (nte × n_bands)
            let mut r_y: CudaSlice<CudaComplex> =
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
                    &r_h,
                    &mut r_y,
                )?;
            }

            // 4g. dy = M^{-1} · r_y  (zgetrs in-place on r_y)
            let mut info_dev2 = stream.alloc_zeros::<i32>(1).map_err(Error::Cuda)?;
            solver.zgetrs(
                cublasOperation_t::CUBLAS_OP_N,
                nte,
                n_bands,
                &vnl_data.lu_m,
                &vnl_data.lu_ipiv,
                &mut r_y,
                &mut info_dev2,
            )?;

            // 4h. y += dy  (temp = temp + r_y via AXPY)
            blas.axpy_c64(
                nte * n_bands,
                CudaComplex { x: 1.0, y: 0.0 },
                &r_y,
                1,
                &mut temp,
                1,
            )?;
        }
    }

    // 5. hpsi = h_in - B·y  (final GEMM + AXPY)
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
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n_pw,
            },
            &vnl_data.b_concat,
            &temp,
            &mut **hpsi_dev,
        )?;
    }
    // hpsi += h_in  (AXPY)
    blas.axpy_c64(
        n_pw * n_bands,
        CudaComplex { x: 1.0, y: 0.0 },
        &h_in,
        1,
        &mut **hpsi_dev,
        1,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// S⁻¹·S identity diagnostic — measures ‖S⁻¹·S·ψ − ψ‖_∞
// ---------------------------------------------------------------------------
/// Verify the global Woodbury S⁻¹ operator by computing
/// `‖S⁻¹·(S·ψ) − ψ‖_∞` for a single-band test vector.
///
/// Returns `max_residual = max_i |(S⁻¹·S·ψ)_i − ψ_i|`.
///
/// The Woodbury formula S⁻¹ = I − B·(Q⁻¹ + B^H·B)⁻¹·B^H is algebraically
/// exact for the finite-dimensional USPP overlap S = I + B·Q·B^H.
/// Any deviation from zero reflects numerical error in:
///   1. Q-matrix conditioning (near-singular Cu 3d projector Q)
///   2. LU factorization precision (cusolver Zgetrf + Zgetrs)
///   3. B^H·B Gram matrix accumulation (cublas Zgemm reduction order)
///
/// Gate 0: ζ = max_residual must be < 1e-10 before Chebyshev filtering
/// can proceed. If ζ > 1e-10, the Woodbury construction must be
/// debugged before any filtering tests.
#[doc(hidden)]
#[cfg(feature = "chebyshev")]
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

    // ---- Step 1: S·psi = psi + Σ_ion β_g · q · (β_g^H · psi) ----
    let mut spsi_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    stream.memcpy_dtod(&psi_dev, &mut spsi_dev).map_err(Error::Cuda)?;

    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

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

    // ---- Step 2: S⁻¹ · (S·psi) via global Woodbury (builder pattern) ----
    let mut spsi_pw = PwCoefficients(spsi_dev);
    unsafe {
        apply_s_inverse()
            .hpsi_dev(&mut spsi_pw)
            .vnl_data(vnl_data)
            .n_bands(1)
            .n_pw(n)
            .blas(blas)
            .stream(stream)
            .solver(solver)
            .call()?;
    }

    // ---- Step 3: D2H and compute ‖S⁻¹·S·ψ − ψ‖_∞ ----
    stream.synchronize()?;
    let result: Vec<CudaComplex> = stream.clone_dtoh(&*spsi_pw).map_err(Error::Cuda)?;
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
// Synthesis test: ZGEMM accumulation order — cuBLAS GPU vs CPU reference
// ---------------------------------------------------------------------------
// The Davidson eigensolver uses ZGEMM(C = A^H · B) extensively:
//   - H_sub = ψ^H · Hψ       (n_pw × k, k = 25..150)
//   - S_overlap = ψ^H · Sψ    (n_pw × k)
//   - S-orthogonalization projection coefficients (n_pw × k vs n_pw × n_ref)
//
// cuBLAS may accumulate the dot products in a different order than a CPU
// reference (different thread/warp scheduling, different reduction trees).
// This test quantifies the per-element numerical difference for realistic
// Davidson subspace sizes.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::blas::{BlasHandle, ZgemmConfig, op};
    use cudarc::driver::CudaContext;

    /// Manual CPU ZGEMM: C = A^H · B (column-major, complex).
    /// A: (n_pw × k), B: (n_pw × k) → C: (k × k) complex Hermitian.
    fn cpu_zgemm_ah_b(
        a: &[CudaComplex],
        b: &[CudaComplex],
        n_pw: usize,
        k: usize,
    ) -> Vec<CudaComplex> {
        let mut c = vec![CudaComplex { x: 0.0, y: 0.0 }; k * k];
        for j in 0..k {
            for i in 0..k {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for g in 0..n_pw {
                    let a_gi = a[g + i * n_pw]; // A(g, i) col-major
                    let b_gj = b[g + j * n_pw]; // B(g, j) col-major
                    // conj(a_gi) * b_gj
                    re += a_gi.x * b_gj.x + a_gi.y * b_gj.y;
                    im += a_gi.x * b_gj.y - a_gi.y * b_gj.x;
                }
                c[i + j * k] = CudaComplex { x: re, y: im };
            }
        }
        c
    }

    /// Test cuBLAS ZGEMM vs CPU reference at realistic Davidson subspace sizes.
    #[test]
    fn zgemm_accumulation_order_realistic_sizes() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).expect("BlasHandle");

        // Davidson subspace sizes: k = current_nblock (25) up to superspace (150).
        let cases = [
            (10_000,  25, "block 25x25, n_pw=10k"),
            (30_000,  50, "iter1 50x50, n_pw=30k"),
            (60_000,  75, "iter2 75x75, n_pw=60k"),
            (60_000, 100, "iter3 100x100, n_pw=60k"),
            (60_000, 125, "iter4 125x125, n_pw=60k"),
            (60_000, 150, "iter5 150x150, n_pw=60k"),
        ];

        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;

        for &(n_pw, k, desc) in &cases {
            let total = n_pw * k;
            // Deterministic pseudo-random data (sin-based for reproducibility)
            let a_host: Vec<CudaComplex> = (0..total)
                .map(|idx| {
                    let phase = (idx as f64 * 0.987654321).sin() * 1000.0;
                    let re = (phase * 1.3).sin();
                    let im = (phase * 1.7).cos();
                    CudaComplex { x: re * 1e-3, y: im * 1e-3 }
                })
                .collect();
            let b_host: Vec<CudaComplex> = (0..total)
                .map(|idx| {
                    let phase = ((idx as f64 + 0.5) * 0.987654321).sin() * 1000.0;
                    let re = (phase * 1.9).cos();
                    let im = (phase * 1.1).sin();
                    CudaComplex { x: re * 1e-3, y: im * 1e-3 }
                })
                .collect();

            // CPU reference
            let c_cpu = cpu_zgemm_ah_b(&a_host, &b_host, n_pw, k);

            // GPU via cuBLAS
            let mut a_dev = stream.alloc_zeros::<CudaComplex>(total).expect("alloc a");
            let mut b_dev = stream.alloc_zeros::<CudaComplex>(total).expect("alloc b");
            let mut c_dev = stream.alloc_zeros::<CudaComplex>(k * k).expect("alloc c");
            stream.memcpy_htod(&a_host, &mut a_dev).expect("H2D a");
            stream.memcpy_htod(&b_host, &mut b_dev).expect("H2D b");

            unsafe {
                blas.gemm_c64(
                    ZgemmConfig {
                        transa: op::C,
                        transb: op::N,
                        m: k as i32,
                        n: k as i32,
                        k: n_pw as i32,
                        alpha: CudaComplex { x: 1.0, y: 0.0 },
                        lda: n_pw as i32,
                        ldb: n_pw as i32,
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                        ldc: k as i32,
                    },
                    &a_dev,
                    &b_dev,
                    &mut c_dev,
                )
                .expect("cuBLAS ZGEMM");
            }
            stream.synchronize().expect("sync");
            let c_gpu: Vec<CudaComplex> = stream.clone_dtoh(&c_dev).expect("D2H c");

            // Compare element-wise
            let mut case_max_abs = 0.0f64;
            let mut case_max_rel = 0.0f64;
            for idx in 0..(k * k) {
                let cpu = c_cpu[idx];
                let gpu = c_gpu[idx];
                let abs_diff = ((cpu.x - gpu.x).powi(2) + (cpu.y - gpu.y).powi(2)).sqrt();
                let cpu_norm = (cpu.x.powi(2) + cpu.y.powi(2)).sqrt();
                let rel_diff = if cpu_norm > 1e-30 { abs_diff / cpu_norm } else { abs_diff };
                if abs_diff > case_max_abs { case_max_abs = abs_diff; }
                if rel_diff > case_max_rel { case_max_rel = rel_diff; }
            }

            let status = if case_max_rel > 1e-6 { "HIGH" }
                else if case_max_rel > 1e-9 { "WARN" }
                else { "ok" };
            eprintln!(
                "ZGEMM {desc}: max|Δ|={:.3e} max|Δ/|C||={:.3e} [{status}]",
                case_max_abs, case_max_rel,
            );

            if case_max_abs > max_abs { max_abs = case_max_abs; }
            if case_max_rel > max_rel { max_rel = case_max_rel; }
        }

        eprintln!(
            "ZGEMM accumulation order: overall max|Δ|={:.3e} max rel={:.3e}",
            max_abs, max_rel,
        );

        assert!(
            max_rel < 1e-9,
            "ZGEMM accumulation order: max relative error {:.3e} exceeds 1e-9.",
            max_rel,
        );
    }

    /// Test cuFFT round-trip: forward C2C FFT → backward C2C FFT = N × identity.
    ///
    /// With no modification between forward and backward transforms, the
    /// round-trip should perfectly reconstruct the input (up to scale factor N).
    /// This quantifies cuFFT numerical error at realistic grid sizes.
    #[test]
    fn cufft_roundtrip_identity() {
        use crate::device::fft::BatchedFftPlan3d;
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = Arc::new(ctx.default_stream());

        // Cu111_CO wave grid: test at full size + smaller sizes
        let cases: &[(i32, i32, i32, i32, &str)] = &[
            (54, 90, 90, 25, "full 54x90x90, 25 bands"),
            (54, 90, 90,  1, "full 54x90x90,  1 band"),
            (27, 45, 45, 25, "half 27x45x45, 25 bands"),
        ];

        let mut worst_abs = 0.0f64;
        let mut worst_rel = 0.0f64;

        for &(ngx, ngy, ngz, n_bands, desc) in cases {
            let grid_size = (ngx * ngy * ngz) as usize;
            let n = grid_size as f64;

            // Generate random complex data on the full 3D grid
            let data_host: Vec<CudaComplex> = (0..(n_bands as usize * grid_size))
                .map(|idx| {
                    let phase = (idx as f64 * 0.987654321).sin() * 1000.0;
                    CudaComplex {
                        x: (phase * 1.3).sin() * 1e-3,
                        y: (phase * 1.7).cos() * 1e-3,
                    }
                })
                .collect();

            let total = n_bands as usize * grid_size;
            let mut data_dev = stream.alloc_zeros::<CudaComplex>(total).expect("alloc data");
            stream.memcpy_htod(&data_host, &mut data_dev).expect("H2D");

            // Build batched C2C FFT plan
            let plan = BatchedFftPlan3d::plan_batched_c2c(
                ngx, ngy, ngz, n_bands, Arc::clone(&stream),
            ).expect("cufft plan");

            let mut tmp_dev = stream.alloc_zeros::<CudaComplex>(total).expect("alloc tmp");

            // Forward FFT (data_dev→tmp_dev, both mut)
            unsafe { plan.c2c_forward(&mut data_dev, &mut tmp_dev).expect("forward"); }
            // Backward FFT (tmp_dev→data_dev, both mut)
            unsafe { plan.c2c_inverse(&mut tmp_dev, &mut data_dev).expect("inverse"); }
            stream.synchronize().expect("sync");

            let data_out: Vec<CudaComplex> = stream.clone_dtoh(&data_dev).expect("D2H");

            // Round-trip: output should equal N × input (cuFFT inverse scales by 1/N)
            let mut max_abs = 0.0f64;
            let mut max_rel = 0.0f64;
            for idx in 0..total {
                let inp = data_host[idx];
                let out = data_out[idx];
                let scaled = CudaComplex { x: out.x / n, y: out.y / n };
                let abs_diff = ((scaled.x - inp.x).powi(2) + (scaled.y - inp.y).powi(2)).sqrt();
                let inp_norm = (inp.x.powi(2) + inp.y.powi(2)).sqrt();
                let rel_diff = if inp_norm > 1e-30 { abs_diff / inp_norm } else { abs_diff };
                if abs_diff > max_abs { max_abs = abs_diff; }
                if rel_diff > max_rel { max_rel = rel_diff; }
            }

            let status = if max_rel > 1e-6 { "HIGH" }
                else if max_rel > 1e-9 { "WARN" }
                else { "ok" };
            eprintln!(
                "cuFFT roundtrip {desc}: max|Δ|={:.3e} rel={:.3e} [{status}]",
                max_abs, max_rel,
            );
            if max_abs > worst_abs { worst_abs = max_abs; }
            if max_rel > worst_rel { worst_rel = max_rel; }
        }

        eprintln!(
            "cuFFT roundtrip overall: worst |Δ|={:.3e} worst rel={:.3e}",
            worst_abs, worst_rel,
        );

        assert!(
            worst_rel < 1e-6,
            "cuFFT roundtrip identity: worst relative error {:.3e} exceeds 1e-6.",
            worst_rel,
        );
    }
}

// ---------------------------------------------------------------------------
// End-to-end synthesis test: GPU H·psi vs CPU reference
// ---------------------------------------------------------------------------
// Loads actual Cu111_CO H_dump fixture data and compares the GPU
// apply_full_hamiltonian output against a CPU reference computation
// for T+V_loc.  V_NL is verified separately via hsub_vs_castep.
//
// This test requires the CASTEP H_dump fixture and ~12 GB GPU VRAM.
// Marked #[ignore] — run manually:
//   cargo test --release -- hpsi_gpu_vs_cpu --ignored --nocapture
#[cfg(test)]
mod hpsi_integration {
    use chemrust_hamiltonian_core::{
        CheckFile, GVectorGrid,
        hamiltonian::apply_local_hamiltonian,
        formatted,
    };

    /// Path to CASTEP H_dump fixture directory.
    const H_DUMP_DIR: &str = "/export/public_castep_jobs/tony/Cu111_CO_H_dump";

    #[test]
    #[ignore = "requires GPU + CASTEP H_dump fixture (~12 GB VRAM)"]
    fn hpsi_gpu_vs_cpu_per_gvector() {
        use super::*;
        use crate::device::fft::BatchedFftPlan3d;
        use crate::device::blas::BlasHandle;
        use crate::device::solver::SolverHandle;
        use crate::eigensolver::kernels::CudaKernelSet;
        use crate::eigensolver::vnl_data::VnlBatchData;
        use cudarc::driver::{CudaContext, CudaStream, DevicePtr};
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = Arc::new(ctx.default_stream());
        let blas = BlasHandle::new(Arc::clone(&stream)).expect("BlasHandle");
        let solver = SolverHandle::new(Arc::clone(&stream)).expect("SolverHandle");

        // --- Load H_dump fixture ---
        let fixture_dir = std::env::var("CASTEP_FIXTURE_DIR")
            .unwrap_or_else(|_| H_DUMP_DIR.to_string());
        let check_path = format!("{fixture_dir}/Cu111_CO.check");
        let check_file = std::fs::File::open(&check_path).expect("open .check");
        let bin = CheckFile::read(std::io::BufReader::new(check_file)).expect("read .check");
        let wfc = bin.wavefunction.as_ref().expect("wavefunction section");
        let kpt = &wfc.kpt_data[0];
        let n_bands = kpt.bands.len();
        let n_pw = kpt.nplw;
        let wave_grid_dims = wfc.grid;
        let cell = &bin.cell;
        let [ngx, ngy, ngz] = wave_grid_dims;
        let ngx_i = ngx as i32;
        let ngy_i = ngy as i32;
        let ngz_i = ngz as i32;
        let grid_size = (ngx * ngy * ngz) as usize;
        let inv_ntotal = 1.0 / grid_size as f64;

        // Build wave grid and Cartesian G-vectors
        let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);
        let recip = cell.recip_lattice.as_array();
        let gcart: Vec<[f64; 3]> = kpt.pw_grid_coord.iter().map(|&[h, k, l]| {
            let gf = [h as f64, k as f64, l as f64];
            std::array::from_fn(|j| (0..3).map(|i| gf[i] * recip[i][j]).sum())
        }).collect();
        let k_cart = {
            let kf = kpt.coords;
            std::array::from_fn(|j| (0..3).map(|i| kf[i] * recip[i][j]).sum())
        };

        // FFT indices: ix-innermost (0-based) for CPU; iz-innermost for GPU
        let fft_indices_3d: Vec<[usize; 3]> = kpt.pw_grid_coord.iter().map(|&[h, k, l]| {
            let ix = if h >= 0 { h as usize } else { (h + ngx as i32) as usize };
            let iy = if k >= 0 { k as usize } else { (k + ngy as i32) as usize };
            let iz = if l >= 0 { l as usize } else { (l + ngz as i32) as usize };
            [iz, iy, ix]
        }).collect();
        let fft_idx_iz: Vec<i32> = kpt.pw_grid_coord.iter()
            .map(|&[h, k, l]| {
                let ix = if h >= 0 { h as usize } else { (h + ngx as i32) as usize };
                let iy = if k >= 0 { k as usize } else { (k + ngy as i32) as usize };
                let iz = if l >= 0 { l as usize } else { (l + ngz as i32) as usize };
                (iz + ngz as usize * (iy + ngy as usize * ix)) as i32
            }).collect();

        // Load V_eff from .pot_fmt
        let pot_path = format!("{fixture_dir}/Cu111_CO.pot_fmt");
        let pot_text = std::fs::read_to_string(&pot_path).expect("read .pot_fmt");
        let (_, v_eff_arr) = formatted::parse_pot_fmt(&pot_text).expect("parse .pot_fmt");
        // v_eff_arr is Array3<f64> with shape (ngx, ngy, ngz), ix-innermost
        // V_eff as EffectivePotential (ix-innermost, for CPU)
        let v_eff_cpu = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(v_eff_arr.clone())
        );
        // V_eff transposed to iz-innermost (for GPU)
        let v_eff_iz: Vec<f64> = {
            let mut out = vec![0.0; grid_size];
            for ix in 0..ngx { for iy in 0..ngy { for iz in 0..ngz {
                out[iz + ngz * (iy + ngy * ix)] = v_eff_arr[[ix, iy, iz]];
            }}}
            out
        };

        // Build GPU infrastructure
        let ctx_arc = Arc::new(ctx.clone());
        let kernels = CudaKernelSet::new(&ctx_arc)
            .expect("kernels");
        let kinetic_host: Vec<f64> = gcart.iter().map(|gc| {
            let kg = [k_cart[0] + gc[0], k_cart[1] + gc[1], k_cart[2] + gc[2]];
            0.5 * (kg[0]*kg[0] + kg[1]*kg[1] + kg[2]*kg[2])
        }).collect();

        // Upload GPU data
        let fft_idx_dev = {
            let mut d = stream.alloc_zeros::<i32>(n_pw).expect("alloc fft_idx");
            stream.memcpy_htod(&fft_idx_iz, &mut d).expect("H2D fft_idx");
            d
        };
        let v_eff_dev = {
            let mut d = stream.alloc_zeros::<f64>(grid_size).expect("alloc v_eff");
            stream.memcpy_htod(&v_eff_iz, &mut d).expect("H2D v_eff");
            d
        };
        let kinetic_dev = {
            let mut d = stream.alloc_zeros::<f64>(n_pw).expect("alloc kinetic");
            stream.memcpy_htod(&kinetic_host, &mut d).expect("H2D kinetic");
            crate::eigensolver::davidson_types::KineticPreconditioner::new(d)
        };

        // Test bands: spread across the full band range
        let n_bands_total = kpt.bands.len();
        let test_bands: Vec<usize> = {
            let n_test = 8usize.min(n_bands_total);
            (0..n_test).map(|i| i * n_bands_total / n_test).collect()
        };
        let n_test = test_bands.len();

        // Build FFT plan with correct batch count
        let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
            ngx_i, ngy_i, ngz_i, n_test as i32, Arc::clone(&stream),
        ).expect("FFT plan");

        // Allocate GPU buffers for all test bands
        let psi_flat: Vec<CudaComplex> = test_bands.iter()
            .flat_map(|&b| kpt.bands[b].iter().map(|&c| CudaComplex{x:c.re,y:c.im}))
            .collect();
        let mut psi_dev = PwCoefficients::new(
            stream.alloc_zeros::<CudaComplex>(n_pw * n_test).expect("alloc psi"));
        stream.memcpy_htod(&psi_flat, &mut psi_dev.0).expect("H2D psi");
        let mut hpsi_dev = PwCoefficients::new(
            stream.alloc_zeros::<CudaComplex>(n_pw * n_test).expect("alloc hpsi"));
        let mut grid_dev = stream.alloc_zeros::<CudaComplex>(n_test * grid_size)
            .expect("alloc grid");

        // GPU: apply T + V_loc (skip V_NL — tested separately in hsub_vs_castep)
        unsafe {
            apply_v_loc_hamiltonian()
                .psi_dev(&psi_dev)
                .hpsi_dev(&mut hpsi_dev)
                .grid_dev(&mut grid_dev)
                .kinetic_dev(&kinetic_dev)
                .fft_idx_dev(&fft_idx_dev)
                .v_eff_dev(&v_eff_dev)
                .n_pw(n_pw as i32)
                .n_bands(n_test as i32)
                .grid_size(grid_size as i32)
                .ngx(ngx_i)
                .ngy(ngy_i)
                .ngz(ngz_i)
                .inv_ntotal(inv_ntotal)
                .fft_plan(&fft_plan)
                .kernels(&kernels)
                .stream(&stream)
                .call()
                .expect("GPU apply_v_loc");
        }
        stream.synchronize().expect("sync");
        let hpsi_gpu: Vec<CudaComplex> = stream.clone_dtoh(&hpsi_dev.0)
            .expect("D2H hpsi");

        // CPU: apply T + V_loc for each test band
        let mut worst_abs = 0.0f64;
        let mut worst_rel = 0.0f64;
        let mut sum_signed = 0.0f64;
        let mut sum_count = 0u64;

        for (ti, &band_idx) in test_bands.iter().enumerate() {
            let psi_band: Vec<num_complex::Complex64> = kpt.bands[band_idx]
                .iter().map(|&c| c).collect();
            let cpu_hpsi = apply_local_hamiltonian(
                &psi_band, &fft_indices_3d, &gcart, k_cart,
                &v_eff_cpu, &wave_grid,
            ).expect("CPU apply_local_hamiltonian");

            let gpu_offset = ti * n_pw;
            let mut band_max_abs = 0.0f64;
            let mut band_max_rel = 0.0f64;
            let mut band_sum_signed = 0.0f64;

            for g in 0..n_pw {
                let cpu = cpu_hpsi[g];
                let gpu = hpsi_gpu[gpu_offset + g];
                let diff = CudaComplex { x: gpu.x - cpu.re, y: gpu.y - cpu.im };
                let abs_diff = (diff.x.powi(2) + diff.y.powi(2)).sqrt();
                let cpu_norm = (cpu.re.powi(2) + cpu.im.powi(2)).sqrt();
                let rel = if cpu_norm > 1e-30 { abs_diff / cpu_norm } else { abs_diff };
                if abs_diff > band_max_abs { band_max_abs = abs_diff; }
                if rel > band_max_rel { band_max_rel = rel; }
                band_sum_signed += diff.x; // track systematic bias (real part)
            }

            let band_mean_signed = band_sum_signed / n_pw as f64;

            // Also compute Rayleigh quotient comparison
            let (mut rq_gpu_re, mut rq_gpu_im) = (0.0f64, 0.0f64);
            let (mut rq_cpu_re, mut rq_cpu_im) = (0.0f64, 0.0f64);
            for g in 0..n_pw {
                let psi_g = kpt.bands[band_idx][g];
                let hpsi_g = hpsi_gpu[gpu_offset + g];
                let hpsi_c = cpu_hpsi[g];
                rq_gpu_re += psi_g.re * hpsi_g.x + psi_g.im * hpsi_g.y;
                rq_gpu_im += psi_g.re * hpsi_g.y - psi_g.im * hpsi_g.x;
                rq_cpu_re += psi_g.re * hpsi_c.re + psi_g.im * hpsi_c.im;
                rq_cpu_im += psi_g.re * hpsi_c.im - psi_g.im * hpsi_c.re;
            }
            let rq_diff = (rq_gpu_re - rq_cpu_re).abs();

            let status = if band_max_rel > 1e-3 { "HIGH" }
                else if band_max_rel > 1e-6 { "WARN" }
                else { "ok" };
            eprintln!(
                "H·psi band {band_idx:3}: max|Δ|={:.3e} max rel={:.3e} mean_signed={:+.3e} RQ_diff={:.6e} [{status}]",
                band_max_abs, band_max_rel, band_mean_signed, rq_diff,
            );

            if band_max_abs > worst_abs { worst_abs = band_max_abs; }
            if band_max_rel > worst_rel { worst_rel = band_max_rel; }
            sum_signed += band_sum_signed;
            sum_count += n_pw as u64;
        }

        let mean_signed = sum_signed / sum_count as f64;
        eprintln!(
            "H·psi GPU vs CPU: worst |Δ|={:.3e} worst rel={:.3e} mean_signed_bias={:+.3e}",
            worst_abs, worst_rel, mean_signed,
        );

        // For acceptable FFT numerical noise, max relative error should be < 1e-3
        // (allowing for the fact that some G-vectors have very small coefficients).
        // Systematic bias (mean_signed) should be near zero — a non-zero bias
        // would indicate a systematic error in the GPU V_loc path.
        assert!(
            worst_rel < 1e-3,
            "H·psi GPU vs CPU: per-element relative error {:.3e} exceeds 1e-3",
            worst_rel,
        );
        assert!(
            mean_signed.abs() < 1e-12,
            "H·psi GPU vs CPU: systematic bias {:.3e} detected — \
             GPU V_loc path has a systematic offset vs CPU.\n\
             This would shift all eigenvalues systematically, explaining \
             the convergence-rate divergence.",
            mean_signed,
        );
    }

    /// Test: D2D memcpy → cuBLAS ZGEMM ordering on the same stream.
    ///
    /// Hypothesis H2: on Pascal (GTX 1080 Ti), cudaMemcpyAsync (copy engine)
    /// and cublasZgemm (compute engine) may not serialize correctly even when
    /// both are issued on the same user stream — the ZGEMM may read stale
    /// data if the copy hasn't completed.
    ///
    /// Test design:
    ///   1. Fill c_proj with non-zero pattern on GPU
    ///   2. D2D memcpy zeros → c_proj (on stream S)
    ///   3. Immediately: cuBLAS ZGEMM reads c_proj, writes result to c_temp
    ///   4. Sync stream S
    ///   5. If c_temp is zero: ZGEMM saw post-copy data ✓ (ordering correct)
    ///   6. If c_temp is non-zero: ZGEMM saw pre-copy data ✗ (ordering broken)
    #[test]
    fn memcpy_dtod_before_cublas_zgemm_ordering() {
        use cudarc::driver::CudaContext;
        use super::*;
        use crate::device::blas::{BlasHandle, op};

        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();  // already Arc<CudaStream>
        let blas = BlasHandle::new(stream.clone()).expect("BlasHandle");

        // Use small but realistic dimensions — large enough that cuBLAS
        // actually launches a kernel, small enough to run quickly.
        let m = 64i32;
        let n = 64i32;
        let k = 64i32;
        let total = (m * n) as usize;

        // Pattern: all 1.0+0i (non-zero, easy to detect)
        let ones: Vec<CudaComplex> = vec![CudaComplex { x: 1.0, y: 0.0 }; total];
        // GPU allocations
        let mut c_proj = stream.alloc_zeros::<CudaComplex>(total).expect("alloc c_proj");
        let mut c_temp = stream.alloc_zeros::<CudaComplex>(total).expect("alloc c_temp");
        let zeros_dev = stream.alloc_zeros::<CudaComplex>(total).expect("alloc zeros");

        // Dummy A matrix for the ZGEMM (identity-like: just reads c_proj)
        // c_temp = A · c_proj  with A = identity
        let mut a_dev = stream.alloc_zeros::<CudaComplex>(total).expect("alloc a");
        // A = I: diagonal elements = 1.0
        let mut a_host = vec![CudaComplex { x: 0.0, y: 0.0 }; total];
        for i in 0..m as usize {
            a_host[i * m as usize + i] = CudaComplex { x: 1.0, y: 0.0 };
        }
        stream.memcpy_htod(&a_host, &mut a_dev).expect("H2D A");

        // Fill c_proj with non-zero pattern
        stream.memcpy_htod(&ones, &mut c_proj).expect("H2D ones");
        stream.synchronize().expect("sync after H2D");

        // --- THE TEST: D2D memcpy followed by cuBLAS ZGEMM ---
        // Step 1: D2D memcpy zeros → c_proj (copy engine, async on stream)
        stream.memcpy_dtod(&zeros_dev, &mut c_proj).expect("D2D zeros");

        // Step 2: IMMEDIATELY read c_proj via cuBLAS ZGEMM
        // c_temp = 1.0 · A · c_proj + 0.0 · c_temp
        // If stream ordering is correct: c_temp = A · 0 = 0
        // If stream ordering is broken: c_temp = A · 1 = 1 (stale data)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::N,
                    transb: op::N,
                    m, n, k,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: m,
                    ldb: k,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: m,
                },
                &a_dev,      // A = I (m×k)
                &c_proj,      // B = c_proj (k×n), should be all zeros
                &mut c_temp,  // C = result (m×n)
            )
            .expect("ZGEMM");
        }

        // Sync and read result
        stream.synchronize().expect("sync after ZGEMM");
        let result: Vec<CudaComplex> = stream.clone_dtoh(&c_temp).expect("D2H result");

        // Check: all elements should be zero (ZGEMM read c_proj after memcpy)
        let max_abs = result.iter()
            .map(|c| (c.x.powi(2) + c.y.powi(2)).sqrt())
            .fold(0.0f64, f64::max);

        eprintln!(
            "memcpy→cuBLAS ordering test: max|result| = {:.3e} (expect 0 if ordered, >0 if race)",
            max_abs,
        );

        if max_abs > 1e-10 {
            eprintln!("  FAIL: cuBLAS read stale c_proj — stream ordering broken!");
            eprintln!("  First 5 elements: {:?}", result.iter().take(5).map(|c| (c.x, c.y)).collect::<Vec<_>>());
        } else {
            eprintln!("  PASS: cuBLAS saw post-copy data — stream ordering correct");
        }

        assert!(
            max_abs < 1e-10,
            "D2D memcpy → cuBLAS ZGEMM ordering failure: cuBLAS read stale data (max|result| = {:.3e})",
            max_abs,
        );
    }
}

