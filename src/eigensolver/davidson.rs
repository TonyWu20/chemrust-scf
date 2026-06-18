// ---------------------------------------------------------------------------
// Production Davidson v1 eigensolver — single-sweep (Phase 1A)
// ---------------------------------------------------------------------------
//
// Single-sweep algorithm (no outer Davidson loop):
//   1. H|ψ⟩ = apply_full_hamiltonian(ψ)
//   2. S|ψ⟩ = ψ; apply_s_times(ψ, S|ψ⟩)
//   3. λ_b = Re⟨ψ_b|Hψ_b⟩ / Re⟨ψ_b|Sψ_b⟩
//   4. r_b = Hψ_b − λ_b·Sψ_b
//   5. sinv_r = S⁻¹·r (batch Woodbury); norm_b = √Re⟨r_b|sinv_r_b⟩
//   6. Lock bands where norm_b < lock_tol
//   7. If all locked: return ψ unchanged
//   8. Gather unconverged ψ + Hψ → contiguous buffers
//   9. Unified k×k subspace diagonalization via ZHEEVD (standard EVP)
//  10. Rotate: ψ_new = ψ_u · X
//  11. S-orthogonalize ψ_new against locked bands (one Gram–Schmidt pass)
//  12. Scatter locked + rotated → ψ_out; return
//
// Rationale: For continuation initial guesses (CASTEP .check or prior SCF
// iterate), ψ lives ε-close to the eigenvectors of (H,S). A single unified
// Subspace diagonalization on the unconverged sub-block gives the exact solution within that
// span. SCF-level convergence is driven by the lock ratchet tightening across
// iterations, not by an outer eigensolver loop.
// See notes/plans/phase-eigensolver-migration/PHASE1A_POSTMORTEM.md §3.
//
// TpaPreconditioner and detect_degenerate_blocks are preserved for future
// cold-start work but are not called in this single-sweep variant.

use std::sync::Arc;

use cudarc::cublas::sys::{
    cublasDiagType_t, cublasHandle_t, cublasOperation_t, cublasSideMode_t,
    cublasZaxpy_v2, cublasZcopy_v2, cublasZdotc_v2, cublasZscal_v2, cublasZtrsm_v2,
};
use cudarc::cusolver::sys::{cublasFillMode_t, cusolverEigMode_t};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::beta_phi_cache::BetaPhiCache;
use crate::eigensolver::davidson_types::*;
use ndarray::Array2;
use num_complex::Complex64;
use crate::eigensolver::hamiltonian::{apply_full_hamiltonian, apply_s_times};
#[cfg(feature = "scf_diag")]
use crate::eigensolver::hamiltonian::apply_v_loc_hamiltonian;
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::preconditioner::{apply_preconditioner, compute_r_vector, prepare_preconditioner, TpaPreconditioner};

use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;
use bon::builder;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Davidson v1 eigensolver.
#[allow(dead_code)]
pub(crate) struct DavidsonConfig {
    /// Maximum number of outer iterations (preserved for API; single-sweep
    /// always returns after 1 pass).
    pub max_outer_iter: usize,
    /// Eigenvalue spacing threshold (Ha) for degenerate-block detection.
    /// Currently unused — per-block subspace diagonalization destroys global eigenvalue
    /// ordering (see postmortem, commit 76e03e9). Retained for future
    /// cold-start variant.
    #[allow(dead_code)]
    pub block_eps_degen: f64,
}

// ---------------------------------------------------------------------------
// Result types and diagnostics
// ---------------------------------------------------------------------------

/// Result of the Davidson v1 diagonalization.
#[allow(dead_code)]
pub(crate) struct DavidsonResult {
    /// Output wavefunctions (column-major, n_pw × n_bands).
    pub psi_out: CudaSlice<CudaComplex>,
    /// Per-band eigenvalues (sorted ascending).
    pub eigenvalues: Vec<f64>,
    /// Number of locked bands on exit.
    pub n_locked: usize,
    /// S⁻¹-weighted residual norms for all bands.
    pub residual_norms_sinv: ResidualSInvNorm,
    /// Number of outer iterations completed (0 for single-sweep v1).
    pub n_outer_iterations: usize,
}

/// Snapshot of Davidson diagnostics after the most recent solve.
#[derive(Clone)]
pub struct DavidsonDiagnostic {
    pub n_locked: usize,
    pub n_unconverged: usize,
    pub locked_indices: Vec<usize>,
    pub unconv_indices: Vec<usize>,
    pub residual_norms_sinv: ResidualSInvNorm,
    pub max_residual_sinv: f64,
    pub lock_tol: f64,
    pub eigenvalue_deltas: Vec<f64>,
}

/// Most recent Davidson v1 diagnostic, accessible for tests.
pub static DAVIDSON_LAST_DIAG: std::sync::Mutex<Option<DavidsonDiagnostic>> =
    std::sync::Mutex::new(None);

// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

/// Run the production Davidson v1 eigensolver — single sweep.
///
/// # Algorithm
///
/// 1. Hψ = apply_full_hamiltonian(ψ)
/// 2. Sψ = ψ; apply_s_times(ψ, Sψ)
/// 3. λ_b = Re⟨ψ_b|Hψ_b⟩ / Re⟨ψ_b|Sψ_b⟩
/// 4. r_b = Hψ_b − λ_b·Sψ_b
/// 5. sinv_r = S⁻¹·r (batch Woodbury); norm_b = √Re⟨r_b|sinv_r_b⟩
/// 6. Lock bands where norm_b < lock_tol
/// 7. If all locked: return ψ unchanged
/// 8. Gather unconverged ψ + Hψ → contiguous buffers
/// 9. Unified k×k subspace diagonalization (ZHEEVD), k = n_unconv
/// 10. Rotate ψ_new = ψ_u · X
/// 11. S-orthogonalize ψ_new against locked bands
/// 12. Scatter output; return eigenvalues, ψ_out
///
/// # Safety
///
/// All device pointers must be valid and of sufficient size. `psi_init` must
/// be S-orthonormal (column-major, n_bands × n_pw).
#[builder]
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
#[cfg(any())] // dead: superseded by davidson_diagonalise
unsafe fn davidson_v1(
    psi_init: &PwCoefficients,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    vnl_data: &VnlBatchData,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    lock_tol: f64,
    blas: &BlasHandle,
    solver: &SolverHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
    _ctx: &Arc<CudaContext>,
    _cfg: &DavidsonConfig,
) -> Result<DavidsonResult, Error> {
    let n_elem = n_bands * n_pw;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;
    let handle = blas.raw_handle();

    // ------------------------------------------------------------------
    // Persistent buffers
    // ------------------------------------------------------------------
    let mut psi_dev = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    stream
        .memcpy_dtod(&**psi_init, &mut psi_dev.0)
        .map_err(Error::Cuda)?;

    let mut hpsi_dev = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut spsi_dev = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let mut residual_dev = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let grid_alloc = n_bands * grid_size;
    let mut grid_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // ------------------------------------------------------------------
    // Steps 1–2: Compute Hψ and Sψ
    // ------------------------------------------------------------------
    unsafe {
        apply_full_hamiltonian()
            .psi_dev(&psi_dev)
            .v_eff_dev(v_eff_dev)
            .kinetic_dev(kinetic_dev)
            .fft_idx_dev(fft_idx_dev)
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(grid_size)
            .inv_ntotal(inv_ntotal)
            .fft_plan(fft_plan)
            .hpsi_dev(&mut hpsi_dev)
            .grid_dev(&mut grid_dev)
            .vnl_data(vnl_data)
            .blas(blas)
            .kernels(kernels)
            .stream(stream)
            .call()?;
    }

    // Sψ: pre-copy ψ → spsi (identity term), then accumulate β·Q·β^H
    stream
        .memcpy_dtod(&*psi_dev, &mut spsi_dev.0)
        .map_err(Error::Cuda)?;
    unsafe {
        apply_s_times()
            .psi_dev(&psi_dev)
            .spsi_dev(&mut spsi_dev)
            .vnl_data(vnl_data)
            .n_bands(n_bands_i32)
            .n_pw(n_pw_i32)
            .blas(blas)
            .stream(stream)
            .call()?;
    }

    // ------------------------------------------------------------------
    // Steps 3–4: Per-band Rayleigh quotient and residual
    // ------------------------------------------------------------------
    let mut eigenvalues = vec![0.0_f64; n_bands];
    {
        let (psi_ptr, _) = psi_dev.device_ptr(stream);
        let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
        let (spsi_ptr, _) = spsi_dev.device_ptr(stream);
        let (residual_mut, _) = residual_dev.device_ptr_mut(stream);

        #[allow(clippy::needless_range_loop)]
        for b in 0..n_bands {
            let psi_b = (psi_ptr as *const CudaComplex).add(b * n_pw);
            let hpsi_b = (hpsi_ptr as *const CudaComplex).add(b * n_pw);
            let spsi_b = (spsi_ptr as *const CudaComplex).add(b * n_pw);
            let r_b = (residual_mut as *mut CudaComplex).add(b * n_pw);

            // Rayleigh quotient: λ_b = Re⟨ψ_b|Hψ_b⟩ / Re⟨ψ_b|Sψ_b⟩
            let mut dot_h = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle,
                n_pw_i32,
                psi_b as *const _,
                1,
                hpsi_b as *const _,
                1,
                &mut dot_h as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;

            let mut dot_s = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle,
                n_pw_i32,
                psi_b as *const _,
                1,
                spsi_b as *const _,
                1,
                &mut dot_s as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;

            let lambda = dot_h.x / dot_s.x;
            eigenvalues[b] = lambda;

            // Residual: r_b = hpsi_b − λ·spsi_b
            // Step 1: copy hpsi_b → r_b
            cublasZcopy_v2(
                handle, n_pw_i32,
                hpsi_b as *const _, 1,
                r_b as *mut _, 1,
            )
            .result()
            .map_err(Error::Blas)?;
            // Step 2: r_b += −λ · spsi_b
            let neg_lambda = CudaComplex { x: -lambda, y: 0.0 };
            cublasZaxpy_v2(
                handle, n_pw_i32,
                &neg_lambda as *const _ as *const _,
                spsi_b as *const _, 1,
                r_b as *mut _, 1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // ------------------------------------------------------------------
    // Step 5: S⁻¹-weighted residual norm (batch Woodbury)
    // ------------------------------------------------------------------
    // residual_dev ← S⁻¹ · residual_dev  (in-place)
    let mut sinv_r_dev = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    stream
        .memcpy_dtod(&*residual_dev, &mut sinv_r_dev.0)
        .map_err(Error::Cuda)?;
    unsafe {
        apply_s_inverse()
            .hpsi_dev(&mut sinv_r_dev)
            .vnl_data(vnl_data)
            .n_bands(n_bands_i32)
            .n_pw(n_pw_i32)
            .blas(blas)
            .stream(stream)
            .solver(solver)
            .call()?;
    }

    let mut residual_norms_sinv = vec![0.0_f64; n_bands];
    {
        let (residual_ptr, _) = residual_dev.device_ptr(stream);
        let (sinv_ptr, _) = sinv_r_dev.device_ptr(stream);
        #[allow(clippy::needless_range_loop)]
        for b in 0..n_bands {
            let r_b = (residual_ptr as *const CudaComplex).add(b * n_pw);
            let sinv_b = (sinv_ptr as *const CudaComplex).add(b * n_pw);
            let mut dot = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle, n_pw_i32,
                r_b as *const _, 1,
                sinv_b as *const _, 1,
                &mut dot as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;
            residual_norms_sinv[b] = dot.x.sqrt();
        }
    }

    // ------------------------------------------------------------------
    // Step 6: Per-band locking (norm criterion only — single sweep)
    // ------------------------------------------------------------------
    let mut locked = vec![false; n_bands];
    let mut unconv_idx: Vec<usize> = Vec::new();
    let mut locked_idx: Vec<usize> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for b in 0..n_bands {
        if residual_norms_sinv[b] < lock_tol {
            locked[b] = true;
            locked_idx.push(b);
        } else {
            unconv_idx.push(b);
        }
    }
    let n_locked = locked_idx.len();
    let n_unconv = unconv_idx.len();

    let max_res = residual_norms_sinv
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);

    // ------------------------------------------------------------------
    // Step 7: Early exit if all bands locked
    // ------------------------------------------------------------------
    if n_unconv == 0 {
        *DAVIDSON_LAST_DIAG.lock().unwrap() = Some(DavidsonDiagnostic {
            n_locked,
            n_unconverged: 0,
            locked_indices: locked_idx,
            unconv_indices: vec![],
            residual_norms_sinv: ResidualSInvNorm::new(residual_norms_sinv.clone()),
            max_residual_sinv: max_res,
            lock_tol,
            eigenvalue_deltas: vec![0.0_f64; n_bands],
        });

        return Ok(DavidsonResult {
            psi_out: psi_dev.0, // ψ unchanged
            eigenvalues,
            n_locked,
            residual_norms_sinv: ResidualSInvNorm::new(residual_norms_sinv),
            n_outer_iterations: 0,
        });
    }

    // ------------------------------------------------------------------
    // Step 8: Gather unconverged ψ and Hψ columns → contiguous buffers
    // ------------------------------------------------------------------
    let k = n_unconv;
    let mut psi_unconv_dev = PwCoefficients::new(
        stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?);
    let mut hpsi_unconv_dev = PwCoefficients::new(
        stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?);

    {
        let (psi_ptr, _) = psi_dev.device_ptr(stream);
        let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
        let (psi_u_mut, _) = psi_unconv_dev.device_ptr_mut(stream);
        let (hpsi_u_mut, _) = hpsi_unconv_dev.device_ptr_mut(stream);

        for (u, &b) in unconv_idx.iter().enumerate() {
            cublasZcopy_v2(
                handle, n_pw_i32,
                (psi_ptr as *const CudaComplex).add(b * n_pw) as *const _, 1,
                (psi_u_mut as *mut CudaComplex).add(u * n_pw) as *mut _, 1,
            )
            .result()
            .map_err(Error::Blas)?;
            cublasZcopy_v2(
                handle, n_pw_i32,
                (hpsi_ptr as *const CudaComplex).add(b * n_pw) as *const _, 1,
                (hpsi_u_mut as *mut CudaComplex).add(u * n_pw) as *mut _, 1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // ------------------------------------------------------------------
    // Steps 9–10: Unified k×k subspace diagonalization + rotation
    // ------------------------------------------------------------------
    let mut psi_unconv_new = PwCoefficients::new(
        stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?);
    let mut eig_dev: CudaSlice<f64> = stream.alloc_zeros(k).map_err(Error::Cuda)?;
    let mut info_dev: CudaSlice<i32> = stream.alloc_zeros(1).map_err(Error::Cuda)?;
    let mut eigenvalues_k = vec![0.0_f64; k];

    diagonalise_subspace()
        .psi_block(&psi_unconv_dev)
        .hpsi_block(&hpsi_unconv_dev)
        .vnl_data(vnl_data)
        .k(k)
        .n_pw(n_pw)
        .blas(blas)
        .solver(solver)
        .stream(stream)
        .eigenvalues_out(&mut eigenvalues_k)
        .eig_dev(&mut eig_dev)
        .info_dev(&mut info_dev)
        .psi_rotated(&mut psi_unconv_new)
        .gamma_point(gamma_point)
        .call()?;

    // Update eigenvalues for unconverged bands
    for (pos, &b) in unconv_idx.iter().enumerate() {
        eigenvalues[b] = eigenvalues_k[pos];
    }

    // ------------------------------------------------------------------
    // Step 11: S-orthogonalize rotated columns against locked bands
    // ------------------------------------------------------------------
    if n_locked > 0 {
        // Single-column temporaries for S-orth (S·ψ_locked computation)
        let mut s_psi_in = PwCoefficients::new(
            stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
        let mut s_psi_out = PwCoefficients::new(
            stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);

        for u in 0..k {
            let (psi_unconv_ptr, _) = psi_unconv_new.device_ptr(stream);
            let unrot_col =
                (psi_unconv_ptr as *const CudaComplex).add(u * n_pw);

            // Orthogonalize against each locked band
            for &lb in &locked_idx {
                // Copy ψ_locked → s_psi_in (input to apply_s_times)
                let (psi_ptr, _) = psi_dev.device_ptr(stream);
                let psi_lb = (psi_ptr as *const CudaComplex).add(lb * n_pw);
                let (s_in_mut, _) = s_psi_in.device_ptr_mut(stream);
                cublasZcopy_v2(
                    handle, n_pw_i32,
                    psi_lb as *const _, 1,
                    s_in_mut as *mut _, 1,
                )
                .result()
                .map_err(Error::Blas)?;

                // s_psi_out = ψ_locked (identity term, pre-copy convention)
                let (s_out_mut, _) = s_psi_out.device_ptr_mut(stream);
                cublasZcopy_v2(
                    handle, n_pw_i32,
                    psi_lb as *const _, 1,
                    s_out_mut as *mut _, 1,
                )
                .result()
                .map_err(Error::Blas)?;
                unsafe {
                    apply_s_times()
                        .psi_dev(&s_psi_in)
                        .spsi_dev(&mut s_psi_out)
                        .vnl_data(vnl_data)
                        .n_bands(1_i32)
                        .n_pw(n_pw_i32)
                        .blas(blas)
                        .stream(stream)
                        .call()?;
                }

                // dot = ⟨S·ψ_locked | ψ_unconv_col⟩
                let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                let (s_out_ptr, _) = s_psi_out.device_ptr(stream);
                cublasZdotc_v2(
                    handle, n_pw_i32,
                    s_out_ptr as *const _, 1,
                    unrot_col as *const _, 1,
                    &mut dot as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;

                // ψ_unconv_col -= dot · ψ_locked
                let neg_dot = CudaComplex { x: -dot.x, y: -dot.y };
                let (unrot_mut, _) = psi_unconv_new.device_ptr_mut(stream);
                let unrot_mut_col =
                    (unrot_mut as *mut CudaComplex).add(u * n_pw);
                cublasZaxpy_v2(
                    handle, n_pw_i32,
                    &neg_dot as *const _ as *const _,
                    psi_lb as *const _, 1,
                    unrot_mut_col as *mut _, 1,
                )
                .result()
                .map_err(Error::Blas)?;
            }

            // The rotated column is already S-normalised from ZHEGVD
            // (X^H·S_sub·X = I, preserved by ψ_u·X rotation). S-orth
            // against locked bands perturbs the norm only slightly.
            // Do NOT re-normalise with L2 (cublasDznrm2) — L2-normalising
            // inflates identity norms from ~0.37 to ~1.0 per band and
            // destabilises the soft density, triggering an SCF cascade.
        }

        core::mem::drop(s_psi_in);
        core::mem::drop(s_psi_out);
    }

    // ------------------------------------------------------------------
    // Step 12: Scatter output — locked bands verbatim, rotated unconv
    // ------------------------------------------------------------------
    let mut psi_out: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;

    // Copy locked bands verbatim
    {
        let (psi_ptr, _) = psi_dev.device_ptr(stream);
        let (psi_out_mut, _) = psi_out.device_ptr_mut(stream);
        for &b in &locked_idx {
            cublasZcopy_v2(
                handle, n_pw_i32,
                (psi_ptr as *const CudaComplex).add(b * n_pw) as *const _, 1,
                (psi_out_mut as *mut CudaComplex).add(b * n_pw) as *mut _, 1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // Scatter rotated unconverged columns
    {
        let (psi_unconv_ptr, _) = psi_unconv_new.device_ptr(stream);
        let (psi_out_mut, _) = psi_out.device_ptr_mut(stream);
        for (u, &b) in unconv_idx.iter().enumerate() {
            cublasZcopy_v2(
                handle, n_pw_i32,
                (psi_unconv_ptr as *const CudaComplex).add(u * n_pw) as *const _, 1,
                (psi_out_mut as *mut CudaComplex).add(b * n_pw) as *mut _, 1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    }

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------
    *DAVIDSON_LAST_DIAG.lock().unwrap() = Some(DavidsonDiagnostic {
        n_locked,
        n_unconverged: n_unconv,
        locked_indices: locked_idx,
        unconv_indices: unconv_idx,
        residual_norms_sinv: ResidualSInvNorm::new(residual_norms_sinv.clone()),
        max_residual_sinv: max_res,
        lock_tol,
        eigenvalue_deltas: vec![0.0_f64; n_bands],
    });

    Ok(DavidsonResult {
        psi_out,
        eigenvalues,
        n_locked,
        residual_norms_sinv: ResidualSInvNorm::new(residual_norms_sinv),
        n_outer_iterations: 0,
    })
}

// ======================================================================
// Inner-loop convergence check (block-level Davidson)
// ======================================================================

/// Result of convergence check for one band in the inner Davidson loop.
///
/// Reference: hamiltonian.f90:1178-1228
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct BandConvStatus {
    /// Band converged by absolute tolerance.
    pub converged: bool,
    /// Band hit optional stop condition (stagnation or relative break).
    pub opt_stopped: bool,
}

/// Check convergence after subspace rotation in the inner Davidson loop.
///
/// Implements CASTEP's inner-loop exit logic (hamiltonian.f90:1178-1228):
///
/// (a) **Absolute tolerance**: band converges when
///     `|ΔE| < max(tol_abs, 2*|new_eig|*EPS)`.
///     The EPS guard prevents false-negatives for large eigenvalues where
///     machine precision exceeds a fixed tol_abs.
///
/// (b) **Relative break condition**:
///     - First step: store `|ΔE|` as `break_cond_tol` (no convergence yet).
///     - Subsequent steps: if `tol_rel > 0` and `|ΔE| < break_cond_tol * tol_rel`
///       → band converged, opt_stopped.
///     - Fallback (tol_rel ≤ 0): if `< break_cond_tol * 0.3` and not last outer
///       iteration → opt_stopped only. The 0.3 factor is CASTEP's heuristic:
///       improvement slowed to <30% of first-step improvement → stagnation.
///
/// (c) **Uphill detection**: if `prev - new < -100·max(EPS, EPS·|prev|)`,
///     the eigenvalue increased (numerical noise). Overrides absolute tolerance
///     — band is NOT marked converged.
///
/// # Arguments
/// - `prev_eig`: eigenvalue before subspace rotation
/// - `new_eig`: eigenvalue after subspace rotation
/// - `tol_abs`: absolute convergence tolerance (Hartree)
/// - `tol_rel`: relative convergence tolerance (ratio, dimensionless).
///   Pass 0.0 to use the CASTEP 0.3 stagnation heuristic.
/// - `break_cond_tol`: accumulator for first-step |ΔE|, used as reference
///   for stagnation detection. Updated in-place on first step.
/// - `is_first_step`: `true` for the first convergence check (sets baseline).
/// - `outer_iter`: current outer Davidson iteration index (0-based).
/// - `max_outer_iter`: maximum outer iterations.
#[doc(hidden)]
pub fn check_inner_convergence(
    prev_eig: f64,
    new_eig: f64,
    tol_abs: f64,
    tol_rel: f64,
    break_cond_tol: &mut f64,
    is_first_step: bool,
    outer_iter: usize,
    max_outer_iter: usize,
) -> BandConvStatus {
    let delta_e = (prev_eig - new_eig).abs();
    let eps_guard = 2.0 * new_eig.abs() * f64::EPSILON;
    let threshold = tol_abs.max(eps_guard);

    let mut converged = false;
    let mut opt_stopped = false;

    // (a) Absolute tolerance check
    if delta_e < threshold {
        converged = true;
    }

    // (b) Relative break condition (CASTEP hamiltonian.f90:563-589)
    if is_first_step {
        *break_cond_tol = delta_e;
    } else if tol_rel > 0.0 && delta_e < *break_cond_tol * tol_rel {
        converged = true;
        opt_stopped = true;
    } else if tol_rel <= 0.0 && outer_iter + 1 < max_outer_iter {
        if delta_e < *break_cond_tol * 0.3 {
            opt_stopped = true;
        }
    }

    // (c) Uphill detection — eigenvalue went UP → numerical noise
    //     Override convergence: band is NOT converged AND not stopped.
    //     CASTEP hamiltonian.f90:593-597 resets both flags.
    let uphill_threshold = -100.0 * (f64::EPSILON).max(f64::EPSILON * prev_eig.abs());
    if prev_eig - new_eig < uphill_threshold {
        converged = false;
        opt_stopped = false;
    }

    BandConvStatus { converged, opt_stopped }
}

// ======================================================================
// Outer Davidson loop with subspace diagonalization (Phase 1B)
// ======================================================================

/// Conditionally emit diagnostic output inside `davidson_diagonalise`.
///
/// User-facing progress message: always prints.  Used for `[davidson]` prefix
/// lines (outer iteration, block loop, convergence).
macro_rules! davidson_diag {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

/// Internal diagnostic: compiles to nothing unless `feature = "scf_diag"`.
#[cfg(feature = "scf_diag")]
macro_rules! diag_detail {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}
#[cfg(not(feature = "scf_diag"))]
macro_rules! diag_detail {
    ($($arg:tt)*) => {};
}

/// Run the outer Davidson loop with subspace diagonalization.
///
/// Implements the outer loop structure from CASTEP's
/// `hamiltonian_diagonalise_ks` (hamiltonian.f90:947-1018):
///
/// 1. Copy psi_init → eigenvectors buffer
/// 2. Set H_correct = false
/// 3. `for iteration in 0..max_outer_iter`:
///    a. If all bands converged: break
///    b. If !H_correct: compute H·ψ via `apply_full_hamiltonian`
///    c. Save previous eigenvalues
///    d. Subspace diagonalization via `diagonalise_subspace` (full n_bands)
///    e. Rotate ψ via ZHEGVD eigenvector matrix
///    f. Convergence check: |prev_eig - new_eig| < max(tol_abs, 2*|new_eig|*EPS)
///    g. Set H_correct = true (H·ψ was just computed)
///    h. After rotation, H·ψ is stale → H_correct = false
///
/// # Notes
/// - The preconditioner call and inner block loop are NOT implemented yet
///   (TODO for future tasks).
/// - Uses the existing `diagonalise_subspace` for the full n_bands × n_bands
///   subspace (not block-by-block).
///
/// # Safety
// ---------------------------------------------------------------------------
// GPU helper: per-band kinetic energies for TPA mean_ek
// ---------------------------------------------------------------------------

const BAND_EK_KERNEL: &str = r#"
extern "C" __global__ void band_ek(
    double* ek_out,
    const double2* psi,
    const double* kinetic,
    int n_pw,
    int n_bands
) {
    int b = blockIdx.x;
    if (b >= n_bands) return;

    int tid = threadIdx.x;
    extern __shared__ double s_sum[];

    double sum = 0.0;
    for (int g = tid; g < n_pw; g += blockDim.x) {
        double2 p = psi[b * n_pw + g];
        sum += (p.x * p.x + p.y * p.y) * kinetic[g];
    }

    s_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0) ek_out[b] = s_sum[0];
}
"#;

/// Compute per-band kinetic energies on GPU.
///
/// `ek[b] = Σ_G |ψ_b(G)|² · T(G)` where `T(G) = 0.5|k+G|²`.
///
/// Uses one CUDA block per band with shared-memory reduction.
///
/// This is the energy scale `ek(b)` that CASTEP averages to get `mean_ek`
/// in `hamiltonian.f90:348`:
///
/// ```fortran
/// ek(b) = Σ_G |ψ_b(G)|² · 0.5|k+G|²
/// mean_ek = sum(ek(1:nbands)) / nbands
/// ```
///
/// The Rust code previously used `Σ_G T(G) / n_pw` (uniform PW average),
/// which overestimates `mean_ek` by 3-5× for typical wavefunction
/// distributions, causing the TPA preconditioner to under-damp high-G
/// components and explode on cold-start initial guesses.
fn compute_band_kinetic_energies(
    psi_dev: &PwCoefficients,
    kinetic_dev: &KineticPreconditioner,
    n_pw: usize,
    n_bands: usize,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> Result<Vec<f64>, Error> {
    let ptx = compile_ptx(BAND_EK_KERNEL).map_err(|e| Error::Nvrtc(e.to_string()))?;
    let module = ctx.load_module(ptx).map_err(Error::Cuda)?;
    let kernel = module.load_function("band_ek").map_err(Error::Cuda)?;

    let mut ek_dev: CudaSlice<f64> = stream.alloc_zeros(n_bands).map_err(Error::Cuda)?;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;
    const BLOCK_DIM: u32 = 256;
    let shared_mem_bytes = BLOCK_DIM as usize * std::mem::size_of::<f64>();

    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&mut ek_dev)
            .arg(&**psi_dev)
            .arg(&**kinetic_dev)
            .arg(&n_pw_i32)
            .arg(&n_bands_i32)
            .launch(LaunchConfig {
                grid_dim: (n_bands as u32, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: shared_mem_bytes as u32,
            })
            .map(|_| ())
    }
    .map_err(Error::Cuda)?;

    let ek_host: Vec<f64> = stream.clone_dtoh(&ek_dev).map_err(Error::Cuda)?;
    Ok(ek_host)
}

/// All device pointers must be valid and of sufficient size. `psi_init` must
/// be S-orthonormal (column-major, n_bands × n_pw).
#[builder]
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn davidson_diagonalise(
    psi_init: &PwCoefficients,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    vnl_data: &VnlBatchData,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    tol_abs: f64,
    /// CASTEP convergence_tols(2): relative convergence tolerance (Hartree).
    /// When > 0, bands with |ΔE| < tol_rel * break_cond_tol are marked as
    /// both converged AND stopped (hamiltonian.f90:563-589).  Default 0.0 (off).
    #[builder(default = 0.0)]
    tol_rel: f64,
    max_outer_iter: usize,
    min_outer_iter: usize,
    blas: &BlasHandle,
    solver: &SolverHandle,
    kernels: &CudaKernelSet,
    tpa_preconditioner: &TpaPreconditioner,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
    /// Use real symmetric DSYEVD for gamma-point (hamiltonian.f90:480-481).
    /// Matches CASTEP `algor_diagonalise(..., 'S')`.  Default: false (ZHEEVD).
    #[builder(default = false)]
    gamma_point: bool,
) -> Result<DavidsonResult, Error> {
    let n_elem = n_bands * n_pw;
    let n_pw_i32 = n_pw as i32;
    let handle = blas.raw_handle();

    // ------------------------------------------------------------------
    // Persistent GPU buffers
    // ------------------------------------------------------------------
    let mut psi_dev = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    stream
        .memcpy_dtod(&**psi_init, &mut psi_dev.0)
        .map_err(Error::Cuda)?;

    let mut hpsi_dev = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
    let grid_alloc = n_bands * grid_size;
    let mut grid_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // Subspace diagonalization GPU buffers (reused across iterations)
    let mut eig_dev: CudaSlice<f64> =
        stream.alloc_zeros(n_bands).map_err(Error::Cuda)?;
    let mut info_dev: CudaSlice<i32> =
        stream.alloc_zeros(1).map_err(Error::Cuda)?;

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    let mut eigenvalues = vec![0.0_f64; n_bands];
    let mut band_converged = vec![false; n_bands];
    let mut h_correct = false;

    let mut n_outer_completed: usize = 0;

    // Per-band break_cond_tol accumulator for inner convergence check.
    // Initialized to 0.0; first inner step sets it to |prev_eig - new_eig|.
    let mut break_cond_tols = vec![0.0_f64; n_bands];

    davidson_diag!("[davidson] start: n_bands={n_bands} n_pw={n_pw} tol_abs={tol_abs:.1e} max_outer={max_outer_iter}");

    // ------------------------------------------------------------------
    // Preconditioner preparation (once per SCF step)
    // ------------------------------------------------------------------
    // Download kinetic energies from GPU (constant across outer iterations)
    let kinetic_host: Vec<f64> = stream.clone_dtoh(&**kinetic_dev).map_err(Error::Cuda)?;
    // Compute per-band kinetic energies on GPU and average over bands.
    // This matches CASTEP hamiltonian.f90:348: mean_ek = sum(ek(1:nbands)) / nbands
    // where ek(b) = Σ_G |ψ_b(G)|² · 0.5|k+G|².
    // The old code averaged over PW (Σ_G T(G) / n_pw), which overestimates
    // mean_ek by 3-5× and causes TPA under-damping on cold-start guesses.
    let band_ek = compute_band_kinetic_energies(
        &psi_dev, kinetic_dev, n_pw, n_bands, stream, ctx,
    )?;
    let mean_ek = band_ek.iter().sum::<f64>() / band_ek.len() as f64;
    // Diagnostic: report mean_ek so we can verify the per-band average matches
    // CASTEP convention.  The old per-PW average (kinetic_host sum / n_pw) is
    // printed alongside for comparison.
    {
        let _mean_ek_pw = kinetic_host.iter().sum::<f64>() / kinetic_host.len() as f64;
        let _ek_min = band_ek.iter().cloned().fold(f64::INFINITY, f64::min);
        let _ek_max = band_ek.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        diag_detail!(
            "[mean_ek] per-band={mean_ek:.6} Ha (min={_ek_min:.4}, max={_ek_max:.4})  per-PW={_mean_ek_pw:.6} Ha  ratio={ratio:.3}",
            ratio = _mean_ek_pw / mean_ek
        );
    }

    // TPA preconditioner R(G) vector on GPU (reused across all blocks/iterations)
    let r_vector = compute_r_vector(kinetic_dev, mean_ek, n_pw, stream)?;

    // USPP preconditioner: download beta_g and q_matrix from GPU, assemble
    // r_beta_per_ion and q_rcq on CPU. For NCPP-only systems this produces
    // empty matrices and the NL correction is a cheap no-op.
    let ion_n_expanded: Vec<usize> = vnl_data.entries.iter()
        .map(|e| e.n_expanded as usize)
        .collect();
    let mixture_weights: Vec<f64> = vec![1.0; vnl_data.entries.len()];

    let mut beta_g_per_ion: Vec<Array2<Complex64>> = Vec::with_capacity(vnl_data.entries.len());
    let mut q_matrices: Vec<Vec<f64>> = Vec::with_capacity(vnl_data.entries.len());
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded as usize;
        if ne > 0 {
            let beta_host: Vec<CudaComplex> = stream.clone_dtoh(&entry.beta_g)
                .map_err(Error::Cuda)?;
            // GPU stores beta_g flat as row-major (ne, n_pw): beta_flat[n*n_pw+G] = beta(n,G).
            // cuBLAS GEMM reads it as column-major (n_pw, ne) with lda=n_pw:
            //   data[G + n*n_pw] = beta(n,G)  ← correct (addition commutes).
            //
            // clone_dtoh preserves the flat layout, so beta_host[n*n_pw+G] = beta(n,G).
            // To get beta[[G, n]] = beta(n,G) in ndarray we must reshape as (ne, n_pw)
            // (restoring the original row-major layout) then transpose to (n_pw, ne):
            //   beta_transposed[[G, n]] = beta_orig[[n, G]] = beta_host[n*n_pw+G] = beta(n,G).
            //
            // The old reshape Array2::from_shape_vec((n_pw, ne), ...) read
            //   beta[[G, n]] = beta_host[G*ne + n] ≠ beta_host[n*n_pw + G],
            // silently transposing the projector data and corrupting all downstream
            // C_matrix, R_beta, and Q_RCQ computations in prepare_preconditioner.
            let beta_correct = Array2::from_shape_vec((ne, n_pw),
                beta_host.iter().map(|c| Complex64::new(c.x, c.y)).collect()
            ).expect("beta_g shape (ne, n_pw) mismatch");
            beta_g_per_ion.push(beta_correct.t().to_owned());
            let q_host: Vec<CudaComplex> = stream.clone_dtoh(&entry.q_matrix)
                .map_err(Error::Cuda)?;
            q_matrices.push(q_host.iter().map(|c| c.x).collect());
        } else {
            beta_g_per_ion.push(Array2::zeros((n_pw, 0)));
            q_matrices.push(Vec::new());
        }
    }

    let precon_prep = prepare_preconditioner()
        .pw_ek(&kinetic_host)
        .mean_ek(mean_ek)
        .n_pw(n_pw)
        .beta_g_per_ion(&beta_g_per_ion)
        .q_matrices(&q_matrices)
        .ion_n_expanded(&ion_n_expanded)
        .mixture_weights(&mixture_weights)
        .maybe_stream(Some(stream))
        .call()?;

    // ------------------------------------------------------------------
    // Outer loop (CASTEP hamiltonian.f90:293–619)
    //
    // CASTEP recomputes H·ψ EVERY outer iteration (line 306), then runs
    // full subspace diagonalization (line 319). If eigenvalue changes from
    // fresh H·ψ exceed the absolute tolerance, bands are un-converged
    // (lines 325–329).  The outer loop exits only when ALL bands are
    // converged for TWO consecutive iterations (lines 291–296).
    //
    // Fresh H·ψ each iteration is critical: it reveals eigenvector
    // stagnation that eigenvalue-delta alone misses (Rayleigh quotient
    // converges quadratically in eigenvector error, so |Δλ| < tol can
    // hold while ‖r‖ ≫ tol).
    // ------------------------------------------------------------------
    // Superspace dimensions (constant across outer iterations)
    // CASTEP hamiltonian.f90:197-205 — nblock = floor(2*sqrt(n_bands)).
    // Only round to even for gamma-point calculations (hamiltonian.f90:203-205).
    // For non-gamma k-points (n_bands=160): nblock=25 in CASTEP vs 26 in Rust.
    // Different block groupings change ZHEEVD eigenvalue ordering → different
    // convergence behavior for near-degenerate bands near the Fermi level.
    let nblock_base = (2.0 * (n_bands as f64).sqrt()).floor() as usize;
    let nblock = if gamma_point && nblock_base % 2 == 1 {
        nblock_base + 1
    } else {
        nblock_base
    };
    let superspace_size = 6_usize;
    let superspace_max_bands = superspace_size * nblock;
    let super_alloc = n_pw * superspace_max_bands;

    // ------------------------------------------------------------------
    // BetaPhiCache: persists β^H·ψ projections across outer iterations.
    // Allocated once outside the outer loop so that cache entries for
    // converged (unchanged) bands survive across iterations.
    let mut beta_phi_cache = BetaPhiCache::new(vnl_data, n_bands, stream)?;

    #[allow(unused_assignments)]
    for iteration in 0..max_outer_iter {
        let n_conv = band_converged.iter().filter(|&&c| c).count();
        davidson_diag!("[davidson] outer iter {iteration}: {n_conv}/{n_bands} converged");


        // CASTEP hamiltonian.f90:306 — recompute H·ψ every outer iteration.
        // This is the implicit residual check: fresh H·ψ reveals true
        // eigenvalue changes and prevents premature convergence from
        // stale subspace-rotated H·ψ.
        if !h_correct || iteration > 0 {
            davidson_diag!("[davidson] computing H·psi...");
            unsafe {
                apply_full_hamiltonian()
                    .psi_dev(&psi_dev)
                    .v_eff_dev(v_eff_dev)
                    .kinetic_dev(kinetic_dev)
                    .fft_idx_dev(fft_idx_dev)
                    .n_pw(n_pw)
                    .n_bands(n_bands)
                    .grid_size(grid_size)
                    .inv_ntotal(inv_ntotal)
                    .fft_plan(fft_plan)
                    .hpsi_dev(&mut hpsi_dev)
                    .grid_dev(&mut grid_dev)
                    .vnl_data(vnl_data)
                    .blas(blas)
                    .kernels(kernels)
                    .stream(stream)
                    .maybe_beta_phi_cache(&mut beta_phi_cache)
                    .call()?;
            }

            h_correct = true;
            davidson_diag!("[davidson] H·psi done");

            // Diagnostic: decompose H_sub[0,0] = T_contrib + V_loc+V_NL_contrib
            #[cfg(feature = "scf_diag")]
            if eigenvalues.iter().all(|&e| e == 0.0) {
                // First outer iteration: compute T_contrib from psi on CPU
                let psi_band0: Vec<CudaComplex> = {
                    let mut tmp = stream.alloc_zeros::<CudaComplex>(n_pw).map_err(Error::Cuda)?;
                    let handle = blas.raw_handle();
                    let (psi_ptr, _) = psi_dev.device_ptr(stream);
                    cublasZcopy_v2(handle, n_pw as i32,
                        psi_ptr as *const _, 1,
                        tmp.device_ptr_mut(stream).0 as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    stream.clone_dtoh(&tmp).map_err(Error::Cuda)?
                };
                let hpsi_band0: Vec<CudaComplex> = {
                    let mut tmp = stream.alloc_zeros::<CudaComplex>(n_pw).map_err(Error::Cuda)?;
                    let handle = blas.raw_handle();
                    let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
                    cublasZcopy_v2(handle, n_pw as i32,
                        hpsi_ptr as *const _, 1,
                        tmp.device_ptr_mut(stream).0 as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    stream.clone_dtoh(&tmp).map_err(Error::Cuda)?
                };
                let mut t_contrib = 0.0f64;
                let mut h_full_re = 0.0f64;
                let mut psi_norm = 0.0f64;
                for g in 0..n_pw {
                    let p = psi_band0[g];
                    let h = hpsi_band0[g];
                    let k = kinetic_host[g];
                    psi_norm += p.x * p.x + p.y * p.y;
                    t_contrib += (p.x * p.x + p.y * p.y) * k;
                    h_full_re += p.x * h.x + p.y * h.y; // Re(conj(p) * h)
                }
                let v_contrib = h_full_re - t_contrib;
                eprintln!(
                    "[H diag] band 0: psi_norm={:.6e} T_contrib={:.6} V_contrib={:.6} H_full={:.6}",
                    psi_norm, t_contrib, v_contrib, h_full_re,
                );

                // Separate V_loc from V_NL: re-compute T+V_loc only
                // (no V_NL) using a fresh temp buffer
                let mut hpsi_tvloc_dev = PwCoefficients::new(
                    stream.alloc_zeros::<CudaComplex>(n_elem).map_err(Error::Cuda)?);
                unsafe {
                    apply_v_loc_hamiltonian()
                        .psi_dev(&psi_dev)
                        .hpsi_dev(&mut hpsi_tvloc_dev)
                        .grid_dev(&mut grid_dev)
                        .kinetic_dev(kinetic_dev)
                        .fft_idx_dev(fft_idx_dev)
                        .v_eff_dev(v_eff_dev)
                        .n_pw(n_pw as i32)
                        .n_bands(n_bands as i32)
                        .grid_size(grid_size as i32)
                        .ngx(fft_plan.nx())
                        .ngy(fft_plan.ny())
                        .ngz(fft_plan.nz())
                        .inv_ntotal(inv_ntotal)
                        .fft_plan(fft_plan)
                        .kernels(kernels)
                        .stream(stream)
                        .call()?;
                }
                let hpsi_tvloc_band0: Vec<CudaComplex> = {
                    let mut tmp = stream.alloc_zeros::<CudaComplex>(n_pw).map_err(Error::Cuda)?;
                    let handle = blas.raw_handle();
                    let (hptr, _) = hpsi_tvloc_dev.device_ptr(stream);
                    cublasZcopy_v2(handle, n_pw as i32,
                        hptr as *const _, 1,
                        tmp.device_ptr_mut(stream).0 as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    stream.clone_dtoh(&tmp).map_err(Error::Cuda)?
                };
                let mut h_tvloc_re = 0.0f64;
                for g in 0..n_pw {
                    let p = psi_band0[g];
                    let h = hpsi_tvloc_band0[g];
                    h_tvloc_re += p.x * h.x + p.y * h.y;
                }
                let v_loc_contrib = h_tvloc_re - t_contrib;
                let v_nl_contrib = h_full_re - h_tvloc_re;
                eprintln!(
                    "[H diag] V_loc={:.6}  V_NL={:.6}  H_TVloc={:.6}",
                    v_loc_contrib, v_nl_contrib, h_tvloc_re,
                );
                // Print first 5 GPU hpsi (T+V_loc only) for comparison with CPU
                eprintln!(
                    "[H diag] GPU hpsi_TVloc[0..5]: {:?}",
                    (0..5).map(|g| {
                        let c = hpsi_tvloc_band0[g];
                        (c.x, c.y)
                    }).collect::<Vec<_>>()
                );
            }

            // Compute initial eigenvalue estimates via Rayleigh quotient
            // ε_b = Re⟨ψ_b|H|ψ_b⟩ for ALL bands. This fills eigenvalues[]
            // with physically correct values before the block loop starts.
            // Without this, bands in blocks 1+ start with e=0, causing the
            // preconditioner to produce H|ψ⟩ (full Hamiltonian) instead of
            // the residual (H−ε)|ψ⟩, contaminating search directions and
            // collapsing unoccupied-band eigenvalues to zero via ZHEGVD's
            // lowest-first sorting.
            {
                let (psi_ptr, _) = psi_dev.device_ptr(stream);
                let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
                for b in 0..n_bands {
                    let psi_b = (psi_ptr as *const CudaComplex).add(b * n_pw);
                    let hpsi_b = (hpsi_ptr as *const CudaComplex).add(b * n_pw);
                    let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                    cublasZdotc_v2(
                        handle,
                        n_pw_i32,
                        psi_b as *const _,
                        1,
                        hpsi_b as *const _,
                        1,
                        &mut dot as *mut _ as *mut _,
                    )
                    .result()
                    .map_err(Error::Blas)?;
                    eigenvalues[b] = dot.x; // Real part = ⟨ψ_b|H|ψ_b⟩
                }
            }
            davidson_diag!(
                "[davidson] initial Rayleigh eigenvalues: [{:.6}, ..., {:.6}]",
                eigenvalues[0],
                eigenvalues[n_bands - 1]
            );
        }

        // ---- Save Rayleigh eigenvalues before full subspace diagonalization ----
        // CASTEP hamiltonian.f90:313-315 — save eigenvalues BEFORE wave_diagonalise
        // so we can detect which bands moved too much from the fresh H·ψ (A4).
        let rayleigh_eigenvalues = eigenvalues.clone();

        // ---- A1: Full n_bands subspace diagonalization (CASTEP wave_diagonalise) ----
        // CASTEP hamiltonian_diagonalise_ks line 1013 (the variant actually called,
        // 33 times per profile):
        //   call wave_diagonalise(eigenvectors,H_eigenvectors,nk,ns,eigenvalues)
        //
        // NOTE: line 1070 has a DIFFERENT wave_diagonalise call that is COMMENTED
        // OUT (inside the band loop).  Only the outer-loop call at line 1013 is
        // active.  Both _slice (line 323) and _ks (line 1013) run A1 before the
        // block loop — this is correct.
        //
        // Builds H_sub = ψ^H·Hψ for ALL n_bands, solves the standard EVP, and
        // rotates all eigenvectors into the globally optimal eigenbasis.
        // Rayleigh quotients alone are poor eigenvalue estimates for cold-start
        // wavefunctions with similar character across blocks — cross-band mixing
        // is never captured.
        let mut psi_full_rotated = PwCoefficients::new(
            stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
        let mut hpsi_full_rotated = PwCoefficients::new(
            stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
        unsafe {
            diagonalise_subspace()
                .psi_block(&psi_dev)
                .hpsi_block(&hpsi_dev)
                .vnl_data(vnl_data)
                .k(n_bands)
                .n_pw(n_pw)
                .blas(blas)
                .solver(solver)
                .stream(stream)
                .eigenvalues_out(&mut eigenvalues)
                .eig_dev(&mut eig_dev)
                .info_dev(&mut info_dev)
                .psi_rotated(&mut psi_full_rotated)
                .hpsi_rotated(&mut hpsi_full_rotated)
                .gamma_point(gamma_point)
                .call()?;
        }
        stream
            .memcpy_dtod(&*psi_full_rotated, &mut psi_dev.0)
            .map_err(Error::Cuda)?;
        stream
            .memcpy_dtod(&*hpsi_full_rotated, &mut hpsi_dev.0)
            .map_err(Error::Cuda)?;

        // A1 rotates ALL bands' psi via subspace diagonalization.
        // All cached β^H·ψ projections are now stale.
        beta_phi_cache.invalidate_all();

        davidson_diag!(
            "[davidson] full subspace diag: eigenvalues [{:.6}, ..., {:.6}]",
            eigenvalues[0],
                eigenvalues[n_bands - 1]
            );

        // ---- D1: S-norm diagnostic (gated — CUDA_LAUNCH_BLOCKING investigation) ----
        #[cfg(feature = "scf_diag")]
        {
            let mut s_in = PwCoefficients::new(
                stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
            let mut s_out = PwCoefficients::new(
                stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
            let (psi_ptr, _) = psi_dev.device_ptr(stream);
            let (s_in_mut, _) = s_in.device_ptr_mut(stream);
            let (s_out_mut, _) = s_out.device_ptr_mut(stream);

            let mut max_deviation: f64 = 0.0;
            let n_check = n_bands.min(10); // sample first 10 bands

            for b in 0..n_check {
                let psi_b = (psi_ptr as *const CudaComplex).add(b * n_pw);
                // Copy psi_b -> s_in, s_out (s_out will become S·psi_b)
                cublasZcopy_v2(handle, n_pw_i32,
                    psi_b as *const _, 1, s_in_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                cublasZcopy_v2(handle, n_pw_i32,
                    psi_b as *const _, 1, s_out_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                // s_out = S · psi_b
                apply_s_times()
                    .psi_dev(&s_in)
                    .spsi_dev(&mut s_out)
                    .vnl_data(vnl_data)
                    .n_bands(1_i32)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .call()?;
                let (s_out_ptr, _) = s_out.device_ptr(stream);
                let mut s_norm = CudaComplex { x: 0.0, y: 0.0 };
                cublasZdotc_v2(handle, n_pw_i32,
                    psi_b as *const _, 1,
                    s_out_ptr as *const _, 1,
                    &mut s_norm as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                let dev = (s_norm.x - 1.0).abs();
                if dev > max_deviation { max_deviation = dev; }
            }
            diag_detail!(
                "[Diag-D1] after A1 ZHEGVD: max |S-norm - 1| = {:.3e} (checked {} bands)",
                max_deviation, n_check
            );
        }

        // ---- A4: Convergence invalidation after fresh H·ψ + subspace rotation ----
        // CASTEP hamiltonian.f90:325-329 — after recomputing H·ψ and running full
        // subspace diagonalization each outer iteration, check if eigenvalue changes
        // exceed the absolute tolerance. Bands that moved too much are UN-CONVERGED.
        // Without this, once a band is marked converged it stays converged even if
        // it drifts (e.g. from V_eff changes in SCF or subspace rotation artifacts).
        for b in 0..n_bands {
            let diff = (rayleigh_eigenvalues[b] - eigenvalues[b]).abs();
            let threshold = tol_abs.max(2.0 * eigenvalues[b].abs() * f64::EPSILON);
            if diff >= threshold {
                if band_converged[b] {
                    davidson_diag!(
                        "[davidson]   band {b}: un-converged — eigenvalue changed {:.3e} > threshold {:.3e}",
                        diff, threshold
                    );
                }
                band_converged[b] = false;
                break_cond_tols[b] = 0.0; // reset stagnation accumulator
            }
        }

        // Step c: save previous eigenvalues for block-loop convergence check
        let prev_eigenvalues = eigenvalues.clone();
        davidson_diag!("[davidson] prev eigenvalues: [{:.6}, ..., {:.6}]",
                  prev_eigenvalues[0], prev_eigenvalues[n_bands-1]);

        // ------------------------------------------------------------------
        // Block loop with superspace management
        //
        // CASTEP hamiltonian.f90:1019-1063 — block loop over groups of nblock
        // bands. Each unconverged block runs an inner Davidson loop (residual,
        // S-orthogonalize, H·search, extend superspace, subspace diagonalization).
        // ------------------------------------------------------------------
        // CASTEP hamiltonian.f90:197 — nblock = floor(2*sqrt(n_bands))
        // (nblock, superspace_max_bands, super_alloc computed once before outer loop)

        // Allocate per-iteration superspace buffers (reused across blocks)
        let mut super_wvfn = PwCoefficients::new(
            stream.alloc_zeros(super_alloc).map_err(Error::Cuda)?);
        let mut h_super_wvfn = PwCoefficients::new(
            stream.alloc_zeros(super_alloc).map_err(Error::Cuda)?);

        // CASTEP hamiltonian.f90:403-408 — separate "slice" workspace.
        // Holds at most nblock columns (compacted subset of current block).
        let slice_alloc = n_pw * nblock;
        let mut slice_wvfn = PwCoefficients::new(
            stream.alloc_zeros(slice_alloc).map_err(Error::Cuda)?);
        let mut slice_h_wvfn = PwCoefficients::new(
            stream.alloc_zeros(slice_alloc).map_err(Error::Cuda)?);

        // CPU-side dense Hermitian super_hamiltonian matrix
        let mut super_hamiltonian = vec![
            CudaComplex { x: 0.0, y: 0.0 };
            superspace_max_bands * superspace_max_bands
        ];

        davidson_diag!("[davidson] block loop: nblock={nblock} superspace_size={superspace_size}");

        for block_start in (0..n_bands).step_by(nblock) {
            let current_nblock = nblock.min(n_bands - block_start);

            // Skip if all bands in this block are converged
            if (block_start..block_start + current_nblock)
                .all(|b| band_converged[b])
            {
                davidson_diag!("[davidson]   block {block_start}..{}: skipped (all converged)", block_start+current_nblock);
                continue;
            }

            davidson_diag!("[davidson]   block {block_start}..{}: current_nblock={current_nblock}", block_start+current_nblock);

            // CASTEP hamiltonian.f90:629-646 — after compaction, the active
            // workspace columns hold a subset of the original block bands.
            // active_indices[compacted_pos] = original_block_relative_index.
            // Global arrays (eigenvalues, psi_dev, hpsi_dev, band_converged,
            // break_cond_tols) always use ORIGINAL band indices throughout.
            // Initialised here (before Stage 1) so that the copy loops can use
            // the correct source/destination offsets from the start.
            let mut active_indices: Vec<usize> = (0..current_nblock).collect();

            // Copy block eigenvectors -> super_wvfn (first current_nblock bands)
            // Copy block H.psi -> h_super_wvfn
            {
                let (psi_ptr, _) = psi_dev.device_ptr(stream);
                let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
                let (super_mut, _) = super_wvfn.device_ptr_mut(stream);
                let (h_super_mut, _) = h_super_wvfn.device_ptr_mut(stream);

                // CASTEP hamiltonian.f90:407-408 — wave_copy(super_wvfn, slice)
                // copies ALL current_nblock columns using sequential 1:1 mapping.
                // After C3-07 slice workspace, super_wvfn is APPEND-ONLY (never
                // compacted) — column i always maps to band block_start + i.  The
                // active_bands() iterator is for compacted workspace access (slice);
                // super_wvfn uses block_bands() for sequential mapping.
                // CHECKLIST: D13-02 revision (2026-06-08), C2-04 revision.
                for (col, gi) in block_bands(block_start, current_nblock) {
                    let src_off = gi * n_pw;
                    let dst_off = col * n_pw;
                    cublasZcopy_v2(
                        handle,
                        n_pw_i32,
                        (psi_ptr as *const CudaComplex).add(src_off) as *const _, 1,
                        (super_mut as *mut CudaComplex).add(dst_off) as *mut _, 1,
                    )
                    .result()
                    .map_err(Error::Blas)?;
                    cublasZcopy_v2(
                        handle,
                        n_pw_i32,
                        (hpsi_ptr as *const CudaComplex).add(src_off) as *const _, 1,
                        (h_super_mut as *mut CudaComplex).add(dst_off) as *mut _, 1,
                    )
                    .result()
                    .map_err(Error::Blas)?;
                }
            }

            // CASTEP hamiltonian_diagonalise_ks (lines 723-1422) — the
            // subroutine actually called for this run (profile confirms 33
            // calls).  This variant has NO cross-block conduction state
            // seeding.  Each block starts with a fresh superspace:
            // super_wvfn%nbands = current_nblock, superspace_index = 1 +
            // current_nblock.  Conduction states are block-local and built
            // from scratch within each block's inner loop.
            let mut superspace_index = current_nblock;

            // Compute initial super_hamiltonian: H_sub = super_wvfn^H · h_super_wvfn
            // for current_nblock columns (CASTEP: wave_dot_all over
            // super_wvfn%nbands = current_nblock columns).
            {
                let k = superspace_index;
                let mut h_init: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(k * k).map_err(Error::Cuda)?;
                unsafe {
                    blas.gemm_c64(
                        ZgemmConfig {
                            transa: op::C,
                            transb: op::N,
                            m: k as i32,
                            n: k as i32,
                            k: n_pw_i32,
                            alpha: CudaComplex { x: 1.0, y: 0.0 },
                            lda: n_pw_i32,
                            ldb: n_pw_i32,
                            beta: CudaComplex { x: 0.0, y: 0.0 },
                            ldc: k as i32,
                        },
                        &super_wvfn,
                        &h_super_wvfn,
                        &mut h_init,
                    )?;
                }
                // D2H: copy H_sub (k×k, k≈30 → ~7 KB) from GPU to CPU.
                //
                // super_hamiltonian is a CPU-side data structure by design — it serves
                // as a record of H in the superspace basis, used only for Hermitian fill
                // and diagonal reset after ZHEGVD. The ZHEGVD itself operates on the raw
                // wavefunction matrices (super_wvfn + H_super_wvfn) on GPU, never on
                // super_hamiltonian. This D2H is therefore inherent to the split design
                // and its cost is negligible: ~7 KB/transfer vs ~2 GB FFT+H per batch.
                let h_init_cpu: Vec<CudaComplex> = stream.clone_dtoh(&h_init).map_err(Error::Cuda)?;
                let diag_vals: Vec<String> = (0..k)
                    .map(|i| format!("{:.6}", h_init_cpu[i * (k + 1)].x))
                    .take(3)
                    .collect();
                davidson_diag!("[davidson]     initial H_sub diag[0..{}]: [{}]",
                    diag_vals.len().min(k), diag_vals.join(", "));

                // CPU dot-product cross-check (diagnostic-only): verify first column
                // ⟨psi|H·psi⟩ against GPU ZGEMM result. Downloads two full n_pw columns
                // (~10⁵ complex numbers) — gated behind scf_diag to avoid wasted D2H.
                #[cfg(feature = "scf_diag")]
                {
                    let sw_col0: Vec<CudaComplex> = {
                        let (super_ptr, _) = super_wvfn.device_ptr(stream);
                        let mut tmp = stream.alloc_zeros::<CudaComplex>(n_pw).map_err(Error::Cuda)?;
                        cublasZcopy_v2(handle, n_pw_i32,
                            super_ptr as *const _, 1,
                            tmp.device_ptr_mut(stream).0 as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                        stream.clone_dtoh(&tmp).map_err(Error::Cuda)?
                    };
                    let hw_col0: Vec<CudaComplex> = {
                        let (hsuper_ptr, _) = h_super_wvfn.device_ptr(stream);
                        let mut tmp = stream.alloc_zeros::<CudaComplex>(n_pw).map_err(Error::Cuda)?;
                        cublasZcopy_v2(handle, n_pw_i32,
                            hsuper_ptr as *const _, 1,
                            tmp.device_ptr_mut(stream).0 as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                        stream.clone_dtoh(&tmp).map_err(Error::Cuda)?
                    };
                    let (mut dot_re, mut dot_im) = (0.0f64, 0.0f64);
                    for g in 0..n_pw {
                        let psi_g = sw_col0[g];
                        let hpsi_g = hw_col0[g];
                        dot_re += psi_g.x * hpsi_g.x + psi_g.y * hpsi_g.y;
                        dot_im += psi_g.x * hpsi_g.y - psi_g.y * hpsi_g.x;
                    }
                    davidson_diag!("[davidson]     CPU dot col0: re={:.6} im={:.6} (GEMM H_sub[0,0] re={:.6})",
                        dot_re, dot_im, h_init_cpu[0].x);
                }
                for i in 0..k {
                    for j in 0..k {
                        // cublasZgemm stores column-major: element (i,j) at i + j*k
                        super_hamiltonian[i * superspace_max_bands + j] = h_init_cpu[i + j * k];
                    }
                }
                // Fill lower triangle via Hermitian conjugate
                for i in 0..k {
                    for j in 0..i {
                        let val = super_hamiltonian[j * superspace_max_bands + i];
                        super_hamiltonian[i * superspace_max_bands + j] = CudaComplex {
                            x: val.x,
                            y: -val.y,
                        };
                    }
                }
            }

            // ------------------------------------------------------------------
            // Inner Davidson loop: build → diagonalize → update ψ (CASTEP-aligned)
            // ------------------------------------------------------------------
            // CASTEP hamiltonian.f90:424 — max_iterations(1), default 10.
            // With early exit (D1 re-check, all-stopped detection), most
            // blocks converge in 1-2 iterations; 10 is a safety ceiling.
            let max_inner_iter = 10_usize;
            let mut ncol = current_nblock;

            // superspace_index and conduction states are seeded above
            // (before initial H_sub computation) — see CASTEP hamiltonian.f90:392-401.

            // CASTEP hamiltonian.f90:403-408 — separate slice workspace holds
            // the active eigenvector estimates.  super_wvfn is append-only;
            // slice_wvfn is the compacted workspace used for search directions.
            // Copy the first current_nblock columns from super_wvfn → slice.
            let mut slice_nbands = current_nblock;
            {
                let (super_ptr, _) = super_wvfn.device_ptr(stream);
                let (h_super_ptr, _) = h_super_wvfn.device_ptr(stream);
                let (slice_mut, _) = slice_wvfn.device_ptr_mut(stream);
                let (h_slice_mut, _) = slice_h_wvfn.device_ptr_mut(stream);
                for i in 0..current_nblock {
                    cublasZcopy_v2(handle, n_pw_i32,
                        (super_ptr as *const CudaComplex).add(i * n_pw) as *const _, 1,
                        (slice_mut as *mut CudaComplex).add(i * n_pw) as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    cublasZcopy_v2(handle, n_pw_i32,
                        (h_super_ptr as *const CudaComplex).add(i * n_pw) as *const _, 1,
                        (h_slice_mut as *mut CudaComplex).add(i * n_pw) as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }

            // CASTEP hamiltonian.f90:926-928 — slice_eigenvalues holds the
            // eigenvalues at the time the slice workspace was last refreshed.
            // CRITICAL: these MUST stay in sync with slice_wvfn/slice_h_wvfn.
            // Using post-ZHEEVD eigenvalues (from the global array after A3)
            // with pre-rotation slice data causes inconsistent residual shifts
            // in the preconditioner → wrong search directions → divergence.
            let mut slice_eigenvalues: Vec<f64> = block_bands(block_start, current_nblock)
                .map(|(_col, gi)| eigenvalues[gi])
                .collect();

            // CASTEP hamiltonian.f90:1106-1107 — super_eigvals pre-initialized
            // with global eigenvalues at block start.  This is dead code in CASTEP
            // (super_eigvals is zeroed at line 1170 before ZHEEVD fills it).
            // Included for CASTEP-fidelity; has no functional effect.
            let _super_eigvals_init: Vec<f64> = block_bands(block_start, current_nblock)
                .map(|(_col, gi)| eigenvalues[gi])
                .collect();

            // CASTEP hamiltonian.f90:255 — allocate previous_eigenvalues (size
            // = current ncol) but do NOT initialize.  Will be set to the current
            // eigenvalue estimates at the top of the inner loop (CASTEP line 431).
            let mut previous_eigenvalues: Vec<f64> = vec![0.0_f64; ncol];
            // Per-band opt_stop_condition (CASTEP: stagnation flag)
            let mut opt_stop_condition = vec![false; ncol];

            // block_ctx allocated ONCE outside the inner loop (GPU buffers reused).
            // Raw pointer casts are used for psi_dev/hpsi_dev copy-back to avoid
            // borrow conflicts — device_ptr_mut needs &mut self on CudaSlice.
            let mut block_ctx = DavidsonBlockCtx::new(
                &psi_dev, n_bands,
                block_start, n_pw, n_pw_i32,
                grid_size, inv_ntotal, superspace_max_bands,
                &r_vector, tpa_preconditioner, vnl_data,
                blas, solver, stream, handle,
                v_eff_dev, kinetic_dev, fft_idx_dev,
                fft_plan, kernels,
                Some(&precon_prep.r_beta_per_ion), Some(&precon_prep.q_rcq),
                precon_prep.q_rcq_gpu.as_ref(),
                precon_prep.r_beta_gpu.as_ref(),
                Some(precon_prep.total_ne).filter(|&n| n > 0),
                active_indices.clone(),
            )?;

            // Raw pointers to psi_dev/hpsi_dev GPU memory.
            // device_ptr returns CUdeviceptr (u64) = the raw device pointer value.
            // Guards must stay alive until all GPU ops on these pointers complete.
            let (psi_dev_cuptr, _psi_sync) = psi_dev.0.device_ptr(stream);
            let psi_dev_raw: *mut CudaComplex = psi_dev_cuptr as *mut CudaComplex;
            let (hpsi_dev_cuptr, _hpsi_sync) = hpsi_dev.0.device_ptr(stream);
            let hpsi_dev_raw: *mut CudaComplex = hpsi_dev_cuptr as *mut CudaComplex;

            for _inner_iter in 0..max_inner_iter {
                davidson_diag!("[davidson]     inner iter {_inner_iter}: superspace_index={superspace_index}");

                // (1) Save previous eigenvalues for convergence tracking
                // CASTEP hamiltonian.f90:431 — save ALL current_nblock
                // eigenvalues, not just active ones.  The D1 re-check
                // tests every band; converged bands must have a
                // recent baseline for delta_e computation.
                for i in 0..current_nblock {
                    previous_eigenvalues[i] = eigenvalues[block_start + i];
                }

                // (2)-(7) Build search directions: preconditioner → S-orth → S-orthonorm → H·search
                let n_added = unsafe {
                    block_ctx.build(
                        &mut super_wvfn,
                        &mut h_super_wvfn,
                        &mut superspace_index,
                        &mut grid_dev,
                        &slice_wvfn,
                        &slice_h_wvfn,
                        &slice_eigenvalues,
                        slice_nbands,
                    )?
                };

                // (8) Extend super_hamiltonian: compute new rows
                if n_added > 0 {
                    let _old_superspace_index = superspace_index;
                    let new_total = superspace_index + n_added;
                    let mut h_new_rows: CudaSlice<CudaComplex> = stream
                        .alloc_zeros(n_added * new_total)
                        .map_err(Error::Cuda)?;
                    unsafe {
                        blas.gemm_c64(
                            ZgemmConfig {
                                transa: op::C,
                                transb: op::N,
                                m: n_added as i32,
                                n: new_total as i32,
                                k: n_pw_i32,
                                alpha: CudaComplex { x: 1.0, y: 0.0 },
                                lda: n_pw_i32,
                                ldb: n_pw_i32,
                                beta: CudaComplex { x: 0.0, y: 0.0 },
                                ldc: n_added as i32,
                            },
                            block_ctx.search_dev(),
                            &h_super_wvfn,
                            &mut h_new_rows,
                        )?;
                    }

                    // D2H: copy new H rows (n_added × new_total, ~30×60 → ~14 KB) from
                    // GPU to CPU. Same justification as the initial H_sub D2H above:
                    // super_hamiltonian is CPU-side by design, ZHEGVD uses raw GPU matrices,
                    // and the transfer cost (~14 KB/extension) is negligible vs FFT+H (~2 GB).
                    let h_new_rows_cpu: Vec<CudaComplex> = stream
                        .clone_dtoh(&h_new_rows)
                        .map_err(Error::Cuda)?;
                    for i in 0..n_added {
                        for j in 0..new_total {
                            super_hamiltonian
                                [(superspace_index + i) * superspace_max_bands + j] =
                                // cublasZgemm stores column-major: element (i,j) at i + j*n_added
                                h_new_rows_cpu[i + j * n_added];
                        }
                    }

                    // Fill Hermitian conjugate correctly.
                    // H_new_rows wrote rows superspace_index..new_total-1
                    // (LOWER triangle for old columns).  We must fill the
                    // UPPER triangle FROM these new rows, NOT read the
                    // (never-written) upper triangle and write zeros down.
                    // Step 1: new rows → fill upper triangle from lower.
                    for i in superspace_index..new_total {
                        for j in 0..i {
                            let val = super_hamiltonian[i * superspace_max_bands + j];
                            super_hamiltonian[j * superspace_max_bands + i] = CudaComplex {
                                x: val.x,
                                y: -val.y,
                            };
                        }
                    }
                    // Step 2: remaining old-block upper triangle
                    // (only needed when old rows have off-diagonal entries).
                    let old_limit = superspace_index.min(new_total);
                    for i in 0..old_limit {
                        for j in 0..i {
                            let val = super_hamiltonian[j * superspace_max_bands + i];
                            super_hamiltonian[i * superspace_max_bands + j] = CudaComplex {
                                x: val.x,
                                y: -val.y,
                            };
                        }
                    }

                    superspace_index += n_added;

                }

                // ---- A3: Subspace diagonalization (ZHEEVD, standard EVP) ----
                // CASTEP hamiltonian.f90:476-480 — algor_diagonalise solves the
                // STANDARD EVP on S-orthonormal superspace vectors.  We match this
                // with ZHEEVD — no overlap matrix needed (S_sub = I implicitly).
                let k_super = superspace_index;
                let mut psi_rotated_inner = PwCoefficients::new(
                    stream.alloc_zeros(n_pw * k_super).map_err(Error::Cuda)?);
                let mut h_rotated_inner = PwCoefficients::new(
                    stream.alloc_zeros(n_pw * k_super).map_err(Error::Cuda)?);
                let mut inner_eigenvalues = vec![0.0_f64; k_super];

                // D10-01: Copy accumulated super_hamiltonian to GPU and
                // pass to ZHEEVD — faithful to CASTEP hamiltonian.f90:472-476.
                let h_sub_gpu: CudaSlice<CudaComplex> = {
                    let mut buf = stream
                        .alloc_zeros(k_super * k_super)
                        .map_err(Error::Cuda)?;
                    let mut host = vec![CudaComplex { x: 0.0, y: 0.0 }; k_super * k_super];
                    for i in 0..k_super {
                        for j in 0..k_super {
                            host[i + j * k_super] =
                                super_hamiltonian[i * superspace_max_bands + j];
                        }
                    }
                    stream.memcpy_htod(&host, &mut buf).map_err(Error::Cuda)?;
                    buf
                };

                unsafe {
                    diagonalise_subspace()
                        .psi_block(&super_wvfn)
                        .hpsi_block(&h_super_wvfn)
                        .h_sub_prebuilt(&h_sub_gpu)
                        .vnl_data(vnl_data)
                        .k(k_super)
                        .n_pw(n_pw)
                        .blas(blas)
                        .solver(solver)
                        .stream(stream)
                        .eigenvalues_out(&mut inner_eigenvalues)
                        .eig_dev(&mut eig_dev)
                        .info_dev(&mut info_dev)
                        .psi_rotated(&mut psi_rotated_inner)
                        .hpsi_rotated(&mut h_rotated_inner)
                        .gamma_point(gamma_point)
                        .call()?;
                }

                // Copy rotated results → super_wvfn, H_super_wvfn
                stream
                    .memcpy_dtod(&*psi_rotated_inner, &mut super_wvfn.0)
                    .map_err(Error::Cuda)?;
                stream
                    .memcpy_dtod(&*h_rotated_inner, &mut h_super_wvfn.0)
                    .map_err(Error::Cuda)?;

                // ---- C1: Reset super_hamiltonian to diagonal ----
                // CASTEP hamiltonian.f90:503-506 — after subspace diagonalization,
                // super_hamiltonian is reset to diag(super_eigvals). This reflects
                // that the rotated superspace columns are now eigenvectors of H_sub.
                super_hamiltonian
                    .iter_mut()
                    .for_each(|c| *c = CudaComplex { x: 0.0, y: 0.0 });
                for i in 0..k_super {
                    super_hamiltonian[i * superspace_max_bands + i] =
                        CudaComplex { x: inner_eigenvalues[i], y: 0.0 };
                }

                // CASTEP hamiltonian.f90:512 — superspace_index accumulates
                // monotonically; higher eigenstates from ZHEGVD (beyond
                // current_nblock) are kept as enrichment for the next inner
                // iteration.  This gives ZHEGVD a larger, richer subspace.
                superspace_index = k_super;

                // ---- A2 REMOVED (2026-06-10): Post-ZHEEVD S-orthogonalization ----
                // CASTEP hamiltonian_diagonalise_ks lines 1238-1248: after ZHEEVD
                // rotation, the _ks variant goes DIRECTLY to A3 (wave_copy +
                // eigenvalue update) with NO intermediate S-orthogonalization or
                // S-orthonormalization of the rotated super_wvfn.  The _slice
                // variant has A2 (lines 517-520), but _ks trusts LAPACK ZHEEVD
                // to preserve S-orthonormality well enough that re-orthogonalization
                // is unnecessary.  We match _ks.
                //
                // Removed: S-orthogonalize against lower eigenvectors
                // Removed: S-orthonormalize among k_super columns
                // Removed: ADR-0005 lockstep hpsi transform (no S-transforms to lockstep)
                // Removed: temp buffer copies super_wvfn→temp→super_wvfn
                //
                // The ZHEEVD rotation (diagonalise_subspace above) already rotates
                // BOTH super_wvfn AND h_super_wvfn by the same eigenvector matrix,
                // keeping them consistent without additional lockstep transforms.

                // ---- A3: Copy updated eigenstates → psi_dev and hpsi_dev ----
                // CASTEP hamiltonian.f90:523-528:
                //   wave_copy(super_wvfn, eigenvectors, nb_src=1, nb_dst=nb,
                //             copy_bands=current_nblock, dataonly=.true.)
                //   do i = 1, current_nblock
                //     eigenvalues(nb+i-1) = super_eigvals(i)
                //   end do
                //
                // A3 copies ALL current_nblock columns sequentially: column i
                // → band block_start + i.  This is correct because super_wvfn
                // is APPEND-ONLY (C3-07 slice workspace, never compacted) and
                // ZHEGVD eigenvectors are sorted by eigenvalue — position i
                // always corresponds to the i-th lowest energy band in the block.
                //
                // CHECKLIST: D13-02 revision (2026-06-08) — supersedes the
                // earlier active_bands() approach.  Converged bands MUST be
                // updated because ZHEGVD rotation produces improved eigenvectors
                // for ALL subspace dimensions.
                // Raw pointers bypass borrow-checker conflict with block_ctx.
                {
                    let (super_ptr, _) = super_wvfn.device_ptr(stream);
                    let (h_super_ptr, _) = h_super_wvfn.device_ptr(stream);

                    for (col, gi) in block_bands(block_start, current_nblock) {
                        cublasZcopy_v2(handle, n_pw_i32,
                            (super_ptr as *const CudaComplex).add(col * n_pw) as *const _, 1,
                            psi_dev_raw.add(gi * n_pw) as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                        cublasZcopy_v2(handle, n_pw_i32,
                            (h_super_ptr as *const CudaComplex).add(col * n_pw) as *const _, 1,
                            hpsi_dev_raw.add(gi * n_pw) as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                        eigenvalues[gi] = inner_eigenvalues[col];
                    }
                }

                // A3 copy-back modified the psi_dev columns for ALL bands in this
                // block (block_bands() covers all current_nblock positions).
                // Invalidate those bands' cached β^H·ψ projections so the next
                // outer iteration's V_NL call recomputes them fresh.
                let modified_bands: Vec<usize> =
                    block_bands(block_start, current_nblock)
                        .map(|(_, gi)| gi)
                        .collect();
                beta_phi_cache.invalidate_bands(&modified_bands);

                davidson_diag!(
                    "[davidson]     inner iter {_inner_iter} eig: [{:.6}, ..., {:.6}]",
                    inner_eigenvalues[0],
                    inner_eigenvalues[current_nblock.min(k_super) - 1]
                );

                // ---- Convergence check (CASTEP hamiltonian.f90:541-617) ----
                // active_indices maps compacted workspace column → original
                // global band position; eigenvalues/band_converged use original indices.
                // CASTEP hamiltonian.f90:541-617 — tests ALL current_nblock
                // bands every inner iteration, not just active ones.
                let mut inner_all_stopped = true;
                for b in 0..current_nblock {
                    let gi = block_start + b;

                    // CASTEP hamiltonian.f90:548-550 — reset band_converged
                    // only for bands NOT stopped by stagnation detector.
                    if !opt_stop_condition[b] {
                        band_converged[gi] = false;
                    }

                    let prev_eig = previous_eigenvalues[b];
                    let new_eig = eigenvalues[gi];
                    let delta_e = (prev_eig - new_eig).abs();
                    let eps_guard = 2.0 * new_eig.abs() * f64::EPSILON;

                    let mut band_conv = false;
                    let mut band_stopped = false;

                    // (a) Absolute tolerance (CASTEP hamiltonian.f90:555 — guard on -epsilon)
                    if tol_abs > -(f64::EPSILON) && delta_e < tol_abs.max(eps_guard) {
                        if !opt_stop_condition[b] {
                            band_conv = true;
                        }
                    }

                    // (b) Relative break condition (CASTEP hamiltonian.f90:563-589)
                    if _inner_iter == 0 {
                        break_cond_tols[gi] = delta_e;
                    } else if tol_rel > -(f64::EPSILON) && delta_e < break_cond_tols[gi] * tol_rel.abs() {
                        band_conv = true;
                        band_stopped = true;
                    } else if tol_rel <= -(f64::EPSILON) {
                        // CASTEP hamiltonian.f90:582 — only apply stagnation check
                        // when NOT on the last outer iteration. On the last iteration,
                        // opt_stop from the penultimate iteration is preserved.
                        if iteration + 1 < max_outer_iter
                            && delta_e < break_cond_tols[gi] * 0.3
                        {
                            band_stopped = true;
                        }
                    }

                    // (c) Uphill detection
                    let uphill_threshold =
                        -100.0 * (f64::EPSILON).max(f64::EPSILON * prev_eig.abs());
                    if prev_eig - new_eig < uphill_threshold {
                        band_conv = false;
                        band_stopped = false;
                    }

                    // Store opt_stop_condition for D1 re-check
                    opt_stop_condition[b] = band_stopped;

                    if band_conv {
                        band_converged[gi] = true;
                    }
                    if !band_conv && !band_stopped {
                        inner_all_stopped = false;
                    }
                }

                // ---- D1: Inner-loop exit re-check ----
                // CASTEP hamiltonian.f90:601-617 — if all bands are converged or
                // opt_stopped, reset convergence flags and re-check using ONLY the
                // strict absolute tolerance (no EPS guard, no stagnation heuristic).
                // This ensures bands are only globally marked converged if they
                // satisfy the strict criterion.
                // The epsilon guard (CASTEP hamiltonian.f90:612) skips the check
                // when tolerance is effectively zero/negative due to numerical noise:
                //   if(convergence_tols(1) > -epsilon(1.0_dp)) then
                if inner_all_stopped && tol_abs > -(f64::EPSILON) {
                    // CASTEP hamiltonian.f90:606-608 — tests ALL current_nblock bands.
                    for b in 0..current_nblock {
                        let gi = block_start + b;
                        band_converged[gi] = false;
                        let prev_eig = previous_eigenvalues[b];
                        let new_eig = eigenvalues[gi];
                        if (prev_eig - new_eig).abs() < tol_abs {
                            band_converged[gi] = true;
                        }
                    }
                    davidson_diag!(
                        "[davidson]     inner iter {_inner_iter}: all stopped, exit after D1 re-check"
                    );
                    break;
                }

                // ---- Compaction: copy unconverged FROM super_wvfn INTO slice ----
                // CASTEP hamiltonian.f90:628-642 — after convergence check,
                // compact unconverged bands FROM super_wvfn INTO a SEPARATE
                // "slice" workspace.  super_wvfn is NEVER modified by compaction
                // (C3-07).  This eliminates all in-place compaction complexity
                // (zero-stale-column, super_hamiltonian diagonal movement, etc.).
                //
                // active_indices is rebuilt to map new compacted positions
                // → original block-relative indices.
                // CASTEP hamiltonian_diagonalise_ks lines 1338-1353:
                // After each inner iteration (unless exit_inner_loop),
                // always copy ACTIVE bands from super_wvfn → slice_wvfn,
                // h_super_wvfn → slice_h_wvfn, and update slice_eigenvalues.
                // super_wvfn is ROTATED by ZHEEVD every iteration, so the
                // data changes even when the active band count doesn't.
                // NOT refreshing the slice when new_slice_nbands == ncol
                // was the root cause of cold-start divergence: after the
                // first inner iteration, the slice held pre-rotation data
                // while super_wvfn held post-rotation Ritz vectors.  The
                // preconditioner saw stale ψ/Hψ → same search directions →
                // duplicate superspace columns → ZHEEVD rank deficiency.
                {
                    // Count active bands (unconverged AND not opt_stopped).
                    let mut new_slice_nbands: usize = 0;
                    for i_src in 0..ncol {
                        let global_idx = block_start + active_indices[i_src];
                        if !band_converged[global_idx] && !opt_stop_condition[i_src] {
                            new_slice_nbands += 1;
                        }
                    }

                    // Always copy active bands from super_wvfn→slice_wvfn
                    // (matching CASTEP 1338-1348, not gated on count change).
                    let (super_ptr, _) = super_wvfn.device_ptr(stream);
                    let (h_super_ptr, _) = h_super_wvfn.device_ptr(stream);
                    let (slice_mut, _) = slice_wvfn.device_ptr_mut(stream);
                    let (h_slice_mut, _) = slice_h_wvfn.device_ptr_mut(stream);

                    // Rebuild active_indices.
                    let mut j: usize = 0;
                    let mut new_active_indices: Vec<usize> =
                        Vec::with_capacity(new_slice_nbands);
                    for i_src in 0..ncol {
                        let global_idx = block_start + active_indices[i_src];
                        if !band_converged[global_idx] && !opt_stop_condition[i_src]
                        {
                            new_active_indices.push(active_indices[i_src]);
                            // Copy super_wvfn[i_src] → slice_wvfn[j]
                            // (always, even when i_src == j — the data
                            //  has changed due to ZHEEVD rotation).
                            unsafe {
                                cublasZcopy_v2(
                                    handle, n_pw_i32,
                                    (super_ptr as *const CudaComplex)
                                        .add(i_src * n_pw) as *const _, 1,
                                    (slice_mut as *mut CudaComplex)
                                        .add(j * n_pw) as *mut _, 1,
                                )
                                .result()
                                .map_err(Error::Blas)?;
                            }
                            // Copy h_super_wvfn[i_src] → slice_h_wvfn[j]
                            unsafe {
                                cublasZcopy_v2(
                                    handle, n_pw_i32,
                                    (h_super_ptr as *const CudaComplex)
                                        .add(i_src * n_pw) as *const _, 1,
                                    (h_slice_mut as *mut CudaComplex)
                                        .add(j * n_pw) as *mut _, 1,
                                )
                                .result()
                                .map_err(Error::Blas)?;
                            }
                            // CASTEP hamiltonian.f90:1346 —
                            // slice_eigenvalues(j) = super_eigvals(i)
                            slice_eigenvalues[j] = inner_eigenvalues[i_src];
                            j += 1;
                        }
                    } // end copy loop

                    // Swap in the rebuilt active_indices mapping.
                    active_indices = new_active_indices;

                    // CASTEP does NOT compact per-band arrays
                    // (hamiltonian.f90:632-641 only copies wavefunction
                    // columns and slice_eigenvalues).  previous_eigenvalues
                    // and opt_stop_condition stay at full current_nblock
                    // size so the D1 re-check can test ALL bands.

                    if new_slice_nbands < ncol {
                        let ncol_old = ncol;
                        davidson_diag!(
                            "[davidson]     inner iter {}: compacted {} -> {} active bands (CASTEP hamiltonian.f90:628-642)",
                            _inner_iter, ncol_old, new_slice_nbands
                        );
                    }
                    ncol = new_slice_nbands;
                    slice_nbands = new_slice_nbands;
                    // current_nblock stays fixed (CASTEP: never reduced
                    // by compaction — only slice%nbands changes).
                    block_ctx.active_indices = active_indices.clone();

                    if new_slice_nbands == 0 {
                        // All bands in this block converged — nothing
                        // left to iterate
                        break;
                    }
                } // end compaction block

                // C15-07: CASTEP hamiltonian.f90:646 — no mid-loop break when
            // n_added == 0. The inner loop always completes its convergence
            // check cycle; if all bands are done, the loop exits at the top
            // via the convergence check.
        } // end inner loop (for _inner_iter)

            // CASTEP hamiltonian_diagonalise_ks (lines 723-1422) — the
            // subroutine actually called (33 times per profile) — has NO
            // cross-block conduction state transfer.  Each block's superspace
            // is built from scratch within its inner loop.  Conduction states
            // (columns > current_nblock in super_wvfn) are discarded at block
            // end.  This avoids conduction states from one block corrupting
            // the ZHEEVD eigenvalue ordering of the next block.

            // ---- D2: S-norm diagnostic (gated — CUDA_LAUNCH_BLOCKING investigation) ----
            #[cfg(feature = "scf_diag")]
            if block_start == 0 {
                // Reuse the D1 buffer pattern but check only the block columns
                let mut s_in = PwCoefficients::new(
                    stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
                let mut s_out = PwCoefficients::new(
                    stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
                let (psi_ptr, _) = psi_dev.device_ptr(stream);
                let (s_in_mut, _) = s_in.device_ptr_mut(stream);
                let (s_out_mut, _) = s_out.device_ptr_mut(stream);

                let mut max_deviation: f64 = 0.0;
                for b in 0..current_nblock {
                    let psi_b = (psi_ptr as *const CudaComplex).add(b * n_pw);
                    cublasZcopy_v2(handle, n_pw_i32,
                        psi_b as *const _, 1, s_in_mut as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    cublasZcopy_v2(handle, n_pw_i32,
                        psi_b as *const _, 1, s_out_mut as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    apply_s_times()
                        .psi_dev(&s_in)
                        .spsi_dev(&mut s_out)
                        .vnl_data(vnl_data)
                        .n_bands(1_i32)
                        .n_pw(n_pw_i32)
                        .blas(blas)
                        .stream(stream)
                        .call()?;
                    let (s_out_ptr, _) = s_out.device_ptr(stream);
                    let mut s_norm = CudaComplex { x: 0.0, y: 0.0 };
                    cublasZdotc_v2(handle, n_pw_i32,
                        psi_b as *const _, 1,
                        s_out_ptr as *const _, 1,
                        &mut s_norm as *mut _ as *mut _,
                    ).result().map_err(Error::Blas)?;
                    let dev = (s_norm.x - 1.0).abs();
                    if dev > max_deviation { max_deviation = dev; }
                }
                diag_detail!(
                    "[Diag-D2] after block 0 inner loop: max |S-norm - 1| = {:.3e} (checked {} bands)",
                    max_deviation, current_nblock
                );
            }
        } // end block loop (for block_start)

        // SURV-01 (deferred): BetaPhiCache was populated here but never
        // consumed (hamiltonian.rs:271 discards the cache).  Stage-level
        // syncs now provide the barriers this dead code used to provide.

        // Step f-g: convergence check using inner-loop convergence criteria
        for b in 0..n_bands {
            let result = check_inner_convergence(
                prev_eigenvalues[b],
                eigenvalues[b],
                tol_abs,
                0.0,
                &mut break_cond_tols[b],
                iteration == 0,
                iteration,
                max_outer_iter,
            );
            band_converged[b] = result.converged;
        }

        let n_conv = band_converged.iter().filter(|&&c| c).count();
        davidson_diag!("[davidson] after convergence check: {n_conv}/{n_bands} converged, eigenvalues: [{:.6}, ..., {:.6}]",
                  eigenvalues[0], eigenvalues[n_bands-1]);

        // CASTEP hamiltonian.f90:958-961 — exit on FIRST all-converged iteration.
        let this_all_converged = band_converged.iter().all(|&c| c);
        if this_all_converged && iteration >= min_outer_iter {
            davidson_diag!("[davidson] all converged, exiting outer loop");
            n_outer_completed = iteration + 1;
            break;
        }

        // CASTEP hamiltonian.f90:306 — H·ψ is always recomputed from scratch
        // each outer iteration.  ψ was rotated by ZHEGVD, so the rotated H·ψ
        // (h_rotated_super) is stale — force fresh computation next iteration.
        h_correct = false;

        n_outer_completed = iteration + 1;
    }

    // ------------------------------------------------------------------
    // Diagnostics: compute S⁻¹-weighted residual norms (gated — investigation)
    // ------------------------------------------------------------------
    let residual_norms_values: Vec<f64> = {
        #[cfg(feature = "scf_diag")]
        {
        // --- Compute S·ψ for residual (USPP: S ≠ I) ---
        // r_b = Hψ_b − λ_b·(Sψ)_b  (not Hψ_b − λ_b·ψ_b)
        // Using ψ instead of Sψ inflates residuals for ultrasoft
        // pseudopotentials because S = I + β·Q·β^H ≠ I.
        let mut spsi_dev = PwCoefficients::new(
            stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
        // Identity term: S = I + β·Q·β^H, pre-fill with ψ
        stream
            .memcpy_dtod(&*psi_dev, &mut spsi_dev.0)
            .map_err(Error::Cuda)?;
        unsafe {
            apply_s_times()
                .psi_dev(&psi_dev)
                .spsi_dev(&mut spsi_dev)
                .vnl_data(vnl_data)
                .n_bands(n_bands as i32)
                .n_pw(n_pw as i32)
                .blas(blas)
                .stream(stream)
                .call()?;
        }

        // Allocate temp buffer for residual = hpsi - λ·Sψ
        let mut residual_dev = PwCoefficients::new(
            stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
        stream
            .memcpy_dtod(&*hpsi_dev, &mut residual_dev.0)
            .map_err(Error::Cuda)?;

        let (spsi_ptr, _) = spsi_dev.device_ptr(stream);
        let (residual_mut, _) = residual_dev.device_ptr_mut(stream);

        for b in 0..n_bands {
            let spsi_b = (spsi_ptr as *const CudaComplex).add(b * n_pw);
            let r_b = (residual_mut as *mut CudaComplex).add(b * n_pw);
            let neg_eig = CudaComplex { x: -eigenvalues[b], y: 0.0 };
            cublasZaxpy_v2(
                handle,
                n_pw as i32,
                &neg_eig as *const _ as *const _,
                spsi_b as *const _,
                1,
                r_b as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;
        }

        // S⁻¹ norm: sinv_r = S⁻¹ · residual
        // NOTE: S⁻¹ application not yet implemented (needs CG or direct solve).
        // Using S·r as a rough proxy for S⁻¹·r in the residual norm.
        let mut sinv_r_dev = PwCoefficients::new(
            stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
        stream
            .memcpy_dtod(&*residual_dev, &mut sinv_r_dev.0)
            .map_err(Error::Cuda)?;

        // ⟨r | r⟩ → sqrt for each band (plain L2 residual, no S⁻¹ weight)
        let (residual_ptr, _) = residual_dev.device_ptr(stream);
        let sinv_ptr = residual_ptr; // alias: use plain r, not S⁻¹·r
        let mut norms = Vec::with_capacity(n_bands);
        for b in 0..n_bands {
            let r_b = (residual_ptr as *const CudaComplex).add(b * n_pw);
            let sinv_b = (sinv_ptr as *const CudaComplex).add(b * n_pw);
            let mut dot = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle,
                n_pw as i32,
                r_b as *const _,
                1,
                sinv_b as *const _,
                1,
                &mut dot as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;
            norms.push(dot.x.sqrt());
        }
        norms
        }
        #[cfg(not(feature = "scf_diag"))]
        {
            vec![0.0_f64; n_bands]
        }
    };

    // ------------------------------------------------------------------
    // Diagnostics: populate DAVIDSON_LAST_DIAG
    // ------------------------------------------------------------------
    let n_locked_final = band_converged.iter().filter(|&&c| c).count();
    let max_res = residual_norms_values.iter().cloned().fold(0.0_f64, f64::max);
    #[cfg(feature = "scf_diag")]
    {
        let max_res_idx = residual_norms_values.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0);
        let min_res = residual_norms_values.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!("[davidson] residual S⁻¹ norms: min={min_res:.4e} max={max_res:.4e} (band {max_res_idx}) n_locked={n_locked_final}");
        // Top-5 worst residuals
        let mut indexed: Vec<(usize, f64)> = residual_norms_values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[davidson] worst 5 residual norms:");
        for (band, norm) in indexed.iter().take(5) {
            eprintln!("[davidson]   band {band}: r_sinv = {norm:.6e} Ha");
        }
    }
    *DAVIDSON_LAST_DIAG.lock().unwrap() = Some(DavidsonDiagnostic {
        n_locked: n_locked_final,
        n_unconverged: n_bands - n_locked_final,
        locked_indices: band_converged
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .map(|(i, _)| i)
            .collect(),
        unconv_indices: band_converged
            .iter()
            .enumerate()
            .filter(|&(_, &c)| !c)
            .map(|(i, _)| i)
            .collect(),
        residual_norms_sinv: ResidualSInvNorm::new(residual_norms_values.clone()),
        max_residual_sinv: max_res,
        lock_tol: tol_abs,
        eigenvalue_deltas: vec![0.0_f64; n_bands],
    });

    // ------------------------------------------------------------------
    // Result
    // ------------------------------------------------------------------
    Ok(DavidsonResult {
        psi_out: psi_dev.0,
        eigenvalues,
        n_locked: n_locked_final,
        residual_norms_sinv: ResidualSInvNorm::new(residual_norms_values),
        n_outer_iterations: n_outer_completed,
    })
}

// ======================================================================
// Helper: subspace diagonalization via standard EVP (ZHEEVD)
// ======================================================================

/// Build H_sub = ψ^H · H·ψ, diagonalize via ZHEEVD or DSYEVD (standard EVP), and rotate ψ.
///
/// Matches CASTEP `algor_diagonalise` (hamiltonian.f90:476-480) — standard EVP
/// on S-orthonormal subspace vectors (S_sub = I implicitly). No overlap matrix
/// is needed because the caller keeps the subspace S-orthonormal.
///
/// `psi_block` — n_pw × k block of wavefunction columns (S-orthonormal)
/// `hpsi_block` — n_pw × k block of H·ψ columns
/// `gamma_point` — when true, uses DSYEVD (real symmetric) matching CASTEP's
///   `'S'` path for `super_wvfn%have_gamma`. When false (default), uses ZHEEVD
///   (complex Hermitian) matching CASTEP's `'H'` path.
/// On return, `psi_rotated` holds the rotated eigenbasis.
#[builder]
#[allow(clippy::too_many_arguments)]
unsafe fn diagonalise_subspace(
    psi_block: &PwCoefficients,
    hpsi_block: &PwCoefficients,
    _vnl_data: &VnlBatchData,
    k: usize,
    n_pw: usize,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
    eigenvalues_out: &mut [f64],
    eig_dev: &mut CudaSlice<f64>,
    info_dev: &mut CudaSlice<i32>,
    psi_rotated: &mut PwCoefficients,
    hpsi_rotated: &mut PwCoefficients,
    /// Use real symmetric DSYEVD for gamma-point (hamiltonian.f90:480-481).
    /// Matches CASTEP `algor_diagonalise(..., 'S')`.  Default: false (ZHEEVD).
    #[builder(default = false)]
    _gamma_point: bool,
    /// Optional: pre-built H_sub matrix (k×k, column-major, GPU-resident).
    /// When provided, the psi^H·hpsi GEMM is skipped and this matrix is
    /// copied into a working buffer for ZHEEVD.  The caller's buffer is NOT
    /// modified in-place — it is internally copied to a working buffer that
    /// ZHEEVD overwrites with eigenvectors.
    ///
    /// Used by the inner Davidson block loop where `super_hamiltonian` is
    /// maintained incrementally (D10-01), matching CASTEP hamiltonian.f90:472
    /// which copies the accumulated super_hamiltonian into the rotation matrix
    /// rather than recomputing H_sub = psi^H·hpsi via fresh GEMM.
    h_sub_prebuilt: Option<&CudaSlice<CudaComplex>>,
) -> Result<(), Error> {
    debug_assert_eq!(k, eigenvalues_out.len());
    let k_i32 = k as i32;
    let n_pw_i32 = n_pw as i32;

    // H_sub = ψ_block^H · hpsi_block  (k × k), unless pre-built
    let mut h_sub: CudaSlice<CudaComplex> =
        stream.alloc_zeros(k * k).map_err(Error::Cuda)?;
    if let Some(prebuilt) = h_sub_prebuilt {
        debug_assert_eq!(
            prebuilt.len(),
            k * k,
            "h_sub_prebuilt size mismatch: expected {} got {}",
            k * k,
            prebuilt.len()
        );
        // D10-01: use incrementally-built super_hamiltonian.
        stream.memcpy_dtod(prebuilt, &mut h_sub).map_err(Error::Cuda)?;

        // ---- Diagnostic: compare D10-01 prebuilt H_sub against fresh GEMM ----
        {
            let mut h_sub_fresh: CudaSlice<CudaComplex> =
                stream.alloc_zeros(k * k).map_err(Error::Cuda)?;
            unsafe {
                blas.gemm_c64(
                    ZgemmConfig {
                        transa: op::C, transb: op::N,
                        m: k_i32, n: k_i32, k: n_pw_i32,
                        alpha: CudaComplex { x: 1.0, y: 0.0 },
                        lda: n_pw_i32, ldb: n_pw_i32, ldc: k_i32,
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                    },
                    psi_block, hpsi_block, &mut h_sub_fresh,
                )?;
            }
            let prebuilt_cpu: Vec<CudaComplex> =
                stream.clone_dtoh(&h_sub).map_err(Error::Cuda)?;
            let fresh_cpu: Vec<CudaComplex> =
                stream.clone_dtoh(&h_sub_fresh).map_err(Error::Cuda)?;
            let mut max_diff: f64 = 0.0;
            let mut max_i = 0usize; let mut max_j = 0usize;
            for i in 0..k {
                for j in 0..k {
                    let d = (prebuilt_cpu[i + j * k].x - fresh_cpu[i + j * k].x).abs()
                          + (prebuilt_cpu[i + j * k].y - fresh_cpu[i + j * k].y).abs();
                    if d > max_diff { max_diff = d; max_i = i; max_j = j; }
                }
            }
            eprintln!(
                "[Diag-D10] D10-01 H_sub vs fresh GEMM: k={} max_diff={:.6e} at ({},{}) \
                 prebuilt=({:.6e},{:.6e}) fresh=({:.6e},{:.6e})",
                k, max_diff, max_i, max_j,
                prebuilt_cpu[max_i + max_j * k].x, prebuilt_cpu[max_i + max_j * k].y,
                fresh_cpu[max_i + max_j * k].x, fresh_cpu[max_i + max_j * k].y,
            );
            if max_diff > 1e-3 {
                eprintln!(
                    "[Diag-D10] D10-01 H_sub vs fresh GEMM max_diff={:.6e} at ({},{})",
                    max_diff, max_i, max_j,
                );
            }
        }
    } else {
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: k_i32,
                    n: k_i32,
                    k: n_pw_i32,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw_i32,
                    ldb: n_pw_i32,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: k_i32,
                },
                psi_block,
                hpsi_block,
                &mut h_sub,
            )?;
        }
    }

    // ---- D4: H_sub condition diagnostic (standard EVP, no S_sub needed) ----
    // CASTEP hamiltonian.f90:476-480 — algor_diagonalise solves STANDARD EVP
    // on S-orthonormalized superspace vectors (S_sub = I implicitly).  We
    // match this faithfully by calling ZHEEVD on H_sub directly.
    {
        let h_sub_cpu: Vec<CudaComplex> = stream.clone_dtoh(&h_sub).map_err(Error::Cuda)?;
        let mut max_h = 0.0f64;
        let mut min_diag = f64::MAX;
        let mut max_diag: f64 = 0.0;
        for i in 0..k {
            let di = h_sub_cpu[i + i * k].x;
            if di < min_diag { min_diag = di; }
            if di > max_diag { max_diag = di; }
            for j in 0..k {
                let v = h_sub_cpu[i + j * k];
                let a = (v.x * v.x + v.y * v.y).sqrt();
                if a > max_h { max_h = a; }
            }
        }
        diag_detail!(
            "[Diag-D4] H_sub (k={}): min_diag={:.3e} max_diag={:.3e} max|entry|={:.3e}",
            k, min_diag, max_diag, max_h,
        );
    }

    // Resize eigenvalue/info GPU buffers if needed
    if eig_dev.len() < k {
        *eig_dev = stream.alloc_zeros(k).map_err(Error::Cuda)?;
    }
    if info_dev.is_empty() {
        *info_dev = stream.alloc_zeros(1).map_err(Error::Cuda)?;
    }

    // CASTEP hamiltonian.f90:476-480 — algor_diagonalise solves the STANDARD
    // EVP on S-orthonormalized superspace vectors.
    //
    // For gamma-point (super_wvfn%have_gamma, hamiltonian.f90:480-481),
    // CASTEP calls algor_diagonalise(..., 'S') → DSYEV (real symmetric).
    // For non-gamma, CASTEP calls algor_diagonalise(..., 'H') → ZHEEV.
    //
    // We use ZHEEVD for both cases: real-symmetric matrices are a special
    // case of complex Hermitian, and ZHEEVD handles them correctly — the
    // eigenvalues are real and eigenvector imaginary parts are ~machine zero.
    // The nblock adjustment at §block-sizing still respects gamma_point to
    // avoid odd block sizes that cause DSYEVD parity issues (historical).

    // ---- ZHEEVD path (complex Hermitian, gamma + non-gamma) ----
    // No overlap matrix needed because the superspace is S-orthonormal
    // (ψ_block is S-orthogonal to lower bands and S-orthonormal among
    // themselves, so S_sub = I implicitly).
    solver.zheevd(
        cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        k_i32,
        &mut h_sub, // overwritten → eigenvectors X
        eig_dev,
        info_dev,
    )?;

    // Check solver info
    let info_cpu: Vec<i32> = stream.clone_dtoh(info_dev).map_err(Error::Cuda)?;
    if info_cpu[0] != 0 {
        return Err(Error::RayleighRitzFailed {
            info: info_cpu[0],
        });
    }

    // D2H eigenvalues
    let blk_eig: Vec<f64> = stream.clone_dtoh(eig_dev).map_err(Error::Cuda)?;
    eigenvalues_out.copy_from_slice(&blk_eig[..k]);

    // Rotate: ψ_new = ψ_block · X  (n_pw × k)
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::N,
                transb: op::N,
                m: n_pw_i32,
                n: k_i32,
                k: k_i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw_i32,
                ldb: k_i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n_pw_i32,
            },
            psi_block,
            &h_sub, // eigenvectors (column-major, k×k)
            psi_rotated,
        )?;

        // Also rotate H·psi (cheap GEMM, no FFT)
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::N,
                transb: op::N,
                m: n_pw_i32,
                n: k_i32,
                k: k_i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw_i32,
                ldb: k_i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n_pw_i32,
            },
            hpsi_block,
            &h_sub, // eigenvectors
            hpsi_rotated,
        )?;
    }

    Ok(())
}

// ======================================================================
// Builder helpers: search direction, S-orthogonalise, S-orthonormalise
// ======================================================================

/// S-orthogonalize search directions against superspace columns.
///
/// CASTEP wave.f90:13388-13414 — batch ZGEMM approach:
///   1. Compute S·search for all columns in one batch
///   2. Compute overlap = super_wvfn^H * (S·search) via ZGEMM
///   3. search -= super_wvfn * overlap via ZGEMM
///   4. (Optional lockstep) hsearch -= hpsi_ref * overlap via ZGEMM
///
/// When `hpsi_dev` and `hpsi_ref` are both provided, the same overlap
/// coefficients are applied to the H·psi counterpart of the search
/// directions. This maintains the invariant hsearch = H·search after
/// the orthogonalization (see ADR-0005).
///
/// This matches CASTEP's wave_Sorthogonalise_wv_slice exactly.
#[builder]
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn s_orthogonalise(
    search_dev: &mut PwCoefficients,
    super_wvfn: &PwCoefficients,
    superspace_index: usize,
    ncol: usize,
    n_pw: usize,
    n_pw_i32: i32,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    s_orth_in: &mut PwCoefficients,
    s_orth_out: &mut PwCoefficients,
    handle: cublasHandle_t,
    /// Optional: H·psi counterpart of `search_dev`. When provided together
    /// with `hpsi_ref`, apply the same overlap coefficients to transform
    /// hsearch in lockstep with search (ADR-0005).
    hpsi_dev: Option<&mut PwCoefficients>,
    /// Optional: H·psi counterpart of `super_wvfn`. When provided together
    /// with `hpsi_dev`, used as the reference columns for the lockstep
    /// H·psi transform. The reference columns are NOT modified.
    hpsi_ref: Option<&PwCoefficients>,
) -> Result<(), Error> {
    if superspace_index == 0 {
        return Ok(());
    }
    // Step 1: Compute S·search in batch → s_orth_out (n_pw × ncol)
    let (search_ptr, _) = search_dev.device_ptr(stream);
    let (s_in_mut, _) = s_orth_in.device_ptr_mut(stream);
    let (s_out_mut, _) = s_orth_out.device_ptr_mut(stream);
    // Copy all ncol search columns to s_orth_in AND s_orth_out.
    // CASTEP wave.f90 (apply_S_operator) applies S = I + β·Q·β^H.
    // Our apply_s_times (hamiltonian.rs:365-367) only adds the β·Q·β^H
    // non-local correction — the identity term MUST already be present
    // in spsi_dev before the call.  Without this pre-copy, s_orth_out
    // contains only the NL correction (missing the PW kinetic part),
    // producing wrong S·search → wrong ZGEMM overlap → eigenvalue explosion.
    for j in 0..ncol {
        let src = (search_ptr as *const CudaComplex).add(j * n_pw);
        let dst_in = (s_in_mut as *mut CudaComplex).add(j * n_pw);
        let dst_out = (s_out_mut as *mut CudaComplex).add(j * n_pw);
        cublasZcopy_v2(handle, n_pw_i32, src as *const _, 1, dst_in as *mut _, 1)
            .result().map_err(Error::Blas)?;
        cublasZcopy_v2(handle, n_pw_i32, src as *const _, 1, dst_out as *mut _, 1)
            .result().map_err(Error::Blas)?;
    }
    unsafe {
        apply_s_times()
            .psi_dev(&*s_orth_in)
            .spsi_dev(&mut *s_orth_out)
            .vnl_data(vnl_data)
            .n_bands(ncol as i32)
            .n_pw(n_pw_i32)
            .blas(blas)
            .stream(stream)
            .call()?;
    }

    // Step 2: Compute overlap = super_wvfn^H * (S·search) via ZGEMM
    // overlap is (superspace_index × ncol)
    let mut overlap_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(superspace_index * ncol).map_err(Error::Cuda)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::C,   // super_wvfn^H
                transb: op::N,   // S·search
                m: superspace_index as i32,
                n: ncol as i32,
                k: n_pw_i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw_i32,
                ldb: n_pw_i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: superspace_index as i32,
            },
            super_wvfn,
            &*s_orth_out,
            &mut overlap_dev,
        )?;
    }

    // Step 3: search -= super_wvfn * overlap via ZGEMM
    // alpha=-1 directly — safe because build() syncs after this function.
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: op::N,
                transb: op::N,
                m: n_pw_i32, n: ncol as i32, k: superspace_index as i32,
                alpha: CudaComplex { x: -1.0, y: 0.0 },
                lda: n_pw_i32, ldb: superspace_index as i32,
                beta: CudaComplex { x: 1.0, y: 0.0 },
                ldc: n_pw_i32,
            },
            super_wvfn, &overlap_dev, search_dev,
        )?;
    }

    // Step 4 (ADR-0005): hsearch -= hpsi_ref * overlap (lockstep transform).
    if let (Some(hpsi_search), Some(hpsi_ref_val)) = (hpsi_dev, hpsi_ref) {
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::N, transb: op::N,
                    m: n_pw_i32, n: ncol as i32, k: superspace_index as i32,
                    alpha: CudaComplex { x: -1.0, y: 0.0 },
                    lda: n_pw_i32, ldb: superspace_index as i32,
                    beta: CudaComplex { x: 1.0, y: 0.0 },
                    ldc: n_pw_i32,
                },
                hpsi_ref_val, &overlap_dev, hpsi_search,
            )?;
        }
    }

    Ok(())
}

/// Modified Gram-Schmidt with S-norm normalization for search directions.
///
/// Uses S-inner products throughout, matching CASTEP's
/// `wave_Sorthonormalise_slice` (wave.f90:11573-11677).
///
/// Algorithm (S-norm MGS):
///   1. Precompute S·search_j for current column j
///   2. Compute S-norm: nrm = sqrt(⟨search_j | S | search_j⟩)
///   3. Skip if below threshold (band already in superspace)
///   4. Orthogonalize against earlier columns i < j using S-inner products:
///      dot = ⟨search_i | S | search_j⟩ = zdotc(search_i, S·search_j)
///      search_j -= dot · search_i
///      hpsi_j   -= dot · hpsi_i           (lockstep, ADR-0005)
///   5. Recompute S·search_j (search_j changed in step 4)
///   6. S-norm normalize (same factor applied to hpsi_j)
///
/// NOTE: This uses O(2 · ncol) S-applications via apply_s_times. With
/// ncol ≤ 4 in typical Davidson blocks the cost is acceptable.
#[builder]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn s_orthonormalise(
    search_dev: &mut PwCoefficients,
    ncol: usize,
    n_pw: usize,
    n_pw_i32: i32,
    vnl_data: &VnlBatchData,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    handle: cublasHandle_t,
    s_orth_in: &mut PwCoefficients,
    s_orth_out: &mut PwCoefficients,
    /// Optional: H·psi counterpart of `search_dev`. When provided, apply the
    /// same MGS orthogonalization coefficients and normalization factor to
    /// the H·psi columns in lockstep with the wavefunction (ADR-0005).
    mut hpsi_dev: Option<&mut PwCoefficients>,
    /// cuSOLVER handle for GPU-resident ZPOTRF (CASTEP Cholesky-primary path).
    solver: &SolverHandle,
) -> Result<(), Error> {

    // ---- C6-D2: Cholesky-primary path (CASTEP algor.F90:1098-1140) ----
    // CASTEP's wave_orthonormalise_over_slice uses:
    //   1. zpotrf('U') — Cholesky: S = U^H·U (cuSOLVER, GPU)
    //   2. ztrtri('U','N') — triangular inverse (replaced by ZTRSM below)
    //   3. ztrmm('R','U','N','N') — rotate (replaced by ZTRSM below)
    //   4. Fall back to single-pass Gram-Schmidt if Cholesky fails.
    //
    // We use GPU ZPOTRF → ZTRSM (psi_new · U = psi ⇒ psi_new = psi · U⁻¹).
    // ZTRSM is more numerically stable than computing the explicit inverse
    // followed by ZGEMM.  All GPU-resident, no CPU↔GPU copies.
    if ncol > 0 {
        // ---- Compute S·psi for all columns (batch) ----
        // apply_s_times accumulates with beta=1: spsi += beta·q.
        // spsi_dev MUST be pre-initialised to psi so that the
        // identity part I·psi is included.  (The MGS fallback
        // does this per column; we do it once for all columns.)
        {
            let (s_in_mut, _) = s_orth_in.device_ptr_mut(stream);
            let (search_ptr, _) = search_dev.device_ptr(stream);
            cublasZcopy_v2(handle, (ncol as i32) * n_pw_i32,
                search_ptr as *const _, 1,
                s_in_mut as *mut _, 1,
            ).result().map_err(Error::Blas)?;
            let (s_out_mut, _) = s_orth_out.device_ptr_mut(stream);
            cublasZcopy_v2(handle, (ncol as i32) * n_pw_i32,
                search_ptr as *const _, 1,
                s_out_mut as *mut _, 1,
            ).result().map_err(Error::Blas)?;
        }
        unsafe {
            apply_s_times()
                .psi_dev(&*s_orth_in)
                .spsi_dev(&mut *s_orth_out)
                .vnl_data(vnl_data)
                .n_bands(ncol as i32)
                .n_pw(n_pw_i32)
                .blas(blas)
                .stream(stream)
                .call()?;
        }

        // ---- Compute S_overlap = search_dev^H · (S·search_dev) on GPU ----
        let nc = ncol as i32;
        let mut s_overlap_gpu: CudaSlice<CudaComplex> =
            stream.alloc_zeros((ncol * ncol) as usize).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: nc, n: nc, k: n_pw_i32,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw_i32, ldb: n_pw_i32, ldc: nc,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                },
                search_dev,
                &*s_orth_out,
                &mut s_overlap_gpu,
            )?;
        }

        // ---- GPU ZPOTRF: Cholesky S_overlap = U^H·U (CASTEP algor.F90:1105) ----
        // The ZGEMM ran on blas.stream(); synchronise it so cuSOLVER
        // (which uses its own internal stream) sees the complete matrix.
        blas.stream().synchronize().map_err(Error::Cuda)?;

        // D2H snapshot of S_overlap + independent ZDOTC check of col 0.
        {
            let diag_host: Vec<CudaComplex> =
                blas.stream().clone_dtoh(&s_overlap_gpu).map_err(Error::Cuda)?;
            let _n = ncol;
            let s00 = diag_host[0].x;
            // Independent ZDOTC: ⟨search_0 | S·search_0⟩ directly
            let (s_ptr, _) = search_dev.device_ptr(stream);
            let (so_ptr, _) = s_orth_out.device_ptr(stream);
            let mut zdotc_sn = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(handle, n_pw_i32,
                s_ptr as *const _, 1,
                so_ptr as *const _, 1,
                &mut zdotc_sn as *mut _ as *mut _,
            ).result().map_err(Error::Blas)?;
            eprintln!(
                "[Diag-C6] ncol={}  S_overlap[0,0]={:.6e}  ZDOTC(psi0,S·psi0)={:.6e}  diff={:.6e}",
                ncol, s00, zdotc_sn.x, (s00 - zdotc_sn.x).abs(),
            );
        }

        let mut chol_info: CudaSlice<i32> =
            solver.stream().alloc_zeros(1).map_err(Error::Cuda)?;
        let chol_result = solver.zpotrf(
            cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            nc,
            &mut s_overlap_gpu,
            &mut chol_info,
        );

        let mut chol_ok = false;
        if chol_result.is_ok() {
            // Wait for cuSOLVER to finish; then check the info code.
            solver.stream().synchronize().map_err(Error::Cuda)?;
            let info_host: Vec<i32> = solver.stream()
                .clone_dtoh(&chol_info)
                .map_err(Error::Cuda)?;
            chol_ok = info_host[0] == 0;
            if !chol_ok {
                eprintln!(
                    "[Diag-C6] ZPOTRF info={} for ncol={}: falling back to MGS",
                    info_host[0], ncol,
                );
            }
        } else {
            eprintln!(
                "[Diag-C6] ZPOTRF error for ncol={}: falling back to MGS",
                ncol,
            );
        }

        if chol_ok {
            // ---- GPU ZTRSM: psi_new · U = psi  →  psi_new = psi · U⁻¹ ----
            // Side=RIGHT, uplo=UPPER, trans=N:  X · U = psi  ⇒  X = psi · U⁻¹.
            // ZTRSM operates in-place on the RHS; we copy psi into a fresh
            // buffer first so search_dev is not overwritten on failure.
            let mut psi_new: CudaSlice<CudaComplex> =
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?;
            {
                let (search_ptr, _) = search_dev.device_ptr(stream);
                let (psi_mut, _) = psi_new.device_ptr_mut(stream);
                cublasZcopy_v2(handle, nc * n_pw_i32,
                    search_ptr as *const _, 1,
                    psi_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
            }

            let alpha = CudaComplex { x: 1.0, y: 0.0 };
            let ztrsm = |rhs: &mut CudaSlice<CudaComplex>| {
                let (a_ptr, _a) = s_overlap_gpu.device_ptr(stream);
                let (b_mut, _b) = rhs.device_ptr_mut(stream);
                use cudarc::cublas::sys as cublas_sys;
                cublasZtrsm_v2(
                    handle,
                    cublasSideMode_t::CUBLAS_SIDE_RIGHT,
                    cublas_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                    n_pw_i32, nc,
                    &alpha as *const _ as *const _,
                    a_ptr as *const _, nc,
                    b_mut as *mut _, n_pw_i32,
                ).result().map_err(Error::Blas)
            };

            ztrsm(&mut psi_new)?;

            // ---- Apply same ZTRSM to hpsi (ADR-0005 lockstep) ----
            if let Some(ref mut hpsi) = hpsi_dev {
                let mut hpsi_new: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?;
                {
                    let (hpsi_ptr, _) = hpsi.device_ptr(stream);
                    let (h_mut, _) = hpsi_new.device_ptr_mut(stream);
                    cublasZcopy_v2(handle, nc * n_pw_i32,
                        hpsi_ptr as *const _, 1,
                        h_mut as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
                ztrsm(&mut hpsi_new)?;
                // Copy back into hpsi_dev
                let (hpsi_mut, _) = hpsi.device_ptr_mut(stream);
                let (new_ptr, _) = hpsi_new.device_ptr(stream);
                cublasZcopy_v2(handle, nc * n_pw_i32,
                    new_ptr as *const _, 1,
                    hpsi_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
            }

            // Copy psi_new back to search_dev
            {
                let (search_mut, _) = search_dev.device_ptr_mut(stream);
                let (new_ptr, _) = psi_new.device_ptr(stream);
                cublasZcopy_v2(handle, nc * n_pw_i32,
                    new_ptr as *const _, 1,
                    search_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
            }

            // ---- Verify post-Cholesky S-norms (CASTEP wave.f90:11631-11668) ----
            // GPU ZPOTRF can return info=0 (success) for near-singular S_overlap
            // matrices while producing numerically inaccurate Cholesky factors.
            // CASTEP detects this via algor_invert status and falls back to
            // per-column Gram-Schmidt with S-normalization.  We do the same:
            // compute S·search[0] → check ⟨search[0]|S|search[0]⟩ ≈ 1.0.
            // If the S-norm deviates significantly, the factorization was
            // inaccurate — fall through to MGS.
            {
                let (search_ptr, _) = search_dev.device_ptr(stream);
                let search_0 = search_ptr as *const CudaComplex;
                let (s_in_mut, _) = s_orth_in.device_ptr_mut(stream);
                cublasZcopy_v2(handle, n_pw_i32,
                    search_0 as *const _, 1,
                    s_in_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                let (s_out_mut, _) = s_orth_out.device_ptr_mut(stream);
                cublasZcopy_v2(handle, n_pw_i32,
                    search_0 as *const _, 1,
                    s_out_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                unsafe {
                    apply_s_times()
                        .psi_dev(&*s_orth_in)
                        .spsi_dev(&mut *s_orth_out)
                        .vnl_data(vnl_data)
                        .n_bands(1_i32)
                        .n_pw(n_pw_i32)
                        .blas(blas)
                        .stream(stream)
                        .call()?;
                }
                let (s_out_ptr, _) = s_orth_out.device_ptr(stream);
                let mut snorm = CudaComplex { x: 0.0, y: 0.0 };
                cublasZdotc_v2(handle, n_pw_i32,
                    search_0 as *const _, 1,
                    s_out_ptr as *const _, 1,
                    &mut snorm as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                if (snorm.x - 1.0).abs() > 0.1 {
                    davidson_diag!(
                        "[Diag-C6] Cholesky post-check FAIL: S-norm[0]={:.6e} > 0.1 off 1.0, falling back to MGS",
                        snorm.x
                    );
                    // Fall through to MGS below
                } else {
                    return Ok(());
                }
            }

            return Ok(());
        }
        // Cholesky failed (ZPOTRF error OR post-check deviation > 0.1)
        // — fall through to Gram-Schmidt fallback
    }

    // ---- Gram-Schmidt fallback (CASTEP wave.f90:11645-11663) ----
    // Single-pass MGS extended to two passes.  CASTEP uses 1-pass but
    // its ZPOTRF succeeds more often (reaching GS less frequently).
    // For large ncol (128-156) with near-linearly-dependent columns,
    // 2-pass MGS is significantly more accurate than 1-pass, preventing
    // the conduction-state corruption cascade that causes eigenvalue
    // explosion at large subspace sizes.
    let (search_mut, _search_guard) = search_dev.device_ptr_mut(stream);
    let (_hpsi_guard, hpsi_raw): (Option<_>, Option<*mut CudaComplex>) =
        if let Some(ref mut hpsi) = hpsi_dev {
            let (ptr, guard) = hpsi.device_ptr_mut(stream);
            (Some(guard), Some(ptr as *mut CudaComplex))
        } else {
            (None, None)
        };
    for _pass in 0..2 {
    for j in 0..ncol {
        let search_j = (search_mut as *mut CudaComplex).add(j * n_pw);
        let hpsi_j = hpsi_raw.map(|raw| raw.add(j * n_pw));

        // ------------------------------------------------------------------
        // Step 1: Precompute S·search_j -> s_orth_out
        // ------------------------------------------------------------------
        {
            let (s_in_mut, _) = s_orth_in.device_ptr_mut(stream);
            cublasZcopy_v2(
                handle,
                n_pw_i32,
                search_j as *const _,
                1,
                s_in_mut as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;

            let (s_out_mut, _) = s_orth_out.device_ptr_mut(stream);
            cublasZcopy_v2(
                handle,
                n_pw_i32,
                search_j as *const _,
                1,
                s_out_mut as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;

            unsafe {
                apply_s_times()
                    .psi_dev(&*s_orth_in)
                    .spsi_dev(&mut *s_orth_out)
                    .vnl_data(vnl_data)
                    .n_bands(1_i32)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .call()?;
            }
        }

        // ------------------------------------------------------------------
        // Step 2: Compute S-norm: nrm = sqrt(⟨search_j | S | search_j⟩)
        // ------------------------------------------------------------------
        let (s_out_ptr, _) = s_orth_out.device_ptr(stream);
        let mut nrm_sq = CudaComplex { x: 0.0, y: 0.0 };
        cublasZdotc_v2(
            handle,
            n_pw_i32,
            search_j as *const _,
            1,
            s_out_ptr as *const _,
            1,
            &mut nrm_sq as *mut _ as *mut _,
        )
        .result()
        .map_err(Error::Blas)?;

        // Step 3: Skip zero residual (band already in superspace)
        if nrm_sq.x < 1e-30 {
            continue;
        }

        // ------------------------------------------------------------------
        // Step 4: Orthogonalize against earlier search columns (S-norm MGS)
        // ------------------------------------------------------------------
        for i in 0..j {
            let search_i = (search_mut as *const CudaComplex).add(i * n_pw);
            let hpsi_i = hpsi_raw.map(|raw| raw.add(i * n_pw) as *const CudaComplex);

            // S-inner product: dot = ⟨search_i | S | search_j⟩
            // Uses precomputed S·search_j in s_orth_out
            let mut dot = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle,
                n_pw_i32,
                search_i as *const _,
                1,
                s_out_ptr as *const _,
                1,
                &mut dot as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;

            let neg_dot = CudaComplex {
                x: -dot.x,
                y: -dot.y,
            };
            // search_j -= dot · search_i
            cublasZaxpy_v2(
                handle,
                n_pw_i32,
                &neg_dot as *const _ as *const _,
                search_i as *const _,
                1,
                search_j as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;

            // (ADR-0005 lockstep): hpsi_j -= dot · hpsi_i
            if let (Some(hj), Some(hi)) = (hpsi_j, hpsi_i) {
                cublasZaxpy_v2(
                    handle,
                    n_pw_i32,
                    &neg_dot as *const _ as *const _,
                    hi as *const _,
                    1,
                    hj as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }

        // ------------------------------------------------------------------
        // Step 5: Recompute S·search_j (search_j changed in step 4)
        // ------------------------------------------------------------------
        {
            let (s_in_mut, _) = s_orth_in.device_ptr_mut(stream);
            cublasZcopy_v2(
                handle,
                n_pw_i32,
                search_j as *const _,
                1,
                s_in_mut as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;

            let (s_out_mut, _) = s_orth_out.device_ptr_mut(stream);
            cublasZcopy_v2(
                handle,
                n_pw_i32,
                search_j as *const _,
                1,
                s_out_mut as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;

            unsafe {
                apply_s_times()
                    .psi_dev(&*s_orth_in)
                    .spsi_dev(&mut *s_orth_out)
                    .vnl_data(vnl_data)
                    .n_bands(1_i32)
                    .n_pw(n_pw_i32)
                    .blas(blas)
                    .stream(stream)
                    .call()?;
            }
        }

        // ------------------------------------------------------------------
        // Step 6: S-norm normalize (lockstep: same factor applied to hpsi_j)
        // ------------------------------------------------------------------
        let (s_out_ptr, _) = s_orth_out.device_ptr(stream);
        let mut nrm2_sq = CudaComplex { x: 0.0, y: 0.0 };
        cublasZdotc_v2(
            handle,
            n_pw_i32,
            search_j as *const _,
            1,
            s_out_ptr as *const _,
            1,
            &mut nrm2_sq as *mut _ as *mut _,
        )
        .result()
        .map_err(Error::Blas)?;

        let inv_norm = 1.0 / nrm2_sq.x.sqrt();
        // D5: track S-norm and L2-norm for first search column
        if j == 0 {
            let mut l2_sq = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(handle, n_pw_i32,
                search_j as *const _, 1,
                search_j as *const _, 1,
                &mut l2_sq as *mut _ as *mut _,
            ).result().map_err(Error::Blas)?;
            eprintln!(
                "[Diag-D5] s_orthonormalise col 0: L2²={:.6e} S²={:.6e} inv_norm={:.6e}",
                l2_sq.x, nrm2_sq.x, inv_norm
            );
        }
        let scale = CudaComplex {
            x: inv_norm,
            y: 0.0,
        };
        cublasZscal_v2(
            handle,
            n_pw_i32,
            &scale as *const _ as *const _,
            search_j as *mut _,
            1,
        )
        .result()
        .map_err(Error::Blas)?;
        // (ADR-0005 lockstep): normalize hpsi_j by the same factor
        if let Some(hj) = hpsi_j {
            cublasZscal_v2(
                handle,
                n_pw_i32,
                &scale as *const _ as *const _,
                hj as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;
        }
    } // end j-loop (one Gram-Schmidt pass)
    } // end pass-loop (2-pass reorthogonalization)
    Ok(())
}

/// Per-block scratch buffers and context for the Davidson inner loop.
///
/// Owns all block-local GPU scratch allocations and holds references to
/// the outer-loop state.  Created once per block, reused across inner
/// iterations.
///
/// Pipeline: precondition → S-orth → S-orthonorm → H·search → superspace copy
struct DavidsonBlockCtx<'a> {
    // --- Block dimensions ---
    block_start: usize,
    n_pw: usize,
    n_pw_i32: i32,
    grid_size: usize,
    inv_ntotal: f64,
    superspace_max_bands: usize,

    // --- Outer state (borrowed) ---
    psi_dev: &'a PwCoefficients,
    n_bands_total: usize,
    /// Maps compacted workspace column → original global band index.
    /// Updated after compaction.
    active_indices: Vec<usize>,
    r_vector: &'a PreconditionerVector,
    tpa_preconditioner: &'a TpaPreconditioner,
    vnl_data: &'a VnlBatchData,
    blas: &'a BlasHandle,
    solver: &'a SolverHandle,
    stream: &'a Arc<CudaStream>,
    handle: cublasHandle_t,
    v_eff_dev: &'a CudaSlice<f64>,
    kinetic_dev: &'a KineticPreconditioner,
    fft_idx_dev: &'a CudaSlice<i32>,
    fft_plan: &'a BatchedFftPlan3d,
    kernels: &'a CudaKernelSet,
    r_beta_per_ion: Option<&'a [Array2<Complex64>]>,
    q_rcq: Option<&'a Array2<Complex64>>,
    // GPU-resident USPP params (pre-uploaded in prepare_preconditioner)
    q_rcq_gpu: Option<&'a CudaSlice<CudaComplex>>,
    r_beta_gpu: Option<&'a CudaSlice<CudaComplex>>,
    total_ne: Option<usize>,

    // --- Owned scratch buffers ---
    block_psi_temp: PwCoefficients,
    block_hpsi_temp: PwCoefficients,
    eig_block_dev: CudaSlice<f64>,
    search_dev: PwCoefficients,
    hsearch_dev: PwCoefficients,
    s_orth_in: PwCoefficients,
    s_orth_out: PwCoefficients,
}

/// Iterator over active (unconverged) bands in a Davidson block.
///
/// Yields `(compacted_col, global_band_index)` pairs, where `compacted_col`
/// is the position in the compacted workspace (`0..active_indices.len()`)
/// and `global_band_index = block_start + active_indices[compacted_col]`
/// is the original global band position.
///
/// This replaces manual `block_start + active_indices[i]` indexing, which
/// is a proven defect vector (see compaction index bug fix):
/// `block_start + i` is NOT the same as `block_start + active_indices[i]`
/// after compaction removes converged bands from the active set.
#[inline]
fn active_bands(
    active_indices: &[usize],
    block_start: usize,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    active_indices
        .iter()
        .enumerate()
        .map(move |(ci, &orig_idx)| (ci, block_start + orig_idx))
}

/// Sequential band iterator for append-only (never-compacted) super_wvfn access.
///
/// Yields `(column, global_band_index)` for ALL `current_nblock` bands using
/// the natural 1:1 mapping: column `i` → band `block_start + i`.  This is the
/// correct iterator for Stage 1 (psi→super_wvfn copy) and A3 (super_wvfn→psi
/// copy-back) because super_wvfn is NEVER compacted (C3-07 slice workspace) —
/// ZHEGVD eigenvectors at position `i` always correspond to the `i`-th lowest
/// energy band in the block.
///
/// Unlike `active_bands()`, this does NOT consult `active_indices`.  After
/// compaction, converged bands' super_wvfn columns remain at their original
/// positions and ZHEGVD still produces eigenvectors sorted by eigenvalue.
/// Sequential mapping is correct regardless of compaction state.
#[inline]
fn block_bands(
    block_start: usize,
    current_nblock: usize,
) -> impl Iterator<Item = (usize, usize)> {
    (0..current_nblock).map(move |i| (i, block_start + i))
}

impl<'a> DavidsonBlockCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        psi_dev: &'a PwCoefficients,
        n_bands_total: usize,
        block_start: usize,
        n_pw: usize,
        n_pw_i32: i32,
        grid_size: usize,
        inv_ntotal: f64,
        superspace_max_bands: usize,
        r_vector: &'a PreconditionerVector,
        tpa_preconditioner: &'a TpaPreconditioner,
        vnl_data: &'a VnlBatchData,
        blas: &'a BlasHandle,
        solver: &'a SolverHandle,
        stream: &'a Arc<CudaStream>,
        handle: cublasHandle_t,
        v_eff_dev: &'a CudaSlice<f64>,
        kinetic_dev: &'a KineticPreconditioner,
        fft_idx_dev: &'a CudaSlice<i32>,
        fft_plan: &'a BatchedFftPlan3d,
        kernels: &'a CudaKernelSet,
        r_beta_per_ion: Option<&'a [Array2<Complex64>]>,
        q_rcq: Option<&'a Array2<Complex64>>,
        q_rcq_gpu: Option<&'a CudaSlice<CudaComplex>>,
        r_beta_gpu: Option<&'a CudaSlice<CudaComplex>>,
        total_ne: Option<usize>,
        active_indices: Vec<usize>,
    ) -> Result<Self, Error> {
        let ncol = active_indices.len();
        Ok(Self {
            block_start,
            n_pw,
            n_pw_i32,
            grid_size,
            inv_ntotal,
            superspace_max_bands,
            psi_dev,
            n_bands_total,
            r_vector,
            tpa_preconditioner,
            vnl_data,
            blas,
            solver,
            stream,
            handle,
            v_eff_dev,
            kinetic_dev,
            fft_idx_dev,
            fft_plan,
            kernels,
            r_beta_per_ion,
            q_rcq,
            q_rcq_gpu,
            r_beta_gpu,
            total_ne,
            active_indices,
            block_psi_temp: PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?),
            block_hpsi_temp: PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?),
            eig_block_dev: stream.alloc_zeros(ncol).map_err(Error::Cuda)?,
            search_dev: PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?),
            hsearch_dev: PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?,
            ),
            s_orth_in: PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?),
            s_orth_out: PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?),
        })
    }

    /// Reference to the search directions (used by the super_hamiltonian
    /// extension stage that follows this pipeline in the caller).
    fn search_dev(&self) -> &PwCoefficients {
        &self.search_dev
    }

    /// Run the full search-direction pipeline for one inner iteration.
    ///
    /// Stages:
    /// 1. Copy ψ and H·ψ for this block, upload eigenvalues
    /// 2. Apply TPA preconditioner → `search_dev`
    /// 3. Check superspace bounds, reset if full
    /// 4. S-orthogonalize against existing superspace
    /// 5. S-orthonormalize among themselves
    /// 6. Apply H → `hsearch_dev`
    /// 7. Copy into superspace buffers, update `superspace_index`
    /// Returns the number of valid search columns added to the superspace
    /// (may be less than `ncol` when bands are near convergence — CASTEP
    /// removes converged bands from the active set; we achieve the same by
    /// skipping near-zero search directions).
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn build(
        &mut self,
        super_wvfn: &mut PwCoefficients,
        h_super_wvfn: &mut PwCoefficients,
        superspace_index: &mut usize,
        grid_dev: &mut CudaSlice<CudaComplex>,
        // CASTEP hamiltonian.f90:403-408 — separate "slice" workspace holding
        // the active eigenvector estimates (copy of super_wvfn's first
        // slice_nbands columns).  Used as source for Stage 1 instead of the
        // global psi_dev/hpsi_dev arrays.
        slice_wvfn: &PwCoefficients,
        slice_h_wvfn: &PwCoefficients,
        // CASTEP hamiltonian.f90:926-928 — slice_eigenvalues consistent with
        // slice_wvfn/slice_h_wvfn.  Must be from the SAME pre-ZHEEVD state.
        slice_eigenvalues: &[f64],
        slice_nbands: usize,
    ) -> Result<usize, Error> {
        // --- Stage 1: Copy ψ and H·ψ from slice workspace to block temps ---
        // CASTEP hamiltonian.f90:404-409 copies slice and H_slice from
        // super_wvfn / H_super_wvfn into the slice workspace.  We follow
        // CASTEP's architecture: the slice (not the global psi_dev) holds
        // the active eigenvector estimates for this inner iteration.
        // With ADR-0005 lockstep transforms, hpsi_dev is kept consistent.
        {
            let (slice_ptr, _) = slice_wvfn.device_ptr(self.stream);
            let (h_slice_ptr, _) = slice_h_wvfn.device_ptr(self.stream);
            let (block_psi_mut, _) = self.block_psi_temp.device_ptr_mut(self.stream);
            let (block_hpsi_mut, _) = self.block_hpsi_temp.device_ptr_mut(self.stream);

            // CASTEP hamiltonian.f90:404-409 — slice columns are
            // sequential (compacted), no global index mapping needed.
            // active_bands maps compacted workspace col ci → global band gi
            // for eigenvalue lookup, but the wavefunction data lives in slice.
            for ci in 0..slice_nbands {
                // Copy slice_wvfn[ci] -> block_psi_temp[ci]
                cublasZcopy_v2(
                    self.handle,
                    self.n_pw_i32,
                    (slice_ptr as *const CudaComplex).add(ci * self.n_pw) as *const _,
                    1,
                    (block_psi_mut as *mut CudaComplex).add(ci * self.n_pw) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
                // Copy slice_h_wvfn[ci] -> block_hpsi_temp[ci]
                cublasZcopy_v2(
                    self.handle,
                    self.n_pw_i32,
                    (h_slice_ptr as *const CudaComplex).add(ci * self.n_pw) as *const _,
                    1,
                    (block_hpsi_mut as *mut CudaComplex).add(ci * self.n_pw) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }

            // Use ZHEEVD subspace eigenvalues for the preconditioner shift.
            // CASTEP hamiltonian.f90:629-642 — passes slice_eigenvalues from
            // the superspace diagonalization to nlpot_apply_precon_ES_slice.
            // Subspace eigenvalues incorporate band coupling via the full H_sub
            // matrix, producing more accurate USPP NL correction weights than
            // per-band Rayleigh quotients from freshly computed H·psi.
            let eig_block_cpu: Vec<f64> = active_bands(&self.active_indices, self.block_start)
                .map(|(ci, _gi)| slice_eigenvalues[ci])
                .collect();
            // Diagnostic: eigenvalues fed to preconditioner (first 3 + count)
            if self.block_start >= 104 {
                let n = eig_block_cpu.len();
                eprintln!("[Diag-PreconEig] block_start={} ncol={} eig[0..3]=[{:.6}, {:.6}, {:.6}] eig[{}..]={:.6}",
                    self.block_start, n,
                    eig_block_cpu.first().copied().unwrap_or(f64::NAN),
                    eig_block_cpu.get(1).copied().unwrap_or(f64::NAN),
                    eig_block_cpu.get(2).copied().unwrap_or(f64::NAN),
                    n.saturating_sub(1),
                    eig_block_cpu.last().copied().unwrap_or(f64::NAN));
            }
            self.stream
                .memcpy_htod(&eig_block_cpu, &mut self.eig_block_dev)
                .map_err(Error::Cuda)?;

            // D9: Check psi and hpsi magnitudes + first elements at Stage 1
            {
                let (psi_ptr, _) = self.block_psi_temp.device_ptr(self.stream);
                let (hpsi_ptr, _) = self.block_hpsi_temp.device_ptr(self.stream);

                // Check first band
                let pcol0 = (psi_ptr as *const CudaComplex).add(0);
                let mut psi_l2 = CudaComplex { x: 0.0, y: 0.0 };
                cublasZdotc_v2(self.handle, self.n_pw_i32,
                    pcol0 as *const _, 1, pcol0 as *const _, 1,
                    &mut psi_l2 as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                let hcol0 = (hpsi_ptr as *const CudaComplex).add(0);
                let mut hpsi_l2 = CudaComplex { x: 0.0, y: 0.0 };
                cublasZdotc_v2(self.handle, self.n_pw_i32,
                    hcol0 as *const _, 1, hcol0 as *const _, 1,
                    &mut hpsi_l2 as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;

                // Download first 5 elements of psi[0] for direct inspection
                let mut tmp_dl = self.stream.alloc_zeros::<CudaComplex>(5).map_err(Error::Cuda)?;
                cublasZcopy_v2(self.handle, 5,
                    psi_ptr as *const _, 1,
                    tmp_dl.device_ptr_mut(self.stream).0 as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                let psi5: Vec<CudaComplex> = self.stream.clone_dtoh(&tmp_dl).map_err(Error::Cuda)?;

                eprintln!(
                    "[Diag-D9] Stage1 (bs={}): psi[0] L2²={:.6e} first5=[{:?}] hpsi[0] L2²={:.6e} eps={:.6e}",
                    self.block_start, psi_l2.x,
                    psi5.iter().map(|c| (c.x, c.y)).collect::<Vec<_>>(),
                    hpsi_l2.x, eig_block_cpu[0]
                );

                // If anomaly detected in any eigenvalue, dump per-band psi/hpsi norms
                let has_anomaly = eig_block_cpu.iter().any(|e| e.abs() > 1e10 || e.is_nan());
                if has_anomaly {
                    eprintln!("[Diag-D9] ANOMALY detected in eigenvalues — dumping per-band psi/hpsi norms:");
                    for i in 0..self.active_indices.len() {
                        let pcol = (psi_ptr as *const CudaComplex).add(i * self.n_pw);
                        let hcol = (hpsi_ptr as *const CudaComplex).add(i * self.n_pw);
                        let mut p_nrm = CudaComplex { x: 0.0, y: 0.0 };
                        let mut h_nrm = CudaComplex { x: 0.0, y: 0.0 };
                        cublasZdotc_v2(self.handle, self.n_pw_i32,
                            pcol as *const _, 1, pcol as *const _, 1,
                            &mut p_nrm as *mut _ as *mut _,
                        ).result().map_err(Error::Blas)?;
                        cublasZdotc_v2(self.handle, self.n_pw_i32,
                            hcol as *const _, 1, hcol as *const _, 1,
                            &mut h_nrm as *mut _ as *mut _,
                        ).result().map_err(Error::Blas)?;
                        // Also download first element of hpsi for this band
                        let mut tmp_h = self.stream.alloc_zeros::<CudaComplex>(1).map_err(Error::Cuda)?;
                        cublasZcopy_v2(self.handle, 1,
                            hcol as *const _, 1,
                            tmp_h.device_ptr_mut(self.stream).0 as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                        let h_first: Vec<CudaComplex> = self.stream.clone_dtoh(&tmp_h).map_err(Error::Cuda)?;
                        eprintln!(
                            "  band {}: eig={:.6e}  |psi|²={:.6e}  |hpsi|²={:.6e}  hpsi[0]=({:.6e},{:.6e})",
                            i, eig_block_cpu[i], p_nrm.x, h_nrm.x,
                            h_first[0].x, h_first[0].y
                        );
                    }
                }
            }

        // --- Stage 2: TPA preconditioner → search_dev ---
        // CASTEP nlpot.f90:15970 — kernel computes (Hψ - ε·ψ) * R(G).
        // USPP correction applied afterwards via NL weights.
        let precon_result = unsafe {
            apply_preconditioner()
                .psi(&self.block_psi_temp)
                .hpsi(&self.block_hpsi_temp)
                .eigenvalues(&self.eig_block_dev)
                .r_vector(self.r_vector)
                .tpa_preconditioner(self.tpa_preconditioner)
                .n_bands(self.active_indices.len())
                .n_pw(self.n_pw)
                .stream(self.stream)
                .vnl_data(self.vnl_data)
                .blas(self.blas)
                .maybe_r_beta_per_ion(self.r_beta_per_ion)
                .maybe_q_rcq(self.q_rcq)
                .maybe_q_rcq_gpu(self.q_rcq_gpu)
                .maybe_r_beta_gpu(self.r_beta_gpu)
                .maybe_total_ne(self.total_ne)
                .maybe_kernels(Some(self.kernels))
                .call()?
        };
        self.stream
            .memcpy_dtod(&*precon_result, &mut self.search_dev.0)
            .map_err(Error::Cuda)?;
        // GPU-resident preconditioner uses temporary per-ion buffers that
        // are dropped at function exit. Sync here so those buffers aren't
        // freed while cublasZcopy is still reading from them.
        self.stream.synchronize().map_err(Error::Cuda)?;

        // Rust-SearchRaw: dump search direction coefficients for CASTEP comparison.
        // Matches CASTEP [CASTEP-SearchRaw] diagnostic at hamiltonian_searchspace_ks.
        // Triggered at SCF iter 1, block 0, inner iter 0 only.
        if self.block_start == 0 {
            let (search_ptr, _) = self.search_dev.device_ptr(self.stream);
            let col0 = (search_ptr as *const CudaComplex).add(0);
            let mut l2_sq = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(self.handle, self.n_pw_i32,
                col0 as *const _, 1, col0 as *const _, 1,
                &mut l2_sq as *mut _ as *mut _,
            ).result().map_err(Error::Blas)?;
            eprintln!("[Rust-SearchRaw] band=0 L2sq={:.16e}", l2_sq.x);

            // First 5 PW coefficients
            let n_dump = 5usize.min(self.n_pw);
            let mut tmp_f = self.stream.alloc_zeros::<CudaComplex>(n_dump).map_err(Error::Cuda)?;
            cublasZcopy_v2(self.handle, n_dump as i32,
                col0 as *const _, 1,
                tmp_f.device_ptr_mut(self.stream).0 as *mut _, 1,
            ).result().map_err(Error::Blas)?;
            let first5: Vec<CudaComplex> = self.stream.clone_dtoh(&tmp_f).map_err(Error::Cuda)?;
            eprint!("[Rust-SearchRaw] band=0 first5:");
            for c in &first5 {
                eprint!(" ({:.12e}, {:.12e})", c.x, c.y);
            }
            eprintln!();

            // Last 5 PW coefficients
            let start = self.n_pw.saturating_sub(n_dump);
            let col_last = (search_ptr as *const CudaComplex).add(start);
            let mut tmp_l = self.stream.alloc_zeros::<CudaComplex>(n_dump).map_err(Error::Cuda)?;
            cublasZcopy_v2(self.handle, n_dump as i32,
                col_last as *const _, 1,
                tmp_l.device_ptr_mut(self.stream).0 as *mut _, 1,
            ).result().map_err(Error::Blas)?;
            let last5: Vec<CudaComplex> = self.stream.clone_dtoh(&tmp_l).map_err(Error::Cuda)?;
            eprint!("[Rust-SearchRaw] band=0 last5:");
            for c in &last5 {
                eprint!(" ({:.12e}, {:.12e})", c.x, c.y);
            }
            eprintln!();
        }

        // D8: Check preconditioner output magnitude + dump first entries
        {
            let (search_ptr, _) = self.search_dev.device_ptr(self.stream);
            let col0 = (search_ptr as *const CudaComplex).add(0);
            let mut l2_sq = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(self.handle, self.n_pw_i32,
                col0 as *const _, 1, col0 as *const _, 1,
                &mut l2_sq as *mut _ as *mut _,
            ).result().map_err(Error::Blas)?;
            // Download first 10 search entries for direct inspection
            let mut tmp_s = self.stream.alloc_zeros::<CudaComplex>(10).map_err(Error::Cuda)?;
            cublasZcopy_v2(self.handle, 10,
                col0 as *const _, 1,
                tmp_s.device_ptr_mut(self.stream).0 as *mut _, 1,
            ).result().map_err(Error::Blas)?;
            let s10: Vec<CudaComplex> = self.stream.clone_dtoh(&tmp_s).map_err(Error::Cuda)?;
            eprintln!(
                "[Diag-D8] after Stage2 (bs={}): search[0] L2²={:.6e} first10=[{:?}]",
                self.block_start, l2_sq.x,
                s10.iter().map(|c| (c.x, c.y)).collect::<Vec<_>>()
            );
        }

        // --- Stage 3: Superspace bounds ---
        // CASTEP hamiltonian.f90:437-439 — when the superspace buffer would
        // overflow, wrap around to the END of the buffer:
        //   superspace_index = 1 + super_wvfn%nbands_max - slice_searchspace%nbands
        // (1-based; 0-based = superspace_max_bands - nbands).
        // This ensures new search directions are stored at the tail of the
        // superspace buffer, keeping the earlier columns (eigenstates from
        // previous iterations) contiguous at the front.
        if *superspace_index + self.active_indices.len() > self.superspace_max_bands {
            *superspace_index = self.superspace_max_bands - self.active_indices.len();
        }

        // --- Stage 3a: S-orthogonalize against ALL eigenvectors ---
        // Matches CASTEP hamiltonian_searchspace_ks → wave_Sorthogonalise(eigenvectors,
        // slice_searchspace) which S-orthogonalizes against the FULL eigenvector set
        // for every inner iteration, every block.
        // Reference: Cu111_CO.0001.profile confirms wave_Sorthogonalise_wv_slice
        // called 555 times (= 33 SCF × ~16.8 inner iters).
        //
        // CASTEP wave.f90:13388-13414 — batch ZGEMM with beta_phi projection.
        // Our code matches: apply_s_times recomputes S·search with beta_phi,
        // then ZGEMM computes overlap = psi^H * (S·search) and search -= psi * overlap.
        // Matches wave_Sorthogonalise_wv_slice exactly.
        if self.n_bands_total > 0 {
            for _pass in 0..1 {
                unsafe {
                    s_orthogonalise()
                        .search_dev(&mut self.search_dev)
                        .super_wvfn(self.psi_dev)
                        .superspace_index(self.n_bands_total)
                        .ncol(self.active_indices.len())
                        .n_pw(self.n_pw)
                        .n_pw_i32(self.n_pw_i32)
                        .vnl_data(self.vnl_data)
                        .blas(self.blas)
                        .stream(self.stream)
                        .s_orth_in(&mut self.s_orth_in)
                        .s_orth_out(&mut self.s_orth_out)
                        .handle(self.handle)
                        .call()?;
                }
            }
        }

        // ---- D6: Search column norm trace after Stage 3a ----
        {
            let (search_ptr, _) = self.search_dev.device_ptr(self.stream);
            let col0 = (search_ptr as *const CudaComplex).add(0);
            let mut l2_sq = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(self.handle, self.n_pw_i32,
                col0 as *const _, 1, col0 as *const _, 1,
                &mut l2_sq as *mut _ as *mut _,
            ).result().map_err(Error::Blas)?;
            // Compute S-norm
            let (s_in_mut, _) = self.s_orth_in.device_ptr_mut(self.stream);
            let (s_out_mut, _) = self.s_orth_out.device_ptr_mut(self.stream);
            cublasZcopy_v2(self.handle, self.n_pw_i32,
                col0 as *const _, 1, s_in_mut as *mut _, 1,
            ).result().map_err(Error::Blas)?;
            cublasZcopy_v2(self.handle, self.n_pw_i32,
                col0 as *const _, 1, s_out_mut as *mut _, 1,
            ).result().map_err(Error::Blas)?;
            unsafe {
                apply_s_times()
                    .psi_dev(&self.s_orth_in)
                    .spsi_dev(&mut self.s_orth_out)
                    .vnl_data(self.vnl_data)
                    .n_bands(1_i32)
                    .n_pw(self.n_pw_i32)
                    .blas(self.blas)
                    .stream(self.stream)
                    .call()?;
            }
            let (s_out_ptr, _) = self.s_orth_out.device_ptr(self.stream);
            let mut s_sq = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(self.handle, self.n_pw_i32,
                col0 as *const _, 1, s_out_ptr as *const _, 1,
                &mut s_sq as *mut _ as *mut _,
            ).result().map_err(Error::Blas)?;
            eprintln!(
                "[Diag-D6] after Stage3a (block_start={}): search[0] L2²={:.6e} S²={:.6e}",
                self.block_start, l2_sq.x, s_sq.x
            );
        }

        // ---- D3: Residual S-overlap diagnostic after Stage 3a ----
        // After S-orthogonalizing search directions against lower bands,
        // verify that the overlap is actually suppressed.  Large residual
        // overlap here means the s_orthogonalise formula (which assumes
        // unit S-norms on reference columns) is failing — the likely
        // proximate cause of subspace eigenvalue explosion in blocks 1+.
        if self.block_start > 0 {
            let (search_ptr, _) = self.search_dev.device_ptr(self.stream);
            let (psi_ptr, _) = self.psi_dev.device_ptr(self.stream);
            let (s_out_mut, _) = self.s_orth_out.device_ptr_mut(self.stream);
            let (s_in_mut, _) = self.s_orth_in.device_ptr_mut(self.stream);

            let n_lower = self.block_start.min(10); // sample first 10 lower bands
            let n_search = self.active_indices.len().min(5);         // sample first 5 search cols
            let mut max_overlap: f64 = 0.0;

            for si in 0..n_lower {
                let psi_si = (psi_ptr as *const CudaComplex).add(si * self.n_pw);
                // Precompute S·psi_si -> s_orth_out
                cublasZcopy_v2(self.handle, self.n_pw_i32,
                    psi_si as *const _, 1, s_in_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                cublasZcopy_v2(self.handle, self.n_pw_i32,
                    psi_si as *const _, 1, s_out_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                unsafe {
                    apply_s_times()
                        .psi_dev(&self.s_orth_in)
                        .spsi_dev(&mut self.s_orth_out)
                        .vnl_data(self.vnl_data)
                        .n_bands(1_i32)
                        .n_pw(self.n_pw_i32)
                        .blas(self.blas)
                        .stream(self.stream)
                        .call()?;
                }
                let (s_out_ptr, _) = self.s_orth_out.device_ptr(self.stream);

                for j in 0..n_search {
                    let search_j = (search_ptr as *const CudaComplex).add(j * self.n_pw);
                    let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                    cublasZdotc_v2(self.handle, self.n_pw_i32,
                        s_out_ptr as *const _, 1,
                        search_j as *const _, 1,
                        &mut dot as *mut _ as *mut _,
                    ).result().map_err(Error::Blas)?;
                    let abs_ov = (dot.x * dot.x + dot.y * dot.y).sqrt();
                    if abs_ov > max_overlap { max_overlap = abs_ov; }
                }
            }
            diag_detail!(
                "[Diag-D3] after Stage3a (block_start={}): max |S-overlap| = {:.3e} ({}×{} grid)",
                self.block_start, max_overlap, n_lower, n_search
            );
        }

        // --- Stage 4: S-orthogonalize against superspace ---
        // Same two-pass iterated Gram-Schmidt as Stage 3a above.
        // CASTEP hamiltonian.f90:442 — single pass of S-orthogonalize
        // against the superspace columns.
        unsafe {
            s_orthogonalise()
                .search_dev(&mut self.search_dev)
                .super_wvfn(&*super_wvfn)
                .superspace_index(*superspace_index)
                .ncol(self.active_indices.len())
                .n_pw(self.n_pw)
                .n_pw_i32(self.n_pw_i32)
                .vnl_data(self.vnl_data)
                .blas(self.blas)
                .stream(self.stream)
                .s_orth_in(&mut self.s_orth_in)
                .s_orth_out(&mut self.s_orth_out)
                .handle(self.handle)
                .call()?;
        }
        // Sync: s_orthogonalise allocated overlap_dev which is dropped at
        // scope exit. Ensure GEMMs reading/writing it are complete.
        self.stream.synchronize().map_err(Error::Cuda)?;

        // --- Stage 5: S-orthonormalize among themselves ---
        // No lockstep hpsi transform needed: Stage 6 overwrites hsearch_dev.
        unsafe {
            s_orthonormalise()
                .search_dev(&mut self.search_dev)
                .ncol(self.active_indices.len())
                .n_pw(self.n_pw)
                .n_pw_i32(self.n_pw_i32)
                .vnl_data(self.vnl_data)
                .blas(self.blas)
                .stream(self.stream)
                .handle(self.handle)
                .solver(self.solver)
                .s_orth_in(&mut self.s_orth_in)
                .s_orth_out(&mut self.s_orth_out)
                .call()?;
        }
        // Sync: s_orthonormalise uses cuSOLVER ZPOTRF/ZTRSM which may
        // use internal working streams.  Ensure they complete and write
        // results to search_dev before Stage 6 reads it.
        self.stream.synchronize().map_err(Error::Cuda)?;

        // --- Stage 6: Apply H to search directions ---
        unsafe {
            apply_full_hamiltonian()
                .psi_dev(&self.search_dev)
                .v_eff_dev(self.v_eff_dev)
                .kinetic_dev(self.kinetic_dev)
                .fft_idx_dev(self.fft_idx_dev)
                .n_pw(self.n_pw)
                .n_bands(self.active_indices.len())
                .grid_size(self.grid_size)
                .inv_ntotal(self.inv_ntotal)
                .fft_plan(self.fft_plan)
                .hpsi_dev(&mut self.hsearch_dev)
                .grid_dev(&mut *grid_dev)
                .vnl_data(self.vnl_data)
                .blas(self.blas)
                .kernels(self.kernels)
                .stream(self.stream)
                // IMPORTANT: Do NOT pass beta_phi_cache here.  The cache
                // stores β^H·super_wvfn projections, but Stage 6 applies H
                // to search directions (S-orthonormalized linear combinations
                // from the preconditioner), NOT super_wvfn columns.  Using
                // cached super_wvfn projections for search directions would
                // produce wrong NL contributions → wrong hsearch → corrupt
                // H_sub → zero eigenvalues in ZHEEVD.
                // (Default is None via bon::builder Option<T> parameter.)
                .call()?;
        }

        // D10: dump H·search entries for direct inspection
        {
            let (hsearch_ptr, _) = self.hsearch_dev.device_ptr(self.stream);
            let hcol0 = (hsearch_ptr as *const CudaComplex).add(0);
            let mut h_l2 = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(self.handle, self.n_pw_i32,
                hcol0 as *const _, 1, hcol0 as *const _, 1,
                &mut h_l2 as *mut _ as *mut _,
            ).result().map_err(Error::Blas)?;
            let mut tmp_h = self.stream.alloc_zeros::<CudaComplex>(10).map_err(Error::Cuda)?;
            cublasZcopy_v2(self.handle, 10,
                hcol0 as *const _, 1,
                tmp_h.device_ptr_mut(self.stream).0 as *mut _, 1,
            ).result().map_err(Error::Blas)?;
            let h10: Vec<CudaComplex> = self.stream.clone_dtoh(&tmp_h).map_err(Error::Cuda)?;
            // Also compute <psi|H|search> coupling for the active band
            let (search_ptr, _) = self.search_dev.device_ptr(self.stream);
            let scol0 = (search_ptr as *const CudaComplex).add(0);
            let mut coupling = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(self.handle, self.n_pw_i32,
                scol0 as *const _, 1,
                hcol0 as *const _, 1,
                &mut coupling as *mut _ as *mut _,
            ).result().map_err(Error::Blas)?;
            eprintln!(
                "[Diag-D10] after Stage6 H·search[0]: L2²={:.6e} ⟨search|H|search⟩=({:.6e},{:.6e}) first10=[{:?}]",
                h_l2.x, coupling.x, coupling.y,
                h10.iter().map(|c| (c.x, c.y)).collect::<Vec<_>>()
            );
        }

        // --- Stage 7: Copy search → superspace ---
        // CASTEP hamiltonian.f90:451-453 — copies ALL slice_searchspace columns
        // unconditionally; no validity pre-filter.
        {
            let (search_ptr, _) = self.search_dev.device_ptr(self.stream);
            let (hsearch_ptr, _) = self.hsearch_dev.device_ptr(self.stream);
            let (super_mut, _) = super_wvfn.device_ptr_mut(self.stream);
            let (h_super_mut, _) = h_super_wvfn.device_ptr_mut(self.stream);

            for i in 0..self.active_indices.len() {
                let dst = *superspace_index + i;
                cublasZcopy_v2(
                    self.handle,
                    self.n_pw_i32,
                    (search_ptr as *const CudaComplex).add(i * self.n_pw) as *const _,
                    1,
                    (super_mut as *mut CudaComplex).add(dst * self.n_pw) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
                cublasZcopy_v2(
                    self.handle,
                    self.n_pw_i32,
                    (hsearch_ptr as *const CudaComplex).add(i * self.n_pw) as *const _,
                    1,
                    (h_super_mut as *mut CudaComplex).add(dst * self.n_pw) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }

        Ok(self.active_indices.len())
    }
}
