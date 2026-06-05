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
    cublasHandle_t, cublasZaxpy_v2, cublasZcopy_v2, cublasZdotc_v2, cublasZscal_v2,
};
use cudarc::cusolver::sys::{cublasFillMode_t, cusolverEigMode_t};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::*;
use ndarray::Array2;
use num_complex::Complex64;
use crate::eigensolver::hamiltonian::{apply_full_hamiltonian, apply_s_times};
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
    //     Override convergence: band is NOT marked converged.
    let uphill_threshold = -100.0 * (f64::EPSILON).max(f64::EPSILON * prev_eig.abs());
    if prev_eig - new_eig < uphill_threshold {
        converged = false;
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
            beta_g_per_ion.push(
                Array2::from_shape_vec((n_pw, ne),
                    beta_host.iter().map(|c| Complex64::new(c.x, c.y)).collect()
                ).expect("beta_g shape (n_pw, ne) mismatch")
            );
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
        // CASTEP hamiltonian.f90:319 — wave_diagonalise(eigenvectors, H_eigenvectors,
        // eigenvalues). Builds H_sub = ψ^H·Hψ for ALL n_bands, solves the standard
        // EVP, and rotates all eigenvectors into the globally optimal eigenbasis.
        // Rayleigh quotients alone are poor eigenvalue estimates for cold-start
        // wavefunctions with similar character across blocks — cross-band mixing
        // is never captured.
        //
        // CASTEP hamiltonian.f90:319 — wave_diagonalise (full subspace rotation).
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
                .call()?;
        }
        stream
            .memcpy_dtod(&*psi_full_rotated, &mut psi_dev.0)
            .map_err(Error::Cuda)?;
        stream
            .memcpy_dtod(&*hpsi_full_rotated, &mut hpsi_dev.0)
            .map_err(Error::Cuda)?;
        davidson_diag!(
            "[davidson] full subspace diag: eigenvalues [{:.6}, ..., {:.6}]",
            eigenvalues[0],
                eigenvalues[n_bands - 1]
            );

        // ---- D1: S-norm diagnostic after A1 full-subspace ZHEGVD ----
        // Verify that rotated eigenvectors maintain ⟨psi|S|psi⟩ ≈ 1.
        // S-norm drift here contaminates lower-band reference columns for
        // subsequent blocks' S-orthogonalization (Stage 3a in build()).
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
        // Round to next even (CASTEP only does this for gamma-point, but
        // cuBLAS batched transforms benefit from even block sizes).
        let nblock_base = (2.0 * (n_bands as f64).sqrt()).floor() as usize;
        let nblock = (nblock_base + 1) / 2 * 2;
        // CASTEP hamiltonian.f90:1079 — superspace_size = 1 + min(max_iterations(1), 5)
        // With max_inner_iter = 10: 1 + min(10, 5) = 6.
        let superspace_size = 6_usize;
        let superspace_max_bands = superspace_size * nblock;

        // Allocate superspace buffers (reused across blocks)
        let super_alloc = n_pw * superspace_max_bands;
        let mut super_wvfn = PwCoefficients::new(
            stream.alloc_zeros(super_alloc).map_err(Error::Cuda)?);
        let mut h_super_wvfn = PwCoefficients::new(
            stream.alloc_zeros(super_alloc).map_err(Error::Cuda)?);

        // CPU-side dense Hermitian super_hamiltonian matrix
        let mut super_hamiltonian = vec![
            CudaComplex { x: 0.0, y: 0.0 };
            superspace_max_bands * superspace_max_bands
        ];

        davidson_diag!("[davidson] block loop: nblock={nblock} superspace_size={superspace_size}");

        // ---- Conduction state buffers (CASTEP hamiltonian.f90:392-401) ----
        let mut cond_wvfn = PwCoefficients::new(
            stream.alloc_zeros(super_alloc).map_err(Error::Cuda)?);
        let mut cond_h_wvfn = PwCoefficients::new(
            stream.alloc_zeros(super_alloc).map_err(Error::Cuda)?);
        let mut cond_count: usize = 0;

        for block_start in (0..n_bands).step_by(nblock) {
            let mut current_nblock = nblock.min(n_bands - block_start);

            // Skip if all bands in this block are converged
            if (block_start..block_start + current_nblock)
                .all(|b| band_converged[b])
            {
                davidson_diag!("[davidson]   block {block_start}..{}: skipped (all converged)", block_start+current_nblock);
                continue;
            }

            davidson_diag!("[davidson]   block {block_start}..{}: current_nblock={current_nblock}", block_start+current_nblock);

            // Copy block eigenvectors -> super_wvfn (first current_nblock bands)
            // Copy block H.psi -> h_super_wvfn
            {
                let (psi_ptr, _) = psi_dev.device_ptr(stream);
                let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
                let (super_mut, _) = super_wvfn.device_ptr_mut(stream);
                let (h_super_mut, _) = h_super_wvfn.device_ptr_mut(stream);

                for i in 0..current_nblock {
                    let src_off = (block_start + i) * n_pw;
                    let dst_off = i * n_pw;
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

            // Compute initial super_hamiltonian: H_sub = super_wvfn^H · h_super_wvfn
            // for the first current_nblock columns (k × k, where k = current_nblock).
            // Reference: hamiltonian.f90:1059 — wave_dot_all(super_wvfn, H_super_wvfn, super_hamiltonian)
            {
                let k = current_nblock;
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
                davidson_diag!("[davidson]     initial H_sub diag[0..3]: [{:.6}, {:.6}, {:.6}]",
                    h_init_cpu[0].x, h_init_cpu[k + 1].x, h_init_cpu[2 * k + 2].x);

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
                        super_hamiltonian[i * superspace_max_bands + j] = h_init_cpu[i * k + j];
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
            // CASTEP hamiltonian.f90:629-646 — after compaction, the active
            // workspace columns hold a subset of the original block bands.
            // active_indices[compacted_pos] = original_block_relative_index.
            // Global arrays (eigenvalues, psi_dev, hpsi_dev, band_converged,
            // break_cond_tols) always use ORIGINAL band indices throughout.
            let mut active_indices: Vec<usize> = (0..current_nblock).collect();
            let mut superspace_index = current_nblock;

            // ---- Conduction state seeding (CASTEP hamiltonian.f90:392-401) ----
            if cond_count > 0 {
                let count = cond_count.min(superspace_max_bands - current_nblock);
                let (super_mut, _) = super_wvfn.device_ptr_mut(stream);
                let (h_super_mut, _) = h_super_wvfn.device_ptr_mut(stream);
                let (cond_ptr, _) = cond_wvfn.device_ptr(stream);
                let (cond_h_ptr, _) = cond_h_wvfn.device_ptr(stream);
                for k in 0..count {
                    cublasZcopy_v2(handle, n_pw_i32,
                        (cond_ptr as *const CudaComplex).add(k * n_pw) as *const _, 1,
                        (super_mut as *mut CudaComplex).add((current_nblock + k) * n_pw) as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    cublasZcopy_v2(handle, n_pw_i32,
                        (cond_h_ptr as *const CudaComplex).add(k * n_pw) as *const _, 1,
                        (h_super_mut as *mut CudaComplex).add((current_nblock + k) * n_pw) as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
                superspace_index = current_nblock + count;
                davidson_diag!(
                    "[davidson]   block {block_start}: seeded {count} conduction states"
                );
            }

            // CASTEP hamiltonian.f90:427 — initialize previous_eigenvalues from
            // the current eigenvalue estimates, not zeros.  Zero initialization
            // causes the first inner-iteration convergence check to compare
            // against 0, which can create false convergence signals for
            // unoccupied bands that start near zero.
            let mut previous_eigenvalues: Vec<f64> =
                eigenvalues[block_start..block_start + ncol].to_vec();
            // Per-band opt_stop_condition (CASTEP: stagnation flag)
            let mut opt_stop_condition = vec![false; ncol];

            // block_ctx allocated ONCE outside the inner loop (GPU buffers reused).
            // Raw pointer casts are used for psi_dev/hpsi_dev copy-back to avoid
            // borrow conflicts — device_ptr_mut needs &mut self on CudaSlice.
            let mut block_ctx = DavidsonBlockCtx::new(
                &psi_dev, n_bands, &eigenvalues,
                block_start, n_pw, n_pw_i32,
                grid_size, inv_ntotal, superspace_max_bands,
                &r_vector, tpa_preconditioner, vnl_data,
                blas, stream, handle,
                v_eff_dev, kinetic_dev, fft_idx_dev,
                fft_plan, kernels,
                Some(&precon_prep.r_beta_per_ion), Some(&precon_prep.q_rcq),
                active_indices.clone(),
            )?;

            // Raw pointers to eigenvalues data (bypass borrow checker for writes)
            let eig_ptr: *mut f64 = eigenvalues.as_ptr() as *mut f64;

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
                // CASTEP hamiltonian.f90:427 — use ORIGINAL band indices
                // via active_indices mapping (compacted→original block position).
                for (ci, gi) in active_bands(&active_indices, block_start) {
                    previous_eigenvalues[ci] = eigenvalues[gi];
                }

                // (2)-(7) Build search directions: preconditioner → S-orth → S-orthonorm → H·search
                let n_added = unsafe {
                    block_ctx.build(
                        &mut super_wvfn,
                        &mut h_super_wvfn,
                        &mut superspace_index,
                        &mut grid_dev,
                    )?
                };

                // (8) Extend super_hamiltonian: compute new rows
                if n_added > 0 {
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
                                h_new_rows_cpu[i * new_total + j];
                        }
                    }

                    // Fill lower triangle via Hermitian conjugate
                    for i in 0..new_total {
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

                unsafe {
                    diagonalise_subspace()
                        .psi_block(&super_wvfn)
                        .hpsi_block(&h_super_wvfn)
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

                // ---- A2 (inner): S-orthogonalize + S-orthonormalize first
                // current_nblock columns after ZHEGVD rotation.
                // CASTEP hamiltonian.f90:515-516 — after ZHEGVD rotation, numerical
                // noise reintroduces lower-band components. Must re-orthogonalize.
                //
                // FIX F1: Always run S-orthonormalize, even for block 0.
                // Previously gated on `block_start > 0`, which skipped S-orthonormalize
                // for block 0.  ZHEGVD's diagonal regularization (1e-10 added to S_sub)
                // produces X^H·S_sub_reg·X = I, but the actual S-overlap becomes
                // I - 1e-10·X^H·X.  When X has large entries (near-linear-dependence),
                // S-norms drift from 1.0.  Without explicit S-orthonormalize, this
                // drift contaminates psi_dev and causes s_orthogonalise (which assumes
                // unit S-norms) to fail for subsequent blocks → near-singular S_sub
                // → subspace eigenvalue explosion.
                {
                    let mut inner_block_temp = PwCoefficients::new(
                        stream.alloc_zeros(current_nblock * n_pw).map_err(Error::Cuda)?);
                    {
                        let (super_ptr, _) = super_wvfn.device_ptr(stream);
                        let (temp_mut, _) = inner_block_temp.device_ptr_mut(stream);
                        for i in 0..current_nblock {
                            cublasZcopy_v2(handle, n_pw_i32,
                                (super_ptr as *const CudaComplex).add(i * n_pw) as *const _, 1,
                                (temp_mut as *mut CudaComplex).add(i * n_pw) as *mut _, 1,
                            ).result().map_err(Error::Blas)?;
                        }
                    }

                    let mut inner_s_orth_in = PwCoefficients::new(
                        stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
                    let mut inner_s_orth_out = PwCoefficients::new(
                        stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);

                    // S-orthogonalize against lower bands (only when there ARE lower bands)
                    if block_start > 0 {
                        for _pass in 0..2 {
                            unsafe {
                                s_orthogonalise()
                                    .search_dev(&mut inner_block_temp)
                                    .super_wvfn(&psi_dev)
                                    .superspace_index(block_start)
                                    .ncol(current_nblock)
                                    .n_pw(n_pw)
                                    .n_pw_i32(n_pw_i32)
                                    .vnl_data(vnl_data)
                                    .blas(blas)
                                    .stream(stream)
                                    .s_orth_in(&mut inner_s_orth_in)
                                    .s_orth_out(&mut inner_s_orth_out)
                                    .handle(handle)
                                    .call()?;
                            }
                        }
                    }

                    // Always S-orthonormalize — eliminates ZHEGVD regularization drift
                    // even for block 0 where there are no lower bands to orthogonalize
                    // against.
                    unsafe {
                        s_orthonormalise()
                            .search_dev(&mut inner_block_temp)
                            .ncol(current_nblock)
                            .n_pw(n_pw)
                            .n_pw_i32(n_pw_i32)
                            .vnl_data(vnl_data)
                            .blas(blas)
                            .stream(stream)
                            .handle(handle)
                            .s_orth_in(&mut inner_s_orth_in)
                            .s_orth_out(&mut inner_s_orth_out)
                            .call()?;
                    }

                    // Copy back to super_wvfn
                    {
                        let (super_mut, _) = super_wvfn.device_ptr_mut(stream);
                        let (temp_ptr, _) = inner_block_temp.device_ptr(stream);
                        for i in 0..current_nblock {
                            cublasZcopy_v2(handle, n_pw_i32,
                                (temp_ptr as *const CudaComplex).add(i * n_pw) as *const _, 1,
                                (super_mut as *mut CudaComplex).add(i * n_pw) as *mut _, 1,
                            ).result().map_err(Error::Blas)?;
                        }
                    }
                }

                // ---- A3: Copy updated eigenstates → psi_dev and hpsi_dev ----
                // CASTEP hamiltonian.f90:519 — wave_copy(super_wvfn, eigenvectors,
                // nb_src=1, nb_dst=nb, copy_bands=current_nblock). Updates the
                // working wavefunction so the NEXT inner iteration builds search
                // directions from the improved eigenvector approximation.
                // active_indices maps compacted workspace column → original
                // global band position (hamiltonian.f90:629-646).
                // Raw pointers bypass borrow-checker conflict with block_ctx.
                {
                    let (super_ptr, _) = super_wvfn.device_ptr(stream);
                    let (h_super_ptr, _) = h_super_wvfn.device_ptr(stream);

                    for (ci, gi) in active_bands(&active_indices, block_start) {
                        let dst_off = gi * n_pw;
                        cublasZcopy_v2(handle, n_pw_i32,
                            (super_ptr as *const CudaComplex).add(ci * n_pw) as *const _, 1,
                            psi_dev_raw.add(dst_off) as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                        cublasZcopy_v2(handle, n_pw_i32,
                            (h_super_ptr as *const CudaComplex).add(ci * n_pw) as *const _, 1,
                            hpsi_dev_raw.add(dst_off) as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                    }
                }

                // Update eigenvalues for current block (raw ptr bypasses block_ctx borrow)
                // active_indices maps super_wvfn column → original global band position.
                for (ci, gi) in active_bands(&active_indices, block_start) {
                    unsafe { *eig_ptr.add(gi) = inner_eigenvalues[ci]; }
                }
                davidson_diag!(
                    "[davidson]     inner iter {_inner_iter} eig: [{:.6}, ..., {:.6}]",
                    inner_eigenvalues[0],
                    inner_eigenvalues[current_nblock.min(k_super) - 1]
                );

                // ---- Convergence check (CASTEP hamiltonian.f90:541-617) ----
                // active_indices maps compacted workspace column → original
                // global band position; eigenvalues/band_converged use original indices.
                let mut inner_all_stopped = true;
                for (ci, gi) in active_bands(&active_indices, block_start) {
                    let prev_eig = previous_eigenvalues[ci];
                    let new_eig = eigenvalues[gi];
                    let delta_e = (prev_eig - new_eig).abs();
                    let eps_guard = 2.0 * new_eig.abs() * f64::EPSILON;

                    // Default: band is not converged
                    let mut band_conv = false;
                    let mut band_stopped = false;

                    // (a) Absolute tolerance
                    if delta_e < tol_abs.max(eps_guard) {
                        if !opt_stop_condition[ci] {
                            band_conv = true;
                        }
                    }

                    // (b) Relative break condition (CASTEP hamiltonian.f90:563-589)
                    if _inner_iter == 0 {
                        break_cond_tols[gi] = delta_e;
                    } else if tol_rel > 0.0 && delta_e < break_cond_tols[gi] * tol_rel {
                        // Relative tolerance: band is both converged AND stopped
                        band_conv = true;
                        band_stopped = true;
                    } else if tol_rel <= 0.0 {
                        // Stagnation check: improvement < 30% of first-step improvement
                        // CASTEP hamiltonian.f90:1217 — skip stagnation detection on
                        // the last outer iteration to avoid trapping the SCF loop.
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
                    if band_stopped {
                        opt_stop_condition[ci] = true;
                    } else {
                        opt_stop_condition[ci] = false;
                    }

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
                if inner_all_stopped {
                    for (ci, gi) in active_bands(&active_indices, block_start) {
                        band_converged[gi] = false;
                        let prev_eig = previous_eigenvalues[ci];
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

                // ---- Compaction: compact unconverged bands to front ----
                // CASTEP hamiltonian.f90:628-642 — after convergence check,
                // compact unconverged bands in super_wvfn / H_super_wvfn
                // (LOCAL workspace only).  The global arrays psi_dev, hpsi_dev,
                // eigenvalues, band_converged, and break_cond_tols are NEVER
                // rearranged — they stay in original band order.
                // Uses in-place COPY (not swap) so that unconverged bands
                // are contiguous at front; converged bands at positions >= j
                // get overwritten by search directions in next iteration.
                //
                // active_indices is rebuilt to map new compacted positions
                // → original block-relative indices.
                {
                    // First pass: count unconverged bands.
                    // active_indices[i_src] gives the ORIGINAL block-relative
                    // position of the band at compacted workspace column i_src.
                    let mut ncol_active: usize = 0;
                    for i_src in 0..ncol {
                        let global_idx = block_start + active_indices[i_src];
                        if !band_converged[global_idx] && !opt_stop_condition[i_src] {
                            ncol_active += 1;
                        }
                    }

                    if ncol_active < ncol {
                        // Raw pointers into super_wvfn / H_super_wvfn
                        // (LOCAL workspace — compaction is safe here)
                        let (super_devptr, _super_guard) =
                            super_wvfn.device_ptr_mut(stream);
                        let (hsuper_devptr, _hsuper_guard) =
                            h_super_wvfn.device_ptr_mut(stream);
                        let super_raw = super_devptr as *mut CudaComplex;
                        let hsuper_raw = hsuper_devptr as *mut CudaComplex;

                        let ssm = superspace_max_bands; // super_hamiltonian stride

                        // Second pass: copy unconverged bands to front
                        // (contiguous in local workspace; no global swaps)
                        // Rebuild active_indices in parallel.
                        let mut j: usize = 0;
                        let mut new_active_indices: Vec<usize> =
                            Vec::with_capacity(ncol_active);
                        for i_src in 0..ncol {
                            let global_idx = block_start + active_indices[i_src];
                            if !band_converged[global_idx] && !opt_stop_condition[i_src]
                            {
                                // Map new compacted position j → original
                                // block-relative index active_indices[i_src].
                                new_active_indices.push(active_indices[i_src]);
                                if i_src != j {
                                    // --- Copy GPU column in super_wvfn ---
                                    // CASTEP: local workspace copy only
                                    unsafe {
                                        cublasZcopy_v2(
                                            handle, n_pw_i32,
                                            super_raw.add(i_src * n_pw) as *const _, 1,
                                            super_raw.add(j * n_pw) as *mut _, 1,
                                        )
                                        .result()
                                        .map_err(Error::Blas)?;
                                    }

                                    // --- Copy GPU column in H_super_wvfn ---
                                    unsafe {
                                        cublasZcopy_v2(
                                            handle, n_pw_i32,
                                            hsuper_raw.add(i_src * n_pw) as *const _, 1,
                                            hsuper_raw.add(j * n_pw) as *mut _, 1,
                                        )
                                        .result()
                                        .map_err(Error::Blas)?;
                                    }

                                    // --- Copy (not swap) super_hamiltonian
                                    // diagonal entry ---
                                    // Diagonal entry tracks the column position
                                    // in super_wvfn. Zero out the source entry.
                                    super_hamiltonian[j * ssm + j] =
                                        super_hamiltonian[i_src * ssm + i_src];
                                    super_hamiltonian[i_src * ssm + i_src] =
                                        CudaComplex { x: 0.0, y: 0.0 };

                                    // --- Copy (not swap) per-block CPU arrays ---
                                    previous_eigenvalues[j] =
                                        previous_eigenvalues[i_src];
                                    opt_stop_condition[j] =
                                        opt_stop_condition[i_src];
                                }
                                j += 1;
                            }
                        } // end copy-to-front loop

                        // Swap in the rebuilt active_indices mapping.
                        active_indices = new_active_indices;

                        // Global arrays (psi_dev, hpsi_dev, eigenvalues,
                        // band_converged, break_cond_tols) are NOT touched —
                        // they remain in original band order throughout.

                        let ncol_old = ncol;
                        davidson_diag!(
                            "[davidson]     inner iter {}: compacted {} -> {} active bands (CASTEP hamiltonian.f90:628-642)",
                            _inner_iter, ncol_old, j
                        );
                        ncol = j;
                        current_nblock = j;
                        block_ctx.active_indices = active_indices.clone();
                        // CASTEP hamiltonian.f90:512 — superspace accumulates.
                        // Higher eigenstates (ncol_old..k_super-1) are untouched
                        // by compaction and remain valid enrichment for the
                        // next inner iteration's ZHEGVD.  Reduce only by the
                        // number of bands that converged/stopped (ncol_old - j).
                        if superspace_index > ncol_old + 1 {
                            superspace_index -= ncol_old.saturating_sub(j);
                        }
                        if j == 0 {
                            // All bands in this block converged — nothing left to iterate
                            break;
                        }
                    }
                } // end compaction block

                if n_added == 0 {
                    // All search columns were zero — nothing more to add
                    break;
                }
            } // end inner loop (for _inner_iter)

            // ---- Save conduction states for next block (CASTEP hamiltonian.f90:392-401) ----
            // Higher eigenstates (current_nblock..superspace_index) from the
            // final ZHEGVD are valid Ritz vectors beyond this block's bands.
            cond_count = superspace_index.saturating_sub(current_nblock);
            if cond_count > 0 {
                let count = cond_count.min(superspace_max_bands);
                let (super_ptr, _) = super_wvfn.device_ptr(stream);
                let (h_super_ptr, _) = h_super_wvfn.device_ptr(stream);
                let (cond_mut, _) = cond_wvfn.device_ptr_mut(stream);
                let (cond_h_mut, _) = cond_h_wvfn.device_ptr_mut(stream);
                for k in 0..count {
                    cublasZcopy_v2(handle, n_pw_i32,
                        (super_ptr as *const CudaComplex).add((current_nblock + k) * n_pw) as *const _, 1,
                        (cond_mut as *mut CudaComplex).add(k * n_pw) as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    cublasZcopy_v2(handle, n_pw_i32,
                        (h_super_ptr as *const CudaComplex).add((current_nblock + k) * n_pw) as *const _, 1,
                        (cond_h_mut as *mut CudaComplex).add(k * n_pw) as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
                davidson_diag!(
                    "[davidson]   block {block_start}: saved {count} conduction states for next block"
                );
            }

            // ---- D2: S-norm diagnostic after block 0 inner loop ----
            // Block 0's A2 (post-ZHEGVD S-orthonormalize) is skipped because
            // block_start == 0 (line 1463 guard).  Check whether ZHEGVD
            // regularization has caused S-norm drift in psi_dev[0..current_nblock].
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
    // Diagnostics: compute S⁻¹-weighted residual norms
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
            let mut sinv_r_dev = PwCoefficients::new(
                stream.alloc_zeros(n_elem).map_err(Error::Cuda)?);
            stream
                .memcpy_dtod(&*residual_dev, &mut sinv_r_dev.0)
                .map_err(Error::Cuda)?;
            unsafe {
                apply_s_inverse()
                    .hpsi_dev(&mut sinv_r_dev)
                    .vnl_data(vnl_data)
                    .n_bands(n_bands as i32)
                    .n_pw(n_pw as i32)
                    .blas(blas)
                    .stream(stream)
                    .solver(solver)
                    .call()?;
            }

            // ⟨r | S⁻¹·r⟩ → sqrt for each band
            let (residual_ptr, _) = residual_dev.device_ptr(stream);
            let (sinv_ptr, _) = sinv_r_dev.device_ptr(stream);
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

/// Build H_sub = ψ^H · H·ψ, diagonalize via ZHEEVD (standard EVP), and rotate ψ.
///
/// Matches CASTEP `algor_diagonalise` (hamiltonian.f90:476-480) — standard EVP
/// on S-orthonormal subspace vectors (S_sub = I implicitly). No overlap matrix
/// is needed because the caller keeps the subspace S-orthonormal.
///
/// `psi_block` — n_pw × k block of wavefunction columns (S-orthonormal)
/// `hpsi_block` — n_pw × k block of H·ψ columns
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
) -> Result<(), Error> {
    debug_assert_eq!(k, eigenvalues_out.len());
    let k_i32 = k as i32;
    let n_pw_i32 = n_pw as i32;

    // H_sub = ψ_block^H · hpsi_block  (k × k)
    let mut h_sub: CudaSlice<CudaComplex> =
        stream.alloc_zeros(k * k).map_err(Error::Cuda)?;
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
    // EVP on S-orthonormalized superspace vectors.  We match this faithfully
    // with ZHEEVD (standard Hermitian EVP) — no overlap matrix needed because
    // the superspace is S-orthonormal (ψ_block is S-orthogonal to lower bands
    // and S-orthonormal among themselves, so S_sub = I implicitly).
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
/// For each superspace column si, precompute S·super_si ONCE, then for
/// each search column j, compute dot = ⟨S·super_si | search_j⟩ and
/// subtract dot·super_si from search_j.
///
/// This avoids O(ncol × superspace_index) calls to the expensive
/// apply_s_times.
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
) -> Result<(), Error> {
    let (search_mut, _) = search_dev.device_ptr_mut(stream);
    let (super_ptr, _) = super_wvfn.device_ptr(stream);

    for si in 0..superspace_index {
        let super_si = (super_ptr as *const CudaComplex).add(si * n_pw);

        // Compute S·super_si ONCE -> s_orth_out
        let (s_in_mut, _) = s_orth_in.device_ptr_mut(stream);
        cublasZcopy_v2(
            handle,
            n_pw_i32,
            super_si as *const _,
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
            super_si as *const _,
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
        let (s_out_ptr, _) = s_orth_out.device_ptr(stream);

        // FIX F2: Compute S-norm of the reference column ONCE.
        // The projection formula search_j -= ⟨S·super_si|search_j⟩ · super_si
        // is correct ONLY when ⟨super_si|S|super_si⟩ = 1.  If the reference
        // column's S-norm has drifted from 1.0 (e.g. from ZHEGVD regularization
        // contamination), the projection is incomplete and residual S-overlap
        // accumulates across reference columns, making S_sub near-singular.
        // Dividing by the actual S-norm makes the formula correct regardless.
        let mut s_norm = CudaComplex { x: 0.0, y: 0.0 };
        cublasZdotc_v2(
            handle,
            n_pw_i32,
            super_si as *const _,
            1,
            s_out_ptr as *const _,
            1,
            &mut s_norm as *mut _ as *mut _,
        )
        .result()
        .map_err(Error::Blas)?;
        let s_norm_scale = if s_norm.x.abs() < 1e-30 {
            0.0 // degenerate: skip projection entirely
        } else {
            1.0 / s_norm.x
        };

        // Project ALL search columns against this superspace column
        // using the precomputed S·super_si
        for j in 0..ncol {
            let search_j = (search_mut as *mut CudaComplex).add(j * n_pw);

            let mut dot = CudaComplex { x: 0.0, y: 0.0 };
            cublasZdotc_v2(
                handle,
                n_pw_i32,
                s_out_ptr as *const _,
                1,
                search_j as *const _,
                1,
                &mut dot as *mut _ as *mut _,
            )
            .result()
            .map_err(Error::Blas)?;

            // search_j -= (dot / s_norm) * super_si
            let neg_dot = CudaComplex {
                x: -dot.x * s_norm_scale,
                y: -dot.y * s_norm_scale,
            };
            cublasZaxpy_v2(
                handle,
                n_pw_i32,
                &neg_dot as *const _ as *const _,
                super_si as *const _,
                1,
                search_j as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;
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
///   5. Recompute S·search_j (search_j changed in step 4)
///   6. S-norm normalize
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
) -> Result<(), Error> {
    let (search_mut, _) = search_dev.device_ptr_mut(stream);

    for j in 0..ncol {
        let search_j = (search_mut as *mut CudaComplex).add(j * n_pw);

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
        // Step 6: S-norm normalize
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
    }
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
    /// Global eigenvalues from the most recent ZHEEVD subspace
    /// diagonalization.  Indexed by global band index.  Passed to
    /// the preconditioner for USPP NL correction weight assembly
    /// (CASTEP hamiltonian.f90:629-642).
    eigenvalues: &'a [f64],
    /// Maps compacted workspace column → original global band index.
    /// Updated after compaction.
    active_indices: Vec<usize>,
    r_vector: &'a PreconditionerVector,
    tpa_preconditioner: &'a TpaPreconditioner,
    vnl_data: &'a VnlBatchData,
    blas: &'a BlasHandle,
    stream: &'a Arc<CudaStream>,
    handle: cublasHandle_t,
    v_eff_dev: &'a CudaSlice<f64>,
    kinetic_dev: &'a KineticPreconditioner,
    fft_idx_dev: &'a CudaSlice<i32>,
    fft_plan: &'a BatchedFftPlan3d,
    kernels: &'a CudaKernelSet,
    r_beta_per_ion: Option<&'a [Array2<Complex64>]>,
    q_rcq: Option<&'a Array2<Complex64>>,

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

impl<'a> DavidsonBlockCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        psi_dev: &'a PwCoefficients,
        n_bands_total: usize,  // total n_bands (for full eigenvector S-orth)
        eigenvalues: &'a [f64], // global eigenvalues from ZHEEVD (for preconditioner)
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
        stream: &'a Arc<CudaStream>,
        handle: cublasHandle_t,
        v_eff_dev: &'a CudaSlice<f64>,
        kinetic_dev: &'a KineticPreconditioner,
        fft_idx_dev: &'a CudaSlice<i32>,
        fft_plan: &'a BatchedFftPlan3d,
        kernels: &'a CudaKernelSet,
        r_beta_per_ion: Option<&'a [Array2<Complex64>]>,
        q_rcq: Option<&'a Array2<Complex64>>,
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
            eigenvalues,
            r_vector,
            tpa_preconditioner,
            vnl_data,
            blas,
            stream,
            handle,
            v_eff_dev,
            kinetic_dev,
            fft_idx_dev,
            fft_plan,
            kernels,
            r_beta_per_ion,
            q_rcq,
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
                stream.alloc_zeros(n_pw).map_err(Error::Cuda)?),
            s_orth_out: PwCoefficients::new(
                stream.alloc_zeros(n_pw).map_err(Error::Cuda)?),
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
    ) -> Result<usize, Error> {
        // --- Stage 1: Copy psi → temps, compute H·psi fresh, upload eigenvalues ---
        // FIX: Compute H·psi from scratch instead of copying stale hpsi_dev.
        // hpsi_dev is stale after A3 (copied from h_super_wvfn which A2 didn't
        // update). CASTEP has the same issue (hamiltonian.f90:404 copies stale
        // H_super_wvfn), but CASTEP's wave_Sorthogonalise_wv_slice corrects with
        // beta_phi projection. Our code doesn't store beta_phi, so we eliminate
        // staleness at the source by recomputing H·psi every build() call.
        {
            let (psi_ptr, _) = self.psi_dev.device_ptr(self.stream);
            let (block_psi_mut, _) = self.block_psi_temp.device_ptr_mut(self.stream);

            for (ci, gi) in active_bands(&self.active_indices, self.block_start) {
                cublasZcopy_v2(
                    self.handle,
                    self.n_pw_i32,
                    (psi_ptr as *const CudaComplex).add(gi * self.n_pw) as *const _,
                    1,
                    (block_psi_mut as *mut CudaComplex).add(ci * self.n_pw) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }

            // Compute H·psi fresh on the block temps
            unsafe {
                apply_full_hamiltonian()
                    .psi_dev(&self.block_psi_temp)
                    .v_eff_dev(self.v_eff_dev)
                    .kinetic_dev(self.kinetic_dev)
                    .fft_idx_dev(self.fft_idx_dev)
                    .n_pw(self.n_pw)
                    .n_bands(self.active_indices.len())
                    .grid_size(self.grid_size)
                    .inv_ntotal(self.inv_ntotal)
                    .fft_plan(self.fft_plan)
                    .hpsi_dev(&mut self.block_hpsi_temp)
                    .grid_dev(grid_dev)
                    .vnl_data(self.vnl_data)
                    .blas(self.blas)
                    .kernels(self.kernels)
                    .stream(self.stream)
                    .call()?;
            }

            // Use ZHEEVD subspace eigenvalues for the preconditioner shift.
            // CASTEP hamiltonian.f90:629-642 — passes slice_eigenvalues from
            // the superspace diagonalization to nlpot_apply_precon_ES_slice.
            // Subspace eigenvalues incorporate band coupling via the full H_sub
            // matrix, producing more accurate USPP NL correction weights than
            // per-band Rayleigh quotients from freshly computed H·psi.
            let eig_block_cpu: Vec<f64> = active_bands(&self.active_indices, self.block_start)
                .map(|(_ci, gi)| self.eigenvalues[gi])
                .collect();
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
                .call()?
        };
        self.stream
            .memcpy_dtod(&*precon_result, &mut self.search_dev.0)
            .map_err(Error::Cuda)?;

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

        // --- Stage 2b: Renormalize search columns to unit L2 norm ---
        // The USPP NL correction can produce search columns with L2² ~ 10^23
        // (Cu β-projectors have large norm).  S-orthogonalization and
        // S-orthonormalization in subsequent stages lose fp64 precision
        // when inputs span 11 orders of magnitude.  We normalize HERE,
        // before any orthogonalization, to keep all subsequent dot products
        // and axpy operations within the fp64 sweet spot (~10^0).
        // The direction is preserved; S-orthonormalization (Stage 5) would
        // normalize to S-norm=1 anyway — we just do it early.
        {
            let (s_ptr, _) = self.search_dev.device_ptr(self.stream);
            for b in 0..self.active_indices.len() {
                let col = (s_ptr as *const CudaComplex).add(b * self.n_pw);
                let mut nrm2 = CudaComplex { x: 0.0, y: 0.0 };
                cublasZdotc_v2(self.handle, self.n_pw_i32,
                    col as *const _, 1, col as *const _, 1,
                    &mut nrm2 as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                let inv_nrm = if nrm2.x > 1.0 { 1.0 / nrm2.x.sqrt() } else { 1.0 };
                if (inv_nrm - 1.0).abs() > 1e-15 {
                    let c = CudaComplex { x: inv_nrm, y: 0.0 };
                    cublasZscal_v2(self.handle, self.n_pw_i32,
                        &c as *const _ as *const _,
                        col as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }
        }

        // --- Stage 3: Superspace bounds ---
        if *superspace_index + self.active_indices.len() > self.superspace_max_bands {
            *superspace_index = self.active_indices.len();
        }

        // --- Stage 3a: S-orthogonalize against ALL eigenvectors ---
        // Matches CASTEP hamiltonian_searchspace_ks → wave_Sorthogonalise(eigenvectors,
        // slice_searchspace) which S-orthogonalizes against the FULL eigenvector set
        // for every inner iteration, every block.
        // Reference: Cu111_CO.0001.profile confirms wave_Sorthogonalise_wv_slice
        // called 555 times (= 33 SCF × ~16.8 inner iters).
        //
        // CASTEP uses batch ZGEMM with beta_phi projection (wave.f90:13388-13414).
        // Our code doesn't store beta_phi, but apply_s_times recomputes it from
        // current PW coefficients. The iterative s_orthogonalise (with F2's S-norm
        // denominator) is mathematically equivalent for S-orthonormal reference.
        //
        // With fresh H·psi from Stage 1, the preconditioner output is well-behaved,
        // making the iterative approach numerically stable even with 160 columns.
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
        for _pass in 0..2 {
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
        }

        // --- Stage 5: S-orthonormalize among themselves ---
        // CASTEP-aligned: columns with near-zero norm after S-orthogonalisation
        // correspond to converged bands.  We zero them out so that
        // S-orthonormalise skips them and they do not contaminate ZHEGVD.
        //
        // FIX F4: Use S-norm (⟨ψ|S|ψ⟩) not L2-norm (⟨ψ|ψ⟩) for the pre-filter.
        // For USPP, Q matrices can have negative eigenvalues, making S-norm ≪ L2-norm
        // for vectors with strong projector character.  A column with L2-norm ≈ 1
        // but S-norm ≈ 1e-20 would pass the L2 filter, then s_orthonormalise would
        // amplify it by 1/sqrt(1e-20) = 1e10, injecting extreme values into H_sub
        // and triggering subspace eigenvalue explosion to -10^78.
        // Threshold: S-norm² < 1e-12 (prevents amplification > 1e6×).
        let mut col_valid = vec![true; self.active_indices.len()];
        {
            let (search_ptr, _) = self.search_dev.device_ptr(self.stream);
            let (s_in_mut, _) = self.s_orth_in.device_ptr_mut(self.stream);
            let (s_out_mut, _) = self.s_orth_out.device_ptr_mut(self.stream);
            for j in 0..self.active_indices.len() {
                let search_j = (search_ptr as *const CudaComplex).add(j * self.n_pw);
                // Compute S·search_j -> s_orth_out
                cublasZcopy_v2(self.handle, self.n_pw_i32,
                    search_j as *const _, 1, s_in_mut as *mut _, 1,
                ).result().map_err(Error::Blas)?;
                cublasZcopy_v2(self.handle, self.n_pw_i32,
                    search_j as *const _, 1, s_out_mut as *mut _, 1,
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
                // S-norm² = ⟨search_j | S | search_j⟩
                let (s_out_ptr, _) = self.s_orth_out.device_ptr(self.stream);
                let mut s_nrm_sq = CudaComplex { x: 0.0, y: 0.0 };
                cublasZdotc_v2(
                    self.handle, self.n_pw_i32,
                    search_j as *const _, 1,
                    s_out_ptr as *const _, 1,
                    &mut s_nrm_sq as *mut _ as *mut _,
                ).result().map_err(Error::Blas)?;
                if s_nrm_sq.x < 1e-12 {
                    col_valid[j] = false;
                    // Zero the column so S-orthonormalise will skip it
                    let (search_mut, _) = self.search_dev.device_ptr_mut(self.stream);
                    let col_mut = (search_mut as *mut CudaComplex).add(j * self.n_pw);
                    cublasZscal_v2(
                        self.handle, self.n_pw_i32,
                        &CudaComplex { x: 0.0, y: 0.0 } as *const _ as *const _,
                        col_mut as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }
        }

        // --- Stage 5: S-orthonormalize among themselves ---
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
                .s_orth_in(&mut self.s_orth_in)
                .s_orth_out(&mut self.s_orth_out)
                .call()?;
        }

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

        // --- Stage 7: Copy search → superspace (only valid columns) ---
        let mut valid_count: usize;
        {
            let (search_ptr, _) = self.search_dev.device_ptr(self.stream);
            let (hsearch_ptr, _) = self.hsearch_dev.device_ptr(self.stream);
            let (super_mut, _) = super_wvfn.device_ptr_mut(self.stream);
            let (h_super_mut, _) = h_super_wvfn.device_ptr_mut(self.stream);

            valid_count = 0;
            for i in 0..self.active_indices.len() {
                if !col_valid[i] {
                    continue;
                }
                let dst = *superspace_index + valid_count;
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
                valid_count += 1;
            }
        }

        Ok(valid_count)
    }
}

// ======================================================================
// CPU unit tests (no GPU required)
// ======================================================================

#[cfg(test)]
mod tests {
    use super::{lock_tol_for_iter, check_band_converged};

    // -----------------------------------------------------------------------
    // lock_tol_for_iter tests (existing)
    // -----------------------------------------------------------------------

    #[test]
    fn lock_tol_for_iter_baseline() {
        // iter 1: lock_tol = 0.2 (initial)
        let tol_1 = lock_tol_for_iter(1, 1e-6);
        assert!(
            (tol_1 - 0.2).abs() < 1e-15,
            "iter 1 lock_tol = {tol_1}, expected 0.2"
        );

        // iter 2: lock_tol = target + (0.2 - target) * 0.5^1
        let expected_2 = 1e-6 + (0.2 - 1e-6) * 0.5_f64.powi(1);
        let tol_2 = lock_tol_for_iter(2, 1e-6);
        assert!(
            (tol_2 - expected_2).abs() < 1e-15,
            "iter 2 lock_tol = {tol_2}, expected {expected_2}"
        );

        // iter 3: lock_tol = target + (0.2 - target) * 0.5^2
        let expected_3 = 1e-6 + (0.2 - 1e-6) * 0.5_f64.powi(2);
        let tol_3 = lock_tol_for_iter(3, 1e-6);
        assert!(
            (tol_3 - expected_3).abs() < 1e-15,
            "iter 3 lock_tol = {tol_3}, expected {expected_3}"
        );

        // iter 10: should still be between target_tol and initial_tol
        let tol_10 = lock_tol_for_iter(10, 1e-6);
        assert!(
            tol_10 > 1e-6 - 1e-15,
            "iter 10 lock_tol = {tol_10} fell below target 1e-6"
        );

        // Edge: scf_iter = 0 (should behave like iter-1)
        let tol_0 = lock_tol_for_iter(0, 1e-6);
        assert!(
            (tol_0 - 0.2).abs() < 1e-15,
            "iter 0 lock_tol = {tol_0}, expected 0.2"
        );
    }

    #[test]
    fn lock_tol_for_iter_convergence_asymptotic() {
        let tol = lock_tol_for_iter(100, 1e-6);
        let diff = (tol - 1e-6).abs();
        assert!(
            diff < 1e-10,
            "iter 100 lock_tol = {tol} is {diff} from target 1e-6, should be very close"
        );
    }

    #[test]
    fn lock_tol_for_iter_zero_target_tol() {
        let tol = lock_tol_for_iter(100, 0.0);
        assert!(
            tol < 1e-10,
            "iter 100 with target 0.0: lock_tol = {tol}, expected near 0"
        );
    }

    // -----------------------------------------------------------------------
    // check_band_converged tests (outer-loop convergence)
    // -----------------------------------------------------------------------

    #[test]
    fn check_band_converged_small_diff_below_tol_abs() {
        // diff < tol_abs → converged
        assert!(check_band_converged(1.0, 1.0 + 1e-10, 1e-8));
    }

    #[test]
    fn check_band_converged_large_diff_above_tol_abs() {
        // diff > tol_abs → not converged
        assert!(!check_band_converged(1.0, 1.1, 1e-8));
    }

    #[test]
    fn check_band_converged_exact_zero_diff() {
        // zero diff → converged
        assert!(check_band_converged(1.0, 1.0, 1e-8));
    }

    #[test]
    fn check_band_converged_eps_guard_below_threshold() {
        // For large eigenvalues, the EPS term dominates tol_abs.
        // threshold ≈ 2*|1e10|*EPS ≈ 4.44e-6
        // diff = threshold * 0.5 < threshold → converged
        let eig = 1e10_f64;
        let threshold = 2.0 * eig.abs() * f64::EPSILON;
        assert!(check_band_converged(eig, eig + threshold * 0.5, 1e-8));
    }

    #[test]
    fn check_band_converged_eps_guard_above_threshold() {
        // diff = threshold * 5.0 > threshold → not converged
        // (Large margin avoids f64 rounding at 1e10 scale)
        let eig = 1e10_f64;
        let threshold = 2.0 * eig.abs() * f64::EPSILON;
        assert!(!check_band_converged(eig, eig + threshold * 5.0, 1e-8));
    }

    #[test]
    fn check_band_converged_tol_abs_dominates_for_small_eig() {
        // For small eigenvalues, tol_abs dominates.
        // threshold = max(1e-6, 2*|1.0|*EPS) = 1e-6 (tol_abs is larger)
        // diff = 5e-7 < 1e-6 → converged
        assert!(check_band_converged(1.0, 1.0 + 5e-7, 1e-6));

        // diff = 2e-6 > 1e-6 → not converged
        assert!(!check_band_converged(1.0, 1.0 + 2e-6, 1e-6));
    }

    #[test]
    fn check_band_converged_negative_eigenvalues() {
        // Negative eigenvalues should be handled correctly
        // |(-10.0 - (-10.0 + 1e-9))| = 1e-9 < 1e-8 → converged
        assert!(check_band_converged(-10.0, -10.0 + 1e-9, 1e-8));

        // |(-10.0 - (-11.0))| = 1.0 > 1e-8 → not converged
        assert!(!check_band_converged(-10.0, -11.0, 1e-8));
    }

    #[test]
    fn check_band_converged_zero_tol_abs() {
        // With zero tol_abs, only EPS guard protects
        // diff = 1e-8, threshold = max(0.0, 2*|1.0|*EPS) ≈ 4.4e-16
        // diff > threshold → not converged
        assert!(!check_band_converged(1.0, 1.0 + 1e-8, 0.0));

        // diff = 0.0 < threshold → converged
        assert!(check_band_converged(1.0, 1.0, 0.0));
    }

    #[test]
    fn check_band_converged_very_large_eigenvalue() {
        // For extremely large eigenvalues, EPS guard dominates
        // threshold ≈ 2 * |1e15| * EPS ≈ 4.4e-1
        let eig = 1e15_f64;
        let threshold = 2.0 * eig.abs() * f64::EPSILON;
        // diff just below threshold → converged
        assert!(check_band_converged(eig, eig + threshold * 0.5, 1e-8));
        // diff just above threshold → not converged
        assert!(!check_band_converged(eig, eig + threshold * 2.0, 1e-8));
    }
}
