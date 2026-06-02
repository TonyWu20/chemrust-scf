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
//   9. Unified k×k ZHEGVD (USPP-augmented S_sub)
//  10. Rotate: ψ_new = ψ_u · X
//  11. S-orthogonalize ψ_new against locked bands (one Gram–Schmidt pass)
//  12. Scatter locked + rotated → ψ_out; return
//
// Rationale: For continuation initial guesses (CASTEP .check or prior SCF
// iterate), ψ lives ε-close to the eigenvectors of (H,S). A single unified
// ZHEGVD on the unconverged sub-block gives the exact solution within that
// span. SCF-level convergence is driven by the lock ratchet tightening across
// iterations, not by an outer eigensolver loop.
// See notes/plans/phase-eigensolver-migration/PHASE1A_POSTMORTEM.md §3.
//
// TpaPreconditioner and detect_degenerate_blocks are preserved for future
// cold-start work but are not called in this single-sweep variant.

use std::sync::Arc;

use cudarc::cublas::sys::{
    cublasZaxpy_v2, cublasZcopy_v2, cublasZdotc_v2, cublasZscal_v2,
};
use cudarc::cusolver::sys::{cublasFillMode_t, cusolverEigMode_t};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::*;
use crate::eigensolver::hamiltonian::{apply_full_hamiltonian, apply_s_inverse, apply_s_times};
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::preconditioner::{apply_preconditioner, compute_r_vector, TpaPreconditioner};
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
    /// Currently unused — per-block ZHEGVD destroys global eigenvalue
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
// Lock tolerance ratchet schedule
// ---------------------------------------------------------------------------

/// Compute Davidson lock tolerance for a given SCF iteration.
///
/// Starts at 0.2 Ha — loose enough to lock most bands at iter-1
/// (max observed S⁻¹ residual ≈ 0.104 Ha for continuation ψ with our V_eff),
/// tightens geometrically toward `target_tol`.
///
/// S⁻¹-weighted norms are ~100× larger than plain L2 norms for the Cu111+CO
/// system. Phase 0's 0.5 Ha (calibrated for L2) maps to ~0.2 Ha for S⁻¹.
pub(crate) fn lock_tol_for_iter(scf_iter: usize, target_tol: f64) -> f64 {
    let initial: f64 = 0.2;
    let decay: f64 = 0.5;
    if scf_iter <= 1 {
        initial
    } else {
        let gap = initial - target_tol;
        (target_tol + gap * decay.powi(scf_iter as i32 - 1)).max(target_tol)
    }
}

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
/// 9. Unified k×k ZHEGVD (USPP-augmented S_sub), k = n_unconv
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
    // Steps 9–10: Unified k×k ZHEGVD + rotation
    // ------------------------------------------------------------------
    let mut psi_unconv_new = PwCoefficients::new(
        stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?);
    let mut eig_dev: CudaSlice<f64> = stream.alloc_zeros(k).map_err(Error::Cuda)?;
    let mut info_dev: CudaSlice<i32> = stream.alloc_zeros(1).map_err(Error::Cuda)?;
    let mut eigenvalues_k = vec![0.0_f64; k];

    solve_block_zhegvd()
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
// Outer-loop convergence check
// ======================================================================

/// Check whether a band has converged based on eigenvalue stability.
///
/// A band is converged if the eigenvalue change after subspace rotation
/// is below the threshold: `|prev - new| < max(tol_abs, 2*|new|*EPS)`.
///
/// The EPS guard prevents near-zero tol_abs from demanding convergence
/// beyond machine precision for large eigenvalues.
///
/// # Arguments
/// - `prev`: eigenvalue before subspace rotation
/// - `new`: eigenvalue after subspace rotation
/// - `tol_abs`: absolute convergence tolerance (Hartree)
///
/// Returns `true` if the band is converged.
pub(crate) fn check_band_converged(prev: f64, new: f64, tol_abs: f64) -> bool {
    let diff = (prev - new).abs();
    let threshold = tol_abs.max(2.0 * new.abs() * f64::EPSILON);
    diff < threshold
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

    // (b) Relative break condition
    if is_first_step {
        *break_cond_tol = delta_e;
    } else if delta_e < *break_cond_tol * 1e-15 {
        // If delta_e is essentially zero relative to break_cond_tol,
        // the band is numerically converged (or delta_e already below
        // machine precision). Neither abs tol nor stagnation matters.
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
///    d. Subspace diagonalization via `solve_block_zhegvd` (full n_bands)
///    e. Rotate ψ via ZHEGVD eigenvector matrix
///    f. Convergence check: |prev_eig - new_eig| < max(tol_abs, 2*|new_eig|*EPS)
///    g. Set H_correct = true (H·ψ was just computed)
///    h. After rotation, H·ψ is stale → H_correct = false
///
/// # Notes
/// - The preconditioner call and inner block loop are NOT implemented yet
///   (TODO for future tasks).
/// - Uses the existing `solve_block_zhegvd` for the full n_bands × n_bands
///   subspace (not block-by-block).
///
/// # Safety
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
    max_outer_iter: usize,
    blas: &BlasHandle,
    solver: &SolverHandle,
    kernels: &CudaKernelSet,
    tpa_preconditioner: &TpaPreconditioner,
    stream: &Arc<CudaStream>,
    _ctx: &Arc<CudaContext>,
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

    // ZHEGVD GPU buffers (reused across iterations in E-3/E-4 block solve)
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

    eprintln!("[davidson] start: n_bands={n_bands} n_pw={n_pw} tol_abs={tol_abs:.1e} max_outer={max_outer_iter}");

    // ------------------------------------------------------------------
    // Outer loop
    // ------------------------------------------------------------------
    #[allow(unused_assignments)]
    for iteration in 0..max_outer_iter {
        let n_conv = band_converged.iter().filter(|&&c| c).count();
        eprintln!("[davidson] outer iter {iteration}: {n_conv}/{n_bands} converged");

        // Step a: exit if all bands converged
        if band_converged.iter().all(|&c| c) {
            eprintln!("[davidson] all converged, exiting outer loop");
            break;
        }

        // Step b: compute H·ψ if needed
        if !h_correct {
            eprintln!("[davidson] computing H·psi...");
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
            eprintln!("[davidson] H·psi done");
        }

        // Step c: save previous eigenvalues
        let prev_eigenvalues = eigenvalues.clone();
        eprintln!("[davidson] prev eigenvalues: [{:.6}, ..., {:.6}]",
                  prev_eigenvalues[0], prev_eigenvalues[n_bands-1]);

        // Compute TPA preconditioner R(G) vector (used by all blocks in this iter)
        let kinetic_host: Vec<f64> = stream.clone_dtoh(&**kinetic_dev).map_err(Error::Cuda)?;
        let mean_ek = kinetic_host.iter().sum::<f64>() / kinetic_host.len() as f64;
        let r_vector = compute_r_vector(kinetic_dev, mean_ek, n_pw, stream)?;

        // ------------------------------------------------------------------
        // Block loop with superspace management
        //
        // CASTEP hamiltonian.f90:1019-1063 — block loop over groups of nblock
        // bands. Each unconverged block runs an inner Davidson loop (residual,
        // S-orthogonalize, H·search, extend superspace, ZHEGVD).
        // ------------------------------------------------------------------
        let nblock_base = (2.0 * (n_bands as f64).sqrt()).ceil() as usize;
        let nblock = (nblock_base + 1) / 2 * 2; // round to next even
        let superspace_size = 4;
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

        eprintln!("[davidson] block loop: nblock={nblock} superspace_size={superspace_size}");

        for block_start in (0..n_bands).step_by(nblock) {
            let current_nblock = nblock.min(n_bands - block_start);

            // Skip if all bands in this block are converged
            if (block_start..block_start + current_nblock)
                .all(|b| band_converged[b])
            {
                eprintln!("[davidson]   block {block_start}..{}: skipped (all converged)", block_start+current_nblock);
                continue;
            }

            eprintln!("[davidson]   block {block_start}..{}: current_nblock={current_nblock}", block_start+current_nblock);

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
                // D2H: copy into CPU super_hamiltonian
                let h_init_cpu: Vec<CudaComplex> = stream.clone_dtoh(&h_init).map_err(Error::Cuda)?;
                eprintln!("[davidson]     initial H_sub diag[0..3]: [{:.6}, {:.6}, {:.6}]",
                    h_init_cpu[0 * k + 0].x, h_init_cpu[1 * k + 1].x, h_init_cpu[2 * k + 2].x);
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
            // Inner Davidson loop: build search directions, expand superspace
            // ------------------------------------------------------------------
            let max_inner_iter = superspace_size - 1;
            let ncol = current_nblock;
            let mut superspace_index = current_nblock;
            let mut previous_eigenvalues: Vec<f64> = vec![0.0_f64; ncol];

            // Temporary buffers for the inner loop
            let mut search_dev = PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?);
            let mut hsearch_dev = PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?);

            // Block-level temp buffers for TPA preconditioner (Gap 2)
            let mut block_psi_temp = PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?);
            let mut block_hpsi_temp = PwCoefficients::new(
                stream.alloc_zeros(n_pw * ncol).map_err(Error::Cuda)?);
            let mut eig_block_dev: CudaSlice<f64> =
                stream.alloc_zeros(ncol).map_err(Error::Cuda)?;

            // Single-column temp buffers for S-orthogonalization (Gap 3)
            let mut s_orth_in = PwCoefficients::new(
                stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);
            let mut s_orth_out = PwCoefficients::new(
                stream.alloc_zeros(n_pw).map_err(Error::Cuda)?);

            for _inner_iter in 0..max_inner_iter {
                eprintln!("[davidson]     inner iter {_inner_iter}: superspace_index={superspace_index}");
                // (1) Save previous eigenvalues for convergence tracking
                for b in 0..ncol {
                    previous_eigenvalues[b] = eigenvalues[block_start + b];
                }

                // (2) Build preconditioned search direction via TPA
                {
                    let (psi_ptr, _) = psi_dev.device_ptr(stream);
                    let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
                    let (block_psi_mut, _) = block_psi_temp.device_ptr_mut(stream);
                    let (block_hpsi_mut, _) = block_hpsi_temp.device_ptr_mut(stream);

                    for i in 0..ncol {
                        let b = block_start + i;
                        cublasZcopy_v2(handle, n_pw_i32,
                            (psi_ptr as *const CudaComplex).add(b * n_pw) as *const _, 1,
                            (block_psi_mut as *mut CudaComplex).add(i * n_pw) as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                        cublasZcopy_v2(handle, n_pw_i32,
                            (hpsi_ptr as *const CudaComplex).add(b * n_pw) as *const _, 1,
                            (block_hpsi_mut as *mut CudaComplex).add(i * n_pw) as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                    }

                    // Upload block eigenvalues to GPU
                    let eig_block_cpu: Vec<f64> = (0..ncol)
                        .map(|i| eigenvalues[block_start + i])
                        .collect();
                    stream.memcpy_htod(&eig_block_cpu, &mut eig_block_dev)
                        .map_err(Error::Cuda)?;

                    // Apply TPA preconditioner: search = (hpsi - lambda·psi) * R(G)
                    let precon_result = unsafe {
                        apply_preconditioner()
                            .psi(&block_psi_temp)
                            .hpsi(&block_hpsi_temp)
                            .eigenvalues(&eig_block_dev)
                            .r_vector(&r_vector)
                            .tpa_preconditioner(tpa_preconditioner)
                            .n_bands(ncol)
                            .n_pw(n_pw)
                            .stream(stream)
                            .call()?
                    };

                    // Copy result to search_dev
                    stream.memcpy_dtod(&*precon_result, &mut search_dev.0)
                        .map_err(Error::Cuda)?;
                }

                // (3) Check superspace bounds: reset if full
                if superspace_index + ncol > superspace_max_bands {
                    superspace_index = ncol;
                }

                // (4) S-orthogonalize search directions against lower superspace
                //     Precompute S·super_si once per superspace column, reuse
                //     for all search columns (avoids O(ncol × superspace_index)
                //     calls to the expensive apply_s_times).
                {
                    let (search_mut, _) = search_dev.device_ptr_mut(stream);
                    let (super_ptr, _) = super_wvfn.device_ptr(stream);

                    for si in 0..superspace_index {
                        let super_si = (super_ptr as *const CudaComplex).add(si * n_pw);

                        // Compute S·super_si ONCE → s_orth_out
                        let (s_in_mut, _) = s_orth_in.device_ptr_mut(stream);
                        cublasZcopy_v2(handle, n_pw_i32,
                            super_si as *const _, 1,
                            s_in_mut as *mut _, 1,
                        ).result().map_err(Error::Blas)?;

                        let (s_out_mut, _) = s_orth_out.device_ptr_mut(stream);
                        cublasZcopy_v2(handle, n_pw_i32,
                            super_si as *const _, 1,
                            s_out_mut as *mut _, 1,
                        ).result().map_err(Error::Blas)?;

                        unsafe {
                            apply_s_times()
                                .psi_dev(&s_orth_in)
                                .spsi_dev(&mut s_orth_out)
                                .vnl_data(vnl_data)
                                .n_bands(1_i32)
                                .n_pw(n_pw_i32)
                                .blas(blas)
                                .stream(stream)
                                .call()?;
                        }
                        let (s_out_ptr, _) = s_orth_out.device_ptr(stream);

                        // Project ALL search columns against this superspace column
                        // using the precomputed S·super_si
                        for j in 0..ncol {
                            let search_j = (search_mut as *mut CudaComplex).add(j * n_pw);

                            let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                            cublasZdotc_v2(handle, n_pw_i32,
                                s_out_ptr as *const _, 1,
                                search_j as *const _, 1,
                                &mut dot as *mut _ as *mut _,
                            ).result().map_err(Error::Blas)?;

                            // search_j -= dot * super_si
                            let neg_dot = CudaComplex { x: -dot.x, y: -dot.y };
                            cublasZaxpy_v2(handle, n_pw_i32,
                                &neg_dot as *const _ as *const _,
                                super_si as *const _, 1,
                                search_j as *mut _, 1,
                            ).result().map_err(Error::Blas)?;
                        }
                    }
                }

                // (5) Orthonormalize search directions among themselves
                //     Modified Gram-Schmidt + L2 normalization
                {
                    let (search_mut, _) = search_dev.device_ptr_mut(stream);

                    for j in 0..ncol {
                        let search_j = (search_mut as *mut CudaComplex).add(j * n_pw);

                        // Skip zero residual (band already in superspace)
                        let mut nrm = CudaComplex { x: 0.0, y: 0.0 };
                        cublasZdotc_v2(handle, n_pw_i32,
                            search_j as *const _, 1,
                            search_j as *const _, 1,
                            &mut nrm as *mut _ as *mut _,
                        ).result().map_err(Error::Blas)?;
                        if nrm.x < 1e-30 {
                            continue;
                        }

                        // Orthogonalize against earlier search columns (MGS)
                        for i in 0..j {
                            let search_i = (search_mut as *const CudaComplex).add(i * n_pw);
                            let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                            cublasZdotc_v2(handle, n_pw_i32,
                                search_i as *const _, 1,
                                search_j as *const _, 1,
                                &mut dot as *mut _ as *mut _,
                            ).result().map_err(Error::Blas)?;

                            let neg_dot = CudaComplex { x: -dot.x, y: -dot.y };
                            cublasZaxpy_v2(handle, n_pw_i32,
                                &neg_dot as *const _ as *const _,
                                search_i as *const _, 1,
                                search_j as *mut _, 1,
                            ).result().map_err(Error::Blas)?;
                        }

                        // Normalize
                        let mut nrm2 = CudaComplex { x: 0.0, y: 0.0 };
                        cublasZdotc_v2(handle, n_pw_i32,
                            search_j as *const _, 1,
                            search_j as *const _, 1,
                            &mut nrm2 as *mut _ as *mut _,
                        ).result().map_err(Error::Blas)?;
                        let inv_norm = 1.0 / nrm2.x.sqrt();
                        let scale = CudaComplex { x: inv_norm, y: 0.0 };
                        cublasZscal_v2(handle, n_pw_i32,
                            &scale as *const _ as *const _,
                            search_j as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                    }
                }

                // (6) Apply H to search directions
                unsafe {
                    apply_full_hamiltonian()
                        .psi_dev(&search_dev)
                        .v_eff_dev(v_eff_dev)
                        .kinetic_dev(kinetic_dev)
                        .fft_idx_dev(fft_idx_dev)
                        .n_pw(n_pw)
                        .n_bands(ncol)
                        .grid_size(grid_size)
                        .inv_ntotal(inv_ntotal)
                        .fft_plan(fft_plan)
                        .hpsi_dev(&mut hsearch_dev)
                        .grid_dev(&mut grid_dev)
                        .vnl_data(vnl_data)
                        .blas(blas)
                        .kernels(kernels)
                        .stream(stream)
                        .call()?;
                }

                // (7) Copy search → super_wvfn and H·search → h_super_wvfn
                {
                    let (search_ptr, _) = search_dev.device_ptr(stream);
                    let (hsearch_ptr, _) = hsearch_dev.device_ptr(stream);
                    let (super_mut, _) = super_wvfn.device_ptr_mut(stream);
                    let (h_super_mut, _) = h_super_wvfn.device_ptr_mut(stream);

                    for i in 0..ncol {
                        let dst = superspace_index + i;
                        cublasZcopy_v2(handle, n_pw_i32,
                            (search_ptr as *const CudaComplex).add(i * n_pw) as *const _, 1,
                            (super_mut as *mut CudaComplex).add(dst * n_pw) as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                        cublasZcopy_v2(handle, n_pw_i32,
                            (hsearch_ptr as *const CudaComplex).add(i * n_pw) as *const _, 1,
                            (h_super_mut as *mut CudaComplex).add(dst * n_pw) as *mut _, 1,
                        ).result().map_err(Error::Blas)?;
                    }
                }

                // (8) Extend super_hamiltonian: compute new rows
                //     new_rows = search^H · h_super_wvfn[:, 0:superspace_index+ncol]
                let new_total = superspace_index + ncol;
                let mut h_new_rows: CudaSlice<CudaComplex> = stream
                    .alloc_zeros(ncol * new_total)
                    .map_err(Error::Cuda)?;
                unsafe {
                    blas.gemm_c64(
                        ZgemmConfig {
                            transa: op::C,
                            transb: op::N,
                            m: ncol as i32,
                            n: new_total as i32,
                            k: n_pw_i32,
                            alpha: CudaComplex { x: 1.0, y: 0.0 },
                            lda: n_pw_i32,
                            ldb: n_pw_i32,
                            beta: CudaComplex { x: 0.0, y: 0.0 },
                            ldc: ncol as i32,
                        },
                        &search_dev,
                        &h_super_wvfn,
                        &mut h_new_rows,
                    )?;
                }

                // D2H: copy new rows into CPU super_hamiltonian
                let h_new_rows_cpu: Vec<CudaComplex> = stream
                    .clone_dtoh(&h_new_rows)
                    .map_err(Error::Cuda)?;
                for i in 0..ncol {
                    for j in 0..new_total {
                        super_hamiltonian
                            [(superspace_index + i) * superspace_max_bands + j] =
                            h_new_rows_cpu[i * new_total + j];
                    }
                }

                // (9) Fill lower triangle via Hermitian conjugate
                for i in 0..new_total {
                    for j in 0..i {
                        let val = super_hamiltonian[j * superspace_max_bands + i];
                        super_hamiltonian[i * superspace_max_bands + j] = CudaComplex {
                            x: val.x,
                            y: -val.y,
                        };
                    }
                }

                superspace_index += ncol;
            }

            // ------------------------------------------------------------------
            // Solve the superspace GEP via ZHEGVD
            // ------------------------------------------------------------------
            let k_super = superspace_index;
            let mut psi_rotated_super = PwCoefficients::new(
                stream.alloc_zeros(n_pw * k_super).map_err(Error::Cuda)?);
            let mut h_rotated_super = PwCoefficients::new(
                stream.alloc_zeros(n_pw * k_super).map_err(Error::Cuda)?);
            let mut super_eigenvalues = vec![0.0_f64; k_super];

            unsafe {
                solve_block_zhegvd()
                    .psi_block(&super_wvfn)
                    .hpsi_block(&h_super_wvfn)
                    .vnl_data(vnl_data)
                    .k(k_super)
                    .n_pw(n_pw)
                    .blas(blas)
                    .solver(solver)
                    .stream(stream)
                    .eigenvalues_out(&mut super_eigenvalues)
                    .eig_dev(&mut eig_dev)
                    .info_dev(&mut info_dev)
                    .psi_rotated(&mut psi_rotated_super)
                    .hpsi_rotated(&mut h_rotated_super)
                    .call()?;
            }

            // Copy first current_nblock eigenvectors back to psi_dev and hpsi_dev
            {
                let (psi_mut, _) = psi_dev.device_ptr_mut(stream);
                let (hpsi_mut, _) = hpsi_dev.device_ptr_mut(stream);
                let (rotated_ptr, _) = psi_rotated_super.device_ptr(stream);
                let (h_rotated_ptr, _) = h_rotated_super.device_ptr(stream);

                for i in 0..current_nblock {
                    let dst_off = (block_start + i) * n_pw;
                    cublasZcopy_v2(handle, n_pw_i32,
                        (rotated_ptr as *const CudaComplex).add(i * n_pw) as *const _, 1,
                        (psi_mut as *mut CudaComplex).add(dst_off) as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                    cublasZcopy_v2(handle, n_pw_i32,
                        (h_rotated_ptr as *const CudaComplex).add(i * n_pw) as *const _, 1,
                        (hpsi_mut as *mut CudaComplex).add(dst_off) as *mut _, 1,
                    ).result().map_err(Error::Blas)?;
                }
            }

            // Update eigenvalues for current block (first current_nblock from solve)
            for i in 0..current_nblock {
                eigenvalues[block_start + i] = super_eigenvalues[i];
            }
            eprintln!("[davidson]     block eigenvalues: [{:.6}, ..., {:.6}]",
                      super_eigenvalues[0], super_eigenvalues[current_nblock-1]);
        }

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
        eprintln!("[davidson] after convergence check: {n_conv}/{n_bands} converged, eigenvalues: [{:.6}, ..., {:.6}]",
                  eigenvalues[0], eigenvalues[n_bands-1]);

        // Step h: after rotation, H·ψ is stale
        h_correct = false;

        n_outer_completed = iteration + 1;
    }

    // ------------------------------------------------------------------
    // Result
    // ------------------------------------------------------------------
    Ok(DavidsonResult {
        psi_out: psi_dev.0,
        eigenvalues,
        n_locked: band_converged.iter().filter(|&&c| c).count(),
        residual_norms_sinv: ResidualSInvNorm::new(vec![0.0_f64; n_bands]),
        n_outer_iterations: n_outer_completed,
    })
}

// ======================================================================
// Helper: solve a block's generalized eigenvalue problem
// ======================================================================

/// Build H_sub, S_sub (USPP-augmented), call ZHEGVD, and rotate ψ.
///
/// `psi_block` — n_pw × k block of wavefunction columns
/// `hpsi_block` — n_pw × k block of H·ψ columns
/// On return, `psi_rotated` holds the rotated eigenbasis.
#[builder]
#[allow(clippy::too_many_arguments)]
unsafe fn solve_block_zhegvd(
    psi_block: &PwCoefficients,
    hpsi_block: &PwCoefficients,
    vnl_data: &VnlBatchData,
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

    // S_sub = ψ_block^H · ψ_block  (bare PW overlap, k × k)
    let mut s_sub: CudaSlice<CudaComplex> =
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
            psi_block,
            &mut s_sub,
        )?;
    }

    // Add USPP overlap: S += Σ_ion c_proj^H · q · c_proj
    for entry in &vnl_data.entries {
        let ne = entry.n_expanded;

        // c_proj = beta_g^H · psi_block  (ne × k)
        let mut c_proj: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * k).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n: k_i32,
                    k: n_pw_i32,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw_i32,
                    ldb: n_pw_i32,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_block,
                &mut c_proj,
            )?;
        }

        // temp = q_matrix · c_proj  (ne × k)
        let mut temp: CudaSlice<CudaComplex> =
            stream.alloc_zeros(ne as usize * k).map_err(Error::Cuda)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::N,
                    transb: op::N,
                    m: ne,
                    n: k_i32,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.q_matrix,
                &c_proj,
                &mut temp,
            )?;
        }

        // S_sub += c_proj^H · temp  (k × k, accumulated)
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: k_i32,
                    n: k_i32,
                    k: ne,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: ne,
                    ldb: ne,
                    beta: CudaComplex { x: 1.0, y: 0.0 }, // accumulate
                    ldc: k_i32,
                },
                &c_proj,
                &temp,
                &mut s_sub,
            )?;
        }
    }

    // Resize eigenvalue/info GPU buffers if needed
    if eig_dev.len() < k {
        *eig_dev = stream.alloc_zeros(k).map_err(Error::Cuda)?;
    }
    if info_dev.is_empty() {
        *info_dev = stream.alloc_zeros(1).map_err(Error::Cuda)?;
    }

    // Regularize S_sub diagonal to ensure positive-definiteness.
    // USPP Q_aug matrices can have negative eigenvalues, making
    // S = I + β^H·Q·β indefinite for some projector combinations.
    // A small ε on the diagonal stabilises ZHEGVD without affecting
    // eigenvalues at the 1e-10 Ha level.
    {
        let mut s_sub_cpu: Vec<CudaComplex> = stream.clone_dtoh(&s_sub).map_err(Error::Cuda)?;
        for i in 0..k {
            s_sub_cpu[i * k + i].x += 1e-10;
        }
        let s_reg = stream.clone_htod(&s_sub_cpu).map_err(Error::Cuda)?;
        core::mem::drop(core::mem::replace(&mut s_sub, s_reg));
    }

    // ZHEGVD: H_sub · X = Λ · S_sub · X
    solver.zhegvd(
        cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        k_i32,
        &mut h_sub, // overwritten → eigenvectors X
        &mut s_sub, // overwritten (scratch)
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
