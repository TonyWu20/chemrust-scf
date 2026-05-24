// ---------------------------------------------------------------------------
// Production Davidson v1 eigensolver (Phase 1A Group C)
// ---------------------------------------------------------------------------
//
// Dead-code allowed: Group D (scf.rs dispatch) wires these types and the main
// driver into the live code path. Until then they are unreachable from any
// public entry point.
#![allow(dead_code)]
//
// Full outer-iteration Davidson with:
//   1. H|ψ⟩ and S|ψ⟩ via apply_full_hamiltonian/apply_s_times
//   2. Per-band Rayleigh quotient, residual, S⁻¹-weighted residual norm
//   3. Per-band locking (residual norm + eigenvalue delta)
//   4. Block partitioning via detect_degenerate_blocks
//   5. Per-block ZHEGVD (USPP-augmented S_sub)
//   6. TPA preconditioned corrections
//   7. S-orthogonalization against subspace (Gram-Schmidt)
//   8. Subspace management with restart
//
// Reference: Zhou (2014), "Davidson eigenvector locking pattern";
//            Seelecke et al. (2021), "Subspace restart in block Davidson".

use std::sync::Arc;

use cudarc::cublas::sys::{
    cublasDznrm2_v2, cublasZaxpy_v2, cublasZcopy_v2, cublasZdscal_v2, cublasZdotc_v2,
};
use cudarc::cusolver::sys::{cublasFillMode_t, cusolverEigMode_t};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::solver::SolverHandle;
use crate::device::CudaComplex;
use crate::eigensolver::hamiltonian::{apply_full_hamiltonian, apply_s_inverse, apply_s_times};
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::preconditioner::TpaPreconditioner;
use crate::eigensolver::rayleigh_ritz::detect_degenerate_blocks;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Davidson v1 eigensolver.
pub(crate) struct DavidsonConfig {
    /// Maximum number of outer iterations (default 30).
    pub max_outer_iter: usize,
    /// Eigenvalue spacing threshold (Ha) for degenerate-block detection (default 0.01).
    pub block_eps_degen: f64,
    /// Max subspace dimension as fraction of n_active (default 3.0).
    pub max_subspace_dim_factor: f64,
    /// TPA diagonal preconditioner.
    pub preconditioner: TpaPreconditioner,
}

// ---------------------------------------------------------------------------
// Result types and diagnostics
// ---------------------------------------------------------------------------

/// Result of the Davidson v1 diagonalization.
pub(crate) struct DavidsonResult {
    /// Output wavefunctions (column-major, n_pw × n_bands).
    pub psi_out: CudaSlice<CudaComplex>,
    /// Per-band eigenvalues (sorted ascending).
    pub eigenvalues: Vec<f64>,
    /// Number of outer iterations completed.
    pub n_outer_iters: usize,
    /// Number of locked bands on exit.
    pub n_locked: usize,
    /// S⁻¹-weighted residual norms for all bands.
    pub residual_norms_sinv: Vec<f64>,
}

/// Snapshot of Davidson diagnostics after the most recent solve.
#[derive(Clone)]
pub struct DavidsonDiagnostic {
    pub n_locked: usize,
    pub n_unconverged: usize,
    pub n_davidson_iters: usize,
    pub locked_indices: Vec<usize>,
    pub unconv_indices: Vec<usize>,
    pub residual_norms_sinv: Vec<f64>,
    pub max_residual_sinv: f64,
    pub lock_tol: f64,
    pub eigenvalue_deltas: Vec<f64>,
    pub blocks: Vec<(usize, usize)>,
    pub n_restarts: usize,
}

/// Most recent Davidson v1 diagnostic, accessible for tests.
pub static DAVIDSON_LAST_DIAG: std::sync::Mutex<Option<DavidsonDiagnostic>> =
    std::sync::Mutex::new(None);

// ---------------------------------------------------------------------------
// Lock tolerance ratchet schedule
// ---------------------------------------------------------------------------

/// Compute Davidson lock tolerance for a given SCF iteration.
///
/// Starts at 0.5 Ha (proven by Phase 0 — all 160 bands lock at iter-1),
/// tightens geometrically toward target_tol.
pub(crate) fn lock_tol_for_iter(scf_iter: usize, target_tol: f64) -> f64 {
    // iter 1: loose enough to lock the trivially-converged bands but tight
    // enough to force ZHEGVD on bands whose residual exceeds ~0.01 Ha.
    // This prevents the vacuous all-lock problem where eigenvectors stay
    // frozen and eigenvalues drift unchecked across SCF iterations.
    let tol_at_iter1: f64 = 0.01;
    let decay: f64 = 0.5;
    if scf_iter <= 1 {
        tol_at_iter1
    } else {
        let gap = tol_at_iter1 - target_tol;
        let tol = target_tol + gap * decay.powi(scf_iter as i32 - 1);
        tol.max(target_tol)
    }
}

// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

/// Run the production Davidson v1 eigensolver.
///
/// # Algorithm
///
/// Outer iteration: while not all bands locked:
///   1. Hψ = apply_full_hamiltonian(ψ_current)
///   2. Sψ = ψ_current; apply_s_times(ψ_current, Sψ)
///   3. Per-band Rayleigh quotient λ_b = Re⟨ψ_b|Hψ_b⟩ / Re⟨ψ_b|Sψ_b⟩
///   4. Residual r_b = Hψ_b − λ_b·Sψ_b
///   5. S⁻¹-weighted norm: sinv_r = S⁻¹·r; norm = √Re⟨r|sinv_r⟩
///   6. Lock bands where norm < lock_tol AND |Δλ_b| < lock_tol
///   7. If all locked: exit
///   8. Block-partition unconverged via detect_degenerate_blocks
///   9. Per-block: H_sub/S_sub (USPP augmented), ZHEGVD, rotate
///  10. Preconditioned correction: t_b = P⁻¹·r_b
///  11. S-orthogonalize corrections against subspace (Gram-Schmidt)
///  12. Append to subspace; restart if dim > max_subspace_dim
///
/// # Safety
///
/// All device pointers must be valid and of sufficient size. `psi_init` must
/// be an S-orthonormalized wavefunction set (column-major, n_bands × n_pw).
///
/// NOTE: `ctx` is accepted for API consistency with other solvers in this
/// crate but is not directly used (allocations go through `stream`).
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn davidson_v1(
    psi_init: &CudaSlice<CudaComplex>,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &CudaSlice<f64>,
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
    cfg: &DavidsonConfig,
    prev_lambdas: Option<&[f64]>,
) -> Result<DavidsonResult, Error> {
    // ----- Convenience constants -----
    let n_elem = n_bands * n_pw;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;
    let handle = blas.raw_handle();
    let max_subspace = (cfg.max_subspace_dim_factor * n_bands as f64).ceil() as usize;

    // ------------------------------------------------------------------
    // Helper: grow subspace buffers when capacity is exhausted
    // ------------------------------------------------------------------
    /// Reallocate subspace and sspace buffers with `new_cap` columns,
    /// preserving existing data. Returns the new capacity on success.
    unsafe fn ensure_subspace_cap(
        subspace_dev: &mut CudaSlice<CudaComplex>,
        sspace_dev: &mut CudaSlice<CudaComplex>,
        current_dim: usize,
        new_cap: usize,
        n_pw: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<usize, Error> {
        let mut new_sub =
            stream.alloc_zeros(n_pw * new_cap).map_err(Error::Cuda)?;
        let mut new_sspace =
            stream.alloc_zeros(n_pw * new_cap).map_err(Error::Cuda)?;
        // Copy existing data (first `current_dim` columns)
        if current_dim > 0 {
            stream
                .memcpy_dtod(subspace_dev, &mut new_sub)
                .map_err(Error::Cuda)?;
            stream
                .memcpy_dtod(sspace_dev, &mut new_sspace)
                .map_err(Error::Cuda)?;
        }
        *subspace_dev = new_sub;
        *sspace_dev = new_sspace;
        Ok(new_cap)
    }

    // ------------------------------------------------------------------
    // Persistent buffer allocation (reused across outer iterations)
    // ------------------------------------------------------------------
    let mut psi_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    stream
        .memcpy_dtod(psi_init, &mut psi_dev)
        .map_err(Error::Cuda)?;

    let mut hpsi_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut spsi_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut residual_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let mut sinv_r_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    let grid_alloc = n_bands * grid_size;
    let mut grid_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(grid_alloc).map_err(Error::Cuda)?;

    // Preconditioner temporary per-band buffer (n_pw)
    let mut t_buf: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    // Preconditioner output buffer (separate from t_buf for borrow-checker)
    let mut precond_buf: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;
    // Single-column S·t temp for subspace S-orthogonalization updates
    let mut s_temp_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw).map_err(Error::Cuda)?;

    // Subspace buffers: start at n_bands columns, grow dynamically.
    // This avoids pre-allocating for the worst-case max_subspace (3× n_bands),
    // which would waste ~4 GB on large systems before any search directions
    // are accumulated.
    let mut subspace_cap: usize = n_bands;
    let mut subspace_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw * subspace_cap).map_err(Error::Cuda)?;
    let mut sspace_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_pw * subspace_cap).map_err(Error::Cuda)?;

    // ------------------------------------------------------------------
    // Initialise subspace with psi_init
    // ------------------------------------------------------------------
    // 1. Copy psi_init → subspace_dev[:, :n_bands]
    stream
        .memcpy_dtod(psi_init, &mut subspace_dev)
        .map_err(Error::Cuda)?;
    // 2. Copy identity term: subspace_dev → sspace_dev
    stream
        .memcpy_dtod(&subspace_dev, &mut sspace_dev)
        .map_err(Error::Cuda)?;
    // 3. Compute S·subspace: accumulates β·Q·β^H into sspace_dev
    unsafe {
        apply_s_times(
            &subspace_dev,
            &mut sspace_dev,
            vnl_data,
            n_bands_i32,
            n_pw_i32,
            blas,
            stream,
        )?;
    }
    let mut subspace_dim: usize = n_bands;

    // ------------------------------------------------------------------
    // Tracking state
    // ------------------------------------------------------------------
    let mut locked: Vec<bool> = vec![false; n_bands];
    let mut eigenvalues: Vec<f64> = vec![0.0; n_bands];
    let mut residual_norms_sinv: Vec<f64> = vec![0.0; n_bands];
    let mut prev_outer_lambdas: Vec<f64> = match prev_lambdas {
        Some(pl) => pl.to_vec(),
        None => vec![0.0; n_bands],
    };
    let mut skip_delta_check: bool = prev_lambdas.is_none();
    let mut n_restarts: usize = 0;

    // Allocate conversion buffers used repeatedly
    let mut eig_dev: CudaSlice<f64> = stream.alloc_zeros(1).map_err(Error::Cuda)?;
    let mut info_dev: CudaSlice<i32> = stream.alloc_zeros(1).map_err(Error::Cuda)?;

    // ==================================================================
    // Outer iteration loop
    // ==================================================================
    for outer_iter in 0..cfg.max_outer_iter {
        // ---------------------------------------------------------------
        // Step 1: Hψ = apply_full_hamiltonian(psi_dev)
        // ---------------------------------------------------------------
        unsafe {
            apply_full_hamiltonian(
                &psi_dev,
                v_eff_dev,
                kinetic_dev,
                fft_idx_dev,
                n_pw,
                n_bands,
                grid_size,
                inv_ntotal,
                fft_plan,
                &mut hpsi_dev,
                &mut grid_dev,
                vnl_data,
                blas,
                kernels,
                stream,
            )?;
        }

        // ---------------------------------------------------------------
        // Step 2: Sψ — pre-copy psi_dev into spsi_dev, then apply_s_times
        // ---------------------------------------------------------------
        stream
            .memcpy_dtod(&psi_dev, &mut spsi_dev)
            .map_err(Error::Cuda)?;
        unsafe {
            apply_s_times(
                &psi_dev,
                &mut spsi_dev,
                vnl_data,
                n_bands_i32,
                n_pw_i32,
                blas,
                stream,
            )?;
        }

        // ---------------------------------------------------------------
        // Step 3: Per-band Rayleigh quotient
        //   λ_b = Re⟨ψ_b|Hψ_b⟩ / Re⟨ψ_b|Sψ_b⟩
        // ---------------------------------------------------------------
        {
            let (psi_ptr, _) = psi_dev.device_ptr(stream);
            let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
            let (spsi_ptr, _) = spsi_dev.device_ptr(stream);

            for (b, lambda) in eigenvalues.iter_mut().enumerate() {
                let psi_b = (psi_ptr as *const CudaComplex).add(b * n_pw);
                let hpsi_b = (hpsi_ptr as *const CudaComplex).add(b * n_pw);
                let spsi_b = (spsi_ptr as *const CudaComplex).add(b * n_pw);

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

                *lambda = dot_h.x / dot_s.x;
            }
        }

        // ---------------------------------------------------------------
        // Step 4: Residual r_b = Hψ_b − λ_b · Sψ_b
        // ---------------------------------------------------------------
        stream
            .memcpy_dtod(&hpsi_dev, &mut residual_dev)
            .map_err(Error::Cuda)?;

        {
            let (spsi_ptr, _) = spsi_dev.device_ptr(stream);
            let (residual_mut, _) = residual_dev.device_ptr_mut(stream);

            for (b, lambda) in eigenvalues.iter().enumerate() {
                let spsi_b = (spsi_ptr as *const CudaComplex).add(b * n_pw);
                let r_b = (residual_mut as *mut CudaComplex).add(b * n_pw);
                let alpha = CudaComplex {
                    x: -lambda,
                    y: 0.0,
                };
                cublasZaxpy_v2(
                    handle,
                    n_pw_i32,
                    &alpha as *const _ as *const _,
                    spsi_b as *const _,
                    1,
                    r_b as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }

        // ---------------------------------------------------------------
        // Step 5: S⁻¹-weighted residual norm
        //   sinv_r = S⁻¹·r   (batch, in-place on sinv_r_dev)
        //   ‖r‖_{S⁻¹} = √Re⟨r_b | sinv_r_b⟩
        // ---------------------------------------------------------------
        // Copy residual → sinv_r_dev (apply_s_inverse works in-place)
        stream
            .memcpy_dtod(&residual_dev, &mut sinv_r_dev)
            .map_err(Error::Cuda)?;

        unsafe {
            apply_s_inverse(
                &mut sinv_r_dev,
                vnl_data,
                n_bands_i32,
                n_pw_i32,
                blas,
                stream,
                solver,
            )?;
        }

        // Per-band: norm = √Re⟨r_b | (S⁻¹·r)_b⟩
        {
            let (residual_ptr, _) = residual_dev.device_ptr(stream);
            let (sinv_r_ptr, _) = sinv_r_dev.device_ptr(stream);

            for (b, rn) in residual_norms_sinv.iter_mut().enumerate() {
                let r_b = (residual_ptr as *const CudaComplex).add(b * n_pw);
                let sinv_b = (sinv_r_ptr as *const CudaComplex).add(b * n_pw);

                let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                cublasZdotc_v2(
                    handle,
                    n_pw_i32,
                    r_b as *const _,
                    1,
                    sinv_b as *const _,
                    1,
                    &mut dot as *mut _ as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;

                *rn = dot.x.sqrt();
            }
        }

        // ---------------------------------------------------------------
        // Step 6: Per-band locking
        //   Lock if:
        //     ‖r_b‖_{S⁻¹} < lock_tol  AND
        //     |λ_b − prev_λ_b| < lock_tol  (skip on first outer iteration when
        //                                    prev_lambdas is None)
        // ---------------------------------------------------------------
        let mut eigenvalue_deltas = vec![0.0_f64; n_bands];
        for b in 0..n_bands {
            let norm_ok = residual_norms_sinv[b] < lock_tol;
            let delta = (eigenvalues[b] - prev_outer_lambdas[b]).abs();
            eigenvalue_deltas[b] = delta;
            let delta_ok = skip_delta_check || delta < lock_tol;
            locked[b] = norm_ok && delta_ok;
        }
        // After first outer iteration, always check deltas
        skip_delta_check = false;

        // Build locked / unconverged index lists
        let mut unconv_idx: Vec<usize> = Vec::new();
        for (b, &is_locked) in locked.iter().enumerate() {
            if !is_locked {
                unconv_idx.push(b);
            }
        }
        let n_locked = locked.iter().filter(|&&l| l).count();
        let n_unconv = unconv_idx.len();
        let locked_indices: Vec<usize> = locked
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l)
            .map(|(i, _)| i)
            .collect();

        // ---------------------------------------------------------------
        // Step 7: Early exit if all bands locked
        // ---------------------------------------------------------------
        if n_unconv == 0 {
            let mut psi_out: CudaSlice<CudaComplex> =
                stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
            stream
                .memcpy_dtod(&psi_dev, &mut psi_out)
                .map_err(Error::Cuda)?;

            let max_res = residual_norms_sinv
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max);
            *DAVIDSON_LAST_DIAG.lock().unwrap() = Some(DavidsonDiagnostic {
                n_locked,
                n_unconverged: n_unconv,
                n_davidson_iters: outer_iter + 1,
                locked_indices: locked_indices.clone(),
                unconv_indices: unconv_idx.clone(),
                residual_norms_sinv: residual_norms_sinv.clone(),
                max_residual_sinv: max_res,
                lock_tol,
                eigenvalue_deltas,
                blocks: vec![],
                n_restarts,
            });

            return Ok(DavidsonResult {
                psi_out,
                eigenvalues,
                n_outer_iters: outer_iter + 1,
                n_locked,
                residual_norms_sinv,
            });
        }

        // ---------------------------------------------------------------
        // Step 8: Block-partition unconverged eigenvalues
        // ---------------------------------------------------------------
        let unconv_eigenvalues: Vec<f64> =
            unconv_idx.iter().map(|&b| eigenvalues[b]).collect();
        let blocks = detect_degenerate_blocks(&unconv_eigenvalues, cfg.block_eps_degen);
        let n_blocks = if blocks.is_empty() {
            // Treat all unconverged as one block
            1
        } else {
            blocks.len()
        };

        // Pre-allocate per-block temporaries reused in the block loop
        let k = n_unconv;

        // Gather unconverged columns (ψ and Hψ) into contiguous buffers
        let mut psi_unconv_dev: CudaSlice<CudaComplex> =
            stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?;
        let mut hpsi_unconv_dev: CudaSlice<CudaComplex> =
            stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?;

        {
            let (psi_ptr, _) = psi_dev.device_ptr(stream);
            let (hpsi_ptr, _) = hpsi_dev.device_ptr(stream);
            let (psi_unconv_mut, _) = psi_unconv_dev.device_ptr_mut(stream);
            let (hpsi_unconv_mut, _) = hpsi_unconv_dev.device_ptr_mut(stream);

            for (u, &b) in unconv_idx.iter().enumerate() {
                cublasZcopy_v2(
                    handle,
                    n_pw_i32,
                    (psi_ptr as *const CudaComplex).add(b * n_pw) as *const _,
                    1,
                    (psi_unconv_mut as *mut CudaComplex).add(u * n_pw) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;

                cublasZcopy_v2(
                    handle,
                    n_pw_i32,
                    (hpsi_ptr as *const CudaComplex).add(b * n_pw) as *const _,
                    1,
                    (hpsi_unconv_mut as *mut CudaComplex).add(u * n_pw) as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }

        // Scatter ZHEGVD eigenvalues back into the main array (per block)
        let mut eigenvalues_k: Vec<f64> = vec![0.0; k];

        // ---------------------------------------------------------------
        // Step 9: Per-block ZHEGVD
        // ---------------------------------------------------------------
        if n_blocks <= 1 {
            // Single block: full k×k problem
            let mut psi_unconv_new: CudaSlice<CudaComplex> =
                stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?;
            solve_block_zhegvd(
                &psi_unconv_dev,
                &hpsi_unconv_dev,
                vnl_data,
                k,
                n_pw,
                blas,
                solver,
                stream,
                &mut eigenvalues_k,
                &mut eig_dev,
                &mut info_dev,
                &mut psi_unconv_new,
            )?;
            psi_unconv_dev = psi_unconv_new;
        } else {
            // Multiple blocks: solve each independently
            let mut psi_rotated: CudaSlice<CudaComplex> =
                stream.alloc_zeros(n_pw * k).map_err(Error::Cuda)?;

            for &(block_lo, block_hi) in blocks.iter() {
                let bk = block_hi - block_lo;
                if bk < 1 {
                    continue;
                }

                // Extract block columns from psi_unconv_dev and hpsi_unconv_dev
                let mut psi_block: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_pw * bk).map_err(Error::Cuda)?;
                let mut hpsi_block: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_pw * bk).map_err(Error::Cuda)?;

                {
                    let (psi_unconv_ptr, _) = psi_unconv_dev.device_ptr(stream);
                    let (hpsi_unconv_ptr, _) = hpsi_unconv_dev.device_ptr(stream);
                    let (psi_block_mut, _) = psi_block.device_ptr_mut(stream);
                    let (hpsi_block_mut, _) = hpsi_block.device_ptr_mut(stream);

                    for i in 0..bk {
                        let src_col = (psi_unconv_ptr as *const CudaComplex)
                            .add((block_lo + i) * n_pw);
                        let hsrc_col = (hpsi_unconv_ptr as *const CudaComplex)
                            .add((block_lo + i) * n_pw);
                        let dst = (psi_block_mut as *mut CudaComplex).add(i * n_pw);
                        let hdst = (hpsi_block_mut as *mut CudaComplex).add(i * n_pw);

                        cublasZcopy_v2(
                            handle,
                            n_pw_i32,
                            src_col as *const _,
                            1,
                            dst as *mut _,
                            1,
                        )
                        .result()
                        .map_err(Error::Blas)?;

                        cublasZcopy_v2(
                            handle,
                            n_pw_i32,
                            hsrc_col as *const _,
                            1,
                            hdst as *mut _,
                            1,
                        )
                        .result()
                        .map_err(Error::Blas)?;
                    }
                }

                // Solve block — use separate result buffer
                let mut block_result: CudaSlice<CudaComplex> =
                    stream.alloc_zeros(n_pw * bk).map_err(Error::Cuda)?;
                let mut block_eig = vec![0.0_f64; bk];
                solve_block_zhegvd(
                    &psi_block,
                    &hpsi_block,
                    vnl_data,
                    bk,
                    n_pw,
                    blas,
                    solver,
                    stream,
                    &mut block_eig,
                    &mut eig_dev,
                    &mut info_dev,
                    &mut block_result,
                )?;

                // Copy block eigenvalues back
                for (i, &val) in block_eig.iter().enumerate() {
                    eigenvalues_k[block_lo + i] = val;
                }

                // Copy rotated block back to psi_rotated
                {
                    let (psi_rot_mut, _) = psi_rotated.device_ptr_mut(stream);
                    let (block_result_ptr, _) = block_result.device_ptr(stream);
                    for i in 0..bk {
                        let src = (block_result_ptr as *const CudaComplex).add(i * n_pw);
                        let dst = (psi_rot_mut as *mut CudaComplex)
                            .add((block_lo + i) * n_pw);
                        cublasZcopy_v2(
                            handle,
                            n_pw_i32,
                            src as *const _,
                            1,
                            dst as *mut _,
                            1,
                        )
                        .result()
                        .map_err(Error::Blas)?;
                    }
                }
            }

            // Replace psi_unconv_dev with the rotated result
            psi_unconv_dev = psi_rotated;
        }

        // Update eigenvalues for unconverged bands
        for (pos, &b) in unconv_idx.iter().enumerate() {
            eigenvalues[b] = eigenvalues_k[pos];
        }

        // Scatter rotated columns back into the main psi_dev buffer
        {
            let (psi_mut, _) = psi_dev.device_ptr_mut(stream);
            let (psi_unconv_ptr, _) = psi_unconv_dev.device_ptr(stream);
            for (u, &b) in unconv_idx.iter().enumerate() {
                let src =
                    (psi_unconv_ptr as *const CudaComplex).add(u * n_pw);
                let dst = (psi_mut as *mut CudaComplex).add(b * n_pw);
                cublasZcopy_v2(
                    handle,
                    n_pw_i32,
                    src as *const _,
                    1,
                    dst as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;
            }
        }

        // ---------------------------------------------------------------
        // Steps 10–12: Preconditioned correction, S-orthogonalization,
        //              and subspace management
        // ---------------------------------------------------------------

        // Reserve space for new search directions collected this iteration
        let n_new_directions = n_unconv;

        // Check whether restart is needed BEFORE adding new directions
        let restart_needed =
            subspace_dim + n_new_directions > max_subspace;

        if restart_needed {
            // Collapse subspace: keep locked bands + top converged bands
            let keep_unconv = std::cmp::min(n_unconv, n_bands.saturating_sub(n_locked));

            // Build new column ordering
            let mut new_col_order: Vec<usize> = Vec::with_capacity(n_bands);
            // Locked bands keep their positions
            for (b, &is_locked) in locked.iter().enumerate() {
                if is_locked {
                    new_col_order.push(b);
                }
            }
            // Best unconverged bands (by eigenvalue order, already sorted)
            for &idx in unconv_idx.iter().take(keep_unconv) {
                new_col_order.push(idx);
            }
            // If there are fewer, fill the rest with the first locked band
            // (should not happen, but defensive):
            while new_col_order.len() < n_bands {
                new_col_order.push(new_col_order[0]);
            }

            // Build new psi buffer, eigenvalues, locked flags
            let mut new_psi: CudaSlice<CudaComplex> =
                stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
            let mut new_eigenvalues = vec![0.0_f64; n_bands];
            let mut new_locked = vec![false; n_bands];

            {
                let (psi_ptr, _) = psi_dev.device_ptr(stream);
                let (new_psi_mut, _) = new_psi.device_ptr_mut(stream);

                for (new_b, &old_b) in new_col_order.iter().enumerate() {
                    let src = (psi_ptr as *const CudaComplex).add(old_b * n_pw);
                    let dst = (new_psi_mut as *mut CudaComplex).add(new_b * n_pw);
                    cublasZcopy_v2(
                        handle,
                        n_pw_i32,
                        src as *const _,
                        1,
                        dst as *mut _,
                        1,
                    )
                    .result()
                    .map_err(Error::Blas)?;

                    new_eigenvalues[new_b] = eigenvalues[old_b];
                    new_locked[new_b] = locked[old_b];
                }
            }

            psi_dev = new_psi;
            eigenvalues = new_eigenvalues;
            locked = new_locked;

            // Rebuild unconv_idx for the collapsed basis
            unconv_idx.clear();
            for (b, &is_locked) in locked.iter().enumerate() {
                if !is_locked {
                    unconv_idx.push(b);
                }
            }

            // Re-initialise subspace from collapsed psi_dev
            stream
                .memcpy_dtod(&psi_dev, &mut subspace_dev)
                .map_err(Error::Cuda)?;
            stream
                .memcpy_dtod(&subspace_dev, &mut sspace_dev)
                .map_err(Error::Cuda)?;
            unsafe {
                apply_s_times(
                    &subspace_dev,
                    &mut sspace_dev,
                    vnl_data,
                    n_bands_i32,
                    n_pw_i32,
                    blas,
                    stream,
                )?;
            }
            subspace_dim = n_bands;

            // Increase restart counter
            n_restarts += 1;

            // No new directions are added after a restart — the fresh subspace
            // is the collapsed psi_dev itself. Continue to next outer iteration.
            continue;
        }

        // ==============================================================
        // No restart: compute corrections and expand subspace
        // ==============================================================

        // Grow subspace buffers if needed for the new directions
        let needed_cap = subspace_dim + n_new_directions;
        if needed_cap > subspace_cap {
            let new_cap = (needed_cap * 2).min(max_subspace);
            subspace_cap = unsafe {
                ensure_subspace_cap(
                    &mut subspace_dev,
                    &mut sspace_dev,
                    subspace_dim,
                    new_cap,
                    n_pw,
                    stream,
                )?
            };
        }

        // ---- Step 10: Preconditioned correction t_b = P⁻¹·r_b ----
        for &b in &unconv_idx {
            let lambda = eigenvalues[b];

            // Copy residual_b → t_buf
            let (residual_ptr, _) = residual_dev.device_ptr(stream);
            let r_b = (residual_ptr as *const CudaComplex).add(b * n_pw);
            let (t_mut, _) = t_buf.device_ptr_mut(stream);
            cublasZcopy_v2(
                handle,
                n_pw_i32,
                r_b as *const _,
                1,
                t_mut as *mut _,
                1,
            )
            .result()
            .map_err(Error::Blas)?;

            // Apply TPA preconditioner: precond_buf = P⁻¹·t_buf
            // (t_buf holds the residual, precond_buf gets the correction)
            unsafe {
                cfg.preconditioner.apply(
                    &mut precond_buf,
                    &t_buf,
                    kinetic_dev,
                    lambda,
                    n_pw,
                    stream,
                )?;
            }

            // ---- Step 11: S-orthogonalize precond_buf against subspace ----
            {
                let (subspace_ptr, _) = subspace_dev.device_ptr(stream);
                let (sspace_ptr, _) = sspace_dev.device_ptr(stream);
                let (precond_ptr, _) = precond_buf.device_ptr(stream);
                let (precond_mut, _) = precond_buf.device_ptr_mut(stream);

                for j in 0..subspace_dim {
                    let s_v_j =
                        (sspace_ptr as *const CudaComplex).add(j * n_pw);
                    let v_j =
                        (subspace_ptr as *const CudaComplex).add(j * n_pw);

                    let mut dot = CudaComplex { x: 0.0, y: 0.0 };
                    cublasZdotc_v2(
                        handle,
                        n_pw_i32,
                        s_v_j as *const _,
                        1,
                        precond_ptr as *const _,
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
                        v_j as *const _,
                        1,
                        precond_mut as *mut _,
                        1,
                    )
                    .result()
                    .map_err(Error::Blas)?;
                }

                // Normalise precond_buf
                let mut norm: f64 = 0.0;
                cublasDznrm2_v2(
                    handle,
                    n_pw_i32,
                    precond_ptr as *const _,
                    1,
                    &mut norm as *mut _,
                )
                .result()
                .map_err(Error::Blas)?;

                if norm > 1e-30 {
                    let inv_norm = 1.0 / norm;
                    cublasZdscal_v2(
                        handle,
                        n_pw_i32,
                        &inv_norm as *const _,
                        precond_mut as *mut _,
                        1,
                    )
                    .result()
                    .map_err(Error::Blas)?;
                }
            }

            // ---- Step 12: Append corrected direction to subspace ----
            // Compute S·precond_buf for the new direction
            stream
                .memcpy_dtod(&precond_buf, &mut s_temp_dev)
                .map_err(Error::Cuda)?;
            unsafe {
                apply_s_times(
                    &precond_buf,
                    &mut s_temp_dev,
                    vnl_data,
                    1, // single band
                    n_pw_i32,
                    blas,
                    stream,
                )?;
            }

            // Append to subspace_dev and sspace_dev
            {
                let (subspace_mut, _) = subspace_dev.device_ptr_mut(stream);
                let (sspace_mut, _) = sspace_dev.device_ptr_mut(stream);
                let (precond_ptr, _) = precond_buf.device_ptr(stream);
                let (s_temp_ptr, _) = s_temp_dev.device_ptr(stream);

                let sub_col =
                    (subspace_mut as *mut CudaComplex).add(subspace_dim * n_pw);
                let s_col =
                    (sspace_mut as *mut CudaComplex).add(subspace_dim * n_pw);

                cublasZcopy_v2(
                    handle,
                    n_pw_i32,
                    precond_ptr as *const _,
                    1,
                    sub_col as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;

                cublasZcopy_v2(
                    handle,
                    n_pw_i32,
                    s_temp_ptr as *const _,
                    1,
                    s_col as *mut _,
                    1,
                )
                .result()
                .map_err(Error::Blas)?;

                subspace_dim += 1;
            }
        }

        // ---------------------------------------------------------------
        // Update prev_outer_lambdas for next iteration's delta check
        // ---------------------------------------------------------------
        prev_outer_lambdas = eigenvalues.clone();
    }

    // ------------------------------------------------------------------
    // Max outer iterations reached — return best available result
    // ------------------------------------------------------------------
    let mut psi_out: CudaSlice<CudaComplex> =
        stream.alloc_zeros(n_elem).map_err(Error::Cuda)?;
    stream
        .memcpy_dtod(&psi_dev, &mut psi_out)
        .map_err(Error::Cuda)?;

    let n_locked_final = locked.iter().filter(|&&l| l).count();
    let max_res = residual_norms_sinv
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);

    *DAVIDSON_LAST_DIAG.lock().unwrap() = Some(DavidsonDiagnostic {
        n_locked: n_locked_final,
        n_unconverged: locked.iter().filter(|&&l| !l).count(),
        n_davidson_iters: cfg.max_outer_iter,
        locked_indices: locked
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l)
            .map(|(i, _)| i)
            .collect(),
        unconv_indices: locked
            .iter()
            .enumerate()
            .filter(|&(_, &l)| !l)
            .map(|(i, _)| i)
            .collect(),
        residual_norms_sinv: residual_norms_sinv.clone(),
        max_residual_sinv: max_res,
        lock_tol,
        eigenvalue_deltas: vec![],
        blocks: vec![],
        n_restarts,
    });

    Ok(DavidsonResult {
        psi_out,
        eigenvalues,
        n_outer_iters: cfg.max_outer_iter,
        n_locked: n_locked_final,
        residual_norms_sinv,
    })
}

// ======================================================================
// Helper: solve a single block's generalized eigenvalue problem
// ======================================================================

/// Build H_sub, S_sub (USPP-augmented), call ZHEGVD, and rotate ψ.
///
/// `psi_block` — n_pw × k block of wavefunction columns
/// `hpsi_block` — n_pw × k block of H·ψ columns
/// On return, `psi_block` is overwritten with the rotated eigenbasis.
#[allow(clippy::too_many_arguments)]
unsafe fn solve_block_zhegvd(
    psi_block: &CudaSlice<CudaComplex>,
    hpsi_block: &CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    k: usize,
    n_pw: usize,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
    eigenvalues_out: &mut [f64],
    eig_dev: &mut CudaSlice<f64>,
    info_dev: &mut CudaSlice<i32>,
    psi_rotated: &mut CudaSlice<CudaComplex>,
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
    }

    Ok(())
}

// ======================================================================
// CPU unit tests (no GPU required)
// ======================================================================

#[cfg(test)]
mod tests {
    use super::lock_tol_for_iter;

    #[test]
    fn lock_tol_for_iter_baseline() {
        // iter 1: lock_tol = 0.01 (initial_tol)
        let tol_1 = lock_tol_for_iter(1, 1e-6);
        assert!(
            (tol_1 - 0.01).abs() < 1e-15,
            "iter 1 lock_tol = {tol_1}, expected 0.01"
        );

        // iter 2: lock_tol = target + (0.01 - target) * 0.5^1
        let expected_2 = 1e-6 + (0.01 - 1e-6) * 0.5_f64.powi(1);
        let tol_2 = lock_tol_for_iter(2, 1e-6);
        assert!(
            (tol_2 - expected_2).abs() < 1e-15,
            "iter 2 lock_tol = {tol_2}, expected {expected_2}"
        );

        // iter 3: lock_tol = target + (0.01 - target) * 0.5^2
        let expected_3 = 1e-6 + (0.01 - 1e-6) * 0.5_f64.powi(2);
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
        assert!(
            tol_10 < 0.5,
            "iter 10 lock_tol = {tol_10} exceeded initial_tol 0.5"
        );

        // Edge: scf_iter = 0 (should behave like iter-1)
        let tol_0 = lock_tol_for_iter(0, 1e-6);
        assert!(
            (tol_0 - 0.01).abs() < 1e-15,
            "iter 0 lock_tol = {tol_0}, expected 0.01"
        );
    }

    #[test]
    fn lock_tol_for_iter_convergence_asymptotic() {
        // For large iter, lock_tol should approach target
        let tol = lock_tol_for_iter(100, 1e-6);
        let diff = (tol - 1e-6).abs();
        assert!(
            diff < 1e-10,
            "iter 100 lock_tol = {tol} is {diff} from target 1e-6, should be very close"
        );
    }

    #[test]
    fn lock_tol_for_iter_zero_target_tol() {
        // target_tol = 0.0: lock_tol should approach 0
        let tol = lock_tol_for_iter(100, 0.0);
        assert!(
            tol < 1e-10,
            "iter 100 with target 0.0: lock_tol = {tol}, expected near 0"
        );
    }
}
