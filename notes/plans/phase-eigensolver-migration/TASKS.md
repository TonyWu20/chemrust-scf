# TASKS — Phase 1A: Production Davidson v1 Eigensolver

**Phase:** Phase 1A of `PHASE_PLAN.md` (Davidson v1)
**Plan:** `/home/tony/.claude/plans/notes-plans-phase-eigensolver-migration-parallel-feigenbaum.md`
**Status:** READY for implementation
**Date:** 2026-05-25
**Predecessor:** Phase 0 Gate 3'' (PASSED — cascade reduced 185×, branch `feat/phase-global-woodbury`)
**Decisions:** `DECISIONS.md` (Phase 0) + plan cross-reconciliation (feigenbaum + glistening-sunrise)

## Architecture overview

Target module structure after Phase 1A:

```
src/eigensolver/
  mod.rs              — submodule declarations
  kernels.rs          — NVRTC kernel sources + CudaKernelSet (extracted from chebyshev.rs)        [NEW]
  hamiltonian.rs      — apply_full_hamiltonian, apply_s_times, apply_s_inverse (extracted)        [NEW]
  chebyshev.rs        — shrunk: spectral bounds, Chebyshev filter driver, transpose
  davidson.rs         — production Davidson v1 (~700 LOC)                                        [NEW]
  preconditioner.rs   — TPA diagonal preconditioner (~70 LOC)                                    [NEW]
  rayleigh_ritz.rs    — preserved: H_sub/S_sub assembly, detect_degenerate_blocks → pub(crate)
  vnl_data.rs         — preserved: VnlBatchData
  d_screening.rs      — preserved (deferred)
```

Import dependency graph:
```
davidson.rs → kernels.rs + hamiltonian.rs + vnl_data.rs + rayleigh_ritz.rs + preconditioner.rs
chebyshev.rs → kernels.rs + hamiltonian.rs + vnl_data.rs + rayleigh_ritz.rs
```

No circular imports. `preconditioner.rs` depends only on `kernels.rs` (CudaContext, CudaFunction).

## Declared fixtures

| Fixture | Path | Purpose |
|---------|------|---------|
| CASTEP wavefunctions | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check` | Reference ψ for self-consistency tests; loaded via `tests/fixtures/cu111_co.rs::load_fixture` |
| CASTEP V_eff | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.pot_fmt` | Pinned-V_eff for trivial self-consistency check |
| CASTEP density | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.castep_bin` | Cell geometry + density; consumed by `build_scf_state` |
| Pseudopotentials | `/export/Potentials/` | Cu_OTF.usp, C_OTF.usp, O_OTF.usp via `CASTEP_POTENTIAL_DIR` env override |

**Path authority:** The loader at `tests/fixtures/cu111_co.rs:18` hardcodes the F8 path. Loader is authoritative; any stale path strings in other documents are documentation bugs.

## Success criteria (Phase 1A acceptance)

1. `cargo check --workspace` and `cargo clippy --workspace -- -D warnings` pass after every group
2. All existing CPU tests (`cargo test --workspace`) pass unmodified — Chebyshev default preserved
3. `CHEMRUST_EIGENSOLVER=davidson` converges to within 1e-3 eV of CASTEP reference in perturbation recovery SCF
4. `rg "davidson_minimal" src/` returns zero hits (Phase 0 code fully removed)
5. `rg "CHEMRUST_EIGENSOLVER" src/scf.rs` still present but calls `davidson_v1`, not `davidson_minimal_single_sweep`

---

## Group A: Split chebyshev.rs into shared primitives (prerequisite refactor)

**Goal:** Extract `kernels.rs` and `hamiltonian.rs` from the 2002-line `chebyshev.rs` monolith. Zero functional change — all existing tests pass unmodified. `davidson.rs` will import clean primitives from these new modules rather than from a moving target.

**Dependencies:** None.

### Task A1: Create `src/eigensolver/kernels.rs`

Extract from `chebyshev.rs`:
- `CUDA_KERNEL_SRC` const (the NVRTC kernel source string, currently ~160 lines inline)
- `CudaKernelSet` struct definition + all its fields
- `CudaKernelSet::new()` constructor — compiles all NVRTC kernels, returns the set
- Any kernel-launch helper methods that live on `CudaKernelSet`

The `CudaKernelSet` struct currently owns NVRTC-compiled kernel functions (`CudaFunction` handles) for: transpose, band_scale_axpy, accumulate_density, scatter, zero_buffer_real, and any others defined in the CUDA source string. All of these must move to `kernels.rs` together with their compilation logic.

**Verify:** `cargo check --workspace` passes (after A1+A2+A3 all done together — intermediate check after A3).

### Task A2: Create `src/eigensolver/hamiltonian.rs`

Extract from `chebyshev.rs`:
- `c2c_inverse_inplace` and `c2c_forward_inplace` (private, ~15 lines each) — C2C FFT wrappers used by `apply_v_loc_hamiltonian`
- `apply_v_loc_hamiltonian` (private, ~85 lines) — T + V_loc via FFT round-trip
- `apply_v_nl_hamiltonian` (private, ~85 lines) — V_NL via cuBLAS gemm with β-projectors
- `apply_full_hamiltonian` (`pub(crate)`, ~35 lines) — composes V_loc + V_NL
- `apply_s_times` (`pub(crate)`, ~80 lines) — S·ψ = ψ + β·Q·β^H·ψ (USPP overlap)
- `apply_s_inverse` (`pub(crate)`, ~70 lines) — S⁻¹ via Woodbury (remove `#[allow(dead_code)]`)
- `check_s_inv_s_identity` (`pub`, ~85 lines) — diagnostic: verifies S⁻¹·S ≈ I
- Any private helper functions these depend on

**Important:** `apply_s_inverse` currently has `#[allow(dead_code)]` at line 851. Remove that attribute; the function becomes `pub(crate) unsafe fn` in `hamiltonian.rs` — it is now load-bearing for Davidson's S⁻¹-weighted norm.

Imports needed: `CudaKernelSet` from `super::kernels`, `VnlBatchData` from `super::vnl_data`, `ZgemmConfig`/`BlasHandle`/`op` from `crate::device::blas`, `SolverHandle` from `crate::device::solver`, `BatchedFftPlan3d` from `crate::device::fft`, `CudaComplex` from `crate::device`, `Error` from `crate::types`.

### Task A3: Shrink `src/eigensolver/chebyshev.rs`

Remove all code extracted to `kernels.rs` and `hamiltonian.rs`. Update imports to pull from `super::kernels` and `super::hamiltonian` instead of local definitions.

Retained in `chebyshev.rs`:
- `FilterMode` enum, `SpectralBounds` struct
- `compute_spectral_bounds`, `lanczos_upper_bound`
- `compute_kinetic_energies` (CPU, used by Davidson dispatch too)
- `apply_scaled_hamiltonian_inplace` (dead code, preserved)
- `compute_frobenius_norm`, `check_norm_stability`
- `launch_band_scale_axpy`, `upload_f64_slice`
- Transpose helpers
- `chebyshev_filter` driver (the main entry point for Chebyshev path)
- Test utilities: `apply_h_components_for_test`, `HComponentsForTest`, `apply_s_for_test`
- All `#[cfg(test)]` modules

Expected shrinkage: 2002 → ~1390 lines.

### Task A4: Update `src/eigensolver/mod.rs`

Add submodule declarations:
```rust
pub(crate) mod kernels;
pub(crate) mod hamiltonian;
```

All existing declarations (`chebyshev`, `d_screening`, `davidson_minimal`, `rayleigh_ritz`, `vnl_data`) remain unchanged.

### Task A5: Verify zero-regression

1. `cargo check --workspace` — compiles with zero errors
2. `cargo clippy --workspace -- -D warnings` — no warnings
3. `cargo test --workspace` — all existing CPU tests pass
4. Spot-check: `cargo test --release --features scf_diag -- subspace_projector_iter1_vs_castep -- --ignored --nocapture` (GPU required) — produces same Cu-3d ratio as pre-refactor baseline

### Acceptance criteria (Group A)

- [ ] `src/eigensolver/kernels.rs` exists with `CudaKernelSet` + `CUDA_KERNEL_SRC`
- [ ] `src/eigensolver/hamiltonian.rs` exists with `apply_full_hamiltonian`, `apply_s_times`, `apply_s_inverse` (all `pub(crate)`)
- [ ] `src/eigensolver/chebyshev.rs` is shrunk to ~1400 lines with updated imports
- [ ] `cargo check --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] All existing tests pass unmodified
- [ ] `apply_s_inverse` is `pub(crate)` — no longer `#[allow(dead_code)]`

---

## Group B: TPA diagonal preconditioner

**Goal:** Implement the Teter-Payne-Allan diagonal preconditioner: P⁻¹_b[g] = (kinetic[g] − λ_b)⁻¹ applied elementwise. New `preconditioner.rs` submodule with a CUDA C kernel + Rust wrapper. The preconditioner accelerates the Davidson outer loop when lock_tol tightens toward 1e-6.

**Dependencies:** Group A (needs `CudaContext`, `CudaFunction` from `kernels.rs`).

### Task B1: Create `src/eigensolver/preconditioner.rs`

Contents:

```rust
/// CUDA kernel: Teter-Payne-Allan diagonal preconditioner.
const TPA_PRECOND_KERNEL: &str = r#"
extern "C" __global__ void tpa_precondition(
    cuDoubleComplex* precond,
    const cuDoubleComplex* residual,
    const double* __restrict__ kinetic,
    double lambda,
    double clamp_eps,
    int n_pw
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = g; i < n_pw; i += stride) {
        double denom = kinetic[i] - lambda;
        if (fabs(denom) < clamp_eps) {
            denom = copysign(clamp_eps, denom);
        }
        double inv = 1.0 / denom;
        precond[i].x = residual[i].x * inv;
        precond[i].y = residual[i].y * inv;
    }
}
"#;

pub(crate) struct TpaPreconditioner {
    kernel: CudaFunction,
    clamp_eps: f64,
}

impl TpaPreconditioner {
    pub fn new(ctx: &Arc<CudaContext>, clamp_eps: f64) -> Result<Self, Error> {
        // Compile TPA_PRECOND_KERNEL via NVRTC, extract "tpa_precondition" function
    }

    /// Apply P⁻¹ · residual → precond (may alias residual for in-place).
    pub(crate) unsafe fn apply(
        &self,
        precond: &mut CudaSlice<CudaComplex>,
        residual: &CudaSlice<CudaComplex>,
        kinetic_dev: &CudaSlice<f64>,
        lambda: f64,
        n_pw: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Error> {
        // Launch kernel with grid/block dims matching existing pattern
    }
}
```

Key design points:
- `clamp_eps` default: `1e-12` — prevents division by zero when kinetic[g] ≈ λ
- Kernel launch pattern: use `next_multiple_of(256)` block size + `(n_pw + block - 1) / block` grid, matching existing kernel launch conventions in `chebyshev.rs`
- The kernel is elementwise — grid-stride loop for arbitrary `n_pw`
- `precond` and `residual` may be the same `CudaSlice` (in-place operation is safe)

### Task B2: Register module in `mod.rs`

Add `pub(crate) mod preconditioner;` to `src/eigensolver/mod.rs`.

### Task B3: Unit test

Add `#[cfg(test)] mod tests` in `preconditioner.rs`:
- `test_tpa_clamps_near_zero` — create input where kinetic[g] == lambda for one element, verify output is clamped (not Inf/NaN)
- `test_tpa_identity_far_from_lambda` — when |kinetic − λ| ≫ 0, verify precond ≈ residual / (kinetic − λ) to 1e-10

### Acceptance criteria (Group B)

- [ ] `src/eigensolver/preconditioner.rs` exists with kernel + wrapper
- [ ] `cargo check --workspace` compiles
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] Unit tests pass (CPU-only, no GPU needed — test the logic without kernel launch)

---

## Group C: Build production davidson.rs

**Goal:** The core `davidson_v1` function implementing full outer Davidson iteration with S⁻¹-weighted residual norm, block partitioning, and subspace management. All GPU-resident — only final psi_out + eigenvalues cross PCIe.

**Dependencies:** Groups A and B complete.

### Task C1: Create `src/eigensolver/davidson.rs`

Define the module structure:

```rust
// ---------------------------------------------------------------------------
// Production Davidson v1 eigensolver (Phase 1A)
// ---------------------------------------------------------------------------
//
// Outer iteration: while not all bands locked:
//   1. Hψ = apply_full_hamiltonian(ψ_current)
//   2. Sψ = ψ_current; apply_s_times(ψ_current, Sψ)  // USPP overlap
//   3. Per-band Rayleigh quotient λ_b = Re⟨ψ_b|Hψ_b⟩ / Re⟨ψ_b|Sψ_b⟩
//   4. Residual r_b = Hψ_b − λ_b·Sψ_b
//   5. S⁻¹-weighted norm: sinv_r = S⁻¹·r; norm = √Re⟨r|sinv_r⟩
//   6. Lock bands where norm < lock_tol AND |Δλ_b| < lock_tol
//   7. If all locked: exit
//   8. Block-partition unconverged via detect_degenerate_blocks
//   9. Per-block: H_sub/S_sub (USPP augmented), ZHEGVD, rotate
//  10. Preconditioned correction: t_b = P⁻¹·r_b
//  11. S-orthogonalize corrections against subspace (Gram-Schmidt)
//  12. Append to subspace; restart if dim > max_subspace_dim
```

### Task C2: Type definitions

```rust
/// Configuration for Davidson v1 eigensolver.
pub(crate) struct DavidsonConfig {
    /// Maximum outer Davidson iterations (default 30).
    pub max_outer_iter: usize,
    /// Eigenvalue spacing threshold for block partitioning (default 0.01 Ha).
    pub block_eps_degen: f64,
    /// Multiplier on n_active for subspace restart (default 3.0 → 3× n_active).
    pub max_subspace_dim_factor: f64,
    /// Preconditioner instance (owns compiled kernel).
    pub preconditioner: TpaPreconditioner,
}

impl Default for DavidsonConfig {
    fn default() -> Self {
        Self {
            max_outer_iter: 30,
            block_eps_degen: 0.01,
            max_subspace_dim_factor: 3.0,
            preconditioner: /* needs Arc<CudaContext> — not Default */,
        }
    }
}

/// Result of a Davidson v1 diagonalization.
pub(crate) struct DavidsonResult {
    pub psi_out: CudaSlice<CudaComplex>,
    pub eigenvalues: Vec<f64>,
    pub n_outer_iters: usize,
    pub n_locked: usize,
    pub residual_norms_sinv: Vec<f64>,
}

/// Enhanced diagnostics (replaces Phase 0's DavidsonDiagnostic).
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

pub static DAVIDSON_LAST_DIAG: std::sync::Mutex<Option<DavidsonDiagnostic>> =
    std::sync::Mutex::new(None);
```

### Task C3: Main driver function signature

```rust
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn davidson_v1(
    psi_init: &CudaSlice<CudaComplex>,     // initial guess, n_bands × n_pw col-major
    v_eff_dev: &CudaSlice<f64>,            // V_eff on wave grid
    kinetic_dev: &CudaSlice<f64>,          // kinetic energies per G-vector
    fft_idx_dev: &CudaSlice<i32>,          // PW-to-FFT index map
    vnl_data: &VnlBatchData,               // β-projectors, Q-matrices, D-matrices
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    lock_tol: f64,                         // from ratchet schedule (Group D)
    blas: &BlasHandle,
    solver: &SolverHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
    cfg: &DavidsonConfig,
    prev_lambdas: Option<&[f64]>,          // for |Δλ| locking criterion (None on first call)
) -> Result<DavidsonResult, Error>
```

### Task C4: Outer iteration loop

Implement the `while` loop body (steps 1-12 from the algorithm above):

**Step 1-2 (Hψ, Sψ):** Reuse `apply_full_hamiltonian` and `apply_s_times` from `hamiltonian.rs`. Identical to Phase 0 minimal code at `davidson_minimal.rs:110-146`. Call `apply_s_times` with the pre-copy pattern (memcpy_dtod psi → spsi, then apply_s_times accumulates).

**Step 3 (Rayleigh quotient):** Per-band `cublasZdotc_v2` for ⟨ψ_b|Hψ_b⟩ and ⟨ψ_b|Sψ_b⟩. λ_b = dot_h.x / dot_s.x. Identical to Phase 0 code at `davidson_minimal.rs:151-190`.

**Step 4 (Residual):** r_b = Hψ_b − λ_b·Sψ_b via `cublasZaxpy_v2` with α = −λ_b. Identical to Phase 0 code at `davidson_minimal.rs:195-224`.

**Step 5 (S⁻¹-weighted norm):** NEW — replaces Phase 0's plain L2 norm.
- Gather all k unconverged residual columns into a single `n_pw × k` buffer
- Call `apply_s_inverse(&mut sinv_r_dev, vnl_data, k, n_pw, blas, stream, solver)?` — operates on the whole batch
- Per-band: `cublasZdotc_v2(residual_b, sinv_r_b)` → norm = √Re(dot)
- This is the batched strategy: one gemm triplet per outer iteration, not per band

**Step 6 (Locking):** Band b is locked when:
- `norm_sinv_b < lock_tol` AND
- `|λ_b − prev_λ_b| < lock_tol` (skip on first Davidson iteration if `prev_lambdas` is None)

**Step 7 (Early exit):** If `n_unconverged == 0`, copy psi_current → psi_out, return.

**Step 8 (Block partitioning):** Call `detect_degenerate_blocks(&unconv_eigenvalues, cfg.block_eps_degen)` from `rayleigh_ritz.rs`. This returns `Vec<(usize, usize)>` of (start, end) indices into the unconverged set. If no blocks detected, treat all unconverged as one block.

**Step 9 (Per-block ZHEGVD):** For each block:
- Gather unconverged ψ columns + Hψ columns into contiguous buffers (use `cublasZcopy_v2`, same pattern as `davidson_minimal.rs:311-348`)
- Build H_sub = ψ_block^H · Hψ_block (gemm, ZgemmConfig with transa=C, transb=N)
- Build S_sub = ψ_block^H · ψ_block (bare PW overlap) + USPP augmentation (β·Q·β^H per VnlBatchData entry). Same pattern as `davidson_minimal.rs:350-466` (rayleigh_ritz.rs:181-310 for reference).
- ZHEGVD: `solver.zhegvd(VECTOR, LOWER, k, &mut h_sub, &mut s_sub, &mut eig, &mut info)?`
- Check info == 0, D2H eigenvalues
- Rotate: ψ_new_block = ψ_block · X (gemm, transa=N, transb=N, k×k)

**Step 10 (Preconditioned correction):** For each unconverged band:
- `preconditioner.apply(&mut t_b, &r_b, kinetic_dev, lambda_b, n_pw, stream)?`
- t_b = P⁻¹·r_b — new search direction

**Step 11 (S-orthogonalization):** For each new search direction `t_b`:
- For each existing direction `v_j` in the subspace:
  - Compute `s_vj = v_j`; `apply_s_times(&v_j, &mut s_vj, ...)` (single-band)
  - `dot = ⟨s_vj | t_b⟩` via `cublasZdotc_v2`
  - `t_b -= dot · v_j` via `cublasZaxpy_v2` with α = −dot
- Normalize t_b: `norm = cublasDznrm2(t_b)`; scale by 1/norm

**Step 12 (Subspace management):**
- Track subspace dimension = n_locked + n_unconv_rotated + n_new_directions
- If `subspace_dim > max_subspace_dim_factor * n_bands`: collapse
  - Keep all locked bands (copy verbatim)
  - Keep the top `n_bands` unconverged Ritz vectors (best eigenvalues)
  - Drop accumulated search directions
  - Increment restart counter
  - Continue with fresh subspace seeded by current eigenvectors

### Task C5: Register in mod.rs

Add `pub(crate) mod davidson;` to `src/eigensolver/mod.rs`.

### Task C6: Expose `detect_degenerate_blocks` as `pub(crate)`

In `src/eigensolver/rayleigh_ritz.rs`, change line 117:
```rust
// Before:
fn detect_degenerate_blocks(eigenvalues: &[f64], eps_degen: f64) -> Vec<(usize, usize)> {
// After:
pub(crate) fn detect_degenerate_blocks(eigenvalues: &[f64], eps_degen: f64) -> Vec<(usize, usize)> {
```

Zero other changes to `rayleigh_ritz.rs`. This function is well-tested (exercised by the existing Rayleigh-Ritz path at line 348).

### Task C7: Compilation gate

`cargo check --workspace` must pass. `cargo clippy --workspace -- -D warnings` must pass.

### Acceptance criteria (Group C)

- [ ] `src/eigensolver/davidson.rs` exists with `davidson_v1` function
- [ ] `DavidsonConfig`, `DavidsonResult`, `DavidsonDiagnostic` types defined
- [ ] Outer iteration loop implemented (steps 1-12)
- [ ] S⁻¹-weighted residual norm uses batch `apply_s_inverse` strategy
- [ ] Block partitioning calls `detect_degenerate_blocks` (now `pub(crate)`)
- [ ] Per-block ZHEGVD with USPP H_sub/S_sub augmentation
- [ ] Subspace restart logic (collapse at 3× n_active)
- [ ] `cargo check --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` passes

---

## Group D: Lock ratchet schedule + SCF wiring

**Goal:** Wire the lock_tol ratchet schedule into the SCF state machine, thread `scf_iter` through `ScfIteration`, and connect `davidson_v1` to the existing `CHEMRUST_EIGENSOLVER=davidson` dispatch path.

**Dependencies:** Group C complete.

### Task D1: Add `scf_iter` field to `ScfIteration`

In `src/scf.rs`, add to the `ScfIteration` struct:
```rust
/// Current SCF iteration number (1-indexed). Updated by run_scf before diagonalize.
pub(crate) scf_iter: usize,
```

Default value: `0` (set to `1` by `run_scf` before the first diagonalize call, or in `new`).

Update ALL `into_phase` transition methods and the `new` constructor to thread this field through. Every `ScfIteration` construction site that currently uses struct-literal syntax must include `scf_iter: self.scf_iter` or the default.

### Task D2: Implement lock ratchet schedule

In `src/eigensolver/davidson.rs` (or a helper in `scf.rs`):

```rust
/// Compute Davidson lock tolerance for a given SCF iteration.
///
/// Starts at 0.5 Ha (proven by Phase 0 — all 160 bands lock at iter-1),
/// tightens geometrically toward target_tol.
pub(crate) fn lock_tol_for_iter(scf_iter: usize, target_tol: f64) -> f64 {
    let initial_tol: f64 = 0.5;
    let decay: f64 = 0.5;
    if scf_iter <= 2 {
        initial_tol  // iter-1 and iter-2: loose lock catches ~all bands
    } else {
        let gap = initial_tol - target_tol;
        let tol = target_tol + gap * decay.powi(scf_iter as i32 - 2);
        tol.max(target_tol)
    }
}
```

### Task D3: Wire Davidson dispatch in `diagonalize_inner`

In `src/scf.rs`, locate the `CHEMRUST_EIGENSOLVER=davidson` branch (currently lines 602-702). Replace the `davidson_minimal_single_sweep` call with `davidson_v1`:

1. Compute `lock_tol` from the ratchet schedule:
   ```rust
   let lock_tol = std::env::var("CHEMRUST_DAVIDSON_LOCK_TOL")
       .ok()
       .and_then(|s| s.parse().ok())
       .unwrap_or_else(|| lock_tol_for_iter(self.scf_iter, 1e-6));
   ```
   (Env-var overrides schedule for testing; schedule is production default.)

2. Build `DavidsonConfig`:
   ```rust
   let davidson_cfg = DavidsonConfig {
       max_outer_iter: 30,
       block_eps_degen: 0.01,
       max_subspace_dim_factor: 3.0,
       preconditioner: TpaPreconditioner::new(&ctx, 1e-12)?,
   };
   ```

3. Get `prev_lambdas` from `self.eigenvalues` (if available — `ScfIteration<VEffBuilt>` has `eigenvalues: Vec<f64>`? Check state. If not in VEffBuilt, pass `None`.)

4. Call `davidson_v1(...)` replacing `davidson_minimal_single_sweep(...)`. The parameter list is similar but adds `cfg: &davidson_cfg` and `prev_lambdas`.

5. Wrap result identically: `Gpu<WavefunctionSet<ColumnDistributed>>` from `result.psi_out`, eigenvalues from `result.eigenvalues`, recompute `beta_psi_gpu`, stash diagnostics via `DAVIDSON_LAST_DIAG`.

### Task D4: Update `run_scf` to increment `scf_iter`

In `run_scf` (or wherever the SCF loop lives), add `self.scf_iter += 1;` before the diagonalize call each iteration. (Or: `self.scf_iter = iteration_number` if the loop already tracks iteration count.)

### Task D5: Compilation + no-regression check

1. `cargo check --workspace` compiles
2. `cargo clippy --workspace -- -D warnings` passes
3. Existing Chebyshev-path tests pass unmodified (env-var default is `chebyshev`)
4. Quick manual check: `CHEMRUST_EIGENSOLVER=davidson cargo test --release --features scf_diag -- gate3_prime -- --ignored --nocapture` (uses production davidson_v1, not minimal)

### Acceptance criteria (Group D)

- [ ] `ScfIteration` has `scf_iter: usize` field, threaded through all constructors
- [ ] `lock_tol_for_iter` function implemented with geometric decay
- [ ] `diagonalize_inner` calls `davidson_v1` (not `davidson_minimal_single_sweep`)
- [ ] `CHEMRUST_DAVIDSON_LOCK_TOL` env-var overrides schedule for ad-hoc testing
- [ ] `cargo check --workspace` + `cargo clippy` pass
- [ ] All existing tests pass unchanged

---

## Group E: Remove Phase 0 davidson_minimal.rs

**Goal:** Delete the Phase 0 minimal Davidson implementation and all associated re-exports, static diagnostics, and test files. The production `davidson.rs` is now the canonical Davidson implementation.

**Dependencies:** Group D complete (new dispatch is in place).

### Task E1: Delete `src/eigensolver/davidson_minimal.rs`

Remove the file entirely. (669 lines deleted.)

### Task E2: Remove from `src/eigensolver/mod.rs`

Delete the line:
```rust
pub(crate) mod davidson_minimal;
```

### Task E3: Remove from `src/lib.rs`

Delete the line (currently line 10):
```rust
pub use eigensolver::davidson_minimal::{DAVIDSON_LAST_DIAG, DavidsonDiagnostic};
```

If `DavidsonDiagnostic` is re-exported for test use, tests should now import from `eigensolver::davidson::DavidsonDiagnostic` instead.

### Task E4: Remove from `src/scf.rs`

Delete the imports:
```rust
use crate::eigensolver::davidson_minimal::{davidson_minimal_single_sweep, DavidsonResult};
use crate::eigensolver::davidson_minimal::{DavidsonDiagnostic, DAVIDSON_LAST_DIAG};
```

Replace with:
```rust
use crate::eigensolver::davidson::{davidson_v1, DavidsonResult, DavidsonDiagnostic, DAVIDSON_LAST_DIAG};
```

### Task E5: Remove test file

Delete `tests/davidson_minimal_validation.rs` if it exists (the Phase 0 gate test file). Check:
```bash
fd davidson_minimal tests/
```
and remove any matches.

### Task E6: Verify zero references

```bash
rg "davidson_minimal" src/ tests/   # must return zero hits
rg "CHEMRUST_EIGENSOLVER" src/scf.rs  # should still be present (env-var dispatch kept)
```

### Task E7: Re-verify compilation

1. `cargo check --workspace` compiles
2. `cargo clippy --workspace -- -D warnings` passes
3. All existing tests pass

### Acceptance criteria (Group E)

- [ ] `src/eigensolver/davidson_minimal.rs` deleted
- [ ] `rg "davidson_minimal" src/` returns zero hits
- [ ] `rg "davidson_minimal" tests/` returns zero hits
- [ ] `scf.rs` imports from `eigensolver::davidson`, not `eigensolver::davidson_minimal`
- [ ] `cargo check --workspace` + `cargo clippy` + `cargo test --workspace` all pass

---

## Group F: Validation test suite

**Goal:** Comprehensive Davidson validation tests proving correctness against CASTEP reference and non-regression of the Chebyshev fallback path.

**Dependencies:** Groups A-E complete.

### Task F1: Self-consistency test (pinned V_eff)

**Test:** `test_davidson_v1_self_consistency_pinned_veff` in `tests/davidson_v1_validation.rs`

**Procedure:**
1. Load CASTEP ψ from `Cu111_CO.check` (via `load_fixture`)
2. Load CASTEP V_eff from `Cu111_CO.pot_fmt`
3. Build `ScfIteration` with CASTEP-pinned V_eff (skip `build_v_eff` — pin the CASTEP V_eff directly)
4. Set `CHEMRUST_EIGENSOLVER=davidson`
5. Run diagonalize
6. Assert: `max |psi_out[b] - psi_castep[b]| < 1e-12` for all bands
7. Assert: `n_locked == n_bands` (all bands lock with CASTEP-pinned V_eff)

**Tolerance:** 1e-12 per band (bitwise preservation, matching Phase 0 Group B unit test tolerance).

### Task F2: Lock progression test (our V_eff, SCF iter 1-3)

**Test:** `test_davidson_v1_lock_progression` in `tests/davidson_v1_validation.rs`

**Procedure:**
1. Load CASTEP ψ, build SCF state with our V_eff
2. Run 3 SCF iterations with `CHEMRUST_EIGENSOLVER=davidson`
3. Assert iter-1: `n_locked == 160` (all lock at 0.5 Ha)
4. Assert iter-2: `n_locked == 160`
5. Assert iter-3: `n_locked >= 151` (matching Phase 0 Gate 3'' data)
6. Assert iter-3 band-0 drift < 0.1 Ha (cascade arrested — matching Phase 0 result of 0.059 Ha)

### Task F3: Chebyshev fallback test

**Test:** `test_davidson_v1_chebyshev_fallback` in `tests/davidson_v1_validation.rs`

**Procedure:**
1. Run diagonalize with `CHEMRUST_EIGENSOLVER=chebyshev` (the default)
2. Verify the result is identical to a saved Chebyshev-path baseline
3. This proves the env-var dispatch correctly routes to the Chebyshev path and the Chebyshev path is unmodified

### Task F4: S⁻¹ norm consistency test

**Test:** `test_davidson_v1_sinv_norm_consistency` in `tests/davidson_v1_validation.rs`

**Procedure:**
1. Create a random test vector `v` on GPU (known values)
2. Compute `||v||_{S⁻¹}` via the Davidson code path (GPU — `apply_s_inverse` + dotc)
3. Compute `||v||_{S⁻¹}` via CPU reference: construct S⁻¹ explicitly for a small test system, compute `sqrt(v^H · S⁻¹ · v)`
4. Assert agreement within 1e-10

**Note:** This test may use a small synthetic system (e.g., 4 bands, 100 PW) to make explicit S⁻¹ construction feasible on CPU.

### Task F5: Max residual monotonicity test

**Test:** `test_davidson_v1_max_residual_monotonic` in `tests/davidson_v1_validation.rs`

**Procedure:**
1. Run Davidson with `CHEMRUST_EIGENSOLVER=davidson` on the standard Cu111+CO fixture
2. Record `max_residual_sinv` from each outer Davidson iteration (via `DavidsonDiagnostic`)
3. Assert: max residual decreases (or stays within 1% of previous) each iteration
4. Allow 1% noise increase specifically at restart iterations (subspace collapse may cause temporary increase)

### Task F6: Perturbation recovery SCF test

**Test:** `test_scf_converges_with_davidson` — add to `tests/ca_scf_convergence.rs` (reuse existing perturbation recovery infrastructure)

**Procedure:**
1. Use the existing perturbation recovery test pattern (perturb ψ, run SCF, check convergence)
2. Set `CHEMRUST_EIGENSOLVER=davidson` via env var
3. Run perturbation recovery SCF
4. Assert total energy at convergence is within 1e-3 eV of CASTEP reference
5. This is the **primary acceptance gate** for Phase 1A

### Task F7: Run all tests

```bash
# CPU tests (always pass)
cargo test --workspace

# GPU tests (requires CUDA GPU)
cargo test --release --test davidson_v1_validation -- --ignored --nocapture
cargo test --release --test ca_scf_convergence -- test_scf_converges_with_davidson -- --ignored --nocapture
```

### Acceptance criteria (Group F)

- [ ] All 6 new tests compile
- [ ] F1 (pinned V_eff): max band error < 1e-12
- [ ] F2 (lock progression): 160/160/151+ locks at iter 1/2/3, drift < 0.1 Ha
- [ ] F3 (Chebyshev fallback): bit-exact match with saved baseline
- [ ] F4 (S⁻¹ norm): < 1e-10 vs CPU reference
- [ ] F5 (max residual monotonic): decreases each iteration (1% noise tolerance at restart)
- [ ] F6 (SCF convergence): total energy within 1e-3 eV of CASTEP reference
- [ ] `cargo test --workspace` — all existing tests pass unchanged

---

## Group sequence

```
Group A (split chebyshev)
  └─> Group B (TPA preconditioner)
        └─> Group C (davidson.rs core)
              └─> Group D (lock ratchet + SCF wiring)
                    └─> Group E (remove davidson_minimal.rs)
                          └─> Group F (validation tests)
```

Groups A through F are strictly sequential — each depends on all prior groups.

## Verification summary

Run these commands after each group and after Phase 1A completion:

```bash
# After every group:
cargo check --workspace 2>&1
cargo clippy --workspace -- -D warnings 2>&1
cargo test --workspace 2>&1

# After Group F (Phase 1A complete):
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 \
  cargo test --release --features scf_diag -- --ignored --nocapture 2>&1

# Phase 1A acceptance:
rg "davidson_minimal" src/          # must return zero hits
rg "davidson_minimal" tests/        # must return zero hits
rg "CHEMRUST_EIGENSOLVER" src/scf.rs  # still present, calls davidson_v1
```
