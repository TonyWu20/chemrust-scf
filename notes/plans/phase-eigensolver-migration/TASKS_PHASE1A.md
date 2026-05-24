# Phase 1A Tasks: Davidson v1 Production Implementation

**Date:** 2026-05-25
**Plan:** `/home/tony/.claude/plans/notes-plans-phase-eigensolver-migration-glistening-sunrise.md`
**Branch:** `feat/phase-1a-davidson-v1` (to be created from `feat/phase-global-woodbury`)
**Estimated duration:** 2 weeks (10-12 working days)

---

## Overview

This TASKS.md decomposes the Phase 1A plan into executable task groups. Each group corresponds to one goal from the plan.

**Dependencies:**
- Group 1 (refactoring) must complete before Groups 2-3
- Group 2 (preconditioner) must complete before Group 3 (Davidson core)
- Group 3 (Davidson core) must complete before Groups 5-6
- Group 5 (enum + ratchet) must complete before Group 6 (cleanup)
- Groups 1-6 must complete before Group 7 (validation)

---

## Group 1: Module Refactoring (prerequisite)

**Goal:** Extract shared primitives from chebyshev.rs monolith before Davidson imports from it.

**Acceptance:** All existing tests pass unmodified. Zero functional change.

### Task 1.1: Create kernels.rs submodule

Extract NVRTC kernel sources and `CudaKernelSet` from `src/eigensolver/chebyshev.rs` to new `src/eigensolver/kernels.rs`.

**What to extract:**
- All CUDA C kernel source strings (currently inline in chebyshev.rs)
- `CudaKernelSet` struct definition
- `CudaKernelSet::new()` implementation (NVRTC compilation)
- Kernel function accessors

**Files:**
- Create: `src/eigensolver/kernels.rs` (~400 lines)
- Modify: `src/eigensolver/chebyshev.rs` (remove extracted code, add `use crate::eigensolver::kernels::*`)
- Modify: `src/eigensolver/mod.rs` (add `pub(crate) mod kernels;`)

**Verification:**
```bash
cargo check --workspace
cargo test --workspace
```

---

### Task 1.2: Create hamiltonian.rs submodule

Extract Hamiltonian/overlap application functions from `src/eigensolver/chebyshev.rs` to new `src/eigensolver/hamiltonian.rs`.

**What to extract:**
- `apply_full_hamiltonian` (currently chebyshev.rs:707)
- `apply_s_times` (currently chebyshev.rs:934)
- `apply_s_inverse` (currently chebyshev.rs:852, **change visibility to `pub(crate) unsafe fn`**)
- Private helper functions these depend on

**Files:**
- Create: `src/eigensolver/hamiltonian.rs` (~300 lines)
- Modify: `src/eigensolver/chebyshev.rs` (remove extracted code, add `use crate::eigensolver::hamiltonian::*`)
- Modify: `src/eigensolver/mod.rs` (add `pub(crate) mod hamiltonian;`)

**Verification:**
```bash
cargo check --workspace
cargo test --workspace
```

---

### Task 1.3: Update imports in existing code

Update all import statements in files that use the extracted functions.

**Files to update:**
- `src/eigensolver/rayleigh_ritz.rs`
- `src/density.rs`
- Any other files that import from chebyshev.rs

**Verification:**
```bash
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

---

## Group 2: TPA Diagonal Preconditioner

**Goal:** Implement Teter-Payne-Allan diagonal preconditioner for Davidson.

**Acceptance:** Preconditioner compiles, unit test passes.

### Task 2.1: Write CUDA kernel for diagonal preconditioner

Create `src/eigensolver/preconditioner.rs` with CUDA C kernel source.

**Kernel signature:**
```cuda
__global__ void apply_diagonal_preconditioner(
    cuDoubleComplex* r_out,
    const cuDoubleComplex* r_in,
    const double* kinetic,
    double eigenvalue,
    double shift,
    int n_pw
)
```

**Implementation:**
- Per-thread: `denom = kinetic[ig] - eigenvalue + shift`
- Safety clamp: `if (fabs(denom) < 1e-10) denom = 1e-10`
- Apply: `r_out[ig] = r_in[ig] / denom`

**Files:**
- Create: `src/eigensolver/preconditioner.rs` (~15 lines kernel source)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 2.2: Write Rust wrapper for preconditioner

Add Rust wrapper function in `src/eigensolver/preconditioner.rs`.

**Function signature:**
```rust
pub(crate) unsafe fn apply_tpa_preconditioner(
    r_out: &mut Gpu<WavefunctionSet<ColumnDistributed>>,
    r_in: &Gpu<WavefunctionSet<ColumnDistributed>>,
    kinetic: &[f64],
    eigenvalue: f64,
    shift: f64,
    cuda_ctx: &CudaContext,
    kernel_set: &CudaKernelSet,
) -> Result<()>
```

**Files:**
- Modify: `src/eigensolver/preconditioner.rs` (~50 lines wrapper)
- Modify: `src/eigensolver/mod.rs` (add `pub(crate) mod preconditioner;`)

**Verification:**
```bash
cargo check --workspace
cargo clippy --workspace -- -D warnings
```

---

### Task 2.3: Unit test for preconditioner

Write unit test verifying preconditioner correctness.

**Test strategy:**
- Create synthetic residual vector
- Apply preconditioner with known kinetic energies and eigenvalue
- Verify output matches formula: `r_out[g] = r_in[g] / (kinetic[g] - eigenvalue + shift)`

**Files:**
- Modify: `src/eigensolver/preconditioner.rs` (add `#[cfg(test)] mod tests`)

**Verification:**
```bash
cargo test --workspace preconditioner
```

---

## Group 3: Production Davidson Core

**Goal:** Implement full Davidson algorithm with outer iteration, subspace management, block partitioning.

**Acceptance:** Davidson compiles, basic convergence test passes.

### Task 3.1: Scaffold davidson.rs with outer loop structure

Create `src/eigensolver/davidson.rs` with main function signature and outer loop skeleton.

**Function signature:**
```rust
pub(crate) unsafe fn davidson_v1(
    psi_in: &Gpu<WavefunctionSet<ColumnDistributed>>,
    lock_tol: f64,
    max_iter: usize,
    vnl_data: &VnlBatchData,
    cuda_ctx: &CudaContext,
    kernel_set: &CudaKernelSet,
    // ... other parameters
) -> Result<(Gpu<WavefunctionSet<ColumnDistributed>>, Vec<f64>)>
```

**Outer loop structure:**
```rust
let mut locked_indices = Vec::new();
let mut unconverged_indices: Vec<usize> = (0..n_bands).collect();

for iter in 0..max_iter {
    // 1. Compute Hψ, Sψ for unconverged bands
    // 2. Compute Rayleigh quotients
    // 3. Compute residuals
    // 4. Check locking criterion
    // 5. If all locked, break
    // 6. Apply preconditioner to residuals
    // 7. S-orthogonalize new search directions
    // 8. Build H_sub, S_sub
    // 9. ZHEGVD on unconverged sub-block
    // 10. Rotate ψ
    // 11. Check restart condition
}
```

**Files:**
- Create: `src/eigensolver/davidson.rs` (~100 lines skeleton)
- Modify: `src/eigensolver/mod.rs` (add `pub(crate) mod davidson;`)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 3.2: Implement S^{-1}-weighted residual norm (batched)

Implement batched S^{-1}-weighted residual norm computation.

**Strategy:**
- Gather all k unconverged residual bands into single `n_pw × k` buffer
- Call `apply_s_inverse` once (from hamiltonian.rs)
- Per-band dotc: `norm_s[b] = sqrt(Re⟨r_b | S^{-1} r_b⟩)`

**Implementation:**
```rust
let s_inv_r = apply_s_inverse(&r_unconverged, vnl_data, ...)?;
for (i, &band_idx) in unconverged_indices.iter().enumerate() {
    let norm_s = r_unconverged.column(i).iter()
        .zip(s_inv_r.column(i).iter())
        .map(|(r, s_inv_r)| (r.conj() * s_inv_r).re)
        .sum::<f64>()
        .sqrt();
    residual_norms[band_idx] = norm_s;
}
```

**Files:**
- Modify: `src/eigensolver/davidson.rs` (~80 lines)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 3.3: Implement locking criterion and band tracking

Implement per-band locking logic.

**Locking criterion:**
- `‖r_b‖_{S^{-1}} < lock_tol` AND `|Δλ_b| < lock_tol`

**Implementation:**
```rust
for &band_idx in &unconverged_indices {
    if residual_norms[band_idx] < lock_tol && 
       (eigenvalues[band_idx] - prev_eigenvalues[band_idx]).abs() < lock_tol {
        locked_indices.push(band_idx);
    }
}
unconverged_indices.retain(|&b| !locked_indices.contains(&b));
```

**Files:**
- Modify: `src/eigensolver/davidson.rs` (~50 lines)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 3.4: Implement block partitioning via detect_degenerate_blocks

Add block partitioning for BLAS3 optimization.

**Implementation:**
- Make `detect_degenerate_blocks` in `rayleigh_ritz.rs` `pub(crate)`
- Call it on unconverged eigenvalues with `eps_degen = 0.01` Ha
- Loop over blocks, per-block ZHEGVD

**Files:**
- Modify: `src/eigensolver/rayleigh_ritz.rs` (change visibility of `detect_degenerate_blocks`)
- Modify: `src/eigensolver/davidson.rs` (~100 lines for block loop)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 3.5: Implement subspace restart/collapse

Add subspace management to bound memory usage.

**Restart condition:**
- When `subspace_dim > max_subspace_dim` (default `2 × n_active_bands`)

**Collapse strategy:**
- Keep all locked bands
- Keep top `n_active` unconverged bands by residual norm
- Discard rest
- Re-orthogonalize via Gram-Schmidt

**Files:**
- Modify: `src/eigensolver/davidson.rs` (~80 lines)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 3.6: Wire preconditioner into Davidson loop

Integrate TPA preconditioner from Group 2.

**Implementation:**
- After computing residuals, apply preconditioner per unconverged band
- `apply_tpa_preconditioner(t_b, r_b, kinetic, eigenvalues[b], shift, ...)`
- S-orthogonalize preconditioned residuals against existing subspace

**Files:**
- Modify: `src/eigensolver/davidson.rs` (~60 lines)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 3.7: Basic convergence test for Davidson

Write integration test verifying Davidson converges on synthetic problem.

**Test strategy:**
- Create diagonal H, S = I
- Known eigenvalues/eigenvectors
- Run Davidson, verify convergence to known solution

**Files:**
- Modify: `src/eigensolver/davidson.rs` (add `#[cfg(test)] mod tests`)

**Verification:**
```bash
cargo test --workspace davidson
```

---

## Group 4: (Integrated into Group 3)

S^{-1}-weighted residual norm is implemented as part of Task 3.2.

---

## Group 5: EigensolverMethod Enum + lock_tol Ratchet

**Goal:** Replace env-var dispatch with typed enum, implement lock_tol ratchet schedule.

**Acceptance:** Enum compiles, dispatch works, ratchet schedule implemented.

### Task 5.1: Define EigensolverMethod enum

Create enum in `src/eigensolver/mod.rs`.

**Definition:**
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigensolverMethod {
    Chebyshev,
    Davidson,
}

impl Default for EigensolverMethod {
    fn default() -> Self {
        Self::Chebyshev  // Preserve existing behavior
    }
}
```

**Files:**
- Modify: `src/eigensolver/mod.rs` (~15 lines)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 5.2: Add eigensolver_method and scf_iter fields to ScfIteration

Extend `ScfIteration` struct.

**Fields to add:**
```rust
pub eigensolver_method: EigensolverMethod,
pub scf_iter: usize,
```

**Files:**
- Modify: `src/scf.rs` (struct definition)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 5.3: Implement lock_tol ratchet schedule

Add ratchet schedule function.

**Implementation:**
```rust
fn compute_lock_tol(scf_iter: usize, target_tol: f64) -> f64 {
    let initial_tol = 0.5;  // Above V_NL noise floor
    let decay = 0.5;  // Geometric tightening
    f64::max(target_tol, initial_tol * decay.powi((scf_iter - 1) as i32))
}
```

**Files:**
- Modify: `src/scf.rs` (~20 lines)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 5.4: Wire enum dispatch into diagonalize_inner

Replace env-var dispatch with enum match.

**Implementation:**
```rust
match self.eigensolver_method {
    EigensolverMethod::Davidson => {
        let lock_tol = compute_lock_tol(self.scf_iter, 1e-6);
        davidson_v1(psi_in, lock_tol, max_iter, vnl_data, ...)?
    }
    EigensolverMethod::Chebyshev => {
        // Existing Chebyshev-RR path (unchanged)
        chebyshev_filter(...)?;
        rayleigh_ritz(...)?
    }
}
```

**Files:**
- Modify: `src/scf.rs` (~100 lines in `diagonalize_inner`)

**Verification:**
```bash
cargo check --workspace
cargo test --workspace
```

---

## Group 6: Cleanup Phase 0 Code

**Goal:** Remove Phase 0 proof-of-concept code.

**Acceptance:** Phase 0 code deleted, all tests pass.

### Task 6.1: Delete davidson_minimal.rs

Remove Phase 0 Davidson implementation.

**Files:**
- Delete: `src/eigensolver/davidson_minimal.rs`
- Modify: `src/eigensolver/mod.rs` (remove `mod davidson_minimal;`)
- Modify: `src/lib.rs` (remove re-exports if any)

**Verification:**
```bash
cargo check --workspace
```

---

### Task 6.2: Delete Phase 0 test file

Remove Phase 0 validation tests.

**Files:**
- Delete: `tests/davidson_minimal_validation.rs`

**Verification:**
```bash
cargo test --workspace
```

---

### Task 6.3: Remove env-var infrastructure from scf.rs

Strip `CHEMRUST_EIGENSOLVER` and `CHEMRUST_DAVIDSON_LOCK_TOL` env-var reads.

**Files:**
- Modify: `src/scf.rs` (remove env-var reads)

**Verification:**
```bash
rg "CHEMRUST_EIGENSOLVER" src/
rg "CHEMRUST_DAVIDSON_LOCK_TOL" src/
# Both should return zero hits
cargo check --workspace
cargo test --workspace
```

---

## Group 7: Phase 2 Validation

**Goal:** Verify Phase 1A reaches CASTEP precision.

**Acceptance:** All Phase 2 tests pass + new Davidson tests pass.

### Task 7.1: Reactivate Phase 2 validation tests

Uncomment/enable existing Phase 2 tests.

**Tests to reactivate:**
1. `iter1_drift_from_castep_state_is_bounded` (1 mHa)
2. `scf_converges_to_castep_energy_at_castep_tolerance` (1e-5 eV)
3. `subspace_projector_iter1_vs_castep` (Cu-3d ratio > 0.999)
4. `overlap_iter2_against_castep` (avg > 0.99)
5. `cascade_iter3_diagnostic_tight` (0.1 Ha)

**Files:**
- Modify: `tests/ca_scf_convergence.rs`

**Verification:**
```bash
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 \
  cargo test --release --features scf_diag -- --ignored
```

---

### Task 7.2: Write Davidson-specific validation tests

Add new tests for Davidson behavior.

**Tests:**
1. `test_davidson_v1_self_consistency_pinned_veff` — CASTEP-pinned V_eff → ψ_out ≡ ψ_in bitwise
2. `test_davidson_v1_lock_progression` — SCF iter 1-3: 160/160/151+ bands lock
3. `test_davidson_v1_chebyshev_fallback` — EigensolverMethod::Chebyshev bit-exact with baseline
4. `test_davidson_v1_sinv_norm_consistency` — S^{-1} norm matches CPU reference (1e-10)
5. `test_davidson_v1_max_residual_monotonic` — Max residual decreases each iteration

**Files:**
- Modify: `tests/ca_scf_convergence.rs` (~200 lines new tests)

**Verification:**
```bash
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 \
  cargo test --release --features scf_diag davidson_v1 -- --ignored
```

---

### Task 7.3: Debug validation failures

Iteratively fix any test failures.

**Expected issues:**
- Tolerance tuning (lock_tol ratchet schedule)
- Preconditioner shift parameter
- Subspace restart threshold
- Convergence criterion edge cases

**Strategy:**
- Run failing test with `--nocapture` to see diagnostics
- Adjust parameters
- Re-run
- Document any deviations from CASTEP in test comments

**Verification:**
```bash
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 \
  cargo test --release --features scf_diag -- --ignored
# All tests must pass
```

---

## Acceptance Criteria (Phase 1A Complete)

Phase 1A is done when:

1. ✅ All 7 groups complete
2. ✅ `cargo check --workspace` passes
3. ✅ `cargo clippy --workspace -- -D warnings` passes
4. ✅ `cargo test --workspace` passes (all existing tests)
5. ✅ All Phase 2 validation tests pass with Davidson
6. ✅ All new Davidson-specific tests pass
7. ✅ `rg "CHEMRUST_EIGENSOLVER" src/` returns zero hits
8. ✅ `rg "davidson_minimal" src/` returns zero hits
9. ✅ Chebyshev-RR still works (default enum value)

---

## Notes

- **Group dependencies:** Groups must be completed in order (1 → 2 → 3 → 5 → 6 → 7)
- **Estimated timeline:** 2 weeks (10-12 working days)
- **Branch:** `feat/phase-1a-davidson-v1`
- **Merge target:** `main` (after `feat/phase-global-woodbury` merges)

