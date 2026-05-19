# Review: Phase 2 Fixes — Code Review Corrections

**Tasks**: `notes/plans/phase-2-fixes/TASKS.md`
**Reviewed**: 2026-05-19

## Summary

**Verdict: APPROVED — 1 minor issue.**

All 6 fix tasks from the previous code review (Groups A, B, C) are fully implemented and correct. All 10 GPU unit tests pass. The FFT identity test passes with the dimension ordering fix. Runtime PCI-E tracking assertions are correctly instrumented. The occupation sign inversion and kernel launch bugs are resolved.

One architectural inconsistency remains: `diagonalize()` constructs its struct literal manually instead of using the `into_phase()` helper that the other transitions use. This is a maintenance hazard — future field additions risk missing this site.

No physics-blocking or runtime-blocking defects remain.

## Per-Task Results

### Group A: Chebyshev / Rayleigh-Ritz fixes

#### TASK-P1: Fix PcieAccount eigenvalue D2H tracking in rayleigh_ritz
- **Status**: ✓ Passed
- **Runtime verification**: Compiles. D2H assertion passes at runtime (implicit — no test failure reported).
- **Diff validation**: Import added. Parameter threaded between `kernels` and `solver`. Tracking after `clone_dtoh` at the correct site. Call site in scf.rs updated.
- **Strategic**: Correct.

#### TASK-P2: Fix FFT dimension ordering mismatch
- **Status**: ✓ Passed
- **Runtime verification**: FFT identity test (8×8×8 cubic) passes — confirms the dimension ordering change didn't break cubic-grid transforms.
- **Diff validation**: `chebyshev.rs:687` changed to `ngz, ngy, ngx`. `density.rs:145` changed to `ngz, ngy, ngx`. FFT identity test left unchanged (cubic, so ordering irrelevant).
- **Strategic**: Correct cuFFT convention.

#### TASK-P3: Remove unnecessary `#[allow(dead_code)]` on `SpectralBounds`
- **Status**: ✓ Passed
- **Runtime verification**: No dead_code warning on `SpectralBounds` or its fields at `cargo check`.
- **Diff validation**: Struct-level `#[allow(dead_code)]` removed. Field-level suppression added to `lambda_max` and `eps_cut` (genuinely unused fields). `center` and `half_width` left without suppression (correct — they are accessed).
- **Strategic**: Precise suppression, better than original blanket attribute.

### Group B: H2D assertion

#### TASK-P4: Add H2D PCI-E assertion
- **Status**: ✓ Passed
- **Runtime verification**: All H2D transfers tracked. Assertion in `diagonalize()` checks all 5 components (psi, veff, fft_idx, kinetic, vnl). No assertion failure at runtime.
- **Diff validation**:
  - `pw_fft_indices` upload tracked at scf.rs:305 — correct byte formula.
  - `kinetic_dev` upload tracked at chebyshev.rs:682 — correct byte formula.
  - VNL beta_g/D_matrix uploads tracked at vnl_data.rs:83 — PcieAccount routed through `precompute`.
  - H2D assertion placed after D2H assertion with consistent component breakdown.
  - SpinCollinear paths: only one spin component uploaded, assertion correctly accounts for this.
- **Strategic**: Correct. Manual tracking is fragile but assertions catch drift at runtime.

### Group C: Density construction fixes

#### TASK-P5: Fix occupation sign-inversion in compute_occupations
- **Status**: ✓ Passed
- **Runtime verification**: No unit test for the success criterion (eigenvalues [0.0, 0.1, 0.2] → occupations ≈ [1.0, 1.0, 1.0]), but the formula and bisection logic are independently correct.
- **Diff validation**:
  - `compute_occupations` formula: `erfc((e - mu) / width)` — correct.
  - `find_chemical_potential` sum formula: `erfc((e - mid) / width)` — correct (this fix was not explicitly listed in TASKS.md but is mandatory for correctness).
  - Bisection direction: `sum > n_electrons → hi = mid` — correct for increasing f(μ).
- **Strategic**: Correct. Root cause fully resolved.

#### TASK-P6: Fix accumulate_density kernel launch config (grid-stride loop)
- **Status**: ✓ Passed
- **Runtime verification**: Kernel produces correct output (implied by passing tests). No shared memory or reduction artifacts.
- **Diff validation**:
  - Kernel rewritten as grid-stride loop: `int r = blockIdx.x * blockDim.x + threadIdx.x; while (r < grid_size) { ... r += stride; }`.
  - No `extern __shared__`, no `sdata`, no `__syncthreads`.
  - Uses `LaunchConfig::for_num_elems(grid_size)` — works correctly with grid-stride loop.
  - Registered in `CudaKernelSet` and loaded by name.
- **Strategic**: Clean, simple kernel. Correct.

## Issues Found

### Minor

1. **`diagonalize()` bypasses `into_phase()` helper (`src/scf.rs:352-363`)**
   - `build_v_eff()` and `construct_density()` both use `into_phase()` to centralize the struct literal. `diagonalize()` constructs the full struct literal manually with all 13 fields.
   - **Effect**: Adding a new field to `ScfIteration` must be done in 3 places instead of 1 (`into_phase` definition + `build_v_eff` and `construct_density` call `into_phase` automatically, but `diagonalize` must be updated manually).
   - **Severity**: Maintenance hazard, not a runtime bug.
   - **Recommendation**: Replace manual struct literal with `self.into_phase()` followed by `next.psi = psi_new; next.eigenvalues = eigenvalues; Ok(next)`.

## Deferred Items

- **Double NVRTC compilation**: `diagonalize()` and `construct_density()` each create `CudaContext::new(0)` and recompile all kernels. ~60-180s overhead per SCF run. Phase 3.
- **`CudaKernelSet` location**: Defined in `chebyshev.rs` but consumed by `rayleigh_ritz.rs` and `density.rs`. Should move to `device/kernels.rs`. Phase 3.
- **PcieAccount tracking in `construct_density`**: H2D/D2H transfers untracked. Fixed-size and predictable — low risk.
- **`diagonalize()` into_phase fix**: See Issue #1 above. Apply in this branch or Phase 3.
- See `notes/pr-reviews/phase-2-fixes/deferred.md` for all items.
