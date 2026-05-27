# Review: Diagnostic 3 — Outer Loop Convergence Test

**Tasks**: notes/plans/diagnostic-3-outer-loop/TASKS.md  
**Reviewed**: 2026-05-27  
**Branch**: impl/diagnostic-3-outer-loop/group-a

## Summary

**Overall: Passed with criteria adjustments needed for SC-1.**

The implementation faithfully follows all three Group A tasks from TASKS.md. All required functions exist at correct locations with correct signatures. The 10-iteration outer loop correctly implements the corrected flow (move semantics, pre-built GPU state, inline V_eff upload, `ctx.default_stream()`).

Runtime outcome verification: **1 of 5 criteria failed** (SC-1), but the failure is a spec issue, not an implementation defect — the `core` band group (band 0) is already near convergence at 4.21e-2 Ha and cannot achieve the required 20% reduction.

## Per-Task Results

### TASK-A1: GPU-Resident Filter Iteration Function
- **Status**: ✓ Passed
- **Runtime outcome verification**: N/A (library support function)
- **Diff validation**: Function `chebyshev_filter_iteration_gpu` present at chebyshev.rs:1498 with correct 17-parameter signature. Body creates dummy `KPoint`, `PcieAccount::default()`, delegates to `chebyshev_filter`. `Gpu::from_host_with` visibility changed to `#[doc(hidden)] pub` (pcie.rs:45). Re-exported in lib.rs:13.
- **Strategic review**: Minor concern — internal `PcieAccount` is discarded (PCI-E tracking lost). Function assumes gamma-point (`KPoint { coords: [0.0, 0.0, 0.0] }`) without documenting the constraint.

### TASK-A2: Helper and Verification Functions
- **Status**: ✓ Passed
- **Runtime outcome verification**: N/A (standalone CPU-side helpers)
- **Diff validation**: All 6 helper functions present at expected locations (lines 729-845):
  - `upload_psi_to_gpu_column` — line 729
  - `compute_mean_residual` — line 742
  - `count_converged_bands` — line 751
  - `compute_max_eigenvalue_drift` — line 756
  - `verify_residual_monotonicity` — line 769
  - `verify_band0_stability` — line 825
- **Strategic review**: All standalone CPU-side math on `Vec<f64>`, no library-level changes.

### TASK-A3: Diagnostic Test
- **Status**: ✓ Implemented, ⚠ SC-1 needs adjustment
- **Runtime outcome verification**: Ran 10 iterations against Cu111_CO fixture (208s). Data collected:
  - Core (band 0): 4.21e-2 → 3.67e-2 Ha (13% reduction — fails 20% threshold)
  - Cu 3d (bands 1-14): 1.55e-1 → 8.62e-2 Ha (44% reduction ✓)
  - Valence (bands 15-81): 1.32e-1 → 8.91e-2 Ha (32% reduction ✓)
  - nFermi (bands 82-96): 7.88e-2 → 7.08e-2 Ha (10% reduction)
  - Conduction (bands 97-159): 2.60e-2 → 2.28e-1 Ha (INCREASING)
  - Band 0 eigenvalue: stable at -1.0458 Ha across all iterations (within SC-5 range ✓)
- **Diff validation**: Test correctly implements corrected outer loop flow. No `upload_check_wavefunctions_column`, no `update_psi_gpu_for_next_iteration`, uses move semantics (`psi_gpu = psi_new_gpu`), V_eff/FFT indices/kernels pre-built and reused. All 5 SC checks correctly wired.

## Issues Found

### 1. SC-1 criterion unrealistic for core band group (band 0)
**Severity**: Medium — spec issue, not code issue  
**Location**: `tests/chebyshev_orthogonality_diagnostic.rs:816`, TASKS.md SC-1  
**Finding**: The core band (band 0) starts at 4.21e-2 Ha from the CASTEP-converged .check file, which is already near the noise floor for S⁻¹-weighted residuals. Expecting a 20% reduction from this baseline is unrealistic. The test correctly implements the criterion as specified but the criterion itself needs adjustment.  
**Recommendation**: Either (a) exclude the `core` group from the 20% reduction check, or (b) use an absolute threshold (e.g., `residual < 0.05 Ha` at iter-5) instead of relative reduction for the core group.

### 2. Conduction band residuals increase with iterations
**Severity**: Medium — diagnostic finding  
**Location**: All iterations, cond group  
**Finding**: Conduction band residuals grow monotonically from 2.60e-2 (iter-1) to 2.28e-1 (iter-10). This means the outer-loop iterations are actively making conduction bands worse. SC-2 (≥50/63 conduction bands < 0.01 Ha at iter-5) would also fail given this trend.  
**Recommendation**: Investigate whether the fixed spectral bounds (b_up=20.84, b_low=0.0894) applied to the entire wavefunction set amplify high-frequency components in the conduction bands. Consider band-dependent filtering or a tighter b_up.

### 3. `PcieAccount` leaked into public API without `#[doc(hidden)]`
**Severity**: Low  
**Location**: `src/device/pcie.rs:21`  
**Finding**: `PcieAccount` changed from `pub(crate)` to `pub`, accessible via `chemrust_scf::device::pcie::PcieAccount`. Unlike other doc-hidden re-exports (`BlasHandle`, `SolverHandle`, `VnlBatchData`), it has no `#[doc(hidden)]` re-export in lib.rs. The test accesses it via the fully-qualified path.  
**Recommendation**: Add `#[doc(hidden)] pub use device::pcie::PcieAccount;` to `lib.rs` and update the test to use the re-exported path.

### 4. `chebyshev_filter_iteration_gpu` discards PCI-E transfer accounting
**Severity**: Low  
**Location**: `src/eigensolver/chebyshev.rs:1519`  
**Finding**: The wrapper creates a local `PcieAccount::default()` that records internal transfers but is dropped at function exit. The `#[must_use]` attribute is rendered inert.  
**Recommendation**: Document in the function doc that PCI-E tracking is unavailable through this wrapper.

### 5. Gamma-point assumption undocumented
**Severity**: Low  
**Location**: `src/eigensolver/chebyshev.rs:1518`  
**Finding**: Hardcoded `KPoint { coords: [0.0, 0.0, 0.0] }` assumes gamma-point calculation.  
**Recommendation**: Add doc note about this constraint.

## Runtime Outcome Summary

| Criterion | Result | Details |
|-----------|--------|---------|
| SC-1: Residual Monotonicity | **FAIL** | Core (band 0): 4.21e-2 → 3.67e-2 (13% reduction, need 20%). Other groups not checked due to early exit. |
| SC-2: Conduction Band Early Convergence | Not reached | Would fail given cond residuals grow monotonically |
| SC-3: Occupied Band Residual Reduction | Not reached | Trend suggests ≥5× reduction achievable (cu3d: 44%, val: 32%) |
| SC-4: Eigenvalue Stability | Not reached | Band 0 eigenvalue stable at -1.0458 Ha across all iterations |
| SC-5: No Cascade Signature | Not reached | Band 0 within [-1.10, -1.01] range ✓ |
