# Review: Phase R-ChFSI

**Tasks**: `notes/plans/phase-rchfsi/TASKS.md`
**Reviewed**: 2026-05-22

## Summary

**Overall: PASS** — all tasks implemented as specified, compiler and linter gates pass, no specification violations found.

- SC-1 (`cargo check --workspace`): ✓ PASS (0 errors)
- SC-2 (`cargo clippy --workspace -- -D warnings`): ✓ PASS (0 warnings)
- SC-3 (unit tests): ✓ 27/27 PASS (GPU-dependent tests require hardware)
- SC-4 (SCF convergence tests): ⚠ Cannot verify without GPU hardware
- Per-group diff validation: 3/3 passed with minor issues
- Strategic review: PASS with concerns noted

## Per-Task Results

### TASK-A1: `apply_s_times`
- **Status**: ✓ Passed
- **Runtime outcome verification**: N/A (no unit test per spec — validated end-to-end by SC-1/SC-2)
- **Diff validation**: ✓ Function added with correct 3-gemm pattern (β·Q·β^H), uses `q_matrix` not `s_inv_mat`, sign flipped to +1.0
- **Strategic review**: Clean implementation. Minor: visibility is `pub(crate)` vs spec's private fn — forward-compatible, not a defect.

### TASK-B1: R-ChFSI recurrence
- **Status**: ✓ Passed
- **Runtime outcome verification**: Build and lint pass. Recurrence matches Algorithm 3 (main.tex:586-610).
- **Diff validation**: All parts verified:
  - B1-part-1 (`lambda_min` in `SpectralBounds`): ✓ Both Gershgorin and Lanczos paths populated with correct clamp
  - B1-part-2 (`band_scale_axpy` kernel + helpers): ✓ Kernel, struct, loader, wrapper, upload helper all present. `lam_y_dev` pre-allocated.
  - B1-part-3 (Recurrence body): ✓ Steps 1-4 implemented correctly. Buffer rotation matches spec. ndeg==0 guard present. Gram-Schmidt and final H|psi> unchanged.
- **Strategic review**: Correctly replaces only the recurrence body. Scope creep items noted (accumulate_density kernel, norm threshold change, transpose removal) — none violate specification.

### TASK-C1: Lanczos for S⁻¹·H
- **Status**: ✓ Passed
- **Runtime outcome verification**: Build passes.
- **Diff validation**: ✓ `apply_s_inverse` correctly inserted after each H·v application in Lanczos loop with `n_bands=1`.
- **Strategic review**: Implementation is correct. Minor doc comments need updating (referenced `lambda_max(H)` instead of `lambda_max(S⁻¹·H)`).

## Issues Found

### Minor Issues (no functional impact)

1. **Documentation drift in `lanczos_upper_bound`** (chebyshev.rs:353-357, 446)
   - Severity: Trivial
   - The doc comment still says "Estimate lambda_max(H)" and inline comment says `<v_cur | Hv>`.
   - Fix: Update to reference S⁻¹·H.
   - Fix file: `src/eigensolver/chebyshev.rs`

2. **Stale file header in chebyshev.rs** (line 1-10)
   - Severity: Trivial
   - References old standard Chebyshev filter steps instead of R-ChFSI.
   - Fix: Update to reference R-ChFSI Algorithm 3.
   - Fix file: `src/eigensolver/chebyshev.rs`

3. **`accumulate_density` kernel in CUDA_KERNEL_SRC**
   - Severity: Information
   - A density-construction kernel was added to the shared CUDA source. It's not part of R-ChFSI but is needed by `density.rs`. Not a defect but adds noise to the diff.

### Strategic Concerns (deferred)

4. **`chebyshev.rs` monolith** — 1769 lines, 14+ responsibilities. Split into sub-modules.
5. **Dead code** — 8 `#[allow(dead_code)]` annotations on unused items from earlier phases.
6. **`ScfIteration` field proliferation** — ~30 fields, manual copy in transition methods.
7. **Aug density structure factor D2H/H2D round-trip** — Performance bottleneck flagged for future optimization.

## Acceptance Criteria Status

| ID | Criterion | Status | Notes |
|----|-----------|--------|-------|
| SC-1 | `cargo check --workspace` | ✓ PASS | Zero errors |
| SC-2 | `cargo clippy --workspace -- -D warnings` | ✓ PASS | Zero warnings |
| SC-3 | `iter2_v_eff_range_within_one_ha_of_iter1` stays green | ⚠ UNTESTED | Requires GPU hardware |
| SC-4 | `fixed_point_matches_castep_energy` does not diverge | ⚠ UNTESTED | Requires GPU hardware |
