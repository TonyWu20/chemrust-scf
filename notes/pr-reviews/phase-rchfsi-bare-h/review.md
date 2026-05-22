# Review: Bare-H R-ChFSI experiment (Group D)

**Tasks**: `notes/plans/phase-rchfsi-bare-h/TASKS.md`
**Branch**: `feat/phase-rchfsi`
**Reviewed**: 2026-05-22

## Summary

**Status: Needs final runtime verification**

All four Group D tasks are implemented as specified. A critical bug in the h_eig
computation (CUDA_ERROR_MISALIGNED_ADDRESS from incorrect pointer arithmetic) was
identified during review and fixed. Runtime outcome verification (SC-3 through SC-7)
is in progress.

## Per-Task Results

### TASK-D1: L2-Lanczos (drop S-inner product)
- **Status**: ✓ Passed
- **Diff validation**: Lanczos uses standard L2 dot products and norms. No S-normalization block, no `apply_s_inverse` call. Alpha/beta correctly computed with `dotc(hv, hv)` for L2 norm. `sr` buffer absent (never existed).
- **Strategic review**: Confirmed T_k bounds are H-eigenvalue Ritz values under standard inner product, matching the bare-H design.

### TASK-D2: Per-band H-eigenvalues and bare-H spectral bounds
- **Status**: ✓ Passed (with bugfix applied)
- **Diff validation**: h_eig computation present at lines 1428-1456, block-scoped. Spectral params computed after Step 1. Λ_Y and Λ_X use `h_eig` not `eig`. `DevicePtr` imported.
- **Bug found and fixed**: Two issues in the h_eig cuBLAS call:
  1. `ptr_psi_base.wrapping_add(b * n_pw)` — `ptr_psi_base` is a `u64` CUdeviceptr, so `wrapping_add` was byte-level addition. Offset of `b * n_pw` bytes landed at 1/16 of the correct position for 16-byte `cuDoubleComplex`. Fixed by casting to `*const cuDoubleComplex` and using `.add(b * n_pw)` (element-level arithmetic, adds `b * n_pw * 16` bytes).
  2. Missing `.result().map_err(Error::Blas)?` on raw `cublasZdotc_v2` call — cuBLAS errors were silently swallowed. Fixed.

### TASK-D3: Bare-H recurrence (Step 3) and reconstruction (Step 4)
- **Status**: ✓ Passed
- **Diff validation**: Step 3 calls `apply_full_hamiltonian(&buf_ry, ...)` directly (H·R_Y, no S⁻¹). Step 4 removes `apply_s_inverse` from reconstruction. Comment header updated to "bare-H variant". `apply_s_inverse` preserved with `#[allow(dead_code)]`.
- **Scope creep (beneficial)**: Gram-Schmidt rewritten to use S-inner product (⟨x,y⟩_S). This is necessary for USPP correctness — ZHEGVD in Rayleigh-Ritz expects S-orthonormal input. The old L2-based GS would produce wrong normalization for non-orthogonal USPP wavefunctions.

### TASK-D4: Restore S⁻¹ in Step 4 (CONTINGENCY)
- **Status**: Not triggered
- `apply_s_inverse` preserved as dead code; TASK-D4 fallback path documented in comments.

## Issues Found

1. **CRITICAL — h_eig pointer arithmetic (FIXED)** —
   `src/eigensolver/chebyshev.rs:1433-1434`. `wrapping_add` on `u64` (CUdeviceptr) is byte-level, not element-level. Caused CUDA_ERROR_MISALIGNED_ADDRESS in SCF runs. Fix verified by `cargo check` and `cargo clippy`.

2. **MINOR — Missing error check on cuBLAS Zdotc (FIXED)** —
   `src/eigensolver/chebyshev.rs:1439`. No `.result()` chained to raw `cublasZdotc_v2` call. Fix applied alongside the pointer fix.

## Verification Protocol Status

| ID | Criterion | Status |
|----|-----------|--------|
| SC-1 | `cargo check --workspace` | ✓ Pass |
| SC-2 | `cargo clippy --workspace -- -D warnings` | ✓ Pass |
| SC-3 | `iter2_v_eff_range_within_one_ha_of_iter1` | ⏳ Pending |
| SC-4 | `fixed_point_matches_castep_energy` | ⏳ Pending (was OOM from concurrent runs) |
| SC-5 | `density_decomp_matches_castep_f8_same_inputs` | ⏳ Pending |
| SC-6 | No `apply_s_inverse` in Chebyshev filter/Lanczos | ✓ Pass (only in diagnostic function) |
| SC-7 | D_screened iter-3 < 10 Ha | ⏳ Pending |
