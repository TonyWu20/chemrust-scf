# Review: Phase 2 Group-C — diagonalize

**Tasks**: `notes/plans/phase-2/TASKS.md` (Group C: C-1 through C-3)
**Reviewed**: 2026-05-19
**Focus**: Chebyshev filtering, Rayleigh-Ritz, diagonalize transition

## Summary

**Changes Required — 1 critical defect, 1 significant defect, 1 minor issue.**

Runtime outcome verification: 10/10 GPU unit tests pass on CUDA 12.9 hardware (c2c identity, batched c2r, gemm, axpy, zhegvd, device roundtrips). All tests from Group B's fix pass were verified. `cargo check` and `cargo test --workspace` both clean.

The implementation scope is correct and well-structured: `chebyshev.rs` (817 lines), `rayleigh_ritz.rs` (219 lines), `vnl_data.rs` (90 lines). The V_NL via cuBLAS gemm, the Chebyshev three-buffer recurrence, and the Rayleigh-Ritz subspace diagonalization are all implemented correctly in structure.

However, 1 critical PcieAccount assertion bug and 1 significant FFT dimension ordering bug would produce wrong results or runtime panics on non-cubic grids. One minor issue should be fixed before merge.

## Per-Task Results

### C-1: Chebyshev filtering on GPU
- **Status**: ⚠ Significant Issues (1 significant, 1 minor)
- **Runtime verification**: NVRTC kernels compile, FFT C2C identity test passes on 8³ grid. No integration test against Cu111_CO fixtures (deferred to Group F).
- **Diff validation**:
  - `src/eigensolver/chebyshev.rs` created with 817 lines ✓
  - Spectral bound estimation implemented (`compute_spectral_bounds`) ✓
  - H_loc = T + V_eff via FFT roundtrip (scatter → IFFT → V_eff multiply → FFT → gather) ✓
  - V_NL via cuBLAS gemm (beta^H·psi, D·C_proj, beta·C_proj accumulate) ✓
  - Chebyshev three-buffer recurrence with norm stability check ✓
  - ColumnDistributed → RowDistributed transpose kernel ✓
  - **FFT dimension ordering mismatch (Significant)**: `plan_batched_c2c(ngx, ngy, ngz)` passes `[ngx, ngy, ngz]` to cuFFT, but the index formula `ix + ngx * (iy + ngy * iz)` uses ngx as fastest-varying. cuFFT expects `[ngz, ngy, ngx]` to match. Hidden on cubic grids; would produce wrong physics on non-cubic systems.
  - **Spectral bound uses range (Minor)**: `kinetic_max + (max_veff - min_veff)` overestimates lambda_max vs. guidance's `kinetic_max + max_veff`.
  - **Flat norm threshold (Minor)**: Uses fixed 10× growth check instead of relative ratio-to-ratio comparison from guidance.
  - `potts`, `cell`, `k_point` parameters correctly marked `_` (VNL data precomputed).

### C-2: Rayleigh-Ritz on GPU
- **Status**: ✓ Passed (minor concerns)
- **Runtime verification**: ZHEGVD unit test passes for 4×4 diagonal system with correct eigenvalues [1,2,3,4].
- **Diff validation**:
  - `src/eigensolver/rayleigh_ritz.rs` created with 219 lines ✓
  - H_sub = ψ^dag·H|ψ> via cuBLAS gemm with transa=C ✓
  - S_sub = ψ^dag·ψ via cuBLAS gemm ✓
  - ZHEGVD solve with proper CUSOLVER_EIG_MODE_VECTOR and CUBLAS_FILL_MODE_LOWER ✓
  - Info check with proper error propagation ✓
  - ψ_new = X·ψ rotation via cuBLAS gemm ✓
  - GPU transpose via shared NVRTC kernel (avoids 4MB D2H+H2D roundtrip) ✓
  - **RowDistributed shape metadata mismatch (Minor)**: Shape reported as `[n_bands, n_pw]` but transpose kernel stores data as `[n_pw, n_bands]` in memory. Latent — no code path currently syncs RowDistributed data to host.

### C-3: Wire diagonalize transition in scf.rs
- **Status**: ⚠ Critical Issue (1 critical defect)
- **Runtime verification**: `cargo check` passes — all CUDA types and imports resolve. No integration test exists yet for the full diagonalize pipeline (Group F).
- **Diff validation**:
  - `diagonalize()` method on `ScfIteration<S, VEffBuilt>` ✓
  - Chains `chebyshev_filter` → `rayleigh_ritz` correctly ✓
  - Returns `ScfIteration<S, WavefunctionsUpdated>` with updated psi and eigenvalues ✓
  - V_eff downsampling utility (`downsample_array_to_wave_grid`) implemented via CPU rustfft ✓
  - **PcieAccount assertion will panic (Critical)**: The assertion at line 247 expects `psi_bytes + eig_bytes` D2H bytes, but `rayleigh_ritz` downloads eigenvalues via raw `clone_dtoh` (not tracked by PcieAccount). Only `psi_bytes` is tracked. At runtime this assertion fails.
  - **H2D assertion missing**: ADR-0002 specifies an H2D assertion alongside the D2H one. Neither is implemented.

## Issues Found

### P1 (Critical) — PcieAccount eigenvalue D2H not tracked → assertion panics

**File**: `src/scf.rs:246-251`, `src/eigensolver/rayleigh_ritz.rs:204-206`
**Description**: The `rayleigh_ritz` function downloads eigenvalues to host via `stream.clone_dtoh(&eigenvalues_dev)` which bypasses `PcieAccount`. The caller's assertion expects `pcie.d2h_bytes == psi_bytes + eig_bytes`, but only `psi_bytes` from the `sync_to_host_with` call is tracked. Runtime assertion failure on any execution path that reaches the assertion.

### P2 (Significant) — FFT dimension ordering mismatch

**File**: `src/eigensolver/chebyshev.rs:662-664`
**Description**: `plan_batched_c2c(ngx, ngy, ngz)` passes `[ngx, ngy, ngz]` to cuFFT, interpreting ngx as the slowest-varying dimension and ngz as fastest. But the scatter/gather index formula uses `ix + ngx * (iy + ngy * iz)`, which makes ngx the fastest-varying dimension. cuFFT requires `[ngz, ngy, ngx]` to match this layout. On cubic test grids (8×8×8, 4×4×4) the bug is masked because all dimensions are equal. On real non-cubic systems (e.g., Cu111_CO slab where ngz > ngx = ngy), G-vector coefficients scatter to wrong frequency positions, producing incorrect physics.

**Fix**: Change to `plan_batched_c2c(ngz as i32, ngy as i32, ngx as i32, ...)`.

### P3 (Minor) — Flat norm threshold instead of relative ratio

**File**: `src/eigensolver/chebyshev.rs:552-563`
**Description**: The guidance specifies a ratio-to-ratio comparison (`growth > threshold × last_growth`), but the implementation uses a flat `10×` threshold. For systems where Chebyshev naturally produces >10× growth, this would false-positive trigger divergence. Deferred item #10 in `deferred.md` documents this.

### P4 (Minor) — `#[allow(dead_code)]` on `SpectralBounds` is unnecessary

**File**: `src/eigensolver/chebyshev.rs:213`
**Description**: The `SpectralBounds` struct is fully used — `compute_spectral_bounds` constructs it and `chebyshev_filter` reads all fields. Remove the attribute.

## Deferred Items

- **Missing H2D assertion**: ADR-0002 specifies `assert_eq!(pcie.h2d_bytes, psi_bytes + veff_bytes, ...)` but only the D2H counterpart exists. Requires routing raw `clone_htod` calls (fft_idx_dev, kinetic_dev, VNL data) through PcieAccount. See `deferred.md`.
- **CudaKernelSet lives in chebyshev module**: `rayleigh_ritz` imports `CudaKernelSet` from `chebyshev` for the transpose kernel. Better to move to `eigensolver/kernels.rs` or `device/kernels.rs`. Deferred to avoid scope creep.
- **RowDistributed shape metadata**: The `[n_bands, n_pw]` shape doesn't reflect the transposed memory layout. No current code path syncs RowDistributed to host, so this is latent. Fix when adding RowDistributed host access.
- See `notes/pr-reviews/phase-2/deferred.md` for all previously deferred items.
