# Review: Phase 2 Group-B — GPU Infrastructure

**Tasks**: `notes/plans/phase-2/TASKS.md` (Group B: B-1 through B-5)
**Reviewed**: 2026-05-19
**Focus mandate**: Violation of newtype pattern, placebo unit tests, cutting corners

## Summary

**Conditional Pass — 2 placebo tests, 2 corner-cutting defects.**

Runtime outcome verification: 8/8 GPU tests pass on CUDA 12.9 hardware. `cargo check` and `cargo clippy` clean. The Gpu/T̵<>/Cpu<> sync architecture, cuFFT, cuBLAS, and cuSOLVER wrappers all execute correctly on real GPU hardware.

However, 2 unit tests are placebo-grade (vacuous assertions that pass even with wrong output), and 2 implementation details cut corners against the TASKS.md guidance. 1 newtype encapsulation gap exists in the new Gpu<T> layer.

## Per-Task Results

### B-1: Add cudarc dependency
- **Status**: ✓ Passed
- **Runtime verification**: `cargo check` resolves cudarc v0.19.7 with CUDA 12.9; `nix develop` provides CUDA env vars.
- **Diff validation**: Cargo.toml adds `cudarc = { version = "0.19.7", features = ["cuda-12090", "cufft", "cusolver"] }`. The `cublas` feature is transitively activated by `cusolver`.
- **Strategic review**: No issues. Correct version targeting CUDA 12.9.

### B-2: Gpu<T>/Cpu<T> real sync
- **Status**: ⚠ Minor Issues
- **Runtime verification**: All 3 device round-trip tests pass (density, wavefunction, type distinction).
- **Diff validation**:
  - `Gpu<T>` moved from `layout.rs` to `device/mod.rs`. Fields are **private** — correct newtype encapsulation. ✓
  - `Gpu<T>` has no `Deref<Target=T>` ✓
  - `Cpu<T>` retains Deref/DerefMut to T ✓
  - `Gpu::from_host()` and `Gpu::from_cpu()` provide H2D construction ✓
  - `unsafe impl Send/Sync` for `Gpu<T>` ✓
  - `sync_to_host(&self, stream) -> Result<Cpu<T>>` works correctly ✓
  - **Corner cut**: No `Cpu<T>::sync_to_device()` method. The TASKS.md specified `Cpu<T>::sync_to_device(...) -> Gpu<T>`. Instead, `Gpu::from_cpu()` provides the equivalent. This is a minor API difference.
  - **Newtype concern**: `Cpu<T>(pub T)` still uses `pub` field — pre-existing issue, not introduced here.
- **Strategic review**: Design is sound. The `DeviceMapped` trait cleanly separates element-type dispatch. No Deref on Gpu prevents accidental CPU reads.

### B-3: cuFFT wrapper
- **Status**: ⚠ Minor Issues (1 placebo test, 1 corner cut)
- **Runtime verification**: `test_c2c_3d_identity` passes (forward+inverse 8³ identity). `test_batched_c2r_4x4x4` passes.
- **Diff validation**:
  - `FftPlan3d` with C2C, D2Z, Z2D plan creation ✓
  - `BatchedFftPlan3d` struct created ✓
  - **PLACEBO TEST**: `test_batched_c2r_4x4x4` uses assertion `result.iter().any(|&v| v != 0.0)` — only checks output is not all-zeros. This would pass even with wrong normalization, wrong batch stride, or corrupted data (as long as any byte is non-zero). A proper test would validate each band's IFFT of a DC-only spectrum produces a uniform field `1.0/(nx*ny*nz)` at every grid point.
  - **Corner cut**: `BatchedFftPlan3d` uses `CudaFft::plan_3d()` (single-plan), NOT `cufftPlanMany` as the guidance specifies. The `batch` parameter is stored but never passed to cuFFT. For density construction, `cufftPlanMany` with proper striding is needed to transform N_bands non-contiguous batches correctly.
  - `BatchedFftPlan3d::plan_batched_c2c` accepts `batch` parameter but ignores it entirely.
- **Strategic review**: The batched plan is structurally identical to the non-batched plan — same inner type, same execute methods. This works for contiguous batch layouts but the guidance specifically called for `cufftPlanMany`. If non-contiguous strides are needed later, this will break silently.

### B-4: cuBLAS wrapper
- **Status**: ⚠ Minor Issues (1 missing test)
- **Runtime verification**: `test_dgemm` (DGEMM 3×2×4) and `test_daxpy` (scale + add) pass with correct reference values.
- **Diff validation**:
  - `BlasHandle` with stream association ✓
  - `gemm_f64`, `gemm_c64`, `gemv_f64`, `axpy_f64`, `axpy_c64`, `dot_f64`, `dotc_c64`, `iamax_f64` all implemented ✓
  - **Missing test**: TASKS.md specifies "Unit test: gemm for small complex matrix multiplication on GPU" — there is no complex GEMM test. Only `test_dgemm` (f64) exists. The `gemm_c64` code path is untested.
  - **Missing test**: `gemv_f64` is implemented but untested.
- **Strategic review**: The `ZgemmConfig` struct duplicates `GemmConfig` fields and uses raw sys FFI. This is acceptable if cudarc's safe `GemmConfig` doesn't support complex types — but the duplicate struct is a maintenance burden.

### B-5: cuSOLVER ZHEGVD wrapper
- **Status**: ⚠ Minor Issues (1 panic path)
- **Runtime verification**: `test_zhegvd_4x4_diagonal` passes (eigenvalues [1,2,3,4] with info=0).
- **Diff validation**:
  - Workspace query via `cusolverDnZhegvd_bufferSize` ✓
  - Workspace allocation ✓
  - Proper enum types (`cusolverEigMode_t`, `cublasFillMode_t`) instead of raw i32 ✓
  - **Corner cut**: Workspace allocation uses `unwrap_or_else(|e| panic!(...))` instead of returning `Result`. This violates the codebase error-handling convention (typed error enums via thiserror, Result propagation).
  - Test uses only diagonal matrix — simplest possible case. No non-diagonal Hermitian matrix test.
  - Signature uses proper cuSOLVER types (improvement over TASKS.md's raw i32) ✓
- **Strategic review**: The solver wrapper is clean and follows cuSOLVER best practices (bufferSize + allocate + solve). The panic path is the only defect.

## Issues Found

### P1 — Placebo test: `test_batched_c2r_4x4x4`

**File**: `src/device/fft.rs:274-275`
**Severity**: Medium
**Description**: Assertion `result.iter().any(|&v| v != 0.0)` only checks output is non-zero. It does not validate correct values, correct normalization, or correct batch handling. A wrong implementation that produces garbage non-zero output would pass.
**Recommendation**: Replace with per-band assertion: for DC-only input spectrum, each band's IFFT output should be `1.0/(nx*ny*nz)` at every grid point within FP tolerance.

### P2 — Missing `cufftPlanMany` for batched transforms

**File**: `src/device/fft.rs:127-136`
**Severity**: Medium
**Description**: `BatchedFftPlan3d` calls `CudaFft::plan_3d()` (single-plan API) ignoring the `batch` parameter. The guidance explicitly calls for wrapping `cufftPlanMany` which supports arbitrary batch strides. The current implementation only works for contiguous batch layouts. If density construction requires strided batch access (non-contiguous bands), this will produce wrong results silently.
**Recommendation**: Implement a separate path using `cufftPlanMany` sys FFI, or document that the current impl only supports contiguous batches and verify that density construction's layout is indeed contiguous.

### P3 — Panic on cuSOLVER workspace allocation failure

**File**: `src/device/solver.rs:70-71`
**Severity**: Medium
**Description**: `unwrap_or_else(|e| panic!(...))` on workspace allocation. This should return `Err(CusolverError)` via `?` instead of panicking, consistent with the rest of the codebase's error handling.
**Recommendation**: Change to `let workspace = self.stream.alloc_zeros::<CudaComplex>(lwork as usize).map_err(|e| ...)?;`.

### P4 — Missing complex GEMM test

**File**: `src/device/blas.rs` (tests)
**Severity**: Low
**Description**: TASKS.md success criteria requires "Unit test: gemm for small complex matrix multiplication on GPU". The `gemm_c64` method is untested.
**Recommendation**: Add a small ZGEMM test (e.g., 2×2 complex matrix multiplication with known reference values).

### P5 — Missing `Cpu<T>::sync_to_device()` method

**File**: `src/device/mod.rs`
**Severity**: Low
**Description**: TASKS.md guidance specified `Cpu<T>::sync_to_device(...) -> Gpu<T>`. The equivalent exists as `Gpu::from_cpu()` but the Cpu-side method is absent. Callers must know to use `Gpu::from_cpu()` instead of `cpu.sync_to_device()`.
**Recommendation**: Either add `Cpu<T>::sync_to_device()` as a thin wrapper or document the asymmetry.

### P6 — `#[allow(dead_code)]` on `BatchedFftPlan3d`

**File**: `src/device/fft.rs:113`
**Severity**: Low
**Description**: `#[allow(dead_code)]` indicates the batched plan struct is never used. Expected consumer (density construction in Group D) is not yet implemented, so this is a forward-reference stub.
**Recommendation**: Remove `#[allow(dead_code)]` once density construction uses it.

## Deferred Items

See `deferred.md`.
