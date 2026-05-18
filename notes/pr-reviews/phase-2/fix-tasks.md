# Fix Tasks: Phase 2 Group-B — GPU Infrastructure

## P1: Fix placebo test `test_batched_c2r_4x4x4`

Replace the vacuous `any(|&v| v != 0.0)` assertion with a per-grid-point check.

**Files:** `src/device/fft.rs`

**Guidance:** For each band in the batched IFFT, the DC-only input spectrum (only the first complex element is `1.0+0i`, rest zero) must produce a uniform real output field. The expected value for each band is `1.0 / (nx*ny*nz)` at every grid point. The test should:
1. Compute expected value
2. For each band, verify all `nx*ny*nz` output elements equal the expected value within `1e-10`
3. Verify inter-band independence (no cross-talk)

**Success Criteria:**
- Single-band DC-only IFFT on 4³ grid produces uniform `1.0/64.0` everywhere
- 4-band batched IFFT produces same per-band as single-band (proves batch stride correctness)
- A band with zero input does not leak into adjacent band's output

---

## P2: Fix batched FFT to use `cufftPlanMany` or add batch-stride verification

**Files:** `src/device/fft.rs`

**Guidance:** The current `BatchedFftPlan3d` uses `plan_3d()` which ignores the batch parameter. Two options:

**Option A (preferred):** Replace with proper `cufftPlanMany` wrapper:
1. Add `cufftPlanMany` sys FFI call to create a plan with proper batch stride parameters
2. Set `idist`/`odist` for batched C2R with `n[0]*n[1]*(n[2]/2+1)` complex input stride and `n[0]*n[1]*n[2]` real output stride
3. Set `inembed`/`onembed` to `[nz, ny, nx]` (row-major FFT convention)
4. Associate stream via `cufftSetStream`

**Option B (minimal):** If contiguous batches are provably correct for density construction, add `#[allow(dead_code)]` removal depends on proper `cufftPlanMany`. Choose Option A.

**Success Criteria:**
- `test_batched_c2r_4x4x4` from P1 fix passes with `cufftPlanMany`-backed plan
- Plan works with non-unit batch strides (test with gap between batches)

---

## P3: Return `Result` instead of panicking on cuSOLVER workspace allocation failure

**Files:** `src/device/solver.rs`

**Guidance:**
- Change `unwrap_or_else(|e| panic!(...))` to use `?` operator
- Map `DriverError` to a new variant on `CusolverError` (or define a `SolverError` that wraps it)
- The workspace allocation failure should propagate up the call stack, not abort the process

**Success Criteria:**
- `cargo check` — no panic paths in solver.rs allocation
- Error propagates as `Result` to caller

---

## P4: Add complex GEMM unit test

**Files:** `src/device/blas.rs` (add to existing `#[cfg(test)] mod tests`)

**Guidance:** Add `test_zgemm_small` that:
1. Creates a 2×2 complex matrix A and a 2×2 complex matrix B on GPU
2. Computes C = A·B via `gemm_c64`
3. D2H result and compares against hand-calculated reference
4. Use known simple values like `A = [[1+i, 0], [0, 1-i]]`, `B = [[1, 0], [0, 1]]`

**Success Criteria:**
- Complex GEMM result matches CPU reference within 1e-10
- Test passes on CUDA GPU

---

## P5: Add Cpu<T>::sync_to_device() thin wrapper

**Files:** `src/layout.rs`

**Guidance:** Add a convenience method to `Cpu<T>`:
```rust
pub fn sync_to_device(&self, stream: &Arc<CudaStream>) -> Result<Gpu<T>, DriverError>
where
    T: DeviceMapped,
{
    Gpu::from_host(&self.0, stream)
}
```
This requires importing `Gpu` and `DeviceMapped` from `device` module — watch for circular dependency.

**Success Criteria:**
- `cpu.sync_to_device(&stream)` returns `Gpu<T>`
- `cargo check` passes
