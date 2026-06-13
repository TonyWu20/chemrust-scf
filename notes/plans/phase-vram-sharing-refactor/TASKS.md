# VRAM Sharing Refactor — Task Groups

**Date**: 2026-06-14
**Branch**: `fix/vram-sharing-refactor`
**Parent plan**: [PLAN.md](./PLAN.md)

Each group is independently compilable (`cargo check` passes) after completion, without waiting for later groups. Groups are ordered by dependency; groups with no mutual dependency can be executed in parallel.

---

## Group 1: Remove dead Woodbury code

**Maps to**: Plan Phase 1
**Files touched**: 1
**Dependencies**: None
**Can be done in parallel with**: Group 3, Group 5 (none of these touch the same regions)

### What to change

**File: `src/eigensolver/vnl_data.rs`** (562 lines → ~430 after removal)

1. **Remove fields from `VnlBatchData` struct** (lines 47-53):
   - `b_concat: CudaSlice<CudaComplex>` — constructed but never read outside Woodbury assembly
   - `lu_m: CudaSlice<CudaComplex>` — LU factor, only consumed by `davidson_v1` behind `#[cfg(any())]`
   - `lu_ipiv: CudaSlice<i32>` — pivot indices for LU
   - `n_total_expanded: i32` — only used to dimension dead Woodbury code

2. **Remove per-ion collector vectors** (lines 193-195):
   - `let mut per_ion_q: Vec<Vec<f64>>`
   - `let mut per_ion_beta_flat: Vec<Vec<CudaComplex>>`
   - `let mut per_ion_ne: Vec<usize>`

3. **Remove per-ion push calls** (lines 355-358):
   - `per_ion_q.push(q_cpu)`
   - `let beta_this_ion = beta_flat.clone(); per_ion_beta_flat.push(beta_this_ion)`
   - `per_ion_ne.push(ne)`

4. **Remove global Woodbury assembly block** (lines 369-494):
   - This includes: block-diagonal Q assembly, Cholesky inversion, beta concatenation, B^H·B ZGEMM, M = Q⁻¹ + B^H·B construction, LU factorization with zgetrf

5. **Remove construction of removed fields** from `Ok(VnlBatchData { ... })` return expression

6. **Remove now-unused imports** (lines 12-17):
   - `use faer::linalg::solvers::{DenseSolveCore, Llt};`
   - `use faer::mat::Mat;`
   - `use faer::Side;`
   - `use crate::device::blas::{self, ZgemmConfig};` — note: `blas::ZgemmConfig` was only used for B^H·B

### Acceptance criterion

```sh
cargo check
```
Must compile with no errors. (Warnings about the now-unused `solver` parameter in `precompute_with_d_override` are acceptable — it is removed in Group 2.)

---

## Group 2: Add KptSharedVnl struct and shared Arc integration

**Maps to**: Plan Phase 2 (steps 4-8)
**Files touched**: 2 (+ 1 trivial wrapper touch)
**Dependencies**: Group 1
**Can be done in parallel with**: Group 3

### What to change

#### File: `src/eigensolver/vnl_data.rs` (major changes, ~80 lines added, ~10 removed)

1. **Define `KptSharedVnl` struct** in `vnl_data.rs` with fields:
   - `screening_cache: WaveScreeningCache` — GPU Q(G) + structure factors (was `Option` on VnlBatchData, always built)
   - `screening_cache_fine: Option<WaveScreeningCache>` — fine-grid variant
   - `per_ion_beta_g: Vec<CudaSlice<CudaComplex>>` — per-ion β(G+k) projectors
   - `per_ion_q: Vec<CudaSlice<CudaComplex>>` — per-ion USPP Q augmentation matrices (GPU)
   - `per_ion_d0_expanded: Vec<Vec<f64>>` — per-ion unscreened D0 (CPU, cheap clone)
   - `per_ion_n_expanded: Vec<i32>` — per-ion expanded projector count
   - `screening_h2d_bytes: usize` — PCI-E accounting

2. **Add `shared: Arc<KptSharedVnl>` field** to `VnlBatchData` struct definition.

3. **Remove `screening_cache`, `screening_cache_fine`, `screening_h2d_bytes`** fields from `VnlBatchData` (they now live in `self.shared`).

4. **Add `shared: Option<Arc<KptSharedVnl>>` parameter** to `precompute_with_d_override`, placed after the `d_override` parameter and before `stream`.

5. **Implement shared-aware logic** in `precompute_with_d_override`:
   - When `shared` is `Some`:
     - Skip building `screening_cache` / `screening_cache_fine` (use from `shared`)
     - Skip `compute_beta_g` / beta upload — clone `CudaSlice` from `shared.per_ion_beta_g[i]`
     - Skip `build_q_expanded` / q upload — clone `CudaSlice` from `shared.per_ion_q[i]`
     - Clone `d0_expanded` from `shared.per_ion_d0_expanded[i]`
     - Still build fresh `d_matrix` (spin-dependent, screened with current V_eff)
     - Set `self.shared = shared.clone()` (Arc bump)
     - Set `self.screening_h2d_bytes` to 0 (no new H2D for screening in reuse path)
   - When `shared` is `None` (fresh build):
     - Build screening caches as before → store in new `KptSharedVnl`
     - Build per-ion beta_g, q_matrix, d0_expanded as before
     - After the per-ion loop: construct `KptSharedVnl`, clone per-ion GPU slices into it
     - Store `Arc::new(shared_vnl)` in returned `VnlBatchData.shared`
     - Entries hold clones from the shared Arc (shallow CudaSlice clone, no VRAM cost)

6. **Update `rescreen_d`**: change `self.screening_cache` / `self.screening_cache_fine` → `self.shared.screening_cache` / `self.shared.screening_cache_fine`. Remove the `Option` unwrap on `screening_cache` (it is non-optional in `KptSharedVnl`).

7. **Update `precompute` wrapper**: pass `None` for the new `shared` parameter.

8. **Remove unused `solver` parameter** from `precompute_with_d_override` and `precompute` (it was only used by the removed Woodbury LU factorization). Update the `solver`-related import if it becomes unused.

#### File: `src/ffi.rs` (1 line mechanical)

- In `step_inner`, line ~449: add `None` argument (for `shared`) to the `VnlBatchData::precompute_with_d_override(...)` call, between `d_override` (`None`) and `&h.stream`.

#### File: `src/scf.rs` (1 line mechanical)

- In `diagonalize_inner`, line ~856: add `None` argument (for `shared`) to the `VnlBatchData::precompute_with_d_override(...)` call. The `precompute` wrapper callers in test helpers (lines 1059, 1130, 1188) are unaffected since `precompute` already passes `None`.

### Acceptance criterion

```sh
cargo check
```
Must compile with no errors. At this point, the sharing mechanism exists but is never engaged — all callers pass `shared=None`, so behavior is identical to Group 1.

---

## Group 3: FFT grid_buf explicit drop

**Maps to**: Plan Phase 5
**Files touched**: 1
**Dependencies**: None (truly independent)
**Can be done in parallel with**: Group 1, Group 2, Group 4, Group 5

### What to change

#### File: `src/ffi.rs` (1 line addition)

- In `step_inner`, after the `apply_full_hamiltonian()` call (line ~703) and before the D2H transfer block (line ~708): insert `drop(grid_buf);`.

Reason: `grid_buf` holds `n_bands * gs` complex elements on GPU. On Cu111_CO with fine grid 160×160×320, `gs ≈ 8M`, `n_bands = 106`, `grid_buf ≈ 13.6 GB` (half-precision complex). Explicit drop frees this before the D2H transfer allocates psi/hpsi host buffers, reducing peak VRAM.

### Acceptance criterion

```sh
cargo check
```
Must compile with no errors. Verify the `drop` is placed between the `apply_full_hamiltonian` call and the `clone_dtoh` calls.

---

## Group 4: Wire FFI path for spin sharing

**Maps to**: Plan Phase 3 (steps 9-12)
**Files touched**: 1
**Dependencies**: Group 2 (needs `KptSharedVnl` type and the `shared` parameter on `precompute_with_d_override`)
**Can be done in parallel with**: Group 5

### What to change

#### File: `src/ffi.rs` (~15 lines added/modified)

1. **Add `shared_vnl: Option<Arc<KptSharedVnl>>` field** to `KptData` struct (line ~55-64). Initialised to `None` in `init_inner` (line ~254).

2. **Update import**: add `use std::sync::Arc;` if not already present (it is, line 7).

3. **Wire lazy-init sharing** in `step_inner` (around line 440-454):
   - Before the `if kd.vnl[isp].is_none()` block: read `let shared_for_this_spin = kd.shared_vnl.clone();`
   - Pass `shared_for_this_spin` instead of `None` in the `precompute_with_d_override` call
   - After successful construction: `kd.shared_vnl = Some(kd.vnl[isp].as_ref().unwrap().shared.clone());`

   Logic flow:
   - First spin-0 call: `kd.shared_vnl` is `None` → fresh build → `precompute_with_d_override` creates and returns `KptSharedVnl` inside the Arc → store Arc on `kd.shared_vnl`
   - Second spin-1 call: `kd.shared_vnl` is `Some(arc)` → `arc.clone()` passed as `shared=Some(...)` → `precompute_with_d_override` skips screening/beta_g/q_matrix and clones from shared

### Acceptance criterion

```sh
cargo check
```
Must compile with no errors. Non-spin path must not break (it exercises the same code with `nspins=1`, so `shared_vnl` is built fresh and never reused — no behavioral change for non-spin).

---

## Group 5: Wire standalone SCF path for spin sharing

**Maps to**: Plan Phase 4 (steps 13-14)
**Files touched**: 1
**Dependencies**: Group 2 (needs `KptSharedVnl` type and the `shared` parameter on `precompute_with_d_override`)
**Can be done in parallel with**: Group 4

### What to change

#### File: `src/scf.rs` (~10 lines added/modified)

1. **Add `shared_vnl_cache: Vec<Option<Arc<KptSharedVnl>>>`** before the spin loop in `diagonalize_inner` (around line 786, before `for ispin in 0..nspins`):
   ```rust
   let mut shared_vnl_cache: Vec<Option<Arc<KptSharedVnl>>> = vec![None; nkpts];
   ```

2. **Wire per-kpt Arc** inside the kpt loop (around line 856):
   - Before the `VnlBatchData::precompute_with_d_override` call: read `let shared_vnl = shared_vnl_cache[ikpt].clone();`
   - Pass `shared_vnl` instead of `None` to `precompute_with_d_override`
   - After successful construction: `shared_vnl_cache[ikpt] = Some(vnl_data.shared.clone());`

   Logic:
   - Spin-0, kpt-i: `shared_vnl_cache[i]` is `None` → fresh build → Arc stored
   - Spin-1, kpt-i: `shared_vnl_cache[i]` is `Some(arc)` → reuse

### Acceptance criterion

```sh
cargo check
```
Must compile with no errors. The standalone SCF tests (behind `#[cfg(test)]` and `#[cfg(feature = "chebyshev")]`) must compile. Non-spin standalone SCF path must be unaffected.

---

## Group 6: Build and integration test

**Maps to**: Plan Phase 6 (steps 16-19)
**Files touched**: 0
**Dependencies**: Groups 1, 2, 3, 4, 5 (all prior groups must be complete)

### Tests to run

1. **`cargo check`** — must compile clean (no errors, no warnings).

2. **`cargo test --release`** — all Rust unit and integration tests must pass. Key tests:
   - Davidson eigensolver tests
   - Preconditioner tests
   - Chebyshev filter tests
   - SCF pipeline tests (non-spin)
   - Hamiltonians/nonlocal V_NL tests for both spin channels

3. **Non-spin NiO FFI cold-start** — must converge in the CASTEP harness. Regresses against pre-refactor behavior (no change expected — non-spin uses fresh build every time).

4. **Spin-polarised Cu111_CO FFI cold-start** — must NOT OOM on GTX1080Ti (11 GB). Must converge (ΔE within tolerance vs CPU reference). This is the primary acceptance criterion of the entire refactor.

### Non-goals for this group

- Cu111_CO energy must match the previous converged reference (within 1e-4 Ha), but exact bit-level reproducibility is not required — the refactor changes only memory management, not arithmetic.
- Performance benchmarks — VRAM reduction is the acceptance criterion, not wall time improvement.

---

## Dependency Graph

```
Group 1 ──┬── Group 2 ──┬── Group 4 ──┬── Group 6
           │             │             │
Group 3 ───┘             └── Group 5 ──┘
```

- Group 1, Group 3 can run in parallel (no shared files)
- Group 4, Group 5 can run in parallel (both depend only on Group 2, touch different files)
- Group 6 requires all prior groups

## Estimated effort

| Group | Files | Est. changed lines | Risk |
|-------|-------|--------------------|------|
| 1 | 1 | -130 (removal) | Low — purely dead code |
| 2 | 3 | +80 / -10 | Medium — new struct, function signature change |
| 3 | 1 | +1 | Low — one-line drop |
| 4 | 1 | +15 | Low — conditionally pass Arc |
| 5 | 1 | +10 | Low — same pattern as Group 4 |
| 6 | 0 | 0 | Verification only |
