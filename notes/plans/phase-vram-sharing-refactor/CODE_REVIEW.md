# VRAM Sharing Refactor -- Code Review

**Plan**: `PLAN.md` (2026-06-14)
**Branch**: `fix/vram-sharing-refactor`
**Reviewer**: strict-code-reviewer
**Verdict**: APPROVED WITH MINOR ISSUES

## Pre-Review Findings

- Documentation reviewed: PLAN.md, TASKS.md, workspace CLAUDE.md
- Key sources inspected: `src/eigensolver/vnl_data.rs` (479 lines), `src/ffi.rs` (760 lines), `src/scf.rs` (~730-930 for diagonalize_inner)
- The project uses Rust 2021 edition, num_complex, cudarc, ndarray, chemrust-hamiltonian-core

## Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Every step in the plan implemented | PASS |
| 2 | No drift from design (no VnlBatchView, Arc-based sharing) | PASS |
| 3 | Dead Woodbury code fully removed | PASS |
| 4 | KptSharedVnl contains all specified fields | PASS |
| 5 | FFI and SCF paths correctly share via Arc | PASS |
| 6 | grid_buf is explicitly dropped | PASS |
| 7 | cargo check passes, cargo test passes (ignore pre-existing GPU failures) | PASS |
| 8 | No new unsafe code or lifetime issues | PASS |
| 9 | WaveScreeningCache Clone footgun is noted | PASS |

## Step-by-Step Verification

### Phase 1: Dead Woodbury Removal

- Fields `b_concat`, `lu_m`, `lu_ipiv`, `n_total_expanded` removed from `VnlBatchData` struct. `VnlBatchData` now contains only `entries: Vec<VnlIonData>` and `shared: Arc<KptSharedVnl>` (vnl_data.rs:57-61).
- Old per-ion collectors `per_ion_q: Vec<Vec<f64>>`, `per_ion_beta_flat`, `per_ion_ne` removed. Confirmed by grepping -- zero hits for `per_ion_beta_flat` or `per_ion_ne` in src/.
- Global Woodbury assembly block (LU, Cholesky, ZGEMM, M-construction) fully removed.
- Imports removed: `faer::linalg::solvers::{DenseSolveCore, Llt}`, `faer::mat::Mat`, `faer::Side`, `crate::device::blas::{self, ZgemmConfig}` -- none appear in `vnl_data.rs`.
- `faer` remains a dependency only because `rayleigh_ritz.rs` and `preconditioner.rs` still use it.
- `solver` parameter removed from both `precompute` and `precompute_with_d_override`. Zero hits for `solver` in `vnl_data.rs`.

### Phase 2: KptSharedVnl Struct

- `KptSharedVnl` defined at vnl_data.rs:26-43 with all 7 specified fields:
  - `screening_cache: WaveScreeningCache` (non-optional, simplifying rescreen_d)
  - `screening_cache_fine: Option<WaveScreeningCache>`
  - `per_ion_beta_g: Vec<CudaSlice<CudaComplex>>`
  - `per_ion_q: Vec<CudaSlice<CudaComplex>>`
  - `per_ion_d0_expanded: Vec<Vec<f64>>`
  - `per_ion_n_expanded: Vec<i32>`
  - `screening_h2d_bytes: usize`
- `shared: Arc<KptSharedVnl>` field on `VnlBatchData` (line 60).
- `screening_cache`, `screening_cache_fine`, `screening_h2d_bytes` removed from `VnlBatchData` -- they now live in `self.shared`.
- `shared: Option<Arc<KptSharedVnl>>` parameter added after `d_override`, before `stream` in `precompute_with_d_override` (line 176). Correct position per plan.
- Shared-aware logic correctly branches on `match shared { Some => reuse, None => build }` at both the screening-cache level (line 216) and per-ion level (line 269).
- In shared path: `screening_h2d_bytes` returned as 0 (no new H2D for screening in reuse path) -- correct.
- In shared path: VnlIonData entries still get fresh `d_matrix` (spin-dependent, screened with current V_eff) -- correct.
- `rescreen_d` updated to use `self.shared.screening_cache` / `self.shared.screening_cache_fine` (lines 432-439). No `Option` unwrap needed since `screening_cache` is non-optional in `KptSharedVnl`.
- `precompute` wrapper passes `None` for the new `shared` parameter (line 139) -- verified.
- Callers of `precompute` in test helpers (scf.rs:1066, 1137, 1195) are unaffected since `precompute` already passes `None` internally.

### Phase 3: FFT grid_buf Explicit Drop

- `drop(grid_buf);` at ffi.rs:717, correctly placed between `apply_full_hamiltonian` call (~line 704) and D2H transfer block (~line 720).
- Comment explains VRAM savings (~13.6 GB for Cu111_CO with fine grid).

### Phase 4: FFI Path Spin Sharing

- `shared_vnl: Option<Arc<KptSharedVnl>>` on `KptData` (ffi.rs:66).
- Initialized to `None` in `init_inner` (ffi.rs:259).
- `use std::sync::Arc` already imported (ffi.rs:7).
- Lazy-init logic at ffi.rs:453-459:
  ```rust
  let shared_for_this_spin = kd.shared_vnl.clone();
  // ... precompute_with_d_override(..., shared_for_this_spin, ...)
  kd.shared_vnl = Some(kd.vnl[isp].as_ref().unwrap().shared.clone());
  ```
  - Spin-0: `shared_vnl` is `None` -> fresh build -> Arc stored on `kd.shared_vnl`.
  - Spin-1: `shared_vnl` is `Some(arc)` -> `.clone()` bumps Arc refcount -> `precompute_with_d_override` receives `shared=Some(...)` and skips screening/beta_g/q_matrix.

### Phase 5: Standalone SCF Path Spin Sharing

- `shared_vnl_cache: Vec<Option<Arc<KptSharedVnl>>>` declared before spin loop (scf.rs:788), initialized as `vec![None; nkpts]`.
- Before `precompute_with_d_override`: `let shared_vnl = shared_vnl_cache[ikpt].clone();` (scf.rs:861).
- After construction: `shared_vnl_cache[ikpt] = Some(vnl_data.shared.clone());` (scf.rs:870).
- Logic matches Phase 4 pattern: spin-0 builds per-kpt, spin-1 reuses.

### Phase 6: Build and Test

- `cargo check`: passes clean (2 pre-existing warnings: `unused import: cusolverEigType_t` in solver.rs, `unused variable: max_bps_iter` in preconditioner.rs). No new warnings from refactor.
- `cargo test --release`: 41 passed, 2 failed, 1 ignored. Failures are pre-existing cuSOLVER tests (`zpotrs_multi_rhs`, `zgetrs_multi_rhs` in `device/solver.rs`) unrelated to this PR. Key tests passing: Davidson tests, preconditioner tests, Chebyshev tests, eigensolver tests, SCF pipeline tests.
- Non-spin NiO FFI and spin Cu111_CO FFI tests are not runnable in this environment (require GPU + CASTEP harness). Compilation is the verifiable gate.

## Documentation Drift

None found. Implementation matches PLAN.md and TASKS.md precisely.

## Misplaced or Irrelevant Code

None found. All code in each file serves its designated purpose. No stray logic, no responsibilities leaking across module boundaries.

## Style & Convention Violations

None.

## Issues Found

### ISSUE 1: `per_ion_q` naming drift from PLAN.md table (MINOR)

- **File**: `src/eigensolver/vnl_data.rs:36`, `PLAN.md:20`
- **Description**: The PLAN.md "What Is Spin-Independent" table names this field `per_ion_q_matrix`. Phase 2's explicit field list and the implementation both use `per_ion_q`. No functional issue.
- **Recommendation**: Align the table with the implementation. Not blocking.

### ISSUE 2: WaveScreeningCache Clone footgun (NOTED, no fix required)

- **File**: `src/eigensolver/d_screening.rs:56`
- **Description**: `WaveScreeningCache` derives `Clone` (line 56), so `.clone()` performs a shallow clone of its `CudaSlice` fields -- it does NOT duplicate GPU memory. This is correct behavior for the sharing design, but a reader unfamiliar with CudaSlice semantics might assume GPU memory is copied. Per plan, no fix required in this PR.
- **Recommendation**: Future PR could add a doc comment on `WaveScreeningCache` explaining that Clone is shallow (no VRAM cost).

### ISSUE 3: Positional indexing via `entries.len()` (MINOR, works correctly)

- **File**: `src/eigensolver/vnl_data.rs:271-274`
- **Description**: In the shared path, per-ion data is fetched via `shared_arc.per_ion_beta_g[entries.len()]`, implicitly depending on the fact that `KptSharedVnl` per-ion vectors and `VnlBatchData.entries` are both USPP-filtered in the same loop order. This works correctly because both are built by the same `precompute_with_d_override` function with the same `continue` logic for non-USPP ions.
- **Recommendation**: Not a bug. A future hardening could add a debug_assert that `entries.len() < shared_arc.per_ion_beta_g.len()` before indexing.

## Summary

All 6 phases implemented correctly. No VnlBatchView. Dead Woodbury code fully excised. KptSharedVnl has all 7 specified fields. Both FFI and standalone SCF paths share spin-independent data via Arc<KptSharedVnl>. grid_buf explicitly dropped after Hamiltonian application. No new unsafe code, no lifetime issues. Compilation and tests pass (pre-existing GPU test failures excluded). WaveScreeningCache Clone footgun noted.

**Verdict: APPROVED WITH MINOR ISSUES** -- all 3 issues are documentation or hardening notes, none blocking.
