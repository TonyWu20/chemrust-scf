# Forensic TASKS.md — phase-global-woodbury

**Branch base:** `feat/phase-rchfsi` @ `6bd2c58`
**Plan source:** `notes/plans/phase-global-woodbury/PHASE_PLAN.md`
**ODD pattern ref:** `~/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

## Declared Fixtures

| Fixture | Path | Used by |
|---------|------|---------|
| Cu(111)+CO CASTEP `.check` | `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/` | G0, G3, G7 |
| CASTEP `.bands` band-1 reference | `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands` | G7 |
| CASTEP F8 instrumented run | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/` | G7 |

## Exploration Notes

- `vnl_data.rs:320` typo confirmed: `m_inv.iter()` uploads the Gauss-Jordan working
  copy (reduced to identity) instead of `s_inv` (holds M⁻¹). Both are `Vec<f64>` —
  silent at compile time.
- `apply_s_inverse` at `chebyshev.rs:821` carries `#[allow(dead_code)]`. Only call
  site is the diagnostic `check_s_inv_s_identity` at line 1068 (n_bands=1). Not
  wired into Lanczos, filter recurrence, or Step-4 reconstruction.
- `SolverHandle` is already constructed at `scf.rs:435`, before both `precompute`
  calls (lines 493, 604) and `chebyshev_filter` (line 512). Threading it is
  mechanical — one new parameter per function in the chain.
- `apply_s_times` at `chebyshev.rs:914` is the structural mirror of `apply_s_inverse`
  (same per-ion GEMM pattern, uses `q_matrix` instead of `s_inv_mat`). Use as
  correctness reference when writing the global Woodbury body.
- `solver.rs` `zhegvd` (lines 47–95) is the exact template for cuSOLVER wrappers:
  `DnHandle`, `SolverError`, buffer-size query, `alloc_zeros` workspace,
  `device_ptr_mut` cast to `*mut sys::cuDoubleComplex`, `.result()?`.
- Switched from Cholesky (`zpotrf`/`zpotrs`) to LU (`zgetrf`/`zgetrs`) for the global
  Woodbury M factorization. M = Q⁻¹ + B^H·B is not guaranteed positive-definite
  because Q⁻¹ has zero rows for projector channels without Q_aug (e.g. Cu d-channels).
  `zpotrf` fails with `info = 2` on the Cu(111)+CO fixture; `zgetrf` handles it.
  See `DECISIONS.md` for the full analysis.
- `Cargo.toml` has no `[features]` section — must be added for `scf_diag`.
- 11 `eprintln!` sites in `chebyshev.rs` at lines 439, 440, 443, 513, 514, 529,
  1354, 1374, 1377, 1396, 1614. 6 in `scf.rs` at lines 469, 536, 687, 704, 708, 758.

---

## Group G0 — Typo fix + baseline rebaseline

**kind:** direct  
**depends on:** nothing  
**blocks:** G2

### TASK-G0-1: Fix `m_inv` → `s_inv` typo with newtype enforcement

**File:** `src/eigensolver/vnl_data.rs`

The `m_inv`/`s_inv` confusion is a silent bug because both are `Vec<f64>`. Fix with
newtypes + a dedicated conversion function so the wrong variable is a compile error:

```rust
struct GjWorkingCopy(Vec<f64>);  // m_inv — row-reduced to identity after GJ
struct GjResult(Vec<f64>);       // s_inv — holds M⁻¹ after GJ

fn gj_result_to_cuda(r: GjResult) -> Vec<CudaComplex> {
    r.0.into_iter().map(|x| CudaComplex { x, y: 0.0 }).collect()
}
```

Changes:
1. Declare `GjWorkingCopy` and `GjResult` newtypes (private to module).
2. Declare `gj_result_to_cuda` conversion function.
3. In the M-inversion block (lines 284–318): rename `m_inv` → `GjWorkingCopy(m_mat.clone())`,
   rename `s_inv` → `GjResult(identity)`. Adjust all indexing to use `.0[...]`.
4. Replace the upload at line 320:
   - Before: `m_inv.iter().map(|&x| CudaComplex { x, y: 0.0 }).collect()`
   - After: `gj_result_to_cuda(s_inv)` — passing `m_inv` (now `GjWorkingCopy`) is a type error.

**Acceptance:** `cargo check --workspace` green.

---

### TASK-G0-2: Exploratory baseline measurement (pre-commit)

**Do not commit yet.** With the typo fix applied, run:

```bash
cargo test --release -p chemrust-scf -- --ignored s_inv_s_identity_test 2>&1 | grep 'S⁻¹·S identity'
```

Record the printed `‖S⁻¹·S·ψ₀ − ψ₀‖_∞` value. This is the true per-ion baseline ζ.

**Decision gate (from PHASE_PLAN.md Goal 0):**
- If ζ < 1e-10: phase reframes from "correctness fix" to "perf optimization".
  Demote G7's iter-3 explosion gate to a regression check. Document inline.
- If ζ ≥ 1e-10: proceed as planned.

---

### TASK-G0-3: Lock baseline + add rebaseline test

**File:** `tests/ca_scf_convergence.rs`

After recording ζ from TASK-G0-2:

1. Add near the top of the test file:
   ```rust
   /// True per-ion Woodbury baseline after m_inv→s_inv typo fix.
   /// Measured on Cu(111)+CO fixture. Becomes historical after G3 (global Woodbury).
   const BASELINE_ZETA_PER_ION: f64 = <measured value>;
   ```

2. Add test:
   ```rust
   #[test]
   #[ignore = "requires GPU and CASTEP fixture data"]
   fn s_inv_baseline_post_typo_fix() {
       // ... same setup as s_inv_s_identity_test ...
       let zeta = check_s_inv_s_identity(&band0, n_pw, &vnl_data, &blas, &stream)
           .expect("check_s_inv_s_identity");
       eprintln!("[baseline] per-ion ζ = {:.6e}  (BASELINE_ZETA_PER_ION = {:.6e})",
                 zeta, BASELINE_ZETA_PER_ION);
       // Assertion: call succeeded. Value is the discriminator, not a threshold.
   }
   ```

**Note:** `s_inv_s_identity_test` threshold stays at `< 1e-6` in this commit.
Tightening to `< 1e-10` belongs in G3 (discriminator for global Woodbury).

**Acceptance:** `cargo check --workspace` green; new test compiles.

**Commit:** `fix(vnl-data): m_inv→s_inv typo; GjWorkingCopy/GjResult newtypes; baseline lock`

---

## Group G1 — cuSOLVER LU bindings (global Woodbury factorisation)

**kind:** direct  
**depends on:** nothing  
**blocks:** G2

**Decision context:** The global Woodbury matrix M = Q⁻¹ + B^H·B is not
guaranteed positive-definite because Q⁻¹ has zero rows for projector channels
where Q_aug has no entries (e.g. Cu d-channels). Cholesky (`zpotrf`) fails
with `info = 2` on the Cu(111)+CO fixture. Switched to LU with partial
pivoting (`zgetrf`/`zgetrs`), which handles any full-rank M. A vestigial
`CHOL_REG = 1e-12` is kept as a floor for exact singularity, but no larger
regularisation is needed. See `DECISIONS.md` for the full analysis.

### TASK-G1-1: Add `zgetrf` / `zgetrs` methods to `SolverHandle`

**File:** `src/device/solver.rs` (append after line 96)

Template: `zpotrf` method (same `DnHandle`, `SolverError`,
`device_ptr_mut` cast, `alloc_zeros` workspace, `.result()?` pattern).

```rust
pub fn zgetrf(
    &self,
    m: i32,
    n: i32,
    a: &mut CudaSlice<CudaComplex>,
    ipiv: &mut CudaSlice<i32>,
    info: &mut CudaSlice<i32>,
) -> Result<(), SolverError>
```
- Query buffer size via `sys::cusolverDnZgetrf_bufferSize`
- Allocate workspace via `self.stream.alloc_zeros::<CudaComplex>(lwork)`
- Call `sys::cusolverDnZgetrf` — factors A = P·L·U, stores L (unit lower)
  and U in `a`, pivot indices in `ipiv`

```rust
pub fn zgetrs(
    &self,
    trans: cublasOperation_t,
    n: i32,
    nrhs: i32,
    a: &CudaSlice<CudaComplex>,   // LU factor (read-only)
    ipiv: &CudaSlice<i32>,         // pivot indices from zgetrf
    b: &mut CudaSlice<CudaComplex>, // RHS in, solution out
    info: &mut CudaSlice<i32>,
) -> Result<(), SolverError>
```
- `zgetrs` does not need a workspace buffer
- Call `sys::cusolverDnZgetrs` directly

### TASK-G1-2: Tests for LU bindings

**File:** `src/device/solver.rs` `#[cfg(test)]` block

```rust
#[test]
fn zgetrs_round_trip() {
    // 4×4 full-rank matrix A, RHS b = A·x_exact, LU solve → x, assert ‖x − x_exact‖ < 1e-12
}

#[test]
fn zgetrs_multi_rhs() {
    // 16×16 matrix, 8 RHS columns, assert max err < 1e-12
}
```

**Acceptance:** `cargo test --release -p chemrust-scf -- zgetrs_round_trip zgetrs_multi_rhs`

**Commit:** `feat(solver): zgetrf/zgetrs cuSOLVER bindings with round-trip tests`

---

## Group G2 — Global Woodbury precompute in `VnlBatchData`

**kind:** direct  
**depends on:** G0, G1  
**blocks:** G3

### TASK-G2-1: Extend `VnlBatchData` struct and drop `s_inv_mat`

**File:** `src/eigensolver/vnl_data.rs`

`VnlBatchData` struct (around line 28): add fields:
```rust
pub b_concat: CudaSlice<CudaComplex>,      // n_pw × n_total_expanded, col-major
pub lu_m: CudaSlice<CudaComplex>,           // LU factor of M = Q⁻¹ + B^H·B
pub lu_ipiv: CudaSlice<i32>,               // pivot indices from zgetrf
pub n_total_expanded: i32,
```

`VnlIonData` struct (around line 19): remove `s_inv_mat` field.

### TASK-G2-2: CPU-side matrix pipeline newtypes

**File:** `src/eigensolver/vnl_data.rs`

Add private newtypes for the new CPU-side matrix pipeline (same module as
`GjWorkingCopy`/`GjResult` from G0):

```rust
struct QInvBlkDiag(Vec<f64>);  // block-diagonal Q⁻¹ assembled from per-ion inverses
struct BhB(Vec<f64>);          // B^H·B cross-ion blocks (GPU gemm → CPU copy)
struct MMatrix(Vec<f64>);      // M = Q⁻¹ + B^H·B, consumed by LU (upload then drop)
```

Add conversion functions:
```rust
fn q_inv_blkdiag_to_cuda(q: &QInvBlkDiag) -> Vec<CudaComplex> { ... }
fn m_matrix_to_cuda(m: MMatrix) -> Vec<CudaComplex> { ... }
```

### TASK-G2-3: Update `precompute` — add `SolverHandle`, build global Woodbury

**File:** `src/eigensolver/vnl_data.rs:97`

Signature change: add `solver: &SolverHandle` parameter.

After the per-ion loop, add:

1. **Assemble `QInvBlkDiag`**: collect per-ion Q⁻¹ blocks (already computed in the
   loop as `inv` from the Gauss-Jordan) into a block-diagonal `Vec<f64>` of shape
   `n_total_expanded × n_total_expanded`.

2. **Concatenate B**: collect all per-ion `beta_g` CPU-side into a flat
   `Vec<CudaComplex>` of shape `n_pw × n_total_expanded` (column-major, ions
   concatenated along the column axis). Upload → `b_concat`. Track H2D bytes in
   `pcie`.

3. **Compute `B^H·B`** via one `cublasZgemm_v2` call:
   - `transa = C` (conjugate transpose), `transb = N`
   - `m = n_total_expanded`, `n = n_total_expanded`, `k = n_pw`
   - Result: `n_total_expanded × n_total_expanded` Hermitian matrix on GPU
   - Copy result to CPU → `BhB`

4. **Build `MMatrix`**: `M[i,j] = QInvBlkDiag[i,j] + BhB[i,j]` (element-wise add;
   off-diagonal blocks of `QInvBlkDiag` are zero, so this just adds the cross-ion
   terms from `BhB`).

5. **LU factor**: upload `MMatrix` to GPU → `m_dev`; call `solver.zgetrf`
   (m = n_total_expanded, n = n_total_expanded) → `lu_m`, `lu_ipiv`.
   Check `info[0] == 0`; if not, return `Err(...)` with a message indicating
   singular M. Add vestigial `1e-12` diagonal regularization as a floor for
   exact rank deficiency (not needed for Cu(111)+CO but harmless and guards
   against edge cases). Drop CPU `MMatrix` after upload.

6. Remove the per-ion `s_inv_mat` upload (lines 320–324).

**Update call sites** in `scf.rs`:
- Line 493: add `&solver` argument
- Line 604: add `&solver` argument

**Acceptance:** `cargo check --workspace` green.

**Commit:** `feat(vnl-data): global Woodbury precompute — B_concat, LU M, SolverHandle`

---

## Group G3 — Replace `apply_s_inverse` body + tighten identity gate

**kind:** direct  
**depends on:** G2  
**blocks:** G4, G5

### TASK-G3-1: Update `apply_s_inverse` signature and body

**File:** `src/eigensolver/chebyshev.rs:821–900`

Signature change: add `solver: &SolverHandle` parameter. Remove `#[allow(dead_code)]`.

Replace the per-ion loop body with global Woodbury:

```
1. t = B^H · v
   cublasZgemm_v2: transa=C, transb=N
   m=n_total_expanded, n=n_bands, k=n_pw
   A=b_concat, B=hpsi_dev → t (n_total_expanded × n_bands)

2. solver.zgetrs(nte, n_bands, &lu_m, &lu_ipiv, &mut t, &mut info)
   Solves the LU system P·L·U · t = rhs in-place

3. v -= B · t
   cublasZgemm_v2: transa=N, transb=N, alpha=-1, beta=1
   m=n_pw, n=n_bands, k=n_total_expanded
   A=b_concat, B=t → hpsi_dev (accumulate)
```

Access `vnl_data.b_concat`, `vnl_data.lu_m`, `vnl_data.lu_ipiv`, `vnl_data.n_total_expanded`.

### TASK-G3-2: Thread `SolverHandle` to `check_s_inv_s_identity`

**File:** `src/eigensolver/chebyshev.rs:1001`

Add `solver: &SolverHandle` to `check_s_inv_s_identity` signature. Pass it through
to the `apply_s_inverse` call at line 1068.

Update the test at `tests/ca_scf_convergence.rs:696`:
- Construct `SolverHandle::new(stream.clone())` before calling `check_s_inv_s_identity`
- Pass `&solver`

### TASK-G3-3: Tighten identity gate threshold

**File:** `tests/ca_scf_convergence.rs:703`

Change `max_residual < 1e-6` → `max_residual < 1e-10`.

This is the discriminator for "global Woodbury actually inverts S to roundoff."
Only valid after G0 + G2 + G3 all land.

**Acceptance:**
- `cargo check --workspace` green
- `cargo test --release -p chemrust-scf -- --ignored s_inv_s_identity_test`
  asserts `< 1e-10`

**Commit:** `feat(chebyshev): global Woodbury apply_s_inverse; tighten identity gate to 1e-10`

---

## Group G4 — Wire `apply_s_inverse` into Lanczos

**kind:** direct  
**depends on:** G3  
**blocks:** G7

### TASK-G4-1: Thread `SolverHandle` through filter chain + insert Lanczos call

**Files:** `src/eigensolver/chebyshev.rs`, `src/scf.rs`

Signature changes (add `solver: &SolverHandle` to each):
- `lanczos_upper_bound` at `chebyshev.rs:394`
- `chebyshev_filter` at `chebyshev.rs:1252`

Call site updates:
- `scf.rs:512`: pass `&solver` to `chebyshev_filter`

Body change in `lanczos_upper_bound` (after `apply_full_hamiltonian` at line 461):
```rust
apply_s_inverse(&mut hv, vnl_data, 1, n_pw as i32, blas, solver, stream)?;
```

**Gate:** Add a temporary `eprintln!` (or gate behind `scf_diag` if G6 lands first)
logging `b_up` before and after the change. Expected: `b_up` decreases for
Cu(111)+CO (S − I is positive-semidefinite for USPP).

**Acceptance:** `cargo check --workspace` green.

**Commit:** `feat(chebyshev): wire apply_s_inverse into lanczos_upper_bound (Goal 1b)`

---

## Group G5 — Filter recurrence A/B gate

**kind:** direct  
**depends on:** G3  
**blocks:** G7

### TASK-G5-1: Add `use_sinv_filter` flag to filter recurrence

**File:** `src/eigensolver/chebyshev.rs`

Add `use_sinv_filter: bool` parameter to `chebyshev_filter` (line 1252). Thread it
to the R-ChFSI recurrence body (around line 1402).

When `use_sinv_filter = true`, insert after each `apply_full_hamiltonian` in the
recurrence:
```rust
if use_sinv_filter {
    apply_s_inverse(&mut hpsi_dev, vnl_data, n_bands_i32, n_pw_i32, blas, solver, stream)?;
}
```

Default call sites pass `use_sinv_filter: false` — no behaviour change.

**Decision criterion (pre-locked, PHASE_PLAN.md Goal 1c):**
- Keep bare-H if iter-3 D_screened amax (S⁻¹·H filter) − bare-H baseline < 5 Ha
  AND iter-1 band-1 |Δ| does not regress > 0.005 Ha.
- Flip default to S⁻¹·H otherwise.

**Acceptance:** `cargo check --workspace` green; both `use_sinv_filter` paths compile.

**Commit:** `feat(chebyshev): use_sinv_filter A/B gate for filter recurrence (Goal 1c)`

---

## Group G6 — `scf_diag` feature gate

**kind:** direct  
**depends on:** nothing (independent)

### TASK-G6-1: Add `[features]` to `Cargo.toml` and gate all diagnostic prints

**Files:** `Cargo.toml`, `src/eigensolver/chebyshev.rs`, `src/scf.rs`

1. `Cargo.toml`: add after `[dependencies]`:
   ```toml
   [features]
   scf_diag = []
   ```

2. Wrap all 17 `eprintln!` sites with `#[cfg(feature = "scf_diag")]`:
   - `chebyshev.rs` lines: 439, 440, 443, 513, 514, 529, 1354, 1374, 1377, 1396, 1614
   - `scf.rs` lines: 469, 536, 687, 704, 708, 758

**Acceptance:**
- `cargo check --workspace` — clean (no diagnostic noise in release)
- `cargo check --workspace --features scf_diag` — also green

**Commit:** `feat(diag): gate 17 eprintln sites behind scf_diag feature (Goal 5)`

---

## Group G7 — Integration gates

**kind:** direct  
**depends on:** G0, G3, G4, G5

### TASK-G7-1: Iter-3 explosion gate + A/B filter test + wall-time gate

**File:** `tests/ca_scf_convergence.rs`

1. **Extend `fixed_point_matches_castep_energy`** (line 53) to depth 3. Add asserts:
   ```rust
   assert!(iter3_d_screened_amax < 10.0,
       "iter-3 Cu D_screened amax = {:.2} Ha ≥ 10 Ha (explosion gate)", ...);
   assert!((iter3_band1 - (-1.05502287_f64)).abs() < 0.05,
       "iter-3 band-1 = {:.4} Ha, |Δ| ≥ 0.05 Ha", ...);
   ```
   Band-1 reference: −1.05502287 Ha from
   `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands`.

2. **Add wall-time gate** `iter2_apply_s_inverse_wall`:
   ```rust
   let t0 = std::time::Instant::now();
   // ... iter-2 apply_s_inverse call ...
   let wall = t0.elapsed().as_secs_f64();
   assert!(wall <= per_ion_baseline_wall,
       "apply_s_inverse wall {:.3}s > per-ion baseline {:.3}s", wall, per_ion_baseline_wall);
   ```

3. **Add A/B filter test** `fixed_point_matches_castep_energy_filter_with_sinv`:
   Same as `fixed_point_matches_castep_energy` but pass `use_sinv_filter: true`.
   Log iter-3 D_screened amax for the Goal 1c decision criterion. No hard assert on
   the value — the decision criterion is applied manually per PHASE_PLAN.md Goal 1c.

**Acceptance:** All new tests compile; `cargo test --release -p chemrust-scf --
--ignored fixed_point_matches_castep_energy` runs to depth 3.

**Commit:** `feat(tests): iter-3 explosion gate + A/B filter test + wall-time gate (Goals 4/1c/6)`

---

## Discriminator Stack (cheap → expensive)

| Gate | Pre-fix | Target | Task |
|------|--------:|-------:|------|
| `cargo check --workspace` | green | green | all |
| `cargo clippy --workspace -- -D warnings` | green | green | all |
| `zgetrs_round_trip` | not present | err < 1e-12 | G1 |
| `zgetrs_multi_rhs` | not present | err < 1e-12 | G1 |
| `s_inv_baseline_post_typo_fix` | 0.014 (contaminated) | **measured, recorded** | G0 |
| `s_inv_s_identity_test` ‖S⁻¹·S·ψ−ψ‖_∞ | `< 1e-6` | `< 1e-10` | G3 |
| Lanczos `b_up` delta (iter-1) | n/a | logged, sign matches expected | G4 |
| iter-1 band-1 vs CASTEP (Ha) | Δ ≈ 0.025 | Δ < 0.005 | G7 |
| iter-2 V_eff range (Ha) | 26.06 (ndeg=4) | < 9.5 Ha | G7 |
| **iter-3 Cu D_screened amax (bare-H filter)** | 362 (ndeg=4) | **< 10 Ha** | G7 |
| **iter-3 band-1 vs CASTEP (Ha)** | −16.74 (ndeg=4) | **\|Δ\| < 0.05** | G7 |
| iter-3 D_screened amax (S⁻¹·H filter, A/B) | n/a | logged; decision per Goal 1c | G7 |
| iter-2 `apply_s_inverse` wall | per-ion baseline | ≤ baseline | G7 |

## SolverHandle threading map

`SolverHandle` constructed at `scf.rs:435`. Full chain:

| Function | File:line | Change |
|----------|-----------|--------|
| `VnlBatchData::precompute` | `vnl_data.rs:97` | add `solver: &SolverHandle` |
| `chebyshev_filter` | `chebyshev.rs:1252` | add `solver: &SolverHandle` |
| `lanczos_upper_bound` | `chebyshev.rs:394` | add `solver: &SolverHandle` |
| `check_s_inv_s_identity` | `chebyshev.rs:1001` | add `solver: &SolverHandle` |
| `apply_s_inverse` | `chebyshev.rs:821` | add `solver: &SolverHandle` |
| `scf.rs:493` | `scf.rs` | pass `&solver` to `precompute` |
| `scf.rs:604` | `scf.rs` | pass `&solver` to `precompute` |
| `scf.rs:512` | `scf.rs` | pass `&solver` to `chebyshev_filter` |
| `tests/ca_scf_convergence.rs:696` | tests | construct + pass `SolverHandle` |

## Dependency graph

```
G0 (typo fix) ────────────────┐
                               ├──> G2 (VnlBatchData + SolverHandle param)
G1 (LU bindings) ────────────────────┘         └──> G3 (apply_s_inverse body + 1e-10)
                                                ├──> G4 (Lanczos wiring)
                                                └──> G5 (filter A/B gate)
G6 (scf_diag) — independent
G7 (integration gates) — after G0 + G3 + G4 + G5
```
