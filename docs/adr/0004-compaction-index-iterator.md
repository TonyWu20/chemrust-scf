# ADR-0004: Compaction Index Mapping — Iterator-Based Access

**Date:** 2026-06-05
**Revised:** 2026-06-08 (add `block_bands()` for append-only super_wvfn access)
**Status:** Accepted

## Context

The Davidson inner loop compacts unconverged bands to the front of the workspace
after each convergence check (CASTEP `hamiltonian.f90:628-646`). After compaction,
an `active_indices: Vec<usize>` maps compacted workspace column positions to original
global band indices. Global arrays (`psi_dev`, `eigenvalues`, `band_converged`) are
never rearranged — they stay in original band order.

Two instances of the **identical bug class** were found where this mapping was not
applied, using raw sequential indices (`block_start + i`) instead of the mapping
(`block_start + active_indices[i]`):

1. **`previous_eigenvalues` load (fixed in commit `c644514`):** The convergence
   check compared a band's current eigenvalue against a different band's previous
   eigenvalue, causing false convergence/divergence signals.

2. **`build()` psi-dev copy (line 2970, this fix):** The preconditioner copied
   stale wavefunctions from already-converged bands into the active workspace,
   producing corrupt search directions that contaminated the ZHEEVD subspace.
   With `max_outer=2`, this cascaded across SCF iterations — eigenvalues drifted
   until only 1/160 bands converged.

The root cause was structural: every call site that accessed global arrays had to
*manually* remember to compute `block_start + active_indices[i]`. Forgetting this
produced silent corruption that the compiler could not catch.

## Decision

### Two iterators for two data structures

After C3-07 (separate slice workspace, `INNER_LOOP_CHECKLIST_20260606.md` §Component 15),
there are TWO distinct workspace buffers with fundamentally different indexing:

| Buffer | Compacted? | Iterator | Mapping |
|--------|-----------|----------|---------|
| **slice_wvfn** | Yes (compacted after convergence) | `active_bands()` | compacted col `ci` → global band `block_start + active_indices[ci]` |
| **super_wvfn** | No (append-only, never compacted) | `block_bands()` | column `col` → global band `block_start + col` |

**`super_wvfn` is append-only.** C3-07's slice workspace means compaction copies
FROM super_wvfn INTO slice without modifying super_wvfn. ZHEGVD eigenvectors are
always sorted by eigenvalue — position `col` always corresponds to the `col`-th
lowest energy band in the block. This is the natural `1:1` sequential mapping.

### `active_bands()` — for compacted workspace (slice) access

```rust
/// Map compacted workspace column positions to global band indices.
///
/// Yields `(compacted_col, global_band_index)`.  This is the ONLY place
/// that computes `block_start + active_indices[ci]`.
///
/// **Use when:** accessing global arrays (psi_dev, eigenvalues,
/// band_converged) from compacted workspace columns (slice_wvfn).
/// **Do NOT use for:** super_wvfn access — super_wvfn is append-only
/// and column `col` always maps to band `block_start + col` (use
/// block_bands() instead).
#[inline]
fn active_bands(
    active_indices: &[usize],
    block_start: usize,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    active_indices
        .iter()
        .enumerate()
        .map(move |(ci, &orig_idx)| (ci, block_start + orig_idx))
}
```

### `block_bands()` — for append-only (super_wvfn) access

```rust
/// Sequential band iterator for append-only super_wvfn access.
///
/// Yields `(column, global_band_index)` for ALL current_nblock bands
/// using the natural 1:1 mapping: column `col` → band `block_start + col`.
///
/// **Use when:** copying between super_wvfn and psi_dev (Stage 1, A3).
/// super_wvfn is NEVER compacted — ZHEGVD eigenvectors at position col
/// always correspond to the col-th lowest energy band in the block.
/// Sequential mapping is correct regardless of compaction state.
///
/// **Do NOT use for:** compacted workspace (slice) access — use
/// active_bands() which maps through active_indices.
#[inline]
fn block_bands(
    block_start: usize,
    current_nblock: usize,
) -> impl Iterator<Item = (usize, usize)> {
    (0..current_nblock).map(move |col| (col, block_start + col))
}
```

### Call-site conversion rules

For **compacted workspace** (slice, active bands only):
```rust
// BEFORE:
for i in 0..ncol {
    let global_idx = block_start + active_indices[i];
}
// AFTER:
for (ci, gi) in active_bands(&active_indices, block_start) {
    // ci: compacted workspace column — safe for slice indexing
    // gi: global band index         — safe for psi_dev, eigenvalues, etc.
}
```

For **append-only superspace** (super_wvfn, all current_nblock bands):
```rust
// BEFORE:
for i in 0..current_nblock {
    let gi = block_start + i;
}
// AFTER:
for (col, gi) in block_bands(block_start, current_nblock) {
    // col: super_wvfn column — safe for super_wvfn/h_super_wvfn indexing
    // gi: global band index   — safe for psi_dev, eigenvalues, etc.
}
```

### Accompanying structural changes

1. **`ncol` field eliminated from `DavidsonBlockCtx`.** It was always equal to
   `active_indices.len()` by construction. Deriving it from the vector length
   eliminates a drift risk.

2. **Free functions, not methods.** Both `active_bands()` and `block_bands()` are
   module-level functions so they serve both `DavidsonBlockCtx::build()` and the
   outer loop in `davidson_diagonalise`.

3. **All outer-loop call sites converted.** Sites accessing compacted data use
   `active_bands()`; sites accessing append-only super_wvfn use `block_bands()`.

## Regression: 2026-06-08 (workflow audit commit `fce7fae`)

A 5-agent adversarial workflow incorrectly applied `active_bands()` to Stage 1 and
A3 — both of which copy between super_wvfn (append-only) and psi_dev (global). The
correct iterator is `block_bands()` because super_wvfn is never compacted.

**Symptoms:**
- Block 130 eigenvalue explosion to -10^52 Ha (D10-01 H_sub discrepancy: 3×10^53)
- Bands 97–116 spuriously un-converged in outer iteration 1 (eigenvalue changes up
  to 0.012 Ha vs correct values)
- Convergence degraded from clean 160/160 in 2 iterations to full numerical collapse

**Root cause:** `active_bands()` maps through `active_indices` — after compaction
the first active band may be at `active_indices[0] = 7`. `active_bands()` writes
ZHEGVD eigenvector 0 (lowest energy, for band 104) to global band
`block_start + 7 = 111`, scrambling the eigenvalue→band mapping. This corrupts
`psi_dev` and `eigenvalues`, contaminating subsequent blocks via conduction state
seeding and causing cascading eigenvalue explosion.

**Lesson:** CASTEP-faithful implementation decisions (sequential A3, append-only
super_wvfn) recorded in `INNER_LOOP_CHECKLIST_20260606.md` as VERIFIED/FIXED must
NEVER be reverted. Future automation/agents must treat that checklist as an
authoritative constraint — "this matches CASTEP" is a permanent invariant, not a
negotiable design choice.

## Consequences

- **Positive:** The compaction-index bug class is structurally eliminated.
- **Positive:** `ncol` and `active_indices` can no longer drift apart.
- **Positive:** The two-iterator design (`active_bands` vs `block_bands`) makes
  the data structure semantics explicit at every call site — `(ci, gi)` signals
  "compacted workspace → global", `(col, gi)` signals "append-only superspace → global".
- **Positive:** Zero runtime cost — `#[inline]` on `Iterator::map()` compiles to
  identical machine code as manual indexing.
- **Negative:** Applying the wrong iterator (e.g., `active_bands` to super_wvfn)
  causes catastrophic eigenvalue explosion. The structural pattern eliminates
  *forgetting* the mapping but doesn't prevent *choosing the wrong mapping*.
  Mitigation: checklist explicitly records which iterator each call site uses.

## Related

- [Memory: Davidson compaction index bug](../../../castep-rust-eigensolve/memory/davidson-compaction-index-bug.md)
- [Checklist: INNER_LOOP_CHECKLIST_20260606.md](../../notes/audit/INNER_LOOP_CHECKLIST_20260606.md)
- C3-07 slice workspace: `hamiltonian.f90:628-646` (compaction to separate slice)
- CASTEP reference: `hamiltonian.f90:523-528` (A3 sequential copy), `hamiltonian.f90:407-408` (Stage 1 sequential copy)
