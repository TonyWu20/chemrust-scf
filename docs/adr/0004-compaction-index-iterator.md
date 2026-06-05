# ADR-0004: Compaction Index Mapping — Iterator-Based Access

**Date:** 2026-06-05
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

Access to global band indices after compaction goes through a **canonical
`active_bands()` iterator**. Direct arithmetic on `block_start` for band indexing
is banned inside `DavidsonBlockCtx::build()`.

```rust
/// Map compacted workspace column positions to global band indices.
///
/// Yields `(compacted_col, global_band_index)`.  This is the ONLY place
/// that computes `block_start + active_indices[ci]` — all call sites
/// destructure the iterator result.
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

All call sites convert from the fragile pattern:
```rust
// BEFORE (two known bugs in this family):
for i in 0..ncol {
    let global_idx = block_start + active_indices[i];
}
```
to the structural pattern:
```rust
// AFTER (compiler-enforced correctness):
for (ci, gi) in active_bands(&active_indices, block_start) {
    // ci: compacted workspace column — safe for local buffers
    // gi: global band index         — safe for psi_dev, eigenvalues, etc.
}
```

### Accompanying structural changes

1. **`ncol` field eliminated from `DavidsonBlockCtx`.** It was always equal to
   `active_indices.len()` by construction (lines 2002-2003 update both atomically).
   Deriving it from the vector length eliminates a drift risk.

2. **Free function, not a method.** `active_bands()` is a module-level function
   so it serves both `DavidsonBlockCtx::build()` (which has `self.active_indices`)
   and the outer loop in `davidson_diagonalise` (which has a local `active_indices`
   variable).

3. **All outer-loop call sites converted.** Five sites in `davidson_diagonalise`
   that computed `block_start + active_indices[i]` manually were converted to
   the iterator, despite being currently correct — the same pattern of
   "looks fine at the time" preceded both known bugs.

## Consequences

- **Positive:** The compaction-index bug class is structurally eliminated. A new
  call site that needs a global band index must use the iterator; the `(ci, gi)`
  destructure makes it obvious which is which.
- **Positive:** `ncol` and `active_indices` can no longer drift apart.
- **Positive:** The iterator's `#[inline]` on `enumerate().map()` compiles to
  identical machine code as manual indexing — zero runtime cost.
- **Negative:** Minor churn converting existing (correct) call sites.
- **Negative:** The `global_band()` convenience method on `DavidsonBlockCtx` was
  added alongside the iterator but has zero call sites — dead code to clean up.

## Related

- [Memory: Davidson compaction index bug](../../../castep-rust-eigensolve/memory/davidson-compaction-index-bug.md)
- [Debug session: debug-20260605-1220](../../../notes/debug/debug-20260605-1220/)
- CASTEP reference: `hamiltonian.f90:628-646` (compaction), `hamiltonian.f90:392-401` (local slice workspace)
