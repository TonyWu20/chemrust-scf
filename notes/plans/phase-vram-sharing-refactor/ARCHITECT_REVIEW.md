# VRAM Sharing Refactor — Architect Review of TASKS.md

**Date**: 2026-06-14
**Reviewer**: Rust architect
**Reviewed**: [TASKS.md](./TASKS.md) against [PLAN.md](./PLAN.md) and source tree

## Source verification summary

Line counts and locations verified against the live tree:

| File | Actual lines | TASKS claim | Status |
|------|-------------|-------------|--------|
| `src/eigensolver/vnl_data.rs` | 562 | 562 | Match |
| `src/ffi.rs` | 748 | ~449, ~703, ~708 | Line-accurate |
| `src/scf.rs` | 3042 | ~786, ~856 | Line-accurate |

**Dead Woodbury fields confirmed unreferenced outside vnl_data.rs**: `b_concat`,
`lu_m`, `lu_ipiv`, `n_total_expanded` appear only in the struct definition
(lines 47-53), the assembly block (lines 369-494), and the return expression
(lines 491-494). No external reader exists — removal is safe.

**Collector vectors** at lines 193-195 (`per_ion_q`, `per_ion_beta_flat`,
`per_ion_ne`) and their push calls at lines 355-358 are correctly identified.
The Woodbury assembly block spans lines 369-494 (125 lines).

**`KptSharedVnl` and `shared_vnl` do not exist anywhere in the codebase** —
confirmed zero hits for both symbols. These are genuine additions.

**`grid_buf` references**: allocation at line 614, last use at line 698
(`.grid_dev(&mut grid_buf)`). The unsafe block containing
`apply_full_hamiltonian` closes at line 705. D2H transfers at lines 708-711.
Drop point between 705 and 708 is valid — no borrow extends past line 705.

## 1. Dependency ordering

The dependency graph is correct:

```
Group 1 (dead code removal) ──┐
                              ├── Group 2 (KptSharedVnl + signature change) ──┬── Group 4 (FFI path) ──┐
Group 3 (grid_buf drop) ──────┘                                              └── Group 5 (SCF path)  ──┤
                                                                                                       ├── Group 6 (test)
```

- **Group 1 before Group 2**: Correct. Group 1 removes dead fields from the
  same struct and function body that Group 2 modifies. Reversing would create
  merge conflicts on the return expression.
- **Group 2 before Groups 4/5**: Correct. Groups 4 and 5 both require the
  `shared: Option<Arc<KptSharedVnl>>` parameter that Group 2 adds to
  `precompute_with_d_override`.
- **Group 3 independent**: Correct. The `drop(grid_buf)` insertion at line
  ~705 does not semantically conflict with any other group's changes.
- **Group 4 parallel with Group 5**: Correct. They touch different files
  (ffi.rs vs scf.rs) with no shared dependency beyond Group 2.

**Minor caveat**: Group 3 and Group 2 both touch `ffi.rs` (different line
ranges: ~449 vs ~705). If Group 2 adds/removes lines before line 705, the
Group 3 insertion point shifts. This is standard rebase mechanics — the
ordering in the dependency graph correctly shows they are independent, and
the implementer will rebase whichever lands second. This is not a dependency
error; it is a scheduling note.

## 2. Group sizing

| Group | Est. lines changed | Concerns touched | Verdict |
|-------|--------------------|------------------|---------|
| 1 | -130 | Dead Woodbury removal only | Correct |
| 2 | +80 / -10 | New type, signature change, 3 files touched | **Correct, but note** |
| 3 | +1 | Single line insertion | Correct |
| 4 | +15 | Arc wiring in ffi.rs | Correct |
| 5 | +10 | Same pattern in scf.rs | Correct |
| 6 | 0 | Verification only | Correct |

**Group 2 is the largest but cannot be split further without breaking the
"independently compilable" rule.** The struct definition change, the
shared-aware logic in `precompute_with_d_override`, the `rescreen_d` update,
and the `precompute` wrapper update form a single atomic change. If you added
the `KptSharedVnl` struct without the shared-aware logic, `rescreen_d` would
fail to compile (it references `self.shared.screening_cache` which would not
exist yet). If you updated `rescreen_d` without adding the `shared` field to
`VnlBatchData`, you'd have an incomplete struct. The group is correctly sized
as a single unit of work.

## 3. Acceptance criteria

| Group | Criterion | Adequacy |
|-------|-----------|----------|
| 1 | `cargo check` | Acceptable. Dead code removal — if it compiles, nothing references the removed items. |
| 2 | `cargo check` | Acceptable for an intermediate group. The sharing mechanism exists but is never engaged. |
| 3 | `cargo check` + placement verification | Acceptable. Trivial insertion. |
| 4 | `cargo check` | **Weak but acceptable.** No behavioral test at this stage. Group 6 covers it. |
| 5 | `cargo check` | Same as Group 4. |
| 6 | `cargo test --release` + FFI convergence | Strong. Tests the full pipeline. |

**Assessment**: Groups 1-5 rely on `cargo check` as their sole gate. This is
acceptable because (a) the structural changes (field removal, signature
change) are compile-time errors if wrong, and (b) Group 6 gates the behavioral
correctness. However, Groups 4 and 5 introduce logic changes that `cargo check`
cannot validate: the lazy-init path and the shared-reuse path. A discriminator
test after Group 4 (non-spin NiO FFI cold-start) would strengthen confidence
before proceeding to Group 5. This is a recommendation, not a defect — the
current criteria are sufficient for an experienced implementer.

## 4. Missing tasks

### 4.1 (Minor) Test callers of `precompute` need solver argument dropped

Group 2.8 says to "Remove unused `solver` parameter from `precompute_with_d_override`
and `precompute`." The `precompute` wrapper is called at three locations in scf.rs
for test-only functions:

- Line 1059 (`diagonalize_with_rr_matrices`)
- Line 1130 (`apply_h_components_for_test`)
- Line 1188 (`apply_s_for_test`)

Each currently passes `&solver` as the last argument. These callers are not
explicitly listed in Group 2's "What to change" section. They will fail `cargo check`
after the solver parameter is removed. The implementer will naturally discover
these through the compiler error, so this is a specification gap, not a logic
error. **Recommendation**: add a bullet under Group 2.8 listing these three lines.

### 4.2 (Minor) KptSharedVnl construction needs collector vectors

Group 2.5 says "After the per-ion loop: construct `KptSharedVnl`, clone per-ion
GPU slices into it." TASKS does not explicitly say to collect `beta_dev` and
`q_dev` CudaSlices during the per-ion loop (for the fresh-build path). The old
collector vectors (`per_ion_beta_flat`, `per_ion_q`) are removed by Group 1, and
replacement collectors are needed for the `Arc<KptSharedVnl>` construction.

This is an implicit requirement — an experienced implementer will add collector
vectors when they reach the "construct KptSharedVnl after the loop" step. The
task description is sufficient for a competent executor; the omission is in
the explicit listing, not in the logical completeness.

### 4.3 (Trivial) File count in Group 2 header

TASKS says "Files touched: 2 (+ 1 trivial wrapper touch)". The actual files are:
`src/eigensolver/vnl_data.rs`, `src/ffi.rs`, `src/scf.rs` — that is 3, not 2.
The `precompute` wrapper is within `vnl_data.rs`, not a separate file.
Correct the header to read "3".

### 4.4 (Observation) Plan step 12 placement inconsistency

Plan Phase 3 step 12 says "Update `rescreen_d` to access `self.shared.screening_cache`
instead of `self.screening_cache`." TASKS correctly moves this into Group 2
(step 6), since the fields are removed from `VnlBatchData` in Group 2 and
`rescreen_d` lives on `VnlBatchData`. The Plan's placement in Phase 3 was
incorrect — `rescreen_d` is not related to KptData or the FFI wiring. TASKS
improves on the Plan here.

## 5. Group independence

Each group is independently compilable after its changes are applied, provided
the implementation is correct:

| Group | Compiles standalone? | Reasoning |
|-------|---------------------|-----------|
| 1 | Yes | Removes dead code only. No caller breaks. |
| 2 | Yes | All syntactic consumers (callers of changed signatures) are updated within the group. The 3 `precompute` test callers must also drop `solver` (see 4.1). |
| 3 | Yes | Single `drop()` insertion. No signature changes. |
| 4 | Yes | Uses types and signatures from Group 2. Adds Arc field to existing struct. |
| 5 | Yes | Same logic as Group 4, different file. |
| 6 | Yes | Verification only — runs the compiled artifact. |

**Verification note for Group 4 and 5 independence**: Each group wires the
lazy-init pattern independently in its own file (ffi.rs vs scf.rs). They do
not share state. The FFI path uses `KptData.shared_vnl` per kpt; the
standalone SCF path uses a local `shared_vnl_cache` vector. Both patterns are
equivalent but mechanically independent — a bug in one does not block
compilation of the other.

## Summary

**Verdict: APPROVED with minor recommendations.**

The task decomposition faithfully maps the Plan to executable groups with
correct dependency ordering and independently compilable boundaries. The two
minor gaps (solver removal not listing test callers explicitly, KptSharedVnl
collector vectors implicit) are specification gaps that the compiler or an
experienced implementer will catch immediately — they are not logic errors.

### Recommendations for the implementation executor

1. When removing the `solver` parameter in Group 2.8, also update the three
   `VnlBatchData::precompute(...)` call sites at scf.rs lines 1059, 1130, 1188
   (drop the `&solver` last argument).

2. When implementing the shared-aware logic in Group 2.5 (fresh-build path),
   add collector vectors for `Vec<CudaSlice<CudaComplex>>` (beta_dev per ion)
   and `Vec<CudaSlice<CudaComplex>>` (q_dev per ion) before the per-ion loop,
   push after upload, and consume when constructing `KptSharedVnl`.

3. Group 2 header: change "Files touched: 2 (+ 1 trivial wrapper touch)" to
   "Files touched: 3".
