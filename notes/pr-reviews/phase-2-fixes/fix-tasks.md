# Fix Tasks: Phase 2 Fixes Review

## Task F1: Use `into_phase()` in `diagonalize()`

**Kind:** direct
**File changes:**
- `src/scf.rs` — replace manual struct literal in `diagonalize()` with `self.into_phase()` pattern

**Changes:**
1. Replace the manual `Ok(ScfIteration { ... })` struct literal at lines 352-363 with:
   ```rust
   let mut next: ScfIteration<S, WavefunctionsUpdated> = self.into_phase();
   next.psi = psi_new;
   next.eigenvalues = eigenvalues;
   Ok(next)
   ```

**Rationale:** Every other phase transition (`build_v_eff`, `construct_density`) uses `into_phase()`. The `diagonalize()` method constructs the full struct literal manually, so adding a new field to `ScfIteration` requires updating 3 sites instead of 1. This is a maintenance hazard that defeats the purpose of the `into_phase()` helper.

**Success Criteria:**
- `cargo check` passes
- `diagonalize()` return type is unchanged (`ScfIteration<S, WavefunctionsUpdated>`)
