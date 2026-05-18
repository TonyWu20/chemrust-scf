# Review: Phase 1 — Type-State SCF Backbone

**Tasks**: notes/plans/phase-1/TASKS.md
**Reviewed**: 2026-05-19

## Summary

**Assessment: PASSED** — all tasks fully implemented, no defects found.

**Outcome verification: 3/3 criteria met**
- `cargo check --workspace` — succeeds, 0 errors
- `cargo clippy --workspace -- -D warnings` — succeeds, 0 warnings
- `cargo test --workspace` — 1 test passes (backbone_compiles, `should_panic`)

## Per-Task Results

### A-1: Create Cargo.toml
- **Status**: ✓ Passed
- **Diff validation**: Edition 2024, `lib` crate-type, path dependency on `chemrust-hamiltonian-core`, all 4 crate deps (ndarray, num-complex, thiserror, bon) present. No unused or missing dependencies.
- **Strategic review**: Minimal and correct.

### B-1: Grid and field types (`src/types.rs`)
- **Status**: ✓ Passed
- **Diff validation**: `WaveGridArray`, `FineGridArray` as opaque newtypes with full accessor suite. `Density`, `EffectivePotential`, `DensityUpsampled` as grid-aware wrappers. Stubs: `KPoint`, `SmearingParams`, `FinalResult`, `Error`. All types `Clone`-derived. Re-exported from `lib.rs`. No dependency on upstream `real_space_field!` macro.
- **Strategic review**: Clean hierarchy prevents grid-level mismatches at compile time.

### B-2: Layout markers and device wrappers (`src/layout.rs`)
- **Status**: ✓ Passed
- **Diff validation**: Sealed `Layout` trait with `RowDistributed`/`ColumnDistributed` ZSTs. `WavefunctionSet<L: Layout>` with `debug_assert_eq!` guard. `Gpu<T>` and `Cpu<T>` with `Deref<Target=T>` identity. `sync_to_host()`/`sync_to_device()` bound on `T: Clone`.
- **Strategic review**: Matches upstream sealed-trait convention.

### B-3: DensityHistory (`src/mixing.rs`)
- **Status**: ✓ Passed
- **Diff validation**: `DensityHistory { densities, max_history }`. `new()` allocates empty vec, `iterations()` returns `.len()`, `mix()` is `todo!()`.
- **Strategic review**: Correct empty-state behavior.

### C-1: Phase markers and `ScfIteration` struct (`src/scf.rs`)
- **Status**: ✓ Passed
- **Diff validation**: Sealed `ScfPhase` trait, 6 ZST markers via `define_phase!` macro. `ScfIteration<S: SpinPolicy = NonSpin, State: ScfPhase = Initialized>` with all 11 fields. `#[bon]` + `#[builder]` on `new()` constructor. `PhantomData<State>`.
- **Strategic review**: Builder compiles and works in integration test. Type defaults prevent incorrect usage.

### C-2: Transitions and `run_scf` (`src/scf.rs`)
- **Status**: ✓ Passed
- **Diff validation**: 5 transitions on correct phase impl blocks — `build_v_eff` (Initialized), `diagonalize` (VEffBuilt), `construct_density` (WavefunctionsUpdated), `mix` (DensityUpdated), `check` (Mixed). Terminal `finalize` (Converged). All consume `self`. `CheckOutcome<S>` enum replaces nested `Result`. `run_scf` loop uses block-expression pattern.
- **Strategic review**: Complete type-state machine. No `clippy::type_complexity` workaround needed.

### D-1: `backbone_compiles` (`tests/backbone_compiles.rs`)
- **Status**: ✓ Passed
- **Runtime outcome verification**: Test panics with `"not yet implemented"` as expected. First `todo!()` in `build_v_eff()` fires, proving the full loop skeleton is wired correctly.
- **Diff validation**: All dummy constructors use only public upstream APIs. Builder pattern for `ScfIteration`. No upstream modifications needed.
- **Strategic review**: End-to-end composability proven with real upstream types.

## Issues Found

*None.* 0 severity-0, 0 severity-1, 0 severity-2 issues across all groups.

## Strategic Observations (Phase 2 preparation)

1. **`ScfIteration` field visibility** — All fields are `pub`. Tighten to `pub(crate)` before non-stub code is added.
2. **Type name collision with upstream** — Both crates define `Density`. Deferred to upstream unification PR.
3. **`DensityHistory::mix()` return values** — Phase 2 transition must use both `(mixed_density, input_snapshot)`.
4. **`check()` must reset `v_eff: None`** — `NotConverged` wraps `Initialized` which expects `v_eff: None`.
5. **`Gpu<T>`/`Cpu<T>` transparent `Deref`** — Correctness risk for later GPU code; sync guards deferred to Tier 3.
