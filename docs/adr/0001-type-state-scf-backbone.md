# ADR-0001: Type-state SCF backbone

**Date:** 2026-05-19

**Status:** Accepted

## Context

The SCF cycle has a fixed topology with strict ordering constraints:
ρ → V_eff → diagonalize H[V_eff] → ψ, ε → ρ_new → mix → check → next ρ.

In CASTEP's Fortran GPU port, 15 distinct crash logs showed the same pattern:
operations called in wrong order, data accessed before it was ready, buffers
reused with stale contents. Fortran's type system cannot express "this function
may only be called after build_v_eff has completed" — these are comment-level
conventions.

We need a way to make illegal states unrepresentable at compile time. Two
approaches:

**Option A: Runtime guards.** Each function checks an enum tag (`Phase`) and
returns `Err` if called in the wrong phase. Lightweight but only catches bugs
at test time.

**Option B: Type-state pattern.** The SCF state carries a zero-sized phase
marker as a generic parameter. Each transition function consumes the current
state and returns a new state with a different phase marker. Illegal transitions
are compile errors.

## Decision

Use Option B (type-state pattern) for the SCF backbone.

The central type is:
```rust
pub struct ScfIteration<State = Initialized> {
    // Immutable for the whole SCF:
    cell: CellGeometry,
    pots: PseudopotentialSet,
    wave_grid: GVectorGrid,
    fine_grid: GVectorGrid,
    k_point: KPoint,
    smearing: SmearingParams,

    // Phase-guarded:
    density: Density,
    psi: WavefunctionSet,
    eigenvalues: Vec<f64>,
    v_eff: Option<EffectivePotential>,
    history: DensityHistory,
    previous_density: Density,

    _phase: PhantomData<State>,
}
```

Phases and their valid transitions:
- `Initialized` → `build_v_eff()` → `VEffBuilt`
- `VEffBuilt` → `diagonalize()` → `WavefunctionsUpdated`
- `WavefunctionsUpdated` → `construct_density()` → `DensityUpdated`
- `DensityUpdated` → `mix()` → `Mixed`
- `Mixed` → `check()` → `Converged | Initialized` (via `Result`)

What the compiler guarantees:
- Cannot call `diagonalize()` before `build_v_eff()`
- Cannot call `construct_density()` before `diagonalize()`
- Cannot re-use stale V_eff after density changes (non-convergence returns
  `Initialized` with `v_eff: None`)
- No unwrap on `v_eff` outside `VEffBuilt` phase

## Consequences

**Positive:**
- The 8 bug categories from the CASTEP GPU port crash logs are prevented at
  compile time, not just at test time
- The SCF loop body is uncluttered — no phase-checking boilerplate
- New developers can see the entire valid-state graph from the type signatures
- Each transition can be implemented independently with known invariants

**Negative:**
- More upfront type machinery (6 phase markers, PhantomData on every struct)
- Error-path plumbing: non-convergence returns `Err(ScfIteration<Initialized>)`,
  which means the density must be `Copy` or cloned (or moved back out)
- Generic parameter on `ScfIteration` propagates through every function that
  touches it — though the `State` parameter is constrained to a sealed trait,
  so the generic is bounded

**Risks:**
- Learning curve for contributors unfamiliar with type-state. Mitigation:
  CONTEXT.md documents the transition graph, and the `run_scf` loop serves as
  the canonical usage example.
- Move semantics: each transition consumes `self`, so the caller must re-bind.
  This is already the pattern in the `run_scf` loop and is idiomatic Rust.

## Alternatives considered

**Runtime enum guards (Option A):**
Rejected because the whole motivation for this project is preventing the bugs
that runtime guards in CASTEP failed to catch. Runtime guards require tests;
type-state is checked by the compiler on every build.

**No state machine (bare functions):**
Rejected because without structure, the caller must manually track which data is
valid. This is exactly the pattern that produced the CASTEP crash logs.

## References

- Bug analysis from 15 CASTEP GPU-resident SCF crash logs at
  `/tmp/cu111_gpu_resident_scf/` — all 8 bug categories are type-state
  violations (order of operations, stale data access, buffer lifecycle)
- `PROJECT_ROOT_PLAN.md` for the full SCF transition design
