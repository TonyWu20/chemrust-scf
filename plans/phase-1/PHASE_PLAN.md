# Phase 1: T1 — Type-State SCF Backbone

**Date:** 2026-05-19
**Status:** Draft

## Goals

### Goal 1: Scaffold the `chemrust-scf` crate with workspace and dependencies

Create `Cargo.toml` at the repo root as a single crate (not a workspace) depending on `chemrust-hamiltonian-core` via path. The crate compiles as a library (`crate-type = ["lib"]`). Nix flake already exists — the `cargo check` workflow should pass with `nix develop`.

**Why now:** No crate exists. T1 is the first code that goes into the repo.

**Effort:** Small (1 file, ~10 lines of Cargo.toml)

### Goal 2: Define the three type-safety layers (grid, device, layout)

Define the newtype wrappers that encode invariants currently tracked only in Fortran comments:

- **Grid layer**: `WaveGridArray(Array3<f64>)` and `FineGridArray(Array3<f64>)` as opaque newtypes. `Density(WaveGridArray)`, `EffectivePotential(FineGridArray)`, and `DensityUpsampled(FineGridArray)` (used during upsampling before V_eff assembly).
- **Device layer**: `Gpu<T>` and `Cpu<T>` with `sync_*` stubs. `Gpu<T>` derefs to `T` for T1 (on CPU everything is "on GPU" as identity). Real transfers in Tier 3.
- **Layout layer**: `RowDistributed` and `ColumnDistributed` zero-sized markers. `WavefunctionSet<Layout>` with `data: Vec<Complex64>`, `n_bands`, `n_pw`, and `PhantomData<Layout>`.

**Why now:** These types appear in every transition signature. Without them, the state machine can't be written.

**Effort:** Medium (~100 lines across types.rs and layout.rs)

### Goal 3: Define the SCF state machine (phase markers, ScfIteration, transitions)

Define 6 phase markers (`Initialized`, `VEffBuilt`, `WavefunctionsUpdated`, `DensityUpdated`, `Mixed`, `Converged`) as zero-sized types implementing a sealed `ScfPhase` trait.

Define `ScfIteration<S: SpinPolicy, State = Initialized>` with all fields, `PhantomData<State>`, and the 5 transition functions — each consuming `self` and returning a new `ScfIteration` with a different phase marker:

```
Initialized → build_v_eff() → VEffBuilt
VEffBuilt → diagonalize(ndeg) → WavefunctionsUpdated
WavefunctionsUpdated → construct_density() → DensityUpdated
DensityUpdated → mix() → Mixed
Mixed → check(tol) → Result<Converged, Initialized>
```

Every transition body is `todo!()`. The `ScfIteration::new()` constructor takes `CellGeometry`, `PseudopotentialSet`, `GVectorGrid` (×2), `Density`, `WavefunctionSet`, `SmearingParams`, `KPoint`.

Generic over `SpinPolicy` (defaulting to `NonSpin` for MVP). This means `v_eff: Option<S::VEff>` where `NonSpin::VEff = EffectivePotential`.

**Why now:** This is the backbone. The compiler checks that no transition can be called in the wrong state.

**Effort:** Medium (~150 lines across scf.rs)

### Goal 4: Define stub types for missing domain concepts

Types that don't exist in `chemrust-hamiltonian` but are needed by the SCF state machine:

- `KPoint` — fractional k-point coordinates, default Gamma for T1
- `SmearingParams` — Fermi-Dirac smearing width and electron temperature
- `DensityHistory` — holds previous densities for mixing; `new(max_history)`, `mix(&mut self, density) -> (Density, Density)`, `iterations(&self) -> usize`
- `FinalResult` — converged output: `Density`, `WavefunctionSet`, `Vec<f64>` eigenvalues, `f64` total energy, optional `Force`

All bodies stubbed except trivial constructors needed for the test fixture.

**Why now:** These appear in the `ScfIteration` constructor and `run_scf` return type.

**Effort:** Small (~50 lines across types.rs and mixing.rs)

### Goal 5: Implement `run_scf` loop and compile-check test

The orchestration function:

```rust
pub fn run_scf<S: SpinPolicy>(
    mut state: ScfIteration<S, Initialized>,
    ndeg: usize,
    tol: f64,
) -> Result<FinalResult, Error> {
    loop {
        let state = state.build_v_eff()?;
        let state = state.diagonalize(ndeg)?;
        let state = state.construct_density()?;
        let state = state.mix();
        match state.check(tol)? {
            Ok(done) => return Ok(done.finalize()),
            Err(next) => state = next,
        }
    }
}
```

A `#[test]` annotated `#[should_panic(expected = "not yet implemented")]` that constructs a minimal `ScfIteration` (using `#[cfg(test)]` helpers to build dummy `CellGeometry`, `GVectorGrid`, and an empty `PseudopotentialSet`) and calls `run_scf`.

**Why now:** This is the verification: does the state machine compile? Does the loop type-check? Does the compiler reject wrong-phase calls?

**Effort:** Small (~40 lines of test infrastructure)

## Scope Boundaries

**In scope:**
- `Cargo.toml` with path dependency on `chemrust-hamiltonian-core`
- `src/lib.rs`, `src/types.rs`, `src/scf.rs`, `src/mixing.rs`, `src/layout.rs`
- Grid-level types (`WaveGridArray`, `FineGridArray`, grid-aware `Density`/`EffectivePotential`)
- Device wrappers (`Gpu<T>`, `Cpu<T>`) with stubs
- Layout markers (`RowDistributed`, `ColumnDistributed`, `WavefunctionSet<Layout>`)
- `ScfPhase` sealed trait and 6 phase markers
- `ScfIteration<S, State>` with all fields and 5 transition stubs
- `KPoint`, `SmearingParams`, `DensityHistory`, `FinalResult` stub types
- `run_scf` function
- `#[should_panic]` compile-check test
- CI via `nix develop -c cargo check` and `nix develop -c cargo test`

**Out of scope:**
- Any transition body implementation (all `todo!()`)
- `chemrust-hamiltonian-core` modifications (grid types live in chemrust-scf for now)
- FFT backend changes (still `rustfft` — Tier 3)
- GPU integration (Tier 3)
- Forces, stress, variable cell (Tier 2+)
- `SpinCollinear` implementation (type param exists, only `NonSpin` tested)
- MPI distribution logic (RowDistributed/ColumnDistributed are markers only)
- Python bindings, C ABI, `cdylib` build
- Loading real pseudopotential files in T1 test (use empty/new `PseudopotentialSet` or dummy)

## Design Notes

### Why grid-aware types in chemrust-scf (not upstreamed now)

Modifying `chemrust-hamiltonian-core/src/types.rs` would change the `real_space_field!` macro and all consumers. T1 should be a self-contained deliverable. Once T1 lands and the grid types are proven correct, a follow-up PR upstreams them to `chemrust-hamiltonian-core` and removes the duplicate definitions from `chemrust-scf`. The adapter layer at the boundary is just type conversions that disappear when the types are unified.

### Why single struct with PhantomData (not per-phase structs)

The CASTEP crash logs show 8 bug categories, all of which involve calling operations at the wrong time. The PhantomData pattern prevents every one of them: the compiler rejects `diagonalize()` before `build_v_eff()`, rejects re-using stale V_eff, and rejects wrong-layout wavefunction operations. Separate per-phase structs would add 200+ lines of conversion boilerplate for zero additional safety. The internal `Option<v_eff>` is imprecise but harmless — the `VEffBuilt` constructor always sets it to `Some`, and no other phase can read it.

### Why generic over SpinPolicy now (even though only NonSpin tested)

`VEffBuilder<S: SpinPolicy>` already exists in `chemrust-hamiltonian`. The SCF transitions need to hold `S::VEff` in the `v_eff` field. If we hardcode `EffectivePotential`, we either (a) lose the ability to add `SpinCollinear` later without refactoring, or (b) need an enum unification layer. The generic adds one type parameter with a default — negligible complexity, keeps the door open.

### Why `Gpu<T>` / `Cpu<T>` now (even for CPU-only T1)

These appear in the transition signatures from the plan design. For T1, `Gpu<T>` is a transparent wrapper — `sync_to_host` is a no-op or stubbed. Defining them now means the type signatures are correct from day one, and Tier 3 only fills in the actual GPU transfer bodies.

### Module structure

```
chemrust-scf/
├── Cargo.toml
├── src/
│   ├── lib.rs          # pub mod types, scf, mixing, layout; re-exports
│   ├── types.rs        # WaveGridArray, FineGridArray, Density, EffectivePotential,
│   │                   # DensityUpsampled, KPoint, SmearingParams, FinalResult
│   ├── layout.rs       # RowDistributed, ColumnDistributed, WavefunctionSet<L>,
│   │                   # Gpu<T>, Cpu<T>
│   ├── scf.rs          # ScfPhase trait, phase markers, ScfIteration<S,State>,
│   │                   # transitions, run_scf, ScfIteration::new()
│   └── mixing.rs       # DensityHistory
└── tests/
    └── backbone_compiles.rs  # #[should_panic] integration test
```

### Path dependency

`chemrust-scf` is at `~/programming/chemrust-scf/`. `chemrust-hamiltonian` is at `~/programming/chemrust-hamiltonian/`. The Cargo.toml path dependency is:

```toml
[dependencies]
chemrust-hamiltonian-core = { path = "../chemrust-hamiltonian/chemrust-hamiltonian-core" }
```

This assumes both repos are checked out side-by-side under `~/programming/`. The Nix flake may need updating to provide the path. For CI, both repos must be available.

### Phase marker trait design

```rust
mod sealed {
    pub trait Sealed {}
}

pub trait ScfPhase: sealed::Sealed {
    /// Human-readable name for error messages.
    fn name() -> &'static str;
}

// 6 implementors: Initialized, VEffBuilt, WavefunctionsUpdated,
//                 DensityUpdated, Mixed, Converged
```

The sealed trait prevents external implementations — only the 6 defined phases can ever exist, which is a compile-time guarantee that the state graph is closed.

## Deferred Items Absorbed

None — this is Phase 1, no prior deferred items exist.

## Domain Terms

No new terms — the CONTEXT.md from the project constitution already covers all T1 types (Density, WavefunctionSet, RowDistributed, ColumnDistributed, Gpu, Cpu, WaveGridArray, FineGridArray, EffectivePotential, ScfIteration, Chebyshev Filtering, Rayleigh-Ritz).

## Verification

1. **`cargo check`**: Must compile with no errors. The state machine types must be self-consistent.
2. **`cargo test`**: One test, `backbone_compiles`, annotated `#[should_panic(expected = "not yet implemented")]`. This proves:
   - `ScfIteration::new()` can be constructed from real types
   - `run_scf()` type-checks (the 5 transitions compose correctly in the loop)
   - The first transition (`build_v_eff`) is reached before the `todo!()` fires
3. **Negative compile test** (optional for T1, can be deferred): `trybuild` tests that assert compile FAILURE for wrong-phase calls, e.g., calling `diagonalize()` on `ScfIteration<Initialized>`.
