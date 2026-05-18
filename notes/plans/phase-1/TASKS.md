# Phase 1: Type-State SCF Backbone — Task Record

## Declared Fixtures

No fixture files are consumed in Phase 1. All transition bodies are `todo!()`.
The sole acceptance criterion is **compiler-enforced type safety** — the Rust
compiler is the fixture.

## Task Groups

### Group A — Crate Scaffold

#### A-1: Create `Cargo.toml`

**Kind:** direct

**Guidance:** Single crate, `edition = "2024"`, `lib` crate-type. Path
dependency on `chemrust-hamiltonian-core`. Deps: ndarray, num-complex, thiserror,
bon.

**Success Criteria:**

- `cargo check` resolves all dependencies
- `cargo test` runs (even if no tests yet)

---

### Group B — Type-Safety Layers

#### B-1: Grid and field types (`src/types.rs`)

**Kind:** direct

**Guidance:** Define `WaveGridArray(Array3<f64>)` and `FineGridArray(Array3<f64>)`
as opaque newtypes with accessor methods. Define `Density(WaveGridArray)`,
`EffectivePotential(FineGridArray)`, `DensityUpsampled(FineGridArray)` as
grid-aware field wrappers. Define stubs: `KPoint`, `SmearingParams`, `FinalResult`,
`Error`.

**Success Criteria:**

- All types compile — no dependencies on upstream's `real_space_field!` macro
  (it is `pub(crate)`)
- `Clone` derives on `Density` (needed for `previous_density = density.clone()`)
- The types are re-exported from `lib.rs`

#### B-2: Layout markers and device wrappers (`src/layout.rs`)

**Kind:** direct

**Guidance:** Sealed `Layout` trait with `RowDistributed`, `ColumnDistributed`
ZSTs. `WavefunctionSet<L: Layout>` with `data`, `n_bands`, `n_pw`,
`PhantomData<L>`. `Gpu<T>` and `Cpu<T>` with `Deref<Target=T>` for T1 identity.

**Success Criteria:**

- `Gpu<T>` derefs transparently to `T`
- `sync_to_host()` / `sync_to_device()` compile for Clone types
- `WavefunctionSet::new()` panics in debug if data length ≠ n_bands × n_pw

#### B-3: DensityHistory (`src/mixing.rs`)

**Kind:** direct

**Guidance:** `DensityHistory { densities: Vec<Density>, max_history: usize }`.
Methods: `new()`, `mix()`, `iterations()`. Body of `mix()` is `todo!()`.

**Success Criteria:**

- Construction with `new(4)` works
- `iterations()` returns 0 on fresh instance

---

### Group C — SCF State Machine

#### C-1: Phase markers and `ScfIteration` struct (`src/scf.rs`)

**Kind:** direct

**Guidance:** Sealed `ScfPhase` trait with 6 ZST markers. `ScfIteration<S, State>`
with all fields. `#[bon]` + `#[builder]` on the `new()` constructor.
`PhantomData<State>` for the phase parameter.

**Success Criteria:**

- Constructor via `ScfIteration::builder().cell(...)...build()` compiles
- `S` defaults to `NonSpin`, `State` defaults to `Initialized`

#### C-2: Transitions and `run_scf` (`src/scf.rs`)

**Kind:** direct

**Guidance:** 5 transition methods on correct phase impl blocks. `run_scf` loop
with block-expression pattern. `CheckOutcome` enum for convergence result.

**Success Criteria:**

- `build_v_eff()` only exists on `ScfIteration<Initialized>`
- `diagonalize()` only exists on `ScfIteration<VEffBuilt>`
- Each transition consumes `self` and returns the next phase
- `CheckOutcome<S>` replaces `Result<Result<..., ...>, Error>` — no
  `clippy::type_complexity` allow

---

### Group D — Integration Test

#### D-1: `backbone_compiles` (`tests/backbone_compiles.rs`)

**Kind:** direct

**Guidance:** Integration test with `#[should_panic(expected = "not yet
implemented")]`. Construct dummy `CellGeometry`, `GVectorGrid`,
`PseudopotentialSet`, `Density`, `WavefunctionSet<ColumnDistributed>` using
public APIs. Use builder to construct `ScfIteration`, call `run_scf`.

**Success Criteria:**

- Test passes with expected panic message
- Proves all types compose correctly at the integration boundary
- Dummy construction uses only public upstream APIs (no upstream changes)

## Exploration Notes

- `real_space_field!` macro is `pub(crate)` in
  `chemrust-hamiltonian-core/src/types.rs` — cannot reuse. Manual newtypes needed.
- `bon` v3.9.1 requires `use bon::bon;` and `#[bon]` on the impl block with
  `#[builder]` on each method, NOT `#[bon::builder]` on the fn (which only works
  for standalone functions).
- `GVectorGrid` is not `Clone` or `Debug` — prevents deriving those on
  `ScfIteration` without upstream changes (deferred).
- `CellGeometry` has all public fields — struct literal construction works for
  tests. No `new()` constructor needed.
- `Gpu<T>` and `Cpu<T>` use `Deref` for Phase 1 identity; this will be replaced
  with explicit sync guards in Tier 3.

## Verification

```bash
cargo check                              # Must succeed
cargo clippy --workspace -- -D warnings  # Must succeed (0 warnings)
cargo test                               # 1 test: backbone_compiles — passes
```
