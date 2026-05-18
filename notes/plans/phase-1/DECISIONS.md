# Phase 1 Design Decisions

## Grill Outcomes

### Grid type naming
**Decision:** Use `Density`, `EffectivePotential` within `chemrust_scf` module
namespace — module paths disambiguate from `chemrust_hamiltonian_core::Density`.

**Why:** Avoiding `ScfDensity`-style prefixes keeps names concise. The two types
have different inner types (chemrust-scf wraps `WaveGridArray`, upstream wraps
`Array3<f64>`), and will be unified by upstreaming in a follow-up PR.

### Error type
**Decision:** New `chemrust_scf::Error` (thiserror). Single variant
`NotImplemented` for Phase 1, expanded in later phases.

**Why:** Upstream `chemrust_hamiltonian_core::Error` has I/O, format, and grid
mismatch variants that are irrelevant to the SCF engine. Keeping them separate
avoids leaking upstream error semantics.

### Test fixture strategy
**Decision:** All dummy construction within `tests/backbone_compiles.rs` using
public upstream APIs. No upstream modifications.

**Why:** Keeps Phase 1 self-contained. `CellGeometry` has public fields,
`GVectorGrid::new()` and `PseudopotentialSet::new()` exist, `RealLattice` and
`RecipLattice` have `from_inner()`. No need for `Default` impls.

### Builder pattern
**Decision:** Use `bon` v3.9.1 for `ScfIteration::new()` — `#[bon]` on impl
block, `#[builder]` on `fn new()`.

**Why:** Upstream already uses `bon = "3"` for `HkBuilder`. Named-arg
construction for 9 parameters is more readable than positional. The
`ScfIteration::builder().cell(...)...build()` call site is self-documenting.

## Architectural Decisions

### Single struct with PhantomData (not per-phase structs)
From ADR-0001. Reaffirmed: separate structs would add 200+ lines of conversion
boilerplate for zero additional safety.

### Gpu<T> derefs to T in Phase 1
Real sync guards replace `Deref` in Tier 3. For now, all data lives on CPU.

### Hardcoded ColumnDistributed on psi field
`RowDistributed` exists as a type but Phase 1 only needs the
Hamiltonian-apply layout. MPI transpose deferred.

### Block-expression loop pattern in run_scf
Needed because `state` is consumed each iteration. Variable shadowing with
`let state = state.build_v_eff()?` fails — the outer `state` is consumed
before the inner binding. The block expression `state = { ... }` evaluates to
the next `ScfIteration<Initialized>`.

### CheckOutcome enum (not nested Result)
Replaces `Result<Result<ScfIteration<S, Converged>, ScfIteration<S,
Initialized>>, Error>` which triggered `clippy::type_complexity`.
Self-documenting variant names `Converged` and `NotConverged`.
