# Deferred Items — Phase 1

Items flagged during review as worth doing but out of scope for Phase 1.

## Structural

1. **All transition body implementations** — All `todo!()` bodies in `ScfIteration` methods (build_v_eff, diagonalize, construct_density, mix, check, finalize). The entire SCF physics pipeline. Phase 2+.

2. **`ScfIteration` field visibility** — Currently all `pub`. Tighten to `pub(crate)` before non-stub transition code is added. Phase 2 start.

3. **`DensityHistory` actual mixing algorithm** — Pulay/DIIS mixing. Phase 2.

## Upstream Dependencies

4. **`GVectorGrid` lacks `Clone` and `Debug`** — Blocks deriving those on `ScfIteration`. Requires upstream change.

5. **Type name collision: both crates have `Density`** — Deferred to upstream unification PR.

## Device Layer

6. **`Gpu<T>`/`Cpu<T>` sync guard replacement** — Transparent `Deref` is fine for identity but must be replaced with explicit sync guards when GPU code is introduced. Tier 3.

7. **MPI transpose support** — `RowDistributed` marker exists but no transpose implementation. Needed when Rayleigh-Ritz on RowDistributed layout is implemented.

## Convenience

8. **Arithmetic ops on field types** — `Add`, `Sub`, etc. for domain types. Needed in Phase 2 when VEff assembly and density mixing operate on grid arrays.

9. **ScfIteration Debug/Clone derives** — Blocked by `GVectorGrid`.

10. **DensityHistory::mix() output wiring** — Phase 2 transition must consume both `(mixed_density, input_snapshot)` return values to set `self.density` and `self.previous_density`.
