# Deferred Items: Phase 2 Group-B

Items flagged during review but intentionally deferred to future phases as they are not blocking for Phase 2 functionality.

## 1. Non-diagonal ZHEGVD test

The solver test uses only a diagonal matrix (`A = diag(1,2,3,4)`, `B = I`), which is the simplest possible case. A non-diagonal Hermitian test (e.g., a 4×4 random Hermitian matrix with known eigenvalues) would increase confidence in the solver wrapper.

**Recommendation for future:** Add during Phase 3 when the full diagonalize pipeline is tested end-to-end.

## 2. `gemv_f64` unit test

`gemv_f64` is implemented but untested. Low priority since GEMV is a thin wrapper over cudarc's safe API.

## 3. `#[allow(dead_code)]` on `BatchedFftPlan3d`

This will be naturally resolved when Group D (density construction) uses the batched plan. Remove the attribute at that point.

## 4. `Cpu<T>(pub T)` public field

Pre-existing newtype violation. Changing `pub T` to `pub(crate) T` or private would require auditing all external access sites. Defer to a dedicated encapsulation cleanup phase.

## 5. AddAssign/SubAssign missing on grid types

Group A-2's guidance specified `AddAssign`/`SubAssign` impls on grid newtypes, but only `Add`/`Sub`/`Mul<f64>` were implemented. These are needed by mix loops (`ρ_mix += c_i * ρ_i`). This was deferred in Group A and impacts Group E (Pulay mixing). Defer to the Group A fix pass or Group E implementation.
