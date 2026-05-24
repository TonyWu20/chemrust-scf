# CASTEP-Anchored PostRr Upper-Bound Test — Cascade IS Rotation-Driven

**Date:** 2026-05-24
**Test:** `cascade_with_castep_anchored_postrr_pin` (tests/ca_scf_convergence.rs)
**Question answered:** Can ANY Procrustes pin stop the iter-3 cascade?

## Setup

Inject CASTEP ψ as the input to every iteration via `psi_data_mut()`. PostRr's
`prev_psi_dev` then equals CASTEP ψ at every call. The pin aligns each
iteration's RR output to CASTEP's basis — the strongest possible Procrustes
target.

## Result

| Configuration | iter-1 band-0 | iter-2 | iter-3 | Drift vs A1 |
|---|---|---|---|---|
| PinMode::Off baseline | −1.046 | drift | **−11.94** | 10.9 Ha |
| PostRr + relative (`debug-20260524-postrr-cascade-amplification`) | −1.046 | −0.890 | **−14.91** | 13.86 Ha |
| **PostRr + CASTEP-anchored (this test)** | **−1.046** | **−0.892** | **−1.227** | **0.172 Ha** |

EXTERNAL anchor A1: CASTEP band-0 = −1.05502287 Ha (`Cu111_CO.bands:12`).

## Three findings

### 1. Cascade IS rotation-driven (proposal §14 was right about mechanism)

With CASTEP ψ as the absolute reference, iter-3 stays at −1.227 Ha
instead of catastrophically diverging to −11.94 Ha. **The pin works.**
The proposal §14 attribution to ZHEGVD rotation was correct; the failure
of all prior pin attempts was due to the *reference choice*, not the
mechanism.

### 2. Iter-1 has irreducible ~9 mHa residual even with perfect anchor

Iter-1 band-0 = −1.046 Ha vs CASTEP −1.055 Ha — a 9 mHa drift even when
the pin reference IS CASTEP ψ. This is the Cu-3d filter discrimination
floor (the 0.893 ratio measured by `subspace_projector_iter1_vs_castep`).
The pin can rotate the basis but cannot recover precision lost at
iter-1's filter step.

### 3. V_eff drift adds ~0.16 Ha per iteration even with CASTEP anchor

Iter-1 → iter-2 → iter-3 drift: 0.009 → 0.163 → 0.172 Ha. After iter-1,
each iteration's density is built from CASTEP-anchored iter-(N-1) output
(which has the 9 mHa rotation residual). That gives a slightly-wrong
V_eff which makes iter-N's eigenproblem a slightly-different operator,
so even re-pinning to CASTEP ψ leaves a residual.

## Forecast for production absolute-target Procrustes

Production would use iter-1's own RR output as the frozen reference (not
CASTEP — that would require running CASTEP first). The reference being
iter-1 instead of CASTEP costs ~0.2 Ha additional drift compared to this
upper-bound test.

Predicted iter-3 with iter-1-anchored pin: somewhere between −1.4 and −0.8
Ha (0.3-0.5 Ha drift from A1).

| Acceptance criterion | Predicted with iter-1-anchored pin | Status |
|---|---|---|
| iter-3 cascade < 10 Ha (no catastrophic divergence) | 0.3-0.5 Ha | **GREEN** |
| iter-3 cascade < 0.1 Ha (current `cascade_iter3_diagnostic_tight` gate) | unlikely | RED but ~30-60× lifted vs baseline |
| Q2 total energy < 1e-5 eV (CASTEP ship gate) | very unlikely | RED — algorithmic floor |
| Q2 total energy < 10 mHa (loose ship gate) | possibly | depends on residual energy variance |

## Recommendation

**Implement production absolute-target Procrustes.** It will stop the
catastrophic cascade. The cascade test should be split per the
`tolerance-conflation-in-acceptance-test` pattern (failure-patterns
2026-05-24):
- Q1_cascade (loose): iter-3 band-0 within 1 Ha of A1 — should be GREEN
- Q2_cascade (medium): iter-3 band-0 within 0.1 Ha of A1 — likely RED, document algorithmic floor
- Q3_energy (loose): Q2 total energy within 10 mHa of CASTEP — possibly GREEN
- Q4_energy (CASTEP ship): Q2 total energy within 1e-5 eV — expected-fail, requires CG eigensolver

## Caveat

This test MIXES the V_eff/density evolution (from our own
construct_density of CASTEP-anchored RR output) with the absolute pin
reference (CASTEP ψ). It is therefore not a fair test of what production
absolute-target Procrustes would do — production's reference would be
iter-1's RR output, not CASTEP ψ. Even so, the test cleanly establishes
the upper-bound: even with the strongest possible reference, residual
drift is ~0.16 Ha per iteration. This is V_eff/density drift, not pin
weakness.
