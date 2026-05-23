# Resolution — T-prime D-injection discriminator: FAIL → cascade is eigensolver-rotation-driven

**Date**: 2026-05-24
**Branch**: `feat/phase-global-woodbury`
**Driver question**: Is the SCF cascade driven by (a) D-screening accuracy
post-rotation, or (b) eigensolver-induced ψ rotation independent of D quality?

## Empirical answer

**T-prime FAIL at 197 mHa**, cascade continues with CASTEP-injected D matrices.

| Configuration | iter-2 band-0 | \|Δ\| vs CASTEP |
|--------------|--------------|------------------|
| Natural cascade (our V_eff, our D) | −0.87 Ha (SUMMARY.md) | 184 mHa |
| **T-prime: our V_eff + CASTEP D injected** | **−0.858 Ha** | **197 mHa** |
| T3: CASTEP V_eff + our D | −1.0452 Ha | 9.8 mHa |
| CASTEP reference | −1.05502310 Ha | 0 |

The discriminator is sharp: **D injection alone does not stop the cascade**;
**V_eff injection does**. The cascade is therefore not D-driven at the
post-rotation state — it's driven upstream of D, in the V_eff that our
iter-2 builds from iter-1's rotated ρ.

## Mechanism (now confirmed empirically)

The chain matches `notes/debug/debug-20260523-2314/SUMMARY.md`'s diagnosis:

1. Iter-1: subspace-RR + Chebyshev filter rotates ψ within Cu 3d
   degenerate manifold (overlap 0.252 at ndeg=0 starting from CASTEP ψ;
   `failure-patterns.md:90`).
2. Rotated ψ produces β·ψ projections that differ from CASTEP's even
   though occupations are identical and soft density is rotation-invariant.
3. Rotated β·ψ produces wrong ρ_aug (29.8/70.2 split vs CASTEP F8 36.8/63.2).
4. Wrong ρ_aug → wrong V_eff (V_eff reconstruction itself is at 4 µHa,
   per `chemrust-hamiltonian` `test_cu111_co_potential_residual`, but it
   is fed wrong inputs).
5. Wrong V_eff drives wrong filter shifts and cascade by iter-3.

**T-prime confirms**: replacing D in step 4 doesn't help. Step 4's V_eff
is wrong because step 3's ρ_aug is wrong, and step 3's ρ_aug is wrong
because step 1 rotated ψ. The only places to intervene are step 1
(prevent rotation) or external substitution at step 4 (T3 trick).

## Egg-or-chicken deadlock: dissolved

| Layer | Blocker for chemrust-scf? | Fix on its side? |
|-------|---------------------------|------------------|
| chemrust-hamiltonian D-screening | No (T-prime FAIL) | N/A — already at post-SCF floor per `66e661d` |
| chemrust-hamiltonian V_eff assembly | No (4 µHa per `test_cu111_co_potential_residual`) | N/A |
| chemrust-scf eigensolver rotation | **Yes** | F3a/F3b/F3c from prior plan |

The deadlock between repos described in the user's framing is now
empirically dissolved: chemrust-hamiltonian is not the blocker.

## Anchor criteria used

- A1: Cu111_CO.bands:12 → band-0 = −1.05502310 Ha (EXTERNAL)
- A2: D_band_debug.dat last 18 blocks (EXTERNAL, parsed via existing
  `tests/fixtures/cu111_co.rs:233::load_castep_d_screened`)
- A3: T3 result (PASS at 9.8 mHa; from
  `notes/debug/debug-20260523-2314/SUMMARY.md`)

## What changed in code

- `src/eigensolver/vnl_data.rs`: added `precompute_with_d_override` taking
  `Option<&[Option<Vec<f64>>]>`; the existing `precompute` now delegates.
- `src/scf.rs`: added `diagonalize_with_d_override` on
  `ScfIteration<S, VEffBuilt, MixingOff>` and refactored
  `diagonalize_with_mode` to share `diagonalize_inner`.
- `tests/ca_scf_convergence.rs:2891+`: added test
  `iter2_band0_with_castep_d_injection`.

## Run command

```
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0523_with_D \
  cargo test --release --test ca_scf_convergence \
  iter2_band0_with_castep_d_injection -- --ignored --nocapture
```

Runtime: ~225 s (compile + GPU run on Cu111_CO).

## Reclassification

- The hypothesis "Cu d-beta2 D-screening drift to 5e-4 Ha is required for
  Q2" — **HYPOTHESIZED** initially (issue #9 first body) → **REFUTED** by
  T-prime. The 17.9 mHa Cu d-beta2 drift, while real, is not load-bearing
  for the cascade.
- The hypothesis "V_eff assembly drifts ~2 Ha and is the second blocker"
  — **HYPOTHESIZED** (Step 5 of plan) → **REFUTED** by
  `test_cu111_co_potential_residual` (4 µHa).
- The hypothesis "eigensolver rotation drives the cascade" — promoted from
  HYPOTHESIZED (`failure-patterns.md` 2026-05-23 entry) to **EXTERNAL** by
  T-prime FAIL.

## Next steps

1. **Filter ψ stabilization investigation** (chemrust-scf-side, Step 5
   path under T-prime FAIL branch in `PLAN.md`):
   - F3a — lock occupied states before GS+RR
   - F3b — pin previous-iteration ψ as GS seed
   - F3c — Davidson-style block update
2. **Update issue #9 a third time** with this resolution and the
   "deadlock dissolved" conclusion. Recommend keeping issue open as a
   tracking artifact for the 17.9 mHa Cu d-beta2 figure (per user
   direction), but with clear note that it is *not* a Q2 blocker.
3. **Carry on with Steps 1, 2, 3, 6** of the original plan (test
   refactoring) — those are independent of this resolution and unblocked.

## Related

- Plan file: `notes/debug/debug-20260524-tprime-d-injection/PLAN.md`
- Prior session SUMMARY: `notes/debug/debug-20260523-2314/SUMMARY.md`
- chemrust-hamiltonian issue: https://github.com/TonyWu20/chemrust-hamiltonian/issues/9
- chemrust-hamiltonian V_eff residual: 4 µHa per
  `test_cu111_co_potential_residual` (commit `0d5dcbf`)
