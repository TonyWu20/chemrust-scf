# Resolution: §13 SCF Cascade — D-screening & V_eff Root-Cause Localisation

**Symptom**: SCF cascade — starting from CASTEP converged ψ and V_eff, iter-1
correct (band-0 = −1.046 Ha), iter-2 drifts (band-0 = −0.869 Ha), iter-3 collapses
(band-0 = −11.94 Ha).

**Root cause**: Stale augmentation density (ρ_aug) from our subspace-rotated ψ
leaks into the next iteration's V_eff via `into_phase()`. Our ψ differs from
CASTEP's within degenerate Cu 3d manifolds, producing ρ_aug that differs from
CASTEP's. When this stale ρ_aug is added to the current soft density in
`build_v_eff_with_energy_impl`, the total density and resulting V_eff are wrong.
The V_eff error amplifies through D-screening (∫Q·V_eff) and the Chebyshev filter.

However, **simply removing ρ_aug from V_eff accelerates divergence** — the aug
density provides important damping by encoding where charge SHOULD concentrate
(at ion cores). The stale aug, while imperfect, is better than none.

**Two contributing factors**:
1. **D-screening discrepancy** (T2): Our `compute_screened_d` with CASTEP V_eff
   differs from CASTEP's D dump by 0.1–2 Ha (up to 22% on diagonal elements).
   This causes V_NL to differ, rotating ψ further. Likely cause: projector
   ordering mismatch between `expanded_projector_lm` and CASTEP's `num_ps_projectors`,
   or Q-on-grid normalization difference.
2. **V_eff sensitivity to aug density** (T3/T4): The cascade stops completely
   when CASTEP V_eff is substituted (T3), but removing aug density entirely (T4
   with cleared aug → cascade fix attempt) makes the cascade WORSE (iter-2
   band-0 = −23.3 Ha).

**Fix location**: Not yet — requires addressing one of:
- (a) Fix D-screening projector ordering / Q normalization to match CASTEP
      element-by-element (reduce ψ rotation at source)
- (b) Damp aug density contribution to V_eff with a mixing weight < 1.0
- (c) Recompute aug density fresh in `build_v_eff_with_energy` using current
      occupations (requires breaking the circular V_eff → diag → occ → aug → V_eff
      dependency)

**Anchor criteria used**:
- A1: Cu111_CO.bands band-0 = −1.05502310 Ha
- A2: D_band_debug.dat per-ion D_screened matrices
- E1: T1 confirming H operator correctness
- E2: Density code correctness for CASTEP ψ

**Test results**:

| Test | Result | Key value | Threshold |
|------|--------|-----------|-----------|
| T1 `h_on_castep_psi_matches_bands` | PASS | RMS 0.0046 Ha, max 0.019 Ha | 0.05/0.10 Ha |
| T2 `d_screened_matches_castep_dump_on_castep_veff` | FAIL | max|Δ| = 2.13 Ha | 5e-4 Ha |
| T3 `cascade_with_castep_veff_substitution` | PASS | iter-2 band-0 = −1.0452 Ha | ±0.05 Ha |
| T4 `cascade_with_castep_density_substitution` (cleared aug) | PASS | iter-2 band-0 = −1.0452 Ha | ±0.05 Ha |
| T4 (original, stale aug) | FAIL | iter-2 band-0 = −3.92 Ha | ±0.05 Ha |
| `cascade_iter3_diagnostic` (remove aug fix attempt) | WORSE | iter-2 band-0 = −23.3 Ha | n/a |

**Prior notes reclassified**:
- "eigenvector-rotation-cascade-divergence" in failure-patterns.md: the rotation
  IS real (T1 confirms H is correct, ψ rotation is downstream of V_eff error),
  but the mechanism is NOT "inherent to subspace methods." The cascade is driven
  by a concrete code defect: stale ρ_aug contamination of V_eff.
- `compute_screened_d formula matches CASTEP` reclassified from EXTERNAL
  (claim) to HYPOTHESIZED (requires per-element verification). T2 disproves
  element-by-element match at the 5e-4 Ha level.

**Date**: 2026-05-24
