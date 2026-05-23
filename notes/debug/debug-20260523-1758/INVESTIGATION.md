# Prior Investigation Classification: Issue #11a

## Symptom
Per-band eigenvalue branches in R-ChFSI (Das Alg 3 lines 598-604) contribute 64% of iter-2 last-band overshoot. When `eigenvalues.is_some()` at iter-2+, the filter uses per-band Λ_Y init and per-band Λ_X updates. Empirical test with `CHEMRUST_FORCE_NO_EIGS=1` (forcing `eigenvalues=None`) reduces iter-2 last-band overshoot from **1.95 Ha → 0.70 Ha** (64% improvement). Band-0 drift unchanged (0.18 Ha).

## Numeric Claims from open-followups.md §11a

| Claim | Value | Source | Classification | Admissible? |
|-------|-------|--------|----------------|-------------|
| Iter-1 band-0 eigenvalue | −1.046 Ha | `open-followups.md:580` | DERIVED (from our RR) | No |
| Iter-2 band-0 eigenvalue | −0.864 Ha | `open-followups.md:580` | DERIVED (from our RR) | No |
| Iter-3 band-0 eigenvalue | −12.68 Ha | `open-followups.md:580` | DERIVED (from our RR) | No |
| Reference band-0 eigenvalue | −1.055 Ha | `open-followups.md:580` | **EXTERNAL** (CASTEP `.bands` or `.castep`) | **Yes** |
| Iter-1 last band | 0.130 Ha | `open-followups.md:581` | DERIVED (from our RR) | No |
| Iter-2 last band | 1.952 Ha | `open-followups.md:581` | DERIVED (from our RR) | No |
| Iter-3 last band | 0.286 Ha | `open-followups.md:581` | DERIVED (from our RR) | No |
| Reference last band | 0.115 Ha | `open-followups.md:581` | **EXTERNAL** (CASTEP `.bands`) | **Yes** |
| Iter-1 V_eff range | 8.69 Ha | `open-followups.md:582` | DERIVED (from our V_eff) | No |
| Iter-2 V_eff range | 20.26 Ha | `open-followups.md:582` | DERIVED (from our V_eff) | No |
| Iter-3 V_eff range | 30.36 Ha | `open-followups.md:582` | DERIVED (from our V_eff) | No |
| Reference V_eff range | 8.69 Ha | `open-followups.md:582` | **EXTERNAL** (fixture `.pot_fmt`) | **Yes** |
| Iter-2 last-band with per-band disabled | 0.70 Ha | `open-followups.md:602` | DERIVED (from our RR with `CHEMRUST_FORCE_NO_EIGS=1`) | No |
| Per-band contribution to overshoot | 64% | `open-followups.md:602` | DERIVED (computed from (1.95-0.70)/1.95) | No |
| Band-0 drift (iter-1 → iter-2) | 0.18 Ha | `open-followups.md:622` | DERIVED (computed from our eigenvalues) | No |
| Global Woodbury S⁻¹ accuracy | ζ = 3.8e-15 | `open-followups.md:569` | DERIVED (from our identity test) | No |

## External Anchor Candidates

Only these claims are admissible as success criteria:

1. **Reference band-0 eigenvalue = −1.055 Ha** (Source: CASTEP `.bands` or `.castep` file)
2. **Reference last-band eigenvalue = 0.115 Ha** (Source: CASTEP `.bands`)
3. **Reference V_eff range = 8.69 Ha** (Source: fixture `.pot_fmt`)

All other numeric claims are DERIVED from our own (potentially buggy) pipeline and cannot be used as criteria without independent corroboration.

## Root Cause Hypothesis from Prior Notes

The prior investigation hypothesized:

> "R-ChFSI's per-band machinery was designed for the regime where ζ = ‖D⁻¹ − B⁻¹‖ > 0 (inexact S⁻¹). After §10's Global Woodbury fix, ζ = 3.8e-15 (machine epsilon). Das main.tex:612 states that when ζ = 0 and the same matrix is used for filter and RR, R-ChFSI ≡ standard ChFSI algebraically. The per-band machinery provides zero benefit but introduces numerical weak points (cancellation in the recurrence, sensitivity to stale eigenvalue labels when V_eff drifts between iterations)."

**Classification**: HYPOTHESIZED. The claim that "per-band machinery introduces numerical weak points" is an inference from the empirical observation (64% improvement when disabled), not a verified fact. The claim that "eigenvalues are stale when V_eff drifts" is plausible but not independently verified.

## Memory Citation Check

No memory entries were cited in the prior investigation notes for Issue #11a. The investigation was conducted within the same session that produced `open-followups.md`.

## Admissible Claims Summary

**EXTERNAL claims only:**
- CASTEP reference band-0 eigenvalue: −1.055 Ha
- CASTEP reference last-band eigenvalue: 0.115 Ha  
- CASTEP reference V_eff range: 8.69 Ha (from fixture `.pot_fmt`)

**All other claims are DERIVED or HYPOTHESIZED and inadmissible as criteria.**
