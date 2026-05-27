# Decisions: Diagnostic 3 — Outer Loop Convergence Test

**Date**: 2026-05-27  
**Participants**: User, grill-me agent  
**Status**: Finalized

---

## Fixture Files

### Primary Fixture
**Path**: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`

**Contents**:
- `Cu111_CO.check` — CASTEP-converged wavefunctions (S-orthonormal under USPP S)
- `Cu111_CO.bands` — Reference eigenvalues (160 bands)
- `Cu111_CO.den_fmt` — Converged density
- `Cu111_CO.pot_fmt` — Converged V_eff

**Starting State**: CASTEP-converged state (same as Diagnostic 2)

**Rationale**: Starting from converged state isolates the outer loop behavior from SCF density-update effects. This tests "does the outer loop prevent the cascade" rather than "can we converge from scratch."

### Reference Data

**No external anchor for per-iteration residuals**. We're establishing baseline behavior for the first time.

**Anchors**:
- **Initial state** (iter-1 of Diagnostic 3): Diagnostic 2 results (conduction mean 0.026 Ha, occupied mean 0.15 Ha)
- **Final state** (iter-10): Residuals should be significantly lower (target: 5× reduction for occupied bands)
- **Eigenvalue reference**: CASTEP's converged eigenvalues from `.bands` file (should stay close throughout)

---

## Success Criteria (Refined)

### SC-1: Residual Monotonicity (per-group)

**Criterion**: For each band group (core, cu3d, val, nFermi, cond), the mean S⁻¹ residual at iteration N must satisfy:
- `residual[N] ≤ residual[N-1] × 1.05` (allow up to 5% increase per iteration, to handle numerical noise)
- No more than 2 non-consecutive violations across iterations 2-10
- At least one of iterations {5, 10} must show `residual[N] < residual[1] × 0.8` (20% reduction from baseline)

**Source**: PARSEC Algorithm 4 (Liou et al. 2020) — subspace iteration with Chebyshev filtering converges monotonically for well-conditioned systems.

**Verification**: Plot mean residual per group vs. iteration. Assert:
1. No more than 2 non-consecutive violations of the 1.05× threshold
2. At least one of {iter-5, iter-10} shows ≥20% reduction from iter-1

**Rationale**: The original criterion ("no sustained increase >2 consecutive iterations with >10% growth") was too loose — it allowed indefinite plateau. The refined criterion catches both plateau (no progress by iter-5 or iter-10) and divergence (sustained increases).

---

### SC-2: Conduction Band Early Convergence

**Primary Criterion**: At least 50 of the 63 conduction bands (79%) must reach S⁻¹ residual < 0.01 Ha within 5 outer iterations.

**Secondary (Diagnostic)**: Track the worst-case conduction band. If max conduction residual > 0.05 Ha at iter-5, flag it (not a failure, but worth investigating).

**Source**: Diagnostic 2 baseline — conduction bands start at mean 0.026 Ha. To reach 0.01 Ha in 5 iterations requires 2.6× reduction in 4 additional iterations (per-iteration reduction factor ~1.27×). Well-separated eigenvalues converge at rate ~(λ_k/λ_{k+1})^m per iteration (Zhou 2014).

**Verification**: 
1. Count bands with residual < 0.01 Ha at iteration 5. Assert count ≥ 50.
2. Report max conduction residual at iter-5 (diagnostic only).

**Rationale**: 1.27× per-iteration reduction is realistic for Chebyshev filtering (T_8 polynomial). Tracking worst-case separately catches the scenario where 50 bands converge but 13 plateau at 0.05 Ha (suggests a subgroup with different convergence characteristics, possibly near-Fermi bands).

---

### SC-3: Occupied Band Residual Reduction

**Primary Criterion**: Mean S⁻¹ residual for occupied bands (bands 0-81) decreases by ≥5× from iter-1 to iter-10.

**Secondary (Diagnostic)**: Track max occupied residual. If max > 0.3 Ha at iter-10 (only 1.4× reduction from Diagnostic 2 max of 0.42 Ha), flag it.

**Source**: Diagnostic 2 baseline — occupied bands start at mean ~0.15 Ha. A 5× reduction → 0.03 Ha at iter-10 (per-iteration reduction factor ~1.19×).

**Verification**:
1. Compute mean residual for bands 0-81 at iter-1 and iter-10. Assert ratio ≥ 5.0.
2. Report max occupied residual at iter-10 (diagnostic only).

**Rationale**: 0.03 Ha is acceptable for a diagnostic test. We're testing "does the outer loop help at all?" not "does it reach production tolerance." If we get 5× reduction, that validates the approach. Phase 1 can target 10× or tighter. Tracking worst-case catches both "mean improves but outliers don't" and "uniform slow convergence."

---

### SC-4: Eigenvalue Stability

**Primary Criterion**: Max eigenvalue drift per iteration must satisfy `max_i |λ_i[N] - λ_i[N-1]| < 0.1 Ha` for N ≥ 4 (after initial settling).

**Diagnostic Tracking** (not failure criteria, but worth reporting):
- Core states (band 0): max drift < 0.01 Ha (should be rock-solid)
- Occupied bands (bands 1-81): max drift < 0.05 Ha (more sensitive due to Cu-3d)
- Conduction bands (bands 82-159): max drift < 0.1 Ha (less critical)

**Source**: Diagnostic 2 eigenvalue MAE = 0.0138 Ha vs CASTEP. Large drift (>0.1 Ha) indicates ZHEGVD rotation instability (the cascade failure mode from HANDOFF.md — band 0 drifted from -1.055 Ha to -11.94 Ha at iter-3).

**Verification**: Track max eigenvalue change per iteration. Assert max_drift < 0.1 Ha for iters 4-10. Report per-group max drift as diagnostic.

**Rationale**: We're testing iteration-to-iteration stability, not absolute accuracy. 0.1 Ha is appropriate for the max across all bands (catches catastrophic drift but allows small numerical variation). Per-group tracking flags red flags early (e.g., if core states drift > 0.01 Ha, that's concerning even if overall max is < 0.1 Ha).

---

### SC-5: No Cascade Signature

**Primary Criterion**: Band 0 eigenvalue must stay within **[-1.10, -1.01] Ha** (±0.05 Ha around CASTEP reference -1.055 Ha) across all 10 iterations.

**Diagnostic Tracking** (report but don't fail on):
- Band 81 (top of occupied, near Fermi): stays within ±0.1 Ha of CASTEP reference
- Max drift across Cu-3d cluster (bands 1-14): < 0.1 Ha from CASTEP reference

**Source**: HANDOFF.md cascade signature — band 0 drifted from -1.055 Ha to -11.94 Ha at iter-3 in the single-pass implementation. Staying within ±0.05 Ha indicates no cascade.

**Verification**: Assert -1.10 ≤ λ_0 ≤ -1.01 for all iterations. Report band 81 and Cu-3d cluster drift as diagnostic.

**Rationale**: Band 0 is the canary in the coal mine. 0.05 Ha tolerance is appropriate for a diagnostic test (if we were testing production code, we'd tighten to ±0.01 Ha, but for "does the outer loop prevent catastrophic cascade" where the failure mode is -11.94 Ha, ±0.05 Ha is sufficient). Diagnostic tracking catches both the cascade (band 0 failure) and Cu-3d rotation instability (cluster drift).

---

## Domain Terms Validated

All terms validated against CONTEXT.md:

- **S⁻¹-weighted residual**: `||r_b||_S^(-1) = √⟨r_b | S^(-1)·r_b⟩` where `r_b = H|ψ_b⟩ - λ_b·S|ψ_b⟩`
- **Chebyshev filter**: Polynomial subspace iteration applying T_m(H) to amplify eigencomponents
- **Rayleigh-Ritz**: Projects H into filtered subspace, solves H_sub·X = ε·S_sub·X (ZHEGVD)
- **USPP**: Ultrasoft pseudopotentials (S ≠ I, augmentation charge present)
- **Cu 3d cluster**: Bands 1-14, near-degenerate (eigenvalues within 0.07 Ha)
- **Conduction bands**: Bands 82-159 (above Fermi level)
- **Occupied bands**: Bands 0-81 (below Fermi level)

---

## Adjustments Made During Interview

1. **SC-1 tightened**: Original allowed indefinite plateau. Refined to require 20% reduction by iter-5 or iter-10.

2. **SC-2 added worst-case tracking**: Original only checked count. Refined to also track max conduction residual (catches subgroup plateau).

3. **SC-3 added worst-case tracking**: Original only checked mean. Refined to also track max occupied residual (catches outlier stagnation).

4. **SC-4 clarified "after iteration 3"**: Original was ambiguous. Refined to mean "iteration-to-iteration drift for N ≥ 4" with per-group diagnostic tracking.

5. **SC-5 fixed asymmetric range**: Original [-1.1, -1.0] Ha was asymmetric around -1.055 Ha. Refined to [-1.10, -1.01] Ha (±0.05 Ha symmetric). Added diagnostic tracking for band 81 and Cu-3d cluster.

---

## Test Scope Confirmed

- **10 outer iterations**: Sufficient to see convergence trends
- **All bands filtered every iteration**: No band-locking (Diagnostic 5 will test that)
- **Standard Rayleigh-Ritz**: No Harmonic RR (Diagnostic 4 will test that)
- **Starting state**: CASTEP-converged (isolates outer loop behavior)
- **Additional metrics tracked**: Per-band eigenvalue history, per-group residual evolution, worst-case residuals per group

---

## Next Steps

Write forensic TASKS.md with:
- Task breakdown for implementing the outer loop test
- Test code structure (outer loop, residual tracking, history storage)
- Acceptance commands
- Success criteria assertions (SC-1 through SC-5 as specified above)
