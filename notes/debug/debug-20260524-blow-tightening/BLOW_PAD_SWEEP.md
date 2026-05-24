# b_low Pad Sweep Results

**Date:** 2026-05-24
**Mechanism:** `CHEMRUST_BLOW_PAD` env-var override of the Rayleigh-quotient-derived b_low (b_low = max_a ⟨ψ_a | H | ψ_a⟩ + pad). Implemented at chebyshev.rs:L1540-L1582.

## Sweep table (iter-1, CASTEP ψ as input)

EXTERNAL anchor: A5 (CASTEP ψ from `Cu111_CO.check`); diagnostic `subspace_projector_iter1_vs_castep`.

| pad (Ha) | b_low (Ha) | Cu-3d/13 ratio | 0..30/30 ratio | 0..40/40 ratio | Run time |
|----------|------------|----------------|----------------|----------------|----------|
| —        | 0.0894 (max_veff baseline, before fix) | **0.893** | 0.940 | 0.939 | (historical) |
| -0.05    | 0.0708     | **0.893**      | 0.940          | 0.939          | 195 s    |
| 0.0      | 0.1208     | **0.891**      | 0.938          | 0.937          | 207 s    |
| +0.5     | 0.6208     | **0.797**      | 0.904          | 0.878          | 182 s    |

`max_h_eig = 0.1208` Ha — the highest tracked-band Rayleigh quotient on input CASTEP ψ. For reference: CASTEP eigenvalues span [-1.06, +0.13] Ha for the 160 tracked bands.

## Observations

1. **b_low at or below max_veff (0.07-0.12 Ha)**: ratio is FLAT at 0.893 for Cu-3d, 0.94 for the 0..30 and 0..40 windows. The differences are within noise (rounding in the 3rd decimal).
2. **b_low above tracked subspace (0.62 Ha)**: ratio gets WORSE. This is the case where all tracked bands sit BELOW b_low → filter amplifies all of them but with similar weight (small dynamic range across tracked subspace).
3. **Even moving b_low from below max_h_eig (0.07) to slightly above (0.12)** — a meaningful 0.05 Ha shift that crosses the band-edge — produces no change in ratio.

## Conclusion: tightening b_low does NOT lift the projector ratio.

The proposal §14 §1.4 attribution — "~6-11% Chebyshev filter pollution from outside-window high-G components" — is **falsified** for the iter-1 / CASTEP-ψ-input case. The 0.893 baseline is NOT a filter-window-discrimination problem.

## What this means for the proposal

The 0.893 → 1.0 gap (10.7% Frobenius loss in the Cu-3d 13-band block) must come from one of:

| Mechanism | Status | Notes |
|-----------|--------|-------|
| Chebyshev filter window choice (proposal §14 §1.4) | **FALSIFIED by this sweep** | Ratio insensitive to b_low across the relevant range [0.07, 0.62] Ha |
| ZHEGVD in-block permutation/rotation (proposal §14 §1.3) | PARTIALLY ADDRESSED by PostRr — iter-1 floor 0.97 ceiling | But cascade-amplification disqualifies it as production fix |
| Numerical precision in S-orthonormalization under low-PW-norm Cu 3d bands | NEW HYPOTHESIS | test_2 measured ‖ψ_3d‖² ≈ 0.14 (vs 1.0 for plain L2-orthonormal); the discriminator value 0.893 ≈ 11.6/13 may be a USPP normalization artifact |
| Augmentation density Q_lm,nm contribution mismatch | NOT INVESTIGATED | The Cu 3d shell's `Q` matrices are the source of the low PW norm; if our diagnostic misuses Q vs CASTEP's S definition, the projector under-reports overlap |

The user-supplied memory entry `relative_target_procrustes_feedback` already records the PostRr / V_eff coupling finding. Combined with this sweep result, the right next direction is:

**Investigate whether 0.893 is a normalization artifact rather than a genuine span loss.**

Specifically: compute `|⟨ψ_castep_a | S | ψ_castep_b⟩|²` for the Cu-3d 13×13 block (this is the diagonal of the S-orthonormality matrix for CASTEP ψ alone). For perfectly S-orthonormal ψ, the block sum should be 13.0. If CASTEP's `.check` ψ is stored with truncated precision such that the block sum is, e.g., 11.6, then 0.893 is not pollution at all — it's just CASTEP's stored-precision floor.

The diagnostic self-test at 2026-05-24 measured `max_diag_err = 5.94e-8` for the diagonal alone, so single-band normalization is fine. But the test did NOT measure the off-diagonal block sums systematically. If the 0.893 number is what CASTEP itself would produce when measuring `Σ_{a,b in 1..14} |⟨ψ_a|S|ψ_b⟩|²` for its own ψ, then we've been chasing a phantom.

## Code change to KEEP or REVERT

The Rayleigh-quotient b_low override at chebyshev.rs:1540-1582 produces NO measurable difference in projector ratio at iter-1. Three options:

1. **REVERT**: remove the override; production stays at `max_veff` (the current code). Reason: the fix has no measurable physical effect.
2. **KEEP as default with pad=0**: hindsight-correct (b_low is the highest tracked Rayleigh quotient, which is the principled value per Zhou §5 step 11 derived from input ψ rather than stale RR output) but doesn't change behavior at iter-1. May help at iter-2+ where input ψ has drifted.
3. **KEEP gated behind env var**: keep the infrastructure for future sweeps (e.g., when investigating iter-2+ behavior or testing with different fixtures); default `CHEMRUST_BLOW_PAD` to a value that reproduces current behavior (e.g., make pad large enough that we hit the b_up*0.95 clamp, effectively disabling).

Recommendation: option 2 (KEEP with pad=0 default). The change is architecturally cleaner (no "stale label" weak point per scf.rs:571-577 comment), self-derived, and doesn't regress iter-1 behavior. If iter-2+ shows different sensitivity, this gives us the right knob.
