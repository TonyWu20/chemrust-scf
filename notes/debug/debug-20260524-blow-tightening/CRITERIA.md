# Anchor Criteria — b_low Tightening (Proposal §5)

**Date:** 2026-05-24
**Symptom:** Chebyshev filter admits out-of-window components into the 40-band tracked subspace; `subspace_projector_iter1_vs_castep` Cu-3d block ratio = 0.893 (target ~1.0).

## Fixture files

| Path | Contains |
|------|----------|
| `Cu111_CO.bands` | Per-band eigenvalues at each k-point, header includes ε_F |
| `Cu111_CO.castep` | CASTEP run log with total energy, smearing config |
| `Cu111_CO.check` | Binary checkpoint with USPP-S-orthonormal ψ ground truth |
| `Cu111_CO.param` | Run parameters: smearing_width, cut_off_energy, etc. |
| Fixture loader | `tests/fixtures/cu111_co.rs` — parses .check via `parse_bands_file` (line 118) and `Cu111CoFixture` |

## Success criteria

All criteria are EXTERNAL and falsifiable.

### Primary criterion (the discriminator)

**C1: Cu-3d block sum ≥ 12.2.**
- Statement: `subspace_projector_iter1_vs_castep` `block_sum(1, 14) ≥ 12.2` (ratio ≥ 0.94).
- Source: A5 (`Cu111_CO.check` S-orthonormal ψ); for two bases spanning the same k-dim subspace, `Σ_{a,b} |⟨ours_a | S | castep_b⟩|² = k = 13`.
- Current baseline: 11.6041 (ratio 0.893).
- Discriminator gap: |12.2 − 11.6041| = 0.6 vs band-0 single-block deviation 0.0079 → 76× signal/noise.
- Why 12.2 and not 12.5 or 13.0: the analyst plan (`~/.claude/plans/analyze-notes-proposals-14-eigensolver-r-precious-sunbeam.md` §6) and the PostRr post-mortem (line 60) both identify a ~3% iter-1 ceiling from filter pollution outside the 40-band window. b_low alone is bounded above by ~12.6 (ratio 0.97). A 12.5 gate would be near the ceiling; 12.2 sits at the realistic operating point.

### Secondary criterion (off-block leakage component isolation)

**C2: Cu-3d off-block leakage decreases.**
- Statement: `Σ_{a∈[1,14], b∉[1,14]} |M[a,b]|² < (baseline value × 0.5)`.
- Source: A5; component decomposition per Step 7.4 of the plan.
- Rationale: the Frobenius residual `13 − ‖P‖_F²` decomposes into in-cluster permutation (does not contribute), off-cluster-inside-40-band leakage (this fix may help), and outside-40-band leakage (this fix targets directly). If C1 lifts but C2 stays flat, then the lift comes from a permutation effect, not from reduced pollution — meaning the b_low fix is misattributed.
- Baseline value: to be measured by Step 7.0 before the fix.

### Advisory criteria (NOT gating)

The following are recorded but do not block plan success.

**A_C3: iter-3 band-0 within 0.1 Ha of A1.**
- Statement: `cascade_iter3_diagnostic_tight` reports iter-3 band-0 = -1.055 ± 0.1 Ha.
- Source: A1 (`Cu111_CO.bands:12` = −1.05502287 Ha).
- Current baseline: −11.94 Ha (drift 10.9 Ha).
- Why advisory: the analyst plan (§4 row "cascade_iter3 < 0.1 Ha") identifies V_eff drift as a separate driver. b_low alone cannot guarantee this; the PostRr attempt made it worse (drift 13.86 Ha). If this stays RED, the next session's direction is the analyst plan's Recommendation 3 (absolute-target Procrustes pinning).

**A_C4: total-energy Q2 within 1e-3 eV of A4.**
- Statement: `scf_converges_to_castep_energy_at_castep_tolerance`.
- Source: A4 (`Cu111_CO.castep` total energy = −24110.96665069 eV).
- Why advisory: Q2 sums all SCF iterations; even iter-1 fidelity gains may not propagate if V_eff drift dominates.

## Anchor freshness check

All anchor sources were re-verified before this session by direct file reads:
- `chebyshev.rs:371-388` and `chebyshev.rs:1406-1419` for the `b_low` formulas.
- `tests/ca_scf_convergence.rs:3475-3550` for the projector diagnostic.
- `tests/ca_scf_convergence.rs:1303-1432` for the bisect harness.

The proposal §5 stale claim (b_low = max_veff + 2.0) has been verified-and-corrected; the actual operating formula at iter-1 is `max_veff.max(0.0)`.
