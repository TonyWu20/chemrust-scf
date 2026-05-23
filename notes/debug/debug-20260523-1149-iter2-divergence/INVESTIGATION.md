# Investigation — Prior-Note Classification

**Symptom slug:** `iter2-divergence`
**Date:** 2026-05-23
**Reference log:** `/tmp/scf-diag-global-woodbury-0523-1021.log`
**Branch:** `feat/phase-global-woodbury` (HEAD `ec2387d`)

This file classifies every numeric claim from §11 of `notes/open-followups.md`,
the §10 resolution at `debug-20260523-0916-iter1-filter-operator-mismatch/RESOLUTION.md`,
and the iter-2 reference log. Per the skill's classification rule, only
EXTERNAL or corroborated-with-tight-gate claims may become success criteria.

## Claim classification

| Claim | Class | Why |
|-------|-------|-----|
| "iter-1 band-0 = −1.046 Ha" | DERIVED | Our pipeline output |
| "iter-2 band-0 = −0.864 Ha" | DERIVED | Our pipeline output (the symptom we're trying to falsify against) |
| "iter-2 last band = 1.952 Ha" | DERIVED | Our pipeline output (the symptom we're trying to falsify against) |
| "iter-3 band-0 = −12.68 Ha", "iter-3 last band = 0.286 Ha" | DERIVED | Our pipeline output |
| "iter-2 V_eff range = 20.26 Ha" | DERIVED | Our diagnostic prints in `scf.rs:486-492` |
| "Cu D_screened amax = 31.4 Ha (iter-2), 315 Ha (iter-3)" | DERIVED | Our diagnostic prints; not yet self-tested |
| "iter-1 smooth ρ ≈ 68 e⁻, aug ρ ≈ 118 e⁻" | DERIVED | Our diagnostic prints |
| "iter-2 smooth ρ ≈ 181 e⁻, aug ρ ≈ 5 e⁻" | DERIVED | Our diagnostic prints |
| "iter-2 V_eff range jumps from 8.69 → 20.26 Ha (2.3×)" | DERIVED | Our diagnostic prints |
| "Lanczos α[0..3] iter-1 = [8.80, 7.50, 7.42, 7.40]; iter-2 = [8.79, 7.49, 7.41, 7.40]" | DERIVED, **corroborated** | Independently verifiable via Gershgorin formula on V_eff — one of the strongest signals: H is essentially unchanged between iterations |
| "iter-1 b_low = 0.089 Ha (max_veff source)" | DERIVED | But corroborated by the §10 resolution that explicitly fixed the b_low bootstrap to use `max_veff` on first iteration |
| "iter-2 b_up = 20.80 Ha, b_low = 0.13 Ha" | DERIVED | Our diagnostic prints |
| "T_8 contrast ratio at filter window ≈ 32,000×" | HYPOTHESIZED | Textbook formula; the value is theoretically correct *if* the operator being filtered matches the spectrum the bounds describe |
| "filter window [0.13, 20.80] Ha should damp anything ≥ 0.13 Ha" | HYPOTHESIZED | Theoretical assertion not measured as polynomial response on real iter-2 inputs |
| "ndeg=8 is insufficient for non-converged starting subspace" (§11 hypothesis A) | HYPOTHESIZED | Stated as a candidate; not measured |
| "iter-1 RR output is not exactly S-orthonormal" (§11 hypothesis B) | HYPOTHESIZED | Stated as a candidate; not measured |
| "CASTEP `.bands` band-0 = −1.0550 Ha" | **EXTERNAL** | `tests/fixtures/cu111_co.rs` parses `.bands` from the CASTEP fixture |
| "CASTEP `.bands` band 9 = −0.4681 Ha" | **EXTERNAL** | Same source |
| "CASTEP `.bands` last (band-159) ≈ 0.115 Ha" | **EXTERNAL** | Same source |
| "CASTEP F8 soft ρ integral = 1,527,256" (= 68.44 e⁻ × Ω in N_e×Ω convention) | **EXTERNAL** | F8 instrumentation runs of CASTEP from §8 RESOLUTION |
| "CASTEP F8 aug ρ integral = 2,623,313" (= 117.56 e⁻ × Ω) | **EXTERNAL** | F8 instrumentation runs |
| "CASTEP `.castep_bin` total density = 4,150,570" (= 186 e⁻ × Ω) | **EXTERNAL** | `.castep_bin` parser |
| "CASTEP `.pot_fmt` V_eff range ≈ 8.69 Ha at converged state" | **EXTERNAL** | `.pot_fmt` fixture |
| "‖S⁻¹·S·ψ − ψ‖_∞ = 3.8e-15" (post-§10 fix) | DERIVED, **corroborated** | Per skill memory rule auto-classifies as DERIVED, but `tests/ca_scf_convergence.rs::s_inv_s_identity_test` is a tight 1e-10 gate, so admissible as evidence the global Woodbury S⁻¹ is exact to machine epsilon |
| "Density code matches CASTEP F8 to 0.0084% on same-input experiment" | **EXTERNAL** | `density_decomp_matches_castep_f8_same_inputs` test gate |
| "Mode B (SinvHKeepHEig) wins iter-1 SC-4-tight" | DERIVED | But the §10 RESOLUTION embedded the result as a regression gate (`iter1_filter_mode_sweep`); admissible as evidence Mode B is correct *for iter-1* — explicitly NOT generalizable to iter-2 |
| "Iter-3 bootstraps b_low = 1.95 Ha because of iter-2 last band" | HYPOTHESIZED | Mechanically obvious from `chebyshev.rs:1411` `eig[last]` selection, but the consequence (cascade) is a hypothesis until measured at iter-3+ post-fix |
| "§10 G2 resolved cross-ion B^H·B imaginary parts via Vec<f64>→Vec<CudaComplex>" | DERIVED, **corroborated** | §10 RESOLUTION; corroborated by the s_inv_s_identity_test passing at 1e-10 |

## Admissible-for-criteria summary

EXTERNAL or corroborated:
- CASTEP `.bands` band-0 −1.0550 Ha, band-9 −0.4681 Ha, last 0.115 Ha
- CASTEP F8 soft 68.44 e⁻, aug 117.56 e⁻, total 186 e⁻
- CASTEP `.pot_fmt` V_eff range 8.69 Ha
- s_inv_s_identity gate at 1e-10 → Woodbury S⁻¹ exact
- density_decomp gate at 1% → density code correct on same-input
- Mode B winner on iter-1 SC-4-tight

DERIVED (admissible only as discriminator targets to falsify, not as anchors):
- Iter-2 band-0 = −0.864 Ha, last = 1.952 Ha → these define the "current
  wrong" state but cannot anchor "what right looks like"; the anchors for
  "right" come from CASTEP fixtures alone.

HYPOTHESIZED (NOT admissible as criteria; admissible only as candidate
hypotheses for Step 7 to test):
- ndeg=8 insufficient
- RR output not S-orthonormal
- T_8 polynomial contrast ratio is what it should be
- Filter window damping behavior matches theory
