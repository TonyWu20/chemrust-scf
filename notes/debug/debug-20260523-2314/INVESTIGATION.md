# Investigation: §13 SCF Cascade — D-screening Comparison via CASTEP Dump

## Prior-Note Classification

Every numeric claim from `PLAN.md` hypothesis ledger and prior sessions, classified
per the ODD classification procedure (trace to origin, not phrasing).

### EXTERNAL claims (admissible as criteria)

| # | Claim | Source | Verification |
|---|-------|--------|-------------|
| E1 | iter-1 band-0 = −1.046 Ha, matches CASTEP band-0 (−1.055 Ha) within 0.05 Ha | `tests/ca_scf_convergence.rs::issue_11a_iter1_band0_matches_castep` + `Cu111_CO.bands:12` | Test passes against `.bands` fixture |
| E2 | Density code correct for CASTEP ψ: soft ratio 1.000000, aug ratio 1.000084 vs CASTEP F8 | `tests/ca_scf_convergence.rs::density_decomp_matches_castep_f8_same_inputs` | Controlled same-input experiment |
| E3 | RR mathematics correct: 160 bands within 0.05 Ha of `.bands` | `tests/rayleigh_ritz_validation.rs` (6 tests, all pass) | §12 validation suite |
| E4 | S⁻¹·S identity within 3.8e-15 | `s_inv_s_identity_test` | Machine-precision check |
| E5 | Mixing scheme Pulay + Kerker matches CASTEP | `Cu111_CO.param:47` | Direct file read |
| E6 | CASTEP band-0 = −1.05502310 Ha (converged) | `Cu111_CO.bands:12` | Fixture file, direct parse |
| E7 | CASTEP final energy = −24110.96665397 eV | `Cu111_CO.castep` | Fixture file |
| E8 | CASTEP converged D_screened per-ion matrices in `D_band_debug.dat` | `D_band_debug.dat` (last 18 blocks of 576) | CASTEP dump, ES24.16 precision |
| E9 | CASTEP species: 1=C (8 proj), 2=O (8 proj), 3=Cu (18 proj) | `Cu111_CO.cell` + `D_band_debug.dat` block headers | Cross-validated |
| E10 | `mixture_weight = 1.0` for all ions (no VCA) | `Cu111_CO.castep` (no VCA blocks) + `Cu111_CO.param` (no mix_amp species blocks) | Explicit absence check |
| E11 | `MIX_CHARGE_AMP = 0.5`, `MIX_CHARGE_GMAX = 1.5` | `Cu111_CO.param:37-39` | Fixture file |

### DERIVED claims (not admissible — computed by our pipeline)

| # | Claim | Origin | Why DERIVED |
|---|-------|--------|-------------|
| D1 | iter-2 band-0 = −0.869 Ha | `cascade_iter3_diagnostic` | Our SCF output |
| D2 | iter-3 band-0 = −11.94 Ha | `cascade_iter3_diagnostic` | Our SCF output |
| D3 | iter-1 density split: soft 29.7%, aug 70.3% | `cascade_iter3_diagnostic` | Our SCF output |
| D4 | iter-2 density split: soft 69.9%, aug 30.1% | `cascade_iter3_diagnostic` | Our SCF output |
| D5 | D_screened amax exploded from 5.8 → 49.97 → 318.9 Ha | Prior debug session | Our D-screening output with our V_eff |
| D6 | CASTEP F8 density split: soft 36.8%, aug 63.2% | `slurm_output_2291.txt` | CASTEP diagnostic output, but extracted by us — intermediate trust |

### HYPOTHESIZED claims (not admissible — inference/estimation)

| # | Claim | Basis | Why HYPOTHESIZED |
|---|-------|-------|-----------------|
| H1 | "Subspace-method rotation is inherent" → produces cascade | §13 RESOLUTION | No external corroboration; pure inference |
| H2 | V_eff drift between iter-1 and iter-2 amplifies through D-screening | §13 RESOLUTION proposal | Inference from cascade pattern; not tested |
| H3 | Q-skip at `q.norm_sqr() < 1e-60` is harmless | Source line inspection | Not tested against reference; existence confirmed but effect unknown |
| H4 | compute_screened_d formula matches CASTEP (sign, norm, conjugate, structure factor) | Side-by-side audit claim | Audit not yet performed; this session must verify |

## Reclassification from Prior Sessions

The failure-patterns.md entry `2026-05-23: eigenvector-rotation-cascade-divergence`
states the root cause as "Gram-Schmidt + Rayleigh-Ritz subspace method produces
rotated eigenvectors." This claim is **HYPOTHESIZED** — it was the working theory
in the prior session but never tested against an EXTERNAL anchor. The claim
`D_screened amax exploded from 5.8 → 49.97 → 318.9 Ha` is **DERIVED** (our
pipeline output with our V_eff).

The `compute_screened_d formula matches CASTEP` claim is reclassified from
implicit EXTERNAL to **HYPOTHESIZED** — the audit is claimed but not independently
verified. T2 in this session performs the actual verification.

## Claim dependency graph

```
E6 (band-0 anchor) ──── T1 (H-on-ψ check)
E8 (D dump) ────────── T2 (D-screening check)
E1 (iter-1 matches) ── T3 (V_eff substitution)
E2 (density correct) ─ T4 (density substitution)
E11 (mixing params) ── T5 (mixing probe)
```

All T1-T5 tests are anchored exclusively to EXTERNAL claims E1-E11. No DERIVED
or HYPOTHESIZED claim appears in any test discriminator.
