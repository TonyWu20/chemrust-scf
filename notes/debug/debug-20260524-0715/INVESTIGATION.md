# Prior Investigation Notes: Issue 15 — Iter-1 total energy off by factor of Ω

## Source: `notes/open-followups.md` Issue 15

### Claim 1: "Energy is off by ~36,500×" — DERIVED
- Source: `notes/open-followups.md:956-958`
- Value: `[Q1] iter-1 total energy = -32407458.84 Ha = -881851807.02 eV`
- Classification: **DERIVED** — this is an output of our own buggy pipeline
- Admissible as criterion? **No**

### Claim 2: "Magnitude matches cell volume in Bohr³ (Ω ≈ 22,315)" — HYPOTHESIZED
- Source: `notes/open-followups.md:961-962`
- Value: Ω ≈ 22,315 Bohr³
- Classification: **HYPOTHESIZED** — derived from scaling argument; not independently corroborated
- Admissible as criterion? **No**

### Claim 3: "CASTEP reference: −886.06 Ha = −24110.97 eV" — EXTERNAL
- Source: `notes/open-followups.md:957`
- Also: `tests/fixtures/cu111_co.rs:26` — `REFERENCE_ENERGY_EV = -24110.96665069`
- Origin: Cu111_CO.castep file line 326 (CASTEP output)
- Classification: **EXTERNAL** — from CASTEP reference output
- Admissible as criterion? **Yes**

### Claim 4: "Suspect: e_hartree formula double-counts Ω" — HYPOTHESIZED
- Source: `notes/open-followups.md:964-979`
- Claim: `e_hartree_raw = Σ ρ[i] × V_H[i] × d_v` with `d_v = Ω/N` gives Ω × correct
- Classification: **HYPOTHESIZED** — not verified against CASTEP source or independent computation
- Will be verified in this session

### Claim 5: "Same pattern likely applies to rho_vxc" — HYPOTHESIZED
- Source: `notes/open-followups.md:981`
- Classification: **HYPOTHESIZED**

### Claim 6: "Q1 threshold = 20 mHa" — EXTERNAL
- Source: `tests/fixtures/cu111_co.rs:40` — `DRIFT_TOLERANCE_HA = 2e-2`
- Origin: fixture calibration from independent diagnostic (T3 cascade_with_castep_veff_substitution)
- Classification: **EXTERNAL** — from controlled experiment with CASTEP V_eff substitution
- Admissible as criterion? **Yes, once energy bug is fixed**

## Summary

| Claim | Classification | Can be criterion? |
|-------|---------------|-------------------|
| Computed energy = −32M Ha | DERIVED | No |
| Ω ≈ 22,315 Bohr³ | HYPOTHESIZED | No (must verify from fixture) |
| CASTEP ref = −886.06 Ha | EXTERNAL | Yes |
| d_v = Ω/N causes Ω factor | HYPOTHESIZED | To be verified |
| rho_vxc also affected | HYPOTHESIZED | To be verified |
| Q1 gate = 20 mHa | EXTERNAL | Yes (post-fix) |
