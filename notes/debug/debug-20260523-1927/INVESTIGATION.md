# Investigation: Issue #12 — Rayleigh-Ritz Zero Validation

## Symptom

135% electron count drift (55 e⁻ → 130 e⁻ vs expected 186 e⁻) between SCF iterations.
Density split ratio wrong (ρ_PW 67.4% of CASTEP, ρ_aug 119%).
Multiple downstream bugs traced to upstream RR, but RR itself never validated.

## Prior Note Claims — Classified

| Claim | Source | Class | Admissible? |
|-------|--------|-------|-------------|
| "135% electron count drift" | open-followups.md §12 | DERIVED | No — from our SCF pipeline |
| "ρ_PW 67.4% of CASTEP" | open-followups.md §12 | DERIVED | No — compares our RR output vs CASTEP |
| "ρ_aug 119% of CASTEP" | open-followups.md §12 | DERIVED | No — compares our augmentation vs CASTEP |
| "55 e⁻ → 130 e⁻ drift" | scf.rs diagnostic | DERIVED | No — from our diagnostic formula |
| "iter-1 band-0 = −1.046 Ha" | open-followups.md §11 | DERIVED | No — from our RR pipeline |
| "CASTEP band-0 = −1.055 Ha" | Cu111_CO.bands | EXTERNAL | Yes |
| "CASTEP last-band = 0.115 Ha" | Cu111_CO.bands | EXTERNAL | Yes |
| "160 reference eigenvalues" | Cu111_CO.bands header | EXTERNAL | Yes |
| "186 electrons" | Cu111_CO.bands header | EXTERNAL | Yes |
| "Fermi energy −0.122443 Ha" | Cu111_CO.bands header | EXTERNAL | Yes |
| "SCF diagnostic formula wrong" | §11b resolution | DERIVED | No |
| "density code correct" | §8 resolution + density_decomp test | DERIVED | No (depends on our RR ψ being correct) |

### Key Observation

The §8 test `density_decomp_matches_castep_f8_same_inputs` feeds CASTEP's ψ through our
density code and confirms density code is correct. This is EXTERNAL-anchored but only proves
the density code is correct given correct inputs. The open question is whether our RR produces
ψ that matches CASTEP's converged ψ in normalization and orthogonality.

### Memory entries — none carry all three required fields (verification granularity, file:line citation, counter-example scope), so all are classified as HYPOTHESIZED.

## Diagnostic Chain Identified

The diagnostic path requires:
1. Expose H_sub, S_sub, X from RR (infrastructure change — done)
2. Test mathematical properties of those matrices (Tests 1-4)
3. Test all eigenvalues vs CASTEP anchor (Test 5)
4. Test S-normalization of rotated ψ_new (Test 6)
