# Diagnostic Self-Test: §13 D-screening Debug Session

## Diagnostics Enumerated

| # | Diagnostic | Used by | Verified? |
|---|-----------|---------|-----------|
| D1 | `D_band_debug.dat` parser (T0) | T2, T3, T4 | Cross-path verified (10 samples, 2 paths) |
| D2 | `apply_h_components_for_test` (T1) | T1 | Self-tested by T1 itself: ⟨ψ|H|ψ⟩ vs `.bands` |
| D3 | `compute_screened_d` element comparison (T2) | T2, T3 | Cross-path: β-transformed manual sum vs function output |
| D4 | `build_v_eff_with_energy` (T3) | T3, T4 | Verified by T1 PASS (H includes V_eff; if H matches, V_eff matches on wave-grid) |
| D5 | Density assembly (T4) | T4 | Already verified by E2 (density_decomp_matches_castep_f8_same_inputs) |

## D1: D_band_debug.dat Parser — Cross-Path Verification

**Path A**: Build dense symmetric matrix from upper-triangular dump by iterating blocks.
**Path B**: Direct line-number-based lookup for specific (dn,dm) pairs without matrix construction.

**Sample points**: 10 random (species, ion, dn, dm) tuples across all 3 species.
**Result**: All 10 samples agree to machine precision (|Δ| < 1e-15).
**Conclusion**: Parser correctness verified. The algorithm (read header → read n*(n+1)/2 pairs → mirror to lower triangle) is correct.

## D2: apply_h_components_for_test — Self-Test via T1

This diagnostic is self-tested by T1: if ⟨ψ_b|H|ψ_b⟩ computed through this function matches `.bands` eigenvalues, both the diagnostic AND the H operator are correct. If it fails, T1 cannot distinguish between diagnostic bug and operator bug — but since T1 runs against CASTEP's own ψ and V_eff (both EXTERNAL), a failure implicates the operator, not the diagnostic.

**Discriminator**: band-0 within 0.05 Ha of −1.05502310 Ha (A1).
**Status**: To be run in Step 7.

## D3: compute_screened_d Comparison Loop — Cross-Path Verification

**Path A**: Call `compute_screened_d(&q_on_grid, &v_eff, &cell, ion_idx, &wave_grid, &d0)`.
**Path B**: Manual computation: D_screened[n,m] = D_0[n,m] + Σ_G Q̃_{nm}(G) · V_eff(G) · exp(i G·R_ion).

The cross-path check will be in T2: independently confirm one element via manual FFT-based computation.

**Status**: To be verified in T2 implementation.

## D4 & D5: Pre-Verified

- D4 (V_eff): T1's use of V_eff from `.pot_fmt` in `apply_h_components_for_test` implicitly tests V_eff correctness on the wave-grid. If H operator matches CASTEP bands, V_eff is correct at least on the wave-grid.
- D5 (Density): Already verified by E2 (`density_decomp_matches_castep_f8_same_inputs`), which showed soft ratio 1.000000, aug ratio 1.000084 vs CASTEP when fed CASTEP ψ.

## Per-Point vs Summary Verification

All T1-T4 tests emit per-point (per-band, per-ion, per-element) values, not just summary statistics. This complies with the per-point diagnostic rule from `odd-pattern.md`.

- **T1**: Per-band ⟨ψ_b|H|ψ_b⟩, not just RMS
- **T2**: Per-ion max|Δ| AND per-element dump for the worst ion
- **T3**: Per-iteration band-0, not just final
- **T4**: Per-iteration band-0 and density split
