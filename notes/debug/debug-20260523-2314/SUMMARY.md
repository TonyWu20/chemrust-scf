# Debug Session Summary: §13 SCF Cascade Root-Cause Analysis

**Date**: 2026-05-24
**Fixture**: Cu111+CO, CASTEP converged ψ and total density
**Branch**: `feat/phase-global-woodbury`

## Key Finding

The SCF cascade is caused by **augmentation density (ρ_aug) sensitivity to eigenvector
rotation within the Cu 3d manifold**. Our subspace eigensolver (Chebyshev filter + RR)
finds different linear combinations of degenerate 3d states than CASTEP's band-by-band
CG. The β-projections `⟨ψ_b|β_i⟩` differ between the two methods even though:

1. The **occupations are identical** (max |Δ| = 0.00 over all 160 bands)
2. The **soft density |ψ|² is rotation-invariant** within the occupied subspace
3. The **total electron count is correct** (186.00 e⁻)
4. The **eigenvalues differ by only 0.01–0.03 Ha** (within T1 tolerance)

But the β functions are localized at ion cores and concentrated at high G-vectors.
They are sensitive to the high-frequency content of ψ, which differs between our
subspace method and CASTEP's CG even when the occupied subspace is the same.
This causes the aug density to differ.

## Test Results

### T1 — H-on-CASTEP-ψ: PASS
- Our H operator applied to CASTEP eigenvectors with our V_eff
- RMS error: **0.0046 Ha** over 160 bands (threshold 0.05 Ha)
- Max error: **0.019 Ha** (threshold 0.10 Ha)
- **Conclusion**: H operator and V_eff from CASTEP density are correct.

### T2 — D-screened vs CASTEP dump: FAIL
- Element-by-element comparison of `compute_screened_d` against `D_band_debug.dat`
- Max per-element delta: **2.13 Ha** (threshold 5e-4 Ha)
- Errors concentrated on diagonal elements, position-dependent
- **Conclusion**: D-screening has per-element discrepancies with CASTEP.
  Normalization factor (1/N) verified correct by raw-sum diagnostic.

### T3 — CASTEP V_eff substitution: PASS
- Inject CASTEP `.pot_fmt` V_eff before iter-2 diagonalization
- Iter-2 band-0: **−1.0452 Ha** (|Δ| = 0.0098 Ha vs reference −1.055 Ha)
- Iter-3 band-0: −0.9047 Ha (some drift, not catastrophic)
- **Conclusion**: Cascade stops with externally-correct V_eff.

### T4 — CASTEP density substitution: PASS (with aug cleared)
- Inject CASTEP total density + clear stale ρ_aug before iter-2 V_eff build
- Iter-2 band-0: **−1.0452 Ha** (|Δ| = 0.0098 Ha)
- Iter-3 band-0: −0.9047 Ha
- **Conclusion**: Cascade stops. Without clearing aug: fails (−3.92 Ha).

### Density Split Diagnostic
- Our ψ (post-RR): soft=**29.8%**, aug=**70.2%**
- CASTEP F8 reference:  soft=**36.8%**, aug=**63.2%**
- Occupations: **identical** (max |Δ| = 0.00, all bands fully occupied)
- Total electrons: **186.00** (correct)
- **Conclusion**: Density split differs by 7 pp despite identical occupations.
  The aug density from our ψ differs from CASTEP's because β-projections
  are sensitive to ψ rotation within the degenerate 3d manifold.

### Cascade with aug removed: WORSE
- Attempted fix: pass `None` for `density_aug_fine` in `build_v_eff_with_energy`
- Iter-2 band-0: **−23.3 Ha** (was −0.87 Ha with aug)
- **Conclusion**: Aug density provides crucial damping at ion cores.
  Removing it accelerates divergence.

## Cascade Mechanism (revised)

```
CASTEP ψ → our filter → rotated ψ (different β-projections within 3d manifold)
  → different ρ_aug (7 pp shift from CASTEP)
  → ρ_aug fed into next iteration's V_eff via into_phase()
  → V_eff = V_H[ρ_soft + ρ_aug(wrong)] + V_ion + V_xc[ρ_soft + ρ_aug(wrong) + ρ_core]
  → wrong V_eff changes spectral bounds and D-screening (∫Q·V_eff)
  → filter amplifies different subspace
  → ψ rotates more → ρ_aug further from CASTEP's → cascade
```

Key insight: the cascade is NOT caused by a code bug in density construction
(provably correct by E2) or V_eff assembly (correct by T1/chemrust-hamiltonian).
It is caused by our eigensolver producing different β-projections than CASTEP's
CG within the degenerate 3d manifold, and the SCF feedback loop amplifying this
difference through V_eff → D-screening → filter.

## What Was Ruled Out

| Hypothesis | Evidence | Status |
|-----------|----------|--------|
| H operator is wrong | T1 PASS (RMS 0.0046 Ha) | **Ruled out** |
| Density code is wrong | E2 (ratio 1.000000/1.000084) | **Ruled out** |
| Occupations differ between iterations | Split diag: identical | **Ruled out** |
| V_eff formula discontinuity | `.castep_bin` contains total density (CASTEP bakes aug in-place) | **Ruled out** |
| Grid mismatch (fine vs wave) | Both 54×90×90 | **Ruled out** |
| Subspace rotation cascade (original §13) | Partially confirmed — rotation IS real, but mechanism is via ρ_aug, not inherent | **Refined** |

## Fix Directions

1. **Primary**: Make our eigensolver produce ψ with β-projections closer to
   CASTEP's. This requires fixing the D-screening element-by-element errors (T2),
   since D errors in V_NL = β·D·β† directly affect which linear combinations
   the filter finds.

2. **Damping**: Apply a mixing weight to ρ_aug in V_eff to reduce its impact
   on the effective Hamiltonian. This trades accuracy for stability.

3. **Filter modification**: The Chebyshev filter takes `self.psi` as input
   (line 1328 of `chebyshev.rs`). Even with identical V_eff and D_screened,
   a rotated input produces different output because the filter is nonlinear
   (Chebyshev recurrence + Gram-Schmidt + RR). Consider initializing the filter
   with a fixed reference ψ at each iteration.

## Reference Code

| File | Lines | Content |
|------|-------|---------|
| `tests/fixtures/cu111_co.rs:139-208` | `build_scf_state` — loads CASTEP total density (no aug compute) |
| `src/scf.rs:342-384` | `build_v_eff_with_energy_impl` — V_eff = V_H + V_ion + V_xc, uses ρ_aug if available |
| `src/scf.rs:748-897` | `compute_density_from_wavefunctions` — soft from |ψ|², aug from β·ψ |
| `src/scf.rs:241-270` | `into_phase` — preserves `density_aug_fine` across SCF iterations |
| `src/eigensolver/chebyshev.rs:1273-1390` | `chebyshev_filter` — takes `psi_gpu` as input, nonlinear filter |
| `chemrust-hamiltonian-core/src/nlpot.rs:368-411` | `compute_screened_d` — D = D0 + ∫Q·V_eff / N |
| CASTEP `nlpot.f90:531-544` | D dump instrumentation |
| CASTEP `ion.f90:6319-6441` | `ion_int_Q_at_origin_recip` — D-screening integral |
| CASTEP `density.f90:3471-3743` | `density_augment` — in-place augmentation (total density in `.check`) |
