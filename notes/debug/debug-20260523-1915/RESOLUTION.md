# Resolution: SCF Divergence — Eigenvector Rotation Cascade

**Symptom**: SCF delivers significantly different soft density/augmented density
split compared to CASTEP F8 dump. Previous work proved density code is correct
for CASTEP wavefunctions (ratios: soft=1.000000, aug=1.000084). Cascade test
shows SCF diverges catastrophically at iter-3 (band-0 eigenvalue = -11.94 Ha
vs CASTEP -1.055 Ha).

**Session date**: 2026-05-23
**Branch**: debug/density-split-audit
**Worktree**: /home/tony/programming/chemrust-scf-debug

## Root cause

The Chebyshev filter + Rayleigh-Ritz subspace method produces eigenvectors that
are energetically correct (eigenvalues match CASTEP within 0.05 Ha at iter-1)
but rotated within degenerate manifolds (avg eigenvector overlap with CASTEP =
0.108 at ndeg=8, 0.252 at ndeg=0). This rotation is inherent to subspace
methods vs CASTEP's band-by-band CG electronic minimization.

The rotation cascade mechanism:
1. **Iter-1**: Gram-Schmidt S-orthonormalization + Rayleigh-Ritz picks different
   eigenvectors within the Cu 3d degenerate manifold (bands 2-10, 0.02 Ha spread)
   than CASTEP's CG. The Chebyshev filter adds additional rotation for bands
   near the occupied subspace boundary. Density from rotated eigenvectors has
   correct total charge (186.0 e⁻) but wrong soft/aug split (29.7/70.3% vs
   36.8/63.2% F8).
2. **Iter-2**: The wrong density → wrong V_eff at ion centres → wrong D-screening
   → wrong S⁻¹·H operator → catastrophic eigenvector evolution. Density split
   INVERTS (soft_frac 69.9%). Eigenvalues begin to drift (band-0 -0.869 Ha).
3. **Iter-3**: Complete collapse (band-0 -11.94 Ha, eigenvectors orthogonal to
   CASTEP, all bands have Σ|c_G|² ≈ 0.036).

## Key diagnostic findings

| Test | Result |
|------|--------|
| `ndeg_zero_with_castep_psi_matches_bands` | PASS (eigenvalues match) |
| `eigenvector_overlap_vs_castep_after_filter` | avg |⟨our|CASTEP⟩|² = 0.108 (ndeg=8) |
| `subspace_overlap_diagnostic` | ndeg=0: 0.252, ndeg=8: 0.108 |
| `cascade_iter3_diagnostic` | Iter-1 ✓, Iter-2 drift, Iter-3 collapse |
| `issue_11a_iter2_lastband_does_not_overshoot` | PASS (0.586 Ha, gate 1.0) |

## Hypothesised next step: D-screening comparison

The cascade is driven by V_eff change at ion centres between iter-1 (CASTEP
converged V_eff) and iter-2 (our V_eff from rotated eigenvectors). The
D-screening integral D_screened = D_0 + ∫Q·V_eff is where V_eff differences
at ion centres are amplified. Key diagnostic to write:

1. **D_screened comparison test**: Run VnlBatchData::precompute with both
   CASTEP V_eff (from .pot_fmt) and our iter-1 V_eff. Compare per-ion
   D_screened matrices. If they differ significantly (>10%), the V_eff change
   at ion centres is the amplification point.

## Date
2026-05-23
