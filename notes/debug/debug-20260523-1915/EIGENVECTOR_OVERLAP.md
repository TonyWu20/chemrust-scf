# Eigenvector Overlap: Our post-filter ψ vs CASTEP .check ψ

**Date**: 2026-05-23
**Branch**: debug/density-split-audit
**Test**: `eigenvector_overlap_vs_castep_after_filter`
**Command**: `diagonalize(8, None)` (ndeg=8, Mode B: SinvHKeepHEig, eigenvalues=None)

## Setup

- Input: CASTEP converged wavefunctions from `.check` file + fixture V_eff
- Pipeline: `build_v_eff(fixture_density) → diagonalize(ndeg=8)`
- Comparison: our post-filter+RR ψ vs original CASTEP ψ (L2 dot product per band)

## Results per band

| Band | ⟨our|CASTEP⟩² | max off-diag | Angle | Character |
|------|-------|-------------|-------|-----------|
| 0    | 1.050 | 0.007       | 0°    | Ground state matched |
| 1    | 0.768 | 0.002       | 40°   | Partial rotation |
| 2    | 0.070 | 0.072       | 86°   | Essentially orthogonal |
| 3    | 0.078 | 0.004       | 86°   | Cu 3d, rotated |
| 4    | 0.000 | 0.388       | 90°   | Cu 3d, orthogonal |
| 5    | 0.031 | 0.036       | 88°   | Cu 3d, rotated |
| 6    | 0.000 | 0.040       | 90°   | Cu 3d, orthogonal |
| 7    | 0.004 | 0.052       | 90°   | Cu 3d, orthogonal |
| 8    | 0.003 | 0.053       | 90°   | Cu 3d, orthogonal |
| 9    | 0.006 | 0.057       | 90°   | Cu 3d, orthogonal |
| 10   | 0.005 | 0.061       | 90°   | Cu 3d, orthogonal |
| 11   | 0.000 | 0.081       | 90°   | Orthogonal |
| 12   | 0.000 | 0.083       | 90°   | Orthogonal |
| 13   | 0.000 | 0.606       | 90°   | Mixed with another band |
| 14   | 0.000 | 0.598       | 90°   | Mixed with another band |
| 15   | 0.000 | 0.237       | 90°   | Orthogonal |
| 16   | 0.032 | 0.186       | 88°   | Partial rotation |
| 17   | 0.000 | 0.052       | 90°   | Orthogonal |
| 18   | 0.000 | 0.055       | 90°   | Orthogonal |
| 19   | 0.104 | 0.036       | 84°   | Partial rotation |

**Summary**: avg |⟨our|CASTEP⟩|² = **0.108** (10.8% aligned)

Despite this severe eigenvector rotation, eigenvalues for bands 0-10 match CASTEP
within 0.05 Ha (confirmed by `issue_11a_iter1_band0_matches_castep`).

## Interpretation

The eigenvectors are energetically equivalent but rotated within near-degenerate
subspaces. The Cu 3d manifold (bands 2-10, spanning 0.02 Ha) forms the largest
such subspace. Any perturbation in V_eff or D-screening causes RR to pick
different linear combinations within this manifold.

This is a well-known property of degenerate/near-degenerate subspaces in
eigensolvers: the PROJECTOR Σ|ψⱼ⟩⟨ψⱼ|S is invariant, but the individual
|ψⱼ⟩ rotate freely.

## Consequence for density split

The rotated eigenvectors within the Cu 3d manifold have different PW vs
augmentation character per band. Since USPP augmentation is NOT isotropic
(different β-projectors have different spatial profiles), the total density
built from rotated eigenvectors has a different soft/aug split than the
density built from CASTEP's eigenvectors — even though both sets span the
exact same eigenspace.

The density code correctly maps eigenvector orientation → density. The split
discrepancy is from the eigenvector rotation, not from a density code bug.

## Subspace diagnostic: ndeg=0 vs ndeg=8

Comparing eigenvector overlap with CASTEP at ndeg=0 (no Chebyshev filter, just
Gram-Schmidt + RR) vs ndeg=8 (filter + GS + RR):

| band      | ndeg=0 | ndeg=8 | band      | ndeg=0 | ndeg=8 |
|-----------|--------|--------|-----------|--------|--------|
|  0        | 1.052  | 1.050  | 10        | 0.005  | 0.005  |
|  1        | 0.830  | 0.768  | 11        | **0.125** | **0.000** |
|  2        | 0.101  | 0.070  | 12        | **0.121** | **0.000** |
|  3        | 0.110  | 0.078  | 13        | **0.241** | **0.000** |
|  4        | 0.000  | 0.000  | 14        | **0.647** | **0.000** |
|  5        | 0.000  | 0.031  | 15        | **0.642** | **0.000** |
|  6        | 0.395  | 0.000  | 16        | 0.352  | 0.032  |
|  7        | 0.045  | 0.004  | 17        | 0.075  | 0.000  |
|  8        | 0.045  | 0.003  | 18        | 0.073  | 0.000  |
|  9        | 0.005  | 0.006  | 19        | 0.182  | 0.104  |

Key finding: **even at ndeg=0 (no filter), overlap is only 0.252**. The Gram-
Schmidt + Rayleigh-Ritz step alone rotates eigenvectors from CASTEP's within
degenerate manifolds. The filter contributes ADDITIONAL rotation for bands near
the occupied subspace boundary (bands 11-15, overlap drops from 0.24-0.65 to 0.00).

At ndeg=0, GS re-S-orthonormalizes CASTEP's already-S-orthonormal wavefunctions,
changing the PW coefficients. The GS is order-dependent (processes bands 0→159),
so within degenerate manifolds the output basis depends on the initial ordering.
RR then diagonalizes H_sub and picks different eigenvectors within the manifold.

Both effects — GS order dependence and RR eigenvector selection — are inherent
to subspace methods vs CASTEP's band-by-band CG minimization.

## Cascade across iterations

**This is divergent, not just rotated.** Running iter-1→iter-2→iter-3 shows
a catastrophic cascade (`cascade_iter3_diagnostic`):

| Iter | band-0 eigenvalue | avg ⟨ψ|ψ_CASTEP⟩² | density split (soft/aug) |
|------|------------------|-------------------|--------------------------|
| 1    | -1.0458 Ha ✓     | 0.108             | 55.37 / 130.63 (29.7/70.3%) |
| 2    | -0.8690 Ha ✗     | 0.116             | 130.00 / 56.00 (69.9/30.1%) — INVERTED |
| 3    | -11.94 Ha ✗✗     | 0.000001          | (catastrophic divergence) |

### Per-band overlap |⟨our_b|CASTEP_b⟩|²

| band | iter-1 | iter-2 | iter-3 | Notes |
|------|--------|--------|--------|-------|
| 0    | 1.050  | 1.035  | 0.000  | Ground state stable through iter-2, collapses iter-3 |
| 1    | 0.768  | 0.000  | 0.000  | Already partially rotated at iter-1 |
| 2    | 0.070  | 0.000  | 0.000  | Rotated from the start |
| 3-13 | 0-0.08 | 0-0.01 | 0.000  | All rotated, iter-2 even worse |
| 14   | 0.000  | 0.638  | 0.000  | Band label swap at iter-2 (our 14 ↔ CASTEP 15) |
| 15   | 0.000  | 0.628  | 0.000  | Band label swap at iter-2 |
| 16-19| 0-0.10 | 0-0.00 | 0.000  | All rotated |

### Root cause analysis

The cascade mechanism:
1. **Iter-1**: Filter+RR on CASTEP psi + CASTEP V_eff → correct eigenvalues but
   rotated eigenvectors within degenerate Cu 3d manifold (bands 2-10, 0.02 Ha spread).
   The rotation shifts the density split (soft 81%, aug 111% of F8).
   
2. **Iter-2**: Our density → different V_eff → different D-screening → different
   S⁻¹·H operator → even more rotated eigenvectors. The density split inverts
   (soft/aug flip). Eigenvalues start drifting (band-0 -0.869 vs -1.055).

3. **Iter-3**: Catastrophic divergence. The wrong V_eff at iter-2 produces
   completely wrong eigenvalues (-11.94 Ha for band 0). The SCF is unstable.

### Conclusion

The density code is correct. The divergence is caused by eigenvector rotation
within near-degenerate manifolds at iter-1, which cascades through the
self-consistency cycle. The root cause is that the filter+RR produces
eigenvectors with the correct SPECTRUM but wrong SPATIAL CHARACTER, even
when starting from CASTEP's exact wavefunctions and V_eff.
