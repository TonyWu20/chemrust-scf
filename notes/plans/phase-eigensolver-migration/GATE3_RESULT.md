# Gate 3 Result — Phase 0 Davidson Decision

**Date:** 2026-05-24
**Branch:** feat/phase-global-woodbury
**Commit:** 5b24b937c329a3367c5a6eed5f7837f0627ce1a3
**Test:** gate3_davidson_minimal_locking_preserves_cu3d_block

## Measurements

| Metric | Value |
|---|---|
| Cu-3d block sum (bands 1..14, S-weighted) | 12.929543 |
| Davidson ratio (sum / 13.0) | 0.994580 |
| Chebyshev-RR baseline (recorded) | 11.610000 (ratio 0.893) |
| CASTEP self-overlap reference | 13.000000 |
| Bands locked / total | 0 / 160 |
| Max residual norm | 1.008e-1 Ha |
| Mean residual norm | (not measured — Phase 0 scratch) |
| Sibling sums | band0=0.992711, 0..30=29.563703, 0..40=39.725167 |

## Decision

**Phase 1 algorithm:** Davidson v1 (Phase 1A) — lean per threshold (ratio 0.995 > 0.97).

**Rationale:** The Davidson single-sweep lifted the Cu-3d block sum from 11.61
(Chebyshev-RR, ratio 0.893) to 12.93 (ratio 0.995), even with **zero bands locked**
and residuals of ~0.1 Ha. The ZHEGVD ran on the full 160-band subspace (k=160),
yet the Cu-3d block survived nearly intact. This proves that the Chebyshev-RR
degradation is NOT caused by the ZHEGVD alone — it's the combination of filter
polynomial distortion + rotation. Davidson's direct Hψ/Rayleigh-quotient path
preserves the subspace structure even without locking.

The 0/160 locked count is expected for Phase 0: lock_tol=1e-6 with our V_eff
(not CASTEP-pinned) produces residuals ~0.1 Ha, well above the tolerance.
Phase 1A adds a preconditioner and outer iteration, which will progressively
tighten residuals and activate locking.

## Next-phase anchor

Proceed with Phase 1A (Davidson v1) from
`notes/plans/phase-eigensolver-migration/PHASE_PLAN.md` section "Phase 1A —
Davidson v1".

Phase 1A adds:
- Preconditioner (Teter-Payne or similar)
- Outer Davidson iteration (converge residuals below lock_tol)
- Subspace management (restart, collapse)
- Locking with progressively tightening tolerance

The Phase 0 single-sweep demonstrated that the core Davidson approach
(direct Hψ → Rayleigh quotient → residual → ZHEGVD on unconverged sub-block)
preserves the Cu-3d block at 0.995 of the CASTEP reference, versus 0.893 for
Chebyshev-RR. The preconditioner + iteration in Phase 1A should close the
remaining 0.005 gap.

## Diagnostic notes

- **0 bands locked:** lock_tol=1e-6 with residuals of ~0.1 Ha. This is expected
  — our V_eff is iter-1 quality (not converged), so the operator H differs
  from CASTEP's converged H. CASTEP ψ are not exact eigenvectors of our H.
  Phase 1A's outer iteration will converge V_eff and drive residuals below
  lock_tol.

- **sibling block sums show sub-unity across the board:** band0=0.993, 0..30=29.56
  (ratio 0.985), 0..40=39.73 (ratio 0.993). The Cu-3d block (0.995) is actually
  slightly better preserved than the low-lying bands. This is the opposite of
  Chebyshev-RR, where the Cu-3d block was the worst-hit.

- **band0 ≠ 1.000:** The first band (lowest eigenvalue) has only 0.993 overlap
  with CASTEP. This suggests a small global rotation even without Chebyshev
  filtering, likely from the ZHEGVD on the full 160×160 subspace. Phase 1A's
  locking should fix this as bands converge and get removed from the active
  subspace.

- **No NaN, no panic:** All 13 algorithm steps executed correctly. The ZHEGVD
  on k=160 (full subspace) ran successfully. The S-orthogonalization pass
  completed without issues. Early-return path (step 8) was not triggered
  (k=160, not 0).

- **V_eff drift is significant:** The iter-1 V_eff differs from CASTEP's
  converged V_eff enough to produce ~0.1 Ha residuals. This is the expected
  behavior — the discriminator is working. If n_unconverged had been <10
  (E1 exploration note), we'd have flagged the run as suspect.

- **Runtime:** ~173 seconds on GPU (including fixture loading, V_eff build,
  D_screening for 18 ions, and the Davidson single-sweep on 160 bands).
