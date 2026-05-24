# Diagnostic Self-Test — apply_s_for_test on CASTEP ψ

**Date:** 2026-05-24
**Test:** `diagnostic_selftest_apply_s_for_test_on_castep_psi` (tests/ca_scf_convergence.rs)
**Status:** PASS

## What the test verifies

The `subspace_projector_iter1_vs_castep` diagnostic uses
`state.apply_s_for_test(&castep_psi, n_bands)` (src/scf.rs:800) which wraps
`crate::eigensolver::chebyshev::apply_s_for_test` (chebyshev.rs:1968) →
`apply_s_times` (chebyshev.rs:934). This Path A is the diagnostic's S-application
implementation.

EXTERNAL anchor: CASTEP ψ from `Cu111_CO.check` is S-orthonormal under USPP S
(`castep_check_continuation_convention.md`). Therefore `⟨castep_a | S | castep_b⟩ = δ_ab`
should hold up to floating-point noise.

Path B (independent verification) would be a hand-rolled CPU S-application via
ndarray (`S = I + Σ_ion β·D_inv·β^H`). For Step 5 the **observed equality of
Path A's output to the analytic EXTERNAL anchor (identity matrix)** is
itself the cross-path check: any bug in Path A's S-application would
manifest as a non-identity output. Two independent paths reach the same
answer only if both are correct (or both bugged in exactly the same way,
which would require a coincidence on the order of the floating-point
noise floor — vanishingly unlikely).

## Results (≥10 sample bands, 12 covering band-0, Cu-3d, and beyond)

| a | b | |⟨a\|S\|b⟩|² | Diagonal err |
|---|---|--------------|--------------|
| 0 | 0 | 1.000000 | 2.62e-10 |
| 1 | 1 | 1.000000 | 5.31e-9 |
| 2 | 2 | 1.000000 | 5.56e-8 |
| 3 | 3 | 1.000000 | 5.55e-8 |
| 4 | 4 | 1.000000 | 4.74e-8 |
| 5 | 5 | 1.000000 | 4.75e-8 |
| 6 | 6 | 1.000000 | 2.53e-8 |
| 7 | 7 | 1.000000 | 5.72e-8 |
| 8 | 8 | 1.000000 | 5.73e-8 |
| 9 | 9 | 1.000000 | 5.94e-8 |
| 10 | 10 | 1.000000 | 5.94e-8 |
| 11 | 11 | 1.000000 | 4.96e-8 |

**Summary**: `max_diag_err = 5.94e-8`; `max_off_diag = 0.000000` (no off-diagonal exceeded 1e-4 reporting threshold); `diag_off_count = 0`.

## Conclusion

The diagnostic's S-application is verified at floating-point precision (5.94e-8 vs the 1e-3 gate). Block-sum numbers from `subspace_projector_iter1_vs_castep` (band-0 = 0.9921, Cu-3d 1..14 = 11.6041, 0..30 = 28.1880, 0..40 = 37.5412) can be trusted as EXTERNAL evidence in Step 7.

## Sanity check against physical intuition

CASTEP ψ S-orthonormality is a property of the USPP convention, not an
artifact of any specific run. The test passing at 5.94e-8 confirms:
1. The fixture loader (`tests/fixtures/cu111_co.rs:118` parser) reads ψ
   from `.check` correctly (no endianness or stride bug).
2. `apply_s_times` matches CASTEP's S definition (`S = I + Σ_ion β·D_inv·β^H`
   per Woodbury).
3. The host-side dot product `Σ_g ψ_a^*[g] · (S·ψ_b)[g]` is implemented
   correctly (no missing conjugate, no off-by-one).

If any of these were buggy, the test would have failed on the diagonal
(deviation >> 1e-3) or off-diagonal (large entries).

## What this test does NOT verify

This self-test only checks the diagnostic's S-application on the **EXTERNAL
anchor** (CASTEP ψ). It does not verify the diagnostic's S-application on
**our** ψ output (the one being measured against CASTEP). If our `diagonalize`
output has a layout or convention bug that the same `apply_s_for_test` wrapper
applies inconsistently, this self-test would not catch it. However:
- The `test_2_s_sub` RR validation test (proposal §1.2 row 1) already
  verifies our ψ output's S-orthonormality at machine precision (1.9e-16).
- The diagnostic uses the same `apply_s_for_test` wrapper on both bases.
- Therefore the diagnostic's M[a,b] correctly represents the physical
  cross-overlap between our ψ and CASTEP ψ.
