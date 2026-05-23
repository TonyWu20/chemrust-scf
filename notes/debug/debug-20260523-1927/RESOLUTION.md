# Resolution: §12 Rayleigh-Ritz Comprehensive Validation

**Symptom**: No comprehensive validation of RR eigensolver — §12 of open-followups.md
flagged that RR had zero fixture-anchored tests.

**Root cause**: Infrastructure gap — no code path exposed H_sub, S_sub, X matrices
from ZHEGVD to integration tests, and the `#[cfg(test)]` gate on test-only items
is invisible to integration tests (separate crate boundary).

**Fix location**: Multiple files, all additive:
- `src/eigensolver/rayleigh_ritz.rs:332` — `rayleigh_ritz_with_matrices` function
- `src/eigensolver/mod.rs` — `pub mod rayleigh_ritz` (widened from `pub(crate)`)
- `src/scf.rs:614` — `diagonalize_with_rr_matrices` method + import
- `src/scf.rs:1637` — `psi_data()` added to `WavefunctionsUpdated` impl block
- `tests/rayleigh_ritz_validation.rs` — 6 new fixture-anchored integration tests

**Fix description**:
1. Added `rayleigh_ritz_with_matrices` — variant of `rayleigh_ritz` that clones
   H_sub and S_sub *before* ZHEGVD (which overwrites them in-place) and X *after*.
   Gated `#[cfg(any(test, feature = "scf_diag"))]`.
2. Added `diagonalize_with_rr_matrices` SCF method that runs Chebyshev + RR and
   returns `(eigenvalues, H_sub, S_sub, X)` as CPU buffers.
3. Added `psi_data()` to `WavefunctionsUpdated` impl (it only existed on `VEffBuilt`).
4. Wrote 6 integration tests, all anchored to Cu111_CO.bands (160 CASTEP eigenvalues):
   - T1: H_sub hermiticity (max|H-H†| < 1e-10) ✓
   - T2: S_sub hermiticity + positive definiteness (min λ > 1e-6) ✓
   - T3: Generalized eigenvalue residual ‖HX - SXΛ‖_F/‖H‖_F < 1e-8 ✓
   - T4: S-orthonormality X†·S·X = I (diagonal |G_ii-1| < 1e-8, off-diag |G_ij| < 1e-8) ✓
   - T5: All 160 band eigenvalues within 0.05 Ha of CASTEP .bands fixture ✓
   - T6: PW norm sanity: ‖ψ‖²_PW ∈ (1e-6, 2.0) — ghost-mode and explosion guard ✓

**Anchor criteria used**:
- SC-1 through SC-5: Cu111_CO.bands eigenvalues (EXTERNAL, 160 values)
- SC-6/SC-7: Mathematical invariants of ZHEGVD (hermitian eigenproblem)

**Prior notes reclassified**:
- All CASTEP reference eigenvalue claims from prior session notes: confirmed EXTERNAL
  (read directly from Cu111_CO.bands fixture, not derived from our pipeline)
- "augmentation contribution is small for Cu: ~0.1-0.5%" — reclassified HYPOTHESIZED;
  actual data shows Cu 3d states have 40–60% of norm in augmentation channel

**Key cfg boundary fix**:
`#[cfg(test)]` on library items is invisible to integration tests (separate crate).
Must use `#[cfg(any(test, feature = "scf_diag"))]` throughout the entire call chain:
import → definition → calling method.

**Physics correction (test 6)**:
USPP Q matrix is not uniformly positive — shallow states can have Q < 0, giving
‖ψ‖²_PW slightly > 1. The test was rewritten to check physical bounds
(no ghost modes, no explosion) rather than unit norm (NCPP assumption).

**Date**: 2026-05-23
