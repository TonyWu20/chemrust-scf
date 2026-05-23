# Debug Session: Issue #12 — Rayleigh-Ritz Comprehensive Validation

**Date**: 2026-05-23 19:27  
**Issue**: open-followups.md §12 — Rayleigh-Ritz has zero comprehensive validation  
**Symptom**: 135% electron count drift, density split ratio wrong (ρ_PW 67.4% of CASTEP, ρ_aug 119%)

## Session Artifacts

- **PLAN.md** — Comprehensive validation test suite design (6 tests covering all mathematical properties)
- **INVESTIGATION.md** — (to be created in implementation session)
- **CRITERIA.md** — (to be created in implementation session)

## Plan Summary

**Scope**: Add 6 validation tests for Rayleigh-Ritz eigensolver mathematical properties:
1. H_sub Hermiticity (threshold: 1e-10 Ha)
2. S_sub Hermiticity & positive-definiteness (threshold: 1e-6)
3. Generalized eigenvalue residual ‖H·X - S·X·Λ‖_F (threshold: 1e-8)
4. Orthonormality X†·S·X = I (threshold: 1e-8)
5. All-band eigenvalue validation (threshold: 0.05 Ha, 160 bands)
6. Wavefunction normalization ⟨ψ|S|ψ⟩ = 1 (threshold: 1e-6)

**Infrastructure required**:
- Expose H_sub, S_sub, X from RR via `rayleigh_ritz_with_matrices()` (test-only)
- Add S-norm computation helper
- Add CPU linear algebra utilities (Frobenius norm, matrix multiply, eigenvalue solver)

**Estimated effort**: 3-4 hours

**Status**: Plan complete, ready for implementation

## Next Session

1. Read `PLAN.md` for full design
2. Implement infrastructure changes in `src/eigensolver/rayleigh_ritz.rs`
3. Implement test suite in `tests/rayleigh_ritz_validation.rs`
4. Run tests and document which properties pass/fail
5. Create `INVESTIGATION.md` with test results
6. If bugs found, file separate issue for fixes (this session is tests-only)

## Coordination

This plan is **independent** from `our-scf-persists-to-tingly-wall.md` (density split audit). Both can proceed in parallel.
