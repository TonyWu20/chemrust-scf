# Resolution: Issue #11a Per-Band Eigenvalue Branches

**Symptom**: R-ChFSI per-band eigenvalue branches cause 64% of iter-2 last-band overshoot (1.95 Ha → 0.70 Ha when disabled via `CHEMRUST_FORCE_NO_EIGS=1`).

**Root cause**: The per-band machinery (Das Algorithm 3 lines 598-604) is designed for inexact S⁻¹ (ζ > 0). After §10's Global Woodbury fix, we have exact S⁻¹ (ζ = 3.8e-15), making the per-band machinery algebraically redundant per Das et al. (2025) main.tex:612. However, it introduces numerical weak points when eigenvalue labels from iter-1 (properties of H[ρ₁]) are used to construct filter shifts for iter-2's different operator H[ρ₂].

**Fix location**: 
- `src/scf.rs:528-541` — always pass `eigenvalues=None` to Chebyshev filter
- `src/eigensolver/chebyshev.rs:1585-1592` — fix `lam_source` to use `eigenvalues.unwrap_or(&h_eig)` consistently

**Fix description**:
1. Removed eigenvalue passthrough logic and `CHEMRUST_FORCE_NO_EIGS` diagnostic flag from `src/scf.rs`
2. Changed `let eig: Option<&[f64]> = None;` to always pass None, effectively running standard ChFSI
3. Fixed `lam_source` bug in Mode B (was hardcoded to `&h_eig`, now uses `eigenvalues.unwrap_or(&h_eig)`)
4. Removed `CHEMRUST_LAMSOURCE_EIG` diagnostic flag from `chebyshev.rs`

**Anchor criteria used**:
- CASTEP reference band-0 eigenvalue: -1.05502343 Ha (Source: `Cu111_CO.bands` line 12)
- CASTEP reference last-band eigenvalue: 0.11531044 Ha (Source: `Cu111_CO.bands` last line)

**Prior notes reclassified**:
- "Iter-2 band-0 = -0.864 Ha" — reclassified from implicit specification to DERIVED (from our buggy pipeline)
- "Iter-2 last-band = 1.952 Ha" — reclassified from implicit specification to DERIVED (from our buggy pipeline)
- "V_eff range = 8.69 Ha" — reclassified from EXTERNAL to DERIVED (needs verification by computing from `.pot_fmt` fixture)

**Empirical validation**:
- **Before fix**: 
  - SC-1 (iter-1 band-0): ✅ PASSED (delta = 0.0092 Ha, gate 0.05 Ha)
  - SC-4 (iter-2 last-band): ❌ FAILED (overshoot = 1.836 Ha, gate 1.0 Ha)
- **After fix**: 
  - SC-1 (iter-1 band-0): ✅ PASSED (delta = 0.0092 Ha, gate 0.05 Ha) — unchanged
  - SC-4 (iter-2 last-band): ✅ PASSED (overshoot = 0.586 Ha, gate 1.0 Ha) — **64% improvement**

**Improvement**: Iter-2 last-band overshoot reduced from 1.95 Ha → 0.70 Ha (64% reduction), exactly matching the empirical prediction from `CHEMRUST_FORCE_NO_EIGS=1` testing.

**Date**: 2026-05-23

**Investigation artifacts**: `notes/debug/debug-20260523-1758/`
- `INVESTIGATION.md` — prior investigation classification
- `CRITERIA.md` — external anchor criteria
- `DIVERGENCE_SURFACE.md` — divergence surface enumeration
- `DIAGNOSTIC_SELFTEST.md` — diagnostic verification strategy
- `UPSTREAM_AUDIT.md` — upstream input audit, Das main.tex:612 analysis
