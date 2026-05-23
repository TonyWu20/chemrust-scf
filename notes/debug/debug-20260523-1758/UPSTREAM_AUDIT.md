# Upstream Audit: Issue #11a Per-Band Eigenvalue Branches

## Algorithm Validation Status

**User response**: "Partially — some parts validated"

**Audit mode**: Hybrid — audit inputs (eigenvalue labels, spectral bounds) AND verify per-band machinery against reference paper.

## Key Finding from Das et al. (2025) main.tex:612

> "When D⁻¹ = B⁻¹ (i.e. the approximate inverse is exact) **and** the same matrix is used for both the Chebyshev filter and the Rayleigh-Ritz projection, ChFSI and R-ChFSI are **algebraically equivalent**."

**Translation to our context**:
- D⁻¹ = our `apply_s_inverse` (Global Woodbury S⁻¹)
- B⁻¹ = exact S⁻¹ for the USPP generalized eigenproblem
- After §10's Global Woodbury fix: ζ = ‖D⁻¹ − B⁻¹‖ = 3.8e-15 (machine epsilon)
- **Conclusion**: D⁻¹ = B⁻¹ exactly (within numerical precision)
- **Implication**: R-ChFSI ≡ standard ChFSI algebraically

**Therefore**: The per-band machinery (Das Algorithm 3 lines 598-604) provides **zero benefit** when ζ ≈ 0. It is designed for the regime where ζ > 0 (inexact S⁻¹), which is NOT our case.

## Root Cause: Stale Eigenvalue Labels

**Hypothesis from prior investigation** (DIVERGENCE_SURFACE.md item A):

The per-band machinery uses eigenvalues from iter-1's RR (properties of H[ρ₁]) to construct filter shifts for iter-2's operator H[ρ₂]. When V_eff changes between iterations (ρ₁ ≠ ρ_castep), the eigenvalue labels are stale and the filter amplifies the wrong subspace.

**Evidence**:
1. **Empirical**: `CHEMRUST_FORCE_NO_EIGS=1` (disabling per-band branches) reduces iter-2 last-band overshoot from 1.95 Ha → 0.70 Ha (64% improvement).
2. **Theoretical**: Das main.tex:612 states R-ChFSI ≡ standard ChFSI when ζ = 0. Our ζ = 3.8e-15, so per-band machinery is redundant.
3. **Code audit**: `chebyshev.rs:1487-1500` — per-band branches execute when `eigenvalues.is_some()`, without checking whether ζ is small enough to make them redundant.

## Conditional-Skip Audit

| Reference condition | Our implementation | Status |
|-------------------|-------------------|--------|
| `if ζ > threshold` (enable per-band machinery only when S⁻¹ is inexact) | **Missing** — no check on ζ before enabling per-band branches | ❌ RED FLAG |
| `if eigenvalues.is_some()` (enable per-band machinery when prior eigenvalues available) | `chebyshev.rs:1487` ✓ | ✓ Present |
| `Λ_Y ← eigenvalues` (Das Alg 3 line 599) | `chebyshev.rs:1593-1598` ✓ | ✓ Present |
| `Λ_X ← diag(X† H X)` (Das Alg 3 line 604) | `chebyshev.rs:1663-1672` ✓ | ✓ Present |

**Missing guard**: The reference paper's algebraic equivalence statement (main.tex:612) implies that per-band machinery should be **disabled** when ζ ≈ 0. Our implementation unconditionally enables per-band branches when `eigenvalues.is_some()`, without checking ζ.

## Input Audit: Eigenvalue Labels

**Input**: `eigenvalues` parameter to `chebyshev_filter` (type: `Option<&[f64]>`)

**Source**: `scf.rs` — eigenvalues from previous iteration's Rayleigh-Ritz

**Unit convention**: Hartree (no conversion needed)

**Staleness check**:
- Iter-1: `eigenvalues = None` (no prior RR) → per-band branches disabled ✓
- Iter-2: `eigenvalues = Some(iter1_eigenvalues)` → per-band branches enabled
  - `iter1_eigenvalues` are properties of H[ρ₁] (iter-1 density)
  - Iter-2 filter operates on H[ρ₂] (iter-2 density, built from iter-1 wavefunctions)
  - **Mismatch**: eigenvalue labels are for H[ρ₁], but filter operates on H[ρ₂]

**Why this causes drift**:
- The per-band shifts `Λ_Y` are initialized to `(σ₁/e) · (eigenvalues - c)` (line 1593-1598)
- These shifts are used to construct the filter polynomial that amplifies eigencomponents near `eigenvalues`
- If `eigenvalues` are stale (from H[ρ₁] but filter operates on H[ρ₂]), the filter amplifies the wrong subspace
- Result: RR produces eigenvalues that drift away from the correct values

## Input Audit: Spectral Bounds

**Input**: `bounds` parameter to `chebyshev_filter` (type: `SpectralBounds`)

**Source**: `chebyshev.rs:1100-1120` — Lanczos upper bound on bare H

**Operator mismatch**:
- Lanczos computes bounds for H (bare Hamiltonian)
- Chebyshev filter (after §10 fix) applies S⁻¹·H (preconditioned operator)
- For USPP, S ≠ I, so eigenvalue spectra differ

**Why this might contribute to drift**:
- The filter polynomial is tuned for the spectrum [b_low, b_up]
- If bounds are for H but filter operates on S⁻¹·H, the polynomial is tuned for the wrong spectrum
- Result: poor separation of wanted/unwanted eigencomponents

**Status**: **Potential compounding issue** — may contribute to the residual 36% drift (band-0 = 0.18 Ha) that remains even with per-band branches disabled.

## Fix Scope

**Primary fix** (addresses 64% of iter-2 overshoot):
- Disable per-band branches when ζ < threshold (e.g., 1e-6)
- Implementation: add a guard condition at `chebyshev.rs:1487` that checks ζ before enabling per-band branches
- Alternatively (simpler): always pass `eigenvalues=None` to the filter, effectively running standard ChFSI

**Secondary fix** (addresses residual 36% drift, if needed):
- Compute spectral bounds for S⁻¹·H instead of H
- Implementation: apply S⁻¹ inside the Lanczos estimator (after each H·v)
- This is a larger change and should only be attempted if the primary fix is insufficient

## Recommendation

**Implement the primary fix first**: Always pass `eigenvalues=None` to the filter. This is the simplest fix and addresses 64% of the observed overshoot. The residual 36% drift (band-0 = 0.18 Ha) may be normal SCF behavior (eigenvalues drifting toward equilibrium) rather than a bug.

**Validation**: After the primary fix, run the tight tests (SC-1 through SC-4) and check whether:
1. Iter-2 last-band overshoot drops from 1.95 Ha to < 1.0 Ha (SC-4)
2. Iter-2 band-0 drift remains < 0.20 Ha (SC-3)
3. SCF converges over subsequent iterations (eigenvalues approach CASTEP reference)

If the residual 36% drift grows over subsequent iterations (SCF diverges), then the secondary fix (spectral bounds for S⁻¹·H) is needed. If it shrinks (SCF converges), then the residual drift is normal SCF behavior and no further fix is needed.
