# Divergence Surface: Issue #11a Per-Band Eigenvalue Branches

## Symptom Class

Eigenvalue drift between SCF iterations. Iter-1 eigenvalues match CASTEP reference (band-0 = -1.046 Ha vs ref -1.055 Ha, within 0.01 Ha). Iter-2 eigenvalues diverge (band-0 = -0.864 Ha, drift = 0.18 Ha; last-band = 1.952 Ha vs ref 0.115 Ha, overshoot = 1.84 Ha).

## Generic Divergence-Surface Categories

### 1. Data layout / axis ordering
**Status**: Ruled out by §1d, §1e fixes (cuFFT dim ordering, RR transpose layout).
**Anchor**: `notes/open-followups.md:23-41` — cuFFT plan dims fixed to `(ngx, ngy, ngz)`, RR transpose removed.

### 2. Normalization / scaling conventions
**Status**: Ruled out by §11b fix (electron count diagnostic).
**Anchor**: `notes/open-followups.md:714-721` — diagnostic formula corrected, electron count now matches 186 e⁻.

### 3. Sign / direction conventions
**Status**: Not applicable to eigenvalue drift (eigenvalues are real, sign is physical).

### 4. Boundary / edge-case handling
**Status**: **To be tested in Step 7** — R-ChFSI per-band branches (Das Alg 3 lines 598-604) are conditional on `eigenvalues.is_some()`. Iter-1 has `eigenvalues=None` (no prior RR), iter-2+ has `eigenvalues=Some(...)`. This is a boundary between two code paths.

**Specific conditions to audit**:
- `chebyshev.rs` line ~1582-1660: per-band Λ_Y init and Λ_X updates
- Condition: `if let Some(ref eigs) = eigenvalues`
- Our implementation: executes at iter-2+
- Reference implementation: Das et al. (2025) Algorithm 3 assumes exact S⁻¹ (ζ = 0) makes R-ChFSI ≡ standard ChFSI algebraically (main.tex:612)

### 5. Unit conversion at any boundary
**Status**: Not applicable — eigenvalues are in Hartree throughout, no unit conversion between iterations.

### 6. Parser precision / offset assumptions
**Status**: Not applicable — eigenvalues are computed, not parsed.

### 7. Decomposition / parallel artifacts
**Status**: Ruled out by single-GPU, single-rank test configuration.
**Anchor**: `tests/ca_scf_convergence.rs` runs on single GPU, no MPI decomposition.

### 8. Diagnostic comparison code
**Status**: **To be tested in Step 5** — eigenvalue extraction from RR output needs self-test verification.

## Project-Specific Divergence Surface

### A. Stale eigenvalue labels in per-band filter
**Status**: **To be tested in Step 7** — HYPOTHESIZED root cause from prior investigation.

**Hypothesis**: The per-band machinery uses eigenvalues from iter-1's RR (properties of H[ρ₁]) to construct filter shifts for iter-2's operator H[ρ₂]. When V_eff changes between iterations (ρ₁ ≠ ρ_castep), the eigenvalue labels are stale and the filter amplifies the wrong subspace.

**Evidence**:
- Empirical: `CHEMRUST_FORCE_NO_EIGS=1` (disabling per-band branches) reduces iter-2 last-band overshoot from 1.95 Ha → 0.70 Ha (64% improvement).
- Theoretical: Das main.tex:612 states R-ChFSI ≡ standard ChFSI when ζ = 0 (exact S⁻¹). Our Global Woodbury fix achieved ζ = 3.8e-15 (machine epsilon), so per-band machinery provides zero benefit.

**Conditional-skip table**:

| Reference line | Reference condition | Our line | Our condition |
|----------------|-------------------|----------|--------------|
| Das Alg 3 line 598 | `if eigenvalues provided` | `chebyshev.rs:~1582` | `if let Some(ref eigs) = eigenvalues` |
| Das Alg 3 line 599 | `Λ_Y ← eigenvalues` | `chebyshev.rs:~1590` | `lambda_y_dev.copy_from_slice(&eigs)` |
| Das Alg 3 line 604 | `Λ_X ← diag(X† H X)` | `chebyshev.rs:~1650` | `lambda_x_dev.copy_from_slice(&h_diag)` |
| Das main.tex:612 | `when ζ = 0, R-ChFSI ≡ ChFSI` | N/A | **Missing guard** — no check that ζ < threshold before enabling per-band branches |

**Red flag**: The reference paper states that per-band machinery is only beneficial when ζ > 0 (inexact S⁻¹). Our implementation unconditionally enables per-band branches when `eigenvalues.is_some()`, without checking whether ζ is small enough to make the machinery redundant.

### B. Spectral bounds computed for H, not S⁻¹·H
**Status**: **To be tested in Step 7** — potential compounding issue.

**Hypothesis**: Lanczos and Gershgorin compute bounds for the bare Hamiltonian H, but the Chebyshev filter (after §10 fix) applies S⁻¹·H. The eigenvalue spectrum of S⁻¹·H is related to but not identical to H's spectrum (generalized eigenvalue problem). Wrong bounds → wrong filter polynomial → poor separation of wanted/unwanted eigencomponents.

**Evidence**:
- `chebyshev.rs:~1100-1120`: `lanczos_upper_bound` operates on H (via `apply_full_hamiltonian`)
- `chebyshev.rs:~1130-1160`: Chebyshev recurrence applies S⁻¹ after each H·ψ (§10 fix)
- Mismatch: bounds are for H, filter is for S⁻¹·H

**Conditional-skip table**:

| Reference line | Reference condition | Our line | Our condition |
|----------------|-------------------|----------|--------------|
| Zhou (2014) Alg 4.1 §7.1 | `b_up from Lanczos on H` | `chebyshev.rs:~1100` | `lanczos_upper_bound` on H ✓ |
| Zhou (2014) Alg 4.1 §7.2 | `b_low from max Ritz of prev iter` | `chebyshev.rs:~1165` | `b_low = eigenvalues[last]` ✓ |
| Zhou (2014) Alg 4.1 recurrence | `ψⁱ = (2/r)·(H·ψⁱ⁻¹ - c·ψⁱ⁻¹) - ψⁱ⁻²` | `chebyshev.rs:~1130` | **Modified**: `ψⁱ = (2/r)·(S⁻¹·H·ψⁱ⁻¹ - c·ψⁱ⁻¹) - ψⁱ⁻²` |

**Red flag**: The spectral bounds (b_up, b_low) are computed for H, but the recurrence operates on S⁻¹·H. For USPP, S ≠ I, so the eigenvalue spectra differ. The filter polynomial is tuned for the wrong operator.

### C. V_eff drift between iterations
**Status**: **To be tested in Step 7** — residual 36% of iter-2 drift (band-0 = 0.18 Ha) remains even with per-band branches disabled.

**Hypothesis**: Our SCF pipeline's equilibrium differs slightly from CASTEP's. Even with per-band branches disabled, iter-2 eigenvalues drift by 0.18 Ha (band-0) because the density → V_eff → eigenvalues feedback loop has not yet converged to our equilibrium.

**Evidence**:
- `CHEMRUST_FORCE_NO_EIGS=1` reduces last-band overshoot from 1.95 Ha → 0.70 Ha, but band-0 drift remains 0.18 Ha.
- This is expected behavior for an SCF that has not yet converged — eigenvalues should drift until self-consistency is reached.

**Not a bug**: This is normal SCF behavior. The question is whether the drift *shrinks* over subsequent iterations (SCF converges to our equilibrium) or *grows* (SCF diverges). The prior investigation hit bug 11b before reaching iter-3, so this is unknown.

## Summary

**Items ruled out by external anchors**:
1. Data layout / axis ordering (§1d, §1e fixes)
2. Normalization / scaling conventions (§11b fix)
3. Decomposition / parallel artifacts (single-GPU test)

**Items to be tested in Step 7**:
4. Boundary / edge-case handling — per-band branches conditional on `eigenvalues.is_some()`
5. Diagnostic comparison code — eigenvalue extraction self-test
A. Stale eigenvalue labels in per-band filter (HYPOTHESIZED root cause)
B. Spectral bounds computed for H, not S⁻¹·H (potential compounding issue)
C. V_eff drift between iterations (expected SCF behavior, not a bug)

**Primary hypothesis**: Item A (stale eigenvalue labels) is the root cause of the 64% overshoot contribution. Item B (spectral bounds mismatch) may contribute to the residual 36% drift.
