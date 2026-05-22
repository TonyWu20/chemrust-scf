# Decisions: R-ChFSI Implementation

## Problem

Standard ChFSI with our Woodbury S⁻¹ (1.4% error) stagnates — the error is applied
to O(1) eigenvectors at every step, producing a persistent O(ε) floor. SCF drifts
after iter-1 (band-1: −0.956 → −14.4 → −8.83 Ha vs reference −1.055 Ha).

## Algorithm Choice

**R-ChFSI (Das et al. 2025, Algorithm 3)** — residual-based reformulation.

The key insight: instead of filtering eigenvectors X, filter the residual
R = H·X − S·X·Λ. As SCF converges, ‖R‖ → 0, so the S⁻¹ approximation error
(proportional to ‖R‖) also vanishes. Standard ChFSI has no such guarantee.

Source: `reference_paper/extracted/das-2025-rchfsi/main.tex` lines 586–610.

## Scope

Replace the Chebyshev recurrence body in `chebyshev_filter` (lines 1200–1301 of
`src/eigensolver/chebyshev.rs`). Everything else stays unchanged:
- Spectral bounds (Lanczos + Gershgorin) — unchanged
- Gram-Schmidt orthonormalization — unchanged
- Rayleigh-Ritz — unchanged
- Density construction, V_eff assembly — unchanged (proven correct)

## Signature Change

`chebyshev_filter` needs the previous iteration's eigenvalues as a diagonal matrix
for the residual computation. They are already passed as `eigenvalues: Option<&[f64]>`.
No signature change needed — the existing parameter is sufficient.

**Fallback**: When `eigenvalues` is `None` (first SCF iteration), R-ChFSI degenerates
to standard ChFSI (residual R = H·X − 0 = H·X, Λ = 0). This is correct behavior.

## New Helper: `apply_s_times`

S·ψ = ψ + Σ_I β_I · Q_I · β_I^H · ψ

Uses `entry.q_matrix` (already on GPU in `VnlIonData`). Mirrors `apply_s_inverse`
but with `+` instead of `−` and `q_matrix` instead of `s_inv_mat`.

## Eigenvalue Matrix Representation

Λ is diagonal (n_bands × n_bands). We store it as a flat Vec<f64> on CPU and
apply it as a scaling operation (element-wise multiply per band column) rather
than a full matrix multiply. This avoids allocating an n_bands² GPU buffer.

The Λ_X, Λ_Y recurrence (eq. `main.tex:604`) operates on diagonal matrices:
  Λ_X ← (2σ₂/e)·Λ_Y·Λ − (2σ₂/e)·c·Λ_Y − σ·σ₂·Λ_X
Since all three are diagonal, this is element-wise on the diagonal vector.

## Fixture

`/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/` — same as existing tests.

## Acceptance Gate

Primary: `iter2_v_eff_range_within_one_ha_of_iter1` (already green, must stay green).
Secondary: `fixed_point_matches_castep_energy` — should converge instead of drift.
