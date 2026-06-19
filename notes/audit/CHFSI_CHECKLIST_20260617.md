# ChFSI Adversarial Audit Checklist

**Date**: 2026-06-17
**Scope**: Chebyshev filtering eigensolver, Rust (`chebyshev.rs`, `scf.rs`, `hamiltonian.rs`, `rayleigh_ritz.rs`, `vnl_data.rs`) vs ABINIT 10.6 (`m_chebfi2.F90`, `m_invovl.F90`, `m_xg_ortho_RR.F90`, `m_vtowfk.F90`)
**Methodology**: 16-agent adversarial workflow audit comparing ABINIT source against Lygatsika et al. 2025 GPU paper and Levitt-Torrent 2015 paper; 8 claim-by-claim verifications
**Status**: Audit complete. 7 divergences identified. 5 fixed, 1 partially fixed, 1 new divergence found.

---

## Fix History

| Date | Fix | IDs | Description |
|------|-----|-----|-------------|
| 2026-06-15 | C5 | Woodbury | Iterative refinement → ζ = 1.36×10⁻¹² (3-pass, pure GPU) |
| 2026-06-16 | C1+C2 | ChFSI algorithm | Switched from R-ChFSI to standard ChFSI (S⁻¹·H operator, eigenvector recurrence) |
| 2026-06-16 | C4 | Ampfactor | GPU-resident ampfactor: per-band cublasZscal, ndeg=0 skip |
| 2026-06-17 | C13 | Orthonormalization | Cholesky QR replacing Gram-Schmidt (ABINIT `xg_Block_xgBlock_xg_QP`) |
| 2026-06-19 | C21 | oracle=0 locking | Converged bands locked (ndeg=0) even with oracle=0 — **divergence from ABINIT** (see C21) |
| 2026-06-20 | — | S-norm normalization | Per-band S-norm normalization before Cholesky QR + ZPOTRF regularization chain + ZHEEVD fallback |

---

## 0. Algorithm Identity: What ABINIT Actually Does

Before auditing components, we must establish the algorithmic identity. The workflow audit verified 8 claims against ABINIT source:

| # | Claim | Verdict | ABINIT Source |
|---|-------|---------|---------------|
| 1 | S⁻¹·H operator (not H·S⁻¹) | **MATCH** | `m_chebfi2.F90:861` — `getBm1X(chebfi%xAXColsRows, ...)` receives H·Ψ, applies S⁻¹ |
| 2 | Standard ChFSI on eigenvectors (not R-ChFSI on residuals) | **MATCH** | `m_chebfi2.F90:644` — recurrence operates on `xXColsRows` (eigenvectors), residuals computed only at line 710 as diagnostic |
| 3 | Matrix-free RR rotates Ψ, HΨ, SΨ | **MATCH** | `m_xg_ortho_RR.F90:516-525` — X, AX, BX all rotated via gemm |
| 4 | λ_plus = ecut (physics cutoff) | **MATCH** | `m_chebfi2.F90:547` — `lambda_plus = chebfi%ecut` |
| 5 | Ampfactor normalizes Ψ, HΨ, SΨ per-band | **PARTIAL** | `m_chebfi2.F90:958-1006` — code does it; 2015 paper only mentions T_n for convergence estimation |
| 6 | H-only during recurrence (paper claim) | **MISMATCH** | `m_chebfi2.F90:660` — `getAX_BX` computes both H+S every step (S unused by recurrence) |
| 7 | Multiple chebfi_run calls per SCF iteration | **MISMATCH** | `m_vtowfk.F90:382` — `nnsclo_now=1` for SCF; multiple calls only for non-SCF |
| 8 | Woodbury S⁻¹: iterative refinement | **MATCH** | `m_invovl.F90:1102-1140` — iterative refinement with block-diagonal preconditioner, targets 1e-16 |

### Bottom line

ABINIT's Chebyshev filtering is:
- **Standard ChFSI** (eigenvector filtering via S⁻¹·H·Ψ, Chebyshev three-term recurrence)
- **Matrix-free RR** (rotates Ψ, HΨ, SΨ — the 2021+ xG abstraction layer innovation)
- **Woodbury S⁻¹ via iterative refinement** to 1e-16 relative error
- **Per-band ampfactor** normalization of all three vectors
- **Single filter call per SCF iteration** (not multiple)
- **λ_plus = ecut from input file** (not computed from grid)

Our code is (as of 2026-06-20):
- **Standard ChFSI** (eigenvector filtering via S⁻¹·H, matching ABINIT) — C1/C2 FIXED
- **Standard RR** (rotates only Ψ; SΨ/HΨ recomputed fresh) — C3 deferred
- **Woodbury S⁻¹ via iterative refinement** (ζ = 1.36×10⁻¹²) — C5 FIXED
- **GPU-resident ampfactor** (per-band cublasZscal) — C4 FIXED
- **Cholesky QR orthonormalization** (replacing Gram-Schmidt) — C13 FIXED
- **Two filter+RR passes** per SCF iteration (ABINIT nnsclo_now=2 cold-start)
- **λ_plus = ecut** (matching ABINIT)
- **NEW: per-band S-norm normalization** before Cholesky QR (not in ABINIT — defense against S-operator mismatch between CPU/GPU)
- **NEW: converged-band locking with oracle=0** (divergence from ABINIT — see C21)

---

## 1. Component Summary Table

| # | Component | ABINIT Source | Rust Source | Status | Severity |
|---|-----------|---------------|-------------|--------|----------|
| 1 | Algorithm identity: standard ChFSI vs R-ChFSI | `m_chebfi2.F90:641-666` | `chebyshev.rs:2474-2660` | **FIXED** | — |
| 2 | Operator ordering: S⁻¹·H vs H·S⁻¹ | `m_chebfi2.F90:858-868` | `chebyshev.rs:2552,2666` | **FIXED** | — |
| 3 | Matrix-free RR: rotate HΨ, SΨ | `m_xg_ortho_RR.F90:516-525` | `rayleigh_ritz.rs:608-629` | **DIVERGE** | **CRITICAL** |
| 4 | Ampfactor normalization in pipeline | `m_chebfi2.F90:676,958-1006` | `chebyshev.rs:2707-2708` | **FIXED** | — |
| 5 | Woodbury S⁻¹ precision | `m_invovl.F90:1072,1102-1140` | `hamiltonian.rs:651-723` | **FIXED** | — |
| 6 | Spectral bounds: ecut for λ_plus | `m_chebfi2.F90:547` | `scf.rs` (ecut from kinetic cutoff) | **MATCH** | — |
| 7 | Per-band oracle (locking) | `m_chebfi2.F90:1128-1214` | `chebyshev.rs:1660-1780` | **MATCH** | — |
| 8 | Rayleigh quotients pre-filter | `m_chebfi2.F90:761-810` | `chebyshev.rs:1864-2132` | **MATCH** | — |
| 9 | Filter degree oracle (cheb_oracle1) | `m_chebfi2.F90:1031-1064` | `chebyshev.rs:1569-1602` | **MATCH** | — |
| 10 | Inner loop structure | `m_vtowfk.F90:382` | `scf.rs:982-1010` | **DIVERGE** | **MEDIUM** |
| 11 | Residual computation: fresh vs prior-iteration | `m_chebfi2.F90:710-716` | `chebyshev.rs:2395-2410` (Phase 2b) | **DIVERGE** | **MEDIUM** |
| 12 | Lambda shift eigenvalues: prior RR vs fresh RQ | `m_chebfi2.F90:528` | `chebyshev.rs:2310` | **DIVERGE** | **MEDIUM** |
| 13 | Orthonormalization: Cholesky QR (was Gram-Schmidt) | ABINIT `xg_Block_xgBlock_xg_QP` | `chebyshev.rs:2709-2772` | **FIXED** (Cholesky QR) | — |
| 21 | oracle=0 converged-band locking | `m_chebfi2.F90:628` | `chebyshev.rs:2457-2472` | **NEW DIVERGE** | **HIGH** |
| 14 | Dead code: compute_spectral_bounds (Gershgorin) | — | `chebyshev.rs:133-180` | **DEAD** | **LOW** |
| 15 | Dead code: chebfi_residual_norms | `m_chebfi2.F90:709-717` | `chebyshev.rs:1413-1514` | **DEAD** | **LOW** |
| 16 | Dead code: lanczos_upper_bound | — | `chebyshev.rs:195-360` | **DEAD** | **LOW** |
| 17 | Dead code: transpose_col_to_row / row_to_col | — | `chebyshev.rs:475-517` | **DEAD** | **LOW** |
| 18 | Dead code: apply_scaled_hamiltonian_inplace | — | `chebyshev.rs:372-404` | **DEAD** | **LOW** |
| 19 | Production filter mode: SinvHFullDas | — | `scf.rs:712`, `chebyshev.rs:68-83` | **INFO** | — |
| 20 | Procrustes pinning: disabled in production | — | `scf.rs:1003` (pin_cfg=None) | **INFO** | — |

### Status Counts

| Status | Count |
|--------|-------|
| **FIXED** (previously DIVERGE, now matches ABINIT) | 5 |
| **DIVERGE** (still must fix for ABINIT parity) | 2 (C3, C21) |
| **MATCH** (verified against ABINIT) | 5 |
| **DEAD** (defined but unused) | 5 |
| **INFO** (documentation) | 2 |

---

## 2. Divergence Details — Critical Path

### C1: Algorithm Identity — R-ChFSI vs Standard ChFSI (FIXED 2026-06-16)

| Property | ABINIT (standard ChFSI) | Our Code (before fix) | Our Code (after fix) |
|----------|------------------------|----------------------|---------------------|
| **What is filtered** | Eigenvectors Ψ | Residuals R = H·Ψ − Ψ·Λ | Eigenvectors Ψ |
| **Recurrence** | Standard three-term on eigenvectors | R-ChFSI on residuals | Standard three-term on eigenvectors via S⁻¹·H |
| **ABINIT source** | `m_chebfi2.F90:641-666, 858-886` | — | — |
| **Our source** | — | Pre-fix `chebyshev.rs` Phase 5 | `chebyshev.rs:2503-2657` |

**Fix**: Replaced R-ChFSI recurrence with standard ChFSI three-term recurrence on eigenvectors via the S⁻¹·H operator (matching ABINIT's `chebfi_computeNextOrderChebfiPolynom`).

---

### C2: Operator Ordering — H·S⁻¹ vs S⁻¹·H (FIXED 2026-06-16)

| Property | ABINIT | Our Code (after fix) |
|----------|--------|---------------------|
| **Operator** | S⁻¹·H (apply H first, then S⁻¹) | S⁻¹·H (apply H first, then S⁻¹) |
| **ABINIT source** | `m_chebfi2.F90:858-868`: `getBm1X(chebfi%xAXColsRows, ...)` receives H·Ψ, applies S⁻¹ | — |
| **Our source** | — | `chebyshev.rs:2569-2599`: applies H to `x_curr`, then S⁻¹ to result |

**Fix**: Switched from H·S⁻¹ to S⁻¹·H — apply `apply_full_hamiltonian` first, then `apply_s_inverse` (Woodbury S⁻¹) to the result.

---

### C3: Matrix-Free RR — Rotating HΨ and SΨ (CRITICAL)

| Property | ABINIT | Our Code |
|----------|--------|----------|
| **What's rotated** | Ψ, H·Ψ, S·Ψ (all three) | Ψ only |
| **ABINIT source** | `m_xg_ortho_RR.F90:516-525`: gemm on X, AX, BX | — |
| **Our source** | — | `rayleigh_ritz.rs:608-629`: `psi_new = psi_row · X` only |

**Cascade contribution**: After RR rotates Ψ_new = Xᵀ·Ψ, our H·Ψ and S·Ψ in GPU memory are STALE — they correspond to pre-rotation Ψ, not post-rotation Ψ_new. The next `chebfi_run_rust` call recomputes H fresh (Phase 1), so H·Ψ is refreshed. But S·Ψ is NOT recomputed in Phase 1 — it's computed via `apply_s_times` on the fresh psi. So this isn't a direct bug for our code (S·Ψ is fresh for the filter recurrence).

However, after switching to standard ChFSI (C1), H·Ψ and S·Ψ must BOTH be kept consistent with Ψ across the RR boundary because ABINIT's pipeline:
1. Computes H·Ψ, S·Ψ at start (Phase 1)
2. Uses H·Ψ in recurrence (via S⁻¹)
3. After RR, expects H·Ψ, S·Ψ to be already rotated and correct for the next iteration

Without matrix-free RR, we'd need to recompute H·Ψ and S·Ψ after every RR, adding cost. With matrix-free RR, we pay O(M²·N) gemm instead of O(N log N + N·N_proj) for FFT-based H application.

**Fix direction**: Add gemm rotation of H·Ψ and S·Ψ in `rayleigh_ritz`. The rotation matrix X is already available from ZHEGVD. Add:
```
hpsi_new = Xᵀ · hpsi_row    (gemm)
spsi_new = Xᵀ · spsi_row    (gemm, if available)
```

---

### C4: Missing Ampfactor Normalization (FIXED 2026-06-16)

| Property | ABINIT | Our Code (after fix) |
|----------|--------|---------------------|
| **Ampfactor called** | Always (`m_chebfi2.F90:676`) | Always (Phase 5b, `chebyshev.rs:2681-2703`) |
| **What's normalized** | X, AX, BX all by `1/T_n(λ_i)` per band | X (psi) by `1/T_n(λ_i)` per band; AX/BX not normalized (recomputed fresh in Phase 6/8) |
| **Method** | Fortran `xgBlock_scale` | GPU-resident `cublasZscal` per band — zero upload/download |

**Fix**: GPU-resident ampfactor using per-band `cublasZscal_v2`. Bands with ndeg=0 skipped. Normalizes only X (psi) — AX (H·psi) and BX (S·psi) are recomputed fresh in subsequent phases, so their ampfactor is unnecessary. Matching ABINIT's `cheb_poly1` and clamping (|amp| < 1e-3 → 1e-3).

---

### C5: Woodbury S⁻¹ Precision — 3.4×10⁻⁶ vs 10⁻¹⁶ (FIXED 2026-06-15)

| Property | ABINIT | Our Code (after fix) |
|----------|--------|---------------------|
| **Method** | Iterative refinement | Iterative refinement (3 passes) |
| **Target precision** | 1e-16 (`m_invovl.F90:1072`) | Achieved ζ = 1.36×10⁻¹² |
| **ABINIT source** | `m_invovl.F90:1102-1140` | — |
| **Our source** | — | `hamiltonian.rs` — `apply_s_inverse` with 3-pass refinement |

**Fix**: Added 3-pass iterative refinement loop matching ABINIT's algorithm:
1. Initial guess: `zgetrs(lu_m, ipiv, h_in)` (direct LU solve)
2. Residual: `r = B^T·h_in - M·y` (exact M = Q⁻¹ + B^H·B)
3. Correction: `dy = zgetrs(lu_m, ipiv, r)`, `y += dy`
4. Pure GPU — Q⁻¹·y via per-ion ZGEMM, all on-device
5. Final: `hpsi = h_in - B·y`

Precision improved from ζ=3.4×10⁻⁶ to ζ=1.36×10⁻¹² (measured at Gate 0 test).

---

### C6: Inner Loop Structure (MEDIUM)

| Property | ABINIT | Our Code |
|----------|--------|----------|
| **Filter calls per SCF iter** | 1 (nnsclo_now=1) | Up to 3 (max_inner=3) |
| **Inner convergence check** | Wavefunction residual < tolwfr | Fresh residual max < tol_res |
| **ABINIT source** | `m_vtowfk.F90:382,690-695` | — |
| **Our source** | — | `scf.rs:982-1010` |

**ABINIT**: For SCF calculations, `nnsclo_now=1` — the Chebyshev filter is called exactly once per SCF iteration. Convergence of the wavefunction residual is left to the SCF outer loop. For non-SCF, `nnsclo_now=nstep` — multiple calls with convergence check.

**Our code**: Always runs up to 3 `chebfi_run_rust` + `rayleigh_ritz` passes per SCF iteration, checking fresh residuals for early exit.

**Practical impact**: ABINIT's approach works because the Chebyshev filter + single RR is sufficient to converge the subspace for the current V_eff. If one pass isn't enough, the outer SCF loop handles it. Our multi-pass approach may be over-converging the subspace for a stale V_eff, wasting work. Or it could be necessary because R-ChFSI convergence is slower than standard ChFSI.

**Fix direction**: After switching to standard ChFSI (C1), evaluate whether single-pass matches ABINIT's convergence behavior. If single-pass is sufficient, remove the inner loop to match ABINIT exactly.

---

### C7: Fresh Residuals vs Prior-Iteration Residuals (MEDIUM)

| Property | ABINIT | Our Code |
|----------|--------|----------|
| **Residual source** | Prior iteration's RR output | Fresh Phase 2b computation |
| **ABINIT source** | `m_chebfi2.F90:528` — eigenvalues from prior call stored in `chebfi%eigenvalues` | — |
| **Our source** | — | `chebyshev.rs:2395-2410` — recomputes H·Ψ from current V_eff |

**ABINIT**: `chebfi%eigenvalues` is set ONCE at the start of `chebfi_run` (line 528: `chebfi%eigenvalues = eigen`). The `eigen` parameter comes from the prior SCF iteration's RR output via `m_vtowfk.F90`. Residuals fed to `chebfi_set_ndeg_from_residu` are based on these prior-iteration eigenvalues.

**Our code**: Phase 2b recomputes H·Ψ fresh from current V_eff and computes residuals with fresh Rayleigh quotients. This was a deliberate fix to avoid the "cascade bug" where stale residuals from a prior SCF iteration poisoned the oracle.

**Practical impact**: Different convergence behavior. ABINIT trusts the prior iteration's eigenvalues as the convergence metric; we insist on fresh verification. Neither is wrong, but they produce different locking decisions.

---

### C21: oracle=0 Converged-Band Locking (HIGH — new divergence, 2026-06-19)

| Property | ABINIT (oracle=0) | Our Code (before fix) | Our Code (after fix) |
|----------|-------------------|----------------------|---------------------|
| **Converged bands** | All get `ndeg_filter_max` | All get `ndeg_filter_max` | ndeg=0 for bands with residual < tolerance |
| **ABINIT source** | `m_chebfi2.F90:628` — `if(oracle>0) call chebfi_set_ndeg_from_residu(...)` | — | — |
| **Our source** | — | `chebyshev.rs:2457-2459` (oracle=0 branch) | `chebyshev.rs:2457-2472` (oracle=0 with locking) |

**Discovery (2026-06-19)**: On Cu111_CO warm start (160 bands, 60k PW), 121/160 bands are already converged (residual < 1e-6). With oracle=0, ALL 160 bands get the same filter degree (~5 based on `cheb_oracle1`). Filtering already-converged bands makes them nearly identical — S_sub off-diagonals reach ~0.999, causing:
- ZPOTRF info=85 (leading minor not positive definite)
- ZHEEVD info=151 (convergence failure)
- ZHEGVD info=245 (leading minor 85 of B not positive definite)

**Why ABINIT doesn't hit this**: ABINIT recomputes H·Ψ and S·Ψ INSIDE the recurrence loop (`m_chebfi2.F90:660` — `getAX_BX` at every step). Our code uses S⁻¹·H via Woodbury LU, which is less numerically stable for near-dependent vectors. Additionally, ABINIT's matrix-free RR (rotating all three vectors) and Gram-Schmidt orthonormalization may provide better numerical stability than our Cholesky QR for this edge case.

**Our fix (2026-06-19)**: Added converged-band locking even with oracle=0 — bands with `fresh_residual[b] < tolerance` get `ndeg=0`. This is a **divergence from ABINIT** (ABINIT oracle=0 does NOT lock converged bands), but it's numerically necessary for our pipeline. The locked bands retain their original S-orthonormal state, eliminating the near-linear-dependence.

**Risk**: If the locking threshold is wrong, bands that appear converged under the current V_eff may actually need filtering. The tolerance used is the same `tolerance` parameter (1e-6), which matches the convergence criterion.

**Long-term fix**: Investigate why ABINIT's oracle=0 pipeline handles near-dependent vectors better. Possible explanations:
1. ABINIT's `getAX_BX` inside the loop recomputes H+S at every step, which may implicitly regularize
2. ABINIT's Gram-Schmidt may handle near-dependent vectors better than our Cholesky QR
3. ABINIT's matrix-free RR may provide additional numerical stabilization

---

## 3. Dead Code Audit

| # | Function | Lines | Why Dead | Recommendation |
|---|----------|-------|----------|----------------|
| D1 | `compute_spectral_bounds` | 133-180 | `chebfi_run_rust` uses ecut+Rayleigh instead; only called by `chebyshev_filter` (test path) | Keep if test path is maintained; otherwise remove |
| D2 | `lanczos_upper_bound` | 195-360 | Only called by `chebyshev_filter` (test path) | Remove — uses S⁻¹·H which is the wrong operator for our pipeline |
| D3 | `apply_scaled_hamiltonian_inplace` | 372-404 | `#[allow(dead_code)]`, never called | Remove |
| D4 | `transpose_col_to_row_on_gpu` | 475-494 | Layout convention change made transpose unnecessary | Remove |
| D5 | `transpose_row_to_col_on_gpu` | 498-517 | Same as D4 | Remove |
| D6 | `chebfi_residual_norms` | 1413-1514 | Phase 2b inside `chebfi_run_rust` replaced it; also exists as duplicate in tests | Keep for test use; add `#[cfg(test)]` |
| D7 | `gram_schmidt_s` | 3027-3098 | Identical logic inlined at two other locations | Consolidate: make other call sites use this function |

---

## 4. Dependency Map

```
C1 (R-ChFSI → standard ChFSI)
├── REQUIRES C2 (H·S⁻¹ → S⁻¹·H operator)
├── REQUIRES C4 (restore ampfactor — mandatory for standard ChFSI)
├── ENABLES C3 (matrix-free RR — needed to avoid recomputing H after RR)
├── AFFECTS C6 (inner loop — standard ChFSI may converge in 1 pass)
└── AFFECTS C7 (fresh residuals — standard ChFSI uses prior eigenvalues)

C3 (matrix-free RR)
├── REQUIRES C1 (standard ChFSI to make HΨ/SΨ rotation meaningful)
└── REQUIRES C4 (ampfactor normalizes HΨ/SΨ alongside Ψ)

C5 (Woodbury precision)
├── INDEPENDENT of C1-C4 (both algorithm types benefit)
└── MORE CRITICAL for R-ChFSI (Das 2025 Theorem 3.4 requires bounded S⁻¹ error)

C6 (inner loop)
├── DEPENDS ON C1 (standard ChFSI may need only 1 pass)
└── INDEPENDENT of C2-C5
```

### Fix Order

| Priority | ID | Description | Depends On | Estimated Effort |
|----------|----|-------------|------------|-----------------|
| **P0** | C5 | Woodbury iterative refinement → 1e-10 precision | None | ~50 lines |
| **P1** | C1+C2+C4 | Switch to standard ChFSI with S⁻¹·H + ampfactor | C5 (better S⁻¹ helps standard ChFSI too) | ~200 lines |
| **P2** | C3 | Matrix-free RR: rotate HΨ, SΨ | C1 (only meaningful for standard ChFSI) | ~40 lines |
| **P3** | C6 | Evaluate single-pass vs multi-pass inner loop | C1 | ~10 lines (change max_inner=1) |
| **P4** | D7 | Consolidate Gram-Schmidt duplication | None | ~30 lines |
| **P5** | D1-D6 | Remove dead code | None | ~20 lines (delete) |

**Rationale for P0 (C5 first)**: The Woodbury fix is self-contained, independently beneficial, and reduces a known noise source (3.4×10⁻⁶ → 1e-10) before we change the algorithm. This eliminates one variable from the cascade diagnosis.

---

## 5. Gate Tests (to write)

Following the Davidson pattern (`nio_spin_scf.rs`, `s_inv_s_identity.rs`, `chebfi_convergence.rs`):

| Gate | Test | What It Verifies | Depends On |
|------|------|-----------------|------------|
| **Gate 0** | `s_inv_s_identity` | Woodbury S⁻¹·S identity → ζ < 1e-10 | C5 fix |
| **Gate 1** | `chebfi_structural` | Pipeline produces monotonic, non-NaN eigenvalues with zero V_eff | None |
| **Gate 2** | `chebfi_convergence_loop` | 3 filter+RR iterations on frozen V_eff don't drift | C1-C4 |
| **Gate 3** | `nio_chebfi_warm_start` | Iter 1 energy matches CASTEP within 0.001 eV (matches existing) | C1-C4 |
| **Gate 4** | `nio_chebfi_convergence_trend` | 5-iteration SCF converges monotonically (< 2e-2 Ha drift) | C1-C5 |
| **Gate 5** | `nio_chebfi_cascade_check` | Iter 3+ eigenvalues don't explode to >1 Ha | C1-C5 |
| **Gate 6** | `chebfi_vs_davidson_eigenvalues` | ChFSI and Davidson eigenvalues agree within 1e-4 Ha at iter 1 | All |

---

## 6. Verification Rules

1. **ABINIT `m_chebfi2.F90` is the authoritative reference for algorithm structure.** The GPU paper (Lygatsika 2025) is a faithful description but has documentation errors in secondary details (H-application pattern, inner loop terminology). When paper and code disagree, trust the code.

2. **ABINIT `m_invovl.F90` is the authoritative reference for Woodbury S⁻¹.** The Levitt-Torrent 2015 paper describes the same algorithm. ABINIT has not changed from iterative refinement.

3. **Every ChFSI function must be verified against its ABINIT counterpart at the same line-number granularity as the Davidson audit.** The Davidson checklist (INNER_LOOP_CHECKLIST_20260606.md) established the standard: component-by-component mapping with exact line numbers in both codebases.

4. **No "simpler" shortcuts.** The Davidson audit taught us that every CASTEP/ABINIT design choice that looks redundant (separate `slice_eigenvalues`, matrix-free RR, per-band ampfactor, iterative refinement) serves a load-bearing purpose. Do not optimize away until the adversarial audit proves it's unnecessary.

5. **Gate tests must use the NiO spin-polarised fixture pattern.** The existing `nio_spin_scf.rs` demonstrates the correct pattern: `build_spin_scf_state`, per-spin-per-kpt GPU buffers, warm-start from CASTEP `.check` file, discriminator tolerances calibrated against CASTEP reference values.

---

## Appendix A: ABINIT `chebfi_run` Pseudocode (verified against source)

```
chebfi_run(X0, getAX_BX, getBm1X, eigen, occ, residu):
  1. spacedim = chebfi%spacedim
  2. eigenProblem = chebfi%eigenProblem
  3. lambda_plus = chebfi%ecut                         // line 547

  // MPI transpose: Row → Column
  4. transpose X to column distribution                // lines 551-570

  // Initial H·Ψ and S·Ψ computation
  5. getAX_BX(xXColsRows, xAXColsRows, xBXColsRows)  // line 579: H·Ψ₀, S·Ψ₀

  // Rayleigh quotients: λ_i = ⟨ψ_i|H|ψ_i⟩ / ⟨ψ_i|S|ψ_i⟩
  6. chebfi_rayleighRitzQuotients(chebfi, maxeig, mineig, ...) // line 607
  7. lambda_minus = maxeig_global                      // line 619

  // Oracle: determine filter degree
  8. ndeg_filter_max = cheb_oracle1(mineig, lambda_minus, lambda_plus, 1e-16, 40) // line 625
  9. ndeg_filter = min(ndeg_filter_max, chebfi%ndeg_filter)
  10. if oracle>0: chebfi_set_ndeg_from_residu(...)    // line 628

  // Filter center and radius
  11. center = (lambda_plus + lambda_minus) / 2        // line 634
  12. radius = (lambda_plus - lambda_minus) / 2        // line 635

  // Chebyshev recurrence loop
  13. for ideg = 0..ndeg_filter-1:
        // Apply S⁻¹·H recurrence step
        chebfi_computeNextOrderChebfiPolynom(...)       // line 644
        // Swap buffers
        chebfi_swapInnerBuffers(...)                    // line 650
        // Compute H·Ψ and S·Ψ for the new vector
        getAX_BX(xXColsRows, xAXColsRows, xBXColsRows) // line 660

  // Amplification factor normalization (per-band)
  14. chebfi_ampfactor(...)                             // line 676

  // MPI transpose: Column → Row
  15. transpose X, AX, BX to row distribution           // lines 684-701

  // Matrix-free Rayleigh-Ritz: rotates X, AX, BX
  16. xg_RayleighRitz(X, AX, BX, eigenvalues, solve_ax_bx=.true.) // line 705

  // Residual computation: ||AX - λ·BX|| or ||AX - λ·X||
  17. residu = colwiseNorm2(AX - eigenvalues * BX_or_X) // lines 710-716

  // Copy result to output
  18. X0 = X                                            // line 720
```

Key observations:
- **Line 660**: `getAX_BX` computes BOTH H and S inside the loop. The paper says H-only. The S result is unused by the recurrence but overwrites SΨ for RR use.
- **Line 676**: Ampfactor runs AFTER the loop, normalizing X, AX, BX. Critical for preventing T_n amplification distortion.
- **Line 705**: `xg_RayleighRitz` rotates all three (X, AX, BX) — the matrix-free optimization.
- **No inner loop**: `chebfi_run` is called exactly once per SCF iteration for SCF mode.

---

## Appendix B: Our `chebfi_run_rust` Pseudocode (for comparison)

```
chebfi_run_rust(psi_gpu, v_eff_dev, wave_grid, pw_coords, vnl_data,
                fft_idx_dev, ecut, min_veff, max_veff, tol_res,
                prev_residuals, prev_eigenvalues, ndeg_global, ...):
  // Phase 1: Compute H·Ψ and S·Ψ
  1. apply_s_times(psi → spsi)                          // S·Ψ
  2. apply_full_hamiltonian(psi → hpsi)                 // H·Ψ

  // Phase 2: Rayleigh quotients
  3. chebfi_rayleigh_ritz_quotients(hpsi, spsi, psi)   // λ_i = ⟨ψ|H|ψ⟩ / ⟨ψ|S|ψ⟩
  4. lambda_minus = max(ritz_values)                    // upper bound of wanted spectrum

  // Phase 2b: Fresh residuals (from current V_eff)
  5. recompute H·Ψ fresh, compute ||H·Ψ − λ·S·Ψ||²

  // Phase 3: Oracle — filter degree
  6. ndeg_max = cheb_oracle1(lambda_min_rq, lambda_minus, ecut, 1e-16, 40)
  7. ndeg_filter = min(ndeg_max, ndeg_global)

  // Phase 4: Per-band degree from fresh residuals
  8. chebfi_set_ndeg_from_residu(fresh_residuals, ritz_values, ...)

  // Phase 5: R-ChFSI recurrence (Das 2025 Algorithm 3)
  9. for k in 2..=ndeg_filter:
       // Step 3: H·S⁻¹·R_Y (apply S⁻¹ to residual copy, then H)
       apply_s_inverse(buf_c → buf_c)                   // S⁻¹·R
       apply_full_hamiltonian(buf_c → buf_c)            // H·S⁻¹·R
       // Three-term recurrence on residuals
       R_next = 2/r · (H·S⁻¹·R_curr − c·R_curr) − R_prev
       // Step 4: Reconstruction
       X_next = S⁻¹·R_Y_next + X·Λ_Y
     // Lock converged bands: copy psi_input columns for ndeg=0 bands

  // Phase 6: Gram-Schmidt S-orthonormalization

  // Phase 8: Final H·Ψ computation
  10. apply_full_hamiltonian(filtered_psi → hpsi_row)

  // Return: (psi_row, hpsi_row, ritz_values, fresh_residuals, ndeg_bands)
```

Key differences from ABINIT:
- **No ampfactor** (Phase 6 → ABINIT line 676)
- **R-ChFSI recurrence** instead of standard ChFSI (Phase 5 → ABINIT lines 641-666)
- **H·S⁻¹** operator instead of **S⁻¹·H** (Phase 5 step 3 → ABINIT line 861)
- **Fresh residuals** instead of prior-iteration (Phase 2b → ABINIT uses prior eigen)
- **Locking** by copying psi_input (Phase 5 lock → no ABINIT equivalent; ABINIT uses ndeg=0 skip)
- **Single H·Ψ at end**, no S·Ψ (Phase 8 → ABINIT recomputes both H+S every step)
