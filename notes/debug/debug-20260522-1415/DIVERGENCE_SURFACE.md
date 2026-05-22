# Divergence Surface: SCF eigenvalue drift at iter-2

## Symptom chain (from `/tmp/scf-diag-0522-1344.log`)

```
Iter-1: fixture V_eff (.pot_fmt) → Lanczos stable → b_up=20.8 Ha ✓ → Chebyshev amplifies ✓
  → RR eigenvalues [-0.95, 1.36] Ha ✓ → density 164.5 e⁻ (deficit) → D_screened normal ✓

Iter-2: our V_eff (from iter-1 density) → Lanczos explodes → b_up=134 Ha ✗
  → Chebyshev damps ✗ → RR eigenvalues [-14.42, 5.45] Ha ✗
  → density 180.8 e⁻ → D_screened 189-455 Ha ✗
```

**Note**: wave_grid == fine_grid == [54, 90, 90] for this fixture. No fine/wave
grid interpolation or downsampling is involved. The grid-resolution hypothesis
(wave-grid vs fine-grid V_eff for D_screened) is ruled out.

## Enumeration

| # | Category | Description | Status |
|---|----------|-------------|--------|
| **1** | **Inner product convention** | `lanczos_upper_bound` at `chebyshev.rs:447` computes αⱼ = Re(⟨v, S⁻¹·H·v⟩) using standard dot product. S⁻¹·H is NOT Hermitian under the standard inner product — Lanczos breaks down. Correct: αⱼ = ⟨v, H·v⟩ (before S⁻¹), β = √⟨r, S·r⟩ (S-norm). | **PRIMARY SUSPECT** |
| **2** | **Per-ion vs global S⁻¹** | `apply_s_inverse` (chebyshev.rs:795-873) applies Woodbury per-ion independently. CASTEP builds global P = -Q·(I+B·Q)⁻¹ including cross-ion projector overlaps. | To be tested after fix #1 |
| **3** | **β-norm (L2 vs S-norm)** | Lines 462-463 and 467-469: β = ‖r‖₂ (L2 norm). Should be β = √⟨r, S·r⟩ (S-norm). S-norm ≥ L2-norm for USPP (S = I + positive correction), so current β is systematically too small. | **Part of fix #1** |
| **4** | **Starting vector normalization** | Lanczos starting vector is L2-normalized (‖v₀‖₂ = 1). For S-inner-product Lanczos, must be S-normalized (‖v₀‖_S = 1). For USPP, S-norm ≈ 1.014 × L2-norm — small but systematic error. | **Part of fix #1** |
| **5** | **Gram-Schmidt uses L2, not S-inner product** | Post-filter orthogonalization (`chebyshev.rs:1569-1613`) uses `cublasZdotc` (L2). For USPP the correct inner product is S-weighted. Post-RR bands are S-orthonormal (from ZHEGVD) but have L2-norm < 1 → density integral deficit. | Secondary — affects density, not Lanczos. To be addressed after #1 |
| **6** | **R-ChFSI recurrence coefficients** | sigma1 = e/(λ_min - c), sigma2 = 1/(γ - σ_cur), coeff = 2σ₂/e, etc. Not validated against small-matrix test case. Sign errors in Lambda_X/Lambda_Y formulas would affect filter action. | To be tested after fix #1 if drift persists |
| **7** | **First-iteration Λ_Y formula (eigenvalues = None)** | Lines 1456-1457: lam_y = -σ₁·c/e for all bands when eigenvalues = None. Derivation from Algorithm 3 needs verification. | To be tested |
| **8** | **D_screened computation** | `compute_screened_d_from_fft` in `chemrust-hamiltonian` uses wave-grid V_eff FFT. For this fixture wave_grid == fine_grid, so no resolution mismatch. The D_screened explosion is a **consequence** of wrong V_eff from wrong eigenvalues. | Ruled out as root cause |
| **9** | **Density FFT normalization factor** | `inv_omega = 1.0` in density construction. Ruled out by `density_decomp_matches_castep_f8_same_inputs` (ratio 1.000000). | Ruled out |
| **10** | **Gershgorin cap behavior** | When Lanczos b_up_raw > Gershgorin, code caps at Gershgorin (134 Ha instead of 244 Ha in iter-2). This is a safety mechanism — fixing Lanczos removes the need for the cap. | Ruled out (safety net, not cause) |

## Prioritized test plan

1. **Fix #1 (Lanczos S-inner product)** — changes 4 code locations in `lanczos_upper_bound`:
   - Move alpha dot product before `apply_s_inverse`
   - S-norm for beta in main loop
   - S-norm for beta in last-step residual
   - S-normalize starting vector
2. **Write tight test**: Assert `max(|alpha|) < 20` and `b_up < 30` at iter-2. Confirm it FAILS on current code.
3. **Apply fix** — confirm tight test PASSES.
4. **Run full SCF test** — verify C1-C6 from CRITERIA.md.
5. **If drift persists**: validate R-ChFSI recurrence coefficients (items #6-7).
6. **If D_screened still explodes**: implement global S⁻¹ (item #2).
