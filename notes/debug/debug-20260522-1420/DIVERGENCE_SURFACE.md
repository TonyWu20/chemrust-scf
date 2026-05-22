# Divergence Surface: D_screened explosion

## Enumeration

| # | Category | Description | Status |
|---|----------|-------------|--------|
| 1 | **Grid resolution for D_screened** | `compute_screened_d_from_fft` receives wave-grid V_eff FFT; CASTEP uses fine-grid V_eff (nlpot.f90:346-355). Q_nm also precomputed on wave_grid. | **PRIMARY SUSPECT — to be fixed** |
| 2 | **Normalization factor n_total** | `n_total = ngz*ngy*ngx` from `gvg.grid()`. If gvg=wave_grid, n_total≈60k; if fine_grid, n_total≈437k. Wrong grid → 7.3× error in ΔD magnitude. | **Consequence of #1** |
| 3 | **Structure factor sign** | `+iG·R` in compute_screened_d (nlpot.rs:362). Ruled out: sign is documented and matches nlpot.f90:383-387. | Ruled out by code comment citing nlpot.f90 |
| 4 | **V_eff downsampling aliasing** | `downsample_array_to_wave_grid` may alias sharp ion-core features of V_eff. In iter-2, the newly computed V_eff has different structure than the fixture V_eff. | **Consequence of #1** |
| 5 | **R-ChFSI norm increase in iter-1** | Norms increase (ratio > 1) in iter-1. This is expected for Chebyshev filter amplifying the target subspace. In iter-2 norms decrease — consistent with different spectral bounds. | Likely benign — monitor |
| 6 | **Gram-Schmidt vs S-orthogonalization** | Post-filter uses plain Gram-Schmidt (ψ†ψ=I), not S-orthogonalization (ψ†Sψ=I). For USPP, the correct inner product is S-weighted. | Secondary concern — not the cause of D_screened explosion |
| 7 | **beta_psi layout (column-major)** | `bp = Array2::from_shape_vec((n_expanded, n_bands).f(), ...)` uses Fortran order. If wrong, ρ_nm would be scrambled. | Ruled out: iter-1 aug density is reasonable (aug_sum=9.4e6) |

## Primary Fix Target

**vnl_data.rs:131-173**: Pass fine-grid V_eff (not wave-grid) to `compute_screened_d_from_fft`,
and use `fine_grid` (not `wave_grid`) for `precompute_q_on_grid`.

This requires:
1. `VnlBatchData::precompute` to accept the fine-grid V_eff (or the fine_grid GVectorGrid)
2. `scf.rs:diagonalize` to pass `self.v_eff` (fine-grid) instead of `v_eff_wave` (downsampled)
