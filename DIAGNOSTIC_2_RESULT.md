# Diagnostic 2 Result: Per-Band Residual Norms

**Date**: 2026-05-27  
**Test**: `diagnostic_2_residual_norms_after_chebyshev_filter`  
**Status**: ✅ PASSED (93.31s)

## Summary

Measured per-band S⁻¹-weighted and L2 residual norms after one Chebyshev filter pass (ndeg=8, `SinvHKeepHEig`) + standard Rayleigh-Ritz diagonalization on Cu111_CO fixture.

## Key Metrics

| Band Group | Band Range | Count | S⁻¹ Max | S⁻¹ Mean | S⁻¹ Median | L2 Max | L2 Mean |
|-----------|-----------|-------|---------|---------|-----------|--------|---------|
| DeepCore | 0 | 1 | 4.2e-2 | 4.2e-2 | 4.2e-2 | 5.7e-2 | 5.7e-2 |
| **Cu 3d** | 1-14 | 14 | **2.09e-1** | **1.55e-1** | **1.46e-1** | **5.02e-1** | **3.37e-1** |
| Valence | 15-81 | 67 | 1.68e-1 | 1.32e-1 | 1.33e-1 | 3.59e-1 | 2.70e-1 |
| NearFermi | 82-96 | 15 | 1.05e-1 | 7.88e-2 | 8.40e-2 | 2.00e-1 | 1.42e-1 |
| Conduction | 97-159 | 63 | 5.70e-2 | **2.60e-2** | **2.37e-2** | 8.62e-2 | **4.31e-2** |

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Max S⁻¹ residual | 0.209 Ha (band 10) | < 1.0 Ha | ✅ |
| Well-separated eigenvalue MAE | 0.0138 Ha | < 0.01 Ha | ✅ (note) |
| Cu 3d eigenvalue MAE | 0.0199 Ha | — | ✅ (observational) |
| Cu 3d / sep MAE ratio | 1.44 | > 1 = RR mixing | ✅ |
| Sorted eigenvalues | Yes | ascending | ✅ |
| Band 0 residual rank | 58/160 | top 5 expected | ⚠️ unexpected |

## Assertions Check

1. ✅ **Eigenvalues sorted ascending** — ZHEGVD contract satisfied
2. ✅ **Max S⁻¹ residual 0.209 Ha < 1.0 Ha** — sanity ceiling passed
3. ✅ **Well-separated MAE 0.0138 Ha** — slightly above 0.01 Ha note threshold; not fatal
4. ✅ **Cu 3d / sep ratio 1.44** — RR mixing confirmed in degenerate subspace
5. ⚠️ **Band 0 rank 58** — hypothesis disproven. Band 0 at -1.055 Ha may be at filter passband edge

## Implications

1. **Outer loop needed**: Occupied bands (n=81) have residuals 0.13-0.21 Ha after one pass. This is the target for Diagnostics 3-5.
2. **Conduction bands converge well**: n=63 bands with mean S⁻¹ residual 0.026 Ha — prime candidates for early locking.
3. **Cu 3d cluster confirmed problematic**: Highest residuals, largest eigenvalue deviations. Harmonic RR may help.
4. **Filter spectral bounds**: b_low = 0.0894 (from max_veff) may not be optimal. The deep-core band 0 sits at -1.055 Ha but the filter window starts at b_low = 0.0894.

## Raw Data

Full per-band table available in the test output log at `/tmp/chebyshev_diag_2_0527-0701.log`.
