# Per-Point Dump: Production Gemm Layout Verification

## Source of data

Synthetic small matrices — `tests/aug_density_layout.rs::production_gemm_path_recovers_bp_truth`
and `test_path_arr_iter_h2d_is_layout_inconsistent`.

## Production path (rayleigh_ritz gemm → compute_aug_density_gpu reinterpret)

**Dimensions**: ne=3, n_pw=8, n_bands=4.
**β_g**: row-major Array2 with deterministic values (`re = i + 0.5 + 0.7`, `im = j + 0.25 - 0.7`).
**ψ**: col-major flat with deterministic values (`re = g + 0.5 + (-0.3)`, `im = b + 0.25 - (-0.3)`).
**Production gemm**: ZgemmConfig{transa=C, transb=N, m=3, n=4, k=8, lda=8, ldb=8, ldc=3}.
**Reinterpret**: `Array2::from_shape_vec((3, 4).f(), bp_complex)`.

| Metric | Value | Conclusion |
|--------|-------|-----------|
| `max |bp_gemm - bp_truth|` | **3.178e-14** | Within numerical noise of ZGEMM. |
| Discriminator threshold | 1e-10 | Production layout is CORRECT. |

## Test data path (arr.iter() H2D → F-order reinterpret)

**Dimensions**: ne=3, n_bands=4.
**bp_logical**: row-major Array2 with deterministic values.
**Test-style upload**: H2D of `bp_logical.iter()` (row-major flat).
**Reinterpret**: `Array2::from_shape_vec((3, 4).f(), bp_complex)`.

| Metric | Value | Conclusion |
|--------|-------|-----------|
| `max |bp_interpreted - bp_logical|` | **2.828e0** | Layout is INCONSISTENT. |
| Elements disagreeing | 10/12 (83%) | Only diagonal-adjacent values happen to match. |

## Implication

1. **Production hot path is correct**: rayleigh_ritz gemm + compute_aug_density_gpu
   F-order reinterpret are layout-consistent. The previously suspected
   bp_dev layout bug is **NOT** the cause of SCF divergence after iter-2.
2. **Existing aug_density test is structurally broken**: the test in
   `tests/ca_scf_convergence.rs::aug_density_gpu_matches_cpu_cu111_co` uses
   a different data path than production and cannot validate production
   semantics. This is a separate defect that should be fixed but is NOT
   the SCF divergence cause.

## What this rules out

- Bug in `rayleigh_ritz` step-5b gemm config.
- Bug in `compute_aug_density_gpu` D2H+reinterpret.
- Bug in `compute_aug_density_gpu` omega indexing.
- Bug in Q gemv layout (Q layout matches gemv lda/m/n config — verified
  by reading `compute_q_nm_flat` source).

## What this leaves open

- ρ_aug spatial values may still be wrong despite correct ω + Q (e.g.
  inverse FFT normalisation, structure factor sign, axis order).
- V_loc apply (FFT roundtrip) may be Hermitian-violating for non-cubic
  grids in some untested path.
- Screened D matrix may not be symmetric in some edge case.
- Mixing or occupation update may corrupt state between iterations.

The strongest remaining hypothesis is that ρ_aug spatial values are
wrong despite the integral matching — but the gemm/reinterpret layer is
not the source. The bug, if it is in ρ_aug, must be in the actual
content of Q_{nm}(G), the inverse FFT, or the structure factor.
