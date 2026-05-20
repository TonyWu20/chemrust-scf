# Resolution: −2 Ha V_loc discrepancy in GPU H|ψ⟩ for Cu111_CO

**Symptom**: GPU eigenvalues for the Cu111_CO fixture come out ~2 Ha too
negative even when given the converged reference V_eff (.pot_fmt). Concrete
witness from prior session: band 1 GPU = −3.13 Ha vs CASTEP reference −1.06 Ha.

**Two root causes** identified and fixed in this session:

## Root cause 1: cuFFT plan dim ordering swapped

`BatchedFftPlan3d::plan_batched_c2c` is called as `(nx, ny, nz, ...)` and
internally passes `n = &[nx, ny, nz]` to `cufftPlanMany`. cuFFT row-major
convention: `n[0]` is the slowest-varying (outermost) dim, `n[rank-1]` is
the fastest-varying (innermost).

Our scatter index formula is `iz + ngz*(iy + ngy*ix)` (in
`pw_coords_to_fft_indices` at `src/scf.rs:1015–1029`), which makes **iz the
innermost dimension** with period `ngz`. To match this layout, the cuFFT
plan must declare `n[2] = ngz` — i.e. call `plan_batched_c2c(ngx, ngy, ngz, ...)`.

The previous code called `plan_batched_c2c(ngz, ngy, ngx, ...)`, declaring
`n[2] = ngx = 54` while our buffer's innermost period is `ngz = 90`. cuFFT
then ran the FFT along the wrong axes for the non-cubic Cu111_CO grid.
Cubic-grid unit tests passed because all three dims are equal there.

**Fix locations**:
- `src/eigensolver/chebyshev.rs:844` — swap to `(ngx, ngy, ngz)`
- `src/density.rs:147` — swap to `(ngx, ngy, ngz)`

**Empirical proof**: `cufft_dim_ordering_isolated_diagnostic` test in
`tests/ca_step_validation.rs` constructs a tiny non-cubic grid
(`ngx=3, ngy=4, ngz=6`), scatters a single δ at `(h, k, l) = (1, 1, 1)`,
runs `BatchedFftPlan3d` IFFT in all 6 axis permutations, and compares to
the analytic exponential `exp(2πi(h·ix/ngx + k·iy/ngy + l·iz/ngz))`. Only
the `(ngx, ngy, ngz)` ordering matches to 1e-15; all 5 other permutations
deviate by O(1).

## Root cause 2: Rayleigh–Ritz transpose kernel produces wrong layout

`rayleigh_ritz` builds `H_sub = ψ^H · Hψ` and `S_sub = ψ^H · ψ` via
`gemm_c64` with `transa = op::C, lda = n_pw`. This config expects the
input matrix to be **col-major (n_pw, n_bands)**: `flat[g + b*n_pw] =
psi[band b, PW g]`.

The `ColumnDistributed` GPU memory layout already satisfies this (each
band occupies a contiguous block of `n_pw` PWs: `psi.data[b*n_pw + g]`).

But `chebyshev_filter` previously used the `transpose_col_to_row` CUDA
kernel:

```c
row[g * n_bands + b] = col[b * n_pw + g];
```

This writes the input col-major (n_pw, n_bands) memory into a
**col-major (n_bands, n_pw)** layout — `flat[b + g*n_bands]`. RR's gemm
then read this with `lda = n_pw`, which scrambled the matrix.

Empirical signature (RR_DUMP_HS env var dump):
- Bare ψ^H·ψ diagonals: 8.68, 0.48, 0.17, 0.06, 0.03 (wrong)
- CPU-computed PW norms: 1.027, 0.918, 0.337, 0.339, 0.394 (right)

The "correct" diagonal per band (= ‖ψ_b‖² in PW basis) was scattered
across multiple matrix entries, producing nonsensical `H_sub` and `S_sub`
matrices that ZHEGVD then "diagonalized" to give ε_1 = −3.48 Ha.

**Fix**:
1. Skip the transpose in `chebyshev.rs:980-988` — replace the two
   `transpose_col_to_row_on_gpu` calls with `memcpy_dtod`, since
   `ColumnDistributed` memory is *already* in the col-major (n_pw,
   n_bands) layout RR expects.
2. Fix the rotation step in `rayleigh_ritz.rs:269-328` — the previous
   code did `psi_new = X · psi_col` after a wrong-layout transpose. The
   correct rotation in col-major (n_pw, n_bands) layout is
   `psi_new = psi_row · X` (gemm with `m=n_pw, n=n_bands, k=n_bands,
   lda=n_pw, ldb=n_bands, ldc=n_pw`).

## Anchor criteria used

EXTERNAL anchors (already in `notes/debug/debug-20260520-1923/CRITERIA.md`):
- `Cu111_CO.bands` band 1 = −1.05502287 Ha
- `Cu111_CO.pot_fmt` real-space V_eff (Hartree, 54×90×90 fine grid)
- `Cu111_CO.castep` total energy = −24110.96665069 eV

## Verification

After both fixes:
- `band1_v_loc_expectation_matches_castep` discriminator: ε_1 = **−1.430 Ha**
  (was −2.568 Ha original / −3.476 Ha after only the cuFFT fix). First
  threshold (−1.5 Ha) **passes**. Tighter 0.1 Ha threshold against −1.055 Ha
  still fails by 0.375 Ha — this residual is methodological (bare D0 vs
  CASTEP's screened D, see `vnl_data.rs:146-149` TODO).
- `compare_eigenvalues_bare_d0`: RMS = **1.06 Ha** (was 216 Ha pre-fixes).
  Per-band first 10 diffs: −0.37, −0.09, +0.04, +0.13, +0.23, ... (sub-eV
  scale, consistent with the missing screened D contribution).
- `compare_eigenvalues_with_reference_veff_and_screening`: RMS = **0.72 Ha**,
  passes its threshold.
- `cpu_band_v_loc_expectation`: CPU brute-force ⟨ψ|T+V_loc|ψ⟩ matches the
  GPU `apply_h_components_for_test` output (T, V_loc, V_NL all to <1e-6
  relative).
- All 27 lib unit tests pass.

## Reclassified prior-session claims

- "GPU `apply_v_loc_hamiltonian` is wrong" — partially right. The cuFFT
  plan dim ordering was indeed wrong, but the `apply_v_loc_hamiltonian`
  kernel sequence itself (scatter → IFFT → multiply → FFT → gather) was
  correct. Once the cuFFT plan was fixed, `apply_v_loc_hamiltonian`
  produced correct ⟨ψ|V_loc|ψ⟩ matching CPU brute-force.
- "Likely a normalization or indexing bug in the FFT roundtrip" —
  partially right. The bug was in the cuFFT plan dim ordering, not in
  the normalization (the `1/N_total` factor was always correct).
- The prior session missed the second bug entirely (the RR transpose
  layout issue), which was the dominant contributor to the wildly-off
  eigenvalues. The cuFFT plan fix alone made early-band eigenvalues
  *worse* than the original because correct-FFT + wrong-RR diverged
  faster than wrong-FFT + wrong-RR (the two bugs partially cancelled).

## Date

2026-05-20
