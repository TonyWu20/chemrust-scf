# Phase G — empirical FFT-plan diagnostic and downstream findings

## Isolated cuFFT plan diagnostic (Phase G test)

`cufft_dim_ordering_isolated_diagnostic` builds a tiny non-cubic grid
`[ngx=3, ngy=4, ngz=6]`, scatters `δ(h,k,l)=(1,1,1)` using our scatter formula
`flat = iz + ngz*(iy + ngy*ix)`, runs `BatchedFftPlan3d::plan_batched_c2c`
inverse FFT in **all 6 axis permutations**, and compares the GPU output to the
analytic real-space exponential `exp(2πi(h·ix/ngx + k·iy/ngy + l·iz/ngz))`.

Result: **exactly one ordering** matches to 1e-15 — `(ngx, ngy, ngz)`. The
other 5 differ by O(1). Confirmed empirically.

So the `(ngx, ngy, ngz)` swap from current `(ngz, ngy, ngx)` is the correct
cuFFT plan ordering for our scatter formula.

## But the swap makes the full Cu111_CO test *worse*

After swapping both `src/eigensolver/chebyshev.rs:844` and
`src/density.rs:147`:

- `band1_v_loc_expectation_matches_castep`: -3.476 Ha (current: -2.568 Ha,
  reference: -1.055 Ha).
- `compare_eigenvalues_bare_d0` RMS: 209 Ha (current: 216 Ha — both wildly
  bad).

The swap test passes in isolation but the full pipeline gets *worse* for band 1.

## Constant-V_eff sanity check (`constant_v_eff_sanity_check`)

Run with V_eff = K (constant) on Cu111_CO, ndeg=0, D=None. Predicted:
ε_b ≈ T_b + V_NL_b + K·‖ψ_b‖² (RR diagonal expectation).

For band 1: T_1=0.83 Ha, V_NL_1=0.56 Ha, ‖ψ_1‖²=1.027 (CPU diagnostic).

| K   | GPU ε_1 | Predicted | Δ |
|-----|---------|-----------|----|
| 0   | -2.51   | +1.39     | -3.90 |
| -2  | -4.43   | -0.66     | -3.77 |
| +10 | +7.05   | +11.66    | -4.61 |

GPU **does** respond to K with slope ≈ ‖ψ‖²·K (correct for V_loc round-trip
for a constant V_eff, which has no spatial pattern to scramble). But the
**baseline is shifted by ≈ -3.9 Ha** relative to T_1 + V_NL_1.

## V_loc with spatial pattern (full pot_fmt) — small response

| Case                    | GPU ε_1 | CPU predicted | Δ |
|-------------------------|---------|---------------|----|
| V_eff = 0               | -2.51   | +1.39         | -3.90 |
| V_eff = full pot_fmt    | -2.57   | -1.28         | -1.30 |
| Δ from K=0 to K=full    | -0.06   | -2.67         |   |

**Going from V_eff=0 to V_eff=full Cu111_CO V_eff, the GPU eigenvalue moves
by only -0.06 Ha — but CPU brute-force computes ⟨ψ_1|V_loc|ψ_1⟩ = -2.67 Ha**.
A constant-V_eff (no spatial pattern) gives the correct round-trip; a
spatially varying V_eff is scrambled.

## Hypothesis: TWO independent bugs

1. **cuFFT plan dim ordering** — confirmed by the isolated Phase G test.
   Need to swap to `(ngx, ngy, ngz)`. **But this swap alone is insufficient.**

2. **Static -3.9 Ha offset** in eigenvalues independent of V_eff. Likely
   in V_NL or kinetic path. CPU CV_NL = +0.56 Ha for band 1; GPU + RR
   produces something effectively -3.4 Ha lower than expected. With
   V_eff=0 the only contributors are T and V_NL → bug in one of those.

3. **V_eff spatial-pattern scrambling** — even with the cuFFT plan dim
   correct, the spatially-varying V_eff isn't producing the expected
   integral. Could be V_eff's **physical layout** vs cuFFT's interpretation
   not aligning despite the plan being right.

## Test artifacts in `tests/ca_step_validation.rs`

- `cufft_dim_ordering_isolated_diagnostic` — 6-permutation analytic FFT test
- `constant_v_eff_sanity_check` — env var VEFF_K to vary; ndeg=0, D=None
- `band1_v_loc_expectation_matches_castep` — discriminator (currently fails)
- `compare_eigenvalues_bare_d0` — bare D0 + reference V_eff (RMS 200+ Ha)
- `cpu_band_v_loc_expectation` — CPU brute-force G-G' diagnostic (band 1: T=0.83, V_loc=-2.67)
- `cpu_vnl_expectation` — CPU V_NL via chemrust-hamiltonian (band 1: V_NL=+0.56)

## Recommended next steps

1. **Pause and consult user** — multiple iterations of the cuFFT-fix
   hypothesis have shown only partial relief; there's clearly a second bug
   compounding.
2. **Audit V_NL + kinetic at the per-band level** — write a CPU-vs-GPU
   `apply_full_hamiltonian` test on a single band that compares
   ⟨ψ_b|H|ψ_b⟩ from GPU's hpsi_dev against CPU brute-force.
3. **Check if `init_kinetic` is double-counting** — the kernel does
   `hpsi[tid].x = psi[tid].x * k` (overwrite, not accumulate). Then V_loc
   gather uses `result[tid].x += v.x * inv_ntotal` (accumulate). So
   T → V_loc → V_NL all accumulate via `+=`. Correct.

Date: 2026-05-20
