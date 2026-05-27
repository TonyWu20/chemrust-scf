# Phase 3: GPU D-Matrix Screening + Aug-Density SF Kernel

**Date:** 2026-05-27
**Status:** Draft

## Goals

1. **Root-cause diagnosis of the GPU D-screening bug** — Write a CPU-vs-GPU discriminator test that feeds identical `V_eff_fft` into both `screen_d_gpu` and `compute_screened_d_from_fft`, asserts per-element `|Δ| < 1e-12` for every ion. If the mismatch reproduces (near-zero screening for non-origin ions, per the 2026-05-23 resolution), add targeted intermediate diagnostics: download GPU structure factors and Q arrays, compare element-by-element against their CPU equivalents, isolate whether the bug is in struct-factor layout, Q flattening convention, gemv call, or kernel. Fix once identified.

2. **Wire GPU D-screening into the production path** — Replace the `compute_screened_d_from_fft` call in `VnlBatchData::precompute_with_d_override` (vnl_data.rs:241) with `screen_d_gpu`. Update `PcieAccount` to track screening H2D bytes. Remove `#![allow(dead_code)]` from `d_screening.rs`.

3. **GPU pointwise-multiply kernel for aug-density SF** — Absorb the deferred item from the phase-rchfsi review: add a `cpx_mul_inplace` CUDA kernel to `CudaKernelSet` and replace the D2H/H2D roundtrip in `compute_aug_density_gpu` where `tmp[g] *= exp(-iG·R_I)` is currently done via CPU.

4. **Validation against CASTEP anchors** — Run the existing anchor tests (SC-3 V_eff range, SC-4 band-1 eigenvalue, SC-5 density decomposition) with GPU D-screening active. Assert no regression against the CPU-path baseline.

## Scope Boundaries

**In scope:**
- Diagnosing and fixing the bug in the existing GPU D-screening code (`src/eigensolver/d_screening.rs`)
- Writing a CPU-vs-GPU discriminator test that compares `screen_d_gpu` output against `compute_screened_d_from_fft` element-by-element
- Intermediate diagnostic assertions (GPU structure factors vs CPU, GPU Q arrays vs CPU) to isolate any residual mismatch
- Wiring the fixed `screen_d_gpu` into `VnlBatchData::precompute_with_d_override`
- A `cpx_mul_inplace` kernel (element-wise complex multiply, in-place: `a[i] *= b[i]`) added to `CudaKernelSet`
- Replacing the aug-density D2H/H2D roundtrip in `compute_aug_density_gpu` with the new kernel
- Updating all callers that construct `VnlBatchData` to pass `kernels: &CudaKernelSet`
- Running existing CASTEP anchor tests to verify no regression

**Out of scope:**
- Porting `precompute_q_on_grid` (radial Bessel transforms) to GPU — stays on CPU, called once per species
- Porting V_eff FFT to GPU — continues using `chemrust-hamiltonian-core` CPU FFT
- Eliminating the D2H of `beta_psi` in `compute_aug_density_gpu` (omega computation on CPU) — separate deferred item
- General `CudaKernelSet` refactoring (moving it out of `chebyshev.rs`) — deferred to the module-split phase

## Design Notes

### Bug diagnosis strategy

The 2026-05-23 bug produced near-zero screening for non-origin ions. Ion 2 at `(0,0,0)` was immune because `exp(±iG·R) = 1` for all G, masking the error. The discriminator test should first reproduce this mismatch, then progressively narrow with intermediate diagnostics:

1. **Structure factor check**: Compute `exp(-iG·R)` on CPU using the same loop order as the GPU builder. Compare against downloaded GPU struct factors. If they disagree, the bug is in the struct-factor computation or iteration order.
2. **Q array check**: Download GPU Q arrays, compare against the CPU `q_on_grid.pairs` flattened in the same order used by the builder.
3. **w buffer check**: After `cpx_conj_mul`, download `w` and compare against CPU-computed `V_eff_fft(g) * exp(+iG·R)` for each G.
4. **tmp buffer check**: After gemv, download `tmp` and compare against CPU-computed `Σ_g conj(Q(p,g)) * w(g)` for each pair.

The most likely root cause candidates, in order of probability:
- **V_eff_fft upload order mismatch**: The CPU FFT data may be laid out in C-order (row-major) while the GPU structure factors are computed in Fortran-order (iz-fastest). The `as_recip_array()` from `ndarray` is in Fortran memory order (the ndarray default), and the GPU loops also iterate in Fortran order — but if the upload serialization doesn't match, every G-vector gets the wrong phase factor.
- **Q flattening convention**: The CPU `q_on_grid` uses `Array3<Complex64>` in Fortran order. The GPU builder iterates with `q_arr.iter()` which traverses in memory (Fortran) order. But if `precompute_q_on_grid` produces Q in a different order than the structure factors expect, the gemv would sum across mismatched G-vectors.

### Phase convention

D screening uses `exp(+iG·R)` (positive sign). The GPU structure factors store `exp(-iG·R)`. The `cpx_conj_mul` kernel computes `dst = a * conj(b)`, so with `a = V_eff_fft` and `b = ion_sf`:
```
w = V_eff_fft * conj(exp(-iG·R)) = V_eff_fft * exp(+iG·R)  ✓
```
This matches the CPU path's `sf = Complex64::from_polar(1.0, +tau * G·R)`.

### cpx_mul_inplace kernel

Simple element-wise complex multiply: `a[i].x = a[i].x * b[i].x - a[i].y * b[i].y`, `a[i].y = a[i].x_old * b[i].y + a[i].y_old * b[i].x`. Accepts the same `(double2*, const double2*, int n)` signature as `cpx_conj_mul` but operates in-place on `a`.

## Deferred Items Absorbed

- **D-6 (GPU D-matrix screening)** from `notes/pr-reviews/phase-rchfsi-bare-h/deferred.md` — Goals 1 and 2
- **GPU pointwise-multiply kernel for aug-density SF** from `notes/pr-reviews/phase-rchfsi/deferred.md` — Goal 3

## Verification

1. **Discriminator test**: CPU-vs-GPU D_screen element-by-element comparison, all ions, `|Δ| < 1e-12` per element
2. `cargo check --workspace` — must pass after each implementation step
3. `cargo clippy --workspace -- -D warnings` — must pass
4. `cargo test --release -- --ignored fixed_point_matches_castep_energy` — SC-4: band-1 iter-2 within 0.05 Ha of CASTEP reference
5. `cargo test --release -- --ignored iter2_v_eff_range_within_one_ha_of_iter1` — SC-3: V_eff range stays bounded
6. `cargo test --release -- --ignored density_decomp_matches_castep_f8_same_inputs` — SC-5: density ratios within 1% of CASTEP

## Domain Terms

None — existing glossary is adequate for this phase.
