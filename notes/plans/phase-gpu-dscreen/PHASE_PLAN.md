# Phase GPU-DScreen: GPU D-Matrix Screening + Aug-Density SF Kernel

**Date:** 2026-05-22
**Status:** Draft

## Goals

1. **GPU D-matrix screening** — Port `compute_screened_d_from_fft` (the formula `D = D0 + (1/N) Re{ Σ_G V_eff_fft(G) · exp(+iG·R) · conj(Q_nm(G)) }`) from a CPU scalar loop to a GPU kernel + cuBLAS gemv. Reuses the existing `QSfCache` pattern (species-shared Q on GPU, per-ion structure factors) but on the wave grid rather than the fine grid. The per-ion D2H shrinks from ~6 MB (full Q arrays) to ~3 KB (final D matrix values). **Target: eliminates the ~3s CPU stall per SCF iteration.**

2. **GPU pointwise-multiply kernel for aug-density SF** — Add a `cpx_mul_inplace` kernel to eliminate the D2H/H2D roundtrip in `compute_aug_density_gpu` (density.rs:551-566) where `tmp[g] *= exp(-iG·R_I)` is done via CPU. A 20-line CUDA kernel. Absorbs the deferred item from the phase-rchfsi review: "Add GPU pointwise-multiply kernel for aug density structure factor."

## Scope Boundaries

**In scope:**
- Two new CUDA kernels added to `CudaKernelSet` in `chebyshev.rs`: `cpx_mul_inplace` (element-wise complex multiply, in-place) and `cpx_conj_mul` (element-wise `a * conj(b)`, into output buffer)
- New module `src/eigensolver/d_screening.rs` with `WaveScreeningCache` (Q on wave grid + ion structure factors on GPU) and `screen_d_gpu` (per-ion screening via kernel + gemv)
- Wire GPU D-screening into `VnlBatchData::precompute`, adding `blas: &BlasHandle` and `kernels: &CudaKernelSet` parameters
- Replace the D2H/H2D roundtrip in `compute_aug_density_gpu` with `cpx_mul_inplace` kernel launch
- Update all callers: `scf.rs` (lines 493 and 604), `ca_scf_convergence.rs` (line 668), and `compute_aug_density_gpu` call sites

**Out of scope:**
- Porting `precompute_q_on_grid` (radial Bessel transforms) to GPU — stays on CPU, called once per species
- Porting V_eff FFT to GPU — continues using chemrust-hamiltonian's CPU FFT; result uploaded to GPU
- Eliminating the D2H of `beta_psi` in `compute_aug_density_gpu` (omega computation on CPU) — separate deferred item
- Fixing bare-H R-ChFSI ndeg>0 convergence — this phase targets the debug cycle, not convergence itself
- General CudaKernelSet refactoring (moving it out of chebyshev.rs) — deferred to the module-split phase
- QSfCache disk serialization for the wave-grid cache — wave-grid Q is small enough to recompute

## Design Notes

### Phase convention

D screening uses `exp(+iG·R)` (positive sign). The existing `ion_sf` stores `exp(-iG·R)`. The `cpx_conj_mul` kernel computes `dst = a * conj(b)` directly, which with `a = V_eff_fft` and `b = ion_sf` gives `V_eff_fft * exp(+iG·R)` — correct without a separate conjugation step or intermediate buffer.

### GPU D-screening algorithm

Per ion:
1. `w[g] = V_eff_fft[g] * conj(ion_sf[g])` via `cpx_conj_mul` kernel (grid-parallel)
2. `tmp[p] = Σ_g conj(Q[p, g]) * w[g]` via cuBLAS `gemv_c64` with `trans=C` (single launch, n_lower_pairs × n_wave_grid)
3. D2H `tmp` (n_lower_pairs × 16 bytes = ~3 KB), finalize on CPU: `D[n,m] = D0[n,m] + Re(tmp[p]) / N`, symmetrize

### Parameter threading

`blas` and `kernels` are created at the top of `ScfIteration::diagonalize` (scf.rs:432-436) — before `VnlBatchData::precompute` is called at line 493. No refactoring of the caller infrastructure needed beyond adding the extra arguments.

## Deferred Items Absorbed

- **D-6 (GPU D-matrix screening)** from `notes/pr-reviews/phase-rchfsi-bare-h/deferred.md` — the primary goal of this phase
- **"Add GPU pointwise-multiply kernel for aug density structure factor"** (MEDIUM) from `notes/pr-reviews/phase-rchfsi/deferred.md` — absorbed as Goal 2

## Verification

1. `cargo check --workspace` — must pass after each implementation step
2. `cargo clippy --workspace -- -D warnings` — must pass
3. `cargo test --release -- --ignored fixed_point_matches_castep_energy` — SC-4: band-1 iter-2 within 0.05 Ha of −1.055 Ha
4. `cargo test --release -- --ignored iter2_v_eff_range_within_one_ha_of_iter1` — SC-3: V_eff range stays bounded
5. `cargo test --release -- --ignored density_decomp_matches_castep_f8_same_inputs` — SC-5: soft/aug ratios within 1% of F8
