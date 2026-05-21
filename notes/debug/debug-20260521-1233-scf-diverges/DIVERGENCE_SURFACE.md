# Divergence Surface: SCF Diverges After Iter-2

Symptom class: spatial distribution of ρ_aug (and/or V_eff downstream of it)
is wrong despite integrals matching. H apply formula is validated pointwise
(prior session, RESOLUTION debug-20260520-1923).

## Divergence-surface enumeration

### Data layout / axis ordering
- **bp_dev (β·ψ) layout out of `rayleigh_ritz` gemm**
  - gemm: `transa=op::C, transb=op::N, m=ne, n=n_bands, k=n_pw, lda=k, ldb=k, ldc=ne`.
  - Output: col-major (ne × n_bands). `bp_dev[i + j*ne]` is element `[i, j]`.
  - **To be tested in Step 7**: cross-check by D2H-ing bp_dev and indexing
    against a CPU `β·ψ` reference computed from same ψ and β_g.

- **`compute_aug_density_gpu` D2H reinterpretation**
  - `Array2::from_shape_vec((n_expanded, n_bands).f(), bp_complex)`.
  - F-order means strides (1, n_expanded). `arr[n, b] = flat[n + b*n_expanded]`.
  - Matches the gemm output layout: consistent.
  - **Ruled out** in production path (gemm + F-reinterpret are paired
    correctly). But still relevant for the `aug_density_gpu_matches_cpu_cu111_co`
    test, which constructs bp_dev differently (see below).

- **Test `aug_density_gpu_matches_cpu_cu111_co` bp_dev construction**
  - `arr` is row-major (n_expanded, n_bands) from `compute_beta_phi`
    (which uses `Array2::zeros((ne, nb))` — default C-order).
  - `arr.iter()` iterates in memory order = row-major flat order.
  - H2D'd as bp_dev → bp_dev in GPU is in **row-major** layout.
  - `compute_aug_density_gpu` then interprets as col-major → WRONG layout.
  - **To be tested in Step 7**: this is a diagnostic-self-test failure. The
    test SHOULD give different rho_aug between CPU and GPU paths but
    apparently passes. Either the cache is stale, or the test isn't
    actually green, or numerical compensation hides the layout swap.

- **`compute_aug_density_gpu` Q-matrix gemv layout**
  - Q flat: `flat[pair_idx * n_fine_grid + g_idx]` per comment at line 516-518.
  - Interpreted as col-major: element `[g, p] = flat[g + n_fine_grid * p]`.
  - This is col-major (n_fine_grid × n_pairs). gemv N takes m=n_fine_grid,
    n=n_pairs, lda=n_fine_grid.
  - **Ruled out for now** by prior fix commit `4da80ca` "correct gemv
    layout for Q_{nm} contraction" — already addressed. But need to verify
    end-to-end against CPU.

### Normalization / scaling conventions
- **ρ_aug ∫dV factor**
  - Production: GPU `compute_aug_density_gpu` returns `rho_arr` directly
    from inverse FFT without normalization (per line 572 comment: "cuFFT
    inverse is unnormalized (same as CPU fft_inverse_3d). No 1/N factor.").
  - Previous commit `fd76076` removed a spurious `1/N` normalization.
  - **To be tested**: compare GPU integral against `fft_inverse_3d` CPU
    integral on the same ρ_aug(G) input.

- **Cell-volume convention in ω**
  - ω^I_{nm} = Σ_b occ_b · conj(βψ)_n,b · (βψ)_m,b. No Ω factor.
  - CASTEP convention may or may not include Ω depending on definition.
  - **To be tested**: reference CASTEP source for ρ_aug assembly.

### Sign / direction conventions
- **Structure factor phase**: `sf = exp(-iG·R_I)` in compute_beta_g (line
  109-110 of beta_phi.rs) uses `-tau` (negative sign). Per CASTEP convention
  (ion.f90:1042-1161 cited), this should be correct.
- **inverse FFT sign**: cuFFT INVERSE uses +i sign; CPU `fft_inverse_3d`
  uses same. Need to verify.

### Boundary / edge-case handling
- **Empty species (n_expanded == 0)**: skipped in compute_aug_density_gpu
  loop. Need to verify all 18 Cu ions are processed correctly.
- **Occupations**: when n_bands > n_occupied, the tail of occupations is
  ~0 but not exactly 0 (Gaussian smearing tail). Bp · diag(occ) · bp^H
  still includes these. Should be negligible. **To be verified.**

### Unit conversion at any boundary
- **Hartree throughout**: V_eff, eigenvalues. CASTEP `.castep` energy in
  eV; converted via HARTREE_TO_EV at test boundary. **Ruled out** as cause
  of spatial-distribution bug.

### Decomposition / parallel artifacts
- **Single-GPU only**: no MPI in current code. Decomposition artifacts
  unlikely.

### Diagnostic comparison code
- **`aug_density_gpu_matches_cpu_cu111_co` (CACHE)**
  - Uses cached `rho_aug_cpu` from `/tmp/cu111_co_rho_aug_cpu.bin`.
  - If this cache was written by a buggy run, comparison is meaningless.
  - **To be checked in Step 5 (Diagnostic Verification)**: re-run CPU
    path without cache to confirm cached value matches fresh computation.

- **`[AugDensity] aug_sum=...` diagnostic in scf.rs:757-761**
  - Reports sum, min, max, and ∫ρ_aug dV ≈ aug_sum × Ω / N.
  - Summary-only; **per-point dump missing**. Per ODD pattern §Per-Point
    Diagnostics, summary without per-point backing cannot detect spatial
    distribution errors.
  - **To be addressed in Step 7**: dump full ρ_aug(r) array for iter-1
    and compare to CASTEP (`Cu111_CO.den_fmt - ρ_PW_built`) pointwise.

### Production hot-path vs test divergence
- **Production path** (`scf.rs:715-720`): `compute_aug_density_gpu(cache,
  beta_psi_gpu_from_RR, ...)`. bp_dev is col-major (gemm output) →
  consistent with `.f()` reinterpret.
- **Test path** (`tests/ca_scf_convergence.rs:311-319`): builds bp_dev_test
  via H2D of `arr.iter()` on a row-major Array2 → row-major bp_dev_test.
  Then calls same `compute_aug_density_gpu` with `.f()` reinterpret → MISMATCH.

**Critical**: the test does NOT exercise the production data path. Two
possible outcomes:
  (a) Test fails silently (assertion checks pass due to cached data
      matching, but live computation gives different result).
  (b) Test was passing in a prior session under different bp layout
      (e.g., the test used to do row-major reinterpret).

Either way, the test cannot validate post-`TASK-C1` `compute_aug_density_gpu`
semantics.

## Items ruled out by anchor (criterion → eliminated divergence)

- **C1 (total energy fixed-point)** — only checks final E; can't isolate
  any single divergence. No items ruled out.
- **H apply formula** — RULED OUT BY PRIOR DEBUG SESSION RESOLUTION
  (debug-20260520-1923). The cuFFT plan, V_loc, V_NL apply, and Chebyshev
  filter are all validated pointwise. The audit must NOT re-examine these.

## Items to test in Step 7 (in priority order)

1. **ρ_aug pointwise vs CASTEP**: compute `ρ_castep_total - ρ_PW_built`
   and compare to our `ρ_aug_built` element-wise. (Criterion C4.)
2. **bp_dev layout in production**: D2H bp_dev right after rayleigh_ritz,
   compare to CPU brute-force `β_g^H · ψ_new` per element. (Diagnostic
   self-test.)
3. **`aug_density_gpu_matches_cpu_cu111_co` re-run without cache**:
   delete `/tmp/cu111_co_rho_aug_cpu.bin`, re-run, see if pass/fail
   changes. (Diagnostic verification.)
4. **iter-2 fixed-point ρ self-consistency**: run iter-1, take output
   density, compare to fixture density pointwise. (Criterion C5.)
