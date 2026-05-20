# Divergence Surface: eigensolver eigenvalue accuracy

Enumeration of every place our pipeline could disagree with CASTEP/paper reference for the observed eigenvalue symptom class.

## 1. Spectral bounds computation

### 1a. Upper bound λ_max — Gershgorin vs Lanczos
- **Our code** (`chebyshev.rs:251`): `lambda_max = kinetic_max + (max_veff - min_veff)`
- **Zhou 2014 §3/Algorithm 4.1**: `b_up = max λ_i(T_k) + ||f||_2` from k-step Lanczos
- **Impact on eigenvalues**: Overestimated λ_max is safe (maps all states further into [-1,1]), but makes filter ineffective (no discrimination)
- **Classification**: Ruled out by anchor C1 — overestimated λ_max cannot cause wrong eigenvalues, only slower convergence. The filter preserves eigencomponents proportionally; Rayleigh-Ritz subspace is unchanged.

### 1b. Lower bound eps_cut — placement relative to Fermi level
- **Our code** (`chebyshev.rs:254-266`):
  - First SCF: `eps_cut = lambda_max / 3.0` (~44 Ha for Cu111_CO → above all eigenvalues)
  - Subsequent: `eps_cut = e_last + 0.2*(e_last - e0)` clamped to `0.95*lambda_max`
- **Zhou 2014 §5**: `b_low = β·min λ_i(T_k) + (1-β)·max λ_i(T_k)` from Lanczos Ritz values
- **Impact**: Our eps_cut is always ABOVE all 160 eigenvalues → ALL states magnified, NONE damped. Filter is a no-op.
- **Classification**: Ruled out by anchor C1 — placing eps_cut above all eigenvalues makes the filter a uniform scaling, which preserves the subspace. Rayleigh-Ritz recovers correct eigenvalues regardless.

### 1c. No scaled filtering (no a_L)
- **Our code**: Uses unscaled Algorithm 3.1 (eq. 8 in Zhou 2014)
- **Zhou 2014 Algorithm 4.1**: Uses scaled Algorithm 3.2 with a_L (eq. 10)
- **Impact**: No overflow possible since eps_cut >> |eigenvalues|. No impact on eigenvalue accuracy.
- **Classification**: Ruled out — scaling only matters when eigenvalues are far from [-1,1], which is already the case with overestimated bounds.

## 2. Hamiltonian application (T + V_loc + V_NL)

### 2a. Kinetic energy per plane-wave
- **Our code** (`chebyshev.rs:286-303`): `compute_kinetic_energies()` computes 0.5*|G|² for each PW using pw_coords + recip_lattice → `KineticEnergies` newtype
- **CASTEP reference**: Same formula (0.5*(G+k)²), computed identically from G-vector positions
- **Fixed in**: `5037e64` — was previously computing from full-grid g2(), giving wrong indices
- **Classification**: TO BE TESTED — the fix changed from grid-based to PW-based indexing, but correctness needs GPU validation against anchor C1. If CUDA kernel uses fft_idx to access kinetic[], the indexing is fixed.

### 2b. FFT index mapping (scatter/gather)
- **Our code** (`scf.rs:1017`): `pw_coords_to_fft_indices` uses Fortran-order: `iz + ngz*(iy + ngy*ix)`
- **cuFFT convention**: cuFFT uses Fortran-order (column-major), dimensions [ngz, ngy, ngx]
- **Fixed in**: `5037e64` — was previously C-order `ix + ngx*(iy + ngy*iz)`
- **Classification**: TO BE TESTED — correct Fortran-order formula should match cuFFT expectation. Needs GPU validation.

### 2c. FFT plan dimensions
- **Our code** (`chebyshev.rs:707`): `plan_batched_c2c(ngz, ngy, ngx, ...)`
- **cuFFT**: First dimension is innermost (Fortran order), so [ngz, ngy, ngx] matches our fft_idx convention
- **Classification**: Ruled out — dimension order is consistent with Fortran-order index computation. If this were wrong, the V_eff multiply would apply at wrong positions, producing garbage.

### 2d. V_eff multiply — normalization factor inv_ntotal
- **Our code**: `inv_ntotal = 1.0 / grid_size` applied in `gather_add_kinetic`
- **Expected**: FFT roundtrip normalization: 1/N_total for forward+inverse pair
- **Classification**: Ruled out — standard FFT normalization. If wrong, all eigenvalues would scale by a constant factor, not show band-dependent errors.

### 2e. V_NL application (non-local pseudopotential)
- **Our code** (`chebyshev.rs:457-541`): cuBLAS gemm with beta_g projectors and D matrix
- **Steps**: C_proj = beta^H · psi, C_proj = D · C_proj, hpsi += beta · C_proj
- **Components that could diverge**:
  - `compute_beta_g` in chemrust-hamiltonian: G-vector matching, beta projector formula
  - `build_d0_expanded`: D0 matrix assembly
  - D matrix screening with occupations (when `occupations` is `Some`)
  - `ZgemmConfig` parameters: lda, ldb, ldc, transa, transb
- **Classification**: TO BE TESTED — V_NL is the most complex part of H|psi>. Errors here would produce band-dependent eigenvalue biases (heavier ions have stronger V_NL contributions for specific bands).

### 2f. Band index layout in CUDA kernels
- **Our code**: PW coefficients stored as (n_bands, n_pw) col-major → `psi[b*n_pw + g]`
- **Classification**: Ruled out by consistency — same indexing in scatter, gather, init_kinetic. If wrong, errors would be catastrophic (all bands wrong), not band-dependent.

## 3. Rayleigh-Ritz subspace diagonalization

### 3a. H_sub and S_sub computation
- **Our code** (`rayleigh_ritz.rs:73-118`): cuBLAS gemm with transa=C (conj-transpose psi), transb=N
- **Expected**: H_sub = psi^dag · H·psi, S_sub = psi^dag · psi, both n_bands×n_bands
- **Classification**: TO BE TESTED — gemm parameters must be correct. psi_row is (n_pw, n_bands) col-major; psi^dag should be (n_bands, n_pw).

### 3b. ZHEGVD solver call
- **Our code** (`rayleigh_ritz.rs:127-135`): cuSOLVER ZHEGVD with CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER
- **Potential issue**: Fill mode (LOWER vs UPPER) — H_sub and S_sub are Hermitian, ZHEGVD reads only the specified triangle
- **Classification**: TO BE TESTED — if the fill mode is wrong, ZHEGVD reads garbage for the other triangle. H_sub and S_sub should be exactly Hermitian (computed from psi^dag·psi and psi^dag·H·psi).

### 3c. Subspace rotation (X · psi)
- **Our code** (`rayleigh_ritz.rs:182-203`): psi_new = X · psi_col via gemm
- **Classification**: Ruled out — straightforward gemm. If wrong, psi would be garbage (not just wrong eigenvalues).

### 3d. Eigenvalue ordering after ZHEGVD
- **cuSOLVER ZHEGVD**: Returns eigenvalues in ascending order (documented behavior)
- **Our code**: No re-sorting, trusts ZHEGVD ordering
- **Classification**: Ruled out — ZHEGVD guarantees ascending order.

## 4. Input preparation (V_eff downsampling, occupations)

### 4a. V_eff downsampling from fine grid to wave grid
- **Our code** (`scf.rs:1042-1072`): FFT → G-space truncation → inverse FFT
- **Classification**: Ruled out by anchor C1 — the test with reference .pot_fmt bypasses V_eff assembly entirely. For anchor C2, V_eff assembly has known ~0.27 Ha RMS residual (documented in chemrust-hamiltonian tests).

### 4b. Occupation numbers for D matrix screening
- **Our code** (`vnl_data.rs:80-106`): D_screen = D0 + Σ_b occ_b · c_proj[b] · c_proj[b]^H
- **Classification**: TO BE TESTED — the screening formula correctness depends on the c_proj computation.

## 5. Spectral bounds effect on eigenvalue accuracy (detailed analysis)

**Question**: Can wrong spectral bounds cause the observed pattern (band 1 off by 0.5 Ha, band 3 off by 0.2 Ha, band 5+ completely wrong)?

**Analysis**: The Chebyshev filter C_m(L(H)) is a polynomial in H. It shares eigenvectors with H. Applying it to a subspace:
- Preserves the eigenvector directions (no mixing)
- Scales each eigencomponent by C_m(L(λ_i))

After normalization, the filtered subspace spans the SAME set of eigenvectors as the input subspace. Rayleigh-Ritz in this subspace should recover correct eigenvalues.

**Exception**: If some eigencomponents are damped to near-zero (C_m(L(λ_i)) ≈ 0 for some i), those components are lost to floating-point precision. But:
- For our bounds (eps_cut ~44 Ha, all λ_i in [-1, 3] Ha): L(λ_i) < -1.94, |C_m(-1.94)| ≫ 1 for m=8
- No component is damped; all are amplified

**Conclusion**: Spectral bounds are RULED OUT as the cause of eigenvalue errors. The filter is a no-op with current bounds. Wrong eigenvalues must originate from the Hamiltonian application or the Rayleigh-Ritz step.

## Summary

| Item | Classification | Next step |
|------|---------------|-----------|
| 1a. λ_max estimation | Ruled out | — |
| 1b. eps_cut placement | Ruled out | — |
| 1c. No scaled filtering | Ruled out | — |
| 2a. Kinetic energy indexing | TO BE TESTED | Run GPU test against anchor C1 |
| 2b. FFT index mapping | TO BE TESTED | Run GPU test against anchor C1 |
| 2c. FFT plan dimensions | Ruled out | — |
| 2d. FFT normalization | Ruled out | — |
| 2e. V_NL application | TO BE TESTED | Compare V_NL contribution per-band |
| 2f. Band index layout | Ruled out | — |
| 3a. H_sub / S_sub gemm | TO BE TESTED | Verify gemm params with small known matrix |
| 3b. ZHEGVD fill mode | TO BE TESTED | Check fill mode vs actual matrix storage |
| 3c. Subspace rotation | Ruled out | — |
| 3d. Eigenvalue ordering | Ruled out | — |
| 4a. V_eff downsampling | Ruled out (C1 bypasses) | — |
| 4b. D matrix screening | TO BE TESTED | Compare screened vs unscreened results |
