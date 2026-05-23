# Divergence Surface: Rayleigh-Ritz Mathematical Validation

## Potential Divergence Points vs CRITERIA.md

### 1. H_sub Assembly (Steps 1-1 of rayleigh_ritz.rs:72-102)
- **Data layout / axis ordering**: psi_row is col-major (n_pw, n_bands); transa=C computes psi†·hpsi
  - If psi is row-major by mistake → H_sub would be garbage
  - **Status**: To be tested (SC-1)

### 2. S_sub Assembly (Steps 2-2b of rayleigh_ritz.rs:104-202)
- **Data layout**: Same psi layout issue as H_sub
- **Augmentation convention**: USPP Q matrix must match the S-norm convention CASTEP uses
  - C_proj†·Q·C_proj is the correct form
  - **Status**: To be tested (SC-2, SC-3)

### 3. ZHEGVD call (rayleigh_ritz.rs:211-226)
- **Fill mode**: CUBLAS_FILL_MODE_LOWER — standard for LAPACK-style packed Hermitian
- **Argument order**: A=H_sub, B=S_sub. ZHEGVD solves A·X = λ·B·X. Correct.
- **In-place overwrite**: h_sub_dev → X after return. Captured in rayleigh_ritz_with_matrices.
- **Status**: To be tested (SC-4, SC-5)

### 4. Eigenvector Rotation psi_new = psi_row · X (rayleigh_ritz.rs:244-265)
- **Layout**: X is col-major (n_bands, n_bands) in h_sub_dev after ZHEGVD
- psi_row is col-major (n_pw, n_bands), gemm with transN×N gives col-major (n_pw, n_bands) = psi_new
- **S-normalization**: If X†·S_sub·X = I and ψ_row is S-orthonormal, then ψ_new = ψ_row·X is S-orthonormal
- **Status**: To be tested (SC-7)

### 5. All 160 Eigenvalues vs CASTEP (SC-6)
- The existing test `ndeg_zero_with_castep_psi_matches_bands` validates first 10 bands
- Extension to all 160 bands is new — ruled out only if SC-1 through SC-5 pass
- **Status**: To be tested (SC-6)

## Items Ruled Out

- **cuFFT ordering bug**: Fixed in commit 5037e64 (failure-patterns.md). Ruled out by anchor.
- **Transpose layout bug**: Fixed in commit (failure-patterns.md cufft-dim-ordering-and-rr-transpose-layout). Ruled out by the existing passing test ndeg_zero_with_castep_psi_matches_bands which proves first 10 RR eigenvalues are correct.
- **D-screening bug**: Fixed (failure-patterns.md iter1-filter-operator-mismatch). Ruled out.
- **Electron count diagnostic formula bug**: Fixed §11b. Ruled out.
- **Per-band eigenvalue branches**: Fixed §11a. Ruled out.
