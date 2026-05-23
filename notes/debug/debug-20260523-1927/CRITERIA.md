# Anchor Criteria: rayleigh-ritz-zero-validation

## Fixture Files

- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands`
  — 160 reference eigenvalues (Hartree), header: 186 electrons, Fermi −0.122443 Ha
- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check`
  — converged wavefunctions (S-orthonormal under USPP S)

## Success Criteria

### SC-1: H_sub Hermiticity
- **Assertion**: max|H_sub[i,j] − conj(H_sub[j,i])| < 1e-10 Ha over all i,j ∈ [0,n_bands)
- **Source**: Mathematical property of Hermitian H — exact for any Hermitian operator
- **Discriminator**: ZHEGVD's solution quality depends on Hermitian input; violation > 1e-10 means layout or accumulation bug

### SC-2: S_sub Hermiticity
- **Assertion**: max|S_sub[i,j] − conj(S_sub[j,i])| < 1e-10
- **Source**: S is symmetric positive-definite by construction (ψ†ψ + USPP augmentation)

### SC-3: S_sub Positive-Definiteness
- **Assertion**: min(CPU eigenvalues of S_sub) > 1e-6
- **Source**: After Gram-Schmidt, S_sub ≈ I so all eigenvalues ≈ 1.0
- **Discriminator**: Negative eigenvalue → ZHEGVD will fail or produce garbage

### SC-4: Generalized Eigenvalue Residual
- **Assertion**: ‖H_sub·X − S_sub·X·Λ‖_F / ‖H_sub‖_F < 1e-8
- **Source**: ZHEGVD direct solver guarantee — residual should be near machine epsilon
- **Discriminator**: correct ≈ 1e-14, wrong > 1e-8 → ratio 1M×

### SC-5: Orthonormality X†·S_sub·X ≈ I
- **Assertion**: max diagonal |G[i,i] − 1| < 1e-8, max off-diagonal |G[i,j]| < 1e-8
- **Source**: ZHEGVD normalizes eigenvectors w.r.t. B-inner product

### SC-6: All 160 Eigenvalues vs CASTEP
- **Assertion**: max|λ_i − CASTEP_bands[i]| < 0.05 Ha for i=0..159
- **Source**: Cu111_CO.bands, "Spin component 1" section
  - Band 0: −1.05502343 Ha
  - Band 159: 0.11531044 Ha
- **Discriminator**: 0.05 Ha gate empirically validated by `ndeg_zero_with_castep_psi_matches_bands`

### SC-7: Wavefunction S-Normalization
- **Assertion**: max|⟨ψ_b|S|ψ_b⟩ − 1| < 1e-6 for b=0..n_bands
- **Source**: CASTEP `.check` stores S-orthonormal wavefunctions (castep_check_continuation_convention.md memory)
  Note: this memory says .check wavefunctions ARE S-orthonormal under USPP S — so RR, starting from
  S-orthonormal input and applying ZHEGVD with S_sub, should produce S-orthonormal output.
- **Discriminator**: correct ≈ 0 (ZHEGVD normalization), wrong could be large (if S_sub convention wrong)
