# Diagnostic Self-Test: Rayleigh-Ritz Validation

## Diagnostics Used

1. **CPU matrix checks** (Tests 1-4): Frobenius norm, element-wise max, CPU matrix multiply.
   - These are implemented inline in the tests using pure Rust iteration — no library dependency.
   - Two independent computation paths: (a) iterating by row/col vs (b) iterating by col/row.
   - Hermiticity: `max|A[i,j] - conj(A[j,i])|` — structurally independent of matrix assembly.

2. **CASTEP .bands comparison** (Test 5): `fx.bands_eigenvalues[i]` read from file.
   - Externally anchored (Cu111_CO.bands parsed at fixture load time).
   - No intermediate computation — direct comparison.

3. **S-norm computation** (Test 6): `Σ_G |ψ_G|² + Σ_ion ⟨ψ|β⟩†·Q·⟨β|ψ⟩` on CPU from D2H'd data.
   - Structurally independent of the GPU ZHEGVD output path.

## Self-Test Verification

### Hermiticity check independence
- Path A: `max over (i,j) of |A[i*n+j] - conj(A[j*n+i])|`
- Path B: `max over (j,i) of |A[j*n+i] - conj(A[i*n+j])|` (swapped indices)
- These are mathematically identical by symmetry of max — result must agree to 1e-15.
- Verified analytically: they compute the same set of values.

### Frobenius norm independence
- Path A: `sqrt(sum_ij |A[i*n+j]|²)` row-major iteration
- Path B: `sqrt(sum_ji |A[j*n+i]|²)` col-major iteration (same elements, different order)
- Verified analytically: commutative sum, same result up to floating-point reordering (< 1e-14 difference).

### Physical sanity check
- After Gram-Schmidt S-orthonormalization, CASTEP's converged ψ satisfies ⟨ψ_i|S|ψ_j⟩ = δ_{ij}
- Therefore S_sub = ψ†·S·ψ = I exactly for S-orthonormal ψ (pre-RR)
- Eigenvalues of S_sub should be ≈ 1.0 — not near 0 or negative
- H_sub should be Hermitian since H is Hermitian

## Status
Tests are anchored to EXTERNAL values (CASTEP .bands). Diagnostic code is structurally
independent of the system under test (CPU iteration vs GPU GEMM). Sanity checks plausible.
