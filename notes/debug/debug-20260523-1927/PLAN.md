# Plan: Comprehensive Rayleigh-Ritz Validation Suite (Issue #12)

## Context

The Rayleigh-Ritz (RR) eigensolver is the core of the SCF loop, responsible for solving the generalized eigenproblem H·X = ε·S·X after Chebyshev filtering. It has ~300 lines of complex CUDA/BLAS code but essentially **zero comprehensive validation**.

**Current symptom**: 135% electron count drift (55 e⁻ → 130 e⁻ vs expected 186 e⁻) between SCF iterations. Density split ratio wrong (ρ_PW 67.4% of CASTEP, ρ_aug 119%). Multiple "downstream" bugs (density construction, mixing, augmentation) traced back to upstream RR issues, but RR itself was never validated.

**Current validation (extremely weak)**:
- `h_sub_off_diagonal_magnitude` — prints H_sub/S_sub matrices, **no assertions**
- `issue_11a_iter1_band0_matches_castep` — checks **one eigenvalue** (band-0) at iter-1
- `issue_11a_iter2_lastband_does_not_overshoot` — checks **one eigenvalue** (last-band) at iter-2
- Total: 2 out of 10 bands checked, no mathematical property validation

**Missing validation** (from open-followups.md §12):
1. ❌ H_sub Hermiticity: H_sub = H_sub† (should be exact for Hermitian H)
2. ❌ S_sub Hermiticity and positive-definiteness: S_sub = S_sub†, eigenvalues > 0
3. ❌ Generalized eigenvalue equation: ‖H·X - S·X·Λ‖ < ε (residual test)
4. ❌ Orthonormality: X†·S·X = I (off-diagonal < ε, diagonal = 1)
5. ❌ All eigenvalues: compare all 10 bands against CASTEP, not just 2
6. ❌ Wavefunction normalization: ⟨ψ_new|S|ψ_new⟩ = 1 after RR
7. ❌ Occupation sum: Σ occ_i = N_e = 186 (electron count conservation)
8. ❌ Density decomposition: ρ_PW + ρ_aug split ratio matches CASTEP (when using same ψ)

## Phase 1: Understanding (Explore agents)

**Goal**: Understand the RR implementation, identify what mathematical properties should hold, and locate the fixture data needed for validation.

I will launch **2 Explore agents in parallel**:

1. **Agent 1**: Map the RR implementation
   - Find `rayleigh_ritz.rs` and trace the full RR pipeline
   - Identify where H_sub/S_sub are assembled
   - Locate ZHEGVD call and eigenvector rotation
   - Find where ⟨ψ|S|ψ⟩ normalization happens (if at all)

2. **Agent 2**: Locate fixture data and existing validation infrastructure
   - Find CASTEP reference eigenvalues (`.bands` file)
   - Locate converged wavefunctions (`.check` file)
   - Identify existing test infrastructure in `tests/ca_scf_convergence.rs`
   - Check if fixture loader provides access to CASTEP's H_sub/S_sub or just final eigenvalues

## Phase 2: Design (Plan agent)

**Goal**: Design a comprehensive validation suite that covers all 8 missing validation categories.

I will launch **1 Plan agent** to:
- Design test structure for each of the 8 validation categories
- Determine which tests can use fixture data directly vs need to run RR
- Specify discriminator thresholds (what values separate correct from incorrect)
- Identify any infrastructure gaps (e.g., need to expose H_sub/S_sub from RR for inspection)
- Consider test ordering (some tests may depend on others passing first)

## Phase 3: Review & User Questions

I will:
- Read critical files identified by agents (`rayleigh_ritz.rs`, fixture loader, existing tests)
- Verify the plan covers all 8 validation categories from §12
- Ask user questions about:
  - Priority: should all 8 categories be implemented, or start with a subset?
  - Scope: should this include fixing any bugs found, or just add tests that expose them?
  - Fixture availability: confirm CASTEP `.bands` and `.check` files are accessible

## Phase 4: Final Plan

The final plan will include:
- **Test structure**: One test per validation category, organized in dependency order
- **Fixture requirements**: Which CASTEP files are needed and how to access them
- **Infrastructure changes**: Any modifications to RR code to expose internal state for testing
- **Discriminator thresholds**: Specific numerical gates for each test
- **Verification**: How to run the new test suite and interpret results

## Exploration Complete

Both Explore agents have completed their investigation:

**Agent 1 (fixture-validator-finder)**: Located all CASTEP fixture data and existing test infrastructure. Key findings:
- CASTEP `.bands` file provides 160 reference eigenvalues (external anchor)
- CASTEP `.check` file provides converged wavefunctions
- Fixture loader (`Cu111CoFixture`) provides full access to all data
- Existing test `ndeg_zero_with_castep_psi_matches_bands` validates RR-only path
- Gap: No tests for H_sub/S_sub Hermiticity, orthonormality, or residuals

**Agent 2 (rr-implementation-mapper)**: Mapped the full RR pipeline from Gram-Schmidt through eigenvector rotation. Key findings:
- RR pipeline: `rayleigh_ritz.rs:72-322` (H_sub/S_sub assembly → ZHEGVD → rotation)
- Gram-Schmidt: `chebyshev.rs:1708-1800` (S-orthonormalization before RR)
- Current validation: Only ZHEGVD convergence check (`info == 0`)
- Gaps: No Hermiticity, orthonormality, residual, or eigenvalue reasonableness checks

## Design Complete

**Agent 3 (rr-validation-designer)**: Designed a comprehensive 6-test validation suite covering all missing mathematical properties.

### Test Suite Architecture (Layered)

**Layer 1 - Matrix Properties** (independent, fast):
1. **H_sub Hermiticity**: H_sub = H_sub† (threshold: 1e-10 Ha)
2. **S_sub Hermiticity & Positive-Definiteness**: S_sub = S_sub†, eigenvalues > 0 (threshold: 1e-6)

**Layer 2 - ZHEGVD Solution Quality** (depends on Layer 1):
3. **Generalized Eigenvalue Residual**: ‖H·X - S·X·Λ‖_F < 1e-8
4. **Orthonormality**: X†·S·X = I (threshold: 1e-8)

**Layer 3 - Full Pipeline Output** (depends on Layers 1-2):
5. **All-Band Eigenvalue Validation**: All 160 bands vs CASTEP (threshold: 0.05 Ha)
6. **Wavefunction Normalization**: ⟨ψ_new|S|ψ_new⟩ = 1 (threshold: 1e-6)

### Infrastructure Changes Required

1. **Expose internal matrices**: Add `rayleigh_ritz_with_matrices()` variant that returns `(H_sub, S_sub, X)` for inspection
2. **S-norm helper**: Add `compute_s_norm_squared()` for Test 6
3. **CPU linear algebra utilities**: Frobenius norm, matrix multiply, eigenvalue solver for validation
4. **Test file**: New `tests/rayleigh_ritz_validation.rs` with all 6 tests

### Scope Clarification

**In scope** (this plan): Tests 1-6 (RR mathematical properties)
**Out of scope**: Tests 7-8 (occupation sum, density decomposition) — these are addressed in the separate plan `our-scf-persists-to-tingly-wall.md` which focuses on the density split discrepancy symptom.

## User Decisions

- **Scope**: All 6 tests (comprehensive validation suite)
- **Fix scope**: Tests only (expose bugs, don't fix them in this session)
- **Coordination**: Independent from density plan (parallel tracks)

---

# Final Plan: Comprehensive Rayleigh-Ritz Validation Suite

## Context

The Rayleigh-Ritz (RR) eigensolver is the core of the SCF loop, solving the generalized eigenproblem H·X = ε·S·X after Chebyshev filtering. It has ~300 lines of complex CUDA/BLAS code but **zero comprehensive validation** of its mathematical properties.

**Symptom** (from open-followups.md §12): 135% electron count drift, density split ratio wrong (ρ_PW 67.4% of CASTEP, ρ_aug 119%). Multiple downstream bugs traced to upstream RR issues, but RR itself was never validated.

**Current validation** (extremely weak):
- Only ZHEGVD convergence check (`info == 0`)
- Two eigenvalue spot-checks (band-0 at iter-1, last-band at iter-2)
- No Hermiticity, orthonormality, residual, or normalization checks

**This plan**: Add 6 comprehensive validation tests covering all mathematical properties that RR must satisfy. Tests will **expose bugs** but not fix them (fixing is a separate session).

## Critical Files

**Implementation**:
- `src/eigensolver/rayleigh_ritz.rs:1-323` — Core RR (H_sub/S_sub assembly, ZHEGVD, rotation)
- `src/eigensolver/chebyshev.rs:1708-1800` — Gram-Schmidt S-orthonormalization (pre-RR)
- `src/scf.rs:553-605` — RR call site in SCF state machine

**Fixtures**:
- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands` — 160 reference eigenvalues
- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check` — Converged wavefunctions
- `tests/fixtures/cu111_co.rs` — Fixture loader (`Cu111CoFixture`)

**Tests** (new):
- `tests/rayleigh_ritz_validation.rs` — All 6 validation tests
- `tests/test_utils.rs` — Shared linear algebra helpers (Frobenius norm, matrix multiply, etc.)

## Test Suite Architecture (Layered)

### Layer 1: Matrix Properties (Independent, Fast)

**Test 1: H_sub Hermiticity**
- **Property**: H_sub = H_sub† (Hamiltonian is Hermitian)
- **Method**: Compute max|H_sub[i,j] - conj(H_sub[j,i])| over all i,j
- **Threshold**: 1e-10 Ha (machine epsilon for f64 complex arithmetic)
- **Fixture**: CASTEP `.check` wavefunctions → run RR → extract H_sub
- **Discriminator rationale**: Hermiticity is exact for Hermitian H. Any violation > 1e-10 indicates layout bug or numerical instability.
- **Runtime**: ~5 seconds (one RR call + CPU matrix comparison)

**Test 2: S_sub Hermiticity & Positive-Definiteness**
- **Property**: S_sub = S_sub† and all eigenvalues > 0 (overlap matrix is Hermitian positive-definite)
- **Method**: 
  - Hermiticity: max|S_sub[i,j] - conj(S_sub[j,i])|
  - Positive-definiteness: compute eigenvalues via CPU LAPACK, check min(λ) > 0
- **Threshold**: Hermiticity < 1e-10, min(λ) > 1e-6
- **Fixture**: CASTEP `.check` wavefunctions → run RR → extract S_sub
- **Discriminator rationale**: After Gram-Schmidt, S_sub ≈ I, so eigenvalues should be ~1.0. Threshold 1e-6 allows for numerical drift but catches catastrophic failures (negative eigenvalues).
- **Runtime**: ~5 seconds (one RR call + CPU eigenvalue solve)

### Layer 2: ZHEGVD Solution Quality (Depends on Layer 1)

**Test 3: Generalized Eigenvalue Residual**
- **Property**: ‖H_sub·X - S_sub·X·Λ‖_F < ε (ZHEGVD solution satisfies the generalized eigenproblem)
- **Method**: 
  - Compute R = H_sub·X - S_sub·X·Λ (CPU matrix multiply)
  - Compute Frobenius norm ‖R‖_F = sqrt(Σ|R[i,j]|²)
  - Normalize by ‖H_sub‖_F to get relative residual
- **Threshold**: Relative residual < 1e-8
- **Fixture**: CASTEP `.check` wavefunctions → run RR → extract H_sub, S_sub, X, Λ
- **Discriminator rationale**: ZHEGVD is a direct solver (not iterative), so residual should be near machine epsilon. Threshold 1e-8 allows for accumulation of rounding errors in matrix multiply.
- **Runtime**: ~5 seconds (one RR call + CPU matrix ops)

**Test 4: Orthonormality (X†·S_sub·X = I)**
- **Property**: Eigenvectors X are orthonormal w.r.t. S-inner product
- **Method**:
  - Compute G = X†·S_sub·X (CPU matrix multiply)
  - Check diagonal: |G[i,i] - 1| < ε for all i
  - Check off-diagonal: |G[i,j]| < ε for all i≠j
- **Threshold**: Diagonal < 1e-8, off-diagonal < 1e-8
- **Fixture**: CASTEP `.check` wavefunctions → run RR → extract S_sub, X
- **Discriminator rationale**: ZHEGVD guarantees X†·S_sub·X = I. Any violation indicates layout bug (row-major vs col-major confusion) or ZHEGVD failure.
- **Runtime**: ~5 seconds (one RR call + CPU matrix ops)

### Layer 3: Full Pipeline Output (Depends on Layers 1-2)

**Test 5: All-Band Eigenvalue Validation**
- **Property**: All 160 eigenvalues match CASTEP reference within tolerance
- **Method**:
  - Run RR on CASTEP `.check` wavefunctions
  - Compare eigenvalues[i] vs CASTEP `.bands` eigenvalues[i] for i=0..159
  - Compute max|Δλ| and RMS(Δλ)
- **Threshold**: max|Δλ| < 0.05 Ha (SC-4-tight gate from existing tests)
- **Fixture**: CASTEP `.check` wavefunctions + `.bands` eigenvalues
- **Discriminator rationale**: Existing test `ndeg_zero_with_castep_psi_matches_bands` validates first 10 bands. This extends to all 160 bands. Threshold 0.05 Ha is empirically validated (iter-1 passes with margin).
- **Runtime**: ~5 seconds (one RR call + CPU comparison)

**Test 6: Wavefunction Normalization (⟨ψ_new|S|ψ_new⟩ = 1)**
- **Property**: Rotated wavefunctions ψ_new are S-orthonormal (each band has S-norm = 1)
- **Method**:
  - For each band b: compute ⟨ψ_b|S|ψ_b⟩ = ⟨ψ_b|ψ_b⟩ + Σ_ion ⟨ψ_b|β⟩† · q · ⟨β|ψ_b⟩
  - Check |⟨ψ_b|S|ψ_b⟩ - 1| < ε for all b
- **Threshold**: max|⟨ψ|S|ψ⟩ - 1| < 1e-6
- **Fixture**: CASTEP `.check` wavefunctions → run RR → compute S-norm of ψ_new
- **Discriminator rationale**: After RR rotation, wavefunctions should be S-orthonormal. Threshold 1e-6 allows for accumulation of rounding errors in rotation gemm and augmentation sum.
- **Runtime**: ~10 seconds (one RR call + per-band S-norm computation)

## Infrastructure Changes

### 1. Expose Internal Matrices from RR

**File**: `src/eigensolver/rayleigh_ritz.rs`

**Add new function** (alongside existing `rayleigh_ritz`):
```rust
#[cfg(test)]
pub(crate) fn rayleigh_ritz_with_matrices(
    psi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    hpsi_row: &Gpu<WavefunctionSet<RowDistributed>>,
    vnl_data: &VnlBatchData,
    n_bands: usize,
    n_pw: usize,
    kernels: &CudaKernelSet,
    pcie: &mut PcieAccount,
    solver: &SolverHandle,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> Result<
    (
        Gpu<WavefunctionSet<ColumnDistributed>>,  // psi_new
        Cpu<Vec<f64>>,                             // eigenvalues
        Vec<CudaSlice<CudaComplex>>,               // beta_psi_per_ion
        Cpu<Vec<CudaComplex>>,                     // H_sub (n_bands × n_bands, col-major)
        Cpu<Vec<CudaComplex>>,                     // S_sub (n_bands × n_bands, col-major)
        Cpu<Vec<CudaComplex>>,                     // X (eigenvectors, n_bands × n_bands, col-major)
    ),
    Error,
> {
    // ... (same as rayleigh_ritz, but D2H transfer H_sub, S_sub, X before ZHEGVD overwrites them)
}
```

**Changes**:
- Before line 204 (ZHEGVD call): clone `h_sub_dev` and `s_sub_dev` to temporary buffers
- After line 226 (ZHEGVD return): clone `h_sub_dev` (now contains X) to temporary buffer
- Before line 306 (return): D2H transfer the three temporary buffers
- Add to return tuple

**D2H overhead**: 3 × (160 × 160 × 16 bytes) = 1.23 MB (acceptable for test-only code)

### 2. S-Norm Computation Helper

**File**: `src/eigensolver/s_operator.rs` (new) or `src/density.rs`

**Add function**:
```rust
/// Compute S-norm squared: ⟨ψ|S|ψ⟩ = ⟨ψ|ψ⟩ + Σ_ion ⟨ψ|β⟩† · q · ⟨β|ψ⟩
pub fn compute_s_norm_squared(
    psi_col: &Gpu<WavefunctionSet<ColumnDistributed>>,
    beta_psi_per_ion: &[CudaSlice<CudaComplex>],
    vnl_data: &VnlBatchData,
    band_index: usize,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<f64, Error> {
    // 1. Bare PW norm: ⟨ψ|ψ⟩ = Σ_G |c_G|²
    let bare_norm_sq = /* cublasZdotc on psi_col[band_index] */;
    
    // 2. Augmentation contribution: Σ_ion ⟨ψ|β⟩† · q · ⟨β|ψ⟩
    let mut aug_contrib = 0.0;
    for (ion_idx, entry) in vnl_data.entries().iter().enumerate() {
        let beta_psi = &beta_psi_per_ion[ion_idx];
        let q = &entry.q_expanded;
        // aug_contrib += beta_psi[band_index]† · q · beta_psi[band_index]
        // (gemv + dot, or direct indexing if q is small)
    }
    
    Ok(bare_norm_sq + aug_contrib)
}
```

**Usage**: Test 6 calls this for each band to validate ⟨ψ_b|S|ψ_b⟩ ≈ 1.

### 3. CPU Linear Algebra Utilities

**File**: `tests/test_utils.rs` (new)

**Add functions**:
```rust
/// Frobenius norm: ‖A‖_F = sqrt(Σ|A[i,j]|²)
pub fn frobenius_norm(matrix: &[CudaComplex], rows: usize, cols: usize) -> f64;

/// Matrix multiply: C = A · B (col-major, complex)
pub fn matmul_complex(
    a: &[CudaComplex], a_rows: usize, a_cols: usize,
    b: &[CudaComplex], b_rows: usize, b_cols: usize,
) -> Vec<CudaComplex>;

/// Hermitian eigenvalue solver (CPU LAPACK wrapper)
pub fn hermitian_eigenvalues(matrix: &[CudaComplex], n: usize) -> Vec<f64>;

/// Check Hermiticity: max|A[i,j] - conj(A[j,i])|
pub fn check_hermiticity(matrix: &[CudaComplex], n: usize) -> f64;
```

**Dependencies**: Use `ndarray-linalg` or `lapack` crate for CPU eigenvalue solver.

## Test Implementation

**File**: `tests/rayleigh_ritz_validation.rs` (new)

**Structure**:
```rust
mod fixtures;
mod test_utils;

use fixtures::cu111_co;
use test_utils::*;

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn test_1_h_sub_hermiticity() {
    // Load fixture, run rayleigh_ritz_with_matrices, check Hermiticity
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn test_2_s_sub_hermiticity_and_positive_definiteness() {
    // Load fixture, run rayleigh_ritz_with_matrices, check Hermiticity + eigenvalues > 0
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn test_3_generalized_eigenvalue_residual() {
    // Load fixture, run rayleigh_ritz_with_matrices, compute ‖H·X - S·X·Λ‖_F
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn test_4_orthonormality() {
    // Load fixture, run rayleigh_ritz_with_matrices, compute X†·S·X, check ≈ I
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn test_5_all_band_eigenvalue_validation() {
    // Load fixture, run rayleigh_ritz, compare all 160 eigenvalues vs CASTEP .bands
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn test_6_wavefunction_normalization() {
    // Load fixture, run rayleigh_ritz, compute ⟨ψ_b|S|ψ_b⟩ for all bands
}
```

**Test ordering**: Run in sequence (1 → 2 → 3 → 4 → 5 → 6). If Layer 1 fails, skip Layers 2-3 to avoid misleading failures.

## Verification

**Run all tests**:
```bash
cargo test --release -p chemrust-scf --test rayleigh_ritz_validation --ignored --nocapture
```

**Expected outcomes**:
- **All pass**: RR implementation is mathematically correct
- **Layer 1 fails**: H_sub or S_sub assembly has layout/indexing bug
- **Layer 2 fails**: ZHEGVD call or eigenvector rotation has bug
- **Layer 3 fails**: Full pipeline integration issue (normalization, occupation, etc.)

**Regression gate** (existing test must still pass):
```bash
cargo test --release -p chemrust-scf --test ca_scf_convergence \
  ndeg_zero_with_castep_psi_matches_bands --ignored --nocapture
```

## Out of Scope

**Not included in this plan** (separate sessions):
- Fixing any bugs exposed by the tests
- Tests 7-8 from §12 (occupation sum, density decomposition) — addressed in `our-scf-persists-to-tingly-wall.md`
- Performance optimization of RR
- Extending validation to multi-k-point or spin-polarized cases

## Estimated Effort

- Infrastructure changes: 1.5 hours
- Test implementation: 1.5 hours
- Debugging test harness: 0.5 hours
- Documentation: 0.5 hours
- **Total**: 3-4 hours

## Success Criteria

✅ All 6 tests compile and run (may fail, but must execute)  
✅ Tests use EXTERNAL anchors (CASTEP fixture data), not DERIVED values  
✅ Discriminator thresholds are tight (2-10× margin between correct/incorrect)  
✅ Existing regression test `ndeg_zero_with_castep_psi_matches_bands` still passes  
✅ Test failures (if any) clearly identify which mathematical property is violated
