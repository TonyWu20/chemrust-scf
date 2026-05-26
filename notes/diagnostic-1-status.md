# Diagnostic 1 & 1b Implementation Status

## ✅ Completed (2025-05-26)

### Diagnostic 1 (Original)
- **Status**: ✅ Complete (commit c471b3d)
- **Filter Mode**: BareH (baseline, physically incorrect for USPP)
- **Result**: κ₂ = 1.0000e0, off-diagonal max = 7.6e-15
- **Verdict**: Orthogonality is EXCELLENT

### Diagnostic 1b (Critical Validation)
- **Status**: ✅ Complete (commit cc99f27)
- **Filter Mode**: SinvHKeepHEig (production mode for USPP)
- **Result**: κ₂ = 1.0000e0, off-diagonal max = 2.9e-15
- **Verdict**: Orthogonality is EXCELLENT ✓ **CRITICAL FINDING**

### Diagnostic 1c (Optional)
- **Status**: ✅ Complete (commit cc99f27)
- **Filter Mode**: SinvHFullDas (full Das Algorithm 3)
- **Result**: κ₂ = 1.0000e0, off-diagonal max = 2.0e-15
- **Verdict**: Orthogonality is EXCELLENT

---

## Key Findings

### Critical Validation Gap Closed

**Problem**: Original Diagnostic 1 used `FilterMode::BareH`, which is physically incorrect for USPP systems. The production code uses `FilterMode::SinvHKeepHEig` (S^{-1}·H via Woodbury), which was never validated for orthogonality preservation.

**Solution**: Implemented Diagnostic 1b to test all three filter modes on the same Cu111_CO fixture.

**Result**: All three filter modes preserve orthogonality perfectly (κ₂ = 1.0).

### Implications

✅ **Factor C (USPP S^{-1} complexity) is NOT a blocker**
- Woodbury-based S^{-1} at ζ=3.8e-15 is numerically stable
- No need to investigate alternative S^{-1} methods
- No need to pivot to Davidson or band-by-band CG due to filter issues

✅ **Cascade root cause confirmed**
- Missing outer loop (no band-locking, no iterative refinement)
- Single sweep per SCF step (vs CASTEP's 19-26 iterations)
- NOT the filter operator or Woodbury precision

✅ **Green light for PARSEC Algorithm 4 implementation**
- Proceed with outer loop + band-locking + Harmonic RR
- Next: Diagnostics 2-5 to validate the outer loop approach

---

## Implementation Details

### Changes Made (commit cc99f27)

1. **Modified `chebyshev_filter_for_test()`** in `src/eigensolver/chebyshev.rs`:
   - Added `filter_mode: FilterMode` parameter
   - Removed hardcoded `FilterMode::BareH`

2. **Exported `FilterMode`** from `src/lib.rs`:
   - Added to test-only exports for integration tests

3. **Refactored test** in `tests/chebyshev_orthogonality_diagnostic.rs`:
   - Created helper function `run_orthogonality_diagnostic(filter_mode, mode_name)`
   - Added three test variants:
     - `diagnostic_1_orthogonality_after_chebyshev_filter()` - BareH
     - `diagnostic_1b_orthogonality_sinvh_keep_h_eig()` - SinvHKeepHEig (production)
     - `diagnostic_1c_orthogonality_sinvh_full_das()` - SinvHFullDas

### Test Infrastructure

- ✅ Test structure and discriminator thresholds
- ✅ Helper functions: `condition_number_svd()`, `compute_s_overlap_matrix()`, `off_diagonal_stats()`
- ✅ Unit tests for helper functions
- ✅ Main diagnostic implementation with filter mode parameter
- ✅ All three filter modes validated

---

## Documentation

- **DIAGNOSTIC_1B_RESULT.md** - Detailed test results and analysis
- **notes/diagnostic-1b-result.md** - Copy in notes directory
- **notes/plans/diagnostic-1b-and-outer-loop-plan.md** - Full implementation plan
- **HANDOFF.md** - Next session instructions

---

## Next Steps

### Diagnostic 2: Per-Band Residual Norms

**Goal**: Measure per-band residual norms after a single Chebyshev filter pass to establish baseline convergence characteristics.

**Implementation**:
1. Load Cu111_CO fixture (converged state)
2. Run one Chebyshev filter pass (ndeg=8, SinvHKeepHEig mode)
3. Apply Gram-Schmidt orthonormalization
4. Run standard Rayleigh-Ritz (ZHEGVD)
5. Compute per-band residuals: r_b = H|ψ_b⟩ - λ_b·S|ψ_b⟩
6. Compute S^{-1}-weighted norms: ||r_b||_{S^{-1}} = √⟨r_b | S^{-1}·r_b⟩
7. Report statistics by band type (Cu 3d cluster vs well-separated)

**Expected Outcome**:
- Cu 3d bands (1-14): High residuals due to degeneracy
- Well-separated bands: Low residuals
- Establishes baseline for "how many bands need more work"

**Estimated Time**: 2-3 hours

---

## Test Commands

```bash
# Run all three diagnostic tests
cargo test --test chebyshev_orthogonality_diagnostic -- --ignored --nocapture

# Run specific tests
cargo test --test chebyshev_orthogonality_diagnostic diagnostic_1_orthogonality_after_chebyshev_filter -- --ignored --nocapture
cargo test --test chebyshev_orthogonality_diagnostic diagnostic_1b_orthogonality_sinvh_keep_h_eig -- --ignored --nocapture
cargo test --test chebyshev_orthogonality_diagnostic diagnostic_1c_orthogonality_sinvh_full_das -- --ignored --nocapture
```

---

## Historical Context

### Original Blockers (Now Resolved)

1. ~~Library compilation errors~~ - Fixed in earlier commits
2. ~~Missing `ndarray-linalg` dependency~~ - Added
3. ~~Eigensolver infrastructure access~~ - `chebyshev_filter_for_test()` created

All blockers have been resolved. The diagnostic infrastructure is complete and validated.
