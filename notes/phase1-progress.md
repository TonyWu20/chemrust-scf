# Phase 1 Progress: Diagnostic 1 Implementation

## Summary

Successfully created the infrastructure for Diagnostic 1 (Orthogonality After Chebyshev Filtering) and fixed blocking compilation errors in the main library.

## Completed Work

### 1. Fixed Library Compilation Errors ✓

Fixed pre-existing compilation errors in `src/scf.rs` and `src/scf_capture.rs` caused by API changes in `chemrust-hamiltonian-core`:

**Changes made:**
- `src/scf.rs:1841`: Fixed `parameters_raw` type mismatch (now `Vec<Vec<u8>>` instead of `Option<Vec<u8>>`)
- `src/scf.rs:1848`: Fixed `CheckFile::write()` argument order (now `write(writer, data)` instead of `write(data, writer)`)
- `src/scf_capture.rs:170-186`: Added missing `CastepBin` fields:
  - `cell_raw: vec![]`
  - `orig_cell_raw: vec![]`
  - `kpoint_weights: vec![1.0]`

**Result**: Main library now compiles successfully.

### 2. Created Diagnostic 1 Test Infrastructure ✓

Created `tests/chebyshev_orthogonality_diagnostic.rs` with:

**Helper functions:**
- `condition_number_svd()`: Computes κ₂(M) via SVD using `faer` library
- `compute_s_overlap_matrix()`: Computes S-overlap matrix M_ij = ⟨ψ_i|S|ψ_j⟩
- `off_diagonal_stats()`: Computes max, median, mean of off-diagonal elements

**Unit tests:**
- `test_condition_number_identity()`: Verifies κ₂ = 1 for identity matrix
- `test_condition_number_ill_conditioned()`: Verifies large κ₂ for ill-conditioned matrix
- `test_off_diagonal_stats()`: Verifies off-diagonal statistics computation

**Main diagnostic test:**
- `diagnostic_1_orthogonality_after_chebyshev_filter()`: Placeholder structure for the full diagnostic

**Dependencies added:**
- `faer = { version = "0.24", features = ["std"] }` in `[dev-dependencies]`

**Result**: Test compiles successfully with 18 warnings (mostly unused imports).

### 3. Documentation ✓

Created `notes/diagnostic-1-status.md` documenting:
- Current implementation status
- Blockers encountered and resolved
- Next steps for full diagnostic implementation

## Current Status

**Compilation**: ✅ All tests compile  
**Unit tests**: ✅ Helper functions have unit tests  
**Main diagnostic**: ⚠️ Placeholder only (needs GPU eigensolver integration)

## Next Steps

### Immediate: Implement Full Diagnostic

The main diagnostic test currently has a placeholder. To complete it, we need to:

1. **Extract `chebyshev_filter()` as standalone function**
   - Currently embedded in `ScfIteration::diagonalize()`
   - Need to expose it in `src/eigensolver/chebyshev.rs`
   - Add GPU upload/download helpers

2. **Implement S-overlap matrix computation**
   - Current `compute_s_overlap_matrix()` assumes S=I (NCPP)
   - Need to call `apply_s_times()` on GPU for USPP
   - Download S·ψ to CPU and compute M_ij = ⟨ψ_i | S·ψ_j⟩

3. **Add Gram-Schmidt vs Cholesky QR comparison**
   - Path A: Use existing Gram-Schmidt (lines 950-1058 in `chebyshev.rs`)
   - Path B: Implement Cholesky QR via cuSOLVER
   - Compare κ₂, max off-diagonal, and performance

4. **Run diagnostic and interpret results**
   - Load Cu111_CO fixture
   - Run Chebyshev filter (ndeg=8)
   - Apply both orthogonalization methods
   - Compute κ₂ for each
   - Report discriminator: κ₂ < 10³ (excellent), κ₂ ~ 10⁶ (acceptable), κ₂ > 10¹⁰ (broken)

### After Diagnostic 1

5. **Diagnostic 2**: Per-band Rayleigh quotient residuals
6. **Diagnostic 3**: Outer iteration convergence rate
7. **Diagnostic 4**: Performance profiling

## Discriminator Thresholds (from Plan)

| κ₂ Range | Interpretation | Action |
|----------|----------------|--------|
| < 10³ | Orthogonality excellent | Either method works, proceed with iterative Chebyshev |
| ~ 10⁶ | Orthogonality acceptable | Cholesky QR may be faster, proceed with iterative Chebyshev |
| > 10¹⁰ | Orthogonality broken | Abandon iterative Chebyshev, use band-by-band CG |

## Files Modified

- `src/scf.rs` (lines 1841, 1848)
- `src/scf_capture.rs` (lines 170-186)
- `Cargo.toml` (added `faer` dev-dependency)
- `tests/chebyshev_orthogonality_diagnostic.rs` (new file, 200+ lines)
- `notes/diagnostic-1-status.md` (new file)

## Compilation Output

```
Finished `release` profile [optimized] target(s) in 8.89s
Executable tests/chebyshev_orthogonality_diagnostic.rs
```

✅ **Ready to proceed with full diagnostic implementation**
