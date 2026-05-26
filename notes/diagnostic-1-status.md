# Diagnostic 1 Implementation Status

## Current State

Created `tests/chebyshev_orthogonality_diagnostic.rs` with:
- ✓ Test structure and discriminator thresholds
- ✓ Helper functions: `condition_number_svd()`, `compute_s_overlap_matrix()`, `off_diagonal_stats()`
- ✓ Unit tests for helper functions
- ✗ Main diagnostic implementation (placeholder)

## Blockers

### 1. Library Compilation Errors (Pre-existing)

The main `chemrust-scf` library has compilation errors unrelated to the diagnostic:

```
error[E0308]: mismatched types
    --> src/scf.rs:1841:65
     |
1841 | ...                   castep_bin.parameters_raw = Some(raw.clone());
     |                       -------------------------   ^^^^^^^^^^^^^^^^^ expected `Vec<Vec<u8>>`, found `Option<Vec<u8>>`

error[E0277]: the trait bound `&CastepBin: std::io::Write` is not satisfied
    --> src/scf.rs:1848:37
     |
1847 | ...                   match chemrust_hamiltonian_core::CheckFile::write(
1848 | ...                       &castep_bin, &mut file,
     |                           ^^^^^^^^^^^ the trait `std::io::Write` is not implemented for `&CastepBin`

error[E0308]: mismatched types
   --> src/scf_capture.rs:178:25
    |
178 |         parameters_raw: None,
    |                         ^^^^ expected `Vec<Vec<u8>>`, found `Option<_>`

error[E0063]: missing fields `cell_raw`, `kpoint_weights` and `orig_cell_raw` in initializer of `CastepBin`
   --> src/scf_capture.rs:170:10
```

**Root cause**: `chemrust-hamiltonian-core` API changed (CastepBin struct fields, CheckFile::write signature).

**Impact**: Cannot compile tests until library compiles.

**Note from ANALYSIS.md**:
> These only affect the `.check`-file writing codepath, not the eigensolver or filter code paths.

### 2. Missing Dependency

The test needs `ndarray-linalg` for SVD computation:

```toml
[dev-dependencies]
ndarray-linalg = { version = "0.17", features = ["openblas-static"] }
```

### 3. Eigensolver Infrastructure Access

The main diagnostic requires:
1. Uploading ψ to GPU
2. Running `chebyshev_filter()` (currently embedded in `ScfIteration::diagonalize()`)
3. Downloading filtered ψ back to CPU
4. Computing S-overlap matrix (USPP-aware via `apply_s_times`)

**Current architecture**: `chebyshev_filter()` is not exposed as a standalone function. It's called internally by `diagonalize()` in `src/scf.rs`.

**Options**:
- **A**: Extract `chebyshev_filter()` into a public API in `src/eigensolver/chebyshev.rs`
- **B**: Call `diagonalize()` and intercept the filtered wavefunctions before Rayleigh-Ritz
- **C**: Duplicate the filter logic in the test (not recommended)

## Next Steps

### Immediate (Fix Blockers)

1. **Fix library compilation errors** in `src/scf.rs` and `src/scf_capture.rs`:
   - Update `CastepBin` field initialization to match new API
   - Fix `CheckFile::write()` argument order
   - Add missing fields: `cell_raw`, `kpoint_weights`, `orig_cell_raw`

2. **Add `ndarray-linalg` to `Cargo.toml`**:
   ```toml
   [dev-dependencies]
   ndarray-linalg = { version = "0.17", features = ["openblas-static"] }
   ```

3. **Extract `chebyshev_filter()` as standalone function**:
   - Move GPU upload/download logic out of `diagonalize()`
   - Expose `chebyshev_filter()` in `src/eigensolver/chebyshev.rs`
   - Add helper to compute S-overlap matrix on GPU

### After Blockers Resolved

4. **Implement main diagnostic**:
   - Load CASTEP fixture wavefunctions
   - Upload to GPU
   - Run Chebyshev filter (ndeg=8)
   - Path A: Apply Gram-Schmidt (existing)
   - Path B: Apply Cholesky QR (new, via cuSOLVER)
   - Download filtered ψ to CPU
   - Compute S-overlap matrix M
   - Compute κ₂(M) via SVD
   - Report statistics

5. **Run diagnostic and interpret results**:
   - κ₂ < 10³: Proceed with either method
   - κ₂ ~ 10⁶: Cholesky QR may be faster
   - κ₂ > 10¹⁰: Abandon iterative Chebyshev, use CG

## Recommendation

**Fix the library compilation errors first.** These are blocking all test compilation, not just the diagnostic. The errors are in the `.check` file writing code, which is orthogonal to the eigensolver work, but must be resolved to proceed.

Once the library compiles, I can implement the full diagnostic.
