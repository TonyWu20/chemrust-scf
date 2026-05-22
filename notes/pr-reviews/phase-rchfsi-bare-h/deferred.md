# Deferred Items — Bare-H R-ChFSI review

Items from the strategic review worth doing but out of scope for the current phase.

## Cleanup

### D-1: Revert `pub(crate)` on `c2c_inverse_inplace`

**File**: `src/eigensolver/chebyshev.rs:34`
**Current**: `pub(crate) unsafe fn c2c_inverse_inplace`
**Should be**: `unsafe fn c2c_inverse_inplace` (private — only called at line 599 within the same file).
**Why**: The `pub(crate)` was necessary during Group A-C when this function was called from `scf.rs`. The call path changed; the visibility was not reverted.

### D-2: Normalize test API boundary pattern

**Files**: `src/density.rs:601` and `src/eigensolver/chebyshev.rs`
**Current**: Mixed — `density.rs` uses `test_api` module for its own test exports, but re-exports `CudaKernelSet` and `check_s_inv_s_identity` from chebyshev via `#[doc(hidden)]`.
**Recommendation**: Pick one convention. `test_api` module is the established pattern.

### D-3: Genericize `c2c_inverse_inplace` over plan type

**Files**: `src/eigensolver/chebyshev.rs:34` and `src/mixing.rs:104`
**Current**: Two versions of `c2c_inverse_inplace`, one for `BatchedFftPlan3d` and one for `FftPlan3d`. They could share a trait.
**Note**: Low priority — the duplication is small and clear.

## Contingency

### D-4: TASK-D4 activation (S⁻¹ in Step 4 only)

If SC-7 fails (D_screened iter-3 > 10 Ha) or SC-3 fails (V_eff range > 1.0 Ha),
restore `apply_s_inverse(&mut buf_a, vnl_data, ...)` in Step 4 of the Chebyshev
filter. The code is preserved at `src/eigensolver/chebyshev.rs:789` with
`#[allow(dead_code)]` and a preservation comment.

## Future investigation

### D-5: Measure h_eig vs generalized eigenvalue deviation

**Why**: Quantify the approximation error of using H-eigenvalues (⟨ψ_j, H·ψ_j⟩)
instead of generalized eigenvalues. For USPP, ‖S − I‖ is small but non-zero.
Measuring this helps decide whether the bare-H approach is viable for production.

### D-6: GPU D-matrix screening

**Why**: `D = D0 + ∫ Q·V_eff` is computed on CPU via a scalar loop over
G-vectors per projector pair. For Cu₁₁₁-CO (20 ions, ~365M MACs per SCF
iteration), this takes ~3 seconds — the current bottleneck.

**Implementation sketch**:
- Pre-upload Q per species to GPU once (struct-optimization-scoped)
- For each SCF iteration: upload V_eff_fft to GPU (~1 MB)
- For each species, compute phase factors `exp(2πi·G·r_ion)` for each ion
  of that species (simple ~20-line kernel)
- Compute D_screening as a single `gemv` per species:
  `Q_matrix (n_pairs × n_pw) · (V_fft ⊙ phase_ion) (n_pw × 1)`
  → `d_screen_flat (n_pairs × 1)`
- Read back ~2 KB per ion

**Est. speedup**: ~3 s → ~1 ms per SCF iteration.
**Est. effort**: ~half-day implementation (Q layout restructure, phase kernel,
gemv integration into `VnlBatchData::precompute`).
