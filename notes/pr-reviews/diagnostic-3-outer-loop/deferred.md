# Deferred Items: Diagnostic 3 — Outer Loop Convergence Test

**Plan slug**: diagnostic-3-outer-loop  
**Date**: 2026-05-27

## Strategic Improvements

### DEF-1: Conduction band residual growth needs investigation

The outer-loop iterations cause conduction band residuals to grow monotonically (2.60e-2 → 2.28e-1 Ha over 10 iterations). This is a physics signal worth investigating but is outside the scope of this diagnostic implementation.

**Possible causes**:
- Fixed spectral bounds (b_up=20.84, b_low=0.0894) amplify high-frequency components in the conduction subspace
- Rayleigh-Ritz rotation mixes conduction band content across iterations
- The Gram-Schmidt re-orthogonalization after each filter pass may redistribute error

**Recommendation**: Add a post-diagnostic investigation task to understand conduction band behavior. This may inform band-dependent filtering or subspace management strategies.

### DEF-2: `chebyshev_filter_iteration_gpu` 17-parameter signature

`#[allow(clippy::too_many_arguments)]` is acceptable for a doc-hidden diagnostic wrapper. If this function is ever promoted to public API, it should take a config struct. Worth revisiting when the eigensolver API stabilizes.

### DEF-3: Accumulated `#[doc(hidden)] pub fn` wrappers

The pattern of doc-hidden pub wrappers for integration tests has accumulated 6 functions. This is a pre-existing architectural tension from making `eigensolver` module `pub(crate)`. Should be revisited when the eigensolver API stabilizes — could use a `#[cfg(test)]` test-helper module instead of re-exports.

### DEF-4: `gpu_available()` uses `catch_unwind` on CUDA FFI

Line 31 of the test file uses `catch_unwind` to detect GPU availability. If the CUDA driver aborts rather than unwinding, this works in practice but is technically UB. Could be replaced with a proper CUDA availability probe, but the current approach has been reliable across diagnostics.
