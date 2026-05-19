# Deferred Improvements — Phase 2 Fixes Review

These items were identified during review but are out of scope for the current fix pass. Candidates for `/define-outcomes` in Phase 3.

## Double NVRTC compilation per SCF iteration

`diagonalize()` and `construct_density()` each independently create `CudaContext::new(0)` and `CudaKernelSet::new()`, recompiling all kernels twice per iteration. On a 30-iteration SCF run with 9 kernels, this is ~60-180s of avoidable overhead. Fix: shared `GpuContext` struct passed through the state machine.

## `CudaKernelSet` location

Defined in `src/eigensolver/chebyshev.rs` but consumed by `rayleigh_ritz.rs` (transpose kernel) and `density.rs` (scatter, accumulate_density). Creates a de-facto `pub(crate)` coupling where `density` depends on `eigensolver::chebyshev` for infrastructure access. Move to `src/device/kernels.rs` or `src/eigensolver/kernels.rs`.

## PcieAccount tracking in `construct_density`

H2D (psi, fft_indices, occupations) and D2H (rho) transfers in `construct_density()` are not tracked by PcieAccount. Fixed-size and predictable transfers — low risk of masking bugs. Add for monitoring parity with `diagonalize()`.
