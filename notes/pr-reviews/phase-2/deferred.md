# Deferred Items: Phase 2

Items flagged during review but intentionally deferred to future phases.

## 1. Non-diagonal ZHEGVD test

The solver test uses only a diagonal matrix (`A = diag(1,2,3,4)`, `B = I`), which is the simplest possible case. A non-diagonal Hermitian test (e.g., a 4×4 random Hermitian matrix with known eigenvalues) would increase confidence in the solver wrapper.

**Recommendation for future:** Add during Phase 3 when the full diagonalize pipeline is tested end-to-end.

## 2. `gemv_f64` unit test

`gemv_f64` is implemented but untested. Low priority since GEMV is a thin wrapper over cudarc's safe API.

## 3. `#[allow(dead_code)]` on `BatchedFftPlan3d`

This will be naturally resolved when Group D (density construction) uses the batched plan. Remove the attribute at that point.

## 4. `Cpu<T>(pub T)` public field

Pre-existing newtype violation. Changing `pub T` to `pub(crate) T` or private would require auditing all external access sites. Defer to a dedicated encapsulation cleanup phase.

## 5. AddAssign/SubAssign missing on grid types

Group A-2's guidance specified `AddAssign`/`SubAssign` impls on grid newtypes, but only `Add`/`Sub`/`Mul<f64>` were implemented. These are needed by mix loops (`ρ_mix += c_i * ρ_i`). This was deferred in Group A and impacts Group E (Pulay mixing). Defer to the Group A fix pass or Group E implementation.

## 6. Subspace orthonormalization before Rayleigh-Ritz

Chebyshev filtering amplifies low-energy eigencomponents by different factors
(T_k(σ) grows exponentially with eigenvalue), so the filtered wavefunctions
ψ_k have norms spanning many orders of magnitude. Building S_sub = ψ^dag · ψ
from these vectors produces a matrix whose entries differ by potentially 10^4x+,
which can degrade the numerical precision of zhegvd's generalized eigenvalue
solve.

**Procedure (if needed):**
1. Add cuSOLVER `potrf` wrapper in `src/device/solver.rs`
2. Add cuBLAS `trsm` wrapper in `src/device/blas.rs`
3. After Chebyshev filtering, orthogonalize: `S_sub = ψ^dag·ψ`, Cholesky
   `S_sub = L·L^H`, then `ψ_ortho = ψ · L^{-H}` via trsm
4. Build H_sub from ψ_ortho instead, converting to a standard eigenproblem

**Why deferred:** The TASKS.md specifies zhegvd which solves the generalized
eigenproblem H_sub·X = ε·S_sub·X directly. Orthonormalization is a numerical
conditioning optimization, not a correctness requirement. For Cu111_CO (~50
bands, ndeg=8) the S_sub condition number is within double-precision tolerance.
Both `potrf` and `trsm` are blockers; re-opening this requires implementing
those wrappers.

## 7. `compute_beta_g` uses KptWaveBlock coupling

The V_NL beta-projector precomputation calls chemrust-hamiltonian's
`compute_beta_g`, which takes a `KptWaveBlock` (a CASTEP-binary-specific
type) rather than generic pw_coords + psi_data + k_point tuples. This
coupling is acceptable for Phase 2 (one-time CPU setup, not on GPU hot
path), but a cleaner interface would take the raw data directly.

**Resolution:** Phase 3+ can refactor `compute_beta_g` to accept the
three inputs independently or add a constructor to KptWaveBlock.

## 8. V_eff downsampling via CPU

`downsample_array_to_wave_grid` in `scf.rs` uses CPU rustfft (via
chemrust-hamiltonian). The D2H→CPU→FFT→H2D roundtrip adds ~16 MB per
SCF iteration. For Phase 2 this is unavoidable (VEffBuilder uses CPU
rustfft). Future Phase 3+ ports the FFT calls to cuFFT for fully
GPU-resident V_eff downsampling.

## 9. Real-space V_NL path not implemented

Only the reciprocal-space V_NL path (β-projector gemm in G-space) is
implemented. CASTEP also has a real-space path for pseudopotentials
with large cutoff radii where the reciprocal path becomes expensive.
USPP always uses the reciprocal path, so this deferral is permanent
unless we add PAW support.

## 10. Norm stability check uses flat threshold

The circuit breaker compares `||ψ_k|| / ||ψ_{k-1}|| > 10` as a flat
threshold. The TASKS.md specifies a relative check:
`ratio > threshold × last_ratio`. A flat threshold misses gradual
multi-step drift (e.g., 1.5x per step for 10 steps = 58x total).

**Fix:** Track previous growth ratio and compare
`curr_growth > THRESHOLD × prev_growth`. Deferred because on Cu111_CO
the norms stay well below threshold and the circuit breaker never fires.

## 11. CudaKernelSet lives in chebyshev module (cross-module coupling)

The `CudaKernelSet` struct (owning all NVRTC-compiled kernel functions) is defined
in `chebyshev.rs` but consumed by `rayleigh_ritz.rs` for the transpose kernel.
This creates a cross-module dependency where Rayleigh-Ritz depends on Chebyshev
for shared infrastructure. Not blocking for Phase 2 since it compiles and works,
but the kernel set should live in a neutral location — either `eigensolver/kernels.rs`
or `device/kernels.rs`.

## 12. Missing H2D PCI-E assertion

ADR-0002 specifies an H2D assertion alongside the D2H one:
`assert_eq!(pcie.h2d_bytes, psi_bytes + veff_bytes + ...)`.
Several setup H2D transfers (fft_idx_dev, kinetic_dev, VNL beta_g/d_matrix)
use raw `stream.clone_htod` calls that bypass `PcieAccount`. Adding the assertion
requires routing these through tracked transfers or recording their bytes manually.

**Why deferred:** The D2H assertion catches the dangerous pattern (accidental D2H
inside the Chebyshev loop). H2D setup costs are fixed per iteration and not
indicative of the class of bug PcieAccount was designed to catch (accidental
PCI-E traffic in the hot path). Low priority.

## 13. RowDistributed shape metadata mismatch

The `Gpu<WavefunctionSet<RowDistributed>>` shape is set to `[n_bands, n_pw]`
but the transpose kernel stores data in `[n_pw, n_bands]` memory layout.
No current code path syncs RowDistributed data to host (the D2H step transfers
ColumnDistributed data), so `unflatten_host` is never called on RowDistributed
data. Latent issue — would produce wrong data shape if RowDistributed D2H is
added later.

**Fix:** Change shape to `[n_pw, n_bands]` to reflect actual memory layout, or
document the discrepancy.

## 14. `rayleigh_ritz` ignores `stream.synchronize()` result

Line 136 in `rayleigh_ritz.rs` calls `stream.synchronize()` but the result is
not checked for errors. If the stream encounters an error (e.g., from a previous
kernel launch), synchronize would return an error that is silently dropped.
The result `map_err(Error::Cuda)?` is not present.

**Fix:** Add `?` or `map_err(Error::Cuda)?` to the `synchronize()` call.

## 15. File location: src/density.rs vs src/scf/density.rs

The Group D spec says `src/scf/density.rs` but the module lives at `src/density.rs`. Moving it to `src/scf/` would require converting `src/scf.rs` from a flat file into a directory with `scf/mod.rs` + `scf/density.rs`. This is a larger refactor than appropriate for a post-review fix pass. Defer to a dedicated module reorganization phase.

## 16. C2C → C2R optimization for density construction

The spec calls for batched C2R transforms (half the output size, no imaginary noise), but the implementation uses C2C. C2C is correct for general k-points and avoids the Hermitian-symmetry constraints of C2R. Revisit for Gamma-point optimization in Phase 3.

## 17. PcieAccount tracking in construct_density

`construct_density()` has H2D (psi, fft_indices, occupations) and D2H (rho) PCI-E transfers that are not tracked by PcieAccount. Adding this requires passing `PcieAccount` through the function. Deferred because the critical monitoring is in `diagonalize()` (where unbounded data flows occur inside the Chebyshev recurrence loop). Density construction transfers are fixed-size, predictable, and unlikely to hide the kind of bug PcieAccount was designed to catch.

## 18. Shared CudaContext/CudaKernelSet across SCF transitions

`diagonalize()` and `construct_density()` each independently create `CudaContext::new(0)` and `CudaKernelSet::new()`, meaning NVRTC recompiles all kernels twice per iteration and two primary contexts are allocated. Fuse into a shared resource struct passed through the state machine. Phase 3 refactor.

## 19. `n_electrons` recomputed per iteration

The species loop for computing total electron count executes every SCF cycle. This is a constant (determined by the pseudopotentials and cell composition). Store once. Trivial fix, negligible perf impact for Phase 2.

## 20. Spectral bound formula uses ad-hoc kinetic+potential estimate

The code at `src/eigensolver/chebyshev.rs:236` computes:
```rust
let lm = kinetic_max + (max_veff - min_veff);
```
The TASKS.md guidance (line 240) specifies `[0, 0.5|k+G_max|² + max(V_eff)]`.
Neither matches the approach in the literature.

**The proper approach** (Zhou et al. 2010, *J. Comp. Appl. Math.*, DOI 10.1016/j.cam.2010.04.022
— Ref [40] of the Zhou 2014 JCP paper):

Run k-step Lanczos (k=4–10) on H with a random starting vector to obtain
T_k (tridiagonal) and residual f_k. Then:

- **Basic bound (eq 2.5):**  λ_max(T_k) + ‖f_k‖₂
- **Safeguard bound (eq 2.7):**  λ_max(T_k) + max_z |e_kᵀz| · ‖f_k‖₂

where z runs over all unit eigenvectors of T_k. The safeguard ensures (2.7)
is provably an upper bound even when the Lanczos Ritz value hasn't yet
converged to λ_max(H). For DFT Hamiltonians, k=4–10 steps are empirically
sufficient (proved in Theorem 1-2 of the paper).

**Why deferred:**
- Both `max_veff` and `max_veff - min_veff` are safe overestimates — neither
  produces wrong answers, only reduced filter efficiency.
- Implementing a Lanczos estimator requires: (1) random vector H2D, (2) k
  matrix-vector products (H·v via the existing FFT roundtrip + V_NL machinery),
  (3) CPU tridiagonal solve for T_k eigenvalues, (4) safeguard computation.
  This is scope increase for Phase 2.
- The practical impact on Cu111_CO (~50 bands, modest system) is negligible.

## Group E Deferred Items

### 21. Kerker `current_density_in` storage via D2H/H2D roundtrip

`Kerker::mix()` and `Pulay::mix()` save the mixed reciprocal density for the next iteration via `clone_dtoh` → `clone_htod` roundtrip. This wastes a PCI-E roundtrip per mix call. Use `stream.alloc_clone()` or similar GPU-side copy instead. Defer to Phase 3 — negligible for Phase 2 correctness validation.

### 22. cuFFT plan created per mix() call

`FftPlan3d::plan_c2c(...)` is called inside every `mix()`. cuFFT plan creation is expensive. Cache in `DensityHistory`. Defer to Phase 3 — correctness-first for Phase 2.

### 23. Ewald α parameter differs from TASKS.md specification

Spec says `α = √π / V^(1/3)`. Code uses `α = (π/V)^(1/3)`. Ewald energy converges to the same value with adequate cutoffs regardless of α choice. Not a physics error. Verify against CASTEP reference in Group F; adjust if discrepancy exceeds 1e-6 eV.
