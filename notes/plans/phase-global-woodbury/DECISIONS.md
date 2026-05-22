# DECISIONS.md — phase-global-woodbury

## D1: Global Woodbury M factorisation — LU over Cholesky

### Status

Adopted 2026-05-23.

### Problem

`zpotrf` fails with `info = 2` (2×2 leading principal minor not positive definite)
on the global Woodbury matrix M = Q⁻¹ + B^H·B for the Cu(111)+CO fixture.

### Investigation

Two competing diagnoses were evaluated:

**Hypothesis A (layout error):** B_concat memory layout is wrong. The per-ion
`beta_g` data (ndarray `Array2<Complex64>`, row-major, shape `(ne, n_pw)`) is
flattened row-major and concatenated via `extend_from_slice` into `b_concat`,
then uploaded to GPU as a single `CudaSlice<CudaComplex>` and interpreted as
column-major with `lda = n_pw`.

*Verdict: Rejected.* A concrete trace (2 ions, ne0=2, ne1=1, n_pw=3) confirms
`B[g,i] = β[i,g]` at every offset. The row-major `(ne, n_pw)` flat data,
interpreted as column-major with `lda = n_pw`, gives `B: n_pw × nte` where
column `i` is projector channel `i` evaluated at all G-vectors. This matches
the per-ion GEMM pattern in the existing `apply_v_nl_hamiltonian` which uses
the exact same `extend_from_slice` + `lda = n_pw` approach. The proposed
transposed concatenation (iterate PWs in outer loop) would require `lda = ne`
and produce `B·B^H` (nte×nte) instead of `B^H·B` (nte×nte) — wrong dimensions
unless the GEMM parameters are also changed.

**Hypothesis B (numerical near-singularity):** M is genuinely near-singular because
Q⁻¹ has zero rows for projector channels without Q_aug entries.

*Verdict: Partially accepted.* `info = 2` from zpotrf is indeed caused by near-singular
M, but switching to LU alone was insufficient. The S⁻¹·S identity test still failed
at 3.4e-6 with LU because of a second bug.

**Hypothesis C (discarded imaginary parts):** The global Woodbury assembly converted
B^H·B to `Vec<f64>` by extracting `.x`, discarding off-diagonal imaginary parts.
Within a single ion, the structure-factor phase `exp(i·(k+g)·R_I)` cancels in
`conj(β_i)·β_j`, so the per-ion code was unaffected. But cross-ion blocks in the
global M have non-zero imaginary parts from `exp(i·(k+g)·(R_I−R_J))`.

*Verdict: Accepted.* Discarding imaginary parts corrupts M, preventing the Woodbury
formula from inverting the correct S. Fix: work in `Vec<CudaComplex>` throughout,
preserving the full complex B^H·B.

### Proposed fixes considered

1. **Increase `CHOL_REG` / `M_REG`**: masks symptoms but wrong — the problem was
   structural (discarded imag parts), not a conditioning issue.

2. **LU with partial pivoting** (`zgetrf`/`zgetrs`): necessary (Cholesky fails on
   near-singular M from zero Q⁻¹ rows) but not sufficient — must be combined with
   keeping full complex B^H·B.

3. **Keep full complex B^H·B**: assemble Q⁻¹ as `Vec<CudaComplex>`, keep B^H·B as
   complex, build M in-place on `Vec<CudaComplex>`. This is the correctness fix.

### Decision

**Adopt LU + full complex B^H·B.** Rationale:

- LU handles near-singular M (zero Q⁻¹ rows, Cholesky fails with info=2).
- Full complex B^H·B preserves cross-ion phase differences (per-ion code was
  unaffected since the phase cancels within each ion).
- Vestigial `1e-12` diagonal regularisation kept as floor for exact-zero pivots.

### Scope of changes

| File | Change |
|------|--------|
| `src/device/solver.rs` | Replace `zpotrf`/`zpotrs` with `zgetrf`/`zgetrs` (and tests) |
| `src/eigensolver/vnl_data.rs` | `chol_m` → `lu_m` + `lu_ipiv`; `zpotrf` → `zgetrf`; remove `CHOL_REG` |
| `src/eigensolver/chebyshev.rs` | `zpotrs` → `zgetrs` in `apply_s_inverse` |

### References

- `solver.rs:47-95` (`zhegvd`) — template for cuSOLVER wrapper pattern
- `solver.rs:110-160` (existing `zpotrf`/`zpotrs`) — direct structural template for LU
- cudarc 0.19.7 sys: `cusolverDnZgetrf`, `cusolverDnZgetrf_bufferSize`, `cusolverDnZgetrs`
