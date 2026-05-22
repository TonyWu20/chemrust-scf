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

**Hypothesis B (numerical):** M is genuinely near-singular because Q⁻¹ has zero
rows for projector channels without Q_aug entries (common for higher-l channels
in USPP), and the remaining B^H·B contribution may be near-singular for nearly
degenerate projector pairs.

*Verdict: Accepted.* `info = 2` means the 2×2 leading principal minor (typically
two m-subshells of the same l-channel, e.g. l=2, m=-1 and m=0) has determinant
near zero. The per-ion Gauss-Jordan handled this by skipping zero pivots
(effectively pseudo-inverse), but Cholesky requires strict positive-definiteness.

### Proposed fixes considered

1. **Increase `CHOL_REG`** from `1e-8` to `1e-4`: masks the symptom but
   introduces O(ε) error in S⁻¹·ψ. For Cu(111)+CO the error is negligible
   (B^H·B gives full-rank M, so ε·I is a small perturbation), but for a
   genuinely rank-deficient channel the error could be O(1) (projector norm² × 1/ε
   amplification in the nullspace). Tuning knob that needs re-verification for
   each new system.

2. **LU with partial pivoting** (`zgetrf`/`zgetrs`): does not require positive
   definiteness. Partial pivoting handles near-singular submatrices naturally.
   No tuning parameters. Same `_bufferSize`/workspace/DnHandle pattern as
   existing cuSOLVER wrappers. ~2× slower than Cholesky for nte ≲ 100
   (negligible in the SCF hot path).

### Decision

**Adopt LU.** Rationale:

- Robust for all systems without tuning (no `CHOL_REG` to calibrate).
- Future-proof: a system with genuinely rank-deficient projectors (e.g. O 2p with
  near-zero Q_aug for certain angular channels) would hit the same `info = k`
  failure with Cholesky, requiring another debug cycle.
- LU bindings (`zgetrf`/`zgetrs`) are available in cudarc 0.19.7 sys module
  with identical calling convention to the existing `zpotrf`/`zpotrs`.
- A vestigial `1e-12` diagonal regularisation is kept as a floor for exact
  rank deficiency (safe: 1e-12 below typical O(1) diagonal from B^H·B is
  negligible, but prevents pivot-exact-zero from zgetrf for a fully zero row).

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
