# Resolution: beta_g GPU→CPU layout transposition

**Symptom**: FFI cold start converged to wrong state — 140 SCF iterations vs 33,
23 eV energy difference, mean eigenvalue diff 0.005 Ha. Search direction L2²
grew without bound (0.62 → 71 → 155 → 214 → ... → 230,000). High-G search
coefficients were 10,000× larger than CASTEP CPU reference.

**Root cause**: Data layout transposition at `chemrust-scf/src/eigensolver/davidson.rs:948`.
GPU stores `beta_g` flat as row-major `(ne, n_pw)`: `beta_flat[n·n_pw + G] = beta(n, G)`.
cuBLAS GEMM reads it column-major `(n_pw, ne)` with `lda=n_pw`:
`data[G + n·n_pw] = beta(n, G)` — correct because addition commutes.
But `clone_dtoh` preserves the flat layout, and the reshape
`Array2::from_shape_vec((n_pw, ne), ...)` reads row-major:
`beta[[G, n]] = data[G·ne + n]` instead of `data[n·n_pw + G]`.
Since `ne = 4 ≪ n_pw = 60067`, every element landed at the wrong G-vector.

This corrupted all downstream computations in `prepare_preconditioner()`:
- `compute_c_global` — wrong Gram matrix
- `assemble_r_beta` — wrong R_beta inverse
- `assemble_q_rcq` — wrong Q·R·C·Q matrix

The corrupted NL correction weights amplified high-G search components by 22×,
creating a feedback loop that grew L2² without bound across SCF iterations.

**Fix location**: `chemrust-scf/src/eigensolver/davidson.rs:948-963`

**Fix description**: Reshape downloaded data as `(ne, n_pw)` (restoring original
row-major layout) then transpose to `(n_pw, ne)`. This gives
`beta[[G, n]] = beta_orig[[n, G]] = beta(n, G)` matching the GPU GEMM convention
and the downstream code's expectation.

**Anchor criteria used**:
- CASTEP CPU cold start reference: final energy = −24110.96676596 eV (Source: `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0604_cold_start_dump/Cu111_CO.castep`)
- H·psi GPU vs CPU per G-vector synthesis test: max rel 1.3e-11, zero bias
- TPA kernel synthesis test: max rel 1.9e-14
- ZGEMM accumulation order test: max rel 2.1e-13
- cuFFT round-trip test: max rel 1.9e-14

**Verification**:
| Metric | Before fix | After fix | CPU reference |
|--------|-----------|-----------|---------------|
| Final energy | ~−24087 eV | −24110.96709863 eV | −24110.96676596 eV |
| Energy diff vs CPU | ~23 eV (0.85 Ha) | −0.00033 eV (−1.2×10⁻⁵ Ha) | — |
| SCF iterations | 140+ | 31 | 33 |

**Prior notes reclassified**: None — this was a new discovery by the audit workflow.

**Date**: 2026-06-10

**Commit**: `2a98448` — `fix(preconditioner): beta_g GPU→CPU layout transposition — root cause of cold start divergence`
