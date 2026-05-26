# Deferred Items: Phase-0 CG

**Review**: `notes/pr-reviews/phase-block-cg-migration/review.md`
**Date**: 2026-05-26

## Items deferred to Phase-1 (GPU batching)

1. **Full S|g⟩ in orthogonalization plumbing** — `apply_s` closure added in
   fix D3, but the full pipeline (applying S via Woodbury for the gradient)
   should be optimized when multiple bands are converged.

2. **Kinetic eigenvalue update in CG loop** — CASTEP `wave_kinetic_eigenvalues`
   line 11942. Not needed for TPA-only preconditioner, but critical for full
   USPP preconditioner in production.

3. **Performance optimization** — current O(n_pw × n_proj) loops per ion per
   band are acceptable for Phase-0 de-risking. Phase-1 should batch across
   bands and use BLAS gemv.

4. **Multi-band S-orthogonalization during CG** — Phase-0 orthogonalizes
   once at convergence. Phase-1 should interleave orthogonalization of
   active bands being solved simultaneously.

## Items deferred to Phase-2+

5. **Mixed precision (FP32) preconditioner** — TPA and R matrix in FP32
   for 2× throughput. Needs Gate-1 parity verification first.

6. **Adaptive locking tolerance** — adjust convergence threshold per band
   based on eigenvalue spacing (avoid near-degenerate convergence stall).

7. **Polak-Ribière CG variant** — CASTEP uses Fletcher-Reeves exclusively.
   PR may converge faster for some systems. Benchmark after Phase-1 works.

8. **Soft fallback logging** — currently line search silently returns step=0
   for cases where CASTEP would abort. Add `tracing::warn!` for production
   monitoring.
