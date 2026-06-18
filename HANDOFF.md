# Handoff — 2026-06-19: Profiling removed; convergence fixes merged from reference

**Branch**: `feat/chebyshev-iterative-eigensolver`
**Status**: `profile-Hpsi` removed ✅ | `compute_all()` restored ✅ | CUDA event timing added ✅ | Cholesky QR under test ⏳

---

## What changed this session

### `profile-Hpsi` diagnostic removed (hamiltonian.rs)

All `#[cfg(feature = "scf_diag")]` profiling code stripped from `hamiltonian.rs`:
- `use std::time::Instant` import removed
- `mod profile` with `VLocProfile` struct and `thread_local!` storage removed
- Per-operation stream sync + CPU timers in `apply_v_loc_hamiltonian` removed
- `_p_h_total_begin` / `_p_vnl_begin` + `eprintln!` print block in `apply_full_hamiltonian` removed
- `BetaPhiCache HIT` eprintln in `apply_v_nl_hamiltonian` removed

This was found to be a cause of divergence — the profiling code added stream
synchronization points that altered GPU execution order.

Reference: `castep-rust-eigensolve/chemrust-scf/src/eigensolver/hamiltonian.rs` (clean version).

### Convergence fixes merged from `castep-rust-eigensolve/chemrust-scf`

Most of the reference's divergence-attack commits were already in our codebase
(from the `feat/phase-7` merge at `79e61dc`). Two missing pieces were ported:

#### Critical: `beta_phi_cache.compute_all()` restored (davidson.rs)

The SURV-01 deferred comment was replaced with an actual `compute_all()` call at
the end of each outer iteration. The reference repo's investigation (`docs/load-bearing-diagnostic-overhead.md`)
proved that the cuBLAS GEMMs queued by `compute_all()` (β^H·ψ for all bands) are
**load-bearing for GPU stream ordering** — removing them causes divergence.
Stage-level syncs alone are not sufficient.

#### Safe profiling: CUDA event-based GPU timing (davidson.rs, scf_diag only)

CUDA event timing infrastructure added at key points in `davidson_diagonalise()`
and `DavidsonBlockCtx::build()`:
- **H·psi** — start/end events around `apply_full_hamiltonian`
- **Rayleigh-ZDOTC** — start/end events around per-band `cublasZdotc`
- **A1-ZHEEVD** — start/end events around full subspace diagonalization
- **A2-ZHEEVD** — per-block events around `diagonalise_subspace`
- **build()** — start/end events for each block's build phase
- **Final report** — aggregate timing grouped by label, printed after outer loop

All behind `#[cfg(feature = "scf_diag")]`. CUDA events record timestamps on the
GPU stream without CPU blocking — NO divergence risk (unlike `Instant::now()` +
`stream.synchronize()`).

### Files already in sync (no changes needed)

| File | Status |
|------|--------|
| `kernels.rs` | `scale_cols_by_eig` kernel + field already present |
| `preconditioner.rs` | GPU-resident USPP path, Q_RCQ/R_beta GPU upload, `total_ne` fields all present |
| `davidson.rs` | GPU preconditioner plumbing, D1/D2 gating, residual norms gating, `eig_ptr` UB fix, `compute_all` comment — all already present |

---

## Current issues still under investigation

### Cu111_CO ZHEGVD crash (info=150-159) on non-cubic grid

Cholesky QR (commit `b3564ab`) replaces Gram-Schmidt in Phase 6 of `chebfi_run_rust`.
Currently under test — no results yet.

### NiO cold start energy oscillation

SCF energy oscillates at ~-7153.6 eV vs CPU reference -7160.3 eV.
Cholesky QR may improve this. Under test.

---

## Key files changed this session

| File | Changes |
|------|---------|
| `src/eigensolver/hamiltonian.rs` | All `scf_diag` profiling removed (~135 lines) |
| `src/eigensolver/davidson.rs` | `compute_all()` restored + CUDA event timing (scf_diag only) |
| `HANDOFF.md` | Updated to reflect current state |

---

## Reference fixtures

| Run | Path | What it proves |
|-----|------|---------------|
| NiO cheby tests | `/export/.../NiO_no_spin_0618_cheby/` | Chebyshev cold-start oscillation |
| Cu111_CO cheby cold | `/export/.../Cu111_CO_Single_Point_0530_rust_eigensolver/` | ZHEGVD crash on non-cubic grid |
| Cu111_CO cheby warm | `/export/.../Cu111_CO_Single_Point_0604_warm_start/` | ZHEGVD crash persists on warm start |

## Pending

- [ ] Cholesky QR test results — Cu111_CO cold/warm start
- [ ] Cholesky QR test results — NiO cold start
- [ ] Test convergence with `profile-Hpsi` removed + `compute_all()` restored
