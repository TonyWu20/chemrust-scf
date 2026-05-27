# Phase Plan: Diagnostic 3 — Outer Loop Convergence Test

**Date**: 2026-05-27  
**Branch**: `diag/iterative-chebyshev-viability`  
**Status**: Ready for implementation

---

## Context

**Diagnostic 2 completed successfully** (2026-05-27, commit `ed8534e`). Key findings:

1. **Conduction bands converge well**: Mean S⁻¹ residual ~0.026 Ha after one Chebyshev pass — prime candidates for early band-locking.
2. **Occupied bands need iteration**: Cu 3d (14 bands) + valence (67 bands) show residuals 0.13-0.21 Ha after one pass.
3. **Cu 3d mixing confirmed**: MAE ratio (cu3d/separated) = 1.44. Degenerate subspace rotation is structural.
4. **Well-separated eigenvalue MAE = 0.0138 Ha**: Slightly above 0.01 Ha threshold, may indicate filter spectral bounds need tuning.
5. **Band 0 anomaly**: Ranked 58th in residual (4.2e-2 Ha), not in top 5 as expected for a deep core state.

**Critical unknown**: Does the outer loop converge? Diagnostic 2 established the starting point (residuals after one pass). Diagnostic 3 will reveal the convergence rate.

---

## Goal

Test if residuals decrease monotonically over 5-10 outer loop iterations (Chebyshev filter → RR → residual check → repeat).

**Decision criteria**:
- **Residuals decrease monotonically, >80% bands lock within 5 iters** → Iterative Chebyshev works, proceed to Diagnostic 5 (band-locking)
- **Residuals decrease but plateau after 6-10 iters, Cu 3d cluster stalls** → Standard RR insufficient for degeneracies, proceed to Diagnostic 4 (Harmonic RR)
- **Residuals increase or oscillate** → Fundamental problem, investigate filter bounds or abandon approach

---

## Success Criteria

### SC-1: Residual Monotonicity (per-group)
For each band group (core, cu3d, val, nFermi, cond), the mean S⁻¹-weighted residual must decrease or stay flat across iterations 1-10. No group may show sustained increase (>2 consecutive iterations with mean residual growth >10%).

**Source**: PARSEC Algorithm 4 (Liou et al. 2020) — subspace iteration with Chebyshev filtering converges monotonically for well-conditioned systems.

**Verification**: Plot mean residual per group vs. iteration. Assert no sustained upward trend.

### SC-2: Conduction Band Early Convergence
At least 50 of the 63 conduction bands (79%) must reach S⁻¹ residual < 0.01 Ha within 5 outer iterations.

**Source**: Diagnostic 2 baseline — conduction bands start at mean 0.026 Ha, well-separated eigenvalues converge at rate ~(λ_k/λ_{k+1})^m per iteration (Zhou 2014).

**Verification**: Count bands with residual < 0.01 Ha at iteration 5. Assert count ≥ 50.

### SC-3: Occupied Band Residual Reduction
The mean S⁻¹ residual for occupied bands (core + cu3d + val, 82 bands total) must decrease by at least 5× from iteration 1 to iteration 10.

**Source**: Diagnostic 2 baseline — occupied bands start at mean ~0.13-0.15 Ha. A 5× reduction → ~0.026-0.03 Ha is the conduction band baseline, indicating the outer loop is doing useful work.

**Verification**: Compute mean residual for bands 0-81 at iter-1 and iter-10. Assert ratio ≥ 5.0.

### SC-4: Eigenvalue Stability
The maximum eigenvalue drift per iteration (|λ_i^(k+1) - λ_i^k|) must be < 0.1 Ha for all bands after iteration 3.

**Source**: Diagnostic 2 eigenvalue MAE = 0.0138 Ha vs CASTEP. Large drift (>0.1 Ha) indicates ZHEGVD rotation instability (the cascade failure mode from HANDOFF.md).

**Verification**: Track max eigenvalue change per iteration. Assert max_drift < 0.1 Ha for iters 4-10.

### SC-5: No Cascade Signature
Band 0 eigenvalue must stay within [-1.1, -1.0] Ha across all 10 iterations.

**Source**: HANDOFF.md cascade signature — band 0 drifted from -1.055 Ha to -11.94 Ha at iter-3 in the single-pass implementation. Staying within ±0.05 Ha of the CASTEP reference (-1.055 Ha) indicates no cascade.

**Verification**: Assert -1.1 ≤ λ_0 ≤ -1.0 for all iterations.

---

## Out of Scope

- **Band-locking**: All bands filtered every iteration (no early exit for converged bands). Diagnostic 5 will test locking.
- **Harmonic Rayleigh-Ritz**: Use standard RR for all bands. Diagnostic 4 will test Harmonic RR for degenerate clusters.
- **Cholesky QR**: Use existing Gram-Schmidt orthonormalization. Cholesky QR is a Phase 1 optimization.
- **Filter bounds tuning**: Use existing spectral bounds (b_low from max_veff, b_up from Lanczos). Tuning is deferred to Phase 1.

---

## Fixtures

**Primary fixture**: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
- `Cu111_CO.castep_bin` — CASTEP-converged wavefunctions (S-orthonormal under USPP S)
- `Cu111_CO.den_fmt` — Converged density
- `Cu111_CO.pot_fmt` — Converged V_eff
- `Cu111_CO.bands` — Reference eigenvalues (160 bands)

**Test starting point**: CASTEP-converged state (same as Diagnostic 2). This isolates the outer loop behavior from SCF density-update effects.

---

## Implementation Approach

### Test Structure

**File**: `tests/chebyshev_orthogonality_diagnostic.rs`  
**Test function**: `diagnostic_3_outer_loop_convergence()`  
**Attributes**: `#[test]`, `#[ignore = "requires GPU and CASTEP fixture data"]`

### Outer Loop Pseudocode

```rust
let n_outer_iters = 10;
let mut residual_history = Vec::new();
let mut eigenvalue_history = Vec::new();

for iter in 0..n_outer_iters {
    // 1. Run Chebyshev filter (all bands, no locking)
    let (psi_filtered, hpsi_filtered, kernels) = 
        chebyshev_filter_for_test_gpu(&state, FilterMode::SinvHKeepHEig);
    
    // 2. Run standard Rayleigh-Ritz
    let (eigenvalues, eigenvectors) = 
        rayleigh_ritz_with_matrices(&psi_filtered, &hpsi_filtered, ...);
    
    // 3. Compute per-band S⁻¹-weighted residuals
    let residuals = compute_residual_norms_for_test(&state, &eigenvectors, &eigenvalues);
    
    // 4. Track history
    residual_history.push(residuals.clone());
    eigenvalue_history.push(eigenvalues.clone());
    
    // 5. Update state.psi_dev for next iteration
    state.psi_dev = eigenvectors;
    
    // Early exit if all converged (optional, for timing data)
    if residuals.iter().all(|&r| r < 0.001) {
        break;
    }
}
```

### Output Format

**Per-iteration table** (printed to stdout):
```
iter |  core_mean |  cu3d_mean |   val_mean | nFermi_mean |  cond_mean | n_locked(<0.01)
-----+------------+------------+------------+-------------+------------+----------------
   1 |   4.21e-2  |   1.55e-1  |   1.32e-1  |    7.88e-2  |   2.60e-2  |      63
   2 |   ...      |   ...      |   ...      |    ...      |   ...      |      ...
  10 |   ...      |   ...      |   ...      |    ...      |   ...      |      ...
```

**Per-group convergence plot** (data for external plotting):
```
Group: Conduction (n=63)
  iter-1: mean=2.60e-2  max=5.70e-2  n_locked=50
  iter-2: mean=...      max=...      n_locked=...
  ...
```

**Eigenvalue drift table**:
```
iter | max_drift (Ha) | band_id | λ_prev (Ha) | λ_curr (Ha)
-----+----------------+---------+-------------+------------
   2 |     1.23e-2    |    42   |   -0.2435   |   -0.2312
   3 |     ...        |   ...   |    ...      |    ...
```

---

## Verification Commands

```bash
# Run Diagnostic 3
cargo test --release --test chebyshev_orthogonality_diagnostic \
  diagnostic_3_outer_loop_convergence -- --ignored --nocapture

# Expected runtime: ~15-20 minutes (10 iterations × ~90s per iteration)
```

---

## Dependencies

**Existing infrastructure** (from Diagnostic 2):
- `chebyshev_filter_for_test_gpu()` — GPU-resident filter returning `(psi_row_gpu, hpsi_row_gpu, kernels)`
- `compute_residual_norms_for_test()` — All-GPU residual computation (gemm rotation, apply_s_times, per-band zcopy+zaxpy, batch S⁻¹ via Woodbury, zdotc norms)
- `rayleigh_ritz_with_matrices()` — Standard RR (ZHEGVD on H_sub, S_sub)

**New helper** (to add):
- `update_psi_dev_for_next_iteration()` — Copy rotated eigenvectors back to `state.psi_dev` for next filter pass

---

## Risk Assessment

### Risk 1: Residuals Plateau After 3-5 Iterations
**Likelihood**: Medium  
**Impact**: High (invalidates iterative Chebyshev hypothesis)  
**Mitigation**: If plateau occurs, check if Cu 3d cluster is the bottleneck. If yes, proceed to Diagnostic 4 (Harmonic RR). If all bands plateau, investigate filter bounds or pivot to band-by-band CG.

### Risk 2: ZHEGVD Rotation Causes Cascade
**Likelihood**: Low (Diagnostic 1b showed κ₂=1.0 for SinvHKeepHEig)  
**Impact**: Critical (same failure mode as single-pass implementation)  
**Mitigation**: SC-5 (band 0 eigenvalue stability) will detect this early. If cascade occurs, the problem is ZHEGVD rotation in degenerate clusters → proceed to Diagnostic 4 (Harmonic RR).

### Risk 3: Memory Exhaustion (10 Iterations × 4 MB per ψ)
**Likelihood**: Low  
**Impact**: Medium (test fails, no data)  
**Mitigation**: Reuse GPU buffers across iterations (don't accumulate subspace). Current implementation already does this.

---

## Next Steps After Diagnostic 3

**If SC-1 to SC-5 all pass**:
- Skip Diagnostic 4 (Harmonic RR not needed if standard RR converges)
- Proceed to Diagnostic 5 (band-locking behavior)
- Estimated timeline: 1 day for Diagnostic 5, then Phase 1 implementation (1-2 weeks)

**If SC-2 fails (conduction bands don't converge quickly)**:
- Investigate filter spectral bounds (b_low may be too tight, band 0 anomaly suggests this)
- Rerun Diagnostic 3 with adjusted bounds
- Estimated timeline: +1 day for bounds tuning

**If SC-3 fails (occupied bands don't improve)**:
- Proceed to Diagnostic 4 (Harmonic RR for Cu 3d cluster)
- Estimated timeline: +2 days for Diagnostic 4, then reassess

**If SC-4 or SC-5 fails (eigenvalue instability or cascade)**:
- Root cause is ZHEGVD rotation in degenerate clusters
- Proceed to Diagnostic 4 (Harmonic RR) as the fix
- Estimated timeline: +2 days for Diagnostic 4, then Phase 1 with Harmonic RR

---

## References

- **HANDOFF.md** — Diagnostic 2 results, cascade failure mode
- **ANALYSIS.md** — Decision matrix for diagnostic outcomes
- **PARSEC paper** (Liou et al. 2020) — Algorithm 4 (CheFSI with outer loop)
- **Zhou 2014** (JCP) — Convergence rate theory for Chebyshev filtering
