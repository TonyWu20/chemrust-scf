# Handoff: Diagnostic 2 Complete — Per-Band Residual Baseline Established

**Date**: 2026-05-27  
**Branch**: `diag/iterative-chebyshev-viability`  
**Status**: ✅ Diagnostic 2 validated, ready for Diagnostic 3

---

## What Was Accomplished This Session

### Diagnostic 2: Per-Band Residual Norms After Chebyshev Filter + RR

**Implementation**:
- `chebyshev_filter_for_test_gpu()` — GPU-resident filter returning `(psi_row_gpu, hpsi_row_gpu, kernels)` for RR chaining
- `compute_residual_norms_for_test()` — all-GPU residual computation following `davidson.rs:270-338`: gemm rotation, apply_s_times, per-band zcopy+zaxpy, batch S^{-1} via Woodbury, zdotc norms
- `diagnostic_2` integration test — 5 anchored assertions, per-band table, per-group statistics

**Result**: ✅ **Test passed (93.31s). All assertions green.**

| Group | Count | S⁻¹ Max | S⁻¹ Mean | L2 Max | L2 Mean |
|-------|-------|---------|---------|--------|---------|
| DeepCore (band 0) | 1 | 4.2e-2 | 4.2e-2 | 5.7e-2 | 5.7e-2 |
| **Cu 3d** (bands 1-14) | 14 | **2.09e-1** | **1.55e-1** | **5.02e-1** | **3.37e-1** |
| Valence | 67 | 1.68e-1 | 1.32e-1 | 3.59e-1 | 2.70e-1 |
| NearFermi | 15 | 1.05e-1 | 7.88e-2 | 2.00e-1 | 1.42e-1 |
| Conduction | 63 | 5.70e-2 | **2.60e-2** | 8.62e-2 | **4.31e-2** |

### Key Findings

1. **Conduction bands converge well**: S⁻¹ residuals ~0.026 Ha mean after one pass — these would lock early with band-locking (Diagnostic 5).

2. **Occupied bands (Cu 3d + valence, 81 bands) need more work**: Residuals ~0.13-0.21 Ha after one pass. The outer loop (Diagnostic 3-5) has a substantial workload — this is not "one sweep is enough."

3. **Cu 3d RR mixing confirmed**: MAE ratio (cu3d/separated) = 1.44, and Cu 3d residuals are the highest of any group. Degenerate subspace rotation is structural.

4. **Well-separated eigenvalue MAE = 0.0138 Ha** (slightly above the 0.01 Ha note threshold, not a failure). Conduction bands recover well despite this.

5. **Band 0 surprise**: Ranked 58th in residual (4.2e-2 Ha), not in top 5. This may be a filter spectral bound issue — band 0 at -1.055 Ha sits near the edge of the Chebyshev passband. The filter amplifies components near the passband center, and band 0 may be too far from center.

6. **n_pw = 60067** (different from earlier 9477 — this is the full PW basis at the actual cutoff, not a test subset)

### What This Means for Diagnostics 3-5

- The outer loop has a clear signal to work with: ~0.1-0.2 Ha residuals for 81 occupied bands
- Conduction bands (~63 bands) already near convergence — band-locking will help
- Filter spectral bounds (b_low=0.0894 from max_veff) may be mis-identifying the lower bound — the filter window may not be centered optimally for the full spectral range
- Harmonic RR (Diagnostic 4) is worth testing: the Cu 3d cluster shows clear mixing

### Files Modified

- `src/eigensolver/chebyshev.rs` — Added `chebyshev_filter_for_test_gpu()` + `compute_residual_norms_for_test()` + type alias
- `src/lib.rs` — Added re-exports for the new wrappers and `rayleigh_ritz_with_matrices`
- `tests/chebyshev_orthogonality_diagnostic.rs` — Added Diagnostic 2 test + helpers

---

## What To Do Next Session

### Immediate Next Step: Diagnostic 3

**Goal**: Test if residuals decrease monotonically with a simple outer loop (5-10 iterations).

**Implementation** (in the same test file or a new one):

```rust
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_3_outer_loop_convergence() {
    // For each iteration 1..N:
    //   1. Run Chebyshev filter (all bands, no locking yet)
    //   2. Run standard Rayleigh-Ritz
    //   3. Compute per-band S⁻¹-weighted residuals
    //   4. Track per-band residual evolution
    //   5. Track eigenvalue drift
    // Report:
    //   - Residual trajectories per band cluster
    //   - Total residual sum over iterations
    //   - Which bands converge and at what rate
}
```

**Decision Point**:
- If residuals decrease → proceed to Diagnostic 4 (Harmonic RR)
- If residuals plateau → need Harmonic RR for degenerate clusters
- If residuals increase → fundamental problem with approach

**Estimated Time**: 1 day

### Subsequent Diagnostics (After Diagnostic 3)

#### Diagnostic 4: Harmonic RR vs. Standard RR

Test if Harmonic Rayleigh-Ritz stabilizes the Cu 3d degenerate cluster by using a shift σ near the cluster center.

**Estimated Time**: 1-2 days

#### Diagnostic 5: Band-Locking Behavior

Test if band-locking works without causing regression. Conduction bands are prime candidates for early locking.

**Estimated Time**: 1 day

### Decision Point After Diagnostics 2-5

**If all diagnostics pass** → Proceed to Phase 1: Full implementation of outer loop + band-locking + Harmonic RR.

**If any diagnostic fails** → Investigate root cause. May need spectrum slicing, different filter bounds, or the already-proven Davidson v1 (which has locking built in).

---

## Reference Documents

- **DIAGNOSTIC_PLAN.md** — Full diagnostic-first implementation plan
- **HANDOFF.md** (this file) — Session log and status
- **DIAGNOSTIC_1B_RESULT.md** — Detailed Diagnostic 1b test results

---

## Commands to Resume Work

```bash
cd /home/tony/programming/chemrust-scf-chebyshev-iter

# Run Diagnostic 2 (verify results)
cargo test --test chebyshev_orthogonality_diagnostic diagnostic_2_residual_norms_after_chebyshev_filter -- --ignored --nocapture

# Run Diagnostics 1-2
cargo test --test chebyshev_orthogonality_diagnostic -- --ignored --nocapture
```
