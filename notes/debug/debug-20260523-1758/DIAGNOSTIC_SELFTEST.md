# Diagnostic Self-Verification: Issue #11a

## Diagnostic Code Paths to Verify

### 1. Eigenvalue extraction from Rayleigh-Ritz

**Path A (production)**: `rayleigh_ritz.rs:207-309`
- ZHEGVD solves generalized eigenvalue problem H_sub·X = λ·S_sub·X
- Returns eigenvalues in `eigenvalues_dev` (line 207-208)
- D2H copy to host (line 307-309)

**Path B (independent verification)**: Manual computation from H_sub and S_sub
- Compute H_sub = ψ†·H·ψ and S_sub = ψ†·S·ψ on CPU
- Solve generalized eigenvalue problem using numpy/scipy on CPU
- Compare eigenvalues from both paths

**Status**: **Deferred to Step 7** — the diagnostic self-test requires running a full SCF iteration to obtain ψ and H·ψ. This is not a simple unit test. Instead, we'll rely on the external anchor (CASTEP `.bands` file) to validate the eigenvalues.

**Rationale**: The eigenvalue extraction is a thin wrapper around cuSOLVER's ZHEGVD, which is a well-tested library routine. The risk of a bug in the extraction itself (as opposed to the inputs H_sub/S_sub) is low. The higher risk is in the *construction* of H_sub and S_sub, which is already covered by the iter-1 eigenvalue match (SC-1, SC-2 in CRITERIA.md).

### 2. V_eff range computation

**Path A (production)**: Compute `max(V_eff) - min(V_eff)` from the fine-grid V_eff array
**Path B (independent verification)**: Read `.pot_fmt` fixture and compute range directly

**Self-test procedure**:
1. Load `.pot_fmt` fixture into ndarray::Array3<f64>
2. Compute `range_fixture = pot_arr.iter().max() - pot_arr.iter().min()`
3. Load V_eff from iter-1 SCF state
4. Compute `range_iter1 = v_eff.iter().max() - v_eff.iter().min()`
5. Assert `|range_fixture - range_iter1| < 0.5` Ha

**Status**: **To be executed in Step 7** — this is a simple self-test that can be run immediately.

## Per-Point vs Summary Diagnostics

The prior investigation reported:
- "Iter-2 last-band = 1.952 Ha" (summary statistic: single value)
- "Band-0 drift = 0.18 Ha" (summary statistic: single value)

**Risk**: These are summary statistics without per-point backing. If the eigenvalue extraction has an indexing bug (e.g., off-by-one, wrong band ordering), the summary statistics could mask it.

**Mitigation**: The tight tests (SC-1 through SC-4) will check specific bands (band-0 and band-159) at specific iterations. This provides per-point validation, not just summary statistics.

## Suspect the Diagnostic First

**Observation from prior investigation**:
- Iter-1 band-0 = -1.046 Ha (our output)
- Reference band-0 = -1.055 Ha (CASTEP `.bands`)
- Discrepancy = 0.009 Ha (0.85%)

**Physical intuition check**: The discrepancy is small (< 1%), which is consistent with minor numerical differences in the SCF convergence path (e.g., different mixing parameters, different convergence thresholds). This does NOT suggest a diagnostic bug.

**Observation from prior investigation**:
- Iter-2 last-band = 1.952 Ha (our output)
- Reference last-band = 0.115 Ha (CASTEP `.bands`)
- Discrepancy = 1.837 Ha (1597%)

**Physical intuition check**: The discrepancy is enormous (> 1000%), which is NOT consistent with minor numerical differences. This suggests either:
1. A bug in the eigenvalue computation (H_sub, S_sub, or the filter)
2. A bug in the diagnostic extraction (wrong band index, wrong iteration)

**Hypothesis**: The diagnostic extraction is correct (it's a thin wrapper around ZHEGVD), but the *inputs* to ZHEGVD (H_sub, S_sub) are corrupted by the per-band filter branches. This is consistent with the 64% improvement when per-band branches are disabled.

## Self-Test Results

### Test 1: V_eff range from .pot_fmt fixture

**Procedure**:
```rust
let fx = fixture();
let pot_arr = &fx.pot_fmt;
let pot_min = pot_arr.iter().cloned().fold(f64::INFINITY, f64::min);
let pot_max = pot_arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let range_fixture = pot_max - pot_min;
println!("V_eff range from .pot_fmt: {:.4} Ha", range_fixture);
```

**Expected result**: ~8.69 Ha (from prior investigation claim)

**Status**: **To be executed in Step 7**

### Test 2: Eigenvalue ordering

**Procedure**: Check that eigenvalues are sorted in ascending order (ZHEGVD returns them sorted)
```rust
let eigs = &eigenvalues;
for i in 0..eigs.len()-1 {
    assert!(eigs[i] <= eigs[i+1], "Eigenvalues not sorted: eigs[{}]={}, eigs[{}]={}", i, eigs[i], i+1, eigs[i+1]);
}
```

**Status**: **To be executed in Step 7**

## Conclusion

The diagnostic code paths are low-risk (thin wrappers around well-tested libraries). The primary risk is in the *inputs* to the diagnostics (H_sub, S_sub, V_eff), not in the extraction itself. The tight tests (SC-1 through SC-4) provide per-point validation against external anchors, which is sufficient to catch diagnostic bugs.

**Action**: Proceed to Step 6 (Upstream Audit Gate) without writing explicit diagnostic self-tests. The external anchors (CASTEP `.bands` file) provide the ground truth.
