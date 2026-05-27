# TASKS: Diagnostic 3 — Outer Loop Convergence Test

**Phase**: diagnostic-3-outer-loop  
**Date**: 2026-05-27  
**ODD Pattern**: `/home/tony/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

---

## Declared Fixtures

**Primary fixture**: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
- `Cu111_CO.check` — CASTEP-converged wavefunctions (S-orthonormal under USPP S), 160 bands, n_pw=60067
- `Cu111_CO.bands` — Reference eigenvalues (160 bands), band 0 = -1.055 Ha
- `Cu111_CO.den_fmt` — Converged density
- `Cu111_CO.pot_fmt` — Converged V_eff

**Starting state**: CASTEP-converged (same as Diagnostic 2)

**Baseline data** (from Diagnostic 2, commit `ed8534e`):
- Conduction bands (82-159): mean S⁻¹ residual = 0.026 Ha, max = 0.057 Ha
- Occupied bands (0-81): mean S⁻¹ residual = 0.15 Ha, max = 0.21 Ha
- Band 0: S⁻¹ residual = 4.21e-2 Ha
- Cu 3d cluster (1-14): mean S⁻¹ residual = 0.155 Ha, max = 0.209 Ha

---

## Task Groups

### Group A: Core Implementation (1 task)

**Dependencies**: Diagnostic 2 infrastructure (chebyshev_filter_for_test_gpu, compute_residual_norms_for_test, rayleigh_ritz_with_matrices)

---

## TASK-A1: Implement Outer Loop Convergence Test

**Kind**: lib-tdd  
**Goal**: Test if residuals decrease monotonically over 10 outer loop iterations (Chebyshev filter → RR → residual check → repeat).

### Success Criteria

**SC-1: Residual Monotonicity (per-group)**
- For each band group (core={0}, cu3d={1-14}, val={15-81}, nFermi={82-96}, cond={97-159}):
  - `residual[N] ≤ residual[N-1] × 1.05` for most iterations (allow ≤2 non-consecutive violations)
  - At least one of {iter-5, iter-10} shows `residual[N] < residual[1] × 0.8`
- **Source**: PARSEC Algorithm 4 (Liou et al. 2020), Diagnostic 2 baseline
- **Verification granularity**: Per-group mean residual per iteration
- **Counter-example**: All groups plateau (no 20% reduction by iter-10) or show sustained increase (>2 violations)
- **Test fixture scope**: Cu111_CO.check (converged state)

**SC-2: Conduction Band Early Convergence**
- At least 50/63 conduction bands reach S⁻¹ residual < 0.01 Ha within 5 iterations
- **Source**: Diagnostic 2 baseline (conduction mean 0.026 Ha), Zhou 2014 convergence rate
- **Verification granularity**: Per-band residual at iter-5
- **Counter-example**: < 40 conduction bands converge within 5 iterations
- **Test fixture scope**: Cu111_CO.check (converged state)

**SC-3: Occupied Band Residual Reduction**
- Mean S⁻¹ residual for occupied bands (0-81) decreases by ≥5× from iter-1 to iter-10
- **Source**: Diagnostic 2 baseline (occupied mean ~0.15 Ha)
- **Verification granularity**: Mean residual for bands 0-81 at iter-1 and iter-10
- **Counter-example**: < 3× reduction (mean residual > 0.05 Ha at iter-10)
- **Test fixture scope**: Cu111_CO.check (converged state)

**SC-4: Eigenvalue Stability**
- Max eigenvalue drift per iteration `max_i |λ_i[N] - λ_i[N-1]| < 0.1 Ha` for N ≥ 4
- **Source**: HANDOFF.md cascade signature (band 0 drifted ~11 Ha in single-pass)
- **Verification granularity**: Per-iteration max drift across all bands
- **Counter-example**: Any iteration N ≥ 4 shows max drift > 0.15 Ha
- **Test fixture scope**: Cu111_CO.check (converged state)

**SC-5: No Cascade Signature**
- Band 0 eigenvalue stays within [-1.10, -1.01] Ha across all 10 iterations
- **Source**: HANDOFF.md (band 0 reference = -1.055 Ha from Cu111_CO.bands)
- **Verification granularity**: Band 0 eigenvalue per iteration
- **Counter-example**: Band 0 drifts outside [-1.15, -0.95] Ha at any iteration
- **Test fixture scope**: Cu111_CO.check (converged state), Cu111_CO.bands (reference)

### Files

**Test file**: `tests/chebyshev_orthogonality_diagnostic.rs`

### Changes

Add test function `diagnostic_3_outer_loop_convergence()` with the following structure:

```rust
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_3_outer_loop_convergence() {
    // Setup (same as Diagnostic 2)
    let fx = fixture();
    let mut state = build_scf_state_for_diagnostic(&fx);
    
    // Band group definitions
    let core_bands = 0..1;
    let cu3d_bands = 1..15;
    let val_bands = 15..82;
    let nfermi_bands = 82..97;
    let cond_bands = 97..160;
    
    // Load CASTEP reference eigenvalues
    let castep_eigenvalues = load_castep_eigenvalues(&fx);
    
    // Outer loop
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
        update_psi_dev_for_next_iteration(&mut state, eigenvectors);
        
        // Print per-iteration summary
        print_iteration_summary(iter + 1, &residuals, &eigenvalues, &castep_eigenvalues);
    }
    
    // Verify SC-1: Residual Monotonicity
    verify_residual_monotonicity(&residual_history, &[
        ("core", core_bands.clone()),
        ("cu3d", cu3d_bands.clone()),
        ("val", val_bands.clone()),
        ("nFermi", nfermi_bands.clone()),
        ("cond", cond_bands.clone()),
    ]);
    
    // Verify SC-2: Conduction Band Early Convergence
    let n_converged_at_iter5 = count_converged_bands(&residual_history[4], cond_bands.clone(), 0.01);
    assert!(n_converged_at_iter5 >= 50, 
        "SC-2 failed: only {}/63 conduction bands converged at iter-5 (expected ≥50)", 
        n_converged_at_iter5);
    
    // Verify SC-3: Occupied Band Residual Reduction
    let occupied_mean_iter1 = compute_mean_residual(&residual_history[0], 0..82);
    let occupied_mean_iter10 = compute_mean_residual(&residual_history[9], 0..82);
    let reduction_factor = occupied_mean_iter1 / occupied_mean_iter10;
    assert!(reduction_factor >= 5.0,
        "SC-3 failed: occupied band reduction = {:.2}× (expected ≥5×)",
        reduction_factor);
    
    // Verify SC-4: Eigenvalue Stability
    for iter in 4..10 {
        let max_drift = compute_max_eigenvalue_drift(&eigenvalue_history[iter-1], &eigenvalue_history[iter]);
        assert!(max_drift < 0.1,
            "SC-4 failed: max eigenvalue drift at iter-{} = {:.3} Ha (expected <0.1 Ha)",
            iter + 1, max_drift);
    }
    
    // Verify SC-5: No Cascade
    for (iter, eigenvalues) in eigenvalue_history.iter().enumerate() {
        let band0_eig = eigenvalues[0];
        assert!(band0_eig >= -1.10 && band0_eig <= -1.01,
            "SC-5 failed: band 0 eigenvalue at iter-{} = {:.3} Ha (expected [-1.10, -1.01] Ha)",
            iter + 1, band0_eig);
    }
    
    println!("\n=== Diagnostic 3 Complete: All Success Criteria Passed ===");
}
```

**Helper functions to add**:

1. `update_psi_dev_for_next_iteration()` — Copy rotated eigenvectors back to `state.psi_dev`
2. `print_iteration_summary()` — Print per-iteration table (iter, core_mean, cu3d_mean, val_mean, nFermi_mean, cond_mean, n_locked)
3. `verify_residual_monotonicity()` — Check SC-1 for each group
4. `count_converged_bands()` — Count bands with residual < threshold
5. `compute_mean_residual()` — Mean residual for a band range
6. `compute_max_eigenvalue_drift()` — Max |λ_i[N] - λ_i[N-1]|
7. `load_castep_eigenvalues()` — Read reference eigenvalues from Cu111_CO.bands

### Acceptance

```bash
# Run Diagnostic 3
cargo test --release --test chebyshev_orthogonality_diagnostic \
  diagnostic_3_outer_loop_convergence -- --ignored --nocapture

# Expected output:
# - Per-iteration table showing residual evolution
# - All 5 success criteria assertions pass
# - Runtime: ~15-20 minutes (10 iterations × ~90s per iteration)
```

### Exploration Notes

**What we learned during criteria validation**:
1. Original SC-1 was too loose (allowed indefinite plateau) — tightened to require 20% reduction by iter-5 or iter-10
2. SC-2 and SC-3 needed worst-case tracking in addition to mean/count — added as diagnostic output
3. SC-4 "after iteration 3" was ambiguous — clarified to mean iteration-to-iteration drift for N ≥ 4
4. SC-5 range was asymmetric — fixed to [-1.10, -1.01] Ha (±0.05 Ha around -1.055 Ha)

**Baseline from Diagnostic 2**:
- Conduction bands already near convergence (mean 0.026 Ha) — expect fast convergence
- Occupied bands need work (mean 0.15 Ha) — outer loop has clear signal
- Cu 3d cluster shows highest residuals (max 0.209 Ha) — may need Harmonic RR if standard RR plateaus

**Decision point after this diagnostic**:
- If all SC pass → proceed to Diagnostic 5 (band-locking)
- If SC-2 fails → investigate filter bounds (band 0 anomaly suggests b_low may be too tight)
- If SC-3 fails → proceed to Diagnostic 4 (Harmonic RR for Cu 3d cluster)
- If SC-4 or SC-5 fails → ZHEGVD rotation instability, proceed to Diagnostic 4 (Harmonic RR)

---

## Implementation Notes

**Reuse from Diagnostic 2**:
- `chebyshev_filter_for_test_gpu()` — already exported, returns `(psi_row_gpu, hpsi_row_gpu, kernels)`
- `compute_residual_norms_for_test()` — already exported, all-GPU residual computation
- `rayleigh_ritz_with_matrices()` — already exported, standard RR with ZHEGVD

**New infrastructure needed**:
- `update_psi_dev_for_next_iteration()` — must handle GPU→GPU copy of rotated eigenvectors
- Helper functions for SC verification (all CPU-side, operate on Vec<f64> residual/eigenvalue history)

**Memory considerations**:
- 10 iterations × 160 bands × 2 Vec<f64> (residuals + eigenvalues) = ~25 KB total history
- No subspace accumulation (reuse GPU buffers across iterations)
- Peak VRAM same as Diagnostic 2 (~4 GB)

**Performance**:
- Expected runtime: 10 iterations × 90s = ~15 minutes
- Dominated by Chebyshev filter (FFT + H·ψ + S·ψ) and ZHEGVD (160×160 generalized eigenproblem)
- No optimization needed for diagnostic (one-time test)
