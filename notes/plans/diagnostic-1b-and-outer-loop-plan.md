# Critical Analysis: Why Previous Chebyshev Implementation Failed

**Date**: 2026-05-26  
**Context**: Reevaluation of eigensolver failure claims and proposed solutions  
**Status**: Analysis complete, recommendations provided

**🚨 CRITICAL FINDING**: Diagnostic 1 (commit c471b3d) validated orthogonality with `FilterMode::BareH`, but the production code uses `FilterMode::SinvHKeepHEig`. The κ₂=1.0 result does NOT validate the production filter mode. This must be revalidated FIRST (Diagnostic 1b) before proceeding with any outer loop implementation.

---

## Executive Summary

After thorough code review and document cross-validation, I found that **the narrative in PROPOSAL.md conflates two different iteration concepts**:

1. **Polynomial recurrence iterations** (currently: 8 steps per filter pass) — this is the "ndeg" parameter
2. **Eigensolver outer loop iterations** (currently: 1 per SCF step) — this is what CASTEP does 19-26 times

The user's critical observation is correct: **Zhou's "one sweep per SCF" claim (if it exists) applies only to insulators with large HOMO-LUMO gaps, not to metals with degenerate bands**. Our test data on Cu111_CO (metal with Cu 3d degeneracy) correctly shows one sweep is insufficient.

The current implementation does:
- ✅ 8-degree Chebyshev polynomial recurrence (iterative filter)
- ❌ Only 1 eigensolver pass per SCF step (no outer loop)
- ❌ No band-locking or deflation

The PARSEC reference paper recommends a **hybrid CheFSI + spectrum slicing** approach with an outer loop (Algorithm 4), not per-band Rayleigh quotients.

---

## Context: What Documents Claim vs. What Code Shows

### Claim 1: "Single-Sweep Eigensolver" (PROPOSAL.md line 40)

**Document says**: "chemrust does 1 eigensolver pass per SCF step"

**Code reality**: `chebyshev_filter()` in `src/eigensolver/chebyshev.rs` (lines 861-946) implements an **8-iteration recurrence loop**:
```rust
for k in 2..=ndeg {  // ndeg = 8 typically
    // 4-term recurrence: R_new = (2σ₂/e)·H·R_Y − (2σ₂/e)·c·R_Y − σ·σ₂·R_X + (2σ₂/e)·Y·Λ_Y
    // Per-band Λ updates on CPU
    // Norm stability checks
}
```

This is **Algorithm 3 from Das et al. (2025)**, not a single matrix-vector multiplication.

**Verdict**: ⚠️ **TERMINOLOGY CONFUSION** — The document conflates two iteration types:
- **Polynomial recurrence iterations** (ndeg=8) — already implemented
- **Eigensolver outer loop iterations** (1 per SCF) — this is what's missing

The "single-sweep" refers to the outer loop (1 filter+orth+RR cycle per SCF), not the polynomial degree.

---

### Claim 2: "CASTEP Does 19-26 Iterations, chemrust Does 1" (PROPOSAL.md lines 42-47)

**Document says**: "CASTEP does 19-26 eigensolver iterations per SCF step; chemrust does 1. This 20× difference is why CASTEP recovers."

**Code reality**: 
- **chemrust**: 1 × (8-iteration Chebyshev filter + Gram-Schmidt + ZHEGVD) per SCF step
- **CASTEP**: 19-26 × (band-by-band CG line search) per SCF step

**Apples-to-oranges comparison**: 
- chemrust's "1" counts **outer loops** (filter → orth → RR cycles)
- CASTEP's "19-26" counts **CG iterations** (gradient descent steps)

**Verdict**: ✅ **CORRECT OBSERVATION** — The real gap is:
- chemrust: **No outer loop** (no band-locking, no deflation)
- CASTEP: **Outer loop with band-locking** (iterate until all bands converge)

**User's insight**: Zhou's papers likely claimed "one sweep per SCF is enough" for **insulators with large HOMO-LUMO gaps** (e.g., Si nanocrystals). This does NOT apply to **metals with degenerate bands** (Cu 3d within 0.07 Ha). Our test data correctly shows one sweep is insufficient for Cu111_CO.

---

### Claim 3: "Chebyshev Filter Preserves Orthogonality (κ₂=1.0)" (HANDOFF.md line 6)

**Document says**: "κ₂ = 1.0, orthogonality is perfect, proceed with iterative Chebyshev"

**Test evidence**: `diagnostic_1_orthogonality_after_chebyshev_filter` (commit `c471b3d`) measured:
- κ₂ = 1.0000e0
- Off-diagonal max = 7.6e-15 (machine epsilon)
- Diagonal elements = 1.0 exactly

**Test scope**: Single Chebyshev pass on CASTEP's **converged state** (Cu111_CO fixture).

**Verdict**: ✅ **TRUE BUT NARROW** — Proves orthogonality in one pass on a converged state. Does NOT test:
- Repeated filtering across multiple SCF iterations
- Starting from non-converged states
- Interaction with ZHEGVD rotation

---

### Claim 4: "Cascade to -24703 eV at iter-3" (PROPOSAL.md line 24)

**Document says**: "chemrust cascades to unphysical energies (-24703 eV at iter-3 vs reference -24111 eV)"

**Test evidence**: `cascade_iter3_diagnostic()` measured band-0 eigenvalue:
- Iter-1: -1.0458 Ha (reference: -1.055 Ha) ✓
- Iter-2: -0.8690 Ha ✗
- Iter-3: **-11.94 Ha** ✗✗ (= -325 eV, not -24703 eV)

**Root cause** (EIGENVECTOR_OVERLAP.md lines 121-142):
1. ZHEGVD rotates eigenvectors within degenerate Cu 3d manifold (bands 1-14)
2. Rotated eigenvectors → different density → different V_eff
3. New V_eff → more rotation → cascade

**Verdict**: ✅ **TRUE** — Cascade is real and reproducible. The -24703 eV figure appears to be from a different test or typo (measured value is -325 eV).

---

## What the PARSEC Reference Paper Actually Recommends

### Algorithm 4: Chebyshev-Filtered Subspace Iteration

```
procedure (V,D) = CHEBYSUBIT(H, V, m, ε_F, λ_ub, λ_lb, maxiter)
  for iter = 1 → maxiter do
    W = ChebyFilter(H, V, m, ε_F, λ_ub, λ_lb);  // Apply polynomial filter
    V = Orth(W);                                 // Cholesky QR
    (V,D) = RayleighRitz(H,V);                   // Extract eigenvalues
```

**Key points**:
1. **Outer loop** (`maxiter` iterations) — currently missing in chemrust
2. **Full Rayleigh-Ritz** (not per-band Rayleigh quotients)
3. **Cholesky QR** for orthonormalization (not Gram-Schmidt)

### Section 3.3: Hybrid Polynomial Filtering

**Recommended strategy**:
1. **First few SCF iterations**: Use CheFSI (Chebyshev filtering for all bands)
2. **Subsequent SCF iterations**: Switch to **Spectrum Slicing** (divide spectrum into slices, use bandpass filters)

**Rationale** (lines 905-919):
> "When a good initial guess of the invariant subspace is available, a bandpass-filtered subspace iteration can be effectively used to refine the approximation."

**For metallic systems** (lines 304-319):
- Use temperature smearing (80 K) for degenerate states
- Include extra unoccupied states (N_s > N_occ)
- Use **Harmonic Rayleigh-Ritz** for interior eigenvalues (Algorithm 5)

---

## Critical Gap Analysis: What's Actually Missing

### Gap 1: No Outer Loop (Band-Locking)

**Current**: 1 × (filter + orth + RR) per SCF step  
**PARSEC**: `maxiter` × (filter + orth + RR) until convergence

**Impact**: Bands that haven't converged get only one refinement pass per SCF step. CASTEP's band-by-band CG iterates until each band's residual < tolerance.

**Fix**: Add outer loop with per-band residual checks and locking (like PARSEC Algorithm 4).

---

### Gap 2: No Spectrum Slicing

**Current**: All 160 bands treated uniformly in one ZHEGVD call  
**PARSEC**: Divide spectrum into slices, compute each slice independently

**Impact**: 
- ZHEGVD on 160×160 matrix is expensive and scales cubically
- No parallelization across spectral slices
- No deflation of converged bands

**Fix**: Implement spectrum slicing (PARSEC Section 3.2) with bandpass filters for interior slices.

---

### Gap 3: Wrong Orthonormalization Method

**Current**: Two-pass classical Gram-Schmidt (lines 967-1058)  
**PARSEC**: Cholesky QR (Algorithm 2)

**Impact**: Gram-Schmidt is less stable and slower for large subspaces.

**Fix**: Replace with Cholesky QR: `A = W^T·W; R = cholesky(A); V = W·R^(-1)`

---

### Gap 4: Standard RR for Degenerate Eigenvalues

**Current**: Standard Rayleigh-Ritz (solves `H_sub·Q = S_sub·Q·Λ`)  
**PARSEC**: Harmonic Rayleigh-Ritz for interior eigenvalues (Algorithm 5)

**Impact**: Standard RR can produce spurious eigenpairs for interior/degenerate eigenvalues.

**Fix**: Use Harmonic RR: solve `(H-σI)·V·c = ξ·(H-σI)²·V·c` where σ is the center of the degenerate cluster.

---

## Why the Proposed "Per-Band Rayleigh Quotient" Approach is Questionable

### PROPOSAL.md's Algorithm (lines 159-226)

```rust
for outer_iter in 0..max_outer_iter {
    let filtered = chebyshev_filter(psi, H, ndeg: 8, ...);
    H.apply(&filtered, &mut hpsi);
    S.apply(&filtered, &mut spsi);
    
    for band in 0..n_bands {
        // Per-band Rayleigh quotient: λ = <ψ|H|ψ> / <ψ|S|ψ>
        eigenvalues[band] = dot(&filtered[band], &hpsi[band]) / dot(&filtered[band], &spsi[band]);
        
        // Compute residual and check convergence
        let residual = hpsi[band] - eigenvalues[band] * spsi[band];
        if residual_norm < lock_tol { locked[band] = true; }
    }
    
    *psi = filtered;  // Update for next iteration
}
```

### Problems with This Approach

#### Problem 1: No Subspace Rotation

**Per-band RQ** computes eigenvalues but **does not rotate eigenvectors**. After filtering, you get:
- New eigenvalue estimates (λ_b)
- Same eigenvectors (ψ_b)

**Standard Rayleigh-Ritz** computes:
- New eigenvalue estimates (λ_b)
- **Rotated eigenvectors** (ψ_new = Σ_j Q_jb · ψ_j)

**Impact**: Without rotation, the eigenvectors don't improve — only the eigenvalue estimates improve. The next Chebyshev filter will operate on the same (unimproved) eigenvectors.

---

#### Problem 2: Degenerate Subspaces Need Rotation

For degenerate manifolds (Cu 3d bands 1-14 with eigenvalues within 0.07 Ha):
- **Any orthonormal basis** spanning the degenerate subspace is physically valid
- **But**: The basis must be consistent across SCF iterations to avoid density oscillations

**Per-band RQ** treats each band independently → no guarantee of consistency within degenerate clusters.

**Rayleigh-Ritz** diagonalizes the subspace → produces a consistent basis (though it may rotate unpredictably, causing the cascade).

**PARSEC's solution**: Use **Harmonic Rayleigh-Ritz** (Algorithm 5) which is stable for interior/degenerate eigenvalues.

---

#### Problem 3: Contradicts PARSEC Reference

The PARSEC paper (Section 3.2.3, lines 729-799) explicitly discusses **subspace iteration vs. Lanczos** for spectrum slicing. It recommends:
- **Subspace iteration** with **full Rayleigh-Ritz** (not per-band RQ)
- **Harmonic Rayleigh-Ritz** for interior eigenvalues

The paper does NOT mention per-band Rayleigh quotients as a viable approach.

---

#### Problem 4: Convergence Theory is Unclear

**Subspace iteration with RR** has well-established convergence theory:
- Eigenvalues converge at rate `(λ_k / λ_{k+1})^m` where m is the polynomial degree
- Eigenvectors converge to the dominant eigenvectors of `p_m(H)`

**Per-band RQ without rotation** has no established convergence theory for this problem. The proposal cites the variational principle (lines 336-343) but doesn't address:
- How eigenvectors improve without rotation
- Convergence rate for degenerate eigenvalues
- Stability for near-degenerate clusters

---

## Recommended Path Forward

Based on the user's critical observation that Zhou's "one sweep per SCF" claim doesn't apply to metals with degenerate bands, and our test data showing the cascade is real, I recommend:

### **Primary Recommendation: Option A (PARSEC Algorithm 4 with Outer Loop)**

**What to implement**:
1. **Outer loop** around existing Chebyshev filter (like PARSEC Algorithm 4)
2. **Per-band residual checks**: `r_b = H|ψ_b⟩ - λ_b·S|ψ_b⟩`
3. **Band-locking**: Skip converged bands in subsequent iterations
4. **Cholesky QR**: Replace Gram-Schmidt for better stability
5. **Harmonic Rayleigh-Ritz**: For degenerate clusters (detect via eigenvalue proximity)

**Why this is the right approach**:
- ✅ Addresses the real gap (no outer loop, not filter quality)
- ✅ Proven algorithm with convergence guarantees (PARSEC, Das 2025)
- ✅ Reuses existing Chebyshev infrastructure (8-degree polynomial is correct)
- ✅ Matches what later papers recommend (iterative CheFSI, not one-sweep)
- ✅ Lower risk than experimental per-band RQ approach

**Estimated effort**: 1-2 weeks

---

### **Alternative: Option B (Hybrid CheFSI + Spectrum Slicing)**

Only consider this if:
- Target system has 1000+ bands (current Cu111_CO has 160)
- Need better scaling on many-core systems
- Willing to invest 3-4 weeks in complex implementation

For 160-band systems, Option A is sufficient.

---

### **NOT Recommended: Option C (Per-Band RQ from PROPOSAL.md)**

**Why not**:
1. ❌ **No eigenvector rotation** — only eigenvalue estimates improve, eigenvectors stay the same
2. ❌ **Unproven for degenerate eigenvalues** — no convergence theory for near-degenerate clusters
3. ❌ **Contradicts PARSEC reference** — paper explicitly uses full Rayleigh-Ritz, not per-band RQ
4. ❌ **High risk of wasting 1-2 weeks** on an approach that may not converge

The PROPOSAL.md's per-band RQ approach is an experimental idea, not a proven algorithm. Given that we now understand the real problem (missing outer loop), we should implement the proven solution (PARSEC Algorithm 4) rather than experiment with unproven alternatives.

---

## Critical Files to Modify

1. **`src/eigensolver/chebyshev.rs`** (lines 532-1103)
   - Add outer loop with convergence check
   - Replace Gram-Schmidt (lines 967-1058) with Cholesky QR
   - Add per-band residual computation
   - Add band-locking logic

2. **`src/eigensolver/rayleigh_ritz.rs`** (if exists, or create new)
   - Implement Harmonic Rayleigh-Ritz (PARSEC Algorithm 5)
   - Detect degenerate clusters via eigenvalue proximity
   - Route degenerate clusters to Harmonic RR, others to standard RR

3. **`tests/ca_scf_convergence.rs`**
   - Add test for outer loop convergence
   - Verify band-locking behavior
   - Check cascade prevention across 30 SCF iterations

---

## Verification Plan

### Phase 1: Unit Tests (1-2 days)

1. **Test outer loop convergence**:
   - Start with CASTEP's converged state
   - Run 10 outer iterations
   - Verify per-band residuals decrease monotonically
   - Check that bands lock when residual < tolerance

2. **Test Cholesky QR**:
   - Compare against Gram-Schmidt on Cu111_CO fixture
   - Verify S-orthonormality: `V^T·S·V = I` to machine precision
   - Check numerical stability (condition number of R)

3. **Test Harmonic RR**:
   - Apply to Cu 3d cluster (bands 1-14)
   - Verify eigenvalues match standard RR
   - Check eigenvector stability (no spurious rotations)

### Phase 2: Integration Tests (3-5 days)

1. **Single SCF iteration with outer loop**:
   - Start from CASTEP iter-0 state
   - Run 1 SCF iteration with 10 outer iterations
   - Compare eigenvalues to CASTEP iter-1
   - Target: all bands within 0.05 Ha

2. **3-iteration cascade test**:
   - Run 3 SCF iterations starting from CASTEP converged state
   - Monitor band-0 eigenvalue drift
   - Target: drift < 0.1 Ha (vs. current -11.94 Ha)

3. **30-iteration SCF convergence**:
   - Run full SCF loop to convergence
   - Compare final energy to CASTEP reference
   - Target: within 0.1 eV (vs. current cascade)

### Phase 3: Performance Validation (1-2 days)

1. **Memory profiling**: Verify peak VRAM < 7 GB
2. **Timing breakdown**: Measure time per outer iteration
3. **Convergence rate**: Plot residual norms vs. outer iteration

---

## Open Questions for User

1. **Do you agree with the diagnostic-first approach?**
   - Start with 6 diagnostic tests (2-3 days), including the critical Diagnostic 1b revalidation
   - Validate outer loop + Harmonic RR + band-locking hypotheses
   - Only proceed to full implementation if diagnostics pass

2. **Which diagnostics should we prioritize?**
   - **Diagnostic 1b (SinvHKeepHEig orthogonality): HIGHEST PRIORITY** — validates the production filter mode
   - Diagnostic 2 (per-band residuals): Establishes baseline
   - Diagnostic 3 (outer loop convergence): Tests core hypothesis
   - Diagnostic 4 (Harmonic RR): Tests degenerate cluster handling
   - Diagnostic 5 (band-locking): Tests optimization strategy
   - Recommend running in order: **1b → 2 → 3 → 4 → 5**

3. **What are the acceptance criteria for diagnostics?**
   - **Diagnostic 1b: κ₂ < 10⁶ for SinvHKeepHEig filter** (if κ₂ > 10¹⁰, pivot immediately)
   - Diagnostic 3: Residuals must decrease by ≥10× after 10 outer iterations
   - Diagnostic 4: Harmonic RR must outperform standard RR on Cu 3d cluster
   - Diagnostic 5: Band-locking must not cause regression in locked bands
   - If any diagnostic fails, we pivot before investing in full implementation

4. **Should we test on CASTEP's converged state or iter-0 state?**
   - Converged state: Easier to debug, bands already close to solution
   - Iter-0 state: More realistic, tests recovery from poor initial guess
   - Recommend: Start with converged state, then test iter-0 if diagnostics pass

5. **CRITICAL: What if Diagnostic 1b shows SinvHKeepHEig destroys orthogonality?**
   - If κ₂ > 10¹⁰ for SinvHKeepHEig but κ₂ = 1.0 for BareH, the problem is the S^{-1} application via Woodbury
   - Options: (a) Investigate Woodbury precision (current ζ = 3.8e-15), (b) Use direct S^{-1} solve instead of Woodbury, (c) Pivot to Davidson or band-by-band CG
   - This is the "Factor C" from HANDOFF.md — USPP-specific complexity that PARSEC (NCPP) doesn't face

---

## Next Steps: Diagnostic-First Approach

**User's recommendation**: Add diagnostic tests to observe per-band residuals and verify whether outer loop + PARSEC's treatment can improve bands until converged and locked, BEFORE committing to full implementation.

This is the right approach — validate the hypothesis with minimal code changes before investing 1-2 weeks in full implementation.

---

### Phase 0: Diagnostic Tests (2-3 days) — **START HERE**

**Goal**: Observe per-band residual behavior and convergence characteristics with a minimal outer loop prototype.

#### Diagnostic 1b: Revalidate Orthogonality with SinvHKeepHEig Filter Mode

**CRITICAL GAP IDENTIFIED**: The existing Diagnostic 1 (commit c471b3d) used `FilterMode::BareH`, which is **physically incorrect for USPP**. The κ₂=1.0 result only validates the BareH filter, not the production SinvHKeepHEig filter that applies S^{-1}·H.

**What to test**:
1. Run the same orthogonality test with `FilterMode::SinvHKeepHEig`
2. Compute S-overlap matrix M after filtering
3. Measure κ₂(M) and compare to BareH result (κ₂=1.0)
4. Check if S^{-1}·H filtering introduces numerical instability

**Expected outcomes**:
- **Best case**: κ₂ ≈ 1.0 (same as BareH) → S^{-1}·H filter is stable, proceed with outer loop
- **Acceptable**: κ₂ ~ 10³-10⁶ → Slightly worse but usable, may need Cholesky QR
- **Failure**: κ₂ > 10¹⁰ → S^{-1}·H filter destroys orthogonality, need to investigate Woodbury precision or pivot to different approach

**Why this matters**: If SinvHKeepHEig produces κ₂ >> 1.0, then the entire PARSEC Algorithm 4 approach may not work for USPP without modifications. The HANDOFF.md correctly identified this as "Factor C: USPP S^{-1} complexity" (lines 57-73).

**Implementation**: Modify `diagnostic_1_orthogonality_after_chebyshev_filter()` to accept a filter_mode parameter and run both BareH and SinvHKeepHEig variants.

---

#### Diagnostic 2: Per-Band Residual Norms After Single Filter Pass

**What to measure**:
1. After one Chebyshev filter pass (ndeg=8) + Gram-Schmidt + standard RR
2. Compute per-band residuals: `r_b = H|ψ_b⟩ - λ_b·S|ψ_b⟩`
3. Compute S⁻¹-weighted norms: `||r_b||_S^(-1) = √⟨r_b | S^(-1)·r_b⟩`
4. Identify which bands would lock at different tolerances (0.1 Ha, 0.01 Ha, 0.001 Ha)

**Expected outcome**:
- Cu 3d bands (1-14): High residuals due to degeneracy
- Well-separated bands: Low residuals
- Establishes baseline for "how many bands need more work"

**Implementation**: Add test function `diagnostic_2_per_band_residuals()` in `tests/chebyshev_orthogonality_diagnostic.rs`

---

#### Diagnostic 3: Outer Loop Convergence (Minimal Prototype)

**What to implement**:
1. Simple outer loop (5-10 iterations) around existing Chebyshev filter
2. No band-locking yet (all bands filtered every iteration)
3. Track per-band residuals across iterations
4. Use standard RR (not Harmonic RR yet)

**What to measure**:
1. Do residuals decrease monotonically?
2. Which bands converge first? (expect well-separated bands)
3. Which bands converge slowly? (expect Cu 3d cluster)
4. How many outer iterations needed to get 80% of bands below 0.01 Ha?

**Expected outcome**:
- If residuals decrease: outer loop is working, proceed to Phase 1
- If residuals plateau: need Harmonic RR for degenerate clusters
- If residuals increase: fundamental problem with approach

**Implementation**: Add test function `diagnostic_3_outer_loop_convergence()` in `tests/chebyshev_orthogonality_diagnostic.rs`

---

#### Diagnostic 4: Harmonic RR vs. Standard RR on Cu 3d Cluster

**What to test**:
1. Extract Cu 3d cluster (bands 1-14, eigenvalues within 0.07 Ha)
2. Run 5 outer iterations with standard RR
3. Run 5 outer iterations with Harmonic RR (σ = center of cluster)
4. Compare residual convergence rates

**What to measure**:
1. Standard RR: Do residuals decrease or oscillate?
2. Harmonic RR: Do residuals decrease monotonically?
3. Eigenvector stability: Do eigenvectors rotate unpredictably?

**Expected outcome**:
- Standard RR: May produce spurious rotations within degenerate cluster
- Harmonic RR: Should stabilize eigenvectors and improve convergence

**Implementation**: Add test function `diagnostic_4_harmonic_rr_comparison()` in `tests/chebyshev_orthogonality_diagnostic.rs`

---

#### Diagnostic 5: Band-Locking Behavior

**What to test**:
1. Implement simple band-locking (skip converged bands in filter)
2. Run 10 outer iterations with lock_tol = 0.01 Ha
3. Track which bands lock and when

**What to measure**:
1. Lock progression: How many bands lock per iteration?
2. Lock stability: Do locked bands stay converged?
3. Speedup: Does locking reduce computation time?

**Expected outcome**:
- Well-separated bands lock early (iterations 1-3)
- Cu 3d cluster locks late (iterations 5-10)
- Locked bands stay converged (no regression)

**Implementation**: Add test function `diagnostic_5_band_locking_behavior()` in `tests/chebyshev_orthogonality_diagnostic.rs`

---

### Decision Point: After Diagnostic Tests

**If diagnostics show**:
- ✅ Residuals decrease monotonically with outer loop
- ✅ Harmonic RR stabilizes Cu 3d cluster
- ✅ Band-locking works and provides speedup

**Then proceed to Phase 1**: Full implementation of PARSEC Algorithm 4

**If diagnostics show**:
- ❌ Residuals plateau or oscillate
- ❌ Harmonic RR doesn't help
- ❌ Band-locking causes regression

**Then pivot**: Investigate root cause (may need spectrum slicing or different approach)

---

### Phase 1: Implement Outer Loop (3-5 days) — **ONLY IF DIAGNOSTICS PASS**

1. **Add outer loop structure** to `chebyshev_filter()`:
   ```rust
   for outer_iter in 0..maxiter {
       // Existing 8-degree Chebyshev filter
       // Gram-Schmidt orthonormalization
       // Rayleigh-Ritz (standard or harmonic)
       // Per-band residual check
       // Band-locking logic
       if all_converged { break; }
   }
   ```

2. **Implement per-band residual computation**:
   - `r_b = H|ψ_b⟩ - λ_b·S|ψ_b⟩`
   - `||r_b||_S^(-1) = √⟨r_b | S^(-1)·r_b⟩`
   - Lock band if `||r_b||_S^(-1) < lock_tol`

3. **Add lock tolerance schedule**:
   - Start: 0.1 Ha (aggressive, lock easy bands quickly)
   - Target: 0.001 Ha (tight, ensure degenerate bands converge)
   - Decay: geometric (0.7× per SCF iteration)

### Phase 2: Replace Gram-Schmidt with Cholesky QR (1-2 days)

1. **Implement Cholesky QR**:
   ```rust
   A = W^T · S · W  // Gram matrix
   R = cholesky(A)  // Upper triangular
   V = W · R^(-1)   // Orthonormalized
   ```

2. **Compare stability** against current Gram-Schmidt
3. **Verify S-orthonormality**: `V^T·S·V = I` to machine precision

### Phase 3: Add Harmonic Rayleigh-Ritz (2-3 days)

1. **Detect degenerate clusters**:
   - Group bands where `|λ_i - λ_j| < threshold` (e.g., 0.05 Ha)
   - Cu 3d cluster: bands 1-14 with eigenvalues within 0.07 Ha

2. **Implement Harmonic RR** (PARSEC Algorithm 5):
   ```rust
   σ = (λ_min + λ_max) / 2  // Center of cluster
   A = V^T · (H - σI) · V
   B = V^T · (H - σI)^2 · V
   Solve: A·Q = B·Q·ξ  // Generalized eigenproblem
   λ = ξ^(-1) + σ      // Harmonic Ritz values
   V_new = V · Q       // Harmonic Ritz vectors
   ```

3. **Route degenerate clusters to Harmonic RR**, others to standard RR

### Phase 4: Integration Testing (3-5 days)

1. **Unit tests**:
   - Outer loop convergence (residuals decrease monotonically)
   - Band-locking behavior (locked bands stay locked)
   - Cholesky QR stability (condition number of R)
   - Harmonic RR on Cu 3d cluster (no spurious rotations)

2. **Integration tests**:
   - Single SCF iteration with outer loop (compare to CASTEP iter-1)
   - 3-iteration cascade test (verify drift < 0.1 Ha)
   - 30-iteration SCF convergence (final energy within 0.1 eV of CASTEP)

3. **Performance validation**:
   - Memory profiling (peak VRAM < 7 GB)
   - Timing breakdown (time per outer iteration)
   - Convergence rate (residual norms vs. iteration)

---

## Success Criteria

### Minimum Viable Product (MVP)

1. ✅ **Prevents cascade**: SCF converges to within 0.1 eV of CASTEP reference
2. ✅ **Fits in memory**: Peak VRAM usage < 7 GB
3. ✅ **Reasonable performance**: Total SCF time < 10 minutes for Cu111_CO

### Stretch Goals

1. ✅ **Matches CASTEP precision**: Final energy within 0.001 eV
2. ✅ **Competitive performance**: Total SCF time < 5 minutes
3. ✅ **Robust**: Works for other systems (not just Cu111_CO)

---

## Conclusion

The user's observation is **critical and correct**: Zhou's "one sweep per SCF" claim (if it exists in the original papers) applies only to **insulators with large HOMO-LUMO gaps**, not to **metals with degenerate bands**. Our test data on Cu111_CO correctly shows one sweep is insufficient.

The current implementation already has:
- ✅ Iterative Chebyshev filter (8-degree polynomial recurrence)
- ✅ USPP-aware S-orthogonalization
- ✅ Three filter modes under test (Das Algorithm 3)

What's missing:
- ❌ Outer loop with band-locking (PARSEC Algorithm 4)
- ❌ Cholesky QR for better stability
- ❌ Harmonic Rayleigh-Ritz for degenerate clusters

**Recommended approach**: **Diagnostic-first validation** (2-3 days) before committing to full implementation (1-2 weeks). Run 6 diagnostic tests to observe:
1. **Diagnostic 1b (CRITICAL)**: Revalidate orthogonality with SinvHKeepHEig filter mode (existing Diagnostic 1 used BareH, which is physically incorrect for USPP)
2. Per-band residual norms after single filter pass
3. Outer loop convergence behavior (do residuals decrease?)
4. Harmonic RR vs. standard RR on Cu 3d cluster
5. Band-locking behavior and stability
6. Integration with full SCF loop

**Decision point**: Only proceed to full PARSEC Algorithm 4 implementation if diagnostics show:
- ✅ **SinvHKeepHEig filter preserves orthogonality (κ₂ < 10⁶)** — if this fails, pivot immediately
- ✅ Residuals decrease monotonically with outer loop
- ✅ Harmonic RR stabilizes degenerate clusters
- ✅ Band-locking works without regression

**NOT recommended**: The per-band Rayleigh quotient approach from PROPOSAL.md is experimental, has no convergence theory for degenerate eigenvalues, and contradicts the PARSEC reference. High risk of failure.

---

## Key Takeaways

1. **Terminology matters**: "Single-sweep" refers to the outer loop (1 filter+orth+RR cycle per SCF), not the polynomial degree (already 8 iterations)

2. **Zhou's claim is conditional**: "One sweep per SCF" likely applies to insulators, not metals with degenerate bands

3. **Test data is correct**: The cascade is real, reproducible, and shows one sweep is insufficient for Cu111_CO

4. **Diagnostic-first is the right approach**: Validate hypotheses with minimal code before investing 1-2 weeks in full implementation

5. **PARSEC Algorithm 4 is the proven solution**: But only implement it if diagnostics confirm it will work for our system

6. **CRITICAL GAP: Diagnostic 1 used wrong filter mode**: The κ₂=1.0 result was measured with `FilterMode::BareH`, which is physically incorrect for USPP. The production code uses `FilterMode::SinvHKeepHEig` (S^{-1}·H operator), which is untested. This must be validated FIRST before any outer loop work.
