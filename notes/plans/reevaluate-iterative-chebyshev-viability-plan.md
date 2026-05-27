# Plan: Reevaluate Iterative Chebyshev Filtering Viability

## Context

The iterative Chebyshev proposal (PROPOSAL.md) was abandoned in favor of band-by-band CG on `feat/phase-block-cg-migration`. However, ANALYSIS.md argues we may have given up too soon, citing two **untested premises** that led to the rejection:

1. **ChASE orthogonality bound (κ₂ ≤ η·|ρ₁|^m)**: Worst-case bound for standard eigenproblems; Cu-3d near-degenerate cluster may be far less severe in practice
2. **Performance estimate (120s/sweep → 52 min/SCF)**: Contaminated by CPU bottlenecks (D_screened ~54s) that are unrelated to the Chebyshev filter itself

The PARSEC paper analysis reveals that our current single-sweep Chebyshev implementation has **three critical divergences** from the proven Algorithm 4:
- **No subspace iteration loop** (we do 1 pass per SCF; paper does 1-5 iterations per SCF)
- **No orthonormalization between filter and Rayleigh-Ritz** (paper uses Cholesky QR)
- **No λ_lb-based stabilization** (paper uses σ = e/(c - λ_lb) recurrence)

The PARSEC paper explicitly states that for **metallic systems** (like our Cu(111)+CO), Rayleigh-Ritz is **mandatory** to resolve occupancy near the Fermi level, and orthonormalization is **essential** for numerical stability.

**Key insight from ANALYSIS.md**: The codebase already has most infrastructure in place:
- ✓ Chebyshev filter with spectral bounds
- ✓ S-orthogonalization (Gram-Schmidt)
- ✓ Locking mechanism (from Davidson v1)
- ✓ Per-band residual computation
- ✗ Outer iteration loop (missing)
- ✗ Per-band Rayleigh quotients instead of ZHEGVD (missing)

**Decision point**: Should we implement iterative Chebyshev + Cholesky QR + per-band Rayleigh quotients, or proceed with band-by-band CG?

## Critical Files

- `src/eigensolver/chebyshev.rs` — Chebyshev filter (lines 41-1058), Gram-Schmidt orthogonalization (lines 950-1058)
- `src/eigensolver/rayleigh_ritz.rs` — ZHEGVD-based Rayleigh-Ritz (lines 162-310)
- `src/eigensolver/davidson.rs` — Locking mechanism (lines 147-400), residual norms (lines 276-338)
- `src/scf.rs` — Eigensolver dispatch (lines 602-716)
- `notes/ANALYSIS.md` — Diagnostic plan and decision matrix
- `notes/debug/debug-20260526-iterative-chebyshev-proposal/PROPOSAL.md` — Original proposal

## Recommended Approach

### Phase 1: Empirical Diagnostics (Validate Premises)

Before committing to either iterative Chebyshev or band-by-band CG, **run the four diagnostics** outlined in ANALYSIS.md to test the untested premises:

#### Diagnostic 1: Orthogonality After Chebyshev Filtering
**Goal**: Measure κ₂(M) where M = ⟨ψ_i|S|ψ_j⟩ after one Chebyshev filter pass (ndeg=8), comparing Gram-Schmidt vs Cholesky QR.

**Method**:
1. Load CASTEP fixture (Cu111_CO iter-2 state)
2. Run one Chebyshev filter pass (existing code)
3. **Path A**: Apply existing Gram-Schmidt S-orthogonalization
4. **Path B**: Apply Cholesky QR via cuSOLVER:
   - Compute A = W^H·S·W (cublasZgemm)
   - Cholesky factorization: A = R^H·R (cusolverDnZpotrf)
   - Triangular solve: V = W·R^{-1} (cublasZtrsm)
5. For each path, compute S-overlap matrix M on CPU
6. Compute condition number κ₂(M) via SVD
7. Report distribution of off-diagonal elements
8. Compare performance (time) and accuracy (κ₂, max |M_ij|)

**Discriminator**:
- κ₂ < 10³: Orthogonality is excellent, either method works
- κ₂ ~ 10⁶: Orthogonality is acceptable, Cholesky QR may be faster
- κ₂ > 10¹⁰: Orthogonality is broken, need Householder QR or tighter filter

**Implementation**: New test in `tests/chebyshev_orthogonality_diagnostic.rs`

**cuSOLVER API needed**:
- `cusolverDnZpotrf` (Cholesky factorization for complex double)
- Already have: `cublasZgemm`, `cublasZtrsm`

#### Diagnostic 2: Per-Band Rayleigh Quotient Residuals
**Goal**: After one Chebyshev filter + Cholesky QR, do per-band Rayleigh quotients produce lockable residuals?

**Method**:
1. Run one Chebyshev filter + Cholesky QR
2. Compute H·ψ and S·ψ
3. For each band: λ_b = ⟨ψ_b|H|ψ_b⟩ / ⟨ψ_b|S|ψ_b⟩
4. Compute residual: r_b = H·ψ_b - λ_b·S·ψ_b
5. Compute S⁻¹-weighted norm: ‖r_b‖_S⁻¹
6. Report histogram of residual norms

**Discriminator**:
- max ‖r‖ < 0.1 Ha: bands can lock quickly
- max ‖r‖ > 0.5 Ha: single pass insufficient, need outer iterations

**Implementation**: Extend `tests/eigenvalue_residual_validation.rs`

#### Diagnostic 3: Outer Iteration Convergence Rate
**Goal**: With iterative Chebyshev (filter → QR → per-band RQ → lock), how many outer iterations to lock >80% of bands?

**Method**:
1. Implement outer loop: filter → Cholesky QR → per-band RQ → lock check
2. Track n_locked per iteration
3. lock_tol = 0.01 Ha (fixed for diagnostic)
4. Report convergence curve

**Discriminator**:
- >80% locked within 3-5 iterations: iterative Chebyshev is viable
- <50% locked after 20 iterations: abandon, use CG instead

**Implementation**: New test in `tests/iterative_chebyshev_convergence.rs`

#### Diagnostic 4: Performance Profiling
**Goal**: Measure actual Chebyshev filter time without CPU bottlenecks.

**Method**:
1. Profile one Chebyshev filter pass (GPU-only)
2. Separate FFT time, cuBLAS time, kernel time
3. Extrapolate to N outer iterations

**Discriminator**:
- <1s per filter pass: 5 iterations = 5s/SCF (acceptable)
- >5s per filter pass: 5 iterations = 25s/SCF (marginal)

**Implementation**: Add timing instrumentation to `chebyshev_filter()`

### Phase 2: Implement Iterative Chebyshev (If Diagnostics Pass)

**Only proceed if Diagnostic 1 shows κ₂ < 10⁶ AND Diagnostic 3 shows >80% lock within 10 iterations.**

#### Step 1: Add Cholesky QR (if Diagnostic 1 shows it's better than Gram-Schmidt)
- Compute A = W^H·S·W via cublasZgemm
- Cholesky factorization: A = R^H·R via cusolverDnZpotrf
- Triangular solve: V = W·R^{-1} via cublasZtrsm
- **Note**: If Diagnostic 1 shows Gram-Schmidt is sufficient, skip this step

**Effort**: ~50 lines, mostly cuSOLVER boilerplate (or 0 lines if Gram-Schmidt wins)

#### Step 2: Add Per-Band Rayleigh Quotients
- After Cholesky QR, compute per-band RQ:
  ```rust
  for b in 0..n_bands {
      let num = dot_product(&psi[b], &hpsi[b]);
      let den = dot_product(&psi[b], &spsi[b]);
      eigenvalues[b] = num / den;
  }
  ```
- Batch all dot products into one cuBLAS call
- Skip ZHEGVD entirely

**Effort**: ~30 lines, reuses Davidson's dot-product pattern

#### Step 3: Add Outer Iteration Loop with Locking
- Wrap `chebyshev_filter()` + Cholesky QR + per-band RQ in outer loop
- Compute residuals: r_b = H·ψ_b - λ_b·S·ψ_b
- Lock bands where ‖r_b‖_S⁻¹ < lock_tol
- Exit when all bands locked or max_outer_iter reached
- **Reuse**: Davidson's residual computation (lines 276-338), lock_tol schedule (lines 110-119)

**Effort**: ~100 lines, mostly orchestration

#### Step 4: Integration
- Add `CHEMRUST_EIGENSOLVER=chebyshev_iterative` dispatch in `scf.rs`
- Run full SCF on Cu111_CO fixture
- Compare to CASTEP reference energy

**Effort**: ~20 lines

### Phase 3: Decision Gate

After diagnostics and (optionally) implementation:

| Outcome | Action |
|---------|--------|
| Diag 1: κ₂ > 10¹⁰ | **Abandon iterative Chebyshev**, proceed with CG |
| Diag 3: <50% lock in 20 iters | **Abandon iterative Chebyshev**, proceed with CG |
| Diag 1+3 pass, but Diag 4: >5s/filter | **Marginal**, compare to CG timeline |
| All diagnostics pass | **Implement iterative Chebyshev**, defer CG |

## Verification

### Diagnostic Tests
1. Run Diagnostic 1: `cargo test --release chebyshev_orthogonality_diagnostic -- --ignored`
   - Check κ₂(M) < 10⁶
   - Check max off-diagonal |M_ij| < 0.1 for i ≠ j

2. Run Diagnostic 2: `cargo test --release per_band_rq_residuals -- --ignored`
   - Check histogram of residual norms
   - Verify max ‖r‖ < 0.5 Ha

3. Run Diagnostic 3: `cargo test --release iterative_chebyshev_convergence -- --ignored`
   - Check convergence curve: n_locked vs iteration
   - Verify >80% lock within 10 iterations

4. Run Diagnostic 4: Profile with `nvprof` or `nsys`
   - Measure Chebyshev filter time (GPU-only)
   - Extrapolate to 5-10 iterations

### Full SCF Test (If Implemented)
1. Run: `CHEMRUST_EIGENSOLVER=chebyshev_iterative cargo test --release scf_cu111_co -- --ignored`
2. Check final energy vs CASTEP reference (within 0.1 eV)
3. Check SCF convergence (no cascade)
4. Monitor GPU memory usage (should stay <7 GB)

## Why This Approach

1. **Empirical validation first**: The two untested premises (orthogonality destruction, performance) can be measured directly on our Cu(111)+CO system. No need to guess.

2. **Low risk**: Diagnostics are read-only tests that don't modify production code. If they fail, we abandon iterative Chebyshev with concrete evidence.

3. **Reuses existing infrastructure**: 80% of the code already exists (filter, orthogonalization, locking, residuals). Only the outer loop orchestration is new.

4. **Avoids premature optimization**: Band-by-band CG is a heavier refactor (new line search, per-band state, sequential optimization). If iterative Chebyshev works, we save that effort.

5. **Aligns with PARSEC paper**: The paper's Algorithm 4 is proven for metallic systems with NCPP. Our USPP generalization is the only unknown, but S-orthogonalization (already implemented) should handle it.

6. **Decision matrix is clear**: ANALYSIS.md provides objective discriminators (κ₂ thresholds, lock percentages) to decide between Chebyshev and CG.

## Risks

1. **Diagnostic 1 may show κ₂ > 10¹⁰**: If orthogonality is severely broken, Cholesky QR won't help. Mitigation: Fall back to Householder QR (more expensive but stable).

2. **Diagnostic 3 may show slow convergence**: If <50% bands lock after 20 iterations, iterative Chebyshev is too slow. Mitigation: Proceed with band-by-band CG as planned.

3. **USPP generalization may fail**: PARSEC uses NCPP (S=I); our USPP (S≠I) is untested. Mitigation: S-orthogonalization (already implemented) should handle it, but verify in Diagnostic 1.

4. **Memory footprint may exceed 8 GB**: Cholesky QR adds n_bands² buffers (~400 KB). Mitigation: Profile in Diagnostic 2, should be negligible.

## Timeline Estimate

- **Phase 1 (Diagnostics)**: 2-3 days
  - Diagnostic 1: 4 hours (S-overlap matrix, SVD)
  - Diagnostic 2: 4 hours (per-band RQ, residuals)
  - Diagnostic 3: 8 hours (outer loop prototype)
  - Diagnostic 4: 2 hours (profiling)

- **Phase 2 (Implementation, if diagnostics pass)**: 3-4 days
  - Cholesky QR: 1 day
  - Per-band RQ: 0.5 day
  - Outer loop: 1.5 days
  - Integration + testing: 1 day

- **Total**: 5-7 days (vs ~2 weeks for band-by-band CG)

## Recommendation

**Start with Phase 1 diagnostics.** They are low-cost, low-risk, and will provide concrete evidence to decide between iterative Chebyshev and band-by-band CG. If diagnostics pass, iterative Chebyshev is the faster path to a working eigensolver. If they fail, we have objective data to justify the CG refactor.
