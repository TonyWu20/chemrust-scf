# Diagnostics Status & Evaluation: Iterative Chebyshev Filtering Viability

**Date**: 2026-05-27  
**Branch**: `diag/iterative-chebyshev-viability`  
**Authoritative tracking document** for the reevaluation of iterative Chebyshev filtering as the chemrust-scf eigensolver.

---

## 1. Why This Reevaluation Exists

The iterative Chebyshev proposal was abandoned (2026-05-24) in favor of band-by-band CG after ~10 sessions of debugging the Chebyshev-RR cascade. Two **untested premises** drove the decision:

1. **ChASE orthogonality bound** (κ₂ ≤ η·|ρ₁|^m): Worst-case bound for standard eigenproblems; never measured on our system.
2. **Performance estimate** (120s/sweep → 52 min/SCF): Contaminated by CPU bottlenecks (D_screened ~54s) unrelated to the Chebyshev filter itself.

The reevaluation plan (`~/.claude/plans/reevaluate-notes-debug-debug-20260526-it-fizzy-rabbit.md`, also saved at `notes/plans/reevaluate-iterative-chebyshev-viability-plan.md`) proposed a diagnostic framework to test both premises empirically before committing to either algorithm.

---

## 2. Diagnostic Results

### ✅ Diagnostic 1: Orthogonality After Chebyshev Filtering (BareH)

| Metric | Value | Verdict |
|--------|-------|---------|
| κ₂ (condition number) | 1.0000e0 | ✅ EXCELLENT |
| Off-diagonal max | 7.6e-15 | ✅ Machine epsilon |
| Diagonal min/max | 1.0 / 1.0 | ✅ Perfect |

**Date**: 2026-05-26  
**Filter mode**: BareH  
**System**: Cu111_CO (160 bands, 60067 PW, USPP), ndeg=8  
**Key finding**: The Chebyshev filter preserves orthogonality to machine precision. The ChASE worst-case bound does NOT apply to our system — all occupied bands are in a narrow energy range, so |ρ₁| ≈ 1.  
**Files**: `DIAGNOSTIC_1B_RESULT.md` (combined 1+1b report), `notes/diagnostic-1b-result.md`  
**Memory**: [[diagnostic_1_result.md]]

### ✅ Diagnostic 1b: Orthogonality with SinvHKeepHEig (Production Mode)

| Filter Mode | κ₂ | Off-Diagonal Max | Diagonal | Verdict |
|-------------|----|-----------------|----------|---------|
| **BareH** (baseline) | 1.0000e0 | 7.6e-15 | 1.0 / 1.0 | ✓ |
| **SinvHKeepHEig** (production) | 1.0000e0 | 2.9e-15 | 1.0 / 1.0 | ✓ |
| **SinvHFullDas** (full Das Alg 3) | 1.0000e0 | 2.0e-15 | 1.0 / 1.0 | ✓ |

**Date**: 2026-05-26  
**Critical gap closed**: Original Diagnostic 1 used BareH, which is physically incorrect for USPP. This validated the production code path.  
**Key finding**: Woodbury-based S⁻¹·H application (ζ = 3.8e-15) does NOT introduce orthogonality loss. Factor C from HANDOFF.md (USPP S⁻¹ complexity) is NOT a blocker.  
**Memory**: [[diagnostic_1b_validated.md]]

### ✅ Diagnostic 2: Per-Band Residual Norms After Single Filter Pass

| Group | Count | S⁻¹ Max | S⁻¹ Mean | L2 Max | L2 Mean |
|-------|-------|---------|---------|--------|---------|
| DeepCore (band 0) | 1 | 4.2e-2 | 4.2e-2 | 5.7e-2 | 5.7e-2 |
| **Cu 3d** | 14 | **2.09e-1** | **1.55e-1** | **5.02e-1** | **3.37e-1** |
| Valence | 67 | 1.68e-1 | 1.32e-1 | 3.59e-1 | 2.70e-1 |
| NearFermi | 15 | 1.05e-1 | 7.88e-2 | 2.00e-1 | 1.42e-1 |
| Conduction | 63 | 5.70e-2 | **2.60e-2** | 8.62e-2 | **4.31e-2** |

**Date**: 2026-05-27  
**Test time**: 93.31s  
**Key findings**:
- Conduction bands converge well (mean 0.026 Ha) — prime candidates for early locking
- Occupied bands (81 bands) need outer loop work (0.13-0.21 Ha)
- Cu 3d residual ratio (cu3d/separated) = 1.44 — RR mixing confirmed
- Band 0 anomaly: ranked 58th in residual (4.2e-2 Ha), not top 5 as expected
- All 5 assertions passed
**Files**: `DIAGNOSTIC_2_RESULT.md`, `HANDOFF.md`

### 🔄 Diagnostic 3: Outer Loop Convergence

**Status**: Implementation in progress. Split into 3 tasks:
- **A1**: `chebyshev_filter_iteration_gpu` wrapper — GPU-resident filter iteration function
- **A2**: Helper functions for the diagnostic test
- **A3**: `diagnostic_3_outer_loop_convergence` test with SC-1 through SC-5

**Success criteria**:
| Criterion | Description | Threshold |
|-----------|-------------|-----------|
| SC-1 | Residual monotonicity per group | ≤2 non-consecutive violations, ≥20% reduction by iter-5 or iter-10 |
| SC-2 | Conduction band early convergence | ≥50/63 bands with S⁻¹ residual < 0.01 Ha at iter-5 |
| SC-3 | Occupied band residual reduction | 5× reduction (iter-1 → iter-10) for bands 0-81 |
| SC-4 | Eigenvalue stability | Max drift < 0.1 Ha for iterations N ≥ 4 |
| SC-5 | No cascade signature | Band 0 eigenvalue in [-1.10, -1.01] Ha across all iterations |

**Plan**: `notes/plans/diagnostic-3-outer-loop/PHASE_PLAN.md`, `notes/plans/diagnostic-3-outer-loop/TASKS.md`

### ⬜ Diagnostic 4: Harmonic RR vs Standard RR

**Status**: Queued. Will test if Harmonic RR stabilizes the Cu 3d degenerate cluster.

### ⬜ Diagnostic 5: Band-Locking Behavior

**Status**: Queued. Will test band-locking with lock_tol = 0.01 Ha.

---

## 3. Updated Decision Matrix

| Diag 1/1b (orth) | Diag 2 (residuals) | Diag 3 (convergence) | Conclusion |
|:---:|:---:|:---:|---|
| **κ₂=1.0 ✅** | **max=0.209 Ha ✅** | **TBD** | **← CURRENT: Awaiting Diag 3** |
| κ₂=1.0 | max <0.1 Ha | >80% lock in 3-5 iters | **Iterative Chebyshev works. Implement.** |
| κ₂=1.0 | max >0.5 Ha | >80% lock in 6-10 iters | **Chebyshev viable but slow. Worth using.** |
| κ₂=1.0 | any | >80% lock in ≤10 iters | **Chebyshev viable. Proceed.** |
| κ₂=1.0 | any | <50% lock in 10 iters | **Chebyshev marginal. Pivot to CG/Davidson.** |

---

## 4. What Has Been Falsified (Premises)

### Originally untested premise 1: "Chebyshev filtering destroys orthogonality"
**Status**: **FALSIFIED** (Diagnostic 1/1b, 2026-05-26)  
**Evidence**: κ₂=1.0 for all three filter modes. Off-diagonal elements at machine epsilon (10⁻¹⁵).  
**Impact**: The ChASE worst-case bound does not apply to our system. The filter operator is NOT the source of the cascade.

### Originally untested premise 2: "Performance is unacceptable (120s/sweep)"
**Status**: **NOT YET TESTED** (Diagnostic 4 deferred)  
**Notes**: The 120s was contaminated by CPU bottlenecks. GPU-only time is expected to be much lower. This is the lowest-priority diagnostic since algorithm viability is the binding constraint.

### Prior conclusion: "Chebyshev-RR architecturally unsuitable" 
**Status**: **SUPERSEDED** by diagnostics. The conclusion was based on single-sweep implementation without outer loop. The PARSEC Algorithm 4 (subspace iteration) is fundamentally different from what was tested.

---

## 5. What Remains Open

### Diagnostic 3 (outer loop): The binding question
If residuals decrease monotonically over 10 outer iterations, iterative Chebyshev is viable. The SC-1 through SC-5 criteria define precisely what "viable" means. This is the highest-priority remaining diagnostic.

### Diagnostic 4 (Harmonic RR): Mitigation for Cu 3d cluster
Only needed if Diagnostic 3 shows the Cu 3d cluster converges significantly slower than other bands. If standard RR handles the degeneracy, Harmonic RR can be deferred.

### Diagnostic 5 (band-locking): Optimization
Only needed if Diagnostics 3-4 show outer loop convergence. Band-locking reduces computation by skipping converged bands.

### b_low spectral bounds: Tuning question
The current b_low = max_veff = 0.089 Ha may not be optimal. Diagnostic 3 log data showed b_low drift (0.089 → 0.048 → -0.109 → ...) when using max_veff fallback. The `eig[n_occ-1]` fix should stabilize this.

---

## 6. Load-Bearing Documents

| Document | Purpose |
|----------|---------|
| `notes/plans/reevaluate-iterative-chebyshev-viability-plan.md` | Original reevaluation plan (Phase 1 diagnostics) |
| `notes/plans/diagnostic-3-outer-loop/PHASE_PLAN.md` | Diagnostic 3 plan with SC-1 through SC-5 |
| `notes/plans/diagnostic-3-outer-loop/TASKS.md` | Diagnostic 3 task breakdown (A1, A2, A3) |
| `docs/DIAGNOSTIC_PLAN.md` | Full diagnostic plan (1b-5) with implementation details |
| `notes/ANALYSIS.md` | Analysis of PARSEC paper vs current implementation |
| `DIAGNOSTIC_1B_RESULT.md` | Combined Diagnostic 1+1b results |
| `DIAGNOSTIC_2_RESULT.md` | Diagnostic 2 per-band residual baseline |
| `HANDOFF.md` | Session handoff with implementation details |
| `docs/chebyshev-filter-paper-findings.md` | Chebyshev filter mechanism from Zhou 2014 — technical reference on b_low, spectral bounds, and the n_occ/s discrepancy fix |
| `docs/abinit-chebyshev-scf-inner-loop.md` | ABINIT's Chebyshev SCF inner loop structure (nnsclo, nnsclo_now logic) — reference for how another production code handles the outer loop |

---

## 7. Change Log

| Date | Change |
|------|--------|
| 2026-05-26 | Original reevaluation plan written |
| 2026-05-26 | Diagnostic 1 ✅ (BareH, κ₂=1.0) |
| 2026-05-26 | Diagnostic 1b ✅ (all modes κ₂=1.0) |
| 2026-05-27 | Diagnostic 2 ✅ (residual baseline) |
| 2026-05-27 | Diagnostic 3 🔄 (implementation started) |
| 2026-05-27 | This tracking document created |
