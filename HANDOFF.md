# Handoff: Diagnostic 1b Complete — Ready for Diagnostic 2

**Date**: 2025-05-26  
**Branch**: `diag/iterative-chebyshev-viability`  
**Status**: ✅ Diagnostic 1b validated, ready for next phase

---

## What Was Accomplished This Session

### Critical Validation Gap Closed

**Problem Identified**: The original Diagnostic 1 (commit c471b3d) used `FilterMode::BareH`, which is physically incorrect for USPP systems. The production code uses `FilterMode::SinvHKeepHEig` (S^{-1}·H via Woodbury), which was never validated for orthogonality preservation.

**Solution Implemented**: 
- Modified `chebyshev_filter_for_test()` to accept a `filter_mode` parameter
- Created three test variants to compare all filter modes
- Ran comprehensive orthogonality tests on Cu111_CO fixture

**Result**: ✅ **All three filter modes preserve orthogonality perfectly (κ₂ = 1.0)**

| Filter Mode | κ₂ | Off-Diagonal Max | Status |
|-------------|-----|------------------|--------|
| BareH (baseline) | 1.0000e0 | 7.6e-15 | ✓ |
| **SinvHKeepHEig (production)** | 1.0000e0 | 2.9e-15 | ✓ |
| SinvHFullDas (full Das) | 1.0000e0 | 2.0e-15 | ✓ |

### Key Findings

1. **Factor C (USPP S^{-1} complexity) is NOT a blocker**
   - Woodbury-based S^{-1} at ζ=3.8e-15 is numerically stable
   - No need to investigate alternative S^{-1} methods
   - No need to pivot away from Chebyshev filtering

2. **Cascade root cause confirmed**
   - Missing outer loop (no band-locking, no iterative refinement)
   - Single sweep per SCF step (vs CASTEP's 19-26 iterations)
   - NOT the filter operator or Woodbury precision

3. **Green light for PARSEC Algorithm 4**
   - Proceed with outer loop + band-locking + Harmonic RR
   - Continue with Diagnostics 2-5 to validate approach

### Files Modified

- `src/eigensolver/chebyshev.rs` - Added `filter_mode` parameter to `chebyshev_filter_for_test()`
- `src/lib.rs` - Exported `FilterMode` for test access
- `tests/chebyshev_orthogonality_diagnostic.rs` - Refactored into helper function + 3 test variants

### Documentation Created

- `DIAGNOSTIC_1B_RESULT.md` - Detailed test results and analysis
- `docs/DIAGNOSTIC_PLAN.md` - Full diagnostic-first implementation plan
- Memory: `diagnostic_1b_validated.md` - Project status update

---

## What To Do Next Session

### Immediate Next Step: Diagnostic 2

**Goal**: Measure per-band residual norms after a single Chebyshev filter pass to establish baseline convergence characteristics.

**Implementation** (in `tests/chebyshev_orthogonality_diagnostic.rs`):

```rust
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_2_per_band_residuals() {
    // 1. Load Cu111_CO fixture (converged state)
    // 2. Run one Chebyshev filter pass (ndeg=8, SinvHKeepHEig mode)
    // 3. Apply Gram-Schmidt orthonormalization
    // 4. Run standard Rayleigh-Ritz (ZHEGVD)
    // 5. Compute per-band residuals: r_b = H|ψ_b⟩ - λ_b·S|ψ_b⟩
    // 6. Compute S^{-1}-weighted norms: ||r_b||_{S^{-1}} = √⟨r_b | S^{-1}·r_b⟩
    // 7. Report statistics:
    //    - Which bands have residuals < 0.1 Ha, < 0.01 Ha, < 0.001 Ha
    //    - Cu 3d cluster (bands 1-14) vs well-separated bands
    //    - Histogram of residual norms
}
```

**Expected Outcome**:
- Cu 3d bands (1-14): High residuals due to degeneracy
- Well-separated bands: Low residuals
- Establishes baseline for "how many bands need more work"

**Estimated Time**: 2-3 hours

---

### Subsequent Diagnostics (After Diagnostic 2)

#### Diagnostic 3: Outer Loop Convergence (Minimal Prototype)

**Goal**: Test if residuals decrease monotonically with a simple outer loop.

**Implementation**:
- Add 5-10 iteration outer loop around existing Chebyshev filter
- No band-locking yet (all bands filtered every iteration)
- Track per-band residuals across iterations
- Use standard RR (not Harmonic RR yet)

**Decision Point**: 
- If residuals decrease → proceed to Diagnostic 4
- If residuals plateau → need Harmonic RR for degenerate clusters
- If residuals increase → fundamental problem with approach

**Estimated Time**: 1 day

---

#### Diagnostic 4: Harmonic RR vs. Standard RR

**Goal**: Test if Harmonic Rayleigh-Ritz stabilizes the Cu 3d degenerate cluster.

**Implementation**:
- Extract Cu 3d cluster (bands 1-14, eigenvalues within 0.07 Ha)
- Run 5 outer iterations with standard RR
- Run 5 outer iterations with Harmonic RR (σ = center of cluster)
- Compare residual convergence rates and eigenvector stability

**Expected Outcome**:
- Standard RR: May produce spurious rotations
- Harmonic RR: Should stabilize eigenvectors

**Estimated Time**: 1-2 days

---

#### Diagnostic 5: Band-Locking Behavior

**Goal**: Test if band-locking works without causing regression.

**Implementation**:
- Implement simple band-locking (skip converged bands in filter)
- Run 10 outer iterations with lock_tol = 0.01 Ha
- Track which bands lock and when
- Verify locked bands stay converged

**Expected Outcome**:
- Well-separated bands lock early (iterations 1-3)
- Cu 3d cluster locks late (iterations 5-10)
- No regression in locked bands

**Estimated Time**: 1 day

---

### Decision Point After Diagnostics 2-5

**If all diagnostics pass**:
- ✅ Residuals decrease monotonically with outer loop
- ✅ Harmonic RR stabilizes Cu 3d cluster
- ✅ Band-locking works without regression

**Then proceed to Phase 1**: Full implementation of PARSEC Algorithm 4 (1-2 weeks)

**If any diagnostic fails**:
- Investigate root cause
- May need spectrum slicing or different approach
- Pivot before investing in full implementation

---

## Reference Documents

- **DIAGNOSTIC_PLAN.md** - Full diagnostic-first implementation plan (copied from `/home/tony/.claude/plans/note-that-handoff-md-and-smooth-fog.md`)
- **DIAGNOSTIC_1B_RESULT.md** - Detailed Diagnostic 1b test results
- **HANDOFF.md** (this file in PARSEC paper analysis) - Original re-evaluation that identified the validation gap
- **PROPOSAL.md** - Original per-band RQ proposal (NOT recommended per plan analysis)

---

## Key Insights from This Session

1. **Terminology matters**: "Single-sweep" refers to the outer loop (1 filter+orth+RR cycle per SCF), not the polynomial degree (already 8 iterations)

2. **Zhou's claim is conditional**: "One sweep per SCF" likely applies to insulators with large HOMO-LUMO gaps, not metals with degenerate bands

3. **PARSEC Algorithm 4 is the proven solution**: Outer loop + band-locking + Harmonic RR, as described in the PARSEC paper

4. **Per-band RQ approach is experimental**: No convergence theory for degenerate eigenvalues, contradicts PARSEC reference, high risk

5. **Diagnostic-first is the right approach**: Validate hypotheses with minimal code before investing 1-2 weeks in full implementation

---

## Commands to Resume Work

```bash
# Navigate to project
cd /home/tony/programming/chemrust-scf-chebyshev-iter

# Check current branch
git status

# Run Diagnostic 2 (once implemented)
cargo test --test chebyshev_orthogonality_diagnostic diagnostic_2_per_band_residuals -- --ignored --nocapture

# View plan
cat docs/DIAGNOSTIC_PLAN.md

# View Diagnostic 1b results
cat DIAGNOSTIC_1B_RESULT.md
```

---

## Questions to Consider

1. Should Diagnostic 2 use CASTEP's converged state or iter-0 state?
   - **Recommendation**: Start with converged state (easier to debug), then test iter-0 if diagnostics pass

2. What lock tolerance should we use for Diagnostic 5?
   - **Recommendation**: Start with 0.01 Ha, then tighten to 0.001 Ha if needed

3. Should we implement Cholesky QR before or after the outer loop?
   - **Recommendation**: After outer loop validation (Diagnostic 3), before full implementation

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

## Contact Points

- **Plan file**: `docs/DIAGNOSTIC_PLAN.md`
- **Memory system**: `/home/tony/.claude/projects/-home-tony-programming-chemrust-scf-chebyshev-iter/memory/`
- **Test file**: `tests/chebyshev_orthogonality_diagnostic.rs`
- **Main implementation**: `src/eigensolver/chebyshev.rs`
