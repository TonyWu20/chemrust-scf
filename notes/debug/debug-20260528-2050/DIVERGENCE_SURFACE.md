# Divergence Surface: CG Implementation vs CASTEP

## Layer 1: Algorithm Implementation (Phase-0 modules)

### 1. USPP Preconditioner

| Item | Status | Evidence |
|------|--------|----------|
| P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹ formula | ✓ Matches CASTEP | uspp_preconditioner.rs:37-43, nlpot.f90:15480 |
| R = (−Q⁻¹ − C)⁻¹ computation | ⚠ To verify | Uses block-diagonal Q across ions (Gate 2 lines 377-382) vs CASTEP's per-ion R |
| TPA diagonal: temp = 27 + 18x + 12x² + 8x³ | ✓ Matches CASTEP | uspp_preconditioner.rs:152-163, wave.f90:29889-29893 |
| Gate 1: TPA-only (β=0 dummy) | ⚠ Simplified | Gate 1 line 307: uses dummy β=0, Q=eye — effectively plain TPA, not full USPP |
| Gate 2: Full USPP preconditioner | ⚠ Block-diagonal Q | Concatenates all ions' β into one matrix; Q is block-diagonal. Cross-ion coupling handled by dense C = β†·T⁻¹·β. Matches CASTEP's per-k-point R computation. |

### 2. Line Search

| Item | Status | Evidence |
|------|--------|----------|
| Quadratic: b·d·s² + 2(a·d−c)·s − b = 0 | ✓ Matches CASTEP | line_search.rs:73-117, electronic.f90:10074-10090 |
| Roots: r1,r2 = (−ad+c±det)/bd | ✓ Matches CASTEP | line_search.rs:116-117 |
| Step negation: step_size = -chosen_r | ✓ Matches CASTEP | line_search.rs:156, electronic.f90:10109 |
| Simple parabola fallback (d=0) | ✓ Matches CASTEP | line_search.rs:158-166, electronic.f90:10135-10148 |
| Clamp at 15.0 | ✓ Matches CASTEP | line_search.rs:172-176, electronic.f90:10151 |
| b = -2·Re⟨Hψ\|d⟩ sign convention | ✓ Matches CASTEP | line_search.rs:66, electronic.f90:10057 |

### 3. CG Direction Update

| Item | Status | Evidence |
|------|--------|----------|
| β = Re⟨g_orth\|r⟩ (not ⟨g\|g⟩) | ✓ Matches CASTEP modified-FR | band_cg.rs:251, electronic.f90:6397 |
| γ = β/β_old (FR) | ✓ Matches CASTEP | band_cg.rs:256, electronic.f90:6407 |
| d = γ·d_old − g_orth | ✓ Matches CASTEP | band_cg.rs:266-269, electronic.f90:6416 |
| SD first 2 steps | ✓ Matches CASTEP | band_cg.rs:244, electronic.f90:11896-11898 |

### 4. S-Orthogonalization

| Item | Status | Evidence |
|------|--------|----------|
| Lower-only S-orth (converged bands) | ✓ Correct MGS in S-metric | cg_helpers.rs:160-215 |
| Direction S-orth against current band | ✓ Correct by-linearity | band_cg.rs:286-312, electronic.f90:6422-6423 |
| Residual = Hψ − ε·Sψ | ✓ Correct | cg_helpers.rs:48-77 |

### 5. Convergence Criterion

| Item | Status | Evidence |
|------|--------|----------|
| \|ε_new − ε_old\| < tol | ✓ Matches CASTEP | band_cg.rs:404, electronic.f90:11970 |
| residual_norm = bare L2 (not S-norm) | ⚠ Approximation | cg_helpers.rs:83-98 — documented as approximation; S-norm would require S⁻¹ solve |

### 6. Hψ/Sψ Linear Update

| Item | Status | Evidence |
|------|--------|----------|
| Hψ_new ← (Hψ_old + s·Hd)/norm | ✓ Matches CASTEP | band_cg.rs:370-384, electronic.f90:11937 |
| Sψ_new ← (Sψ_old + s·Sd)/norm | ✓ Matches CASTEP | band_cg.rs:370-384 |

## Layer 2: Integration Gap (SCF wiring)

### 7. SCF Loop Integration

| Item | Status | Evidence |
|------|--------|----------|
| CG wired into SCF dispatch | ❌ NOT DONE | scf.rs:224 — hardcoded `chebyshev_filter` call, no dispatch |
| DiagonalizeMode enum | ❌ NOT CREATED | No such enum exists in src/ |
| Block CG (batched) | ❌ NOT DONE | Only serial band-by-band |
| Block-end Rayleigh-Ritz | ❌ NOT DONE | CASTEP rotates within-block after CG sweep |
| Full-orth trigger (residual < 0.2·n_bands) | ❌ NOT DONE | CASTEP electronic.f90:506-508 |
| V_NL in CG H-closure | ⚠ Gate-test-specific | Gate tests construct per-ion loops manually; no shared H/S closure |
| Warm-start (reuse ψ from previous SCF iter) | ❌ NOT DONE | Phase-0 tests start fresh each time |
| Lock mask reset per SCF iter | ❌ NOT DONE | CG operates per-band, but no lock mask concept |

## Layer 3: Gate Test Status

| Item | Status | Evidence |
|------|--------|----------|
| Gate 1 (consistency check) | ⚠ Unknown | Test exists but results not verified in this session. Uses TPA-only preconditioner. |
| Gate 2 (random init convergence) | ⚠ Unknown | Test exists but results not verified. Uses full USPP H/S and preconditioner. |
| Gate 3+ (SCF-level tests) | ❌ NOT REACHED | Requires SCF integration first |

## Summary

**Algorithm core**: The CG modules (band_cg, line_search, uspp_preconditioner, cg_helpers) follow CASTEP character-by-character. All critical formulas match. No identified algorithm bugs from code inspection.

**Unvalidated items**: Gate 1 and Gate 2 test results are unknown. USPP preconditioner correctness not confirmed by running tests.

**Integration gap**: This is the main gap. The CG modules exist as standalone library code but are not wired into the SCF loop. The SCF loop (`scf.rs:224`) is hardcoded to `chebyshev_filter`. No `DiagonalizeMode` dispatch exists. The CG code is tested only in isolation (Gate 1/2), not as part of a full SCF cycle.
