# Review: Phase-0 Band-by-Band CG Algorithm De-risking

**Tasks**: `notes/plans/phase-block-cg-migration/TASKS.md`
**Reviewed**: 2026-05-26
**Verification source**: CASTEP 6.11 Fortran source (electronic.f90, nlpot.f90, wave.f90)

## Summary

**Overall**: NEEDS FIXES — Groups A and B algorithms are faithful to CASTEP, but two
critical defects block Gate 1 pass and Gate 2 is unimplemented (9 compilation errors,
all hallucinated APIs).

**Outcome verification**: 37/37 unit tests pass (synthetic data). Gate 1
integration test cannot pass due to line-search singularity. Gate 2 does not compile.

**CASTEP fidelity**: Algorithms A1-A3, B1 are character-by-character matches to
CASTEP Fortran source (verified against electronic.f90:9959-10185, 6238-6437,
11639-12019; nlpot.f90:15396-15680; wave.f90:29889-29893).

## Per-Task Results

### TASK-A1: USPP Preconditioner Module — ✓ Passed

- **CASTEP match**: TPA formula (wave.f90:29889-29893) — exact.
  R = (−Q⁻¹ − C)⁻¹ with C = β†·T⁻¹·β (nlpot.f90:15480) — exact.
  Apply: P⁻¹·r = T⁻¹·r + T⁻¹·β·R·β†·(T⁻¹·r) (nlpot.f90:15640-15664) — exact.
- **Unit tests**: 6/6 pass — R Hermiticity, C Hermiticity, self-consistency,
  TPA floor > 0.01, β=0 reduction to T⁻¹, non-trivial result.
- **Issue**: Not used in Gate 1/2 tests — both pass `beta_g_dummy = zeros`.

### TASK-A2: Line Search Module — ✓ Passed (with CASTEP caveat)

- **CASTEP match**: Quadratic (electronic.f90:10056-10184) — exact match on:
  a, b=−2·Re⟨Hψ|d⟩, c, d_s; ad,bd variables; det formula; r1,r2 roots;
  root selection (lower E); step = −r_chosen; clamp at 15.0; parabola fallback.
- **Soft fallback difference**: CASTEP calls `io_abort` on complex roots or
  zero bd; Rust returns `(s=0, ε=a, status=-1)`. Acceptable for library code.
- **Unit tests**: 6/6 pass — synthetic parabola, rational function, energy decrease,
  clamping, no-minimum, monotonicity.
- **Critical issue**: When ‖d‖ → 0 (near-converged state), r1 ∝ 1/‖d‖ → ∞.
  This is the Gate 1 root cause (see Issues #1).

### TASK-A3: CG Helper Functions — ✓ Passed

- **CASTEP match**: Residual r = Hψ − ε·Sψ (electronic_CG_direction_bks:6356-6359).
  S-normalization, S-orthogonalization (modified Gram-Schmidt in S-metric).
- **Unit tests**: 19/19 pass — S-normalize identity, residual orthogonality,
  S-orthogonalization (S=I and S≠I), eigenvalue computation, panic guards.
- **Issue**: `s_orthogonalize_against` in `band_cg.rs:217` passes `&g` for both
  `psi` and `spsi` — implicitly S=I. For USPP, S|g⟩ ≠ g (see Issues #3).

### TASK-B1: Band-by-Band CG Minimizer — ✓ Passed (algorithm correct, guard missing)

- **CASTEP match (verified line-by-line)**:
  - Sign convention: d = P⁻¹·r (uphill) in Rust vs d = −P⁻¹·r (downhill) in
    CASTEP `electronic_SD_direction_bks:5426`. **Self-compensating**: the line
    search negates its root, so ψ_new = ψ + step·d is identical in both conventions
    (verified algebraically).
  - Fletcher-Reeves β = Re⟨g|r⟩ (electronic_CG_direction_bks:6397) — exact.
  - γ = β/β_old (line 6407) — exact.
  - CG update d = γ·d_old − g (line 6416, `wave_add(gamma, -cmplx_1)`) — exact.
  - SD first 2 steps, CG thereafter — exact.
  - Convergence by |ε_new − ε_old| < tol (electronic_find_eigenstate:11970) — exact.
  - ψ update: linear H|ψ⟩ update (line 11937) — exact.
  - S-orthogonalize d against current band (line 6422) — exact.
- **Unit tests**: 6/6 pass — SD decreases eigenvalue, energy monotonicity,
  residual decrease, convergence flag, converged-band isolation.
- **Critical issue**: Zero-gradient guard (line 200) uses `f64::EPSILON * 1000.0`
  (≈ 2.2e-13), far too tight. A residual of 1e-8 Ha from H_Rust ≠ H_CASTEP
  passes through, enters the line-search singularity, and produces step_size ≈ −1e10.
  See Issues #1.

### TASK-C1: Gate 1 Consistency Check — ✗ Failed

- **Status**: Test compiles but cannot pass.
- **Root cause**: CASTEP's converged ψ has r ≈ 0 internally, but Rust's H and S
  operators differ from CASTEP's (separate code paths), producing r ≉ 0
  (small but non-zero). The line search then explodes via the ‖d‖ → 0 singularity.
- **Expected failure mode**: step_size ≈ −1e10, ψ gets destroyed, eigenvalue
  drifts by ≫ 1e-10 Ha.
- **Fix**: Add residual-norm guard before line search (see fix-tasks.md).

### TASK-C2: Gate 2 Convergence Test — ✗ Not implemented

- **Status**: 9 compilation errors — all hallucinated APIs.
- `AugConfig` does not exist in `chemrust_hamiltonian_core::atomic_solver::augmented`.
- `RealLattice::volume()` does not exist (field access, not method).
- `LocalPotential.v_data` does not exist (tuple struct).
- `BesselBasis::new` takes 1 argument, not 2.
- `UspData` trait methods accessed as inherent methods (`.projectors()`, `.q_aug()`,
  `.q_func()`, `.radial_grid()`, `.core_charge()`).
- Wrong element: uses C 2s instead of Cu 3d as specified in TASKS.md.
- Uses TPA-only preconditioner (zero beta) instead of full USPP.

## Issues Found

### Issue #1 (CRITICAL): Line search singularity for near-converged states

**Location**: `src/eigensolver/line_search.rs:90-91` via `band_cg.rs:318`
**Severity**: Blocks Gate 1 and Gate 2.
**Root cause**: The quadratic `bd·r² + 2(ad−c)·r − b = 0` has r1 ∝ 1/‖d‖
when ‖d‖ → 0. For r ≈ 1e-10 (from H_Rust ≠ H_CASTEP), bd ≈ 2e-30, r1 ≈ 1.6e10,
step_size ≈ −1.6e10. ψ gets destroyed: `(ψ − 1.6·unit)/1.9`.

CASTEP avoids this because its internal H/S loop produces genuinely-zero residual
for its own converged states; there is no cross-codebase operator mismatch.

**Fix**: Add `‖r‖ < MIN_RESIDUAL` guard in `band_cg_minimize` BEFORE the line
search call. Threshold: `max(tol * 0.01, 1e-12 * |ε|)` returns current state
as converged. Also add a guard in `line_search_2d_quadratic` itself: if
`|bd| < 1e-30`, return `(s=0, ε=a, status=-2)`.

### Issue #2 (HIGH): Gate 1/2 tests use TPA-only preconditioner, not full USPP

**Location**: `tests/phase0_gate1_consistency.rs:306-308`, `tests/phase0_gate2_convergence.rs:386-388`
**Severity**: Tests do not verify the full USPP preconditioner (Task A1).
**Fix**: Construct `UsppPreconditioner` from per-ion `beta_g` and `q_exp` arrays.

### Issue #3 (MEDIUM): Gradient orthogonalization assumes S|g⟩ = g

**Location**: `src/eigensolver/band_cg.rs:217`
**Severity**: Wrong for USPP; masked because Gate 1/2 use empty converged_bands.
**Fix**: Either extend `apply_hs` to a separate `apply_s` closure, or compute
S|g⟩ inside `band_cg_minimize`.

### Issue #4 (LOW): Missing kinetic eigenvalue update

**Location**: `src/eigensolver/band_cg.rs` (absent)
**Severity**: CASTEP line 11942 updates `ek` for preconditioning after each step.
Not needed with TPA-only preconditioner, but needed for full USPP preconditioner.
**Defer to**: Phase-1 (when full USPP preconditioner is used).

### Issue #5 (LOW): Soft fallbacks mask bugs

**Location**: `src/eigensolver/line_search.rs:82-99`, `122-141`
**Severity**: CASTEP would `io_abort`, alerting the developer. Rust silently
returns step=0 with status=-1. Acceptable in production, but during Phase-0
de-risking, these should at minimum log a warning.

## Deferred Items

1. **Full S|g⟩ computation in orthogonalization** — needs apply_s closure plumbing.
   Can be done when converged_bands becomes non-empty (Phase-1).
2. **Kinetic eigenvalue update** — needed for USPP preconditioner with non-zero beta.
3. **Gate 2: pseudoatomic guess for Cu 3d** — requires `solve_atom` for Cu, not C.
4. **Gate 2: Bessel projection for l=2 (Cu 3d)** — only l=0 implemented.
5. **Parity test for line search** — verify Rust and CASTEP produce identical
   step_size for a set of synthetic (a,b,c,d_s) tuples.
