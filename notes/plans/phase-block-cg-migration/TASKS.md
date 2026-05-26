# Phase-0 Tasks: Band-by-Band CG Algorithm De-risking

**Phase**: Block CG Migration — Phase-0 (Algorithm De-risking)  
**Date**: 2026-05-26  
**Status**: Ready for implementation  
**ODD Pattern**: `/home/tony/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

## Declared Fixtures

1. **Cu111_CO CASTEP converged state**
   - Path: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.check` (155 MB)
   - Contains: Converged wavefunctions ψ, eigenvalues ε, k-points, occupations
   - Format: CASTEP binary .check file (Fortran unformatted)
   - Source: CASTEP 6.11 CPU-only run, converged to 1e-5 eV total energy tolerance

2. **Cu111_CO CASTEP V_eff**
   - Path: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.pot_fmt` (17 MB)
   - Contains: Effective potential V_eff on fine grid
   - Format: CASTEP formatted potential file
   - Source: Same CASTEP run as .check file

3. **CASTEP source code (authoritative reference)**
   - Path: `~/Downloads/CASTEP-6.11-nixos/Source/Functional/` (electronic.f90, nlpot.f90)
   - Path: `~/Downloads/CASTEP-6.11-nixos/Source/Fundamental/` (wave.f90)
   - Files:
     - `electronic.f90` — CG eigensolver (lines 11639-12019, 6238-6437, 9959-10185)
     - `nlpot.f90` — USPP preconditioner (lines 15396-15680)
     - `wave.f90` — TPA formula (lines 29889-29893)
   - Purpose: Character-by-character algorithm verification

## Success Criteria Summary

**Gate 1 (Consistency Check)**:
- Starting from CASTEP converged ψ + V_eff
- Run 1 CG iteration
- **Criterion**: max|ε_out - ε_in| < 1e-10 Ha across all bands
- **Source**: PHASE_PLAN.md line 168-169

**Gate 2 (Convergence from Pseudoatomic Guess)**:
- Starting from pseudoatomic SCF guess + CASTEP V_eff
- Run CG until convergence (max 50 steps)
- **Criterion**: ‖r_0‖_S < 1e-6 Ha where r_0 = H|ψ_0⟩ − ε_0·S|ψ_0⟩
- **Target**: ε_0 within 1e-6 Ha of CASTEP A1 = −1.05502287 Ha
- **Source**: PHASE_PLAN.md line 170-172

## Task Groups

### Group A: Infrastructure (Setup)

**Dependencies**: None  
**Estimated effort**: 2-3 hours

#### TASK-A1: USPP Preconditioner Module

**Kind**: `lib-tdd`

**Description**:
Implement USPP-aware preconditioner P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹ where R = (−Q⁻¹ − C)⁻¹.

**Algorithm** (CASTEP `nlpot.f90:15396-15480`):
1. Precompute C = β†·T⁻¹·β (n_proj × n_proj dense matrix)
   - T⁻¹ is diagonal TPA: `tpa(g) = 1/(1 + 16x⁴/temp)` where `temp = 27 + 18x + 12x² + 8x³`, `x = E_k / E_k,mean`
   - Source: `wave.f90:29889-29893`
2. Compute R = (−Q⁻¹ − C)⁻¹ via dense matrix inversion (faer)
   - Q is the USPP augmentation matrix from pseudopotential
3. Apply: `P⁻¹·r = T⁻¹·r + T⁻¹·(β·R·β†·(T⁻¹·r))`
   - Two small dense GEMVs (n_proj × n_pw), negligible cost vs H-apply

**Files**:
- Create: `src/eigensolver/uspp_preconditioner.rs`

**Changes**:
- `struct UsppPreconditioner` with fields: `r_matrix: Mat<Complex64>`, `tpa_diag: Vec<f64>`, `beta_g: Array2<Complex64>`
- `fn new(beta_g, q_matrix, kinetic_g, k_cart) -> Self` — precompute R
- `fn apply(&self, residual: &[Complex64]) -> Vec<Complex64>` — apply P⁻¹

**Success Criteria**:
1. **R matrix symmetry**: `max|R - R†| < 1e-12` (Hermitian)
2. **Preconditioner identity**: Apply P⁻¹ then P to a random vector → `‖v - P·P⁻¹·v‖ / ‖v‖ < 1e-10`
3. **TPA floor**: For Cu111_CO, `min(tpa_diag) > 0.01` (no singularities)

**Test fixture**: Cu111_CO k-point 1, ion 0 (Cu atom with 3d projectors)

**Verification granularity**: Per-function unit tests + integration test with real fixture

#### TASK-A2: Line Search Module

**Kind**: `lib-tdd`

**Description**:
Implement closed-form 2×2 quadratic line search from CASTEP `electronic.f90:9959-10185`.

**Algorithm** (CASTEP `electronic_ideal_step_size:10046-10125`, source-audited 2026-05-26):

Given current band ψ and search direction d (preconditioned steepest descent),
find step size s that minimizes:
```
E(s) = ⟨ψ + s·d|H|ψ + s·d⟩ / ⟨ψ + s·d|S|ψ + s·d⟩
     = (a + s·b + s²·c) / (1 + s²·d_)
```
where (CASTEP convention):
- a = Re⟨Hψ|ψ⟩ (= ⟨ψ|H|ψ⟩, always real)
- b = −2·Re⟨Hψ|d⟩ (CASTEP: negative sign because d = −P⁻¹·r, already a descent direction)
- c = Re⟨Hd|d⟩ (= ⟨d|H|d⟩)
- d_ = Re⟨d|S|d⟩ (CASTEP uses wave_Sdot, NOT plain dot — this is the S-overlap norm)
- ⟨ψ|S|ψ⟩ = 1 (S-normalized)
- ⟨ψ|S|d⟩ = 0 (S-orthogonal)

**Correct quadratic** (derived from dE/ds = 0, verified against CASTEP lines 10074-10090):
```
b·d_·s² + 2(a·d_ − c)·s − b = 0
```
Roots (CASTEP): `r = (−a·d_ + c ± det) / (b·d_)` where `det = sqrt((a·d_ − c)² + b²·d_)`.
CASTEP evaluates E(−r1), E(−r2), picks the one with smaller energy, then `step = −r_chosen`.
(CASTEP negates because d already points downhill — the raw root of the derivative is negative.)

**Fallback**: If d_ ≈ 0 (simple parabola, no S-norm): step = −b/(2c) if c > 0.

Clamp: if s > 15.0, set s = 15.0 (CASTEP safety limit, line 10151).

**Files**:
- Create: `src/eigensolver/line_search.rs`

**Changes**:
- `fn line_search_2d_quadratic(psi, hpsi, spsi, dir, hdir, sdir) -> Result<(f64, f64), Error>`
  - Returns: `(step_size, new_eigenvalue)`
  - Inputs: all `&[Complex64]` host arrays

**Success Criteria**:
1. **Synthetic parabola**: For E(s) = 1 + 2s + 3s², minimum at s = -1/3, verify `|s_found - (-1/3)| < 1e-12`
2. **CASTEP iter-1 step**: Starting from CASTEP ψ, one CG step → `|s - s_castep| < 1e-6` (if CASTEP logs available)
3. **Energy decrease**: For any non-converged band, `E(s) < E(0)` (downhill step)

**Test fixture**: Synthetic 2×2 generalized eigenvalue problem + Cu111_CO band-0

**Verification granularity**: Unit test (synthetic) + integration test (real fixture)

#### TASK-A3: CG Helper Functions

**Kind**: `lib-tdd`

**Description**:
Implement CG helper functions: residual computation, S-normalization, S-orthogonalization.

**Algorithm**:
1. **Residual**: r = H|ψ⟩ − ε·S|ψ⟩ where ε = ⟨ψ|H|ψ⟩ / ⟨ψ|S|ψ⟩
2. **S-norm**: ‖r‖_S = sqrt(⟨r|S⁻¹|r⟩) — requires S⁻¹ via Woodbury (chemrust-hamiltonian)
3. **S-normalize**: ψ ← ψ / sqrt(⟨ψ|S|ψ⟩)
4. **S-orthogonalize**: Modified Gram-Schmidt in S-metric
   - For each converged band ψ_i: ψ ← ψ − ⟨ψ_i|S|ψ⟩·ψ_i
   - Then S-normalize

**Files**:
- Create: `src/eigensolver/cg_helpers.rs`

**Changes**:
- `fn compute_residual(psi, hpsi, spsi) -> (Vec<Complex64>, f64)` → (residual, eigenvalue)
- `fn s_norm_residual(residual, s_overlap_matrix) -> f64` → ‖r‖_S
- `fn s_normalize(psi, s_overlap_matrix) -> Vec<Complex64>`
- `fn s_orthogonalize_against(psi, converged_bands, s_overlap_matrix) -> Vec<Complex64>`

**Success Criteria**:
1. **S-norm identity**: For CASTEP converged band-0, ⟨ψ|S|ψ⟩ = 1.0000 ± 1e-10
2. **Residual at convergence**: For CASTEP converged band-0, ‖r‖_S < 1e-8 Ha
3. **Orthogonality**: After S-orthogonalize, |⟨ψ_new|S|ψ_i⟩| < 1e-10 for all converged bands

**Test fixture**: Cu111_CO bands 0-5 from .check file

**Verification granularity**: Per-function unit tests

### Group B: Core CG Algorithm

**Dependencies**: Group A  
**Estimated effort**: 4-6 hours

#### TASK-B1: Band-by-Band CG Minimizer

**Kind**: `lib-tdd`

**Description**:
Implement serial band-by-band CG minimizer following CASTEP `electronic.f90:11639-12019`.

**Algorithm** (Modified Fletcher-Reeves CG with USPP preconditioning, SD/CG stepping):

CASTEP uses SD (steepest descent) for the first 2 steps, then switches to CG (conjugate gradient).
The CG direction update uses a modified Fletcher-Reeves formula: β = Re⟨d_new|r⟩, γ = β/β_old,
d_cg = γ·d_old − d_new.  This is FR, not Polak-Ribière (PR would use (r−r_old)·d_new in numerator).

CASTEP's sign convention: the search direction d is the preconditioned gradient (NOT negated —
d = P⁻¹·r points UPHILL).  The CG update d_cg = γ·d_old − d_new preserves this orientation.
The line search then finds the step where ψ + s·d minimizes the Rayleigh quotient (s is typically
negative since d points uphill).  CASTEP handles this by computing root r of the derivative and
setting step = −r, which yields a downhill step.

CASTEP convergence criterion: eigenvalue change |ε_new − ε_old| < tol (NOT residual norm).
The residual norm ‖r‖_S is checked in Gate-2, not in the per-iteration convergence check.

H|ψ_new⟩ is computed linearly (H is linear): H|ψ_new⟩ ← (H|ψ_old⟩ + s·H|d⟩) / norm.
S|ψ_new⟩ is updated similarly: S|ψ_new⟩ ← (S|ψ_old⟩ + s·S|d⟩) / norm.
```
Input: ψ_init, preconditioner, converged_bands, max_steps, tol, apply_hs
Output: CgResult { psi, eigenvalue, n_steps, converged, residual_norm, step_type }

1. Compute H|ψ⟩, S|ψ⟩ via apply_hs
2. S-normalize: ψ ← ψ / sqrt(⟨ψ|S|ψ⟩), scale H|ψ⟩, S|ψ⟩ by same factor
3. ε = Re⟨ψ|H|ψ⟩
4. Compute residual r = H|ψ⟩ − ε·S|ψ⟩
5. For step = 1..max_steps:
   a. Precondition: g = P⁻¹·r (preconditioned gradient, uphill direction)
   b. S-orthogonalize g against converged lower bands
   c. Step type: SD (step ≤ 2) or CG (step ≥ 3)
   d. if SD: d = g (steepest descent, d_old = d, β = ⟨g|r⟩ for later CG)
   e. if CG: β = ⟨g|r⟩, γ = β/β_old, d = γ·d_old − g (FR CG update)
   f. Apply H and S to d: compute Hd, Sd via apply_hs
   g. Line search: s = line_search_2d_quadratic(ε, Re⟨Hψ|d⟩, Re⟨Hd|d⟩, Re⟨d|Sd⟩)
   h. Update: ψ ← (ψ + s·d) / norm  where norm = sqrt(1 + s²·⟨d|S|d⟩)
   i. Update H|ψ⟩ ← (H|ψ⟩ + s·Hd) / norm (linear)
   j. Update S|ψ⟩ ← (S|ψ⟩ + s·Sd) / norm (linear)
   k. Recompute ε_new = Re⟨ψ|H|ψ⟩, r_new
   l. Check convergence: |ε_new − ε_old| < tol → converged
   m. ε_old = ε_new, r = r_new, d_old = d (for next CG step)
6. Return (max steps reached without convergence)

NOTE: CASTEP default is 2 SD + 4 CG = 6 total steps per call, repeated across SCF iterations.
For Phase-0 Gate tests, max_steps is set by the gate (1 for Gate-1, 50 for Gate-2).
```

**Source audit (verified 2026-05-26)**:
- Main loop: `electronic.f90:11865-11980` (S-normalize, loop, convergence by eigenvalue change)
- Direction update: `electronic.f90:6238-6437` — FR CG: β = Re⟨g|r⟩, γ = β/β_old (lines 6393-6416)
- Line search: `electronic.f90:9959-10185` (called per-iteration)
- Sign convention: d = P⁻¹·r (uphill), CG update d = γ·d_old − g (line 6416: wave_add with gamma,-1)
- H update: `electronic.f90:11937-11938` — linear mixing: H|ψ⟩ ← (H|ψ⟩ + s·H|d⟩) / norm
- Convergence: `electronic.f90:11965-11973` — |ε_new − ε_old| < eigenvalue_tol

**CASTEP vs TASKS.md corrections (2026-05-26 audit)**:
1. Polak-Ribière → Fletcher-Reeves: CASTEP computes ⟨g|r⟩ not ⟨r−r_old|g⟩ (verified lines 6393-6397)
2. SD first 2 steps → CG thereafter (lines 11896-11898, electronic_step_type)
3. Convergence by eigenvalue diff, not residual norm (line 11970)
4. H|ψ⟩ linear update, not re-apply (line 11937)
5. d points uphill (not negated), line search negates r to get downhill step (line 10109)

**Files**:
- Create: `src/eigensolver/band_cg.rs`

**Changes**:
- `fn band_cg_minimize(psi_init, v_eff, precond, converged_bands, max_steps, tol) -> CgResult`
- `struct CgResult { psi: Vec<Complex64>, eigenvalue: f64, n_steps: usize, converged: bool }`

**Success Criteria**:
1. **Steepest descent check**: First step with γ = 0 → direction is −P⁻¹·r
2. **Energy monotonicity**: ε_n ≤ ε_{n-1} for all steps (within numerical noise 1e-12)
3. **Residual decrease**: ‖r_n‖_S ≤ ‖r_{n-1}‖_S (monotonic convergence)

**Test fixture**: Cu111_CO band-0, pseudoatomic guess

**Verification granularity**: Integration test (full CG loop)

### Group C: Gate Tests

**Dependencies**: Group B  
**Estimated effort**: 3-4 hours

#### TASK-C1: Gate 1 — Consistency Check

**Kind**: `lib-tdd`

**Description**:
Verify that one CG iteration on CASTEP's converged state produces negligible eigenvalue drift.

**Test procedure**:
1. Load Cu111_CO .check file → extract ψ, ε for all bands at k-point 1
2. Load Cu111_CO .pot_fmt → extract V_eff
3. For each band b = 0..159:
   - Run 1 CG iteration: `band_cg_minimize(ψ_b, V_eff, precond, [], max_steps=1, tol=1e-20)`
   - Compute eigenvalue drift: Δε_b = |ε_out - ε_in|
4. Assert: max(Δε_b) < 1e-10 Ha

**Success Criteria**:
- **Gate 1 PASS**: max|ε_out - ε_in| < 1e-10 Ha across all 160 bands
- **Source**: PHASE_PLAN.md line 168-169

**Files**:
- Create: `tests/phase0_gate1_consistency.rs`

**Acceptance command**:
```bash
cargo test --test phase0_gate1_consistency -- --nocapture
```

**Expected output**:
```
Gate 1: Consistency Check
  Loaded 160 bands from Cu111_CO.check
  Max eigenvalue drift: 3.2e-11 Ha (band 47)
  PASS: max drift < 1e-10 Ha
```

**Counter-example**: If max drift > 1e-10 Ha, the CG algorithm has a bug (likely in line search or direction update).

**Test fixture scope**: Full Cu111_CO .check file (160 bands, 1 k-point)

**Verification granularity**: Per-band eigenvalue comparison

#### TASK-C2: Gate 2 — Convergence from Pseudoatomic Guess

**Kind**: `lib-tdd`

**Description**:
Verify that CG converges band-0 from pseudoatomic guess to within 1e-6 Ha of CASTEP reference.

**Test procedure**:
1. Load Cu111_CO .pot_fmt → extract V_eff
2. Generate pseudoatomic guess for band-0 using chemrust-hamiltonian's Hubbard U module
   - Call `atomic_solver::solve_pseudoatomic_scf()` for Cu 3d orbital
   - Project onto plane-wave basis
   - S-normalize
3. Run CG: `band_cg_minimize(ψ_guess, V_eff, precond, [], max_steps=50, tol=1e-6)`
4. Assert:
   - Converged within 50 steps
   - ‖r_0‖_S < 1e-6 Ha
   - |ε_0 - (-1.05502287)| < 1e-6 Ha (CASTEP A1 reference)

**Success Criteria**:
- **Gate 2 PASS**: Band-0 converges within 50 CG steps, ‖r_0‖_S < 1e-6 Ha
- **Target eigenvalue**: ε_0 = −1.05502287 ± 1e-6 Ha
- **Source**: PHASE_PLAN.md line 170-172

**Files**:
- Create: `tests/phase0_gate2_convergence.rs`

**Acceptance command**:
```bash
cargo test --test phase0_gate2_convergence -- --nocapture
```

**Expected output**:
```
Gate 2: Convergence from Pseudoatomic Guess
  Generated pseudoatomic guess for Cu 3d (band-0)
  Initial eigenvalue: -0.82 Ha
  CG iteration 1: ε = -0.95 Ha, ‖r‖_S = 3.2e-2 Ha
  CG iteration 2: ε = -1.01 Ha, ‖r‖_S = 8.1e-3 Ha
  ...
  CG iteration 12: ε = -1.055023 Ha, ‖r‖_S = 4.3e-7 Ha
  CONVERGED in 12 steps
  Final eigenvalue: -1.055023 Ha (target: -1.055023 Ha, diff: 3.2e-8 Ha)
  PASS: converged within 50 steps, ‖r‖_S < 1e-6 Ha
```

**Counter-example**: If convergence takes > 50 steps, the preconditioner is ineffective or the line search is suboptimal.

**Test fixture scope**: Cu111_CO V_eff + random initialization (CASTEP method='R')

**Verification granularity**: Per-iteration residual norm + final eigenvalue comparison

## Exploration Notes

### Exploration 1: chemrust-hamiltonian CPU API

**Finding**: chemrust-hamiltonian-core provides complete CPU-only H/S operators:
- `apply_full_hamiltonian(wave_g, fft_indices, gcart, k_cart, v_eff, gvg, vnl_psi_g)` → `Vec<Complex64>`
- `apply_nlpot(beta_g_ion, beta_phi_ion, d_matrix, band_idx, vnl_psi_g)` → accumulates V_NL
- `compute_s_overlap_matrix(wave_block, pots, cell, gvg_wave, k_cart)` → `Mat<Complex64>`
- `inner_product(psi1, psi2)` → `Complex64`

All functions operate on `&[Complex64]` host arrays. No GPU code in chemrust-hamiltonian.

**Implication**: Phase-0 can use chemrust-hamiltonian directly without wrappers. The "CPU-only" requirement is naturally satisfied.

### Exploration 2: CASTEP CG Algorithm Structure

**Source audit** (CASTEP `electronic.f90`, verified 2026-05-26):
- Main loop: `electronic_find_eigenstate` (11639-12019)
  - Outer loop: SD steps (default 2) then CG steps (default 4)
  - Inner loop: per-band CG with **Fletcher-Reeves** direction update (not Polak-Ribière)
  - Convergence: **eigenvalue change** |ε_new − ε_old| < tol (not residual norm)
- Direction update: `electronic_CG_direction_bks` (6238-6437)
  - β = Re⟨d_n|r_n⟩ (preconditioned direction · raw residual)
  - γ = β / β_old (Fletcher-Reeves, not Polak-Ribière)
  - d_n = γ·d_{n-1} − g_n (CASTEP sign convention: g = P⁻¹·r, d already points uphill)
  - The PR formula ⟨r_n − r_{n-1}|d_n⟩ / ⟨r_{n-1}|d_{n-1}⟩ is NOT used; CASTEP uses FR
- Line search: `electronic_ideal_step_size` (9959-10185)
  - Closed-form 2×2 quadratic: E(s) = (a + sb + s²c)/(1 + s²d)
  - b = −2·Re⟨Hψ|d⟩ (negative convention, d is uphill)
  - **Correct quadratic**: b·d·s² + 2(a·d − c)·s − b = 0
  - Two roots, pick the one with smaller E, step = −r_chosen (negate for downhill)
  - Clamp s ≤ 15.0
- H|ψ_new⟩ update: linear mixing — H|ψ_new⟩ ← (H|ψ_old⟩ + s·H|d⟩) / norm (exact, H is linear)

**Key corrections to initial TASKS.md (2026-05-26 audit)**:
1. **Polak-Ribière → Fletcher-Reeves**: CASTEP uses ⟨d|r⟩ in numerator, not ⟨r−r_old|d⟩ (line 6393-6397)
2. **Quadratic equation wrong**: TASKS.md gave `(ad−c)s² + bd·s + (ad−c) = 0`; correct is `bd·s² + 2(ad−c)·s − b = 0` (line 10074-10090)
3. **Convergence by eigenvalue diff, not residual norm**: CASTEP checks |ε_new − ε_old| < tol (line 11970)
4. **SD first 2 steps**: CASTEP uses steepest descent for the first 2 steps, then switches to CG
5. **wave.f90 path**: `Source/Fundamental/wave.f90`, not `Source/Functional/wave.f90`

### Exploration 3: USPP Preconditioner Formula

**Source audit** (CASTEP `nlpot.f90:15396-15480`):
```fortran
! The preconditioning is tpa + tpa sum_nm |beta_n> q_nm <beta_m| tpa
```

This is P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹ where R = (−Q⁻¹ − C)⁻¹.

**Key finding**: The formula is a rank-n_proj Woodbury correction to diagonal TPA, structurally identical to chemrust-scf's existing `apply_s_inverse` (S⁻¹ = I + β·M⁻¹·β†).

**Implication**: Can reuse the Woodbury pattern. R is precomputed once per k-point (small dense matrix, ~10-50 projectors).

### Exploration 4: Convergence Criterion Adjustment

**Initial assumption**: Gate 2 requires "converge band-0 to within 1e-6 Ha of CASTEP A1".

**User clarification**: The convergence metric is residual norm ‖r‖_S < 1e-6 Ha, not eigenvalue difference. The eigenvalue will automatically be correct if the wavefunction converges.

**Implication**: The test should check ‖r_0‖_S < 1e-6 Ha as the primary criterion. The eigenvalue comparison |ε_0 - (-1.05502287)| < 1e-6 Ha is a secondary check to verify the converged state matches CASTEP.

### Exploration 5: Initial Guess Strategy

**Initial assumption**: Gate 2 uses "random ψ" but unclear what that means.

**User challenge**: "This is not physical. How can you expect you can converge without generating guesses for C and O atom too?"

**Source audit** (CASTEP `wave.f90:1900-1949`):
CASTEP's `wave_initialise` with method='R' generates random complex coefficients:
- Uniform distribution: ψ[g] = (rn1 - 0.5) + i·(rn2 - 0.5) where rn1, rn2 ~ U(0,1)
- Only for plane waves with E_k < 3.307 Ha (90 eV cutoff, hardcoded line 1737)
- Higher energy plane waves are zeroed
- Then S-normalized

**Finding**: CASTEP does NOT use LCAO/pseudoatomic superposition for initial guess in the standard path. It uses simple random initialization with a kinetic energy cutoff.

**Implication**: Gate 2 will use CASTEP's random initialization (method='R'), not pseudoatomic SCF. This is the actual CASTEP algorithm.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| USPP preconditioner R matrix singular | LOW | Add regularization ε·I if det(R) < 1e-12 |
| Line search fails to find downhill step | LOW | Fallback to steepest descent (γ = 0) |
| Gate 1 fails due to numerical noise | MEDIUM | Loosen tolerance to 1e-9 Ha if needed |
| Gate 2 requires > 50 steps | MEDIUM | Acceptable; report actual step count |
| Pseudoatomic guess not available | LOW | Fallback to perturbed CASTEP ψ |
| chemrust-hamiltonian API mismatch | LOW | Verified during exploration |

## Dependencies

**External crates**:
- `chemrust-hamiltonian-core` (existing) — H/S operators, pseudoatomic solver
- `faer` (existing) — small dense matrix inversion for R
- `num-complex` (existing) — Complex64 arithmetic

**Internal modules**:
- `src/eigensolver/hamiltonian.rs` (existing) — GPU H/S apply (not used in Phase-0)
- `src/eigensolver/vnl_data.rs` (existing) — VNL data structures (reference only)

## Out of Scope

**Explicitly deferred to Phase-1**:
- GPU batching (Phase-0 is CPU-only)
- Block CG (Phase-0 is serial band-by-band)
- Multi-band S-orthogonalization during CG loop (Phase-0 orthogonalizes once at convergence)
- Performance optimization (Phase-0 focuses on correctness)
- Integration with SCF loop (Phase-0 is standalone)

**Explicitly deferred to Phase-2+**:
- Adaptive locking tolerance
- Full reorthogonalization fallback
- Mixed precision (FP32 preconditioner)
- Distributed multi-GPU

## Acceptance

Phase-0 is **ACCEPTED** when:
1. Gate 1 PASS: max|ε_out - ε_in| < 1e-10 Ha
2. Gate 2 PASS: Band-0 converges within 50 steps, ‖r_0‖_S < 1e-6 Ha

If both gates pass, proceed to Phase-1 (GPU batching). If either gate fails, investigate root cause before GPU work.

