# Fix Tasks: Phase-0 Band-by-Band CG Algorithm De-risking

**Parent Review**: `notes/pr-reviews/phase-block-cg-migration/review.md`
**Date**: 2026-05-26
**Status**: Ready for implementation
**Input**: `notes/pr-reviews/phase-block-cg-migration/review.md` issues

## Declared Fixtures

Same as parent TASKS.md.

## Task Groups

### Group D: Critical Fixes (Gate 1 unblock)

**Dependencies**: None (these are fixes to existing code)
**Estimated effort**: 1-2 hours

#### TASK-D1: Add residual-based convergence guard before line search

**Kind**: `lib-tdd`

**Description**:
The line search quadratic has a singularity when ‖d‖ → 0 (r1 ∝ 1/‖d‖).
Add an early-return guard in `band_cg_minimize` that checks residual norm
before entering the line search.

**Algorithm**:
```rust
const MIN_RESIDUAL_FOR_STEP: f64 = 1e-12; // Ha — below this, state is converged

// Before line search (after computing g and d):
let r_norm = residual_bare_norm(&residual);
if r_norm < MIN_RESIDUAL_FOR_STEP * eigenvalue.abs().max(1.0) {
    return CgResult {
        psi, eigenvalue, n_steps: step,
        converged: true, residual_norm: r_norm,
        step_type: current_step_type,
    };
}
```

**Files**:
- Edit: `src/eigensolver/band_cg.rs`

**Changes**:
- Insert guard between direction computation (after §3f-g, before §3h line search)
- Threshold: `r_norm < tol * 0.01` where `tol` is the user-provided eigenvalue
  tolerance, with a floor of `1e-12 * |ε|`

**Success Criteria**:
1. Gate 1 passes: max|ε_out − ε_in| < 1e-10 Ha across all bands
2. For synthetic H=0 test, residual=0, guard fires and returns converged=true
3. For synthetic non-trivial H, guard does NOT fire (CG proceeds normally)

**Acceptance command**:
```bash
cargo test --test phase0_gate1_consistency -- --nocapture
```

#### TASK-D2: Add guard in line_search_2d_quadratic for near-zero bd

**Kind**: `lib-tdd`

**Description**:
In `line_search_2d_quadratic`, when `|bd| < 1e-30`, return early with
`(s=0, ε=a, status=-2)` instead of proceeding to divide by near-zero.

**Files**:
- Edit: `src/eigensolver/line_search.rs`

**Changes**:
- After computing `bd = b * d`, check `if bd.abs() < 1e-30 { return early }`.
  This supplements the existing `f64::MIN_POSITIVE` guard which is far too
  tight (2.2e-308 vs practical bd ≈ 1e-24 for near-converged states).

**Success Criteria**:
1. For near-converged synthetic inputs (a=−1, b=1e-12, c=1e-24, d=1e-24),
   returns status=-2 with step=0 instead of producing step=−1e12.
2. Existing unit tests continue to pass.
3. `clamping_at_15` test still triggers (‖d‖ is large enough for bd to be above threshold).

#### TASK-D3: Fix gradient orthogonalization to use actual S|g⟩

**Kind**: `lib-tdd`

**Description**:
In `band_cg_minimize` line 217, the call `s_orthogonalize_against(&g, &g, converged_bands)`
passes `g` as both `psi` and `spsi`, implicitly assuming S|g⟩ = g (i.e., S=I).
For USPP, this is incorrect: the S operator includes the augmentation S_aug = β·Q·β†.

**Fix**: Change `band_cg_minimize` signature to also accept an `apply_s` closure,
or extend `apply_hs` to return a 3-tuple `(H|v⟩, S|v⟩, S⁻¹|v⟩)`. The gradient
S-application is needed for correct S-orthogonalization against converged bands.

**Minimal fix (Phase-0)**:
Replace the closure type from `(H|v⟩, S|v⟩)` to include `apply_s_only`:
```rust
pub fn band_cg_minimize(
    psi_initial: &[Complex64],
    precond: &UsppPreconditioner,
    converged_bands: &[(Vec<Complex64>, Vec<Complex64>)],
    max_steps: usize,
    tol: f64,
    apply_hs: &impl Fn(&[Complex64]) -> (Vec<Complex64>, Vec<Complex64>),
    apply_s: &impl Fn(&[Complex64]) -> Vec<Complex64>,
) -> CgResult
```

Then at the gradient orthogonalization site (line 217):
```rust
let sg = apply_s(&g);
let (g_orth, _sg_orth) = s_orthogonalize_against(&g, &sg, converged_bands);
```

**Files**:
- Edit: `src/eigensolver/band_cg.rs`

**Changes**:
- Add `apply_s` parameter to `band_cg_minimize`
- Use `apply_s(&g)` for gradient S-application in orthogonalization
- Update all call sites (Gate 1 test, Gate 2 test, unit tests)

**Success Criteria**:
1. For S=I (synthetic test), S|g⟩ = g → same results as before
2. For non-trivial S (USPP test), orthogonalization uses correct S-inner product
3. Existing unit tests pass after adding `apply_s = |v| v.to_vec()` closure

### Group E: Gate 2 Implementation

**Dependencies**: Group D
**Estimated effort**: 3-5 hours

#### TASK-E1: Fix Gate 2 compilation errors (hallucinated APIs)

**Kind**: `lib-tdd`

**Description**:
The Gate 2 test has 9 compilation errors — all hallucinated APIs that don't exist
in chemrust-hamiltonian-core. Fix each one by checking the actual API.

**Errors to fix**:

1. `atomic_solver::augmented::AugConfig` → does not exist. Check actual
   `solve_atom` signature and construct arguments directly.

2. `RealLattice::volume()` → field access `cell.real_lattice.volume` (or check
   actual field name via `cargo doc`).

3. `LocalPotential.v_data` → tuple struct, access as `c_usp.v_loc.0`.

4. `BesselBasis::new(rcut, 400)` → takes 1 argument `(cutoff: f64)`. The
   number-of-points is determined internally. Check constructor signature
   at `chemrust-hamiltonian-core/src/atomic_solver/bessel.rs:182`.

5. `c_usp.projectors()` → field access `c_usp.projectors`, or import
   `HasAugmentationData` trait and call `c_usp.projectors()`.

6. `c_usp.q_aug()` → same pattern as #5.

7. `c_usp.q_func()` → same pattern as #5.

8. `c_usp.radial_grid()` → field access `c_usp.radial_grid`.

9. `c_usp.core_charge()` → field access `c_usp.core_charge`.

**Files**:
- Edit: `tests/phase0_gate2_convergence.rs`

**Success Criteria**:
1. `cargo test --test phase0_gate2_convergence --no-run` compiles cleanly.

#### TASK-E2: Use correct element for pseudoatomic guess (Cu 3d instead of C 2s)

**Kind**: `lib-tdd`

**Description**:
TASKS.md specifies "Call `atomic_solver::solve_pseudoatomic_scf()` for Cu 3d orbital".
The current test uses Carbon 2s. Cu has 11 valence electrons (3d¹⁰ 4s¹ in CASTEP
convention for the pseudopotential used). The pseudoatomic SCF generates radial
wavefunctions for each l-channel, and the l=2 (3d) orbital is projected onto plane
waves.

**Changes**:
1. Load Cu pseudopotential instead of C.
2. Set `occ_by_l` for Cu valence configuration (check USP file for valence charge
   and l-channels; typically l=0: 1.0, l=2: 10.0 for Cu).
3. Set `valence_charge` to Cu USP's `z_valence`.
4. Change `s_idx` from l=0 to l=2 (Cu 3d).
5. Extend `project_radial_to_pw` to handle l=2 via spherical Bessel j_2.
6. If necessary, pull the CASTEP reference eigenvalue for the Cu-3d-like band from
   the `.check` file eigenvalues, or use the hardcoded `-1.05502287`.

**Files**:
- Edit: `tests/phase0_gate2_convergence.rs`

**Success Criteria**:
1. Test generates pseudoatomic guess for Cu 3d (l=2), not C 2s (l=0).
2. Pseudoatomic SCF converges for Cu.
3. Plane-wave projection produces non-zero coefficients.

#### TASK-E3: Wire full USPP preconditioner in Gate 1 and Gate 2 tests

**Kind**: `lib-tdd`

**Description**:
Both Gate 1 and Gate 2 tests construct `UsppPreconditioner` with `beta_g_dummy = zeros`
(TPA-only). Task A1 specifies the full USPP preconditioner P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹.
Wire the actual per-ion `beta_g` and `q_exp` arrays.

**Changes in Gate 1** (`tests/phase0_gate1_consistency.rs`):
```rust
// Replace:
let beta_g_dummy = Array2::<Complex64>::zeros((nplw, 1));
let q_matrix_dummy = Array2::<Complex64>::eye(1);
let precond = UsppPreconditioner::new(beta_g_dummy, q_matrix_dummy, kinetic_g, k_cart);

// With: concatenate all ion beta_g horizontally, assemble block-diagonal Q
let all_beta_g = concatenate_beta_g(&ion_data, nplw);
let all_q = assemble_q_matrix(&ion_data);
let precond = UsppPreconditioner::new(all_beta_g, all_q, kinetic_g, k_cart);
```

**Changes in Gate 2**: Same replacement.

**Helper functions**:
```rust
/// Concatenate per-ion beta_g horizontally: (nplw, total_n_exp)
fn concatenate_beta_g(ion_data: &[IonData], nplw: usize) -> Array2<Complex64> { ... }

/// Assemble block-diagonal Q matrix from per-ion q_exp
fn assemble_q_matrix(ion_data: &[IonData]) -> Array2<Complex64> { ... }
```

**Files**:
- Edit: `tests/phase0_gate1_consistency.rs`
- Edit: `tests/phase0_gate2_convergence.rs`

**Success Criteria**:
1. USPP preconditioner R matrix is non-trivial (nonzero beta → nonzero C → R ≠ −Q).
2. Gate 1 test compiles and uses full USPP preconditioner.
3. Gate 2 test compiles and uses full USPP preconditioner.

#### TASK-E4: Add noise-to-convergence Gate 2 variant

**Kind**: `lib-tdd`

**Description**:
The definitive Gate 2 test: start from pseudoatomic Cu 3d guess, run CG with
USPP preconditioner, verify convergence to CASTEP reference within 50 steps.

The primary criterion: ‖r‖_S < 1e-6 Ha.
Secondary: |ε − ε_CASTEP_A1| < 1e-6 Ha.

**Acceptance criteria** (from TASKS.md C2):
1. Converged within 50 steps (or report actual step count if > 50).
2. ‖r_0‖_S < 1e-6 Ha.
3. |ε_0 − (−1.05502287)| < 1e-6 Ha.

**Files**:
- Edit: `tests/phase0_gate2_convergence.rs` (the existing file, fixed and completed)

**Acceptance command**:
```bash
cargo test --test phase0_gate2_convergence -- --nocapture
```

### Group F: Verification (CASTEP parity)

**Dependencies**: Group E
**Estimated effort**: 1-2 hours

#### TASK-F1: Parity test for line search against CASTEP

**Kind**: `lib-tdd`

**Description**:
Generate a synthetic test matrix of (a, b, c, d_s) tuples and verify Rust and
CASTEP produce identical step_size. Run CASTEP's `electronic_ideal_step_size` on
the same inputs (via a minimal Fortran driver or by extracting the Fortran output
from a known CASTEP log) and assert agreement.

Simpler variant: for 3 hand-computed cases where the algebra is verified (already
done in synthetic_parabola, synthetic_rational tests), add explicit verification
that the step_size sign convention matches CASTEP.

**Files**:
- Create or edit: `src/eigensolver/line_search.rs` (tests section)

**Success Criteria**:
1. For (a=0, b=1, c=1, d=1): Rust and CASTEP step_size agree to 1e-12.
2. For (a=−0.4, b=0.3, c=1.2, d=0.8): Rust and CASTEP step_size agree to 1e-12.

#### TASK-F2: Residual-drift tolerance calibration

**Kind**: `lib-tdd`

**Description**:
After Task D1, run Gate 1 and measure max eigenvalue drift. If drift > 1e-10 Ha,
the residual guard threshold may need tuning. If drift < 1e-12 Ha, the guard can
be tightened.

Document the actual drift achieved and the guard threshold used.

**Files**:
- No code changes; runtime measurement only.

**Success Criteria**:
1. Gate 1 passes: max drift < 1e-10 Ha.
2. Guard threshold documented in code comment.

## Acceptance

All fixes are **ACCEPTED** when:
1. Gate 1 PASS: max|ε_out − ε_in| < 1e-10 Ha (after residual guard)
2. Gate 2 PASS: CG converges from Cu 3d pseudoatomic guess within 50 steps,
   ‖r_0‖_S < 1e-6 Ha
3. All existing 37 unit tests continue to pass
4. `cargo test --test phase0_gate2_convergence --no-run` compiles cleanly
5. USPP preconditioner uses actual ion beta_g and Q matrices (non-dummy)
