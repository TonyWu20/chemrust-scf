# Plan — Reframe `fixed_point_matches_castep_energy` as an Algorithm-Fidelity Test

## Context

For multiple days we have debugged a "catastrophic SCF cascade" against the
acceptance test `fixed_point_matches_castep_energy` (`tests/ca_scf_convergence.rs:59`).
The test:

- Loads CASTEP's converged state (ρ from `.castep_bin`, ψ from `.check`,
  eigenvalues from `.bands`).
- Calls `run_scf_with_energy_gated(state, max_iter=8, tol=1e-8 Ha)`.
- Asserts `|computed − CASTEP| < 2e-4 eV` (≈ 7e-6 Ha).

User's hypothesis (now under test): the bug may not be in our SCF code — the
**test itself may be wrong from the start**.

### What the evidence shows

**1. The "V_eff formula discontinuity" hypothesis in REVIEW_PROMPT.md is wrong.**

CASTEP's `density.f90:1115-1148` shows the write order: compute soft (PW)
density → dump F8 soft sum → compute aug → add aug in-place to `den%charge` →
write to `.castep_bin`. So `.castep_bin` stores ρ_total = ρ_PW + ρ_aug.

Verified empirically from the fixture:

- `slurm_output_2295.txt` converged dump: `F8_RHO_SOFT_SUM ≈ 2.994e7`,
  `F8_RHO_AUG_SUM ≈ 5.142e7`, sum ≈ `8.135e7`.
- `electron_count_diagnostic_correct` (`tests/ca_scf_convergence.rs:1717`)
  asserts `rho_sum / n_grid ≈ 186 ± 5 e⁻`. With N_grid = 54×90×90 = 437,400,
  rho_sum ≈ 8.13e7 — consistent with F8_SOFT + F8_AUG.

So loading `.castep_bin` density into the `density` field and leaving
`density_aug_fine = None` is **correct for iter-1**: the upsample step
(which is identity since wave grid == fine grid at `grid_scale = fine_grid_scale = 1.5`)
yields ρ_total, and `build_v_eff_with_energy_impl` adds 0 — exactly right.

At iter-2 our code rebuilds ρ_PW (≈ 37%) from rotated ψ and ρ_aug (≈ 63%)
from rotated β·ψ; their sum is also ρ_total. **Same V_eff formula at both
iters**; no discontinuity.

**2. The real test-setup problem is the tolerance.**

`TOLERANCE_EV = 2e-4 eV` (`tests/fixtures/cu111_co.rs:29`) demands fixed-point
*stability*, not convergence. To pass, our pipeline must reproduce CASTEP's
converged ψ to the precision that energy is invariant to within 7e-6 Ha.

But our eigensolver is subspace Rayleigh–Ritz (Chebyshev-filtered), while
CASTEP uses band-by-band CG. Subspace RR rotates eigenvectors within
degenerate manifolds (Cu 3d, 9 bands within 0.02 Ha) on each call, even when
fed converged ψ. The rotation is recorded in `notes/failure-patterns.md:89-93`
(avg overlap 0.252 at ndeg=0 starting from CASTEP ψ). Subsequent SCF mixing
amplifies this rotation through ρ_aug (which depends on β·ψ phase choice
within degenerate manifolds), and the cascade observed in T-tests is the
algorithm-mismatch signature, not a code defect.

In other words: **CASTEP's converged ψ is not a fixed point of our SCF
operator.** A bug-free implementation of subspace-RR + Pulay would still
fail this test at 2e-4 eV.

### What needs to change

Replace the single "fixed point at 2e-4 eV" test with two narrower questions
whose answers are well-defined for our algorithm:

- **(Q1) Algorithm-fidelity probe**: does our SCF *stay close* to CASTEP's
  converged state for a small number of iterations? Defines a tight, narrow
  drift bound (per-iter energy drift; per-iter electron-count preservation;
  per-iter |ΔV_eff| in some norm) starting from CASTEP's ψ+ρ.
- **(Q2) Convergence probe**: does our SCF, starting from a *generic* initial
  density (e.g. CASTEP `.castep_bin` ρ as-is), converge to within a reasonable
  bound of CASTEP's total energy (e.g. 10 mHa ≈ 0.27 eV) in some iteration
  budget?

These split the conflated questions and let each one have an appropriate,
algorithm-aware tolerance.

This is a **plan-mode investigation only** — no code changes yet. The next
step after user approval is to draft the new test pair and decide where the
old test lives (delete vs. relax-and-keep-as-baseline).

## Confirmed findings (read-only investigation)

The prior debug session's `notes/debug/debug-20260523-2314/SUMMARY.md`
(written 2026-05-24) independently arrived at most of the same conclusions
through a different route. Both lines of evidence agree:

| Claim | Status | Evidence |
|-------|--------|---------|
| `.castep_bin` density = ρ_PW + ρ_aug (combined) | EXTERNAL ✓ | `density.f90:1115-1148` write order; F8_SOFT+F8_AUG = 8.135e7 matches `electron_count_diagnostic_correct`; **SUMMARY.md row 4 in "Ruled Out" table independently confirms** |
| Wave grid == fine grid for this fixture | EXTERNAL ✓ | `~/Downloads/CASTEP-6.11-nixos/grid_scale_investigation.md` § 4; both 54×90×90 at grid_scale=1.5; **SUMMARY.md row 5 confirms** |
| No V_eff formula discontinuity between iter-1 and iter-2 | EXTERNAL ✓ | `build_v_eff_with_energy_impl` adds aug iff present; with iter-1 ρ already including aug and aug=None, total = ρ_total. With iter-2 split into ρ_PW + ρ_aug, sum = ρ_total. Same input. |
| Our soft + aug code is correct for CASTEP ψ | EXTERNAL ✓ | `density_decomp_matches_castep_f8_same_inputs` ratios 1.000000 / 1.000084 |
| Subspace-RR rotates within degenerate manifolds | EXTERNAL ✓ | `failure-patterns.md:89-93` (overlap 0.252 at ndeg=0) |
| **Rotation mechanism is via β·ψ projection differences**, not just ⟨ψ\|H\|ψ⟩ | EXTERNAL ✓ | SUMMARY.md: occupations identical, soft density rotation-invariant in occupied subspace, but aug density (Σ β·ψ·Q·β·ψ) is NOT rotation-invariant for projector-pair indices. Density split 29.8/70.2 vs CASTEP F8 36.8/63.2 — 7 pp shift |
| Removing ρ_aug to "fix" the cascade makes it **worse** | EXTERNAL ✓ | SUMMARY.md "Cascade with aug removed": iter-2 band-0 = −23.3 Ha (was −0.87 Ha with stale aug). Aug provides essential damping at ion cores. |
| Test tolerance 2e-4 eV is fixed-point-stability tight | EXTERNAL ✓ | `tests/fixtures/cu111_co.rs:29` |

## Recommended approach (to be confirmed with user)

### Step 1 — Reclassify the test

Move `fixed_point_matches_castep_energy` into a deprecation note. Mark it
`#[ignore = "algorithm-fidelity overspec — see plan/RESOLUTION.md"]` rather
than deleting it (preserves the artefact and the historical bug-hunt context).

### Step 2 — Add the algorithm-fidelity test (Q1: per-iter noise floor)

New test name (proposal): `iter1_drift_from_castep_state_is_bounded`.

- Setup: `build_scf_state` (same as today).
- Action: run exactly **one** SCF iteration
  (`build_v_eff_with_energy → diagonalize → construct_density_off → mix → check`).
- Anchors (all EXTERNAL):
  - A1: total energy after iter-1 within **5 mHa** of CASTEP final.
  - A2: electron count preserved within `|N − 186| < 0.01 e⁻`.
  - A3: per-band ⟨ψ|H|ψ⟩ RMS over 160 bands within 50 mHa of `.bands`
    (already proved by T1 in `notes/debug/debug-20260523-2314/`).

**Calibration**: 5 mHa = ~5× the observed iter-1 drift of 0.0098 Ha
(REVIEW_PROMPT.md T3) wait — that's *backwards*. Observed 9.8 mHa exceeds
proposed 5 mHa threshold, so the test as written would FAIL even on what we
currently consider the most-correct configuration. Two valid responses:

- **(2a) Calibrate the threshold to the observed value × 2**: set threshold
  at 20 mHa, document that "iter-1 drift up to 20 mHa is the current noise
  floor of our subspace-RR + Chebyshev pipeline" — this is the *honest*
  floor. The test then becomes a regression bar, not a quality bar.
- **(2b) Set threshold at 5 mHa as a goal**: test fails today, documents the
  gap between current state and "drop-in for CASTEP" requirements. Test
  serves as a clear improvement target, not a regression bar.

**(2a)** is the right ODD framing — the test must distinguish "correct
implementation passes" from "buggy implementation fails" by ≥ 2× margin. The
**discriminator** is current-pipeline (9.8 mHa) vs broken-pipeline (the iter-3
band-0 = -11.94 Ha cascade), which is a 4-order-of-magnitude gap, so 20 mHa
is a strong threshold with massive margin.

This becomes Step 2 in the new plan.

### Step 3 — Add the convergence test (Q2: drop-in fidelity gate)

New test name (proposal): `scf_converges_to_castep_energy_at_castep_tolerance`.

- **Motivation**: This crate is intended as a drop-in GPU-accelerated SCF
  solver for CASTEP, exposed via C bindings. The tolerance must match
  CASTEP's own `ELEC_ENERGY_TOL = 1e-5 eV` (`Cu111_CO.param:39`), because
  that is the tolerance downstream CASTEP code (forces, stress, properties)
  expects of any SCF backend.
- Setup: same `build_scf_state`.
- Action: `run_scf_with_energy_gated(state, max_iter=64, tol=1e-7 Ha)`
  (Ha tol slightly tighter than the eV tolerance to leave headroom for
  unit-conversion roundoff).
- Anchor (EXTERNAL):
  - A4: `|computed_total_energy − CASTEP_REFERENCE| < 1e-5 eV`
    where CASTEP_REFERENCE = -24110.96665069 eV (`Cu111_CO.castep`).
- Mark `#[ignore]` until passing — this test is **the actual ship gate**
  and is expected to fail today; that's correct ODD process, not a problem.

**Important subtlety — partial rotation invariance**: subspace-RR rotates ψ
within degenerate manifolds (`failure-patterns.md:90`), and the soft-density
contribution to total energy *is* stationary with respect to such rotations
(|ψ|² is rotation-invariant in an equally-occupied manifold). **However, the
augmentation contribution is NOT rotation-invariant** because it integrates
β·ψ outer products with explicit projector indices (`SUMMARY.md` evidence:
identical occupations + rotation-invariant soft density, yet density split
29.8/70.2 vs CASTEP 36.8/63.2 — a 7 pp shift attributable solely to the
β·ψ projection sensitivity to high-frequency content of ψ inside ion cores).

So Q2 at 1e-5 eV is *theoretically achievable* even with subspace-RR, but
only if:

- the SCF reaches a self-consistent ρ where rotation no longer changes
  β·ψ projections (i.e., ψ is at our algorithm's fixed point, which is
  not necessarily CASTEP's fixed point);
- mixing damps transients faster than rotation-induced ρ_aug noise
  accumulates between iterations;
- D-screening (issue #9) is fixed so V_NL no longer rotates ψ further on
  each filter application.

This is *not* the same as the original `fixed_point_matches_castep_energy`,
which required *pointwise ψ stability*. The new test requires only
*self-consistent total-energy convergence*, which is what CASTEP's downstream
code actually cares about — but it is meaningfully harder than "energy is
degeneracy-invariant" would suggest, because aug density breaks that
invariance.

### Step 4 — File D-screening discrepancy as an issue against `chemrust-hamiltonian`

**STATUS: filed, twice-commented, kept open as tracking artifact (per
user direction 2026-05-24). Body claims about Cu d-beta2 are superseded
by Step 5 below — V_eff residual is 4 µHa empirically, and chemrust-hamiltonian's
own diagnosis (commit `66e661d`) shows D-screening is at the post-SCF
self-consistency floor, not actionable on hamiltonian's side without
running SCF.**

**Issue URL: https://github.com/TonyWu20/chemrust-hamiltonian/issues/9**
- Initial filing comment: `#issuecomment-4526632424`
- Empirical-V_eff update comment: `#issuecomment-4526666938`

The T2 finding (`d_screened_matches_castep_dump_on_castep_veff` FAILs at
max|Δ| = 2.13 Ha vs threshold 5e-4 Ha) implicates code in the **separate**
workspace `~/programming/chemrust-hamiltonian/`, specifically
`chemrust-hamiltonian-core/src/nlpot.rs:368` (`compute_screened_d`) and
`:417` (`compute_screened_d_from_fft`).

**UPDATE (2026-05-24, after filing)**: a new independent test in
`chemrust-hamiltonian-core/tests/d_screening_vs_castep_dump.rs:216`
(`test_cu111co_d_screening_vs_castep_band_dump`, commit `0d5dcbf`) was
written to verify D-screening. It reports:

| Species | Max\|diff\| (Ha) |
|---------|------------------|
| C (s+p) | 3.4e-3 |
| O (s+p) | 1.11e-2 |
| Cu s/p | < 1e-4 |
| Cu d-beta1 | < 3e-3 |
| **Cu d-beta2** | **1-2e-2 (worst 1.79e-2 at D[13][13])** |

This is **two orders of magnitude tighter** than T2's 2.13 Ha. The
methodological difference: the new test builds V_eff via
`VEffBuilder::<NonSpin>::new(cell, &pots, &gvg).with_density(.castep_bin density, None).assemble()`
(line 250-253), while T2 fed `.pot_fmt` V_eff directly into
`compute_screened_d`.

**Implication**: the 2.13 Ha gap in T2 was *not* purely a D-screening defect.
It included drift between **our V_eff[ρ_castep_total]** and **CASTEP's
`.pot_fmt` V_eff**, propagated through D-screening. The actual D-screening
formula error is ≤ 17.9 mHa, concentrated on Cu d-beta2 channels (high-energy
scattering states sensitive to small V_eff differences).

**What this means for the issue**:
- Issue #9's body claims "0.1–2 Ha per element" — this is incorrect for
  D-screening per se. The body should be updated to reflect 17.9 mHa global
  max, with the audit checklist refocused on the **Cu d-beta2 channel
  specifically** rather than a general projector-ordering investigation.
- Audit checklist items 1 (D_0 source) and 4 (projector ordering) are
  largely satisfied by the new test's pattern: if projector ordering were
  wrong, errors would distribute across all channels, not concentrate on
  d-beta2. The remaining suspects are:
  - **(F1a)** Q-on-grid sampling for d-beta2 specifically — high-energy
    radial functions may exceed accurate range of our radial-grid
    interpolation.
  - **(F1b)** V_eff high-G-vector representation — d-beta2 integrals
    sample G-vectors near the cutoff, where our V_eff may differ from
    CASTEP's converged V_eff representation.

This also surfaces a **second, larger error source** that was hiding
behind the original 2.13 Ha T2 number: **our V_eff[ρ_castep_total]
differs from CASTEP's `.pot_fmt` V_eff** by enough to add ~2 Ha of error
when fed into D-screening. This is a chemrust-scf-side issue (V_eff
assembly path), not a chemrust-hamiltonian one. **Recommendation: file
a separate issue or add this as a new "Step 5" investigation in this
plan.**

Rather than entangling the D-screening fix with the test-reframing in this
plan (which is purely a `chemrust-scf` test-and-tolerance change), file
the D-screening discrepancy as a GitHub issue against
`TonyWu20/chemrust-hamiltonian` (verified accessible via `gh`).

**Issue scope** (to be drafted as part of this step):

- Title: `D-screening drifts from CASTEP dump by up to 2.13 Ha — need T2 setup audit + fix`
- Symptom: `compute_screened_d`'s output (with CASTEP's converged V_eff from
  `.pot_fmt`) diverges from CASTEP's `D_band_debug.dat` by 0.1–2 Ha per
  element, up to 22% on diagonal elements. Reference test:
  `chemrust-scf/tests/ca_scf_convergence.rs::d_screened_matches_castep_dump_on_castep_veff`.
- Evidence anchors (EXTERNAL):
  - `D_band_debug.dat` (576 blocks = 32 SCF iters × 18 ions, ES24.16 format
    from CASTEP `nlpot.f90:531-544`)
  - `Cu111_CO.pot_fmt` (converged V_eff on fine grid)
  - `compute_screened_d` formula must reproduce CASTEP `nlpot.f90:531-544`
    write-out value element-by-element to ≤ 5e-4 Ha
- **Pre-fix audit checklist** (this is the test-setup-skepticism Path B from
  my prior analysis — *must precede* implementation work):
  1. Verify `D_0` source matches CASTEP `ps_D0` element-by-element before
     any screening is added.
  2. Verify V_eff grid alignment between our `compute_screened_d`'s G-vector
     sampling and CASTEP's `nlpot_calculate_d`'s G-vector sampling.
  3. Verify SCF-iteration alignment: which block of `D_band_debug.dat`
     corresponds to the V_eff in `.pot_fmt`? (Off-by-one is a known hazard.)
  4. Verify projector ordering: CASTEP's `expanded_projector_lm` analogue
     (likely `nlpot.f90`) vs `chemrust-hamiltonian-core/src/nlpot.rs`
     (l/n/m ordering convention).
  5. Reformulate T2 to decompose error pattern by (ion, projector-pair)
     so the *shape* of the discrepancy self-classifies the bug type:
     - Diagonal-heavy → projector ordering permutation
     - Uniform scale → normalization
     - Specific m-pairs → m-ordering
     - Ion-position-dependent → structure factor / fractional-Cartesian
       conversion
- Cross-link: this issue should reference both
  `chemrust-scf/notes/debug/debug-20260523-2314/RESOLUTION.md` (T2 evidence)
  and `chemrust-scf/notes/failure-patterns.md` (pattern of misattribution
  in this codebase family).

**Why file it as an issue rather than fix it here**:
- D-screening lives in a separate crate/repo, separate Cargo workspace.
- The fix surface is bounded but the audit surface is larger; tracking that
  as an issue with subtasks is cleaner than carrying it inside a `chemrust-scf`
  test-refactoring plan.
- Q1 and Q2 from this plan should *not* be blocked on D-screening — they
  can land now and serve as the regression bar; Q2's failure becomes the
  forcing function for the D-screening issue's priority.

**Action in this plan**: run `gh issue create --repo TonyWu20/chemrust-hamiltonian`
with the body above, then add the resulting issue URL to
`notes/debug/debug-20260524-XXXX/RESOLUTION.md` and to the `Q2 blockers`
section of the new `failure-patterns.md` entry.

### Step 5 — Investigate eigensolver ψ rotation surface (revised 2026-05-24)

**Originally framed as "V_eff assembly drift investigation". That hypothesis
was refuted empirically.**

Running `cargo test --release --test integration test_cu111_co_potential_residual -- --nocapture`
in `chemrust-hamiltonian` reports:

```
VEffBuilder residual: max=3.973265e-6 Ha  mean=1.009731e-9 Ha
```

V_eff reconstruction against `.pot_fmt` is at **4 µHa max** — six orders of
magnitude better than my extrapolated 2 Ha figure. The original Step 5
hypothesis is dead.

**chemrust-hamiltonian's own diagnosis** (commit `66e661d`,
`notes/failure-patterns.md:902-922`) independently arrives at:

- V_eff residual: 4.44 mHa interstitial, 1 µHa atom-site — **not a blocker**.
- CASTEP D injection: changes V_NL by ≤ 1.2 µHa for 160 bands at converged ψ
  — Cu D-screening is silent at converged state.
- Band residual 1.27e-2 Ha L1 is at **post-SCF self-consistency floor** —
  cannot improve without running SCF (which is chemrust-scf's job).

**Egg-or-chicken problem stated explicitly**: chemrust-hamiltonian's
D-screening can't tighten without true SCF; chemrust-scf's true SCF can't
converge if it requires tight D. The deadlock resolves once you observe
that at converged ψ, D errors are silent (the µHa V_NL change above).
The cascade in our SCF is therefore **eigensolver-rotation-driven, not
D-driven**: subspace methods rotate ψ within degenerate Cu 3d manifolds,
and β·ψ projections (which feed D-screening accumulation and ρ_aug) are
sensitive to that rotation in a way that occupied-subspace ⟨ψ\|H\|ψ⟩ is not.

**Rotation surface in `src/eigensolver/`** (read-only audit, 2026-05-24):

1. **Chebyshev filter** (`chebyshev.rs:1273-1700`): polynomial p(H) acting
   on ψ. Eigenvector-preserving in exact arithmetic. Numerical noise in
   the Chebyshev recurrence can introduce rotation but at machine ε scale.
   **Not the dominant rotation source.**
2. **Classical Gram-Schmidt** (`chebyshev.rs:1708-1800`, two passes):
   column-order-dependent. Within a 9-fold degenerate manifold, GS picks
   a basis specific to the column ordering and to the post-filter starting
   basis. **Real but partial rotation source.**
3. **Rayleigh-Ritz / ZHEGVD** (`rayleigh_ritz.rs:56-128`): for degenerate
   eigenvalues, LAPACK's eigenvector choice within each degenerate block
   is implementation-defined. **Most likely dominant rotation source** when
   input ψ projects onto degenerate manifolds.

None of these are bugs. All produce mathematically correct outputs of their
respective algorithms. But none preserves CASTEP's specific choice of basis
within degenerate manifolds.

**Revised Step 5 work** (read-only investigation then write a discriminator):

1. **Discriminator T-prime (D injection from `D_band_debug.dat`)**: extend
   T3 (V_eff substitution) symmetrically — inject CASTEP's converged D
   matrices into our SCF before iter-2's diagonalization. The result
   discriminates two interpretations:
   - **T-prime PASSES**: cascade stops with CASTEP D. Confirms D-screening
     is the dominant blocker post-rotation. Egg-or-chicken deadlock is real
     and the right move is to file a chemrust-hamiltonian issue for
     **tight-coupled iterative D refinement** (the only way out of the
     deadlock).
   - **T-prime FAILS**: cascade continues with CASTEP D. Confirms eigensolver
     rotation drives the cascade independently of D quality. Deadlock
     dissolves. Fix is on chemrust-scf side: stabilize GS+RR against
     degenerate-manifold rotation.

   T-prime is the highest-information experiment available. It is a new
   test in `tests/ca_scf_convergence.rs`; ≈ 50 lines using the existing
   `set_d_screened` mutator (or `set_v_eff` analogue if no D mutator exists
   — need to check).

2. **If T-prime FAILS**, investigate eigensolver stabilization techniques:
   - **(F3a) Lock occupied states**: project out converged states before
     GS+RR; only iterate on the not-yet-converged subspace.
   - **(F3b) Pin reference basis**: modified Gram-Schmidt with the previous
     iteration's ψ as the seed; only newly-introduced vectors are
     orthogonalized against it.
   - **(F3c) Davidson-style block update**: feed only the residual
     component of ψ to GS+RR; the bulk of ψ stays put.

   Each is a substantial eigensolver change; pick the lightest one that
   stabilizes rotation. None require chemrust-hamiltonian changes.

3. **If T-prime PASSES**, file a chemrust-hamiltonian issue (or reopen #9)
   for in-SCF iterative D refinement. This is a fundamentally different
   ask than what #9 currently requests — it requires the hamiltonian to
   expose a per-iteration D-screening API that takes the current SCF
   iteration's V_eff (which it can), rather than a one-shot post-SCF
   validation.

**Acceptance criterion**: T-prime PASS/FAIL determined empirically, with
per-iteration band-0 dump. The result determines whether F3 (chemrust-scf
side) or in-SCF D refinement (chemrust-hamiltonian side) is the lever.

### Step 6 — Document the algorithm-mismatch reality

Append a new entry to `notes/failure-patterns.md`:

```
## 2026-05-24: test-tolerance-empirically-loose-not-algorithm-bound
**Symptom**: `fixed_point_matches_castep_energy` set at 2e-4 eV tolerance was
treated as a fixed-point-stability test, driving days of debugging into an
SCF cascade. The cascade is real (iter-3 band-0 = -11.94 Ha) and indicates
real bugs (D-screening element-by-element drift up to 2.13 Ha, stale ρ_aug
propagation). But the test as written can never distinguish "drop-in
CASTEP-replacement quality" from "in-basin convergence" because the tolerance
was chosen empirically before the SCF's behavior was characterized.
**Pattern**: tolerance-conflation. A single test with a single threshold
cannot probe both algorithm-fidelity (does the eigensolver preserve CASTEP's
fixed point?) and convergence (does the SCF reach 1e-5 eV from a generic
start?). These are different questions with different threshold requirements.
**Resolution**: split into per-iter drift test (Q1) calibrated to current
noise floor, and total-energy convergence test (Q2) calibrated to CASTEP's
ELEC_ENERGY_TOL = 1e-5 eV (the drop-in fidelity requirement).
```

## Critical files

| Path | Role | Action |
|------|------|--------|
| `tests/ca_scf_convergence.rs:59-92` | The over-specified test | Mark `#[ignore]` with deprecation note |
| `tests/ca_scf_convergence.rs` (new) | Add Q1 and Q2 tests | Add ≈ 80 lines |
| `tests/fixtures/cu111_co.rs:26-29` | `REFERENCE_ENERGY_EV` and `TOLERANCE_EV` constants | Add `DRIFT_TOLERANCE_HA = 2e-2` (iter-1 noise floor) and `CASTEP_TOLERANCE_EV = 1e-5` (drop-in gate matching `Cu111_CO.param` `ELEC_ENERGY_TOL`) constants. Keep the old `TOLERANCE_EV = 2e-4` as `LEGACY_TOLERANCE_EV` with a deprecation comment. |
| `notes/failure-patterns.md` | Track patterns | Append `tolerance-conflation` entry |
| `notes/debug/debug-20260524-XXXX/RESOLUTION.md` | This investigation's resolution | Write after user approval; include link to the chemrust-hamiltonian issue URL |
| GitHub: `TonyWu20/chemrust-hamiltonian` | D-screening discrepancy | File issue per Step 4; record URL in RESOLUTION.md |

## Existing helpers to reuse (no new code paths needed)

- `fixtures::cu111_co::build_scf_state` (`tests/fixtures/cu111_co.rs:140`)
- `chemrust_scf::run_scf_with_energy_gated` (`src/scf.rs:1348`)
- The `build_v_eff_with_energy → diagonalize → construct_density_off →
  mix → check` chain used in `iter2_v_eff_range_within_one_ha_of_iter1`
  (`tests/ca_scf_convergence.rs:174-234`) — already exists and works.
- Per-band ⟨ψ|H|ψ⟩ computation already proven by T1 in
  `tests/eigenvalue_residual_validation.rs` (untracked, present per
  `git status`).
- `electron_count_diagnostic_correct` (`tests/ca_scf_convergence.rs:1717`)
  for the N_e preservation anchor.

## Verification plan

After user approval and implementation:

1. Run `cargo check --workspace --release` — must compile.
2. Run the new Q1 test
   (`cargo test --release -- --ignored iter1_drift_from_castep_state_is_bounded`)
   — expect PASS at 20 mHa threshold (calibrated to current ~9.8 mHa noise
   floor with 2× margin per ODD discriminator rule).
3. Run the new Q2 test
   (`cargo test --release -- --ignored scf_converges_to_castep_energy_at_castep_tolerance`)
   — expected to FAIL today; documents the drop-in-fidelity gap. Record
   the actual final-energy delta in the failure message so the gap can be
   tracked across fix iterations.
4. File the `chemrust-hamiltonian` D-screening issue via `gh issue create`;
   confirm the issue URL is returned and recorded in
   `notes/debug/debug-20260524-XXXX/RESOLUTION.md`.
5. Confirm the old `fixed_point_matches_castep_energy` still runs (with
   `#[ignore]`) and that its failure is documented as tolerance-conflation
   in `notes/failure-patterns.md`.

## Open questions for the user

All four directional questions answered (2026-05-24):

1. **Retention policy**: keep the old test as `#[ignore]` deprecation
   artefact. ✓
2. **Iteration count for Q1**: exactly 1 SCF iteration (cleanest signal). ✓
3. **Q2 tolerance**: match CASTEP's `ELEC_ENERGY_TOL = 1e-5 eV` because
   this crate is intended as a drop-in GPU-accelerated SCF for CASTEP via
   C bindings. ✓
4. **Provenance of 2e-4 eV**: chosen empirically before SCF behavior was
   characterized — i.e., a "we don't know yet, leave it loose" placeholder.
   This means there's no design intent we'd be violating by replacing it. ✓

## Implications of (3) — drop-in fidelity mode

This is the largest single constraint on the rest of the work. Specifically:

- **Q2 at 1e-5 eV is the actual ship gate.** Until Q2 passes, the crate is
  not drop-in-replacement-ready. Every other test is a stepping stone or
  a regression bar.

- **Each remaining bug from the prior session must reduce to "does it block
  Q2?".** Per `notes/debug/debug-20260523-2314/SUMMARY.md` "Fix Directions":
  - **(F1) D-screening element-by-element drift** (T2 max|Δ| = 2.13 Ha):
    **blocks Q2**. Filed as
    [chemrust-hamiltonian#9](https://github.com/TonyWu20/chemrust-hamiltonian/issues/9).
    Per SUMMARY.md, this is the **primary** lever — D errors in V_NL =
    β·D·β† directly affect which linear combinations the filter selects
    within the degenerate manifold.
  - **(F2) ρ_aug damping in V_eff** (mixing weight < 1.0 on aug
    contribution): **may block Q2** depending on whether F1 alone closes
    the gap. SUMMARY.md notes the trade-off — damping aug accuracy for
    SCF stability. Not yet implemented; keep as fallback.
  - **(F3) Filter ψ stabilization** (initialize Chebyshev with a fixed
    reference ψ, or filter the *change* in ψ rather than ψ itself):
    **may block Q2**. SUMMARY.md observes the Chebyshev recurrence is
    nonlinear via Gram-Schmidt + RR, so even with bug-free V_eff and D
    a rotated input ψ produces a rotated output. Not yet investigated
    in detail; keep as second fallback.

- **Subspace-RR rotation within degenerate manifolds is partially
  refinable, not eliminable.** The rotation mechanism (per SUMMARY.md)
  is: β·ψ projections are sensitive to high-frequency content of ψ
  inside ion cores, which is *not* invariant under unitary rotation in
  the occupied subspace even when soft density is. The 7 pp density-split
  shift (29.8/70.2 vs CASTEP 36.8/63.2) is the algorithmic signature.
  After F1 fixes V_NL, the rotation magnitude may shrink to negligible;
  if not, F2/F3 are needed.

- **Q1 at 20 mHa drift is a regression bar, not a quality bar.** Once Q2 is
  passing, Q1's threshold will tighten naturally (post-fix iter-1 drift will
  be much smaller than today's 9.8 mHa, and the threshold should ratchet
  down to track it).

## Risk in this plan

The only meaningful risk is in Step 2's **(2a) vs (2b) calibration choice**.
If we choose (2a) — set Q1 threshold at 20 mHa to match current noise floor
— we lose the ability for Q1 to *fail* and signal a regression on a fix that
should have improved iter-1 drift. To mitigate: each iteration of fixes
should ratchet the threshold down (e.g., post-D-screening-fix, drop to 5 mHa;
post-aug-handling-fix, drop to 1 mHa). The threshold becomes a living
specification, not a static one.

If we choose (2b) — set Q1 at 5 mHa as a goal — Q1 fails today and we lose
the regression-bar property until it first passes. That's also acceptable
ODD methodology (the test documents the gap between current state and goal),
just less useful as a quick canary.

**Recommendation**: (2a) with explicit ratchet schedule documented in
`notes/failure-patterns.md` so the threshold is updated whenever a fix
changes the noise floor.
