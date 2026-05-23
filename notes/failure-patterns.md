# Failure Patterns

## 2026-05-20: spectral-bounds-not-root-cause-of-eigenvalue-errors
**Root cause**: Spectral bounds were implicated as cause of eigenvalue errors, but bounds only affect convergence rate — the filter is a polynomial in H that preserves eigenvectors. Real cause was G-vector FFT index order + kinetic energy indexing (fixed in `5037e64`).
**Fix**: `src/eigensolver/chebyshev.rs:246-283` — improved `compute_spectral_bounds` to use physically-grounded b_low on first iteration and `max_i Ritz_i` on subsequent iterations (matching CheFSI §5 step 11).
**Pattern**: wrong-attribution — symptom (wrong eigenvalues) mismatched to hypothesis (spectral bounds). The debug-outcomes divergence-surface enumeration correctly ruled out spectral bounds before implementing the fix.

## 2026-05-20: cufft-dim-ordering-and-rr-transpose-layout
**Root cause**: Two compounding bugs that partially cancelled on the Cu111_CO non-cubic grid:
  1. cuFFT plan dim ordering passed as `(ngz, ngy, ngx)` but our scatter formula `iz + ngz*(iy + ngy*ix)` makes `ngz` innermost — cuFFT expected `n[rank-1]` innermost so the plan should be `(ngx, ngy, ngz)`.
  2. `transpose_col_to_row` CUDA kernel produced col-major (n_bands, n_pw) memory while Rayleigh-Ritz's gemm expected col-major (n_pw, n_bands). Result: H_sub and S_sub matrices were scrambled, ZHEGVD diagonalized garbage.
**Fix**:
  - `src/eigensolver/chebyshev.rs:844` and `src/density.rs:147` — swap to `(ngx, ngy, ngz)`.
  - `src/eigensolver/chebyshev.rs:980` — replace transpose calls with memcpy (ColumnDistributed memory is already in the layout RR expects).
  - `src/eigensolver/rayleigh_ritz.rs:269` — replace step 4-5 with direct `psi_new = psi_row · X` gemm in col-major (n_pw, n_bands) layout.
**Pattern**: bug-cancellation. Fixing only the cuFFT plan dim made the Cu111_CO test result *worse* (band 1: -2.57 → -3.48 Ha), because the wrong cuFFT happened to feed the wrong RR transpose in a way that partially cancelled. This misled debugging until an isolated `cufft_dim_ordering_isolated_diagnostic` test (Phase G) and a brute-force `apply_h_components_for_test` that returns per-band `T`/`V_loc`/`V_NL` decompositions (Phase H) decoupled the two bugs. Lesson: when a "fix" makes things worse, suspect a second compounding bug rather than reverting.
**Diagnostic anchors**: `tests/ca_step_validation.rs` → `cufft_dim_ordering_isolated_diagnostic`, `diagonal_h_expectation_gpu_vs_cpu`, `h_sub_off_diagonal_magnitude` (with `RR_DUMP_HS=1`).
**Resolution**: see `notes/debug/debug-20260520-1923/RESOLUTION.md`.

## 2026-05-20: dont-revert-empirically-correct-fix-on-regression
**Pattern**: When a fix is independently verified correct by an isolated unit test (e.g. analytic match to 1e-15) but applying it to the full pipeline makes the end-to-end result *worse*, the right response is to **keep the fix and hunt for a second compounding bug** — not to revert. Two independent wrongs can partially cancel; reverting the correct fix restores the cancellation, hides the second bug, and traps the debug session in a loop blaming the already-correct component.
**Concrete instance (this session)**: cuFFT plan dim swap from `(ngz, ngy, ngx)` to `(ngx, ngy, ngz)` was verified correct by `cufft_dim_ordering_isolated_diagnostic` (analytic match to 1e-15, only ordering of 6 permutations to do so). Applying the swap moved Cu111_CO band 1 from −2.57 Ha (closer to ref −1.06 Ha) to −3.48 Ha (further). Looked like a regression. Was actually unmasking the Rayleigh–Ritz transpose layout bug that had been hidden because wrong-cuFFT × wrong-RR partially cancelled. User explicitly directed: *"do not let the currently worse result intimidate you. Or things might be further convoluted with mutual canceling, shadowing the real cause."* That call was load-bearing — without it the RR bug stays hidden.
**Counter-example boundary**: This applies when the fix has strong isolated evidence (analytic match, independent reference). A fix supported only by intuition or a single empirical check still warrants reconsideration on regression.
**Action when triggered**: After a "fix-but-worse" outcome, the next diagnostic should isolate components downstream of the fix (here: H_sub/S_sub dump after gemm, per-band ⟨ψ|H|ψ⟩ decomposition) — not roll back.

## 2026-05-22: spin-deg-wrong-for-erfc-occupations
**Root cause**: Added `spin_deg=2.0` to ω_{nm} accumulation in `compute_aug_density_fine` and `compute_aug_density_gpu`, matching CASTEP `ion.f90:7114`. But CASTEP's `occ ∈ [0,1]` per spin channel; our `erfc` smearing produces `occ ∈ [0,2]` with spin degeneracy already encoded. The extra factor doubled ρ_aug, breaking the correct total density (pre-fix: rho_PW + rho_aug = CASTEP total ✓; post-fix: 1.75× CASTEP total ✗).
**Fix**: Reverted `spin_deg * acc` from both CPU (`src/density.rs:425-437`) and GPU (`src/density.rs:504-518`) paths.
**Pattern**: reference-mismatch — CASTEP reference formula is correct for CASTEP's occupation convention [0,1], but inapplicable when occupations already encode spin degeneracy [0,2]. Always verify the occupation convention before applying a spin_deg factor from a reference implementation.

## 2026-05-21: range-only-acceptance-misses-pointwise-divergence
**Root cause**: Issue #8 (ρ_aug wiring) was verified by `iter2_v_eff_range_within_one_ha_of_iter1` — a range-only check (|max - min| of V_eff). Range matched between iter-1 and iter-2 (8.69 vs 8.55 Ha), but V_eff *values at ion centres* did not. ∫Q·V_eff (with Q sharply peaked at ions) is sensitive to V_eff at ion centres, not its global range. Result: D_screened on Cu ions exploded from amax ≈ 5.8 Ha (iter-1) to 49.97 Ha (iter-2) to 318.9 Ha (iter-3) — 5× the bare D_0 amax of 61.6 Ha.
**Fix**: not yet — investigation localised the cause to ρ_aug spatial distribution at ion centres or V_eff downsampling artefacts there. See `notes/debug/debug-20260521-1233-scf-diverges/RESOLUTION.md`.
**Pattern**: ANCHOR-WEAK-DISCRIMINATOR. A discriminator metric that aggregates spatial information (range, ‖·‖_∞ on a coarse summary) cannot detect localised pointwise errors. For physics tied to localised operators (β projectors, Q augmentation, V_NL), the acceptance test must compare against the reference at the LOCAL points where the operator acts — not at global summaries. Specifically: if Q is peaked at ion centres, V_eff(ion_centre) and ρ(ion_centre) must be in the criterion set.
**Diagnostic anchors**: D_screened per-ion amax tracking in `src/eigensolver/vnl_data.rs:175-184`. When D_screened amax > 2× D_0 amax, ∫Q·V_eff is corrupted regardless of what global V_eff range says.
**Lesson**: range-only and integral-only checks pass for any spatial distribution that preserves the aggregate. They are necessary but not sufficient for operators sensitive to local values.

## 2026-05-22: density-normalization-bug-misattribution
**Root cause**: Two claims of "normalization bugs" in density construction code
(rho_PW 32.6% too small, rho_aug 19% too large) were made based on comparing
our iter-2 density decomposition against CASTEP's converged-state F8 dumps.
The comparison was between different wavefunction states (different SCF
iteration), not a controlled same-input comparison.
**Fix**: A controlled experiment feeding CASTEP's converged wavefunctions
through our density code showed ratios of 1.000000 (soft) and 1.000084 (aug)
vs CASTEP F8 dumps. The density code is correct. The real issue is upstream
(RR produces different eigenvectors than CASTEP's).
**Pattern**: misattribution — comparing outputs at different SCF states is
not evidence of a normalization bug. Always validate with same-input
controlled experiment before asserting a code bug.
**Anchor**: `tests/ca_scf_convergence.rs::density_decomp_matches_castep_f8_same_inputs`

## 2026-05-23: iter1-filter-operator-mismatch
**Root cause**: Three compounding bugs masked by each other:
  1. GPU D-screening (`screen_d_gpu`) produced near-zero screening terms for non-origin ions, causing `d_screened ≈ d0_expanded` (10–50× too large). Ion at origin was immune, masking the bug.
  2. `b_low` bootstrap on iter-1 used `max_veff + 2.0 = 2.09 Ha`, placing the filter cutoff above all 160 tracked bands (highest ≈ 0.13 Ha), making the filter non-selective.
  3. Chebyshev recurrence used bare H (Mode A) while Lanczos bounds were on S⁻¹·H — filter window/operator mismatch.
**Fix**:
  - `src/eigensolver/vnl_data.rs` — reverted D-screening to CPU `compute_screened_d_from_fft`.
  - `src/eigensolver/chebyshev.rs` — b_low: `max_veff + 2.0` → `max_veff`; production filter: `FilterMode::BareH` → `FilterMode::SinvHKeepHEig`.
  - `src/scf.rs:433` — production default updated to Mode B.
**Pattern**: gpu-port-silent-correctness-regression. GPU port of a CPU function produced wrong results for all non-trivial inputs (non-origin ions) while passing for the trivial case (origin ion). The trivial case was the first ion processed, masking the bug in all diagnostic logs. Always include a non-trivial test case when porting numerical code to GPU.
**Resolution**: `notes/debug/debug-20260523-0916-iter1-filter-operator-mismatch/RESOLUTION.md`

## 2026-05-23: electron-count-diagnostic-double-volume
**Root cause**: Diagnostic formula applied cell volume to density already in CASTEP raw units (ρ×Ω)
**Fix**: src/scf.rs:486 and src/scf.rs:732 — remove extra `* cell.volume` factor
**Pattern**: unit-convention-mismatch (diagnostic)
**Lesson**: When debugging normalization errors, trace the full dataflow including diagnostics — the measurement code can be wrong even when the physics is correct. In this case, the density construction was correct all along (both fixture and computed paths use CASTEP ρ×Ω convention consistently). The bug was in the diagnostic formula that multiplied by Ω again, applying the volume factor twice and producing a 22,300× error (exactly the cell volume in Bohr³). The "raw_conv" diagnostic already gave the correct answer (186 e⁻), but "phys_conv" was wrong and prominently used, misleading the investigation.
**Resolution**: `notes/debug/debug-20260523-1149-iter2-divergence/RESOLUTION.md`

## 2026-05-23: per-band-eigenvalue-branches-redundant-when-zeta-zero
**Root cause**: R-ChFSI per-band eigenvalue machinery (Das Algorithm 3 lines 598-604) is designed for inexact S⁻¹ (ζ > 0). After §10's Global Woodbury fix, ζ = ‖D⁻¹ − B⁻¹‖ = 3.8e-15 (machine epsilon). Das et al. (2025) main.tex:612 proves that when ζ = 0, R-ChFSI ≡ standard ChFSI algebraically. The per-band machinery provides zero benefit but introduces numerical weak points: eigenvalue labels from iter-1 (properties of H[ρ₁]) are used to construct filter shifts for iter-2's different operator H[ρ₂]. When V_eff drifts between iterations, stale labels cause the filter to amplify the wrong subspace.
**Fix**: 
  - `src/scf.rs:528-541` — always pass `eigenvalues=None` to Chebyshev filter (standard ChFSI path)
  - `src/eigensolver/chebyshev.rs:1585-1592` — fix `lam_source` to use `eigenvalues.unwrap_or(&h_eig)` consistently (defensive fix, no-op when eigenvalues is always None)
**Empirical evidence**: `CHEMRUST_FORCE_NO_EIGS=1` (disabling per-band branches) reduced iter-2 last-band overshoot from 1.95 Ha → 0.70 Ha (64% improvement).
**Pattern**: algorithm-redundancy-after-upstream-fix. An algorithm feature designed for a specific regime (inexact S⁻¹) becomes redundant after an upstream fix (exact S⁻¹ via Global Woodbury). The feature should be disabled when its precondition no longer holds. Missing guard condition: reference paper implies per-band machinery should be disabled when ζ ≈ 0, but implementation unconditionally enables it when `eigenvalues.is_some()`.
**Tight tests**: `tests/ca_scf_convergence.rs::issue_11a_iter1_band0_matches_castep` (SC-1), `issue_11a_iter2_lastband_does_not_overshoot` (SC-4)
**Resolution**: `notes/debug/debug-20260523-1758/RESOLUTION.md`

## 2026-05-23: scf-gate-electron-count-diagnostic-double-volume
**Root cause**: SCF gate diagnostic at `src/scf.rs:1414` applied cell volume to density already in CASTEP raw units (ρ×Ω), producing a 15,000× error (2.9M electrons vs expected 186). This is the SAME bug as the `[NewDensity]` diagnostic fixed in commit `5a6f598`, but the SCF gate diagnostic was added in that same commit with the bug still present.
**Fix**: `src/scf.rs:1414` — change `rho_arr.iter().sum::<f64>() * cell_volume / n_grid` to `rho_arr.iter().sum::<f64>() / n_grid`
**Pattern**: diagnostic-copy-paste-error. When adding a new diagnostic that computes the same quantity as an existing one, the new diagnostic must use the same unit convention. In this case, the `[NewDensity]` diagnostic was fixed to remove `* cell_volume`, but the SCF gate diagnostic (added in the same commit) still had it.
**Lesson**: When fixing a unit-convention bug in one diagnostic, audit ALL diagnostics that compute the same quantity. A grep for the quantity name (e.g., "electron count", "total_e") would have caught this.
**After fix**: Electron count diagnostic now reports correct values (55–130 e⁻ range vs expected 186 e⁻), revealing a REAL physics bug (135% drift from iter-1 to iter-2) that was hidden by the 15,000× measurement error.

## 2026-05-23: eigenvector-rotation-cascade-divergence
**Root cause**: Gram-Schmidt + Rayleigh-Ritz subspace method produces rotated eigenvectors within degenerate Cu 3d manifolds (bands 2-10, 0.02 Ha spread), even starting from CASTEP's exact wavefunctions and V_eff at ndeg=0 (avg overlap 0.252). The Chebyshev filter adds further rotation (avg 0.108 at ndeg=8). The rotated eigenvectors produce a slightly wrong density at iter-1 (correct total 186 e⁻ but soft/aug split 29.7/70.3% vs F8 36.8/63.2%). This density cascades through the SCF: wrong V_eff → wrong D-screening → wrong S⁻¹·H → catastrophic divergence by iter-3 (band-0 = -11.94 Ha).
**Fix**: Not yet — hypothesised fix is to compare D-screening between iter-1 (CASTEP V_eff) and iter-2 (our V_eff) to identify the amplification point.
**Pattern**: subspace-vs-CG-algorithm-mismatch. Chebyshev filter + subspace Rayleigh-Ritz is a fundamentally different eigensolver than CASTEP's band-by-band conjugate gradient minimization. Subspace methods rotate eigenvectors within degenerate manifolds, which CG preserves naturally. This is expected algorithmic behaviour, not a code bug.
**Resolution**: `notes/debug/debug-20260523-1915/RESOLUTION.md`

## 2026-05-23: rr-validation-infrastructure-cfg-boundary
**Root cause**: `#[cfg(test)]` on library items (`rayleigh_ritz_with_matrices`, its import in `scf.rs`) is invisible to integration tests in `tests/` which compile as a separate crate. All test-only infrastructure that integration tests need must be gated with `#[cfg(any(test, feature = "scf_diag"))]`, not just `#[cfg(test)]`. Simultaneously, `psi_data()` existed only on the `VEffBuilt` phase impl, not on `WavefunctionsUpdated`, despite accessing the same struct field.
**Fix**: `src/scf.rs:18` — change `#[cfg(test)]` to `#[cfg(any(test, feature = "scf_diag"))]` on import; `src/scf.rs:1637` — add `psi_data()` to `WavefunctionsUpdated` impl block.
**Pattern**: cfg-boundary-mismatch. The rule: any test-only item accessed from `tests/` needs `any(test, feature = "...")`, not bare `#[cfg(test)]`. The typestate pattern means phase-specific impl blocks don't share methods even when accessing the same field.
**Resolution**: `notes/debug/debug-20260523-1927/RESOLUTION.md`

## 2026-05-23: uspp-pw-norm-not-unit-ncpp-assumption
**Root cause**: Test 6 assumed `‖ψ‖²_PW ≈ 1` for USPP wavefunctions. In USPP the constraint is `⟨ψ|S|ψ⟩ = 1` where `S = 1 + Σ_ion |β_I⟩ Q_I ⟨β_I|`. For Cu 3d states Q contributes 40–60% of total norm so `‖ψ‖²_PW` ranges 0.14–1.03. Additionally Q can be negative for shallow s/p states, allowing `‖ψ‖²_PW` slightly above 1. The `≤ 1` upper bound is a norm-conserving PP assumption that does not hold for USPP.
**Fix**: `tests/rayleigh_ritz_validation.rs` — replace `|‖ψ‖²_PW - 1| < 1e-6` assertion with physical bounds `‖ψ‖²_PW ∈ (1e-6, 2.0)` (no ghost modes, no explosion).
**Pattern**: ncpp-assumption-in-uspp-code. When porting validation logic from norm-conserving PP literature, audit every statement of the form "⟨ψ|ψ⟩ = 1" — in USPP this is `⟨ψ|S|ψ⟩ = 1`, which is a strictly weaker constraint on the bare PW norm.
**Resolution**: `notes/debug/debug-20260523-1927/RESOLUTION.md`

## 2026-05-24: stale-aug-density-cascade

**Root cause**: ρ_aug from subspace-rotated ψ leaks into next SCF iteration's V_eff via `into_phase()`. The stale ρ_aug (computed with ψ that differs from CASTEP's within degenerate 3d manifolds) contaminates the total density used for Hartree/XC. Removing ρ_aug accelerates divergence — it provides needed damping at ion cores. T3 proves the cascade stops with CASTEP V_eff. T2 reveals a secondary issue: D-screening diverges from CASTEP dump by 0.1–2 Ha per element.

**Fix**: Not yet. Two directions: fix D-screening element-by-element (reduces ψ rotation), or damp aug contribution to V_eff.

**Pattern**: stale-state-propagation. A cached intermediate from iteration N is consumed in iteration N+1 where the underlying state (ψ) has already changed. The reference implementation (CASTEP CG) avoids this because its ψ stays closer to the fixed point; our subspace RR rotates more.

**Diagnostic anchors**: T1 (RMS 0.0046 Ha), T2 (max|Δ| 2.13 Ha), T3 (PASS, cascade stopped), T4 (PASS with cleared aug).

**Resolution**: `notes/debug/debug-20260523-2314/RESOLUTION.md`

## 2026-05-24: tolerance-conflation-in-acceptance-test

**Symptom**: `fixed_point_matches_castep_energy` (`tests/ca_scf_convergence.rs:59`)
asserted `|E_total - CASTEP| < 2e-4 eV` after 8 SCF iterations from CASTEP's
converged state. The test drove days of debugging into the SCF cascade as if
it were a single-cause bug. The cascade is real (iter-3 band-0 = -11.94 Ha)
and indicates real symptoms, but the test as written cannot distinguish two
fundamentally different questions:

- algorithm-fidelity: does our eigensolver preserve CASTEP's converged ψ?
  (Answer: no — subspace-RR + Chebyshev rotates within Cu 3d degenerate
  manifolds; this is intrinsic to the algorithm, not a code bug. A bug-free
  port still fails at 2e-4 eV.)
- convergence: does our SCF reach CASTEP's `ELEC_ENERGY_TOL = 1e-5 eV` from a
  generic starting density? (Answer: blocked by F3 — eigensolver rotation
  stabilization, see open-followups §14.)

**Pattern**: `tolerance-conflation`. A single test with a single threshold
cannot probe both algorithm-fidelity (operator-preservation) and convergence
(energy-functional reaching a target) simultaneously. The thresholds for
those two questions differ by orders of magnitude in this regime, and the
acceptance criterion that conflates them defaults to the tighter of the two,
making the loose-question failure look like a code bug.

**Lesson**: When porting between eigensolvers (band-by-band CG → subspace
Rayleigh-Ritz here), the acceptance test must be formulated in
operator-invariant quantities (total energy, electron count, band-RMS in
non-degenerate manifolds) at appropriate per-question thresholds. Avoid
"do everything correctly to 2e-4 eV" tests for any pipeline whose
algorithmic fixed point is not bitwise-identical to the reference.

**Provenance**: The 2e-4 eV tolerance was chosen empirically before the SCF's
behavior was characterized — a "leave it loose, we don't know yet" placeholder
that became a load-bearing acceptance criterion.

**Resolution**: split into Q1 (`iter1_drift_from_castep_state_is_bounded`,
20 mHa per-iter drift bound, regression bar) and Q2
(`scf_converges_to_castep_energy_at_castep_tolerance`, 1e-5 eV ship gate,
expected-fail until F3).

**Discriminator anchors**: T-prime FAIL at 197 mHa
(`notes/debug/debug-20260524-tprime-d-injection/RESOLUTION.md`) — confirms
chemrust-hamiltonian is not the blocker, F3 is.

## 2026-05-24: t-prime-d-injection-discriminator

**Symptom**: chemrust-scf SCF cascade had three plausible attribution candidates:
(a) chemrust-hamiltonian D-screening accuracy, (b) chemrust-scf V_eff assembly
drift, (c) chemrust-scf eigensolver rotation. (a) and (b) form an apparent
egg-or-chicken deadlock — chemrust-hamiltonian's "post-SCF self-consistency
floor" can't tighten without running SCF, but our SCF can't run if it requires
tight D.

**Pattern**: `cross-repo-deadlock-dissolved-by-symmetric-substitution`. T3
(V_eff substitution from `.pot_fmt`) had already shown the cascade stops with
externally-correct V_eff. T-prime adds the symmetric experiment: inject CASTEP
converged D from `D_band_debug.dat` into iter-2 (`diagonalize_with_d_override`
in `src/scf.rs`, `precompute_with_d_override` in
`src/eigensolver/vnl_data.rs`).

**Result**: T-prime FAIL at 197 mHa. Cascade continues with CASTEP-injected D.
Comparison:

| Substitution | Iter-2 band-0 | \|Δ\| vs CASTEP |
|--------------|---------------|------------------|
| Natural cascade | −0.87 Ha | 184 mHa |
| T-prime (CASTEP D) | −0.86 Ha | 197 mHa |
| T3 (CASTEP V_eff) | −1.0452 Ha | 9.8 mHa |

V_eff injection stops the cascade; D injection does not. The cascade is
upstream of D, in the V_eff that iter-2 builds from iter-1's rotated density.

**Lesson**: When two repos appear deadlocked through "I need your tighter
output to validate mine," try **symmetric substitution** of each repo's
output into the other. If only one substitution stops the failure, that
side owns the cause and the deadlock is illusory. Cross-repo coupling
through "tightness" alone is a sign that the actual mechanism hasn't been
isolated yet.

**Lesson 2**: `chemrust-hamiltonian`'s 4 µHa V_eff residual against `.pot_fmt`
(`test_cu111_co_potential_residual`) and the 17.9 mHa Cu d-beta2 D-screening
drift are both real but **not the cascade's source**. The cascade comes from
chemrust-scf eigensolver rotation in degenerate manifolds, full stop.

**Resolution**: `notes/debug/debug-20260524-tprime-d-injection/RESOLUTION.md`.
chemrust-hamiltonian issue #9 left open as tracking artifact (per user
direction) but flagged as not-blocking for Q2.
