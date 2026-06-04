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

## 2026-05-24: energy-integral-d_v-unit-bug (Issue 15)
**Root cause**: Energy integrals `e_hartree` and `rho_vxc` used `d_v = Ω/N` as the
discrete sum weight, but ρ is stored in CASTEP raw units (ρ_phys × Ω). The correct
weight is `1/N` — using `Ω/N` overcounts by a factor of Ω (~22,310 Bohr³), producing
an iter-1 total energy of −32M Ha instead of −886 Ha.
**Fix**: `src/scf.rs:373` — `let d_v = 1.0 / n_grid` (was `cell.volume / n_grid`).
**Pattern**: density-unit-convention-mismatch — a ρ×Ω density used with a dV = Ω/N
weight gives Ω²/N per grid point instead of the correct Ω/N, introducing one extra
power of Ω into the integral.
**Diagnostic**: Q1 (`iter1_drift_from_castep_state_is_bounded`) catches the energy
drift at a 20 mHa gate — a 7-order-of-magnitude discriminator ratio (3.2407e7 Ha
drift vs 0.02 Ha gate).
**Prior root-cause dead ends (§13, T-prime)**: The energy bug (this issue) has
been present since the code's inception. Many prior debugging sessions attributed
SCF divergence to eigensolver rotation, D-screening, or augmentation density —
and those *are* real follow-on effects — but the first symptom (wrong iter-1
energy from the fixture-converged state) was always this bug. It was missed
because no test checked iter-1 total energy against CASTEP at any precision
until Q1 was written.

## 2026-05-24: ewald-cutoff-and-background (co-discovered with Issue 15)
**Root cause**: Two compounding bugs:
1. Real-space Ewald cutoff was hardcoded at 8 Bohr, but for α = (π/Ω)^(1/3) ≈ 0.052,
   `erfc(0.052 × 8) = 0.556`, truncating ~56% of the long-range real-space sum.
2. The G=0 background correction `−π·Q²/(2α²V)` was absent (CASTEP ewald.f90:585-587).
**Fix**: `src/energy.rs` — adaptive cutoff computed from precision `5.5/α` (≈ 106 Bohr
for Cu111+CO); added `background_correction` term matching CASTEP formula.
**Pattern**: ewald-truncation-error — a real-space cutoff that doesn't account for
the slow decay of `erfc(α·r)` at small α truncates the long-range Coulomb sum,
producing an error proportional to the missing tail. Combined with the missing
G=0 term, the total Ewald was off by 346 Ha (1193 vs 847 Ha). α-invariance of
the Ewald total is a useful cross-check: if changing α changes the total, the
cutoffs are inadequate.
**Diagnostic**: CASTEP iprint=3 energy component breakdown (hartree, xc, ewald)
exposed which component was wrong. Head-to-head component comparison in eV
showed E_H, E_xc, and rho_vxc matching exactly — only Ewald differed.

## 2026-05-24: postrr-cascade-amplification (Procrustes pin against ψ_prev)
**Root cause**: The PostRr pin (SVD-polar of M = ψ_prev^H · S · ψ_after_GS · X,
applied as R = U·V^H on X columns within near-degenerate blocks) fixes iter-1
in-block gauge rotation almost completely (1e-6 outliers → 0.94 floor), but
**amplifies** cascade across SCF iterations: iter-3 band-0 drift grew from
−11.94 Ha (PinMode::Off baseline) to −14.91 Ha under PostRr+eps_degen=0.05
(both vs CASTEP A1 = −1.055 Ha). Hypothesis: pin against the previous
iteration's ψ creates self-reinforcing feedback through V_eff — iter-t pin
aligns to iter-(t−1) which was itself pinned, so error compounds rather than
re-anchoring to ground truth.
**Resolution**: `notes/debug/debug-20260524-postrr-cascade-amplification/RESOLUTION.md`
**Implementation**: `src/eigensolver/rayleigh_ritz.rs` (PinMode enum, postrr
path, typed faer SVD); `src/scf.rs` (`RrPinConfig::from_env()` plumbing, PCI-E
budget extensions). Default `PinMode::Off` — PostRr is opt-in via
`CHEMRUST_PIN_MODE=postrr`. PreRr declared but unimplemented.
**Pattern**: relative-target-procrustes-feedback — using a moving reference
(ψ_prev) for Procrustes alignment is unstable when the reference itself is
the previous iteration's output of the same algorithm. The chain `ψ_t aligns
to ψ_{t−1}, ψ_{t−1} aligns to ψ_{t−2}, ...` drifts cumulatively rather than
converging to a fixed anchor. Candidate fix: pin against an **absolute**
reference (e.g., the initial guess or iter-1 RR output frozen for the SCF run).
**Sibling failure**: iter-1 `> 0.999` gate is unreachable due to Chebyshev
filter pollution from bands outside the 40-band window (proposal §1.3
documented this as ~6–11% span pollution). Tighter `b_low` (proposal §5
follow-up) is needed before any pin variant can hit the strict gate.

## 2026-05-24: blow-tightening-falsified (proposal §5 lever exhausted)

**Symptom investigated**: "tighten `b_low` (proposal §5) to reduce filter pollution" — the postrr-cascade-amplification post-mortem (line 145-148) recommended this as the binding constraint on the iter-1 ceiling (PostRr floor 0.945, ceiling 0.977 vs gate 0.999). The proposal §14 §1.4 attributed ~6–11% Cu-3d span pollution to filter window choice.

**Pad-sweep result (definitive)**:

| pad above max_h_eig (Ha) | b_low (Ha) | Cu-3d/13 ratio |
|---|---|---|
| -0.05 | 0.0708 | 0.893 |
| (max_veff baseline) | 0.0894 | 0.893 |
| 0.0 | 0.1208 | 0.891 |
| +0.5 | 0.6208 | 0.797 |

Tightening b_low across the relevant range produces NO improvement; lifting it well above the tracked subspace makes things WORSE. The proposal §14 §1.4 attribution is **falsified**.

**Phantom check (decisive)**: `diagnostic_selftest_castep_self_overlap_block_sums` measures `Σ_{a,b∈1..14} |⟨ψ_castep_a | S | ψ_castep_b⟩|² = 13.0000` exactly — CASTEP ψ produces the perfect block sum against itself. So the 0.893 IS real pipeline loss; it's NOT a CASTEP stored-precision floor.

**Pattern**: `wrong-mechanism-attribution`. Two independent investigation paths (the parent §14 proposal team and this debug session) both fingered b_low as the lever for the same symptom, citing the same theoretical justification. Both were wrong. The 10.7% Cu-3d block sum loss is a numerical-precision floor in the filter→GS→RR pipeline at f64, NOT a filter-window-discrimination issue. The ~120 untracked-physical bands inside the damp window claim is doubly false: (a) `b_low = max_veff = 0.089` Ha sits ABOVE all 160 tracked bands → no discrimination occurs at all, ALL bands are amplified uniformly; (b) sweeping b_low to actually discriminate (0.1208 Ha — at the highest tracked band) gives the SAME ratio. The mechanism is upstream of where everyone was looking.

**Architectural finding (load-bearing)**: `scf.rs:577` and `scf.rs:716` deliberately hardcode `eig: Option<&[f64]> = None;` per the §10 Global Woodbury fix's intent (per-band eigenvalue machinery is provably redundant when ζ ≈ 0; eigenvalue-source dependency introduces stale-label noise). This means the proposal-text claim "iter-2+ b_low = eig[last] puts ~120 untracked bands inside the damped window" is RUBBISH at the architectural level — that branch is dead code.

**Reclassified prior claims**:
- "Current `b_low = max_veff + 2.0`" (proposal §5, line 267) → **STALE** — actual value since 2026-05-23 is `max_veff` (no +2.0).
- "Iter-3 b_low = 1.95 Ha would cause cascade" → **REFUTED** — production iter-3 b_low = 0.0894 Ha (same as iter-1, eig is always None).
- "Tightening b_low at iter-2+ should reduce ~3% iter-1 residual" → **REFUTED** by sweep.

**Concrete next directions** (in order of cost-to-test):

1. **Pivot to absolute-target Procrustes (P1)** — pin against iter-1 RR output frozen for the SCF run. Whole-40-band-window granularity (not per-cluster). Reuses PostRr `PinMode` infrastructure. Medium effort. The cluster-boundary rotation that the per-band S-norm test localized is at band 14/15 within the OCCUPIED 1..93 manifold (CASTEP `perc_extra_bands=72` → 93 occupied + 67 buffer = 160), not at the buffer edge — so buffer enlargement does NOT help.
2. **Davidson eigensolver (P2)** — never builds H_sub; sidesteps cluster mixing. Heavy refactor. Reserve for if P1 insufficient.
3. **Band-by-band CG (P3)** — replaces eigensolver entirely with CASTEP's algorithm. Heaviest. Last resort.

**Do NOT pursue further**:
- Filter window tuning (this session falsified it).
- Per-cluster Procrustes pinning (cluster-boundary rotation is a 40-band-window problem).
- Buffer enlargement (CASTEP convention `perc_extra_bands=72` pins 160 tracked bands; ours matches).
- Augmentation-convention or S-application audit (S-norms confirmed = 1.0000).
- f64-precision blame (other DFT codes work fine at f64; per-band S-norm test confirms no precision floor).

**Resolution**: `notes/debug/debug-20260524-blow-tightening/RESOLUTION.md`

## 2026-06-04: ffi-fortran-1-based-index-mismatch-in-scatter-gather

**Root cause**: CASTEP passes `fft_idx` as 1-based Fortran grid indices (`pw_grid_index`,
range `1..grid_size`) through the FFI boundary with no conversion. Rust copies them raw
at `ffi.rs:418` and uploads to GPU. The scatter/gather kernels at `kernels.rs:53,68,97`
use them directly as 0-based C indices (`grid[b*grid_size + fft_idx[g]]`), causing a
+1 offset in every real-space grid access:

- `grid[0]` (DC G=0 component) is NEVER populated — every V_loc operation loses the
  spatially-averaged potential
- `grid[grid_size]` is accessed out-of-bounds (reads uninitialized GPU memory for the
  last band, next band's DC position for others)
- All local potential contributions are computed at wrong spatial positions
- V_contrib for band 0: **+0.712 Ha** instead of **−1.879 Ha** (sign flip + magnitude error)
- Band 0 eigenvalue: **+1.536 Ha** instead of **−1.055 Ha** (wrong sign)
- Wrong eigenvalues feed the preconditioner as ε-estimates → NL correction weight
  explodes → |hpsi|² after V_loc hits 10⁶→10⁶³→10⁷⁴→10⁸⁶→10¹⁰⁸ → ZHEGVD fails

**Fix**: `ffi.rs:418` — convert 1-based Fortran indices to 0-based C indices during
the host-side copy with `.map(|&i| i - 1)`. Added `debug_assert!` for range validation.

**Pattern**: `1-based-index-leak-across-ffi`. Fortran passes 1-based grid indices
through a C FFI boundary; Rust receives them as raw `c_int` values with no semantic
conversion. The bug is silent because:
1. No diagnostic cross-checked V_contrib against CASTEP's internal H·ψ decomposition
2. The standalone SCF path generates its own 0-based FFT indices via
   `pw_coords_to_fft_indices`, so the bug only manifests through the FFI path
3. The KE factor diagnostic (which passes) tests the `fft_idx_to_coord` path at
   `ffi.rs:40-49` — which DOES subtract 1 at line 41. This is a DIFFERENT code path
   from the scatter/gather kernels. Two functions consuming the same `fft_idx` raw
   array: one correct (subtracts 1 internally), one wrong (assumes 0-based).
4. The V_eff round-trip check (which passes) tests the V_eff upload path — a third
   independent code path unaffected by the scatter/gather index bug

**Lesson**: When receiving integer index arrays across an FFI boundary, NEVER assume
the indexing convention matches. Add an explicit conversion AND a range assertion.
A `debug_assert!` that min≥0 and max<grid_size catches the mismatch before any
computation.

**Lesson 2**: Two functions can consume the same `fft_idx` raw data with different
assumptions, and both can have diagnostic cross-checks that pass — because each
cross-check tests only ONE function's path. Proof that a value is correct in one
function does NOT prove it's correctly consumed in another. Audit ALL consumers
of FFI integer arrays, not just the one with the first diagnostic.

**Resolution**: `notes/debug/eigenvalue-explosion-20260604/INVESTIGATION.md` (FFI Data Boundary Hypothesis)
**Commit**: `b58d33a` (fix), `c644514` (prior eigensolver fixes)
**Duration**: ~1 day from symptom identification to root cause

## 2026-06-04: ffi-warm-start-test-is-critical-discriminator

**Context**: The two current test modes for the Rust eigensolver are:
1. **Standalone SCF test**: Loads CASTEP checkpoint data from disk (`.orbitals`,
   `.pot_fmt`, etc.) and runs the full SCF loop in pure Rust. Validates the Rust
   eigensolver/hamiltonian/density code against CASTEP data. ALL diagnostics pass
   for converged wavefunctions — eigenvalues match to ~10⁻⁸ Ha.
2. **FFI cold-start test**: CASTEP calls `chemrust_diagonalise_h` via cdylib with
   atomic-guess wavefunctions. Validates the full Fortran→Rust FFI boundary from
   zero. Eigenvalues are poor (cold-start), but tests the data transfer path.

The warm-start test fills the gap between them: CASTEP calls `chemrust_diagonalise_h`
with **converged wavefunctions** from a CASTEP checkpoint. This tests:
- The full FFI data transfer path (like cold-start)
- With known-good wavefunctions where the Rust eigensolver is proven correct (like standalone)
- Isolates ANY data corruption at the Fortran→Rust boundary as the sole variable

**Why this test is critical**: Without the warm-start test, the `1-based-index-leak`
bug above would remain invisible indefinitely:
- Standalone test: correct (uses 0-based indices from `pw_coords_to_fft_indices`)
- Cold-start test: explodes (but wrongly attributed to cold-start being "expected to
  be noisy" — the explosion masked the systematic index offset)
- Warm-start test: explodes **with known-good wavefunctions** — this discriminates
  "FFI data path" from "bad initial guess" unambiguously

The warm-start test isolated the FFI boundary as the failure site within one run.
Without it, debugging would require instrumenting both the Fortran and Rust sides
to compare intermediate values — a multi-day effort for the same conclusion.

**Pattern**: `missing-discriminator-allows-silent-ffi-corruption`. When a system has
two code paths (standalone and FFI), a test that exercises only the standalone path
cannot catch FFI-specific bugs. The warm-start test — same wavefunctions, different
entry point — creates a controlled experiment where the ONLY variable is the data
transfer mechanism. Any divergence in eigenvalues between standalone and warm-start
ISOLATES the FFI boundary as the cause.

**Setup**: `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0604_warm_start/`
- Uses CASTEP checkpoint from a converged SCF run (continuation)
- `slurm_job_Cu111_CO.sh` submits with `sbatch`
- Rust cdylib must be rebuilt with `nix develop --command "make-castep-chemrust"` before each test
- Output: `slurm_output_*.txt` — key diagnostics at lines 70-80 (initial eigenvalues, H·psi decomposition)
- Compare initial eigenvalues against standalone test: must match within ~10⁻⁸ Ha
- Compare V_contrib for band 0: must have correct sign (negative for occupied bands)
- Monitor |hpsi|² after V_loc: must stay bounded (< 1e4) for all iterations

## 2026-06-05: ffi-fortran-grid-layout-convention-mismatch (RESOLVED — FIRST SUCCESS)

**Root cause**: CASTEP uses Fortran ix-innermost grid convention (column-major,
`ix + ngx·iy + ngx·ngy·iz`). The Rust code uses iz-innermost (z-fastest,
`iz + ngz·iy + ngz·ngy·ix`, from `pw_coords_to_fft_indices`). The FFI boundary
received ix-innermost data from CASTEP but treated it as iz-innermost.
Four independent manifestations across three data channels:

1. **fft_idx 1→0 based** (`b58d33a`): `pw_grid_index` is 1-based Fortran;
   scatter/gather kernels are 0-based. grid[0] never populated, grid[grid_size] OOB.

2. **fft_idx ix→iz transpose** (`22c97a2`): CASTEP's flat index formula
   `ix + ngx·iy + ngx·ngy·iz` differs from Rust's `iz + ngz·iy + ngz·ngy·ix`.
   On non-cubic grids (ngx=54≠ngz=90 for Cu111_CO), G-vectors map to wrong
   real-space positions.

3. **V_eff x↔z transpose** (`22c97a2`): cuFFT with plan (ngx,ngy,ngz) uses
   n[rank-1]=ngz innermost. V_eff with ix-innermost layout has ngx innermost.
   Axis swap needed for V_eff×ψ multiplication to be physically correct.

4. **D-screening uses original V_eff** (`22c97a2`): The screening integral
   ∫Q·V_eff needs the physical (x,y,z) layout, not the FFT-transposed (z,y,x)
   layout. D matrices were computed with x↔z-swapped V_eff, corrupting V_NL.

**Cumulative error**: +2.591 Ha (V_contrib went from −1.879 Ha correct to
+0.712 Ha). Each individual bug contributed ~0.2–2.4 Ha of error. The compound
bug was: V_loc effectively computed at random spatial positions → averaged to
V_eff_mean (~−0.07 Ha) instead of ~−2.6 Ha at ion cores → wrong eigenvalues
→ preconditioner amplification → search direction explosion (10⁶→10⁶³→10¹⁰⁸).

**Fix**: Three changes in `src/ffi.rs`:
- V_eff transpose: `ndarray::Array3::from_shape_fn((ngz,ngy,ngx), |(iz,iy,ix)| arr_ix_fast[[ix,iy,iz]])` — matching `scf.rs:598-608`
- fft_idx transpose: decode ix-innermost → (ix,iy,iz) → re-encode iz-innermost during 1→0 conversion
- D-screening: pass original `arr_ix_fast.clone()` (not transposed `arr_iz_fast`)

**Pattern**: `ffi-grid-layout-convention-mismatch`. When receiving multi-dimensional
grid data across a Fortran→C FFI boundary, EVERY channel must be audited independently
for: (a) base convention (0 vs 1-based indices), (b) axis ordering (innermost dimension),
(c) derived quantities that need the physical grid layout, not the FFT layout.

**Critical discriminator**: The warm-start test (`/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0604_warm_start/`) isolates FFI boundary corruption from eigensolver correctness. It feeds CONVERGED CASTEP wavefunctions through the FFI boundary and asserts eigenvalues match the standalone Rust test exactly.

**Resolution**: `notes/ffi-grid-layout-resolution.md`
**Commit**: `b58d33a` (1→0), `22c97a2` (transposes + D-screening)
**Duration**: ~2 days from symptom to resolution across ~12 job submissions
**Related**: §2026-06-04 ffi-fortran-1-based-index-mismatch-in-scatter-gather (Bug 1),
§2026-05-20 cufft-dim-ordering-and-rr-transpose-layout (prior cuFFT ordering bug in Chebyshev path)

