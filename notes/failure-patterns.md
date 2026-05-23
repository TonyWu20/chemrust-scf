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
