# Open Follow-Ups

## 1. G-vector mapping in diagonalize produces wrong eigenvalues

**Symptom:** Starting from CASTEP's converged V_eff and wavefunctions (loaded from
`.check`), `diagonalize()` produces eigenvalues that are systematically wrong beyond
the first 2 bands.

**Status (2026-05-20):** Partially resolved. Multiple root causes found:

### 1a. FFT index order — FIXED (commit 5037e64)
`pw_coords_to_fft_indices` was using C-order indexing. Changed to Fortran-order
`iz + ngz*(iy + ngy*ix)` to match cuFFT convention. Also fixed per-PW kinetic
energies to index by plane-wave index rather than full grid.

### 1b. Pseudopotentials loaded as Recpot — FIXED (test fixture)
`PseudopotentialSet::from_dir` picks first alphabetical match. Cu_00_OP.recpot
sorts before Cu_00.usp → all 18 ions loaded as Recpot (no V_NL, no augmentation).
Fixed by hardcoding `{species}_00.usp` paths in test fixture.

### 1c. V_eff downsampling indexing — FIXED (scf.rs:1053)
When fine_grid == wave_grid, added early return to skip FFT roundtrip entirely.
The FFT roundtrip for non-cubic grids has a dimension-mismatch bug (chemrust#8).

### 1d. V_loc + T discrepancy of −2 Ha — RESOLVED (2026-05-20)

See `notes/debug/debug-20260520-1923/RESOLUTION.md` for details. **Two
independent bugs** found and fixed:

1. **cuFFT plan dim ordering swapped**: `plan_batched_c2c(ngz, ngy, ngx,
   ...)` declared `n[2] = ngx = 54` while our scatter index formula
   `iz + ngz*(iy + ngy*ix)` makes the buffer's innermost period `ngz =
   90`. Fix: pass `(ngx, ngy, ngz, ...)` instead. Affects
   `src/eigensolver/chebyshev.rs:844` and `src/density.rs:147`. Cubic
   grids hide the bug; non-cubic grids do not.
2. **Rayleigh–Ritz transpose layout bug**: `transpose_col_to_row` kernel
   produced col-major (n_bands, n_pw) memory while RR's gemm expected
   col-major (n_pw, n_bands), scrambling H_sub and S_sub. Fix: skip the
   transpose entirely (ColumnDistributed memory is already what RR
   expects) and replace the rotation step in `rayleigh_ritz.rs` with a
   direct `psi_new = psi_row · X` gemm.

**Post-fix eigenvalue status** (bare D0, ndeg=0, reference .pot_fmt):
- Band 1: −1.43 Ha (ref −1.06 Ha) — **was −2.57 Ha pre-fix**
- Bare D0 RMS: **1.06 Ha** — was 216 Ha pre-fix
- With screening: RMS = 0.72 Ha
- Remaining residual ~0.4 Ha for band 1 is expected from bare-D0 vs
  CASTEP's screened-D V_NL (see 1e below).

### 1e. D matrix screening — FIXED (2026-05-20)
`compute_screened_d` now handles non-cubic grids correctly via the `RealGrid`/
`RecipGrid` type system (chemrust-hamiltonian commit `8691500`). Wired into
production V_NL path in `src/eigensolver/vnl_data.rs` (commit `85685b5`).
The `permuted_axes([2,1,0])` workaround in `diagnose_d_screening_values` removed.
Tight test `screened_d_band_1_residual_improves` added (GPU, `#[ignore]`):
- band-1 residual < 0.30 Ha (bare-D0 gives 0.37 Ha)
- RMS first 10 bands < 0.85 Ha (bare-D0 gives 1.06 Ha)

## 2. Density unit convention across crates

**Symptom:** The CASTEP binary stores density as `ρ × Ω` (raw grid values,
"electrons/grid_point × number_of_grid_points" per the `.den_fmt` header). The
GPU density construction (`construct_density_gpu`) divides by cell volume to
produce Ha/Bohr³.  The chemrust-hamiltonian `solve_poisson` and `compute_pbe_xc`
both expect density in these raw binary units (no volume normalization), as
confirmed by the upstream integration tests.

**Resolution (2026-05-20):** The fixture loader (`build_scf_state`) passes the
binary density as-is (raw ρ × Ω), matching VEffBuilder's expectation.  The density
comparison test separately normalises by volume for a like-for-like comparison.
No further action needed, but the unit split between crates should be documented
in CONTEXT.md to prevent regressions.

## 3. VEffWithEnergy API migration (chemrust-hamiltonian `feat/expose-energy`)

**Symptom:** The merge of chemrust-hamiltonian `main` into `feat/expose-energy`
removed the `e_hartree` and `rho_vxc` fields from `VEffWithEnergy`.  These are
now computed explicitly in `chemrust-scf/src/scf.rs` (`BuildVEffWithEnergy::build_v_eff_with_energy_impl`)
using the upsampled density, `v_h` from the VEffBuilder, and a separate
`compute_pbe_xc` call for the XC potential.

**Status:** Resolved by the manual energy computation in `scf.rs:327-336`.
No further action needed as long as the VEffBuilder contract (`Built::assemble_with_energy`
returns `v_eff`, `e_xc`, `v_h`) remains stable.

## 4. PseudopotentialSet::from_dir prefers .recpot over .usp

**Symptom:** `from_dir` picks first alphabetical match per species. Directories
with both `.usp` and `.recpot` files always load `.recpot` (e.g. Cu_00_OP.recpot
before Cu_00.usp) → no USPP augmentation data.

**Status:** FIXED (2026-05-20). chemrust-hamiltonian commit `5e3bef3` changed
`from_dir` to take explicit `pot_files` slice (from `cell.species_pot_files`).
`tests/fixtures/cu111_co.rs` updated to use `PseudopotentialSet::from_dir` with
`bin.cell.species_pot_files` — the hardcoded `{symbol}_00.usp` workaround removed.

## 5. compute_screened_d panics on non-cubic grids

**Symptom:** `fft_forward_3d` reverses array axes (a,b,c)→(c,b,a). `compute_screened_d`
Zips V_eff_fft with Q_arr assuming same shape, but they differ for non-cubic grids.
All unit tests use cubic [4,4,4] so the bug is invisible.

**Status:** FIXED (2026-05-20). chemrust-hamiltonian commit `8691500` encodes FFT
grid layout in the type system (`RealGrid<T>` / `RecipGrid<T>`). `fft_forward_3d`
now takes `&RealGrid<f64>` and returns `RecipGrid<Complex64>` — the axis ordering
is correct by construction. `compute_screened_d` wired into production V_NL path
in `src/eigensolver/vnl_data.rs` (commit `85685b5`). The `permuted_axes([2,1,0])`
workaround in `diagnose_d_screening_values` removed.

## 6. V_eff assembly: NLCC core charge inclusion

**Symptom:** V_eff from VEffBuilder had max 27 Ha at some grid points vs reference
max 0.09 Ha. Caused by missing NLCC core charge (ρ_core) in XC evaluation.

**Status:** FIXED (2026-05-20). V_eff now matches reference RMS 0.004 Ha.

## 7. Chebyshev filter diverges in SCF loop — Lanczos b_up stuck at Gershgorin cap

**Symptom:** `fixed_point_matches_castep_energy` fails. The Chebyshev filter
works correctly on SCF iteration 1 (b_up ≈ 23 Ha, ratios ~4×/step, RR gives
band-1 = -1.03 Ha). From iteration 2 onwards, `b_up = 147 Ha` (Gershgorin cap)
every time, giving `half_width ≈ 74 Ha` and ratios ~20-28×/step. RR then
produces unphysical eigenvalues (band-1 ≈ -29 Ha, drifting negative each
iteration).

**Status (2026-05-21): ROOT CAUSE LOCALISED — see §8.** The Lanczos
estimator is *not* the bug. After instrumenting `lanczos_upper_bound` and the
call site:
- `norm0 = 245.09` healthy (no early-return).
- Iter-1 `alpha = [8.80, 7.50, 7.41, 7.36, 7.13, 6.18]`, `beta` small ⇒
  textbook well-conditioned T_k, `b_up_raw = 20.76 Ha`. RR band-1 = −1.03 Ha.
- Iter-2 `alpha[0] = 9.61` (sane) then **alpha[1] = −391 Ha** — impossible for
  a unit vector under any Hermitian H. The Lanczos kernel is correct; **H
  itself is corrupted** in iter 2.
- Iter-1 gershgorin = 116 Ha, iter-2 gershgorin = 147 Ha ⇒ `(max_veff −
  min_veff)` jumped by 31 Ha between iterations. V_eff range: iter-1 = 8.69
  Ha vs iter-2 = 39.95 Ha. iter-2 V_eff degenerates into bare V_loc (deep
  −35 Ha wells with no V_H/V_xc smoothing) because the density fed into
  iter-2's V_eff assembly is the smooth-PW-only density, missing USPP
  augmentation. See §8.

The original "next steps" list below is preserved for historical context;
items 1-3 are no longer relevant (root cause is upstream of Lanczos).

**Root cause analysis (2026-05-21):**

The Lanczos upper-bound estimator (`lanczos_upper_bound` in
`src/eigensolver/chebyshev.rs`) uses a deterministic pseudo-random starting
vector (complex exponential). After `cargo clean` and rebuild, iteration 2
still returns `b_up = 147.7446 Ha` (Gershgorin cap), identical to before the
fix. This means the Lanczos estimator is still falling back to Gershgorin on
every post-first-iteration call.

**Suspected cause:** The Lanczos estimator returns `(b_up, ritz_min, ritz_max)`
and the call site uses `b_up_raw = b_up_lanczos.min(gershgorin_b_up)`. If the
Lanczos `b_up` is already ≥ Gershgorin (e.g. due to large residual norm from
a poorly-conditioned starting vector), `b_up_raw = gershgorin`. Alternatively,
the condition `b_up > b_low` may be failing because `b_low = eig[last] = 0.13 Ha`
and the Lanczos `b_up` is somehow ≤ 0.13 Ha.

**What is known:**
- Iteration 1: Lanczos works correctly (b_up = 22.8 Ha, b_low = 7.13 Ha from
  T_k midpoint). RR produces physically correct eigenvalues (band-1 = -1.03 Ha,
  band-160 = 0.13 Ha).
- Iteration 2: b_low = eig[last] = 0.13 Ha (correct per Alg 4.1 §7.2).
  Lanczos b_up falls back to Gershgorin 147 Ha. The 160-band subspace after
  Gram-Schmidt + RR rotation is the lowest 160 eigenstates — the starting
  vector for Lanczos (random complex exponential) should still find λ_max ≈
  12-15 Ha. Why it doesn't is unresolved.

**Diagnostic prints still active** (remove before production):
- `[Chebyshev] b_up=... b_low=...` in `chebyshev_filter`
- `[RR] eigenvalues: first=... last=...` in `scf.rs:474`
- `[Chebyshev] k=N norm_prev=... norm_curr=... ratio=...` in filter loop

**Next steps for dedicated session:**
1. Add a diagnostic print inside `lanczos_upper_bound` showing the raw
   `(b_up_lanczos, ritz_min, ritz_max)` tuple before the Gershgorin cap is
   applied, to determine whether Lanczos is returning a bad value or the
   cap logic is wrong.
2. Check whether `norm0 < 1e-30` early-return is triggering (v_cur all-zeros
   despite the fix). Add an eprintln for `norm0` inside the estimator.
3. Consider whether the Gram-Schmidt step is consuming the `final_psi_buf`
   in a way that leaves it in a state where the random vector upload to
   `v_cur` is racing with a stream operation.
4. Reference: Zhou (2014) Algorithm 4.1 §7.1 — b_up from Lanczos, b_low
   from max Ritz of previous iteration. Algorithm 5.1 — first-step bootstrap
   using T_k midpoint for b_low. Paper at
   `reference_paper/zhou2014-chebyshev-filtered-subspace-iteration-jcp.zip`.

## 8. USPP augmentation density missing in `construct_density_gpu`

**Symptom:** Iter-2 V_eff range jumps from iter-1's 8.69 Ha to 39.95 Ha. The
density built by `construct_density_gpu` (`src/density.rs:93`) integrates to
~0.002 in raw `ρ × Ω` units (`sum/N`), vs the fixture's 186.0 (= N_electrons).
Equivalently, `construct_density_gpu`'s `rho_sum = 904.59` while the fixture's
`rho_sum = 8.1356e7` for the same physical electron count. Ratio ≈ 89,944 ≈
**4 · Ω** (Ω ≈ 22,310 Bohr³). The factor of 4 above the simple
convention-mismatch factor Ω is the smoking gun.

**Root cause (2026-05-21):** `accumulate_density` computes only the smooth
plane-wave term `Σ_b occ_b · |ψ_b(r)|² / Ω`. CASTEP `.check` wavefunctions
are S-orthonormal under USPP S (`⟨ψ_b|S|ψ_b⟩ = 1`), so the plane-wave norm
`Σ_G |c_{b,G}|² = ⟨ψ_b|ψ_b⟩` is *less* than 1 per band — the rest of the
charge lives in the augmentation channels. The full USPP density is

  ρ(r) = Σ_b occ_b [|ψ_b(r)|² + Σ_{I,L,L'} Q^I_{LL'}(r) ⟨ψ_b|β_{IL}⟩ ⟨β_{IL'}|ψ_b⟩]
       = smooth ρ_PW + augmentation Σ_I,L,L' Q^I_{LL'}(r) · ω^I_{LL'}

where `ω^I_{LL'} = Σ_b occ_b ⟨ψ_b|β_{IL}⟩ ⟨β_{IL'}|ψ_b⟩` is the occupancy
matrix. For Cu 3d states the augmentation contributes the bulk of the
charge — empirically, `⟨ψ|ψ⟩_avg ≈ 0.25` across occupied bands, matching the
observed factor of 4.

The fixture density (CASTEP `.den_fmt`/`.castep_bin`) already contains
augmentation, so iter-1 (which uses the fixture density) is correct. Iter-2
takes the smooth-only output of `construct_density_gpu`, which collapses V_H
(linear in ρ, so ~5 orders of magnitude smaller than expected after the
convention/augmentation mismatch) and V_xc (nonlinear, but also tiny), leaving
V_eff ≈ V_loc — bare pseudopotential wells, no Hartree/XC smoothing. Lanczos
faithfully reports the spectrum of this corrupted H (`lambda_min_tk ≈ −1170
Ha` near the deepest V_loc point) and Chebyshev amplifies it.

**Crude rescale experiment (reverted; only the diagnostic prints remain in
tree):** scaling the smooth ρ uniformly so `sum/N` matches `N_electrons`
reduces iter-2 V_eff range to 32.80 Ha and band-1 to −13.34 Ha — directionally
improved (39.95 → 32.80 Ha range, −29 → −13 Ha band-1) but still ~4× too wide
vs the correct 8.69 Ha range. Iter-3 `[CrudeScale] scale` drops from 89,937 to
76,438 — the required rescale factor is *iteration-dependent*, confirming that
no single uniform scaling can substitute for the spatially-localised
augmentation contribution.

**Diagnostic evidence in tree:** the eprintln prints in `chebyshev.rs`,
`density.rs`, and `scf.rs` (tagged `[Lanczos]`, `[Lanczos@call]`,
`[ConstructDensity]`, `[V_eff]`, `[Density]`, `[NewDensity]`, `[psi]`) are
the artifacts of this diagnosis. A reference run is preserved at
`/tmp/lanczos-diag-0115.log` (and the conversation transcript). Decision
tree from the diagnosis is recorded in `notes/plans/notes-open-followups-md-issue-after-car-snug-ripple.md`.

**Fix scope:** implement USPP augmentation density on top of
`construct_density_gpu`. Reusable infrastructure already exists:

1. **β projector matrix elements `⟨β_{IL}|ψ_b⟩`** — computed inside
   `VnlBatchData::precompute` (`src/eigensolver/vnl_data.rs`) for the
   D-matrix application path. The same projections need to be exposed for
   density construction (CPU reduction across bands, weighted by `occ_b`).

2. **`Q^I_{LL'}(r)` real-space augmentation functions** — the screened-D
   path (`chemrust-hamiltonian-core::compute_screened_d`) already FFTs
   `Q^I_{LL'}(G)` onto the fine grid for `∫Q·V_eff`. Same pipeline can be
   reused for density assembly.

3. **Augmentation accumulation kernel** — new CUDA kernel (or CPU code,
   since occupancy matrices are small per ion). For each ion I and
   channel-pair (L,L'), add `ω^I_{LL'} · Q^I_{LL'}(r)` to the smooth
   density buffer.

**Affected files for the fix:**
- `src/density.rs` — add augmentation pass after `accumulate_density`
- `src/eigensolver/vnl_data.rs` — expose β projection matrix elements
- `chemrust-hamiltonian-core` — reuse Q-FFT pipeline from screened-D code
- Tests: extend `tests/ca_scf_convergence.rs::fixed_point_matches_castep_energy`
  with a tighter assertion on iter-2 V_eff range (should match iter-1's ~9 Ha).

**Next steps for dedicated session:**
1. Read CASTEP source (`~/programming/CASTEP-GPU-port/`) for the
   reference `density.F90` augmentation accumulation — specifically
   `aug_charge_addition` or equivalent. Note the convention CASTEP uses for
   storing `ω^I_{LL'}` and the indexing of `Q^I_{LL'}(r)` on the fine grid.
2. Verify the β projections currently stored in `VnlBatchData` carry the
   raw `⟨β_{IL}|ψ_b⟩` values (not contracted into D · ⟨β|ψ⟩ for V_NL apply).
   If they're pre-contracted, the augmentation density path needs its own
   projection pass.
3. After implementing, the discriminator test is iter-2 V_eff range —
   should land within ±1 Ha of iter-1's 8.69 Ha (the fixture-based value).
   Loose acceptance: iter-2 band-1 within ±0.2 Ha of CASTEP's −1.06 Ha
   reference.

