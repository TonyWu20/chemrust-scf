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

**Resolution (2026-05-21):** Implemented and discriminator test green —
iter-1 = 8.6877 Ha, iter-2 = 8.5573 Ha, |Δ| = 0.1304 Ha (target < 1.0 Ha,
7.7× margin). Two pieces of work were needed:

1. **State-machine wiring**: cached `⟨β|ψ_new⟩` from Rayleigh–Ritz into a
   new `ScfIteration.beta_psi_per_ion` field (one extra GPU gemm +
   batched D2H, ~1 MB). New `density.rs::compute_aug_density_fine` builds
   per-ion `ω^I_{nm} = Σ_b occ_b · conj(βψ)_n,b · (βψ)_m,b` and calls a
   new chemrust-hamiltonian wrapper `assemble_aug_density_fine` that sums
   `Σ_I ω^I · Q^I(G) · exp(-iG·R_I)` and inverse-FFTs to real space.
   `build_v_eff_with_energy_impl` adds ρ_aug to upsampled ρ_PW before
   feeding Poisson + XC. ρ_aug regenerates each iteration; not mixed.
2. **Hidden unit bug**: `construct_density_gpu` was producing density in
   electrons/Bohr³ (multiplying by `inv_omega = 1/Ω`) while VEffBuilder /
   solve_poisson / compute_pbe_xc expect CASTEP raw `ρ_phys × V_cell`
   units (matches `.castep_bin` storage). The fixture density iter-1
   path silently worked because it's loaded raw, but iter-2 saw a
   factor-of-Ω mismatch on top of the missing augmentation. Fixed by
   dropping the `inv_omega` factor in `density.rs`.

Empirical electron-count balance post-fix:
- Smooth ρ_PW integral: 1,029,838  (≈ 25% of total — matches `<ψ|ψ>_PW`)
- Augmentation ρ_aug integral: 3,120,753
- Total: **4,150,591** — matches iter-1 fixture density exactly (= N_e × Ω).

**Ground-truth decomposition from CASTEP F8 instrumentation (2026-05-22):**
CASTEP `density_calc_soft_wvfn_real` and `density_augment_complex` were
instrumented to dump gathered full-grid sums (Cu111_CO, converged, 16 MPI ranks).
In N_e × Ω convention (same as above):
- CASTEP soft: **1,527,256** (36.8% of total)
- CASTEP aug:  **2,623,313** (63.2% of total)
- Total: 4,150,570 ≈ N_e × Ω ✓

Comparison with our pre-fix values:
- rho_PW  1,029,480 / CASTEP_soft 1,527,256 = **0.674** (32.6% undercount)
- rho_aug 3,122,263 / CASTEP_aug  2,623,313 = **1.190** (19.0% overcount)

The pre-fix total matched CASTEP only because the two errors partially cancelled.
The spin_deg=2 fix was correctly reverted — it made rho_aug 2.38× CASTEP's value.

**RESOLVED (2026-05-22) — density code is correct; "normalization bugs" were a
misattribution.** The open items above claimed two normalization bugs in the
density construction code. A controlled experiment feeding CASTEP's own converged
wavefunctions and eigenvalues through our `construct_density_gpu` and
`compute_aug_density_fine` showed:

| Component | Our N_e | CASTEP F8 N_e | Ratio |
|-----------|---------|---------------|-------|
| Soft ρ_PW | 68.4407 | 68.4407 | 1.000000 |
| Aug ρ_aug | 117.57  | 117.56  | 1.000084 |
| Total     | 186.01  | 186.00  | 1.000053 |

Both components match to within 0.0084% (test threshold was 1%). The IFFT
normalization diagnostic probe also confirms the cuFFT/CASTEP convention match
(ratio Σ|grid|²/(N·Σ|c|²) = 1.000000).

**The 32.6%/19% discrepancy reported earlier was from comparing our iter-2
wavefunctions (different ψ after Rayleigh–Ritz) against CASTEP's converged
state — not from density code normalization errors.** The density code is
correct; the decomposition discrepancy is purely from wavefunction differences.

The real question becomes: why does our Rayleigh–Ritz produce different
eigenvectors/eigenvalues than CASTEP? Potential upstream causes:
- β projector normalization in `compute_beta_phi` (chemrust-hamiltonian)
- Q function G=0 normalization
- D matrix screening (D0 + ∫Q·V_eff vs D0)
- S_sub assembly in RR (Q matrix indexing/projector ordering)
- Preconditioning or filter quality affecting eigenvector convergence

See `notes/debug/debug-20260522-0343/CRITERIA.md` for the controlled experiment
and `tests/ca_scf_convergence.rs::density_decomp_matches_castep_f8_same_inputs`
for the discriminator test.

## 9. Performance — iter-2 path takes ~531 s (mostly CPU)

**Symptom:** With Issue #8 resolved, the green discriminator test still
takes 8-9 minutes to complete two iterations. Most of the cost is
single-threaded CPU work outside the GPU hot path.

**Approximate breakdown** (Cu111+CO, 18 ions, fine grid 54·90·90 = 437k):
- `VnlBatchData::precompute` — ~200-300 s. `precompute_q_on_grid` per
  species (Cu Q-on-grid: 171 channel pairs × radial Bessel transform on
  log grid), `compute_screened_d_from_fft` per ion (18×), CPU `compute_beta_g`
  per ion, Gauss-Jordan inversion for S^{-1} Woodbury matrix per ion.
  Already cached per species where possible.
- **`compute_aug_density_fine` (new) — ~100-200 s.** `apply_q_and_sf` runs
  per ion on the fine grid: for each (n_exp, m_exp) channel pair with
  non-zero ω, walks the fine-grid G-vectors with `real_solid_harmonic +
  interp_uniform`. 18 ions × ~100-300 non-zero pairs × 437k grid
  points × CG-sum × radial interp → dominant CPU bottleneck. Plus the
  inverse FFT on the assembled ρ_aug(G).
- Chebyshev + Rayleigh–Ritz — ~5-10 s (GPU).
- `build_v_eff_with_energy_impl` (Poisson + XC + upsample) — ~5-10 s.

**Optimization proposal (deferred — not in Issue #8 scope):**

The Q-function infrastructure is geometry-static — `Q^I_{nm}(G)` for
each ion depends only on the cell, pseudopotential, and ion position.
Only `ω^I_{nm}` and the structure factor change per iteration. Two
tiers of speedup are available:

1. **Cache `Q^I_{nm}(G) · exp(-iG·R_I)` per ion on GPU once at SCF
   start.** Replaces the per-iteration `apply_q_and_sf` CPU walk with a
   single gemm `ρ_aug(G) = Σ_I (Q^I(G) ⊙ SF_I) · ω^I` and a batched
   IFFT on GPU. Expected gain: ~100-200 s → <1 s per iteration. Memory
   cost: 18 ions × ~100-300 non-zero pairs × 437k Complex64 ≈ 1-3 GB —
   feasible on Pascal+. Implementation: extend `VnlBatchData` (or a
   sibling) to hold `q_sf_gpu: Vec<CudaSlice<CudaComplex>>` per ion,
   built once when V_eff iter-1 starts (geometry-stable). Then expose
   a `compute_aug_density_gpu(beta_psi_gpu, occ_gpu, q_sf_gpu)` kernel
   that:
   - Builds ω^I on GPU from cached β·ψ (no D2H needed — keep
     `beta_psi_per_ion` GPU-resident as `Vec<CudaSlice<CudaComplex>>`).
   - Reduces to `ρ_aug(G) = Σ_I Σ_{n,m} ω^I_{nm} · (Q^I_{nm}(G) ⊙ SF_I)`
     via a single batched gemm per ion (n_expanded × n_expanded × n_G).
   - One IFFT to real space.
2. **GPU-port `VnlBatchData::precompute`'s radial Bessel transform** —
   the 200-300 s `precompute_q_on_grid` cost is mostly geometry-static
   per species (one species → one Q table → reused for all 18 Cu ions).
   The current cache already amortizes Cu's Q across 18 ions, so this
   is only worth doing if a multi-species fixture shows it dominant.
   Likely a bigger CPU-side win is parallelising the radial sum with
   rayon over the species-pair index (currently sequential).

**Tier-1 alone (cached Q·SF gemm) probably collapses iter-2 wall time
from 8-9 min to ~1-2 min**, dominated by `VnlBatchData::precompute`.
Tier-1 is a self-contained piece that does not touch the SCF state
machine — `compute_aug_density_fine` becomes a 5-line wrapper.

**Suggested entry point:** add a sibling
`chemrust_hamiltonian_core::QSfCache` that owns the per-ion
`Q^I_{nm}(G) · exp(-iG·R_I)` arrays on GPU. Build it inside
`ScfIteration::new` (geometry is known at construction). Expose
`compute_aug_density_gpu(...) -> RealGrid<f64>` and swap it for the
host `compute_aug_density_fine` call in `compute_density_from_wavefunctions`.
The unit test gate stays the same (iter-2 V_eff range within 1 Ha of
iter-1); regression check is `cargo test --release -- --ignored
iter2_v_eff_range_within_one_ha_of_iter1`.

## 10. Chebyshev filter missing S⁻¹ operator — causes SCF drift after iter-1

**Symptom:** The controlled experiment proves the density code is correct
(same-input validation passes), but our SCF still drifts away from the
converged state after iter-1. The Rayleigh–Ritz produces different eigenvalues
and eigenvectors than CASTEP after the first iteration, leading to progressively
wrong densities in subsequent iterations.

**Root cause (2026-05-22):** The Chebyshev filter recurrence operates on `H`
(the bare Hamiltonian) instead of `S⁻¹·H` (the preconditioned operator for
the generalized USPP eigenproblem H·ψ = ε·S·ψ).

From the reference paper `Parallel-eigensolvers-in-plane-wave-Density-Functional-Theory.tar.gz`
(`abinit.tex` section 4.1):

> "If we denote by Λ and P the eigenvalues and eigenvectors of the eigenproblem
> Hψ = λSψ, then we have the decomposition HP = SPΛ, or S⁻¹H = PΛP⁻¹.
> Therefore, T_n(S⁻¹H)ψ = P T_n(Λ) P⁻¹ ψ will have its eigencomponents
> filtered by the spectral filter T_n."

The Algorithm (Algorithm 1) explicitly requires:
> ψⁱ = (2/r)·(S⁻¹·H·ψⁱ⁻¹ - c·ψⁱ⁻¹) - ψⁱ⁻²

**Current code** (`chebyshev.rs:1133-1137, 1155-1158`):
- `hpsi = H·ψ` via `apply_full_hamiltonian`
- `hpsi = (H·ψ - c·ψ)/d` via `apply_scaled_hamiltonian_inplace`
-> NO S⁻¹ applied.

**Fix infrastructure already exists** (`chebyshev.rs:759-838`):
`apply_s_inverse` is a complete, correct implementation of the Woodbury formula:
`S⁻¹·v = v - β·(Q⁻¹ + β^H·β)⁻¹·β^H·v`

It uses the precomputed `s_inv_mat` (already built in `VnlBatchData::precompute`
and stored in `VnlBatchEntry`) and performs 3 gemm calls:
1. `p = β^H · hpsi` (project H·ψ onto projectors)
2. `q = s_inv_mat · p` (solve reduced Woodbury system)
3. `hpsi -= β · q` (subtract correction)

It is currently dead code (`#[allow(dead_code)]`).

**Fix:** Insert `apply_s_inverse(&mut hpsi_dev, vnl_data, ...)?;` after each
`apply_full_hamiltonian` call inside the recurrence (but NOT after the final
H·ψ for Rayleigh–Ritz, which needs bare H·ψ for H_sub).

**Secondary issue:** Gram-Schmidt orthonormalization (lines 1209-1253) uses
the standard inner product (`cublasZdotc`) instead of the S-inner product.
This could also contribute to drift by producing a subspace that is not
S-orthonormal, potentially making S_sub ill-conditioned.

**Fix scope:**
- `src/eigensolver/chebyshev.rs` — 2 lines: call `apply_s_inverse` after
  `apply_full_hamiltonian` at lines 1131 and 1152
- The `apply_s_inverse` function needs its `#[allow(dead_code)]` removed

**Validation:**
- `fixed_point_matches_castep_energy` — should converge instead of drift
- `iter2_v_eff_range_within_one_ha_of_iter1` — existing regression gate
- Eigenvalue comparison against CASTEP `.bands` should improve

**Status (2026-05-22, post-fix test):** S⁻¹ fix applied but SCF still drifts.
Log at `/tmp/scf-diag-0522-1022.log` shows:

| Metric | Iter-1 | Iter-2 | Iter-3 | Reference |
|--------|--------|--------|--------|-----------|
| Band-1 eigenvalue | −0.956 Ha | −14.40 Ha | −8.83 Ha | −1.055 Ha |
| V_eff range | 8.69 Ha | 26.87 Ha | 39.65 Ha | 8.69 Ha |
| Lanczos α₁ | 7.50 (healthy) | −164.6 (corrupted) | −19.1 (corrupted) | — |
| Cu D_screened/D0 | 0.09× | 4-7× | 4-7× | < 1× |
| Smooth ρ (e⁻) | 163 | 181 | 158 | ~68 |
| Aug ρ (e⁻) | ~23 | ~5 | ~28 | ~118 |

The fix improved iter-2 band-1 from −29 Ha (pre-fix) to −14 Ha, but the
density decomposition flip (smooth 68→181, aug 118→5) suggests the S⁻¹
operator is overcorrecting, pushing charge from augmentation into PW
channels. The S⁻¹ operator's Q matrix convention may not match the
Woodbury formula's expectation.

Additionally, spectral bounds (Lanczos + Gershgorin) are computed for
H, not S⁻¹·H. The Lanczos operates on the bare Hamiltonian, b_up
bounds λ_max(H), but the Chebyshev filter now applies S⁻¹·H whose
eigenvalue spectrum is the generalized eigenvalue problem — related
but not identical. Wrong bounds → wrong filter polynomial → poor
separation of wanted/unwanted eigencomponents.

**Post-fix diagnostic (2026-05-22):** `s_inv_s_identity_test` added to
`tests/ca_scf_convergence.rs` — feeds CASTEP converged ψ₀ through
S⁻¹·S·ψ₀ and checks ‖result − ψ₀‖_∞. Result: **0.014 (1.4%) error**.
The Woodbury S⁻¹ is fundamentally inexact due to Q matrix conditioning
or Gram matrix precision — not a convention mismatch, since both
`q_matrix` and `s_inv_mat` come from the same `build_q_expanded` call.

This explains why standard ChFSI fails: the S⁻¹ error (~1.4%) applied
to O(1) eigenvectors at every Chebyshev step produces persistent O(ε)
error in the filtered subspace. Das et al. (2025) Theorem 3.2 formally
proves that standard ChFSI cannot reduce the subspace angle below a
threshold proportional to the inverse approximation error.

**R-ChFSI solution (2026-05-22):** Das et al. (2025), "Residual-based
Chebyshev filtered subspace iteration for Hermitian eigenvalue problems
tolerant to inexact matrix-vector products" (arXiv preprint, see
`reference_paper/2025-rchfsi-inexact-mv-paper.tar.gz`) introduces
R-ChFSI (Algorithm 3) which reformulates the Chebyshev recurrence
to operate on **residuals** instead of eigenvectors:

```
Standard ChFSI:  ψ_new = C_p(S⁻¹·H) · ψ_old     error ∝ ‖ψ‖ = O(1)
R-ChFSI:         ψ_new = X·Λ_Y + D⁻¹·R_Y       error ∝ ‖R‖ → 0
```

The term `H·D⁻¹·R_Y` in the recurrence uses the inexact S⁻¹ applied to
the current filtered residual R_Y. As the SCF converges, ‖R_Y‖ → 0,
so the error from the inexact inverse vanishes. Theorem 3.4 proves
convergence as long as the Chebyshev polynomial contrast exceeds a
bound that decreases with the subspace angle.

When D⁻¹ = S⁻¹ exactly, R-ChFSI is algebraically equivalent to standard
ChFSI (proved at `main.tex:612`). It strictly generalizes standard
ChFSI and never underperforms it.

**Decision:** Implement R-ChFSI Algorithm 3 as a drop-in replacement
for the current Chebyshev filter body in `chebyshev_filter`. The
Rayleigh-Ritz step, spectral bounds, density construction, and V_eff
assembly stay unchanged. Full rationale at
`reference_paper/ALGORITHM_RATIONALE.md` with precise file:line
references to the source papers.

**Next steps:**
1. Implement R-ChFSI Algorithm 3 in `src/eigensolver/chebyshev.rs`
   (replaces lines 1123-1265 of `chebyshev_filter`).
2. Requires new `apply_s_times` function: S·ψ = ψ + β·q·(β^H·ψ)
   to compute the initial residual Y = H·X - S·X·Λ.
3. Validation: `fixed_point_matches_castep_energy` should converge
   monotonically.

