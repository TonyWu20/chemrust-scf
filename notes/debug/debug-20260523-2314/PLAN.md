# Debug Plan: §13 SCF Cascade — D-screening Comparison via CASTEP Dump

## Context

`notes/open-followups.md` §13 documents an unresolved SCF cascade: starting from
CASTEP's converged wavefunctions and V_eff for Cu111+CO, our SCF cascades:

| Iter | band-0 (Ha) | density split (soft / aug) | Status |
|------|-------------|---------------------------|--------|
| 1    | −1.046      | 29.7 / 70.3 %              | eigenvalues match CASTEP, density split wrong |
| 2    | −0.869      | 69.9 / 30.1 % (inverted)   | drifting |
| 3    | −11.94      | (collapse)                | catastrophic |

The §13 RESOLUTION attributes the cascade to *"eigenvector rotation within
degenerate Cu 3d manifolds inherent to subspace methods vs CASTEP's band-by-band
CG."* This claim is **HYPOTHESIZED**, not EXTERNAL: CASTEP also subspace-projects
inside its CG/RR loop, yet its converged state is a fixed point of its own SCF.
A pure subspace-rotation explanation cannot account for the *catastrophic* drift
of band-0 from −1.046 Ha → −11.94 Ha in two iterations.

**What changed since the last debug session**: the user has switched
`~/Downloads/CASTEP-6.11-nixos/` to branch `dumo/vxc-full-grid-gather`, which
contains a screened-D matrix dump instrumented at `Source/Functional/nlpot.f90:531-544`:

```fortran
open(unit=9877, file='D_band_debug.dat', position='append', status='unknown')
write(9877,'(2I6)') nsp, num_ps_projectors(nsp)
do dn = 1, num_ps_projectors(nsp)
  do dm = dn, num_ps_projectors(nsp)
    write(9877,'(2I6,ES24.16)') dn, dm, nl_d(dm,dn,ni,nsp,ns)
  end do
end do
```

This is written **after** `nl_d(m,n) += ps_D0(m,n)` at line 523, so the dump
contains `D_screened = D_0 + ∫Q·V_eff` — exactly the quantity our
`chemrust_hamiltonian_core::compute_screened_d` returns. **Direct
element-by-element comparison is now possible**, with CASTEP as the EXTERNAL
ground-truth anchor.

**Critical caveat — `mixture_weight`**: at `nlpot.f90:523`, CASTEP applies
`nl_d(m,n) = (nl_d(m,n) + ps_D0(m,n)) * current_cell%mixture_weight(ni,nsp)`
*before* the dump. So the recorded value is `D_screened × w_mix`. For Cu111+CO
(no VCA), `mixture_weight = 1.0` and the dump equals `D_screened` directly. Any
future fixture with VCA-mixed species must divide the dumped value by
`mixture_weight` before comparison.

**Companion CASTEP dumps in the same branch** (commit `c7182ce`): the parallel
work for NiO writes per-SCF-iteration `psat%D(nn,mm)` to `NiO.D_debug.dat`,
`D_0` to `NiO.D0_debug.dat`, and a radial V_eff(r) profile from `ion_atom.f90`.
These are not needed for Cu111+CO this session, but they offer a *per-iteration*
D anchor (vs our converged-only anchor) for future use.

**Goal of this session**: Use the new D dump to (a) falsify the §13 "rotation is
inherent" hypothesis, (b) localise the cascade root cause to one of {V_eff
fidelity, D-screening computation, density assembly}, and (c) write a tight test
gating the fix.

## Hypothesis ledger

| Claim | Source | Class |
|-------|--------|-------|
| §13 cascade band-0 = −11.94 Ha at iter-3 | `cascade_iter3_diagnostic` | **EXTERNAL** (our test, but reproducible) |
| iter-1 band-0 within 0.05 Ha of CASTEP | `issue_11a_iter1_band0_matches_castep` + `Cu111_CO.bands` | **EXTERNAL** |
| Density code correct for CASTEP ψ (ratios 1.000000/1.000084) | `density_decomp_matches_castep_f8_same_inputs` | **EXTERNAL** |
| RR mathematics correct (160 bands within 0.05 Ha of `.bands`) | §12 (six tests in `tests/rayleigh_ritz_validation.rs`) | **EXTERNAL** |
| S⁻¹·S identity within 3.8e-15 | `s_inv_s_identity_test` | **EXTERNAL** |
| Mixing scheme (Pulay + Kerker) matches CASTEP | `Cu111_CO.param` line `MIXING_SCHEME : Pulay` | **EXTERNAL** |
| compute_screened_d formula matches CASTEP nlpot_calculate_d (sign, normalization, conjugate, structure factor) | side-by-side audit (this session) | **EXTERNAL** |
| `q.norm_sqr() < 1e-60` skip in our compute_screened_d | source line 396 | **EXTERNAL** existence; HYPOTHESIZED harmlessness |
| "Subspace-method rotation is inherent" → produces cascade | §13 RESOLUTION | **HYPOTHESIZED** — no external corroboration |
| V_eff drift between iter-1 and iter-2 amplifies through D-screening | §13 RESOLUTION proposal | **HYPOTHESIZED** — to be tested |

## Pre-flight (Phase 0): Generate the EXTERNAL D anchor

Before writing any test code, we need `D_band_debug.dat` for the Cu111+CO
fixture. The dump file is currently absent from
`/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`.

**Steps** (user-driven, this is the only out-of-process work):

1. Build CASTEP-6.11 on branch `dumo/vxc-full-grid-gather` (CPU/MPI build —
   the dump runs on `on_root_node` only).
2. Re-run the Cu111+CO single-point job in the fixture directory with the
   newly-built binary (`sbatch slurm_job_Cu111_CO.sh`).
3. Verify `D_band_debug.dat` appears, accumulates over SCF iterations
   (~50 × 18 ions × N_proj records), and the **last** block per
   `(species, ion)` corresponds to the converged iteration.

**Exit criterion for Phase 0**:
- `D_band_debug.dat` exists, parses cleanly, and per-ion blocks have
  consistent record counts (`num_ps_projectors(nsp)·(num_ps_projectors(nsp)+1)/2`
  records per block).
- The last 18 blocks (one per ion, single spin) are taken as the converged
  D-screened reference.

## Test specifications (Phase 1: localisation)

All tests gated `#[ignore]` + `#[cfg(feature = "scf_diag")]`, run via
`cargo test --release --features scf_diag -- --ignored --nocapture --test-threads=1`.

### T0 — `parse_castep_d_dump` (anchor extraction)

**Purpose**: Parse `D_band_debug.dat` into `Vec<Array2<f64>>` (one matrix per
ion). Take the *last* block per (species, ion) tuple as the converged D.

**Location**: New helper `tests/fixtures/cu111_co.rs::load_castep_d_screened`.

**Body sketch**:
- Read the file line-by-line.
- Each block starts with `(nsp, num_proj)`; followed by
  `num_proj·(num_proj+1)/2` triples `(dn, dm, value)`.
- Build symmetric `Array2<f64>` per ion (mirror to upper triangle: matches
  CASTEP `nl_d(n,m) = nl_d(m,n)` at line 526).
- Return `HashMap<(species_idx, ion_idx_in_species), Array2<f64>>` —
  the *latest* block per key (file is append-only, last wins = converged).

**Discriminator**: Round-trip — parse the file, re-format the first block,
and verify byte-for-byte match against a slice of the input. No physics
threshold; this is a parser correctness test.

### T1 — `h_on_castep_psi_matches_bands` (operator falsification)

**Purpose**: Falsify "rotation is inherent" by checking whether our `H` operator
agrees with CASTEP's eigenvalue spectrum *on CASTEP's exact eigenvectors*. If
yes, our H is correct on the un-rotated basis — the cascade cannot be a pure
"subspace rotation cascade" because the operator agrees pointwise with CASTEP
in CASTEP's basis.

**Scope**: Diagonal only — `⟨ψ_b | H | ψ_b⟩` per band. §12 already validates the
full H_sub assembly and the RR generalised-eigenvalue residual to machine
precision (`tests/rayleigh_ritz_validation.rs`), so T1's job is narrower:
isolate the operator on the un-rotated CASTEP basis.

**Location**: `tests/ca_scf_convergence.rs` after `cascade_iter3_diagnostic`
(~L2245).

**Body sketch**:
- Load fixture (`Cu111_CO.check`, `Cu111_CO.pot_fmt`, `Cu111_CO.den_fmt`).
- Build initial state: `build_scf_state(fx)` → `build_v_eff_with_energy()` →
  `precompute_vnl_data()` (so D_screened is loaded with CASTEP V_eff).
- Reuse `apply_h_components_for_test` (already exists in `src/scf.rs:776-810`)
  to apply H to each band of CASTEP ψ.
- Per band b: compute `⟨ψ_b | H | ψ_b⟩` = `inner_product(&psi_b, &h_psi_b).re`.
- Compare against `fx.bands_eigenvalues[b]` (loaded from `Cu111_CO.bands`).

**Discriminator**: RMS over 160 bands ≤ **0.05 Ha**, per-band max ≤ **0.10 Ha**.
Anchor: `Cu111_CO.bands`, line 12 (band-0 = −1.05502343 Ha) through last line.
The 0.05 Ha threshold matches §12 RR validation precedent.

**Decision branch**:
- **PASS**: H operator is correct on CASTEP's eigenvectors. Proceed to T2.
  The §13 "inherent rotation" hypothesis is downgraded — the operator matches.
- **FAIL**: H operator is wrong on CASTEP's fixed point. Open §14 ("H-on-CASTEP-ψ
  mismatch") and pivot to upstream audit of V_NL / V_loc / V_H / V_xc components.
  All other tests become unanchored.

### T2 — `d_screened_matches_castep_dump_on_castep_veff` (D anchor sanity)

**Purpose**: With CASTEP V_eff loaded as input, our `compute_screened_d` should
exactly reproduce the CASTEP D-matrix dump. This is the **upstream-audit
sanity gate** — if our function differs from CASTEP's value when fed the *same*
V_eff, the bug is in `compute_screened_d` itself, not in V_eff fidelity.

**Location**: `tests/ca_scf_convergence.rs` ~L2300.

**Body sketch**:
- Load fixture; convert `fx.pot_fmt` to `EffectivePotential`.
- For each ion (18 ions in Cu111+CO):
  - Get the species-index Q-on-grid (precomputed once).
  - Call `compute_screened_d(&q_grid, &fx_veff, &cell, ion_idx, &gvg, &d0)`.
- Load CASTEP D dump via T0's helper; align indexing
  (CASTEP `(species, ion_in_species)` → our `ion_idx`).
- **Apply `mixture_weight`**: CASTEP dumps `nl_d × w_mix` (see Context). For
  Cu111+CO `w_mix = 1.0` for all ions. T0's parser must read
  `current_cell.mixture_weight` from the fixture and divide the dumped value
  before comparison. Fail fast if `w_mix == 0.0` for any ion (would imply a
  parsing bug).
- Compare each (n, m) element.

**Discriminator (per-ion)**: max |D_ours[n,m] − D_CASTEP[n,m]| ≤ **5e-4 Ha**.
Anchor: `D_band_debug.dat` (machine-precision dump, ES24.16 format).
Threshold rationale: CASTEP's nominal precision is double; numerical drift from
FFT differences between fftw3 (CASTEP) and our cuFFT/CPU FFT bounded around
1e-6 Ha relative to D values that range from ~0.01 to ~10 Ha. 5e-4 gives ≥ 2×
margin against expected numerical noise of ~1e-6 to ~5e-5.

**Decision branch**:
- **PASS**: `compute_screened_d` is correct. The cascade is in V_eff
  fidelity (proceed to T3). The 1e-60 Q-skip is empirically harmless.
- **FAIL**: Localise the formula divergence — log per-ion, per-(n,m) deltas;
  identify whether failure is concentrated on (a) cross-m pairs (Q skip
  hypothesis), (b) high-l channels (radial integration hypothesis), or
  (c) all pairs uniformly (unit/normalization hypothesis).

### T3 — `cascade_with_castep_veff_substitution` (V_eff fidelity isolation)

**Purpose**: Substitute CASTEP V_eff for *our* iter-1 output V_eff before iter-2.
If the cascade vanishes, the bug is in `build_v_eff_with_energy` (Hartree, XC,
upsample/downsample, augmentation density assembly). If the cascade persists,
the bug is downstream of V_eff (in diagonalize → density given correct V_eff).

**Phase-targeting note**: the existing `set_v_eff` at `src/scf.rs:1610` is on
the `VEffBuilt` phase, but T3 needs to inject V_eff *after* iter-1's density
mixing — i.e., on the `DensityUpdated<MixingOff>` (or whichever Mixed phase
follows iter-1) phase, *replacing* iter-2's `build_v_eff_with_energy` output.
This requires a **new mutator** on a different state-machine phase, not reuse.

**T3 body sketch**:
- Replicate `cascade_iter3_diagnostic` iter-1 block (L2178-2185).
- Before iter-2's `build_v_eff_with_energy`, call the new `DensityUpdated`-phase
  `set_v_eff` mutator to inject CASTEP `.pot_fmt` V_eff. (Equivalently:
  `build_v_eff_with_energy` runs to completion, then we *overwrite* with
  CASTEP V_eff — implementation choice for whichever is type-cleaner.)
- Continue iter-2 → iter-3.

**Location**: `tests/ca_scf_convergence.rs` ~L2360.

**Discriminator**: iter-2 band-0 within **0.05 Ha** of −1.055 Ha (anchor:
`Cu111_CO.bands`). Iter-2 density soft fraction within **±2%** of 36.8% (anchor:
F8 dump in `slurm_output_2291.txt`).

**Decision branch**:
- **PASS** (cascade stops): V_eff assembly is the bug. Audit
  `build_v_eff_with_energy` — Hartree, XC kernel, fine-grid upsampling,
  ρ_aug assembly.
- **FAIL** (cascade persists): V_eff is *not* the only driver. Bug is in the
  diagonalize → density path even when V_eff is correct. Run T4 to isolate
  density assembly from RR.

### T4 — `cascade_with_castep_density_substitution` (density-assembly isolation)

**Purpose**: Substitute CASTEP density (`Cu111_CO.den_fmt`) for our iter-1
output density before iter-2's V_eff rebuild. If iter-2 stays correct, density
construction from our (rotated) ψ is the cause. If it cascades, V_eff
assembly is *also* a contributor independent of density.

**Location**: `tests/ca_scf_convergence.rs` ~L2420.

**Body sketch**:
- Same setup as T3, but after iter-1's mix, replace mixed density with
  `Density::from_inner(WaveGridArray::from_inner(...))` built from
  `fx.den_fmt` downsampled to wave-grid (use existing `downsample_array_to_wave_grid`
  in `scf.rs:1565-1596`).
- Requires `pub fn set_density(&mut self, d: Density)` analogous to `set_v_eff`
  at `scf.rs:1610-1622`. **Only new mutator**, no GPU code.

**Discriminator**: iter-2 band-0 within 0.05 Ha. Anchor: `Cu111_CO.bands`.

### T5 — `cascade_off_mixing_only` (cheap mixing-effect probe)

**Purpose**: Force `mix_charge_amp = 0.0` (skip Off → Kerker transition; treat
all iterations as no-mix) for one run. Cross-check whether mixing is masking or
amplifying the cascade. Useful only if T3/T4 leave residual ambiguity.

**Location**: `tests/ca_scf_convergence.rs` ~L2480.

**Discriminator**: qualitative — does the cascade get worse, better, or
unchanged? No EXTERNAL anchor; this is a probe, not a falsifier. Skip if T3/T4
are decisive.

## Test ordering (signal strength, lowest cost first)

Execute in order; abort on the first failure that decisively localises the bug.

```
T0  parse_castep_d_dump                    parser correctness only
T1  h_on_castep_psi_matches_bands          operator falsification
T2  d_screened_matches_castep_dump_…       D-screening function falsification
T3  cascade_with_castep_veff_substitution  V_eff-fidelity isolation
T4  cascade_with_castep_density_…          density-assembly isolation
T5  cascade_off_mixing_only                mixing-effect probe (only if needed)
```

**Decision tree**:

```
T1 FAIL  → pivot to §14 (H operator wrong); abandon §13 hypothesis tree.
T1 PASS, T2 FAIL → bug in compute_screened_d formula; fix per per-(n,m) delta pattern.
T1 PASS, T2 PASS, T3 PASS → bug in our build_v_eff_with_energy; audit Hartree/XC/upsample/ρ_aug.
T1 PASS, T2 PASS, T3 FAIL, T4 PASS → bug in density assembly from rotated ψ.
T1 PASS, T2 PASS, T3 FAIL, T4 FAIL → bug downstream of both V_eff and density:
                                      diagonalize/RR/Chebyshev — but T1+T2 already excluded
                                      H+D-screening, so this points to occupations or filter.
```

## File:line edit list

All edits happen in tests + minimal SCF setters. **No new GPU kernels, no
algorithm code changes.**

| File | Location | Change |
|------|----------|--------|
| `tests/fixtures/cu111_co.rs` | end of file | Add `pub fn load_castep_d_screened(fx: &Fixture) -> HashMap<(usize, usize), Array2<f64>>` parser for `D_band_debug.dat`. **Must read `mixture_weight` per ion and divide the dumped value before returning** (see Context caveat). |
| `tests/fixtures/cu111_co.rs` | end of file | Add `pub fn castep_veff_as_effective(fx: &Fixture) -> EffectivePotential` helper (factor 4 lines repeated across tests). |
| `src/scf.rs` | ~L1622 (next to existing `set_v_eff` on `VEffBuilt`) | Add `pub fn set_v_eff` on the `DensityUpdated<MixingOff>` (or equivalent post-mix) phase. Used by T3 only. The existing `VEffBuilt`-phase setter at L1610 is *not* sufficient because T3 needs to inject V_eff after iter-1's density mixing. |
| `src/scf.rs` | ~L1624 | Add `pub fn set_density(&mut self, d: Density)` mutator on the appropriate post-mix phase. Used by T4 only. |
| `tests/ca_scf_convergence.rs` | ~L2245 | Add **T1** `h_on_castep_psi_matches_bands` (diagonal-only ⟨ψ\|H\|ψ⟩). |
| `tests/ca_scf_convergence.rs` | ~L2300 | Add **T2** `d_screened_matches_castep_dump_on_castep_veff`. |
| `tests/ca_scf_convergence.rs` | ~L2360 | Add **T3** `cascade_with_castep_veff_substitution`. |
| `tests/ca_scf_convergence.rs` | ~L2420 | Add **T4** `cascade_with_castep_density_substitution`. |
| `tests/ca_scf_convergence.rs` | ~L2480 | Add **T5** `cascade_off_mixing_only` (optional). |

**Existing infrastructure being reused** (no new code):
- `apply_h_components_for_test` at `src/scf.rs:776-810` (T1)
- `set_v_eff` at `src/scf.rs:1610` — pattern (not reuse) for the new `DensityUpdated`-phase mutator (T3)
- `compute_screened_d` at `chemrust-hamiltonian-core/src/nlpot.rs:368-411` (T2)
- `precompute_q_on_grid` at `chemrust-hamiltonian-core/src/nlpot.rs` (T2)
- `downsample_array_to_wave_grid` at `src/scf.rs:1565-1596` (T4)
- `diagnose_d_screening_values` at `tests/ca_step_validation.rs:580` (T2 — reference template for per-ion D iteration)
- Test-call template for `compute_screened_d` at
  `chemrust-hamiltonian-core/tests/integration.rs:2014-2017` and `2478-2481`

**Failure pattern citation**: `range-only-acceptance-misses-pointwise-divergence`
(in `notes/failure-patterns.md` and the auto-memory). T2 directly remediates
this pattern at the D-screened layer with the new CASTEP element-by-element
anchor.

## Step-7 (Loose-then-Tighten) compliance

Per `/home/tony/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/debug-outcomes/SKILL.md`:

1. **Loose run**: existing `cascade_iter3_diagnostic` already shows the cascade.
   Capture current numbers (band-0 = −11.94 Ha at iter-3) as baseline.
2. **Tight test (must fail on broken code)**: T2 with anchor 5e-4 Ha against
   the CASTEP D dump. Run *before* any fix is applied.
3. **Falsify the §13 "rotation is inherent" hypothesis**: T1.
4. **Localise**: T2 → T3 → T4.
5. **Implement fix** (scope determined by which test fails). Edit→check→fix
   loop.
6. **Green tight test**: re-run T1 through T4 plus `cascade_iter3_diagnostic`.
   Iter-3 band-0 must stay within **0.05 Ha** of `.bands` band-0.

## Exit criteria

- `cascade_iter3_diagnostic` iter-3 band-0 within 0.05 Ha of −1.055 Ha.
- Iter-2 density soft fraction within ±2% of 36.8%.
- T1, T2, T3, T4 all pass.
- The fix is documented in `notes/debug/<slug>/RESOLUTION.md` with the
  failing test (T1/T2/T3/T4) cited as discriminator.
- §13 entry in `notes/open-followups.md` updated with RESOLUTION pointer.

## Rollback / off-ramp triggers

| Condition | Action |
|-----------|--------|
| Phase 0 cannot produce `D_band_debug.dat` (CASTEP build fails / job fails) | Pause; user investigates CASTEP build. T2 cannot run without it. |
| T1 fails | Abandon §13 hypothesis tree. Open §14 "H operator wrong on CASTEP eigenvectors". The other tests are unanchored. |
| T1 PASS, T2 PASS, T3 PASS, T4 PASS, T5 inconclusive | The cascade is *not* localised by these substitution tests — re-examine the assumption set. Likely candidates: occupation smearing (Gaussian erfc width / temperature) drift between iterations, or a Chebyshev filter window bootstrap issue. |

## Notes / open questions

- **D-dump file size**: append-only across SCF iterations. For a converged
  Cu111+CO run (~50 SCF iters × 18 ions × ~10 projectors × ~55 unique pairs),
  `D_band_debug.dat` ≈ 100–500 KB. Tractable.
- **Per-iteration anchor (future)**: T0's parser keeps only the *last* block
  per `(species, ion)`. A future enhancement could keep every block and let
  T2/T3 anchor each of our SCF iterations against the corresponding CASTEP
  iteration. Not in scope for this session — the converged-iteration anchor
  is sufficient to discriminate. Note that the same CASTEP branch
  (`dumo/vxc-full-grid-gather`, commit `c7182ce`) also instruments
  `Source/Fundamental/ion_atom.f90` with per-iteration `psat%D(nn,mm)` →
  `<seed>.D_debug.dat`, `D_0` → `<seed>.D0_debug.dat`, and a radial V_eff(r)
  profile. These are written for the NiO test case but become available for
  Cu111+CO if we re-enable the same hooks. Available as a richer anchor
  source in a follow-on session.
- **`mixture_weight` caveat**: the parser must divide by
  `current_cell.mixture_weight` per ion before comparison. For Cu111+CO this
  is 1.0; for any future VCA fixture it matters.
- **Q-on-grid caching across tests**: T2 calls `precompute_q_on_grid` once per
  unique species (Cu, C, O); reuses across 18 ion calls. ~30 s / species on
  CPU. Acceptable.
- **No edits to `chemrust-hamiltonian`** in scope. If T2 fails and the fix is
  in `compute_screened_d`, the fix lands there in a follow-on session.

## Verification

Single command to run the full test set after Phase 0 produces the dump:

```bash
cargo test --release --features scf_diag --test ca_scf_convergence \
  -- --ignored --nocapture --test-threads=1 \
  h_on_castep_psi_matches_bands \
  d_screened_matches_castep_dump_on_castep_veff \
  cascade_with_castep_veff_substitution \
  cascade_with_castep_density_substitution \
  cascade_iter3_diagnostic
```

Each test prints its discriminator value; failures point to the next narrow
fix scope. The plan is complete when `cascade_iter3_diagnostic` shows iter-3
band-0 within 0.05 Ha of −1.055 Ha — i.e., the cascade is gone.
