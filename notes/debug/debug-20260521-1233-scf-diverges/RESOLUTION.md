# Resolution: SCF Diverges After Iter-2 — Root Cause Localised

**Symptom**: `fixed_point_matches_castep_energy` test starts from CASTEP-converged
fixture state. SCF total energy oscillates {4.15M, 1.03M, 2.98M, 1.12M, 2.96M} Ha
across iterations; eigenvalue range widens dramatically; Lanczos b_up jumps
22.8 → 107.5 → 128 Ha despite global V_eff range remaining at 8.5 Ha
through iter-2.

## What was found

### Bug 1 (real, fixed): test data layout in `aug_density_gpu_matches_cpu_cu111_co`

Test built `beta_psi_gpu` from `arr.iter()` on a row-major `Array2`, but
`compute_aug_density_gpu` reinterprets the buffer as col-major. The
production gemm output IS col-major, but the test was using a different
data path → test was structurally unable to validate production.

With original bug: `‖ρ_aug_gpu − ρ_aug_cpu‖_∞ = 1.04e5` (massive).
After fix: `‖ρ_aug_gpu − ρ_aug_cpu‖_∞ = 2.18e-11`.

**Fix location**: `tests/ca_scf_convergence.rs:311-330`. Loop `arr[[n,b]]`
in (b, n) order to write col-major flat.

### Smoking gun for the SCF divergence: D_screened explodes

D_screened tracing in `vnl_data.rs:175-184` showed (per Cu ion, amax):

| Iteration | V_eff range | Cu D_screened amax (max over ions) | Lanczos b_up |
|-----------|-------------|------------------------------------|--------------|
| iter-1    | 8.69 Ha     | 5.82 Ha                            | 22.8 Ha      |
| iter-2    | 8.55 Ha     | **49.97 Ha**                       | 107.5 Ha     |
| iter-3    | 20.57 Ha    | **318.94 Ha**                      | 128 Ha (capped) |
| iter-4    | 20.04 Ha    | **148.67 Ha**                      | 128 Ha (capped) |

D_0 (bare D matrix) for Cu has amax ≈ 61.6 Ha. By iter-3, D_screened amax
is **5× D_0**. This is unphysical — D_screened = D_0 + ∫Q·V_eff. For the
integral term to be 4× D_0, V_eff at ion centres (where Q is concentrated)
must have changed enormously even though global V_eff min/max barely shifted.

The Lanczos b_up tracks ‖V_NL‖ = Σ_I |β⟩ D^I ⟨β|, which scales with
‖D‖. b_up went 5× because ‖D‖ went 10×. So the Lanczos jump is correctly
reporting the operator norm — H really does have spectral radius ~107 Ha
in iter-2.

### What this implies

The divergence cause is NOT layout, NOT non-Hermiticity, NOT mixing.
It is iter-2 V_eff having very different VALUES at ion centres compared
to iter-1, despite global range being similar.

Two paths explain this:
1. **ρ_aug spatial distribution differs from CASTEP**, causing V_eff at
   ion centres to differ. Even with correct integral, wrong spatial
   distribution at ion centres makes ∫Q·V_eff wildly different. The
   `aug_density_gpu_matches_cpu_cu111_co` test only verifies GPU
   matches our CPU implementation; neither is checked against CASTEP's
   ρ_aug pointwise.
2. **V_eff downsampling fine→wave grid** introduces high-frequency
   artefacts at ion centres (Gibbs-like) that don't show in the global
   range diagnostic but corrupt ∫Q·V_eff because Q is sharply peaked.

## Anchor criteria

EXTERNAL anchors used:
- `Cu111_CO.bands` band-1 = -1.05502287 Ha (matched in iter-1 at -1.03 Ha;
  drifts to -0.886 Ha in iter-2, then -12.75 Ha in iter-3).
- `Cu111_CO.den_fmt` (full ρ on fine grid) — could be used to compare
  iter-2's ρ_PW + ρ_aug pointwise; not yet executed.
- `Cu111_CO.pot_fmt` (full V_eff on fine grid) — could be used to compare
  iter-2's V_eff pointwise; not yet executed.

D-matrix screening anchor:
- D_screened amax should be O(D_0) = 6-60 Ha. Anything ≥ 100 Ha indicates
  V_eff at ion centres is corrupted.

## Reclassified prior-session claims

- "SCF diverges due to ρ_aug missing in iter-2" (open-followups #8, May 21):
  partial. The augmentation IS now wired in, but the SPATIAL DISTRIBUTION
  may still be wrong in a way that doesn't show in the integral check.
- "iter-2 V_eff range within 1 Ha of iter-1 → fix verified" (Issue #8
  resolution): range-only check missed the root cause. V_eff range matches,
  but V_eff *values at ion centres* don't. The discriminator test was
  insufficient.

## Next steps (not yet implemented)

1. Pointwise compare iter-2 ρ_PW + ρ_aug vs `Cu111_CO.den_fmt` and iter-2
   V_eff vs `Cu111_CO.pot_fmt` in a small ROI around one Cu ion.
2. Check whether `compute_screened_d_from_fft` does the right thing for
   non-cubic fine grids (the prior session's commit `8691500` claimed to
   fix this, but with a range-only acceptance test).
3. Add an acceptance test that compares post-iter-2 V_eff to CASTEP
   pointwise within ROI(R_I, 2 Å) — this is the test that would have
   caught this bug at fix time.

## Files modified during this session

- **Test fix**: `tests/ca_scf_convergence.rs:311-330` (col-major bp upload).
- **Debug accessor**: `src/scf.rs:1340-1343` (`psi_data()` for tests).
- **New tests**: `tests/aug_density_layout.rs`, `tests/hamiltonian_hermiticity.rs`.
- **Diagnostic prints (NOT TO COMMIT)**: `src/eigensolver/vnl_data.rs:175-184`
  (D_screened amax per ion). Useful for ongoing investigation; remove
  before merging.

## Date

2026-05-21
