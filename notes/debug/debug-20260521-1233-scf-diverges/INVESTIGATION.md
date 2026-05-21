# Investigation: SCF Diverges After Iter-2

**Symptom**: `fixed_point_matches_castep_energy` test starts from CASTEP-converged
fixture state. Iter-1 reproduces fixture V_eff range (8.69 Ha) and CASTEP
eigenvalues (band-1 = -1.03 Ha). Iter-2 V_eff range stays sane (8.55 Ha) but
its Lanczos b_up jumps from 22.8 Ha → 107.5 Ha; Chebyshev filter then barely
amplifies (ratios 1.01-1.10) and iter-2 eigenvalues drift to [-0.886, 1.7169].
Iter-3 V_eff explodes to 20.57 Ha range; total energies oscillate
{4.15e6, 1.03e6, 2.98e6, 1.12e6} Ha across iters.

## Prior numeric claims classification

| Claim | Class | Source | Admissible? |
|-------|-------|--------|-------------|
| `iter-1 V_eff range = 8.6877 Ha` | DERIVED | log + open-followups #8 | No (our output) |
| `iter-2 V_eff range = 8.5573 Ha` | DERIVED | log + open-followups #8 | No (our output) |
| `Smooth ρ_PW integral = 1,029,838` | DERIVED | log + open-followups #8 | No |
| `Augmentation ρ_aug integral = 3,120,753` | DERIVED | log + open-followups #8 | No |
| `Total ρ integral = 4,150,591` | DERIVED → corroborated by `= N_e × Ω = 186 × 22310 ≈ 4.15e6` | Open-followups #8 + cell volume | Partially anchored |
| `CASTEP reference total energy = -24110.96665069 eV` | EXTERNAL | `Cu111_CO.castep` line 326 | YES |
| `CASTEP band-1 eigenvalue = -1.05502287 Ha` | EXTERNAL | `Cu111_CO.bands` | YES |
| `CASTEP V_eff (54×90×90 fine grid, Hartree)` | EXTERNAL | `Cu111_CO.pot_fmt` | YES |
| `CASTEP density (with augmentation)` | EXTERNAL | `Cu111_CO.den_fmt`, `Cu111_CO.castep_bin` | YES |

## Prior-session memory citations

- Open-followups #7: "Lanczos kernel is correct; H itself is corrupted in iter 2."
  - **Reclassified**: DERIVED — this was the diagnosis from a prior session
    based on alpha[1]=-391 Ha. The current symptom is different (alphas are
    positive but inflated starting at k=2). Cannot be used as a criterion.
- Open-followups #8 resolution: "iter-2 V_eff range 8.5573 Ha, |Δ| = 0.1304 Ha
  ... 7.7× margin". DERIVED. But it was verified by the dedicated test
  `iter2_v_eff_range_within_one_ha_of_iter1` which is **range-only** — it
  does NOT verify spatial distribution of iter-2 ρ_aug or V_eff. Therefore
  the "fix" may be incomplete: integrals match but spatial pointwise values
  could still be wrong.

## Smoking gun for new investigation

Between Lanczos call #1 (iter-1 diagonalize) and Lanczos call #2 (iter-2
diagonalize), V_eff range is essentially identical (8.69 → 8.55 Ha), the
Lanczos starting vector is byte-identical (norm0=2.450857e2 in both),
β_g is geometry-static, kinetic T is fixed, yet b_up jumps from 22.8 →
107.5 Ha (5× higher). This indicates the Hamiltonian apply differs in a
way that the global V_eff range does not capture.

Candidate sources:
- **Spatial distribution of V_eff** (range matches but values at specific
  grid points differ). If V_eff near ion cores changes, D-matrix screening
  (`D = D0 + ∫Q·V_eff` with Q localized at ions) changes.
- **ρ_aug spatial distribution**. The cached test
  `aug_density_gpu_matches_cpu_cu111_co` uses a layout-mismatched test
  setup: it H2Ds row-major `arr.iter()` into bp_dev, but the production
  code in `compute_aug_density_gpu` does `D2H + Array2::from_shape_vec((ne,
  n_bands).f(), ...)` which interprets the buffer as col-major. The
  production hot path is layout-consistent (gemm produces col-major and
  reinterprets as col-major), but it means the test does NOT validate
  production semantics. The test may pass by relying on a cached
  rho_aug_cpu/rho_aug_gpu match that doesn't represent the real
  production call.

The hot path (in `diagonalize`, `src/scf.rs:493`) rebuilds `vnl_data` every
iteration. `compute_beta_g` is geometry-static (does not depend on bands).
`compute_screened_d` depends only on V_eff and Q (geometry-static).
Therefore the path most likely to introduce iteration-dependent errors is
the spatial distribution of ρ_aug → V_eff → D screening.

## Resolution status

Open. Hypotheses to test in Step 7 once external anchor criteria are
established and upstream-audit gate is resolved.
