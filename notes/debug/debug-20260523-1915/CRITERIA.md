# Anchor Criteria: Soft/Aug Density Split

## Fixture Files
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands` — 160 CASTEP reference eigenvalues
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.check` — converged wavefunctions  
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.pot_fmt` — converged V_eff
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/slurm_output_2291.txt` — F8-instrumented run output
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.castep` — total energy (line 326)

## Success Criteria (EXTERNAL anchors only)

### SC-1: Band-0 eigenvalue at iter-1 matches CASTEP
- **Assertion**: `|eigenvalues[0] - (-1.05502343)| < 0.05` Ha
- **Source**: Cu111_CO.bands line 12
- **Status**: PASS (|Δ| = 0.0092 Ha, measured 2026-05-23)

### SC-2: Density code correct with CASTEP wavefunctions
- **Assertion**: soft ratio ∈ [0.99, 1.01] and aug ratio ∈ [0.99, 1.01]
- **Source**: density_decomp_matches_castep_f8_same_inputs test
- **Status**: PASS (soft=1.000000, aug=1.000084, from prior session)

### SC-3: Iter-2 density split vs F8 anchors
- **Assertion**: soft fraction (N_e_soft / N_e_total) within 5% of F8 value
- **F8 soft fraction**: 68.44 / 186.00 = 36.8%
- **F8 aug fraction**: 117.56 / 186.00 = 63.2%
- **Source**: F8 instrumented output at converged iteration

### SC-4: Iter-2 last-band overshoot within 1.0 Ha
- **Assertion**: `|eigenvalues_iter2[159] - 0.11531044| < 1.0` Ha
- **Source**: Cu111_CO.bands last line
