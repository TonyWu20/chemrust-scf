# Anchor Criteria: D_screened explosion + eigenvalue drift

## Fixture Files
- `fixtures/cu111_co.rs` — loads CASTEP reference data
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.castep` — CASTEP text output
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands` — reference eigenvalues

## Success Criteria

1. **D_screened magnitude**: After iter-1, `d_screen_amax` for Cu ions must be ≤ 2× `d0_amax` (6.1637e1 Ha).
   - Correct: d_screen_amax ≈ 3–6 Ha (as in iter-1 of the log)
   - Wrong: d_screen_amax ≈ 300–450 Ha (as in iter-2 of the log)
   - Source: CASTEP nlpot.f90:346-355 (fine-grid V_eff for D_screened); log iter-1 baseline

2. **Eigenvalue range**: After iter-2 RR, first eigenvalue must be > -5 Ha.
   - Correct: first ≈ -1.05 Ha (Source: Cu111_CO.bands Band 1: -1.05502343 Ha)
   - Wrong: first = -14.42 Ha (observed in log iter-2)
   - Source: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands`, Band 1

3. **V_eff range stability**: iter-2 V_eff range must be within 2× of iter-1 range.
   - Correct: range ≈ 8–15 Ha
   - Wrong: range = 26.59 Ha (observed — already 3× iter-1's 8.69 Ha)
   - Source: iter-1 log baseline (DERIVED from fixture, not EXTERNAL — used as sanity check only)

4. **Total energy convergence**: After 8 SCF iterations, |E_computed - E_ref| < 2e-4 eV.
   - E_ref = -24110.96665069 eV
   - Source: `tests/fixtures/cu111_co.rs:25`, CASTEP Cu111_CO.castep line 326
