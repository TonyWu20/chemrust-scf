# Anchor Criteria: Lanczos S⁻¹·H inner-product bug

## Fixture Files

- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.castep` — CASTEP text output (final energy, SCF table)
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands` — reference eigenvalues (160 bands, 1 kpt, 1 spin)
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.den_fmt` — reference density on fine grid
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.pot_fmt` — reference V_eff on fine grid
- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/` — F8-instrumented run with per-iteration soft/aug density sums

## Success Criteria

### C1: Lanczos alpha stability across SCF iterations (smoking gun)
- **Assertion**: In iter-2 Lanczos, all alpha[j] values must be < 20 Ha and monotonic.
- **Correct** (iter-1 baseline): alpha = [8.80, 7.50, 7.42, 7.41, 7.42, 7.41] — stable, bounded by Gershgorin radius of S⁻¹·H.
- **Wrong** (current iter-2): alpha = [9.15, 10.14, **96.15**, 41.45, **79.85**, 59.44] — non-Hermitian Lanczos produces spurious eigenvalues.
- **Discriminator**: `max(|alpha_j|) < 20 Ha` — 4.8× separation from 96.15 Ha.
- **Source**: Log observation, classification DERIVED (used as sanity check, not EXTERNAL anchor).

### C2: Spectral bound b_up must be stable across SCF iterations
- **Assertion**: In iter-2, `b_up` must be < 30 Ha.
- **Correct** (iter-1): b_up = 20.81 Ha (Lanczos × 1.1, not Gershgorin-capped).
- **Wrong** (current iter-2): b_up = 134.39 Ha (Gershgorin-capped at 134.39, raw Lanczos would give 243.88 Ha).
- **Discriminator**: `b_up < 30 Ha` — 4.5× separation from 134.39 Ha.
- **Source**: Log observation, classification DERIVED.

### C3: D_screened amax must stay within 2× D_0 amax
- **Assertion**: For Cu ions, `d_screen_amax` ≤ 2 × 6.1637e1 = 123 Ha.
- **Correct** (iter-1): d_screen_amax = 3.7–6.0 Ha (well below D_0=61.6 Ha).
- **Wrong** (current iter-2): d_screen_amax = 189–455 Ha (3–7× D_0).
- **Discriminator**: `d_screen_amax < 123 Ha` — 1.5× separation from 189 Ha.
- **Source**: CASTEP nlpot.f90:346-355 (CASTEP uses fine-grid V_eff for D_screened; our convention matches for this fixture since wave_grid == fine_grid). Monitor only — D_screened explosion is a **downstream consequence** of wrong V_eff, not a root cause.

### C4: Eigenvalue range must not catastrophically shift
- **Assertion**: After iter-2 RR, first eigenvalue must be > -5 Ha.
- **Correct**: first ≈ -1.055 Ha.
- **Wrong** (current iter-2): first = -14.42 Ha.
- **Source**: `Cu111_CO.bands`, Band 1 = -1.05502287 Ha. **EXTERNAL**.

### C5: R-ChFSI residual norms must grow (amplification), not shrink (damping)
- **Assertion**: In iter-2, `norm_curr / norm_prev > 1` for the first few k-steps (amplification phase).
- **Correct** (iter-1): ratios = 2.21, 1.48, 1.31, 1.23, 1.18, 1.14, 1.12 (all > 1).
- **Wrong** (current iter-2): ratios = 0.64, 0.63, 0.81, 0.75, 0.72, 0.71, 0.71 (all < 1).
- **Source**: Log observation, classification DERIVED. The shrinking norms indicate the filter's half-width is too large (66.5 Ha), placing the tracked eigenvalue region at |x| ≈ 1.0 where T_k(1) = 1 (no amplification).

### C6: Total energy convergence (end-to-end)
- **Assertion**: After 8 SCF iterations, |E_computed - E_ref| < 2e-4 eV.
- **E_ref**: -24110.96665069 eV.
- **Source**: `tests/fixtures/cu111_co.rs:25`, CASTEP Cu111_CO.castep line 326. **EXTERNAL**.
