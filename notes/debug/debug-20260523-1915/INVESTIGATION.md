# Investigation: Soft/Aug Density Split Discrepancy

**Symptom**: SCF delivers significantly different soft density/augmented density
split compared to CASTEP's F8 dump. Previous test proved density code is correct
when fed CASTEP's converged wavefunctions (ratios: soft=1.000000, aug=1.000084).
This implies our Rayleigh-Ritz wavefunctions are producing a different density
split.

**Session date**: 2026-05-23
**Branch**: debug/density-split-audit (from feat/phase-global-woodbury)
**Worktree**: /home/tony/programming/chemrust-scf-debug

## Prior-note claim classification

| Claim | Source | Class | Admissible? |
|-------|--------|-------|-------------|
| CASTEP F8 soft sum = 2.99359524940157e7 | slurm_output_2291 (converged iter) | **EXTERNAL** | Yes |
| CASTEP F8 aug sum = 5.14204469948334e7 | slurm_output_2291 (converged iter) | **EXTERNAL** | Yes |
| CASTEP band-0 = -1.05502343 Ha | Cu111_CO.bands line 12 | **EXTERNAL** | Yes |
| CASTEP last-band = 0.11531044 Ha | Cu111_CO.bands last line | **EXTERNAL** | Yes |
| Density code correct (ratio 1.000000/1.000084) | `density_decomp_matches_castep_f8_same_inputs` | **EXTERNAL** | Yes |
| RR-only (ndeg=0) matches CASTEP within 0.05 Ha | `ndeg_zero_with_castep_psi_matches_bands` | **EXTERNAL** | Yes |
| Das Alg 3 Step 4 S⁻¹·R_Y deviation | Divergence surface analysis | **HYPOTHESIZED** | No (empirically disproved — fix made eigenvalues worse) |
| Das Alg 3 Step 3 operator order deviation | Divergence surface analysis | **HYPOTHESIZED** | No — S⁻¹·(H·R_Y) = (S⁻¹·H)·R_Y which is correct |
| iter-1 band-0 = -1.0458 Ha (after all §11 fixes) | This session, `issue_11a_iter1_band0_matches_castep` | **EXTERNAL** | Yes |
| Filter correct at iter-1 (|Δband0| = 0.0092 Ha) | This session | **EXTERNAL** | Yes |
| S⁻¹ exact (ζ = 3.8e-15) | `s_inv_s_identity_test` | **EXTERNAL** | Yes |
| Step 4 fix made band-0 = -0.814 Ha (regression from -1.046) | This session, test run with fix | **DERIVED** | N/A — fix reverted |

## Key empirical findings

1. **Step 4 S⁻¹·R₂ fix is wrong for our implementation**: Applying S⁻¹·R₂ in
   reconstruction degrades band-0 from -1.046 Ha → -0.814 Ha. The Das Alg 3
   formula assumes different metric convention (L2 Gram-Schmidt). Our S-inner-
   product Gram-Schmidt makes S⁻¹·R₂ redundant/harmful.

2. **Per-band Σ|c_G|² diagnostic reveals correct USPP normalization**: Low bands
   (Cu 3d at bands 2-10) have Σ|c_G|² = 0.22-0.36, meaning ~64-78% of their
   S-norm comes from USPP augmentation. This is physically correct.

3. **The filter code is correct** — iter-1 band-0 within 0.0092 Ha of CASTEP.
   The 0.70 Ha iter-2 overshoot (after §11a fix) is NOT from a filter formula bug.

4. **The density code is provably correct** for CASTEP wavefunctions
   (same-input experiment ratio 0.0084%). If our RR wavefunctions produce
   a different density split, the RR wavefunctions genuinely differ from
   CASTEP's — the density code computes the correct density FOR those wavefunctions.

## Remaining hypotheses

1. **Iter-2 V_eff drifts enough to change D-screening significantly**, which
   changes the filter operator (different S⁻¹·H), producing different eigenvectors
   at iter-2. This is expected SCF transient behavior, not a bug.

2. **Occupation differences at iter-2** (different eigenvalues → different
   occupation profile via erfc smearing) contribute to the density split
   discrepancy. The same-input experiment used CASTEP eigenvalues (converged),
   not our iter-2 eigenvalues.

3. **Density mixing on soft-only density** may cause spatial inconsistency
   between the mixed soft density and fresh aug density.

## Next steps

Run `issue_11a_iter2_lastband_does_not_overshoot` with `scf_diag` to capture
the actual DensitySplit diagnostic at iter-2. This will show whether the
soft/aug split is within F8 ratios or not.
