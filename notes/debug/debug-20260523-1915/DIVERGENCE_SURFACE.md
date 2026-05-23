# Divergence Surface: Soft/Aug Density Split Drift

## What the diagnostic shows
At iter-2, the density split shifts from F8 reference:
- soft_frac: 29.8% (F8: 36.8%) — PW character under-represented
- aug_frac: 70.2% (F8: 63.2%) — augmentation over-represented
- Total charge: exactly 186 e⁻ (perfect conservation)

Last 10 bands at iter-2 have Σ|c_G|² ≈ 0.20-0.21 (expected ~1.0 for high-energy
bands). These bands acquired spurious augmentation character.

## Ruled out
- **Density code normalization bug**: total e⁻ = 186.0, exact. Code matches F8
  for same-input wavefunctions (ratio 1.000000/1.000084).
- **RR code formula error**: ndeg=0 test passes (RR-only matches CASTEP within
  0.05 Ha). RR is correct on clean input.
- **Filter formula error**: iter-1 produces band-0 within 0.0092 Ha. Filter is
  correct on near-converged input.
- **Step 4 S⁻¹·R₂ fix**: empirically disproved (made eigenvalues worse). The
  Das Alg 3 formula assumes different Gram-Schmidt convention.
- **Per-band eigenvalue branches**: disabled by §11a fix (eigenvalues=None).

## Active candidates
1. **V_eff at iter-2 differs from CASTEP converged V_eff** at ion centres,
   changing D-screening → changing S⁻¹·H operator → different filtered subspace.
   This is a convergence quality issue, not a code bug.

2. **Gram-Schmidt sensitivity to input quality**: the S-inner-product Gram-Schmidt
   may produce different results with iter-2's filtered vectors vs iter-1's,
   because the augmentation character of the input vectors differs.

3. **Normalization drift in Gram-Schmidt**: the RRNorm at iter-2 shows
   Σ|c_G|² = 0.20 for last 10 bands — these bands have lost PW character.
   This might indicate the Gram-Schmidt S-norm is miscomputing the augmentation
   contribution when the filtered vectors have wrong character.

4. **Self-consistency problem**: the SCF converges to a different fixed point
   than CASTEP because our D-screening convention (via CPU FFT) may differ
   subtly from CASTEP's convention. The converged state would then have a
   different soft/aug split by definition.

## Next steps
1. Run more SCF iterations (iter-3, iter-4) to see if the split converges
   toward or away from the F8 ratio.
2. Compare Gram-Schmidt norms between iter-1 and iter-2 for specific bands.
3. Compare D_screened matrices between iter-1 (CASTEP V_eff) and iter-2 (our V_eff).
