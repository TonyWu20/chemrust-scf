# Anchor Criteria: Issue #11a Per-Band Eigenvalue Branches

## Fixture Files

- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands` — CASTEP reference eigenvalues (160 bands, Hartree)
- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.pot_fmt` — CASTEP reference V_eff on fine grid
- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.castep_bin` — CASTEP binary output with density, cell, eigenvalues
- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check` — CASTEP checkpoint with wavefunctions

## Success Criteria (EXTERNAL anchors only)

### SC-1: Band-0 eigenvalue at iter-1 matches CASTEP reference
- **Assertion**: `|eigenvalues[0] - (-1.05502343)| < 0.05` Ha
- **Source**: `Cu111_CO.bands` line 12 (first eigenvalue after "Spin component 1" header)
- **Discriminator ratio**: correct ≈ -1.055 Ha, typical wrong (from §11 table) ≈ -0.864 Ha → ratio ≈ 1.2×
- **Note**: This is a boundary-value criterion (ratio < 2×). Tightening to 0.05 Ha tolerance provides a clear pass/fail signal while allowing for minor numerical differences in the SCF convergence path.

### SC-2: Last-band eigenvalue at iter-1 matches CASTEP reference
- **Assertion**: `|eigenvalues[159] - 0.11531044| < 0.05` Ha
- **Source**: `Cu111_CO.bands` last line (160th eigenvalue, 0-indexed as 159)
- **Discriminator ratio**: correct ≈ 0.115 Ha, typical wrong (from §11 table) ≈ 1.952 Ha → ratio ≈ 17× ✓

### SC-3: Iter-2 eigenvalues do not diverge from iter-1
- **Assertion**: `|eigenvalues_iter2[0] - eigenvalues_iter1[0]| < 0.20` Ha
- **Rationale**: If iter-1 matches CASTEP (SC-1), then iter-2 should remain close to iter-1 as the SCF converges. The prior investigation showed iter-1 band-0 = -1.046 Ha, iter-2 band-0 = -0.864 Ha (drift = 0.18 Ha). A threshold of 0.20 Ha allows for this observed drift while catching catastrophic divergence (e.g., iter-3 band-0 = -12.68 Ha would fail).
- **Source**: Derived from CASTEP convergence behavior — eigenvalues should monotonically approach the reference, not oscillate wildly.
- **Classification**: This is a DERIVED criterion (based on our iter-1 output), but it's the only way to test the "per-band branches cause drift" hypothesis without running a full multi-iteration SCF to convergence.

### SC-4: Last-band eigenvalue at iter-2 does not overshoot by > 1.0 Ha
- **Assertion**: `|eigenvalues_iter2[159] - 0.11531044| < 1.0` Ha
- **Source**: `Cu111_CO.bands` last line, with tolerance chosen to catch the observed 1.95 Ha overshoot (from §11a table) while allowing for reasonable SCF drift.
- **Discriminator ratio**: correct ≈ 0.115 Ha, wrong ≈ 1.952 Ha → ratio ≈ 17× ✓

## V_eff Range Criterion (Deferred)

The prior investigation claimed "Reference V_eff range = 8.69 Ha" but did not cite a specific field in `.pot_fmt`. The `.castep` file does not contain a "V_eff range" summary statistic. 

**Action required**: Compute V_eff range from the `.pot_fmt` fixture file directly (max - min of the 3D array) to establish the EXTERNAL anchor. Until then, this criterion is DERIVED and inadmissible.

## Limitations

1. **Iter-2 and iter-3 criteria are DERIVED**: We cannot establish EXTERNAL anchors for iter-2/iter-3 eigenvalues because CASTEP does not output per-iteration eigenvalues — only the final converged values. The best we can do is assert that iter-2 remains close to iter-1 (which we've anchored to CASTEP via SC-1/SC-2).

2. **V_eff range needs verification**: The "8.69 Ha" claim needs to be verified by reading the `.pot_fmt` fixture and computing `max(V_eff) - min(V_eff)` directly.

3. **Per-band machinery hypothesis is HYPOTHESIZED**: The claim that "per-band branches introduce numerical weak points" is an inference from the empirical observation (64% improvement when disabled), not a verified fact. The success criteria above test the *symptom* (eigenvalue drift), not the root cause.

## Next Steps

1. Write a diagnostic script to compute V_eff range from `.pot_fmt` and verify the 8.69 Ha claim.
2. Write tight tests for SC-1 through SC-4.
3. Confirm the tight tests fail on the current code (with per-band branches enabled).
4. Implement the fix (disable per-band branches by always passing `eigenvalues=None` to the filter).
5. Confirm the tight tests pass after the fix.
