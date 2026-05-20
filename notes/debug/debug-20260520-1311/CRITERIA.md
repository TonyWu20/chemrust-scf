# Anchor Criteria: eigensolver-eigenvalue-accuracy

## Fixture Files

- `Cu111_CO.bands` — CASTEP converged Kohn-Sham eigenvalues (1 k-point, 160 bands) in Hartree, text format
  - Band 1 (deepest): -1.05502287 Ha
  - Band 2 (Cu semi-core): -0.49721145 Ha
  - Band 3 (Cu semi-core): -0.48877156 Ha
  - Band 93 (highest occupied, 186 e⁻ with 2 e⁻/band): ~ -0.12258940 Ha
  - Band 94 (lowest unoccupied): ~ -0.12245575 Ha
  - Fermi energy: -0.122443 Ha (from .bands header)
  - Band 160 (highest computed): 0.11244608 Ha

- `Cu111_CO.pot_fmt` — CASTEP converged V_eff on fine grid, text format
  - Used as ground truth V_eff (not our own VEffBuilder assembly)

- `Cu111_CO.castep` — CASTEP output, contains reference total energy
  - Total energy: -24110.96665069 eV

- `Cu111_CO.check` — CASTEP converged wavefunctions + fine_grid metadata

## Success Criteria

### C1: Single-iteration eigenvalue accuracy with reference V_eff
Given CASTEP's converged .pot_fmt V_eff and screened D matrices from reference occupations:
- RMS eigenvalue error across all 160 bands < 1.0 Ha
  (Source: CASTEP `.bands` file, cross-validated against `.pot_fmt`)
- Per-band error for band 1 < 0.1 Ha (value: -1.05502287 Ha)
  (Source: CASTEP `.bands` line 12)
- Per-band error for band 93 (Fermi level) < 0.1 Ha
  (Source: CASTEP `.bands` — last occupied state)
- Eigenvalue ordering matches reference (monotonically increasing)
  (Source: CASTEP `.bands`)

### C2: Single-iteration eigenvalue accuracy with our V_eff
Given our VEffBuilder's assembled V_eff (known residual ~0.27 Ha RMS vs .pot_fmt):
- RMS eigenvalue error across bands 1-160 < 2.0 Ha
  (Source: extrapolated from C1 + known V_eff residual)

### C3: Total energy convergence
- Converged SCF total energy within 2e-4 eV of -24110.96665069 eV
  (Source: CASTEP `.castep` line 326)

## Non-Criteria (explicitly excluded)

The following are DERIVED and NOT admissible as criteria:
- "Band 1 should be -1.60 Ha" — this was our buggy pipeline output, not a target
- "RMS error should be < 12 Ha" — 12 Ha was the pre-fix error, not a threshold
- "Total energy should be -10192 eV" — this was our buggy pipeline output
