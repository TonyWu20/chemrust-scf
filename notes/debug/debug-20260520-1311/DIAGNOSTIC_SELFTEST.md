# Diagnostic Self-Test: eigensolver eigenvalue accuracy

## Diagnostic enumeration

### D1: `compare_eigenvalues_against_bands` (ca_step_validation.rs:141-188)
- Calls build_v_eff → diagonalize → compare eigenvalues with .bands
- **Status**: SKIP — requires GPU, not runnable
- **Trust**: Not yet verified

### D2: `compare_eigenvalues_with_reference_veff_and_screening` (ca_step_validation.rs:269-347)
- Injects reference .pot_fmt V_eff, uses screened D matrices, calls diagonalize
- **Status**: SKIP — requires GPU, not runnable
- **Trust**: Not yet verified

### D3: `compare_density_against_castep_bin` (ca_step_validation.rs:196-242)
- Compares constructed density against .castep_bin reference
- **Status**: SKIP — requires GPU, not runnable

### D4: `compare_v_eff_against_pot_fmt` (ca_step_validation.rs:40-88)
- Compares VEffBuilder output against .pot_fmt on CPU
- **Status**: Requires GPU (build_v_eff needs GPU for NLCC reconstruction)
- **Trust**: Not yet verified

### D5: `diagnose_veff_components` (ca_step_validation.rs:93-133)
- CPU-only, computes V_H, V_ion, V_xc individually and prints component ranges
- **Status**: Runnable on CPU — let me verify

### D6: `fixed_point_matches_castep_energy` (ca_scf_convergence.rs:29-52)
- Full SCF convergence test
- **Status**: SKIP — requires GPU

### D7: `perturbation_recovers_castep_energy` (ca_scf_convergence.rs:59-104)
- Perturbation recovery test
- **Status**: SKIP — requires GPU

## Self-test of D5 (diagnose_veff_components)

Let me run this CPU-only diagnostic and verify its output against physical intuition.

### Physical expectation for Cu111_CO:
- V_H (Hartree): ~0-1 Ha (repulsive electron-electron)
- V_ion (ionic): ~-5 to -2 Ha near Cu nuclei, ~0 far away (attractive, deep)
- V_xc (exchange-correlation): ~-1 to 0 Ha (negative, smooth)
- V_eff = V_H + V_ion + V_xc: ~-2 to -5 Ha near nuclei
- V_eff reference: from .pot_fmt, should have similar range

### Cross-path verification:
Cannot be done without writing an independent computation. The diagnostic only prints summary statistics (min, max, mean). Per the ODD pattern, summary statistics without per-point backing are inadmissible as empirical evidence. However, the component ranges themselves serve as a sanity check.

**TODO**: Write a brute-force per-point dump if GPU validation is needed after the fix.
