# Resolution: Electron Count Diagnostic Bug (§11b)

**Symptom**: Electron count diagnostic reports 4.15M e⁻ (should be 186 e⁻), total energy −881M eV (should be −24k eV)

**Root cause**: The `total_e_phys_conv` diagnostic at `scf.rs:722` (and line 486) multiplied by cell volume (Ω) when the density was already in CASTEP raw units (ρ×Ω). This applied Ω twice, producing a 22,300× error (exactly the cell volume in Bohr³).

**Fix location**: `src/scf.rs:486` and `src/scf.rs:732`

**Fix description**: Changed `total_e_phys_conv = rho_sum * Ω / n_grid` to `total_e_phys_conv = rho_sum / n_grid`

**Why the initial analysis was wrong**: The error magnitude (~6,600× in some logs) suggested a wavefunction normalization bug. But careful dataflow tracing revealed:
1. Iter-1 (which uses fixture density and never touches Gram-Schmidt) showed the same wrong electron count
2. The "raw_conv" diagnostic already gave the correct answer (186 e⁻)
3. The bug was in the diagnostic formula, not in density construction
4. The actual ratio was 22,300× (= Ω), not 6,600× (≠ (N_grid/Ω)²)

**Anchor criteria used**:
- Electron count = 186 e⁻ (from cell file: 11 Cu @ 11 e⁻, 1 C @ 4 e⁻, 1 O @ 6 e⁻)
- Diagnostic consistency: both raw_conv and phys_conv should agree (density is already in raw units)
- Iter-1 vs iter-2 consistency: both use same density convention

**Date**: 2026-05-23
