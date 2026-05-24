# Anchor Criteria: energy-assembly-unit-bug

## Fixture Files
- `Cu111_CO.castep_bin` — cell geometry, density (wave grid), eigenvalues
- `Cu111_CO.check` — wavefunctions, fine grid definition
- `Cu111_CO.pot_fmt` — reference V_eff on fine grid
- `Cu111_CO.den_fmt` — reference density on fine grid
- `Cu111_CO.bands` — eigenvalues in Hartree (160 bands)
- `Cu111_CO.castep` — reference total energy = −24,110.96665069 eV

## External Anchors

### A1: CASTEP total energy
- Source: `tests/fixtures/cu111_co.rs:26` — `REFERENCE_ENERGY_EV = -24110.96665069`
- Origin: Cu111_CO.castep line 326
- Value: **−886.0571 Ha** (= −24110.96665069 / 27.211384569)

### A2: Q1 drift gate
- Source: `tests/fixtures/cu111_co.rs:40` — `DRIFT_TOLERANCE_HA = 2e-2`
- Origin: T3 cascade_with_castep_veff_substitution (CASTEP V_eff substitution experiment)
- Description: With correct V_eff, our pipeline produces energy within 9.8 mHa of CASTEP. The 2e-2 gate is 2× this empirical noise floor.
- Value: **drift < 0.02 Ha** after one SCF iteration

### A3: Cell volume (Ω)
- Source: `tests/fixtures/cu111_co.rs` — loaded from `Cu111_CO.castep_bin`
- The `bin.cell.volume` field is loaded from CASTEP's own `.castep_bin` file
- Value: approximately **22,310 Bohr³** (exact value loaded at runtime from fixture)

### A4: Density convention (ρ × Ω)
- Source: `solve_poisson` docstring (`poisson.rs:5-6`) — "The charge density array stores ρ(r) × V_cell"
- `compute_pbe_xc` line 152: `xc_pbe_point(*r * inv_vol, ...)` — divides input by volume before kernel
- `construct_density_gpu` uses `inv_omega = 1.0` — no Ω division
- All confirm CASTEP raw density convention

## Success Criteria

### SC1 — Correct e_hartree after fix
- After changing `d_v = 1.0 / n_grid`, per-point contribution to e_hartree should match:
  `E_H = 0.5 × (1/N) × Σ ρ_grid[i] × V_H[i]`
- The factor of Ω (~22,310) must NOT be present

### SC2 — Correct rho_vxc after fix  
- After changing `d_v = 1.0 / n_grid`, rho_vxc becomes:
  `rho_vxc = (1/N) × Σ ρ_grid[i] × v_xc[i]`
- Factor of Ω eliminated

### SC3 — Q1 passes at 20 mHa gate
- Source: A1, A2
- After fix: `iter1_drift_from_castep_state_is_bounded` should pass with drift < 20 mHa
- This is only reachable if SC1 + SC2 are correct, i.e. all energy components are in the right convention
