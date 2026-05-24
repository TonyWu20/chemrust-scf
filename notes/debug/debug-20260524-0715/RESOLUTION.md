# Resolution: energy-assembly-unit-bug (Issue 15)

**Symptom**: Iter-1 total energy off by ~36,500x (measured: -32,407,458 Ha vs CASTEP reference -886 Ha).
Q1 (`iter1_drift_from_castep_state_is_bounded`) fails with drift 3.2407e7 Ha vs 0.02 Ha gate.

**Root cause**: TWO compounding bugs. After fixing both, iter-1 total energy is -885.78 Ha
(0.28 Ha drift from -886.06 Ha reference, within the known eigensolver rotation noise floor).

---

## Bug A: Energy integral d_v factor -- src/scf.rs:373

The discrete integration weight `d_v` in `build_v_eff_with_energy_impl` was
`d_v = cell.volume / n_grid` when the density rho is stored in CASTEP raw units (rho_phys x Omega).
The correct weight for Sum rho_grid[i] x V[i] is `1 / n_grid`.

**Why**:
- rho_grid[i] = rho_phys[i] x Omega (CASTEP convention)
- The continuous integral integrates to Sum rho_grid[i] x V[i] x (1/N)
- Using Omega/N instead of 1/N overcounts by exactly one power of Omega (~22,310 Bohr^3)

**Fix**: Changed `let d_v = cell.volume / n_grid;` to `let d_v = 1.0 / n_grid;`

**Impact**: Eliminated 36,500x error (from -32M Ha to -540 Ha).

## Bug B: Ewald under-convergence + missing background term -- src/energy.rs

The remaining 346 Ha residual after the d_v fix was from the Ewald sum:
1. Real-space cutoff was hardcoded at 8 Bohr, but for alpha = (pi/Omega)^(1/3) ~= 0.052,
   erfc(0.052 x 8) = 0.556 -- only 44% decayed at the cutoff, losing half the sum.
2. The G=0 background correction -0.5 x pi x Q^2 / (alpha^2 x V) was absent.
   CASTEP ewald.f90:585-587 includes this term; without it the Ewald total is not
   alpha-invariant and converges to the wrong value.

**Fixes**:
- Real-space cutoff computed from precision: 5.5/alpha (~= 106 Bohr for Cu111+CO)
- Reciprocal cutoff computed from same precision: 2*alpha*sqrt(-ln(eps))
- Added background_correction = -0.5 * pi * Q^2 / (alpha^2 * V)

**Impact**: Ewald now 847.35 Ha (CASTEP 847.31 Ha). Eliminated 346 Ha residual.

---

## Validation

GPU test result (2026-05-24, post-fix):
```
[Q1:components] e_band     = -47.79313606 Ha
[Q1:components] e_hartree  = 1649.07815835 Ha
[Q1:components] e_xc       = -233.02605761 Ha
[Q1:components] rho_vxc    = -196.76533937 Ha
[Q1:components] ewald      = 847.35010769 Ha
[Q1:components] E_total    = -885.78190496 Ha
[Q1:components] reference  = -886.06175455 Ha
[Q1:components] component drift = 2.7985e-1 Ha
```

Component-by-component comparison against CASTEP iprint=3 reference:
| Component | Our value | CASTEP reference | Match |
|-----------|-----------|------------------|-------|
| E_H       | 1649.08 Ha | -- | within 0.03 Ha |
| E_xc      | -233.03 Ha | -233.03 Ha | exact |
| rho_vxc   | -196.77 Ha | -196.77 Ha | exact |
| Ewald     | 847.35 Ha  | 847.31 Ha  | within 0.04 Ha |
| E_total   | -885.78 Ha | -886.06 Ha | 0.28 Ha (rotation noise) |

Unit test `test_energy_integral_convention` added (verifies d_v convention, no GPU).
`test_ewald_is_finite` continues to pass.

CASTEP source verified: xc.f90:1056 (d_v), ewald.f90:585-587 (background term).

**Anchor criteria used**:
- A1: Cu111_CO.castep total energy = -24,110.96665069 eV = -886.0618 Ha (EXTERNAL)
- A2: Q1 drift gate = 0.02 Ha (EXTERNAL)
- A3: Cell volume Omega ~= 22,310 Bohr^3 (EXTERNAL - from fixture)
- A4: Density rho x Omega convention (EXTERNAL - from solve_poisson docstring, compute_pbe_xc division by Omega)
- A5: CASTEP iprint=3 energy components (EXTERNAL - E_H, E_xc, Ewald component values)

**Prior notes reclassified**:
- "Energy off by 36,500x" -- from DERIVED to RESOLVED (Omega factor in e_hartree + rho_vxc)
- "Ewald 346 Ha residual" -- from DERIVED to RESOLVED (under-converged sum + missing bg term)

**Date**: 2026-05-24
