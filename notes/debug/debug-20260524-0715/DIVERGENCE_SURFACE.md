# Divergence Surface: energy-assembly-unit-bug

## Generic Surface Categories

### 1. Normalization / scaling conventions — **TARGET**
- **Issue**: `d_v = cell.volume / n_grid` multiplies the e_hartree and rho_vxc sums by Ω
- **Evidence**: ρ is in ρ×Ω convention (CASTEP raw). The correct discrete integration weight for `e_hartree = 0.5 × ∫ ρ_phys V_H d³r` is `1/N`, not `Ω/N`. See `CRITERIA.md` A4.
- **Classification**: **TO BE TESTED** (this is the primary hypothesis)

### 2. Data layout / axis ordering — **RULED OUT**
- `solve_poisson` and VEffBuilder handle all FFT layout internally
- Energy computation is a simple pointwise zip-over-iterators, no indexing sensitivity
- Classification: Ruled out by code inspection

### 3. Sign / direction conventions — **RULED OUT**  
- Energy formula signs verified against `assemble_total_energy` unit test (`energy.rs:317-328`)
- CASTEP formula: `E_total = E_band - E_hartree + E_xc - ∫ρV_xc + E_ewald`
- Classification: Ruled out by unit test

### 4. Boundary / edge-case handling — **RULED OUT**
- No conditionals in the energy sum
- Classification: Ruled out by code inspection

### 5. Unit conversion at any boundary — **RULE NOT YET ASSESSED**
- `e_hartree` and `rho_vxc` are computed on the fine grid
- `v_h` from `solve_poisson` is in Hartree (confirmed by `poisson.rs:14` docstring)
- `v_xc.v_xc` is in Hartree (confirmed by `xc/pipeline.rs:184`)
- Classification: Ruled out by reading API docs

### 6. Parser precision / offset assumptions — **RULED OUT**
- Density is loaded directly from `.castep_bin` without transformation
- Classification: Ruled out by `build_scf_state` code

### 7. Diagnostic comparison code — **RULED OUT**
- The energy comparison is a simple subtraction, no intermediate analysis
- Classification: Ruled out

## Project-Specific Items

### 8. Density convention mismatch between smooth PW and augmentation
- Both ρ_PW and ρ_aug use the same ρ×Ω convention
- Their sum feeds `rho_total_fine` which flows into the dot product
- Classification: Ruled out (single convention throughout)

### 9. XC energy returned by `compute_pbe_xc`
- `e_xc` is computed as `energy_density.iter().sum::<f64>() * volume / total_points`
- The kernel receives ρ_phys (after ÷Ω in the kernel call), so e_xc is in Hartree
- Classification: Ruled out by reading `xc/pipeline.rs:159`
