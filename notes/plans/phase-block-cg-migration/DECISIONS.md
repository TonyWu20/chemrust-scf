# Phase-0 Design Decisions

**Date**: 2026-05-26  
**Phase**: Block CG Migration — Phase-0 (Algorithm De-risking)  
**Status**: Grilling in progress

## Context

Phase-0 implements CPU-only, serial, band-by-band CG to verify the algorithm reaches CASTEP precision before committing to GPU batching work (Phase-1). This is a de-risking phase: if the algorithm itself cannot reach CASTEP precision, there's no point in GPU optimization.

## User-Confirmed Decisions

### D1: Fixture Files
**Decision**: Use Cu111_CO .check file at `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`

The .check file contains converged wavefunctions and eigenvalues. V_eff comes from `.pot_fmt` in the same directory.

**Files**:
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.check` (155 MB)
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.pot_fmt` (17 MB)

### D2: Gate 1 Metric
**Decision**: Eigenvalue difference: max|ε_out - ε_in| < 1e-10 Ha

Gate 1 is a consistency check: starting from CASTEP's converged ψ + V_eff, one CG iteration should return ψ within 1e-10 of input. "Within 1e-10" means the maximum eigenvalue difference across all bands.

**Rationale**: Eigenvalues are the Rayleigh quotients ⟨ψ|H|ψ⟩/⟨ψ|S|ψ⟩. If the CG algorithm is correct, applying it to an already-converged state should produce negligible eigenvalue drift.

### D3: Gate 2 Scope
**Decision**: Wavefunction only (eigenvalue follows)

Gate 2 requires band-0 to converge to within 1e-6 Ha of CASTEP A1 (−1.05502287 Ha). The primary check is wavefunction overlap with CASTEP's band-0. If the wavefunction converges (high overlap, e.g., > 0.999), the eigenvalue will automatically be correct.

**Rationale**: The eigenvalue is a functional of the wavefunction. Checking wavefunction convergence is the stronger condition.

### D4: Preconditioner Implementation
**Decision**: Create new UsppPreconditioner module

Add `src/eigensolver/uspp_preconditioner.rs` as a separate implementation. Keep `TpaPreconditioner` unchanged for future reference.

**Rationale**: The USPP-aware preconditioner P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹ is structurally different from diagonal-only TPA. A separate module keeps the implementations clean and allows side-by-side comparison.

### D5: CPU Execution Strategy
**Decision**: Use chemrust-hamiltonian as-is (may use GPU internally)

**Clarification from user**: "chemrust-hamiltonian does not have GPU code"

Phase-0 will call chemrust-hamiltonian's existing functions. Since chemrust-hamiltonian is CPU-only, this naturally satisfies the "CPU-only" requirement.

### D6: Line Search Implementation
**Decision**: Full CASTEP line search (closed-form quadratic)

Implement the exact closed-form 2×2 quadratic line search from CASTEP `electronic.f90:9959-10185` (two roots, pick the one with smaller energy).

**Rationale**: This is the authoritative reference. Simplifying the line search would introduce an unnecessary variable in the de-risking experiment.

### D7: Convergence Strategy
**Decision**: Early exit on convergence (max 50 steps)

Check residual norm after each CG step and exit early if it drops below the convergence threshold. Report the number of steps taken. Maximum 50 steps as specified in Gate 2.

**Rationale**: Early exit provides diagnostic information (how many steps were actually needed) and avoids wasting computation on already-converged bands.

### D8: Test Structure
**Decision**: Unit tests + integration tests

Write unit tests for individual CG components (direction update, line search, residual computation) plus integration tests for the full gates.

**Rationale**: Unit tests allow debugging individual components in isolation. Integration tests verify the full algorithm against CASTEP reference data.

## Resolved Questions

### Q1: chemrust-hamiltonian API for CPU H-apply
**Status**: RESOLVED

chemrust-hamiltonian-core provides CPU functions:
- `apply_full_hamiltonian(wave_g, fft_indices, gcart, k_cart, v_eff, gvg, vnl_psi_g)` → `Vec<Complex64>`
- `apply_local_hamiltonian(wave_g, fft_indices, gcart, k_cart, v_eff, gvg)` → `Vec<Complex64>` (T + V_loc)
- `apply_nlpot(beta_g_ion, beta_phi_ion, d_matrix, band_idx, vnl_psi_g)` → accumulates V_NL
- `compute_s_overlap_matrix(wave_block, pots, cell, gvg_wave, k_cart)` → `Mat<Complex64>` (S-overlap)
- `inner_product(psi1, psi2)` → `Complex64`

All functions operate on `&[Complex64]` host arrays. Phase-0 will use these directly.

### Q2: R Matrix Precomputation
**Status**: RESOLVED

**Decision**: Precompute R in setup (cached)

R = (−Q⁻¹ − C)⁻¹ will be precomputed once in `UsppPreconditioner::new()` and stored in the struct. The apply function will use the cached R.

**Linear algebra**: Use `faer` for small dense matrix inversion (already a dependency via chemrust-hamiltonian).

### Q3: Convergence Threshold
**Status**: RESOLVED

**Decision**: Residual norm: ‖r‖_S < 1e-6 Ha

Convergence criterion is residual norm in the S-metric: ‖r‖_S = sqrt(⟨r|S⁻¹|r⟩) where r = H|ψ⟩ − ε·S|ψ⟩. This matches CASTEP's criterion.

### Q4: S-orthogonalization
**Status**: RESOLVED

**Decision**: S-orthogonalize once at convergence

After each band converges, S-orthogonalize it against all previously converged bands once. No S-orthogonalization during the CG loop. This matches CASTEP's "lower-only" default.

### Q5: Initial Guess for Gate 2
**Status**: RESOLVED

**Decision**: Use pseudoatomic SCF from chemrust-hamiltonian's Hubbard U module

The user clarified: "chemrust-hamiltonian ports the pseudoatomic SCF of CASTEP in its Hubbard U module. That should be also how CASTEP generates initial guess too?"

Phase-0 will use the pseudoatomic SCF initial guess, not random coefficients. This is more realistic and matches CASTEP's actual initialization.

## Implementation Plan (Draft)

### Files to Create

1. **`src/eigensolver/band_cg.rs`** (~300 lines)
   - `band_cg_minimize_single_band()` — main CG loop for one band
   - Polak-Ribière direction update
   - Residual computation
   - Convergence check

2. **`src/eigensolver/uspp_preconditioner.rs`** (~150 lines)
   - `UsppPreconditioner` struct (holds R matrix, TPA cache)
   - `precompute_r_matrix()` — R = (−Q⁻¹ − C)⁻¹
   - `apply()` — P⁻¹·r = T⁻¹·r + T⁻¹·(β·R·β†·T⁻¹·r)

3. **`src/eigensolver/line_search.rs`** (~100 lines)
   - `line_search_2d_quadratic()` — closed-form 2×2 line search
   - Mirrors CASTEP `electronic.f90:10046-10125`

4. **`tests/phase0_cg_gates.rs`** (~200 lines)
   - `gate1_consistency_check()` — converged state → 1 CG iter → eigenvalue drift < 1e-10 Ha
   - `gate2_convergence_from_random()` — random ψ → CG → band-0 converges within 50 steps

5. **`tests/phase0_cg_units.rs`** (~150 lines)
   - Unit tests for direction update, line search, residual computation

### Dependencies

- **chemrust-hamiltonian-core**: CPU H-apply, S-apply (need to verify API)
- **ndarray** or **faer**: Small dense matrix inversion for R matrix
- **rand**: Random wavefunction initialization for Gate 2

## Next Steps

1. Read chemrust-hamiltonian-core API to find CPU H/S apply functions
2. Specify R matrix precomputation strategy
3. Clarify convergence threshold metric
4. Specify S-orthogonalization strategy for serial band-by-band
5. Specify random initialization for Gate 2
6. Write forensic TASKS.md with anchored success criteria
