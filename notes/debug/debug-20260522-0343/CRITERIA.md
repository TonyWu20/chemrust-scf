# Anchor Criteria: missing-spin-deg-rho-nm

## Fixture Files
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.den_fmt` — converged ρ on fine grid (includes ρ_aug)
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands` — converged eigenvalues

## Reference Implementation
- `~/Downloads/CASTEP-6.11-nixos/Source/Fundamental/ion.f90:7114` — `weighting = 2.0*occ*kpoint_weights`
- `~/programming/chemrust-hamiltonian/chemrust-hamiltonian-core/src/augment/mod.rs:227-243` — `weight = spin_deg * weight_k`

## Success Criteria

1. **Formula correctness**: `ω^I_{nm} = Σ_b spin_deg * occ[b] * β_ψ[n,b]* · β_ψ[m,b]`
   with `spin_deg = 2.0` for nspins=1.
   (Source: `ion.f90:7114`, `augment/mod.rs:227`)

2. **Discriminator**: After fix, `compute_aug_density_fine` result must equal
   `2 × (pre-fix result)` for nspins=1 — a 2× ratio is the discriminator value.
   Correct and incorrect implementations differ by exactly 2×.

3. **Regression**: `aug_density_gpu_matches_cpu_cu111_co` must still pass
   (GPU and CPU paths must agree after both are fixed).

4. **SCF stability proxy**: D_screened amax on Cu ions must remain O(D_0) ≈ 5-60 Ha
   through iter-2 (not explode to 49.97 Ha as before).
   (Source: `Cu111_CO.bands` band-1 = -1.05502287 Ha as convergence proxy)
