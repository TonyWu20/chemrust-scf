# Divergence Surface: missing-spin-deg-rho-nm

## Items

| Item | Status |
|------|--------|
| Missing `spin_deg=2.0` in CPU `compute_aug_density_fine` (`density.rs:430`) | **To fix** |
| Missing `spin_deg=2.0` in GPU `compute_aug_density_gpu` (`density.rs:507`) | **To fix** |
| Missing `kpoint_weight` multiplication (gamma-only: weight=1.0, latent) | **Ruled out by fixture** — Cu111_CO is gamma-only; weight_k=1.0 so no effect now |
| `accumulate_density_matrix` in hamiltonian-core already correct | **Ruled out** — `augment/mod.rs:227-243` confirmed correct |
| Diagnostic comparison code layout | **Ruled out** — prior session fixed col-major bp upload in test |

## Root cause

Both ω^I_{nm} accumulation loops (CPU and GPU) are missing the `spin_deg` factor.
For nspins=1 this produces ρ_aug that is 2× too small, corrupting D_screened
and causing SCF divergence after iter-1.
