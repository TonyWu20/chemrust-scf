# Investigation: Missing spin_deg in ρ_nm accumulation

**Symptom**: `compute_aug_density_fine` (CPU path, `src/density.rs:399-441`) and
`compute_aug_density_gpu` (`src/density.rs:458-...`) both compute ω^I_{nm} as:

```rust
acc += occupations[b] * bp[[n, b]].conj() * bp[[m, b]];
```

Missing `spin_deg = 2.0` factor for nspins=1 (closed-shell), matching CASTEP
`ion.f90:7114`: `weighting = 2.0 * occ * kpoint_weights`.

## Prior numeric claims classification

| Claim | Class | Source | Admissible? |
|-------|-------|--------|-------------|
| `CASTEP ion.f90:7114 weighting = 2.0*occ*kpoint_weights` | EXTERNAL | `~/Downloads/CASTEP-6.11-nixos/Source/Fundamental/ion.f90:7114` | YES |
| `chemrust-hamiltonian-core accumulate_density_matrix uses spin_deg*weight_k` | EXTERNAL | `~/programming/chemrust-hamiltonian/chemrust-hamiltonian-core/src/augment/mod.rs:227-243` | YES |
| `spin_deg = 2.0 for nspins=1` | EXTERNAL | `ion.f90:7114` comment + `augment/mod.rs:227` | YES |
| `kpoint_weight = 1.0 for gamma-only` | HYPOTHESIZED | Gamma-point convention; not verified from fixture | No (latent, not active for Cu111_CO) |
| `D_screened explodes 5.82 → 49.97 → 318.94 Ha` | DERIVED | Prior session diagnostic output | No (but consistent with 2× ρ_aug error) |

## Classification of the finding

The finding from `chemrust-hamiltonian` exploration is:
- **EXTERNAL** for the CASTEP reference (`ion.f90:7114`) — verified directly.
- **EXTERNAL** for the hamiltonian-core reference (`augment/mod.rs:227-243`) — verified directly.
- The missing factor is **confirmed**: both `compute_aug_density_fine` and
  `compute_aug_density_gpu` lack `spin_deg` multiplication.

## Impact assessment

For nspins=1 (Cu111_CO fixture): ρ_aug is **2× too small** everywhere.
This means ∫Q·V_eff uses half the correct ρ_aug, making D_screened wrong.
The SCF divergence (D_screened 5.82 → 49.97 Ha) is consistent with a 2×
undercount in ρ_aug causing V_eff at ion centres to be wrong after iter-1.

## Scope

Both CPU and GPU paths have the same bug at the same accumulation loop.
Fix is identical: multiply `acc` by `spin_deg` before storing.
