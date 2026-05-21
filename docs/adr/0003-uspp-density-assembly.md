# ADR-0003: USPP Density Assembly — Two-Channel Architecture

**Date:** 2026-05-21
**Status:** Accepted

## Context

Two prior bugs exposed architectural invariants that lived only in implementers'
heads:

- **§8 `inv_omega` bug:** The `accumulate_density` kernel was dividing by Ω,
  but `solve_poisson` and `compute_pbe_xc` downstream expected raw `ρ × Ω`.
  The fix (pass `inv_omega = 1.0`) was non-obvious because the unit convention
  was undocumented.

- **§1d FFT plan ordering bug:** cuFFT plan dims were passed in the wrong order
  (`(ngz, ngy, ngx)` instead of `(ngx, ngy, ngz)`), causing a silent layout
  mismatch between the scatter formula and the FFT output. The correct ordering
  is `(ngx, ngy, ngz)` (innermost first) to match `iz + ngz*(iy + ngy*ix)`.

Both bugs shared the same root cause: the augmentation density path has two
channels (smooth PW ρ and ρ_aug) with different unit conventions and different
geometry-staticity, and neither was documented.

## Decision

Smooth PW ρ and ρ_aug are maintained as **separate channels** with a defined
assembly path:

1. `construct_density_gpu` produces smooth-only ρ_PW on the wave grid
   (raw `ρ × Ω` convention, no Ω division).
2. `compute_aug_density_gpu` produces ρ_aug on the fine grid (same raw
   convention). Uses `QSfCache` (geometry-static, built once per cell) to
   avoid recomputing `Q_{nm}(G)·exp(-iG·R_I)` every iteration.
3. `build_v_eff_with_energy_impl` sums ρ_PW (upsampled to fine grid) + ρ_aug
   before Poisson + XC evaluation.

The `.castep_bin` density fixture already stores the sum of both channels.

## Rationale

The two channels have fundamentally different properties:

| Property | ρ_PW | ρ_aug |
|---|---|---|
| Grid | Wave grid | Fine grid |
| Geometry-static? | No (per-iteration) | Yes (per-cell) |
| Unit convention | raw ρ × Ω | raw ρ × Ω |
| Source | `construct_density_gpu` | `compute_aug_density_gpu` |

Keeping them separate prevents the class of bug where one channel's convention
is silently applied to the other. The `QSfCache` separation also makes the
geometry-static nature of `Q_{nm}(G)·exp(-iG·R_I)` explicit in the type
system — it is built once and reused, not recomputed per iteration.

## Consequences

- Any future refactor that merges the two channels into a single accumulation
  must explicitly handle the unit convention difference and the different
  geometry-staticity.
- The `QSfCache` must be invalidated if the cell geometry changes (ion
  positions, lattice vectors). Currently this is handled by the SCF state
  machine: `q_sf_cache` is `None` at `ScfIteration::new` and rebuilt lazily.
  A geometry-update path must reset it to `None`.
- The `inv_omega = 1.0` convention must be preserved in any new density
  accumulation kernel. Document it at the kernel call site.
