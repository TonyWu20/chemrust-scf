# Anchor Criteria: SCF Diverges After Iter-2

## Fixture Files (EXTERNAL anchors)

- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.castep`
  — CASTEP run output. Reference total energy.
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.castep_bin`
  — Binary cell + density (raw ρ × Ω) on wave grid.
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.den_fmt`
  — Formatted density on fine grid.
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.pot_fmt`
  — Formatted V_eff on fine grid (Hartree).
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands`
  — Eigenvalues per band (Hartree).
- `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.check`
  — Wavefunction set (S-orthonormal under USPP S).

## Success Criteria (anchored EXTERNAL only)

### C1 — total energy fixed-point
`run_scf(state_from_fixture, 8, 1e-8)` returns total_energy such that
|computed_eV - REFERENCE_ENERGY_EV| < TOLERANCE_EV.
(Source: `tests/fixtures/cu111_co.rs:25` `REFERENCE_ENERGY_EV =
-24110.96665069`, TOLERANCE_EV = 2e-4.)

### C2 — eigenvalue first band fixed-point
After iter-1 (driven by fixture density), `eigenvalues[0]` must satisfy
|ε_0 - (-1.05502287)| < 0.05 Ha.
(Source: `Cu111_CO.bands`, first eigenvalue of k-point 1, spin component 1.)

### C3 — V_eff pointwise matches reference V_eff
After iter-1, the assembled V_eff on the fine grid must satisfy
||V_eff_built - V_eff_ref||_∞ < 0.05 Ha at every grid point.
(Source: `Cu111_CO.pot_fmt`, full 54×90×90 array.)
**Diagnostic ratio**: typical wrong-V_eff (deep V_loc wells alone) is
~30-40 Ha range vs reference 8.69 Ha range — 3-5× discriminator.

### C4 — ρ_aug spatial distribution matches CASTEP augmentation
For iter-2's ρ_aug computed from iter-1's ψ:
||ρ_aug_built - (ρ_castep_total - ρ_PW_built)||_∞ < 1e-3 × max(ρ_castep_total).
(Source: `Cu111_CO.den_fmt` total ρ minus our ρ_PW_built smooth part.)
**Diagnostic ratio**: wrong spatial distribution can give correct integral
but ∞-norm off by 50% of max.

### C5 — fixed-point self-consistency
For iter-2 driven by iter-1's output: ρ_iter2_total = ρ_PW_iter2 + ρ_aug_iter2
must satisfy ||ρ_iter2 - ρ_iter1||_2 / ||ρ_iter1||_2 < 1e-6 (fixed-point
condition: starting from the converged state, the SCF must reproduce it).
(Source: `Cu111_CO.castep_bin` density as ρ_iter1 reference.)
**Discriminator**: a self-consistent fixed point reproduces the input
density to machine precision; any deviation > 1e-4 indicates a bug.

### C6 — Lanczos b_up stability
For nearly-identical V_eff (within 0.5 Ha range delta), Lanczos b_up
estimate should not change by more than 2×. Concretely: |b_up_iter2 -
b_up_iter1| / b_up_iter1 < 0.5.
Anchor: with V_eff range 8.69 → 8.55 Ha (1.5% change), b_up should remain
near 22.8 Ha, not jump to 107.5 Ha (5× change).
(Source: physical principle — H operator norm is approximately
||T||_∞ + max(V_eff) - min(V_eff) + ||V_NL||; small V_eff change
should give small ||H|| change. Derived from H = T + V_loc + V_NL.)

## What we cannot anchor

- per-iteration ψ direction (gauge freedom: ψ is determined up to U(N_b)
  unitary in degenerate subspaces). Cannot directly compare to CASTEP ψ.
- per-iteration energy decomposition (E_H, E_xc separately). CASTEP does
  not export these.

## Decision

C5 (fixed-point self-consistency) is the strongest criterion. If iter-2's
density doesn't reproduce iter-1's, the bug is in either ρ_PW or ρ_aug
construction. C4 isolates ρ_aug pointwise (vs fixture ρ_aug = ρ_total -
ρ_PW). C6 is the smoking-gun signal at Lanczos level.

The Loose-then-Tighten cycle in Step 7 will check C4 first (most direct
test of ρ_aug spatial distribution).
