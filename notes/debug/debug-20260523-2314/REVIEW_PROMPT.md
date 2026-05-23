# SCF Cascade Root-Cause Analysis — Request for Review

Project: `/home/tony/programming/chemrust-scf` (CASTEP-compatible DFT in Rust)
Key files: `tests/fixtures/cu111_co.rs`, `src/scf.rs`, `tests/ca_scf_convergence.rs`

## Symptom

Starting from CASTEP's converged wavefunctions and soft density for Cu111+CO,
our SCF cascades:

| Iter | band-0 (Ha) | Status |
|------|-------------|--------|
| 1    | −1.046      | matches CASTEP (−1.055) |
| 2    | −0.869      | drifting |
| 3    | −11.94      | catastrophic |

## Evidence (four tests, all against real CASTEP fixture data)

**T1** `h_on_castep_psi_matches_bands`: Apply our H operator to CASTEP's exact ψ
using V_eff built from CASTEP's soft density. RMS over 160 bands = 0.0046 Ha,
max = 0.019 Ha. **PASS** — H operator is correct.

**T2** `d_screened_matches_castep_dump_on_castep_veff`: Compare our D_screened
(D0 + ∫Q·V_eff) element-by-element against CASTEP's `D_band_debug.dat` dump.
Max per-element |Δ| = 2.13 Ha (threshold 5e-4 Ha). **FAIL** — D-screening has
per-element errors. (Note: normalization factor 1/N verified correct via raw-sum
diagnostic; discrepancies concentrated on diagonal elements, position-dependent.)

**T3** `cascade_with_castep_veff_substitution`: After iter-1, inject CASTEP's
converged V_eff (from `.pot_fmt`) before iter-2's diagonalization. Iter-2
band-0 = −1.0452 Ha (|Δ| = 0.0098 Ha). **PASS** — cascade stops with
CASTEP V_eff.

**T4** `cascade_with_castep_density_substitution` (with `clear_density_aug_fine`):
After iter-1, inject CASTEP soft density AND clear stale ρ_aug. Iter-2 band-0 =
−1.0452 Ha. **PASS**. Without clearing aug: iter-2 band-0 = −3.92 Ha (**FAIL**).

## Proposed Root Cause

`build_scf_state` at `tests/fixtures/cu111_co.rs:169-172` loads CASTEP's soft
density from `.castep_bin` but leaves `density_aug_fine = None`.

This creates a **V_eff formula discontinuity** between iterations:

- **Iter-1**: `density_aug_fine = None` → V_eff = V_H[ρ_soft] + V_ion +
  V_xc[ρ_soft + ρ_core]. No aug contribution.

- **Iter-2**: After `construct_density_off` runs, `density_aug_fine =
  Some(ρ_aug)` → V_eff = V_H[ρ_soft + ρ_aug] + V_ion + V_xc[ρ_soft + ρ_aug +
  ρ_core]. Aug suddenly appears.

The effective Hamiltonian formula changes between iter-1 and iter-2 (adding
ρ_aug to the Hartree and XC potentials). This systematic shift — independent of
ψ accuracy — changes which subspace the Chebyshev filter amplifies, causing the
cascade. The aug density at iter-2 comes from our (rotated) ψ, not CASTEP's,
making the shift worse.

A secondary factor: T2 shows D-screening (∫Q·V_eff) has element-by-element
errors vs CASTEP. These errors in V_NL = β·D·β† cause additional ψ rotation
within degenerate manifolds.

## Proposed Fix

During initialization, compute density fresh from CASTEP wavefunctions rather
than loading CASTEP's mixed soft density from `.castep_bin`:

1. Load ψ from `.check` (already done)
2. Compute occupations from CASTEP eigenvalues (loaded from `.bands`)
3. Compute soft density from |ψ|² on wave grid — E2 proves ratio 1.000000 vs CASTEP
4. Compute β·ψ projections and aug density on fine grid — E2 proves ratio 1.000084
5. Store both soft and aug density in the initial state

This populates `density_aug_fine` from the start, so iter-1 and iter-2 use the
same V_eff formula.

## Questions

1. Does this root-cause analysis hold up against the four test results?
2. Does the "V_eff formula discontinuity" hypothesis explain why T3 works
   (CASTEP V_eff was built WITH aug, matching iter-2's formula)?
3. Does T2's D-screening discrepancy need to be fixed first, or is it a separate
   issue that doesn't directly cause the cascade?
4. Is computing density from CASTEP ψ at initialization the correct fix, or is
   there a simpler approach?

## Reference code locations

- `build_scf_state`: `tests/fixtures/cu111_co.rs:139-208`
- `build_v_eff_with_energy_impl`: `src/scf.rs:342-384`
- `build_v_eff_with_energy`: `src/scf.rs:411-427`
- `compute_density_from_wavefunctions`: `src/scf.rs:748-897`
- `into_phase` (preserves `density_aug_fine`): `src/scf.rs:241-270`
- `apply_h_components_for_test`: `src/scf.rs:691-732`
- `compute_screened_d`: `chemrust-hamiltonian-core/src/nlpot.rs:368-411`
- CASTEP reference: `nlpot.f90:531-544` (D dump), `ion.f90:6319-6441` (D-screening integral)
- Debug session: `notes/debug/debug-20260523-2314/`
