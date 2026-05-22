# Resolution: missing-spin-deg-rho-nm

**Symptom**: SCF diverges after iter-2 with D_screened exploding to 413–658 Ha on Cu ions.

**Root cause**: The spin_deg fix was incorrect for this codebase. Our `compute_occupations` uses `erfc((e-μ)/w)` which returns values in [0,2] — spin degeneracy is already encoded in the occupations. CASTEP's `ion.f90:7114` multiplies by `2.0` because its `occ ∈ [0,1]` per spin channel. Adding `spin_deg=2.0` on top of our occupations doubled ρ_aug, breaking the correct total density.

**Evidence from fixture**:
- CASTEP `den_fmt` total sum = 8.1356e7 = N_e × N_fine = 186 × 437400
- Pre-fix: rho_PW(2.02e7) + rho_aug(6.12e7) = 8.14e7 ≈ CASTEP total ✓
- Post-fix: rho_PW(2.02e7) + rho_aug(1.22e8) = 1.43e8 ≠ CASTEP total ✗

**Fix location**: Reverted `spin_deg=2.0` from both:
- `src/density.rs:425-437` (CPU path, `compute_aug_density_fine`)
- `src/density.rs:504-518` (GPU path, `compute_aug_density_gpu`)

**Fix description**: Removed the `spin_deg * acc` multiplication. The ω accumulation is:
`ω_{nm} = Σ_b occ[b] * β_ψ[n,b]* · β_ψ[m,b]` with `occ[b] ∈ [0,2]`.

**Anchor criteria used**:
- `Cu111_CO.den_fmt` total sum = 8.1356e7 = N_e × N_fine (EXTERNAL)
- Pre-fix total density matched CASTEP; post-fix did not (EXTERNAL comparison)

**Prior notes reclassified**:
- "CASTEP ion.f90:7114 weighting=2.0*occ" — EXTERNAL but inapplicable: CASTEP's occ ∈ [0,1], ours ∈ [0,2]. The factor is already in our occupations.
- "hamiltonian-core accumulate_density_matrix uses spin_deg*weight_k" — EXTERNAL but that function reads from `.check` file occupancies which are in [0,1] CASTEP convention, not our erfc occupations.
- D_screened explosion (5.82 → 49.97 → 318.94 Ha) — was caused by the incorrect spin_deg fix doubling ρ_aug.

**Stale cache**: `/tmp/cu111_co_rho_aug_cpu.bin` must be deleted again after this revert (it was rebuilt with the wrong spin_deg=2 values).

**Date**: 2026-05-22

---

## Post-resolution followup: density "normalization bugs" reclassified (2026-05-22)

The open-followups.md §9 claimed two independent "normalization bugs" in the
density construction code (rho_PW 32.6% too small, rho_aug 19% too large).
A controlled same-input experiment resolved this:

**Feed CASTEP's own converged wavefunctions + eigenvalues through our density
code** → the component sums match CASTEP F8 dumps to within 0.0084%.
See `tests/ca_scf_convergence.rs::density_decomp_matches_castep_f8_same_inputs`.

**Conclusion**: The density code is correct. The 32.6%/19% discrepancy in the
original report was from comparing our iter-2 wavefunctions (different
eigenvectors after Rayleigh-Ritz) against CASTEP's independently converged
state. It is a wavefunction accuracy issue, not a density normalization bug.

**Prior notes reclassified**:
- "rho_PW is 32.6% too small — bug in `construct_density_gpu`" — DERIVED:
  based on comparison against different-wavefunction state
- "rho_aug is 19% too large — bug in ω accumulation or Q_nm normalization" —
  DERIVED: same reason
- "Two independent normalization bugs remain open" — HYPOTHESIZED:
  asserted without same-input validation; refuted by the controlled experiment

