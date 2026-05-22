# Anchor Criteria — Iter-1 Filter Operator Mismatch

All criteria below are EXTERNAL (admitted by `INVESTIGATION.md`). DERIVED
values from prior sessions are excluded by construction.

---

## Fixture Files

| Path | Content | Anchor for |
|------|---------|------------|
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.castep_bin` | Cell, density on wave grid, eigenvalues | Density layout, lattice |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check` | CASTEP converged wavefunctions (S-orthonormal under USPP) | Initial ψ for SCF replay |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.pot_fmt` | Reference V_eff on fine grid | V_eff range gate |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands` | Reference eigenvalues in Hartree (plain text) | **Per-band gate (SC-4)** |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.den_fmt` | Reference electron density on fine grid | Density decomposition gate |

These are CASTEP-binary outputs from a converged single-point SCF (CPU branch,
F8 build with the F8 density-augment instrumentation patch).

---

## Success Criteria

### Discriminator (the one that picks A/B/C)

| ID | Criterion | Source | Threshold |
|----|-----------|--------|-----------|
| **SC-4-tight** | iter-1 RR per-band eigenvalue matches CASTEP for the lowest 10 bands | `Cu111_CO.bands` (lines 1-10 after header) | `\|band_j_iter1 − band_j_castep\| < 0.05 Ha for j ∈ [0, 10)` |

This is the new gate added by this debug session. It supersedes the existing
single-band SC-4 (`tests/ca_scf_convergence.rs:53-80`, threshold 0.05 Ha on
band-1 only) by widening to 10 bands. Single-band agreement under one mode
could be coincidental; ten-band agreement is signal.

### Pre-existing regression gates (must not regress)

| ID | Criterion | Source | Threshold |
|----|-----------|--------|-----------|
| SC-density | density decomposition matches CASTEP F8 with same input ψ | `tests/ca_scf_convergence.rs::density_decomp_matches_castep_f8_same_inputs` | soft/aug ratios within 1% of F8 (commit `fa68980`) |
| SC-S⁻¹ | global-Woodbury inverse identity | `tests/ca_scf_convergence.rs::s_inv_s_identity_test` | `‖S⁻¹·S·ψ − ψ‖_∞ < 1e-10` (commit `f21127f`) |
| SC-Veff | iter-2 V_eff range close to iter-1's | `tests/ca_scf_convergence.rs::iter2_v_eff_range_within_one_ha_of_iter1` | `\|iter-2 range − iter-1 range\| < 1.0 Ha` |
| SC-build | build/lint sanity | `cargo check --workspace` / `cargo clippy --workspace -- -D warnings` | zero errors / zero warnings |

### Tie-break (if SC-4-tight passes for multiple modes)

| ID | Criterion | Source | Threshold |
|----|-----------|--------|-----------|
| SC-7-tight | iter-3 Cu D_screened amax stays below D_0 amax | `notes/plans/phase-rchfsi-bare-h/TASKS.md:24` | `D_screened(iter-3, Cu) < 10 Ha` (max D_0 ≈ 6.16 Ha; SC-7 was prelocked at 10 to allow some screening growth) |

If two modes pass SC-4-tight, run iter-3 and apply SC-7-tight to break the tie.

### Algorithm-level constraints (paper-anchored)

| ID | Constraint | Source | Implication |
|----|-----------|--------|-------------|
| ALG-1 | Filter must apply `S⁻¹·H` (not `H`) on the recurrence vector | Levitt-Torrent `abinit.tex:678-680` | Mode A is *prima facie* algorithmically wrong under exact S⁻¹ |
| ALG-2 | `λ_+`, `λ_−`, `λ_T` are bounds on **S⁻¹·H** spectrum | Levitt-Torrent `abinit.tex:670-675`; Das `main.tex:591` | Lanczos must apply S⁻¹·H (already does, `chebyshev.rs:474`); recurrence must apply S⁻¹·H to be consistent |
| ALG-3 | R-ChFSI Step 4 reconstruction: `X = D⁻¹·R_Y + X·Λ_Y` | Das `main.tex:607` | Mode C requirement |
| ALG-4 | When D⁻¹ = B⁻¹ exact, R-ChFSI ≡ standard ChFSI | Das `main.tex:612` | At ζ = 3.8e-15, Mode C is algebraically standard ChFSI on `S⁻¹·H` |

ALG-1..ALG-4 are admissible as criteria because they are stated in
peer-reviewed/preprint sources, independent of our pipeline. They argue for
Mode B or C *a priori* but cannot fire the discriminator alone — that is
SC-4-tight's job.

---

## Why these and not others

Excluded from criteria:
- **Iter-1 band-1 = −1.06 Ha (single number)** — too narrow; one band could match
  by coincidence under any mode. Promoted to per-band over 10 bands (SC-4-tight).
- **Total energy match to −24,110.96665 eV** — only achievable after full SCF
  convergence (mixing, occupancies, multi-iter). Out of scope per
  `phase-global-woodbury/PHASE_PLAN.md:208-210`.
- **R-ChFSI norm ratio ≤ some bound** — derived from our recurrence, not from
  CASTEP. Could constrain numerically but does not anchor against ground truth.

These exclusions follow the upstream-audit rule: criteria must come from a
source that is independent of our (potentially buggy) pipeline. Our log lines
are observations, not anchors.
