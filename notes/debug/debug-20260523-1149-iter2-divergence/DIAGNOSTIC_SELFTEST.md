# Diagnostic Self-Test — iter2-divergence

**Slug:** `debug-20260523-1149-iter2-divergence`
**Date:** 2026-05-23

Per `odd-pattern.md` Diagnostic Self-Verification: every diagnostic in the
tree is verified against an independent computation path before its output
is read as evidence in any later step. The skill's standard is ≥10 sample
points compared through two structurally independent code paths.

## Diagnostics in tree (enumerated)

Located via `rg -n "eprintln!" src/eigensolver/chebyshev.rs src/density.rs src/scf.rs`:

| Tag | Location | Quantity | Independent path |
|-----|----------|----------|------------------|
| `[V_eff]` `min max range` | `scf.rs:486-492` | V_eff aggregate stats | ndarray fold `(min, max)` vs cuBLAS `IZAMAX`/`amax` on the GPU buffer pre-downsample |
| `[Density] rho_sum rho_min rho_max` | `scf.rs:486-492` | density aggregate | Independent CPU loop summing the same `Array3<f64>`; sample 10 grid points at known offsets |
| `[NewDensity] rho_sum aug_sum total_e chem_pot` | `scf.rs:660-755` (compute_density_from_wavefunctions output) | iter-output density | CPU back-substitution: load the GPU buffer to host, sum directly; verify within 1e-12 of the diagnostic-printed value |
| `[RR] eigenvalues first last count` | `scf.rs:555-558` | RR ZHEGVD output | CPU back-substitution: compute `‖H_sub · X − S_sub · X · Λ‖` for first 5 bands |
| `[psi] \|c\|_min \|c\|_max` | `scf.rs:486-492` | ψ amplitude bounds | ndarray fold vs ZNRM2 on GPU buffer |
| `[Chebyshev] b_up b_low center half_width lambda_min` | `chebyshev.rs:1446-1448` | spectral bounds | Compare against Lanczos T_k diagonalization done in `lanczos_upper_bound` body |
| `[Lanczos@call] b_up_lanczos ritz_min ritz_max gershgorin scaled b_up capped_by_gershgorin` | `chebyshev.rs:1381-1384` | Lanczos output | Run Lanczos on the Layer B fixture H (where analytic eigenvalues are known); compare ritz_min/ritz_max to the synthetic H's eigenvalues |
| `[Lanczos@call] b_low source` | `chebyshev.rs:1421` | b_low selection | Source-check: `eig[last]` should be the same value as `[RR] eigenvalues: last` from the previous iteration |
| `[Lanczos@call] guard pass` | `chebyshev.rs:1426-1431` | Lanczos guard | Boolean composition check |
| `[R-ChFSI] k norm_prev norm_curr ratio` | `chebyshev.rs:1677` | Per-step Chebyshev norm | Direct cuBLAS `ZNRM2` on buf_ry and buf_rx outside the diagnostic context; compare values |
| `[ConstructD]` per-ion D_screened amax (likely in `vnl_data.rs`) | TBD | D-screening output per ion | CPU `entry.d_matrix.iter().map(\|c\| c.norm()).fold(0., f64::max)` on the host copy of d_matrix |

## Per-band per-step gap (the critical one)

**All current diagnostics emit summary statistics only.** The §11 symptom
table contains aggregates: V_eff range, ρ_sum, last band, D_screened amax.
For observing the polynomial response per-band per-Chebyshev-step (which is
exactly what Layer B will require to validate the recurrence body), we need
a per-step Rayleigh-quotient probe.

**Action:** add `eprintln!` at `chebyshev.rs:1674` vicinity (inside the
recurrence loop, after buffer rotation) that emits per-band
`⟨ψ_b, H ψ_b⟩` per step. This is required *before* running Layer B
because without it, B-T8-DIAG will only see the final filter output, not
the per-step polynomial response. With the per-step dump, we can:
- Verify each step's amplification matches the analytic
  `T_k((λ−c)/e)` polynomial value.
- Localize *which* Chebyshev step (k=2..8) introduces divergence.
- Distinguish "the recurrence body is wrong" (Step 8 narrows to formula
  bug) from "the recurrence body is right but inputs are wrong" (Step 8
  narrows to upstream).

## Verification protocol per diagnostic

For each diagnostic above, before any production output is admitted as
evidence in Steps 7+:

1. Construct or identify ≥10 sample points where independent computation
   is feasible (small synthetic input, or known fixture point).
2. Compute the same quantity through the two paths.
3. Assert agreement to 1e-12 (machine precision for double-precision
   arithmetic on reasonably-conditioned operations).
4. If any sample point disagrees, the diagnostic itself is the bug. Fix
   the diagnostic *first*; do not interpret its production output.

## Sanity check against physical intuition

Per `odd-pattern.md` Suspect the Diagnostic First:

| Observation in §11 | Likely diagnostic defect? |
|--------------------|---------------------------|
| iter-2 last band = 1.95 Ha, filter window upper = 20.8 Ha | Unlikely — 1.95 Ha is well below 20.8 Ha; the surprise is that 1.95 > b_low = 0.13, but RR output isn't subject to the filter window |
| iter-2 V_eff range = 20.26 Ha (vs iter-1's 8.69 Ha) | Plausibly real (ρ_aug crashed → V_H/V_xc smoothing reduced → bare V_loc wells exposed). Cross-check: iter-2 V_eff range matches the §11 reported value 20.26 Ha exactly — if our diagnostic is reproducing prior DERIVED output, it's the diagnostic that's stuck. Mitigation: Step 7 re-runs with `feature = "scf_diag"` and verifies the diagnostic fires per-iteration, not just once |
| iter-2 D_screened amax 31.4 Ha for Cu vs iter-1 5.8 Ha | Order-of-magnitude jump but stays below iter-3's 315 Ha; consistent with the cascade pattern. Real. Self-test by CPU recomputation on the host d_matrix copy |
| iter-2 smooth ρ = 181 e⁻ vs iter-1 68 e⁻ (with same total ρ ≈ 186) | Suspicious: the inversion of soft/aug ratio (iter-1 was 68/118 ≈ 0.58 ratio, iter-2 is 181/5 ≈ 36×). This needs the most careful diagnostic verification before being treated as fact. If our `[NewDensity]` diagnostic counts soft and aug separately, the field-by-field sum of "smooth + aug" must equal the field "total" — verify with a direct addition probe |

## Plan-mode acknowledgment

The plan's Step 7.0 (ndeg=0 baseline) does not require the per-band
per-step probe — it only requires `[RR] eigenvalues` (already present and
verifiable). So Step 7.0 can run *before* the probe is added.

The plan's Step 7.1 (Layer B) does require the per-band per-step probe to
make `T_8 vs Rayleigh-quotient` measurable. So the probe must be added
between Step 7.0 and Step 7.1.

This ordering is preserved by the Step 7 sub-step numbering already in the
plan file.
