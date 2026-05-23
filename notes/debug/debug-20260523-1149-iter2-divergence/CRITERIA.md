# Anchor Criteria — iter2-divergence

**Slug:** `debug-20260523-1149-iter2-divergence`
**Date:** 2026-05-23

Each criterion is EXTERNAL (fixture, spec, or independent reference) or
corroborated by an existing tight test gate. DERIVED claims are excluded.
Each anchor has a stable ID for traceable citation in Step 7 tests.

## Layer A — End-to-end SCF criteria

The SCF must reach a state consistent with CASTEP's converged answer for
Cu111+CO. Sources are CASTEP fixtures at
`/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/` parsed by
`tests/fixtures/cu111_co.rs`.

| ID | Assertion | Source |
|----|-----------|--------|
| **A-BAND0** | Iter-2 band-0 within 0.05 Ha of −1.0550 Ha | `.bands` fixture, band index 0 |
| **A-BAND9** | Iter-2 band-9 within 0.05 Ha of −0.4681 Ha | `.bands` fixture, band index 9 |
| **A-BANDS-10** | Iter-2 lowest 10 eigenvalues within 0.05 Ha of `.bands` band 0–9 | `.bands` fixture |
| **A-LAST** | Iter-2 last (band-159) within 0.05 Ha of iter-1 last (which already passes the band-159 ≈ 0.115 Ha gate) | `.bands` fixture |
| **A-VEFF-RNG** | Iter-2 V_eff range within 1.0 Ha of iter-1 (≈ 8.69 Ha) | `.pot_fmt` fixture, parsed via VEffBuilder |
| **A-RHO-SOFT** | Iter-2 smooth ρ integral = 68.44 e⁻ within ±1% | CASTEP F8 instrumentation, §8 RESOLUTION |
| **A-RHO-AUG** | Iter-2 aug ρ integral = 117.56 e⁻ within ±1% | CASTEP F8 instrumentation, §8 RESOLUTION |
| **A-RHO-TOT** | Iter-2 total ρ = 186 e⁻ | `.castep_bin` total electrons |
| **A-DISCRIM** | Iter-2 last band < 0.5 Ha (current 1.95 Ha → 4× margin) | Discriminator chosen per skill rule: `wrong_value / target_value ≥ 2×` |

## Layer B — Filter-as-transform criteria

The R-ChFSI body in `chebyshev.rs:1273-1697` must reproduce the analytic
Chebyshev polynomial response on a controlled isolated input. This is the
algorithm-validation test the user requested at Step 6. The test is
constructed so that it has no SCF, no USPP, no V_NL, no S-augmentation —
only the filter recurrence acting on a known H.

Source: Das et al. (2025) "Residual-based Chebyshev filtered subspace
iteration" Algorithm 3, `reference_paper/2025-rchfsi-inexact-mv-paper.tar.gz`,
specifically `main.tex:586-610`.

| ID | Assertion | Source |
|----|-----------|--------|
| **B-T8-DIAG** | With `H = diag([λ_1..λ_n])` spanning [-1.5, 20.0] Ha, `S = I`, no V_NL: max relative error between filter output Rayleigh quotient and analytic `T_8((λ−c)/e)` amplification < 1% per band | Das main.tex:586-610 |
| **B-T8-SWEEP** | B-T8-DIAG holds across pollution sweep `α ∈ {0.01, 0.1, 0.25, 0.5}` of the highest eigenvector mixed into the lowest band | Iter-1 starts from near-zero pollution (CASTEP ψ); iter-2 has unknown larger pollution |
| **B-NDEG-SWEEP** | B-T8-DIAG holds for ndeg ∈ {2, 4, 8, 16, 32} | Polynomial-degree sensitivity check |

## Layer C — H/S spectrum bridging criteria

Only relevant if Layers A and B both have unresolved failures. Verifies
whether `b_low = eig[last]` (a generalized eigenvalue from
`H_sub·X = λ·S_sub·X`) is the right spectral cutoff for a filter that
operates on either bare-H or S⁻¹·H. The Chebyshev polynomial maps
`[b_low, b_up]` onto `[-1, 1]` for *some* operator; the question is whether
that operator's eigenvalues align with the `eig[last]` framing.

| ID | Assertion | Source |
|----|-----------|--------|
| **C-SPECTRUM** | On iter-1 RR output's 160-band subspace: \|λ_max(H projected) − λ_max(S⁻¹·H projected)\| < 5% × (b_up − b_low) | Generalized vs operator eigenvalue identity |

## Discriminator value selection rationale

Per `odd-pattern.md` Discriminator Value Selection: thresholds must place
correct and incorrect implementations ≥ 2× apart. Boundary thresholds are
brittle.

- **A-DISCRIM** (last band < 0.5 Ha): correct ≈ 0.13 Ha, current wrong =
  1.95 Ha. Threshold 0.5 Ha sits in the middle. Discriminator ratio:
  current 1.95 / threshold 0.5 = 3.9× over; correct 0.13 / threshold 0.5 =
  3.85× under. Both directions ≥ 2×. Acceptable.
- **A-VEFF-RNG** (Δ < 1.0 Ha): correct = 8.69 Ha (iter-1 baseline), current
  iter-2 = 20.26 Ha → Δ = 11.57 Ha. Threshold 1.0 Ha gives ratio 11.6×
  over. Strong discriminator.
- **B-T8-DIAG** (max rel error < 1%): if the filter body is correct, error
  should be at machine precision (~1e-10 to 1e-6 typical for double-
  precision T_8 evaluations). Threshold 1% gives ratio ≥ 10,000×.

## Falsifiability checks

Each Layer A criterion must be capable of failing on the current state:

| ID | Current state | Threshold | Will fail? |
|----|--------------|-----------|------------|
| A-BAND0 | iter-2 band-0 = −0.864 Ha | within 0.05 Ha of −1.055 Ha | YES (|Δ| = 0.191 Ha > 0.05) |
| A-LAST | iter-2 last = 1.95 Ha | within 0.05 Ha of iter-1's 0.13 Ha | YES (|Δ| = 1.82 Ha > 0.05) |
| A-VEFF-RNG | iter-2 V_eff range = 20.26 Ha | within 1.0 Ha of 8.69 Ha | YES (|Δ| = 11.57 Ha > 1.0) |
| A-RHO-SOFT | iter-2 smooth ≈ 181 e⁻ | 68.44 e⁻ ± 1% | YES |
| A-RHO-AUG | iter-2 aug ≈ 5 e⁻ | 117.56 e⁻ ± 1% | YES |
| A-DISCRIM | iter-2 last = 1.95 Ha | < 0.5 Ha | YES |

All Layer A discriminators currently fail on the broken state. Falsifiable. Anchored.
