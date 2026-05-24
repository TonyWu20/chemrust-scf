# Prior-Note Classification — b_low Tightening (Proposal §5)

**Date:** 2026-05-24
**Slug:** `debug-20260524-blow-tightening`
**Symptom (user-provided):** "`b_low` tightening (proposal §5) to reduce filter pollution"
**Origin trigger:** `notes/debug/debug-20260524-postrr-cascade-amplification/RESOLUTION.md` §"Recommendation for next session" item 1.

## Scope

The PostRr Procrustes pin attempt is concluded as a partial-success/falsified.
The post-mortem hands off three priorities; this session executes priority 1
(tighten `b_low` to reduce out-of-window Chebyshev filter pollution). The
~3% iter-1 ceiling on per-band S-overlap (PostRr empirical floor 0.945,
ceiling ~0.977) is the binding constraint that this fix targets.

## Classification table

Every numeric claim referenced in prior notes about `b_low` or filter
pollution is classified per `odd-pattern.md` (EXTERNAL / DERIVED /
HYPOTHESIZED). Only EXTERNAL claims become success criteria in CRITERIA.md.

| # | Claim | Source | Class | Reason |
|---|-------|--------|-------|--------|
| 1 | Current `b_low = max_veff + 2.0` at iter-1 | proposal §14 line 267; postrr-cascade-amplification:64 | **DERIVED (stale)** | Code at `chebyshev.rs:1417` reads `max_veff.max(0.0)` (no `+2.0`); the 2026-05-23 `iter1-filter-operator-mismatch` resolution already replaced it. Proposal text was not updated. |
| 2 | Iter-1 `max_veff ≈ 0.089 Ha` for Cu111+CO | `chebyshev.rs:1401` inline comment; debug-20260523-0916/RESOLUTION.md | **EXTERNAL** (code-evident + corroborated) | Direct read of the production-comment that drove the prior fix; cited as the operating value in the upstream resolution. |
| 3 | `b_low = eig[eig.len()-1]` at iter-2+ | `chebyshev.rs:1411` | **EXTERNAL** | Direct code read. |
| 4 | Iter-3 `b_low` ≈ 1.95 Ha would cause cascade | debug-20260523-1149-iter2-divergence/INVESTIGATION.md:43 | **HYPOTHESIZED** | The mechanism is mechanically obvious from the formula but the cascade as a consequence is unverified post-fix. |
| 5 | Subspace projector Cu-3d 1..14 = 11.6041 (ratio 0.893) | proposal §1.3 table; tests/ca_scf_convergence.rs:3475 emits per run | **EXTERNAL** | Empirical readout from a passing diagnostic (subject to Step 5 self-test). |
| 6 | ~6–11% span pollution from outside the 40-band window | proposal §1.4 | **EXTERNAL** | Decomposition of (5): block_sum_0_to_40 = 37.5412 vs target 40 → 6.1% beyond-40 pollution. |
| 7 | "Filter pollution limits iter-1 to ~0.97 floor regardless of pin" | postrr-cascade-amplification:111 | **EXTERNAL** | Falsification result from PostRr empirical run; the floor IS the residual filter pollution. |
| 8 | iter-3 band-0 = −11.94 Ha under PinMode::Off | postrr-cascade-amplification:75; `cascade_iter3_diagnostic_tight` | **EXTERNAL** | Direct test output, recent. |
| 9 | iter-3 band-0 = −14.91 Ha under PostRr | postrr-cascade-amplification:76 | **EXTERNAL** | Direct test output, recent. |
| 10 | A1: CASTEP band-0 = −1.05502287 Ha | `Cu111_CO.bands:12` | **EXTERNAL** | Fixture file. |
| 11 | A2: ε_F = −0.122443 Ha | `Cu111_CO.bands:5` (header) | **EXTERNAL** | Fixture file. |
| 12 | A3: smearing width = 0.1 eV ≈ 3.6749 mHa | `Cu111_CO.param:35` | **EXTERNAL** | Fixture file. |
| 13 | A5: CASTEP ψ S-orthonormal | `Cu111_CO.check`; user memory `castep_check_continuation_convention` | **EXTERNAL** | Verified empirically by the iter-1 sanity gate at PostRr eps_degen=0.05 (floor 0.945 — implies CASTEP ψ is non-degenerate enough for block alignment). |
| 14 | Variant A/B pollution probes show filter denoises 460× / 33× at iter-1 path | debug-20260523-1149-iter2-divergence/STEP_7_1_RESULT.md | **EXTERNAL** | Direct test result; rules out filter polynomial math as the bug. |
| 15 | "n_bands buffer (160 tracked, ~93 occupied)" — bands 94-160 sit below current `b_low = eig[last]` | debug-20260523-1149-iter2-divergence/DIVERGENCE_SURFACE.md:38 | **DERIVED + HYPOTHESIZED** | DERIVED for "160 tracked" (code-evident); HYPOTHESIZED for the consequence "filter preferentially amplifies a band-1xx eigencomponent over band-93". |
| 16 | The +2.0 in `compute_spectral_bounds:379` is "dead code" under ndeg>0 paths | this session's read of `chebyshev.rs:1356-1417` | **EXTERNAL** | Direct code-read: `compute_spectral_bounds` provides initial `bounds`, then ndeg>0 overrides via L1435-1441 inside `chebyshev_filter`. Only ndeg=0 callers (e.g., FilterMode::SinvHKeepHEig with ndeg=0) hit the +2.0 path. |

## Memory citations (auto-classify as DERIVED unless triple-anchored)

| Memory entry | Citation | Verification granularity | Scope | Class |
|--------------|---------|--------------------------|-------|-------|
| `relative_target_procrustes_feedback.md` | (no file:line) | Per-iteration cascade test result | Cu111_CO fixture, single SCF run | **EXTERNAL** by exception — corroborated independently by the analyst plan's section 2a. Triple-anchored. |
| `castep_check_continuation_convention.md` | (no file:line) | Pseudopotential / Woodbury identity | All CASTEP `.check` files | **EXTERNAL** by exception — same convention has been tested at machine precision in `test_2_s_sub` (proposal §1.2 row 1). |
| `dont_revert_empirically_correct_fix_on_regression.md` | (no file:line) | General methodology | — | **EXTERNAL** as guidance (not as a numeric claim). |
| `range_only_acceptance_misses_pointwise.md` | (no file:line) | General methodology | — | **EXTERNAL** as guidance. |
| `rayon_fine_grid_memory_overhead.md` | (no file:line) | General methodology | — | **EXTERNAL** as guidance — not load-bearing for this session. |
| `castep_gaussian_smearing.md` | (no file:line) | Smearing function | All CASTEP runs | **EXTERNAL** by exception — needed if Step 7.1 sweep uses ε_F-anchored candidates. |

## Reclassifications applied in CRITERIA.md and DIVERGENCE_SURFACE.md

- Claim #1 ("current `b_low = max_veff + 2.0`") → **moved to RULED OUT in divergence surface** rather than treated as an open lever. The actual current value at iter-1 is `max_veff` (no `+2.0`); the proposal text is stale and will be corrected in this session's RESOLUTION.md.
- Claim #4 ("Iter-3 b_low ≈ 1.95 Ha would cause cascade") → **HYPOTHESIZED**; not used as a success criterion. The cascade gate is advisory (per analyst plan's V_eff drift finding).
- Claim #15 (bands 94-160 leaking) → **DERIVED for the count, HYPOTHESIZED for the consequence**. The consequence is the working hypothesis Step 7.1 will test by sweeping `eig[last]`.

## Output

Only EXTERNAL claims (1*, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16) become success-criterion inputs in CRITERIA.md. Claim 1 enters as a CORRECTED-STALE entry (the actual current value).
