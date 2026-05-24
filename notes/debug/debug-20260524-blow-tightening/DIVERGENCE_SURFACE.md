# Divergence Surface — b_low Tightening (Proposal §5)

**Date:** 2026-05-24
**Approach:** breadth-first enumeration of every plausible disagreement between our Chebyshev filter window choice and a CASTEP-equivalent choice; per item, either rule out (citing an EXTERNAL anchor) or list for Step 7 testing.

## Surface enumeration

### A. b_low at iter-1 (no prior eigenvalues)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| A1 | `b_low = max_veff.max(0.0) + 2.0` (the proposal §5 text claim) | **RULED OUT** | EXTERNAL: `chebyshev.rs:1417` reads `max_veff.max(0.0)` (no +2.0) on the Lanczos path; `compute_spectral_bounds:379` still has +2.0 but that branch is dead-code when ndeg>0 (the production path) per `chebyshev.rs:1356-1441`. Proposal text is stale; reclassify in RESOLUTION.md. |
| A2 | `b_low = max_veff` is still too loose (Fermi-level-aware would be tighter) | **TO BE TESTED** | The Fermi level ε_F = −0.122443 Ha sits **below** max_veff ≈ 0.089 Ha by 0.21 Ha. If the relevant tracked-band ceiling is ε_F + N·smearing, max_veff overshoots by ~0.2 Ha and admits bands from positions [N_occ+ε, N_occ+ε+0.2 Ha] into the amplification region. |
| A3 | `b_low` should be Lanczos-derived `ritz_max * 0.8` (mirror of the upper bound) | **TO BE TESTED** | The Lanczos `ritz_max` is the largest eigenvalue of the Lanczos T_k matrix on the wavefunction subspace. For 6-step Lanczos starting from band 0, `ritz_max` approximates the largest tracked-band eigenvalue more directly than `max_veff` (which is just the maximum of V_eff in real space). |

### B. b_low at iter-2+ (eigenvalues available)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| B1 | `b_low = eig[eig.len() - 1]` puts ~120 untracked-physical bands inside the damped window | **PRIMARY CANDIDATE — TO BE TESTED** | For n_bands = 160 with ~93 occupied bands, `eig[159]` is the 160th-band eigenvalue, which can sit far above ε_F (e.g., observed 0.131 Ha at iter-1 clean baseline, 1.95 Ha at iter-2 after pollution). Bands 41-159 (positions outside the 40-band tracking window) end up *below* `b_low` → amplified by the filter → reseed the subspace each iteration. |
| B2 | `b_low = eig[N_track - 1]` (use eigenvalue at the tracking-window boundary) | **TO BE TESTED** | Replaces `eig[159]` with `eig[39]` or similar; this puts only the tracked-and-physical bands below `b_low`. The trade-off: if the tracking window is set too narrow, occupied + small-buffer bands at the top of the window get damped. |
| B3 | `b_low = ε_F + 3·smearing_width` (Fermi-anchored, μ-aware) | **TO BE TESTED** | The Fermi level + 3 smearing widths is the textbook "Fermi window" cutoff for fractional occupation. Above this, occupation is < 10^-4; bands above can be damped without loss of physical fidelity. |
| B4 | `b_low = 0.5 × (eig[N_occ] + eig[N_occ+1])` (gap-aware midpoint) | **TO BE TESTED** | Sits halfway between the highest occupied and lowest unoccupied eigenvalue. For an insulator this is the band-gap midpoint; for a metal at finite temperature it approximates ε_F. |
| B5 | The "+2.0" stale clamp in `compute_spectral_bounds:379` could affect ndeg=0 callers | **PARTIAL — TO BE TESTED** | The ndeg=0 path (`FilterMode::SinvHKeepHEig` with ndeg=0) is used by Variant A/B baselines and `pollution_bisect_blow_vs_per_band_shifts`. If those tests rely on `+2.0` semantics, removing it could regress them. Verify by reading `pollution_bisect` and `Variant A/B` outputs after a candidate-fix dry-run. |

### C. Lanczos b_up (the upper end of the filter window)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| C1 | `b_up = ritz_max * 1.1` cap by Gershgorin | **RULED OUT** by EXTERNAL anchor | Step 5 Variant A test shows filter denoises 460× when polluter is at high energy — meaning bands far above b_low are properly damped. The upper bound is doing its job; no symptom traced here. |
| C2 | Lanczos starting vector = band 0 (well-converged) → overshoots | **RULED OUT** | The `scaled.min(gershgorin_b_up)` cap at L1377 catches Lanczos overshoot. EXTERNAL: code-read. |

### D. R-ChFSI spectral-shift parameters (σ, c, e, γ)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| D1 | `lambda_min = ritz_min * 0.8` for σ-shift | **RULED OUT** | `subspace_projector` ratio is insensitive to `lambda_min` (the R-ChFSI shift affects per-band Λ_X update, not the filter window). EXTERNAL: post-mortem narrowed the symptom to filter window only. |
| D2 | `center = (b_up + b_low) / 2.0` formula | **RULED OUT** | Standard affine map for [b_low, b_up] → [−1, +1]. Math-verified. |

### E. Diagnostic itself (subject of Step 5 verification)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| E1 | `apply_s_for_test` produces correct S·ψ | **TO BE VERIFIED** by Step 5 Path-B independent CPU computation | EXTERNAL anchor: ψ_castep is S-orthonormal → `⟨ψ_a | S | ψ_a⟩ ≈ 1.0`. If Path A and Path B disagree on this, the diagnostic is bugged. |
| E2 | Block-sum boundary `block_sum(1, 14)` matches Cu-3d physical extent | **EXTERNAL** | tests/ca_scf_convergence.rs:3533 + proposal §1.3 — bands 1..14 are the 13 Cu-3d states. Code-evident. |

## Items to test in Step 7

The primary candidates to sweep in Step 7.1 are **B1 vs B2 vs B3 vs B4** (the iter-2+ `b_low` source). Items A2 and A3 are secondary because the iter-1 path is also called once per SCF run; their effect on iter-1 sanity gate is interleaved with the iter-2+ effect on `eig[last]`.

The sweep harness `pollution_bisect_blow_vs_per_band_shifts` already supports arbitrary `eig[last]` override, which is the cleanest way to test B-family items without changing production code.

## Items ruled out by anchor

A1, C1, C2, D1, D2, E2 — see EXTERNAL evidence per row.

## Items pending diagnostic verification (Step 5)

E1 — to be verified before any block-sum number from the diagnostic is trusted.
