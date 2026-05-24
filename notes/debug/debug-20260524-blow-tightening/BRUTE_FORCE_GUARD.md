# Brute-Force Guard Triggered — Sweep Harness Bug Discovered

**Date:** 2026-05-24
**Trigger:** First sweep of `b_low_sweep_subspace_projector` produced flat ratios (0.8926 across all candidates from -0.2435 Ha to +5.0 Ha b_low_override). The skill's brute-force guard requires reporting to the user.

## What was supposed to vary

The sweep injects `eig[last]` overrides via `state.set_eigenvalues(eigs)` and expects the iter-2+ branch at `chebyshev.rs:1411` (`b_low = eig[eig.len() - 1]`) to consume the override.

## What actually varied

Per the captured Lanczos@call diagnostic on every iteration:

```
[Lanczos@call] b_low=0.0894  source=max_veff
```

→ The sweep candidate was IGNORED. Every run took the `_ => max_veff.max(0.0)` branch at chebyshev.rs:1417, regardless of what was set on the state object.

## Root cause

`src/scf.rs:577` and `src/scf.rs:716`:

```rust
// Always pass eigenvalues=None. Das et al. (2025) main.tex:612 proves
// that when ζ = ‖D⁻¹ − B⁻¹‖ = 0 (exact S⁻¹), R-ChFSI ≡ standard ChFSI
// algebraically. After §10's Global Woodbury fix, ζ = 3.8e-15 (machine
// epsilon). Per-band eigenvalue machinery is provably redundant and
// introduces numerical weak points from stale eigenvalue labels when
// V_eff drifts between SCF iterations.
let eig: Option<&[f64]> = None;
```

The production SCF path **deliberately** passes `eig = None` to `chebyshev_filter`. The `state.set_eigenvalues(...)` test API writes to `self.eigenvalues` but the value is never threaded through to the filter. The "iter-2+ path" referenced in proposal §5 and the parent debug session is **dead code in the production SCF**.

## Reframe of the problem

The proposal §5 hypothesis was: "tighten `b_low` at iter-2+ where `b_low = eig[last]` puts ~120 untracked bands inside the damped window".

But the actual production b_low formula at every iteration is:

```rust
// chebyshev.rs:1417 (iter-1 OR iter-2+)
(max_veff.max(0.0), "max_veff")
```

— always taking the "no eigenvalues available" branch because `eig` is None at every call.

So the leverage points are:
- **NOT** `eig[last]` (dead code).
- The **constant** `max_veff.max(0.0) ≈ 0.089 Ha` for Cu111+CO.

This means tightening `b_low` is a single-formula change: replace `max_veff` with something tighter. Candidates:
1. **Fermi-anchored**: `ε_F + N·smearing` where `ε_F = -0.122443 Ha`, `smearing = 3.6749e-3 Ha`.
   - `ε_F + 3w ≈ -0.111 Ha`
   - `ε_F + 10w ≈ -0.086 Ha` (~0.17 Ha tighter than max_veff)
   - `ε_F + 30w ≈ -0.012 Ha` (~0.10 Ha tighter)
2. **Lanczos-derived**: `ritz_max * α` (α < 1, mirror of b_up's `*1.1`).
   - The Lanczos@call diagnostic shows `ritz_max = 15.09 Ha` — too loose unless α is very small.
   - Would need a separate Lanczos run targeted at the *tracked-subspace* bands, not the full spectrum.
3. **Constant offset from ε_F or μ**: simpler, but less physically motivated.

Of these, (1) is most physically motivated. The Fermi level + N·smearing is the textbook Fermi window cutoff for fractional occupation: above this, occupation is < 10^-N, so damping these states does no physical harm.

## Hypotheses to test next

| Hypothesis | Test | Expected lift in Cu-3d ratio |
|------------|------|-----------------------------|
| H1: Fermi-anchored b_low (ε_F + 10w ≈ -0.086 Ha) reduces filter pollution | Modify chebyshev.rs:1417 directly, re-run subspace_projector | ratio 0.893 → 0.93+ if hypothesis is correct |
| H2: Tighter still (ε_F + 3w ≈ -0.111 Ha) lifts further | Same, with tighter constant | ratio 0.95+ |
| H3: Even tighter (ε_F + 0.5w ≈ -0.121 Ha) is too tight (admits occupied bands into damp region) | Same; expect REGRESSION (ratio drops) | ratio 0.85 or worse |

The empirical lever is: lower (more negative) b_low values widen the *amplification* region (everything below b_low is amplified). At ε_F itself, occupied bands (< ε_F by ~smearing) sit *above* b_low → damped → cascade. So the "right" b_low should sit ABOVE ε_F by enough margin to keep occupied bands in the amplification region.

The threshold to test is therefore: how much margin above ε_F is needed?

## Decision: pivot the sweep harness

Rather than sweeping via the (broken) `set_eigenvalues` knob, sweep by directly modifying chebyshev.rs:1417 → recompile → re-run the projector. This is heavier (full rebuild per candidate, ~20s × 4 candidates) but accurate.

Alternative: refactor chebyshev.rs to accept a `b_low_source` enum parameter, threaded through `chebyshev_filter`. This would expose b_low as a first-class A/B knob without per-candidate rebuilds. Heavier refactor; defer unless the sweep needs more than 4 iterations.

## Status of prior-note classifications

The following claims now need RECLASSIFICATION:

| Claim | Old class | New class | Reason |
|-------|-----------|-----------|--------|
| "Iter-2+ `b_low = eig[eig.len() - 1]`" (INVESTIGATION.md row 3) | EXTERNAL | **EXTERNAL but DEAD CODE** | The match arm exists (chebyshev.rs:1411) but `eigenvalues = None` at every production call (scf.rs:577, 716) → branch never taken |
| "Iter-3 b_low ≈ 1.95 Ha would cause cascade" (INVESTIGATION.md row 4) | HYPOTHESIZED | **REFUTED** | Iter-3 b_low = 0.0894 Ha = max_veff = same as iter-1, regardless of eig[last] |
| "n_bands buffer (160 tracked, ~93 occupied) — bands 94-160 sit below b_low = eig[last]" (DIVERGENCE_SURFACE.md B1) | DERIVED + HYPOTHESIZED | **REFUTED** | Production b_low = 0.0894 < ε_F + 30w ≈ -0.012, so all 160 bands sit BELOW b_low and ARE all amplified. The "out-of-window pollution" is real but its mechanism is different: bands 41-160 are amplified BY THE SAME FACTOR as bands 0-40 → no discrimination |

## Recommendation to user

**Pivot Step 7 from the sweep harness to direct constant edits.** Test 3 candidates in chebyshev.rs:1417:

1. **C1**: `ε_F + 10·smearing` (-0.086 Ha) — a touch tighter than max_veff
2. **C2**: `ε_F + 30·smearing` (-0.012 Ha) — much tighter, but still above ε_F
3. **C3**: `ε_F + 100·smearing` (+0.245 Ha) — looser than max_veff (sanity baseline)

Run `subspace_projector_iter1_vs_castep` between each. Each cycle: ~20s rebuild + 6 min run = ~7 min × 3 = 21 min total.

If C1 or C2 lifts the Cu-3d ratio from 0.893 toward 0.94+, write the tight test (Step 7.2) and commit. If neither lifts substantively, this is a different bug entirely (filter polynomial degree? spectral conditioning?) — escalate.

Threading `μ` and `smearing_width` to `chebyshev_filter`: these are already on `self.smearing` (visible at scf.rs:879). Plumbing is ~5 lines per call site.

## What this proves about the proposal

Proposal §14 §1.4 attributes "~6–11% Chebyshev filter pollution from outside-window high-G components" to a separate mechanism addressable by tightening `b_low`. This is **partially correct in motivation but wrong in mechanism** — the outside-window leakage is real (visible in subspace_projector ratios), but it's not because b_low cuts at `eig[last]`. It's because **b_low cuts at +0.089 Ha = above all 160 tracked bands**, so the filter amplifies *every* band uniformly without discrimination. Inside-window bands (occupied + small buffer) and outside-window bands (41-160) get the same amplification weight.

The fix is to bring b_low ABOVE the tracked-band ceiling but BELOW the untracked region — i.e., into the gap above ε_F + N·smearing. This is exactly the "Fermi window" cutoff convention.
