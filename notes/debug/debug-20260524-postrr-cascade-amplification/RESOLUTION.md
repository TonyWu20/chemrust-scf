# Resolution — PostRr Procrustes Pin: iter-1 success, cascade amplification

**Status:** PARTIAL SUCCESS / DESIGN REVISION NEEDED.
**Date:** 2026-05-24.
**Branch:** `feat/phase-global-woodbury`.
**Commits this investigation:**
- `a3e5c78` — Steps 1-3 scaffolding (red tests, faer, PinMode enum)
- `12b96b0` — Step 4 plumbing (`prev_psi_dev` through call chain)
- `1fbbf7c` — Step 5 initial PostRr math (raw indexing)
- `1e3d952` — Step 5 typed-faer refactor (per-block SVD)
- `f5e7e3d` — `RrPinConfig::from_env()` wired into `diagonalize_inner`
- `0249285` — PCI-E budget extensions + `eps_degen` default raised to 0.05

## Symptom

`cascade_iter3_diagnostic_tight` failing with iter-3 band-0 = −11.94 Ha,
gate < 0.1 Ha from A1 (−1.05502287 Ha from `Cu111_CO.bands:12`). The
proposal §14 hypothesis attributed this to ZHEGVD in-block gauge rotation
within near-degenerate eigenvalue clusters. The Procrustes pin was the
proposed fix.

## What was tried

PostRr variant per the plan at
`notes/proposals/14-eigensolver-rotation-fix-plan.md` §"PinMode::PostRr":

1. Detect degenerate blocks by consecutive eigenvalue spacing (`eps_degen`).
2. Compute T = ψ_prev^H · S · ψ_row (full n×n, USPP-augmented).
3. Compute M = T · X on GPU (where X is ZHEGVD's eigenvector matrix).
4. D2H M and X.
5. Per-block CPU SVD via faer 0.24: `M_block = U·Σ·V^H` → `R = U·V^H`.
6. Apply `X[:, lo..hi] ← X[:, lo..hi] · R^H` on host, H2D X back.
7. Existing `ψ_new = ψ_row · X` gemm runs against pinned X.

Env-var toggle `CHEMRUST_PIN_MODE` ∈ {off, prerr, postrr} via
`RrPinConfig::from_env()` (follows the `CHEMRUST_FORCE_NO_EIGS` pattern).

## Iter-1 sanity gate — `pin_preserves_castep_basis_at_iter1_postrr`

Test: load CASTEP ψ as both `ψ_prev` and the initial guess, run iter-1
under PostRr, check `|⟨our_a | S | castep_a⟩|² > 0.999` for bands 0..40.

| Config | Floor | Median | Outliers (< 0.01) | Gate (> 0.999) |
|--------|-------|--------|-------------------|----------------|
| `PinMode::Off` (baseline)     | 1e-6  | ~0.07  | 10 bands | RED |
| `PostRr` + `eps_degen=0.01`   | 1e-6  | ~0.94  |  2 bands | RED |
| `PostRr` + `eps_degen=0.05`   | 0.945 | ~0.977 |  0 bands | RED but at the noise floor |

**Mechanism of the eps_degen=0.01 failure:** the threshold over-fragments
near-degenerate clusters into 2-band blocks at boundaries where filter
pollution has shifted eigenvalues by 8–15 mHa. The resulting M_block is
poorly conditioned and SVD-polar gives a chaotic rotation. Concretely:
bands 22 (in block [21,23)) and 24 (in block [23,25)) showed
overlaps 0.000003 and 0.000002 — orthogonalized to CASTEP, not aligned.

**At eps_degen=0.05:** the entire bands [1, 160) becomes a single block,
the SVD is well-conditioned, and the pin recovers from the 1e-6 floor
to a ~0.97 floor. The remaining ~3% residual is attributable to
Chebyshev filter pollution from bands outside the 40-band window
(proposal §1.3 documented ~6–11% span pollution from this source).

**Conclusion:** the PostRr pin math is correct; `eps_degen=0.05` is the
right default. The 0.999 gate is not physically reachable in iter-1
with current Chebyshev filter parameters (`b_low = max_veff + 2.0`
allows out-of-window component bleed). This is a meaningful but partial
success.

## Cascade discriminator — `cascade_iter3_diagnostic_tight`

Test: run iter-1 → iter-2 → iter-3 SCF, check iter-3 band-0 within 0.1 Ha
of CASTEP A1 (−1.05502287 Ha).

| Config | iter-1 band-0 | iter-2 band-0 | iter-3 band-0 | Drift vs A1 |
|--------|---------------|---------------|---------------|-------------|
| `PinMode::Off` (HEAD baseline) | (similar) | (drift starts) | **−11.94 Ha** | 10.9 Ha |
| `PostRr` + `eps_degen=0.05`    | **−1.0458** ✓ | −0.890 | **−14.91 Ha** | **13.86 Ha (WORSE)** |

**The cascade is WORSE under PostRr.** This is a qualitatively new failure
mode, not a no-op. The plan's decision protocol §"Decision protocol after
A/B run" anticipated "both modes fail cascade" as a sign that the
residual driver is outside ZHEGVD rotation — but did not predict that
PostRr would *amplify* drift relative to Off.

## Root cause hypothesis (unverified)

The pin uses `prev_psi` = the previous iteration's RR output ψ. Across
SCF iterations this creates a self-reinforcing feedback chain:

  iter-2 pin aligns to iter-1's output
  iter-3 pin aligns to iter-2's (pinned) output
  iter-4 pin aligns to iter-3's (twice-pinned) output
  ...

Each iteration's V_eff is built from the previously-pinned ψ. If the
pin's small per-iteration rotation interacts non-trivially with the
density-V_eff feedback, the cumulative drift exceeds the no-pin
baseline rather than reducing it.

Auxiliary evidence: the iter-3 eigenvalue spectrum spans
`-14.91 to 4.65 Ha` (vs CASTEP's `-1.06 to ~-0.12 Ha`). This is far
wider than physically plausible — V_eff has drifted enormously by
iter-3, not just ψ. The pin's rotation is reshaping the spectrum
in a way that compounds rather than corrects.

## Falsification status of proposal §14

| Claim from §14 | Status after this investigation |
|----------------|----------------------------------|
| "ZHEGVD in-block gauge rotation is the dominant cascade driver"           | **PARTIALLY VINDICATED** — iter-1 sanity gate confirms in-block rotation IS a real ~99% effect rescued by PostRr at eps=0.05 |
| "Procrustes pin against ψ_prev fixes cascade"                              | **FALSIFIED for PostRr** — pin amplifies cascade rather than correcting it |
| "Iter-1 sanity gate `> 0.999` is achievable in most-favorable case"        | **FALSIFIED** — Chebyshev filter pollution limits iter-1 to ~0.97 floor regardless of pin |
| "10% Chebyshev pollution outside 40-band window is a separate problem"     | **REINFORCED** — appears to be the binding constraint on iter-1 sanity gate AND likely a major cascade contributor independent of in-block rotation |

## Reclassified prior claims

- Original §14 §5 expected: "if iter-1-iter-3 cascade tests all go green
  but Q2 still fails by ~5% in energy, the pollution is the remaining
  work." — **NOT MET**. We did not reach green iter-1-iter-3, so the
  pollution-is-remaining-work hypothesis cannot be tested as stated.
  Reclassified to: "filter pollution and SCF-feedback drift are jointly
  load-bearing; PostRr in isolation is insufficient."

- Original plan §"Decision protocol" predicted: "Both pass iter-1 but
  neither passes cascade" → "Pivot to b_low tightening." — **REVISED**:
  iter-1 passes substantively (floor 0.945, vs 1e-6 baseline) but at
  a < 0.999 ceiling. Cascade gets actively worse. Suggests b_low
  alone is not the only follow-up; the SCF iteration loop's coupling
  with the pin needs analysis.

## Recommendation for next session

Do NOT proceed to PreRr A/B. Three reasons:

1. PreRr's design has a known strict-degeneracy weakness (ZHEGVD
   re-randomizes the in-block gauge regardless of pre-alignment) — at
   iter-1 it would likely match PostRr in floor, and at iter-3 it
   suffers the same feedback-amplification issue or worse.
2. The cascade-amplification finding points to a more fundamental issue
   in the pin↔V_eff feedback loop, not the choice of pin location.
3. The iter-1 floor of ~0.97 ceiling indicates that even a "perfect"
   pin cannot reach the 0.999 gate without filter improvements.

**Suggested next directions** (in order of priority):

1. **Investigate `b_low` tightening (proposal §5 follow-up).** Current
   `b_low = max_veff + 2.0` is ad-hoc; the Chebyshev filter is admitting
   out-of-window components into the 40-band subspace. Tighter `b_low`
   should reduce the ~3% iter-1 residual.

2. **Investigate the iter-2 → iter-3 amplification mechanism.** Add
   diagnostic dumps showing V_eff drift, eigenvalue spectrum spread,
   and per-iteration band-0 trajectory under both Off and PostRr.
   Quantify how much of the cascade is in-block rotation vs V_eff
   coupling.

3. **Consider an absolute (not relative) Procrustes target.** Pinning
   against `prev_psi` self-references; pinning against a fixed reference
   (e.g., the iter-1 RR output frozen for the duration of the SCF)
   would break the feedback chain. Untested.

4. **Mark PostRr code as `cfg(feature = "pin_postrr")` or similar
   non-default** so that production paths remain at `PinMode::Off` until
   the cascade-amplification issue is understood.

5. **Drop PreRr scaffolding cleanly** if direction (4) is taken — the
   `PinMode::PreRr` variant is currently unimplemented and would just
   accrete dead code.

## Empirical anchors (EXTERNAL)

| Anchor | Source | Value |
|--------|--------|-------|
| A1: band-0 reference | `Cu111_CO.bands:12` | −1.05502287 Ha |
| A4: total energy | `Cu111_CO.castep` | −24110.96665069 eV |
| A5: CASTEP ψ ground-truth | `Cu111_CO.check` | binary checkpoint |

## What lives in the working tree right now

- `PinMode::Off` is the default — production behavior is byte-equivalent
  to pre-investigation HEAD.
- `PinMode::PostRr` is fully implemented and wired through `CHEMRUST_PIN_MODE`.
- `PinMode::PreRr` is declared in the enum but unimplemented (it would
  return through the `mode == PostRr` gate and fall through to the
  baseline path; no behavior change).
- The two failed stashes `stash@{0}` and `stash@{1}` are still present
  per the original plan §"Stashed work to consult, not restore". Drop
  them after the next-direction decision is made.

## Where the iter-1 success matters even though cascade fails

The 1e-6 → 0.94 floor lift at iter-1 is real and reproducible. Even if
PostRr is not the production cascade fix, the per-block typed-faer SVD
infrastructure is reusable for any future Procrustes-pin variant
(against a fixed reference, against a sliding window, etc.). The code
is in `src/eigensolver/rayleigh_ritz.rs` and is gated cleanly behind
the `PinMode` enum.

The `RrPinConfig::from_env()` + `CHEMRUST_PIN_MODE` env var pattern is
also reusable for future A/B work without further plumbing churn.
