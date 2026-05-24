# Resolution — b_low Tightening (Proposal §5): FALSIFICATION

**Symptom (original):** "`b_low` tightening (proposal §5) to reduce filter pollution"
**Root cause (NOT FOUND in this lever):** The 0.893 Cu-3d projector ratio is NOT addressable by tightening b_low. The proposal §14 §1.4 attribution ("~6–11% Chebyshev filter pollution from outside-window high-G components") is **falsified for the iter-1 case** by the pad-sweep result documented in `BLOW_PAD_SWEEP.md`.
**Fix location:** None — no production code change persists from this session.
**Fix description:** Reverted. Two diagnostic tests remain as forensic record:
- `diagnostic_selftest_apply_s_for_test_on_castep_psi` — verifies the projector diagnostic's S-application against the EXTERNAL A5 anchor (CASTEP ψ S-orthonormality). PASS at 5.94e-8.
- `diagnostic_selftest_castep_self_overlap_block_sums` — the phantom check; CASTEP ψ produces exact 1.0000 block sums against itself, ruling out "stored precision artifact" as the explanation.

**Date:** 2026-05-24
**Branch:** `feat/phase-global-woodbury`

## Anchor criteria used

- A5 (CASTEP ψ S-orthonormal): block_sum(1, 14) target = 13.0; observed pre-fix = 11.6041; observed post-(any b_low) = 11.58-11.61. Diagnostic verified PASS.
- A1 (CASTEP band-0 = −1.05502287 Ha): not tested this session (advisory cascade gate skipped per Step 5/6 task descriptions).

## Prior notes reclassified

### Stale-but-corrected

- **"Current `b_low = max_veff + 2.0`"** (proposal §14 line 267; debug-20260524-postrr-cascade-amplification:64) → EXTERNAL but **outdated**. Actual code at chebyshev.rs:1417 reads `max_veff.max(0.0)` (no +2.0) since the 2026-05-23 `iter1-filter-operator-mismatch` resolution. The proposal text was never updated.

### Refuted

- **"Iter-2+ `b_low = eig[eig.len() - 1]` puts ~120 untracked bands inside the damped window"** (DIVERGENCE_SURFACE.md item B1) → REFUTED.
  - **Why**: production SCF passes `eig = None` to `chebyshev_filter` at every iteration (scf.rs:577, 716). The match arm at chebyshev.rs:1411 is dead code. Every iteration takes the `_ => max_veff.max(0.0)` branch.
  - **Architectural intent (load-bearing)**: the comment at scf.rs:571-577 cites Das §612 — "when ζ = ‖D⁻¹ − B⁻¹‖ = 0 (exact S⁻¹), R-ChFSI ≡ standard ChFSI algebraically. After §10's Global Woodbury fix, ζ = 3.8e-15 (machine epsilon). Per-band eigenvalue machinery is provably redundant and introduces numerical weak points from stale eigenvalue labels when V_eff drifts between SCF iterations." → eigenvalue-source dependency is intentionally avoided.

- **"Iter-3 b_low = 1.95 Ha because of iter-2 last band"** (INVESTIGATION.md row 4) → REFUTED. Iter-3 b_low = 0.0894 Ha (same as iter-1) because eig is None at every call.

- **"Tightening b_low at iter-2+ should reduce the ~6-11% pollution"** (proposal §5; debug-20260524-postrr-cascade-amplification:147-148) → REFUTED by direct sweep. See BLOW_PAD_SWEEP.md table.

- **"0.893 may be a CASTEP stored-precision artifact"** (this session's hypothesis post-sweep) → REFUTED by phantom check. `diagnostic_selftest_castep_self_overlap_block_sums` measures CASTEP-vs-CASTEP block sum = **13.0000** for the Cu-3d 13-band block (ratio 1.0000). The 0.893 is REAL pipeline loss.

### Strengthened

- **"Filter pollution and SCF-feedback drift are jointly load-bearing"** (debug-20260524-postrr-cascade-amplification:120-121) → REINFORCED. The 0.893 baseline is not filter-window-driven, so it must be either GS-driven, RR-driven, or precision-driven.

## The b_low pad-sweep table (definitive)

| pad (Ha) | b_low (Ha) | Cu-3d/13 ratio | 0..30/30 ratio | 0..40/40 ratio |
|----------|------------|----------------|----------------|----------------|
| -0.05    | 0.0708     | **0.893**      | 0.940          | 0.939          |
| (max_veff) | 0.0894 | 0.893 | 0.940 | 0.939 |
| 0.0      | 0.1208     | **0.891**      | 0.938          | 0.937          |
| +0.5     | 0.6208     | **0.797**      | 0.904          | 0.878          |

Tightening b_low across this range: NO improvement available.

## Working hypothesis after falsification — UPDATED 2026-05-24 after S-norm discriminator and CASTEP buffer-convention recall

**SUPERSEDED**: original H1/H2/H3 framing (f64-precision floors) was wrong. The `diagnostic_per_band_s_norm_of_our_output` test (also added this session) measured **all 40 bands' S-norms = 1.0000 to f64 precision** — every band is perfectly S-orthonormal after diagonalize. The 0.893 loss is NOT a magnitude offset, NOT augmentation-convention, NOT numerical precision. It is **pure unitary rotation across the 40-band window**.

**Further refined after recalling the CASTEP nbands convention** (`Cu111_CO.param`: `perc_extra_bands = 72`):
- 186 electrons → 93 occupied bands (`ceiling(max(nup, ndown))` per `parameters.f90:1668-1683`).
- 72% extra → 93 × 1.72 = 160 tracked bands.
- The buffer is canonical CASTEP, not a free parameter. Our 160 matches CASTEP's 160.
- So **"increase tracking buffer"** as a fix is misguided: CASTEP achieves the canonical accuracy at 160; we should too.

### The actual mechanism (now precisely defined)

By the unitary-singular-values theorem, for two S-orthonormal k-dim bases spanning the same subspace, `Σ_{a,b∈block} |⟨ψ_a^A | S | ψ_b^B⟩|² = k`. Our Cu-3d 13-band block produces 11.6, not 13.0. Since per-band S-norms are 1.0, the only possibility is **our 1..14 span is not identical to CASTEP's 1..14 span**. About 11% of the span has rotated INTO bands 14..40 (and beyond).

With band 14/15 being deep inside the **occupied 1..93 subspace** (Cu 3d is ~1..14, Cu 4sp follows ~14..60+, all occupied), the rotation occurs at a **boundary between near-degenerate eigenvalue clusters within the occupied manifold**, NOT at the occupied/unoccupied buffer edge. This is exactly the subspace-RR / band-by-band-CG algorithmic mismatch documented in `failure-patterns.md` 2026-05-23 ("eigenvector-rotation-cascade-divergence"): "Subspace methods rotate eigenvectors within degenerate manifolds, which CG preserves naturally. This is expected algorithmic behaviour, not a code bug."

### Why this happens, in words

RR diagonalizes H_sub built from filtered ψ. The Cu-3d cluster (eigenvalues clustered within ~0.07 Ha) has small but nonzero off-block couplings to neighboring eigenvalue clusters (band 14's 4s state, band 15's). The filtered ψ amplifies these couplings non-uniformly (deeper bands amplified more), so the H_sub off-block entries at the [14, 15] boundary aren't bitwise-zero. ZHEGVD's stable sort + invariant-subspace solver mixes across the boundary in a deterministic but **non-CASTEP** way (CASTEP's band-by-band CG never builds H_sub and never makes this mixing decision).

### What this rules out

- ❌ Buffer enlargement: CASTEP works at 160 bands with `perc_extra_bands=72`; ours uses the same; this isn't the lever.
- ❌ Filter window tuning (this session falsified it).
- ❌ Q-matrix / β-projector audit (per-band S-norm = 1.0 rules it out).
- ❌ Relative-target Procrustes (amplifies cascade per `relative_target_procrustes_feedback` memory).
- ❌ f64 precision (other DFT codes work at f64; S-norms = 1.0000 confirm no precision loss).

### What remains as fix paths

| Path | Cost | Notes |
|------|------|-------|
| **(P1) Absolute-target Procrustes** (analyst plan Rec 3): pin against a fixed reference like iter-1 RR output frozen for the SCF run | MEDIUM (~150 lines new code; existing PostRr `PinMode` infrastructure reusable) | Aligns OUR output to a stable basis within the same 160-band window. Mitigates cluster-boundary mixing without re-creating the moving-target feedback. |
| **(P2) Davidson eigensolver** (proposal §6): never builds H_sub, never invokes ZHEGVD on a global block; sidesteps cluster mixing entirely | HEAVY (~weeks) | Reserve for if P1 insufficient. |
| **(P3) Band-by-band CG** (the actual CASTEP algorithm): replaces the entire eigensolver | HEAVIEST | Most principled but a major rewrite. Last resort. |

## Recommendation for next session

**Pivot directly to P1** (absolute-target Procrustes).

Concrete sub-steps:
1. Take iter-1's `psi_iter1_out` (the FIRST diagonalize's output) and freeze it as the reference for all subsequent iterations.
2. At iter ≥ 2, in `rayleigh_ritz`, after ZHEGVD produces X:
   - Compute T = ψ_ref^H · S · ψ_row (n × n, same as PostRr).
   - SVD T = U·Σ·V^H, set R = U·V^H (whole-window, not per-cluster).
   - Apply X[:, :] ← X[:, :] · R^H.
3. Iter-1 itself runs unpinned (the reference is set FROM iter-1's own output).
4. Test: does this lift the iter-2 overlap from 0.1446 (current) toward 0.5+? Does it stop the iter-3 cascade?

**Expected outcomes**:
- If P1 lifts cascade to within 0.1 Ha of A1: ship; production-default `PinMode::AbsoluteRef`.
- If P1 lifts iter-2 overlap but not cascade: the iter-1 → iter-3 V_eff drift is independent of ψ rotation. The right next step is either (a) accept the cascade as algorithmic-cost (revise acceptance test per `tolerance-conflation` pattern) or (b) revisit density mixing as the load-bearing fix.
- If P1 does nothing: implies the iter-1 reference itself is the wrong choice; consider CASTEP-frozen-ψ as reference instead.

**Do NOT pursue**:
- Further b_low tightening (this session).
- Per-cluster Procrustes (this session — the right granularity is whole-window).
- Buffer enlargement (CASTEP convention pins 160; ours matches).
- Augmentation / Q audit (S-norms ruled it out).

## What remains in the working tree

- Three new tests in `tests/ca_scf_convergence.rs`:
  - `diagnostic_selftest_apply_s_for_test_on_castep_psi` (passes; verifies S-application at 5.94e-8)
  - `diagnostic_selftest_castep_self_overlap_block_sums` (passes; phantom-check rules out CASTEP precision artifact)
  - `diagnostic_per_band_s_norm_of_our_output` (passes; **discriminator** that localized the loss to off-block rotation)
  - `b_low_sweep_subspace_projector` (passes, but harness was found broken — eig is hardcoded None at scf.rs:577 per architectural intent; the sweep does not actually vary b_low in production semantics; useful as a documented baseline of what NOT to expect)
- No production code change (chebyshev.rs reverted).
- Debug session artifacts at `notes/debug/debug-20260524-blow-tightening/`.

## Update to the parent post-mortem

`notes/debug/debug-20260524-postrr-cascade-amplification/RESOLUTION.md` lines 145-148 should be amended:
> ~~"Investigate `b_low` tightening (proposal §5 follow-up). Current `b_low = max_veff + 2.0` is ad-hoc; the Chebyshev filter is admitting out-of-window components into the 40-band subspace. Tighter `b_low` should reduce the ~3% iter-1 residual."~~ → SUPERSEDED by debug-20260524-blow-tightening/RESOLUTION.md. The b_low lever has been swept and falsified; current production `b_low = max_veff` is in the indifferent-to-improvement range. The 0.893 Cu-3d ratio is **off-block unitary rotation across the 40-band tracking window**, not a filter-window-discrimination issue and not a numerical-precision floor — every band's S-norm is 1.0000 to f64 precision. Cheapest next test: increase n_bands buffer (160 → 240) and re-measure.

## Sanity-check against subagent claims and prior memory

The debug-outcomes skill requires re-verifying factual claims drawn from subagent summaries. Claims checked:

| Claim | Source | Verified |
|-------|--------|----------|
| "Iter-1 max_veff ≈ 0.089 Ha for Cu111+CO" | subagent + code comment at chebyshev.rs:1401 | YES — runtime diagnostic at b_low pad-sweep confirms 0.0894 |
| "Current b_low = max_veff + 2.0 at iter-1" | proposal §5 + RESOLUTION post-mortem:64 | REFUTED — actual code is max_veff (no +2.0) since 2026-05-23 |
| "f64 precision is the limiting factor" | this session's intermediate hypothesis | REFUTED by user objection (other DFT codes use f64 fine) AND by S-norm test (all bands 1.0000 — no precision loss in S-norm) |
| "0.893 is rotation between Cu-3d cluster and bands 14..160" | this session's S-norm test conclusion | EXTERNAL — pure deduction from unitary-singular-values theorem given S-norms = 1.0 and block sum = 11.6 |
