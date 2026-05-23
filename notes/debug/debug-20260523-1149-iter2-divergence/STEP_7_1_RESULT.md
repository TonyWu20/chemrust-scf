# Step 7.1 Result Update — Reconciling Pollution Probes with Audit

**Date:** 2026-05-23

## What we know with certainty (from probes)

| Test | Result | What it proves |
|------|--------|----------------|
| ndeg=0 baseline (filter bypassed) | PASS | Downstream of filter is clean (RR, S_sub, ZHEGVD, β·ψ, density) |
| Variant A: high-energy polluter, eigs=None | Filter damps 460× | iter-1 path works for far-from-band-0 pollution |
| Variant B: low-energy polluter, eigs=None | Filter damps 33× | iter-1 path works for near-band-0 pollution |

**Both pollution variants use eigenvalues=None** (because `build_scf_state` returns
Initialized state with empty eigenvalues). This is the **iter-1** filter code path.

## What we still don't know

The §11 symptom is at **iter-2**, where:
- `eigenvalues = Some(iter-1 RR output)`
- This activates per-band code paths the pollution probes never exercised:
  - `chebyshev.rs:1411`: `b_low = eig[last]` (instead of `max_veff`)
  - `chebyshev.rs:1583`: `lam_y init = (σ₁/e)·(λ_b − c)` per band (instead of scalar `−σ₁·c/e`)
  - `chebyshev.rs:1654`: `Λ_X update += coeff·ly·λ_b` per band (instead of falling through to `coeff_c·ly + σ·σ₂·lx`)

## Updated divergence-surface status

| Item | Status | Rationale |
|------|--------|-----------|
| Filter body for eigs=None (iter-1 path) | **RULED OUT** by pollution probes |
| Filter body for eigs=Some (iter-2 path) — per-band branches | **PRIMARY CANDIDATE**, currently being tested |
| Operator order S⁻¹ before vs after H (`chebyshev.rs:1602-1611`) | Audit claim NOT confirmed by Variant B; status REDUCED to PLAUSIBLE because Variant B used eigs=None, not the full iter-2 path |
| Step 4 reconstruction Mode B omits S⁻¹ (Das main.tex:606 includes D⁻¹·R_Y) | Newly suspected: Rust Mode B at `chebyshev.rs:1687-1691` applies S⁻¹ only in Mode C, but Das Algorithm 3 line 606 applies D⁻¹ unconditionally. Mode C deviation: zero. Mode B deviation: yes. |

## Two Das-deviations now suspected

1. **Line 1602-1611 (Step 3 operator order):** Rust = `S⁻¹·(H·R_Y)`; Das = `H·(S⁻¹·R_Y) = A·D⁻¹·R_Y`.
2. **Line 1687-1691 (Step 4 reconstruction):** Mode B Rust = `R_Y + X·Λ_Y`; Das = `D⁻¹·R_Y + X·Λ_Y`. Mode C does apply D⁻¹ correctly.

In §10's resolution, Mode B *won* iter-1 SC-4-tight while Mode C lost. So:
- Mode B = "mostly Das but skips some D⁻¹ applications"
- Mode C = "full Das"

Why did Mode B beat Mode C at iter-1? Possibly because the bug in line 1602-1611
(wrong operator order) compensates partially for the missing D⁻¹ at line 1687-1691,
producing approximately-correct iter-1 output for near-converged ψ. At iter-2, with
non-trivial residuals, both bugs compound and the cascade starts.

## Currently running

`pollution_with_eigenvalues_exercises_iter2_path` — pre-populates eigenvalues from
ndeg=0 baseline, injects pollution, runs filter (Mode B, ndeg=8). Expected ~25 min.

Discriminator: low-energy polluter (band-1 → band-0) WITH eigs=Some.
- If filter still denoises: bug is upstream of filter (inputs at iter-2).
- If filter partially denoises: per-band branches have a sensitivity bug.
- If filter fails to denoise: per-band branches contain a formula error.

## Fix candidates (ranked, post-probe-pending)

1. **HIGH if iter-2-path test fails:** Add S⁻¹ application in Step 4 (line 1687-1691)
   for Mode B, matching Das main.tex line 606.
2. **MEDIUM:** Fix operator order in Step 3 (line 1602-1611): apply S⁻¹ to R_Y first.
3. **LOW:** Per-band branch formula at line 1654 (already verified to match Das line 604).
