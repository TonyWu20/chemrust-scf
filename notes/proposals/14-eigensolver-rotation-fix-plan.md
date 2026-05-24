# Proposal §14 — Detailed Implementation Plan (A/B PreRr vs PostRr Procrustes Pin)

**Status:** Implementation plan derived from `notes/proposals/14-eigensolver-rotation-fix.md`. Approved 2026-05-24 via the `/rust-development-pipeline:debug-outcomes` workflow. Supersedes §3–§4 of the parent proposal: it implements **both** the parent's PreRr design AND the swift-wolf agent's PostRr design behind a `PinMode` enum, with a cascade-test discriminator to falsify between them.

**Read order:** Read the parent proposal (`14-eigensolver-rotation-fix.md`) first for empirical state, constraints, and what failed previously. Read this plan for the executable design and verification protocol.

---

## Context

**Symptom (load-bearing).** `cascade_iter3_diagnostic_tight` — iter-3 band-0 = **−11.94 Ha** vs CASTEP A1 = **−1.05502287 Ha** (drift > 10 Ha; gate 0.1 Ha). Concurrently `overlap_iter2_against_castep` reports avg per-band S-overlap = **0.1446** vs gate **0.5**, and `subspace_projector_iter1_vs_castep` shows the Cu-3d block ratio at **0.893** (~11% span pollution + a 4-band cyclic permutation pattern within near-degenerate clusters).

**Why this matters now.** Branch `feat/phase-global-woodbury` has cleared all formula-level bugs (RR validation suite all-green at machine precision, density/V_eff machine-precision against CASTEP, T3 PASS at 9.8 mHa with V_eff substituted). The remaining cascade is **eigenvector-rotation-driven** (T-prime FAIL at 197 mHa with only D substituted confirms cascade is NOT in chemrust-hamiltonian). Without a fix here, Q2 (total-energy convergence at CASTEP tolerance) cannot land. This unblocks Phase Global-Woodbury close-out.

**Why the next attempt must be different.** Two implementations on this branch failed and were stashed:
- **Attempt 1 (F3d-narrow identity-target):** SVD'd `M = X[lo..hi, lo..hi]` (k×k diagonal sub-block of X). M is generically non-unitary → polar factor solves the wrong Procrustes problem.
- **Attempt 2 (Option B Procrustes):** SVD'd the right thing (T = ψ_prev^H · S · ψ_after_GS) but used **μ-window** block detection. By iter-2 μ drifts out of window → detector fires zero times → no-op.

The corrected approach uses **eigenvalue-spacing-only block detection** (μ-free, robust to drift) and applies the SVD result on a **mathematically sound target** (the SVD's R = U·V^H is unitary by construction). Plus an iter-1 sanity gate that gives a falsifiable per-block unit check.

**A/B variant decision (user-confirmed).** Two pin locations are theoretically motivated, with a substantive math disagreement at strict degeneracy:
- **Post-RR pin (`PinMode::PostRr`)** — apply R^H on X columns *after* ZHEGVD. ψ_new = (ψ_row · X) · R^H_full. Strict-degeneracy-robust: directly cancels ZHEGVD's arbitrary in-block gauge.
- **Pre-RR pin (`PinMode::PreRr`)** — rotate ψ_row before assembling H_sub/S_sub; rerun ZHEGVD on pinned basis. Two ZHEGVD passes. Theory says ZHEGVD re-randomizes the in-block gauge from the residual off-diagonal coupling (so this variant may be unable to cancel the gauge regardless of pre-alignment).

We implement both behind a `PinMode` enum and A/B them on the same cascade test. The discriminator is `cascade_iter3_diagnostic_tight`: whichever variant brings iter-3 band-0 within 0.1 Ha of A1 (or closer) wins. If both fail, the cascade has a contributor beyond ZHEGVD's in-block rotation (e.g., the residual ~6% filter pollution outside the 40-band window — separate follow-up).

**Anchor criteria (EXTERNAL).** From `notes/proposals/14-eigensolver-rotation-fix.md` §1.1:

| Anchor | Source | Value |
|--------|--------|-------|
| A1: band-0 reference | `Cu111_CO.bands:12` | −1.05502287 Ha |
| A2: ε_F | `Cu111_CO.bands:5` | −0.122443 Ha |
| A3: smearing width | `Cu111_CO.param:35` | 0.1 eV ≈ 3.6749 mHa |
| A4: total energy | `Cu111_CO.castep` | −24110.96665069 eV (= −886.06175 Ha) |
| A5: CASTEP ψ ground-truth | `Cu111_CO.check` (S-orthonormal) | binary checkpoint |

**Diagnostic anchors (verified pre-existing on HEAD — DO NOT duplicate).**
- `apply_s_for_test` — `src/scf.rs:779`, `src/eigensolver/chebyshev.rs:1968`. Already used by `subspace_projector_iter1_vs_castep` and `overlap_iter2_against_castep`. (swift-wolf's Step 4 proposes adding this — already on HEAD; skip.)
- `apply_s_times` — `src/eigensolver/chebyshev.rs:934` (per-block primitive used inside Gram-Schmidt at line 1736).
- Three §14 cascade tests already exist at `tests/ca_scf_convergence.rs:3182, 3253, 3475`.

---

## What is NOT the bug (per §14 §1.5, all empirically validated)

Chebyshev filter math, R-ChFSI residual update, classical 2-pass Gram-Schmidt, RR mathematical correctness (6 validation tests PASS), density formula, V_eff assembly, chemrust-hamiltonian D-screening (T-prime FAIL + T3 PASS jointly localise the cascade to chemrust-scf eigenvector rotation, **not** the upstream Hamiltonian).

The Upstream Audit Gate is therefore **already discharged**: the algorithm is verified, the cascade is downstream of the algorithm, the bug is **rotation-stability of ZHEGVD eigenvectors across SCF iterations**.

---

## Approach — Procrustes pin with A/B mode switch

### Shared math (both variants)

Block detection (identical for both modes): walk eigenvalues, find maximal contiguous runs of consecutive `|ε[j+1] − ε[j]| < eps_degen` with block size ≥ 2. `eps_degen` default 0.01 Ha (catches Cu-3d Δε ≈ 7 mHa and Fermi cluster Δε ≈ 0.3 mHa). No μ window.

For each block `[lo, hi)`:
1. Compute the overlap-derived k×k complex matrix that will be SVD'd (mode-specific — see below).
2. CPU SVD via `faer`: matrix = U·Σ·V^H.
3. `R_block = U · V^H` — unitary by construction.
4. Apply R_block to the appropriate target (mode-specific).
5. Sanity: `‖R^H · R − I‖_F < 1e-10`; on failure, skip block (no-op).

### `PinMode::PostRr` (recommended-first variant, swift-wolf's design)

Apply pin **after ZHEGVD**, on the eigenvector matrix X (held in `h_sub_dev` post-ZHEGVD), before the `ψ_new = ψ_row · X` rotation:

1. After ZHEGVD: `h_sub_dev` contains X, eigenvalues in `eigenvalues_dev`.
2. D2H eigenvalues, detect blocks.
3. If any block of size k ≥ 2 exists:
   - Compute `T = ψ_prev^H · S · ψ_row` (full n×n complex, USPP-augmented: includes the Σ_ion C_prev^H · q · C_new contribution analogous to S_sub augmentation in `rayleigh_ritz.rs:129-202`).
   - Compute `M = T · X` (full n×n) — physically `M[a, c] = ⟨ψ_prev_a | S | ψ_new_c⟩` (where ψ_new = ψ_row·X is what RR would produce un-pinned).
   - D2H `M` → `m_full_host` (n×n complex, ~205 KB for n=160).
   - For each block: extract `M_block = M[lo..hi, lo..hi]`, CPU SVD → R, then apply `X[:, lo..hi] ← X[:, lo..hi] · R^H` via small GPU gemm (the rotation matches the column convention so that `ψ_new_pinned = ψ_row · X_pinned`).
4. Then run the existing `ψ_new = ψ_row · X` gemm at `rayleigh_ritz.rs:244-265` unchanged.

**Why post-RR is theoretically robust in the strictly-degenerate limit.** In a block where H_sub = ε·I_k, ZHEGVD produces an arbitrary k×k unitary in X[:, lo..hi]. M_block = T_block · X_block; SVD-polar of M_block gives R such that ψ_new · R^H = (ψ_row · X) · R^H is Procrustes-aligned to ψ_prev's block. R can cancel any unitary in X[:, lo..hi].

### `PinMode::PreRr` (A/B comparison variant, proposal §3.1's design)

Apply pin **before assembling H_sub/S_sub**, on the post-GS `ψ_row` that `rayleigh_ritz` receives:

1. Run a "pre-pass" ZHEGVD on un-pinned ψ_row to get eigenvalues (X discarded). This is the second ZHEGVD that PreRr requires.
2. Detect blocks. If any block of size k ≥ 2 exists:
   - Compute `T_block = ψ_prev[:, lo..hi]^H · S · ψ_row[:, lo..hi]` directly (k×k complex, USPP-augmented).
   - CPU SVD → `R_block = U · V^H`.
   - Apply `ψ_row[:, lo..hi] ← ψ_row[:, lo..hi] · R_block` (mutates psi_row in place).
3. Rebuild `H_sub_new = R_full^H · H_sub_old · R_full` via algebraic block transform (R_full = identity outside detected blocks, R_block inside) — mathematically exact under any unitary basis rotation, avoids `apply_full_hamiltonian` re-call.
4. Rebuild `S_sub_new` from pinned ψ_row (cheap n×n gemm).
5. Final ZHEGVD on pinned H_sub_new, S_sub_new; use its X for `ψ_new = ψ_row_pinned · X`.

**Theoretical concern.** In a strictly-degenerate block (H_sub = ε·I_k), pre-RR alignment leaves H_sub_new still = ε·I_k inside the block. The second ZHEGVD's in-block rotation is then unconstrained — the pin's effort is **erased**. PreRr can only help when the off-diagonal residual is small enough that the second ZHEGVD's in-block rotation is small. If A1 cascade gate is hit only by PostRr, this is the explanation.

### `PinMode::Off`

Pin code path is bypassed; `rayleigh_ritz` runs identically to current HEAD. Used for the red-test baseline and for ablation runs.

### Where ψ_prev comes from (both variants)

ψ_prev = **input to chebyshev_filter at this iteration** = the basis we hand to Chebyshev before filter/GS produce ψ_after_GS. At iter-t≥2 it's the previous iteration's RR output; at iter-1 production it's the initial guess; at iter-1 in the sanity test it's CASTEP ψ. The pin runs every iteration uniformly. Cross-iteration state: scf.rs holds the iter-t Chebyshev input ψ until the same iteration's RR consumes it — no new buffer between iterations.

ψ_prev is S-orthonormal (RR output or CASTEP, both satisfy ⟨ψ|S|ψ⟩ = 1), as is ψ_after_GS (post-GS). Both in the same S-metric → standard SVD of the overlap solves orthogonal Procrustes correctly.

### Iter-1 sanity gate (the red test that must exist before any fix lands)

`pin_preserves_castep_basis_at_iter1`:
1. Load CASTEP ψ from `Cu111_CO.check`; feed as both `ψ_prev` AND the input to `chebyshev_filter`.
2. Run iter-1 with the current `PinMode` (one test instantiation per mode: `_postrr`, `_prerr`).
3. Assert per-block S-overlap `|⟨our_a | S | castep_a⟩|² > 0.999` for every band in every detected block.

Theory for each mode:
- `PostRr`: T ≈ I (Chebyshev+GS approximately preserve subspace), X ≈ I (ZHEGVD on near-diagonal H_sub), so M ≈ I → R ≈ I, X_pinned ≈ I, ψ_new ≈ ψ_row ≈ CASTEP ψ.
- `PreRr`: T_block ≈ I → R_block ≈ I → ψ_row unchanged → second ZHEGVD's X ≈ first ZHEGVD's X ≈ I → ψ_new ≈ CASTEP ψ.

**Both variants must pass iter-1 sanity gate to be considered correctly implemented.** Failure means the variant's code path has a bug; A/B comparison on cascade tests is meaningless until iter-1 passes for both.

Pre-fix baseline: with `PinMode::Off`, this test fails (`subspace_projector_iter1_vs_castep` already shows Cu-3d block ratio 0.893 → some bands fall well below 0.999).

---

## Files to modify

- `Cargo.toml` — add `faer = "0.24"` (CPU SVD for k×k blocks).
- `src/scf.rs:560-597` — save the iter-t Chebyshev input ψ slice into `prev_psi_dev` (already on device — just keep the slice alive past the chebyshev_filter call). Build `RrPinConfig` from environment (env var for A/B switching during development). Pass both to `rayleigh_ritz`. Update PCI-E byte accounting if/as needed.
- `src/eigensolver/rayleigh_ritz.rs` — add `RrPinConfig { eps_degen: f64, mode: PinMode }`, `PinMode { Off, PreRr, PostRr }`, helpers `detect_degenerate_blocks`, `compute_prev_new_overlap` (full n×n), `apply_right_unitary_to_slab` (k columns), `polar_pin_block_postrr`, `polar_pin_block_prerr`. Extend `rayleigh_ritz` signature with `prev_psi_dev: Option<&CudaSlice<CudaComplex>>` + `pin_cfg: Option<&RrPinConfig>`. Branch on mode for pin placement.
- `src/eigensolver/rayleigh_ritz.rs` — `rayleigh_ritz_with_matrices` (test-only variant at line 334): extend signature identically (`Option<>` params; behavior unchanged when None). Tests on HEAD that use this variant continue to pass.
- `tests/ca_scf_convergence.rs` — append:
  - `pin_preserves_castep_basis_at_iter1_postrr` (~50 lines)
  - `pin_preserves_castep_basis_at_iter1_prerr` (~50 lines, mirrors the postrr test with `PinMode::PreRr`)
  - One small `polar_unitary_unit_test` (CPU-only, synthetic SVD round-trip) as diagnostic self-test.

**Critical primitives that already exist (reuse, do not reimplement):**
- `apply_s_times` for column-batch S·ψ (`src/eigensolver/chebyshev.rs:934`).
- `apply_s_for_test` for CASTEP-side S·ψ_castep in the iter-1 sanity test (`src/scf.rs:779`, `src/eigensolver/chebyshev.rs:1968`). swift-wolf's Step 4 proposes re-adding these — they exist; do not duplicate.
- `BlasHandle::gemm_c64` + `op::C` / `op::N` for the small k×k overlap and rotation gemms (pattern at `rayleigh_ritz.rs:84-101`).
- USPP augmentation pattern for any prev_psi^H · S · ψ overlap: mirror `rayleigh_ritz.rs:129-202` (S_sub assembly) but with prev_psi on the left.

---

## Implementation steps (edit→check→fix loop)

1. **Add red tests first** (debug-outcomes Step 7.1 red-step requirement). Author `pin_preserves_castep_basis_at_iter1_postrr` and `_prerr` against the current code path with `PinMode::Off`. Verify both FAIL (Cu-3d block per-band overlap < 0.999). Quantify the gap.

2. **Add `faer` dependency** + diagnostic self-test `polar_unitary_unit_test`: feed a synthetic `T = U_known · diag(σ) · V_known^H` (with σ > 0 entries, k=4 or 8), check `polar_unitary(T) == U_known · V_known^H` to 1e-12. Verifies the math primitive in isolation before it's wired in (Step 5 debug-outcomes diagnostic verification).

3. **Add `RrPinConfig`, `PinMode`, and `detect_degenerate_blocks` to `rayleigh_ritz.rs`.** Implement `PinMode::Off` path = current behavior (so the existing test suite stays green with both `Option`-None and `Some(PinMode::Off)`).

4. **Plumb `prev_psi_dev` through `scf.rs`.** Save the slice that goes into `chebyshev_filter` (already on device); pass to `rayleigh_ritz` as `Option<&CudaSlice<CudaComplex>>`. No new H2D bytes.

5. **Implement `PinMode::PostRr` path.** Order: D2H eigenvalues earlier (move from current position at line 307 to right after the info check at 222), detect blocks, compute T (full n×n S-overlap, USPP-augmented), compute M = T·X (full n×n gemm), D2H M, per-block CPU SVD + `apply_right_unitary_to_slab` on X. Final `psi_new = psi_row · X_pinned` reuses the existing gemm.

6. **Verify postrr iter-1 sanity gate goes GREEN.** Run `pin_preserves_castep_basis_at_iter1_postrr`. Failure here means PostRr code path has a bug. Do not proceed to cascade tests.

7. **Implement `PinMode::PreRr` path.** First-pass ZHEGVD on un-pinned ψ_row → eigenvalues; block detection; T_block + R_block computation; in-place rotation of ψ_row blocks; H_sub block transform `R_full^H · H_sub · R_full` (block-diagonal R_full multiplication; can be done with two small gemms instead of one full n×n gemm); S_sub rebuild; second ZHEGVD; final rotation.

8. **Verify prerr iter-1 sanity gate goes GREEN.** Run `pin_preserves_castep_basis_at_iter1_prerr`. Same gate as Step 6, different code path.

9. **A/B cascade comparison.** With both iter-1 gates green, run `cascade_iter3_diagnostic_tight` and `overlap_iter2_against_castep` under each mode:
   - `PinMode::Off` (baseline, must remain at current RED values)
   - `PinMode::PreRr`
   - `PinMode::PostRr`
   
   Record iter-3 band-0 and avg S-overlap for each. The mode with smaller iter-3 drift and higher S-overlap is the winner. Theory predicts PostRr wins on cascade-iter-3 (strict-degeneracy argument); if both pass the 0.1 Ha gate, both are acceptable and we keep PostRr as default for the simpler 1-ZHEGVD code path.

10. **Run regression suite.** `iter1_drift_from_castep_state_is_bounded` (Q1) must remain GREEN; full RR validation suite must remain PASS; `rayleigh_ritz_with_matrices` callers must compile with the extended signature (Option=None preserves behavior). Q2 may still be RED if the ~6% filter pollution outside the 40-band window dominates — that's a separate follow-up.

11. **Drop both stashes** once the corrected fix lands and gates above are green: `git stash drop stash@{0} && git stash drop stash@{1}`.

---

## Verification (end-to-end)

```bash
# 0. Compile.
cargo check --workspace 2>&1

# 1. Diagnostic self-test (faer SVD round-trip).
cargo test --release polar_unitary_unit_test

# 2. Iter-1 sanity gates (the red→green discriminators, per mode).
cargo test --release --features scf_diag pin_preserves_castep_basis_at_iter1_postrr -- --nocapture --ignored
cargo test --release --features scf_diag pin_preserves_castep_basis_at_iter1_prerr  -- --nocapture --ignored

# 3. Cascade A/B (run each via env var CHEMRUST_PIN_MODE = off | prerr | postrr).
CHEMRUST_PIN_MODE=off    cargo test --release --features scf_diag cascade_iter3_diagnostic_tight -- --nocapture --ignored
CHEMRUST_PIN_MODE=prerr  cargo test --release --features scf_diag cascade_iter3_diagnostic_tight -- --nocapture --ignored
CHEMRUST_PIN_MODE=postrr cargo test --release --features scf_diag cascade_iter3_diagnostic_tight -- --nocapture --ignored

CHEMRUST_PIN_MODE=postrr cargo test --release --features scf_diag overlap_iter2_against_castep -- --nocapture --ignored
CHEMRUST_PIN_MODE=postrr cargo test --release --features scf_diag subspace_projector_iter1_vs_castep -- --nocapture --ignored

# 4. Regression: must remain GREEN under the winning mode.
CHEMRUST_PIN_MODE=postrr cargo test --release --features scf_diag iter1_drift_from_castep_state_is_bounded -- --nocapture --ignored
cargo test --release rayleigh_ritz_validation -- --nocapture
```

Discriminator gates per §14 §5:

| Test | Current (RED, `PinMode::Off`) | Target (GREEN) | Anchor |
|------|-------------------------------|----------------|--------|
| `pin_preserves_castep_basis_at_iter1_postrr` (NEW) | per-block overlap < 0.999 | > 0.999 | A5 |
| `pin_preserves_castep_basis_at_iter1_prerr` (NEW) | per-block overlap < 0.999 | > 0.999 | A5 |
| `subspace_projector_iter1_vs_castep` Cu-3d block | 0.893 | > 0.999 | A5 |
| `overlap_iter2_against_castep` avg | 0.1446 | > 0.95 | A5 |
| `cascade_iter3_diagnostic_tight` iter-3 band-0 | −11.94 Ha | within 0.1 Ha of −1.055 | A1 |
| `scf_converges_to_castep_energy_at_castep_tolerance` (Q2) | RED | within 1e-5 eV of CASTEP | A4 |

Q2 may still be RED post-fix if the residual ~6% Chebyshev filter pollution outside the 40-band window dominates; that's the §14 §5 follow-up (tighten `b_low`), tracked separately.

---

## Decision protocol after A/B run

- **Both modes pass iter-1 sanity gate AND PostRr beats PreRr on `cascade_iter3_diagnostic_tight`:** PostRr is the default. PreRr code is kept (gated by `PinMode::PreRr`) for one commit cycle as an empirical falsifier, then removed if no use case emerges. The strict-degeneracy theory is confirmed.
- **Both modes pass iter-1 AND PreRr beats PostRr on cascade:** Surprising. Theory of strict-degeneracy was wrong (perhaps the regime isn't strictly degenerate, or PostRr's M·X gemm introduces a precision issue we didn't model). Keep PreRr as default. Document the empirical reversal in `notes/failure-patterns.md`.
- **Both pass iter-1 but neither passes cascade:** The cascade has a contributor beyond ZHEGVD in-block rotation. The §1.3 finding of ~10% subspace pollution from outside the 40-band window is likely the residual driver. Pivot to `b_low` tightening (proposal §5 follow-up). Keep PostRr code as the partial fix and document.
- **One mode fails iter-1 sanity:** That mode has an implementation bug. Fix bug or remove that mode. The other mode is the default.

---

## Out of scope for this debug session

- Replacing subspace-RR with band-by-band CG (Davidson, F3c).
- Performance work on `compute_aug_density_fine` (open-followup §9).
- chemrust-hamiltonian D-screening tightening (T-prime FAIL proves it's not load-bearing here).
- Tightening Chebyshev filter `b_low` (separate follow-up after pin lands).

---

## Resolved configuration (from intake)

- **Pin variants:** Implement both `PinMode::PreRr` and `PinMode::PostRr`. A/B them on `cascade_iter3_diagnostic_tight`. Theory favors PostRr at strict degeneracy; empirics decide.
- **iter-1 production behavior:** Pin runs every iteration with ψ_prev = ψ_input. No iteration-count special-casing.
- **`eps_degen` exposure:** Field of `RrPinConfig` with default 0.01 Ha. Also overridable via env var `CHEMRUST_PIN_EPS_DEGEN` if needed during A/B (optional).
- **Mode switching:** Env var `CHEMRUST_PIN_MODE` ∈ {`off`, `prerr`, `postrr`} (follows the `CHEMRUST_FORCE_NO_EIGS` pattern at scf.rs:573). `scf.rs` reads it and builds `RrPinConfig`.
- **Stashes:** Drop both `stash@{0}` and `stash@{1}` after the winning mode lands and all gates are green.

---

## Sources

- `notes/proposals/14-eigensolver-rotation-fix.md` — original proposal (proposed PreRr as §3.1).
- `/home/tony/.claude/plans/notes-proposals-14-eigensolver-rotation-swift-wolf.md` — sibling agent's plan (proposed PostRr; this plan adopts that variant as a co-equal candidate).
- `stash@{0}` — failed Option B implementation; carries the prev_psi plumbing and the M = T·X pattern (post-RR style) that PostRr will adapt.
- `notes/debug/debug-20260524-tprime-d-injection/RESOLUTION.md` — T-prime/T3 cascade localization proof.
- `notes/failure-patterns.md` — prior debug entries.
