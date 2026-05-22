# Divergence-Surface Enumeration

Every place where the local pipeline could disagree with CASTEP for the
iter-1-band-1 symptom class. Each item is classified as either ruled out by
an anchor from `CRITERIA.md` or to be tested in the A/B/C discriminator
sweep.

The point of breadth-first enumeration: prevent depth-first hypothesis
chaining from prior sessions' biases. The §10 followup heavily pre-favored
"S⁻¹ accuracy" as the bug; once that was fixed, we want all *other* surfaces
named explicitly so we don't redirect into "must be the next single thing" too
quickly.

---

## A. Filter operator framing (the prime suspect)

### A1. Step 3 recurrence operator: `H` vs `S⁻¹·H`

- **Code:** `src/eigensolver/chebyshev.rs:1554` calls `apply_full_hamiltonian`;
  line 1559 conditionally calls `apply_s_inverse` if `use_sinv_filter=true`.
  Production has `use_sinv_filter=false` (`src/scf.rs:524`).
- **Status:** **TO BE TESTED** in Mode B and Mode C.
- **Rationale:** Lanczos already applies S⁻¹·H (`chebyshev.rs:472-476` per
  commit `c8846cc`). Filter operator mismatch with bound estimator is the
  most direct operator-framing surface.

### A2. Step 4 reconstruction operator

- **Code:** `chebyshev.rs:1640-1644`: `X_new = R_Y + X·Λ_Y`. Comment at line
  1633 acknowledges paper specifies `S⁻¹·R_Y` and notes "if iter-3 diverges,
  TASK-D4 restores S⁻¹ here."
- **Status:** **TO BE TESTED** in Mode C only (Mode B keeps bare reconstruction).
- **Rationale:** Das Algorithm 3 line 607 explicitly puts `D⁻¹` here. Omitting
  it leaves R_Y carrying an implicit S-factor (from Y = H·X − S·X·Λ in Step 1)
  that X·Λ_Y does not carry — dimensional asymmetry the paper resolves by
  applying S⁻¹ to R_Y on the way out.

### A3. Λ source for band shifts

- **Code:** `chebyshev.rs:1467-1496` computes `h_eig` per-band Rayleigh
  quotients ⟨ψ_j, H·ψ_j⟩; `:1534-1539` uses `h_eig` to initialize Λ_Y.
- **Status:** **TO BE TESTED** in Mode C (replaces `h_eig` with the
  `eigenvalues` slice from previous RR's ZHEGVD generalized solve).
- **Rationale:** Das Algorithm 3 line 600 specifies `Λ_Y = (σ₁/e)·(Λ − c·I)`
  where Λ is the **generalized** eigenvalue diagonal from the previous RR. On
  iter-1 with `eigenvalues=None` we fall back to `h_eig`; on iter-2+ we should
  use the RR output. Currently `h_eig` is used unconditionally even when
  generalized eigenvalues are available, which is its own framing inconsistency.

### A4. Lanczos operator (already on S⁻¹·H)

- **Code:** `chebyshev.rs:467-476`. Applies H then S⁻¹.
- **Status:** **RULED OUT BY ANCHOR ALG-2** — the algorithm wants Lanczos
  bounds on S⁻¹·H, and that is what we have. Not a divergence surface.

### A5. Gershgorin fallback

- **Code:** `chebyshev.rs:1346-1349, 350-357`. `gmax² + (max_veff − min_veff)`.
- **Status:** **RULED OUT for the iter-1 explosion**. The current log shows
  Lanczos is used (`scaled = 21.20 Ha < gershgorin = 116.49 Ha`, capped_by =
  false). Gershgorin is only a cap, not the active b_up. The Gershgorin formula
  itself is bare-H-flavored (max_veff is from real-space V_eff which appears
  in bare H), so if the cap fired it would be wrong for S⁻¹·H — but it didn't.

### A6. b_low first-iteration choice (T_k midpoint vs Gershgorin)

- **Code:** `chebyshev.rs:1373-1377`. On first call, b_low = `0.5·(ritz_min +
  ritz_max) = 7.34 Ha`.
- **Status:** **AUDIT BUT DEFER** (per `FIX_PLAN.md` §4b). Reduces filter
  selectivity (low unoccupied states between ~0.5 and 7.34 Ha get amplified)
  but does not by itself produce divergence. Revisit only if Mode B/C wins
  iter-1 but iter-3+ stalls. Cannot be fixed before Step 3 operator is picked
  because the spectrum b_low bounds depends on the operator.

### A7. R-ChFSI vs standard ChFSI form

- **Code:** Entire `chebyshev_filter` body uses Algorithm 3 (residual recurrence,
  buffers buf_y/buf_rx/buf_ry, two swaps per step).
- **Status:** **RULED OUT BY ANCHOR ALG-4** — at ζ = 3.8e-15 the two are
  algebraically equivalent. Not a divergence surface. (Future simplification
  noted in PHASE_PLAN.md:204-207.)

---

## B. Spectral bounds

### B1. b_up Lanczos value

- Already 21.20 Ha (S⁻¹·H spectrum). Consistent with operator if Modes B/C are
  picked. Inconsistent with Mode A's bare-H polynomial. **RESOLVED BY THE A/B/C
  CHOICE itself** — not a separate surface.

### B2. b_up cap by Gershgorin

- See A5. **RULED OUT** for current log.

### B3. λ_min in σ formula

- `chebyshev.rs:1392`: `raw_lambda_min = ritz_min * 0.8`.
- **Status:** TO BE OBSERVED in the diagnostic. If σ explodes (γ < |σ|) the
  recurrence becomes unstable independent of operator. Current log gives
  finite σ so this is not the active cause, but the diagnostic should print
  σ/c/e/γ for every mode to confirm.

---

## C. RR (Rayleigh–Ritz) downstream of filter

### C1. S_sub assembly: `Ψ^H · S · Ψ` vs `Ψ^H · Ψ`

- **Code:** `src/eigensolver/rayleigh_ritz.rs` (not read in this session).
- **Status:** TO BE INSPECTED if all three modes fail SC-4. Per
  `notes/open-followups.md:341-346` this is one of the listed candidates for
  upstream audit. Out of scope for the diagnostic-first sweep but on the
  escalation path.

### C2. ZHEGVD vs ZHEEVD

- **Status:** TO BE INSPECTED on escalation. ZHEGVD is the generalized
  eigensolver; ZHEEVD with implicit S = I would be wrong for USPP. Verifiable
  by reading the rayleigh_ritz.rs solver call site.

### C3. β projector cache contracted vs raw

- **Code:** `src/eigensolver/vnl_data.rs::VnlBatchData`.
- **Status:** Per `notes/open-followups.md §8 Next Steps #2`: "Verify the β
  projections currently stored in VnlBatchData carry the raw `⟨β_{IL}|ψ_b⟩`
  values (not contracted into D · ⟨β|ψ⟩)." TO BE INSPECTED on escalation.

---

## D. Gram–Schmidt orthonormalization (post-filter)

### D1. S-Gram-Schmidt vs L²-Gram-Schmidt

- **Code:** `chebyshev.rs:1649-1722`. Uses S-inner-product (line 1675-1687).
- **Status:** **AUDIT BUT DEFER** (per `FIX_PLAN.md` §4a). Correct under any
  mode. Revisit only if iter-4+ drift occurs.

### D2. Two-pass classical vs modified GS

- Two-pass classical (lines 1666 outer loop). Numerically stable enough for our
  scale. **RULED OUT for iter-1 explosion**.

---

## E. Initial wavefunction conventions

### E1. CASTEP `.check` ψ S-orthonormal vs L²-orthonormal

- Memory `castep_check_continuation_convention.md` says S-orthonormal, no
  re-orthonormalization on read. The S-Gram-Schmidt at the end of
  `chebyshev_filter` (D1) re-S-orthonormalizes the filtered output anyway, so
  the input convention only affects iter-1 *before* filtering.
- **Status:** RULED OUT for the explosion (the explosion happens *during*
  filtering, before output GS); but flagged as a re-verification target if all
  three modes fail and we escalate.

### E2. Density unit convention (raw `ρ × Ω` vs e/Bohr³)

- Resolved in `notes/open-followups.md §2`. Per the same-input controlled
  experiment (commit `fa68980`), density code is correct. Independent of
  filter operator. **RULED OUT BY ANCHOR SC-density.**

---

## F. Reference-implementation conditional-skip table

Per the debug-outcomes upstream-audit rule (Step 6 of the skill), the
conditional branches in CASTEP's filter / RR and our handling:

| Reference (CASTEP `wave_diag.F90`, `chefsi.F90`, etc.) | Our handling | Status |
|---|---|---|
| Generalized eigenvalue branch in `wave_orthog`/`zhegv` | We use ZHEGVD (need to verify in C2) | TO BE VERIFIED on escalation |
| `do_gauge_transform` after RR | Not present in our code | Probably orthogonal — gauge fixing affects degenerate subspaces only |
| Augmentation density Q-on-grid path | Wired in `src/density.rs::compute_aug_density_fine`, GPU-cached via `QSfCache` (see `notes/open-followups.md §9`) | RESOLVED |
| `kpoint_weighted_*` accumulation | Single Γ-point in our test fixture; no k-point sum | RULED OUT for this fixture |

**Empty cells in this table become red flags only on escalation.** Currently
we have a primary hypothesis (operator framing, A1/A2/A3) and the discipline
of the debug-outcomes pattern is to test that first before broadening.

---

## Summary of testing assignments

| Surface | Mode A tests | Mode B tests | Mode C tests | Comment |
|---|---|---|---|---|
| A1 (Step 3 op) | H | S⁻¹·H | S⁻¹·H | The prime variable |
| A2 (Step 4 op) | bare | bare | S⁻¹ | Mode C only |
| A3 (Λ source) | h_eig | h_eig | RR Λ | Mode C only |
| All others | unchanged | unchanged | unchanged | Held constant for clean attribution |

Three modes test 1–3 variables simultaneously. The discriminator (SC-4-tight)
distinguishes which combination matters. Two-pass tie-break (SC-7-tight)
distinguishes B from C if both pass iter-1.
