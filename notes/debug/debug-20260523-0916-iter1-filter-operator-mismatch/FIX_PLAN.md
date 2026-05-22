# Fix Plan — Iter-1 Filter Operator Mismatch

**Branch:** `feat/phase-global-woodbury` (HEAD `e1d1424`)
**Symptom:** `fixed_point_matches_castep_energy` log
`/tmp/scf-diag-global-woodbury-0523-0524.log` — iter-1 RR band-1 = −1.69 Ha
vs CASTEP reference −1.06 Ha (|Δ| = 0.63 Ha → **13× the 0.05 Ha SC-4 gate**),
R-ChFSI norm ratio ~5×/step k=2..8, iter-2 V_eff range = 29.2 Ha vs target
≤ 9.5 Ha.

**Strategy:** Diagnostic-first A/B/C sweep over the filter-operator framing.
No production commit until the discriminator picks a winner.

This file consolidates the merged plan (effervescent-journal + zazzy-wren
history + lobster's b_low concern) and references the four sibling artifacts:
- [INVESTIGATION.md](./INVESTIGATION.md) — prior-note classification
- [CRITERIA.md](./CRITERIA.md) — EXTERNAL anchors + SC-4-tight gate
- [DIVERGENCE_SURFACE.md](./DIVERGENCE_SURFACE.md) — surfaces enumerated
- [DIAGNOSTIC_SELFTEST.md](./DIAGNOSTIC_SELFTEST.md) — diagnostic verification

---

## Context

Three things changed the landscape since the bare-H R-ChFSI experiment was
justified:

1. **Goal 1a (commit `986bc96`) collapsed ζ = ‖S⁻¹·S − I‖_∞ from 0.014 → 3.8e-15** —
   exact global Woodbury inverse via cuSOLVER LU. Confirmed by
   `s_inv_s_identity_test`, gate tightened to 1e-10 (commit `f21127f`).
2. **The original bare-H rationale is empirically stale.** When bare-H produced
   cleaner numbers than S⁻¹·H at run-1010, the operative S⁻¹ carried 1.4%
   per-application noise (`notes/plans/phase-global-woodbury/PHASE_PLAN.md:16-17`).
   Das 2025 §3.2 Theorem 3.4 (`reference_paper/extracted/das-2025-rchfsi/main.tex:573-583`)
   shows naïve ChFSI stagnates at O(ζ); Das line 612 shows ChFSI ≡ R-ChFSI
   when ζ → 0. The "bare H beats S⁻¹·H" observation lived inside the regime
   where Das's stagnation theorem applied. That regime is gone.
3. **Both reference algorithms prescribe S⁻¹·H, not H.** Levitt-Torrent
   (`abinit.tex:678-680`) Algorithm 1 explicitly applies `S⁻¹·H` in the
   recurrence, with `λ_+` an upper bound on the **generalized** spectrum
   (= spectrum of `S⁻¹·H`). Das (`main.tex:597-607`) Algorithm 3 applies
   `D⁻¹·A` (= `S⁻¹·H`) in Step 3 and **also** `D⁻¹·R_Y` in Step 4
   reconstruction. Neither paper sanctions a "bare H" R-ChFSI variant.

Our current `use_sinv_filter=false` mode (`src/scf.rs:524`) computes spectral
parameters σ/c/e from Lanczos bounds on `S⁻¹·H` (chebyshev.rs:472-476, post
commit `c8846cc`) but applies them to a polynomial of bare H — a guaranteed
mismatch between filter window `[7.34, 21.20] Ha` and bare-H spectrum
`[~0, ~117] Ha`. That is the most likely engine of the 5×/step norm growth in
the log (k=2 ratio 3.97, k=3-8 settling near 4.9).

The bare-H R-ChFSI phase
(`notes/plans/phase-rchfsi-bare-h/TASKS.md:31-32`) pre-locked the decision tree:
- Iter-2 OK + iter-3 D_screened > 10 Ha → "Apply TASK-D4 (restore S⁻¹ in Step 4 only)"
- **SC-4 fails (band-1 > 0.05 Ha) → "Bare-H hypothesis insufficient. Investigate second compounding bug."**

The current log triggers SC-4 outright. Per the decision gate, the bare-H
hypothesis is falsified. The "second compounding bug" was the ζ = 1.4% inexact
S⁻¹ — already fixed. The plan reverses the experiment: re-run the A/B
comparison the bare-H phase originally killed, plus a third mode that fully
aligns with Das Algorithm 3, and pick by data.

This is not "trust the paper over the empirical result." It is "the empirical
result that motivated bare-H came from a regime that no longer exists;
reproduce the comparison in the current regime and let the discriminator pick."

### History of ChFSI Adaptation for USPP (how we got here)

| Phase | Recurrence operator | Lanczos operator | S⁻¹ accuracy ζ | Outcome |
|-------|--------------------|--------------------|------------------|---------|
| Standard ChFSI, bare H | H | H | N/A | Wrong: ψ_n are eigenvectors of S⁻¹·H, not H — `T_p(H)` filters the wrong subspace |
| ChFSI with S⁻¹·H (first attempt) | S⁻¹·H | H (Gershgorin) | ~0.014 (m_inv→s_inv typo) | Wrong: bounds for H, S⁻¹ buggy → drift via Das Thm 3.2 |
| R-ChFSI (inexact-tolerant) | S⁻¹·H approx | H | ~0.014 | Wrong: S⁻¹ still buggy; R-ChFSI's tolerance can't rescue a sign-level error |
| Bare-H R-ChFSI (commit `4e54d71`–`6bd2c58`) | H | H | ~0.014 | Cleared filter/Lanczos S⁻¹ paths; *empirically* beat S⁻¹·H at run-1010 under ζ ≈ 0.014 |
| **Global Woodbury (current HEAD)** | **H (bare)** | **S⁻¹·H** | **3.8e-15** | **Mismatch — bounds and operator now disagree** |

The "bare-H beats S⁻¹·H" observation belonged to row 4 (ζ ≈ 0.014). The
current row's mismatch — Lanczos on S⁻¹·H but recurrence on bare H — is
unprecedented and was introduced by commit `c8846cc` wiring S⁻¹ into Lanczos
(Goal 1b) while keeping the filter A/B gate locked to bare H (Goal 1c).

**Why bare H is wrong for USPP in principle** (Levitt-Torrent `abinit.tex:653-657`):
> "If we denote by Λ and P the eigenvalues and eigenvectors of the eigenproblem
> Hψ = λSψ, then S⁻¹H = PΛP⁻¹. Therefore, T_n(S⁻¹H)ψ = PT_n(Λ)P⁻¹ψ will have
> its eigencomponents filtered by the spectral filter T_n."

H·ψ_n = ε_n·S·ψ_n ≠ ε_n·ψ_n — ψ_n are not eigenvectors of bare H. `T_p(H)`
does not amplify the wanted generalized eigenvectors; under exact S⁻¹, it
should be strictly worse. The bare-H phase's empirical win required the
ζ = 0.014 regime; we are no longer there.

---

## Strategy: Diagnostic-first A/B/C

Add a single test that runs **iter-1 only** under three filter-operator modes,
dumps lowest-10 RR band energies vs CASTEP `.bands`, and gates on a tight
per-band threshold. The winner becomes the production path; the other two get
deleted along with their dead-code branches.

### Mode definitions

| Mode | Step 3 filter operator (line 1554) | Step 4 reconstruction (line 1640) | Λ source for band shifts |
|------|-------------------------------------|-----------------------------------|--------------------------|
| **A — current (bare-H)** | `H·R_Y` | `X_new = R_Y + X·Λ_Y` | `h_eig` per-band Rayleigh quotients ⟨ψ_j, H·ψ_j⟩ (chebyshev.rs:1467-1496) |
| **B — minimal flip** | `S⁻¹·H·R_Y` (toggle `use_sinv_filter` at line 1559) | unchanged: `X_new = R_Y + X·Λ_Y` | `h_eig` (unchanged) |
| **C — full Das Alg 3** | `S⁻¹·H·R_Y` | `X_new = S⁻¹·R_Y + X·Λ_Y` (insert `apply_s_inverse(&mut buf_a, ...)` between line 1640 dtod and line 1641 axpy) | RR `eigenvalues` slice (generalized Λ from previous iteration's ZHEGVD; falls back to `h_eig` only when `eigenvalues = None` on iter-1 first call) |

Mode A is the current code (no change). Mode B is one bool flip. Mode C
touches three sites: filter-Step-3 conditional → unconditional, reconstruction
adds S⁻¹, and the `h_eig` shift list at line 1534 is replaced by the
generalized eigenvalues already passed in via `eigenvalues: Option<&[f64]>`.

The Lanczos estimator already operates on `S⁻¹·H` (chebyshev.rs:472-476), so
Modes B and C make σ/c/e parameters consistent with the operator they apply.
Mode A keeps the existing inconsistency.

### Discriminator gate (SC-4-tight)

**Per-band |Δ| over the lowest 10 bands vs CASTEP `.bands`, iter-1 only:**

```
|band_j_iter1 − band_j_castep| < 0.05 Ha   for j ∈ [0, 10)
```

CASTEP reference values are in
`/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands`.
Three bands isn't enough to distinguish algebraic equivalence vs accidental
agreement on one state; ten gives a statistically meaningful signal at low cost.

**Tie-break (if two modes pass):** prefer Mode C (paper-aligned per ALG-3/4).
If A and C both pass at iter-1 but A passes SC-7 (D_screened iter-3 < 10 Ha)
and C does not, that is itself diagnostic of a *third* bug we have not yet
identified — escalate, do not commit.

**Falsification:** if all three modes fail SC-4-tight, the bug is upstream of
the filter (RR S_sub assembly, β projector convention, Q matrix indexing, or
Lanczos T_k construction). Branch into upstream audit per
`DIVERGENCE_SURFACE.md` §C.

---

## Implementation steps

### Step 1: Diagnostic harness (no production code change)

Add an ignored, release-only test `iter1_filter_mode_sweep` next to
`fixed_point_matches_castep_energy` in `tests/ca_scf_convergence.rs`. The test:

1. Builds `ScfIteration` from the CASTEP fixture (reuses `build_scf_state`
   from `tests/fixtures/cu111_co.rs`).
2. Runs `diagonalize(8, None)` three times with the mode injected per
   `FilterMode` enum (defined below).
3. Captures the eigenvalue array from each run, prints a per-band table to
   stderr, writes a CSV to `/tmp/iter1-mode-sweep-<timestamp>.csv` for
   follow-on analysis.
4. Self-test gate: Mode A's band-1 must reproduce −1.69 Ha within 0.1 Ha (per
   `DIAGNOSTIC_SELFTEST.md` D6) before reading B/C.
5. Asserts SC-4-tight per mode (collect failures, panic at the end with a
   summary table — so we get pass/fail labels for all three even if A fails).

To inject the Mode C variations cleanly without permanent API churn, add an
`enum FilterMode { BareH, SinvHKeepHEig, SinvHFullDas }` parameter to
`chebyshev_filter` (replaces the bool `use_sinv_filter`). `diagonalize` gains
an analogous parameter with a default. The diagnostic test passes the mode
variants in turn; the production call site at `src/scf.rs:524` keeps the
existing default for now (we only commit the production switch after the
gate picks a winner).

### Step 2: Run, decide

```bash
cargo test --release --features scf_diag -p chemrust-scf -- \
  --ignored iter1_filter_mode_sweep --nocapture \
  | tee /tmp/iter1-mode-sweep-$(date +%Y%m%d-%H%M).log
```

Tabulate band-0 through band-9 |Δ| for each mode. Pick the mode that lands
all 10 bands within 0.05 Ha. If C passes by paper alignment but A also
passes, run iter-3 SC-7-tight against both as tiebreak (B and C should agree
if Das line 612 holds at our ζ).

### Step 3: Commit the winner, delete the losers

Once the discriminator picks a single mode:

- Replace `chebyshev_filter`'s `FilterMode` parameter with the winning code
  path (no enum, no flag — one path).
- Delete the two losing branches and the `h_eig` per-band Rayleigh-quotient
  block at `chebyshev.rs:1467-1496` if Mode C wins (Λ now comes from RR;
  h_eig is dead).
- Update the SC-4 / SC-7 assertions in `fixed_point_matches_castep_energy` to
  the same 0.05 Ha gate, run end-to-end.
- Remove `#[allow(dead_code)]` from `apply_s_inverse` (no longer dead).
- Update `notes/open-followups.md §10` with the resolution and link to the
  new test.
- Write `RESOLUTION.md` in this directory with the winning mode, the fix
  location, and a reclassification of any prior claims that were stale.

### Step 4: Audit but defer — two secondary candidates

Per user clarification, two adjacent concerns are *flagged* but not changed in
this fix. Adding them to the same commit conflates discriminator signal across
multiple variables; we want the iter-1 A/B/C gate to attribute cleanly to the
filter-operator choice alone.

**4a. S-Gram-Schmidt** (`chebyshev.rs:1649-1722`). Uses S-inner-product
`⟨x, y⟩_S = x†·S·y` (chebyshev.rs:1675-1687), which is correct under any of
Mode A/B/C. The two-pass classical Gram-Schmidt loop allocates a 60k-PW
scratch per band — irrelevant for correctness here. Add a follow-up note to
`notes/open-followups.md §11` to revisit if the discriminator winner passes
iter-1 but full SCF drifts at iter-4+.

**4b. b_low choice on first iteration** (`chebyshev.rs:1373-1377`). On the
first Chebyshev call, `eigenvalues` is `None` so b_low falls back to the T_k
midpoint of the Lanczos-on-S⁻¹·H tridiagonal: `0.5·(ritz_min + ritz_max) =
7.34 Ha` in the current log. The Fermi level for Cu111+CO is ≈ 0.18 Ha (read
from `.bands` header), so the band `[b_low=7.34, b_up=21.20]` to be
*attenuated* sits well above the band gap, and unoccupied states between
roughly 0.5 and 7.34 Ha are *amplified* along with the occupied set. This
reduces filter selectivity but does not by itself produce the iter-1 explosion
(it produces sub-optimal filtering, not divergence).

The Gershgorin estimate `max_veff + 2.0 ≈ 2.09 Ha` (line 350-357 of
`compute_spectral_bounds`) is tighter and more physically grounded for b_low.
Adopting it would require the spectrum to be **S⁻¹·H** (since b_up is already
on S⁻¹·H), not bare H — currently the Gershgorin path mixes `max_veff` (a
real-space quantity bounding V in bare H) with the S⁻¹·H spectrum, which is
its own inconsistency.

Defer: if the A/B/C discriminator picks Mode B or C and iter-1 SC-4 passes,
b_low remains a candidate for the next round only if iter-3+ shows convergence
stalling. If Mode A wins, b_low is irrelevant. If *all three modes fail*
iter-1, b_low joins the upstream-audit list (along with β projector
normalization, Q G=0, S_sub assembly). Recorded in the same
`open-followups.md §11` follow-up entry as 4a, with the explicit caveat that
fixing b_low without first picking a recurrence operator is a category mistake
— the spectrum it bounds depends on the operator choice in Step 3.

---

## Critical files

| Path | Change |
|------|--------|
| `tests/ca_scf_convergence.rs` | New `iter1_filter_mode_sweep` test (ignored, release). |
| `src/eigensolver/chebyshev.rs:1271` (signature), `:1559-1563` (Step 3 conditional), `:1633-1644` (Step 4 reconstruction), `:1467-1496` (h_eig block), `:1530-1540` (Λ_Y init) | Add `FilterMode` enum parameter; gate Step-3 S⁻¹ apply, Step-4 S⁻¹ apply, and Λ source per mode. |
| `src/scf.rs:518-525` | `chebyshev_filter` call site: pass mode param (default = current behavior until winner picked). |
| `src/scf.rs:427-432` | `diagonalize` signature: optional `FilterMode` arg, defaults to current. |
| Post-decision (Step 3): all of the above collapse to a single code path; `FilterMode` deleted. | |

## Reference papers cited

- **Levitt & Torrent (2015)** — `reference_paper/extracted/levitt-torrent-2015/abinit.tex:653-704`. Algorithm 1 lines 678-680 are the literal recurrence for `S⁻¹·H`; lines 706-735 derive the Woodbury formula for `S⁻¹` (which we now implement exactly). The "filter on `S⁻¹·H`, not `H`" requirement is at lines 653-657.
- **Das et al. (2025)** — `reference_paper/extracted/das-2025-rchfsi/main.tex:586-612`. Algorithm 3 lines 597-608 are the residual-form recurrence; line 603 has `D⁻¹` in Step 3, line 607 has `D⁻¹` in Step 4 reconstruction. Line 612 ("ChFSI and R-ChFSI are algebraically equivalent" when D⁻¹ = B⁻¹) is what makes Mode C the safe paper-aligned target now that ζ = 4e-15.
- **Zhou (2014)** — `reference_paper/zhou2014-chebyshev-filtered-subspace-iteration-jcp.zip`. Algorithm 4.1 §7.2 (b_low from previous iter's max Ritz) and Algorithm 5.1 eq.(13) (T_k midpoint bootstrap) — already correctly implemented at `chebyshev.rs:1366-1377`, no change.

## Reuse / existing infrastructure

- `apply_s_inverse` (`chebyshev.rs:830`): exact global-Woodbury inverse. Already wired into Lanczos at line 472-476. Modes B/C just call it from one or two more places.
- `apply_s_times` (`chebyshev.rs:912`): used by Step 1 residual `Y = H·X − S·X·Λ` and Gram-Schmidt; no change needed for any mode.
- `eigenvalues: Option<&[f64]>` already plumbed through `chebyshev_filter` (line 1265). On iter-1 it is `None`; on iter-2+ it carries the previous RR's generalized eigenvalues. Mode C uses these directly in place of `h_eig`.
- `fixed_point_matches_castep_energy` test loads the same fixtures we need; the new `iter1_filter_mode_sweep` reuses `build_scf_state` (`tests/fixtures/cu111_co.rs`).
- CASTEP `.bands` reference parsing: plain text; minimal helper in test (per `DIAGNOSTIC_SELFTEST.md` D2).

---

## Risks (cross-referenced from DIVERGENCE_SURFACE)

1. **All three modes fail SC-4-tight.** Bug is upstream — most likely
   candidates per `DIVERGENCE_SURFACE.md` §C: β projector normalization in
   `compute_beta_phi`, Q matrix G=0 normalization, S_sub assembly in RR.
   Stop, do not commit, escalate to upstream audit. The diagnostic table from
   Step 2 is itself the artifact that justifies escalation; the per-band Δ
   pattern (constant offset vs scaling vs specific bands) narrows the suspect
   set.
2. **Mode B passes but Mode C does not.** Step-4 S⁻¹ reconstruction
   destabilizes against an otherwise-correct filter. Pick B, file a new
   follow-up that audits why the line-607 reconstruction fails. Most likely
   cause: a normalization or ordering mismatch between `apply_s_inverse`
   output and the `band_scale_axpy` it feeds into.
3. **Discriminator threshold (0.05 Ha) too loose.** If all modes pass,
   retighten to 0.01 Ha and rerun. A non-tightenable gate that all modes
   clear signals the discriminator was too weak to falsify — re-think.
4. **S-Gram-Schmidt has its own bug.** Possible but orthogonal; deferred per
   §4a above. Add to `open-followups.md §11` for the next round if the chosen
   mode passes iter-1 but drifts at iter-2/3.
5. **The diagnostic harness itself is wrong.** D6 self-test (Mode A reproduces
   band-1 = −1.69 Ha) catches this. Do not read B/C results until D6 passes.

---

## Verification

End-to-end (run after Step 1 lands as a single non-committed working-copy diff):

1. **Sanity:** `cargo check --workspace 2>&1` — green.
2. **Sanity:** `cargo clippy --workspace -- -D warnings 2>&1` — green.
3. **Unchanged baseline:** `cargo test --release -p chemrust-scf -- --ignored s_inv_s_identity_test` — passes at 1e-10 (commit `f21127f` baseline).
4. **Discriminator:** `cargo test --release --features scf_diag -p chemrust-scf -- --ignored iter1_filter_mode_sweep --nocapture` — emits per-mode per-band table; one mode passes the 0.05 Ha gate.
5. **Post-decision regression:** after Step 3 commit, `cargo test --release --features scf_diag -p chemrust-scf -- --ignored fixed_point_matches_castep_energy` — iter-1 SC-4 green; iter-3 SC-7 green if applicable.
6. **Cross-check:** re-run `iter2_v_eff_range_within_one_ha_of_iter1` on the winner — should now pass instead of seeing iter-2 V_eff range = 29.2 Ha.

The discriminator step is the only mandatory pre-commit gate. Steps 5/6
confirm the choice was correct downstream of iter-1.

---

## Resolution capture (post-fix)

After Step 3 commits, write `RESOLUTION.md` in this directory containing:
- Winning mode (A/B/C) and the per-band Δ table that selected it
- Fix location (file:line) — diff summary of the surviving code path
- Reclassified claims from `INVESTIGATION.md` (which DERIVED claims became
  EXTERNAL after this resolution; which prior assertions are now stale)
- Append a one-line entry to `notes/failure-patterns.md` with the root-cause
  pattern slug.
