# INVESTIGATION: Prior-Note Classification

**Symptom:** `fixed_point_matches_castep_energy` diverges at iter-1 after the
global-Woodbury commits (HEAD `e1d1424` on `feat/phase-global-woodbury`).
Iter-1 RR band-1 = −1.69 Ha vs CASTEP reference −1.06 Ha (|Δ| = 0.63 Ha → 13×
the 0.05 Ha SC-4 gate). R-ChFSI norm ratio ~5×/step, k=2..8.

**Log:** `/tmp/scf-diag-global-woodbury-0523-0524.log`

**Reference reading scope:**
- `notes/open-followups.md §10` (post-Woodbury status)
- `notes/plans/phase-global-woodbury/PHASE_PLAN.md` (Goals 0/1a/1b/1c/3)
- `notes/plans/phase-rchfsi-bare-h/TASKS.md` (decision gates SC-1..SC-7)
- `notes/plans/phase-rchfsi/` (R-ChFSI Algorithm 3 implementation phase)
- Memory entries cited inline below

---

## Classification table

| # | Claim | Source | Class | Used as criterion? |
|---|-------|--------|-------|-------|
| 1 | Iter-1 RR band-1 in current run = −1.69 Ha | `/tmp/scf-diag-global-woodbury-0523-0524.log:65` | DERIVED (our pipeline output) | No — observed value, the thing we are explaining |
| 2 | CASTEP `.bands` band-1 reference = −1.055 Ha (≈ −1.06 Ha) | `Cu111_CO.bands` (text file at fixture path) | EXTERNAL | **Yes — anchor for SC-4** |
| 3 | Iter-2 V_eff range should be ≈ 8.69 Ha | `Cu111_CO.den_fmt` + VEffBuilder, recorded in `notes/open-followups.md §8` | EXTERNAL (fixture density round-trip) | **Yes — regression gate (`iter2_v_eff_range_within_one_ha_of_iter1`)** |
| 4 | CASTEP F8 instrumentation: soft ρ = 1,527,256 / aug ρ = 2,623,313 (N_e × Ω units) | `notes/open-followups.md:307-310` (CASTEP F8 dump from converged WF, recorded in same-input controlled experiment commit `fa68980`) | EXTERNAL (CASTEP source-code instrumented dump) | Yes — used in `density_decomp_matches_castep_f8_same_inputs` |
| 5 | Same-input ratios soft 1.000000 / aug 1.000084 (commit `fa68980`) | `tests/ca_scf_convergence.rs::density_decomp_matches_castep_f8_same_inputs` | DERIVED (our pipeline, but anchored against EXTERNAL #4 with tight gate) | Yes — proves density code is correct *for matching ψ* |
| 6 | S⁻¹ identity residual ‖S⁻¹·S·ψ−ψ‖_∞ = 3.8e-15 | `s_inv_s_identity_test`, commit `f21127f` (gate at 1e-10) | DERIVED (our test) | Yes (regression check); proven robust by tight gate |
| 7 | Pre-Woodbury identity residual = 0.014 (1.4%) | `notes/open-followups.md:504` ("post-fix diagnostic 2026-05-22") | DERIVED (was contaminated by `m_inv→s_inv` typo per PHASE_PLAN.md:32-38) | No — superseded by #6 |
| 8 | "Bare-H R-ChFSI cleared filter/Lanczos paths" — implied to mean "produces correct results" | `notes/open-followups.md §12` and bare-H phase notes | DERIVED + HYPOTHESIZED (run-1010 was on ζ ≈ 0.014; conclusion was conditional on that regime) | **No — DOES NOT classify as evidence in current regime** |
| 9 | Bare-H R-ChFSI converged to band-1 within 0.05 Ha at run-1010 | `notes/plans/phase-rchfsi-bare-h/TASKS.md:21` (SC-4 declared as the gate) and run-1010 log | DERIVED (single-run observation, no per-band table preserved in tree, no replication) | No — would require re-run with current ζ to be EXTERNAL-equivalent |
| 10 | Levitt-Torrent Algorithm 1: filter on `S⁻¹·H` not `H` | `reference_paper/extracted/levitt-torrent-2015/abinit.tex:653-657, 678-680` | EXTERNAL (peer-reviewed paper, JCP) | Yes — sets the Mode B/C target |
| 11 | Das 2025 Algorithm 3: D⁻¹ in both Step 3 (`main.tex:603`) and Step 4 (`main.tex:607`) | `reference_paper/extracted/das-2025-rchfsi/main.tex:586-610` | EXTERNAL (preprint with formal proofs) | Yes — sets Mode C target; line 612 (ChFSI ≡ R-ChFSI when D⁻¹=B⁻¹) sanctions Mode C as "standard ChFSI" under our ζ |
| 12 | Das Theorem 3.2: standard ChFSI stagnates at O(ζ) | `main.tex:861` (proof) + `notes/plans/phase-global-woodbury/PHASE_PLAN.md:223-230` | EXTERNAL (theorem) | Reasoning input for "bare-H rationale is empirically stale" |
| 13 | Lanczos `b_up_lanczos = 19.28 Ha`, scaled = 21.20 Ha (`/tmp/...0524.log:53`) | Current session log line 53 | DERIVED (this session's diagnostic) | Reasoning input for divergence-surface enumeration |
| 14 | Bare-H Gershgorin upper bound = 116.49 Ha (same line 53) | Same log | DERIVED | Reasoning input — establishes that H spectrum extends well above filter window |
| 15 | Iter-1 R-ChFSI norm ratios: k=2 → 3.98, k=3-8 → ~4.9 | `/tmp/...0524.log:57-63` | DERIVED | Reasoning input — geometric growth pattern |
| 16 | "Bare-H R-ChFSI hypothesis insufficient when SC-4 fails" | `notes/plans/phase-rchfsi-bare-h/TASKS.md:32` | EXTERNAL within project (decision-gate pre-locked by author before knowing outcome) | Yes — its triggering condition (SC-4 fail) is met in current log; instructs us to investigate compounding bug |
| 17 | "S⁻¹·H bounds applied to bare-H polynomial → mismatch" | This investigation's reasoning (chebyshev.rs:472-476 wires S⁻¹·H into Lanczos but `use_sinv_filter=false` at scf.rs:524 keeps recurrence on bare H) | HYPOTHESIZED (root-cause hypothesis to be tested by Mode B/C of A/B/C sweep) | No — to-be-tested |
| 18 | Memory: "CASTEP .check wavefunctions are S-orthonormal under USPP S, no re-orthogonalisation on read" | `~/.claude/projects/-home-tony-programming-chemrust-scf/memory/castep_check_continuation_convention.md` | DERIVED (memory entry) — but lacks the three required fields (verification-granularity, file:line citation, counter-example/scope) | Treat as hypothesis; verifiable by reading CASTEP `wave_read.F90` if it becomes load-bearing — currently not load-bearing for this fix |

---

## Memory-citation rule application

The most consequential prior memory at risk of contaminating this debug session:

> "When an isolated test proves a fix correct but the full pipeline gets worse,
> hunt for a second compounding bug instead of reverting"
> (`~/.claude/projects/-home-tony-programming-chemrust-scf/memory/dont_revert_empirically_correct_fix_on_regression.md`)

Applies *directly* here. The bare-H R-ChFSI fix was empirically correct under
ζ ≈ 0.014. The full pipeline has now gotten worse — but only after a separate
upstream change (Goal 1a global-Woodbury) collapsed ζ to 3.8e-15. The memory
forbids reverting bare-H on regression; it does *not* forbid re-running an
A/B comparison once the regime that justified bare-H has changed. The
distinction is:

- **Forbidden**: "bare-H produces band-1 = −1.69 Ha → revert bare-H to S⁻¹·H"
- **Permitted**: "ζ regime that motivated bare-H is gone; re-run the A/B
  comparison and let the data pick which mode wins under the new ζ"

The plan does the latter. The diagnostic-first A/B/C sweep is precisely the
empirical re-validation the memory's intent requires.

---

## Reclassified-as-stale claims (consequence of regime change)

Claim #8 ("bare-H R-ChFSI cleared filter/Lanczos paths") and #9 ("run-1010
band-1 success") are reclassified from "evidence" to "evidence under
ζ ≈ 0.014, not under ζ = 3.8e-15." They no longer support keeping bare H in
the recurrence; they document the historical justification.

---

## Anchor inventory (for CRITERIA.md)

EXTERNAL claims admissible as success criteria:
- #2: CASTEP `.bands` band-1 = −1.055 Ha (and the next 9 bands)
- #3: iter-2 V_eff range ≈ 8.69 Ha (existing regression gate)
- #4: CASTEP F8 ρ decomposition (existing same-input gate)
- #6: S⁻¹ identity ≤ 1e-10 (existing baseline gate; cannot regress)
- #10: Levitt-Torrent Alg 1 prescribes S⁻¹·H (algorithm-level)
- #11: Das Alg 3 prescribes D⁻¹ in both Step 3 and Step 4 (algorithm-level)
- #16: Bare-H phase decision gate fires when SC-4 fails

DERIVED/HYPOTHESIZED claims **not** admissible:
- #1, #7, #8, #9, #13, #14, #15, #17, #18

This separation is the contract between INVESTIGATION.md and CRITERIA.md.
