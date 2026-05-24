# Davidson Phase 1A Standing — Pre-C2 Assessment

**Date:** 2026-05-24
**Branch:** `feat/phase-global-woodbury`
**Commit at write:** `f8fc21d` (last Gate 3 commit), uncommitted C2 test in working tree
**Purpose:** Snapshot of what the accumulated Phase 0 evidence does and does
not establish at the moment C2 (`gate3_prime_prime_davidson_stops_cascade_through_scf3`)
is about to run.

This file is the companion to:
- `GATE3_RESULT.md` — original Group C (SUPERSEDED, n_locked = 0).
- `GATE3_TWEAKS_REPORT.md` — Gate 3' (C1) tweaks and synthetic-lock pass.
- `DECISIONS.md` — original Phase 0 grill decisions.
- Future `GATE3_RESULT.md` revision (post-C2) overrides this file.

---

## 1. Evidence ledger

| Test / probe | Result | Status | Source |
|---|---|---|---|
| T-prime D-injection (CASTEP D matrices into our V_eff path) | iter-2 band-0 drift = **197 mHa** | EXTERNAL | `notes/debug/debug-20260524-tprime-d-injection/RESOLUTION.md` |
| T3 V_eff injection (CASTEP V_eff into our SCF) | iter-2 band-0 drift = **9.8 mHa** | EXTERNAL | `notes/debug/debug-20260523-2314/SUMMARY.md` |
| Natural cascade (our V_eff, our D, Chebyshev-RR) | iter-2 band-0 drift = **184 mHa**; iter-3 diverges | EXTERNAL | `tests/ca_scf_convergence.rs::cascade_iter3_diagnostic` |
| Gate 3 original (Davidson, our V_eff, no perturbation) | n_locked = **0 / 160**, ratio 0.995 | SUPERSEDED | `GATE3_RESULT.md` |
| Gate 3' / C1 (Davidson, CASTEP-pinned V_eff, eps=0.01 perturbation, lock_tol = 0.5) | n_locked = **13 / 160**, bitwise-preserved Cu-3d | PASS (primitive) | `GATE3_TWEAKS_REPORT.md` |
| Gate 3'' / C2 (Davidson, our V_eff, SCF iter 1→2→3, lock_tol = 0.5) | **TBD** | IN-PROGRESS | uncommitted in `tests/ca_scf_convergence.rs` |

**Subspace-RR rotation cascade**, the underlying failure, is settled
EXTERNAL by the failure-patterns ledger: Chebyshev-RR rotates ψ inside
the Cu 3d degenerate manifold, ρ_aug splits 29.8/70.2 vs CASTEP F8
36.8/63.2, V_eff drifts, cascade by iter-3
(`notes/failure-patterns.md`, entry 2026-05-23).

## 2. What is and is not blocking Phase 1A

### 2.1. D-screening drift — NOT blocking

The Gate 3' tweaks report flagged a "0.07 Ha V_NL noise floor" and
framed it as a finding worth its own investigation. **That investigation
already happened.** T-prime D-injection ran the orthogonal substitution
test (replace our D with CASTEP D in our SCF) and the cascade continued
at 197 mHa. T3 ran the V_eff substitution test and the cascade stopped
at 9.8 mHa. The discriminator is sharp:

```
D wrong   + V_eff right → 9.8 mHa  (T3, cascade stopped)
D right   + V_eff wrong → 197 mHa  (T-prime, cascade alive)
D wrong   + V_eff wrong → 184 mHa  (natural, cascade alive)
```

The chemrust-hamiltonian Cu d-beta2 17.9 mHa D-drift is **real** but
**not load-bearing** for the cascade. Phase 1A may proceed without
fixing it. The Gate 3' "noise floor" is the same drift, re-observed
through a different probe; its only operational consequence is that
lock_tol cannot be set below it at iter-1.

### 2.2. lock_tol = 0.5 Ha as "calibration" — open

The Gate 3' report frames 0.5 Ha as a 7× safety margin above the
observed Cu-3d residual ceiling (~0.07 Ha). Two facts about this number:

- **It is observation-fit, not theory-derived.** The 0.5 Ha was chosen
  after seeing the bimodal residual distribution that the test's
  perturbation scheme (eps=0.01 on bands 0 and 14) *constructed*.
- **It happens to also pass the T-prime/T3 logic check.** Iter-1 ψ are
  CASTEP ψ. They are not exact eigenvectors of our H even with
  CASTEP-pinned V_eff (because our H_NL ≠ CASTEP H_NL by the
  D-screening drift). The 0.07 Ha residual is the *real* mismatch from
  D, not a numerical artifact. Setting lock_tol > 0.07 to lock Cu-3d
  bands at iter-1 is therefore physically motivated, not just
  test-engineered.

C2 uses the same lock_tol = 0.5 Ha for iter-1, iter-2, iter-3. This is
a Phase 0 simplification. **Phase 1A must replace this with a ratchet
schedule** — start near the D-floor (~1e-1 Ha), tighten as V_eff
converges and residuals drop. The Phase 0 number does not survive to
Phase 1A.

### 2.3. Tweak 3 GPU non-determinism — open, low priority

The 1e-7 → 1e-5 block-sum tolerance relaxation in C1 was attributed to
two `VnlBatchData::precompute` calls producing slightly different S
buffers (~6e-7 excess over predicted f64 noise). Untested alternative
hypotheses:
- ZGEMM accumulation-order non-determinism across kernel launches.
- C1's host-side reference Sψ path vs Davidson's GPU Sψ path having
  different precision profiles.

This deserves a debug-assert check ("same V_eff → bytewise-equal
D_screened on two consecutive precompute calls"), but only if the
discriminator gap shrinks below 1 mHa in some future test. At current
gate scales (Cu-3d block sum 12.999999 vs Chebyshev's 11.61), the 1e-5
tolerance has 4 orders of magnitude of headroom and does not threaten
any decision.

### 2.4. C1 vs C2 — what each does and does not prove

- **C1 (synthetic-lock identity preservation)** proves the locking
  primitive bitwise-preserves bands it locks. It does **not** touch
  ρ_aug, β·ψ, V_eff feedback, or cascade dynamics. C1 passing is
  necessary, not sufficient, for Phase 1A.
- **C2 (SCF iter 1→2→3 with Davidson + our V_eff)** is the cascade
  test. It is the *only* test in Phase 0 that exercises the ψ → β·ψ →
  ρ_aug → V_eff → next-iter ψ feedback loop with Davidson active. If
  C2 stops the cascade, the entire chain from rotation hypothesis to
  fix is empirically closed. If C2 does not, Phase 1A is wrong-shape
  regardless of what C1 says.

## 3. C2 pre-registered decision matrix

Recorded here **before** C2 runs, to prevent post-hoc threshold drift.
Anchored against the T-prime/T3 figures in section 1.

The probe is iter-3 band-0 in Ha, compared to CASTEP reference
−1.05502310 Ha.

| iter-3 |band-0 drift| | Verdict | Implication |
|---|---|---|
| ≤ 50 mHa | **PASS** | Davidson + locking arrests the cascade. Proceed to Phase 1A as defined in `PHASE_PLAN.md`. |
| ∈ (50, 100] mHa | **PASS-WITH-CAVEAT** | Locking helps significantly. Phase 1A's outer-iteration tightening should close the gap. Document the iter-1/2/3 n_locked and max_res trajectory. |
| ∈ (100, 197] mHa | **MIXED** | Locking helps but does not match T3 V_eff-injection floor. Phase 1A is viable but Phase 1B (block CG) cannot be ruled out. Run additional probes before committing. |
| > 197 mHa | **FAIL** | Locking is not the load-bearing mechanism. The cascade survives with Davidson too. Fall back to Phase 1B (block CG) per `notes/plans/phase-block-cg-migration/PHASE_PLAN.md`. |
| panic / NaN / OOM | **VOID** | C2 itself is broken. Diagnose; do not infer about Davidson. |

**Anchor reasoning:**

- The 50 mHa PASS threshold reflects: T3 is 9.8 mHa with CASTEP V_eff
  (best achievable with our H_NL). C2 uses our V_eff at iter-1 and our
  rebuilt V_eff at iter-2, iter-3. Each iter introduces some additional
  V_eff drift on top of T3's baseline. A factor of ~5× over the T3 floor
  is the realistic ceiling for "cascade arrested."
- The 197 mHa FAIL threshold is exact: T-prime's number with CASTEP D
  but our V_eff. If Davidson + locking + our D + our V_eff produces a
  drift ≥ T-prime, then locking has zero discriminating power over
  Chebyshev-RR for cascade purposes.
- The (50, 100] PASS-WITH-CAVEAT band exists because a single-sweep
  Phase 0 Davidson is *less powerful* than Phase 1A's outer iteration.
  A figure in this range means "Phase 1A's missing pieces (preconditioner,
  iteration, ratchet) are exactly what's needed to close the remainder";
  this is a reasonable design hypothesis, not a punt.
- The MIXED band (100, 197] is the danger zone — locking moved the
  needle but not enough to clearly beat T-prime. Specific follow-up
  probes to run before deciding:
  - Re-run C2 with lock_tol = 0.1 (closer to D-floor); does iter-2/3
    lock more bands and shrink drift?
  - Re-run C2 with eigenvalue ratchet (tighten lock_tol by 10× each
    iter); does the trajectory improve monotonically?
  - Measure iter-1 ρ_aug split (Cu vs interstitial); is it closer to
    CASTEP F8 36.8/63.2 with Davidson than with Chebyshev-RR's 29.8/70.2?

## 4. Surviving open questions for Phase 1A (regardless of C2 outcome)

These are independent of C2 and must be addressed when Phase 1A starts:

1. **lock_tol ratchet schedule.** Phase 0 uses one static value. Phase 1A
   needs a per-iter schedule. Open question: derive from residual decay
   theory (e.g., target locking rate 50% bands per iter after iter-3),
   or from observed iter-by-iter residual histograms.
2. **TPA preconditioner.** Skipped in Phase 0 (DECISIONS.md D2). Required
   for Phase 1A outer iteration to converge. CUDA kernel + Rust wrapper,
   ~50 LOC, modeled on Chebyshev filter's GPU pattern.
3. **Subspace management.** Phase 0 has no restart, no collapse, no
   trial-vector growth. Phase 1A needs all three.
4. **`EigensolverMethod` typed enum.** Replaces `CHEMRUST_EIGENSOLVER`
   env-var (DECISIONS.md D3). Touches every `diagonalize` call site.
5. **S^{-1}-weighted residual norm.** Phase 0 uses plain L2
   (DECISIONS.md A6). Phase 1A production version uses Woodbury-based
   weighted norm via `apply_s_inverse` (`chebyshev.rs:852-920`).
6. **GPU determinism audit** (low priority). Bytewise reproducibility
   of `VnlBatchData::precompute` and `apply_s_times` on identical inputs.

## 5. What this assessment does NOT claim

- It does **not** claim Phase 1A will succeed. C2 is unrun. The
  decision matrix in §3 is the falsifiable claim.
- It does **not** claim C1 was a strong test. C1 proves the locking
  primitive, not the cascade fix. The amended TASKS.md called this out
  explicitly (Group C1 marked "necessary, not sufficient").
- It does **not** claim lock_tol = 0.5 Ha is the right Phase 1A value.
  It is the Phase 0 simplification; Phase 1A must replace it.
- It does **not** retract the original Gate 3' critique of the
  superseded Group C (n_locked = 0). That critique stands and produced
  the amendment that gave us C1 and C2.
- It does **not** claim D-screening accuracy is unimportant for the
  project as a whole. The 17.9 mHa Cu d-beta2 drift is a real
  chemrust-hamiltonian quality issue tracked in their issue #9. It is
  not on Phase 0's critical path; it is not denied as a concern.

## 6. Next action

Run C2:

```bash
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 \
  cargo test --release --features scf_diag \
  gate3_prime_prime_davidson_stops_cascade_through_scf3 \
  -- --ignored --nocapture
```

When the test completes, read iter-3 band-0 from the printed output,
look up the verdict in the §3 table, and write the revised
`GATE3_RESULT.md` recording:
- The iter-1 / iter-2 / iter-3 n_locked, max_res, band-0 trajectory.
- The verdict bucket.
- The Phase 1 algorithm choice (Davidson v1 or block CG fallback).
- Citation of this file's §3 matrix as the pre-registration anchor.

This file is then superseded by the revised `GATE3_RESULT.md` and may
be retained as forensic context or deleted at the user's discretion.
