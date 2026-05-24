# Decisions — Phase 0 Davidson Gate 3

**Date:** 2026-05-24
**Source:** `/drive-outcomes` grill against
`notes/plans/phase-eigensolver-migration/PHASE_PLAN.md`.
**Scope:** Phase 0 only (decisive gate that picks Phase 1A vs Phase 1B).
Phase 0.5 / 1A / 1B / 2-4 explicitly deferred per user instruction
"Create the TASKS.md for the Davidson test first."

## Goal

Answer one yes/no question on Cu111+CO with the cheapest possible code:

> **Does per-band locking (skip rotation of converged bands inside ZHEGVD)
> preserve the Cu-3d block sum at 13.0, where Chebyshev-RR's structural
> rotation forces it down to 11.6?**

Outcome decides whether Phase 1A (Davidson v1) or Phase 1B (block CG)
becomes the production eigensolver migration target. Wrong answer
costs ~2-3 weeks of misdirected work; right answer eliminates one of
the two algorithm options for v1.

## Declared fixtures

The Phase 0 test consumes the same Cu111+CO fixture that all existing
diagnostic tests use, accessed through the established loader.

| Fixture | Path | Purpose |
|---------|------|---------|
| CASTEP wavefunctions | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check` | Source of CASTEP ψ at 160 bands; consumed via `tests/fixtures/cu111_co.rs::load_fixture` (lines 72-88) which calls `chemrust_hamiltonian_core::CheckFile::read`. S-orthonormal under USPP S, no re-orthogonalisation on read (per [[castep_check_continuation_convention]]). |
| CASTEP V_eff | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.pot_fmt` | Reference V_eff on fine grid. **Not directly used by Gate 3** (which builds our V_eff via `build_v_eff_with_energy`); exists for sibling diagnostics. |
| CASTEP density | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.castep_bin` | Cell geometry + density on wave grid; used by `build_scf_state` to construct the `ScfIteration<Initialized>`. |
| CASTEP eigenvalues | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands` | Band-0 reference value −1.05502287 Ha; not consumed by Gate 3 directly but anchors sibling cascade tests. |
| Pseudopotentials | `/export/Potentials/` | Cu_OTF.usp, C_OTF.usp, O_OTF.usp via `CASTEP_POTENTIAL_DIR` env override. |

**Path-name discrepancy noted**: PHASE_PLAN.md line 60 cites
`Cu111_CO_SinglePoint/Cu111_CO.check` but the loader at
`tests/fixtures/cu111_co.rs:18` hardcodes the
`Cu111_CO_Single_Point_0522_F8` path. **Loader is authoritative**;
PHASE_PLAN.md and CLAUDE.md path strings are stale documentation. All
existing diagnostic tests (`subspace_projector_iter1_vs_castep`,
`diagnostic_per_band_s_norm_of_our_output`,
`cascade_with_castep_anchored_postrr_pin`, etc.) report against the F8
fixture; using the same one keeps Gate 3's 13.0/11.6 ratio comparison
direct.

## Success criteria

### Primary (Gate 3 outcome)

| Cu-3d block sum (bands 1..14 vs CASTEP, S-weighted) | Phase 1 algorithm |
|---|---|
| ∈ [12.999, 13.001] | Phase 1A — Davidson v1 (per-band locking sufficient) |
| ≤ 11.700 | Phase 1B — block CG (locking insufficient; cascade upstream of ZHEGVD rotation) |
| ∈ (11.700, 12.999) | Lean Davidson, re-evaluate at Phase 2 (mixed signal) |

Source: PHASE_PLAN.md Gate 3 table (lines 64-68) plus Risks row 1
(line 254): "any ratio < 0.97 → fall back to CG; ≥ 0.97 → proceed
Davidson". The thresholds **must be pre-decided** before reading the
test output to prevent post-hoc rationalization.

### Secondary (correctness anchors)

| Criterion | Anchor |
|---|---|
| Self-consistency: pinned-V_eff Davidson reproduces input ψ to ≤ 1e-12 | Group B unit test: feed CASTEP ψ + CASTEP V_eff (`fx.pot_fmt`-derived); all bands lock; output ≡ input |
| Existing baseline `subspace_projector_iter1_vs_castep` reports Cu-3d ratio = 0.893 unchanged when `CHEMRUST_EIGENSOLVER` is unset | No-regression check: env-var dispatch must default to Chebyshev path verbatim |
| `cargo check --workspace` passes after Group A dispatch wire-up | Compilation gate |
| `cargo clippy --workspace -- -D warnings` passes after Group B | Lint gate matching CONTEXT.md tooling |

### Decision artifact

A human-readable `notes/plans/phase-eigensolver-migration/GATE3_RESULT.md`
captures the block sum value, residual statistics, and chosen Phase 1
direction. This file (not the test log) is the canonical record that
unblocks the next phase.

## Architectural decisions

### A1. Single-sweep Davidson, no outer iteration

**Decision (D1):** Phase 0 implements a *single* sweep — compute
residuals, mark locked, ZHEGVD on unconverged sub-block, S-orthogonalize,
stop. No outer Davidson loop.

**Why:** Gate 3 starts from CASTEP ψ. Residuals are tiny (CASTEP ψ are
near-eigenvectors of *our* H, drift only by V_eff difference). The
question being answered — "does isolating ZHEGVD to the unconverged
sub-block prevent locked-band rotation?" — is fully exercised in one
sweep. Iterating would derisk Davidson convergence rate, which is
Phase 1A's job, not Phase 0's.

**Code budget:** ~120 LOC for the algorithm; total Phase 0 ~230 LOC.

### A2. No preconditioner

**Decision (D2):** Skip the Teter-Payne-Allan diagonal preconditioner
P^{-1}_b[g] = (kinetic[g] − λ_b)^{-1}.

**Why:** Preconditioner matters when residuals are large and need
acceleration; for CASTEP-near eigenvectors residuals are tiny in any
norm. Phase 0 isn't measuring convergence rate, it's measuring whether
locking works as a structural property of the algorithm. Saves ~50 LOC
CUDA kernel + Rust wrapper that Phase 1A will write properly anyway.

### A3. Env-var dispatch, not typed enum

**Decision (D3):** Add `CHEMRUST_EIGENSOLVER=davidson|chebyshev`
(default `chebyshev`) read once in
`src/scf.rs::diagonalize_inner`. Mirrors the established
`CHEMRUST_PIN_MODE` pattern at `src/eigensolver/rayleigh_ritz.rs:36, 64-81`.

**Why:** PHASE_PLAN.md explicitly schedules `EigensolverMethod` enum
introduction for Phase 1A. Adding it now would (a) require touching
every `diagonalize` call site, (b) commit to a public-API surface
before the algorithm is validated, (c) need to be removed/refactored
when Phase 1A actually lands. Env-var dispatch removes cleanly:
Phase 1A deletes the env-var read and replaces with the typed enum.

### A4. Use our V_eff, not CASTEP-pinned V_eff

**Decision:** Gate 3 calls `state.build_v_eff()?` (our V_eff path) rather
than pinning V_eff from `fx.pot_fmt`.

**Why (load-bearing):** If V_eff were CASTEP-pinned, our H ≡ CASTEP's H
exactly → CASTEP ψ are exact eigenvectors → residuals = 0 → all bands
lock → ZHEGVD sub-block never runs → output ≡ input → block sum
trivially 13.0. Test learns nothing.

With our V_eff, our H differs slightly from CASTEP's (~1 mHa eigenvalue
drift). Some bands develop residuals > 1e-6 → ZHEGVD runs on the
unconverged sub-block → if the locked Cu-3d bands survive untouched,
block sum stays near 13.0; if Chebyshev-RR's failure mode reappears
(rotation through ZHEGVD), block sum drops toward 11.6.

This matches the existing baseline:
`subspace_projector_iter1_vs_castep` uses our V_eff and reports the
11.6 floor. Direct ratio comparison demands identical V_eff source.

### A5. ndeg ignored in davidson branch

**Decision:** `diagonalize(ndeg, occupations)` keeps its existing
signature. In davidson mode, `ndeg` is documented-ignored — the new
path bypasses `chebyshev_filter` entirely.

**Why:** Avoids changing the public typestate transition signature
just for Phase 0. Phase 1A's typed enum will replace the ndeg parameter
properly; Phase 0 does not need to.

### A6. Plain L2 residual norm, not S^{-1}-weighted

**Decision:** Group B computes ‖r_b‖₂ via `cublasDznrm2`, not the proper
S^{-1}-weighted norm Phase 1A will use.

**Why:** For CASTEP-near ψ, r_b is small in any norm. Discriminator
between locked/unconverged is unaffected; off-diagonal weight is
second-order. Phase 1A's production version uses the correct
S^{-1}-weighted norm via Woodbury (apply_s_inverse already exists at
`chebyshev.rs:852-920`).

### A7. No hard assertion in Gate 3 test

**Decision:** Group C prints the block sum and a derived `Decision:`
text line; does NOT `assert!` against a threshold.

**Why:** Phase 0 is a decision gate, not a correctness gate. The
threshold IS the decision. Hard-asserting `sum > 12.999` would
crystallize Phase 1A before the user reads the number, defeating the
gate's purpose. Matches the existing convention at
`subspace_projector_iter1_vs_castep` (lines 3475-3573 — diagnostic
prints, no assertion). Group D captures the human decision in
`GATE3_RESULT.md`.

## Domain terms validated against CONTEXT.md

| Term | CONTEXT.md status | Phase 0 usage |
|---|---|---|
| WavefunctionSet | Defined (line 30) | Davidson input/output type |
| ColumnDistributed / RowDistributed | Defined (lines 54-58) | Davidson lives in ColumnDistributed; sub-block ZHEGVD on host |
| Rayleigh-Ritz | Defined (lines 47-50) | Reference for H_sub/S_sub assembly with USPP augmentation |
| Chebyshev Filtering | Defined (lines 41-46) | NOT called in davidson branch |
| EffectivePotential | Defined (line 73) | Built by `build_v_eff()`, consumed by `apply_full_hamiltonian` |

**New terms (Phase 0 only, not for CONTEXT.md commitment):**

- **per-band locking** — a band b is *locked* when ‖r_b‖ < `lock_tol`
  (1e-6); locked bands are excluded from sub-block diagonalization,
  preventing rotation. Source: PHASE_PLAN.md line 27. Promote to
  CONTEXT.md only if Phase 1A ships with Davidson.
- **unconverged sub-block** — the (k×k) submatrix of H_sub and S_sub
  built from the columns of ψ corresponding to unlocked bands; ZHEGVD
  acts only on this submatrix. Phase 0 internal vocabulary; not yet
  CONTEXT.md material.

## Pre-existing infrastructure relied upon (not modified)

These are load-bearing reused primitives. Source-audited in this
session:

| Primitive | Path | Lines | Verified behavior |
|---|---|---|---|
| `apply_full_hamiltonian` | `src/eigensolver/chebyshev.rs` | 707-739 | T+V_loc via FFT (call to `apply_v_loc_hamiltonian`) followed by V_NL via cuBLAS gemm (call to `apply_v_nl_hamiltonian`). Returns Hψ in `hpsi_dev` |
| `apply_s_times` | `src/eigensolver/chebyshev.rs` | 934-1014 | **Caller must pre-copy ψ into spsi_dev** for the identity term (per comment at line 936). Then three gemm calls: p = β^H·ψ, q = Q·p, spsi += β·q with β=+1 accumulator |
| `apply_s_inverse` | `src/eigensolver/chebyshev.rs` | 852-920 | Woodbury S^{-1} = I − B·M^{-1}·B^H. Pre-factored LU in vnl_data. Not used by Phase 0 (plain L2 norm — see A6) |
| Rayleigh-Ritz H_sub/S_sub assembly | `src/eigensolver/rayleigh_ritz.rs` | 181-310 | H_sub = ψ^†·Hψ, S_sub = ψ^†·ψ + Σ_ion C_proj^†·Q·C_proj where C_proj = β^H·ψ. Phase 0 adapts this restricted to the unconverged index set |
| Gram-Schmidt (2-pass S-orth) | `src/eigensolver/chebyshev.rs` | 1708-1799 | Inlined in chebyshev_filter; per-band loop with `cublasZdotc` and `cublasZaxpy`. Phase 0 takes single-pass cross-orthogonalization (locked vs unconverged) only |
| CASTEP fixture loader | `tests/fixtures/cu111_co.rs` | 72-88 (check), 90-93 (pot_fmt), 158 (build_scf_state) | Caches via `OnceLock`; loads `.check`, `.pot_fmt`, `.castep_bin`, `.den_fmt`, `.bands` into `Cu111CoFixture`; `build_scf_state` returns `ScfIteration<S, Initialized, MixingOff>` |
| Cu-3d block sum logic | `tests/ca_scf_convergence.rs` | 3522-3535 (in `subspace_projector_iter1_vs_castep`) | Pattern: bands 1..14 (0-indexed, 13 bands), S-augmented overlap with CASTEP ψ |

## Risks acknowledged

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Block sum lands in mixed bucket (0.892 < ratio < 0.999) | MEDIUM | Pre-decided in PHASE_PLAN.md Risks row 1: any ratio < 0.97 → CG. Decision IS the threshold |
| Davidson branch breaks the no-env-var Chebyshev default | LOW | Group A acceptance: existing tests pass unchanged with no env var set |
| `apply_s_times` identity-term pre-copy missed | LOW (caught by Group B unit test) | Group B's pinned-V_eff trivial check catches missing `Sψ_b = ψ_b + (Q-aug)·ψ_b` term; pre-copy explicit in algorithm step 3 |
| `cublasZgemm` mu lda/ldb confusion in sub-block (k×k) gather | MEDIUM | Adapt verbatim from `rayleigh_ritz.rs:181-236` lda=n_pw, ldb=n_pw, ldc=n_bands pattern; restrict by index. Group B unit test catches |
| ZHEGVD on degenerate sub-block returns NaN if k=0 (all bands locked) | LOW | Group B early-returns input ψ unmodified when k=0 (degenerate case = self-consistency check passing trivially) |
