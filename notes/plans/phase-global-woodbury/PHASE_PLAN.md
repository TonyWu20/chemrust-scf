# Phase: Global Woodbury S⁻¹

**Date:** 2026-05-22 (revised post-review)
**Status:** Draft (revised)
**Branch base:** `feat/phase-rchfsi` (after bare-H R-ChFSI gates green at HEAD = `6bd2c58`)
**Plan source:** `notes/open-followups.md §12`
**Revision driver:** `notes/plans/phase-global-woodbury/review.md` (three critical findings + design concerns absorbed below)

## Context

The bare-H R-ChFSI experiment (`notes/plans/phase-rchfsi-bare-h/`) cleared the
filter and Lanczos paths of S⁻¹ but did not stop SCF divergence at iter-3. The
ndeg sweep (0/4/8/16/32) at `notes/open-followups.md:715-740` localised the
defect away from the filter quality and onto the per-ion Woodbury inverse used
for `S⁻¹·v`. The discriminator probe `s_inv_s_identity_test` measures
‖S⁻¹·S·ψ₀ − ψ₀‖_∞ ≈ **0.014** — but that measurement is **contaminated** (see
revision finding 2 below); it must be remeasured before being trusted.

Three points the original plan understated or missed, all verified against the
source on `feat/phase-rchfsi @ 6bd2c58`:

1. **`apply_s_inverse` is currently dead code.** It carries
   `#[allow(dead_code)]` at `src/eigensolver/chebyshev.rs:820` and is called
   from exactly one site — the diagnostic at line 1068, with `n_bands=1`. It
   is *not* called from `lanczos_upper_bound` (line 461 calls
   `apply_full_hamiltonian` — bare H), the R-ChFSI recurrence (line 1402),
   `chebyshev_filter`, or the Step-4 reconstruction (line 1620 comment:
   *"no S⁻¹ — see Risk §2 in plan"*). `notes/open-followups.md §12:805-806`
   already states *"the bare-H R-ChFSI phase removed the call sites; restoring
   them is part of the fix."* Replacing the body of a dead function is not
   sufficient — the wiring is part of this phase.
2. **`m_inv` ↔ `s_inv` typo at `vnl_data.rs:320`.** After Gauss-Jordan,
   `m_inv` is reduced to identity and `s_inv` holds M⁻¹. Line 320 uploads
   `m_inv`, so the per-ion `s_inv_mat` on GPU is the identity matrix and
   `apply_s_inverse` reduces to `hpsi −= Σ β·(β^H·v)` with no Q⁻¹ term. The
   0.014 baseline ζ is therefore not the per-ion-Woodbury baseline; it is the
   identity-Woodbury baseline. A clean per-ion baseline must be measured
   first (Goal 0).
3. **Lanczos uses bare H, not S⁻¹·H.** For the generalized eigenproblem
   H·x = λ·S·x, the Chebyshev filter bounds (`b_up`, `b_low`) should bracket
   λ(S⁻¹·H), not λ(H). The current Lanczos at `chebyshev.rs:461` calls
   `apply_full_hamiltonian` (bare H), and `notes/open-followups.md §12:824-825`
   confirms Lanczos needs S⁻¹ for the spectrum to land where the algorithm
   assumes.

Replacing the per-ion `S̃⁻¹` with the exact global Sherman–Morrison–Woodbury
inverse

```
S = I + B·Q·B^H
S⁻¹ = I − B · (Q⁻¹ + B^H·B)⁻¹ · B^H,   B = [β_1 | β_2 | … | β_N]  (~350 cols)
```

drives ζ to roundoff. With ζ → 0, R-ChFSI is algebraically identical to
standard ChFSI (R-ChFSI paper, `main.tex:612`), and the existing convergence
theory covers the USPP case directly.

## Goals

### Goal 0 — Prelim: `m_inv` → `s_inv` typo + baseline rebaseline

One-character source change at `src/eigensolver/vnl_data.rs:320`
(`m_inv.iter()` → `s_inv.iter()`). Lands as its own commit before any other
change in this phase. Re-run `s_inv_s_identity_test` (asserts `< 1e-6`); record
the new ‖S⁻¹·S·ψ−ψ‖_∞ as the true pre-fix baseline. This number replaces the
contaminated 0.014 in the discriminator stack. New unit test
`s_inv_baseline_post_typo_fix` locks the value for the rest of the phase.

If the rebaselined ζ is already < 1e-10, the global Woodbury work reframes
from "correctness fix" to "perf optimization + cross-ion completeness for
non-Cu systems" — Goal 4's iter-3 explosion gate then demotes to a
regression check rather than the load-bearing falsifier.

*Effort:* trivial. One-character source edit, one CI run, one commit, one
test.

### Goal 1a — Global Woodbury inverse for `apply_s_inverse`

Replace the per-ion loop in `chebyshev.rs:821-900` with a single-shot global
solve:
- precomputed concatenated `B = [β_1 | β_2 | … | β_N]` (n_pw × ~350)
- one `cublasZgemm` for `B^H · v`
- one `Zpotrs` against the cached Cholesky factor of `M = Q⁻¹ + B^H·B`
- one `cublasZgemm` for `B · temp` accumulated with α = −1

The Cholesky factor is built **once per run** (not once per SCF iteration)
inside `VnlBatchData::precompute` (`src/eigensolver/vnl_data.rs:97`) from
concatenated per-ion `q_matrix` and `beta_g`. `B^H·B` is computed by a single
`cublasZgemm` on the concatenated B (GPU), not per-ion CPU triple-loops.
Drop `#[allow(dead_code)]` from `apply_s_inverse` once Goals 1b/1c are
wired.

*Effort:* medium. Touches one struct, one apply path, adds three cuSOLVER
bindings, replaces two CPU Gauss-Jordan inversions with one GPU Cholesky.

### Goal 1b — Wire `apply_s_inverse` into `lanczos_upper_bound` (mandatory)

Insert a single `apply_s_inverse(&mut hv, …)` call immediately after
`apply_full_hamiltonian` at `chebyshev.rs:461` so the Lanczos operator
becomes S⁻¹·H (per §12:824-825). Reuses Goal 1a infrastructure; affects
only the n_bands=1 path inside the Lanczos loop.

New gate: bare-H vs S⁻¹·H Lanczos `b_up` delta on iter-1 logged. Expected
sign — `b_up` shifts toward the spectrum of S⁻¹·H (smaller upper bound for
Cu(111)+CO since S − I is positive-semidefinite for USPP).

*Effort:* small. One additional call inside an existing loop.

### Goal 1c — Filter recurrence S⁻¹ wiring as a gated A/B experiment

Add `apply_s_inverse` wiring inside the R-ChFSI recurrence behind a runtime
boolean threaded from the test entry point (no Cargo feature needed since
both paths share the compile path). Run paired iter-3 SCF: bare-H filter
(current default) vs S⁻¹·H filter (gated), all other variables fixed.

Decision criterion (locked at plan time, not at experiment time):
- **Keep bare-H** if iter-3 D_screened amax (S⁻¹·H filter) − bare-H baseline
  < 5 Ha **AND** iter-1 band-1 |Δ| does not regress > 0.005 Ha. Close the
  experiment, file a follow-up if the delta is independently interesting.
- **Flip default to S⁻¹·H** otherwise. Retire the flag.

*Effort:* small (wiring) + one paired SCF run.

### Goal 2 — cuSOLVER LU bindings in `src/device/solver.rs`

Add safe wrappers for `cusolverDnZgetrf_bufferSize`, `cusolverDnZgetrf`,
`cusolverDnZgetrs`. Mirror the existing `Zhegvd` template at
`src/device/solver.rs:47-95` (handle, buffer-size query, workspace alloc,
solve, error type). Cholesky (`zpotrf`/`zpotrs`) was the original choice but
fails on near-singular M (zero Q⁻¹ rows). Two added tests:
- 4×4 full-rank round-trip (`Zgetrf` → `Zgetrs` against synthetic RHS) asserting
  err < 1e-12.
- n_bands=8 multi-RHS `Zgetrs` against a 16×16 matrix — guards against
  single-RHS overfitting.

*Effort:* small. Self-contained binding layer.

### Goal 3 — Tighten `s_inv_s_identity_test` to roundoff

Change the assertion at `tests/ca_scf_convergence.rs:698-703` from `< 1e-6`
(lenient) to `< 1e-10` (roundoff for double precision on a 350×350 system).
Lead discriminator. Lands in the same commit as Goal 1a.

*Effort:* trivial. One line.

### Goal 4 — Iter-3 SCF convergence gate

Run `fixed_point_matches_castep_energy` to depth 3. Assert iter-3 Cu
D_screened amax < 10 Ha **AND** |band-1 − (−1.055 Ha)| < 0.05 Ha. The §12
falsification gate.

If Goal 0 rebaselines ζ below 1e-10, demote this gate to a regression check
rather than the correctness gate (the SCF feedback explosion will already
be gone before any of Goals 1a–1c land).

*Effort:* small. Extend an existing test, no new fixtures.

### Goal 5 — Diagnostic prints behind `cfg(feature = "scf_diag")`

Move 17 diagnostic eprintln sites (corrected from "ten" in original plan)
behind a feature gate:
- `src/eigensolver/chebyshev.rs` — 11 sites (`[Lanczos]`, `[Lanczos@call]`,
  `[Chebyshev]`, `[R-ChFSI]`)
- `src/scf.rs` — 6 sites (`[V_eff]`, `[RR]`, `[NewDensity]`, `[QSfCache]`,
  `[AugDensity]`)

Add a `[features]` section to the crate-root `Cargo.toml` (currently absent)
with `scf_diag = []`. Default builds: clean stderr; debug sessions re-enable
with `cargo test --release --features scf_diag`.

*Effort:* small. No semantic changes; release build does not bring back the
diagnostic noise.

### Goal 6 — Wall-time observability gate

Capture iter-2 `apply_s_inverse` total wall in the discriminator log; assert
≤ per-ion baseline. One `Instant::now()` pair plus an `eprintln!` gated by
`scf_diag`. Surfaces accidental regressions (e.g. uploading `B^H·B` every
iteration when it should be cached).

*Effort:* trivial.

## Scope Boundaries

**In scope:**
- `src/eigensolver/vnl_data.rs:320` — Goal 0 typo fix.
- `src/eigensolver/vnl_data.rs:28` (struct), `:97` (precompute), `:213-325`
  (replace Gauss-Jordan + per-ion s_inv path) — concatenated B, cached
  `B^H·B`, Cholesky factor of M; drop `s_inv_mat`.
- `src/eigensolver/chebyshev.rs:820-900` — replace `apply_s_inverse` body
  with global Woodbury; remove `#[allow(dead_code)]`.
- `src/eigensolver/chebyshev.rs:461` — insert `apply_s_inverse` after
  `apply_full_hamiltonian` for Lanczos (Goal 1b).
- `src/eigensolver/chebyshev.rs` filter recurrence body — gated
  `apply_s_inverse` insertion controlled by runtime/test flag (Goal 1c).
- `src/device/solver.rs` — `Zpotrf` / `Zpotrs` wrappers + tests.
- `tests/ca_scf_convergence.rs` — tighten to 1e-10; iter-3 D_screened +
  band-1 asserts; new `s_inv_baseline_post_typo_fix` test.
- `Cargo.toml` (crate root) — add `[features]` section with
  `scf_diag = []`.
- 17 diagnostic eprintln sites gated behind `#[cfg(feature = "scf_diag")]`.

**Out of scope:**
- **R-ChFSI → ChFSI simplification.** Das line-612 corollary lets us
  collapse the residual recurrence once ζ is roundoff, but coupling it to
  this phase makes blame attribution hard if a gate fails. File a
  follow-up.
- **Full SCF convergence to CASTEP total energy** (`-24110.96665069 eV`).
  Convergence past iter-3 depends on density mixing, occupancy update,
  and potential/charge mixing — orthogonal to S⁻¹ correctness.
- **Tier-1 augmentation Q·SF GPU caching (§9, ~100-200 s/iter).**
  Self-contained perf phase; no S⁻¹ dependency.
- **GPU D-matrix screening (D-6, ~3 s/iter).** Independent of S⁻¹
  correctness; defer.
- **D-1/D-2 cosmetic cleanups.** Bundle into a future cleanup pass.
- **D-5 H-eigenvalue vs generalized-eigenvalue quantification.** Research
  question, not a phase-blocker.

## Design Notes

### Why exact (not "smaller ζ") matters

Das 2025 Theorem 3.4 proves R-ChFSI converges to the correct *residual norm*
under ζ ∈ [10⁻⁴, 10⁻²]. But the eigenvectors of `S̃⁻¹·H` differ from those
of `S⁻¹·H` by O(ζ), and the SCF feedback loop (§12, eqn cluster after the
gate table) amplifies that into ρ_nm corruption → V_eff range explosion →
D_screened runaway. We are not solving the eigenvalue problem the paper
analyzes; we are solving a *different* eigenvalue problem on `S̃⁻¹·H` and
calling it H/S. The fix has to drive ζ to roundoff. (This argument
**only matters if Goal 0 rebaseline still shows ζ > 1e-10.**)

### Block structure

`Q⁻¹` is block-diagonal in ions (each ion's Q is independent). `B^H·B`
introduces the cross-ion blocks — that is the entire mathematical content
of the fix. Concretely:

```
Q⁻¹ = blkdiag(Q_1⁻¹, Q_2⁻¹, …, Q_N⁻¹)        # already known per-ion
B^H·B has N×N blocks, block (I,J) = β_I^H·β_J  # currently dropped for I≠J
M = Q⁻¹ + B^H·B is dense Hermitian PD ~350×350 # one Cholesky per RUN (not per iter)
```

Cholesky of a 350×350 Hermitian PD matrix is microseconds on Pascal+. The
expensive part is the one-time `B^H·B` GPU gemm at SCF setup.

### `B^H·B` computation strategy

Single `cublasZgemm` on the concatenated B (shape (n_pw × 350)ᴴ × (n_pw × 350)).
GPU; microseconds; aligned with the rest of the apply path. Does not
extend the existing CPU triple-loop pattern at `vnl_data.rs:220-230`
(would be ~6.5B ops for 18 ions and seconds per call).

### Caching policy

`M = Q⁻¹ + B^H·B` is built **once per run** inside `VnlBatchData::precompute`
(`vnl_data.rs:97`). Q is pseudopotential-static; β depends only on ionic
positions and the G-grid. Neither changes during SCF. Cholesky recomputes
only on geometry change (single-point run = never).

### Memory budget

| Item | Size |
|------|-----:|
| Concatenated B (350 × n_pw=50000 × 16 B/cplx) | **280 MB** |
| Cholesky factor of M (350×350 cplx) | **1.96 MB** |
| `B^H·v` temp (350 × n_bands=160 cplx) | **0.9 MB** |
| `B^H·B` precompute scratch (350×350 cplx) | **1.96 MB** |
| **Total incremental GPU footprint** | **≈ 285 MB** |

Within Pascal 8–16 GB budget; well below the SCF state already resident
(wavefunctions, V_eff FFT scratch, Q·SF caches).

### Cost comparison (per `S⁻¹·v` apply, n_pw ≈ 50k, n_bands = 160)

- **Per-ion (current):** 18 ions × 3 gemms × per-ion-shape ≈ 18 × O(n_pw·ne²),
  with launch overhead dominating.
- **Global Woodbury (this phase):** 1 gemm `B^H·v` (n_pw × 350 × n_bands), one
  `Zpotrs` (350 × 350 × n_bands triangular solve), 1 gemm `B·temp`. Strictly
  fewer launches, larger arithmetic intensity. Expected: faster, not slower.
  Goal 6 verifies.

### Risks

- **Goal 0 rebaseline already collapses ζ.** If post-typo-fix ζ < 1e-10,
  global Woodbury becomes a perf/completeness step rather than a
  correctness fix. Action: still ship Goals 1a/1b/1c (Lanczos
  correctness, perf, A/B data); demote Goal 4 explosion gate to a
  regression check, document the reframe inline.
- **Cholesky failure on near-singular M.** `Zpotrf` returns `info > 0` on
  the failing leading minor (observed: info=2 on Cu(111)+CO due to zero Q⁻¹
  rows making the 2×2 leading minor non-PD). **Resolution:** switch to LU
  (`zgetrf`/`zgetrs`), which handles any full-rank matrix. See DECISIONS.md.
- **Cross-ion B^H·B imaginary parts.** The global Gram matrix has non-zero
  off-diagonal imaginary parts from structure-factor phase differences
  `exp(i·(k+g)·(R_I−R_J))`. The per-ion code was unaffected (phase cancels
  within each ion). **Must work in `Vec<CudaComplex>` throughout** — extracting
  `.x` to `Vec<f64>` corrupts M and breaks S⁻¹·S identity at 3.4e-6.
- **A/B filter experiment shows no useful difference.** Decision pre-locked
  in Goal 1c: keep bare-H filter; consistent with the bare-H R-ChFSI
  conclusion at §12.
- **Q convention mismatch.** The current per-ion Woodbury inverts `Q`
  directly (Gauss-Jordan, `vnl_data.rs:227-277`); the global form needs the
  same convention. The 1e-10 identity gate falsifies any drift before SCF.

### Why not bypass S⁻¹ entirely via direct ZHEGVD

Discussed at `notes/open-followups.md:822-825` and rejected. ZHEGVD is for
the RR projection; the filter recurrence and Lanczos still need the
spectral operator's eigenvalues to land where the algorithm assumes.
Bypassing S⁻¹ moves the bug to a wronger place.

## Discriminator Stack (cheap → expensive)

| Gate | Pre-fix | Target | Falsifies |
|------|--------:|-------:|-----------|
| `cargo check --workspace` | green | green | build sanity |
| `cargo clippy --workspace -- -D warnings` | green | green | lint sanity |
| `Zpotrf/Zpotrs` round-trip on synthetic 4×4 HPD | not present | err < 1e-12 | "Cholesky bindings work" |
| `Zpotrs` n_bands=8 multi-RHS on 16×16 HPD | not present | err < 1e-12 | "multi-RHS solve works" |
| `s_inv_baseline_post_typo_fix` (Goal 0) | 0.014 (contaminated) | **measured value, recorded** | "true per-ion baseline established" |
| `s_inv_s_identity_test` ‖S⁻¹·S·ψ−ψ‖_∞ | **\<rebaselined ζ from Goal 0\>** | < 1e-10 | "global Woodbury actually inverts S" |
| Lanczos S⁻¹·H vs bare-H `b_up` delta (iter-1) | n/a | logged, sign matches expected | "Lanczos sees the right operator" (Goal 1b) |
| iter-1 band-1 vs CASTEP `.bands` (Ha) | Δ ≈ 0.025 (filter on) / 0.009 (ndeg=0) | Δ < 0.005 | "filter no longer perturbs eigenvectors" |
| iter-2 V_eff range (Ha) | 26.06 (ndeg=4) / 8.71 (ndeg=0) | < 9.5 Ha | "no iter-2 ρ_nm shift" |
| **iter-3 Cu D_screened amax (Ha), bare-H filter** | 362 (ndeg=4) / 64 (ndeg=0) | **< 10 Ha** | **"§12 explosion fixed by S⁻¹ in apply path"** |
| **iter-3 Cu D_screened amax, S⁻¹·H filter (Goal 1c A/B)** | n/a | logged; decision per Goal 1c criterion | "filter recurrence needs S⁻¹?" |
| **iter-3 band-1 vs CASTEP (Ha)** | −16.74 (ndeg=4) / −2.59 (ndeg=0) | **\|Δ\| < 0.05** | **"SCF actually heads to convergence"** |
| iter-2 `apply_s_inverse` wall (s) | per-ion baseline | ≤ baseline | "global Woodbury isn't a perf regression" |

The Goal-0 cell stays as `<rebaselined ζ from Goal 0>` until that commit
lands; the Goal-0 commit fills the value into this table. The first four
gates are unit-level and run in seconds — they lead, and a failure there
falsifies the implementation before the multi-iteration SCF tests run.

## Critical Files

| Path | Lines | Change |
|------|-------|--------|
| `src/eigensolver/vnl_data.rs` | 320 | **Goal 0** — typo `m_inv.iter()` → `s_inv.iter()` |
| `src/eigensolver/vnl_data.rs` | 28 (struct), 97 (precompute), 213-325 (replace Gauss-Jordan + per-ion s_inv path) | concatenated B, cached `B^H·B`, Cholesky factor of M; drop `s_inv_mat` |
| `src/eigensolver/chebyshev.rs` | 820-900 | replace `apply_s_inverse` body with global Woodbury; remove `#[allow(dead_code)]` (Goal 1a) |
| `src/eigensolver/chebyshev.rs` | 461 | insert `apply_s_inverse` after `apply_full_hamiltonian` for Lanczos (Goal 1b) |
| `src/eigensolver/chebyshev.rs` | filter recurrence body | gated `apply_s_inverse` insertion controlled by runtime/test flag (Goal 1c) |
| `src/device/solver.rs` | append after :95 | `cusolverDnZpotrf_bufferSize` / `cusolverDnZpotrf` / `cusolverDnZpotrs` wrappers + tests |
| `tests/ca_scf_convergence.rs` | 698-703 (tighten); `fixed_point_matches_castep_energy` body (extend); new `s_inv_baseline_post_typo_fix` test | tighten to 1e-10; iter-3 D_screened + band-1 asserts; baseline rebaseline test |
| `Cargo.toml` (crate root) | new `[features]` section | add `scf_diag = []` |
| `src/eigensolver/chebyshev.rs` (11 sites), `src/scf.rs` (6 sites) | per `notes/open-followups.md §11` | gate behind `#[cfg(feature = "scf_diag")]` |

## Reuse / Existing Infrastructure

- `src/device/solver.rs:47-95` — existing `Zhegvd` wrapper is the template
  for Cholesky wrappers (same handle, same error type, same buffer-size
  pattern).
- `src/device/blas.rs:90, 146` — `cublasZgemm_v2` and `cublasZgemv_v2` for
  the gemm calls in the apply path and for `B^H·B` at precompute.
- `src/eigensolver/chebyshev.rs:914+` — `apply_s_times` already exposes the
  per-ion β projection pattern; useful as a check for `B·v` correctness
  during dev.
- `tests/ca_scf_convergence.rs:626-704` — `s_inv_s_identity_test`
  discriminator pattern stays as-is, just tightens.
- `chemrust-hamiltonian` — no API change required. Per-ion β and Q are
  already GPU-resident in `VnlIonData`.

## Deferred Items Absorbed

- **D-4 (TASK-D4 contingency)** — *resolved*. The per-ion `apply_s_inverse`
  fallback is what we are *replacing*, so the contingency dissolves into
  this phase.

## Domain Terms

- **Per-ion Woodbury (S̃⁻¹) — pre-Goal-0** — uploads identity as `s_inv_mat`
  due to the line-320 typo, so `apply_s_inverse` reduces to
  `hpsi −= Σ β·(β^H·v)`. This is *not* the per-ion Woodbury; it is the
  identity-Woodbury. Contaminates the 0.014 baseline.
- **Per-ion Woodbury (S̃⁻¹) — post-Goal-0** — true per-ion approximation:
  `Σ_I β_I·(M_I⁻¹)·β_I^H·v` with `M_I = Q_I⁻¹ + β_I^H·β_I`. Cross-ion
  block neglect remains; ζ is whatever Goal 0 measures.
- **Global Woodbury (S⁻¹)** — exact Sherman–Morrison–Woodbury inverse from
  the full `(Q⁻¹ + B^H·B)`. ζ → roundoff. Replaces per-ion Woodbury.
- **Identity gate** — ‖S⁻¹·S·ψ − ψ‖_∞, asserted at < 1e-10 after Goal 1a.
- **Iter-3 explosion gate** — D_screened amax at iter-3 < 10 Ha.
  Distinguishes "S⁻¹ correctness" from "full SCF convergence".
- **A/B filter gate (new)** — paired iter-3 SCF with bare-H vs S⁻¹·H filter
  recurrence (Goal 1c). Pre-locked decision criterion.

## Verification

End-to-end, all run after Goal 0 lands:

- Synthetic Cholesky tests (fast, non-ignored):
  `cargo test --release -p chemrust-scf -- zpotrs_round_trip zpotrs_multi_rhs`.
- Pre-fix baseline lock:
  `cargo test --release -p chemrust-scf -- --ignored s_inv_baseline_post_typo_fix`.
- Identity gate at 1e-10:
  `cargo test --release -p chemrust-scf -- --ignored s_inv_s_identity_test`.
- Iter-3 explosion gate (bare-H filter):
  `cargo test --release -p chemrust-scf -- --ignored fixed_point_matches_castep_energy`.
- Goal 1c A/B paired run (S⁻¹·H filter):
  `cargo test --release -p chemrust-scf -- --ignored fixed_point_matches_castep_energy_filter_with_sinv`.
- Iter-2 V_eff range:
  `cargo test --release -p chemrust-scf -- --ignored iter2_v_eff_range_within_one_ha_of_iter1`.
- Density decomposition match against CASTEP F8:
  `cargo test --release -p chemrust-scf -- --ignored density_decomp_matches_castep_f8_same_inputs`.

All green = phase complete. The Goal 1c paired run produces the A/B data;
the locked decision criterion in Goal 1c determines whether the filter S⁻¹
wiring stays default or reverts. Iter-3 D_screened printed by the test
stderr (release build, no diagnostic feature flag) provides the explosion
gate.

## Reference Data

- CASTEP F8 instrumented run:
  `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/`
- CASTEP `.bands` for band-1 reference:
  `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.bands`
  → band-1 = −1.05502287 Ha
- R-ChFSI paper line-612 corollary (proves R-ChFSI ≡ ChFSI when ζ = 0):
  `reference_paper/2025-rchfsi-inexact-mv-paper.tar.gz` → `main.tex:612`
- Bare-H R-ChFSI commits (this phase builds on top):
  `e16924b`, `902a9ab`, `a0c8dc2`, `0cd46e7`, `6bd2c58`
- Review document driving this revision:
  `notes/plans/phase-global-woodbury/review.md` (three critical findings,
  two design concerns, four minor issues — all absorbed above).
