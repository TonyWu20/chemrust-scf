# Phase: Global Woodbury S⁻¹

**Date:** 2026-05-22
**Status:** Draft
**Branch base:** `feat/phase-rchfsi` (after bare-H R-ChFSI gates green at HEAD = `6bd2c58`)
**Plan source:** `notes/open-followups.md §12`

## Context

The bare-H R-ChFSI experiment (`notes/plans/phase-rchfsi-bare-h/`) cleared the
filter and Lanczos paths of S⁻¹ but did not stop SCF divergence at iter-3. The
ndeg sweep (0/4/8/16/32) at `notes/open-followups.md:715-740` localised the
defect away from the filter quality and onto the per-ion Woodbury inverse used
for `S⁻¹·v`. The discriminator probe `s_inv_s_identity_test` measures
‖S⁻¹·S·ψ₀ − ψ₀‖_∞ ≈ **0.014** — a 1.4% structural error from neglecting the
cross-ion projector overlaps `⟨β_I | β_J⟩` for `I ≠ J`. For the dense Cu(111)+CO
slab (18 ions in 22k Bohr³), those overlaps are non-negligible, and the Das
2025 R-ChFSI tolerance theorem (Theorem 3.4, ζ ∈ [10⁻⁴, 10⁻²]) covers the *norm
of the residual*, not the *eigenvector accuracy* — so the per-ion approximation
sends our SCF to a systematically wrong fixed point.

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

1. **Global Woodbury inverse for `apply_s_inverse`.** Replace the per-ion loop
   in `src/eigensolver/chebyshev.rs:821-900` with a single-shot global solve:
   one batched `B^H·v` gemm, one `(Q⁻¹+B^H·B)⁻¹·(B^H·v)` Hermitian-PD solve via
   precomputed Cholesky, one `B·…` gemm. The Cholesky factor is built once per
   SCF iteration inside `VnlBatchData::precompute` (`src/eigensolver/vnl_data.rs`)
   from concatenated per-ion `q_matrix` and `beta_g`.

   *Effort:* medium. Touches one struct, one apply path, adds two cuSOLVER
   bindings, replaces one CPU Gauss-Jordan inversion with GPU Cholesky.

2. **cuSOLVER Cholesky bindings in `src/device/solver.rs`.** Add safe wrappers
   for `cusolverDnZpotrf_bufferSize` / `cusolverDnZpotrf` /
   `cusolverDnZpotrs`. Mirrors the existing Zhegvd wrapper at
   `src/device/solver.rs:69-100` for shape and error handling.

   *Effort:* small. Self-contained binding layer; one added test that round-trips
   a 4×4 Hermitian PD matrix.

3. **Tighten `s_inv_s_identity_test` to roundoff.** Change the assertion at
   `tests/ca_scf_convergence.rs:698-703` from its current `< 1e-6` lenient
   threshold to `< 1e-10`. This is the lead discriminator: anything looser
   passes a partial fix, anything tighter is unreasonable for double precision
   on a 350×350 system.

   *Effort:* trivial. One line. Lives in the same commit as Goal 1.

4. **Iter-3 SCF convergence gate.** Run
   `fixed_point_matches_castep_energy` to depth 3 and assert: D_screened amax
   at iter-3 < 10 Ha **and** band-1 |Δ vs CASTEP −1.055 Ha| < 0.05 Ha. This is
   the §12 falsification gate — proves the global Woodbury removes the SCF
   feedback explosion.

   *Effort:* small. Extend the existing test, no new fixtures.

5. **Diagnostic prints behind `cfg(feature = "scf_diag")`.** Move the ten
   `[Lanczos]`, `[Lanczos@call]`, `[Chebyshev]`, `[R-ChFSI]`, `[V_eff]`,
   `[RR]` eprintlns from the §11 investigation into a feature-gated path.
   Default builds get clean stderr; debug sessions re-enable with
   `cargo test --release --features scf_diag`.

   *Effort:* small. No semantic changes; release build does not bring back the
   diagnostic noise.

6. **Wall-time observability gate.** Capture iter-2 `apply_s_inverse` total
   wall time in the discriminator log. Asserts only that it is *not worse* than
   the per-ion baseline. Surfaces accidental regressions (e.g. uploading
   `B^H·B` every iteration when it should be cached). Lightweight — one
   `Instant::now()` pair plus an eprintln behind `scf_diag`.

   *Effort:* trivial.

## Scope Boundaries

**In scope:**
- `src/eigensolver/chebyshev.rs:821-900` — replace `apply_s_inverse` body.
- `src/eigensolver/vnl_data.rs` — add concatenated `B`, `Q⁻¹+B^H·B` Cholesky
  factor, drop per-ion `s_inv_mat` (becomes dead → delete).
- `src/device/solver.rs` — add `Zpotrf` and `Zpotrs` safe wrappers.
- `tests/ca_scf_convergence.rs` — tighten `s_inv_s_identity_test` to 1e-10;
  extend `fixed_point_matches_castep_energy` to assert iter-3 gates.
- Remove `#[allow(dead_code)]` from `apply_s_inverse` once it is live again.
- Feature-gate diagnostic eprintlns under `scf_diag`.

**Out of scope:**
- **R-ChFSI simplification to standard ChFSI** — Das line-612 corollary lets
  us collapse the residual recurrence once ζ is roundoff, but doing it in the
  same phase couples two unrelated changes and makes blame attribution hard if
  a gate fails. File a follow-up.
- **Full SCF convergence to CASTEP total energy** (`-24110.96665069 eV`).
  Convergence past iter-3 depends on density mixing, occupancy update, and
  potential/charge mixing parameters — orthogonal to S⁻¹ correctness. Iter-3
  D_screened gate is the falsifier here.
- **Tier-1 augmentation Q·SF GPU caching (§9, ~100-200 s/iter savings).**
  Self-contained perf phase; no S⁻¹ dependency.
- **GPU D-matrix screening (deferred D-6, ~3 s/iter savings).** Independent of
  S⁻¹ correctness; defer.
- **Reverting `pub(crate) c2c_inverse_inplace` (D-1) and normalising `test_api`
  (D-2)** — cosmetic; bundle into a future cleanup pass.
- **Quantifying H-eigenvalue vs generalized-eigenvalue deviation (D-5)**
  — research question, not a phase-blocker.

## Design Notes

### Why exact (not "smaller ζ") matters
Das 2025 Theorem 3.4 proves R-ChFSI converges to the correct *residual norm*
under ζ ∈ [10⁻⁴, 10⁻²]. But the eigenvectors of `S̃⁻¹·H` differ from those of
`S⁻¹·H` by O(ζ), and the SCF feedback loop (§12, eqn cluster after the gate
table) amplifies that into ρ_nm corruption → V_eff range explosion → D_screened
runaway. We are not solving the eigenvalue problem the paper analyzes; we are
solving a *different* eigenvalue problem on `S̃⁻¹·H` and calling it H/S. The
fix has to drive ζ to roundoff.

### Block structure
`Q⁻¹` is block-diagonal in ions (each ion's Q is independent). `B^H·B`
introduces the cross-ion blocks — that is the entire mathematical content of
the fix. Concretely:

```
Q⁻¹ = blkdiag(Q_1⁻¹, Q_2⁻¹, …, Q_N⁻¹)        # already known per-ion
B^H·B has N×N blocks, block (I,J) = β_I^H·β_J  # currently dropped for I≠J
M = Q⁻¹ + B^H·B is dense Hermitian PD ~350×350 # one Cholesky per SCF iter
```

Cholesky of a 350×350 Hermitian PD matrix is microseconds on Pascal+. The
expensive part is the one-time `B^H·B` gemm at SCF setup.

### Cost comparison (per `S⁻¹·v` apply, n_pw ≈ 50k, n_bands = 160)
- **Per-ion (current):** 18 ions × 3 gemms × per-ion-shape ≈ 18 × O(n_pw · ne²)
  ≈ 18 × O(50000 · 64) launches, with launch overhead dominating.
- **Global Woodbury (this phase):** 1 gemm `B^H·v` (n_pw × 350 × n_bands), one
  `Zpotrs` (350 × 350 × n_bands triangular solve), 1 gemm `B·temp`.
  Strictly fewer launches, larger arithmetic intensity. Expected: faster, not
  slower. Goal 6 verifies.

### Risks
- **Q convention mismatch.** The current per-ion Woodbury inverts `Q` directly
  (Gauss-Jordan, `vnl_data.rs:227-277`); the global form needs the same
  convention. The 1e-10 identity gate falsifies any convention drift before SCF
  ever runs.
- **Cholesky failure on near-singular M.** If `B^H·B` plus `Q⁻¹` is poorly
  conditioned, `Zpotrf` can fail with `info > 0`. Mitigation: report the leading
  minor and add the failure as a separate gate.
- **Memory.** 350×350 complex double = 1.96 MB per Cholesky factor. Negligible.

### Why not bypass S⁻¹ entirely via direct ZHEGVD
Discussed at `notes/open-followups.md:822-825` and rejected. ZHEGVD is for the
RR projection; the filter recurrence and Lanczos still need the spectral
operator's eigenvalues to land where the algorithm assumes. Bypassing S⁻¹
moves the bug to a wronger place.

## Discriminator Stack (cheap → expensive)

| Gate | Pre-fix | Target | Falsifies |
|------|--------:|-------:|-----------|
| `cargo check --workspace` | green | green | build sanity |
| `cargo clippy --workspace -- -D warnings` | green | green | lint sanity |
| `Zpotrf/Zpotrs` round-trip on synthetic 4×4 HPD matrix | not present | err < 1e-12 | "Cholesky bindings work" |
| `s_inv_s_identity_test` ‖S⁻¹·S·ψ−ψ‖_∞ | 0.014 | < 1e-10 | "global Woodbury actually inverts S" |
| iter-1 band-1 vs CASTEP `.bands` (Ha) | Δ ≈ 0.025 (filter on) / 0.009 (ndeg=0) | Δ < 0.005 | "filter no longer perturbs eigenvectors" |
| iter-2 V_eff range (Ha) | 26.06 (ndeg=4) / 8.71 (ndeg=0) | < 9.5 Ha | "no iter-2 ρ_nm shift" |
| **iter-3 Cu D_screened amax (Ha)** | 362 (ndeg=4) / 64 (ndeg=0) | **< 10 Ha** | **"no SCF feedback explosion (§12)"** |
| **iter-3 band-1 vs CASTEP (Ha)** | −16.74 (ndeg=4) / −2.59 (ndeg=0) | **\|Δ\| < 0.05** | **"SCF actually heads to convergence"** |
| iter-2 `apply_s_inverse` wall (s) | per-ion baseline | ≤ baseline | "global Woodbury isn't a perf regression" |

The first four gates are unit-level and run in seconds. They lead — a failure
there falsifies the implementation before the multi-iteration SCF test runs.

## Critical Files

| Path | Lines | Change |
|------|-------|--------|
| `src/eigensolver/chebyshev.rs` | 820-900 | replace `apply_s_inverse` body with global Woodbury; drop `#[allow(dead_code)]` |
| `src/eigensolver/vnl_data.rs` | 18-35 (struct), 94-329 (precompute) | replace per-ion `s_inv_mat` with concatenated `B` + Cholesky factor of `Q⁻¹+B^H·B` |
| `src/device/solver.rs` | append after :100 | add `cusolverDnZpotrf_bufferSize`, `cusolverDnZpotrf`, `cusolverDnZpotrs` wrappers |
| `tests/ca_scf_convergence.rs` | 626-704 | tighten `s_inv_s_identity_test` threshold to 1e-10; extend `fixed_point_matches_castep_energy` with iter-3 D_screened + band-1 assertions |
| `Cargo.toml` (crate root) | features section | add `scf_diag` feature flag |
| `src/eigensolver/chebyshev.rs`, `src/scf.rs` | diagnostic eprintln sites listed in `notes/open-followups.md §11.6` | gate behind `#[cfg(feature = "scf_diag")]` |

## Reuse / Existing Infrastructure

- `src/device/solver.rs:49-100` — existing `Zhegvd` wrapper is the template for
  Cholesky wrappers (same handle, same error type, same buffer-size pattern).
- `src/device/blas.rs:90, 146` — `cublasZgemm_v2` and `cublasZgemv_v2` for the
  two gemm calls in the apply path.
- `src/eigensolver/chebyshev.rs:914+` — `apply_s_times` already exposes the per-ion
  β projection pattern; useful as a check for `B·v` correctness during dev.
- `tests/ca_scf_convergence.rs:626-704` — `s_inv_s_identity_test` discriminator
  pattern stays as-is, just tightens.
- `chemrust-hamiltonian` — no API change required. Per-ion β and Q are already
  GPU-resident in `VnlIonData`.

## Deferred Items Absorbed

- **D-4 (TASK-D4 contingency)** — *resolved*. The per-ion `apply_s_inverse`
  fallback is what we are *replacing*, so the contingency dissolves into this
  phase.

## Domain Terms

- **Per-ion Woodbury (S̃⁻¹)** — the approximation currently in tree at
  `chebyshev.rs:821-900`. Treats `B^H·B` as block-diagonal in ions, neglecting
  cross-ion overlaps `⟨β_I | β_J⟩` for `I ≠ J`. ζ ≈ 0.014 measured. Used by
  `[[castep_check_continuation_convention]]` as the inner-product matrix.
- **Global Woodbury (S⁻¹)** — the exact Sherman–Morrison–Woodbury inverse
  computed from the full `(Q⁻¹ + B^H·B)`. ζ → roundoff. Replaces per-ion
  Woodbury in this phase.
- **Identity gate** — the test ‖S⁻¹·S·ψ − ψ‖_∞, a direct falsifier of the
  inverse implementation independent of any SCF dynamics. Currently asserted at
  1e-6 (lenient); tightened to 1e-10 (roundoff) in this phase.
- **Iter-3 explosion gate** — the §12 falsifier: D_screened amax at iter-3 must
  stay below 10 Ha. Distinguishes "S⁻¹ correctness" from "full SCF
  convergence" (the latter depends on mixing, which is out of scope).

## Verification

End-to-end: `cargo test --release -p chemrust-scf -- --ignored
fixed_point_matches_castep_energy s_inv_s_identity_test
iter2_v_eff_range_within_one_ha_of_iter1
density_decomp_matches_castep_f8_same_inputs`. All four green = phase
complete. Iter-3 D_screened printed by the test stderr (release build, no
diagnostic feature flag) provides the explosion gate.

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
