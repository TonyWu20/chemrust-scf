# Phase 2 Perf Opt: Collapse iter-wall-time to enable end-to-end SCF debugging

**Date:** 2026-05-21
**Status:** Draft
**Branch:** `feat/phase-2-perf-opt` (chemrust-scf), `feat/expose-energy` (chemrust-hamiltonian)
**Parent:** `feat/phase-2` (F-0/F-1 prep committed at `a01323d`)

## Goals

### Goal 1 — GPU `QSfCache` for augmentation density (§9 Tier-1)

Move `compute_aug_density_fine` off the CPU. Currently `apply_q_and_sf`
walks the fine grid per ion per (n_exp, m_exp) channel pair on CPU
(~100–200 s/iter on Cu111_CO, ~50% of iter-2's 531 s budget). Cache the
geometry-static `Q^I_{nm}(G) · exp(-iG·R_I)` per ion on GPU once at SCF
init; per iteration, reduce `Σ_I (Q^I ⊙ SF_I) · ω^I` via batched gemm and
inverse-FFT to real space. Keep `beta_psi_per_ion` GPU-resident
(`Vec<CudaSlice<CudaComplex>>`) so `ω^I_{nm}` is built on GPU without the
D2H/H2D introduced at §8's resolution.

Why now: the user cannot meaningfully test SCF convergence (or
`fixed_point_matches_castep_energy`) when one iteration costs 4 minutes.
Reaching iter-2 took 531 s, which made the §8 augmentation-density debug
session painfully slow even with a single fixture. The Tier-1 fix is
§9's recommended self-contained entry point — does not touch the SCF
state machine, with `compute_aug_density_fine` becoming a thin GPU
wrapper.

### Goal 2 — Speed up `VnlBatchData::precompute` (§9 Tier-2, rayon-first)

After Goal 1 collapses iter wall-time, `VnlBatchData::precompute`
(~200–300 s) becomes the new bottleneck — radial Bessel transform on log
grid, per species, currently sequential. User has multi-species
converged jobs ready for testing (NiO, Fe2O3, ZnO + larger), so the
single-species amortization (Cu's Q reused across 18 ions) will not
hide this on those fixtures.

Two implementation options, pick during exploration:

- **Goal 2a (cheap, first):** rayon-parallelize the radial sum over the
  species-pair index (`compute_beta_g`, `precompute_q_on_grid`,
  `compute_screened_d_from_fft` ion loops). On 8-core, expect 4–8×.
- **Goal 2b (only if needed):** GPU-port `precompute_q_on_grid` radial
  Bessel transform kernel. Only do this if Goal 2a + a multi-species
  fixture profile shows the radial transform still dominant.

### Goal 3 — Document USPP / FFT / unit invariants (CONTEXT.md + ADR-0003)

The §8 `inv_omega` factor-of-Ω bug was exactly the regression §2 warned
about and the §1d (FFT plan ordering swap) bug compounded with the RR
transpose bug. Both were architectural invariants that lived only in
implementer's heads. Codify them before the next regression:

- **CONTEXT.md additions:** "Density unit convention" (raw `ρ × Ω` not
  Ha/Bohr³ for ρ entering Poisson/XC), "FFT axis convention"
  (`RealGrid<T>` / `RecipGrid<T>` invariant), "Augmentation Density"
  domain term.
- **ADR-0003: USPP density assembly.** Records the decision that
  smooth-PW ρ and augmentation ρ_aug live as separate channels with a
  defined assembly path (`construct_density_gpu` produces smooth-only,
  `compute_aug_density_gpu` adds ρ_aug, `build_v_eff_with_energy_impl`
  sums them before Poisson + XC). Rationale: §8's resolution shows the
  channels have different unit conventions, different geometry-staticity
  (smooth = per-iter, Q·SF = once-per-cell), and the fixture density
  already contains both.

### Goal 4 — Wall-time regression guard

Single `#[ignore]`-gated release test asserts iter-2 wall time stays
within budget. Without this, the next refactor silently re-introduces a
CPU roundtrip in the density path and we discover it months later.
Empirically measure on test GPU after Goal 1 lands, then set budget at
1.5× the measured value. Test name (proposed):
`iter2_wall_time_within_budget_post_q_cache` in `tests/perf_gate.rs`.

## Scope Boundaries

**In scope:**

- GPU `QSfCache` (per-cell, geometry-static) under chemrust-hamiltonian.
- `compute_aug_density_gpu` consuming GPU-resident `beta_psi_per_ion` +
  cached `Q·SF` + occupations.
- Keep `beta_psi_per_ion` GPU-resident across the RR → density boundary
  (drop the §8 D2H/H2D roundtrip).
- Rayon-parallelization of `VnlBatchData::precompute` per-species and
  per-ion loops.
- Multi-species fixtures: at least one of NiO / Fe2O3 / ZnO loaded and
  the perf gate measured against it.
- CONTEXT.md updates + ADR-0003 (USPP density assembly).
- One wall-time regression test (`#[ignore]`, release, GPU-gated).

**Out of scope:**

- Diagnostic `eprintln!` cleanup (`[Lanczos]`, `[Chebyshev]`, `[RR]`,
  `[ConstructDensity]`, `[V_eff]`, `[Density]`, `[NewDensity]`, `[psi]`)
  — defer; user wants to keep them until end-to-end convergence is
  decided in a later phase.
- `fixed_point_matches_castep_energy` ≤ 2e-4 eV total-energy assertion
  — testable cheaply only *after* Goals 1+2 land; may add a smoke check
  but not a tight assertion in this phase.
- Cold-start initial density (atomic superposition / CASTEP C-binding
  kickoff) — Phase 3+, see open-followups §3 caveat in TASKS.md F-1.
- Pulay/DIIS/Kerker mixing changes — Phase 3 if convergence demands it.
- Mixed-precision FP32/TF32 inner Chebyshev loop (`suggested_algorithm.md`
  §"Performance Advantages") — separate phase.
- GPU port of `compute_screened_d_from_fft` beyond rayon — falls out
  naturally if Goal 2a is enough; otherwise revisit.
- F-2..F-N test cases against Cu111_CO (depend on Group F, separate
  branch).

## Design Notes

- **`QSfCache` ownership:** lives in chemrust-hamiltonian-core (next to
  `compute_screened_d_from_fft` which already FFTs `Q^I_{nm}(G)`). The
  scf crate constructs it once inside `ScfIteration::new` (geometry is
  known at construction; pseudopotentials are immutable). Memory budget:
  18 ions × ~100–300 non-zero (n,m) pairs × 437k Complex64 ≈ 1–3 GB on
  Cu111_CO. Verify VRAM headroom on Pascal cc 6.1 before locking in.
- **β·ψ GPU residency:** §8 stored `beta_psi_per_ion` as host-side
  `Vec<Vec<Complex64>>`. Replace with `Vec<CudaSlice<CudaComplex>>` (or
  a single contiguous `[n_ions × n_proj × n_bands]` device slice) so
  ω^I_{nm} = `Σ_b occ_b · conj(βψ)_n,b · (βψ)_m,b` is one batched gemm.
- **`compute_aug_density_gpu` signature:** takes
  `(&QSfCache, &Gpu<BetaPsiPerIon>, &Gpu<Occupations>, &mut PcieAccount,
  &Stream)` and returns `RealGrid<f64>` (post-IFFT). Add D2H tracking on
  the final result transfer; the gemm + IFFT stay GPU-resident.
- **PcieAccount:** new `Q·SF` upload at SCF init is one-shot H2D
  (geometry-static), tracked separately from per-iter assertion. ω^I and
  ρ_aug stay GPU-resident → zero per-iter PCI-E for the augmentation
  channel.
- **rayon scope (Goal 2a):** thread the per-species `precompute_q_on_grid`
  loop and the per-ion `compute_screened_d_from_fft` loop with
  `par_iter`. Note: chemrust-hamiltonian has a "no `for` loop rule"
  (commit `6b16508`) — combinators are already idiomatic, rayon
  composes via `par_iter().map(...).collect()`. The recently reverted
  `compute_screened_d_from_fft` rayon (commit `d73d42a`) should be
  investigated: why was it reverted? Re-enable carefully if the regression
  it caused is now understood.
- **Multi-species fixture choice:** prefer NiO (2 species, 4 atoms,
  smaller cell than Fe2O3) for the first cross-check — fast to load
  and exercises species-pair index without slowing the iteration loop.
  Fe2O3 / ZnO add as the perf gate stabilizes.
- **Discriminator for Goal 1 correctness:** existing iter-2 V_eff range
  test (`|range_iter2 - range_iter1| < 1.0 Ha`, currently 0.13 Ha) — if
  GPU augmentation matches host, this stays green. Plus a tighter
  `‖ρ_aug_gpu − ρ_aug_cpu‖_∞ < 1e-10` discriminator on the augmentation
  buffer.
- **Discriminator for Goal 2 correctness:** β·ψ projections and screened-D
  matrix entries match the pre-rayon serial run to bit-exact tolerance on
  rerun; results match within 1e-12 across runs (rayon non-determinism
  guard).

## Deferred Items Absorbed

- **`notes/pr-reviews/phase-2-fixes/deferred.md` #1** — Double NVRTC
  compilation per SCF iteration (shared `GpuContext`). **Not absorbed**;
  this is iter-level orchestration, not the augmentation hot path.
  Re-list for Phase 3.
- **`notes/pr-reviews/phase-2-fixes/deferred.md` #2** — `CudaKernelSet`
  location (move out of `eigensolver/chebyshev.rs`). **Partially
  absorbed**: any new GPU kernels added for `compute_aug_density_gpu`
  go into `src/device/kernels.rs` (or a new module), not into chebyshev.
- **`notes/pr-reviews/phase-2-fixes/deferred.md` #3** — PcieAccount
  tracking in `construct_density`. **Absorbed** for the new aug-density
  path (PcieAccount-instrumented).
- **`notes/pr-reviews/phase-2/deferred.md` #21** — Kerker
  `current_density_in` D2H/H2D roundtrip. Not absorbed; mixing-loop perf
  is a Phase 3 concern.
- **`notes/pr-reviews/phase-2/deferred.md` #22** — cuFFT plan creation
  cached per mix call. Not absorbed; same reason as #21.

## Domain Terms

- **Augmentation Density (ρ_aug)** — the USPP charge contribution
  `ρ_aug(r) = Σ_I Σ_{n,m} ω^I_{nm} · Q^I_{nm}(r)` where ω^I_{nm} = Σ_b
  occ_b · ⟨β_{I,n}|ψ_b⟩⟨ψ_b|β_{I,m}⟩. Lives as a separate channel from
  the smooth plane-wave ρ_PW. Total ρ = ρ_PW + ρ_aug. CASTEP `.den_fmt`
  and `.castep_bin` already store the sum. Resolves the prior ambiguity
  where "density" alone could mean either channel — call out the
  channel explicitly when context is not obvious.
- **QSfCache** — per-cell GPU-resident cache of `Q^I_{nm}(G) ·
  exp(-iG·R_I)` (Q-functions in reciprocal space multiplied by
  per-ion structure factor). Geometry-static: built once at SCF init,
  invariant under SCF iterations. Used by `compute_aug_density_gpu` to
  reduce ω^I_{nm} into ρ_aug(G).
- **Q-on-grid** — the real-space (or reciprocal-space) tabulation of
  USPP augmentation function `Q^I_{nm}(r)` for ion I and channel-pair
  (n,m). `precompute_q_on_grid` builds this per species via radial
  Bessel transform on the log grid. Distinct from `q_ij` which usually
  denotes the integrated `∫Q^I_{nm}(r) d³r` quantity.

## Validation criteria summary

| Criterion | Target | Discriminator |
|-----------|--------|---------------|
| Iter-2 V_eff range matches Goal-1 baseline | `|Δ| < 0.2 Ha` | Stricter than current 1 Ha gate |
| ρ_aug_gpu vs ρ_aug_cpu reference | `‖·‖_∞ < 1e-10` | Bit-equivalence within FP64 |
| Iter-2 wall time on Cu111_CO | empirically set 1.5× of post-G1 measurement | Perf gate test, `#[ignore]` release |
| `cargo test --workspace` (CPU tests) | all green | Regression guard |
| `cargo clippy -- -D warnings` | clean | Lint guard |
| Multi-species fixture (NiO) loads + runs ≥ 2 iters without crash | required for Goal 2 acceptance | New `tests/nio_perf_smoke.rs` |
| ADR-0003 + CONTEXT.md sections committed | required for Goal 3 acceptance | Inspection |
