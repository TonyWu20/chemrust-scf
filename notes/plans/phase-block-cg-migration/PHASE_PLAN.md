# Phase Plan: Block CG Eigensolver Migration

**Status:** DRAFT. Conceived 2026-05-24 after the Chebyshev-RR architecture
was determined structurally unsuitable for the project's four sharpened
requirements (GPU + USPP + metals + CASTEP precision).
**Predecessor:** `feat/phase-global-woodbury` branch (Chebyshev-RR
investigation). All learnings recorded in
`notes/debug/debug-20260520-1311/` through
`notes/debug/debug-20260524-blow-tightening/`.

## Context — Why this phase

The project's four sharpened requirements (per the 2026-05-24
strategic conversation) are:

1. GPU-accelerated SCF
2. USPP support (ultra-soft pseudopotentials with augmentation)
3. Metallic systems (fractional occupations, Fermi smearing)
4. CASTEP precision (1e-5 eV total-energy parity for Cu111+CO)

The previous architecture (Chebyshev filtering + subspace Rayleigh-Ritz)
cannot satisfy (4) for (2)+(3) due to ZHEGVD's degenerate-cluster gauge
choice coupling with V_eff/density. This was empirically established
across ~10 sessions of investigation. See
[[chebyshev_rr_architecturally_unsuitable]] memory.

CASTEP-GPU-port is unavailable as a working reference (per user 2026-05-24:
"haunted by missing a significant piece of energy contribution in SCF").
So we cannot port-from-working-GPU. We must derive from working-CPU-CASTEP
(`~/Downloads/CASTEP-6.11-nixos/Source/Functional/electronic.f90`),
Payne et al. 1992, and the Abinit GPU paper at
`~/programming/CASTEP-GPU-port/2604.11139v1`.

## Algorithmic choice: block CG

Block CG (batched band-by-band CG) keeps the GPU-friendly batched kernels
(apply_H, apply_S, batched FFT, batched gemm) while preserving CG's
per-band convergence semantics. Each band optimizes its own Rayleigh
quotient with its own search direction and conjugation parameter; no
shared H_sub diagonalization. The rotation cascade that defeated
Chebyshev-RR cannot occur.

USPP support is native (residual `r_b = (H − ε_b·S)·ψ_b` uses existing
apply_H and apply_S primitives). Metals work natively (occupations enter
via density formula, outside the eigensolver). Convergence to CASTEP
precision is proven by CASTEP itself.

The trade-off vs Chebyshev: lower per-step arithmetic intensity (1
apply_H per inner iteration vs k per Chebyshev step). Compensated by
fewer inner iterations and residual-driven per-band early exit. Net
wall-clock is competitive on GPU (per Abinit paper benchmarks).

## Phase 0 — Algorithm de-risking (1-3 days)

Goal: verify that band-by-band CG reaches CASTEP precision on Cu111+CO
**before** committing to GPU batching work.

Approach: implement a CPU-only, serial, band-by-band CG using existing
`chemrust-hamiltonian` apply_H + apply_S + the Woodbury S⁻¹. Process one
band at a time. Slow (n_bands × per-band CG iterations) but unambiguous.
Run on Cu111+CO fixture with CASTEP's converged ψ as initial guess (a
near-noop check) and with a random initial guess (the convergence
test).

**Gate**: starting from CASTEP ψ + CASTEP V_eff, iter-1 CG must return
ψ within 1e-10 of input (consistency check). Starting from random ψ +
CASTEP V_eff, CG must converge band-0 to within 1e-6 Ha of CASTEP A1
(−1.05502287 Ha) within 50 inner CG steps.

**If Phase 0 fails**: investigate before scaling. Possible failure modes
include:
- Teter-Payne-Allan preconditioner wrong → bands converge slowly or to
  wrong values
- Per-band line-search numerics fragile → infinite loops or oscillation
- USPP convention mismatch in apply_S → bands converge to wrong basis
  (already ruled out by per-band S-norm test at 1.0000, but Phase 0
  validates end-to-end)

**If Phase 0 passes**: proceed to Phase 1 with confidence that algorithm
+ existing kernels are correct.

## Phase 1 — Block CG implementation (~1 week)

Goal: replace serial band-by-band loop with batched form. Same algorithm,
batched primitives.

**Files to create**:
- `src/eigensolver/block_cg.rs` (new module, ~300 lines)
  - `pub fn block_cg_iteration(ψ_in, v_eff_dev, vnl_data, ...) -> Result<(ψ_out, eigenvalues), Error>`
  - Inner loop: 5-15 CG steps per outer SCF iteration
  - Per-iteration steps: batched H·ψ + S·ψ → batched residual r =
    H·ψ − S·ψ·diag(ε), batched preconditioner z = P⁻¹·r, per-band β
    update, batched search direction d, per-band line search, batched
    update ψ ← ψ + diag(α)·d, S-orthonormalize, batched Rayleigh
    quotient

**Files to modify**:
- `src/scf.rs`: replace `chebyshev_filter` call (L586, L723) with
  `block_cg_iteration`. The signature is similar (psi_gpu, v_eff_gpu,
  vnl_data, ...) so the surrounding plumbing stays. Remove
  `rayleigh_ritz` call (L597-604) — block CG produces eigenvalues
  directly, no separate RR needed.
- `src/eigensolver/mod.rs`: add `pub mod block_cg;`
- Optionally keep `pub mod chebyshev` for diagnostic uses, but the
  production SCF stops calling it

**Files to keep, fully reusable**:
- `chemrust-hamiltonian/` — V_NL, V_loc, β, Q, D matrices, all unchanged
- `src/eigensolver/chebyshev.rs:707-833` — `apply_full_hamiltonian` is
  called from block_cg.rs unchanged
- `src/eigensolver/chebyshev.rs:934-1057` — `apply_s_times` called
  unchanged
- `src/eigensolver/chebyshev.rs:758-933` — `apply_s_inverse` (Woodbury)
  called unchanged from preconditioner
- `src/density.rs` — density assembly, occupations, V_eff assembly all
  unchanged
- `src/mixing.rs` — density mixing logic unchanged
- `src/device/` — GPU plumbing fully reusable
- `tests/fixtures/cu111_co.rs` — fixture loader unchanged
- `tests/ca_scf_convergence.rs` — most tests reusable; the cascade gate
  becomes the convergence gate

**Files to remove or demote**:
- `src/eigensolver/chebyshev.rs:1273-1849` — `chebyshev_filter` and
  associated R-ChFSI machinery. Either delete (cleanest) or mark
  `#[deprecated]` and gate behind a feature flag for research use.
- `src/eigensolver/rayleigh_ritz.rs` — `PinMode` enum, `RrPinConfig`,
  the entire PostRr scaffolding. Delete. `rayleigh_ritz` itself can stay
  for diagnostic use (e.g., post-CG basis polish) but is removed from
  production path.
- Stashes `stash@{0}`, `stash@{1}` — drop after Phase 1 ships.
- All `CHEMRUST_PIN_MODE`, `CHEMRUST_BLOW_PAD`, `RrPinConfig::from_env()`
  env-var infrastructure.

## Phase 2 — Validation (~3-5 days)

Goal: re-establish CASTEP-precision convergence on Cu111+CO using the
new eigensolver.

Tests to reactivate / rewrite:
- `iter1_drift_from_castep_state_is_bounded` (Q1) — should pass at 1 mHa,
  not 20 mHa
- `scf_converges_to_castep_energy_at_castep_tolerance` (Q2) — should
  pass at 1e-5 eV
- `subspace_projector_iter1_vs_castep` — should produce Cu-3d ratio
  >0.999 (the 0.893 Chebyshev-RR floor was structural; CG should not have
  this)
- `overlap_iter2_against_castep` — should pass at avg > 0.99
- `cascade_iter3_diagnostic_tight` — should pass at the original 0.1 Ha
  gate (no need for the loose-tier split discussed in
  `tolerance-conflation-in-acceptance-test` failure-pattern)
- All RR validation tests — repurpose as diagnostic of any optional
  post-CG basis polish

If any of these fail at the original tight tolerances after Phase 1, the
issue is implementation, not algorithm. Debug per usual.

## Phase 3 — GPU performance tuning (~3-5 days)

Goal: ensure block CG matches or exceeds Chebyshev-RR's GPU performance.

The kernels are the same (batched FFT, batched gemm, batched preconditioner).
What changes is the orchestration:
- Chebyshev did k=8 H apply per outer iteration with 1 MPI transpose
- Block CG does 5-10 inner iterations × 1 H apply per outer, with
  inner-iter S-orthonormalization

Profile vs Chebyshev-RR baseline on the same Cu111+CO fixture. Optimize
S-orthonormalization (currently classical 2-pass GS at chebyshev.rs:1708-1799
— possibly fuse the two passes for fewer GPU launches).

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Block CG itself fails to reach CASTEP precision | LOW — CASTEP and VASP both prove it works at this precision | Phase 0 derisks before GPU work |
| Phase 0 reveals USPP convention bug in apply_S | LOW — per-band S-norm test already passes at 1.0000 | Investigate; bug would be common to both algorithms |
| GPU block CG slower than Chebyshev-RR | MEDIUM — Chebyshev has higher arithmetic intensity per iteration | Accept; CASTEP precision is the goal, not raw FLOPS |
| Convergence harder for metals than for insulators | MEDIUM — fractional occupations near ε_F may need more inner CG steps | Use CASTEP's empirical "5-10 inner steps per SCF" as a starting point |
| Phase 0 takes >3 days (algorithm investigation rabbit hole) | MEDIUM | Time-box. After 3 days, escalate to user for go/no-go on Phase 1 |

## What stays from `feat/phase-global-woodbury` branch

After Phase 1 lands, the branch will retain:
- All chemrust-hamiltonian work (V_NL/V_loc/β/Q/D)
- All density / V_eff / Hartree / V_xc / Ewald work
- All GPU plumbing, fixture loaders, CASTEP comparison anchors
- All forensic notes in `notes/debug/*` (institutional memory)
- The diagnostic-test infrastructure (subspace_projector, per_band_S_norm,
  etc. — these tests are algorithm-agnostic and will validate block CG too)

What's lost:
- ~600 lines of Chebyshev-RR code
- ~150 lines of Procrustes pin / PinMode scaffolding
- Two stashed failed pin attempts

## When to begin

After user explicit go-ahead. The plan is heavy on Phase 0 derisking
because the previous architecture took ~10 sessions to falsify. We
should not commit weeks of GPU work to a new architecture without first
validating in 1-3 days that the algorithm itself can reach CASTEP
precision on the existing CPU + Woodbury infrastructure.

## Reading list before Phase 0 starts

1. Payne, Teter, Allan, Arias, Joannopoulos (1992) "Iterative
   minimization techniques for ab initio total-energy calculations",
   RMP 64:1045 — the original CG description
2. `~/Downloads/CASTEP-6.11-nixos/Source/Functional/electronic.f90` —
   band-loop structure; specific Fortran routines to mirror
3. `~/programming/CASTEP-GPU-port/2604.11139v1` (Abinit paper) — block
   CG on GPU, specifically the preconditioner choice and the
   S-orthonormalization frequency
4. `notes/debug/debug-20260524-blow-tightening/RESOLUTION.md` and
   `CASTEP_ANCHORED_PIN_PROBE.md` — why we are here

## Out of scope for this phase

- Davidson eigensolver (alternative considered; not chosen because BigDFT
  documented that Davidson struggles with USPP metals)
- LOBPCG (preconditioner choice is open research for USPP metals)
- Direct minimization (CDM/OD) — works for insulators, not for the
  fractional-occupation metals you care about
- Replacing density mixing (current Pulay + Kerker is fine; not the
  bottleneck)
- Tuning chemrust-hamiltonian (D-screening, Q assembly — both validated
  to 4 µHa)
