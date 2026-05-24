# Phase Plan: Eigensolver Migration (Davidson-First, CG-Fallback) + DFT+U Type Design

**Status:** DRAFT, awaits explicit go-ahead.
**Date:** 2026-05-24
**Predecessor:** `feat/phase-global-woodbury` branch (Chebyshev-RR investigation).
All learnings recorded in `notes/debug/debug-20260520-*` through
`notes/debug/debug-20260524-blow-tightening/` plus the relevant memory entries.

## Context — Why this phase

Five sharpened requirements (per the 2026-05-24 strategic conversation):

1. GPU-accelerated SCF
2. USPP support (ultra-soft pseudopotentials with augmentation)
3. Metallic systems (fractional occupations, Fermi smearing)
4. CASTEP precision (1e-5 eV total-energy parity for Cu111+CO)
5. **DFT+U support** (the group's daily workload includes correlated systems)

The Chebyshev-RR architecture cannot satisfy (4) under (2)+(3). The cascade
is structural — see [[chebyshev_rr_architecturally_unsuitable]] memory.

**The load-bearing finding** (per [[locking_is_the_load_bearing_eigensolver_property]]):
the eigensolver must provide **per-band locking**. Once a band converges
(`‖r_b‖_S < tol`), no subsequent ZHEGVD must rotate it.

Two algorithm classes provide locking:
- **Block CG** (batched band-by-band CG): implicit locking via no shared H_sub
- **Block Davidson with explicit lock list**: explicit locking via diagonalizing
  only the unconverged sub-block

A 1-day Phase 0 test decides between them.

## Phase 0 — Locking-mechanism derisk gate (1-2 days)

**Goal**: determine whether per-band locking is sufficient to fix the cascade
on Cu111+CO. This single test decides whether Davidson or CG is the v1 path.

### Implementation: minimal Davidson without Chebyshev preconditioning

A bare-bones Davidson using only existing kernels:
1. Take filtered ψ (or even unfiltered ψ_in) as the trial subspace.
2. Compute residuals `r_b = (H − λ_b·S)·ψ_b` per band using existing
   apply_full_hamiltonian + apply_s_times.
3. Mark bands with `‖r_b‖_S < 1e-6` as LOCKED.
4. Build H_sub, S_sub for unconverged bands only; ZHEGVD on this sub-block.
5. Update unconverged bands; locked bands stay fixed.
6. S-orthonormalize against locked bands (single-pass GS).
7. Iterate until all locked.

Skip the Chebyshev preconditioner for Phase 0 — use a diagonal kinetic-energy
preconditioner `P^{-1}_b[g] = (kinetic[g] − λ_b)^{-1}` (Teter-Payne-Allan
form). This is ~50 lines of CUDA kernel + Rust wrapper.

The Phase 0 implementation is a **scratch test** — does not need to be
production-quality. It exists to answer the gate question; if the gate
passes, the production-quality version follows in Phase 1.

### Gate 3 (THE decisive test)

Load CASTEP ψ from `Cu111_CO.check`. Run minimal Davidson with ndeg=0 (no
Chebyshev filter; locking + diagonal-precond Davidson only) starting from
CASTEP ψ.

| Outcome | Decision | Confidence |
|---------|----------|------------|
| Cu-3d block sum = 13.0 ± 1e-6 (ratio 1.0) | **Locking IS sufficient** → proceed to Davidson v1 (Phase 1A) | High |
| Cu-3d block sum stays at 11.6 (ratio 0.893) | Locking alone insufficient → fall back to block CG (Phase 1B) | High |
| Cu-3d block sum lifts but doesn't reach 1.0 (e.g., 12.5) | Mixed signal — locking helps but isn't complete. Prefer Davidson; re-evaluate during Phase 2 if needed. | Medium |

This gate uses the existing `subspace_projector_iter1_vs_castep` diagnostic
and the per-band-S-norm test — both already passing infrastructure.

### Phase 0 budget

- Day 1: write minimal Davidson + diagonal preconditioner. ~150 lines.
- Day 2: run Gate 3. Read result. Decide algorithm. Document.

If neither path is decisively chosen by end of Day 2, escalate to user.

## Phase 0.5 — HubbardPolicy trait scaffolding (3-5 days, REQUIRED)

This phase establishes the type system for DFT+U **before** the eigensolver
implementation lands, so DFT+U is a compile-time-checked policy from day one
rather than a retrofit.

### Trait design

```rust
/// Compile-time policy: does this calculation include DFT+U Hubbard correction?
pub trait HubbardPolicy: 'static + Clone {
    /// Per-site density matrix container; `()` for NoHubbard.
    type DensityMatrices: MixableMatrix;
    /// Augmented density combining ρ and {n_I}; `Density` for NoHubbard.
    type AugmentedDensity: MixableDensity;
    type ConvergenceTol;  // `()` for NoHubbard, `f64` for WithHubbard

    fn n_correlated_sites(&self) -> usize;
    fn evaluate_density_matrices(&self, ψ: &WavefunctionSet<RowDistributed>,
                                  occ: &[f64]) -> Self::DensityMatrices;
    fn v_u_apply(&self, ψ: &WavefunctionSet<ColumnDistributed>,
                 dm: &Self::DensityMatrices) -> Result<HpsiContribution>;
    fn u_term_energy(&self, dm: &Self::DensityMatrices) -> f64;
}

pub struct NoHubbard;
pub struct WithHubbard {
    /// Per-species U, J parameters (runtime config from .param)
    pub params: Vec<HubbardParameters>,
    /// Pre-computed local atomic projectors |φ_Im⟩ on GPU (analogous to β)
    pub projectors: HubbardProjectorData,
}
```

`ScfIteration` becomes `ScfIteration<S: SpinPolicy, H: HubbardPolicy = NoHubbard, ...>`.
The default-type-parameter pattern means existing call sites do not change.

### Key type-safety guarantees

1. **Per-site, per-l angular-momentum dimension** locked at compile time:
   ```rust
   pub struct DensityMatrix<L: AngularMomentum, Spin: SpinChannel> {
       n: nalgebra::Matrix<Complex<f64>, L::Dim, L::Dim>,
       ion_idx: usize,
   }
   pub trait AngularMomentum { const L: u32; type Dim: nalgebra::Dim; }
   pub struct DOrbital;
   impl AngularMomentum for DOrbital { const L: u32 = 2; type Dim = U5; }
   ```
   Cannot multiply a 5×5 d-orbital matrix by a 7×7 f-orbital projector.

2. **Staleness phantom on density matrices**:
   ```rust
   pub struct DensityMatrices<Status, ...> { ... }
   impl<...> DensityMatrices<Stale, ...> {
       pub fn refresh_from(...) -> DensityMatrices<Fresh, ...> { ... }
   }
   impl HubbardPotential<Fresh> {
       pub fn from_density_matrices(dm: &DensityMatrices<Fresh, ...>) -> Self
   }
   ```
   V_U construction refuses `Stale`. Compiler ensures `n_I` is freshly
   computed before V_U is built each SCF iteration.

3. **`MixableDensity` trait for augmented mixing**:
   ```rust
   pub trait MixableDensity {
       fn flatten(&self) -> Vec<f64>;
       fn reconstruct(flat: &[f64]) -> Self;
   }
   impl MixableDensity for Density { ... }
   impl MixableDensity for AugmentedDensity { ... }  // ρ + {n_I} combined
   ```
   Pulay/DIIS mixing operates on `AugmentedDensity` for DFT+U; the mixer
   doesn't branch on policy.

4. **`HamiltonianApply` composition**:
   ```rust
   pub trait HamiltonianApply<S, H> {
       fn apply_full(&self, ψ) -> Result<Hpsi>;  // composes T+V_loc+V_NL+V_U
   }
   ```
   The eigensolver calls `hamiltonian.apply_full(ψ)`. It does not know whether
   V_U is in there; the trait impl arranges for V_U iff `H: WithHubbard`.

### Phase 0.5 deliverables

- `src/hubbard/mod.rs` (new module): traits, types, `NoHubbard`/`WithHubbard` markers
- `src/scf.rs`: extend `ScfIteration` generic parameters; default to NoHubbard
- All existing tests compile unchanged (default-parameter pattern)
- No DFT+U functionality yet — that's Phase 4
- ~400 lines of trait definitions + plumbing

## Phase 1A — Davidson v1 (chosen if Gate 3 passes, ~1 week)

Production Davidson based on the other agent's plan
(`~/.claude/plans/we-seems-to-have-peppy-patterson.md`):

- Block partitioning by eigenvalue spacing (eps_degen = 0.01 Ha default)
- Chebyshev preconditioner (k=3-8) using EXISTING `chebyshev_filter` machinery
- Locking criterion: `‖r_b‖_S < tol` AND `|Δλ_b| < tol`
- Subspace management: restart at `2 × n_active_bands`
- `EigensolverMethod` enum dispatch in `diagonalize_inner`
- Trait composition with `HubbardPolicy`

Files affected (per the other agent's plan, with HubbardPolicy threading):
- `src/eigensolver/davidson.rs` (new, ~700 lines)
- `src/eigensolver/rayleigh_ritz.rs` (+60 lines for `subspace_eigenvalues_only`)
- `src/eigensolver/chebyshev.rs` (+70 lines for diagonal preconditioner)
- `src/scf.rs` (~+140 lines for dispatch + EigensolverMethod field)

**Critical reuse**: 100% of `chebyshev_filter`, `apply_full_hamiltonian`,
`apply_s_times`, `apply_s_inverse`, Gram-Schmidt, `VnlBatchData` —
preserved as load-bearing infrastructure.

## Phase 1B — Block CG v1 (chosen if Gate 3 fails, ~1 week)

If locking-via-Davidson is insufficient (cascade still occurs even with
locked bands), fall back to band-by-band CG batched across bands. The CG
plan is in `notes/plans/phase-block-cg-migration/PHASE_PLAN.md` (the
predecessor draft of this document).

Phase 1B retains the `HubbardPolicy` infrastructure from Phase 0.5; the
trait composition is identical (CG also calls `hamiltonian.apply_full`).

## Phase 2 — Validation (~3-5 days)

Goal: re-establish CASTEP-precision convergence on Cu111+CO using the new
eigensolver (Davidson or CG, whichever Phase 1 chose).

Tests to reactivate / rewrite:
- `iter1_drift_from_castep_state_is_bounded` (Q1) at 1 mHa
- `scf_converges_to_castep_energy_at_castep_tolerance` (Q2) at 1e-5 eV
- `subspace_projector_iter1_vs_castep` Cu-3d ratio > 0.999
- `overlap_iter2_against_castep` avg > 0.99
- `cascade_iter3_diagnostic_tight` at the original 0.1 Ha gate
- All RR validation tests (repurpose for diagnostic post-eigensolver basis polish)

If any fail at original tolerances after Phase 1, the issue is implementation,
not algorithm. Debug per usual.

## Phase 3 — GPU performance tuning (~3-5 days)

For Davidson: profile against Chebyshev-RR baseline. Optimize block detection
overhead, S-orthonormalization (fuse 2-pass GS), Chebyshev preconditioner
inner-iteration cost.

For CG (fallback): same kernels, fewer to optimize. Likely shorter phase.

## Phase 4 — DFT+U implementation (~1-2 weeks)

Implements `WithHubbard` policy. Adds:

- Local atomic projectors `|φ_Im⟩` storage in `HubbardProjectorData`
  (sibling of `VnlBatchData`). Precomputed once per geometry, GPU-resident,
  analogous to β projectors.
- Per-site density-matrix evaluation:
  `n_I^σ_mm' = Σ_n,k f_{nk}^σ ⟨ψ_{nk}^σ | φ_{Im}⟩ ⟨φ_{Im'} | ψ_{nk}^σ⟩`
- V_U apply (composed into `HamiltonianApply::apply_full` via trait impl)
- U-term energy contribution to total energy
- `AugmentedDensity` mixing through `MixableDensity` trait
- DFT+U fixtures from CASTEP for testing (Cu+U or NiO single-point reference)

DFT+U-specific risks:
- Metastable solutions (different magnetic/orbital orderings) — defer
  occupation matrix control to v1.1
- Convergence is harder; may need DIIS history depth > current default
- Works with both Davidson and CG eigensolvers (HubbardPolicy is composition,
  not algorithm-specific)

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Phase 0 Gate 3 ambiguous (block sum 0.93-0.97) | MEDIUM | Pre-decide: any ratio < 0.97 → fall back to CG; ≥ 0.97 → proceed Davidson. Document decision before running. |
| Davidson locking has implementation edge cases | MEDIUM | Phase 1A v1 ships without optimization; correctness via existing diagnostic suite |
| Davidson convergence harder for metals than insulators | MEDIUM | Use Chebyshev preconditioner (the other agent's plan); CASTEP/VASP precedent for similar designs |
| HubbardPolicy trait surface too complex | LOW | Phase 0.5 uses default type parameter; no existing call sites change |
| DFT+U metastable solutions block convergence | MEDIUM (Phase 4) | Defer to v1.1; document workarounds with .param-level guidance |
| CASTEP-GPU-port reference unavailable | KNOWN (per user 2026-05-24) | Derive from CASTEP CPU + Payne 1992 + Abinit GPU paper; rely on CASTEP CPU for reference values |

## What stays from `feat/phase-global-woodbury` branch

- All chemrust-hamiltonian work (V_NL, V_loc, β, Q, D)
- All density / V_eff / Hartree / V_xc / Ewald work
- All GPU plumbing, fixture loaders, CASTEP comparison anchors
- All forensic notes in `notes/debug/*` (institutional memory)
- Diagnostic-test infrastructure (`subspace_projector_iter1_vs_castep`,
  `diagnostic_per_band_s_norm_of_our_output`,
  `diagnostic_selftest_castep_self_overlap_block_sums`,
  `cascade_with_castep_anchored_postrr_pin`) — all algorithm-agnostic;
  reusable as Davidson/CG validation
- **If Davidson chosen**: 100% of `chebyshev_filter` (preconditioner)
- **If CG chosen**: ~600 lines of `chebyshev_filter` removed; the rest
  reused

## What's removed regardless

- `PinMode` enum + `RrPinConfig` + PostRr scaffolding (~150 lines of
  `rayleigh_ritz.rs`)
- `CHEMRUST_PIN_MODE`, `CHEMRUST_BLOW_PAD` env-var infrastructure
- Stashes `stash@{0}`, `stash@{1}`

## When to begin

After user explicit go-ahead. Phase 0 is the explicit derisk gate before any
multi-week work. Phase 0.5 (HubbardPolicy scaffolding) can run in parallel
with Phase 1 once Phase 0 has decided the algorithm.

Total estimated calendar time: ~4-6 weeks for v1 (eigensolver + plain DFT
at CASTEP precision) + ~1-2 weeks for v1.1 (DFT+U).

## Reading list before Phase 0 starts

1. The other agent's plan: `~/.claude/plans/we-seems-to-have-peppy-patterson.md`
   (Davidson + Chebyshev preconditioner architecture)
2. `notes/debug/debug-20260524-blow-tightening/RESOLUTION.md` —
   why Chebyshev-RR is wrong; specifically the per-band S-norm = 1.0 test
   that localizes the loss to off-block rotation
3. `notes/debug/debug-20260524-blow-tightening/CASTEP_ANCHORED_PIN_PROBE.md` —
   CASTEP-anchored upper-bound test showing rotation IS the cascade driver
4. Saad (2003) "Iterative Methods for Sparse Linear Systems" — Davidson
   chapter
5. Payne et al. (1992) RMP 64:1045 — band-by-band CG for the fallback path
6. Abinit GPU paper at `~/programming/CASTEP-GPU-port/2604.11139v1` —
   block-CG GPU implementation details
7. CASTEP DFT+U: `~/Downloads/CASTEP-6.11-nixos/Source/Functional/electronic.f90`
   for `dft+u` keyword handling and the specific U projection convention

## Out of scope for v1 + v1.1

- LOBPCG (preconditioner research-grade for USPP metals)
- Direct minimization (CDM/OD; insulator-only)
- Replacing density mixing
- Tuning chemrust-hamiltonian (already validated)
- Self-consistent U (linear-response Cococcioni method) — defer to v2
- DFT+U+V (inter-site Hubbard) — defer to v2
- Non-collinear / spin-orbit DFT+U — defer to v2
