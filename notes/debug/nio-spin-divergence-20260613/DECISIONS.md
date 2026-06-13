# DECISIONS: NiO Spin-Polarised Cold-Start Divergence at Iter 16

**Date**: 2026-06-13 / **corrected 2026-06-14**
**Context**: NiO spin-polarised FFI cold-start diverges at SCF iteration 16 (commit `69b3bc4`), while non-spin converges in 10 iterations and warm-start passes all discriminator criteria. This document synthesizes findings from 8 deep-dive agents (profile, prep-H, ham-diag, checklist-gap, occupation-audit, divergence-surface, diagnostic-design, abort-design) into actionable decisions.

**Input reports**: profile-deep-dive, prep-h-deep-dive, ham-diag-deep-dive, checklist-gap, occupation-audit, divergence-surface, diagnostic-design, abort-design (2026-06-13 phase-7-castep-audit workflow).

---

## 0. Post-Mortem Correction (2026-06-14)

The original analysis in Sections 1-2 below attributed the FFI cold-start divergence
to spin density mixing amplitude mismatch and wrong spin_fix value. **Both of these
are standalone Rust SCF path issues — they do NOT affect the FFI path.** In the FFI
path (`chemrust_eigensolve_step`), CASTEP handles all mixing, occupation search, and
density construction on the Fortran side. Rust only diagonalizes H[V_eff].

### What was actually fixed (standalone Rust SCF path)

| Fix | What changed | Commit |
|-----|-------------|--------|
| P0-DONE | Mixing amplitude: `cpx_full_update` kernel now applies per-spin `amp`; `DensityHistory` stores per-spin `mixing_amplitude`; spin-polarised systems default to 2.0 (matching CASTEP `spin_density_mixing_amplitude`) | `mixing/cuda_kernels.rs`, `mixing.rs`, `scf.rs` |
| P1-DONE | `spin_fix`: corrected 1-based→0-based off-by-one in `scf.rs` (`>= spin_fix - 1`); test value changed from 10 to 6 (matching NiO `.param`) | `scf.rs`, `nio_spin_scf.rs`, `types.rs` |

### What remains: FFI cold-start divergence at iter 16

The FFI path divergence is unaffected by these fixes. The non-spin-converges/spin-diverges
gap must be in the per-spin diagonalization path itself. The key differential between
spin and non-spin in the FFI step is the **D-screening**:

- For non-spin: `rescreen_d` called once with V_eff_total → one set of D-matrices
- For spin: `rescreen_d` called twice with V_eff_up and V_eff_dn → two sets of D-matrices
- CASTEP profile: `nlpot_calculate_d` = 66 calls (per-SCF, NOT per-spin) — shared D-matrices

This is the primary investigative target for the actual FFI divergence.

### What was wrong with the original root cause analysis

| Original claim | Why wrong |
|---------------|-----------|
| "70% probability: spin mixing amplitude mismatch" | Mixing is CASTEP's responsibility in FFI path. Rust never calls `mix()` from `chemrust_eigensolve_step`. |
| "15% probability: wrong spin_fix value" | `spin_fix` controls occupation search, which CASTEP handles (not Rust) in the FFI path. |
| "10% probability: Davidson compaction index bug" | Spin-agnostic — would affect non-spin equally. Non-spin converges. No mechanism to produce spin-specific divergence. |
| "D-screening per-spin introduces fake spin-dependence" | **This is the one plausible FFI-path finding that wasn't pursued.** The profile shows CASTEP calls `nlpot_calculate_d` 66 times (per-SCF, shared across spins), while Rust calls `rescreen_d` per-spin. |

### Corrected fix priority

| Priority | ID | Fix | Path | Status |
|----------|----|-----|------|--------|
| **P0** | C8-S3 (mixing amplitude) | `cpx_full_update` + `amp` parameter; spin-polarised default 2.0 | Standalone Rust SCF | **DONE** |
| **P1** | C10-S6 (spin_fix) | Correct 1-based→0-based off-by-one; set to 6 matching NiO param | Standalone Rust SCF | **DONE** |
| **P2** | FFI D-screening | Investigate per-spin vs per-SCF D-matrix screening discrepancy | FFI path | **TO DO** |

---

## 1. Root Cause Hypothesis (original, partially incorrect — see Section 0)

### Primary Candidate: Per-Spin Density Mixing Amplitude Mismatch (C8-S3 / Gap G1)

**What**: Rust mixes spin channels (rho_up, rho_dn) with the same mixing amplitude as charge density (0.5), while CASTEP uses a **separate `spin_density_mixing_amplitude=2.0`** for spin channels.

**Evidence**:
- **CASTEP source**: NiO `.castep` line 109508-109512 shows `charge density mixing amplitude: 0.5000` and `spin density mixing amplitude: 2.000`. These are distinct parameters passed to `dm_mix_density` separately.
- **Rust code**: `src/scf.rs:1606-1613` — per-spin mixing loop `for ispin in 0..nspins { self.history.mix(self.density[ispin].clone()) }` uses the same `DensityHistory` instance for both spin channels, with the same Pulay parameters (amplitude, g-vector, history depth). No per-spin amplitude override exists.
- **Profile**: NiO CPU profile shows CASTEP converges in 67 SCF iterations with `spin_fix=5` at runtime. The spin density mixing amplitude of 2.0 (4x the charge amplitude of 0.5) means CASTEP's spin density evolves 4x faster than charge density during early iterations.
- **Divergence signature**: Divergence at iter 16, not iter 2 or 3. This is a **slow accumulation** pattern, consistent with a mixing parameter mismatch (each iteration's spin density is slightly farther from self-consistent, accumulating over ~15 iterations until the V_eff_up vs V_eff_dn separation becomes unphysical).

**Causal chain**: 
1. Iter 0: paramagnetic guess (zero spin) → correct for both Rust and CASTEP.
2. Iter 1-5: spin emerges from density construction. Rust mixes spin density with amplitude 0.5, CASTEP mixes with amplitude 2.0. Rust's spin density evolves 4x slower.
3. Iter 6-10: CASTEP switches to `fermi_free` (shared Fermi energy). Rust may still be in `fermi_fix` if `spin_fix` value is wrong (see secondary candidate below). Compound effect: wrong mixing + wrong occupations.
4. Iter 11-16: accumulated spin density error becomes large enough that V_eff_up and V_eff_dn no longer produce a physically meaningful exchange splitting. "No empty bands" warning appears (observed in prior divergence patterns — failure-patterns.md line 14-15: "no empty bands warning → runaway energy gains").
5. Iter 16+: eigenvalues explode, SCF diverges catastrophically.

**Estimated probability**: **70%** this is the primary cause. C8-S3 is the most direct mismatch between Rust and CASTEP that would produce exactly the observed slow-accumulation-then-divergence pattern.

### Secondary Candidate: Wrong `spin_fix` Value (C10-S6 / Gap G2)

**What**: Rust `SmearingParams.spin_fix` defaults to 10, while CASTEP NiO runtime value is 5 (per profile: 5 calls to `electronic_find_fermi_fix`).

**Evidence**:
- **CASTEP profile**: `NiO.0001.profile` shows 5 calls to `electronic_find_fermi_fix` and 62 to `electronic_find_fermi_free` — total 67 occupation searches, 5 fixed-spin → `spin_fix` = 5 at runtime.
- **CASTEP source**: `electronic.f90:294-296`: `if(scf_cycle >= spin_fix.and.spin_polarised...) spin_freed = .true.` — for `spin_fix=5`, iterations 5+ use free spin. Also `electronic.f90:516-518`: `if(scf_cycle == spin_fix) call fermi_fix` — the spin_fix iteration itself still calls fermi_fix.
- **Rust code**: `scf.rs:194` — `state.spin_freed = scf_iter > smearing.spin_fix`. For `spin_fix=10`, iterations 1-10 use `fermi_fix`, iter 11+ use `fermi_free`. For `spin_fix=5` (correct), iter 1-5 use fix, iter 6+ use free.
- **Impact**: For iterations 6-10, Rust uses `fermi_fix` (per-spin independent Fermi levels, fixed spin populations) while CASTEP uses `fermi_free` (shared Fermi level, variable spin populations). Wrong occupations → wrong density per spin → cascade amplification of G1.

**Causal chain**: The wrong `spin_fix` value alone would NOT cause divergence at iter 16 (it would cause wrong occupations starting at iter 6, but would not produce a monotonically growing error). Combined with G1 (wrong mixing amplitude), the compound effect accelerates divergence: wrong spin density + wrong occupations = doubly wrong density for V_eff construction.

**Estimated probability**: **15%** this is a contributing factor. Fixing G1 alone may still diverge if `spin_fix=10` keeps wrong occupations for iterations 6-10, but the divergence would likely be slower (iter 20+ instead of 16).

### Tertiary Candidate: Davidson Compaction Index Bug (INNER_LOOP_CHECKLIST C1-D3 / Memory)

**What**: After inner-loop compaction removes converged bands, `previous_eigenvalues` is overwritten with sequential indices that no longer match compacted column positions (davidson-compaction-index-bug.md).

**Evidence**:
- **Memory**: `davidson-compaction-index-bug.md` — MUST-FIX. "previous_eigenvalues[b] = eigenvalues[block_start + b] using sequential indices that no longer match compacted column positions."
- **Manifestation**: The bug corrupts eigenvalue index mapping for convergence checks in subsequent inner iterations. For warm-start (converged psi, small residuals, few inner iterations), compaction rarely removes bands → bug rarely triggers. For cold-start (random psi, large residuals, many inner iterations), compaction is frequent → index corruption compounds across blocks and outer iterations.
- **NiO specificity**: The bug affects all spin channels equally. Why non-spin converges but spin diverges: the spin-polarised case has twice as many eigensolves per SCF iteration (one per spin), each with different eigenvalue distributions → twice the opportunity for compaction to trigger the bug → faster accumulation of corrupted index mapping.

**Estimated probability**: **10%** this is a contributing factor. The bug would cause slower convergence and occasional eigenvalue jumps but is unlikely to be the sole cause of monotonic divergence at iter 16. Fixing G1 + G2 would likely mask this bug (fewer inner iterations needed when V_eff is correct → less compaction → bug rarely triggers).

### Risk Tier: What Else Could Be Wrong

| Risk | Description | Likelihood | Impact if True |
|------|-------------|-----------|----------------|
| R1 | Per-iteration per-spin V_eff assembly accumulation error — small errors in V_eff_up vs V_eff_dn at each SCF iter compound across 16 iters | 5% | High: would require debugging the entire V_eff assembly pipeline for spin-specific errors |
| R2 | Occupation search numerical instability for cold-start eigenvalues (eigenvalues span [-5, +5] Ha during early iterations vs [-1, +0.2] Ha for converged) | 3% | Medium: bisection search correct but occupancy formula hyper-sensitive to extreme eigenvalues |
| R3 | D-screening fine-grid cache contamination between spin channels (ffI.rs uses per-spin cache but `rescreen_d` may reuse data across spins within same SCF iter) | 3% | High: wrong D-matrices → wrong V_NL → wrong eigenvalues for both spin channels |
| R4 | Multi-kpt wavefunction layout transposition specific to spin-polarised (14 kpts x 2 spins = 28 per-iteration eigensolves, any layout bug compounds 28x) | 2% | High: if kpt ordering differs between spins in some data structure |
| R5 | `E_nonCoulomb` hardcoded constant differs from actual PP-derived value across SCF iterations (PP local part energy depends on density via augmentation) | 1% | Low: only affects energy convergence criterion, not eigenvalue correctness |
| R6 | Per-spin Pulay history shared across spin channels — density history stores mixed rho_up and rho_dn in the same `DensityHistory` instance, potentially cross-contaminating | 2% | Medium: Pulay history stores previous iterations' densities; if spin channels share history, the mixing weight computation for rho_up uses rho_dn history entries |

---

## 2. Decision: What to Fix (updated 2026-06-14)

### Fix Status

| Priority | ID | Fix | Path | Status |
|----------|----|-----|------|--------|
| **P0** | C8-S3 | Spin mixing amplitude: `cpx_full_update` kernel + `amp` parameter; `DensityHistory::with_amplitude()`; spin-polarised default 2.0 matching CASTEP `spin_density_mixing_amplitude` | Standalone Rust SCF | **DONE** |
| **P1** | C10-S6 | `spin_fix`: corrected 1-based→0-based off-by-one (`>= spin_fix - 1`); defaults and test set to 6 matching NiO `.param` | Standalone Rust SCF | **DONE** |
| **P2** | FFI D-screening | Investigate per-spin `rescreen_d` vs CASTEP's per-SCF `nlpot_calculate_d` (66 calls, shared across spins) | FFI path | **INVESTIGATING** (wf_7bcf30e3-589) |
| ~~P3~~ | INNER_LOOP | ~~Davidson compaction index bug~~ — spin-agnostic; non-spin converges, so this cannot explain spin-specific divergence. User requested iterator-based fix separately. | Both | **DEFERRED** (not relevant to this divergence)

### Do NOT Fix First (reasons below)

| Deferred | ID | Reason |
|----------|----|--------|
| D-first | G6 (diagnostics) | Diagnostics MUST be implemented BEFORE applying fixes to confirm the fix changes divergence behavior as predicted. But the fix code and diagnostic code can be developed in parallel. |
| D-first | G5 (E_nonCoulomb) | Only affects energy convergence criterion, not eigenvalue correctness. Not a cause of eigenvalue divergence. |
| D-first | D3 (per-kpt OccupationSet) | Known simplification. Does not affect SCF convergence — the correct multi-kpt weighted occupations are used in density construction. Only the stored `OccupationSet` loses per-kpt info. |

---

## 3. Decision: What Diagnostic Tests to Implement

### Diagnostic-1: Per-Iteration Spin Density Tracker (P1-1)

**Purpose**: Track spin density evolution across cold-start SCF iterations. Compare Rust spin density trajectory against CASTEP's trajectory.

**Implementation**:
- Feature-gated (`scf_diag`) diagnostic in `src/scf.rs`
- At each SCF iteration, after `construct_density_off`, compute and print:
  - `net_spin = Σ(ρ_up - ρ_down) / N_grid` (integrated spin)
  - `max_spin = max(|ρ_spin[i]|) / max(|ρ_total[i]|)` (spin polarization ratio)
  - `rms_spin_change = RMS(current_spin_density - previous_spin_density)`
  - `n_up = Σ occ_up`, `n_dn = Σ occ_dn` (electron counts per spin)
- Dump to stderr with `[SpinTrack iter=N]` prefix for grep-ability

**Discriminator value**: If the spin density magnitude at iter 10 of Rust is 4x smaller than CASTEP's, G1 (wrong mixing amplitude) is confirmed. If spin density magnitude matches but electron counts are wrong at iters 6-10, G2 (wrong spin_fix) is confirmed.

### Diagnostic-2: Cold-Start Divergence Gate Test (P1-2)

**Purpose**: Automated divergence-detection test for cold-start NiO SCF. Runs 40-50 iterations — enough to observe the divergence pattern (which CASTEP reference shows converges in 67). Does NOT assert convergence; only asserts eigenvalues stay bounded.

**Implementation**:
- New test: `tests/nio_spin_scf.rs::nio_cold_start_divergence_gate`
- Start from paramagnetic guess (uniform density, random wavefunctions)
- Run **40-50** SCF iterations via `run_scf_with_energy::<SpinCollinear>` (CASTEP CPU reference takes 67 iterations to converge — 20 is insufficient to observe anything meaningful)
- At each iteration, assert:
  - `max(|eigenvalue|) < 10 Ha` (no explosion)
  - `|n_up + n_down - N_total| < 1e-3` (no electron loss)
  - `|n_up - n_down - expected_spin| < 5` (no spin flip catastrophe)
  - Track but do NOT assert: energy, spin density, Fermi level — these should TREND toward reference values but may not converge in 50 iterations
- If any assertion fails, print full state dump for that iteration and abort

**Acceptance criterion**: Pass = eigenvalues remain bounded and no "no empty bands" warning for 40+ iterations. Do NOT assert convergence within N iterations. The purpose is divergence detection, not convergence speed benchmarking.

### Diagnostic-3: Per-Iteration Occupation Audit (P1-1 extension)

**Purpose**: Verify the `spin_freed` transition produces correct occupations.

**Implementation**:
- In `src/density.rs`, add `scf_diag` print at each `compute_occupations_weighted` call:
  - `spin_freed` flag value
  - `n_up_expected, n_dn_expected` (from density-integrated spin)
  - `n_up_actual, n_dn_actual` (from occupation sums)
  - `E_F_up, E_F_dn` (Fermi energies, shared or per-spin)
  - Number of partially occupied bands per spin
- Assert `|n_up_expected - n_up_actual| < 1e-6` and `|n_dn_expected - n_dn_actual| < 1e-6` (occupation search must find correct electron counts)

### Diagnostic-4: Per-Block Davidson Statistics (extends existing diagnostics)

**Purpose**: Track whether the Davidson compaction index bug manifests in cold-start.

**Implementation**:
- In `src/eigensolver/davidson.rs`, add `scf_diag` print after each block's inner loop:
  - Block index, spin, kpt
  - Number of inner iterations
  - Number of bands compacted (bands removed mid-block)
  - Number of bands converged at end of outer iteration
  - Max eigenvalue change for ANY band being tracked (not just compacted active bands — INNER_LOOP_CHECKLIST C14-03 fix)

**Discriminator value**: If `previous_eigenvalues` corruption manifests, the "max eigenvalue change" metric will show anomalous jumps (>0.01 Ha) for bands that were not in the compacted active set.

---

## 4. Decision: What Abort Mechanism to Use

### Abort Trigger: Eigenvalue Explosion Detector

Based on the failure pattern observed across multiple divergence sessions (failure-patterns.md: "|hpsi|^2 after V_loc hits 10^6->10^63->10^74->10^86->10^108 -> ZHEGVD fails", and "no empty bands warning -> runaway energy gains"):

**Abort condition** (inside `src/eigensolver/davidson.rs`, after each inner iteration):
1. **Immediate abort**: If `max(|H·ψ|^2) > 1e10` for any band after V_loc multiplication — indicates the H·ψ computation has entered an unrecoverable numerical regime. This catches the ZPOTRF/ZHEGVD failure cascade BEFORE it floods the GPU with NaN/Inf.
2. **Warning threshold**: If `max(|eigenvalue|) > 5 Ha` for any band (physical bound: NiO eigenvalues span [-1.0, +0.2] Ha when correct). Log warning with spin, kpt, band index, and eigenvalue value.
3. **SCF-level abort**: If any SCF iteration produces eigenvalues with `max(|ε|) > 100 Ha` — the SCF driver at `src/scf.rs` should abort and return an error containing the current state for post-mortem analysis.

**CASTEP comparison**: CASTEP `hamiltonian.f90` does not have an explicit eigenvalue explosion detector — its band-by-band CG minimizer is inherently more stable against eigenvalue explosion because each band is converged independently. Our subspace Rayleigh-Ritz method is more susceptible to explosion because a single bad conduction state can corrupt ZHEEVD for an entire block.

### Abort Implementation Location

In `src/eigensolver/davidson.rs`, after `apply_full_hamiltonian()` (Stage 6), add:

```rust
// Abort check: H·psi norm sanity
let hpsi_norm2 = compute_column_norms_squared(hsearch_dev, current_nblock, n_pw, stream);
for b in 0..current_nblock {
    if hpsi_norm2[b] > 1e10 {
        return Err(DavidsonError::Explosion {
            block_start,
            spin,
            kpt,
            band: b,
            hpsi_norm2: hpsi_norm2[b],
            eigenvalue: eigenvalues[block_start + b],
        });
    }
}
```

In `src/scf.rs`, wrap the diagonalize call in a per-iteration eigenvalue bound check:

```rust
// SCF-level abort: eigenvalue bounds
if let Some(max_eig) = eigenvalues.iter().flat_map(|v| v.iter()).cloned().reduce(f64::max) {
    if max_eig > 100.0 {
        return Err(ScfError::EigenvalueExplosion {
            scf_iter,
            max_eigenvalue_ha: max_eig,
        });
    }
}
```

### Error Type Design

```rust
#[derive(Debug, thiserror::Error)]
pub enum DavidsonError {
    // ... existing variants ...
    #[error("H·psi norm explosion: block_start={block_start} spin={spin} kpt={kpt} band={band} |Hψ|^2={hpsi_norm2:.2e} ε={eigenvalue:.6f}")]
    Explosion {
        block_start: usize,
        spin: usize,
        kpt: usize,
        band: usize,
        hpsi_norm2: f64,
        eigenvalue: f64,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ScfError {
    // ... existing variants ...
    #[error("SCF eigenvalue explosion at iter {scf_iter}: max |ε| = {max_eigenvalue_ha:.2f} Ha")]
    EigenvalueExplosion {
        scf_iter: usize,
        max_eigenvalue_ha: f64,
    },
}
```

---

## 5. Decision: What Checklist Items Need Re-Verification

These items are currently marked WARM-START-VERIFIED. After implementing P0 and P1 fixes, they must be re-verified against the cold-start discriminator test (Diagnostic-2) to upgrade to FIXED status:

### Must Re-Verify After P0 (spin mixing amplitude)

| ID | Component | What to Re-Verify | Acceptance Criterion |
|----|-----------|-------------------|---------------------|
| C8-S3 | Per-spin density mixing | Mixing applies `spin_density_mixing_amplitude=2.0` to up/down channels | V12: Spin density trajectory matches CASTEP (RMS residual < 0.1 e-/Bohr^3) after spin_freed transition |
| C6-S1 | Per-spin density construction | Density from wavefunctions with kpt weights produces correct spin density at each iter | V13: Occupation sums N_up+N_dn = N_total at every cold-start iter |
| C5-S1 | BuildVEffWithEnergy | V_eff_up/V_eff_dn from cold-start densities maintain physically meaningful spin splitting | V11: Eigenvalues bounded within [-10, +5] Ha at all cold-start iters |

### Must Re-Verify After P1 (spin_fix value)

| ID | Component | What to Re-Verify | Acceptance Criterion |
|----|-----------|-------------------|---------------------|
| C10-S6 | spin_freed transition | Transition from fermi_fix to fermi_free occurs at scf_iter > 5 | V14: Exactly 5 iterations use fermi_fix, remainder use fermi_free |
| C10-S1 | Occupation search strategy | fermi_free produces correct shared E_F with variable spin population | V7: net_spin from occupations matches integrated spin density |

### Must Re-Verify After P2 (Davidson compaction fix)

| ID | Component | What to Re-Verify | Acceptance Criterion |
|----|-----------|-------------------|---------------------|
| INNER C1-D3 | slice_eigenvalues consistency | previous_eigenvalues not overwritten after compaction | Diagnostic-4: No anomalous eigenvalue jumps (>0.01 Ha) for non-active bands |
| INNER C14-03 | previous_eigenvalues scope | All bands' previous eigenvalues saved, not just active | Convergence check uses correct baseline |

### Already Verified (No Re-Verification Needed)

| ID | Reason |
|----|--------|
| C1-S1 through C7 | Field type correctness verified by compiler. SpinChannelData<T> newtypes cannot silently coerce. |
| C2-S1 | into_phase() field copy verified by compiler. |
| C3-S1 through S5 | Spin+kpt loop structure verified against CASTEP profile (134 calls). |
| C11, C12 | FFI parameter wiring verified by CASTEP build + single-call warm-start. `cache_reuse=false` eliminates the only multi-iteration FFI concern. |
| C4 | Paramagnetic guess only used at iter-0. After iter-1, density has spin, C5 path exercised. |

---

## 6. Risk Assessment: What Else Could Be Wrong

### Cross-Cutting Risks (affect multiple checklist components)

| Risk | Components Affected | Manifestation | Detection Strategy |
|------|---------------------|---------------|-------------------|
| **Per-iteration V_eff assembly accumulation** | C5 (BuildVEffWithEnergy), C6 (density), C3 (diagonalize) | Small per-spin V_eff errors at each SCF iter compound: iter-N V_eff uses density from iter-(N-1) eigenvalues, which had small eigenvalue errors from iter-(N-2) V_eff errors, etc. | Diagnostic-2: compare eigenvalue RMS drift per iteration against CASTEP reference. If drift grows monotonically, V_eff assembly has per-iteration error accumulation independent of mixing. |
| **Occupation formula hyper-sensitivity to extreme eigenvalues** | C10 (occupation search) | During early cold-start iterations, eigenvalues span [-5, +5] Ha (wide range from random psi). The `erfc` smearing formula may produce non-physical occupations for eigenvalues far from E_F. | Diagnostic-3: check that Σ_occ = N_total at every iteration, even with eigenvalue range of ±5 Ha. If Σ_occ ≠ N_total, the bisection search is not handling wide eigenvalue ranges. |
| **D-screening cache contamination between spins** | C3-S4 (VNL per-spin), C11, C12 (FFI per-spin caches) | ffi.rs uses per-spin `vnl[isp]` cache. If `rescreen_d` internally reuses data across spine calls (e.g., shares fine-grid FFT buffers), spin-up D-matrices contaminate spin-down. | Diagnostic-1: if V_eff_up ≠ V_eff_dn but eigenvalues are identical between spins, D-screening is identical (should differ for spin-polarised V_eff). |
| **Multi-kpt layout transposition specific to spin** | C1 (ScfIteration fields), C3 (diagonalize) | 14 kpts x 2 spins = 28 per-iteration eigensolves. Any kpt ordering mismatch between spins (e.g., kpt_data for spin-up stored as [k0..k13] and spin-down as [k13..k0]) would silently map wrong wavefunctions to wrong k-points. | Diagnostic-1: at iter 0 (paramagnetic guess), eigenvalues for kpt_i should be identical between spin-up and spin-down. If they differ, kpt ordering is wrong. |
| **Pulay history cross-contamination** | C8 (mix), C7 (construct_density_pulay) | Per-spin density mixing currently uses the same `DensityHistory` instance for both channels. Pulay mixing stores density residual vectors from previous iterations. If both channels share the same history store, rho_up history could contain rho_dn residuals. | Diagnostic-1: track per-spin density residual norms separately. If `RMS(Δrho_up)` and `RMS(Δrho_dn)` are identical (should differ for spin-polarised), history is shared. |

### Mitigation for Each Risk

| Risk | Mitigation | When to Apply |
|------|-----------|---------------|
| V_eff accumulation | After P0+P1+P2 fixes, if divergence persists, implement per-iteration V_eff difference diagnostic: compare V_eff[N] from Rust's density[N] vs V_eff[N] from CASTEP's density[N] (requires CASTEP intermediate density dumps). | After P0-P2, if Diagnostic-2 still fails |
| Occupation hyper-sensitivity | Add eigenvalue clamping in bisection: `eig_clamped = eig.clamp(-10*smearing_width, +10*smearing_width)`. Prevents extreme eigenvalues from dominating occupation sums. | After P0-P2, if Diagnostic-3 shows Σ_occ ≠ N_total for early iterations |
| D-screening cache contamination | Per-spin FFT buffers: each spin gets its own fine-grid FFT workspace. Verify by computing |D_up - D_dn| at ion 0 (Ni) — must be non-zero for spin-polarised V_eff. | After P0-P2, if eigenvalues identical between spin channels |
| Multi-kpt layout | Add debug assertion: at iter 0 (paramagnetic), assert eigenvalues[0][kpt_i] ≈ eigenvalues[1][kpt_i] (within 1e-6 Ha). Paramagnetic V_eff is identical for both spins, so eigenvalues should match. | Before P0, as a pre-condition check |
| Pulay history contamination | Separate `DensityHistory` instances: one per spin channel. Minimal change — in `mix()`, index `self.history[ispin]` instead of sharing a single history. | Alongside P0 (spin mixing amplitude fix), since both touch the same `mix()` code path |

---

## Appendix A: Profile Evidence Summary

From NiO CPU spin-polarised profile (`NiO.0001.profile`, 67 SCF iterations):

| Metric | CASTEP Value | Rust Value | Divergence |
|--------|-------------|------------|------------|
| SCF iterations to converge | 67 | 16 (diverges before convergence) | N/A |
| `hamiltonian_diagonalise_ks` calls | 134 (67 x 2 spins) | TBD (diverges before completion) | N/A |
| `electronic_find_fermi_fix` calls | 5 | Depends on `spin_fix` (default 10 → 10 calls) | **MISMATCH** (C10-S6, G2) |
| `electronic_find_fermi_free` calls | 62 | Depends on `spin_fix` (default 10 → would be 6) | **MISMATCH** (C10-S6, G2) |
| `density_calculate_soft_wvfn` calls | 65 | TBD | Should match (one per SCF iter) |
| `dm_mix_density` calls | 67 | TBD | Should match |
| `spin_density_mixing_amplitude` | 2.0 | 0.5 (same as charge) | **MISMATCH** (C8-S3, G1) |
| `spin_density_mixing_g_vector` | 1.5 A^-1 | 1.5 A^-1 (same as charge) | Matches |

---

## Appendix B: Failure Pattern Correlation

The divergence at iter 16 matches the general pattern documented in `failure-patterns.md`:

1. **"SCF iter 1 looks reasonable, iter 2 shows eigenvalue drift, iter 3 triggers catastrophic divergence"** (failure-patterns.md line 11-14, V_eff cache staleness pattern).
   - Current divergence: drift accumulates over 15 iterations before catastrophic at iter 16. Slower accumulation because the error source (mixing amplitude) accumulates per-iteration rather than being triggered by a single stale-cache event.

2. **"no empty bands" warning** (failure-patterns.md line 14, cascade signature).
   - Observed in prior Cu111_CO divergence sessions. This warning means the occupation search found MORE occupied bands than physically possible for the given electron count — a sign that the eigenvalue spectrum has shifted unphysically (V_eff wrong → eigenvalues wrong → Fermi level wrong).

3. **"Warm-start test passes, cold-start diverges"** (failure-patterns.md beta-g-layout section, lines 47-52).
   - The warm-start test is insufficient for catching bugs that scale with residual magnitude. The spin mixing amplitude mismatch is exactly such a bug: at the converged state, mixing amplitude doesn't matter (density residual is ~1e-6). During cold-start, mixing amplitude determines the trajectory to convergence (density residual is ~1 Ha at iter 0 → correct mixing amplitude is essential).
