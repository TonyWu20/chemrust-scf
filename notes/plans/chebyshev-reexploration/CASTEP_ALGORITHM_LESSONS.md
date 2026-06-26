# Why CASTEP's Algorithm Succeeds — Consolidated Lessons

**Date:** 2026-06-15
**Status:** Research complete; consolidated understanding before any new implementation
**Source:** CASTEP hamiltonian.F90 (lines 48-680), Rust davidson.rs (3921 lines faithful port),
          10+ debug sessions (2026-05-20 → 2026-06-10)

---

## 1. CASTEP's Algorithm: The Two-Level Iterative Scheme

CASTEP's `hamiltonian_diagonalise_slice` (hamiltonian.F90:48-680) uses a nested
two-level loop. This is NOT the "band-by-band CG" I previously described in
memory `[[chebyshev_rr_architecturally_unsuitable]]` — that description was
wrong. CASTEP uses **subspace Davidson with block-grouped superspace**.

### Outer loop (lines 293-666)

```
outer_loop: do iteration = 1, max_iterations(2)
  if (all(band_converged) .and. prev_all_converged) exit  ← double-check guard

  ! 1. Recompute H·ψ from scratch — fresh operator application
  call hamiltonian_apply(eigenvectors, local_pot, nl_d, H_eigenvectors, eigenvalues, ...)

  ! 2. Save pre-rotation eigenvalues for convergence invalidation
  previous_full_eigenvalues(i) = eigenvalues(i)

  ! 3. Full subspace diagonalization (wave_diagonalise = ZHEGVD on n_bands×n_bands)
  call wave_diagonalise(eigenvectors, H_eigenvectors, eigenvalues)

  ! 4. Convergence invalidation: un-converge bands whose eigenvalues changed
  !    after fresh H·ψ + subspace rotation
  if (abs(previous_eigenvalues(i) - eigenvalues(i)) >= tol_abs) then
    band_converged(i) = .false.
  end if

  ! 5. Build preconditioner from mean kinetic energy
  mean_ek = sum(ek) / nbands
  call wave_calc_precon(...)
  call nlpot_prepare_precon(...)

  ! 6. Block loop: iterate over groups of nblock = floor(2*sqrt(n_bands)) bands
  band_loop: do nb = 1, nbands, nblock
    ...
  end do band_loop
end do outer_loop
```

### Block loop — inner Davidson with superspace (lines 361-662)

For each block of `nblock` bands:

```
! 1. Skip if all bands in this block are already converged
if (all(band_converged(nb:nb+current_nblock-1))) cycle

! 2. Copy block ψ and Hψ into superspace buffers
call wave_copy(eigenvectors, super_wvfn, ...)   ! first current_nblock columns
call wave_copy(H_eigenvectors, H_super_wvfn, ...)

! 3. Compute initial super_hamiltonian = super_wvfn^H · H_super_wvfn
call wave_dot_all(super_wvfn, H_super_wvfn, super_hamiltonian)

inner_loop: do elecmin_step = 1, max_iterations(1)

  ! 4. Build search direction: preconditioned steepest descent
  !    search = P⁻¹ · (H - εS) · ψ       [hamiltonian_searchspace → nlpot_apply_precon_ES]
  call hamiltonian_searchspace(...)

  ! 5. S-orthogonalize search against superspace
  call wave_Sorthogonalise_to_lower(super_wvfn, slice_searchspace, superspace_index)
  call wave_Sorthonormalise(slice_searchspace)

  ! 6. Apply H to search directions
  call hamiltonian_apply(slice_searchspace, ..., H_slice)

  ! 7. Extend superspace
  call wave_copy(slice_searchspace, super_wvfn, ...)
  call wave_copy(H_slice, H_super_wvfn, ...)

  ! 8. Update super_hamiltonian (Hermitian)
  call wave_dot_all(slice_searchspace, H_super_wvfn, new_rows)

  ! 9. Diagonalize superspace (ZHEEVD or DSYEVD for gamma-point)
  call algor_diagonalise(super_wvfn%nbands, rotation, super_eigvals, ...)

  ! 10. Rotate superspace wfn and H·wfn
  call wave_rotate(super_wvfn, rotation)
  call wave_rotate(H_super_wvfn, rotation)

  ! 11. S-orthogonalize against full eigenvectors
  call wave_Sorthogonalise_to_lower(eigenvectors, super_wvfn, nb)
  call wave_Sorthonormalise(super_wvfn)

  ! 12. Copy rotated results back to eigenvectors
  call wave_copy(super_wvfn, eigenvectors, ...)
  call wave_copy(H_super_wvfn, H_eigenvectors, ...)

  ! 13. Convergence check: per-band eigenvalue change
  do i = 1, current_nblock
    if (abs(previous_eigenvalues(i) - eigenvalues(nb+i-1)) < tol_abs) then
      band_converged(nb+i-1) = .true.
    end if
    ! Also check stagnation via relative criterion
    if (abs(ΔE) < break_cond_tol(i) * tol_rel) then
      opt_stop_condition(i) = .true.
    end if
    ! Upward eigenvalue change → definitely not converged
    if (previous_eigenvalues(i) - eigenvalues(nb+i-1) < -ε) then
      band_converged(nb+i-1) = .false.
    end if
  end do

  ! 14. Exit if all bands converged or stagnated
  ! 15. Otherwise, compact unconverged bands into slice workspace
  do i = 1, current_nblock
    if (.not. band_converged(nb+i-1) .and. .not. opt_stop_condition(i)) then
      call wave_copy(H_super_wvfn, H_slice, nb_src=i, nb_dst=j, ...)
      call wave_copy(super_wvfn, slice, nb_src=i, nb_dst=j, ...)
      slice_eigenvalues(j) = super_eigvals(i)
    end if
  end do

end do inner_loop
```

---

## 2. What Makes This Algorithm Work — The Five Load-Bearing Properties

### Property 1: Fresh H·ψ EVERY outer iteration (not just eigenvalue tracking)

**CASTEP location:** hamiltonian.F90:306 — `call hamiltonian_apply(eigenvectors, ...)`

**Why it matters:** The Rayleigh quotient λ_b = ⟨ψ_b|H|ψ_b⟩ converges quadratically
in eigenvector error ‖ψ_b − ψ_b^exact‖₂. If you only track eigenvalue deltas without
recomputing H·ψ, you can have ‖Δλ‖ < 1e-8 while ‖r_b‖ = ‖Hψ_b − λ_b Sψ_b‖ ≈ 0.01 Ha.
The quadratic convergence MASKES the true residual.

Fresh H·ψ exposes the true operator action. Convergence invalidation (line 325-329)
then correctly un-converges bands whose eigenvalues drifted after the fresh H·ψ.

**What we got wrong before:** Chebyshev-RR used the same stale H·ψ from the
Chebyshev filter step for the entire SCF iteration. No fresh recomputation meant
eigenvalue deltas were meaningless as convergence indicators.

### Property 2: Full n_bands subspace diagonalization EVERY outer iteration

**CASTEP location:** hamiltonian.F90:319 — `call wave_diagonalise(eigenvectors, H_eigenvectors, eigenvalues)`

**Why it matters:** Subspace ZHEGVD on the full n_bands × n_bands problem finds
the globally optimal eigenbasis within the span of all n_bands columns. Without
this, blocks with similar eigenvalue character (e.g., Cu-3d bands spanning
0.07 Ha) develop independent internal gauge choices that are mutually
inconsistent.

**This is the OPPOSITE of what the memory `[[chebyshev_rr_architecturally_unsuitable]]`
claimed.** That memory said "ZHEGVD on full subspace is the problem." In CASTEP's
actual code, full-subspace ZHEGVD is **essential** — it runs BEFORE the block
loop every outer iteration. The failure mode is NOT full-subspace ZHEGVD; it's
running ZHEGVD WITHOUT subsequent verification (convergence invalidation) and
WITHOUT the inner Davidson refinement loop.

### Property 3: Eigenvalue-change-based convergence, not residual-norm-based

**CASTEP location:** hamiltonian.F90:548-593

**Why it matters:** CASTEP converges bands based on |Δλ_b| < tol_abs, NOT
‖r_b‖_S⁻¹ < tol. The eigenvalue change is the physically correct metric: it
measures how much the Rayleigh quotient of band b improved after an inner
iteration of superspace diagonalization. If Δλ is below tolerance, the band
has reached the best possible eigenvector within the current superspace span.

The residual norm ‖r_b‖ is a different quantity — it measures the error in
the **current** eigenvector relative to the **current** operator H. If H is
changing across SCF iterations (which it always does), residual norms are
not monotonic and not reliable for convergence tracking.

**What we got wrong before:** Phase 0 Gate 3 (original) used residual-norm-based
locking and set lock_tol = 1e-6. This was 100,000× tighter than the residuals
our V_eff actually produces (~0.1 Ha), so zero bands ever locked — the
mechanism was vacuous.

### Property 4: Preconditioned steepest descent within each block

**CASTEP location:** hamiltonian_searchspace (lines 2075-2131) → nlpot_apply_precon_ES

```fortran
! Convert H|slice> to (H-ES)|slice> and precondition
call nlpot_apply_precon_ES(H_slice, slice, slice_eigenvalues, slice_searchspace)
call wave_Sorthogonalise(eigenvectors, slice_searchspace)
```

**Why it matters:** The search direction is the **preconditioned residual**:
t = P⁻¹ · (H − εS) · ψ, S-orthogonalized against all existing eigenvectors.

For USPP, P⁻¹ includes BOTH the TPA kinetic-energy preconditioner AND the
nonlocal pseudopotential correction (nlpot_apply_precon_ES). The TPA
preconditioner alone (P⁻¹[G] = (kinetic[G] − λ_b)⁻¹) is diagonal in G-space
and handles the kinetic-energy-dominated part of the residual. The USPP
correction handles the augmentation-charge-induced part.

**What we got wrong before:** The Chebyshev-RR path had no preconditioner at
all. Chebyshev filtering was used as a crude substitute — amplify wanted
eigenvalues, damp unwanted ones — but the exponential condition-number growth
(κ₂ ∝ |ρ₁|^m from the ChASE paper) made this a net loss for metallic systems.

### Property 5: Two-level convergence with stagnation detection

**CASTEP location:** hamiltonian.F90:542-617

CASTEP's convergence has THREE independent checks:

1. **Absolute:** |λ_new − λ_old| < tol_abs → band converged
2. **Relative:** |λ_new − λ_old| < tol_rel × initial_Δλ → band stagnated (opt_stop_condition)
3. **Anti-uphill:** λ_new > λ_old → definitely NOT converged (eigenvalues must decrease)

The stagnation check (opt_stop_condition) is critical: when a band can't improve
further within the current superspace, continuing to iterate wastes time. The
inner loop exits when ALL bands are either converged OR stagnated, but only
bands that satisfy the absolute criterion get marked as globally converged.

---

## 3. What Changed From the Failed Chebyshev-RR Attempts

### The "phase-eigensolver-migration" timeline

| Stage | Date | Algorithm | Result |
|---|---|---|---|
| Chebyshev-RR | ~May 20-24 | Chebyshev filter → ZHEGVD on full subspace → no locking | Cascade: iter-3 band-0 drift = 10.9 Ha |
| Gate 3 (Phase 0) | May 24 | Single-sweep: RQ + ZHEGVD on unconverged-only + locking | 185× reduction (drift = 0.059 Ha) |
| Phase 1A (failed) | May 25 | Outer-loop Davidson with per-block ZHEGVD + subspace accumulation | 3 bugs: vacuous locking, eigenvalue-ordering destroyed, subspace inert |
| Phase 1A (fixed) | May 25-26 | Single-sweep: RQ + ZHEGVD on unconverged-only (no outer loop) | Working baseline |
| Current (post-migration) | ~May 26+ | **Full CASTEP port**: outer loop + full subspace ZHEGVD + block-loop inner Davidson + TPA+USPP preconditioner | Converges with FFI to CASTEP |

### What the "single-sweep" got right vs wrong

The single-sweep (PHASE1A_POSTMORTEM.md) WORKED as a Phase 0 gate but was
INSUFFICIENT as a production eigensolver:

- **Right:** Locking via unconverged-only ZHEGVD arrested the cascade
- **Right:** Per-band Rayleigh quotients bypassed stale-ZHEGVD issues
- **Wrong:** No outer iteration meant no fresh H·ψ recomputation
- **Wrong:** No preconditioner meant no search-direction acceleration
- **Wrong:** No block-grouping meant all 160 bands in one ZHEGVD (degenerate-cluster gauge drift)
- **Wrong:** No stagnation detection

### What the "Phase 1A outer-loop Davidson" got wrong vs right

The outer-loop Davidson in TASKS.md (never shipped — the postmortem documents
its failure) had three structural bugs:

1. **Vacuous locking:** lock_tol=0.5 Ha was above max residual (0.104 Ha),
   so all bands locked without any eigensolver work
2. **Block ZHEGVD destroys global ordering:** per-block ZHEGVD sorts eigenvalues
   internally but places them at original positions → cross-band λ comparison broken
3. **Subspace accumulation is inert:** corrections accumulated in subspace buffer
   but never fed into ZHEGVD — they consumed memory without algorithmic purpose

### What the CURRENT working Davidson gets right

The current `davidson.rs` (3921 lines, shipped on `main`) is a faithful port of
CASTEP's algorithm with ALL five load-bearing properties:

1. ✓ Fresh H·ψ every outer iteration (line 1029-1052)
2. ✓ Full n_bands subspace diagonalization before block loop (line 1195-1230, "A1")
3. ✓ Eigenvalue-change-based convergence invalidation (lines 1300-1319, "A4")
4. ✓ TPA + USPP preconditioner (lines 906-981, preconditioner.rs)
5. ✓ Two-level convergence with stagnation (lines 606-693, `check_inner_convergence`)

Plus: BetaPhiCache for β^H·ψ projection reuse (avoids recomputing V_NL for
unchanged bands), superspace management (block-local search directions),
S-orthonormalization, and the block-grouping scheme (nblock = 2√n_bands).

---

## 4. Implications for Chebyshev Filtering Exploration

### What Chebyshev filtering CANNOT replace

Given CASTEP's algorithm, Chebyshev filtering CANNOT replace:

1. **The outer loop's fresh H·ψ recomputation** — Chebyshev filtering operates
   on ψ, not on the operator. It doesn't reveal eigenvalue drift from evolving V_eff.
2. **Full subspace diagonalization** — Chebyshev amplifies wanted eigenvectors
   but doesn't find the optimal eigenbasis within the span.
3. **The convergence check** — Chebyshev doesn't produce eigenvalue deltas.

### Where Chebyshev filtering MIGHT fit

Chebyshev filtering could serve as:

1. **Preconditioner replacement/augmentation:** Replace the TPA+USPP preconditioner
   step in the inner loop with a Chebyshev filter applied to the residual. Instead of
   t = P⁻¹ · (H − εS) · ψ, use t = C_m(H) · ψ where C_m is a degree-m Chebyshev
   polynomial. This would amplify residual components in the wanted energy window.

2. **Initial subspace enrichment:** Before the first outer iteration, apply a
   Chebyshev filter to ψ_init to enrich the initial subspace with better
   approximations to the true eigenvectors. This could reduce the number of
   outer iterations.

3. **Cold-start acceleration:** When starting from random wavefunctions (not
   CASTEP .check), use Chebyshev filtering to quickly build a reasonable
   initial subspace. This is what PARSEC does (Algorithm 4 in Banerjee et al.).

### The critical constraint from the current architecture

The current Davidson implementation is already faithful to CASTEP and working.
Any Chebyshev integration must:
- NOT break the five load-bearing properties
- NOT replace the outer loop, subspace diagonalization, or convergence check
- BE a drop-in augmentation at a specific, well-defined point in the algorithm
- BE testable with the existing Cu111+CO fixture and discriminator-value tests

---

## 5. The Knowledge Gap That Caused Repeated Failures

### What I repeatedly misunderstood

1. **"Chebyshev-RR is structurally wrong"** — Partially true (the single-sweep
   approach without locking was wrong) but the blanket condemnation of subspace
   methods was incorrect. CASTEP uses subspace diagonalization on every outer
   iteration and it works because of the surrounding safeguards.

2. **"Band-by-band CG is what CASTEP uses"** — WRONG. CASTEP uses block-grouped
   superspace Davidson, not band-by-band CG. The hamiltonian_searchspace
   subroutine computes preconditioned steepest descent, not conjugate gradient.
   There is no CG line search, no β computation, no conjugate direction update.
   Corrected memory: [[locking_is_the_load_bearing_eigensolver_property]] needs revision.

3. **"Locking is the load-bearing property"** — Partially true but incomplete.
   Locking is ONE of five load-bearing properties. Fresh H·ψ recomputation
   and convergence invalidation are equally critical. The Phase 0 single-sweep
   had locking but still produced 0.059 Ha drift because it lacked the other four.

4. **"ZHEGVD on the full subspace causes the cascade"** — The most damaging
   misunderstanding. CASTEP runs ZHEGVD on the full n_bands subspace EVERY outer
   iteration. The cascade wasn't caused by ZHEGVD; it was caused by:
   (a) Chebyshev filter polynomial distorting the subspace before ZHEGVD,
   (b) No convergence invalidation after ZHEGVD,
   (c) No inner Davidson refinement loop.

### Why "simpler" implementations failed

Every "simpler" approach I tried omitted at least one load-bearing property:

| Attempt | Fresh Hψ | Full subspace ZHEGVD | Eig-change convergence | Preconditioner | Stagnation |
|---|---|---|---|---|---|
| Chebyshev-RR (original) | ✗ | ✓ (but on distorted ψ) | ✗ | ✗ | ✗ |
| Gate 3 single-sweep | ✗ | ✓ (unconv only) | ✗ | ✗ | ✗ |
| Phase 1A multi-block | ✗ | ✓ (per-block, buggy) | ✗ | ✓ | ✗ |
| Current Davidson | ✓ | ✓ | ✓ | ✓ | ✓ |

Only the current implementation has all five. This is why it works.

---

## 6. References

- CASTEP hamiltonian.F90: `~/programming/CASTEP-GPU-port/Source/Functional/hamiltonian.F90`
  - `hamiltonian_diagonalise_slice`: lines 48-680
  - `hamiltonian_searchspace_slice`: lines 2075-2131
  - `hamiltonian_searchspace_ks`: lines 2133-2189
- Rust Davidson: `src/eigensolver/davidson.rs` (3921 lines)
- Phase 1A postmortem: `notes/plans/phase-eigensolver-migration/PHASE1A_POSTMORTEM.md`
- Gate 3 result: `notes/plans/phase-eigensolver-migration/GATE3_RESULT.md`
- Memory: [[chebyshev_rr_architecturally_unsuitable]] (needs revision — claims about ZHEGVD were wrong)
- Memory: [[locking_is_the_load_bearing_eigensolver_property]] (needs revision — incomplete)
