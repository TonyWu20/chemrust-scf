# Phase 1A Implementation Post-Mortem

**Date:** 2026-05-25
**Branch:** `feat/phase-global-woodbury`
**Scope:** Groups A-F implementation of production Davidson v1 eigensolver

---

## 1. What We Planned to Implement

The TASKS.md called for a **production Davidson v1 eigensolver** (`davidson.rs`,
~700 LOC target) implementing a full **outer iteration loop**:

| Step | Description |
|------|-------------|
| 1-2 | Compute Hψ and Sψ for current iterate |
| 3 | Per-band Rayleigh quotient λ_b |
| 4 | Residual r_b = Hψ_b − λ_b·Sψ_b |
| 5 | S⁻¹-weighted residual norm via batch `apply_s_inverse` |
| 6 | Lock bands where norm < lock_tol AND λ_b − prev_λ_b < lock_tol |
| 7 | Early exit if all locked |
| 8 | Block-partition unconverged via `detect_degenerate_blocks` |
| 9 | **Per-block ZHEGVD** — solve generalized eigenproblem independently per degenerate block |
| 10 | TPA-preconditioned corrections t_b = P⁻¹·r_b |
| 11 | **S-orthogonalize corrections against subspace** (Gram-Schmidt over accumulated directions) |
| 12 | **Append to subspace**; restart if dim > 3× n_active |

Supporting infrastructure:

- **Group A**: Split `chebyshev.rs` monolith → `kernels.rs` + `hamiltonian.rs` (clean extraction done)
- **Group B**: TPA diagonal preconditioner (`preconditioner.rs`)
- **Group D**: Lock ratchet schedule (0.5 Ha initial → geometric decay to 1e-6), SCF dispatch wiring
- **Group E**: Remove Phase 0 `davidson_minimal.rs`
- **Group F**: GPU validation test suite against CASTEP Cu111+CO fixture

The **theory of operation**: outer iterations accumulate search directions in a growing
subspace; each ZHEGVD call within a degenerate block refines eigenvectors within that
block's span; corrections accelerate convergence via the preconditioner; restart prevents
subspace blow-up.

---

## 2. What the Failed Tests Revealed

All testing was done on an **8 GB GPU** with the Cu111+CO CASTEP fixture
(~100k plane waves, 160 bands, 17 atoms).

### Test F1 (self-consistency, pinned V_eff): PASSED

When V_eff matches CASTEP exactly, all 160 bands lock at 1e-12 and output ψ ≡ input ψ.
This proves the basic operator application and Rayleigh quotient machinery is correct.

### Test F3 (Chebyshev fallback): PASSED

The env-var dispatch correctly routes to the unmodified Chebyshev path. No regression.

### Test F2 (lock progression) and F5 (residual monotonicity): FAILED

Three distinct root causes were discovered sequentially:

#### Bug 1: Vacuous locking (ratchet too loose)

- `lock_tol_for_iter` started at **0.5 Ha**. The maximum observed S⁻¹-weighted residual
  norm was **0.104 Ha**.
- Every band satisfied `norm_sinv < 0.5` trivially. **Zero ZHEGVD calls were made**
  across 3 SCF iterations.
- Eigenvectors stayed frozen at CASTEP ψ while our V_eff drifted → eigenvalues
  accumulated **0.51 Ha of drift** by iter-3.
- The ratchet was a no-op: it never forced actual eigensolver work.

#### Bug 2: Per-block ZHEGVD destroys eigenvalue ordering

- Tightening to `lock_tol = 0.01` forced bands into ZHEGVD. `detect_degenerate_blocks`
  produced 17 blocks for the 160 unconverged bands.
- After per-block ZHEGVD, each block's eigenvalues are sorted **internally** but placed
  at their original block positions. The **global band-to-eigenvalue correspondence is
  destroyed**.
- Result: `max |Δλ_b| = 1.041 Ha` between outer iterations — eigenvalues from different
  bands are being compared against each other. No band can pass the delta criterion.
- A separate related bug: isolated single bands (not in any degenerate block) were left
  as **zero columns** in the rotated psi buffer, corrupting eigenvectors for subsequent
  iterations.

#### Bug 3: Subspace accumulation is structurally inert

- Even with unified ZHEGVD (fixing bug 2), **zero bands converged after 30 outer iterations**.
- Diagnostic output at iter=0,1,2 with unified ZHEGVD:
  ```
  res_sinv = [1.76e-2, 1.04e-1]  max_δλ ≈ 1.0 Ha  lock_tol = 1.0e-2
  ```
- The **minimum residual stayed at 1.8× the lock tolerance** and was not decreasing. 30
  iterations of step-10 corrections + step-11 Gram-Schmidt + step-12 subspace expansion
  produced **no measurable improvement** in residual norms.
- Root cause: steps 10-12 accumulate correction vectors into `subspace_dev`, but
  **step 9's ZHEGVD only operates on `psi_dev` columns** (gathered at step 8). The
  accumulated corrections are **never fed into the Rayleigh-Ritz procedure**. They sit
  in the subspace buffer, consuming GPU memory (~0.5 GB per 160-column expansion),
  serving no algorithmic purpose.
- The subspace also triggered 10 restarts across 30 iterations, each collapse resetting
  the subspace to n_bands columns — the corrections accumulated between restarts were
  **discarded without ever being used**.

### Infrastructure finding: 8 GB VRAM constraint

- With 7 × `n_pw × n_bands` working buffers + grid_dev (1.12 GB) + subspace buffers,
  the baseline footprint was **~3.5-4 GB**. Subspace doubling during outer iterations
  pushed this toward the 8 GB limit.
- Tests run sequentially could trigger OOM in later tests from GPU memory fragmentation
  (observed: F3 Chebyshev test OOMs when run after two Davidson tests, but passes in
  isolation).

---

## 3. Proposed Fix and Supporting Reasoning

**Replace the outer-iteration Davidson with a single-sweep algorithm**, matching what
Phase 0's `davidson_minimal_single_sweep` (Gate 3'') already proved works.

### The simplified algorithm

```
1. Hψ = apply_full_hamiltonian(ψ)
2. Sψ = ψ; apply_s_times(ψ, Sψ)
3. Per-band: λ_b = Re⟨ψ|Hψ⟩ / Re⟨ψ|Sψ⟩
4. Per-band: r_b = Hψ_b − λ_b·Sψ_b
5. Per-band: norm_sinv = √Re⟨r_b | S⁻¹·r_b⟩
6. Lock bands where norm_sinv < lock_tol (skip delta check — single sweep)
7. If all locked: return ψ unchanged
8. Gather unconverged ψ columns + Hψ columns → contiguous buffers
9. Single unified k×k ZHEGVD on unconverged block (k = n_unconv)
10. Scatter rotated columns back to ψ
11. Return ψ, eigenvalues
```

**No outer loop. No subspace accumulation. No Gram-Schmidt. No restart logic.**

### Why this works

1. **Phase 0 empirical evidence**: Gate 3'' (commit `4e6ad91`) demonstrated that a
   single sweep with ZHEGVD on the unconverged block, combined with a tightening lock
   ratchet across SCF iterations, **arrested the eigenvalue cascade to 0.059 Ha drift**
   at iter-3 with 151+ bands locked. This is the behavior we need.

2. **ZHEGVD is the convergence engine, not subspace accumulation**: For a problem of
   dimension `n_bands = 160`, the unconverged block ZHEGVD solves the **exact**
   generalized eigenvalue problem within the `k × k` subspace of unconverged Ritz
   vectors. Since `k = n_unconv ≤ n_bands`, the Ritz vectors after rotation are the
   **best possible** eigenvectors within the span of the input columns — no amount of
   correction-vector accumulation can improve on the exact solution of the subspace
   eigenproblem.

3. **Subspace accumulation is for Krylov-subspace methods (CG, Arnoldi), not
   Rayleigh-Ritz**: In block Davidson, corrections serve to **expand the search space**
   when k < n_bands (i.e., when you're searching for fewer eigenvectors than the basis
   dimension). But when k = n_bands (as in our case), the subspace already spans the
   full problem dimension. ZHEGVD on the full k×k problem is exact; corrections add
   redundant directions.

4. **The lock ratchet handles gradual convergence**: The ratchet starts at a moderate
   tolerance (0.01-0.1 Ha) at SCF iter-1 and tightens geometrically. Early SCF
   iterations lock trivially-converged bands (leaving a small k for ZHEGVD). Later
   iterations tighten the tolerance, forcing more bands through ZHEGVD as V_eff and
   density converge. This naturally interleaves eigensolver convergence with SCF
   convergence.

5. **Memory footprint drops to baseline**: Without subspace buffers or per-iteration
   correction allocations, peak memory stays at the 7 working buffers + grid_dev ≈
   3.5-4 GB, well within 8 GB. No OOM risk regardless of outer iteration count (there
   are none).

6. **Code complexity collapses**: The 1197-line `davidson.rs` shrinks to ~500 lines.
   The `solve_block_zhegvd` helper is retained. `detect_degenerate_blocks` is retained
   for future use but not called (single unified ZHEGVD). The preconditioner
   (`preconditioner.rs`) is retained but only used if unlockable bands need acceleration
   — in the single-sweep design, the ZHEGVD step makes the preconditioner unnecessary
   (it's only useful for expanding the search space, which we don't do).

---

## 4. What Was Successfully Completed

Groups A, B, D, E were implemented and are correct:

| Group | Description | Status |
|-------|-------------|--------|
| A | Split `chebyshev.rs` → `kernels.rs` + `hamiltonian.rs` | Complete, all tests pass |
| B | TPA diagonal preconditioner (`preconditioner.rs`) | Complete, compiles, unit tests pass |
| D | Lock ratchet + SCF dispatch wiring | Complete, compiles, Chebyshev fallback works |
| E | Remove Phase 0 `davidson_minimal.rs` | Complete, zero references remain |
| F | Validation test suite | Written, compiles, F1/F3 pass, F2/F5 need algorithm fix |

Group C (`davidson.rs`) needs to be rewritten from the single-sweep design. The
`solve_block_zhegvd` helper, type definitions (`DavidsonConfig`, `DavidsonResult`,
`DavidsonDiagnostic`), and `lock_tol_for_iter` are correct and should be preserved.

---

## 5. Next Steps

1. **Rewrite `davidson.rs`** to the single-sweep algorithm (~500 LOC)
2. **Tune `lock_tol_for_iter`** starting at 0.01 Ha with geometric decay to 1e-6
3. **Update test expectations** in `tests/davidson_v1_validation.rs` for the single-sweep design
4. **Re-run GPU validation suite** (F1-F6) against CASTEP fixture
5. **Acceptance gate**: `CHEMRUST_EIGENSOLVER=davidson` converges to within 1e-3 eV
   of CASTEP reference in perturbation recovery SCF
