# Analysis: Re-evaluating Iterative Chebyshev Filtering for chemrust-scf

**Date**: 2026-05-26
**Branch**: `diag/iterative-chebyshev-viability`, forked from `feat/phase-global-woodbury`
**Purpose**: Empirically test whether iterative Chebyshev filtering with per-band
Rayleigh quotients (no ZHEGVD) is viable, or whether we should proceed with
band-by-band CG (as planned on `feat/phase-block-cg-migration`).

**Related**: See [`docs/chebyshev-filter-paper-findings.md](../docs/chebyshev-filter-paper-findings.md) for the Zhou 2014 spectral bounds and n_occ/s analysis. See [`docs/abinit-chebyshev-scf-inner-loop.md](../docs/abinit-chebyshev-scf-inner-loop.md) for ABINIT's inner-loop implementation reference.

---

## 1. What the PARSEC Paper Teaches

Key reference: Liou, Yang & Chelikowsky (2020),
"Scalable Implementation of Polynomial Filtering for DFT Calculation in PARSEC"

### Algorithm 4: Chebyshev-filtered Subspace Iteration (CheFSI)

```
for iter = 1 to maxiter:
    W = ChebyFilter(H, V, m, εF, λub, λlb)
    V = Orth(W)                         ← Cholesky QR
    (V, D) = RayleighRitz(H, V)        ← extract eigenpairs
```

This is NOT what we do. Our `diagonalize()` does:
```
1. Chebyshev filter (once)
2. Rayleigh-Ritz (once)
3. Done
```

Three divergences from the paper:
1. **No subspace iteration loop** — single pass vs Paper's `maxiter` cycles per SCF step
2. **No orthonormalization** — Paper uses Cholesky QR (`A = W^T W, A = R^T R, V = W R⁻¹`) between filter and RR
3. **No λlb-based stabilization** — Paper's Algorithm 1 uses σ = e/(c − λlb), τ = 2/σ for a stabilized recurrence; we use standard T_k = 2σ(H)T_{k−1} − T_{k−2}

Two domain mismatches (Paper's assumptions that don't hold for us):
4. **NCPP vs USPP**: Paper uses Troullier-Martins norm-conserving PPs (S=I).
   We use ultrasoft PPs (S = I + Σ β·Q·β^H).
5. **Insulator vs metal**: Paper's test systems (Si nanocrystals) have a large
   HOMO-LUMO gap. They note (p. 7): *"For systems with a relatively large gap...
   it may not be necessary to perform the Rayleigh-Ritz procedure because the
   basis vectors can be any orthonormal basis vectors that span the same invariant
   subspace."* For our Cu(111)+CO metal with ~13 Cu-3d bands within 0.07 Ha,
   we DO need individual eigenpairs for Fermi-Dirac occupancy.

The paper's success on (4)+(5) is why Chebyshev filtering works for them:
Near-degenerate clusters are irrelevant when any subspace basis works (S=I, insulator).

---

## 2. What `feat/phase-global-woodbury` Found

### Gate 3 (Davidson locking test, commit `4e6ad91`): PASSED

**Gate 3' (synthetic lock identity)**:
- All 13 Cu-3d bands preserved byte-for-byte after locking
- Cu-3d block sum = 12.9999993918 (target 13.0), deviation 6e-7
- Locking invariant holds

**Gate 3'' (SCF-3 cascade test)**:
- Band-0 drift: 0.059 Ha (Chebyshev-RR baseline: 10.9 Ha — **185× reduction**)
- At iter-1 and iter-2: all 160 bands locked (max residual < 0.5 Ha), ZHEGVD never ran
- At iter-3: 9/160 bands exceeded lock_tol, ZHEGVD on 9×9 subspace only

**Conclusion from Gate 3**: Locking IS the load-bearing property. The cascade is
primarily driven by the **Chebyshev filter polynomial distorting the subspace**
before ZHEGVD ever runs. Eliminating the filter (Davidson's direct Hψ + Rayleigh quotient)
reduces drift by 185×. Locking then keeps converged bands stable.

### Phase 1A (iterative Davidson with subspace accumulation): FAILED

Attempted full iterative Davidson with:
- TPA preconditioner + subspace accumulation + per-block ZHEGVD + Gram-Schmidt

Three root causes:
1. **Per-block ZHEGVD destroys eigenvalue ordering** — blocks sorted internally,
   placed at original positions → bands compared with wrong reference eigenvalues
2. **Subspace accumulation structurally inert** — correction vectors stored in
   `subspace_dev` but ZHEGVD only operates on `psi_dev`; corrections never fed into RR
3. **OOM on 8 GB GPU** — baseline 3.5-4 GB, subspace expansion pushes past 8 GB

**Fundamental limit**: Any algorithm that accumulates a growing subspace
(iterative Davidson, iterative Chebyshev with QR+subspace) will struggle with
8 GB VRAM for our problem size (~100k PW × 160 bands).

### Iterative Chebyshev proposal: KILLED — two untested premises

The iterative Chebyshev proposal (iterative Chebyshev + Cholesky QR + per-band
Rayleigh quotients, no ZHEGVD) was killed by review on `feat/phase-global-woodbury`.
Three reasons were cited, two of which are based on untested or contaminated data:

| Reason | Status | Notes |
|--------|--------|-------|
| ChASE bound: κ₂ ≤ η·|ρ₁|^m (orthogonality destruction) | **UNTESTED** | Worst-case bound for standard eigenproblems; Cu-3d near-degenerate cluster may be far less severe |
| Performance: 120s/sweep → 26 sweeps = 52 min/SCF step | **CONTAMINATED** | 120s dominated by CPU D_screened (~54s) + CPU augmented density, not Chebyshev filter |
| Unproven for USPP generalized eigenproblem | True, but testable | ChASE handles generalized eigenproblems; only S-matrix is new |

---

## 3. Why Re-evaluate Iterative Chebyshev

The woodbury branch jumped from "Phase 1A (iterative Davidson) failed" directly to
"band-by-band CG is the only path" without thoroughly testing the intermediate option:
**iterative Chebyshev + Cholesky QR + per-band Rayleigh quotients + locking**.

This approach is distinct from both:
- **Phase 1A Davidson**: No subspace accumulation → constant memory
- **Current single-sweep Chebyshev**: Multiple passes per SCF step → converges the subspace
- **It avoids ZHEGVD entirely** → no gauge drift in near-degenerate clusters

Key advantages over CG:
- Chebyshev filter does more work per pass than CG's line search (may need fewer outer iterations)
- Already GPU-accelerated (FFT, cuBLAS) — no new kernel development needed
- Constant memory (no subspace accumulation, no per-band state)
- Band-by-band CG on the woodbury branch doesn't exist yet (it's on the other branch)

Memory analysis (iterative Chebyshev):
```
Buffer             Size                           Already exists
─────────          ──────────                     ──────────────
ψ (input/output)   n_pw × n_bands × 16            yes
buf_a, b, c (recur) 3 × n_pw × n_bands × 16      yes
hpsi_dev           n_pw × n_bands × 16            yes
grid_dev           n_bands × grid_size × 16       yes
Sψ                 1 × n_pw × n_bands × 16        reuse buf_a
Cholesky QR        ~1 MB (n_bands²)               new, but tiny
Residual (1 band)  1 × n_pw × 16                  reuse buf_c
───────────────────────────────────────────────────────────────
Total: ~same as current (no subspace accumulation)
```

The Cholesky QR adds only n_bands² ≈ 25600 complex entries ≈ ~400 KB per buffer
(for A = W^H·S·W, need one temp buffer for R⁻¹). Negligible.

---

## 4. What We Need to Diagnose

### Diagnostic 1: Orthogonality After Chebyshev Filtering ✅ **PASSED**

**Question**: For our specific Cu(111)+CO system with ndeg=8, how badly does
Chebyshev filtering destroy orthogonality?

**Result** (2026-05-26):
- **κ₂ = 1.0** (literally perfect, not just < 10³)
- **Off-diagonal max = 7.6e-15** (machine epsilon)
- **Diagonal elements = 1.0** exactly
- **R-ChFSI amplification = 1.9-3.3×** per iteration (not 10×)

**Conclusion**: The Chebyshev filter **preserves orthogonality perfectly** for our
system. The ChASE worst-case bound κ₂ ≤ η·|ρ₁|^m does not apply because the
occupied band cluster (Cu 3d + CO states) spans a narrow energy range, so all
bands get amplified roughly equally → |ρ₁| ≈ 1 → κ₂ stays at 1.0.

**Critical fix**: The first diagnostic run failed with SVD NoConvergence because
we used dummy occupations (`vec![1.0; n_bands]`) instead of proper Fermi-Dirac
occupations. After fixing to use `chemrust_scf::density::compute_occupations()`
(same as `ca_scf_convergence.rs`), D-screening computed correctly and orthogonality
became perfect.

**Recommendation**: Proceed with iterative Chebyshev implementation. The filter is
NOT the source of the iter-2 divergence — the bug must be elsewhere in the SCF loop.

### Diagnostic 2: Memory Footprint

**Question**: Do all buffers fit in 8 GB?

**Answer**: Almost certainly yes based on analysis above. Can be confirmed by:
- Tracing GPU allocations during the diagnostic (nvidia-smi peak memory)
- Counting buffer sizes in the code

### Diagnostic 3: Per-Band Rayleigh Quotient Residuals

**Question**: After one Chebyshev filter pass + Cholesky QR, do per-band
Rayleigh quotients produce residuals below a lockable tolerance?

**Approach**:
1. Run one Chebyshev filter pass + Cholesky QR
2. For each band b: compute λ_b = ⟨ψ_b|H|ψ_b⟩ / ⟨ψ_b|S|ψ_b⟩ (Rayleigh quotient)
3. Compute residual r_b = H|ψ_b⟩ − λ_b·S|ψ_b⟩
4. Compute ‖r_b‖_S⁻¹ = √⟨r_b | S⁻¹·r_b⟩
5. Report histogram of residual norms

**Discriminator**:
- max ‖r_b‖_S⁻¹ < 0.1 Ha: bands can lock quickly at loose tolerance
- max ‖r_b‖_S⁻¹ > 0.5 Ha: single pass insufficient, need outer iterations

### Diagnostic 4: Outer Iteration Convergence Rate

**Question**: With iterative Chebyshev (filter → QR → RQ → lock), how many
outer iterations are needed to lock >80% of bands?

**Approach**:
1. Run outer loop: filter → Cholesky QR → H+apply → per-band RQ → lock
2. Track n_locked per iteration
3. lock_tol = 0.01 Ha (fixed for diagnostic)
4. Report convergence curve

**Discriminator**:
- >80% locked within 3-5 outer iterations: iterative Chebyshev is viable
- <50% locked after 20 iterations: abandon, use CG instead

---

## 5. Decision Matrix

| Diag 1 (orth) | Diag 2 (residual baseline) | Diag 3 (convergence) | Conclusion |
|:---:|:---:|:---:|---|
| **κ₂ = 1.0 ✅** | **max ‖r‖_S⁻¹ = 0.209 Ha ✅** (reasonable, not catastrophic) | TBD | **Iterative Chebyshev still viable. Awaiting Diagnostic 3.** ← **CURRENT** |
| κ₂ = 1.0 | max ‖r‖ < 0.1 Ha | >80% lock in 3-5 iters | **Iterative Chebyshev works. Implement.**
| κ₂ = 1.0 | max ‖r‖ > 0.5 Ha | >80% lock in 6-10 iters | **Iterative Chebyshev viable but slower. Worth using.**
| κ₂ < 10³ | any | >80% lock in ≤10 iters | **Iterative Chebyshev viable. May need Cholesky QR.**
| any | any | <50% lock in 10 iters | **Iterative Chebyshev marginal. Use CG.** |

**Status** (2026-05-27):
- ✅ **Diagnostic 1**: κ₂ = 1.0 (perfect orthogonality)
- ✅ **Diagnostic 1b**: κ₂ = 1.0 for ALL filter modes including production SinvHKeepHEig
- ✅ **Diagnostic 2**: Per-band residuals established — max S⁻¹ residual 0.209 Ha, reasonable baseline
- 🔄 **Diagnostic 3**: Outer loop convergence — being implemented/tested
- ⬜ **Diagnostic 4**: Harmonic RR — queued
- ⬜ **Diagnostic 5**: Band-locking — queued

---

## 6. Codebase State on `feat/phase-global-woodbury`

### What's available

| Module | Purpose | Compiles? |
|--------|---------|-----------|
| `src/scf.rs` | SCF state machine with diagonalize | Yes (lib) |
| `src/eigensolver/chebyshev.rs` | Chebyshev filter, spectral bounds, FilterMode | Yes |
| `src/eigensolver/hamiltonian.rs` | H·ψ, S·ψ, S⁻¹·ψ (GPU) | Yes |
| `src/eigensolver/kernels.rs` | CUDA kernels (compiled via NVRTC) | Yes |
| `src/eigensolver/preconditioner.rs` | TPA diagonal preconditioner | Yes |
| `src/eigensolver/davidson.rs` | Davidson v1 with locking | Yes |
| `src/eigensolver/rayleigh_ritz.rs` | ZHEGVD, H_sub/S_sub build | Yes |
| `src/eigensolver/vnl_data.rs` | V_NL batch data (β, Q, D, Woodbury M) | Yes |
| `tests/fixtures/cu111_co.rs` | Cu111_CO fixture loader | Yes |
| `tests/eigenvalue_residual_validation.rs` | Template for residual diagnostics | Yes |
| `tests/rayleigh_ritz_validation.rs` | CPU linear algebra helpers (matmul, hermiticity) | Yes |

### What doesn't compile (non-critical)

- `src/scf.rs` lines 1845-1850: `CheckFile::write` argument order mismatch
  (expects `(writer, data)`, our code passes `(data, writer)`)
- `src/scf_capture.rs` lines 170-178: `CastepBin` struct field mismatch
  (newer chemrust-hamiltonian API has additional fields)

These only affect the `.check`-file writing codepath, not the eigensolver
or filter code paths.

### Test infrastructure patterns

The test files (`eigenvalue_residual_validation.rs`, `davidson_v1_validation.rs`)
follow a consistent pattern:
1. `fixtures::cu111_co::fixture()` → get cached Cu111_CO data
2. `fixtures::cu111_co::build_scf_state(fx)` → create ScfIteration<Initialized>
3. `state.build_v_eff_with_energy()` → ScfIteration<VEffBuilt>
4. `state.diagonalize(ndeg, None)` → ScfIteration<WavefunctionsUpdated>
5. Extract results from the state

For GPU-specific tests, the pattern is:
1. Check `gpu_available()` at the start
2. Use `#[ignore = "requires GPU and CASTEP fixture data"]` for CI skip
3. The SCF state internal GPU setup is handled by `diagonalize()`
