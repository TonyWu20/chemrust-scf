# Chebyshev Filtering IS an Eigensolver — Corrected Understanding

**Date:** 2026-06-15
**Status:** Literature review complete — CheFSI vs CASTEP Davidson comparison
**Primary sources:**
- Zhou et al. (2014), "Chebyshev-filtered subspace iteration" — foundational CheFSI Algorithm 4.1
- Levitt & Torrent (2015), "Parallel eigensolvers in plane-wave DFT" — S⁻¹·H derivation, USPP
- Das et al. (2025), "Residual-based ChFSI tolerant to inexact matrix-vector products" — R-ChFSI Alg 3
- Di Napoli & Wu, "Estimating the condition number of Chebyshev filtered vectors" — κ₂ growth analysis
- Liou et al. (2020), "Scalable implementation of polynomial filtering in PARSEC" — production DFT CheFSI

---

## 1. Correction to CASTEP_ALGORITHM_LESSONS.md

In `CASTEP_ALGORITHM_LESSONS.md` §4 (written earlier today), I claimed:
> "Chebyshev filtering CANNOT replace the outer loop's fresh H·ψ recomputation"
> "Chebyshev filtering CANNOT replace full subspace diagonalization"
> "Chebyshev filtering could serve as a preconditioner replacement"

**These claims are wrong.** Chebyshev filtered subspace iteration (CheFSI) is a
COMPLETE eigensolver algorithm, not a preconditioner. The three-step algorithm
(Algorithm 4 in PARSEC paper, Algorithm 1 in Levitt-Torrent):

```
for iter = 1 to maxiter:
    1. W = ChebyFilter(H, V, p, εF, λub, λlb)   // polynomial subspace enrichment
    2. V = Orth(W)                                 // Cholesky QR orthonormalization
    3. (V, Λ) = RayleighRitz(H, V)                // subspace diagonalization
       if converged: exit
```

This IS the full algorithm. It solves the eigenproblem (H − λS)·ψ = 0 to
machine precision. The convergence mechanism is fundamentally different from
Davidson:

| Mechanism | Davidson (CASTEP) | CheFSI |
|---|---|---|
| Subspace enrichment | Preconditioned steepest descent | Chebyshev polynomial C_p(S⁻¹·H) |
| Convergence driver | Eigenvalue-change tracking | Subspace angle reduction |
| Convergence rate | tan θ^(i+1) ∝ convergence_rate · tan θ^(i) | tan θ^(i+1) ∝ |C_p(λ_{n+1})/C_p(λ_n)| · tan θ^(i) |
| Orthonormalization | S-Gram-Schmidt per block | CholeskyQR or Gram-Schmidt per iteration |
| Outer iterations | 0-30 per SCF step | 1-5 per SCF step |
| Spectral info needed | mean_ek for TPA | λ_max, λ_min, λ_T (εF) |
| USPP handling | S in preconditioner + S-orth | S⁻¹·H in filter recurrence |

**Both are valid eigensolvers.** Both solve generalized EVP. Both use subspace
diagonalization (ZHEGVD). Both need spectral bounds. Both can converge to
CASTEP-level precision.

## 2. Why Our Previous Chebyshev-RR Failed (The Real Reason)

The failure was NOT because "Chebyshev filtering can't work for USPP metals."
The failure was because we only did **one pass** of the CheFSI algorithm per SCF
iteration — omitting the outer iteration loop.

What the papers say:
- PARSEC Algorithm 4: "for iter = 1 → maxiter" — **1-5 iterations** per SCF step
- ChASE paper (Di Napoli & Wu): convergence to 10⁻¹² within 3-5 outer iterations
- R-ChFSI paper (Das 2025) §4: 5-15 outer iterations for DFT benchmarks

What we did:
- **One** Chebyshev filter pass → Gram-Schmidt → ZHEGVD → done
- No residual check
- No second iteration when first didn't converge

A single Chebyshev filter pass enriches the subspace but doesn't guarantee
convergence. The PARSEC paper explicitly states (Algorithm 4 line 2-5):
a full subspace iteration wraps filter+QR+RR in a convergence loop.

## 3. The Five Properties Reframed for CheFSI

Property 1 from CASTEP ("fresh H·ψ every outer iteration"):
→ CheFSI equivalent: recompute C_p(H) every outer iteration with current H
  (the filter polynomial depends on H through the recurrence)

Property 2 ("full subspace ZHEGVD"):
→ CheFSI equivalent: Rayleigh-Ritz (Algorithm 3 in PARSEC) — ZHEGVD on
  H_sub = W^H·H·W, S_sub = W^H·S·W

Property 3 ("eigenvalue-change convergence"):
→ CheFSI equivalent: residual norm convergence ‖H·ψ − ε·S·ψ‖ < τ
  (standard in all CheFSI papers)

Property 4 ("preconditioned steepest descent"):
→ CheFSI equivalent: the Chebyshev polynomial C_p(S⁻¹·H) IS the
  "preconditioner" — it amplifies the wanted subspace exponentially

Property 5 ("stagnation detection"):
→ CheFSI equivalent: built into the convergence check; if residuals
  stop decreasing, exit

## 4. What We Already Have Implemented

Our `chebyshev_filter` (chebyshev.rs:539-1179) implements R-ChFSI Algorithm 3
(Das 2025). Checking against the paper:

| R-ChFSI Algorithm 3 Step | Our implementation | Status |
|---|---|---|
| Spectral parameters: e, c, σ, γ | Lines 829-837 (compute_spectral_bounds + Lanczos) | ✓ |
| Step 1: Y = H·X − S·X·Λ | Lines 743-814 (apply_full_hamiltonian, apply_s_times, band_scale_axpy) | ✓ |
| Step 2: R_X=0, R_Y=(σ₁/e)·Y, Λ_X=I, Λ_Y=(σ₁/e)(Λ−cI) | Lines 842-885 | ✓ |
| Step 3: k=2..p recurrence | Lines 890-993 | ✓ |
| Step 4: X_new = D⁻¹·R_Y + X·Λ_Y | Lines 997-1020 | ✓ |
| Gram-Schmidt S-orthonormalization | Lines 1023-1122 | ✓ |
| Final H·ψ for RR | Lines 1124-1143 | ✓ |

**What's missing for a complete CheFSI eigensolver:**

1. **Outer iteration loop** — wrap filter+RR in a convergence loop
2. **Residual norm computation** — after RR, compute ‖H·ψ_new − ε·S·ψ_new‖
3. **Convergence gate** — exit when max residual < τ
4. **The S⁻¹ mode correctness** — verify FilterMode::SinvHKeepHEig matches
   the paper's Algorithm 2 (not Algorithm 3; Algorithm 2 is standard ChFSI
   which operates on eigenvectors with S⁻¹·H; Algorithm 3 is R-ChFSI which
   operates on residuals)

**Critical detail about the filter mode:**
Our default `FilterMode::SinvHKeepHEig` applies S⁻¹·H in the **standard ChFSI**
recurrence (Algorithm 2). But R-ChFSI (Algorithm 3) uses **bare H** in the
recurrence and applies S⁻¹ only at Step 4 (reconstruction). This is load-bearing:
R-ChFSI's residual formulation multiplies the S⁻¹ error by ‖R‖ which → 0.

Our code at lines 916-927 applies S⁻¹ to H·R_Y only when filter_mode is
SinvHKeepHEig or SinvHFullDas. For the R-ChFSI Alg 3 recurrence (which uses
H·R_Y, not S⁻¹·H·R_Y), we should be using the bare H mode (FilterMode::BareH).

**Wait — actually, Algorithm 3 uses H·D⁻¹·R_Y, not H·R_Y.** Looking at line 603:
```
R_X = (2σ₂/e) · A·D⁻¹·R_Y − (2σ₂c/e) · R_Y − σ·σ₂·R_X + (2σ₂/e) · Y·Λ_Y
```

So Algorithm 3 DOES apply D⁻¹ in the recurrence, but only to the residual R_Y
(which is small), not to the eigenvectors. In Algorithm 2 (standard ChFSI):
```
Y_k+1 = (2σ₂/e) · H·B⁻¹·Y_k − (2σ₂c/e) · Y_k − σ·σ₂·Y_k-1
```
Here B⁻¹ is applied to Y_k (which is O(1)).

The difference: B⁻¹ error goes to 0 in Alg 3 (multiplied by ‖R‖), but stays
constant in Alg 2 (multiplied by ‖Y‖ ≈ 1).

## 5. The Exploration Path

Given that our code already implements R-ChFSI Algorithm 3, the exploration is:

**Goal:** Wrap `chebyshev_filter` + `rayleigh_ritz` in an outer convergence loop
to implement the complete CheFSI subspace iteration (Algorithm 4 from PARSEC).

**What to implement:**
1. Outer loop: for iter in 1..maxiter { filter → RR → check convergence }
2. Residual norm computation: ‖H·ψ_b − ε_b·S·ψ_b‖_S⁻¹ per band
3. Convergence gate: max(residuals) < τ → exit
4. Verify FilterMode validity: confirm BareH is correct for R-ChFSI
   (the recurrence operates on residuals of bare H; S⁻¹ only at reconstruction)

**What's reusable without changes:**
- `chebyshev_filter()` — the R-ChFSI filter (1887 lines, working)
- `rayleigh_ritz()` — the RR step (953 lines)
- `lanczos_upper_bound()` + `compute_spectral_bounds()` — spectral estimation
- `apply_full_hamiltonian()`, `apply_s_times()`, `apply_s_inverse()` — operators
- `gram_schmidt_s()` — S-orthonormalization
- All GPU infrastructure (cuBLAS, cuFFT, NVRTC kernels, Woodbury S⁻¹)

**What needs to be written:**
- ~100 lines: outer iteration loop + residual computation + convergence check
- Integration in `scf.rs`: new `CHEMRUST_EIGENSOLVER=chebyshev_iterative` path

**Risk assessment:**
- The κ₂ growth concern (ChASE paper) is handled by Gram-Schmidt between iterations
- The S⁻¹ inexactness concern is handled by R-ChFSI's residual formulation
- The metallic system concern (degenerate clusters) was tested in the R-ChFSI paper:
  they tested on Mo, Si, C supercells — metallic and insulating — with convergence
  to 10⁻⁸ in all cases

## 6. References

| Paper | Key content | Our code mapping |
|---|---|---|
| Zhou 2014 | Algorithm 4.1 — complete CheFSI eigensolver | Not yet — need outer loop |
| Levitt-Torrent 2015 | S⁻¹·H rationale, Woodbury S⁻¹ for USPP | `apply_s_inverse()` = Woodbury |
| Das 2025 | Algorithm 3 — R-ChFSI | `chebyshev_filter()` lines 539-1179 |
| Das 2025 | Theorem 3.4 — convergence with inexact S⁻¹ | Our S⁻¹ has ~1.4% error (ALGORITHM_RATIONALE.md §4) |
| Di Napoli & Wu | κ₂ ≤ η·|ρ₁|^p — condition number bound | Handled by Gram-Schmidt (lines 1023-1122) |
| Liou et al. 2020 | PARSEC Algorithm 4 — production DFT CheFSI | Our target architecture |
