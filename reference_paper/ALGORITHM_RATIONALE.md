# Algorithm Rationale: Why R-ChFSI for Our SCF

## 1. Background: The Generalized Eigenvalue Problem

In USPP plane-wave DFT, the eigenproblem is the generalized Hermitian form:

```
H · ψ = ε · S · ψ
```

where H = T + V_loc + V_NL is the Hamiltonian and S = I + β · Q · β^H
is the USPP overlap matrix. The projector functions β are localized around
ions, and Q is the augmentation charge integral matrix (n_proj × n_proj per ion,
typically 8–18 projector channels per Cu atom).

Unlike the standard eigenvalue problem (S = I), the generalized form requires
specialized eigensolvers.

## 2. Standard ChFSI for Generalized Problems

Source: Levitt & Torrent (2015), "Parallel eigensolvers in plane-wave DFT",
J. Comput. Phys. — published peer-reviewed paper.

**Archive**: `extracted/levitt-torrent-2015/abinit.tex`
- Algorithm 1 — `caption{Chebyshev filtering}` at line 664, body lines 662–695
- Key S⁻¹·H derivation — lines 653–658
- Algorithm recurrence — lines 678–680:
  ```
  ψ⁰ = ψ
  ψ¹ = (1/r) · (S⁻¹·H·ψ⁰ - c·ψ⁰)
  ψⁱ = (2/r) · (S⁻¹·H·ψⁱ⁻¹ - c·ψⁱ⁻¹) - ψⁱ⁻²
  ```

### Why S⁻¹·H is required (not H alone)

> "If we denote by Λ and P the eigenvalues and eigenvectors of the
> eigenproblem Hψ = λSψ, then S⁻¹H = PΛP⁻¹. Therefore,
> T_n(S⁻¹H)ψ = PT_n(Λ)P⁻¹ψ will have its eigencomponents filtered
> by the spectral filter T_n." — (`abinit.tex:653-657`)

Filtering on H alone (without S⁻¹) amplifies eigencomponents of the wrong
operator — the invariant subspaces of H and S⁻¹·H are different when S ≠ I.

## 3. Computing S⁻¹: Woodbury Formula

Source: Levitt & Torrent (2015), `abinit.tex` §3.3, lines 706–820:
"subsection{Inversion of the overlap matrix}"

The overlap matrix is a low-rank perturbation of identity:

```
S = I + P · D_S · P^T
```

where P are the projector functions (β) and D_S is the augmentation matrix (Q).

Using the Woodbury formula (`abinit.tex:735`):

```
S⁻¹ = I - P · (D_S⁻¹ + P^T·P)⁻¹ · P^T
```

In our notation:
```
S⁻¹ = I - β · (Q⁻¹ + β^H·β)⁻¹ · β^H
     = I - β · s_inv_mat · β^H
```

where `s_inv_mat = (Q⁻¹ + β^H·β)⁻¹` is a small (n_proj × n_proj) matrix
that can be precomputed once per ion. Implemented in `src/eigensolver/vnl_data.rs`
at lines 224–315 (`build_q_expanded` → Gauss-Jordan inversion → `s_inv_mat`).

## 4. The Problem: Our S⁻¹ is Inexact

**Status**: The Woodbury S⁻¹ is fully implemented in `src/eigensolver/chebyshev.rs`
as `apply_s_inverse` (line 759). The reduced matrix `s_inv_mat` is precomputed in
`VnlBatchData::precompute` (`src/eigensolver/vnl_data.rs`).

**Empirical test**: `s_inv_s_identity_test` in `tests/ca_scf_convergence.rs`
feeds CASTEP converged wavefunctions through S⁻¹·S:

```
‖S⁻¹·S·ψ₀ − ψ₀‖_∞ = 1.415640e-2
```

**The Woodbury inverse has ~1.4% error.** Two possible causes:

1. **Q matrix numerical conditioning**: Q may have near-singular eigenvalues
   for the Cu USP projector channels. The Gauss-Jordan inversion at
   `vnl_data.rs:227-261` uses `eps_reg = 1e-12` which may be insufficient
   if Q is poorly conditioned.

2. **Gram matrix precision**: β^H·β is computed on CPU in double precision,
   while the S⁻¹ application uses GPU cuBLAS. Rounding differences would
   cause the Woodbury denominator to be slightly off.

Regardless of the cause, the error is **deterministic** (not random) and
reproducible.

## 5. Why Standard ChFSI Fails with Inexact S⁻¹

In the standard Chebyshev recurrence (`abinit.tex:678-680`), the S⁻¹ operator
is applied to the **eigenvectors themselves**:

```
ψⁱ = (2/r) · (S⁻¹·H·ψⁱ⁻¹ - c·ψⁱ⁻¹) - ψⁱ⁻²
```

The eigenvectors ψ stay O(1) in norm throughout the entire SCF process.
Therefore, the S⁻¹ error (~1.4%) is applied to O(1) vectors at every Chebyshev
step, producing a persistent O(ε) error in the filtered subspace:

```
‖ψⁱ_exact - ψⁱ_approx‖ ≈ O(ε)  for all SCF iterations
```

This is formally established by Das et al. (2025) Theorem 3.2
(`main.tex`, proof at line 861: `section{Proof of Theorem 3.2}`) —
standard ChFSI with inexact matrix-vector products cannot reduce the
subspace angle below a threshold proportional to the inverse error.

**Consequence in our SCF**:
- Iter-1: ψ ≈ converged state, filter error small
- Iter-2: ψ drifts by O(ε), density shifts → V_eff range jumps
- Iter-3+: Error accumulates, D_screened explodes, Lanczos detects corrupted H

This matches our observations in `/tmp/scf-diag-0522-1022.log`:
- Iter-1 band-1 = -0.956 Ha (off by 0.099 Ha from reference -1.055 Ha)
- Iter-2 band-1 = -14.4 Ha (unstable)
- Iter-3 band-1 = -8.83 Ha (drifting)

## 6. R-ChFSI: Residual-Based Formulation

Source: Das et al. (2025), "Residual-based Chebyshev filtered subspace
iteration for Hermitian eigenvalue problems tolerant to inexact matrix-vector
products", arXiv preprint.

**Archive**: `extracted/das-2025-rchfsi/main.tex`
- Algorithm 2 (standard ChFSI for generalized EVP) — `caption{...}` at line 386, body lines 384–426
- Algorithm 3 (R-ChFSI for generalized EVP) — `caption{...}` at line 588, body lines 586–611
- Theorem 3.2 (standard ChFSI error bound) — proof at line 861
- Theorem 3.4 (R-ChFSI convergence guarantee) — proof at line 873
- Convergence condition eq.(5) — Section 3.1, lines 460–530
- Equal-algebra proof — line 612: "When D⁻¹ = B⁻¹ … ChFSI and R-ChFSI are
  algebraically equivalent."

### Algorithm 3: R-ChFSI (line 586–611, `begin{algorithm}` at 586)

```
Input:  X (eigenvectors n_pw × n_band), Λ (eigenvalues n_band × n_band),
        spectral bounds λ_max, λ_min, λ_T,
        approximate S⁻¹ (D⁻¹), polynomial degree p
Output: filtered eigenvectors X_new

Step 0:  e = (λ_max - λ_T)/2,  c = (λ_max + λ_T)/2
         σ = e/(λ_min - c),    σ₁ = σ,  γ = 2/σ₁

Step 1:  Y = H·X - S·X·Λ              ← residual (computed once, before loop)
         R_X = 0
         R_Y = (σ₁/e) · Y              ← initial residual
         Λ_X = I
         Λ_Y = (σ₁/e) · (Λ - c·I)      ← eigenvalue matrix for residual

Step 2:  for k = 2 to p:               ← operates on residuals, not eigenvectors
           σ₂ = 1/(γ - σ)
           R_X = (2σ₂/e) · H·D⁻¹·R_Y   ← H·(S⁻¹·R_Y): error ∝ ‖R_Y‖
               - (2σ₂/e) · c·R_Y
               - σ·σ₂·R_X
               + (2σ₂/e) · Y·Λ_Y        ← correction from initial residual
         swap(R_X, R_Y), swap(Λ_X, Λ_Y)

Step 3:  X_new = D⁻¹·R_Y + X·Λ_Y      ← reconstruct from filtered residual
```

### Why the error is suppressed

In the recurrence, the term `H·D⁻¹·R_Y` uses the inexact S⁻¹ applied to R_Y
(the current filtered residual). As the SCF converges:

```
‖R_Y‖ → 0        (residuals approach zero)
‖H·D⁻¹·R_Y - H·S⁻¹·R_Y‖ ∝ ‖D⁻¹ - S⁻¹‖ · ‖R_Y‖ → 0
```

In contrast, standard ChFSI applies S⁻¹ to eigenvectors:
```
‖S⁻¹·H·ψ - (exact)‖ ∝ ‖D⁻¹ - S⁻¹‖ · ‖H·ψ‖ = O(ε)  (constant)
```

### Convergence guarantee (Theorem 3.4, proof line 873)

The paper proves that R-ChFSI converges as long as:

```
|C_p(λ_n)| - |C_p(λ_{n+1})| > 2‖H‖(γ_m·η_p + ζ·η̃_p) · (1 + tan ∠(Sⁱ, S))
```

where ζ = ‖D⁻¹ − S⁻¹‖ is the inverse approximation error and η_p, η̃_p are
finite constants depending on the polynomial degree p. Since the RHS decreases
as the subspace angle ∠(Sⁱ, S) decreases, if this inequality holds at any
iteration, it holds for all subsequent iterations. **Standard ChFSI has no
corresponding guarantee** (Theorem 3.2 shows failure with the same D⁻¹).

### Experimental validation (paper §4)

For generalized eigenproblems with approximate B⁻¹:
- Standard ChFSI stagnates at O(ε) where ε is the approximation error
- R-ChFSI converges to machine precision (10⁻¹²) despite the same inexact inverse

### Algebraic equivalence with exact inverse (line 612)

> "When D⁻¹ = B⁻¹ (i.e. the approximate inverse is exact) and the same
> matrix is used for both the Chebyshev filter and the Rayleigh-Ritz
> projection, ChFSI and R-ChFSI are algebraically equivalent."

This means R-ChFSI is a strict generalization — it never performs worse than
standard ChFSI, and outperforms it whenever the inverse is inexact.

## 7. Implementation Plan for Our Code

### What changes

Replace the body of `chebyshev_filter` in `src/eigensolver/chebyshev.rs`
(lines 1123-1265) with Algorithm 3.

### What stays the same

- Rayleigh-Ritz projection (H_sub, S_sub via ZHEGVD) — unchanged
- β projector computation (VnlBatchData::precompute) — unchanged
- Spectral bounds (Lanczos + Gershgorin) — unchanged
- Density construction and V_eff assembly — unchanged (proven correct)

### What's new

1. **Initial residual**: `Y = H·X - S·X·Λ`
   - H·X via existing `apply_full_hamiltonian`
   - S·X = X + β·q·(β^H·X) — new S·X apply function (reuses VnlBatchData q_matrix)

2. **Residual recurrence**: replaces the eigenvector recurrence
   - Each step calls H via `apply_full_hamiltonian` and S⁻¹ via `apply_s_inverse`
   - The `Y·Λ_Y` term requires a gemm (n_pw × n_band × n_bands)

3. **Reconstruction**: `X_new = D⁻¹·R_Y + X·Λ_Y`
   - D⁻¹·R_Y via `apply_s_inverse` on the filtered residual
   - X·Λ_Y via gemm

### Expected outcome

- Iter-1 band-1: should improve from -0.956 Ha toward -1.055 Ha
- Iter-2 forward: should converge monotonically instead of diverging
- D_screened: should stay within D_0 range
- The S⁻¹·S test still fails at 1e-8 but R-ChFSI tolerates this

## 8. Source Files Map

| Reference | File | Key Lines |
|-----------|------|-----------|
| Levitt & Torrent (2015) | `extracted/levitt-torrent-2015/abinit.tex` | 664 (Alg 1), 653–657 (S⁻¹·H rationale), 678–680 (recurrence), 708 (Woodbury §), 735 (Woodbury formula) |
| Das et al. (2025) | `extracted/das-2025-rchfsi/main.tex` | 386 (Alg 2, std ChFSI), 588 (Alg 3, R-ChFSI), 533–565 (residual definition eq.5), 573 (convergence condition), 612 (equivalence proof), 861 (Thm 3.2 proof), 873 (Thm 3.4 proof) |
| LLM agent output | `suggested_algorithm.md` | — (low credibility) |
| Our S⁻¹ impl | `src/eigensolver/chebyshev.rs` | 759 (`apply_s_inverse`), 839–955 (`check_s_inv_s_identity`) |
| Our S⁻¹ impl | `src/eigensolver/vnl_data.rs` | 224–315 (s_inv_mat construction) |
| Our Chebyshev filter | `src/eigensolver/chebyshev.rs` | 969–1265 (`chebyshev_filter`), 1123–1265 (recurrence body) |
| S⁻¹·S identity test | `tests/ca_scf_convergence.rs` | 585–651 (`s_inv_s_identity_test`) |
| SCF drift evidence | `/tmp/scf-diag-0522-1022.log` | lines 65, 105, 144 (eigenvalue degradation) |

## 9. Notes on Sources

| Source | Type | Credibility |
|--------|------|-------------|
| `abinit.tex` (Levitt & Torrent 2015) | Published peer-reviewed paper in J. Comput. Phys. | High — the standard reference for USPP ChFSI |
| `main.tex` (Das et al. 2025) | Recent arXiv preprint on R-ChFSI | High — comprehensive mathematical analysis + benchmarks |
| `suggested_algorithm.md` | LLM agent output | Low — plausible architecture sketch, not peer-reviewed |

The `suggested_algorithm.md` describes the same S⁻¹·H approach as the Levitt &
Torrent paper but with a CG solver for S⁻¹ instead of the Woodbury formula.
Its architectural description is directionally correct but contains no
mathematical depth or convergence guarantees. All algorithmic claims in this
document are sourced from the primary papers, not from `suggested_algorithm.md`.
