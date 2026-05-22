# Investigation: Lanczos S⁻¹·H inner-product bug → eigenvalue drift

**Symptom**: R-ChFSI is implemented but SCF diverges after iter-1. Lanczos alpha
explodes in iter-2, b_up balloons from 20.8 to 134 Ha, Chebyshev filter damps
instead of amplifies, eigenvalues drift, D_screened explodes.

**Log**: `/tmp/scf-diag-0522-1344.log`

## Prior-note claim classification

| Claim | Source | Class | Admissible? |
|-------|--------|-------|-------------|
| CASTEP band-1 = -1.05502287 Ha | `Cu111_CO.bands` | **EXTERNAL** | Yes |
| CASTEP total energy = -24110.96665069 eV | `Cu111_CO.castep` line 326 | **EXTERNAL** | Yes |
| CASTEP F8 soft sum = 2.99359524940157e7 | F8-instrumented run stderr | **EXTERNAL** | Yes |
| CASTEP F8 aug sum = 5.14204469948334e7 | F8-instrumented run stderr | **EXTERNAL** | Yes |
| D_screened amax > 2× D_0 amax signals V_eff corruption | memory entry, `vnl_data.rs:175-184` | DERIVED (our diagnostic convention) | Conditionally — useful as monitoring signal, not as anchor |
| Density construction code is correct (ratio 1.000000 vs CASTEP F8) | `density_decomp_matches_castep_f8_same_inputs` test | EXTERNAL (reproducible against fixture) | Yes |
| S⁻¹ identity test shows 1.4% error | `s_inv_s_identity_test` | DERIVED | No — but indicates S⁻¹ is inexact |
| iter-1 Lanczos alpha [7.4-8.8] stable | log observation | DERIVED | No — but serves as baseline for comparison |
| iter-2 Lanczos alpha [9.2, 10.1, 96.2, 41.4, 79.8, 59.4] | log observation | DERIVED | No — is the symptom itself |

## Root cause (confirmed by code read)

`lanczos_upper_bound` (`src/eigensolver/chebyshev.rs:426-486`) estimates
λ_max(S⁻¹·H) for the generalized eigenproblem H·ψ = ε·S·ψ. It applies
S⁻¹ after each H·v and then uses the **standard dot product** for both
αⱼ = Re(⟨vⱼ, S⁻¹·H·vⱼ⟩) and β_{j+1} = ‖rⱼ‖₂.

**The bug**: S⁻¹·H is Hermitian only under the S-inner product ⟨x, y⟩_S = x†·S·y.
Under the standard dot product (S⁻¹·H)† = H·S⁻¹ ≠ S⁻¹·H because S⁻¹ and H don't
commute. Running Lanczos on a non-Hermitian operator with the standard inner
product is known to produce spurious eigenvalues (ghost convergence), and the
forced-symmetric tridiagonal matrix has wrong eigenvalue estimates.

### Why iter-1 works but iter-2 explodes

In iter-1, the fixture V_eff (loaded from `.pot_fmt`) is CASTEP's converged
potential. S⁻¹·H is well-conditioned and the non-Hermitian component is small
because the β-projector components of the Lanczos vectors are aligned with the
eigenspace. The standard-dot-product Lanczos happens to produce reasonable alpha
values.

In iter-2, our V_eff (built from our density) has different values at ion
centers. This changes V_NL through D_screening, and the non-Hermitian component
of S⁻¹·H becomes significant. Lanczos with the wrong inner product breaks down.

### Correct approach (S-inner-product Lanczos)

For operator A = S⁻¹·H with S-inner product:
- αⱼ = ⟨vⱼ, A·vⱼ⟩_S = ⟨vⱼ, S·S⁻¹·H·vⱼ⟩ = **⟨vⱼ, H·vⱼ⟩** (standard dot of v with H·v, NOT S⁻¹·H·v)
- β_{j+1} = ‖rⱼ‖_S = **√⟨rⱼ, S·rⱼ⟩** (S-norm, NOT L2-norm)

The fix is localized to ~40 lines in `lanczos_upper_bound`.

### Rejected hypotheses

- **Grid resolution mismatch** (wave-grid vs fine-grid V_eff for D_screened):
  Rejected. wave_grid == fine_grid == [54, 90, 90] for this fixture — no
  downsampling is involved. The D_screened computation uses the same grid for
  both V_eff and Q_nm, so the integral is self-consistent at wave-grid
  resolution. The D_screened explosion is a **consequence** of the wrong V_eff
  (from wrong eigenvalues → wrong density), not the cause.

- **Density normalization bug**: Ruled out by `density_decomp_matches_castep_f8_same_inputs`
  — density code is correct when fed CASTEP's wavefunctions (ratio 1.000000).

- **Spectral bounds (Gershgorin cap)**: The Gershgorin cap is a safety mechanism
  that limits damage from the Lanczos explosion, not the cause. Fixing Lanczos
  makes the cap unnecessary.

## Fix applied (2026-05-22)

### Fix 1: Lanczos S-inner-product (`chebyshev.rs:388-513`)

**Change 1 (line 390)**: Pre-allocate `sr` scratch buffer for S-norm computation, reused across all Lanczos steps.

**Change 2 (lines 425-440)**: S-normalize the Lanczos starting vector after L2-normalization.

**Change 3 (lines 456-470)**: Move alpha dot product to BEFORE `apply_s_inverse`. αⱼ = ⟨vⱼ, H·vⱼ⟩ (standard dot, not S⁻¹·H·vⱼ).

**Change 4 (lines 482-497)**: Compute beta as S-norm ‖r‖_S = √⟨r, S·r⟩ using pre-allocated `sr` buffer + `apply_s_times`.

**Result** (from `/tmp/scf-diag-0522-1517.log`): Alpha spike eliminated (96.2 → gone), but beta values 2-3× larger due to S-norm, and Gershgorin-capped b_up unchanged at 134 Ha. D_screened still explodes (183-455 Ha). **Fix was necessary but not sufficient** — the density deficit (164.5 vs 186 e⁻) from L2 Gram-Schmidt causes wrong V_eff at ion centers.

### Fix 2: S-Gram-Schmidt (`chebyshev.rs:1593-1666`)

Replaced L2-inner-product Gram-Schmidt with S-inner-product version:
1. For each column b, copy to scratch, compute S·col_b via `apply_s_times`
2. Compute initial S-norm: ‖col_b‖²_S = ⟨col_b, S·col_b⟩
3. For each j<b: dotⱼ = ⟨col_j, S·col_b⟩; col_b -= dotⱼ·col_j
4. ‖col_b_new‖²_S = ‖col_b_init‖²_S − Σⱼ|dotⱼ|² (no need to recompute S·col_b)
5. Normalize: col_b /= √(‖col_b_new‖²_S)

Uses 2 scratch buffers (gs_col, gs_s_col) and 1 `apply_s_times` call per column per pass.

### Verification

```bash
cargo test --release -p chemrust-scf --test ca_scf_convergence \
  -- fixed_point_matches_castep_energy --ignored --nocapture 2>&1 | tee /tmp/scf-diag-fix2.log
```

Expected after both fixes: iter-1 density integral 186 (not 164.5), iter-2 Lanczos stable, b_up < 25 Ha, D_screened stable at Cu sites.
