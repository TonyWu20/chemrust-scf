# Divergence Surface — iter2-divergence

**Slug:** `debug-20260523-1149-iter2-divergence`
**Date:** 2026-05-23

Per the user's "don't pre-bias" Step 4 directive: enumerate the full surface
from the skill's standard checklist plus project-specific candidates from
the iter-1→iter-2 trace. Each item is **RULED OUT** (cite anchor),
**PRIMARY CANDIDATE** (high prior + cheap to falsify), or **To be tested**.

Items cannot be ruled out by §10 memory alone — that memory predates §11's
symptom and may not have exercised the full surface.

## Standard categories from skill checklist

| Item | Status | Evidence / next action |
|------|--------|------------------------|
| Data layout / axis ordering (ψ ColumnDistributed handoff iter-N→iter-(N+1)) | **To be tested** | `chebyshev.rs:1700` exits with col-major (n_pw, n_bands) memory; RR wraps with `shape = [n_bands, n_pw]` at `rayleigh_ritz.rs:315-320`; next-iteration `scf.rs:503` H2D uploads. Verify wrap is consistent through full round-trip |
| Normalization / scaling (ZHEGVD X B-orthonormality → ψ_new S-orthonormality) | **To be tested** | If S_sub ≠ I entering RR (it isn't — has aug term), rotated ψ_new inherits whatever cuSOLVER ZHEGVD produced. cuSOLVER convention: ZHEGVD returns `X^H · B · X = I`, so rotated `ψ_new = ψ_row · X` should satisfy `ψ_new^H · S · ψ_new = X^H · S_sub · X = I` *iff* S_sub matches the operator's S exactly |
| Sign / direction (σ sign at `chebyshev.rs:1546`: `e/(λ_min − c)` is negative when λ_min < c) | **To be tested** | Verify against Das `main.tex:599`. Recurrence `σ_{k+1} = 1/(γ − σ_k)` and `γ = 2/σ_1` may have sign-dependent stability |
| Filter mode dispatch (A/B/C S⁻¹ gating at `chebyshev.rs:1578-1581, 1607-1611, 1687-1691`) | **To be tested** | Production uses Mode B (SinvHKeepHEig) post §10 fix. Iter-1 success may be mode-dependent; the iter-1 ladder showed Mode B winning by 5–10× over Mode A. But "wins on near-converged input" ≠ "wins on far-from-converged input" |
| Unit conversion at boundaries | **Ruled out** | All eigenvalues stay in Ha throughout the loop. No unit conversion sites between RR output, Chebyshev input, and density |
| Parser precision / density mixing | **Ruled out** | §8 `density_decomp_matches_castep_f8_same_inputs` validates density code to 0.0084% on a same-input controlled experiment. Density is not a source of iter-2 corruption when given correct ψ |
| Decomposition / parallel artifacts | **Ruled out** | Single-GPU run, no MPI. Only `RowDistributed`/`ColumnDistributed` phantom-type wrapping, which the type system enforces |
| Diagnostic comparison code | **To be tested** | Step 5 self-test required before any diagnostic output is read as evidence |

## Project-specific candidates (from iter-1→iter-2 trace)

| Item | Status | Evidence / next action |
|------|--------|------------------------|
| **H drifting between iterations** | **RULED OUT** | Lanczos α[0..3] identical to 4 decimals across iter-1→iter-2: [8.80, 7.50, 7.42, 7.40] → [8.79, 7.49, 7.41, 7.40]. The Hamiltonian's diagonal Rayleigh quotients on the Lanczos starting vector are stable. The subspace is drifting; H is not |
| **`b_low = eig[last]` semantic** (generalized eigenvalue used as bare-H or S⁻¹·H filter cutoff) | **PRIMARY CANDIDATE** | Layer C anchor C-SPECTRUM. Filter polynomial maps `[b_low, b_up]` for *some* operator. `eig[last]` is a generalized eigenvalue (eigenvalue of S⁻¹·H, modulo USPP convention). Mode B applies S⁻¹·H in Step 3, so this *should* be consistent. But Mode B also uses `h_eig = ⟨ψ, H ψ⟩` (bare-H Rayleigh quotient) for Λ shifts at `chebyshev.rs:1503-1538`. The mismatch: filter operator = S⁻¹·H, but Λ shifts derived from H. If ‖S − I‖ is non-trivial, this introduces O(‖S − I‖) error per Chebyshev step; over 8 steps with USPP at ~64% augmentation, the accumulated error could be exactly the 1.95 Ha overshoot observed |
| **R-ChFSI recurrence body correctness as a transform** | **PRIMARY CANDIDATE** | Layer B anchors B-T8-*. The body has never been black-box validated against analytic Chebyshev response on a clean H. Iter-1 success only proves it works for already-converged inputs (a near-identity case where the transform is approximately the identity polynomial regardless of internal arithmetic) |
| **R-ChFSI Λ shifts for non-converged inputs** | **PRIMARY CANDIDATE** | At `chebyshev.rs:1503-1538` we compute per-band `h_eig = ⟨ψ_b, H ψ_b⟩`. On near-converged ψ (iter-1 from CASTEP), `h_eig ≈ true H-eigenvalue`. On far-from-converged ψ (iter-2 from RR rotation of iter-1), `h_eig` is the Rayleigh quotient of *whatever ψ is*, not the H-eigenvalue. The recurrence at `chebyshev.rs:1614-1647` assumes `h_eig` represents the spectrum it's about to filter against. Layer B's α-sweep (B-T8-SWEEP) measures exactly this assumption |
| **S_sub assembly** (`rayleigh_ritz.rs:104-202`) — Q convention vs Woodbury Q | **To be tested** | `S_sub = ψ†ψ + Σ C_proj†·q·C_proj` uses `entry.q_matrix` (line 176). `apply_s_times` in `chebyshev.rs:934-1010` uses the same `entry.q_matrix`. They both pull from `VnlBatchEntry.q_matrix`. Verify identical field, no copies, no transposes between RR call site and Woodbury call site |
| **β_g phase / structure-factor at iter-2** | **To be tested** | `VnlBatchData::precompute` rebuilt fresh each iter (`scf.rs:510`); §10 G2 fix changed `Vec<f64>→Vec<CudaComplex>` for cross-ion imaginary parts. If V_eff-dependent D-screening (commit `85685b5`) introduces a per-iteration phase shift not exercised by iter-1's near-CASTEP V_eff, surfaces only at iter-2's larger V_eff range |
| **Gram-Schmidt S-inner product** (`chebyshev.rs:1700-1773` post-recurrence) | **To be tested** | Two passes of classical GS using `apply_s_times` with per-ion Q. If Q convention differs from RR S_sub, GS produces wrong metric and hands non-S-orthonormal subspace to RR, corrupting S_sub assembly there |
| **n_bands buffer (160 tracked, ~93 occupied)** | **To be tested** | Bands 94-160 are unoccupied tracked bands sitting *below* `b_low = eig[last]`. The filter amplifies them along with occupied; only RR's generalized eigensolve discriminates. If filter preferentially amplifies a band-1xx eigencomponent over band-93, the discriminator is broken upstream of RR |
| **§10 G2 cross-ion imaginary parts at iter-2 vnl_data** | **To be tested** | At iter-2, `VnlBatchData::precompute` uses iter-1's V_eff (passed via `v_eff_for_d` at `scf.rs:507-516`). If iter-1's V_eff has a different angular distribution than the fixture V_eff §10 fixed for, cross-ion B^H·B blocks could differ. Verify by computing `‖S⁻¹·S − I‖_∞` at iter-2 vnl_data; should still be ~3e-15 |
| **ψ ColumnDistributed → RowDistributed transpose at iter-2** | **To be tested** | iter-2's input ψ goes through `Gpu::from_host_with` at `scf.rs:503`. iter-1's output ψ went through `psi_new_dev` wrapping at `rayleigh_ritz.rs:315-320`. The host buffer between iterations is `ScfIteration.psi.data` (Vec<Complex64>) — verify the layout invariant survives the host-side ScfIteration handoff |

## Status summary

- **Ruled out:** 4 items (H drift, unit conversion, density code, MPI)
- **Primary candidates:** 4 items (R-ChFSI body correctness, R-ChFSI Λ shifts, b_low semantic, filter mode dispatch under non-converged inputs)
- **To be tested:** 8 items spanning data layout, normalization, sign convention, S_sub assembly, β phase, Gram-Schmidt, n_bands buffer, vnl_data iter-2 build, ψ layout round-trip

The primary candidates are testable cheaply by Layer B + Layer C + Step 7.0
ndeg=0 baseline. Step 7's order is exactly designed to bisect the primary
candidates first, then fall through to the to-be-tested items if all
primaries pass.
