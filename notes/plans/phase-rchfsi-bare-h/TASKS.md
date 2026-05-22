# Forensic TASKS.md — Bare-H R-ChFSI Experiment

ODD pattern reference: `/home/tony/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

## Declared Fixtures

| Path | Description |
|------|-------------|
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.castep_bin` | Cell, density on wave grid, eigenvalues |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check` | Wavefunctions + fine grid (155 MB, cached) |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.pot_fmt` | Reference V_eff on fine grid |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands` | Reference eigenvalues in Hartree |

## Success Criteria

| ID | Criterion | Source | Threshold |
|----|-----------|--------|-----------|
| SC-1 | `cargo check --workspace` passes | Build gate | zero errors |
| SC-2 | `cargo clippy --workspace -- -D warnings` passes | Lint gate | zero warnings |
| SC-3 | `iter2_v_eff_range_within_one_ha_of_iter1` stays green | `tests/ca_scf_convergence.rs:157-217` | \|iter-2 range − iter-1 range\| < 1.0 Ha |
| SC-4 | `fixed_point_matches_castep_energy` converges (not diverges) | `tests/ca_scf_convergence.rs:53-80` | band-1 iter-2 within 0.05 Ha of −1.055 Ha |
| SC-5 | `density_decomp_matches_castep_f8_same_inputs` stays green | `tests/ca_scf_convergence.rs:423-614` | soft/aug ratios within 1% of F8 |
| SC-6 | No `apply_s_inverse` calls from `chebyshev_filter` or `lanczos_upper_bound` | `src/eigensolver/chebyshev.rs` | grep for `apply_s_inverse` in filter + Lanczos |
| SC-7 | D_screened at iter-3 stays below 10 Ha (late-divergence gate) | Run-1010 log at `notes/open-followups.md:559` | D_screened iter-3 < 10 Ha |

## Decision Gates

| Outcome | Action |
|---------|--------|
| All gates green | Success — bare-H R-ChFSI converges. Commit. |
| Iter-2 gates green, SC-7 fails (iter-3 D_screened > 10 Ha) | Apply TASK-D4 (restore S⁻¹ in Step 4 only) |
| SC-4 fails (band-1 > 0.05 Ha error) | Bare-H hypothesis insufficient. Investigate second compounding bug. |
| SC-3 fails (> 1.0 Ha) | Density corruption. Revert and investigate. |

## Exploration Notes

- **Run 1010 precedent**: Bare-H standard ChFSI diverged at iter-3 (D_screened 50→319 Ha).
  This experiment tests whether R-ChFSI's residual formulation rescues bare-H filtering.

- **Evaluation feedback absorbed** (`~/.claude/plans/evaluate-the-plan-from-swift-river.md`):
  C1 (Lanczos mischaracterization): clarified — current Lanczos already operates on H;
  the change drops S-inner product, not an S⁻¹·H operator.
  C2 (mathematical inconsistency): resolved — committed fully to H-spectrum framing;
  all shifts/bounds use H-eigenvalues, not generalized. eps_cut is the only generalized
  quantity, used as an approximation (O(‖S − I‖) error, small for USPP).
  C3 (overstated evidence): reframed as experiment with explicit risk/decision gates.
  C4 (loose thresholds): tightened to 0.05 Ha band-1 gate, added iter-3 D_screened gate.
  C5 (Step 4 metric mismatch): acknowledged in comments, TASK-D4 pre-committed.

- **H-eigenvalue computation**: After Step 1 computes H·X, per-band dot products
  ⟨ψ_j, H·ψ_j⟩ are computed on GPU via cuBLAS. These approximate H-Rayleigh quotients
  (‖ψ_j‖_S = 1 ≈ ‖ψ_j‖_2 for USPP). Used in place of generalized eigenvalues throughout
  the recurrence for consistent H-spectrum framing.

- **Spectral bounds**: b_up = λ_max(H) from L2-Lanczos; b_low = eps_cut (generalized
  eigenvalue, approximate for H-spectrum cutoff); lambda_min = λ_min_tk(H) × 0.8.
  c = (b_up + b_low)/2; e = (b_up − b_low)/2; σ = e/(lambda_min − c).

---

## Task Groups

### Group D: Bare-H R-ChFSI experiment

**TASK-D1** — Lanczos: drop S-inner product, use standard L2 throughout

- **Kind**: `direct`
- **File**: `src/eigensolver/chebyshev.rs`
- **What**: The previous Lanczos (`lanczos_upper_bound`) operated on H but used S-inner
  product (S-norms for normalization and beta, `apply_s_inverse` in residual recurrence).
  TASK-D1 switches to standard L2-Lanczos on bare H:

  1. **Lines 425-440** (now deleted): Removed S-normalization block. The L2
     normalization at lines 405-423 already sets ‖v₀‖₂ = 1.
  2. **Lines 446-454** (now deleted): Removed `apply_s_inverse` call. The Lanczos
     residual recurrence no longer needs S⁻¹·H·v; `hv = H·v` stays as-is.
  3. **Lines 466-481** (rewritten): Both beta computations switched from S-norm
     (`‖r‖_S` via `apply_s_times` + dotc) to L2 norm (`‖r‖_2` via `dotc(hv, hv)`).
  4. **Line 390** (deleted): Removed `sr` scratch buffer (was pre-allocated for
     S-norm work, now unused).
  5. Alpha comment updated to reflect standard L2 dot product.

  Result: Lanczos returns `(λ_max(H), λ_min_tk(H), λ_max_tk(H))` — Ritz values of H
  under standard inner product.

- **Acceptance**: `cargo check --workspace` passes.

---

**TASK-D2** — Per-band H-eigenvalues and bare-H spectral bounds

- **Kind**: `direct`
- **File**: `src/eigensolver/chebyshev.rs`
- **What**:

  1. **Lines 1419-1447 (new)**: After Step 1 (after line 1418), added per-band
     H-eigenvalue computation:
     ```rust
     let h_eig: Vec<f64> = {
         let (ptr_psi_base, _sync_psi) = psi_input.device_ptr(stream);
         let (ptr_hpsi_base, _sync_hpsi) = hpsi_dev.device_ptr(stream);
         (0..n_bands).map(|b| {
             let off = (b * n_pw) as u64;
             // cublasZdotc_v2 on column b of psi_input vs hpsi_dev
             ...result.x
         }).collect()
     };
     ```
     Block-scoped to release `device_ptr` borrows before the recurrence loop
     (which mutably borrows `hpsi_dev` at Step 3).

  2. **Lines 1449-1455 (moved)**: Spectral parameter computation (e, c, σ, γ)
     moved from before-Step-1 to after-Step-1+h_eig. Uses `bounds.lambda_max`
     (bare-H Lanczos result) and `bounds.lambda_min` directly — no generalized
     RQ blending.

  3. **Step 2 Λ_Y** (line ~1497): Changed from `eig.iter().map(|l| sigma1_over_e * (l - c))`
     to `h_eig.iter().map(|l| sigma1_over_e * (l - c))`. Uses H-eigenvalues, not
     generalized eigenvalues.

  4. **Step 3 Λ_X** (line ~1549): Changed from `eig.iter()` to `h_eig.iter()`.
     All per-band shifts now use H-spectrum quantities.

  5. **Import**: Added `DevicePtr` to `use cudarc::driver::{...}` for immutable
     `device_ptr` access.

- **Acceptance**: `cargo check --workspace` passes. `cargo clippy --workspace -- -D warnings` passes.

---

**TASK-D3** — Bare-H recurrence (Step 3) and reconstruction (Step 4)

- **Kind**: `direct`
- **File**: `src/eigensolver/chebyshev.rs`
- **What**:

  1. **Step 3** (lines 1500-1511, now ~1505-1511): Deleted the `memcpy(buf_ry→buf_c)` +
     `apply_s_inverse(&mut buf_c)` sequence. Changed `apply_full_hamiltonian` to
     operate on `&buf_ry` directly (H·R_Y instead of H·S⁻¹·R_Y). Updated comments
     from "H·S⁻¹·R_Y" to "H·R_Y (bare-H operator, no S⁻¹ in recurrence)".

  2. **Step 4** (lines 1577-1584, now ~1577-1590): Removed `apply_s_inverse(&mut buf_a)`.
     Reconstruction becomes `X_new = R_Y + X·Λ_Y` (no S⁻¹). Added comment documenting
     the metric mismatch (R_Y carries implicit S-factor from Step 1; X·Λ_Y does not)
     and referencing TASK-D4 fallback.

  3. **Comment block** (lines 1366-1377): Updated R-ChFSI algorithm header to note
     "bare-H variant" and explain that spectral parameters are computed after Step 1.

  4. **`apply_s_inverse`** (lines 780-791): Added `#[allow(dead_code)]` attribute and
     preservation comment:
     ```
     /// Preserved as dead code: used by `check_s_inv_s_identity` diagnostic and
     /// potential TASK-D4 fallback (restore S⁻¹ in Step 4 only). Do not reap.
     ```

- **Acceptance**: `cargo check --workspace` passes. `cargo clippy --workspace -- -D warnings` passes.

---

**TASK-D4** (CONTINGENCY) — Restore S⁻¹ in Step 4 only

- **Kind**: `direct`
- **File**: `src/eigensolver/chebyshev.rs`
- **Trigger**: SC-7 fails (D_screened iter-3 > 10 Ha) OR SC-3 fails (V_eff range > 1.0 Ha)
- **What**: Restore `apply_s_inverse(&mut buf_a, vnl_data, ...)` call in Step 4
  (before the `band_scale_axpy` at line ~1584). Keep Step 3 on bare H (no S⁻¹ in loop).
- **Rationale**: S⁻¹ in reconstruction only (not in the loop) applies the Woodbury
  error once per iteration instead of ndeg times. Middle ground between full S⁻¹
  (diverges early) and no S⁻¹ (may diverge late).

---

## Dependency Order

```
TASK-D1 (L2-Lanczos)
    ↓
TASK-D2 (H-eigenvalues + bare-H bounds)
    ↓
TASK-D3 (Bare-H recurrence + reconstruction)
    ↓ (conditional)
TASK-D4 (S⁻¹ in Step 4 fallback)
```

## Verification Protocol

1. `cargo check --workspace` — must pass
2. `cargo clippy --workspace -- -D warnings` — must pass
3. Verify no S⁻¹ in hot path: `rg "apply_s_inverse" src/eigensolver/chebyshev.rs` —
   should show only: function definition (~line 789), `check_s_inv_s_identity` call
   (~line 1036), and Step 4 comment (~line 1588)
4. `cargo test --release -- --ignored iter2_v_eff_range_within_one_ha_of_iter1` — SC-3
5. `cargo test --release -- --ignored fixed_point_matches_castep_energy` — SC-4
6. `cargo test --release -- --ignored density_decomp_matches_castep_f8_same_inputs` — SC-5
7. Check D_screened at iter-3 in the test output — SC-7

## Key File References

| File | Lines | Purpose |
|------|-------|---------|
| `src/eigensolver/chebyshev.rs` | 16-19 | Import: added `DevicePtr` |
| `src/eigensolver/chebyshev.rs` | 362-519 | `lanczos_upper_bound` — L2-Lanczos (D1) |
| `src/eigensolver/chebyshev.rs` | 780-791 | `apply_s_inverse` — `#[allow(dead_code)]` + comment (D3) |
| `src/eigensolver/chebyshev.rs` | 1366-1387 | R-ChFSI comment block (D3) |
| `src/eigensolver/chebyshev.rs` | 1419-1447 | H-eigenvalue computation (D2) |
| `src/eigensolver/chebyshev.rs` | 1449-1455 | Spectral params moved after Step 1 (D2) |
| `src/eigensolver/chebyshev.rs` | 1487-1499 | Step 2 Λ_Y uses h_eig (D2) |
| `src/eigensolver/chebyshev.rs` | 1505-1511 | Step 3: H·R_Y directly (D3) |
| `src/eigensolver/chebyshev.rs` | 1549-1560 | Step 3 Λ_X uses h_eig (D2) |
| `src/eigensolver/chebyshev.rs` | 1577-1590 | Step 4: no S⁻¹, metric mismatch comment (D3) |
| `tests/ca_scf_convergence.rs` | 53-80 | `fixed_point_matches_castep_energy` |
| `tests/ca_scf_convergence.rs` | 157-217 | `iter2_v_eff_range_within_one_ha_of_iter1` |
| `tests/ca_scf_convergence.rs` | 423-614 | `density_decomp_matches_castep_f8_same_inputs` |
| `notes/open-followups.md` | 552-704 | Section 11 — evidence table |
| `~/.claude/plans/notes-open-followups-md-section-11-gleaming-wind.md` | — | Full plan with design rationale |
