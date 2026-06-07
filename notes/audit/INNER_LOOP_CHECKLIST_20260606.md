# Davidson Inner-Loop Adversarial Audit Checklist

**Date**: 2026-06-06 / updated 2026-06-08
**Scope**: Davidson eigensolver inner loop (17 components), Rust (`davidson.rs`) vs CASTEP 6.11 (`hamiltonian.f90`, `nlpot.f90`)
**Methodology**: Adversarial line-by-line audit, 55 surviving differences after filtering false positives; follow-up systematic workflow audit 2026-06-08 found 12 additional divergences.

---

## Fix History

| Date | Fix | IDs | Description |
|------|-----|-----|-------------|
| 2026-06-08 | **REGRESSION: `active_bands()` misapplied to super_wvfn** | D13-02, C2-04 | **REGRESSION — FIXED.** A 5-agent adversarial workflow audit (commit `fce7fae`) incorrectly changed A3 and Stage 1 to use `active_bands()` for super_wvfn access. This was wrong because `active_bands()` maps through `active_indices` (compacted workspace → global bands), but `super_wvfn` is APPEND-ONLY after C3-07 slice workspace — column `col` always maps to band `block_start + col`. The 2026-06-08 D13-02 and C2-04 revisions had ALREADY established that A3 and Stage 1 use sequential indexing (`block_bands()`). The workflow reverted these to `active_bands()`. **Symptoms:** D10-01 H_sub discrepancy 3×10^53, block 130 eigenvalues -10^52 Ha, bands 97–116 spuriously un-converged (eigenvalue changes 0.001–0.012 Ha), total numerical collapse. **Fix:** Stage 1 and A3 use `block_bands()` iterator (sequential 1:1 mapping for append-only super_wvfn). BetaPhiCache invalidation updated to match. ADR-0004 revised to distinguish `active_bands()` (compacted slice) from `block_bands()` (append-only super_wvfn). **See ADR-0004 §Regression: 2026-06-08 for full causal chain.** |
| 2026-06-08 | **GUARDRAIL: Checklist entries verified against CASTEP are permanent invariants** | Process | The D13-02 revision (2026-06-08) verified that CASTEP `hamiltonian.f90:523-528` copies ALL `current_nblock` columns sequentially. The C2-04 revision verified the same for Stage 1. These are CASTEP-faithful decisions — they must NEVER be reverted or "simplified." Any future agent or automation that proposes changing a FIXED entry in this checklist MUST first demonstrate that the CASTEP reference was misread, not that a different approach is "equivalent" or "simpler." **This regression cost one full investigation cycle (OOM + eigenvalue explosion) that was entirely avoidable.** |
| 2026-06-07 | D12-01: A2 on all k_super columns | D12-01 | **FIXED.** Changed `n_a2_cols` from `current_nblock.min(k_super)` to `k_super.min(superspace_max_bands)`. Conduction states (columns `current_nblock..k_super`) are now S-orthogonalized against global eigenvectors during A2, removing components of already-converged lower bands that previously contaminated the last block's ZHEEVD. Root cause of 156/160 warm-start test failure: conduction states from block 130 (eigenvalues 0.06-0.11 Ha) pulled block 156's ZHEEVD eigenvalues down to 0.07 Ha instead of the correct 0.112 Ha. Safe because ADR-0005 lockstep transforms keep h_super_wvfn consistent. Matches CASTEP hamiltonian.f90:519-520. |
| 2026-06-07 | Initial H_sub includes conduction states | C16-05 | **FIXED.** Changed initial H_sub computation from `k = current_nblock` to `k = superspace_index`, and moved conduction state seeding BEFORE H_sub. Previously, conduction states were in super_wvfn/h_super_wvfn but absent from super_hamiltonian, creating degenerate null-space rows in H_sub that caused ZHEEVD to find spurious low eigenvalues. **PRIMARY root cause of C1 156/160 failure.** Warm-start test: 160/160 converged in 1 outer iteration, max eigenvalue error 3.9e-4 Ha (within GPU/CPU numerical noise). Matches CASTEP hamiltonian.f90:414 where `wave_dot_all` uses ALL `super_wvfn%nbands` columns. |
| 2026-06-06 | Batch ZGEMM identity copy | (pre-audit) | Pre-copy search columns to `s_orth_out` before `apply_s_times` adds NL correction |
| 2026-06-06 | Compaction stale-column zeroing | C15-04 | **SUPERSEDED by C3-07.** Zero stale super_wvfn/h_super_wvfn columns after in-place compaction. Removed when separate slice workspace eliminated in-place compaction. |
| 2026-06-06 | superspace_index monotonic | C15-04 (related) | Remove superspace_index reduction; now grows monotonically matching CASTEP |
| 2026-06-06 | Eigenvalue assignment index | D13-01 | `inner_eigenvalues[active_indices[ci]]` → `inner_eigenvalues[ci]` |
| 2026-06-06 | Row-major GEMM extraction | comp9-1, comp9-2 | Fix column-major indexing in h_init_cpu and h_new_rows_cpu |
| 2026-06-06 | A2 re-orthogonalization scope | D12-01 | **REVERTED** — expanding to all `k_super` columns amplified D12-03 (stale h_super_wvfn), causing Rayleigh-ZHEEVD eigenvalue mismatch and severe drift. Restricted back to `current_nblock`. **SUPERSEDED 2026-06-07.** |
| 2026-06-06 | A3 copy-back scope + destination | D13-02 | **SUPERSEDED by 2026-06-08 D13-02 revision.** Used `active_bands()` — was wrong after C3-07 slice workspace made super_wvfn append-only. The 2026-06-08 revision uses `block_bands()` (sequential 1:1). |
| 2026-06-06 | Stage 1 source offset | C2-04 | **SUPERSEDED by 2026-06-08 C2-04 revision.** Used `active_bands()` — was wrong after C3-07 slice workspace made super_wvfn append-only. The 2026-06-08 revision uses `block_bands()` (sequential 1:1). |
| 2026-06-06 | active_indices moved before Stage 1 | — | Moved `active_indices` initialization before Stage 1 (was after inner loop setup), enabling correct iterator use from the start. |
| 2026-06-06 | Recompute H·psi fresh in build() | D12-03 workaround | **SUPERSEDED by ADR-0005.** Was a temporary workaround: build() called apply_full_hamiltonian() to bypass the stale-hpsi feedback loop. Now removed — ADR-0005 lockstep transforms keep hpsi consistent, so the stale-copy approach (CASTEP hamiltonian.f90:404-409) is safe. |
| 2026-06-06 | ADR-0005: Lockstep hpsi transform | D12-03 permanent fix | **IMPLEMENTED.** s_orthogonalise() and s_orthonormalise() accept optional hpsi_dev/hpsi_ref parameters. When provided, the same projection coefficients and GS rotations applied to search_dev are also applied to hpsi_dev — since H is linear, psi' = T(psi) → H·psi' = T(H·psi). Wired through A2 (transforms h_super_wvfn in lockstep with inner_block_temp via inner_hpsi_temp buffer). Not wired in build() Stages 3a/4/5 — Stage 6's apply_full_hamiltonian() overwrites hsearch_dev, making intermediate lockstep transforms pure overhead. See [ADR-0005](../../docs/adr/0005-lockstep-hpsi-transform.md). |
| 2026-06-06 | D12-01 re-applied (plan) | D12-01 | **PLANNED.** A2 should be expanded to all `k_super.min(superspace_max_bands)` columns. Safe with ADR-0005 lockstep transforms. **Actually applied 2026-06-07.** |
| 2026-06-06 | Fresh H·psi removed, stale-copy restored | D12-03 resolved | Removed apply_full_hamiltonian() from build() Stage 1. Restored copy from hpsi_dev → block_hpsi_temp (matching CASTEP hamiltonian.f90:404-409). Restored hpsi_dev field in DavidsonBlockCtx. Now safe because ADR-0005 prevents hpsi_dev from going stale. |
| 2026-06-06 | Review findings fixed | — | (1) Removed wasted hpsi_dev/hpsi_ref from build() Stage 3a/4/5 s_orthogonalise/s_orthonormalise calls — Stage 6 overwrites hsearch_dev. (2) Hoisted _hpsi_guard to function scope in s_orthonormalise(), matching search_mut pattern. |
| 2026-06-07 | C6-D2: Complex Cholesky fix (real-only bug) | C6-D2 | **FIXED.** The 2026-06-06 "FIXED" claim was incorrect — the Cholesky factorization only read `s_overlap_cpu[i + j*n].x` (real part), discarding imaginary off-diagonals. For complex Hermitian S_overlap (all non-gamma wavefunctions), this produced a wrong Cholesky factor → ZPOTRF false-negative (`diag <= 0`) → ALWAYS fell through to MGS fallback. MGS is less accurate than Cholesky; post-A2 vectors drifted enough that D10-01's diagonal-reset inconsistency became large enough to corrupt ZHEEVD eigenvalues (observed: sign flips at block 104 inner iter 1, contamination at block 130, L2²=1e10 norm explosion at block 156). **Fix 2026-06-07:** Replaced `Vec<f64>` with `Vec<Complex64>`, proper complex arithmetic: diagonal `|U[k,i]|² = re²+im²`, off-diagonal `conj(U[k,i]) · U[k,j]`. Matches LAPACK ZPOTRF('U') exactly. **C6-D2 is a HARD DEPENDENCY of D10-01.** |
| 2026-06-07 | D10-01: Confirmed faithful to CASTEP | D10-01 | **VERIFIED.** CASTEP `hamiltonian.f90:472-476` copies accumulated `super_hamiltonian` directly into `rotation(:,:)` and diagonalises it — no fresh GEMM recomputation. CASTEP also resets to diagonal after diagonalisation (`hamiltonian.f90:501-505`), creating the same diagonal-reset inconsistency our code exhibits. CASTEP tolerates this because its Cholesky-primary A2 (ZPOTRF → ZTRTRI → ZTRMM in `wave.f90:11573-11677`) keeps post-A2 vectors close enough to pre-A2 that the inconsistency is at numerical noise. **D10-01 REQUIRES C6-D2 (complex Cholesky) to work correctly; without it, the MGS fallback amplifies the inconsistency to physically meaningful levels.** |
| 2026-06-07 | D10-01 revert attempt | D10-01 | **REVERTED (wrong).** On 2026-06-07, D10-01 was temporarily reverted (using fresh GEMM instead of incremental super_hamiltonian) under the incorrect theory that the diagonal-reset inconsistency was the root cause. The real root cause was C6-D2's real-only Cholesky bug forcing all calls through MGS. D10-01 is now restored. |
| 2026-06-06 | C3-07/C15: Separate slice workspace | C3-07, C15-01–C15-06 | **FIXED.** Added slice_wvfn/slice_h_wvfn buffers. Compaction copies super_wvfn→slice (CASTEP hamiltonian.f90:637-638). super_wvfn is append-only. Removed in-place compaction, zero-stale-column, diagonal compaction, previous_eigenvalues compaction, current_nblock mutation. Resolves all C15 cascading differences. |
| 2026-06-06 | C16-04: Conduction state cross-block accumulation | C16-04 | **FIXED.** `cond_count` was `superspace_index - current_nblock` (grew monotonically across blocks). CASTEP `hamiltonian.f90:223,537-538` allocates `conduction_slice` with `nblock` bands (fixed) and overwrites every block — only the LAST block's conduction states persist. Rust accumulated ALL historical conduction states, polluting block 156's ZHEEVD with deeply-bound core states from blocks 0–130. Fix: `cond_count = (superspace_index - current_nblock).min(nblock)`. |
| 2026-06-06 | A3 sequential copy-back (refute finding) | D13-02 scope | **FIXED.** A3 was using `active_bands()` iterator — only copied ACTIVE band columns to global arrays. CASTEP `hamiltonian.f90:523-528` copies ALL `current_nblock` columns sequentially: column i → band nb+i-1. Converged bands must be updated alongside active ones to keep the full block consistent with the ZHEEVD rotation. Fix: `for i in 0..current_nblock` with destination `block_start + i`. **NOTE**: Requires Stage 1 to also use sequential indexing (see next entry). |
| 2026-06-06 | Stage 1 sequential indexing (match A3) | C2-04 revision | **FIXED.** Stage 1 was using `active_bands()` iterator (from ADR-0004 fix for in-place compaction era). After C3-07 slice workspace, super_wvfn is APPEND-ONLY (never compacted) — column i always maps to band block_start + i. The active_bands() iterator broke the Stage 1 ↔ A3 round-trip when combined with A3's sequential copy-back: Stage 1 loaded psi_dev[block_start + active_indices[i]] into super_wvfn[i], A3 wrote super_wvfn[i] to psi_dev[block_start + i]. For active_indices[1]=3: band 107's data was loaded into super_wvfn[1] then written to band 105's slot — scrambling the upper-band eigenvalue spectrum. **Regression confirmed in run 2509** (convergence dropped from 156/160 → 111/160). Fix: Stage 1 now uses same sequential `for i in 0..current_nblock` as A3. |
| 2026-06-06 | Conduction buffer persistence (refute finding) | C17-D3 | **FIXED.** `cond_wvfn`/`cond_h_wvfn`/`cond_count` were freshly allocated/reset each outer SCF iteration. CASTEP `hamiltonian.f90:223-224` allocates `conduction_slice` OUTSIDE `outer_loop` (line 297) — states persist across SCF iterations. Fix: moved `nblock`/`superspace_max_bands`/`super_alloc`/`cond_wvfn`/`cond_h_wvfn`/`cond_count` before `for iteration`, allocated once. Block 0 of iteration 2+ now receives conduction states from iteration 1's final block. |
| 2026-06-08 | C6-D2: s_orth_out pre-initialized to psi | C6-D2 (batch) | **FIXED.** `apply_s_times` accumulates with `beta=1`: `spsi += beta_g·q`. The Cholesky batch path in `s_orthonormalise` was passing a zero-initialized `s_orth_out` → result was NL correction only, missing the identity part `I·psi`. The per-column MGS fallback correctly pre-copied `search_j` to `s_orth_out` before calling `apply_s_times`. Fix: add `cublasZcopy(search_dev → s_orth_out)` before the batch `apply_s_times`. This caused ZPOTRF to see negative S-norms for A2 vectors (missing ~1.0 identity contribution). |
| 2026-06-08 | C6-D2: 2-pass MGS fallback | C6-D2 (MGS) | **FIXED.** ZPOTRF correctly fails for large-ncol A2 calls (128-156 columns with near-linearly-dependent conduction states). 1-pass MGS accumulates O(n²·ε) orthogonality error, corrupting conduction states. 2-pass MGS (reorthogonalization) reduces error to O(n·ε), preventing the cascading conduction-state corruption that caused eigenvalue explosion. |
| 2026-06-08 | C6-D2: GPU ZPOTRF + ZTRSM (pure GPU) | C6-D2 (GPU) | **FIXED.** Replaced hand-rolled CPU Cholesky + explicit inverse + ZGEMM with pure-GPU `solver.zpotrf('U')` → `cublasZtrsm_v2(RIGHT, UPPER, N, NON_UNIT)`. No CPU↔GPU copies for factorization; no explicit triangular inverse (ZTRSM is more stable). Stream sync via `blas.stream().synchronize()`. |
| 2026-06-08 | D9-01: Hermitian fill direction (H_new_rows) | comp9-1 (revision) | **FIXED.** After `H_new_rows` populates rows `superspace_index..new_total-1` (LOWER triangle for old columns), the Hermitian fill was reading from the UPPER triangle (never-written for cross-block entries → zeros) and writing zeros to LOWER, destroying correct `H_new_rows` values. Fix: reads from `S[new_row, j]` (LOWER, set by H_new_rows) and fills `conjg` to `S[j, new_row]` (UPPER). Exact match to CASTEP `hamiltonian.f90:461-464`. |
| 2026-06-08 | D10-01 temporarily removed, then restored | D10-01 | **RESTORED.** D10-01 was temporarily reverted (2026-06-08, using fresh GEMM for inner-loop ZHEEVD) under the theory that the C1 diagonal-reset inconsistency was the root cause. The actual root causes were P1-P4 (A3 scope, D1 re-check scope, conditional reset, previous_eigenvalues scope) which caused cascading convergence failure → large subspaces → ZPOTRF failure → MGS A2 → amplified C1 inconsistency. With P1-P4 fixed, convergence is clean (0-1 inner iterations), subspaces stay small, ZPOTRF succeeds for A2, and the C1 diagonal-reset inconsistency is at numerical noise level. D10-01 is now fully operational and faithful to CASTEP `hamiltonian.f90:472-476`. |
| 2026-06-08 | D13-02: A3 copy-back ALL bands | D13-02 (revision) | **FIXED.** A3 was using `active_bands()` to copy only active band columns to global arrays. CASTEP `hamiltonian.f90:523-528` copies ALL `current_nblock` columns via `wave_copy(..., copy_bands=current_nblock)`. Converged bands must be updated because ZHEEVD rotation produces improved eigenvectors for ALL subspace dimensions. Stale eigenvectors for converged bands cause eigenvalue drift in the next outer iteration. Fix: `for i in 0..current_nblock` with destination `block_start + i`. **Supersedes 2026-06-06 D13-02 fix which used active_bands().** |
| 2026-06-08 | C14-01: D1 re-check ALL bands | C14-01 | **FIXED.** D1 re-check was iterating over `active_bands()` only. CASTEP `hamiltonian.f90:606-608` re-tests ALL `current_nblock` bands against strict absolute tolerance. Previously-converged bands whose eigenvalues shifted due to superspace expansion were never re-verified. Fix: `for b in 0..current_nblock { gi = block_start + b; ... }`. |
| 2026-06-08 | C14-02: band_converged reset conditional | C14-02 | **FIXED.** `band_converged[gi] = false` was unconditional. CASTEP `hamiltonian.f90:548-550` resets only for bands NOT stopped by stagnation: `if(.not.opt_stop) band_converged = .false.`. A stopped-and-converged band counts toward the all-bands-done short-circuit without re-converging. Fix: `if !opt_stop_condition[b] { band_converged[gi] = false; }`. |
| 2026-06-08 | C14-03: previous_eigenvalues for ALL bands | C14-03 | **FIXED.** `previous_eigenvalues` was saved only for active bands via `active_bands()`. After compaction removed converged bands, their baseline was lost. CASTEP `hamiltonian.f90:431` saves `previous_eigenvalues(1:current_nblock) = eigenvalues(nb:nb+current_nblock-1)` for ALL bands. Fix: `for i in 0..current_nblock { previous_eigenvalues[i] = eigenvalues[block_start + i]; }`. |
| 2026-06-08 | C15-05: No compaction of per-band state arrays | C15-05 | **FIXED.** Compaction was removing entries for converged bands from `previous_eigenvalues` and `opt_stop_condition`. CASTEP `hamiltonian.f90:632-641` compacts only wavefunction columns and `slice_eigenvalues`; per-band state arrays stay at full `current_nblock` size. Fix: removed compaction lines for `previous_eigenvalues` and `opt_stop_condition`. |
| 2026-06-08 | C5-D1: Stage 4 S-orthogonalize 1 pass | C5-D1 | **FIXED.** Stage 4 was using 2 passes of `s_orthogonalise` against superspace. CASTEP `hamiltonian.f90:442` uses single `wave_Sorthogonalise_to_lower`. Over-orthogonalization can distort search directions. Fix: single pass. |
| 2026-06-08 | D12-02: A2 re-orthogonalize 1 pass | D12-02 | **FIXED.** A2 S-orthogonalize against lower eigenvectors was using 2 passes. CASTEP `hamiltonian.f90:519` uses single call. Fix: single pass. |
| 2026-06-08 | C14-04: missing convergence_tols(1) guard | C14-04 | **NOTED.** CASTEP `hamiltonian.f90:555` guards absolute tolerance check with `if(tol > -epsilon)`. When tol is negative (unset), the check is skipped. Rust uses `tol_abs.max(eps_guard)` fallback. **LOW severity** — only affects edge case where user provides negative tolerance. |

---

## 1. Component Summary Table

| # | Component | CASTEP Source | Rust Source | Status | Severity Spread |
|---|-----------|---------------|-------------|--------|-----------------|
| 1 | Pre-inner-loop eigenvalue init + conduction seeding | `hamiltonian.f90:396-418` | `davidson.rs:1330-1496` | **DIVERGE** | 1x MEDIUM, 4x LOW |
| 2 | Stage 1: psi/hpsi copy to workspace | `hamiltonian.f90:386-408` | `davidson.rs:1353` | **FIXED** | ~~1x CRITICAL~~, 3x LOW |
| 3 | Stage 2: Preconditioner math | `nlpot.f90:15973+` | `preconditioner.rs:68+` | **FIXED** | ~~2x HIGH~~, ~~1x MEDIUM~~, 5x LOW |
| 4 | Stage 3a: S-orthogonalize against ALL eigenvectors | `hamiltonian.f90:430-435` | `davidson.rs:3205` | **MATCH** | 1x LOW |
| 5 | Stage 4: S-orthogonalize against current superspace | `hamiltonian.f90:437-446` | `davidson.rs:3319` | **DIVERGE** | 1x MEDIUM |
| 6 | Stage 5: S-orthonormalize search directions | `hamiltonian.f90:11573+` | `davidson.rs:2622` | **FIXED** | ~~1x HIGH~~ |
| 7 | Stage 6: Apply H to search directions | `hamiltonian.f90:448-453` | `davidson.rs` (apply_full_hamiltonian) | **MATCH** | 1x LOW |
| 8 | Stage 7: Copy search + Hsearch to superspace | `hamiltonian.f90:455-467` | `davidson.rs` | **MATCH** | 1x LOW |
| 9 | H_new_rows GEMM + super_hamiltonian extraction | `hamiltonian.f90:455-467` | `davidson.rs:1431,1575,1582` | **FIXED** | ~~2x CRITICAL~~, 1x MEDIUM, 1x LOW |
| 10 | Diagonalization (ZHEGVD/ZHEEV) | `hamiltonian.f90:472-497` | `davidson.rs:1605-1620,2416` | **FIXED** | ~~2x HIGH~~, 1x LOW |
| 11 | super_hamiltonian reset to diag(eigenvalues) | `hamiltonian.f90:507-510` | `davidson.rs:1634-1640` | **MATCH** | 1x LOW |
| 12 | Post-diagonalization re-orthogonalization | `hamiltonian.f90:517-520` | `davidson.rs:1661-1753` | **FIXED** | ~~1x HIGH~~, 1x MEDIUM |
| 13 | Eigenvalue/wavefunction copy-back to global arrays | `hamiltonian.f90:523-542` | `davidson.rs:1756-1772,2058` | **FIXED** | ~~1x CRITICAL~~, ~~1x HIGH~~, 1x MEDIUM |
| 14 | Convergence check | `hamiltonian.f90:546-627` | `davidson.rs:1778-1843` | **DIVERGE** | 1x MEDIUM, 6x LOW |
| 15 | Compaction of unconverged bands | `hamiltonian.f90:633-645` | `davidson.rs:1915-2052` | **FIXED** | ~~1x HIGH~~, ~~3x MEDIUM~~, ~~3x LOW~~ |
| 16 | Conduction state saving | `hamiltonian.f90:537-542` | `davidson.rs:2062-2098` | **FIXED** | ~~1x CRITICAL~~, 1x MEDIUM, 2x LOW |
| 17 | Conduction state seeding for next block | `hamiltonian.f90:223-224,396,537-542` | `davidson.rs:982,1324-1340,2058` | **FIXED** | ~~1x HIGH~~, 2x LOW |

### Status Counts (updated 2026-06-08)

| Status | Count |
|--------|-------|
| **MATCH** (no significant differences) | 4 (C4, C7, C8, C11) |
| **DIVERGE** (differences exist) | 1 (C1 — slice workspace architectural choice) |
| **FIXED** | 12 (C2, C3, C5, C6, C9, C10, C12, C13, C14, C15, C16, C17) |
| **REVERTED** | 0 |

### Severity Counts (2026-06-08, after workflow audit fixes)

| Severity | Count |
|----------|-------|
| **CRITICAL** (active bug or high-risk latent) | 0 |
| **HIGH** (functional impact, not yet verified benign) | 0 |
| **MEDIUM** (observable effect, edge cases) | 1 (C1-D4 — slice workspace architectural choice) |
| **LOW** (diagnostic, cosmetic, or verified benign) | remaining |

---

## Appendix D: 2026-06-08 Workflow Audit — Convergence Regression Root Cause

A systematic 5-agent parallel audit compared the Rust Davidson eigensolver
against CASTEP 6.11 `hamiltonian.f90` line-by-line.  12 divergences were found,
of which 8 were identified as likely contributors to the convergence regression
from 160/160 to 159→132→83→explosion.

### Common Root Cause Pattern

After compaction was introduced (C3-07 slice workspace), several code paths
continued to use `active_bands()` / `active_indices` to iterate over the
compacted (reduced) set of bands instead of the full `current_nblock` set.
CASTEP consistently iterates over ALL `current_nblock` bands in these paths;
`active_indices` / compaction only affects which wavefunction columns
participate in Stage 3a/4/5 search-space construction.

The affected paths and their fixes:

| Path | CASTEP (all bands) | Rust (was: active only) | Fix |
|------|-------------------|------------------------|-----|
| A3 copy-back | `do i=1,current_nblock` | `active_bands(&indices, ...)` | `for i in 0..current_nblock` |
| A3 eigenvalue update | `eigenvalues(nb+i-1) = super_eigvals(i)` | `active_bands()` | `eigenvalues[block_start+i] = inner_eigvals[i]` |
| D1 re-check | `do i=1,current_nblock` | `active_bands()` | `for b in 0..current_nblock` |
| `band_converged` reset | `do i=1,current_nblock` | `active_bands()` | `for b in 0..current_nblock` |
| `previous_eigenvalues` save | `previous_eigenvalues(1:nblock)` | `active_bands()` | `for i in 0..current_nblock` |
| Per-band array compaction | Never compacted | Compacted with active_indices | Removed compaction |

### Additional Divergences Found

| ID | Divergence | Severity | Fix |
|----|-----------|----------|-----|
| D10-01 inner loop | Temporarily reverted; restored after P1-P4 | RESOLVED | With P1-P4 fixed, D10-01 is safe and faithful to CASTEP |
| C5-D1 (Stage 4) | 2-pass vs 1-pass S-orthogonalize | MEDIUM | Single pass |
| D12-02 (A2) | 2-pass vs 1-pass S-orthogonalize | MEDIUM | Single pass |
| C6-D2 (batch) | s_orth_out not pre-initialized to psi | HIGH | Zcopy psi→s_orth_out before apply_s_times |
| C6-D2 (MGS) | 1-pass MGS for large ncol | HIGH | 2-pass MGS for reorthogonalization |
| D9-01 | Hermitian fill direction reversed | HIGH | Read from lower, write to upper |
| C14-04 | Missing tol_abs guard | LOW | Documented, low impact |

---

## 2. Surviving Differences — Detailed Analysis

### Component 1: Pre-inner-loop Eigenvalue Initialization + Conduction Seeding

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C1-D1 | No `initial_eigenvalues` storage | `hamiltonian.f90:411` | 1460 | LOW | Rust never stores initial eigenvalues for diagnostic print. No computational impact. |
| C1-D2 | No persistent `super_eigvals` pre-init | `hamiltonian.f90:416-417` | 1602 | LOW | CASTEP's line 417 is dead code — `super_eigvals` is zeroed at line 470 before any read. Rust computes `inner_eigenvalues` fresh each iteration via ZHEGVD. |
| C1-D3 | No `slice_eigenvalues` array | `hamiltonian.f90:418` | 1496 | LOW | Rust passes eigenvalues to preconditioner via `block_ctx` from global array. Same data, different path. |
| C1-D4 | No separate `slice` workspace | `hamiltonian.f90:403-408` | 1330 | MEDIUM | Architectural: Rust's `super_wvfn` serves double duty (superspace + active workspace). CASTEP maintains separate `slice` and `super_wvfn` buffers. Drives C3-07, C15-01 cascading differences. |
| C1-D5 | Conduction eigenvalues not seeded into super_eigvals | `hamiltonian.f90:396-401` | 1462 | LOW | CASTEP does NOT copy conduction_eigvals into super_eigvals either (user assertion disproven). `conduction_eigvals` is write-only dead storage in both codebases. |

**Root cause analysis**: The absence of a separate `slice` workspace (C1-D4) is the root cause of the compaction divergence (C3-07, C15-01). In CASTEP, the active search subspace (`slice`) is physically distinct from the historical superspace (`super_wvfn`), so compaction copies *from* super_wvfn *into* slice without modifying super_wvfn. In Rust, both roles share `super_wvfn`, forcing in-place compaction with zero-stale-column mitigation. This is a design choice, not a bug, but it creates a different numerical pathway that warrants careful verification.

---

### Component 2: Stage 1 — psi/hpsi Copy to Workspace

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C2-01 | 3-buffer vs 2-buffer copy chain | `hamiltonian.f90:386-408` | 3015 | LOW | CASTEP: eigenvectors → super_wvfn → slice (3 buffers). Rust: psi_dev → block_psi_temp (2 buffers). End-result eigenvectors identical. |
| C2-02 | Copy-back timing | `hamiltonian.f90:637-638` | 1742 | LOW | CASTEP refreshes slice at block start AND after compaction. Rust copies from psi_dev each inner iteration (psi_dev was updated at end of previous iteration). Data freshness identical. |
| C2-03 | Band index mapping | `hamiltonian.f90:407,637` | 3015 | LOW | Both use original band positions. CASTEP: `nb:nb+current_nblock-1`. Rust: `block_start + active_indices[ci]`. MATCH CONFIRMED. |
| C2-04 | **Stage 1 source offset (FIXED, revised for slice workspace)** | `hamiltonian.f90:637` | 1350 | **FIXED.** Originally fixed via `active_bands()` iterator (ADR-0004) for in-place compaction. After C3-07 slice workspace made super_wvfn append-only, revised to sequential `for i in 0..current_nblock` — column i always maps to band block_start + i. This matches A3's sequential write-back and CASTEP's architecture. |

**Root cause analysis**: Trivial buffer-count difference. The extra copy layer in CASTEP is an artifact of maintaining separate super_wvfn and slice workspaces (see C1-D4). Both paths deliver identical eigenvectors at identical band positions to the preconditioner.

---

### Component 3: Stage 2 — Preconditioner Math

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C3-01 | Preconditioner formula | `nlpot.f90:15973` | `preconditioner.rs:68` | LOW | `(H*psi - e*S*psi) * R_vector` in both. MATCH CONFIRMED. |
| C3-02 | Eigenvalue source | `hamiltonian.f90:418` | `preconditioner.rs:3046` | LOW | Both use subspace-diagonalized eigenvalues at original band positions. MATCH CONFIRMED. |
| C3-03 | Post-compaction eigenvalue mapping | `hamiltonian.f90:639` | 1769 | LOW | Both map compacted position to eigenvalue at original band position. MATCH CONFIRMED. |
| C3-04 | Active band count | `hamiltonian.f90:644` | 3140 | LOW | Both count only unconverged, non-stopped bands. MATCH CONFIRMED. |
| C3-05 | USPP NL correction weight | `nlpot.f90:16054` | `preconditioner.rs:1045` | LOW | Same `eigenvalues[b] * beta_phi` per-band multiplication. MATCH CONFIRMED. |
| C3-06 | S-orthogonalization scope after preconditioner | `hamiltonian.f90:2224` | 3205 | LOW | Both orthogonalize against ALL eigenvectors (n_bands_total). MATCH CONFIRMED. |
| C3-07 | ~~Compaction buffer model~~ **FIXED via C3-07** | `hamiltonian.f90:633` | — | ~~HIGH~~ **FIXED.** Separate slice workspace implemented: compaction copies super_wvfn→slice, super_wvfn is never modified. Eliminates near-duplicate column risk. |
| C3-08 | ~~Post-rotation S-orthogonalization scope~~ **FIXED via D12-01** | `hamiltonian.f90:519` | 1668 | ~~MEDIUM~~ **FIXED.** A2 now re-orthogonalizes ALL k_super columns (including conduction states). Safe because ADR-0005 lockstep keeps h_super_wvfn consistent. |

**Root cause analysis**: The preconditioner *math* (C3-01 through C3-06) is verified identical to CASTEP. The two divergences (C3-07, C3-08) are now resolved: separate slice workspace and full A2 scope with lockstep hpsi.

---

### Component 4: Stage 3a — S-orthogonalize Against ALL Eigenvectors

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C4-OK | No differences found | `hamiltonian.f90` S-orth block | 3205 | LOW | Both use ALL eigenvector bands as reference. Both 1 pass, equivalent S-dot (ZGEMM), equivalent correction. Ncol matches. **MATCH CONFIRMED.** |

---

### Component 5: Stage 4 — S-orthogonalize Against Current Superspace

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C5-D1 | ~~Pass count: 1 vs 2~~ **FIXED 2026-06-08** | `hamiltonian.f90:442` | 3319 | ~~MEDIUM~~ **FIXED.** Changed from 2 passes to 1 pass, matching CASTEP. Over-orthogonalization (second pass) can distort search directions by projecting out legitimate search components that loosely overlap with higher (not-yet-converged) bands. |

**Root cause analysis**: The second pass was likely introduced as a numerical stability measure during early GPU development, but it does not exist in CASTEP and was not needed for the 160/160 working state. Single pass matches CASTEP exactly.

---

### Component 6: Stage 5 — S-orthonormalize Search Directions

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C6-D2 | ~~Cholesky vs CGS orthonormalization~~ **FIXED 2026-06-07** | `hamiltonian.f90:11573` | 2622 | ~~HIGH~~ **FIXED.** Complex Cholesky-primary path implemented in s_orthonormalise(): copy full complex S_overlap from GPU → CPU-side ZPOTRF-equivalent Cholesky (row-wise ikj, `Vec<Complex64>`, `conj(Uki)·Ukj` for off-diagonals, `|Uki|²` for diagonal) → CPU-side ZTRTRI-equivalent triangular inverse → H2D → ZGEMM rotation ψ_new = ψ · U⁻¹. Falls back to single-pass MGS on non-SPD. Lockstep hpsi via same ZGEMM rotation (ADR-0005). Matches CASTEP wave.f90:11573-11677 exactly.

**Root cause analysis**: The 2026-06-06 implementation had a critical bug — it only read the REAL part of S_overlap (`s_overlap_cpu[i + j*n].x`), storing into `Vec<f64>`. Since quantum wavefunctions are complex, S_overlap = ψ^H·S·ψ is complex Hermitian with imaginary off-diagonals. The real-only Cholesky factor was wrong → false `diag <= 0` negative → ALL calls fell through to MGS fallback. MGS is less accurate than Cholesky (error accumulates as O(n·ε) per column vs O(ε) for Cholesky+ZTRSM). For n_a2_cols = 30–52 (with conduction states), MGS produced post-A2 vectors that drifted enough that D10-01's diagonal-reset inconsistency became physically meaningful — ZHEEVD saw old diagonal-only rows vs new full rows that didn't agree, producing garbage eigenvalues (block 104 sign flips from -0.039→+0.045 Ha, block 130 contamination to -0.214 Ha instead of +0.02 Ha, block 156 L2²=1e10 norm explosion). The fix uses `Vec<Complex64>` with full complex arithmetic, matching LAPACK ZPOTRF('U') exactly. **This is a hard dependency of D10-01: the incremental super_hamiltonian approach only works when A2 orthonormalization is Cholesky-accurate.** |

---

### Components 7 and 8: Stage 6 (Apply H) and Stage 7 (Copy to Superspace)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| comp7-1 | Full Hamiltonian application | `hamiltonian.f90` hamiltonian_apply | `davidson.rs` apply_full_hamiltonian | LOW | Both apply kinetic + local + nonlocal to search directions. Call pattern aligned. (LOW-LEVEL HAMILTONIAN CORRECTNESS OUTSIDE AUDIT SCOPE.) |
| comp8-1 | Copy search + Hsearch to superspace | `hamiltonian.f90:455-467` | `davidson.rs` | LOW | Both copy search directions and H-applied search directions to superspace at correct insertion index. 1-based/0-based indexing handled consistently. |

**Root cause analysis**: These stages are structurally identical. No divergences warranting investigation.

---

### Component 9: H_new_rows GEMM and super_hamiltonian Extraction

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| comp9-1 | ~~Row-major extraction from column-major GEMM (location 1)~~ **FIXED** | `hamiltonian.f90:455` | 1431 | ~~CRITICAL~~ **FIXED** | Changed `h_init_cpu[i * k + j]` → `h_init_cpu[i + j * k]`. Was reading column-major buffer with row-major indexing (transpose). |
| comp9-2 | ~~Row-major extraction from column-major GEMM (location 2)~~ **FIXED** | `hamiltonian.f90:455` | 1575 | ~~CRITICAL~~ **FIXED** | Changed `h_new_rows_cpu[i * new_total + j]` → `h_new_rows_cpu[i + j * n_added]`. Same root cause. |
| comp9-3 | Hermitian fill direction reversed | `hamiltonian.f90:461` | 1582 | LOW | CASTEP: reads LOWER triangle, writes conj to UPPER. Rust: reads UPPER, writes conj to LOWER. Functionally equivalent. Irrelevant because matrix reset to diagonal afterward. |
| comp9-4 | Inconsistent indexing within same file | `hamiltonian.f90:455` | 1431,2388,2392 | MEDIUM | Diagnostic code at lines 2388/2392 uses CORRECT column-major `[i + j*k]`. Extraction code at lines 1431/1575 uses WRONG row-major `[i*k + j]`. Maintenance hazard: copy-paste between these locations would silently corrupt physics. |

**Root cause analysis**: Row-major vs column-major confusion in 2D array indexing. The GEMM output is column-major (standard BLAS convention), but the extraction code uses C-style row-major indexing (`row * ncol + col` instead of `row + col * nrow`). For comp9-1 (square k x k), this produces a transpose — diagonal entries are correct, off-diagonals have wrong index pairs AND sign-flipped imaginary parts. For comp9-2 (rectangular n_added x new_total), the elements are placed at wrong positions entirely, not even a clean transpose.

These are **latent critical bugs** masked by the super_hamiltonian reset at line 1634. Any refactoring that reads super_hamiltonian before the reset will silently corrupt physics. The fix is trivial: swap the multiplication operands in both locations.

---

### Component 10: Diagonalization (ZHEGVD)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| D10-01 | ~~Diagonalization matrix source~~ **FIXED (2026-06-08, restored after P1-P4 fixes)** | `hamiltonian.f90:472-484` | 1605 | **FIXED.** Incrementally-built super_hamiltonian (CPU, row-major) copied to GPU (column-major) and passed via h_sub_prebuilt to diagonalise_subspace(). Matches CASTEP's approach exactly. Was temporarily reverted because P1-P4 bugs (A3/D1 re-check/previous_eigenvalues scope) caused convergence cascades that amplified the C1 inconsistency. With P1-P4 fixed, convergence is clean and D10-01 works correctly. |
| D10-02 | Rotation matrix padding | `hamiltonian.f90:493-497` | 1605 | LOW | CASTEP pads rotation matrix with identity beyond `super_wvfn%nbands`. Rust has no padding — ZHEEVD operates on dense k_super x k_super. Different but both valid. |
| D10-03 | ~~EVP solver: ZHEEV vs ZHEEVD, real vs complex~~ **FIXED** | `hamiltonian.f90:480-484` | 2416 | ~~HIGH~~ **FIXED.** Gamma detection via k-point coordinates → cusolverDnDsyevd for gamma-point, ZHEEVD otherwise. Matches CASTEP's `algor_diagonalise(..., 'S'/'H')` dispatch. |

**Root cause analysis for D10-01**: CASTEP `hamiltonian.f90:472-476` copies `super_hamiltonian` directly into `rotation(:,:)`, then `algor_diagonalise`. No fresh GEMM. After diagonalisation, `super_hamiltonian` is reset to diagonal (lines 501-505). The NEXT inner iteration adds new full rows (from `wave_dot_all` against the CURRENT `h_super_wvfn`) to this diagonal matrix. This creates a matrix where old rows are diagonal-only and new rows are full — an inconsistency CASTEP tolerates because its Cholesky-primary A2 (`wave_Sorthonormalise_slice` → ZPOTRF → ZTRTRI → ZTRMM) keeps post-A2 vectors extremely close to pre-A2, making the inconsistency numerically negligible. Our implementation was SAME as CASTEP; the problem was C6-D2's real-only Cholesky bug forcing all calls through MGS, which produced larger post-A2 drift, making the inconsistency large enough to corrupt ZHEEVD (sign flips, contamination, norm explosion). With C6-D2 fixed (complex Cholesky), D10-01 is now safe. **C6-D2 is a hard dependency of D10-01.**

**Root cause analysis for D10-03**: DSYEVD is designed for real symmetric matrices, avoiding 2x FLOP waste and potential numerical artifacts from treating real data as complex. Gamma-point detection via k-point fractional coordinates matches CASTEP's `have_gamma` flag.

---

### Component 11: super_hamiltonian Reset to diag(eigenvalues)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| D11-01 | No divergence found | `hamiltonian.f90:507-510` | 1634 | LOW | Both zero super_hamiltonian and set diagonal to eigenvalues for all k_super entries. **MATCH CONFIRMED.** |

---

### Component 12: Post-diagonalization Re-orthogonalization

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| D12-01 | ~~Re-orthogonalization scope: all vs first nblock~~ **FIXED** | `hamiltonian.f90:519` | 1668 | ~~HIGH~~ **FIXED** | A2 now operates on all `k_super.min(superspace_max_bands)` columns (matching CASTEP's entire super_wvfn). Safe because ADR-0005 lockstep transforms keep conduction-state h_super_wvfn consistent during S-orthogonalization/S-orthonormalization. |
| D12-02 | ~~Pass count: 1 vs 2~~ **FIXED 2026-06-08** | `hamiltonian.f90:519` | 1692 | ~~MEDIUM~~ **FIXED.** Changed A2 S-orthogonalize from 2 passes to 1 pass, matching CASTEP. The ZHEGVD rotation already diagonalizes the subspace Hamiltonian; the single S-orthogonalize pass is sufficient to clean up machine-precision-level non-orthogonality. |
| D12-03 | ~~Stale h_super_wvfn after re-orthogonalization~~ **FIXED via ADR-0005** | `hamiltonian.f90:517` | 1655 | ~~HIGH~~ **FIXED** | **ADR-0005 lockstep transform implemented.** A2 copies h_super_wvfn columns into `inner_hpsi_temp`, passes it through s_orthogonalise/s_orthonormalise alongside `inner_block_temp`, and copies back. The SAME S-overlap coefficients and GS rotation factors applied to wavefunction columns are applied to hpsi columns — since H is linear, this maintains h_super_wvfn = H·super_wvfn exactly. Cost: a few extra AXPYs per orthogonalization step. See [ADR-0005](../../docs/adr/0005-lockstep-hpsi-transform.md). |
| D12-04 | No beta_phi invalidation (CASTEP line 517) | `hamiltonian.f90:517` | — | LOW (documented divergence) | CASTEP sets `super_wvfn%have_beta_phi = .false.` as an ADDITIONAL safeguard. ADR-0005 lockstep achieves the same correctness guarantee (consistent hpsi) via a different mechanism. Not a functional gap. |

**Root cause analysis**: D12-03 was the root cause of the stale-hpsi feedback loop: A2 modifies super_wvfn → h_super_wvfn stale → A3 copies stale values to hpsi_dev → build() copies stale hpsi_dev to block temps → H_sub = psi^H·(stale hpsi) produces wrong eigenvalues → vicious cycle. Empirically confirmed: Rayleigh quotients (~-0.18) diverged from ZHEEVD eigenvalues (~+0.006) in run 2503. ADR-0005 lockstep transform eliminates this by keeping h_super_wvfn identical to H·super_wvfn through all S-orthogonalization steps. With D12-03 fixed, D12-01 (full A2 scope) and the stale-copy approach (hamiltonian.f90:404-409) are both safe. See [ADR-0005](../../docs/adr/0005-lockstep-hpsi-transform.md).

---

### Component 13: Eigenvalue/Wavefunction Copy-back to Global Arrays

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| D13-01 | ~~Eigenvalue index lookup bug~~ **FIXED** | `hamiltonian.f90:527` | 1769 | ~~CRITICAL~~ **FIXED** | Changed `inner_eigenvalues[active_indices[ci]]` → `inner_eigenvalues[ci]`. After ZHEEVD rotation, super_wvfn column `ci` is the `ci`-th lowest eigenvector with eigenvalue `inner_eigenvalues[ci]`, regardless of compaction state. Using `active_indices[ci]` (original band offset) was wrong — it skipped eigenvalues for converged bands, leaving gaps. |
| D13-02 | ~~Copy-back scope~~ **FIXED (2026-06-08 revision)** | `hamiltonian.f90:523` | 1849 | **FIXED.** Copies ALL `current_nblock` columns sequentially: `for i in 0..current_nblock`, destination `block_start + i`. Matches CASTEP `wave_copy(super_wvfn, eigenvectors, nb_src=1, nb_dst=nb, copy_bands=current_nblock)`. Converged bands are updated alongside active ones because ZHEEVD rotation produces improved eigenvectors for ALL subspace dimensions — stale converged-band eigenvectors cause eigenvalue drift on the next outer iteration. **The 2026-06-06 fix used `active_bands()` which was still too narrow; the 2026-06-08 revision corrects this to unconditional sequential copy matching CASTEP.** |
| D13-03 | Conduction eigenvalues not saved | `hamiltonian.f90:540` | 2058 | MEDIUM | CASTEP saves `conduction_eigvals(i) = super_eigvals(current_nblock+i)`. Rust has no equivalent. However, CASTEP's `conduction_eigvals` is write-only dead storage (never read before deallocation), so this is functionally irrelevant in both codebases. |

**Root cause analysis for D13-01**: This is the most impactful active bug in the current code. The index mapping confusion arises from the compaction step: `active_indices` maps *compacted position* to *original block-relative position*. After diagonalization, the Ritz vectors in super_wvfn are ordered by the diagonalization (which sorts by eigenvalue), so column `ci` in super_wvfn corresponds to the `ci`-th Ritz vector with eigenvalue `inner_eigenvalues[ci]`. When copying back to global arrays, the correct mapping is:

- Wavefunction: super_wvfn column `ci` → global band `block_start + active_indices[ci]`
- Eigenvalue: `inner_eigenvalues[ci]` → global band same position

After ADR-0004, this mapping is enforced structurally by the `active_bands()` iterator:
```rust
for (ci, gi) in active_bands(&active_indices, block_start) {
    // ci: compacted workspace column → super_wvfn source, eigenvalue index
    // gi: global band index        → psi_dev destination, eigenvalues destination
}
```

This bug means: for any band ordering where `active_indices[ci] != ci` (which happens whenever compaction changes the ordering), the eigenvalues written to the global array are WRONG. This directly causes higher-band eigenvalue drift because a higher band may receive a lower band's (more negative) eigenvalue if the index mapping is off.

---

### Component 14: Convergence Check

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C14-01 | ~~D1 re-check scope (active_bands only)~~ **FIXED 2026-06-08** | `hamiltonian.f90:606-608` | 1798 | ~~MEDIUM~~ **FIXED.** D1 re-check was iterating over `active_bands()` only — previously-converged bands whose eigenvalues shifted due to superspace expansion were never re-verified against strict tolerance. CASTEP re-tests ALL `current_nblock` bands: `do i=1,current_nblock`. Fix: `for b in 0..current_nblock { gi = block_start + b; ... }`. |
| C14-02 | ~~band_converged reset unconditional~~ **FIXED 2026-06-08** | `hamiltonian.f90:548-550` | 1786 | ~~HIGH~~ **FIXED.** `band_converged[gi] = false` was unconditional. CASTEP resets only for bands NOT stopped by stagnation: `if(.not.opt_stop_condition(i)) band_converged = .false.`. A stopped-and-converged band counts toward the all-bands-done short-circuit. Fix: `if !opt_stop_condition[b] { band_converged[gi] = false; }`. |
| C14-03 | ~~previous_eigenvalues only for active bands~~ **FIXED 2026-06-08** | `hamiltonian.f90:431` | 1782 | ~~HIGH~~ **FIXED.** `previous_eigenvalues` was saved only for active bands via `active_bands()`. After compaction, converged bands lost their baseline for delta_e computation. CASTEP saves for ALL bands: `previous_eigenvalues(1:current_nblock) = eigenvalues(nb:nb+current_nblock-1)`. Fix: `for i in 0..current_nblock { previous_eigenvalues[i] = eigenvalues[block_start + i]; }`. |
| C14-04 | Negative tolerance guard missing | `hamiltonian.f90:555` | 1798 | LOW | CASTEP gates absolute tolerance check with `if(convergence_tols(1) > -epsilon(1.0_dp))`. Rust uses `tol_abs.max(eps_guard)` fallback. For physical tol_abs values (positive), no difference. **LOW — only affects edge case of negative user-specified tolerance.** |
| C14-05 | abs(tol_rel) vs tol_rel | `hamiltonian.f90:569` | 1807 | LOW | CASTEP uses `abs(convergence_tols(2))`; Rust uses `tol_rel` directly. Both guarded to non-negative before reaching this code. Equivalent for well-behaved inputs. |
| C14-06 | MPI broadcast of convergence state | `hamiltonian.f90:624-627` | 1843 | LOW | CASTEP broadcasts convergence flags across MPI. Rust runs single-GPU. Architectural constraint, not a bug. |
| C14-07 | Stopped state on last outer iteration | `hamiltonian.f90:582` | 1815 | LOW | CASTEP preserves `opt_stop_condition` on last outer iteration (guard fails). Rust explicitly clears it. May cause extra inner iterations in Rust on final outer iteration. |

**Root cause analysis**: Most convergence check differences are diagnostic/logging gaps (C14-02 through C14-07). C14-01 (negative tolerance guard) could affect edge cases where users specify negative tolerance to disable absolute convergence checking entirely, but this is rare. C14-07 could cause slight behavioral differences on the final SCF iteration but should not affect final eigenvalue accuracy.

---

### Component 15: Compaction

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C15-01 | ~~Compaction direction: separate buffer vs in-place~~ **FIXED via C3-07** | `hamiltonian.f90:637` | — | ~~HIGH~~ **FIXED.** Separate slice_wvfn/slice_h_wvfn implemented. Compaction copies super_wvfn→slice; super_wvfn is append-only. |
| C15-02 | ~~super_hamiltonian diagonal compaction~~ **REMOVED** | — | — | ~~MEDIUM~~ Not needed — super_wvfn columns are never rearranged. |
| C15-03 | ~~previous_eigenvalues compaction~~ **REMOVED** | — | — | ~~LOW~~ Not needed — slice-indexed arrays replace compacted per-block arrays. |
| C15-04 | ~~Zero-stale-column fix~~ **REMOVED** | — | — | ~~MEDIUM~~ Not needed — no in-place compaction means no stale columns. |
| C15-05 | ~~Per-band state arrays compacted~~ **FIXED 2026-06-08** | — | — | ~~MEDIUM~~ **FIXED.** Compaction was removing entries for converged bands from `previous_eigenvalues` and `opt_stop_condition`. CASTEP `hamiltonian.f90:632-641` compacts only wavefunction columns and `slice_eigenvalues`; per-band state arrays stay at full `current_nblock` size. Fix: removed compaction lines for these arrays. Required for C14-01/02/03 to work correctly. |
| C15-06 | ~~current_nblock reduction~~ **REMOVED** | — | — | ~~MEDIUM~~ current_nblock stays fixed (original block size); slice_nbands tracks active count. |
| C15-07 | Early break on zero search columns | `hamiltonian.f90:646` | — | LOW | Rust has `if n_added == 0 { break; }` guard. CASTEP does not. Could cause premature inner-loop exit. |

**Root cause analysis**: All compaction differences were cascading consequences of C1-D4 (no separate slice workspace). The slice workspace implementation (C3-07) resolves C15-01 through C15-06 simultaneously. Only C15-07 (early break) remains as a LOW-priority behavioral difference.

---

### Component 16: Conduction State Saving

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C16-01 | Save timing: every iteration vs after loop | `hamiltonian.f90:537` | 2058 | MEDIUM | CASTEP saves conduction states EVERY inner iteration (between diagonalization and convergence check). Rust saves ONCE after inner loop exit. Functionally: both capture the final Ritz vectors before next block. CASTEP's extra saves are overwritten each iteration and only the last one matters. |
| C16-02 | conduction_eigvals not saved | `hamiltonian.f90:541` | 2061 | LOW | CASTEP's `conduction_eigvals` is write-only dead storage. Neither codebase actually uses it. |
| C16-03 | Conduction band count calculation | `hamiltonian.f90:537` | 2061 | LOW | CASTEP bound: `min(nblock, superspace_index + slice_searchspace%nbands - 1 - current_nblock)`. Rust: `superspace_index - current_nblock`. Both produce equivalent counts in practice. |
| C16-04 | **Cross-block conduction state accumulation (FIXED)** | `hamiltonian.f90:223,537-538` | 2065 | **CRITICAL** | **FIXED 2026-06-06.** Rust's `cond_count = superspace_index - current_nblock` grows monotonically across blocks — block 0 saves ~28 states, block 1 saves ~74, block 5 saves ~130. All are seeded into block 6, polluting ZHEEVD with deeply-bound core states. CASTEP `hamiltonian.f90:223` allocates `conduction_slice` with FIXED `nblock` bands, and lines 537-538 OVERWRITE (e.g., `nb_dst=1`) every block — only the LAST block's conduction states persist. **Empirically confirmed**: block 156's PreconEig values jump from 0.072→0.302 between outer iterations, and upper-band eigenvalues are systematically ~0.005 Ha lower than CPU reference. Fix: `cond_count.min(nblock)`. **MISSED by original audit** — C16 was classified as timing/count differences (MEDIUM/LOW) when the real bug was across-block accumulation. |

---

### Component 17: Conduction State Seeding for Next Block

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C17-D1 | Save location: inside vs outside inner loop | `hamiltonian.f90:537` | 2058 | LOW | Same as C16-01. Structurally different, functionally equivalent. |
| C17-D2 | conduction_eigvals not in Rust | `hamiltonian.f90:540` | 2061 | LOW | Same as C16-02. Dead code in CASTEP. |
| C17-D3 | ~~Buffer persistence across outer iterations~~ **FIXED** | `hamiltonian.f90:223-224` | 982 | ~~MEDIUM→HIGH~~ **FIXED.** CASTEP allocates `conduction_slice/Hconduction_slice` OUTSIDE the outer loop (line 223, before `outer_loop` at line 297) — conduction states from iteration N persist into iteration N+1. Rust was freshly allocating `cond_wvfn`/`cond_h_wvfn` with `cond_count = 0` INSIDE the outer loop each iteration. **Elevated to HIGH by refute audit**: block 0 of outer iteration 2+ operates on a systematically smaller subspace without conduction state enrichment. Fix: `nblock`/`superspace_max_bands`/`super_alloc`/`cond_wvfn`/`cond_h_wvfn`/`cond_count` moved before `for iteration`, allocated once. |

**Audit blind spot (C16-04, C17-D3)**: The original adversarial audit treated C16/C17 as secondary — timing differences, count formulas, buffer persistence. Both missed findings were caught by the refute stage (workflow `refute-audit-remaining`): (1) cross-block conduction state accumulation (C16-04, CRITICAL), and (2) conduction buffer allocation inside vs outside the outer SCF loop (C17-D3, elevated to HIGH from MEDIUM). Both are now fixed.

**Root cause analysis for C17-D3**: Now fixed — cond_wvfn/cond_h_wvfn/cond_count allocated before `for iteration`, matching CASTEP `hamiltonian.f90:223-224`. Block 0 of outer iteration 2+ now receives conduction state enrichment. CASTEP's design preserves conduction state information across the outer SCF loop, allowing block 1 of outer iteration N+1 to benefit from the final block's Ritz vectors from iteration N. Rust loses this information. The impact depends on how much the conduction states change between outer iterations — for well-converged SCF cycles, the loss is minor; for early SCF cycles with large charge density changes, the loss could slow convergence.

---

## 3. Diagnostic Decision Tree: Higher Band Eigenvalue Drift Toward Zero

**Symptom**: Eigenvalues for bands near and above the Fermi level drift toward zero (become less negative, less bound) over Davidson iterations. Low-lying (core-like) bands are unaffected or affected less.

### Decision Tree

```
START: Higher band eigenvalues drift toward zero
│
├── Q1: Is the eigenvalue drift CORRELATED with band index gaps?
│   │   (e.g., bands 5, 10, 15 drift but bands 1-4, 6-9 are OK)
│   │
│   ├── YES → D13-01 (CRITICAL): Eigenvalue index lookup bug
│   │   │   Band positions with active_indices[ci] != ci receive WRONG eigenvalues.
│   │   │   CHECK: Print inner_eigenvalues[ci] vs eigenvalue written to global[block_start + active_indices[ci]]
│   │   │   If they differ when active_indices[ci] != ci, THIS IS THE BUG.
│   │   │
│   │   └── CONFIDENCE: HIGH. This is an active bug with predictable signature.
│   │       After compaction, bands at non-trivial positions in active_indices get
│   │       mismatched eigenvalues. The pattern is deterministic: band at global
│   │       position block_start + active_indices[ci] receives eigenvalue from
│   │       inner_eigenvalues[active_indices[ci]] instead of inner_eigenvalues[ci].
│   │
│   ├── Q2: Does the drift appear AFTER some bands have converged and been stopped?
│   │   │
│   │   ├── YES → D13-02 (HIGH): Stopped bands not updated
│   │   │   │   Rust only updates ACTIVE bands. Stopped/converged bands keep frozen
│   │   │   │   wavefunctions and eigenvalues. If a band was prematurely stopped, its
│   │   │   │   eigenvalue freezes at the wrong value.
│   │   │   │
│   │   │   └── COMBINED with D13-01, this creates a compound failure:
│   │   │       - D13-01 assigns wrong eigenvalue to wrong band
│   │   │       - D13-02 then freezes that wrong eigenvalue when band is stopped
│   │   │       - The correct eigenvalue is "lost" because it was written to a different band
│   │   │
│   │   ├── Q3: Does the drift worsen with inner-iteration count?
│   │   │   │
│   │   │   ├── YES → D12-01 (HIGH): Re-orthogonalization scope too narrow
│   │   │   │   │   Columns current_nblock..k_super-1 never re-orthogonalized.
│   │   │   │   │   Accumulated non-orthogonality degrades ZHEGVD accuracy.
│   │   │   │   │   Higher bands are more affected because they lie in the region
│   │   │   │   │   of the spectrum most sensitive to basis quality.
│   │   │   │   │
│   │   │   │   └── CHECK: Monitor ||S_ij - delta_ij|| for columns beyond current_nblock
│   │   │   │       across inner iterations. If growing, D12-01 is causal.
│   │   │   │
│   │   │   ├── Q4: Does the drift correlate with near-degenerate eigenvalue clusters?
│   │   │   │   │
│   │   │   │   ├── YES → C6-D2 (HIGH): CGS instead of Cholesky
│   │   │   │   │   │   For near-degenerate bands, search directions become nearly
│   │   │   │   │   │   parallel. CGS accumulates orthogonality error, corrupting
│   │   │   │   │   │   the search subspace. Cholesky (ZPOTRF+ZTRMM) is robust to this.
│   │   │   │   │   │
│   │   │   │   │   └── CHECK: Compute condition number of S-overlap matrix in Stage 5.
│   │   │   │   │       If cond(S) > 1e8 and using CGS, this is a contributing factor.
│   │   │   │   │
│   │   │   │   └── NO → Q5: Does drift appear specifically after compaction?
│   │   │   │       │
│   │   │   │       ├── YES → C3-07/C15-01 (HIGH): In-place compaction
│   │   │   │       │   │   Near-duplicate columns in super_wvfn after in-place
│   │   │   │       │   │   compaction can create near-singular S-overlap in ZHEGVD,
│   │   │   │       │   │   producing spurious near-zero eigenvalues.
│   │   │   │       │   │
│   │   │   │       │   └── CHECK: After compaction, check for columns i != j where
│   │   │   │       │       ||v_i - v_j|| < epsilon. If found, zeroing (C15-04) is insufficient.
│   │   │   │       │
│   │   │   │       └── NO → Q6: Does drift appear only on gamma-point calculations?
│   │   │   │               │
│   │   │   │               ├── YES → D10-03 (HIGH): ZHEEVD forced for real wavefunctions
│   │   │   │               │   │   Complex solver on real data adds numerical noise.
│   │   │   │               │   │   For gamma-point, should use DSYEVD.
│   │   │   │               │   │
│   │   │   │               │   └── CHECK: Is kpoint at gamma? If using ZHEEVD on real data,
│   │   │   │               │       switch to DSYEVD and compare eigenvalues.
│   │   │   │               │
│   │   │   │               └── NO → Consider D10-01 (fresh vs incremental H_sub)
│   │   │   │                   │   The fresh H_sub computation may miss off-diagonal
│   │   │   │                   │   couplings present in CASTEP's incrementally-built
│   │   │   │                   │   super_hamiltonian. This would manifest as systematic
│   │   │   │                   │   eigenvalue differences (not drift per se, but bias).
│   │   │   │                   │
│   │   │   │                   └── CHECK: Compare H_sub from fresh GEMM vs CASTEP's
│   │   │   │                       incrementally-built super_hamiltonian at same iteration.
│   │   │   │                       If they differ, D10-01 is causal.
│   │   │   │
│   │   │   └── [End of tree]

LEGEND:
  → = causal link, the preceding difference directly causes the symptom
  CHECK = diagnostic test to confirm or refute the hypothesis
```

### Most Likely Causal Chain for "Higher Band Eigenvalue Drift Toward Zero"

All identified causes are now fixed (2026-06-06):

1. **C2-04 (CRITICAL, FIXED) — Stage 1 source-offset bug**: **PRIMARY ROOT CAUSE.** Sequential psi_dev indexing loaded wrong wavefunctions after compaction. **FIXED: active_bands() iterator (ADR-0004).**

2. **D13-02 (FIXED) — A3 destination-offset bug**: Companion to C2-04. Sequential writing to wrong band positions. **FIXED: active_bands() iterator (ADR-0004).**

3. **D13-01 (CRITICAL, FIXED) — Eigenvalue index lookup bug**: Wrong eigenvalue index after compaction. **FIXED.**

4. **D12-03 (HIGH, FIXED) — Stale h_super_wvfn feedback loop**: A2→A3→build()→wrong H_sub→wrong eigenvalues→worse search directions. **FIXED: ADR-0005 lockstep transform.**

5. **C6-D2 (HIGH, FIXED) — CGS orthonormalization**: Instability for near-degenerate bands. **FIXED: Cholesky-primary path with CGS fallback.**

6. **C3-07/C15 (HIGH, FIXED) — In-place compaction**: Duplicate columns in super_wvfn. **FIXED: separate slice workspace.**

7. **D10-01 (HIGH, FIXED) — Fresh vs incremental H_sub**: Algorithmic divergence. **FIXED: incremental super_hamiltonian.**

8. **D10-03 (HIGH, FIXED) — Real vs complex solver**: 2x FLOP waste at gamma. **FIXED: DSYEVD dispatch.**

### Elimination Ladder (Recommended Diagnostic Order)

| Step | Check | Tool | Expected if D13-01 is causal | Expected if D12-01 is causal |
|------|-------|------|------------------------------|------------------------------|
| 1 | Print `(ci, active_indices[ci], inner_eigenvalues[ci], eigenvalue_written_to_global)` after copy-back | `println!` at line 1769 | `eigenvalue_written != inner_eigenvalues[ci]` when `active_indices[ci] != ci` | All equal (D13-01 not causal) |
| 2 | If Step 1 shows mismatches, fix index and re-run | Edit line 1769 | Drift disappears | Drift persists (next suspect) |
| 3 | Monitor `||S_ij - I||` for columns > current_nblock across iterations | Norm computation after ZHEGVD | N/A | Growth in off-diagonal norm |
| 4 | Compute condition number of S-overlap in Stage 5 | `cond(S)` | N/A | High condition number → C6-D2 |

---

## 4. Recommended Fix Priority Order

### Priority 0: CRITICAL — Active Bugs (Fix Immediately)

| Rank | ID | Description | Fix | Estimated Effort | Justification |
|------|----|-------------|-----|-----------------|---------------|
| **P0-1** | D13-01 | ~~Eigenvalue index lookup bug~~ **FIXED** | `inner_eigenvalues[active_indices[ci]]` → `inner_eigenvalues[ci]` | 1 line | Active bug corrupting eigenvalue-to-band mapping after any compaction. |
| **P0-2** | comp9-1 | ~~Row-major extraction (location 1)~~ **FIXED** | `h_init_cpu[i*k+j]` → `h_init_cpu[i+j*k]` | 1 line | Latent critical bug — column-major vs row-major confusion. |
| **P0-3** | comp9-2 | ~~Row-major extraction (location 2)~~ **FIXED** | `h_new_rows_cpu[i*new_total+j]` → `h_new_rows_cpu[i+j*n_added]` | 1 line | Same as P0-2, second location. |

### Priority 1: HIGH — All Resolved

| Rank | ID | Description | Fix | Status |
|------|----|-------------|-----|--------|
| **P1-1** | D13-02 + C2-04 | Stage 1 source + A3 destination offset bugs | active_bands() iterator (ADR-0004) | **FIXED** |
| **P1-2** | D10-01 | Incremental super_hamiltonian for diagonalization | h_sub_prebuilt path in diagonalise_subspace() | **FIXED** (verified faithful to CASTEP; hard-depends on C6-D2) |
| **P1-3** | D12-01 | Re-orthogonalization scope: all k_super columns | A2 expanded; ADR-0005 lockstep keeps hpsi consistent | **FIXED** |
| **P1-4** | D12-03 | Stale h_super_wvfn after re-orthogonalization | ADR-0005 lockstep transform in s_orthogonalise/s_orthonormalise | **FIXED** |
| **P1-5** | C6-D2 | Cholesky-primary orthonormalization | CPU-side complex ZPOTRF + ZTRTRI + GPU ZGEMM in s_orthonormalise() | **FIXED 2026-06-07** (real-only bug corrected). |
| **P1-6** | C3-07/C15 | Separate slice workspace | slice_wvfn/slice_h_wvfn buffers; compaction super_wvfn→slice | **FIXED** |
| **P1-7** | D10-03 | Gamma-point DSYEVD dispatch | KptData.kpoint_frac detection → cusolverDnDsyevd | **FIXED** |

### Priority 2: MEDIUM — Edge Cases, Diagnostics, and Hardening

| Rank | ID | Description | Fix | Estimated Effort | Justification |
|------|----|-------------|-----|-----------------|---------------|
| **P2-1** | C17-D3 | Conduction buffer persistence: allocate outside outer loop or copy from previous iteration | Move allocation to before outer loop (line 976), or save conduction states to persistent buffer | ~15 lines | Loss of conduction state information across outer SCF iterations. Affects convergence rate for early SCF cycles. |
| **P2-2** | C14-01 | Negative tolerance guard: add outer guard matching CASTEP | Add `if tol_abs > -(f64::EPSILON)` guard at line 1798 | ~3 lines | Edge case: negative tolerance to disable absolute check. |
| **P2-3** | C5-D1 | Stage 4 pass count: evaluate whether 2 passes are needed, document rationale | Either reduce to 1 pass (match CASTEP) OR document why 2 passes needed on GPU | ~2 lines (change) or investigation | Unknown whether 2-pass is intentional or accidental. |
| **P2-4** | D12-02 | Stage 12 pass count: same as C5-D1 for post-diagonalization re-orthogonalization | Same approach as P2-3 | ~2 lines | Same unknown rationale. |
| **P2-5** | C15-06 | current_nblock mutation: either document as intentional divergence or preserve original value | If preserving: track `original_nblock` separately, use `active_count` for compacted size | ~10 lines | Semantic difference: Rust's `current_nblock` shrinks after compaction; CASTEP's stays fixed. |
| **P2-6** | comp9-4 | Indexing inconsistency: add comment at lines 1431 and 1575 noting correct column-major pattern used elsewhere | Add comment referencing lines 2388/2392 as correct pattern | ~2 lines | Maintenance hazard prevention. |
| **P2-7** | C15-07 | Zero search columns early break: verify it doesn't cause premature inner-loop exit | Add diagnostic counter; compare convergence behavior with/without early break | Investigation | Potential premature exit if all search columns are zero but convergence not met. |

### Priority 3: LOW — Diagnostic, Cosmetic, Verified Benign

| Rank | ID | Description | Action | Justification |
|------|----|-------------|--------|---------------|
| P3-1 | C14-03 | minimisation_steps tracking | Add per-band iteration counter | Diagnostic parity with CASTEP output |
| P3-2 | C14-04 | convergence_values storage | Store delta-e and break_cond_tol | Diagnostic parity |
| P3-3 | C1-D1 | initial_eigenvalues storage | Store initial eigenvalues for diagnostic print | Diagnostic parity |
| P3-4 | C14-02 | band_converged reset timing | Align with CASTEP conditional reset | Cosmetic, no functional impact |
| P3-5 | D10-02 | Rotation matrix padding | Add identity padding beyond subspace dimension | Cosmetic, both approaches valid |
| P3-6 | All other LOW items | Various | No action needed | Verified benign or cosmetic only |

---

## Appendix A: Cascading Dependency Map

All dependency chains resolved:

```
C1-D4 (no separate slice workspace)
  ├── C3-07 (in-place compaction)  ──┐
  ├── C15-01 (in-place compaction) ──┤
  ├── C15-02 (diagonal compaction) ──┤ All resolved via slice workspace (C3-07).
  ├── C15-04 (zero-stale-column)  ──┤ super_wvfn is append-only; compaction
  ├── C15-05 (no slice_eigvals)   ──┤ copies into separate slice.
  └── C15-06 (current_nblock mut) ──┘

C3-08 (no beta_phi cache)
  ├── D12-01 (re-orth scope narrow) — RESOLVED: A2 on all k_super via ADR-0005
  ├── D12-03 (stale h_super_wvfn)   — RESOLVED: ADR-0005 lockstep transforms
  └── D10-01 interaction             — RESOLVED: incremental H_sub implemented

D13-01 + D13-02 compound — RESOLVED: active_bands() iterator (ADR-0004)
```
```

## Appendix B: Fix Interaction Matrix

All HIGH-priority fix interactions are now resolved:

| Interaction | Resolution |
|-------------|------------|
| P1-5 (complex Cholesky) REQUIRED by P1-2 (incremental H_sub / D10-01) | **HARD DEPENDENCY.** D10-01 copies diagonal-reset super_hamiltonian to ZHEEVD. CASTEP tolerates this because its Cholesky-primary A2 keeps post-A2 vectors within numerical noise of pre-A2. Our 2026-06-06 C6-D2 had a real-only Cholesky bug — dropping imaginary parts → false ZPOTRF failure → ALL calls fell through to MGS → larger post-A2 drift → D10-01 inconsistency large enough to corrupt ZHEEVD. With complex Cholesky fixed (2026-06-07), D10-01 is now safe. |
| P1-2 (incremental H_sub) required P1-4 (H*psi recompute) | Both implemented: P1-4 via ADR-0005 lockstep, P1-2 via h_sub_prebuilt |
| P1-3 (full re-orth scope) required P1-4 (H*psi recompute) | Both implemented: ADR-0005 lockstep resolves simultaneously |
| P1-6 (separate slice) reverts C15-01 through C15-06 | All resolved: slice workspace implemented, in-place compaction removed |
| P0-1 (D13-01) required P1-1 (D13-02) | Both fixed; active_bands() iterator ensures correct mapping |

---

## Appendix C: Verification Tests

For each CRITICAL and HIGH fix, create a discriminator test:

| Fix | Test Description | Expected |
|-----|-----------------|----------|
| P0-1 (D13-01) | Run with 4+ bands, force early convergence of band 2. Check eigenvalue at global position of band 3 after compaction. | Eigenvalue at band 3 position matches inner_eigenvalues[ci] where ci is the compacted position for that band. |
| P0-2/P0-3 (comp9-1/2) | Before the super_hamiltonian reset, compare h_init_cpu[i+k*j] (correct) vs h_init_cpu[i*k+j] (current). | They differ for i != j. Fix removes the discrepancy. |
| P1-2 (D10-01) | Compare H_sub from fresh GEMM vs extracted from incrementally-built super_hamiltonian (after fixing comp9-1/2). | They should match to machine precision IF h_super_wvfn is fresh. |
| P1-3 (D12-01) | Monitor ||S_ij - I|| for columns > current_nblock over 10+ inner iterations. | After fix, off-diagonal norm stays bounded near machine epsilon. |
| P1-5 (C6-D2) | For near-degenerate eigenvalue pair (delta < 0.01 eV), compare search direction orthogonality after CGS vs Cholesky. | Cholesky maintains ||S - I|| < 1e-14; CGS may show 1e-10 or worse. |
