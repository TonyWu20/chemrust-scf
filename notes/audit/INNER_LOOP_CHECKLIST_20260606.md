# Davidson Inner-Loop Adversarial Audit Checklist

**Date**: 2026-06-06
**Scope**: Davidson eigensolver inner loop (17 components), Rust (`davidson.rs`) vs CASTEP 6.11 (`hamiltonian.f90`, `nlpot.f90`)
**Methodology**: Adversarial line-by-line audit, 55 surviving differences after filtering false positives

---

## 1. Component Summary Table

| # | Component | CASTEP Source | Rust Source | Status | Severity Spread |
|---|-----------|---------------|-------------|--------|-----------------|
| 1 | Pre-inner-loop eigenvalue init + conduction seeding | `hamiltonian.f90:396-418` | `davidson.rs:1330-1496` | **DIVERGE** | 1x MEDIUM, 4x LOW |
| 2 | Stage 1: psi/hpsi copy to workspace | `hamiltonian.f90:386-408` | `davidson.rs:3015` | **MATCH** | 3x LOW |
| 3 | Stage 2: Preconditioner math | `nlpot.f90:15973+` | `preconditioner.rs:68+` | **DIVERGE** | 2x HIGH, 1x MEDIUM, 5x LOW |
| 4 | Stage 3a: S-orthogonalize against ALL eigenvectors | `hamiltonian.f90:430-435` | `davidson.rs:3205` | **MATCH** | 1x LOW |
| 5 | Stage 4: S-orthogonalize against current superspace | `hamiltonian.f90:437-446` | `davidson.rs:3319` | **DIVERGE** | 1x MEDIUM |
| 6 | Stage 5: S-orthonormalize search directions | `hamiltonian.f90:11573+` | `davidson.rs:2622` | **DIVERGE** | 1x HIGH |
| 7 | Stage 6: Apply H to search directions | `hamiltonian.f90:448-453` | `davidson.rs` (apply_full_hamiltonian) | **MATCH** | 1x LOW |
| 8 | Stage 7: Copy search + Hsearch to superspace | `hamiltonian.f90:455-467` | `davidson.rs` | **MATCH** | 1x LOW |
| 9 | H_new_rows GEMM + super_hamiltonian extraction | `hamiltonian.f90:455-467` | `davidson.rs:1431,1575,1582` | **DIVERGE** | 2x CRITICAL, 1x MEDIUM, 1x LOW |
| 10 | Diagonalization (ZHEGVD/ZHEEV) | `hamiltonian.f90:472-497` | `davidson.rs:1605-1620,2416` | **DIVERGE** | 2x HIGH, 1x LOW |
| 11 | super_hamiltonian reset to diag(eigenvalues) | `hamiltonian.f90:507-510` | `davidson.rs:1634-1640` | **MATCH** | 1x LOW |
| 12 | Post-diagonalization re-orthogonalization | `hamiltonian.f90:517-520` | `davidson.rs:1648-1727` | **DIVERGE** | 1x HIGH, 2x MEDIUM |
| 13 | Eigenvalue/wavefunction copy-back to global arrays | `hamiltonian.f90:523-542` | `davidson.rs:1742-1770,2058` | **DIVERGE** | 1x CRITICAL, 1x HIGH, 1x MEDIUM |
| 14 | Convergence check | `hamiltonian.f90:546-627` | `davidson.rs:1778-1843` | **DIVERGE** | 1x MEDIUM, 6x LOW |
| 15 | Compaction of unconverged bands | `hamiltonian.f90:633-645` | `davidson.rs:1915-2052` | **DIVERGE** | 1x HIGH, 3x MEDIUM, 3x LOW |
| 16 | Conduction state saving | `hamiltonian.f90:537-542` | `davidson.rs:2058-2081` | **DIVERGE** | 1x MEDIUM, 2x LOW |
| 17 | Conduction state seeding for next block | `hamiltonian.f90:223-224,396,537-542` | `davidson.rs:976,1309-1313,2058-2081` | **DIVERGE** | 1x MEDIUM, 2x LOW |

### Status Counts

| Status | Count |
|--------|-------|
| **MATCH** (no significant differences) | 4 (C2, C4, C7, C8) |
| **DIVERGE** (differences exist) | 13 (C1, C3, C5, C6, C9, C10, C12, C13, C14, C15, C16, C17) |
| **FIXED** | 0 |
| **REVERTED** | 0 |

### Severity Counts

| Severity | Count |
|----------|-------|
| **CRITICAL** (active bug or high-risk latent) | 3 (D13-01, comp9-1, comp9-2) |
| **HIGH** (functional impact, not yet verified benign) | 7 (C3-07, C6-D2, D10-01, D10-03, D12-01, D13-02, C15-01) |
| **MEDIUM** (observable effect, edge cases) | 10 |
| **LOW** (diagnostic, cosmetic, or verified benign) | 31 |

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
| C3-07 | Compaction buffer model | `hamiltonian.f90:633` | 1968 | HIGH | **In-place vs separate-workspace compaction**. CASTEP copies from super_wvfn INTO separate slice. Rust compacts IN-PLACE within super_wvfn, then zeros stale columns. Creates near-duplicate column risk for ZHEGVD. Cascades from C1-D4. |
| C3-08 | Post-rotation S-orthogonalization scope | `hamiltonian.f90:519` | 1648 | MEDIUM | CASTEP S-orthogonalizes ENTIRE super_wvfn (including conduction states). Rust only S-orthogonalizes first `current_nblock` columns. Rust comment at line 1648 acknowledges: lack of beta_phi cache means extending to conduction states would leave `h_super_wvfn` stale. |

**Root cause analysis**: The preconditioner *math* (C3-01 through C3-06) is verified identical to CASTEP. All divergences are buffer-architecture issues (C3-07, C3-08) flowing from C1-D4 and the lack of a beta_phi cache.

---

### Component 4: Stage 3a — S-orthogonalize Against ALL Eigenvectors

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C4-OK | No differences found | `hamiltonian.f90` S-orth block | 3205 | LOW | Both use ALL eigenvector bands as reference. Both 1 pass, equivalent S-dot (ZGEMM), equivalent correction. Ncol matches. **MATCH CONFIRMED.** |

---

### Component 5: Stage 4 — S-orthogonalize Against Current Superspace

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C5-D1 | Pass count: 1 vs 2 | `hamiltonian.f90:442` | 3319 | MEDIUM | CASTEP uses 1 pass (`wave_Sorthogonalise_to_lower`). Rust uses 2 passes (`for _pass in 0..2`). Number of reference columns is the same after 0-based/1-based indexing adjustment. Rust's second pass is an unvalidated numerical stability addition. |

**Root cause analysis**: Unknown whether the 2-pass variant was introduced intentionally (e.g., observed instability with 1 pass on GPU) or accidentally. If the algorithm converged with 1 pass in tests, the second pass may mask a different bug. Needs justification or removal.

---

### Component 6: Stage 5 — S-orthonormalize Search Directions

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C6-D2 | Cholesky vs CGS orthonormalization | `hamiltonian.f90:11573` | 2622 | HIGH | CASTEP uses Cholesky-based `wave_Sorthonormalise_overlap` (ZPOTRF + ZTRMM), falling back to CGS only on Cholesky failure. Rust always uses CGS Gram-Schmidt (`s_orthonormalise`). Cholesky is more stable when the S-overlap matrix is well-conditioned; CGS can accumulate error for near-linearly-dependent search directions. |

**Root cause analysis**: CGS is simpler to implement on GPU but less numerically stable. For poorly conditioned search subspaces (which occur when search directions become nearly parallel, common for nearly-converged or near-degenerate bands), CGS can fail to maintain orthogonality, leading to eigenvalue drift. The Cholesky approach uses the overlap matrix S_ij = <psi_i|S|psi_j>, factorizes via ZPOTRF, then rotates via ZTRMM — this is the LAPACK-recommended stable path. Rust should implement the Cholesky-primary, CGS-fallback pattern.

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
| comp9-1 | Row-major extraction from column-major GEMM (location 1) | `hamiltonian.f90:455` | 1431 | **CRITICAL** | `h_init_cpu[i * k + j]` reads kxk column-major buffer with row-major indexing → TRANSPOSE. Diagonal OK, off-diagonal imaginary parts sign-flipped. Correct: `h_init_cpu[i + j * k]`. Currently BENIGN because super_hamiltonian is reset to diag(eigenvalues) at line 1634 before any read. **Latent bug**: any future code that reads off-diagonals before the reset will see a transposed matrix with wrong imaginary signs. |
| comp9-2 | Row-major extraction from column-major GEMM (location 2) | `hamiltonian.f90:455` | 1575 | **CRITICAL** | `h_new_rows_cpu[i * new_total + j]` reads n_added x new_total column-major buffer with row-major indexing → SCRAMBLED (not even a transpose). Correct: `h_new_rows_cpu[i + j * n_added]`. Same root cause as comp9-1. Same benign status. |
| comp9-3 | Hermitian fill direction reversed | `hamiltonian.f90:461` | 1582 | LOW | CASTEP: reads LOWER triangle, writes conj to UPPER. Rust: reads UPPER, writes conj to LOWER. Functionally equivalent. Irrelevant because matrix reset to diagonal afterward. |
| comp9-4 | Inconsistent indexing within same file | `hamiltonian.f90:455` | 1431,2388,2392 | MEDIUM | Diagnostic code at lines 2388/2392 uses CORRECT column-major `[i + j*k]`. Extraction code at lines 1431/1575 uses WRONG row-major `[i*k + j]`. Maintenance hazard: copy-paste between these locations would silently corrupt physics. |

**Root cause analysis**: Row-major vs column-major confusion in 2D array indexing. The GEMM output is column-major (standard BLAS convention), but the extraction code uses C-style row-major indexing (`row * ncol + col` instead of `row + col * nrow`). For comp9-1 (square k x k), this produces a transpose — diagonal entries are correct, off-diagonals have wrong index pairs AND sign-flipped imaginary parts. For comp9-2 (rectangular n_added x new_total), the elements are placed at wrong positions entirely, not even a clean transpose.

These are **latent critical bugs** masked by the super_hamiltonian reset at line 1634. Any refactoring that reads super_hamiltonian before the reset will silently corrupt physics. The fix is trivial: swap the multiplication operands in both locations.

---

### Component 10: Diagonalization (ZHEGVD)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| D10-01 | Diagonalization matrix source | `hamiltonian.f90:472-484` | 1605 | HIGH | CASTEP diagonalizes the incrementally-built `super_hamiltonian` (contains historical search direction contributions). Rust recomputes `H_sub = psi^H * hpsi` from scratch via GEMM, then calls ZHEEVD on this fresh matrix. The incrementally-built matrix preserves off-diagonal information from all historical search directions; the fresh computation only captures the current subspace. **This is the most likely source of eigenvalue accuracy differences between CASTEP and Rust.** |
| D10-02 | Rotation matrix padding | `hamiltonian.f90:493-497` | 1605 | LOW | CASTEP pads rotation matrix with identity beyond `super_wvfn%nbands`. Rust has no padding — ZHEEVD operates on dense k_super x k_super. Different but both valid. |
| D10-03 | EVP solver: ZHEEV vs ZHEEVD, real vs complex | `hamiltonian.f90:480-484` | 2416 | HIGH | CASTEP calls `algor_diagonalise` which selects DSYEV for gamma-point (real) or ZHEEV for complex. Rust always calls cusolver ZHEEVD (complex only, divide-and-conquer variant). Gamma-point calculations in Rust would use the complex path unnecessarily, adding 2x computational cost and potential numerical differences from different algorithm (ZHEEV QR vs ZHEEVD divide-and-conquer). |

**Root cause analysis for D10-01**: This is a fundamental algorithmic divergence. In CASTEP's Davidson, `super_hamiltonian` is built incrementally: each inner iteration appends new rows/columns from the search direction overlap `H_new_rows = <search|H|super_wvfn>`. The diagonalization of this accumulated matrix captures couplings between current search directions AND all historical Ritz vectors in superspace. Rust's approach recomputes `H_sub` from the current superspace vectors via `psi_super^H * hpsi_super` (a fresh GEMM), which *should* produce the same matrix IF `h_super_wvfn = H * super_wvfn` holds exactly. However, if `h_super_wvfn` is stale for any columns (see D12-03), the fresh GEMM uses wrong H*psi values, while CASTEP's incrementally-built matrix uses historically-correct values. The incremental approach is also more robust to gradual loss of H*psi accuracy.

**Root cause analysis for D10-03**: ZHEEVD is a divide-and-conquer algorithm, generally faster but with different rounding behavior than ZHEEV (QR). For well-conditioned matrices the results should agree to machine precision, but for near-degenerate eigenvalues (common in solid-state), the two algorithms can produce different eigenvector rotations. The gamma-point real/complex mismatch doubles the FLOP cost for real wavefunctions.

---

### Component 11: super_hamiltonian Reset to diag(eigenvalues)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| D11-01 | No divergence found | `hamiltonian.f90:507-510` | 1634 | LOW | Both zero super_hamiltonian and set diagonal to eigenvalues for all k_super entries. **MATCH CONFIRMED.** |

---

### Component 12: Post-diagonalization Re-orthogonalization

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| D12-01 | Re-orthogonalization scope: all vs first nblock | `hamiltonian.f90:519` | 1661 | HIGH | CASTEP S-orthogonalizes and S-orthonormalizes ALL k_super columns (including historical search directions and conduction states). Rust only re-orthogonalizes the first `current_nblock` columns (the updated Ritz vectors). Columns `current_nblock..k_super-1` are never re-orthogonalized. Over many inner iterations, un-re-orthogonalized columns accumulate non-orthogonality, degrading the quality of the superspace basis and the eigenvalue accuracy of subsequent diagonalizations. |
| D12-02 | Pass count: 1 vs 2 | `hamiltonian.f90:519` | 1681 | MEDIUM | CASTEP: 1 pass of S-orthogonalize. Rust: 2 passes. Same issue as C5-D1. |
| D12-03 | Stale h_super_wvfn after re-orthogonalization | `hamiltonian.f90:517` | 1648 | MEDIUM | CASTEP sets `super_wvfn%have_beta_phi = .false.` to force H*psi recomputation when the wavefunction is modified. Rust has no beta_phi cache and no mechanism to recompute `h_super_wvfn` after re-orthogonalization modifies `super_wvfn`. H(super_wvfn_new) != h_super_wvfn for modified columns. This interacts with D10-01: if Rust switched to using the incremental super_hamiltonian, stale h_super_wvfn columns would corrupt the H_new_rows computation. |

**Root cause analysis**: The scope restriction (D12-01) is the Rust comment's acknowledged divergence (line 1648): "Rust lacks beta_phi caching so extending re-orthogonalization to conduction states would leave h_super_wvfn stale." This creates a circular dependency: re-orthogonalization is skipped to avoid stale h_super_wvfn, but the skipped re-orthogonalization degrades the superspace basis. The proper fix requires either (a) implementing beta_phi cache tracking with invalidation, or (b) recomputing H for affected columns after re-orthogonalization.

---

### Component 13: Eigenvalue/Wavefunction Copy-back to Global Arrays

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| D13-01 | Eigenvalue index lookup bug | `hamiltonian.f90:527` | 1769 | **CRITICAL** | **Active bug, not latent.** `inner_eigenvalues[active_indices[ci]]` is written to band `block_start + active_indices[ci]`, but the wavefunction at that position came from super_wvfn column `ci`, which has eigenvalue `inner_eigenvalues[ci]`. When `active_indices` has gaps (e.g., `[0, 2, 4]` after compaction), `ci=1` (wavefunction from super_wvfn col 1) gets eigenvalue `inner_eigenvalues[2]` (which belongs to column 2's wavefunction). The eigenvalue and wavefunction at the global position are **mismatched**. Correct index: `ci`, not `active_indices[ci]`. |
| D13-02 | Copy-back scope: active-only vs all | `hamiltonian.f90:523` | 1742 | HIGH | CASTEP copies ALL `current_nblock` columns from super_wvfn to eigenvectors every inner iteration (including converged/stopped bands). Rust only copies active bands (those in `active_indices`). Stopped/converged bands keep stale wavefunctions and eigenvalues from when they were last active. This means: (a) stopped bands' wavefunctions are not updated even though the Ritz vectors in superspace have changed, and (b) stopped bands' eigenvalues are frozen, potentially preventing them from being re-evaluated if their convergence was premature. |
| D13-03 | Conduction eigenvalues not saved | `hamiltonian.f90:540` | 2058 | MEDIUM | CASTEP saves `conduction_eigvals(i) = super_eigvals(current_nblock+i)`. Rust has no equivalent. However, CASTEP's `conduction_eigvals` is write-only dead storage (never read before deallocation), so this is functionally irrelevant in both codebases. |

**Root cause analysis for D13-01**: This is the most impactful active bug in the current code. The index mapping confusion arises from the compaction step: `active_indices` maps *compacted position* to *original block-relative position*. After diagonalization, the Ritz vectors in super_wvfn are ordered by the diagonalization (which sorts by eigenvalue), so column `ci` in super_wvfn corresponds to the `ci`-th Ritz vector with eigenvalue `inner_eigenvalues[ci]`. When copying back to global arrays, the correct mapping is:

- Wavefunction: super_wvfn column `ci` → global band `block_start + original_position_of_column_ci`
- Eigenvalue: `inner_eigenvalues[ci]` → global band same position

The current code uses `active_indices[ci]` for both, but `active_indices[ci]` is the ORIGINAL position, not the Ritz vector position. After compaction, a band at original position 5 may be at compacted position 2, meaning `active_indices[2] = 5`. Its eigenvalue is in `inner_eigenvalues[2]`, not `inner_eigenvalues[5]`. The correct eigenvalue index is `ci` (the compacted/Ritz position), and the correct global destination is `block_start + active_indices[ci]` (the original band position).

This bug means: for any band ordering where `active_indices[ci] != ci` (which happens whenever compaction changes the ordering), the eigenvalues written to the global array are WRONG. This directly causes higher-band eigenvalue drift because a higher band may receive a lower band's (more negative) eigenvalue if the index mapping is off.

---

### Component 14: Convergence Check

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C14-01 | Negative tolerance guard missing | `hamiltonian.f90:555` | 1798 | MEDIUM | CASTEP gates absolute tolerance check with `if(convergence_tols(1) > -epsilon(1.0_dp))` — negative tol_abs skips the entire check. Rust always computes `delta_e < tol_abs.max(eps_guard)`, so negative tol_abs still applies eps_guard. For physical tol_abs values (positive), no difference. |
| C14-02 | band_converged reset timing | `hamiltonian.f90:550` | 1786 | LOW | CASTEP resets `band_converged` only for non-stopped bands. Rust resets unconditionally for all. D1 re-check resets all afterward anyway, so inner-loop exit logic unaffected. Only affects intermediate uses within convergence loop body (minimisation_steps tracking). |
| C14-03 | minimisation_steps not tracked | `hamiltonian.f90:547` | 1782 | LOW | CASTEP tracks per-band minimisation steps and reports via `iterations`. Rust does not. Diagnostic only. |
| C14-04 | convergence_values not stored | `hamiltonian.f90:577,599` | 1778 | LOW | CASTEP stores delta-e and break_cond_tol per-band for diagnostics. Rust does not. |
| C14-05 | abs(tol_rel) vs tol_rel | `hamiltonian.f90:569` | 1807 | LOW | CASTEP uses `abs(convergence_tols(2))`; Rust uses `tol_rel` directly. Both guarded to non-negative before reaching this code. Equivalent for well-behaved inputs. |
| C14-06 | MPI broadcast of convergence state | `hamiltonian.f90:624-627` | 1843 | LOW | CASTEP broadcasts convergence flags across MPI. Rust runs single-GPU. Architectural constraint, not a bug. |
| C14-07 | Stopped state on last outer iteration | `hamiltonian.f90:582` | 1815 | LOW | CASTEP preserves `opt_stop_condition` on last outer iteration (guard fails). Rust explicitly clears it. May cause extra inner iterations in Rust on final outer iteration. |

**Root cause analysis**: Most convergence check differences are diagnostic/logging gaps (C14-02 through C14-07). C14-01 (negative tolerance guard) could affect edge cases where users specify negative tolerance to disable absolute convergence checking entirely, but this is rare. C14-07 could cause slight behavioral differences on the final SCF iteration but should not affect final eigenvalue accuracy.

---

### Component 15: Compaction

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C15-01 | Compaction direction: separate buffer vs in-place | `hamiltonian.f90:637` | 1928 | HIGH | **Fundamental architectural divergence.** CASTEP copies FROM super_wvfn INTO separate slice — super_wvfn unchanged. Rust moves columns in-place within super_wvfn. Cascades from C1-D4 (no separate slice workspace). |
| C15-02 | super_hamiltonian diagonal compaction | `hamiltonian.f90:507` | 1952 | MEDIUM | Rust compacts super_hamiltonian diagonal entries during in-place compaction. CASTEP does not (it rebuilds from super_eigvals each iteration). Necessary consequence of in-place design. |
| C15-03 | previous_eigenvalues/opt_stop_condition compaction | `hamiltonian.f90:639` | 1958 | LOW | Rust compacts these per-block arrays in-place. CASTEP does not compact them. Rust rebuilds them each block anyway, so compaction is unnecessary but harmless. |
| C15-04 | Zero-stale-column fix | `hamiltonian.f90:644` | 1987 | MEDIUM | Rust zeros columns at positions j..ncol_before-1 after in-place compaction to prevent near-rank-deficient ZHEGVD matrices from duplicate/stale column vectors. CASTEP does not need this (separate slice workspace). This is a mitigation for C3-07, not a divergence in itself. |
| C15-05 | slice_eigenvalues not compacted | `hamiltonian.f90:639` | 1915 | LOW | CASTEP copies `slice_eigenvalues(j) = super_eigvals(i)`. Rust has no separate slice_eigenvalues — eigenvalues are recomputed in next ZHEGVD. |
| C15-06 | current_nblock reduction | `hamiltonian.f90:644` | 2034 | MEDIUM | CASTEP: `slice%nbands = j` (separate workspace), `current_nblock` unchanged. Rust: `current_nblock = j` (in-place). This means Rust's `current_nblock` is fluid (shrinks as bands converge), while CASTEP's is fixed per block. May affect loop bounds that use `current_nblock` elsewhere. |
| C15-07 | Early break on zero search columns | `hamiltonian.f90:646` | 2052 | LOW | Rust has `if n_added == 0 { break; }` guard. CASTEP does not. Could cause premature inner-loop exit if orthogonalization produces zero search columns but convergence not yet met. |

**Root cause analysis**: Compaction differences are almost entirely consequences of the C1-D4 architectural choice (no separate slice workspace). The in-place compaction (C15-01) is the root cause, and C15-02 through C15-06 are cascading adaptations. The zero-stale-column fix (C15-04) is a valid mitigation but adds complexity. C15-06 (current_nblock mutation) is a semantic difference that could affect subsequent iterations — in CASTEP, `current_nblock` is the original block size, while in Rust it becomes the count of remaining unconverged bands.

---

### Component 16: Conduction State Saving

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C16-01 | Save timing: every iteration vs after loop | `hamiltonian.f90:537` | 2058 | MEDIUM | CASTEP saves conduction states EVERY inner iteration (between diagonalization and convergence check). Rust saves ONCE after inner loop exit. Functionally: both capture the final Ritz vectors before next block. CASTEP's extra saves are overwritten each iteration and only the last one matters. |
| C16-02 | conduction_eigvals not saved | `hamiltonian.f90:541` | 2061 | LOW | CASTEP's `conduction_eigvals` is write-only dead storage. Neither codebase actually uses it. |
| C16-03 | Conduction band count calculation | `hamiltonian.f90:537` | 2061 | LOW | CASTEP bound: `min(nblock, superspace_index + slice_searchspace%nbands - 1 - current_nblock)`. Rust: `superspace_index - current_nblock`. Both produce equivalent counts in practice. |

**Root cause analysis**: Timing difference (C16-01) is structurally different but functionally equivalent — the last save before next block is what matters. C16-02 is dead code in CASTEP too. No functional impact.

---

### Component 17: Conduction State Seeding for Next Block

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C17-D1 | Save location: inside vs outside inner loop | `hamiltonian.f90:537` | 2058 | LOW | Same as C16-01. Structurally different, functionally equivalent. |
| C17-D2 | conduction_eigvals not in Rust | `hamiltonian.f90:540` | 2061 | LOW | Same as C16-02. Dead code in CASTEP. |
| C17-D3 | Buffer persistence across outer iterations | `hamiltonian.f90:223-224` | 1309 | MEDIUM | **Real functional gap.** CASTEP allocates `conduction_slice/Hconduction_slice` OUTSIDE the outer loop (line 223) — they persist across outer SCF iterations. On outer iteration N+1, conduction states from iteration N's final block are available for seeding. Rust allocates `cond_wvfn/cond_h_wvfn` INSIDE the outer loop (line 976) with `alloc_zeros` and `cond_count = 0` — fresh zeroed each outer iteration. Rust CANNOT seed conduction states at the first block of outer iteration 2+, losing information from previous outer iterations. |

**Root cause analysis for C17-D3**: This is the only verified functional gap in the conduction seeding pipeline. CASTEP's design preserves conduction state information across the outer SCF loop, allowing block 1 of outer iteration N+1 to benefit from the final block's Ritz vectors from iteration N. Rust loses this information. The impact depends on how much the conduction states change between outer iterations — for well-converged SCF cycles, the loss is minor; for early SCF cycles with large charge density changes, the loss could slow convergence.

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

The most probable causal chain, in order of likelihood:

1. **D13-01 (CRITICAL) — Eigenvalue index lookup bug**: This is the primary suspect. After any compaction that changes band ordering (`active_indices[ci] != ci`), higher bands receive wrong eigenvalues. If a higher band at compacted position `ci` has `active_indices[ci] > ci`, it would receive a lower (more deeply bound) eigenvalue from a different band — but the *visible* symptom depends on which band ends up holding the wrong eigenvalue. The drift pattern is that eigenvalues at the global positions `block_start + active_indices[ci]` are wrong by `inner_eigenvalues[active_indices[ci]] - inner_eigenvalues[ci]`.

2. **D13-02 (HIGH) + D13-01 interaction**: If a band is stopped (due to convergence or stagnation), D13-02 leaves its wavefunction and eigenvalue frozen. If D13-01 has already written a wrong eigenvalue to that band, the wrong value is permanently frozen. The correct eigenvalue for that band was written to a different position, which may still be actively updated — creating the appearance of "drift" when comparing the two.

3. **D12-01 (HIGH) — Re-orthogonalization scope**: Over many inner iterations, the un-re-orthogonalized historical search directions lose orthogonality to the active subspace. This pollutes the ZHEGVD diagonalization, increasingly biasing higher eigenvalues (which are more sensitive to basis incompleteness). This is a gradual effect, not a sudden one.

4. **C6-D2 (HIGH) — CGS orthonormalization**: For near-degenerate bands (common near Fermi level in metals or near band crossings), CGS fails to maintain orthogonality of search directions, degrading the search subspace quality.

5. **C3-07/C15-01 (HIGH) — In-place compaction**: If the zero-stale-column fix (C15-04) is insufficient, near-duplicate columns in super_wvfn can create rank-deficient ZHEGVD input, producing spurious eigenvalues near zero.

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
| **P0-1** | D13-01 | Eigenvalue index lookup bug: `inner_eigenvalues[active_indices[ci]]` should be `inner_eigenvalues[ci]` | Change line 1769: `inner_eigenvalues[active_indices[ci]]` → `inner_eigenvalues[ci]` | 1 line | Active bug corrupting eigenvalue-to-band mapping after any compaction. Every single Davidson run with converged bands produces wrong eigenvalues for some bands. This is the **most likely cause of observed higher-band eigenvalue drift**. |
| **P0-2** | comp9-1 | Row-major extraction: `h_init_cpu[i * k + j]` should be `h_init_cpu[i + j * k]` | Change line 1431: swap multiplication operands | 1 line | Latent critical bug. Currently masked by super_hamiltonian reset at line 1634. If any refactoring reads off-diagonals before the reset (or if the reset is removed), physics silently corrupts. Fix now to prevent future catastrophe. |
| **P0-3** | comp9-2 | Row-major extraction: `h_new_rows_cpu[i * new_total + j]` should be `h_new_rows_cpu[i + j * n_added]` | Change line 1575: swap operands AND use `n_added` instead of `new_total` | 1 line | Same as P0-2, second location. Elements are scrambled (not even a clean transpose). |

### Priority 1: HIGH — Functional Impact, Observable in Current Runs

| Rank | ID | Description | Fix | Estimated Effort | Justification |
|------|----|-------------|-----|-----------------|---------------|
| **P1-1** | D13-02 | Copy-back scope: copy ALL current_nblock columns (not just active) to eigenvectors | Extend copy-back loop at lines 1742-1752 to include non-active bands, or CASTEP-style `wave_copy` of all first `current_nblock` columns from super_wvfn | ~10 lines | Stopped/converged bands keep stale wavefunctions. Combined with D13-01, this freezes wrong eigenvalues permanently. After fixing D13-01, still need to propagate correct eigenvalues to ALL bands. |
| **P1-2** | D10-01 | Diagonalization source: use incrementally-built super_hamiltonian for diagonalization instead of fresh GEMM | Pass accumulated super_hamiltonian to ZHEEVD instead of recomputing H_sub = psi^H * hpsi | ~20 lines (with D12-03 fix) | Most significant algorithmic deviation from CASTEP. The fresh GEMM is only correct if h_super_wvfn = H * super_wvfn exactly for all columns. With D12-03 (stale h_super_wvfn), the fresh computation is using wrong H*psi values. Switching to incremental super_hamiltonian matches CASTEP's proven approach. **WARNING**: This fix REQUIRES P1-4 (D12-03) to be fixed first, otherwise the incremental super_hamiltonian would be built from stale H*psi values. |
| **P1-3** | D12-01 | Re-orthogonalization scope: extend to ALL k_super columns | Modify lines 1661-1727 to re-orthogonalize all columns, not just first current_nblock | ~15 lines + D12-03 interaction | Un-re-orthogonalized historical columns accumulate non-orthogonality, degrading ZHEGVD accuracy over inner iterations. This is the most likely cause of gradual eigenvalue quality degradation. |
| **P1-4** | D12-03 | Stale h_super_wvfn after re-orthogonalization: recompute H for modified super_wvfn columns | After re-orthogonalization modifies super_wvfn, recompute H*psi for affected columns, OR implement beta_phi cache with invalidation | ~20-50 lines (full H recompute) or ~100+ lines (beta_phi cache) | Currently, any modification to super_wvfn (re-orthogonalization, copy-back) leaves h_super_wvfn stale. This means H(super_wvfn) != h_super_wvfn. For P1-2 to work correctly, stale columns must be refreshed. The simpler fix: recompute H for modified columns after re-orthogonalization. The CASTEP approach: beta_phi cache invalidation triggers lazy recompute. |
| **P1-5** | C6-D2 | Orthonormalization algorithm: implement Cholesky-primary with CGS fallback | Add ZPOTRF + ZTRMM path for Stage 5, keep CGS as fallback | ~30-50 lines | CGS is less stable for near-linearly-dependent search directions. Cholesky is the LAPACK-recommended approach. Important for systems with near-degenerate bands (metals, band crossings). |
| **P1-6** | C3-07/C15-01 | In-place compaction: verify zero-stale-column fix is sufficient, or switch to separate slice workspace | Option A: Add rigorous verification that zeroed columns don't leak into ZHEGVD. Option B: Implement separate slice workspace matching CASTEP architecture. | Option A: ~10 lines (diagnostic). Option B: ~100+ lines | In-place compaction is the single largest architectural divergence. The zero-stale-column fix (C15-04) is a mitigation, not a solution. If D13-01 fix doesn't resolve eigenvalue drift, this is the next most likely cause. |
| **P1-7** | D10-03 | Gamma-point real support: implement DSYEVD path for gamma-point calculations | Add real/complex dispatch based on gamma-point flag | ~30 lines | For gamma-point, forcing complex ZHEEVD on real data is 2x cost and may introduce unnecessary numerical differences. |

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

```
C1-D4 (no separate slice workspace)
  ├── C3-07 (in-place compaction)  ──┐
  ├── C15-01 (in-place compaction) ──┤
  ├── C15-02 (diagonal compaction) ──┤
  ├── C15-04 (zero-stale-column)  ──┤ All flow from C1-D4
  ├── C15-05 (no slice_eigvals)   ──┤
  └── C15-06 (current_nblock mut) ──┘

C3-08 (no beta_phi cache)
  ├── D12-01 (re-orth scope narrow)
  ├── D12-03 (stale h_super_wvfn)
  └── D10-01 interaction (incremental H needs fresh H*psi)

D12-03 (stale h_super_wvfn)
  └── D10-01 (fresh GEMM uses stale columns)
        └── If P1-2 (incremental H_sub) is implemented without P1-4
            (H*psi recompute), the incremental matrix would be wrong

D13-01 (eigenvalue index bug) + D13-02 (active-only copy)
  └── Compound: wrong eigenvalue frozen at wrong band position
```

## Appendix B: Fix Interaction Matrix

| If you fix... | You must also... | Because... |
|---------------|-----------------|------------|
| P1-2 (incremental H_sub) | P1-4 (H*psi recompute) | Incremental H_sub uses h_super_wvfn columns which may be stale |
| P1-3 (full re-orth scope) | P1-4 (H*psi recompute) | Re-orthogonalization modifies super_wvfn, making h_super_wvfn stale |
| P1-6 (separate slice workspace) | Reverts C15-01 through C15-06 | In-place compaction no longer needed |
| P0-1 (D13-01 eigenvalue index) | P1-1 (D13-02 all-band copy) | Fixing the wrong eigenvalue assignment reveals that stopped bands still have stale data |
| P1-5 (Cholesky orthonormalization) | None (independent) | Cholesky path is self-contained in Stage 5 |

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
