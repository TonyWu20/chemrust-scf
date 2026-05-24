# Proposal §14 — Eigensolver Rotation Cascade Fix

**Status (2026-05-24):** PROPOSAL DRAFT. Two implementation attempts on
`feat/phase-global-woodbury` have been stashed; this document captures
the empirical state, what was tried, and what the next attempt must do
differently.

**Implementation plan:** A detailed, A/B-augmented implementation plan
lives at `notes/proposals/14-eigensolver-rotation-fix-plan.md`. That
plan supersedes §3–§4 of this document by introducing a `PinMode` enum
and implementing **both** `PreRr` (proposal §3.1) and `PostRr` (recommended
based on strict-degeneracy theory) so the cascade test can falsify between
them. Read this proposal for the empirical motivation and constraints; read
the plan for the executable design.

**Cross-reference:** `notes/open-followups.md` §14 (one-line index entry).
This document is the dedicated unit; load it instead of the index when
working on this issue.

---

## 1. Empirical state — what we know today

### 1.1 Hard discriminators (anchors EXTERNAL to our pipeline)

| Anchor | Source | Value |
|--------|--------|-------|
| A1: band-0 reference | `Cu111_CO.bands:12` | −1.05502287 Ha |
| A2: ε_F | `Cu111_CO.bands:5` (header) | −0.122443 Ha |
| A3: smearing width | `Cu111_CO.param:35` | 0.1 eV ≈ 3.6749 mHa |
| A4: total energy | `Cu111_CO.castep` reference | −24110.96665069 eV (= −886.06175 Ha) |
| A5: CASTEP ψ ground-truth | `Cu111_CO.check` (S-orthonormal: ⟨ψ\|S\|ψ⟩ = 1) | binary checkpoint |

### 1.2 Current empirical readouts (commit `44796f5`, after GS volume_factor fix)

| Diagnostic | Result |
|------------|--------|
| `test_2_s_sub` (RR validation) | S_sub eigenvalues = 1.0 ± 1.9e-16 (was 0.3715 pre-fix) ✓ |
| All 6 RR validation tests | all PASS — eigensolver math is correct |
| `iter1_drift_from_castep_state_is_bounded` (Q1) | PASS at 20 mHa — energy iter-1 fidelity is fine |
| `cascade_iter3_diagnostic_tight` | iter-3 band-0 = −11.94 Ha (gate < 0.1 Ha from −1.055) — **RED** |
| `overlap_iter2_against_castep` | avg S-overlap = 0.1446 (gate > 0.5) — **RED** |
| `scf_converges_to_castep_energy_at_castep_tolerance` (Q2) | RED (expected) |

### 1.3 Discriminator that flipped the §14 framing: `subspace_projector_iter1_vs_castep`

For two bases spanning the same `k`-dim subspace, the projector matrix
`P[a,b] = ⟨our_a | S | castep_b⟩` has unitary singular values, so
`‖P‖_F² = k`. This is the cleanest possible rotation-vs-span-loss test.

| Block | k | Σ \|M[a,b]\|² | ratio | reading |
|-------|---|---------------|-------|---------|
| band 0 alone | 1 | 0.9921 | 0.992 | single-band block, near-perfect |
| Cu-3d 1..14 | 13 | 11.6041 | **0.893** | rotation + ~11% span pollution |
| 0..30 | 30 | 28.1880 | 0.940 | most of the subspace preserved |
| 0..40 | 40 | 37.5412 | 0.939 | ~6% pollution beyond the 40-band window |

Per-row arg-max from the same test shows a **permutation pattern**:
- our_a=4 → castep_b=6, our_a=5 → 7, our_a=6 → 8, our_a=7 → 9, our_a=8 → 10
  (a 4-band cyclic shift)
- our_a=9 → b=5, our_a=10 → b=4 (adjacent swap)
- our_a=11 → b=12, our_a=12 → b=11 (adjacent swap)
- our_a=13 → 14, …, our_a=16 → 13 (4-band cyclic shift)

This is **not** random rotation — it's **block-localized
permutation + rotation within near-degenerate clusters**. ZHEGVD's stable
sort gives non-CASTEP ordering, and within each tight cluster, the
Procrustes alignment to CASTEP's basis has max coefficients of 0.5–0.97
instead of 1.0.

**Verdict:** The cascade is 90% in-block ZHEGVD rotation + permutation
(matching the original §14 hypothesis), 10% genuine subspace pollution
from outside the 40-band tracking window.

### 1.4 Rotation-vs-span-loss decomposition

| Mechanism | Empirical contribution | Addressable by |
|-----------|------------------------|----------------|
| ZHEGVD permutation across near-degenerate eigenvalues | dominant | basis pinning (F3 family) |
| ZHEGVD in-block rotation within near-degenerate clusters | dominant | basis pinning (F3 family) |
| Chebyshev filter pollution from outside-window high-G components | ~6–11% | tighter `b_low`, or filter quality |

### 1.5 What is NOT the bug

- Chebyshev filter polynomial math (per-eigenstate `T_k(scaled_ε)` — bands stay in their own column)
- R-ChFSI residual update (per-band scalar `Λ_Y`, no inter-band mixing in the kernel)
- Gram-Schmidt classical 2-pass (validated by test_2 with `S_sub = I`)
- Rayleigh-Ritz mathematical correctness (6 validation tests all PASS)
- Density formula (machine-precision against CASTEP ψ at iter-1)
- V_eff assembly (matches CASTEP ρ at machine precision)
- chemrust-hamiltonian D-screening (T-prime FAIL at 197 mHa, T3 PASS at 9.8 mHa
  — see `notes/debug/debug-20260524-tprime-d-injection/RESOLUTION.md`)

---

## 2. What was tried this session (both failed)

Both implementations are stashed:
- `stash@{0}: failed-optionB-procrustes-prev-psi`
- `stash@{1}: failed-f3d-narrow-identity-target`

### 2.1 Attempt 1 — F3d-narrow identity-target polar pin

**Idea:** For each near-degenerate eigenvalue block (size `k`, within the
Fermi window `[μ − 3w, μ + 3w]`), extract the k×k diagonal sub-block
`M = X[lo..hi, lo..hi]` from ZHEGVD's eigenvector matrix, compute its
SVD `M = U·Σ·V^H`, and replace `X[:, lo..hi]` with `X[:, lo..hi] · V·U^H`.
This is the orthogonal Procrustes solution with target = identity in
X-coords.

**Why it failed:** Mathematically wrong. `M` (the k×k diagonal sub-block
of X) is **not unitary** in general. So `R = V·U^H` makes `M·R = U·Σ·U^H`
which is not the identity — it's the closest **Hermitian** matrix to a
unitary, which has nothing to do with what we want.

**Empirical signature:** iter-1 pinned 1 block, iter-2 had μ jumped to
+0.267 Ha (out of Fermi window), iter-3 band-0 = −15.02 Ha.

### 2.2 Attempt 2 — Option B Procrustes against pre-Chebyshev ψ_prev

**Idea:** At iter t ≥ 2, the input ψ to `chebyshev_filter` is the
iter-(t−1) RR-rotated ψ — a canonical reference basis for continuity.
For each near-degenerate block, compute the overlap matrix
`T = ψ_prev^H · S · ψ_row[:, lo..hi]`, take its SVD `T = U·Σ·V^H`, and
set `R = U·V^H`. Apply `X[:, lo..hi] ← X[:, lo..hi] · R`.

**Why it failed:** The framing assumed the block boundaries are known
from current eigenvalues, but at iter-2+ the eigenvalues drift far
enough that the block detection fires zero times (μ jumps outside the
Fermi window).

**Empirical signature:** Same as Attempt 1 — 0 blocks pinned at iter-2,
cascade continues.

### 2.3 Lessons from the failures

1. **The k×k diagonal sub-block of X is not generally unitary.** SVD-polar
   on a non-unitary matrix gives the closest unitary, which is the polar
   factor, but it solves the wrong Procrustes problem when the "target"
   is identity-in-X-coords.

2. **μ-driven Fermi window detection is fragile.** When V_eff drifts
   between iterations, μ drifts with it, and the window-based block
   detection can silently fire zero times. Block detection needs to be
   **eigenvalue-spacing-based** (consecutive Δε < threshold), not
   absolute-energy-window-based.

3. **The pin's success criterion was unclear.** Both attempts had no
   iter-1 sanity check: "with CASTEP's ψ as input and CASTEP's V_eff
   built from CASTEP's ρ, does the pin reproduce CASTEP's ψ within
   1e-10?" If iter-1 doesn't pin to identity in this case, no later
   iteration will either.

---

## 3. Proposed next-attempt design

### 3.1 Correct math — Orthogonal Procrustes against ψ_prev in the S-metric

For each near-degenerate eigenvalue block `[lo, hi)` of size `k`:

1. **Block detection** by consecutive eigenvalue spacing only:
   ```
   maximal contiguous runs where Δε < eps_degen (= 0.01 Ha)
   AND k ≥ 2
   ```
   No μ window. Constant-occupation blocks deep below μ get pinned too
   (no-op for density, but ψ continuity is real and feeds into β·ψ).

2. **Target = ψ_prev (the input to chebyshev_filter at this iteration)**:
   compute the full overlap `T = ψ_prev^H · S · ψ_after_GS`, take the
   k×k sub-block `T_block = T[lo..hi, lo..hi]`. SVD `T_block = U·Σ·V^H`.
   Procrustes-optimal rotation is `R_block = U·V^H` (a true unitary).

3. **Apply** `ψ_after_GS[:, lo..hi] ← ψ_after_GS[:, lo..hi] · R_block`.
   Then run RR (ZHEGVD) on the pinned basis. The RR step still rotates
   within blocks, but because ψ is now pre-aligned with ψ_prev, the RR
   rotation has minimal effect within blocks.

4. **Sanity check** after pin: `‖R_block^H · R_block − I‖_F < 1e-10`.
   If it fails, fall back to identity (= no pin) for that block.

### 3.2 Why this is different from Attempts 1 and 2

- **Math correctness**: `T_block` is generally not unitary, but
  `R_block = U·V^H` from its SVD is **always** unitary (by construction
  of the SVD). The polar pin is mathematically sound.
- **Block detection**: spacing-based, not μ-window-based. Catches all
  near-degenerate clusters at any energy, including deep core states.
- **Apply timing**: AFTER Gram-Schmidt, BEFORE the RR S_sub/H_sub gemms.
  This way RR diagonalizes the *pinned* basis, not the un-pinned one.

### 3.3 Iter-1 sanity check (load-bearing)

Add a new test `pin_preserves_castep_basis_at_iter1`:
1. Load CASTEP ψ as input ψ_prev AND as input to chebyshev_filter.
2. Run iter-1 with the pin active.
3. Verify: per-block S-overlap `|⟨our_a|S|castep_a⟩|² > 0.999`.

If this test doesn't go to ≈1.0, the pin doesn't preserve CASTEP's basis
in the most favorable case (CASTEP V_eff, CASTEP ψ, no V_eff drift) and
any iter-2+ behavior is meaningless.

---

## 4. Implementation plan

### 4.1 Call site

`src/eigensolver/rayleigh_ritz.rs`, between Gram-Schmidt (which lives in
chebyshev.rs at lines 1708-1799, output `final_psi_buf`) and the H_sub /
S_sub gemms in rayleigh_ritz.rs:74-127.

`ψ_prev` must be plumbed through:
- `src/scf.rs:550-625` — `diagonalize` saves the pre-Chebyshev ψ as
  `prev_psi_dev` (already done in the stashed Option B attempt; reuse
  that bit of plumbing).
- `rayleigh_ritz` signature gets `prev_psi_dev: Option<&CudaSlice<CudaComplex>>`
  + `pin_cfg: Option<&RrPinConfig>`.
- `RrPinConfig` struct: `eps_degen: f64` (default 0.01 Ha).

No `smearing_width`/`n_electrons` plumbing needed (μ-free design).

### 4.2 Implementation steps

1. **Detect blocks** from eigenvalues (CPU walk, < 1 µs).
2. For each block of size k ≥ 2:
   - Compute `T_block = ψ_prev[:, lo..hi]^H · S · ψ_after_GS[:, lo..hi]`
     on GPU (one gemm + S·ψ via `apply_s_times` on the k columns).
   - D2H `T_block` (k×k complex, ≤ 15×15 = 225 elements).
   - CPU SVD via `faer` (already proposed dep, no new GPU code).
   - Compute `R = U·V^H` on CPU (k×k matmul).
   - H2D `R` to a temp slice.
   - Apply `ψ_after_GS[:, lo..hi] ← ψ_after_GS[:, lo..hi] · R` (one
     small gemm).
3. **Assert** `‖R^H · R − I‖_F < 1e-10` per block (otherwise rollback
   that block).

### 4.3 Files to modify

- `Cargo.toml` — add `faer = "0.24"` (already proposed; not currently
  added — the stashed attempts had it).
- `src/scf.rs` — plumb `prev_psi_dev` upload + PCI-E bookkeeping
  (~10 lines; reuse from stashed Option B).
- `src/eigensolver/rayleigh_ritz.rs` — new helper
  `pin_blocks_against_prev_psi`, call site after GS / before RR
  gemms (~80 lines).
- `tests/ca_scf_convergence.rs` — new test
  `pin_preserves_castep_basis_at_iter1` (the iter-1 sanity gate).

---

## 5. Discriminator gates (red→green)

| Test | Current (RED) | Target (GREEN) | Anchor |
|------|---------------|----------------|--------|
| `pin_preserves_castep_basis_at_iter1` (NEW) | n/a | per-block S-overlap > 0.999 | A5 |
| `subspace_projector_iter1_vs_castep` Cu-3d block ratio | 0.893 | > 0.999 | A5 |
| `overlap_iter2_against_castep` avg | 0.1446 | > 0.95 | A5 |
| `cascade_iter3_diagnostic_tight` iter-3 band-0 | −11.94 Ha | within 0.1 Ha of −1.055 | A1 |
| `scf_converges_to_castep_energy_at_castep_tolerance` (Q2) | RED | within 1e-5 eV of CASTEP | A4 |

The 10% Chebyshev-filter pollution surfaced by the subspace-projector
diagnostic is a *separate* problem; if iter-1-iter-3 cascade tests all
go green but Q2 still fails by ~5% in energy, the pollution is the
remaining work and should be addressed by tightening `b_low` (currently
`max_veff + 2.0` at iter-1 — possibly too high).

---

## 6. Fallback ladder

If the corrected Procrustes-against-ψ_prev pin doesn't suffice:

1. **F3-extended** — extend pin to all eigenvalue spacings ≤ 0.05 Ha
   (looser). Catches more clusters.
2. **F3a — Deflation lock** (Saad ch. 8). After each SCF iteration, mark
   bands with ε < ε_F − 5w as "converged" and project them out of the
   next iteration's GS+RR. Only the active subspace (near ε_F and above)
   gets re-orthonormalized. Heavier — needs convergence test per band.
3. **Davidson-style block update** — replace filter+RR+GS with a Davidson
   residual loop. Most expensive option; reserved for last resort.

---

## 7. Out of scope

- Replacing subspace-RR with band-by-band CG (CASTEP's algorithm). Much
  larger refactor; explicitly out of §14.
- Performance work on `compute_aug_density_fine` (§9 in open-followups).
- chemrust-hamiltonian D-screening tightening (T-prime FAIL proves it's
  not load-bearing for this cascade).

---

## 8. Stashed work to consult, not restore

- `stash@{0}` — Option B Procrustes against pre-Chebyshev ψ_prev. Has
  the `prev_psi_dev` plumbing in scf.rs and `RrPinConfig` struct. The
  block-detection function uses μ-window (wrong); replace with
  spacing-only detection. The SVD/Procrustes math itself is mostly right
  but applied at the wrong tangent point (block of `T = ψ_prev^H · S · ψ_row`
  for the relevant column range, not the diagonal sub-block of X).
- `stash@{1}` — F3d-narrow identity-target. Don't restore; the math is
  wrong (M not unitary).

Both stashes can be dropped after this proposal is implemented and the
relevant plumbing is re-derived from scratch in the new attempt.

---

## 9. Estimated effort

- Plumbing `prev_psi_dev` through scf.rs: ~15 lines.
- `pin_blocks_against_prev_psi` helper: ~80 lines.
- Iter-1 sanity test: ~50 lines.
- Total new code: ~150 lines.
- Verification: 4-5 test runs at ~6 min/run (CA + Q1 + Q2).
- Best-case session: 1 focused session (~2-3 hours) if math is right
  the first time.
- Worst-case: 2-3 sessions if S-metric polar fallback is needed for
  ill-conditioned T_block at iter-2+.

---

## 10. Open questions to resolve at the start of the next session

1. Does ZHEGVD's eigenvector phase convention (the arbitrary `e^{iθ}`
   on each non-degenerate band) need a separate phase pin? Pure rotation
   theory says no (density and β·ψ projections are phase-invariant for
   non-degenerate bands), but it's worth checking empirically with a
   "phase-only pin for k=1 blocks" toggle.

2. Should the pin run also at iter-1 (where ψ_prev = ψ_input =
   CASTEP ψ)? Yes — the iter-1 sanity test in §5 requires it, and it's
   the cleanest test of the implementation. If iter-1 pin → identity
   (no rotation needed because ψ_input already equals ψ_prev), the test
   should be trivially green.

3. Is `eps_degen = 0.01 Ha` tight enough? Cu-3d cluster spans
   `[−0.49, −0.41]` Ha — Δε between adjacent bands is ~7 mHa. The Fermi
   cluster has Δε ≈ 0.3 mHa. So `eps_degen = 0.01 Ha = 10 mHa` catches
   both. A looser value (e.g. 0.05 Ha) might fragment fewer blocks but
   risks pinning across genuinely non-degenerate band gaps.
