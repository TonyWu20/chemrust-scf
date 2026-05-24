# Plan — §14 Eigenvector-Rotation Cascade: F3d-Narrow Identity Pinning at the Fermi-Level Smear-Cluster

## Context

`notes/open-followups.md` §14 (the only active item — §15 was resolved in
commit `2819624` two commits ago) records that SCF diverges by iter-3 in
`tests/ca_scf_convergence.rs::cascade_iter3_diagnostic` even after every
known formula bug is fixed. T-prime FAIL at 197 mHa
(`notes/debug/debug-20260524-tprime-d-injection/RESOLUTION.md`) and T3 PASS
at 9.8 mHa together root-causally confirm the cause is **chemrust-scf
eigenvector rotation between SCF iterations**, not D-screening, not V_eff
assembly, not chemrust-hamiltonian. All 6 RR validation tests pass
(`tests/rayleigh_ritz_validation.rs`); the eigensolver is mathematically
correct. The bug is a **stability defect**: ZHEGVD's eigenvector choice
within near-degenerate eigenvalue blocks differs each iteration.

§14 names three candidate fixes (F3a deflation-lock, F3b GS-pin, F3c
Davidson) drawn from the standard subspace-iteration literature
(Saad, Knyazev, Davidson). All three are heavyweight reshapes of the
eigensolver and the prior session deferred them.

The user's "why don't the papers discuss this?" question (asked during plan
intake) reframed the surface. The answer makes the problem narrower than
§14 implies:

- **Plane-wave density `ρ = Σ_b occ_b |ψ_b|²` is rotation-invariant** under
  unitary mixing within any equal-occupation block. Standard NCPP papers
  benchmark on this density and ignore basis rotation by construction.
- **USPP augmentation density** `ρ_aug = Σ_b occ_b Σ_{nm} Q^I_{nm} (β·ψ)*_{n,b} (β·ψ)_{m,b}`
  is also rotation-invariant in equal-occupation blocks (the inner sum
  contracts to `ω^I_{nm}` which absorbs U†U = I).
- **Fractional-occupation blocks are not rotation-invariant**: two
  near-degenerate bands at ε ≈ μ with occupations 0.7 / 0.3 produce
  occupation-weighted contributions to `ω^I` that depend on the basis
  choice. This is what cascades.

Empirical confirmation from `Cu111_CO.bands`:
- ε_F = −0.122443 Ha (file header), smearing width = 0.1 eV ≈ 3.68 mHa
- Cu 3d cluster: bands 2-13 at ε ≈ −0.49 to −0.41 Ha — depth (μ-ε)/w ≈ 80
  in units of width ⇒ erfc((ε−μ)/w) saturates at 2.0; **constant-occupation,
  rotation-invariant**
- Fermi-level cluster: bands 85-95 at ε ≈ −0.16 to −0.12 Ha, with bands
  92/93/94 at Δε < 0.3 mHa from each other and Δε < 0.4 mHa from μ —
  **occupations ∈ (0.4, 1.6), fractional, rotation-sensitive**

So the cascade is driven by ~10 bands within ±3·width of μ, not the full
160-band subspace. The §14 candidates F3a/F3b/F3c all reshape the
eigensolver globally; we can do something targeted, lighter, and
literature-justifiable instead.

The user chose option 4 in plan intake: explore F3a/F3b/F3c/F3d (Procrustes
alignment) before committing. This plan therefore proposes **a sequenced
F3-investigation, with F3d-narrow as the first attempt** because (a) it is
the most surgical, (b) it directly targets the rotation surface identified
above, and (c) it has the smallest rollback cost if it doesn't suffice.

The discriminator (also user-chosen) is dual: tighten
`cascade_iter3_diagnostic` to assert iter-3 band-0 within 0.1 Ha of CASTEP,
**and** add an iter-2 eigenvector-overlap test against CASTEP. Both must
go red on current code, both must go green after the fix. Two anchors
guard against false-positive fixes (e.g., damping that obscures the
rotation without addressing it).

## Anchor criteria (EXTERNAL, from this debug session)

All from fixture files, not derived intermediates:

| Anchor | Source | Value |
|--------|--------|-------|
| A1: band-0 (ground state) | `Cu111_CO.bands:12` | −1.05502287 Ha |
| A2: ε_F | `Cu111_CO.bands:5` (header) | −0.122443 Ha |
| A3: smearing width | `Cu111_CO.param:35` | 0.1 eV = 3.6749e-3 Ha |
| A4: smearing scheme | `Cu111_CO.castep:154` | Gaussian (erfc) |
| A5: number of electrons | `Cu111_CO.bands:3` (header) | 186.0 |
| A6: total energy | `Cu111_CO.castep` reference | −24110.96665069 eV (= −886.06175 Ha) |
| A7: T3 cascade-stops baseline | `notes/debug/debug-20260524-tprime-d-injection/RESOLUTION.md` | iter-2 band-0 = −1.0452 Ha when V_eff substituted |
| A8: T-prime cascade-continues | same RESOLUTION.md | iter-2 band-0 = −0.858 Ha when only D substituted |
| A9: CASTEP ψ for overlap | `Cu111_CO.check` (USPP S-orthonormal: ⟨ψ\|S\|ψ⟩ = 1) | binary checkpoint |

Bands within ±3w of μ (the rotation-sensitive cluster) computed offline
from `Cu111_CO.bands`: bands ~85–98, eigenvalues ε ∈ [−0.197, −0.094] Ha.
This is what the targeted fix needs to stabilize.

## Divergence-surface enumeration (for the F3 fix)

The fix surface is "what additional step inside the eigensolver pipeline
breaks the iter-to-iter rotation in fractional-occupation blocks?"

| Candidate | Where it acts | Estimated LOC | Iter-2 band-0 expected |
|-----------|---------------|---------------|------------------------|
| **F3d-narrow (identity pinning, Fermi-cluster only)** | Post-ZHEGVD, pre-rotation in `rayleigh_ritz` | ~70 lines | aim ≤ 0.05 Ha drift |
| F3d-full (identity pinning, every degenerate block) | Same | ~80 lines | same |
| F3a (deflation-lock occupied bands) | Wraps Chebyshev + GS + RR; needs convergence test per band | ~250 lines | same, but converges faster |
| F3b (GS-pin to previous ψ) | Inside Gram-Schmidt block of `chebyshev.rs:1708-1800` | ~60 lines | partial — leaves ZHEGVD rotation untouched |
| F3c (Davidson residual update) | Replaces filter+RR+GS with Davidson loop | ~500+ lines | same effect, much heavier |

**Identity pinning, not previous-iteration alignment.** The fix replaces each
near-degenerate block of ZHEGVD's eigenvector matrix `X_block` with its
**unitary polar factor** `U_polar = U_svd · V_svd^H` (where `X_block = U·Σ·V^H`
is the SVD). The polar factor is the closest unitary matrix to `X_block`
in Frobenius norm — equivalently, the orthogonal Procrustes solution with
target = I in the X-coordinate system. Since the *input* ψ to RR is already
the previous-iteration basis (it's been Gram-Schmidted from the prior RR's
output), pinning X_block to identity-in-X-coords pins ψ_new to the previous
basis automatically. **No `prev_beta_psi` plumbing through `ScfIteration`
needed** — μ is recomputed inline from current eigenvalues.

**Why F3d-narrow first:** It is the *only* candidate that directly addresses
the empirically isolated surface (fractional-occupation bands within 3w of
μ) without touching code that the RR validation suite already proves
correct. F3a/F3b/F3c are global reshapes that bring more risk per LOC.

**Why F3d's absence from the literature is consistent, not suspicious:** ChFSI
papers benchmark on systems where the load-bearing failure mode (frac-occ
at degeneracy + USPP augmentation) is not present. Their density is
rotation-invariant in their benchmark systems, so a Procrustes/polar
alignment step would do nothing and would not be reported.

**S-metric subtlety to verify numerically.** ZHEGVD eigenvectors satisfy
`X^H · S_sub · X = I` (S-orthonormal). The L2 SVD of X_block produces a
polar factor unitary in the L2 sense, not the S-sense. For exactly degenerate
blocks the two coincide (X_block itself is unitary; SVD gives Σ = I and the
pin is a near no-op). For approximately-degenerate blocks the difference is
O(‖S_sub_block − I‖). In USPP this can be non-trivial. The implementation
must verify `‖X_block_pinned^H · S_sub_block · X_block_pinned − I‖_F < 1e-8`
after each pin; if it fails, fall back to the **S-metric polar**: Cholesky
`L · L^H = S_sub_block`, SVD `L · X_block = U·Σ·V^H`, set `X_block_pinned =
L^{-1} · U · V^H`. Both blocks are k×k with k ≤ ~15, so the extra Cholesky
is microseconds.

## Recommended approach

**Sequenced F3 investigation, F3d-narrow first.** Each attempt has a clear
red→green gate; if it fails, advance to the next candidate.

### Step 1 — Write the discriminator tests (must fail today)

Two new tests in `tests/ca_scf_convergence.rs`, both red against current
code at `feat/phase-global-woodbury` HEAD. Both `#[ignore]` + feature
`scf_diag` to match the existing convention.

**T-cascade-tight** — converts `cascade_iter3_diagnostic` (currently a
diagnostic with no hard assertion) to assert:

```rust
assert!(
    (iter3_band0 - (-1.05502287)).abs() < 0.1,
    "iter-3 band-0 = {:.4} Ha (gate 0.1 Ha vs CASTEP -1.05502; current ~-11.94)",
    iter3_band0,
);
```

Current value is iter-3 band-0 = −11.94 Ha; threshold 0.1 Ha gives a
~109× discriminator. **Anchor**: A1.

**T-overlap-iter2** — new test, runs 2 SCF iterations from CASTEP state,
computes per-band **S-inner-product** overlap with CASTEP ψ and asserts
average > 0.5:

```rust
let avg_overlap = ...;  // average across first 20 bands
assert!(
    avg_overlap > 0.5,
    "iter-2 avg S-overlap = {:.3} (gate 0.5; current ~0.11 L2)",
    avg_overlap,
);
```

**Critical: must use `apply_s_times` from `chebyshev.rs:934`, NOT raw
`cublasZdotc` L2.** Cu 3d PW norms are 0.14–1.03 (per
`rayleigh_ritz_validation` test 6), so the *ceiling* of the L2 overlap
`|⟨ψ_b|ψ_b⟩|²` for self-comparison is `‖ψ_b‖_PW⁴ ≈ 0.02` for the
lowest-PW-norm bands — making any `> 0.5` gate physically unreachable.
The S-inner product `|⟨ψ_b|S|ψ_castep_b⟩|²` correctly gives 1.0 for
identical USPP-normalized wavefunctions because `⟨ψ|S|ψ⟩ = 1` by
construction.

**Anchor**: A9 (Cu111_CO.check). **Discriminator**: ~5× (0.5 vs 0.11
current).

**Diagnostic self-test for T-overlap-iter2** (run before trusting the
test result):

1. Feed CASTEP ψ as both "our" and "castep" → must give `|⟨ψ|S|ψ⟩|² ≈ 1.0`
   per band (S-orthonormal property).
2. Feed CASTEP ψ band 0 vs band 1 → must give ~0 (orthogonal eigenstates
   of a Hermitian operator).
3. Run against deliberately rotated ψ (swap columns 2,3) → must detect
   the swap (low self-overlap for bands 2 and 3, high cross-overlap).

If any of these fails, the diagnostic itself is buggy and must be fixed
before its output is trusted as the green/red signal. This satisfies the
`/debug-outcomes` Step 5 diagnostic-verification requirement.

### Step 2 — Implement F3d-narrow identity pinning

**Where:** `src/eigensolver/rayleigh_ritz.rs`, between the ZHEGVD info
check (after line 226) and the `psi_new = psi_row · X` rotation gemm
(currently lines 247-265). Mirror the same change in
`rayleigh_ritz_with_matrices` near line 487.

**Dependency:** Add `faer = "0.24"` to `Cargo.toml` (pure-Rust linalg
already used in chemrust-hamiltonian-core workspace) for the small-matrix
SVD and Cholesky. Alternative: `nalgebra` — also already an indirect
dependency.

**New function in `src/eigensolver/rayleigh_ritz.rs`:**

```rust
/// Pin near-degenerate eigenblocks near the Fermi level to the identity
/// basis via unitary polar factor.
///
/// After ZHEGVD, X_block (the submatrix of eigenvectors for a near-degenerate
/// block) is unique only up to a unitary rotation R within the block. We
/// replace X_block with its polar factor U·V^H (where X_block = U·Σ·V^H is
/// the SVD), which is the closest unitary matrix to X_block in Frobenius
/// norm. Since the input ψ to this RR call is the previous iteration's
/// basis (it was the rotated ψ from the prior RR's `psi_new = psi_row · X`
/// gemm), pinning X_block to identity in X-coordinates pins ψ_new to the
/// previous basis.
///
/// Only blocks where (a) consecutive eigenvalues differ by < `eps_degen`
/// AND (b) the block intersects [μ − fermi_window, μ + fermi_window] are
/// pinned. Constant-occupation blocks are rotation-invariant for ρ_aug
/// and need no pinning.
///
/// S-metric fallback: if `‖X_block_pinned^H · S_sub_block · X_block_pinned
/// − I‖_F > 1e-8` after the L2 polar pin, switch to S-metric polar via
/// Cholesky `L·L^H = S_sub_block`, SVD `L·X_block = U·Σ·V^H`, then
/// `X_block_pinned = L^{-1} · U · V^H`. This preserves the
/// X^H·S_sub·X = I property that the downstream rotation gemm assumes.
fn pin_fermi_degenerate_blocks(
    eigenvalues: &[f64],
    x_dev: &mut CudaSlice<CudaComplex>,      // n_bands × n_bands col-major (eigenvectors from ZHEGVD)
    s_sub_block_provider: impl Fn(usize, usize) -> Vec<CudaComplex>,
                                              // returns S_sub[block, block] for the S-metric fallback
    n_bands: usize,
    eps_degen: f64,    // 0.01 Ha — eigenvalue spacing threshold
    fermi_window: f64, // 3 × smearing_width ≈ 0.011 Ha
    mu: f64,           // chemical potential
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    // 1. Walk eigenvalues, find contiguous runs [lo, hi) where consecutive
    //    eigenvalues differ by < eps_degen AND the run intersects
    //    [μ − fermi_window, μ + fermi_window].
    //
    // 2. For each qualifying run of size k = hi − lo ≥ 2:
    //
    //    The gauge freedom within a degenerate eigenvalue block is a
    //    k × k unitary R: X[:, lo..hi] · R is also a valid eigenvector
    //    set for the same eigenvalue cluster. We choose R to pin the
    //    block to the input basis (identity column ordering).
    //
    //    Orthogonal Procrustes against the identity target on the
    //    k × k diagonal sub-block:
    //      Let M = X[lo..hi, lo..hi] (the k × k sub-block of X at the
    //      block's row indices). SVD M = U·Σ·V^H. The polar factor of
    //      M is U·V^H. Setting R = V·U^H makes M·R = U·V^H — the closest
    //      unitary to the original M in Frobenius norm, which is the
    //      identity-in-X-coords solution.
    //
    //    Steps:
    //      a. D2H the k × k diagonal sub-block M = X[lo..hi, lo..hi].
    //      b. SVD M = U·Σ·V^H via faer.
    //      c. Compute R = V · U^H (k × k complex).
    //      d. Apply R to the full column slab: X[:, lo..hi] ← X[:, lo..hi] · R.
    //         (Single GPU gemm: m = n_bands, n = k, k_inner = k.)
    //      e. Verify ‖X[lo..hi, lo..hi]^H · S_sub_block · X[lo..hi, lo..hi]
    //         − I‖_F < 1e-8. If fails → S-metric polar:
    //           - Cholesky S_sub_block = L · L^H
    //           - SVD L · M = U' · Σ' · V'^H
    //           - R = V' · U'^H · L (with the L absorbed so M · R is
    //             S-orthonormal); re-apply at step (d).
    //      f. H2D the modified slab back to x_dev.
    //
    // 3. Eigenvalues unchanged (R unitary preserves the diagonal Λ block).
    todo!()
}
```

~70-80 lines including the S-metric fallback path.

**Call site in `rayleigh_ritz`** (replaces lines 226-228 area):

```rust
// After ZHEGVD info check, D2H eigenvalues early (was deferred to line 307)
let eigenvalues_host: Vec<f64> = stream.clone_dtoh(&eigenvalues_dev)
    .map_err(Error::Cuda)?;
pcie.d2h_bytes += eigenvalues_host.len() * std::mem::size_of::<f64>();

// Compute μ inline via the bisection from density.rs::find_chemical_potential.
// Factor that out to a shared helper or inline it (~25 lines, microsecond runtime).
let mu = chemical_potential_bisection(
    &eigenvalues_host, smearing_width, n_electrons,
)?;

let fermi_window = 3.0 * smearing_width;
let eps_degen = 0.01;  // Ha — looser than the smearing width to catch genuine clusters

pin_fermi_degenerate_blocks(
    &eigenvalues_host,
    &mut h_sub_dev,           // ZHEGVD wrote X into h_sub_dev
    |lo, hi| { /* D2H s_sub_dev[lo..hi, lo..hi] on demand */ },
    n_bands,
    eps_degen,
    fermi_window,
    mu,
    stream,
)?;
```

Then the existing `psi_new = psi_row · X` gemm at lines 247-265 runs
unchanged using the pinned `h_sub_dev`.

**Iter-1 caveat:** at iter-1, the input ψ to RR is CASTEP's ψ (loaded
from `.check`). Pinning to identity in X-coordinates pins ψ_new to
CASTEP's ψ — exactly what we want for the Q1 algorithm-fidelity probe.
At iter-2+, the input ψ is the prior pin's output, so the chain is
stable by induction.

**Signature plumbing:** `smearing_width` and `n_electrons` are not
currently in `rayleigh_ritz`'s signature. Either:
(a) Add them as parameters (touch `diagonalize_inner` at `src/scf.rs:592-597`
    and `src/scf.rs:704-718`), or
(b) Bundle them into a small `RrPinConfig { smearing_width: f64,
    n_electrons: f64, eps_degen: f64 }` and pass as `Option<RrPinConfig>`
    so iter-1's first call (where μ doesn't strictly need pinning) can
    pass `None` cleanly. Prefer (b) for clarity.

### Step 3 — Run discriminator tests and verify the diagnostic self-test

```bash
cargo test --release --test ca_scf_convergence \
  cascade_iter3_diagnostic_tight overlap_iter2_against_castep \
  --features scf_diag -- --ignored --nocapture
```

Three possible outcomes:

(a) **Both green** ⇒ proceed to verify Q1 (`iter1_drift_from_castep_state_is_bounded`)
still passes. Then attempt Q2
(`scf_converges_to_castep_energy_at_castep_tolerance`) — if Q2 also goes
green, F3d-narrow is sufficient and §14 closes.

(b) **T-overlap-iter2 green, T-cascade-tight red** ⇒ alignment works but
something downstream still cascades. Likely a stale-state issue (e.g.,
`previous_density` mixing inside `into_phase` chain). Open subsidiary
investigation under same SLUG; do not advance to F3a/b/c yet.

(c) **Both red** ⇒ F3d-narrow insufficient. Promote to F3d-full (no
fractional-occupation gate; align every degenerate block of size ≥ 2 at
any energy). If still red, advance to F3a (deflation-lock) per §14's list.

### Step 4 — RESOLUTION capture and §14 update

Regardless of which F3 candidate finally passes, write
`notes/debug/<slug>/RESOLUTION.md` with:
- Root cause (rotation in fractional-occupation manifolds + USPP
  augmentation, **not** generic subspace-iteration rotation as §14
  framed it).
- Fix location (`rayleigh_ritz.rs` polar pin + signature plumbing).
- Reclassified claim from §14: "rotation cascade is inherent to subspace
  vs band-by-band CG" — partially refuted. Subspace-RR rotation is real
  but only cascades through the augmentation pathway in
  fractional-occupation blocks. Constant-occupation blocks (Cu 3d depth,
  empty bands) are rotation-invariant by construction.
- Append entry to `notes/failure-patterns.md` with pattern label
  `frac-occ-aug-rotation-cascade` documenting the load-bearing insight
  (constant-occ blocks are invariant; only fractional-occ at μ cascades
  via `ρ_aug`). Future debug sessions touching SCF stability for metallic
  USPP systems should grep for this pattern first.
- Update `notes/open-followups.md` §14 status from DEFERRED to RESOLVED
  (or mark partially-resolved + open §16 if F3d-narrow needed escalation
  to F3a).

## Edge cases and risks

1. **μ bisection at iter-1**: eigenvalues from ZHEGVD may differ slightly
   from CASTEP `.bands`. μ derived from them may be off by a few mHa.
   The fermi_window = 3·width = 11 mHa provides ample margin so this
   doesn't shift block membership.

2. **Block fragmentation**: if V_eff changes shift eigenvalues so adjacent
   bands differ by > eps_degen (0.01 Ha), the block fragments and the
   pin becomes a no-op for those bands — correct behavior (genuinely
   non-degenerate eigenvectors are unique up to phase, and a 1-band
   "block" is unaffected by the polar factor since it's already unitary).

3. **faer SVD on small CPU matrices**: blocks are k ≤ 15 typically.
   D2H per block is negligible (≤ 225 × 16 bytes = 3.6 KB). SVD via faer
   takes microseconds. The full `n_bands × k` slab D2H is also small
   (≤ 160 × 15 × 16 bytes = 38 KB).

4. **Phase ambiguity at non-degenerate bands**: ZHEGVD eigenvectors carry
   arbitrary complex phase `e^{iθ}` even for non-degenerate eigenvalues
   (a 1×1 "unitary"). The polar pin only acts on size ≥ 2 blocks, so
   phase drift on non-degenerate bands is unaddressed. This is consistent
   with §14's analysis: |ψ_b|² and (β·ψ)*(β·ψ) are phase-invariant for
   non-degenerate bands. If this turns out to matter empirically (it
   shouldn't), extend the pin to size-1 blocks with a phase-only
   correction `e^{−i·arg(X[b,b])}`.

5. **S-metric polar fallback overhead**: if many blocks need the S-metric
   path, an extra Cholesky per block runs on CPU. Still microseconds for
   k ≤ 15. The fallback should be the exception, not the rule, because
   the S_sub_block → I deviation is bounded by the augmentation strength
   in the block, which is small for high-energy bands far from any
   atomic core.

## Critical files to be modified

- `src/eigensolver/rayleigh_ritz.rs` — new `pin_fermi_degenerate_blocks`
  helper, call site after ZHEGVD info check (~line 226), inline μ
  bisection (or shared call to a helper extracted from
  `density.rs::find_chemical_potential`). Mirror the change in
  `rayleigh_ritz_with_matrices` (~line 487) to keep the test variant
  parallel.
- `src/scf.rs:592-597` and `src/scf.rs:704-718` — pass `RrPinConfig`
  (smearing_width, n_electrons, eps_degen) into `rayleigh_ritz`.
- `Cargo.toml` — add `faer = "0.24"` (or use `nalgebra` already in tree).
- `tests/ca_scf_convergence.rs` — two new tests
  (`cascade_iter3_diagnostic_tight`, `overlap_iter2_against_castep`)
  plus the diagnostic self-test for the overlap helper. Reuse the
  existing per-band overlap routine from
  `eigenvector_overlap_vs_castep_after_filter:1971`, but route through
  `apply_s_times` for S-inner product.
- `notes/debug/<slug>/RESOLUTION.md`, `notes/failure-patterns.md`,
  `notes/open-followups.md` (§14 status update).

## Functions / utilities to reuse

- `compute_occupations` — `src/density.rs:41`. Derives μ via
  `find_chemical_potential` (bisection, line 61). The bisection routine
  is the simplest piece to factor out into a shared helper that both
  density.rs and rayleigh_ritz.rs can call.
- `apply_s_times` — `src/eigensolver/chebyshev.rs:934`. Computes `S·v`
  via the existing β·Q·β^H Woodbury machinery. Both the T-overlap-iter2
  test and the S-metric polar fallback's S_sub_block × small-vector
  product can use it.
- `eigenvector_overlap_vs_castep_after_filter` —
  `tests/ca_scf_convergence.rs:1971`. Already implements per-band overlap
  scaffolding against CASTEP ψ. Adapt for T-overlap-iter2 by swapping
  L2 dot for S-inner.
- `cascade_iter3_diagnostic` — already runs the 3-iter loop and prints
  band-0 per iter; just add the assertion at the end.
- `s_inv_s_identity_test` — `tests/ca_scf_convergence.rs` (search by
  name). Demonstrates the existing pattern for combining β·ψ + Q to form
  the augmentation-aware overlap; the same shape applies to the polar
  fallback's `S_sub_block` extraction.

## Verification (end-to-end)

After F3d-narrow lands and tests are green, the following sequence must
pass:

```bash
# 0. Diagnostic self-test (run once before trusting Step 3 results).
#    Confirm the S-overlap helper gives 1.0 for self, ~0 for orthogonal,
#    and detects column swaps. Run as a unit test or one-off script.

# 1. RR mathematical correctness untouched
cargo test --release --test rayleigh_ritz_validation \
  --features scf_diag -- --ignored --nocapture --test-threads=1

# 2. Q1 algorithm-fidelity probe (must stay green; gate is 20 mHa)
cargo test --release --test ca_scf_convergence \
  iter1_drift_from_castep_state_is_bounded --features scf_diag \
  -- --ignored --nocapture

# 3. New discriminator tests (must turn green from red)
cargo test --release --test ca_scf_convergence \
  cascade_iter3_diagnostic_tight overlap_iter2_against_castep \
  --features scf_diag -- --ignored --nocapture

# 4. Q2 ship gate (was expected-fail; now must pass at 1e-5 eV)
cargo test --release --test ca_scf_convergence \
  scf_converges_to_castep_energy_at_castep_tolerance --features scf_diag \
  -- --ignored --nocapture

# 5. Full release suite — no regressions
cargo test --release
```

If step 4 passes, §14 closes and Q2 becomes the green ship gate. If step 4
fails but steps 1-3 pass, the basis-stabilization fix is correct but
something else (likely density mixing convergence rate) blocks Q2 — that
is a separate issue, file as §16.

## Fallback ladder

If F3d-narrow passes T-cascade-tight and T-overlap-iter2 but Q2 still
doesn't reach CASTEP tolerance:

1. **F3d-full**: drop the Fermi-window gate; pin every degenerate block
   regardless of distance from μ. Same mechanism, broader application.
   Cheap escalation.
2. **F3a**: deflation-lock occupied bands per Saad ch. 8. Heavier;
   requires a per-band convergence test and dynamic `n_active`.
3. **F3c**: Davidson-style residual update (full eigensolver replacement).
   Last resort.

Each step should be tried independently against the same two
discriminator tests before advancing.

## Out of scope for this debug session

- Replacing subspace-RR with band-by-band CG (the CASTEP algorithm).
  That is a much larger refactor and §14 explicitly chose to stay with
  ChFSI.
- Tightening chemrust-hamiltonian D-screening or V_eff residuals
  further. T-prime FAIL already proves these are not load-bearing.
- Performance work (cached Q·SF gemm, etc., listed in §9).
