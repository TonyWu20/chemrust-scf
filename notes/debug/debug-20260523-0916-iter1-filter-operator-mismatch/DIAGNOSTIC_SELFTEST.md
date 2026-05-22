# Diagnostic Self-Test Plan

Per the debug-outcomes Step 5 rule: any diagnostic used to interpret the A/B/C
results must itself be verified against the external anchors before its
output is trusted.

---

## Diagnostics in scope

| Diagnostic | Purpose | Code path |
|------------|---------|-----------|
| D1 | iter-1 RR eigenvalue extraction (per band) | `tests/ca_scf_convergence.rs::iter1_filter_mode_sweep` (new) → `ScfIteration::diagonalize()` → `chebyshev_filter` → `rayleigh_ritz` → returned `eigenvalues_cpu` |
| D2 | CASTEP `.bands` parser (read 10 lowest band energies in Hartree) | New helper in test, parses `Cu111_CO.bands` plain-text |
| D3 | Per-band Δ comparison + per-mode CSV emission | Test body |
| D4 | R-ChFSI norm-ratio printout (`[R-ChFSI] k=N norm_prev=... norm_curr=... ratio=...`) | `chebyshev.rs:1626-1627` (already present, gated `scf_diag`) |
| D5 | Spectral parameter printout (b_up, b_low, c, e, σ, lambda_min) | `chebyshev.rs:1402-1406` (already present, gated `scf_diag`) |
| D6 | Mode-A reproducibility check | Test body — first run of the sweep must reproduce `band_1 = −1.69 Ha` from `/tmp/scf-diag-global-woodbury-0523-0524.log:65` |

---

## Self-test for each diagnostic

### D1: iter-1 RR eigenvalue extraction

**Risk:** the test invokes `diagonalize` which returns eigenvalues; we trust
the returned slice directly. The slice is produced by `cusolverDnZhegvd` inside
`rayleigh_ritz`. Bug-class risk: indexing (n_bands offset), unit (eV vs Ha),
ordering (ascending vs descending), or eigenvector layout (col-major vs
row-major).

**Self-test:**
1. **Path A:** read the eigenvalue array from `diagonalize().eigenvalues`.
2. **Path B:** for one selected band j, recompute Rayleigh quotient on CPU:
   `ε_j ≈ ⟨ψ_j, H·ψ_j⟩ / ⟨ψ_j, S·ψ_j⟩` using a reference H·ψ and S·ψ from
   `apply_full_hamiltonian` and `apply_s_times` (call the kernels through the
   public test API). Pull both values D2H and compute on CPU with f64.
3. **Assertion:** `|ε_j(A) − ε_j(B)| < 1e-6 Ha` for j ∈ {0, 5, 9}.

If the assertion fails, the `diagonalize` output is suspect and the entire
sweep is meaningless. Fix D1 before reading B/C.

### D2: CASTEP `.bands` parser

**Risk:** wrong file format, wrong unit, wrong band ordering, wrong k-point.

**Self-test:**
1. The `.bands` header includes `Number of k-points`, `Number of spin
   components`, `Number of electrons`, and `Fermi energies (in atomic units)`.
   Assert: parsed n_kpoints = 1 (Γ-only fixture); n_spins = 1 (non-spin-polarized).
2. Parse the eigenvalues block; expect 160 bands at k-point 1.
3. Compare band-1 against the known reference value −1.055 Ha. If the parsed
   value disagrees with `-1.055 ± 0.001`, the parser is broken.
4. Compare band-160 against a known upper band reference (read from the file
   directly during test setup, then assert it against the same parse). This
   second check guards against silently truncating the band list.

The reference value −1.055 Ha for band-1 comes from
`notes/plans/phase-rchfsi-bare-h/TASKS.md:21` and is stable across runs of
the same fixture (verified: it appears in multiple session logs in
`notes/debug/debug-20260520-1923/` etc.). If a parser self-test reads
−1.055, both the file format and the parser logic are correct.

### D3: per-band Δ comparison

**Risk:** mode mislabeling, mode contamination (one mode's eigenvalues showing
under another mode's tag), order-of-operations bug in the assertion.

**Self-test:**
1. Print mode tag *before* the eigenvalue extraction in the test loop, so the
   stderr stream interleaves `[Mode X] band_j = ...` deterministically.
2. CSV emission to `/tmp/iter1-mode-sweep-<timestamp>.csv` with columns
   `mode, band_idx, eigenvalue_Ha, castep_ref_Ha, abs_delta_Ha, pass_at_0.05`.
3. Run Mode A, then Mode A again (same mode twice). Both runs must produce
   bit-identical CSV rows. Any drift between two A-runs means mode state is
   leaking between calls (e.g., a static cache).

### D4: R-ChFSI norm-ratio printout

**Risk:** already-present diagnostic; just confirm it surfaces under each mode.

**Self-test:** the existing log shows growth ~5×/step under Mode A. After
flipping to Mode B, expect the ratio to drop dramatically (toward the
Chebyshev polynomial's expected expansion factor of `≈ (b_up + b_low)/(b_up −
b_low)` per step, which for 21.20/7.34 = 1.4× growth — orders of magnitude
slower than the observed 5× under bare H). If Mode B still shows 5×/step, A1
is *not* the bug and the root cause is elsewhere; do not commit Mode B.

### D5: spectral parameter printout

**Risk:** σ becomes infinite if `bounds.lambda_min` ≈ `bounds.center`, making
the recurrence unstable independent of operator.

**Self-test:** assert `|σ| < 10` and `bounds.half_width > 0.5 Ha` for every
mode in the sweep. The current log shows σ implicitly via the printed center
(14.27) and half_width (6.93), giving σ = 6.93/(−0.41 − 14.27) ≈ −0.47 — sane.
Modes B/C should produce similar values since they share the Lanczos bound.

### D6: Mode-A reproducibility check

**Risk:** the FilterMode enum scaffolding inadvertently changes Mode A's
behavior. If Mode A in the new test produces something other than band-1 ≈
−1.69 Ha, the test infrastructure is the bug, not the algorithm.

**Self-test:** assert `|band_1_ModeA − (−1.69)| < 0.1 Ha` as the *first*
assertion in the test. Only proceed to read Mode B and Mode C if Mode A
reproduces the production-path value within 0.1 Ha. (Tolerance allows for
nondeterminism from CUDA reduction order; ±0.1 Ha is wide enough to absorb
that without admitting algorithmic difference.)

---

## Sanity-check against physical intuition

For Cu111+CO at convergence:
- Band 1 (deepest) ≈ −1.06 Ha (Cu 4s/3d state, deep)
- Band 10 (still occupied) typically in the [−0.5, −0.3] Ha range
- Band-160 (top of subspace) typically positive, < 5 Ha (otherwise the filter
  has way more bands than physically meaningful)
- Fermi energy ≈ 0.18 Ha (read from `.bands` header)

If a mode produces band-1 = +50 Ha or band-160 = −100 Ha, that fails the
smell test before we even apply SC-4-tight — the diagnostic has a bug or the
mode produces an unphysical answer. Reject and investigate.

---

## Brute-force fallback

Per debug-outcomes Step 7 sub-step 0 and the brute-force guard: if D1–D6 all
self-test green but two cycles of A/B/C produce no clear winner (all three
within 0.1 Ha of each other but none below 0.05 Ha), report and request user
guidance before proceeding. Possible brute-force probe: dump every iter-1 RR
eigenvalue (all 160) to a CSV and visually compare against `.bands`. The
divergence pattern (constant offset? scaling? specific bands wrong?) narrows
the upstream suspect.

The `iter1_filter_mode_sweep` test should be written so it can dump all 160
bands when given a `--features bands_full_dump` flag, even if the routine
gate only checks 10. This keeps the brute-force option one-flag away.
