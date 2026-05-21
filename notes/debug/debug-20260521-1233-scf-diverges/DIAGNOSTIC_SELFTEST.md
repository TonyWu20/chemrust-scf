# Diagnostic Self-Test: SCF Diverges After Iter-2

## Diagnostics in scope

Three diagnostics emit data we have been treating as ground truth in this
session. Each must be self-verified before downstream reasoning trusts it.

### D1 — `aug_density_gpu_matches_cpu_cu111_co` integration test
- **Claims**: GPU `compute_aug_density_gpu` produces identical ρ_aug as CPU
  `compute_aug_density_fine` for the Cu111_CO fixture (`max_diff < 1e-6`,
  `sum_diff < 1e-4 × |∫ρ_aug|`).
- **Self-test result**: ⚠ **STRUCTURALLY UNSOUND — does not exercise the
  production data path**.

**Reasoning** (paper analysis — code paths read at
`tests/ca_scf_convergence.rs:311-319` and `src/density.rs:475-499`):

The test constructs `beta_psi_gpu` from `Array2::zeros((n_expanded,
n_bands))` (default ndarray C-order, row-major) via `arr.iter()`:

```rust
let flat: Vec<CudaComplex> = arr.iter()...collect();
stream.clone_htod(&flat)
```

`arr.iter()` for a C-order Array2 iterates in memory order:
`flat[i*n_bands + j] = arr[i, j]`. So `bp_dev_test` has row-major
(n_e, n_bands) layout.

`compute_aug_density_gpu` then does:
```rust
let bp_host: Vec<CudaComplex> = stream.clone_dtoh(bp_dev)...;
let bp = Array2::from_shape_vec((n_expanded, n_bands).f(), bp_complex)?;
```

`(n_e, n_bands).f()` declares F-order (col-major) strides `(1, n_e)`. So
`bp[i, j] = bp_complex[i + j*n_e]`. For the row-major test buffer, this
indexing **does NOT recover the original `arr[i, j]`** — it indexes a
permuted location: `arr[(i + j*n_e) / n_bands, (i + j*n_e) mod n_bands]`.

**The production path** (`src/scf.rs:715-721` calling
`compute_aug_density_gpu`) sources `bp_dev` from
`rayleigh_ritz`'s gemm:
- gemm config: `transa=op::C, lda=n_pw, ldc=ne` → output is col-major
  (ne × n_bands).
- `bp_dev[i + j*ne] = result[i, j]`. ✓ matches the F-order reinterpret.

So the production code is layout-consistent, but **the test does not
exercise it** — the test feeds a row-major buffer to a function that
expects col-major.

**Required outcome of test re-run with cache deleted**:
- If `max_diff` reported is small (< 1e-6), the test is somehow
  numerically masking the layout swap (or the cached CPU output was
  written with the same wrong layout assumption — unlikely since the
  CPU path uses layout-agnostic `bp[[n,b]]` indexing).
- If `max_diff` is large, the test is broken as I claim.

**Verdict**: this test cannot be used as evidence that
`compute_aug_density_gpu` works correctly in the production hot path.
It tests a different data path.

### D2 — `[AugDensity] aug_sum=...` print at `src/scf.rs:757-761`
- **Claims**: ρ_aug integral matches expected `∫ρ_aug dV ≈ 3.12e6`.
- **Self-test result**: ⚠ **SUMMARY-ONLY, no per-point backing**.

Per ODD pattern Diagnostic Self-Verification (Per-Point vs Summary
Diagnostics table): `aug_sum` masks spatial distribution errors. A wrong
spatial distribution can give correct integral. The diagnostic is
admissible only with an independent per-point comparison.

### D3 — `[V_eff] min=..., max=..., range=...` print
- **Claims**: V_eff range matches iter-1 vs iter-2.
- **Self-test result**: ⚠ **SUMMARY-ONLY, no per-point backing**.

Same issue as D2: range matches while spatial distribution diverges.
This is precisely what makes the iter-2 → iter-3 transition look
"healthy" by V_eff range (8.55 vs 8.69 Ha) yet Lanczos sees a wildly
different H.

## Cross-path verification — not feasible without code changes

To self-test D1 properly, the cleanest approach is:
1. Build `β_g` and `ψ` from the .check fixture (real geometry).
2. Compute `bp_via_gemm` using the same `gemm_c64` call as production.
3. Compute `bp_via_cpu_brute_force` by `bp[n, b] = Σ_G conj(β_g[n, G]) ·
   ψ[G, b]` (explicit logical sum).
4. Assert `bp_via_gemm[n, b] == bp_via_cpu_brute_force[n, b]` for all n, b.

If these match, layout is OK and the bug lies elsewhere. If they
differ, we have the bug. Either way, this is a NEW test to write —
neither path currently exists in the codebase.

## Sanity-check against physical intuition

The reported aug_sum (6.117e7) gives ∫ρ_aug dV ≈ 3.12e6, which matches
`N_e_aug × Ω` where `N_e_aug ≈ 0.75 × 186 ≈ 140` electrons. This passes
the integral check.

But the **smoking gun** is that aug_sum **changes** between iterations:
6.117e7 → 2.286e7 → 5.940e7 across iters 2-4. If βψ were correct and
stable, aug_sum should be stable too (the wavefunctions converge to a
fixed point).

The integral-of-ρ_aug oscillation suggests ω^I (= Σ_b occ_b · bp̄·bp)
is fluctuating between iterations because:
(a) ψ itself is unstable (caused by some upstream bug), OR
(b) bp computation is wrong AND iteration-dependent.

The Lanczos b_up jump (22.8 → 107.5 → 128 Ha) is the leading indicator
that H itself is unstable, which traces back to V_eff spatial values
(despite range stability) which traces back to ρ_aug spatial values.

## Conclusion

The diagnostics currently in use cannot validate the spatial distribution
of ρ_aug. Before any further hypothesis testing, a per-point comparison
of `bp_via_gemm` against `bp_via_cpu_brute_force` must be written. This
is the smallest unit test that can either confirm or deny the layout
hypothesis.
