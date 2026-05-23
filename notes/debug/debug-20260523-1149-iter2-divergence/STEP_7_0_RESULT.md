# Step 7.0 Result — ndeg=0 baseline

**Date:** 2026-05-23
**Test:** `tests/ca_scf_convergence.rs::ndeg_zero_with_castep_psi_matches_bands`
**Outcome:** GREEN

## Per-band table

| band | CASTEP (Ha) | ndeg=0 (Ha) | |Δ| (Ha) | Gate 0.05 |
|------|-------------|-------------|---------|-----------|
| 0 | -1.055023 | -1.045966 | 0.009057 | PASS |
| 1 | -0.497212 | -0.509915 | 0.012703 | PASS |
| 2 | -0.488772 | -0.490003 | 0.001231 | PASS |
| 3 | -0.488264 | -0.488713 | 0.000449 | PASS |
| 4 | -0.484833 | -0.485903 | 0.001069 | PASS |
| 5 | -0.484757 | -0.485002 | 0.000245 | PASS |
| 6 | -0.478984 | -0.478239 | 0.000745 | PASS |
| 7 | -0.473451 | -0.473221 | 0.000230 | PASS |
| 8 | -0.473009 | -0.473184 | 0.000175 | PASS |
| 9 | -0.468084 | -0.467739 | 0.000345 | PASS |

Mode A/B/C identical to 1e-10 (filter body never executes when ndeg=0 → mode dispatch is dead code).

Last band: 0.131108 Ha (matches §11's iter-1 "last band 0.130 Ha" within 1 mHa).

## Diagnosis

**The bug is INSIDE the Chebyshev filter recurrence** (`chebyshev.rs:1451-1697`).

Downstream pipeline (RR, S_sub assembly, ZHEGVD, β·ψ caching, ψ rotation, density assembly) is correct on this fixture input. The §10 fixes hold.

## Ruled-OUT items (move from To-be-tested to Ruled out in DIVERGENCE_SURFACE.md)

- Data layout / axis ordering (ψ ColumnDistributed handoff) — RR produces correct bands, so handoff is correct
- ZHEGVD X B-orthonormality → ψ_new S-orthonormality — RR produces correct bands
- S_sub assembly (`rayleigh_ritz.rs:104-202`) — RR produces correct bands
- Gram-Schmidt S-inner product (`chebyshev.rs:1700-1773`) — runs after the recurrence; if recurrence is bypassed and result is correct, GS isn't the issue *for ndeg=0*. (Still worth checking for ndeg>0 — GS could mask or interact with recurrence-body bugs.)
- β_g phase / structure-factor at iter-2 — irrelevant for iter-1-only fixture run, but the precompute path is exercised and ψ→bands round-trips correctly
- §10 G2 cross-ion imaginary parts — corroborated by clean RR result

## Confirmed primary candidates (narrow Step 8 to these)

`src/eigensolver/chebyshev.rs:1451-1697`:
- Step 1 residual `Y = H·X − S·X·Λ` (`chebyshev.rs:1474-1500`)
- `h_eig` per-band Rayleigh quotients (`chebyshev.rs:1503-1538`)
- Spectral parameters `(σ, c, e, γ)` derivation (`chebyshev.rs:1540-1548`)
- `lam_y` initial value mode-dependence (`chebyshev.rs:1578-1587`)
- Step 3 recurrence loop sign/coefficient correctness (`chebyshev.rs:1593-1679`)
- Step 4 reconstruction mode-dependence (`chebyshev.rs:1681-1697`)

Next: Layer B (Step 7.1) — synthetic H controlled test of the recurrence body alone.
