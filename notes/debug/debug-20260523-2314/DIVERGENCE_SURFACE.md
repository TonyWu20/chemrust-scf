# Divergence Surface: D-screening Pipeline vs CASTEP

Enumerates every plausible point where our `compute_screened_d` could diverge
from CASTEP's `nlpot_calculate_d`. Classification per Step 4 rules.

## Legend

- **Ruled out by anchor X** — EXTERNAL anchor eliminates this possibility
- **To be tested** — not yet ruled out; will be investigated

---

## 1. Data layout / axis ordering

| # | Divergence point | Classification |
|---|-----------------|----------------|
| 1a | Fortran column-major vs Rust row-major in D matrix indexing | **To be tested** — T2 compares element-by-element; any mismatch surfaces as max|Δ| |
| 1b | Upper-triangle storage order: CASTEP loops `dn=1..np, dm=dn..np` writing `nl_d(dm,dn)` | **To be tested** — parser must mirror to `(dn,dm)` for symmetry; verified by parser round-trip test T0 |

## 2. Normalization / scaling conventions

| # | Divergence point | Classification |
|---|-----------------|----------------|
| 2a | `mixture_weight` scaling: CASTEP dumps `nl_d × w_mix` | **Ruled out by A2/E10** — `w_mix = 1.0` for Cu111+CO; parser divides by it defensively |
| 2b | `spin_deg` factor in D_screened | **Ruled out by E3/E4** — spin_polarized=false, no spin factor applies |
| 2c | Cell volume scaling in Q integral | **To be tested** — if our Q normalization differs from CASTEP's by a volume factor, per-element Δ would be uniform across all ions |

## 3. Sign / direction conventions

| # | Divergence point | Classification |
|---|-----------------|----------------|
| 3a | Sign of `∫Q·V_eff` term added to D_0 | **To be tested** — T2's per-element comparison catches sign flips (Δ ≈ 2×|value|) |
| 3b | Complex conjugate convention in structure factor | **To be tested** — CASTEP uses `conjg(SF)` in some contexts; if we conjugate the wrong factor, cross-terms are wrong |
| 3c | FFT sign convention (forward vs inverse) | **To be tested** — affects Q-on-grid computation, which has already been validated for density correctness (E2). If Q-on-grid had a sign bug, density would also be wrong. BUT: density test E2 uses CASTEP ψ, so Q-on-grid errors could cancel in density while persisting in D-screening. |

## 4. Boundary / edge-case handling

| # | Divergence point | Classification |
|---|-----------------|----------------|
| 4a | `q.norm_sqr() < 1e-60` skip in our `compute_screened_d` | **HYPOTHESIZED harmless (H3)** — T2 tests empirically whether skipping near-zero Q causes mismatch. If T2 PASSES, this skip is confirmed harmless. If T2 FAILS and delta pattern shows cross-m pairs (m≠n), this skip is the cause |
| 4b | G-vector sphere cutoff: differences in which G-vectors are included | **Ruled out by E2** — density code uses the same G-vector set and is correct for CASTEP ψ. However: D-screening uses a DIFFERENT set of Q-on-grid G-vectors (species-dependent, not density-dependent). |
| 4c | Radial quadrature grid for β projectors | **To be tested** — if our βᵢ(r) evaluation uses different radial grid than CASTEP's pseudopotential generation grid |
| 4d | Zero G-vector handling in Q→real-space transform | **To be tested** — CASTEP may handle G=0 component differently in Q-on-grid |

## 5. Unit conversion at boundaries

| # | Divergence point | Classification |
|---|-----------------|----------------|
| 5a | Q units: CASTEP Q̃(G) in Bohr³ vs our convention | **Ruled out by E2** — density augmentation uses same Q-on-grid; if Q units were wrong, aug density would not match CASTEP F8 (ratio 1.000084) |
| 5b | V_eff units: Ha vs eV | **To be tested** — `.pot_fmt` stores in Ha; our parser reads as Ha. If a unit conversion is applied spuriously, D_screened scales uniformly |
| 5c | G-vector units: Å⁻¹ vs Bohr⁻¹ | **Ruled out by E2** — density FFT uses these; if G-vectors were wrong, the entire reciprocal-space computation would be wrong |

## 6. Parser precision / offset assumptions

| # | Divergence point | Classification |
|---|-----------------|----------------|
| 6a | `.pot_fmt` parsing: grid dimension ordering | **To be tested** — if V_eff is read with wrong axis ordering, ion-centre values are wrong → D_screened wrong |
| 6b | `D_band_debug.dat` parsing: record count per block | **Ruled out by T0** — parser round-trip test verifies parsing correctness |
| 6c | `.pot_fmt` upsampling/downsampling to wave-grid | **To be tested** — if the upsampling introduces interpolation error at ion centres where Q is sharply peaked |

## 7. Decomposition / parallel artifacts

| # | Divergence point | Classification |
|---|-----------------|----------------|
| 7a | MPI vs serial: our code is serial, CASTEP ran MPI | **Ruled out by CASTEP dump location** — dump runs on `on_root_node` only, so `D_band_debug.dat` is the gathered/serial result |

## 8. Algorithm-specific divergence

| # | Divergence point | Classification |
|---|-----------------|----------------|
| 8a | Q-on-grid precomputation: our `precompute_q_on_grid` vs CASTEP's inline computation in `nlpot_calculate_d` | **To be tested** — T2 tests the end-to-end result; if Q-on-grid precomputation is wrong but V_eff-on-grid is correct, T2 catches it |
| 8b | Structure factor S_I(G) = Σ_i exp(-i G·R_i): ion position indexing | **To be tested** — if ion positions are read with wrong fractional→Cartesian conversion |
| 8c | CASTEP's `nl_d` is accumulated over SCF: `nl_d(m,n) += ps_D0(m,n)` vs our fresh computation each call | **Ruled out by E10/A5** — CASTEP dump is per-iteration (append mode), not accumulated. Each block in `D_band_debug.dat` is the D_screened for that SCF iteration only |

## Priority for Investigation

Ordered by likelihood × impact:

1. **4a** (Q-skip) — easiest to test, directly in the hot path
2. **6c** (upsampling interpolation at ion centres) — Q is sharply peaked, interpolation error at ions is the most likely subtle bug
3. **2c/5b** (scaling/normalization) — would produce uniform mismatch, easy to detect
4. **3b/3c** (conjugate/FFT sign) — would produce structured mismatch patterns
5. **8b** (ion position indexing) — would produce ion-dependent mismatch
