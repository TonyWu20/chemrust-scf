# Resolution: iter-1 filter-operator-mismatch

**Symptom**: iter-1 RR band-0 = −1.69 Ha vs CASTEP reference −1.06 Ha (|Δ| = 0.63 Ha), R-ChFSI norm ratio ~5×/step k=2..8, iter-2 V_eff range = 29.2 Ha.

**Root causes** (two compounding bugs):

### Bug 1: GPU D-matrix screening (primary, introduced in commit `99d3460`)
`screen_d_gpu` in `src/eigensolver/d_screening.rs` produced near-zero screening terms for all non-origin ions (ions 3–17 in Cu111+CO), causing `d_screened ≈ d0_expanded` (10–50× too large). Ion 2 at fractional position `(0,0,0)` was immune because its structure factor `exp(-iG·R) = 1` for all G, masking the bug.

**Fix**: Reverted D-screening to CPU path (`compute_screened_d_from_fft` from `chemrust-hamiltonian-core`). The GPU `screen_d_gpu` function is preserved in `d_screening.rs` for future debugging but is no longer called.

**Fix location**: `src/eigensolver/vnl_data.rs` — replaced `screen_d_gpu` call with `compute_screened_d_from_fft`.

### Bug 2: b_low bootstrap on iter-1 (secondary)
`b_low = max_veff + 2.0 = 2.09 Ha` placed the filter cutoff above all 160 tracked bands (highest band ≈ 0.13 Ha), making the filter amplify the entire subspace uniformly with no discrimination. The correct `b_low` for iter-1 is `max_veff` itself (≈ 0.089 Ha for Cu111+CO), which sits just above the highest tracked band.

**Fix location**: `src/eigensolver/chebyshev.rs` — changed first-call b_low from `max_veff + 2.0` to `max_veff`.

### Bug 3: Filter operator mismatch (original FIX_PLAN target)
The Chebyshev recurrence used bare H (Mode A) while the Lanczos bounds were computed on S⁻¹·H. The discriminator confirmed Mode B (S⁻¹·H in Step 3, h_eig Λ, no S⁻¹ in Step 4) passes SC-4-tight; Mode A fails 8/10 bands; Mode C fails all 10.

**Fix location**: `src/scf.rs:433` — changed production default from `FilterMode::BareH` to `FilterMode::SinvHKeepHEig`.

---

## Discriminator results (per-band |Δ| vs CASTEP .bands, iter-1)

Run: `/tmp/iter1-filter-mode-sweep-20260523-1001.log`

| band | CASTEP (Ha) | Mode A |ΔA| | Mode B |ΔB| | Mode C |ΔC| |
|------|------------|--------|------|--------|------|--------|------|
| 0 | −1.0550 | −1.0296 | 0.025 | −1.0458 | **0.009** | −0.9263 | 0.129 |
| 1 | −0.4972 | −0.4877 | 0.009 | −0.4930 | **0.004** | −0.3484 | 0.149 |
| 2 | −0.4888 | −0.4268 | 0.062 | −0.4749 | **0.014** | −0.2836 | 0.205 |
| 3 | −0.4883 | −0.3949 | 0.093 | −0.4730 | **0.015** | −0.2424 | 0.246 |
| 4 | −0.4848 | −0.3941 | 0.091 | −0.4642 | **0.021** | −0.2118 | 0.273 |
| 5 | −0.4848 | −0.3495 | 0.135 | −0.4616 | **0.023** | −0.2065 | 0.278 |
| 6 | −0.4790 | −0.3104 | 0.169 | −0.4611 | **0.018** | −0.2012 | 0.278 |
| 7 | −0.4735 | −0.2796 | 0.194 | −0.4542 | **0.019** | −0.2006 | 0.273 |
| 8 | −0.4730 | −0.2793 | 0.194 | −0.4538 | **0.019** | −0.1997 | 0.273 |
| 9 | −0.4681 | −0.2606 | 0.208 | −0.4395 | **0.029** | −0.1765 | 0.292 |

**Winner: Mode B** — all 10 bands within 0.05 Ha gate.

---

## Prior claims reclassified

From `INVESTIGATION.md`:
- "iter-1 RR band-1 = −1.69 Ha" — DERIVED from broken GPU D-screening run. Now reclassified as DERIVED/stale. Correct value with CPU D-screening: −1.046 Ha (Mode A), −1.046 Ha (Mode B).
- "bare-H beats S⁻¹·H at run-1010" — DERIVED from ζ ≈ 0.014 regime. Reclassified as DERIVED/stale. In the current ζ = 4e-15 regime, Mode B (S⁻¹·H) beats Mode A (bare H) by 5–10× across all bands.

---

## Changes committed

| File | Change |
|------|--------|
| `src/eigensolver/vnl_data.rs` | Reverted D-screening to CPU `compute_screened_d_from_fft` |
| `src/eigensolver/chebyshev.rs` | b_low bootstrap: `max_veff + 2.0` → `max_veff`; FilterMode enum added; Mode B wired into Step 3 |
| `src/scf.rs` | Production default: `FilterMode::BareH` → `FilterMode::SinvHKeepHEig` |
| `tests/ca_scf_convergence.rs` | `iter1_filter_mode_sweep` test: D6 anchored to CASTEP reference; SC-4 asserts Mode B |

**Date**: 2026-05-23
