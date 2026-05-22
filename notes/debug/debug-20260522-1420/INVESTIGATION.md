# Investigation: D_screened explosion + eigenvalue drift

**Symptom**: After iter-1, D_screened for Cu ions jumps from ~5 Ha to ~300-450 Ha.
Eigenvalues drift from [-0.95, 1.36] Ha to [-14.42, 5.45] Ha. SCF diverges.

## Prior-note claim classification

| Claim | Source | Class | Admissible? |
|-------|--------|-------|-------------|
| R-ChFSI norms increasing in iter-1 (ratio > 1) is expected | log observation | DERIVED | No — needs verification |
| D_screened amax > 2× D_0 amax signals V_eff corruption | memory entry | EXTERNAL (memory cites CASTEP nlpot.f90 convention) | Conditionally yes |
| CASTEP uses fine-grid V_eff for D_screened | CASTEP nlpot.f90:346-355 | EXTERNAL | Yes |
| wave_grid V_eff is passed to compute_screened_d_from_fft | code read vnl_data.rs:131-173 | EXTERNAL (code) | Yes |

## Root cause hypothesis (to be confirmed)

`VnlBatchData::precompute` (vnl_data.rs:131-133) FFTs `v_eff_wave` (the
**downsampled wave-grid** V_eff) and passes it to `compute_screened_d_from_fft`.
CASTEP uses the **fine-grid** V_eff (nlpot.f90:346-355).

In iter-1, the fixture V_eff is loaded from `.pot_fmt` (fine-grid resolution),
then downsampled to wave-grid before being passed. The downsampling loses
high-frequency components of V_eff near ion cores — but the error is small
because the fixture V_eff is already converged and smooth.

In iter-2, the newly computed V_eff (from the corrupted density) is on the
fine grid. When downsampled to wave-grid and used for D_screened, the
normalization factor `n_total = ngz*ngy*ngx` uses wave-grid dimensions
(~60k points) instead of fine-grid dimensions (~437k points). This gives a
factor of ~437400/60067 ≈ 7.3× error in the screening integral magnitude.

**This explains the ~50-100× D_screened explosion**: the screening term
ΔD = Σ_G V_eff(G)·Q(G) / n_total uses the wrong n_total.

Wait — actually the grid mismatch is more subtle. Let me re-examine:
- `v_eff_wave.as_fine_array()` returns the wave-grid array (confusingly named)
- The FFT of this wave-grid array has `n_total = n_wave_grid` points
- But `precompute_q_on_grid` also uses `wave_grid` → Q_nm(G) is on wave-grid
- So both V_eff and Q are on wave_grid → the integral is self-consistent on wave_grid

The real CASTEP uses fine-grid for both. Our code uses wave-grid for both.
This is a resolution mismatch, not a normalization mismatch.

**Revised hypothesis**: The wave-grid V_eff misses the sharp features of V_eff
near ion cores (the pseudopotential local part has sharp features that require
fine-grid resolution). This causes ΔD to be underestimated in iter-1 (smooth
fixture V_eff), but in iter-2 the newly computed V_eff has different structure
that, when downsampled, produces aliasing artifacts → D_screened explosion.

**The fix**: Pass the fine-grid V_eff to `VnlBatchData::precompute` and use
`fine_grid` (not `wave_grid`) for both `precompute_q_on_grid` and
`compute_screened_d_from_fft`.
