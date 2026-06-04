# FFI Grid Layout Resolution — First Successful Warm-Start Eigensolve

**Date**: 2026-06-05
**Status**: ✅ RESOLVED — 160/160 bands locked, eigenvalues match standalone test exactly
**Commits**: `b58d33a` (1→0 fft_idx), `22c97a2` (x↔z transposes + D-screening)
**Run**: `slurm_output_2464.txt` at `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0604_warm_start/`

## Milestone

This is the **first successful FFI warm-start eigensolve** — CASTEP calling
`chemrust_diagonalise_h` via cdylib with converged wavefunctions, producing
eigenvalues identical to the standalone Rust test and matching CASTEP's
internal eigensolver.

- Band 0 eigenvalue: −1.055 Ha (matches CASTEP and standalone)
- V_contrib: −1.879 Ha (matches standalone to 6 decimal places)
- 160/160 bands locked
- No eigenvalue explosion, no ZHEGVD failure

## Root Cause: FFT Grid Layout Convention Mismatch

CASTEP uses the Fortran ix-innermost grid convention. The Rust code
(standalone path via `pw_coords_to_fft_indices`) uses iz-innermost.
The FFI boundary was receiving ix-innermost data from CASTEP but
treating it as Rust's iz-innermost convention. Four independent
manifestations of the same convention mismatch:

### Bug 1: fft_idx 1-based → 0-based (`b58d33a`)

CASTEP passes `fft_idx = pw_grid_index(i, nk)` as 1-based Fortran indices
(range 1..grid_size). Rust scatter/gather kernels use 0-based C indices.
grid[0] (G=0 DC component) was never populated; grid[grid_size] was
out-of-bounds. All V_loc real-space operations accessed +1 offset positions.

**Fix**: `ffi.rs` subtract 1 during host-side copy. Added `debug_assert!`
range check.

### Bug 2: fft_idx ix→iz transpose (`22c97a2`)

CASTEP's fft_idx uses ix-innermost flat indexing:
`idx = 1 + ix + ngx·iy + ngx·ngy·iz`

Rust's scatter/gather and cuFFT expect iz-innermost (matching
`pw_coords_to_fft_indices`):
`idx = iz + ngz·(iy + ngy·ix)`

On Cu111_CO (ngx=54, ngy=90, ngz=90 — non-cubic), the x↔z axis swap
maps G-vectors to wrong grid positions, scrambling ψ(r) and reducing
V_loc to the spatial average of V_eff.

**Fix**: Decode ix-innermost → (ix,iy,iz) coordinates, re-encode as
iz-innermost during host-side conversion.

### Bug 3: V_eff x↔z transpose (`22c97a2`)

CASTEP stores V_eff as ix-innermost flat array (Fortran column-major).
cuFFT with plan `(ngx, ngy, ngz)` has n[rank-1]=ngz innermost, matching
the scatter formula `iz + ngz·(iy + ngy·ix)`. Without transposing V_eff
to z-innermost, the potential at grid position (ix,iy,iz) lands at the
wrong flat index relative to what cuFFT expects.

**Fix**: `ndarray::Array3::from_shape_fn((ngz, ngy, ngx), |(iz,iy,ix)| arr[[ix,iy,iz]])`
— identical to the standalone path's transpose at `scf.rs:598-608`.

### Bug 4: D-screening used transposed V_eff (`22c97a2`)

D-screening computes ∫Q·V_eff at ion positions — it needs the physical
(x,y,z) grid layout, not the transposed cuFFT layout. The D matrices
were computed with x↔z-swapped V_eff, corrupting V_NL by ~0.23 Ha.

**Fix**: Pass the original ix-innermost V_eff to `rescreen_d`,
separate from the transposed V_eff used for GPU FFT operations.

## Cumulative Error Budget

| Stage | V_contrib (band 0) | V_loc | V_NL | H_full |
|-------|-------------------|-------|------|--------|
| Before fixes | +0.712 Ha | −0.074 | +0.786 | +1.536 |
| After 1→0 fft_idx | +0.712 Ha | −0.074 | +0.786 | +1.536 |
| After V_eff transpose only | +0.527 Ha | −0.030 | +0.556 | +1.351 |
| After V_eff+fft_idx transpose | −2.110 Ha | −2.666 | +0.556 | −1.285 |
| **After D-screening fix** | **−1.879 Ha** | **−2.666** | **+0.786** | **−1.055** |
| Standalone reference | −1.879 Ha | — | ~0.70 | −1.055 |

The V_NL difference (0.786 vs ~0.70) is the known β_phi per-l sign
convention bug in chemrust-hamiltonian-core — a separate pre-existing
issue (~0.09 Ha, tracked in `chemrust-scf-chebyshev-iter/notes/`).

## Pattern

**Fortran→C grid layout convention mismatch**: CASTEP's ix-innermost
(Fortran column-major) grid convention differs from Rust's iz-innermost
convention. Three independent data channels (ψ FFT indices, V_eff grid,
D-screening integrals) each need explicit convention conversion at the
FFI boundary. The cumulative error from all three mismatches was 2.6 Ha —
two orders of magnitude larger than any individual numerical tolerance.

**Lesson**: When receiving multi-dimensional grid data across a
Fortran→C FFI boundary, audit EVERY channel independently:
- Integer index arrays: verify base (0 vs 1) AND axis ordering
- Real-valued grid arrays: verify axis ordering AND verify against
  a known reference (shape reshape must be identity-transpose, not
  identity-reshape)
- Derived quantities (integrals, screening): verify they use the
  CORRECT grid layout (physical vs FFT-transposed)

**Resolution**: `notes/ffi-grid-layout-resolution.md`

## Warm-Start Test

The warm-start test (`/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0604_warm_start/`)
is now a **critical regression gate**. It feeds converged CASTEP
wavefunctions through the FFI boundary and verifies that the Rust
eigensolver produces eigenvalues matching the standalone test. This
test isolates the FFI data transfer from eigensolver correctness.

## Related

- [[failure-patterns]] § 2026-06-04: ffi-fortran-1-based-index-mismatch-in-scatter-gather
- [[failure-patterns]] § 2026-05-20: cufft-dim-ordering-and-rr-transpose-layout
- `chemrust-scf-chebyshev-iter/notes/vnl-error-isolation-status.md`: β_phi per-l sign bug
