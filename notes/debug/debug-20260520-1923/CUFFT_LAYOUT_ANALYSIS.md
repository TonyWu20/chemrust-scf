# Phase D — cuFFT layout audit

## Reference convention (cuFFT)

cuFFT plan_many docs (CUDA Toolkit reference): "cuFFT uses C-style (row-major)
layout. For a multi-dimensional transform of rank `R` with sizes `n[0]…n[R-1]`,
the data is stored such that `n[0]` is the slowest-varying (outermost) dim and
`n[R-1]` is the fastest-varying (innermost) dim."

That is: cuFFT treats `flat[k*n[1]*n[2] + j*n[2] + i] = arr[k][j][i]` where
`i ∈ [0, n[2])` is innermost.

In `src/device/fft.rs:131` the docstring on `plan_batched_z2d` says the
opposite ("`n[0]` innermost"). That comment is wrong; the **code** in
`plan_batched_z2d` correctly assumes `n[rank-1]=nz` is the half-compressed
innermost dim (input shape `[nx, ny, nz_half]`), so the comment is contradicted
by the code itself. The comment is a documentation bug; the **layout
convention used by cuFFT is the standard one** (`n[rank-1]` innermost).

## Cu111_CO concrete numbers

- Wave grid (per `wave_grid.grid()` in `chemrust_hamiltonian_core::GVectorGrid`):
  `[ngz, ngy, ngx] = [90, 90, 54]`. Lattice: a=10.225 Å (54), b=17.710 Å (90),
  c=18.261 Å (90).
- V_eff source: `Cu111_CO.pot_fmt`, parsed by `parse_pot_fmt` into
  `Array3<f64>` shape `(ngx=54, ngy=90, ngz=90)` C-layout, indexed
  `arr[ix, iy, iz]` = V_eff at lattice point (a-idx ix, b-idx iy, c-idx iz).
  Flatten via `.iter()` → flat layout `flat[iz + 90*(iy + 90*ix)] =
  V_eff(ix, iy, iz)` — **iz (c-axis) is innermost**, period 90.

## Pipeline layouts

### V_eff buffer (uploaded via `Gpu::from_host_with`)

`flatten_f64(arr)` calls `arr.iter()` which is logical row-major over (axis 0,
axis 1, axis 2). For pot_fmt's C-layout `(ngx, ngy, ngz)`, this yields
**flat[iz + ngz*(iy + ngy*ix)] = V_eff(ix, iy, iz)** with iz innermost
(period ngz=90).

### Wavefunction scatter (`scatter_pw_to_grid`)

`pw_coords_to_fft_indices` (src/scf.rs:1015–1029) computes
`flat = iz + ngz*(iy + ngy*ix) = l + 90*(k + 90*h)` for each PW (h, k, l).
Innermost = iz, period 90. Matches V_eff layout. ✓

### cuFFT plan (`plan_batched_c2c`)

Called at `src/eigensolver/chebyshev.rs:844` and `src/density.rs:147` as
`plan_batched_c2c(ngz, ngy, ngx, batch, stream)`. Inside `plan_batched_c2c`
(`src/device/fft.rs:177`) these become `&[nx=ngz, ny=ngy, nz=ngx]` passed to
`cufftPlanMany` as `n[0..3]`.

cuFFT's convention: `n[2]=ngx=54` is innermost; cuFFT expects
`flat[i + ngx*(j + ngy*k)] = arr[k][j][i]` with `i ∈ [0, ngx=54)` innermost.

**Mismatch**: our actual layout has innermost period 90 (=ngz, =ngy), but
cuFFT thinks innermost period is 54 (=ngx).

### Real-space output (`Array3::from_shape_vec`)

In `density.rs:180`: `Array3::from_shape_vec((ngx, ngy, ngz), rho_host)` —
C-layout shape `(ngx, ngy, ngz)`, so flat layout `flat[ix*ngy*ngz + iy*ngz +
iz]` = `arr[ix, iy, iz]`. Innermost = iz (period 90). ✓ Matches scatter, but
**not** matches cuFFT — same bug.

## Why the bug doesn't show on cubic grids

All unit tests use `ngx = ngy = ngz`. When ngx == ngz the cuFFT-vs-data
strides are equal and the FFT computes the same numerical result regardless
of which axis labels are which. The dim labels matter for non-cubic.

## Numerical signature of the bug

For Cu111_CO (ngx=54, ngy=90, ngz=90), the FFT axes are misaligned:
- cuFFT's "innermost" axis (length 54) is run over data whose innermost
  stride period is 90.
- Since 90 is not a multiple of 54, the 1-D FFTs along cuFFT's nominal
  innermost axis include data points that, in the physical lattice, lie
  along combinations of the c-axis and the a-axis.
- The forward+inverse FFT roundtrip with V_eff multiplication in between is
  **not** a simple permutation — it's a rotation in a basis cuFFT made up.
- Result: ⟨ψ|V_loc|ψ⟩ comes out approximately **−3.13 Ha** vs reference
  −1.06 Ha. The 3× ratio is consistent with mixing of components across
  axes of disparate sizes.

## Proposed fix

Change the cuFFT plan dims so that **cuFFT's innermost dim equals our
physical innermost stride period**. Our scatter and V_eff use
`flat[iz + ngz*(iy + ngy*ix)]` with `iz` innermost (period ngz=90).
Therefore cuFFT's `n[2]` must equal `ngz`. The call should be:

```rust
let [ngz, ngy, ngx] = wave_grid.grid();
// cuFFT dims = [outermost=ngx, middle=ngy, innermost=ngz]
let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
    ngx as i32, ngy as i32, ngz as i32, n_bands_i32, stream.clone(),
)?;
```

(swap of ngz and ngx in the arg list)

This matches the call style at `src/density.rs:147` once the same swap is
applied there. Keep the function signature `(nx, ny, nz)` semantically
labelled as `(outermost, middle, innermost)`.

The call site swap is a one-token change. The function signature does not
need to change. The contradictory comment on `plan_batched_z2d:131` should
be fixed as a bonus (separately).

## Why this works

After the swap, cuFFT interprets the buffer as
`flat[i + ngz*(j + ngy*k)] = arr[k][j][i]` with `i ∈ [0, ngz=90)` innermost,
`j ∈ [0, ngy=90)`, `k ∈ [0, ngx=54)`. That matches our actual layout exactly.
The 3D FFT then runs along the correct axes, V_eff multiplication aligns at
correct spatial positions, the inverse FFT reconstructs G-space, and the
gather extracts PWs at correct (h, k, l) coordinates.

## Sanity check on cubic grids

For `ngx = ngy = ngz = N`, the swap is a no-op — the dims are still `(N, N,
N)`. All existing unit tests on cubic grids remain green. ✓

## Sanity check on `compute_screened_d` (chemrust-hamiltonian#8)

The chemrust-hamiltonian issue tracks a separate bug: `fft_forward_3d`
reverses axes `(a,b,c)→(c,b,a)`, breaking `compute_screened_d` on non-cubic
grids. That is a CPU FFT bug in chemrust-hamiltonian, **independent** of
this cuFFT plan/scatter mismatch. They are separate root causes; both must
be fixed for full Cu111_CO eigenvalue accuracy. This plan only fixes the
GPU side.

The diagonalize path passes `Some(&v_eff_for_d)` to `VnlBatchData::precompute`
at `src/scf.rs:439`, which would invoke `compute_screened_d`. If
`compute_screened_d` is currently broken, the V_NL term will be off.
However, the −2 Ha symptom on band 1 (T≈0) is dominantly V_loc, so this
fix is expected to bring band 1 from −3.13 Ha toward −1.06 Ha even with
broken D screening. Higher bands may remain off until chemrust-hamiltonian#8
is fixed.

## Acceptance criteria for this fix

EXTERNAL anchors (from prior `CRITERIA.md`):

- After the cuFFT plan swap, the test
  `compare_eigenvalues_with_reference_veff_and_screening` should produce
  **band 1 ε ∈ [−1.10, −1.00] Ha** (loose, allowing any residual from
  V_NL screening or from finite Chebyshev degree).
- For a true tight test, run with `ndeg=0` and `occupations=None` so the
  Chebyshev filter is a no-op and screening is disabled. Then
  `Rayleigh-Ritz on the input wavefunctions` should give band 1 at
  approximately the reference value −1.055 Ha.

## Risk

Density construction (`density.rs:147`) uses the same buggy plan call. If
density was constructed via this path during prior runs, the constructed
density would also be wrong-shape relative to the lattice. However, the
existing tests use density loaded from `.castep_bin` (parsed correctly into
shape `[ngx, ngy, ngz]` C-layout), not constructed from the GPU path. So the
GPU density bug is latent in the SCF loop but invisible to the eigenvalue
test. Both call sites must be fixed together.

Date: 2026-05-20
