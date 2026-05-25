# Design Decisions: CASTEP `.check` Discriminator

## Cross-repo architecture

The `.check` serialization belongs in `chemrust-hamiltonian-core` (it mirrors
the read path and uses the same type system). chemrust-scf depends on
hamiltonian-core via path dependency. The serialization work was filed as
[issue #10](https://github.com/TonyWu20/chemrust-hamiltonian/issues/10) and
implemented in the same session.

## Feature gate

Capture logic gated behind `scf_diag` (default-on, zero-cost when
`--no-default-features`). Matches existing pattern for `DavidsonDiagnostic`.

## Wavefunction indexing

`ColumnDistributed` is a ZST phantom type marker; data is band-major
`data[b*n_pw + g]`. Confirmed at `chebyshev.rs:1071` and production usage
at `vnl_data.rs:164-165`.

## Occupations

`compute_occupations()` is called inside `check()` but the result is consumed
for energy assembly and not stored on `ScfIteration`. The capture module
recomputes them.

## Grid dimension convention

`GVectorGrid::grid()` returns `[ngz, ngy, ngx]` (Fortran order), but
`CastepBin` fields use `[ngx, ngy, ngz]` (row-major). Every grid access
in the capture code applies this swap.

## Density on fine grid

`.check` stores density on the fine grid. The pipeline:
`rho_wave_grid → upsample_density_to_fine_grid → + rho_aug_fine → total_rho`
mirrors the existing `build_v_eff_with_energy_impl` path.

## Iter-2 capture point

Inserted at line 1811 of `scf.rs` — after divergence gate checks and before
`next` is moved into `s`. At this point `fermi_energy` and `total_energy`
are populated (set by `check()`).

## File format findings

1. `parse_orig_cell` unconditionally reads the next record as a cell tag.
   orig_cell must ALWAYS be written, even when it's the default identity.
2. Parameters section expects separate Fortran records (one for version
   string, one for `END_PARAMETERS_DUMP`), not a single multi-line record.
3. Cell tags are 256-byte space-padded Fortran character records.
4. Density columns use 1-based indices and separate Fortran records per
   (nx, ny) pair.
