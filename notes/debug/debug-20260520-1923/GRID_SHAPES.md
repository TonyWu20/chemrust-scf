# Phase A1 — Cu111_CO grid shapes

## Source

Both extracted from binary fixture files (NOT prior notes — EXTERNAL anchors).

## Wave (standard) FFT grid

From `Cu111_CO.check` parsed Fortran records (record 722 after tag `b'wave'`):

```
ngx=54  ngy=90  ngz=90
ngw (n_pw) = 60608
nbands = 160  nkpts = 1  nspins = 1
```

## Fine FFT grid

From `Cu111_CO.pot_fmt` header line 9:

```
54    90    90    ! fine FFT grid along <a,b,c>
```

Identical to the wave grid.

## Implication

`tests/fixtures/cu111_co.rs:156` builds `wave_grid = GVectorGrid::new([ngz, ngy,
ngx], ...)` = `[90, 90, 54]`. Line 164 builds `fine_grid` analogously =
`[90, 90, 54]`. They are **bit-identical**.

In `src/scf.rs:1052-1055`, `downsample_array_to_wave_grid` early-returns the
fine array as-is when `ngz==ngz_f && ngy==ngy_f && ngx==ngx_f`. **For Cu111_CO
this branch fires.** The FFT roundtrip in `downsample_array_to_wave_grid` does
NOT execute on the test path.

## Divergence surface — narrowed

Component (2) `downsample_array_to_wave_grid normalization` is **ruled out for
this fixture**. The bug must be in:

1. **`plan_batched_c2c` dimension order vs scatter-index layout** (still strongest
   suspect)
2. **`veff_multiply` kernel index alignment** (V_eff stored in `Array3<f64>` 
   uploaded as a flat slice — the storage order matters)
3. **Kinetic / V_NL** (lower priority)

## Sanity check on cell volume

From `.pot_fmt` header: a=10.225 Å, b=17.710 Å, c=18.261 Å (orthorhombic).
Volume ≈ 3306 Å³. Confirms a real-space asymmetric lattice — non-cubic. Any
layout mismatch between scatter-index ordering and cuFFT's interpretation
**will manifest** because the dims (54 vs 90) differ.

Date: 2026-05-20
