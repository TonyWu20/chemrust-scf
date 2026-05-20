# Resolution: eigensolver eigenvalue accuracy

**Symptom**: Eigensolver produces wrong eigenvalues; previous session suspected spectral bound handling.

**Root cause**: Multiple independent issues found. Spectral bounds were NOT the cause — wrong bounds affect convergence rate but not eigenvalue accuracy. Actual issues:

1. **Pseudopotentials loaded as Recpot instead of USP** — `PseudopotentialSet::from_dir` picks first alphabetical match. `Cu_00_OP.recpot` sorts before `Cu_00.usp` → all 18 ions loaded as Recpot (no V_NL, no augmentation). Fixed in test fixture by hardcoding `{species}_00.usp` paths.

2. **`compute_screened_d` panics on non-cubic grids** — `fft_forward_3d` reverses axes `(a,b,c)→(c,b,a)`. `compute_screened_d` assumes FFT output shape matches input. All unit tests use cubic `[4,4,4]` grids where reversal is invisible. Panics for `[54,90,90]`. Filed as chemrust-hamiltonian#8.

3. **V_loc + T discrepancy of −2 Ha** — even with correct V_eff (.pot_fmt) and matching CPU V_NL, GPU eigenvalues are ~2 Ha too negative vs CASTEP reference. Band 1 (T≈0): GPU V_loc = −3.13 Ha, ref V_loc ≈ −1.06 Ha. The GPU FFT roundtrip that applies V_loc|ψ> likely has a normalization or indexing bug.

4. **S-overlap is ~I** — `compute_s_overlap_matrix` shows S deviates from identity by < 0.9% for this system. S⁻¹ in Chebyshev recurrence (required by PAW paper for USPP) and S-augmentation in Rayleigh-Ritz are negligible for Cu111_CO.

**Fix locations**:
- `tests/fixtures/cu111_co.rs:87` — hardcoded USP file paths
- `src/scf.rs:1053` — early return in `downsample_array_to_wave_grid` when fine_grid == wave_grid
- `src/eigensolver/vnl_data.rs` — `build_q_expanded`, Q matrix upload, S-aug prep (inactive — needs chemrust-hamiltonian#8)
- `src/eigensolver/rayleigh_ritz.rs` — USPP S-augmentation in S_sub (inactive — S≈I)
- `src/eigensolver/chebyshev.rs` — `apply_s_inverse` via Woodbury (inactive — needs chemrust-hamiltonian#8)

**Remaining work**:
1. Fix GPU FFT roundtrip causing −2 Ha V_loc discrepancy (highest priority)
2. Fix chemrust-hamiltonian#8 to enable D screening via `compute_screened_d`
3. After #2, enable S⁻¹ in Chebyshev recurrence for USPP systems (PAW paper requirement)

**Date**: 2026-05-20
