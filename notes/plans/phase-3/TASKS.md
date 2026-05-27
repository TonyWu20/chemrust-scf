# Forensic TASKS.md — phase-3 (GPU D-Matrix Screening)

**Branch base:** `diag/iterative-chebyshev-viability`
**Plan source:** `plans/phase-3/PHASE_PLAN.md`
**Decisions:** `notes/plans/phase-3/DECISIONS.md`
**ODD pattern ref:** `~/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

## Declared Fixtures

| Fixture | Path | Used by |
|---------|------|---------|
| Cu(111)+CO CASTEP bin | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.castep_bin` | A, D |
| Cu(111)+CO CASTEP check | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check` | A, D |
| Cu(111)+CO CASTEP pot_fmt | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.pot_fmt` | A, D |
| Cu(111)+CO CASTEP bands | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands` | D |
| Pseudopotentials | `/export/Potentials/` | A, D |
| CASTEP D_band_debug.dat | `CASTEP_FIXTURE_DIR/D_band_debug.dat` (must be generated) | A (secondary anchor) |

## Exploration Notes

- **Goal 3 already done**: `cpx_mul_inplace` kernel exists at `kernels.rs:152-164`
  and is used in `compute_aug_density_gpu:548-556`. SF multiplication is GPU-resident.
  No D2H/H2D roundtrip for the structure factor step. Goal removed from scope.
- **`screen_d_gpu` algorithm review**: The phase convention is internally consistent:
  stores `exp(-iG·R)` in `WaveScreeningCache`, `cpx_conj_mul` computes
  `V_eff_fft * conj(exp(-iG·R)) = V_eff_fft * exp(+iG·R)`, matching CPU path.
  The `gemv` call uses `trans=C` (conjugate-transpose), computing
  `tmp[p] = Σ_g conj(Q[p,g]) * w[g]`. All math checks out. Bug is in integration
  (V_eff_fft upload convention), not kernel logic.
- **Origin-ion immunity confirmed**: For R=0, `exp(-iG·R) = (1,0)` for all G,
  so `cpx_conj_mul` reduces to `w = a` regardless of layout. Non-origin ions
  require V_eff_fft and ion_sf to index the same G-vectors.
- **`CudaKernelSet` is already passed through most call paths**: `scf.rs`
  `diagonalize_inner` creates it at line ~435; `diagonalize_with_rr_matrices`
  creates it; test files create their own. The only call site without `kernels`
  in scope is `apply_s_for_test` in `scf.rs`.
- **CPU D-screening (reference)**: `compute_screened_d_from_fft` in
  `chemrust-hamiltonian-core/src/nlpot.rs:417-454`. Iterates G-points in
  Fortran order, computes `exp(+iG·R)` directly, sums
  `Σ_G Re(V_eff_fft(G) * exp(+iG·R) * conj(Q_nm(G))) / N`.

---

## Group A — Discriminator Test: CPU vs GPU D-Screening

**kind:** lib-tdd
**depends on:** nothing
**blocks:** B (if bug found), C (if no bug)

### TASK-A1: Export d_screening items for test access

**Goal:** G1
**Files:** `src/eigensolver/d_screening.rs`
**Depends on:** none
**Kind:** direct

**Changes:**
- Add `#[cfg(any(test, feature = "scf_diag"))]` re-export block at bottom of
  `d_screening.rs` following the `density.rs:586-600` test_api pattern.
  Export: `screen_d_gpu`, `build_wave_screening_cache`, `WaveScreeningCache`,
  `WaveQSpeciesEntry`, `WaveSfEntry`.
- These items are currently `pub(crate)` — change to `pub` within the cfg-gated
  re-export block, or add explicit `pub use` statements.

**Acceptance:** `cargo check --workspace` passes with items visible to integration tests.

---

### TASK-A2: Write discriminator test

**Goal:** G1
**Files:** `tests/d_screening_cpu_vs_gpu.rs` (create)
**Depends on:** TASK-A1
**Kind:** lib-tdd

**Success Criteria:**
- For every ion in the Cu111_CO fixture, `max|GPU_D[n,m] - CPU_D[n,m]| < 1e-12`
  where CPU_D is computed by `compute_screened_d_from_fft` (verified correct).
  (Source: CPU path is EXTERNAL anchor — independently validated against CASTEP
  nlpot.f90 algorithm in chemrust-hamiltonian tests)
- GPU D matrix is real-symmetric: `|D[n,m] - D[m,n]| < 1e-15` per ion.
  (Source: D is a real-symmetric matrix by construction — screening preserves this)
- GPU D matrix has no NaN/inf entries for any ion.
  (Source: finite-precision arithmetic should produce finite results)
- Non-origin ions have non-zero screening: at least one `|D[n,m] - D0[n,m]| > 1e-10`.
  (Source: the 2026-05-23 bug produced near-zero screening; this catches regression)
- Origin ion (R=0) diagonal matches CPU to 1e-15.
  (Source: at R=0, `exp(±iG·R) = 1` — screening is layout-independent)

**Test Interface:**
- **Test file:** `tests/d_screening_cpu_vs_gpu.rs`
- **Test module:** integration test (top-level)
- **Test function:** `fn d_screening_cpu_vs_gpu_element_by_element()`
- **Test code:**
  ```rust
  #[test]
  #[ignore = "requires GPU and CASTEP fixture data"]
  fn d_screening_cpu_vs_gpu_element_by_element() {
      // 1. GPU context + fixture load
      let ctx = Arc::new(CudaContext::new(0).expect("CUDA device 0"));
      let stream = ctx.default_stream();
      let fx = fixtures::cu111_co::fixture();
      let state = fixtures::cu111_co::build_scf_state(&fx);

      // 2. Build V_eff on CPU, downsample to wave grid
      let veff_built = state.build_v_eff_with_energy().expect("build_v_eff");
      let v_eff = veff_built.v_eff().as_ref().expect("V_eff");

      // 3. FFT V_eff → reciprocal space (CPU)
      let v_eff_fft = chemrust_hamiltonian_core::fft::fft_forward_3d(
          v_eff.as_real_grid()
      ).expect("FFT forward");
      let [ngz, ngy, ngx] = state.wave_grid.grid();
      let n_wave = ngz * ngy * ngx;

      // 4. Upload V_eff_fft to GPU in Fortran order
      let v_eff_flat: Vec<CudaComplex> = v_eff_fft.as_recip_array()
          .iter()
          .map(|c| CudaComplex { x: c.re, y: c.im })
          .collect();
      let v_eff_dev = stream.clone_htod(&v_eff_flat).expect("H2D V_eff_fft");

      // 5. Build WaveScreeningCache on GPU
      let mut pcie = PcieAccount::default();
      let kernels = CudaKernelSet::new(&ctx).expect("kernels");
      let blas = BlasHandle::new(Arc::clone(&stream)).expect("blas");
      let cache = build_wave_screening_cache(
          &state.pots, &state.cell, &state.wave_grid,
          &stream, &mut pcie,
      ).expect("build cache");

      // 6. Per-ion CPU vs GPU comparison
      for ion_idx in 0..state.cell.num_ions {
          let species_idx = state.cell.ion_species[ion_idx];
          let symbol = &state.cell.species_symbols[species_idx];
          let pot = state.pots.get(symbol).expect("pot");
          let aug: &dyn HasAugmentationData = match pot {
              Pseudopotential::Usp(d) => d,
              _ => continue,
          };

          let d0 = build_d0_expanded(aug);
          let d0_flat: Vec<f64> = d0.iter().cloned().collect();
          let n_exp = d0.shape()[0];

          // CPU reference (anchor)
          let q_on_grid = precompute_q_on_grid(aug, &state.wave_grid).expect("Q");
          let cpu_d = compute_screened_d_from_fft(
              &q_on_grid, &v_eff_fft, &state.cell, ion_idx,
              &state.wave_grid, &d0,
          );

          // GPU computation
          let gpu_d = screen_d_gpu(
              &cache, &v_eff_dev, ion_idx, species_idx,
              &d0_flat, n_wave, &kernels, &blas, &stream,
          ).expect("screen_d_gpu");

          // Structural assertions
          for n in 0..n_exp {
              for m in 0..n_exp {
                  assert!((gpu_d[[n,m]] - gpu_d[[m,n]]).abs() < 1e-15,
                      "ion={ion_idx}: D not symmetric at ({n},{m})");
                  assert!(gpu_d[[n,m]].is_finite(),
                      "ion={ion_idx}: D has non-finite at ({n},{m})");
              }
          }

          // Element-by-element comparison
          let max_delta = (0..n_exp).flat_map(|n| (0..n_exp).map(move |m| {
              (gpu_d[[n,m]] - cpu_d[[n,m]]).abs()
          })).fold(0.0_f64, f64::max);
          assert!(max_delta < 1e-12,
              "ion={ion_idx}: max|GPU-CPU| = {:.4e} > 1e-12",
              max_delta);

          // Non-origin must have non-trivial screening
          let pos = state.cell.ionic_positions.row(ion_idx);
          let is_origin = pos.iter().all(|&r| r.abs() < 1e-10);
          if !is_origin {
              let max_screening = (0..n_exp).flat_map(|n| (0..n_exp).map(move |m| {
                  (gpu_d[[n,m]] - d0[[n,m]]).abs()
              })).fold(0.0_f64, f64::max);
              assert!(max_screening > 1e-10,
                  "ion={ion_idx}: non-origin has near-zero screening ({:.4e})",
                  max_screening);
          }
      }
  }
  ```

**Diagnostic tiers** (inlined after the per-ion assertion failure, gated by
`#[cfg(debug_assertions)]` or env var `CHEMRUST_DIAGNOSE_D_SCREENING=ion_idx`):

- **T1 (struct factors)**: Download `cache.ion_sf[ion_idx].sf` via `clone_dtoh`.
  Recompute CPU struct factors in Fortran order using same loop as
  `build_wave_screening_cache:158-171`: `phase = -tau * (gx*rx + gy*ry + gz*rz)`.
  Assert `|GPU_sf[g] - CPU_sf[g]| < 1e-15` for all g.

- **T2 (Q arrays)**: Download `cache.species_entries[species_idx].q_nm`.
  Recompute CPU Q via `precompute_q_on_grid`, flatten in pair-major Fortran order
  matching builder lines 133-138. Assert `|GPU_Q - CPU_Q| < 1e-12` for sampled
  grid points.

- **T3 (w buffer)**: Modify `screen_d_gpu` to accept an optional diagnostic output
  slice for the `w` buffer. Compute CPU `w_cpu[g] = V_eff_fft(g) * exp(+iG·R)`
  by conjugating CPU struct factor. Assert `|GPU_w - CPU_w| < 1e-15`.

- **T4 (tmp buffer)**: Similarly download tmp after gemv. Compute CPU
  `tmp_cpu[p] = Σ_g conj(Q(p,g)) * w(g)`. Assert `|GPU_tmp - CPU_tmp| < 1e-12`.

**Acceptance:**
```bash
cargo test --release -- d_screening_cpu_vs_gpu --ignored --nocapture
```

---

## Group B — Fix the Bug

**kind:** direct
**depends on:** Group A (diagnostic output identifies root cause)

### TASK-B1: Apply fix based on diagnostic findings

**Goal:** G1
**Files:** `src/eigensolver/d_screening.rs` (most likely)
**Depends on:** TASK-A2 (diagnostic results)
**Kind:** direct

**Changes (by diagnostic outcome):**

1. **V_eff_fft upload order** (T1 passes, T3 fails): The V_eff_fft was uploaded
   in wrong flattening order. In `d_screening.rs` (or the wiring code), ensure
   `as_recip_array().iter()` is used for flattening — this traverses in Fortran
   (memory) order, matching struct factor layout.

2. **Struct factor computation** (T1 fails): G-vector iteration order in
   `build_wave_screening_cache:158-171` doesn't match FFT output. Check whether
   `gvecs()[[iz, iy, ix]]` returns G-vectors in the same order as FFT. Fix loop
   or indexing accordingly.

3. **gemv parameters** (T1-T3 pass, T4 fails): Check `gemv_c64` call at
   `d_screening.rs:243-261`. Verify `m = n_wave_grid`, `n = n_lower_pairs`,
   `lda = n_wave_grid`, `trans = C`. The Q matrix is stored pair-major,
   interpreted as col-major `(n_wave × n_lower_pairs)`. If `m` and `n` are
   swapped, gemv sums across wrong dimension.

4. **Pair indexing** (T1-T4 pass, final D mismatch): Check `pair_indices`
   construction in `build_wave_screening_cache:110-130`. Verify the pair-to-
   (n,m) mapping matches CPU `compute_screened_d_from_fft` which iterates the
   full `(n_expanded, n_expanded)` matrix.

**Acceptance:** TASK-A2 discriminator test passes with `|Δ| < 1e-12` for all ions.

---

## Group C — Wire GPU D-Screening into Production Path

**kind:** direct
**depends on:** Group B (or Group A passes without fix needed)

### TASK-C1: Add kernels parameter to precompute_with_d_override

**Goal:** G2
**Files:** `src/eigensolver/vnl_data.rs`
**Depends on:** Group A or B (D-screening verified correct)
**Kind:** direct

**Changes:**
- Add `kernels: &CudaKernelSet` parameter to `precompute_with_d_override`
  signature (between `blas` and `solver`). Use the fully-qualified path
  `&crate::eigensolver::chebyshev::CudaKernelSet` to avoid unresolved-type
  errors (the import isn't added until TASK-C2).
- In `precompute` wrapper (line 116): add `kernels` parameter, forward to
  `precompute_with_d_override`.

**Acceptance:** `cargo check --workspace` — expected: compile errors at all call
sites (they don't pass `kernels` yet), but declaration-site in vnl_data.rs must
compile. Proceed to TASK-C2 (which adds the shorter import).

---

### TASK-C2: Build WaveScreeningCache and wire screen_d_gpu

**Goal:** G2
**Files:** `src/eigensolver/vnl_data.rs`
**Depends on:** TASK-C1
**Kind:** direct

**Changes:**
1. Add imports at top: `use crate::eigensolver::d_screening::{build_wave_screening_cache, screen_d_gpu, WaveScreeningCache};`
   and `use crate::eigensolver::chebyshev::CudaKernelSet;`.

2. After `v_eff_fft` is computed (line 184):
   - Build `WaveScreeningCache` via `build_wave_screening_cache(pots, cell, wave_grid, stream, pcie)`.
     Store as `Option<WaveScreeningCache>`.
   - Flatten `v_eff_fft.as_recip_array()` to `Vec<CudaComplex>` via `.iter()`,
     upload to GPU via `stream.clone_htod()`, store as `Option<CudaSlice<CudaComplex>>`.
     Record H2D bytes in `pcie`.
   - Remove the `q_on_grid_cache: HashMap<String, Option<QOnGrid>>` entirely.
   - Remove `use chemrust_hamiltonian_core::nlpot::precompute_q_on_grid;` import.

3. Replace lines 239-244:
   ```rust
   // Before:
   None => match (&v_eff_fft, q_on_grid_cache.get(symbol).and_then(|o| o.as_ref())) {
       (Some(fft), Some(q_on_grid)) => {
           compute_screened_d_from_fft(q_on_grid, fft, cell, ion_idx, wave_grid, &d0_expanded)
       }
       _ => d0_expanded.clone(),
   },

   // After:
   None => match (&screening_cache, &v_eff_fft_dev) {
       (Some(cache), Some(fft_dev)) => {
           let d0_flat: Vec<f64> = d0_expanded.iter().cloned().collect();
           screen_d_gpu(
               cache, fft_dev, ion_idx, species_idx,
               &d0_flat, n_wave_grid,
               kernels, blas, stream,
           )?
       }
       _ => d0_expanded.clone(),
   },
   ```
   Where `n_wave_grid = ngz * ngy * ngx` for the wave grid.

4. Update `screening_h2d_bytes`:
   ```rust
   let screening_h2d_bytes: usize =
       // V_eff_fft H2D
       (if v_eff_fft.is_some() { n_wave_grid * std::mem::size_of::<CudaComplex>() } else { 0 })
       // Q arrays + struct factors (already tracked in build_wave_screening_cache via pcie)
       + pcie.h2d_bytes; // captured during cache build + V_eff upload
   ```

5. Remove the reverted-GPU-path comment at lines 182-183.

**Acceptance:** `cargo check --workspace` — expected: some callers still fail
(missing `kernels`), but vnl_data.rs compiles. Proceed to TASK-C3.

---

### TASK-C3: Update all callers to pass kernels

**Goal:** G2
**Files:** `src/scf.rs`, `tests/ca_scf_convergence.rs`, `tests/rayleigh_ritz_validation.rs`, `tests/chebyshev_orthogonality_diagnostic.rs`
**Depends on:** TASK-C2
**Kind:** direct

**Changes (each call site adds `&kernels` argument):**

| File | Function | Line (approx) | Has `kernels` in scope? | Action |
|------|----------|------|------------------------|--------|
| `src/scf.rs` | `diagonalize_inner` | ~579 | Yes (created ~line 435) | Add `&kernels` arg |
| `src/scf.rs` | `diagonalize_with_rr_matrices` | ~843 | Yes (created locally) | Add `&kernels` arg |
| `src/scf.rs` | `apply_h_components_for_test` | ~910 | Yes (passed in) | Add `&kernels` arg |
| `src/scf.rs` | `apply_s_for_test` | ~964 | **No** | Add `let kernels = CudaKernelSet::new(&ctx)?;` before call |
| `tests/ca_scf_convergence.rs` | `s_inv_baseline_post_typo_fix` | ~711 | Yes | Add `&kernels` arg |
| `tests/ca_scf_convergence.rs` | `s_inv_s_identity_test` | ~802 | Yes | Add `&kernels` arg |
| `tests/rayleigh_ritz_validation.rs` | ritz test | ~543 | **No** | Add `let kernels = CudaKernelSet::new(&ctx)?;` before call |
| `tests/chebyshev_orthogonality_diagnostic.rs` | diagnostic 1 | ~244 | **No** | Add `let kernels = CudaKernelSet::new(&ctx)?;` before call |
| `tests/chebyshev_orthogonality_diagnostic.rs` | diagnostic 2 | ~582 | **No** | Add `let kernels = CudaKernelSet::new(&ctx)?;` before call |
| `tests/chebyshev_orthogonality_diagnostic.rs` | diagnostic 3 | ~1103 | **No** | Add `let kernels = CudaKernelSet::new(&ctx)?;` before call |

**Acceptance:** `cargo check --workspace` passes with no errors.

---

### TASK-C4: Update PcieAccount H2D assertion

**Goal:** G2
**Files:** `src/scf.rs`
**Depends on:** TASK-C3
**Kind:** direct

**Changes:**
- In `diagonalize_inner` (near line 786-791), update the PcieAccount H2D
  assertion to include `vnl_data.screening_h2d_bytes`:
  ```rust
  let expected_h2d = psi_bytes
      + vnl_data.screening_h2d_bytes
      + fft_idx_bytes
      + kinetic_bytes
      + ...; // existing terms
  assert_eq!(pcie.h2d_bytes, expected_h2d, "PCI-E H2D budget mismatch");
  ```

**Acceptance:** `cargo test --release -- iter1_drift_from_castep_state_is_bounded --ignored --nocapture`
— the PcieAccount assertion passes (H2D bytes within budget).

---

### TASK-C5: Remove #[allow(dead_code)] attributes

**Goal:** G2
**Files:** `src/eigensolver/d_screening.rs`, `src/eigensolver/kernels.rs`
**Depends on:** TASK-C3
**Kind:** direct

**Changes:**
1. `d_screening.rs:10`: Remove `#![allow(dead_code)]` from module header.
2. `d_screening.rs`: Remove `#[allow(dead_code)]` from individual fields
   (`ion_species`, `wave_grid`) in `WaveScreeningCache` if present.
3. `kernels.rs:202-203`: Remove `#[allow(dead_code)]` from `cpx_conj_mul` field.

**Acceptance:** `cargo check --workspace` and `cargo clippy --workspace -- -D warnings`
pass with no dead_code warnings on these items.

---

## Group D — Anchor Validation

**kind:** direct
**depends on:** Group C

### TASK-D1: Run anchor tests against GPU D-screening

**Goal:** G3
**Files:** none (run only)
**Depends on:** TASK-C5
**Kind:** direct

**Changes:** None. Run existing tests.

**Acceptance (all must pass):**
```bash
# SC-3: V_eff range bounded across iterations
cargo test --release -- --ignored iter2_v_eff_range_within_one_ha_of_iter1 --nocapture

# SC-4: screened-D improves band-1 residual over bare-D0 (ca_step_validation.rs)
cargo test --release -- --ignored screened_d_band_1_residual_improves --nocapture

# SC-5: density decomposition matches CASTEP F8
cargo test --release -- --ignored density_decomp_matches_castep_f8_same_inputs --nocapture
```

**Regression handling:** If any test regresses vs CPU baseline, GPU and CPU
D matrices differ for some ion. Use Group A diagnostic infrastructure
(`CHEMRUST_DIAGNOSE_D_SCREENING=ion_idx`) to identify which ion and which
diagnostic tier fails. Fix in `d_screening.rs` and re-run.

---

## Group E — Cleanup

**kind:** direct
**depends on:** Group D

### TASK-E1: Remove stale comments and dead-code annotations

**Goal:** G2
**Files:** `src/eigensolver/d_screening.rs`, `src/eigensolver/vnl_data.rs`
**Depends on:** TASK-D1
**Kind:** direct

**Changes:**
1. `d_screening.rs:7-10`: Remove the comment block:
   ```
   // NOTE: This module is currently unused — the GPU D-screening path was reverted
   // to the CPU `compute_screened_d_from_fft` path due to a bug producing near-zero
   // screening terms for non-origin ions. Kept for future debugging.
   ```
   Replace with a brief one-liner: `// GPU D-matrix screening for USPP V_NL.`

2. `vnl_data.rs:182-183`: Remove:
   ```
   // GPU D-screening (screen_d_gpu) was reverted due to a bug producing
   // near-zero screening terms for non-origin ions. CPU path is correct.
   ```

3. Verify no other `#[allow(dead_code)]` remains on D-screening items.

**Acceptance:**
```bash
cargo check --workspace
cargo clippy --workspace -- -D warnings
```
No dead_code or unused_import warnings related to D-screening.
