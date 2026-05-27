# TASKS: Diagnostic 3 — Outer Loop Convergence Test

**Phase**: diagnostic-3-outer-loop  
**Date**: 2026-05-27  
**ODD Pattern**: `/home/tony/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

---

## Declared Fixtures

**Primary fixture**: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
- `Cu111_CO.check` — CASTEP-converged wavefunctions (S-orthonormal under USPP S), 160 bands, n_pw=60067
- `Cu111_CO.bands` — Reference eigenvalues (160 bands), band 0 = -1.055 Ha
- `Cu111_CO.den_fmt` — Converged density
- `Cu111_CO.pot_fmt` — Converged V_eff

**Starting state**: CASTEP-converged (same as Diagnostic 2)

**Baseline data** (from Diagnostic 2, commit `ed8534e`):
- Conduction bands (82-159): mean S⁻¹ residual = 0.026 Ha, max = 0.057 Ha
- Occupied bands (0-81): mean S⁻¹ residual = 0.15 Ha, max = 0.21 Ha
- Band 0: S⁻¹ residual = 4.21e-2 Ha
- Cu 3d cluster (1-14): mean S⁻¹ residual = 0.155 Ha, max = 0.209 Ha

---

## Task Groups

### Group A: Core Implementation (3 tasks)

**Dependencies**: Diagnostic 2 infrastructure (chebyshev_filter_for_test_gpu, compute_residual_norms_for_test, rayleigh_ritz_with_matrices)

---

## TASK-A1: GPU-Resident Filter Iteration Function

**Kind**: direct  
**Goal**: Add `chebyshev_filter_iteration_gpu` — a `#[doc(hidden)] pub fn` that wraps the internal `chebyshev_filter` (chebyshev.rs:542), accepting pre-built GPU state instead of building from scratch.

### Changes

**New function** in `src/eigensolver/chebyshev.rs` (after `chebyshev_filter_for_test_gpu` at line 1484):

```rust
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_filter_iteration_gpu(
    psi_gpu: &Gpu<WavefunctionSet<ColumnDistributed>>,
    v_eff_gpu: &Gpu<EffectivePotential>,
    fft_idx_dev: &CudaSlice<i32>,
    kernels: &CudaKernelSet,
    wave_grid: &GVectorGrid,
    pw_coords: &[[i32; 3]],
    cell: &CellGeometry,
    pots: &PseudopotentialSet,
    vnl_data: &VnlBatchData,
    min_veff: f64,
    max_veff: f64,
    ndeg: usize,
    eigenvalues: Option<&[f64]>,
    filter_mode: FilterMode,
    blas: &BlasHandle,
    solver: &SolverHandle,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> ChebyshevResult
```

Body: constructs dummy `KPoint { coords: [0.0, 0.0, 0.0] }`, creates internal `PcieAccount`, calls `chebyshev_filter(...)`. ~5 lines.

**Visibility change** in `src/device/pcie.rs`:
- `Gpu::from_host_with` changed from `pub(crate)` to `#[doc(hidden)] pub` — needed so integration tests can upload V_eff to GPU for pre-built state. All parameter types and return types were already public.

**Re-export** in `src/lib.rs`:
- Added `chebyshev_filter_iteration_gpu` to the `#[doc(hidden)] pub use` list.

---

## TASK-A2: Helper and Verification Functions

**Kind**: direct  
**Goal**: Add CPU-side helper functions used by the diagnostic test.

### Changes

All added to `tests/chebyshev_orthogonality_diagnostic.rs`:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `upload_psi_to_gpu_column` | `(psi_host, n_bands, n_pw, stream, pcie) -> Gpu<WavefunctionSet<ColumnDistributed>>` | Uploads CPU wavefunction data to GPU as ColumnDistributed layout |
| `compute_mean_residual` | `(norms, band_indices) -> f64` | Mean residual over band indices |
| `count_converged_bands` | `(norms, band_range, threshold) -> usize` | Count bands with residual below threshold |
| `compute_max_eigenvalue_drift` | `(eigs_a, eigs_b) -> f64` | Max |λ_i[N] - λ_i[N-1]| |
| `verify_residual_monotonicity` | `(history, groups)` | SC-1: ≤2 non-consecutive violations, 20% reduction by iter-5 or iter-10 |
| `verify_band0_stability` | `(history, lo, hi)` | SC-5: band 0 eigenvalue in [lo, hi] across all iterations |

No library-level changes needed — all verification helpers are standalone CPU-side math on `Vec<f64>`.

---

## TASK-A3: Diagnostic Test

**Kind**: direct  
**Goal**: Add `diagnostic_3_outer_loop_convergence` test that runs 10 iterations of Chebyshev filter → RR → residual check, verifying SC-1 through SC-5.

### Corrected Outer Loop Flow

Key differences from the original pseudocode:

1. **No `upload_check_wavefunctions_column`** — replaced with `upload_psi_to_gpu_column` (takes raw `&[Complex64]`, not fixture reference)
2. **No `update_psi_gpu_for_next_iteration`** — move semantics: `psi_gpu = psi_new_gpu` drops the old `Gpu` (frees its device memory) and takes ownership of the RR output
3. **V_eff upload done inline** in the test using public types (`WaveGridArray`, `FineGridArray`, `EffectivePotential`, `Gpu`)
4. **FFT indices uploaded inline** via `chemrust_scf::pw_coords_to_fft_indices` + `stream.clone_htod`
5. **Pre-built GPU state** (`v_eff_gpu`, `fft_idx_dev`, `kernels`) reused across all 10 iterations — no re-upload of V_eff, no kernel recompilation
6. **First iteration uses same flow** as subsequent iterations (no special first-iter code path)
7. **`CudaStream::new(&ctx)` replaced with `ctx.default_stream()`** — matches Diagnostic 2 pattern

### Corrected Test Structure

```rust
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_3_outer_loop_convergence() {
    // === Setup (same as Diagnostic 2) ===
    // - GPU check, fixture, dimensions, wave_grid, pw_coords, k_point
    // - GPU context, stream (=ctx.default_stream()), blas, solver
    // - VnlBatchData::precompute (14 params)
    // - min_veff, max_veff from fx.pot_fmt
    // - ndeg = 8
    
    // === One-time GPU setup ===
    // - Upload V_eff: Array3::from_shape_vec → WaveGridArray → FineGridArray → EffectivePotential → Gpu::from_host_with
    // - Upload FFT indices: pw_coords_to_fft_indices → stream.clone_htod
    // - Compile kernels: CudaKernelSet::new(&ctx)
    // - Upload initial psi: upload_psi_to_gpu_column(psi_input, n_bands, n_pw, &stream, &mut pcie)
    
    // === Outer loop (10 iterations) ===
    for iter in 0..n_outer_iters {
        // Step 1: chebyshev_filter_iteration_gpu(&psi_gpu, &v_eff_gpu, &fft_idx_dev, &kernels, ...)
        //         → (psi_row_gpu, hpsi_row_gpu)  [GPU-resident]
        // Step 2: rayleigh_ritz_with_matrices(psi_row, hpsi_row, ...)
        //         → (psi_new, Cpu(eigenvalues), ..., X)
        // Step 3: compute_residual_norms_for_test(psi_new, hpsi_row, eigenvalues, x, ...)
        //         → (sinv_norms, l2_norms)
        // Step 4: track history
        // Step 5: psi_gpu = psi_new  (move)
    }
    
    // === Verify SC-1 through SC-5 ===
}
```

### Band Group Definitions

```rust
let core: Vec<usize> = (0..1).collect();    // band 0
let cu3d: Vec<usize> = (1..15).collect();    // bands 1-14
let val: Vec<usize> = (15..82).collect();    // bands 15-81
let nfermi: Vec<usize> = (82..97).collect(); // bands 82-96
let cond: Vec<usize> = (97..160).collect();  // bands 97-159
```

Note: groups are `Vec<usize>` (not `Range<usize>`) since `compute_mean_residual` takes `&[usize]`.

### Acceptance

```bash
# Compilation check
cargo check --workspace

# Run Diagnostic 3 (requires GPU + fixture data)
cargo test --release --test chebyshev_orthogonality_diagnostic \
  diagnostic_3_outer_loop_convergence -- --ignored --nocapture

# Expected output:
# - Per-iteration table showing residual evolution (6-column: iter, core, cu3d, val, nFermi, cond, band0_eig)
# - All 5 success criteria assertions pass
# - Runtime: ~15-20 minutes (10 iterations × ~90s per iteration)
```

### Success Criteria

**SC-1: Residual Monotonicity (per-group)**
- For each band group (core={0}, cu3d={1-14}, val={15-81}, nFermi={82-96}, cond={97-159}):
  - `residual[N] ≤ residual[N-1] × 1.05` for most iterations (allow ≤2 non-consecutive violations)
  - At least one of {iter-5, iter-10} shows `residual[N] < residual[1] × 0.8`
- **Source**: PARSEC Algorithm 4 (Liou et al. 2020), Diagnostic 2 baseline
- **Verification granularity**: Per-group mean residual per iteration
- **Counter-example**: All groups plateau (no 20% reduction by iter-10) or show sustained increase (>2 violations)
- **Test fixture scope**: Cu111_CO.check (converged state)

**SC-2: Conduction Band Early Convergence**
- At least 50/63 conduction bands reach S⁻¹ residual < 0.01 Ha within 5 iterations
- **Source**: Diagnostic 2 baseline (conduction mean 0.026 Ha), Zhou 2014 convergence rate
- **Verification granularity**: Per-band residual at iter-5
- **Counter-example**: < 40 conduction bands converge within 5 iterations
- **Test fixture scope**: Cu111_CO.check (converged state)

**SC-3: Occupied Band Residual Reduction**
- Mean S⁻¹ residual for occupied bands (0-81) decreases by ≥5× from iter-1 to iter-10
- **Source**: Diagnostic 2 baseline (occupied mean ~0.15 Ha)
- **Verification granularity**: Mean residual for bands 0-81 at iter-1 and iter-10
- **Counter-example**: < 3× reduction (mean residual > 0.05 Ha at iter-10)
- **Test fixture scope**: Cu111_CO.check (converged state)

**SC-4: Eigenvalue Stability**
- Max eigenvalue drift per iteration `max_i |λ_i[N] - λ_i[N-1]| < 0.1 Ha` for N ≥ 4
- **Source**: HANDOFF.md cascade signature (band 0 drifted ~11 Ha in single-pass)
- **Verification granularity**: Per-iteration max drift across all bands
- **Counter-example**: Any iteration N ≥ 4 shows max drift > 0.15 Ha
- **Test fixture scope**: Cu111_CO.check (converged state)

**SC-5: No Cascade Signature**
- Band 0 eigenvalue stays within [-1.10, -1.01] Ha across all 10 iterations
- **Source**: HANDOFF.md (band 0 reference = -1.055 Ha from Cu111_CO.bands)
- **Verification granularity**: Band 0 eigenvalue per iteration
- **Counter-example**: Band 0 drifts outside [-1.15, -0.95] Ha at any iteration
- **Test fixture scope**: Cu111_CO.check (converged state), Cu111_CO.bands (reference)

### Files

**Test file**: `tests/chebyshev_orthogonality_diagnostic.rs`

### Changes

Add test function `diagnostic_3_outer_loop_convergence()` with the following structure:

```rust
#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn diagnostic_3_outer_loop_convergence() {
    // Setup (same as Diagnostic 2)
    let fx = fixture();
    
    // Initialize GPU context and handles
    let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
    let stream = Arc::new(CudaStream::new(&ctx).expect("Failed to create CUDA stream"));
    let blas = BlasHandle::new(&ctx).expect("Failed to create cuBLAS handle");
    let solver = SolverHandle::new(&ctx).expect("Failed to create cuSOLVER handle");
    let mut pcie = PcieAccount::new();
    
    // Build VNL data
    let vnl_data = VnlBatchData::precompute(
        &fx.pots, &fx.bin.cell, fx.bin.n_bands, fx.bin.n_pw,
        &blas, &stream, &ctx,
    ).expect("Failed to precompute VNL data");
    
    let n_bands = fx.bin.n_bands;
    let n_pw = fx.bin.n_pw;
    
    // Band group definitions
    let core_bands = 0..1;
    let cu3d_bands = 1..15;
    let val_bands = 15..82;
    let nfermi_bands = 82..97;
    let cond_bands = 97..160;
    
    // Load CASTEP reference eigenvalues from fixture
    let castep_eigenvalues = &fx.bands_eigenvalues;
    
    // Initialize psi_input_gpu from CASTEP .check file (ColumnDistributed)
    let mut psi_input_gpu = upload_check_wavefunctions_column(&fx, &ctx, &stream)
        .expect("Failed to upload initial wavefunctions");
    
    // Outer loop
    let n_outer_iters = 10;
    let mut residual_history = Vec::new();
    let mut eigenvalue_history = Vec::new();
    
    for iter in 0..n_outer_iters {
        println!("\n=== Iteration {} ===", iter + 1);
        
        // 1. Run Chebyshev filter (all bands, no locking)
        let (psi_filtered, hpsi_filtered, kernels) = 
            chebyshev_filter_for_test_gpu(
                &psi_input_gpu, &fx.pot_fmt, &fx.bin.cell, &fx.pots,
                n_bands, n_pw, FilterMode::SinvHKeepHEig,
                &blas, &solver, &stream, &ctx, &mut pcie,
            ).expect("Chebyshev filter failed");
        
        // 2. Run standard Rayleigh-Ritz
        let rr = rayleigh_ritz_with_matrices(
            &psi_filtered, &hpsi_filtered, &vnl_data, n_bands, n_pw,
            &kernels, &mut pcie, &solver, &blas, &stream, &ctx,
            None,  // no Procrustes pinning
            None,  // no pin config
        ).expect("Rayleigh-Ritz failed");
        
        let psi_new_gpu = rr.0;  // Gpu<WavefunctionSet<ColumnDistributed>>
        let Cpu(eigenvalues): Cpu<Vec<f64>> = rr.1;
        let x = chemrust_scf::device::cuda_vec_to_complex(rr.5.0);
        
        // 3. Compute per-band S⁻¹-weighted residuals
        let (sinv_norms, _l2_norms) = compute_residual_norms_for_test(
            &psi_new_gpu, &hpsi_filtered, &eigenvalues, &x,
            n_bands, n_pw, &vnl_data, &blas, &solver, &stream,
        ).expect("Residual computation failed");
        
        // 4. Track history
        residual_history.push(sinv_norms.clone());
        eigenvalue_history.push(eigenvalues.clone());
        
        // 5. Update psi_input_gpu for next iteration (device-to-device memcpy)
        update_psi_gpu_for_next_iteration(&mut psi_input_gpu, &psi_new_gpu, &stream)
            .expect("Failed to update psi_input_gpu");
        
        // Print per-iteration summary
        print_iteration_summary(iter + 1, &sinv_norms, &eigenvalues, castep_eigenvalues,
                                &core_bands, &cu3d_bands, &val_bands, &nfermi_bands, &cond_bands);
    }
    
    // Verify SC-1: Residual Monotonicity
    verify_residual_monotonicity(&residual_history, &[
        ("core", core_bands.clone()),
        ("cu3d", cu3d_bands.clone()),
        ("val", val_bands.clone()),
        ("nFermi", nfermi_bands.clone()),
        ("cond", cond_bands.clone()),
    ]);
    
    // Verify SC-2: Conduction Band Early Convergence
    let n_converged_at_iter5 = count_converged_bands(&residual_history[4], cond_bands.clone(), 0.01);
    assert!(n_converged_at_iter5 >= 50, 
        "SC-2 failed: only {}/63 conduction bands converged at iter-5 (expected ≥50)", 
        n_converged_at_iter5);
    
    // Verify SC-3: Occupied Band Residual Reduction
    let occupied_mean_iter1 = compute_mean_residual(&residual_history[0], 0..82);
    let occupied_mean_iter10 = compute_mean_residual(&residual_history[9], 0..82);
    let reduction_factor = occupied_mean_iter1 / occupied_mean_iter10;
    assert!(reduction_factor >= 5.0,
        "SC-3 failed: occupied band reduction = {:.2}× (expected ≥5×)",
        reduction_factor);
    
    // Verify SC-4: Eigenvalue Stability
    for iter in 4..10 {
        let max_drift = compute_max_eigenvalue_drift(&eigenvalue_history[iter-1], &eigenvalue_history[iter]);
        assert!(max_drift < 0.1,
            "SC-4 failed: max eigenvalue drift at iter- = {:.3} Ha (expected <0.1 Ha)",
            iter + 1, max_drift);
    }
    
    // Verify SC-5: No Cascade
    verify_band0_stability(&eigenvalue_history, -1.10, -1.01);
    
    println!("\n=== Diagnostic 3 Complete: All Success Criteria Passed ===");
}
```

**Helper functions to add**:

1. `upload_check_wavefunctions_column()` — Upload CASTEP .check wavefunctions to GPU in ColumnDistributed layout
2. `update_psi_gpu_for_next_iteration()` — Device-to-device memcpy from RR output to psi_input_gpu
   ```rust
   fn update_psi_gpu_for_next_iteration(
       psi_input: &mut Gpu<WavefunctionSet<ColumnDistributed>>,
       psi_new: &Gpu<WavefunctionSet<ColumnDistributed>>,
       stream: &Arc<CudaStream>,
   ) -> Result<(), Error>
   ```
3. `print_iteration_summary()` — Print per-iteration table (iter, core_mean, cu3d_mean, val_mean, nFermi_mean, cond_mean)
4. `verify_residual_monotonicity()` — Check SC-1 for each group
5. `count_converged_bands()` — Count bands with residual < threshold
6. `compute_mean_residual()` — Mean residual for a band range
7. `compute_max_eigenvalue_drift()` — Max |λ_i[N] - λ_i[N-1]|
8. `verify_band0_stability()` — Check SC-5 (band 0 eigenvalue stays within range across all iterations)

### Acceptance

```bash
# Run Diagnostic 3
cargo test --release --test chebyshev_orthogonality_diagnostic \
  diagnostic_3_outer_loop_convergence -- --ignored --nocapture

# Expected output:
# - Per-iteration table showing residual evolution
# - All 5 success criteria assertions pass
# - Runtime: ~15-20 minutes (10 iterations × ~90s per iteration)
```

### Exploration Notes

**What we learned during criteria validation**:
1. Original SC-1 was too loose (allowed indefinite plateau) — tightened to require 20% reduction by iter-5 or iter-10
2. SC-2 and SC-3 needed worst-case tracking in addition to mean/count — added as diagnostic output
3. SC-4 "after iteration 3" was ambiguous — clarified to mean iteration-to-iteration drift for N ≥ 4
4. SC-5 range was asymmetric — fixed to [-1.10, -1.01] Ha (±0.05 Ha around -1.055 Ha)

**API corrections from strict-code-reviewer audit (2026-05-27)**:
1. Removed non-existent `build_scf_state_for_diagnostic()` — follow Diagnostic 2 pattern with direct GPU buffer management
2. Fixed `rayleigh_ritz_with_matrices()` call — replaced `...` placeholder with actual 13 parameters
3. Fixed `compute_residual_norms_for_test()` call — corrected parameter order to match actual signature (10 parameters)
4. Removed redundant `load_castep_eigenvalues()` — use `fx.bands_eigenvalues` directly from fixture
5. Clarified `update_psi_gpu_for_next_iteration()` — device-to-device memcpy using `stream.memcpy_dtod`, no host round-trip

**Baseline from Diagnostic 2**:
- Conduction bands already near convergence (mean 0.026 Ha) — expect fast convergence
- Occupied bands need work (mean 0.15 Ha) — outer loop has clear signal
- Cu 3d cluster shows highest residuals (max 0.209 Ha) — may need Harmonic RR if standard RR plateaus

**Decision point after this diagnostic**:
- If all SC pass → proceed to Diagnostic 5 (band-locking)
- If SC-2 fails → investigate filter bounds (band 0 anomaly suggests b_low may be too tight)
- If SC-3 fails → proceed to Diagnostic 4 (Harmonic RR for Cu 3d cluster)
- If SC-4 or SC-5 fails → ZHEGVD rotation instability, proceed to Diagnostic 4 (Harmonic RR)

---

## Implementation Notes

**Reused from Diagnostic 2**:
- `chebyshev_filter_for_test_gpu()` — used as reference for setup code pattern (src/eigensolver/chebyshev.rs:1407-1484)
- `compute_residual_norms_for_test()` — already exported (src/eigensolver/chebyshev.rs:1497-1665), returns `(sinv_norms, l2_norms)`
- `rayleigh_ritz_with_matrices()` — already exported (src/eigensolver/rayleigh_ritz.rs:696-947), returns 6-tuple
- `fixture()` — already exists (tests/fixtures/cu111_co.rs:70-72), returns `&'static Cu111CoFixture`
- `fx.bands_eigenvalues` — CASTEP reference eigenvalues already loaded in fixture
- `classify_bands`, `group_stats`, `mean_abs_err` — existing helpers in test file

**New infrastructure**:
- `chebyshev_filter_iteration_gpu()` (src/eigensolver/chebyshev.rs:1486) — GPU-resident filter wrapper accepting pre-built state
- `Gpu::from_host_with` made `#[doc(hidden)] pub` (src/device/pcie.rs:45) — exposed for integration test V_eff upload
- `upload_psi_to_gpu_column()` — uploads psi host data to GPU as ColumnDistributed (test file)
- Verification helpers (`verify_residual_monotonicity`, `count_converged_bands`, `compute_mean_residual`, `compute_max_eigenvalue_drift`, `verify_band0_stability`) — all in test file

**What changed vs original pseudocode**:
- No `upload_check_wavefunctions_column(fx, ctx, stream)` — replaced with `upload_psi_to_gpu_column(psi_host, n_bands, n_pw, stream, pcie)` that takes raw slice
- No `update_psi_gpu_for_next_iteration()` — move semantics: `psi_gpu = psi_new_gpu`
- No `PcieAccount::new()` — uses `PcieAccount::default()` (matches Diagnostic 2)
- No `CudaStream::new(&ctx)` — uses `ctx.default_stream()` (matches Diagnostic 2)
- V_eff upload/FFT index/kernel compilation done inline in test using library types
- `chebyshev_filter_iteration_gpu()` called directly with pre-built state (17 params)
- Groups are `Vec<usize>` not `Range<usize>`
- `print_iteration_summary` inlined in loop body instead of separate function

**Memory considerations**:
- 10 iterations × 160 bands × 2 Vec<f64> (residuals + eigenvalues) = ~25 KB total history
- No subspace accumulation (reuse GPU buffers across iterations)
- Peak VRAM same as Diagnostic 2 (~4 GB)

**Performance**:
- Expected runtime: 10 iterations × 90s = ~15 minutes
- Dominated by Chebyshev filter (FFT + H·ψ + S·ψ) and ZHEGVD (160×160 generalized eigenproblem)
- No optimization needed for diagnostic (one-time test)
