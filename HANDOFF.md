# Handoff — 2026-06-14: Eigensolver verified; pivot to standalone GPU SCF

**Branch**: `feat/phase-7`
**Status**: Eigensolver ✅ | Standalone SCF 🔬 | FFI GPU SCF bottleneck identified

---

## What changed this session

### Eigensolver is now verified faithful to CASTEP

The FFI path (`ffi.rs` → `davidson.rs` → `hamiltonian.rs`) converges for both NiO non-spin and Cu111_CO spin-polarised. Two optimizations were added:

1. **Profiling instrumentation** (`hamiltonian.rs`, feature-gated behind `scf_diag`): per-operation GPU timings for `apply_full_hamiltonian` (init_kinetic, zero_buffer, scatter, cuFFT IFFT, veff_multiply, cuFFT FFT, gather_add, V_NL).

2. **BetaPhiCache** (`beta_phi_cache.rs`): caches β^H·ψ projections across Davidson outer iterations (matches CASTEP's `have_beta_phi` pattern). Populated by `compute_all()` at end of each outer iteration; consumed via compute-engine `copy_buffer` kernel (NOT `cudaMemcpyAsync` — Pascal GPUs have a copy-engine/compute-engine coherence gap). Reduces V_NL from 246ms to 110ms per cache-hit call.

### GPU eigensolver profiling — INITIAL ANALYSIS REFUTED by Fortran profiler (2026-06-14)

**The initial bottleneck analysis below was WRONG.** See `.profile` correction below.

~~Per SCF iteration for Cu111_CO (n_pw=60067, n_bands=174, grid=54×90×90):~~

~~| Component | Time per call | Calls/iter | Total/iter |~~
~~|-----------|--------------|------------|------------|~~
~~| H·ψ (full-band, n_bands=174) | 337–350ms | 4 | 1.4s |~~
~~| H·psi (search-direction, n_bands=26) | 126ms | ~100 | 12.6s |~~
~~| H·psi (other) | 80–100ms | ~34 | 3.0s |~~
~~| **All H·psi** | | **~138** | **~17s** |~~
~~| **Fortran SCF cycle** (density, mixing, V_eff, D-screening) | | | **~223s** |~~

~~H·psi is only **7% of SCF iteration time**. Every optimization targeting H·psi (Phase 1 G-vector split, Phase 2 V_NL overlap, Phase 3 grid decomposition, BetaPhiCache) targets a minority component. The 223 seconds in CASTEP Fortran are the real bottleneck.~~

### Correction: CASTEP Fortran profiler reveals the eigensolver IS the bottleneck

The CASTEP Fortran profiler (`NiO.0001.profile`, FFI-converged non-spin run, 10 iters, 201s total) traces every Fortran subroutine. The Rust GPU eigensolver is called via FFI — it is **not a Fortran subroutine**, so the profiler cannot attribute its time to any child function. The time appears as an "untraced gap" under `electronic_minimisation`.

Evidence from `/export/public_castep_jobs/tony/NiO_no_u_finer_grid_no_spin/NiO.0001.profile`:

```
electronic_minimisation                         214.43s  (93% untraced)
├── electronic_initialise                         5.06s
├── electronic_prepare_H        (V_eff assembly)  2.55s
│   ├── locpot_calculate                          1.35s
│   └── nlpot_calculate_d                        1.20s
├── wave_beta_phi_wv_ks        (β^H·ψ projections) 2.67s
├── electronic_apply_H_energy_eigen               2.75s
├── density_calculate_soft_wvfn                   0.40s
├── density_augment                               0.48s
├── electronic_dump                               0.37s
├── dm_mix_density              (Pulay/Kerker)    0.01s
├── all other Fortran children                   ~0.1s
└── ═══════ UNTRACED (Rust GPU eigensolver) ═══ ~200.0s (93%)
```

**Total identifiable Fortran SCF time: ~14s of 214s = 6.5%.** The Rust GPU eigensolver is **93%** of SCF time, not 7%. The previous "223s in Fortran" was a residual (total − measured H·psi kernel time) that incorrectly attributed Rust Davidson overhead, CPU↔GPU transfers, and FFI boundary costs to CASTEP Fortran.

**NiO FFI run**: 10 SCF iterations, Final energy = −7160.229816426 eV, Total time = 201.06s, compiled Sun Jun 14 10:46:16 2026. This is a real, completed FFI-converged run with machine-readable profiling.

### CPU serial vs GPU — the eigensolver dominates BOTH (2026-06-15)

The CPU serial run for Cu111_CO spin-polarised (`Cu111_CO_Single_Point_0614_spin_cpu_serial/`) completed 53 SCF iterations in 3516s (Final energy = −24111.28142512 eV). The `.profile` provides a direct comparison with the GPU FFI NiO run:

| Component | CPU Cu111_CO (spin, 53 iters) | GPU FFI NiO (non-spin, 10 iters) |
|-----------|-------------------------------|----------------------------------|
| **Eigensolver** | **2959s (86.6%)** | **~200s (93%)** |
| V_eff (D-screening + Hartree + XC) | 198s (5.8%) | 2.6s (1.2%) |
| Density (augment + soft) | 167s (4.9%) | 0.9s (0.4%) |
| Mixing (Pulay/Kerker) | 3s (0.1%) | 0.01s (<0.1%) |
| Other | 91s (2.7%) | ~11s (5.4%) |
| **Total SCF** | **3418s** | **214s** |

**Key finding**: The eigensolver dominates SCF time on **both** CPU (86.6%) and GPU (93%). The previous claim that "Fortran SCF is the bottleneck" was wrong for both platforms. The Fortran SCF components (V_eff, density, mixing) are minority costs everywhere.

### CPU eigensolver internal breakdown (from `hamiltonian_diagonalise_ks`, 2959s total)

| Phase | CPU time | % of eigensolver | GPU equivalent |
|-------|----------|-------------------|----------------|
| V_NL projector (ion_beta_add_multi_recip_all) | 820s | 27.7% | cuBLAS ZGEMM (β^H·ψ) |
| Rotation (wave_rotate_slice) | 625s | 21.1% | ZGEMM (ψ_new = ψ_old · X) |
| H·search (hamiltonian_apply_slice) | 584s | 19.7% | apply_full_hamiltonian (search dirs) |
| H·ψ full-band (hamiltonian_apply_ks) | 553s | 18.7% | apply_full_hamiltonian (all bands) |
| Preconditioner (nlpot_apply_precon_ES_slice) | 420s | 14.2% | TPA + USPP preconditioner |
| S-orthogonalization (wave_Sorthogonalise_wv_slice) | 393s | 13.3% | cuBLAS-based Gram-Schmidt |
| Subspace diagonalization (wave_diagonalise_H_ks) | 137s | 4.6% | ZHEEVD |
| Other (copy, dot, init, dealloc) | ~155s | 5.2% | cublasZcopy, D2H, etc. |

**Inside V_loc** (pot_nongamma_apply_slice, 502s): IFFT 262s (52%), FFT 213s (42%), other 27s (6%).

**Inside V_eff** (electronic_prepare_H, 198s): D-screening (nlpot_calculate_d) 149s (75%), Hartree+XC (locpot_calculate) 49s (25%).

### GVEC_PARALLELISM_PROPOSAL fully reviewed and rejected

See `notes/plans/GVEC_PARALLELISM_REVIEW.md` — a three-agent review (not nine; the MEMORY.md entry was incorrect) rejected the proposal. The review correctly identified Error 1 (cuFFT cost doesn't scale with G-vector sparsity) as genuinely fatal, but Errors 2 and 3 are mitigatable. The review's own quantitative analysis was based on the same incorrect bottleneck assumption as the proposal.

**Post-review correction (2026-06-14)**: The `.profile` data proves the review AND the HANDOFF both misidentified the bottleneck. H·psi + full Davidson overhead is **93%** of SCF time, not 7%. The FFT fraction (~19.5% of H·psi) is correct, but the conclusion that "H·psi optimizations are marginal" was wrong — the eigensolver IS the bottleneck. Phase 2 overlap and grid decomposition should be re-evaluated.

---

## Current state of the standalone GPU SCF path

The standalone path (`scf.rs`) already implements the full SCF cycle in Rust/GPU:

| Phase | Implementation | Location |
|-------|---------------|----------|
| V_eff assembly | chemrust-hamiltonian (CPU): Poisson, PBE XC, V_loc | `build_v_eff_with_energy()` |
| Eigensolver | Davidson (GPU) or Chebyshev (bitrotted) | `diagonalize()` |
| Density construction | GPU: scatter + IFFT + accumulate | `construct_density_gpu()` in `density.rs` |
| Augmentation density | GPU: gemv + structure factor | `compute_aug_density_gpu()` |
| Density mixing | Kerker (GPU FFT) and Pulay DIIS (GPU + CPU 7×7 solve) | `mixing.rs` |
| Energy calculation | Ewald (CPU), total energy assembly | `energy.rs` |

All CPU-level unit tests pass (mixing, energy, convergence logic). The standalone path was abandoned because SCF divergence couldn't be attributed — was it the eigensolver, density, or mixing? Now that the eigensolver is verified correct (FFI path converges), we can debug the standalone path with confidence.

The integration tests (`tests/ca_scf_convergence.rs`) compile fine under default features (22 test binaries pass `cargo test --no-run`). The 48 compilation errors only manifest under `--features scf_diag` (API drift in `KPoint.weight`, `FilterMode`, etc.) or `--features chebyshev` (17 errors). Fix the feature-gated bitrot rather than discarding the test file.

---

## Next steps (revised 2026-06-14 based on `.profile` evidence)

### 1. Run standalone NonSpin SCF with verified eigensolver [IMMEDIATE]

Construct `ScfIteration<NonSpin>` from Cu111_CO (or NiO) checkpoint, call `run_scf_with_energy_gated`. One GPU run answers: does the pipeline converge? Uses existing `ca_scf_convergence.rs` infrastructure (which compiles fine under default features — the "48 compilation errors" were `scf_diag`-specific). Do NOT write a 20 mHa smoke test — use existing NiO discriminator tolerances (1e-4 Ha eigenvalues, 1e-6 Ha total energy).

### 2. Extend scf_diag profiling to cover full Davidson iteration [HIGH PRIORITY]

The `.profile` revealed that `apply_full_hamiltonian` kernel time (~17s) is only a fraction of the ~200s eigensolver time. The ~183s gap is in: ZHEEVD subspace diagonalization, cuBLAS ZGEMM H_sub construction, S-orthogonalization, preconditioner application, BetaPhiCache `compute_all`, and cublasZcopy data movement. Add per-phase GPU timers to `davidson.rs` (under `scf_diag` feature gate) to decompose the 200s.

### 3. Fix FFI D-screening per-spin mismatch [HIGH PRIORITY]

In `ffi.rs`, CASTEP calls `nlpot_calculate_d` once per SCF (shared across spins), but Rust calls `rescreen_d` per-spin. This is the only identified unfixed bug in the FFI path post-pipeline-unification. Fix before adding infrastructure.

### 4. Complete Phase 7 validation [MEDIUM PRIORITY]

Run the NiO discriminator test (`nio_spin_scf.rs`) on GPU. Phase 7 has 20+ commits of nearly-complete spin-polarised SCF work. Add a NonSpin regression gate: `run_scf<NonSpin>` on Cu111_CO before and after Phase 7 refactor, asserting identical total energy and eigenvalues.

### 5. Re-evaluate H·psi optimizations [AFTER STANDALONE PROFILE]

Phase 2 V_NL/FFT overlap and grid decomposition were shelved based on the wrong bottleneck analysis. After P1 and P2 establish the real standalone SCF profile, re-evaluate:
- Phase 2 overlap: V_NL callbacks are ~14s within eigensolver; async overlap with compute could save 30-40% of that
- H·psi kernel optimization: scf_diag profiling already shows V_NL dominates at 246ms vs 38ms per FFT — target V_NL first

### 6. Profile Rust standalone SCF path [BEFORE OPTIMIZATION]

Add `scf_diag`-style timing instrumentation to `scf.rs` measuring: V_eff assembly, diagonalization, density construction, mixing, energy check. Do NOT profile CASTEP Fortran — the standalone path replaces all of that code.

### 7. Repair scf_diag feature-gated tests [LOW PRIORITY]

Default-feature integration tests compile fine — preserve them. Fix `scf_diag`-specific bitrot (KPoint.weight, FilterMode, check_s_inv_s_identity) rather than discarding the test file.

---

## Key files

| File | What |
|------|------|
| `src/eigensolver/hamiltonian.rs` | H·psi application (V_loc via FFT, V_NL via cuBLAS), profiling, BetaPhiCache |
| `src/eigensolver/davidson.rs` | Davidson eigensolver with BetaPhiCache lifecycle |
| `src/eigensolver/beta_phi_cache.rs` | β^H·ψ cache (compute_all, invalidate_all, invalidate_bands) |
| `src/eigensolver/kernels.rs` | CUDA kernels including `copy_buffer` (compute-engine copy) |
| `src/ffi.rs` | FFI boundary: CASTEP Fortran ↔ Rust GPU eigensolver |
| `src/scf.rs` | Standalone GPU SCF cycle (full pipeline) |
| `src/density.rs` | Density construction (soft + aug, GPU) |
| `src/mixing.rs` | Kerker and Pulay mixing (GPU) |
| `src/energy.rs` | Ewald and total energy |
| `notes/plans/GVEC_PARALLELISM_REVIEW.md` | Formal review of GVEC_PARALLELISM_PROPOSAL |
| `notes/failure-patterns.md` | Catalog of resolved bugs and patterns |
| `tests/ca_scf_convergence.rs` | Integration tests (compile under default features; `scf_diag`/`chebyshev` features have API drift) |
| `/export/public_castep_jobs/tony/NiO_no_u_finer_grid_no_spin/` | FFI-converged non-spin run with `.profile` timing data |

## Key decisions (revised 2026-06-14)

1. **Eigensolver IS the bottleneck** — CASTEP Fortran profiler (`.profile`) proves the Rust GPU eigensolver is 93% of SCF time; CASTEP Fortran V_eff/density/mixing is ~7%. The previous "Fortran is the bottleneck" conclusion was based on an untraced residual.
2. **H·psi optimizations are NOT shelved** — Phase 2 overlap and grid decomposition target the dominant cost. Re-evaluate after standalone SCF profiling.
3. **BetaPhiCache** remains a correct optimization (55% V_NL reduction per cache-hit call).
4. **Standalone GPU SCF path** is still the correct architectural direction, but for eliminating FFI overhead, not because Fortran is slow.
5. **scf_diag profiling must extend beyond `apply_full_hamiltonian`** to cover ZHEEVD, ZGEMM, preconditioner, and data movement in the Davidson inner loop.
6. **The NiO FFI run** (`NiO_no_u_finer_grid_no_spin/`) is the first verifiable completed FFI-converged run with machine-readable profiling data. It should be the reference fixture going forward.
