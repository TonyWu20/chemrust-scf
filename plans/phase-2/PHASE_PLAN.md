# Phase 2: Fill the SCF transitions (GPU-direct)

**Date:** 2026-05-19
**Status:** Draft
**Input:** phase-1 delivered the type-state SCF backbone (grid types, device/layout wrappers, 6 phase markers, 5 transition stubs, `run_scf` loop, compile-check test). All transition bodies are `todo!()`. No physics code. Phase 2 fills the holes against real GPU libraries.

## Context

Phase 1 proved the state machine compiles. Phase 2 makes it compute. The five transitions (`build_v_eff`, `diagonalize`, `construct_density`, `mix`, `check`) need real implementations against GPU libraries — cudarc (cuFFT + cuBLAS + cuSOLVER) — so the SCF loop produces correct physics output validated against CASTEP Cu111_CO reference fixtures.

Why GPU-direct (not CPU-first as PROJECT_ROOT_PLAN suggested): the user's motivation is preventing GPU correctness bugs. chemrust-hamiltonian already validates the H|ψ> pipeline on CPU against CASTEP. Writing CPU transition bodies then rewriting for GPU doubles the work. The hot path (eigensolver, density construction) goes directly against cuFFT/cuBLAS/cuSOLVER. V_eff assembly uses chemrust-hamiltonian's VEffBuilder (CPU) with H2D/D2H for the XC step (libxc is CPU-only regardless).

Risk-weighted ordering: diagonalize first because it carries ALL the project risk — Chebyshev filtering composing H_loc FFT + V_NL β-projector gemm has never been tested in this architecture. build_v_eff is 30 lines of glue around validated code and proves nothing new.

## Goals

### Goal 1: Lock type-state safety + deferred Phase 1 items (Small)

Change all `ScfIteration` fields from `pub` to `pub(crate)`. Add `pub(crate)` accessor methods for the integration test. Add `Add`, `Sub`, `Mul<f64>` operator impls on `Density`/`WaveGridArray`/`FineGridArray` (needed by mix, check, and Pulay metric tensor). Wire `DensityHistory::mix()` return values so both `(mixed_density, input_snapshot)` are consumed and assigned to `self.density` and `self.previous_density`.

**Why:** The type-state safety is the project's entire reason for existing. It is currently broken — `pub` fields let any code bypass transitions. Fixing this is the foundation every line of Phase 2 depends on.

**Dependencies:** None. Pure refactoring on existing code.

**Files:** `src/scf.rs`, `src/types.rs`, `src/layout.rs`, `src/mixing.rs`, `tests/backbone_compiles.rs`

### Goal 2: GPU infrastructure layer (Medium)

Integrate cudarc and set up the GPU compute primitives.

**Prerequisite: flake.nix CUDA configuration.** The current `flake.nix` has no CUDA support — only Rust via fenix. cudarc links against `libcudart`, `libcublas`, `libcusolver`, `libcufft`, and needs `nvcc` in PATH. Port the CUDA overlay pattern from `~/programming/CASTEP-GPU-port/flake.nix`:
- `cudaSupport = true`, `cudaCapability = ["6.1"]`, `cudaVersion = "12.9"` (matches system GPU: Pascal cc 6.1)
- `cudaOverlay` with `enableParallelBuilding = true` on cuda_nvcc, cuda_cudart, cuda_cccl, libcublas, libcusolver (plus cuDNN override at v9.11.1.4 — required for cc 6.1 toolchain compatibility)
- `pkgsCuda.cudaPackages_12_9` symlinkJoin for the devShell (cuFFT + cuBLAS + cuSOLVER + cuRAND + NVRTC)
- CUDA env vars: `CUDA_HOME`, `CUDA_INCLUDE`, `CUDA_LIB` pointing at the symlinkJoin paths
- Keep fenix for Rust toolchain (not in CASTEP flake)

**Rust dependency:**
- Add `cudarc` dependency (with `cuda-12050` feature or appropriate CUDA version)
- GPU FFT wrapper: `DeviceFftPlan` wrapping cuFFT 3D plans (C2C, C2R, R2C). Exposes `forward_3d`, `inverse_3d`, and batched 3D transforms (via `cufftPlanMany` for density construction)
- GPU BLAS wrapper: thin safe wrappers around cuBLAS `gemm`, `gemv`, `axpy`, `dot` via cudarc sys
- cuSOLVER `ZHEGVD` wrapper: safe Rust function calling `cusolverDnZhegvd` via cudarc sys, following the workspace-query pattern from PROJECT_ROOT_PLAN (SS1246-1285)
- Upgrade `Gpu<T>`/`Cpu<T>`: Phase 1 uses transparent `Deref<Target=T>` (CPU identity). Phase 2 replaces with real sync guards backed by `CudaStream` and `DeviceBuffer<T>`. `Gpu<T>` holds a `DeviceBuffer`; `sync_to_host()` does D2H and returns `Cpu<T>`; `sync_to_device()` does H2D. `Gpu<T>` no longer `Deref` to `T` — explicit access only through stream-aware operations.
- `Gpu<WavefunctionSet<L>>`: specialized GPU wavefunction type with stream-aware layout transpose stubs (single-rank for Phase 2, real MPI later)

**Why:** Every transition goal depends on GPU compute primitives existing.

**Dependencies:** cudarc crate available on the system (CUDA toolkit must be installed).

**Files:** `flake.nix`, new `src/device/` module (`mod.rs`, `fft.rs`, `blas.rs`, `solver.rs`), `Cargo.toml` (cudarc dep), modifications to `src/layout.rs` for `Gpu<T>`/`Cpu<T>` upgrade

### Goal 3: Implement diagonalize (Large — critical path)

Implement `ScfIteration::diagonalize()` end-to-end on GPU:

**Chebyshev filtering** (inner loop, `ndeg` iterations):
1. Compute filter bounds: estimate eigenvalue range from H|ψ> application (power iteration on a random vector or use the kinetic energy estimate)
2. For each degree k = 1..ndeg:
   - `H_loc|ψ>`: G-space kinetic + V_eff multiplication, FFT roundtrip via cuFFT (requires V_eff on GPU fine grid, upsampled from wave grid data)
   - `V_NL|ψ>`: pre-computed β-projectors (from PseudopotentialSet), cuBLAS gemm for β·ψ, then scatter β-weighted result via cuBLAS gemm
   - Chebyshev recurrence: ψ_{k+1} = 2 * Ĥ_norm * H|ψ_k> - ψ_{k-1} (cuBLAS axpy/scal on GPU arrays)
   - Check norm stability between iterations (guard against exponential blowup — CASTEP Bug 4)
3. Transpose: ColumnDistributed → RowDistributed (single-rank: permute data layout in-place. No MPI yet.)

**Rayleigh-Ritz** (on RowDistributed ψ):
1. Build subspace matrices H_sub = ψ† H ψ, S_sub = ψ† S ψ (cuBLAS gemm, n_bands × n_bands)
2. Solve H_sub · X = ε · S_sub · X via cuSOLVER ZHEGVD (GPU-resident, zero PCIe)
3. Rotate ψ ← X · ψ (cuBLAS gemm)
4. D2H eigenvalues (n_bands × 8 bytes — tiny)

**New code:** ~800 lines for Chebyshev filtering, ~200 lines for Rayleigh-Ritz

**Why:** Highest risk, highest complexity. The only transition that exercises the full GPU Hamiltonian pipeline. Getting this right proves the architecture works for the hard case.

**Dependencies:** Goal 1 (pub(crate) safety), Goal 2 (GPU infrastructure).

To test diagonalize before Goal 4 (build_v_eff): load V_eff from `.pot_fmt` fixture + pseudopotentials from USP files (`PseudopotentialSet`) for V_NL. The full H = T + V_eff[ρ] + V_NL is applied via chemrust-hamiltonian's `apply_full_hamiltonian` (hamiltonian.rs) which composes `apply_local_hamiltonian` (T + V_loc FFT roundtrip) + `apply_nlpot` (V_NL via β-projector gemm). Goal 4 is NOT a prerequisite.

**Files:** new `src/eigensolver/` module (`mod.rs`, `chebyshev.rs`, `rayleigh_ritz.rs`), modify `src/scf.rs` for the `diagonalize` body

### Goal 4: Implement construct_density + build_v_eff (Medium)

**construct_density:**
1. Transpose ψ: RowDistributed → ColumnDistributed (single-rank layout permute)
2. Batched cuFFT `cufftPlanMany`: N_bands simultaneous 3D C2R IFFTs (wave grid)
3. |ψ(r)|² computation (element-wise on GPU, or fused kernel)
4. Occupation-weighted accumulation: Σ_i f(ε_i) · |ψ_i(r)|² → ρ_new (GPU axpy loop or fused kernel)
5. symmetrize ρ (if crystal symmetry available — skip for Phase 2)

**build_v_eff:**
1. D2H density from GPU to CPU (VEffBuilder uses rustfft — CPU)
2. Upsample density: wave grid → fine grid (via chemrust-hamiltonian `upsample_density_to_fine_grid`, CPU rustfft)
3. Call `VEffBuilder::assemble_on_fine_grid()` — assembles V_H (Poisson) + V_xc (native Rust PBE) + V_ion (Ewald, CPU). All pure Rust, no libxc.
4. H2D V_eff to GPU fine grid
5. Store as `v_eff: Some(Gpu<EffectivePotential>)`

**New code:** ~300 lines for density construction, ~40 lines for V_eff assembly wrapper

**Why:** These complete the forward pass (ρ → V_eff → ψ → ρ_new). Together with diagonalize, a single SCF iteration can be validated step-by-step against CASTEP.

**Dependencies:** Goal 3 (diagonalize) for real ψ and eigenvalues. Goal 2 (GPU FFT for batched IFFT).

**Files:** new `src/scf/density.rs` for density construction, modify `src/scf.rs` for both transition bodies

### Goal 5: Implement mix + check (Medium)

**mix (Pulay mixing):**
1. Compute residual: R = ρ_new - ρ_old (GPU vector subtraction)
2. Store ρ and R in DensityHistory (ring buffer or sliding window of `history_size` entries)
3. Build Pulay metric tensor M_ij = <R_i | R_j> (GPU inner product reductions, produce history_size × history_size matrix)
4. D2H metric tensor (history_size² doubles, e.g., 64 doubles for history=8)
5. Solve M · c = α on CPU (constrained linear system, ~25 lines)
6. H2D coefficients c_i
7. ρ_mix = Σ c_i ρ_i (GPU AXPY combination) + α Σ c_i R_i

**check:**
1. GPU reduction: max|ρ_mix - ρ_old| → single f64
2. D2H the scalar
3. Return `Converged` or `NotConverged(density=ρ_mix, psi=ψ, v_eff=None, ...)`

**Why:** These close the SCF loop. Pulay (not linear mixing) is essential because Cu111_CO (a metal) won't converge with linear mixing.

**Dependencies:** Goal 4 (construct_density) for real densities to mix. Goal 1 (arithmetic ops on field types).

**Files:** modify `src/mixing.rs` for Pulay algorithm, modify `src/scf.rs` for both transition bodies

### Goal 6: SCF convergence + CASTEP validation (Medium)

**Integration test** that validates the full SCF cycle against Cu111_CO CASTEP reference:

1. **Fixture loading**: Parse `.cell`, `.param`, `.usp`, `.castep_bin` from `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/` to construct initial state. chemrust-hamiltonian already has CASTEP binary I/O (`formatted/` and `castep_bin/` modules) — reuse where possible.
2. **Per-step validation** (single SCF iteration):
   - After `build_v_eff`: compare V_eff against `.pot_fmt` (read with chemrust-hamiltonian's CASTEP formatted I/O)
   - After `diagonalize`: compare eigenvalues against `.bands` or `.castep` reference
   - After `construct_density`: compare ρ_new against `.den_fmt`
3. **Full SCF validation**: Run to convergence (or max iterations), compare:
   - Total energy against CASTEP reference: **-24110.96665069 eV** (18 ions: Cu(111) slab + CO adsorbate). Tolerance: 2e-4 eV total (~1e-5 eV/atom with guard factor for GPU FP associativity)
   - Final eigenvalues within 1e-4 eV of CASTEP `.bands` reference values
   - NOTE: CASTEP's own SCF convergence tolerance is 1e-5 eV/atom. We allow 2× guard for GPU FP differences.

**Why:** Without external validation, the code produces numbers of unknown correctness. The project's value is "correct DFT with type safety" — correctness must be proven.

**Dependencies:** All previous goals (full working SCF loop). chemrust-hamiltonian CASTEP I/O modules.

**Files:** new `tests/cu111_co_validation.rs` (or `tests/fixtures/` with helper modules for fixture loading)

## Scope Boundaries

**In scope:**
- `pub` → `pub(crate)` on all ScfIteration fields
- Arithmetic ops (Add/Sub/Mul) on grid types and domain fields
- GPU FFT (cuFFT via cudarc), cuBLAS (gemm/axpy/dot), cuSOLVER (ZHEGVD)
- `Gpu<T>` → `Cpu<T>` real sync with DeviceBuffer + CudaStream
- Chebyshev filtering (ndeg iterations, norm guards)
- Rayleigh-Ritz via cuSOLVER ZHEGVD
- Batched density construction via cuFFT `cufftPlanMany`
- V_eff assembly wrapping chemrust-hamiltonian VEffBuilder (CPU, with H2D/D2H)
- Pulay mixing with DensityHistory ring buffer
- GPU-reduction convergence check
- run_scf loop executing all real transitions
- CASTEP fixture loading and validation for Cu111_CO
- `ScfIteration<SpinPolicy, State>` stays generic (user choice)

**Out of scope:**
- MPI support (single-rank only — layout transpose is in-place permute)
- GPU kernel writing (no custom CUDA kernels, no cuda-oxide)
- Forces, stress, geometry optimization
- SpinCollinear (type param exists, only NonSpin path validated)
- k-points (Gamma-point only)
- Variable-cell relaxation
- Convergence acceleration beyond Pulay
- Python bindings, C ABI cdylib
- AMD GPU support (Vulkan/gpufft backend — CUDA only for Phase 2)
- Performance tuning (correctness-first)
- Upstream modifications to chemrust-hamiltonian (use as-is via its public API)

## Design Notes

### What "GPU-direct" means in Phase 2

Not zero PCIe. The realistic residency model:

| Data | Residency | Transfer per SCF iter |
|------|-----------|----------------------|
| ψ during Chebyshev filtering | GPU | 0 |
| ψ during Rayleigh-Ritz | GPU | 0 |
| ρ during SCF loop | GPU | D2H once for VEffBuilder (CPU rustfft) |
| V_eff construction | CPU (chemrust-hamiltonian VEffBuilder) | H2D: ρ. D2H: V_eff to GPU for diagonalize |
| V_eff during diagonalize | GPU | 0 (already on GPU from build_v_eff H2D) |
| β-projectors (V_NL) | GPU (pre-computed, stays) | 0 after initialization |
| Pulay metric tensor | GPU → CPU (history² doubles) | ~512 bytes |
| Pulay coefficients | CPU → GPU (history doubles) | ~64 bytes |
| Convergence scalar | GPU → CPU (1 f64) | 8 bytes |
| Final density/eigenvalues | GPU → CPU (at convergence only) | once |

Total PCIe per SCF iteration: ~2 grid arrays (density H2D for VEffBuilder + V_eff D2H for diagonalize) + ~600 bytes of scalars. For 100³ wave grid: ~16 MB/iter. Acceptable.

### Why not modify chemrust-hamiltonian for GPU V_eff

VEffBuilder is pure Rust — the PBE XC is a direct handwritten port from CASTEP (`xc/kernels.rs::xc_pbe_point`), not libxc. V_eff assembly runs entirely in native code with no C FFI. However, VEffBuilder's FFT-dependent operations (density gradient for GGA XC in `pipeline.rs::compute_density_gradient`, reciprocal-space Poisson solve) use `rustfft` (CPU). Porting these to cuFFT requires changes to chemrust-hamiltonian's internal FFT dispatch. Rather than partially port VEffBuilder (creating a split CPU/GPU FFT pipeline inside chemrust-hamiltonian mid-Phase), Phase 2 wraps the existing CPU VEffBuilder and accepts the H2D/D2H for V_eff assembly. This makes V_eff assembly the least GPU-optimized part of Phase 2 — V_eff is ~15% of runtime vs 70% for diagonalize.

Future Phase 3+ can port VEffBuilder's FFT calls to cuFFT for fully GPU-resident V_eff assembly.

### Why Chebyshev filtering + cuSOLVER ZHEGVD (not LOBPCG + CPU LAPACK)

PROJECT_ROOT_PLAN SS71-78 established this. In brief: Chebyshev + cuSOLVER achieves GPU-resident RR with zero PCIe (only n_bands eigenvalues cross). LOBPCG does MPI transpose per block and requires CPU LAPACK for per-block orthogonalization. For Phase 2 (single-rank), MPI doesn't matter, but cuSOLVER ZHEGVD still avoids the n_bands² PCIe transfer of CPU LAPACK.

### Diagonalize before build_v_eff in implementation order

Goal 3 (diagonalize) is listed before Goal 4 (build_v_eff + construct_density) because:
- diagonalize carries all the project risk (Chebyshev + full H|ψ> pipeline composition)
- build_v_eff is ~40 lines of glue around validated code — no new risk
- construct_density needs real ψ from diagonalize for meaningful testing
- Implementing diagonalize first also validates the GPU infrastructure layer (Goal 2) comprehensively

However, to test diagonalize independently, an initial V_eff must be available on GPU. Options:
(a) Load V_eff from CASTEP `.pot_fmt` fixture directly (H2D to GPU)
(b) Run VEffBuilder once on CPU and H2D
Either works; the point is that diagonalize doesn't need build_v_eff to be implemented to be testable.

### Pulay mixing details

History size: 5-8 (configurable via `ScfIteration::new()` max_history parameter). Stored as ring buffer in DensityHistory:
```
densities: [ρ_i], residuals: [R_i = ρ_i - ρ_{i-1}]
Metric tensor: M_ij = <R_i | R_j> (GPU reductions → D2H)
Solve: [M  ones] [c] = [0]  (constrained: Σ c_i = 1)
       [ones 0  ] [α]   [1]
Mixed: ρ_mix = Σ c_i ρ_i + α Σ c_i R_i
```

### Module structure additions

```
chemrust-scf/src/
├── device/
│   ├── mod.rs          # Gpu<T>/Cpu<T> real sync, CudaStream management
│   ├── fft.rs          # DeviceFftPlan, cufftPlanMany wrapper
│   └── blas.rs         # cuBLAS gemm/axpy/dot wrappers
│   └── solver.rs       # cuSOLVER ZHEGVD wrapper
├── eigensolver/
│   ├── mod.rs          
│   ├── chebyshev.rs    # Chebyshev polynomial filtering on GPU ψ
│   └── rayleigh_ritz.rs # H_sub/S_sub build + ZHEGVD + ψ rotation
├── scf/
│   └── density.rs      # build_density_from_wavefunctions (batched cuFFT)
├── types.rs            # (add Add/Sub/Mul impls)
├── mixing.rs           # (Pulay algorithm replaces todo!())
├── scf.rs              # (transition bodies filled)
├── layout.rs           # (Gpu<T>/Cpu<T> real sync replaces Deref identity)
└── lib.rs
```

### GPU FFT: cudarc raw cuFFT vs gpufft

PROJECT_ROOT_PLAN SS1214-1225 evaluated gpufft. For Phase 2 we use cudarc's raw cuFFT bindings directly (`cudarc::cufft::sys`) instead of gpufft because:
- cudarc is already required for cuBLAS and cuSOLVER (single dependency)
- gpufft's batch limitation (batch=1 for 3D) blocks `cufftPlanMany` for density construction
- cudarc provides `cufftPlanMany` natively via raw sys bindings

The wrapper in `device/fft.rs` is ~50 lines of safe Rust around raw cuFFT calls — same one-time cost as gpufft integration.

## Deferred Items Absorbed

| # | Item | Absorbed into |
|---|-------|---------------|
| 1 | All transition body implementations | Goals 3, 4, 5 — this IS Phase 2 |
| 2 | `pub` → `pub(crate)` ScfIteration fields | Goal 1 |
| 3 | DensityHistory actual mixing (Pulay) | Goal 5 |
| 7 | Arithmetic ops on field types | Goal 1 |
| 8 | DensityHistory::mix() output wiring | Goal 1 (mix signature + scf.rs assignment) |

**Remaining deferred (not in Phase 2):**
| # | Item | Reason |
|---|-------|--------|
| 4 | GVectorGrid lacks Clone/Debug | Annoyance, not blocking. Manual impl if needed. |
| 5 | Type name collision (both crates define Density) | Module paths disambiguate; upstream PR separate |
| 6 | Gpu<T>/Cpu<T> sync guard replacement | Actually done in Goal 2 (GPU-direct requires it) |
| 9 | MPI transpose support | Phase 3+ (single-rank for now) |
| 10 | ScfIteration Debug/Clone | Convenience, not functionality |

## Verification

### Per-goal verification

| Goal | How verified |
|------|-------------|
| 1 (pub(crate) safety) | `cargo check` — all existing code compiles. `cargo test` — backbone_compiles still passes |
| 2 (GPU infra) | Unit tests for FFT plan creation, BLAS correctness on small matrices, ZHEGVD on 4×4 test system |
| 3 (diagonalize) | Integration test: load CASTEP V_eff from `.pot_fmt` fixture, run one diagonalize, compare eigenvalues against `.bands` reference |
| 4 (construct_density + build_v_eff) | Integration test: single SCF iteration (build_v_eff → diagonalize → construct_density), compare each output against CASTEP fixtures |
| 5 (mix + check) | Integration test: two consecutive iterations, verify ρ mixing reduces residual, verify convergence check fires within expected iterations |
| 6 (full SCF validation) | `cargo test cu111_co_full_scf` — converges within CASTEP's iteration count, total energy within 1e-5 eV/atom |

### End-to-end acceptance criteria

1. `cargo check --workspace` — 0 errors
2. `cargo clippy --workspace -- -D warnings` — 0 warnings
3. `cargo test --workspace` — all tests pass (GPU tests require CUDA-capable GPU)
4. Cu111_CO full SCF converges to within 1e-5 eV/atom of CASTEP reference total energy
5. No `unsafe` outside `device/` module boundary functions (quarantined)
