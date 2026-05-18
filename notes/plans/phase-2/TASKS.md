# Phase 2: Fill the SCF transitions (GPU-direct) — Task Record

## Declared Fixtures

- **Cu111_CO CASTEP reference:** `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
  Contains `.cell`, `.param`, `.castep_bin`, `.den_fmt`, `.pot_fmt`, `.bands` files.
  - Reference total energy: **-24110.96665069 eV** (18 ions)
  - Validation tolerance: 2e-4 eV total (~1e-5 eV/atom with guard factor for GPU FP associativity)
- **Pseudopotentials:** `~/Downloads/Potentials/` — USP files for Cu, C, O species.
- **GPU crash logs (informative):** `/tmp/cu111_gpu_resident_scf/` — CASTEP Bug 1-8 patterns.
- **Cu111_CO V_eff fixture:** `.pot_fmt` loaded via chemrust-hamiltonian `parse_pot_fmt` for testing diagonalize before build_v_eff.

## Task Groups

### Group A — Infrastructure (Goal 1 + flake.nix prerequisite)

#### A-1: Lock type-state safety (`pub` → `pub(crate)`)

**Kind:** direct

**Guidance:** Change all fields on `ScfIteration` from `pub` to `pub(crate)`.
Add `pub(crate)` accessor methods needed by the integration test. The
`backbone_compiles` test constructs `ScfIteration` via builder — it should
not need field access. If any test code reads fields directly, add accessor
methods rather than reverting to `pub`.

**Files:** `src/scf.rs`

**Success Criteria:**
- `cargo check` passes — no `pub` fields on `ScfIteration` remain
- `backbone_compiles` test still passes (builder API unchanged)

---

#### A-2: Arithmetic ops on grid types

**Kind:** direct

**Guidance:** Add `Add`, `Sub`, `Mul<f64>`, `AddAssign`, `SubAssign` impls on
`WaveGridArray`, `FineGridArray`, `Density`, `EffectivePotential`,
`DensityUpsampled`. These are needed by mix (ρ_new - ρ_old), check
(max|ρ_mix - ρ_old|), and Pulay metric tensor (<R_i|R_j>). All ops delegate
to the underlying `Array3<f64>` operations element-wise.

**Files:** `src/types.rs`

**Success Criteria:**
- `Density::from_inner(a) + Density::from_inner(b)` compiles
- `Density::from_inner(a) * 0.5_f64` compiles
- `Density::from_inner(a) - Density::from_inner(b)` compiles

---

#### A-3: Wire DensityHistory::mix() return values

**Kind:** direct

**Guidance:** `DensityHistory::mix()` already returns `(Density, Density)` —
`(mixed_density, input_snapshot)`. The `mix()` transition in `scf.rs` must
consume both: `self.density = mixed_density`, `self.previous_density =
input_snapshot`. The return type of `mix()` transition remains
`ScfIteration<S, Mixed>`.

**Files:** `src/scf.rs`, `src/mixing.rs`

**Success Criteria:**
- `cargo check` — no unused-variable warnings on mix return values

---

#### A-4: flake.nix CUDA configuration

**Kind:** direct

**Guidance:** Add CUDA toolkit support to `flake.nix`. Model on CASTEP-GPU-port
pattern:
- `cudaSupport = true`, `cudaCapability = ["6.1"]`, `cudaVersion = "12.9"`
- `cudaOverlay` with `enableParallelBuilding` on cuda_nvcc, cuda_cudart,
  cuda_cccl, libcublas, libcusolver
- cuDNN override at v9.11.1.4 (required for Pascal cc 6.1 toolchain compatibility)
- `pkgsCuda.cudaPackages_12_9` symlinkJoin as a single devShell input
  including cuFFT, cuBLAS, cuSOLVER, cuRAND, NVRTC
- CUDA env vars: `CUDA_HOME`, `CUDA_INCLUDE`, `CUDA_LIB`
- Keep fenix overlay for Rust toolchain (unlike CASTEP flake which uses custom
  overlays.nix)

**Reference:** `~/programming/CASTEP-GPU-port/flake.nix` lines 37-80 (CUDA overlay).

**Files:** `flake.nix`

**Success Criteria:**
- `nix develop` enters shell with `nvcc` in PATH
- `CUDA_HOME` points to the symlinkJoin
- `cargo check` works from within the nix shell

---

### Group B — GPU Infrastructure (Goal 2)

#### B-1: Add cudarc dependency

**Kind:** direct

**Guidance:** Add `cudarc` to `Cargo.toml` with appropriate version and feature
flags. Target CUDA 12.x (`cuda-12050` or matching version). The crate provides
FFI bindings for cuFFT, cuBLAS, cuSOLVER, and CUDA driver/runtime.

**Files:** `Cargo.toml`

**Success Criteria:**
- `cargo check` from within nix shell resolves cudarc
- No CUDA header or library not-found errors

---

#### B-2: `src/device/mod.rs` — Gpu<T>/Cpu<T> real sync

**Kind:** design

**Guidance:** Replace Phase 1's transparent `Deref`-to-`T` with real sync guards.
`Gpu<T>` holds `CudaStream` handle + `DeviceBuffer<T>` for the inner type.
`sync_to_host(&self, stream) -> Cpu<T>` does D2H transfer.
`Cpu<T>::sync_to_device(...) -> Gpu<T>` does H2D transfer. No `Deref` to `T`
on `Gpu<T>` — explicit access only.

Requirements:
- `Gpu<T>` must be constructable from `T` (for H2D on creation)
- `Cpu<T>` derefs to `T` (CPU data is always accessible)
- `Gpu<T>` provides `as_device_buffer(&self) -> &DeviceBuffer<T>` for
  low-level access
- Both are `Send` and `Sync` (required by stream ownership model)

**Files:** new `src/device/mod.rs`, modify `src/layout.rs`

**Success Criteria:**
- `Gpu<Density>::sync_to_host(&stream)` returns `Cpu<Density>`
- No `Deref<Target=T>` impl on `Gpu<T>`
- `Cpu<T>` still derefs to `T` (CPU convenience)

---

#### B-3: `src/device/fft.rs` — cuFFT wrapper

**Kind:** design

**Guidance:** Wrap cudarc's raw cuFFT sys bindings. Expose:
- `DeviceFftPlan<T>` parametrized over real/complex types
- `forward_3d(plan, input, output, stream) -> Result<()>`
- `inverse_3d(plan, input, output, stream) -> Result<()>`
- `batched_3d(plan, input, output, n_batches, stream) -> Result<()>`
  (wraps `cufftPlanMany` for density construction)
- `Plan1d`, `Plan2d`, `Plan3d` construction helpers with `cufftCreate` +
  `cufftSetStream`

The wrapper must handle:
- C2C (complex→complex), R2C (real→complex), C2R (complex→real)
- `cufftPlanMany` for strided multi-batch 3D C2R (density construction needs
  n_bands simultaneous IFFTs)
- Stream association (all transforms are async on the given stream)

**Files:** new `src/device/fft.rs`

**Success Criteria:**
- Unit test: C2C forward+inverse on 8³ grid returns identity
- Unit test: batched C2R for 4 bands on 4³ grid

---

#### B-4: `src/device/blas.rs` — cuBLAS wrapper

**Kind:** design

**Guidance:** Wrap cudarc's raw cuBLAS sys bindings. Expose:
- `gemm(...)` — general matrix multiply (C = α·A·B + β·C)
- `gemv(...)` — matrix-vector multiply
- `axpy(...)` — y = α·x + y (scalar-vector addition)
- `dot(...)` — inner product reduction

All operations are stream-associated. Handle layout: cuBLAS uses column-major;
our `WavefunctionSet` stores per-band data contiguously (n_pw × n_bands). The
ghost layout of data in device memory respects ColumnDistributed layout.

**Files:** new `src/device/blas.rs`

**Success Criteria:**
- Unit test: gemm for small complex matrix multiplication on GPU
- Unit test: axpy for vector scaling on GPU

---

#### B-5: `src/device/solver.rs` — cuSOLVER ZHEGVD wrapper

**Kind:** design

**Guidance:** Wrap cudarc's raw `cusolverDnZhegvd` sys binding. The wrapper:
1. Queries workspace size via `cusolverDnZhegvd_bufferSize`
2. Allocates workspace `DeviceBuffer`
3. Launches `cusolverDnZhegvd` on the stream (blocking for the caller)
4. Reads `info` (DeviceBuffer) to check success
5. Returns eigenvalues as `Cpu<Vec<f64>>` (D2H, n_bands × 8 bytes) and
   overwrites input matrix with eigenvectors in-place

Signature:
```rust
pub fn zhegvd(
    handle: &DnHandle,
    jobz: i32,   // CUSOLVER_EIG_MODE_VECTOR = 1
    uplo: i32,   // CUBLAS_FILL_MODE_LOWER = 1
    n: i32,
    a: &mut DeviceBuffer<Complex64>,  // H_sub → X (overwritten in-place)
    b: &mut DeviceBuffer<Complex64>,  // S_sub (overwritten, workspace)
    eigenvalues: &mut DeviceBuffer<f64>,
    info: &mut DeviceBuffer<i32>,
    stream: &CudaStream,
) -> Result<(), SolverError>
```

**Files:** new `src/device/solver.rs`

**Success Criteria:**
- Unit test: solve 4×4 Hermitian generalized eigenproblem on GPU
- Eigenvalues match CPU LAPACK reference

---

### Group C — diagonalize (Goal 3)

#### C-1: Chebyshev filtering on GPU

**Kind:** design

**Guidance:** Implement Chebyshev polynomial subspace iteration on GPU.

Input: `Gpu<WavefunctionSet<ColumnDistributed>>`, `Gpu<EffectivePotential>`
(on fine grid), `PseudopotentialSet`, `CellGeometry`, ndeg (polynomial degree).

Steps:
1. Estimate eigenvalue bounds: apply H to a random vector, compute
   Rayleigh quotient bounds via one H|ψ> + inner product. Alternatively,
   use kinetic energy estimate: [0, 0.5|k+G_max|² + max(V_eff)].
2. Compute Chebyshev coefficients a_k for the polynomial that amplifies
   eigenvalues above the filter window (the upper ~50% of the spectrum).
3. For k = 1..ndeg:
   - Apply H_loc = T + V_eff: FFT roundtrip via cuFFT. Scatter G-vector
     coefficients to FFT grid indices, IFFT to real space, multiply by
     V_eff(r), FFT back, gather back to coefficient array.
   - Apply V_NL: β-projector gemm (cuBLAS) using pre-computed β_phi from
     PseudopotentialSet. Screen D-matrix at the selected level
     (ScreenLevel::None for Phase 2, matching chemrust-hamiltonian's
     default in nlpot.rs).
   - Full H|ψ> = H_loc|ψ> + V_NL|ψ>.
   - Chebyshev recurrence: ψ_{k+1} = 2 · Ĥ_norm · (H|ψ_k>) - ψ_{k-1}
     via cuBLAS axpy/scal.
   - Norm stability check: if ||ψ_k||/||ψ_{k-1}|| > threshold × last
     ratio, return Err(NormDiverged) (circuit breaker for exponential
     blowup — CASTEP Bug 4).
4. ColumnDistributed → RowDistributed transpose (single-rank: reorder
   data in-place from [n_pw, n_bands] to [n_bands, n_pw] layout).

**Reference:** chemrust-hamiltonian `hamiltonian.rs::apply_local_hamiltonian` for
the FFT roundtrip pattern. `nlpot.rs::apply_nlpot` for V_NL apply.

**Files:** new `src/eigensolver/chebyshev.rs`

**Success Criteria:**
- Chebyshev filtering on Cu111_CO test system runs without divergence
  (using V_eff loaded from `.pot_fmt` fixture)
- Filtered subspace has lower eigenvalue residuals (||Hψ - εψ||) than
  initial random ψ

---

#### C-2: Rayleigh-Ritz on GPU

**Kind:** design

**Guidance:** Compute subspace eigenvalues and rotate ψ.

Input: `Gpu<WavefunctionSet<RowDistributed>>` (from Chebyshev filtering),
`Gpu<EffectivePotential>`, PseudopotentialSet, CellGeometry.

Steps:
1. Build subspace matrices:
   - H_sub = ψ† · H · ψ: apply H to ψ (same H_loc + V_NL as Chebyshev),
     then compute ψ† · (Hψ) via cuBLAS gemm: C = ψ·conjugate_transpose × (Hψ).
     Shape: n_bands × n_bands (complex).
   - S_sub = ψ† · ψ: cuBLAS gemm with the two-argument identity case.
     For USPP (not PAW), S = I, so S_sub = ψ†·ψ is diagonal-dominant but
     not exactly identity for non-orthogonal trial ψ.
2. Solve H_sub · X = ε · S_sub · X via `zhegvd` from device/solver.rs.
   - Input matrices overwritten in-place
   - Eigenvalues ε returned as `Cpu<Vec<f64>>`
   - Eigenvectors X (n_bands × n_bands complex) in-place
3. Rotate: ψ_new = X · ψ via cuBLAS gemm.
   - X is n_bands × n_bands, ψ is n_bands × n_pw (RowDistributed)
   - Result: n_bands × n_pw, same layout
4. Store eigenvalues in `self.eigenvalues`.

**Files:** new `src/eigensolver/rayleigh_ritz.rs`

**Success Criteria:**
- Eigenvalues match CASTEP `.bands` reference within 1e-3 eV after one
  diagonalize call (with V_eff from `.pot_fmt`)
- ψ norm ||ψ_i|| ~= 1.0 for each band after rotation

---

#### C-3: Wire diagonalize transition in scf.rs

**Kind:** direct

**Guidance:** Fill `ScfIteration::diagonalize()` body. It:
1. Takes `self` (VEffBuilt phase, guaranteed `v_eff: Some(...)`)
2. Calls Chebyshev filtering then Rayleigh-Ritz
3. Returns `ScfIteration<S, WavefunctionsUpdated>` with updated
   `self.psi` and `self.eigenvalues`

The `ndeg` parameter is an argument (not part of state) to allow caller
control. For Phase 2, `ndeg=8` is the default (matching Abinit paper's
recommendation).

**Files:** `src/scf.rs`

**Success Criteria:**
- `cargo check` — diagonalize body compiles with real CUDA calls
- The transition only exists on `ScfIteration<..., VEffBuilt>`

---

### Group D — construct_density + build_v_eff (Goal 4)

#### D-1: Batched density construction on GPU

**Kind:** design

**Guidance:** Implement `build_density_from_wavefunctions` on GPU.

Input: `Gpu<WavefunctionSet<ColumnDistributed>>`, `&[f64]` eigenvalues,
`SmearingParams`, `GVectorGrid`.

Steps:
1. Compute occupation numbers: Fermi-Dirac smearing with chemical potential
   search (bisection or analytic integration). This is CPU (n_bands ≪ grid).
2. Batched IFFT via `cufftPlanMany`: n_bands simultaneous C2R 3D transforms.
   ColumnDistributed layout has per-band contiguity — no data reordering needed
   (unlike Abinit paper's D2H step).
3. |ψ_i(r)|² for each band on GPU: element-wise complex-square accumulation.
4. ρ_new(r) = Σ_i occ_i × |ψ_i(r)|²: scalar-weighted sum, one element-wise
   reduction (fused kernel or axpy loop).
5. Optional: symmetrize density (deferred for Phase 2).

**Files:** new `src/scf/density.rs`

**Success Criteria:**
- Density preserves total charge: ∫ ρ(r) dr = total_electrons (within 1e-8)
- ρ_new matches CASTEP `.den_fmt` within 1e-6 e/Bohr³

---

#### D-2: V_eff assembly wrapper

**Kind:** direct

**Guidance:** Fill `ScfIteration::build_v_eff()` body. Steps:
1. D2H density from GPU to CPU (`density.sync_to_host(&stream)`)
2. Call `VEffBuilder::<S>::new(&cell, &pots, &fine_grid)
   .assemble_on_fine_grid(&rho, &wave_grid, &fine_grid)?`
3. H2D V_eff from CPU to GPU: wrap in `Gpu<EffectivePotential>` via
   `sync_to_device`
4. Store as `self.v_eff = Some(gpu_v_eff)`

For Phase 2, the D2H/H2D cost is unavoidable (VEffBuilder uses rustfft).
Future Phase 3+ ports VEffBuilder's FFT calls.

**Files:** `src/scf.rs`

**Success Criteria:**
- V_eff after H2D matches CPU V_eff from direct VEffBuilder call (bitwise
  within FP associativity)
- V_eff on GPU can be consumed by diagonalize transition

---

### Group E — mix + check (Goal 5)

#### E-1: Pulay mixing

**Kind:** design

**Guidance:** Replace `todo!()` in `DensityHistory::mix()` with Pulay algorithm.

Input: new ρ (Density), stored history of ρ_i and R_i = ρ_i - ρ_{i-1}.

Algorithm:
1. Compute residual R_new = ρ_new - ρ_old (GPU subtraction via cuBLAS axpy)
2. Append to history ring buffer (max `max_history` slots)
3. Build Pulay metric tensor M_ij = <R_i | R_j> for all i,j in history
   (GPU dot product reductions via cuBLAS). D2H the history_size² matrix.
4. Solve constrained linear system on CPU:
   ```
   [M  1] [c] = [0]    (Σ c_i = 1 constraint)
   [1ᵀ  0] [α]   [1]
   ```
   Use a small LAPACK solve or hand-rolled Gaussian elimination
   (history_size ≤ 8, system is tiny).
5. H2D coefficients c_i (history_size floats)
6. ρ_mix = Σ c_i ρ_i + α Σ c_i R_i (GPU AXPY combination via cuBLAS)
7. Return: `(ρ_mix, ρ_new)` — the mixed density and the input snapshot

**Files:** `src/mixing.rs`

**Success Criteria:**
- ρ_mix is a linear combination of history densities (coefficients sum to 1)
- Residual norm ||R_mix|| ≤ ||R_new|| (mixing reduces residual)

---

#### E-2: Convergence check

**Kind:** direct

**Guidance:** Fill `ScfIteration::check()` body.

1. Compute density delta: Δρ = ρ_mix - ρ_old (GPU subtraction)
2. Reduction: max|Δρ| over all grid points (GPU: `cublasIdamax`
   equivalent, or a custom reduction kernel)
3. D2H: single f64 crosses PCIe
4. Compare against tol:
   - diff < tol: return `Ok(CheckOutcome::Converged(final_state))`
   - else: return `Ok(CheckOutcome::NotConverged(restart_state))`
     where restart_state has `v_eff: None` (caller must rebuild)

**Files:** `src/scf.rs`

**Success Criteria:**
- Single-iteration check on Cu111_CO returns NotConverged (won't converge
  in one iteration)
- Converged test with artificially low tol returns Converged

---

### Group F — CASTEP Validation (Goal 6)

#### F-1: Fixture loading infrastructure

**Kind:** design

**Guidance:** Build test helpers to load Cu111_CO reference data.

Reuse chemrust-hamiltonian's:
- `castep_bin::CastepBinFile::read()` — reads `.castep_bin` for cell data,
  wavefunction coefficients, eigenvalues
- `formatted::parse_den_fmt()` — reads `.den_fmt` → reference density
- `formatted::parse_pot_fmt()` — reads `.pot_fmt` → reference V_eff
- `PseudopotentialSet::from_directory()` — loads USP files

Constants:
- Fixture dir: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
- Reference energy: `-24110.96665069 eV` (18 atoms: Cu(111) slab + CO)
- Tolerance: 2e-4 eV total

**Files:** `tests/fixtures/cu111_mod.rs` (module with loading helpers)

**Success Criteria:**
- Parse CASTEP binary format, extract cell, ψ, eigenvalues
- Load reference V_eff from `.pot_fmt` matches internal VEffBuilder output
- Load reference density from `.den_fmt`

---

#### F-2: Per-step comparison test

**Kind:** direct

**Guidance:** Integration test that runs one SCF iteration and compares
each step against CASTEP reference:
1. Load initial state from fixtures
2. `build_v_eff` → compare V_eff against `.pot_fmt` (H2D then D2H for
   comparison — use absolute tolerance for FP differences)
3. `diagonalize` → compare eigenvalues against `.bands` (within 1e-4 eV)
4. `construct_density` → compare ρ_new against `.den_fmt`

**Files:** `tests/single_iteration_validation.rs`

**Success Criteria:**
- All three comparisons pass with per-type tolerances

---

#### F-3: Full SCF convergence test

**Kind:** direct

**Guidance:** Integration test that runs `run_scf` to convergence and
compares final total energy against CASTEP reference.

```
let state: ScfIteration = load_cu111_state()?;
let result = run_scf(state, ndeg=8, tol=1e-8)?;
let diff = (result.total_energy - (-24110.96665069)).abs();
assert!(diff < 2e-4, "...");
```

**Files:** `tests/full_scf_validation.rs`

**Success Criteria:**
- SCF converges within 50 iterations (CASTEP converges in 17)
  (First run: may need more iterations due to GPU prelim stages —
  the 50-iteration limit is generous)
- Total energy within 2e-4 eV of -24110.96665069 eV

## Exploration Notes

- **Rust edition 2024 note:** Path dependencies in Cargo.toml must use
  `resolver = "2"` (edition 2024 defaults to this). cudarc may require
  `edition 2021` features — verify during B-1.
- **VEffBuilder API:** `VEffBuilder::assemble_on_fine_grid(ρ, &wave_grid,
  &fine_grid)` returns `EffectivePotential` on the fine grid. The density
  upsampling is handled internally. GPU wrapper D2H→VEffBuilder→H2D must
  restore the GPU buffer with `sync_to_device`.
- **cudarc version:** Latest stable is v0.19.8 (CUDA 12.x). Verify
  feature flags: `cuda-12050` matches the system CUDA version.
- **cuSOLVER ZHEGVD:** The workspace-size query requires calling
  `cusolverDnZhegvd_bufferSize` first. Allocate `workspace` on the same
  stream. The solver call is blocking on the stream — the caller must
  synchronize before reading any results D2H'd from the same stream.
- **ColumnDistributed layout for cuFFT:** Per-band coefficients are
  contiguous in memory. For batched cuFFT: n_bands = batch count,
  each band's n_pw maps to the FFT grid indices via the G-vector index
  mapping from GVectorGrid. The scatter/gather pattern for PW ↔ FFT grid
  is the same as chemrust-hamiltonian's `apply_local_hamiltonian` but
  performed on GPU-resident device buffers.

## Verification

```bash
# From nix develop shell:
cargo check                              # Must succeed
cargo clippy --workspace -- -D warnings  # Must succeed (0 warnings)
cargo test                               # GPU tests require CUDA-capable GPU

# Specific validation:
cargo test cu111_co_full_scf             # Full SCF convergence test
cargo test single_iteration_validation   # Per-step comparison against CASTEP

# Expected output (first run, before all transitions are filled):
# - A-1, A-2, A-3: compile-only, no runtime test
# - A-4: nix develop enters CUDA shell
# - B-2, B-3, B-4, B-5: per-cuFFT/cuBLAS/cuSOLVER unit tests
# - C-1, C-2: diagonalize integration test
# - D-1, D-2: density + V_eff tests
# - E-1, E-2: mixing + check tests
# - F-1, F-2, F-3: full validation against CASTEP
```
