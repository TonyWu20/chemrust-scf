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

> **Source-audited 2026-05-19.**  Original E-1/E-2 descriptions were written
> from general domain knowledge without source citations.  Every line was
> wrong: constrained Pulay → unconstrained DIIS, real-space → reciprocal-space,
> density differences → residual differences, max|Δρ| → energy-window.
> Rewritten against CASTEP `dm.f90:895-1093` and `electronic.f90:7536-7644`.

**Source references:**
- `~/programming/CASTEP-GPU-port/Source/Functional/dm.f90:895-1093` — `dm_mix_density_pulay`
- `dm.f90:620-719` — `dm_mix_density_kerker`
- `dm.f90:2296-2365` — `dm_mix_density_dot` (inner product)
- `dm.f90:2367-2542` — real↔reciprocal space conversion
- `dm.f90:2544-2598` — `dm_apply_kerker` (preconditioner application)
- `~/programming/CASTEP-GPU-port/Source/Functional/electronic.f90:7536-7644` — `electronic_store_energy`
- `~/programming/CASTEP-GPU-port/Source/Fundamental/parameters.f90:210-216` — mixing parameters

#### Mixing Phase Type-State Machine

CASTEP's mixing has three distinct modes that form a natural state machine,
encoded as a type parameter `M: MixingPhase` on `DensityHistory<M>` and
`DensityUpdated<M>`:

```
MixingOff ──(energy stabilizes)──→ Kerker ──(first mix done)──→ Pulay/DIIS
    │                                  │                          │
    └── density passes through         └── n_new = n_in + K·R     └── full DIIS
        unchanged (no mixing)              (preconditioned)            (history ≥ 1)
```

A runtime `MixingPhaseKind` enum (`Off`, `Kerker`, `Pulay`) is stored on
`ScfIteration` for the `run_scf` loop to dispatch to the correct
`construct_density_*` method.  This is the only runtime branch; all other
code is monomorphized at compile time.

#### E-1: Kerker preconditioner setup

**Kind:** design

**Guidance:** Precompute K(G) = G²/(G²+q²) on GPU.  K(G=0) = 0 (no DC
mixing).  K(G) → 1 as |G| → ∞.  For Phase 2, use q = 1.5 a.u. (typical
for metals; CASTEP autocomputes from Thomas-Fermi screening).

**Files:** new `src/mixing/kerker.rs`

**Success Criteria:**
- K(G=0) = 0.0, K(high G) ≈ 1.0
- Precomputed `DeviceBuffer<f64>` on GPU

---

#### E-2: Reciprocal-space DIIS + Kerker mixing (GPU-native)

**Kind:** design

**Guidance:** Implement `DensityHistory<M>::mix()` matching CASTEP's
`dm_mix_density_pulay`.  All GPU-resident except the tiny DIIS linear
solve (CPU, history_size ≤ 7).

**CASTEP algorithm (source-audited):**

Mixing is done in **reciprocal space**: density is FFT'd (R2C), PW
coefficients are mixed, then inverse-FFT'd back.  High-frequency components
(G > mix_cut_off_energy) pass through unchanged.  For Phase 2 we use the
full reciprocal grid (the FFT grid cutoff is sufficient).

Residual: **R = n_out - n_in** (output minus input density), both in
reciprocal PW-coefficient representation.

History stores **deltas, not absolutes** (dm.f90:999-1000):
- `density_history(i) = n_in(current) - n_in(previous)`  (Δn_in)
- `residual_history(i) = R(current) - R(previous)`       (ΔR)

DIIS is **unconstrained** (dm.f90:1011-1028): builds matrix
`M_ij = <ΔR_j | ΔR_i>` and RHS `b_i = -<ΔR_i | R_current>`, solves
`M · c = b` via `dgesv`.  No Lagrange multiplier (Σc_i = 1 is NOT enforced).

Update formula (dm.f90:1045-1063):
```
n_new = n_in + Σc_i·Δn_i + K·[R_current + Σc_i·ΔR_i]
```
where K is the Kerker preconditioner applied per PW coefficient.

On `dgesv` failure → fallback to Kerker mixing for this step (dm.f90:1029-1041).

Inner product (dm.f90:2342-2343):
```
<den1 | den2> = Σ_ipw conjg(den1%charge(ipw)) · den2%charge(ipw) · mix_metric(ipw)
```
For Phase 2, use uniform metric (= 1.0) since Kerker is applied separately.

**Type-state API:**
```rust
impl DensityHistory<MixingOff> {
    fn new(kerker: Gpu<KerkerPreconditioner>) -> Self;
    fn into_kerker(self) -> DensityHistory<Kerker>;  // no mix() — passthrough
}
impl DensityHistory<Kerker> {
    fn mix(&mut self, density: Density) -> (Density, Density);  // Kerker
    fn into_pulay(self) -> DensityHistory<Pulay>;
}
impl DensityHistory<Pulay> {
    fn mix(&mut self, density: Density) -> (Density, Density);  // DIIS
}
```

**Files:** rewrite `src/mixing.rs`, new `src/mixing/reciprocal_density.rs`

**Success Criteria:**
- `DensityHistory<MixingOff>` has no `mix()` (compile-time safety)
- Kerker: first mix uses K·(n_out - n_in), residual norm decreases
- Pulay: DIIS with growing history, residual decreases monotonically
- Solve failure: falls back to Kerker, no panic
- All GPU-resident except 7×7 DIIS matrix D2H (negligible)

---

#### E-3: Total energy computation

**Kind:** design

**Guidance:** Compute E_total = Σ_i f_i ε_i - E_H + E_xc - ∫ρV_xc + E_ewald.

Components:
1. **E_band** = Σ_i f_i · ε_i (eigenvalues × occupations, already on CPU)
2. **E_H** = 1/2 · Σ_r ρ(r) · V_H(r) · dV (GPU: element-wise multiply + sum)
3. **E_xc** from `PbeXcResult.energy` (currently discarded by VEffBuilder)
4. **∫ρV_xc** = Σ_r ρ(r) · V_xc(r) · dV
5. **E_ewald**: standard Ewald summation (CPU, ~50 lines)

For E_H, E_xc, and ∫ρV_xc: branch chemrust-hamiltonian as
`feat/expose-energy` to add `VEffAssemblyResult { v_eff, e_hartree, e_xc,
v_xc_integral }` return type.  Depend on the branch from chemrust-scf.
Submit PR upstream later.

**Files:** new `src/energy.rs`, modify `chemrust-hamiltonian-core/src/band_structure.rs`,
modify `src/scf.rs` (store `total_energy: Option<f64>`)

**Success Criteria:**
- E_ewald for Cu111_CO matches CASTEP reference within 1e-6 eV
- Total energy physically reasonable (~-2.4×10⁴ eV for Cu111_CO)
- All energy components non-NaN

---

#### E-4: Energy-window convergence check

**Kind:** direct

**Guidance:** Fill `ScfIteration::check()` matching CASTEP's
`electronic_store_energy` (electronic.f90:7536-7644).

Algorithm:
1. Push `total_energy` into cyclic `energies[elec_convergence_win]` buffer
2. Track `mixed_status` (CASTEP lines 7614-7638): mark when mixing is active,
   require full convergence window to use mixed densities
3. `max_E - min_E ≤ elec_energy_tol × num_ions`? → check mixed_status → Converged/NotConverged
4. Determine next `MixingPhaseKind`:
   - `Off → Kerker` when energy diff < `mixing_convergence_tol` (0.1 eV default)
   - `Kerker → Pulay` after first mix completes
   - `Pulay → Pulay` for normal DIIS
5. `elec_energy_tol`: 1e-5 eV (metals), `elec_convergence_win`: 3

**Files:** `src/scf.rs` (rewrite `check()`, modify `CheckOutcome`, add energy
buffer + mixed_status fields)

**Success Criteria:**
- Single iteration: NotConverged with next_mixing=Off
- After energy stabilizes: next_mixing transitions Kerker→Pulay
- Converged when energy window satisfies tolerance with mixed densities

---

#### E-5: Wire mixing phase into run_scf loop

**Kind:** direct

**Guidance:** Add `MixingPhaseKind` runtime enum.  Split `construct_density()`
into `construct_density_off/kerker/pulay` methods on `WavefunctionsUpdated`.
`run_scf` matches on `state.next_mixing` to dispatch.  `CheckOutcome::NotConverged`
carries `next_mixing`.

**Files:** `src/scf.rs` (add `MixingPhaseKind`, modify `run_scf`, add three
`construct_density_*` methods)

**Success Criteria:**
- First iterations: passthrough (no mixing)
- After threshold: Kerker, then Pulay
- `cargo check` and `cargo test` pass

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
- **Group E CASTEP source audit (2026-05-19):** Read `dm.f90:895-1093`,
  `dm.f90:620-719`, `dm.f90:2296-2365`, `dm.f90:2367-2542`,
  `dm.f90:2544-2598`, `electronic.f90:7536-7644`, and
  `parameters.f90:210-216`.  Key findings:
  - Mixing is in **reciprocal space** (PW coefficients), not real-space grid
  - Algorithm is **unconstrained DIIS** (`dgesv`), not constrained Pulay
    (Lagrange multiplier)
  - Residual is **R = n_out - n_in**, not ρ_new - ρ_old
  - History stores **Δn_in and ΔR** (deltas), not absolute densities
  - **Kerker preconditioning** K(G) = G²/(G²+q²) is mandatory, not optional
  - First iteration uses **Kerker**, not Pulay (no history yet yet)
  - dgesv **failure fallback** to Kerker (singular DIIS matrix)
  - Convergence is **energy-window** based, not max|Δρ|
  - `mixing_convergence_tol = 0.1 eV` controls when mixing starts
  - `elec_convergence_win = 3`, `mix_history_length = 7`
  - Mixed-status tracking prevents false convergence when mixing is off
  - Original E-1/E-2 were written from general knowledge; every line was wrong.
    Rewritten against actual CASTEP source.
- **chemrust-hamiltonian energy exposure:** VEffBuilder computes XC energy
  (`PbeXcResult.energy`) but discards it in `assemble()`.  Hartree energy
  (E_H = 1/2 ∫ ρ·V_H dr) is never computed.  Ewald summation is completely
  absent.  Plan: branch `feat/expose-energy` in chemrust-hamiltonian to add
  `VEffAssemblyResult { v_eff, e_hartree, e_xc, v_xc_integral }` return type.
  Depend on branch from chemrust-scf; submit PR upstream after Phase 2.
- **Ewald summation:** Standard real/reciprocal split with erfc screening.
  α = √π / V^(1/3).  Real-space cutoff ~10 Å, reciprocal cutoff ~20 G-vectors
  per direction.  ~50 lines CPU-side, negligible cost.
- **GPU mixing FFT cost:** One cuFFT R2C + one C2R per mix call.  Acceptable
  for Phase 2 since VEffBuilder already does D2H/H2D for CPU FFT (rustfft).
  Phase 3 can fuse the V_eff FFT with the mixing FFT.

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
