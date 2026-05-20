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

> **Source-audited 2026-05-20.**  Read the actual CASTEP `.castep` file
> (lines 286–324) for the SCF output format, verified `.check` stores
> only converged state, checked grid-indexing conventions in
> `castep_bin/density.rs` and `scf.rs`, traced `compute_occupations`
> call graph to find missing Fermi energy plumbing.

**Source references:**
- `Cu111_CO.castep:286-324` — SCF convergence output format
- `castep_bin/density.rs:193-223` — `reconstruct_density` grid transposition
- `castep_bin/parameters.rs:118` — `field_meta.grid` stored as `[ngx, ngy, ngz]`
- `scf.rs:917-931` — `pw_coords_to_fft_indices` C-order FFT index convention
- `density.rs:39-46,50-75` — `compute_occupations` discards chemical potential μ

**`.check` caveat:** The `.check` file stores the **converged** state (dumped at
end of calculation, `Cu111_CO.castep` line 333).  Loading it and running one SCF
iteration tests fixed-point stability, not convergence dynamics.  CASTEP's initial
SCF (pseudoatomic calculations, atomic-superposition starting density) is not
implemented in chemrust-hamiltonian.  F-3 addresses this with a perturbation-recovery
test.

#### F-0: Per-iteration SCF tracing output

**Kind:** direct

**Guidance:** Add `tracing`-based per-iteration output to `run_scf_with_energy`
matching CASTEP's SCF column format (`.castep` lines 286–324):

```
------------------------------------------------------------------------ <-- SCF
SCF loop      Energy           Fermi           Energy gain       Timer   <-- SCF
                               energy          per atom          (sec)   <-- SCF
------------------------------------------------------------------------ <-- SCF
      1  -2.30069691E+004  2.36299163E+000   6.15319444E+001      25.03  <-- SCF
```

Columns: iteration, total energy (eV), Fermi energy (eV), energy gain per atom
(eV), wall time (sec).  CASTEP prints free energy (E−TS) in this table; we print
total electronic energy.

At `tracing::debug!` level, emit energy decomposition:
```
  E_band={:.8E}  -E_H={:.8E}  E_xc={:.8E}  -∫ρVxc={:.8E}  E_ewald={:.8E}
```

**Prerequisites (included in this task):**

1. Add `tracing = { version = "0.1", default-features = false, features =
   ["std"] }` to `Cargo.toml`.

2. Add domain newtypes to `src/types.rs`:
   - `Occupations(Vec<f64>)` — occupation numbers per band
   - `ChemicalPotential(f64)` — chemical potential μ from smearing search (Hartree)

3. Add unit constants to `src/energy.rs`:
   - `pub(crate) const EV_TO_HARTREE: f64 = 1.0 / 27.211384;`
   - `pub(crate) const HARTREE_TO_EV: f64 = 27.211384;`
   
   Re-export both from `src/lib.rs`.

4. Add `fermi_energy: Option<f64>` field to `ScfIteration`, plumb through
   `into_phase` and constructor.

5. Change `compute_occupations` return type from `Vec<f64>` to
   `(Occupations, ChemicalPotential)`.  Update both call sites in `scf.rs`
   (`compute_density_from_wavefunctions` and `check()`) to store `mu.0`
   as `self.fermi_energy`.

6. In `run_scf_with_energy`: add `Instant` timer, iteration counter, header
   print on first iteration, per-iteration `tracing::info!` after `check()`.

Output is gated on `RUST_LOG=info` (no output by default).

**Files:** `Cargo.toml`, `src/types.rs`, `src/energy.rs`, `src/density.rs`,
`src/scf.rs`, `src/lib.rs`

**Success Criteria:**
- `RUST_LOG=info cargo test ...` emits one line per SCF iteration in CASTEP format
- `RUST_LOG=debug cargo test ...` additionally emits energy decomposition
- Fermi energy column is non-zero and physically reasonable (~ -0.12 Hartree for Cu111_CO)
- `cargo check` — no unused field warnings on `fermi_energy`

---

#### F-1: Fixture loading infrastructure

**Kind:** design

**Guidance:** Build test helpers to load Cu111_CO reference data.

Reuse chemrust-hamiltonian's:
- `CastepBinFile::read()` — reads `.castep_bin` for cell, density, eigenvalues
- `CheckFile::read()` — reads `.check` for wavefunctions, fine_grid
- `formatted::parse_den_fmt()` — reads `.den_fmt` → reference density (fine grid)
- `formatted::parse_pot_fmt()` — reads `.pot_fmt` → reference V_eff (fine grid)
- `ParsedotentialSet::from_dir()` — loads USP files

**Grid reordering (critical):**

| Source | Convention |
|--------|-----------|
| CASTEP `FieldMetadata.grid` | `[ngx, ngy, ngz]` |
| CASTEP `ElectronDensity.charge` (Array3 shape) | `(ngx, ngy, ngz)` |
| scf `GVectorGrid::grid()` | `[ngz, ngy, ngx]` |
| scf `Density` (Array3 shape, from `construct_density_gpu:177`) | `(ngx, ngy, ngz)` |

**Conversion rules:**
- `GVectorGrid::new([ngz, ngy, ngx], recip)` where `[ngx, ngy, ngz] = field_meta.grid`
- Density Array3: same shape convention — **no transpose needed**
- `WavefunctionCoeffs.grid` and `.check` `fine_grid`: reorder `[ngx, ngy, ngz]` → `[ngz, ngy, ngx]`

Verify grid shapes with runtime assertions at fixture load time.

**Cached struct** (`OnceLock` — stable since Rust 1.80, available in edition 2024):

```rust
pub struct Cu111CoFixture {
    pub bin: CastepBin,              // .castep_bin
    pub check: CastepBin,            // .check (155 MB, wavefunction + fine_grid)
    pub pot_fmt: Array3<f64>,        // .pot_fmt reference V_eff (fine grid)
    pub den_fmt: ElectronDensity,    // .den_fmt reference density (fine grid)
    pub bands_eigenvalues: Vec<f64>, // .bands eigenvalues in Hartree
    pub pots: PseudopotentialSet,
}
```

**`build_scf_state` helper:**
1. Cell from `fx.bin.cell`, pots from `fx.pots`
2. Wave grid: reorder `field_meta.grid` → `GVectorGrid::new([ngz, ngy, ngx], recip)`
3. Fine grid: reorder `fx.check.fine_grid.unwrap()` → `GVectorGrid::new(...)`
4. Density: wrap `fx.bin.density.charge.as_array().clone()` — no transpose
5. Wavefunctions from `fx.check.wavefunction.unwrap().kpt_data[0]` (Gamma-only):
   - `pw_coords = kpt_block.pw_grid_coord.clone()`
   - `pw_fft_indices = pw_coords_to_fft_indices(&pw_coords, &wave_grid)`
   - `psi = WavefunctionSet::<ColumnDistributed>::new(kpt_block.bands.concat(), n_bands, nplw)`
6. Smearing: Gaussian, 0.1 eV width (CASTEP default, `.castep` line 154)
7. `ScfIteration::builder()...build()`

**Visibility:** Change `pw_coords_to_fft_indices` from `pub(crate)` to
`#[doc(hidden)] pub` in `src/scf.rs:917`; re-export from `src/lib.rs`.

**Constants:**
- Fixture dir: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
  (overridable via `CASTEP_FIXTURE_DIR` env var)
- Pseudopotential dir: `/export/Potentials/` (overridable via `CASTEP_POTENTIAL_DIR`)
- Reference energy: `-24110.96665069 eV` (Cu111_CO.castep line 326, 18 ions)
- Tolerance: 2e-4 eV total

**Files:** `tests/fixtures/mod.rs`, `tests/fixtures/cu111_co.rs`,
`src/scf.rs` (visibility), `src/lib.rs` (re-export)

**Success Criteria:**
- Parse `.castep_bin` + `.check` → cell, density, ψ, eigenvalues, fine_grid
- Parse `.pot_fmt` → reference V_eff array on fine grid
- Parse `.den_fmt` → reference density on fine grid
- Parse `.bands` → 160 eigenvalues in Hartree
- Load USP pseudopotentials for Cu, C, O
- `build_scf_state` produces valid `ScfIteration<NonSpin, Initialized, MixingOff>`
- Grid shape assertions pass at load time

---

#### F-2: Per-step comparison test

**Kind:** direct

**Guidance:** Integration test (`#[ignore]` — requires GPU) running three
sub-tests, each comparing one SCF step against CASTEP reference.

GPU detection: `cudarc::driver::CudaContext::new(0).is_ok()` inside
`catch_unwind` → skip if unavailable.

**Test 2a — `compare_v_eff_against_pot_fmt`:**
1. `state.build_v_eff()` → extract V_eff as `&Array3<f64>`
2. Reference: `fx.pot_fmt` (fine grid)
3. Assert `max_abs_diff < 1e-3` Hartree, `rms_diff < 1e-4` Hartree
   (pseudopotential interpolation differences, same physics)

**Test 2b — `compare_eigenvalues_against_bands`:**
1. `state.build_v_eff()?.diagonalize(8)`
2. Compare `diag_state.eigenvalues` vs `fx.bands_eigenvalues` (both Hartree)
3. Assert RMS diff < 5e-3 Hartree (~0.14 eV)
4. Assert fewer than 5% of bands exceed 0.1 eV individual diff

**Test 2c — `compare_density_against_castep_bin`:**
1. `state.build_v_eff()?.diagonalize(8)?.construct_density_off()`
2. Compare computed density vs `fx.bin.density.charge.as_array()` (wave grid)
3. Assert RMS diff < 1e-5 e/Bohr³
4. Assert |∫ρ_computed − ∫ρ_ref| < 0.1 e

**Files:** `tests/ca_step_validation.rs`

**Success Criteria:**
- All three sub-tests pass on GPU-equipped machine
- Tolerances calibrated against known CASTEP–chemrust differences

---

#### F-3: SCF convergence tests

**Kind:** direct

**Guidance:** Two sub-tests (`#[ignore]` — requires GPU).  The `.check`
caveat means we cannot test convergence from CASTEP's actual starting point,
so we test both fixed-point stability and perturbation recovery.

**Test 3a — `fixed_point_matches_castep_energy`:**
Load converged state, run `run_scf_with_energy(state, ndeg=8, tol=1e-8)`.
Assert `|result.total_energy * HARTREE_TO_EV - (-24110.96665069)| < 2e-4`.
Converges in 1–3 iterations (state is already at the fixed point).

**Test 3b — `perturbation_recovers_castep_energy`:**
1. Load converged state
2. Apply 5% multiplicative noise to density: `ρ(r) *= 1.0 ± 0.05`
3. Renormalize to preserve total charge
4. Run `run_scf_with_energy` → assert energy recovers within 2e-4 eV
5. Assert iteration count > 1 (noise pushed off fixed point)

The noise amplitude (5%) is small enough that DIIS can recover but large enough
to test real convergence dynamics.  CASTEP's Pulay mixing amplitude of 0.5
suggests the SCF can handle ~50% perturbations.

**Files:** `tests/ca_scf_convergence.rs`

**Success Criteria:**
- Fixed-point: energy matches CASTEP reference within 2e-4 eV
- Perturbation: energy recovers within 2e-4 eV, more than 1 iteration
- Both tests pass on GPU-equipped machine

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
- **Group F source audit (2026-05-20):** Read `Cu111_CO.castep` lines 286–324
  for actual SCF output format (not the generic format assumed before).  Key
  findings:
  - CASTEP's SCF loop took **33 iterations** to converge (not 50), from initial
    energy −21899.39 eV to final free energy −24111.22 eV.
  - Columns: iteration, free energy E−TS (eV), Fermi energy (eV), energy gain
    per atom (eV), cumulative wall time (sec).
  - CASTEP SCF free energy differs from `Final energy, E` by ~0.25 eV (TS term
    from Gaussian smearing).  We output total electronic energy, not free energy.
  - `.check` file stores **converged** state only — loading it tests the fixed
    point, not convergence dynamics.  Perturbation-recovery test (F-3b) is the
    practical workaround until chemrust has its own initial-SCF.
  - Grid conventions: `FieldMetadata.grid` is `[ngx, ngy, ngz]`; `GVectorGrid`
    expects `[ngz, ngy, ngx]`.  Density Array3 shapes match (both `(ngx, ngy, ngz)`)
    so no transpose needed for density — only for grid construction.
  - `compute_occupations` computed chemical potential μ (Fermi energy) but
    discarded it.  Added `ChemicalPotential` newtype + `fermi_energy` field to
    `ScfIteration` so the per-iteration output can display it.
  - Added `Occupations(Vec<f64>)` and `ChemicalPotential(f64)` newtypes to
    avoid bare tuples — consistent with project's granular newtype convention.
  - Added `EV_TO_HARTREE` / `HARTREE_TO_EV` constants to `src/energy.rs` as
    single source of truth, re-exported from `lib.rs`.

## Verification

```bash
# From nix develop shell:
cargo check                              # Must succeed
cargo clippy --workspace -- -D warnings  # Must succeed (0 warnings)
cargo test                               # GPU tests require CUDA-capable GPU

# Specific validation (GPU required, per-iteration output enabled):
RUST_LOG=info cargo test compare_v_eff_against_pot_fmt -- --ignored
RUST_LOG=info cargo test compare_eigenvalues_against_bands -- --ignored
RUST_LOG=info cargo test compare_density_against_castep_bin -- --ignored
RUST_LOG=info cargo test fixed_point_matches_castep_energy -- --ignored
RUST_LOG=info cargo test perturbation_recovers_castep_energy -- --ignored

# All GPU tests with energy decomposition:
RUST_LOG=debug cargo test -- --ignored
```
