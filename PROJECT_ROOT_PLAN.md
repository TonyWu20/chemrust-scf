# Assessment: GPU-native plane-wave DFT in Rust, referencing CASTEP infrastructure

## Context

The user is deep in a GPU-resident SCF port of CASTEP (Fortran+CUDA), hitting systemic correctness bugs: sync/layout errors that the Fortran type system cannot catch. Separately, they have built `TonyWu20/chemrust-hamiltonian`, a Rust crate that already reconstructs V_eff and H(k) from electron density + USP pseudopotentials + GGA-PBE XC, validated against CASTEP reference eigenvalues. The question: **is a complete GPU-native plane-wave DFT rewrite in Rust defensible, and how should it be assessed?**

## New information incorporated

1. **Abinit GPU port paper** (Lygiatsika et al., arXiv:2604.11139): OpenMP-target-offload GPU port of Abinit, using cuFFT+cuBLAS+cuSOLVER exclusively (no custom CUDA kernels). Key findings: Chebyshev filtering >> LOBPCG on GPU (k× higher arithmetic intensity per MPI transposition), Rayleigh-Ritz is the memory-bound bottleneck, energy savings on NVIDIA are excellent.
2. **NVlabs/cuda-oxide** (1,906 stars, active development): NVIDIA's official rustc backend that compiles `#[kernel]` Rust functions directly to PTX. Single-source compilation, generic kernels with closures. Currently alpha/experimental.
3. **chemrust-hamiltonian** already implements: V_eff assembly (Poisson V_H, PBE XC, V_ion, screened V_NL), H_loc|ψ> apply via dual-space approach, full H(k) expectation-value pipeline validated against CASTEP.

## Ground truths (irreducible)

| # | Truth | Why it matters |
|---|-------|---------------|
| 1 | **FFT dominates compute** (50-70% of runtime in Hamiltonian apply) | Solved by cuFFT/VkFFT FFI; Rust doesn't need to do anything special here |
| 2 | **Rayleigh-Ritz is the scaling bottleneck** — `hegvd` is memory-bound, gets *worse* with more GPU nodes, tanks on AMD | This is a vendor-library problem; Rust can't magic cuSOLVER/rocSOLVER faster |
| 3 | **Chebyshev filtering is the correct GPU eigensolver** — maximizes compute-bound work between MPI transpositions; LOBPCG's per-block orthogonalization is communication-bound | Algorithm choice, not language choice. But Rust's type system can enforce the correct API contract (e.g., row-vs-column distributed wavefunctions) |
| 4 | **You don't need custom CUDA kernels for MVP** — Abinit uses zero custom kernels; everything is vendor libraries | cuda-oxide is nice-to-have, not a blocker |
| 5 | **chemrust-hamiltonian already has the hard part** — validated H(k) construction, pseudopotential machinery, XC, V_eff assembly | The gap to runnable SCF is ~4-5K lines, not 50K |
| 6 | **GPU memory is the hard constraint** — VRAM (~80GB H100) vs system RAM (~2TB) | True GPU-resident only up to some system size; need an out-of-core path for large systems |
| 7 | **The user base is Fortran/Python-native** — adoption requires Python bindings or a compelling performance story | `maturin` + `pyo3` make Python bindings from Rust straightforward |

## Assessment by axis

### Axis A: Type-system value (the primary motivation)

**Assessment: Strong positive.**

The user's core motivation — using Rust's type system to prevent the class of bugs currently plaguing the CASTEP GPU port — is well-founded. Concrete examples of what Rust can enforce at compile time that Fortran cannot:

| Fortran pain point | Rust solution |
|---|---|
| Mixing row-distributed vs column-distributed wavefunction arrays | `WavefunctionRowDistributed` and `WavefunctionColumnDistributed` as distinct types; transposition consumes one and returns the other |
| Passing the wrong 3D field (e.g., HartreePotential where Density is expected) | `real_space_field!` macro in `chemrust-hamiltonian` already generates opaque newtypes — compiler rejects mismatches |
| MPI buffer size errors (n_bands vs n_plane_waves swapped) | Type-level encoding of array dimensions or runtime-checked newtypes |
| Layout-sensitive operations (contiguous-by-band vs contiguous-by-PW) | Types carry layout information; operations that require a specific layout only accept that type |
| Missing GPU→CPU sync before CPU reads GPU data | `GpuResident<T>` vs `HostAccessible<T>` wrapper types; accessing host data requires explicit `.sync_to_host()` that returns the right type |

The `chemrust-hamiltonian` crate already demonstrates this pattern with its `real_space_field!` macro and sealed `SpinPolicy` trait.

### Axis B: What's already built vs. what's missing

**Assessment: The hard part is done.**

Already in `chemrust-hamiltonian`:
- USP pseudopotential parser and PseudopotentialSet
- V_H (Poisson solver in G-space)
- V_ion reconstruction
- PBE XC (GGA)
- V_NL with D-matrix screening (3 levels: None/Wave/Fine)
- β_phi projection onto plane-wave basis
- `apply_local_hamiltonian`: dual-space T+V_eff application with correct FFT normalization
- `nlpot_expectation` and `nlpot_apply` for V_NL
- Full H(k) expectation-value pipeline validated against CASTEP reference
- Type-safe V_eff assembly (`VEffBuilder`)
- CASTEP binary/formatted file I/O

Missing for runnable SCF (~4-5K lines):
- **Chebyshev filtering eigensolver** (~2K lines): subspace iteration with polynomial filtering, Rayleigh-Ritz procedure, eigenvector rotation
- **SCF cycle** (~500 lines): density mixing (linear + Pulay), convergence check, outer loop
- **Electron density from wavefunctions** (~300 lines): Σ_occ |ψ_i(r)|² with occupation smearing (Fermi-Dirac)
- **GPU FFT backend** (swap `rustfft` for VkFFT/cuFFT via FFI)
- **Forces** (~1K lines): Hellmann-Feynman + Pulay corrections

### Axis C: Algorithm choice (lesson from Abinit paper)

**Assessment: Chebyshev filtering is the right choice.**

The Abinit paper provides empirical evidence:
- Chebfi with `ndeg=8`: 198.8s wall time, 15 SCF iterations to convergence
- LOBPCG with `nline=4`: 236.3s wall time, 14 SCF iterations (but worse eigenvalue convergence per iteration)
- Chebfi achieves k× higher arithmetic intensity because it only does MPI transpose once per k Hamiltonian applications
- LOBPCG does MPI transpose k times per block (the "Comm-H-Comm" pattern per block)
- The Rayleigh-Ritz `hegvd` call is memory-bound on GPU — minimizing its frequency is critical

For Rust implementation, Chebyshev filtering has a cleaner API surface:
```rust
fn chebyshev_filtering(
    ham: &impl HamiltonianApply,  // H|ψ> and S|ψ>
    psi: &mut WavefunctionColumn, // n_pw × n_bands, column-distributed
    ndeg: usize,                   // polynomial degree
) -> Result<FilteredSubspace, Error>
```
vs. LOBPCG's more complex block management, orthogonalization, and CG line search.

### Axis D: cuda-oxide (Rust GPU kernel capability)

**Assessment: Promising but not needed for MVP.**

cuda-oxide is a genuine advance — it compiles safe(ish) Rust `#[kernel]` functions to PTX through a full Rust-native pipeline (Rust MIR → Pliron IR → LLVM IR → PTX). Features: generic kernels, closure capture, type-safe shared memory, atomics, barriers, TMA.

However, the Abinit paper shows **zero custom kernels are needed** for GPU DFT — cuFFT + cuBLAS + cuSOLVER cover the entire SCF cycle. cuda-oxide becomes valuable for:
- Fused Chebyshev recurrence kernels (reduce launch overhead)
- Custom density-accumulation kernels (|ψ|² weighted by occupations)
- Strided batched operations for nonlocal projectors

None of these are MVP blockers.

cuda-oxide is alpha — expect bugs, API breakage, and missing features. NVIDIA Research maintains it, not NVIDIA Product. By the time a Rust DFT code reaches maturity (1-2 years), cuda-oxide will likely be stable.

### Axis E: Scope discipline

**Assessment: Define the MVP boundary explicitly.**

MVP scope (defensible 1-person, 3-6 month project):
- Γ-point only (no k-points initially)
- USPP (not PAW initially — but the infrastructure supports it)
- GGA-PBE only
- Non-spin-polarized + spin-collinear (already supported in `SpinPolicy`)
- No DFT+U, no hybrids
- Fixed cell (no variable-cell relaxation initially)
- Forces yes (Hellmann-Feynman is straightforward), stress maybe later

Out of scope for MVP:
- k-points (requires complex wavefunctions — `chemrust-hamiltonian` already handles this)
- PAW (requires overlap operator S ≠ I — more complex eigensolver)
- Phonons, NMR, EELS, TDDFT, constraints
- Non-collinear spin, spin-orbit coupling
- Hybrid functionals (requires exact exchange — Fock operator is expensive)

### Axis F: Validation strategy

**Assessment: Already partially in place.**

`chemrust-hamiltonian` already validates against CASTEP reference eigenvalues. For full SCF validation:
1. Run CASTEP on a small test system (Cu bulk, Si, GaAs), save the `.castep_bin` file
2. Run the Rust SCF with the same input parameters
3. Compare total energy, eigenvalues, forces to CASTEP within numerical tolerance
4. Accept that GPU FP associativity will produce slightly different results (this is normal, and the Abinit paper doesn't even address it — they compare order-of-magnitude speedups, not bitwise accuracy)

## Conclusion

**This is defensible.** The type-system motivation is genuine — the bugs you're hitting in CASTEP's GPU port (layout mismatches, missing syncs, wrong array ordering) are exactly the class of bugs Rust's type system prevents at compile time. The hard part (validated Hamiltonian construction) is already done in `chemrust-hamiltonian`. The gap to a runnable SCF is ~4-5K lines.

**Key risks acknowledged:**
- The Abinit paper shows RR `hegvd` is the bottleneck — Rust can't fix AMD's poor rocSOLVER performance
- Feature parity with CASTEP is a 50-person-year effort; don't aim for it
- cuda-oxide is alpha; rely on vendor library FFI for MVP, adopt cuda-oxide for custom kernels later
- Validation burden is real — the CASTEP test suite has thousands of inputs; test strategically

**Recommended path:**
1. CPU-first SCF prototype in Rust using existing `rustfft` + Chebyshev filtering
2. Validate total energy vs CASTEP on 3-5 small test systems
3. Swap FFT to VkFFT/cuFFT for GPU
4. GPU-resident wavefunction during eigensolver loop
5. Forces (Hellmann-Feynman is easy; Pulay corrections need nonlocal derivatives)

---

# Design: Type-Safe SCF Backbone

## Strategy: backbone-first, not piece-by-piece

Bottom-up (write pieces then wire together) inevitably hits integration bugs — layouts don't match, data is in wrong state, sync not called. Every one of the CASTEP GPU bugs in memory is an *interface* violation, not an algorithm error.

Top-down (backbone first) defines the contracts via Rust's type system, then fills the holes. The compiler checks the wiring before any physics code runs. Each transition function can be stubbed with `todo!()` and implemented independently, with known invariants at every boundary.

The SCF cycle has a fixed topology:

```
ρ (wave grid) → V_eff[ρ] (fine grid) → diagonalize H[V_eff] → ψ, ε
→ ρ_new from ψ → mix(ρ_new, history) → check convergence → next ρ
```

## Type-safety layers

Three independent axes of type distinction, each encoding an invariant that CASTEP tracks only in comments:

### Layer 1: Grid level

```rust
// Opaque grid wrappers — prevents passing wave-grid data where fine-grid is expected
pub struct WaveGridArray(Array3<f64>);
pub struct FineGridArray(Array3<f64>);

// Density lives on the wave grid
pub struct Density(pub WaveGridArray);

// Upsampled density for XC evaluation
pub struct DensityUpsampled(pub FineGridArray);

// Effective potential lives on the fine grid
pub struct EffectivePotential(pub FineGridArray);
```

**Existing:** `chemrust-hamiltonion-core/src/types.rs` already has `real_space_field!` macro generating `Density`, `HartreePotential`, `EffectivePotential`, etc. as opaque newtypes. The `WaveGridArray`/`FineGridArray` distinction is the missing piece — currently `Density` wraps `Array3<f64>` directly with no grid-level information.

**CASTEP bugs this prevents:** The `gpu-resident-density-bug` (memory file) — density computation read G-vector values instead of real-space data because the grid level (who holds which array, on which grid) was unclear.

### Layer 2: Device location

```rust
pub struct Gpu<T>(pub T);
pub struct Cpu<T>(pub T);

impl<T: Clone> Gpu<Array3<T>> {
    pub fn sync_to_host(&self, stream: &CudaStream) -> Cpu<Array3<T>> {
        // D2H transfer, returns Cpu<Array3<T>>
    }
}

impl<T: Clone> Cpu<Array3<T>> {
    pub fn sync_to_device(&self, stream: &CudaStream) -> Gpu<Array3<T>> {
        // H2D transfer, returns Gpu<Array3<T>>
    }
}
```

**CASTEP bugs this prevents:** Missing D2H sync before CPU code reads GPU-resident data (the `gpu-resident-density-bug` root cause). If density construction takes `Cpu<WaveGridArray>`, the compiler enforces the sync.

### Layer 3: Wavefunction layout (MPI distribution)

```rust
pub struct RowDistributed;
pub struct ColumnDistributed;

pub struct WavefunctionSet<Layout = ColumnDistributed> {
    data: Vec<Complex64>,
    n_bands: usize,
    n_pw_local: usize,   // plane waves per MPI rank
    _marker: PhantomData<Layout>,
}
```

| Operation | Requires | Returns |
|-----------|----------|---------|
| Hamiltonian apply (FFT+multiply) | `ColumnDistributed` | `ColumnDistributed` |
| Rayleigh-Ritz (subspace solve) | `RowDistributed` | `RowDistributed` |
| Nonlocal projector apply | `ColumnDistributed` | `ColumnDistributed` |
| Transpose: all-to-all MPI | `RowDistributed` | `ColumnDistributed` (or vice versa) |

**CASTEP bugs this prevents:** Operating on row-distributed data in a function that assumes column-distributed layout. In CASTEP this is a comment convention (`! This routine assumes wavefunction is in the waveform layout`); in Rust it's a compile error.

## The SCF state machine

```rust
// ── Phase markers (zero-sized, compile-time only) ──

pub struct Initialized;
pub struct VEffBuilt;
pub struct WavefunctionsUpdated;
pub struct DensityUpdated;
pub struct Mixed;
pub struct Converged;

// ── The SCF iteration struct ──

pub struct ScfIteration<State = Initialized> {
    // ─── Immutable across the whole SCF ───
    cell: CellGeometry,
    pots: PseudopotentialSet,
    wave_grid: GVectorGrid,
    fine_grid: GVectorGrid,
    k_point: KPoint,
    smearing: SmearingParams,

    // ─── Fields guarded by phase (Some in some phases, None in others) ───
    density: Density,
    psi: WavefunctionSet,
    eigenvalues: Vec<f64>,
    v_eff: Option<EffectivePotential>,
    history: DensityHistory,
    previous_density: Density,

    _phase: PhantomData<State>,
}
```

### Transition 1: Initialized → VEffBuilt

```rust
impl ScfIteration<Initialized> {
    pub fn build_v_eff(self) -> Result<ScfIteration<VEffBuilt>, Error> {
        // Upsample density from wave grid to fine grid
        let rho_fine = upsample_density_to_fine_grid(
            &self.density, &self.wave_grid, &self.fine_grid,
        )?;
        // Assemble V_eff using existing VEffBuilder from chemrust-hamiltonian
        let v_eff = VEffBuilder::<NonSpin>::new(&self.cell, &self.pots, &self.fine_grid)
            .with_density(Density::from_inner(rho_fine), None)?
            .assemble()?;
        Ok(ScfIteration {
            v_eff: Some(v_eff),
            density: self.density,
            psi: self.psi,
            // ...
            _phase: PhantomData,
        })
    }
}
```

**Reuses:** `chemrust-hamiltonian-core/src/band_structure.rs` `VEffBuilder`, `assemble`, `upsample_density_to_fine_grid` from `fft.rs`.

### Transition 2: VEffBuilt → WavefunctionsUpdated

```rust
impl ScfIteration<VEffBuilt> {
    pub fn diagonalize(self, ndeg: usize) -> Result<ScfIteration<WavefunctionsUpdated>, Error> {
        // v_eff is guaranteed Some — the type state enforces this
        let v_eff = self.v_eff.as_ref().expect("VEffBuilt guarantees v_eff is Some");

        // 1. MPI transpose: RowDistributed → ColumnDistributed
        let psi_column = self.psi.transpose_to_column();

        // 2. Chebyshev filtering: k× H|ψ> applications in column layout
        //    Each iteration: H_loc|ψ> (FFT roundtrip) + V_NL|ψ> (β-projector gemm)
        let psi_filtered = chebyshev_filtering(
            &psi_column, v_eff, &self.pots, &self.cell, ndeg,
        )?;

        // 3. MPI transpose: ColumnDistributed → RowDistributed
        let psi_row = psi_filtered.transpose_to_row();

        // 4. Rayleigh-Ritz on row-distributed ψ
        //    H_sub = ψ† H ψ, S_sub = ψ† S ψ (for USPP S=I, paper step)
        //    Solve H_sub X = ε S_sub X (hegvd-equivalent)
        //    Rotate: ψ ← X ψ
        let (psi_row, eigenvalues) = rayleigh_ritz(&psi_row, v_eff, &self.pots, &self.cell)?;

        Ok(ScfIteration {
            psi: psi_row,
            eigenvalues,
            v_eff: self.v_eff,
            density: self.density,
            // ...
            _phase: PhantomData,
        })
    }
}
```

**New code needed here** (~2K lines): `chebyshev_filtering` and `rayleigh_ritz`. Everything else (H_loc apply, V_NL apply, FFT) already exists in `chemrust-hamiltonian`.

### Transition 3: WavefunctionsUpdated → DensityUpdated

```rust
impl ScfIteration<WavefunctionsUpdated> {
    pub fn construct_density(self) -> Result<ScfIteration<DensityUpdated>, Error> {
        // Σ_{i} f(ε_i) |ψ_i(r)|² on the wave grid
        // f(ε_i) = occupations from Fermi-Dirac smearing
        let rho_new = build_density_from_wavefunctions(
            &self.psi, &self.eigenvalues, &self.wave_grid,
            self.smearing, self.k_point,
        )?;
        Ok(ScfIteration {
            density: rho_new,    // ← replaces old density
            // ...
            _phase: PhantomData,
        })
    }
}
```

**New code needed** (~300 lines): `build_density_from_wavefunctions` — scatter PW coeffs → IFFT → |ψ|² → weighted sum → density on wave grid. Plus occupation number determination (Fermi-Dirac smearing with chemical potential search).

### Transition 4: DensityUpdated → Mixed

```rust
impl ScfIteration<DensityUpdated> {
    pub fn mix(self) -> ScfIteration<Mixed> {
        let (mixed, prev) = self.history.mix(self.density, self.mixing_scheme);
        ScfIteration {
            density: mixed,
            previous_density: prev,
            // ...
            _phase: PhantomData,
        }
    }
}
```

**New code needed** (~300 lines): `DensityHistory::mix` — simple linear mixing for MVP (`ρ_mix = β ρ_new + (1-β) ρ_old`), Pulay/DIIS later.

### Transition 5: Mixed → Converged | Initialized

```rust
impl ScfIteration<Mixed> {
    pub fn check(self, tol: f64) -> Result<ScfIteration<Converged>, ScfIteration<Initialized>> {
        let diff = max_abs_diff(&self.density, &self.previous_density);
        if diff < tol && self.history.iterations >= 2 {
            Ok(ScfIteration {
                _phase: PhantomData,
                // ...
            })
        } else {
            // Rewind — discard V_eff, go back to Initialized
            Err(ScfIteration {
                v_eff: None,            // ← must rebuild
                _phase: PhantomData,
                // ...
            })
        }
    }
}
```

**The return type is the key invariant:** `Result<Converged, Initialized>`. Convergence returns `Converged` (terminal). Non-convergence returns `Initialized` — V_eff is dropped, the caller *must* call `build_v_eff` again. The compiler enforces this.

### The SCF loop (compile-time checked orchestration)

```rust
pub fn run_scf(mut state: ScfIteration<Initialized>, tol: f64) -> Result<FinalResult, Error> {
    loop {
        state = state.build_v_eff()?;           // Initialized → VEffBuilt
        state = state.diagonalize(ndeg)?;        // VEffBuilt → WavefunctionsUpdated
        state = state.construct_density()?;      // WavefunctionsUpdated → DensityUpdated
        state = state.mix();                     // DensityUpdated → Mixed

        match state.check(tol)? {                // Mixed → Converged | Initialized
            Ok(done) => return Ok(done.finalize()),
            Err(next) => state = next,           // ← type resets to Initialized
        }
    }
}
```

**What the compiler guarantees at each step:**
- Cannot call `diagonalize()` before `build_v_eff()` — `VEffBuilt` marker required
- Cannot call `construct_density()` before `diagonalize()` — `WavefunctionsUpdated` required
- Cannot call `mix()` before `construct_density()` — `DensityUpdated` required
- Cannot re-use stale V_eff after density changes — `check()` returns `Initialized` which must restart from `build_v_eff`
- No `unwrap()` on `v_eff` outside of `VEffBuilt` phase — the type state eliminates the panic path

## Full GPU-resident SCF backbone

### How the three type axes compose

`Gpu<T>` / `Cpu<T>` wraps any type. It composes with grid-level types and layout types:

```rust
// A GPU-resident density on the wave grid:
Gpu<Density>              // expands to Gpu<WaveGridArray(Array3<f64>)>

// A GPU-resident V_eff on the fine grid:
Gpu<EffectivePotential>   // expands to Gpu<FineGridArray(Array3<f64>)>

// Wavefunction on GPU, row-distributed (for Rayleigh-Ritz):
Gpu<WavefunctionSet<RowDistributed>>

// Same wavefunction, column-distributed (for Hamiltonian apply):
Gpu<WavefunctionSet<ColumnDistributed>>
```

The compiler rejects operations that mix these. `solve_poisson(Gpu<FineGridArray>)` won't accept `Cpu<WaveGridArray>` — grid level AND device location must both match.

### Data residency per phase

**The density never leaves GPU.**

```
┌────────────────────────────────────────────────────────────────────────────┐
│ GPU                   THE DENSITY STAYS ON GPU. PERIOD.                    │
│                                                                      │
│  Phase: Initialized                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐     │
│  │ ρ (wave grid)    │  │ ψ: RowDistributed (bands×pw, for RR) │     │
│  └────────┬─────────┘  └──────────────────┬───────────────────┘     │
│           │                               │                          │
│  ┌────────▼───────────────────────────────▼──────────────────┐      │
│  │ build_v_eff (Initialized → VEffBuilt)                     │      │
│  │  1. upsample ρ: wave grid → fine grid (FFT on GPU)        │      │
│  │  2. solve_poisson: V_H on fine grid (GPU)                 │      │
│  │  3. compute_pbe_xc: V_xc on fine grid (GPU, or libxc FFI) │      │
│  │  4. reconstruct_v_ion: V_ion on fine grid (GPU)          │      │
│  │  5. reconstruct_rho_core: NLCC if needed (GPU)            │      │
│  │  6. sum: V_eff = V_H + V_xc + V_ion                     │      │
│  │  → V_eff on fine grid                                     │      │
│  └──────────────────────────────┬───────────────────────────┘      │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────┐      │
│  │ diagonalize (VEffBuilt → WavefunctionsUpdated)           │      │
│  │                                                          │      │
│  │  ╔══════════════════════════════════════════════════════╗│      │
│  │  ║ ψ STAYS ON GPU FOR ENTIRE EIGENSOLVER LOOP          ║│      │
│  │  ║ (matches Abinit sec 2.2 "GPU-resident wave          ║│      │
│  │  ║  function calculations")                             ║│      │
│  │  ╚══════════════════════════════════════════════════════╝│      │
│  │                                                          │      │
│  │  1. all-to-all transpose: RowDist → ColumnDist (GPU MPI) │      │
│  │  2. Chebyshev filtering (k× iterations):                 │      │
│  │     for i in 0..ndeg:                                    │      │
│  │       H_loc|ψ>: FFT roundtrip (cuFFT/VkFFT on GPU)       │      │
│  │       V_NL|ψ>: β-projector gemm (cuBLAS on GPU)          │      │
│  │       Chebyshev recurrence: X ← aX + Y (fused kernel)    │      │
│  │  3. all-to-all transpose: ColumnDist → RowDist (GPU MPI) │      │
│  │                                                          │      │
│  │  4. Rayleigh-Ritz (via cuSOLVER — zero PCIe):           │      │
│  │     ┌─────────────────────────────────────────┐          │      │
│  │     │ H_sub = ψ† H ψ  (GPU)                    │          │      │
│  │     │ S_sub = ψ† S ψ  (GPU)                    │          │      │
│  │     │ cuSOLVER ZHEGVD: H_sub·X = ε·S_sub·X     │          │      │
│  │     │   → X, ε on GPU (no D2H/H2D)              │          │      │
│  │     │ ψ_new = X·ψ  (cuBLAS gemm on GPU)         │          │      │
│  │     │   → ε D2H (optional, n_bands f64s)        │          │      │
│  │     └─────────────────────────────────────────┘          │      │
│  │                                                          │      │
│  │  → ψ on GPU, RowDistributed; ε on CPU (tiny)             │      │
│  └──────────────────────────────┬───────────────────────────┘      │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────┐      │
│  │ construct_density (WavefunctionsUpdated → DensityUpdated)│      │
│  │                                                          │      │
│  │  1. all-to-all transpose: RowDist → ColumnDist (GPU MPI) │      │
│  │  2. bulk batched FFT: cufftPlanMany(N_bands, grid)       │      │
│  │     ┌───────────────────────────────────────────┐        │      │
│  │     │ Single kernel launch:                      │        │      │
│  │     │   for all bands simultaneously:            │        │      │
│  │     │     scatter c_G → FFT grid (strided batch) │        │      │
│  │     │     batched IFFT → ψ_i(r) for all bands    │        │      │
│  │     │     batched |ψ_i(r)|²                      │        │      │
│  │     │     weighted sum: Σ f(ε_i) × |ψ_i(r)|²     │        │      │
│  │     │     → ρ_new(r) single real-space array     │        │      │
│  │     └───────────────────────────────────────────┘        │      │
│  │  3. symmetrize ρ (GPU, if crystal symmetry enables it)   │      │
│  │                                                          │      │
│  │  → ρ_new on GPU, wave grid                               │      │
│  └──────────────────────────────┬───────────────────────────┘      │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────┐      │
│  │ mix (DensityUpdated → Mixed)                             │      │
│  │                                                          │      │
│  │  ─── CPU solves small Pulay system ───                   │      │
│  │  The Pulay/DIIS linear system is history_size ×           │      │
│  │  history_size (typically 5-8), solved on CPU:             │      │
│  │    ┌────────────────────────────────────┐                 │      │
│  │    │ build metric tensor M_ij on GPU    │                 │      │
│  │    │   → D2H: M (8×8 = 64 doubles)      │                 │      │
│  │    │ solve M·c = b on CPU → coeffs c_i  │                 │      │
│  │    │   → H2D: c (8 doubles)             │                 │      │
│  │    │ ρ_mix = Σ c_i ρ_i (AXPYs on GPU)   │                 │      │
│  │    └────────────────────────────────────┘                 │      │
│  │  (Linear mixing: even simpler — one GPU AXPY)             │      │
│  │                                                          │      │
│  │  → ρ_mix on GPU, wave grid                                │      │
│  └──────────────────────────────┬───────────────────────────┘      │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────┐      │
│  │ check (Mixed → Converged | Initialized)                  │      │
│  │                                                          │      │
│  │  GPU reduction: max|ρ_mix - ρ_old|                       │      │
│  │    → single f64 crosses PCIe                             │      │
│  │                                                          │      │
│  │  diff < tol?                                              │      │
│  │    → yes: Converged (done, D2H ρ for final output)       │      │
│  │    → no:  Initialized (ρ stays on GPU, V_eff = None)     │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Transfer accounting per SCF iteration (with cuSOLVER)

| Transfer | Direction | Size | When |
|----------|-----------|------|------|
| ~~H_sub, S_sub~~ | ~~GPU→CPU~~ | **0 — cuSOLVER on GPU** | Rayleigh-Ritz |
| ~~X (eigenvectors)~~ | ~~CPU→GPU~~ | **0 — cuSOLVER on GPU** | Rayleigh-Ritz rotation |
| Eigenvalues (optional) | GPU→CPU | n_bands × 8 bytes | Only if CASTEP needs them |
| M_ij (Pulay metric) | GPU→CPU | history_size² × 8 bytes | Density mixing (~64 doubles for history=8) |
| c_i (Pulay coefficients) | CPU→GPU | history_size × 8 bytes | Density mixing (~8 doubles) |
| diff scalar | GPU→CPU | 1 × f64 (8 bytes) | Convergence check |
| **ρ never leaves GPU** | — | **0 bytes** | — |

For a 1000-band system: total PCIe traffic **per SCF iteration** is ~8 KB (eigenvalues + mixing + convergence). **The largest array is O(history_size²) — nothing scales with grid or band count.**

### Batched density-construction FFT

The key to GPU-resident density: instead of per-band FFT (which would need D2H for layout reorganization, as Abinit does), use a single bulk batched FFT:

```rust
// Pseudocode — cufftPlanMany equivalent
// Input: psi_col is Gpu<WavefunctionSet<ColumnDistributed>>
//   Layout: each band's n_pw coefficients are contiguous
// Output: Gpu<Density> — single real-space array, wave grid

fn build_density_batched_gpu(
    psi_col: &Gpu<WavefunctionSet<ColumnDistributed>>,
    eigenvalues: &[f64],
    wave_grid: &GVectorGrid,
    smearing: SmearingParams,
    stream: &CudaStream,
) -> Result<Gpu<Density>, Error> {
    let n_bands = psi_col.n_bands;
    let [nx, ny, nz] = wave_grid.grid();

    // 1. Compute occupations: f(ε_i) via Fermi-Dirac (CPU, tiny array)
    let occ = compute_occupations(eigenvalues, smearing);

    // 2. Single batched scatter + IFFT:
    //    cufftPlanMany: N_bands × (nx, ny, nz) 3D IFFTs, strided
    //    Each band's PW coefficients are contiguous in ColumnDistributed layout
    let psi_r_batched = cufft_exec_many_inverse(
        psi_col.data(), n_bands, [nx, ny, nz], stream,
    )?;
    // → N_bands real-space grids, shape (n_bands, nx, ny, nz)

    // 3. Fused kernel: |ψ|² + weighted accumulation
    //    Single GPU kernel: for each grid point, sum over bands with occupation weights
    let rho = accumulate_density_kernel(
        &psi_r_batched, &occ, n_bands, [nx, ny, nz], stream,
    )?;
    // → single (nx, ny, nz) real-space array

    // 4. Symmetrize (if crystal symmetry applies)
    let rho_sym = symmetrize_density_gpu(&rho, wave_grid, stream)?;

    Ok(Gpu(Density(rho_sym)))
}
```

This eliminates the Abinit-paper D2H for "waveform layout reorganization" — the `ColumnDistributed` layout already gives per-band contiguity, so the batched FFT operates directly on GPU-resident data.

### What the type system enforces for GPU residency

```rust
// ─── build_v_eff: GPU → GPU, no transfers ───
impl ScfIteration<Initialized> {
    pub fn build_v_eff(
        self, stream: &CudaStream,
    ) -> Result<ScfIteration<VEffBuilt>, Error> {
        let rho_fine = upsample_density_to_fine_grid_gpu(
            &self.density, &self.wave_grid, &self.fine_grid, stream,
        )?;  // → Gpu<FineGridArray>

        let v_eff = assemble_v_eff_gpu(
            &self.cell, &self.pots, &self.fine_grid, &rho_fine, stream,
        )?;  // → Gpu<EffectivePotential>

        Ok(ScfIteration {
            v_eff: Some(v_eff),     // Gpu<EffectivePotential>
            density: self.density,   // Gpu<Density>, preserved
            _phase: PhantomData, ..
        })
    }
}

// ─── diagonalize: ψ stays GPU, RR via cuSOLVER (zero PCIe) ───
impl ScfIteration<VEffBuilt> {
    pub fn diagonalize(
        self, ndeg: usize, handle: &DnHandle, stream: &CudaStream,
    ) -> Result<ScfIteration<WavefunctionsUpdated>, Error> {
        let v_eff = self.v_eff.as_ref().unwrap();

        let psi_col = gpu_transpose_to_column(&self.psi, stream);
        let psi_filtered = chebyshev_filtering_gpu(
            &psi_col, v_eff, &self.pots, ndeg, stream,
        )?;
        let psi_row = gpu_transpose_to_row(&psi_filtered, stream);

        // Rayleigh-Ritz entirely on GPU — zero PCIe traffic
        let (h_sub, s_sub) = build_subspace_matrices_gpu(&psi_row, v_eff, stream);

        // cuSOLVER ZHEGVD: H_sub·X = ε·S_sub·X, all on GPU
        let (x, eigenvalues_gpu) = zhegvd_gpu(handle, h_sub, s_sub, stream)?;

        // ψ_rotated = X·ψ — cuBLAS gemm on GPU
        let psi_rotated = rotate_wavefunctions_gpu(&psi_row, &x, stream);

        // Eigenvalues are needed by CASTEP — D2H once (n_bands × 8 bytes)
        let eigenvalues = eigenvalues_gpu.sync_to_host(stream);

        Ok(ScfIteration {
            psi: psi_rotated,        // Gpu<WavefunctionSet<RowDistributed>>
            eigenvalues,             // Vec<f64>, CPU (n_bands entries — tiny)
            v_eff: self.v_eff,       // Gpu<EffectivePotential>
            density: self.density,   // Gpu<Density>
            _phase: PhantomData, ..
        })
    }
}

// ─── construct_density: GPU batched FFT, ρ STAYS on GPU ───
impl ScfIteration<WavefunctionsUpdated> {
    pub fn construct_density(
        self, stream: &CudaStream,
    ) -> Result<ScfIteration<DensityUpdated>, Error> {
        let psi_col = gpu_transpose_to_column(&self.psi, stream);

        // Bulk batched FFT: all bands in one cuFFT call
        let rho_new = build_density_batched_gpu(
            &psi_col, &self.eigenvalues, &self.wave_grid,
            self.smearing, stream,
        )?;
        // → Gpu<Density> — STAYS ON GPU

        Ok(ScfIteration {
            density: rho_new,           // Gpu<Density>, not Cpu<Density>
            psi: self.psi,              // Gpu<WavefunctionSet>
            eigenvalues: self.eigenvalues, // Vec<f64>, CPU
            _phase: PhantomData, ..
        })
    }
}

// ─── mix: tiny Pulay system on CPU, AXPY combination on GPU ───
impl ScfIteration<DensityUpdated> {
    pub fn mix(
        self, stream: &CudaStream,
    ) -> ScfIteration<Mixed> {
        // Build metric tensor on GPU, D2H the tiny matrix
        let m_gpu = build_pulay_metric_gpu(&self.history, stream);
        let m_cpu = m_gpu.sync_to_host(stream);  // history_size² × 8 bytes (~64 doubles)

        // Solve on CPU, get mixing coefficients
        let coeffs_cpu = solve_pulay_coefficients(&m_cpu);
        let coeffs_gpu = coeffs_cpu.sync_to_device(stream); // history_size × 8 bytes

        // Mixing AXPYs on GPU — element-wise on wave-grid arrays
        let (rho_mixed, prev) = self.history.mix_gpu(
            self.density, &coeffs_gpu, stream,
        );
        // → both Gpu<Density>

        ScfIteration {
            density: rho_mixed,            // Gpu<Density>
            previous_density: prev,        // Gpu<Density>
            _phase: PhantomData, ..
        }
    }
}

// ─── check: GPU reduction, single f64 crosses PCIe ───
impl ScfIteration<Mixed> {
    pub fn check(
        self, tol: f64, stream: &CudaStream,
    ) -> Result<ScfIteration<Converged>, ScfIteration<Initialized>> {
        // GPU reduction: max|ρ_mix - ρ_old|
        let diff = max_abs_diff_gpu(&self.density, &self.previous_density, stream);
        // → single f64, 8 bytes over PCIe

        if diff < tol {
            Ok(ScfIteration { _phase: PhantomData, ..self })
        } else {
            // Loop back — ρ stays on GPU, V_eff is reset to None
            Err(ScfIteration {
                density: self.density,   // Gpu<Density>, never left GPU
                v_eff: None,             // ← must rebuild
                psi: self.psi,           // Gpu<WavefunctionSet>
                _phase: PhantomData, ..
            })
        }
    }
}
```

### The GPU-resident SCF loop

```rust
pub fn run_scf_gpu(
    mut state: ScfIteration<Initialized>,  // density and psi are Gpu<...>
    ndeg: usize, tol: f64,
) -> Result<FinalResult, Error> {
    let stream = CudaStream::new()?;

    loop {
        state = state.build_v_eff(&stream)?;         // GPU: upsample + assemble
        state = state.diagonalize(ndeg, &stream)?;    // GPU: eigensolver
                                                      //      → H_sub/S_sub/X cross PCIe (n_bands²)
        state = state.construct_density(&stream)?;    // GPU: batched FFT, ρ stays GPU
        state = state.mix(&stream);                   // GPU: Pulay AXPYs (coeffs from CPU)
        match state.check(tol, &stream)? {             // GPU: reduction, 1 f64 → CPU
            Ok(done) => return Ok(done.finalize()),
            Err(next) => state = next,                 // ρ stays GPU, V_eff=None
        }
    }
}
```

**ρ is never D2H'd or H2D'd during the SCF loop.** Only at final convergence is it transferred for output.

### What can never go wrong (compiler-enforced)

| Bug class | CASTEP symptom | Rust prevention |
|-----------|---------------|----------------|
| Read GPU data without D2H | Read G-vectors instead of real-space density | `mix()` takes `Gpu<Density>`, not `Cpu<Density>` — compiler verifies ρ never accidentally lands on CPU |
| Forget to rebuild V_eff after density changes | Using stale V_eff with new density | `check()` non-convergence path returns `Initialized` with `v_eff: None`; `diagonalize()` requires `VEffBuilt` |
| Apply H to row-distributed ψ | FFT on wrong axis, garbage eigenvalues | `apply_hamiltonian()` takes `WavefunctionSet<ColumnDistributed>` only |
| Call RR on column-distributed ψ | Band-band inner products across wrong dimension | `rayleigh_ritz()` takes `WavefunctionSet<RowDistributed>` only |
| Mix density from wrong grid | Upsample noise in V_eff | `Density(WaveGridArray)`, `EffectivePotential(FineGridArray)` — distinct types |
| Forget to symmetrize density | Broken forces | `build_density_batched_gpu` returns symmetrized result internally |
| D2H ρ then forget to H2D next iter | Stale density on GPU | Never happens: ρ never D2H'd in the loop at all |

### Alignment with Abinit paper, improved

| Abinit pattern (sec 2.2) | Abinit limitation | Rust realization (improved) |
|--------------------------|-------------------|---------------------------|
| "Single H2D of ψ at start of SCF" | — | `ScfIteration<Initialized>` holds `Gpu<WavefunctionSet>`, constructed once |
| "Keep ψ on GPU during diagonalization" | — | `diagonalize()` takes/returns `Gpu<WavefunctionSet>` |
| "D2H ψ for waveform layout reorg before density FFT" | Requires PCIe transfer of ψ (O(N_pw × N_bands)) | **No transfer needed.** `ColumnDistributed<Gpu>` layout already has per-band contiguity; batched cuFFT operates directly |
| "D2H ρ for CPU-side operations" | Density crosses PCIe every SCF iteration | **ρ stays GPU.** Mixing coefficients from CPU (tens of bytes), AXPY combination on GPU |
| "RR is memory-bound bottleneck" | — | Only O(n_bands²) data crosses PCIe; RR frequency is once per outer SCF iter |

### Full-residency comparison

| Data | CASTEP GPU (current) | Abinit GPU (paper) | This design |
|------|---------------------|-------------------|-------------|
| ψ during diagonalization | GPU | GPU | GPU |
| ψ for density construction | D2H + H2D | D2H (layout reorg) + H2D | **GPU** (batched cuFFT on column-distributed layout) |
| ρ during SCF loop | D2H + H2D every iter | D2H every iter | **GPU** (never leaves) |
| Mixing | CPU | CPU | **GPU** (only tiny coeffs from CPU) |
| Convergence check | CPU | CPU | **GPU** reduction, 1 scalar → CPU |
| Subspace matrices (RR) | D2H | D2H | D2H (unavoidable — LAPACK hegvd on CPU is optimal) |

## Hybrid deployment: Rust cdylib called from CASTEP Fortran

The design compiles as a `cdylib` (`.so`). CASTEP calls it via `iso_c_binding`. The Rust internals stay fully typed; only the `extern "C"` boundary functions strip to C ABI.

### Architecture

```
┌───────────────────────────────────────────────────────────┐
│ CASTEP Fortran (castep.mpi)                               │
│                                                           │
│  ┌─────────────────────────────────────────────────┐     │
│  │ Initialisation: cell, symmetry, k-points,       │     │
│  │ pseudopotential I/O, wavefunction init          │     │
│  └──────────────────────┬──────────────────────────┘     │
│                         │                                  │
│  ┌──────────────────────▼──────────────────────────┐     │
│  │ SCF loop (Fortran)                              │     │
│  │   IF (use_rust_scf) THEN                        │     │
│  │     CALL scf_run(ctx, rho, E, eps, f)           │ ← iso_c_binding
│  │   ELSE                                          │     │
│  │     CALL original_fortran_scf(...)              │     │
│  │   END IF                                        │     │
│  └──────────────────────┬──────────────────────────┘     │
│                         │                                  │
│  ┌──────────────────────▼──────────────────────────┐     │
│  │ Post-processing: DOS, bands, ELNES, NMR, ...    │     │
│  └─────────────────────────────────────────────────┘     │
│                                                           │
│  Owns: cell, symmetry, k-points, I/O buffers (CPU)       │
└──────────────────────┬────────────────────────────────────┘
                       │  C ABI (extern "C" + iso_c_binding)
┌──────────────────────▼────────────────────────────────────┐
│ libscf_backend.so (Rust cdylib)                           │
│                                                           │
│  ┌─────────────────────────────────────────────────┐     │
│  │ extern "C" boundary functions (thin wrappers)    │     │
│  │   scf_context_create / scf_run / scf_destroy     │     │
│  │   ┌───────────────────────────────────────┐     │     │
│  │   │ unsafe { raw → typed }  (one block)   │     │     │
│  │   │        ↓                              │     │     │
│  │   │ Fully typed Rust internals:           │     │     │
│  │   │   ScfIteration<Initialized>           │     │     │
│  │   │   Gpu<Density>, Gpu<WavefunctionSet>  │     │     │
│  │   │   Gpu<EffectivePotential>             │     │     │
│  │   │        ↓                              │     │     │
│  │   │ unsafe { typed → raw }  (one block)   │     │     │
│  │   └───────────────────────────────────────┘     │     │
│  └─────────────────────────────────────────────────┘     │
│                                                           │
│  Owns: ρ (GPU), ψ (GPU), V_eff (GPU), cuFFT plans,       │
│        PseudopotentialSet (parsed from USP files)         │
└──────────────────────────────────────────────────────────┘
```

### The C ABI boundary

```rust
// ─── lib.rs — cdylib public API ───

use std::ffi::CStr;
use std::os::raw::c_char;

/// Opaque handle. Fortran sees only `type(c_ptr)`.
pub struct ScfContext {
    state: ScfIteration<Initialized>,
    ndeg: usize,
    tol: f64,
    wave_grid: GVectorGrid,
    fine_grid: GVectorGrid,
    stream: CudaStream,
    // Pseudopotentials parsed by Rust from CASTEP's file directory
    pots: PseudopotentialSet,
}

/// Create context. Fortran passes: density array, grid dims, cell params,
/// pseudopotential directory path, smearing width, polynomial degree, tolerance.
#[no_mangle]
pub extern "C" fn scf_context_create(
    density_ptr: *const f64,        // ρ(r) on wave grid, column-major
    grid_dims: *const i32,          // [nx, ny, nz]
    cell_real_lat: *const f64,      // 3×3 real-space lattice
    cell_recip_lat: *const f64,    // 3×3 reciprocal lattice
    psi_ptr: *const f64,           // initial wavefunction (real+imag interleaved)
    n_bands: i32,
    n_pw: i32,
    pseudopot_dir: *const c_char,   // path to USP files
    smearing_width: f64,
    ndeg: i32,
    tol: f64,
) -> *mut ScfContext {
    // ─── UNSAFE: unpack C ABI into typed Rust (one block) ───
    let density = unsafe {
        let dims = std::slice::from_raw_parts(grid_dims, 3);
        let [nx, ny, nz] = [dims[0] as usize, dims[1] as usize, dims[2] as usize];
        let len = nx * ny * nz;
        let slice = std::slice::from_raw_parts(density_ptr, len);
        Density(WaveGridArray(Array3::from_shape_vec(
            (nz, ny, nx).f(), slice.to_vec(),
        ).unwrap()))
    };

    let psi = unsafe {
        // PW coefficients: n_pw × n_bands, complex interleaved
        let len = (n_pw as usize) * (n_bands as usize);
        let slice = std::slice::from_raw_parts(psi_ptr, len * 2);
        let data: Vec<Complex64> = slice
            .chunks_exact(2)
            .map(|c| Complex64::new(c[0], c[1]))
            .collect();
        WavefunctionSet::<RowDistributed>::new(data, n_bands as usize, n_pw as usize)
    };

    let cell = unsafe {
        let rl = std::slice::from_raw_parts(cell_real_lat, 9);
        let rcp = std::slice::from_raw_parts(cell_recip_lat, 9);
        CellGeometry::new(
            RealLattice::from_inner([rl[0..3].try_into().unwrap(), ...]),
            RecipLattice::from_inner([rcp[0..3].try_into().unwrap(), ...]),
        )
    };

    let pseud_dir = unsafe { CStr::from_ptr(pseudopot_dir) }
        .to_str()
        .expect("valid UTF-8 path");

    // ─── SAFE: typed Rust from here down ───
    let pots = PseudopotentialSet::from_directory(pseud_dir)
        .expect("failed to parse pseudopotentials");

    let wave_grid = GVectorGrid::new([nz, ny, nx]);
    let fine_grid = GVectorGrid::new([2*nz, 2*ny, 2*nx]); // or from cell

    let stream = CudaStream::new().expect("CUDA stream");

    let state = ScfIteration::<Initialized>::new(
        Gpu(density),
        Gpu(psi),
        cell,
        pots,
        wave_grid.clone(),
        fine_grid.clone(),
        SmearingParams::fermi_dirac(smearing_width),
        KPoint::gamma(),
    );

    // ─── UNSAFE: box and return opaque pointer ───
    Box::into_raw(Box::new(ScfContext {
        state, ndeg: ndeg as usize, tol, wave_grid, fine_grid, stream, pots,
    }))
}

/// Run full SCF to convergence. Returns 0 on success, >0 for iterations, <0 for error.
#[no_mangle]
pub extern "C" fn scf_run(
    ctx: *mut ScfContext,
    rho_out: *mut f64,           // output density (Fortran-allocated)
    total_energy: *mut f64,       // output scalar
    eigenvalues: *mut f64,        // output array, n_bands
    forces: *mut f64,             // output array, 3*n_atoms (nullable)
) -> i32 {
    // ─── UNSAFE: unpack opaque pointer ───
    let ctx = unsafe { &mut *ctx };
    let ndeg = ctx.ndeg;
    let tol = ctx.tol;

    // ─── SAFE: typed Rust SCF ───
    // Can't move out of ctx.state directly (behind &mut), swap in a dummy
    let state = std::mem::replace(
        &mut ctx.state,
        // dummy — will be overwritten by loop result or error path
        ScfIteration::<Initialized>::dummy(),
    );

    let result = run_scf_gpu(state, ndeg, tol, &ctx.stream);

    match result {
        Ok(final_result) => {
            // ─── UNSAFE: pack typed results into C pointers ───
            unsafe {
                // D2H density
                let rho_cpu = final_result.density.sync_to_host(&ctx.stream);
                let rho_array = rho_cpu.0.as_array();
                let len = rho_array.len();
                std::ptr::copy_nonoverlapping(rho_array.as_ptr(), rho_out, len);

                // Scalar energy
                std::ptr::write(total_energy, final_result.total_energy);

                // Eigenvalues
                let n_bands = final_result.eigenvalues.len();
                std::ptr::copy_nonoverlapping(
                    final_result.eigenvalues.as_ptr(), eigenvalues, n_bands,
                );

                // Forces (if requested)
                if !forces.is_null() {
                    if let Some(ref f) = final_result.forces {
                        std::ptr::copy_nonoverlapping(f.as_ptr(), forces, f.len());
                    }
                }

                // Store state back for potential reuse
                ctx.state = final_result.into_initialized();
            }
            0 // converged
        }
        Err(state) => {
            // Not converged within max iterations — state is Initialized
            ctx.state = state;
            -1 // error: not converged
        }
    }
}

/// Free GPU memory and drop everything.
#[no_mangle]
pub extern "C" fn scf_context_destroy(ctx: *mut ScfContext) {
    if ctx.is_null() { return; }
    unsafe {
        // Box::from_raw takes ownership back, Drop runs (GPU memory freed)
        let _ = Box::from_raw(ctx);
    }
}
```

### Fortran calling side

```fortran
! ─── scf_rust_interface.f90 ───
module scf_rust_interface
  use iso_c_binding
  implicit none
  private
  public :: scf_context_create, scf_run, scf_context_destroy

  interface
    function scf_context_create(density, grid_dims, cell_real, cell_recip, &
                                psi, n_bands, n_pw, pseud_dir, &
                                smearing, ndeg, tol) &
             bind(C, name="scf_context_create")
      import c_ptr, c_double, c_int, c_char
      type(c_ptr) :: scf_context_create
      real(c_double), intent(in) :: density(*)
      integer(c_int), intent(in) :: grid_dims(*)
      real(c_double), intent(in) :: cell_real(*), cell_recip(*)
      real(c_double), intent(in) :: psi(*)
      integer(c_int), value     :: n_bands, n_pw
      character(c_char), intent(in) :: pseud_dir(*)
      real(c_double), value     :: smearing
      integer(c_int), value     :: ndeg
      real(c_double), value     :: tol
    end function

    function scf_run(ctx, rho_out, total_energy, eigenvalues, forces) &
             bind(C, name="scf_run")
      import c_ptr, c_double, c_int
      integer(c_int) :: scf_run
      type(c_ptr), value        :: ctx
      real(c_double), intent(out) :: rho_out(*), total_energy, eigenvalues(*)
      real(c_double), intent(out) :: forces(*)
    end function

    subroutine scf_context_destroy(ctx) bind(C, name="scf_context_destroy")
      import c_ptr
      type(c_ptr), value :: ctx
    end subroutine
  end interface
end module scf_rust_interface

! ─── In CASTEP's SCF module ───
subroutine scf_cycle(...)
  use scf_rust_interface, only: scf_context_create, scf_run, scf_context_destroy
  type(c_ptr) :: rust_ctx
  integer :: status

  if (use_rust_scf) then
    rust_ctx = scf_context_create( &
      c_loc(density), c_loc(grid), c_loc(real_lat), c_loc(recip_lat), &
      c_loc(psi), n_bands, n_pw, &
      c_loc(pseud_dir // c_null_char), smearing_width, ndeg, tol)

    status = scf_run(rust_ctx, &
      c_loc(new_density), c_loc(total_energy), c_loc(eigenvalues), &
      c_loc(forces))

    call scf_context_destroy(rust_ctx)

    if (status /= 0) then
      ! Fall back to Fortran SCF, or handle error
    end if
  else
    call original_fortran_scf(...)
  end if
end subroutine
```

### What lives where

| Concern | Fortran (CASTEP) | Rust (cdylib) |
|---------|-----------------|---------------|
| Cell/symmetry setup | Owns, passes to Rust | Receives copies |
| Pseudopotential files | Passes directory path | Parses internally (`PseudopotentialSet::from_directory`) |
| Initial density | Owns CPU copy, passes to Rust | Copies to GPU, owns GPU copy |
| Initial wavefunction | Owns CPU copy, passes to Rust | Copies to GPU from interleaved complex |
| SCF iteration | Calls `scf_run` | Owns the loop, GPU-resident |
| Density during SCF | — | GPU only, never D2H'd |
| Density after SCF | Receives D2H'd result | GPU → CPU transfer at boundary |
| Forces/energy/eigenvalues | Receives | GPU → CPU transfer at boundary |
| Post-processing (DOS, etc.) | Owns | — |
| Error/fallback | Catches `status != 0`, falls back to Fortran SCF | Returns error code |

### Internal types untouched

The boundary functions repack at the edges. Every type from the design — `ScfIteration<Initialized>`, `Gpu<Density>`, `WavefunctionSet<RowDistributed>`, `EffectivePotential` — lives exclusively inside Rust. The C ABI only sees raw pointers and primitives. The `unsafe` blocks are quarantined to the `extern "C"` functions; the internal code never uses raw pointers.

### Build integration

```make
# In CASTEP's Makefile (or CMakeLists.txt)
RUST_SCF_DIR = ../rust_scf_backend

librust_scf_backend:
	cd $(RUST_SCF_DIR) && cargo build --release
	cp $(RUST_SCF_DIR)/target/release/libscf_backend.so $(LIB_DIR)/

# Link like any other library
LIBS += -L$(LIB_DIR) -lscf_backend
```

Cargo profile for HPC (compatible with the Fortran compiler's flags):

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"        # no unwinding across C ABI
```

## Reuse from chemrust-hamiltonian

| Module | What to reuse | Where it plugs in |
|--------|--------------|-------------------|
| `types.rs` | `real_space_field!` macro, `Density`, `EffectivePotential`, `CellGeometry`, `Error` | Extend with `WaveGridArray`/`FineGridArray` distinction |
| `band_structure.rs` | `VEffBuilder`, `SpinPolicy` (NonSpin/SpinCollinear) | `build_v_eff()` transition |
| `hamiltonian.rs` | `apply_local_hamiltonian`, `kinetic_expectation` | Chebyshev filtering inner loop |
| `nlpot.rs` | `nlpot_apply`, `nlpot_expectation`, `compute_beta_phi`, `build_d0_expanded` | Chebyshev filtering inner loop (V_NL|ψ>) |
| `poisson.rs` | `solve_poisson` | Already called inside `VEffBuilder::assemble()` |
| `vion.rs` | `reconstruct_v_ion` | Already called inside `VEffBuilder::assemble()` |
| `xc/` | `compute_pbe_xc` | Already called inside `VEffBuilder::assemble()` |
| `fft.rs` | `fft_inverse_3d`, `fft_forward_3d`, upsample/downsample | Both `build_v_eff` and `construct_density` |
| `gvec.rs` | `GVectorGrid` | Grid management throughout |
| `pseudopotential.rs` | `PseudopotentialSet`, `HasAugmentationData` | Throughout |
| `nlcc.rs` | `reconstruct_rho_core` | Inside `VEffBuilder::assemble()` |

## Module structure

```
chemrust-hamiltonian-core/src/
├── scf/
│   ├── mod.rs              // Re-exports, ScfIteration type
│   ├── state.rs             // Phase markers (Initialized, VEffBuilt, ...)
│   ├── transitions.rs       // build_v_eff, diagonalize, construct_density, mix, check
│   ├── mixing.rs            // DensityHistory, linear mixing, Pulay
│   └── density.rs           // build_density_from_wavefunctions, occupations
├── eigensolver/
│   ├── mod.rs
│   ├── chebyshev.rs         // Chebyshev filtering algorithm
│   └── rayleigh_ritz.rs     // Subspace diagonalization
├── layout/
│   ├── mod.rs               // RowDistributed, ColumnDistributed
│   └── transpose.rs         // Row↔Column transposition (MPI all-to-all)
├── device/
│   ├── mod.rs               // Gpu<T>, Cpu<T>, sync primitives
│   └── fft_backend.rs       // Binds cuFFT or VkFFT
└── ... (existing modules)
```

## Implementation order (two-tier)

### Tier 1 — Backbone (first, everything stubbed)

1. Define `WaveGridArray`/`FineGridArray` grid-level types in `types.rs`
2. Define `Gpu<T>`/`Cpu<T>` device wrappers in `device/mod.rs`
3. Define `RowDistributed`/`ColumnDistributed` layout markers
4. Define `ScfIteration<State>` with all phase markers
5. Implement each transition with `todo!()` bodies (not even stubs — just the type signatures)
6. Write the `run_scf` loop — verify it **compiles** (all phases line up)
7. Write a unit test that calls `run_scf` → compiles and panics on first `todo!`

### Tier 2 — Fill the holes (one transition at a time)

1. `build_v_eff` — trivial, almost entirely reuses existing `VEffBuilder`
2. `construct_density` — new code, but simple: FFT + |ψ|² + occupations
3. `mix` / `check` — new code, also simple: linear algebra on Density arrays
4. `diagonalize` — the big one: Chebyshev filtering + Rayleigh-Ritz

### Tier 3 — GPU

1. Swap `rustfft` backend for `gpufft` (Vulkan via VkFFT, or CUDA via cuFFT)
2. Wrap in `Gpu<T>` — FFT calls now take `Gpu<Array3<Complex64>>`
3. GPU-resident ψ during eigensolver loop
4. Adopt `cuda-oxide` for custom fused kernels if needed

#### GPU FFT crate selection: `gpufft`

After evaluating four Rust GPU FFT crates, **`gpufft` (alejandro-soto-franco/gpufft)** is the choice:

| Candidate | Dealbreaker |
|-----------|------------|
| `gpu-fft` (eugenehp) | f32 only (no double precision); radix-2 only (DFT grids aren't powers of 2); custom compute-shader FFT, not vendor-tuned |
| `vkfft-rs` (semio-ai) | Vulkan only (no CUDA — cuFFT is 2-3× faster on NVIDIA); complex CMake build; safety warnings in README |
| `scirs2-fft` (cool-japan) | 50+ crate framework; GPU FFT is sparse-FFT-focused, not dense batched cuFFT |
| **`gpufft`** | Clean trait API; dual CUDA+Vulkan backend; C2C/R2C/C2R; 1D/2D/3D; f64; type-safe per-backend buffers/plans. Limitation: batch only for 1D (2D/3D `batch=1`). Workaround: decompose 3D FFT into batched 1D+transpose (standard DFT approach, already what CASTEP does). |

For production: `gpufft` with CUDA backend (`cargo add gpufft -F cuda`). For AMD fallback: Vulkan backend. Swap via feature flag; same trait surface.

#### cuSOLVER: GPU-resident Rayleigh-Ritz

The Rayleigh-Ritz step needs `ZHEGVD` — complex Hermitian generalized eigenvalue decomposition. **cuSOLVER provides this directly on GPU**, making the entire step GPU-resident:

**Before (CPU LAPACK):**
```
H_sub, S_sub (n_bands², GPU) → D2H → CPU ZHEGVD → X, ε → H2D X → ψ = X·ψ
```
PCIe: 3 × n_bands² per SCF iteration (2 D2H + 1 H2D)

**After (cuSOLVER):**
```
H_sub, S_sub (n_bands², GPU) → cuSOLVER ZHEGVD (GPU) → X, ε (GPU) → ψ = X·ψ
```
PCIe: 0 (eigenvalues D2H only if CASTEP needs them — n_bands f64s)

**Implementation via cudarc:**

`cudarc` (v0.19.7) includes `cusolverDn.h` in its bindgen wrapper, so `sys::cusolverDnZhegvd()` already exists as a raw FFI function. The safe wrapper layer (v0.19.7) only covers handle creation/destruction, not solver functions — but the raw sys binding is ready to use:

```rust
use cudarc::cusolver::sys;
use cudarc::driver::{CudaStream, DeviceBuffer};
use cudarc::cublas::sys::cublasFillMode_t;

/// Solve H·X = ε·S·X on GPU. H, S overwritten; X and ε on GPU.
pub fn zhegvd(
    handle: &DnHandle,
    jobz: i32,           // CUSOLVER_EIG_MODE_VECTOR = 1
    uplo: i32,           // CUBLAS_FILL_MODE_LOWER = 1
    n: i32,              // n_bands
    a: &mut DeviceBuffer<Complex64>,  // H_sub → X (in-place)
    b: &mut DeviceBuffer<Complex64>,  // S_sub (overwritten)
    eigenvalues: &mut DeviceBuffer<f64>,
    info: &mut DeviceBuffer<i32>,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    // Query workspace size (CUDA convention: pass nullptr for workspace, get lwork)
    let mut lwork = 0i64;
    sys::cusolverDnZhegvd_bufferSize(
        handle.cu(), jobz, uplo, n, a.as_dev_ptr(), n,
        b.as_dev_ptr(), n, &mut lwork, // ... info
    ).result()?;

    // Allocate workspace
    let mut workspace = DeviceBuffer::new(lwork as usize)?;

    // Execute (synchronous on the stream)
    sys::cusolverDnZhegvd(
        handle.cu(), jobz, uplo, n,
        a.as_dev_ptr(), n, b.as_dev_ptr(), n,
        eigenvalues.as_dev_ptr(),
        workspace.as_dev_ptr(), lwork,
        info.as_dev_ptr(),
    ).result()?;

    // info[0] = 0 means success
    Ok(info.stream_read(&[0])?[0] == 0)
}
```

**Impact on transfers:**

| Transfer | Before (CPU LAPACK) | After (cuSOLVER) |
|----------|--------------------|------------------|
| H_sub | D2H: n_bands² × 16 bytes | **GPU stays GPU** |
| S_sub | D2H: n_bands² × 16 bytes | **GPU stays GPU** |
| X eigenvectors | H2D: n_bands² × 16 bytes | **GPU stays GPU** |
| Eigenvalues ε | — | D2H: n_bands × 8 (tiny, only for CASTEP output) |

For a 1000-band system: ~48 MB/iter saved. For the Rayleigh-Ritz step: **zero PCIe cost.**

**Risks:**
- cuSOLVER's `ZHEGVD` is a blocking operation (synchronous on the stream). For n_bands ≤ 4000 this is negligible (~10ms on A100). For larger problems, cuSOLVER's batched or iterative solvers are alternatives.
- cudarc leaves safe solver wrappers to the user — you write ~30 lines of FFI boilerplate. One time.
- For AMD GPUs (future): no cuSOLVER. Fallback to CPU LAPACK for the cross-vendor path.

## Real-world motivation: 15 crash logs from CASTEP GPU-resident SCF

The following analysis is based on 15 distinct crash records from `/tmp/cu111_gpu_resident_scf/slurm_output_*.txt`, all from attempts to run CASTEP's GPU-resident diagonalization (branch `impl/phase-2/gpu-resident-scf`) on a Cu111_CO system. Each bug category maps to a Fortran limitation that Rust's type system closes.

### Bug 1: Undetected transfer failure
*Evidence: 7 of 15 logs. `wave_gpu_memcpy_d2h: failed` / `wave_gpu_memcpy_h2d: failed` — error printed, execution continues with garbage. Eigenvalues become `-1.9537E-16` (machine epsilon), then `-1.2821-313` (subnormal junk), then `Infinity`.*

**Fortran root cause:** GPU→CPU or CPU→GPU transfer returns an error code. The `print` statement executes but the code does not stop. Every subsequent computation reads corrupted data.

**Rust prevention:** `sync_to_host()` returns `Result<Cpu<T>, Error>`. The `?` operator forces propagation. No implicit "ignore the error and plow ahead with garbage" path exists.

| Layer | Type involved | Compiler guarantee |
|-------|--------------|-------------------|
| Device | `Gpu<WaveGridArray>` vs `Cpu<WaveGridArray>` | Wrong-location data is a type mismatch, not a runtime surprise |
| Transfer | `sync_to_host(stream) -> Result<Cpu<T>, Error>` | Error must be handled; unhandled errors kill the process in `abort` mode |

### Bug 2: In-place FFT work buffer self-destruct
*Evidence: `BATCH_FFT_WARN: d_work_R band 0 norm = 0.00E+00 (IN-PLACE?)`. First H|ψ> produces correct norm (0.99), subsequent inner iterations produce zero. The in-place buffer writes over its own input.*

**Fortran root cause:** A single large array `d_work_R` serves as both input and scratch across all inner iterations. Its contents are silently consumed on the second call. The calling convention ("caller guarantees buffer is valid") is enforced only by comments.

**Rust prevention:** The state machine makes buffer lifecycle explicit. `ScfIteration<WavefunctionsUpdated>` holds a `d_work_R` buffer with clear semantics: consuming it produces new data. Re-calling `hpsi_apply` without going through the `VEffBuilt` phase is a compile error because the phase marker forbids it.

| Layer | Type involved | Compiler guarantee |
|-------|--------------|-------------------|
| State | `fn chebyshev_iter(self) -> Result<Self, Error>` | Each H|ψ> consumes `self`; a second call without going back through `VEffBuilt` is impossible |
| Phase | `ScfIteration<WavefunctionsUpdated>` | This type has `v_eff` in it; another `hpsi_apply` would need to return to `VEffBuilt` first |

### Bug 3: Undetected Cholesky failure (ZPOTRF info=1)
*Evidence: `wave_gpu_sorthonormalise: ZPOTRF info=1` in every failing log. The Gram matrix ψ†ψ is not positive-definite because ψ is garbage from Bug 1 or Bug 2. ZPOTRF returns `info=1` meaning the matrix isn't positive-definite, but the code continues with the invalid decomposition.*

**Fortran root cause:** LAPACK convention: error codes in an `info` integer argument. The caller prints `info` but does not return from the subroutine. The diagonalization proceeds with a non-orthogonal basis.

**Rust prevention:** `orthonormalise()` takes `Gpu<WavefunctionSet<RowDistributed>>` and returns `Result<Gpu<WavefunctionSet<RowDistributed>>, OrthonormalisationError>`. The `info=1` case is a variant of the error enum. The calling code must `?` or `match` it.

| Layer | Type involved | Compiler guarantee |
|-------|--------------|-------------------|
| Return type | `Result<_, OrthonormalisationError>` | `info` is not just printed — it terminates the transition |

### Bug 4: Exponential blowup cascade
*Evidence: `BATCH_FFT_DIAG: band 0 norm grows 0.99 → 8.05E+3 → 6.52E+7 → ... → 6.42E+50` (logs 2213, 2215). Each H|ψ> amplifying the garbage from the previous iteration. No circuit breaker.*

**Fortran root cause:** The Chebyshev filtering loop runs for `ndeg` iterations regardless of ψ corruption. There is no guard that checks ψ norm before proceeding.

**Rust prevention:** The Chebyshev filtering function is a single transition, not a loop in user code. If any step produces `Result::Err`, the entire transition returns `Err`. Alternatively, a norm check between inner iterations returns `Result<Gpu<WavefunctionSet>, Error>` when norm diverges past a threshold.

| Layer | Type involved | Compiler guarantee |
|-------|--------------|-------------------|
| Transition | `fn diagonalize(self) -> Result<ScfIteration<WavefunctionsUpdated>, Error>` | All inner iterations are inside this function; any failure propagates out |
| Data flow | Input is `ScfIteration<VEffBuilt>` which guarantees `v_eff: Some` and valid ψ | If prior state is corrupted, the `VEffBuilt` constructor catches it |

### Bug 5: Zero-size allocation
*Evidence: `wave_gpu_malloc: failed, size=0` (log 2215). Dimension computation gave zero; calling code didn't check.*

**Fortran root cause:** `allocate(arr(size))` where `size` was computed from a prior calculation that produced zero. Fortran `allocate` with `size=0` is invalid, but no type-level guard against it.

**Rust prevention:** `DeviceBuffer::new(size: NonZeroUsize)` makes zero a compile-time rejection. Or `DeviceBuffer::new(size: usize)` returns `Result<_, ZeroSizeError>`.

| Layer | Type involved | Compiler guarantee |
|-------|--------------|-------------------|
| Allocation | `DeviceBuffer::new(NonZeroUsize) -> Result<_, CudaError>` | Zero-size arguments fail at runtime safely; `NonZeroUsize` at compile time |

### Bug 6: Nonlocal projector H2D failure mid-diagonalization
*Evidence: `nlpot_gpu_apply_add_recip_resident: H2D d_recip_beta failed` (log 2217). GPU memory exhausted or stream synchronised incorrectly during the Chebyshev loop.*

**Fortran root cause:** Nonlocal projector arrays are copied to GPU on-demand during the diagonalization. If the copy fails (e.g., running out of GPU memory), the GPU-resident β-projectors are in an undefined state but still used.

**Rust prevention:** β-projectors are allocated once during `ScfContext::new()` and live on GPU for the entire SCF cycle. `compute_beta_phi_gpu()` returns `Result<Gpu<BetaPhi>, Error>` at initialization. If it fails, the context is never created.

| Layer | Type involved | Compiler guarantee |
|-------|--------------|-------------------|
| Initialization | `scf_context_create` returns `*mut ScfContext` or null | Failure -> null pointer -> CASTEP can fall back to Fortran SCF |

### Bug 7: 26 GB PCIe traffic per SCF cycle for a 160-band system
*Evidence: `GPU_TRANSFER_BREAKDOWN: H2D 16.5 GB, D2H 9.6 GB` for a system where total GPU data is ~3.4 GB. | The code is transferring ~8× the buffer contents per cycle.*

**Fortran root cause:** Transfers are implicit — each subroutine copies data to/from GPU independently. No single function has visibility into the aggregate transfer pattern. The diagnostic counters had to be added to discover the thrashing.

**Rust prevention:** Every transfer is an explicit `.sync_to_host()` or `.sync_to_device()` call at a `Gpu<T>` ↔ `Cpu<T>` boundary. The compiler guarantees that no transfer happens without a visible `.sync_*()` call. A code review of total transfer calls gives the exact byte count.

| Layer | Type involved | Compiler guarantee |
|-------|--------------|-------------------|
| Device | `Gpu<T>::sync_to_host(stream) -> Cpu<T>` | Transfer is explicit in every call site; no implicit D2H on a `Gpu<T>` value |
| Accounting | Only `mix`, `check`, and final output call `sync_to_host` | Transfer count is derivable from the code structure |

### Bug 8: Block restart inherits corrupted global state
*Evidence: Log 2221. Inner loop steps 1-3 produce reasonable eigenvalues (~-2.0E+04), then step 4 onward collapses to `-1.2821-313`. The block restart (`nb=1, nb=26, nb=51, ...`) re-uses a global scratch buffer with garbage from the first block.*

**Fortran root cause:** The block restart logic spans 50+ lines of Fortran with global-scope buffer reuse. Each restart's data validity depends on every previous iteration having cleaned up properly. Writing to the same buffer array from different blocks is safe if and only if no previous block left intermediate data behind.

**Rust prevention:** Block restart is a loop over `ScfIteration<State>`. Each iteration produces a clean `State` by consuming the previous one. No global scratch buffers — the per-iteration state owns its work array. When the type transitions, the previous iteration's data is dropped.

| Layer | Type involved | Compiler guarantee |
|-------|--------------|-------------------|
| State machine | `ScfIteration<WavefunctionsUpdated>` owns `d_work_R` | Each iteration produces a new state; the previous allocation is consumed/dropped |
| No globals | All state is in `ScfIteration<State>` | No unowned buffers that could persist corrupted data across restarts |

### Summary

| Bug category | Count in logs | Fortran fix required (manual) | Rust protection (automatic) |
|-------------|:---:|---|---|
| Undetected D2H/H2D failure | 7/15 | Add error check after every `cudaMemcpy` | `sync_to_host()` returns `Result`; `?` forces handling |
| FFT work buffer self-destruct | 7/15 | Vertify FFT plan is out-of-place, add guard | State machine consumes buffer at phase boundary |
| ZPOTRF info=1 ignored | 6/15 | Check `info` after every LAPACK call | `Result<_, OrthonormalisationError>` propagates |
| Exponential blowup | 2/15 | Add norm guard between Chebyshev iterations | Transition returns `Err` if any inner step fails |
| Zero-size allocation | 1/15 | Guard allocation size | `NonZeroUsize` or `Result` |
| Nlpot H2D failure mid-loop | 1/15 | Pre-allocate all beta projectors | Allocated once in `scf_context_create` |
| Hidden PCIe thrashing | 1/15 | Add counters, find hot path, refactor | `Gpu<T>` / `Cpu<T>` makes every transfer visible |
| Block restart corrupt state | 2/15 | Manual buffer reset at each restart | Ownership guarantees clean state per iteration |

## Verification plan

| What to verify | How |
|---------------|-----|
| State machine compiles | Build check: `run_scf()` with stub bodies must not have type errors |
| Transitions can't be called out of order | Negative compile test: `trybuild` tests that assert *failure* for wrong-phase calls |
| V_eff assembly matches CASTEP | Already validated in existing `chemrust-hamiltonian` tests |
| Density construction matches CASTEP | Run on Cu111 reference `.castep_bin` (in memory), compare ρ_new |
| Chebyshev filtering eigenvalues match CASTEP | Compare per-band eigenvalues for a single SCF iteration |
| Full SCF total energy matches CASTEP | Run on Cu111, Si, GaAs; compare final total energy within 1e-5 eV/atom |
| GPU-resident path produces same result as CPU | Bitwise or near-bitwise identical for the same SCF input |

## Reference files

- arXiv article: `/home/tony/programming/CASTEP-GPU-port/2604.11139v1` (gzipped tar of `main.tex`)
- User's Rust crate: `TonyWu20/chemrust-hamiltonian` on GitHub
- cuda-oxide: `NVlabs/cuda-oxide` on GitHub (1,906 stars, Apache-2.0, alpha)
- CASTEP GPU port (current work): `/home/tony/programming/CASTEP-GPU-port/` (Fortran, branch `impl/phase-2/gpu-resident-scf`)
