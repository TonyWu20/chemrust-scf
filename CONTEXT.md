# Chemrust SCF

GPU-native, type-safe plane-wave DFT SCF engine in Rust. Builds on
`chemrust-hamiltonian` for V_eff assembly, pseudopotential machinery, and
Hamiltonian apply. Compiles as a `cdylib` callable from CASTEP Fortran.

Eliminates the class of correctness bugs (layout mismatches, missing syncs,
wrong-array ordering) that CASTEP's Fortran GPU port cannot catch at compile
time.

## Language

### Core types

**Density**:
Electron density ρ(r) on the wave grid, wrapped as `Density(WaveGridArray)`.
_Avoid_: "charge density" (conflates ρ with total charge, a scalar)

**Wave Grid**:
FFT grid whose size is determined by the wavefunction cutoff (G-vector cutoff),
typically ~1.5× the PW basis size per dimension.
_Avoid_: "G-grid", "coarse grid"

**Fine Grid**:
2× upsampled FFT grid used for V_eff assembly — XC evaluation and Hartree
potential require higher real-space resolution.
_Avoid_: "density grid"

**WavefunctionSet**:
N_bands complex plane-wave coefficients, typed by MPI distribution layout:
`WavefunctionSet<RowDistributed>` or `WavefunctionSet<ColumnDistributed>`.
_Avoid_: "psi", "wavefunction array" (untyped), "wvfn"

**GVectorGrid**:
Stores G-vector indices (Miller indices) for the plane-wave basis, mapping
between reciprocal-space indices and FFT grid points.

### Eigensolver

**Chebyshev Filtering**:
Polynomial subspace iteration for eigenvalue solving. Applies a Chebyshev
polynomial of H to the trial subspace, amplifying eigencomponents above the
chosen filter window. Achieves high arithmetic intensity on GPU since it does
MPI transpose once per k Hamiltonian applications (vs LOBPCG's per-block
transpose).

**Rayleigh-Ritz**:
Projects H into the subspace spanned by filtered ψ, solves the generalized
eigenproblem H_sub·X = ε·S_sub·X (ZHEGVD), and rotates ψ ← X·ψ.
_Avoid_: "subspace diagonalization" (too generic)

### Type-layer distinctions

**RowDistributed** / **ColumnDistributed**:
Phantom-type markers on `WavefunctionSet` encoding MPI distribution layout.
Hamiltonian apply requires ColumnDistributed; Rayleigh-Ritz requires RowDistributed.
The compiler rejects mismatched layouts.
_Avoid_: "layout mode", "waveform layout" (untagged, error-prone)

**Gpu\<T\>** / **Cpu\<T\>**:
Device-location wrappers. Accessing data on the wrong side requires explicit
`.sync_to_host()` / `.sync_to_device()`. Transfers are visible at every call site.
_Avoid_: `DevicePointer`, `HostArray`, raw `*const f64` provenance

**WaveGridArray** / **FineGridArray**:
Opaque newtypes over `Array3<f64>` encoding which grid the data lives on.
Prevents passing wave-grid data where fine-grid data is expected.
_Avoid_: raw `Array3<f64>` through the type system

### Physics

**EffectivePotential**:
V_eff[ρ] = V_H + V_xc + V_ion assembled on the fine grid, type
`EffectivePotential(FineGridArray)`. Built by `VEffBuilder` from
chemrust-hamiltonian.
_Avoid_: "potential", "V_eff" as a bare array

**SpinPolicy**:
Sealed trait with `NonSpin` and `SpinCollinear` variants. Controls how density
loops and XC evaluation handle spin.

**PseudopotentialSet**:
Parsed USP pseudopotentials (from CASTEP `.usp` files). Owns β-projectors,
D-matrix, V_loc, and augmentation data for all species in the calculation.

## Relationships

- A **Density** lives on the **Wave Grid**.
- **EffectivePotential** is assembled on the **Fine Grid** from density
  upsampled from the wave grid.
- **Chebyshev Filtering** operates on `WavefunctionSet<ColumnDistributed>`
  (Hamiltonian apply layout).
- **Rayleigh-Ritz** operates on `WavefunctionSet<RowDistributed>` (band-band
  inner product layout).
- **ScfIteration\<State\>** transitions through 5 non-terminal phases
  (Initialized → VEffBuilt → WavefunctionsUpdated → DensityUpdated → Mixed)
  with Converged as the terminal phase.

## Example dialogue

> **Dev:** "During the Rayleigh-Ritz step, what layout does the wavefunction need
> to be in?"
> **Domain:** "RowDistributed — because we're computing ψ†·H·ψ, which is a
> band-band inner product. ColumnDistributed would give us plane-wave inner
> products instead. The compiler rejects the wrong one."
> **Dev:** "And what about the density after mixing — does it stay on GPU?"
> **Domain:** "Yes. The entire mixing step runs on GPU. Only `history_size²`
> doubles cross PCIe to solve the Pulay system on CPU. The density never leaves
> GPU during the SCF loop."

## Flagged ambiguities

- "psi" / "wavefunction array" was used both for the full set and for a single
  band — resolved: full set is `WavefunctionSet`, individual `Wavefunction`.
- "potential" conflated V_eff, V_H, V_xc, V_ion — resolved: each has its own
  opaque newtype.
- "subspace diagonalization" conflated Rayleigh-Ritz with generic
  diagonalization — resolved: Rayleigh-Ritz is the specific procedure after
  Chebyshev filtering.

## Tooling

**Language:** Rust (nightly, for cuda-oxide compatibility in future phases)
**Build command:** `cargo check --workspace 2>&1`
**Test command:** `cargo test --workspace 2>&1`
**Lint command:** `cargo clippy --workspace -- -D warnings 2>&1`
**File extensions:** `.rs`
**Test file patterns:** `tests/**/*.rs src/**/*.rs`
**Test filter flag:** `-p <package> <test_name>`
**Package manager:** `cargo`
**Package boundaries:** single crate, scoped modules under `src/`:
  `scf/` (SCF state machine + transitions),
  `eigensolver/` (Chebyshev filtering + Rayleigh-Ritz),
  `layout/` (RowDistributed/ColumnDistributed + transpose),
  `device/` (Gpu/Cpu wrappers, GPU FFT backend),
  plus existing chemrust-hamiltonian modules (types, hamiltonian, nlpot,
  poisson, vion, xc, fft, gvec, pseudopotential)

## Coding patterns

**Prefer iterators over explicit loops:**
Combinators (`map`, `fold`, `zip`, `filter`) over `for`/`while` loops.
Accumulation uses `fold` or `.sum()` rather than `a += b` in a loop body.

**Granular newtypes:**
Every semantically distinct quantity gets its own newtype, even if the
underlying representation is the same (`Vec<f64>`, `&[f64]`, `f64`). Function
parameters and return types must use these newtypes — never pass raw `Vec<f64>`
or `&[f64]` where a domain quantity is intended.

**Error handling:**
Typed error enums via `thiserror`. No `anyhow` in the core library — callers
decide how to handle errors. All `?` propagation in internal code.

**Module visibility:**
Flat module tree with re-exports from `lib.rs`. Items are `pub(crate)` by
default; only the SCF public API is `pub`.

**Async:**
Sync-only. CUDA streams are blocking in this design; no async runtime needed.

**No unsafe in internal code:**
`unsafe` quarantined to the `extern "C"` boundary functions. Internal
transitions and algorithms are safe Rust.

**Testing:**
Integration tests in `tests/` against CASTEP fixture files. Unit tests inline
(`#[cfg(test)] mod tests`). Every numeric assertion cites its ground-truth
source (fixture file, reference formula, published value).

## Fixtures

**CPU-only reference:**
`/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
Contains `.castep_bin`, `.den_fmt`, `.pot_fmt`, `.bands`, `.castep`.

**GPU-resident crash logs (informative, not ground truth):**
`/tmp/cu111_gpu_resident_scf/`
Contains slurm output from CASTEP GPU-resident SCF crash logs.

**Pseudopotentials:**
`~/Downloads/Potentials/`
USP files for all species.

## Physics Conventions

**Density unit convention:**
Raw `ρ × Ω` not Ha/Bohr³ for ρ entering Poisson/XC. The `accumulate_density`
kernel uses `inv_omega = 1.0` (no Ω division). `solve_poisson` and
`compute_pbe_xc` downstream expect this convention. The `.castep_bin` density
storage uses the same convention.

**FFT axis convention:**
`RealGrid<T>` stores data in Fortran layout `(ngz, ngy, ngx).f()`.
`RecipGrid<T>` stores the forward FFT result in the same layout. cuFFT plan
dims are `(ngz, ngy, ngx)` (innermost first) to match the scatter formula
`iz + ngz*(iy + ngy*ix)`. See failure-patterns.md:
`cufft-dim-ordering-and-rr-transpose-layout`.

**Augmentation Density domain term:**
ρ_aug(r) = Σ_I Σ_{n,m} ω^I_{nm} · Q^I_{nm}(r). Separate channel from smooth
PW ρ_PW. Total ρ = ρ_PW + ρ_aug. The `.castep_bin` density already stores the
sum. In the SCF loop: `construct_density_gpu` produces smooth-only ρ_PW;
`compute_aug_density_gpu` adds ρ_aug; `build_v_eff_with_energy_impl` sums them
before Poisson + XC.
