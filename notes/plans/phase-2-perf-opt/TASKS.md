# Forensic TASKS.md — Phase 2 Perf Opt

**Date:** 2026-05-21
**Branch:** `feat/phase-2-perf-opt`
**ODD pattern reference:** `/home/tony/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

## Declared Fixtures

| Path | Description |
|------|-------------|
| `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/` | CPU-only CASTEP reference. `.castep_bin`, `.den_fmt`, `.pot_fmt`, `.bands`, `.castep`. 18 Cu + 1 CO ions. |

## Dependency Map

```
GROUP-A (rayon in core)
  └─► GROUP-B (QSfCache + GPU aug density)
        └─► GROUP-C (beta_psi GPU residency)
              └─► GROUP-D (docs: CONTEXT.md + ADR-0003)
```

GROUP-A and GROUP-B can be developed in parallel on separate sub-branches; GROUP-C depends on GROUP-B's `QSfCache` type and GROUP-A's rayon changes being merged.

---

## GROUP-A: Rayon parallelization of chemrust-hamiltonian-core

**Branch:** `impl/phase-2-perf-opt/group-a`
**Merge target:** `feat/phase-2-perf-opt`

### TASK-A1: Add rayon dependency to chemrust-hamiltonian-core

**Kind:** direct
**Files:**
- `chemrust-hamiltonian/chemrust-hamiltonian-core/Cargo.toml`

**Changes:**
Add `rayon = "1"` to `[dependencies]` in `chemrust-hamiltonian-core/Cargo.toml`.

**Acceptance:**
```bash
cd /home/tony/programming/chemrust-hamiltonian && cargo check --workspace 2>&1 | tail -5
```

**Per-task memory footprint:** N/A — dependency addition only.

---

### TASK-A2: Parallelize `precompute_q_on_grid` radial Bessel transforms

**Kind:** lib-tdd
**Files:**
- `chemrust-hamiltonian/chemrust-hamiltonian-core/src/nlpot.rs`

**Context:**
`precompute_q_on_grid` (nlpot.rs:212) has two nested serial loops:
1. Outer: `qlnm_g.iter_mut().enumerate().for_each(|(ll, qlnm_ll)| ...)` — iterates over `ll` from 0 to `ll_max` (= 2×lmax). For Cu (d-orbitals, lmax=2), ll_max=4, so 5 iterations. For each `ll`, iterates over `(n, m)` radial pairs and calls `radial_bessel_transform` (nqpts × nrpts work, ~200 × 200 = 40k ops per pair).
2. Inner: `pairs` construction via `flat_map` over expanded projector pairs — each pair builds a `Q_arr` of shape `(ngz, ngy, ngx)` (wave-grid size, ~50k points for Cu111_CO wave grid). This is the **fine-grid** loop — do NOT parallelize (see memory constraint below).

**Eligible for rayon:** The outer `qlnm_g` loop (radial Bessel transforms). Each task's working set is `nqpts` floats (~200 × 8 bytes = 1.6 KB per pair). Safe to parallelize.

**NOT eligible:** The `pairs` construction loop that builds `Array3<Complex64>` of shape `(ngz, ngy, ngx)`. Each task allocates a full wave-grid array (~50k × 16 bytes = 800 KB). With 8 threads × 171 pairs = 1.4 GB peak. This is the same class of bug as the reverted `a50cc05` (see memory/rayon_fine_grid_memory_overhead.md).

**Change:**
Replace the serial `qlnm_g.iter_mut().enumerate().for_each(...)` with `qlnm_g.par_iter_mut().enumerate().for_each(...)` using `rayon::prelude::*`.

The `pairs` construction (the `flat_map` + `filter_map` chain at nlpot.rs:264) stays serial.

**Per-task memory footprint:** `nqpts` floats per (ll, pair) task ≈ 200 × 8 = 1.6 KB. Well within the "few MB" guard.

**Success Criteria:**
- `precompute_q_on_grid` on Cu111_CO fixture produces identical `QOnGrid.pairs` vs serial run: for each pair `(n_exp, m_exp)`, `‖q_arr_parallel − q_arr_serial‖_∞ < 1e-12`.
  Source: determinism requirement, PHASE_PLAN.md Design Notes.
- Results match within 1e-12 across 3 independent runs (rayon non-determinism guard).
  Source: PHASE_PLAN.md Design Notes.
- `cargo test --workspace` all green.

**Test file:** `chemrust-hamiltonian-core/tests/integration.rs` (add test `precompute_q_on_grid_rayon_matches_serial`).

**Test code sketch:**
```rust
#[test]
fn precompute_q_on_grid_rayon_matches_serial() {
    // Load Cu USP from fixture
    let usp = load_cu_usp("/export/Potentials/Cu_00.usp");
    let cell = load_cu111_co_cell("/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.castep_bin");
    let gvg = GVectorGrid::new(...); // wave grid from fixture

    // Run twice (rayon uses thread pool, results must be deterministic)
    let q1 = precompute_q_on_grid(&usp, &gvg).unwrap();
    let q2 = precompute_q_on_grid(&usp, &gvg).unwrap();

    assert_eq!(q1.pairs.len(), q2.pairs.len());
    for ((nm1, arr1), (nm2, arr2)) in q1.pairs.iter().zip(q2.pairs.iter()) {
        assert_eq!(nm1, nm2);
        let max_diff = arr1.iter().zip(arr2.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-12,
            "pair {:?}: max diff {:.2e} across runs", nm1, max_diff);
    }
}
```

**Acceptance:**
```bash
cd /home/tony/programming/chemrust-hamiltonian && cargo test -p chemrust-hamiltonian-core precompute_q_on_grid_rayon_matches_serial 2>&1 | tail -10
cd /home/tony/programming/chemrust-hamiltonian && cargo clippy --workspace -- -D warnings 2>&1 | tail -10
```

---

### TASK-A3: Parallelize `compute_beta_g` per-ion G-vector projection

**Kind:** lib-tdd
**Files:**
- `chemrust-hamiltonian/chemrust-hamiltonian-core/src/augment/beta_phi.rs`

**Context:**
`compute_beta_g` (beta_phi.rs:37) iterates over `wave_block.pw_grid_coord` (nplw G-vectors, ~5k for Cu111_CO wave grid) via `filter_map` + `for_each`. For each G-vector, it computes `n_expanded` projector values (spherical Bessel + Y_lm). Per-task working set: a few scalars (no array allocation). Safe to parallelize.

The function currently writes into `beta_g_out: Array2<Complex64>` via `for_each` with mutable access. To parallelize, replace with `par_iter().filter_map(...).collect::<Vec<_>>()` then scatter into the output array, or use `par_iter().for_each(|(g_idx, ...)| { ... })` with index-based writes (safe since each `g_idx` is unique).

**Per-task memory footprint:** ~`n_expanded` Complex64 values per G-vector task ≈ 18 × 16 = 288 bytes. Well within guard.

**Change:**
Replace the serial `wave_block.pw_grid_coord.iter().enumerate().filter_map(...).for_each(...)` with a parallel version using `rayon::prelude::*`. Use `par_iter()` on the G-vector index range, with index-based writes into a pre-allocated `beta_g_out` (safe: each thread writes to a unique column `g_idx`).

**Note on "no for loop" rule:** The existing code already uses combinators. The rayon version should use `par_iter().filter_map(...).for_each(...)` — same combinator style, just parallel. If index-based writes require unsafe, use `ndarray`'s `axis_iter_mut` with `par_bridge()` or collect into a `Vec` and scatter.

**Success Criteria:**
- `compute_beta_g` on Cu111_CO fixture produces identical `Array2<Complex64>` vs serial run: `‖beta_g_parallel − beta_g_serial‖_∞ < 1e-12`.
  Source: determinism requirement, PHASE_PLAN.md Design Notes.
- Results match within 1e-12 across 3 independent runs.
- `cargo test --workspace` all green.

**Test file:** `chemrust-hamiltonian-core/tests/integration.rs` (add test `compute_beta_g_rayon_matches_serial`).

**Acceptance:**
```bash
cd /home/tony/programming/chemrust-hamiltonian && cargo test -p chemrust-hamiltonian-core compute_beta_g_rayon_matches_serial 2>&1 | tail -10
cd /home/tony/programming/chemrust-hamiltonian && cargo clippy --workspace -- -D warnings 2>&1 | tail -10
```

---

## GROUP-B: QSfCache + compute_aug_density_gpu

**Branch:** `impl/phase-2-perf-opt/group-b`
**Merge target:** `feat/phase-2-perf-opt`

### TASK-B1: Define QSfCache type and build function

**Kind:** lib-tdd
**Files:**
- `chemrust-scf/src/density.rs` (or new `chemrust-scf/src/aug_cache.rs`)

**Context:**
`QSfCache` caches `Q^I_{nm}(G) · exp(-iG·R_I)` per ion on GPU. It is geometry-static: built once at SCF init, invariant under SCF iterations.

The current CPU path in `assemble_aug_density_fine` (chemrust-hamiltonian-core/src/augment/mod.rs:47) calls `apply_q_and_sf` per ion per iteration. `apply_q_and_sf` (q_apply.rs:18) computes:
1. Radial Bessel transforms `Q^L_{rad_n,rad_m}(q)` for all (L, radial pair) — geometry-static.
2. For each G-vector: interpolate radial part, apply Y_LM and phase, multiply by structure factor `exp(-iG·R_I)` — geometry-static.

The cache stores the result of step 2 per ion: a flat `Vec<CudaSlice<CudaComplex>>` where each slice has shape `[n_pairs_ion × n_fine_grid]` (all (n_exp, m_exp) pairs for that ion, flattened).

**Memory budget:** For Cu111_CO: 18 ions × 171 pairs × 437k fine-grid points × 16 bytes = ~21 GB. This exceeds VRAM on Pascal cc 6.1 (typically 12–16 GB). **Revised approach:** Store only the non-zero pairs (those with `rho_nm.norm() > 1e-30` threshold). In practice, the density matrix `ω^I_{nm}` is sparse — only pairs with the same `l` channel are non-zero. For Cu (d-orbitals), this reduces to ~25 non-zero pairs per ion × 18 ions × 437k × 16 bytes ≈ 3 GB. Verify at runtime.

**Alternative if VRAM is insufficient:** Store `Q^I_{nm}(G)` without the structure factor (species-shared, not ion-specific), and apply `exp(-iG·R_I)` per-iteration on GPU. This reduces storage to `n_species × n_pairs × n_fine_grid` (1 species × 171 pairs × 437k × 16 bytes ≈ 1.2 GB for Cu). The structure factor application is a cheap element-wise multiply on GPU.

**Decision at implementation time:** Measure VRAM usage after building the cache. If > 8 GB, switch to the species-shared approach.

**Type definition:**
```rust
pub(crate) struct QSfCache {
    /// Per-ion GPU slices. Each slice is flat [n_pairs × n_fine_grid] Complex128.
    /// `pair_counts[i]` gives the number of (n_exp, m_exp) pairs for ion i.
    pub entries: Vec<QSfIonEntry>,
    pub fine_grid: [usize; 3],  // [ngz, ngy, ngx]
}

pub(crate) struct QSfIonEntry {
    /// Flat GPU slice: [n_pairs × n_fine_grid] CudaComplex.
    /// Q^I_{nm}(G) · exp(-iG·R_I) for all non-zero (n_exp, m_exp) pairs.
    pub q_sf: CudaSlice<CudaComplex>,
    /// Expanded projector pair indices: Vec<(n_exp, m_exp)>.
    pub pairs: Vec<(usize, usize)>,
    pub n_expanded: usize,
}
```

**Build function:**
```rust
pub(crate) fn build_q_sf_cache(
    pots: &PseudopotentialSet,
    cell: &CellGeometry,
    fine_grid: &GVectorGrid,
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<QSfCache, Error>
```

Calls `apply_q_and_sf` (from chemrust-hamiltonian-core) per ion, uploads result to GPU.

**Success Criteria:**
- `build_q_sf_cache` on Cu111_CO fixture completes without error.
- Total GPU memory allocated ≤ 8 GB (log the actual value).
  Source: PHASE_PLAN.md Design Notes (VRAM headroom on Pascal cc 6.1).
- `cargo check --workspace` clean.

**Acceptance:**
```bash
cd /home/tony/programming/chemrust-scf && cargo check --workspace 2>&1 | tail -5
```

---

### TASK-B2: Implement compute_aug_density_gpu

**Kind:** lib-tdd
**Files:**
- `chemrust-scf/src/density.rs`

**Context:**
`compute_aug_density_gpu` replaces the CPU `compute_aug_density_fine` (density.rs:248). It takes GPU-resident `beta_psi_per_ion` and the `QSfCache`, and produces `ρ_aug(r)` on the fine grid.

**Algorithm:**
1. For each ion `I`:
   a. Compute `ω^I_{nm} = Σ_b occ_b · conj(βψ_I)_{n,b} · (βψ_I)_{m,b}` via batched gemm on GPU.
      - `beta_psi_I` shape: `(n_expanded × n_bands)` col-major.
      - `ω^I` shape: `(n_expanded × n_expanded)`.
      - gemm: `ω = conj(βψ) · diag(occ) · βψ^T` — or equivalently, scale each column of `βψ` by `sqrt(occ_b)` then `ω = (scaled_βψ) · (scaled_βψ)^H`.
   b. Contract `ρ_aug(G) += Σ_{n,m} ω^I_{nm} · Q^I_{nm}(G)·exp(-iG·R_I)` using `QSfCache.entries[I].q_sf`.
      - This is a dot product of the flat `ω^I` vector with the `[n_pairs × n_fine_grid]` matrix — a gemv on GPU.
2. Inverse FFT `ρ_aug(G)` → `ρ_aug(r)` on GPU (cuFFT, fine grid).
3. D2H `ρ_aug(r)` → `RealGrid<f64>`.

**Signature:**
```rust
pub(crate) fn compute_aug_density_gpu(
    q_sf_cache: &QSfCache,
    beta_psi_per_ion: &[CudaSlice<CudaComplex>],
    n_expanded_per_ion: &[usize],
    occupations: &[f64],
    n_bands: usize,
    cell: &CellGeometry,
    fine_grid: &GVectorGrid,
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<RealGrid<f64>, Error>
```

**Success Criteria:**
- `‖ρ_aug_gpu − ρ_aug_cpu‖_∞ < 1e-6` on Cu111_CO fixture.
  Source: user-confirmed tolerance (DECISIONS.md AD-1).
- `ρ_aug_gpu` sum matches `ρ_aug_cpu` sum within 1e-4 (integral conservation).
  Source: physics invariant — augmentation charge is conserved.
- `cargo test --workspace` all green.

**Test file:** `chemrust-scf/tests/aug_density_gpu_test.rs` (new file).

**Test code sketch:**
```rust
#[test]
fn aug_density_gpu_matches_cpu_cu111_co() {
    let bin = load_castep_bin("/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.castep_bin");
    // Build beta_psi_per_ion on CPU (from fixture wavefunctions)
    // Run CPU path: compute_aug_density_fine(...)
    // Run GPU path: build_q_sf_cache(...) + compute_aug_density_gpu(...)
    // Compare:
    let max_diff = rho_aug_gpu.iter().zip(rho_aug_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-6,
        "‖ρ_aug_gpu − ρ_aug_cpu‖_∞ = {:.2e}, expected < 1e-6", max_diff);
}
```

**Acceptance:**
```bash
cd /home/tony/programming/chemrust-scf && cargo test aug_density_gpu_matches_cpu_cu111_co 2>&1 | tail -10
cd /home/tony/programming/chemrust-scf && cargo clippy --workspace -- -D warnings 2>&1 | tail -5
```

---

### TASK-B3: Wire QSfCache into ScfIteration and replace CPU aug density call

**Kind:** direct
**Files:**
- `chemrust-scf/src/scf.rs`
- `chemrust-scf/src/density.rs`

**Changes:**
1. Add `q_sf_cache: Option<QSfCache>` field to `ScfIteration`.
2. Build `QSfCache` inside `ScfIteration::new` (or lazily on first `build_v_eff`).
3. In `compute_density_from_wavefunctions` (scf.rs:627), replace the call to `compute_aug_density_fine` with `compute_aug_density_gpu` when `beta_psi_per_ion` is `Some` and `q_sf_cache` is `Some`.
4. Keep `compute_aug_density_fine` as a fallback for the iter-1 fixture path (when `beta_psi_per_ion` is `None`).

**Acceptance:**
```bash
cd /home/tony/programming/chemrust-scf && cargo check --workspace 2>&1 | tail -5
cd /home/tony/programming/chemrust-scf && cargo test --workspace 2>&1 | tail -20
```

---

## GROUP-C: beta_psi GPU residency

**Branch:** `impl/phase-2-perf-opt/group-c`
**Merge target:** `feat/phase-2-perf-opt`
**Depends on:** GROUP-B merged (needs `QSfCache` type and `compute_aug_density_gpu` signature)

### TASK-C1: Change beta_psi_per_ion to GPU-resident in rayleigh_ritz

**Kind:** direct
**Files:**
- `chemrust-scf/src/eigensolver/rayleigh_ritz.rs`
- `chemrust-scf/src/scf.rs`

**Context:**
Currently `rayleigh_ritz` step 5b (rayleigh_ritz.rs:268–318) computes `βψ_I = β_g^H · ψ_new` on GPU, then immediately D2H to host (`clone_dtoh`), converts to `Array2<Complex64>`, and returns `Cpu<Vec<Array2<Complex64>>>`.

The D2H is unnecessary: `compute_aug_density_gpu` (TASK-B2) takes `&[CudaSlice<CudaComplex>]` directly.

**Changes:**
1. In `rayleigh_ritz`, remove the `clone_dtoh` + `Array2` construction in step 5b. Instead, keep `bp_dev: CudaSlice<CudaComplex>` and collect into `Vec<CudaSlice<CudaComplex>>`.
2. Change `RayleighRitzResult` type alias (rayleigh_ritz.rs:33) from:
   ```rust
   Cpu<Vec<ndarray::Array2<num_complex::Complex64>>>
   ```
   to:
   ```rust
   Vec<CudaSlice<CudaComplex>>
   ```
3. Update `ScfIteration.beta_psi_per_ion` field type from `Option<Vec<Array2<Complex64>>>` to `Option<Vec<CudaSlice<CudaComplex>>>`.
4. Update `compute_density_from_wavefunctions` to pass `beta_psi_per_ion` directly to `compute_aug_density_gpu` (no conversion needed).
5. Remove `pcie.d2h_bytes` tracking for `beta_psi` (no longer a D2H transfer).

**PcieAccount impact:** The D2H for `beta_psi` was tracked at rayleigh_ritz.rs:307. Remove that line. The `pcie.d2h_bytes` assertion in `diagonalize` must be updated to reflect the reduced D2H budget.

**Success Criteria:**
- `cargo check --workspace` clean.
- `cargo test --workspace` all green (existing tests must pass).
- The `[AugDensity]` diagnostic line still appears in SCF output (GPU path produces same result as CPU path, verified by TASK-B2 test).

**Acceptance:**
```bash
cd /home/tony/programming/chemrust-scf && cargo check --workspace 2>&1 | tail -5
cd /home/tony/programming/chemrust-scf && cargo test --workspace 2>&1 | tail -20
cd /home/tony/programming/chemrust-scf && cargo clippy --workspace -- -D warnings 2>&1 | tail -5
```

---

## GROUP-D: Documentation

**Branch:** `impl/phase-2-perf-opt/group-d`
**Merge target:** `feat/phase-2-perf-opt`
**Depends on:** GROUP-B merged (ADR-0003 describes the assembly path that GROUP-B implements)

### TASK-D1: CONTEXT.md additions

**Kind:** direct
**Files:**
- `chemrust-scf/CONTEXT.md`

**Changes:**
Add three sections to CONTEXT.md under a new "## Physics Conventions" heading (or append to existing "### Physics" section):

1. **Density unit convention:** "Raw `ρ × Ω` not Ha/Bohr³ for ρ entering Poisson/XC. The `accumulate_density` kernel uses `inv_omega = 1.0` (no Ω division). `solve_poisson` and `compute_pbe_xc` downstream expect this convention. The `.castep_bin` density storage uses the same convention."

2. **FFT axis convention:** "`RealGrid<T>` stores data in Fortran layout `(ngz, ngy, ngx).f()`. `RecipGrid<T>` stores the forward FFT result in the same layout. cuFFT plan dims are `(ngx, ngy, ngz)` (innermost first) to match the scatter formula `iz + ngz*(iy + ngy*ix)`. See failure-patterns.md: `cufft-dim-ordering-and-rr-transpose-layout`."

3. **Augmentation Density domain term:** "ρ_aug(r) = Σ_I Σ_{n,m} ω^I_{nm} · Q^I_{nm}(r). Separate channel from smooth PW ρ_PW. Total ρ = ρ_PW + ρ_aug. The `.castep_bin` density already stores the sum. In the SCF loop: `construct_density_gpu` produces smooth-only ρ_PW; `compute_aug_density_gpu` adds ρ_aug; `build_v_eff_with_energy_impl` sums them before Poisson + XC."

**Acceptance:**
```bash
rg "Density unit convention" /home/tony/programming/chemrust-scf/CONTEXT.md
rg "FFT axis convention" /home/tony/programming/chemrust-scf/CONTEXT.md
rg "Augmentation Density" /home/tony/programming/chemrust-scf/CONTEXT.md
```

---

### TASK-D2: ADR-0003 — USPP density assembly

**Kind:** direct
**Files:**
- `chemrust-scf/docs/adr/0003-uspp-density-assembly.md` (new file)

**Content:**
Write ADR-0003 recording:
- **Context:** The §8 `inv_omega` bug and the §1d FFT plan ordering bug were both architectural invariants that lived only in implementer's heads. The augmentation density path has two channels (smooth PW ρ and ρ_aug) with different unit conventions and different geometry-staticity.
- **Decision:** Smooth PW ρ and ρ_aug live as separate channels with a defined assembly path:
  1. `construct_density_gpu` produces smooth-only ρ_PW (wave-grid convention, raw ρ × Ω).
  2. `compute_aug_density_gpu` produces ρ_aug on the fine grid (same raw convention).
  3. `build_v_eff_with_energy_impl` sums them before Poisson + XC.
- **Rationale:** The channels have different unit conventions, different geometry-staticity (smooth = per-iter, Q·SF = once-per-cell), and the fixture density already contains both. Keeping them separate prevents the class of bug where one channel's convention is silently applied to the other.
- **Consequences:** Any future refactor that merges the channels must explicitly handle the unit convention difference.

**Acceptance:**
```bash
ls /home/tony/programming/chemrust-scf/docs/adr/0003-uspp-density-assembly.md
```

---

## Workspace Validation (after all groups merged)

```bash
cd /home/tony/programming/chemrust-scf && cargo check --workspace 2>&1
cd /home/tony/programming/chemrust-scf && cargo clippy --workspace -- -D warnings 2>&1
cd /home/tony/programming/chemrust-scf && cargo test --workspace 2>&1 | tail -40
```

## Exploration Notes

### E-1: QSfCache memory budget

The plan estimated 1–3 GB for Cu111_CO. Actual calculation:
- 18 ions × 171 expanded pairs × 437k fine-grid points × 16 bytes = **21 GB** (all pairs).
- With sparsity (only same-l pairs non-zero): Cu has 9 s-projectors (l=0) + 9 d-projectors (l=2) per ion in the expanded basis. Non-zero pairs: s-s (9×9=81) + d-d (9×9=81) = 162 pairs per ion. Still ~19 GB.
- **Revised approach:** Store species-shared `Q_{nm}(G)` (no structure factor) — 1 species × 171 pairs × 437k × 16 bytes ≈ 1.2 GB. Apply `exp(-iG·R_I)` per-iteration on GPU (cheap element-wise multiply). This is the recommended implementation path.

This was discovered during exploration. TASK-B1 should implement the species-shared approach by default, with a runtime check.

### E-2: rayon in compute_beta_g — write safety

`compute_beta_g` writes into `beta_g_out[[n_exp, g_idx]]` where `g_idx` is the column index. Each G-vector task writes to a unique column. This is safe for parallel writes if we use raw pointer access or `ndarray`'s `axis_iter_mut`. The cleanest approach: collect `(g_idx, Vec<Complex64>)` pairs in parallel, then scatter into `beta_g_out` serially. The scatter is O(nplw × n_expanded) = O(5k × 18) = 90k ops — negligible.

### E-3: precompute_q_on_grid rayon scope

The `qlnm_g` outer loop has `ll_max + 1` iterations (5 for Cu). With 8 cores, this gives at most 5× speedup on the radial transform step. The dominant cost is the `pairs` construction (wave-grid array per pair), which stays serial. Expected speedup on the full `precompute_q_on_grid`: modest (the radial transform is a small fraction of total time). The main speedup comes from parallelizing the per-ion `compute_beta_g` calls in `VnlBatchData::precompute` (18 ions × serial → 18/8 ≈ 2× speedup on the ion loop).

**Recommendation:** After TASK-A2 and TASK-A3 land, also parallelize the per-ion loop in `VnlBatchData::precompute` (vnl_data.rs:150) using `rayon::iter::IntoParallelIterator`. Each ion's task: `compute_beta_g` + `compute_screened_d_from_fft` + matrix inversions. Per-task memory: `beta_g` array (n_expanded × nplw × 16 bytes = 18 × 5k × 16 = 1.4 MB) + `d_screened` (18 × 18 × 8 = 2.6 KB). Total per thread: ~1.4 MB. With 8 threads: ~11 MB. Well within guard.

This is a TASK-A4 candidate if TASK-A2/A3 don't provide enough speedup.
