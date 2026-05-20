# Decisions — Phase 2 Perf Opt

**Date:** 2026-05-21
**Phase:** phase-2-perf-opt

## Fixture Files

| Path | Description |
|------|-------------|
| `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/` | CPU-only CASTEP reference run. Contains `.castep_bin`, `.den_fmt`, `.pot_fmt`, `.bands`, `.castep`. Accessible from test runner. |

## Success Criteria (per goal)

### Goal 1 — GPU QSfCache + compute_aug_density_gpu

- `‖ρ_aug_gpu − ρ_aug_cpu‖_∞ < 1e-6` on Cu111_CO fixture.
  Source: user-confirmed tolerance (GPU path uses same CPU Q·SF computation, reorganized; 1e-6 allows for minor FP reordering).
- Existing iter-2 V_eff range test `|range_iter2 − range_iter1| < 0.2 Ha` stays green.
  Source: PHASE_PLAN.md validation table.
- `cargo test --workspace` all green.
- `cargo clippy --workspace -- -D warnings` clean.

### Goal 2a — Rayon parallelization of VnlBatchData::precompute

- `compute_beta_g` and `precompute_q_on_grid` produce bit-identical results vs serial run on Cu111_CO fixture.
  Source: determinism requirement from PHASE_PLAN.md Design Notes.
- Results match within 1e-12 across multiple runs (rayon non-determinism guard).
  Source: PHASE_PLAN.md Design Notes.
- Per-task memory footprint documented in commit message.
  Source: PHASE_PLAN.md Design Notes (rayon scope guard).

### Goal 3 — Documentation

- CONTEXT.md additions committed: "Density unit convention", "FFT axis convention", "Augmentation Density" domain term.
- ADR-0003 committed: USPP density assembly decision.

## Architectural Decisions

### AD-1: QSfCache lives in chemrust-scf (not chemrust-hamiltonian-core)

**Decision:** `QSfCache` is defined in `chemrust-scf/src/density.rs` (or a new `src/aug_cache.rs`), not in `chemrust-hamiltonian-core`.

**Rationale:** `QSfCache` holds `Vec<CudaSlice<CudaComplex>>` (GPU-resident data), which requires `cudarc` as a dependency. Adding `cudarc` to `chemrust-hamiltonian-core` would couple the physics library to a GPU runtime. The user confirmed: keep GPU types in `chemrust-scf`.

**Consequence:** `QSfCache` is built from `assemble_aug_density_fine`'s inputs (the per-ion `Q·SF` arrays computed by `apply_q_and_sf`) at SCF init, then stored on `ScfIteration`.

### AD-2: beta_psi_per_ion becomes GPU-resident

**Decision:** Replace `beta_psi_per_ion: Option<Vec<Array2<Complex64>>>` (host) with `beta_psi_per_ion: Option<Vec<CudaSlice<CudaComplex>>>` (device) on `ScfIteration`.

**Rationale:** Eliminates the D2H roundtrip in `rayleigh_ritz` step 5b (currently `clone_dtoh` per ion). The `ω^I_{nm}` contraction runs on GPU via batched gemm.

**Consequence:** `rayleigh_ritz` return type changes from `Cpu<Vec<Array2<Complex64>>>` to `Vec<CudaSlice<CudaComplex>>`. The `compute_aug_density_gpu` function takes GPU-resident `beta_psi` directly.

### AD-3: Rayon added to chemrust-hamiltonian-core

**Decision:** Add `rayon` as a dependency to `chemrust-hamiltonian-core`. Parallelize `precompute_q_on_grid` (per-species, outer `ll` loop over radial Bessel transforms) and `compute_beta_g` (per-ion, outer G-vector loop).

**Rationale:** User confirmed: add rayon to core, not just to scf. Both functions have small per-task working sets (log-grid arrays, not fine-grid FFT scratch).

**Constraint:** Do NOT parallelize the per-ion FFT loop inside `compute_screened_d_from_fft` — each thread needs ~28 MB fine-grid FFT scratch (see memory/rayon_fine_grid_memory_overhead.md).

### AD-4: compute_aug_density_gpu signature

**Decision:**
```rust
pub(crate) fn compute_aug_density_gpu(
    q_sf_cache: &QSfCache,
    beta_psi_per_ion: &[CudaSlice<CudaComplex>],
    occupations: &[f64],
    cell: &CellGeometry,
    fine_grid: &GVectorGrid,
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<RealGrid<f64>, Error>
```

The function:
1. For each ion: compute `ω^I_{nm} = Σ_b occ_b · conj(βψ_I)_{n,b} · (βψ_I)_{m,b}` via batched gemm on GPU.
2. Contract `ρ_aug(G) = Σ_I Σ_{n,m} ω^I_{nm} · Q^I_{nm}(G) · exp(-iG·R_I)` using cached `QSfCache`.
3. Inverse FFT to real space on GPU (cuFFT).
4. D2H the final `ρ_aug(r)` array.

### AD-5: QSfCache construction

**Decision:** `QSfCache` is built once at SCF init (inside `ScfIteration::new` or lazily on first `build_v_eff`). It caches `Q^I_{nm}(G) · exp(-iG·R_I)` per ion as `Vec<CudaSlice<CudaComplex>>` (one slice per ion, flattened over all (n,m) pairs × fine-grid points).

**Memory budget check required:** 18 ions × ~171 Q-pairs × 437k Complex128 ≈ 1–3 GB. Verify VRAM headroom before locking in.

### AD-6: Goal 4 deferred

**Decision:** Wall-time regression guard (`#[ignore]`-gated perf gate test) is deferred until after Goals 1+2 are validated. Not in this TASKS.md.

## Domain Terms Validated

- **QSfCache** — per-cell GPU-resident cache of `Q^I_{nm}(G) · exp(-iG·R_I)`. Geometry-static. Lives in `chemrust-scf`.
- **ρ_aug** — USPP augmentation density. Separate channel from smooth PW ρ. Total ρ = ρ_PW + ρ_aug.
- **ω^I_{nm}** — per-ion occupancy matrix `Σ_b occ_b · conj(⟨β_{I,n}|ψ_b⟩) · ⟨β_{I,m}|ψ_b⟩`. Built from GPU-resident `beta_psi_per_ion`.
