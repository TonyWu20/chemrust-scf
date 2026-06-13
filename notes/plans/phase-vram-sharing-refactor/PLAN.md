# VRAM Sharing Refactor — Implementation Plan

**Date**: 2026-06-14
**Branch**: `fix/vram-sharing-refactor`
**Problem**: Cu111_CO spin-polarised FFI cold-start OOM on GTX1080Ti (11 GB). Non-spin uses 7.6 GB VRAM; per-spin duplication of spin-independent VNL data pushes past 11 GB.
**Design**: Hybrid of Design A (Arc-based sharing with lazy init) + Design B (dead Woodbury removal, beta_g sharing).

## Architecture Decision

Use `Arc<KptSharedVnl>` to share spin-independent GPU data across spin channels.
Lazy-init pattern: shared data built on first `step_inner` call for each kpt, reused by subsequent spin calls. No `VnlBatchView<'a>` — existing `VnlBatchData` type is modified in-place to hold an `Arc<KptSharedVnl>`.

## What Is Spin-Independent (moves to shared)

| Field | Est. size (Cu111_CO) | Why |
|-------|---------------------|-----|
| `screening_cache` | ~1.5 GB | Q(G) + structure factors, depends on pots/cell/grid only |
| `screening_cache_fine` | ~1.5 GB | Same, on fine grid |
| `per_ion_beta_g` | ~1.27 GB | β(G) depends on pseudopotential projectors + kpt, not V_eff or spin |
| `per_ion_q` | ~340 KB | Q augmentation matrix, depends on pseudopotential only |
| `per_ion_d0_expanded` | ~340 KB | Unscreened D0, depends on pseudopotential only |
| `screening_h2d_bytes` | trivial | PCI-E accounting |

## What Is Removed (dead code)

| Field | Size | Evidence |
|-------|------|----------|
| `b_concat` | ~327 MB | Constructed in precompute_with_d_override, never read |
| `lu_m` | small | Woodbury LU, only consumed by `davidson_v1` behind `#[cfg(any())]` |
| `lu_ipiv` | tiny | Same |
| `n_total_expanded` | trivial | Only used to dimension dead Woodbury code |

## Implementation Steps

### Phase 1: Remove dead Woodbury code
1. Remove fields from `VnlBatchData`: `b_concat`, `lu_m`, `lu_ipiv`, `n_total_expanded`
2. Remove Woodbury assembly code (vnl_data.rs lines currently building M = Q⁻¹ + B^H·B + LU)
3. Remove `per_ion_q`, `per_ion_beta_flat`, `per_ion_ne` collector vectors from per-ion loop

### Phase 2: Add KptSharedVnl struct
4. Define `KptSharedVnl` in vnl_data.rs with fields:
   - `screening_cache: WaveScreeningCache`
   - `screening_cache_fine: Option<WaveScreeningCache>`
   - `per_ion_beta_g: Vec<CudaSlice<CudaComplex>>`
   - `per_ion_q: Vec<CudaSlice<CudaComplex>>`
   - `per_ion_d0_expanded: Vec<Vec<f64>>`
   - `per_ion_n_expanded: Vec<i32>`
   - `screening_h2d_bytes: usize`
5. Add `shared: Arc<KptSharedVnl>` field to `VnlBatchData`
6. Add `shared: Option<Arc<KptSharedVnl>>` parameter to `precompute_with_d_override`
7. When `shared` is `Some`, skip building screening caches, beta_g, q_matrix; use shared instead
8. When `shared` is `None`, build everything and return the shared Arc inside VnlBatchData

### Phase 3: Wire FFI path
9. Add `shared_vnl: Option<Arc<KptSharedVnl>>` to `KptData` (ffi.rs)
10. In `step_inner`: on first call per kpt, build with `shared=None`; store Arc on KptData
11. On subsequent spin calls: clone Arc, pass `shared=Some(arc)` to `precompute_with_d_override`
12. Update `rescreen_d` to access `self.shared.screening_cache` instead of `self.screening_cache`

### Phase 4: Wire standalone SCF path
13. In `diagonalize_inner` (scf.rs): add `shared_vnl_cache: Vec<Option<Arc<KptSharedVnl>>>` before spin loop
14. Build shared on first spin-0 call per kpt; reuse for spin-1

### Phase 5: FFT grid_buf explicit drop
15. In ffi.rs `step_inner`: `drop(grid_buf)` after `apply_full_hamiltonian` call, before D2H transfers

### Phase 6: Build and test
16. `cargo check` — must compile clean
17. `cargo test --release` — must pass
18. Non-spin NiO FFI cold-start — must converge
19. Spin Cu111_CO FFI cold-start — must not OOM, must converge
