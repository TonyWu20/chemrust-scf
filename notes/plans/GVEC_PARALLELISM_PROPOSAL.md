# GPU G-Vector Parallelism — Multi-Stream Banded FFT

**Date**: 2026-06-14
**Status**: Proposal
**Problem**: Single-GPU eigensolver is slower than 16-core CPU for Cu111_CO because cuFFT on a full 437k-point grid is bandwidth-bound and cannot saturate the SMs.

## Root Cause

CASTEP distributes G-vectors across MPI ranks. Each rank FFTs only its own G-vector subset
(437k / 16 ≈ 27k points). 16 concurrent MKL FFTs on 27k-point grids achieve higher aggregate
throughput than 1 cuFFT on 437k points, because:

1. **Superlinear FFT complexity**: O(N log N). 16 × (27k log 27k) < 437k log 437k.
2. **cuFFT bandwidth bottleneck**: 3D FFT on 437k points doesn't saturate GPU SMs; memory
   bandwidth limits throughput. Smaller FFTs fit in L2 cache.
3. **No launch concurrency**: cuFFT serializes batch dimensions internally.

## Design

Replicate CASTEP's G-vector distribution on a single GPU using multiple CUDA streams
and batched FFTs on G-vector slices.

### Phase 1: G-Vector Block Parallelism

Split G-vectors into N equal blocks. Each block runs on its own stream. The FFT grid is
the same size for all blocks, but each block only processes its G-vector subset.

```
┌─────────────────────────────────────────────────────────────┐
│ Stream 0: G[     0..⌊npw/4⌋] → IFFT → V_eff·ψ → FFT → hpsi_slice[0]
│ Stream 1: G[⌊npw/4⌋..⌊npw/2⌋] → IFFT → V_eff·ψ → FFT → hpsi_slice[1]
│ Stream 2: G[⌊npw/2⌋..⌊3npw/4⌋] → IFFT → V_eff·ψ → FFT → hpsi_slice[2]
│ Stream 3: G[⌊3npw/4⌋..npw    ] → IFFT → V_eff·ψ → FFT → hpsi_slice[3]
│                                                                   │
│ Stream 4 (parallel to all above): T·ψ + V_NL·ψ → hpsi_vnl        │
│                                                                   │
│ sync all → scatter_kernel: ∀ig, hpsi[ig] = hpsi_slice[s][ig]    │
│          → hpsi[ig] += hpsi_vnl[ig]                               │
└─────────────────────────────────────────────────────────────┘
```

**New components:**
- `scatter_kernel`: CUDA kernel that scatters per-slice hpsi_slice back to full hpsi buffer.
  Each thread handles one G-vector index. 437k threads, single launch, ~50µs.
- `StreamPool`: pool of N+1 reusable `CudaStream`s for G-vector blocks + V_NL pass.
- `BatchedFftPlan3d` per stream: each stream needs its own FFT plan (cuFFT handles are
  per-stream). Plans are created once per kpt, cached alongside the existing single-stream
  plan.

**Parameters:**
- N = min(4, SM count / 8) ≈ 4 for GTX1080Ti (28 SMs)
- G-vectors per block ≈ npw / N
- Grid: full ngx×ngy×ngz (same for all blocks — each block has the same FFT grid size;
  only the G-vector indices processed differ)

### Phase 2: Overlap V_NL + Kinetic with G-Vector FFT

V_NL (beta·D·beta^H) and kinetic (T·ψ) don't need FFT — they operate directly in G-space.
Run them on a dedicated stream parallel to the G-vector FFT streams:

```
Time →  ════IFFT══╗═══V_eff══╗═══FFT══╗══════════════════════
              ║               ║          ║
        ══════╬═══IFFT═══════╬═══FFT════╬════════════════════
              ║               ║          ║
        ══════╬═══IFFT═══════╬═══FFT════╬════════════════════
              ║               ║          ║
        ══════╬═══IFFT═══════╬═══FFT════╬════════════════════
              ║               ║          ║
        ══════╩═══T+V_NL═════╩═══════════╩═══scatter+combine══
              ↑                           ↑
          all streams                sync event on each stream
          launched here              before scatter
```

**Estimated Cu111_CO timing (174-band, 437k-grid):**

| Operation | 1 stream | 4 G-streams + 1 V_NL stream |
|---|---|---|
| IFFT (per block) | 85ms | 22ms (÷4 smaller FFT) |
| V_eff·ψ | 15ms | 5ms (÷4 points per stream) |
| FFT (per block) | 85ms | 22ms |
| T·ψ + V_NL·ψ | 53ms | 53ms (parallel to FFTs) |
| scatter | — | 2ms |
| **Total H·psi** | **250ms** | **~80ms** |
|---|---|---|
| Per SCF iter (16 calls) | 4.0s | 1.3s |
| 33 SCF iters | 132s | **43s** |

Combined with VNL precompute (~20s) and davidson diagonalization (~10s), total eigensolver
time ≈ **72s** vs CPU's ~290s for the same 33 SCF iterations. **4× speedup over 16-core CPU.**

### Phase 3 (future): Persistent Stream Pool

Allocate streams once at init, reuse across all `step_inner` calls. Stream creation
overhead (~50µs per stream) is negligible compared to FFT time, so this is low priority.

## Implementation

### New types

```rust
struct GBlockConfig {
    n_blocks: usize,
    pw_ranges: Vec<(usize, usize)>,     // [start, end) per block
    streams: Vec<Arc<CudaStream>>,       // n_blocks + 1 (V_NL) streams
    fft_plans: Vec<BatchedFftPlan3d>,   // per G-block (plan per stream)
    scatter_kernel: CudaFunction,
    grid_bufs: Vec<CudaSlice<CudaComplex>>, // per G-block
}
```

### Changes to `step_inner` (ffi.rs)

Current: single FFT, single stream.
Proposed: launch N G-block FFTs + V_NL on separate streams, then scatter-assemble.

### Changes to `apply_full_hamiltonian` (hamiltonian.rs)

Add G-block variant: `apply_full_hamiltonian_gblock` that takes `GBlockConfig`.

### Memory

Each G-block needs its own `grid_buf` (437k / 4 × 174 × 16B ≈ 304 MB each, total ~1.2 GB).
V_NL stream doesn't need a grid buffer. Peak memory increases by ~3 GB over current
allocation but stays within 11 GB for Cu111_CO (current peak ~8 GB, now ~11 GB).

### Algorithmic correctness

Same as single-stream: each G-vector is FFT'd independently. The scatter kernel just
copies results to the correct index. No floating-point differences — cuFFT is deterministic
for identical plan parameters. The stream splits are purely a scheduling optimization.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| cuFFT plan per stream increases init time | Plans created once, cached in KptData alongside existing plan |
| Stream pool memory overhead pushes Cu111_CO past 11 GB | Implement memory budget check; fall back to 2 blocks if >10 GB |
| cudarc multi-stream API immaturity | cuFFT and cuBLAS are inherently stream-safe; only our custom kernels need stream parameter passing (already supported) |
| Synchronization bugs from missing stream events | Add cuEventRecord→cuStreamWaitEvent at all stream merge points; panic on timeout in debug mode |

## Non-Goals

- Multi-GPU (requires NCCL/NVSHMEM, separate hardware)
- G-vector reordering for load balance (Cu111_CO 1 kpt has uniform PW density)
- Wavefunction batching across k-points (separate optimization)
