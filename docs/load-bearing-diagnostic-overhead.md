# Load-Bearing Diagnostic Overhead: Async Stream-Ordering Bug

**Date:** 2026-06-18 (root-cause revision: 2026-07)
**Status:** ROOT CAUSE IDENTIFIED (2026-07). Pending one CASTEP-level
verification run of the production (no-`scf_diag`) config.
**Commits:** `ad717d7` (gated diagnostics → diverges), `007a294` (ungated → converges), `0f7da96` (regated for investigation)

## Root cause (2026-07 revision)

All GPU libraries in this pipeline run on ONE user stream:

- cuBLAS: `cublasSetStream_v2` in cudarc `CudaBlas::new`.
- cuSOLVER: `cusolverDnSetStream` in cudarc `DnHandle::new`.
- cuFFT: `cufftPlanSetStream` in cudarc `CudaFft::plan_3d`.
- NVRTC kernels: launched on the same `stream`.

So GPU-GPU ordering is safe, and the `cuMemFreeAsync` cross-stream
hypotheses (1 and 2 below) are dead. The hazard class is HOST reads of
asynchronously-produced data. Two probes pin down which reads are racy:

### Probe 1: `clone_dtoh` into a plain Vec is effectively synchronous

`tests/async_dtoh_probe.rs` shows `cuMemcpyDtoHAsync` into pageable host
memory blocks the host until the DMA data is available (D2H enqueue
wall-time ≈ full DMA time, ~3-4 ms for 16 MB, on both the legacy
default stream and a fresh non-blocking stream; immediate host reads
match in all four configs). D2H/H2D of plain `Vec` buffers is NOT the
racy class.

Note for future work: switching D2H to `PinnedHostSlice` (cudarc
pinned buffers) makes D2H truly async and REINTRODUCES the read race.
Any pinned-buffer optimization must pair every host read with an event
or stream sync.

### Probe 2: `cublasZdotc_v2` host-pointer results are the racy class

`tests/async_dtoh_mechanism.rs` proves it deterministically: a zdotc
enqueued behind a GEMM delay returns to the host immediately; the host
pointer still holds the stale value; after `stream.synchronize()` the
result matches.

The load-bearing racy read in the live block Davidson path was the
initial Rayleigh estimate (`davidson_diagonalise`, per-band `cublasZdotc_v2`
into `eigenvalues`). It is now enqueue-all + one sync + read.
The residual-norms zdotc block (scf_diag only) got the same treatment.

## What the "wall" actually did

The ungated D1/D2 S-norm blocks and the residual-norms block queue
`apply_s_times` GEMMs plus per-band zdotc reads. The GEMMs delay the
host relative to the GPU, which kept the async zdotc results settled by
the time the host read them. Gating the blocks removed the delay, the
host outran the GPU, and the stale zdotc reads corrupted the initial
eigenvalues → divergence. `CUDA_LAUNCH_BLOCKING=1` masked the same
race by serializing every launch.

## Applied fix

- `davidson_diagonalise`: Rayleigh zdotc loop is enqueue-all + single
  `stream.synchronize()` + read. Residual-norms zdotc loop: same.
- D1/D2 S-norm blocks gated behind `#[cfg(feature = "scf_diag")]`
  (true diagnostics; no longer load-bearing).
- `compute_all` kept (it populates the β^H·ψ cache for the next
  iteration); its "load-bearing wall" comment is corrected.
- `SyncAudit` diagnostics (scf_diag): at every host read of GPU data in
  the block Davidson path (entry kinetic/beta/q, Rayleigh zdotc, H_sub
  D2Hs, FFI final D2H). They log `first_read vs post_sync maxdiff`.
  A nonzero value at any site names a remaining race.

## Verification status

- `cargo test` (default + `chebyshev` + `scf_diag` builds): pending.
- CASTEP run with `scf_diag`: read the `[SyncAudit]` lines; expect
  `maxdiff = 0` at every site.
- CASTEP run without `scf_diag` (production config): must converge.
  This is the gate for the D1/D2 gating change.

## Summary

Three outer-loop diagnostic blocks (D1, D2, residual-norm computation) are
unintentionally load-bearing: gating them behind `#[cfg(feature = "scf_diag")]`
causes production divergence.  These diagnostics queue `apply_s_times` GEMMs
on the GPU stream but provide **no synchronization** — cudarc 0.19.7 uses
fully async memory management on modern GPUs (see Mechanism below).
`CUDA_LAUNCH_BLOCKING=1` restores convergence, confirming a timing-dependent
bug at the host-GPU boundary.

## Diagnostic blocks involved

| Block | Location (davidson.rs) | GPU ops queued |
|-------|----------------------|----------------|
| D1 | After A1 ZHEGVD, outer loop | 10 × `apply_s_times` (3 ZGEMMs each) + `cublasZdotc` |
| D2 | After block 0 inner loop | `current_nblock` × `apply_s_times` + `cublasZdotc` |
| Residual norms | After loop exit | 1 × `apply_s_times` (all bands) + per-band `cublasZaxpy`/`cublasZdotc` |

`apply_s_times` computes S·ψ = ψ + β·Q·β^H·ψ (USPP overlap operator),
allocating and freeing temporary GPU buffers for each ion's NL projector.

## cudarc 0.19.7 memory model (key findings)

### Context creation determines async path

```rust
// src/driver/safe/core.rs:78-84
let has_async_alloc = device_attribute(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED) > 0;
```

Modern GPUs (CC 6.0+) report `MEMORY_POOLS_SUPPORTED = 1` → `has_async_alloc = true`.

### Allocation (has_async_alloc = true)

```rust
// src/driver/result.rs:818-825
cuMemAllocAsync(dev_ptr, num_bytes, stream)   // stream-ordered allocation
```

### Drop (has_async_alloc = true)

```rust
// src/driver/safe/core.rs:809-812
if ctx.has_async_alloc {
    result::free_async(self.cu_device_ptr, self.stream.cu_stream)  // → cuMemFreeAsync
}
```

**No `stream.synchronize()` call.**  The free is deferred until the stream
completes.  No implicit host-device synchronization point.

### Drop (has_async_alloc = false) — for reference

```rust
// src/driver/safe/core.rs:813-816
ctx.record_err(self.stream.synchronize());   // ← implicit sync!
ctx.record_err(unsafe { result::free_sync(self.cu_device_ptr) });
```

On older GPUs, **every `CudaSlice` drop synchronizes the stream**.  The bug
would be masked on such hardware.

### clone_dtoh

```rust
// src/driver/safe/core.rs:1630-1641 → result.rs:1035-1047
cuMemcpyDtoHAsync_v2(dst, src, stream)    // async, no sync
```

cudarc exclusively exposes `*_async` variants.  `clone_dtoh` returns a `Vec<T>`
whose contents may be **stale** (GPU hasn't written yet).  Our code reads these
Vecs without syncing — a latent UB that affects all builds equally.

### cublasZdotc_v2

No safe wrapper in cudarc 0.19.7.  Raw FFI, host-pointer mode.  Result written
asynchronously to a stack variable.  Same issue as `clone_dtoh`.

## Experimental results

| Configuration | Result |
|--------------|--------|
| D1/D2/residual-norms **ungated** (production, `007a294`) | **Converges** |
| D1/D2/residual-norms **gated** (`0f7da96`, no scf_diag) | **Diverges** |
| Gated + `CUDA_LAUNCH_BLOCKING=1` | **Converges** |
| `compute-sanitizer --tool racecheck` (gated build) | Running (46+ min/SCF iter, 0 hazards so far) |

## Hypothesis

The bug is a **host-GPU interaction**, not a cross-stream GPU race (racecheck
would have caught that).  `CUDA_LAUNCH_BLOCKING` serializes every kernel launch
so the host cannot outrun the GPU.  The diagnostic GEMMs, while not providing
synchronization, queue enough work on the stream to change the relative timing
between the host's buffer-free operations and the GPU's buffer-use operations.

Candidate mechanisms under investigation:
1. **cuMemFreeAsync ordering**: a buffer freed via `cuMemFreeAsync(ptr, stream_A)`
   might be reclaimed prematurely if stream_A has no pending work, even though
   another stream still references `ptr`
2. **Implicit stream interactions**: cuSOLVER (ZHEEVD, ZPOTRF) may use internal
   streams not tracked by `cuMemFreeAsync`
3. **Host-side pointer invalidation**: a `CUdeviceptr` cached from a freed
   `CudaSlice` is reused in a subsequent kernel launch

## Next steps

1. Wait for `compute-sanitizer --tool racecheck` to complete or produce output
2. If racecheck finds hazards: analyze the specific lines flagged
3. If racecheck completes clean: the bug is definitively host-GPU interaction —
   investigate stream-ordered free semantics and cuSOLVER internal stream usage
4. Alternative diagnostic: `compute-sanitizer --tool initcheck` to catch
   uninitialized memory reads (which `clone_dtoh` without sync would produce)

## Related

- [[davidson-audit-20260604]] — initial UB audit
- [[eigenvalue-explosion-cold-start]] — same class of timing-sensitive bug
- `ad717d7` — commit that introduced the gating
- `eed0c64` — earlier revert of S-orth optimizations (same pattern: optimizations
  that remove GPU work break convergence)
