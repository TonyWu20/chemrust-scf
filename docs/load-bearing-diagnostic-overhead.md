# Load-Bearing Diagnostic Overhead: Async Stream-Ordering Bug

**Date:** 2026-06-18
**Status:** Under investigation (racecheck running)
**Commits:** `ad717d7` (gated diagnostics → diverges), `007a294` (ungated → converges), `0f7da96` (regated for investigation)

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
