# GVEC_PARALLELISM_PROPOSAL -- Formal Review

**Document under review**: `GVEC_PARALLELISM_PROPOSAL.md` (2026-06-14)
**Review process**: Adversarial three-agent panel (Agent-A fact-gatherer, Agent-B adversarial reviewer, Agent-J synthesis judge), two rebuttal rounds, unanimous consensus.
**Final verdict**: **REJECT** (the proposal as written is fatally flawed; Phase 2 only is APPROVED as an alternative)
**Review date**: 2026-06-14

---

## 1. Executive Summary

The proposal seeks to accelerate the single-GPU eigensolver by splitting G-vectors into four blocks, each running an independent cuFFT pipeline on its own CUDA stream, overlapping these with a fifth stream performing V_NL via cuBLAS. It claims a 3.1x speedup (250ms to 80ms per H*psi application). The adversarial review found that the proposal contains a fatal structural contradiction: it assumes quarter-sized FFT cost (22ms = 85ms / 4) while simultaneously specifying full-sized FFT grids per block. Since cuFFT time is determined by grid dimensions, not input sparsity, the claimed speedup is physically impossible under the stated architecture. The only viable optimization is Phase 2 alone -- overlapping V_NL (cuBLAS ZGEMM) with the existing single-stream FFT pipeline -- which provides at most ~1.3x speedup assuming verified cuFFT/cuBLAS concurrency on the target GPU.

---

## 2. Factual Errors in the Proposal

### Error 1 (FATAL): FFT cost claimed as quarter-sized while grid is full-sized

**Proposal claim** (lines 53-54, 80): "Grid: full ngx x ngy x ngz (same for all blocks)" AND "IFFT per block: 22ms (div-4 smaller FFT)".

**Reality**: The Rust code at `fft.rs:184` creates `cufftPlanMany` with `odist = nx * ny * nz = grid_size` and `batch = n_bands`. The cuFFT plan is parameterized by the full grid dimensions; it traverses the complete butterfly network over all `grid_size` points per batch element regardless of how many input entries are non-zero. There is no sparse-FFT path in cuFFT. The 22ms figure (85/4) is fabricated -- it has no derivation from the stated architecture, no empirical measurement, and no theoretical justification.

**Impact**: FATAL. This contradiction alone invalidates every timing number in the proposal's table and the claimed 3.1x speedup.

**Verification**: `chemrust-scf/src/device/fft.rs:184-196` -- `odist = (nx * ny * nz) as usize`, `cufftType::CUFFT_Z2Z`, `batch`. The transform size is fixed to the full grid. `kernels.rs:73-86` -- `veff_multiply` iterates over `n_bands * grid_size` elements (full grid). The FFT at `hamiltonian.rs:134,149` calls `c2c_inverse_inplace` / `c2c_forward_inplace` on a buffer of `n_bands * grid_size` elements.

---

### Error 2 (FATAL): CASTEP MPI distributes the GRID, not G-vectors

**Proposal claim** (lines 9-11): "CASTEP distributes G-vectors across MPI ranks. Each rank FFTs only its own G-vector subset (437k / 16 approx 27k points). 16 concurrent MKL FFTs on 27k-point grids achieve higher aggregate throughput."

**Reality**: CASTEP uses a 3-stage column decomposition with interleaved `comms_transpose` calls, not G-vector subset decomposition. Verified at:

- `fft.mkl.F90:212`: `DftiCreateDescriptor(plan_p(...), DFTI_DOUBLE, DFTI_COMPLEX, 1, ng(i,igrid))` -- the `1` parameter means **1D** FFT descriptors, not 3D.
- `fft.mkl.F90:215`: `DFTI_NUMBER_OF_TRANSFORMS = num_columns(i, iwt, igrid)` -- each plan processes `num_x_columns` independent 1D transforms of length `ngx`, not one 3D transform.
- `fft.mkl.F90:462-470`: Three-stage pipeline -- 1D FFT along x, `comms_transpose('XY')`, 1D FFT along y, `comms_transpose('YZ')`, 1D FFT along z.
- `basis.f90:1465`: `max_x_columns = (ngy*ngz + ngy + 2 + (num_in_gv_group - 1))/num_in_gv_group + 2` -- determines per-node column partition.
- `basis.f90:2941`: `max_grid_points = max(max_x_data, max_y_data, max_z_data)` -- per-node GRID BUFFER size, NOT G-vector count.
- `basis.f90:188`: `num_plane_waves_kp` is "The number of plane wave components on the **local node**" -- confirming G-vectors are per-node, but this is a consequence of grid decomposition, not the mechanism.

The proposal's "27k" figure is approximately `max_grid_points` (grid buffer size per rank), which coincidentally approximates `n_pw / 16` for this specific system (Cu111_CO, 54x90x90, 16 ranks). These are distinct quantities. The FFT operates on `max_grid_points` grid entries (line `basis.f90:4490`: `zlaset('A', max_grid_points, ...)` zeroes the grid buffer), not on `num_plane_waves_kp` G-vector coefficients.

**Impact**: FATAL. The proposal's central analogy (replicating MPI G-vector distribution on a single GPU) is based on a misunderstanding of what CASTEP actually distributes. The MPI speedup comes from physically smaller FFT grids (column decomposition), not from processing fewer G-vectors per rank on a full-sized grid.

---

### Error 3 (FATAL): Memory calculation is internally inconsistent

**Proposal claim** (line 124): "Each G-block needs its own grid_buf (437k / 4 x 174 x 16B approx 304 MB each, total approx 1.2 GB)."

**Reality**: The formula `(437k / 4) * 174` divides `grid_size` by 4 while keeping `n_bands = 174`. This is algebraically identical to `(174 / 4) * 437k` (band-level splitting). It implicitly assumes quarter-sized grids OR quarter-band batches, both of which contradict the stated architecture (G-vector splitting with full grid, all bands per block).

Under the architecture as diagrammed (full grid, all bands, G-vector range split):
- Per block: `grid_size * n_bands * 16 bytes = 437,400 * 174 * 16 = 1.22 GB`
- Total for 4 blocks: **4.88 GB** (not 1.2 GB)
- Plus psi (1.22 GB) + hpsi (1.22 GB) + hpsi_vnl_slices + V_NL beta/D = ~7.5 GB additional
- Total peak: exceeds GTX 1080 Ti 11 GB VRAM limit

**Impact**: FATAL. The proposal's own memory budget (11 GB peak) does not close under its stated architecture.

---

### Error 4 (SERIOUS): cuFFT determinism is not guaranteed

**Proposal claim** (lines 130-132): "No floating-point differences -- cuFFT is deterministic for identical plan parameters. The stream splits are purely a scheduling optimization."

**Reality**: NVIDIA cuFFT documentation (CUDA Toolkit Documentation, cuFFT Library, "Results Reproducibility" section) states: "cuFFT results are not guaranteed to be bitwise reproducible across different GPU architectures or even across different runs on the same GPU." This applies to the same plan on the same hardware. Four independent cuFFT plans on four different streams may produce results differing in the last 1-3 mantissa bits due to different thread/warp scheduling, different internal scratch buffer states, and different floating-point accumulation orders within the butterfly network.

The existing cuFFT roundtrip test at `hamiltonian.rs:608-694` shows `worst_rel < 1e-6` for a single-plan roundtrip, which establishes acceptable numerical accuracy but NOT bitwise determinism (which would require `< 1e-16`).

**Impact**: SERIOUS but not fatal on its own. The numerical differences are likely small enough not to affect SCF convergence (the existing tests show `< 1e-6` roundtrip error), but the claim of "no floating-point differences" is false and the risk is unquantified.

---

### Error 5 (SERIOUS): Arithmetic inconsistency in timing table

**Proposal claim** (lines 78-88): Single-stream total = 250ms, multi-stream total = 80ms. But the proposal's own per-component numbers (22+5+22 = 49ms per block, with 53ms V_NL parallel) give max(49, 53) + 2 = 55ms, not 80ms. Neither 55ms nor 80ms is derivable from the stated architecture.

**Reality**: Under the architecture as diagrammed (full grid per block, all bands), the correct per-block timing is:
- zero_buffer: ~2.5ms (`kernels.rs:17-21`, `n_bands * grid_size` elements)
- scatter: ~0.5ms (`kernels.rs:58-71`, `n_bands * n_pw/4` elements)
- cuFFT IFFT: ~85ms (`fft.rs:184-196`, full grid x n_bands, unchanged from single-stream)
- V_eff multiply: ~15ms (`kernels.rs:73-86`, full grid x n_bands, unchanged)
- cuFFT FFT: ~85ms (unchanged)
- gather_add_kinetic: ~2ms (`kernels.rs:88-104`, iterates over ALL `n_bands * n_pw`, not per-block subset -- this is 4x overhead when 4 blocks each call it)
- Per-block total (sequential): ~190ms
- With 4 concurrent blocks on a single memory bus: cuFFT plans serialize, wall time ~195-211ms (worse than single-stream due to 4x zero_buffer overhead of ~10ms total, 4x gather overhead of ~8ms total, and stream synchronization overhead)
- V_NL parallel: 53ms
- Total: max(~200ms, 53ms) + 2ms = ~202ms, speedup = 240/202 = 1.19x

**Impact**: FATAL. The multi-block approach provides no meaningful speedup and may be slower than single-stream.

---

## 3. CASTEP vs Proposal: Algorithmic Comparison

| Aspect | CASTEP MPI (16 ranks) | Proposal (4 streams) | Rust single-stream (current) |
|--------|----------------------|---------------------|------------------------------|
| **Decomposition** | Column (rod) decomposition of 3D grid | G-vector index range split | None (full grid, one stream) |
| **FFT type** | 3-stage 1D transforms with interleaved `comms_transpose` | Independent 3D cuFFT plans | Single 3D cuFFT plan |
| **Grid per unit** | `max_grid_points` ~ 27k grid points per rank | Full `ngx*ngy*ngz = 437k` per stream (contradicts cost model) | Full 437k grid |
| **FFT cost per unit** | `num_x_columns * ngx * log(ngx)` + transpose + y-stage + transpose + z-stage | Full `grid_size * log(grid_size)` per batch element, same as single-stream | Full `grid_size * log(grid_size)` per batch element |
| **G-vectors per unit** | `num_plane_waves_kp(nk)` ~ 27k PWs per rank | `n_pw / 4` ~ 109k PWs per stream | All 437k PWs |
| **How speedup arises** | Physically smaller grid -> less FFT work; 16x parallelism | Claim: fewer G-vectors -> faster FFT (FALSE); 4x concurrency (unverified) | Baseline |
| **Key source line** | `fft.mkl.F90:212` -- 1D descriptors, `basis.f90:2941` -- `max_grid_points` | `fft.rs:184` -- `odist = nx*ny*nz`, `batch = n_bands` | `hamiltonian.rs:66-168` |
| **Correctness of analogy** | N/A (ground truth) | **INVALID** -- confuses grid decomposition with G-vector decomposition | N/A |

**The critical difference**: CASTEP's MPI speedup comes from each rank processing a physically smaller FFT grid (column decomposition reduces `max_grid_points` from 437k to ~27k per rank). The proposal keeps the grid at 437k per stream while claiming the FFT cost drops to 1/4. These are mutually exclusive. If the grid is quarter-sized, the FFT cost does drop (approximately linearly for small grids where compute dominates). If the grid is full-sized, the FFT cost is unchanged regardless of how many G-vectors populate it.

---

## 4. Performance Analysis

### 4.1 Corrected timing for the proposal's architecture as diagrammed

| Component | Single-stream | 4 G-blocks (stated arch) | Notes |
|-----------|--------------|--------------------------|-------|
| zero_buffer | 2.5ms | 10ms (4x) | 4 blocks x full grid per block |
| scatter | 2ms | 2ms (4 concurrent x 0.5ms each) | 1/4 elements per block |
| cuFFT IFFT | 85ms | 85ms per block (serialized or minimal concurrency) | Same grid_size, same batch |
| V_eff multiply | 15ms | 15ms per block | Same grid_size, same batch |
| cuFFT FFT | 85ms | 85ms per block | Same grid_size, same batch |
| gather_add_kinetic | 2ms | 8ms (4x) | `kernels.rs:94` loops over ALL `n_bands * n_pw` |
| V_NL (cuBLAS) | 53ms | 53ms (parallel) | Independent stream |
| scatter assembly | -- | 2ms | New kernel |
| **Total** | **~240ms** | **~195-211ms** | Speedup: 1.14-1.23x |

The gather kernel at `kernels.rs:88-104` is the hidden killer: `while (tid < n_bands * n_pw)` iterates over ALL bands and ALL G-vectors, not just those belonging to the calling block. Four blocks each launch this kernel at full size, producing 4x the gather work.

### 4.2 Phase 2 only (V_NL/FFT overlap)

| Component | Time |
|-----------|------|
| V_loc pipeline (IFFT + V_eff + FFT + scatter/gather) | ~185ms |
| V_NL (cuBLAS ZGEMM) | ~53ms |
| Sync + final add | ~2ms |
| **Total (perfect overlap)** | **~187ms** |
| **Total (50% overlap)** | **~214ms** |
| **Total (0% overlap)** | **~240ms** |

Speedup range: 1.12x to 1.28x, depending on measured cuFFT/cuBLAS concurrency on the GTX 1080 Ti Pascal architecture.

### 4.3 Why the proposal's 3.1x is impossible

The proposal's claimed speedup of 3.1x requires either:
1. Quarter-sized FFT grids per block (contradicting lines 53-54), OR
2. FFT cost that scales with non-zero input count (no such mechanism exists in cuFFT), OR
3. 4x concurrency among bandwidth-bound cuFFT plans on a single memory bus (Maxwell/Pascal/Volta architectures serialize large concurrent kernels)

None of these conditions hold under the stated architecture. The maximum achievable speedup from the proposal's ideas, even with ideal concurrency, is ~1.3x from Phase 2 alone.

---

## 5. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Architecture invalid** -- full grid vs quarter FFT cost contradiction | CRITICAL | Reject G-vector decomposition. Do not implement multi-FFT-stream. |
| **Memory budget exceeded** -- 4 full-grid buffers exceed 11GB VRAM | CRITICAL | Would require band-level splitting (unverified) or grid decomposition (major redesign). Do not implement. |
| **Gather kernel 4x overhead** -- `kernels.rs:94` destroys any concurrency gain | CRITICAL | Requires kernel redesign to skip out-of-range G-vectors. Complex and fragile. |
| **cuFFT/cuBLAS concurrency unverified** on Pascal GTX 1080 Ti | HIGH | Measure before implementing Phase 2. If <20% overlap, Phase 2 benefit drops below 1.1x and may not justify complexity. |
| **cuFFT non-determinism** -- 4 plans may produce slightly different outputs | MEDIUM | Existing roundtrip test (`hamiltonian.rs:608-694`) bounds single-plan noise at <1e-6 relative. Multi-plan variation unmeasured. Not expected to affect SCF at <1e-12 per-element threshold, but must be measured. |
| **Nyquist-plane G-vector splitting** -- conjugate pairs in different blocks | LOW | For C2C FFT on non-gamma k-points, Nyquist-conjugate pairs map to distinct grid indices and do not require co-location. The `nyq_*` parameters in `kernels.rs:58-71` are dead code (development artifact from C2R context). Safe for Cu111_CO, but document limitation for gamma-point. |
| **cuFFT plan creation time** -- one-time per-kpt cost | LOW | Amortized across SCF iterations (30+). Even 500ms per plan is negligible vs 30+ seconds of SCF. However, internal workspace memory per plan is not publicly documented and may push VRAM limits. |
| **V_eff concurrent read contention** -- 4 streams reading same 3.5MB array | LOW | V_eff fits in aggregate L2 cache. Crossbar traffic increase is small relative to cuFFT bandwidth dominance. |

---

## 6. Final Recommendation

**Verdict: REJECT the proposal as written.**

The proposal contains three fatal errors that cannot be resolved through modification:

1. **The FFT cost contradiction** (Error 1): The proposal simultaneously claims full-sized grids and quarter-sized FFT costs. These are mutually exclusive. No amount of restructuring can reconcile them without abandoning the stated architecture.

2. **The CASTEP analogy is invalid** (Error 2): The proposal's central motivating claim -- that CASTEP achieves speedup by distributing G-vectors -- is incorrect. CASTEP distributes the FFT grid itself via column decomposition, which reduces the physical grid size per rank. The proposal's approach (keeping the full grid while splitting G-vectors) does not replicate this mechanism and cannot achieve the claimed speedup.

3. **The memory budget does not close** (Error 3): Under the stated architecture, peak VRAM exceeds the GTX 1080 Ti 11 GB limit. The proposal's own memory formula implies a different (band-level splitting) architecture.

---

## 7. Approved Alternative: Phase 2 Only

**What to implement**: Overlap the existing single-stream V_loc FFT pipeline with V_NL (cuBLAS ZGEMM) on a separate stream.

**Required changes**:
1. Create one additional `CudaStream` for V_NL
2. Allocate one additional `hpsi_vnl` buffer (`n_pw * n_bands * 16 bytes = ~1.22 GB` for Cu111_CO)
3. In `apply_full_hamiltonian` at `hamiltonian.rs:175-236`:
   - Launch `apply_v_loc_hamiltonian` on stream_0 (current behavior)
   - Launch `apply_v_nl_hamiltonian` on stream_1 immediately after, while V_loc pipeline runs
   - `cuEventRecord` on both streams
   - `cuStreamWaitEvent` on stream_0 for stream_1's completion
   - Accumulate `hpsi_vnl` into `hpsi` on stream_0
4. No changes to FFT plans, no G-vector splitting, no scatter assembly kernel, no new kernel types.

**Expected speedup**: 1.12x to 1.28x (240ms to 187-214ms), depending on measured cuFFT/cuBLAS overlap on GTX 1080 Ti.

**Pre-conditions**:
- Profile current `apply_full_hamiltonian` to get real component timings (cuFFT IFFT, V_eff, cuFFT FFT, V_NL, scatter/gather).
- Benchmark cuBLAS ZGEMM + cuFFT C2C concurrency on separate streams on the actual GTX 1080 Ti. If overlap is <20%, the benefit of Phase 2 drops below 1.1x and may not justify the added complexity and VRAM.
- Verify that the additional `hpsi_vnl` buffer stays within the 11 GB VRAM budget.

---

## 8. What Not to Implement

| Approach | Why not |
|----------|---------|
| G-vector range splitting | FFT cost unchanged (full grid per block); gather kernel 4x overhead (`kernels.rs:94`); zero_buffer 4x overhead; no speedup possible |
| Multi-FFT-plan concurrency | 4 bandwidth-bound cuFFT plans serialize on single memory bus; no speedup |
| Band-level splitting (n_bands/4 per block, full grid each) | cuFFT batch scaling at batch=43 is unverified; may be worse than single batch=174 due to amortized launch overhead; does not overlap with V_NL; added complexity for unmeasured gain |
| True grid decomposition (like CASTEP MPI) | Requires partial FFT grids, column transposes between streams, kernel changes throughout the pipeline. Major architectural redesign. No Rust-side grid decomposition infrastructure exists. |

---

## 9. Evidence Index

### Primary sources verified

| File | Lines | What it proves |
|------|-------|---------------|
| `fft.mkl.F90` | 212 | CASTEP creates 1D FFT descriptors (column decomposition), not 3D |
| `fft.mkl.F90` | 215 | `DFTI_NUMBER_OF_TRANSFORMS = num_columns(...)` -- batch of 1D transforms per dimension |
| `fft.mkl.F90` | 462-470 | 3-stage distributed FFT with `comms_transpose` interleaved |
| `fft.mkl.F90` | 437-441 | Serial branch: `DftiComputeForward(plan3d_s_for, a)` -- single 3D plan |
| `basis.f90` | 188 | `num_plane_waves_kp` = "on the local node" (per-node, not global) |
| `basis.f90` | 1465 | `max_x_columns` formula -- per-node column count from ceil division |
| `basis.f90` | 2941 | `max_grid_points = max(max_x_data, max_y_data, max_z_data)` -- per-node grid buffer size |
| `basis.f90` | 3503 | Serial case: `max_grid_points = total_grid_points` (full grid) |
| `basis.f90` | 4490 | `zlaset('A', max_grid_points, ...)` -- zeroes per-node buffer, not full grid |
| `basis.f90` | 4494-4499 | Scatters `num_plane_waves_kp` PWs into `grid` of size `max_grid_points` |
| `basis.f90` | 11250-11294 | `pw_grid_index` assigned from local `point` counter, range `1..num_recip_columns*ngx` |
| `fft.rs` | 184-196 | `odist = nx * ny * nz`, `batch = n_bands` -- cuFFT on full grid per batch element |
| `kernels.rs` | 17-21 | `zero_buffer` writes `n_bands * grid_size` elements (full grid) |
| `kernels.rs` | 73-86 | `veff_multiply` iterates `n_bands * grid_size` (full grid) |
| `kernels.rs` | 88-104 | `gather_add_kinetic` iterates ALL `n_bands * n_pw` (not per-block) -- 4x overhead source |
| `kernels.rs` | 58-71 | `scatter_pw_to_grid_nyq` accepts `nyq_*` but never uses them (dead code, C2C safe) |
| `hamiltonian.rs` | 608-694 | cuFFT roundtrip test: `worst_rel < 1e-6` for single plan (accuracy, not determinism) |
| `hamiltonian.rs` | 462-599 | ZGEMM accumulation order test: `max_rel < 1e-9` for cuBLAS dot products |

### NVIDIA documentation (paraphrased)

- **cuFFT Results Reproducibility**: "cuFFT results are not guaranteed to be bitwise reproducible across different GPU architectures or even across different runs on the same GPU." Source: CUDA Toolkit Documentation, cuFFT Library, Reproducibility section.

---

## 10. Review Process Notes

This review was conducted by a three-agent adversarial panel (Agent-A: fact-gatherer and primary assessor, Agent-B: adversarial reviewer, Agent-J: synthesis judge) across two rebuttal rounds. All three agents converged on the REJECT verdict with full consensus. The only remaining quantitative disagreement (190ms best-case vs 211ms worst-case for the multi-block architecture) does not affect the recommendation and is within the noise of unmeasured parameters.

**Agent-A final score**: 8.5/10 -- comprehensive analysis, promptly conceded cuFFT determinism error.
**Agent-B final score**: 9.0/10 -- identified the gather kernel 4x overhead (`kernels.rs:94`) which mathematically proves the multi-block approach cannot deliver speedup.

No third round is needed. Proceed to Phase 2 profiling and implementation.

---

## 11. Post-Review: What Profiling Revealed (2026-06-14)

After the review, profiling instrumentation was added to `apply_full_hamiltonian` (see `hamiltonian.rs:389-480`) and the following was measured on actual hardware (GTX 1080 Ti, Cu111_CO, n_pw=60067, n_bands=174, grid=54×90×90):

### 11.1 Real per-H·psi component timings

| Component | Review estimate | Measured | Error |
|-----------|----------------|----------|-------|
| cuFFT IFFT | 85ms | **38ms** | Review overestimated 2.2× |
| cuFFT FFT | 85ms | **38ms** | Review overestimated 2.2× |
| V_eff multiply | 15ms | **8.6ms** | — |
| V_NL (cuBLAS) | 53ms | **246ms** | Review underestimated 4.6× |
| **Total H·psi** | 240ms | **350ms** | — |
| **FFT fraction** | 71% | **19.5%** | Review completely wrong about bottleneck |

**Finding**: The review (and proposal) both assumed FFT was the bottleneck. In reality, V_NL dominates at **65% of H·psi time**, and the FFT is only **19.5%**. Every optimization targeting FFT (Phase 1 G-vector decomposition, Phase 3 grid decomposition) targets a minority component.

### 11.2 H·psi is only 7% of SCF iteration time

Per SCF iteration for Cu111_CO (240s wall clock):
- **~138 H·psi calls** totaling **~17 seconds**
- Remaining **~223 seconds** in CASTEP Fortran: density FFT, density mixing (Pulay), Hartree+XC potential, D-screening, wavefunction→density transforms, MPI communication

| Call type (n_bands) | Count/iter | Each (ms) | Total (s) |
|---|---|---|---|
| 174 (full-band) | ~4 | 337 | 1.3 |
| 26 (search-direction) | ~100 | 126 | 12.6 |
| 18 | ~19 | ~100 | 1.9 |
| other | ~15 | ~80 | 1.2 |
| **Total** | **~138** | | **~17s** |

**Finding**: Even a miracle 10× speedup in H·psi would only save 15 seconds in a 240-second iteration (6%). The Phase 2 overlap (1.12–1.28× on H·psi) saves ~2 seconds — **invisible against the 240s baseline**.

### 11.3 Phase 2 overlap estimate was wrong

The review estimated V_NL at 53ms and V_loc at 185ms, giving a 1.12–1.28× overlap benefit. Real numbers:
- V_loc: **92ms** (FFT 76ms + kernels 16ms, much faster than estimated)
- V_NL: **246ms** (4.6× larger than estimated)

Overlap: `max(92, 246) + 2 = 248ms` vs sequential `92 + 246 = 338ms` → **1.36× speedup on H·psi, ~0.7s saved per SCF iter**. Still negligible.

### 11.4 BetaPhiCache investigation

The BetaPhiCache (skipping β^H·ψ ZGEMM when ψ hasn't changed between Davidson outer iterations) was wired up and tested. Key findings:
- Cached β^H·ψ is **bitwise identical** to fresh ZGEMM (verified at `{:.15e}` precision across all ions)
- Copy-engine→compute-engine coherence gap on Pascal GPUs: `cudaMemcpyAsync` (copy engine) writes are not coherent with subsequent `cublasZgemm` (compute engine) reads on the same user stream. Fixed by replacing D2D memcpy with a compute-engine `copy_buffer` CUDA kernel (`kernels.rs:28-36`).
- V_NL reduced from 246ms to 110ms per cache-hit call (55% reduction)
- Only 2–4 calls per SCF iteration benefit → total saving ~0.5s/iter

### 11.5 CPU vs GPU for Cu111_CO

| | CPU (1 core, serial) | GPU (GTX 1080 Ti, FFI) |
|---|---|---|
| SCF iter time | **118s** | 240s |
| Speed | **2× faster** | — |

The GPU eigensolver is faster per H·psi call, but the Fortran-side SCF cycle (density, mixing, V_eff) dominates runtime, and the GPU FFI path adds CPU↔GPU transfer overhead. The serial CPU is faster overall for this system.

### 11.6 Updated recommendations

**The review's REJECT verdict stands.** All H·psi-level optimizations (Phase 1, Phase 2, Phase 3, BetaPhiCache) are marginal because H·psi is only 7% of SCF runtime. The path to significant GPU speedup requires:

1. **Move the full SCF cycle to GPU** — the standalone `scf.rs` path already does density construction, mixing, and V_eff assembly on GPU. Porting the FFI path to use the GPU-resident SCF cycle would eliminate Fortran-side bottlenecks.
2. **Profile the CASTEP Fortran SCF** — understand where the 223 CPU seconds go (density FFT, Pulay mixing, Hartree potential, XC, D-screening).
3. **Consider multi-GPU or newer GPU hardware** — the GTX 1080 Ti (2017, Pascal, 11 GB) is memory-constrained and lacks hardware features (tensor cores for FP64, improved copy/compute engine coherence) available on newer architectures.

**The review's own timing estimates (85ms FFT, 53ms V_NL) were also wrong** — based on the proposal's fabricated numbers rather than measurement. This is a meta-lesson: all performance claims in the proposal AND the review should have been gated on profiling before any implementation was attempted.
