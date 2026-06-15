# Handoff — 2026-06-15: Bottleneck corrected; profiling fragility documented

**Branch**: `main`
**Status**: Eigensolver bottleneck confirmed ✅ | Profiling fragility documented ⚠️ | Phase 2 overlap → P0

---

## What changed this session

### Eigensolver is the bottleneck — proved by two independent Fortran profilers

The CASTEP Fortran profiler cannot see inside Rust FFI calls. The "untraced gap" under
`electronic_minimisation` IS the Rust GPU eigensolver.

**GPU FFI NiO** (non-spin, 10 SCF iters, 201s total, Final E = −7160.2298 eV):

```
electronic_minimisation                         214.43s
├── EIGENSOLVER (untraced Rust FFI gap)         ~200s   (93%)
├── electronic_initialise                         5.1s
├── electronic_prepare_H (V_eff)                  2.6s
├── density (soft + augment)                      0.9s
├── mixing (Pulay/Kerker)                         0.01s
└── other                                        ~6s
```

**CPU serial Cu111_CO** (spin-polarised, 53 SCF iters, 3516s, Final E = −24111.2814 eV):

```
electronic_minimisation                         3417.5s
├── hamiltonian_diagonalise_ks (EIGENSOLVER)    2959.0s (86.6%)
│   ├── V_NL projector                           819.7s (27.7%)
│   ├── Rotation                                 624.8s (21.1%)
│   ├── H·search                                 584.4s (19.7%)
│   ├── H·ψ full-band                            552.7s (18.7%)
│   ├── Preconditioner                           420.5s (14.2%)
│   ├── S-orthogonalization                      392.8s (13.3%)
│   ├── Subspace diagonalization                 137.0s ( 4.6%)
│   └── Other (copy, dot, init)                 ~155s   ( 5.2%)
├── electronic_prepare_H (V_eff)                 198.0s ( 5.8%)
│   ├── nlpot_calculate_d (D-screening)          149.2s
│   └── locpot_calculate (Hartree+XC)             48.5s
├── density (augment + soft)                     167.3s ( 4.9%)
└── other                                        ~93s
```

**Cross-platform conclusion**: The eigensolver dominates everywhere — 87% CPU, 93% GPU.
The previously claimed "Fortran is the bottleneck" was based on subtracting measured H·psi
kernel time from total SCF time and attributing the residual to Fortran. That residual was
the Rust Davidson eigensolver's non-H·psi overhead (ZHEEVD, ZGEMM, preconditioner, copy,
BetaPhiCache).

### Optimization priority (from CPU eigensolver breakdown + GPU scf_diag per-call timings)

| Priority | Phase | CPU fraction | GPU per-call timing | Action |
|----------|-------|-------------|---------------------|--------|
| **P0** | V_NL (β·β^H projector) | 28% | 246ms (65% of H·ψ) | Phase 2 overlap: V_NL on dedicated stream |
| P1 | Preconditioner | 14% | unknown (TPA+USPP on GPU) | Profile separately; may be GPU bottleneck |
| P2 | H·psi FFT+V_loc | 17% | 38ms FFT + 92ms V_loc | Already fast; low priority |
| P3 | Rotation + S-orth | 34% | cheap on GPU (ZGEMM) | Low priority |

### Profiling fragility in `davidson_diagonalise`

**7 hours of bisect testing** established that `davidson_diagonalise()` (2000+ lines,
dozens of `unsafe` blocks, raw CUDA pointers) is extremely sensitive to code-generation
changes:

- Adding ANY local variable (even a 16-byte `Instant`) changes SCF convergence behavior
  non-deterministically
- Adding a field to `DavidsonResult` changes convergence even when the field is always `None`
- The same profiling code sometimes converges and sometimes diverges on rebuild
- This is a **latent correctness bug** — the function has undefined behavior that manifests
  differently under different compiler optimizations

**What profiling IS safe**: the `scf_diag` feature in `hamiltonian.rs` (per-call H·psi
GPU timings). This code is in a separate function and has been stable across builds.

**The `ffi.rs` builder form also matters**: the `let h_builder = ...; h_builder.call()`
form converges while `apply_full_hamiltonian()...call()` diverges (for the post-diag
H·psi computation). Both produce identical runtime behavior (post_cache is always None).
This is the same code-generation sensitivity as `davidson_diagonalise`.

### GVEC_PARALLELISM_PROPOSAL: rejection confirmed, but for the right reasons

Error 1 (cuFFT cost doesn't scale with G-vector sparsity) is genuinely fatal. The review's
own bottleneck assumption (FFT = 71% of H·psi) was wrong — FFT is 19.5%, V_NL is 65%.
But the rejection stands because Error 1 alone is sufficient.

**The review was three agents, not nine** (MEMORY.md was incorrect). The review document
is now committed at `notes/plans/GVEC_PARALLELISM_REVIEW.md`.

---

## Reference fixtures

| Run | Path | What it proves |
|-----|------|---------------|
| NiO FFI non-spin (GPU) | `/export/.../NiO_no_u_finer_grid_no_spin/` | Eigensolver = 93% of SCF |
| Cu111_CO CPU serial (spin) | `/export/.../Cu111_CO_Single_Point_0614_spin_cpu_serial/` | Eigensolver = 87% of SCF; internal breakdown |

---

## Current state

- FFI path (`ffi.rs` → `davidson.rs` → `hamiltonian.rs`) **converges** for NiO non-spin
- BetaPhiCache works correctly for internal Davidson use (reduces V_NL 246ms→110ms per
  cache-hit call within the eigensolver)
- Returning BetaPhiCache from `DavidsonResult` to `ffi.rs` for post-diag H·psi causes
  divergence — returned as `None` (always fresh V_NL in post-diag path)
- Profiling instrumentation (`scf_diag`) in `hamiltonian.rs` is stable and gives per-call
  GPU timings
- `scf_diag` in `davidson.rs` is **not safely addable** due to code-generation sensitivity

## Key files

| File | What |
|------|------|
| `src/eigensolver/hamiltonian.rs` | H·psi (V_loc FFT + V_NL cuBLAS), `scf_diag` profiling |
| `src/eigensolver/davidson.rs` | Davidson eigensolver, BetaPhiCache lifecycle |
| `src/eigensolver/beta_phi_cache.rs` | β^H·ψ cache (compute_all, invalidate_*) |
| `src/eigensolver/kernels.rs` | CUDA kernels incl. `copy_buffer` (Pascal coherence) |
| `src/ffi.rs` | FFI boundary (⚠️ builder form matters — do not refactor) |
| `notes/plans/GVEC_PARALLELISM_REVIEW.md` | Formal review of GVEC_PARALLELISM proposal |

## Key decisions

1. **V_NL is the #1 optimization target** (28% CPU, 65% of GPU H·ψ) — Phase 2 stream overlap
2. **Porting V_eff/density to GPU saves ≤7%** — not worth prioritizing over eigensolver
3. **Do NOT add code to `davidson_diagonalise`** — the function needs a `cuda-memcheck` audit
   before any modifications. When instrumenting, use `#[inline(never)]` helper functions.
4. **Use CASTEP .profile gaps for profiling** — they already measure everything except the
   Rust FFI call. The gap IS the eigensolver.
5. **The `ffi.rs` builder form must be preserved** — `let h_builder = ... ; h_builder.call()`
   form, not the simple chain form.
