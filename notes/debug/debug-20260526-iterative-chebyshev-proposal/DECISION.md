# Decision: Pivot from Iterative Chebyshev to Band-by-Band CG

**Date**: 2026-05-26  
**Context**: Resolution of SCF cascade issue in chemrust  
**Decision**: Abandon iterative Chebyshev + per-band RQ proposal; implement band-by-band CG instead

---

## Summary

After thorough investigation including:
- Analysis of CASTEP recovery from corrupted states (both Davidson and Chebyshev)
- Review of ChASE library paper on Chebyshev filtering
- Critical assessment of orthogonality preservation
- Realistic performance analysis with actual timing data

We have decided to **pivot from iterative Chebyshev filtering to band-by-band conjugate gradient** as the solution to the SCF cascade problem.

---

## Why Iterative Chebyshev Was Abandoned

### 1. Orthogonality Loss is Fundamental

The Di Napoli & Wu paper (ChASE library) proves that Chebyshev filtering produces highly ill-conditioned subspaces:

```
κ₂(p_m(A)·V) ≤ η·|ρ₁|^m
```

For filter degree m=8, condition numbers reach **10^8** in early iterations. This means:
- Chebyshev filtering DESTROYS orthogonality, not preserves it
- Explicit S-orthonormalization is required between every filter pass
- Our proposal's claim that "Chebyshev preserves orthonormality" was **fundamentally wrong**

### 2. Performance is Unacceptable Without Full GPU Acceleration

**Current timing** (partially GPU-accelerated):
- Single-sweep: ~120 seconds per SCF step
- 26 iterations: ~3120 seconds (52 minutes) per SCF step
- Total for 40 SCF iterations: **26-43 hours** (unacceptable)

**With full GPU acceleration** (estimated):
- Single iteration: ~300 ms
- 26 iterations: ~7.8 seconds per SCF step
- Total for 40 SCF iterations: ~5 minutes (acceptable)

**But**: Full GPU acceleration is a large upfront investment with no guarantee of success.

### 3. Algorithm is Unproven for USPP

- ChASE paper validates iterative Chebyshev for **standard eigenproblems** (Ax = λx)
- chemrust needs **generalized eigenproblems** (Hψ = λSψ) with USPP
- S-orthonormalization adds complexity and cost
- No reference implementation exists for this case

### 4. High Implementation Risk

- Requires full GPU acceleration to be viable
- Requires validation of S-orthonormality preservation
- Requires proving per-band RQ convergence for degenerate manifolds
- Debugging would be difficult (algorithm vs implementation issues)

---

## Why Band-by-Band CG is the Right Choice

### 1. Proven by CASTEP

CASTEP recovers from **both** Davidson and Chebyshev corrupted states:
- Davidson .check: 33 SCF iterations to converge
- Chebyshev .check: 115+ SCF iterations, stabilizing around reference energy

CASTEP does **19-26 band-by-band CG passes per SCF step** (not a fixed number — iterates until convergence).

### 2. Simpler Algorithm

Band-by-band CG for each band:
```
while ||residual|| > tolerance:
    1. Compute gradient: g = H·ψ - λ·S·ψ
    2. Apply preconditioner: t = P^(-1)·g
    3. Line search: minimize Rayleigh quotient along ψ + α·t
    4. Update: ψ ← ψ + α·t
    5. Orthogonalize against other bands
```

**No subspace accumulation, no RR rotation, constant memory.**

### 3. Natural GPU Fit

- One band at a time: constant memory footprint
- Each band is independent: natural parallelization
- Gradient computation: reuse existing H·ψ infrastructure
- Line search: simple 1D optimization (GPU-friendly)
- Orthogonalization: batched dot products + axpy

### 4. Incremental Development Path

1. **Phase 1**: Implement CPU version
   - Reuse existing gradient computation
   - Implement line search
   - Validate against CASTEP

2. **Phase 2**: Move to GPU
   - GPU gradient computation
   - GPU line search
   - GPU orthogonalization

3. **Phase 3**: Optimize
   - Batch operations
   - Minimize CPU-GPU transfers
   - Profile and tune

### 5. Lower Risk

- Algorithm is proven (CASTEP)
- No orthogonality concerns (explicit orthogonalization per band)
- No condition number issues
- Easier to debug (band-by-band is sequential)

---

## Key Findings from Investigation

### CASTEP Source Code Analysis

From `hamiltonian.F90`:
```fortran
outer_loop: do iteration=1,max_iterations(2)
  if(all(band_converged).and.prev_all_converged) exit
  
  inner_loop: do elecmin_step=1,max_iterations(1)
    ! Band-by-band CG optimization
    ...
  end do inner_loop
end do outer_loop
```

**Key insight**: CASTEP iterates until convergence, not a fixed number of iterations. The "26 iterations" we observed is an emergent result.

### ChASE Paper Findings

The Di Napoli & Wu paper on Chebyshev filtering proves:
1. Filtered vectors have exponentially growing condition numbers
2. Explicit QR factorization is required between every filter pass
3. ChASE uses CholeskyQR variants to handle high condition numbers efficiently
4. The paper's focus is on **making QR fast**, not eliminating it

**This directly refutes our proposal's assumption that Chebyshev preserves orthogonality.**

### Actual Performance Data

From `/tmp/chebyshev-iter-2-dump-iter-3-energy-0526-0655.log`:
- SCF iter-1: 171.62 seconds
- SCF iter-2: 122.85 seconds

**Not** the 25-55 ms we estimated. Current implementation has significant CPU bottlenecks.

---

## What We Learned

### Correct Understanding

1. **CASTEP's recovery mechanism**: 19-26 eigensolver iterations per SCF step, not 1
2. **Chebyshev filtering**: Destroys orthogonality, requires explicit re-orthonormalization
3. **ChASE algorithm**: Iterative Chebyshev + QR + RR + locking (standard pattern)
4. **Performance bottleneck**: Current implementation is CPU-bound, not GPU-accelerated

### Incorrect Assumptions in Proposal

1. ❌ "Chebyshev preserves orthonormality" — **FALSE** (condition number grows exponentially)
2. ❌ "Per-band RQ is sufficient without orthogonalization" — **FALSE** (density errors accumulate)
3. ❌ "Current single-sweep takes 25-55 ms" — **FALSE** (actually 120+ seconds)
4. ❌ "Iterative approach is viable without full GPU acceleration" — **FALSE** (would take 26-43 hours)

### What the Proposal Got Right

1. ✓ CASTEP iterates eigensolver until convergence (not fixed iterations)
2. ✓ Single-sweep is insufficient (1 vs 26 iterations per SCF step)
3. ✓ RR rotation can mix degenerate bands (but orthogonalization is still needed)
4. ✓ Locking converged bands prevents regression

---

## Next Steps

### Immediate Actions

1. **Archive iterative Chebyshev proposal** as "investigated but not viable"
2. **Study CASTEP's band-by-band CG implementation** in detail
3. **Plan band-by-band CG implementation** for chemrust

### Implementation Plan (High-Level)

**Phase 1: Algorithm Design**
- Extract CASTEP's CG algorithm from source code
- Design chemrust-specific variant for USPP
- Define convergence criteria and locking strategy

**Phase 2: CPU Prototype**
- Implement band-by-band CG on CPU
- Reuse existing gradient computation (H·ψ, S·ψ)
- Implement line search (Rayleigh quotient minimization)
- Validate against CASTEP on Cu111+CO fixture

**Phase 3: GPU Migration**
- Move gradient computation to GPU
- Implement GPU line search
- Implement GPU orthogonalization (batched)
- Profile and optimize

**Phase 4: Integration**
- Replace current eigensolver with band-by-band CG
- Tune convergence tolerances
- Validate on multiple systems
- Performance benchmarking

---

## References

1. **CASTEP source**: `~/programming/CASTEP-GPU-port/Source/Functional/hamiltonian.F90`
   - Band-by-band CG implementation
   - Convergence criteria
   - Locking and deflation strategy

2. **ChASE paper**: Di Napoli & Wu, "Estimating the condition number of Chebyshev filtered vectors"
   - Proves condition number growth: κ₂ ≤ η·|ρ₁|^m
   - Validates iterative Chebyshev + QR + RR pattern
   - Shows QR is load-bearing, not eliminable

3. **DG-CheFSI paper**: Banerjee et al., "Chebyshev polynomial filtered subspace iteration in the discontinuous Galerkin method"
   - Discusses eigenvector alignment for changing basis
   - Confirms multiple CheFSI cycles per SCF step is expensive
   - Shows CheFSI is standard practice for DFT

4. **CASTEP recovery experiments**:
   - Davidson .check: `notes/debug/debug-20260526-0635/RESOLUTION.md`
   - Chebyshev .check: `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0526_from_chemrust_chebyshev/`

5. **Iterative Chebyshev proposal**: `notes/debug/debug-20260526-iterative-chebyshev-proposal/PROPOSAL.md`
   - Detailed algorithm description
   - Performance analysis (now known to be incorrect)
   - Mathematical derivations (correct, but orthogonality assumption was wrong)

---

## Lessons Learned

### Technical Lessons

1. **Always check actual performance data** before making estimates
2. **Orthogonality preservation is not free** — Chebyshev filtering destroys it
3. **Proven algorithms are lower risk** than novel approaches
4. **GPU acceleration is load-bearing** for iterative methods

### Process Lessons

1. **Critical review is essential** — the other agent's feedback caught fundamental flaws
2. **Reference implementations matter** — CASTEP provides a working solution
3. **Incremental development reduces risk** — band-by-band CG can be built incrementally
4. **Performance profiling before optimization** — don't optimize based on estimates

---

## Conclusion

The iterative Chebyshev + per-band RQ proposal was a valuable investigation that:
- Deepened our understanding of Chebyshev filtering
- Revealed the importance of orthogonalization
- Identified CASTEP's actual recovery mechanism (26 iterations per SCF step)
- Led us to the correct solution: band-by-band CG

**Band-by-band CG is the pragmatic, proven, and implementable solution to the SCF cascade problem.**

We will proceed with band-by-band CG implementation in a clean session.

---

**End of Decision Document**
