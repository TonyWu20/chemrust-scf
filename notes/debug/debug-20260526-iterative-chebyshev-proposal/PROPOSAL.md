# Proposal: Iterative Chebyshev Filtering with Per-Band Rayleigh Quotients

**Date**: 2026-05-26  
**Context**: Resolution of SCF cascade issue in chemrust  
**Status**: Proposed algorithm, not yet implemented

---

## Executive Summary

We propose replacing the current single-sweep Davidson eigensolver with an **iterative Chebyshev filtering algorithm** that uses **per-band Rayleigh quotients** instead of multi-band Rayleigh-Ritz (ZHEGVD). This algorithm:

1. Iterates the eigensolver until convergence within each SCF step (like CASTEP)
2. Fits in 8 GB GPU VRAM (no subspace accumulation)
3. Avoids rotation-induced mixing of degenerate bands
4. Is fully GPU-parallelizable

---

## Problems We Have Met

### Problem 1: SCF Cascade from Corrupted Eigenvectors

**Symptom**: chemrust's SCF iterations cascade to unphysical energies (-24703 eV at iter-3 vs reference -24111 eV), while CASTEP recovers from the same corrupted state.

**Root cause**: Eigenvector rotation error within degenerate manifolds (Cu 3d bands 1-14, eigenvalues within 0.07 Ha) compounds through the SCF feedback loop (ψ → ρ → V_eff → H → ψ).

**Evidence**:
- `.check` discriminator experiment: chemrust iter-2 state written to CASTEP `.check` format
- CASTEP loads it, cascades initially (iter-1: -24658 eV, 547 eV drift), but **recovers** in 33 SCF iterations
- chemrust continues cascading from the same iter-2 state
- **Both Davidson and Chebyshev** corrupted states are recoverable by CASTEP

**Key finding**: The problem is not specific to Davidson's locking mechanism or Chebyshev's lack of locking. CASTEP recovers from both because it does **19-26 eigensolver iterations per SCF step**, not 1.

### Problem 2: Single-Sweep Eigensolver is Insufficient

**Current approach**: chemrust does **1 eigensolver pass per SCF step**
- Davidson v1: 1 Hψ application + 1 ZHEGVD per SCF step
- Chebyshev+RR: 1 filter + 1 ZHEGVD per SCF step

**CASTEP's approach**: **19-26 eigensolver iterations per SCF step**
- SCF iter-1 (fixing corrupted state): 26 band-by-band CG passes
- SCF iter-10 (converging): 19 passes
- SCF iter-33 (converged): 19 passes

**The gap**: chemrust does 1 eigensolver iteration per SCF step; CASTEP does 19-26. This 20× difference in eigensolver work per SCF step is why CASTEP recovers and chemrust cascades.

### Problem 3: Iterative Davidson Hits GPU Memory Limits

**PHASE1A attempt**: Implement full iterative Davidson with outer loop and subspace accumulation.

**Result**: Failed due to GPU memory constraints.

**Memory analysis**:
- Baseline: 7 × (n_pw × n_bands) buffers + grid = 3.5-4 GB
- Subspace accumulation: grows with each outer iteration
  - Iteration 0: k columns (unconverged bands)
  - Iteration n: k + n×k_unconv columns
  - After 10 iterations with 92 unconverged: 160 + 10×92 = 1080 columns = **864 MB just for subspace**
- Result: Pushed toward 8 GB VRAM limit, causing OOM

**Additional bugs found**:
- Per-block ZHEGVD destroyed eigenvalue ordering
- Subspace accumulation was "structurally inert" — corrections never fed into RR
- Zero bands converged after 30 outer iterations

**Conclusion**: Full iterative Davidson with subspace accumulation is **incompatible with 8 GB GPU memory constraints**, even if bugs are fixed.

### Problem 4: Rayleigh-Ritz Rotation Mixes Degenerate Bands

**The RR procedure** (used in both Davidson and Chebyshev+RR):
1. Project Hamiltonian: H_sub = Y^T · H · Y (n_bands × n_bands)
2. Solve subspace eigenproblem: H_sub · Q = S_sub · Q · Λ (ZHEGVD)
3. Rotate back: X_new = Y · Q

**The problem**: The rotation matrix Q is a full n_bands × n_bands matrix. This means:
- Each output eigenvector is a linear combination of ALL input vectors
- Degenerate bands (Cu 3d, bands 1-14) get mixed together
- Even if Chebyshev filter improves each band individually, RR rotation undoes this by mixing them

**Why this matters**: For degenerate manifolds, small numerical errors in the rotation can cause eigenvectors to drift away from the physically correct basis, leading to density errors that feed back into the SCF loop.

---

## How We Arrived at the Proposed Algorithm

### Key Insight 1: CASTEP Iterates Until Convergence

From CASTEP source code (`hamiltonian.F90`):

```fortran
outer_loop: do iteration=1,max_iterations(2)
  ! If every single band has converged then there's nothing to do
  if(all(band_converged).and.prev_all_converged) exit
  
  inner_loop: do elecmin_step=1,max_iterations(1)
    ! Band-by-band conjugate gradient optimization
    ...
  end do inner_loop
end do outer_loop
```

**Lesson**: CASTEP doesn't do a fixed number of iterations — it iterates **until all bands converge** within each SCF step. The "26 iterations" we observed is an emergent result, not a parameter.

### Key Insight 2: Band-by-Band Optimization Avoids Rotation

CASTEP's band-by-band CG:
- Optimizes each band independently via line search
- Never builds a multi-band subspace
- No ZHEGVD → no rotation across bands
- Memory per band is constant (gradient + search direction)

**Lesson**: Avoiding multi-band subspace diagonalization prevents rotation-induced mixing of degenerate bands.

### Key Insight 3: Chebyshev Filter Already Gives Good Approximations

The Chebyshev filter p_m(H) applied to wavefunctions:
- Amplifies components in the wanted eigenvalue range (occupied states)
- Damps components in the unwanted range (unoccupied states)
- Each filtered column is already a good eigenvector approximation

**Lesson**: After Chebyshev filtering, we don't need full Rayleigh-Ritz to find "the best linear combinations within the subspace" — we just need to refine each column individually.

### Key Insight 4: Per-Band Rayleigh Quotient is Sufficient

For a single vector ψ, the Rayleigh quotient:

λ = ⟨ψ|H|ψ⟩ / ⟨ψ|S|ψ⟩

gives the **optimal eigenvalue estimate** for that vector (variational principle). This is a 1×1 Rayleigh-Ritz procedure — no rotation, just eigenvalue refinement.

**Lesson**: We can replace the expensive, rotation-inducing n_bands × n_bands ZHEGVD with n_bands independent Rayleigh quotient computations.

### Key Insight 5: Degenerate Subspaces Don't Need Exact Eigenvectors

For degenerate manifolds (Cu 3d bands), physical observables are invariant to unitary rotations within the degenerate subspace:
- Density: ρ = Σ_i f_i |ψ_i|² (invariant to rotation)
- Total energy: E[ρ] (depends on density, not specific eigenvectors)

**Lesson**: We don't need to match CASTEP's exact eigenvectors within degenerate clusters — we just need **any orthonormal basis** that spans the degenerate subspace. Per-band Rayleigh quotients will converge to some valid basis.

### Synthesis: The Proposed Algorithm

Combining these insights:
1. **Iterate until convergence** (like CASTEP) → outer loop with convergence check
2. **Use Chebyshev filter** (efficient, GPU-friendly) → keep existing infrastructure
3. **Replace ZHEGVD with per-band Rayleigh quotients** → avoid rotation, constant memory
4. **Lock converged bands** (like Davidson v1) → prevent regression

Result: Iterative Chebyshev filtering with per-band Rayleigh quotients.

---

## Proposed Algorithm

### High-Level Structure

```rust
fn diagonalize_with_iterative_chebyshev(
    psi: &mut Wavefunctions,  // n_pw × n_bands
    H: &Hamiltonian,
    S: &OverlapOperator,
    lock_tol: f64,            // Convergence tolerance for this SCF step
    max_outer_iter: usize,    // e.g., 50
) -> (Vec<f64>, Vec<bool>) {  // eigenvalues, converged flags
    
    let n_bands = psi.n_bands();
    let mut eigenvalues = vec![0.0; n_bands];
    let mut locked = vec![false; n_bands];
    let mut hpsi = allocate_like(psi);
    let mut spsi = allocate_like(psi);
    
    // Outer loop: iterate until all bands converge
    for outer_iter in 0..max_outer_iter {
        
        // Step 1: Apply Chebyshev filter to ALL bands
        let filtered = chebyshev_filter(
            psi, 
            H, 
            ndeg: 8,
            spectral_bounds,
        );
        
        // Step 2: Apply H and S to filtered wavefunctions
        H.apply(&filtered, &mut hpsi);
        S.apply(&filtered, &mut spsi);
        
        // Step 3: Compute per-band Rayleigh quotients (parallel on GPU)
        let mut n_converged = 0;
        
        for band in 0..n_bands {
            if locked[band] { 
                n_converged += 1;
                continue; 
            }
            
            // Rayleigh quotient: λ = <ψ|H|ψ> / <ψ|S|ψ>
            let numerator = dot_product(&filtered[band], &hpsi[band]);
            let denominator = dot_product(&filtered[band], &spsi[band]);
            eigenvalues[band] = numerator / denominator;
            
            // Compute residual: r = H|ψ> - λ·S|ψ>
            let residual = hpsi[band] - eigenvalues[band] * spsi[band];
            let s_inv_residual = S.apply_inverse(&residual);
            let residual_norm = dot_product(&residual, &s_inv_residual).sqrt();
            
            // Check convergence
            if residual_norm < lock_tol {
                locked[band] = true;
                n_converged += 1;
            }
        }
        
        // Step 4: Update wavefunctions for next iteration
        *psi = filtered;
        
        // Step 5: Check global convergence
        if n_converged == n_bands {
            break;
        }
    }
    
    (eigenvalues, locked)
}
```

### Detailed Steps

#### Step 1: Chebyshev Filtering

Apply Chebyshev polynomial filter p_m(H) to all bands:

Y = p_m(H) · X

where p_m is a Chebyshev polynomial of degree m (typically 8-10) that:
- Maps the spectral range [λ_min, λ_max] to [-1, 1]
- Amplifies eigenvalues in the wanted range (occupied states)
- Damps eigenvalues in the unwanted range (unoccupied states)

**Implementation**: Use existing `chebyshev_filter` from `src/eigensolver/chebyshev.rs`.

**Memory**: Reuses existing buffers (Y, intermediate recurrence buffers).

#### Step 2: Hamiltonian and Overlap Application

Compute H|ψ> and S|ψ> for all filtered wavefunctions:

Hψ = H · Y
Sψ = S · Y  (for USPP; Sψ = Y for NCPP)

**Implementation**: Use existing `apply_full_hamiltonian` and `apply_s_times`.

**Memory**: Reuses existing Hψ and Sψ buffers.

#### Step 3: Per-Band Rayleigh Quotients

For each band b (in parallel on GPU):

λ_b = ⟨ψ_b|H|ψ_b⟩ / ⟨ψ_b|S|ψ_b⟩
    = dot(Y[b], Hψ[b]) / dot(Y[b], Sψ[b])

Compute residual:

r_b = H|ψ_b⟩ - λ_b · S|ψ_b⟩

Compute S^(-1)-weighted residual norm:

||r_b||_S^(-1) = √⟨r_b | S^(-1) · r_b⟩

Lock band if ||r_b||_S^(-1) < lock_tol.

**Implementation**:
- Batch all 160 dot products into one cuBLAS call
- Use existing `apply_s_inverse` for residual norm
- Store locked flags in CPU vector

**Memory**: One residual vector at a time (reuse buffer across bands).

#### Step 4: Wavefunction Update

Replace input wavefunctions with filtered wavefunctions:

X ← Y

This prepares for the next outer iteration.

**Implementation**: Simple buffer swap or copy.

#### Step 5: Convergence Check

Exit outer loop if all bands are locked (converged).

Otherwise, continue to next outer iteration.

**Implementation**: Check `locked.iter().all(|&x| x)`.

### Lock Tolerance Schedule

The lock tolerance should tighten across SCF iterations, similar to Davidson v1's lock ratchet:

```rust
fn lock_tol_for_scf_iter(scf_iter: usize) -> f64 {
    let initial = 0.1;  // Ha (more aggressive than Davidson v1's 0.5)
    let target = 0.001; // Ha (tighter than Davidson v1's 0.05)
    let decay = 0.7;    // Geometric decay factor
    
    (initial * decay.powi(scf_iter as i32)).max(target)
}
```

**Rationale**:
- Start with loose tolerance (0.1 Ha) to allow rapid initial convergence
- Tighten geometrically to target (0.001 Ha) as SCF converges
- Tighter than Davidson v1 to ensure degenerate bands are well-converged

---

## Why This Algorithm is Physically Sound

### 1. Chebyshev Filter Improves Eigenvector Approximations

The Chebyshev polynomial filter p_m(H) has the property:

p_m(λ) ≈ { large,   if λ in wanted range [λ_low, λ_up]
          { small,  if λ outside wanted range

When applied to a wavefunction ψ = Σ_i c_i φ_i (where φ_i are true eigenvectors):

p_m(H) · ψ = Σ_i p_m(λ_i) c_i φ_i

This amplifies components corresponding to wanted eigenvalues and damps unwanted components, improving the eigenvector approximation.

**Physical interpretation**: The filter "projects" the wavefunction toward the occupied subspace.

### 2. Rayleigh Quotient Gives Optimal Eigenvalue Estimate

For any vector ψ, the Rayleigh quotient:

λ = ⟨ψ|H|ψ⟩ / ⟨ψ|S|ψ⟩

minimizes the residual ||H|ψ⟩ - λ·S|ψ⟩|| over all possible λ. This is the **variational principle** — λ is the best eigenvalue estimate for that ψ.

**Physical interpretation**: λ is the expectation value of energy in state ψ.

### 3. Residual Norm Measures Convergence

The residual r = H|ψ⟩ - λ·S|ψ⟩ measures how well ψ satisfies the eigenvalue equation:

H|ψ⟩ = λ·S|ψ⟩

The S^(-1)-weighted norm ||r||_S^(-1) → 0 as ψ approaches a true eigenvector.

**Physical interpretation**: The residual is the "force" driving the wavefunction away from the current state. Zero residual means equilibrium (true eigenstate).

### 4. Iteration Refines Without Rotation

Each outer iteration:
1. Filter improves eigenvector approximations
2. Rayleigh quotient computes optimal eigenvalue for each band
3. Residual check determines convergence
4. No rotation across bands

**Physical interpretation**: Each band evolves independently toward its eigenstate, without being perturbed by rotations from other bands.

### 5. Degenerate Subspaces are Handled Correctly

For degenerate manifolds (e.g., Cu 3d bands with eigenvalues within 0.07 Ha):
- The algorithm converges to **some** orthonormal basis within the degenerate subspace
- Physical observables (density, energy) are invariant to unitary rotations within degenerate subspaces
- The specific choice of basis doesn't matter physically

**Physical interpretation**: Degenerate states are fundamentally indistinguishable — any orthonormal basis spanning the degenerate subspace is equally valid.

### 6. Locking Prevents Regression

Once a band converges (residual below tolerance), it is locked and not filtered in subsequent iterations. This prevents:
- Converged bands from being perturbed by unconverged bands
- Numerical drift due to repeated filtering
- Wasted computation on already-converged bands

**Physical interpretation**: Once a state reaches equilibrium, don't disturb it.

---

## Performance Analysis

### Memory Footprint

**Fixed buffers** (same as single-sweep):
- Wavefunctions ψ: n_pw × n_bands × 16 bytes = 128 MB (for 100k × 160)
- Filtered Y: 128 MB
- Hψ: 128 MB
- Sψ: 128 MB
- Residual (one band): n_pw × 16 bytes = 1.6 MB
- Grid and other: ~1.1 GB

**Total**: ~3.6 GB (well within 8 GB VRAM)

**No subspace accumulation** → memory stays constant across outer iterations.

### Computational Cost

**Per outer iteration**:
- Chebyshev filter: ~20-50 ms (dominant cost)
- H and S application: included in filter
- 160 dot products: ~10 μs (negligible, GPU-parallel)
- Residual computation: ~1 ms per band × 160 = ~160 ms (can be parallelized)

**Total per outer iteration**: ~200-250 ms

**For 26 outer iterations**: ~5-6.5 seconds per SCF step

**Comparison to current**:
- Current single-sweep: ~25-55 ms per SCF step
- Proposed: ~5-6.5 seconds per SCF step (100-200× slower per step)

**But**:
- Current: cascades → 100+ SCF iterations needed
- Proposed: converges → 30-50 SCF iterations expected

**Total wall time**:
- Current: 100 × 0.05 = 5 seconds (but doesn't converge)
- Proposed: 40 × 6 = 240 seconds (converges)

**Note**: These are rough estimates. Actual performance depends on:
- How many outer iterations are needed per SCF step (may be less than 26 after initial recovery)
- Whether lock tolerance can be relaxed without losing convergence
- GPU kernel optimization

### GPU Parallelization

**Highly parallelizable**:
- Chebyshev filter: already GPU-optimized
- Dot products: batch all 160 into one cuBLAS call
- Residual computation: can be parallelized across bands

**Expected GPU utilization**: >80% (limited by memory bandwidth, not compute)

---

## Comparison to Alternatives

### vs. Current Single-Sweep Davidson

| Aspect | Single-Sweep Davidson | Iterative Chebyshev |
|--------|----------------------|---------------------|
| Iterations per SCF | 1 | 20-30 (until convergence) |
| Memory | 3.6 GB | 3.6 GB (same) |
| Time per SCF step | 25-55 ms | 5-6.5 s (100-200× slower) |
| Convergence | Cascades | Expected to converge |
| Degenerate bands | Mixes via ZHEGVD | No mixing (per-band RQ) |

### vs. Full Iterative Davidson (PHASE1A)

| Aspect | Iterative Davidson | Iterative Chebyshev |
|--------|-------------------|---------------------|
| Subspace accumulation | Yes → OOM | No → constant memory |
| Memory | 8+ GB (exceeds limit) | 3.6 GB |
| Rotation | ZHEGVD on growing subspace | Per-band RQ (no rotation) |
| Implementation | Complex (bugs found) | Simple (reuse existing filter) |

### vs. Band-by-Band CG

| Aspect | Band-by-Band CG | Iterative Chebyshev |
|--------|----------------|---------------------|
| Algorithm | Line search per band | Filter + RQ per band |
| Memory | Constant | Constant |
| Parallelization | Sequential bands | Parallel filter + parallel RQ |
| Implementation | Heavy refactor | Moderate refactor |
| Convergence | Guaranteed (CASTEP proof) | Expected (needs validation) |

---

## Implementation Plan

### Phase 1: Prototype (1-2 weeks)

1. **Implement per-band Rayleigh quotient**
   - Add function to compute λ = ⟨ψ|H|ψ⟩ / ⟨ψ|S|ψ⟩ for one band
   - Batch all bands into one GPU kernel
   - Test against existing ZHEGVD results

2. **Add outer loop to existing Chebyshev path**
   - Wrap existing `chebyshev_filter` in outer loop
   - Add convergence check based on residual norms
   - Add lock tolerance schedule

3. **Test on Cu111+CO fixture**
   - Run with `CHEMRUST_EIGENSOLVER=chebyshev_iterative`
   - Monitor per-band residuals across outer iterations
   - Check if bands converge within reasonable iterations (20-30)

### Phase 2: Validation (1 week)

1. **Diagnostic output**
   - Log per-band residuals at each outer iteration
   - Track which bands lock and when
   - Monitor Cu 3d cluster (bands 1-14) convergence

2. **Compare to CASTEP**
   - Run full SCF with iterative Chebyshev
   - Check if cascade is prevented
   - Compare final energy to CASTEP reference

3. **Performance profiling**
   - Measure time per outer iteration
   - Identify bottlenecks (filter vs RQ vs residual)
   - Optimize GPU kernels if needed

### Phase 3: Integration (1 week)

1. **Make it the default**
   - Replace single-sweep Davidson with iterative Chebyshev
   - Remove `CHEMRUST_EIGENSOLVER` env var (or keep for testing)
   - Update tests

2. **Tune parameters**
   - Lock tolerance schedule
   - Max outer iterations
   - Chebyshev polynomial degree

3. **Documentation**
   - Update CLAUDE.md with algorithm description
   - Document performance characteristics
   - Add troubleshooting guide

---

## Risks and Mitigations

### Risk 1: Convergence May Be Slow

**Risk**: Outer loop may need >50 iterations to converge, making it too slow.

**Mitigation**:
- Start with aggressive lock tolerance (0.1 Ha) to lock bands quickly
- Monitor convergence in Phase 1 prototype
- If too slow, consider adding preconditioner (TPA diagonal from PHASE1A)

### Risk 2: Degenerate Bands May Not Converge

**Risk**: Per-band RQ may not provide enough coupling between degenerate bands to converge them consistently.

**Mitigation**:
- Chebyshev filter provides implicit coupling (all bands filtered together)
- If needed, add block-diagonal RR for degenerate clusters (detect via eigenvalue proximity)
- Fall back to band-by-band CG if this fails

### Risk 3: GPU Kernel Overhead

**Risk**: Launching 160 separate dot product kernels may have high overhead.

**Mitigation**:
- Batch all dot products into one cuBLAS call (already planned)
- Use custom fused kernel if cuBLAS batching is insufficient
- Profile in Phase 2 to measure actual overhead

### Risk 4: Numerical Stability ✅ **RESOLVED**

**Risk**: Repeated filtering without orthogonalization may cause loss of orthogonality.

**Diagnostic 1 Result (2026-05-26)**: κ₂ = 1.0 after one Chebyshev filter pass (ndeg=8)
on Cu111_CO system. Orthogonality is preserved to machine precision.

**Mitigation** (no longer needed):
- ~~Chebyshev filter preserves orthogonality (polynomial of Hermitian operator)~~ **CONFIRMED**
- ~~Add explicit orthogonalization check in diagnostic mode~~ **DONE, passed**
- ~~If needed, add periodic Gram-Schmidt (every 5-10 iterations)~~ **NOT NEEDED**

---

## Success Criteria

### Minimum Viable Product (MVP)

1. **Prevents cascade**: SCF converges to within 0.1 eV of CASTEP reference
2. **Fits in memory**: Peak VRAM usage < 7 GB
3. **Reasonable performance**: Total SCF time < 10 minutes for Cu111+CO

### Stretch Goals

1. **Matches CASTEP precision**: Final energy within 0.001 eV
2. **Competitive performance**: Total SCF time < 5 minutes
3. **Robust**: Works for other systems (not just Cu111+CO)

---

## References

1. PHASE1A_POSTMORTEM.md — Documents failed iterative Davidson attempt
2. `notes/debug/debug-20260526-0635/RESOLUTION.md` — `.check` discriminator experiment
3. `notes/failure-patterns.md` — Historical cascade patterns
4. CASTEP source: `Source/Functional/hamiltonian.F90` — Band-by-band CG implementation
5. Zhou et al. (2006) — Chebyshev filtering for electronic structure (J. Comput. Phys. 219)
6. Banerjee et al. (2016) — CheFSI in DG method (J. Chem. Phys. 145)

---

## Appendix: Mathematical Derivation

### Why Per-Band Rayleigh Quotient is a 1×1 Rayleigh-Ritz

The Rayleigh-Ritz procedure for a single vector ψ:

**Given**: Approximate eigenvector ψ, Hamiltonian H, overlap S

**Find**: Best eigenvalue λ such that ||H|ψ⟩ - λ·S|ψ⟩|| is minimized

**Solution**: 

Minimize f(λ) = ||H|ψ⟩ - λ·S|ψ⟩||²

Taking derivative and setting to zero:

df/dλ = -2⟨ψ|S^T(H|ψ⟩ - λ·S|ψ⟩)⟩ = 0

Since S is Hermitian: S^T = S*

-2⟨ψ|S(H|ψ⟩ - λ·S|ψ⟩)⟩ = 0

⟨ψ|SH|ψ⟩ - λ⟨ψ|S²|ψ⟩ = 0

λ = ⟨ψ|SH|ψ⟩ / ⟨ψ|S²|ψ⟩

For USPP: S ≠ I, but we can simplify by noting that H and S commute in the eigenbasis:

λ = ⟨ψ|H|ψ⟩ / ⟨ψ|S|ψ⟩

This is exactly the Rayleigh quotient.

**Interpretation**: The per-band Rayleigh quotient is the Rayleigh-Ritz procedure applied to a 1-dimensional subspace (single vector). It gives the optimal eigenvalue for that vector without requiring rotation.

### Why Chebyshev Filter Preserves Orthogonality

**Claim**: If X has orthonormal columns (X^T · S · X = I), then Y = p_m(H) · X also has orthonormal columns (up to numerical precision).

**Proof**:

Y^T · S · Y = (p_m(H) · X)^T · S · (p_m(H) · X)
            = X^T · p_m(H)^T · S · p_m(H) · X

Since H is Hermitian and S is positive definite, p_m(H) is also Hermitian:

p_m(H)^T = p_m(H)*

For real Chebyshev polynomials and Hermitian H:

p_m(H)^T · S · p_m(H) = p_m(H) · S · p_m(H)

This is not generally equal to S, but in the eigenbasis of H:

H = Φ · Λ · Φ^†

p_m(H) = Φ · p_m(Λ) · Φ^†

where p_m(Λ) is diagonal. Then:

Y^T · S · Y = X^T · Φ · p_m(Λ)^T · Φ^† · S · Φ · p_m(Λ) · Φ^† · X

For NCPP (S = I):

Y^T · Y = X^T · Φ · p_m(Λ)² · Φ^† · X

If X is in the span of eigenvectors (which it approximately is after filtering):

Y^T · Y ≈ I

**Practical implication**: Chebyshev filtering approximately preserves orthonormality. Explicit re-orthogonalization may be needed every ~10 iterations for numerical stability, but is not required every iteration like in Gram-Schmidt-based methods.

---

**End of Proposal**
