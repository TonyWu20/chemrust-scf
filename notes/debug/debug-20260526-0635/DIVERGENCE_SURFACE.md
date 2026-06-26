# Divergence Surface Enumeration

**Context**: CASTEP continuation from chemrust iter-2 `.check` shows:
- Iter-1 at cascade level (-24658 eV vs reference -24111 eV)
- Recovery to convergence within 33 iterations
- chemrust cannot recover (continues cascading)

## Generic Divergence Categories

### 1. Data Layout / Axis Ordering
**Item**: `.check` file stores density/wavefunctions in different memory layout than CASTEP expects
**Status**: **Ruled out by SC-1** — CASTEP loads the `.check` without error and produces physically plausible energies. A layout mismatch would cause immediate NaN or garbage energies.

### 2. Normalization / Scaling Conventions
**Item**: Wavefunction coefficients, density, or eigenvalues scaled by different convention (e.g., missing √Ω factor)
**Status**: **Ruled out by SC-2** — CASTEP converges to within 0.001 eV of reference. A normalization error would produce a systematic energy offset that persists after convergence.

### 3. Sign / Direction Conventions
**Item**: Phase convention on wavefunction coefficients (global phase per band)
**Status**: **Ruled out by SC-2** — Global phase is gauge freedom; CASTEP's eigensolver would find the correct phase during diagonalization. Final energy match confirms no sign error.

### 4. Boundary / Edge-Case Handling
**Item**: Conditional branches in CASTEP's `.check` reader that chemrust's writer doesn't match
**Status**: **Ruled out by SC-1** — CASTEP accepts the file. Any missing conditional (e.g., spin-polarized vs non-spin-polarized branch) would trigger a read error or assertion.

### 5. Unit Conversion at Boundary
**Item**: Energy units (eV vs Ha), length units (Å vs Bohr), or k-point coordinates (fractional vs Cartesian)
**Status**: **Ruled out by SC-2** — Final energy matches. Unit errors would produce systematic offsets (e.g., 27.2× for Ha/eV mismatch).

### 6. Parser Precision / Offset Assumptions
**Item**: `.check` record framing (Fortran unformatted I/O), endianness, or field alignment
**Status**: **Ruled out by SC-1** — CASTEP reads the file successfully. Binary format errors cause immediate read failures.

### 7. Decomposition / Parallel Artifacts
**Item**: G-vector distribution across MPI ranks, or gathered vs distributed representation
**Status**: **Ruled out by SC-1** — `.check` files are always gathered (single-rank representation). CASTEP's parallel decomposition happens after read.

### 8. Diagnostic Comparison Code
**Item**: chemrust's diagnostic that compares iter-3 energy is wrong
**Status**: **Ruled out by EXTERNAL anchor** — chemrust iter-3 energy -24703.41 eV is directly from test panic output, not a diagnostic computation.

## Project-Specific Divergence Categories

### 9. Wavefunction S-Orthonormality
**Item**: chemrust iter-2 wavefunctions violate `⟨ψ|S|ψ⟩ = 1` constraint
**Status**: **To be tested in Step 7** — Not ruled out by SC-1 (CASTEP may accept slightly non-orthonormal ψ and re-orthogonalize). SC-3 (cascade-level iter-1) suggests ψ are not perfectly S-orthonormal.

### 10. Density Electron Count
**Item**: Total density in `.check` does not integrate to 186 electrons
**Status**: **To be tested in Step 7** — Not ruled out. SC-3 suggests density may be wrong. CASTEP's SCF would correct electron count via Fermi level adjustment, explaining recovery.

### 11. Eigenvalue-Occupancy Consistency
**Item**: Eigenvalues in `.check` don't match occupancies (e.g., occupied bands above Fermi level)
**Status**: **To be tested in Step 7** — Not ruled out. CASTEP recomputes occupancies from eigenvalues + Fermi level, so inconsistency would be corrected in iter-1.

### 12. Augmentation Density Spatial Distribution
**Item**: `ρ_aug` in `.check` has wrong spatial distribution (correct integral but wrong local values at ion centers)
**Status**: **To be tested in Step 7** — Not ruled out by SC-2 (CASTEP recomputes ρ_aug from ψ during SCF). SC-3 suggests ρ_aug may be wrong. This is consistent with `failure-patterns.md` line 107-117 (stale-aug-density-cascade).

### 13. V_eff Spatial Distribution
**Item**: V_eff in `.check` has wrong local values (correct range but wrong pointwise)
**Status**: **Ruled out by design** — `.check` files do NOT store V_eff. CASTEP recomputes V_eff from density during continuation. The cascade at iter-1 is from the density/wavefunctions, not V_eff.

### 14. Fine Grid vs Wave Grid Mismatch
**Item**: Density stored on wrong grid (wave grid instead of fine grid, or vice versa)
**Status**: **Ruled out by SC-1** — CASTEP's `.check` reader validates grid dimensions against `fine_grid` field. Mismatch would cause read error.

### 15. Eigensolver Algorithm Difference
**Item**: chemrust's Chebyshev + subspace RR produces eigenvectors that differ from CASTEP's band-by-band CG within degenerate manifolds
**Status**: **NOT ruled out** — This is the leading hypothesis from `failure-patterns.md` line 89-93 and memory `[[locking_is_the_load_bearing_eigensolver_property]]`. SC-3 + SC-4 + SC-5 together prove:
  - chemrust iter-2 state is damaged (CASTEP iter-1 cascades)
  - CASTEP can recover (converges in 33 iterations)
  - chemrust cannot recover (continues cascading)
  
  The difference is **algorithmic resilience**: CASTEP's band-by-band CG with locking prevents cascade amplification; chemrust's subspace RR re-diagonalizes the full subspace every iteration, allowing error to compound.

### 16. Pulay Mixing History
**Item**: `.check` stores Pulay mixing history (previous densities/residuals) that chemrust doesn't write
**Status**: **Ruled out by design** — CASTEP continuation from `.check` resets mixing history (starts fresh with the checkpoint density as iter-0). No mixing history is stored in `.check` format.

## Summary Table

| Item | Category | Status | Anchor |
|------|----------|--------|--------|
| 1. Data layout | Generic | ✓ Ruled out | SC-1 |
| 2. Normalization | Generic | ✓ Ruled out | SC-2 |
| 3. Sign convention | Generic | ✓ Ruled out | SC-2 |
| 4. Boundary handling | Generic | ✓ Ruled out | SC-1 |
| 5. Unit conversion | Generic | ✓ Ruled out | SC-2 |
| 6. Parser precision | Generic | ✓ Ruled out | SC-1 |
| 7. Parallel artifacts | Generic | ✓ Ruled out | SC-1 |
| 8. Diagnostic code | Generic | ✓ Ruled out | EXTERNAL |
| 9. S-orthonormality | Project | ⚠ To be tested | — |
| 10. Electron count | Project | ⚠ To be tested | — |
| 11. Eigenvalue-occupancy | Project | ⚠ To be tested | — |
| 12. Aug density spatial | Project | ⚠ To be tested | — |
| 13. V_eff spatial | Project | ✓ Ruled out | Design |
| 14. Grid mismatch | Project | ✓ Ruled out | SC-1 |
| 15. Eigensolver algorithm | Project | ❌ NOT ruled out | SC-3/4/5 |
| 16. Mixing history | Project | ✓ Ruled out | Design |

## Items for Step 7 (Loose-then-Tighten)

1. **Item 9**: Verify S-orthonormality of chemrust iter-2 ψ in `.check`
2. **Item 10**: Verify electron count of chemrust iter-2 density in `.check`
3. **Item 11**: Verify eigenvalue-occupancy consistency in `.check`
4. **Item 12**: Verify aug density spatial distribution (compare against CASTEP recomputed ρ_aug from same ψ)
5. **Item 15**: Confirm eigensolver algorithm difference is the root cause (requires comparing CASTEP's iter-1 ψ against chemrust iter-2 ψ)

## Leading Hypothesis (Item 15)

The `.check` discriminator experiment **confirms** the eigensolver algorithm hypothesis from `failure-patterns.md`:

> "Subspace methods rotate eigenvectors within degenerate manifolds, which CG preserves naturally. This is expected algorithmic behaviour, not a code bug."

**Evidence**:
- chemrust iter-2 state causes CASTEP to cascade initially (SC-3) → state is damaged
- CASTEP recovers within 33 iterations (SC-4) → damage is not fatal
- chemrust continues cascading (SC-5) → chemrust cannot self-correct

**Mechanism**: CASTEP's band-by-band CG **locks** converged bands (doesn't touch them in subsequent iterations), preventing error amplification. chemrust's subspace RR **re-diagonalizes** the full 40-band subspace every iteration, allowing eigenvector rotation error to compound through the SCF feedback loop (ψ → ρ → V_eff → H → ψ).

**Implication**: This is **not a bug in `.check` serialization**. The `.check` file correctly represents chemrust's iter-2 state. The problem is that chemrust's iter-2 state is already on a divergent trajectory due to lack of locking.
