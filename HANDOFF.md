# Handoff — 2026-06-26: ZTRSM column mixing fixed; unoccupied band inflation remains

**Branch**: `feat/chebyshev-iterative-eigensolver`
**Status**: Warm start converges in 4 iters (-24110.967 vs -24110.967 eV ref) ✅ | Unoccupied band eigenvalues inflated ⚠️ | Cold start fails (overflow) ⚠️

---

## Resolved issues

### ZTRSM column mixing (root cause of unoccupied band corruption)

**Symptom**: Bands 120+ had H-expectations diverging from pre-filter ritz values
(ritz 0.04 → H-expectation 1.08). Locked bands (ndeg=0, copied from input)
also showed wrong H-expectation.

**Root cause**: Cholesky QR's `ZTRSM` step (`X = X·R⁻¹`) mixes columns with
very different bare `|ψ|²` magnitudes. For USPP (S ≠ I), occupied bands have
`|ψ|² ≈ 1.0` while unoccupied bands have `|ψ|² ≈ 0.06`. ZTRSM redistributes
norm from occupied columns to unoccupied columns.

**Fix** (`c837b16`): Skip ZTRSM when ZPOTRF succeeds (info=0). Per-band S-norm
normalization already gives `S_sub ≈ I` (max|off| ≈ 4×10⁻⁴), so `rayleigh_ritz`
ZHEGVD handles it directly. Regularization retry + ZHEEVD fallback keep ZTRSM
since nearly-diagonal R after regularization makes mixing negligible.

**Verification**: Pre/post ZTRSM psi coefficients are now identical (no mixing).

### ZPOTRF failure (info=85—160) on Cu111_CO non-cubic grid

**Root cause**: All 160 bands filtered with oracle=0 (including 121 already-converged
warm-start bands), producing near-linearly-dependent vectors with `S_sub` off-diagonals ≈ 0.999.

**Fix** (`3e75331`): Lock converged bands even with oracle=0 — `ndeg=0` for bands
with `fresh_residual[b] < tolerance`. This is a **divergence from ABINIT** (ABINIT's
oracle=0 broadcasts same ndeg to all bands), necessary because ABINIT's `getAX_BX`
inside the recurrence loop + matrix-free RR provide numerical stabilization our
pipeline lacks.

### ABINIT-matching per-ion block-diagonal S⁻¹

**Fix** (`56b6dec`): Replace global `M = Q⁻¹ + B^H·B` LU preconditioner with
per-ion block-diagonal `Q⁻¹` preconditioner, matching ABINIT `m_invovl.F90:1100-1140`.
Iterative refinement up to 30 iterations with per-band residual convergence tracking.

---

## Remaining issues

### 1. Unoccupied band eigenvalue inflation (HIGH)

**Symptom**: Cu111_CO warm start `.bands` shows bands 125+ with eigenvalues up to
1.86 Ha vs 0.115 Ha CPU reference. Occupied bands (1-120) match reference within 0.0001 Ha.
Final energy converges correctly (-24110.967 vs -24110.967 eV).

**Mechanism**: The filter's Chebyshev recurrence operates on all 160 bands simultaneously
(column-independent). The 39 unlocked bands (with residual > tolerance) get corrupted
by the filter. After rayleigh_ritz ZHEGVD, the eigenvector rotation matrix X mixes the
corruption into ALL bands. In pass 2, the "locked" bands copy corrupted psi_input from
pass 1's RR output. The corruption self-amplifies across SCF iterations.

**Why ZTRSM fix wasn't sufficient**: ZTRSM was ONE source of column mixing. The RR
ZHEGVD's eigenvector rotation matrix is ANOTHER source — it rotates all 160 columns
simultaneously, propagating any corruption in the H_sub matrix.

**Why S⁻¹ fix wasn't sufficient**: The T_1 diagnostic confirmed S⁻¹ produces correct
results (matching ritz) for all bands. The corruption enters during the full Chebyshev
recurrence (T_2 through T_n), not the first S⁻¹·H step.

**Lock-check diagnostic** (`a3579c8`): Confirmed locked-band copy is correct
(`⟨x_curr|psi_input⟩ = |psi|²`). The locked bands receive already-corrupted data
from the previous RR output.

**Next step**: Investigate whether the Chebyshev recurrence itself is corrupting
unlocked bands, or whether the RR ZHEGVD rotation is propagating corruption.

### 2. Cold start overflow (HIGH)

**Symptom**: Cu111_CO cold start: `max|psi| = 2×10¹⁹`, ZPOTRF info=63, ZHEEVD info=155,
RR ZHEGVD info=223. All 160 bands have ritz clustered at 1.89-1.94 Ha.

**Root cause**: Random initial guess vectors contain deep-core components (λ ≈ -10 Ha,
xred ≈ -1.5). T_40(1.5) ≈ 10¹⁶ causes double-precision overflow. Ampfactor divides by
T_n(average ritz) ≈ 6.3 — insufficient against 10¹⁶.

**Partial fix** (`ddff9d9`): Safe degree cap based on eigenvalue bounds:
`n_max = floor(ln(2×10¹⁰) / ln(|x| + sqrt(x²-1)))`. For Cu111_CO, caps ndeg at ~19.
Prevents overflow but doesn't fix convergence — cold start needs many SCF iterations
with such small ndeg.

**Rejected approach** (`ccf53d7`, reverted by `67fa50b`): Davidson fallback for first
SCF iteration. Caused warm start divergence because Davidson→Chebyshev transition
on iter 2 produced eigenvalue shifts that destabilized SCF convergence.

**Proper fix**: Improve initial guess quality (LCAO) or use iterative subspace
expansion with per-band degree differentiation.

---

## Key files changed

| File | Key changes |
|------|-------------|
| `src/eigensolver/chebyshev.rs` | ZTRSM skip, converged-band locking, safe degree cap, per-band S-norm normalization, ZPOTRF regularization chain + ZHEEVD fallback, lock-check diagnostic |
| `src/eigensolver/hamiltonian.rs` | Per-ion block-diagonal S⁻¹ iterative refinement (replaces global LU) |
| `src/eigensolver/davidson.rs` | `compute_all()` restoration, CUDA event timing |
| `src/eigensolver/rayleigh_ritz.rs` | S_sub diagnostic before ZHEGVD |
| `src/ffi.rs` | (reverted) Davidson fallback for first SCF iteration |
| `HANDOFF.md` | This file |

## Key decisions

1. **ZTRSM skipped for well-conditioned S_sub** — per-band normalization + small
   off-diagonals is sufficient; ZHEGVD handles S_sub ≈ I
2. **Converged-band locking is necessary** — divergence from ABINIT oracle=0, but
   ABINIT's `getAX_BX`-inside-loop provides implicit regularization we lack
3. **Davidson fallback breaks warm start** — eigenvalue shift on iter 2 causes
   SCF oscillation; cold start needs a different solution
4. **Per-ion S⁻¹ matches ABINIT algorithm** — eliminates global LU dependency

## Reference fixtures

| Run | Path | What it proves |
|-----|------|---------------|
| Cu111_CO warm start | `/export/.../Cu111_CO_Single_Point_0604_warm_start/slurm_output_cheby_2838.txt` | ZTRSM skip confirmed, lock-check passes, unoccupied bands still inflated |
| Cu111_CO cold start | `/export/.../Cu111_CO_Single_Point_0530_rust_eigensolver/slurm_output_cheby_2832.txt` | Safe degree cap works for pass 1 (no overflow), pass 2 overflows without additional fixes |
| CPU reference | `/export/.../Cu111_CO_H_dump/Cu111_CO.bands` | Ground truth eigenvalues for comparison |

## Pending

- [ ] Investigate Chebyshev recurrence corruption of unlocked bands (or RR propagation)
- [ ] Fix cold start (LCAO initial guess, Davidson fallback that works, or iterative degree increase)
- [ ] Clean up verbose diagnostics once unoccupied band issue is resolved
- [ ] Test NiO cold/warm start with all fixes
