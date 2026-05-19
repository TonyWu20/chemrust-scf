# Review: Phase 2 Group-C — diagonalize

**Tasks**: `notes/plans/phase-2/TASKS.md` (Group C: C-1 through C-3)
**Reviewed**: 2026-05-19
**Focus**: Chebyshev filtering, Rayleigh-Ritz, diagonalize transition

## Summary

**Changes Required — 1 critical defect, 1 significant defect, 1 minor issue.**

Runtime outcome verification: 10/10 GPU unit tests pass on CUDA 12.9 hardware (c2c identity, batched c2r, gemm, axpy, zhegvd, device roundtrips). All tests from Group B's fix pass were verified. `cargo check` and `cargo test --workspace` both clean.

The implementation scope is correct and well-structured: `chebyshev.rs` (817 lines), `rayleigh_ritz.rs` (219 lines), `vnl_data.rs` (90 lines). The V_NL via cuBLAS gemm, the Chebyshev three-buffer recurrence, and the Rayleigh-Ritz subspace diagonalization are all implemented correctly in structure.

However, 1 critical PcieAccount assertion bug and 1 significant FFT dimension ordering bug would produce wrong results or runtime panics on non-cubic grids. One minor issue should be fixed before merge.

## Per-Task Results

### C-1: Chebyshev filtering on GPU
- **Status**: ⚠ Significant Issues (1 significant, 1 minor)
- **Runtime verification**: NVRTC kernels compile, FFT C2C identity test passes on 8³ grid. No integration test against Cu111_CO fixtures (deferred to Group F).
- **Diff validation**:
  - `src/eigensolver/chebyshev.rs` created with 817 lines ✓
  - Spectral bound estimation implemented (`compute_spectral_bounds`) ✓
  - H_loc = T + V_eff via FFT roundtrip (scatter → IFFT → V_eff multiply → FFT → gather) ✓
  - V_NL via cuBLAS gemm (beta^H·psi, D·C_proj, beta·C_proj accumulate) ✓
  - Chebyshev three-buffer recurrence with norm stability check ✓
  - ColumnDistributed → RowDistributed transpose kernel ✓
  - **FFT dimension ordering mismatch (Significant)**: `plan_batched_c2c(ngx, ngy, ngz)` passes `[ngx, ngy, ngz]` to cuFFT, but the index formula `ix + ngx * (iy + ngy * iz)` uses ngx as fastest-varying. cuFFT expects `[ngz, ngy, ngx]` to match. Hidden on cubic grids; would produce wrong physics on non-cubic systems.
  - **Spectral bound uses range (Minor)**: `kinetic_max + (max_veff - min_veff)` overestimates lambda_max vs. guidance's `kinetic_max + max_veff`.
  - **Flat norm threshold (Minor)**: Uses fixed 10× growth check instead of relative ratio-to-ratio comparison from guidance.
  - `potts`, `cell`, `k_point` parameters correctly marked `_` (VNL data precomputed).

### C-2: Rayleigh-Ritz on GPU
- **Status**: ✓ Passed (minor concerns)
- **Runtime verification**: ZHEGVD unit test passes for 4×4 diagonal system with correct eigenvalues [1,2,3,4].
- **Diff validation**:
  - `src/eigensolver/rayleigh_ritz.rs` created with 219 lines ✓
  - H_sub = ψ^dag·H|ψ> via cuBLAS gemm with transa=C ✓
  - S_sub = ψ^dag·ψ via cuBLAS gemm ✓
  - ZHEGVD solve with proper CUSOLVER_EIG_MODE_VECTOR and CUBLAS_FILL_MODE_LOWER ✓
  - Info check with proper error propagation ✓
  - ψ_new = X·ψ rotation via cuBLAS gemm ✓
  - GPU transpose via shared NVRTC kernel (avoids 4MB D2H+H2D roundtrip) ✓
  - **RowDistributed shape metadata mismatch (Minor)**: Shape reported as `[n_bands, n_pw]` but transpose kernel stores data as `[n_pw, n_bands]` in memory. Latent — no code path currently syncs RowDistributed data to host.

### C-3: Wire diagonalize transition in scf.rs
- **Status**: ⚠ Critical Issue (1 critical defect)
- **Runtime verification**: `cargo check` passes — all CUDA types and imports resolve. No integration test exists yet for the full diagonalize pipeline (Group F).
- **Diff validation**:
  - `diagonalize()` method on `ScfIteration<S, VEffBuilt>` ✓
  - Chains `chebyshev_filter` → `rayleigh_ritz` correctly ✓
  - Returns `ScfIteration<S, WavefunctionsUpdated>` with updated psi and eigenvalues ✓
  - V_eff downsampling utility (`downsample_array_to_wave_grid`) implemented via CPU rustfft ✓
  - **PcieAccount assertion will panic (Critical)**: The assertion at line 247 expects `psi_bytes + eig_bytes` D2H bytes, but `rayleigh_ritz` downloads eigenvalues via raw `clone_dtoh` (not tracked by PcieAccount). Only `psi_bytes` is tracked. At runtime this assertion fails.
  - **H2D assertion missing**: ADR-0002 specifies an H2D assertion alongside the D2H one. Neither is implemented.

## Issues Found

### P1 (Critical) — PcieAccount eigenvalue D2H not tracked → assertion panics

**File**: `src/scf.rs:246-251`, `src/eigensolver/rayleigh_ritz.rs:204-206`
**Description**: The `rayleigh_ritz` function downloads eigenvalues to host via `stream.clone_dtoh(&eigenvalues_dev)` which bypasses `PcieAccount`. The caller's assertion expects `pcie.d2h_bytes == psi_bytes + eig_bytes`, but only `psi_bytes` from the `sync_to_host_with` call is tracked. Runtime assertion failure on any execution path that reaches the assertion.

### P2 (Significant) — FFT dimension ordering mismatch

**File**: `src/eigensolver/chebyshev.rs:662-664`
**Description**: `plan_batched_c2c(ngx, ngy, ngz)` passes `[ngx, ngy, ngz]` to cuFFT, interpreting ngx as the slowest-varying dimension and ngz as fastest. But the scatter/gather index formula uses `ix + ngx * (iy + ngy * iz)`, which makes ngx the fastest-varying dimension. cuFFT requires `[ngz, ngy, ngx]` to match this layout. On cubic test grids (8×8×8, 4×4×4) the bug is masked because all dimensions are equal. On real non-cubic systems (e.g., Cu111_CO slab where ngz > ngx = ngy), G-vector coefficients scatter to wrong frequency positions, producing incorrect physics.

**Fix**: Change to `plan_batched_c2c(ngz as i32, ngy as i32, ngx as i32, ...)`.

### P3 (Minor) — Flat norm threshold instead of relative ratio

**File**: `src/eigensolver/chebyshev.rs:552-563`
**Description**: The guidance specifies a ratio-to-ratio comparison (`growth > threshold × last_growth`), but the implementation uses a flat `10×` threshold. For systems where Chebyshev naturally produces >10× growth, this would false-positive trigger divergence. Deferred item #10 in `deferred.md` documents this.

### P4 (Minor) — `#[allow(dead_code)]` on `SpectralBounds` is unnecessary

**File**: `src/eigensolver/chebyshev.rs:213`
**Description**: The `SpectralBounds` struct is fully used — `compute_spectral_bounds` constructs it and `chebyshev_filter` reads all fields. Remove the attribute.

## Deferred Items

- **Missing H2D assertion**: ADR-0002 specifies `assert_eq!(pcie.h2d_bytes, psi_bytes + veff_bytes, ...)` but only the D2H counterpart exists. Requires routing raw `clone_htod` calls (fft_idx_dev, kinetic_dev, VNL data) through PcieAccount. See `deferred.md`.
- **CudaKernelSet lives in chebyshev module**: `rayleigh_ritz` imports `CudaKernelSet` from `chebyshev` for the transpose kernel. Better to move to `eigensolver/kernels.rs` or `device/kernels.rs`. Deferred to avoid scope creep.
- **RowDistributed shape metadata**: The `[n_bands, n_pw]` shape doesn't reflect the transposed memory layout. No current code path syncs RowDistributed to host, so this is latent. Fix when adding RowDistributed host access.
- See `notes/pr-reviews/phase-2/deferred.md` for all previously deferred items.

---

# Review: Phase 2 Group D — construct_density + build_v_eff

**Tasks**: `notes/plans/phase-2/TASKS.md` (Group D, lines 330-381)
**Reviewed**: 2026-05-19
**Review scope**: Group D only (D-1: Batched density construction on GPU, D-2: V_eff assembly wrapper)

## Summary

**Verdict: CHANGES REQUIRED** — 2 critical bugs, 2 moderate issues.

The architectural direction is sound. The `BuildVEff` trait cleanly handles NonSpin/SpinCollinear dispatch. The `into_phase` helper centralizes state transitions. Crate boundaries are respected. However, the density construction has two critical bugs that would produce incorrect physics: an inverted occupation formula (wrong density from wrong bands) and a broken kernel launch (shared memory not allocated + wrong grid dimensions).

Outcome verification: `cargo check` passes. All 10 GPU unit tests pass (Group B tests). `cargo test --workspace` clean. No Group D integration tests exist yet (Group F handles fixture-anchored validation).

## Per-Task Results

### D-1: Batched density construction on GPU
- **Status**: ✗ Failed — 2 critical bugs
- **File**: `src/density.rs` (spec says `src/scf/density.rs` — minor drift)
- **Runtime verification**: No dedicated test (Group F). Compilation passes.
- **Diff validation**:
  - Occupation computation with bisection search for μ ✓ (but formula is inverted)
  - Scatter sparse PW → FFT grid, reusing `scatter_pw_to_grid` kernel ✓
  - Batched C2C IFFT via `BatchedFftPlan3d` ✓
  - ρ[r] = 1/Ω · Σ occ_b · |ψ_b[r]|² via `accumulate_density` kernel ✓ (but launch config wrong)
  - D2H → Density ✓
  - **Critical — occupation sign inversion**: `erfc((μ - ε)/w)` should be `erfc((ε - μ)/w)`. Bands below μ (physically occupied) get near-zero weight; bands above μ get near-full weight.
  - **Critical — accumulate_density launch broken**: `LaunchConfig::for_num_elems` gives wrong grid dimension (ceil(grid_size/1024) blocks instead of grid_size blocks) AND `shared_mem_bytes = 0` while kernel uses `extern __shared__`.
  - C2C rather than spec'd C2R (acceptable for Phase 2; C2C is correct for general k-points)
  - Function named `construct_density_gpu` not `build_density_from_wavefunctions` (internally consistent)

### D-2: V_eff assembly wrapper
- **Status**: ✓ Passed
- **File**: `src/scf.rs`
- **Runtime verification**: Compiles. No runtime test (needs VEffBuilder + fixtures, Group F).
- **Diff validation**:
  - `build_v_eff()` on `ScfIteration<S, Initialized>` implemented ✓
  - V_eff computed via `VEffBuilder::assemble_on_fine_grid` ✓
  - `BuildVEff` trait unifies NonSpin and SpinCollinear dispatch ✓
  - V_eff stored in `self.v_eff` with transition to `VEffBuilt` phase ✓
  - `into_phase` helper method centralizes struct literal ✓
  - Spec's D2H/H2D steps not literally present (V_eff/V_eff are CPU-side types — architecturally correct)
  - SpinCollinear path uses zero-spin placeholder (acceptable for Phase 2)

## Issues Found

### Critical

1. **Occupations formula inverted (`src/density.rs:43`)**
   - `erfc((μ - ε_b) / width)` should be `erfc((ε_b - μ) / width)`. Bisection comparison at line 68 must also flip.
   - Effect: bands below μ (occupied) get near-zero weight; bands above μ (empty) get near-full weight. Density built from wrong wavefunctions → wrong total energy.
   - **Fix**: Change erfc argument order; flip bisection: `if sum > n_electrons { hi = mid; } else { lo = mid; }`.
   - **Severity**: Physics-correctness blocking.

2. **`accumulate_density` kernel launch broken (`src/density.rs:169`)**
   - Two compounding bugs in one `LaunchConfig::for_num_elems(grid_size)` call:
     - **2a — Grid dimension**: Formula computes `grid_dim = (ceil(grid_size/1024), 1, 1)`, but kernel uses `int r = blockIdx.x` as grid-point index, expecting one block per grid point. For a 64³ grid (262,144 points), only 256 blocks launched → 261,888 points uncomputed.
     - **2b — Shared memory**: `shared_mem_bytes = 0` but kernel uses `extern __shared__ double sdata[]` for block reduction. UB — writes to unmapped shared memory, silent garbage or illegal address error.
   - **Fix**: Replace with `LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 8 }`.
   - **Severity**: Runtime correctness blocking.

### Moderate

3. **File location drift (`src/density.rs` vs `src/scf/density.rs`)**
   - Spec says `src/scf/density.rs` but module lives at `src/density.rs`.
   - **Recommendation**: Move to `src/scf/` for consistency, or document the deviation.

4. **No PcieAccount tracking in `construct_density`**
   - H2D transfers (psi, fft_indices, occupations) and D2H (rho) are not tracked. `diagonalize()` has PcieAccount tracking; density construction should too for monitoring parity.
   - **Recommendation**: Add PcieAccount tracking.

### Minor

5. C2C rather than spec'd C2R — functionally correct and works for general k-points; less efficient but acceptable for Phase 2.
6. Function name `construct_density_gpu` differs from spec's `build_density_from_wavefunctions`.
7. Smearing changed from Fermi-Dirac (spec) to Gaussian/erfc — intentional correction: CASTEP defaults to Gaussian.
8. D-2 spec's D2H/H2D steps not literally present — correct by architecture (types are CPU-side).
9. `BuildVEff` trait + `SpinCollinear` dispatch are scope additions beyond D-2.

## Deferred Items

See `deferred.md` (Group D section).

## Fix Tasks

See `fix-tasks.md` (Group D section).

---

# Review: Phase 2 Group E — mix + check

**Tasks**: `notes/plans/phase-2/TASKS.md` (Group E, lines 384-573)
**Reviewed**: 2026-05-19
**Post-hoc tests**: `notes/plans/phase-2/POST_GroupE_TASKS.md`
**Review scope**: Group E (E-1 through E-5, plus E-6 unit tests)

## Summary

**Verdict: CHANGES REQUIRED — 1 blocking gap (now fixed), 2 deferred items, 1 observation.**

The architecture is sound: type-state `DensityHistory<M>` correctly encodes the three mixing phases, the Kerker formula matches CASTEP, the DIIS solve matches the source-audited algorithm in `dm.f90:895-1093`, and the Ewald+total-energy assembly follows the KS-DFT formula. 22 unit tests all pass, with 5 new phase-transition tests added during this review.

The blocking gap was that `check()` never advanced `next_mixing` from `Off` — the SCF loop always used pass-through density construction.  This has been fixed in this review session.

Outcome verification: `cargo test --lib` → 27/27 pass (was 22). `cargo clippy --workspace -- -D warnings` → clean. No fixture-anchored tests exist yet (Group F).

## Per-Task Results

### E-1: Kerker preconditioner setup
- **Status**: ✓ Passed
- **File**: `src/mixing/kerker.rs` (146 lines)
- **Runtime verification**: 4 CPU unit tests pass (G=0 zero, monotonic, high-G asymptote, q² gives 0.5). GPU allocation not tested at unit level (deferred to Group F).
- **Diff validation**:
  - K(G) = G²/(G²+q²) with q² = 2.25 (q=1.5 a.u.) ✓
  - K(G=0) = 0.0 (no DC mixing) ✓
  - Kernel computed on CPU from `GVectorGrid::g2()`, then H2D ✓
  - Fortran layout matches cuFFT C2C plan ordering ✓
  - **Tests are formula-only**: validate Kerker math, not GPU allocation or mixing behaviour. Acceptable for CPU-only testing.

### E-2: Reciprocal-space DIIS + Kerker mixing
- **Status**: ✓ Passed
- **Files**: `src/mixing.rs` (768 lines), `src/mixing/cuda_kernels.rs` (100 lines), `src/mixing/reciprocal_density.rs` (45 lines)
- **Runtime verification**: 5 DIIS solve tests pass (2×2, 3×3 identity, singular fallback, near-singular fallback, empty). No GPU-resident mixing test (deferred to Group F).
- **Diff validation**:
  - Type-state `DensityHistory<M>` with `MixingOff`, `Kerker`, `Pulay` markers ✓
  - Sealed trait prevents external implementations ✓
  - `MixingOff::mix()` pass-through ✓
  - `Kerker::mix()`: density → C2C FFT → R=n_out−n_in → K·R → C2C-IFFT → real ✓
  - `Pulay::mix()`: full DIIS with delta history, dgesv solve, Kerker preconditioned update ✓
  - cuBLAS `axpy_c64` for accumulating Σc_i·ΔR_i, Σc_i·Δn_i ✓
  - NVRTC-compiled `cpx_sub`, `cpx_full_update` kernels ✓
  - `build_and_solve_diis` builds M_ij = Re[zdotc(ΔR_j, ΔR_i)] on GPU, solves on CPU ✓
  - DIIS fallback to Kerker on singular matrix ✓
  - Ring-buffer history eviction at DIIS_MAX_HISTORY=7 ✓
  - **Mixing is in reciprocal space** as specified ✓
  - **Residual is R = n_out − n_in** (not ρ_new − ρ_old) ✓
  - **History stores deltas** (Δn, ΔR), not absolutes ✓
  - **DIIS is unconstrained** (no Lagrange multiplier) ✓
  - **Kerker preconditioning is mandatory** (applied per PW coefficient) ✓
  - `cpx_full_update` implements n_new = n_in + Σc_i·Δn_i + K·(R + Σc_i·ΔR_i) ✓
  - First-call pass-through for both Kerker and Pulay (no n_in to compare against) ✓

### E-3: Total energy computation
- **Status**: ✓ Passed (minor spec deviation)
- **Files**: `src/energy.rs` (323 lines)
- **Runtime verification**: 2 CPU tests pass. `test_ewald_is_finite` tests zero-charge case (E=0). `test_assemble_total_energy_basic` verifies arithmetic.
- **Diff validation**:
  - Ewald summation: real-space + reciprocal-space + self-energy ✓
  - Real-space sum with erfc screening, cutoff=8 Bohr ✓
  - Reciprocal sum with exp(−G²/4α²) factor, cutoff G_max=7α ✓
  - Structure factor S(G) = Σ Z_I exp(iG·r_I) ✓
  - Self-energy: −α/√π · Σ Z_I² ✓
  - `assemble_total_energy`: E_band − E_H + E_xc − ∫ρV_xc + E_ewald ✓
  - chemrust-hamiltonian `VEffWithEnergy` provides e_xc, e_hartree, rho_vxc from `assemble_with_energy()` ✓
  - Hartree and ∫ρV_xc use valence density (NLCC core is frozen/non-variational) ✓
  - **Ewald α parameter**: code uses `(π/V)^(1/3)`, TASKS.md specifies `√π/V^(1/3)`. Both converge to the same Ewald energy with adequate cutoffs — not a physics error. Deferred.
  - **test_ewald_is_finite is borderline placebo**: tests Ewald with Z_I=0 (no pseudopotential charges), so all three Ewald terms are zero. Validates the code doesn't panic but doesn't test any of the Ewald sub-terms. Not blocking — real-charge testing requires fixture pseudopotentials (Group F).

### E-4: Energy-window convergence check
- **Status**: ⚠ Fixed during review — was ✗ Failed
- **File**: `src/scf.rs` (check() method, lines 720-807)
- **Original state**: `check()` copied `self.next_mixing` through unchanged — no phase transitions. Convergence did not require mixing to be active.
- **Fixed state**:
  - Energy convergence uses max−min over 3-entry window (matches CASTEP `electronic_store_energy`) ✓
  - Off → Kerker when energy variation < 0.1 eV (`MIXING_CONV_TOL_EV`) ✓
  - Kerker → Pulay unconditionally (first DIIS step after one Kerker mix) ✓
  - Pulay → Pulay (stay in DIIS) ✓
  - Convergence requires `mixing_was_active` (next_mixing ≠ Off at entry) ✓
  - Mixed-status guard prevents false convergence when mixing is off ✓
- **Tests added**: 5 new tests exercise all transitions and the mixed-status guard:
  - `test_check_off_to_kerker_transition` — energy settled → Kerker
  - `test_check_off_stays_off_when_energy_unstable` — energy varying → stays Off
  - `test_check_kerker_to_pulay_transition` — Kerker → Pulay
  - `test_check_pulay_stays_pulay` — Pulay → Pulay
  - `test_check_no_converge_with_mixing_off` — flat energy + zero RMS but Off → NOT converged

### E-5: Wire mixing phase into run_scf loop
- **Status**: ✓ Passed
- **File**: `src/scf.rs` (run_scf, lines 828-890)
- **Runtime verification**: `cargo check` passes. Phase transition tests validate the dispatch indirectly.
- **Diff validation**:
  - `MixingPhaseKind` runtime enum (Off, Kerker, Pulay) ✓
  - `run_scf` matches on `wfn.next_mixing` to dispatch ✓
  - `construct_density_off/kerker/pulay` methods on `WavefunctionsUpdated` ✓
  - Each `construct_density_*` correctly wires the history type: Off→MixingOff, Kerker→Kerker, Pulay→(Off→)Kerker→Pulay ✓
  - After `mix()`, history normalized back to `MixingOff` via `into_off()` ✓
  - `run_scf_with_energy` variant uses `build_v_eff_with_energy` for total energy ✓
  - `CheckOutcome::NotConverged` carries `next_mixing` to next iteration ✓

### E-6: CPU unit tests
- **Status**: ✓ Passed (now 27 tests, was 22)
- **Coverage**: 5 DIIS solve + 4 Kerker formula + 1 check (original) + 5 phase transition (added) + 3 device + 3 FFT + 3 BLAS + 1 solver + 2 energy = 27

## Issues Found

### Fixed during review

1. **Mixing phase never transitions (P1 — was Critical)**
   - `check()` at `src/scf.rs:779` (old) copied `self.next_mixing` through, never advancing from Off.
   - Effect: SCF loop always called `construct_density_off()?.mix()` — pass-through, no Kerker or DIIS mixing ever engaged.
   - Fix: Implemented phase transition logic with `MIXING_CONV_TOL_EV = 0.1 eV`. Off→Kerker→Pulay state machine now advances correctly. Mixed-status guard (`mixing_was_active`) prevents false convergence when mixing is off.
   - Verified: 5 new tests pass, all 27 tests green.

### Deferred

2. **Kerker mix() D2H/H2D roundtrip for current_density_in storage (P3 — Performance)**
   - `Kerker::mix()` and `Pulay::mix()` save `current_density_in` via `clone_dtoh` → `clone_htod` roundtrip.
   - Wastes a PCI-E roundtrip per mix call. Should use `stream.alloc_clone(&result_dev)` or similar GPU-side copy.
   - **Defer to Phase 3**: negligible for Phase 2 correctness validation.

3. **New cuFFT plan allocated per mix() call (P3 — Performance)**
   - `FftPlan3d::plan_c2c(...)` is called inside every `mix()`. cuFFT plan creation is expensive.
   - Should be cached in `DensityHistory`.
   - **Defer to Phase 3**: correctness-first for Phase 2.

4. **Ewald α parameter differs from TASKS.md (P3 — Minor spec deviation)**
   - Spec: `α = √π / V^(1/3)`. Code: `α = (π/V)^(1/3)`.
   - Ewald energy converges to same value with adequate cutoffs. Not a physics error.
   - **Defer**: verify against CASTEP reference in Group F; adjust if discrepancy exceeds 1e-6 eV.

### Observations (no fix needed)

5. **test_ewald_is_finite tests Z_I=0 case only**: Validates code structure, not physics. Acceptable for CPU-only testing — real-charge testing needs fixture pseudopotentials (Group F). Not placebo per se, but low signal.

6. **check() convergence formula uses max−min instead of pairwise diffs**: This was changed during the review fix and actually aligns BETTER with CASTEP's `electronic_store_energy` which uses `max_E − min_E`. The old pairwise approach was stricter but not wrong.

## Deferred Items

See `deferred.md` (Group E section) for the 3 deferred items listed above.

## Fix Tasks

See `fix-tasks.md` (Group E section).
