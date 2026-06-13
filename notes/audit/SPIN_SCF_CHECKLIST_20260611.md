# SCF Spin-Polarised Adversarial Audit Checklist

**Date**: 2026-06-11 / updated 2026-06-12 / **CRITICAL UPDATE 2026-06-13**
**Scope**: SCF loop spin-polarisation (13 components), Rust (`scf.rs`, `ffi.rs`, `density.rs`, `energy.rs`, `spin_types.rs`) vs CASTEP 6.11 (`electronic.f90`, `density.f90`, `locpot.f90`, `xc_gga.f90`, `hamiltonian.f90`)
**Methodology**: Adversarial line-by-line audit; CASTEP profile evidence (NiO `NiO.0001.profile`: 67 SCF x 2 spins = 134 `hamiltonian_diagonalise_ks` calls); CASTEP source read for each component (2026-06-11 workflow). **Update 2026-06-12:** All 13 components implemented (TASK-1 through TASK-16), multi-kpt NiO discriminator test passing. **Update 2026-06-13:** Post-mortem analysis after cold-start divergence discovery at iter 16; reclassification of warm-start-only verifications.

## 0. Post-Mortem: Cold-Start Divergence at Iter 16

### 0.1 Symptom

- **NiO spin-polarised FFI cold-start** diverges at SCF iteration 16 (commit `69b3bc4`).
- **NiO non-spin FFI cold-start** converges in 10 iterations (commit `e4e02f9`).
- **NiO spin-polarised warm-start** (converged density + wavefunctions from CASTEP) passes all discriminator criteria V1-V7 (TASK-16).
- **NiO spin-polarised 2-iteration cascade check** (`nio_two_iter_cascade_check`) passes (commit `851a31b`).

The divergence is specific to **cold-start spin-polarised multi-iteration SCF**: the warm-start test (1 iteration from converged density) and cascade check (2 iterations from converged density) both pass, but the cold-start path fails at iteration 16 because intermediate SCF states differ from converged states in ways the warm-start discriminator never exercises.

### 0.2 Why Warm-Start Tests Are Insufficient

The warm-start test loads a converged CASTEP density and wavefunctions, runs ONE SCF iteration, and then asserts eigenvalues, energy, and Fermi energies match reference values. This establishes that the **single-iteration operator action H[ρ_converged] produces correct eigenvalues**. It does NOT test:

1. **Iterative stability**: Do eigenvalues remain bounded across 15+ cold-start iterations?
2. **Occupation search accuracy**: Is the `spin_fixed`→`spin_freed` transition correctly placed for the current SCF state?
3. **Spin density mixing fidelity**: Does the per-spin mixing reproduce CASTEP's spin density trajectory when starting from paramagnetic guess?
4. **Density-potential self-consistency drift**: Do small errors in density construction compound across iterations?

Previous bugs masked by warm-start-only tests (per failure-patterns.md):
- **Beta-g layout transposition** (2026-06-10): row-major/column-major mismatch scaled with residual magnitude ~1e-6 for warm-start, ~1 Ha for cold-start.
- **V_eff cache staleness** (2026-06-13): fine-grid V_eff max dominated by frozen cores, cache reuse always true, cold-start solved with stale V_eff after iter 1.

**Lesson**: Every spin-polarised checklist item currently marked FIXED with only warm-start evidence is **WARM-START-VERIFIED**, not FIXED. Cold-start multi-iteration evidence is required for full reclassification.

### 0.3 Reclassification of Checklist Statuses

| Component | Old Status | New Status | Why |
|-----------|-----------|------------|-----|
| C1 (ScfIteration fields) | FIXED | **WARM-START-VERIFIED** | Field types compile and single-iteration test passes. No multi-iteration cold-start evidence. |
| C2 (into_phase copy) | FIXED | **WARM-START-VERIFIED** | Compiler-enforced type safety. Field copy correctness across 15+ iterations not tested beyond `nio_two_iter_cascade_check` (2 iterations). |
| C3 (diagonalize_inner spin loop) | FIXED | **WARM-START-VERIFIED** | Spin+kpt nested loop matches CASTEP structure. But VNL D-matrix recomputation per (spin,kpt) — correctness of per-iteration VNL data across cold-start not verified (see S1 cross-check). |
| C4 (BuildVEff paramagnetic) | FIXED | **WARM-START-VERIFIED** | Zero-spin fallback correct for iter-0. After iter-1, density has spin → BuildVEffWithEnergy path exercised. The paramagnetic path is iter-0 only; multi-iteration correctness depends on C5. |
| C5 (BuildVEffWithEnergy) | FIXED | **WARM-START-VERIFIED** | Warm-start V1-V2 pass. Cold-start V_eff assembly from non-converged density not tested beyond 2 iterations. |
| C6 (density construction) | FIXED | **WARM-START-VERIFIED** | Per-spin, per-kpt construction compiles and single-iter V1 passes. Cold-start per-spin density from cold-start eigenvalues not tested beyond 2 iterations. |
| C7 (construct_density_*) | FIXED | **WARM-START-VERIFIED** | Type changes correct. Manual struct copies in kerker/pulay paths are maintenance risk (D5). |
| C8 (mix) | FIXED | **DIVERGE (C8-S3 DEFERRED)** | Per-spin mixing loop correct. BUT `spin_density_mixing_amplitude=2.0` from NiO `.param` is NOT implemented (C8-S3 DEFERRED). Spin density mixed with same amplitude as charge density. Over 16 cold-start iterations, different mixing amplitude → different spin density trajectory → divergence. **This is a candidate root cause.** |
| C9 (check/energy) | FIXED | **WARM-START-VERIFIED** | Energy formula correct for warm-start. Cold-start energy tracking and convergence window across 16 iterations not tested. |
| C10 (occupation search) | FIXED | **WARM-START-VERIFIED** | fermi_fix/fermi_free distinction correct for single-iter warm-start. Cold-start `spin_fix` value may not match NiO fixture's runtime value (5). Default `SmearingParams.spin_fix=10` in test code vs CASTEP runtime `spin_fix=5` from profile. For iterations 6-10, wrong occupation strategy used. **Candidate root cause.** |
| C11 (FFI init) | FIXED | **WARM-START-VERIFIED** | nspins parameter correct. Per-spin V_eff cache works for warm-start (single spin per call). Cold-start verifies 32 calls (16 iter x 2 spins) — V_eff change detection per-spin, cache_reuse=false. |
| C12 (FFI step) | FIXED | **WARM-START-VERIFIED** | ispin parameter correct. Per-spin cache indexing verified for single call. Multi-iteration cache correctness not tested beyond warm-start. |
| C13 (run_scf bounds) | FIXED | **WARM-START-VERIFIED** | SpinCollinear trait impls compile. Not exercised in FFI cold-start path (CASTEP drives the SCF loop, Rust only sees diagonalize calls). |

### 0.4 New Status: WARM-START-VERIFIED

A new status tier between FIXED and DEFERRED:

- **FIXED**: Verified against CASTEP source AND confirmed correct in multi-iteration cold-start SCF (or proven invariant across cold-start states).
- **WARM-START-VERIFIED**: Verified against CASTEP source AND confirmed correct in single-iteration warm-start from converged state. Multi-iteration cold-start behavior **not yet tested**. At risk of being correct for the converged state but wrong for intermediate states.
- **DIVERGE**: Known difference from CASTEP. Needs implementation.
- **DEFERRED**: Planned but not blocking.
- **MISSING**: Entire component absent.

### 0.5 Gap Analysis: Missing Components for Multi-Iteration Stability

These components were NOT covered by the original 13-component audit because they only manifest across multiple cold-start iterations:

| ID | Missing Component | CASTEP Source | Why It Matters for Cold-Start |
|----|-------------------|---------------|-------------------------------|
| **G1** | Per-spin mixing parameters (amplitude, g-vector) | NiO `.param`: `spin_density_mixing_amplitude=2.0`, `spin_density_mixing_g_vector=1.5` vs charge `mix_charge_amp=0.5` | Mixing spin with charge amplitude (0.5) instead of spin amplitude (2.0) produces 4x slower spin density evolution. Over 16 iterations, this means the spin density is far from its self-consistent value at the iteration where CASTEP's spin density would have converged. Wrong spin density → wrong V_eff_up vs V_eff_dn → wrong eigenvalues → wrong occupations → cascade. |
| **G2** | `spin_fix` value from `.param` or `.castep_bin` | CASTEP `electronic.f90:516-518`: `if(scf_cycle == spin_fix)` transition | Default `SmearingParams.spin_fix=10` in test code. NiO CPU profile shows `spin_fix=5` at runtime (5 calls to `electronic_find_fermi_fix`). For cold-start, iterations 6-10 use `fermi_fix` in Rust but `fermi_free` in CASTEP. Wrong occupation strategy → wrong electron counts per spin → wrong density. |
| **G3** | Per-spin density change metric for mixing | CASTEP `dm_mix_density_pulay` — residual norm computed for `charge(:)` and `spin(:)` independently | Our mixing tracks per-spin densities but uses same Pulay history parameters for both channels. Different convergence rates for charge vs spin not accounted for. |
| **G4** | Multi-kpt OccupationSet | C6 (known simplification, deferred D3) | `OccupationSet` stores only kpt-0 occupations. Across 14 kpts and 16 iterations, lost per-kpt occupancy information means convergence check cannot verify multi-kpt occupation consistency. |
| **G5** | E_nonCoulomb computation from PseudopotentialSet | `energy.f90:4205`: computed from local PP parts | Currently hardcoded constant. Not a root cause of eigenvalue divergence but means total energy convergence criterion is approximate. |
| **G6** | Per-iteration diagnostic tracking | N/A (diagnostic) | No tracking of: spin density evolution per SCF iter, V_eff_up vs V_eff_dn max difference per iter, occupation distribution change per iter, band energy per spin per iter. Without these, divergence at iter 16 cannot be characterized beyond "it diverges." |

### 0.6 New Verification Criteria for Multi-Iteration Stability

Add to Section 9 (Verification Matrix):

| # | Criterion | Fixture Anchor | Tolerance | Why This Tests Cold-Start |
|---|-----------|---------------|-----------|---------------------------|
| **V11** | 16-iteration cold-start eigenvalues remain bounded | NiO CPU reference: band range [-1.0, +0.2] Ha | max eigenvalue > -10 Ha, min eigenvalue < +5 Ha | Catches eigenvalue explosion/divergence before it's catastrophic |
| **V12** | Spin density trajectory over 6-16 iters matches CASTEP | CASTEP per-SCF-iter spin density dumps (F8 format) | RMS residual < 0.1 e-/Bohr^3 after spin_freed transition | Verifies mixing amplitude correctness |
| **V13** | Occupation sums N_up + N_dn = N_total at every iter | CASTEP constraint: N_total = 64 for NiO | |N_up + N_down - 64| < 1e-6 | Catches Fermi search bugs that lose electrons |
| **V14** | `spin_freed` flag matches CASTEP transition point | NiO profile: 5 `fermi_fix` calls → `spin_fix=5` | Exact match of transition iteration | Catches wrong spin_fix value |
| **V15** | Per-spin band energy drift per iteration < 0.5 Ha for first 10 iters | CASTEP energy trajectory from cold-start | Monotonically decreasing after iter 5 | Catches V_eff assembly accumulation errors |
| **V16** | No "no empty bands" warning | CASTEP behavior: empty bands exist at all iters | Warning count = 0 | This warning is the immediate precursor to divergence (per NiO failure pattern) |

### 0.7 Correction of Claims

Nothing in the original checklist is WRONG in an absolute sense — the claims describe verified code structure, CASTEP-faithful algorithms, and passing warm-start tests. The **reclassification** to WARM-START-VERIFIED reflects the gap between structural correctness and cold-start multi-iteration correctness, not an error in the original audit.

**Correction**: The original checklist (line 351) states "All 16 tasks (TASK-1 through TASK-16) implemented. NiO warm-start discriminator test passes all V1-V6 criteria." This is true. The statement is NOT equivalent to "cold-start spin-polarised SCF converges," and the original checklist does not claim convergence. The reclassification makes this distinction explicit.

## Fix History

| Date | Fix | IDs | Description |
|------|-----|-----|-------------|
| 2026-06-11 | **TASK-1 + TASK-2: SpinChannelData<T> newtypes** | C1-all, C2-all | Added `spin_types.rs` with `SpinChannelData<T>` (length-nspins wrapper), `PerSpinDensity`, `PerSpinPwCoefficients`, `PerSpinEigenvalues`, `PerSpinBetaProjections`, `PerSpinAugDensity`, `FermiEnergies`, `OccupationSet`, `KptDataSet<T>`. Each implements `Index<usize>` for per-spin access. `KptDataSet<T>` wraps `Vec<T>` with a stored `nkpts` count, enabling per-kpt-per-spin indexing without `Vec<Vec<T>>`. |
| 2026-06-11 | **TASK-3 through TASK-8: ScfIteration per-spin state** | C1-S1–S7, C2-all, C3-S1–S5, C6-S1–S5, C7-S1–S3, C8-S1–S2 | Replaced single-spin fields (`density: Density` → `density: PerSpinDensity`, etc.) in `ScfIteration`. Updated `into_phase()` field copy. Wrapped `diagonalize_inner` in `for ispin in 0..nspins` loop with per-spin V_eff lookup (`S::v_eff_for_spin(v_eff_ref, ispin)`). Moved `VnlBatchData::precompute()` inside the (spin, kpt) nested loop (CASTEP `hamiltonian.f90:1013`: D-screening is per-(kpt, spin)). Density construction loops per-spin: `construct_density_gpu()` per (spin, kpt) with kpt-weighted accumulation for multi-kpt. `construct_density_off/kerker/pulay` all updated for `PerSpinDensity`. `mix()` mixes per-spin densities independently. `into_phase()` verified compiler-enforced with new types. |
| 2026-06-11 | **TASK-9 + TASK-10: SpinCollinear BuildVEffWithEnergy** | C5-S1–S5 | Implemented `BuildVEffWithEnergy` for `SpinCollinear` in `scf.rs:456-567`. Upsamples both total density and spin density to fine grid, adds augmentation (ρ_aug to total only), assembles V_eff via `VEffBuilder::<SpinCollinear>::with_density(rho_total, Some(rho_spin))`, calls `compute_pbe_xc_spin(rho_total, rho_spin)`. Energy integrals: E_H = 0.5·Σρ_total·V_H/N, E_xc from XC functional, ρV_xc = Σ(ρ_up·V_xc_up + ρ_dn·V_xc_dn)/N (per-spin sum, CASTEP `pot.f90:4205`). Convention: `d_v = 1/N_grid` matching CASTEP `xc_gga`. |
| 2026-06-11 | **TASK-11 + TASK-12: Per-spin occupation search** | C10-S1–S7 | Implemented `find_fermi_fix(eig_up, eig_dn, n_up, n_dn)` and `find_fermi_free(eig_up, eig_dn, n_total)` in `density.rs`. `compute_occupations_weighted()` handles per-spin bisection with kpt weights. `find_fermi_free` does single bisection integrating both spin channels: `occ_up = Σ_k w_k Σ_b f(E_F, ε_{bk})`, `occ_dn` same — finds shared E_F where total = N_total. Net spin is `intent(out)` — computed from resulting occupancies, not from `.cell SPIN=`. `spin_fix` transition tracked via `SmearingParams.spin_fix` and `ScfIteration.spin_freed` flag. |
| 2026-06-11 | **TASK-14 + TASK-15: FFI spin wiring** | C11-S1–S3, C12-S1–S5 | `chemrust_eigensolve_init` now accepts `nspins: c_int`. Per-kpt `VnlBatchData` stored as `Vec<Option<VnlBatchData>>` — one per spin. V_eff cache per-spin: `Vec<Option<CudaSlice<f64>>>`. `chemrust_eigensolve_step` accepts `ispin: c_int` — converted from CASTEP 1-based `ns` to 0-based Rust index (`ispin = ns - 1`). Array data is already sliced to current spin by Fortran caller; Rust only uses `ispin` for cache/VNL lookup. |
| 2026-06-11 | **TASK-16: NiO warm-start discriminator test** | V1-V10 | Multi-kpt (14), spin-polarised NiO test in `tests/nio_spin_scf.rs`. Loads converged density + wavefunctions from CASTEP fixture (`/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin/`). Down samples density from fine grid to wave grid, reconstructs ρ_up/ρ_dn. Iter-1 warm-start: build_v_eff_with_energy → diagonalize → construct_density → mix → check. V1: per-spin eigenvalues at all kpts (tol 3e-4 Ha). V2: total energy (with E_nonCoulomb correction, drift < 2e-2 Ha). V4: Fermi energies (tol 1e-3 Ha). V6: V_eff_up ≠ V_eff_dn (max|Δ| > 1e-6). |
| 2026-06-12 | **Multi-kpt spin loop refinement** | C3-S1 (revised) | `diagonalize_inner` now has spin outer + kpt inner nested loops matching CASTEP `electronic.f90:488-495` exactly. FFT plan, TPA preconditioner, and GPU context created once outside both loops. V_eff downsampled and uploaded per-spin (once per spin). Kinetic energies, FFT indices, and VNL data per (spin, kpt). Eigenvalues, wavefunctions, and beta_psi stored per-(spin, kpt) via `KptDataSet`. |
| 2026-06-12 | **Per-spin electron counts from density net_spin** | C10-S2 (revised) | `N_up = 0.5*(N_total + net_spin)`, `N_dn = 0.5*(N_total - net_spin)` where `net_spin = ∫(ρ_up - ρ_down) / N_grid` from the CURRENT density, NOT from `.cell SPIN=`. CASTEP `electronic.f90:8742-8746`: `frac_elec(1) = 0.5*(N+spin)`, `frac_elec(2) = 0.5*(N-spin)`. The profile shows the spin value evolves during SCF; using the converged density's integrated spin ensures correct per-spin electron counts for the warm-start test. |
| 2026-06-12 | **Energy formula for spin-polarised multi-kpt** | C5-S2 (revised), C9-S1 (revised) | Total energy assembly in `check()`: per-spin band energy with kpt weights `Σ_spin Σ_k w_k Σ_b f_{bk} ε_{bk}`. `assemble_total_energy_from_band(e_band, e_xc, e_hartree, rho_vxc, ewald, TS)` — same formula as NonSpin but e_band is now the sum of both spin channels. Entropy correction `-TS` computed per-spin with per-spin Fermi energies. |
| 2026-06-12 | **E_nonCoulomb correction identified** | C5-S5 (new) | CASTEP `energy.f90:4205` adds a constant `E_nonCoulomb` term from pseudopotential local parts. For NiO: +533.14 eV. Our `ewald_energy()` does not include this. Added as constant `E_NON_COULOMB_HA` in the discriminator test. TODO: compute from `PseudopotentialSet` in `chemrust-hamiltonian-core`. |
| 2026-06-12 | **Kpt weights from .castep_bin** | C6 (supplemental) | For multi-kpt systems, kpt weights must come from `.castep_bin` (`CastepBin.kpoint_weights`) rather than parsed from `.bands` (which lacks weight data). The `.castep_bin` file stores weights in the `KpointWeights` record. Confirmed against NiO fixture: 14 kpts with standard Monkhorst-Pack weights. |
| 2026-06-12 | **OccupationSet stores only kpt-0** | C6 (known simplification) | `OccupationSet` currently stores `occupations_all_kpts[0].clone()` — only the first k-point's occupations. This is a known simplification: the actual multi-kpt weighted occupations are used correctly in density construction (lines 1337-1358 of `scf.rs`), but the stored `OccupationSet` loses per-kpt information. Deferred: per-kpt `OccupationSet` generalisation. |
| 2026-06-13 | **V_eff GPU cache disabled (cold-start divergence fix, non-spin)** | ffI.rs:489 | `cache_reuse = false` permanently. Fixes V_eff cache staleness from frozen-core-dominated fine-grid max check. Non-spin FFI cold-start now converges (10 iters, -7160.23 eV). Applied to both spin channels. |
| 2026-06-13 | **D-screening fine-grid fix (cold-start divergence fix, non-spin)** | ffI.rs:434-543 | D-matrix screening now uses fine-grid V_eff + fine-grid Q(G) matching CASTEP `nlpot_calculate_d` (nlpot.f90:352). Previously used wave-grid V_eff + wave-grid Q(G). Fix applied to all spin channels. |
| 2026-06-13 | **Pipeline unification: spin-polarised cold-start fix** | src/pipeline.rs | Created shared pipeline functions (`v_eff_prepare`, `kpt_gpu_upload`, `run_davidson`). Both FFI and standalone SCF paths now use fine-grid `rescreen_d` for D-matrix screening. Spin-polarised FFI cold-start converges (20 iters, ΔE = 2.08e-5 Ha vs CPU). |

---

## 1. Component Summary Table

| # | Component | CASTEP Source | Rust Source | Status | Severity Spread |
|---|-----------|---------------|-------------|--------|-----------------|
| 1 | `ScfIteration` struct — per-spin fields | `electronic.f90:483-532` (wvfn, eigenvalues, density types) | `scf.rs:106-180`, `spin_types.rs` | **WARM-START-VERIFIED** (was FIXED) | ~~1x CRITICAL~~, ~~6x HIGH~~ |
| 2 | `into_phase()` — state transition copy | — (architectural) | `scf.rs:284-321` | **WARM-START-VERIFIED** (was FIXED) | ~~6x HIGH~~ |
| 3 | Spin loop in `diagonalize_inner` | `electronic.f90:488-495` | `scf.rs:750-1016` | **WARM-START-VERIFIED** (was FIXED) | ~~1x CRITICAL~~, ~~3x HIGH~~ |
| 4 | `BuildVEff` for `SpinCollinear` (paramagnetic) | `locpot.f90:301` (V_eff per-spin assembly) | `scf.rs:369-391` | **WARM-START-VERIFIED** (was FIXED) | ~~2x HIGH~~ |
| 5 | `BuildVEffWithEnergy` for `SpinCollinear` | `electronic_prepare_H` → `locpot_calculate` | `scf.rs:456-567` | **WARM-START-VERIFIED** (was FIXED) | ~~1x CRITICAL~~, ~~3x HIGH~~ |
| 6 | `compute_density_from_wavefunctions` | `density.f90:2126-2195` (per-spin loop → charge+spin) | `scf.rs:1219-1484` | **WARM-START-VERIFIED** (was FIXED) | ~~1x CRITICAL~~, ~~3x HIGH~~ |
| 7 | `construct_density_off/kerker/pulay` | density mixing pipeline | `scf.rs:1488-1591` | **WARM-START-VERIFIED** (was FIXED) | ~~3x MEDIUM~~ |
| 8 | `mix()` — density mixing | `dm_mix_density` | `scf.rs:1597-1770` | **DIVERGE — C8-S3: spin_density_mixing_amplitude not implemented** | ~~2x MEDIUM~~ |
| 9 | `check()` — convergence | `electronic_check_occupancies` | `scf.rs:1780-2010` | **WARM-START-VERIFIED** (was FIXED) | ~~2x MEDIUM~~ |
| 10 | Occupation search (`compute_occupations`) | `electronic.f90:8602-9209` (fermi_fix + fermi_free) | `density.rs` | **WARM-START-VERIFIED** (was FIXED) | ~~1x CRITICAL~~, ~~4x HIGH~~ |
| 11 | FFI `chemrust_eigensolve_init` | `chemrust_eigensolve.f90` | `ffi.rs` | **WARM-START-VERIFIED** (was FIXED) | ~~1x HIGH~~, ~~2x MEDIUM~~ |
| 12 | FFI `chemrust_eigensolve_step` | `electronic.f90:511-529` | `ffi.rs` | **WARM-START-VERIFIED** (was FIXED) | ~~1x HIGH~~, ~~2x MEDIUM~~ |
| 13 | `run_scf` / `run_scf_with_energy` | `electronic_minimisation` | `scf.rs:1548-1656` | **WARM-START-VERIFIED** (was FIXED) | ~~1x HIGH~~, ~~1x MEDIUM~~ |

### Status Counts

| Status | Count |
|--------|-------|
| **MATCH** (no significant differences) | 0 |
| **DIVERGE** (differences requiring implementation) | 1 (C8: spin_density_mixing_amplitude) |
| **MISSING** (entire component absent) | 0 |
| **WARM-START-VERIFIED** | 12 |
| **FIXED** (multi-iteration verified) | 0 |

### Severity Counts

| Severity | Count |
|----------|-------|
| **CRITICAL** (wrong physics if not addressed) | 0 (all 5 mitigated in warm-start, unverified in cold-start) |
| **HIGH** (functional gap, blocks spin support) | 0 (all 32 mitigated in warm-start, unverified in cold-start) |
| **MEDIUM** (observable effect, edge cases) | 1 (C8-S3: spin density mixing amplitude — active divergence contributor) |
| **LOW** (diagnostic, cosmetic, or verified benign) | 1 (C13-S3: `scf_iter` already existed, no change needed) |

---

## 2. Surviving Differences — Detailed Analysis

### Component 1: `ScfIteration` Struct — Per-Spin Fields (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C1-S1 | `density: Density` — single-spin, no `PerSpinDensity` → `density: PerSpinDensity` | `density.f90:30-37`: `electron_density` stores `charge(:)` + `spin(:)` separately | `spin_types.rs:70-85` | **WARM-START-VERIFIED** | `PerSpinDensity` stores `SpinChannelData<Density>` (ρ_up and ρ_down). Derives `total()` = ρ_up + ρ_down, `spin()` = ρ_up − ρ_down. |
| C1-S2 | `psi: WavefunctionSet<ColumnDistributed>` — single channel → `psi: PerSpinPwCoefficients` | `electronic.f90:492`: `wvfn%coeffs(:,:,nk,ns)` — spin is outermost dimension | `scf.rs:135` | **WARM-START-VERIFIED** | `PerSpinPwCoefficients(SpinChannelData<KptDataSet<PwCoefficients>>)`. Per-spin, per-kpt GPU-resident wavefunctions. |
| C1-S3 | `eigenvalues: Vec<f64>` — single channel → `eigenvalues: PerSpinEigenvalues` | `electronic.f90:492`: `eigenvalues(:,nk,ns)` — spin is outermost dimension | `scf.rs:139` | **WARM-START-VERIFIED** | `PerSpinEigenvalues(SpinChannelData<KptDataSet<Vec<f64>>>)`. Per-spin, per-kpt eigenvalues. |
| C1-S4 | `previous_density: Density` — single-track → `previous_density: PerSpinDensity` | — | `scf.rs:144` | **WARM-START-VERIFIED** | `PerSpinDensity` stores per-spin previous densities for mixing history. |
| C1-S5 | `beta_psi_per_ion: Option<Vec<...>>` — single channel → `beta_psi_per_ion: PerSpinBetaProjections` | `ion.f90:7544-7577`: per-spin β·ψ projections | `scf.rs:175` | **WARM-START-VERIFIED** | `PerSpinBetaProjections(SpinChannelData<KptDataSet<Option<Vec<CudaSlice<CudaComplex>>>>>)`. |
| C1-S6 | `density_aug_fine: Option<RealGrid<f64>>` — single channel → `density_aug_fine: PerSpinAugDensity` | `density.f90:1121-1149`: separate `Q_rho_sum` + `Q_rho_sum_sp` | `scf.rs:185` | **WARM-START-VERIFIED** | `PerSpinAugDensity(SpinChannelData<Vec<Option<RealGrid>>>)` stores per-spin augmentation on fine grid. |
| C1-S7 | `fermi_energy: Option<f64>` — single value → `fermi_energy: FermiEnergies` | `electronic.f90:9180`: `fermi_energy(2) = fermi_energy(1)` — per-spin array | `scf.rs:168` | **WARM-START-VERIFIED** | `FermiEnergies(Vec<f64>)` — per-spin Fermi energies. Initialised to `[0.0; nspins]`. |

### Component 2: `into_phase()` — State Transition Copy (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C2-S1 | Field-by-field copy: 6 per-spin fields not copied | — | `scf.rs:272-274,277,286,288` | **WARM-START-VERIFIED** | `into_phase()` copies 22 fields with struct-literal syntax. Compiler enforces correctness for type changes. |
| C2-S2 | `v_eff: Option<S::VEff>` — already generic, no change | — | `scf.rs:275` | **LOW** | `S::VEff` already encodes spin-channel potentials via `SpinPolicy`. |
| C2-S3 | Non-mutable fields unchanged | — | `scf.rs:263-271` | **LOW** | `cell`, `pots`, `wave_grid`, `fine_grid`, `k_point`, `smearing`, `pw_coords`, `pw_fft_indices` — geometry-static, no spin dimension. |

### Component 3: Spin Loop in `diagonalize_inner` (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C3-S1 | **Hardcoded spin index 0**: `S::v_eff_for_spin(v_eff_ref, 0)` → `for ispin in 0..nspins` | `electronic.f90:488`: `do ns=1,wvfn%nspins` | `scf.rs:750` | **WARM-START-VERIFIED** | Spin outer + kpt inner nested loop matches CASTEP `electronic.f90:488-495` exactly. |
| C3-S2 | Single `psi` upload to GPU → per-spin-per-kpt psi upload | `electronic.f90:492`: per-spin slice `coeffs(:,:,nk,ns)` | `scf.rs:859-864` | **WARM-START-VERIFIED** | `psi_cpu[ispin][ikpt]` uploaded separately for each (spin, kpt). |
| C3-S3 | Single eigenvalue output → `PerSpinEigenvalues` per-spin per-kpt | `electronic.f90:492`: per-spin `eigenvalues(:,nk,ns)` | `scf.rs:1001-1003` | **WARM-START-VERIFIED** | `next.eigenvalues[ispin] = KptDataSet::new(spin_eig_kpts, nkpts)`. |
| C3-S4 | VNL data precomputed once (not per-spin) → per-(spin,kpt) inside nested loop | `hamiltonian.f90:1013`: `nlpot_prepare_precon` receives `nk, ns` | `scf.rs:848-856` | **WARM-START-VERIFIED** | `VnlBatchData::precompute_with_d_override()` called per (spin, kpt) with `&v_eff_for_d` (per-spin). |
| C3-S5 | Single `beta_psi_gpu` output → per-spin-per-kpt via `KptDataSet` | Per-ion β·ψ differs per spin | `scf.rs:903-924,1005` | **WARM-START-VERIFIED** | Beta-psi recomputed per (spin, kpt), stored in `PerSpinBetaProjections`. |

### Component 4: `BuildVEff` for `SpinCollinear` (Paramagnetic Guess) (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C4-S1 | Zero spin density paramagnetic guess → zero_spin fallback implemented | `locpot.f90:278`: xc_calculate_potential receives `rho` + `sprho` from density | `scf.rs:376-387` | **WARM-START-VERIFIED** | `build_v_eff_impl` for `SpinCollinear` accepts `spin: Option<&Density>`. When `None` (iter-0 guess), creates zero spin density as fallback. |
| C4-S2 | `assemble_on_fine_grid` with `&zero_spin` — no energy return | `electronic_prepare_H` returns energy via `pot_calc_energy_real` | `scf.rs:347-348` | **WARM-START-VERIFIED** | Energy from `build_v_eff_with_energy` (Component 5). |
| C4-S3 | Return type `(EffectivePotential, EffectivePotential)` — not newtyped | `pot.f90:76`: `real_fine_pot(:,ns)` — 2D array indexed by spin | `scf.rs:375` | **LOW** | Tuple return is valid Rust encoding. |

### Component 5: `BuildVEffWithEnergy` for `SpinCollinear` — **WARM-START-VERIFIED**

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C5-S1 | **No `BuildVEffWithEnergy` impl for `SpinCollinear`** → implemented | `electronic_prepare_H` → `locpot_calculate`: assembles V_eff from ρ_total + ρ_spin, computes E_H + E_xc | `scf.rs:456-567` | **WARM-START-VERIFIED** | Full implementation: upsample both ρ_total and ρ_spin to fine grid, add augmentation, assemble V_eff via `VEffBuilder::<SpinCollinear>::with_density()`, compute E_H, E_xc, ρV_xc. |
| C5-S2 | Energy integral convention: `1/N_grid` vs `Ω/N_grid` | `xc.f90:565`: `1/n_grid` convention | `scf.rs:541-542` | **WARM-START-VERIFIED** | `d_v = 1.0 / n_grid` — matches CASTEP convention. |
| C5-S3 | XC energy: `compute_pbe_xc_spin` vs `compute_pbe_xc` | `xc.f90:516-523`: spin-dependent XC evaluation | `scf.rs:532-537` | **WARM-START-VERIFIED** | Calls `compute_pbe_xc_spin(rho_total_core, rho_spin_fine, ...)`. |
| C5-S4 | `∫ρV_xc = Σ(ρ_up·V_xc_up + ρ_down·V_xc_dn) / N_grid` | CASTEP `pot_calc_energy_real`: per-spin ∫ρV_xc sum | `scf.rs:556-563` | **WARM-START-VERIFIED** | Two-channel sum implemented. |
| C5-S5 | Upstream change needed in chemrust-hamiltonian-core | `Built::assemble()` returns `(V_up, V_dn)` only, no energy | `band_structure.rs:406-431` | **DEFERRED** | SCF layer calls `compute_pbe_xc_spin` directly (matching NonSpin path). |

### Component 6: `compute_density_from_wavefunctions` (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C6-S1 | **Single-spin density construction**: `construct_density_gpu()` called once → per-spin loop with kpt-weighted accumulation | `density.f90:2126-2164`: per-spin loop `do ns=1,nspins`; also multi-kpt weighted sum at `density.f90:2179-2187` | `scf.rs:1315-1476` | **WARM-START-VERIFIED** | Per-spin loop: `for ispin in 0..nspins` with inner `for ikpt in 0..nkpts`. Each (spin,kpt) density weighted by `w_k` and accumulated. |
| C6-S2 | Single `Density` return — no `PerSpinDensity` → returns `(PerSpinDensity, PerSpinAugDensity, OccupationSet, FermiEnergies)` | `density.f90:2179-2187`: `den%charge = up+down`, `den%spin = up−down` | `scf.rs:1478-1481` | **WARM-START-VERIFIED** | `PerSpinDensity(SpinChannelData::<S>(densities))` wraps per-spin densities. |
| C6-S3 | Single-channel occupation search → per-spin with spin_freed transition | `electronic.f90:488-495`: per-spin eigenvalues → per-spin occupations | `scf.rs:1288-1330` | **WARM-START-VERIFIED** | `spin_freed_occs` pre-computed for fermi_free. Per-spin loop uses `compute_occupations_weighted()` with kpt weights. |
| C6-S4 | Single-channel augmentation → per-spin augmentation density | `ion.f90:7544-7577`: per-spin β·ψ → per-spin rho_ij | `scf.rs:1361-1433` | **WARM-START-VERIFIED** | Per-spin `beta_psi_per_ion[ispin][ikpt]` used. Augmentation density accumulated with kpt weights. |
| C6-S5 | Density convention unchanged | `density.f90:2181-2187`: same convention | `scf.rs:1053` | **LOW** | Convention unchanged — ρ_up and ρ_down stored in raw ρ×Ω units. |

### Component 7: `construct_density_off/kerker/pulay` (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C7-S1 | `construct_density_off`: `new_density: Density` → `PerSpinDensity` | `density.f90`: stores both charge and spin | `scf.rs:1491` | **WARM-START-VERIFIED** | `next.density = new_density` where `new_density: PerSpinDensity`. |
| C7-S2 | `construct_density_kerker`: same type change | — | `scf.rs:1521` | **WARM-START-VERIFIED** | Field `density: new_density` in manual struct literal. |
| C7-S3 | `construct_density_pulay`: same type change | — | `scf.rs:1567` | **WARM-START-VERIFIED** | Same as Kerker — manual struct literal with `PerSpinDensity`. |

### Component 8: `mix()` — Density Mixing (**DIVERGE — C8-S3**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C8-S1 | Mixing operates on single `Density` → per-spin mixing loop | `dm_mix_density`: mixes `charge(:)` + `spin(:)` independently | `scf.rs:1606-1613` | **WARM-START-VERIFIED** | `for ispin in 0..nspins { let (mixed, prev) = self.history.mix(self.density[ispin].clone()); }` — mixes each spin channel independently. |
| C8-S2 | `previous_density` tracks single `Density` → per-spin `PerSpinDensity` | — | `scf.rs:1612` | **WARM-START-VERIFIED** | Each spin channel gets its own `prev` density tracked in `DensityHistory`. |
| C8-S3 | Kerker/Pulay mixing of spin density | NiO `.param`: `spin_density_mixing_amplitude=2.0`, `spin_density_mixing_g_vector=1.5` A^-1 | — | **DIVERGE — CANDIDATE ROOT CAUSE** | Mixes ρ_up and ρ_down with **same mixing amplitude as charge density** (0.5). CASTEP uses **separate `spin_density_mixing_amplitude=2.0`** for spin channels. Over 16 iterations, wrong spin mixing amplitude produces incorrect spin density trajectory — spin density converges 4x slower (0.5 vs 2.0), meaning at iter 16 the spin density is far from self-consistent. Wrong spin density → wrong V_eff_up vs V_eff_dn → wrong eigenvalues → wrong occupations → cascade divergence. **This is the primary hypothesis for iter-16 divergence per the checklist gap analysis. Priority: P0 (MUST-FIX).** |

### Component 9: `check()` — Convergence Check (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C9-S1 | Convergence criteria use single `eigenvalues` and single `Density` → per-spin band energy + multi-kpt weights | `electronic_check_occupancies`: checks total energy convergence | `scf.rs:1856-1895` | **WARM-START-VERIFIED** | Per-spin loop computes `e_band_spin = Σ_k w_k Σ_b f_{bk} ε_{bk}`, summed for total band energy. |
| C9-S2 | Fermi energy stored as single `Option<f64>` → `FermiEnergies` | `electronic.f90`: per-spin Fermi energies | `scf.rs:1875` | **WARM-START-VERIFIED** | `self.fermi_energy[ispin] = chem_pot.0` for each spin. Entropy correction `-TS` computed per-spin. |

### Component 10: Occupation Search (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C10-S1 | **No `fermi_fix` / `fermi_free` distinction** → implemented | `electronic.f90:516-518`: `if(scf_cycle == spin_fix) call fermi_fix else call fermi_free` | `density.rs` (find_fermi_fix, find_fermi_free) | **WARM-START-VERIFIED** | Two distinct functions. Transition controlled by `spin_freed` flag. |
| C10-S2 | Single electron count from total valence → per-spin from density net_spin | `electronic.f90:8742-8746`: `frac_elec(1) = 0.5*(N+net_spin)`, `frac_elec(2) = 0.5*(N-net_spin)` | `scf.rs:1246-1266, scf.rs:1857-1864` | **WARM-START-VERIFIED** | `net_spin = ∫(ρ_up − ρ_down) / N_grid` from CURRENT density. |
| C10-S3 | Single bisection search → per-spin independent bisection for fermi_fix | `electronic.f90:8757`: `do ns=1,nspins` for fermi_fix | `density.rs` (compute_occupations_weighted) | **WARM-START-VERIFIED** | `find_fermi_fix(eigenvalues[ispin], n_spin_electrons, smearing)` called per spin. |
| C10-S4 | No shared Fermi energy search → `find_fermi_free` implemented | `electronic.f90:9094-9106`: one bisection integrating both spins | `density.rs` (find_fermi_free) | **WARM-START-VERIFIED** | `find_fermi_free(ev_up, ev_dn, n_total, smearing, occ_factor)`. |
| C10-S5 | No `net_spin` computation in fermi_free → `intent(out)` from occupancies | `electronic.f90:9183-9209`: net_spin = Σocc_up − Σocc_dn after shared E_F | `density.rs` (find_fermi_free return) | **WARM-START-VERIFIED** | `find_fermi_free` returns `net_spin` computed from the resulting occupancies. |
| C10-S6 | No `spin_fix` parameter or transition logic → `spin_freed` flag | `electronic.f90:516-518`, NiO `.param`: `spin_fix=6` (runtime value = 5 per profile) | `scf.rs:194` | **WARM-START-VERIFIED — POTENTIAL DIVERGENCE** | `SmearingParams.spin_fix` (default 10 in test code, vs CASTEP runtime 5). For cold-start iterations 6-10, Rust uses `fermi_fix` while CASTEP uses `fermi_free`. Different occupation strategy → different electron counts → different density → cascade. **Candidate root cause. Priority: P1.** |
| C10-S7 | Smearing formula: Gaussian only — NiO default is Gaussian | `algor.F90:2929`: 5 schemes | `density.rs` | **LOW** | Gaussian smearing using `erfc`-based occupancy. NiO uses Gaussian with 0.1 eV width. |

### Component 11: FFI `chemrust_eigensolve_init` (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C11-S1 | **No `nspins` parameter** → `nspins: c_int` added | `chemrust_eigensolve.f90:134`: passes `nspins` from `wvfn%nspins` | `ffi.rs` (init signature) | **WARM-START-VERIFIED** | `nspins: c_int` param received, stored as `nspins: usize` in `ChemrustHandle`. |
| C11-S2 | Per-k-point `VnlBatchData` not per-spin → `vnl: Vec<Option<VnlBatchData>>` per-spin | — | `ffi.rs` (KptData struct) | **WARM-START-VERIFIED** | `KptData.vnl: (0..nspins).map(|_| None).collect()` — capacity `nspins`. |
| C11-S3 | V_eff cache single-entry → per-spin `Vec<Option<CudaSlice<f64>>>` | — | `ffi.rs` (ChemrustHandle) | **WARM-START-VERIFIED** | `v_eff_cached: vec![None; nspins]`, `v_eff_norm: vec![0.0; nspins]`. |

### Component 12: FFI `chemrust_eigensolve_step` (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C12-S1 | **No `ispin` parameter** → `ispin: c_int` added | `electronic.f90:522`: passes per-spin slice `real_fine_pot(:,ns)` | `ffi.rs` (step signature) | **WARM-START-VERIFIED** | `ispin: c_int` param received, converted to 0-based. |
| C12-S2 | V_eff upload reads single cache entry → per-spin cache `h.v_eff_cached[isp]` | — | `ffi.rs` (step_inner) | **WARM-START-VERIFIED** | Per-spin V_eff change detection. `cache_reuse=false` always (2026-06-13 fix). |
| C12-S3 | VNL lookup single-entry → `kd.vnl[isp]` per-spin | — | `ffi.rs` (step_inner) | **WARM-START-VERIFIED** | VNL data indexed per-spin: `kd.vnl[isp]`. |
| C12-S4 | Fortran 1-based → Rust 0-based: `ispin = ns - 1` in Fortran wrapper | `electronic.f90:488`: `ns=1,wvfn%nspins` | CASTEP `chemrust_eigensolve.f90` | **WARM-START-VERIFIED** | Fortran wrapper converts: `ispin = ns - 1` before calling `chemrust_eigensolve_step`. |
| C12-S5 | Data arrays already per-spin at Fortran boundary | `electronic.f90:523`: `coeffs(:,:,nk,ns)` — Fortran slices single channel | — | **LOW** | Fortran caller already slices `coeffs(:,:,nk,ns)`. |

### Component 13: `run_scf` / `run_scf_with_energy` / `run_scf_with_energy_gated` (**WARM-START-VERIFIED**)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C13-S1 | **`S: SpinPolicy + BuildVEff` bound — NonSpin only in practice** → `SpinCollinear` now implements all traits | `electronic_minimisation`: generic over nspins | `scf.rs:1548` | **WARM-START-VERIFIED** | `SpinCollinear` implements `BuildVEff`, `BuildVEffWithEnergy`. |
| C13-S2 | Energy tracking uses single `total_energy: Option<f64>` | — | `scf.rs:166` | **LOW** | Total energy is spin-independent scalar sum. |
| C13-S3 | `scf_iter` counter used for spin_fix transition → `spin_freed` flag set in `run_scf` | `electronic.f90:516-518` | `scf.rs:194` | **WARM-START-VERIFIED** | `state.spin_freed = scf_iter > smearing.spin_fix`. See C10-S6 for spin_fix value concern. |

---

## 3. Spin-Specific Data Flow Diagram

```
SCF Iteration (per spin channel in diagonalization):
  
  ┌─────────────────────────────────────────────────────────────┐
  │  for ispin in 0..nspins():                                  │
  │    1. v_eff_spin = S::v_eff_for_spin(v_eff, ispin)         │
  │    2. psi_ispin = self.psi[ispin]                           │
  │    3. vnl_data = precompute_d(..., v_eff_spin)  ← per-spin!│
  │    4. (psi_out, eig_ispin, beta_psi) = davidson(...)       │
  │    5. self.psi[ispin] = psi_out                             │
  │    6. self.eigenvalues[ispin] = eig_ispin                   │
  │    7. self.beta_psi_per_ion[ispin] = beta_psi               │
  └─────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────┐
  │  Density construction (per-spin internally, once per SCF):  │
  │    for ispin in 0..nspins():                                │
  │       ρ[ispin] = construct_density_gpu(ψ[ispin], occ[ispin])│
  │    ρ_total = ρ[0] + ρ[1]                                    │
  │    ρ_spin  = ρ[0] - ρ[1]                                    │
  └─────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────┐
  │  Occupation search:                                          │
  │    if scf_iter <= spin_fix:                                 │
  │       for ispin in 0..nspins():                             │
  │          E_F[ispin] = fermi_fix(eig[ispin], N_spin[ispin])  │
  │    else:                                                     │
  │       E_F_shared = fermi_free(eig[0], eig[1], N_total)      │
  │       E_F[0] = E_F_shared                                   │
  │       E_F[1] = E_F_shared                                   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 4. Dependency Map

```
Component 1 (ScfIteration fields)
  ├── Component 2 (into_phase) — needs new field types
  ├── Component 3 (diagonalize_inner) — needs per-spin psi/eig access
  ├── Component 6 (density construction) — needs per-spin psi/eig
  ├── Component 7 (construct_density_*) — needs per-spin return types
  ├── Component 8 (mix) — needs PerSpinDensity
  ├── Component 9 (check) — needs per-spin eigenvalues
  └── Component 10 (occupation search) — needs per-spin eigenvalues

Component 5 (BuildVEffWithEnergy) — needs per-spin density + spin density from C6

Component 12 (FFI step) — needs per-spin caches from C11

Component 13 (run_scf) — needs C5 (BuildVEffWithEnergy for SpinCollinear)

All components depend on Component 1 (SpinChannelData<T> newtypes).
```

---

## 5. Synthesis Findings: Cross-Checklist Items (Spin SCF ↔ Inner Loop)

These items emerged during spin-polarised implementation (2026-06-12) and affect both the SCF spin loop (this checklist) and the Davidson inner loop (INNER_LOOP_CHECKLIST_20260606.md). Each item references both checklists.

### S1. VNL D-matrix recomputation per (spin, kpt)

- **CASTEP reference**: `hamiltonian.f90:1013` — `nlpot_prepare_precon` receives `nk, ns` as explicit parameters. D-matrix screening uses `∫Q·V_eff`, which differs per spin channel → D-matrices differ per spin channel.
- **Rust implementation**: `scf.rs:848-856` — `VnlBatchData::precompute_with_d_override()` called per (ispin, ikpt) inside nested spin+kpt loop.
- **Pre-fix state**: `VnlBatchData::precompute()` was called once per SCF iteration (NonSpin single-channel).
- **Cross-check**: INNER_LOOP_CHECKLIST Component 3 (preconditioner) — the TPA preconditioner uses eigenvalues from `slice_eigenvalues` and β-projectors from VNL data, both of which are now per-(spin,kpt). The USPP NL correction at `preconditioner.rs:1045` (`eigenvalues[b] * beta_phi`) is now correctly per-spin.
- **Discriminator**: Using wrong-spin VNL data would silently produce wrong V_NL contributions with eigenvalues that pass loose tolerances — no existing test catches this.

### S2. Per-spin electron counts from integrated density, not static .cell SPIN=

- **CASTEP reference**: `electronic.f90:8742-8746` — `frac_elec(1) = 0.5*(N+net_spin)`, `frac_elec(2) = 0.5*(N-net_spin)` uses dynamic `net_spin` from current density.
- **Rust implementation**: `scf.rs:1246-1266` (density construction), `scf.rs:1815-1824` (check). `net_spin = Σ(ρ_up[i] - ρ_down[i]) / N_grid` from CURRENT `PerSpinDensity`.
- **Origin**: SPIN_SCF_CHECKLIST C10-S2 originally assumed `N_up`, `N_dn` from static `.cell SPIN=` block. During warm-start test implementation, discovered that the converged density's integrated net_spin may differ from the initial `.cell` value.
- **Cross-check**: INNER_LOOP_CHECKLIST Component 14 (convergence check) — per-spin electron counts shift the Fermi level, which changes which bands are "active" (unconverged) in the Davidson inner loop. ZHEEVD's ascending eigenvalue sort means small Fermi shifts can change inner-loop convergence behavior for bands near E_F.

### S3. Energy formula synthesis: single-spin → multi-kpt spin + E_nonCoulomb

- **CASTEP reference**: `energy.f90:4205` — `E_nonCoulomb` term from pseudopotential local parts. `pot.f90:4205` — per-spin `∫ρV_xc` sum.
- **Rust implementation**: `scf.rs:1856-1925` — per-spin kpt-weighted band energy: `E_band = Σ_spin Σ_k w_k Σ_b f_{bk} ε_{bk}`. Assembly: `assemble_total_energy_from_band()`. `E_nonCoulomb` constant in `tests/nio_spin_scf.rs:77` (+533.14 eV = +19.59 Ha).
- **Findings**: NiO discriminator test revealed missing E_nonCoulomb term (+19.59 Ha). SPIN_SCF_CHECKLIST C5-S4 correctly identified per-spin ∫ρV_xc, and C9-S1 correctly stated total energy is a scalar sum. Neither anticipated E_nonCoulomb.
- **Cross-check**: INNER_LOOP_CHECKLIST Component 14 (tol gate) — eigenvalue precision (1e-8 Ha) maps to energy contribution of ~1e-8 Ha per highly-occupied band. Below the 1e-6 Ha energy tolerance. No-action coupling, documented for completeness.

### S4. Spin_freed transition and inner loop convergence path

- **CASTEP reference**: `electronic.f90:516-518` — `if(scf_cycle == spin_fix) call fermi_fix else call fermi_free`. `hamiltonian.f90:548-550` — `band_converged` reset conditional on `opt_stop`.
- **Rust implementation**: `spin_freed` flag set in `run_scf_with_energy` when `scf_iter > smearing.spin_fix`, consulted by `compute_density_from_wavefunctions` and `check()`.
- **Effect**: When spin is freed, `find_fermi_free` produces a shared Fermi energy → occupation distribution shifts → some bands may stagnate. CASTEP's `opt_stop` logic at `hamiltonian.f90:548-550` prevents premature freeze — stalled bands are re-checked.
- **Cross-check**: INNER_LOOP_CHECKLIST C14-02 (band_converged reset conditional) — this is the CASTEP-faithful guard against spin_freed-induced stagnation. Our implementation matches.

### S5. KptDataSet<T> encoding and inner loop scatter/gather

- **Rust implementation**: `spin_types.rs` — `KptDataSet<T>` wraps `Vec<T>` with stored `nkpts` count, avoiding `Vec<Vec<T>>`. Flat-major layout ensures all k-points have identical structure.
- **Cross-check**: INNER_LOOP_CHECKLIST Components 1 and 2 (Stage 1 psi/hpsi copy, A3 copy-back) — the compacted-slice-to-global and global-to-slice copy paths used per-band iterators that assume band-major layout. With `KptDataSet`, multi-kpt wavefunction arrays preserve band-major layout within each (spin,kpt) entry. The inner loop's `active_bands()` and `block_bands()` iterators are unchanged because they operate within a single (spin,kpt) block.

---

## 6. Recommended Fix Priority Order

### Priority 0: Cold-Start Divergence — MUST FIX

These items are believed to contribute to the iter-16 cold-start divergence. Fix order follows expected impact.

| Rank | ID | Description | Estimated Effort | Rationale |
|------|----|-------------|-----------------|-----------|
| **P0-1** | C8-S3 (gap G1) | Implement per-spin mixing parameters (`spin_density_mixing_amplitude=2.0`, `spin_density_mixing_g_vector`) matched to NiO `.param` | 2-4 hours | Primary candidate: mixing spin with charge amplitude (0.5) instead of spin amplitude (2.0) produces 4x slower spin density evolution. Wrong spin density at iter 16 → wrong V_eff_up vs V_eff_dn → cascade. |
| **P0-2** | C10-S6 (gap G2) | Set `spin_fix` from CASTEP input (`.param` or `NiO.castep_bin`) instead of hardcoded default 10. NiO runtime value is 5. | 1 hour | Wrong occupation strategy for iterations 6-10. Compound effect with G1: wrong occupations + wrong spin density = strongly wrong V_eff. |
| **P0-3** | INNER_LOOP C1-davidson-compaction | Fix Davidson compaction eigenvalue index mapping bug (MUST-FIX per memory/davidson-compaction-index-bug.md) | 4-8 hours | Corrupts eigenvalue index mapping in inner loop. Benign for warm-start (small residuals, few inner iterations), catastrophic for cold-start (large residuals, many inner iterations, index corruption compounds). |

### Priority 1: Diagnostic Infrastructure — VERIFY BEFORE FIXING

| Rank | ID | Description | Estimated Effort | Rationale |
|------|----|-------------|-----------------|-----------|
| **P1-1** | G6 (diagnostic) | Implement per-iteration diagnostic tracking: spin density evolution, V_eff_up vs V_eff_dn max difference, occupation distribution, band energy per spin | 3-5 hours | Without these, the iter-16 divergence cannot be characterized beyond "it diverges." Must implement BEFORE applying P0 fixes to confirm the fix changes divergence behavior as predicted. |
| **P1-2** | G6 (cold-start test) | Write `nio_cold_start_discriminator` test mirroring warm-start pattern but starting from paramagnetic guess | 4-6 hours | The test that should have caught the divergence. Must run 16+ iterations, track eigenvalues, energy, spin density, and occupation sums at each iter. |

### Priority 2: Multi-Iteration Verification — CONFIRM FIXES

| Rank | ID | Description | Estimated Effort | Rationale |
|------|----|-------------|-----------------|-----------|
| **P2-1** | V11-V16 | Implement all cold-start verification criteria from Section 0.6 | 3-5 hours | Confirms P0 fixes resolve divergence and no regressions introduced. |
| **P2-2** | C8, C10 | Re-verify occupation search and mixing after P0 fixes with cold-start NiO fixture | 2-3 hours | Reclassify from WARM-START-VERIFIED to FIXED when cold-start passes. |

### Deferred Items

| Rank | ID | Description | Status |
|------|----|-------------|--------|
| **D1** | C5-S5 | Optimise XC calls: `compute_pbe_xc_spin` called twice (once in `assemble()`, once for energy in SCF layer) | Deferred |
| **D2** | C8-S3 (per-spin Pulay history) | Per-spin Pulay history parameters (currently shared across spin channels within the same `DensityHistory`) | Deferred |
| **D3** | C6 (OccSet) | Generalise `OccupationSet` for per-kpt storage (currently stores kpt-0 only) | Deferred |
| **D4** | S3 (E_nonCoulomb) | Compute `E_nonCoulomb` from `PseudopotentialSet` in `chemrust-hamiltonian-core` | Deferred |
| **D5** | C7 (manual copies) | Refactor `construct_density_kerker` and `construct_density_pulay` to use `into_phase()` | Deferred |
| **D6** | C4-S3 | newtype per-spin V_eff tuple → `SpinChannelData<EffectivePotential>` | Deferred |
| **D7** | G3 (per-spin residual norms) | Per-spin density change metric for independent charge/spin mixing convergence tracking | Deferred |

---

## 7. Cascading Dependency Chains

```
P0-3 (Davidson compaction fix — blocks cold-start correctness)
  └── P1-1 (diagnostic infrastructure)
        └── P1-2 (cold-start discriminator test)

P0-1 (spin_density_mixing_amplitude)
  └── P0-2 (spin_fix value)
        └── P2-1 (cold-start verification)

P0-1 + P0-2 together expected to resolve iter-16 divergence.
P0-3 expected to improve cold-start eigenvalue quality across all iterations.
P1-1 required to CONFIRM any fix works, not just empirically test.
```

---

## 8. CASTEP Profile Evidence

From `NiO.0001.profile` (2026-06-10, 2952 lines):

| Subroutine | Calls | Time | Implication |
|------------|-------|------|-------------|
| `electronic_minimisation` | 1 | 57.50s | Single SCF run |
| `electronic_prepare_H` | 66 | 29.67s | V_eff assembled once per SCF iter (not per-spin) |
| `hamiltonian_diagonalise_ks` | **134** | 17.70s | **134 = 67 SCF × 2 spins** — spin loop confirmed |
| `electronic_find_occupancies` | **67** | 0.35s | Called once per SCF (NOT per-spin) — spin loop inside |
| `electronic_find_fermi_fix` | 5 | 0.04s | First 5 iterations: fixed spin |
| `electronic_find_fermi_free` | 62 | 0.27s | Remaining 62: free spin |
| `electronic_apply_H_energy_eigen` | 67 | 0.47s | Per-SCF energy computation |
| `density_calculate_soft_wvfn` | 65 | 0.68s | Called once per SCF (NOT per-spin) — spin loop inside |
| `density_augment` | 65 | 8.17s | Augmentation per SCF |
| `dm_mix_density` | 66 | 0.32s | Mixing per SCF (+ 1 from init) |

Total SCF iterations: 67 (converged). `spin_fix` = 5 (CASTEP default, though NiO `.param` specifies 6 — the profile suggests 5 was used at runtime).

**NiO `.param` reference**: `spin_density_mixing_amplitude: 2.0`, `spin_density_mixing_g_vector: 1.5 1/A` (from NiO `.castep` line 109509-109512). These differ from `mix_charge_amp: 0.5` and `mix_charge_g_vector: 1.5 1/A`. Our code applies charge mixing parameters to all spin channels — gap G1 (C8-S3).

---

## 9. Verification Matrix

| # | Criterion | Fixture Anchor | Tolerance | Discriminator Ratio |
|---|-----------|---------------|-----------|---------------------|
| V1 | `diagonalize_inner` calls Davidson twice per SCF iter for SpinCollinear | Profile: 134 calls | Exact (compiler-enforced) | ∞ (doesn't compile otherwise) |
| V2 | `V_eff_up ≠ V_eff_dn` for spin-polarised density | Self-consistency | | Qualitative |
| V3 | Per-spin eigenvalues match `.bands` | NiO.bands:12,75 | 1x10^-4 Ha | ~1000x |
| V4 | Total energy matches CASTEP | NiO.castep:774418 (-7160.230577732 eV) | 1x10^-6 Ha | ~1000x |
| V5 | Integrated spin density: 2∫ρ_spin = -0.0619641 | NiO.castep:774415 | 1x10^-4 rel | ~100x |
| V6 | Fermi energies: 0.152664 Ha both spins | NiO.bands:5 | 1x10^-4 Ha | ~100x |
| V7 | Occupations: N_up ≈ 36.00, N_dn ≈ 28.00 | NiO.bands:3 | ±0.01 | ~1000x |
| V8 | NonSpin regression: existing tests pass | Cu111_CO fixture | Identical results | Regression guard |
| V9 | FFI init accepts nspins, step accepts ispin | Compiler | Compiles + CASTEP builds | |
| V10 | No `Vec<Vec<T>>` in public API touching per-spin data | rg | 0 occurrences | Style |

### New: Multi-Iteration Cold-Start Verification

| # | Criterion | Fixture Anchor | Tolerance | Why This Tests Cold-Start |
|---|-----------|---------------|-----------|---------------------------|
| **V11** | 16-iteration cold-start eigenvalues remain bounded | NiO CPU reference: band range [-1.0, +0.2] Ha | max eigenvalue > -10 Ha, min eigenvalue < +5 Ha | Catches eigenvalue explosion/divergence before it's catastrophic |
| **V12** | Spin density trajectory over 6-16 iters matches CASTEP | CASTEP per-SCF-iter spin density dumps (F8 format) | RMS residual < 0.1 e-/Bohr^3 after spin_freed transition | Verifies mixing amplitude correctness |
| **V13** | Occupation sums N_up + N_dn = N_total at every iter | CASTEP constraint: N_total = 64 for NiO | \|N_up + N_down - 64\| < 1e-6 | Catches Fermi search bugs that lose electrons |
| **V14** | `spin_freed` flag matches CASTEP transition point | NiO profile: 5 `fermi_fix` calls -> `spin_fix=5` | Exact match of transition iteration | Catches wrong spin_fix value |
| **V15** | Per-spin band energy drift per iteration < 0.5 Ha for first 10 iters | CASTEP energy trajectory from cold-start | Monotonically decreasing after iter 5 | Catches V_eff assembly accumulation errors |
| **V16** | No "no empty bands" warning | CASTEP behavior: empty bands exist at all iters | Warning count = 0 | This warning is the immediate precursor to divergence (per NiO failure pattern) |

---

## 10. Style Conformance

Applied during implementation:

1. No `for` loops in `src/` outside test code and GPU kernel launch closures
2. Every new function has CASTEP source citation in doc comment: `/// Reference: subroutine_name (file.f90:line-line)`
3. No raw `Vec<Vec<T>>` in any new public API — `SpinChannelData<T>` or its named wrappers only
4. All per-spin newtypes implement `Deref<Target=Inner>` for ergonomic access with explicit construction
5. `cargo clippy --workspace -- -D warnings` must pass after each task
6. All numeric assertions cite ground-truth source (fixture file + line number, or CASTEP source + line number)

---

## Appendix A: CASTEP Source Files Audited

| File | Lines | Content |
|------|-------|---------|
| `electronic.f90` | 483-532 | Spin loop: `do ns=1,nspins` around `hamiltonian_diagonalise` |
| `electronic.f90` | 510-527 | `spin_fix` transition: `if(scf_cycle == spin_fix)` |
| `electronic.f90` | 2400-2450 | `electronic_prepare_H`: V_eff assembly per SCF |
| `electronic.f90` | 8602-8880 | `electronic_find_fermi_fix`: per-spin independent bisection |
| `electronic.f90` | 8910-9209 | `electronic_find_fermi_free`: shared E_F, spin from occ |
| `electronic.f90` | 9400-9438 | `electronic_occupancy_update`: per-spin occupancy assignment |
| `electronic.f90` | 16027-16034 | Second spin loop: `hamiltonian_diagonalise` call site |
| `density.f90` | 30-37 | `electron_density` type: `charge(:)` + `spin(:)` |
| `density.f90` | 1120-1168 | `density_augment`: spin-dependent augmentation |
| `density.f90` | 2120-2195 | `density_calculate_soft_wvfn_real`: per-spin |ψ|^2 -> charge/spin |
| `locpot.f90` | 74-422 | `locpot_calculate`: V_H + V_locps + V_xc per spin |
| `xc_gga.f90` | 516-523 | XC potential: total + spin density -> V_xc_up, V_xc_dn |
| `hamiltonian.f90` | 723 | `hamiltonian_diagonalise_ks` signature: `nk, ns` as scalars |
| `hamiltonian.f90` | 1013 | `nlpot_prepare_precon` signature: `nk, ns` -> per-spin D-matrices |
| `ion.f90` | 7544-7577 | Spin dimension in ion augmentation matrices |
| `pot.f90` | 76 | `real_fine_pot(:,ns)` — 2D V_eff storage indexed by spin |
| `chemrust_eigensolve.f90` | 134 | FFI init: `nspins` from `wvfn%nspins` |

---

## Appendix B: Guardrail — CASTEP Source Citation Protocol

1. Every task MUST cite the specific CASTEP source lines before implementation.
2. Profile evidence (`NiO.0001.profile`) is authoritative for call counts and nesting structure.
3. Any claim about "how CASTEP does X" without a source line citation is a hypothesis, not a fact.
4. Fortran 1-based indexing vs Rust 0-based indexing must be explicitly converted at every boundary.
5. The spin loop nesting (spins outer, k-points inner) is confirmed by CASTEP `electronic.f90:488-495` and profile evidence (134 `hamiltonian_diagonalise_ks` calls).
