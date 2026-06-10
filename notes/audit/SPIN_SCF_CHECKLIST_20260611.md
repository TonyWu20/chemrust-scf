# SCF Spin-Polarised Adversarial Audit Checklist

**Date**: 2026-06-11
**Scope**: SCF loop spin-polarisation (13 components), Rust (`scf.rs`, `ffi.rs`, `density.rs`) vs CASTEP 6.11 (`electronic.f90`, `density.f90`, `locpot.f90`, `xc_gga.f90`, `hamiltonian.f90`)
**Methodology**: Adversarial line-by-line audit; CASTEP profile evidence (NiO `NiO.0001.profile`: 67 SCF × 2 spins = 134 `hamiltonian_diagonalise_ks` calls); CASTEP source read for each component (2026-06-11 workflow).

**Pre-populated from adversarial audit (2026-06-11 workflow: `wf_10b795ec-dbd`):**
- CLAIM-1 (density storage): PARTIALLY CORRECT — CASTEP stores `den%charge(:)` + `den%spin(:)` as separate 1D arrays, not `density(:,:,ns)`
- CLAIM-6 (occupation search): PARTIALLY CORRECT — `fermi_free` uses ONE shared Fermi energy, NOT per-spin independent
- CLAIM-9 (USPP augmentation): PARTIALLY CORRECT — `Q_nm` is spin-independent but `rho_ij` carries spin dimension → augmentation IS spin-dependent

---

## Fix History

| Date | Fix | IDs | Description |
|------|-----|-----|-------------|
| — | *(to be populated during implementation)* | — | — |

---

## 1. Component Summary Table

| # | Component | CASTEP Source | Rust Source | Status | Severity Spread |
|---|-----------|---------------|-------------|--------|-----------------|
| 1 | `ScfIteration` struct — per-spin fields | `electronic.f90:483-532` (wvfn, eigenvalues, density types) | `scf.rs:106-180` | **DIVERGE** — single-spin only | 1× CRITICAL, 6× HIGH |
| 2 | `into_phase()` — state transition copy | — (architectural) | `scf.rs:262-294` | **DIVERGE** — single-spin copy only | 6× HIGH |
| 3 | Spin loop in `diagonalize_inner` | `electronic.f90:488-495` | `scf.rs:512-711` | **DIVERGE** — hardcoded `v_eff_for_spin(v_eff_ref, 0)` | 1× CRITICAL, 3× HIGH |
| 4 | `BuildVEff` for `SpinCollinear` (paramagnetic) | `locpot.f90:301` (V_eff per-spin assembly) | `scf.rs:335-350` | **DIVERGE** — zero spin density guess | 2× HIGH |
| 5 | `BuildVEffWithEnergy` for `SpinCollinear` | `electronic_prepare_H` → `locpot_calculate` | `scf.rs:355-362` | **MISSING** — trait only, no impl | 1× CRITICAL, 3× HIGH |
| 6 | `compute_density_from_wavefunctions` | `density.f90:2126-2195` (per-spin loop → charge+spin) | `scf.rs:1024-1173` | **DIVERGE** — single spin, no ρ_spin | 1× CRITICAL, 3× HIGH |
| 7 | `construct_density_off/kerker/pulay` | density mixing pipeline | `scf.rs:1177-1360` | **DIVERGE** — single `Density`, no `PerSpinDensity` | 3× MEDIUM |
| 8 | `mix()` — density mixing | `dm_mix_density` | `scf.rs:1282-1360` | **DIVERGE** — single `Density`, no spin density mixing | 2× MEDIUM |
| 9 | `check()` — convergence | `electronic_check_occupancies` | `scf.rs:1421-1528` | **DIVERGE** — single `Density`, single `eigenvalues` | 2× MEDIUM |
| 10 | Occupation search (`compute_occupations`) | `electronic.f90:8602-9209` (fermi_fix + fermi_free) | `density.rs` | **DIVERGE** — no spin awareness, no fermi_fix/free distinction | 1× CRITICAL, 4× HIGH |
| 11 | FFI `chemrust_eigensolve_init` | `chemrust_eigensolve.f90` | `ffi.rs` | **DIVERGE** — no `nspins` param | 1× HIGH, 2× MEDIUM |
| 12 | FFI `chemrust_eigensolve_step` | `electronic.f90:511-529` | `ffi.rs` | **DIVERGE** — no `ispin` param | 1× HIGH, 2× MEDIUM |
| 13 | `run_scf` / `run_scf_with_energy` | `electronic_minimisation` | `scf.rs:1548-1656` | **DIVERGE** — `NonSpin`-only bounds | 1× HIGH, 1× MEDIUM |

### Status Counts

| Status | Count |
|--------|-------|
| **MATCH** (no significant differences) | 0 |
| **DIVERGE** (differences requiring implementation) | 12 |
| **MISSING** (entire component absent) | 1 |
| **FIXED** | 0 |

### Severity Counts

| Severity | Count |
|----------|-------|
| **CRITICAL** (wrong physics if not addressed) | 5 (C1-S1, C3-S1, C5-S1, C6-S1, C10-S1) |
| **HIGH** (functional gap, blocks spin support) | 32 |
| **MEDIUM** (observable effect, edge cases) | 13 |
| **LOW** (diagnostic, cosmetic, or verified benign) | 0 |

---

## 2. Surviving Differences — Detailed Analysis

### Component 1: `ScfIteration` Struct — Per-Spin Fields

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C1-S1 | `density: Density` — single-spin, no `PerSpinDensity` | `density.f90:30-37`: `electron_density` stores `charge(:)` + `spin(:)` separately | `scf.rs:122` | **CRITICAL** | SCF loop has no concept of spin channels. `Density` stores only total ρ. Spin density ρ_spin = ρ_up − ρ_down has no storage location. For SpinCollinear, V_eff assembly needs both ρ_total and ρ_spin. |
| C1-S2 | `psi: WavefunctionSet<ColumnDistributed>` — single channel | `electronic.f90:492`: `wvfn%coeffs(:,:,nk,ns)` — spin is outermost dimension | `scf.rs:123` | **HIGH** | Per-spin wavefunctions are not stored. `diagonalize_inner` must produce separate ψ_up and ψ_dn. |
| C1-S3 | `eigenvalues: Vec<f64>` — single channel | `electronic.f90:492`: `eigenvalues(:,nk,ns)` — spin is outermost dimension | `scf.rs:124` | **HIGH** | Per-spin eigenvalues not stored. Occupation search needs separate eigenvalue arrays. Fermi energy differs per spin in fermi_fix. |
| C1-S4 | `previous_density: Density` — single-track | — | `scf.rs:129` | **HIGH** | Density mixing history needs per-spin tracking. `ρ_up` and `ρ_down` evolve independently through the SCF cycle. |
| C1-S5 | `beta_psi_per_ion: Option<Vec<CudaSlice<CudaComplex>>>` — single channel | `ion.f90:7544-7577`: per-spin β·ψ projections | `scf.rs:160` | **HIGH** | USPP β-projections differ per spin channel (different wavefunctions). Augmentation density construction needs per-spin β·ψ. |
| C1-S6 | `density_aug_fine: Option<RealGrid<f64>>` — single channel | `density.f90:1121-1149`: separate `Q_rho_sum` + `Q_rho_sum_sp` | `scf.rs:170` | **HIGH** | USPP augmentation has per-spin contributions (density matrix `rho_ij` differs per spin). Deferred for Phase 7 per scope boundaries but field must be per-spin for future use. |
| C1-S7 | `fermi_energy: Option<f64>` — single value | `electronic.f90:9180`: `fermi_energy(2) = fermi_energy(1)` — per-spin array | `scf.rs:153` | **HIGH** | Fermi energy needs per-spin storage: independent values for fermi_fix, equal values for fermi_free. |

**Root cause analysis**: The `ScfIteration` struct was designed for `NonSpin` (nspins=1) and never generalised. All 7 mutable state fields that have a spin dimension must be wrapped in `SpinChannelData<T>` or replaced with per-spin newtypes. The CASTEP evidence is unambiguous: every field maps to a CASTEP type that carries a spin dimension (`(:,:,nk,ns)`, `(:,nk,ns)`, or `charge(:)` + `spin(:)`).

**Fix**: Apply TASK-3 from TASKS.md — replace all single-track fields with `PerSpinDensity`, `PerSpinPwCoefficients`, `PerSpinEigenvalues`, `PerSpinBetaProjections`, `PerSpinAugDensity`, and `FermiEnergies`.

---

### Component 2: `into_phase()` — State Transition Copy

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C2-S1 | Field-by-field copy: 6 per-spin fields not copied | — | `scf.rs:272-274,277,286,288` | **HIGH** | `into_phase()` copies 22 fields with struct-literal syntax. After C1-S1 through C1-S6, the field types change (e.g. `Density` → `PerSpinDensity`), but the struct-literal copy syntax is the same — the compiler enforces correctness for type changes. No explicit copy logic needed. |
| C2-S2 | `v_eff: Option<S::VEff>` — already generic, no change | — | `scf.rs:275` | **LOW** | `S::VEff` already encodes spin-channel potentials via `SpinPolicy`. For `SpinCollinear`, `VEff = (EffectivePotential, EffectivePotential)`. No change needed. |
| C2-S3 | Non-mutable fields unchanged | — | `scf.rs:263-271` | **LOW** | `cell`, `pots`, `wave_grid`, `fine_grid`, `k_point`, `smearing`, `pw_coords`, `pw_fft_indices` — geometry-static, no spin dimension. |

**Root cause analysis**: `into_phase()` is structurally correct — it just needs the new field types from C1. The compiler will catch any missed fields. The only risk is that `PhantomData<State>` erases the concrete type, so a field with the wrong type but correct name would compile silently. Mitigation: `SpinChannelData<T>` vs `T` are different types — the compiler won't silently coerce `Density` into `PerSpinDensity`.

---

### Component 3: Spin Loop in `diagonalize_inner`

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C3-S1 | **Hardcoded spin index 0**: `S::v_eff_for_spin(v_eff_ref, 0)` | `electronic.f90:488`: `do ns=1,wvfn%nspins` | `scf.rs:527` | **CRITICAL** | Only diagonalizes spin-up channel. For SpinCollinear (nspins=2), spin-down channel is never processed → eigenvalues only for spin-up, density from spin-up only → wrong total energy. |
| C3-S2 | Single `psi` upload to GPU | `electronic.f90:492`: per-spin slice `coeffs(:,:,nk,ns)` | `scf.rs:572-577` | **HIGH** | Only one set of wavefunctions is uploaded. Spin-down ψ needs separate upload → Davidson call. |
| C3-S3 | Single eigenvalue output | `electronic.f90:492`: per-spin `eigenvalues(:,nk,ns)` | `scf.rs:695` | **HIGH** | `result.eigenvalues` is a single `Vec<f64>`. Must be stored per-spin in `PerSpinEigenvalues`. |
| C3-S4 | VNL data precomputed once (not per-spin) | `hamiltonian.f90:1013`: `nlpot_prepare_precon` receives `nk, ns` | `scf.rs:589-596` | **HIGH** | VNL data (D-matrices) are per-(kpt, spin) because D-screening uses `∫Q·V_eff` and V_eff differs per spin channel. Must recompute inside spin loop. |
| C3-S5 | Single `beta_psi_gpu` output | Per-ion β·ψ differs per spin | `scf.rs:697-711` | **HIGH** | `beta_psi_gpu` computed from `result.psi_out` of single spin channel. Must store per-spin in `PerSpinBetaProjections`. |

**Root cause analysis**: The `diagonalize_inner` method was written when only `NonSpin` existed. The fix is to wrap the entire body (V_eff extraction through Davidson call through D2H and β·ψ recomputation) in `for ispin in 0..S::nspins()`, with per-spin indexing into `self.psi[ispin]`, `self.v_eff` (via `S::v_eff_for_spin(v_eff_ref, ispin)`), storing results into `self.eigenvalues[ispin]`, etc.

**Critical consideration**: VNL data (`VnlBatchData::precompute_with_d_override`) must be recomputed inside the spin loop because the D-matrices differ per spin channel. The screening potential `∫Q·V_eff` differs for V_eff_up vs V_eff_dn. Using the wrong spin channel's D-matrices would produce wrong V_NL contributions and wrong eigenvalues.

**CASTEP alignment**: `nlpot_prepare_precon` at `hamiltonian.f90:1013` receives `nk, ns` as explicit parameters. The D-matrix screening is per-(kpt, spin). Our current code precomputes `vnl_data` once outside the spin loop — this is only correct for NonSpin (single channel). For SpinCollinear, it must move inside.

---

### Component 4: `BuildVEff` for `SpinCollinear` (Paramagnetic Guess)

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C4-S1 | Zero spin density paramagnetic guess | `locpot.f90:278`: xc_calculate_potential receives `rho` + `sprho` from density | `scf.rs:343-346` | **HIGH** | `zero_spin` is correct for iter-0 initial guess (no wavefunctions yet). But the comment "Phase 3+ will compute proper spin density" is stale — Phase 7 is the implementation. |
| C4-S2 | `assemble_on_fine_grid` with `&zero_spin` — no energy return | `electronic_prepare_H` returns energy via `pot_calc_energy_real` | `scf.rs:347-348` | **HIGH** | Only returns `(V_up, V_dn)`, no energy components. `build_v_eff` (iter-0) doesn't need energy, but `build_v_eff_with_energy` (iter ≥1) does. This is handled by Component 5. |
| C4-S3 | Return type `(EffectivePotential, EffectivePotential)` — not newtyped | `pot.f90:76`: `real_fine_pot(:,ns)` — 2D array indexed by spin | `scf.rs:340` | **LOW** | The tuple return is a valid Rust encoding of per-spin V_eff. No semantic issue, but a `SpinChannelData<EffectivePotential>` wrapper would be more self-documenting. |

**Root cause analysis**: The paramagnetic guess is correct for the initial SCF iteration. CASTEP itself starts with a paramagnetic guess (zero spin density) and only develops spin polarisation through the SCF cycle. No fix needed for Component 4 — the deficiency is in Component 5 (energy-aware assembly for subsequent iterations).

---

### Component 5: `BuildVEffWithEnergy` for `SpinCollinear` — **MISSING**

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C5-S1 | **No `BuildVEffWithEnergy` impl for `SpinCollinear`** | `electronic_prepare_H` → `locpot_calculate`: assembles V_eff from ρ_total + ρ_spin, computes E_H + E_xc | `scf.rs:355-362` (trait only) | **CRITICAL** | `run_scf_with_energy::<SpinCollinear>(...)` won't compile. No path to track E_xc, E_H, ρV_xc for spin-polarised systems. |
| C5-S2 | Energy integral convention: `1/N_grid` vs `Ω/N_grid` | `xc.f90:565`: `1/n_grid` convention | `scf.rs:364-414` (NonSpin impl) | **HIGH** | Must use same `d_v = 1.0 / n_grid` as NonSpin. CASTEP consistently uses `1/N_grid` for energy integrals. Existing NonSpin impl already uses this — just needs SpinCollinear to follow same pattern. |
| C5-S3 | XC energy: `compute_pbe_xc_spin` vs `compute_pbe_xc` | `xc.f90:516-523`: spin-dependent XC evaluation | `scf.rs:365` (NonSpin calls `compute_pbe_xc`) | **HIGH** | Must call `compute_pbe_xc_spin(rho_total, rho_spin)` instead of `compute_pbe_xc(rho)`. The existing `compute_pbe_xc_spin` already returns `PbeXcSpinResult { v_xc_up, v_xc_dn, energy }`. |
| C5-S4 | `∫ρV_xc = Σ(ρ_up·V_xc_up + ρ_dn·V_xc_dn) / N_grid` | CASTEP `pot_calc_energy_real`: per-spin ∫ρV_xc sum | — | **HIGH** | Formula differs from NonSpin: two spin channels contribute independently. The discrete integral sums both channels: `rho_vxc = Σ_i (ρ_up[i]·V_xc_up[i] + ρ_dn[i]·V_xc_dn[i]) / n_grid`. |
| C5-S5 | Upstream change needed in chemrust-hamiltonian-core | `Built::assemble()` returns `(V_up, V_dn)` only, no energy | `band_structure.rs:406-431` | **MEDIUM** | Per TASKS.md TASK-9, the SCF layer calls `compute_pbe_xc_spin` directly (matching NonSpin path) rather than modifying upstream API. This avoids a breaking change but calls XC twice (once for potential in `assemble()`, once for energy in SCF layer). Deferred optimisation. |

**Root cause analysis**: This is the largest single missing piece. `BuildVEffWithEnergy` for `NonSpin` already exists and computes `E_H = 0.5·Σρ·V_H/N`, `ρV_xc = Σρ·V_xc/N`, `E_xc` from `PbeXcResult`. The SpinCollinear impl needs the same pattern but with: (1) spin density upsampled to fine grid, (2) `VEffBuilder::<SpinCollinear>::new(...).with_density(rho_total, Some(rho_spin)).assemble()`, (3) `compute_pbe_xc_spin` for XC energy, (4) two-channel `∫ρV_xc` sum.

**Implementation approach**: TASK-9 from TASKS.md — implement `build_v_eff_with_energy_impl` for `SpinCollinear` in `scf.rs`. TASK-10 — generalise the `build_v_eff_with_energy` public method from `impl ... NonSpin ...` to `impl<S: SpinPolicy + BuildVEffWithEnergy>`.

---

### Component 6: `compute_density_from_wavefunctions`

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C6-S1 | **Single-spin density construction**: `construct_density_gpu()` called once | `density.f90:2126-2164`: per-spin loop `do ns=1,nspins` | `scf.rs:1053-1063` | **CRITICAL** | Only ρ_up is constructed (from spin-0 ψ). ρ_down is never computed. Missing the spin combination step: ρ_total = ρ_up + ρ_down, ρ_spin = ρ_up − ρ_down. |
| C6-S2 | Single `Density` return — no `PerSpinDensity` | `density.f90:2179-2187`: `den%charge = up+down`, `den%spin = up−down` | `scf.rs:1027-1028` | **HIGH** | Return type `(Density, Option<RealGrid>, ChemicalPotential)` must become `(PerSpinDensity, Option<PerSpinAugDensity>, PerSpinOccupations, FermiEnergies)`. |
| C6-S3 | Single-channel occupation search: `compute_occupations(&self.eigenvalues, ...)` | `electronic.f90:488-495`: per-spin eigenvalues → per-spin occupations | `scf.rs:1046-1047` | **HIGH** | Occupations must be computed per spin channel. For fermi_fix: independent per-spin bisection. For fermi_free: shared Fermi energy integrating both channels. See Component 10. |
| C6-S4 | Single-channel augmentation: `beta_psi_per_ion` used as-is | `ion.f90:7544-7577`: per-spin β·ψ → per-spin rho_ij | `scf.rs:1093-1170` | **HIGH** | USPP augmentation density must be computed per spin channel. β·ψ differs for ψ_up vs ψ_dn → rho_ij differs → Q_rho_sum and Q_rho_sum_sp differ. Deferred for Phase 7 scope but the data flow must support per-spin augmentation. |
| C6-S5 | Density convention: `Density` wraps `WaveGridArray` (raw ρ×Ω) | `density.f90:2181-2187`: same convention | `scf.rs:1053` | **LOW** | Convention unchanged — both ρ_up and ρ_down stored in raw ρ×Ω units. Total density = ρ_up + ρ_down inherits same units. No convention change needed. |

**Root cause analysis**: The density construction pipeline was designed for NonSpin where there is one set of wavefunctions → one density. For SpinCollinear, the pipeline must: (1) loop over spin channels constructing per-spin soft density, (2) combine to total and spin density using the CASTEP formula (`ρ_total = ch_up + ch_down`, `ρ_spin = ch_up − ch_down`), (3) compute per-spin occupations (or shared-Fermi occupations), (4) compute augmentation per spin channel.

**CASTEP alignment verification** (density.f90:2179-2187):
```fortran
if(wvfn%nspins.eq.2) then
   den%real_charge(i) = den%real_charge(i) + den%real_spin(i)   ! = up+down
   den%real_spin(i)   = den%real_charge(i) - 2.0*den%real_spin(i)  ! = up-down
end if
```
This formula transforms accumulated |ψ_up|² (stored in `den%real_spin` during the per-spin loop) and |ψ_down|² (added to `den%real_charge` during the same loop) into the final charge and spin density arrays.

---

### Component 7: `construct_density_off/kerker/pulay`

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C7-S1 | `construct_density_off`: `new_density: Density` → must be `PerSpinDensity` | `density.f90`: stores both charge and spin | `scf.rs:1183` | **MEDIUM** | The density assignment `next.density = new_density` must become per-spin. |
| C7-S2 | `construct_density_kerker`: same type change | — | `scf.rs:1190` | **MEDIUM** | Same as C7-S1 — density stored in ScfIteration changes type. |
| C7-S3 | `construct_density_pulay`: same type change | — | `scf.rs:1230` | **MEDIUM** | Same as C7-S1. |

**Root cause analysis**: These three methods are thin wrappers around `compute_density_from_wavefunctions()` with different mixing-phase transitions. They don't contain algorithm logic — they just assign the returned density to the state and transition to the appropriate phase. The changes are purely mechanical: `Density` → `PerSpinDensity`, `Option<RealGrid>` → `PerSpinAugDensity`.

---

### Component 8: `mix()` — Density Mixing

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C8-S1 | Mixing operates on single `Density` | `dm_mix_density`: mixes `charge(:)` + `spin(:)` independently | `scf.rs:1282-1360` | **HIGH** | Mixing must handle both ρ_total and ρ_spin. CASTEP mixes them independently with potentially different mixing amplitudes. |
| C8-S2 | `previous_density` tracks single `Density` | — | `scf.rs:129` | **MEDIUM** | Density history needs per-spin tracking. `ρ_up` and `ρ_down` diverge during SCF — mixing each independently requires separate previous values. |
| C8-S3 | Kerker/Pulay mixing of spin density | NiO `.param`: `spin_density_mixing_amplitude=2.0`, `spin_density_mixing_g_vector=1.5` | — | **MEDIUM** | Spin density mixing may use different parameters than charge density mixing. For Phase 7 minimum, mix ρ_spin with the same parameters as ρ_total. Deferred: per-component mixing amplitudes. |

**Root cause analysis**: CASTEP's `electron_density` type stores `charge(:)` and `spin(:)` separately, and the mixing pipeline mixes both independently. For NonSpin, `spin(:)` is not allocated — only `charge(:)` is mixed. For SpinCollinear, both are mixed. Our `PerSpinDensity` stores `ρ_up` and `ρ_down` — the mixing should operate on derived `ρ_total` and `ρ_spin`, or directly on the per-spin densities. The existing `DensityHistory<M>` stores a `Vec<Density>` — this becomes a history of `PerSpinDensity`.

---

### Component 9: `check()` — Convergence Check

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C9-S1 | Convergence criteria use single `eigenvalues` and single `Density` | `electronic_check_occupancies`: checks total energy convergence | `scf.rs:1421-1528` | **MEDIUM** | Convergence uses total energy (sum of both spin contributions) — this is already correct once total energy is properly computed from per-spin contributions. |
| C9-S2 | Fermi energy stored as single `Option<f64>` | `electronic.f90`: per-spin Fermi energies | `scf.rs:153` | **MEDIUM** | Must store `FermiEnergies` (per-spin). Convergence diagnostics may want to report per-spin Fermi energies. |

**Root cause analysis**: The convergence check (`check()`) monitors total energy stability across SCF iterations. This is spin-independent — total energy is a scalar regardless of spin channels. However, the energy components feeding into total energy must be correctly computed from per-spin contributions. Once Components 5 and 6 are fixed (correct V_eff energy and density), `check()` naturally works because total energy is a scalar sum.

---

### Component 10: Occupation Search

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C10-S1 | **No `fermi_fix` / `fermi_free` distinction** | `electronic.f90:516-518`: `if(scf_cycle == spin_fix) call fermi_fix else call fermi_free` | `density.rs` | **CRITICAL** | Current `compute_occupations` does a single bisection for one electron count — equivalent to fermi_fix for NonSpin. No fermi_free path at all. No spin_fix transition logic. |
| C10-S2 | Single electron count: `n_electrons` from total valence | `electronic.f90:8742-8746`: `frac_elec(1) = 0.5*(N+spin)`, `frac_elec(2) = 0.5*(N−spin)` | `scf.rs:1034-1044` | **HIGH** | Must derive `N_up` and `N_dn` from total electrons + net spin (from `.cell SPIN=` block). For NiO: N_up=36, N_dn=28, net_spin=8. |
| C10-S3 | Single bisection search — no per-spin bisection | `electronic.f90:8757`: `do ns=1,nspins` for fermi_fix | `density.rs` (compute_occupations) | **HIGH** | fermi_fix requires independent bisection per spin channel with separate electron counts. See TASK-11. |
| C10-S4 | No shared Fermi energy search (fermi_free) | `electronic.f90:9094-9106`: one bisection integrating both spins | — | **HIGH** | fermi_free requires bisection that integrates occupancies over BOTH spin channels in each trial, finding one shared E_F. See TASK-12. |
| C10-S5 | No `net_spin` computation in fermi_free | `electronic.f90:9183-9209`: net_spin = Σocc_up − Σocc_dn after shared E_F | — | **HIGH** | After finding shared E_F in fermi_free, net spin is computed from the resulting occupancies. This is `intent(out)` — it differs from the `.cell` SPIN value. |
| C10-S6 | No `spin_fix` parameter or transition logic | `electronic.f90:516-518`, NiO `.param`: `spin_fix=6` | — | **MEDIUM** | The SCF loop must track iteration count and switch from fermi_fix to fermi_free after `spin_fix` iterations. Default: 5. NiO uses 6. |
| C10-S7 | Smearing formula: Gaussian only | `algor.F90:2929`: 5 schemes (GAUSSIAN, FERMIDIRAC, HERMITEPOLYNOMIALS, COLDSMEARING, GAUSSIANSPLINES) | `density.rs` | **LOW** | NiO uses Gaussian smearing (0.1 eV width). The existing `compute_occupations` uses `erfc`-based Gaussian occupancy. No change needed for Phase 7. |

**Root cause analysis**: The occupation search is the most algorithmically complex component. CASTEP has two distinct search strategies controlled by `spin_fix`:
- **fermi_fix** (electronic.f90:8602-8880): Independent per-spin bisection. `net_spin` is `intent(in)` — fixed from `.cell SPIN=`. Each spin channel gets its own E_F. Electron counts: `frac_elec(1)=0.5*(N+spin)`, `frac_elec(2)=0.5*(N-spin)`.
- **fermi_free** (electronic.f90:8910-9209): One shared Fermi energy. `net_spin` is `intent(out)` — computed from occupancies after E_F is found. Both channels get the same E_F: `fermi_energy(2) = fermi_energy(1)`.

The profile evidence confirms the transition: 5 `fermi_fix` calls + 62 `fermi_free` calls = 67 SCF iterations. The `spin_fix` keyword (default 5, NiO uses 6 per `.param`) controls when the switch occurs.

---

### Component 11: FFI `chemrust_eigensolve_init`

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C11-S1 | **No `nspins` parameter** | `chemrust_eigensolve.f90:134`: passes `nspins` from `wvfn%nspins` | `ffi.rs` | **HIGH** | Must accept `nspins: c_int` to allocate per-spin data arrays. Without it, all allocations are single-channel. |
| C11-S2 | Per-k-point `VnlBatchData` not per-spin | — | `ffi.rs` (KptData struct) | **MEDIUM** | `KptData` gains `vnl: Vec<Option<VnlBatchData>>` — one per spin channel. VNL data is per-(kpt, spin). |
| C11-S3 | V_eff cache single-entry, not per-spin | — | `ffi.rs` (ChemrustHandle) | **MEDIUM** | `v_eff_cached` becomes `Vec<Option<CudaSlice<f64>>>` — one GPU cache per spin channel. `v_eff_norm` becomes `Vec<f64>` — one norm per spin channel. |

**Root cause analysis**: The FFI init path allocates all GPU-resident data structures. Without `nspins`, all allocations are single-channel. With `nspins: c_int`:
- K-point data arrays use spin-major layout: `kpts[ispin * nkpts + ikpt]`
- Per-k-point `VnlBatchData` stored per-spin
- V_eff cache per-spin (two separate GPU buffers for change detection)

The Fortran side already has `nspins` available from `wvfn%nspins` — just needs to pass it.

---

### Component 12: FFI `chemrust_eigensolve_step`

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C12-S1 | **No `ispin` parameter** | `electronic.f90:522`: passes per-spin slice `real_fine_pot(:,ns)` | `ffi.rs` | **HIGH** | Must accept `ispin: c_int` to index correct spin channel for V_eff cache, VNL data, wavefunction buffer. |
| C12-S2 | V_eff upload reads single cache entry | — | `ffi.rs` (step_inner) | **HIGH** | V_eff upload must use per-spin cache: `h.v_eff_cached[ispin]` instead of single entry. |
| C12-S3 | VNL lookup single-entry, not per-spin | — | `ffi.rs` | **HIGH** | VNL data lookup: `kd.vnl[ispin]` instead of single entry. |
| C12-S4 | Fortran → Rust spin index conversion: CASTEP `ns` is 1-based (1=up, 2=down) | `electronic.f90:488`: `ns=1,wvfn%nspins` | `ffi.rs` | **MEDIUM** | Rust `ispin` is 0-based. Fortran wrapper must convert: `ispin = ns - 1`. Critical: getting this wrong swaps up/down channels → wrong eigenvalues + wrong density. |
| C12-S5 | Data arrays already per-spin at Fortran boundary | `electronic.f90:523`: `coeffs(:,:,nk,ns)` — Fortran slices single channel | — | **LOW** | The wavefunction and V_eff arrays are already sliced to the current spin channel by Fortran before the FFI call. No Rust-side offset computation needed for array data — only for cache lookup. |

**Root cause analysis**: The FFI step path is called inside CASTEP's spin loop. CASTEP already slices `wvfn%coeffs(:,:,nk,ns)` and `local_pot%real_fine_pot(:,ns)` to the current spin channel. The Rust side just needs to know WHICH channel it's receiving to index its internal caches correctly. The 1-based → 0-based conversion is a classic off-by-one risk (see failure-patterns.md: `ffi-1based-index-leak`).

---

### Component 13: `run_scf` / `run_scf_with_energy` / `run_scf_with_energy_gated`

| ID | Description | CASTEP | Rust Line | Severity | Root Cause |
|----|-------------|--------|-----------|----------|------------|
| C13-S1 | **`S: SpinPolicy + BuildVEff` bound — NonSpin only in practice** | `electronic_minimisation`: generic over nspins | `scf.rs:1548` | **HIGH** | `run_scf` already has the right trait bound `S: SpinPolicy + BuildVEff`. But `SpinCollinear` doesn't implement `BuildVEffWithEnergy` yet (Component 5). Compiler prevents calling `run_scf_with_energy::<SpinCollinear>()`. |
| C13-S2 | Energy tracking uses single `total_energy: Option<f64>` | — | `scf.rs:151` | **MEDIUM** | Total energy is spin-independent (scalar sum). No change needed to the energy tracking machinery — only the energy computation (Component 5). |
| C13-S3 | `scf_iter` counter used for spin_fix transition | `electronic.f90:516-518` | `scf.rs:177` | **LOW** | `scf_iter` already exists and is 1-indexed. Can be used directly for `if scf_iter <= spin_fix { fermi_fix } else { fermi_free }`. |

**Root cause analysis**: The high-level SCF loop drivers (`run_scf`, `run_scf_with_energy`, `run_scf_with_energy_gated`) are mostly correct — they're generic over `S: SpinPolicy` and the phase transitions are type-driven. The main gap is that `BuildVEffWithEnergy` has no `SpinCollinear` impl, which blocks `run_scf_with_energy::<SpinCollinear>()` at compile time. Once Component 5 is implemented, these functions work as-is.

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

## 5. Recommended Fix Priority Order

### Priority 0: Foundation — Block All Other Work

| Rank | ID | Description | Fix | Estimated Effort |
|------|----|-------------|-----|-----------------|
| **P0-1** | C1-all | Add `SpinChannelData<T>` + per-spin newtypes | TASK-1, TASK-2 | New file `spin_types.rs`, ~150 lines |
| **P0-2** | C1-S1–S7 | Replace single-spin fields in `ScfIteration` | TASK-3 | ~30 lines of field type changes |
| **P0-3** | C2-all | Update `into_phase()` for new field types | TASK-4 | Compiler-driven; field count unchanged |

### Priority 1: Spin Loop — Make SCF Work for SpinCollinear

| Rank | ID | Description | Fix | Estimated Effort |
|------|----|-------------|-----|-----------------|
| **P1-1** | C3-S1–S5 | Wrap `diagonalize_inner` in spin loop | TASK-6 | ~30 lines: add `for ispin` loop, index per-spin |
| **P1-2** | C6-S1–S4 | Per-spin density construction + combine | TASK-7, TASK-13 | ~60 lines: loop + ρ_total/ρ_spin combine |
| **P1-3** | C5-S1–S5 | Implement `BuildVEffWithEnergy` for `SpinCollinear` | TASK-9, TASK-10 | ~80 lines: upsampling + XC call + energy integrals |
| **P1-4** | C10-S1–S7 | Occupation search: fermi_fix + fermi_free | TASK-11, TASK-12 | ~150 lines: two bisection functions + spin_fix logic |

### Priority 2: FFI — Wire Spin Through Fortran Boundary

| Rank | ID | Description | Fix | Estimated Effort |
|------|----|-------------|-----|-----------------|
| **P2-1** | C11-S1–S3 | Add `nspins` to init, allocate per-spin caches | TASK-14 | ~30 lines Rust + ~10 lines Fortran |
| **P2-2** | C12-S1–S5 | Add `ispin` to step, per-spin cache indexing | TASK-15 | ~20 lines Rust + ~10 lines Fortran |

### Priority 3: Polish — Mixing, Convergence, Integration Test

| Rank | ID | Description | Fix | Estimated Effort |
|------|----|-------------|-----|-----------------|
| **P3-1** | C7, C8 | Per-spin density mixing + construct_density_* methods | TASK-8 | ~40 lines mechanical changes |
| **P3-2** | C9 | Convergence check with per-spin data | TASK-7 (includes) | ~10 lines |
| **P3-3** | C13 | Generalise `run_scf` bounds | TASK-10 (includes) | ~5 lines |
| **P3-4** | Integration | NiO discriminator test | TASK-16 | New file `nio_spin_scf.rs`, ~200 lines |

---

## 6. Cascading Dependency Chains

```
P0-1 (SpinChannelData<T>)
  └── P0-2 (ScfIteration fields)
        └── P0-3 (into_phase)
              ├── P1-1 (diagonalize_inner spin loop)
              │     └── P1-2 (density construction) ─── P1-3 (V_eff energy)
              │           └── P1-4 (occupation search)
              ├── P2-1 (FFI init)
              │     └── P2-2 (FFI step)
              └── P3-1 (mixing)
                    └── P3-2 (convergence)
```

Every path goes through `SpinChannelData<T>`. No parallel work possible before P0 is complete.

---

## 7. CASTEP Profile Evidence

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

---

## 8. Verification Matrix

| # | Criterion | Fixture Anchor | Tolerance | Discriminator Ratio |
|---|-----------|---------------|-----------|---------------------|
| V1 | `diagonalize_inner` calls Davidson twice per SCF iter for SpinCollinear | Profile: 134 calls | Exact (compiler-enforced) | ∞ (doesn't compile otherwise) |
| V2 | `V_eff_up ≠ V_eff_dn` for spin-polarised density | Self-consistency | | Qualitative |
| V3 | Per-spin eigenvalues match `.bands` | NiO.bands:12,75 | 1×10⁻⁴ Ha | ~1000× |
| V4 | Total energy matches CASTEP | NiO.castep:774418 (−7160.230577732 eV) | 1×10⁻⁶ Ha | ~1000× |
| V5 | Integrated spin density: 2∫ρ_spin = −0.0619641 | NiO.castep:774415 | 1×10⁻⁴ rel | ~100× |
| V6 | Fermi energies: 0.152664 Ha both spins | NiO.bands:5 | 1×10⁻⁴ Ha | ~100× |
| V7 | Occupations: N_up ≈ 36.00, N_dn ≈ 28.00 | NiO.bands:3 | ±0.01 | ~1000× |
| V8 | NonSpin regression: existing tests pass | Cu111_CO fixture | Identical results | Regression guard |
| V9 | FFI init accepts nspins, step accepts ispin | Compiler | Compiles + CASTEP builds | |
| V10 | No `Vec<Vec<T>>` in public API touching per-spin data | grep | 0 occurrences | Style |

---

## 9. Style Conformance

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
| `density.f90` | 2120-2195 | `density_calculate_soft_wvfn_real`: per-spin |ψ|² → charge/spin |
| `locpot.f90` | 74-422 | `locpot_calculate`: V_H + V_locps + V_xc per spin |
| `xc_gga.f90` | 516-523 | XC potential: total + spin density → V_xc_up, V_xc_dn |
| `hamiltonian.f90` | 723 | `hamiltonian_diagonalise_ks` signature: `nk, ns` as scalars |
| `hamiltonian.f90` | 1013 | `nlpot_prepare_precon` signature: `nk, ns` → per-spin D-matrices |
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
