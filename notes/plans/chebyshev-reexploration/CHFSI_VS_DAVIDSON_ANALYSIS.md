# ChFSI vs Davidson — Fundamental Differences and ABINIT's Cascade Prevention

**Date:** 2026-06-16
**Source:** ABINIT source code audit (m_chebfi2.F90, m_vtowfk.F90, m_scfcv_core.F90,
           m_inwffil.F90, m_newrho.F90, m_chebfi.F90, m_chebfiwf.F90)
**Previous analysis:** `CASTEP_ALGORITHM_LESSONS.md` (Davidson), `CHEFSI_AS_EIGENSOLVER.md` (Chebyshev theory)

---

## 1. Fundamental Algorithmic Differences

### Davidson (CASTEP: hamiltonian.F90, our davidson.rs)

```
Outer loop (0-30 iterations per SCF step):
  1. Fresh H·ψ, S·ψ
  2. Full n_bands ZHEGVD (wave_diagonalise)
  3. Convergence invalidation (|Δλ| > tol → un-converge)
  4. Block loop (groups of 2√n_bands):
     Inner loop (0-max_inner):
       a. Preconditioned residual: t = P⁻¹·(H−εS)·ψ
       b. S-orthogonalize search direction
       c. Extend superspace (accumulate search dirs)
       d. ZHEGVD on superspace
       e. Rotate, check convergence (eigenvalue-change criterion)
       f. Compact unconverged bands → repeat
```

**Key property:** Subspace grows with iterations. Memory = O(n_pw × n_bands × superspace_factor).

### ChFSI (ABINIT: m_chebfi2.F90, chebfi_run)

```
Single pass per SCF iteration (NO outer loop):
  1. Fresh H·ψ, S·ψ
  2. Per-band Rayleigh quotients: λ_b
  3. λ_minus = max(λ_b), λ_plus = ecut
  4. Dynamic filter degree via oracle: ndeg = min(cheb_oracle1(...), 40)
  5. For ideg = 0..ndeg-1:
     a. X_next = S⁻¹·H·X_cur − c·X_cur
     b. X_next = (1/r or 2/r)·X_next − X_prev
     c. Swap buffers, fresh H·ψ + S·ψ
  6. Amplification factor: X *= 1/T_n(ε_b)  (normalize per band)
  7. Rayleigh-Ritz: ZHEGVD on H_sub/S_sub
  8. Residual computation: ‖H·ψ − λ·S·ψ‖₂
```

**Key property:** Fixed memory. 3 buffers for 3-term recurrence. No subspace accumulation.

### Head-to-Head

| Property | Davidson | ChFSI |
|---|---|---|
| Subspace enrichment | Preconditioned steepest descent | Chebyshev polynomial C_p(S⁻¹·H) |
| Memory | Grows: superspace × n_bands | Fixed: 3 × n_bands buffers |
| Convergence metric | Eigenvalue change |Δλ| | Residual norm ‖r‖ |
| H·ψ recomputation | Every outer iteration (~1-30×) | Every filter degree (ndeg ×, up to 40×) |
| Band locking | Explicit lock_mask | Per-band ndeg=0 (excluded from filter) |
| Inner convergence loop | Yes (block-level inner loop) | No (external SCF loop drives convergence) |
| Preconditioner | TPA + USPP | Chebyshev polynomial IS the accelerator |
| Amplitude control | Built into preconditioned residual | Explicit ampfactor normalization |
| Spectral info needed | mean_ek for TPA | λ_minus, λ_plus for filter window |

---

## 2. ABINIT's Cascade Prevention Mechanisms

### 2.1 Amplification Factor (THE load-bearing mechanism)

**Location:** `m_chebfi2.F90:958-1006` (`chebfi_ampfactor`)

After the Chebyshev polynomial iterations, each band's wavefunction has been amplified by T_n(ε_b) where ε_b is the eigenvalue. For the lowest eigenvalue with ndeg=20, T_20(ε_b) can be 10^4-10^8.

The ampfactor normalizes each band:
```fortran
ampfactor = max(|T_n(ε_b)|, 1e-3)    ! clamp floor
X_b /= ampfactor                       ! normalize
HX_b /= ampfactor
SX_b /= ampfactor
```

**Why this prevents cascading:** Without normalization, the Chebyshev polynomial amplifies wanted eigencomponents exponentially. If ε_b from a cascade-corrupted SCF iteration is wrong, the ampfactor is wrong, and the wavefunction norm becomes inconsistent with the density normalization — the density changes spuriously, V_eff drifts, and the cascade self-amplifies. The ampfactor floor (1e-3) prevents division by zero for eigenvalues at the filter boundary.

**Vulnerability:** If eigenvalues from a cascade-corrupted state are used to compute the ampfactor, the normalization is wrong. ABINIT mitigates this by computing ampfactor from **current SCF iteration** Rayleigh quotients, not from stored eigenvalues.

### 2.2 Per-Band Oracle (Dynamic Filter Degree)

**Location:** `m_chebfi2.F90:1128-1214` (`chebfi_set_ndeg_from_residu`)

Sets per-band filter degree based on residual level:

```fortran
if res_iband < tolerance:     ndeg_filter_bands(iband) = 0   ! converged → lock
if band in buffer:            ndeg_filter_bands(iband) = 0   ! buffer → lock
if low occupancy (nbdbuf=-101): ndeg_filter_bands(iband) = 0  ! unoccupied → lock
otherwise:  ndeg_filter_bands(iband) = compute from oracle(1 or 2)
```

**Why this prevents cascading:** Bands that are already converged are EXCLUDED from the Chebyshev recurrence entirely (ndeg=0). This prevents the filter polynomial from distorting already-correct eigenvectors — the same purpose as Davidson's locking mechanism. The buffer bands (highest-energy states) are also excluded — their eigenvalues are near the filter boundary and would be minimally amplified anyway.

### 2.3 Quality Random Initialization

**Location:** `m_inwffil.F90:247-252`

When using Chebyshev filtering, ABINIT uses a higher-quality random number generator for initial wavefunctions:
```fortran
if(wfoptalg == 1 .or. wfoptalg == 111) randalg = 1  ! better RNG for Chebyshev
```

**Why this prevents cascading:** Poor initial wavefunctions in metal systems produce near-equal Rayleigh quotients across the occupied manifold. The filter window [λ_minus, λ_plus] becomes too narrow or mispositioned, amplifying wrong spectral components from the start.

### 2.4 Rayleigh-Ritz as Recovery

**Location:** `m_chebfi2.F90:705-706` (`xg_RayleighRitz`)

After filtering, the subspace diagonalization extracts the optimal eigenbasis. Any spectral pollution introduced during the polynomial recurrence is discarded because ZHEGVD finds the best eigenvectors within the filtered span.

**Why this prevents cascading:** The Chebyshev polynomial amplifies the wanted subspace but also introduces numerical noise (κ₂ growth, see ChASE paper). RR recovers the correct eigenbasis before density construction.

### 2.5 SCF Density Mixing (Pulay/Anderson)

**Location:** `m_newrho.F90:168`, `m_abi_mixing.F90:677`

ABINIT's density mixing (Pulay/Anderson) damps SCF oscillations. The mixing object stores a history of densities and uses the `npulayit` parameter to control the Pulay history length.

**Why this prevents cascading:** An oscillating SCF creates an oscillating V_eff, which creates an oscillating Hamiltonian. The Chebyshev filter window depends on accurate λ_minus from Rayleigh quotients. If λ_minus oscillates between SCF iterations, the filter window shifts, amplifying different spectral components each iteration. Density mixing smooths this.

### 2.6 Filter Degree Cap (ndeg ≤ 40)

**Location:** `m_chebfi2.F90:625`

```fortran
ndeg_filter_max = cheb_oracle1(mineig, lambda_minus, lambda_plus, 1D-16, 40)
```

**Why this prevents cascading:** Limits numerical blowup. T_40(x) for x slightly outside [-1,1] can be 10^16. Higher degrees add negligible eigenvalue improvement but massive numerical noise.

---

## 3. What chemrust-scf's Chebyshev Path Is Missing

### CRITICAL (will cause cascade without these):

| Mechanism | ABINIT | chemrust-scf | Consequence of absence |
|---|---|---|---|
| **Amplification factor** | `chebfi_ampfactor` (lines 958-1006) | NOT IMPLEMENTED | Wavefunction amplitudes grow exponentially → density normalization breaks → cascade |
| **Per-band oracle** | `chebfi_set_ndeg_from_residu` (lines 1128-1214) | NOT IMPLEMENTED | All bands get same filter degree; converged bands are distorted; no locking |
| **Residual-norm convergence** | `‖H·ψ − λ·S·ψ‖₂` per band (line 710-716) | NOT IN CHEBYSHEV PATH | No way to know if filter succeeded or failed |
| **Rayleigh quotients pre-filter** | `chebfi_rayleighRitzQuotients` (line 761) | Lanczos estimator instead | λ_minus from Lanczos may miss the correct filter window position |

### IMPORTANT (will degrade convergence without these):

| Mechanism | ABINIT | chemrust-scf | Consequence of absence |
|---|---|---|---|
| **Band buffer (nbdbuf)** | Excludes top bands from convergence (line 89) | NOT IMPLEMENTED | High-energy bands drive convergence criteria needlessly |
| **Occupancy-weighted residuals** | `nbdbuf=-101` mode scales residual by occupancy | NOT IMPLEMENTED | Unoccupied bands force higher filter degree, wasting compute |
| **Filter degree cap (40)** | `ndeg_filter_max = min(cheb_oracle1(...), 40)` | User-specified ndeg | No protection against numerical blowup |
| **SCF density mixing** | Pulay with npulayit history | Pulay exists (scf.rs) but Chebyshev path bypasses it | Same mixing infrastructure, but not tested with Chebyshev |

---

## 4. The Cascade Loop — How Each Mechanism Breaks the Cycle

The cascade in chemrust-scf (old Chebyshev-RR) was:
```
Chebyshev filter → wrong ψ → wrong ρ → wrong V_eff → wrong H
    → wrong λ estimates → wrong filter window → worse ψ → ...
```

ABINIT breaks this cycle at multiple points:

```
[INIT] Better RNG → better initial ψ → more accurate first λ estimates  (2.3)
                                                    ↓
[WINDOW] Rayleigh quotients → λ_minus = max(λ_b) → correct filter window  (2.6)
                                                    ↓
[DEGREE] Per-band oracle → ndeg=0 for converged bands → locking equivalent  (2.2)
                                                    ↓
[AMPLITUDE] ampfactor = 1/T_n(ε_b) → prevents exponential blowup  (2.1)
                                                    ↓
[RECOVERY] Rayleigh-Ritz → ZHEGVD discards pollution  (2.4)
                                                    ↓
[MIXING] Pulay density mixing → damps oscillations → stable V_eff  (2.5)
                                                    ↓
[CONVERGENCE] Residual norms → accurate convergence detection → stops when done  (2.3)
```

Our old Chebyshev-RR had NONE of these. The Davidson path (current production) has: locking instead of oracle, eigenvalue-change convergence instead of residual norms, TPA preconditioner instead of ampfactor, and Pulay mixing.

---

## 5. Implementation Priority for chemrust-scf Chebyshev Path

**Must implement (Gate 1 — basic viability):**
1. Amplification factor — prevents exponential blowup
2. Per-band residual computation — enables convergence detection
3. Rayleigh quotients pre-filter — correct filter window positioning

**Should implement (Gate 2 — convergence quality):**
4. Per-band oracle — locking equivalent, dynamic filter degree
5. Filter degree cap — safety against numerical blowup

**Nice to have (Gate 3 — production quality):**
6. Band buffer (nbdbuf) — performance optimization
7. Occupancy-weighted residuals — better convergence for metals

---

## 6. Differences Consolidated

**ChFSI vs Davidson is NOT about which eigensolver is "better."** Both work for USPP metallic systems when properly guarded. The difference is:

- **Davidson** converges through **preconditioned gradient descent** — each iteration reduces the residual by ~constant factor. Needs many iterations per SCF step (5-30). Builds a growing subspace.

- **ChFSI** converges through **polynomial subspace amplification** — each filter degree exponentially amplifies the wanted subspace. Needs fewer iterations per SCF step (1-5 in practice, sometimes just 1). Uses fixed memory.

- **Both need locking** — Davidson uses explicit `band_converged` flags. ABINIT Chebyshev uses per-band `ndeg=0`.

- **Both need amplitude control** — Davidson through the TPA preconditioner's bounded amplification. ChFSI through explicit ampfactor normalization.

- **Both need convergence detection** — Davidson through eigenvalue-change tracking. ChFSI through residual-norm tracking.

The cascade in our old Chebyshev-RR wasn't because "Chebyshev can't work for metals." It was because we implemented the Chebyshev filter in isolation — without ampfactor, without oracle, without residual-based convergence, and with a mis-positioned filter window. Any eigensolver would fail under those conditions.
