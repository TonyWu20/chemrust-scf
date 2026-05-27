# ABINIT Chebyshev Filtering in the SCF Loop

Reference document mapping ABINIT's Chebyshev filtering eigensolver within its SCF
iteration structure. Source tree: `~/programming/abinit/`.

## SCF Loop Structure

The SCF convergence loop lives in `scfcv_core` (`src/94_scfcv/m_scfcv_core.F90`). The
density/wavefunction solve for a fixed Hamiltonian is delegated to `vtorho`, which is
called once per SCF iteration. `vtorho` contains a second, inner loop that re-applies
the eigensolver against the *same* Hamiltonian multiple times.

Full annotated structure:

```
scfcv_core (src/94_scfcv/m_scfcv_core.F90:1058-2200)
│
┌─ do istep=1, max(1, nstep)          ← SCF outer loop (line 1058)
│                                       Hamiltonian V_trial is UPDATED each iteration
│                                       via density mixing (newrho) + potential mixing (newvtr)
│
│    [precompute atomic data, FFT grids, PAW data — only on istep=1 or moved_atm_inside]
│
│    call vtorho(...)                   (line 1662) — src/79_seqpar_mpi/m_vtorho.F90
│    │
│    │  nnsclo_now ← set by logic below (lines 590-631)
│    │
│    │  for each k-point, spin:
│    │    call vtowfk(...)               src/79_seqpar_mpi/m_vtowfk.F90
│    │    │
│    │    ┌─ do inonsc=1, nnsclo_now    ← inner loop (line 382) — FIXED H
│    │    │
│    │    │    [pre-filter: zero wave-vector components above kinetic energy cutoff]
│    │    │
│    │    │    select case(wfopta10):
│    │    │      case 1:  chebfi / chebfiwf2 / chebfiwf2_cprj   (lines 493-506)
│    │    │                  ↑ Chebyshev filter
│    │    │      case 4:  lobpcgwf / lobpcgwf2 / lobpcgwf2_cprj (lines 449-483)
│    │    │      default: cgwf / cgwf_cprj                      (lines 513-535)
│    │    │
│    │    │    [orthogonalisation — skipped for Chebyshev (wfopta10==1)]
│    │    │
│    │    │    max_resid = maxval(resid_k(1:nband_k-nbdbuf))     (line 557)
│    │    │              ↑ buffer bands excluded from convergence metric
│    │    │
│    │    │    if (max_resid < tolwfr) exit ← early exit on WF convergence
│    │    │
│    │    └─ end do                       (line 696)
│    │
│    │    [accumulate density rho from wavefunctions]
│    │
│    end vtorho
│
│    if (dtset%iscf < 0) exit           ← NSCF: skip mixing, done (line 1717)
│    ↑ already did nstep inner loops inside vtorho for NSCF
│
│    call scprqt → check SCF convergence (deltae, diffor, res2, residm)  (line 1781)
│    if (quit == 1) exit                ← SCF converged (line 1803)
│
│    call newrho  → mix density         (line 1867)
│    call newvtr  → construct new trial potential V_trial for next istep  (line 2123)
│
└─ end do ! istep                       (line 2200)
```

## The Inner Loop: `nnsclo_now` Logic

The number of inner eigensolver iterations per SCF step is controlled by `nnsclo_now`,
set in `vtorho` (`m_vtorho.F90:590-631`):

```
     iscf < 0 (NSCF)?
       → nnsclo_now = dtset%nstep          ← many iterations, tolwfr-driven exit
     iscf > 0 (SCF)?
       dtset%nnsclo > 0?
         → nnsclo_now = dtset%nnsclo        ← user-specified
       dtset%nnsclo < 0?
         → nnsclo_now = 1
           if (istep <= abs(nnsclo)) nnsclo_now = 5 (or useria)
       dtset%nnsclo == 0 (default)?
         → nnsclo_now = 1
           if (istep <= 2) nnsclo_now = 2   ← first 2 SCF steps get 2 iterations
           (unless structural relaxation & RMM-DIIS, which reduces back to 1)
     dbl_nnsclo != 0?
       → nnsclo_now *= 2
```

**SCF defaults** (the common case for production Chebyshev runs):

| SCF step | Default `nnsclo_now` | Rationale |
|---|---|---|
| 1 | 2 | WFs initialised from random/atoms; extra refinement helps |
| 2 | 2 | WFs still settling from potential change |
| 3+ | 1 | One Chebyshev pass per SCF iteration |

The intent is that after the first two SCF iterations the wavefunctions from the
previous SCF step (slightly stale due to potential mixing, but close) are good enough
that a single Chebyshev filter pass per SCF iteration suffices.

### Convergence Check in the Inner Loop

After the eigensolver call, `max_resid` is computed at `vtowfk:557`:

```fortran
max_resid = maxval(resid_k(1:max(1, nband_k - nbdbuf)))
```

Then at line 690:

```fortran
if (max_resid < dtset%tolwfr) then
  exit  ← exits the `inonsc` loop
end if
```

Behaviour:
- **`max_resid > tolwfr`**: loop continues to next `inonsc` iteration (up to `nnsclo_now`)
- **`max_resid < tolwfr`**: loop exits early (wavefunctions converged for this k-point)
- **`nnsclo_now == 1` (default SCF)**: the convergence check exists but is inert — the
  loop runs exactly once regardless

This means setting `nnsclo > 1` changes the eigensolver from a single-pass filter to a
genuine convergence-driven inner loop within each SCF step.

## Residual Definition and `tolwfr`

`tolwfr` is a user-specified input parameter (not computed). The comparison is:

```
max( resid[1], resid[2], ..., resid[nband_k - nbdbuf] )  <  tolwfr
```

### What the residual IS

The residual is a **squared L2 norm** of the residual vector. There are two variants
depending on which Chebyshev implementation is used:

**V1** (`wfoptalg=1`, `m_chebfi.F90:544-566`):

```
                        residual vector r_i = H|ψ_i⟩ - ε_i·S|ψ_i⟩
                                      │
                                      ▼
                    r_i = r_i × pcon         (Teter et al. preconditioner)
                                      │
                                      ▼
            resid(iband) = dotprod_g(r_i, r_i, option=1)
                        = BLAS ddot(2·npw·nspinor, r_i, 1, r_i, 1)
                        = Σ Re(r_i)² + Σ Im(r_i)²
                        = |r_i|²                                       ← SQUARED norm
```
So in V1: `resid = ||pcon · (Hψ - εSψ)||²` — **preconditioned** squared L2 norm.

**V2** (`wfoptalg=111`, `m_chebfi2.F90:710-716`):

```
            AX = Hψ - λ·Sψ               (no preconditioner applied)
            resid = xgBlock_colwiseNorm2(AX) = ||AX||²                ← SQUARED norm
```
So in V2: `resid = ||Hψ - εSψ||²` — **unpreconditioned** squared L2 norm.

### The Preconditioner (`pcon`)

Used only in V1. Defined at `m_chebfi.F90:167`:

```fortran
pcon = (27 + T(18 + T(12 + 8T))) / (27 + T(18 + T(12 + 8T)) + 16T⁴)
```

where `T = kinpw` is the kinetic energy of each plane-wave component. This is the
Teter et al. preconditioner (also used by the CG eigensolver). It down-weights
high-kinetic-energy G-vector components in the residual norm, so the convergence
metric is dominated by low-G components where the wavefunction amplitude is largest.

### Practical Consequences

- `tolwfr` has units of **(Hartree)²** (since it's a squared norm comparison).
- A typical SCF setting like `tolwfr=1e-6` corresponds to an unpreconditioned RMS
  residual of roughly 1e-3 Ha per plane-wave component — though the actual meaning
  differs between V1 (preconditioned) and V2 (unpreconditioned).
- Because V2 omits the preconditioner, the same `tolwfr` value is a stricter threshold
  in V2 than in V1 (the preconditioner attenuates large kinetic-energy components,
  making the norm smaller for the same absolute wavefunction error).

### Buffer Bands (nbdbuf)

The `nbdbuf` parameter specifies how many of the highest bands to exclude from the
convergence metric (`max_resid`). These buffer bands are not expected to converge;
excluding them prevents the inner loop from running extra iterations chasing
unconverged high-empty bands that are irrelevant to the density.

## Chebyshev Filter Implementation: Two Versions

ABINIT has two Chebyshev filtering implementations, selected by `wfoptalg`:

| `wfoptalg` | Implementation | Module | Key Routine | xG abstraction? |
|---|---|---|---|---|
| 1 | Chebyshev filter V1 (2014) | `66_wfs/m_chebfi.F90` | `chebfi` | No |
| 111 | Chebyshev filter V2 (2021) | `48_diago/m_chebfi2.F90` + `79_seqpar_mpi/m_chebfiwf.F90` | `chebfiwf2` / `chebfiwf2_cprj` | Yes (via `m_xg`) |

Both implement the same algorithm described in Levitt & Torrent (2015). V2 adds GPU
support (Kokkos/YAKL backend) and uses the `xG` abstraction layer for linear algebra.

### Internal Structure of `chebfi` (V1, `wfoptalg=1`)

File: `src/66_wfs/m_chebfi.F90`, subroutine `chebfi` at line 102.

```
chebfi(cg, dtset, eig, enlx, gs_hamk, gsc, kinpw, mpi_enreg, nband, npw, nspinor, prtvol, resid)
│
│  1. INITIALISATION
│     - Set preconditioner pcon = f(kinpw)  (line 167)
│     - If paral_kgb: transpose CG data from (npw, nband) → (npw_filter, nband_filter)
│     - Allocate _filter, _next, _prev buffers for 3-term Chebyshev recurrence
│     - Allocate PAW cprj buffers if USPP
│
│  2. FIRST H|ψ⟩ APPLICATION (line 269-276)
│     call getghc(cpopt, cg, cwaveprj, ghc, gsc, gs_hamk, gvnlxc, ...)
│     ↑ Computes H·ψ AND S·ψ (if PAW) for all bands, in one batched call
│
│  3. COMPUTE RAYLEIGH QUOTIENTS (line 291-314)
│     eig(iband) = <ψ|H|ψ>  (NCPP)  or  <ψ|H|ψ> / <ψ|S|ψ> (PAW)
│     resids_filter(iband) = |pcon · (Hψ - eig·Sψ)|²  (preconditioned squared norm)
│     filter_low = maxeig  ← upper bound for the filter interval
│
│  4. PER-BAND ORACLE — DETERMINE POLYNOMIAL DEGREE (lines 318-327)
│     for each band:
│       ndeg_tolwfr   = cheb_oracle(eig, filter_low, ecut, tolwfr_diago/resid,  mdeg)
│       ndeg_decrease = cheb_oracle(eig, filter_low, ecut, 0.1,                 mdeg)
│       ndeg_filter_bands(iband) = max(min(ndeg_tolwfr, ndeg_decrease, ndeg_max, mdeg), 1)
│       [overridden to: ndeg_filter_bands = mdeg_filter  ← locking variant]
│
│  5. CHEBYSHEV POLYNOMIAL APPLICATION LOOP (lines 347-443)
│     do ideg = 1, mdeg_filter
│       ↑ THIS IS THE POLYNOMIAL DEGREE, NOT AN ITERATIVE CONVERGENCE LOOP
│
│       filter_center = (ecut + filter_low)/2
│       filter_radius = (ecut - filter_low)/2
│
│       a) Apply S⁻¹ to (Hψ) if PAW (line 370-377)
│          call apply_invovl(ghc → gsm1hc)
│
│       b) Chebyshev 3-term recurrence (lines 379-410):
│          ideg=1: ψ_next =  (S⁻¹Hψ - center·ψ) / radius
│          ideg≥2: ψ_next = 2(S⁻¹Hψ - center·ψ) / radius - ψ_prev
│          ↑ Updates ONLY bands with ndeg_filter_bands(iband) >= ideg
│
│       c) Apply H·ψ_next via getghc (line 428-442)
│          ↑ One getghc call per polynomial degree step
│
│     end do  ← mdeg_filter H applications total
│
│  6. AMPLITUDE NORMALISATION (lines 446-458)
│     Each band divided by T_{ndeg}(eig) where T_n is the Chebyshev polynomial
│     ↑ Prevents the polynomial amplification at the target eigenvalule
│
│  7. TRANSPOSE BACK (if paral_kgb)
│  8. RAYLEIGH-RITZ SUBSPACE DIAGONALISATION (lines 537-540)
│     call rayleigh_ritz_distributed(cg, ghc, gsc, gvnlxc, eig, ...)
│     ↑ Solves the projected nband×nband eigenproblem, rotates WFs
│
│  9. COMPUTE POST-RR RESIDUALS (lines 544-566)
│     resid(iband) = |pcon · (Hψ - eig·Sψ)|²  (preconditioned squared norm)
│     ↑ Uses the Teter et al. kinetic-energy preconditioner (pcon)
│       V2 omits this preconditioning step
```

Key observations:

- **No internal convergence loop**: `chebfi` applies the polynomial once, does one
  Rayleigh-Ritz, and returns. All `mdeg_filter` H applications are part of the single
  polynomial filter pass, not an outer convergence loop.
- **The oracle determines per-band polynomial degree**: converged bands get a low
  degree (or even 1, meaning a single Chebyshev iteration = essentially no filtering).
- **Locking**: Line 326 shows a commented locking path where
  `ndeg_filter_bands(iband) = dtset%mdeg_filter` for all bands, including converged
  ones. The user comment says "fiddle with this to use locking" — this was an
  experimental feature.
- **H is applied `mdeg_filter` times per call** (once per polynomial degree step).
  Default `mdeg_filter` is typically 10-20.

### Internal Structure of `chebfiwf2` (V2, `wfoptalg=111`)

File: `src/48_diago/m_chebfi2.F90`, subroutine `chebfi_run` at line 473.
Wrapper: `src/79_seqpar_mpi/m_chebfiwf.F90`, subroutine `chebfiwf2`.

The same logical structure as V1, but using the `xG` abstraction layer for
GPU-compatible linear algebra:

```
chebfi_run(chebfi, X0, getAX_BX, getBm1X, eigen, occ, residu, nspinor)
│
│  1. Transpose if paral_kgb
│  2. call getAX_BX (compute H|ψ⟩ and S|ψ⟩)
│  3. Rayleigh-Ritz quotients (eig = <ψ|H|ψ> / <ψ|S|ψ>)
│  4. Oracle: determine ndeg_filter from residuals (or fix at ndeg_filter)
│  5. do ideg = 0, ndeg_filter-1   ← Chebyshev polynomial degree loop
│       call chebfi_computeNextOrderChebfiPolynom
│       call getAX_BX                ← one H application per polynomial degree
│  6. Amplitude factor correction
│  7. Transpose back
│  8. xg_RayleighRitz  ← subspace diagonalisation
│  9. Compute residuals: resid = ||Hψ - eig·Sψ||²  (no preconditioner)
```

Key V2 additions:
- **`oracle` modes** (lines 1196-1204):
  - `oracle=1`: determine degree from residual-to-tolerance ratio
  - `oracle=2`: determine degree from a constant residual decrease factor
- **Band skipping via `nbdbuf` or occupancy** (lines 1188-1192):
  - Bands already converged (`resid < tolerance`)
  - Bands in the buffer (`iband > neigenpairs - nbdbuf`)
  - Low-occupancy bands (`nbdbuf=-101` and `occ < oracle_min_occ`)
- **Occupancy-aware oracle** when `nbdbuf=-101`, using occupancy thresholds

## The `cheb_oracle` Function

Computes the minimum Chebyshev polynomial degree `n` such that
`1 / T_n(x_red)² < tol`, i.e. the polynomial amplifies the eigenvector enough to
reduce the residual below tolerance. The mapped coordinate is:

```
x_red = 2(x - (a+b)/2) / (b-a)
```

where `a = filter_low` (max eigenvalue), `b = ecut` (filter upper bound).

For well-separated eigenvalues near the filter upper bound, the required degree is
small (often 1). For tightly clustered eigenvalues near the lower bound of the
occupied spectrum, the required degree can be large.

## Key Parameters

| Parameter | Scope | Default | Description |
|---|---|---|---|
| `wfoptalg` | `dtset` | 0 (CG) | `1` = Chebyshev V1; `111` = Chebyshev V2 |
| `mdeg_filter` | `dtset` | ~10-20 | Max Chebyshev polynomial degree (H applications per filter pass) |
| `nnsclo` | `dtset` | 0 (auto) | Number of inner NSCF loops per SCF step. >0 overrides default logic |
| `nbdbuf` | `dtset` | 0 | Number of buffer bands excluded from convergence metric |
| `tolwfr` | `dtset` | varies | Threshold on max squared residual norm: `max(resid) < tolwfr` exits inner loop. Units: (Hartree)². In V1 the residual is preconditioned (`pcon·(Hψ-εSψ)`); in V2 it is not (`Hψ-εSψ`) |
| `tolwfr_diago` | `dtset` | same as `tolwfr` | Tolerance used by oracle to compute required polynomial degree (V1 only). Ratio `tolwfr_diago/resid` controls how aggressively the oracle reduces the filter degree |
| `ecut` | `dtset` | user-set | Upper bound of Chebyshev filter interval (plane-wave cutoff) |
| `oracle` | chebfi2 | 0 | V2 only: oracle mode (0=fix ndeg; 1=resid→tol; 2=resid→factor) |
| `oracle_factor` | chebfi2 | — | Residual decrease factor for oracle mode 2 |

## Data Flow Summary

```
                  dtset%nnsclo, dtset%nstep, istep, dbl_nnsclo
                              │
                              ▼
                    ┌─────────────────┐
                    │ nnsclo_now = ... │   vtorho:590-631
                    └────────┬────────┘
                             │
    SCF loop         ┌───────▼───────────┐
    (scfcv_core)     │  do inonsc=1,     │
                     │     nnsclo_now    │
                     │                   │       ┌──────────────────────┐
                     │   chebfi/call     │◄──────┤ do ideg=1,mdeg_filter│
                     │   → getghc (×mdeg)│       │  3-term recurrence   │
                     │   → RR            │       │  getghc each step    │
                     │   → resid(iband)  │       └──────────────────────┘
                     │                   │
                     │   max_resid = ... │
                     │   if converged    │
                     │     exit          │
                     └───────────────────┘
                              │  (density ρ)
                              ▼
                    ┌─────────────────┐
                    │ SCF converged?  │──→ yes → exit SCF loop
                    └────────┬────────┘
                             │ no
                             ▼
                    ┌─────────────────┐
                    │ newrho (mix ρ)  │
                    │ newvtr (V_trial)│──→ next istep
                    └─────────────────┘
```

## Summary

ABINIT's Chebyshev filtering in SCF mode:

1. **Default: one filter pass per SCF iteration** (`nnsclo_now=1` for istep≥3).
2. The Chebyshev polynomial is applied at degree `mdeg_filter` (typically 10-20 H
   applications), followed by one Rayleigh-Ritz subspace diagonalisation. This is all
   within a single `chebfi` call — no convergence loop.
3. `nnsclo` controls an outer wrapper loop that re-applies the entire filter+RR
   sequence against the same Hamiltonian, with a `tolwfr`-based convergence check
   between iterations.
4. The oracle can reduce the polynomial degree for already-converged bands, making
   the filter effectively skip bands that need no further correction.
