# Phase 5: Production-Ready GPU-Resident Eigensolver

**Date:** 2026-05-31
**Status:** Draft

## Goals

### Goal 1: Root-cause the density normalization and cycle-5 divergence (Medium)

The FFI layer (`ffi.rs`) implements the full ABINIT Chebyshev recurrence with
T+V_loc+V_NL+S-operator on GPU, but the SCF diverges after 4 cycles and
returned wavefunctions produce ~2.7x the expected density. The filter itself
works — cycle 1-4 eigenvalues improve — but something compounds.

Diagnose by:
(a) Dumping the FFT normalization factors on Rust and CASTEP sides for the same
    wavefunction: check that `inv_ntotal` matches the cuFFT forward/inverse
    scaling convention expected by CASTEP's density builder,
(b) Verifying `<psi|psi>` in real space matches between sides — is the 2.7x
    coming from FFT normalization, density construction convention, or
    wavefunction normalization,
(c) Computing H·psi and S·psi through the Chebyshev recurrence with the ABINIT
    convention and verifying the per-band Rayleigh quotients `<psi_b|H|psi_b>`
    track CASTEP's eigenvalues,
(d) Testing n_inner=2 or 3 to see if the explosion is convergence-iteration-
    dependent or SCF-cycle-count-dependent,
(e) Checking whether the S-normalization (lines 889-935 of ffi.rs) `1/√<psi|S|psi>`
    correctly restores unit S-norm for bands that are superpositions of multiple
    eigenstates after T_n amplification.

**Why next**: Correctness blocks everything. You cannot measure convergence or
performance if the answer is wrong.

**Dependencies**: None.

### Goal 2: Fix the D-matrix sign convention for non-origin ions (Small)

The `chemrust-hamiltonian-core` crate has a known bug where V_NL D-matrix has
the wrong sign for ions not at the origin. Ion 1 (at the origin) produces
correct results because the phase factor `exp(±iG·R) = 1`, masking the sign
error. Non-origin ions contribute with wrong sign, corrupting V_NL.

This manifests as the NL energy diagnostic showing ~9 eV vs the CASTEP
reference of ~2991 eV (per the diagnostic output in ffi.rs lines 506-553).

Fix by: (a) comparing Rust's per-ion D-matrix elements against CASTEP's
`nl_d(m,n,ion,species,spin)` element-by-element for all ions, (b) isolating
whether the sign flip is in beta-projector construction, D-matrix indexing, or
the `exp(±iG·R)` phase factor in `chemrust-hamiltonian-core`.

**Why next**: Can be done in parallel with Goal 1. Even after fixing density
normalization, the answer will be wrong without this.

**Dependencies**: None — independent of Goal 1.

### Goal 3: Enable the inner convergence loop (n_inner > 1) (Medium)

Current `ffi.rs:406`: `let n_inner: usize = 1;` — a single Chebyshev filter
followed by one Rayleigh-Ritz. This cannot converge bands near the filter edge
because one filter pass doesn't fully suppress unwanted eigencomponents.

Implement Zhou et al. Algorithm 5.1 adaptive iteration:
1. Filter + RR → get eigenvalues → update `b_low` from RR eigenvalues
2. Repeat filter+RR with updated spectral bounds
3. Stop when `max|ε_i^(k) - ε_i^(k-1)| < tol` or `k > max_inner` (3-4)
4. Re-enable warm start (`if false` → `if true` at line 560) — use
   H·psi as the input vector for the next filter iteration, which builds
   beta-character through V_NL = beta·D·beta^H

Requires: (a) fixing any correctness bugs that prevent warm start from
producing valid wavefunctions, (b) passing `prev_rr_eig` between iterations
(already partially wired at line 595-623), (c) convergence check on eigenvalue
deltas (Zhou Algorithm 5.1 step 12).

**Why next**: Once correctness is established (Goals 1+2), the eigensolver
needs to actually converge. n_inner=1 is a debugging simplification.

**Dependencies**: Goals 1, 2.

### Goal 4: Eliminate SCF cycle-1 Davidson dependency (Small)

Current `electronic.f90`: SCF cycle 1 uses native CASTEP block Davidson, cycles
>= 2 use Rust Chebyshev. The comment says "build wavefunction character
(beta_phi, augmentation)" — but ABINIT initializes Chebyshev from random vectors
and converges from nothing.

Remove the `scf_cycle <= 1` branch so Rust runs from cycle 1. This requires:
(a) verifying that warm start (Goal 3) or direct Chebyshev from trial
wavefunctions produces correct eigenvalues on cycle 1,
(b) ensuring the S-operator works correctly on the initial trial wavefunctions
(which may have less beta-character than cycle-2 wavefunctions).

If the trial wavefunctions genuinely lack beta-character (USPP augmentation):
either run one H-application to build it, or initialize with random vectors
and let Chebyshev converge.

**Why next**: True GPU residency means the eigensolve never touches Fortran
Davidson. The hybrid approach is a workaround masking initialization bugs.

**Dependencies**: Goals 1, 3.

### Goal 5: Keep V_eff on GPU between SCF steps (Small)

Currently `ffi.rs:330-362` uploads V_eff from host to GPU on every SCF step,
including round-trip verification. In a typical 30-iteration SCF cycle, V_eff
changes slowly after the first few iterations.

Track a hash or max-norm of the current V_eff in the `ChemrustHandle`. On each
`chemrust_eigensolve_step` call: if `||V_eff_new - V_eff_prev||_∞ < threshold`
(threshold ~1e-8 Ha), skip the H2D transfer and reuse the GPU copy. Skip the
round-trip verification as well when reusing.

This eliminates one `ngx*ngy*ngz * sizeof(f64)` PCIe transfer per (kpt, spin)
per SCF iteration — the last non-eigensolve data movement across the boundary.

**Why next**: Pure optimization with zero correctness risk. The only remaining
PCIe transfers are: psi upload (input), eigenvalues + psi + hpsi download
(output). V_eff is the largest single transfer (~33 MB for 128³ grid).

**Dependencies**: None — pure optimization.

## Scope Boundaries

**In scope:**
- Diagnostic instrumentation to isolate the density normalization root cause
- Per-element comparison of D-matrices between Rust and CASTEP `nl_d`
- Adaptive inner convergence loop (Zhou Algorithm 5.1)
- Removal of the SCF cycle-1 Davidson workaround
- V_eff caching on GPU between SCF steps

**Out of scope:**
- MPI gather/scatter for multi-rank operation (deferred — single-rank operation
  is sufficient for current system sizes)
- Gamma-point real optimization (deferred — complex path works for all kpts)
- LDA+U occupancy matrix pass-through (deferred — LDA+U systems are rare in
  the current testing scope)
- NLXC (exact exchange) support — target systems are GGA-PBE only; fall back to Fortran Davidson when nlxc_on
- Per-band convergence tracking (Rust returns `converged=1` unconditionally
  at line 1348 — deferred until after inner loop converges reliably)
- General diagnostic cleanup (the extensive ep-printing in ffi.rs is load-bearing
  for debugging and should remain until goals 1-4 are verified)

## Design Notes

### How n_inner=1 survives 4 cycles

A single Chebyshev filter of degree ~12 amplifies eigencomponents above `b_low`
by factors of `T_12(x_i)`. Bands with `x_i > cos(π/12)` get amplified ~10³ or
more, so one Rayleigh-Ritz after one filter pass captures them well. The problem
is bands near the filter edge (`x_i ≈ 1`) where `T_12(1) = 1` — they get no
amplification and degrade as the subspace rotates toward the filter-window
eigenstates. After ~4 SCF cycles, these edge bands have accumulated enough error
to corrupt the entire subspace.

The fix (Goal 3) is standard: after RR gives improved eigenvalues, recompute
`b_low` from the RR output and run another filter+RR. This is Algorithm 5.1
from the ABINIT paper (Zhou et al.).

### Why not fix convergence first, then correctness?

The architect argued that n_inner > 1 is meaningless with wrong answers. True,
but the reverse is also true: you can't diagnose density normalization or D-matrix
bugs if the filter is exploding after 4 cycles. The order should be:

1. **First**: Fix what's diagnosable without running to cycle 5. D-matrix sign
   (Goal 2) can be verified element-by-element at init time — no SCF loop needed.
   Density normalization (Goal 1) may require only a single filter+RR — compare
   `<psi|psi>` immediately after one H-application.
2. **Second**: Enable inner loop (Goal 3) with correctness fixes in place.
3. **Third**: Remove Davidson fallback (Goal 4) — must converge from cycle 1.

### Density normalization candidates (Goal 1)

The 2.7x factor could come from several places:
- **FFT normalization**: cuFFT C2C inverse multiplies by `1/(ngx*ngy*ngz)` by
  default. If both Rust and CASTEP use cuFFT with the same plan, they agree.
  But if one side uses FFTW (Fortran) and the other cuFFT (Rust), the
  normalization conventions may differ.
- **Density construction convention**: CASTEP's density builder may use
  `ρ(G) = Σ_b Σ_G' conj(ψ_b(G')) * ψ_b(G'+G) / Ω` or without the 1/Ω factor.
  The CONTEXT.md says "inv_omega = 1.0 (no Ω division)" — if CASTEP uses 1/Ω
  in the density builder, that's exactly the discrepancy.
- **Wavefunction normalization**: After S-normalization in ffi.rs:889-935,
  each band satisfies `<ψ_b|S|ψ_b> = 1`. But CASTEP may expect `<ψ_b|ψ_b> = 1`
  (identity norm) or `<ψ_b|S|ψ_b> = 1` inconsistently.
- **Chebyshev T_n amplification**: The normalization factor `1/T_n(x_i)` at
  ffi.rs:838 may over- or under-correct for bands that are superpositions,
  since T_n amplifies each eigencomponent differently.

The diagnostic approach: for one (kpt, spin) after one Chebyshev filter + RR:
1. Dump `<psi_b|psi_b>` (identity norm) and `<psi_b|S|psi_b>` (S-norm) on Rust
2. Upload the same psi to CASTEP's density builder and dump `ρ(G=0)`
3. Compare to CASTEP's own `ρ(G=0)` from the same wavefunctions

### D-matrix sign convention candidates (Goal 2)

Per the failure-patterns note, the `chemrust-hamiltonian-core` V_NL uses a
D-matrix sign convention that matches CASTEP for the origin ion but differs
for non-origin ions. This points to a sign error in `exp(±iG·R)`:
- `beta_g(G, ion) = beta_g(G, origin) * exp(iG·R_ion)` — the projector at
  non-origin ions gains a phase factor
- `D_screened(n,m,ion) = Σ_G Q*(n,G) * V_eff(G) * Q(m,G)` where Q includes
  `exp(±iG·R)` factors from structure-factor multiplication

Likely fix: trace the `exp(iG·R)` sign through `chemrust-hamiltonian-core`'s
beta-projector construction and D-matrix screening, comparing against CASTEP's
`ion_beta_recip_set` + `nlpot_calculate_packed_nl` + `ion_beta_add_multi_recip_all`
convention.

### V_eff caching (Goal 5)

Add to `ChemrustHandle`:
```rust
v_eff_cached: Option<CudaSlice<f64>>,
v_eff_norm: f64,  // max-norm of cached V_eff
```

In `step_inner`:
```rust
let ve_norm: f64 = ve_host.iter().map(|v| v.abs()).fold(0.0, f64::max);
let reuse = h.v_eff_cached.as_ref().is_some_and(|_| (ve_norm - h.v_eff_norm).abs() < 1e-8);
if !reuse {
    // upload V_eff to GPU, update cache
    h.v_eff_cached = Some(v_eff_gpu.clone());  // need Clone for CudaSlice
    h.v_eff_norm = ve_norm;
}
```

If `cudarc::CudaSlice` doesn't implement Clone, store a second copy and `memcpy_dtod`.

## Deferred Items Absorbed

None — this phase addresses bugs discovered during Phase 4 (ABINIT Chebyshev
integration) that were not captured as formal deferred items.

## Domain Terms

**ABINIT Chebyshev recurrence**: The 3-term recurrence `ψ_{k+1} = 2σ(H)·ψ_k - ψ_{k-1}`
with `σ(H) = (H - c·I) / e`, followed by per-band normalization `ψ_b /= T_n(x_b)`.
Contrasts with the older unscaled recurrence used in Phase 4's `phase_a.rs`.

**Inner loop vs outer loop**: The "inner loop" iterates `filter → RR → update_b_low` within
one SCF step (Zhou Algorithm 5.1). The "outer loop" is the SCF cycle. Currently n_inner=1
(no inner convergence), but the Chebyshev recurrence itself runs `ndeg` polynomial iterations.

**S-subspace orthonormality**: After Rayleigh-Ritz, rotated eigenvectors satisfy
`X^H · S_sub · X = I` (the generalized eigenproblem guarantees this). The S-operator is
then implicitly orthonormalized — no Gram-Schmidt needed. The diagnostic at ffi.rs:1038-1114
verifies this holds.

**T_n amplification**: The Chebyshev polynomial `T_n(x)` amplifies eigencomponents with
`x > 1` by `cosh(n·arccosh(x))` ≈ `½·(x + √(x²-1))^n`. Bands with `x ≈ 1` get `T_n ≈ 1`
(no amplification). The normalization `1/T_n(x_i)` restores the original eigencomponent
amplitudes for band `i`, but only exactly if band `i` is a pure eigenstate.

## Verification

1. D-matrix per-element match: `|nl_d(m,n,ion,sp,spin) - D_rust(m,n,ion,sp,spin)| < 1e-12` for all ions, all species
2. Single-filter density normalization: `|ρ_rust(G=0) / ρ_castep(G=0) - 1| < 0.01` for same input wavefunctions
3. `cargo check --workspace` — must pass after each implementation step
4. `cargo clippy --workspace -- -D warnings` — must pass
5. SCF converges to same total energy within 1e-6 Hartree of CASTEP reference for Cu111_CO
6. Cycle-1 eigenvalues match CASTEP eigenvalues within `elec_eigenvalue_tol`
7. V_eff cache hit rate ≥ 60% for a typical 30-iteration SCF cycle
