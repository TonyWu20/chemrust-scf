# TASKS — Phase 0: Davidson Gate 3

**Phase:** Phase 0 of `notes/plans/phase-eigensolver-migration/PHASE_PLAN.md`
**Status:** READY for implementation. Awaits user `continue` after checkpoint review.
**Date:** 2026-05-24
**Predecessor:** `feat/phase-global-woodbury` branch (Chebyshev-RR forensic baseline)
**Decisions:** see `DECISIONS.md` (sibling file)
**ODD pattern reference:** `~/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

## Amendment log

**2026-05-24 (initial-run retrospective).** The original
`gate3_davidson_minimal_locking_preserves_cu3d_block` (Group C below)
executed and produced `GATE3_RESULT.md` (block sum 12.929543, ratio
0.995, **n_locked = 0**). The locking branch never fired: lock_tol =
1e-6 with our (un-converged) V_eff produces residuals ~0.1 Ha, far
above the tolerance. The 0.995 ratio reflects "filter-free single-sweep
ZHEGVD" not "per-band locking works." Block CG would have produced a
similar number for the same reason (no Chebyshev filter). The test
passes the pre-registered 0.97 threshold *bureaucratically* but does
not validate the load-bearing per-band-locking property the gate was
supposed to derisk.

This amendment adds the discriminating tests:

- **Group B'** — `DavidsonDiagnostics.locked_indices: Vec<usize>`
  exposed alongside `n_locked`; `CHEMRUST_DAVIDSON_LOCK_TOL` env-var
  override for per-test tolerance configuration.
- **Group C1** — `gate3_prime_davidson_synthetic_lock_preserves_locked_bands`.
  Synthetic construction forces the Cu-3d cluster into the locked set
  by injecting noise into all other bands; asserts bitwise preservation
  of locked Cu-3d bands. Necessary condition for Davidson v1.
- **Group C2** — `gate3_prime_prime_davidson_stops_cascade_through_scf3`.
  Integrated SCF-3 test with Davidson dispatched; measures whether the
  iter-3 band-0 drift that blew up under Chebyshev-RR (-11.94 Ha) is
  arrested. Sufficient condition for Davidson v1.
- **Group D revision** — `GATE3_RESULT.md` retrospective + revised
  decision based on C1/C2 outcomes.

The original Group C remains in the file for forensic continuity but is
**superseded as a decision gate** — see banner on Group C below.

## Goal

Decide between Phase 1A (Davidson v1) and Phase 1B (block CG) by
running a single-sweep minimal Davidson on Cu111+CO and reading the
Cu-3d block sum (bands 1..14 vs CASTEP, S-weighted).

## Declared fixtures

All groups consume the Cu111+CO CASTEP single-point reference, accessed
through the established loader:

```text
/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/
  Cu111_CO.check        — CASTEP ψ at 160 bands, S-orthonormal under USPP S
  Cu111_CO.castep_bin   — cell, density on wave grid (used by build_scf_state)
  Cu111_CO.pot_fmt      — V_eff on fine grid (used by sibling tests, not Gate 3 directly)
  Cu111_CO.bands        — eigenvalue references (band 0 = -1.05502287 Ha)

/export/Potentials/
  Cu_OTF.usp, C_OTF.usp, O_OTF.usp
```

Loader entry point: `tests/fixtures/cu111_co.rs::fixture()` (cached
via `OnceLock`) and `build_scf_state(fx) -> ScfIteration<S, Initialized,
MixingOff>` (line 158).

Env overrides: `CASTEP_FIXTURE_DIR`, `CASTEP_POTENTIAL_DIR`.

**Path-name note:** PHASE_PLAN.md and CLAUDE.md cite stale path
`Cu111_CO_SinglePoint/`; loader-hardcoded `Cu111_CO_Single_Point_0522_F8`
is authoritative.

## Cu-3d cluster

Bands 1..14 (0-indexed, 13 bands total). Confirmed by:

- `diagnostic_selftest_castep_self_overlap_block_sums`
  (`tests/ca_scf_convergence.rs:3928-3996`) reports 13.000000 exactly
  for CASTEP-vs-CASTEP block.
- `subspace_projector_iter1_vs_castep`
  (`tests/ca_scf_convergence.rs:3522-3535`) uses `block_sum(1, 14)`.

## Dependency map

```
group-dispatch (A) ──┐
                     ├─→ group-davidson-diagnostics (B') ──→ group-gate3-prime (C1) ──┐
group-davidson (B) ──┘                                                                  │
                                                                                        ├─→ group-decision (D)
                                                                  group-gate3-pp (C2) ──┘
                                                                                ▲
                                                                                │
                                       (gated: only run if C1 passes — too expensive otherwise)
```

Original `group-gate3` (Group C) is superseded; kept in this file for
forensic continuity and as the wire-up template for C1/C2. A and B
implement in parallel; B' is small and depends on B's struct;
C1 depends on A + B + B'; C2 depends on C1 passing (gating saves ~500 s
of GPU time on a Davidson-broken run). D depends on C1 + C2.

---

## Group A — Dispatch wire-up (`group-dispatch`)

**Kind:** direct
**Branch:** `impl/phase-eigensolver-migration/group-dispatch`
**Estimated LOC:** ~30
**Dependency:** none

### Context

`src/scf.rs::diagonalize_inner` (lines 484-622) currently hardcodes
the call chain:

```text
chebyshev_filter(...) → rayleigh_ritz(...)
```

Phase 0 needs to optionally divert this to a Davidson single-sweep call
without touching the public `diagonalize()` typestate transition.

### Source-audit instructions

**Before editing,** read:
- `src/scf.rs:444-484` — confirm `diagonalize` → `diagonalize_with_mode`
  → `diagonalize_inner` chain.
- `src/scf.rs:484-622` — note exact location of:
  - GPU/blas/solver/kernels setup (≈ 491-542)
  - V_NL precomputation (≈ 557-569)
  - chebyshev_filter call (≈ 586-593)
  - rayleigh_ritz call (≈ 597-604)
  - psi sync to host (≈ 612)
- `src/eigensolver/rayleigh_ritz.rs:36, 64-81` — `RrPinConfig::from_env`
  pattern. Mirror the env-var read style.

### Change

Read `CHEMRUST_EIGENSOLVER` once, just before the chebyshev_filter call
site (after V_NL precomputation, after kinetic_dev upload). Default
`"chebyshev"`. Branch:

- `"davidson"` → call new `davidson_minimal::davidson_minimal_single_sweep`
  (defined by Group B). Skip `chebyshev_filter` and `rayleigh_ritz`.
  The Davidson result struct exposes `psi_out` (ColumnDistributed
  CudaSlice) and `eigenvalues_cpu`; convert to the same return tuple
  the existing path produces (psi_new_gpu, eigenvalues_cpu,
  beta_psi_gpu_or_None).
- `"chebyshev"` (or any other value) → existing path verbatim.

**Critical:** `ndeg` parameter is ignored in davidson branch. Add a
single-line comment at the davidson branch:

```rust
// ndeg ignored — davidson_minimal is single-sweep (Phase 0 scratch test).
```

**`beta_psi_gpu` consequence:** the chebyshev path returns
`beta_psi_gpu` (β·ψ projector overlaps) for downstream density
construction. Davidson's single-sweep does not produce this. Two
options — pick whichever is simpler at implementation time:

1. Recompute β·ψ via a dedicated cuBLAS gemm call after Davidson
   returns (β_g^H · psi_out → C_proj). The β_g matrices live in
   `vnl_data.entries`. ~10 LOC.
2. Pass `None` and let downstream code re-derive (check whether the
   `WavefunctionsUpdated` typestate accepts `Option<beta_psi>` —
   audit before deciding).

**Default to (1)** unless audit shows (2) is free.

### Files

| File | Edit | Lines |
|---|---|---|
| `src/scf.rs` | modify `diagonalize_inner` | ~+25 lines around 580-610 |

### Success criteria

| Criterion | Source | Verification |
|---|---|---|
| `cargo check --workspace` passes | CONTEXT.md tooling | `cargo check --workspace` |
| `cargo clippy --workspace -- -D warnings` passes | CONTEXT.md tooling | `cargo clippy` |
| Existing tests pass with no env var set (Chebyshev default preserved) | DECISIONS.md A3 | `CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 cargo test --release --features scf_diag subspace_projector_iter1_vs_castep -- --ignored --nocapture` reports Cu-3d/13 ≈ 0.893 unchanged |
| Davidson branch unreachable when env var unset | dispatch correctness | grep ensures no test runs Davidson branch unless explicit env override |

### Commit

`feat(eigensolver): wire CHEMRUST_EIGENSOLVER env-var dispatch in diagonalize_inner`

---

## Group B — Minimal Davidson single-sweep (`group-davidson`)

**Kind:** lib-tdd (ODD)
**Branch:** `impl/phase-eigensolver-migration/group-davidson`
**Estimated LOC:** ~120
**Dependency:** none (independent of Group A)

### Context

Implements the Phase 0 scratch algorithm: one residual computation,
one lock-list construction, one ZHEGVD on the unconverged sub-block,
one S-orthogonalization pass. No outer iteration, no preconditioner.

### Source-audit instructions (READ before writing code)

These are not decoration. The algorithm description below mirrors
what the source actually does; the implementation must reuse, not
reimplement, these primitives.

| Primitive | Path | Lines | What to verify by reading |
|---|---|---|---|
| `apply_full_hamiltonian` | `src/eigensolver/chebyshev.rs` | 707-739 | Signature; that it composes T+V_loc via FFT then V_NL via cuBLAS; that `hpsi_dev` is pre-allocated by caller |
| `apply_s_times` | `src/eigensolver/chebyshev.rs` | 934-1014 | **Comment at line 936: "caller pre-copies psi into this".** Identity term `Sψ = ψ + Σ_ion β·Q·β^H·ψ` requires the pre-copy or it silently drops. Three gemm cycle: p = β^H·ψ, q = Q·p, spsi += β·q (β-accumulator = +1 at line 1004) |
| Rayleigh-Ritz H_sub/S_sub assembly | `src/eigensolver/rayleigh_ritz.rs` | 181-310 | The exact gemm pattern for H_sub = ψ^†·Hψ (line 193-211, transa=op::C, transb=op::N), S_sub = ψ^†·ψ + Σ_ion C_proj^†·Q·C_proj (line 213-310). C_proj = β_g^H·ψ at line 245-266. β-accumulator on the augmentation gemm is +1 (line 303) |
| Gram-Schmidt 2-pass | `src/eigensolver/chebyshev.rs` | 1708-1799 | Per-band cuBLAS pattern: `apply_s_times` to get S·col_b (line 1736), `cublasZdotc` for inner products (line 1770), `cublasZaxpy` for subtraction (line 1778), `cublasDznrm2` + `cublasZscal` for normalization (line 1791). Phase 0 takes a single-pass restricted version (locked-vs-unconverged only) |
| Generalized eigensolve | `src/eigensolver/rayleigh_ritz.rs` | (search for `zhegvd` or `faer_la::Hermitian`) | Exact API: H_sub, S_sub layout; eigenvalue/eigenvector output convention. Phase 0's sub-block eigensolve must match the layout convention the existing code uses |

If any of the above signatures or behaviors differ from this table at
audit time, **stop and report**: the table comes from a one-shot read
during planning, not a tested compile.

### Algorithm — single sweep, ColumnDistributed in/out

Inputs: `psi_in` (ColumnDistributed CudaSlice, n_bands × n_pw col-major
flat), `v_eff_dev`, `kinetic_dev`, `fft_idx_dev`, `vnl_data`, sundry
GPU handles.

```text
1. Allocate workspaces:
   hpsi_dev    : CudaSlice<CudaComplex>, n_bands * n_pw, zeros
   spsi_dev    : CudaSlice<CudaComplex>, n_bands * n_pw, zeros
   grid_dev    : CudaSlice<CudaComplex>, n_bands * grid_size, zeros (FFT scratch)

2. Hψ:
   apply_full_hamiltonian(
     psi_in, v_eff_dev, kinetic_dev, fft_idx_dev,
     n_pw, n_bands, grid_size, inv_ntotal, fft_plan,
     &mut hpsi_dev, &mut grid_dev, vnl_data,
     blas, kernels, stream)?;

3. Sψ:
   // CRITICAL: pre-copy ψ into spsi_dev for the identity term.
   stream.memcpy_dtod(psi_in, &mut spsi_dev)?;
   apply_s_times(psi_in, &mut spsi_dev, vnl_data, n_bands, n_pw, blas, stream)?;

4. Per-band Rayleigh quotient λ_b = Re⟨ψ_b | Hψ_b⟩ / Re⟨ψ_b | Sψ_b⟩:
   for b in 0..n_bands:
     lambda_h = cublasZdotc(psi_in[b*n_pw..(b+1)*n_pw], hpsi_dev[b*n_pw..]).x
     lambda_s = cublasZdotc(psi_in[b*n_pw..(b+1)*n_pw], spsi_dev[b*n_pw..]).x
     lambdas[b] = lambda_h / lambda_s
   (Imaginary parts of these inner products should be ~ULP; assert in debug.)

5. Per-band residual r_b = Hψ_b − λ_b · Sψ_b on GPU:
   residual_dev := hpsi_dev.clone()       // OR allocate fresh and memcpy
   for b in 0..n_bands:
     alpha = -lambdas[b]
     cublasZaxpy(alpha, &spsi_dev[b*n_pw..], &mut residual_dev[b*n_pw..])

6. Per-band L2 norm:
   for b in 0..n_bands:
     residual_norms[b] = cublasDznrm2(&residual_dev[b*n_pw..])

7. Lock list:
   locked: Vec<bool> = residual_norms.iter().map(|r| *r < lock_tol).collect();
   unconv_idx: Vec<usize> = (0..n_bands).filter(|b| !locked[*b]).collect();
   k = unconv_idx.len();

8. EARLY RETURN if k == 0:
   // All bands self-consistent — typical when V_eff is CASTEP-pinned.
   return DavidsonResult {
     psi_out: psi_in.clone_to_owned(),
     eigenvalues: lambdas,
     n_locked: n_bands, n_unconverged: 0, residual_norms,
   };

9. Sub-block ZHEGVD on unconverged (k×k):
   a. Gather: build psi_unconv (n_pw × k) and hpsi_unconv (n_pw × k) on GPU
      by copying the relevant column slices from psi_in / hpsi_dev
      (each column is contiguous in ColumnDistributed: psi[b*n_pw..(b+1)*n_pw]).
   b. Build H_sub_k (k×k) and S_sub_k (k×k) using the SAME gemm pattern as
      rayleigh_ritz.rs:181-310, restricted to k bands:
        H_sub_k = psi_unconv^† · hpsi_unconv          (transa=C, transb=N)
        S_sub_k = psi_unconv^† · psi_unconv           (bare PW)
        FOR EACH ion entry in vnl_data.entries:
          C_proj_unconv = beta_g^H · psi_unconv       (n_expanded × k)
          temp = q_matrix · C_proj_unconv             (n_expanded × k)
          S_sub_k += C_proj_unconv^† · temp           (k×k accumulator)
   c. D2H: copy H_sub_k, S_sub_k to host (k² complex doubles each).
   d. Solve generalized Hermitian eigenproblem on host with the SAME
      facility rayleigh_ritz uses (faer or cuSOLVER zhegvd — match the
      audit). Returns:
        lambda_unconv: Vec<f64> of length k (sorted ascending)
        x_sub: Mat<Complex<f64>> of shape (k, k), eigenvectors as columns
   e. H2D: upload x_sub to a CudaSlice (k × k col-major).

10. Rotate unconverged bands:
    psi_unconv_new = psi_unconv · x_sub_dev          // (n_pw × k)
    via cublasZgemm(transa=N, transb=N, m=n_pw, n=k, k=k, alpha=1, beta=0).

11. Single-pass S-orthogonalize unconverged-against-locked:
    // X_sub already gave unconverged-among-themselves S-orthonormality.
    // We need to remove components along the locked subspace.
    // Pattern: for each unconverged column, subtract its locked-projector overlap.
    for u in 0..k:
      // Compute S·psi_unconv_new[u] in-place
      memcpy_dtod(psi_unconv_new[u], &mut s_col_dev)
      apply_s_times(psi_unconv_new[u..u+1], &mut s_col_dev, vnl_data, 1, n_pw, blas, stream)?;
      for j in locked_indices:
        // dot = ⟨psi_locked[j] | s_col_dev⟩
        dot = cublasZdotc(psi_in[j*n_pw..], &s_col_dev)
        // psi_unconv_new[u] -= dot * psi_in[j]
        cublasZaxpy(-dot, &psi_in[j*n_pw..(j+1)*n_pw], &mut psi_unconv_new[u*n_pw..])

12. Concatenate output:
    psi_out: CudaSlice = allocated zeros, n_bands * n_pw
    for b in 0..n_bands:
      if locked[b]:
        memcpy_dtod(&psi_in[b*n_pw..(b+1)*n_pw], &mut psi_out[b*n_pw..(b+1)*n_pw])
        eigenvalues_out[b] = lambdas[b]
      else:
        u = position of b in unconv_idx
        memcpy_dtod(&psi_unconv_new[u*n_pw..(u+1)*n_pw], &mut psi_out[b*n_pw..(b+1)*n_pw])
        eigenvalues_out[b] = lambda_unconv[u]

13. Return DavidsonResult { psi_out, eigenvalues: eigenvalues_out,
    n_locked, n_unconverged: k, residual_norms }.
```

### File creation

```rust
// src/eigensolver/davidson_minimal.rs

use std::sync::Arc;
use cudarc::{cublas::CudaBlas, driver::*};
use crate::{...};

pub(crate) struct DavidsonResult {
    pub psi_out: CudaSlice<CudaComplex>,
    pub eigenvalues: Vec<f64>,
    pub n_locked: usize,
    pub n_unconverged: usize,
    pub residual_norms: Vec<f64>,
}

/// Phase 0 minimal Davidson — single sweep, no preconditioner, no
/// outer iteration. Returns rotated unconverged bands and untouched
/// locked bands. See `notes/plans/phase-eigensolver-migration/TASKS.md`
/// for the algorithm spec.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn davidson_minimal_single_sweep(
    psi_in: &CudaSlice<CudaComplex>,
    v_eff_dev: &CudaSlice<f64>,
    kinetic_dev: &CudaSlice<f64>,
    fft_idx_dev: &CudaSlice<i32>,
    vnl_data: &VnlBatchData,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    lock_tol: f64,
    blas: &BlasHandle,
    solver: &SolverHandle,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
) -> Result<DavidsonResult, Error> {
    // Steps 1-13 per algorithm spec above.
    todo!()
}
```

```rust
// src/eigensolver/mod.rs (existing file, add line)
pub(crate) mod davidson_minimal;
```

### Success criteria

| Criterion | Source | Verification |
|---|---|---|
| `cargo check --workspace` passes | CONTEXT.md tooling | `cargo check --workspace` |
| `cargo clippy --workspace -- -D warnings` passes | CONTEXT.md tooling | `cargo clippy` |
| Self-consistency unit test: pinned-V_eff Davidson reproduces CASTEP ψ to ≤ 1e-12 | DECISIONS.md secondary criteria | Test below |
| Trivial early-return path: when all residuals < lock_tol, output ≡ input bitwise (no ZHEGVD) | algorithm step 8 | Test below |

### Test (lib-tdd ODD anchor)

`tests/davidson_minimal_validation.rs` (NEW file, ~80 LOC):

```rust
//! Davidson-minimal scratch test: self-consistency with pinned CASTEP V_eff.
//!
//! When V_eff is pinned to CASTEP's, our H ≡ CASTEP's H exactly.
//! CASTEP ψ are exact eigenvectors of H, so all residuals = 0,
//! all bands lock, sub-block ZHEGVD never runs, output ≡ input.
//!
//! This test verifies the Davidson plumbing is wired correctly. It is
//! NOT the Gate 3 test (which uses our V_eff and exposes the locking
//! discrimination). See tests/ca_scf_convergence.rs::gate3_*.

#[cfg(feature = "scf_diag")]
mod tests {
    use chemrust_scf::*;
    // ... fixture loader imports ...

    #[test]
    #[ignore]
    fn davidson_minimal_self_consistency_with_pinned_castep_veff() {
        // SAFETY: env vars are tested with serial_test or scopeguard.
        unsafe { std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson"); }
        let _guard = scopeguard::guard((), |_| {
            unsafe { std::env::remove_var("CHEMRUST_EIGENSOLVER"); }
        });

        let fx = fixtures::cu111_co::fixture();
        let psi_castep_in = fixtures::cu111_co::castep_psi_first_kpoint(fx);
        // Build state with V_eff PINNED to CASTEP's (not built fresh).
        let veff_state = fixtures::cu111_co::build_state_with_castep_veff(fx)
            .expect("fixture state with CASTEP V_eff");

        let diag = veff_state.diagonalize(0, None).expect("davidson diag");
        let psi_out = diag.psi_data();

        // Compare ψ_out vs ψ_castep_in band-by-band, plain L2.
        for b in 0..160 {
            let band_in = &psi_castep_in[b*n_pw..(b+1)*n_pw];
            let band_out = &psi_out[b*n_pw..(b+1)*n_pw];
            let diff_norm: f64 = band_in.iter().zip(band_out.iter())
                .map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>().sqrt();
            assert!(diff_norm < 1e-12,
                "band {b}: ‖ψ_out − ψ_castep_in‖₂ = {diff_norm:.3e}, want ≤ 1e-12");
        }
    }
}
```

**`build_state_with_castep_veff` may not yet exist as a fixture
helper**; if so, this is a sub-task: add a thin helper to
`tests/fixtures/cu111_co.rs` that pins V_eff from `fx.pot_fmt`-derived
data into the `ScfIteration<VEffBuilt>` state. Audit during implementation:
search for existing test patterns that pin V_eff (sibling test
`cascade_with_castep_anchored_postrr_pin` may already do this).

### Commit

`feat(eigensolver): minimal Davidson single-sweep for Phase 0 Gate 3`

---

## Group C — Gate 3 test (`group-gate3`) — SUPERSEDED

> **AMENDED 2026-05-24:** This test ran and produced
> `GATE3_RESULT.md`, but `n_locked = 0` made it a "filter-free
> single-sweep" test rather than a "locking" test. Kept in this file
> as the wire-up template for C1/C2 (env-var dispatch, fixture
> loading, block-sum machinery). The decision-gate role passes to
> Group C1.

**Kind:** lib-tdd (ODD)
**Branch:** `impl/phase-eigensolver-migration/group-gate3`
**Estimated LOC:** ~80
**Dependency:** Group A merged + Group B merged

### Context

The decisive test. Loads CASTEP ψ, builds **our** V_eff (per
DECISIONS.md A4), calls Davidson-minimal via env-var dispatch, computes
Cu-3d block sum against CASTEP ψ, prints the Decision text.

### Source-audit instructions

**Before editing,** read:
- `tests/ca_scf_convergence.rs:83-107` (`fixed_point_matches_castep_energy`) — template for `build_scf_state` + `build_v_eff` + `diagonalize` flow.
- `tests/ca_scf_convergence.rs:3475-3573` (`subspace_projector_iter1_vs_castep`) — exact Cu-3d block-sum computation and the reference 11.6/13.0 baseline. **Mirror the block_sum closure verbatim** so the comparison is apples-to-apples.
- `tests/ca_scf_convergence.rs:3522-3535` — the `block_sum(1, 14)` call itself.

### Test definition

`tests/ca_scf_convergence.rs` (append, alongside sibling diagnostics):

```rust
#[test]
#[ignore]
#[cfg(feature = "scf_diag")]
fn gate3_davidson_minimal_locking_preserves_cu3d_block() {
    // Phase 0 Gate 3 (PHASE_PLAN.md lines 58-72): does per-band locking
    // preserve the Cu-3d block at 13.0 where Chebyshev-RR's ZHEGVD-rotation
    // forces it to 11.6?
    //
    // CHEMRUST_EIGENSOLVER=davidson dispatches to davidson_minimal::single_sweep
    // (Group B). V_eff is OUR V_eff, not CASTEP-pinned (DECISIONS.md A4):
    // pinning V_eff trivializes the test (all residuals = 0 → no ZHEGVD).
    //
    // No hard assertion: this is a decision gate, not a correctness gate.
    // The Decision: log line drives the next phase choice.

    unsafe { std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson"); }
    let _guard = scopeguard::guard((), |_| {
        unsafe { std::env::remove_var("CHEMRUST_EIGENSOLVER"); }
    });

    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);

    let veff_state = state.build_v_eff().expect("build_v_eff");
    let diag = veff_state.diagonalize(0, None).expect("diagonalize davidson");
    let psi_out = diag.psi_data();

    let psi_castep = fixtures::cu111_co::castep_psi_first_kpoint(fx);
    let n_pw = fx.n_pw_first_kpoint();
    let vnl_data = /* obtain from diag or rebuild — audit during implementation */;

    // S-augmented inner product ⟨ψ_out_a | S | ψ_castep_b⟩, verbatim from
    // subspace_projector_iter1_vs_castep at lines 3522-3535.
    let block_sum = |i0: usize, i1: usize| -> f64 {
        let mut total = 0.0_f64;
        for a in i0..i1 {
            for b in i0..i1 {
                let s_ab = compute_s_inner_product(
                    &psi_out[a*n_pw..(a+1)*n_pw],
                    &psi_castep[b*n_pw..(b+1)*n_pw],
                    &vnl_data,
                );
                total += s_ab.norm_sqr();
            }
        }
        total
    };

    let s_band0 = block_sum(0, 1);
    let s_cu3d = block_sum(1, 14);   // 13 bands, Cu-3d cluster
    let s_30 = block_sum(0, 30);
    let s_40 = block_sum(0, 40);

    let ratio = s_cu3d / 13.0;
    let decision = if s_cu3d >= 12.999 {
        "PASS — locking sufficient. Proceed Phase 1A (Davidson v1)."
    } else if s_cu3d <= 11.700 {
        "FAIL — locking insufficient. Fall back Phase 1B (block CG)."
    } else {
        "MIXED — locking helps but incomplete. Lean Davidson; re-evaluate Phase 2."
    };

    println!("[Gate 3] Cu-3d block sum (bands 1..14 vs CASTEP): {s_cu3d:.6}");
    println!("[Gate 3] Self-overlap reference (CASTEP vs CASTEP): 13.000000");
    println!("[Gate 3] Chebyshev-RR baseline (recorded): 11.610000 (ratio 0.893)");
    println!("[Gate 3] Davidson ratio: {ratio:.6}");
    println!("[Gate 3] Sibling block sums: band0={s_band0:.6}, 0..30={s_30:.6}, 0..40={s_40:.6}");
    println!("[Gate 3] Locked bands: {} / 160", /* extract from DavidsonResult — audit */);
    println!("[Gate 3] Max residual: {:.3e}", /* from DavidsonResult */);
    println!("[Gate 3] Decision: {decision}");
}
```

**Audit during implementation:** the `vnl_data` access from a
post-diagonalize state, and the `n_locked`/`max_residual` extraction
from `DavidsonResult`. Group B's signature returns `DavidsonResult` to
the dispatch in Group A — Group A must propagate enough of it for this
test to read. If propagation is awkward, fall back to environment
variable diagnostics (Davidson writes
`CHEMRUST_LAST_DAVIDSON_LOCKED=N` to a thread-local or stderr;
inelegant but Phase-0-scratch-acceptable).

### Success criteria

| Criterion | Source | Verification |
|---|---|---|
| Test compiles and runs without panic | bare correctness | `cargo test --release --features scf_diag gate3_davidson_minimal_locking_preserves_cu3d_block -- --ignored --nocapture` |
| Cu-3d block sum is a finite f64 in [0, 14] | sanity | `assert!(s_cu3d.is_finite() && (0.0..=14.0).contains(&s_cu3d))` (the only hard assertion in the test) |
| One of the three Decision branches prints | logic completeness | inspect captured stdout |
| Existing `subspace_projector_iter1_vs_castep` still reports 11.6 ratio 0.893 | no regression | run sibling test with no env var, verify unchanged |

### Commit

`test(scf): Phase 0 Gate 3 — Davidson-minimal Cu-3d block-sum probe`

---

## Group B' — Davidson diagnostics surface (`group-davidson-diagnostics`)

**Kind:** direct
**Branch:** `impl/phase-eigensolver-migration/group-davidson-diagnostics`
**Estimated LOC:** ~40
**Dependency:** Group B merged

### Context

Group B's `DavidsonResult` exposed `n_locked` and `residual_norms`. C1
and C2 need additional surfaces:

- `locked_indices: Vec<usize>` — explicit which-bands-locked list
  (currently derivable from `residual_norms < lock_tol` but recomputing
  in tests duplicates the source of truth and risks drift).
- `lock_tol` overridable per call via env var
  `CHEMRUST_DAVIDSON_LOCK_TOL` (defaults to 1e-6 = current hardcoded
  value). C1 needs `lock_tol = 1e-3`; C2 may sweep.
- A way for the post-`diagonalize` test code to read these fields.
  `diag.davidson_diagnostics() -> Option<&DavidsonDiagnostics>` on the
  `WavefunctionsUpdated` typestate; `Some` only when Davidson ran,
  `None` for the chebyshev path.

### Source-audit instructions

**Before editing,** read:
- The `DavidsonResult` definition produced by Group B
  (`src/eigensolver/davidson_minimal.rs`).
- `src/scf.rs` `WavefunctionsUpdated` impl block (~ line 1700+) —
  the typestate where `psi_data()` lives. Add a sibling diagnostic
  accessor.

### Change

1. Rename `DavidsonResult` field surface or add a wrapper:

```rust
pub(crate) struct DavidsonDiagnostics {
    pub n_locked: usize,
    pub n_unconverged: usize,
    pub locked_indices: Vec<usize>,        // NEW
    pub unconv_indices: Vec<usize>,        // NEW
    pub residual_norms: Vec<f64>,
    pub max_residual: f64,                 // NEW (mean optional)
    pub lock_tol: f64,                     // record actual value used
}
```

`DavidsonResult { psi_out, eigenvalues, diagnostics: DavidsonDiagnostics }`.

2. In `davidson_minimal_single_sweep`, read `lock_tol` from arg as
   today, but the **caller** (Group A's dispatch) reads from env:

```rust
let lock_tol = std::env::var("CHEMRUST_DAVIDSON_LOCK_TOL")
    .ok()
    .and_then(|s| s.parse::<f64>().ok())
    .unwrap_or(1e-6);
```

Record the value used into `DavidsonDiagnostics.lock_tol`.

3. Stash `DavidsonDiagnostics` into the `ScfIteration<…,
   WavefunctionsUpdated, …>` so the test can retrieve it after
   `diagonalize` returns. Add field
   `last_davidson_diagnostics: Option<DavidsonDiagnostics>` (or
   equivalent) to whatever struct backs `WavefunctionsUpdated`.
   Chebyshev path leaves this `None`.

4. Public accessor:

```rust
impl<...> ScfIteration<S, WavefunctionsUpdated, M> {
    pub fn davidson_diagnostics(&self) -> Option<&DavidsonDiagnostics> {
        self.last_davidson_diagnostics.as_ref()
    }
}
```

`#[cfg(any(test, feature = "scf_diag"))]` on the accessor — matches
the cfg-boundary discipline from failure-pattern
`rr-validation-infrastructure-cfg-boundary` (failure-patterns.md).

### Files

| File | Edit |
|---|---|
| `src/eigensolver/davidson_minimal.rs` | refactor `DavidsonResult` to embed `DavidsonDiagnostics`; populate `locked_indices`, `unconv_indices`, `max_residual`, `lock_tol` |
| `src/scf.rs` | Group A dispatch reads `CHEMRUST_DAVIDSON_LOCK_TOL`; pass through; stash diagnostics on `WavefunctionsUpdated`; add accessor |

### Success criteria

| Criterion | Verification |
|---|---|
| `cargo check --workspace` passes | tooling |
| `cargo clippy --workspace -- -D warnings` passes | tooling |
| Group B's `davidson_minimal_self_consistency_with_pinned_castep_veff` test, updated to read `diag.davidson_diagnostics()`, sees `Some` with `n_locked == 160`, `locked_indices == 0..160`, `lock_tol == 1e-6` | retrofit Group B's test |
| With `CHEMRUST_DAVIDSON_LOCK_TOL=0.5` set, the existing `gate3_davidson_minimal_locking_preserves_cu3d_block` (Group C, superseded) sees `n_locked > 0` | smoke test the env override |
| Chebyshev path (no env var) leaves `davidson_diagnostics()` returning `None` | dispatch isolation |

### Commit

`feat(eigensolver): expose Davidson diagnostics + lock_tol env override for Phase 0 tests`

---

## Group C1 — Synthetic-lock identity preservation (`group-gate3-prime`)

**Kind:** lib-tdd (ODD)
**Branch:** `impl/phase-eigensolver-migration/group-gate3-prime`
**Estimated LOC:** ~120
**Dependency:** Group A + Group B + Group B' merged

### Context

The decisive gate (replaces superseded Group C). Forces the locking
branch to fire by construction: the Cu-3d cluster is hand-placed in
the locked set, all other bands are perturbed into the unconverged
set. Asserts that locked bands come out **bitwise identical** to
input.

This is the necessary condition for Davidson v1: if the algorithm's
locking branch rotates the locked Cu-3d bands, Davidson cannot fix
the cascade regardless of preconditioner or outer iteration design.

### Construction (from grill design discussion)

```text
1. Load fixture: psi_castep = fx.castep_psi_first_kpoint()  (160 bands)
2. Pin V_eff to CASTEP V_eff (NOT our V_eff — we want CASTEP ψ to be
   exact eigenvectors of H so that residuals on UNTOUCHED Cu-3d bands
   are < 1e-12, securely below lock_tol = 1e-3)
3. Inject noise into non-Cu-3d bands:
     for b in 0..160 where b ∉ 1..14:
         eta_b = random complex vector, ‖eta_b‖₂ = epsilon  (default 0.01)
         psi_perturbed[b] = psi_castep[b] + eta_b
     for b in 1..14:
         psi_perturbed[b] = psi_castep[b]            // bitwise identical
4. Re-S-orthonormalize perturbed bands AGAINST Cu-3d only:
     for b in 0..160 where b ∉ 1..14:
         for j in 1..14:
             dot = ⟨ψ_castep[j] | S | psi_perturbed[b]⟩
             psi_perturbed[b] -= dot · ψ_castep[j]
         normalize psi_perturbed[b] by √⟨psi_perturbed[b]|S|psi_perturbed[b]⟩
     // Cu-3d bands UNTOUCHED in this loop — they remain bitwise == psi_castep
5. Configure: CHEMRUST_EIGENSOLVER=davidson, CHEMRUST_DAVIDSON_LOCK_TOL=1e-3
6. Run davidson_minimal_single_sweep via diagonalize(0, None)
7. Assertions:
     a. dr.n_locked == 13
     b. dr.locked_indices == [1, 2, ..., 13]
     c. For b in 1..14, for g in 0..n_pw:
            ψ_out[b][g] == psi_perturbed[b][g]   // bitwise (==, not approx)
     d. Cu-3d block sum (vs psi_castep) ∈ [13.0 - 1e-7, 13.0 + 1e-7]
     e. Sanity: Cu-3d residuals (computed externally on psi_perturbed) < 1e-10
```

### Why CASTEP V_eff in step 2 (vs. Group C's "use our V_eff")

In Group C, "our V_eff" was load-bearing because it created the
discriminating residual structure. Here, the discriminator is the
**construction**, not the residual structure. We need:

- Cu-3d residuals to be deeply locked (< 1e-12, way under 1e-3)
- Non-Cu-3d residuals to be above lock_tol = 1e-3

CASTEP V_eff makes Cu-3d residuals on `psi_castep` (which we left
untouched in step 3) exactly zero modulo floating-point. Our V_eff
would make Cu-3d residuals ~ 0.1 Ha, putting Cu-3d in the unconverged
set — which is the opposite of what the test needs.

This is *not* the same trivialization-by-V_eff-pin that Group C had to
avoid. Group C asked "does locking save the cascade?" Group C1 asks
"does locking preserve locked bands?" — these are different
questions. Group C2 below restores the integrated cascade question
with our V_eff.

### Why epsilon = 0.01

- **Above lock_tol = 1e-3**: 0.01 perturbation on a unit-S-norm USPP
  wavefunction produces residual ~ 0.01 × |λ_b| ≈ 0.001-0.05 Ha for
  occupied bands. Sample: lowest-band λ ≈ -1.05 Ha → residual ~ 0.01
  Ha; midband λ ≈ -0.1 Ha → residual ~ 0.001 Ha (borderline); plenty
  of margin for the buffer bands above the Fermi level (residual
  scales with the larger of |λ_b| and |λ_perturbation|, the latter
  being ~ε).
- **Below rotation amplitude**: small enough that the perturbed-band
  subspace is still close to the eigenvector basis — ZHEGVD's output
  is a small-rotation correction, not a noise-driven diagonalization
  on garbage.

Parameterize via `CHEMRUST_GATE3_PERTURB_EPS=0.01` env override to
sweep at run-time if the default chooses badly.

### Why re-S-orthonormalize unconverged-against-locked only

Full 160×160 S-Gram-Schmidt would back-react on the Cu-3d bands to
maintain global S-orthonormality, breaking the bitwise-identity check
before Davidson runs. Restricting GS to "perturbed-vs-Cu-3d,
perturbed-vs-other-perturbed" leaves Cu-3d coefficients untouched and
produces input that satisfies Davidson's expected pre-condition (the
unconverged set is S-orthogonal to the locked set).

### Source-audit instructions

**Before editing,** read:
- `tests/ca_scf_convergence.rs` Group C body (the superseded test) —
  reuse the env-var dispatch + scopeguard pattern, the CASTEP-V_eff
  pinning, and the Cu-3d block-sum closure.
- `tests/ca_scf_convergence.rs:4138-4218`
  (`cascade_with_castep_anchored_postrr_pin`) — for the
  `psi_data_mut()` injection pattern used to swap in custom ψ.
- Failure-pattern `uspp-pw-norm-not-unit-ncpp-assumption`
  (failure-patterns.md) — confirms ‖ψ‖²_PW for Cu 3d is not 1.0
  under USPP; do NOT use plain PW norm for the noise injection,
  use S-norm via `apply_s_times` + `cublasZdotc`.

### Test snippet

`tests/ca_scf_convergence.rs` (append):

```rust
#[test]
#[ignore]
#[cfg(feature = "scf_diag")]
fn gate3_prime_davidson_synthetic_lock_preserves_locked_bands() {
    // Phase 0 Gate 3' — synthetic-lock identity preservation. Forces Cu-3d
    // into the locked set by construction; asserts bitwise preservation.
    // See notes/plans/phase-eigensolver-migration/TASKS.md Group C1.

    unsafe {
        std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson");
        std::env::set_var("CHEMRUST_DAVIDSON_LOCK_TOL", "1e-3");
    }
    let _guard = scopeguard::guard((), |_| unsafe {
        std::env::remove_var("CHEMRUST_EIGENSOLVER");
        std::env::remove_var("CHEMRUST_DAVIDSON_LOCK_TOL");
    });

    let fx = fixtures::cu111_co::fixture();
    let psi_castep = fixtures::cu111_co::castep_psi_first_kpoint(fx);
    let n_pw = fx.n_pw_first_kpoint();
    let n_bands: usize = 160;

    let cu3d = 1..14usize;
    let epsilon: f64 = std::env::var("CHEMRUST_GATE3_PERTURB_EPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.01);

    // Steps 3-4: construct psi_perturbed
    let psi_perturbed = construct_synthetic_locked_input(
        psi_castep, n_pw, n_bands, cu3d.clone(), epsilon, /*seed=*/42, &fx,
    );
    // construct_synthetic_locked_input internally:
    //   - clones psi_castep into psi_perturbed
    //   - for b ∉ cu3d: adds RNG-seeded complex noise of S-norm ε
    //   - for b ∉ cu3d: subtracts S-projection onto each j ∈ cu3d (using
    //     apply_s_times for the inner product), normalizes by S-norm
    //   - returns psi_perturbed; bands in cu3d are bitwise == psi_castep

    // Sanity: Cu-3d bands bitwise unchanged in input
    for b in cu3d.clone() {
        assert_eq!(
            &psi_perturbed[b * n_pw..(b + 1) * n_pw],
            &psi_castep[b * n_pw..(b + 1) * n_pw],
            "construct_synthetic_locked_input modified Cu-3d band {b}"
        );
    }

    // Step 2 + 5-6: pin V_eff to CASTEP, inject psi_perturbed, run Davidson
    let veff_state = fixtures::cu111_co::build_state_with_castep_veff_and_psi(
        fx, &psi_perturbed,
    ).expect("pinned state");
    let diag = veff_state.diagonalize(0, None).expect("davidson diag");
    let dr = diag.davidson_diagnostics().expect("davidson diagnostics set");
    let psi_out = diag.psi_data();

    // Step 7: assertions
    assert_eq!(dr.n_locked, 13,
        "expected 13 locked bands (Cu-3d cluster), got {}", dr.n_locked);
    let expected_locked: Vec<usize> = cu3d.clone().collect();
    assert_eq!(dr.locked_indices, expected_locked,
        "locked set should be exactly Cu-3d");
    for &b in &expected_locked {
        let band_in = &psi_perturbed[b * n_pw..(b + 1) * n_pw];
        let band_out = &psi_out[b * n_pw..(b + 1) * n_pw];
        for g in 0..n_pw {
            assert_eq!(band_out[g], band_in[g],
                "band {b} G {g}: locked band rotated. \
                 in = {:?}, out = {:?}", band_in[g], band_out[g]);
        }
    }
    let cu3d_sum = compute_s_block_sum(
        psi_out, psi_castep, n_pw, cu3d.clone(), &fx,
    );
    assert!(
        (cu3d_sum - 13.0).abs() < 1e-7,
        "Cu-3d block sum = {cu3d_sum:.10}, want 13.0 ± 1e-7"
    );

    println!("[Gate 3'] PASS — locking preserves Cu-3d block bitwise");
    println!("[Gate 3'] n_locked = {} / 160", dr.n_locked);
    println!("[Gate 3'] Cu-3d block sum = {cu3d_sum:.10} (target 13.0)");
    println!("[Gate 3'] max residual on unconverged = {:.3e} Ha", dr.max_residual);
    println!("[Gate 3'] perturbation epsilon = {epsilon}");
}
```

### Helper functions to add

In `tests/fixtures/cu111_co.rs` (or a new
`tests/fixtures/davidson_synthetic.rs`):

- `construct_synthetic_locked_input(...)` — steps 3-4
- `build_state_with_castep_veff_and_psi(...)` — pin V_eff to CASTEP
  AND inject custom ψ_in (sibling of `build_state_with_castep_veff`
  from Group B's test; if neither exists, this is the parent helper)
- `compute_s_block_sum(...)` — extract the closure used in
  `subspace_projector_iter1_vs_castep` lines 3522-3535 into a sharable
  function

### Success criteria (THE primary Davidson v1 gate)

| Criterion | Source | Verification |
|---|---|---|
| `cargo build --release --features scf_diag` succeeds | tooling | build |
| `n_locked == 13`, `locked_indices == [1..14]` | construction | hard assert |
| **For all b ∈ 1..14 and all g: ψ_out[b][g] == ψ_perturbed[b][g] bitwise** | locking invariant | hard assert |
| Cu-3d block sum (vs CASTEP) ∈ [13.0 - 1e-7, 13.0 + 1e-7] | identity ⇒ block sum | hard assert |
| Non-Cu-3d max residual > 1e-3 (i.e. did NOT spuriously lock) | dispatch correctness | inspect log |

If C1 passes: Davidson v1 (Phase 1A) unblocked on the necessary
correctness condition. Proceed to C2.

If C1 fails on bitwise assertion: the locking branch IS rotating
locked bands (gather/scatter index bug, S-orth-against-locked
back-reaction, or fundamental algorithm mistake). Davidson v1 is
**unsafe to ship without fixing**. Investigate before C2 — do not
spend the C2 GPU budget on a broken algorithm.

If C1 fails on `n_locked != 13`: residual computation is wrong, or
construction does not actually leave Cu-3d at residual ~ 1e-12.
Diagnose the residual computation against a hand-computed
expectation.

### Commit

`test(scf): Phase 0 Gate 3' — synthetic-lock identity preservation for Davidson`

---

## Group C2 — SCF-3 cascade with Davidson (`group-gate3-pp`)

**Kind:** lib-tdd (ODD)
**Branch:** `impl/phase-eigensolver-migration/group-gate3-pp`
**Estimated LOC:** ~50
**Dependency:** Group C1 PASSED (verified with green test run, not just merged)

### Context

The integrated test. C1 establishes that Davidson's locking preserves
locked bands; C2 establishes that this preservation actually arrests
the cascade through real SCF iterations (not just construction).

The cascade signature: under Chebyshev-RR, `cascade_iter3_diagnostic_tight`
(`tests/ca_scf_convergence.rs:3182-3237`) reports band-0 = -11.94 Ha
at iter-3 (vs CASTEP -1.055 Ha, drift = 10.9 Ha). Stale ρ_aug from
rotated bands cascades into V_eff over three iterations.

If Davidson with locking actually fixes this, iter-3 band-0 drift
should drop dramatically.

### Construction

Reuse the `cascade_iter3_diagnostic_tight` test infrastructure
verbatim, but with `CHEMRUST_EIGENSOLVER=davidson`. Compare against
the recorded baseline.

```rust
#[test]
#[ignore]
#[cfg(feature = "scf_diag")]
fn gate3_prime_prime_davidson_stops_cascade_through_scf3() {
    // Phase 0 Gate 3'' — does Davidson's locking arrest the iter-3
    // cascade that defeats Chebyshev-RR? Sufficient condition for
    // Davidson v1 (Phase 1A). See TASKS.md Group C2.

    unsafe {
        std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson");
        // Lean lock_tol — production Phase 1A would tighten progressively;
        // for Phase 0 single-sweep, use a value that captures bands as
        // they reach iter-stable residuals. 1e-2 trades Phase-1A's
        // would-be tighter convergence for Phase-0's single-sweep budget.
        std::env::set_var("CHEMRUST_DAVIDSON_LOCK_TOL", "1e-2");
    }
    let _guard = scopeguard::guard((), |_| unsafe {
        std::env::remove_var("CHEMRUST_EIGENSOLVER");
        std::env::remove_var("CHEMRUST_DAVIDSON_LOCK_TOL");
    });

    let fx = fixtures::cu111_co::fixture();
    let mut state = fixtures::cu111_co::build_scf_state(fx);

    let mut iter3_band0_ha = f64::NAN;
    let mut iter_locks: Vec<usize> = Vec::new();
    for iter_idx in 1..=3 {
        let veff_state = state.build_v_eff().expect("v_eff");
        let diag = veff_state.diagonalize(0, None).expect("diag");
        if let Some(dr) = diag.davidson_diagnostics() {
            iter_locks.push(dr.n_locked);
            println!(
                "[Gate 3''] iter {iter_idx}: n_locked = {} / 160, max_res = {:.3e}",
                dr.n_locked, dr.max_residual,
            );
        }
        let evs = diag.eigenvalues();
        if iter_idx == 3 {
            iter3_band0_ha = evs[0];
        }
        let densified = diag.construct_density().expect("density");
        let mixed = densified.mix().expect("mix");
        state = mixed.check().unwrap_or_else(|s| s);  // continue regardless
    }

    let castep_band0 = -1.05502287_f64;
    let drift = (iter3_band0_ha - castep_band0).abs();
    let chebyshev_baseline_drift = 10.9_f64;  // iter-3 = -11.94 Ha

    let decision = if drift < 0.5 {
        "PASS — cascade arrested. Phase 1A Davidson v1 unblocked."
    } else if drift < 2.0 {
        "PARTIAL — cascade reduced but not eliminated. Phase 1A scope must include preconditioner before further validation."
    } else if drift < chebyshev_baseline_drift * 0.5 {
        "WEAK — cascade reduced < 50%. Davidson alone insufficient; consider CG."
    } else {
        "FAIL — cascade unaffected. Davidson does NOT fix the cascade. Fall back to Phase 1B (block CG)."
    };

    println!("[Gate 3''] iter-3 band-0 = {iter3_band0_ha:.6} Ha");
    println!("[Gate 3''] CASTEP band-0  = {castep_band0:.6} Ha");
    println!("[Gate 3''] drift          = {drift:.6} Ha");
    println!("[Gate 3''] Chebyshev-RR baseline drift: {chebyshev_baseline_drift:.6} Ha");
    println!("[Gate 3''] Locks per iter: {iter_locks:?}");
    println!("[Gate 3''] Decision: {decision}");

    assert!(iter3_band0_ha.is_finite(), "iter-3 band-0 not finite");
}
```

### Pre-registered decision matrix

| iter-3 drift | Decision |
|---|---|
| < 0.5 Ha | PASS — Phase 1A Davidson v1 (no further test) |
| 0.5 – 2.0 Ha | PARTIAL — Phase 1A but scope MUST include preconditioner before further test |
| 2.0 – 5.45 Ha (< 50% of baseline) | WEAK — preferential CG over Davidson; revisit at Phase 2 |
| ≥ 5.45 Ha | FAIL — Phase 1B block CG; Davidson does not fix cascade |

These thresholds are pre-registered per the discipline that bit Group
C: post-hoc threshold rationalization undermines the gate. **If the
result lands between buckets, the worse interpretation wins.**

### Why lock_tol = 1e-2 not 1e-3

C2 runs the single-sweep Davidson three times. With our V_eff,
residuals are ~0.1 Ha at iter-1. At lock_tol = 1e-3, we'd lock zero
bands — same situation as the original Group C, defeating the test.
At lock_tol = 1e-2, ~half the bands lock per iter (the safer subspace
distant from V_eff drift); the rest go through ZHEGVD. This is the
operating point that exercises the locking branch in the integrated
SCF.

This is a deliberate compromise specific to Phase 0's single-sweep
budget. Phase 1A's outer iteration tightens residuals progressively
(start lock_tol = 1e-2, ratchet down), so production won't have this
fixed-tolerance issue.

### Source-audit instructions

**Before editing,** read:
- `tests/ca_scf_convergence.rs:3182-3237`
  (`cascade_iter3_diagnostic_tight`) — exact iter loop pattern,
  state-passing convention, eigenvalue extraction.
- `tests/ca_scf_convergence.rs` — find `eigenvalues()` accessor on the
  `WavefunctionsUpdated` typestate; if it doesn't exist, this is a
  pre-task (small, ~5 LOC).

### Cost

~500 s GPU on Cu111+CO (3 SCF iterations × ~170 s/iter). Gated behind
C1 passing because: a Davidson with broken locking would still produce
a result here, but we'd be paying GPU time to confirm what C1 already
told us. Sequential dependency, not parallel.

### Success criteria

| Criterion | Verification |
|---|---|
| Test runs to completion without panic | `cargo test --release --features scf_diag gate3_prime_prime_davidson_stops_cascade_through_scf3 -- --ignored --nocapture` |
| iter-3 band-0 is finite | hard assert |
| Decision text printed maps to decision matrix | inspect log |
| Per-iter `n_locked` printed for forensic record | inspect log |

### Commit

`test(scf): Phase 0 Gate 3'' — Davidson SCF-3 cascade behavior`

---

## Group D — Decision artifact (`group-decision`)

**Kind:** direct
**Branch:** `impl/phase-eigensolver-migration/group-decision`
**Estimated LOC:** ~60 (the file is hand-written, not generated)
**Dependency:** Group C1 + C2 run to completion (C2 only if C1 passes)

### Context

Captures the human Phase 1 algorithm choice in a stable artifact. The
revised `GATE3_RESULT.md` retrospects on the superseded Group C run
and records the C1 + C2 outcomes. The test log alone is too ephemeral
to anchor weeks of subsequent work.

### Procedure

1. Run Gate 3' (necessary condition):

```bash
cd /home/tony/programming/chemrust-scf
cargo build --release --features scf_diag
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 \
  cargo test --release --features scf_diag \
  gate3_prime_davidson_synthetic_lock_preserves_locked_bands \
  -- --ignored --nocapture 2>&1 | tee /tmp/gate3-prime.log
```

2. If Gate 3' fails on the bitwise assertion, **stop** — debug the
   Davidson locking branch (gather/scatter, S-orth-against-locked
   back-reaction). Do NOT spend C2 GPU budget. Update `GATE3_RESULT.md`
   with the failure mode and `Phase 1B (block CG)` decision.

3. If Gate 3' passes, run Gate 3'' (sufficient condition):

```bash
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 \
  cargo test --release --features scf_diag \
  gate3_prime_prime_davidson_stops_cascade_through_scf3 \
  -- --ignored --nocapture 2>&1 | tee /tmp/gate3-prime-prime.log
```

4. Capture both test logs' Decision lines and metrics. Apply the
   pre-registered decision matrix (Group C2) to pick Phase 1
   algorithm.

5. Write `notes/plans/phase-eigensolver-migration/GATE3_RESULT.md` using
   the template below, REPLACING the existing one (preserve the
   existing one's content under a "Superseded run" section as
   forensic record).

### File template (revised)

```markdown
# Gate 3 Result — Phase 0 Davidson Decision

**Date:** YYYY-MM-DD
**Branch:** feat/phase-eigensolver-migration
**Commit:** <git rev-parse HEAD>
**Tests:**
- gate3_prime_davidson_synthetic_lock_preserves_locked_bands (C1)
- gate3_prime_prime_davidson_stops_cascade_through_scf3 (C2)

## Gate 3' (necessary condition — synthetic-lock identity)

| Metric | Value |
|---|---|
| n_locked / target 13 | NN / 13 |
| locked_indices match [1..14] | yes / no |
| Cu-3d bands bitwise preserved | yes / no — band X G Y diverged by NNN |
| Cu-3d block sum vs CASTEP | NN.NNNNNNNNNN |
| Block sum within [13.0 ± 1e-7] | yes / no |
| Max residual on unconverged set | N.NNNe-NN Ha |
| Perturbation epsilon used | 0.NN |

**Outcome:** [PASS — locking invariant holds | FAIL — see diagnostic notes]

## Gate 3'' (sufficient condition — SCF-3 cascade)

(Only if C1 PASSED. If C1 FAILED, this section reads "Not run — C1
prerequisite failed.")

| Metric | Value |
|---|---|
| Iter-3 band-0 | NN.NNNNNN Ha |
| CASTEP band-0 reference | -1.05502287 Ha |
| Drift | N.NNNNNN Ha |
| Chebyshev-RR baseline drift | 10.9 Ha (recorded) |
| Locks per iter | [N, N, N] |
| Max residual per iter | [N.NNe-N, N.NNe-N, N.NNe-N] Ha |

**Outcome (per pre-registered decision matrix):**
[PASS | PARTIAL | WEAK | FAIL]

## Decision

**Phase 1 algorithm:** [Davidson v1 (Phase 1A) | block CG (Phase 1B) | partial-Davidson-with-preconditioner-scope-expansion]

**Rationale:** [2-3 sentences citing both C1 (locking-invariant
correctness) and C2 (cascade-arrest sufficiency). Pre-registered
threshold cited explicitly. If C1 failed, rationale explains the
locking-branch failure mode. If C2 was WEAK or FAIL despite C1 PASS,
rationale explains why locking alone is insufficient.]

## Next-phase anchor

`/drive-outcomes notes/plans/phase-eigensolver-migration/PHASE_PLAN.md`
section "Phase 1A — Davidson v1" — IF C1+C2 both PASS

OR

`/drive-outcomes notes/plans/phase-block-cg-migration/PHASE_PLAN.md`
section "Phase 0" — IF C1 FAILED or C2 was FAIL/WEAK

(delete the wrong one after the decision is made)

## Diagnostic notes

[Anything surprising in the runs — e.g., epsilon sensitivity, lock
count variance across iters, sibling block sums diverging from
expected plateaus. These notes are forensic anchors for Phase 1 if it
encounters issues.]

## Superseded run (forensic record)

(Preserve here the contents of the prior GATE3_RESULT.md from the
original Group C — block sum 12.929543, n_locked = 0, etc. Explain
in 1 paragraph why this run was insufficient as a decision gate:
locking branch never fired, ratio measured filter-free-ZHEGVD not
locking. Reference TASKS.md amendment log.)
```

### Success criteria

| Criterion | Source | Verification |
|---|---|---|
| GATE3_RESULT.md exists with all fields populated for both C1 and C2 (or C1+"not run" for C2 if C1 failed) | revised template | `ls notes/plans/phase-eigensolver-migration/GATE3_RESULT.md` |
| Decision matches the conjunction of C1 and C2 outcomes via the pre-registered matrix | consistency | inspect file vs test logs |
| Anchor section unambiguously names ONE phase plan | unblock Phase 1 | reader picks next plan without re-running tests |
| Superseded-run section preserves the original GATE3_RESULT.md content | forensic record | grep for "12.929543" in file |

### Commit

`docs(plans): record Gate 3'/3'' result — Phase 1 algorithm decision`

---

## Exploration notes

These are the surprises and adjustments captured during planning
(per ODD pattern). Implementation should treat them as institutional
context, not rework them.

### E1. The "use our V_eff" choice is load-bearing

PHASE_PLAN.md line 60-61 phrases Gate 3 as "Load CASTEP ψ from
Cu111_CO.check. Run minimal Davidson with ndeg=0 starting from CASTEP ψ."
Read literally, this could include CASTEP V_eff (since CASTEP ψ + CASTEP
V_eff is the natural "starting from CASTEP" reading). DECISIONS.md A4
documents why this would defeat the test.

The discriminator is built by V_eff drift. If during implementation
the drift is very small (~µHa) then few bands have residuals > tol and
ZHEGVD acts on a tiny subspace — the test signal weakens. Mitigation:
log `n_unconverged` and `mean_residual`; if `n_unconverged < 10` flag
the run as suspect and consider increasing the iter count or using a
slightly perturbed starting density.

### E2. ndeg parameter pollution risk

`diagonalize(ndeg, occupations)` keeps ndeg in its signature. A reader
encountering `diagonalize(0, None)` with `CHEMRUST_EIGENSOLVER=davidson`
might assume ndeg=0 means "no Davidson iterations" since for Chebyshev
ndeg=0 means "no filter polynomial." The dispatch comment ("ndeg
ignored — davidson_minimal is single-sweep") in Group A is explicitly
intended to forestall this misreading.

### E3. `apply_s_times` identity-term pre-copy

The most common silent bug in adapting `apply_s_times` is forgetting
the pre-copy `spsi := psi`. The function only adds the augmentation
term `Σ_ion β·Q·β^H·ψ`; the identity `I·ψ = ψ` term must be in
`spsi_dev` before the call. Algorithm step 3 includes the explicit
`memcpy_dtod`. Group B's self-consistency unit test catches a missing
pre-copy: without it, `Sψ_b ≈ 0.4·ψ_b + augmentation` for Cu 3d states,
the Rayleigh quotient diverges from CASTEP eigenvalues by 60%, and the
self-consistency check fails immediately.

Reference: failure-pattern `uspp-pw-norm-not-unit-ncpp-assumption`
(2026-05-23) shows ‖ψ‖²_PW for Cu 3d ranges 0.14-1.03 under USPP.

### E4. ZHEGVD k=0 degenerate case

If V_eff is CASTEP-pinned (Group B trivial test) all bands lock, k=0,
and the gather→ZHEGVD path would either NaN or panic. Algorithm step 8
returns early with input ≡ output. This makes the trivial test pass
without exercising the unconverged path.

The Gate 3 test (Group C) uses our V_eff specifically to exercise k > 0
for a meaningful subset of bands. If the Gate 3 run reports k = 0,
that itself is a finding (V_eff is too close to CASTEP's; investigate
or accept).

### E5. `RrPinConfig::from_env` should not be invoked in davidson branch

The chebyshev path passes `Some(&pin_cfg)` to `rayleigh_ritz`. The
davidson path skips this entirely. **Important:** do not let the env
var read in Group A also trigger `RrPinConfig::from_env`. Two env vars
that interact would create combinatorial test surface. Davidson runs
with `CHEMRUST_PIN_MODE` ignored (the davidson branch does not call
the post-RR Procrustes pin).

### E6. `serial_test` or scopeguard for env-var tests

Tests that mutate `CHEMRUST_EIGENSOLVER` need either `serial_test`
attribute (forces single-threaded test execution) or a scopeguard that
clears the var on test exit. The Group C and Group B test snippets use
scopeguard. If `serial_test` is already a dev-dependency, prefer it
(less boilerplate). If not, scopeguard works fine.

Rust 2024+ requires `unsafe` around `std::env::set_var` in
multi-threaded contexts. Match the surrounding codebase's pattern.

### E7. Group C (superseded) was structurally undertested

**Lesson from the actual Group C run** (block sum 12.929543,
**n_locked = 0**): an unconstrained natural-residual approach cannot
test a locking mechanism if the operating point doesn't put any bands
below `lock_tol`. The original Group C plan documented residuals as
"tiny for CASTEP-near eigenvectors" but our V_eff produces ~0.1 Ha
residuals — three orders of magnitude above 1e-6. With zero bands
locked, the test became "filter-free single-sweep ZHEGVD" not
"locking-aware Davidson," and Block CG would have produced an
identical number for the same reason. The 0.995 ratio passed the
0.97 threshold by accident of the threshold not the algorithm.

**Test-design principle (encoded in C1):** when testing a conditional
mechanism, the test must **force the condition to fire by
construction**, not hope for the natural distribution to place the
operating point in the right regime. C1's synthetic-lock construction
hand-places Cu-3d in the locked set; C2's `lock_tol = 1e-2` is
calibrated so residuals at iter-1 produce a meaningful lock count.

### E8. Bitwise identity beats epsilon comparisons for locking tests

Locking is defined as "no rotation of locked bands." The natural
metric is `‖ψ_out_b − ψ_in_b‖_S < eps`. But this admits accidental
near-zero diffs from spurious back-reaction (e.g.,
S-orth-against-locked subtracts ⟨ψ_j | S | ψ_b⟩ but the dot is
~1e-12 instead of 0, so ψ_j changes by ~1e-12 — would pass `eps =
1e-10` but is a bug). Bitwise `==` catches this: a correctly-locked
band MUST be the input value byte-for-byte (the locking algorithm
should `memcpy_dtod` it through, no float math). Group C1 uses `==`
on `CudaComplex` (which is `Complex<f64>` ≅ `[f64; 2]`). Any
arithmetic on the locked band raises a hard fail.

### E9. C2's `lock_tol = 1e-2` is a Phase-0-specific compromise

Phase 1A's outer iteration tightens residuals progressively, so
production Davidson would start at lock_tol ≈ 1e-2 and ratchet down
to 1e-6 as bands converge. Phase 0's single-sweep can't ratchet —
we get one tolerance for the entire SCF. 1e-2 is the operating
point where the single-sweep locks a meaningful fraction of bands
at iter-1 without being so loose that nearly-divergent bands lock.
If C2 fails at 1e-2, sweep lock_tol manually (1e-3, 5e-3, 5e-2) via
env var before declaring Davidson broken — the failure may be the
tolerance-sweep being too coarse.

## Risks (pre-mitigation)

| Risk | Mitigation |
|------|-----------|
| ZHEGVD on small k subspace returns degenerate eigenvectors that re-rotate the unconverged Cu 3d bands among themselves | Gate 3's block-sum reads the rotation if it happens; that's the test signal. No mitigation needed — that IS the answer to the gate question |
| Implementation discovers `apply_full_hamiltonian` requires kinetic_dev not in our state | Audit during Group A; if missing, `compute_kinetic_energies` at chebyshev.rs:581-599 + `stream.clone_htod` produces it (~5 LOC) |
| `cargo build --features scf_diag` slow due to fixture loading | Acceptable; tests are `#[ignore]`-gated and run on demand |
| Group C cannot extract `n_locked` / `max_residual` from current ScfIteration return type | Group A propagates DavidsonResult fields out, OR Davidson writes them to a thread-local diagnostic struct readable post-hoc. Decide at Group A implementation; do not gold-plate |

## Reading list (PHASE_PLAN.md line 292-307)

1. PHASE_PLAN.md (sibling) — Phase 0 spec, Gate 3 table, decision matrix
2. `notes/debug/debug-20260524-blow-tightening/RESOLUTION.md` — why Chebyshev-RR is wrong
3. `notes/debug/debug-20260524-blow-tightening/CASTEP_ANCHORED_PIN_PROBE.md` — rotation IS the cascade driver
4. failure-pattern `blow-tightening-falsified` (failure-patterns.md last entry) — full audit of what's been falsified

Skip for Phase 0:
- Saad chapter (relevant for Phase 1A subspace management — not for single sweep)
- Payne 1992 (Phase 1B)
- Abinit GPU paper (Phase 1A optimizations + Phase 1B)
- CASTEP DFT+U source (Phase 4)
