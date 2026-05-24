# TASKS — Phase 0: Davidson Gate 3

**Phase:** Phase 0 of `notes/plans/phase-eigensolver-migration/PHASE_PLAN.md`
**Status:** READY for implementation. Awaits user `continue` after checkpoint review.
**Date:** 2026-05-24
**Predecessor:** `feat/phase-global-woodbury` branch (Chebyshev-RR forensic baseline)
**Decisions:** see `DECISIONS.md` (sibling file)
**ODD pattern reference:** `~/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

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
                     ├─→ group-gate3 (C) ──→ group-decision (D)
group-davidson (B) ──┘
```

A and B can implement in parallel on independent sub-branches; C
requires both. D is post-execution write-up.

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

## Group C — Gate 3 test (`group-gate3`)

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

## Group D — Decision artifact (`group-decision`)

**Kind:** direct
**Branch:** `impl/phase-eigensolver-migration/group-decision`
**Estimated LOC:** ~30 (the file is hand-written, not generated)
**Dependency:** Group C merged AND Gate 3 test executed

### Context

Captures the human Phase 1 algorithm choice in a stable artifact. The
test log alone is too ephemeral to anchor weeks of subsequent work.

### Procedure

1. Run Gate 3:

```bash
cd /home/tony/programming/chemrust-scf
cargo build --release --features scf_diag
CASTEP_FIXTURE_DIR=/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8 \
  cargo test --release --features scf_diag \
  gate3_davidson_minimal_locking_preserves_cu3d_block \
  -- --ignored --nocapture 2>&1 | tee /tmp/gate3.log
```

2. Capture the printed `Decision:` line, block sum, lock count, and
   max residual.

3. Write `notes/plans/phase-eigensolver-migration/GATE3_RESULT.md` using
   the template below.

### File template

```markdown
# Gate 3 Result — Phase 0 Davidson Decision

**Date:** YYYY-MM-DD
**Branch:** feat/phase-eigensolver-migration
**Commit:** <git rev-parse HEAD>
**Test:** gate3_davidson_minimal_locking_preserves_cu3d_block

## Measurements

| Metric | Value |
|---|---|
| Cu-3d block sum (bands 1..14, S-weighted) | NN.NNNNNN |
| Davidson ratio (sum / 13.0) | 0.NNNNNN |
| Chebyshev-RR baseline (recorded) | 11.610000 (ratio 0.893) |
| CASTEP self-overlap reference | 13.000000 |
| Bands locked / total | NN / 160 |
| Max residual norm | N.NNNe-NN |
| Mean residual norm | N.NNNe-NN |
| Sibling sums | band0=N.NNNNNN, 0..30=NN.NNNNNN, 0..40=NN.NNNNNN |

## Decision

**Phase 1 algorithm:** [Davidson v1 (Phase 1A) | block CG (Phase 1B) | mixed]

**Rationale:** [1-2 sentences citing the threshold from
PHASE_PLAN.md Risks row 1: ratio ≥ 0.97 → Davidson, < 0.97 → CG.]

## Next-phase anchor

`/drive-outcomes notes/plans/phase-eigensolver-migration/PHASE_PLAN.md`
section "Phase 1A — Davidson v1"

OR

`/drive-outcomes notes/plans/phase-block-cg-migration/PHASE_PLAN.md`

(delete the wrong one after the decision is made)

## Diagnostic notes

[Anything surprising in the run — e.g., bands locked > expected, max
residual unusually large, sibling sums diverging from expected
plateaus. These notes are forensic anchors for Phase 1 if it
encounters issues.]
```

### Success criteria

| Criterion | Source | Verification |
|---|---|---|
| GATE3_RESULT.md exists with all fields populated | template | `ls notes/plans/phase-eigensolver-migration/GATE3_RESULT.md` |
| Decision matches Group C output | consistency | grep `Decision:` stdout matches MD |
| Anchor section unambiguously names ONE phase plan | unblock Phase 1 | reader picks next plan without re-running Gate 3 |

### Commit

`docs(plans): record Gate 3 result — Phase 1 algorithm decision`

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
