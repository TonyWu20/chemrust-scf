# Fix Tasks: Phase 2 Group-C — diagonalize

## P1: Fix PcieAccount eigenvalue D2H tracking

The `rayleigh_ritz` function downloads eigenvalues via raw `clone_dtoh` which bypasses `PcieAccount`. The caller's D2H assertion expects `psi_bytes + eig_bytes` but only `psi_bytes` is tracked.

**Files:** `src/scf.rs`, `src/eigensolver/rayleigh_ritz.rs`

**Guidance:**
Option A (preferred): Pass `&mut PcieAccount` into `rayleigh_ritz` and instrument the eigenvalue D2H:
1. Add `pcie: &mut PcieAccount` parameter to `rayleigh_ritz`
2. After the `clone_dtoh` on line 204-206, add `pcie.d2h_bytes += eigenvalues.len() * 8;`
3. Update the call site in `scf.rs` to pass `&mut pcie`

Option B (minimal): Remove `eig_bytes` from the expected D2H count in `scf.rs` and only assert `psi_bytes`. Document why eigenvalues are excluded.

**Success Criteria:**
- Assertion `pcie.d2h_bytes == psi_bytes + eig_bytes` passes at runtime
- Eigenvalues are correctly tracked in PcieAccount

---

## P2: Fix FFT dimension ordering mismatch

`plan_batched_c2c(ngx, ngy, ngz)` passes `[ngx, ngy, ngz]` to cuFFT, but the index formula `ix + ngx * (iy + ngy * iz)` has ngx as fastest-varying. cuFFT expects `[ngz, ngy, ngx]`.

**Files:** `src/eigensolver/chebyshev.rs`

**Guidance:**
Change line 662-664 from:
```rust
let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
    ngx as i32, ngy as i32, ngz as i32, n_bands_i32, stream.clone(),
)?;
```
to:
```rust
let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
    ngz as i32, ngy as i32, ngx as i32, n_bands_i32, stream.clone(),
)?;
```

Also audit all other `plan_c2c` / `plan_batched_c2c` call sites in the codebase for correct dimension ordering.

**Success Criteria:**
- FFT identity test (8³ cubic) still passes after ordering change
- For non-cubic grids (e.g., 8×8×16), a non-uniform test pattern confirms correct frequency placement

## P3: Remove unnecessary `#[allow(dead_code)]` on `SpectralBounds`

**Files:** `src/eigensolver/chebyshev.rs`

**Guidance:**
Remove `#[allow(dead_code)]` from line 213. The `SpectralBounds` struct is fully used — `compute_spectral_bounds` constructs it and `chebyshev_filter` reads all fields.

**Success Criteria:**
- `cargo check` — no dead_code warning on `SpectralBounds`
- No dead_code warning on `lambda_max` or `eps_cut` fields

---

## P4: Add H2D assertion (optional, from strategic review)

**Files:** `src/scf.rs`

**Guidance:**
Add H2D assertion after D2H assertion, tracking PCI-E uploads:
```rust
assert_eq!(
    pcie.h2d_bytes,
    psi_bytes + veff_bytes + fft_idx_bytes + kinetic_bytes + vnl_bytes,
    "H2D tracking check failed",
);
```

Note: this requires routing all H2D transfers through `PcieAccount` (currently `clone_htod` calls for fft_idx, kinetic, VNL beta/D matrices bypass tracking). Add `pcie.h2d_bytes += ...` after each setup transfer, or route them through `Gpu::from_host_with`.

**Success Criteria:**
- H2D assertion passes when diagonalize is called
- All setup H2D transfers are tracked
