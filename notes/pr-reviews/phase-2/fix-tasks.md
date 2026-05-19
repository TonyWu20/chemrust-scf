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

---

# Group D Fix Tasks

## P5: Fix occupation sign-inversion in compute_occupations

`erfc((μ - ε_b) / width)` at `src/density.rs:43` is the inversion of the correct formula `erfc((ε_b - μ) / width)`. The bisection search in `find_chemical_potential` must also flip its comparison direction.

**Files:** `src/density.rs`

**Guidance:**

1. Change line 43 from:
   ```rust
   .map(|&e| libm::erfc((mu - e) / smearing.width))
   ```
   to:
   ```rust
   .map(|&e| libm::erfc((e - mu) / smearing.width))
   ```

2. Change line 68 from:
   ```rust
   if sum > n_electrons {
       lo = mid;
   } else {
       hi = mid;
   }
   ```
   to:
   ```rust
   if sum > n_electrons {
       hi = mid;
   } else {
       lo = mid;
   }
   ```

**Rationale:** The correct Gaussian-smearing occupation formula is `erfc((ε_b - μ) / w)` where:
- States far below μ (ε_b << μ): erfc(-large) ≈ 2 → fully occupied
- States at μ (ε_b = μ): erfc(0) = 1 → half occupied
- States far above μ (ε_b >> μ): erfc(large) ≈ 0 → empty

`erfc((μ - ε_b)/w) = 2 - erfc((ε_b - μ)/w)` which inverts the occupation: states below μ get near-zero weight, states above μ get near-full weight.

With the correct formula, `f(μ) = Σ erfc((ε_b - μ)/w)` is **increasing** in μ (positive derivative). So:
- f(μ) > N → μ is too high → set `hi = mid`
- f(μ) < N → μ is too low → set `lo = mid`

**Success Criteria:**
- `cargo check` passes
- Unit test: for a sorted eigenval array [0.0, 0.1, 0.2], width=0.1, n_electrons=3.0, the computed occupations should be approximately [1.0, 1.0, 1.0], not inverted
- The bisection should converge to μ ≈ 0.1 (the midpoint) for the above case

---

## P6: Fix `accumulate_density` kernel launch config

The `accumulate_density` kernel at `src/density.rs:169` is launched with `LaunchConfig::for_num_elems(grid_size)` which has two bugs:
1. Grid dimension: computes `ceil(grid_size/1024)` blocks instead of `grid_size` blocks (kernel uses `blockIdx.x` as grid-point index, expecting one block per point)
2. Shared memory: `shared_mem_bytes = 0` but kernel uses `extern __shared__ double sdata[]`

**Files:** `src/density.rs`, `src/eigensolver/chebyshev.rs` (kernel might need adjustment)

**Guidance:**

Option A (preferred — minimal change to kernel): Replace the `LaunchConfig::for_num_elems` call with an explicit config that allocates shared memory and uses one block per grid point:

```rust
let block_size = 256u32;
let cfg = LaunchConfig {
    grid_dim: (grid_size as u32, 1, 1),
    block_dim: (block_size, 1, 1),
    shared_mem_bytes: block_size * std::mem::size_of::<f64>() as u32,
};
```

Then change the launch call to use `cfg` instead of `LaunchConfig::for_num_elems`.

Option B (modern — rewrite kernel with grid-stride loop, avoiding shared memory entirely): Rewrite the kernel to be a simple grid-stride loop without reduction:

```cuda
extern \"C\" __global__ void accumulate_density(
    const double2* psi_r, const double* occ,
    double* rho, int n_bands, int grid_size, double inv_omega
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (r < grid_size) {
        double sum = 0.0;
        for (int b = 0; b < n_bands; b++) {
            double2 psi = psi_r[b * grid_size + r];
            sum += occ[b] * (psi.x * psi.x + psi.y * psi.y);
        }
        rho[r] = sum * inv_omega;
        r += stride;
    }
}
```

This eliminates the shared memory and reduction logic entirely. Launch with `for_num_elems(grid_size)` after the rewrite (the original launch config works because the grid-stride loop handles the block → element mapping correctly).

**Option B is recommended** — simpler kernel, no shared memory, no reduction bugs, one less thing to track.

**Success Criteria:**
- `cargo check` passes
- Small-scale test on CPU: construct a known density from a small set of PW coefficients and verify against an NDArray reference computation
- The kernel produces correct output when called with actual data on GPU

---

# Group E Fix Tasks

## P7: Mixing phase transition logic in check()  ✅ FIXED

**Status: Applied during review session 2026-05-19.**

`check()` at `src/scf.rs:779` (old) copied `self.next_mixing` through unchanged — no phase transitions. The SCF loop always called `construct_density_off()?.mix()` (pass-through). Kerker and Pulay mixing were never engaged during an SCF run.

**Files:** `src/scf.rs`

**What was changed:**

1. Energy convergence formula changed from pairwise-diff to max−min over 3-entry window (aligns with CASTEP `electronic_store_energy`).
2. Phase transition logic added:
   - `Off → Kerker` when energy variation < 0.1 eV (`MIXING_CONV_TOL_EV`)
   - `Kerker → Pulay` after first Kerker mix completes
   - `Pulay → Pulay` for normal DIIS
3. Mixed-status guard: convergence requires `mixing_was_active` (next_mixing ≠ Off at entry), preventing false convergence when no mixing is perturbing the density.
4. 5 new unit tests exercise all transitions and the guard.

**Verification:**
- `cargo test --lib` → 27/27 pass
- `cargo clippy --workspace -- -D warnings` → clean
