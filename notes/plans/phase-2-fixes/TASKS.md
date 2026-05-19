# TASKS: Phase 2 Fixes — Code Review Corrections

## Declared Fixtures

- CPU-only reference: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
- GPU FFT-only ver: `/tmp/cu111_gpu_resident_scf/`

These are used for end-to-end SCF verification. Individual unit tests use synthetic
data with concrete expected values.

## Source-audited reference

- CASTEP GPU port: `~/programming/CASTEP-GPU-port/` (used for FFT convention verification)
- Abinit algorithm arxiv paper: `~/programming/CASTEP-GPU-port/2604.11139v1`

---

## Group A: Chebyshev / Rayleigh-Ritz fixes

### TASK-P1: Fix PcieAccount eigenvalue D2H tracking in rayleigh_ritz

**Kind:** direct
**File changes:**
- `src/scf.rs` — no change needed (assertion already correct, just needs PcieAccount
  to be threaded through)
- `src/eigensolver/rayleigh_ritz.rs` — add `pcie: &mut PcieAccount` parameter,
  instrument eigenvalue D2H after line 205

**Changes:**
1. Add `use crate::device::pcie::PcieAccount;` to rayleigh_ritz.rs
2. Add `pcie: &mut PcieAccount,` parameter to `rayleigh_ritz` function signature
   (between `kernels` and `solver`)
3. After line 205 (`.clone_dtoh(&eigenvalues_dev)`), add:
   `pcie.d2h_bytes += eigenvalues.len() * 8;`
4. Update the call site in `scf.rs` to pass `&mut pcie` as the new argument

**Success Criteria:**
- `cargo check` passes
- D2H assertion `pcie.d2h_bytes == psi_bytes + eig_bytes` passes at runtime
- Eigenvalues are correctly tracked in PcieAccount

---

### TASK-P2: Fix FFT dimension ordering mismatch

**Kind:** direct
**File changes:**
- `src/eigensolver/chebyshev.rs` — fix chebyshev's `plan_batched_c2c` call
- `src/density.rs` — fix density construction's `plan_batched_c2c` call
- `src/device/fft.rs` — check `plan_c2c` in identity test

**Changes:**
1. `src/eigensolver/chebyshev.rs:686-688`: Change `ngx, ngy, ngz` to `ngz, ngy, ngx`
2. `src/density.rs:144-147`: Change `ngx, ngy, ngz` to `ngz, ngy, ngx`
3. `src/device/fft.rs:249`: Identity test is 8×8×8 (cubic), so dimension order
   is irrelevant — no change needed

**Rationale:** cuFFT expects dimensions in `[z, y, x]` order (most-significant first),
but our index formula `ix + ngx * (iy + ngy * iz)` has x as fastest-varying and z as
slowest-varying. Passing `[ngx, ngy, ngz]` transposes the FFT axes.

**Success Criteria:**
- `cargo check` passes
- FFT identity test (8³ cubic) still passes after ordering change
- For non-cubic grids (e.g., 8×8×16), a non-uniform test pattern would confirm
  correct frequency placement

---

### TASK-P3: Remove unnecessary `#[allow(dead_code)]` on `SpectralBounds`

**Kind:** direct
**File changes:**
- `src/eigensolver/chebyshev.rs`

**Changes:**
1. Remove `#[allow(dead_code)]` from line 237 (preceding `pub(crate) struct SpectralBounds`)

**Rationale:** `SpectralBounds` is fully used — `compute_spectral_bounds` constructs it
and `chebyshev_filter` reads all fields (`lambda_max`, `eps_cut`, `center`, `half_width`).

**Success Criteria:**
- `cargo check` — no dead_code warning on `SpectralBounds` or its fields

---

## Group B: H2D assertion

### TASK-P4: Add H2D PCI-E assertion

**Kind:** direct
**File changes:**
- `src/scf.rs` — add H2D assertion
- `src/eigensolver/chebyshev.rs` — track kinetic energy upload, VNL uploads
- Potentially VNL path (`src/eigensolver/chebyshev.rs` or vnl_data builder)

**Changes:**
1. Track all H2D transfers:
   - `pw_fft_indices` upload (scf.rs:303): add `pcie.h2d_bytes += ...`
   - `kinetic_dev` upload (chebyshev.rs:683): add `pcie.h2d_bytes += ...`
   - VNL beta_g / D_matrix uploads in `VnlBatchData::precompute`: route through
     PcieAccount or track bytes manually
2. Add H2D assertion after the D2H assertion:
   ```rust
   assert_eq!(
       pcie.h2d_bytes,
       psi_bytes + veff_bytes + fft_idx_bytes + kinetic_bytes + vnl_bytes,
       "H2D tracking check failed",
   );
   ```

**Success Criteria:**
- H2D assertion passes when diagonalize is called
- All setup H2D transfers are tracked

---

## Group C: Density construction fixes

### TASK-P5: Fix occupation sign-inversion in compute_occupations

**Kind:** direct
**File changes:**
- `src/density.rs`

**Changes:**
1. Line 43: Change `erfc((mu - e) / smearing.width)` to `erfc((e - mu) / smearing.width)`
2. Line 68-72: Change bisection logic:
   - `if sum > n_electrons { lo = mid; }` → `if sum > n_electrons { hi = mid; }`
   - `else { hi = mid; }` → `else { lo = mid; }`

**Rationale:** The correct Gaussian-smearing occupation formula is `erfc((ε_b - μ) / w)`
where states far below μ (ε_b << μ): erfc(-large) ≈ 2 → fully occupied. The old formula
`erfc((μ - ε_b) / w) = 2 - erfc((ε_b - μ)/w)` inverts this.

**Success Criteria:**
- `cargo check` passes
- Unit test: for eigenvalues [0.0, 0.1, 0.2], width=0.1, n_electrons=3.0,
  occupations should be approximately [1.0, 1.0, 1.0], bisection converges to μ ≈ 0.1

---

### TASK-P6: Fix accumulate_density kernel launch config (grid-stride loop)

**Kind:** direct
**File changes:**
- `src/eigensolver/chebyshev.rs` — CUDA kernel source
- `src/density.rs` — launch config

**Changes:**
1. Rewrite `accumulate_density` kernel in `src/eigensolver/chebyshev.rs` as a
   grid-stride loop without shared memory or reduction:
   ```cuda
   extern "C" __global__ void accumulate_density(
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
2. The existing `LaunchConfig::for_num_elems(grid_size as u32)` now works correctly
   because the grid-stride loop handles the block→element mapping.

**Success Criteria:**
- `cargo check` passes
- Kernel produces correct output when called with actual data on GPU

---

## Exploration notes

- Fix-tasks.md was produced by `/make-judgement` code review. All 6 tasks have
  verified guidance that matches the actual codebase state.
- P1: `rayleigh_ritz` currently returns `Cpu<Vec<f64>>` for eigenvalues from an
  untracked `clone_dtoh`. The D2H assertion in `scf.rs` includes `eig_bytes` but
  the bytes are never counted.
- P2: Both FFT plan call sites (`chebyshev.rs:686`, `density.rs:144`) pass
  `(ngx, ngy, ngz)` but cuFFT expects `(ngz, ngy, ngx)`.
- P3: `#[allow(dead_code)]` is on `SpectralBounds` struct (line 237) — not line 213
  as stated in fix-tasks.md (line numbers drifted).
- P5: Current formula inverts occupation — confirmed against Gaussian smearing
  definition: erfc((ε_b - μ)/w) with bisection on f(μ) increasing.
- P6: Current kernel uses `blockIdx.x` as grid-point index with
  `LaunchConfig::for_num_elems` which computes `ceil(grid_size/1024)` blocks.
  For grid_size < 1024, only 1 block is launched, so multi-band accumulation at
  each grid point works by accident. For grid_size > 1024, grid points beyond 1024
  are silently dropped.
