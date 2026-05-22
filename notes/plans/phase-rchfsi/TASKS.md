# Forensic TASKS.md — R-ChFSI Implementation

ODD pattern reference: `/home/tony/.claude/plugins/cache/my-claude-marketplace/rust-development-pipeline/4.0.0/skills/drive-outcomes/references/odd-pattern.md`

## Declared Fixtures

| Path | Description |
|------|-------------|
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.castep_bin` | Cell, density on wave grid, eigenvalues |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.check` | Wavefunctions + fine grid (155 MB, cached) |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.pot_fmt` | Reference V_eff on fine grid |
| `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.bands` | Reference eigenvalues in Hartree |

## Success Criteria

| ID | Criterion | Source | Threshold |
|----|-----------|--------|-----------|
| SC-1 | `iter2_v_eff_range_within_one_ha_of_iter1` stays green | Existing test, `tests/ca_scf_convergence.rs` | \|iter-2 range − iter-1 range\| < 1.0 Ha |
| SC-2 | `fixed_point_matches_castep_energy` does not diverge after iter-2 | `notes/open-followups.md §10` | band-1 iter-2 within ±2 Ha of −1.055 Ha (loose gate; tighter once converging) |
| SC-3 | `cargo check --workspace` passes with no errors | Build gate | zero errors |
| SC-4 | `cargo clippy --workspace -- -D warnings` passes | Lint gate | zero warnings |

## Exploration Notes

- **Algorithm source verified**: Read `main.tex` lines 384–610 directly. Algorithm 3
  (R-ChFSI) is at lines 586–610. The recurrence uses `A` (= H in our notation) and
  `B` (= S), with `D⁻¹` as the approximate `S⁻¹`. The paper's `B·X·Λ` term in the
  residual definition (line 598: `Y ← A·X − B·X·Λ`) requires an explicit `S·X` apply.

- **Three distinct spectral bounds** (source: `main.tex:591`): The INPUTS list THREE
  bounds — `λ_max` (upper bound of full eigenspectrum), `λ_min` (lower bound of full
  eigenspectrum), and `λ_T` (upper bound of wanted spectrum). The σ formula at line
  597 uses `λ_min`, NOT `λ_T`. Our `SpectralBounds` currently has `lambda_max` (= `λ_max`)
  and `eps_cut` (= `λ_T`), but **no `lambda_min` field**. We must add it.

- **Current recurrence verified**: Read `chebyshev.rs` lines 1200–1301. The current
  code already applies `S⁻¹` after each `H·ψ` (lines 1222–1227, 1250–1254). This is
  the standard ChFSI with inexact S⁻¹ that stagnates per Theorem 3.2.

- **`apply_s_inverse` signature** (`chebyshev.rs:758`):
  ```rust
  unsafe fn apply_s_inverse(
      hpsi_dev: &mut CudaSlice<CudaComplex>,
      vnl_data: &VnlBatchData,
      n_bands: i32, n_pw: i32,
      blas: &BlasHandle, stream: &Arc<CudaStream>,
  ) -> Result<(), Error>
  ```
  Implements: `v ← v − β · (Q⁻¹ + β^H·β)⁻¹ · β^H · v` using `entry.s_inv_mat`.

- **`q_matrix` field confirmed**: `VnlIonData.q_matrix` is `CudaSlice<CudaComplex>`
  of shape `(n_expanded × n_expanded)`, already on GPU. Used by Rayleigh-Ritz for
  S_sub assembly. Available for `apply_s_times`.

- **Eigenvalue parameter**: `chebyshev_filter` already receives `eigenvalues: Option<&[f64]>`.
  No signature change needed. When `None` (first iteration), Λ = 0, so Y = H·X (skip
  S·X computation), Λ_Y = −σ₁·c/e (constant across bands). Reconstruction:
  X_new = D⁻¹·R_Y + X·Λ_Y. Source: `main.tex:612` — "When D⁻¹ = B⁻¹ ... ChFSI and
  R-ChFSI are algebraically equivalent."

- **Λ matrix representation**: Diagonal only. Store as `Vec<f64>` on CPU, apply as
  per-band column scaling on GPU (no n_bands² allocation needed).

- **`apply_scaled_hamiltonian_inplace`** (`chebyshev.rs:934`): computes
  `(H·ψ − c·ψ) / e` in-place. Not reusable for R-ChFSI recurrence (different
  structure), but the scalar arithmetic pattern is reusable.

- **Lanczos lambda_min_tk**: `lanczos_upper_bound` (line 335) already computes
  `lambda_min_tk` (Gershgorin lower bound of T_k tridiagonal) and returns it as the
  second tuple element. It is currently discarded at the call site (line 1135). This
  provides the `λ_min` estimate needed by R-ChFSI — no new computation required.

---

## Task Groups

### Group A: `apply_s_times` helper

**TASK-A1** — Implement `apply_s_times`

- **Kind**: `direct`
- **File**: `src/eigensolver/chebyshev.rs`
- **What**: Add a new private function `apply_s_times` that computes `S·ψ` in-place
  (or into a separate buffer). S = I + Σ_I β_I · Q_I · β_I^H.

  Algorithm (mirrors `apply_s_inverse` at line 758, but uses `q_matrix` and `+`):
  ```
  for each ion entry:
      p = beta_g^H · psi          (ne × n_bands, gemm: C^H · N)
      q = q_matrix · p            (ne × n_bands, gemm: N · N)
      spsi += beta_g · q          (n_pw × n_bands, gemm: N · N, alpha=+1)
  ```
  The output is `spsi = psi + Σ_I β_I · Q_I · β_I^H · psi`.

  Signature to implement:
  ```rust
  unsafe fn apply_s_times(
      psi_dev: &CudaSlice<CudaComplex>,   // input ψ (n_pw × n_bands, col-major)
      spsi_dev: &mut CudaSlice<CudaComplex>, // output S·ψ (caller pre-copies psi into this)
      vnl_data: &VnlBatchData,
      n_bands: i32,
      n_pw: i32,
      blas: &BlasHandle,
      stream: &Arc<CudaStream>,
  ) -> Result<(), Error>
  ```
  Caller copies `psi_dev` into `spsi_dev` first (identity term), then calls this
  to accumulate the β·Q·β^H·ψ correction.

- **Reference**: `apply_s_inverse` at `chebyshev.rs:758–837` — same 3-gemm pattern,
  replace `s_inv_mat` with `q_matrix`, flip sign from `−1.0` to `+1.0` in the
  accumulation gemm.

- **Acceptance**: `cargo check` passes. No unit test needed at this stage — correctness
  is validated end-to-end by SC-1/SC-2.

---

### Group B: R-ChFSI recurrence

**TASK-B1** — Replace Chebyshev recurrence body with R-ChFSI Algorithm 3

- **Kind**: `direct`
- **File**: `src/eigensolver/chebyshev.rs`

  **B1-part-1: Add `lambda_min` to `SpectralBounds`**

  Add `pub lambda_min: f64` to `SpectralBounds` (line 237). Populate in **both**
  code paths where the struct is constructed:

  - **Lanczos path** (lines 1135–1193): capture the second element of the Lanczos
    result tuple (`ritz_min`) and use `lambda_min = ritz_min * 0.8` (multiplicative
    safety factor to ensure the estimate is strictly below the true minimum).
  - **Gershgorin path** (`compute_spectral_bounds`, line 281): `lambda_min = min_veff`
    (since T ≥ 0, λ₁(H) ≥ min(V_loc)). The `min_veff` parameter is already available;
    add `lambda_min` to the struct literal at line 314.

  Clamp: `lambda_min = lambda_min.min(eps_cut - 1e-3)`. This ensures `λ_min < eps_cut`
  even when both are negative (typical for DFT eigenvalues). Using `eps_cut * 0.95`
  would produce a value *larger* than `eps_cut` for negative values — e.g.,
  `eps_cut = −10` gives `−9.5 > −10`, violating the `λ_min < c` invariant required
  by `σ = e/(λ_min − c)` (`main.tex:597`).

  > **Source**: `main.tex:591` INPUTS list three distinct bounds — `λ_max`, `λ_min`
  > (full-spectrum bounds), and `λ_T` (wanted-spectrum upper bound). Line 597 uses
  > `λ_min` in `σ = e/(λ_min − c)`, NOT `λ_T`. Our `eps_cut` = `λ_T`. Conflating
  > the two gives `σ = −1.0` always — mathematically wrong.

  **B1-part-2: `band_scale_axpy` kernel + helpers**

  Add kernel to `CUDA_KERNEL_SRC` (before closing, line ~186):
  ```c
  // dst[b*n_pw + g] += alpha * src[b*n_pw + g] * scale[b]
  // Used for: Y·Λ_Y term (Step 3), S·X·Λ subtraction (Step 1), X·Λ_Y reconstruction (Step 4)
  extern "C" __global__ void band_scale_axpy(
      cuDoubleComplex* dst,
      const cuDoubleComplex* src,
      const double* scale,
      double alpha,
      int n_pw, int n_bands
  ) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = n_pw * n_bands;
      if (idx >= total) return;
      int b = idx / n_pw;
      double s = alpha * scale[b];
      dst[idx].x += s * src[idx].x;
      dst[idx].y += s * src[idx].y;
  }
  ```

  Add `pub(crate) band_scale_axpy: CudaFunction` to `CudaKernelSet` struct and
  `band_scale_axpy: load("band_scale_axpy")?` to `CudaKernelSet::new()`.

  Rust wrapper:
  ```rust
  fn launch_band_scale_axpy(
      dst: &mut CudaSlice<CudaComplex>, src: &CudaSlice<CudaComplex>,
      scale_dev: &CudaSlice<f64>, alpha: f64,
      n_pw: usize, n_bands: usize,
      kernels: &CudaKernelSet, stream: &Arc<CudaStream>,
  ) -> Result<(), Error>
  ```

  GPU scalar upload helper:
  ```rust
  fn upload_f64_slice(v: &[f64], stream: &Arc<CudaStream>) -> Result<CudaSlice<f64>, Error>
  ```

  Pre-allocate `lam_y_dev` once (length `n_bands`) before the recurrence loop. On each
  step, update in-place via `stream.memcpy_htod(&lam_y, &mut lam_y_dev)` — do NOT
  re-allocate on every step (avoids 10–20 small GPU allocations per filter call).

  **B1-part-3: Recurrence body replacement**

  Replace lines 1200–1301 (the recurrence body). Everything before (spectral bounds,
  lines 1128–1198) and after (Gram-Schmidt, lines 1308–1356; final H|ψ>, lines
  1358–1380) stays unchanged.

  **Buffer strategy**: Replace `buf_a/buf_b/buf_c` with purpose-named buffers. Reuse
  `buf_a` and `buf_c` from the existing allocation for swap roles (saves 1 buffer):
  ```rust
  let mut buf_y   = stream.alloc_zeros(n_elem)...;  // Y = H·X − S·X·Λ, persisted
  let mut buf_sx  = stream.alloc_zeros(n_elem)...;  // S·X for residual computation
  let mut buf_rx  = stream.alloc_zeros(n_elem)...;  // R_X (swap role with buf_c)
  let mut buf_ry  = stream.alloc_zeros(n_elem)...;  // R_Y (swap role)
  // buf_c: reused for R_new computation in recurrence, then swap with buf_ry
  // buf_a: reused for X_new reconstruction at Step 4
  // hpsi_dev, grid_dev: kept as-is
  ```

  **Spectral parameters** (computed inline, source: `main.tex:597`):
  ```rust
  let e     = bounds.half_width;                       // = (λ_max − λ_T)/2
  let c     = bounds.center;                           // = (λ_max + λ_T)/2
  let sigma = e / (bounds.lambda_min - c);             // σ = e/(λ_min − c)
  let sigma1 = sigma;
  let gamma  = 2.0 / sigma1;
  ```

  **Step 1 — Initial residual Y = H·X − S·X·Λ** (source: `main.tex:598`):
  ```
  hpsi_dev = H·psi_input                    (apply_full_hamiltonian)
  buf_sx   = psi_input (copy)               (memcpy_dtod)
  apply_s_times(psi_input, &mut buf_sx)     (buf_sx = S·X = X + β·Q·β^H·X)
  buf_y    = hpsi_dev (copy)                (memcpy_dtod)
  if eigenvalues is Some:
      launch_band_scale_axpy(&mut buf_y, &buf_sx, lambda_dev, alpha=-1.0)
                                             (buf_y -= buf_sx * λ[b])
  // else eigenvalues=None → Λ=0 → Y = H·X already in buf_y
  ```

  **Step 2 — Initialize recurrence** (source: `main.tex:599-600`):
  ```
  buf_rx = zeros (already zero-allocated)    // R_X = 0
  buf_ry = (σ₁/e) · buf_y  (copy + zscal)    // R_Y = (σ₁/e)·Y
  lam_x = [1.0; n_bands]                     // Λ_X = I
  lam_y[b] = (σ₁/e) * (λ[b] − c)             // Λ_Y = (σ₁/e)·(Λ − c·I)
      or   (−σ₁·c/e) if eigenvalues is None
  ```

  **Step 3 — Recurrence for k = 2..=ndeg** (source: `main.tex:601-606`):
  ```
  σ₂    = 1 / (γ − σ)
  coeff = 2·σ₂ / e                              // = 2σ₂/e
  coeff_c = -coeff * c                           // = −2σ₂·c/e
  sigma_sigma2 = -sigma * σ₂                     // = −σ·σ₂

  // H·D⁻¹·R_Y (source: main.tex:603, first term):
  buf_c = buf_ry (copy)        (memcpy_dtod)
  apply_s_inverse(&mut buf_c)  (buf_c = S⁻¹·R_Y)
  apply_full_hamiltonian(buf_c → hpsi_dev)  (hpsi_dev = H·S⁻¹·R_Y)

  // R_new = (2σ₂/e)·H·S⁻¹·R_Y − (2σ₂/e)·c·R_Y − σ·σ₂·R_X + (2σ₂/e)·Y·Λ_Y
  buf_c = coeff * hpsi_dev    (copy hpsi_dev → buf_c, then zscal by coeff)
  blas::axpy(&mut buf_c, coeff_c, &buf_ry)          // − (2σ₂·c/e) · R_Y
  blas::axpy(&mut buf_c, sigma_sigma2, &buf_rx)      // − σ·σ₂ · R_X
  launch_band_scale_axpy(&mut buf_c, &buf_y, lam_y_dev, coeff)
                                                      // + (2σ₂/e) · Y · Λ_Y

  // Λ_X_new on CPU (source: main.tex:604):
  new_lam_x[b] = coeff*lam_y[b]*lambda[b] + coeff_c*lam_y[b] + sigma_sigma2*lam_x[b]

  // Buffer rotation (source: main.tex:605, swap(R_X,R_Y); swap(Λ_X,Λ_Y)):
  std::mem::swap(&mut buf_rx, &mut buf_ry);   // buf_rx ← old_R_Y, buf_ry ← old_R_X
  std::mem::swap(&mut buf_ry, &mut buf_c);    // buf_ry ← R_new, buf_c ← old_R_X
  lam_x = std::mem::replace(&mut lam_y, new_lam_x);
      // impl of main.tex:604-605: Λ_X←new, then swap(Λ_X,Λ_Y)
      // Effect: lam_x ← old Λ_Y, lam_y ← new Λ_X

  // Update Λ_Y on GPU in-place (pre-allocated, no re-allocation):
  memcpy_htod(&lam_y, &mut lam_y_dev)?;

  sigma = sigma2

  // Norm check on the newly computed residual R_Y (buf_ry after swap rotation):
  // After the double-swap: buf_rx = previous-step R_Y, buf_ry = new R_X (now called R_Y)
  norm_curr = frobenius(&buf_ry);
  norm_prev = frobenius(&buf_rx);
  check_norm_stability(norm_curr, norm_prev, k)?;
  ```

  **Step 4 — Reconstruct X_new** (source: `main.tex:607` `X ← D⁻¹·R_Y + X·Λ_Y`):
  ```
  buf_a = buf_ry (copy)          (memcpy_dtod)
  apply_s_inverse(&mut buf_a)    (buf_a = S⁻¹·R_Y ≡ D⁻¹·R_Y)
  launch_band_scale_axpy(&mut buf_a, &psi_input, lam_y_dev, alpha=1.0)
                                 (buf_a += X · Λ_Y)
  ```
  `buf_a` now holds X_new. Set `final_psi_buf = &mut buf_a`.

  **ndeg == 0 guard**: The recurrence loop never executes and Step 4 is skipped —
  `buf_a` remains zero-allocated. Add an explicit guard to prevent Gram-Schmidt
  from orthonormalizing a zero matrix:
  ```rust
  if ndeg == 0 {
      stream.memcpy_dtod(&psi_input, &mut buf_a)?;
      // skip directly to Gram-Schmidt
  } else {
      // Steps 1–4 as described above
  }
  ```

  **Fallback (eigenvalues=None, first SCF iter)**: Λ=0 → Y=H·X (skip S·X computation
  in Step 1). Λ_Y = −σ₁·c/e (constant across all bands). All other steps identical.

  > Source: `main.tex:612` — "When D⁻¹ = B⁻¹ ... ChFSI and R-ChFSI are algebraically
  > equivalent." With Λ=0, R-ChFSI's residual Y = H·X coincides with the standard ChFSI
  > starting vector. The first SCF iteration is near-converged (wavefunctions loaded
  > from `.check`), so the residual is small and the method is well-behaved.

- **What stays unchanged**:
  - Spectral bounds computation (lines 1128–1198) — only adds `lambda_min` population
  - Gram-Schmidt orthonormalization (lines 1308–1356) — unchanged
  - Final `H|psi>` computation for Rayleigh-Ritz (lines 1358–1380) — unchanged

- **Acceptance commands**:
  ```bash
  cd /home/tony/programming/chemrust-scf && cargo check --workspace 2>&1
  cd /home/tony/programming/chemrust-scf && cargo clippy --workspace -- -D warnings 2>&1
  # Run the discriminator test (GPU, ~8 min):
  cd /home/tony/programming/chemrust-scf && cargo test --release -- --ignored iter2_v_eff_range_within_one_ha_of_iter1 2>&1 | tail -20
  ```

---

### Group C: Spectral bounds for S⁻¹·H

**TASK-C1** — Update Lanczos to estimate λ_max of S⁻¹·H (not H)

- **Kind**: `direct`
- **File**: `src/eigensolver/chebyshev.rs`
- **What**: The Lanczos estimator (`lanczos_upper_bound`, called at line 1135)
  currently estimates λ_max of H. R-ChFSI filters S⁻¹·H, whose spectrum is the
  generalized eigenvalue problem H·ψ = ε·S·ψ. The spectral bounds should bound
  the generalized eigenvalues, not the standard ones.

  The simplest correct fix: after computing `hpsi = H·v` in the Lanczos kernel,
  apply `apply_s_inverse` to get `S⁻¹·H·v`. This makes the Lanczos Ritz values
  estimate λ_max(S⁻¹·H) = λ_max(generalized).

  **Scope**: Modify `lanczos_upper_bound` to accept `vnl_data` and apply
  `apply_s_inverse` after each H·v application. The function signature currently
  does not take `vnl_data` — add it.

  Current signature (`chebyshev.rs` around line 335):
  ```rust
  unsafe fn lanczos_upper_bound(
      v_eff_dev: &CudaSlice<f64>,
      kinetic_dev: &CudaSlice<f64>,
      fft_idx_dev: &CudaSlice<i32>,
      n_pw: usize, grid_size: usize, inv_ntotal: f64,
      ngx: usize, ngy: usize, ngz: usize,
      vnl_data: &VnlBatchData,   // already present for V_NL
      blas: &BlasHandle,
      kernels: &CudaKernelSet,
      stream: &Arc<CudaStream>,
      k_steps: usize,
  ) -> Result<(f64, f64, f64), Error>
  ```
  `vnl_data` is already passed (for V_NL in `apply_full_hamiltonian`). Add
  `apply_s_inverse` call after `apply_full_hamiltonian` inside the Lanczos loop.
  The Lanczos vector is a single band (n_bands=1), so `apply_s_inverse` with
  `n_bands=1` works directly.

- **Note**: This task is lower priority than TASK-B1. If R-ChFSI already converges
  with the current H-based bounds (Gershgorin cap), skip this task. The Gershgorin
  bound is conservative but safe — it overestimates λ_max, which widens the filter
  interval but does not break correctness.

- **Acceptance**: `cargo check --workspace` passes.

---

## Dependency Order

```
TASK-A1 (apply_s_times)
    ↓
TASK-B1 (R-ChFSI recurrence)  ← primary deliverable
    ↓
TASK-C1 (Lanczos for S⁻¹·H)  ← optional, only if B1 doesn't converge
```

## Verification Protocol

1. `cargo check --workspace` — must pass after each task
2. `cargo clippy --workspace -- -D warnings` — must pass after B1
3. `cargo test --release -- --ignored iter2_v_eff_range_within_one_ha_of_iter1`
   — primary discriminator (SC-1). Must stay green.
4. `cargo test --release -- --ignored fixed_point_matches_castep_energy`
   — convergence gate (SC-2). Expected to improve from diverging to converging.

## Key File References

| File | Lines | Purpose |
|------|-------|---------|
| `src/eigensolver/chebyshev.rs` | 237–244 | `SpectralBounds` struct — add `lambda_min` field |
| `src/eigensolver/chebyshev.rs` | 281–320 | `compute_spectral_bounds` — add `lambda_min = min_veff` to Gershgorin path |
| `src/eigensolver/chebyshev.rs` | 335–475 | `lanczos_upper_bound` — captures `lambda_min_tk` (already computed, currently discarded) |
| `src/eigensolver/chebyshev.rs` | 758–837 | `apply_s_inverse` — template for `apply_s_times` |
| `src/eigensolver/chebyshev.rs` | 1058–1077 | `chebyshev_filter` signature |
| `src/eigensolver/chebyshev.rs` | 1135–1198 | Lanczos call site — populate `lambda_min` from `ritz_min * 0.8` |
| `src/eigensolver/chebyshev.rs` | 1200–1301 | Current recurrence body — replace with R-ChFSI Algorithm 3 |
| `src/eigensolver/chebyshev.rs` | 1308–1356 | Gram-Schmidt — keep unchanged |
| `src/eigensolver/chebyshev.rs` | 1358–1380 | Final H\|psi> for RR — keep unchanged |
| `src/eigensolver/vnl_data.rs` | 18–34 | `VnlIonData` / `VnlBatchData` struct fields |
| `reference_paper/extracted/das-2025-rchfsi/main.tex` | 586–610 | Algorithm 3 (R-ChFSI) — full pseudocode |
| `reference_paper/extracted/das-2025-rchfsi/main.tex` | 591 | INPUTS: three distinct bounds (`λ_max`, `λ_min`, `λ_T`) |
| `reference_paper/extracted/das-2025-rchfsi/main.tex` | 597 | `σ = e/(λ_min − c)` — uses `λ_min`, NOT `λ_T` |
| `reference_paper/extracted/das-2025-rchfsi/main.tex` | 612 | Equivalence to standard ChFSI when D⁻¹ = B⁻¹ |
| `reference_paper/ALGORITHM_RATIONALE.md` | — | Full rationale and source map |
