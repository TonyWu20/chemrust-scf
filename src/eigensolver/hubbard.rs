// ---------------------------------------------------------------------------
// DFT+U Hubbard potential (GPU) — pre-augmented projector path
// ---------------------------------------------------------------------------
//
// Implements the per-band DFT+U potential application using pre-augmented
// LCAO projectors (S|φ_n>) to eliminate S-inner products and S-augmentation
// from the hot path.  Mathematically equivalent to CASTEP's
// nlxc_ldau_apply_band + nlpot_apply_S_band.
//
// Design: docs/design/dft-plus-u-design.md §3 (GPU kernel design)
// Audit:  docs/design/dft-plus-u-audit-report.md

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use crate::device::blas::{self, BlasHandle, ZgemmConfig};
use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::PwCoefficients;
use crate::eigensolver::hamiltonian::apply_s_times;
pub(crate) use crate::eigensolver::hubbard_types::HubbardBatchData;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;
use bon::builder;

// ---------------------------------------------------------------------------
// Pre-computation: aug_lcao = S · lcao (once per k-point)
// ---------------------------------------------------------------------------

/// Compute `aug_lcao = S · lcao` via the existing `apply_s_times`,
/// reusing `VnlBatchData` for the β·Q·β^H correction.
///
/// Called lazily from `ldau_ffi_upload_u` after `VnlBatchData` is available
/// (i.e. after the first `step_inner` call has populated per-kpt VNL data).
///
/// Takes **ownership** of both `CudaSlice`s to avoid `CudaSlice::clone()`,
/// which performs a D2D deep copy (cudarc 0.19.7 `clone()` → `try_clone()` →
/// `stream.clone_dtod()`).  Returns both slices so the caller can store
/// them back into `HubbardBatchData`.
///
/// The caller is responsible for pre-copying `lcao_coeffs_dev` into
/// `aug_lcao_dev` (identity term of `S = I + Σ β·Q·β^H`) via `memcpy_dtod`
/// **before** calling this function.  This function accumulates the
/// β·Q·β^H·lcao correction into `aug_lcao_dev`.
pub(crate) unsafe fn compute_aug_lcao(
    lcao_coeffs_dev: CudaSlice<CudaComplex>,
    aug_lcao_dev: CudaSlice<CudaComplex>,
    vnl_data: &VnlBatchData,
    n_orb_total: usize,
    n_pw: usize,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(CudaSlice<CudaComplex>, CudaSlice<CudaComplex>), Error> {
    // Build PwCoefficients from the OWNED CudaSlices — no clone needed.
    let lcao_pw = PwCoefficients::new(lcao_coeffs_dev);
    let mut aug_pw = PwCoefficients::new(aug_lcao_dev);
    unsafe {
        apply_s_times()
            .psi_dev(&lcao_pw)
            .spsi_dev(&mut aug_pw)
            .vnl_data(vnl_data)
            .n_bands(n_orb_total as i32)
            .n_pw(n_pw as i32)
            .blas(blas)
            .stream(stream)
            .call()?;
    }
    // Extract the CudaSlices back from PwCoefficients and return both.
    // lcao_pw.0 is unchanged (apply_s_times only reads from psi_dev).
    // aug_pw.0 now contains S·lcao = lcao + β·Q·β^H·lcao.
    Ok((lcao_pw.0, aug_pw.0))
}

// ---------------------------------------------------------------------------
// Hot-path: apply V_U^AE |psi> (every Hamiltonian application)
// ---------------------------------------------------------------------------

/// Apply `V_U^AE |psi>` and accumulate into `hpsi`.
///
/// Uses pre-augmented LCAO projectors (`S|φ_n>`) so no S-operator is
/// needed in the hot path.  Mathematically equivalent to CASTEP's
/// `nlxc_ldau_apply_band` + `nlpot_apply_S_band`, but the S-augmentation
/// is absorbed into the projectors.
///
/// ## Steps (design doc §3.2)
///
/// **H1:** `T = psi^H · aug_lcao` — cuBLAS ZGEMM (op::C × op::N).
/// Plain dot product — no S-inner product because aug_lcao carries S.
///
/// **H3:** `W = conjg(T) · UNnm^T` — computed on CPU rather than cuBLAS
/// ZGEMM.  n_bands × n_orb is tiny (~260 elements for NiO, ~2600 FLOPs),
/// well below GPU kernel launch overhead.  The CPU triple loop is
/// algebraically equivalent to ZGEMM with transb=T for real-symmetric
/// UNnm (the only case in CASTEP 6.11: N_matrix is real for collinear
/// systems, so UNnm = U·(½δ - N) is real-symmetric).
/// Formula matches CASTEP `nlxc.f90:3991`:
///   weight(n) = Σ_m UNnm(n, m, ns) · conjg(T(m))
/// equivalent to W(b, j) = Σ_i conjg(T(b, i)) · UNnm(i, j, ns).
///
/// **H4:** `hpsi += aug_lcao · W^T` — cuBLAS ZGEMM (op::N × op::T, β=1.0).
/// Accumulates Hubbard action into existing hpsi.
///
/// **H2 (eigenvalues):** → Deferred to Phase 7B.
/// Per-band Hubbard eigenvalues `eig_U(b)` are not needed for Gate 1
/// (augmented projector equivalence) or for the Davidson Rayleigh-Ritz
/// path (where Hubbard is baked into hpsi).  They will be implemented
/// with the standalone `ldau_ffi_apply` diagnostic wrapper in Phase 7B.
///
/// ## Constraints
///
/// - `n_bands ≤ hubbard.max_n_bands` (asserted).
/// - VCA safety gate: aborts if any ion has mixture_weight ≠ 1.0
///   (audit F-C2; `apply_s_times` does not yet handle VCA scaling).
/// - `n_bands` is a call-time parameter reflecting actual band count,
///   which varies between outer loop (~62) and inner loop (~10) as
///   proven by the NiO profile.
///
/// ## Safety
///
/// `psi_dev`, `hpsi_dev`, and `hubbard.aug_lcao_dev` must all be valid
/// GPU buffers with the correct shapes.
#[builder]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_v_u_hamiltonian(
    psi_dev: &PwCoefficients,
    hpsi_dev: &mut PwCoefficients,
    hubbard: &HubbardBatchData,
    n_bands: i32,
    ns: i32,
    blas: &BlasHandle,
    stream: &Arc<CudaStream>,
) -> Result<(), Error> {
    let aug_lcao = hubbard
        .aug_lcao_dev
        .as_ref()
        .ok_or_else(|| Error::Io("aug_lcao not yet computed".to_string()))?;

    // ---- Bounds check ----
    assert!(
        n_bands as usize <= hubbard.max_n_bands,
        "n_bands {} exceeds max_n_bands {}",
        n_bands, hubbard.max_n_bands
    );

    // ---- VCA Safety Gate (Audit F-C2) ----
    // `apply_s_times` does not yet apply mixture_weight scaling to the
    // S-augmentation of LCAO projectors.  For single-species systems
    // (mixture_weight ≡ 1.0) this is harmless; for VCA it produces wrong
    // Hubbard potentials.  Abort if any ion has non-unity mixture_weight.
    if !hubbard.mixture_weights.is_empty() {
        for (i, &w) in hubbard.mixture_weights.iter().enumerate() {
            if (w - 1.0_f64).abs() > 1e-12 {
                return Err(Error::Io(format!(
                    "VCA not yet supported: ion {} has mixture_weight={} (!= 1.0). \
                     See design doc §4.6 and audit F-C2.",
                    i, w
                )));
            }
        }
    }

    let n_pw = hubbard.n_pw as i32;
    let n_orb = hubbard.n_orb_total as i32;
    let nb = n_bands as usize;
    let no = n_orb as usize;

    // ---- H1: T_matrix = psi^H · aug_lcao  (n_bands × n_orb) ----
    //
    // cuBLAS ZGEMM: C = A^H · B
    //   A = psi (n_pw × n_bands), transa=C → (n_bands × n_pw)
    //   B = aug_lcao (n_pw × n_orb), transb=N
    //   C = t_matrix (n_bands × n_orb), column-major, ldc=n_bands
    let mut t_matrix: CudaSlice<CudaComplex> =
        stream.alloc_zeros(nb * no).map_err(Error::Cuda)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: blas::op::C,
                transb: blas::op::N,
                m: n_bands,
                n: n_orb,
                k: n_pw,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                beta: CudaComplex { x: 0.0, y: 0.0 },
                lda: n_pw,
                ldb: n_pw,
                ldc: n_bands,
            },
            psi_dev,
            aug_lcao,
            &mut t_matrix,
        )?;
    }

    // ---- H3: W = conjg(T) · UNnm^T  (CPU side, tiny matrices) ----
    //
    // Layout conventions:
    //   t_host: column-major, ldc=n_bands → T(b, i) = t_host[b + i*n_bands]
    //   unnm_host: row-major per spin → UNnm(i, j, ns) = unnm_host[ns*stride + i*no + j]
    //   w_host: column-major, ldc=n_bands → W(b, j) = w_host[b + j*n_bands]
    //
    // This avoids a custom GPU conjugate kernel and ZGEMM launch overhead
    // for what is ~2600 FLOPs in the worst case.
    let t_host: Vec<CudaComplex> = stream.clone_dtoh(&t_matrix).map_err(Error::Cuda)?;
    let stride = no * no;
    let spin_off = ns as usize * stride;
    let mut w_host = vec![CudaComplex { x: 0.0, y: 0.0 }; nb * no];
    for b in 0..nb {
        for j in 0..no {
            let mut acc = CudaComplex { x: 0.0, y: 0.0 };
            for i in 0..no {
                let t = t_host[b + i * nb]; // T(b, i), col-major
                let u = hubbard.unnm_host[spin_off + i * no + j]; // UNnm(i, j, ns)
                // acc += conjg(t) * u
                acc.x += t.x * u.x + t.y * u.y; // Re(conj(t)*u) = t.x*u.x + t.y*u.y
                acc.y += t.x * u.y - t.y * u.x; // Im(conj(t)*u) = t.x*u.y - t.y*u.x
            }
            w_host[b + j * nb] = acc; // W(b, j), col-major
        }
    }
    let mut w_dev: CudaSlice<CudaComplex> =
        stream.alloc_zeros(nb * no).map_err(Error::Cuda)?;
    stream
        .memcpy_htod(&w_host, &mut w_dev)
        .map_err(Error::Cuda)?;

    // ---- H4: hpsi += aug_lcao · W^T  (accumulate, beta=1.0) ----
    //
    // cuBLAS ZGEMM: C = A · B^T  (+ accumulate into existing C)
    //   A = aug_lcao (n_pw × n_orb), transa=N
    //   B = W (n_bands × n_orb), transb=T → (n_orb × n_bands)
    //   C = hpsi (n_pw × n_bands), column-major, ldc=n_pw, beta=1.0 (accumulate)
    //
    // This computes hpsi(p, b) += Σ_j aug_lcao(p, j) · W(b, j)
    // which matches CASTEP's per-band Hubbard potential accumulation.
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: blas::op::N,
                transb: blas::op::T,
                m: n_pw,
                n: n_bands,
                k: n_orb,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                beta: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw,
                ldb: n_bands,
                ldc: n_pw,
            },
            aug_lcao,
            &w_dev,
            hpsi_dev,
        )?;
    }

    Ok(())
}
