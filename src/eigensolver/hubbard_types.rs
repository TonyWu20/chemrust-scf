// ---------------------------------------------------------------------------
// DFT+U Hubbard types — data structures for GPU-resident Hubbard U data
// ---------------------------------------------------------------------------
//
// Design: docs/design/dft-plus-u-design.md §4 (Rust data structures)
// Audit:  docs/design/dft-plus-u-audit-report.md

use cudarc::driver::CudaSlice;

use crate::device::CudaComplex;

// ---------------------------------------------------------------------------
// HubbardBatchData — GPU-resident Hubbard U data for one k-point
// ---------------------------------------------------------------------------

/// GPU-resident Hubbard U data for one k-point configuration.
///
/// The augmented LCAO projectors `aug_lcao_dev` absorb the S-operator,
/// eliminating per-band S-inner products and S-augmentation from the hot path.
///
/// Pre-computation: `aug_lcao_dev = S · lcao_coeffs_dev` (once per k-point).
/// Hot path: `hpsi += aug_lcao · W^T` where `W = conjg(psi^H · aug_lcao) · UNnm^T`.
pub struct HubbardBatchData {
    /// Bare LCAO projector plane-wave coefficients: shape (n_pw, n_orb_total).
    /// Column-major (matching Fortran `lcao_basis%coeffs(:, orb, nk_lb, ns_lb)`).
    /// Uploaded once per k-point via ldau_ffi_upload_basis.
    pub lcao_coeffs_dev: CudaSlice<CudaComplex>,

    /// Augmented LCAO projectors S|lcao>: shape (n_pw, n_orb_total).
    /// Computed lazily via compute_aug_lcao — None until the first
    /// ldau_ffi_upload_u call triggers lazy computation (VnlBatchData
    /// must be available first).
    pub aug_lcao_dev: Option<CudaSlice<CudaComplex>>,

    /// UNnm effective Hubbard matrices:
    /// (n_spins * n_orb_total * n_orb_total) row-major per spin.
    /// Uploaded once per SCF cycle via ldau_ffi_upload_u.
    pub unnm_dev: CudaSlice<CudaComplex>,

    /// CPU copy of UNnm for H3 weight computation (tiny: max ~800 complex
    /// values for d-shell = 6.4 KB).  Used to avoid GPU kernel launch
    /// overhead for conjugate-and-multiply of n_bands × n_orb matrices.
    pub unnm_host: Vec<CudaComplex>,

    /// Per-channel starting orbital index (0-based).  Length = n_channels.
    /// CASTEP stores orb-1 at nlxc.f90:4607 — no conversion needed.
    pub channel_offset: Vec<usize>,

    /// Per-channel angular momentum quantum number l.  Length = n_channels.
    pub channel_l: Vec<i32>,

    /// Total number of LCAO orbitals across all channels.
    pub n_orb_total: usize,

    /// Number of Hubbard channels.
    pub n_channels: usize,

    /// Number of plane-wave coefficients for this k-point.
    pub n_pw: usize,

    /// Number of spins.  1 (non-spin-polarised) or 2 (collinear spin-polarised).
    /// CASTEP 6.11 does not support non-collinear Hubbard magnetism
    /// (UNnm hardcoded to dimension 2 at nlxc.f90:3456).
    pub n_spins: usize,

    /// Zeeman Lz matrix placeholder.  Allocated zero-sized in Phase 7A
    /// (Zeeman term is gated on external_field != 0).
    pub lz_matrix_dev: CudaSlice<CudaComplex>,

    /// Per-ion mixture_weight for VCA safety gate.
    ///
    /// Audit F-C2 / design §4.6 mandates a runtime assertion: if any ion has
    /// `mixture_weight != 1.0`, the Hubbard potential is incorrect because
    /// `apply_s_times` does not yet apply mixture_weight scaling to the
    /// S-augmentation of LCAO projectors.  Empty Vec = single-species (safe).
    ///
    /// Populated from CASTEP's `current_cell%mixture_weight(:, :)` via FFI
    /// in Phase 7B; for Phase 7A all test systems are single-species.
    pub mixture_weights: Vec<f64>,

    /// Maximum number of bands this k-point can hold (nbands_max from CASTEP).
    /// Asserted at call time in apply_v_u_hamiltonian: n_bands ≤ max_n_bands.
    pub(crate) max_n_bands: usize,
}
