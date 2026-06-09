// ---------------------------------------------------------------------------
// BetaPhiCache — persistent β^H·ψ projection cache across Davidson outer
// iterations (matches CASTEP's have_beta_phi caching pattern).
// ---------------------------------------------------------------------------
//
// CASTEP behaviour (wave.f90):
//   - have_beta_phi(:,:) is a 2D logical array indexed by (kpt, spin).
//   - After wave_beta_phi computes β^H·ψ, have_beta_phi(nk, ns) = .true.
//   - Invalidation happens per-(kpt,spin): when any band's ψ changes at a
//     given k-point/spin, the ENTIRE flag for that (kpt, spin) is set to
//     .false. (wave.f90 lines 1012, 1199, 1263, 17608).
//
// Rust deviation from CASTEP:
//   This implementation tracks per-band validity (band_valid[b]: bool) for
//   finer-grained cache reuse.  When only a subset of bands are modified
//   (e.g. during inner Davidson block copies), we invalidate only those
//   bands rather than discarding the entire cache.
//
// Lifecycle (one Davidson outer iteration):
//   1. Start of outer iter:  cache is fully valid from previous iter's
//      compute_all() call.
//   2. apply_full_hamiltonian() → apply_v_nl_hamiltonian() can read cached
//      β^H·ψ (Case 1), skipping per-ion ZGEMM.
//   3. After A1 full-subspace ZHEGVD: invalidate_all() — all bands rotated.
//   4. Inner block loop: after each block's A3 copy-back,
//      invalidate_bands(&modified) marks those bands stale.
//   5. End of outer iter: compute_all() recomputes β^H·ψ for the final ψ_dev
//      of this iteration, populating the cache for the NEXT outer iteration.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::CudaComplex;
use crate::eigensolver::davidson_types::PwCoefficients;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::types::Error;

/// Cached `β^H · ψ` for one ion: an `(n_expanded × n_bands)` complex matrix,
/// stored column-major (each column = one band's projection).
struct IonBetaEntry {
    projections: CudaSlice<CudaComplex>,
    n_expanded: i32,
}

/// Cache of `β^H · ψ` projections across Davidson outer iterations.
///
/// # CASTEP correspondence
///
/// | CASTEP (`wavefunction` type) | Rust `BetaPhiCache`      |
/// |-------------------------------|---------------------------|
/// | `have_beta_phi(nk, ns)`       | `band_valid[b]: bool`     |
/// | `beta_phi(:, nb, nk, ns)`     | `ion_entries[i].projections[:, nb]` |
/// | `wave_beta_phi` (compute)     | `compute_all()`           |
/// | `.false.` on alloc/change     | `invalidate_all()` / `invalidate_bands()` |
pub struct BetaPhiCache {
    ion_entries: Vec<IonBetaEntry>,
    band_valid: Vec<bool>,
    n_bands: usize,
}

impl BetaPhiCache {
    /// Allocate GPU storage for per-ion `β^H · ψ` projections.
    /// All bands initially invalid (CASTEP wave.f90:1012).
    pub fn new(
        vnl_data: &VnlBatchData,
        n_bands: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, Error> {
        let ion_entries: Vec<IonBetaEntry> = vnl_data
            .entries
            .iter()
            .map(|entry| {
                let ne = entry.n_expanded;
                let size = (ne as usize) * n_bands;
                let projections = stream
                    .alloc_zeros::<CudaComplex>(size)
                    .map_err(Error::Cuda)?;
                Ok(IonBetaEntry {
                    projections,
                    n_expanded: ne,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Self {
            ion_entries,
            band_valid: vec![false; n_bands],
            n_bands,
        })
    }

    /// Mark all cached bands as stale (CASTEP: have_beta_phi = .false.).
    pub fn invalidate_all(&mut self) {
        self.band_valid.fill(false);
    }

    /// Mark specific bands as stale.
    ///
    /// # Panics
    /// Panics if any band index exceeds `n_bands - 1`.
    pub fn invalidate_bands(&mut self, bands: &[usize]) {
        for &b in bands {
            assert!(
                b < self.n_bands,
                "BetaPhiCache::invalidate_bands: band {b} out of range (n_bands={})",
                self.n_bands,
            );
            self.band_valid[b] = false;
        }
    }

    /// Recompute `β^H · ψ` for all ions and all bands.
    /// Matches CASTEP `wave_beta_phi_wv_ks` → `ion_all_beta_multi_phi_recip`.
    ///
    /// # Safety
    /// Calls cuBLAS `gemm_c64` (unsafe FFI).
    pub unsafe fn compute_all(
        &mut self,
        psi_dev: &PwCoefficients,
        vnl_data: &VnlBatchData,
        n_pw: usize,
        blas: &BlasHandle,
        _stream: &Arc<CudaStream>,
    ) -> Result<(), Error> {
        let n_pw_i32 = n_pw as i32;
        let n_bands_i32 = self.n_bands as i32;

        for (ion_idx, entry) in vnl_data.entries.iter().enumerate() {
            let ion_cache = &mut self.ion_entries[ion_idx];
            let ne = ion_cache.n_expanded;

            // C = beta_g^H · psi (ne × n_bands)
            unsafe {
                blas.gemm_c64(
                    ZgemmConfig {
                        transa: op::C,
                        transb: op::N,
                        m: ne,
                        n: n_bands_i32,
                        k: n_pw_i32,
                        alpha: CudaComplex { x: 1.0, y: 0.0 },
                        lda: n_pw_i32,
                        ldb: n_pw_i32,
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                        ldc: ne,
                    },
                    &entry.beta_g,
                    psi_dev,
                    &mut ion_cache.projections,
                )
                .map_err(Error::Blas)?;
            }
        }

        self.band_valid.fill(true);
        Ok(())
    }

    /// Returns `true` if band `b` has valid cached projections.
    #[inline]
    pub fn is_valid(&self, band: usize) -> bool {
        self.band_valid[band]
    }

    /// Returns `true` when every band has valid cached projections.
    #[inline]
    pub fn are_all_valid(&self) -> bool {
        self.band_valid.iter().all(|&v| v)
    }

    /// Number of bands in this cache.
    #[inline]
    pub fn n_bands(&self) -> usize {
        self.n_bands
    }

    /// Reference to cached `β^H · ψ` for ion `ion_idx`.
    /// Returns `None` if `ion_idx` is out of range.
    #[inline]
    pub fn ion_projections(&self, ion_idx: usize) -> Option<(&CudaSlice<CudaComplex>, i32)> {
        self.ion_entries.get(ion_idx).map(|entry| {
            (&entry.projections, entry.n_expanded)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalidate_all() {
        let mut cache = BetaPhiCache {
            ion_entries: Vec::new(),
            band_valid: vec![true; 3],
            n_bands: 3,
        };
        assert!(cache.are_all_valid());
        cache.invalidate_all();
        assert!(!cache.are_all_valid());
    }

    #[test]
    fn test_invalidate_bands_partial() {
        let mut cache = BetaPhiCache {
            ion_entries: Vec::new(),
            band_valid: vec![true; 5],
            n_bands: 5,
        };
        cache.invalidate_bands(&[1, 3]);
        assert!(cache.is_valid(0));
        assert!(!cache.is_valid(1));
        assert!(cache.is_valid(2));
        assert!(!cache.is_valid(3));
        assert!(cache.is_valid(4));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_invalidate_bands_out_of_range() {
        let mut cache = BetaPhiCache {
            ion_entries: Vec::new(),
            band_valid: vec![true; 3],
            n_bands: 3,
        };
        cache.invalidate_bands(&[3]);
    }

    #[test]
    fn test_are_all_valid_empty() {
        let cache = BetaPhiCache {
            ion_entries: Vec::new(),
            band_valid: Vec::new(),
            n_bands: 0,
        };
        assert!(cache.are_all_valid());
    }
}
