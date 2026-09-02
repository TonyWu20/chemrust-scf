// ---------------------------------------------------------------------------
// Kerker mixing preconditioner: K(G) = G² / (G² + q²)
// ---------------------------------------------------------------------------
//
// Precomputed on the reciprocal-space FFT grid and stored on GPU.
//
//     K(G=0)  = 0.0    — no DC component mixing (total charge conservation)
//     K(G→∞)  → 1.0    — short wavelengths pass through
//     q       = 1.5 a.u. (fixed for Phase 2; CASTEP autocomputes from TF
//                         screening length)
//
// The kernel is laid out in GPU memory matching cuFFT conventions for a
// C2C (full grid) plan of dimensions (ngz, ngy, ngx).  For each grid point:
//
//     offset = ix * ngy * ngz + iy * ngz + iz
//
// where (ix, iy, iz) are the grid indices and iz is the fastest-varying
// dimension (matching cuFFT plan_3d(ngz, ngy, ngx) ordering).

use std::sync::Arc;

use chemrust_hamiltonian_core::GVectorGrid;
use cudarc::driver::{CudaSlice, CudaStream};

use crate::types::Error;

/// Kerker mixing preconditioner K(G) = G² / (G² + q²).
///
/// - K(G=0) = 0.0 (DC component excluded — total charge conservation)
/// - K(G) → 1.0 as |G| → ∞
/// - q = mix_charge_gmax (CASTEP default 1.5 /Å = 2.8346 a₀⁻¹), matching
///   CASTEP `dm_assign_plane_wave_indices`:
///   `energy_ch_q0sq = 0.5·mix_charge_gmax²` and kernel
///   `mix_charge_amp/(1 + E_q0/E)` = amp·G²/(G² + gmax²) with E = G²/2.
#[derive(Debug)]
pub struct KerkerPreconditioner {
    pub(crate) kernel: CudaSlice<f64>,
    pub(crate) shape: [usize; 3],  // [ngz, ngy, ngx]
    /// Mixing-basis mask: 1.0 for G components inside the CASTEP mix cutoff
    /// (|G|² ≤ g2_cutoff, including G=0), 0.0 above. Mirrors CASTEP's
    /// `num_mix_plane_waves` band-limit of the mix density object.
    pub(crate) mask: CudaSlice<f64>,
}

/// CASTEP default `mix_charge_gmax` = 1.5 /Å in a₀⁻¹
/// (`io_unit_to_atomic(1.5, "1/ang")`), see parameters.f90:1906.
pub const KERKER_GMAX_DEFAULT: f64 = 1.5 * 1.88972612545;

impl KerkerPreconditioner {
    /// Build the Kerker preconditioner on GPU from a `GVectorGrid`.
    ///
    /// The kernel is computed on CPU from `GVectorGrid::g2()` (which already
    /// provides |G|² for every reciprocal grid point in the cuFFT-compatible
    /// Fortran layout) and then transferred to GPU memory.
    ///
    /// `g2_cutoff`: if `Some(max_g2)`, G-vectors with |G|² > max_g2 are set
    /// to K=0 (no mixing).  CASTEP's `dm_apply_kerker` only mixes up to
    /// `num_mix_plane_waves` (= G-vectors within `mix_charge_gmax`, typically
    /// the wave-function cutoff at 380 eV).  On the fine grid, high-frequency
    /// G-vectors beyond this cutoff carry numerical noise and must be excluded
    /// from mixing.  Pass `gvg_wave.g2().iter().cloned().fold(0.0, f64::max)`
    /// when building the Kerker kernel on the fine grid.
    pub fn new(
        stream: &Arc<CudaStream>,
        gvg: &GVectorGrid,
        g2_cutoff: Option<f64>,
        gmax: f64,
    ) -> Result<Self, Error> {
        let shape = gvg.grid();  // [ngz, ngy, ngx]
        // CASTEP kernel: K(G) = G²/(G² + gmax²), gmax = mix_charge_gmax in a₀⁻¹.
        // (energy_ch_q0sq = 0.5·gmax²; kernel = amp/(1 + E_q0/E), E = G²/2.)
        let q2 = gmax * gmax;

        // Build kernel on CPU.
        // g2() returns &Array3<f64> in Fortran layout (ngz, ngy, ngx).
        // .iter() visits elements in Fortran memory order: iz fastest,
        // iy middle, ix slowest.  This is exactly the same layout as cuFFT
        // plan_3d(ngz, ngy, ngx) offset = ix * ngy * ngz + iy * ngz + iz,
        // so the flattened Vec is directly usable as a cuFFT-compatible kernel.
        let kernel_host: Vec<f64> = gvg.g2().iter().map(|&g2_val| {
            if g2_val == 0.0 {
                // G = 0: no DC component mixing
                0.0
            } else if let Some(cut) = g2_cutoff {
                if g2_val > cut {
                    0.0  // beyond mixing cutoff — CASTEP num_mix_plane_waves
                } else {
                    g2_val / (g2_val + q2)
                }
            } else {
                g2_val / (g2_val + q2)
            }
        }).collect();

        // Mixing-basis mask: 1.0 where the G component is in CASTEP's mix
        // density object (G2/2 <= mix_ecut).  mix_cut_off_energy defaults to
        // cut_off_energy (the wave cutoff) when unset in the .param, so the
        // external g2_cutoff (wave-grid G2 max) sets the mix/carry split.
        // Components above the cutoff are NOT mixed; their content is carried
        // from the fresh output density (dm_mix_density_to_density).
        let mask_host: Vec<f64> = gvg.g2().iter().map(|&g2_val| {
            if g2_val == 0.0 {
                1.0
            } else if let Some(cut) = g2_cutoff {
                if g2_val <= cut { 1.0 } else { 0.0 }
            } else {
                1.0
            }
        }).collect();
        let mask = stream
            .clone_htod(&mask_host)
            .map_err(Error::Cuda)?;

        // Transfer to GPU
        let kernel = stream
            .clone_htod(&kernel_host)
            .map_err(Error::Cuda)?;

        Ok(Self { kernel, mask, shape })
    }

    /// Access the precomputed kernel on GPU for use in CUDA kernels.
    pub fn as_device_slice(&self) -> &CudaSlice<f64> {
        &self.kernel
    }

    /// Mixing-basis mask (1.0 = G component inside the CASTEP mix cutoff,
    /// incl. G=0; 0.0 = high-frequency content carried from the fresh
    /// density, not mixed).
    pub fn as_mask_slice(&self) -> &CudaSlice<f64> {
        &self.mask
    }

    /// The grid shape `[ngz, ngy, ngx]`.
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Number of elements in the kernel (ngz × ngy × ngx).
    pub fn len(&self) -> usize {
        self.kernel.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kernel.is_empty()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    /// Kerker formula: K(G) = G² / (G² + q²).
    /// K(G=0) must be zero (no DC mixing, total charge conservation).
    /// K(G→∞) approaches 1.0.
    const Q2: f64 = 2.25; // q = 1.5 a.u.

    fn kerker_formula(g2: f64) -> f64 {
        if g2 == 0.0 { 0.0 } else { g2 / (g2 + Q2) }
    }

    #[test]
    fn test_kerker_g0_zero() {
        assert_eq!(kerker_formula(0.0), 0.0,
            "K(G=0) must be zero to conserve total charge");
    }

    #[test]
    fn test_kerker_monotonic() {
        let g2_vals = [0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
        for w in g2_vals.windows(2) {
            let k1 = kerker_formula(w[0]);
            let k2 = kerker_formula(w[1]);
            assert!(k2 > k1,
                "K(G²) must be monotonic: K({})={} < K({})={}",
                w[0], k1, w[1], k2);
        }
    }

    #[test]
    fn test_kerker_high_g_approaches_one() {
        for &g2 in &[1e3, 1e6, 1e12] {
            let k = kerker_formula(g2);
            let diff = (k - 1.0).abs();
            // 1 - K(G) = q²/(G²+q²) ≈ q²/G² for large G
            let expected_diff = Q2 / (g2 + Q2);
            assert!(diff < expected_diff * 1.1,
                "K({g2}) = {k}, diff={diff}, expected_diff~{expected_diff}");
        }
    }

    #[test]
    fn test_kerker_fixed_q_1p5() {
        // At G² = q², K = 0.5
        let k = kerker_formula(Q2);
        assert!((k - 0.5).abs() < 1e-15,
            "K(q²) should be 0.5, got {k}");
    }
}
