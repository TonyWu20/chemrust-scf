// ---------------------------------------------------------------------------
// Kerker mixing preconditioners: K(G) = G² / (G² + q²)
// ---------------------------------------------------------------------------
//
// Precomputed on the reciprocal-space FFT grid and stored on GPU.
//
// Two kernels per G (CASTEP `kerker_matrix(num_pw, 1:2)`,
// dm_sub_base.f90:689-690):
//
//   charge:  Kc(G) = amp_c·G²/(G² + gc²),  Kc(G=0) = 0
//            (DC charge is not mixed — total charge conservation)
//   spin:    Ks(G) = amp_s·G²/(G² + gs²),  Ks(G=0) = amp_s
//            (uniform spin mixes fully — CASTEP sets the G=0 spin kernel
//            to mix_spin_amp explicitly, dm_sub_base.f90:681-682)
//
// The slices store the PURE kernels (amplitudes applied as scalars in the
// CUDA update): charge slice G=0 → 0.0, spin slice G=0 → 1.0, so that
// amp·slice reproduces CASTEP's `kerker_matrix` values at every G, G=0
// included.
//
//     K(G→∞)  → amp·1.0    — short wavelengths pass through
//     gc, gs   = mix_charge_gmax / mix_spin_gmax (a₀⁻¹; CASTEP default
//                1.5 /Å = 2.8346 a₀⁻¹ each, parameters.f90:1906-1918)
//
// The kernels are laid out in GPU memory matching cuFFT conventions for a
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

/// Kerker mixing preconditioners Kc(G), Ks(G) = G² / (G² + q²).
///
/// - charge slice: G=0 → 0.0 (DC charge excluded — charge conservation),
///   G>0 → G²/(G² + gc²)
/// - spin slice: G=0 → 1.0 (×amp_s gives CASTEP's Ks(0) = amp_s),
///   G>0 → G²/(G² + gs²)
/// - gc = mix_charge_gmax, gs = mix_spin_gmax (a₀⁻¹).
///
/// CASTEP `dm_assign_plane_wave_indices`:
///   `energy_ch_q0sq = 0.5·mix_charge_gmax²`,
///   `energy_sp_q0sq = 0.5·mix_spin_gmax²`,
///   kernel = amp/(1 + E_q0/E) = amp·G²/(G² + gmax²) with E = G²/2.
#[derive(Debug)]
pub struct KerkerPreconditioner {
    /// Pure charge kernel (×amp_c in the update). G=0 → 0.0.
    pub(crate) kernel: CudaSlice<f64>,
    /// Pure spin kernel (×amp_s in the update). G=0 → 1.0.
    /// Unused for nspins=1 (spin object is zero).
    pub(crate) spin_kernel: CudaSlice<f64>,
    pub(crate) shape: [usize; 3],  // [ngz, ngy, ngx]
    /// Mixing-basis mask: 1.0 for G components inside the CASTEP mix cutoff
    /// (|G|² ≤ g2_cutoff, including G=0), 0.0 above. Mirrors CASTEP's
    /// `num_mix_plane_waves` band-limit of the mix density object.
    pub(crate) mask: CudaSlice<f64>,
    /// DIIS inner-product weight for the charge part (CASTEP `mix_metric`):
    /// 1.0 + E_q1sq/E with q1 = 0 → 1.0 for G>0; CASTEP sets it to 0.0 at
    /// G=0 (dm_sub_base.f90:684). Equals `mask` with the G=0 entry zeroed.
    pub(crate) metric_mask: CudaSlice<f64>,
}

/// CASTEP default `mix_charge_gmax` = 1.5 /Å in a₀⁻¹
/// (`io_unit_to_atomic(1.5, "1/ang")`), see parameters.f90:1906.
pub const KERKER_GMAX_DEFAULT: f64 = 1.5 * 1.88972612545;
/// CASTEP default `mix_spin_gmax` = 1.5 /Å in a₀⁻¹ (parameters.f90:1916).
pub const KERKER_SPIN_GMAX_DEFAULT: f64 = KERKER_GMAX_DEFAULT;

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
    ///
    /// `gmax_c` / `gmax_s`: CASTEP `mix_charge_gmax` and `mix_spin_gmax` in
    /// a₀⁻¹ (both default to 1.5 /Å; NiO .param sets both to 1.5 /Å).
    pub fn new(
        stream: &Arc<CudaStream>,
        gvg: &GVectorGrid,
        g2_cutoff: Option<f64>,
        gmax_c: f64,
        gmax_s: f64,
    ) -> Result<Self, Error> {
        let shape = gvg.grid();  // [ngz, ngy, ngx]
        // CASTEP pure kernels (dm_sub_base.f90:689-690, E = G²/2):
        //   Kc(G) = G²/(G² + gc²), G=0 → 0
        //   Ks(G) = G²/(G² + gs²), G=0 → 1 (×amp_s gives Ks(0) = amp_s,
        //          CASTEP's explicit G=0 spin kernel, dm_sub_base.f90:681-682)
        let q2_c = gmax_c * gmax_c;
        let q2_s = gmax_s * gmax_s;

        // Build kernels on CPU.
        // g2() returns &Array3<f64> in Fortran layout (ngz, ngy, ngx).
        // .iter() visits elements in Fortran memory order: iz fastest,
        // iy middle, ix slowest.  This is exactly the same layout as cuFFT
        // plan_3d(ngz, ngy, ngx) offset = ix * ngy * ngz + iy * ngz + iz,
        // so the flattened Vec is directly usable as a cuFFT-compatible kernel.
        let mut kernel_host: Vec<f64> = Vec::with_capacity(gvg.g2().len());
        let mut spin_kernel_host: Vec<f64> = Vec::with_capacity(gvg.g2().len());
        let mut mask_host: Vec<f64> = Vec::with_capacity(gvg.g2().len());
        let mut metric_mask_host: Vec<f64> = Vec::with_capacity(gvg.g2().len());
        for &g2_val in gvg.g2().iter() {
            let in_mix_basis = match g2_cutoff {
                Some(cut) => g2_val == 0.0 || g2_val <= cut,
                None => true,
            };
            // CASTEP pure charge kernel (dm_sub_base.f90:689):
            // Kc(G) = G2/(G2 + gc2); G=0 -> 0.
            let kernel = if g2_val == 0.0 {
                0.0
            } else if in_mix_basis {
                g2_val / (g2_val + q2_c)
            } else {
                0.0  // beyond mixing cutoff — CASTEP num_mix_plane_waves
            };
            // CASTEP pure spin kernel: Ks(G) = G2/(G2 + gs2); the G=0 entry
            // is 1.0 so amp_s x 1.0 = CASTEP's Ks(0) = mix_spin_amp
            // (dm_sub_base.f90:681-682: "need to mix the G=0 spin").
            let spin_kernel = if g2_val == 0.0 {
                1.0
            } else if in_mix_basis {
                g2_val / (g2_val + q2_s)
            } else {
                0.0
            };
            // Mixing-basis mask: 1.0 where the G component is in
            // CASTEP's mix density object.  mix_cut_off_energy
            // defaults to cut_off_energy (the wave cutoff) when
            // unset in the .param, so the external g2_cutoff
            // (wave-grid G2 max) sets the mix/carry split.
            // Components above the cutoff are NOT mixed; their
            // content is carried from the fresh output density
            // (dm_mix_density_to_density).
            let mask = if in_mix_basis { 1.0 } else { 0.0 };
            // DIIS charge inner-product weight (CASTEP mix_metric,
            // dm_sub_base.f90:684,687): G=0 -> 0.0, G>0 -> 1.0 +
            // E_q1sq/E with q1 = 0 -> 1.0.
            let metric_mask = if g2_val == 0.0 { 0.0 } else { mask };
            kernel_host.push(kernel);
            spin_kernel_host.push(spin_kernel);
            mask_host.push(mask);
            metric_mask_host.push(metric_mask);
        }
        let mask = stream
            .clone_htod(&mask_host)
            .map_err(Error::Cuda)?;
        let metric_mask = stream
            .clone_htod(&metric_mask_host)
            .map_err(Error::Cuda)?;
        let spin_kernel = stream
            .clone_htod(&spin_kernel_host)
            .map_err(Error::Cuda)?;

        // Transfer to GPU
        let kernel = stream
            .clone_htod(&kernel_host)
            .map_err(Error::Cuda)?;

        Ok(Self { kernel, spin_kernel, shape, mask, metric_mask })
    }

    /// Access the precomputed charge kernel on GPU for use in CUDA kernels.
    pub fn as_device_slice(&self) -> &CudaSlice<f64> {
        &self.kernel
    }

    /// Access the precomputed spin kernel on GPU (×amp_s in the update;
    /// G=0 entry is 1.0 → Ks(0) = amp_s, CASTEP's full DC spin mixing).
    pub fn as_spin_kernel_slice(&self) -> &CudaSlice<f64> {
        &self.spin_kernel
    }

    /// Mixing-basis mask (1.0 = G component inside the CASTEP mix cutoff,
    /// incl. G=0; 0.0 = high-frequency content carried from the fresh
    /// density, not mixed).
    pub fn as_mask_slice(&self) -> &CudaSlice<f64> {
        &self.mask
    }

    /// DIIS inner-product weight for the charge part (CASTEP `mix_metric`):
    /// mask with the G=0 entry zeroed (G>0 → 1.0 for q1 = 0).
    pub fn as_metric_mask_slice(&self) -> &CudaSlice<f64> {
        &self.metric_mask
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
    /// Pure charge kernel: Kc(G) = G² / (G² + q²).  K(G=0) must be zero
    /// (no DC charge mixing, total charge conservation).  K(G→∞) → 1.0.
    const Q2: f64 = 2.25; // q = 1.5 a.u.

    fn kerker_formula(g2: f64) -> f64 {
        if g2 == 0.0 { 0.0 } else { g2 / (g2 + Q2) }
    }

    /// Pure spin kernel: Ks(G) = G² / (G² + q²) for G>0; G=0 → 1.0 so the
    /// effective spin kernel Ks_eff(0) = amp_s × 1.0 = amp_s (CASTEP's
    /// explicit G=0 spin kernel value, dm_sub_base.f90:681-682).
    fn spin_kerker_formula(g2: f64) -> f64 {
        if g2 == 0.0 { 1.0 } else { g2 / (g2 + Q2) }
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

    #[test]
    fn test_spin_kerker_g0_full() {
        // G=0 uniform spin mixes fully: pure slice = 1.0, so the
        // effective kernel at G=0 equals the spin amplitude (amp_s × 1.0).
        assert_eq!(spin_kerker_formula(0.0), 1.0,
            "Ks(G=0) pure slice must be 1.0 (effective Ks(0) = amp_s)");
    }

    #[test]
    fn test_spin_kerker_g_positive() {
        // For G>0 the spin kernel equals the charge formula (same G² shape,
        // different G=0 value).  At G² = q² both equal 0.5.
        let k = spin_kerker_formula(Q2);
        assert!((k - 0.5).abs() < 1e-15, "Ks(q²) should be 0.5, got {k}");
    }

    #[test]
    fn test_spin_kerker_high_g_approaches_one() {
        for &g2 in &[1e3, 1e6, 1e12] {
            let k = spin_kerker_formula(g2);
            let diff = (k - 1.0).abs();
            let expected_diff = Q2 / (g2 + Q2);
            assert!(diff < expected_diff * 1.1,
                "Ks({g2}) = {k}, diff={diff}, expected_diff~{expected_diff}");
        }
    }
}
