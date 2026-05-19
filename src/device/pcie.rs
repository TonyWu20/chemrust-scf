#![allow(dead_code)]
use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::{result::DriverError, CudaSlice, CudaStream};

use crate::device::{DeviceMapped, Gpu};
use crate::layout::Cpu;

// ---------------------------------------------------------------------------
// PcieAccount — tracks GPU ↔ CPU data transfers
// ---------------------------------------------------------------------------
//
// Thread an &mut PcieAccount through any function that touches GPU memory.
// At the end of the critical path, assert() on the counts.
// Any unexpected D2H/H2D shows up as a mismatch.

/// Accumulated PCI-E transfer volume.
#[derive(Debug, Clone, Copy, Default)]
#[must_use]   // ← forces every call site to handle the account
pub(crate) struct PcieAccount {
    pub h2d_bytes: usize,   // host → device
    pub d2h_bytes: usize,   // device → host
}

impl PcieAccount {
    #[allow(dead_code)]
    pub fn record_h2d<T>(&mut self, slice: &CudaSlice<T>) {
        self.h2d_bytes += slice.len() * size_of::<T>();
    }
    pub fn record_d2h<T>(&mut self, slice: &CudaSlice<T>) {
        self.d2h_bytes += slice.len() * size_of::<T>();
    }
    pub fn reset(&mut self) {
        self.h2d_bytes = 0;
        self.d2h_bytes = 0;
    }
}

// ---------------------------------------------------------------------------
// Integration: Gpu::from_host now records transfers
// ---------------------------------------------------------------------------

impl<T: DeviceMapped> Gpu<T> {
    pub(crate) fn from_host_with(
        value: &T,
        stream: &Arc<CudaStream>,
        acc: &mut PcieAccount,
    ) -> Result<Self, DriverError> {
        let shape = value.shape_metadata();
        let host_data = value.flatten_host();
        let n = host_data.len();
        acc.h2d_bytes += n * size_of::<T::Elem>();
        let mut slice = stream.alloc_zeros::<T::Elem>(n)?;
        stream.memcpy_htod(&host_data, &mut slice)?;
        let ctx = stream.context();
        Ok(Self { slice, shape, ctx: ctx.clone(), _marker: PhantomData })
    }

    pub(crate) fn sync_to_host_with(
        &self,
        stream: &Arc<CudaStream>,
        acc: &mut PcieAccount,          // ← new parameter
    ) -> Result<Cpu<T>, DriverError> {
        let n = self.slice.len() * size_of::<T::Elem>();
        acc.d2h_bytes += n;                             // ← record
        let data = stream.clone_dtoh(&self.slice)?;
        let value = T::unflatten_host(data, &self.shape);
        Ok(Cpu(value))
    }
}

// ---------------------------------------------------------------------------
// How the diagonalize call chain reads with PcieAccount
// ---------------------------------------------------------------------------
//
// The orchestrator owns the account and asserts:
//
// pub fn diagonalize(self, ndeg: usize) -> Result<ScfIteration<..., WavefunctionsUpdated>> {
//     let mut pcie = PcieAccount::default();
//
//     // ── Expected: 2 H2D, 0 D2H in setup ──
//     let v_eff_gpu = Gpu::from_host_with(&v_eff_wave, &stream, &mut pcie)?;
//     let psi_gpu   = Gpu::from_cpu_with(&Cpu(self.psi), &stream, &mut pcie)?;
//     //                                pcie.h2d_bytes ≈ 256KB + 4MB
//     //                                pcie.d2h_bytes ≈ 0
//
//     // ── Chebyshev recurrence — MUST have ZERO PCI-E ──
//     let (filtered, hpsi) = chebyshev_filter(
//         &psi_gpu, &v_eff_gpu, ..., &mut pcie,
//     )?;
//     //                          pcie.h2d_bytes unchanged ← VICTORY
//     //                          pcie.d2h_bytes unchanged ← VICTORY
//
//     // ── Rayleigh-Ritz — MUST have ZERO PCI-E ──
//     let (psi_new, eigenvalues_cpu) = rayleigh_ritz(
//         &filtered, &hpsi, ..., &mut pcie,
//     )?;
//     //                          pcie.h2d_bytes unchanged ← VICTORY
//     //                          pcie.d2h_bytes has ~4KB for eigenvalues
//
//     // ── Final D2H — EXPECTED ──
//     let Cpu(psi_new) = psi_new.sync_to_host_with(&stream, &mut pcie)?;
//
//     // ── TRAP: any function that silently adds a D2H gets caught here ──
//     assert_eq!(pcie.h2d_bytes, psi_bytes + veff_bytes,
//         "H2D: expected exactly 2 uploads in diagonalize");
//     assert_eq!(pcie.d2h_bytes, psi_bytes + eig_bytes,
//         "D2H: expected exactly 2 downloads in diagonalize");
// }

// ---------------------------------------------------------------------------
// How the violations would appear
// ---------------------------------------------------------------------------

// VIOLATION 1 (chebyshev.rs:218): compute_spectral_bounds D2H entire V_eff
//
//   fn compute_spectral_bounds(
//       ...,
//       v_eff_slice: &CudaSlice<f64>,
//       stream: &Arc<CudaStream>,
//       pcie: &mut PcieAccount,          // ← added
//   ) -> Result<SpectralBounds> {
//       let data: Vec<f64> = stream.clone_dtoh(v_eff_slice)?;
//       pcie.d2h_bytes += data.len() * 8;  // ← explicit: can't hide
//       let max = data.iter().fold(...);    // ← CPU, not cuBLAS iamax
//
//   The assert in diagonalize FAILS:
//     assertion `pcie.d2h_bytes == psi_bytes + eig_bytes` failed
//     added 256KB from spectral bounds → d2h_bytes now = psi+eig+256KB
//
//   Fix: use cuBLAS iamax (zero D2H for the reduction, 1 f64 for the result).
//   With PcieAccount the fix is obvious — the 256KB spike tells you.

// VIOLATION 2 (rayleigh_ritz.rs:153): transpose via host roundtrip
//
//   fn rayleigh_ritz(..., pcie: &mut PcieAccount) {
//       let cpu_data = stream.clone_dtoh(psi_slice)?;
//       pcie.record_d2h(psi_slice);        // ← D2H 4MB spike
//       // ... CPU transpose ...
//       let gpu_data = stream.clone_htod(&cpu_data)?;
//       pcie.record_h2d(&gpu_data);        // ← H2D 4MB spike
//
//   Assert fails: both h2d_bytes and d2h_bytes are inflated by 4MB each.
//   Fix: use the GPU transpose kernel (already exists in chebyshev.rs!)
//   With PcieAccount the pattern is undeniable — double the expected D2H.
