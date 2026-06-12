// ---------------------------------------------------------------------------
// Pipeline orchestration: V_eff preparation, k-point GPU upload, Davidson run
// ---------------------------------------------------------------------------
//
// Extracts shared patterns from ffi.rs and scf.rs into reusable functions
// for the pure-Rust SCF path and the FFI binding layer.

use std::sync::Arc;

use chemrust_hamiltonian_core::{CellGeometry, GVectorGrid, PseudopotentialSet};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use ndarray::Array3;

use crate::device::blas::BlasHandle;
use crate::device::fft::BatchedFftPlan3d;
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::eigensolver::d_screening::{build_wave_screening_cache, WaveScreeningCache};
use crate::eigensolver::davidson::davidson_diagonalise;
use crate::eigensolver::davidson::DavidsonResult;
use crate::eigensolver::davidson_types::{KineticPreconditioner, PwCoefficients};
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::preconditioner::TpaPreconditioner;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::scf::downsample_array_to_wave_grid;
use crate::types::Error;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// GPU-resident V_eff data prepared from fine-grid potential.
///
/// Holds the downsampled wave-grid array (iz-innermost for cuFFT), the
/// original fine-grid array, and the GPU device slice.
pub struct VEffData {
    /// GPU-side iz-innermost V_eff on the standard wave grid.
    pub gpu_slice: CudaSlice<f64>,
    /// Original fine-grid V_eff (ix-innermost).
    pub fine_arr: Array3<f64>,
    /// Downsampled wave-grid V_eff (iz-innermost), shape (ngz, ngy, ngx).
    pub wave_arr: Array3<f64>,
    /// Total number of grid points on the wave grid (ngz * ngy * ngx).
    pub grid_size: usize,
}

/// Per-k-point GPU data: kinetic preconditioner and FFT index map.
pub struct KptGpuData {
    pub kinetic_precond: KineticPreconditioner,
    pub fft_idx_dev: CudaSlice<i32>,
}

/// Combined screening caches for D-matrix evaluation on wave and fine grids.
#[derive(Clone)]
pub struct ScreeningCaches {
    pub wave: WaveScreeningCache,
    pub fine: Option<WaveScreeningCache>,
}

impl ScreeningCaches {
    /// Build both wave-grid and optional fine-grid screening caches.
    pub fn build(
        pots: &PseudopotentialSet,
        cell: &CellGeometry,
        wave_grid: &GVectorGrid,
        fine_grid: Option<&GVectorGrid>,
        stream: &Arc<CudaStream>,
        pcie: &mut PcieAccount,
    ) -> Result<Self, Error> {
        let wave = build_wave_screening_cache(pots, cell, wave_grid, stream, pcie)?;
        let fine = match fine_grid {
            Some(fg) => Some(build_wave_screening_cache(pots, cell, fg, stream, pcie)?),
            None => None,
        };
        Ok(Self { wave, fine })
    }
}

// ---------------------------------------------------------------------------
// v_eff_prepare
// ---------------------------------------------------------------------------

/// Downsample fine-grid V_eff to wave grid, transpose ix→iz for cuFFT,
/// and upload to GPU.
///
/// Matches the FFI path in ffi.rs (lines 457–518): FFT-based downsampling via
/// `downsample_array_to_wave_grid`, then transposition to iz-innermost, then
/// upload via `alloc_zeros` + `memcpy_htod`.
pub fn v_eff_prepare(
    fine_arr: &Array3<f64>,
    fine_grid: &GVectorGrid,
    wave_grid: &GVectorGrid,
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<VEffData, Error> {
    // FFT-based downsampling (matches CASTEP basis_real_fine_to_std_grid)
    let v_eff_wave = downsample_array_to_wave_grid(fine_arr, fine_grid, wave_grid)?;
    // Result is ix-innermost on standard wave grid
    let wave_ix = v_eff_wave.as_fine_array();

    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = ngz * ngy * ngx;

    // Transpose ix→iz for cuFFT (CASTEP / cuFFT expect iz-innermost ordering).
    // Build both the iz-innermost flat buffer for GPU upload and the
    // corresponding Array3 for the VEffData struct.
    let mut ve_std_iz = vec![0.0_f64; grid_size];
    let mut wave_arr = Array3::zeros((ngz, ngy, ngx));
    for iz in 0..ngz {
        for iy in 0..ngy {
            for ix in 0..ngx {
                let val = wave_ix[[ix, iy, iz]];
                // iz-innermost linear index: iz + ngz * (iy + ngy * ix)
                let idx_iz = iz + ngz * (iy + ngy * ix);
                ve_std_iz[idx_iz] = val;
                wave_arr[[iz, iy, ix]] = val;
            }
        }
    }

    // Upload fresh V_eff to GPU every call — CASTEP recomputes V_eff from
    // density each SCF iteration with no caching.
    let mut gpu_slice = stream.alloc_zeros::<f64>(grid_size).map_err(Error::Cuda)?;
    stream.memcpy_htod(&ve_std_iz, &mut gpu_slice).map_err(Error::Cuda)?;
    pcie.h2d_bytes += ve_std_iz.len() * std::mem::size_of::<f64>();

    Ok(VEffData {
        gpu_slice,
        fine_arr: fine_arr.clone(),
        wave_arr,
        grid_size,
    })
}

// ---------------------------------------------------------------------------
// kpt_gpu_upload
// ---------------------------------------------------------------------------

/// Upload kinetic energies and FFT index map to GPU for a k-point.
///
/// Matches the per-k-point upload pattern in ffi.rs (lines 553–570) and
/// scf.rs (lines 858–874).
pub fn kpt_gpu_upload(
    kinetic: &[f64],
    fft_idx: &[i32],
    stream: &Arc<CudaStream>,
    pcie: &mut PcieAccount,
) -> Result<KptGpuData, Error> {
    let kinetic_dev: CudaSlice<f64> = stream.clone_htod(kinetic).map_err(Error::Cuda)?;
    let kinetic_precond = KineticPreconditioner::new(kinetic_dev);
    pcie.h2d_bytes += kinetic.len() * std::mem::size_of::<f64>();

    let fft_idx_dev: CudaSlice<i32> = stream.clone_htod(fft_idx).map_err(Error::Cuda)?;
    pcie.h2d_bytes += fft_idx.len() * std::mem::size_of::<i32>();

    Ok(KptGpuData {
        kinetic_precond,
        fft_idx_dev,
    })
}

// ---------------------------------------------------------------------------
// run_davidson
// ---------------------------------------------------------------------------

/// Run Davidson diagonalisation with the full builder chain.
///
/// Wraps `davidson_diagonalise()` with all required parameters, matching the
/// call sites in ffi.rs (lines 615–639) and scf.rs (lines 899–921).
///
/// # Safety
///
/// All device pointers must be valid and of sufficient size.
pub(crate) unsafe fn run_davidson(
    psi_init: &PwCoefficients,
    v_eff_slice: &CudaSlice<f64>,
    kinetic_precond: &KineticPreconditioner,
    fft_idx_dev: &CudaSlice<i32>,
    vnl_data: &VnlBatchData,
    n_pw: usize,
    n_bands: usize,
    grid_size: usize,
    inv_ntotal: f64,
    fft_plan: &BatchedFftPlan3d,
    blas: &BlasHandle,
    solver: &SolverHandle,
    kernels: &CudaKernelSet,
    tpa: &TpaPreconditioner,
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
    max_outer_iter: usize,
    gamma_point: bool,
) -> Result<DavidsonResult, Error> {
    unsafe {
        davidson_diagonalise()
            .psi_init(psi_init)
            .v_eff_dev(v_eff_slice)
            .kinetic_dev(kinetic_precond)
            .fft_idx_dev(fft_idx_dev)
            .vnl_data(vnl_data)
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(grid_size)
            .inv_ntotal(inv_ntotal)
            .fft_plan(fft_plan)
            .tol_abs(1e-8)
            .max_outer_iter(max_outer_iter)
            .min_outer_iter(0)
            .blas(blas)
            .solver(solver)
            .kernels(kernels)
            .tpa_preconditioner(tpa)
            .stream(stream)
            .ctx(ctx)
            .gamma_point(gamma_point)
            .call()
    }
}
