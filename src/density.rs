// ---------------------------------------------------------------------------
// Density construction from wavefunctions (GPU)
// ---------------------------------------------------------------------------
//
// 1. Compute Gaussian-smearing occupation numbers on CPU
// 2. Scatter sparse PW → full FFT grid on GPU
// 3. Batched C2C IFFT → ψ(r) for all bands
// 4. ρ(r) = (1/Ω) · Σ_b occ_b · |ψ_b(r)|²
// 5. D2H → Density

use std::sync::Arc;

use bon::builder;
use chemrust_hamiltonian_core::GVectorGrid;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ndarray::Array3;
use num_complex::Complex64;

use crate::device::fft::BatchedFftPlan3d;
use crate::device::{complex_slice_to_cuda, CudaComplex};
use crate::eigensolver::chebyshev::CudaKernelSet;
use crate::types::{Density, Error, SmearingParams, SmearingScheme, WaveGridArray};

// ---------------------------------------------------------------------------
// Occupation numbers (Gaussian smearing, CASTEP default)
// ---------------------------------------------------------------------------

/// Compute occupation numbers via Gaussian smearing.
///
/// occ_b = erfc((μ - ε_b) / w)
///
/// The chemical potential μ satisfies Σ_b occ_b = N_electrons.
pub(crate) fn compute_occupations(
    eigenvalues: &[f64],
    smearing: &SmearingParams,
    n_electrons: f64,
) -> Result<Vec<f64>, Error> {
    match smearing.scheme {
        SmearingScheme::Gaussian => {
            let mu = find_chemical_potential(eigenvalues, smearing.width, n_electrons)?;
            Ok(eigenvalues
                .iter()
                .map(|&e| libm::erfc((mu - e) / smearing.width))
                .collect())
        }
    }
}

/// Bisection search for μ such that Σ erfc((μ - ε_b) / w) = N_electrons.
fn find_chemical_potential(
    eigenvalues: &[f64],
    width: f64,
    n_electrons: f64,
) -> Result<f64, Error> {
    let emin = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax = eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if eigenvalues.is_empty() || emax < emin {
        return Err(Error::NotImplemented);
    }
    let mut lo = emin - 10.0 * width;
    let mut hi = emax + 10.0 * width;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let sum: f64 = eigenvalues
            .iter()
            .map(|&e| libm::erfc((mid - e) / width))
            .sum();
        if sum > n_electrons {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

// ---------------------------------------------------------------------------
// GPU density construction (builder API via bon)
// ---------------------------------------------------------------------------

/// Build electron density from wavefunctions on GPU.
///
/// Pipeline:
/// 1. H2D psi, fft_indices, occupations
/// 2. Scatter sparse PW → full FFT grid (reuse `scatter_pw_to_grid` kernel)
/// 3. Batched C2C IFFT (reuse `BatchedFftPlan3d`)
/// 4. ρ[r] = (1/Ω) Σ_b occ_b |ψ_b[r]|² (`accumulate_density` kernel)
/// 5. D2H → Density(WaveGridArray)
#[builder]
pub(crate) fn construct_density_gpu(
    psi_data: &[Complex64],
    occupations: &[f64],
    fft_indices: &[i32],
    wave_grid: &GVectorGrid,
    cell_volume: f64,
    n_bands: usize,
    n_pw: usize,
    kernels: &CudaKernelSet,
    stream: &Arc<CudaStream>,
) -> Result<Density, Error> {
    let [ngz, ngy, ngx] = wave_grid.grid();
    let grid_size = (ngz * ngy * ngx) as i32;
    let inv_omega = 1.0 / cell_volume;

    let n_bands_i = n_bands as i32;
    let n_pw_i = n_pw as i32;

    // 1. H2D: psi, fft_indices, occupations
    let psi_slice: Vec<CudaComplex> = complex_slice_to_cuda(psi_data);
    let psi_dev: CudaSlice<CudaComplex> = stream
        .clone_htod(&psi_slice)
        .map_err(Error::Cuda)?;
    let fft_idx_dev: CudaSlice<i32> = stream
        .clone_htod(fft_indices)
        .map_err(Error::Cuda)?;
    let occ_dev: CudaSlice<f64> = stream
        .clone_htod(occupations)
        .map_err(Error::Cuda)?;

    // 2. Allocate + zero the full FFT grid: n_bands × grid_size complex
    let mut grid_dev: CudaSlice<CudaComplex> = {
        let g: CudaSlice<CudaComplex> =
            stream.alloc_zeros(n_bands * grid_size as usize).map_err(Error::Cuda)?;
        g
    };

    // 3. Scatter sparse PW → full FFT grid
    unsafe {
        stream
            .launch_builder(&kernels.scatter_pw_to_grid)
            .arg(&psi_dev)
            .arg(&fft_idx_dev)
            .arg(&mut grid_dev)
            .arg(&n_pw_i)
            .arg(&n_bands_i)
            .arg(&grid_size)
            .launch(LaunchConfig::for_num_elems(
                (n_bands_i * n_pw_i) as u32,
            ))
    }
    .map_err(Error::Cuda)?;

    // 4. Batched C2C IFFT (in-place on grid_dev)
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32,
        n_bands as i32, Arc::clone(stream),
    )?;
    // In-place IFFT: same buffer for input and output via raw pointer
    unsafe {
        let ptr = &mut grid_dev as *mut CudaSlice<CudaComplex>;
        fft_plan.c2c_inverse(&mut *ptr, &mut *ptr)?
    };

    // 5. Accumulate density: ρ[r] = inv_omega × Σ_b occ[b] × |ψ_b[r]|²
    let mut rho_dev: CudaSlice<f64> = {
        let r: CudaSlice<f64> =
            stream.alloc_zeros(grid_size as usize).map_err(Error::Cuda)?;
        r
    };
    unsafe {
        stream
            .launch_builder(&kernels.accumulate_density)
            .arg(&grid_dev)
            .arg(&occ_dev)
            .arg(&mut rho_dev)
            .arg(&n_bands_i)
            .arg(&grid_size)
            .arg(&inv_omega)
            .launch(LaunchConfig::for_num_elems(grid_size as u32))
    }
    .map_err(Error::Cuda)?;

    // 6. D2H density
    let rho_host: Vec<f64> = stream
        .clone_dtoh(&rho_dev)
        .map_err(Error::Cuda)?;
    let array = Array3::from_shape_vec((ngx, ngy, ngz), rho_host)
        .map_err(|_| Error::NotImplemented)?;

    Ok(Density::from_inner(WaveGridArray::from_inner(array)))
}
