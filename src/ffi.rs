// ---------------------------------------------------------------------------
// FFI: CASTEP Fortran ↔ Rust GPU eigensolver
// ---------------------------------------------------------------------------

use std::ffi::{c_char, c_void, CStr};
use std::os::raw::{c_double, c_int};
use std::sync::Arc;

use chemrust_hamiltonian_core::{CellGeometry, GVectorGrid, PseudopotentialSet, RealLattice, RecipLattice};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::device::blas::BlasHandle;
use crate::device::fft::BatchedFftPlan3d;
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::device::{CudaComplex, Gpu};
use crate::eigensolver::davidson::davidson_diagonalise;
use crate::eigensolver::davidson_types::{KineticPreconditioner, PwCoefficients};
use crate::eigensolver::hamiltonian::apply_full_hamiltonian;
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::preconditioner::TpaPreconditioner;
use crate::eigensolver::vnl_data::{build_handle_shared_vnl, HandleSharedVnl, KptSharedVnl, VnlBatchData};
use crate::layout::{ColumnDistributed, WavefunctionSet};
use crate::types::{EffectivePotential, FineGridArray, KPoint};

pub const CHEM_EIG_OK: c_int = 0;
pub const CHEM_EIG_CUDA_ERROR: c_int = 3;
pub const CHEM_EIG_NULL_HANDLE: c_int = 4;

// ---- Helper: recover integer Miller indices from Cartesian G-vectors -------

/// Convert a 1-based CASTEP grid index to signed integer grid coordinates
/// (nx_coord, ny_coord, nz_coord).
///
/// CASTEP's grid index uses the COLUMN ordering from `basis_assign_plane_wave_indexes`:
///   point = 1 + nx-1 + ngx*((column_y-1) + ngy*(column_z-1))
/// where nx is innermost, then ny (via column_y), then nz (via column_z).
/// This corresponds to:
///   idx_0based = (nx-1) + ngx*(ny-1) + ngx*ngy*(nz-1)
fn fft_idx_to_coord(idx_1based: i32, ngx: i32, ngy: i32, ngz: i32) -> [i32; 3] {
    let idx = (idx_1based - 1).max(0);
    let nx_off = idx % ngx;
    let ny_off = (idx / ngx) % ngy;
    let nz_off = idx / (ngx * ngy);
    let nx = if nx_off <= ngx / 2 { nx_off } else { nx_off - ngx };
    let ny = if ny_off <= ngy / 2 { ny_off } else { ny_off - ngy };
    let nz = if nz_off <= ngz / 2 { nz_off } else { nz_off - ngz };
    [nx, ny, nz]
}

// ---- Per-k-point precomputed data ------------------------------------------

struct KptData {
    /// Fractional k-point coordinates [kx, ky, kz].
    /// Used for gamma-point detection (all coords zero → DSYEVD path).
    kpoint_frac: [f64; 3],
    /// k-point descriptor (coords only; pw_coords stored separately).
    k_point: KPoint,
    /// Plane-wave Miller indices for this k-point.
    pw_coords: Vec<[i32; 3]>,
    /// Lazily-initialised V_NL batch data per spin.  Created on the first call
    /// to step_inner with the REAL psi and n_bands — never dummy values.
    vnl: Vec<Option<VnlBatchData>>,
    /// Shared spin-independent VNL state per k-point, built on first
    /// step_inner call and reused by subsequent spin calls.
    shared_vnl: Option<Arc<KptSharedVnl>>,
}

// ---- Eigensolver mode ------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
enum EigensolverMode {
    Davidson = 0,
    #[cfg(feature = "chebyshev")]
    Chebyshev = 1,
}

// ---- Opaque handle ---------------------------------------------------------

struct ChemrustHandle {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: BlasHandle,
    solver: SolverHandle,
    kernels: CudaKernelSet,
    ngx: i32, ngy: i32, ngz: i32,           // FINE grid
    ngx_std: i32, ngy_std: i32, ngz_std: i32, // STANDARD grid
    max_n_pw: usize,
    nspins: usize,
    kpts: Vec<KptData>,
    /// Cached GPU copies of V_eff per spin for skip-upload optimization.
    v_eff_cached: Vec<Option<CudaSlice<f64>>>,
    /// Max-norm of the cached V_eff per spin, used for change detection.
    v_eff_norm: Vec<f64>,
    /// Kpt-independent V_NL shared state (screening caches, Q, D0).  Built once
    /// at init, shared via Arc across all k-points and spin channels.
    handle_shared_vnl: Option<Arc<HandleSharedVnl>>,
    /// Shared across k-points: pseudopotentials, cell geometry, wave grid.
    /// Stored here so that VnlBatchData can be created in step_inner
    /// (where real psi/n_bands are available) instead of init_inner
    /// (where they are not).
    pots: PseudopotentialSet,
    cell: CellGeometry,
    wave_grid: GVectorGrid,  // STANDARD grid — matches CASTEP's internal FFT grid
    fine_grid: GVectorGrid,  // FINE grid — for V_eff downsampling only
    eigensolver_mode: EigensolverMode,
    /// Per-(spin,kpt) flag: true until the first successful eigensolve.
    /// Forces Davidson on SCF iter 0 regardless of eigensolver_mode.
    first_scf_iter: Vec<Vec<bool>>,
}

// ---- Init ------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_eigensolve_init(
    num_species: c_int, species_symbols: *const *const c_char, species_pots: *const *const c_char,
    real_lattice: *const c_double, recip_lattice: *const c_double,
    num_ions: c_int, ion_species: *const c_int, ion_positions: *const c_double,
    nkpts: c_int, num_pw_per_kpt: *const c_int, gvec_all_kpt: *const c_double,
    pw_grid_idx: *const c_int,
    kpt_coords: *const c_double,
    ngx: c_int, ngy: c_int, ngz: c_int,
    ngx_std: c_int, ngy_std: c_int, ngz_std: c_int,
    nspins: c_int,
    handle_out: *mut *mut c_void,
) -> c_int {
    if handle_out.is_null() { return CHEM_EIG_NULL_HANDLE; }
    let h = match init_inner(num_species, species_symbols, species_pots,
        real_lattice, recip_lattice, num_ions, ion_species, ion_positions,
        nkpts, num_pw_per_kpt, gvec_all_kpt, pw_grid_idx, kpt_coords,
        ngx, ngy, ngz, ngx_std, ngy_std, ngz_std, nspins)
    {
        Ok(h) => h,
        Err(code) => return code,
    };
    unsafe { *handle_out = h as *mut c_void };
    CHEM_EIG_OK
}

fn init_inner(
    num_species: c_int, species_symbols: *const *const c_char, species_pots: *const *const c_char,
    real_lattice: *const c_double, recip_lattice: *const c_double,
    num_ions: c_int, ion_species: *const c_int, ion_positions: *const c_double,
    nkpts: c_int, num_pw_per_kpt: *const c_int, gvec_all_kpt: *const c_double,
    pw_grid_idx: *const c_int,
    kpt_coords: *const c_double,
    ngx: c_int, ngy: c_int, ngz: c_int,
    ngx_std: c_int, ngy_std: c_int, ngz_std: c_int,
    nspins: c_int,
) -> Result<*mut ChemrustHandle, c_int> {
    let ctx: Arc<CudaContext> = CudaContext::new(0).map_err(|e| { eprintln!("[chemrust] init: CudaContext failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).map_err(|e| { eprintln!("[chemrust] init: Blas failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    let solver = SolverHandle::new(stream.clone()).map_err(|e| { eprintln!("[chemrust] init: Solver failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    let kernels = CudaKernelSet::new(&ctx).map_err(|e| { eprintln!("[chemrust] init: Kernels failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    // Read USP pseudopotentials from CASTEP-provided paths
    let ns = num_species as usize;
    let syms: Vec<String> = (0..ns).map(|i| unsafe { CStr::from_ptr(*species_symbols.add(i)) }.to_str().unwrap_or("").to_string()).collect();
    let pot_paths: Vec<String> = (0..ns).map(|i| unsafe { CStr::from_ptr(*species_pots.add(i)) }.to_str().unwrap_or("").to_string()).collect();
    let mut pots = PseudopotentialSet::new();
    for (sym, path) in syms.iter().zip(pot_paths.iter()) {
        let pp = chemrust_hamiltonian_core::pseudopotential::Pseudopotential::from_path(&path)
            .map_err(|e| { eprintln!("[chemrust] failed {path}: {e}"); CHEM_EIG_CUDA_ERROR })?;
        pots.insert(sym.clone(), pp);
    }

    // Cell geometry
    let rl = unsafe { std::slice::from_raw_parts(real_lattice as *const f64, 9) };
    let rp = unsafe { std::slice::from_raw_parts(recip_lattice as *const f64, 9) };
    let mut rlat = [[0.0f64; 3]; 3];
    let mut rcip = [[0.0f64; 3]; 3];
    for i in 0..3 { for j in 0..3 { rlat[i][j] = rl[i*3+j]; rcip[i][j] = rp[i*3+j]; } }
    let real_lat = RealLattice::from_inner(rlat);
    let recip_lat = RecipLattice::from_inner(rcip);
    let vol = (rlat[0][0]*(rlat[1][1]*rlat[2][2]-rlat[1][2]*rlat[2][1])
              +rlat[0][1]*(rlat[1][2]*rlat[2][0]-rlat[1][0]*rlat[2][2])
              +rlat[0][2]*(rlat[1][0]*rlat[2][1]-rlat[1][1]*rlat[2][0])).abs();

    let ni = num_ions as usize;
    let ion_sp: Vec<usize> = (0..ni).map(|i| unsafe { *(ion_species.add(i)) } as usize).collect();
    let pos_f = unsafe { std::slice::from_raw_parts(ion_positions as *const f64, 3*ni) };
    let mut pos = ndarray::Array2::<f64>::zeros((ni, 3));
    for i in 0..ni { for j in 0..3 { pos[[i, j]] = pos_f[3*i + j]; } }

    let cell = CellGeometry {
        real_lattice: real_lat,
        recip_lattice: recip_lat,
        volume: vol, num_species: ns, num_ions: ni,
        ionic_positions: pos, species_symbols: syms,
        species_pot_files: vec![], num_ions_in_species: vec![],
        ion_species: ion_sp, max_ions_in_species: 0, species_lcao_states: vec![],
    };

    // Per k-point VnlBatchData precompute
    let nk = nkpts as usize;
    let npwk = unsafe { std::slice::from_raw_parts(num_pw_per_kpt as *const i32, nk) };
    let kfrac = unsafe { std::slice::from_raw_parts(kpt_coords as *const f64, 3*nk) };
    let maxpw = *npwk.iter().max().unwrap_or(&0) as usize;
    let _gv = unsafe { std::slice::from_raw_parts(gvec_all_kpt as *const f64, 3*maxpw*nk) };
    let gidx = unsafe { std::slice::from_raw_parts(pw_grid_idx as *const i32, maxpw*nk) };

    // wave_grid uses STANDARD grid dims — matches CASTEP's internal FFT grid
    // after pot_interpolate downsamples V_eff from fine to standard grid.
    let wave_grid = GVectorGrid::new(ngx_std as usize, ngy_std as usize, ngz_std as usize, RecipLattice::from_inner(rcip));
    // fine_grid uses FINE grid dims — for receiving/downsampling CASTEP's real_fine_pot.
    let fine_grid = GVectorGrid::new(ngx as usize, ngy as usize, ngz as usize, RecipLattice::from_inner(rcip));

    let nspins_u = nspins as usize;

    let mut kpts = Vec::with_capacity(nk);
    for ik in 0..nk {
        let n_pw = npwk[ik] as usize;
        let kf = [kfrac[3*ik], kfrac[3*ik+1], kfrac[3*ik+2]];
        let kpt = KPoint { coords: kf, weight: 1.0 };

        // Convert k-point fractional → Cartesian for diagnostic
        let mut k_cart_diag = [0.0f64; 3];
        for a in 0..3 {
            for b in 0..3 {
                k_cart_diag[b] += kf[a] * rcip[a][b];
            }
        }

        // Compute pw_coords from the FFT grid index (1-based from CASTEP)
        // using the FFTW-ordering convention.  This avoids the fragile
        // Cartesian→Miller conversion which gives wrong results for the
        // first few PWs (where Cartesian components are near-zero but the
        // grid index correctly encodes a non-Gamma grid point).
        // CASTEP's pw_grid_index is encoded on the STANDARD grid with
        // strides [1, ngx_std, ngx_std*ngy_std] (ix-innermost).
        // Decode with standard grid dimensions for correct Miller indices.
        let mut pw_coords = Vec::with_capacity(n_pw);
        for ipw in 0..n_pw {
            let idx_1based = gidx[ik*maxpw + ipw];
            pw_coords.push(fft_idx_to_coord(idx_1based, ngx_std, ngy_std, ngz_std));
        }

        // Diagnostic: compare our pw_coords→Cartesian vs CASTEP's gvec
        // CASTEP's gvec_all_kpt is pw_g_vector = G+k (Cartesian)
        // Our pw_coords are G Miller indices → G_cart = G * rcip
        // Difference should be k_cart
        #[cfg(feature = "scf_diag")]
        {
            let gv = unsafe { std::slice::from_raw_parts(gvec_all_kpt as *const f64, 3*maxpw*nk) };
            let n_check = n_pw.min(10);
            let mut max_err = 0.0f64;
            for ipw in 0..n_check {
                let coord = pw_coords[ipw];
                let gx = coord[0] as f64 * rcip[0][0] + coord[1] as f64 * rcip[1][0] + coord[2] as f64 * rcip[2][0];
                let gy = coord[0] as f64 * rcip[0][1] + coord[1] as f64 * rcip[1][1] + coord[2] as f64 * rcip[2][1];
                let gz = coord[0] as f64 * rcip[0][2] + coord[1] as f64 * rcip[1][2] + coord[2] as f64 * rcip[2][2];
                let gv_x = gv[ik*maxpw*3 + ipw*3 + 0];
                let gv_y = gv[ik*maxpw*3 + ipw*3 + 1];
                let gv_z = gv[ik*maxpw*3 + ipw*3 + 2];
                let dx = (gx + k_cart_diag[0] - gv_x).abs();
                let dy = (gy + k_cart_diag[1] - gv_y).abs();
                let dz = (gz + k_cart_diag[2] - gv_z).abs();
                max_err = max_err.max(dx).max(dy).max(dz);
                if ipw < 5 {
                    eprintln!("[Diag-Gvec] ik={} ipw={} pw_coord=({},{},{}) G_cart+k=({:.6},{:.6},{:.6}) gvec=({:.6},{:.6},{:.6}) err=({:.2e},{:.2e},{:.2e})",
                        ik, ipw, coord[0], coord[1], coord[2],
                        gx + k_cart_diag[0], gy + k_cart_diag[1], gz + k_cart_diag[2],
                        gv_x, gv_y, gv_z, dx, dy, dz);
                }
            }
            eprintln!("[Diag-Gvec] ik={} max|(G_cart + k_cart) - gvec| = {:.3e} (checked {} PWs)",
                ik, max_err, n_check);
        }

        // VnlBatchData is deferred to step_inner where real psi/n_bands are
        // available.  Store only the ion-independent per-k-point data here.
        kpts.push(KptData {
            vnl: (0..nspins_u).map(|_| None).collect(),
            shared_vnl: None,
            kpoint_frac: kf,
            k_point: kpt,
            pw_coords,
        });
    }

    let handle_shared_vnl = build_handle_shared_vnl(
        &pots, &cell, &wave_grid, Some(&fine_grid),
        &stream, &mut PcieAccount::default(),
    ).map_err(|e| { eprintln!("[chemrust] HandleSharedVnl build failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    let pots_clone = pots.clone();
    let cell_clone = cell.clone();
    Ok(Box::into_raw(Box::new(ChemrustHandle {
        ctx, stream, blas, solver, kernels,
        ngx, ngy, ngz, ngx_std, ngy_std, ngz_std, max_n_pw: maxpw, nspins: nspins_u, kpts,
        v_eff_cached: (0..nspins_u).map(|_| None).collect(),
        v_eff_norm: vec![0.0; nspins_u],
        handle_shared_vnl: Some(handle_shared_vnl),
        pots: pots_clone,
        cell: cell_clone,
        wave_grid,
        fine_grid,
        eigensolver_mode: {
            #[cfg(feature = "chebyshev")]
            if std::env::var("CHEMRUST_EIGENSOLVER").unwrap_or_default() == "chebyshev" {
                EigensolverMode::Chebyshev
            } else {
                EigensolverMode::Davidson
            }
            #[cfg(not(feature = "chebyshev"))]
            EigensolverMode::Davidson
        },
        first_scf_iter: vec![vec![true; nkpts as usize]; nspins_u],
    })))
}

// ---- Set eigensolver mode ---------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_eigensolve_set_mode(
    handle: *mut c_void,
    mode: c_int,
) -> c_int {
    let h = match unsafe { (handle as *mut ChemrustHandle).as_mut() } {
        Some(h) => h,
        None => return CHEM_EIG_NULL_HANDLE,
    };
    match mode {
        0 => h.eigensolver_mode = EigensolverMode::Davidson,
        #[cfg(feature = "chebyshev")]
        1 => h.eigensolver_mode = EigensolverMode::Chebyshev,
        _ => return CHEM_EIG_CUDA_ERROR,
    }
    CHEM_EIG_OK
}

// ---- Destroy ---------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_eigensolve_destroy(handle: *mut c_void) -> c_int {
    if !handle.is_null() { unsafe { drop(Box::from_raw(handle as *mut ChemrustHandle)); } }
    CHEM_EIG_OK
}

// ---- Step ------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_eigensolve_step(
    handle: *mut c_void,
    psi_data: *mut CudaComplex, v_eff_data: *const c_double,
    kinetic_data: *const c_double, fft_idx_data: *const c_int,
    eigenvalues_ptr: *mut c_double, hpsi_out: *mut CudaComplex,
    npw: c_int, nbands: c_int, ikpt: c_int, ispin: c_int,
    max_deg: c_int,
    converged: *mut c_int,
) -> c_int {
    let h = match unsafe { (handle as *mut ChemrustHandle).as_mut() } {
        Some(h) => h,
        None => return CHEM_EIG_NULL_HANDLE,
    };
    #[cfg(feature = "chebyshev")]
    if h.eigensolver_mode == EigensolverMode::Chebyshev && !h.first_scf_iter[ispin as usize][ikpt as usize] {
        let result = match unsafe { step_inner_chebyshev(handle, psi_data, v_eff_data, kinetic_data,
            fft_idx_data, eigenvalues_ptr, hpsi_out, npw, nbands, ikpt, ispin, max_deg, converged) }
        {
            Ok(()) => CHEM_EIG_OK,
            Err(c) => c,
        };
        if result != CHEM_EIG_OK {
            return result;
        }
    }
    let result = match unsafe { step_inner(handle, psi_data, v_eff_data, kinetic_data, fft_idx_data,
        eigenvalues_ptr, hpsi_out, npw, nbands, ikpt, ispin, max_deg, converged) }
    {
        Ok(()) => CHEM_EIG_OK,
        Err(c) => c,
    };
    h.first_scf_iter[ispin as usize][ikpt as usize] = false;
    result
}

#[allow(clippy::too_many_arguments)]
unsafe fn step_inner(
    handle: *mut c_void,
    psi_data: *mut CudaComplex, v_eff_data: *const c_double,
    kinetic_data: *const c_double, fft_idx_data: *const c_int,
    eigenvalues_ptr: *mut c_double, hpsi_out: *mut CudaComplex,
    npw: c_int, nbands: c_int, ikpt: c_int, ispin: c_int,
    _max_deg: c_int,
    converged: *mut c_int,
) -> Result<(), c_int> {
    let h = unsafe { (handle as *mut ChemrustHandle).as_mut() }.ok_or(CHEM_EIG_NULL_HANDLE)?;
    if psi_data.is_null() || v_eff_data.is_null() || kinetic_data.is_null()
        || fft_idx_data.is_null() || eigenvalues_ptr.is_null() || hpsi_out.is_null() || converged.is_null()
    { return Err(CHEM_EIG_NULL_HANDLE); }

    let ik = ikpt as usize;
    if ik >= h.kpts.len() { return Err(CHEM_EIG_CUDA_ERROR); }
    let isp = ispin as usize;
    if isp >= h.nspins { return Err(CHEM_EIG_CUDA_ERROR); }
    let n_pw = npw as usize;
    let n_bands = nbands as usize;
    // Use STANDARD grid for FFT — matches CASTEP's internal convention after
    // pot_interpolate downsamples V_eff from fine to standard grid.
    let gs = (h.ngx_std * h.ngy_std * h.ngz_std) as usize;
    let n_bands_i32 = n_bands as i32;
    // cuFFT C2C inverse does NOT divide by N_grid (confirmed by the
    // round-trip identity test at device/fft.rs:262-283, which requires
    // explicit division by N to recover identity).  Both forward and
    // inverse transforms are unnormalized, so the full round-trip
    // IFFT → V_eff·pointwise → FFT introduces a factor of N_grid.
    // inv_ntotal compensates: FFT[V_eff · IFFT[ψ]](G) = N_grid · V_loc(G),
    // and multiplying by 1/N_grid recovers the correct V_loc contribution.
    // This matches the pattern used in the integration test
    // (hamiltonian.rs:752), standalone SCF (scf.rs), and Chebyshev solver.
    // N.B. gs uses STANDARD grid = ngx_std * ngy_std * ngz_std, matching
    // the FFT plan dimensions on the standard grid.
    let inv_ntotal = 1.0 / gs as f64;
    let kd = &mut h.kpts[ik];

    // Gamma-point detection: all fractional coords near zero → DSYEVD path.
    // Matches CASTEP hamiltonian.f90:480 — super_wvfn%have_gamma.
    let have_gamma = kd.kpoint_frac.iter().all(|&c| c.abs() < 1e-12);

    // Use CASTEP-provided kinetic energies (pw_ek_data = 0.5*|G+k|^2)
    let ke_castep: Vec<f64> = unsafe { std::slice::from_raw_parts(kinetic_data, n_pw) }.to_vec();

    #[cfg(feature = "scf_diag")]
    // Diagnostic: verify KE consistency
    {
        let n_print = kd.pw_coords.len().min(5);
        let ke_rust = crate::eigensolver::davidson_types::compute_kinetic_energies(&kd.pw_coords, h.wave_grid.recip_lattice(), kd.kpoint_frac);
        let mut all_ok = true;
        for i in 0..n_print {
            let diff = (ke_castep[i] - ke_rust.0[i]).abs();
            if diff > 1e-8 { all_ok = false; }
        }
        eprintln!("[chemrust] KE match: {} (n={} pw_coords[0]=({},{},{})) ke_castep[0]={:.6} ke_rust[0]={:.6}",
            if all_ok { "OK" } else { "MISMATCH" },
            n_pw,
            kd.pw_coords[0][0], kd.pw_coords[0][1], kd.pw_coords[0][2],
            ke_castep[0], ke_rust.0[0]);
    }

    // Upload psi.  CASTEP stores wvfn%coeffs(:,:,nk,ns) with leading dimension
    // max_n_pw (global max plane-wave count across all k-points), not n_pw
    // (this k-point's count).  Stride by max_n_pw to read each band correctly.
    let max_n_pw = h.max_n_pw;
    let raw_psi = unsafe { std::slice::from_raw_parts(psi_data as *const CudaComplex, max_n_pw * n_bands) };
    let mut psi_host: Vec<num_complex::Complex64> = Vec::with_capacity(n_pw * n_bands);
    for b in 0..n_bands {
        let start = b * max_n_pw;
        psi_host.extend(raw_psi[start..start + n_pw].iter().map(|c| num_complex::Complex64::new(c.x, c.y)));
    }

    // ---- FFI boundary diagnostic: dump raw ψ[0] coefficients ----
    // Compare against standalone test values to detect phase / layout mismatches.
    #[cfg(feature = "scf_diag")]
    {
        let psi_band0 = &psi_host[0..n_pw];
        let l2_sq: f64 = psi_band0.iter().map(|c| c.norm_sqr()).sum();
        eprintln!(
            "[Diag-FFI-psi] band 0: L2²={:.6e} n_pw={} |psi[0]|²={:.6e}",
            l2_sq, n_pw, psi_band0[0].norm_sqr(),
        );
        eprintln!(
            "[Diag-FFI-psi] band 0 first5=[[({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e})]]",
            psi_band0[0].re, psi_band0[0].im,
            psi_band0[1].re, psi_band0[1].im,
            psi_band0[2].re, psi_band0[2].im,
            psi_band0[3].re, psi_band0[3].im,
            psi_band0[4].re, psi_band0[4].im,
        );
        // Dump band 1 for sign-consistency check
        let psi_band1 = &psi_host[n_pw..2*n_pw];
        let l2_sq1: f64 = psi_band1.iter().map(|c| c.norm_sqr()).sum();
        eprintln!(
            "[Diag-FFI-psi] band 1: L2²={:.6e} first5=[[({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e})]]",
            l2_sq1,
            psi_band1[0].re, psi_band1[0].im,
            psi_band1[1].re, psi_band1[1].im,
            psi_band1[2].re, psi_band1[2].im,
            psi_band1[3].re, psi_band1[3].im,
            psi_band1[4].re, psi_band1[4].im,
        );
        // Dump band 25 (first band of second block) for sign-consistency
        let psi_b25 = &psi_host[25*n_pw..26*n_pw];
        let l2_sq25: f64 = psi_b25.iter().map(|c| c.norm_sqr()).sum();
        eprintln!(
            "[Diag-FFI-psi] band 25: L2²={:.6e} first5=[[({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e}), ({:.15e}, {:.15e})]]",
            l2_sq25,
            psi_b25[0].re, psi_b25[0].im,
            psi_b25[1].re, psi_b25[1].im,
            psi_b25[2].re, psi_b25[2].im,
            psi_b25[3].re, psi_b25[3].im,
            psi_b25[4].re, psi_b25[4].im,
        );
    }
    // Verify psi data at FFI boundary for mid-band
    // BEFORE any processing by WavefunctionSet/Gpu::from_host
    if n_bands > 1 {
        let mid = n_bands / 2;
        let psi_mid = &psi_host[mid * n_pw..(mid + 1) * n_pw];
        let l2sq: f64 = psi_mid.iter().map(|c| c.norm_sqr()).sum();
        eprintln!("[Diag-FFI-raw] band {mid} (0-based, mid-band): L2²={:.6e} first5=[({:+.6e},{:+.6e}), ({:+.6e},{:+.6e}), ({:+.6e},{:+.6e}), ({:+.6e},{:+.6e}), ({:+.6e},{:+.6e})]",
            l2sq,
            psi_mid[0].re, psi_mid[0].im,
            psi_mid[1].re, psi_mid[1].im,
            psi_mid[2].re, psi_mid[2].im,
            psi_mid[3].re, psi_mid[3].im,
            psi_mid[4].re, psi_mid[4].im,
        );
    }

    // Lazily initialise VnlBatchData per spin on the first call to step_inner,
    // where real psi and n_bands are available.  init_inner cannot create this
    // because CASTEP doesn't pass wavefunctions at init time.
    if kd.vnl[isp].is_none() {
        let mut pcie = PcieAccount::default();
        // Use precompute_with_d_override with fine_grid for D-screening.
        // CASTEP's nlpot_calculate_d (nlpot.f90:352) uses poten%real_fine_pot
        // (fine grid), forward FFT on fine grid normalized by 1/N_fine, and Q(G)
        // on fine half-grid.  Passing Some(&h.fine_grid) builds a fine-grid
        // screening cache that matches CASTEP's convention.
        // Empirically: fine-grid D-screening gives iter-2 spin-polarised energy
        // (-7160.63 eV) close to reference (-7160.23 eV); wave-grid gives -7187 eV.
        let shared_for_this_spin = kd.shared_vnl.clone();
        kd.vnl[isp] = Some(VnlBatchData::precompute_with_d_override(
            &kd.pw_coords, &h.pots, &h.cell, &h.wave_grid, Some(&h.fine_grid), &kd.k_point,
            &psi_host, n_bands, n_pw, None, None, None, shared_for_this_spin,
            h.handle_shared_vnl.clone(),
            None,  // solver_thunk — no Woodbury in FFI path
            &h.stream, &mut pcie, &h.blas, &h.kernels,
        ).map_err(|e| { eprintln!("[chemrust] VnlBatchData init failed: {e}"); CHEM_EIG_CUDA_ERROR })?);
        kd.shared_vnl = Some(kd.vnl[isp].as_ref().unwrap().shared.clone());
    }

    let wfn = WavefunctionSet::<ColumnDistributed>::new(psi_host, n_bands, n_pw);
    let psi_gpu = Gpu::from_host(&wfn, &h.stream).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Upload V_eff
    // CASTEP's real_fine_pot is on the FINE grid, Fortran-ordered (x fastest).
    // CASTEP internally calls pot_interpolate → basis_real_fine_to_std_grid to
    // downsample V_eff from fine to standard grid before Hamiltonian application.
    // We replicate that downsampling here so the eigensolver operates on the
    // same standard-grid V_eff that CASTEP uses internally.
    let gs_fine = (h.ngx * h.ngy * h.ngz) as usize;
    let ve_host: Vec<f64> = unsafe { std::slice::from_raw_parts(v_eff_data, gs_fine) }.to_vec();
    let ve_norm: f64 = ve_host.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    #[cfg(feature = "scf_diag")]
    {
        let ve_min = ve_host.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let ve_max = ve_host.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let ve_mean = ve_host.iter().sum::<f64>() / gs_fine as f64;
        eprintln!("[Diag-Veff] ngx={} ngy={} ngz={} gs_fine={} ve_min={:.6e} ve_max={:.6e} ve_mean={:.6e}",
            h.ngx, h.ngy, h.ngz, gs_fine, ve_min, ve_max, ve_mean);
        eprintln!("[Diag-Veff] first 5 raw: {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}",
            ve_host[0], ve_host[1], ve_host[2], ve_host[3], ve_host[4]);
    }

    // V_eff GPU caching: DISABLED.
    // ve_norm is max(|V_eff|) on the FINE grid, dominated by frozen-core
    // pseudopotentials near nuclei (~8.55 Ha).  Valence charge redistribution
    // changes V_eff shape but leaves the fine-grid maximum essentially unchanged
    // (< 1e-8 Ha between SCF iterations).  This causes false cache HITs that
    // feed stale V_eff to the eigensolver, creating a growing density-potential
    // inconsistency that triggers catastrophic eigenvalue divergence at SCF
    // iteration 3.  The H2D transfer of V_eff (~1M doubles) is negligible
    // compared to the eigensolver cost.
    let cache_reuse = false;

    // Downsample V_eff from fine grid to standard (wave) grid.
    // arr_ix_fast has shape [ngx, ngy, ngz] with arr[[ix, iy, iz]] = V_fine(ix, iy, iz).
    let ngx_f = h.ngx as usize;
    let ngy_f = h.ngy as usize;
    let ngz_f = h.ngz as usize;
    let arr_ix_fast = crate::device::unflatten_f64(ve_host, &[ngx_f, ngy_f, ngz_f]);

    // Downsample: FFT on fine grid → truncate high G → IFFT on wave grid.
    let veff_wave = crate::downsample_array_to_wave_grid(
        &arr_ix_fast, &h.fine_grid, &h.wave_grid,
    ).map_err(|e| { eprintln!("[chemrust] V_eff downsampling failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    // veff_wave is EffectivePotential(FineGridArray(wave_arr)) on the wave grid.
    // The Array3 inside has shape (ngz_std, ngy_std, ngx_std) — see the
    // downsample function which creates the output as Array3::zeros((ngz, ngy, ngx)).
    {
        let ve_w_min = veff_wave.as_fine_array().iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let ve_w_max = veff_wave.as_fine_array().iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let wave_shape = veff_wave.as_fine_array().shape();
        eprintln!("[chemrust-Veff] downsampled fine→wave: ve_min={:.6e} ve_max={:.6e} wave_shape=({},{},{}) gs_fine={} gs_wave={}",
            ve_w_min, ve_w_max, wave_shape[0], wave_shape[1], wave_shape[2], gs_fine, gs);
    }

    // Transpose V_eff to iz-innermost layout for cuFFT/scatter/gather.
    // The downsampled array is ix-innermost in memory (C order with x fastest).
    // cuFFT expects iz-innermost even on the standard grid, so we transpose.
    let wave_arr = veff_wave.as_fine_array();
    let ngx_s = h.ngx_std as usize;
    let ngy_s = h.ngy_std as usize;
    let ngz_s = h.ngz_std as usize;
    let arr_iz_fast = ndarray::Array3::from_shape_fn(
        (ngz_s, ngy_s, ngx_s),
        |(iz, iy, ix)| wave_arr[[ix, iy, iz]],
    );
    let veff_wave_iz = EffectivePotential(FineGridArray(arr_iz_fast));

    #[cfg(feature = "scf_diag")]
    {
        let ve_w_min = veff_wave.as_fine_array().iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let ve_w_max = veff_wave.as_fine_array().iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        eprintln!("[Diag-Veff] downsampled to wave grid: ve_min={:.6e} ve_max={:.6e} wave_grid=({},{},{})",
            ve_w_min, ve_w_max, ngx_s, ngy_s, ngz_s);
    }

    let v_eff_gpu = if cache_reuse {
        if cfg!(feature = "scf_diag") { eprintln!("[chemrust] V_eff cache HIT norm={:.6e}", ve_norm); }
        let mut slice: CudaSlice<f64> = h.stream.alloc_zeros::<f64>(gs).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        h.stream.memcpy_dtod(h.v_eff_cached[isp].as_ref().unwrap(), &mut slice).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        Gpu::<EffectivePotential> {
            slice,
            shape: vec![ngz_s, ngy_s, ngx_s],
            ctx: h.ctx.clone(),
            _marker: std::marker::PhantomData,
        }
    } else {
        if cfg!(feature = "scf_diag") { eprintln!("[chemrust] V_eff cache MISS norm={:.6e} prev={:.6e}", ve_norm, h.v_eff_norm[isp]); }
        let vg = Gpu::from_host(&veff_wave_iz, &h.stream).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        // Update cache: preserve GPU copy for next SCF step
        if h.v_eff_cached[isp].is_none() {
            let mut cache = h.stream.alloc_zeros::<f64>(gs).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            h.stream.memcpy_dtod(vg.as_device_slice(), &mut cache).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            h.v_eff_cached[isp] = Some(cache);
        } else {
            h.stream.memcpy_dtod(vg.as_device_slice(), h.v_eff_cached[isp].as_mut().unwrap()).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        }
        h.v_eff_norm[isp] = ve_norm;
        vg
    };

    // Re-screen D matrices using FINE-grid V_eff, matching CASTEP.
    // CASTEP's nlpot_calculate_d (nlpot.f90:352) uses poten%real_fine_pot
    // (fine grid), forward FFT on fine grid with 1/N_fine normalization, and
    // Q(G) on fine half-grid.  The fine-grid screening cache was built above
    // via precompute_with_d_override.  Pass the raw fine-grid V_eff in its
    // original ix-innermost layout for D-screening.
    kd.vnl[isp].as_mut().unwrap().rescreen_d(&arr_ix_fast, &h.stream, &h.kernels, &h.blas)
        .map_err(|e| { eprintln!("[chemrust] D re-screen failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    // Upload FFT index.  CASTEP passes 1-based Fortran grid indices encoded
    // on the STANDARD grid (ngx_std × ngy_std × ngz_std): idx = 1 + ix + ngx_std*iy + ngx_std*ngy_std*iz.
    // Our FFT plan is now on the STANDARD grid to match CASTEP's V_eff after
    // pot_interpolate downsamples from fine to standard.  Decode the standard-grid
    // position and re-encode with iz-innermost layout for cuFFT on the standard grid.
    let ngx_s = h.ngx_std;
    let ngy_s = h.ngy_std;
    let ngz_s = h.ngz_std;
    let fft_idx: Vec<i32> = unsafe { std::slice::from_raw_parts(fft_idx_data as *const c_int, n_pw) }
        .iter()
        .map(|&idx_1based| {
            // Decode CASTEP's ix-innermost STANDARD-grid position (0-based)
            let idx0 = (idx_1based - 1).max(0);
            let ix = idx0 % ngx_s;
            let iy = (idx0 / ngx_s) % ngy_s;
            let iz = idx0 / (ngx_s * ngy_s);
            // Re-encode with iz-innermost layout on STANDARD grid (no scaling needed)
            iz + ngz_s * (iy + ngy_s * ix)
        })
        .collect();
    debug_assert!(
        fft_idx.iter().all(|&i| i >= 0 && (i as usize) < gs),
        "fft_idx out of range after 1→0+transpose: min={} max={} gs={gs}",
        fft_idx.iter().min().unwrap_or(&0),
        fft_idx.iter().max().unwrap_or(&0),
    );
    let mut fft_idx_dev: CudaSlice<i32> = h.stream.alloc_zeros(n_pw).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    h.stream.memcpy_htod(&fft_idx, &mut fft_idx_dev).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // ---- Davidson diagonalisation (default eigensolver path) ----
    // Pre-borrow handle components to avoid borrow conflicts
    let blas = &h.blas;
    let solver = &h.solver;
    let kernels = &h.kernels;
    let stream = &h.stream;
    let ctx = &h.ctx;

    let n_elem = n_pw * n_bands;
    let kinetic_raw: CudaSlice<f64> = stream.clone_htod(&ke_castep).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let kinetic_dev = KineticPreconditioner::new(kinetic_raw);

    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        h.ngx_std, h.ngy_std, h.ngz_std, n_bands_i32, stream.clone(),
    ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Davidion requires grid buffer for Hamiltonian application
    let mut grid_buf: CudaSlice<CudaComplex> = stream.alloc_zeros(n_bands * gs)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Davidson superspace memory requirement:
    //   nblock ≈ ceil(2·√n_bands), rounded to next even integer
    //   superspace_size = 6 (CASTEP hamiltonian.f90:1019-1063)
    //   Each block allocates superspace_max_bands = superspace_size · nblock
    //   → super_wvfn and h_super_wvfn each of size n_pw · superspace_max_bands
    //   davidson_diagonalise handles internal buffer allocation.

    // Initialise TPA preconditioner (NVRTC kernel compilation at init time)
    let tpa = TpaPreconditioner::new(ctx).map_err(|e| {
        eprintln!("[chemrust] TpaPreconditioner init failed: {e}");
        CHEM_EIG_CUDA_ERROR
    })?;

    let psi_init = PwCoefficients::new(psi_gpu.as_device_slice().clone());

    // FFI cold-start diagnostic: verify psi data at the boundary before
    // Davidson solve.  Download band 0 psi to CPU and check L2² + first 5
    // G-vector entries.  Compare against CPU CASTEP's [CASTEP-A1] values.
    {
        use crate::device::CudaComplex;
        let n_pw_psi = psi_gpu.as_device_slice().len() / n_bands;
        for &b in &[0usize, 104, 105] {
            if b >= n_bands { continue; }
            let start = b * n_pw_psi;
            let end = (start + n_pw).min(start + n_pw_psi);
            let band_cpu: Vec<CudaComplex> = stream
                .clone_dtoh(&psi_gpu.as_device_slice().slice(start..end))
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            let l2_sq: f64 = band_cpu.iter().take(n_pw).map(|c| c.x*c.x + c.y*c.y).sum();
            eprintln!("[FFI-diag] band {b}: psi L2²={:.6e} first5=[{:?}]",
                l2_sq,
                band_cpu.iter().take(5).map(|c| (c.x, c.y)).collect::<Vec<_>>());
        }
        let v_lo: Vec<f64> = stream
            .clone_dtoh(&v_eff_gpu.as_device_slice().slice(0..5usize.min(v_eff_gpu.as_device_slice().len())))
            .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        eprintln!("[FFI-diag] V_eff[0..5]={:?}", v_lo);
    }

    let davidson_result = unsafe {
        davidson_diagonalise()
            .psi_init(&psi_init)
            .v_eff_dev(v_eff_gpu.as_device_slice())
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(&fft_idx_dev)
            .vnl_data(kd.vnl[isp].as_ref().unwrap())
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(gs)
            .inv_ntotal(inv_ntotal)
            .fft_plan(&fft_plan)
            .tol_abs(1e-8)
            .max_outer_iter(2)
            .min_outer_iter(0)
            .blas(blas)
            .solver(solver)
            .kernels(kernels)
            .tpa_preconditioner(&tpa)
            .stream(stream)
            .ctx(ctx)
            .gamma_point(have_gamma)
            .call()
            .map_err(|e| { eprintln!("[chemrust] davidson_diagonalise failed: {e}"); CHEM_EIG_CUDA_ERROR })?
    };

    // Compute H·psi_new for hpsi_out.
    // Reuse the BetaPhiCache from the Davidson solve — psi_out is the same
    // psi that compute_all ran against at the end of the last outer iteration,
    // so cached β^H·ψ projections are still valid.
    let psi_out_pw = PwCoefficients::new(davidson_result.psi_out.clone());
    let mut hpsi_new = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?);
    let mut post_cache: Option<crate::eigensolver::beta_phi_cache::BetaPhiCache> = None;
    unsafe {
        let h_builder = apply_full_hamiltonian()
            .psi_dev(&psi_out_pw)
            .v_eff_dev(v_eff_gpu.as_device_slice())
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(&fft_idx_dev)
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(gs)
            .inv_ntotal(inv_ntotal)
            .fft_plan(&fft_plan)
            .hpsi_dev(&mut hpsi_new)
            .grid_dev(&mut grid_buf)
            .vnl_data(kd.vnl[isp].as_ref().unwrap())
            .blas(blas)
            .kernels(kernels)
            .stream(stream);
        if let Some(ref mut cache) = post_cache {
            h_builder.maybe_beta_phi_cache(cache).call()
        } else {
            h_builder.call()
        }
        .map_err(|e| { eprintln!("[chemrust] H*psi after Davidson failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    }

    // Free FFT grid buffer before D2H transfers to reduce peak VRAM.
    // grid_buf holds n_bands * gs complex elements (~13.6 GB for Cu111_CO
    // with fine grid).  Dropping here frees ~13.6 GB before we allocate
    // psi_host_out and hpsi_host on the host.
    drop(grid_buf);

    // Download and write back to CASTEP buffers
    let psi_host_out: Vec<CudaComplex> = stream.clone_dtoh(&davidson_result.psi_out)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let hpsi_host: Vec<CudaComplex> = stream.clone_dtoh(&*hpsi_new)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Write back with max_n_pw stride — CASTEP's wvfn%coeffs(:,:,nk,ns) has
    // leading dimension max_plane_waves, not this k-point's n_pw.
    let raw_psi_out = unsafe { std::slice::from_raw_parts_mut(psi_data, max_n_pw * n_bands) };
    for b in 0..n_bands {
        let start = b * max_n_pw;
        raw_psi_out[start..start + n_pw].copy_from_slice(&psi_host_out[b * n_pw..(b + 1) * n_pw]);
    }
    let raw_hpsi_out = unsafe { std::slice::from_raw_parts_mut(hpsi_out, max_n_pw * n_bands) };
    for b in 0..n_bands {
        let start = b * max_n_pw;
        raw_hpsi_out[start..start + n_pw].copy_from_slice(&hpsi_host[b * n_pw..(b + 1) * n_pw]);
    }
    unsafe {
        std::slice::from_raw_parts_mut(eigenvalues_ptr, n_bands)
            .copy_from_slice(&davidson_result.eigenvalues);
    }

    // Diagnostic: print the returned eigenvalues for the first and last bands
    #[cfg(feature = "scf_diag")]
    {
        if n_bands > 0 {
            let ev = &davidson_result.eigenvalues;
            eprintln!(
                "[chemrust-eig-return] ispin={} ikpt={} n_bands={} ev[0]={:.10e} ev[mid]={:.10e} ev[last]={:.10e} n_locked={}/{}",
                isp, ik, n_bands, ev[0], ev[n_bands/2], ev[n_bands-1],
                davidson_result.n_locked, n_bands,
            );
        }
    }

    // Set converged flag: true if all bands locked (residual below tolerance)
    let all_converged = davidson_result.n_locked >= n_bands;
    unsafe { *converged = all_converged as c_int; };

    Ok(())
}

// ---- Chebyshev eigensolver variant -----------------------------------------

#[cfg(feature = "chebyshev")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_eigensolve_step_chebyshev(
    handle: *mut c_void,
    psi_data: *mut CudaComplex, v_eff_data: *const c_double,
    kinetic_data: *const c_double, fft_idx_data: *const c_int,
    eigenvalues_ptr: *mut c_double, hpsi_out: *mut CudaComplex,
    npw: c_int, nbands: c_int, ikpt: c_int, ispin: c_int,
    max_deg: c_int,
    converged: *mut c_int,
) -> c_int {
    match unsafe { step_inner_chebyshev(handle, psi_data, v_eff_data, kinetic_data, fft_idx_data,
        eigenvalues_ptr, hpsi_out, npw, nbands, ikpt, ispin, max_deg, converged) }
    {
        Ok(()) => CHEM_EIG_OK,
        Err(c) => c,
    }
}

#[cfg(feature = "chebyshev")]
#[allow(clippy::too_many_arguments)]
unsafe fn step_inner_chebyshev(
    handle: *mut c_void,
    psi_data: *mut CudaComplex, v_eff_data: *const c_double,
    kinetic_data: *const c_double, fft_idx_data: *const c_int,
    eigenvalues_ptr: *mut c_double, hpsi_out: *mut CudaComplex,
    npw: c_int, nbands: c_int, ikpt: c_int, ispin: c_int,
    max_deg: c_int,
    converged: *mut c_int,
) -> Result<(), c_int> {
    use crate::eigensolver::chebyshev::{FilterMode, chebfi_run_rust};
    use crate::eigensolver::rayleigh_ritz::rayleigh_ritz;

    let h = unsafe { (handle as *mut ChemrustHandle).as_mut() }.ok_or(CHEM_EIG_NULL_HANDLE)?;
    if psi_data.is_null() || v_eff_data.is_null() || kinetic_data.is_null()
        || fft_idx_data.is_null() || eigenvalues_ptr.is_null() || hpsi_out.is_null() || converged.is_null()
    { return Err(CHEM_EIG_NULL_HANDLE); }

    let ik = ikpt as usize;
    if ik >= h.kpts.len() { return Err(CHEM_EIG_CUDA_ERROR); }
    let isp = ispin as usize;
    if isp >= h.nspins { return Err(CHEM_EIG_CUDA_ERROR); }
    let n_pw = npw as usize;
    let n_bands = nbands as usize;
    let kd = &mut h.kpts[ik];
    let have_gamma = kd.kpoint_frac.iter().all(|&c| c.abs() < 1e-12);

    // Use CASTEP-provided kinetic energies
    let ke_castep: Vec<f64> = unsafe { std::slice::from_raw_parts(kinetic_data, n_pw) }.to_vec();

    // Upload psi with max_n_pw stride
    let max_n_pw = h.max_n_pw;
    let raw_psi = unsafe { std::slice::from_raw_parts(psi_data as *const CudaComplex, max_n_pw * n_bands) };
    let mut psi_host: Vec<num_complex::Complex64> = Vec::with_capacity(n_pw * n_bands);
    for b in 0..n_bands {
        let start = b * max_n_pw;
        psi_host.extend(raw_psi[start..start + n_pw].iter().map(|c| num_complex::Complex64::new(c.x, c.y)));
    }

    // Lazily initialise VnlBatchData
    if kd.vnl[isp].is_none() {
        let mut pcie = PcieAccount::default();
        let shared_for_this_spin = kd.shared_vnl.clone();
        kd.vnl[isp] = Some(VnlBatchData::precompute_with_d_override(
            &kd.pw_coords, &h.pots, &h.cell, &h.wave_grid, Some(&h.fine_grid), &kd.k_point,
            &psi_host, n_bands, n_pw, None, None, None, shared_for_this_spin,
            h.handle_shared_vnl.clone(),
            None,
            &h.stream, &mut pcie, &h.blas, &h.kernels,
        ).map_err(|e| { eprintln!("[chemrust-chebyshev] VnlBatchData init failed: {e}"); CHEM_EIG_CUDA_ERROR })?);
        kd.shared_vnl = Some(kd.vnl[isp].as_ref().unwrap().shared.clone());
    }

    let wfn = WavefunctionSet::<ColumnDistributed>::new(psi_host, n_bands, n_pw);
    let psi_gpu = Gpu::from_host(&wfn, &h.stream).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Upload V_eff and downsample fine->standard grid (same as step_inner)
    let gs_fine = (h.ngx * h.ngy * h.ngz) as usize;
    let ve_host: Vec<f64> = unsafe { std::slice::from_raw_parts(v_eff_data, gs_fine) }.to_vec();

    let ngx_f = h.ngx as usize;
    let ngy_f = h.ngy as usize;
    let ngz_f = h.ngz as usize;
    let arr_ix_fast = crate::device::unflatten_f64(ve_host, &[ngx_f, ngy_f, ngz_f]);

    let veff_wave = crate::downsample_array_to_wave_grid(
        &arr_ix_fast, &h.fine_grid, &h.wave_grid,
    ).map_err(|e| { eprintln!("[chemrust-chebyshev] V_eff downsampling failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    let wave_arr = veff_wave.as_fine_array();
    let ngx_s = h.ngx_std as usize;
    let ngy_s = h.ngy_std as usize;
    let ngz_s = h.ngz_std as usize;
    let arr_iz_fast = ndarray::Array3::from_shape_fn(
        (ngz_s, ngy_s, ngx_s),
        |(iz, iy, ix)| wave_arr[[ix, iy, iz]],
    );
    let veff_wave_iz = EffectivePotential(FineGridArray(arr_iz_fast));
    let v_eff_gpu = Gpu::from_host(&veff_wave_iz, &h.stream)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Re-screen D matrices
    kd.vnl[isp].as_mut().unwrap().rescreen_d(&arr_ix_fast, &h.stream, &h.kernels, &h.blas)
        .map_err(|e| { eprintln!("[chemrust-chebyshev] D re-screen failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    // Upload FFT index (same transpose as step_inner)
    let fft_idx: Vec<i32> = unsafe { std::slice::from_raw_parts(fft_idx_data as *const c_int, n_pw) }
        .iter()
        .map(|&idx_1based| {
            let idx0 = (idx_1based - 1).max(0);
            let ix = idx0 % ngx_s as i32;
            let iy = (idx0 / ngx_s as i32) % ngy_s as i32;
            let iz = idx0 / (ngx_s as i32 * ngy_s as i32);
            iz + ngz_s as i32 * (iy + ngy_s as i32 * ix)
        })
        .collect();
    let mut fft_idx_dev: CudaSlice<i32> = h.stream.alloc_zeros(n_pw)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    h.stream.memcpy_htod(&fft_idx, &mut fft_idx_dev)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // ---- Chebyshev filtering -------------------------------------------------
    let blas = &h.blas;
    let solver = &h.solver;
    let kernels = &h.kernels;
    let stream = &h.stream;
    let ctx = &h.ctx;

    // ecut from maximum kinetic energy (physical energy cutoff in Ha)
    let ecut = ke_castep.iter().cloned().fold(0.0_f64, f64::max);

    // Compute V_eff statistics for spectral bounds
    let ve_min = arr_ix_fast.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let ve_max = arr_ix_fast.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Tolerance for residual convergence
    // ABINIT m_chebfi2.F90: ndeg capped at 40 (Phase 3 hard cap), not CASTEP's
    // max_deg (=12).  Conservative oracle (mode 1) caps ALL bands at
    // min(global_oracle, ndeg_filter_max) — with max_deg=12, bands needing
    // >12 iterations get under-filtered, causing eigenvalue oscillation.
    let tolerance = 1e-6_f64;
    let ndeg_filter_max = 40usize; // ABINIT hard cap (not CASTEP's max_deg=12)
    let oracle_mode = 0usize;      // ABINIT default: oracle disabled
    let oracle_factor = 0.0_f64;   // unused when oracle=0
    let oracle_min_occ = 0.0_f64;  // unused when oracle=0

    // Chebyshev filtering — SinvHKeepHEig for USPP (S⁻¹·H operator).
    // ABINIT m_vtorho.F90:610: nnsclo_now=2 for istep<=2 (cold start).
    // Zhou 2014 Algorithm 5.1: 3-4 iters of filter→orthonormalize→RR to
    // converge the initial random subspace before density reconstruction.
    let mut pcie = PcieAccount::default();

    // Pass 1: initial filter + RR
    let (pf1, mut hf1, _ritz1, _res1, _ndeg1) = chebfi_run_rust(
        &psi_gpu,
        v_eff_gpu.as_device_slice(),
        &h.wave_grid, &kd.pw_coords, kd.vnl[isp].as_ref().unwrap(),
        &fft_idx_dev,
        ecut, ve_min, ve_max,
        tolerance, None, None,
        ndeg_filter_max, oracle_mode, oracle_factor, oracle_min_occ,
        kernels, &mut pcie, blas, solver, stream, ctx,
        FilterMode::SinvHKeepHEig, Some(&ke_castep),
        0, n_bands, !have_gamma,
    ).map_err(|e| { eprintln!("[chemrust-chebyshev] chebfi_run_rust pass1 failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    let (psi_col1, eig_cpu1, _beta1) = rayleigh_ritz(
        &pf1, &mut hf1, kd.vnl[isp].as_ref().unwrap(),
        n_bands, n_pw, kernels, &mut pcie, solver, blas, stream, ctx,
        None, None, None,
    ).map_err(|e| { eprintln!("[chemrust-chebyshev] rayleigh_ritz pass1 failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    // Pass 2: refine subspace (ColumnDistributed from pass1 as input)
    let (pf2, mut hf2, _ritz2, _res2, _ndeg2) = chebfi_run_rust(
        &psi_col1,
        v_eff_gpu.as_device_slice(),
        &h.wave_grid, &kd.pw_coords, kd.vnl[isp].as_ref().unwrap(),
        &fft_idx_dev,
        ecut, ve_min, ve_max,
        tolerance, None, None,
        ndeg_filter_max, oracle_mode, oracle_factor, oracle_min_occ,
        kernels, &mut pcie, blas, solver, stream, ctx,
        FilterMode::SinvHKeepHEig, Some(&ke_castep),
        0, n_bands, !have_gamma,
    ).map_err(|e| { eprintln!("[chemrust-chebyshev] chebfi_run_rust pass2 failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    let (psi_col, eig_cpu, _beta2) = rayleigh_ritz(
        &pf2, &mut hf2, kd.vnl[isp].as_ref().unwrap(),
        n_bands, n_pw, kernels, &mut pcie, solver, blas, stream, ctx,
        None, None, None,
    ).map_err(|e| { eprintln!("[chemrust-chebyshev] rayleigh_ritz pass2 failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    // ---- Compute H·psi for hpsi_out (CASTEP stores into wvfn_gradient) ----
    let gs = (h.ngx_std * h.ngy_std * h.ngz_std) as usize;
    let inv_ntotal = 1.0 / gs as f64;
    let n_elem = n_pw * n_bands;
    let kinetic_raw: CudaSlice<f64> = stream.clone_htod(&ke_castep).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let kinetic_dev = KineticPreconditioner::new(kinetic_raw);
    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        h.ngx_std as i32, h.ngy_std as i32, h.ngz_std as i32, n_bands as i32, stream.clone(),
    ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut grid_buf: CudaSlice<CudaComplex> = stream
        .alloc_zeros(n_bands * gs)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut hpsi_dev = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?);
    let psi_out_pw = PwCoefficients::new(psi_col.as_device_slice().clone());
    unsafe {
        apply_full_hamiltonian()
            .psi_dev(&psi_out_pw)
            .v_eff_dev(v_eff_gpu.as_device_slice())
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(&fft_idx_dev)
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(gs)
            .inv_ntotal(inv_ntotal)
            .fft_plan(&fft_plan)
            .hpsi_dev(&mut hpsi_dev)
            .grid_dev(&mut grid_buf)
            .vnl_data(kd.vnl[isp].as_ref().unwrap())
            .blas(blas)
            .kernels(kernels)
            .stream(stream)
            .call()
    }.map_err(|e| { eprintln!("[chemrust-chebyshev] H*psi after Chebyshev+RR failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    drop(grid_buf);

    // ---- Synchronize before D2H (cudarc 0.19.7 async memory) ----
    stream.synchronize().map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Download and write back psi
    let psi_host_out: Vec<CudaComplex> = stream.clone_dtoh(psi_col.as_device_slice())
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Write eigenvalues
    unsafe {
        std::slice::from_raw_parts_mut(eigenvalues_ptr, n_bands)
            .copy_from_slice(&eig_cpu.0);
    }

    // Write back psi with max_n_pw stride
    let raw_psi_out = unsafe { std::slice::from_raw_parts_mut(psi_data, max_n_pw * n_bands) };
    for b in 0..n_bands {
        let start = b * max_n_pw;
        raw_psi_out[start..start + n_pw]
            .copy_from_slice(&psi_host_out[b * n_pw..(b + 1) * n_pw]);
    }

    // Download and write back H·psi for wvfn_gradient
    let hpsi_host: Vec<CudaComplex> = stream.clone_dtoh(&*hpsi_dev)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let raw_hpsi_out = unsafe { std::slice::from_raw_parts_mut(hpsi_out, max_n_pw * n_bands) };
    for b in 0..n_bands {
        let start = b * max_n_pw;
        raw_hpsi_out[start..start + n_pw]
            .copy_from_slice(&hpsi_host[b * n_pw..(b + 1) * n_pw]);
    }

    // converged flag: unused by CASTEP (declared+passed but never read in electronic.f90)
    unsafe { *converged = 1; };

    Ok(())
}
