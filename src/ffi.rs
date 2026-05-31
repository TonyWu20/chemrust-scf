// ---------------------------------------------------------------------------
// FFI: CASTEP Fortran ↔ Rust GPU eigensolver
// ---------------------------------------------------------------------------

use std::ffi::{c_char, c_void, CStr};
use std::os::raw::{c_double, c_int};
use std::sync::Arc;

use chemrust_hamiltonian_core::{CellGeometry, GVectorGrid, PseudopotentialSet, RealLattice, RecipLattice};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::cublas::sys::cublasOperation_t;

use crate::device::blas::{BlasHandle, ZgemmConfig};
use crate::device::fft::BatchedFftPlan3d;
use crate::device::pcie::PcieAccount;
use crate::device::solver::SolverHandle;
use crate::device::{CudaComplex, DeviceMapped, Gpu};
use crate::eigensolver::chebyshev::{chebyshev_filter, FilterMode};
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::rayleigh_ritz::rayleigh_ritz_with_matrices;
use crate::eigensolver::vnl_data::VnlBatchData;
use crate::layout::{ColumnDistributed, WavefunctionSet};
use crate::types::{EffectivePotential, Error, FineGridArray, KPoint};

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
    vnl: VnlBatchData,
    wave_grid: GVectorGrid,
    pw_coords: Vec<[i32; 3]>,
    pcie: PcieAccount,
}

// ---- Opaque handle ---------------------------------------------------------

struct ChemrustHandle {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: BlasHandle,
    solver: SolverHandle,
    kernels: CudaKernelSet,
    ngx: i32, ngy: i32, ngz: i32,
    fft_plan: Option<BatchedFftPlan3d>,
    kpts: Vec<KptData>,
    /// Cached GPU copy of V_eff for skip-upload optimization.
    v_eff_cached: Option<CudaSlice<f64>>,
    /// Max-norm of the cached V_eff, used for change detection.
    v_eff_norm: f64,
}

impl ChemrustHandle {
    fn fft(&mut self, n_bands: i32) -> Result<&BatchedFftPlan3d, c_int> {
        let ok = self.fft_plan.as_ref().map(|p| p.batch() == n_bands).unwrap_or(false);
        if !ok {
            self.fft_plan = Some(BatchedFftPlan3d::plan_batched_c2c(
                self.ngx, self.ngy, self.ngz, n_bands, self.stream.clone(),
            ).map_err(|_| CHEM_EIG_CUDA_ERROR)?);
        }
        Ok(self.fft_plan.as_ref().unwrap())
    }
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
    handle_out: *mut *mut c_void,
) -> c_int {
    if handle_out.is_null() { return CHEM_EIG_NULL_HANDLE; }
    let h = match init_inner(num_species, species_symbols, species_pots,
        real_lattice, recip_lattice, num_ions, ion_species, ion_positions,
        nkpts, num_pw_per_kpt, gvec_all_kpt, pw_grid_idx, kpt_coords, ngx, ngy, ngz)
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

    let mut kpts = Vec::with_capacity(nk);
    for ik in 0..nk {
        let n_pw = npwk[ik] as usize;
        let kf = [kfrac[3*ik], kfrac[3*ik+1], kfrac[3*ik+2]];
        let kpt = KPoint { coords: kf };

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
        let mut pw_coords = Vec::with_capacity(n_pw);
        for ipw in 0..n_pw {
            let idx_1based = gidx[ik*maxpw + ipw];
            pw_coords.push(fft_idx_to_coord(idx_1based, ngx, ngy, ngz));
        }

        // Diagnostic: compare our pw_coords→Cartesian vs CASTEP's gvec
        // CASTEP's gvec_all_kpt is pw_g_vector = G+k (Cartesian)
        // Our pw_coords are G Miller indices → G_cart = G * rcip
        // Difference should be k_cart
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

        let wg = GVectorGrid::new(ngx as usize, ngy as usize, ngz as usize, RecipLattice::from_inner(rcip));
        let psi_dummy = vec![num_complex::Complex64::new(0.0, 0.0); n_pw];
        let mut pcie = PcieAccount::default();

        let vnl = VnlBatchData::precompute(
            &pw_coords, &pots, &cell, &wg, &kpt,
            &psi_dummy, 1, n_pw, None, None,
            &stream, &mut pcie, &blas, &kernels, &solver,
        ).map_err(|e| { eprintln!("[chemrust] precompute kpt {ik} failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

        kpts.push(KptData { vnl, wave_grid: wg, pw_coords, pcie });
    }

    Ok(Box::into_raw(Box::new(ChemrustHandle {
        ctx, stream, blas, solver, kernels,
        ngx, ngy, ngz, fft_plan: None, kpts,
        v_eff_cached: None,
        v_eff_norm: 0.0,
    })))
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
    npw: c_int, nbands: c_int, ikpt: c_int,
    max_deg: c_int,
    converged: *mut c_int,
) -> c_int {
    match unsafe { step_inner(handle, psi_data, v_eff_data, kinetic_data, fft_idx_data,
        eigenvalues_ptr, hpsi_out, npw, nbands, ikpt, max_deg, converged) }
    {
        Ok(()) => CHEM_EIG_OK,
        Err(c) => c,
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn step_inner(
    handle: *mut c_void,
    psi_data: *mut CudaComplex, v_eff_data: *const c_double,
    kinetic_data: *const c_double, fft_idx_data: *const c_int,
    eigenvalues_ptr: *mut c_double, hpsi_out: *mut CudaComplex,
    npw: c_int, nbands: c_int, ikpt: c_int,
    max_deg: c_int,
    converged: *mut c_int,
) -> Result<(), c_int> {
    let h = unsafe { (handle as *mut ChemrustHandle).as_mut() }.ok_or(CHEM_EIG_NULL_HANDLE)?;
    if psi_data.is_null() || v_eff_data.is_null() || kinetic_data.is_null()
        || fft_idx_data.is_null() || eigenvalues_ptr.is_null() || hpsi_out.is_null() || converged.is_null()
    { return Err(CHEM_EIG_NULL_HANDLE); }

    let ik = ikpt as usize;
    if ik >= h.kpts.len() { return Err(CHEM_EIG_CUDA_ERROR); }
    let n_pw = npw as usize;
    let n_bands = nbands as usize;
    let ndeg = max_deg as usize;
    let gs = (h.ngx * h.ngy * h.ngz) as usize;
    let n_pw_i32 = n_pw as i32;
    let n_bands_i32 = n_bands as i32;
    let n_elem_i32 = (n_pw * n_bands) as i32;
    let inv_ntotal = 1.0 / gs as f64;
    let kd = &mut h.kpts[ik];

    // Use CASTEP-provided kinetic energies (pw_ek_data = 0.5*|G+k|^2)
    let ke_castep: Vec<f64> = unsafe { std::slice::from_raw_parts(kinetic_data, n_pw) }.to_vec();

    // Diagnostic: verify KE consistency
    {
        let n_print = kd.pw_coords.len().min(5);
        let ke_rust = crate::eigensolver::chebyshev::compute_kinetic_energies(&kd.pw_coords, kd.wave_grid.recip_lattice());
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

    // Upload psi
    let psi_host: Vec<num_complex::Complex64> = unsafe {
        std::slice::from_raw_parts(psi_data as *const CudaComplex, n_pw * n_bands)
    }.iter().map(|c| num_complex::Complex64::new(c.x, c.y)).collect();
    let wfn = WavefunctionSet::<ColumnDistributed>::new(psi_host, n_bands, n_pw);
    let psi_gpu = Gpu::from_host(&wfn, &h.stream).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Upload V_eff
    // CASTEP's real_fine_pot is Fortran-ordered (x fastest). unflatten_f64
    // now expects Fortran order so the GPU upload via flatten_f64 produces
    // the correct x-fastest layout that cuFFT expects.
    let ve_host: Vec<f64> = unsafe { std::slice::from_raw_parts(v_eff_data as *const f64, gs) }.to_vec();
    let ve_raw = unsafe { std::slice::from_raw_parts(v_eff_data as *const f64, gs) };
    let ve_norm: f64 = ve_host.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let ve_min = ve_host.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let ve_max = ve_host.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let ve_mean = ve_host.iter().sum::<f64>() / gs as f64;
    eprintln!("[Diag-Veff] ngx={} ngy={} ngz={} gs={} ve_min={:.6e} ve_max={:.6e} ve_mean={:.6e}",
        h.ngx, h.ngy, h.ngz, gs, ve_min, ve_max, ve_mean);
    eprintln!("[Diag-Veff] first 5 raw: {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}",
        ve_host[0], ve_host[1], ve_host[2], ve_host[3], ve_host[4]);

    // V_eff GPU caching: skip H2D transfer if V_eff unchanged since last step.
    // Uses max-norm for cheap change detection (threshold 1e-8 Ha).
    let cache_reuse = h.v_eff_cached.as_ref().is_some_and(|_| (ve_norm - h.v_eff_norm).abs() < 1e-8);

    let arr = crate::device::unflatten_f64(ve_host, &[h.ngx as usize, h.ngy as usize, h.ngz as usize]);
    let veff = EffectivePotential(FineGridArray(arr));

    let v_eff_gpu = if cache_reuse {
        eprintln!("[chemrust] V_eff cache HIT norm={:.6e}", ve_norm);
        // Allocate fresh GPU buffer and copy from cache via memcpy_dtod.
        let mut slice: CudaSlice<f64> = h.stream.alloc_zeros::<f64>(gs).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        h.stream.memcpy_dtod(h.v_eff_cached.as_ref().unwrap(), &mut slice).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        Gpu::<EffectivePotential> {
            slice,
            shape: vec![h.ngx as usize, h.ngy as usize, h.ngz as usize],
            ctx: h.ctx.clone(),
            _marker: std::marker::PhantomData,
        }
    } else {
        eprintln!("[chemrust] V_eff cache MISS norm={:.6e} prev={:.6e}", ve_norm, h.v_eff_norm);
        // Verify round-trip: flatten back and compare
        let ve_rt: Vec<f64> = crate::device::flatten_f64(veff.0.as_array());
        let mut rt_err = 0.0f64;
        for i in 0..gs.min(10) {
            let d = (ve_rt[i] - ve_raw[i]).abs();
            rt_err = rt_err.max(d);
        }
        eprintln!("[Diag-Veff] round-trip max error (first 10): {:.3e}", rt_err);
        let vg = Gpu::from_host(&veff, &h.stream).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        // Verify GPU upload: read back from GPU and compare
        {
            let ve_gpu_back: Vec<f64> = h.stream.clone_dtoh(vg.as_device_slice())
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            let mut gpu_err = 0.0f64;
            for i in 0..gs.min(10) {
                let d = (ve_gpu_back[i] - ve_raw[i]).abs();
                gpu_err = gpu_err.max(d);
            }
            eprintln!("[Diag-Veff] GPU upload verified: max error (first 10): {:.3e}", gpu_err);
            eprintln!("[Diag-Veff] GPU first 5: {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}",
                ve_gpu_back[0], ve_gpu_back[1], ve_gpu_back[2], ve_gpu_back[3], ve_gpu_back[4]);
        }
        // Update cache: preserve GPU copy for next SCF step
        if h.v_eff_cached.is_none() {
            let mut cache = h.stream.alloc_zeros::<f64>(gs).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            h.stream.memcpy_dtod(vg.as_device_slice(), &mut cache).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            h.v_eff_cached = Some(cache);
        } else {
            h.stream.memcpy_dtod(vg.as_device_slice(), h.v_eff_cached.as_mut().unwrap()).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        }
        h.v_eff_norm = ve_norm;
        vg
    };

    // Re-screen D matrices using the current V_eff (must happen before
    // Hamiltonian application so V_NL reflects the updated potential).
    kd.vnl.rescreen_d(veff.as_fine_array(), &h.stream, &h.kernels, &h.blas)
        .map_err(|e| { eprintln!("[chemrust] D re-screen failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    // Upload FFT index
    let fft_idx: Vec<i32> = unsafe { std::slice::from_raw_parts(fft_idx_data as *const c_int, n_pw) }.to_vec();
    let mut fft_idx_dev: CudaSlice<i32> = h.stream.alloc_zeros(n_pw).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    h.stream.memcpy_htod(&fft_idx, &mut fft_idx_dev).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // ---- Direct Chebyshev filter (ABINIT-style) ----
    // Step 0: Compute generalized Rayleigh quotients and spectral bounds.
    let ecut = ke_castep.iter().cloned().fold(0.0f64, f64::max);

    // Pre-borrow handle components to avoid borrow conflicts
    let blas = &h.blas;
    let solver = &h.solver;
    let kernels = &h.kernels;
    let stream = &h.stream;
    let ctx = &h.ctx;

    let mut pcie_step = PcieAccount::default();
    let n_elem = n_pw * n_bands;
    let n_elem_i32 = n_elem as i32;
    let kinetic_dev: CudaSlice<f64> = stream.clone_htod(&ke_castep).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    let fft_plan = BatchedFftPlan3d::plan_batched_c2c(
        h.ngx, h.ngy, h.ngz, n_bands_i32, stream.clone(),
    ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Allocate GPU work buffers
    let mut psi_prev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut psi_curr: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut hpsi: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut spsi: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut spsi_prev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut sm1hpsi: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut grid_buf: CudaSlice<CudaComplex> = stream.alloc_zeros(n_bands * gs).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Cold-start inner loop: repeat filter+RR with b_low from RR eigenvalues.
    // Zhou et al. Algorithm 5.1 — itmax=3-4 iterations of filter→RR→update_b_low
    // bootstraps spectral bounds without an initial diagonalization.
    let n_inner: usize = 3;
    let mut prev_rr_eig: Option<Vec<f64>> = None;
    let mut psi_col_holder: Option<CudaSlice<CudaComplex>> = None;

    for i_iter in 0..n_inner {
        // Determine psi_input: first iter uses CASTEP input, later use RR output
        let psi_input_slice: &CudaSlice<CudaComplex> = if i_iter == 0 {
            psi_gpu.as_device_slice()
        } else {
            psi_col_holder.as_ref().unwrap()
        };

        // Compute H*psi and S*psi for current psi_input
        unsafe {
            crate::eigensolver::hamiltonian::apply_full_hamiltonian(
                psi_input_slice, v_eff_gpu.as_device_slice(), &kinetic_dev, &fft_idx_dev,
                n_pw, n_bands, gs, inv_ntotal, &fft_plan, &mut hpsi, &mut grid_buf,
                &kd.vnl, blas, kernels, stream,
            ).map_err(|e| { eprintln!("[chemrust] H*psi iter={i_iter} failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
            // S·psi: identity + augmentation
            stream.memcpy_dtod(psi_input_slice, &mut spsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            crate::eigensolver::hamiltonian::apply_s_times(
                psi_input_slice, &mut spsi, &kd.vnl, n_bands_i32, n_pw_i32, blas, stream,
            ).map_err(|e| { eprintln!("[chemrust] S*psi iter={i_iter} failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
            // ABINIT: also save S·psi₀ for the Chebyshev recurrence of gsc
            stream.memcpy_dtod(&spsi, &mut spsi_prev).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        }

        // Diagnostic: compare <psi|H_ours|psi> against CASTEP eigenvalues.
        // If H_ours == H_castep, the per-band expectation values must match
        // the eigenvalues from the previous SCF cycle for the same wavefunctions.
        // This isolates H-operator mismatch from filter/RR issues.
        if i_iter == 0 {
            let eig_castep: Vec<f64> = unsafe {
                std::slice::from_raw_parts(eigenvalues_ptr as *const f64, n_bands).to_vec()
            };
            let psi_diag: Vec<CudaComplex> = stream.clone_dtoh(psi_input_slice)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            let hpsi_diag: Vec<CudaComplex> = stream.clone_dtoh(&hpsi)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            let spsi_diag: Vec<CudaComplex> = stream.clone_dtoh(&spsi)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            let mut e_h = vec![0.0f64; n_bands];
            let mut e_hs = vec![0.0f64; n_bands];
            let mut e_kin = vec![0.0f64; n_bands];
            let mut n2 = vec![0.0f64; n_bands];
            for b in 0..n_bands {
                let mut hdot = (0.0f64, 0.0f64);
                let mut sdot = (0.0f64, 0.0f64);
                let mut nd = 0.0f64;
                let mut kd = 0.0f64;
                for g in 0..n_pw {
                    let p = &psi_diag[b * n_pw + g];
                    let h = &hpsi_diag[b * n_pw + g];
                    let s = &spsi_diag[b * n_pw + g];
                    let p2 = p.x * p.x + p.y * p.y;
                    hdot.0 += p.x * h.x + p.y * h.y;
                    hdot.1 += p.x * h.y - p.y * h.x;
                    sdot.0 += p.x * s.x + p.y * s.y;
                    nd += p2;
                    kd += p2 * ke_castep[g];
                }
                e_h[b] = if nd > 1e-30 { hdot.0 / nd } else { 0.0 };
                e_hs[b] = if sdot.0.abs() > 1e-30 { hdot.0 / sdot.0 } else { e_h[b] };
                e_kin[b] = if nd > 1e-30 { kd / nd } else { 0.0 };
                n2[b] = nd;
            }
            // Decompose: E = T + V (kinetic + local pot + nonlocal), all per-unit-norm
            let v_sum: Vec<f64> = e_h.iter().zip(e_kin.iter()).map(|(e, t)| e - t).collect();
            let mut diffs: Vec<f64> = e_h.iter().zip(eig_castep.iter())
                .map(|(e, ec)| (e - ec).abs()).collect();
            diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let d_min = diffs[0];
            let d_max = diffs[n_bands - 1];
            let d_med = diffs[n_bands / 2];
            let d_mean = diffs.iter().sum::<f64>() / n_bands as f64;
            let eig_min = eig_castep.iter().cloned().fold(f64::INFINITY, f64::min);
            let eig_max = eig_castep.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e_h_min = e_h.iter().cloned().fold(f64::INFINITY, f64::min);
            let e_h_max = e_h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            eprintln!("[Diag-HOp] ik={} |E_ours - E_castep|: min={:.3e} med={:.3e} mean={:.3e} max={:.3e}",
                ikpt, d_min, d_med, d_mean, d_max);
            let kin_min = e_kin.iter().cloned().fold(f64::INFINITY, f64::min);
            let kin_max = e_kin.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let kin_mean = e_kin.iter().sum::<f64>() / n_bands as f64;
            let v_min = v_sum.iter().cloned().fold(f64::INFINITY, f64::min);
            let v_max = v_sum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let v_mean = v_sum.iter().sum::<f64>() / n_bands as f64;
            eprintln!("[Diag-HOp] ik={} E_castep range=[{:.6}, {:.6}]  E_ours range=[{:.6}, {:.6}]",
                ikpt, eig_min, eig_max, e_h_min, e_h_max);
            eprintln!("[Diag-HOp] ik={} T(kinetic) range=[{:.6}, {:.6}] mean={:.6}  V(pot) range=[{:.6}, {:.6}] mean={:.6}",
                ikpt, kin_min, kin_max, kin_mean, v_min, v_max, v_mean);
            // Show worst 5 bands by |E_ours - E_castep|, with T/V decomposition
            let mut idxs: Vec<usize> = (0..n_bands).collect();
            idxs.sort_by(|&a, &b| diffs[b].partial_cmp(&diffs[a]).unwrap());
            for rank in 0..5usize.min(n_bands) {
                let b = idxs[rank];
                eprintln!("[Diag-HOp] ik={} band={} E_castep={:.6} E_ours={:.6} diff={:.3e} T={:.6} V={:.6} |psi|^2={:.6}",
                    ikpt, b, eig_castep[b], e_h[b], (e_h[b] - eig_castep[b]).abs(), e_kin[b], v_sum[b], n2[b]);
            }
            // Compute V_NL per band directly from beta_phi^H * D * beta_phi for first ion
            // to compare with CASTEP's Diag-Vnl-CASTEP.
            let ne0 = kd.vnl.entries[0].n_expanded as usize;
            if ne0 > 0 && n_pw > 0 {
                let mut bp: CudaSlice<CudaComplex> = stream.alloc_zeros(ne0 * n_bands)
                    .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                unsafe {
                    blas.gemm_c64(
                        ZgemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_C,
                            transb: cublasOperation_t::CUBLAS_OP_N,
                            m: ne0 as i32, n: n_bands_i32, k: n_pw_i32,
                            alpha: CudaComplex { x: 1.0, y: 0.0 },
                            lda: n_pw_i32, ldb: n_pw_i32,
                            beta: CudaComplex { x: 0.0, y: 0.0 },
                            ldc: ne0 as i32,
                        },
                        &kd.vnl.entries[0].beta_g,
                        psi_input_slice,
                        &mut bp,
                    ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                }
                let bp_host: Vec<CudaComplex> = stream.clone_dtoh(&bp)
                    .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                let d_host: Vec<CudaComplex> = stream.clone_dtoh(&kd.vnl.entries[0].d_matrix)
                    .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                eprintln!("# [Diag-Vnl-Rust-D] ik={} ion0 ne={} D from GPU:", ikpt, ne0);
                for m in 0..ne0.min(6) {
                    for n in 0..ne0.min(6) {
                        eprintln!("  D[{m},{n}]={:.15E}", d_host[m * ne0 + n].x);
                    }
                }
                eprintln!("# [Diag-Vnl-Rust] ik={} ion0 ne={} nbands={}", ikpt, ne0, n_bands);
                for b in 0..n_bands.min(10) {
                    let mut vnl = 0.0f64;
                    for m in 0..ne0 {
                        for n in 0..ne0 {
                            let d = d_host[m * ne0 + n].x;
                            let pn = &bp_host[n + b * ne0];
                            let pm = &bp_host[m + b * ne0];
                            vnl += d * (pm.x * pn.x + pm.y * pn.y + (pm.y * pn.x - pm.x * pn.y) * 0.0);
                            // conjg(pm) * pn = (pm.x - i*pm.y) * (pn.x + i*pn.y)
                            // Re = pm.x*pn.x + pm.y*pn.y
                        }
                    }
                    eprintln!("{:6}  {:.15E}", b, vnl);
                }
            }
        }

        // Power-iteration warm start for the first inner iteration:
        // Use H·psi as the input vector instead of psi. One application of H
        // builds beta-character through V_NL = beta·D·beta^H, which introduces
        // atomic projector overlap that the S-operator requires.
        let (warm_psi_ptr, warm_hpsi_ptr, warm_spsi_ptr) = if i_iter > 0 { // warm start enabled for inner iterations > 0
            // Save H·psi → psi_curr (warm_psi = H·psi₀)
            stream.memcpy_dtod(&hpsi, &mut psi_curr).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            // Compute H·(H·psi₀) → hpsi (overwrite; warm_hpsi)
            unsafe {
                crate::eigensolver::hamiltonian::apply_full_hamiltonian(
                    &psi_curr, v_eff_gpu.as_device_slice(), &kinetic_dev, &fft_idx_dev,
                    n_pw, n_bands, gs, inv_ntotal, &fft_plan, &mut hpsi, &mut grid_buf,
                    &kd.vnl, blas, kernels, stream,
                ).map_err(|e| { eprintln!("[chemrust] warm H²*psi failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
            }
            // Compute S·warm_psi → spsi (overwrite) and spsi_prev
            stream.memcpy_dtod(&psi_curr, &mut spsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            unsafe {
                crate::eigensolver::hamiltonian::apply_s_times(
                    &psi_curr, &mut spsi, &kd.vnl, n_bands_i32, n_pw_i32, blas, stream,
                ).map_err(|e| { eprintln!("[chemrust] warm S*Hpsi failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
            }
            stream.memcpy_dtod(&spsi, &mut spsi_prev).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            // Return GPU pointers for RQ computation and recurrence
            let warm_ptr = psi_curr.device_ptr(stream).0;
            let warm_hptr = hpsi.device_ptr(stream).0;
            let warm_sptr = spsi.device_ptr(stream).0;
            (warm_ptr, warm_hptr, warm_sptr)
        } else {
            // Non-first iteration: use existing psi_input and hpsi/spsi
            let p = psi_input_slice.device_ptr(stream).0;
            let h = hpsi.device_ptr(stream).0;
            let s = spsi.device_ptr(stream).0;
            (p, h, s)
        };

        // Determine per-band eigenvalues for filtering and normalization.
        // Iteration 0: use warm Rayleigh quotients from H·psi.
        // Iteration 1+: use RR eigenvalues from previous iteration (Algorithm 5.1 step 11).
        let eig: Vec<f64> = if let Some(ref prev_eig) = prev_rr_eig {
            prev_eig.clone()
        } else {
            let mut h_dot = vec![0.0f64; n_bands];
            let mut s_dot = vec![0.0f64; n_bands];
            unsafe {
                for b in 0..n_bands {
                    let mut hd = CudaComplex { x: 0.0, y: 0.0 };
                    let mut sd = CudaComplex { x: 0.0, y: 0.0 };
                    let psi_base = warm_psi_ptr as *const CudaComplex;
                    let hpsi_base = warm_hpsi_ptr as *const CudaComplex;
                    let spsi_base = warm_spsi_ptr as *const CudaComplex;
                    cudarc::cublas::sys::cublasZdotc_v2(
                        blas.raw_handle(), n_pw_i32,
                        psi_base.add(b * n_pw) as *const _, 1,
                        hpsi_base.add(b * n_pw) as *const _, 1,
                        &mut hd as *mut _ as *mut _,
                    ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                    cudarc::cublas::sys::cublasZdotc_v2(
                        blas.raw_handle(), n_pw_i32,
                        psi_base.add(b * n_pw) as *const _, 1,
                        spsi_base.add(b * n_pw) as *const _, 1,
                        &mut sd as *mut _ as *mut _,
                    ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                    h_dot[b] = hd.x;
                    s_dot[b] = sd.x.max(1e-30);
                }
            }
            h_dot.iter().zip(s_dot.iter()).map(|(h, s)| h / s).collect()
        };
        let b_low = eig.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let center = (ecut + b_low) / 2.0;
        let radius = (ecut - b_low) / 2.0;
        if radius <= 0.0 {
            eprintln!("[chemrust] step_inner bounds FAILED ik={} ecut={ecut:.4} b_low={b_low:.4} radius={radius:.4}", ik);
            return Err(CHEM_EIG_CUDA_ERROR);
        }
        // Pre-filter beta_phi diagnostic (track if filter destroys atomic character)
        if i_iter == n_inner - 1 {
            let ne0 = kd.vnl.entries[0].n_expanded as usize;
            if ne0 > 0 {
                let mut bp_pre: CudaSlice<CudaComplex> = stream.alloc_zeros(ne0 * n_bands)
                    .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                // Use psi_input_slice for iter>0, or psi_gpu for iter==0
                let psi_for_bp: &CudaSlice<CudaComplex> = if i_iter == 0 {
                    psi_input_slice
                } else {
                    psi_col_holder.as_ref().unwrap()
                };
                unsafe {
                    blas.gemm_c64(
                        ZgemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_C,
                            transb: cublasOperation_t::CUBLAS_OP_N,
                            m: ne0 as i32, n: n_bands_i32, k: n_pw_i32,
                            alpha: CudaComplex { x: 1.0, y: 0.0 },
                            lda: n_pw_i32, ldb: n_pw_i32,
                            beta: CudaComplex { x: 0.0, y: 0.0 },
                            ldc: ne0 as i32,
                        },
                        &kd.vnl.entries[0].beta_g,
                        psi_for_bp,
                        &mut bp_pre,
                    ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                }
                let bp_host: Vec<CudaComplex> = stream.clone_dtoh(&bp_pre)
                    .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                let mut max_bp = 0.0f64;
                let mut sum_bp = 0.0f64;
                for idx in 0..(ne0 * n_bands) {
                    let v = (bp_host[idx].x*bp_host[idx].x + bp_host[idx].y*bp_host[idx].y).sqrt();
                    max_bp = max_bp.max(v);
                    sum_bp += v;
                }
                let mean_bp = sum_bp / (ne0 * n_bands) as f64;
                // Also compute psi L2 norm
                let psi_host: Vec<CudaComplex> = stream.clone_dtoh(psi_for_bp)
                    .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                let mut norms2_pre = vec![0.0f64; n_bands];
                for b in 0..n_bands {
                    for g in 0..n_pw {
                        let c = &psi_host[b * n_pw + g];
                        norms2_pre[b] += c.x * c.x + c.y * c.y;
                    }
                }
                let min_n2 = norms2_pre.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_n2 = norms2_pre.iter().cloned().fold(0.0f64, f64::max);
                let mean_n2 = norms2_pre.iter().sum::<f64>() / n_bands as f64;
                eprintln!("[Diag-BetaPhi-PreFilt] ik={} ion0 pre-filter |beta_phi| max={:.3e} mean={:.3e} |psi|^2 range=[{:.6},{:.6}] mean={:.6}",
                    ikpt, max_bp, mean_bp, min_n2, max_n2, mean_n2);
            }
        }

        eprintln!("[DirectCheb] ecut={:.4} b_low={:.4}(from {}) center={:.4} radius={:.4} ndeg={} iter={}/{}",
            ecut, b_low, if prev_rr_eig.is_some() { "RR eig" } else { "max λ_i" }, center, radius, ndeg, i_iter, n_inner);

        // Precomputed x_i = (λ_i - center) / radius for normalization
        let xred: Vec<f64> = eig.iter().map(|l| (l - center) / radius).collect();

    // Diagnostic: spectral bounds + per-band Rayleigh quotients
    {
        let min_eig = eig.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_eig = eig.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_x = xred.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_x = xred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let above_b_low: Vec<_> = eig.iter().enumerate().filter(|&(_, &l)| l > b_low).collect();
        eprintln!("[Diag-Bounds] ik={} ecut={:.6} b_low={:.6} center={:.6} radius={:.6}",
            ikpt, ecut, b_low, center, radius);
        eprintln!("[Diag-Bounds] ik={} eig range=[{:.6}, {:.6}] xred range=[{:.6}, {:.6}] n_bands_above_b_low={}",
            ikpt, min_eig, max_eig, min_x, max_x, above_b_low.len());
        if !above_b_low.is_empty() {
            let show = above_b_low.iter().take(5).map(|(b, l)| format!("b{b}={l:.6}")).collect::<Vec<_>>().join(" ");
            eprintln!("[Diag-Bounds] ik={ikpt} BANDS ABOVE b_low: {show}", ikpt=ikpt, show=show);
        }
    }

    // ---- Chebyshev recurrence: psi_k = (2/r)*(S^-1*H - c)*psi_{k-1} - psi_{k-2} ----
    let inv_r = 1.0 / radius;
    let two_inv_r = 2.0 * inv_r;
    let neg_two_c_r = -2.0 * center * inv_r;
    let neg_c = CudaComplex { x: -center, y: 0.0 };
    let inv_r_c = CudaComplex { x: inv_r, y: 0.0 };
    let two_inv_r_c = CudaComplex { x: two_inv_r, y: 0.0 };
    let neg_two_c_r_c = CudaComplex { x: neg_two_c_r, y: 0.0 };
    let minus_one = CudaComplex { x: -1.0, y: 0.0 };

    unsafe {
        // k=1: psi_1 = (1/r)*(S^-1*H - c)*psi_0
        stream.memcpy_dtod(&hpsi, &mut sm1hpsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        crate::eigensolver::hamiltonian::apply_s_inverse(
            &mut sm1hpsi, &kd.vnl, n_bands_i32, n_pw_i32, blas, stream, solver,
        ).map_err(|e| { eprintln!("[chemrust] S^-1 k=1 failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

        stream.memcpy_dtod(psi_input_slice, &mut psi_prev).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        stream.memcpy_dtod(&sm1hpsi, &mut psi_curr).map_err(|_| CHEM_EIG_CUDA_ERROR)?;

        let blas_raw = blas.raw_handle();
        cudarc::cublas::sys::cublasZaxpy_v2(
            blas_raw, n_elem_i32, &neg_c as *const _ as *const _,
            psi_prev.device_ptr(stream).0 as *const _, 1,
            psi_curr.device_ptr_mut(stream).0 as *mut _, 1,
        ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        cudarc::cublas::sys::cublasZscal_v2(
            blas_raw, n_elem_i32, &inv_r_c as *const _ as *const _,
            psi_curr.device_ptr_mut(stream).0 as *mut _, 1,
        ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;

        // ABINIT: spsi recurrence for k=1
        // spsi_1 = (1/r)*(H·psi_0 - c*S·psi_0) = (1/r)*(hpsi - c*spsi_prev)
        // where hpsi = H·psi_0, spsi_prev = S·psi_0
        // Reuse sm1hpsi as temp (will be overwritten in k=2 step anyway)
        // sm1hpsi already contains S⁻¹·H·psi_0 from above, need hpsi = H·psi_0
        // hpsi still holds H·psi_0 at this point (not yet overwritten)
        stream.memcpy_dtod(&hpsi, &mut sm1hpsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?; // sm1hpsi = H·psi_0
        cudarc::cublas::sys::cublasZaxpy_v2(
            blas_raw, n_elem_i32, &neg_c as *const _ as *const _,
            spsi_prev.device_ptr(stream).0 as *const _, 1,
            sm1hpsi.device_ptr_mut(stream).0 as *mut _, 1,
        ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?; // sm1hpsi = H·psi_0 - c*S·psi_0
        cudarc::cublas::sys::cublasZscal_v2(
            blas_raw, n_elem_i32, &inv_r_c as *const _ as *const _,
            sm1hpsi.device_ptr_mut(stream).0 as *mut _, 1,
        ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?; // sm1hpsi = spsi_1
        // Update spsi buffers: spsi_prev stays as S·psi_0, spsi becomes S·psi_1
        stream.memcpy_dtod(&spsi, &mut spsi_prev).map_err(|_| CHEM_EIG_CUDA_ERROR)?; // spsi_prev = S·psi_0
        stream.memcpy_dtod(&sm1hpsi, &mut spsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?;   // spsi = S·psi_1

        // k=2..ndeg
        for k in 2..=ndeg {
            crate::eigensolver::hamiltonian::apply_full_hamiltonian(
                &psi_curr, v_eff_gpu.as_device_slice(), &kinetic_dev, &fft_idx_dev,
                n_pw, n_bands, gs, inv_ntotal, &fft_plan, &mut hpsi, &mut grid_buf,
                &kd.vnl, blas, kernels, stream,
            ).map_err(|e| { eprintln!("[chemrust] H*psi k={k} failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

            stream.memcpy_dtod(&hpsi, &mut sm1hpsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            crate::eigensolver::hamiltonian::apply_s_inverse(
                &mut sm1hpsi, &kd.vnl, n_bands_i32, n_pw_i32, blas, stream, solver,
            ).map_err(|e| { eprintln!("[chemrust] S^-1 k={k} failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

            cudarc::cublas::sys::cublasZscal_v2(
                blas_raw, n_elem_i32, &two_inv_r_c as *const _ as *const _,
                sm1hpsi.device_ptr_mut(stream).0 as *mut _, 1,
            ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            cudarc::cublas::sys::cublasZaxpy_v2(
                blas_raw, n_elem_i32, &neg_two_c_r_c as *const _ as *const _,
                psi_curr.device_ptr(stream).0 as *const _, 1,
                sm1hpsi.device_ptr_mut(stream).0 as *mut _, 1,
            ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            cudarc::cublas::sys::cublasZaxpy_v2(
                blas_raw, n_elem_i32, &minus_one as *const _ as *const _,
                psi_prev.device_ptr(stream).0 as *const _, 1,
                sm1hpsi.device_ptr_mut(stream).0 as *mut _, 1,
            ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;

            // Swap psi buffers (psi_curr now holds psi_k, sm1hpsi = stale psi_{k-1})
            std::mem::swap(&mut psi_prev, &mut psi_curr);
            std::mem::swap(&mut psi_curr, &mut sm1hpsi);

            // ABINIT: spsi recurrence for k≥2
            // spsi_k = (2/r)*(H·psi_{k-1} - c*S·psi_{k-1}) - S·psi_{k-2}
            //        = (2/r)*hpsi + (-2c/r)*spsi + (-1)*spsi_prev
            // Reuse sm1hpsi as temp (it now holds stale psi_{k-1}, will be
            // overwritten at start of next iteration by memcpy from hpsi).
            stream.memcpy_dtod(&hpsi, &mut sm1hpsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            cudarc::cublas::sys::cublasZscal_v2(
                blas_raw, n_elem_i32, &two_inv_r_c as *const _ as *const _,
                sm1hpsi.device_ptr_mut(stream).0 as *mut _, 1,
            ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            cudarc::cublas::sys::cublasZaxpy_v2(
                blas_raw, n_elem_i32, &neg_two_c_r_c as *const _ as *const _,
                spsi.device_ptr(stream).0 as *const _, 1,
                sm1hpsi.device_ptr_mut(stream).0 as *mut _, 1,
            ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            cudarc::cublas::sys::cublasZaxpy_v2(
                blas_raw, n_elem_i32, &minus_one as *const _ as *const _,
                spsi_prev.device_ptr(stream).0 as *const _, 1,
                sm1hpsi.device_ptr_mut(stream).0 as *mut _, 1,
            ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;

            // Swap spsi buffers: spsi_prev ← spsi, spsi ← sm1hpsi (now holds spsi_k)
            stream.memcpy_dtod(&spsi, &mut spsi_prev).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            stream.memcpy_dtod(&sm1hpsi, &mut spsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        }
    }

    // ---- Normalize: psi_i /= T_ndeg(x_i) ----
    // ABINIT (m_chebfi.F90:447-453): after the Chebyshev recurrence, divide
    // each band by T_n((λ_i - c)/r) to undo the differential amplification.
    // T_n(x) computed on CPU, normalization applied per-band.
    let mut norm_factors = vec![1.0f64; n_bands];
    for b in 0..n_bands {
        let x = xred[b];
        // Chebyshev polynomial T_n(x) via recurrence
        let mut t_prev = 1.0; // T_0 = 1
        let mut t_curr = x;   // T_1 = x
        for _ in 2..=ndeg {
            let t_next = 2.0 * x * t_curr - t_prev;
            t_prev = t_curr;
            t_curr = t_next;
        }
        let ampfactor = t_curr;
        norm_factors[b] = if ampfactor.abs() < 1e-3 { 1e3 } else { 1.0 / ampfactor };
    }
    // Diagnostic: per-band T_n amplification summary
    {
        let mut t_vals: Vec<f64> = Vec::with_capacity(n_bands);
        for b in 0..n_bands {
            let x = xred[b];
            let mut tp = 1.0;
            let mut tc = x;
            for _ in 2..=ndeg {
                let tn = 2.0 * x * tc - tp;
                tp = tc; tc = tn;
            }
            t_vals.push(tc);
        }
        let t_abs: Vec<f64> = t_vals.iter().map(|t| t.abs()).collect();
        let t_min = t_abs.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = t_abs.iter().cloned().fold(0.0f64, f64::max);
        let n_capped = t_abs.iter().filter(|&&t| t < 1e-3).count();
        let n_neg = t_vals.iter().filter(|&&t| t < 0.0).count();
        eprintln!("[Diag-Tn] ik={} ndeg={} |T_n| range=[{:.3e}, {:.3e}] n_capped={} n_neg_sign={}",
            ikpt, ndeg, t_min, t_max, n_capped, n_neg);
        if n_capped > 0 {
            for b in 0..n_bands {
                if t_abs[b] < 1e-3 {
                    eprintln!("[Diag-Tn] ik={} band={} λ={:.6} x={:.6} T_n={:.3e} cap_activated",
                        ikpt, b, eig[b], xred[b], t_vals[b]);
                }
            }
        }
    }
    // Apply per-band scaling on GPU — ABINIT normalizes psi, H·psi, S·psi all by T_n
    unsafe {
        let blas_raw = blas.raw_handle();
        let psi_ptr = psi_curr.device_ptr_mut(stream).0 as *mut CudaComplex;
        let spsi_ptr = spsi.device_ptr_mut(stream).0 as *mut CudaComplex;
        for b in 0..n_bands {
            let scale = CudaComplex { x: norm_factors[b], y: 0.0 };
            if scale.x != 1.0 {
                cudarc::cublas::sys::cublasZscal_v2(
                    blas_raw, n_pw_i32, &scale as *const _ as *const _,
                    psi_ptr.add(b * n_pw) as *mut _, 1,
                ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                cudarc::cublas::sys::cublasZscal_v2(
                    blas_raw, n_pw_i32, &scale as *const _ as *const _,
                    spsi_ptr.add(b * n_pw) as *mut _, 1,
                ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            }
        }
    }

    // S-normalize each band so <psi_i|S|psi_i> = 1 before entering RR.
    // The Chebyshev filter amplifies different eigencomponents by different
    // amounts; single-number T_n normalization cannot perfectly restore the
    // original S-norm for bands that are superpositions of multiple eigenstates.
    // Explicit S-normalization guarantees all bands have equal weight in the RR
    // subspace and prevents high-norm bands from dominating the overlap matrix.
    {
        let mut s_norms = vec![0.0f64; n_bands];
        unsafe {
            let blas_raw = blas.raw_handle();
            let psi_ptr = psi_curr.device_ptr(stream).0 as *const CudaComplex;
            let spsi_ptr = spsi.device_ptr(stream).0 as *const CudaComplex;
            for b in 0..n_bands {
                let mut sdot = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas_raw, n_pw_i32,
                    psi_ptr.add(b * n_pw) as *const _, 1,
                    spsi_ptr.add(b * n_pw) as *const _, 1,
                    &mut sdot as *mut _ as *mut _,
                ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                s_norms[b] = sdot.x.max(1e-30);
            }
        }
        let min_s = s_norms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_s = s_norms.iter().cloned().fold(0.0f64, f64::max);
        let mean_s = s_norms.iter().sum::<f64>() / n_bands as f64;
        eprintln!("[Diag-SNorm] ik={} <psi|S|psi> before S-norm: min={:.6e} max={:.6e} mean={:.6e}",
            ikpt, min_s, max_s, mean_s);

        unsafe {
            let blas_raw = blas.raw_handle();
            let psi_ptr = psi_curr.device_ptr_mut(stream).0 as *mut CudaComplex;
            let spsi_ptr = spsi.device_ptr_mut(stream).0 as *mut CudaComplex;
            for b in 0..n_bands {
                let inv_sqrt = 1.0 / s_norms[b].sqrt();
                let scale = CudaComplex { x: inv_sqrt, y: 0.0 };
                cudarc::cublas::sys::cublasZscal_v2(
                    blas_raw, n_pw_i32, &scale as *const _ as *const _,
                    psi_ptr.add(b * n_pw) as *mut _, 1,
                ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                cudarc::cublas::sys::cublasZscal_v2(
                    blas_raw, n_pw_i32, &scale as *const _ as *const _,
                    spsi_ptr.add(b * n_pw) as *mut _, 1,
                ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            }
        }

        // Diagnostic: compute identity norm <psi|psi> AFTER S-normalization.
        // For USPP, <psi|S|psi> = <psi|psi> + <psi|aug|psi>. Since we normalize
        // <psi|S|psi> = 1, the identity norm <psi|psi> is < 1. If CASTEP's
        // density builder expects <psi|psi> = 1 (identity normalization), the
        // density will be too small — but we observe the opposite (2.7x too large).
        // Tracking this helps isolate whether the density discrepancy comes from
        // wavefunction normalization or augmentation density weights.
        let mut id_norms = vec![0.0f64; n_bands];
        unsafe {
            let blas_raw = blas.raw_handle();
            let psi_ptr = psi_curr.device_ptr(stream).0 as *const CudaComplex;
            for b in 0..n_bands {
                let mut idot = CudaComplex { x: 0.0, y: 0.0 };
                cudarc::cublas::sys::cublasZdotc_v2(
                    blas_raw, n_pw_i32,
                    psi_ptr.add(b * n_pw) as *const _, 1,
                    psi_ptr.add(b * n_pw) as *const _, 1,
                    &mut idot as *mut _ as *mut _,
                ).result().map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                id_norms[b] = idot.x.max(1e-30);
            }
        }
        let id_min = id_norms.iter().cloned().fold(f64::INFINITY, f64::min);
        let id_max = id_norms.iter().cloned().fold(0.0f64, f64::max);
        let id_mean = id_norms.iter().sum::<f64>() / n_bands as f64;
        let aug_fraction = 1.0 - id_mean; // fraction of norm in augmentation channel
        eprintln!("[Diag-SNorm] ik={} <psi|psi> after S-norm: min={:.6e} max={:.6e} mean={:.6e} aug_frac={:.4}",
            ikpt, id_min, id_max, id_mean, aug_fraction);
    }

    // Diagnostic: track beta_phi from post-filter psi_curr (before RR)
    if i_iter == n_inner - 1 {
        let ne0 = kd.vnl.entries[0].n_expanded as usize;
        if ne0 > 0 {
            let mut bp_filt: CudaSlice<CudaComplex> = stream.alloc_zeros(ne0 * n_bands)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            unsafe {
                blas.gemm_c64(
                    ZgemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_C,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: ne0 as i32, n: n_bands_i32, k: n_pw_i32,
                        alpha: CudaComplex { x: 1.0, y: 0.0 },
                        lda: n_pw_i32, ldb: n_pw_i32,
                        beta: CudaComplex { x: 0.0, y: 0.0 },
                        ldc: ne0 as i32,
                    },
                    &kd.vnl.entries[0].beta_g,
                    &psi_curr,
                    &mut bp_filt,
                ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            }
            let bp_host: Vec<CudaComplex> = stream.clone_dtoh(&bp_filt)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            let mut max_bp = 0.0f64;
            let mut sum_bp = 0.0f64;
            for idx in 0..(ne0 * n_bands) {
                let v = (bp_host[idx].x*bp_host[idx].x + bp_host[idx].y*bp_host[idx].y).sqrt();
                max_bp = max_bp.max(v);
                sum_bp += v;
            }
            let mean_bp = sum_bp / (ne0 * n_bands) as f64;
            eprintln!("[Diag-BetaPhi-PostFilt] ik={} ion0 post-filter |beta_phi| max={:.3e} mean={:.3e}",
                ikpt, max_bp, mean_bp);
            // Also print L2 norm of post-filter psi_curr
            let psi_filt: Vec<CudaComplex> = stream.clone_dtoh(&psi_curr)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            let mut norms2: Vec<f64> = Vec::with_capacity(n_bands);
            for b in 0..n_bands {
                let mut n2 = 0.0f64;
                for g in 0..n_pw {
                    let c = &psi_filt[b * n_pw + g];
                    n2 += c.x * c.x + c.y * c.y;
                }
                norms2.push(n2);
            }
            let min_n2 = norms2.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_n2 = norms2.iter().cloned().fold(0.0f64, f64::max);
            let mean_n2 = norms2.iter().sum::<f64>() / n_bands as f64;
            eprintln!("[Diag-Norm-PostFilt] ik={} post-filter |psi|^2 range=[{:.6}, {:.6}] mean={:.6}",
                ikpt, min_n2, max_n2, mean_n2);
        }
    }

    // ---- Final H*psi for Rayleigh-Ritz (after normalization, no GS) ----
    // ABINIT goes filter → normalize(T_n) → H*psi → RR.
    // The RR solves H_sub·X = λ·S_sub·X which implicitly S-orthonormalizes
    // the subspace via the eigenvectors X.
    unsafe {
        crate::eigensolver::hamiltonian::apply_full_hamiltonian(
            &psi_curr, v_eff_gpu.as_device_slice(), &kinetic_dev, &fft_idx_dev,
            n_pw, n_bands, gs, inv_ntotal, &fft_plan, &mut hpsi, &mut grid_buf,
            &kd.vnl, blas, kernels, stream,
        ).map_err(|e| { eprintln!("[chemrust] final H*psi failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    }

    // ---- Rayleigh-Ritz and download ----
    let psi_host: Vec<num_complex::Complex64> = {
        let v = stream.clone_dtoh(&psi_curr).map_err(|_| { eprintln!("[chemrust] clone_dtoh psi_curr failed ik={}", ik); CHEM_EIG_CUDA_ERROR })?;
        v.into_iter().map(|c| num_complex::Complex64::new(c.x, c.y)).collect()
    };
    let hpsi_host_rr: Vec<num_complex::Complex64> = {
        let v = stream.clone_dtoh(&hpsi).map_err(|_| { eprintln!("[chemrust] clone_dtoh hpsi failed ik={}", ik); CHEM_EIG_CUDA_ERROR })?;
        v.into_iter().map(|c| num_complex::Complex64::new(c.x, c.y)).collect()
    };
    let psi_wfn = WavefunctionSet::<crate::layout::RowDistributed>::new(psi_host, n_bands, n_pw);
    let hpsi_wfn = WavefunctionSet::<crate::layout::RowDistributed>::new(hpsi_host_rr, n_bands, n_pw);
    let psi_gpu_rr = crate::device::Gpu::from_host(&psi_wfn, stream).map_err(|_| { eprintln!("[chemrust] from_host psi failed ik={}", ik); CHEM_EIG_CUDA_ERROR })?;
    let hpsi_gpu_rr = crate::device::Gpu::from_host(&hpsi_wfn, stream).map_err(|_| { eprintln!("[chemrust] from_host hpsi failed ik={}", ik); CHEM_EIG_CUDA_ERROR })?;

    let (psi_col, eig_cpu, _, _, _, x_cpu) = rayleigh_ritz_with_matrices(
        &psi_gpu_rr, &hpsi_gpu_rr, &kd.vnl,
        n_bands, n_pw, kernels, &mut pcie_step, solver, blas, stream, ctx,
        None, None,
    ).map_err(|e| { eprintln!("[chemrust] RR failed: {e}"); CHEM_EIG_CUDA_ERROR })?;

    // Convergence check (Zhou Algorithm 5.1 step 12):
    // if max|eps_i^(k) - eps_i^(k-1)| < tol, break early.
    let conv_tol: f64 = 1e-5; // Hartree
    let converged_this_iter = if let Some(ref prev) = prev_rr_eig {
        let max_delta = eig_cpu.0.iter().zip(prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("[Diag-Conv] ik={} iter={} max|Delta_eps|={:.3e} tol={:.3e}",
            ikpt, i_iter, max_delta, conv_tol);
        max_delta < conv_tol
    } else {
        false
    };

    // Save RR eigenvalues for next inner iteration (Algorithm 5.1 step 11)
    prev_rr_eig = Some(eig_cpu.0.clone());

    // Copy psi_col to psi_col_holder for the next inner iteration
    // (psi_col lives on GPU; we need a persistent copy)
    if i_iter < n_inner - 1 && !converged_this_iter {
        if psi_col_holder.is_none() {
            psi_col_holder = Some(stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?);
        }
        stream.memcpy_dtod(psi_col.as_device_slice(), psi_col_holder.as_mut().unwrap())
            .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    }

    // Run diagnostics and write-back on final iteration OR early convergence
    let is_final = i_iter == n_inner - 1 || converged_this_iter;
    if is_final {
    // Diagnostic: verify RR output S-orthonormality
    // Compute S·psi_new on GPU and check psi_new^H·(S·psi_new) ≈ I
    {
        // Re-use spsi buffer for S·psi_new
        stream.memcpy_dtod(psi_col.as_device_slice(), &mut spsi).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        unsafe {
            crate::eigensolver::hamiltonian::apply_s_times(
                psi_col.as_device_slice(), &mut spsi, &kd.vnl,
                n_bands_i32, n_pw_i32, blas, stream,
            ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        }
        // Download spsi and psi_col for CPU check
        let spsi_host: Vec<CudaComplex> = stream.clone_dtoh(&spsi)
            .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        let psi_out_host: Vec<CudaComplex> = stream.clone_dtoh(psi_col.as_device_slice())
            .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        let mut max_offdiag = 0.0f64;
        let mut max_diag_err = 0.0f64;
        for i in 0..n_bands {
            for j in 0..n_bands {
                let mut dot = num_complex::Complex64::new(0.0, 0.0);
                for g in 0..n_pw {
                    let a = num_complex::Complex64::new(psi_out_host[i * n_pw + g].x, psi_out_host[i * n_pw + g].y);
                    let b = num_complex::Complex64::new(spsi_host[j * n_pw + g].x, spsi_host[j * n_pw + g].y);
                    dot += a.conj() * b;
                }
                if i == j {
                    let err = (dot.re - 1.0).abs();
                    if err > max_diag_err { max_diag_err = err; }
                } else {
                    let v = dot.norm();
                    if v > max_offdiag { max_offdiag = v; }
                }
            }
        }
        eprintln!("[Diag-Sortho] ik={} max|psi^H·(S·psi) - I|_diag={:.3e} max_offdiag={:.3e}",
            ikpt, max_diag_err, max_offdiag);

        // Also check: did apply_s_times actually modify spsi?
        // If S ≈ I, then |spsi - psi| ≈ 0.
        let mut max_spi_diff = 0.0f64;
        let mut max_spi_abs = 0.0f64;
        for idx in 0..(n_bands * n_pw) {
            let a = num_complex::Complex64::new(psi_out_host[idx].x, psi_out_host[idx].y);
            let b = num_complex::Complex64::new(spsi_host[idx].x, spsi_host[idx].y);
            let d = (a - b).norm();
            if d > max_spi_diff { max_spi_diff = d; }
            let m = a.norm();
            if m > max_spi_abs { max_spi_abs = m; }
        }
        eprintln!("[Diag-Sortho] ik={} max|S·psi - psi|={:.3e} max|psi|={:.3e}",
            ikpt, max_spi_diff, max_spi_abs);

        // Diagnostic: verify S⁻¹·(S·psi) ≈ psi (Woodbury self-consistency)
        {
            // spsi currently holds S·psi. Apply S⁻¹ to it, should get back psi.
            unsafe {
                crate::eigensolver::hamiltonian::apply_s_inverse(
                    &mut spsi, &kd.vnl, n_bands_i32, n_pw_i32, blas, stream, solver,
                ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            }
            let sinv_s_host: Vec<CudaComplex> = stream.clone_dtoh(&spsi)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;

            let mut max_sinv_err = 0.0f64;
            let mut rms_sinv_err = 0.0f64;
            for idx in 0..(n_bands * n_pw) {
                let orig = num_complex::Complex64::new(psi_out_host[idx].x, psi_out_host[idx].y);
                let sinv = num_complex::Complex64::new(sinv_s_host[idx].x, sinv_s_host[idx].y);
                let d = (orig - sinv).norm();
                if d > max_sinv_err { max_sinv_err = d; }
                rms_sinv_err += d * d;
            }
            rms_sinv_err = (rms_sinv_err / (n_bands * n_pw) as f64).sqrt();
            eprintln!("[Diag-SinvS] ik={} max|S⁻¹·S·psi - psi|={:.3e} rms={:.3e}",
                ikpt, max_sinv_err, rms_sinv_err);
        }

        // Dump Q matrix from GPU for the first ion
        if !kd.vnl.entries.is_empty() {
            let ne0 = kd.vnl.entries[0].n_expanded as usize;
            if ne0 > 0 {
                let q_host: Vec<CudaComplex> = stream.clone_dtoh(&kd.vnl.entries[0].q_matrix)
                    .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
                eprintln!("[Diag-Qmat] ion0 ne={} q[0..min(5,ne)]: {}",
                    ne0,
                    (0..ne0.min(5)).map(|i| format!("{:.6e}", q_host[i * ne0 + i].x))
                        .collect::<Vec<_>>().join(" "));
                // Also print first row for cross-check
                if ne0 > 1 {
                    eprintln!("[Diag-Qmat] ion0 q[row=0,col=0..min({},4)]: {}",
                        ne0,
                        (0..ne0.min(4)).map(|j| format!("{:.6e}", q_host[j].x))
                            .collect::<Vec<_>>().join(" "));
                }
            }
        }
    }

    // Diagnostic: dump a few raw beta_g values for comparison with CASTEP
    {
        let ne_first = kd.vnl.entries[0].n_expanded as usize;
        if ne_first > 0 {
            let nbeta = n_pw * ne_first;
            let bg_host: Vec<CudaComplex> = stream.clone_dtoh(&kd.vnl.entries[0].beta_g)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            // beta_g is (npw × ne) column-major: index[proj * npw + pw]
            eprintln!("[Diag-BetaG] ion0 n_pw={} ne={} beta_g[ipw=0,proj=0]=({:.6e},{:.6e}) beta_g[ipw=1,proj=0]=({:.6e},{:.6e})",
                n_pw, ne_first,
                bg_host[0].x, bg_host[0].y,
                bg_host[1].x, bg_host[1].y);
            if ne_first >= 2 && n_pw > 1 {
                eprintln!("[Diag-BetaG] ion0 beta_g[ipw=0,proj=1]=({:.6e},{:.6e}) beta_g[ipw=1,proj=1]=({:.6e},{:.6e})",
                    bg_host[n_pw].x, bg_host[n_pw].y,
                    bg_host[n_pw + 1].x, bg_host[n_pw + 1].y);
            }
        }
    }

    // Diagnostic: compute beta_phi = beta_g^H * psi_col using our beta_g
    // This should match what CASTEP's ion_all_beta_multi_phi_recip computes.
    {
        let ne_first = kd.vnl.entries[0].n_expanded as usize;
        let mut bp_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(ne_first * n_bands)
            .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        unsafe {
            blas.gemm_c64(
                ZgemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_C,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: ne_first as i32,
                    n: n_bands_i32,
                    k: n_pw_i32,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw_i32,
                    ldb: n_pw_i32,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne_first as i32,
                },
                &kd.vnl.entries[0].beta_g,
                psi_col.as_device_slice(),
                &mut bp_dev,
            ).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        }
        let bp_host: Vec<CudaComplex> = stream.clone_dtoh(&bp_dev)
            .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        // Print first few (proj, band) values and summary stats
        let n_proj_show = ne_first.min(5);
        let n_band_show = n_bands.min(3);
        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        for ip in 0..ne_first {
            for ib in 0..n_bands {
                let v = (bp_host[ib * ne_first + ip].x.powi(2)
                       + bp_host[ib * ne_first + ip].y.powi(2)).sqrt();
                max_abs = max_abs.max(v);
                sum_abs += v;
            }
        }
        eprintln!("[Diag-BetaPhi] ik={} ion0 n_proj={} n_bands={} max|beta_phi|={:.3e} mean|beta_phi|={:.3e}",
            ikpt, ne_first, n_bands, max_abs, sum_abs / (ne_first * n_bands) as f64);
        for ib in 0..n_band_show {
            let vals: Vec<String> = (0..n_proj_show).map(|ip| {
                let c = &bp_host[ib * ne_first + ip];
                format!("({:.3e},{:.3e})", c.x, c.y)
            }).collect();
            eprintln!("[Diag-BetaPhi] ik={} ion0 band={}: {}", ikpt, ib, vals.join(" "));
        }
    }

    // Diagnostic: per-band G-space L2 norms of RR output
    {
        let psi_out_host: Vec<CudaComplex> = stream.clone_dtoh(psi_col.as_device_slice())
            .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        let mut norms2: Vec<f64> = Vec::with_capacity(n_bands);
        for b in 0..n_bands {
            let mut n2 = 0.0f64;
            for g in 0..n_pw {
                let c = &psi_out_host[b * n_pw + g];
                n2 += c.x * c.x + c.y * c.y;
            }
            norms2.push(n2);
        }
        let min_n2 = norms2.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_n2 = norms2.iter().cloned().fold(0.0f64, f64::max);
        let mean_n2 = norms2.iter().sum::<f64>() / n_bands as f64;
        eprintln!("[Diag-Norm] ik={} |psi_b(G)|^2 range=[{:.6}, {:.6}] mean={:.6} (expected ~1.0 per band)",
            ikpt, min_n2, max_n2, mean_n2);
        let total_n2: f64 = norms2.iter().sum();
        // Expected soft density sum: total_n2 × inv_ntotal × 1/inv_omega = total_n2 × 1.0
        // (inv_omega = 1.0, inv_ntotal = 1/gs). This is the smooth PW density
        // component before augmentation. Compare against CASTEP F8_RHO_SOFT_SUM.
        eprintln!("[Diag-RhoSum] ik={} total |psi|^2 = {:.6e} n_bands={} (rho_soft_sum ~ {:.6e} x occupations)",
            ikpt, total_n2, n_bands, total_n2);
        // Also print occupied band norms (first few)
        let n_show = n_bands.min(10);
        let detail: Vec<String> = norms2.iter().take(n_show).enumerate()
            .map(|(b, n)| format!("b{b}={n:.4}")).collect();
        eprintln!("[Diag-Norm] ik={} first {} bands: {}", ikpt, n_show, detail.join(" "));
    }

    // Diagnostic: compute S_sub explicitly and print augmentation contribution
    {
        let psi_out_host: Vec<num_complex::Complex64> = {
            let v = stream.clone_dtoh(psi_col.as_device_slice())
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            v.into_iter().map(|c| num_complex::Complex64::new(c.x, c.y)).collect()
        };
        let mut s_diag_min = f64::INFINITY;
        let mut s_diag_max = f64::NEG_INFINITY;
        let mut s_off_max = 0.0f64;
        for i in 0..n_bands {
            for j in i..n_bands {
                let mut dot = num_complex::Complex64::new(0.0, 0.0);
                for g in 0..n_pw {
                    let a = psi_out_host[i * n_pw + g];
                    let b = psi_out_host[j * n_pw + g];
                    dot += a.conj() * b;
                }
                if i == j {
                    s_diag_min = s_diag_min.min(dot.re);
                    s_diag_max = s_diag_max.max(dot.re);
                } else {
                    s_off_max = s_off_max.max(dot.norm());
                }
            }
        }
        eprintln!("[Diag-Ssub] ik={} psi^H·psi diag range=[{:.6e}, {:.6e}] max_offdiag={:.3e}",
            ikpt, s_diag_min, s_diag_max, s_off_max);

        // Also compute beta_phi^H · Q · beta_phi per ion explicitly
        let mut aug_total = 0.0f64;
        for (ion_idx, entry) in kd.vnl.entries.iter().enumerate() {
            let ne = entry.n_expanded as usize;
            if ne == 0 { continue; }
            let q_host: Vec<CudaComplex> = stream.clone_dtoh(&entry.q_matrix)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            // Compute beta_phi for this ion on CPU from psi_out_host
            let bg_host: Vec<CudaComplex> = stream.clone_dtoh(&entry.beta_g)
                .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
            // beta_phi[n_exp, band] = sum_g conj(beta_g[n_exp, g]) * psi[band, g]
            let mut bp: Vec<num_complex::Complex64> = vec![num_complex::Complex64::new(0.0, 0.0); ne * n_bands];
            for n in 0..ne {
                for b in 0..n_bands {
                    let mut sum = num_complex::Complex64::new(0.0, 0.0);
                    for g in 0..n_pw {
                        // beta_g stored col-major: beta_g[g + n*n_pw] (index grows g fastest)
                        let bg = num_complex::Complex64::new(
                            bg_host[n * n_pw + g].x, bg_host[n * n_pw + g].y);
                        let psi_g = psi_out_host[b * n_pw + g];
                        sum += bg.conj() * psi_g;
                    }
                    bp[n * n_bands + b] = sum;
                }
            }
            // S_aug[b1,b2] = sum_{n,m} q[n,m] * conj(bp[n,b1]) * bp[m,b2]
            let mut aug_diag_sum = 0.0f64;
            for b in 0..n_bands {
                for n in 0..ne {
                    for m in 0..ne {
                        let q = q_host[n * ne + m].x;
                        if q.abs() < 1e-30 { continue; }
                        aug_diag_sum += q * (bp[n * n_bands + b].conj() * bp[m * n_bands + b]).re;
                    }
                }
            }
            // Per-band average
            let aug_per_band = aug_diag_sum / n_bands as f64;
            eprintln!("[Diag-Ssub-Aug] ik={} ion={} ne={} sum_aug_diag={:.6e} avg_per_band={:.6e}",
                ikpt, ion_idx, ne, aug_diag_sum, aug_per_band);
            aug_total += aug_diag_sum;
        }
        let aug_total_per_band = aug_total / n_bands as f64;
        eprintln!("[Diag-Ssub-Aug] ik={} total_aug_diag={:.6e} avg_per_band={:.6e} (all ions)",
            ikpt, aug_total, aug_total_per_band);
    }
    } // end of diagnostics (if i_iter == n_inner - 1)

    // Rotate hpsi and write back (inside loop for variable access)
    if i_iter == n_inner - 1 {
    // Rotate hpsi: hpsi_rotated = hpsi · X
    let x_host: Vec<CudaComplex> = x_cpu.0.iter().map(|c| CudaComplex { x: c.x, y: c.y }).collect();
    let mut x_dev: CudaSlice<CudaComplex> = stream.alloc_zeros(n_bands * n_bands)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    stream.memcpy_htod(&x_host, &mut x_dev).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let mut hpsi_rotated: CudaSlice<CudaComplex> = stream.alloc_zeros(n_elem)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    unsafe {
        blas.gemm_c64(
            ZgemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_pw_i32, n: n_bands_i32, k: n_bands_i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw_i32, ldb: n_bands_i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
                ldc: n_pw_i32,
            },
            &hpsi,
            &x_dev,
            &mut hpsi_rotated,
        )
    }.map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    // Download and write back
    let psi_host_out: Vec<CudaComplex> = stream.clone_dtoh(psi_col.as_device_slice())
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let hpsi_host: Vec<CudaComplex> = stream.clone_dtoh(&hpsi_rotated)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    unsafe {
        std::slice::from_raw_parts_mut(psi_data as *mut CudaComplex, n_pw * n_bands).copy_from_slice(&psi_host_out);
        std::slice::from_raw_parts_mut(hpsi_out as *mut CudaComplex, n_pw * n_bands).copy_from_slice(&hpsi_host);
        std::slice::from_raw_parts_mut(eigenvalues_ptr as *mut f64, n_bands).copy_from_slice(eig_cpu.0.as_slice());
    }

    // Set converged flag: true if inner loop converged within tolerance
    // or if n_inner=1 (single filter, no delta to check).
    let actual_converged = if n_inner > 1 { converged_this_iter } else { true };
    unsafe { *converged = actual_converged as c_int };
    } // end of if is_final (writeback)

    if converged_this_iter {
        eprintln!("[Diag-Conv] ik={} breaking inner loop after iter={}", ikpt, i_iter);
        break;
    }
    } // end of inner loop (for i_iter in 0..n_inner)
    Ok(())
}
