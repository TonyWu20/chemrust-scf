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
use crate::eigensolver::vnl_data::VnlBatchData;
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
    vnl: VnlBatchData,
    wave_grid: GVectorGrid,
    pw_coords: Vec<[i32; 3]>,
    kpoint_frac: [f64; 3],
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

        kpts.push(KptData { vnl, wave_grid: wg, pw_coords, kpoint_frac: kf, pcie });
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
    _max_deg: c_int,
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
    let gs = (h.ngx * h.ngy * h.ngz) as usize;
    let n_bands_i32 = n_bands as i32;
    let inv_ntotal = 1.0 / gs as f64;
    let kd = &mut h.kpts[ik];

    // Use CASTEP-provided kinetic energies (pw_ek_data = 0.5*|G+k|^2)
    let ke_castep: Vec<f64> = unsafe { std::slice::from_raw_parts(kinetic_data, n_pw) }.to_vec();

    #[cfg(feature = "scf_diag")]
    // Diagnostic: verify KE consistency
    {
        let n_print = kd.pw_coords.len().min(5);
        let ke_rust = crate::eigensolver::davidson_types::compute_kinetic_energies(&kd.pw_coords, kd.wave_grid.recip_lattice(), kd.kpoint_frac);
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
    let ve_host: Vec<f64> = unsafe { std::slice::from_raw_parts(v_eff_data, gs) }.to_vec();
    let ve_raw = unsafe { std::slice::from_raw_parts(v_eff_data, gs) };
    let ve_norm: f64 = ve_host.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    #[cfg(feature = "scf_diag")]
    {
        let ve_min = ve_host.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let ve_max = ve_host.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let ve_mean = ve_host.iter().sum::<f64>() / gs as f64;
        eprintln!("[Diag-Veff] ngx={} ngy={} ngz={} gs={} ve_min={:.6e} ve_max={:.6e} ve_mean={:.6e}",
            h.ngx, h.ngy, h.ngz, gs, ve_min, ve_max, ve_mean);
        eprintln!("[Diag-Veff] first 5 raw: {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}",
            ve_host[0], ve_host[1], ve_host[2], ve_host[3], ve_host[4]);
    }

    // V_eff GPU caching: skip H2D transfer if V_eff unchanged since last step.
    // Uses max-norm for cheap change detection (threshold 1e-8 Ha).
    let cache_reuse = h.v_eff_cached.as_ref().is_some_and(|_| (ve_norm - h.v_eff_norm).abs() < 1e-8);

    let arr = crate::device::unflatten_f64(ve_host, &[h.ngx as usize, h.ngy as usize, h.ngz as usize]);
    let veff = EffectivePotential(FineGridArray(arr));

    let v_eff_gpu = if cache_reuse {
        eprintln!("[chemrust] V_eff cache HIT norm={:.6e}", ve_norm);
        // TODO(phase-5 deferred): Pre-allocate a persistent scratch buffer in
        // ChemrustHandle to avoid per-HIT alloc_zeros + memcpy_dtod below.
        // Currently each HIT allocates a fresh GPU buffer and copies from cache,
        // which is wasteful for every SCF iteration after the first.
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
        #[cfg(feature = "scf_diag")]
        {
            let ve_rt: Vec<f64> = crate::device::flatten_f64(veff.0.as_array());
            let mut rt_err = 0.0f64;
            for i in 0..gs.min(10) {
                let d = (ve_rt[i] - ve_raw[i]).abs();
                rt_err = rt_err.max(d);
            }
            eprintln!("[Diag-Veff] round-trip max error (first 10): {:.3e}", rt_err);
        }
        let vg = Gpu::from_host(&veff, &h.stream).map_err(|_| CHEM_EIG_CUDA_ERROR)?;
        // Verify GPU upload: read back from GPU and compare
        #[cfg(feature = "scf_diag")]
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

    // Upload FFT index (CASTEP passes 1-based Fortran indices; convert to 0-based)
    let fft_idx: Vec<i32> = unsafe { std::slice::from_raw_parts(fft_idx_data as *const c_int, n_pw) }
        .iter()
        .map(|&i| i - 1)
        .collect();
    debug_assert!(
        fft_idx.iter().all(|&i| i >= 0 && (i as usize) < gs),
        "fft_idx out of range after 1→0 conversion: min={} max={} gs={gs}",
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
        h.ngx, h.ngy, h.ngz, n_bands_i32, stream.clone(),
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
    let davidson_result = unsafe {
        davidson_diagonalise()
            .psi_init(&psi_init)
            .v_eff_dev(v_eff_gpu.as_device_slice())
            .kinetic_dev(&kinetic_dev)
            .fft_idx_dev(&fft_idx_dev)
            .vnl_data(&kd.vnl)
            .n_pw(n_pw)
            .n_bands(n_bands)
            .grid_size(gs)
            .inv_ntotal(inv_ntotal)
            .fft_plan(&fft_plan)
            .tol_abs(1e-8)
            .max_outer_iter(10)
            .min_outer_iter(0)
            .blas(blas)
            .solver(solver)
            .kernels(kernels)
            .tpa_preconditioner(&tpa)
            .stream(stream)
            .ctx(ctx)
            .call()
            .map_err(|e| { eprintln!("[chemrust] davidson_diagonalise failed: {e}"); CHEM_EIG_CUDA_ERROR })?
    };

    // Compute H·psi_new for hpsi_out
    let psi_out_pw = PwCoefficients::new(davidson_result.psi_out.clone());
    let mut hpsi_new = PwCoefficients::new(
        stream.alloc_zeros(n_elem).map_err(|_| CHEM_EIG_CUDA_ERROR)?);
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
            .hpsi_dev(&mut hpsi_new)
            .grid_dev(&mut grid_buf)
            .vnl_data(&kd.vnl)
            .blas(blas)
            .kernels(kernels)
            .stream(stream)
            .call()
            .map_err(|e| { eprintln!("[chemrust] H*psi after Davidson failed: {e}"); CHEM_EIG_CUDA_ERROR })?;
    }

    // Download and write back to CASTEP buffers
    let psi_host_out: Vec<CudaComplex> = stream.clone_dtoh(&davidson_result.psi_out)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;
    let hpsi_host: Vec<CudaComplex> = stream.clone_dtoh(&*hpsi_new)
        .map_err(|_| CHEM_EIG_CUDA_ERROR)?;

    unsafe {
        std::slice::from_raw_parts_mut(psi_data, n_pw * n_bands)
            .copy_from_slice(&psi_host_out);
        std::slice::from_raw_parts_mut(hpsi_out, n_pw * n_bands)
            .copy_from_slice(&hpsi_host);
        std::slice::from_raw_parts_mut(eigenvalues_ptr, n_bands)
            .copy_from_slice(&davidson_result.eigenvalues);
    }

    // Set converged flag: true if all bands locked (residual below tolerance)
    let all_converged = davidson_result.n_locked >= n_bands;
    unsafe { *converged = all_converged as c_int; };

    Ok(())
}
