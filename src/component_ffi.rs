// ---------------------------------------------------------------------------
// Component FFI: CASTEP ↔ Rust per-component swap-in
//
// Exposes individual components of the pure-Rust SCF to CASTEP via C ABI so
// that each component can be swapped in one at a time inside the CASTEP SCF
// loop:
//
//   1. V_eff assembly     (Hartree + PBE XC + ionic, CPU)  — chemrust_comp_locpot
//   2. Density from wavefunctions (soft + augmented, GPU)  — chemrust_comp_density_kpt
//   3. Pulay/DIIS density mixing (Kerker preconditioned)   — chemrust_comp_mix_init/step
//   4. Occupations + Fermi level (Gaussian smearing, CPU)  — chemrust_comp_occupations
//
// A single `ComponentCtx` handle holds the shared cell/potential/grid state
// plus lazily-created GPU state and mixing history.  CPU components never
// touch the GPU.
// ---------------------------------------------------------------------------

use std::ffi::{c_char, c_void, CStr};
use std::os::raw::{c_double, c_int};
use std::sync::{Arc, Mutex};

use chemrust_hamiltonian_core::{
    upsample_density_to_fine_grid, CellGeometry, GVectorGrid, PseudopotentialSet, RealLattice,
    RecipLattice, VEffBuilder,
    fft::RealGrid,
    nlcc, poisson, xc,
    Density as CoreDensity,
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use num_complex::Complex64;

use crate::device::blas::{op, BlasHandle, ZgemmConfig};
use crate::device::pcie::PcieAccount;
use crate::device::unflatten_f64;
use crate::device::{CudaComplex, Gpu};
use crate::density::{build_q_sf_cache, compute_aug_density_gpu, QSfCache};
use crate::eigensolver::kernels::CudaKernelSet;
use crate::eigensolver::vnl_data::{
    build_handle_shared_vnl, HandleSharedVnl, KptSharedVnl, VnlBatchData,
};
use crate::layout::{ColumnDistributed, WavefunctionSet};
use crate::mixing::{DensityHistory, MixingOff, Pulay};
use crate::types::{Density, KPoint, SmearingParams, SmearingWidth, WaveGridArray};

pub const CHEM_COMP_OK: c_int = 0;
pub const CHEM_COMP_CUDA_ERROR: c_int = 3;
pub const CHEM_COMP_NULL_HANDLE: c_int = 4;

// ---- Handle ----------------------------------------------------------------

/// Per-k-point data for component FFI.
struct CompKpt {
    k_point: KPoint,
    pw_coords: Vec<[i32; 3]>,
    n_pw: usize,
}

/// Lazily-created GPU state for GPU components (density).
struct GpuState {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: CudaKernelSet,
    blas: BlasHandle,
    handle_shared_vnl: Arc<HandleSharedVnl>,
    /// Per-k-point VnlBatchData (lazy, built on first density call).
    vnl: Vec<Option<VnlBatchData>>,
    shared_vnl: Vec<Option<Arc<KptSharedVnl>>>,
    /// QSfCache for augmented density (geometry-static).
    qsf: Option<QSfCache>,
}

/// Mixing state (Pulay/DIIS with Kerker preconditioner).
struct MixState {
    history: DensityHistory<Pulay>,
    /// Last mixed fine-grid density per spin (x-fastest flat), used for the
    /// no-mix path (CASTEP keeps the previous mixed density).
    last_mixed: Vec<Vec<f64>>,
}

pub struct ComponentCtx {
    // --- CPU core (cell, pots, grids, k-points) ---
    cell: CellGeometry,
    pots: PseudopotentialSet,
    wave_grid: GVectorGrid,
    fine_grid: GVectorGrid,
    /// FINE grid dims [ngz, ngy, ngx] (order returned by GVectorGrid::grid()).
    fine_dims: [usize; 3],
    /// WAVE grid dims [ngz, ngy, ngx].
    wave_dims: [usize; 3],
    kpts: Vec<CompKpt>,
    nspins: usize,
    n_electrons: f64,
    smearing: SmearingParams,

    // --- Lazy GPU + mixing state ---
    gpu: Mutex<Option<GpuState>>,
    mix: Mutex<Option<MixState>>,
}

// ---- Init / destroy ---------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_comp_init(
    num_species: c_int,
    species_symbols: *const *const c_char,
    species_pots: *const *const c_char,
    real_lattice: *const c_double,
    recip_lattice: *const c_double,
    num_ions: c_int,
    ion_species: *const c_int,
    ion_positions: *const c_double,
    nkpts: c_int,
    num_pw_per_kpt: *const c_int,
    pw_grid_idx: *const c_int,
    kpt_coords: *const c_double,
    kpt_weights: *const c_double,
    ngx: c_int,
    ngy: c_int,
    ngz: c_int,
    ngx_std: c_int,
    ngy_std: c_int,
    ngz_std: c_int,
    nspins: c_int,
    n_electrons: c_double,
    smearing_width_ev: c_double,
    handle_out: *mut *mut c_void,
) -> c_int {
    if handle_out.is_null() {
        return CHEM_COMP_NULL_HANDLE;
    }
    let ctx = match init_comp_inner(
        num_species,
        species_symbols,
        species_pots,
        real_lattice,
        recip_lattice,
        num_ions,
        ion_species,
        ion_positions,
        nkpts,
        num_pw_per_kpt,
        pw_grid_idx,
        kpt_coords,
        kpt_weights,
        ngx,
        ngy,
        ngz,
        ngx_std,
        ngy_std,
        ngz_std,
        nspins,
        n_electrons,
        smearing_width_ev,
    ) {
        Ok(ctx) => ctx,
        Err(code) => {
            eprintln!("[chemrust-comp] init failed code={code}");
            return code;
        }
    };
    eprintln!(
        "[chemrust-comp] init ok: fine=({},{},{}) wave=({},{},{}) nkpts={} nspins={}",
        ngx, ngy, ngz, ngx_std, ngy_std, ngz_std, nkpts, nspins
    );
    unsafe { *handle_out = Box::into_raw(ctx) as *mut c_void };
    CHEM_COMP_OK
}

fn init_comp_inner(
    num_species: c_int,
    species_symbols: *const *const c_char,
    species_pots: *const *const c_char,
    real_lattice: *const c_double,
    recip_lattice: *const c_double,
    num_ions: c_int,
    ion_species: *const c_int,
    ion_positions: *const c_double,
    nkpts: c_int,
    num_pw_per_kpt: *const c_int,
    pw_grid_idx: *const c_int,
    kpt_coords: *const c_double,
    kpt_weights: *const c_double,
    ngx: c_int,
    ngy: c_int,
    ngz: c_int,
    ngx_std: c_int,
    ngy_std: c_int,
    ngz_std: c_int,
    nspins: c_int,
    n_electrons: c_double,
    smearing_width_ev: c_double,
) -> Result<Box<ComponentCtx>, c_int> {
    // Pseudopotentials
    let ns = num_species as usize;
    let syms: Vec<String> = (0..ns)
        .map(|i| unsafe { CStr::from_ptr(*species_symbols.add(i)) }.to_str().unwrap_or("").to_string())
        .collect();
    let pot_paths: Vec<String> = (0..ns)
        .map(|i| unsafe { CStr::from_ptr(*species_pots.add(i)) }.to_str().unwrap_or("").to_string())
        .collect();
    let mut pots = PseudopotentialSet::new();
    for (sym, path) in syms.iter().zip(pot_paths.iter()) {
        let pp = chemrust_hamiltonian_core::pseudopotential::Pseudopotential::from_path(path)
            .map_err(|e| {
                eprintln!("[chemrust-comp] failed {path}: {e}");
                CHEM_COMP_CUDA_ERROR
            })?;
        pots.insert(sym.clone(), pp);
    }

    // Cell geometry
    let rl = unsafe { std::slice::from_raw_parts(real_lattice as *const f64, 9) };
    let rp = unsafe { std::slice::from_raw_parts(recip_lattice as *const f64, 9) };
    let mut rlat = [[0.0f64; 3]; 3];
    let mut rcip = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rlat[i][j] = rl[i * 3 + j];
            rcip[i][j] = rp[i * 3 + j];
        }
    }
    let real_lat = RealLattice::from_inner(rlat);
    let recip_lat = RecipLattice::from_inner(rcip);
    let vol = (rlat[0][0] * (rlat[1][1] * rlat[2][2] - rlat[1][2] * rlat[2][1])
        + rlat[0][1] * (rlat[1][2] * rlat[2][0] - rlat[1][0] * rlat[2][2])
        + rlat[0][2] * (rlat[1][0] * rlat[2][1] - rlat[1][1] * rlat[2][0]))
        .abs();

    let ni = num_ions as usize;
    let ion_sp: Vec<usize> = (0..ni).map(|i| unsafe { *(ion_species.add(i)) } as usize).collect();
    let pos_f = unsafe { std::slice::from_raw_parts(ion_positions as *const f64, 3 * ni) };
    let mut pos = ndarray::Array2::<f64>::zeros((ni, 3));
    for i in 0..ni {
        for j in 0..3 {
            pos[[i, j]] = pos_f[3 * i + j];
        }
    }
    let cell = CellGeometry {
        real_lattice: real_lat,
        recip_lattice: recip_lat,
        volume: vol,
        num_species: ns,
        num_ions: ni,
        ionic_positions: pos,
        species_symbols: syms,
        species_pot_files: vec![],
        num_ions_in_species: vec![],
        ion_species: ion_sp,
        max_ions_in_species: 0,
        species_lcao_states: vec![],
    };

    // Grids.  FINE grid = CASTEP density grid; WAVE grid = CASTEP FFT grid
    // for wavefunctions (STANDARD grid in CASTEP terminology).
    let fine_grid = GVectorGrid::new(ngx as usize, ngy as usize, ngz as usize, recip_lat);
    let wave_grid = GVectorGrid::new(
        ngx_std as usize,
        ngy_std as usize,
        ngz_std as usize,
        recip_lat,
    );

    // Per-k-point data.  pw_grid_idx is encoded on the STANDARD grid with
    // strides [1, ngx_std, ngx_std*ngy_std] (ix-innermost), 1-based.
    let nk = nkpts as usize;
    let npwk = unsafe { std::slice::from_raw_parts(num_pw_per_kpt as *const i32, nk) };
    let kfrac = unsafe { std::slice::from_raw_parts(kpt_coords as *const f64, 3 * nk) };
    let kwt = unsafe { std::slice::from_raw_parts(kpt_weights as *const f64, nk) };
    let maxpw = *npwk.iter().max().unwrap_or(&0) as usize;
    let gidx = unsafe { std::slice::from_raw_parts(pw_grid_idx as *const i32, maxpw * nk) };

    let ngx_s = ngx_std as i32;
    let ngy_s = ngy_std as i32;
    let ngz_s = ngz_std as i32;

    let mut kpts = Vec::with_capacity(nk);
    for ik in 0..nk {
        let n_pw = npwk[ik] as usize;
        let kf = [kfrac[3 * ik], kfrac[3 * ik + 1], kfrac[3 * ik + 2]];
        let weight = if kwt.len() == nk {
            kwt[ik]
        } else {
            1.0
        };
        let mut pw_coords = Vec::with_capacity(n_pw);
        for ipw in 0..n_pw {
            let idx_1based = gidx[ik * maxpw + ipw];
            pw_coords.push(fft_idx_to_coord(idx_1based, ngx_s, ngy_s, ngz_s));
        }
        kpts.push(CompKpt {
            k_point: KPoint { coords: kf, weight },
            pw_coords,
            n_pw,
        });
    }

    let smearing = SmearingParams::builder()
        .width(SmearingWidth::ev(smearing_width_ev as f64))
        .build();

    Ok(Box::new(ComponentCtx {
        cell,
        pots,
        wave_grid,
        fine_grid,
        fine_dims: [ngz as usize, ngy as usize, ngx as usize],
        wave_dims: [ngz_std as usize, ngy_std as usize, ngx_std as usize],
        kpts,
        nspins: nspins as usize,
        n_electrons: n_electrons as f64,
        smearing,
        gpu: Mutex::new(None),
        mix: Mutex::new(None),
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_comp_destroy(handle: *mut c_void) -> c_int {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle as *mut ComponentCtx));
        }
    }
    CHEM_COMP_OK
}

/// Convert a 1-based CASTEP standard-grid index to signed Miller indices,
/// identical to the FFI eigensolver path (see ffi.rs).
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

// ---- Component 1: V_eff assembly (CPU) --------------------------------------

/// Assemble V_eff = V_H + V_xc(PBE) + V_ion on the fine grid from CASTEP's
/// fine-grid density, mirroring the pure-Rust SCF `build_v_eff_with_energy`
/// path.  `dens` is CASTEP's fine-grid charge density (x-fastest flat,
/// CASTEP column order): total density for spin-polarised runs, the single
/// channel for non-spin runs.  `dens_spin` is ρ_up − ρ_dn (null for
/// non-spin).
///
/// Writes V_eff into `veff_out` (same layout) and, when the output
/// pointers are non-null, the energy pieces (E_xc, E_Hartree, ∫ρV_xc) in
/// Hartree.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_comp_locpot(
    handle: *mut c_void,
    ispin: c_int,
    dens: *const c_double,
    dens_spin: *const c_double,
    veff_out: *mut c_double,
    e_xc_out: *mut c_double,
    e_hartree_out: *mut c_double,
    rho_vxc_out: *mut c_double,
) -> c_int {
    match unsafe { locpot_inner(handle, ispin, dens, dens_spin, veff_out, e_xc_out, e_hartree_out, rho_vxc_out) } {
        Ok(()) => CHEM_COMP_OK,
        Err(code) => code,
    }
}

fn locpot_inner(
    handle: *mut c_void,
    ispin: c_int,
    dens: *const c_double,
    dens_spin: *const c_double,
    veff_out: *mut c_double,
    e_xc_out: *mut c_double,
    e_hartree_out: *mut c_double,
    rho_vxc_out: *mut c_double,
) -> Result<(), c_int> {
    eprintln!("[chemrust-comp] locpot called: handle={:?} ispin={ispin} dens_null={} veff_null={}", handle, dens.is_null(), veff_out.is_null());
    let ctx = unsafe { (handle as *mut ComponentCtx).as_ref() }.ok_or(CHEM_COMP_NULL_HANDLE)?;
    if dens.is_null() || veff_out.is_null() {
        eprintln!("[chemrust-comp] locpot null pointer arg");
        return Err(CHEM_COMP_NULL_HANDLE);
    }

    let [ngz_f, ngy_f, ngx_f] = ctx.fine_dims;
    let n_fine = ngx_f * ngy_f * ngz_f;

    // CASTEP flat order (x-fastest) → Rust Array3 (ngx, ngy, ngz).
    let dens_vec: Vec<f64> =
        unsafe { std::slice::from_raw_parts(dens as *const f64, n_fine) }.to_vec();
    let rho_fine_arr = unflatten_f64(dens_vec, &[ngx_f, ngy_f, ngz_f]);
    let rho_fine = RealGrid::from_inner(rho_fine_arr);

    let mut e_xc = 0.0f64;
    let mut e_hartree = 0.0f64;
    let mut rho_vxc = 0.0f64;

    let spin_grid = if ctx.nspins == 2 {
        let spin_vec: Vec<f64> = if dens_spin.is_null() {
            vec![0.0; n_fine]
        } else {
            unsafe { std::slice::from_raw_parts(dens_spin as *const f64, n_fine) }.to_vec()
        };
        Some(unflatten_f64(spin_vec, &[ngx_f, ngy_f, ngz_f]))
    } else {
        None
    };

    let spin_real = match &spin_grid {
        Some(arr) => Some(RealGrid::from_inner(arr.clone())),
        None => None,
    };
    compute_energy_pieces(ctx, &rho_fine, spin_real.as_ref(), &mut e_xc, &mut e_hartree, &mut rho_vxc);

    let arr = if ctx.nspins == 1 {
        let rho = CoreDensity::from_inner(rho_fine.clone());
        let pot = VEffBuilder::<chemrust_hamiltonian_core::NonSpin>::new(
            &ctx.cell,
            &ctx.pots,
            &ctx.fine_grid,
        )
        .with_density(rho, None)
        .assemble()
        .map_err(|e| {
            eprintln!("[chemrust-comp] locpot NonSpin assemble failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?;
        pot.as_real_grid().as_real_array().to_owned()
    } else {
        let rho = CoreDensity::from_inner(rho_fine.clone());
        let spin = spin_real
            .clone()
            .map(|rg| CoreDensity::from_inner(rg))
            .expect("nspins=2 requires spin density grid");
        let (pot_up, pot_dn) = VEffBuilder::<chemrust_hamiltonian_core::SpinCollinear>::new(
            &ctx.cell,
            &ctx.pots,
            &ctx.fine_grid,
        )
        .with_density(rho, Some(spin))
        .assemble()
        .map_err(|e| {
            eprintln!("[chemrust-comp] locpot SpinCollinear assemble failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?;
        if ispin == 0 {
            pot_up.as_real_grid().as_real_array().to_owned()
        } else {
            pot_dn.as_real_grid().as_real_array().to_owned()
        }
    };

    write_realgrid_xfastest(arr, veff_out);

    if !e_xc_out.is_null() {
        unsafe { *e_xc_out = e_xc };
    }
    if !e_hartree_out.is_null() {
        unsafe { *e_hartree_out = e_hartree };
    }
    if !rho_vxc_out.is_null() {
        unsafe { *rho_vxc_out = rho_vxc };
    }
    Ok(())
}

/// E_H = 0.5·Σ ρV_H /N, E_xc (PBE functional), ∫ρV_xc/N on the fine grid.
/// Matches scf.rs `build_v_eff_with_energy_impl` integrals (weight 1/N,
/// density in CASTEP raw units ρ_phys·Ω).
fn compute_energy_pieces(
    ctx: &ComponentCtx,
    rho_fine: &RealGrid<f64>,
    spin: Option<&RealGrid<f64>>,
    e_xc: &mut f64,
    e_hartree: &mut f64,
    rho_vxc: &mut f64,
) {
    let v_h = poisson::solve_poisson(&CoreDensity::from_inner(rho_fine.clone()), &ctx.fine_grid)
        .expect("poisson solve");
    let rho_core = nlcc::reconstruct_rho_core(&ctx.cell, &ctx.pots, &ctx.fine_grid)
        .expect("core density")
        .into_inner();
    let density_total = CoreDensity::from_inner(rho_fine.clone() + &rho_core);
    let n_grid = density_total.as_real_grid().as_real_array().len() as f64;
    let d_v = 1.0 / n_grid;

    *e_hartree = 0.5
        * rho_fine
            .as_real_array()
            .iter()
            .zip(v_h.as_real_grid().as_real_array().iter())
            .map(|(&rv, &vh)| rv * vh * d_v)
            .sum::<f64>();

    if let Some(m) = spin {
        let v_xc = xc::compute_pbe_xc_spin(
            density_total.as_real_grid().as_real_array(),
            m.as_real_array(),
            &ctx.fine_grid,
            ctx.cell.volume,
        )
        .expect("PBE spin xc");
        *e_xc = v_xc.energy;
        // Per-spin double counting: (ρ+m)/2·V_xc_up + (ρ-m)/2·V_xc_dn.
        *rho_vxc = rho_fine
            .as_real_array()
            .iter()
            .zip(m.as_real_array().iter())
            .zip(v_xc.v_xc_up.iter().zip(v_xc.v_xc_dn.iter()))
            .map(|((&rv, &rm), (&vu, &vd))| 0.5 * ((rv + rm) * vu + (rv - rm) * vd) * d_v)
            .sum::<f64>();
    } else {
        let v_xc = xc::compute_pbe_xc(
            density_total.as_real_grid().as_real_array(),
            &ctx.fine_grid,
            ctx.cell.volume,
        )
        .expect("PBE xc");
        *e_xc = v_xc.energy;
        *rho_vxc = rho_fine
            .as_real_array()
            .iter()
            .zip(v_xc.v_xc.iter())
            .map(|(&rv, &vxc)| rv * vxc * d_v)
            .sum::<f64>();
    }
}

/// Write an Array3 (shape (ngx, ngy, ngz), x-first index order) as CASTEP
/// x-fastest flat order.
fn write_realgrid_xfastest(arr: ndarray::Array3<f64>, out: *mut c_double) {
    let (ngx, ngy, ngz) = arr.dim();
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out as *mut f64, ngx * ngy * ngz) };
    let mut idx = 0;
    for iz in 0..ngz {
        for iy in 0..ngy {
            for ix in 0..ngx {
                out_slice[idx] = arr[[ix, iy, iz]];
                idx += 1;
            }
        }
    }
}

// ---- GPU state management ---------------------------------------------------

/// Create the lazy GPU state if absent.  The returned guard must be held
/// for the whole FFI call scope.
fn ensure_gpu<'a>(ctx: &'a ComponentCtx, pcie: &mut PcieAccount) -> Result<std::sync::MutexGuard<'a, Option<GpuState>>, c_int> {
    let mut guard = ctx.gpu.lock().expect("gpu lock poisoned");
    if guard.is_none() {
        let cctx = CudaContext::new(0).map_err(|e| {
            eprintln!("[chemrust-comp] CudaContext failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?;
        let stream = cctx.default_stream();
        let kernels = CudaKernelSet::new(&cctx).map_err(|e| {
            eprintln!("[chemrust-comp] kernels failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?;
        let blas = BlasHandle::new(stream.clone()).map_err(|e| {
            eprintln!("[chemrust-comp] blas failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?;
        let handle_shared_vnl = build_handle_shared_vnl(
            &ctx.pots,
            &ctx.cell,
            &ctx.wave_grid,
            Some(&ctx.fine_grid),
            &stream,
            pcie,
        )
        .map_err(|e| {
            eprintln!("[chemrust-comp] HandleSharedVnl build failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?;
        *guard = Some(GpuState {
            ctx: cctx,
            stream,
            kernels,
            blas,
            handle_shared_vnl,
            vnl: (0..ctx.kpts.len()).map(|_| None).collect(),
            shared_vnl: (0..ctx.kpts.len()).map(|_| None).collect(),
            qsf: None,
        });
    }
    Ok(guard)
}

// ---- Component 2: density from wavefunctions (GPU) --------------------------

/// Compute the fine-grid (soft + augmented) density for one (k-point, spin)
/// from CASTEP's wavefunction coefficients, matching the pure-Rust SCF
/// `compute_density_from_wavefunctions` + `combine_soft_aug_on_fine`.
///
/// `psi` points at CASTEP `wvfn%coeffs(1,1,nk,ispin)`: band blocks of
/// `n_pw` with leading dimension `max_n_pw` (Fortran column-major).
/// `occ` is CASTEP's occupancy array for this (nk, ispin), length `n_bands`.
/// Output `dens_out`: fine-grid total density, x-fastest flat.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_comp_density_kpt(
    handle: *mut c_void,
    ikpt: c_int,
    ispin: c_int,
    psi: *const CudaComplex,
    max_n_pw: c_int,
    n_pw: c_int,
    n_bands: c_int,
    occ: *const c_double,
    dens_out: *mut c_double,
) -> c_int {
    match unsafe {
        density_kpt_inner(
            handle, ikpt, ispin, psi, max_n_pw, n_pw, n_bands, occ, dens_out,
        )
    } {
        Ok(()) => CHEM_COMP_OK,
        Err(code) => code,
    }
}

fn density_kpt_inner(
    handle: *mut c_void,
    _ikpt: c_int,
    _ispin: c_int,
    psi: *const CudaComplex,
    max_n_pw: c_int,
    n_pw: c_int,
    n_bands: c_int,
    occ: *const c_double,
    dens_out: *mut c_double,
) -> Result<(), c_int> {
    let ctx = unsafe { (handle as *mut ComponentCtx).as_ref() }.ok_or(CHEM_COMP_NULL_HANDLE)?;
    if psi.is_null() || occ.is_null() || dens_out.is_null() {
        return Err(CHEM_COMP_NULL_HANDLE);
    }
    eprintln!(
        "[chemrust-comp] density_kpt called: ik={_ikpt} ispin={_ispin} n_pw={n_pw} n_bands={n_bands}"
    );
    let ik = _ikpt as usize;
    if ik >= ctx.kpts.len() {
        return Err(CHEM_COMP_CUDA_ERROR);
    }
    let kp = &ctx.kpts[ik];
    let n_bands = n_bands as usize;
    let n_pw = n_pw as usize;
    let max_n_pw = max_n_pw as usize;

    // Compact psi: band-stride [n_pw × n_bands] layout.
    let raw = unsafe { std::slice::from_raw_parts(psi, max_n_pw * n_bands) };
    let mut psi_compact = Vec::with_capacity(n_pw * n_bands);
    for b in 0..n_bands {
        let start = b * max_n_pw;
        for ipw in 0..n_pw {
            let c = &raw[start + ipw];
            psi_compact.push(Complex64::new(c.x, c.y));
        }
    }

    let occ_vec: Vec<f64> =
        unsafe { std::slice::from_raw_parts(occ as *const f64, n_bands) }.to_vec();

    let mut pcie = PcieAccount::default();
    let mut gpu_guard = ensure_gpu(ctx, &mut pcie)?;
    let gpu = gpu_guard
        .as_mut()
        .expect("ensure_gpu created the state");

    // 1. Soft (PW) wave-grid density.
    let pw_fft_indices = crate::scf::pw_coords_to_fft_indices(&kp.pw_coords, &ctx.wave_grid);
    let soft_wave = crate::density::construct_density_gpu()
        .psi_data(&psi_compact)
        .occupations(&occ_vec)
        .fft_indices(&pw_fft_indices)
        .wave_grid(&ctx.wave_grid)
        .cell_volume(ctx.cell.volume)
        .n_bands(n_bands)
        .n_pw(n_pw)
        .kernels(&gpu.kernels)
        .stream(&gpu.stream)
        .call()
        .map_err(|e| {
            eprintln!("[chemrust-comp] construct_density_gpu failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?;

    // 2. Upsample soft density to the fine grid (matches
    //    combine_soft_aug_on_fine in scf.rs).
    let soft_wave_arr = soft_wave.as_wave_array().to_owned();
    let soft_fine = upsample_density_to_fine_grid(
        &RealGrid::from_inner(soft_wave_arr),
        &ctx.wave_grid,
        &ctx.fine_grid,
    )
    .map_err(|e| {
        eprintln!("[chemrust-comp] upsample failed: {e}");
        CHEM_COMP_CUDA_ERROR
    })?;

    // 3. Augmented density from β|ψ> projections.
    let [ngz_f, ngy_f, ngx_f] = ctx.fine_dims;
    let aug_fine = if ctx.cell.num_species > 0 {
        let beta_psi_per_ion =
            compute_beta_psi_per_ion(ctx, gpu, &psi_compact, n_bands, n_pw, ik, &mut pcie)?;
        if gpu.qsf.is_none() {
            let cache = build_q_sf_cache(
                &ctx.pots,
                &ctx.cell,
                &ctx.fine_grid,
                &gpu.stream,
                &mut pcie,
            )
            .map_err(|e| {
                eprintln!("[chemrust-comp] QSfCache build failed: {e}");
                CHEM_COMP_CUDA_ERROR
            })?;
            gpu.qsf = Some(cache);
        }
        let qsf = gpu
            .qsf
            .as_ref()
            .expect("QSfCache built above");
        compute_aug_density_gpu(
            qsf,
            &beta_psi_per_ion,
            &occ_vec,
            &gpu.stream,
            &mut pcie,
            &gpu.kernels,
        )
        .map_err(|e| {
            eprintln!("[chemrust-comp] aug density failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?
    } else {
        RealGrid::from_inner(ndarray::Array3::zeros((ngx_f, ngy_f, ngz_f)))
    };

    // 4. Combine: fine-grid total = soft_fine + aug_fine (matches
    //    combine_soft_aug_on_fine).
    let total = soft_fine + &aug_fine;

    // 5. Write x-fastest flat.
    write_realgrid_xfastest(total.as_real_array().to_owned(), dens_out);
    Ok(())
}

/// Per-ion β|ψ> projections: p_i = β_i^H · ψ for every ion in the cell.
/// Returns GPU-resident slices in the layout expected by
/// `compute_aug_density_gpu` (per-ion col-major n_expanded × n_bands).
fn compute_beta_psi_per_ion(
    ctx: &ComponentCtx,
    gpu: &mut GpuState,
    psi_compact: &[Complex64],
    n_bands: usize,
    n_pw: usize,
    ik: usize,
    pcie: &mut PcieAccount,
) -> Result<Vec<CudaSlice<CudaComplex>>, c_int> {
    // Lazily build VnlBatchData for this k-point (mirrors ffi.rs step_inner).
    if gpu.vnl[ik].is_none() {
        let shared = gpu.shared_vnl[ik].clone();
        let vnl = VnlBatchData::precompute_with_d_override(
            &ctx.kpts[ik].pw_coords,
            &ctx.pots,
            &ctx.cell,
            &ctx.wave_grid,
            Some(&ctx.fine_grid),
            &ctx.kpts[ik].k_point,
            psi_compact,
            n_bands,
            n_pw,
            None,
            None,
            None,
            shared,
            Some(gpu.handle_shared_vnl.clone()),
            None, // solver_thunk — no Woodbury in the component path
            &gpu.stream,
            pcie,
            &gpu.blas,
            &gpu.kernels,
        )
        .map_err(|e| {
            eprintln!("[chemrust-comp] VnlBatchData build failed: {e}");
            CHEM_COMP_CUDA_ERROR
        })?;
        gpu.shared_vnl[ik] = Some(vnl.shared.clone());
        gpu.vnl[ik] = Some(vnl);
    }
    let vnl = gpu.vnl[ik].as_ref().ok_or(CHEM_COMP_CUDA_ERROR)?;

    let psi_gpu: Gpu<WavefunctionSet<ColumnDistributed>> = Gpu::from_host(
        &WavefunctionSet::new(psi_compact.to_vec(), n_bands, n_pw),
        &gpu.stream,
    )
    .map_err(|e| {
        eprintln!("[chemrust-comp] psi upload failed: {e}");
        CHEM_COMP_CUDA_ERROR
    })?;

    let mut beta_psi_per_ion: Vec<CudaSlice<CudaComplex>> = Vec::with_capacity(vnl.entries.len());
    for entry in &vnl.entries {
        let ne = entry.n_expanded;
        let mut p: CudaSlice<CudaComplex> = gpu
            .stream
            .alloc_zeros(ne as usize * n_bands)
            .map_err(|e| {
                eprintln!("[chemrust-comp] alloc failed: {e}");
                CHEM_COMP_CUDA_ERROR
            })?;
        // p = beta_g^H · psi — same gemm config as apply_s_times in
        // hamiltonian.rs.
        unsafe {
            gpu.blas.gemm_c64(
                ZgemmConfig {
                    transa: op::C,
                    transb: op::N,
                    m: ne,
                    n: n_bands as i32,
                    k: n_pw as i32,
                    alpha: CudaComplex { x: 1.0, y: 0.0 },
                    lda: n_pw as i32,
                    ldb: n_pw as i32,
                    beta: CudaComplex { x: 0.0, y: 0.0 },
                    ldc: ne,
                },
                &entry.beta_g,
                psi_gpu.as_device_slice(),
                &mut p,
            )
            .map_err(|e| {
                eprintln!("[chemrust-comp] beta gemm failed: {e}");
                CHEM_COMP_CUDA_ERROR
            })?;
        }
        beta_psi_per_ion.push(p);
    }
    Ok(beta_psi_per_ion)
}

// ---- Component 3: Pulay/DIIS mixing (GPU) -----------------------------------

/// Initialize the Rust mixing history.  `mixing_amp` is per spin
/// (length nspins), `g2_cutoff` the Kerker G² mask cutoff in atomic units
/// (max G² on the wave grid; 0.0 disables the mask), `dens_mixed_in` the
/// per-spin initial mixed fine-grid density (x-fastest flat, nspins ×
/// n_fine_points).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_comp_mix_init(
    handle: *mut c_void,
    mixing_amp: *const c_double,
    g2_cutoff: c_double,
    dens_mixed_in: *const c_double,
    n_fine_points: c_int,
    status: *mut c_int,
) -> c_int {
    match unsafe { mix_init_inner(handle, mixing_amp, g2_cutoff, dens_mixed_in, n_fine_points) } {
        Ok(code) => {
            if !status.is_null() {
                unsafe { *status = code };
            }
            code
        }
        Err(code) => {
            if !status.is_null() {
                unsafe { *status = code };
            }
            code
        }
    }
}

fn mix_init_inner(
    handle: *mut c_void,
    mixing_amp: *const c_double,
    g2_cutoff: c_double,
    dens_mixed_in: *const c_double,
    n_fine_points: c_int,
) -> Result<c_int, c_int> {
    let ctx = unsafe { (handle as *mut ComponentCtx).as_ref() }.ok_or(CHEM_COMP_NULL_HANDLE)?;
    if mixing_amp.is_null() || dens_mixed_in.is_null() {
        return Err(CHEM_COMP_NULL_HANDLE);
    }
    let n_fine = n_fine_points as usize;

    let g2_opt = if g2_cutoff > 0.0 {
        Some(g2_cutoff as f64)
    } else {
        // Match scf.rs: mask Kerker beyond the wave-grid G^2 max.
        Some(
            ctx.wave_grid
                .g2()
                .iter()
                .cloned()
                .fold(0.0f64, f64::max),
        )
    };
    let history = match DensityHistory::<MixingOff>::with_amplitude(ctx.nspins, 0.5)
        .into_kerker(&ctx.fine_grid, g2_opt)
    {
        Ok(h) => h.into_pulay(),
        Err(e) => {
            eprintln!("[chemrust-comp] into_kerker failed: {e}");
            return Err(CHEM_COMP_CUDA_ERROR);
        }
    };

    // Per-spin amplitude from Fortran.
    let amp_slice =
        unsafe { std::slice::from_raw_parts(mixing_amp as *const f64, ctx.nspins) };
    let mut history = history;
    for (ispin, &amp) in amp_slice.iter().enumerate() {
        history.set_mixing_amplitude(ispin, amp);
    }

    // Snapshot the initial mixed density per spin (no-mix path source).
    let nspins = ctx.nspins;
    let mixed_base =
        unsafe { std::slice::from_raw_parts(dens_mixed_in as *const f64, nspins * n_fine) };
    let last_mixed: Vec<Vec<f64>> = (0..nspins)
        .map(|s| mixed_base[s * n_fine..(s + 1) * n_fine].to_vec())
        .collect();

    let mut mix_guard = ctx.mix.lock().expect("mix lock poisoned");
    *mix_guard = Some(MixState {
        history,
        last_mixed,
    });
    Ok(CHEM_COMP_OK)
}

/// One mixing step for one spin channel.
///
/// `dens_in`: new fine-grid density (x-fastest flat) from the current
/// wavefunctions.  `do_mix` mirrors CASTEP's `mix_density` gate: 0 keeps the
/// previously mixed density (no history update), 1 runs Pulay/DIIS.
/// `mixed_out`: resulting mixed density.  `res_norm_out`: norm of the
/// density change (sqrt(ΣΔ²/N); -1 when no mixing).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_comp_mix_step(
    handle: *mut c_void,
    ispin: c_int,
    dens_in: *const c_double,
    do_mix: c_int,
    mixed_out: *mut c_double,
    res_norm_out: *mut c_double,
) -> c_int {
    match unsafe { mix_step_inner(handle, ispin, dens_in, do_mix, mixed_out, res_norm_out) } {
        Ok(()) => CHEM_COMP_OK,
        Err(code) => code,
    }
}

fn mix_step_inner(
    handle: *mut c_void,
    ispin: c_int,
    dens_in: *const c_double,
    do_mix: c_int,
    mixed_out: *mut c_double,
    res_norm_out: *mut c_double,
) -> Result<(), c_int> {
    let ctx = unsafe { (handle as *mut ComponentCtx).as_ref() }.ok_or(CHEM_COMP_NULL_HANDLE)?;
    if dens_in.is_null() || mixed_out.is_null() {
        return Err(CHEM_COMP_NULL_HANDLE);
    }
    let ispin = ispin as usize;
    if ispin >= ctx.nspins {
        return Err(CHEM_COMP_CUDA_ERROR);
    }
    let [ngz_f, ngy_f, ngx_f] = ctx.fine_dims;
    let n_fine = ngx_f * ngy_f * ngz_f;

    // Spin-polarised callers pass a combined [rho_up; rho_dn] buffer;
    // non-spin callers pass a single-channel buffer.
    let dens_in_base = unsafe { dens_in.add(ispin * n_fine) };
    let mixed_out_base = unsafe { mixed_out.add(ispin * n_fine) };
    let dens_vec: Vec<f64> =
        unsafe { std::slice::from_raw_parts(dens_in_base as *const f64, n_fine) }.to_vec();

    let mut mix_guard = ctx.mix.lock().expect("mix lock poisoned");
    let mix_state = match mix_guard.as_mut() {
        Some(s) => s,
        None => {
            // Uninitialized: seed the no-mix snapshot and build a default
            // history so the pass-through stays valid.
            let last = vec![dens_vec.clone(); ctx.nspins];
            let history =
                match DensityHistory::<MixingOff>::with_amplitude(ctx.nspins, 0.5)
                    .into_kerker(
                        &ctx.fine_grid,
                        Some(
                            ctx.wave_grid
                                .g2()
                                .iter()
                                .cloned()
                                .fold(0.0f64, f64::max),
                        ),
                    )
                {
                    Ok(k) => k.into_pulay(),
                    Err(e) => {
                        eprintln!("[chemrust-comp] mix lazy init failed: {e}");
                        return Err(CHEM_COMP_CUDA_ERROR);
                    }
                };
            *mix_guard = Some(MixState { history, last_mixed: last });
            mix_guard.as_mut().expect("just set")
        }
    };

    if do_mix == 0 {
        // No mixing: previous mixed density unchanged (CASTEP behavior).
        write_realgrid_xfastest(
            unflatten_f64(mix_state.last_mixed[ispin].clone(), &[ngx_f, ngy_f, ngz_f]),
            mixed_out_base,
        );
        if !res_norm_out.is_null() {
            unsafe { *res_norm_out = -1.0 };
        }
        return Ok(());
    }

    let in_arr = unflatten_f64(dens_vec.clone(), &[ngx_f, ngy_f, ngz_f]);
    let dens = Density::from_inner(WaveGridArray::from_inner(in_arr.clone()));
    let (mixed, _snapshot) = mix_state.history.mix(dens, ispin);
    let mixed_arr = mixed.as_wave_array().to_owned();

    // Norm of the density change (diagnostic; mirrors CASTEP res_norm role).
    let mut norm_sq = 0.0f64;
    for (i, j) in mixed_arr.iter().zip(in_arr.iter()) {
        let d = i - j;
        norm_sq += d * d;
    }
    let res_norm = (norm_sq / n_fine as f64).sqrt();

    // Store last mixed (x-fastest flat) for the no-mix path.
    let mut flat = vec![0.0f64; n_fine];
    {
        let mut idx = 0;
        for iz in 0..ngz_f {
            for iy in 0..ngy_f {
                for ix in 0..ngx_f {
                    flat[idx] = mixed_arr[[ix, iy, iz]];
                    idx += 1;
                }
            }
        }
    }
    mix_state.last_mixed[ispin] = flat;
    write_realgrid_xfastest(mixed_arr, mixed_out_base);
    if !res_norm_out.is_null() {
        unsafe { *res_norm_out = res_norm };
    }
    Ok(())
}

// ---- Component 4: occupations + Fermi level (CPU) ---------------------------

/// Compute CASTEP-style Gaussian-smearing occupations for one spin channel
/// from all k-point eigenvalues.  `eig` points at CASTEP
/// `eigenvalues(1,1,ispin)`: per-k-point blocks of `n_bands_kpt` with
/// leading dimension `max_n_bands`.  Writes occupations to `occ_out`
/// (same layout) and the Fermi level to `fermi_out` (Hartree).
///
/// `n_electrons_spin`: per-spin electron count for this channel (CASTEP
/// `frac_elec` convention: 0.5·(N ± net_spin) for spin-polarised, N for
/// non-spin).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chemrust_comp_occupations(
    handle: *mut c_void,
    ispin: c_int,
    eig: *const c_double,
    max_n_bands: c_int,
    n_bands_per_kpt: *const c_int,
    occ_out: *mut c_double,
    n_electrons_spin: c_double,
    fermi_out: *mut c_double,
) -> c_int {
    match unsafe {
        occupations_inner(
            handle, ispin, eig, max_n_bands, n_bands_per_kpt, occ_out, n_electrons_spin, fermi_out,
        )
    } {
        Ok(()) => CHEM_COMP_OK,
        Err(code) => code,
    }
}

fn occupations_inner(
    handle: *mut c_void,
    ispin: c_int,
    eig: *const c_double,
    max_n_bands: c_int,
    n_bands_per_kpt: *const c_int,
    occ_out: *mut c_double,
    n_electrons_spin: c_double,
    fermi_out: *mut c_double,
) -> Result<(), c_int> {
    let ctx = unsafe { (handle as *mut ComponentCtx).as_ref() }.ok_or(CHEM_COMP_NULL_HANDLE)?;
    if eig.is_null() || n_bands_per_kpt.is_null() || occ_out.is_null() || fermi_out.is_null() {
        return Err(CHEM_COMP_NULL_HANDLE);
    }
    let _ = ispin;
    let nk = ctx.kpts.len();
    let max_nb = max_n_bands as usize;
    let nb_per_kpt: Vec<usize> =
        unsafe { std::slice::from_raw_parts(n_bands_per_kpt as *const i32, nk) }
            .iter()
            .map(|&b| b as usize)
            .collect();

    // Per-k-point eigenvalue blocks (stride max_n_bands).
    let eig_base = eig as *const f64;
    let mut per_kpt_eigs: Vec<Vec<f64>> = Vec::with_capacity(nk);
    let mut kpt_weights = Vec::with_capacity(nk);
    for ik in 0..nk {
        if nb_per_kpt[ik] > max_nb {
            eprintln!(
                "[chemrust-comp] occupations: n_bands_per_kpt[{ik}]={:?} > max_n_bands={max_nb}",
                nb_per_kpt[ik]
            );
            return Err(CHEM_COMP_CUDA_ERROR);
        }
        let start = ik * max_nb;
        let eigs =
            unsafe { std::slice::from_raw_parts(eig_base.add(start), nb_per_kpt[ik]) }.to_vec();
        per_kpt_eigs.push(eigs);
        kpt_weights.push(ctx.kpts[ik].k_point.weight);
    }

    // CASTEP stores per-spin-channel occupations in [0,1] for both non-spin
    // and spin-polarised runs. The spin-degeneracy factor (2/nspins) is
    // applied by the density and energy consumers, not the occupations.
    let occ_factor = 0.5;
    let (per_kpt_occs, mu) = crate::density::compute_occupations_weighted(
        &per_kpt_eigs,
        &kpt_weights,
        &ctx.smearing,
        n_electrons_spin as f64,
        occ_factor,
    )
    .map_err(|e| {
        eprintln!("[chemrust-comp] occupations failed: {e}");
        CHEM_COMP_CUDA_ERROR
    })?;

    let occ_base = occ_out as *mut f64;
    for ik in 0..nk {
        let start = ik * max_nb;
        unsafe {
            std::slice::from_raw_parts_mut(occ_base.add(start), nb_per_kpt[ik])
                .copy_from_slice(&per_kpt_occs[ik]);
        }
    }
    unsafe { *fermi_out = mu.0 };
    Ok(())
}
