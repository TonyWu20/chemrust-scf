//! Minimal diagnostic: load .check, compute H·ψ once, Rayleigh quotients.
#![allow(non_snake_case)]
use std::sync::Arc;
use chemrust_hamiltonian_core::{CastepBinFile, CheckFile, GVectorGrid, PseudopotentialSet};
use chemrust_scf::{
    apply_full_hamiltonian, apply_s_times, compute_kinetic_energies,
    BlasHandle, KineticPreconditioner, KPoint, PwCoefficients,
    PcieAccount, SolverHandle, VnlBatchData,
    CudaKernelSet,
    downsample_array_to_wave_grid, pw_coords_to_fft_indices,
};
use chemrust_scf::device::CudaComplex;
use chemrust_scf::device::fft::BatchedFftPlan3d;
use cudarc::driver::{CudaContext, CudaSlice};

const DIR: &str = "/export/public_castep_jobs/tony/Cu111_CO_H_dump";
const PDIR: &str = "/export/Potentials";

#[test]
#[ignore = "requires Cu111_CO_H_dump fixture at /export/public_castep_jobs/tony/Cu111_CO_H_dump"]
fn debug_rayleigh() {
    let bin = CastepBinFile::read(std::io::BufReader::new(
        std::fs::File::open(format!("{DIR}/Cu111_CO.castep_bin")).unwrap())).unwrap();
    let chk = CheckFile::read(std::io::BufReader::new(
        std::fs::File::open(format!("{DIR}/Cu111_CO.check")).unwrap())).unwrap();
    let bands_ref: Vec<f64> = std::fs::read_to_string(format!("{DIR}/Cu111_CO.bands")).unwrap()
        .lines().skip_while(|l| !l.trim().starts_with("Spin component")).skip(1)
        .filter_map(|l| l.trim().parse().ok()).collect();

    let wfc = chk.wavefunction.as_ref().unwrap();
    let kpt = &wfc.kpt_data[0];
    let nb = kpt.bands.len();
    let npw = kpt.nplw;
    eprintln!("nb={nb} npw={npw}");

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).unwrap();
    let solver = SolverHandle::new(stream.clone()).unwrap();
    let kernels = CudaKernelSet::new(&ctx).unwrap();

    let cell = &bin.cell;
    let [ngx, ngy, ngz] = wfc.grid;
    let wg = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);
    let gs = (ngx * ngy * ngz) as usize;
    let invn = 1.0 / gs as f64;

    let pwc = kpt.pw_grid_coord.clone();
    let fft_idx = pw_coords_to_fft_indices(&pwc, &wg);
    let fft_idx_dev: CudaSlice<i32> = stream.clone_htod(&fft_idx).unwrap();

    // V_eff
    let pot_text = std::fs::read_to_string(format!("{DIR}/Cu111_CO.pot_fmt")).unwrap();
    let (_g, pot_arr) = chemrust_hamiltonian_core::formatted::parse_pot_fmt(&pot_text).unwrap();
    let veff_inner = chemrust_hamiltonian_core::EffectivePotential::from_inner(
        chemrust_hamiltonian_core::fft::RealGrid::from_inner(pot_arr.clone()));
    let veff_down = downsample_array_to_wave_grid(&pot_arr, &wg, &wg).unwrap();
    let veff_flat: Vec<f64> = veff_down.as_fine_array().iter().copied().collect();
    let veff_dev: CudaSlice<f64> = stream.clone_htod(&veff_flat).unwrap();

    // ψ
    let psi_flat: Vec<num_complex::Complex64> = kpt.bands.concat();
    let psi_cuda: Vec<CudaComplex> = psi_flat.iter()
        .map(|c| CudaComplex{x:c.re,y:c.im}).collect();
    let psi_dev: CudaSlice<CudaComplex> = stream.clone_htod(&psi_cuda).unwrap();
    let mut hpsi_dev = stream.alloc_zeros::<CudaComplex>(nb*npw).unwrap();
    let mut spsi_dev = stream.alloc_zeros::<CudaComplex>(nb*npw).unwrap();
    let mut grid_dev = stream.alloc_zeros::<CudaComplex>(nb*gs).unwrap();

    // Kinetic
    let kcpu = compute_kinetic_energies(&pwc, wg.recip_lattice(), kpt.coords);
    let kdev: CudaSlice<f64> = stream.clone_htod(&kcpu.0).unwrap();
    let kpre = KineticPreconditioner::new(kdev);

    // FFT
    let fft = BatchedFftPlan3d::plan_batched_c2c(
        ngx as i32, ngy as i32, ngz as i32, nb as i32, stream.clone()).unwrap();

    // V_NL
    let pots = PseudopotentialSet::from_dir(PDIR, &cell.species_symbols, &cell.species_pot_files).unwrap();
    let kp = KPoint{coords:kpt.coords};
    let mut pcie = PcieAccount::default();
    let vnl = VnlBatchData::precompute(
        &pwc, &pots, cell, &wg, &kp, &psi_flat, nb, npw, None,
        Some(&veff_inner), &stream, &mut pcie, &blas, &kernels, &solver,
    ).unwrap();

    // === H·ψ ===
    let psi = PwCoefficients::new(psi_dev);
    let mut hpsi = PwCoefficients::new(hpsi_dev);
    let mut spsi = PwCoefficients::new(spsi_dev);
    unsafe {
        apply_full_hamiltonian()
            .psi_dev(&psi).v_eff_dev(&veff_dev).kinetic_dev(&kpre)
            .fft_idx_dev(&fft_idx_dev).n_pw(npw).n_bands(nb).grid_size(gs)
            .inv_ntotal(invn).fft_plan(&fft).hpsi_dev(&mut hpsi).grid_dev(&mut grid_dev)
            .vnl_data(&vnl).blas(&blas).kernels(&kernels).stream(&stream).call().unwrap();
    }
    // === S·ψ ===
    stream.memcpy_dtod(&*psi, &mut *spsi).unwrap();
    unsafe {
        apply_s_times()
            .psi_dev(&psi).spsi_dev(&mut spsi).vnl_data(&vnl)
            .n_bands(nb as i32).n_pw(npw as i32).blas(&blas).stream(&stream).call().unwrap();
    }

    // D2H
    let hpsi_host: Vec<CudaComplex> = stream.clone_dtoh(&*hpsi).unwrap();
    let spsi_host: Vec<CudaComplex> = stream.clone_dtoh(&*spsi).unwrap();

    // Rayleigh quotients
    let mut max_err = 0.0f64;
    eprintln!("band   λ_our         λ_ref         Δλ");
    for b in 0..nb.min(20) {
        let (mut dh, mut ds) = (0.0f64, 0.0f64);
        for g in 0..npw {
            let p = psi_flat[b*npw+g];
            let hp = num_complex::Complex64::new(hpsi_host[b*npw+g].x, hpsi_host[b*npw+g].y);
            let sp = num_complex::Complex64::new(spsi_host[b*npw+g].x, spsi_host[b*npw+g].y);
            dh += (p.conj() * hp).re;
            ds += (p.conj() * sp).re;
        }
        let lam = dh / ds;
        let r = bands_ref[b];
        let e = (lam - r).abs();
        eprintln!("{b:4}  {lam:+.8e}  {r:+.8e}  {e:.4e}");
        if e > max_err { max_err = e; }
    }
    eprintln!("max |Δλ| = {max_err:.4e} Ha");
    assert!(max_err < 0.01, "Hamiltonian gives wrong Rayleigh quotients");
}
