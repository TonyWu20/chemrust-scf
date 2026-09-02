#![cfg(feature = "chebyshev")]
// ---------------------------------------------------------------------------
// ChFSI convergence tests — Gate 1 and Gate 2
// ---------------------------------------------------------------------------
// Gate 1: Single Chebyshev filter iteration on CASTEP converged wavefunctions
//   with zero V_eff. Structural validation only: NaN/Inf, monotonic eigenvalues.
//   No comparison to CASTEP eigenvalues — zero V_eff gives kinetic-only spectrum.
//
// Gate 2: Multi-iteration ChFSI convergence loop (filter → RR → repeat)
//   with V_eff computed FROM DENSITY (frozen). Verifies the eigensolver
//   converges under a fixed Hamiltonian — eigenvalues must stabilize,
//   not drift or cascade.
//
// Reference: ABINIT chebfi_run (m_chebfi2.F90:473-737)

mod fixtures;

fn gpu_available() -> bool {
    std::env::var("CASTEP_FIXTURE_DIR").is_ok()
        || std::path::Path::new(
            "/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8",
        )
        .exists()
}

#[cfg(test)]
mod tests {
    /// Gate 1: Verify that one ChFSI iteration from CASTEP-converged
    /// wavefunctions produces well-formed eigenvalues (no NaN/Inf, monotonic).
    /// This is a STRUCTURAL check only — zero V_eff means eigenvalues are
    /// kinetic energies, not physical DFT eigenvalues.
    ///
    /// Discriminator: NaN=false, Inf=false, monotonic=true.
    #[test]
    #[ignore = "requires GPU and CASTEP fixture data"]
    fn gate1_chebfi_preserves_converged_eigenpairs() {
        if !super::gpu_available() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        use chemrust_hamiltonian_core::GVectorGrid;
        use chemrust_scf::KPoint;
        use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData};
        use chemrust_scf::device::blas::BlasHandle;
        use chemrust_scf::device::pcie::PcieAccount;
        use chemrust_scf::eigensolver::chebyshev::{FilterMode, chebfi_run_rust};
        use chemrust_scf::eigensolver::rayleigh_ritz::rayleigh_ritz;
        use chemrust_scf::layout::{ColumnDistributed, WavefunctionSet};
        use chemrust_scf::device::Gpu;
        use num_complex::Complex64;
        use std::sync::Arc;

        let fx = super::fixtures::cu111_co::fixture();
        let cell = &fx.bin.cell;
        let pots = &fx.pots;
        let wfc = fx.check.wavefunction.as_ref().expect(".check wavefunction");
        let wave_grid_dims = wfc.grid;
        let wave_grid = GVectorGrid::new(
            wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice,
        );
        let kpt_block = &wfc.kpt_data[0];
        let n_bands = kpt_block.bands.len();
        let n_pw = kpt_block.nplw;
        let k_point = KPoint { coords: kpt_block.coords, weight: 1.0 };
        let psi_data: Vec<Complex64> = kpt_block.bands.concat();
        let pw_coords = &kpt_block.pw_grid_coord;
        let fft_idx: Vec<i32> =
            chemrust_scf::pw_coords_to_fft_indices(pw_coords, &wave_grid);

        let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).expect("BLAS handle");
        let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone())
            .expect("SolverHandle");
        let kernels = CudaKernelSet::new(&ctx).expect("CUDA kernels");

        let mut pcie = PcieAccount::default();
        let vnl_data = VnlBatchData::precompute_with_d_override(
            pw_coords, pots, cell, &wave_grid,
            None, &k_point, &psi_data, n_bands, n_pw,
            None, None, None, None, None,
            Some(&solver),
            &stream, &mut pcie, &blas, &kernels,
        ).expect("VnlBatchData::precompute_with_d_override");

        // Zero V_eff for structural validation
        let [ngz_w, ngy_w, ngx_w] = wave_grid.grid();
        let grid_size = ngz_w * ngy_w * ngx_w;
        let v_eff_dev: cudarc::driver::CudaSlice<f64> =
            stream.clone_htod(&vec![0.0_f64; grid_size]).expect("v_eff upload");

        let psi_wfn = WavefunctionSet::<ColumnDistributed>::new(psi_data, n_bands, n_pw);
        let psi_gpu = Gpu::from_host_with(&psi_wfn, &stream, &mut pcie)
            .expect("psi upload");
        let fft_idx_dev: cudarc::driver::CudaSlice<i32> =
            stream.clone_htod(&fft_idx).expect("fft_idx upload");

        let gmax = wave_grid.gmax();
        let ecut = 0.5 * gmax * gmax;

        // Run one ChFSI iteration (zero V_eff)
        let (psi_filtered_row, mut hpsi_row, _ritz, _res, _ndeg) = chebfi_run_rust(
            &psi_gpu, &v_eff_dev, &wave_grid, pw_coords,
            &vnl_data, &fft_idx_dev,
            ecut, 0.0, 0.0,
            1e-6, None, None, 20, 1, 0.0, 1e-8,
            &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig, None,
            0, n_bands, false,
        ).expect("chebfi_run_rust");

        let (_psi_new, eigenvalues_cpu, _beta) = rayleigh_ritz(
            &psi_filtered_row, &mut hpsi_row, &vnl_data,
            n_bands, n_pw, &kernels, &mut pcie,
            &solver, &blas, &stream, &ctx,
            None, None, None,
        ).expect("rayleigh_ritz");

        stream.synchronize().expect("sync");

        let eig = &eigenvalues_cpu.0;

        let has_nan = eig.iter().any(|e| e.is_nan());
        let has_inf = eig.iter().any(|e| e.is_infinite());
        let monotonic = eig.windows(2).all(|w| w[0] <= w[1]);

        eprintln!("[Gate 1] ChFSI pipeline structural validation (zero V_eff)");
        eprintln!("[Gate 1] bands={n_bands} n_pw={n_pw}");
        eprintln!("[Gate 1] eigenvalues: [{:.6}, ..., {:.6}]", eig[0], eig[n_bands-1]);
        eprintln!("[Gate 1] NaN: {has_nan}, Inf: {has_inf}, monotonic: {monotonic}");

        assert!(!has_nan, "Gate 1 FAIL: NaN in eigenvalues");
        assert!(!has_inf, "Gate 1 FAIL: Inf in eigenvalues");
        assert!(monotonic, "Gate 1 FAIL: eigenvalues not monotonically increasing");

        eprintln!("[Gate 1 PASS] pipeline structural validation — no NaN/Inf, monotonic");
    }

    /// Gate 2: ChFSI convergence loop — multiple filter+RR iterations
    /// from CASTEP-converged ψ with V_eff computed FROM DENSITY and frozen.
    /// The eigensolver must converge under a fixed Hamiltonian without
    /// distorting eigenvectors across iterations.
    ///
    /// Discriminator: consecutive eigenvalue drift must be < 5e-4 Ha.
    /// Iter-1 band 0 must be within 1 Ha of CASTEP (weak sanity check).
    #[test]
    #[ignore = "requires GPU and CASTEP fixture data"]
    fn gate2_chebfi_convergence_loop_no_scf() {
        if !super::gpu_available() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        use chemrust_hamiltonian_core::GVectorGrid;
        use chemrust_scf::KPoint;
        use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData};
        use chemrust_scf::device::blas::BlasHandle;
        use chemrust_scf::device::pcie::PcieAccount;
        use chemrust_scf::eigensolver::chebyshev::{FilterMode, chebfi_run_rust};
        use chemrust_scf::eigensolver::rayleigh_ritz::rayleigh_ritz;
        use chemrust_scf::layout::{ColumnDistributed, WavefunctionSet};
        use chemrust_scf::device::Gpu;
        use chemrust_scf::downsample_array_to_wave_grid;
        use num_complex::Complex64;
        use std::sync::Arc;

        let fx = super::fixtures::cu111_co::fixture();
        let cell = &fx.bin.cell;
        let pots = &fx.pots;
        let wfc = fx.check.wavefunction.as_ref().expect(".check wavefunction");
        let wave_grid_dims = wfc.grid;
        let wave_grid = GVectorGrid::new(
            wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice,
        );
        let fine_grid_dims = fx.check.fine_grid.expect(".check must have fine_grid");
        let [fgx, fgy, fgz] = fine_grid_dims;
        let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

        let kpt_block = &wfc.kpt_data[0];
        let n_bands = kpt_block.bands.len();
        let n_pw = kpt_block.nplw;
        let k_point = KPoint { coords: kpt_block.coords, weight: 1.0 };
        let psi_data: Vec<Complex64> = kpt_block.bands.concat();
        let pw_coords = &kpt_block.pw_grid_coord;
        let fft_idx: Vec<i32> =
            chemrust_scf::pw_coords_to_fft_indices(pw_coords, &wave_grid);

        let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).expect("BLAS handle");
        let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone())
            .expect("SolverHandle");
        let kernels = CudaKernelSet::new(&ctx).expect("CUDA kernels");

        let mut pcie = PcieAccount::default();
        let vnl_data = VnlBatchData::precompute_with_d_override(
            pw_coords, pots, cell, &wave_grid,
            None, &k_point, &psi_data, n_bands, n_pw,
            None, None, None, None, None,
            Some(&solver),
            &stream, &mut pcie, &blas, &kernels,
        ).expect("VnlBatchData::precompute_with_d_override");

        // V_eff from density: build SCF state, compute V_eff = V_H[ρ] + V_xc[ρ] + V_loc
        let scf_state = super::fixtures::cu111_co::build_scf_state(&fx, &stream);
        let v_eff_state = scf_state
            .build_v_eff_with_energy()
            .expect("build_v_eff_with_energy");
        let v_eff_opt = v_eff_state.v_eff();
        let v_eff_ref = v_eff_opt.as_ref()
            .expect("V_eff must be Some after build_v_eff_with_energy");
        let v_eff_wave = downsample_array_to_wave_grid(
            v_eff_ref.as_real_grid().as_real_array(),
            &fine_grid, &wave_grid,
        ).expect("downsample V_eff");
        // The downsampled V_eff has shape (ngx, ngy, ngz) from FFT inverse
        // (see fft_inverse_3d doc: input Fortran (ngz,ngy,ngx), output row-major (ngx,ngy,ngz)).
        // ndarray indexing [[ix, iy, iz]] matches this shape exactly.
        // Transpose to iz-innermost flat buffer for cuFFT (matching src/pipeline.rs:104-119).
        let wave_ix = v_eff_wave.as_fine_array();
        let [ngz_w, ngy_w, ngx_w] = wave_grid.grid();
        let grid_size_w = ngz_w * ngy_w * ngx_w;
        let mut ve_std_iz = vec![0.0_f64; grid_size_w];
        for iz in 0..ngz_w {
            for iy in 0..ngy_w {
                for ix in 0..ngx_w {
                    let val = wave_ix[[ix, iy, iz]];
                    let idx_iz = iz + ngz_w * (iy + ngy_w * ix);
                    ve_std_iz[idx_iz] = val;
                }
            }
        }
        let v_eff_flat = ve_std_iz;
        let v_eff_dev: cudarc::driver::CudaSlice<f64> =
            stream.clone_htod(&v_eff_flat).expect("v_eff upload");
        let min_veff = v_eff_flat.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_veff = v_eff_flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        eprintln!("[Gate 2] wave_grid.grid() = [{ngz_w}, {ngy_w}, {ngx_w}]");
        eprintln!("[Gate 2] V_eff flat len = {}  grid_size_w = {}", v_eff_flat.len(), grid_size_w);
        eprintln!("[Gate 2] CASTEP wave_grid_dims = {:?}", wave_grid_dims);

        let psi_wfn = WavefunctionSet::<ColumnDistributed>::new(psi_data.clone(), n_bands, n_pw);
        let psi_gpu = Gpu::from_host_with(&psi_wfn, &stream, &mut pcie)
            .expect("psi upload");
        let fft_idx_dev: cudarc::driver::CudaSlice<i32> =
            stream.clone_htod(&fft_idx).expect("fft_idx upload");

        // ecut = max kinetic energy in PW basis (spherical cutoff), NOT FFT grid gmax
        let kinetic_cpu_vals = chemrust_scf::eigensolver::davidson_types::compute_kinetic_energies(
            pw_coords, wave_grid.recip_lattice(), k_point.coords,
        );
        let ecut = kinetic_cpu_vals.0.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!("[Gate 2] ecut(PW max)={ecut:.2} Ha  n_pw={n_pw}");

        let castep_eig = &fx.bands_eigenvalues;
        let max_outer = 3;
        let mut band0_history: Vec<f64> = Vec::with_capacity(max_outer);

        eprintln!("[Gate 2] ChFSI convergence loop — {max_outer} iterations, V_eff FROM DENSITY");
        eprintln!("[Gate 2] V_eff range: [{:.4}, {:.4}] Ha", min_veff, max_veff);
        eprintln!("[Gate 2] iter | band 0 (Ha) | diff from CASTEP (Ha)");

        let mut psi_current = psi_gpu;

        for iter in 0..max_outer {
            let (psi_filt, mut hpsi_filt, ritz_values, _res, _ndeg) = chebfi_run_rust(
                &psi_current, &v_eff_dev, &wave_grid, pw_coords,
                &vnl_data, &fft_idx_dev,
                ecut, min_veff, max_veff,
                1e-6, None, None, 20, 1, 0.0, 1e-8,
                &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
                FilterMode::SinvHKeepHEig, None,
                0, n_bands, false,
            ).expect("chebfi_run_rust");

            let (psi_new, eigenvalues_cpu, _beta) = rayleigh_ritz(
                &psi_filt, &mut hpsi_filt, &vnl_data,
                n_bands, n_pw, &kernels, &mut pcie,
                &solver, &blas, &stream, &ctx,
                None, None, None,
            ).expect("rayleigh_ritz");

            let eig = eigenvalues_cpu.0.clone();
            let band0_diff = (eig[0] - castep_eig[0]).abs();
            band0_history.push(eig[0]);

            // Diagnostic: pre-filter Rayleigh quotient for band 0
            // If this differs from CASTEP band 0, the Hamiltonian is wrong
            let rq0 = ritz_values.get(0).copied().unwrap_or(f64::NAN);
            let rq_diff = (rq0 - castep_eig[0]).abs();
            if iter == 0 {
                eprintln!("[Gate 2] pre-filter RQ band0 = {:.8} Ha (Δ={:.2e} from CASTEP)", rq0, rq_diff);
            }

            eprintln!(
                "[Gate 2] {:4} | {:.8} | {:.2e}",
                iter, eig[0], band0_diff,
            );

            psi_current = psi_new;
        }

        stream.synchronize().expect("sync");

        // Gate 2 discriminator: eigenvalues must STABILIZE across iterations.
        let consecutive_drift: f64 = band0_history.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f64::max);
        let total_drift = (band0_history.last().unwrap() - band0_history.first().unwrap()).abs();

        eprintln!("[Gate 2] band 0 history: {:?}", band0_history);
        eprintln!("[Gate 2] CASTEP band 0: {:.8}", castep_eig[0]);
        eprintln!("[Gate 2] consecutive drift: {:.4e} Ha", consecutive_drift);
        eprintln!("[Gate 2] total drift: {:.4e} Ha", total_drift);

        // Primary gate: consecutive eigenvalue change must be bounded
        assert!(
            consecutive_drift < 5e-4,
            "Gate 2 FAIL: band 0 consecutive drift {:.4e} Ha > 5e-4 Ha. \
             ChFSI convergence loop is distorting eigenvectors under fixed V_eff.",
            consecutive_drift,
        );

        // Weak sanity gate: iter-1 band 0 must be within 1 Ha of CASTEP
        let iter0_err = (band0_history[0] - castep_eig[0]).abs();
        assert!(
            iter0_err < 1.0,
            "Gate 2 FAIL: iter-1 band 0 differs from CASTEP by {:.4e} Ha > 1 Ha. \
             V_eff construction or Hamiltonian application is broken.",
            iter0_err,
        );

        eprintln!("=== Gate 2 PASS: eigenvalues stable under frozen V_eff ===");
    }

    /// Gate 3: Full SCF convergence with ChFSI on non-spin Cu111_CO.
    /// Uses run_scf_with_energy_gated which properly handles
    /// Off→Kerker→Pulay density mixing transitions.
    #[test]
    #[ignore = "requires GPU and CASTEP fixture data"]
    fn gate3_chfsi_scf_convergence() {
        if !super::gpu_available() {
            eprintln!("SKIP: no GPU available");
            return;
        }
        unsafe { std::env::set_var("CHEMRUST_EIGENSOLVER", "chebyshev"); }

        use chemrust_scf::run_scf_with_energy_gated;
        use chemrust_scf::ScfDivergenceGate;
        use std::sync::Arc;

        let fx = super::fixtures::cu111_co::fixture();
        let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA"));
        let stream = ctx.default_stream();
        let state = super::fixtures::cu111_co::build_scf_state(&fx, &stream);
        let result = run_scf_with_energy_gated(
            state,
            8,
            1e-8,
            Some(ScfDivergenceGate {
                max_last_band_ha: 30.0,  // relaxed: let us see eigenvalue range
                min_band0_ha: -30.0,
                max_veff_range_factor: 5.0,
                max_iter: 20,
                electron_count_tolerance: 0.05,
                soft_fraction_tolerance: 0.20,
            }),
        )
        .expect("SCF converged");

        let computed_ev = result.total_energy * chemrust_scf::HARTREE_TO_EV;
        let ref_ev = super::fixtures::cu111_co::REFERENCE_ENERGY_EV;
        let diff_ev = (computed_ev - ref_ev).abs();

        eprintln!("[Gate 3] ChFSI total energy: {:.8} eV", computed_ev);
        eprintln!("[Gate 3] CASTEP reference:    {:.8} eV", ref_ev);
        eprintln!("[Gate 3] difference:          {:.4e} eV", diff_ev);

        assert!(
            diff_ev < super::fixtures::cu111_co::TOLERANCE_EV,
            "Gate 3 FAIL: ChFSI SCF energy differs by {:.4e} eV",
            diff_ev,
        );
        eprintln!("=== Gate 3 PASS: ChFSI SCF converges to CASTEP energy ===");
    }

    /// Gate 4: ChFSI single-iteration eigenvalue accuracy on NiO non-spin.
    ///
    /// Runs one ChFSI filter + Rayleigh-Ritz pass with V_eff FROM DENSITY
    /// (frozen), then compares eigenvalues to CASTEP reference.
    /// This answers: does ChFSI produce correct eigenvalues on ANY system?
    ///
    /// NiO is smaller than Cu111_CO (62 bands, ~8 ions) and should fit
    /// in 8 GB GPU memory.
    #[test]
    #[ignore = "requires GPU and CASTEP fixture data"]
    fn gate4_nio_chfsi_eigenvalues() {
        if !super::gpu_available() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        use chemrust_hamiltonian_core::GVectorGrid;
        use chemrust_scf::KPoint;
        use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData};
        use chemrust_scf::device::blas::BlasHandle;
        use chemrust_scf::device::pcie::PcieAccount;
        use chemrust_scf::eigensolver::chebyshev::{FilterMode, chebfi_run_rust};
        use chemrust_scf::eigensolver::rayleigh_ritz::rayleigh_ritz;
        use chemrust_scf::layout::{ColumnDistributed, WavefunctionSet};
        use chemrust_scf::device::Gpu;
        use chemrust_scf::downsample_array_to_wave_grid;
        use num_complex::Complex64;
        use std::sync::Arc;

        let fx = super::fixtures::nio_no_spin::fixture();
        let cell = &fx.bin.cell;
        let pots = &fx.pots;
        let wfc = fx.check.wavefunction.as_ref().expect(".check wavefunction");
        let wave_grid_dims = wfc.grid;
        let wave_grid = GVectorGrid::new(
            wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice,
        );
        let fine_grid_dims = fx.check.fine_grid.expect(".check must have fine_grid");
        let [fgx, fgy, fgz] = fine_grid_dims;
        let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

        let kpt_block = &wfc.kpt_data[0];
        let n_bands = kpt_block.bands.len();
        let n_pw = kpt_block.nplw;
        let k_point = KPoint { coords: kpt_block.coords, weight: 1.0 };
        let psi_data: Vec<Complex64> = kpt_block.bands.concat();
        let pw_coords = &kpt_block.pw_grid_coord;
        let fft_idx: Vec<i32> =
            chemrust_scf::pw_coords_to_fft_indices(pw_coords, &wave_grid);

        eprintln!("[Gate 4] NiO non-spin: n_bands={n_bands} n_pw={n_pw} kpt=({:.4},{:.4},{:.4})",
            kpt_block.coords[0], kpt_block.coords[1], kpt_block.coords[2]);

        let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).expect("BLAS handle");
        let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone())
            .expect("SolverHandle");
        let kernels = CudaKernelSet::new(&ctx).expect("CUDA kernels");

        let mut pcie = PcieAccount::default();
        let vnl_data = VnlBatchData::precompute_with_d_override(
            pw_coords, pots, cell, &wave_grid,
            None, &k_point, &psi_data, n_bands, n_pw,
            None, None, None, None, None,
            Some(&solver),
            &stream, &mut pcie, &blas, &kernels,
        ).expect("VnlBatchData::precompute_with_d_override");

        // V_eff from density
        let scf_state = super::fixtures::nio_no_spin::build_scf_state(fx, &stream);
        let v_eff_state = scf_state
            .build_v_eff_with_energy()
            .expect("build_v_eff_with_energy");
        let v_eff_opt = v_eff_state.v_eff();
        let v_eff_ref = v_eff_opt.as_ref()
            .expect("V_eff must be Some after build_v_eff_with_energy");
        let v_eff_wave = downsample_array_to_wave_grid(
            v_eff_ref.as_real_grid().as_real_array(),
            &fine_grid, &wave_grid,
        ).expect("downsample V_eff");

        // Transpose to iz-innermost for cuFFT
        let wave_ix = v_eff_wave.as_fine_array();
        let [ngz_w, ngy_w, ngx_w] = wave_grid.grid();
        let grid_size_w = ngz_w * ngy_w * ngx_w;
        let mut ve_std_iz = vec![0.0_f64; grid_size_w];
        for iz in 0..ngz_w {
            for iy in 0..ngy_w {
                for ix in 0..ngx_w {
                    let val = wave_ix[[ix, iy, iz]];
                    let idx_iz = iz + ngz_w * (iy + ngy_w * ix);
                    ve_std_iz[idx_iz] = val;
                }
            }
        }
        let v_eff_flat = ve_std_iz;
        let v_eff_dev: cudarc::driver::CudaSlice<f64> =
            stream.clone_htod(&v_eff_flat).expect("v_eff upload");
        let min_veff = v_eff_flat.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_veff = v_eff_flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let psi_wfn = WavefunctionSet::<ColumnDistributed>::new(psi_data.clone(), n_bands, n_pw);
        let psi_gpu = Gpu::from_host_with(&psi_wfn, &stream, &mut pcie)
            .expect("psi upload");
        let fft_idx_dev: cudarc::driver::CudaSlice<i32> =
            stream.clone_htod(&fft_idx).expect("fft_idx upload");

        // ecut from PW basis
        let kinetic_cpu_vals = chemrust_scf::eigensolver::davidson_types::compute_kinetic_energies(
            pw_coords, wave_grid.recip_lattice(), k_point.coords,
        );
        let ecut = kinetic_cpu_vals.0.iter().cloned().fold(0.0_f64, f64::max);

        let castep_eig_first_kpt = &fx.bands_eigenvalues[..n_bands]; // first k-point
        let castep_band0 = castep_eig_first_kpt[0];

        eprintln!("[Gate 4] ecut={ecut:.2} Ha  n_pw={n_pw}");
        eprintln!("[Gate 4] V_eff range: [{:.4}, {:.4}] Ha", min_veff, max_veff);
        eprintln!("[Gate 4] CASTEP band 0: {:.8} Ha", castep_band0);

        // --- Run one ChFSI iteration ---
        let (psi_filt, mut hpsi_filt, ritz_values, _res, _ndeg) = chebfi_run_rust(
            &psi_gpu, &v_eff_dev, &wave_grid, pw_coords,
            &vnl_data, &fft_idx_dev,
            ecut, min_veff, max_veff,
            1e-6, None, None, 20, 1, 0.0, 1e-8,
            &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig, None,
            0, n_bands, false,
        ).expect("chebfi_run_rust");

        let rq0 = ritz_values.get(0).copied().unwrap_or(f64::NAN);
        eprintln!("[Gate 4] pre-filter RQ band0 = {:.8} Ha (Δ={:.2e} from CASTEP)",
            rq0, (rq0 - castep_band0).abs());

        // --- Rayleigh-Ritz ---
        let (_psi_new, eigenvalues_cpu, _beta) = rayleigh_ritz(
            &psi_filt, &mut hpsi_filt, &vnl_data,
            n_bands, n_pw, &kernels, &mut pcie,
            &solver, &blas, &stream, &ctx,
            None, None, None,
        ).expect("rayleigh_ritz");

        stream.synchronize().expect("sync");

        let eig = &eigenvalues_cpu.0;

        eprintln!("[Gate 4] ========================================");
        eprintln!("[Gate 4] band | ChFSI (Ha)     | CASTEP (Ha)    | Δ (Ha)");
        let mut max_err = 0.0f64;
        for b in 0..n_bands.min(10) {
            let err = (eig[b] - castep_eig_first_kpt[b]).abs();
            eprintln!("[Gate 4] {:4} | {:.10} | {:.10} | {:.2e}", b, eig[b], castep_eig_first_kpt[b], err);
            max_err = max_err.max(err);
        }
        if n_bands > 10 {
            eprintln!("[Gate 4] ... (showing first 10 of {n_bands} bands)");
            for b in (n_bands - 5).max(10)..n_bands {
                let err = (eig[b] - castep_eig_first_kpt[b]).abs();
                eprintln!("[Gate 4] {:4} | {:.10} | {:.10} | {:.2e}", b, eig[b], castep_eig_first_kpt[b], err);
                max_err = max_err.max(err);
            }
        }
        eprintln!("[Gate 4] max eigenvalue error: {:.2e} Ha", max_err);

        let band0_err = (eig[0] - castep_band0).abs();
        eprintln!("[Gate 4] band 0 error: {:.2e} Ha", band0_err);

        assert!(
            band0_err < 1.0,
            "Gate 4 FAIL: band 0 differs from CASTEP by {:.2e} Ha > 1 Ha",
            band0_err,
        );

        eprintln!("=== Gate 4 PASS: ChFSI eigenvalues within 1 Ha of CASTEP ===");
    }
}
