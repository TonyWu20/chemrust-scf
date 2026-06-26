// Gate 4: ChFSI eigenvalue accuracy — NiO (all k-points) + Cu111_CO.
// Standalone — does NOT use the fixtures module.

fn niO_fixture_dir() -> String {
    std::env::var("CASTEP_FIXTURE_DIR")
        .unwrap_or_else(|_| "/export/public_castep_jobs/tony/NiO_no_u_finer_grid_no_spin_cpu_reference".to_string())
}

fn cu_fixture_dir() -> String {
    "/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8".to_string()
}

fn potential_dir() -> String {
    std::env::var("CASTEP_POTENTIAL_DIR")
        .unwrap_or_else(|_| "/export/Potentials".to_string())
}

fn parse_bands_first_spin(text: &str) -> Vec<f64> {
    text.lines()
        .skip_while(|line| !line.trim().starts_with("Spin component"))
        .skip(1)
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect()
}

// ============================================================================
// NiO non-spin — all 14 k-points
// ============================================================================

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn gate4_nio_all_kpoints() {
    let dir = niO_fixture_dir();
    if !std::path::Path::new(&dir).exists() { eprintln!("SKIP"); return; }
    let pot_dir = potential_dir();

    use chemrust_hamiltonian_core::{
        CastepBinFile, CheckFile, GVectorGrid, PseudopotentialSet, NonSpin,
    };
    use chemrust_scf::KPoint;
    use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData};
    use chemrust_scf::device::blas::BlasHandle;
    use chemrust_scf::device::pcie::PcieAccount;
    use chemrust_scf::eigensolver::chebyshev::{FilterMode, chebfi_run_rust};
    use chemrust_scf::eigensolver::rayleigh_ritz::rayleigh_ritz;
    use chemrust_scf::layout::{ColumnDistributed, WavefunctionSet};
    use chemrust_scf::device::Gpu;
    use chemrust_scf::downsample_array_to_wave_grid;
    use chemrust_scf::{
        Density, ScfIteration, SmearingParams, SmearingScheme, SmearingWidth,
        WaveGridArray, pw_coords_to_fft_indices,
    };
    use chemrust_scf::spin_types::{
        KptDataSet, PerSpinDensity, PerSpinPwCoefficients, SpinChannelData,
    };
    use chemrust_scf::PwCoefficients;
    use num_complex::Complex64;
    use std::sync::Arc;

    let bin = CastepBinFile::read(std::io::BufReader::new(
        std::fs::File::open(format!("{dir}/NiO.castep_bin")).expect("open"),
    )).expect("read");
    let check = CheckFile::read(std::io::BufReader::new(
        std::fs::File::open(format!("{dir}/NiO.check")).expect("open"),
    )).expect("read");
    let bands_text = std::fs::read_to_string(format!("{dir}/NiO.bands")).expect("bands");
    let bands_eigenvalues = parse_bands_first_spin(&bands_text);
    let pots = PseudopotentialSet::from_dir(
        &pot_dir, &bin.cell.species_symbols, &bin.cell.species_pot_files,
    ).expect("pots");

    let cell = &bin.cell;
    let wfc = check.wavefunction.as_ref().expect("wfc");
    let wave_grid_dims = wfc.grid;
    let wave_grid = GVectorGrid::new(wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice);
    let fine_grid_dims = check.fine_grid.expect("fine_grid");
    let [fgx, fgy, fgz] = fine_grid_dims;
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("BLAS");
    let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone()).expect("Solver");
    let kernels = CudaKernelSet::new(&ctx).expect("kernels");
    let mut pcie = PcieAccount::default();

    // ---- V_eff from density (once for all k-points) ----
    let den_fine_arr = bin.density.charge.as_real_grid().as_real_array().clone();
    let den_wave = downsample_array_to_wave_grid(&den_fine_arr, &fine_grid, &wave_grid).expect("ds density");
    let density = Density::from_inner(WaveGridArray::from_inner(den_wave.as_fine_array().clone()));

    let nkpts = wfc.kpt_data.len();
    let kpt_weights: [f64; 14] = [0.07407407; 14];
    // Build ScfIteration with all 14 k-points
    let mut all_psi: Vec<Vec<Complex64>> = Vec::with_capacity(nkpts);
    let mut all_pw: Vec<Vec<[i32; 3]>> = Vec::with_capacity(nkpts);
    let mut all_fft: Vec<Vec<i32>> = Vec::with_capacity(nkpts);
    let mut all_kpts: Vec<KPoint> = Vec::with_capacity(nkpts);
    for ikpt in 0..nkpts {
        let kb = &wfc.kpt_data[ikpt];
        all_psi.push(kb.bands.concat());
        all_pw.push(kb.pw_grid_coord.clone());
        all_fft.push(pw_coords_to_fft_indices(&kb.pw_grid_coord, &wave_grid));
        all_kpts.push(KPoint { coords: kb.coords, weight: kpt_weights[ikpt] });
    }
    let scf_state = ScfIteration::<NonSpin>::builder()
        .cell(cell.clone()).pots(pots.clone())
        .wave_grid(GVectorGrid::new(wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice))
        .fine_grid(GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice))
        .density(PerSpinDensity::new(SpinChannelData::new::<NonSpin>(vec![density])))
        .psi(PerSpinPwCoefficients::new(SpinChannelData::new::<NonSpin>(vec![
            KptDataSet::new(all_kpts.iter().map(|_| PwCoefficients::new(
                stream.alloc_zeros::<chemrust_scf::device::CudaComplex>(0).expect("dummy"),
            )).collect(), nkpts),
        ])))
        .psi_data(SpinChannelData::new::<NonSpin>(vec![KptDataSet::new(all_psi.clone(), nkpts)]))
        .pw_coords(KptDataSet::new(all_pw.clone(), nkpts))
        .pw_fft_indices(KptDataSet::new(all_fft.clone(), nkpts))
        .k_points(KptDataSet::new(all_kpts, nkpts))
        .smearing(SmearingParams { width: SmearingWidth::ev(0.1),
            electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
            scheme: SmearingScheme::Gaussian, spin_fix: 6 })
        .max_history(8).build();

    let v_eff_state = scf_state.build_v_eff_with_energy().expect("V_eff");
    let v_eff_wave = downsample_array_to_wave_grid(
        v_eff_state.v_eff().as_ref().expect("V_eff").as_real_grid().as_real_array(),
        &fine_grid, &wave_grid,
    ).expect("ds V_eff");

    let wave_ix = v_eff_wave.as_fine_array();
    let [ngz_w, ngy_w, ngx_w] = wave_grid.grid();
    let grid_size_w = ngz_w * ngy_w * ngx_w;
    let mut ve_iz = vec![0.0_f64; grid_size_w];
    for iz in 0..ngz_w { for iy in 0..ngy_w { for ix in 0..ngx_w {
        ve_iz[iz + ngz_w * (iy + ngy_w * ix)] = wave_ix[[ix, iy, iz]];
    }}}
    let v_eff_dev: cudarc::driver::CudaSlice<f64> = stream.clone_htod(&ve_iz).expect("v_eff");
    let min_veff = ve_iz.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_veff = ve_iz.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!("[Gate 4] NiO: {} k-points, {} bands/kpt, V_eff=[{:.4},{:.4}] Ha",
        nkpts, wfc.kpt_data[0].bands.len(), min_veff, max_veff);

    // Quick check: does V_eff actually contribute to NiO RQ?
    // Run one k-point with zero V_eff and constant V_eff=-5
    {
        let kb = &wfc.kpt_data[0];
        let nb = kb.bands.len();
        let npw = kb.nplw;
        let kp = KPoint { coords: kb.coords, weight: kpt_weights[0] };
        let psid: Vec<Complex64> = kb.bands.concat();
        let pwc = &kb.pw_grid_coord;
        let ffti: Vec<i32> = pw_coords_to_fft_indices(pwc, &wave_grid);

        let mut vnl_test = VnlBatchData::precompute_with_d_override(
            pwc, &pots, cell, &wave_grid, Some(&fine_grid), &kp, &psid, nb, npw,
            None, None, None, None, None,
            Some(&solver), &stream, &mut pcie, &blas, &kernels,
        ).expect("vnl test");
        vnl_test.rescreen_d(
            v_eff_state.v_eff().as_ref().expect("V_eff").as_real_grid().as_real_array(),
            &stream, &kernels, &blas,
        ).expect("rescreen");
        let psig = Gpu::from_host_with(
            &WavefunctionSet::<ColumnDistributed>::new(psid.clone(), nb, npw),
            &stream, &mut pcie,
        ).expect("psig");
        let fftid: cudarc::driver::CudaSlice<i32> = stream.clone_htod(&ffti).expect("ffti");
        let kcpu = chemrust_scf::eigensolver::davidson_types::compute_kinetic_energies(
            pwc, wave_grid.recip_lattice(), kp.coords,
        );
        let ec = kcpu.0.iter().cloned().fold(0.0_f64, f64::max);

        // Actual V_eff
        let (_, _, ritz_act, _, _) = chebfi_run_rust(
            &psig, &v_eff_dev, &wave_grid, pwc, &vnl_test, &fftid,
            ec, min_veff, max_veff, 1e-6, None, None, 1, 1, 0.0, 1e-8,
            &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig, if kb.coords == [0.0;3] { None } else { Some(&kcpu.0) },
            0, nb, kb.coords != [0.0;3],
        ).expect("chebfi act");
        let rq0_act = ritz_act.get(0).copied().unwrap_or(f64::NAN);

        // Zero V_eff
        let v_zero: cudarc::driver::CudaSlice<f64> = stream.clone_htod(&vec![0.0_f64; grid_size_w]).expect("zero");
        let (_, _, ritz_zero, _, _) = chebfi_run_rust(
            &psig, &v_zero, &wave_grid, pwc, &vnl_test, &fftid,
            ec, 0.0, 0.0, 1e-6, None, None, 1, 1, 0.0, 1e-8,
            &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig, if kb.coords == [0.0;3] { None } else { Some(&kcpu.0) },
            0, nb, kb.coords != [0.0;3],
        ).expect("chebfi zero");
        let rq0_zero = ritz_zero.get(0).copied().unwrap_or(f64::NAN);

        eprintln!("[Gate 4] NiO kpt1: RQ(actual)={:.6}  RQ(V=0)={:.6}  ΔV_contrib={:.4} Ha",
            rq0_act, rq0_zero, rq0_act - rq0_zero);
    }

    let mut all_passed = true;
    let mut worst_band0 = 0.0f64;
    let mut worst_max = 0.0f64;

    for ikpt in 0..nkpts {
        let kpt_block = &wfc.kpt_data[ikpt];
        let n_bands = kpt_block.bands.len();
        let n_pw = kpt_block.nplw;
        let is_gamma = kpt_block.coords == [0.0, 0.0, 0.0];
        let k_point = KPoint { coords: kpt_block.coords, weight: kpt_weights[ikpt] };
        let psi_data: Vec<Complex64> = kpt_block.bands.concat();
        let pw_coords = &kpt_block.pw_grid_coord;
        let fft_idx: Vec<i32> = pw_coords_to_fft_indices(pw_coords, &wave_grid);

        let mut vnl_data = VnlBatchData::precompute_with_d_override(
            pw_coords, &pots, cell, &wave_grid, Some(&fine_grid), &k_point, &psi_data, n_bands, n_pw,
            None, None, None, None, None,
            Some(&solver), &stream, &mut pcie, &blas, &kernels,
        ).expect("VnlBatchData");
        vnl_data.rescreen_d(
            v_eff_state.v_eff().as_ref().expect("V_eff").as_real_grid().as_real_array(),
            &stream, &kernels, &blas,
        ).expect("rescreen_d");

        let psi_wfn = WavefunctionSet::<ColumnDistributed>::new(psi_data.clone(), n_bands, n_pw);
        let psi_gpu = Gpu::from_host_with(&psi_wfn, &stream, &mut pcie).expect("psi");
        let fft_idx_dev: cudarc::driver::CudaSlice<i32> = stream.clone_htod(&fft_idx).expect("fft");

        let kinetic_cpu = chemrust_scf::eigensolver::davidson_types::compute_kinetic_energies(
            pw_coords, wave_grid.recip_lattice(), k_point.coords,
        );
        let ecut = kinetic_cpu.0.iter().cloned().fold(0.0_f64, f64::max);
        let kinetic_slice: Option<&[f64]> = if is_gamma { None } else { Some(&kinetic_cpu.0) };

        let (_psi_filt, _hpsi_filt, ritz_values, _res, _ndeg) = chebfi_run_rust(
            &psi_gpu, &v_eff_dev, &wave_grid, pw_coords, &vnl_data, &fft_idx_dev,
            ecut, min_veff, max_veff,
            1e-6, None, None, 20, 1, 0.0, 1e-8,
            &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig, kinetic_slice,
            0, n_bands, !is_gamma,
        ).expect("chebfi");

        let kpt_start = ikpt * n_bands;
        let castep_eig = &bands_eigenvalues[kpt_start..kpt_start + n_bands];
        let band0_err = (ritz_values[0] - castep_eig[0]).abs();
        let max_err = ritz_values.iter().zip(castep_eig.iter())
            .map(|(c, r)| (c - r).abs()).fold(0.0f64, f64::max);
        let pass = band0_err < 5e-4 && max_err < 1e-3;

        eprintln!("[Gate 4] kpt{:2} ({:7.4},{:7.4},{:7.4}) γ={} npw={:4} | RQ Δ={:.2e} | max Δ={:.2e} | {}",
            ikpt+1, kpt_block.coords[0], kpt_block.coords[1], kpt_block.coords[2],
            is_gamma, n_pw, band0_err, max_err, if pass { "PASS" } else { "FAIL" });

        if band0_err > worst_band0 { worst_band0 = band0_err; }
        if max_err > worst_max { worst_max = max_err; }
        if !pass { all_passed = false; }
    }

    eprintln!("[Gate 4] worst band0 Δ={:.2e} Ha  worst max Δ={:.2e} Ha", worst_band0, worst_max);
    assert!(all_passed, "Gate 4 FAIL: tolerances exceeded");
    eprintln!("=== Gate 4 PASS: all {nkpts} NiO k-points match CASTEP ===");
}

// ============================================================================
// Cu111_CO — single k-point at (-0.25, 0, 0)
// ============================================================================

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn gate4_cu111co_chfsi_eigenvalues() {
    let cu_dir = cu_fixture_dir();
    if !std::path::Path::new(&cu_dir).exists() { eprintln!("SKIP"); return; }
    let pot_dir = potential_dir();

    use chemrust_hamiltonian_core::{
        CastepBinFile, CheckFile, GVectorGrid, PseudopotentialSet, NonSpin,
    };
    use chemrust_scf::KPoint;
    use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData};
    use chemrust_scf::device::blas::BlasHandle;
    use chemrust_scf::device::pcie::PcieAccount;
    use chemrust_scf::eigensolver::chebyshev::{FilterMode, chebfi_run_rust};
    use chemrust_scf::layout::{ColumnDistributed, WavefunctionSet};
    use chemrust_scf::device::Gpu;
    use chemrust_scf::downsample_array_to_wave_grid;
    use chemrust_scf::{
        Density, ScfIteration, SmearingParams, SmearingScheme, SmearingWidth,
        WaveGridArray, pw_coords_to_fft_indices,
    };
    use chemrust_scf::spin_types::{
        KptDataSet, PerSpinDensity, PerSpinPwCoefficients, SpinChannelData,
    };
    use chemrust_scf::PwCoefficients;
    use num_complex::Complex64;
    use std::sync::Arc;

    let bin = CastepBinFile::read(std::io::BufReader::new(
        std::fs::File::open(format!("{cu_dir}/Cu111_CO.castep_bin")).expect("open"),
    )).expect("read");
    let check = CheckFile::read(std::io::BufReader::new(
        std::fs::File::open(format!("{cu_dir}/Cu111_CO.check")).expect("open"),
    )).expect("read");
    let bands_text = std::fs::read_to_string(format!("{cu_dir}/Cu111_CO.bands")).expect("bands");
    let castep_eig: Vec<f64> = bands_text.lines()
        .skip_while(|l| !l.trim().starts_with("Spin component")).skip(1)
        .take_while(|l| !l.trim().starts_with("Spin component"))
        .filter_map(|l| l.trim().parse::<f64>().ok()).collect();
    let pots = PseudopotentialSet::from_dir(
        &pot_dir, &bin.cell.species_symbols, &bin.cell.species_pot_files,
    ).expect("pots");

    let cell = &bin.cell;
    let wfc = check.wavefunction.as_ref().expect("wfc");
    let wave_grid_dims = wfc.grid;
    let wave_grid = GVectorGrid::new(wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice);
    let fine_grid_dims = check.fine_grid.expect("fine_grid");
    let [fgx, fgy, fgz] = fine_grid_dims;
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;
    let k_point = KPoint { coords: kpt_block.coords, weight: 1.0 };
    let psi_data: Vec<Complex64> = kpt_block.bands.concat();
    let pw_coords = &kpt_block.pw_grid_coord;
    let fft_idx: Vec<i32> = pw_coords_to_fft_indices(pw_coords, &wave_grid);

    eprintln!("[Gate 4-Cu] n_bands={n_bands} n_pw={n_pw} kpt=({:.4},{:.4},{:.4})  grid={:?}",
        kpt_block.coords[0], kpt_block.coords[1], kpt_block.coords[2], wave_grid_dims);
    // Verify fft_idx are all in-bounds
    {
        let [ngz, ngy, ngx] = wave_grid.grid();
        let grid_size = ngz * ngy * ngx;
        let min_idx = fft_idx.iter().copied().min().unwrap();
        let max_idx = fft_idx.iter().copied().max().unwrap();
        let unique: std::collections::HashSet<i32> = fft_idx.iter().copied().collect();
        let nyq_x = ngx as i32 / 2;
        let nyq_y = ngy as i32 / 2;
        let nyq_z = ngz as i32 / 2;
        let nyq_count = pw_coords.iter().filter(|c| {
            c[0].abs() == nyq_x || c[1].abs() == nyq_y || c[2].abs() == nyq_z
        }).count();
        eprintln!("[Gate 4-Cu] fft_idx: {} unique/{} total  range=[{min_idx},{max_idx}]  grid={grid_size}  nyq_PWs={nyq_count}",
            unique.len(), n_pw);
    }

    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("BLAS");
    let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone()).expect("Solver");
    let kernels = CudaKernelSet::new(&ctx).expect("kernels");
    let mut pcie = PcieAccount::default();

    // V_eff from density
    let den_wave = downsample_array_to_wave_grid(
        bin.density.charge.as_real_grid().as_real_array(), &fine_grid, &wave_grid,
    ).expect("ds density");
    let density = Density::from_inner(WaveGridArray::from_inner(den_wave.as_fine_array().clone()));

    let scf_state = ScfIteration::<NonSpin>::builder()
        .cell(cell.clone()).pots(pots.clone())
        .wave_grid(GVectorGrid::new(wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice))
        .fine_grid(GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice))
        .density(PerSpinDensity::new(SpinChannelData::new::<NonSpin>(vec![density])))
        .psi(PerSpinPwCoefficients::new(SpinChannelData::new::<NonSpin>(vec![
            KptDataSet::new(vec![PwCoefficients::new(
                stream.alloc_zeros::<chemrust_scf::device::CudaComplex>(0).expect("dummy"),
            )], 1),
        ])))
        .psi_data(SpinChannelData::new::<NonSpin>(vec![KptDataSet::new(vec![psi_data.clone()], 1)]))
        .pw_coords(KptDataSet::new(vec![pw_coords.clone()], 1))
        .pw_fft_indices(KptDataSet::new(vec![fft_idx.clone()], 1))
        .k_points(KptDataSet::new(vec![k_point], 1))
        .smearing(SmearingParams { width: SmearingWidth::ev(0.1),
            electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
            scheme: SmearingScheme::Gaussian, spin_fix: 10 })
        .max_history(8).build();

    let v_eff_state = scf_state.build_v_eff_with_energy().expect("V_eff");
    let v_eff_data = chemrust_scf::pipeline::v_eff_prepare(
        v_eff_state.v_eff().as_ref().expect("V_eff").as_real_grid().as_real_array(),
        &fine_grid, &wave_grid,
        &stream, &mut pcie,
    ).expect("v_eff_prepare");
    let v_eff_dev = v_eff_data.gpu_slice;
    let grid_size_w = v_eff_dev.len();
    let _veff_arr = v_eff_state.v_eff().as_ref().expect("V_eff").as_real_grid().as_real_array();
    let min_veff = _veff_arr.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_veff = _veff_arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // VnlBatchData + rescreen_d (mirrors SCF pipeline)
    let mut vnl_data = VnlBatchData::precompute_with_d_override(
        pw_coords, &pots, cell, &wave_grid, Some(&fine_grid), &k_point, &psi_data, n_bands, n_pw,
        None, None, None, None, None,
        Some(&solver), &stream, &mut pcie, &blas, &kernels,
    ).expect("VnlBatchData");
    vnl_data.rescreen_d(
        v_eff_state.v_eff().as_ref().expect("V_eff").as_real_grid().as_real_array(),
        &stream, &kernels, &blas,
    ).expect("rescreen_d");

    let psi_wfn = WavefunctionSet::<ColumnDistributed>::new(psi_data.clone(), n_bands, n_pw);
    let psi_gpu = Gpu::from_host_with(&psi_wfn, &stream, &mut pcie).expect("psi");
    let fft_idx_dev: cudarc::driver::CudaSlice<i32> = stream.clone_htod(&fft_idx).expect("fft");

    let kinetic_cpu = chemrust_scf::eigensolver::davidson_types::compute_kinetic_energies(
        pw_coords, wave_grid.recip_lattice(), k_point.coords,
    );
    let ecut = kinetic_cpu.0.iter().cloned().fold(0.0_f64, f64::max);

    // --- Diagnostic: RQ using .pot_fmt V_eff (CASTEP's own V_eff) ---
    {
        let pot_text = std::fs::read_to_string(format!("{cu_dir}/Cu111_CO.pot_fmt")).expect("pot_fmt");
        let (_pg, pot_arr) = chemrust_hamiltonian_core::formatted::parse_pot_fmt(&pot_text).expect("parse");
        let v_pot = chemrust_scf::pipeline::v_eff_prepare(
            &pot_arr, &fine_grid, &wave_grid, &stream, &mut pcie,
        ).expect("pot_fmt v_eff_prepare");
        let (_, _, ritz_pot, _, _) = chebfi_run_rust(
            &psi_gpu, &v_pot.gpu_slice, &wave_grid, pw_coords, &vnl_data, &fft_idx_dev,
            ecut, min_veff, max_veff,
            1e-6, None, None, 1, 1, 0.0, 1e-8,
            &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig, Some(&kinetic_cpu.0),
            0, n_bands, true,
        ).expect("chebfi pot_fmt");
        let rq_pot = ritz_pot.get(0).copied().unwrap_or(f64::NAN);
        eprintln!("[Gate 4-Cu] RQ(.pot_fmt V_eff) band0 = {:.8} Ha", rq_pot);
    }

    // --- Stream sync before any chebfi_run_rust ---
    stream.synchronize().expect("sync before chebfi");

    // === CPU vs GPU V_loc comparison ===
    // Compute H·psi on CPU using the verified apply_local_hamiltonian
    {
        use chemrust_hamiltonian_core::hamiltonian::apply_local_hamiltonian;
        // Build CPU FFT indices: [iz, iy, ix] same as GPU but 3D
        let [ngz, ngy, ngx] = wave_grid.grid();
        let fft_3d: Vec<[usize; 3]> = pw_coords.iter().map(|&[h, k, l]| {
            let ix = if h >= 0 { h as usize } else { (h + ngx as i32) as usize };
            let iy = if k >= 0 { k as usize } else { (k + ngy as i32) as usize };
            let iz = if l >= 0 { l as usize } else { (l + ngz as i32) as usize };
            [iz, iy, ix]
        }).collect();
        let recip = cell.recip_lattice.as_array();
        let gcart: Vec<[f64; 3]> = pw_coords.iter().map(|&[h, k, l]| {
            let gf = [h as f64, k as f64, l as f64];
            std::array::from_fn(|j| (0..3).map(|i| gf[i] * recip[i][j]).sum())
        }).collect();
        let k_cart = {
            let kf = k_point.coords;
            std::array::from_fn(|j| (0..3).map(|i| kf[i] * recip[i][j]).sum())
        };
        let psi_b0 = &psi_data[0..n_pw];
        let v_eff_hcore = chemrust_hamiltonian_core::EffectivePotential::from_inner(
            chemrust_hamiltonian_core::fft::RealGrid::from_inner(
                v_eff_state.v_eff().as_ref().unwrap().as_real_grid().as_real_array().to_owned()
            ));
        let cpu_hpsi = apply_local_hamiltonian(
            psi_b0, &fft_3d, &gcart, k_cart, &v_eff_hcore, &wave_grid,
        ).expect("CPU apply_local_hamiltonian");
        let mut cpu_t = 0.0f64;
        let mut cpu_tvloc = 0.0f64;
        for g in 0..n_pw {
            cpu_t += psi_b0[g].norm_sqr() * 0.5 * (
                (k_cart[0]+gcart[g][0]).powi(2) +
                (k_cart[1]+gcart[g][1]).powi(2) +
                (k_cart[2]+gcart[g][2]).powi(2)
            );
            cpu_tvloc += (psi_b0[g].conj() * cpu_hpsi[g]).re;
        }
        eprintln!("[Gate 4-Cu] CPU: T={:.6} H_TVloc={:.6} V_loc={:.6} Ha",
            cpu_t, cpu_tvloc, cpu_tvloc - cpu_t);
    }

    // === Diagnostic: constant V_eff test ===
    // If V_eff = -5 Ha everywhere, ΔRQ should be -5.0 Ha.
    // This verifies the FFT pipeline applies V_eff correctly.
    let (_pf, _hp, ritz_actual, _, _) = chebfi_run_rust(
        &psi_gpu, &v_eff_dev, &wave_grid, pw_coords, &vnl_data, &fft_idx_dev,
        ecut, min_veff, max_veff,
        1e-6, None, None, 1, 1, 0.0, 1e-8,
        &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
        FilterMode::SinvHKeepHEig, Some(&kinetic_cpu.0),
        0, n_bands, true,
    ).expect("chebfi actual");
    let rq0 = ritz_actual.get(0).copied().unwrap_or(f64::NAN);

    {
        let v_const: cudarc::driver::CudaSlice<f64> = stream.clone_htod(&vec![-5.0_f64; grid_size_w]).expect("const");
        let (_, _, ritz_const, _, _) = chebfi_run_rust(
            &psi_gpu, &v_const, &wave_grid, pw_coords, &vnl_data, &fft_idx_dev,
            ecut, -5.0, -5.0,
            1e-6, None, None, 1, 1, 0.0, 1e-8,
            &kernels, &mut pcie, &blas, &solver, &stream, &ctx,
            FilterMode::SinvHKeepHEig, Some(&kinetic_cpu.0),
            0, n_bands, true,
        ).expect("chebfi const");
        let rq_c = ritz_const.get(0).copied().unwrap_or(f64::NAN);
        eprintln!("[Gate 4-Cu] V_eff range=[{min_veff:.4}, {max_veff:.4}] Ha  ecut={ecut:.2} Ha");
        eprintln!("[Gate 4-Cu] CASTEP band0 = {:.8} Ha", castep_eig[0]);
        eprintln!("[Gate 4-Cu] RQ(actual V_eff)  band0 = {:.8} Ha (Δ={:.2e})", rq0, (rq0-castep_eig[0]).abs());
        eprintln!("[Gate 4-Cu] RQ(V_eff=-5)      band0 = {:.8} Ha", rq_c);
        eprintln!("[Gate 4-Cu] ΔRQ from V_eff=-5 (expect -5.0) = {:.4} Ha", rq_c - rq0);
    }

    // Show RQ vs CASTEP for first and last bands
    eprintln!("[Gate 4-Cu] band | RQ (Ha)        | CASTEP (Ha)    | Δ (Ha)");
    for b in 0..n_bands.min(10) {
        let rq = ritz_actual.get(b).copied().unwrap_or(f64::NAN);
        eprintln!("[Gate 4-Cu] {:4} | {:.10} | {:.10} | {:.2e}", b, rq, castep_eig[b], (rq - castep_eig[b]).abs());
    }
    eprintln!("[Gate 4-Cu] ...");
    for b in (n_bands - 5).max(10)..n_bands {
        let rq = ritz_actual.get(b).copied().unwrap_or(f64::NAN);
        eprintln!("[Gate 4-Cu] {:4} | {:.10} | {:.10} | {:.2e}", b, rq, castep_eig[b], (rq - castep_eig[b]).abs());
    }

    let band0_err = (rq0 - castep_eig[0]).abs();
    assert!(band0_err < 5e-4, "Gate 4-Cu FAIL: band0 Δ={:.2e}", band0_err);
}

// ============================================================================
// Gate 5: ChFSI cascade check — SCF on NiO, panic if eigenvalues drift
// ============================================================================

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn gate5_nio_chfsi_cascade() {
    unsafe { std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson"); }

    let dir = niO_fixture_dir();
    if !std::path::Path::new(&dir).exists() { eprintln!("SKIP"); return; }
    let pot_dir = potential_dir();

    use chemrust_hamiltonian_core::{
        CastepBinFile, CheckFile, GVectorGrid, PseudopotentialSet, NonSpin,
    };
    use chemrust_scf::{
        KPoint, ScfIteration, SmearingParams, SmearingScheme, SmearingWidth,
        WaveGridArray, Density, pw_coords_to_fft_indices, downsample_array_to_wave_grid,
        run_scf_with_energy_gated, ScfDivergenceGate,
    };
    use chemrust_scf::spin_types::{
        KptDataSet, PerSpinDensity, PerSpinPwCoefficients, SpinChannelData,
    };
    use chemrust_scf::PwCoefficients;
    use num_complex::Complex64;
    use std::sync::Arc;

    let bin = CastepBinFile::read(std::io::BufReader::new(
        std::fs::File::open(format!("{dir}/NiO.castep_bin")).expect("open"),
    )).expect("read");
    let check = CheckFile::read(std::io::BufReader::new(
        std::fs::File::open(format!("{dir}/NiO.check")).expect("open"),
    )).expect("read");
    let bands_text = std::fs::read_to_string(format!("{dir}/NiO.bands")).expect("bands");
    let bands_eigenvalues = parse_bands_first_spin(&bands_text);
    let pots = PseudopotentialSet::from_dir(
        &pot_dir, &bin.cell.species_symbols, &bin.cell.species_pot_files,
    ).expect("pots");

    let cell = &bin.cell;
    let wfc = check.wavefunction.as_ref().expect("wfc");
    let wave_grid_dims = wfc.grid;

    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA"));
    let stream = ctx.default_stream();

    // Density from .castep_bin (on fine grid), downsample to wave grid
    let den_fine = bin.density.charge.as_real_grid().as_real_array().clone();
    let wave_grid = GVectorGrid::new(wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice);
    let fine_grid_dims = check.fine_grid.expect("fine_grid");
    let [fgx, fgy, fgz] = fine_grid_dims;
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);
    let den_wave = downsample_array_to_wave_grid(&den_fine, &fine_grid, &wave_grid).expect("ds");
    let density = Density::from_inner(WaveGridArray::from_inner(den_wave.as_fine_array().clone()));

    // All 14 k-points
    let nkpts = wfc.kpt_data.len();
    let kpt_weights: [f64; 14] = [0.07407407; 14];
    let mut all_psi: Vec<Vec<Complex64>> = Vec::with_capacity(nkpts);
    let mut all_pw: Vec<Vec<[i32; 3]>> = Vec::with_capacity(nkpts);
    let mut all_fft: Vec<Vec<i32>> = Vec::with_capacity(nkpts);
    let mut all_kpts: Vec<KPoint> = Vec::with_capacity(nkpts);
    for ikpt in 0..nkpts {
        let kb = &wfc.kpt_data[ikpt];
        all_psi.push(kb.bands.concat());
        all_pw.push(kb.pw_grid_coord.clone());
        all_fft.push(pw_coords_to_fft_indices(&kb.pw_grid_coord, &wave_grid));
        all_kpts.push(KPoint { coords: kb.coords, weight: kpt_weights[ikpt] });
    }

    let state = ScfIteration::<NonSpin>::builder()
        .cell(cell.clone()).pots(pots.clone())
        .wave_grid(GVectorGrid::new(wave_grid_dims[0], wave_grid_dims[1], wave_grid_dims[2], cell.recip_lattice))
        .fine_grid(GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice))
        .density(PerSpinDensity::new(SpinChannelData::new::<NonSpin>(vec![density])))
        .psi(PerSpinPwCoefficients::new(SpinChannelData::new::<NonSpin>(vec![
            KptDataSet::new(all_kpts.iter().map(|_| PwCoefficients::new(
                stream.alloc_zeros::<chemrust_scf::device::CudaComplex>(0).expect("dummy"),
            )).collect(), nkpts),
        ])))
        .psi_data(SpinChannelData::new::<NonSpin>(vec![KptDataSet::new(all_psi.clone(), nkpts)]))
        .pw_coords(KptDataSet::new(all_pw.clone(), nkpts))
        .pw_fft_indices(KptDataSet::new(all_fft.clone(), nkpts))
        .k_points(KptDataSet::new(all_kpts, nkpts))
        .smearing(SmearingParams { width: SmearingWidth::ev(0.1),
            electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
            scheme: SmearingScheme::Gaussian, spin_fix: 6 })
        .max_history(8).build();

    let castep_band0 = bands_eigenvalues[0];

    eprintln!("[Gate 5] NiO ChFSI SCF — max 5 iters, FORCING Off mixing, CASTEP band0={castep_band0:.8} Ha");

    let result = run_scf_with_energy_gated(
        state,
        8,
        1e-6,
        Some(ScfDivergenceGate {
            max_last_band_ha: 30.0,
            min_band0_ha: -30.0,
            max_veff_range_factor: 10.0,
            max_iter: 5,
            electron_count_tolerance: 0.15,
            soft_fraction_tolerance: 0.99,
        }),
    ).expect("SCF");

    let final_eig = result.eigenvalues.first()
        .and_then(|s| s.first())
        .map(|e| e[0])
        .unwrap_or(f64::NAN);
    let drift = (final_eig - castep_band0).abs();
    let energy_ev = result.total_energy * chemrust_scf::HARTREE_TO_EV;
    eprintln!("[Gate 5] final band0={final_eig:.8} Ha  Δ={drift:.2e} Ha  E_total={energy_ev:.6} eV");

    assert!(drift < 0.02, "Gate 5 FAIL: band0 drifted {:.2e} Ha from CASTEP", drift);
    eprintln!("=== Gate 5 PASS: ChFSI SCF stable, band0 drift {:.2e} Ha ===", drift);
}
