// ---------------------------------------------------------------------------
// S⁻¹ precision decomposition — identify the numerical bottleneck
// ---------------------------------------------------------------------------
// Measures for Cu111_CO and NiO:
//   ζ = ‖S⁻¹·S·ψ − ψ‖_∞        (full Woodbury error)
//   η = ‖M·y − Bᴴ·(S·ψ)‖ / ‖Bᴴ·(S·ψ)‖  (zgetrs relative residual)
//   κ = ‖M‖_∞ · ‖M⁻¹‖_∞         (condition number)

mod fixtures;

fn gpu_available() -> bool {
    std::path::Path::new("/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8").exists()
}

fn condest(m: &[f64], n: usize) -> f64 {
    let mut norm_m: f64 = 0.0;
    for i in 0..n {
        let s: f64 = (0..n).map(|j| m[i*n + j].abs()).sum();
        norm_m = norm_m.max(s);
    }
    if norm_m < 1e-30 { return f64::INFINITY; }

    let solve = |b: &[f64], x: &mut [f64]| {
        let mut a = m.to_vec();
        let mut aug = b.to_vec();
        for k in 0..n {
            let mut max_r = k;
            let mut max_v = a[k*n + k].abs();
            for r in (k+1)..n { let v = a[r*n + k].abs(); if v > max_v { max_v = v; max_r = r; } }
            if max_v < 1e-30 { continue; }
            if max_r != k {
                for c in 0..n { a.swap(k*n + c, max_r*n + c); }
                aug.swap(k, max_r);
            }
            let pivot = a[k*n + k];
            for r in (k+1)..n {
                let factor = a[r*n + k] / pivot;
                if factor.abs() < 1e-30 { continue; }
                for c in k..n { a[r*n + c] -= factor * a[k*n + c]; }
                aug[r] -= factor * aug[k];
            }
        }
        for i in (0..n).rev() {
            let mut s = aug[i];
            for j in (i+1)..n { s -= a[i*n + j] * x[j]; }
            x[i] = if a[i*n + i].abs() > 1e-30 { s / a[i*n + i] } else { 0.0 };
        }
    };

    let mut x = vec![1.0_f64; n];
    let mut norm_inv: f64 = 0.0;
    for _ in 0..20 {
        let mut x_new = vec![0.0_f64; n];
        solve(&x, &mut x_new);
        let xn = x_new.iter().map(|&v| v.abs()).fold(0.0, f64::max);
        let bn = x.iter().map(|&v| v.abs()).fold(0.0, f64::max);
        if bn > 1e-30 { norm_inv = norm_inv.max(xn / bn); }
        if xn > 1e-30 { for v in &mut x_new { *v /= xn; } }
        x.copy_from_slice(&x_new);
    }
    norm_m * norm_inv
}

/// Measure zgetrs relative residual: η = ‖M·y − b‖/‖b‖ where b = Bᴴ·(S·ψ).
/// Downloads y from GPU after zgetrs, computes M·y exactly on CPU via stored Q⁻¹ and Bᴴ·B.
fn zgetrs_residual(
    psi_cpu: &[chemrust_scf::device::CudaComplex],
    n_pw: usize, n_bands: usize,
    vnl_data: &chemrust_scf::density::test_api::VnlBatchData,
    blas: &chemrust_scf::device::blas::BlasHandle,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    solver: &chemrust_scf::device::solver::SolverHandle,
) -> f64 {
    use chemrust_scf::device::{CudaComplex, blas};
    use chemrust_scf::eigensolver::davidson_types::PwCoefficients;
    use cudarc::cusolver::sys::cublasOperation_t;

    let nte = vnl_data.n_total_expanded as usize;
    let nte_i = nte as i32;
    let n_pw_i = n_pw as i32;
    let nb_i = n_bands as i32;

    // 1. hpsi = S·psi (start with psi, then apply Woodbury S)
    let psi_dev = PwCoefficients::new(stream.clone_htod(psi_cpu).unwrap());
    let mut hpsi = PwCoefficients::new(stream.alloc_zeros::<CudaComplex>(n_pw * n_bands).unwrap());
    stream.memcpy_dtod(&*psi_dev, &mut *hpsi).unwrap();
    unsafe {
        chemrust_scf::eigensolver::hamiltonian::apply_s_times()
            .psi_dev(&psi_dev).spsi_dev(&mut hpsi)
            .vnl_data(vnl_data).n_bands(nb_i).n_pw(n_pw_i)
            .blas(blas).stream(stream)
            .call().unwrap();
    }
    // hpsi now = S·psi

    // 2. temp = Bᴴ·hpsi
    let mut temp = stream.alloc_zeros::<CudaComplex>(nte * n_bands).unwrap();
    unsafe {
        blas.gemm_c64(blas::ZgemmConfig {
            transa: blas::op::C, transb: blas::op::N,
            m: nte_i, n: nb_i, k: n_pw_i,
            alpha: CudaComplex { x: 1.0, y: 0.0 },
            lda: n_pw_i, ldb: n_pw_i, ldc: nte_i,
            beta: CudaComplex { x: 0.0, y: 0.0 },
        }, &vnl_data.b_concat, &*hpsi, &mut temp).unwrap();
    }
    let rhs: Vec<CudaComplex> = stream.clone_dtoh(&temp).unwrap();

    // 3. Solve M·y = temp via zgetrs
    let mut info_dev = stream.alloc_zeros::<i32>(1).unwrap();
    solver.zgetrs(cublasOperation_t::CUBLAS_OP_N, nte_i, nb_i,
        &vnl_data.lu_m, &vnl_data.lu_ipiv, &mut temp, &mut info_dev).unwrap();
    let y_sol: Vec<CudaComplex> = stream.clone_dtoh(&temp).unwrap();

    // 4. Compute M·y on CPU and compare to rhs
    let eps: f64 = 1e-15;
    let mut max_rel: f64 = 0.0;
    for col in 0..n_bands {
        let off = col * nte;
        // M·y = Q⁻¹·y + (Bᴴ·B)·y + ε·y
        let mut my = vec![CudaComplex { x: 0.0, y: 0.0 }; nte];
        let mut row = 0usize;
        for q_inv in &vnl_data.q_inv_per_ion {
            let ne2 = q_inv.len();
            let ne = (ne2 as f64).sqrt() as usize;
            for i in 0..ne {
                for j in 0..ne {
                    let yj = y_sol[off + row + j];
                    my[row + i].x += q_inv[i*ne + j] * yj.x;
                    my[row + i].y += q_inv[i*ne + j] * yj.y;
                }
            }
            row += ne;
        }
        let bhb = &vnl_data.bhb_cpu;
        for i in 0..nte {
            for j in 0..nte {
                let yj = y_sol[off + j];
                my[i].x += bhb[i*nte + j] * yj.x;
                my[i].y += bhb[i*nte + j] * yj.y;
            }
        }
        for i in 0..nte {
            my[i].x += eps * y_sol[off + i].x;
            my[i].y += eps * y_sol[off + i].y;
        }
        let mut r_sq: f64 = 0.0;
        let mut b_sq: f64 = 0.0;
        for i in 0..nte {
            let dx = my[i].x - rhs[off + i].x;
            let dy = my[i].y - rhs[off + i].y;
            r_sq += dx*dx + dy*dy;
            b_sq += rhs[off + i].x*rhs[off + i].x + rhs[off + i].y*rhs[off + i].y;
        }
        let rel = r_sq.sqrt() / b_sq.sqrt().max(1e-30);
        max_rel = max_rel.max(rel);
    }
    max_rel
}

#[cfg(test)]
mod tests {
    use chemrust_hamiltonian_core::GVectorGrid;
    use chemrust_scf::KPoint;
    use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData};
    use chemrust_scf::device::blas::BlasHandle;
    use chemrust_scf::device::pcie::PcieAccount;
    use chemrust_scf::eigensolver::hamiltonian::check_s_inv_s_identity;
    use chemrust_scf::device::CudaComplex;
    use num_complex::Complex64;
    use std::sync::Arc;

    fn run_precision(
        label: &str,
        cell: &chemrust_hamiltonian_core::CellGeometry,
        pots: &chemrust_hamiltonian_core::PseudopotentialSet,
        wave_grid: &GVectorGrid,
        kpt_block: &chemrust_hamiltonian_core::types::KptWaveBlock,
    ) {
        let n_bands = kpt_block.bands.len();
        let n_pw = kpt_block.nplw;
        let k_point = KPoint { coords: kpt_block.coords, weight: 1.0 };
        let psi_data: Vec<Complex64> = kpt_block.bands.concat();

        let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA"));
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).expect("BLAS");
        let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone()).expect("Solver");
        let kernels = CudaKernelSet::new(&ctx).expect("Kernels");
        let mut pcie = PcieAccount::default();

        let vnl_data = VnlBatchData::precompute_with_d_override(
            &kpt_block.pw_grid_coord, pots, cell, &wave_grid,
            None, &k_point, &psi_data, n_bands, n_pw,
            None, None, None, None, None,
            Some(&solver), &stream, &mut pcie, &blas, &kernels,
        ).expect("VnlBatchData");

        let nte = vnl_data.n_total_expanded as usize;

        // ζ = full Woodbury error
        let band0: Vec<Complex64> = psi_data.iter().take(n_pw).copied().collect();
        let zeta = check_s_inv_s_identity(&band0, n_pw, &vnl_data, &blas, &stream, &solver).unwrap();
        // Iterative refinement (ABINIT m_invovl.F90:1102-1140) must improve
        // Woodbury S⁻¹ precision below 1e-8.  Without refinement, ζ ≈ 3.4e-6.
        assert!(zeta < 1e-8, "[precision] {label}: ζ={zeta:.6e} exceeds 1e-8 gate — Woodbury refinement broken");

        // η = zgetrs relative residual
        let band0_cplx: Vec<CudaComplex> = band0.iter()
            .map(|&c| CudaComplex { x: c.re, y: c.im }).collect();
        let eta = super::zgetrs_residual(&band0_cplx, n_pw, 1, &vnl_data, &blas, &stream, &solver);

        // κ = condition number of M
        let eps: f64 = 1e-15;
        let mut m_cpu = vec![0.0_f64; nte*nte];
        let mut row = 0usize;
        for q_inv in &vnl_data.q_inv_per_ion {
            let ne = (q_inv.len() as f64).sqrt() as usize;
            for i in 0..ne { for j in 0..ne { m_cpu[(row+i)*nte + (row+j)] += q_inv[i*ne + j]; } }
            row += ne;
        }
        for i in 0..nte { for j in 0..nte { m_cpu[i*nte + j] += vnl_data.bhb_cpu[i*nte + j]; } }
        for i in 0..nte { m_cpu[i*nte + i] += eps; }
        let cond = super::condest(&m_cpu, nte);
        let ne_sizes: Vec<usize> = vnl_data.q_inv_per_ion.iter().map(|q| (q.len() as f64).sqrt() as usize).collect();

        // Measure cancellation: download S·ψ, B·y, and final result
        use chemrust_scf::eigensolver::davidson_types::PwCoefficients;
        use chemrust_scf::device::blas;
        use cudarc::cusolver::sys::cublasOperation_t;

        // S·ψ = ψ + β·Q·βᴴ·ψ, stored in hpsi before the B·y subtraction
        let mut hpsi_before = PwCoefficients::new(stream.alloc_zeros::<CudaComplex>(n_pw).unwrap());
        let psi_dev = PwCoefficients::new(stream.clone_htod(&band0_cplx).unwrap());
        stream.memcpy_dtod(&*psi_dev, &mut *hpsi_before).unwrap();
        unsafe {
            chemrust_scf::eigensolver::hamiltonian::apply_s_times()
                .psi_dev(&psi_dev).spsi_dev(&mut hpsi_before)
                .vnl_data(&vnl_data).n_bands(1).n_pw(n_pw as i32)
                .blas(&blas).stream(&stream).call().unwrap();
        }
        let spsi_data: Vec<CudaComplex> = stream.clone_dtoh(&*hpsi_before).unwrap();
        // B·y contribution: download temp (which = y = M⁻¹·Bᴴ·(S·ψ)), run B·y on GPU
        let mut temp2 = stream.alloc_zeros::<CudaComplex>(nte).unwrap();
        let mut b_y = stream.alloc_zeros::<CudaComplex>(n_pw).unwrap();
        unsafe {
            // Recompute Bᴴ·(S·ψ)
            blas.gemm_c64(blas::ZgemmConfig {
                transa: blas::op::C, transb: blas::op::N,
                m: nte as i32, n: 1, k: n_pw as i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw as i32, ldb: n_pw as i32, ldc: nte as i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
            }, &vnl_data.b_concat, &*hpsi_before, &mut temp2).unwrap();
        }
        let mut info_dev = stream.alloc_zeros::<i32>(1).unwrap();
        solver.zgetrs(cublasOperation_t::CUBLAS_OP_N, nte as i32, 1,
            &vnl_data.lu_m, &vnl_data.lu_ipiv, &mut temp2, &mut info_dev).unwrap();
        // B·y
        unsafe {
            blas.gemm_c64(blas::ZgemmConfig {
                transa: blas::op::N, transb: blas::op::N,
                m: n_pw as i32, n: 1, k: nte as i32,
                alpha: CudaComplex { x: 1.0, y: 0.0 },
                lda: n_pw as i32, ldb: nte as i32, ldc: n_pw as i32,
                beta: CudaComplex { x: 0.0, y: 0.0 },
            }, &vnl_data.b_concat, &temp2, &mut b_y).unwrap();
        }
        let by_data: Vec<CudaComplex> = stream.clone_dtoh(&b_y).unwrap();

        // Compute norms
        let psi_norm: f64 = band0_cplx.iter().map(|c| c.x*c.x + c.y*c.y).sum::<f64>().sqrt();
        let spsi_norm: f64 = spsi_data.iter().map(|c| c.x*c.x + c.y*c.y).sum::<f64>().sqrt();
        let by_norm: f64 = by_data.iter().map(|c| c.x*c.x + c.y*c.y).sum::<f64>().sqrt();
        // The augmentation part: aug = S·ψ − ψ
        let mut aug_norm_sq = 0.0_f64;
        for i in 0..n_pw {
            let dx = spsi_data[i].x - band0_cplx[i].x;
            let dy = spsi_data[i].y - band0_cplx[i].y;
            aug_norm_sq += dx*dx + dy*dy;
        }
        let aug_norm = aug_norm_sq.sqrt();
        // How well does B·y cancel S·ψ? The difference should be ≈ ψ
        let mut diff_norm_sq = 0.0_f64;
        for i in 0..n_pw {
            let dx = spsi_data[i].x - by_data[i].x - band0_cplx[i].x;
            let dy = spsi_data[i].y - by_data[i].y - band0_cplx[i].y;
            diff_norm_sq += dx*dx + dy*dy;
        }
        let woodbury_err = diff_norm_sq.sqrt();

        eprintln!("[precision] {label}: n_pw={n_pw} nte={nte} ζ={zeta:.6e} η={eta:.6e} κ={cond:.2e}");
        eprintln!("[precision]   ‖ψ‖={psi_norm:.6e}  ‖Sψ‖={spsi_norm:.6e}  ‖aug‖={aug_norm:.6e}  ‖B·y‖={by_norm:.6e}  ‖Sψ−B·y−ψ‖={woodbury_err:.6e}");
        eprintln!("[precision]   ‖Sψ‖/‖ψ‖={:.2e}  ‖aug‖/‖ψ‖={:.2e}  ne_sizes={ne_sizes:?}",
            spsi_norm/psi_norm, aug_norm/psi_norm);
        stream.synchronize().unwrap();
    }

    #[test]
    #[ignore = "requires GPU and CASTEP fixture data"]
    fn precision_cu111_co() {
        if !super::gpu_available() { eprintln!("SKIP"); return; }
        let fx = super::fixtures::cu111_co::fixture();
        let wfc = fx.check.wavefunction.as_ref().expect("wfc");
        let [ngx, ngy, ngz] = wfc.grid;
        let wave_grid = GVectorGrid::new(ngx, ngy, ngz, fx.bin.cell.recip_lattice);
        run_precision("Cu111_CO", &fx.bin.cell, &fx.pots, &wave_grid, &wfc.kpt_data[0]);
    }

    #[test]
    #[ignore = "requires GPU and CASTEP fixture data"]
    fn precision_nio() {
        if !super::gpu_available() { eprintln!("SKIP"); return; }
        let nio_dir = "/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin";
        use chemrust_hamiltonian_core::{CastepBinFile, CheckFile, PseudopotentialSet};
        let bin = CastepBinFile::read(std::io::BufReader::new(
            std::fs::File::open(format!("{nio_dir}/NiO.castep_bin")).unwrap())).unwrap();
        let check = CheckFile::read(std::io::BufReader::new(
            std::fs::File::open(format!("{nio_dir}/NiO.check")).unwrap())).unwrap();
        let pots = PseudopotentialSet::from_dir(
            "/export/Potentials", &bin.cell.species_symbols, &bin.cell.species_pot_files).unwrap();
        let wfc = check.wavefunction.as_ref().expect("wfc");
        let [ngx, ngy, ngz] = wfc.grid;
        let wave_grid = GVectorGrid::new(ngx, ngy, ngz, bin.cell.recip_lattice);
        run_precision("NiO", &bin.cell, &pots, &wave_grid, &wfc.kpt_data[0]);
    }
}
