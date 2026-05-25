//! Quick diagnostic: compare Davidson iter-1 eigenvalues and density split
//! against CASTEP reference, without running full SCF.

mod fixtures;

fn gpu_available() -> bool {
    std::env::var("CUDA_VISIBLE_DEVICES").map_or(true, |s| s != "-1")
}

#[test]
#[ignore = "requires GPU + CASTEP fixtures"]
fn davidson_iter1_eigenvalues_vs_castep() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU");
        return;
    }
    unsafe {
        std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson");
    }

    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);
    let veff = state
        .build_v_eff_with_energy()
        .expect("build_v_eff");
    let wfn = veff.diagonalize(8, None).expect("diagonalize");

    // Compare eigenvalues
    let ev = wfn.eigenvalues().to_vec();
    let ref_ev = &fx.bands_eigenvalues;

    eprintln!("=== Eigenvalue comparison (Davidson vs CASTEP) ===");
    eprintln!("band | Davidson (Ha) | CASTEP (Ha) | Δ (Ha)    | Δ (eV)");
    let mut max_delta = 0.0f64;
    let mut rms_delta = 0.0f64;
    for b in 0..ev.len().min(20) {
        let d = (ev[b] - ref_ev[b]).abs();
        eprintln!(" {:>3} | {:>13.8} | {:>11.8} | {:>9.6} | {:>8.6}",
            b, ev[b], ref_ev[b], d, d * 27.2114);
        if d > max_delta { max_delta = d; }
        rms_delta += d * d;
    }
    if ev.len() > 20 {
        for b in 20..ev.len() {
            let d = (ev[b] - ref_ev[b]).abs();
            if d > max_delta { max_delta = d; }
            rms_delta += d * d;
        }
    }
    rms_delta = (rms_delta / ev.len() as f64).sqrt();
    eprintln!("max Delta_lambda = {:.6} Ha = {:.3} eV", max_delta, max_delta * 27.2114);
    eprintln!("rms Delta_lambda = {:.6} Ha = {:.3} eV", rms_delta, rms_delta * 27.2114);

    // Check n_locked
    if let Some(diag) = wfn.davidson_diagnostics() {
        eprintln!("n_locked = {}, n_unconv = {}", diag.n_locked, diag.n_unconverged);
        eprintln!("max_residual_sinv = {:.6e} Ha", diag.max_residual_sinv);
    }

    // Construct density from Davidson psi to get soft + aug split
    let dens = wfn
        .construct_density_off()
        .expect("construct_density");

    let rho_arr = dens.density().as_wave_array();
    let soft_sum: f64 = rho_arr.iter().sum();
    let n_soft = rho_arr.len() as f64;
    let soft_e = soft_sum / n_soft;
    eprintln!("soft density: sum={:.4e}  electrons={:.4}", soft_sum, soft_e);

    let aug_e = dens.density_aug_fine()
        .map(|aug| {
            let arr = aug.as_real_array();
            let sum: f64 = arr.iter().sum();
            eprintln!("aug  density: sum={:.4e}  electrons={:.4}", sum, sum / arr.len() as f64);
            sum / arr.len() as f64
        })
        .unwrap_or(0.0);
    let total_e = soft_e + aug_e;
    let soft_fraction = if total_e > 0.0 { soft_e / total_e } else { 0.0 };
    eprintln!("total electrons = {:.4}  soft fraction = {:.4}", total_e, soft_fraction);
    eprintln!("CASTEP reference: RHO_SOFT ~3.0e7  RHO_AUG ~5.1e7  soft frac ~0.368");

    // ---- Advance to iter-2 without density perturbation ----
    use chemrust_scf::CheckOutcome;
    let mixed = dens.mix();
    let outcome = mixed.check(1e-8).expect("iter-1 check");
    let state_2 = match outcome {
        CheckOutcome::Converged(_) => {
            eprintln!("iter-1 unexpectedly converged");
            unsafe { std::env::remove_var("CHEMRUST_EIGENSOLVER"); }
            return;
        }
        CheckOutcome::NotConverged { state, .. } => state,
    };

    // iter-2: build V_eff from iter-1's post-mix density + aug
    let veff_2 = state_2.build_v_eff_with_energy().expect("iter-2 build_v_eff");
    let wfn_2 = veff_2.diagonalize(8, None).expect("iter-2 diagonalize");

    // Compare eigenvalues iter-1 vs iter-2
    let ev_2 = wfn_2.eigenvalues();
    eprintln!("\n=== iter-2 eigenvalue comparison ===");
    let mut max_d2 = 0.0f64;
    let mut rms_d2 = 0.0f64;
    for b in 0..ev.len() {
        let d = (ev_2[b] - ev[b]).abs();
        if d > max_d2 { max_d2 = d; }
        rms_d2 += d * d;
    }
    rms_d2 = (rms_d2 / ev.len() as f64).sqrt();
    eprintln!("max |iter2 - iter1| = {:.6} Ha", max_d2);
    eprintln!("rms |iter2 - iter1| = {:.6} Ha", rms_d2);
    if let Some(diag) = wfn_2.davidson_diagnostics() {
        eprintln!("n_locked = {}, n_unconv = {}", diag.n_locked, diag.n_unconverged);
    }

    // iter-2 density
    let dens_2 = wfn_2.construct_density_off().expect("iter-2 construct_density");
    let soft2_sum: f64 = dens_2.density().as_wave_array().iter().sum();
    let soft2_e = soft2_sum / dens_2.density().as_wave_array().len() as f64;
    let aug2_e = dens_2.density_aug_fine()
        .map(|aug| { let a = aug.as_real_array(); a.iter().sum::<f64>() / a.len() as f64 })
        .unwrap_or(0.0);
    let total2_e = soft2_e + aug2_e;
    eprintln!("iter-2: soft={:.4} aug={:.4} total={:.4} soft_frac={:.4}",
        soft2_e, aug2_e, total2_e,
        if total2_e > 0.0 { soft2_e / total2_e } else { 0.0 });

    unsafe {
        std::env::remove_var("CHEMRUST_EIGENSOLVER");
    }
}
