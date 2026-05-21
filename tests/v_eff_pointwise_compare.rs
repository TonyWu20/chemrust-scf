//! Pointwise comparison of iter-1 and iter-2 V_eff against CASTEP `.pot_fmt`.
//!
//! Drives one SCF iteration, dumps iter-1 V_eff and iter-2 V_eff to disk,
//! and prints per-point statistics:
//! - Global L_inf and L_2 diff vs CASTEP reference.
//! - Per-Cu-ion ROI (within 2.0 Bohr of nucleus) L_inf and L_2.
//!
//! Hypothesis: iter-1 V_eff matches CASTEP closely (built from fixture
//! density which IS the CASTEP density), but iter-2 V_eff has localised
//! errors AT ion centres that don't show in the global range diagnostic.

mod fixtures;

use std::io::Write;

#[allow(dead_code)]
fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

/// Dump a 3-D f64 array to disk as raw little-endian bytes.
fn dump_array_3d(arr: &ndarray::Array3<f64>, path: &str) {
    let bytes: Vec<u8> = arr.iter().flat_map(|&v| v.to_le_bytes()).collect();
    std::fs::write(path, &bytes).expect("write");
    eprintln!("[dump] wrote {} bytes to {path}", bytes.len());
}

/// Compute fractional → Cartesian using the real lattice.
fn frac_to_cart(frac: [f64; 3], real_lat: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            // CASTEP convention: a, b, c stored as rows; r_cart = sum_j frac[j] * lat[j][i]
            c[i] += frac[j] * real_lat[j][i];
        }
    }
    c
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn iter1_iter2_v_eff_pointwise_vs_pot_fmt() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixtures::cu111_co::fixture();
    let pot_ref = &fx.pot_fmt;
    let cell = &fx.bin.cell;
    let real_lat = cell.real_lattice.as_array();
    eprintln!(
        "[fixture] pot_fmt shape: {:?}  num_ions: {}",
        pot_ref.shape(),
        cell.num_ions
    );
    let pot_min = pot_ref.iter().cloned().fold(f64::INFINITY, f64::min);
    let pot_max = pot_ref.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[fixture] pot_fmt min={pot_min:.4} max={pot_max:.4} range={:.4} Ha",
        pot_max - pot_min
    );

    let state = fixtures::cu111_co::build_scf_state(fx);

    // iter-1: V_eff from fixture density.
    let iter1_v = state.build_v_eff_with_energy().expect("iter-1 build_v_eff");
    let iter1_arr = iter1_v
        .v_eff()
        .as_ref()
        .expect("v_eff present")
        .as_real_grid()
        .as_real_array()
        .clone();
    dump_array_3d(&iter1_arr, "/tmp/cu111_v_eff_iter1.bin");

    // Drive iter-1 to produce iter-2's input density.
    let iter1_diag = iter1_v.diagonalize(8, None).expect("iter-1 diagonalize");
    let iter1_dens = iter1_diag.construct_density_off().expect("construct_density");
    let iter1_mixed = iter1_dens.mix();
    let iter2_init = match iter1_mixed.check(1e-8).expect("iter-1 check") {
        chemrust_scf::CheckOutcome::Converged(_) => panic!("converged unexpectedly"),
        chemrust_scf::CheckOutcome::NotConverged { state, .. } => state,
    };

    // iter-2: V_eff from iter-1 output density.
    let iter2_v = iter2_init.build_v_eff_with_energy().expect("iter-2 build_v_eff");
    let iter2_arr = iter2_v
        .v_eff()
        .as_ref()
        .expect("v_eff present")
        .as_real_grid()
        .as_real_array()
        .clone();
    dump_array_3d(&iter2_arr, "/tmp/cu111_v_eff_iter2.bin");

    // Sanity: shapes match.
    assert_eq!(iter1_arr.shape(), pot_ref.shape(), "iter-1 shape mismatch");
    assert_eq!(iter2_arr.shape(), pot_ref.shape(), "iter-2 shape mismatch");
    let [ngx, ngy, ngz] = [iter1_arr.shape()[0], iter1_arr.shape()[1], iter1_arr.shape()[2]];
    eprintln!("[grid] {ngx}×{ngy}×{ngz}");

    // Global pointwise stats vs reference.
    let global_stats = |arr: &ndarray::Array3<f64>, label: &str| {
        let mut linf = 0.0_f64;
        let mut l2 = 0.0_f64;
        let mut sum_ref = 0.0_f64;
        let n = arr.len();
        for (a, &r) in arr.iter().zip(pot_ref.iter()) {
            let d = (a - r).abs();
            if d > linf {
                linf = d;
            }
            l2 += d * d;
            sum_ref += r * r;
        }
        l2 = (l2 / n as f64).sqrt();
        let l2_rel = (l2 * (n as f64)).sqrt() / sum_ref.sqrt();
        eprintln!(
            "[{label} vs pot_fmt] L_inf = {linf:.4e} Ha  L_2_rms = {l2:.4e} Ha  L_2_rel = {l2_rel:.4e}"
        );
        (linf, l2, l2_rel)
    };
    let (g1_inf, _, _) = global_stats(&iter1_arr, "iter-1");
    let (g2_inf, _, _) = global_stats(&iter2_arr, "iter-2");

    // Per-Cu-ion ROI: cells within 2.0 Bohr of each Cu nucleus.
    // Grid points are at fractional (ix/ngx, iy/ngy, iz/ngz). Convert to Cart.
    let lat_arr = [
        [real_lat[0][0], real_lat[0][1], real_lat[0][2]],
        [real_lat[1][0], real_lat[1][1], real_lat[1][2]],
        [real_lat[2][0], real_lat[2][1], real_lat[2][2]],
    ];

    // Find the first Cu ion (skip C and O if present).
    let cu_indices: Vec<usize> = (0..cell.num_ions)
        .filter(|&i| {
            let sp = cell.ion_species[i];
            cell.species_symbols[sp] == "Cu"
        })
        .collect();
    eprintln!("[ions] Cu ion indices (first 4 sampled): {:?}", &cu_indices[..cu_indices.len().min(4)]);

    let radius_bohr = 2.0;
    let radius_sq = radius_bohr * radius_bohr;

    // Use the lattice to convert fractional grid points to Cartesian.
    // Sample the first 3 Cu ions to keep output manageable.
    for &ion_idx in cu_indices.iter().take(3) {
        let frac = [
            cell.ionic_positions[[ion_idx, 0]],
            cell.ionic_positions[[ion_idx, 1]],
            cell.ionic_positions[[ion_idx, 2]],
        ];
        let r_ion = frac_to_cart(frac, &lat_arr);

        let mut roi_count = 0usize;
        let mut roi_iter1_inf = 0.0_f64;
        let mut roi_iter1_l2 = 0.0_f64;
        let mut roi_iter2_inf = 0.0_f64;
        let mut roi_iter2_l2 = 0.0_f64;
        let mut roi_iter1_minmax = (f64::INFINITY, f64::NEG_INFINITY);
        let mut roi_iter2_minmax = (f64::INFINITY, f64::NEG_INFINITY);
        let mut roi_pot_minmax = (f64::INFINITY, f64::NEG_INFINITY);

        // Search a bounding box (efficient — we only need ~4 Bohr cube).
        for ix in 0..ngx {
            for iy in 0..ngy {
                for iz in 0..ngz {
                    let frac_pt = [
                        ix as f64 / ngx as f64,
                        iy as f64 / ngy as f64,
                        iz as f64 / ngz as f64,
                    ];
                    // Apply minimum-image convention (PBC).
                    let mut dr_frac = [
                        frac_pt[0] - frac[0],
                        frac_pt[1] - frac[1],
                        frac_pt[2] - frac[2],
                    ];
                    for d in dr_frac.iter_mut() {
                        if *d > 0.5 { *d -= 1.0; }
                        if *d < -0.5 { *d += 1.0; }
                    }
                    let dr_cart = frac_to_cart(dr_frac, &lat_arr);
                    let d2 = dr_cart[0]*dr_cart[0] + dr_cart[1]*dr_cart[1] + dr_cart[2]*dr_cart[2];
                    if d2 > radius_sq { continue; }

                    let v1 = iter1_arr[[ix, iy, iz]];
                    let v2 = iter2_arr[[ix, iy, iz]];
                    let vr = pot_ref[[ix, iy, iz]];

                    let d1 = (v1 - vr).abs();
                    let d2v = (v2 - vr).abs();
                    if d1 > roi_iter1_inf { roi_iter1_inf = d1; }
                    if d2v > roi_iter2_inf { roi_iter2_inf = d2v; }
                    roi_iter1_l2 += d1 * d1;
                    roi_iter2_l2 += d2v * d2v;

                    if v1 < roi_iter1_minmax.0 { roi_iter1_minmax.0 = v1; }
                    if v1 > roi_iter1_minmax.1 { roi_iter1_minmax.1 = v1; }
                    if v2 < roi_iter2_minmax.0 { roi_iter2_minmax.0 = v2; }
                    if v2 > roi_iter2_minmax.1 { roi_iter2_minmax.1 = v2; }
                    if vr < roi_pot_minmax.0 { roi_pot_minmax.0 = vr; }
                    if vr > roi_pot_minmax.1 { roi_pot_minmax.1 = vr; }
                    roi_count += 1;
                }
            }
        }

        if roi_count > 0 {
            roi_iter1_l2 = (roi_iter1_l2 / roi_count as f64).sqrt();
            roi_iter2_l2 = (roi_iter2_l2 / roi_count as f64).sqrt();
        }
        eprintln!(
            "[ROI Cu ion={ion_idx} r<{radius_bohr}Bohr  N={roi_count}]\n\
             \treference V_eff: [{:.4} .. {:.4}] Ha\n\
             \titer-1   V_eff: [{:.4} .. {:.4}] Ha   diff_inf={:.4e} Ha  diff_rms={:.4e} Ha\n\
             \titer-2   V_eff: [{:.4} .. {:.4}] Ha   diff_inf={:.4e} Ha  diff_rms={:.4e} Ha",
            roi_pot_minmax.0, roi_pot_minmax.1,
            roi_iter1_minmax.0, roi_iter1_minmax.1, roi_iter1_inf, roi_iter1_l2,
            roi_iter2_minmax.0, roi_iter2_minmax.1, roi_iter2_inf, roi_iter2_l2,
        );
    }

    // Output a flush so the prints land before the test exits.
    std::io::stderr().flush().ok();

    eprintln!(
        "\n[summary]\n  iter-1 global L_inf = {g1_inf:.4e} Ha\n  iter-2 global L_inf = {g2_inf:.4e} Ha"
    );
}
