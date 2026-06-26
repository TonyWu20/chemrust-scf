#![cfg(feature = "chebyshev")]
//! Compare iter-1 output (ρ_PW upsampled + ρ_aug) vs CASTEP `.den_fmt` pointwise.
//!
//! This test localises whether the ρ_PW (smooth) or ρ_aug (augmentation)
//! component is responsible for the spatial mismatch that drives V_eff
//! at ion centres away from CASTEP's reference (~1-2 Ha shift, see
//! v_eff_pointwise_compare).
//!
//! Three comparisons:
//! - iter-1 (ρ_PW + ρ_aug) vs CASTEP total ρ (`.den_fmt`).
//! - iter-1 ρ_PW upsampled vs (CASTEP total ρ − iter-1 ρ_aug).
//!   If chemrust ρ_PW differs from "what's left after subtracting our aug",
//!   the smooth path (construct_density_gpu) is the culprit.
//! - iter-1 ρ_aug vs (CASTEP total ρ − iter-1 ρ_PW upsampled).
//!   The complementary localisation.

mod fixtures;

use std::io::Write;

#[allow(dead_code)]
fn gpu_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

fn frac_to_cart(frac: [f64; 3], real_lat: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i] += frac[j] * real_lat[j][i];
        }
    }
    c
}

fn dump_array_3d(arr: &ndarray::Array3<f64>, path: &str) {
    let bytes: Vec<u8> = arr.iter().flat_map(|&v| v.to_le_bytes()).collect();
    std::fs::write(path, &bytes).expect("write");
    eprintln!("[dump] wrote {} bytes to {path}", bytes.len());
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn iter1_density_pointwise_vs_den_fmt() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }

    let fx = fixtures::cu111_co::fixture();
    let cell = &fx.bin.cell;
    let real_lat = cell.real_lattice.as_array();

    // Reference: CASTEP `.den_fmt` total density on the fine grid (raw ρ × Ω units).
    let den_fmt_arr = fx.den_fmt.charge.as_real_grid().as_real_array().clone();
    let den_min = den_fmt_arr.iter().cloned().fold(f64::INFINITY, f64::min);
    let den_max = den_fmt_arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let den_sum: f64 = den_fmt_arr.iter().sum();
    eprintln!(
        "[fixture .den_fmt] shape={:?} sum={:.4e} min={:.4} max={:.4}",
        den_fmt_arr.shape(), den_sum, den_min, den_max
    );

    // Build state and run iter-1.
    let ctx = cudarc::driver::CudaContext::new(0).expect("cuda");
    let stream = std::sync::Arc::new(ctx.default_stream());
    let state = fixtures::cu111_co::build_scf_state(fx, &stream);
    let iter1_v = state.build_v_eff_with_energy().expect("iter-1 build_v_eff");
    let iter1_diag = iter1_v.diagonalize(8, None).expect("iter-1 diagonalize");
    let iter1_dens_state = iter1_diag.construct_density_off().expect("construct_density");

    // After the fine-grid mixing refactor, construct_density_off combines soft+aug
    // on the fine grid and stores the combined density.  density() returns the
    // combined fine-grid density directly — no separate upsampling step needed.
    let rho_total_iter1_arr = iter1_dens_state.density().as_fine_array().clone();

    assert_eq!(rho_total_iter1_arr.shape(), den_fmt_arr.shape(),
        "shape mismatch: iter1 combined {:?} vs den_fmt {:?}",
        rho_total_iter1_arr.shape(), den_fmt_arr.shape());

    // Aug is baked into the density — verify it's absent from density_aug_fine.
    assert!(iter1_dens_state.density_aug_fine().is_none(),
        "after combine_soft_aug_on_fine, density_aug_fine must be None");

    let rho_total_iter1 = rho_total_iter1_arr;
    let total_sum: f64 = rho_total_iter1.iter().sum();
    eprintln!(
        "[iter-1 output] combined soft+aug on fine grid, total sum={total_sum:.4e}"
    );

    // Dump for offline analysis.
    dump_array_3d(&rho_total_iter1, "/tmp/cu111_iter1_rho_combined.bin");
    dump_array_3d(&den_fmt_arr, "/tmp/cu111_castep_rho_total.bin");

    // ---- Global pointwise comparisons ----
    let l_inf_l2 = |arr: &ndarray::Array3<f64>, ref_arr: &ndarray::Array3<f64>| -> (f64, f64, f64) {
        let mut linf = 0.0_f64;
        let mut l2 = 0.0_f64;
        let mut sum_ref_sq = 0.0_f64;
        let n = arr.len();
        for (a, &r) in arr.iter().zip(ref_arr.iter()) {
            let d = (a - r).abs();
            if d > linf { linf = d; }
            l2 += d * d;
            sum_ref_sq += r * r;
        }
        let l2_rms = (l2 / n as f64).sqrt();
        let l2_rel = (l2 / sum_ref_sq).sqrt();
        (linf, l2_rms, l2_rel)
    };

    eprintln!("\n=== Global pointwise comparisons ===");
    let (linf, l2, l2rel) = l_inf_l2(&rho_total_iter1, &den_fmt_arr);
    eprintln!("[iter-1 total ρ vs CASTEP .den_fmt]   L_inf={linf:.4e}  L_2_rms={l2:.4e}  L_2_rel={l2rel:.4e}");

    // ---- Per-Cu-ion ROI ----
    let lat_arr = [
        [real_lat[0][0], real_lat[0][1], real_lat[0][2]],
        [real_lat[1][0], real_lat[1][1], real_lat[1][2]],
        [real_lat[2][0], real_lat[2][1], real_lat[2][2]],
    ];
    let cu_indices: Vec<usize> = (0..cell.num_ions)
        .filter(|&i| cell.species_symbols[cell.ion_species[i]] == "Cu")
        .collect();

    let radius_bohr = 2.0_f64;
    let radius_sq = radius_bohr * radius_bohr;
    let [ngx, ngy, ngz] = [
        den_fmt_arr.shape()[0],
        den_fmt_arr.shape()[1],
        den_fmt_arr.shape()[2],
    ];

    eprintln!("\n=== Per-Cu-ion ROI (r < {radius_bohr} Bohr) ===");

    for &ion_idx in cu_indices.iter().take(3) {
        let frac = [
            cell.ionic_positions[[ion_idx, 0]],
            cell.ionic_positions[[ion_idx, 1]],
            cell.ionic_positions[[ion_idx, 2]],
        ];

        let mut n_roi = 0usize;
        let mut sum_ref = 0.0_f64;
        let mut sum_total = 0.0_f64;
        let mut max_ref = f64::NEG_INFINITY;
        let mut max_total = f64::NEG_INFINITY;
        let mut diff_total_inf = 0.0_f64;
        let mut sum_diff_total_sq = 0.0_f64;

        for ix in 0..ngx {
            for iy in 0..ngy {
                for iz in 0..ngz {
                    let frac_pt = [ix as f64 / ngx as f64, iy as f64 / ngy as f64, iz as f64 / ngz as f64];
                    let mut dr_frac = [frac_pt[0] - frac[0], frac_pt[1] - frac[1], frac_pt[2] - frac[2]];
                    for d in dr_frac.iter_mut() {
                        if *d > 0.5 { *d -= 1.0; }
                        if *d < -0.5 { *d += 1.0; }
                    }
                    let dr_cart = frac_to_cart(dr_frac, &lat_arr);
                    let d2 = dr_cart[0]*dr_cart[0] + dr_cart[1]*dr_cart[1] + dr_cart[2]*dr_cart[2];
                    if d2 > radius_sq { continue; }

                    let rref = den_fmt_arr[[ix, iy, iz]];
                    let rtot = rho_total_iter1[[ix, iy, iz]];

                    sum_ref += rref;
                    sum_total += rtot;
                    if rref > max_ref { max_ref = rref; }
                    if rtot > max_total { max_total = rtot; }

                    let dt = (rtot - rref).abs();
                    if dt > diff_total_inf { diff_total_inf = dt; }
                    sum_diff_total_sq += dt * dt;

                    n_roi += 1;
                }
            }
        }

        let inv_n = if n_roi > 0 { 1.0 / n_roi as f64 } else { 0.0 };
        let total_rms = (sum_diff_total_sq * inv_n).sqrt();

        eprintln!(
            "[ROI Cu ion={ion_idx}  N={n_roi}]\n\
             \tsum: ref={:.3e}  combined={:.3e}\n\
             \tmax: ref={:.3e}  combined={:.3e}\n\
             \tdiff (combined vs ref): inf={:.3e}  rms={:.3e}",
            sum_ref, sum_total,
            max_ref, max_total,
            diff_total_inf, total_rms,
        );
    }

    std::io::stderr().flush().ok();
}
