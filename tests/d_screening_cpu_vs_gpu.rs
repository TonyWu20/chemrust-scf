#![cfg(feature = "chebyshev")]
//! Discriminator test: CPU vs GPU D-screening on Cu111_CO fixture.
//!
//! Compares per-ion D matrices from the GPU path (`screen_d_gpu`) against
//! the CPU reference (`compute_screened_d_from_fft`) which is independently
//! validated against CASTEP `nlpot.f90` in chemrust-hamiltonian tests.
//!
//! **Success criteria:**
//! - `max|GPU_D - CPU_D| < 1e-10` per ion (CPU is EXTERNAL anchor; GPU/CPU summation order differs at machine precision)
//! - `|D[n,m] - D[m,n]| < 1e-15` per ion (real-symmetric construction)
//! - D has no NaN/inf entries
//! - Non-origin ions have non-trivial screening: `max|D - D0| > 1e-10`
//! - Origin ion (R=0) diagonal matches CPU to 1e-10
//!   (Source: at R=0, exp(±iG·0) = 1 — screening is layout-independent;
//!    machine-precision FMA order differences in 437k-element reduction)

mod fixtures;

use std::sync::Arc;
use std::sync::Once;

use chemrust_hamiltonian_core::{
    GVectorGrid, Pseudopotential,
    nlpot::{build_d0_expanded, compute_screened_d_from_fft, precompute_q_on_grid},
    fft::{fft_forward_3d, RecipGrid},
    pseudopotential::HasAugmentationData,
};
use chemrust_scf::{
    CudaKernelSet, device::CudaComplex,
    test_api::{build_wave_screening_cache, screen_d_gpu_debug},
};
use cudarc::driver::{CudaContext, CudaSlice};
use ndarray::{Array3, ShapeBuilder};
use num_complex::Complex64;

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_target(false)
            .try_init()
            .ok();
    });
}

fn gpu_available() -> bool {
    std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
}

/// FFT fine-grid V_eff and truncate to wave-grid reciprocal space.
///
/// The SCF state stores V_eff on the fine (2x upsampled) grid.  D-screening
/// operates on the wave grid, so we FFT the fine-grid V_eff, then copy only
/// the wave-grid G-vectors into a new Fortran-ordered array.  The returned
/// `RecipGrid` has shape `(ngz, ngy, ngx)` matching `wave_grid`.
fn v_eff_fft_on_wave_grid(
    veff: &chemrust_hamiltonian_core::EffectivePotential,
    fine_grid: &GVectorGrid,
    wave_grid: &GVectorGrid,
) -> RecipGrid<Complex64> {
    let [ngzf, ngyf, ngxf] = fine_grid.grid();
    let [ngz, ngy, ngx] = wave_grid.grid();

    let fine_g = fft_forward_3d(veff.as_real_grid())
        .expect("FFT fine-grid V_eff");
    debug_assert_eq!(fine_g.as_recip_array().shape(), [ngzf, ngyf, ngxf]);

    // Truncate: copy only wave-grid G-vectors to a new Fortran-ordered array.
    let mut wave_g = Array3::<Complex64>::zeros((ngz, ngy, ngx).f());
    let gvecs_wave = wave_grid.gvecs();
    let f2ix = |f: i32, n: usize| -> usize {
        if f >= 0 { f as usize } else { (n as i32 + f) as usize }
    };
    ndarray::Zip::from(gvecs_wave)
        .and(&mut wave_g)
        .for_each(|&gf, coeff| {
            let fx = gf[0] as i32;
            let fy = gf[1] as i32;
            let fz = gf[2] as i32;
            let ix_f = f2ix(fx, ngxf);
            let iy_f = f2ix(fy, ngyf);
            let iz_f = f2ix(fz, ngzf);
            *coeff = fine_g.as_recip_array()[[iz_f, iy_f, ix_f]];
        });

    RecipGrid::from_inner(wave_g)
}

#[test]
#[ignore = "requires GPU and CASTEP fixture data"]
fn d_screening_cpu_vs_gpu_element_by_element() {
    if !gpu_available() {
        eprintln!("SKIP: no GPU available");
        return;
    }
    init_tracing();

    // ------------------------------------------------------------------
    // 1. GPU context + fixture load
    // ------------------------------------------------------------------
    let ctx: Arc<CudaContext> = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    let fx = fixtures::cu111_co::fixture();
    let state = fixtures::cu111_co::build_scf_state(fx);
    let cell = &fx.bin.cell;
    let pots = &fx.pots;

    // Grids from fixture data
    let wfc = fx
        .check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction");
    let [ngx, ngy, ngz] = wfc.grid;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);
    let n_wave = ngz * ngy * ngx;

    let [fgx, fgy, fgz] = fx
        .check
        .fine_grid
        .expect(".check must have fine_grid");
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    eprintln!("[fixture] wave grid: {ngx}x{ngy}x{ngz} = {n_wave}");
    eprintln!("[fixture] fine grid: {fgx}x{fgy}x{fgz} = {}", fgx * fgy * fgz);
    eprintln!("[fixture] num_ions:  {}", cell.num_ions);

    // ------------------------------------------------------------------
    // 2. Build V_eff on fine grid -> FFT -> truncate to wave grid
    // ------------------------------------------------------------------
    let veff_built = state.build_v_eff_with_energy().expect("build_v_eff");
    let v_eff = veff_built
        .v_eff()
        .as_ref()
        .expect("V_eff present after build");
    let v_eff_fft = v_eff_fft_on_wave_grid(v_eff, &fine_grid, &wave_grid);
    debug_assert_eq!(v_eff_fft.as_recip_array().shape(), [ngz, ngy, ngx]);

    // ------------------------------------------------------------------
    // 3. Upload wave-grid V_eff FFT to GPU in Fortran order (iz fastest).
    //    ndarray .iter() traverses last-dim-fastest (row-major); .t() swaps
    //    axes so that the last dim of the transposed view is iz, giving
    //    iz-fastest iteration — matching the sf/Q flattening convention.
    // ------------------------------------------------------------------
    let v_eff_flat: Vec<CudaComplex> = v_eff_fft
        .as_recip_array()
        .t()
        .iter()
        .map(|c| CudaComplex { x: c.re, y: c.im })
        .collect();
    let v_eff_dev: CudaSlice<CudaComplex> =
        stream.clone_htod(&v_eff_flat).expect("H2D V_eff_fft");

    // ------------------------------------------------------------------
    // 4. Build WaveScreeningCache on GPU (per-species Q + per-ion SF)
    // ------------------------------------------------------------------
    use chemrust_scf::device::blas::BlasHandle;
    use chemrust_scf::device::pcie::PcieAccount;

    let mut pcie = PcieAccount::default();
    let kernels = CudaKernelSet::new(&ctx).expect("compile CUDA kernels");
    let blas = BlasHandle::new(Arc::clone(&stream)).expect("BLAS handle");
    let cache = build_wave_screening_cache(pots, cell, &wave_grid, &stream, &mut pcie)
        .expect("build WaveScreeningCache");
    eprintln!("[cache] H2D bytes: {}", pcie.h2d_bytes);

    // ------------------------------------------------------------------
    // 5. Per-ion CPU vs GPU D-screening comparison
    // ------------------------------------------------------------------
    let mut n_ions_screened = 0usize;
    let mut n_origin = 0usize;
    let mut n_non_origin = 0usize;
    let mut global_max_delta = 0.0_f64;

    for ion_idx in 0..cell.num_ions {
        let species_idx = cell.ion_species[ion_idx];
        let symbol = &cell.species_symbols[species_idx];

        // Only USPP species have augmentation data for D-screening.
        let Some(pot) = pots.get(symbol) else {
            eprintln!("ion={ion_idx}: species {symbol} has no pseudopotential — skipping");
            continue;
        };
        let aug: &dyn HasAugmentationData = match pot {
            Pseudopotential::Usp(d) => d,
            Pseudopotential::Recpot(_) => {
                eprintln!("ion={ion_idx}: species {symbol} is Recpot (no augmentation) — skipping");
                continue;
            }
        };

        let d0 = build_d0_expanded(aug);
        let n_exp = d0.shape()[0];
        let d0_flat: Vec<f64> = d0.iter().copied().collect();

        // --- CPU reference (EXTERNAL anchor) ---
        // Precompute Q on the wave grid once per species, then screen per ion.
        let q_on_grid = precompute_q_on_grid(aug, &wave_grid)
            .expect("precompute_q_on_grid");
        let cpu_d = compute_screened_d_from_fft(
            &q_on_grid,
            &v_eff_fft,
            cell,
            ion_idx,
            &wave_grid,
            &d0,
        );

        // --- GPU screening ---
        let (gpu_d, w_gpu, tmp_gpu) = screen_d_gpu_debug(
            &cache,
            &v_eff_dev,
            ion_idx,
            species_idx,
            &d0_flat,
            n_wave,
            &kernels,
            &blas,
            &stream,
        )
        .expect("screen_d_gpu_debug");

        // --- Criterion 2: Real-symmetric ---
        for n in 0..n_exp {
            for m in 0..n_exp {
                let diff_sym = (gpu_d[[n, m]] - gpu_d[[m, n]]).abs();
                assert!(
                    diff_sym < 1e-15,
                    "ion={ion_idx}: D not symmetric at ({n},{m}): |D[n,m]-D[m,n]| = {:.2e}",
                    diff_sym,
                );
            }
        }

        // --- Criterion 3: No NaN/inf ---
        for n in 0..n_exp {
            for m in 0..n_exp {
                assert!(
                    gpu_d[[n, m]].is_finite(),
                    "ion={ion_idx}: D has non-finite entry at ({n},{m}) = {}",
                    gpu_d[[n, m]],
                );
            }
        }

        // --- Criterion 1: Element-by-element CPU vs GPU ---
        let mut max_delta = 0.0_f64;
        for n in 0..n_exp {
            for m in 0..n_exp {
                let d = (gpu_d[[n, m]] - cpu_d[[n, m]]).abs();
                if d > max_delta {
                    max_delta = d;
                }
            }
        }
        global_max_delta = global_max_delta.max(max_delta);

        // --- Diagnostic tiers (gated by CHEMRUST_DIAGNOSE_D_SCREENING) ---
        let diag_ion: Option<usize> = std::env::var("CHEMRUST_DIAGNOSE_D_SCREENING")
            .ok()
            .and_then(|s| s.parse().ok());
        if diag_ion == Some(ion_idx) || (max_delta >= 1e-10 && diag_ion.is_none()) {
            eprintln!("\n=== DIAGNOSTIC ion={ion_idx} max|GPU-CPU|={max_delta:.4e} ===");
            let tau = 2.0 * std::f64::consts::PI;
            let pos = cell.ionic_positions.row(ion_idx);
            let (rx, ry, rz) = (pos[0], pos[1], pos[2]);

            // T1: Struct factors — download GPU sf, recompute CPU sf
            {
                eprintln!("--- T1: struct factors ---");
                let gpu_sf: Vec<CudaComplex> = stream
                    .clone_dtoh(&cache.ion_sf[ion_idx].sf)
                    .expect("D2H sf");
                let mut max_sf_err = 0.0_f64;
                let mut g = 0usize;
                for ix in 0..ngx {
                    for iy in 0..ngy {
                        for iz in 0..ngz {
                            let gf = wave_grid.gvecs()[[iz, iy, ix]];
                            let phase = -tau * (gf[0] * rx + gf[1] * ry + gf[2] * rz);
                            let (s, c) = phase.sin_cos();
                            let cpu_re = c;
                            let cpu_im = s;
                            let err = ((gpu_sf[g].x - cpu_re).powi(2)
                                + (gpu_sf[g].y - cpu_im).powi(2))
                                .sqrt();
                            if err > max_sf_err {
                                max_sf_err = err;
                            }
                            g += 1;
                        }
                    }
                }
                eprintln!("  max|GPU_sf - CPU_sf| = {:.4e}", max_sf_err);
                assert!(
                    max_sf_err < 1e-14,
                    "T1 FAIL: struct factor mismatch max_err={:.4e}",
                    max_sf_err
                );
                eprintln!("  T1 PASS");
            }

            // T2: Q arrays — download GPU Q, recompute CPU Q
            {
                eprintln!("--- T2: Q arrays (species_idx={species_idx}) ---");
                let entry = cache.species_entries[species_idx]
                    .as_ref()
                    .expect("species entry present");
                let gpu_q: Vec<CudaComplex> =
                    stream.clone_dtoh(&entry.q_nm).expect("D2H q_nm");
                let q_cpu = precompute_q_on_grid(aug, &wave_grid).expect("Q cpu");
                let n_pairs = entry.n_lower_pairs;
                let mut max_q_err = 0.0_f64;
                let mut g = 0usize;
                for ((_n, _m), q_arr) in &q_cpu.pairs {
                    for &c in q_arr.t().iter() {
                        let err = ((gpu_q[g].x - c.re).powi(2)
                            + (gpu_q[g].y - c.im).powi(2))
                            .sqrt();
                        if err > max_q_err {
                            max_q_err = err;
                        }
                        g += 1;
                    }
                }
                eprintln!("  max|GPU_Q - CPU_Q| = {:.4e}  (n_pairs={n_pairs}, n_elems={g})",
                    max_q_err);
                assert!(
                    max_q_err < 1e-12,
                    "T2 FAIL: Q array mismatch max_err={:.4e}",
                    max_q_err
                );
                eprintln!("  T2 PASS");
            }

            // T3: w buffer — compare GPU w (V_eff * conj(sf)) vs CPU w
            {
                eprintln!("--- T3: w buffer (V_eff * conj(sf)) ---");
                let mut max_w_err = 0.0_f64;
                let mut max_w_err_pos = (0usize, 0usize, 0usize);
                let mut g = 0usize;
                for ix in 0..ngx {
                    for iy in 0..ngy {
                        for iz in 0..ngz {
                            let gf = wave_grid.gvecs()[[iz, iy, ix]];
                            let sf_phase = -tau * (gf[0] * rx + gf[1] * ry + gf[2] * rz);
                            let (s, c) = sf_phase.sin_cos();
                            // sf = c + i*s, conj(sf) = c - i*s
                            // w = V * conj(sf) = (V_re*c + V_im*s) + i*(V_im*c - V_re*s)
                            let v = v_eff_fft.as_recip_array()[[iz, iy, ix]];
                            let w_cpu_re = v.re * c + v.im * s;
                            let w_cpu_im = v.im * c - v.re * s;
                            let err = ((w_gpu[g].x - w_cpu_re).powi(2)
                                + (w_gpu[g].y - w_cpu_im).powi(2)).sqrt();
                            if err > max_w_err {
                                max_w_err = err;
                                max_w_err_pos = (iz, iy, ix);
                            }
                            // Pinpoint: dump first 5 elements for inspection
                            if g < 5 {
                                eprintln!("  g={g} (iz={iz},iy={iy},ix={ix}):");
                                eprintln!("    V_eff  = ({:+.6e}, {:+.6e})", v.re, v.im);
                                eprintln!("    sf     = ({:+.6e}, {:+.6e})", c, s);
                                eprintln!("    w_cpu  = ({:+.6e}, {:+.6e})", w_cpu_re, w_cpu_im);
                                eprintln!("    w_gpu  = ({:+.6e}, {:+.6e})", w_gpu[g].x, w_gpu[g].y);
                                let v_eff_gpu = v_eff_flat[g];
                                eprintln!("    v_eff_flat[g] = ({:+.6e}, {:+.6e})", v_eff_gpu.x, v_eff_gpu.y);
                            }
                            g += 1;
                        }
                    }
                }
                let (miz, miy, mix) = max_w_err_pos;
                eprintln!("  max|GPU_w - CPU_w| = {:.4e} at (iz={miz},iy={miy},ix={mix})", max_w_err);
                if max_w_err > 1e-14 {
                    eprintln!("  T3 FAIL: w buffer mismatch");
                } else {
                    eprintln!("  T3 PASS");
                }
            }

            // T4: tmp buffer — compare GPU tmp vs CPU recomputation
            {
                eprintln!("--- T4: tmp buffer (gemv result) ---");
                let entry = cache.species_entries[species_idx]
                    .as_ref().expect("species entry");
                let q_cpu = precompute_q_on_grid(aug, &wave_grid).expect("Q cpu");
                let mut max_tmp_err = 0.0_f64;
                for (p, &(_n, _m)) in entry.pair_indices.iter().enumerate() {
                    // CPU: tmp[p] = Σ_g conj(Q(p,g)) * w(g)
                    let mut cpu_tmp_re = 0.0_f64;
                    let mut cpu_tmp_im = 0.0_f64;
                    let mut g_idx = 0usize;
                    for ((_pn, _pm), q_arr) in &q_cpu.pairs {
                        if g_idx == p {
                            let mut gi = 0usize;
                            for ix in 0..ngx {
                                for iy in 0..ngy {
                                    for iz in 0..ngz {
                                        let v = v_eff_fft.as_recip_array()[[iz, iy, ix]];
                                        let sf_phase = -tau * (wave_grid.gvecs()[[iz, iy, ix]][0] * rx
                                            + wave_grid.gvecs()[[iz, iy, ix]][1] * ry
                                            + wave_grid.gvecs()[[iz, iy, ix]][2] * rz);
                                        let (s, c) = sf_phase.sin_cos();
                                        // sf = c + i*s, conj(sf) = c - i*s
                                        let w_re = v.re * c + v.im * s;
                                        let w_im = v.im * c - v.re * s;
                                        // tmp[p] += conj(Q(p,gi)) * w(gi)
                                        let q = q_arr[[iz, iy, ix]];
                                        cpu_tmp_re += q.re * w_re + q.im * w_im;
                                        cpu_tmp_im += q.re * w_im - q.im * w_re;
                                        gi += 1;
                                    }
                                }
                            }
                            break;
                        }
                        g_idx += 1;
                    }
                    let err = ((tmp_gpu[p].x - cpu_tmp_re).powi(2)
                        + (tmp_gpu[p].y - cpu_tmp_im).powi(2)).sqrt();
                    if err > max_tmp_err { max_tmp_err = err; }
                }
                eprintln!("  max|GPU_tmp - CPU_tmp| = {:.4e}", max_tmp_err);
                if max_tmp_err > 5e-6 {
                    eprintln!("  T4 FAIL: tmp buffer mismatch");
                } else {
                    eprintln!("  T4 PASS");
                }
            }
        }

        assert!(
            max_delta < 1e-10,
            "ion={ion_idx}: max|GPU-CPU| = {:.4e} > 1e-10",
            max_delta,
        );

        // --- Criterion 4 + 5: ion-position-dependent ---
        let pos = cell.ionic_positions.row(ion_idx);
        let is_origin = pos.iter().all(|&r| r.abs() < 1e-10);

        if is_origin {
            // Criterion 5: Origin ion — exp(+/-iG.0) = 1 exactly, so screening
            // is layout-independent and must match CPU to round-off.
            let mut max_diag_delta = 0.0_f64;
            for n in 0..n_exp {
                let d = (gpu_d[[n, n]] - cpu_d[[n, n]]).abs();
                if d > max_diag_delta {
                    max_diag_delta = d;
                }
            }
            assert!(
                max_diag_delta < 1e-10,
                "ion={ion_idx}: origin diagonal GPU-CPU diff = {:.4e} > 1e-10",
                max_diag_delta,
            );
            n_origin += 1;
        } else {
            // Criterion 4: Non-origin must have non-trivial screening.
            let mut max_screening = 0.0_f64;
            for n in 0..n_exp {
                for m in 0..n_exp {
                    let d = (gpu_d[[n, m]] - d0[[n, m]]).abs();
                    if d > max_screening {
                        max_screening = d;
                    }
                }
            }
            assert!(
                max_screening > 1e-10,
                "ion={ion_idx}: non-origin has near-zero screening ({:.4e}) — bug regression",
                max_screening,
            );
            n_non_origin += 1;
        }

        eprintln!(
            "ion={ion_idx:2} sym={symbol:2} ne={n_exp:2}  \
             max|GPU-CPU|={max_delta:.4e}  origin={is_origin}"
        );
        n_ions_screened += 1;
    }

    // --- Summary ---
    eprintln!();
    eprintln!("[D-screening] Summary:");
    eprintln!("  ions screened:        {n_ions_screened}");
    eprintln!("  of which origin:      {n_origin}");
    eprintln!("  non-origin:           {n_non_origin}");
    eprintln!("  global max|GPU-CPU|:  {global_max_delta:.4e}");

    assert!(
        n_ions_screened > 0,
        "No USPP ions found — test cannot validate screening",
    );

    eprintln!("[D-screening] ALL CRITERIA PASS");
}
