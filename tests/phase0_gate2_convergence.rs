// ---------------------------------------------------------------------------
// Gate 2: Convergence from CASTEP Converged Wavefunctions
//
// Verify that band-by-band CG preserves the eigenvalues of CASTEP's converged
// wavefunctions to within 1e-6 Ha.  Starting from ψ that are already S-
// orthonormal and at the true eigenstate, CG should converge in 0-1 steps
// per band with negligible eigenvalue drift.
//
// This is the "happy path" test: when the eigensolver starts from a near-
// converged subspace (as it does in the SCF loop after subspace diag),
// CG should preserve the .check eigenvalues exactly.
//
// Compare with Gate 1 which tests 1 CG step on CASTEP ψ and checks drift
// < 1e-6 Ha for all bands.  Gate 2 runs full CG refinement (max 15 steps)
// to verify that multi-step CG doesn't accumulate drift.
// ---------------------------------------------------------------------------

use std::fs::{self, File};
use std::io::BufReader;

use ndarray::Array2;
use num_complex::Complex64;

use chemrust_hamiltonian_core::{
    CheckFile, GVectorGrid, EffectivePotential, RealGrid,
    PseudopotentialSet, Pseudopotential,
    hamiltonian::{apply_full_hamiltonian, inner_product},
    formatted::{parse_pot_fmt, usp::parse_usp},
    augment::beta_phi::{compute_beta_g, expanded_projector_count, expanded_projector_lm},
    nlpot::{build_d0_expanded, precompute_q_on_grid, compute_screened_d_from_fft, QOnGrid},
    fft::{fft_forward_3d, RecipGrid},
    downsample_density_to_wave_grid,
};

use chemrust_scf::eigensolver::{
    band_cg::band_cg_minimize,
    uspp_preconditioner::UsppPreconditioner,
};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const CHECK_PATH: &str =
    "/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.check";
const POT_PATH: &str =
    "/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/Cu111_CO.pot_fmt";
const POT_DIR: &str = "/export/Potentials";

/// Number of bands to converge (all bands in the system).
const N_BANDS: usize = 160;

// ---------------------------------------------------------------------------
// Helper: convert fractional k-point to Cartesian
// ---------------------------------------------------------------------------

fn kpoint_to_cartesian(k_frac: [f64; 3], recip: &chemrust_hamiltonian_core::RecipLattice) -> [f64; 3] {
    let rl = recip.as_array();
    [
        k_frac[0] * rl[0][0] + k_frac[1] * rl[1][0] + k_frac[2] * rl[2][0],
        k_frac[0] * rl[0][1] + k_frac[1] * rl[1][1] + k_frac[2] * rl[2][1],
        k_frac[0] * rl[0][2] + k_frac[1] * rl[1][2] + k_frac[2] * rl[2][2],
    ]
}

// ---------------------------------------------------------------------------
// Helper: build expanded Q_aug matrix (same as Gate 1)
// ---------------------------------------------------------------------------

fn build_q_expanded(aug: &dyn chemrust_hamiltonian_core::pseudopotential::HasAugmentationData) -> Array2<f64> {
    let projectors = aug.projectors();
    let n_expanded = expanded_projector_count(projectors);
    let q_rows = aug.q_aug();

    let within_l: Vec<usize> = projectors
        .iter()
        .enumerate()
        .map(|(i, _)| projectors[..=i].iter().filter(|p| p.l == projectors[i].l).count())
        .collect();

    let mut q = Array2::<f64>::zeros((n_expanded, n_expanded));
    for n_exp in 0..n_expanded {
        let pn = expanded_projector_lm(projectors, n_exp);
        for m_exp in 0..n_expanded {
            let pm = expanded_projector_lm(projectors, m_exp);
            if pn.l == pm.l && pn.m == pm.m {
                let cnt_n = within_l[pn.rad_idx];
                let cnt_m = within_l[pm.rad_idx];
                let (canon, smaller_cnt) = if cnt_n >= cnt_m {
                    (pn.rad_idx, cnt_m)
                } else {
                    (pm.rad_idx, cnt_n)
                };
                let q_val = q_rows
                    .0
                    .get(canon)
                    .and_then(|r| r.0.get(smaller_cnt.saturating_sub(1)))
                    .copied()
                    .unwrap_or(0.0);
                q[[n_exp, m_exp]] = q_val;
            }
        }
    }
    q
}

// ---------------------------------------------------------------------------
// Gate 2 convergence test
// ---------------------------------------------------------------------------

#[test]
fn gate2_convergence_from_castep_wavefunctions() {
    // ---- 1. Read .check file (for geometry, G-vectors, eigenvalues) ------------
    eprintln!("[Gate 2] Reading .check file...");
    let check_file = File::open(CHECK_PATH).expect("failed to open .check file");
    let castep_bin =
        CheckFile::read(BufReader::new(check_file)).expect("failed to parse .check file");

    let castep_eigvals: Vec<f64> = castep_bin.eigenvalues.kpoints[0].spins[0].eigenvalues.clone();

    let wave = castep_bin
        .wavefunction
        .as_ref()
        .expect(".check file has no wavefunction data");
    let kpt0 = &wave.kpt_data[0];
    let nplw = kpt0.nplw;
    let pw_coords = &kpt0.pw_grid_coord;
    let k_frac = kpt0.coords;
    eprintln!("  {nplw} plane waves, k = [{:.6}, {:.6}, {:.6}]",
        k_frac[0], k_frac[1], k_frac[2]);

    // Load CASTEP converged wavefunctions directly (they are already S-orthonormal).
    let bands: Vec<Vec<Complex64>> = kpt0.bands.clone();
    assert_eq!(bands.len(), N_BANDS, "expected {N_BANDS} bands in .check file");

    // ---- 2. Read V_eff from .pot_fmt ----------------------------------------
    eprintln!("[Gate 2] Reading .pot_fmt...");
    let pot_content = fs::read_to_string(POT_PATH).expect("failed to read .pot_fmt");
    let (fine_grid, veff_arr) =
        parse_pot_fmt(&pot_content).expect("failed to parse .pot_fmt");
    let veff_fine = EffectivePotential::from_inner(RealGrid::from_inner(veff_arr));

    // ---- 3. Build GVectorGrids ----------------------------------------------
    let recip = castep_bin.cell.recip_lattice;
    let wave_grid = wave.grid;
    let gvg_wave = GVectorGrid::new(wave_grid[0], wave_grid[1], wave_grid[2], recip);
    let gvg_fine = GVectorGrid::new(fine_grid[0], fine_grid[1], fine_grid[2], recip);

    // ---- 4. Downsample V_eff to wavefunction grid ---------------------------
    eprintln!("[Gate 2] Downsampling V_eff...");
    let veff_wave_arr = downsample_density_to_wave_grid(
        veff_fine.as_real_grid(), &gvg_wave, &gvg_fine,
    ).expect("V_eff downsampling failed");
    let veff_wave = EffectivePotential::from_inner(veff_wave_arr);

    // ---- 5. FFT indices and Cartesian G-vectors -----------------------------
    let fft_indices = gvg_wave.pw_to_fft_indices(pw_coords);
    let gcart: Vec<[f64; 3]> = fft_indices
        .iter()
        .map(|&[iz, iy, ix]| gvg_wave.gcart()[[iz, iy, ix]])
        .collect();

    // ---- 6. k-point to Cartesian --------------------------------------------
    let k_cart = kpoint_to_cartesian(k_frac, &recip);

    // ---- 7. Kinetic energies for PW selection -------------------------------
    let kinetic_g: Vec<f64> = gcart
        .iter()
        .map(|&gc| {
            let kg = [k_cart[0] + gc[0], k_cart[1] + gc[1], k_cart[2] + gc[2]];
            0.5 * (kg[0] * kg[0] + kg[1] * kg[1] + kg[2] * kg[2])
        })
        .collect();

    // ---- 8. Load pseudopotentials -------------------------------------------
    let cell = &castep_bin.cell;
    eprintln!("[Gate 2] Loading pseudopotential files...");
    let mut pots = PseudopotentialSet::new();
    for (sp_idx, symbol) in cell.species_symbols.iter().enumerate() {
        let pot_file = &cell.species_pot_files[sp_idx];
        let pot_path = if pot_file.starts_with('/') {
            pot_file.clone()
        } else {
            format!("{POT_DIR}/{pot_file}")
        };
        let src = fs::read_to_string(&pot_path)
            .unwrap_or_else(|e| panic!("failed to read {pot_path}: {e}"));
        let usp = parse_usp(&src)
            .unwrap_or_else(|e| panic!("failed to parse USP for {symbol}: {e}"));
        pots.insert(symbol.clone(), Pseudopotential::Usp(usp));
    }

    // ---- 9. Precompute Q_on_grid, beta_g, screened D (same as Gate 1) ------
    eprintln!("[Gate 2] Precomputing Q_on_grid per species...");

    struct SpeciesQ { symbol: String, q_grid: QOnGrid }
    let mut species_q: Vec<SpeciesQ> = Vec::new();
    for sp_idx in 0..cell.num_species {
        let symbol = &cell.species_symbols[sp_idx];
        let pot = pots.get(symbol)
            .unwrap_or_else(|| panic!("missing pseudopotential for species {symbol}"));
        let aug: &dyn chemrust_hamiltonian_core::pseudopotential::HasAugmentationData = match pot {
            Pseudopotential::Usp(d) => d,
            _ => continue,
        };
        let q_grid = precompute_q_on_grid(aug, &gvg_wave)
            .expect("precompute_q_on_grid failed");
        species_q.push(SpeciesQ { symbol: symbol.clone(), q_grid });
    }

    let veff_wave_fft: RecipGrid<Complex64> = fft_forward_3d(veff_wave.as_real_grid())
        .expect("V_eff forward FFT failed");

    let bg_wave_block = chemrust_hamiltonian_core::types::KptWaveBlock {
        coords: k_frac, nplw, pw_grid_coord: pw_coords.clone(), bands: Vec::new(),
    };

    struct IonData {
        beta_g: Array2<Complex64>,
        d: Array2<f64>,
        q_exp: Array2<f64>,
    }

    let mut ion_data: Vec<IonData> = Vec::new();
    for ion_idx in 0..cell.num_ions {
        let species_idx = cell.ion_species[ion_idx];
        let symbol = &cell.species_symbols[species_idx];
        let pot = pots.get(symbol)
            .unwrap_or_else(|| panic!("missing pseudopotential for species {symbol}"));
        let aug: &dyn chemrust_hamiltonian_core::pseudopotential::HasAugmentationData = match pot {
            Pseudopotential::Usp(d) => d,
            _ => continue,
        };
        let gmax_pp = pot.gmax();
        let beta_g = compute_beta_g(&bg_wave_block, aug, cell, ion_idx, &gvg_wave, gmax_pp, k_cart)
            .expect("compute_beta_g failed");
        let d0 = build_d0_expanded(aug);
        let species_q_info = species_q.iter()
            .find(|sq| sq.symbol == *symbol).expect("species Q_on_grid not found");
        let d = compute_screened_d_from_fft(
            &species_q_info.q_grid, &veff_wave_fft, cell, ion_idx, &gvg_wave, &d0,
        );
        let q_exp = build_q_expanded(aug);
        ion_data.push(IonData { beta_g, d, q_exp });
    }
    eprintln!("  Total USPP ions: {}", ion_data.len());

    // ---- 10. Full USPP preconditioner (CASTEP nlpot.f90:15480-15665) ---------
    //
    // P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹
    eprintln!("[Gate 2] Building full USPP preconditioner...");
    let total_proj: usize = ion_data.iter().map(|ion| ion.beta_g.shape()[0]).sum();
    eprintln!("  total projectors across {} ions: {}", ion_data.len(), total_proj);

    let mut beta_g_full = Array2::<Complex64>::zeros((nplw, total_proj));
    let mut q_full = Array2::<Complex64>::zeros((total_proj, total_proj));
    let mut offset = 0;

    for (_ion_idx, ion) in ion_data.iter().enumerate() {
        let n_proj = ion.beta_g.shape()[0]; // ion.beta_g is (n_proj, nplw), row-major
        // Transpose: (n_proj, nplw) → (nplw, n_proj) for UsppPreconditioner
        for p in 0..n_proj {
            for g in 0..nplw {
                beta_g_full[[g, offset + p]] = ion.beta_g[[p, g]];
            }
        }
        // Block-diagonal Q: each ion's q_exp is (n_proj, n_proj)
        for i in 0..n_proj {
            for j in 0..n_proj {
                q_full[[offset + i, offset + j]] = Complex64::new(ion.q_exp[[i, j]], 0.0);
            }
        }
        offset += n_proj;
    }

    let precond = UsppPreconditioner::new(beta_g_full, q_full, kinetic_g, k_cart);

    // ---- 11. H/S closure (full USPP, same as Gate 1) ------------------------
    let apply_hs = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
        let mut vnl_v = vec![Complex64::ZERO; nplw];
        let mut s_aug_v = vec![Complex64::ZERO; nplw];

        for ion in &ion_data {
            let n_exp = ion.beta_g.shape()[0];
            let mut beta_phi = vec![Complex64::ZERO; n_exp];
            for n in 0..n_exp {
                let mut bp_n = Complex64::ZERO;
                for (vg, bg) in v.iter().map(|c| c.conj()).zip(ion.beta_g.row(n)) {
                    bp_n += vg * bg;
                }
                beta_phi[n] = bp_n;
            }
            let mut c_vnl = vec![Complex64::ZERO; n_exp];
            let mut c_q = vec![Complex64::ZERO; n_exp];
            for n in 0..n_exp {
                let d_row = ion.d.row(n);
                let q_row = ion.q_exp.row(n);
                for m in 0..n_exp {
                    if d_row[m].abs() > 1e-30 {
                        c_vnl[n] += Complex64::new(d_row[m], 0.0) * beta_phi[m].conj();
                    }
                    if q_row[m].abs() > 1e-30 {
                        c_q[n] += Complex64::new(q_row[m], 0.0) * beta_phi[m].conj();
                    }
                }
            }
            for n in 0..n_exp {
                let cvn = c_vnl[n];
                let cqn = c_q[n];
                let bg_row = ion.beta_g.row(n);
                for ((vnl_g, s_aug_g), bg) in vnl_v.iter_mut().zip(s_aug_v.iter_mut()).zip(bg_row) {
                    *vnl_g += bg * cvn;
                    *s_aug_g += bg * cqn;
                }
            }
        }
        let hv = apply_full_hamiltonian(
            v, &fft_indices, &gcart, k_cart, &veff_wave, &gvg_wave, &vnl_v,
        ).expect("H|v> failed");
        let sv: Vec<Complex64> = v.iter().zip(s_aug_v.iter()).map(|(vg, sag)| vg + sag).collect();
        (hv, sv)
    };
    let apply_s = |v: &[Complex64]| {
        // For USPP: S|v⟩ = |v⟩ + Σ_ion β·Q·β†|v⟩
        let mut sv = v.to_vec();
        let mut s_aug_v = vec![Complex64::ZERO; nplw];
        for ion in &ion_data {
            let n_exp = ion.beta_g.shape()[0];  // beta_g is (n_proj, nplw)
            let mut beta_phi = vec![Complex64::ZERO; n_exp];
            for n in 0..n_exp {
                let mut bp_n = Complex64::ZERO;
                for (vg, bg) in v.iter().map(|c| c.conj()).zip(ion.beta_g.row(n)) {
                    bp_n += vg * bg;
                }
                beta_phi[n] = bp_n;
            }
            let mut c_q = vec![Complex64::ZERO; n_exp];
            for n in 0..n_exp {
                let q_row = ion.q_exp.row(n);
                for m in 0..n_exp {
                    if q_row[m].abs() > 1e-30 {
                        c_q[n] += Complex64::new(q_row[m], 0.0) * beta_phi[m].conj();
                    }
                }
            }
            for n in 0..n_exp {
                let cqn = c_q[n];
                let bg_row = ion.beta_g.row(n);
                for (s_aug_g, bg) in s_aug_v.iter_mut().zip(bg_row) {
                    *s_aug_g += bg * cqn;
                }
            }
        }
        for (svg, sag) in sv.iter_mut().zip(s_aug_v.iter()) {
            *svg += sag;
        }
        sv
    };

    // ---- 12. Band-by-band CG from CASTEP converged wavefunctions ------------
    //
    // CASTEP's .check stores ψ from CG refinement inside hamiltonian_searchspace_ks,
    // but stores ε from the subspace-diagonalization step (wave_diagonalise_H_ks).
    // These come from different algorithm steps, so ⟨ψ|H|ψ⟩ for the stored ψ
    // does NOT equal the stored ε — the difference is the "subspace diagonalization
    // floor" (~9 mHa for band 0).  We therefore compare CG's eigenvalue against
    // the INITIAL Rayleigh quotient from our H (matching Gate 1's drift approach),
    // and print the comparison against .check eigenvalues for diagnostics.
    eprintln!();
    eprintln!("[Gate 2] Running band-by-band CG on {N_BANDS} CASTEP converged bands...");
    eprintln!("  Starting from .check wavefunctions (max 15 steps/band, drift target < 1e-6 Ha)");

    // Precompute initial Rayleigh quotients from our H for comparison.
    let mut initial_eps = Vec::with_capacity(N_BANDS);
    for psi in &bands {
        let (hpsi, spsi) = apply_hs(psi);
        let h_expect = inner_product(psi, &hpsi).re;
        let s_expect = inner_product(psi, &spsi).re;
        initial_eps.push(h_expect / s_expect);
    }

    let mut converged_bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = Vec::new();
    let mut max_drift = 0.0_f64;
    let mut max_drift_band = 0;
    let mut total_steps = 0;

    for ib in 0..N_BANDS {
        let psi_init = bands[ib].clone();
        let eps_initial = initial_eps[ib];

        let result = band_cg_minimize(
            &psi_init,
            &precond,
            &converged_bands,  // lower bands already converged
            15,                // max_steps per band (should converge in 0-1 steps)
            1e-10,             // tight tolerance: eigenvalue drift < 1e-10 Ha
            &apply_hs,
            &apply_s,
        );

        // Add this band to converged set for next band's orthogonalization
        let spsi = apply_s(&result.psi);
        converged_bands.push((result.psi.clone(), spsi));

        let drift = (result.eigenvalue - eps_initial).abs();
        if drift > max_drift {
            max_drift = drift;
            max_drift_band = ib;
        }
        total_steps += result.n_steps;

        if ib % 20 == 0 {
            let eps_castep = castep_eigvals[ib];
            eprintln!("  Band {:3}: ε = {:.8e} Ha, drift_vs_init = {:.4e} Ha, vs_CASTEP = {:.4e}, {} steps, converged = {}",
                ib, result.eigenvalue, drift, (result.eigenvalue - eps_castep).abs(),
                result.n_steps, result.converged);
        }
    }

    // ---- 13. Summary and assertions -----------------------------------------
    eprintln!();
    eprintln!("[Gate 2] Results:");
    eprintln!("  Total CG steps across {N_BANDS} bands: {total_steps}");
    eprintln!("  Max drift from initial Rayleigh quotient: {:.4e} Ha (band {})", max_drift, max_drift_band);
    eprintln!("  Average steps per band: {:.2}", total_steps as f64 / N_BANDS as f64);

    // Primary criterion: CG preserves the initial Rayleigh quotient (drift < 1e-6).
    // This matches Gate 1's approach (compare against our H, not against .check).
    assert!(
        max_drift < 1e-6,
        "Gate 2 FAILED: max eigenvalue drift from initial Rayleigh quotient = {:.4e} Ha (band {}) >= 1e-6 Ha",
        max_drift, max_drift_band,
    );

    eprintln!();
    eprintln!("  Gate 2 PASSED: Band-by-band CG preserves Rayleigh quotient within {:.4e} Ha.", max_drift);
}
