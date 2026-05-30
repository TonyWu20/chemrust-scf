// ---------------------------------------------------------------------------
// Gate 1: Consistency Check — FULL USPP H and S
//
// Verify that one CG iteration on CASTEP's converged state produces negligible
// eigenvalue drift (max|ε_out - ε_in| < 1e-10 Ha across all bands).
//
// This tests the full H = T + V_loc + V_NL Hamiltonian and S = I + β·Q·β†
// overlap operator with complete USPP augmentation via the beta_phi pipeline.
// ---------------------------------------------------------------------------

use std::fs::{self, File};
use std::io::BufReader;

use ndarray::Array2;
use num_complex::Complex64;

use chemrust_hamiltonian_core::{
    CheckFile, GVectorGrid, EffectivePotential, RealGrid,
    PseudopotentialSet, Pseudopotential,
    hamiltonian::{apply_full_hamiltonian, inner_product, local_potential_expectation},
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

/// CASTEP-checkpoint file with converged wavefunctions and eigenvalues.
/// Must match the CASTEP run that produced the H_sub dump.
const CHECK_PATH: &str =
    "/export/public_castep_jobs/tony/Cu111_CO_H_dump/Cu111_CO.check";

/// Formatted local potential V_eff on the fine FFT grid.
/// Must match the CASTEP run that produced the H_sub dump.
const POT_PATH: &str =
    "/export/public_castep_jobs/tony/Cu111_CO_H_dump/Cu111_CO.pot_fmt";

/// Pseudopotential file directory.
const POT_DIR: &str = "/export/Potentials";

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
// Helper: build expanded Q_aug matrix (same selection rule as D0)
// ---------------------------------------------------------------------------

/// Build the expanded Q_aug matrix for the USPP S-overlap correction.
///
/// Q[n_exp, m_exp] = Q_aug[rad_n, rad_m]   if l_n == l_m && m_n == m_m
///                    0                     otherwise
///
/// Q_aug data layout (from USP file, same as `HasAugmentationData::q_aug`):
/// Lower-triangular per-l-channel: Q_aug.0[canon].0[j] gives the j-th
/// element in the same-l channel for the canon-th radial projector.
fn build_q_expanded(aug: &dyn chemrust_hamiltonian_core::pseudopotential::HasAugmentationData) -> Array2<f64> {
    let projectors = aug.projectors();
    let n_expanded = expanded_projector_count(projectors);
    let q_rows = aug.q_aug();

    // Precompute within-l count for each radial projector.
    // within_l[i] = number of projectors up to and including i with same l.
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
// Gate 1 consistency test
// ---------------------------------------------------------------------------

#[test]
fn gate1_consistency_check() {
    // ---- 1. Read .check file ------------------------------------------------
    eprintln!("[Gate 1] Reading .check file...");
    let check_file = File::open(CHECK_PATH).expect("failed to open .check file");
    let castep_bin =
        CheckFile::read(BufReader::new(check_file)).expect("failed to parse .check file");

    // ---- 2. Extract wavefunction data --------------------------------------
    // nspins=1, only one k-point in this system.
    let wave = castep_bin
        .wavefunction
        .as_ref()
        .expect(".check file has no wavefunction data");
    let kpt0 = &wave.kpt_data[0];
    let nplw = kpt0.nplw;
    let nbands = kpt0.bands.len();
    let pw_coords = &kpt0.pw_grid_coord;
    let k_frac = kpt0.coords;

    eprintln!(
        "  {nbands} bands, {nplw} plane waves, k-point = [{:.6}, {:.6}, {:.6}]",
        k_frac[0], k_frac[1], k_frac[2]
    );
    eprintln!("  wavefunction grid: {:?}", wave.grid);

    // ---- 3. Read V_eff from .pot_fmt ---------------------------------------
    eprintln!("[Gate 1] Reading .pot_fmt...");
    let pot_content = fs::read_to_string(POT_PATH).expect("failed to read .pot_fmt");
    let (fine_grid, veff_arr) =
        parse_pot_fmt(&pot_content).expect("failed to parse .pot_fmt");
    let veff_fine = EffectivePotential::from_inner(RealGrid::from_inner(veff_arr));
    eprintln!("  fine grid: {:?}", fine_grid);

    let check_fine = castep_bin
        .fine_grid
        .expect("no fine grid in .check file");
    assert_eq!(
        fine_grid, check_fine,
        "fine grid mismatch: .pot_fmt={fine_grid:?}, .check={check_fine:?}"
    );

    // ---- 4. Build GVectorGrids ---------------------------------------------
    let recip = castep_bin.cell.recip_lattice;
    let wave_grid = wave.grid;
    let gvg_wave = GVectorGrid::new(wave_grid[0], wave_grid[1], wave_grid[2], recip);
    let gvg_fine = GVectorGrid::new(fine_grid[0], fine_grid[1], fine_grid[2], recip);

    // ---- 5. Downsample V_eff to wavefunction grid (no-op if grids equal) ----
    eprintln!("[Gate 1] Downsampling V_eff to wavefunction grid...");
    let veff_wave_arr = downsample_density_to_wave_grid(
        veff_fine.as_real_grid(),
        &gvg_wave,
        &gvg_fine,
    )
    .expect("V_eff downsampling failed");
    let veff_wave = EffectivePotential::from_inner(veff_wave_arr);

    // ---- 6. Compute FFT indices and Cartesian G-vectors ---------------------
    let fft_indices = gvg_wave.pw_to_fft_indices(pw_coords);
    let gcart: Vec<[f64; 3]> = fft_indices
        .iter()
        .map(|&[iz, iy, ix]| gvg_wave.gcart()[[iz, iy, ix]])
        .collect();

    // ---- 7. Convert k-point to Cartesian ------------------------------------
    let k_cart = kpoint_to_cartesian(k_frac, &recip);

    // ---- 8. Kinetic energies for TPA preconditioner -------------------------
    let kinetic_g: Vec<f64> = gcart
        .iter()
        .map(|&gc| {
            let kg = [
                k_cart[0] + gc[0],
                k_cart[1] + gc[1],
                k_cart[2] + gc[2],
            ];
            0.5 * (kg[0] * kg[0] + kg[1] * kg[1] + kg[2] * kg[2])
        })
        .collect();

    // ---- 9. Load pseudopotential files --------------------------------------
    eprintln!("[Gate 1] Loading pseudopotential files...");
    let cell = &castep_bin.cell;
    let mut pots = PseudopotentialSet::new();
    for (sp_idx, symbol) in cell.species_symbols.iter().enumerate() {
        let pot_file = &cell.species_pot_files[sp_idx];
        // If it's a full path, use it; otherwise assume in POT_DIR.
        let pot_path = if pot_file.starts_with('/') {
            pot_file.clone()
        } else {
            format!("{POT_DIR}/{pot_file}")
        };
        eprintln!("  loading {symbol}: {pot_path}");
        let src = fs::read_to_string(&pot_path)
            .unwrap_or_else(|e| panic!("failed to read {pot_path}: {e}"));
        let usp = parse_usp(&src)
            .unwrap_or_else(|e| panic!("failed to parse USP for {symbol}: {e}"));
        pots.insert(symbol.clone(), Pseudopotential::Usp(usp));
    }

    // ---- 10. Precompute Q_on_grid per species, then screened D per ion ------
    eprintln!("[Gate 1] Precomputing Q_on_grid per species...");

    /// Precomputed Q_nm(G) on the wavefunction grid, per species.
    struct SpeciesQ {
        symbol: String,
        q_grid: QOnGrid,
    }
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
        eprintln!("  {symbol}: {} Q pairs on wave grid {:?}",
            q_grid.pairs.len(), q_grid.grid);
        species_q.push(SpeciesQ { symbol: symbol.clone(), q_grid });
    }

    // FFT V_eff on the wavefunction grid to G-space for D screening.
    eprintln!("[Gate 1] FFT V_eff to G-space for D screening...");
    let veff_wave_fft: RecipGrid<Complex64> = fft_forward_3d(veff_wave.as_real_grid())
        .expect("V_eff forward FFT failed");

    eprintln!("[Gate 1] Precomputing per-ion beta_g and screened D...");

    // Minimal KptWaveBlock for beta_g (only uses pw_grid_coord, not bands).
    let bg_wave_block = chemrust_hamiltonian_core::types::KptWaveBlock {
        coords: k_frac,
        nplw,
        pw_grid_coord: pw_coords.clone(),
        bands: Vec::new(),
    };

    /// Per-ion precomputed data for the V_NL+S pipeline.
    struct IonData {
        beta_g: Array2<Complex64>,  // (n_expanded, nplw)
        d: Array2<f64>,            // (n_expanded, n_expanded) — D0 + V_eff screening
        q_exp: Array2<f64>,        // (n_expanded, n_expanded)
    }

    let mut ion_data: Vec<IonData> = Vec::new();
    for ion_idx in 0..cell.num_ions {
        let species_idx = cell.ion_species[ion_idx];
        let symbol = &cell.species_symbols[species_idx];
        let pot = pots.get(symbol).unwrap_or_else(|| {
            panic!("missing pseudopotential for species {symbol}")
        });
        let aug: &dyn chemrust_hamiltonian_core::pseudopotential::HasAugmentationData = match pot {
            Pseudopotential::Usp(d) => d,
            _ => continue, // skip non-USP (e.g. recpot)
        };
        let gmax_pp = pot.gmax();
        let beta_g = compute_beta_g(&bg_wave_block, aug, cell, ion_idx, &gvg_wave, gmax_pp, k_cart)
            .expect("compute_beta_g failed");
        let d0 = build_d0_expanded(aug);

        // Find Q_on_grid for this species and compute screened D.
        let species_q_info = species_q.iter()
            .find(|sq| sq.symbol == *symbol)
            .expect("species Q_on_grid not found");
        let d = compute_screened_d_from_fft(
            &species_q_info.q_grid,
            &veff_wave_fft,
            cell,
            ion_idx,
            &gvg_wave,
            &d0,
        );

        let q_exp = build_q_expanded(aug);
        let n_exp = beta_g.shape()[0];

        // Estimate memory for this ion
        let ion_bytes = (n_exp * nplw * 16 + n_exp * n_exp * 8 * 2) as f64 / 1_048_576.0;
        eprintln!(
            "  ion {ion_idx:3} ({symbol}): n_exp={n_exp:3}, {:>6.1} MiB",
            ion_bytes,
        );
        ion_data.push(IonData { beta_g, d, q_exp });
    }
    eprintln!("  Total USPP ions: {}", ion_data.len());

    // ---- 11. Build TPA-only preconditioner -----------------------------------
    let beta_g_dummy = Array2::<Complex64>::zeros((nplw, 1));
    let q_matrix_dummy = Array2::<Complex64>::eye(1);
    let precond = UsppPreconditioner::new(beta_g_dummy, q_matrix_dummy, kinetic_g, k_cart);

    // ---- 12. Create the (H|v>, S|v>) closure with full USPP ------------------
    //
    //   H|v> = T|v> + V_loc|v> + V_NL|v>
    //   V_NL|v>(G) = Σ_{ion} Σ_{n,m} β^I_n(G) · D^I_nm · <β^I_m|v>
    //
    //   S|v> = |v> + S_aug|v>
    //   S_aug|v>(G) = Σ_{ion} Σ_{n,m} β^I_n(G) · Q^I_nm · <β^I_m|v>

    let apply_hs = |v: &[Complex64]| -> (Vec<Complex64>, Vec<Complex64>) {
        let mut vnl_v = vec![Complex64::ZERO; nplw];
        let mut s_aug_v = vec![Complex64::ZERO; nplw];

        for ion in &ion_data {
            let n_exp = ion.beta_g.shape()[0];

            // ---- beta_phi[n] = Σ_g conj(v[g]) · beta_g[n, g] ----------------
            let mut beta_phi = vec![Complex64::ZERO; n_exp];
            for n in 0..n_exp {
                let mut bp_n = Complex64::ZERO;
                for (vg, bg) in v.iter().map(|c| c.conj()).zip(ion.beta_g.row(n)) {
                    bp_n += vg * bg;
                }
                beta_phi[n] = bp_n;
            }

            // ---- c_vnl[n] = Σ_m D0[n,m] · <β_m|v> ----------------------
            // ---- c_q[n]   = Σ_m Q[n,m] · <β_m|v> -----------------------
            // NOTE: beta_phi[m] = <v|β_m> = conj(<β_m|v>), so we conjugate.
            let mut c_vnl = vec![Complex64::ZERO; n_exp];
            let mut c_q = vec![Complex64::ZERO; n_exp];
            for n in 0..n_exp {
                let d_row = ion.d.row(n);
                let q_row = ion.q_exp.row(n);
                for m in 0..n_exp {
                    let dm = d_row[m];
                    if dm.abs() > 1e-30 {
                        c_vnl[n] += Complex64::new(dm, 0.0) * beta_phi[m].conj();
                    }
                    let qm = q_row[m];
                    if qm.abs() > 1e-30 {
                        c_q[n] += Complex64::new(qm, 0.0) * beta_phi[m].conj();
                    }
                }
            }

            // ---- Accumulate into vnl_v and s_aug_v --------------------------
            // V_NL|v>(G) += Σ_n β_n(G) · c_vnl[n]
            // S_aug|v>(G) += Σ_n β_n(G) · c_q[n]
            for n in 0..n_exp {
                let cvn = c_vnl[n];
                let cqn = c_q[n];
                let bg_row = ion.beta_g.row(n);
                for ((vnl_g, s_aug_g), bg) in vnl_v
                    .iter_mut()
                    .zip(s_aug_v.iter_mut())
                    .zip(bg_row)
                {
                    *vnl_g += bg * cvn;
                    *s_aug_g += bg * cqn;
                }
            }
        }

        // H|v> = T|v> + V_loc|v> + V_NL|v>
        let hv = apply_full_hamiltonian(
            v, &fft_indices, &gcart, k_cart, &veff_wave, &gvg_wave, &vnl_v,
        )
        .expect("H|v> in closure failed");

        // S|v> = |v> + S_aug|v>
        let sv: Vec<Complex64> = v.iter().zip(s_aug_v.iter()).map(|(vg, sag)| vg + sag).collect();

        (hv, sv)
    };

    // S = I for Phase-0 (norm-conserving approximation; full USPP S is applied
    // inside apply_hs but the band_cg S-orthogonalization uses S = I).
    let apply_s = |v: &[Complex64]| v.to_vec();

    // ---- 12a. Compute H_sub = <ψ_i|H|ψ_j> for all bands (reference CPU path) ---
    // This is the reference H_sub using the same CPU Hamiltonian as the CG loop.
    // We dump it to a file and optionally compare against CASTEP's H_sub dump.
    const CHEMRUST_H_SUB_PATH: &str = "h_sub_chemrust_debug.dat";
    const CASTEP_H_SUB_PATH: &str =
        "/export/public_castep_jobs/tony/Cu111_CO_H_dump/Cu111_CO.H_sub_debug.dat";

    eprintln!("[Gate 1] Computing full H_sub matrix for {nbands} bands...");
    let mut h_psi_all: Vec<Vec<Complex64>> = Vec::with_capacity(nbands);
    for b in 0..nbands {
        let psi_b = &kpt0.bands[b];
        let (hv, _sv) = apply_hs(psi_b);
        h_psi_all.push(hv);
    }
    let mut h_sub = vec![vec![Complex64::ZERO; nbands]; nbands];
    for j in 0..nbands {
        for i in 0..nbands {
            h_sub[i][j] = inner_product(&kpt0.bands[i], &h_psi_all[j]);
        }
    }

    // Dump H_sub to file (same format as CASTEP: "nbands\n i j re im")
    if let Ok(mut file) = std::fs::File::create(CHEMRUST_H_SUB_PATH) {
        use std::io::Write;
        let _ = writeln!(file, "{nbands}");
        for j in 0..nbands {
            for i in 0..nbands {
                let val = h_sub[i][j];
                let _ = writeln!(
                    file,
                    "{:6} {:6} {:26.16e} {:26.16e}",
                    i + 1,
                    j + 1,
                    val.re,
                    val.im
                );
            }
        }
        eprintln!("[Gate 1] H_sub dumped to {CHEMRUST_H_SUB_PATH}");
    }

    // Compare against CASTEP H_sub dump (if accessible)
    if let Ok(data) = std::fs::read_to_string(CASTEP_H_SUB_PATH) {
        let castep_lines: Vec<&str> = data.lines().collect();
        if let Ok(castep_nbands) = castep_lines[0].trim().parse::<usize>() {
            if castep_nbands == nbands {
                let mut castep_h_sub = vec![vec![Complex64::ZERO; nbands]; nbands];
                for line in &castep_lines[1..] {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 4 {
                        let i: usize = parts[0].parse().unwrap_or(1) - 1;
                        let j: usize = parts[1].parse().unwrap_or(1) - 1;
                        if i < nbands && j < nbands {
                            let re: f64 = parts[2].parse().unwrap_or(0.0);
                            let im: f64 = parts[3].parse().unwrap_or(0.0);
                            castep_h_sub[i][j] = Complex64::new(re, im);
                        }
                    }
                }

                // Element-wise comparison
                let mut max_diff = 0.0_f64;
                let mut max_i = 0;
                let mut max_j = 0;
                for j in 0..nbands {
                    for i in 0..nbands {
                        let diff = (h_sub[i][j] - castep_h_sub[i][j]).norm();
                        if diff > max_diff {
                            max_diff = diff;
                            max_i = i;
                            max_j = j;
                        }
                    }
                }
                eprintln!("[Gate 1] H_sub vs CASTEP: max|Δ| = {max_diff:.6e} Ha at ({max_i},{max_j})");

                // Diagonal element difference
                let max_diag_diff = (0..nbands)
                    .map(|i| (h_sub[i][i] - castep_h_sub[i][i]).norm())
                    .fold(0.0_f64, f64::max);
                eprintln!("[Gate 1] H_sub vs CASTEP: max|Δ_diag| = {max_diag_diff:.6e} Ha");

                // ---- Component decomposition: T, V_NL, V_loc ----
                eprintln!();
                eprintln!("[Gate 1] Decomposing H_sub into T, V_NL, V_loc components...");

                // T_sub[i][j] from kinetic formula: <ψ_i|0.5|k+G|²|ψ_j>
                // Compute kinetic energy per PW: 0.5|k+G|²
                let t_per_pw: Vec<f64> = gcart.iter().map(|&gc| {
                    let kg = [k_cart[0] + gc[0], k_cart[1] + gc[1], k_cart[2] + gc[2]];
                    0.5 * (kg[0] * kg[0] + kg[1] * kg[1] + kg[2] * kg[2])
                }).collect();
                let mut t_sub = vec![vec![Complex64::ZERO; nbands]; nbands];
                let mut t_psi_all: Vec<Vec<Complex64>> = Vec::with_capacity(nbands);
                for b in 0..nbands {
                    let psi_b = &kpt0.bands[b];
                    let t_psi: Vec<Complex64> = psi_b.iter().zip(t_per_pw.iter())
                        .map(|(&c, &ek)| c * ek).collect();
                    t_psi_all.push(t_psi);
                }
                for j in 0..nbands {
                    for i in 0..nbands {
                        t_sub[i][j] = inner_product(&kpt0.bands[i], &t_psi_all[j]);
                    }
                }

                // V_NL_sub[i][j] = Σ_ion Σ_nm D_nm · conj(<ψ_i|β_n>) · <β_m|ψ_j>
                let mut vnl_sub = vec![vec![Complex64::ZERO; nbands]; nbands];
                for ion in &ion_data {
                    let n_exp = ion.d.shape()[0];
                    // Pre-compute beta_phi for this ion: beta_phi[n][b] = <ψ_b|β_n>
                    let mut bp = vec![vec![Complex64::ZERO; nbands]; n_exp];
                    for n in 0..n_exp {
                        let bg_row = ion.beta_g.row(n);
                        for b in 0..nbands {
                            let psi_b = &kpt0.bands[b];
                            bp[n][b] = psi_b.iter().map(|c| c.conj())
                                .zip(bg_row.iter())
                                .map(|(cg, &bg)| cg * bg)
                                .sum();
                        }
                    }
                    // V_NL contribution from this ion
                    for n in 0..n_exp {
                        for m in 0..n_exp {
                            let d_nm = ion.d[[n, m]];
                            if d_nm.abs() > 1e-30 {
                                let cd = Complex64::new(d_nm, 0.0);
                                for i in 0..nbands {
                                    let bphi_ni_conj = bp[n][i].conj();
                                    for j in 0..nbands {
                                        vnl_sub[i][j] += cd * bphi_ni_conj * bp[m][j];
                                    }
                                }
                            }
                        }
                    }
                }

                // V_loc_sub = H_sub - T_sub - V_NL_sub
                let mut vloc_sub = vec![vec![Complex64::ZERO; nbands]; nbands];
                for i in 0..nbands {
                    for j in 0..nbands {
                        vloc_sub[i][j] = h_sub[i][j] - t_sub[i][j] - vnl_sub[i][j];
                    }
                }

                // Print decomposition for the worst bands
                fn fmt_sign(v: f64) -> String {
                    format!("{:+10.6e}", v)
                }

                // Find top 5 bands by diagonal |Δ|
                let mut diag_diffs: Vec<(usize, f64)> = (0..nbands)
                    .map(|i| (i, (h_sub[i][i] - castep_h_sub[i][i]).norm()))
                    .collect();
                diag_diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                eprintln!("  Band    H_sub(chemrust)    H_sub(CASTEP)        Δ       Δ(T)      Δ(V_loc)    Δ(V_NL)");
                for &(b, _) in diag_diffs.iter().take(5) {
                    let delta = h_sub[b][b] - castep_h_sub[b][b];
                    let delta_t = (t_sub[b][b] - castep_h_sub[b][b]).re;
                    let delta_vloc = vloc_sub[b][b].re;
                    let delta_vnl = vnl_sub[b][b].re;
                    eprintln!(
                        "  {b:4}  {}  {}  {}  {}  {}  {}",
                        fmt_sign(h_sub[b][b].re),
                        fmt_sign(castep_h_sub[b][b].re),
                        fmt_sign(delta.re),
                        fmt_sign(delta_t),
                        fmt_sign(delta_vloc),
                        fmt_sign(delta_vnl),
                    );
                }

                // ---- Sanity: verify V_NL_sub against nlpot_expectation ----------
                let mut vnl_check_ok = true;
                for b in 0..nbands.min(3) {
                    let mut vnl_expect = 0.0_f64;
                    for ion in &ion_data {
                        let n_exp = ion.d.shape()[0];
                        let mut bp = vec![Complex64::ZERO; n_exp];
                        for n in 0..n_exp {
                            let bg_row = ion.beta_g.row(n);
                            bp[n] = kpt0.bands[b].iter().map(|c| c.conj())
                                .zip(bg_row.iter())
                                .map(|(cg, &bg)| cg * bg)
                                .sum();
                        }
                        for n in 0..n_exp {
                            let bphi_n_conj = bp[n].conj();
                            for m in 0..n_exp {
                                let d_nm = ion.d[[n, m]];
                                if d_nm.abs() > 1e-30 {
                                    vnl_expect += d_nm * (bphi_n_conj * bp[m]).re;
                                }
                            }
                        }
                    }
                    let vnl_decomp = vnl_sub[b][b].re;
                    let vnl_diff = (vnl_expect - vnl_decomp).abs();
                    if vnl_diff > 1e-10 {
                        vnl_check_ok = false;
                        eprintln!("[Gate 1] V_NL MISMATCH band {b}: nlpot_expect={vnl_expect:.10e} vs decomp={vnl_decomp:.10e} diff={vnl_diff:.10e}");
                    }
                }
                if vnl_check_ok {
                    eprintln!("[Gate 1] V_NL sanity check: OK (nlpot_expectation matches decomposition)");
                }

                // ---- Cross-check: V_loc from local_potential_expectation ----------
                for &b in &[0usize, 1] {
                    let psi_b = &kpt0.bands[b];
                    let vloc_expect = local_potential_expectation(
                        psi_b, &fft_indices, &veff_wave, &gvg_wave,
                    ).unwrap_or(0.0);
                    let vloc_decomp = vloc_sub[b][b].re;
                    let vloc_diff = (vloc_expect - vloc_decomp).abs();
                    if vloc_diff > 1e-10 {
                        eprintln!("[Gate 1] V_loc MISMATCH band {b}: expect={vloc_expect:.10e} vs decomp={vloc_decomp:.10e} diff={vloc_diff:.10e}");
                    } else {
                        eprintln!("[Gate 1] V_loc OK band {b}: expect={vloc_expect:.10e} decomp={vloc_decomp:.10e}");
                    }
                }
            }
        }
    }

    // ---- 13. Main loop: 1 CG step per band, track max drift -----------------
    let castep_eigs = &castep_bin.eigenvalues.kpoints[0].spins[0].eigenvalues;
    let mut max_drift = 0.0_f64;
    let mut max_drift_band = 0;
    let mut sum_drift = 0.0_f64;

    eprintln!();
    eprintln!("[Gate 1] Running 1 CG step per band (full USPP H/S)...");

    for b in 0..nbands.min(3) {
        let psi_b = &kpt0.bands[b];

        // ---- Debug: compute per-ion S_aug contribution ----------------------
        let mut per_ion_saug: Vec<(String, f64)> = Vec::new();
        for ion_idx in 0..ion_data.len() {
            let species_idx = cell.ion_species[ion_idx];
            let symbol = &cell.species_symbols[species_idx];
            let ion = &ion_data[ion_idx];
            let n_exp = ion.beta_g.shape()[0];
            // beta_phi
            let mut beta_phi = vec![Complex64::ZERO; n_exp];
            for n in 0..n_exp {
                for (vg, bg) in psi_b.iter().map(|c| c.conj()).zip(ion.beta_g.row(n)) {
                    beta_phi[n] += vg * bg;
                }
            }
            // <psi|S_aug|psi> = Σ_{n,m} Q_nm · <ψ|β_n> · <β_m|ψ>
            //                  = Σ_{n,m} Q_nm · beta_phi[n] · conj(beta_phi[m])
            let mut saug_expect = 0.0_f64;
            for n in 0..n_exp {
                for m in 0..n_exp {
                    let qv = ion.q_exp[[n, m]];
                    if qv.abs() > 1e-30 {
                        saug_expect += qv * (beta_phi[n] * beta_phi[m].conj()).re;
                    }
                }
            }
            per_ion_saug.push((symbol.clone(), saug_expect));
        }

        let l2_norm_sq: f64 = psi_b.iter().map(|c| c.norm_sqr()).sum();
        let total_saug: f64 = per_ion_saug.iter().map(|(_, v)| v).sum();
        eprintln!(
            "  DEBUG band {b}: ||psi||^2 = {:.6e}, total S_aug = {:.4e}, <S> = {:.6e}",
            l2_norm_sq, total_saug, l2_norm_sq + total_saug,
        );
        for (symbol, sv) in &per_ion_saug {
            eprintln!("    {symbol:4}: S_aug = {:.4e}", sv);
        }

        // ---- 13a. Compute eps_in = <ψ|H|ψ> / <ψ|S|ψ> -----------------------
        let (hpsi_init, spsi_init) = apply_hs(psi_b);
        let h_expect = inner_product(psi_b, &hpsi_init).re;
        let s_overlap = inner_product(psi_b, &spsi_init).re;
        let eps_in = h_expect / s_overlap;

        // ---- 13b. Run 1 CG step ---------------------------------------------
        let result = band_cg_minimize(
            psi_b, &precond, &[], 1, 1e-20, &apply_hs, &apply_s,
        );

        let drift = (result.eigenvalue - eps_in).abs();
        sum_drift += drift;
        if drift > max_drift {
            max_drift = drift;
            max_drift_band = b;
        }

        eprintln!(
            "  band {b:4}: eps_in = {:.6e}, eps_CASTEP = {:.6e}, \
             <S> = {:.6e} (ΔS={:.4e}), drift = {:.4e}, n_steps={}",
            eps_in,
            castep_eigs[b],
            s_overlap,
            s_overlap - 1.0,
            drift,
            result.n_steps,
        );
    }

    // ---- 13c. Remaining bands (minimal diagnostics) -------------------------
    for b in 3..nbands {
        let psi_b = &kpt0.bands[b];

        // ---- 13a. Compute eps_in = <ψ|H|ψ> / <ψ|S|ψ> -----------------------
        let (hpsi_init, spsi_init) = apply_hs(psi_b);
        let h_expect = inner_product(psi_b, &hpsi_init).re;
        let s_overlap = inner_product(psi_b, &spsi_init).re;
        let eps_in = h_expect / s_overlap;

        // ---- 13b. Run 1 CG step ---------------------------------------------
        let result = band_cg_minimize(
            psi_b, &precond, &[], 1, 1e-20, &apply_hs, &apply_s,
        );

        let drift = (result.eigenvalue - eps_in).abs();
        sum_drift += drift;
        if drift > max_drift {
            max_drift = drift;
            max_drift_band = b;
        }

        // Print progress for a representative subset of bands
        if b < 5
            || b == 10
            || b == 20
            || b == 30
            || b % 32 == 0
            || b == nbands - 1
        {
            eprintln!(
                "  band {b:4}: eps_in = {:.6e}, eps_CASTEP = {:.6e}, \
                 <S> = {:.6e} (ΔS={:.4e}), drift = {:.4e}, n_steps={}",
                eps_in,
                castep_eigs[b],
                s_overlap,
                s_overlap - 1.0,
                drift,
                result.n_steps,
            );
        }
    }

    let avg_drift = sum_drift / nbands as f64;

    eprintln!();
    eprintln!("[Gate 1] Results (full USPP H = T+V_loc+V_NL, S = I+β·Q·β†):");
    eprintln!("  Max drift  = {:.4e} Ha (band {})", max_drift, max_drift_band);
    eprintln!("  Avg drift  = {:.4e} Ha", avg_drift);
    eprintln!("  Target     = 1.0e-10 Ha");

    assert!(
        max_drift < 1e-10,
        "Gate 1 FAILED: max drift = {:.4e} Ha >= 1e-10 Ha (band {})",
        max_drift,
        max_drift_band,
    );

    eprintln!("  PASSED");
}
