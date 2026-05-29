// ---------------------------------------------------------------------------
// Gate 2: Convergence from Random Initialization (CASTEP method='R')
//
// Verify that CG converges band-0 from a CASTEP method='R' random initial
// guess to within 1e-6 Ha of CASTEP reference. Primary: residual norm < 1e-6 Ha.
// Secondary: |ε - (-1.05502287)| < 1e-6 Ha.
//
// CASTEP method='R' (wave.f90:1900-1949):
//   1. For each PW with E_k < 3.307 Ha (90 eV hardcoded cutoff, line 1737):
//      - ψ[g] = (rn1 - 0.5) + i·(rn2 - 0.5), rn1, rn2 ~ Uniform(0,1)
//   2. For PWs with E_k >= 3.307 Ha: ψ[g] = 0
//   3. S-orthonormalize all bands (wave_Sorthonormalise, line 2036)
//
// This test initializes N_INIT_BANDS random bands, S-orthonormalizes them,
// then converges band-0 while treating bands 1..N-1 as "converged" for
// S-orthogonalization during CG.
// ---------------------------------------------------------------------------

use std::fs::{self, File};
use std::io::BufReader;

use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;

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

/// CASTEP hardcoded initialisation cutoff: 3.307 Ha = 90 eV (wave.f90:1737).
const INIT_CUTOFF_HA: f64 = 3.307;

/// Number of bands to converge (all bands in the system).
/// Cu111_CO has 160 bands; we converge all of them sequentially (band-by-band CG)
/// and check band-0 as the probe against CASTEP reference.
const N_BANDS: usize = 160;

// ---------------------------------------------------------------------------
// Minimal LCG random number generator
//
// Used to match CASTEP's uniform random intialisation (method='R') without
// pulling in the `rand` crate. Deterministic across runs (fixed seed).
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return a uniform random f64 in (0, 1) using POSIX rand48 constants.
    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(25214903917).wrapping_add(11) & 0xFFFFFFFFFFFF;
        (self.state as f64) / 281474976710656.0 // divide by 2^48
    }
}

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
// Generate CASTEP method='R' random wavefunction
//
// CASTEP wave.f90:1900-1949 (non-alt_random_init path):
//   For each PW with E_k < 3.307 Ha:
//     ψ[g] = (rn1 - 0.5) + i·(rn2 - 0.5),  rn1, rn2 ~ Uniform(0,1)
//   For PWs with E_k >= 3.307 Ha: ψ[g] = 0
//   Then S-orthonormalise all bands (wave.f90:2036)
// ---------------------------------------------------------------------------

fn generate_random_wavefunction(kinetic_g: &[f64], seed: u64) -> Vec<Complex64> {
    let nplw = kinetic_g.len();
    let mut psi = vec![Complex64::ZERO; nplw];
    let mut rng = Lcg::new(seed);

    for (g, &ek) in kinetic_g.iter().enumerate() {
        if ek < INIT_CUTOFF_HA {
            let rn1 = rng.next_f64() - 0.5;
            let rn2 = rng.next_f64() - 0.5;
            psi[g] = Complex64::new(rn1, rn2);
        }
        // else: ψ[g] = 0 (already zero from Vec initialization)
    }

    psi
}

// ---------------------------------------------------------------------------
// S-orthonormalize a set of bands using Modified Gram-Schmidt
//
// CASTEP wave_Sorthonormalise (wave.f90:2036) applies Modified Gram-Schmidt
// in the S-metric to ensure ⟨ψᵢ|S|ψⱼ⟩ = δᵢⱼ.
//
// Algorithm:
//   For each band i = 0..N-1:
//     1. S-orthogonalize ψᵢ against all previous bands j < i:
//        ψᵢ ← ψᵢ - Σⱼ ⟨ψⱼ|S|ψᵢ⟩·ψⱼ
//     2. S-normalize: ψᵢ ← ψᵢ / sqrt(⟨ψᵢ|S|ψᵢ⟩)
// ---------------------------------------------------------------------------

fn s_orthonormalize_bands(
    bands: &mut [Vec<Complex64>],
    apply_s: &(impl Fn(&[Complex64]) -> Vec<Complex64> + Sync),
) {
    let n_bands = bands.len();
    if n_bands == 0 {
        return;
    }
    let _n_pw = bands[0].len();

    // Precompute S|bands[j]> for all j (parallel, uses O(n²) S-applies → O(n)).
    let mut s_bands: Vec<Vec<Complex64>> = (0..n_bands)
        .into_par_iter()
        .map(|j| apply_s(&bands[j]))
        .collect();

    for i in 0..n_bands {
        // Split psi and s arrays at i for split-borrow access
        let (psi_lower, psi_upper) = bands.split_at_mut(i);
        let (s_lower, s_upper) = s_bands.split_at_mut(i);

        let psi_i = &mut psi_upper[0];
        let s_psi_i = &mut s_upper[0];

        // Step 1: S-orthogonalize against all previous bands
        // Uses linearity of S: S|ψ_i − overlap·ψ_j⟩ = S|ψ_i⟩ − overlap·S|ψ_j⟩
        // This avoids recomputing apply_s in the inner loop (the O(n³) killer).
        for j in 0..i {
            let psi_j = &psi_lower[j];
            let s_psi_j = &s_lower[j];

            let overlap = inner_product(psi_j, s_psi_i);

            // ψ_i ← ψ_i − overlap · ψ_j
            psi_i
                .par_iter_mut()
                .zip(psi_j.par_iter())
                .for_each(|(p_i, p_j)| *p_i -= overlap * p_j);

            // S|ψ_i⟩ ← S|ψ_i⟩ − overlap · S|ψ_j⟩ (linear update)
            s_psi_i
                .par_iter_mut()
                .zip(s_psi_j.par_iter())
                .for_each(|(s_i, s_j)| *s_i -= overlap * s_j);
        }

        // Step 2: S-normalize
        let s_norm_sq = inner_product(psi_i, s_psi_i).re;
        let inv_norm = 1.0 / s_norm_sq.sqrt();
        psi_i.par_iter_mut().for_each(|p| *p *= inv_norm);
        s_psi_i.par_iter_mut().for_each(|s| *s *= inv_norm);
    }
}

// ---------------------------------------------------------------------------
// Gate 2 convergence test
// ---------------------------------------------------------------------------

#[test]
fn gate2_convergence_from_random_init() {
    // ---- 1. Read .check file (for geometry, G-vectors) ----------------------
    eprintln!("[Gate 2] Reading .check file...");
    let check_file = File::open(CHECK_PATH).expect("failed to open .check file");
    let castep_bin =
        CheckFile::read(BufReader::new(check_file)).expect("failed to parse .check file");

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

    // ---- 8. Generate N_BANDS random wavefunctions (CASTEP method='R') -------
    eprintln!("[Gate 2] Generating {N_BANDS} random wavefunctions (method='R', cutoff={INIT_CUTOFF_HA} Ha)...");
    let mut bands: Vec<Vec<Complex64>> = (0..N_BANDS)
        .map(|i| generate_random_wavefunction(&kinetic_g, 42 + i as u64))
        .collect();
    let n_init: usize = kinetic_g.iter().filter(|&&ek| ek < INIT_CUTOFF_HA).count();
    eprintln!("  {n_init}/{nplw} plane waves below cutoff per band");

    // ---- 9. Load pseudopotentials -------------------------------------------
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

    // ---- 10. Precompute Q_on_grid, beta_g, screened D (same as Gate 1) ------
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

    // ---- 11. Full USPP preconditioner (CASTEP nlpot.f90:15480-15665) ---------
    //
    // P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹
    //
    // Concatenate all per-ion β projectors (each row-based) into a single
    // column-major matrix and build a block-diagonal Q.  This is equivalent to
    // CASTEP's per-ion R matrix summation.
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

    // ---- 12. H/S closure (full USPP, same as Gate 1) ------------------------
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

    // ---- 13. S-orthonormalize all bands (CASTEP wave_Sorthonormalise) -------
    eprintln!("[Gate 2] S-orthonormalizing {N_BANDS} bands...");
    s_orthonormalize_bands(&mut bands, &apply_s);
    eprintln!("  S-orthonormalization complete");

    // ---- 14a. Compute H|ψ⟩ and S|ψ⟩ for ALL bands (subspace diag) -----------
    eprintln!("[Gate 2] Computing H|ψ⟩, S|ψ⟩ for all {N_BANDS} bands...");
    let h_bands: Vec<Vec<Complex64>> = bands
        .par_iter()
        .map(|psi| apply_hs(psi).0)
        .collect();
    let s_bands: Vec<Vec<Complex64>> = bands
        .par_iter()
        .map(|psi| apply_hs(psi).1)
        .collect();

    // ---- 14b. Build H_sub and S_sub -----------------------------------------
    eprintln!("[Gate 2] Building H_sub and S_sub...");
    use chemrust_scf::eigensolver::subspace_diag::{
        build_h_sub, build_s_sub, diagonalize_and_rotate,
    };
    let h_sub = build_h_sub(&bands, &h_bands);
    let s_sub = build_s_sub(&bands, &s_bands);

    // ---- 14c. Diagonalize (Cholesky reduction → standard EV) ----------------
    eprintln!("[Gate 2] Diagonalizing subspace ({N_BANDS}×{N_BANDS}) and rotating...");
    let diag_eigs = diagonalize_and_rotate(&h_sub, &s_sub, &mut bands);

    // ---- 14d. Diagnostic: compare subspace diag eigenvalues vs CASTEP .check ---
    // NOTE: With random starting wavefunctions, the subspace spans a random
    // 160-dimensional subspace of a 60067-dimensional PW space.  The Ritz
    // values (eigenvalues of H_sub) are NOT the true eigenvalues — they only
    // converge to the true eigenvalues as the subspace approaches the invariant
    // subspace over SCF iterations.  CASTEP's wave_diagonalise_H_ks is called
    // with wavefunctions from the PREVIOUS SCF iteration (already near-converged).
    // We print the comparison for diagnostics but do NOT assert against .check.
    let castep_eigvals: Vec<f64> = castep_bin.eigenvalues.kpoints[0].spins[0].eigenvalues.clone();
    let mut max_delta = 0.0_f64;
    let mut min_delta = f64::MAX;
    let mut max_band = 0;
    for (i, (diag_e, castep_e)) in diag_eigs.iter().zip(castep_eigvals.iter()).enumerate() {
        let delta = (diag_e - castep_e).abs();
        if delta > max_delta { max_delta = delta; max_band = i; }
        if delta < min_delta { min_delta = delta; }
    }
    eprintln!("  Subspace diag eigenvalues (random ψ subspace):");
    eprintln!("    max|Δε| = {max_delta:.4e} (band {max_band}), min|Δε| = {min_delta:.4e}");

    // ---- 15. Band-by-band CG refinement (from subspace-diag basis) ----------
    eprintln!();
    eprintln!("[Gate 2] Running band-by-band CG refinement on all {N_BANDS} bands...");
    eprintln!("  Starting from subspace-diagonalized basis (max 50 steps/band)");

    let mut converged_bands: Vec<(Vec<Complex64>, Vec<Complex64>)> = Vec::new();
    let mut band_results = Vec::new();

    for ib in 0..N_BANDS {
        let psi_init = bands[ib].clone();

        let result = band_cg_minimize(
            &psi_init,
            &precond,
            &converged_bands,  // lower bands already converged
            50,                // max_steps per band
            1e-6,              // tol (eigenvalue change)
            &apply_hs,
            &apply_s,
        );

        // Add this band to converged set for next band's orthogonalization
        let spsi = apply_s(&result.psi);
        converged_bands.push((result.psi.clone(), spsi));
        band_results.push(result);

        if ib % 20 == 0 || ib == 0 {
            eprintln!("  Band {:3}: ε = {:.8e} Ha, {} steps, converged = {}",
                ib, band_results[ib].eigenvalue, band_results[ib].n_steps, band_results[ib].converged);
        }
    }

    // ---- 16. Check band-0 as probe against CASTEP .check reference ------------
    eprintln!();
    eprintln!("[Gate 2] Band-0 (probe) results:");
    let result = &band_results[0];
    let castep_eps_0 = castep_eigvals[0];
    eprintln!("  converged              = {}", result.converged);
    eprintln!("  n_steps                = {}", result.n_steps);
    eprintln!("  ε_final                = {:.8e} Ha", result.eigenvalue);
    eprintln!("  ‖r‖_S                  = {:.4e}", result.residual_norm);
    eprintln!("  ε_CASTEP (.check band0) = {:.8e} Ha", castep_eps_0);
    eprintln!("  |Δε| (CG vs CASTEP)     = {:.4e} Ha", (result.eigenvalue - castep_eps_0).abs());

    // ---- Gate 2 assertions --------------------------------------------------
    assert!(
        result.converged,
        "Gate 2 FAILED: CG did not converge within 50 steps (n_steps={})",
        result.n_steps,
    );

    assert!(
        result.residual_norm < 1e-6,
        "Gate 2 FAILED: residual norm {:.4e} >= 1e-6 Ha",
        result.residual_norm,
    );

    let delta_eps = (result.eigenvalue - castep_eps_0).abs();
    assert!(
        delta_eps < 1e-6,
        "Gate 2 FAILED: |ε_CG - ε_CASTEP| = {:.4e} >= 1e-6 Ha",
        delta_eps,
    );

    eprintln!();
    eprintln!("  Gate 2 PASSED: Subspace diag + CG refinement matches CASTEP precision.");
}
