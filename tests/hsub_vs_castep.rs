//! Compare H_sub = ψ^dag · H · ψ from Rust CPU computation against
//! CASTEP's H_sub_debug.dat from the H_dump fixture.
//!
//! ## Physics context
//!
//! For S-normalized USPP wavefunctions satisfying:
//!   H|ψ_b⟩ = ε_b · S|ψ_b⟩
//! the subspace matrix H_sub[i,j] = ⟨ψ_i|H|ψ_j⟩ should equal ε_i · δ_ij
//! at exact SCF convergence.  In practice, off-diagonal elements are
//! non-zero due to finite SCF tolerance and near-degenerate bands.
//!
//! This test isolates whether the CPU Hamiltonian application (T + V_loc
//! via FFT + V_NL via beta_phi + D) reproduces CASTEP's H_sub elementwise
//! to high precision.
//!
//! ## Components
//!
//! 1. **H_loc** = T + V_loc: applied via `apply_local_hamiltonian` (CPU
//!    FFT-based, one IFFT→V·→FFT roundtrip per band).
//! 2. **V_NL** = non-local pseudopotential contribution: computed via
//!    `beta_phi` (projector overlaps per USPP ion) + screened D matrices
//!    (D0 + ∫ Q·V_eff screening).
//!
//! No GPU code is used.
//!
//! Fixtures: Cu111_CO.check, Cu111_CO.castep_bin, Cu111_CO.pot_fmt,
//!           Cu111_CO.H_sub_debug.dat

use std::collections::HashMap;

use chemrust_hamiltonian_core::{
    CastepBinFile, CheckFile, EffectivePotential, GVectorGrid,
    augment::beta_phi::compute_beta_phi,
    fft::{fft_forward_3d, RealGrid},
    formatted,
    hamiltonian::apply_local_hamiltonian,
    nlpot::{build_d0_expanded, compute_screened_d_from_fft, precompute_q_on_grid, QOnGrid},
    pseudopotential::HasAugmentationData,
    types::{CellGeometry, KptWaveBlock, RecipLattice},
    Pseudopotential, PseudopotentialSet,
};
use num_complex::Complex64;

/// Directory containing the Cu111_CO H_dump fixture files.
/// Override via CASTEP_FIXTURE_DIR environment variable.
const H_DUMP_DIR: &str = "/export/public_castep_jobs/tony/Cu111_CO_H_dump";

/// Directory containing pseudopotential files.
/// Override via CASTEP_POTENTIAL_DIR environment variable.
const POTENTIAL_DIR: &str = "/export/Potentials";

/// Convert fractional G-vectors [h,k,l] to Cartesian using the reciprocal lattice.
/// (Duplicated from chemrust-hamiltonian-core::hamiltonian::fractional_to_cartesian
/// which is `pub(crate)`, not accessible from integration tests.)
fn fractional_to_cartesian_impl(
    pw_coords: &[[i32; 3]],
    recip_lattice: &RecipLattice,
) -> Vec<[f64; 3]> {
    let rl = recip_lattice.as_array();
    pw_coords
        .iter()
        .map(|&[h, k, l]| {
            let gf = [h as f64, k as f64, l as f64];
            std::array::from_fn(|j| (0..3).map(|i| gf[i] * rl[i][j]).sum())
        })
        .collect()
}

/// Compute H_sub[i,j] = ⟨ψ_i|T+V_loc+V_NL|ψ_j⟩ for the full band subspace.
///
/// Returns the n_bands × n_bands real matrix.  At Gamma point, H_sub is
/// real-symmetric because ψ(b) is real (CASTEP Gamma convention) and all
/// Hamiltonian components are Hermitian with zero imaginary part.
///
/// # Arguments
///
/// * `psi` — wavefunction coefficients, `psi[b][g]` for band b, PW g.
/// * `fft_indices` — PW-to-FFT-grid index mapping from `GVectorGrid::pw_to_fft_indices`.
/// * `gcart` — Cartesian G-vectors for each PW.
/// * `k_cart` — k-point in Cartesian (Gamma → [0,0,0]).
/// * `v_eff` — effective potential on the wave-function grid.
/// * `wave_grid` — G-vector grid matching the wavefunction.
/// * `cell` — cell geometry (ion positions, species).
/// * `pots` — pseudopotential set (species-indexed).
/// * `wave_block` — KptWaveBlock for beta_phi projector overlap computation.
///
/// Returns n_bands × n_bands matrix `h_sub[i][j]` (0-indexed).
fn compute_hsub_cpu(
    psi: &[Vec<Complex64>],
    fft_indices: &[[usize; 3]],
    gcart: &[[f64; 3]],
    k_cart: [f64; 3],
    v_eff: &EffectivePotential,
    wave_grid: &GVectorGrid,
    cell: &CellGeometry,
    pots: &PseudopotentialSet,
    wave_block: &KptWaveBlock,
) -> Vec<Vec<f64>> {
    let n_bands = psi.len();

    // -----------------------------------------------------------------------
    // Part 1: H_loc|ψ_j⟩ = (T + V_loc)|ψ_j⟩  (CPU FFT, one per band)
    // -----------------------------------------------------------------------
    let mut h_loc_psi: Vec<Vec<Complex64>> = Vec::with_capacity(n_bands);
    for b in 0..n_bands {
        let hpsi = apply_local_hamiltonian(
            &psi[b], fft_indices, gcart, k_cart, v_eff, wave_grid,
        )
        .expect("apply_local_hamiltonian failed");
        h_loc_psi.push(hpsi);
    }

    // -----------------------------------------------------------------------
    // Part 2: V_NL[i,j]  (via beta_phi + screened D matrices)
    //
    // For each USPP ion:
    //   1. Compute beta_phi[n,b] = ⟨ψ_b|β_n⟩  (projector overlap, real at Γ)
    //   2. Build D_screen = D0 + ΔD(V_eff)  (screening integral via Q_on_grid)
    //   3. VNL_ij += Σ_{n,m} D_screen[n,m] · beta_phi[n,i] · beta_phi[m,j]
    //
    // The matrix form: VNL = Σ_ion beta_phi^T · D_screen · beta_phi
    // -----------------------------------------------------------------------

    // Pre-compute V_eff FFT once (used by all D-screening calls).
    let v_eff_fft = fft_forward_3d(v_eff.as_real_grid())
        .expect("V_eff FFT for D screening failed");

    // Per-species caches: QOnGrid (heavy, depends only on aug data) and D0.
    let mut q_cache: HashMap<String, QOnGrid> = HashMap::new();
    let mut d0_cache: HashMap<String, ndarray::Array2<f64>> = HashMap::new();

    // Prepopulate caches for all USP species.
    for species_idx in 0..cell.num_species {
        let symbol = &cell.species_symbols[species_idx];
        let pot = pots.get(symbol).unwrap_or_else(|| {
            panic!("pseudopotential for species '{symbol}' not found");
        });
        if !pot.has_augmentation() {
            continue;
        }
        if q_cache.contains_key(symbol) {
            continue; // already cached
        }
        let aug: &dyn HasAugmentationData = match pot {
            Pseudopotential::Usp(d) => d,
            _ => continue,
        };
        let q_on_grid =
            precompute_q_on_grid(aug, wave_grid).expect("precompute_q_on_grid failed");
        let d0 = build_d0_expanded(aug);
        q_cache.insert(symbol.clone(), q_on_grid);
        d0_cache.insert(symbol.clone(), d0);
    }

    // Accumulate VNL matrix over all USPP ions.
    let mut vnl_mat = vec![vec![0.0_f64; n_bands]; n_bands];

    for ion_idx in 0..cell.num_ions {
        let species_idx = cell.ion_species[ion_idx];
        let symbol = &cell.species_symbols[species_idx];
        let pot = pots.get(symbol).unwrap_or_else(|| {
            panic!("pseudopotential for species '{symbol}' (ion {ion_idx}) not found");
        });

        // NC pseudopotentials have no augmentation → no V_NL contribution.
        if !pot.has_augmentation() {
            continue;
        }

        let aug: &dyn HasAugmentationData = match pot {
            Pseudopotential::Usp(d) => d,
            _ => continue,
        };

        // Beta-phi projector overlaps (⟨ψ_b|β_n⟩).
        let beta_phi = compute_beta_phi(wave_block, aug, cell, ion_idx, wave_grid, pot.gmax(), k_cart)
            .expect("compute_beta_phi failed");

        // Screened D matrix for this ion.
        let d0 = d0_cache.get(symbol).expect("D0 cache missing");
        let q_on_grid = q_cache.get(symbol).expect("QOnGrid cache missing");
        let d_screen = compute_screened_d_from_fft(q_on_grid, &v_eff_fft, cell, ion_idx, wave_grid, d0);

        let ne = d_screen.shape()[0]; // n_expanded projectors

        // V_NL[i,j] = Σ_{ion} Σ_{n,m} D[n,m] · Re(conj(β[n,i]) · β[m,j])
        //
        // CASTEP nlpot.f90:2856 (nlpot_apply_add_slice, non-Gamma path):
        //   nl_eigenvalues(nb) += real(tmp * conjg(beta_phi(n, nb)))
        //   where tmp = Σ_m nl_d(m,n,...) * beta_phi(m, nb)
        //
        // At non-Gamma k-points, beta_phi is complex-valued.
        // The full Re(conj(β[n])·β[m]) = Re(β[n])·Re(β[m]) + Im(β[n])·Im(β[m])
        // must be used, not just Re(β[n])·Re(β[m]).
        for n in 0..ne {
            // db[j] = Σ_m D[n,m] · β[m,j]  (complex accumulator)
            let mut db = vec![Complex64::ZERO; n_bands];
            for m in 0..ne {
                let d_nm = d_screen[[n, m]];
                if d_nm.abs() <= 1e-30 {
                    continue;
                }
                for j in 0..n_bands {
                    db[j] += d_nm * beta_phi[[m, j]];
                }
            }
            // VNL[i,j] += Re(conj(β[n,i]) · db[j])
            for i in 0..n_bands {
                let bn_i = beta_phi[[n, i]];
                if bn_i.norm_sqr() <= 1e-30 {
                    continue;
                }
                for j in 0..n_bands {
                    vnl_mat[i][j] += (bn_i.conj() * db[j]).re;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Part 3: H_sub[i,j] = ⟨ψ_i|H_loc|ψ_j⟩ + VNL_ij
    // -----------------------------------------------------------------------
    let mut h_sub = vec![vec![0.0_f64; n_bands]; n_bands];
    for i in 0..n_bands {
        let psi_i = &psi[i];
        for j in 0..n_bands {
            // ⟨ψ_i|T+V_loc|ψ_j⟩ via inner product with precomputed H_loc|ψ_j⟩
            let hloc_ij: f64 = psi_i
                .iter()
                .zip(h_loc_psi[j].iter())
                .map(|(&c, &hp)| (c.conj() * hp).re)
                .sum();
            h_sub[i][j] = hloc_ij + vnl_mat[i][j];
        }
    }

    // Component decomposition for band 0 (diagnostic).
    {
        let i = 0;
        let psi_i = &psi[i];
        // T_00 = <psi_0|T|psi_0> — pure G-space, no FFT
        let mut t_00 = 0.0_f64;
        for g in 0..gcart.len() {
            let c2 = psi_i[g].norm_sqr();
            let gk = [
                gcart[g][0] + k_cart[0],
                gcart[g][1] + k_cart[1],
                gcart[g][2] + k_cart[2],
            ];
            let ekin = 0.5 * (gk[0] * gk[0] + gk[1] * gk[1] + gk[2] * gk[2]);
            t_00 += c2 * ekin;
        }
        // H_loc_00 = <psi_0|H_loc|psi_0> (FFT-based, Path A)
        let hloc_00: f64 = psi_i
            .iter()
            .zip(h_loc_psi[i].iter())
            .map(|(&c, &hp)| (c.conj() * hp).re)
            .sum();
        let vloc_path_a = hloc_00 - t_00;
        // Path B: local_potential_expectation (independent IFFT → |ψ|²*V_eff → integrate)
        let vloc_path_b = chemrust_hamiltonian_core::hamiltonian::local_potential_expectation(
            psi_i, fft_indices, v_eff, wave_grid,
        ).unwrap_or(f64::NAN);
        let vnl_00 = vnl_mat[i][i];
        eprintln!(
            "[hsub_vs_castep] Band 0 decomposition: T={:.10e}  Vloc(A)={:.10e}  Vloc(B)={:.10e}  VNL={:.10e}  H_sub={:.10e}  (expected ε_0={:.10e})",
            t_00, vloc_path_a, vloc_path_b, vnl_00, hloc_00 + vnl_00, -1.05503050_f64,
        );
        // Wavefunction norm check
        let norm2: f64 = psi_i.iter().map(|c| c.norm_sqr()).sum();
        eprintln!(
            "[hsub_vs_castep] Band 0 norm² = {:.10e} (expected 1.0 for S-orthonormal)",
            norm2,
        );
        // V_eff grid statistics
        let v_arr = v_eff.as_real_grid().as_real_array();
        let (v_min, v_max, v_mean) = {
            let n = v_arr.len() as f64;
            let mut min = f64::MAX;
            let mut max = f64::MIN;
            let mut sum = 0.0;
            for &v in v_arr.iter() {
                min = min.min(v);
                max = max.max(v);
                sum += v;
            }
            (min, max, sum / n)
        };
        eprintln!(
            "[hsub_vs_castep] V_eff grid stats: min={:.6e}  max={:.6e}  mean={:.6e}  n_grid={}",
            v_min, v_max, v_mean, v_arr.len(),
        );

        // ---- Verify .pot_fmt parser against raw file values ----
        // Read raw .pot_fmt to verify parser correctness
        let pot_fmt_path = format!("{H_DUMP_DIR}/Cu111_CO.pot_fmt");
        if let Ok(raw_pot) = std::fs::read_to_string(&pot_fmt_path) {
            // Find a few specific grid points in the raw text and compare
            // Look for lines: "    1     1     1    -2.674890" etc.
            let raw_v111 = raw_pot.lines()
                .find(|l| l.trim().starts_with("1     1     1 ") || l.trim().starts_with("1 1 1 "))
                .and_then(|l| l.split_whitespace().last()?.parse::<f64>().ok());
            let raw_v221 = raw_pot.lines()
                .find(|l| l.trim().starts_with("2     2     1 ") || l.trim().starts_with("2 2 1 "))
                .and_then(|l| l.split_whitespace().last()?.parse::<f64>().ok());
            let raw_v112 = raw_pot.lines()
                .find(|l| l.trim().starts_with("1     1     2 ") || l.trim().starts_with("1 1 2 "))
                .and_then(|l| l.split_whitespace().last()?.parse::<f64>().ok());
            eprintln!(
                "[hsub_vs_castep] Raw .pot_fmt: V(1,1,1)={raw}  V(2,2,1)={raw2}  V(1,1,2)={raw3}",
                raw = raw_v111.map_or("N/A".to_string(), |v| format!("{:.6e}", v)),
                raw2 = raw_v221.map_or("N/A".to_string(), |v| format!("{:.6e}", v)),
                raw3 = raw_v112.map_or("N/A".to_string(), |v| format!("{:.6e}", v)),
            );
            eprintln!(
                "[hsub_vs_castep] Our parsed: V(1,1,1)={:.6e}  V(2,2,1)={:.6e}  V(1,1,2)={:.6e}",
                v_arr[[0, 0, 0]],
                v_arr[[1, 1, 0]],
                v_arr[[0, 0, 1]],
            );
        }
    }

    h_sub
}

#[test]
fn hsub_vs_castep() {
    // ---- Load fixtures ----
    let fixture_dir =
        std::env::var("CASTEP_FIXTURE_DIR").unwrap_or_else(|_| H_DUMP_DIR.to_string());
    let potential_dir =
        std::env::var("CASTEP_POTENTIAL_DIR").unwrap_or_else(|_| POTENTIAL_DIR.to_string());

    // 1. .check file (converged wavefunctions + grid metadata)
    let check_path = format!("{fixture_dir}/Cu111_CO.check");
    let check_file = std::fs::File::open(&check_path).unwrap_or_else(|e| {
        panic!("cannot open {check_path}: {e} — set CASTEP_FIXTURE_DIR if needed");
    });
    let check = CheckFile::read(std::io::BufReader::new(check_file)).unwrap_or_else(|e| {
        panic!("failed to parse {check_path}: {e}");
    });

    // 2. .castep_bin file (cell geometry, species)
    let bin_path = format!("{fixture_dir}/Cu111_CO.castep_bin");
    let bin_file = std::fs::File::open(&bin_path).unwrap_or_else(|e| {
        panic!("cannot open {bin_path}: {e} — set CASTEP_FIXTURE_DIR if needed");
    });
    let bin = CastepBinFile::read(std::io::BufReader::new(bin_file)).unwrap_or_else(|e| {
        panic!("failed to parse {bin_path}: {e}");
    });

    // 3. .pot_fmt file (reference V_eff on wave grid)
    let pot_path = format!("{fixture_dir}/Cu111_CO.pot_fmt");
    let pot_text = std::fs::read_to_string(&pot_path).unwrap_or_else(|e| {
        panic!("cannot read {pot_path}: {e} — set CASTEP_FIXTURE_DIR if needed");
    });
    let (_pot_grid, pot_arr) = formatted::parse_pot_fmt(&pot_text).unwrap_or_else(|e| {
        panic!("failed to parse {pot_path}: {e}");
    });

    // 4. Pseudopotentials
    let pots = PseudopotentialSet::from_dir(
        potential_dir,
        &bin.cell.species_symbols,
        &bin.cell.species_pot_files,
    )
    .unwrap_or_else(|e| {
        panic!("failed to load pseudopotentials: {e} — set CASTEP_POTENTIAL_DIR if needed");
    });

    // ---- Extract wavefunction data ----
    let wfc = check
        .wavefunction
        .as_ref()
        .expect(".check must have wavefunction section");
    // Cu111_CO is gamma-point only, non-spin-polarised
    assert_eq!(wfc.kpt_data.len(), 1, "expected 1 k-point");
    let kpt = &wfc.kpt_data[0];
    let n_bands = kpt.bands.len();
    let n_pw = kpt.nplw;
    let pw_coords = &kpt.pw_grid_coord;

    eprintln!(
        "[hsub_vs_castep] n_bands={n_bands}, n_pw={n_pw}, grid={:?}",
        wfc.grid,
    );

    // ---- Build computation objects ----
    let [ngx, ngy, ngz] = wfc.grid;
    let cell = &bin.cell;
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);
    let fft_indices = wave_grid.pw_to_fft_indices(pw_coords);
    let gcart = fractional_to_cartesian_impl(pw_coords, &cell.recip_lattice);
    // Convert k-point from fractional to Cartesian using the reciprocal lattice.
    // The calculation uses an MP grid [2,1,1] with k_frac = [-0.25, 0, 0]
    // (non-Gamma), so hardcoding [0.0; 3] would evaluate T and beta at wrong |G+k|.
    let rl = cell.recip_lattice.as_array();
    let k_frac = kpt.coords;
    let k_cart: [f64; 3] = std::array::from_fn(|j| (0..3).map(|i| k_frac[i] * rl[i][j]).sum());
    let v_eff = EffectivePotential::from_inner(RealGrid::from_inner(pot_arr));

    // ---- KptWaveBlock for beta_phi projector calculations ----
    let wave_block = KptWaveBlock {
        coords: kpt.coords,
        nplw: n_pw,
        pw_grid_coord: pw_coords.clone(),
        bands: kpt.bands.clone(),
    };

    // ---- Compute H_sub on CPU ----
    eprintln!("[hsub_vs_castep] Computing H_sub on CPU (this may take a while)...");
    let psi_slices: Vec<Vec<Complex64>> = kpt.bands.clone();
    let h_sub = compute_hsub_cpu(
        &psi_slices,
        &fft_indices,
        &gcart,
        k_cart,
        &v_eff,
        &wave_grid,
        cell,
        &pots,
        &wave_block,
    );
    eprintln!("[hsub_vs_castep] H_sub computed.");

    // ---- Parse H_sub_debug.dat ----
    let dat_path = format!("{fixture_dir}/Cu111_CO.H_sub_debug.dat");
    let dat_text = std::fs::read_to_string(&dat_path).unwrap_or_else(|e| {
        panic!("cannot read {dat_path}: {e} — set CASTEP_FIXTURE_DIR if needed");
    });
    let ref_lines: Vec<&str> = dat_text.lines().collect();
    let n_bands_ref: usize = ref_lines[0].trim().parse().expect("parse n_bands from header");
    assert_eq!(
        n_bands_ref, n_bands,
        "n_bands from H_sub_debug.dat ({n_bands_ref}) does not match .check ({n_bands})",
    );

    let mut h_sub_ref = vec![vec![0.0_f64; n_bands]; n_bands];
    for line in &ref_lines[1..] {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }
        let i: usize = parts[0].parse::<usize>().expect("parse band_i") - 1; // 1→0 based
        let j: usize = parts[1].parse::<usize>().expect("parse band_j") - 1;
        let re: f64 = parts[2].parse().expect("parse H_sub_re");
        h_sub_ref[i][j] = re;
    }

    // ---- Per-band diagonal comparison vs CASTEP reference ----
    eprintln!("[hsub_vs_castep] Per-band diagonal comparison (first 10 bands):");
    for i in 0..10.min(n_bands) {
        let psi_b = &psi_slices[i];
        let mut t_b = 0.0_f64;
        for g in 0..gcart.len() {
            let c2 = psi_b[g].norm_sqr();
            let gk = [gcart[g][0] + k_cart[0], gcart[g][1] + k_cart[1], gcart[g][2] + k_cart[2]];
            t_b += c2 * 0.5 * (gk[0] * gk[0] + gk[1] * gk[1] + gk[2] * gk[2]);
        }
        let vloc_vnl_rust = h_sub[i][i] - t_b;
        let vloc_vnl_ref = h_sub_ref[i][i] - t_b;
        eprintln!(
            "[hsub_vs_castep]   band {i}: T={:.6e}  Vloc+VNL(rust)={:.6e}  Vloc+VNL(ref)={:.6e}  H(rust)={:.6e}  H(ref)={:.6e}  diff={:.6e}",
            t_b, vloc_vnl_rust, vloc_vnl_ref, h_sub[i][i], h_sub_ref[i][i], h_sub[i][i] - h_sub_ref[i][i],
        );
    }

    // ---- Compare ----
    let mut max_diff = 0.0_f64;
    let mut max_diff_i = 0;
    let mut max_diff_j = 0;
    let mut max_diff_diag = 0.0_f64;
    let mut max_diff_offdiag = 0.0_f64;

    for i in 0..n_bands {
        for j in 0..n_bands {
            let diff = (h_sub[i][j] - h_sub_ref[i][j]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_i = i;
                max_diff_j = j;
            }
            if i == j {
                max_diff_diag = max_diff_diag.max(diff);
            } else {
                max_diff_offdiag = max_diff_offdiag.max(diff);
            }
        }
    }

    eprintln!("[hsub_vs_castep] H_sub max|diff| = {:.6e} Ha (at [{max_diff_i},{max_diff_j}])", max_diff);
    eprintln!("[hsub_vs_castep]   max|diff| diagonal    = {:.6e}", max_diff_diag);
    eprintln!("[hsub_vs_castep]   max|diff| off-diagonal = {:.6e}", max_diff_offdiag);
    eprintln!(
        "[hsub_vs_castep]   H_sub[0,0]: rust={:.10e}  ref={:.10e}",
        h_sub[0][0], h_sub_ref[0][0],
    );
    eprintln!(
        "[hsub_vs_castep]   H_sub[{max_diff_i},{max_diff_j}]: rust={:.10e}  ref={:.10e}",
        h_sub[max_diff_i][max_diff_j], h_sub_ref[max_diff_i][max_diff_j],
    );

    // Diagonal tolerance: 5e-4 Ha.  O-dominated bands (0-9) match to
    // ~17 µHa; higher bands show ~0.4 meV from SCF convergence floor.
    // Off-diagonal tolerance: 1e-3 Ha (pre- vs post-rotation ψ mismatch).
    // Source: H_sub_debug.dat from CASTEP H_dump fixture.
    assert!(
        max_diff_diag < 5e-4,
        "H_sub max|diff| diagonal = {:.6e} Ha exceeds 5e-4 Ha",
        max_diff_diag,
    );
    assert!(
        max_diff < 1e-3,
        "H_sub max|diff| = {:.6e} Ha exceeds 1e-3 Ha (at [{max_diff_i},{max_diff_j}]: \
         rust={:.10e} ref={:.10e})",
        max_diff,
        h_sub[max_diff_i][max_diff_j],
        h_sub_ref[max_diff_i][max_diff_j],
    );
}
