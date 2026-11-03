//! Pure cold-start SCF on the non-spin-polarised `Cu111_CO_H` fixture.
//!
//! What this proves: the full pure-Rust SCF loop (`run_scf_with_energy_gated`,
//! GPU diagonalization) started from a **non-CASTEP initial guess** reaches
//! the same SCF fixed point as the CASTEP cold-start job that produced the
//! `Cu111_CO_H_dump` fixture. The converged eigenvalues are compared against
//! the reference `.bands` file.
//!
//! # Cold-start protocol (no CASTEP electronic data)
//!
//! - Default `CHEMRUST_COLD_INIT=lcao`: the CASTEP-faithful LCAO init, via
//!   `chemrust_scf::build_lcao_initial_state` (the pure-Rust
//!   `chemrust_hamiltonian_core::init` chain). The initial density is the
//!   LCAO atomic superposition (the pseudo-atom charge, the pure-Rust
//!   `atomic_solver` output) and the initial wavefunctions are the LCAO 'R'
//!   random fill S-orthonormalised against the LCAO overlap. This is what
//!   the CASTEP FFI run uses; it screens the first V_eff with V_H / V_xc so
//!   the TPA preconditioner stays damped.
//! - `CHEMRUST_COLD_INIT=zero`: the bare-ionic diagnostic (zero density +
//!   empty-orbital or random wavefunctions, `CHEMRUST_COLD_GUESS`). With no
//!   initial charge the first V_eff is bare ionic and the TPA filter
//!   amplifies the deep V_loc wells (band-0 -> -8627 Ha, the observed
//!   explosion). Kept only to characterise that failure.
//! - Eigensolver: Davidson (the default). The Chebyshev cold start is known
//!   to overflow on pass 2 (HANDOFF.md "Pending: fix cold start"). To force
//!   the Chebyshev path anyway, run with `CHEMRUST_EIGENSOLVER=chebyshev`.
//!
//! # CASTEP reference (Cu111_CO_H_dump job, `Cu111_CO.param`)
//!
//! - 1 k-point (-0.25, 0, 0), 160 bands, 186 electrons, `SPIN_POLARIZED: false`
//! - Gaussian smearing 0.1 eV; `MIXING_SCHEME: Pulay` (amp 0.5, gmax 1.5 1/Å,
//!   history 20); `ELEC_ENERGY_TOL: 1e-5 eV` per atom, window 3 cycles.
//! - CASTEP was itself a cold start (LCAO): 12 SCF cycles to E-TS =
//!   -24111.21907515 eV. The reference eigenvalues in `.bands` come from
//!   that converged state.
//!
//! # VRAM
//!
//! 160 bands × 60067 plane waves. `davidson_hdump_validation` budgets ~12 GB
//! for this fixture. A GTX 1080 Ti (11 GB) is expected to fit the Davidson
//! path; watch for OOM. The Hamiltonian application alone was measured on a
//! GTX 1080 Ti for the same system (notes/plans/GVEC_PARALLELISM_REVIEW.md).
//!
//! # Run (nixos-2, GPU machine)
//!
//! ```sh
//! cd chemrust-scf && cargo test --release --test cu111_co_cold_start -- --ignored
//! ```
//!
//! Env overrides:
//! - `CHEMRUST_EIGENSOLVER` — eigensolver (`davidson` default; `chebyshev` known-broken for cold start)
//! - `CHEMRUST_COLD_INIT`   — initial-state kind (`lcao` default; `zero` the bare-ionic diagnostic)
//! - `CHEMRUST_COLD_SEED`   — LCAO 'R' (or zero+random) guess seed (u64 decimal, `0xC0FFEE` default)
//! - `CHEMRUST_COLD_GUESS`  — zero-init wavefunction kind (`empty_orbital` default; `random` diagnostic)
//! - `CHEMRUST_COLD_TOL_HA` — energy-window convergence tolerance, Hartree
//! - `CHEMRUST_COLD_MAX_ITER` — iteration ceiling for the divergence gate
//! - `CHEMRUST_NVRTC_ARCH` — NVRTC kernel target arch. Unset (portable
//!   default: NVRTC-default PTX, JITs on any modern GPU incl. Pascal
//!   sm_61). Set `sm_120` on the Blackwell dev box (RTX PRO 5000).
//!   A hardcoded `sm_120` fails on Pascal with CUDA_ERROR_INVALID_PTX.
//! - `CASTEP_FIXTURE_DIR` / `CASTEP_POTENTIAL_DIR` — fixture locations

use std::sync::Arc;

use chemrust_hamiltonian_core::{
    CastepBin, CheckFile, GVectorGrid, PseudopotentialSet, NonSpin,
};
use chemrust_scf::{
    Density, KPoint, KptDataSet, PwCoefficients, PerSpinDensity,
    PerSpinPwCoefficients, ScfDivergenceGate, ScfIteration, SmearingParams, SmearingScheme,
    SmearingWidth, SpinChannelData, WaveGridArray,
    compute_kinetic_energies, run_scf_with_energy_gated,
};
use cudarc::driver::CudaContext;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Directory containing the Cu111_CO H_dump fixture files.
const H_DUMP_DIR: &str = "/export/public_castep_jobs/tony/Cu111_CO_H_dump";

/// Pseudopotential library.
const POTENTIAL_DIR: &str = "/export/Potentials";

/// Seed for the deterministic cold-start initial guess.
const COLD_SEED: u64 = 0xC0FFEE;

/// Eigenvalue hard gate vs `.bands` (Ha). A cold-start SCF converged to
/// CASTEP's 1e-5 eV/atom window should agree to ~1e-4 Ha; 1e-3 Ha is the
/// fail threshold (a broken eigensolver misses by O(0.1) Ha).
const EIGVAL_TOL_HA: f64 = 1e-3;

/// Soft threshold: reported as WARN (still passes) — the level a healthy
/// converged run should clear.
const EIGVAL_WARN_HA: f64 = 1e-4;

/// CASTEP `.param` density-mixing parameters for the H_dump job.
const MIX_CHARGE_GMAX_INV_A: f64 = 1.5; // 1/Å
const INV_ANG_TO_INV_BOHR: f64 = 1.88972612545;
const MIX_HISTORY_LENGTH: usize = 20;

/// CASTEP ELEC_ENERGY_TOL in eV PER ATOM (Cu111_CO.param line 27).
const CONV_TOL_EV_PER_ATOM: f64 = 1e-5;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse eigenvalues from a CASTEP `.bands` file (single k-point,
/// single spin component: the "Spin component N" block of the H_dump job).
fn parse_bands_file(text: &str) -> Result<Vec<f64>, String> {
    let eigenvalues: Vec<f64> = text
        .lines()
        .skip_while(|line| !line.trim().starts_with("Spin component"))
        .skip(1)
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect();
    if eigenvalues.is_empty() {
        return Err("no eigenvalues parsed from .bands file".into());
    }
    Ok(eigenvalues)
}

/// Seeded random initial guess: `n_bands` complex vectors of length `n_pw`,
/// uniform [-1, 1) real/imaginary parts, L2-normalised per band.
///
/// DIAGNOSTIC ONLY: a random subspace makes the TPA preconditioner's
/// mean-ek estimate poor, which under-damps and explodes on cold start
/// (davidson.rs "cold-start" comments; observed as the [Diag-PreconEig]
/// cascade to ~-170 Ha on nixos-2). Kept behind CHEMRUST_COLD_GUESS=random.
fn random_initial_guess(n_bands: usize, n_pw: usize, seed: u64) -> Vec<Complex64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n_bands * n_pw);
    for _ in 0..n_bands {
        let band: Vec<Complex64> = (0..n_pw)
            .map(|_| {
                Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
            })
            .collect();
        let norm = band
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let inv_norm = 1.0 / norm.max(1e-300);
        out.extend(band.iter().map(|c| *c * inv_norm));
    }
    out
}

/// Empty-orbital cold-start initial guess: band i is the plane-wave basis
/// vector at the i-th lowest-kinetic-energy G point (0.5|G+k|^2 order).
///
/// Deterministic and non-CASTEP: it uses only the PW basis and the cell.
/// This is the standard empty-orbital initialisation of plane-wave codes.
/// It is the diagnostic fallback (CHEMRUST_COLD_INIT=zero); the default
/// LCAO init is preferred because a zero initial density leaves the first
/// V_eff bare-ionic and the TPA filter amplifies the deep V_loc wells.
fn empty_orbital_guess(n_bands: usize, n_pw: usize, kinetic: &[f64]) -> Vec<Complex64> {
    let mut order: Vec<usize> = (0..n_pw).collect();
    order.sort_by(|&a, &b| kinetic[a].partial_cmp(&kinetic[b]).unwrap());
    let mut out = vec![Complex64::ZERO; n_bands * n_pw];
    for (i, &g) in order.iter().take(n_bands).enumerate() {
        out[i * n_pw + g] = Complex64::new(1.0, 0.0);
    }
    out
}

/// Returns `true` if a CUDA-capable GPU is available at device 0.
fn gpu_available() -> bool {
    std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires GPU (~12 GB VRAM) and CASTEP fixture data at /export/public_castep_jobs/tony/Cu111_CO_H_dump"]
fn cu111_co_h_cold_start_vs_bands() {
    if !gpu_available() {
        eprintln!("SKIP: no CUDA-capable GPU available at device 0");
        return;
    }

    // Eigensolver selection: Davidson by default. An explicit user
    // CHEMRUST_EIGENSOLVER is honoured (the Chebyshev cold start is known
    // to overflow on pass 2 — see HANDOFF.md).
    if std::env::var("CHEMRUST_EIGENSOLVER").is_err() {
        unsafe {
            std::env::set_var("CHEMRUST_EIGENSOLVER", "davidson");
        }
        eprintln!("[cold] eigensolver=davidson (default; CHEMRUST_EIGENSOLVER unset)");
    } else {
        eprintln!(
            "[cold] eigensolver={} (user override)",
            std::env::var("CHEMRUST_EIGENSOLVER").unwrap()
        );
    }

    // -------------------------------------------------------------------
    // Load fixture: problem setup only (cell, grids, PW coords, pots).
    // CheckFile::read also loads the CASTEP wavefunctions; they are NOT
    // used by this test (cold start).
    // -------------------------------------------------------------------
    let fixture_dir =
        std::env::var("CASTEP_FIXTURE_DIR").unwrap_or_else(|_| H_DUMP_DIR.to_string());
    let potential_dir =
        std::env::var("CASTEP_POTENTIAL_DIR").unwrap_or_else(|_| POTENTIAL_DIR.to_string());

    let check: CastepBin = CheckFile::read(
        std::io::BufReader::new(
            std::fs::File::open(format!("{fixture_dir}/Cu111_CO.check"))
                .expect("open Cu111_CO.check"),
        ),
    )
    .expect("read Cu111_CO.check");

    let bands_ref = parse_bands_file(
        &std::fs::read_to_string(format!("{fixture_dir}/Cu111_CO.bands"))
            .expect("read Cu111_CO.bands"),
    )
    .expect("parse Cu111_CO.bands");

    let wfc = check.wavefunction.as_ref().expect(".check must have wavefunction");
    assert_eq!(wfc.kpt_data.len(), 1, "H_dump fixture must be single-k-point");
    let kpt = &wfc.kpt_data[0];
    let n_bands = kpt.bands.len();
    let n_pw = kpt.nplw;
    let [ngx, ngy, ngz] = wfc.grid;
    let cell = &check.cell;
    assert_eq!(
        bands_ref.len(),
        n_bands,
        ".bands has {} eigenvalues but .check has {} bands",
        bands_ref.len(),
        n_bands
    );

    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);
    let [fgx, fgy, fgz] = check.fine_grid.expect(".check must have fine_grid");
    let fine_grid = GVectorGrid::new(fgx, fgy, fgz, cell.recip_lattice);

    // Plane-wave kinetic cutoff (Ha): bounds the early Rayleigh-Ritz of the
    // random initial guess. Computed before `wave_grid` is moved into the
    // builder.
    let kinetic = compute_kinetic_energies(
        &kpt.pw_grid_coord,
        wave_grid.recip_lattice(),
        kpt.coords,
    );
    let ecut_ha = kinetic.0.iter().cloned().fold(0.0_f64, f64::max);

    let pots = PseudopotentialSet::from_dir(
        &potential_dir,
        &cell.species_symbols,
        &cell.species_pot_files,
    )
    .expect("load pseudopotentials");

    eprintln!(
        "[cold] n_bands={n_bands} n_pw={n_pw} wave_grid=[{ngx},{ngy},{ngz}] fine_grid=[{fgx},{fgy},{fgz}] k=({:.4},{:.4},{:.4})",
        kpt.coords[0], kpt.coords[1], kpt.coords[2]
    );
    eprintln!(
        "[cold] reference .bands: first={:.6} Ha last={:.6} Ha (n={})",
        bands_ref.first().copied().unwrap_or(f64::NAN),
        bands_ref.last().copied().unwrap_or(f64::NAN),
        bands_ref.len()
    );

    // -------------------------------------------------------------------
    // Cold-start initial state.
    //
    // Default `CHEMRUST_COLD_INIT=lcao`: the CASTEP-faithful LCAO init (a
    // physical LCAO atomic-superposition initial density + LCAO 'R'
    // S-orthonormalised initial wavefunctions, all pure Rust). This is what
    // the CASTEP FFI run uses and what keeps the first V_eff screened by
    // V_H/V_xc. `CHEMRUST_COLD_INIT=zero` reverts to the bare-ionic
    // diagnostic (zero density + empty-orbital or random wavefunctions,
    // `CHEMRUST_COLD_GUESS`): it reproduces the TPA V_loc-well amplification
    // (band-0 -> -8627 Ha) and is kept only to characterise the failure.
    // -------------------------------------------------------------------
    let init_kind = std::env::var("CHEMRUST_COLD_INIT").unwrap_or_else(|_| "lcao".into());
    let seed: u64 = std::env::var("CHEMRUST_COLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(COLD_SEED);

    let (psi0, guess_label, density): (Vec<Complex64>, String, Density) = if init_kind == "lcao" {
        let lcao = chemrust_scf::build_lcao_initial_state(
            &fixture_dir,
            &potential_dir,
            n_bands,
            n_pw,
            &kpt.pw_grid_coord,
            &kpt.coords,
            [fgx as usize, fgy as usize, fgz as usize],
            &wave_grid,
            &fine_grid,
            &check.cell.ionic_positions,
            seed as i64,
        )
        .expect("LCAO cold-start initial state");
        eprintln!(
            "[cold] LCAO init: charge pre-VCA={:.6} post-VCA={:.6} (expect 186.0); wave = LCAO 'R' seed={seed}",
            lcao.elec_total_pre_vca, lcao.elec_total_post_vca
        );
        (lcao.wave, format!("LCAO 'R' S-orthonormalised (seed={seed})"), lcao.density)
    } else {
        let guess_kind =
            std::env::var("CHEMRUST_COLD_GUESS").unwrap_or_else(|_| "empty_orbital".into());
        let (psi, label) = match guess_kind.as_str() {
            "random" => (
                random_initial_guess(n_bands, n_pw, seed),
                format!("seeded random (seed={seed})"),
            ),
            _ => (
                empty_orbital_guess(n_bands, n_pw, &kinetic.0),
                "empty-orbital".to_string(),
            ),
        };
        let zero = Density::from_inner(WaveGridArray::from_inner(
            ndarray::Array3::<f64>::zeros([ngx, ngy, ngz]),
        ));
        eprintln!("[cold] zero-density diagnostic: guess={label}");
        (psi, label, zero)
    };
    eprintln!("[cold] initial state: psi={guess_label}");

    let k_point = KPoint {
        coords: kpt.coords,
        weight: 1.0,
    };
    let pw_coords = KptDataSet::new(vec![kpt.pw_grid_coord.clone()], 1);
    let pw_fft_indices_data = chemrust_scf::pw_coords_to_fft_indices(
        &kpt.pw_grid_coord,
        &wave_grid,
    );
    let pw_fft_indices = KptDataSet::new(vec![pw_fft_indices_data], 1);

    let density_ps = PerSpinDensity::new(SpinChannelData::new::<NonSpin>(vec![
        density,
    ]));

    // CPU-side psi (diagonalize uploads from here). GPU-side psi:
    // placeholder, as in the fixture builders.
    let psi_cpu: SpinChannelData<KptDataSet<Vec<Complex64>>> =
        SpinChannelData::new::<NonSpin>(vec![KptDataSet::new(vec![psi0], 1)]);
    let ctx = Arc::new(CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();
    let psi_gpu = PwCoefficients::new(
        stream
            .alloc_zeros::<chemrust_scf::device::CudaComplex>(0)
            .expect("dummy psi alloc"),
    );
    let psi_gpu_ps = PerSpinPwCoefficients::new(SpinChannelData::new::<NonSpin>(
        vec![KptDataSet::new(vec![psi_gpu], 1)],
    ));

    // Smearing + mixing exactly as Cu111_CO.param: Gaussian 0.1 eV,
    // Pulay mixing (the loop's automatic Off→Kerker→Pulay transition
    // mirrors CASTEP's dm_sub_mix gating).
    let smearing = SmearingParams {
        width: SmearingWidth::ev(0.1),
        electron_temperature: 0.1 * chemrust_scf::EV_TO_HARTREE,
        scheme: SmearingScheme::Gaussian,
        spin_fix: -1, // non-spin: CASTEP gates spin_fix on spin_polarised
        mixing_scheme: chemrust_scf::MixingScheme::Pulay,
        net_spin: 0.0,
    };

    // The explicit annotation pins S = NonSpin (the `ScfIteration` alias
    // default) before the generic builder chain is type-checked.
    let state: chemrust_scf::ScfIteration = ScfIteration::builder()
        .cell(cell.clone())
        .pots(pots)
        .wave_grid(wave_grid)
        .fine_grid(fine_grid)
        .density(density_ps)
        .psi(psi_gpu_ps)
        .psi_data(psi_cpu)
        .pw_coords(pw_coords)
        .pw_fft_indices(pw_fft_indices)
        .k_points(KptDataSet::new(vec![k_point], 1))
        .smearing(smearing)
        .max_history(MIX_HISTORY_LENGTH)
        .mix_gmax(MIX_CHARGE_GMAX_INV_A * INV_ANG_TO_INV_BOHR)
        .build();

    // -------------------------------------------------------------------
    // Divergence gate: cold-start tolerant bounds.
    // -------------------------------------------------------------------
    // CASTEP tol is per atom: window on TOTAL energy = tol_per_atom × n_ions.
    let tol_ha = std::env::var("CHEMRUST_COLD_TOL_HA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(CONV_TOL_EV_PER_ATOM * cell.num_ions as f64 * chemrust_scf::EV_TO_HARTREE);
    let max_iter: u64 = std::env::var("CHEMRUST_COLD_MAX_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    eprintln!(
        "[cold] gate: ecut={ecut_ha:.4} Ha (max_last_band={:.4}), tol_ha={tol_ha:.3e} (window, total), max_iter={max_iter}",
        ecut_ha + 5.0
    );

    let gate = ScfDivergenceGate {
        max_last_band_ha: ecut_ha + 5.0,
        min_band0_ha: -30.0,
        max_veff_range_factor: 10.0,
        max_iter,
        electron_count_tolerance: 0.05,
        soft_fraction_tolerance: 0.20,
        #[cfg(feature = "scf_diag")]
        check_raw_sections: None,
    };

    // -------------------------------------------------------------------
    // Run the pure-Rust SCF loop to convergence.
    // -------------------------------------------------------------------
    let t0 = std::time::Instant::now();
    eprintln!("[cold] running pure-Rust cold-start SCF ...");
    let result = run_scf_with_energy_gated(state, 8, tol_ha, Some(gate))
        .expect("cold-start SCF converged (or gate did not trip)");
    let wall_s = t0.elapsed().as_secs_f64();

    // -------------------------------------------------------------------
    // Compare converged eigenvalues against the reference .bands.
    // -------------------------------------------------------------------
    let eigs = &result.eigenvalues[0][0]; // NonSpin → channel 0, kpt 0
    assert_eq!(
        eigs.len(),
        bands_ref.len(),
        "eigenvalue count mismatch: {} vs reference {}",
        eigs.len(),
        bands_ref.len()
    );

    let errors: Vec<(usize, f64, f64, f64)> = eigs
        .iter()
        .zip(bands_ref.iter())
        .enumerate()
        .map(|(i, (&e, &r))| (i, e, r, (e - r).abs()))
        .collect();
    let mut worst: Vec<(usize, f64, f64, f64)> = errors.clone();
    worst.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    let max_err = worst[0].3;
    let max_band = worst[0].0;

    eprintln!("[cold] === Eigenvalue comparison vs Cu111_CO_H_dump/.bands ===");
    eprintln!("[cold]   Band   This run (Ha)   Reference (Ha)   |Δ| (Ha)");
    for (band, e, r, err) in worst.iter().take(10) {
        let flag = if *err > EIGVAL_WARN_HA { " *" } else { "" };
        eprintln!(
            "[cold]   {band:4}  {e:+.8e}  {r:+.8e}  {err:.4e}{flag}"
        );
    }
    eprintln!(
        "[cold]   --- max |Δ| = {:.4e} Ha at band {max_band} (worst-10 shown above)",
        max_err
    );
    eprintln!(
        "[cold]   band0:  this={:.8} Ha  ref={:.8} Ha",
        eigs[0], bands_ref[0]
    );

    // Diagnostics: total energy (loop convention; the offset vs CASTEP's
    // E-TS is not asserted — it depends on the ion-constant convention).
    eprintln!(
        "[cold] total energy (loop) = {:.8} eV; CASTEP reference E-TS = -24111.21907515 eV",
        result.total_energy * chemrust_scf::HARTREE_TO_EV
    );
    eprintln!("[cold] SCF wall time = {wall_s:.1} s");

    assert!(
        max_err < EIGVAL_TOL_HA,
        "COLD START FAIL: max|ε_i − ε_i^ref| = {max_err:.4e} Ha at band {max_band} \
         (this={:.8e}, ref={:.8e}) exceeds {EIGVAL_TOL_HA:.0e} Ha. \
         The converged eigenvalues do not match the CASTEP H_dump reference.",
        eigs[max_band], bands_ref[max_band]
    );

    if max_err < EIGVAL_WARN_HA {
        eprintln!("[cold] PASS: max |Δ| = {max_err:.4e} Ha < {EIGVAL_WARN_HA:.0e} Ha (clean agreement)");
    } else {
        eprintln!("[cold] OK (above warn threshold): max |Δ| = {max_err:.4e} Ha < {EIGVAL_TOL_HA:.0e} Ha");
    }
}

/// CPU-only smoke test: the LCAO cold-start initial-state chain (parse,
/// pseudo-atom, LCAO density, LCAO 'R' wavefunctions) runs and yields the
/// fixture electron count. No GPU and no SCF — it validates the
/// `chemrust_scf::build_lcao_initial_state` integration and its `.check`
/// consistency asserts.
#[test]
#[ignore = "requires the CASTEP fixture at /export/public_castep_jobs/tony/Cu111_CO_H_dump and /export/Potentials"]
fn lcao_init_smoke() {
    let fixture_dir =
        std::env::var("CASTEP_FIXTURE_DIR").unwrap_or_else(|_| H_DUMP_DIR.to_string());
    let potential_dir =
        std::env::var("CASTEP_POTENTIAL_DIR").unwrap_or_else(|_| POTENTIAL_DIR.to_string());

    let check: CastepBin = CheckFile::read(
        std::io::BufReader::new(
            std::fs::File::open(format!("{fixture_dir}/Cu111_CO.check")).expect("open .check"),
        ),
    )
    .expect("read .check");
    let wfc = check.wavefunction.as_ref().expect(".check wavefunction");
    let kpt = &wfc.kpt_data[0];
    let n_bands = kpt.bands.len();
    let n_pw = kpt.nplw;
    let [ngx, ngy, ngz] = wfc.grid;
    let cell = &check.cell;
    let fine_grid = GVectorGrid::new(
        check.fine_grid.expect("fine_grid")[0],
        check.fine_grid.expect("fine_grid")[1],
        check.fine_grid.expect("fine_grid")[2],
        cell.recip_lattice,
    );
    let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);
    let fine_dims = [
        check.fine_grid.expect("fine_grid")[0] as usize,
        check.fine_grid.expect("fine_grid")[1] as usize,
        check.fine_grid.expect("fine_grid")[2] as usize,
    ];

    let state = chemrust_scf::build_lcao_initial_state(
        &fixture_dir,
        &potential_dir,
        n_bands,
        n_pw,
        &kpt.pw_grid_coord,
        &kpt.coords,
        fine_dims,
        &wave_grid,
        &fine_grid,
        &check.cell.ionic_positions,
        0xC0FFEE as i64,
    )
    .expect("LCAO init");

    eprintln!(
        "[lcao-smoke] charge pre-VCA={:.6} post-VCA={:.6} (expect 186.0); psi0 len={}",
        state.elec_total_pre_vca,
        state.elec_total_post_vca,
        state.wave.len()
    );
    assert_eq!(state.wave.len(), n_bands * n_pw, "psi0 length");
    // The LCAO charge is the 186 valence electrons: the post-VCA total is
    // `nelectrons` by construction, and the pre-VCA G = 0 coefficient of the
    // LCAO inverse FFT already sums to 186 (allow a small FFT slack).
    assert!(
        (state.elec_total_post_vca - 186.0).abs() < 1e-6,
        "LCAO post-VCA charge {} != 186.0",
        state.elec_total_post_vca
    );
    assert!(
        (state.elec_total_pre_vca - 186.0).abs() < 0.5,
        "LCAO pre-VCA G=0 coefficient {} != 186.0",
        state.elec_total_pre_vca
    );
    // The wavefunctions are S-orthonormalised: no band is the zero vector.
    let max_abs: f64 = state.wave.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
    assert!(max_abs > 1e-3, "LCAO psi0 has a zero band (max|c| = {max_abs})");

    // The downsampled wave-grid density is a finite, positive-peak charge
    // distribution (not all-zero, not NaN). Its G = 0 coefficient is the
    // electron count, so sum / n_points ~= 186.
    let arr = state.density.as_wave_array();
    let n_pts: usize = arr.iter().count();
    let (sum, max, min): (f64, f64, f64) = arr.iter().fold(
        (0.0, f64::MIN, f64::MAX),
        |(s, mx, mn), &v| (s + v, mx.max(v), mn.min(v)),
    );
    assert!(arr.iter().all(|v| v.is_finite()), "LCAO wave-grid density has a non-finite value");
    assert!(max > 0.0, "LCAO wave-grid density is all <= 0 (max = {max})");
    eprintln!(
        "[lcao-smoke] wave-grid density: n_points={n_pts} sum/n={:.4} min={min:.4e} max={max:.4e}",
        sum / n_pts as f64
    );
    assert!(
        (sum / n_pts as f64 - 186.0).abs() < 1.0,
        "LCAO wave-grid density sum/n_points = {sum:.6}/{n_pts} != 186"
    );
}
