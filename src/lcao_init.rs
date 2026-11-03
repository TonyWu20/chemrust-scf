//! LCAO cold-start initial state for the pure-Rust GPU SCF pipeline.
//!
//! A CASTEP *cold start* does not read any electronic data (no `.check`
//! wavefunctions, no `.castep_bin` density). Instead it initialises from an
//! LCAO atomic superposition: a radial pseudo-atom charge per species (the
//! pure-Rust `atomic_solver` output) smeared over the fine grid, and a
//! seeded random wavefunction fill S-orthonormalised against the LCAO
//! overlap. This module wraps the verified `chemrust_hamiltonian_core::init`
//! chain into a single call that returns the initial density (wave grid) and
//! the initial wavefunctions in the form the [`crate::ScfIteration`]
//! builder consumes.
//!
//! The LCAO density screens the bare ionic potential with V_H / V_xc from
//! iteration 1. A zero initial density leaves the first V_eff bare-ionic and
//! the TPA preconditioner amplifies the deep V_loc wells (band-0 runs to
//! O(-10^3) Ha); the LCAO init is what keeps the first operator physical.
//!
//! No CASTEP electronic data: the only inputs are the `.cell`, the `.param`,
//! and the pseudopotential library.

use num_complex::Complex64;

use chemrust_hamiltonian_core::{
    downsample_density_to_wave_grid, GVectorGrid, RealGrid,
};

use crate::types::{Density, WaveGridArray};

/// The LCAO cold-start initial state: the wave-grid initial density and the
/// band-major initial wavefunctions, with the charge bookkeeping.
pub struct LcaoInitialState {
    /// The LCAO atomic-superposition charge on the wave grid (the
    /// `density_initialise_cell` fine-grid charge downsampled).
    pub density: Density,
    /// Band-major `n_bands * n_pw` initial wavefunctions (the LCAO 'R'
    /// random fill + LCAO S-orthonormalisation), in the `.check`
    /// plane-wave order.
    pub wave: Vec<Complex64>,
    /// The fine-grid total charge before the VCA background correction (the
    /// G = 0 coefficient of the un-normalised inverse FFT).
    pub elec_total_pre_vca: f64,
    /// The post-VCA-corrected total charge (equals the electron count).
    pub elec_total_post_vca: f64,
}

/// Build the CASTEP LCAO cold-start initial state from the fixture `.cell`
/// and `.param` (plus the pseudopotential library).
///
/// The `.check`-derived values (`n_bands`, `n_pw`, `pw_grid_coord`,
/// `k_coords`, `fine_dims`, `wave_grid`, `fine_grid`) are used to assert the
/// LCAO `basis_initialise` output matches the `.check` the SCF builder will
/// run on, so a grid or plane-wave drift fails fast rather than silently.
///
/// `seed` is the LCAO 'R' random-fill seed (`wave_initialise_wv`); a fixed
/// value keeps the run reproducible. The SCF fixed point is seed-independent.
pub fn build_lcao_initial_state(
    fixture_dir: &str,
    potential_dir: &str,
    n_bands: usize,
    n_pw: usize,
    pw_grid_coord: &Vec<[i32; 3]>,
    k_coords: &[f64; 3],
    fine_dims: [usize; 3],
    wave_grid: &GVectorGrid,
    fine_grid: &GVectorGrid,
    check_ionic_positions: &ndarray::Array2<f64>,
    seed: i64,
) -> Result<LcaoInitialState, String> {
    use chemrust_hamiltonian_core::init::basis::basis_initialise;
    use chemrust_hamiltonian_core::init::cell::cell_read;
    use chemrust_hamiltonian_core::init::density::density_initialise_cell;
    use chemrust_hamiltonian_core::init::ion::{cell_ion_counts, ion_read};
    use chemrust_hamiltonian_core::init::kpoints::kpoints_read;
    use chemrust_hamiltonian_core::init::model::model_initialise;
    use chemrust_hamiltonian_core::init::params::parameters_read;
    use chemrust_hamiltonian_core::init::wave::wave_initialise_wv;

    let cell_text = std::fs::read_to_string(format!("{fixture_dir}/Cu111_CO.cell"))
        .map_err(|e| format!("read {fixture_dir}/Cu111_CO.cell: {e}"))?;
    let param_text = std::fs::read_to_string(format!("{fixture_dir}/Cu111_CO.param"))
        .map_err(|e| format!("read {fixture_dir}/Cu111_CO.param: {e}"))?;

    // Parse with the local `castep-cell-fmt` (the path version the core
    // uses). The document types are inferred from the core `init` calls, so
    // the registry `castep-cell-io 0.6` name is never referenced here.
    let cdoc = castep_cell_fmt::parse(&cell_text).map_err(|e| format!("parse .cell: {e}"))?;
    let cell = cell_read(&cdoc).map_err(|e| format!("cell_read: {e}"))?;
    let pdoc = castep_cell_fmt::parse(&param_text).map_err(|e| format!("parse .param: {e}"))?;

    // The `.cell` SPECIES_POT block carries absolute paths from the machine
    // that wrote it (e.g. /Users/tony/Downloads/Potentials/C_00PBE.usp).
    // Resolve against `potential_dir` by basename so the same library works
    // on every host.
    let usp_paths: Vec<std::path::PathBuf> = cell
        .species_pot_files
        .iter()
        .map(|f| {
            let fname = std::path::Path::new(f)
                .file_name()
                .map(|b| b.to_os_string())
                .unwrap_or_else(|| std::ffi::OsString::from(f.clone()));
            std::path::PathBuf::from(potential_dir).join(fname)
        })
        .collect();
    let (n_atoms, ion_charge) =
        cell_ion_counts(&cell, &usp_paths).map_err(|e| format!("cell_ion_counts: {e}"))?;
    let prm = parameters_read(&pdoc, &param_text, ion_charge, n_atoms as f64);
    let mixture: Vec<Option<(usize, f64)>> = vec![None; cell.num_ions];
    let ions = ion_read(&cell, &usp_paths, &mixture, &prm).map_err(|e| format!("ion_read: {e}"))?;
    let kpt = kpoints_read(&cdoc, &cell).map_err(|e| format!("kpoints_read: {e}"))?;
    let basis =
        basis_initialise(&cell, &kpt.kpoints, &kpt.weights, &prm).map_err(|e| format!("basis_initialise: {e}"))?;
    let nbands = vec![n_bands; kpt.kpoints.len()];
    let model = model_initialise(
        &prm,
        &cell,
        &ions,
        &basis,
        &kpt.kpoints,
        &kpt.weights,
        &nbands,
    )
    .map_err(|e| format!("model_initialise: {e}"))?;

    // Consistency against the `.check` the SCF builder will use.
    if [basis.ngx_fine, basis.ngy_fine, basis.ngz_fine] != fine_dims {
        return Err(format!(
            "LCAO fine grid [{}x{}x{}] != .check fine grid {:?}: the .param \
             grid scale drifted",
            basis.ngx_fine, basis.ngy_fine, basis.ngz_fine, fine_dims
        ));
    }
    if basis.num_plane_waves_kp[0] != n_pw {
        return Err(format!(
            "LCAO plane-wave count {} != .check nplw {n_pw}: the .param \
             ELEC_CUTOFF drifted",
            basis.num_plane_waves_kp[0]
        ));
    }
    // The LCAO `basis_initialise` and the `.check` enumerate the same G set
    // (same ELEC_CUTOFF, same k-point) but possibly in different orders.
    // Verify the G set matches, then map each `.check` plane wave to its LCAO
    // index so the LCAO wavefunction coefficients are emitted in `.check`
    // plane-wave order.
    let lcao_pw = &basis.pw_grid_coord[0];
    if lcao_pw.len() != n_pw {
        return Err(format!(
            "LCAO plane-wave coord length {} != nplw {n_pw}",
            lcao_pw.len()
        ));
    }
    let mut g_to_lcao: std::collections::HashMap<[i32; 3], usize> =
        lcao_pw.iter().enumerate().map(|(i, g)| (*g, i)).collect();
    if g_to_lcao.len() != n_pw {
        return Err(format!(
            "LCAO plane-wave grid has duplicate G indices ({} entries, {} unique)",
            lcao_pw.len(),
            g_to_lcao.len()
        ));
    }
    for g in pw_grid_coord.iter() {
        if !g_to_lcao.contains_key(g) {
            return Err(format!(
                "LCAO plane-wave grid is missing .check G index {g:?}: the G set drifted"
            ));
        }
    }
    for ax in 0..3 {
        if (kpt.kpoints[0][ax] - k_coords[ax]).abs() > 1e-6 {
            return Err(format!(
                "LCAO k-point axis {ax} {} != .check {}: the .cell k-point drifted",
                kpt.kpoints[0][ax], k_coords[ax]
            ));
        }
    }
    // The LCAO `.cell` ionic positions must describe the same structure as
    // the `.check` cell, or the LCAO density would smear charge onto the
    // wrong sites and V_H would be wrong. The two files may list ions in
    // different orders, so compare the position set (rounded, sorted)
    // rather than element-wise.
    let lcao_pos = &cell.ionic_positions;
    if lcao_pos.shape() != check_ionic_positions.shape() {
        return Err(format!(
            "LCAO ionic position shape {:?} != .check {:?}: the .cell ion count drifted",
            lcao_pos.shape(),
            check_ionic_positions.shape()
        ));
    }
    fn rounded_sorted(pos: &ndarray::Array2<f64>) -> Vec<[f64; 3]> {
        let mut v: Vec<[f64; 3]> = (0..pos.nrows())
            .map(|i| {
                [
                    (pos[[i, 0]] * 1e6).round() / 1e6,
                    (pos[[i, 1]] * 1e6).round() / 1e6,
                    (pos[[i, 2]] * 1e6).round() / 1e6,
                ]
            })
            .collect();
        v.sort_by(|a, b| {
            a[0].total_cmp(&b[0])
                .then_with(|| a[1].total_cmp(&b[1]))
                .then_with(|| a[2].total_cmp(&b[2]))
        });
        v
    }
    let lcao_sorted = rounded_sorted(lcao_pos);
    let check_sorted = rounded_sorted(check_ionic_positions);
    for (i, (l, c)) in lcao_sorted.iter().zip(check_sorted.iter()).enumerate() {
        if l != c {
            return Err(format!(
                "LCAO ionic position set differs from .check at slot {i}: LCAO {l:?} != .check {c:?}: the .cell structure drifted"
            ));
        }
    }

    // The LCAO initial density (fine grid -> wave grid) and wavefunctions.
    let den =
        density_initialise_cell(&model).map_err(|e| format!("density_initialise_cell: {e}"))?;
    let wave = wave_initialise_wv(&model, seed).map_err(|e| format!("wave_initialise_wv: {e}"))?;

    let fine_arr =
        ndarray::Array3::<f64>::from_shape_vec((basis.ngx_fine, basis.ngy_fine, basis.ngz_fine), den.real_charge.clone())
            .map_err(|e| format!("LCAO fine-grid density array: {e}"))?;
    let rho_fine = RealGrid::from_inner(fine_arr);
    let rho_wave = downsample_density_to_wave_grid(&rho_fine, wave_grid, fine_grid)
        .map_err(|e| format!("downsample LCAO density to wave grid: {e}"))?;
    let density = Density::from_inner(WaveGridArray::from_inner(rho_wave.into_inner()));

    if wave.coeffs[0][0].len() != n_bands {
        return Err(format!(
            "LCAO band count {} != .check band count {n_bands}",
            wave.coeffs[0][0].len()
        ));
    }
    let mut psi0 = vec![Complex64::ZERO; n_bands * n_pw];
    for b in 0..n_bands {
        for p in 0..n_pw {
            // Emit in `.check` plane-wave order via the LCAO index map.
            let lcao_p = g_to_lcao[&pw_grid_coord[p]];
            psi0[b * n_pw + p] = wave.coeffs[0][0][b][lcao_p];
        }
    }

    // The post-VCA total charge is `nelectrons` by construction (the VCA
    // background correction drives the fine-grid G = 0 coefficient to the
    // electron count). The pre-VCA value is the raw G = 0 coefficient of the
    // LCAO inverse FFT.
    let elec_total_post_vca = prm.nelectrons;

    Ok(LcaoInitialState {
        elec_total_pre_vca: den.num_elec_total_pre_correction,
        density,
        wave: psi0,
        elec_total_post_vca,
    })
}
