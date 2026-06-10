//! Total energy computation.
//!
//! Implements the two components needed for the full KS-DFT total energy:
//!
//! 1. **Ewald summation** (`ewald_energy`) — ion-ion electrostatic energy
//!    via the standard Ewald split into real-space, reciprocal-space, and
//!    self-energy terms.
//!
//! 2. **Total energy assembly** (`assemble_total_energy`) — combines band
//!    energy, Hartree energy, exchange-correlation energy, the
//!    double-counting correction ∫ρV_xc, and the Ewald energy:
//!
//! ```text
//! E_total = E_band - E_H + E_xc - ∫ρV_xc + E_ewald
//! ```

use chemrust_hamiltonian_core::{CellGeometry, PseudopotentialSet};

/// eV to Hartree conversion (1 eV = 1/27.211384 Hartree).
pub const EV_TO_HARTREE: f64 = 1.0 / 27.211384;
/// Hartree to eV conversion (1 Hartree = 27.211384 eV).
pub const HARTREE_TO_EV: f64 = 27.211384;

// ---------------------------------------------------------------------------
// Ewald summation
// ---------------------------------------------------------------------------

/// Compute the Ewald energy (ion-ion electrostatic) in Hartree.
///
/// Standard Ewald summation formula with uniform neutralising background:
///
/// ```text
/// E_ewald = 1/2 Σ_{R} Σ_{I,J} Z_I Z_J erfc(α|r_IJ - R|) / |r_IJ - R|
///         + 2π/V Σ_{G≠0} |S(G)|² exp(-G²/4α²) / G²
///         - α/√π Σ_I Z_I²
///         - π/(2α²V) × (Σ_I Z_I)²      ← background self-energy (G=0)
/// ```
///
/// Where:
/// - α = √π / V^(1/3)  (Ewald splitting parameter)
/// - S(G) = Σ_I Z_I exp(iG·r_I) is the structure factor
/// - Z_I = ionic charge from pseudopotential
/// - V = cell volume
/// - R = lattice translation vectors (real-space, nearest-neighbour shells)
/// - G = reciprocal lattice vectors (up to ~7α).
///
/// The last term (G=0 background correction) is required because the
/// reciprocal-space sum explicitly excludes G=0.  CASTEP applies the same
/// correction (ewald.f90:585-587); without it the periodic Ewald sum of a
/// charged unit cell would diverge.
pub fn ewald_energy(cell: &CellGeometry, pots: &PseudopotentialSet) -> f64 {
    let volume = cell.volume;
    let alpha = (std::f64::consts::PI / volume).powf(1.0 / 3.0);

    // Compute Z_I for each ion.
    let ionic_charges: Vec<f64> = cell
        .ion_species
        .iter()
        .map(|&idx| {
            let symbol = &cell.species_symbols[idx];
            pots.get(symbol)
                .and_then(|p| p.ionic_charge())
                .unwrap_or(0.0)
        })
        .collect();

    // Total charge Q = Σ Z_I (used for the G=0 background correction).
    let q_total: f64 = ionic_charges.iter().sum();
    #[cfg(feature = "scf_diag")]
    eprintln!("[Ewald] q_total={} alpha={:.6} volume={:.2} alpha2V_ovpi={:.6} bg_term_ha={:.4}",
        q_total, alpha, volume,
        0.5 * std::f64::consts::PI / (alpha * alpha * volume),
        -0.5 * std::f64::consts::PI * q_total * q_total / (alpha * alpha * volume));

    // Self-energy: -α/√π · Σ_I Z_I²
    let self_energy = {
        let sum_z2: f64 = ionic_charges.iter().map(|z| z * z).sum();
        -alpha / std::f64::consts::PI.sqrt() * sum_z2
    };

    // G=0 background correction: -π·Q²/(2·α²·V)
    // Removes the self-interaction of the uniform neutralising background
    // that is implicitly excluded when G=0 is skipped in the recip sum.
    let background_correction =
        -0.5_f64 * std::f64::consts::PI * q_total * q_total / (alpha * alpha * volume);

    // Real-space sum: compute cutoff to guarantee erfc(α×cutoff) < 5e-15.
    // For α = (π/Ω)^(1/3) ≈ 0.052 this gives cutoff ≈ 106 Bohr, which
    // requires about 11×7×7 = 539 image cells — still tractable.
    let erfc_precision = 5e-15_f64;
    // erfc(x) < ε for x > sqrt(-ln(ε·√π)) roughly, but a safe bound is:
    // erfc(x) ≈ exp(-x²) / (x·√π) for large x.  Solve exp(-x²) = ε·√π·x.
    // For ε = 5e-15, x ≈ 5.5 gives erfc(5.5) = 5.4e-15.
    let erf_inv = 5.5_f64;
    let real_cutoff = erf_inv / alpha;
    let real_energy = ewald_real_space(cell, &ionic_charges, alpha, real_cutoff);

    // Reciprocal-space sum (G-vectors up to |G|_max).
    // The Gaussian weight exp(-G²/4α²) must be < ε for convergence.
    // |G|_max = 2α · sqrt(-ln(ε))  →  for ε = 5e-15, sqrt(-ln(ε)) ≈ 5.74.
    let recip_g_max = 2.0 * alpha * (-erfc_precision.ln()).sqrt();
    let recip_energy = ewald_reciprocal_space(cell, &ionic_charges, alpha, recip_g_max);

    real_energy + recip_energy + self_energy + background_correction
}

/// Real-space Ewald sum over image-cell pairs.
///
/// Iterates over all ion pairs (I,J) and lattice translation vectors R within
/// `cutoff` Bohr.  The factor ½ is applied at the end.
///
/// `cutoff` must be large enough that `erfc(α·cutoff)` is negligible (typically
/// < 1e-14) for the Ewald sum to converge to machine precision.
fn ewald_real_space(
    cell: &CellGeometry,
    charges: &[f64],
    alpha: f64,
    cutoff: f64,
) -> f64 {
    let real_lattice = cell.real_lattice.as_array();
    let a1 = real_lattice[0];
    let a2 = real_lattice[1];
    let a3 = real_lattice[2];

    // Norms for bounding lattice-translation search.
    let a1_norm = (a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2]).sqrt();
    let a2_norm = (a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2]).sqrt();
    let a3_norm = (a3[0] * a3[0] + a3[1] * a3[1] + a3[2] * a3[2]).sqrt();

    let n1_max = (cutoff / a1_norm).ceil() as i32;
    let n2_max = (cutoff / a2_norm).ceil() as i32;
    let n3_max = (cutoff / a3_norm).ceil() as i32;

    // Convert fractional positions to Cartesian.
    let cart_pos: Vec<[f64; 3]> = (0..cell.num_ions)
        .map(|i| {
            let xf = cell.ionic_positions[[i, 0]];
            let yf = cell.ionic_positions[[i, 1]];
            let zf = cell.ionic_positions[[i, 2]];
            let x = xf * a1[0] + yf * a2[0] + zf * a3[0];
            let y = xf * a1[1] + yf * a2[1] + zf * a3[1];
            let z = xf * a1[2] + yf * a2[2] + zf * a3[2];
            [x, y, z]
        })
        .collect();

    let mut energy = 0.0_f64;

    for i in 0..cell.num_ions {
        for j in 0..cell.num_ions {
            let zi = charges[i];
            let zj = charges[j];
            let ri = cart_pos[i];
            let rj = cart_pos[j];

            for n1 in -n1_max..=n1_max {
                for n2 in -n2_max..=n2_max {
                    for n3 in -n3_max..=n3_max {
                        // Skip self-interaction.
                        if i == j && n1 == 0 && n2 == 0 && n3 == 0 {
                            continue;
                        }

                        // R = n1·a1 + n2·a2 + n3·a3
                        let rx = n1 as f64 * a1[0] + n2 as f64 * a2[0] + n3 as f64 * a3[0];
                        let ry = n1 as f64 * a1[1] + n2 as f64 * a2[1] + n3 as f64 * a3[1];
                        let rz = n1 as f64 * a1[2] + n2 as f64 * a2[2] + n3 as f64 * a3[2];

                        let dx = rj[0] + rx - ri[0];
                        let dy = rj[1] + ry - ri[1];
                        let dz = rj[2] + rz - ri[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                        if dist < cutoff {
                            energy += zi * zj * libm::erfc(alpha * dist) / dist;
                        }
                    }
                }
            }
        }
    }

    0.5 * energy
}

/// Reciprocal-space Ewald sum.
///
/// Iterates over integer G-vector triplets `(h,k,l)` up to `g_max`
/// and accumulates `|S(G)|² exp(-G²/4α²) / G²`.
fn ewald_reciprocal_space(
    cell: &CellGeometry,
    charges: &[f64],
    alpha: f64,
    g_max: f64,
) -> f64 {
    let volume = cell.volume;
    let recip = cell.recip_lattice.as_array();
    let b1 = recip[0];
    let b2 = recip[1];
    let b3 = recip[2];

    // Bounding box for (h,k,l) from the reciprocal-space norms.
    let b_norm = |b: &[f64; 3]| (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    let nb1 = (g_max / b_norm(&b1)).ceil() as i32;
    let nb2 = (g_max / b_norm(&b2)).ceil() as i32;
    let nb3 = (g_max / b_norm(&b3)).ceil() as i32;

    let two_pi = 2.0 * std::f64::consts::PI;

    let mut energy = 0.0_f64;

    for h in -nb1..=nb1 {
        for k in -nb2..=nb2 {
            for l in -nb3..=nb3 {
                if h == 0 && k == 0 && l == 0 {
                    continue; // G=0 term is handled by the real-space sum.
                }

                // Cartesian G-vector: G = h·b1 + k·b2 + l·b3
                let gx = h as f64 * b1[0] + k as f64 * b2[0] + l as f64 * b3[0];
                let gy = h as f64 * b1[1] + k as f64 * b2[1] + l as f64 * b3[1];
                let gz = h as f64 * b1[2] + k as f64 * b2[2] + l as f64 * b3[2];
                let g2 = gx * gx + gy * gy + gz * gz;

                if g2 < 1e-30 || g2 > g_max * g_max {
                    continue;
                }

                // Structure factor S(G) = Σ_I Z_I exp(iG·r_I)
                // G·r_I = 2π · (h·x_frac + k·y_frac + l·z_frac)
                let mut s_re = 0.0_f64;
                let mut s_im = 0.0_f64;
                for (ion, &z) in charges.iter().enumerate() {
                    let x = cell.ionic_positions[[ion, 0]];
                    let y = cell.ionic_positions[[ion, 1]];
                    let z_frac = cell.ionic_positions[[ion, 2]];
                    let phase = two_pi * (h as f64 * x + k as f64 * y + l as f64 * z_frac);
                    s_re += z * phase.cos();
                    s_im += z * phase.sin();
                }
                let s2 = s_re * s_re + s_im * s_im;

                let exp_factor = (-g2 / (4.0 * alpha * alpha)).exp();
                energy += two_pi / volume * s2 * exp_factor / g2;
            }
        }
    }

    energy
}

// ---------------------------------------------------------------------------
// Total energy assembly
// ---------------------------------------------------------------------------

/// Assemble the KS-DFT total energy from its components.
///
/// # Formula
///
/// ```text
/// E_total = E_band - E_H + E_xc - ∫ρV_xc + E_ewald
/// ```
///
/// Where:
/// - `E_band = Σ_i f_i ε_i` — band energy (sum of occupied eigenvalues)
/// - `E_H = ½ × Σ_r ρ(r) V_H(r) × dV` — Hartree energy (precomputed)
/// - `E_xc` — exchange-correlation energy (from `VEffWithEnergy.e_xc`)
/// - `∫ρV_xc = Σ_r ρ(r) V_xc(r) × dV` — double-counting correction (precomputed)
/// - `E_ewald` — ion-ion electrostatic (Ewald) energy
///
/// # Arguments
///
/// * `eigenvalues` — KS eigenvalues ε_i (sorted, one per band).
/// * `occupations` — occupation numbers f_i (same length as eigenvalues).
/// * `e_xc` — exchange-correlation energy from the XC evaluation.
/// * `e_hartree` — Hartree energy E_H (precomputed on the fine grid,
///   from the **valence** density to match the non-variational NLCC core).
/// * `rho_vxc` — the integral ∫ρ V_xc dr (precomputed on the fine grid,
///   also from valence density).
/// * `ewald` — Ewald ion–ion energy.
#[allow(clippy::too_many_arguments)]
pub fn assemble_total_energy(
    eigenvalues: &[f64],
    occupations: &[f64],
    e_xc: f64,
    e_hartree: f64,
    rho_vxc: f64,
    ewald: f64,
) -> f64 {
    let e_band: f64 = eigenvalues
        .iter()
        .zip(occupations.iter())
        .map(|(&eps, &f)| f * eps)
        .sum();

    e_band - e_hartree + e_xc - rho_vxc + ewald
}

/// Assemble the KS-DFT total energy from pre-computed band energy.
///
/// Same formula as `assemble_total_energy` but accepts the band energy
/// directly rather than computing it from eigenvalues and occupations.
/// Useful when the band energy involves a kpt-weighted sum.
///
/// ```text
/// E_total = e_band - E_H + E_xc - ∫ρV_xc + E_ewald
/// ```
pub fn assemble_total_energy_from_band(
    e_band: f64,
    e_xc: f64,
    e_hartree: f64,
    rho_vxc: f64,
    ewald: f64,
) -> f64 {
    e_band - e_hartree + e_xc - rho_vxc + ewald
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use chemrust_hamiltonian_core::{RealLattice, RecipLattice};

    /// Helper: build a minimal 1-atom CellGeometry.
    fn simple_cell(volume: f64) -> CellGeometry {
        let a = volume.powf(1.0 / 3.0);
        CellGeometry {
            real_lattice: RealLattice::from_inner([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
            recip_lattice: RecipLattice::from_inner([
                [2.0 * std::f64::consts::PI / a, 0.0, 0.0],
                [0.0, 2.0 * std::f64::consts::PI / a, 0.0],
                [0.0, 0.0, 2.0 * std::f64::consts::PI / a],
            ]),
            volume,
            num_species: 1,
            num_ions: 1,
            ionic_positions: Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap(),
            species_symbols: vec!["H".into()],
            species_pot_files: vec!["H_00.usp".into()],
            num_ions_in_species: vec![1],
            ion_species: vec![0],
            max_ions_in_species: 1,
            species_lcao_states: vec![],
        }
    }

    /// Ewald energy should be finite for a simple cubic cell with one ion.
    #[test]
    fn test_ewald_is_finite() {
        let cell = simple_cell(1000.0);
        let pots = PseudopotentialSet::new();
        // No pots → Z_I = 0 → Ewald = 0.
        let e = ewald_energy(&cell, &pots);
        assert!(e.is_finite(), "Ewald should be finite with no charges");
        assert!(
            (e).abs() < 1e-12,
            "Zero charges should give zero Ewald, got {e}"
        );
    }

    /// Total energy assembly: simple check that band energy dominates.
    #[test]
    fn test_assemble_total_energy_basic() {
        let eigs = vec![-0.5, -0.3, 0.0, 0.2];
        let occs = vec![1.0, 1.0, 0.0, 0.0];
        let e = assemble_total_energy(&eigs, &occs, -1.0, 0.5, 0.8, 0.2);
        // E_band = -0.5*1 + -0.3*1 = -0.8
        // e = -0.8 - 0.5 + (-1.0) - 0.8 + 0.2 = -2.9
        let expected = -2.9;
        assert!(
            (e - expected).abs() < 1e-12,
            "E_total expected {expected}, got {e}"
        );
    }
}
