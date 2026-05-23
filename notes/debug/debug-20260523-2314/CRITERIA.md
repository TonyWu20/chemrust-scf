# Anchor Criteria: §13 SCF Cascade D-screening Debug

## Fixture Files

| File | Location | Content |
|------|----------|---------|
| `Cu111_CO.bands` | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0523_with_D/` | Converged eigenvalues (160 bands, 1 k-point, 1 spin) |
| `Cu111_CO.check` | same dir | Converged wavefunctions (S-orthonormal, USPP convention) |
| `Cu111_CO.pot_fmt` | same dir | Converged V_eff on fine grid (formatted) |
| `Cu111_CO.den_fmt` | same dir | Converged density on fine grid (formatted) |
| `D_band_debug.dat` | same dir | D_screened per-ion per-SCF-iter (CASTEP `nl_d` dump, ES24.16) |
| `Cu111_CO.castep` | same dir | CASTEP text output (energies, convergence) |
| `Cu111_CO.param` | same dir | SCF parameters (mixing, smearing, etc.) |
| `Cu111_CO.cell` | same dir | Lattice, ion positions, species |

## Species Mapping (from `.cell` %BLOCK SPECIES_POT order)

- Species 1 = C (8 β projectors)
- Species 2 = O (8 β projectors)
- Species 3 = Cu (18 β projectors)
- Ions: 1×C, 1×O, 16×Cu = 18 total

## External Anchor Values

### A1 — Eigenvalue spectrum (Source: `Cu111_CO.bands:12-171`)

```
band-0  = -1.05502310 Ha   (Fermi energy = -0.122443 Ha)
band-1  = -0.49721165 Ha
band-2  = -0.48877159 Ha
...
band-159 = last eigenvalue
```

Success criterion: any test computing ⟨ψ_b|H|ψ_b⟩ must produce band-0 within
**0.05 Ha** of −1.05502310 Ha. RMS over 160 bands ≤ 0.05 Ha, per-band max ≤ 0.10 Ha.

### A2 — Converged D_screened matrices (Source: `D_band_debug.dat`, last 18 blocks)

Values are D_screened = D_0 + ∫Q·V_eff_converged, multiplied by mixture_weight (=1.0 for Cu111+CO, no VCA). Format: upper-triangular `(dn, dm, value)`, symmetric.

Per-ion max |D_screened[i,j]|:

| Species | Ion | Proj | max\|D\| (Ha) |
|---------|-----|------|----------------|
| C (sp=1) | 1 | 8 | 3.678691 |
| O (sp=2) | 1 | 8 | 6.038601 |
| Cu (sp=3) | 1 | 18 | 5.717674 |
| Cu (sp=3) | 2 | 18 | 4.661067 |
| Cu (sp=3) | 3 | 18 | 4.661067 |
| Cu (sp=3) | 4 | 18 | 4.791034 |
| Cu (sp=3) | 5 | 18 | 4.032666 |
| Cu (sp=3) | 6 | 18 | 4.464826 |
| Cu (sp=3) | 7 | 18 | 4.464826 |
| Cu (sp=3) | 8 | 18 | 4.589186 |
| Cu (sp=3) | 9 | 18 | 4.040514 |
| Cu (sp=3) | 10 | 18 | 4.657051 |
| Cu (sp=3) | 11 | 18 | 4.657051 |
| Cu (sp=3) | 12 | 18 | 4.789412 |
| Cu (sp=3) | 13 | 18 | 3.948576 |
| Cu (sp=3) | 14 | 18 | 5.822567 |
| Cu (sp=3) | 15 | 18 | 5.822567 |
| Cu (sp=3) | 16 | 18 | 5.823973 |

Success criterion (T2): per-ion `max|D_ours[n,m] − D_castep[n,m]| ≤ 5e-4 Ha`.

### A3 — CASTEP SCF parameters (Source: `Cu111_CO.param`)

```
MIXING_SCHEME: Pulay
MIX_CHARGE_AMP: 0.5
MIX_CHARGE_GMAX: 1.5 Bohr⁻¹
MIX_HISTORY_LENGTH: 20
SMEARING_WIDTH: 0.1 Ha (Gaussian erfc)
SPIN_POLARIZED: false
CUT_OFF_ENERGY: 400 eV
FINE_GRID_SCALE: 1.5
ELEC_ENERGY_TOL: 1e-5 eV
```

### A4 — CASTEP converged total energy (Source: `Cu111_CO.castep`)

```
Final energy = -24110.96665397 eV = -886.0794 Ha
```

### A5 — Number of SCF iterations to convergence (Source: `D_band_debug.dat`)

32 SCF iterations (576 blocks / 18 ions per iter = 32).

## Non-Criteria

The following are explicitly NOT success criteria (all DERIVED):

- Iter-2 band-0 = −0.869 Ha — our pipeline output, not ground truth
- Density split percentages (29.7/70.3 at iter-1, 69.9/30.1 at iter-2) — our pipeline output
- "CASTEP F8 density split = 36.8/63.2" — CASTEP output but intermediate diagnostic, not the final physical result
