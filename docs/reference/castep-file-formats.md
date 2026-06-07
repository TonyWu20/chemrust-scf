# CASTEP File Formats

Reference for which CASTEP output files contain what data. Last updated 2026-06-08.

## `.check` — Binary checkpoint

Contains a full snapshot of the SCF state for warm-start continuation:

- **Wavefunctions** (`eigenvectors%pw_coeff`): complex plane-wave coefficients for all bands
  and k-points. This is the authoritative source of wavefunction data — both the direct
  test (`davidson_hdump_validation`) and the FFI warm-start read wavefunctions from this file.
- **Eigenvalues**: band eigenvalues at the time of checkpoint
- **Density**: electron density on the FFT grid
- **SCF history**: previous SCF step data for convergence acceleration

Format: Fortran unformatted sequential (record-based binary), read by CASTEP's
`electronic_read_checkpoint` / `checkpoint_read_*` routines.

## `.bands` — Band eigenvalues (text)

Text file containing ONLY the converged band eigenvalues, one value per line for each
k-point and spin channel. Does NOT store wavefunctions. Used by the direct test for
eigenvalue reference comparison only (the C1 check).

## `.pot_fmt` — Local potential (formatted text)

Formatted text file containing the local potential `V_loc` on the fine FFT grid.
Used by the direct test to reconstruct `V_eff = V_loc + V_H + V_xc` for
computing H·psi. One value per line (or per row), Fortran real format.

## `.castep` — Main output (text)

Human-readable SCF convergence log containing:
- Total energy per SCF iteration
- Band eigenvalues
- Convergence metrics
- Timing information

Not used for data input — diagnostic only.

## `.cell` — Crystal structure (text)

Unit cell vectors, atomic positions, k-point grid, pseudopotential species.

## `.param` — Calculation parameters (text)

SCF convergence thresholds, eigensolver settings (`max_iterations`), smearing, etc.
