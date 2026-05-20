# Prior Investigation Note Classification

## Source: `notes/open-followups.md`

### Claim 1: "G-vector mapping in diagonalize produces wrong eigenvalues"
- **Symptom**: "Starting from CASTEP's converged V_eff and wavefunctions (loaded from .check), diagonalize() produces eigenvalues that are systematically wrong beyond the first 2 bands. RMS error ~12 Ha across 160 bands."
- **Symptom details**: Band 1: -1.60 Ha (ref -1.06 Ha), Band 3: -0.29 Ha (ref -0.49 Ha), Band 5+: positive and climbing (ref remains negative until band ~104)
- **Claimed root cause**: "G-vector index mapping (pw_fft_indices) between the sparse PW coefficient array and the full FFT grid does not match the convention that apply_local_hamiltonian uses"
- **Classification**: **DERIVED** — computed by our own pipeline (the symptom was observed from our code, and the root cause hypothesis was generated from our own analysis)
- **Why not EXTERNAL**: The eigenvalue values -1.60, -0.29, etc. are from our own pipeline output. The reference values -1.06, -0.49 exist in CASTEP .bands (EXTERNAL), but the symptom comparison is a DERIVED observation.
- **Post-fix status**: Commit `5037e64` addressed the scatter/gather index order and per-PW kinetic energies. Whether this resolved the eigenvalue issue is UNVERIFIED (tests are `#[ignore]`, require GPU).

### Claim 2: "Likely root cause — scatter/gather index order"
- **Classification**: **HYPOTHESIZED** — inferred from code analysis, not confirmed with external data
- **Verification quality**: No discriminator-value test existed before the fix. The fix in `5037e64` added `compare_eigenvalues_with_reference_veff_and_screening` but it has not been run (GPU required).

### Claim 3: "Total energy is ~13919 eV too high"
- **Classification**: **DERIVED** — computed from our pipeline's wrong eigenvalues
- **Post-fix status**: Depends on whether eigenvalue fix resolved the issue.

## Source: Prior session memory (auto-memory)

### Claim: "CASTEP defaults to Gaussian (erfc) smearing"
- **Classification**: **EXTERNAL** — confirmed by .castep file and .bands file header
- **Verification**: `.bands` header shows 186 electrons with Fermi energy -0.122443 Ha; our smearing uses Gaussian with 0.1 eV width, matching CASTEP default.

## Summary

| Claim | Class | Admissible? |
|-------|-------|-------------|
| G-vector mapping causes wrong eigenvalues | DERIVED | No |
| Scatter/gather index order is root cause | HYPOTHESIZED | No |
| Total energy 13919 eV too high | DERIVED | No |
| CASTEP uses Gaussian smearing | EXTERNAL | Yes |

Only the CASTEP reference eigenvalues from `.bands` and `.pot_fmt` are EXTERNAL and admissible as criteria.
