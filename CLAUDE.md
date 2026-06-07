## Resource location

- GPU porting CASTEP source code (full of bugs in the full GPU-resident stage):
  `~/programming/CASTEP-GPU-port/`
- Original CASTEP source code: `~/Downloads/CASTEP-6.11-nixos/`
- Current fixtures for CASTEP run with CPU-only ver: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/`
- Fixtures for CASTEP run with GPU FFT-only ver: `/tmp/cu111_gpu_resident_scf/`
- Abinit algorithm arxiv paper: `~/programming/CASTEP-GPU-port/2604.11139v1` (tarball)
- Pseudopotential files that CASTEP use: `/export/Potentials/`
- `chemrust-hamiltonian`: `~/programming/chemrust-hamiltonian/`
- `slurm` submission script for reference:
  - CPU version: `/export/public_castep_jobs/tony/Cu111_CO_SinglePoint/slurm_job_Cu111_CO.sh`
  - GPU version: `/tmp/cu111_gpu_resident_scf/slurm_job_Cu111_CO.sh`

## Useful commands

- How to run CASTEP GPU ver: `cd` to a directory with `.cell`, `.param` and
  pseudopotential files stated in `%BLOCK SPECIES_POT` section of `.cell`, then
  run `sbatch slurm_job_*.sh`
- When testing this crate, always use `--release` for speed, as we're
  computation heavy

## Hot tips

- A properly run CASTEP job won't produce anything to stderr/stdout. Outputs are
  in `.castep` file.
