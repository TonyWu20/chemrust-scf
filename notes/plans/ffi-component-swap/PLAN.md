# FFI component-swap debugging plan

Status: stage 0 in progress (AOCL build on the AMD machine).

## Goal

Find which Rust SCF-loop component diverges from CASTEP. The block Davidson
solver is proven correct through the FFI route. The remaining loop
components (initial density, V_eff, energies, occupations, mixing) are
unverified. Method: swap one component to the Rust side at a time, through
the FFI boundary, and compare the SCF trajectory against the CASTEP
reference. The first swap that breaks the trajectory names the culprit.

## Why not debug the pure-Rust loop directly

Two months of attempts (Opus, DeepSeek-v4-pro) failed on the pure-Rust
loop. The loop has interacting components. A component-swap over the
proven FFI boundary isolates each component against a working CASTEP
reference. Each step is small and individually verifiable.

## Stage 0: build (in progress)

This machine is AMD (Ryzen 9950X3D2). The old CASTEP build was tuned for
Intel (MKL). The build uses the `castep_611_aocl` derivation
(AOCL BLIS + AOCL FFTW).

- `castep/flake.nix`: new `aocl` devShell with commands:
  - `patch-aocl` — idempotent sed of `obj/Makefile` (/bin/cp) and
    `obj/platforms/linux_x86_64_gfortran.mk` (-lacml, -lfftw3) plus
    removal of the `obj/linux_x86_64_gfortran/exists` stamp. The
    top-level Makefile re-copies the patched platform .mk into the
    build dir when the stamp is missing.
  - `make-castep-aocl` — `make -j ARCH=linux_x86_64_gfortran MATHLIBS=acml
    FFT=fftw3 MATHLIBDIR=<aocl lib> FFTLIBDIR=<aocl lib> castep`
  - `make-test-aocl`, `make-castep-chemrust-aocl` (adds
    `CHEMRUST_SCF_DIR=<../chemrust-scf> USE_CHEMRUST=1`).
- Shell: `nix shell .#devShells.x86_64-linux.aocl --command sh -c '<cmd>'`
- chemrust cdylib: `cargo build --release` in `chemrust-scf/` (default
  features, sm_120 NVRTC target).

Gate: `make-castep-chemrust-aocl` produces `obj/linux_x86_64_gfortran/castep.mpi`
and a NiO CASTEP job with `USE_CHEMRUST` runs to completion.

## Stage 1: baselines (NiO, fits the free GPU memory)

| Run | Config | Output |
|---|---|---|
| L1a | CASTEP CPU Davidson (native, no FFI) | `davidson_cpu_serial_NiO_castep_scf_log.txt` (exists; converged -7160.27064 eV) |
| L1b | CASTEP + GPU Davidson FFI, default features (D1/D2 wall OFF) | per-iteration energy / Fermi / gain table |
| L1c | CASTEP + GPU Davidson FFI, scf_diag (wall ON + SyncAudit) | L1b + `[SyncAudit] maxdiff` lines |

Gate L1b: L1b trajectory matches L1a within a small band. This also
settles the Davidson wall question on real hardware: if L1b matches and
a no-wall run diverges, the SyncAudit (L1c) names the load-bearing read
site.

## Stage 2: component-swap ladder

One component per run. All other components stay CASTEP. Order follows
likelihood of being the culprit (evidence: the NiO harness run diverged
by iter 3-4 with a ~719 eV iter-1 offset and a different gain sequence).

| Step | Component swapped to Rust | FFI entry needed | Check |
|---|---|---|---|
| S2a | Initial density (fine->wave transfer, .bin charge) | `chemrust_initial_density` | per-iter energies track L1a from iter 1 |
| S2b | Energy bookkeeping (Ewald, core constants, E_H, E_xc) | none (pure comparison of logged terms) | iter-1 total energy within tolerance of L1a |
| S2c | V_eff construction (Hartree + XC + USPP augmentation) | `chemrust_veff_step` | V_eff arrays and per-iter energies track L1a |
| S2d | Occupations / Fermi (Gaussian smearing 0.1 eV) | `chemrust_occupations` | occupation columns track L1a |
| S2e | Density mixing (Kerker/Pulay: amp 0.5, G-cutoff 1.5, history 20, delay 1, kick-in 0.1 Ha/atom) | `chemrust_mix_density` | trajectory matches L1a to convergence |

For each step: run the NiO job, diff the per-iteration energy / Fermi /
gain against L1a. Record the result (match / break + first broken
iteration) in the progress table below. A broken step isolates that
component. Fix it against CASTEP source, re-run, then move on.

## Wall A/B design (run on the proven FFI path)

The cdylib built before this session printed the `[Diag-D*]` blocks
unconditionally (zdotc + D2H + print per call) — the accidental wall
documented in `docs/load-bearing-diagnostic-overhead.md`. Those blocks
are now gated behind `scf_diag` (D3, D4, D5, D6, D8, D9, D10-01,
D10-Stage6, C6, Rust-SearchRaw). The D1/D2 blocks were gated in the
prior session.

- **A (wall ON)**: the running L1b job. It loaded the pre-gating .so.
  Archived: `/tmp/libchemrust_scf_wall_on.so`.
- **B (wall OFF)**: the quiet default-features cdylib (rebuilt after
  the gating; the `davidson_diag!` progress lines are gated too). Same
  NiO job dir, log `ffi_run_B_wall_off.log`. Compare energy trajectories
  A vs B. If B diverges, run C (scf_diag build with SyncAudit) to name
  the load-bearing host read, then add the targeted sync at that site
  only.
- B is running: cycle 1 complete (NiO.wvfn.1), outer-iter checks show
  0/62 converged at the early cycles (normal HFO start), eigenvalues in
  the physical -0.6..+1.1 Ha range.
- The Fortran-side WIP diagnostics in the worktree (`F8 DIAGNOSTIC`
  block in nlpot.f90, `[Diag-Vnl-CASTEP]` blocks in electronic.f90)
  flood B's stdout. They exist in the A baseline too, so they do not
  contaminate the A/B comparison. Disable them for clean runs only with
  user approval.

## Progress

- [x] stage 0: AOCL worktree build + chemrust cdylib
  - `castep/flake.nix` `aocl` devShell: `patch-aocl`, `make-castep-aocl`,
    `make-test-aocl`, `make-castep-chemrust-aocl`.
  - Plain `make-castep-aocl` fails at the link (the FFI stub is always
    compiled; only the libchemrust_scf link is gated on USE_CHEMRUST).
    Use `make-castep-chemrust-aocl`.
  - L1b job dir: `/export/public_castep_jobs/tony/NiO_GPU_Davidson_pro5000`
    (NiO input files from the CPU reference dir, `NiO.param` with
    `devel_code: PROF:*:ENDPROF CHEMRUST`).
- [~] L1b: FFI default-features run, trajectory vs L1a (running; first
  attempt died at a session boundary — process group cleanup. Restarted
  with `setsid nohup`, exit code logged to `ffi_run.log`).
- [ ] L1c: scf_diag SyncAudit run (only if L1b diverges)
- [ ] S2a: initial density
- [ ] S2b: energy bookkeeping
- [ ] S2c: V_eff
- [ ] S2d: occupations
- [ ] S2e: mixing

## Notes

- NiO reference uses 14 k-points (`KPOINTS_LIST` in `NiO.cell`, weight
  1/14 each). The Rust `nio_no_spin` fixture uses k-point 0 only. The
  harness fixture must cover all 14 k-points before any trajectory
  comparison is meaningful. The FFI job reads `NiO.cell` and uses all
  14 k-points, so L1b vs L1a is comparable.
- CASTEP mixing kick-in (verified in source): `electronic.f90:7737`
  `mixing_convergence_tol = 0.1` Ha; mixing on when
  `|E_now - E_prev| <= 0.1 Ha * num_ions` and `scf_cycle > mixing_delay`
  (=1, `dm_sub_mix.f90:75`). First mix pass uses Kerker, then Pulay/DIIS.
- GPU is shared: sglang holds ~42.6 GB of 48.9 GB. NiO (40^3 fine grid)
  fits in the remaining memory; Cu111_CO does not.
