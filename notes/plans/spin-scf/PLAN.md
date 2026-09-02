# Spin-polarised SCF test plan (NiO, no U, finer grid)

## Context and goal chain

- Workspace purpose: debug `chemrust-scf` against CASTEP via the FFI path.
- CASTEP is the gold reference. The FFI hooks are the microscope.
- `chemrust-scf` exists because `chemrust-hamiltonian` cannot reproduce the
  DFT+U Hamiltonian from a dumped wavefunction without an SCF utility.
- The U term is state-dependent. A non-self-consistent dump gives an
  incomparable Hamiltonian. The SCF utility produces the consistent state.
- The spin-polarised case of the Rust SCF loop is untested. All loop work so
  far (mixer fixes, Pulay loop, control arms) is non-spin.
- This plan covers the spin-polarised loop test. It must finish before the
  DFT+U Hamiltonian reproduction work.

## Reference job

Path: `/export/public_castep_jobs/tony/NiO_no_u_finer_grid_spin/`
Regenerated: 2026-09-02 17:29 (the authoritative run for this plan).

| Quantity | Value |
| --- | --- |
| SCF cycles | 77 |
| E (.check total_energy) | -263.135104400 Ha = -7160.270369717 eV |
| E-TS (log) | -7160.270529747 eV |
| Fermi energy | 0.152973595 Ha (both channels) |
| Electrons | up 36 / down 28, net spin 8 |
| k-points | 14 |
| Wave grid | [20, 20, 20] |
| Fine grid | [40, 40, 40] (fine_grid_scale 3.0) |
| Cutoff | 380 eV |
| Mixing | Pulay, charge amp 0.5, spin amp 2.0, both gmax 1.5/Å, history 20 |
| Smearing | 0.1 eV, metals_method dm |
| spin_fix | 6 (fixed-spin cycles 1-6, then spin freed) |

Note: an older spin reference gave -7160.23058 eV. The current run is
40 meV lower (a different magnetic local minimum). The Sep 2 17:29 files are
authoritative. Any reference re-run must re-verify constants via
`nio_spin_reference_values`.

## Phase 1 — reference and baseline: DONE

- [x] Updated stale constants in `tests/nio_spin_scf.rs`
      (energy, Fermi, band-0 spin channels).
- [x] Added `nio_spin_reference_values` pin test (prints `.check` values;
      no GPU needed).
- [x] Baseline PASSES: `nio_warm_start_discriminator` on today's `.check`.
      Iter-1 eigenvalues: max 2.74e-4 Ha (gate 3e-4).
      Fermi delta 7.6e-5 Ha (gate 1e-3). V_eff_up != V_eff_dn.
      Energy drift at iter 1: 1.49e-3 Ha (gate 2e-2, warm-start).

## Phase 2 — CASTEP-faithful spin mixer: PENDING

Spec, read from CASTEP source (`castep/Source/Functional/dm/`):

- Mix object per kpt: complex PW pair (charge, spin) on the fine G-lattice,
  band-limited at the wave cutoff (380 eV), same mask as non-spin.
  - charge = FFT(rho_up + rho_down)
  - spin   = FFT(rho_up - rho_down)
- Two Kerker kernels (`dm_sub_base.f90:689-690`):
  - Kc(G) = 0.5 * G^2/(G^2+gc^2); G=0 → 0 (charge conservation).
  - Ks(G) = 2.0 * G^2/(G^2+gs^2); G=0 → 2.0 (uniform spin mixes fully).
  - gc = gs = 1.5/Å = 2.8346 a0^-1 (gmax² = 8.035 a0^-2).
- Residual R = (Rc, Rs). DIIS deltas on (c, s). Update
  (`dm_sub_mix.f90:815-1013`):
  - n_new = n_in + 1.0 * (sum c_i * delta_n)   [charge and spin parts x1.0]
  - plus Kerker part: Kc*(Rc + sum c_i*dRc) and Ks*(Rs + sum c_i*dRs).
  - Fallback on DIIS solve failure: plain Kerker step n_in + K*R.
- DIIS inner product (`dm_sub_base.f90:1112-1180`):
  - charge part weighted by mix_metric (1 + E_q1sq/E, q1 = 0).
  - spin part unweighted.
- Recombine to channels (`dm_sub_base.f90:1272-1358`):
  - rho_up = (c + s)/2, rho_dn = (c - s)/2.
  - High-G content above the cutoff carries over from the fresh
    wavefunction density (same mask rule as non-spin).
- `dm_flush_history()` clears the DIIS/Kerker history
  (`dm_sub_mix.f90` history ring + indices).

Work items:

- [ ] `src/mixing/kerker.rs`: add spin kernel slice. Store two kernels per
      G (charge, spin). Spin G=0 value = amp_s.
- [ ] `src/mixing/cuda_kernels.rs`: generalise `cpx_full_update` to two
      kernel amplitudes (charge part, spin part). Keep the non-spin call
      path working (spin kernel unused when nspins=1).
- [ ] `src/mixing.rs`: `DensityHistory` spin mode stores (c, s) objects.
      Two-kernel DIIS update. Charge-weighted + spin-unweighted inner
      product. New `flush()` method. `into_pulay` must preserve the spin
      history (same rule as the non-spin fix).
- [ ] Non-spin regression: all non-spin tests stay green after the change.

## Phase 3 — loop plumbing: PENDING

Spec from `electronic.f90`:

- Cycles 1..spin_fix (6): fixed occupations 36/28.
  (`electronic.f90:340` — spin_freed set only when scf_cycle >= spin_fix.)
- At scf_cycle == spin_fix (6):
  - spin_freed = true
  - dm_flush_history()
  - dm_mix_density(dens, dens) — store the fresh density as next input
  - re-occupancy: unrestricted filling at a common Fermi level
    (`electronic.f90:986-992`, occ_eigenvalues_free path).
- If the system converges while spin is still fixed: free the spin,
  reset the converged flag, continue (`electronic.f90:982-997`).
- Convergence counts only after spin_freed (energy window of 3 cycles).

Work items:

- [ ] `src/scf.rs`: fixed-occupation phase for the first 6 cycles
      (36/28 split, existing spin_fix field in SmearingParams).
- [ ] Release at scf_iter == spin_fix - 1 (0-based): flush the mixer
      history, re-mix, re-derive occupations at a common Fermi level.
- [ ] Convergence gate: `spin_freed && energy_converged`
      (extend the post-non-spin-fix check, scf.rs:2354 area).
- [ ] Verify the occupation code handles the fixed 36/28 fill per channel
      plus the post-release common-Fermi fill with 0.1 eV smearing
      (both channels metallic, ~17% spilling).

## Phase 4 — full-loop test: PENDING

- [ ] New test `nio_spin_pulay_full_loop` in `tests/nio_spin_scf.rs`
      (or `tests/` file of its own): SpinCollinear, 14 kpts, Pulay scheme,
      charge amp 0.5 / spin amp 2.0, spin gmax and charge gmax 2.8346 a0^-1,
      history 20, spin_fix 6, smearing 0.1 eV, max_iter ~120.
- [ ] Target: E_total ≈ -7693.40640 eV
      (= E-TS -7160.27053 minus the 533.13587 ion non-Coulomb constant,
      loop convention).
- [ ] Invariants: electron counts 36/28 stable after release. V_eff range
      bounded. Net spin 8 at convergence.
- [ ] Run locally on the GPU first. Then one SLURM confirmation run
      (root flake, `sbatch -w nixos-pro5000`, one GPU job at a time,
      fresh subfolder, append-only .castep).

## Phase 5 — FFI cross-check and follow-ups: PENDING

- [ ] Spin DENS/MIX component swap via `component_ffi` (nspins=2 hooks):
      verify the per-channel conventions (Fortran passes spin-summed
      real_charge + real_spin; Rust converts to channels with (c±s)/2).
- [ ] Control arms if the loop misbehaves: Kerker-pinned spin, no-mix
      spin. Assert expected gate trips, as in the non-spin A/B suite.
- [ ] After the spin loop passes: start the DFT+U Hamiltonian
      reproduction (the original `chemrust-hamiltonian` goal) using the
      verified spin SCF state. `hubbard.rs` / `hubbard_types.rs` already
      exist in `src/eigensolver/`.

## Risks and open questions

- Multiple magnetic local minima. The spin state reached depends on the
  mixing path. The CASTEP reference is one such minimum. If the Rust loop
  converges to a different minimum, compare spin-resolved band energies
  and total energies before declaring failure.
- The spin DIIS inner product mixes a weighted charge part and an
  unweighted spin part. A wrong weight here silently changes the
  trajectory. Verify against CASTEP `dm_mix_density_dot` line by line.
- `ca_scf_convergence` compile debt (scf_diag/chebyshev features) is
  unrelated; do not let it block this plan.

## Status

- Phase 1: DONE (2026-09-02).
- Phases 2-4: next up.
- Phase 5: after the loop passes.
