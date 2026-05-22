# Graph Report - .  (2026-05-22)

## Corpus Check
- 93 files · ~101,834 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 591 nodes · 877 edges · 45 communities (29 shown, 16 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 55 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Test Fixtures & Cell Setup|Test Fixtures & Cell Setup]]
- [[_COMMUNITY_GPU Density & CASTEP Reference|GPU Density & CASTEP Reference]]
- [[_COMMUNITY_GPU Data Transfer & Density Core|GPU Data Transfer & Density Core]]
- [[_COMMUNITY_Chebyshev Filtering & S⁻¹ Woodbury|Chebyshev Filtering & S⁻¹ Woodbury]]
- [[_COMMUNITY_WavefunctionSet & Density Data Model|WavefunctionSet & Density Data Model]]
- [[_COMMUNITY_GPU-Resident Augmentation Kernel|GPU-Resident Augmentation Kernel]]
- [[_COMMUNITY_Bug Patterns & Fortran Interop|Bug Patterns & Fortran Interop]]
- [[_COMMUNITY_Augmentation Density Pipeline|Augmentation Density Pipeline]]
- [[_COMMUNITY_DIISKerker Mixing Pipeline|DIIS/Kerker Mixing Pipeline]]
- [[_COMMUNITY_Batched cuFFT Plans|Batched cuFFT Plans]]
- [[_COMMUNITY_Validation Test Suite|Validation Test Suite]]
- [[_COMMUNITY_Hamiltonian Application Pipeline|Hamiltonian Application Pipeline]]
- [[_COMMUNITY_cuBLAS Wrapper|cuBLAS Wrapper]]
- [[_COMMUNITY_Rayleigh-Ritz Eigensolver|Rayleigh-Ritz Eigensolver]]
- [[_COMMUNITY_Kerker Preconditioner|Kerker Preconditioner]]
- [[_COMMUNITY_cuSOLVER Error Handling|cuSOLVER Error Handling]]
- [[_COMMUNITY_SCF State Builder & Fixtures|SCF State Builder & Fixtures]]
- [[_COMMUNITY_Reciprocal Density Mixing|Reciprocal Density Mixing]]
- [[_COMMUNITY_Architecture Decision Records|Architecture Decision Records]]
- [[_COMMUNITY_Paper Publication Metadata|Paper Publication Metadata]]
- [[_COMMUNITY_CpuT Host Wrapper|Cpu<T> Host Wrapper]]
- [[_COMMUNITY_PcieAccount PCI-E Tracking|PcieAccount PCI-E Tracking]]
- [[_COMMUNITY_VnlBatchData Preprocessing|VnlBatchData Preprocessing]]
- [[_COMMUNITY_Hamiltonian Hermiticity Tests|Hamiltonian Hermiticity Tests]]
- [[_COMMUNITY_Iter1 Density Pointwise Tests|Iter1 Density Pointwise Tests]]
- [[_COMMUNITY_V_eff Pointwise Tests|V_eff Pointwise Tests]]
- [[_COMMUNITY_USPP Density Assembly Architecture|USPP Density Assembly Architecture]]
- [[_COMMUNITY_Mixing CUDA Kernels|Mixing CUDA Kernels]]
- [[_COMMUNITY_ScfIteration Type-State Core|ScfIteration Type-State Core]]
- [[_COMMUNITY_WavefunctionSet Constructor|WavefunctionSet Constructor]]
- [[_COMMUNITY_DensityHistory Mixing State|DensityHistory Mixing State]]
- [[_COMMUNITY_DensityUpdated State Marker|DensityUpdated State Marker]]
- [[_COMMUNITY_ScfIteration Eigenvalue Accessor|ScfIteration Eigenvalue Accessor]]
- [[_COMMUNITY_SpectralBounds Struct|SpectralBounds Struct]]
- [[_COMMUNITY_CudaComplex Re-export|CudaComplex Re-export]]
- [[_COMMUNITY_HARTREE_TO_EV Constant|HARTREE_TO_EV Constant]]
- [[_COMMUNITY_Phase 1 Deferred Items|Phase 1 Deferred Items]]
- [[_COMMUNITY_Phase 2 Deferred Items|Phase 2 Deferred Items]]
- [[_COMMUNITY_Phase 2 Fixes Deferred Items|Phase 2 Fixes Deferred Items]]
- [[_COMMUNITY_PBE XC Native Rust Decision|PBE XC Native Rust Decision]]

## God Nodes (most connected - your core abstractions)
1. `ScfIteration` - 30 edges
2. `BlasHandle` - 14 edges
3. `R-ChFSI Algorithm 3 (Das et al. 2025)` - 13 edges
4. `run_scf()` - 12 edges
5. `gpu_available()` - 12 edges
6. `Das et al. 2025 R-ChFSI paper` - 12 edges
7. `run_scf_with_energy()` - 11 edges
8. `WaveGridArray` - 11 edges
9. `Density` - 11 edges
10. `FftPlan3d` - 11 edges

## Surprising Connections (you probably didn't know these)
- `cuFFT Dimension Ordering Convention` --conceptually_related_to--> `Beta-Psi Gemm Layout Convention`  [INFERRED]
  CONTEXT.md → tests/aug_density_layout.rs
- `tests/backbone_compiles.rs` --implements--> `Type-State SCF Machine`  [EXTRACTED]
  tests/backbone_compiles.rs → PROJECT_ROOT_PLAN.md
- `tests/iter1_density_pointwise.rs` --references--> `USPP Augmentation Density (rho_aug)`  [EXTRACTED]
  tests/iter1_density_pointwise.rs → notes/open-followups.md
- `R-ChFSI Algorithm 3 (Das et al. 2025)` --references--> `Toy convergence plot for generalized eigenproblem (ChFSI vs R-ChFSI)`  [EXTRACTED]
  notes/plans/phase-rchfsi/DECISIONS.md → reference_paper/extracted/das-2025-rchfsi/toy_convergence_generalized.pdf
- `Toy convergence plot for generalized eigenproblem (ChFSI vs R-ChFSI)` --references--> `Das et al. 2025 R-ChFSI paper`  [EXTRACTED]
  reference_paper/extracted/das-2025-rchfsi/toy_convergence_generalized.pdf → notes/plans/phase-rchfsi/TASKS.md

## Hyperedges (group relationships)
- **SCF Phase State Machine** — src_scf_initialized, src_scf_veffbuilt, src_scf_wavefunctionsupdated, src_scf_densityupdated, src_scf_mixed, src_scf_converged, src_scf_scfiteration [EXTRACTED 1.00]
- **Mixing Phase State Machine** — src_mixing_mixingoff, src_mixing_kerker, src_mixing_pulay, src_mixing_densityhistory, src_scf_off_to_kerker_to_pulay [EXTRACTED 1.00]
- **Hamiltonian Application Pipeline (T+V_loc+V_NL+S_inv)** — src_eigensolver_chebyshev_apply_v_loc_hamiltonian, src_eigensolver_chebyshev_apply_v_nl_hamiltonian, src_eigensolver_chebyshev_apply_s_inverse, src_eigensolver_chebyshev_apply_full_hamiltonian [EXTRACTED 1.00]
- **GPU-Resident Hot-Path Performance Invariant** — src_device_pcie_pcieaccount, src_device_pcie_hot_path_guard, src_eigensolver_chebyshev_chebyshev_filter, src_eigensolver_rayleigh_ritz_rayleigh_ritz, src_scf_scfiteration [EXTRACTED 1.00]
- **Three Layers of Type Safety** — gpu_t_cpu_t_wrappers, rowdistributed_columndistributed, wavegrid_finegrid_newtypes [EXTRACTED 1.00]
- **SCF State Transition Machinery** — typestate_scf_machine, into_phase_helper, check_outcome_enum, density_history_mixing [INFERRED 0.95]
- **GPU Eigensolver Pipeline** — chebyshev_filtering, rayleigh_ritz, spectral_bounds_lanczos, rchfsi, s_inverse_woodbury [EXTRACTED 1.00]
- **R-ChFSI convergence architecture** — rchfsi_algorithm, apply_s_inverse, apply_s_times, spectral_bounds, lanczos_estimator, gpu_device_layer [EXTRACTED 0.95]
- **USPP density assembly pipeline** — uspp_density_assembly, q_sf_cache, beta_psi_gpu_residency, omega_occupancy_matrix, gpu_device_layer [EXTRACTED 0.95]
- **SCF bugs requiring corrections** — occupation_formula_bug, fft_dimension_ordering, accumulate_density_kernel, pseudo_loading_bug, spin_deg_rho_aug_bug [EXTRACTED 0.85]
- **SCF divergence root cause investigation chain (May 2025)** — cufft_plan_dim_ordering_bug, rr_transpose_layout_bug, d_screened_diagnostic, rho_aug_spatial_distribution_hypothesis [INFERRED 0.85]
- **R-ChFSI vs Standard ChFSI trade-off: inexact S^{-1} tolerance** — rchfsi_algorithm, standard_chfsi_algorithm, s_inv_s_identity_test, woodbury_s_inverse [EXTRACTED 1.00]
- **GPU-resident SCF architecture pillars** — type_state_scf_backbone, column_distributed_layout, pcieaccount_runtime_tracker, two_channel_density_architecture [EXTRACTED 1.00]
- **R-ChFSI GPU performance across real and complex GEP (residuals + speedups)** — das_2025_rchfsi_gep_residuals_real_lp_figure, das_2025_rchfsi_perf_gpu_real_gep_figure, das_2025_rchfsi_gep_residuals_complex_lp_figure, das_2025_rchfsi_perf_gpu_complex_gep_figure [EXTRACTED 1.00]
- **PAW overlap inversion: Gram matrix structure determines feasibility of Woodbury-based S^{-1} iterative solve** — levitt_torrent_2015_gram_projs_figure, paw_projector_overspill, woodbury_overlap_inverse [EXTRACTED 1.00]

## Communities (45 total, 16 thin omitted)

### Community 0 - "Test Fixtures & Cell Setup"
Cohesion: 0.07
Nodes (33): downsample_array_to_wave_grid(), dummy_cell(), dummy_grid(), mixed_state(), NonSpin, run_scf(), run_scf_with_energy(), ScfIteration<NonSpin, Initialized, MixingOff> (+25 more)

### Community 1 - "GPU Density & CASTEP Reference"
Cohesion: 0.06
Nodes (50): Das 2025 R-ChFSI paper (reference), CASTEP raw density convention (rho x V_cell), construct_density_gpu(), BlasHandle (cuBLAS wrapper), DeviceMapped (CPU-GPU transfer trait), BatchedFftPlan3d (cuFFT batched), FftPlan3d (cuFFT wrapper), Gpu<T> (device-resident wrapper) (+42 more)

### Community 2 - "GPU Data Transfer & Density Core"
Cohesion: 0.07
Nodes (20): complex_slice_to_cuda(), complex_to_cuda(), cuda_to_complex(), cuda_vec_to_complex(), Density, DensityUpsampled, DeviceMapped, EffectivePotential (+12 more)

### Community 3 - "Chebyshev Filtering & S⁻¹ Woodbury"
Cohesion: 0.07
Nodes (47): S^-1 via Woodbury identity, S*psi computation (I + beta*Q*beta^H), Standard Chebyshev Filtered Subspace Iteration, ColumnDistributed GPU wavefunction layout (n_pw, n_bands) col-major, cuFFT plan dim ordering bug: (ngz, ngy, ngx) vs (ngx, ngy, ngz), D_screened amax diagnostic for SCF divergence detection, R-ChFSI residual convergence (complex Hermitian, low precision), R-ChFSI residual convergence (real symmetric, low precision) (+39 more)

### Community 4 - "WavefunctionSet & Density Data Model"
Cohesion: 0.05
Nodes (8): WavefunctionSet, Density, DensityUpsampled, EffectivePotential, FineGridArray, WaveGridArray, Cu111CoFixture (test fixtures), REFERENCE_ENERGY_EV (-24110.96665069)

### Community 5 - "GPU-Resident Augmentation Kernel"
Cohesion: 0.08
Nodes (38): Grid-stride loop accumulate_density kernel, Beta-Psi Gemm Layout Convention, GPU-resident beta_psi_per_ion, CASTEP GPU port source code, Chebyshev Filtering, Cu111_CO CASTEP reference fixture, Eigensolver eigenvalue accuracy debug, SCF diverges after iter-2 debug (+30 more)

### Community 6 - "Bug Patterns & Fortran Interop"
Cohesion: 0.09
Nodes (35): Bug Cancellation Pattern, cdylib Fortran Interop Strategy, CheckOutcome Enum (not nested Result), cuFFT Dimension Ordering Convention, DensityHistory Mixing State Machine, Density Normalization Misattribution Pattern, Density rho-times-Omega Convention, F8 Controlled Validation Experiment (+27 more)

### Community 7 - "Augmentation Density Pipeline"
Cohesion: 0.10
Nodes (26): build_q_sf_cache(), compute_aug_density_fine(), compute_aug_density_gpu(), compute_occupations(), find_chemical_potential(), IonSfEntry, load_q_sf_cache_from_disk(), QSfCache (+18 more)

### Community 8 - "DIIS/Kerker Mixing Pipeline"
Cohesion: 0.12
Nodes (18): build_and_solve_diis(), c2c_forward_inplace(), c2c_inverse_inplace(), DensityHistory<Kerker>, DensityHistory<MixingOff>, DensityHistory<Pulay>, Unconstrained DIIS (no Lagrange multiplier), Kerker (+10 more)

### Community 9 - "Batched cuFFT Plans"
Cohesion: 0.10
Nodes (5): BatchedFftPlan3d, FftPlan3d, test_batched_c2r_4x4x4(), test_batched_c2r_nonuniform(), test_c2c_3d_identity()

### Community 10 - "Validation Test Suite"
Cohesion: 0.16
Nodes (15): band1_v_loc_expectation_matches_castep(), compare_density_against_castep_bin(), compare_eigenvalues_bare_d0(), compare_eigenvalues_with_reference_veff_and_screening(), compare_v_eff_against_pot_fmt(), compute_mu(), constant_v_eff_sanity_check(), cufft_dim_ordering_isolated_diagnostic() (+7 more)

### Community 11 - "Hamiltonian Application Pipeline"
Cohesion: 0.17
Nodes (18): apply_full_hamiltonian(), apply_h_components_for_test(), apply_s_inverse(), apply_scaled_hamiltonian_inplace(), apply_v_loc_hamiltonian(), apply_v_nl_hamiltonian(), c2c_forward_inplace(), c2c_inverse_inplace() (+10 more)

### Community 12 - "cuBLAS Wrapper"
Cohesion: 0.18
Nodes (5): BlasHandle, test_daxpy(), test_dgemm(), test_zgemm_small(), ZgemmConfig

### Community 13 - "Rayleigh-Ritz Eigensolver"
Cohesion: 0.15
Nodes (7): Gpu<T>, rayleigh_ritz(), ColumnDistributed, Cpu, Layout, RowDistributed, Sealed

### Community 14 - "Kerker Preconditioner"
Cohesion: 0.21
Nodes (5): kerker_formula(), KerkerPreconditioner, test_kerker_fixed_q_1p5(), test_kerker_high_g_approaches_one(), test_kerker_monotonic()

### Community 15 - "cuSOLVER Error Handling"
Cohesion: 0.32
Nodes (3): SolverError, SolverHandle, test_zhegvd_4x4_diagonal()

### Community 16 - "SCF State Builder & Fixtures"
Cohesion: 0.38
Nodes (6): build_scf_state(), Cu111CoFixture, fixture(), load_fixture(), parse_bands_file(), pw_coords_to_fft_indices()

### Community 18 - "Architecture Decision Records"
Cohesion: 0.29
Nodes (7): ADR-0001: Type-state SCF backbone, ADR-0002: PCI-E Transfer Tracking with PcieAccount, PcieAccount runtime PCI-E transfer counter, Phase 1: T1 Type-State SCF Backbone Plan, Phase 2: Fill the SCF Transitions (GPU-direct) Plan, Pulay mixing with DensityHistory ring buffer, Type-state SCF backbone (ScfIteration<State> pattern)

### Community 19 - "Paper Publication Metadata"
Cohesion: 0.33
Nodes (5): process, compiler, sources, spec_version, texlive_version

### Community 22 - "VnlBatchData Preprocessing"
Cohesion: 0.50
Nodes (3): build_q_expanded(), VnlBatchData, VnlIonData

### Community 23 - "Hamiltonian Hermiticity Tests"
Cohesion: 0.80
Nodes (4): compute_h_sub_for_pairs(), gpu_available(), hamiltonian_apply_is_hermitian_on_fixture_state(), hamiltonian_apply_is_hermitian_on_iter2_state()

### Community 24 - "Iter1 Density Pointwise Tests"
Cohesion: 0.70
Nodes (4): dump_array_3d(), frac_to_cart(), gpu_available(), iter1_density_pointwise_vs_den_fmt()

### Community 25 - "V_eff Pointwise Tests"
Cohesion: 0.70
Nodes (4): dump_array_3d(), frac_to_cart(), gpu_available(), iter1_iter2_v_eff_pointwise_vs_pot_fmt()

### Community 26 - "USPP Density Assembly Architecture"
Cohesion: 0.50
Nodes (4): ADR-0003: USPP Density Assembly Two-Channel Architecture, QSfCache: geometry-static Q_{nm}(G)*exp(-iG*R_I) cache, rho_aug spatial distribution hypothesis for SCF divergence, Two-channel USPP density architecture (smooth PW rho + rho_aug)

## Knowledge Gaps
- **86 isolated node(s):** `sources`, `spec_version`, `texlive_version`, `compiler`, `Sealed` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ScfIteration` connect `GPU Density & CASTEP Reference` to `Test Fixtures & Cell Setup`, `WavefunctionSet & Density Data Model`, `Augmentation Density Pipeline`?**
  _High betweenness centrality (0.307) - this node is a cross-community bridge._
- **Why does `Rayleigh-Ritz` connect `GPU-Resident Augmentation Kernel` to `Test Fixtures & Cell Setup`, `Bug Patterns & Fortran Interop`?**
  _High betweenness centrality (0.232) - this node is a cross-community bridge._
- **Why does `Chebyshev Filtering` connect `GPU-Resident Augmentation Kernel` to `Chebyshev Filtering & S⁻¹ Woodbury`, `Bug Patterns & Fortran Interop`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `R-ChFSI Algorithm 3 (Das et al. 2025)` (e.g. with `S^-1 via Woodbury identity` and `D_screened amax diagnostic for SCF divergence detection`) actually correct?**
  _`R-ChFSI Algorithm 3 (Das et al. 2025)` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `run_scf()` (e.g. with `backbone_compiles()` and `run_scf_with_energy()`) actually correct?**
  _`run_scf()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `sources`, `spec_version`, `texlive_version` to the rest of the system?**
  _101 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Test Fixtures & Cell Setup` be split into smaller, more focused modules?**
  _Cohesion score 0.06516290726817042 - nodes in this community are weakly interconnected._