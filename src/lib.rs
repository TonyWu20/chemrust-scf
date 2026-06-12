pub mod types;
pub mod layout;
pub mod scf;
pub mod mixing;
pub mod density;
pub mod device;
#[doc(hidden)]
pub mod eigensolver;
pub mod ffi;
pub mod pipeline;
pub mod spin_types;

// Re-export test-only helpers for integration tests
#[doc(hidden)]
pub use eigensolver::kernels::CudaKernelSet;
#[cfg(feature = "chebyshev")]
#[doc(hidden)]
pub use eigensolver::chebyshev::{
    apply_h_components_for_test, apply_s_for_test, chebyshev_filter_for_test,
    chebyshev_filter_for_test_gpu, chebyshev_filter_iteration_gpu,
    compute_residual_norms_for_test, FilterMode, HComponentsForTest,
};
#[doc(hidden)]
pub use eigensolver::preconditioner::tpa;
#[doc(hidden)]
pub use eigensolver::preconditioner::compute_c_matrix;
#[doc(hidden)]
pub use eigensolver::preconditioner::compute_c_global;
#[doc(hidden)]
pub use eigensolver::preconditioner::invert_q_matrix;
#[doc(hidden)]
pub use eigensolver::preconditioner::assemble_r_beta;
#[doc(hidden)]
pub use eigensolver::preconditioner::assemble_q_rcq;
#[doc(hidden)]
pub use eigensolver::preconditioner::apply_preconditioner;
#[doc(hidden)]
pub use eigensolver::preconditioner::TpaPreconditioner;
#[doc(hidden)]
pub use eigensolver::hamiltonian::{apply_full_hamiltonian, apply_s_times};
#[doc(hidden)]
pub use eigensolver::davidson_types::PwCoefficients;
#[doc(hidden)]
pub use eigensolver::davidson_types::PreconditionerVector;
#[doc(hidden)]
pub use eigensolver::davidson_types::KineticPreconditioner;
#[doc(hidden)]
pub use eigensolver::davidson_types::compute_kinetic_energies;
#[doc(hidden)]
pub use eigensolver::davidson::{check_inner_convergence, BandConvStatus};
/// Rayleigh-Ritz test wrapper (returns subspace matrices for diagnostics).
/// Gated behind `chebyshev` feature + `scf_diag` (default feature) or `test` cfg.
#[cfg(all(any(test, feature = "scf_diag"), any(test, feature = "chebyshev")))]
pub use eigensolver::rayleigh_ritz::rayleigh_ritz_with_matrices;
#[cfg(any(test, feature = "scf_diag"))]
pub use eigensolver::d_screening::test_api;
#[doc(hidden)]
pub use eigensolver::vnl_data::VnlBatchData;
#[doc(hidden)]
pub use device::blas::BlasHandle;
#[doc(hidden)]
pub use device::solver::SolverHandle;
#[doc(hidden)]
pub use device::pcie::PcieAccount;
pub mod scf_capture;

pub use types::{
    Density, DensityUpsampled, EffectivePotential, Error, FinalResult, FineGridArray, KineticEnergies, KPoint,
    SmearingParams, SmearingScheme, WaveGridArray,
};
pub use spin_types::{
    ElectronCounts, FermiEnergies, KptDataSet, OccupationSet, PerSpinAugDensity, PerSpinBetaProjections,
    PerSpinDensity, PerSpinEigenvalues, PerSpinPwCoefficients, SpinChannelData, SpinDensity,
};
pub use layout::{Cpu, ColumnDistributed, Layout, RowDistributed, WavefunctionSet};
pub use device::{DeviceMapped, Gpu};
pub mod energy;
pub use energy::{EV_TO_HARTREE, HARTREE_TO_EV};
pub use scf::{
    downsample_array_to_wave_grid, pw_coords_to_fft_indices, run_scf, run_scf_with_energy,
    run_scf_with_energy_gated, BuildVEffWithEnergy, CheckOutcome, Converged, DensityUpdated,
    Initialized, Mixed, MixingPhaseKind, ScfDivergenceGate, ScfIteration, ScfPhase, VEffBuilt,
    WavefunctionsUpdated,
};
pub use mixing::{DensityHistory, Kerker, MixingOff, MixingPhase, Pulay};
