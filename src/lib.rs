pub mod types;
pub mod layout;
pub mod scf;
pub mod mixing;
pub mod density;
pub mod device;
pub(crate) mod eigensolver;

// Re-export test-only helpers for integration tests
#[doc(hidden)]
pub use eigensolver::chebyshev::{apply_s_for_test, chebyshev_filter_for_test, FilterMode};
#[doc(hidden)]
pub use eigensolver::vnl_data::VnlBatchData;
#[doc(hidden)]
pub use device::blas::BlasHandle;
#[doc(hidden)]
pub use device::solver::SolverHandle;
pub mod scf_capture;

pub use types::{
    Density, DensityUpsampled, EffectivePotential, Error, FinalResult, FineGridArray, KineticEnergies, KPoint,
    SmearingParams, SmearingScheme, WaveGridArray,
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
