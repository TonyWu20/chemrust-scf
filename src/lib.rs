pub mod types;
pub mod layout;
pub mod scf;
pub mod mixing;
pub mod density;
pub mod device;
pub(crate) mod eigensolver;

#[doc(hidden)]
pub use eigensolver::davidson_minimal::{DAVIDSON_LAST_DIAG, DavidsonDiagnostic};

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
