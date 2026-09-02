#![cfg(feature = "chebyshev")]
// Temporary probe: VnlBatchData sizes for the Cu111_CO fixture.
mod fixtures;

use std::sync::Arc;

fn gpu_available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}

#[test]
fn probe_vnl_sizes() {
    if !gpu_available() {
        eprintln!("no GPU, skip");
        return;
    }
    use chemrust_hamiltonian_core::GVectorGrid;
    use chemrust_scf::KPoint;
    use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData};
    use chemrust_scf::device::blas::BlasHandle;
    use chemrust_scf::device::pcie::PcieAccount;
    use num_complex::Complex64;

    let fx = fixtures::cu111_co::fixture();
    let cell = &fx.bin.cell;
    let pots = &fx.pots;
    let wfc = fx.check.wavefunction.as_ref().expect(".check wavefunction");
    let wave_grid = GVectorGrid::new(
        wfc.grid[0], wfc.grid[1], wfc.grid[2], cell.recip_lattice,
    );
    let kpt_block = &wfc.kpt_data[0];
    let n_bands = kpt_block.bands.len();
    let n_pw = kpt_block.nplw;
    let k_point = KPoint { coords: kpt_block.coords, weight: 1.0 };
    let psi_data: Vec<Complex64> = kpt_block.bands.concat();
    let pw_coords = &kpt_block.pw_grid_coord;

    let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();
    let blas = BlasHandle::new(stream.clone()).expect("BLAS handle");
    let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone())
        .expect("SolverHandle");
    let kernels = CudaKernelSet::new(&ctx).expect("CUDA kernels");

    let mut pcie = PcieAccount::default();
    let vnl_data = VnlBatchData::precompute_with_d_override(
        pw_coords, pots, cell, &wave_grid,
        None, &k_point, &psi_data, n_bands, n_pw,
        None, None, None, None, None,
        Some(&solver),
        &stream, &mut pcie, &blas, &kernels,
    ).expect("VnlBatchData::precompute_with_d_override");

    eprintln!(
        "n_total_expanded = {}, per_ion_ne = {:?} ({} ions), entries = {}",
        vnl_data.n_total_expanded,
        vnl_data.shared.handle.per_ion_n_expanded,
        cell.num_ions,
        vnl_data.entries.len(),
    );
}
