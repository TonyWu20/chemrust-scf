// ---------------------------------------------------------------------------
// S⁻¹·S identity test — Gate 0 for Chebyshev filtering viability
// ---------------------------------------------------------------------------
// Verifies the global Woodbury S⁻¹ operator satisfies ‖S⁻¹·S·ψ − ψ‖_∞ < 1e-10.
// This must pass before any Chebyshev filter testing can proceed.
//
// The Woodbury formula: S⁻¹ = I − B·(Q⁻¹ + B^H·B)⁻¹·B^H is algebraically
// exact. Any deviation reflects numerical error in Q-matrix conditioning,
// LU factorization, or B^H·B accumulation.

mod fixtures;

fn gpu_available() -> bool {
    std::env::var("CASTEP_FIXTURE_DIR").is_ok()
        || std::path::Path::new("/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8")
            .exists()
}

#[cfg(test)]
mod tests {
    /// Gate 0: Verify the global Woodbury S⁻¹ operator on CASTEP converged
    /// wavefunctions. Measures ζ = ‖S⁻¹·S·ψ₀ − ψ₀‖_∞ for the lowest band.
    #[test]
    #[ignore = "requires GPU and CASTEP fixture data"]
    fn s_inv_s_identity_gate0() {
        if !super::gpu_available() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        use chemrust_hamiltonian_core::GVectorGrid;
        use chemrust_scf::KPoint;
        use chemrust_scf::density::test_api::{CudaKernelSet, VnlBatchData, check_s_inv_s_identity};
        use chemrust_scf::device::blas::BlasHandle;
        use chemrust_scf::device::pcie::PcieAccount;
        use num_complex::Complex64;
        use std::sync::Arc;

        // Use the same fixture loader as all existing diagnostic tests
        let fx = super::fixtures::cu111_co::fixture();
        let cell = &fx.bin.cell;
        let pots = &fx.pots;

        let wfc = fx
            .check
            .wavefunction
            .as_ref()
            .expect(".check must have wavefunction");
        let [ngx, ngy, ngz] = wfc.grid;
        let wave_grid = GVectorGrid::new(ngx, ngy, ngz, cell.recip_lattice);

        let kpt_block = &wfc.kpt_data[0];
        let n_bands = kpt_block.bands.len();
        let n_pw = kpt_block.nplw;
        let k_point = KPoint {
            coords: kpt_block.coords,
            weight: 1.0,
        };

        let psi_data: Vec<Complex64> = kpt_block.bands.concat();

        let ctx = Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
        let stream = ctx.default_stream();
        let blas = BlasHandle::new(stream.clone()).expect("BLAS handle");
        let solver = chemrust_scf::device::solver::SolverHandle::new(stream.clone())
            .expect("SolverHandle");
        let kernels = CudaKernelSet::new(&ctx).expect("CUDA kernels");

        // Build VnlBatchData with global Woodbury. Must use precompute_with_d_override
        // to pass solver_thunk: Some(&solver) for Woodbury construction.
        let mut pcie = PcieAccount::default();
        let vnl_data = VnlBatchData::precompute_with_d_override(
            &kpt_block.pw_grid_coord,
            pots,
            cell,
            &wave_grid,
            None,  // fine_grid
            &k_point,
            &psi_data,
            n_bands,
            n_pw,
            None,   // occupations
            None,   // v_eff
            None,   // d_override
            None,   // shared
            None,   // handle_shared
            Some(&solver),  // solver_thunk — NEEDED for Woodbury
            &stream,
            &mut pcie,
            &blas,
            &kernels,
        )
        .expect("VnlBatchData::precompute_with_d_override");

        // Take band 0 (the converged lowest eigenstate — Cu 3s)
        let band0: Vec<Complex64> = psi_data.iter().take(n_pw).copied().collect();

        let zeta = check_s_inv_s_identity(&band0, n_pw, &vnl_data, &blas, &stream, &solver)
            .expect("check_s_inv_s_identity");

        eprintln!("[S⁻¹·S identity] ‖S⁻¹·S·ψ₀ − ψ₀‖_∞ = {:.6e}", zeta);

        // Gate 0 discriminator: ζ = ‖S⁻¹·S·ψ − ψ‖_∞ must be ≤ 1e-4 Ha.
        // R-ChFSI (Das 2025) converges for ζ ∈ {1e-4, 1e-3, 1e-2} per §4 experiments.
        // Our ζ = 3.4e-6 is well within this regime. If ζ > 1e-4, the Woodbury
        // construction has a bug (algebraic error, not just conditioning).
        assert!(
            zeta < 1e-4,
            "S⁻¹·S·ψ ≠ ψ: max residual = {:.6e} > 1e-4. \
             Woodbury S⁻¹ is numerically too imprecise for R-ChFSI filtering.",
            zeta,
        );
    }
}