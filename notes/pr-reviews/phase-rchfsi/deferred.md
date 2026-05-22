# Deferred — Phase R-ChFSI

Items identified during review that are worth doing but out of scope for this phase. Candidates for the next `/define-outcomes` discussion.

| Priority | Item | Rationale |
|----------|------|-----------|
| MEDIUM | Split `src/eigensolver/chebyshev.rs` into sub-modules (kernels, spectral, s_ops, filter, hamiltonian_apply) | 1769-line monolith with 14+ responsibilities. Refactor, no functional change. |
| LOW | Remove `apply_scaled_hamiltonian_inplace` | Dead code from old Chebyshev. Keep as reference during R-ChFSI stabilization. |
| LOW | Remove transpose-related dead code and `zero_buffer_real` | From earlier phases, never wired. `#[allow(dead_code)]` suppresses warnings but not maintenance. |
| MEDIUM | Add GPU pointwise-multiply kernel for aug density structure factor | Eliminates D2H/H2D round-trip in `compute_aug_density_gpu`. Single biggest hot-path perf win. |
| LOW | Extract energy computation from `scf.rs` to dedicated module or upstream to `chemrust-hamiltonian-core` | Physics computation mixed with orchestrator. |
| DEFERRED | SpinCollinear energy support | Phase 3+ scope per existing planning. `run_scf_with_energy` is hardcoded to `NonSpin`. |
