# Fix Tasks — Bare-H R-ChFSI review

## Applied During Review

### FIX-1: h_eig pointer arithmetic + missing error check

Already applied to `src/eigensolver/chebyshev.rs` (lines 1428-1456):

1. Changed `ptr_psi_base.wrapping_add((b * n_pw) as u64)` to
   `(ptr_psi_base as *const cuDoubleComplex).add(b * n_pw)` — casts the `u64`
   CUdeviceptr to a typed pointer and uses element-level addition.

2. Added `.result().map_err(Error::Blas)?` to the raw `cublasZdotc_v2` call.

**Verification**: `cargo check --workspace` and `cargo clippy --workspace -- -D warnings` pass.
