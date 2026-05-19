# ADR-0002: PCI-E transfer tracking with PcieAccount

**Date:** 2026-05-19

**Status:** Adopted (Phase 2)

## Context

The Phase 2 PHASE_PLAN.md defines a specific GPU residency model: which data
lives on GPU vs CPU during each SCF stage, and how many bytes cross PCI-E per
iteration. Violating this model (e.g., D2H inside the Chebyshev loop) silently
destroys performance.

The code review of our Group C implementation found two real violations:
1. `compute_spectral_bounds` D2H the entire V_eff grid (~256 KB) instead of
   using a GPU reduction
2. `rayleigh_ritz` D2H the whole wavefunction (~4 MB) for a CPU transpose
   instead of using the GPU transpose kernel that already existed

Neither broke correctness, but both were invisible to reviewers without manual
PCI-E auditing. We need a mechanism that surfaces unexpected transfers.

## Decision: Runtime counter with `#[must_use]`

We introduce `PcieAccount` — a struct that accumulates H2D and D2H byte counts
when threaded through the critical path:

```rust
#[derive(Debug, Clone, Copy, Default)]
#[must_use]        // ← compile-time warning if dropped without inspection
pub(crate) struct PcieAccount {
    pub h2d_bytes: usize,
    pub d2h_bytes: usize,
}
```

`Gpu<T>` gains two optional-entry methods in `device/pcie.rs`:

```rust
impl<T: DeviceMapped> Gpu<T> {
    pub(crate) fn from_host_with(... acc: &mut PcieAccount) -> ...;
    pub(crate) fn sync_to_host_with(... acc: &mut PcieAccount) -> ...;
}
```

The orchestrator (`scf.rs::diagonalize`) creates the account, passes it to each
transfer call, and asserts the final counts match expectations:

```rust
let mut pcie = PcieAccount::default();
let v_eff_gpu = Gpu::from_host_with(&v_eff, &stream, &mut pcie)?;
let psi_gpu   = Gpu::from_host_with(&psi, &stream, &mut pcie)?;
// ... hot path (zero transfers by construction) ...
let Cpu(psi) = psi_new_gpu.sync_to_host_with(&stream, &mut pcie)?;
assert_eq!(pcie.d2h_bytes, psi_bytes + eig_bytes, "...");
```

### What it catches

- **Unexpected D2H** in `compute_spectral_bounds` (full V_eff grid → CPU)
- **Unexpected H2D** in Rayleigh-Ritz transpose (host roundtrip)
- **Any future refactoring** that adds a transfer in the hot path
- Extraneous transfers from workspace allocations that accidentally sync

### What it does NOT enforce at compile time

The counter is **runtime**, not compile-time. The compiler does not prevent
calling `stream.clone_dtoh` without going through `PcieAccount`. The
guarantees are:

| Mechanism | When enforced | What it prevents |
|-----------|--------------|------------------|
| `Gpu<T>` has no `Deref<Target=T>` | **Compile time** | Accidental CPU reads of GPU data |
| `#[must_use]` on `PcieAccount` | **Compile time (warning)** | Dropping the account without checking |
| `assert_eq!` on byte counts | **Runtime** | Unexpected transfers in the hot path |
| `from_host_with` / `sync_to_host_with` | **Convention** | Untracked transfers (callers can still use `from_host` / `sync_to_host`) |

## Full compile-time enforcement and why we do NOT pursue it

A true compile-time PCI-E counter would need one of:

1. **Typestate on Gpu:** `Gpu<T, const H2D: usize, const D2H: usize>` with
   const-generic counters incremented per operation. Rust's const generics
   do not support mutation or trait-bound arithmetic (as of 2026).

2. **Linear types.** Rust's ownership system is affine, not linear. Dropping
   a value without inspecting its counter is allowed. `#[must_use]` is a best
   effort.

3. **Affine effect types.** Languages with effect typing (Koka, F*)
   could encode "this function has 2 H2D + 2 D2H side effects." Rust does
   not have effect types.

None are practical today. The runtime counter + `#[must_use]` gives ~90% of
the benefit for near-zero implementation cost.

## Consequences

Positive:
- Every `diagonalize()` call asserts its transfer budget
- Refactoring that adds a transfer gets caught immediately (test failure)
- `#[must_use]` forces callers to at least acknowledge the account

Negative:
- Requires threading `&mut PcieAccount` through the call chain
- Does not cover `stream.clone_htod` / `stream.clone_dtoh` called directly
  (only `Gpu::from_host_with` / `sync_to_host_with`)
- The assertion is in test/verification code, not in the type itself

Neutral:
- Existing `Gpu::from_host` / `sync_to_host` remain for non-critical paths
  (test setup, fixture loading, one-off transfers)
- New code should prefer `_with` variants on the hot path
