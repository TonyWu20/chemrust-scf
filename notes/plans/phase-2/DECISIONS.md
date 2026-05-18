# Phase 2 Design Decisions

## Grill Outcomes

### GPU-direct (not CPU-first)
**Decision:** Skip CPU prototype. Implement all transition bodies directly against
cudarc (cuFFT + cuBLAS + cuSOLVER) on the GPU.

**Why:** chemrust-hamiltonian already validates the H|ψ> pipeline on CPU against
CASTEP. Writing CPU transition bodies then rewriting for GPU doubles work without
reducing risk. The hot path (eigensolver, density construction) goes straight
against GPU libraries.

**Replaces:** PROJECT_ROOT_PLAN "Tier 2: CPU-first prototype, Tier 3: GPU." The
tiers are merged.

### Risk-weighted ordering: diagonalize first
**Decision:** Goal 3 (diagonalize) before Goal 4 (build_v_eff + construct_density).

**Why:** diagonalize carries ALL the project risk — Chebyshev filtering composing
H_loc FFT + V_NL β-projector gemm has never been tested in this architecture.
build_v_eff is ~40 lines of glue around VEffBuilder (already validated). Risk
weight, not line count, determines order.

### Pulay mixing (not linear mixing)
**Decision:** Implement Pulay/DIIS mixing directly. Skip linear mixing.

**Why:** Cu111_CO (a metal) won't converge with linear mixing. Pulay adds ~25
lines of linear algebra (history_size × history_size solve) on top of the same
infrastructure. No throwaway code.

### cudarc (not gpufft)
**Decision:** Use cudarc's raw cuFFT/cuBLAS/cuSOLVER sys bindings instead of
the gpufft crate.

**Why:** cudarc already required for cuBLAS and cuSOLVER (single dependency).
gpufft has batch=1 limitation for 3D FFT, which blocks `cufftPlanMany` for
density construction. cudarc exposes raw sys bindings for all three vendor
libraries — the wrapper is ~50 lines of safe Rust either way.

### PBE XC is native Rust (not libxc)
**Decision (corrected):** The PBE XC is a direct handwritten port from CASTEP
(`xc/kernels.rs::xc_pbe_point`), not libxc.

**Why:** The plan originally assumed libxc dependency for XC. This was wrong.
VEffBuilder's CPU barrier is FFT (rustfft), not XC. No forced D2H for XC.

### VEffBuilder stays CPU in Phase 2
**Decision:** Wrap chemrust-hamiltonian's CPU VEffBuilder as-is. Accept
H2D/D2H for V_eff assembly.

**Why:** VEffBuilder uses rustfft for FFT-dependent operations (density gradient
for GGA, Poisson solve). Porting to cuFFT requires changes to
chemrust-hamiltonian. Deferred to Phase 3+.

### flake.nix CUDA configuration with cuDNN override
**Decision:** Port CUDA overlay from CASTEP-GPU-port/flake.nix. `cudaSupport =
true`, `cudaCapability = ["6.1"]`, `cudaVersion = "12.9"`. cuDNN override at
v9.11.1.4 required for Pascal cc 6.1 toolchain compatibility.

**Why:** System GPU is Pascal (cc 6.1). Newer CUDA toolkit drops cc 6.1 support
in some packages without the override.

### Keep SpinPolicy generic
**Decision:** `ScfIteration<S: SpinPolicy, State>` stays generic. `S` defaults
to `NonSpin`.

**Why:** User explicitly chose this. VEffBuilder and HkBuilder upstream already
use the same generic. CheckOutcome would need either an enum layer or a separate
structural type if removed. The type parameter is low-cost with the default.

### Lock field visibility before any physics code
**Decision:** `pub` → `pub(crate)` on all ScfIteration fields must be Goal 1
(before any transition body work).

**Why:** The type-state safety is the project's entire reason for existing. All-
pub fields let any code write `state.v_eff = Some(...)` bypassing build_v_eff.
The safety guarantee is ornamental until this is locked.

## Architectural Decisions (overrides to ADR-0001)

### `HkBuilder` for H(k) assembly (not just V_eff)
The diagonalize transition applies the full H = T + V_eff + V_NL. chemrust-
hamiltonian provides `apply_full_hamiltonian` (hamiltonian.rs:74) which composes
`apply_local_hamiltonian` (FFT roundtrip for T + V_loc) + `apply_nlpot`
(β-projector gemm). Testing diagonalize independently (before build_v_eff
exists) uses V_eff from `.pot_fmt` fixture + pseudopotentials.

### Device layer no longer transparent Deref
Phase 1 `Gpu<T>` implemented `Deref<Target=T>` (CPU identity). Phase 2 replaces
this with real `DeviceBuffer`-backed sync. `Gpu<T>` no longer derefs to `T` —
all access is stream-aware through methods on `Gpu<T>`. `sync_to_host()` returns
`Cpu<T>`, `sync_to_device()` returns `Gpu<T>`.
