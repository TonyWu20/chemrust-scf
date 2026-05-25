# TASKS: CASTEP `.check` Discriminator

## Fixtures

- Cu111_CO.check — CASTEP 6.11 gamma-point Cu111+CO `.check` file
- Cu111_CO.castep_bin — matching density file on wave grid

## Tasks

### Group A: chemrust-hamiltonian serialization (PREREQUISITE)

**Status**: ✅ Done (commit 8c03fbc)

**Success criteria**:
- [x] `write_check()` serializes all 14 sections in correct order
- [x] Fortran record framing: 4-byte BE length prefix/suffix
- [x] Round-trip test passes: read → write → re-read → key fields match
- [x] Determinism test: same input → same byte stream
- [x] All 202 crate tests pass

**Test code**: `tests::test_write_read_roundtrip` in `write_check.rs`

### Group B: chemrust-scf capture module

**Status**: ✅ Done (commit 25cb72b)

**Success criteria**:
- [x] `capture_as_castep_bin()` constructs `CastepBin` from `ScfIteration`
- [x] Wavefunction extraction uses band-major indexing (`psi.data[b*n_pw..(b+1)*n_pw]`)
- [x] Density assembled via `upsample_density_to_fine_grid()` + aug addition
- [x] Occupations recomputed (not stored on ScfIteration)
- [x] Fine grid dims converted from `[ngz,ngy,ngx]` (GVectorGrid) to `[ngx,ngy,ngz]`
- [x] Feature-gated behind `scf_diag`
- [x] No warnings, compiles cleanly

### Group C: Insertion point in SCF loop

**Status**: ✅ Done (commit 25cb72b)

**Success criteria**:
- [x] Capture fires at `iter_count == 2` in NotConverged branch
- [x] Output path from `CHEMRUST_CHECK_DUMP` env var (default `chemrust_iter2.check`)
- [x] `n_electrons` computed from cell species + PP ionic charges (same formula as `check()`)
- [x] Error handling: capture failures are non-fatal (tracing::warn only)

## Verification

### 1. Round-trip test (hamiltonian) ✅
Tests in `write_check.rs`: `test_write_read_roundtrip`, `test_write_check_is_deterministic`

### 2. Structural validation (scf)
To run (requires GPU + CUDA):
```
CHEMRUST_CHECK_DUMP=/tmp/chemrust_iter2.check cargo test --release test_scf_converges_with_davidson -- --ignored --nocapture
```
Verify: file written, non-zero size, check file sections parse correctly.

### 3. CASTEP oracle test (discriminator)
After producing `/tmp/chemrust_iter2.check`:
1. Copy to CASTEP job directory
2. Run CASTEP with `continuation : /tmp/chemrust_iter2.check` and `max_scf_cycles : 5`
3. Inspect `.castep` output for iter-3:
   - Cascade (V_eff > 20 Ha, D_screened > 50 Ha) → wavefunctions corrupted
   - Converge (V_eff ~9 Ha, energy descends) → V_eff/D_screened assembly bug
