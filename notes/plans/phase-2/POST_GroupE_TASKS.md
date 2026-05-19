# POST Group E Implementation: refined task record

## Group E — mix + check (Goal 5)

All tasks implemented and verified with CPU-only unit tests. See
`/home/tony/.claude/plans/notes-plans-phase-2-tasks-md-group-e-au-merry-pearl.md`
for the design plan.

### Implementation status

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| E-1 | Kerker preconditioner `K(G)=G²/(G²+q²)` on GPU | `src/mixing/kerker.rs` | ✅ |
| E-2 | Reciprocal-space DIIS + Kerker mixing with type-state | `src/mixing.rs` + `src/mixing/reciprocal_density.rs` | ✅ |
| E-3 | Total energy computation (Ewald + assembly) | `src/energy.rs` | ✅ |
| E-4 | Energy-window convergence check | `src/scf.rs` | ✅ |
| E-5 | Wire mixing phase into `run_scf` loop | `src/scf.rs` | ✅ |
| E-6 | CPU unit tests for mixing + energy + check | various | ✅ |

### E-6: Unit tests — what each catches

**`mixing::tests::test_diis_*` (5 tests)**

| Test | What it catches | Failure mode |
|------|----------------|-------------|
| `test_diis_2x2_known_system` | M=[[2,1],[1,2]], b=[5,4] → x=[2,1] | Wrong Gaussian elimination, indexing errors |
| `test_diis_3x3_identity` | M=I, b=[1,2,3] → x=[1,2,3] | Back-substitution stride errors |
| `test_diis_empty` | n=0 → empty vec | Edge case with no history |
| `test_diis_singular_fallback` | zero matrix → fallback=true | Missed singular matrix detection |
| `test_diis_near_singular_fallback` | pivot < 1e-30 → fallback=true | Threshold too tight |

**`mixing::kerker::tests::test_kerker_*` (4 tests)**

| Test | What it catches | Failure mode |
|------|----------------|-------------|
| `test_kerker_g0_zero` | K(G=0)=0 for charge conservation | Nonzero DC would drift total charge |
| `test_kerker_monotonic` | K strictly increasing with G² | Wrong screening wavelength behavior |
| `test_kerker_high_g_approaches_one` | Asymptotic K(G)→1 | Scaling factor error in formula |
| `test_kerker_fixed_q_1p5` | K(q²)=0.5 | Wrong q² mixing parameter |

**`scf::tests::test_check_not_converged_no_energy` (1 test)**
- check() returns `NotConverged { next_mixing: Off }` with no energy data.
- Catches: wrong CheckOutcome variant, moved self after into_phase(), stale next_mixing.

**`energy::tests::test_*` (2 tests)**
- Ewald finite with zero charges; total energy arithmetic.

### How to run

```bash
# All CPU tests (no GPU needed)
cargo test --lib

# Individual categories
cargo test mixing::tests::      # DIIS solve tests
cargo test mixing::kerker::     # Kerker formula tests
cargo test scf::tests::         # check() test
cargo test energy::tests::      # energy tests

# Full suite
cargo test --lib                # 22 tests, all pass
```

### What is deferred to Group F (needs Cu111_CO fixture + GPU)

| Test | Why deferred |
|------|-------------|
| Ewald energy vs CASTEP reference | Needs fixture cell + pseudopotential loading |
| DIIS residual norm decreases monotonically | Needs real wavefunctions on GPU |
| Density RMS after mix < before mix | Needs real GPU density construction |
| Full SCF convergence energy | Needs full pipeline + GPU |
