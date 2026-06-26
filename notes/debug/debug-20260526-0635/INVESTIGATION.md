# Prior Investigation Notes Classification

**Symptom**: CASTEP continuation from chemrust iter-2 `.check` produces iter-3 energy at similar cascade level to chemrust's iter-3, then converges smoothly.

**Source documents**:
- `~/.claude/plans/i-have-an-idea-sunny-bunny.md` — `.check` discriminator proposal
- `/tmp/iter-2-dump-iter-3-energy.log` — chemrust iter-2 → iter-3 energy output
- `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0525_from_chemrust/Cu111_CO.castep` — CASTEP continuation log
- `notes/failure-patterns.md` — prior cascade investigations
- `notes/debug/debug-20260523-2314/RESOLUTION.md` — stale-aug-density-cascade
- `notes/debug/debug-20260524-blow-tightening/RESOLUTION.md` — eigenvector rotation

## Numeric Claims Classification

### EXTERNAL (admissible as criteria)

| Claim | Value | Source | Verification |
|-------|-------|--------|--------------|
| CASTEP reference final energy | -24110.96665077 eV | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0522_F8/Cu111_CO.castep` | Direct fixture read |
| CASTEP continuation final energy | -24110.96563535 eV | `/export/public_castep_jobs/tony/Cu111_CO_Single_Point_0525_from_chemrust/Cu111_CO.castep` | Direct fixture read |
| CASTEP continuation iter-1 energy | -24658.0671 eV | Same file, line "1  -2.46580671E+004" | Direct fixture read |
| chemrust iter-2 total energy | -24189.8598 eV | `/tmp/iter-2-dump-iter-3-energy.log` line 1 | Direct test output |
| chemrust iter-3 total energy | -24703.4148 eV | `/tmp/iter-2-dump-iter-3-energy.log` line 26 | Direct test output |
| chemrust iter-2 V_eff range | 8.9099 Ha | `/tmp/iter-2-dump-iter-3-energy.log` line 3 | Direct test output |
| chemrust iter-2 band-0 eigenvalue | -0.9957889622 Ha | `/tmp/iter-2-dump-iter-3-energy.log` line 33 | Direct test output |
| chemrust iter-2 Fermi energy | -0.0816739416 Ha | `/tmp/iter-2-dump-iter-3-energy.log` line 32 | Direct test output |

### DERIVED (not admissible without independent corroboration)

| Claim | Value | Source | Why DERIVED |
|-------|-------|--------|-------------|
| "iter-2 wavefunctions are corrupted" | hypothesis | `i-have-an-idea-sunny-bunny.md` line 9 | Inferred from cascade behavior, not measured independently |
| "V_eff/D_screened assembly bug" | hypothesis | `i-have-an-idea-sunny-bunny.md` line 10 | Inferred from cascade behavior, not measured independently |
| "Stale aug density contaminates V_eff" | hypothesis | `debug-20260523-2314/RESOLUTION.md` line 7-12 | Inferred from T3/T4 substitution tests, not directly measured |
| "D-screening diverges by 0.1-2 Ha" | 2.13 Ha max | `debug-20260523-2314/RESOLUTION.md` line 48 | Computed by our own `compute_screened_d`, not CASTEP output |
| "Cu-3d projector ratio 0.893" | 0.893 | `debug-20260524-blow-tightening/RESOLUTION.md` line 4 | Computed by our overlap diagnostic, not CASTEP |
| "Eigenvector rotation within degenerate manifolds" | qualitative | `failure-patterns.md` line 89-93 | Inferred from overlap measurements, not CASTEP eigenvector dump |

### HYPOTHESIZED (not admissible)

| Claim | Value | Source | Why HYPOTHESIZED |
|-------|-------|--------|------------------|
| "If CASTEP cascades at iter-3, wavefunctions are corrupted" | conditional | `i-have-an-idea-sunny-bunny.md` line 9 | Prediction, not observation |
| "If CASTEP converges, bug is in V_eff assembly" | conditional | `i-have-an-idea-sunny-bunny.md` line 10 | Prediction, not observation |
| "Subspace RR rotates eigenvectors, CG preserves" | qualitative | `failure-patterns.md` line 92 | Algorithmic reasoning, not measured on this system |

## Key Observation

The `.check` discriminator proposal (line 8-10) frames two mutually exclusive hypotheses:
1. **H1**: iter-2 wavefunctions are corrupted → CASTEP will cascade at iter-3
2. **H2**: iter-2 wavefunctions are healthy, bug is in V_eff assembly → CASTEP will converge

**Actual result**: CASTEP's iter-1 (continuation from chemrust iter-2) produces energy **-24658.0671 eV**, which is at the same cascade level as chemrust's iter-3 **-24703.4148 eV** (both ~600 eV below reference -24110.97 eV). But CASTEP then **converges smoothly** to -24110.96563535 eV (within 0.001 eV of reference).

This result **does not cleanly fit either H1 or H2**:
- Against H1: CASTEP does NOT cascade further after iter-1; it recovers.
- Against H2: CASTEP's iter-1 IS at cascade-level energy, suggesting the iter-2 state is not "healthy."

**Interpretation**: The iter-2 state is **damaged but recoverable**. CASTEP's eigensolver + SCF feedback can pull out of the cascade, while chemrust's cannot.

## Memory Citation Audit

No memory entries were cited in the `.check` discriminator proposal. The proposal references:
- `handoff-iter3-cascade-2026-05-25.md` (not found in repo)
- Prior debug sessions (classified above)

All numeric anchors are EXTERNAL (fixture files) or DERIVED (our own diagnostics).
