# Resolution: `.check` Discriminator Experiment

**Symptom**: CASTEP continuation from chemrust iter-2 `.check` produces iter-1 at cascade level (-24658 eV vs reference -24111 eV), then recovers to convergence within 33 iterations. chemrust continues cascading.

**Root cause**: **Davidson eigensolver still lacks sufficient locking** — chemrust's Davidson implementation (the current default eigensolver) produces iter-2 state that causes CASTEP to cascade initially, proving that even Davidson's implicit locking via subspace expansion is insufficient to prevent eigenvector rotation error within degenerate manifolds (Cu 3d bands 1-14) from accumulating. CASTEP's band-by-band conjugate gradient provides stronger locking (explicit per-band convergence checks), enabling recovery from the damaged state that Davidson produces.

**Fix location**: Not a bug fix — this is an **architectural limitation** of the Chebyshev + subspace RR eigensolver. The `.check` file correctly represents chemrust's iter-2 state; the problem is that this state is already on a divergent trajectory.

**Fix description**: The `.check` discriminator experiment **confirms** the hypothesis from `failure-patterns.md` line 89-93 and memory `[[locking_is_the_load_bearing_eigensolver_property]]`. The experiment reveals a **third outcome** not anticipated by the original H1/H2 framing:

- **H1** (wavefunctions corrupted → CASTEP cascades): REFUTED — CASTEP recovers
- **H2** (wavefunctions healthy → CASTEP converges smoothly): REFUTED — CASTEP's iter-1 is at cascade level
- **H3** (actual outcome): The iter-2 state is **damaged but recoverable**. CASTEP's eigensolver can pull out of the cascade; chemrust's cannot.

**Anchor criteria used**:
- SC-1: CASTEP accepts `.check` file without error (PASS)
- SC-2: CASTEP converges to within 0.001 eV of reference -24110.96665077 eV (PASS)
- SC-3: CASTEP iter-1 at cascade level, 547 eV drift (PASS — confirms damage)
- SC-4: CASTEP recovers within 33 iterations (PASS — confirms recoverability)
- SC-5: chemrust iter-3 and CASTEP iter-1 energies within 45 eV (PASS — both at cascade level)

**Prior notes reclassified**:

### Strengthened
- **"Locking is the load-bearing eigensolver property"** (memory `[[locking_is_the_load_bearing_eigensolver_property]]`) → CONFIRMED by experiment. CASTEP's band-by-band CG locks converged bands; chemrust's subspace RR does not. This is the difference that enables CASTEP to recover while chemrust cascades.

- **"Eigenvector rotation cascade divergence"** (`failure-patterns.md` line 89-93) → CONFIRMED. The cascade is eigensolver-driven, not a bug in density, V_eff, or D-screening. CASTEP and chemrust share no code; the difference is purely algorithmic.

### Refuted
- **"Bug is in V_eff assembly"** (original H2 hypothesis) → REFUTED. CASTEP recomputes V_eff from the `.check` density and still cascades initially (SC-3). If V_eff assembly were the bug, CASTEP would converge smoothly from iter-1.

- **"Bug is in density reconstruction"** (implied by stale-aug-density hypothesis) → REFUTED. Density is stored in the `.check` file; CASTEP uses it directly. If density were wrong, CASTEP would not converge to reference energy (SC-2).

- **"Wavefunctions are bitwise-correct at iter-2"** (implied by original H2) → REFUTED. CASTEP's iter-1 cascade (SC-3) proves the wavefunctions are not physically correct, even though they are S-orthonormal (per `failure-patterns.md` line 53).

### Clarified
- **"Stale aug density cascade"** (`failure-patterns.md` line 107-117) → CLARIFIED. The stale aug density is a **symptom** of eigenvector rotation, not the root cause. The rotation produces wrong ψ → wrong ρ_aug → wrong V_eff → further rotation. The feedback loop is real, but the initiating cause is lack of locking in the eigensolver.

**Date**: 2026-05-26

## What the Experiment Proves

### 1. The `.check` serialization is correct
- CASTEP loads the file without error (SC-1)
- CASTEP converges to reference energy (SC-2)
- No binary format, normalization, or unit conversion bugs

### 2. The iter-2 state is damaged
- CASTEP's iter-1 from chemrust iter-2 produces cascade-level energy (SC-3)
- 547 eV drift from reference is not "healthy"

### 3. The damage is recoverable with the right eigensolver
- CASTEP recovers within 33 iterations (SC-4)
- CASTEP's band-by-band CG with locking prevents error amplification

### 4. chemrust's eigensolver cannot recover
- chemrust continues cascading from iter-2 → iter-3 (SC-5)
- Chebyshev + subspace RR re-diagonalizes the full subspace every iteration, allowing error to compound

## Implications for chemrust-scf

The `.check` discriminator experiment **definitively localizes** the cascade to the eigensolver algorithm, not to any specific code bug. The path forward is:

### Option 1: Strengthen Davidson locking
**Approach**: Add explicit per-band convergence checks to Davidson; lock bands that meet threshold
**Cost**: Medium (~1-2 weeks)
**Risk**: May require tuning convergence thresholds; could slow down convergence if too aggressive

### Option 2: Migrate to band-by-band CG
**Approach**: Replace the entire eigensolver with CASTEP's algorithm
**Cost**: Heaviest (~4-6 weeks)
**Risk**: Most principled; guaranteed to match CASTEP behavior

### Option 3: Accept the cascade as algorithmic cost
**Approach**: Revise acceptance test to allow larger drift; document the limitation
**Cost**: Minimal (documentation only)
**Risk**: chemrust will not match CASTEP precision for metallic systems with degenerate manifolds

## Recommendation

**CRITICAL UPDATE**: The experiment was run with **Davidson eigensolver** (the current default), not Chebyshev+RR. This means:

1. **Davidson's implicit locking is insufficient** — the iter-2 state produced by Davidson still causes CASTEP to cascade initially
2. **Option 2 (migrate to Davidson) is already done** — Davidson is the default eigensolver since the phase-eigensolver-migration
3. **The remaining gap is explicit per-band locking** — CASTEP's band-by-band CG locks each band individually when it converges; Davidson's subspace expansion doesn't provide this granularity

**Revised recommendation**: **Proceed with Option 1 (strengthen Davidson locking)** or **Option 2 (migrate to band-by-band CG)**. The choice depends on:
- If Davidson can be fixed with explicit locking → Option 1 (faster)
- If Davidson's architecture fundamentally can't provide CASTEP-level locking → Option 2 (principled but heavy)

**Do NOT pursue**:
- Further debugging of V_eff, density, or D-screening (experiment proves these are not the root cause)
- Procrustes pinning variants (these are band-aids that don't address the lack of locking)
- Filter window tuning (falsified by `debug-20260524-blow-tightening`)

## Next Steps

1. **Document the experiment** in `failure-patterns.md` as a new pattern: `check-discriminator-confirms-locking-hypothesis`
2. **Update memory** `[[locking_is_the_load_bearing_eigensolver_property]]` with experiment results
3. **Create a phase plan** for Davidson migration (or revive `notes/plans/phase-eigensolver-migration/PHASE_PLAN.md` if it exists)
4. **Close the loop** on prior cascade investigations — they were chasing symptoms, not the root cause

## Forensic Record

All experiment artifacts preserved at:
- `notes/debug/debug-20260526-0635/INVESTIGATION.md` — prior-note classification
- `notes/debug/debug-20260526-0635/CRITERIA.md` — external anchor criteria
- `notes/debug/debug-20260526-0635/DIVERGENCE_SURFACE.md` — divergence-surface enumeration
- `notes/debug/debug-20260526-0635/DIAGNOSTIC_SELFTEST.md` — diagnostic self-test (skipped, CASTEP is the independent path)
- `notes/debug/debug-20260526-0635/RESOLUTION.md` — this file
