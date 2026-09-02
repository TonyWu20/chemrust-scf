# Investigation: CG Implementation Gap vs CASTEP

**Date**: 2026-05-28
**Symptom**: What's our gap in CG implementation compared to CASTEP now?

## Classification of Prior Claims

### EXTERNAL (from fixture files, CASTEP source, or published specs)

| Claim | Source | Verification |
|-------|--------|-------------|
| CASTEP band-0 eigenvalue = -1.05502287 Ha | Cu111_CO.castep | EXTERNAL — CASTEP output |
| CASTEP total energy = -24110.96665077 eV | Cu111_CO.castep | EXTERNAL — CASTEP output |
| CG algorithm: FR direction update, 2 SD + CG, eigenvalue-diff convergence | electronic.f90:11639-12019, 6238-6437 | EXTERNAL — verified during 2026-05-26 audit |
| Line search: b·d·s² + 2(a·d−c)·s − b = 0 | electronic.f90:10074-10090 | EXTERNAL — verified during 2026-05-26 audit |
| USPP preconditioner: P⁻¹ = T⁻¹ + T⁻¹·β·R·β†·T⁻¹ | nlpot.f90:15480 | EXTERNAL — verified during 2026-05-26 audit |
| TPA formula: tpa = 1/(1 + 16x⁴/temp), temp = 27 + 18x + 12x² + 8x³ | wave.f90:29889-29893 | EXTERNAL — verified during 2026-05-26 audit |

### DERIVED (from our own pipeline, possibly buggy)

| Claim | Origin | Issue |
|-------|--------|-------|
| "Gate 1 and Gate 2 tests pass" | Unknown — test results not available | Tests may fail; need to verify |
| "CG converges to CASTEP precision" | Implicit in Gate 2 design | Not yet proven by passing Gate 2 |
| "USPP preconditioner with concatenated per-ion β" | Gate 2 test implementation | May differ from CASTEP's per-ion application |

### HYPOTHESIZED (inferred from scaling or reasoning)

| Claim | Inference chain |
|-------|----------------|
| "CG will fix the cascade" | Prior debug session confirmed locking is load-bearing; CG has implicit per-band locking |
| "Phase-0 CG is algorithmically correct" | Code follows CASTEP line-by-line, but not yet validated end-to-end |
