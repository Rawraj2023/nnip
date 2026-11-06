# Table 3 oracle v2 (session 3, 2026-11-06) — item B of `../runs/OPEN-ITEMS-PREREGISTRATION.md`

Files: `table3_oracle.py` (solver; bundled `../oracle/` untouched), `table3_oracle.json` (raw,
all methods x L in {10,12,16,20,24} x S in {2.0..2.4} x branch), `table3_oracle.stderr.log`,
`judge.py` -> `judge.md` (pre-registered criteria applied mechanically).
Environment: CPU, scipy 1.18.1, numpy 2.5.4 (`/workspace/.venv-pinn`). Runtime ~25 min.
Eq. 31 re-checked against the PDF page-10 image this session: the coded operator is the exact
linearisation of steady Eq. 13 with `+alpha g'`; the printed equation drops primes (e.g. it shows
`+2ff''g0` and `-2ff''g0`, which would cancel), so the linearisation is the authority.

## Results (beta=0.1, K=0.1, A*=0.1, Pr=2, B*=0)

| S | lower alpha1 (converged) | paper lower | upper lowest finite-L root | alpha* (far-field) | paper upper |
|---|---|---|---|---|---|
| 2.0 | **-0.49257** | -0.0376726 | 0.7348 (converged) | 0.8205 | 0.8810518 |
| 2.1 | **-0.61135** | -0.1290175 | 1.0493 (converged) | 1.0882 | 1.1498579 |
| 2.2 | **-0.70803** | -0.2006274 | ~1.39 (0.36 % drift L20->24) | 1.4016 | 1.4609727 |
| 2.3 | **-0.79412** | -0.2601093 | not converged (1.79 -> 1.92) | 1.7828 | 1.8396721 |
| 2.4 | **-0.87670** | -0.3115262 | not converged (2.28 -> 2.39) | 2.2618 | 2.3116364 |

1. **Lower branch — converged, pre-registered criterion met at every S.** Shooting with g'(L)=0,
   corrected Chebyshev, and shooting with the exact asymptotic BC agree to < 1e-7 relative, and
   change < 1e-5 between L=20 and 24. The lower branch is unstable (sign agrees with the paper),
   but the magnitudes are 2.8x-13x larger than Table 3. **Table 3's lower column does not
   reproduce** from the paper's own Eq. 31/33.
2. **Upper branch — pre-registered hypothesis REFUTED at S=2.0 and 2.1.** The lowest finite-L root
   does not approach alpha* from above: it is converged (< 0.02 % change, L=16..24) and lies
   below alpha* (-10 %, -4 %). Higher roots do crowd toward alpha* as L grows. The refutation
   condition as written ("converged and > 2 % above alpha*") was not met either: the outcome was a
   case the pre-registration did not foresee. Interpretation (not tested further): for K < alpha <
   alpha* both far-field modes decay monotonically, and g'(L)=0 at large L forces the
   slow-decaying mode's coefficient to vanish, so the converged root is the "fast-decay"
   eigenvalue; under the stated condition g'(inf)=0 alone, every alpha > K still qualifies
   (continuous spectrum). At S=2.2..2.4, where the root sits at or above alpha*, neither method
   converges: Chebyshev becomes ill-conditioned for L >= 16..20 (spurious |alpha| ~ 1e2) and
   shooting drifts with L. **No reliable upper magnitude is claimed for S >= 2.2.** What is robust:
   every upper candidate root is > K > 0, so the upper branch is stable (sign agrees with the
   paper).
3. **Attribution (informative only).** The bundled `oracle/ref_eig_cheb.py` boundary placement
   (wall BCs applied at eta=L, far-field BC at the wall) at its own default L=12 gives upper
   values 0.88253, 1.14325, 1.45075, 1.82675, 2.30169: within 0.17-0.70 % of all five Table 3
   upper values. The same placement gives positive values for the lower branch, so it does not
   reproduce the lower column. This is consistent with, but does not prove, the upper column
   having come from a computation with that BC placement.
4. **Energy disturbance (Eq. 32 with g0=0), not examined before.** All finite-L energy eigenvalues
   are positive (lower branch: 0.03-0.12 at L=24, decreasing with L). They do not change either
   sign conclusion. On the upper branch at S >= 2.3 they fall below the momentum root at L=24,
   which is a further reason not to quote an upper "smallest eigenvalue" magnitude.

## Consequence for the PINN evaluation
Lower-branch PINN alpha1 (baseline -0.48820, candidate-005 -0.48808 at S=2.0, session 2) is within
0.9 % of the converged -0.49257. The evaluator's Table 3 magnitude gates compare against values
that this oracle does not reproduce. The evaluator stays unedited (program-evolution contract);
these numbers are the recommended reference if a new evaluator version is ever locked.
