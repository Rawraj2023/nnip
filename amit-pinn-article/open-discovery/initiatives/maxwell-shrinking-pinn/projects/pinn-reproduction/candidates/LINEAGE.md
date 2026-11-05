# Candidate lineage (GPU search, 2026-11-05)

| id | parent | change (see each folder's NOTE.md / patch.diff) | full total s | vs baseline 1451.8 s | gates (pre-registered rule) |
|---|---|---|---|---|---|
| 001 | baseline | Taylor-mode derivative jets instead of nested autograd | 823.5 | 1.76x | pass |
| 002 | baseline | adaptive LBFGS stop (50-iter chunks, <1 % gain -> stop) | 967.7 | 1.50x | pass |
| 003 | baseline | collocation xi_max 0.9995 -> 0.99995 | 1374.9 | none (same work; noise) | pass; CPU accuracy gain NOT reproduced in full mode |
| 004 | baseline | auto-scale pilot 2500/200 -> 800/100 | 1230.7 | ~1.18x, within run-to-run noise | pass |
| 005 | 001 + patches of 002, 003, 004 | combination | 503.5 | 2.88x | pass |

All single runs, same GTX 1080 Ti, unmodified evaluator. Evaluator `valid=False` on every row
(baseline included) only because of the Table 3 alpha1 magnitude gate — see
`../runs/GPU-SEARCH-PREREGISTRATION.md`. Single-run stage timings vary ~15 % (fold stage is
untouched by 003/004 yet ran 368-387 s vs 456 s). Winner confirmation: `../runs/GPU-SEARCH-RESULTS.md`.
