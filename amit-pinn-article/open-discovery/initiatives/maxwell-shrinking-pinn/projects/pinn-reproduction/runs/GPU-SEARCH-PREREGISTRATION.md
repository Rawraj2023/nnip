# GPU candidate search — pre-registered acceptance rule (written before any candidate result)

Date: 2026-11-05. Hardware: NVIDIA GeForce GTX 1080 Ti (sm_61), driver 580.178.04,
torch 2.14.1+cu126 (separate venv /workspace/.venv-pinn; the image's torch 2.11+cu128 has no sm_61 kernels).

## Reference (unmodified baseline, unmodified evaluator, same GPU)
`runs/full_confirmation_gpu.json` (`--mode full --oracle --fold --stability`): total 1451.8 s
(table1/2 404.5 s, fold 456.0 s, stability 591.2 s). Table 1/2 + S_c gates PASS; Table 3 alpha1
magnitude gates FAIL (upper 0.958 vs 0.881, lower -0.488 vs -0.0377); sign checks pass.

## Why the alpha1 magnitude gate is not used for acceptance
Two independent solves of the paper's own Eq. 31 (verified term-by-term against the PDF, p.10)
agree with each other and not with Table 3:
- `diagnostics/eig_shooting.py` (IVP shooting, g''(0)=1, g'(edge)=0): S=2.0 upper 0.7358,
  lower -0.4938 (edge 10); lower stable to 3 digits for edge 6..12.
- `diagnostics/eig_cheb_fixed.py` (bundled Chebyshev oracle with its boundary rows moved to the
  correct ends — the bundled `oracle/ref_eig_cheb.py` applies the wall BCs at eta=L and vice versa):
  identical roots at L=10; lower -0.4926 at L=16.
No correct solver can satisfy |alpha_lower + 0.0377| < 1 %. The evaluator is NOT edited
(program-evolution contract); the gate is simply reported, not used to accept/reject.

## Acceptance rule for every candidate
1. smoke gate valid;
2. full mode: no Table 1/2 failure (tol 1 %), no S_c failure (tol 3 %);
3. stability: both sign checks pass (upper > 0, lower < 0); alpha1 reported vs independent
   values (upper 0.7358, lower -0.4938 at S=2.0) as information;
4. speed claim = same-GPU total objective (table1/2 + fold + stability seconds) vs 1451.8 s;
5. a winner must be reconstructed from the baseline + recorded patch and re-confirmed on fresh
   parameter points (`confirm_fresh.py`, Table 1 rows not used in search) over several seeds.

## Addendum (before any candidate result): upper-branch alpha1 is not identifiable by this PINN
Far field: g' ~ exp(-mu eta) with (1 - b f_inf^2) mu^2 - f_inf mu + (alpha - K) = 0. For alpha > K
both roots have Re mu > 0, so g'(inf)=0 holds for a continuum of alpha on the semi-infinite domain;
the PINN trial function imposes g'(inf)=0 exactly, so the upper-branch alpha1 (~0.74 > K=0.1) is not
pinned (CPU test, candidate-001 math: inits 0.0..1.0 all -> 0.66..0.69; GPU baseline -> 0.958).
The lower branch (alpha ~ -0.49 < K) has one growing mode -> discrete eigenvalue -> PINN matches the
independent value to ~1 %. Only the upper SIGN is a meaningful PINN output; no candidate targets the
upper magnitude.
