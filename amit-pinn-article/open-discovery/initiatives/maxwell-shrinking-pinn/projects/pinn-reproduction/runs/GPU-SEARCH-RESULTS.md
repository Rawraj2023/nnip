# GPU candidate search — winner confirmation (candidate-005)

Environment: NVIDIA GeForce GTX 1080 Ti (sm_61), driver 580.178.04, CUDA 12.6 wheel,
torch 2.14.1+cu126, Python 3.12 (`confirm-005/env.txt`, `confirm-005/nvidia-smi.txt`).
Winner rebuilt from `baseline-lock-gpu/baseline/` + `candidates/candidate-005/patch.diff`;
sha256 `cae54498…0661`, byte-identical to the evaluated candidate.
Timeline 17:49–18:44 UTC (`confirm-005/timeline.txt`); machine otherwise idle except light git
work during the first ~6 min of the baseline run.

## 1. Paired full-mode runs (unmodified evaluator, `--oracle --fold --stability`)

| stage | baseline (s) | candidate-005 (s) | speed-up |
|---|---|---|---|
| Table 1/2 (both branches, incl. pilots) | 392.8 | 158.1 | 2.48x |
| fold / S_c | 396.4 | 161.9 | 2.45x |
| stability (2 branch solves + 2 eigen) | 600.7 | 191.7 | 3.13x |
| **total** | **1389.9** | **511.6** | **2.72x** |

Algorithmic (same-GPU) improvement: **2.72x**. Hardware effect is separate and NOT summed:
Windows-CPU smoke lock 126.8 s vs GPU smoke lock 14.8 s (different machines; smoke only).

Accuracy at the search point is unchanged within noise:

| metric | baseline | 005 | reference |
|---|---|---|---|
| upper Cf (err vs paper) | 1.935161 (0.004 %) | 1.934442 (0.041 %) | paper 1.93523180 |
| lower Cf (err vs paper) | 0.755627 (0.42 %) | 0.755643 (0.43 %) | paper 0.75244081; eta_inf->inf oracle 0.753702 |
| upper / lower Nu err vs paper | 0.24 % / 0.38 % | 0.27 % / 0.39 % | oracle itself is 0.20 % / 0.29 % from paper |
| S_c | 1.84910 (0.54 %) | 1.84947 (0.56 %) | paper 1.8391 |
| alpha1 lower | -0.48820 | -0.48808 | independent -0.4938 (paper -0.0377) |
| alpha1 upper (sign only meaningful) | 0.958 | 1.051 | independent leading 0.7358 (paper 0.881) |

Pre-registered rule: Table 1/2 + S_c + both alpha1 signs pass for both. Evaluator `valid=False`
for both solely from the Table 3 magnitude gate.

## 2. Fresh-parameter check (`confirm_fresh.py`, seeds 1–3; search used seed 0 and rows S=1.95 / S=2.0 only)

Errors vs the independent `scipy.solve_bvp` oracle at eta_inf=40 (all oracle solves converged):

| row | upper Cf | upper Nu | lower Cf | lower Nu |
|---|---|---|---|---|
| S=2.10, K=0.10, beta=0.10 | 0.05–0.10 % | ≤0.08 % | **0.04 / 0.72 / 1.00 %** | ≤0.22 % |
| S=1.95, K=0.13, beta=0.10 | 0.03–0.21 % | ≤0.13 % | 0.40–0.49 % | ≤0.34 % |
| S=1.95, K=0.10, beta=0.15 | 0.03–0.15 % | ≤0.11 % | 0.04–0.67 % | ≤0.07 % |

Against the paper's own Table 1 Cf: 16/18 within 1 %; the two exceptions are the lower branch at
S=2.1 (seeds 2, 3: 1.51 %, 1.79 %), where the paper value itself is 0.78 % below the converged
oracle (eta_inf=10 truncation; truncation error grows as the lower branch decays more slowly).

## Verdict
- **Speed: confirmed.** 2.72x end-to-end on the same idle GPU, accuracy at the search point unchanged.
- **Upper branch, Nu on both branches: robust** on unseen rows and seeds (≤0.34 %).
- **Lower-branch Cf: seed-sensitive**, up to ~1.0 % vs the converged oracle on unseen rows. Whether
  this spread is inherent to the baseline architecture or increased by 005 (e.g. the adaptive LBFGS
  stop) is **untested** — the standard plan did not run the baseline through `confirm_fresh.py`.
  That comparison (~60 min GPU) is the next step before calling 005 accuracy-equivalent off the search point.
- **Table 3 does not reproduce** (baseline and 005 alike); independent solves of the paper's Eq. 31
  give lower alpha1 ≈ -0.49, upper ≈ 0.74 at S=2.0. The bundled `oracle/ref_eig_cheb.py` applies its
  boundary rows at swapped ends (see `../diagnostics/`).

## Update (session 3, 2026-11-06): open items closed
- Baseline vs 005 on `confirm_fresh.py` (same RTX 2060 SUPER, seeds 1-3): **equivalent** under a
  pre-registered rule; lower-branch Cf seed spread is inherent to the baseline.
  `fresh-paired-2060s/ANALYSIS.md`.
- Correct Table 3 oracle: `../oracle-v2/README.md` (lower alpha1 -0.4926 at S=2.0, converged).
