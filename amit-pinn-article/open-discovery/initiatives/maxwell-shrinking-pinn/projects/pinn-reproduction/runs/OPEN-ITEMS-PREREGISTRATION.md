# Open items after the GPU search — pre-registered rules (written 2026-11-06, before any result)

Session 3. Machine: NVIDIA GeForce RTX 2060 SUPER (sm_75), driver 615.71.09; rebuilt venv
/workspace/.venv-pinn from `requirements-gpu-cu126.txt` (torch 2.14.1+cu126, scipy 1.18.1 —
identical to session 2). Different GPU from session 2 (GTX 1080 Ti): no timing from this
session is compared with session-2 timings.

Programs (sha256 verified before running):
- baseline `baseline-lock-gpu/baseline/pinn_maxwell_shrinking.py` 8f145d68…b180
- candidate-005 `runs/confirm-005/pinn_maxwell_shrinking.py` cae54498…0661

## Item A — does candidate-005 change lower-branch seed sensitivity?

Run `confirm_fresh.py` (unmodified) on BOTH programs, seeds 1 2 3, serially on the same idle GPU
(baseline first). 9 (row, seed) points per program.

Primary metric: lower-branch `Cf_err_oracle` (vs scipy.solve_bvp, eta_inf=40).
Let m_B, m_5 = mean over the 9 points; x_B, x_5 = max over the 9 points.
- **equivalent**: |m_5 - m_B| <= 0.20 percentage points AND x_5 <= x_B + 0.30 pp
- **005 worse**: m_5 > m_B + 0.20 pp, or x_5 > x_B + 0.30 pp
- **005 better**: m_5 < m_B - 0.20 pp (and not worse)
Secondary (same thresholds, reported only): upper Cf, upper Nu, lower Nu.
With n=9 per arm this detects only gross differences; a result of "equivalent" means "no gross
difference detected", not proof of equality. Seeds are shared but the two programs consume the
RNG differently, so points are not truly paired.
Informative only: 005 here vs 005 in session 2 (same seeds) = cross-GPU reproducibility; and the
fresh-row wall-clock ratio baseline/005 on this GPU.

## Item B — a correct Table 3 (stability) oracle

Hypothesis to test (from the session-2 far-field addendum): on the semi-infinite domain the
disturbance problem has continuous spectrum for alpha > K; finite-domain solvers then produce
roots that accumulate as L grows, and the lowest such root tends to the onset of oscillatory
far-field decay, alpha* = K + f_inf^2 / (4 (1 - beta f_inf^2)), not to a discrete eigenvalue.

Oracle deliverable: a new file `oracle-v2/table3_oracle.py` (the bundled `oracle/` stays
byte-identical) that, for Table 3's rows (beta=0.1, K=0.1, A*=0.1, Pr=2, S=2.0..2.4):
1. lower branch: discrete alpha1 by two independent methods (shooting; corrected Chebyshev),
   with a domain-length study. Call it **converged** only if both methods agree to 0.5 % and the
   value changes < 0.5 % between the two largest domains.
2. upper branch: report f_inf, alpha*, and the lowest finite-L root vs L. The hypothesis is
   **supported** if, at every S, the lowest finite-L root decreases toward alpha* with L and
   lies above alpha*; **refuted** if a root stays converged (< 0.5 % change across the two
   largest domains) while clearly above alpha* (> 2 %), i.e. a genuine discrete eigenvalue.
3. Bug attribution (informative): run the bundled `oracle/ref_eig_cheb.py` placement (swapped
   BC rows) and report whether it reproduces any Table 3 value to 1 %. This does not establish
   how the paper's numbers were produced.
The PINN's stability stage is never altered or re-tuned against this oracle.
