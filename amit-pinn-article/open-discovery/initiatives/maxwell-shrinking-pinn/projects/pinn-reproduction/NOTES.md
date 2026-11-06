# Project: PINN reproduction of the Maxwell shrinking-sheet paper

Owner: single explorer (this session). Workflow: `paper-implementer` for the
reproduction claim, `evolve-program` / `program-evolution/` for the locked
optimization baseline. See `../../BRIEF.md` for the full request and
provenance hashes.

## Layout

```
pinn-reproduction/
├── NOTES.md                 (this file)
├── paper/
│   ├── pagination_FINMEC_100352-1.pdf   (copy of the source PDF)
│   └── paper-summary.md     (transcribed equations, BCs, Tables 1-3, gated vs. reference-only)
├── baseline/
│   └── pinn_maxwell_shrinking.py   (copy of the candidate program to evolve; unmodified)
├── oracle/
│   ├── reference_bvp.py     (independent scipy.solve_bvp ground truth; unmodified)
│   └── ref_eig_cheb.py      (independent Chebyshev-spectral eigenvalue ground truth; unmodified)
├── evaluator.py              (candidate-proof evaluator; implements the program-evolution contract)
├── baseline-lock/            (output of lock_baseline.py — immutable once created)
├── candidates/                (empty; for future evolve-program candidate-NNN folders)
└── runs/                      (evaluator run logs/outputs kept as evidence)
```

## What was done this session

1. Read the full paper (12 pages) and all three scripts in detail.
2. Transcribed the governing ODEs, BCs, solution method, and Tables 1-3 into
   `paper/paper-summary.md`, marking exactly which numbers are used as
   automated evaluator gates (cross-checked against the paper co-author's own
   inline comments in `pinn_maxwell_shrinking.py`) versus transcribed for
   reference only.
3. Copied the three scripts and the PDF read-only into this project (sha256
   hashes recorded in `../../BRIEF.md`); the originals outside
   `open-discovery/` were not modified.
4. Wrote `evaluator.py`: a standalone, import-based (not print-parsing)
   evaluator that
   - loads a candidate PINN module via `importlib`,
   - checks it exposes the required contract
     (`MLP, train, fields, wall_values, solve_branch, trace_fold, smallest_eigen`),
   - in **smoke mode** (default, cheap): monkeypatches `train` to a hard
     120-Adam/40-LBFGS budget and checks the dual-solution pipeline runs and
     returns physically ordered, finite values (`Cf_upper > Cf_lower > 0`,
     both `Nu > 0`). This is a pipeline sanity gate, not an accuracy claim.
   - in **full mode** (expensive): runs the candidate's own unmodified
     `solve_branch` at the paper's Table 1/2 baseline row
     (`beta=0.1, K=0.1, S=1.95, Pr=2, A*=0.1, B*=0.1`) and gates on relative
     error to the published `Cf`/`Nu`. Optional `--fold` gates the critical
     suction parameter `S_c` against `1.8391`; optional `--stability` gates
     the Table 3 leading eigenvalues (including a hard sign check: upper
     branch stable, lower branch unstable) against `0.8810518` /
     `-0.0376726`; optional `--oracle` cross-checks the paper's own numbers
     against the independent `scipy.solve_bvp` oracle (informative only,
     never gates the candidate).
5. Validated the evaluator (smoke mode) directly against the baseline:
   **valid, SUPPORTED**, Cf_upper≈1.913, Cf_lower≈0.772, Nu_upper≈1.854,
   Nu_lower≈1.203 after a 120+40-iteration budget — correctly ordered,
   finite, ~126.8s wall-clock on this CPU-only machine (`torch 2.13.0+cpu`,
   no CUDA). These numbers are **not** expected to match the paper (the
   smoke budget is far below what the architecture needs to converge); they
   only confirm the pipeline and evaluator wiring are correct.
6. Locked the baseline:
   `program-evolution/scripts/lock_baseline.py --baseline baseline/pinn_maxwell_shrinking.py --evaluator evaluator.py --goal "..." --output baseline-lock --timeout-seconds 280`
   → **`baseline-lock/lock.json`: `"status": "locked"`, `"error": null`**.
   Baseline sha256 `40d7e8611b8d417153eea254e7531eb5017b1502edd2a7fd5ed174234ae8c28`.
   This baseline + evaluator pair is now ready for an `evolve-program`
   candidate search (e.g. faster CPU convergence, fewer total iterations to
   reach paper-level accuracy, a better far-field map/architecture) whenever
   the human requests one — see `program-evolution/prompts/propose-candidate.md`.
7. Launched the expensive **full-mode, paper-accuracy confirmation** run in
   the background (`python evaluator.py baseline/pinn_maxwell_shrinking.py
   --mode full --oracle --tol 0.01`), writing to
   `runs/full_table12_confirmation.json` /
   `runs/full_table12_confirmation.stderr.log`. This is the run whose result
   actually supports or refutes the "faithful reproduction" claim at
   publication-level accuracy (paper's own script comments target the exact
   8-significant-figure Table 1/2 values). Expected runtime: tens of minutes
   to roughly an hour+ on this CPU (no GPU detected); the smoke run's
   160-total-iteration budget alone took ~127s, and full mode runs
   4000+1500 (momentum) + an auto-scale pilot of 2500+200, then 2000+1000
   (energy) iterations **per branch**, times two branches.

## Current status: **in progress**

- Smoke gate: **passed** (pipeline verified correct).
- Baseline lock: **done** (`baseline-lock/lock.json` status `locked`).
- Full paper-accuracy confirmation: **running in background**; check
  `runs/full_table12_confirmation.json` for the result (one JSON object per
  the evaluator contract) once the process exits. If the file is empty or
  the process is still listed as running, it has not finished yet — do not
  report pass/fail before it completes.
- Stability (Table 3) and fold-tracing (`S_c`) full-mode confirmations have
  **not** been run yet (they are additional ~tens-of-minutes-each runs via
  `--stability` / `--fold`); available on request, not run automatically
  given the cumulative CPU time involved.

## How to reproduce / continue this work

```bash
# cheap pipeline sanity check (~2 min on CPU)
python evaluator.py baseline/pinn_maxwell_shrinking.py

# expensive paper-accuracy confirmation (Table 1/2 baseline row; ~tens of minutes+)
python evaluator.py baseline/pinn_maxwell_shrinking.py --mode full --oracle

# also confirm the critical suction S_c and Table 3 stability eigenvalues
python evaluator.py baseline/pinn_maxwell_shrinking.py --mode full --fold --stability

# (re-)lock the baseline after any intentional change to baseline/ or evaluator.py
python ../../../../program-evolution/scripts/lock_baseline.py \
  --baseline baseline/pinn_maxwell_shrinking.py --evaluator evaluator.py \
  --goal "<state the goal>" --output baseline-lock-v2 --timeout-seconds 280
```

To start an `evolve-program` candidate search against the locked baseline
(e.g. "make the momentum-stage LBFGS converge in fewer iterations while
keeping `--mode full` Cf/Nu within 1% of the paper"), invoke the
`$evolve-program` workflow and point it at this project's `baseline-lock/`
and `evaluator.py`; follow `program-evolution/prompts/propose-candidate.md`
for candidate workers, and `docs/CANDIDATE-LIFECYCLE.md` for candidate
bookkeeping (one folder per hypothesis under `candidates/`, never overwrite
an evaluated candidate).

## Known limitations / things a candidate search must not silently break

- The dual-solution existence boundary (`S_c`) and the stable/unstable sign
  of the two branches are the paper's central physical claims; a "faster"
  candidate that converges to a single branch, swaps branch labels, or flips
  an eigenvalue sign has not improved anything even if `elapsed_seconds`
  drops — `evaluator.py --stability` enforces the sign check for exactly
  this reason.
- `L_MAP` auto-scaling (`solve_branch(auto_scale=True)`) is what gets the
  lower branch to the paper's stated ~0.1% accuracy floor instead of a
  ~1-2% floor (see the baseline script's own docstring); a candidate that
  removes or weakens it may look fine on the upper branch and silently fail
  the lower-branch gate.
- This machine is CPU-only (`torch 2.13.0+cpu`); no GPU speedup claim can be
  made here. A candidate's `elapsed_seconds` objective is only comparable to
  other runs on the same machine/environment, per
  `program-evolution/README.md`'s hash-and-environment-tracking guidance.

## GPU session (2026-11-05, GTX 1080 Ti) — supersedes the CPU "in progress" status above

- The earlier background CPU full-mode run never completed (`runs/full_table12_confirmation.json` is empty).
- GPU lock: `baseline-lock-gpu/` (smoke 14.8 s, `locked`). Full GPU reference:
  `runs/full_confirmation_gpu.json` — Table 1/2 and S_c reproduce; **Table 3 does not**.
- Acceptance rule fixed before any candidate result: `runs/GPU-SEARCH-PREREGISTRATION.md`
  (why the alpha1 magnitude gate cannot be met by any correct solver; upper alpha1 not identifiable).
- Independent eigen diagnostics: `diagnostics/eig_shooting.py`, `diagnostics/eig_cheb_fixed.py`.
- Candidates 001–005: `candidates/LINEAGE.md`. Winner candidate-005 confirmed **2.72x faster**
  same-GPU with unchanged search-point accuracy: `runs/GPU-SEARCH-RESULTS.md`.
- Open: baseline vs 005 on `confirm_fresh.py` (lower-branch seed spread); a correct Table 3 oracle.
- Environment: torch 2.14.1+cu126 venv (image torch cu128 lacks sm_61);
  `../../../../../requirements-gpu-cu126.txt`.

## Session 3 (2026-11-06, RTX 2060 SUPER) — closes both open items above

Pre-registration (before any result): `runs/OPEN-ITEMS-PREREGISTRATION.md`. Venv rebuilt from the
cu126 requirements (torch 2.14.1+cu126, identical to session 2). Different GPU: no timing here is
compared with session-2 timings.

- **Item A — lower-branch seed spread: closed, equivalent.** Baseline and 005 through
  `confirm_fresh.py` (seeds 1-3, same GPU): lower Cf error vs converged oracle mean/max
  0.46/0.96 % (baseline) vs 0.39/0.99 % (005); all secondaries equivalent. The spread is inherent
  to the baseline, not introduced by 005. `runs/fresh-paired-2060s/ANALYSIS.md`.
  Candidate-005 is now accuracy-equivalent to the baseline both at the search point and on
  unseen rows/seeds (to the resolution of n=9).
- **Item B — Table 3 oracle: built, `oracle-v2/` (README has the full table).** Lower-branch alpha1
  converged by three methods: -0.4926, -0.6114, -0.7080, -0.7941, -0.8767 (S=2.0..2.4) vs paper
  -0.0377..-0.3115 — sign agrees (unstable), magnitudes do not reproduce. PINN lower alpha1 at
  S=2.0 (-0.488) is within 0.9 % of the oracle. Upper branch: the pre-registered "roots tend to
  alpha*" hypothesis was refuted at S=2.0/2.1 (converged 0.7348/1.0493, below alpha*); no reliable
  magnitude at S>=2.2; all roots > K > 0, so "stable" holds. The bundled `oracle/ref_eig_cheb.py`
  BC placement (rows at swapped ends) at L=12 lands within 0.7 % of all five paper upper values —
  consistent with, not proof of, the paper's upper column having come from that placement.
  Energy-disturbance eigenvalues (Eq. 32, new) are all positive; no sign change.
- Evaluator not edited. Its Table 3 magnitude gates remain unmeetable by a correct solver;
  `oracle-v2` values are the recommended reference if a new evaluator version is locked.

Status: **reproduction complete to the extent the paper allows** — Table 1/2 and S_c reproduce;
Table 3 signs reproduce, magnitudes do not (paper-side discrepancy, evidence above). Candidate-005
confirmed 2.72x faster (session 2) and accuracy-equivalent. No open items.
