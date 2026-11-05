# Candidate mechanisms for beating the Maxwell-PINN baseline

Concrete, code-specific hypotheses for `evolve-program` candidates against
`initiatives/maxwell-shrinking-pinn/projects/pinn-reproduction/baseline/pinn_maxwell_shrinking.py`.
Each item names the mechanism, why a GPU specifically unlocks or rewards it,
and what the evaluator already protects so a candidate cannot cheat its way
to a lower `elapsed_seconds`. Line numbers refer to the locked baseline copy.

## A. Same-hardware throughput (reduce elapsed_seconds, keep accuracy)

- **A1. Batch the two branches into one training pass.** `solve_branch`
  (136–154) trains the upper and lower branch fully sequentially, each with
  its own `MLP` pair. On GPU, stack both branches' parameters along a leading
  batch dimension (`torch.func.vmap`/`functorch`, or simply a wider shared
  trunk with a 2-way branch-selector input) and train both in one optimizer
  loop. CPU has little to gain here (serial anyway); GPU occupancy is
  currently wasted training one tiny branch network at a time.
- **A2. Cut the pilot-run cost in `auto_scale`.** `solve_branch` (144–153)
  runs a full 2500-Adam/200-LBFGS pilot network just to read off `f(∞)` and
  pick `L_MAP`. Replace or shrink the pilot: either a much shorter/coarser
  fit (fewer collocation points, fewer iterations — only the asymptotic
  value matters), or compute `f_inf` from a cheap fixed-point iteration on
  `far_field_decay` (128–133) plus the known `S`, `K`, `β`, bypassing the
  neural pilot entirely for the common case.
- **A3. Early-exit LBFGS instead of a fixed `max_iter`.** `train` (107–124)
  always runs LBFGS for the full `lbfgs_iters` budget. Instrument how many
  iterations actually reduce loss before `tolerance_grad`/`tolerance_change`
  would have stopped it anyway, and cap accordingly — a candidate that
  reaches the same loss floor in fewer iterations is a legitimate win.
- **A4. Rebalance Adam vs. LBFGS.** GPU makes many cheap Adam steps closer to
  "free" relative to LBFGS's multiple closure evaluations per step (strong
  Wolfe line search). Try Adam-only with cosine/longer annealing, or a
  different Adam/LBFGS iteration split, and measure wall-clock at equal
  accuracy rather than assuming the current 4000/1500 split is GPU-optimal.
- **A5. Vectorize `trace_fold`'s continuation loop.** `trace_fold` (201–226)
  retrains sequentially over 8 prescribed `f''(0)` values via a Python
  `for`-loop with warm starts. A batched network with an extra
  continuation-index input could fit all 8 points' collocation + loss terms
  in one GPU batch instead of 8 sequential trainings.
- **A6. Grow the collocation batch, not just iteration count.** `n=300` in
  `collocation()` (56–67) is a CPU-sized budget. GPU headroom may let a much
  larger `n` converge in fewer *optimizer steps* for the same or better
  accuracy — trade point count for iteration count and measure wall-clock,
  not step count.

## B. Accuracy / robustness (tighten the gate, don't just pass it)

- **B1. Closed-form `L_MAP` instead of a neural pilot.** Combine A2's removal
  of the pilot network with `far_field_decay` (128–133) applied directly;
  this both speeds up *and* removes a source of pilot-run noise that
  currently feeds into the production run's `L_MAP` choice.
- **B2. More reliable branch selection.** The `f2_anchor` warm-start trick
  (157–179) is a single rough target; a candidate that converges to the
  wrong branch can still pass the smoke gate's `Cf_upper > Cf_lower > 0`
  check only by luck, not by correct branch identity. A short continuation
  sweep (trace_fold-style) seeding `solve_branch`, or an explicit check that
  the recovered `f''(0)` lands near the intended target, closes this gap.
  Full-mode `--stability`'s sign check is the real guard against this —
  prioritize candidates against that gate, not just the smoke gate.
- **B3. Stress-test near the `(1 - β f²)` singularity.** Both the PINN
  residual (`residuals`, 89–96) and the oracle's explicit `fppp` formula
  (`oracle/reference_bvp.py:17`) divide by `(1 - β f²)`-equivalent terms.
  Neither script tests `β` beyond the paper's range (up to 0.2). A candidate
  that improves numerical stability for larger `β` (where this factor can
  approach zero) is real robustness work, validated by extending the
  evaluator's parameter grid — not a change to `evaluator.py` itself from
  inside a candidate folder.
- **B4. Target the lower branch specifically.** The baseline script's own
  docstring (136–143) notes a ~1–2% accuracy floor on the lower branch
  without `L_MAP` matching, dropping to ~0.1% with it. Report per-branch
  relative error, not just a combined pass/fail, and prioritize candidates
  that close the gap on whichever branch the current full-mode run (once it
  finishes on CPU, or after the first GPU reference run) shows is further
  from the paper's published value.

## C. GPU-native restructuring (same public contract, different internals)

A candidate may change anything *inside* `solve_branch`, `trace_fold`,
`smallest_eigen`, `train`, `fields`, `residuals` as long as the public
function signatures and return shapes stay exactly what `evaluator.py`
expects (see its `REQUIRED_CONTRACT` and the dict keys read from
`solve_branch`'s second return value). A1, A5, and A6 above are the concrete
instances of this; no evaluator change is needed to validate them.

## D. Novel-work track — new project, not a `candidates/` entry

These are genuinely new contributions beyond the paper and the existing
script (both re-solve one ODE system per parameter point); pursue only as a
new project per `SKILL.md` section 3, with its own evaluator.

- **D1. Parametric / meta-PINN.** Train one network
  `f(η; S, β, K)`, `θ(η; S, β, K, Pr, A*, B*)` taking the physical parameters
  as extra inputs, so one GPU training run reproduces the entire Table 1/2
  sweep (and extends past it, e.g. finer `β` resolution, `S` below 1.8391 to
  map the non-existence region) instead of one network per parameter tuple.
- **D2. Continuous fold-tracking.** Using D1's parametric network,
  differentiate with respect to `S` to locate `S_c(β, K)` as the point where
  the branch parametrization becomes singular, replacing the discrete
  `trace_fold` sweep over prescribed `f''(0)` values with an exact,
  continuous method.
- **D3. Parametric stability.** Extend `smallest_eigen` to be parametric in
  `S`, producing a continuous `α₁(S)` and directly locating `α₁ = 0` as an
  independent cross-check that the turning point (D2) and the
  stability-exchange point coincide — the paper asserts this via separate
  tables, never verifies it as one continuous claim.
- **D4. Ensemble uncertainty.** A small multi-seed ensemble per branch,
  GPU-cheap, reporting spread in `Cf`/`Nu` as a robustness companion to the
  point estimates — useful evidence for any eventual write-up, not a speed
  or accuracy claim against the existing evaluator.

Any D-track result needs held-out `(S, β, K)` combinations the parametric
network was *not* trained on, compared against the oracle
(`oracle/reference_bvp.py`, `oracle/ref_eig_cheb.py`) at those points — table
values memorized during training are not evidence of a working parametric
solution.
