# Maxwell-fluid shrinking-sheet PINN: reproduction + optimization baseline

## Original request

The human supplied, in one chat session:

1. A paper PDF: *"Insight of thermal transport in Maxwell fluid flow on a
   porous contracting flat surface inside a porous material with non-uniform
   heat source/sink: Multiple solutions and stability treatment"*, A.K. Pandey,
   P. Goswami, S. Rajput, K. Bhattacharyya, A.J. Chamkha, D. Yadav, *Forces in
   Mechanics* 100352 (uncorrected proof), DOI
   10.1016/j.finmec.2026.100352.
2. Three Python files implementing and validating a Physics-Informed Neural
   Network (PINN) solution of the paper's governing similarity ODEs:
   - `pinn_maxwell_shrinking.py` — the PINN (three stages: branch solve, fold
     tracing, stability eigenvalue).
   - `reference_bvp.py` — an independent `scipy.solve_bvp` numerical solver
     for the same BVP, used as ground truth.
   - `ref_eig_cheb.py` — an independent Chebyshev-spectral solve of the
     linearized stability eigenvalue problem, a second ground truth for the
     stability claim.
3. The instruction, after review: *"go through the folder open-discovery and
   the files in it. make all necessary changes in the file there, baseline,
   skills etc. for the work related to the article you have gone through just
   now."*

Interpreted outcome: use Open Discovery's existing harness (paper-implementer
pattern for the reproduction claim, program-evolution for a lockable
optimization baseline) to give this PINN reproduction a durable, evidence-
preserving home, with the PINN set up as a **locked program-evolution
baseline** so future candidate search (faster training, better accuracy,
different architectures) can be run against an external, non-gameable
evaluator.

## Source material — not modified

The paper PDF and all three scripts originated outside this repository, at
`C:\Users\LENOVO\Downloads\amit-pinn-article\` (the parent of this
`open-discovery` checkout). Per `AGENTS.md`, external work is not modified;
this initiative copies the material read-only into
`projects/pinn-reproduction/` for a self-contained, reproducible record.
SHA-256 of the originals at copy time:

| file | sha256 |
|---|---|
| `pinn_maxwell_shrinking.py` | `8f145d68377b7be4bbbc73fc7d643d745ef063e4156be2c7fea089016176b1` |
| `reference_bvp.py` | `a1e042fd16bdccd19cdac981992f87878cf7e40eafba497303b4ee93e8f203` |
| `ref_eig_cheb.py` | `081318bf7cbb5ab4a74f97dd3e9d7027cac31f347aa5cf02cb81ba320e260b1` |
| `pagination_FINMEC_100352-1.pdf` | `005acd8e27c6533be8e9e4e996e7567bd9eb4d7671f31472bbc6d7d3fe4f5e3` |

The copies under `projects/pinn-reproduction/baseline/` and
`projects/pinn-reproduction/oracle/` are byte-identical to these originals and
must stay that way (the baseline copy becomes immutable once locked; see
`baseline-lock/lock.json` after locking).

## Desired outcome

1. A durable, provenance-preserving record of the paper's governing equations,
   boundary conditions, and the exact numeric claims in Tables 1–3 that the
   code is meant to reproduce.
2. An external, candidate-proof evaluator that checks a PINN script's outputs
   against (a) the paper's own published numbers and (b) the two independent
   numerical oracles — not just against itself.
3. A locked, hashed baseline (`pinn_maxwell_shrinking.py` + evaluator) via
   `program-evolution/scripts/lock_baseline.py`, ready for an `evolve-program`
   candidate search (e.g. faster CPU training, fewer iterations to reach
   paper-level accuracy, better far-field architecture) whenever the human
   requests one.

## Constraints

- Local, zero-cost, non-destructive, CPU-only (no CUDA detected in this
  environment: `torch 2.13.0+cpu`). No GPU/cloud spend authorized.
- Do not weaken or edit the evaluator/oracle to make a candidate pass; the
  evaluator is immutable once a candidate search begins (program-evolution
  contract).
- Do not claim "reproduction" from a reduced-iteration smoke run; only a
  `--mode full` evaluator run against the paper's exact baseline parameter set
  supports that claim.
- This is a reproduction + harness-setup initiative, not a new scientific
  claim about the physical system; no novel result is asserted here.

## Status

Single project, `projects/pinn-reproduction/`. See its `NOTES.md` for
methodology, evidence, and current state.
