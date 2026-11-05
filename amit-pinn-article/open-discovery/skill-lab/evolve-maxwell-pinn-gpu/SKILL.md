---
name: evolve-maxwell-pinn-gpu
description: Run a GPU-accelerated evolve-program candidate search to beat the locked CPU baseline for the Maxwell-fluid shrinking-sheet PINN (Pandey et al., Forces in Mechanics 100352) at initiatives/maxwell-shrinking-pinn/projects/pinn-reproduction/. Use when the user has connected this session to a CUDA GPU (cloud or local) and wants to search for a faster and/or more accurate version of pinn_maxwell_shrinking.py than the locked baseline, or wants to extend the work into novel parametric/meta-PINN territory the paper itself does not cover.
---

# Evolve the Maxwell-fluid shrinking-sheet PINN on GPU

Prototype, repository-local skill. Invoke by path (`skill-lab/evolve-maxwell-pinn-gpu/SKILL.md`)
until promoted. It composes the repository's own `evolve-program` skill and
`program-evolution/` harness with work specific to this one initiative — read
`.agents/skills/evolve-program/SKILL.md` and `program-evolution/README.md`
first if this session has not already internalized them.

## 0. Confirm the GPU is real before claiming any speedup

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

If `torch.cuda.is_available()` is `False`, or the installed wheel is a
`+cpu` build, stop and tell the human — this skill assumes genuine CUDA
access. `pinn_maxwell_shrinking.py` already auto-selects
`DEV = "cuda" if torch.cuda.is_available() else "cpu"`; no code change is
needed to use the GPU, only a working CUDA-enabled `torch` install.

## 1. Do not compare wall-clock across hardware

The existing lock at
`initiatives/maxwell-shrinking-pinn/projects/pinn-reproduction/baseline-lock/lock.json`
was measured **on CPU** (smoke mode). Per `program-evolution/README.md`,
"always retain benchmark and evaluator hashes: an apparent model improvement
after the benchmark changed is not a comparable result" — the same rule
applies to the *environment*. Moving to a GPU will produce a large wall-clock
win on its own; that is a hardware effect, not an evolved program, and must
not be reported as "beating the baseline."

Before searching for candidates, establish a fresh, same-hardware reference:

```bash
cd initiatives/maxwell-shrinking-pinn/projects/pinn-reproduction
python ../../../../program-evolution/scripts/lock_baseline.py \
  --baseline baseline/pinn_maxwell_shrinking.py \
  --evaluator evaluator.py \
  --goal "GPU reference measurement: unmodified baseline, full-mode Table 1/2 accuracy + wall-clock on <GPU model>" \
  --output baseline-lock-gpu \
  --timeout-seconds 1800
```

The default evaluator invocation inside `lock_baseline.py` runs `--mode smoke`
(cheap, ~2 min). Immediately after, also run and keep the **full**-mode
reference separately (this is the real number candidates must beat):

```bash
python evaluator.py baseline/pinn_maxwell_shrinking.py \
  --mode full --oracle --fold --stability \
  > runs/full_confirmation_gpu.json
```

Record the GPU model/driver (`nvidia-smi` output) next to
`baseline-lock-gpu/lock.json` — the lock schema does not capture GPU identity,
only Python version and OS platform.

## 2. Candidate search

Follow `.agents/skills/evolve-program/SKILL.md` exactly: one candidate folder
per hypothesis under `projects/pinn-reproduction/candidates/candidate-NNN/`,
each a full copy of `baseline/pinn_maxwell_shrinking.py` with one change,
evaluated with the unmodified `evaluator.py` and never allowed to edit it.
Gate cheaply first (`--mode smoke`, default) and only spend full-mode budget
on candidates that already pass smoke. A candidate is only a real win if
`--mode full` (ideally with `--fold --stability` too) still passes within
`--tol`, compared against `baseline-lock-gpu`'s `runs/full_confirmation_gpu.json`,
on the same GPU.

See `references/candidate-mechanisms.md` for the concrete list of mechanisms
to try, grouped by what kind of win they target. Treat it as a menu, not a
checklist to exhaust in order — prioritize by expected mechanism strength and
assign independent candidates to different mechanisms rather than repeatedly
editing one winner (`AGENTS.md`'s program-evolution guidance).

## 3. Novel-work track is a different project, not a candidate

Section D of `references/candidate-mechanisms.md` (parametric/meta-PINN,
continuous fold-tracking, parametric stability) is genuinely new science
beyond the paper, not a behavior-preserving optimization of the existing
baseline — it changes what is being claimed, not just how fast the existing
claim is produced. If the human wants this pursued, create a **new project**
(e.g. `initiatives/maxwell-shrinking-pinn/projects/parametric-meta-pinn/`)
with its own `evaluator.py` (the existing one only validates one fixed
parameter point; a parametric claim needs held-out (S, β, K) grid points, not
reused training points) rather than adding it under `candidates/`.

## 4. Finish like evolve-program

Reconstruct any claimed winner from the locked GPU baseline, apply only its
recorded patch, rerun `--mode full --oracle --fold --stability` on fresh
confirmation parameters not reused from search, and report: the algorithmic
improvement (same-GPU, candidate vs. `baseline-lock-gpu`), the hardware
improvement (CPU lock vs. GPU lock, labeled separately and never summed with
the algorithmic one), accuracy gates passed/failed, and environment
(GPU model, CUDA version, `torch` version). Local GPU use the human already
connected is covered by their stated authority; any *additional* cloud spend
or a different/bigger instance still needs the human's explicit go-ahead
per `AGENTS.md`.
