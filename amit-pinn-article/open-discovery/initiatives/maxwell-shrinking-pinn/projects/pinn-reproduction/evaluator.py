"""Evaluator for the Maxwell-fluid shrinking-sheet PINN.

Reproduces Pandey, Goswami, Rajput, Bhattacharyya, Chamkha & Yadav,
"Insight of thermal transport in Maxwell fluid flow on a porous contracting
flat surface inside a porous material with non-uniform heat source/sink:
Multiple solutions and stability treatment", Forces in Mechanics 100352.
See ../paper/paper-summary.md for the transcribed equations, boundary
conditions, and the exact table values used below.

Candidate contract (same as ../baseline/pinn_maxwell_shrinking.py): a .py
module exposing `MLP`, `train`, `fields`, `wall_values`, `solve_branch`,
`trace_fold`, `smallest_eigen` with the same signatures. A candidate may
change architecture, training schedule, or numerics inside those functions,
but not their external contract.

Ground truth, two independent sources (never edited by candidates):
  1. The paper's own published Table 1 / Table 2 / Table 3 numbers, hardcoded
     below from paper-summary.md.
  2. An independent oracle re-solve bundled in ../oracle/: `reference_bvp.py`
     (scipy.solve_bvp collocation) and `ref_eig_cheb.py` (Chebyshev-spectral
     eigenvalue solve), used as a cross-check that (1) is transcribed
     correctly and that the candidate agrees with a second numerical method,
     not only with the printed paper numbers.

Modes
-----
--mode smoke (default): drastically reduced training budget (monkeypatches
  the candidate's own `train`), only checks the pipeline runs end to end and
  returns physically sane, correctly ordered dual-solution values (upper
  Cf > lower Cf > 0, both Nu > 0). This is what `lock_baseline.py` runs and
  what should gate rapid candidate screening. It is NOT a paper-accuracy
  claim and must not be reported as a reproduction.

--mode full: runs the candidate's own solve_branch unmodified (its own
  architecture and iteration counts) at the paper's Table 1/2 baseline row
  (beta=0.1, K=0.1, S=1.95, Pr=2, A*=0.1, B*=0.1) and gates on relative error
  to the published Cf and Nu. This is the actual paper-reproduction claim.
  Expensive (minutes on CPU) -- run deliberately, not on every candidate.
  Add --fold for the critical-suction (S_c) claim and --stability for the
  Table 3 leading-eigenvalue claim; both run the candidate's own unmodified
  trace_fold / smallest_eigen and add their wall-clock time to the objective.

Contract: exits 0, prints exactly one JSON object to stdout, in the shape
documented by ../../../program-evolution/README.md.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
ORACLE_DIR = HERE / "oracle"

# ---------------------------------------------------------------- paper ground truth
# Baseline row, Table 1 (Cf Re^1/2) and Table 2 (Nu Re^-1/2 = -theta'(0)),
# paper page 7. beta=0.1, K=0.1, S=1.95, Pr=2, A*=0.1, B*=0.1.
BASELINE_PARAMS = dict(beta=0.1, K=0.1, S=1.95, Pr=2.0, A=0.1, B=0.1)
PAPER_TABLE_1_2 = {
    "upper": {"Cf": 1.93523180, "Nu": 3.35807824},
    "lower": {"Cf": 0.75244081, "Nu": 3.14427924},
}

# Critical suction parameter for beta=0.1 (Section 4.1 text / Fig. 2 caption).
PAPER_SC = 1.8391
FOLD_PARAMS = dict(beta=0.1, K=0.1, S=1.95, Pr=2.0, A=0.1, B=0.1)
FOLD_F2_VALUES = [2.4, 2.0, 1.6, 1.3, 1.1, 0.9, 0.7, 0.5]

# Table 3 stability base case, paper page 10: beta=0.1, K=0.1, A*=0.1, Pr=2,
# B*=0, S=2.0.
STABILITY_PARAMS = dict(beta=0.1, K=0.1, S=2.0, Pr=2.0, A=0.1, B=0.0)
PAPER_TABLE_3 = {"upper": 0.8810518, "lower": -0.0376726}

REQUIRED_CONTRACT = (
    "MLP", "train", "fields", "wall_values",
    "solve_branch", "trace_fold", "smallest_eigen",
)


# ---------------------------------------------------------------- loading
def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def load_oracle():
    """Load the bundled, never-edited oracle (scipy BVP + Chebyshev spectral)."""
    ref_bvp = load_module(ORACLE_DIR / "reference_bvp.py", "oracle_reference_bvp")
    # ref_eig_cheb.py does `from reference_bvp import solve`; make our copy
    # resolvable under that name regardless of sys.path/cwd.
    sys.modules["reference_bvp"] = ref_bvp
    eig_cheb = load_module(ORACLE_DIR / "ref_eig_cheb.py", "oracle_ref_eig_cheb")
    return ref_bvp, eig_cheb


def require_contract(candidate: ModuleType) -> list[str]:
    return [name for name in REQUIRED_CONTRACT if not hasattr(candidate, name)]


def rel_err(actual: float, expected: float) -> float:
    if expected == 0:
        return abs(actual)
    return abs(actual - expected) / abs(expected)


# ---------------------------------------------------------------- smoke gate
def fast_train_factory(adam_iters: int, lbfgs_iters: int):
    """Drop-in replacement for candidate.train() with a hard-capped budget,
    used only for the smoke gate. Ignores whatever adam_iters/lbfgs_iters the
    candidate's own call sites request, so every stage stays cheap."""

    def fast_train(net, loss_fn, extra_params=(), **kwargs):
        import torch

        params = list(net.parameters()) + list(extra_params)
        opt = torch.optim.Adam(params, lr=kwargs.get("lr", 2e-3))
        for it in range(adam_iters):
            opt.zero_grad()
            loss = loss_fn(it)
            loss.backward()
            opt.step()
        opt2 = torch.optim.LBFGS(
            params, lr=1.0, max_iter=lbfgs_iters,
            tolerance_grad=1e-10, tolerance_change=1e-12,
            history_size=20, line_search_fn="strong_wolfe",
        )

        def closure():
            opt2.zero_grad()
            loss = loss_fn(10**9)
            loss.backward()
            return loss

        opt2.step(closure)
        return loss_fn(10**9).item()

    return fast_train


def run_smoke(candidate: ModuleType, p: dict) -> dict:
    started = time.perf_counter()
    original_train = candidate.train
    candidate.train = fast_train_factory(adam_iters=120, lbfgs_iters=40)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, up = candidate.solve_branch(dict(p), "upper", auto_scale=False)
            _, lo = candidate.solve_branch(dict(p), "lower", auto_scale=False)
    finally:
        candidate.train = original_train
    elapsed = time.perf_counter() - started

    failures = []
    for label, row in (("upper", up), ("lower", lo)):
        for key in ("Cf", "Nu"):
            value = row[key]
            if not math.isfinite(value):
                failures.append(f"smoke: {label}.{key} is not finite ({value!r})")
    if not failures:
        if not (up["Cf"] > lo["Cf"] > 0):
            failures.append(
                "smoke: expected upper Cf > lower Cf > 0, got "
                f"upper={up['Cf']!r} lower={lo['Cf']!r}"
            )
        if not (up["Nu"] > 0 and lo["Nu"] > 0):
            failures.append(
                f"smoke: expected both-branch Nu > 0, got upper={up['Nu']!r} lower={lo['Nu']!r}"
            )
    return {"elapsed_seconds": elapsed, "upper": up, "lower": lo, "failures": failures}


# ---------------------------------------------------------------- full gates
def run_full_table12(candidate: ModuleType, p: dict, tol: float) -> dict:
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        _, up = candidate.solve_branch(dict(p), "upper")
        _, lo = candidate.solve_branch(dict(p), "lower")
    elapsed = time.perf_counter() - started

    errors = {
        "upper_Cf_rel_err": rel_err(up["Cf"], PAPER_TABLE_1_2["upper"]["Cf"]),
        "upper_Nu_rel_err": rel_err(up["Nu"], PAPER_TABLE_1_2["upper"]["Nu"]),
        "lower_Cf_rel_err": rel_err(lo["Cf"], PAPER_TABLE_1_2["lower"]["Cf"]),
        "lower_Nu_rel_err": rel_err(lo["Nu"], PAPER_TABLE_1_2["lower"]["Nu"]),
    }
    failures = [f"{name} {err:.4%} exceeds tol {tol:.4%}" for name, err in errors.items() if err > tol]
    return {"elapsed_seconds": elapsed, "upper": up, "lower": lo, "errors": errors, "failures": failures}


def run_full_fold(candidate: ModuleType, tol_sc: float) -> dict:
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        rows = candidate.trace_fold(dict(FOLD_PARAMS), list(FOLD_F2_VALUES))
    elapsed = time.perf_counter() - started
    sc = min(row[1] for row in rows)
    err = rel_err(sc, PAPER_SC)
    failures = [] if err <= tol_sc else [f"S_c relative error {err:.4%} exceeds tol {tol_sc:.4%}"]
    return {"elapsed_seconds": elapsed, "S_c": sc, "rows": rows, "error": err, "failures": failures}


def run_full_stability(candidate: ModuleType, tol: float) -> dict:
    p = dict(STABILITY_PARAMS)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        u_net, _ = candidate.solve_branch(p, "upper")
        l_net, _ = candidate.solve_branch(p, "lower")
        a_up = candidate.smallest_eigen(u_net, p, 1.0)
        a_lo = candidate.smallest_eigen(l_net, p, -0.1)
    elapsed = time.perf_counter() - started

    errors = {
        "upper_alpha1_rel_err": rel_err(a_up, PAPER_TABLE_3["upper"]),
        "lower_alpha1_rel_err": rel_err(a_lo, PAPER_TABLE_3["lower"]),
    }
    failures = [f"{name} {err:.4%} exceeds tol {tol:.4%}" for name, err in errors.items() if err > tol]
    # The sign is the physically load-bearing claim (stable vs. unstable
    # branch); flag a sign flip even if a loose magnitude tolerance passed.
    if a_up <= 0:
        failures.append(f"upper branch alpha1 not positive (got {a_up!r}); stability claim violated")
    if a_lo >= 0:
        failures.append(f"lower branch alpha1 not negative (got {a_lo!r}); instability claim violated")
    return {
        "elapsed_seconds": elapsed, "alpha1_upper": a_up, "alpha1_lower": a_lo,
        "errors": errors, "failures": failures,
    }


def run_oracle_crosscheck(p: dict) -> dict:
    """Independent scipy.solve_bvp re-solve of the Table 1/2 baseline row,
    purely informative: sanity-checks that PAPER_TABLE_1_2 above was
    transcribed correctly. Never gates the candidate."""
    ref_bvp, _eig_cheb = load_oracle()
    out = {}
    for branch in ("upper", "lower"):
        with contextlib.redirect_stdout(io.StringIO()):
            sol = ref_bvp.solve(dict(p), branch)
        f2, t1 = sol.sol(0)[2], sol.sol(0)[4]
        cf = (1 + p["beta"]) * f2
        nu = -t1
        out[branch] = {
            "ok": bool(sol.success),
            "Cf": cf, "Nu": nu,
            "Cf_vs_paper_rel_err": rel_err(cf, PAPER_TABLE_1_2[branch]["Cf"]),
            "Nu_vs_paper_rel_err": rel_err(nu, PAPER_TABLE_1_2[branch]["Nu"]),
        }
    return out


# ---------------------------------------------------------------- result shape
def result(*, valid, improved, status, objective_value, baseline_value,
           objective_name, direction, metrics, failures) -> dict:
    return {
        "valid": valid,
        "improved": improved,
        "status": status,
        "objective": {
            "name": objective_name,
            "value": objective_value,
            "baseline_value": baseline_value,
            "direction": direction,
        },
        "metrics": metrics,
        "failures": failures,
    }


def fail(message: str) -> dict:
    return result(
        valid=False, improved=False, status="FAILED",
        objective_value=0.0, baseline_value=0.0,
        objective_name="elapsed_seconds", direction="minimize",
        metrics={}, failures=[message],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tol", type=float, default=0.01,
                         help="relative-error tolerance for Cf/Nu/alpha1 gates (default 1%%)")
    parser.add_argument("--fold", action="store_true", help="full mode: also gate S_c (fold tracing)")
    parser.add_argument("--stability", action="store_true", help="full mode: also gate Table 3 alpha1")
    parser.add_argument("--oracle", action="store_true",
                         help="full mode: also run the independent scipy.solve_bvp cross-check (informative only)")
    parser.add_argument("--baseline-seconds", type=float, default=None,
                         help="prior recorded elapsed_seconds, to report objective.value as a speedup claim")
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    if candidate_path.is_dir():
        pys = sorted(candidate_path.glob("*.py"))
        if len(pys) != 1:
            print(json.dumps(fail(f"expected exactly one .py file in {candidate_path}")))
            return 0
        candidate_path = pys[0]

    try:
        candidate = load_module(candidate_path, "pinn_candidate")
    except Exception as error:  # noqa: BLE001 - must always emit valid JSON
        print(json.dumps(fail(f"import failed: {type(error).__name__}: {error}")))
        return 0

    missing = require_contract(candidate)
    if missing:
        print(json.dumps(fail(f"candidate missing required contract members: {missing}")))
        return 0

    try:
        if args.mode == "smoke":
            smoke = run_smoke(candidate, BASELINE_PARAMS)
            failures = list(smoke["failures"])
            metrics = {"mode": "smoke", "smoke": smoke}
            objective_value = smoke["elapsed_seconds"]
        else:
            table12 = run_full_table12(candidate, BASELINE_PARAMS, args.tol)
            failures = list(table12["failures"])
            metrics = {"mode": "full", "table1_2": table12}
            objective_value = table12["elapsed_seconds"]

            if args.oracle:
                metrics["oracle_vs_paper"] = run_oracle_crosscheck(BASELINE_PARAMS)
            if args.fold:
                fold = run_full_fold(candidate, tol_sc=max(args.tol, 0.03))
                failures += fold["failures"]
                metrics["fold"] = fold
                objective_value += fold["elapsed_seconds"]
            if args.stability:
                stability = run_full_stability(candidate, args.tol)
                failures += stability["failures"]
                metrics["stability"] = stability
                objective_value += stability["elapsed_seconds"]

        valid = not failures
    except Exception as error:  # noqa: BLE001 - must always emit valid JSON
        print(json.dumps(fail(f"evaluation raised {type(error).__name__}: {error}")))
        return 0

    baseline_value = args.baseline_seconds if args.baseline_seconds is not None else objective_value
    improved = bool(valid and args.baseline_seconds is not None and objective_value < baseline_value)

    print(json.dumps(result(
        valid=valid, improved=improved, status="SUPPORTED" if valid else "FAILED",
        objective_value=objective_value, baseline_value=baseline_value,
        objective_name="elapsed_seconds", direction="minimize",
        metrics=metrics, failures=failures,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
