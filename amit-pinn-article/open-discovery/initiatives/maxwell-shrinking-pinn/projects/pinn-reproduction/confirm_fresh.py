"""Fresh-confirmation check (separate from, and never modifying, evaluator.py).

Runs a candidate's own unmodified `solve_branch` at Table 1 rows NOT used during candidate
search (search used only the S=1.95 baseline row and the S=2.0 stability row), over several
torch seeds, and compares Cf/Nu with the independent scipy.solve_bvp oracle solved with a
long domain (eta_inf=40, converged) as well as with the paper's Table 1 value (eta_inf=10).
Usage: python confirm_fresh.py <candidate.py> <seed> [<seed> ...]   -> one JSON object.
"""
import contextlib, importlib.util, io, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROWS = [  # (S, K, beta, paper upper Cf, paper lower Cf)   [Table 1, reference-only rows]
    (2.10, 0.1, 0.1, 2.62119412, 0.56244284),
    (1.95, 0.13, 0.1, 1.98493433, 0.69930309),
    (1.95, 0.1, 0.15, 2.77435660, 0.71841216),
]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m

def main():
    cand_path, seeds = Path(sys.argv[1]).resolve(), [int(s) for s in sys.argv[2:]] or [0]
    sys.path.insert(0, str(HERE / "oracle"))
    import reference_bvp as ref
    out = {"candidate": str(cand_path), "seeds": seeds, "rows": []}
    for S, K, beta, pu, pl in ROWS:
        p = dict(beta=beta, K=K, S=S, Pr=2.0, A=0.1, B=0.1)
        oracle = {}
        for br in ("upper", "lower"):
            sol = ref.solve(dict(p), br, eta_inf=40.0)
            oracle[br] = dict(ok=bool(sol.success), Cf=(1 + beta) * sol.sol(0)[2], Nu=-sol.sol(0)[4])
        row = dict(params=p, paper_Cf=dict(upper=pu, lower=pl), oracle=oracle, runs=[])
        for seed in seeds:
            cand = load(cand_path, f"cand_{seed}")
            cand.torch.manual_seed(seed)
            t = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                _, up = cand.solve_branch(dict(p), "upper")
                _, lo = cand.solve_branch(dict(p), "lower")
            run = dict(seed=seed, seconds=time.perf_counter() - t)
            for br, r in (("upper", up), ("lower", lo)):
                run[br] = dict(Cf=r["Cf"], Nu=r["Nu"],
                               Cf_err_oracle=abs(r["Cf"] / oracle[br]["Cf"] - 1),
                               Nu_err_oracle=abs(r["Nu"] / oracle[br]["Nu"] - 1),
                               Cf_err_paper=abs(r["Cf"] / (pu if br == "upper" else pl) - 1))
            row["runs"].append(run)
            print(json.dumps(dict(S=S, K=K, beta=beta, **run)), file=sys.stderr, flush=True)
        out["rows"].append(row)
    print(json.dumps(out))

if __name__ == "__main__":
    main()
