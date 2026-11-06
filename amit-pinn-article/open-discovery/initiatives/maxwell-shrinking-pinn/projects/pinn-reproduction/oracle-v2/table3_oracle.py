"""Table 3 (stability) oracle, v2 — independent of the PINN; bundled oracle/ files untouched.

Paper Eq. 31 (momentum disturbance, = linearisation of the steady Eq. 13 with +alpha g'):
  (1-b f^2) g''' + (2b f f' + f) g'' + (2b f f'' - 2f' - K + alpha) g' + (2b f' f'' - 2b f f''' + f'') g = 0
  g(0)=0, g'(0)=0, g'(inf)=0
Paper Eq. 32 with g0=0 (energy-only disturbance):  phi'' + Pr f phi' + (A* + Pr alpha) phi = 0, phi(0)=phi(inf)=0.

Far field (f -> f_inf, f',f''->0): g' ~ exp(-mu eta), (1-b f_inf^2) mu^2 - f_inf mu + (alpha-K) = 0.
  alpha < K : one decaying + one growing root -> discrete eigenvalues (well-posed on [0,inf)).
  alpha > K : both roots decay -> every alpha > K satisfies g'(inf)=0 (continuous spectrum);
              finite-L solvers return roots that crowd toward
              alpha* = K + f_inf^2 / (4 (1 - b f_inf^2))   (onset of oscillatory decay).
Methods for the momentum problem:
  cheb   : Chebyshev collocation on [0,L], BC rows at the CORRECT ends (wall = x=+1 = index 0)
  shoot  : IVP shooting g(0)=g'(0)=0, g''(0)=1, roots of g'(L; alpha)
  asym   : shooting with the exact asymptotic BC g''(L) + mu_dec(alpha) g'(L) = 0 (alpha < K only)
  swapped: bundled oracle/ref_eig_cheb.py placement (BC rows at the wrong ends) — attribution only
Usage: python table3_oracle.py > table3_oracle.json      (progress on stderr)
"""
import json, sys, numpy as np, scipy.linalg as sl
from pathlib import Path
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import reference_bvp as ref

PAPER = {2.0: (0.8810518, -0.0376726), 2.1: (1.1498579, -0.1290175), 2.2: (1.4609727, -0.2006274),
         2.3: (1.8396721, -0.2601093), 2.4: (2.3116364, -0.3115262)}
LS = (10.0, 12.0, 16.0, 20.0, 24.0)


def base_flow(p, br, L, _cache={}):
    key = (p["S"], br, L)
    if key not in _cache:
        if L <= 10:
            sol = ref.solve(dict(p), br, eta_inf=L)
        else:  # continuation in L from the eta_inf=10 solution (constant extension of f, decaying f')
            prev = base_flow(p, br, 10.0)
            eta = np.linspace(0, L, 4001); y = prev.sol(np.minimum(eta, 10.0))
            y[1:3] *= np.exp(-np.maximum(eta - 10.0, 0))[None]; y[3:] *= np.exp(-np.maximum(eta - 10.0, 0))[None]
            sol = solve_bvp(lambda x, yy: ref.rhs(x, yy, p), lambda a, b: ref.bc(a, b, p), eta, y,
                            tol=1e-8, max_nodes=400000)
        assert sol.success, (key, sol.message)
        _cache[key] = sol
    return _cache[key]


def coeffs(sol, eta, b, K):
    f, fp, fpp = sol.sol(eta)[:3]; fppp = sol.sol(eta, 1)[2]
    return f, (1 - b*f*f), (2*b*f*fp + f), (2*b*f*fpp - 2*fp - K), (2*b*fp*fpp - 2*b*f*fppp + fpp)


def cheb(N):
    x = np.cos(np.pi*np.arange(N+1)/N); c = np.r_[2, np.ones(N-1), 2]*(-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1)).T; dX = X - X.T
    D = np.outer(c, 1/c)/(dX + np.eye(N+1)); D -= np.diag(D.sum(1)); return D, x


def momentum_cheb(p, br, L, N, swap=False, k=4):
    sol = base_flow(p, br, L); D, x = cheb(N); eta = (1 - x)*L/2
    D1 = -D*2/L; D2 = D1 @ D1; D3 = D2 @ D1; I = np.diag
    _, a3, a2, a1, a0 = coeffs(sol, eta, p["beta"], p["K"])
    A = I(a3) @ D3 + I(a2) @ D2 + I(a1) @ D1 + I(a0); B = -D1.copy()
    i0, iL = (N, 0) if swap else (0, N)          # swap reproduces the bundled oracle's placement
    for row, vec in [(i0, np.eye(N+1)[i0]), (i0 + (-1 if swap else 1), D1[i0]), (iL, D1[iL])]:
        A[row] = vec; B[row] = 0
    w = sl.eigvals(A, B); w = w[np.isfinite(w) & (np.abs(w) < 1e3)]
    w = w[np.abs(w.imag) < 1e-6*np.maximum(1, np.abs(w.real))].real   # real eigenvalues only
    return np.sort(w)[:k].tolist()


def _shoot(p, br, L, a, sol):
    b, K = p["beta"], p["K"]
    def rhs(e, y):
        _, a3, a2, a1, a0 = coeffs(sol, e, b, K)
        return [y[1], y[2], -(a2*y[2] + (a1 + a)*y[1] + a0*y[0]) / a3]
    return solve_ivp(rhs, [0, L], [0, 0, 1], rtol=1e-10, atol=1e-12).y[:, -1]


def roots(fun, grid):
    v = [fun(a) for a in grid]; out = []
    for i in range(len(grid) - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i+1]) and np.sign(v[i]) != np.sign(v[i+1]):
            out.append(brentq(fun, grid[i], grid[i+1], xtol=1e-11))
    return out


def momentum_shoot(p, br, L, grid):
    sol = base_flow(p, br, L)
    return roots(lambda a: _shoot(p, br, L, a, sol)[1], grid)


def momentum_asym(p, br, L, f_inf):
    """alpha < K only: kill the growing far-field mode exactly."""
    sol = base_flow(p, br, L); b, K = p["beta"], p["K"]; c2 = 1 - b*f_inf**2
    def res(a):
        mu = (f_inf + np.sqrt(f_inf**2 - 4*c2*(a - K))) / (2*c2)   # decaying root (>0)
        y = _shoot(p, br, L, a, sol); return (y[2] + mu*y[1]) / np.hypot(y[1], y[2])
    return roots(res, np.linspace(-2.0, K - 1e-6, 141))


def energy_cheb(p, br, L, N, k=3):
    sol = base_flow(p, br, L); D, x = cheb(N); eta = (1 - x)*L/2
    D1 = -D*2/L; D2 = D1 @ D1; f = sol.sol(eta)[0]; Pr, A = p["Pr"], p["A"]
    M = D2 + np.diag(Pr*f) @ D1 + A*np.eye(N+1); B = -Pr*np.eye(N+1)
    for i in (0, N):
        M[i] = np.eye(N+1)[i]; B[i] = 0
    w = sl.eigvals(M, B); w = w[np.isfinite(w) & (np.abs(w) < 1e3)]
    return np.sort(w.real)[:k].tolist()


def main():
    out = {"params": dict(beta=0.1, K=0.1, Pr=2.0, A=0.1, B=0.0), "domains": LS, "rows": []}
    for S, (pu, pl) in PAPER.items():
        p = dict(out["params"], S=S); row = {"S": S, "paper": {"upper": pu, "lower": pl}}
        for br in ("upper", "lower"):
            f_inf = float(base_flow(p, br, 40.0 if br == "upper" else 24.0).sol(40.0 if br == "upper" else 24.0)[0])
            c2 = 1 - p["beta"]*f_inf**2
            a_star = p["K"] + f_inf**2/(4*c2) if c2 > 0 else None
            r = {"f_inf": f_inf, "one_minus_b_finf2": c2, "alpha_star": a_star, "by_L": {}}
            for L in LS:
                N = int(10*L)
                lo, hi = (-2.0, p["K"] - 1e-6) if br == "lower" else (p["K"], 4.0)
                d = {"cheb": momentum_cheb(p, br, L, N),
                     "shoot": momentum_shoot(p, br, L, np.linspace(-2.0, 4.0, 241))[:4],
                     "swapped_bundled": momentum_cheb(p, br, L, N, swap=True, k=2),
                     "energy_cheb": energy_cheb(p, br, L, N)}
                if br == "lower":
                    d["asym"] = momentum_asym(p, br, L, f_inf)
                r["by_L"][str(L)] = d
                print(json.dumps({"S": S, "branch": br, "L": L, **{k: np.round(v, 5).tolist() for k, v in d.items()}}),
                      file=sys.stderr, flush=True)
            row[br] = r
        out["rows"].append(row)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
