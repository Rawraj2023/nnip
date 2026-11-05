"""Independent diagnostic (not a gate): Harris-style eigenvalue by shooting.
Base flow from oracle/reference_bvp.py at eta_inf=Lb. Disturbance (paper Eq. 31 momentum part):
 (1-b f^2) g''' + (2b f f' + f) g'' + (2b f f'' - 2f' - K + a) g' + (2b f' f'' - 2b f f''' + f'') g = 0
 g(0)=0, g'(0)=0, g''(0)=1 (normalisation), and g'(Lg)=0 at a finite edge Lg.
Scans alpha for roots of g'(Lg; alpha)."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import reference_bvp as ref
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def roots(p, branch, Lb, Lg, grid=np.linspace(-1.5, 3.0, 181)):
    base = ref.solve(dict(p), branch, eta_inf=Lb)
    b, K = p["beta"], p["K"]
    def rhs(e, y, a):
        f, fp, fpp = base.sol(min(e, Lb))[:3]; fppp = base.sol(min(e, Lb), 1)[2]
        g, g1, g2 = y
        g3 = -((2*b*f*fp + f)*g2 + (2*b*f*fpp - 2*fp - K + a)*g1 + (2*b*fp*fpp - 2*b*f*fppp + fpp)*g) / (1 - b*f*f)
        return [g1, g2, g3]
    def edge(a):
        s = solve_ivp(rhs, [0, Lg], [0, 0, 1], args=(a,), rtol=1e-10, atol=1e-12)
        return s.y[1, -1]
    v = [edge(a) for a in grid]
    out = []
    for i in range(len(grid) - 1):
        if np.sign(v[i]) != np.sign(v[i + 1]):
            out.append(brentq(edge, grid[i], grid[i + 1], xtol=1e-10))
    return out

if __name__ == "__main__":
    p = dict(beta=0.1, K=0.1, S=2.0, Pr=2.0, A=0.1, B=0.0)
    for Lg in (6.0, 8.0, 10.0, 12.0):
        for br in ("upper", "lower"):
            r = roots(p, br, 10.0, Lg)
            print(f"eta_edge={Lg:5.1f} {br}: alpha roots in [-1.5,3]: {np.round(r, 5).tolist()}", flush=True)
