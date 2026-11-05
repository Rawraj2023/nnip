"""
Reference (non-PINN) solver for Pandey et al., Forces in Mechanics 100352.
Used to (a) check how the paper's tables are defined and (b) give ground truth
for validating the PINN.

  (1 - b f^2) f''' + 2 b f f' f'' + f f'' - f'^2 - K f' = 0
  theta'' + Pr f theta' + A theta + B exp(-eta) = 0
  f(0)=S, f'(0)=-1, f'(inf)=0, theta(0)=1, theta(inf)=0
"""
import numpy as np
from scipy.integrate import solve_bvp


def rhs(eta, y, p):
    f, fp, fpp, th, thp = y
    b, K, Pr, A, B = p["beta"], p["K"], p["Pr"], p["A"], p["B"]
    fppp = (fp**2 + K * fp - f * fpp - 2 * b * f * fp * fpp) / (1 - b * f**2)
    thpp = -Pr * f * thp - A * th - B * np.exp(-eta)
    return np.vstack([fp, fpp, fppp, thp, thpp])


def bc(ya, yb, p):
    return np.array([ya[0] - p["S"], ya[1] + 1, yb[1], ya[3] - 1, yb[3]])


def solve(p, branch="upper", eta_inf=10.0, guess=None):
    eta = np.linspace(0, eta_inf, 2001)
    if guess is None:
        # upper branch: fast decay; lower branch: slow decay of f'
        lam = 3.0 if branch == "upper" else 0.6
        fp = -np.exp(-lam * eta)
        f = p["S"] - (1 - np.exp(-lam * eta)) / lam
        fpp = lam * np.exp(-lam * eta)
        th = np.exp(-2 * eta)
        thp = -2 * np.exp(-2 * eta)
        guess = (eta, np.vstack([f, fp, fpp, th, thp]))
    sol = solve_bvp(lambda x, y: rhs(x, y, p), lambda a, b: bc(a, b, p),
                    guess[0], guess[1], tol=1e-8, max_nodes=200000)
    return sol


if __name__ == "__main__":
    base = dict(beta=0.1, K=0.1, S=1.95, Pr=2.0, A=0.1, B=0.1)
    for br in ["upper", "lower"]:
        s = solve(base, br)
        f2, t1 = s.sol(0)[2], s.sol(0)[4]
        print(f"{br:5s} ok={s.success}  f''(0)={f2:.8f}  (1+b)f''(0)={(1+base['beta'])*f2:.8f}"
              f"  theta'(0)={t1:.8f}")
