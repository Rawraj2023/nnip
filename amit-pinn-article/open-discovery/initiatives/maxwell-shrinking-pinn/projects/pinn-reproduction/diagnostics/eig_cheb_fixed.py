"""Diagnostic copy of oracle/ref_eig_cheb.py with ONE change: boundary rows at the correct ends.
In the original, x = cos(pi k/N) so x[0]=1 -> eta=0 (the WALL) and x[N]=-1 -> eta=L, but the code
sets i0=N (wall) and iL=0, i.e. imposes g(L)=0, g'(L)=0, g'(0)=0 instead of g(0)=0, g'(0)=0, g'(L)=0.
`swap=True` reproduces the original (buggy) placement for comparison. The oracle file is untouched."""
import sys, numpy as np, scipy.linalg as sl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
from reference_bvp import solve

def cheb(N):
    x = np.cos(np.pi*np.arange(N+1)/N); c = np.r_[2, np.ones(N-1), 2]*(-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1)).T; dX = X-X.T; D = np.outer(c, 1/c)/(dX+np.eye(N+1)); D -= np.diag(D.sum(1)); return D, x

def spectrum(p, branch, L=12, N=120, swap=False):
    base = solve(p, branch, eta_inf=L)
    D, x = cheb(N); eta = (1-x)*L/2; D1 = -D*2/L; D2 = D1@D1; D3 = D2@D1
    f, fp, fpp = base.sol(eta)[:3]; fppp = base.sol(eta, 1)[2]; b, K = p['beta'], p['K']; I = np.diag
    Lop = I(1-b*f**2)@D3+I(2*b*f*fp+f)@D2+I(2*b*f*fpp-2*fp-K)@D1+I(2*b*fp*fpp-2*b*f*fppp+fpp)
    Bop = -D1.copy()
    i0, iL = (N, 0) if swap else (0, N)
    assert swap or abs(eta[i0]) < 1e-12
    for row, vec in [(i0, np.eye(N+1)[i0]), (i0+(-1 if swap else 1), D1[i0]), (iL, D1[iL])]:
        Lop[row] = vec; Bop[row] = 0
    w = sl.eigvals(Lop, Bop); w = w[np.isfinite(w)]; w = w[np.abs(w) < 1e3]
    w = w[np.argsort(w.real)]; return w[:3]

if __name__ == "__main__":
    for S in (2.0, 2.1, 2.2, 2.3, 2.4):
        p = dict(beta=0.1, K=0.1, S=S, Pr=2.0, A=0.1, B=0.0)
        for L, N in ((10, 120), (12, 120), (16, 160)):
            u = spectrum(p, 'upper', L, N); l = spectrum(p, 'lower', L, N)
            print(f"S={S} L={L:2d} N={N}: upper {np.round(u.real,5).tolist()}  lower {np.round(l.real,5).tolist()}", flush=True)
