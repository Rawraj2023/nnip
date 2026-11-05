import numpy as np, scipy.linalg as sl
from reference_bvp import solve
def cheb(N):
    x=np.cos(np.pi*np.arange(N+1)/N); c=np.r_[2,np.ones(N-1),2]*(-1)**np.arange(N+1)
    X=np.tile(x,(N+1,1)).T; dX=X-X.T; D=np.outer(c,1/c)/(dX+np.eye(N+1)); D-=np.diag(D.sum(1)); return D,x
def spectrum(p,branch,L=12,N=120):
    base=solve(p,branch,eta_inf=L)
    D,x=cheb(N); eta=(1-x)*L/2; D1=-D*2/L; D2=D1@D1; D3=D2@D1
    f,fp,fpp=base.sol(eta)[:3]; fppp=base.sol(eta,1)[2]; b,K=p['beta'],p['K']
    I=np.diag
    Lop=I(1-b*f**2)@D3+I(2*b*f*fp+f)@D2+I(2*b*f*fpp-2*fp-K)@D1+I(2*b*fp*fpp-2*b*f*fppp+fpp)
    Bop=-D1.copy()   # L g + alpha g' = 0  ->  L g = alpha (-g')
    # BCs: g(0)=0 (row at eta=0 is last index), g'(0)=0, g'(L)=0
    i0=N; iL=0
    for row,vec in [(i0,np.eye(N+1)[i0]),(i0-1,D1[i0]),(iL,D1[iL])]:
        Lop[row]=vec; Bop[row]=0
    w=sl.eigvals(Lop,Bop); w=w[np.isfinite(w)]; w=w[np.abs(w)<1e3]
    w=w[np.argsort(w.real)]; return w[:3]
for S in [2.0,2.2,2.4]:
    p=dict(beta=0.1,K=0.1,S=S,Pr=2.0,A=0.1,B=0.0)
    print(S,np.round(spectrum(p,'upper'),5),np.round(spectrum(p,'lower'),5))
