"""
PINN for: Pandey et al., "Insight of thermal transport in Maxwell fluid flow on a porous
contracting flat surface ... Multiple solutions and stability treatment",
Forces in Mechanics 100352 (uncorrected proof).

Governing similarity ODEs (paper Eqs. 13-15, with f''' written out):
    (1 - beta f^2) f''' + 2 beta f f' f'' + f f'' - f'^2 - K f' = 0
    theta'' + Pr f theta' + A* theta + B* exp(-eta) = 0
    f(0)=S, f'(0)=-1, f'(inf)=0, theta(0)=1, theta(inf)=0
Outputs:  Cf Re^1/2 = (1+beta) f''(0)    NuRe^-1/2 = -theta'(0)

Three stages, each a separate PINN:
  1. solve_branch()    -> upper / lower branch for one parameter set
  2. trace_fold()      -> both branches + critical S_c by treating S as unknown and
                          prescribing f''(0) (passes smoothly through the turning point)
  3. smallest_eigen()  -> leading eigenvalue alpha_1 of the linear stability problem

Run:  python pinn_maxwell_shrinking.py
"""
import math
import time
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)   # 3rd derivatives + small residuals -> use float64
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ETA_INF = 10.0   # same truncation as the paper
L_MAP = 2.0      # length scale of the map xi = 1 - exp(-eta/L)


# ---------------------------------------------------------------- network
class MLP(nn.Module):
    def __init__(self, n_in=1, n_out=1, width=40, depth=4):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]   # smooth activation: we need f'''
            d = width
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def d(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def collocation(n=300, xi_max=0.9995):
    """
    Uniform points in the mapped variable xi = 1 - exp(-eta/L), i.e. eta up to ~15 for L=2.
    This (a) clusters points near the wall automatically and (b) puts residual points far
    out in the free stream. IMPORTANT: the BCs at infinity are imposed by the trial function
    at xi -> 1, so the residual must also be enforced far from the wall; collocating only on
    [0, 10] lets the optimiser find spurious solutions that satisfy theta(inf)=0 only
    "outside" the training domain.
    """
    xi = torch.linspace(0, xi_max, n, device=DEV).unsqueeze(1)
    eta = -L_MAP * torch.log(1 - xi)
    return eta.requires_grad_(True)


# ---------------------------------------------------------------- hard-constrained trial functions
def fields(net, eta, S):
    """
    net = (net_f, net_t): two separate networks (momentum is independent of energy, so they
    are trained one after the other).  xi = 1 - exp(-eta/L) in [0,1).  Trial forms satisfy
    EVERY boundary condition exactly:
        f     = S - L*xi + xi^2 * N1(xi)          f(0)=S, f'(0)=-1, f'(inf)=0 (since dxi/deta -> 0)
        theta = (1 - xi) * (1 + xi * N2(xi))      theta(0)=1, theta(inf)=0
    So the loss contains only the ODE residuals (no BC weights to tune).
    """
    net_f, net_t = net
    L = getattr(net_f, "L", L_MAP)        # each trained net remembers the map it was trained with
    xi = 1 - torch.exp(-eta / L)
    z = 2 * xi - 1                        # feed networks a [-1,1] input
    f = S - L * xi + xi**2 * net_f(z)
    th = (1 - xi) * (1 + xi * net_t(z))
    return f, th


def residuals(net, eta, S, p):
    f, th = fields(net, eta, S)
    f1 = d(f, eta); f2 = d(f1, eta); f3 = d(f2, eta)
    t1 = d(th, eta); t2 = d(t1, eta)
    b, K, Pr, A, B = p["beta"], p["K"], p["Pr"], p["A"], p["B"]
    r_mom = (1 - b * f**2) * f3 + 2 * b * f * f1 * f2 + f * f2 - f1**2 - K * f1
    r_eng = t2 + Pr * f * t1 + A * th + B * torch.exp(-eta)
    return r_mom, r_eng


def wall_values(net, S, p):
    eta0 = torch.zeros(1, 1, device=DEV, requires_grad=True)
    f, th = fields(net, eta0, S)
    f2 = d(d(f, eta0), eta0)
    t1 = d(th, eta0)
    return f2.item(), t1.item()


def train(net, loss_fn, extra_params=(), adam_iters=4000, lbfgs_iters=1500, lr=2e-3, log=True):
    params = list(net.parameters()) + list(extra_params)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)
    for it in range(adam_iters):
        opt.zero_grad(); loss = loss_fn(it); loss.backward(); opt.step(); sched.step()
        if log and it % 1000 == 0:
            print(f"   adam {it:5d}  loss {loss.item():.3e}")
    # candidate-002: same LBFGS, run in chunks (optimizer state/history persists across
    # step() calls); stop once a chunk improves the loss by < 1 %, never exceed lbfgs_iters.
    CHUNK, REL = 50, 0.01
    opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=CHUNK, tolerance_grad=1e-12,
                             tolerance_change=1e-15, history_size=100, line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad(); l = loss_fn(10**9); l.backward(); return l
    prev = loss_fn(10**9).item()
    done = 0
    while done < lbfgs_iters:
        opt2.param_groups[0]["max_iter"] = min(CHUNK, lbfgs_iters - done)
        opt2.step(closure)
        done += CHUNK
        cur = loss_fn(10**9).item()
        if not (cur < prev * (1 - REL)):
            break
        prev = cur
    final = loss_fn(10**9).item()
    if log:
        print(f"   lbfgs final loss {final:.3e}")
    return final


# ---------------------------------------------------------------- stage 1: one branch
def far_field_decay(f_inf, p):
    """Linearise the momentum eq. for large eta (f -> f_inf, f' ~ -C exp(-lam*eta)):
       (1 - b f_inf^2) lam^2 - f_inf lam - K = 0  ->  positive root lam."""
    b, K = p["beta"], p["K"]
    a = 1 - b * f_inf**2
    return (f_inf + math.sqrt(f_inf**2 + 4 * K * a)) / (2 * a)


def solve_branch(p, branch="upper", f2_anchor=None, anchor_iters=1500, auto_scale=True):
    """
    auto_scale: a short pilot run estimates f(inf); the map length is then set to
    L = m/lam (m the smallest integer giving L >= 1) so that the true far-field decay
    exp(-lam*eta) = (1-xi)^m is a SMOOTH function of xi. With a mismatched L the decay is
    (1-xi)^(non-integer) -> derivative singular at xi=1 -> error floor ~1-2 % on the lower
    branch. With the matched L the error drops to ~0.1 %.
    """
    global L_MAP
    if auto_scale:
        L_MAP = 2.0
        pilot, _ = _solve_branch(p, branch, f2_anchor, anchor_iters, quick=True)
        with torch.no_grad():
            f_inf = fields(pilot, torch.tensor([[60.0]], device=DEV),
                           torch.tensor(p["S"], device=DEV))[0].item()
        lam = far_field_decay(f_inf, p)
        L_MAP = math.ceil(lam) / lam
        print(f"[{branch}] pilot f(inf)={f_inf:.4f}, decay rate {lam:.4f} -> L={L_MAP:.4f}")
    return _solve_branch(p, branch, f2_anchor, anchor_iters)


def _solve_branch(p, branch, f2_anchor, anchor_iters, quick=False):
    """
    Branch selection: for the first `anchor_iters` Adam steps an extra loss term pulls
    f''(0) toward a rough target (large -> upper, small -> lower). The anchor is then
    switched OFF, so the converged solution satisfies the ODEs alone; the anchor only
    picks which basin of attraction the optimiser falls into.
    """
    if f2_anchor is None:
        f2_anchor = 2.0 if branch == "upper" else 0.4
    net = (MLP().to(DEV), MLP().to(DEV))
    net[0].L = L_MAP
    eta = collocation()
    eta0 = torch.zeros(1, 1, device=DEV, requires_grad=True)
    S = torch.tensor(p["S"], device=DEV)

    def loss_mom(it):
        rm, _ = residuals(net, eta, S, p)
        loss = (rm**2).mean()
        if it < anchor_iters:
            f, _ = fields(net, eta0, S)
            f2 = d(d(f, eta0), eta0)
            loss = loss + 1.0 * (f2 - f2_anchor).pow(2).sum()
        return loss

    def loss_eng(it):
        _, re = residuals(net, eta, S, p)
        return (re**2).mean()

    if quick:
        train(net[0], loss_mom, adam_iters=2500, lbfgs_iters=200, log=False)
        return net, None
    print(f"[{branch}] momentum ...")
    train(net[0], loss_mom)
    for q in net[0].parameters():          # freeze f, then solve the (linear) energy equation
        q.requires_grad_(False)
    print(f"[{branch}] energy ...")
    train(net[1], loss_eng, adam_iters=2000, lbfgs_iters=1000)
    for q in net[0].parameters():
        q.requires_grad_(True)
    f2, t1 = wall_values(net, S, p)
    return net, dict(f2=f2, Cf=(1 + p["beta"]) * f2, Nu=-t1, theta1=t1)


# ---------------------------------------------------------------- stage 2: trace through the fold
def trace_fold(p, f2_values):
    """
    Natural-parameter switch: prescribe c = f''(0), learn S as a trainable scalar.
    S(c) is single-valued and smooth, so both branches AND the turning point are reached
    without any branch-selection trick. min_c S(c) = critical suction S_c.
    """
    global L_MAP
    L_MAP = 1.3                            # compromise between upper/lower decay rates
    rows = []
    net = (MLP().to(DEV), MLP().to(DEV))   # only the momentum net is trained here
    net[0].L = L_MAP
    eta = collocation()
    eta0 = torch.zeros(1, 1, device=DEV, requires_grad=True)
    S = torch.nn.Parameter(torch.tensor(p["S"], device=DEV))
    for k, c in enumerate(f2_values):
        def loss_fn(it, c=c):
            rm, _ = residuals(net, eta, S, p)
            f, _ = fields(net, eta0, S)
            f2 = d(d(f, eta0), eta0)
            return (rm**2).mean() + 10.0 * (f2 - c).pow(2).sum()
        # warm start from the previous point (continuation) -> fewer iterations after the first
        train(net[0], loss_fn, extra_params=[S], adam_iters=4000 if k == 0 else 800,
              lbfgs_iters=800, log=False)
        rows.append((c, S.item(), (1 + p["beta"]) * c))
        print(f"   f''(0)={c:.3f}  ->  S={S.item():.5f}   CfRe^1/2={(1+p['beta'])*c:.5f}")
    return rows


# ---------------------------------------------------------------- stage 3: stability eigenvalue
def smallest_eigen(base_net, p, alpha_init):
    """
    Linearised disturbance problem at tau=0 (paper Eq. 31, momentum part):
      (1-b f^2) g''' + (2b f f' + f) g'' + (2b f f'' - 2f' - K + alpha) g'
                     + (2b f' f'' - 2b f f''' + f'') g = 0
      g(0)=0, g'(0)=0, g'(inf)=0
    alpha is a trainable scalar. Trivial g=0 is excluded by the normalisation g''(0)=1
    (built into the trial function), as in Harris et al. Energy disturbance is decoupled and
    does not change alpha_1. Base flow f comes from a trained (frozen) branch network.
    """
    for q in base_net[0].parameters():
        q.requires_grad_(False)
    gnet = MLP(1, 1).to(DEV)
    alpha = torch.nn.Parameter(torch.tensor(alpha_init, device=DEV))
    eta = collocation()
    S = torch.tensor(p["S"], device=DEV)
    b, K = p["beta"], p["K"]

    def loss_fn(it):
        xi = 1 - torch.exp(-eta / L_MAP)
        # g = L^2 xi^2/2 + xi^3 N(xi): g(0)=0, g'(0)=0, g''(0)=1, g'(inf)=0 automatically
        g = 0.5 * L_MAP**2 * xi**2 + xi**3 * gnet(2 * xi - 1)
        g1 = d(g, eta); g2 = d(g1, eta); g3 = d(g2, eta)
        f, _ = fields(base_net, eta, S)
        f1 = d(f, eta); f2 = d(f1, eta); f3 = d(f2, eta)
        r = ((1 - b * f**2) * g3 + (2 * b * f * f1 + f) * g2
             + (2 * b * f * f2 - 2 * f1 - K + alpha) * g1
             + (2 * b * f1 * f2 - 2 * b * f * f3 + f2) * g)
        return (r**2).mean()

    train(gnet, loss_fn, extra_params=[alpha], adam_iters=3000, lbfgs_iters=1000, log=False)
    return alpha.item()


# ---------------------------------------------------------------- driver
if __name__ == "__main__":
    t0 = time.time()
    base = dict(beta=0.1, K=0.1, S=1.95, Pr=2.0, A=0.1, B=0.1)  # paper baseline

    print("=== Stage 1: dual solutions at the paper's baseline ===")
    up_net, up = solve_branch(base, "upper")
    lo_net, lo = solve_branch(base, "lower")
    print(f"upper: CfRe^1/2={up['Cf']:.6f} (paper Table 1: 1.93523180)  "
          f"theta'(0)={up['theta1']:.6f} (paper Table 2: -3.35807824)")
    print(f"lower: CfRe^1/2={lo['Cf']:.6f} (paper Table 1: 0.75244081)  "
          f"theta'(0)={lo['theta1']:.6f} (paper Table 2: -3.14427924)")

    print("\n=== Stage 2: trace the fold (S unknown, f''(0) prescribed) ===")
    rows = trace_fold(base, [2.4, 2.0, 1.6, 1.3, 1.1, 0.9, 0.7, 0.5])
    Sc = min(r[1] for r in rows)
    print(f"estimated critical S_c ~ {Sc:.4f} for beta=0.1 (paper: 1.8391)")

    print("\n=== Stage 3: stability (beta=0.1, K=0.1, S=2.0) ===")
    p2 = dict(base, S=2.0, B=0.0)
    u2, _ = solve_branch(p2, "upper")
    l2, _ = solve_branch(p2, "lower")
    a_up = smallest_eigen(u2, p2, 1.0)
    a_lo = smallest_eigen(l2, p2, -0.1)
    print(f"alpha_1 upper = {a_up:.5f}  (paper Table 3: 0.8810518)")
    print(f"alpha_1 lower = {a_lo:.5f}  (paper Table 3: -0.0376726)")
    print(f"\ntotal time {time.time()-t0:.0f}s on {DEV}")
