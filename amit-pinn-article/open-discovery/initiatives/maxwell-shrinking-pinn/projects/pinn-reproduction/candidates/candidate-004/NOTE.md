# candidate-004 — cheaper auto-scale pilot
parent: baseline · mechanism A2 (throughput)
Hypothesis: the pilot only needs f(inf) to pick L = ceil(lam)/lam. CPU pre-check (seed 0):
pilot budgets 2500/200, 1500/100, 800/100, 400/60 give lower L 1.3111/1.3095/1.3110/1.3105 and
upper L 1.1148/1.1401/1.1112/1.1445 — the choice is insensitive to budget. Cut pilot from
2500 Adam + 200 LBFGS to 800 + 100 (~3x cheaper pilot, ~24 % of branch cost originally).
Change: one line in `_solve_branch(quick=True)`.
