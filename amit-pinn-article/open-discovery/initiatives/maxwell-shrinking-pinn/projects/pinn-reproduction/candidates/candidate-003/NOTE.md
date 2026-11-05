# candidate-003 — extend collocation reach
parent: baseline · mechanism B4 (accuracy, lower branch)
Hypothesis: `collocation(xi_max=0.9995)` only enforces the residual to eta = -L ln(5e-4) ~ 7.6 L
(~10 for the lower branch), while the lower branch's slow decay (lambda~0.77) still matters to
eta~20 (oracle: lower Cf changes 0.17 % between eta_inf=10 and 20). Raising xi_max to 0.99995
(eta reach ~13 L/1.3) should remove the far-field error floor.
CPU pre-check (300-iter LBFGS cap, 1 seed): lower Cf err vs eta_inf->inf oracle 0.19 % -> 0.018 %,
upper Nu err 0.17 % -> 0.019 %. Change: one default argument.
