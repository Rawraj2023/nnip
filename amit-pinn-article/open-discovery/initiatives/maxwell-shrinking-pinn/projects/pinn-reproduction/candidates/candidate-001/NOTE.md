# candidate-001 — Taylor-mode derivative jets

parent: baseline (baseline-lock-gpu, sha256 of baseline file 8f145d68…b180)
mechanism class: A/C (same-hardware throughput, GPU-native restructuring; behavior-preserving)

## Hypothesis
Every optimizer step of the baseline builds f', f'', f''' (and theta', theta'', g', g'', g''')
with three nested `torch.autograd.grad(create_graph=True)` passes, then backpropagates
through that nested graph. For a 4x40 tanh MLP on 300 points this is pure kernel-launch
overhead on GPU. Propagating the value and its first k eta-derivatives forward in closed
form (Faa di Bruno through tanh, Leibniz for the trial-function products) gives the
identical residual with a much smaller graph -> fewer kernels per step -> lower wall-clock.

## Changes (only `pinn_maxwell_shrinking.py`)
- New helpers `_mlp_jet`, `_xi_jet`, `_mul`, `f_jet`, `theta_jet`.
- `residuals(..., which="both")` computes via jets; optional `which` lets the momentum
  stage skip the (unused) energy residual and vice-versa. Public contract unchanged.
- Branch anchor and fold anchor use `f_jet(...)[2]` at eta=0 instead of nested autograd.
- `smallest_eigen`: frozen base-flow jet computed once under no_grad; g jet in closed form.
- Unchanged: architecture, init, seeds, collocation, iteration counts, optimizer settings,
  auto-scale pilot, `fields`, `wall_values`, `train`.

## Pre-evaluation check (not an evaluator result)
On random weights, jet residuals and their parameter gradients equal the baseline autograd
values to <=2e-14 (float64); eigen trial-function jet equal to <=1.1e-15.
Expected: same converged numbers as baseline up to floating-point path differences.
