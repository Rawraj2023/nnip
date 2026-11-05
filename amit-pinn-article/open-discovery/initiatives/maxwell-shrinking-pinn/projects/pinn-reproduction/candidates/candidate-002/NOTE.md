# candidate-002 — adaptive LBFGS stopping
parent: baseline · mechanism A3 (throughput)
Hypothesis: LBFGS tolerances (1e-12 grad / 1e-15 change) never trigger, so every stage burns
its full fixed budget. A CPU trace of the branch solve (scratchpad conv_trace) showed momentum
converges in ~100 LBFGS evals and energy in ~200; the remaining ~1700/1050 evals move Cf/Nu by
<1e-5 relative. Run the same LBFGS in 50-iteration chunks (history persists) and stop when a
chunk improves the loss by <1 %, never exceeding the original `lbfgs_iters`.
Change: `train()` LBFGS phase only. Risk: eigen/fold stages may converge more slowly than
the branch solve; full-mode --fold/--stability gates check that.
