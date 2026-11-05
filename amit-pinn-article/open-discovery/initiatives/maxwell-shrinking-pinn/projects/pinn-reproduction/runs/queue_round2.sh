#!/bin/bash
# Wait for candidate-001's in-flight full evaluation (orphaned from the round-1 runner), then
# evaluate the round-2 combination first, then the remaining round-1 singles. Serial GPU use only.
cd "$(dirname "$0")/.."
while pgrep -f "evaluator.py candidates/candidate-001" >/dev/null; do sleep 15; done
source /workspace/.venv-pinn/bin/activate
python summarize_run.py candidates/candidate-001/full.json
./run_candidates.sh 1451.7908697790044 candidate-005 candidate-002 candidate-003 candidate-004
