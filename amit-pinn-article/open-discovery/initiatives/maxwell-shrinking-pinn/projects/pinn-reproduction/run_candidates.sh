#!/bin/bash
# Sequential GPU evaluation of candidates with the UNMODIFIED evaluator.
# usage: run_candidates.sh <baseline_full_seconds> candidate-NNN [candidate-NNN ...]
# Writes candidates/<c>/smoke.json and (if smoke valid) candidates/<c>/full.json.
set -u
cd "$(dirname "$0")"
source /workspace/.venv-pinn/bin/activate
BASE=$1; shift
for c in "$@"; do
  d=candidates/$c
  [ -s $d/full.json ] && { echo "$c already evaluated, skipping"; continue; }
  python evaluator.py $d/pinn_maxwell_shrinking.py > $d/smoke.json 2> $d/smoke.stderr.log
  ok=$(python -c "import json;print(json.load(open('$d/smoke.json'))['valid'])")
  echo "$c smoke valid=$ok"
  [ "$ok" = "True" ] || continue
  python evaluator.py $d/pinn_maxwell_shrinking.py --mode full --oracle --fold --stability \
     --baseline-seconds $BASE > $d/full.json 2> $d/full.stderr.log
  python summarize_run.py $d/full.json
done
