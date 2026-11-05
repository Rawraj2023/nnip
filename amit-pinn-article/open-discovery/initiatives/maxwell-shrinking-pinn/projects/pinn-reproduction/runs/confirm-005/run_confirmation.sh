#!/bin/bash
# Winner confirmation for candidate-005 (standard plan). Serial, idle GPU, unmodified evaluator.
# 1) smoke on the rebuilt winner  2) paired full runs: locked GPU baseline, then rebuilt winner
# 3) fresh-parameter multi-seed check (Table 1 rows unused in search; seeds 1-3; search used seed 0)
set -u
cd "$(dirname "$0")/../.."
source /workspace/.venv-pinn/bin/activate
C=runs/confirm-005
nvidia-smi > $C/nvidia-smi.txt 2>&1; python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())" > $C/env.txt
date -u +"start %FT%TZ" | tee $C/timeline.txt
python evaluator.py $C/pinn_maxwell_shrinking.py > $C/smoke.json 2> $C/smoke.stderr.log
echo "smoke valid=$(python -c "import json;print(json.load(open('$C/smoke.json'))['valid'])")"; date -u +"smoke done %FT%TZ" >> $C/timeline.txt
python evaluator.py baseline-lock-gpu/baseline/pinn_maxwell_shrinking.py --mode full --oracle --fold --stability > $C/baseline_full_paired.json 2> $C/baseline_full_paired.stderr.log
date -u +"baseline full done %FT%TZ" >> $C/timeline.txt; python summarize_run.py $C/baseline_full_paired.json
BASE=$(python -c "import json;print(json.load(open('$C/baseline_full_paired.json'))['objective']['value'])")
python evaluator.py $C/pinn_maxwell_shrinking.py --mode full --oracle --fold --stability --baseline-seconds $BASE > $C/winner_full_paired.json 2> $C/winner_full_paired.stderr.log
date -u +"winner full done %FT%TZ" >> $C/timeline.txt; python summarize_run.py $C/winner_full_paired.json
python confirm_fresh.py $C/pinn_maxwell_shrinking.py 1 2 3 > $C/fresh_seeds123.json 2> $C/fresh_seeds123.stderr.log
date -u +"fresh done %FT%TZ" >> $C/timeline.txt; cat $C/fresh_seeds123.stderr.log | tail -9; cat $C/timeline.txt
