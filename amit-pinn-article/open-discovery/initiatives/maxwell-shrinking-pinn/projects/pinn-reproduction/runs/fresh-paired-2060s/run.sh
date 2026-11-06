#!/bin/bash
# Item A (runs/OPEN-ITEMS-PREREGISTRATION.md): baseline vs candidate-005 on confirm_fresh.py, seeds 1-3, serial, same GPU.
set -u
cd "$(dirname "$0")/../.."
source /workspace/.venv-pinn/bin/activate
C=runs/fresh-paired-2060s
nvidia-smi > $C/nvidia-smi.txt 2>&1
python -c "import torch,scipy;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),'scipy',scipy.__version__)" > $C/env.txt
sha256sum baseline-lock-gpu/baseline/pinn_maxwell_shrinking.py runs/confirm-005/pinn_maxwell_shrinking.py > $C/sha256.txt
date -u +"start %FT%TZ" > $C/timeline.txt
python confirm_fresh.py baseline-lock-gpu/baseline/pinn_maxwell_shrinking.py 1 2 3 > $C/baseline_seeds123.json 2> $C/baseline_seeds123.stderr.log
date -u +"baseline done %FT%TZ" >> $C/timeline.txt
python confirm_fresh.py runs/confirm-005/pinn_maxwell_shrinking.py 1 2 3 > $C/c005_seeds123.json 2> $C/c005_seeds123.stderr.log
date -u +"c005 done %FT%TZ" >> $C/timeline.txt
