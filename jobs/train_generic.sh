#!/usr/bin/env bash
set -eo pipefail
module load conda/4.11.0
module load cuda/12.6.0
conda activate nuplan-critic-rl
FORK=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2
cd "$FORK"; export PYTHONPATH="$FORK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
: "${H:?}"; : "${REPLAY:?}"; : "${CKPT:?}"
echo "host=$(hostname) H=$H updates=${UPDATES:-4000} replay=$REPLAY"
nvidia-smi -L || true
python -u train.py --config config/eval.yaml --replay "$REPLAY" \
  --visible-horizon "$H" --checkpoint "$CKPT" \
  --updates "${UPDATES:-4000}" --warmup 200 --batch-size "${BS:-48}" \
  --bootstrap-candidates "${KBOOT:-6}" --log-interval 25
echo "TRAINGEN_${TAG:-q$H}_DONE_OK -> $CKPT"
