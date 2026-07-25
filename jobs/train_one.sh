#!/usr/bin/env bash
set -eo pipefail
module load conda/4.11.0
module load cuda/12.6.0
conda activate nuplan-critic-rl

FORK=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2
REPLAY=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2/critic-training-data/replay_balanced_h1.zarr
OUT=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/critic2_runs/checkpoints_balanced
cd "$FORK"
export PYTHONPATH="$FORK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUT"

: "${H:?set H (visible horizon)}"
echo "host=$(hostname) H=$H updates=${UPDATES:-4000} bs=${BS:-48} K=${K:-6}"
nvidia-smi -L || true

python -u train.py --config config/eval.yaml --replay "$REPLAY" \
  --visible-horizon "$H" --checkpoint "$OUT/q${H}.pt" \
  --updates "${UPDATES:-4000}" --warmup 200 --batch-size "${BS:-48}" \
  --bootstrap-candidates "${K:-6}" --log-interval 25

echo "TRAIN_Q${H}_DONE_OK -> $OUT/q${H}.pt"
