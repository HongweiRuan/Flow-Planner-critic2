#!/usr/bin/env bash
set -eo pipefail
module load conda/4.11.0
module load cuda/12.6.0
conda activate nuplan-critic-rl

FORK=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2
RUN=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/critic2_runs
cd "$FORK"
export PYTHONPATH="$FORK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

: "${TAG:?}"; : "${SCORER:?}"; : "${NC:?}"
SI=${SHARD_INDEX:-0}; SC=${SHARD_COUNT:-1}
if [ "$SC" -gt 1 ]; then OUTDIR="$RUN/eval_${TAG}_s${SI}of${SC}"; else OUTDIR="$RUN/eval_${TAG}"; fi
CKARGS=""
if [ "$SCORER" = "critic" ]; then CKARGS="--checkpoint ${CKPT:?} --visible-horizon ${H:?}"; fi

echo "host=$(hostname) TAG=$TAG scorer=$SCORER nc=$NC workers=${WORKERS:-12} shard=$SI/$SC"
nvidia-smi -L || true
T0=$(date +%s)
python -u evaluate.py --config config/eval.yaml --scorer "$SCORER" $CKARGS \
  --num-candidates "$NC" --episodes "${EPISODES:-1118}" --workers "${WORKERS:-12}" \
  --shard-index "$SI" --shard-count "$SC" --output-dir "$OUTDIR"
T1=$(date +%s)
echo "EVAL_${TAG}_s${SI}of${SC}_DONE_OK elapsed=$((T1-T0))s -> $OUTDIR/summary.json"
