#!/usr/bin/env bash
set -eo pipefail
module load conda/4.11.0
module load cuda/12.6.0
conda activate nuplan-critic-rl
FORK=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2
DEST=$FORK/critic-training-data/aqc1008
cd "$FORK"; export PYTHONPATH="$FORK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAY_DEDUP_LOGS=1
mkdir -p "$DEST"
: "${K:?set K}"; : "${PASSES:?set PASSES}"
SC=${SHARD_COUNT:-1}; SI=${SHARD_INDEX:-0}
OUT=$DEST/replay_aqc1008_h${K}_s${SI}of${SC}.zarr

# Idempotent + dedup across queues: skip if this shard is already collected, or if
# another job holds the atomic lock (mkdir is atomic).
if [ -d "$OUT" ]; then echo "shard $OUT already collected; skipping"; exit 0; fi
LOCK=$DEST/.lock_h${K}_s${SI}of${SC}
if ! mkdir "$LOCK" 2>/dev/null; then echo "shard h${K} s${SI}/${SC} is being collected elsewhere; skipping"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

echo "host=$(hostname) AQC1008 K=$K passes=$PASSES shard=$SI/$SC -> $OUT"; nvidia-smi -L || true

BASE=/tmp/collect_aqc1008/replay_s${SI}.zarr
python -u collect.py --config config/collect_aqc_1008.yaml \
  --execution-horizon "$K" --passes "$PASSES" \
  --shard-index "$SI" --shard-count "$SC" --out "$BASE"

SRC=/tmp/collect_aqc1008/replay_s${SI}_h${K}.zarr
cp -a "$SRC" "$OUT.tmp"
rm -rf "$OUT"
mv "$OUT.tmp" "$OUT"
python -u -c "from flow_planner.critic_rl.replay import ZarrReplayReader as R; print('h${K} s${SI}/${SC} persisted transitions:', len(R('$OUT')))"
echo "COLLECT1008_h${K}_s${SI}of${SC}_DONE_OK -> $OUT"
