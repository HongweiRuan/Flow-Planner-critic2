#!/usr/bin/env bash
set -eo pipefail
module load conda/4.11.0
module load cuda/12.6.0
conda activate nuplan-critic-rl
FORK=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2
DEST=$FORK/critic-training-data/aqc
cd "$FORK"; export PYTHONPATH="$FORK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAY_DEDUP_LOGS=1
mkdir -p "$DEST"
: "${K:?set K (execution/commit horizon)}"

# Idempotent + dedup: same k may be queued on multiple queues (h200 AND a100_mig).
# Skip if already collected, or if another job holds the atomic lock (mkdir is atomic).
if [ -d "$DEST/replay_aqc_h${K}.zarr" ]; then echo "h${K} already collected; skipping"; exit 0; fi
if ! mkdir "$DEST/.lock_h${K}" 2>/dev/null; then echo "h${K} is being collected by another job; skipping"; exit 0; fi
trap 'rmdir "$DEST/.lock_h${K}" 2>/dev/null || true' EXIT

echo "host=$(hostname) AQC collect K=$K target=10000"; nvidia-smi -L || true

python -u collect.py --config config/collect_aqc.yaml --execution-horizon "$K" --target 10000

SRC=/tmp/collect_aqc/replay_h${K}.zarr
cp -a "$SRC" "$DEST/replay_aqc_h${K}.zarr.tmp"
rm -rf "$DEST/replay_aqc_h${K}.zarr"
mv "$DEST/replay_aqc_h${K}.zarr.tmp" "$DEST/replay_aqc_h${K}.zarr"
python -u -c "from flow_planner.critic_rl.replay import ZarrReplayReader as R; print('h${K} persisted transitions:', len(R('$DEST/replay_aqc_h${K}.zarr')))"
echo "COLLECT_AQC_h${K}_DONE_OK -> $DEST/replay_aqc_h${K}.zarr"
