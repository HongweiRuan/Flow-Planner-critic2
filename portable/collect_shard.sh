#!/usr/bin/env bash
# Portable single-shard balanced collector. Wrap this in your cluster's scheduler
# (one job per (K, SHARD_INDEX)). No module loads / no LSF -- activate your env first.
#
# Required env:
#   REPO        path to the Flow-Planner-critic repo (contains collect.py, patched with --shard-*)
#   COLLECT_CFG collect config yaml (see collect_balanced_template.yaml)
#   DEST        persistent dir for the output shard replay
#   K           execution/commit horizon (1,10,20,40,80)
#   PASSES      how many times to run EVERY scenario (round-robin balanced)
# Optional env:
#   SHARD_INDEX (default 0)   SHARD_COUNT (default 1)   contiguous episode sub-range
#   TMP         node-local scratch for the in-progress replay (default /tmp/collect_balanced)
set -eo pipefail
: "${REPO:?}"; : "${COLLECT_CFG:?}"; : "${DEST:?}"; : "${K:?}"; : "${PASSES:?}"
SI=${SHARD_INDEX:-0}; SC=${SHARD_COUNT:-1}
TMP=${TMP:-/tmp/collect_balanced}
cd "$REPO"; export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAY_DEDUP_LOGS=1
mkdir -p "$DEST"

OUT=$DEST/replay_h${K}_s${SI}of${SC}.zarr
if [ -d "$OUT" ]; then echo "shard $OUT already exists; skipping"; exit 0; fi
LOCK=$DEST/.lock_h${K}_s${SI}of${SC}          # atomic dedup if you queue duplicates
if ! mkdir "$LOCK" 2>/dev/null; then echo "shard held by another job; skipping"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

echo "host=$(hostname) K=$K passes=$PASSES shard=$SI/$SC -> $OUT"

BASE=$TMP/replay_s${SI}.zarr
python -u collect.py --config "$COLLECT_CFG" \
  --execution-horizon "$K" --passes "$PASSES" \
  --shard-index "$SI" --shard-count "$SC" --out "$BASE"

SRC=$TMP/replay_s${SI}_h${K}.zarr
cp -a "$SRC" "$OUT.tmp"; rm -rf "$OUT"; mv "$OUT.tmp" "$OUT"
python -u -c "from flow_planner.critic_rl.replay import ZarrReplayReader as R; print('h${K} s${SI}/${SC} transitions:', len(R('$OUT')))"
echo "COLLECT_h${K}_s${SI}of${SC}_DONE_OK -> $OUT"
