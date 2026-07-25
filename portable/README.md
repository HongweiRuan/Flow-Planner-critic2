# Balanced dataset expansion + round-robin (passes-mode) collection — portable

Reusable pipeline to build a **category-balanced** scenario filter and collect a
**large, balanced** replay for critic training, on any cluster. High-horizon
collections are **sharded** across parallel jobs and merged.

## Why round-robin / passes mode
A 15s scenario yields ~`150/k` AQC transitions (k=1 → ~150, k=80 → ~2). The old
`--target N` ring-buffer stopped as soon as the buffer hit N, so at small k it
covered only the first few scenarios/categories → **not balanced**. Passes mode
(`--passes P`) runs **every scenario exactly P times**, so the set is balanced by
construction regardless of k or stop point.

## Prerequisites on the target cluster
- The `Flow-Planner-critic` repo (this bundle plugs into it). `collect.py` must have
  the `--shard-index/--shard-count` patch (see "collect.py patch" below).
- nuPlan v1.1 dataset (trainval split + maps), the Flow-Planner original ckpt
  (`model.pth` + `model_config.yaml`), and the `nuplan-critic-rl` conda env.
- A train-split log list JSON (e.g. `nuplan_train.json`) disjoint from val14.

## Files here
| file | role |
|---|---|
| `build_balanced_filter.py` | scan train logs → balanced filter (`<OUT_TAG>.yaml` + token2db + summary) |
| `collect_balanced_template.yaml` | collect config; fill the `<PLACEHOLDER>` paths |
| `collect_shard.sh` | portable single-(k,shard) collector; wrap in your scheduler |
| `merge_replays.py` | merge shard replays → one balanced replay (with disjointness self-check) |

## Step 1 — build the balanced filter (CPU only)
```bash
export DATA_ROOT=<NUPLAN_DATASET_ROOT>/nuplan-v1.1/splits/trainval
export MAP_ROOT=<NUPLAN_DATASET_ROOT>/maps
export TRAIN_JSON=<path>/nuplan_train.json
export OUT_DIR=<repo>/critic-training-data
export VAL14_YAML=<repo>/flow_planner/nuplan_simulation/scenario_filter/val14.yaml   # optional, for overlap check
export N_PER_TYPE=72 N_LOGS=15000 OUT_TAG=critic_train_balanced_1008
python build_balanced_filter.py
# then install it into the hydra config group so `scenario_filter=<OUT_TAG>` resolves:
cp $OUT_DIR/critic_train_balanced_1008.yaml <repo>/flow_planner/nuplan_simulation/scenario_filter/
```
Check `<OUT_TAG>_summary.json`: `val14_overlap_count` must be 0; rare categories may
land below `N_PER_TYPE` (that's the max available — still the balanced ceiling).

## Step 2 — collect 100k balanced per k (GPU)
Fill `collect_balanced_template.yaml` paths and point its `scenario_filter=` at your
`OUT_TAG`. Then, **one scheduler job per (k, shard)** running `collect_shard.sh`:

| k | passes | shards | jobs | ~transitions |
|---|--------|--------|------|--------------|
| 1  | 1  | 1 | 1 | ~150k |
| 10 | 7  | 1 | 1 | ~106k |
| 20 | 14 | 1 | 1 | ~106k |
| 40 | 27 | 2 | 2 | ~102k |
| 80 | 54 | 4 | 4 | ~102k |

(passes chosen so `passes × 1008 × 150/k ≳ 100k`; recompute if your filter has a
different scenario count.) Example launch of one shard:
```bash
export REPO=<repo> COLLECT_CFG=$REPO/config/collect_balanced.yaml
export DEST=<persistent>/aqc1008 K=80 PASSES=54 SHARD_INDEX=2 SHARD_COUNT=4
bash collect_shard.sh          # -> $DEST/replay_h80_s2of4.zarr
```
Contiguous shard ranges are whole passes, so each shard stays balanced and
`episode_id`s never collide across shards (safe to merge).

## Step 3 — merge shards → one replay per k
```bash
# k=1/10/20 are single-shard: just use replay_h${k}_s0of1.zarr directly (or rename).
python merge_replays.py $DEST/replay_h40.zarr $DEST/replay_h40_s0of2.zarr $DEST/replay_h40_s1of2.zarr
python merge_replays.py $DEST/replay_h80.zarr $DEST/replay_h80_s{0,1,2,3}of4.zarr
```
`merge_replays.py` prints `dup_(ep,step)_keys=0` — if it's not 0, the shards were not
disjoint and it aborts. Feed `replay_h${k}.zarr` to your trainer.

## collect.py patch (if your repo copy lacks it)
Add two args and shard the passes-mode episode range:
```python
ap.add_argument("--shard-index", type=int, default=0)
ap.add_argument("--shard-count", type=int, default=1)
...
# inside the `if passes:` block, replace `for j, eid in enumerate(range(total_eps))`:
C = max(1, int(args.shard_count)); S = int(args.shard_index) % C
lo = S * total_eps // C; hi = (S + 1) * total_eps // C
for j, eid in enumerate(range(lo, hi)):
    assignments[j % workers].append(eid)
episodes_done = hi - lo
```
