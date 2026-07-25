#!/usr/bin/env bash
# Track-B v2 (AQC, 100k balanced per k, 1008-scenario filter) full pipeline:
#   wait 1008 filter -> install into hydra group -> collect (passes mode, high-k sharded)
#   -> merge shards -> train Q^k on 100k -> eval val14 -> aggregate table.
module load conda/4.11.0 >/dev/null 2>&1
conda activate nuplan-critic-rl
FORK=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2
RUN=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/critic2_runs
CTD=$FORK/critic-training-data
DEST=$CTD/aqc1008
SF=$FORK/flow_planner/nuplan_simulation/scenario_filter
CK=$RUN/checkpoints_aqc1008
cd "$FORK"; export PYTHONPATH="$FORK"; mkdir -p "$DEST" "$CK"
KS="1 10 20 40 80"
# per-k passes (>=100k transitions; yield ~ 150/k over 1008 scenarios) and shard counts
declare -A P=( [1]=1 [10]=7 [20]=14 [40]=27 [80]=54 )
declare -A SH=( [1]=1 [10]=1 [20]=1 [40]=2 [80]=4 )

echo "[v2] $(date) waiting for 1008 filter ..."
while [ ! -f "$CTD/critic_train_balanced_1008.yaml" ]; do sleep 60; done
# install into the hydra scenario_filter config group so `scenario_filter=critic_train_balanced_1008` resolves
cp -f "$CTD/critic_train_balanced_1008.yaml" "$SF/critic_train_balanced_1008.yaml"
echo "[v2] filter installed. per-category:"; python3 -c "import json;d=json.load(open('$CTD/critic_train_balanced_1008_summary.json'));print(' n=',d['n_scenarios'],'overlap=',d['val14_overlap_count']);print(d['per_category_counts'])"

# ---- submit collection jobs (batch_a100_mig), one per (k, shard) ----
echo "[v2] submitting collection jobs ..."
for K in $KS; do
  for SI in $(seq 0 $(( ${SH[$K]} - 1 )) ); do
    J=$RUN/coll1008_h${K}_s${SI}.lsf
    cat > "$J" <<LSF
#!/usr/bin/env bash
#BSUB -J c2_coll1008_h${K}_s${SI}
#BSUB -q batch_a100_mig
#BSUB -n 10
#BSUB -R "span[hosts=1]"
#BSUB -M 12000
#BSUB -W 12:00
#BSUB -gpu "num=1:mig=4:mps=yes"
#BSUB -P BH-000425-08-09
#BSUB -oo $RUN/coll1008_h${K}_s${SI}.%J.out
#BSUB -eo $RUN/coll1008_h${K}_s${SI}.%J.err
export K=${K} PASSES=${P[$K]} SHARD_INDEX=${SI} SHARD_COUNT=${SH[$K]}
bash $FORK/collect_aqc1008_one.sh
LSF
    bsub < "$J"
  done
done

# ---- wait for all shards ----
echo "[v2] waiting for all collection shards ..."
while :; do
  done=0; total=0
  for K in $KS; do for SI in $(seq 0 $(( ${SH[$K]} - 1 )) ); do
    total=$((total+1))
    [ -d "$DEST/replay_aqc1008_h${K}_s${SI}of${SH[$K]}.zarr" ] && done=$((done+1))
  done; done
  echo "[v2] shards $done/$total $(date)"
  [ "$done" -ge "$total" ] && break; sleep 180
done

# ---- merge shards -> canonical replay per k ----
echo "[v2] merging shards ..."
for K in $KS; do
  C=${SH[$K]}
  MERGED=$DEST/replay_aqc1008_h${K}.zarr
  if [ "$C" -eq 1 ]; then
    [ -d "$MERGED" ] || cp -a "$DEST/replay_aqc1008_h${K}_s0of1.zarr" "$MERGED"
  else
    if [ ! -d "$MERGED" ]; then
      SHARDS=""; for SI in $(seq 0 $((C-1)) ); do SHARDS="$SHARDS $DEST/replay_aqc1008_h${K}_s${SI}of${C}.zarr"; done
      python -u merge_replays.py "$MERGED" $SHARDS
    fi
  fi
  python -u -c "from flow_planner.critic_rl.replay import ZarrReplayReader as R;print('h${K} merged transitions:',len(R('$MERGED')))"
done

# ---- train Q^k on the merged 100k replays (UPDATES=20000, same regime as Track A retrain) ----
echo "[v2] submitting trainings ..."
for K in $KS; do
  J=$RUN/train1008_q${K}.lsf
  cat > "$J" <<LSF
#!/usr/bin/env bash
#BSUB -J c2_tr1008_q${K}
#BSUB -q batch_a100_mig
#BSUB -n 10
#BSUB -R "span[hosts=1]"
#BSUB -M 12000
#BSUB -W 12:00
#BSUB -gpu "num=1:mig=4:mps=yes"
#BSUB -P BH-000425-08-09
#BSUB -oo $RUN/train1008_q${K}.%J.out
#BSUB -eo $RUN/train1008_q${K}.%J.err
export H=${K} REPLAY=$DEST/replay_aqc1008_h${K}.zarr CKPT=$CK/q${K}.pt UPDATES=20000 BS=48 KBOOT=6 TAG=aqc1008_q${K}
bash $FORK/train_generic.sh
LSF
  bsub < "$J"
done

echo "[v2] waiting for 5 checkpoints ..."
while :; do n=0; for K in $KS; do [ -f "$CK/q${K}.pt" ] && n=$((n+1)); done; [ "$n" -ge 5 ] && break; sleep 180; done

# ---- eval val14 ----
echo "[v2] submitting 5 val14 evals ..."
for K in $KS; do
  J=$RUN/eval1008_q${K}.lsf
  cat > "$J" <<LSF
#!/usr/bin/env bash
#BSUB -J c2_ev1008_q${K}
#BSUB -q batch_a100_mig
#BSUB -n 10
#BSUB -R "span[hosts=1]"
#BSUB -M 14000
#BSUB -W 12:00
#BSUB -gpu "num=1:mig=4:mps=yes"
#BSUB -P BH-000425-08-09
#BSUB -oo $RUN/eval1008_q${K}.%J.out
#BSUB -eo $RUN/eval1008_q${K}.%J.err
export TAG=aqc1008_q${K} SCORER=critic CKPT=$CK/q${K}.pt H=${K} NC=16 WORKERS=8 EPISODES=1118
bash $FORK/eval_one.sh
LSF
  bsub < "$J"
done

echo "[v2] waiting for 5 evals ..."
while :; do c=0; for K in $KS; do ls $RUN/eval1008_q${K}.*.out >/dev/null 2>&1 && grep -qE "EVAL_aqc1008_q${K}_.*DONE_OK" $RUN/eval1008_q${K}.*.out 2>/dev/null && c=$((c+1)); done; [ "$c" -ge 5 ] && break; sleep 180; done

echo "===================== FINAL AQC-v2 (100k balanced, 1008 filter) VAL14 ====================="
for T in base aqc1008_q1 aqc1008_q10 aqc1008_q20 aqc1008_q40 aqc1008_q80; do python3 agg_shards.py $T 1 2>/dev/null | head -1; done
echo "ALL_AQC1008_DONE"
