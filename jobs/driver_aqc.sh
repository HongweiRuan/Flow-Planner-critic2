#!/usr/bin/env bash
# AQC line orchestration: wait 5 k-step replays -> train Q^k on each -> eval val14 -> aggregate.
module load conda/4.11.0 >/dev/null 2>&1
conda activate nuplan-critic-rl
FORK=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2
RUN=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/critic2_runs
AQCD=$FORK/critic-training-data/aqc
CKA=$RUN/checkpoints_aqc
cd "$FORK"; export PYTHONPATH="$FORK"; mkdir -p "$CKA"
KS="1 10 20 40 80"

echo "[aqc] waiting for 5 k-step replays ... $(date)"
while :; do n=0; for K in $KS; do [ -d "$AQCD/replay_aqc_h${K}.zarr" ] && n=$((n+1)); done; [ "$n" -ge 5 ] && break; sleep 180; done
echo "[aqc] all replays ready $(date); submitting 5 trainings"
for K in $KS; do bsub < jobs_trainaqc_q${K}.lsf; done

echo "[aqc] waiting for 5 checkpoints ..."
while :; do n=0; for K in $KS; do [ -f "$CKA/q${K}.pt" ] && n=$((n+1)); done; [ "$n" -ge 5 ] && break; sleep 180; done
echo "[aqc] all trained $(date); submitting 5 val14 evals"
for K in $KS; do bsub < jobs_evalaqc_q${K}.lsf; done

echo "[aqc] waiting for 5 evals ..."
while :; do c=0; for K in $KS; do ls $RUN/evalaqc_q${K}.*.out >/dev/null 2>&1 && grep -qE "EVAL_aqc_q${K}_.*DONE_OK" $RUN/evalaqc_q${K}.*.out 2>/dev/null && c=$((c+1)); done; [ "$c" -ge 5 ] && break; sleep 180; done

echo "===================== FINAL AQC VAL14 RESULTS ====================="
echo "(commit-horizon critics: trained on execution_horizon=k data, k-step reward target)"
for T in base aqc_q1 aqc_q10 aqc_q20 aqc_q40 aqc_q80; do python3 agg_shards.py $T 1 2>/dev/null | head -1; done
echo "ALL_AQC_DONE"
