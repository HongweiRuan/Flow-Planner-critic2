#!/usr/bin/env bash
# Track A finisher: q1-q40 evals already submitted manually. Submit q80 eval when
# its checkpoint lands, wait for all 5 bal evals, print the Track-A table.
module load conda/4.11.0 >/dev/null 2>&1
conda activate nuplan-critic-rl
FORK=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2
RUN=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/critic2_runs
CK=$RUN/checkpoints_balanced
cd "$FORK"; export PYTHONPATH="$FORK"
HS="1 10 20 40 80"

echo "[balA] waiting for q80.pt ..."
while [ ! -f "$CK/q80.pt" ]; do sleep 60; done
# submit q80 eval only if not already running/done
if ! ls $RUN/evalbal_q80.*.out >/dev/null 2>&1; then bsub < jobs_evalbal_q80.lsf; fi
echo "[balA] q80 eval submitted; waiting for all 5 bal evals ..."
while :; do
  c=0; for H in $HS; do ls $RUN/evalbal_q${H}.*.out >/dev/null 2>&1 && grep -qE "EVAL_bal_q${H}_.*DONE_OK" $RUN/evalbal_q${H}.*.out 2>/dev/null && c=$((c+1)); done
  [ "$c" -ge 5 ] && break; sleep 120
done
echo "============ FINAL TRACK A (look-ahead, retrained 20k updates) VAL14 ============"
echo "(planner=model.pth, balanced train, encoder-finetuned, commit=1 look-ahead=k)"
for T in base bal_q1 bal_q10 bal_q20 bal_q40 bal_q80; do python3 agg_shards.py $T 1 2>/dev/null | head -1; done
echo "ALL_BALANCED_DONE"
