# critic-training-data

Reusable, balanced, TRAIN-split scenario filter for critic data collection.
Provably disjoint from the val14 eval set (val14_overlap_count = 0).

## Files
- critic_train_balanced.yaml          nuPlan ScenarioFilter: 70 explicit scenario
                                       tokens = 14 val14 categories x 5 each, train
                                       split (nuplan_train.json). Also copied to
                                       flow_planner/nuplan_simulation/scenario_filter/
                                       so it loads via  scenario_filter=critic_train_balanced
- critic_train_balanced_token2db.json  token -> db basename (fast per-worker build)
- critic_train_balanced_summary.json   per-category counts + val14 overlap (=0)
- collect_balanced.yaml                collect config using this filter (train split,
                                       execution_horizon=1); mirror of config/collect_balanced.yaml

## Use
  python collect.py --config config/collect_balanced.yaml     # -> /tmp/collect_balanced/replay_h1.zarr

## Regenerate (one-time scan)
  N_LOGS=6000 N_PER_TYPE=5 python build_balanced_filter.py     # (via jobs_build_filter.lsf on batch_cpu)
