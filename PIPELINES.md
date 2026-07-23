# Critic pipelines

Two pipelines, two entry scripts, two config files. That's it.

| I want to...                       | Run                                          | Config              |
|------------------------------------|----------------------------------------------|---------------------|
| Collect critic training data       | `python collect.py --config config/collect.yaml`  | `config/collect.yaml` |
| Evaluate on val14 (closed-loop)    | `python evaluate.py --config config/eval.yaml`    | `config/eval.yaml`    |

Both scripts print `--help` with copy-paste examples. Both read a single
self-contained YAML (no config inheritance) — edit paths directly in the file.

## Collect

```bash
python collect.py --config config/collect.yaml                       # defaults from the config
python collect.py --config config/collect.yaml --execution-horizon 10 --target 20000 --workers 64
```

- Rolls out the planner across nuPlan scenarios in parallel and saves
  transitions (scene + N candidates + reward) to a zarr replay buffer.
- Stops once the buffer holds `--target` transitions.
- Each execution horizon K writes its own file: `replay.zarr` → `replay_h<K>.zarr`.

## Evaluate

```bash
python evaluate.py --config config/eval.yaml --scorer candidate0     # baseline: raw planner
python evaluate.py --config config/eval.yaml --scorer random         # random candidate
python evaluate.py --config config/eval.yaml --scorer critic \
    --checkpoint /path/to/critic.pt --visible-horizon 10             # a trained critic
```

- Runs val14 closed-loop, sharded across workers by scenario token (each worker
  builds only its slice and opens only its dbs — this is what keeps it fast).
- Writes `metrics_per_scenario.csv`, `metrics_per_scenario.json`, and
  `summary.json` to `output_dir`; prints a running mean.

## What's a stub

The critic **model** (`flow_planner/critic_rl/critic.py`) and its **training
algorithm** (`flow_planner/critic_rl/trainer.py`) are intentionally left as
stubs — fill them in. `collect.py` needs neither. `evaluate.py` needs them only
for `--scorer critic`; `candidate0` and `random` work today.

## Layout

```
collect.py, evaluate.py            # the two entry points
config/collect.yaml, eval.yaml     # one flat config each
config/val14_token2db.json         # token -> db map used for eval sharding
flow_planner/critic_rl/
  factory.py    # builds the nuPlan env (planner + simulator) from a config
  env.py        # one step = commit K sim steps of a chosen candidate
  reward.py, dense_reward.py   # per-step reward + official nuPlan score
  replay.py     # zarr replay buffer (writer + reader/sampler)
  workers.py    # Ray actors: CollectorWorker, EvaluationWorker
  types.py      # CandidateBatch / CriticObservation / Transition
  critic.py     # critic model architecture  ---- STUB
  trainer.py    # critic training algorithm   ---- STUB
```

Candidate sampling itself lives in the planner (patched from the base repo):
`planner.py` → `core/flow_matching_core.py` → `model/flow_planner_model/flow_planner.py`
(`compute_candidate_trajectories` / `forward_inference_candidates`).
