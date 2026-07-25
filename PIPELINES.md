# Flow-Planner Critic — pipelines

A **critic** that scores Flow-Planner candidate trajectories for closed-loop
autonomous driving, evaluated on the nuPlan **val14** benchmark. Given a scene and
the planner's *N* candidate maneuvers, the critic predicts each candidate's expected
return so we can pick a better one than the planner's default at test time.

Three entry scripts, one flat YAML each — no config inheritance, edit paths in the file.

| I want to…                         | Run                                                        |
|------------------------------------|-----------------------------------------------------------|
| Collect critic training data       | `python collect.py   --config config/collect.yaml`        |
| Train a critic                     | `python train.py     --config config/eval.yaml --replay …`|
| Evaluate on val14 (closed-loop)    | `python evaluate.py  --config config/eval.yaml --scorer critic --checkpoint …` |

All three print `--help` with copy-paste examples.

## Two experiment tracks

Both share one collection + training codebase; they differ only in the horizon the
transitions are collected at and what the critic is asked to predict.

| Track | Name | Collection | Critic predicts |
|-------|------|-----------|-----------------|
| **A** | **Look-ahead** | `execution_horizon=1` (commit 1 step, replan) | value of committing 1 step then bootstrapping; candidate truncated to a *visible* horizon *k* |
| **B** | **AQC / commit-horizon** | `execution_horizon=k` (commit *k* steps) | *k*-step accumulated real return of actually committing the candidate for *k* steps |

A 15 s scenario (~150 sim steps) yields **~150/k** transitions, so at k=80 you get ~2
per scenario — Track B needs many more rollouts per k than Track A.

## Data collection

`collect.py` rolls the planner out across nuPlan scenarios in parallel (Ray) and
writes transitions (**raw scene encoder-inputs** + the *N* candidates + reward +
done) to a zarr replay (schema v3, `flow_planner/critic_rl/replay.py`). Storing raw
encoder-inputs (not frozen tokens) lets the critic **fine-tune its own copy of the
scene encoder** during training.

Two modes:

- **Target mode** (`--target N`): stop once the buffer holds *N* transitions.
- **Balanced / round-robin mode** (`--passes P`): run **every scenario exactly P
  times** (distinct seeds → distinct rollouts). Balanced by construction across
  categories — this is what you want for a class-balanced dataset. *Target mode's
  early stop is not category-balanced when scenarios are grouped by type.*

**Sharding** (`--shard-index S --shard-count C`): splits the `[0, total_eps)` episode
range into *C* contiguous blocks (whole passes → each shard still balanced;
`episode_id`s stay disjoint across shards → safe to merge). Use it to spread a large
high-*k* collection over parallel jobs, then combine with `merge_replays.py` (which
self-checks that no `(episode, step)` key collides).

### Balanced scenario filter

`jobs/build_balanced_filter.py` scans **train-split** logs (disjoint from val14) and
picks *N* scenarios of each of the 14 val14 categories, emitting a reproducible
nuPlan `ScenarioFilter` (explicit token list) + a `token2db.json` for fast per-worker
rebuild + a summary with a val14-overlap check (**must be 0**). The shipped filter is
`critic-training-data/critic_train_balanced_1008.yaml` — **72/category × 14 = 1008
scenarios, val14 overlap 0**. Install it into the hydra config group so
`scenario_filter=<name>` resolves:
```
cp critic-training-data/<name>.yaml flow_planner/nuplan_simulation/scenario_filter/
```

## Training

`train.py` loads the original Flow-Planner checkpoint, builds a `HorizonCritic`
(`flow_planner/critic_rl/critic.py`) whose scene encoder is **warm-started from the
planner encoder and trainable** (`--freeze-encoder` to A/B it), and runs a plain
**online 1-step Q-learning** trainer (`flow_planner/critic_rl/trainer.py`): twin-Q,
`target = r + γ·max_{a'~planner(s')} Q_target(s',a')`, Polyak target, next candidates
re-inferred from the planner at each step, replay buffer to decorrelate. The encoder
gets a smaller LR than the heads.

## Evaluation

`evaluate.py` runs val14 closed-loop, **sharded by scenario token** across workers
(each worker builds only its slice / opens only its dbs — this is what keeps it fast).
Scorers: `candidate0` (raw planner baseline), `random`, `critic` (a trained
checkpoint). Writes `metrics_per_scenario.{csv,json}` + a summary; `agg_shards.py`
merges sharded eval outputs into one full-val14 number. Baseline (candidate0) val14
mean score ≈ **0.8807**.

## Repo layout

```
collect.py  evaluate.py  train.py       # entry points
agg_shards.py  merge_replays.py         # eval aggregation / replay-shard merge utils
tests_smoke_critic.py                   # end-to-end smoke test
config/                                 # flat YAML configs (collect*, eval)
flow_planner/
  critic_rl/                            # critic, trainer, replay, workers, factory, model_loader, types
  nuplan_simulation/scenario_filter/    # hydra scenario_filter config group (val14, balanced filters)
  model/ …                              # the Flow-Planner planner itself
critic-training-data/                   # balanced filter yamls/json + summaries (zarr replays are gitignored)
jobs/                                   # cluster-specific: LSF job files, orchestration drivers, RNG-path builders
portable/                               # cluster-agnostic, env-var-driven version of the balanced-collection pipeline
```

## Reproduce (portable)

`portable/` is the scheduler-agnostic, env-var-driven version of the balanced
dataset + round-robin collection pipeline — see `portable/README.md` for the full
recipe (build filter → collect per-k with the passes/shards table → merge). Point it
at your nuPlan dataset, the Flow-Planner `model.pth`, and a train-split log list.
