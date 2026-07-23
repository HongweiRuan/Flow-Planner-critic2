#!/usr/bin/env python
"""Closed-loop evaluation on nuPlan val14.

Runs each val14 scenario closed-loop, picks a candidate every step with the
chosen scorer, and reports the official nuPlan score. Scenarios are sharded
across Ray workers by token so each worker builds only its own slice.

    # baseline: always execute the raw planner sample (candidate 0):
    python evaluate.py --config config/eval.yaml --scorer candidate0

    # random candidate baseline:
    python evaluate.py --config config/eval.yaml --scorer random

    # a trained critic (needs critic.py implemented + a checkpoint):
    python evaluate.py --config config/eval.yaml --scorer critic \
        --checkpoint /path/to/critic.pt --visible-horizon 10

Writes per-scenario metrics (CSV + JSON) and a summary.json to `output_dir`,
refreshed as it goes, and prints a running mean to the console.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import ray
import yaml

from flow_planner.critic_rl.workers import EvaluationWorker


# --------------------------------------------------------------------------- #
# val14 token sharding: read the scenario token list + the token->db map so
# each worker builds only its own scenarios and opens only its own dbs.
# --------------------------------------------------------------------------- #
def _override_value(overrides: Sequence[str], key: str) -> Optional[str]:
    for o in overrides:
        t = str(o).strip().lstrip("+").strip()
        if t.startswith(key + "="):
            return t.split("=", 1)[1].strip()
    return None


def scenario_filter_name(factory_kwargs: Dict[str, Any]) -> Optional[str]:
    return _override_value(factory_kwargs.get("overrides", []), "scenario_filter")


def load_tokens(name: str) -> Optional[List[str]]:
    """Explicit scenario token list of the filter (val14 -> its 1118 tokens)."""
    path = os.path.join("flow_planner", "nuplan_simulation", "scenario_filter", f"{name}.yaml")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    tokens = spec.get("scenario_tokens")
    return [str(t) for t in tokens] if isinstance(tokens, list) and tokens else None


def load_token2db(name: str) -> Optional[Dict[str, str]]:
    """token -> db filename map, cached at config/<name>_token2db.json."""
    path = os.path.join("config", f"{name}_token2db.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def with_token_subset(factory_kwargs: Dict[str, Any], tokens: Sequence[str], token2db: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Copy factory_kwargs, pinning the filter to exactly `tokens` (and, if known,
    restricting the open dbs to just those holding the tokens)."""
    fk = dict(factory_kwargs)
    drop = ("scenario_filter.scenario_tokens", "scenario_filter.limit_total_scenarios", "scenario_builder.db_files")
    overrides = [o for o in fk.get("overrides", []) if not str(o).strip().startswith(drop)]
    # Quote tokens: some look like floats (e.g. 5953...e137) and hydra would mis-parse them.
    overrides.append("scenario_filter.scenario_tokens=[" + ",".join('"%s"' % t for t in tokens) + "]")
    overrides.append("scenario_filter.limit_total_scenarios=null")
    if token2db:
        data_root = _override_value(overrides, "scenario_builder.data_root")
        dbs = sorted({token2db[t] for t in tokens if t in token2db})
        if dbs and data_root:
            paths = [os.path.join(data_root, db) for db in dbs]
            overrides.append("scenario_builder.db_files=[" + ",".join('"%s"' % p for p in paths) + "]")
    fk["overrides"] = overrides
    return fk


# --------------------------------------------------------------------------- #
# output writing
# --------------------------------------------------------------------------- #
def write_outputs(output_dir: Optional[str], records: List[Dict[str, Any]], evaluation: Dict[str, Any]) -> Dict[str, Any]:
    scores = [r["official_score"] for r in records]
    metric_names = sorted({m for r in records for m in r.get("metrics", {})})
    metric_means = {}
    for m in metric_names:
        vals = [r["metrics"][m] for r in records if m in r.get("metrics", {})]
        metric_means[m] = sum(vals) / len(vals) if vals else None
    summary = {
        "episodes": len(scores),
        "mean_official_score": sum(scores) / len(scores) if scores else None,
        "metric_means": metric_means,
        "scorer": evaluation.get("scorer"),
        "visible_horizon": evaluation.get("visible_horizon"),
        "checkpoint": evaluation.get("checkpoint"),
        "num_candidates": evaluation.get("num_candidates"),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cols = ["episode_id", "scenario_token", "scenario_type", "log_name", "steps", "official_score"] + metric_names
        tmp = os.path.join(output_dir, ".metrics_per_scenario.csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in records:
                w.writerow(
                    [r.get(k) for k in ("episode_id", "scenario_token", "scenario_type", "log_name", "steps", "official_score")]
                    + [r.get("metrics", {}).get(m, "") for m in metric_names]
                )
        os.replace(tmp, os.path.join(output_dir, "metrics_per_scenario.csv"))
        with open(os.path.join(output_dir, "metrics_per_scenario.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to an eval config (e.g. config/eval.yaml)")
    ap.add_argument("--scorer", choices=("candidate0", "random", "critic"), help="candidate selection rule (overrides config)")
    ap.add_argument("--checkpoint", help="critic checkpoint (critic scorer only)")
    ap.add_argument("--visible-horizon", type=int, help="candidate steps the critic sees (critic scorer only)")
    ap.add_argument("--episodes", type=int, help="cap number of scenarios (overrides config)")
    ap.add_argument("--num-candidates", type=int, help="candidates the scorer ranks (overrides config)")
    ap.add_argument("--workers", type=int, help="number of Ray eval workers (overrides config)")
    ap.add_argument("--output-dir", help="where to write per-scenario metrics + summary (overrides config)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ev = dict(cfg["evaluate"])
    for key, val in (
        ("scorer", args.scorer), ("checkpoint", args.checkpoint), ("visible_horizon", args.visible_horizon),
        ("episodes", args.episodes), ("num_candidates", args.num_candidates), ("workers", args.workers),
        ("output_dir", args.output_dir),
    ):
        if val is not None:
            ev[key] = val

    if ev.get("scorer") == "critic" and (ev.get("visible_horizon") is None or ev.get("checkpoint") is None):
        raise SystemExit("critic scorer needs --visible-horizon and --checkpoint")

    factory_kwargs = dict(cfg["factory"]["kwargs"])
    if ev.get("num_candidates") is not None:
        factory_kwargs["num_candidates"] = int(ev["num_candidates"])

    total = int(ev["episodes"])
    n_workers = int(ev["workers"])
    output_dir = ev.get("output_dir")
    print(f"[eval] scorer={ev.get('scorer')} num_candidates={ev.get('num_candidates')} "
          f"visible_horizon={ev.get('visible_horizon')} checkpoint={ev.get('checkpoint')}", flush=True)

    ray.init(address=cfg.get("ray", {}).get("address") or None, ignore_reinit_error=True, log_to_driver=False)

    name = scenario_filter_name(factory_kwargs)
    tokens = load_tokens(name) if name else None
    if not tokens:
        raise SystemExit(f"scenario_filter '{name}' has no token list; expected "
                         f"flow_planner/nuplan_simulation/scenario_filter/{name}.yaml with scenario_tokens")
    token2db = load_token2db(name)
    tokens = tokens[:total]
    shards = [tokens[i::n_workers] for i in range(n_workers)]
    shards = [s for s in shards if s]

    def spawn(shard: Sequence[str], index: int):
        return ray.remote(EvaluationWorker).options(
            num_cpus=float(ev["cpus_per_worker"]), num_gpus=float(ev["gpus_per_worker"])
        ).remote(
            factory_path=cfg["factory"]["path"],
            factory_kwargs=with_token_subset(factory_kwargs, shard, token2db),
            scorer=ev["scorer"],
            seed=int(cfg.get("seed", 0)) + index,
            critic_kwargs=cfg.get("critic"),
            visible_horizon=ev.get("visible_horizon"),
            checkpoint=ev.get("checkpoint"),
            q_reduction=ev.get("q_reduction", "mean"),
        )

    workers = [spawn(s, i) for i, s in enumerate(shards)]
    # Each worker's remaining local scenario indices; dispatch one at a time so we
    # get per-scenario progress and can retry a worker that faults (e.g. CUDA OOM).
    remaining = {i: list(range(len(shards[i]))) for i in range(len(workers))}
    fails = {i: 0 for i in range(len(workers))}
    dead = set()
    max_fails = 5
    in_flight = {}  # future -> (worker_index, local_scenario_index)

    def dispatch(i: int):
        if i in dead or not remaining[i]:
            return
        j = remaining[i].pop(0)
        in_flight[workers[i].evaluate.remote([j])] = (i, j)

    for i in range(len(workers)):
        dispatch(i)

    records: List[Dict[str, Any]] = []
    score_sum = 0.0
    write_every = max(1, len(tokens) // 50)
    since_write = 0
    while in_flight:
        done, _ = ray.wait(list(in_flight.keys()), num_returns=1)
        i, j = in_flight.pop(done[0])
        try:
            recs = ray.get(done[0]).get("records", [])
        except Exception as exc:  # worker fault: respawn its shard and retry this scenario
            fails[i] += 1
            sys.stderr.write(f"\n[eval] worker {i} fault on scen {j} (try {fails[i]}/{max_fails}): {type(exc).__name__}\n")
            remaining[i].insert(0, j)
            try:
                ray.kill(workers[i])
            except Exception:
                pass
            if fails[i] >= max_fails:
                sys.stderr.write(f"[eval] worker {i} exceeded max fails; dropping {len(remaining[i])} scenarios\n")
                dead.add(i)
                remaining[i] = []
            else:
                try:
                    workers[i] = spawn(shards[i], i)
                    dispatch(i)
                except Exception:
                    dead.add(i)
                    remaining[i] = []
            continue
        for rec in recs:
            records.append(rec)
            score_sum += rec["official_score"]
            print(f"[{len(records)}/{len(tokens)}] {rec.get('scenario_token', '?')}  "
                  f"score={rec.get('official_score', float('nan')):.3f}  running_mean={score_sum/len(records):.4f}", flush=True)
        since_write += len(recs)
        if since_write >= write_every:
            write_outputs(output_dir, records, ev)
            since_write = 0
        dispatch(i)

    summary = write_outputs(output_dir, records, ev)
    ray.shutdown()
    print(f"[eval] DONE  {json.dumps(summary)}")
    if output_dir:
        print(f"[eval] results -> {output_dir}")


if __name__ == "__main__":
    main()
