#!/usr/bin/env python
"""Collect critic training data.

Rolls out the Flow-Planner across nuPlan scenarios in parallel (Ray) and saves
every decision as a transition (scene + N candidate trajectories + reward) into
a replay buffer on disk. This is the data the critic trains on.

    # collect with the defaults in the config (execution_horizon=1, target=10k):
    python collect.py --config config/collect.yaml

    # collect for a longer commit horizon, more transitions, more workers:
    python collect.py --config config/collect.yaml --execution-horizon 10 --target 20000 --workers 64

Collection stops once the buffer holds `--target` transitions (it is a ring
buffer, so it never grows past its capacity). Each execution horizon K writes to
its own file:  replay.zarr -> replay_h<K>.zarr, so runs for different K never
clash and can run at the same time.
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import ray
import yaml

from flow_planner.critic_rl.replay import ReplaySpec, create_replay_writer_actor
from flow_planner.critic_rl.workers import CollectorWorker


def replay_path_for_horizon(path: str, horizon: int) -> str:
    """One replay file per execution horizon:  replay.zarr -> replay_h10.zarr."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_h{int(horizon)}{p.suffix}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a collect config (e.g. config/collect.yaml)")
    ap.add_argument("--execution-horizon", type=int, help="sim steps committed per decision (overrides config)")
    ap.add_argument("--target", type=int, help="stop once the buffer holds this many transitions (overrides config)")
    ap.add_argument("--workers", type=int, help="number of parallel Ray collector actors (overrides config)")
    ap.add_argument("--out", help="replay path (overrides config replay.path); _h<K> is appended per horizon")
    ap.add_argument("--summary", help="where to write the JSON run summary (default: alongside the replay)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    c = cfg["collect"]
    horizon = args.execution_horizon if args.execution_horizon is not None else int(c["execution_horizon"])
    target = args.target if args.target is not None else int(c["target"])
    workers = args.workers if args.workers is not None else int(c["workers"])
    base_path = args.out or cfg["replay"]["path"]
    path = replay_path_for_horizon(base_path, horizon)

    # Tell the env how many sim steps to commit per decision.
    factory_kwargs = copy.deepcopy(cfg["factory"]["kwargs"])
    factory_kwargs["execution_horizon"] = horizon

    print(f"[collect] horizon={horizon} target={target} workers={workers}")
    print(f"[collect] replay -> {path}")

    ray.init(address=cfg.get("ray", {}).get("address") or None, ignore_reinit_error=True)

    # One writer actor owns the replay; workers stream records to it.
    writer = create_replay_writer_actor(
        path, ReplaySpec(**cfg["replay"]["spec"]), overwrite=bool(cfg["replay"].get("overwrite", True))
    )
    worker_cls = ray.remote(CollectorWorker).options(
        num_cpus=float(c["cpus_per_worker"]), num_gpus=float(c["gpus_per_worker"])
    )
    actors = [worker_cls.remote(cfg["factory"]["path"], factory_kwargs) for _ in range(workers)]
    print(f"[collect] {workers} workers built; collecting...")

    # Collect in rounds until the buffer reaches `target`. Round 1 runs one
    # episode per worker to measure the yield (transitions/episode); later rounds
    # are sized from that rate to hit the target in as few rounds as possible.
    episodes_done = 0
    round_size = workers
    start = time.time()
    size = 0
    while size < target:
        assignments = [[] for _ in range(workers)]
        for j, eid in enumerate(range(episodes_done, episodes_done + round_size)):
            assignments[j % workers].append(eid)
        ray.get([actors[i].collect.remote(writer, ids) for i, ids in enumerate(assignments) if ids])
        episodes_done += round_size
        size = ray.get(writer.stats.remote())["size"]
        rate = size / max(episodes_done, 1)
        print(f"[collect]   episodes={episodes_done} size={size} ({rate:.1f} tx/ep) {time.time()-start:.0f}s", flush=True)
        if size >= target:
            break
        need = target - size
        round_size = max(workers, min(int(math.ceil(need / max(rate, 0.05) * 1.15)), 40 * workers))

    stats = ray.get(writer.stats.remote())
    stats.update(horizon=horizon, episodes=episodes_done, seconds=round(time.time() - start, 1), replay_path=path)
    ray.shutdown()

    summary_path = args.summary or str(Path(path).with_suffix(".summary.json"))
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[collect] DONE  {json.dumps(stats)}")
    print(f"[collect] summary -> {summary_path}")


if __name__ == "__main__":
    main()
