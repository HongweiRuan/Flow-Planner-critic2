#!/usr/bin/env python
"""Aggregate sharded eval CSVs into one full-val14 summary.
  python agg_shards.py <TAG> <SHARD_COUNT>
Reads $RUN/eval_<TAG>_s{i}of{N}/metrics_per_scenario.csv for i in [0,N), and any
$RUN/eval_<TAG>/metrics_per_scenario.csv (unsharded), dedups by scenario_token."""
import csv
import glob
import json
import os
import sys

RUN = "/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/critic2_runs"
META = {"episode_id", "scenario_token", "scenario_type", "log_name", "steps", "official_score"}


def main():
    tag, n = sys.argv[1], int(sys.argv[2])
    paths = [f"{RUN}/eval_{tag}_s{i}of{n}/metrics_per_scenario.csv" for i in range(n)]
    paths.append(f"{RUN}/eval_{tag}/metrics_per_scenario.csv")
    rows, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                tok = r.get("scenario_token")
                if tok and tok not in seen:
                    seen.add(tok)
                    rows.append(r)
    if not rows:
        print(f"{tag}: NO ROWS found"); return
    scores = [float(r["official_score"]) for r in rows if r.get("official_score") not in (None, "")]
    metrics = sorted(k for k in rows[0].keys() if k not in META)
    mm = {}
    for m in metrics:
        vals = [float(r[m]) for r in rows if r.get(m) not in (None, "")]
        mm[m] = round(sum(vals) / len(vals), 4) if vals else None
    mean = round(sum(scores) / len(scores), 4)
    out = {"tag": tag, "scenarios": len(scores), "mean_official_score": mean, "metric_means": mm}
    json.dump(out, open(f"{RUN}/eval_{tag}_full.json", "w"), indent=2)
    print(f"{tag}: scenarios={len(scores)}  mean_official_score={mean}")
    for m, v in mm.items():
        print(f"    {m}: {v}")


if __name__ == "__main__":
    main()
