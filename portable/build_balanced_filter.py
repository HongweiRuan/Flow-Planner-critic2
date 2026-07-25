"""Build a reusable, balanced, TRAIN-split scenario filter for critic training.

Scans nuPlan TRAIN logs and picks up to N_PER_TYPE scenarios of each of the 14
val14 categories, producing a nuPlan ScenarioFilter with an explicit token list
(reproducible + fast per-worker rebuild). Optionally checks the picked tokens are
disjoint from a val14 eval set.

Everything is driven by environment variables so this runs on any cluster:

  DATA_ROOT   nuPlan trainval split dir (contains <log>.db files)         [required]
  MAP_ROOT    nuPlan maps dir                                             [required]
  TRAIN_JSON  JSON list of train log names (no .db); we scan TRAIN_JSON[:N_LOGS]
              e.g. Diffusion-Planner/nuplan_train.json                    [required]
  OUT_DIR     where to write the 3 output files                          [required]
  VAL14_YAML  a val14 ScenarioFilter yaml, for the disjointness check     [optional]
  N_PER_TYPE  scenarios per category to keep (e.g. 72)                    [default 24]
  N_LOGS      how many train logs to scan (more -> rarer categories fill) [default 15000]
  OUT_TAG     basename for outputs                                        [default critic_train_balanced]
  MAP_VERSION nuPlan map version                                          [default nuplan-maps-v1.0]

Outputs (in OUT_DIR):
  <OUT_TAG>.yaml            nuPlan ScenarioFilter with explicit scenario_tokens
  <OUT_TAG>_token2db.json   token -> "<log>.db" (fast per-worker scenario rebuild)
  <OUT_TAG>_summary.json    per-category counts + val14 disjointness result
"""
import json
import os

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

DATA_ROOT = os.environ["DATA_ROOT"]
MAP_ROOT = os.environ["MAP_ROOT"]
TRAIN_JSON = os.environ["TRAIN_JSON"]
OUT_DIR = os.environ["OUT_DIR"]
VAL14_YAML = os.environ.get("VAL14_YAML", "")
MAP_VERSION = os.environ.get("MAP_VERSION", "nuplan-maps-v1.0")
N_PER_TYPE = int(os.environ.get("N_PER_TYPE", "24"))
N_LOGS = int(os.environ.get("N_LOGS", "15000"))
OUT_TAG = os.environ.get("OUT_TAG", "critic_train_balanced")

# The 14 val14 closed-loop categories.
CATEGORIES = [
    "starting_left_turn", "starting_right_turn",
    "starting_straight_traffic_light_intersection_traversal", "stopping_with_lead",
    "high_lateral_acceleration", "high_magnitude_speed", "low_magnitude_speed",
    "traversing_pickup_dropoff", "waiting_for_pedestrian_to_cross", "behind_long_vehicle",
    "stationary_in_traffic", "near_multiple_vehicles", "changing_lane", "following_lane_with_lead",
]


def val14_tokens():
    if not VAL14_YAML or not os.path.exists(VAL14_YAML):
        return set()
    import yaml
    spec = yaml.safe_load(open(VAL14_YAML))
    return set(str(t) for t in (spec.get("scenario_tokens") or []))


def main():
    train_logs = json.load(open(TRAIN_JSON))[:N_LOGS]
    db_files = [os.path.join(DATA_ROOT, f"{log}.db") for log in train_logs]
    db_files = [p for p in db_files if os.path.exists(p)]
    print(f"scanning {len(db_files)} train-log dbs for {N_PER_TYPE}/category x {len(CATEGORIES)} categories", flush=True)

    builder = NuPlanScenarioBuilder(DATA_ROOT, MAP_ROOT, None, db_files, MAP_VERSION)
    sfilter = ScenarioFilter(
        scenario_types=CATEGORIES, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=N_PER_TYPE, limit_total_scenarios=None, timestamp_threshold_s=15.0,
        ego_displacement_minimum_m=None, expand_scenarios=False, remove_invalid_goals=True,
        shuffle=False, ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(sfilter, worker)
    print(f"got {len(scenarios)} scenarios", flush=True)

    tokens, token2db, per_cat = [], {}, {}
    for s in scenarios:
        tok = s.token
        tokens.append(tok)
        token2db[tok] = f"{s.log_name}.db"
        per_cat.setdefault(s.scenario_type, []).append(tok)

    v14 = val14_tokens()
    overlap = sorted(set(tokens) & v14)

    os.makedirs(OUT_DIR, exist_ok=True)
    fy = os.path.join(OUT_DIR, f"{OUT_TAG}.yaml")
    with open(fy, "w") as f:
        f.write('_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter\n')
        f.write('_convert_: "all"\n\n')
        f.write("scenario_types:\n")
        for c in CATEGORIES:
            f.write(f"  - {c}\n")
        f.write("\nscenario_tokens:\n")
        for t in tokens:
            f.write(f'  - "{t}"\n')
        f.write("\nlog_names: null\nmap_names: null\n")
        f.write("num_scenarios_per_type: null\nlimit_total_scenarios: null\n")
        f.write("timestamp_threshold_s: 15\nego_displacement_minimum_m: null\n")
        f.write("ego_start_speed_threshold: null\nego_stop_speed_threshold: null\nspeed_noise_tolerance: null\n")
        f.write("expand_scenarios: false\nremove_invalid_goals: false\nshuffle: false\n")
    json.dump(token2db, open(os.path.join(OUT_DIR, f"{OUT_TAG}_token2db.json"), "w"), indent=2)
    summary = {
        "n_scenarios": len(tokens), "n_per_type_target": N_PER_TYPE, "n_logs_scanned": len(db_files),
        "per_category_counts": {c: len(per_cat.get(c, [])) for c in CATEGORIES},
        "val14_overlap_count": len(overlap), "val14_overlap_tokens": overlap,
        "split": "train (TRAIN_JSON)",
    }
    json.dump(summary, open(os.path.join(OUT_DIR, f"{OUT_TAG}_summary.json"), "w"), indent=2)

    print("=== per-category counts ===")
    for c in CATEGORIES:
        print(f"  {c}: {len(per_cat.get(c, []))}")
    print(f"total scenarios: {len(tokens)}")
    print(f"val14 overlap: {len(overlap)} (MUST be 0 if VAL14_YAML given)")
    print("BUILD_FILTER_DONE_OK")


if __name__ == "__main__":
    main()
