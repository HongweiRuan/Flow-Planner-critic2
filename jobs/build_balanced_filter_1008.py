"""Build a reusable, balanced, TRAIN-split scenario filter for critic training.

Scans a subset of nuPlan TRAIN logs (from Diffusion-Planner/nuplan_train.json --
disjoint from the val14 eval set) and picks up to N_PER_TYPE scenarios of each of
the 14 val14 categories. Writes, into critic-training-data/:
  - critic_train_balanced_1008.yaml         (a nuPlan ScenarioFilter with the explicit
                                        token list -> fast, reproducible collection)
  - critic_train_balanced_1008_token2db.json (token -> db basename, for fast per-worker build)
  - critic_train_balanced_1008_summary.json  (per-category counts + disjointness check)

One-time. After this, collection just points scenario_filter at these tokens.
"""
import json
import os

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

DATA_ROOT = "/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/datasets/public/nuplan/dataset/nuplan-v1.1/splits/trainval"
MAP_ROOT = "/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/datasets/public/nuplan/dataset/maps"
TRAIN_JSON = "/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Diffusion-Planner/nuplan_train.json"
CTD = "/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2/critic-training-data"
VAL14_YAML = "/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/Flow-Planner-critic2/flow_planner/nuplan_simulation/scenario_filter/val14.yaml"

CATEGORIES = [
    "starting_left_turn", "starting_right_turn",
    "starting_straight_traffic_light_intersection_traversal", "stopping_with_lead",
    "high_lateral_acceleration", "high_magnitude_speed", "low_magnitude_speed",
    "traversing_pickup_dropoff", "waiting_for_pedestrian_to_cross", "behind_long_vehicle",
    "stationary_in_traffic", "near_multiple_vehicles", "changing_lane", "following_lane_with_lead",
]
N_PER_TYPE = int(os.environ.get("N_PER_TYPE", "5"))
N_LOGS = int(os.environ.get("N_LOGS", "1500"))  # how many train logs to scan


def val14_tokens():
    import yaml
    spec = yaml.safe_load(open(VAL14_YAML))
    return set(str(t) for t in (spec.get("scenario_tokens") or []))


def main():
    train_logs = json.load(open(TRAIN_JSON))[:N_LOGS]
    db_files = [os.path.join(DATA_ROOT, f"{log}.db") for log in train_logs]
    db_files = [p for p in db_files if os.path.exists(p)]
    print(f"scanning {len(db_files)} train-log dbs for {N_PER_TYPE}/category x {len(CATEGORIES)} categories", flush=True)

    builder = NuPlanScenarioBuilder(DATA_ROOT, MAP_ROOT, None, db_files, "nuplan-maps-v1.0")
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

    os.makedirs(CTD, exist_ok=True)
    # scenario_filter yaml (explicit tokens -> reproducible + fast)
    fy = os.path.join(CTD, "critic_train_balanced_1008.yaml")
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
        f.write(f"num_scenarios_per_type: null\nlimit_total_scenarios: null\n")
        f.write("timestamp_threshold_s: 15\nego_displacement_minimum_m: null\n")
        f.write("ego_start_speed_threshold: null\nego_stop_speed_threshold: null\nspeed_noise_tolerance: null\n")
        f.write("expand_scenarios: false\nremove_invalid_goals: false\nshuffle: false\n")
    json.dump(token2db, open(os.path.join(CTD, "critic_train_balanced_1008_token2db.json"), "w"), indent=2)
    summary = {
        "n_scenarios": len(tokens), "n_per_type_target": N_PER_TYPE, "n_logs_scanned": len(db_files),
        "per_category_counts": {c: len(per_cat.get(c, [])) for c in CATEGORIES},
        "val14_overlap_count": len(overlap), "val14_overlap_tokens": overlap,
        "split": "train (nuplan_train.json)",
    }
    json.dump(summary, open(os.path.join(CTD, "critic_train_balanced_1008_summary.json"), "w"), indent=2)

    print("=== per-category counts ===")
    for c in CATEGORIES:
        print(f"  {c}: {len(per_cat.get(c, []))}")
    print(f"total scenarios: {len(tokens)}")
    print(f"val14 overlap: {len(overlap)} (MUST be 0)")
    print("BUILD_FILTER_DONE_OK")


if __name__ == "__main__":
    main()
