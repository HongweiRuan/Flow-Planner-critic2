#!/usr/bin/env python
"""Train a HorizonCritic with online Q-learning.

    python train.py --config config/eval.yaml --replay /path/replay_h1.zarr \
        --visible-horizon 10 --checkpoint out/q10.pt --updates 20000

- Loads the Flow-Planner model standalone (for the critic's embedded encoder and
  for re-inferencing bootstrap candidates).
- Fills the online trainer's replay buffer from a collected zarr replay.
- Trains, then saves a checkpoint ({"critic": state_dict}) that
  `evaluate.py --scorer critic --checkpoint ...` can load.

`--freeze-encoder` toggles the A/B knob (frozen planner tokens vs fine-tuned).
Planner config_path / ckpt_path and the `critic:` block come from --config.
"""
import argparse
import os
from typing import Optional, Sequence

import numpy as np
import torch
import yaml

from flow_planner.critic_rl.critic import HorizonCritic, SCENE_KEYS
from flow_planner.critic_rl.model_loader import load_flow_planner_model
from flow_planner.critic_rl.replay import ZarrReplayReader
from flow_planner.critic_rl.trainer import OnlineQLearningTrainer


def _override_value(overrides: Sequence[str], key: str) -> Optional[str]:
    for o in overrides:
        t = str(o).strip().lstrip("+").strip()
        if t.startswith(key + "="):
            return t.split("=", 1)[1].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config with planner paths + critic block (e.g. config/eval.yaml)")
    ap.add_argument("--replay", required=True, help="zarr replay from collect.py")
    ap.add_argument("--visible-horizon", type=int, required=True, help="critic look-ahead L (the k in Q^k)")
    ap.add_argument("--checkpoint", required=True, help="output checkpoint path")
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--freeze-encoder", action="store_true", help="freeze the scene encoder (A/B baseline)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bootstrap-candidates", type=int, default=8, help="K candidates re-inferenced per next scene")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr", type=float, default=3e-5, help="smaller LR for the (unfrozen) encoder")
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--buffer-capacity", type=int, default=200000)
    ap.add_argument("--max-transitions", type=int, default=None, help="cap transitions loaded (debug)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = cfg["factory"]["kwargs"]["overrides"]
    config_path = _override_value(overrides, "planner.flow_planner.config_path")
    ckpt_path = _override_value(overrides, "planner.flow_planner.ckpt_path")
    print(f"[train] planner config={config_path}\n[train] planner ckpt={ckpt_path}", flush=True)

    model = load_flow_planner_model(config_path, ckpt_path, device=args.device)
    critic = HorizonCritic(
        scene_encoder=model.model_encoder, freeze_encoder=args.freeze_encoder, **dict(cfg["critic"])
    )
    print(f"[train] critic built (freeze_encoder={args.freeze_encoder}, visible_horizon={args.visible_horizon})", flush=True)

    def candidate_sampler(scene_batch, num_candidates):
        enc = {k: scene_batch[k].to(args.device) for k in SCENE_KEYS}
        with torch.no_grad():
            return model.sample_candidates_from_encoder_inputs(enc, num_candidates)

    trainer = OnlineQLearningTrainer(
        critic=critic,
        candidate_sampler=candidate_sampler,
        visible_horizon=args.visible_horizon,
        learning_rate=args.lr,
        encoder_lr=args.encoder_lr,
        batch_size=args.batch_size,
        num_bootstrap_candidates=args.bootstrap_candidates,
        buffer_capacity=args.buffer_capacity,
        device=args.device,
    )

    # Fill the buffer from the collected replay.
    reader = ZarrReplayReader(args.replay)
    n = 0
    for t in reader.iter_transitions():
        scene = {k: torch.from_numpy(np.asarray(t["scene"][k])) for k in t["scene"]}
        nxt = None if t["next_scene"] is None else {k: torch.from_numpy(np.asarray(t["next_scene"][k])) for k in t["next_scene"]}
        trainer.buffer.add(scene, torch.from_numpy(np.asarray(t["action"])), t["reward"], t["done"], nxt)
        n += 1
        if args.max_transitions and n >= args.max_transitions:
            break
    print(f"[train] buffer filled: {len(trainer.buffer)} transitions (read {n})", flush=True)

    trainer.train(updates=args.updates, warmup=args.warmup, log_interval=args.log_interval)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    trainer.save_checkpoint(args.checkpoint)
    print(f"[train] saved {args.checkpoint}")
    print("TRAIN_DONE_OK")


if __name__ == "__main__":
    main()
