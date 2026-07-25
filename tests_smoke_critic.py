"""Smoke test for the critic pipeline code (critic / trainer / replay).

Uses a DummyEncoder that mimics FlowPlannerEncoder's output contract, so it
validates all the NEW code (tokenization, cross-attn, pool, twin-Q, freeze knob,
online Q-learning loop, zarr round-trip) WITHOUT needing nuPlan / the planner.
Run on the GPU pod:  PYTHONPATH=<fork> python tests_smoke_critic.py
"""
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from flow_planner.critic_rl.critic import HorizonCritic, SCENE_KEYS
from flow_planner.critic_rl.trainer import OnlineQLearningTrainer
from flow_planner.critic_rl.replay import ReplaySpec, ZarrReplayWriter, ZarrReplayReader
from flow_planner.critic_rl.types import REWARD_COMPONENT_NAMES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, SD, CTX = 80, 4, 192
NB, NS, NL = 32, 5, 70          # neighbor / static / lane token counts -> S = 107


def make_scene(B):
    return {
        "neighbors": torch.randn(B, NB, 21, 9),
        "static": torch.randn(B, NS, 11),
        "lanes": torch.randn(B, NL, 20, 12),
        "lanes_speed_limit": torch.randn(B, NL, 1),
        "lanes_has_speed_limit": (torch.rand(B, NL, 1) > 0.5).float(),
        "routes": torch.randn(B, 25, 20, 12),
    }


class DummyEncoder(nn.Module):
    """Mimics FlowPlannerEncoder: returns encodings=(a,b), masks=(ma,mb) with True=valid."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(12, CTX)

    def forward(self, neighbors, static, lanes, lanes_speed_limit, lanes_has_speed_limit, routes):
        B, dev = neighbors.shape[0], neighbors.device
        bias = self.lin(lanes[:, :1, :1, :]).mean()  # keep a real param in the graph
        a = torch.randn(B, NB + NS, CTX, device=dev) + bias
        b = torch.randn(B, NL, CTX, device=dev) + bias
        ma = torch.ones(B, NB + NS, dtype=torch.bool, device=dev)
        mb = torch.ones(B, NL, dtype=torch.bool, device=dev)
        return {"encodings": (a, b), "masks": (ma, mb)}


def test_critic():
    enc = DummyEncoder()
    critic = None
    for freeze in (True, False):
        critic = HorizonCritic(scene_encoder=enc, horizon=H, freeze_encoder=freeze).to(DEV)
        B, N, L = 4, 16, 10
        scene = {k: v.to(DEV) for k, v in make_scene(B).items()}
        cand = torch.randn(B, N, H, SD, device=DEV)
        q1, q2 = critic(scene, cand, visible_horizon=L)
        assert q1.shape == (B, N) and q2.shape == (B, N), (q1.shape, q2.shape)
        assert critic.score(scene, cand, L, "mean").shape == (B, N)
        assert critic.score(scene, cand, L, "min").shape == (B, N)
        n_enc_grad = sum(p.requires_grad for p in critic.scene_encoder.parameters())
        print(f"[critic] freeze={freeze} q{tuple(q1.shape)} encoder_trainable_params={n_enc_grad} OK")
    return critic


def test_replay():
    path = os.path.join(tempfile.mkdtemp(), "r.zarr")
    w = ZarrReplayWriter(path, ReplaySpec(capacity=100), overwrite=True)
    recs = []
    for ep in range(2):
        for st in range(5):
            recs.append(
                {
                    "scene": {k: v[0].numpy() for k, v in make_scene(1).items()},
                    "action": np.random.randn(H, SD).astype("f4"),
                    "reward": float(np.random.randn()),
                    "reward_components": np.random.randn(len(REWARD_COMPONENT_NAMES)).astype("f4"),
                    "done": (st == 4),
                    "episode_id": ep,
                    "step_index": st,
                }
            )
    w.append_records(recs)
    r = ZarrReplayReader(path)
    trs = list(r.iter_transitions())
    with_next = sum(1 for t in trs if t["next_scene"] is not None)
    print(f"[replay] size={len(r)} yielded={len(trs)} with_next={with_next} keys={r.scene_keys}")
    assert len(r) == 10 and with_next == 8, (len(r), with_next)
    # a next_scene must actually be a dict of the 6 keys
    first_next = next(t["next_scene"] for t in trs if t["next_scene"] is not None)
    assert set(first_next.keys()) == set(SCENE_KEYS)


def test_trainer(critic):
    def sampler(scene_batch, K):
        return torch.randn(scene_batch["neighbors"].shape[0], K, H, SD)

    tr = OnlineQLearningTrainer(
        critic=critic, candidate_sampler=sampler, visible_horizon=10,
        batch_size=8, num_bootstrap_candidates=6, buffer_capacity=200, device=DEV,
    )
    for _ in range(60):
        sc, ns = make_scene(1), make_scene(1)
        tr.buffer.add(
            {k: v[0] for k, v in sc.items()}, torch.randn(H, SD),
            float(np.random.randn()), bool(np.random.rand() < 0.2), {k: v[0] for k, v in ns.items()},
        )
    out = tr.train(updates=5, warmup=8, log_interval=5)
    assert np.isfinite(out["loss"]), out
    print(f"[trainer] loss={out['loss']:.4f} q_mean={out['q_mean']:.4f} target_mean={out['target_mean']:.4f} OK")


if __name__ == "__main__":
    print(f"device={DEV} torch={torch.__version__}")
    critic = test_critic()
    test_replay()
    test_trainer(critic)
    print("SMOKE_OK")
