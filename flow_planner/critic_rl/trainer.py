"""Online Q-learning trainer for HorizonCritic.

Plain 1-step Q-learning, with one twist that matters here: the bootstrap actions
are NOT the candidates stored at collection time. Each update, for every next
scene s' in the batch, we RE-INFERENCE fresh candidate trajectories from the
(fixed) planner and take the max over them:

    target = r + gamma * (1 - done) * max_{a' ~ planner(s')} Q_target(s', a')
    loss   = MSE(Q(s, a), target)          (a = the action actually executed)

Re-sampling a' each step (instead of scoring a frozen stored candidate set)
keeps the critic from overfitting to one particular action sample, and a replay
buffer decorrelates the updates.

Everything planner-specific is injected as `candidate_sampler`, so this file
has no dependency on the planner internals:

    candidate_sampler(scene_batch: dict[str, Tensor(B, ...)], num_candidates: int)
        -> Tensor[B, num_candidates, H, state_dim]

`scene_batch` is a dict with the SCENE_KEYS tensors (see critic.py); it must
return that many fresh planner candidates per scene, on the same device.
"""

import copy
import random
from collections import deque
from typing import Callable, Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F

from flow_planner.critic_rl.critic import SCENE_KEYS, HorizonCritic


class ReplayBuffer:
    """In-memory ring buffer of transitions (scene / action / reward / done / next scene)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._data: deque = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add(
        self,
        scene: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        done: bool,
        next_scene: Optional[Mapping[str, torch.Tensor]],
    ) -> None:
        """Store one transition. Tensors are kept on CPU. For terminal steps
        next_scene may be None (it is masked out of the bootstrap anyway)."""
        self._data.append(
            {
                "scene": {k: scene[k].detach().cpu() for k in SCENE_KEYS},
                "action": action.detach().cpu(),
                "reward": float(reward),
                "done": bool(done),
                # terminal: reuse the current scene as a harmless placeholder; the
                # (1 - done) mask zeroes its bootstrap contribution.
                "next_scene": {k: (next_scene or scene)[k].detach().cpu() for k in SCENE_KEYS},
            }
        )

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, object]:
        items = random.sample(self._data, batch_size)
        scene = {k: torch.stack([it["scene"][k] for it in items]).to(device) for k in SCENE_KEYS}
        next_scene = {k: torch.stack([it["next_scene"][k] for it in items]).to(device) for k in SCENE_KEYS}
        action = torch.stack([it["action"] for it in items]).to(device)  # [B,H,state_dim]
        reward = torch.tensor([it["reward"] for it in items], dtype=torch.float32, device=device)
        done = torch.tensor([it["done"] for it in items], dtype=torch.float32, device=device)
        return {"scene": scene, "next_scene": next_scene, "action": action, "reward": reward, "done": done}


class OnlineQLearningTrainer:
    def __init__(
        self,
        critic: HorizonCritic,
        candidate_sampler: Callable[[Mapping[str, torch.Tensor], int], torch.Tensor],
        visible_horizon: int,
        gamma: float = 1.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        encoder_lr: Optional[float] = None,  # smaller LR for the (unfrozen) encoder; None -> learning_rate
        batch_size: int = 256,
        num_bootstrap_candidates: int = 16,
        target_tau: float = 0.005,
        max_grad_norm: float = 10.0,
        buffer_capacity: int = 100000,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.target_critic.eval()
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.candidate_sampler = candidate_sampler
        self.visible_horizon = int(visible_horizon)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.num_bootstrap_candidates = int(num_bootstrap_candidates)
        self.target_tau = float(target_tau)
        self.max_grad_norm = float(max_grad_norm)
        self.buffer = ReplayBuffer(buffer_capacity)

        # Optionally give the encoder its own (smaller) LR so it doesn't drift far
        # from the planner representation. Frozen encoder params are skipped.
        enc_ids = {id(p) for p in self.critic.scene_encoder.parameters()}
        head_params = [p for p in self.critic.parameters() if p.requires_grad and id(p) not in enc_ids]
        enc_params = [p for p in self.critic.scene_encoder.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": learning_rate}]
        if enc_params:
            groups.append({"params": enc_params, "lr": encoder_lr if encoder_lr is not None else learning_rate})
        self.optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)

    def _q_taken(self, scene: Mapping[str, torch.Tensor], action: torch.Tensor) -> tuple:
        """Q(s, a) for the single executed action a: [B,H,D] -> two [B] tensors."""
        q1, q2 = self.critic(scene, action[:, None], self.visible_horizon)  # N=1
        return q1[:, 0], q2[:, 0]

    @torch.no_grad()
    def _bootstrap_target(self, batch: Dict[str, object]) -> torch.Tensor:
        next_scene = batch["next_scene"]
        # Fresh planner candidates at each next scene -> [B, K, H, D].
        next_candidates = self.candidate_sampler(next_scene, self.num_bootstrap_candidates).to(self.device)
        tq1, tq2 = self.target_critic(next_scene, next_candidates, self.visible_horizon)  # [B,K]
        q_next = 0.5 * (tq1 + tq2)  # plain Q-learning: mean of twin heads
        best = q_next.max(dim=1).values  # max over candidates
        return batch["reward"] + self.gamma * (1.0 - batch["done"]) * best

    def train_step(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.device)
        target = self._bootstrap_target(batch)  # [B]
        q1, q2 = self._q_taken(batch["scene"], batch["action"])
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in self.critic.parameters() if p.requires_grad), self.max_grad_norm
        )
        self.optimizer.step()
        self._soft_update_target()
        return {
            "loss": float(loss.detach()),
            "q_mean": float(q1.mean().detach()),
            "target_mean": float(target.mean().detach()),
            "grad_norm": float(grad_norm),
        }

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.mul_(1.0 - self.target_tau).add_(self.target_tau * p.data)

    def train(self, updates: int, warmup: int = 1000, log_interval: int = 0, logger=None) -> Dict[str, float]:
        if len(self.buffer) < max(self.batch_size, warmup):
            raise RuntimeError(
                f"buffer has {len(self.buffer)} < warmup {max(self.batch_size, warmup)}; fill it first"
            )
        last: Dict[str, float] = {}
        for step in range(1, int(updates) + 1):
            last = self.train_step()
            if log_interval and (step % log_interval == 0 or step == updates):
                last["update_step"] = step
                print(f"[train] step={step} " + " ".join(f"{k}={v:.4f}" for k, v in last.items()), flush=True)
                if logger is not None:
                    logger(last)
        return last

    def save_checkpoint(self, path: str) -> None:
        torch.save({"critic": self.critic.state_dict()}, path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["critic"])
