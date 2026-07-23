"""Ray actors for the two pipelines.

Only two workers exist here:

    CollectorWorker   -- used by collect.py: rolls out episodes and appends
                         transitions to the replay writer.
    EvaluationWorker  -- used by evaluate.py: runs closed-loop scenarios and
                         reports the official nuPlan score per scenario.

Each worker builds its own nuPlan factory (GPU planner + CPU simulator) from a
factory path + kwargs, so it is self-contained on its Ray actor.
"""

import importlib
import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import ray
import torch

from flow_planner.critic_rl.critic import HorizonCritic
from flow_planner.critic_rl.replay import transition_to_record

logger = logging.getLogger(__name__)


def load_object(path: str) -> Any:
    """Import a `module:attribute` path (e.g. the factory in the config)."""
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError("object path must use module:attribute syntax")
    return getattr(importlib.import_module(module_name), attribute)


def split_episode_ids(count: int, workers: int):
    """Round-robin split of episode ids across workers."""
    return [list(range(index, count, workers)) for index in range(workers)]


class CollectorWorker:
    """Roll out episodes and stream transitions to the replay writer.

    The planner proposes N candidate trajectories per decision; we always
    execute candidate 0 (an ordinary planner sample) and record the full
    transition, which includes all N candidates for later critic training.
    """

    def __init__(self, factory_path: str, factory_kwargs: Mapping[str, Any]) -> None:
        self.factory = load_object(factory_path)(**dict(factory_kwargs))

    def collect(self, writer: Any, episode_ids: Sequence[int]) -> Dict[str, Any]:
        transitions = 0
        completed = 0
        scores = []
        pending = []
        for episode_id in episode_ids:
            env = self.factory(int(episode_id))
            env.reset(episode_id=int(episode_id))
            done = False
            while not done:
                transition = env.step(0)  # execute candidate 0
                pending.append(transition_to_record(transition))
                if len(pending) == 8:  # batch writes to cut actor round-trips
                    ray.get(writer.append_records.remote(pending))
                    pending.clear()
                transitions += 1
                done = transition.done
                if done:
                    scores.append(float(transition.info["official_score"]))
                    completed += 1
        if pending:
            ray.get(writer.append_records.remote(pending))
        return {"episodes": completed, "transitions": transitions, "official_scores": scores}


class EvaluationWorker:
    """Run closed-loop scenarios and report the official nuPlan score.

    `scorer` decides which candidate to execute each step:
        "candidate0" -- always candidate 0 (the raw planner; the baseline).
        "random"     -- a random candidate.
        "critic"     -- the argmax of a trained critic's scores (needs a
                        checkpoint + visible_horizon; critic.py must be
                        implemented for this to run).
    """

    def __init__(
        self,
        factory_path: str,
        factory_kwargs: Mapping[str, Any],
        scorer: str,
        seed: int,
        critic_kwargs: Optional[Mapping[str, Any]] = None,
        visible_horizon: Optional[int] = None,
        checkpoint: Optional[str] = None,
        q_reduction: str = "mean",
        device: str = "cuda",
    ) -> None:
        self.factory = load_object(factory_path)(**dict(factory_kwargs))
        self.scorer = scorer
        self.rng = np.random.default_rng(seed)
        self.visible_horizon = visible_horizon
        self.q_reduction = q_reduction
        self.device = torch.device(device)
        self.critic = None
        if scorer == "critic":
            if critic_kwargs is None or visible_horizon is None or checkpoint is None:
                raise ValueError("critic scorer requires critic_kwargs, visible_horizon, and checkpoint")
            self.critic = HorizonCritic(**dict(critic_kwargs)).to(self.device).eval()
            state = torch.load(checkpoint, map_location=self.device)
            self.critic.load_state_dict(state["critic"])

    def _action(self, observation: Any) -> int:
        if self.scorer == "candidate0":
            return 0
        if self.scorer == "random":
            return int(self.rng.integers(0, observation.batch.candidates.shape[0]))
        if self.scorer == "critic":
            batch = observation.batch
            scores = self.critic.score(
                batch.scene_tokens[None].to(self.device),
                batch.scene_mask[None].to(self.device),
                batch.candidates[None].to(self.device),
                visible_horizon=int(self.visible_horizon),
                reduction=self.q_reduction,
            )
            return int(scores.argmax(dim=1).item())
        raise ValueError(f"unknown scorer {self.scorer}")

    def evaluate(self, episode_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        # None -> every scenario this worker built (used when scenarios are sharded
        # across workers so each worker only builds its own slice).
        if episode_ids is None:
            episode_ids = list(range(len(self.factory)))
        records = []
        scores = []
        steps = 0
        for episode_id in episode_ids:
            env = self.factory(int(episode_id))
            scenario = getattr(getattr(env, "simulation", None), "scenario", None)
            observation = env.reset(episode_id=int(episode_id))
            scenario_steps = 0
            while True:
                transition = env.step(self._action(observation))
                steps += 1
                scenario_steps += 1
                if transition.done:
                    info = transition.info
                    score = float(info["official_score"])
                    scores.append(score)
                    records.append(
                        {
                            "episode_id": int(episode_id),
                            "scenario_token": getattr(scenario, "token", None),
                            "scenario_type": getattr(scenario, "scenario_type", None),
                            "log_name": getattr(scenario, "log_name", None),
                            "steps": scenario_steps,
                            "official_score": score,
                            "metrics": {k: float(v) for k, v in dict(info.get("official_metric_scores", {})).items()},
                        }
                    )
                    break
                observation = transition.next_observation
        return {
            "episodes": len(scores),
            "steps": steps,
            "mean_official_score": float(np.mean(scores)) if scores else float("nan"),
            "records": records,
        }
