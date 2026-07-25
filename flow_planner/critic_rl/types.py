from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch


# Canonical order of the dense reward components persisted per transition. Keeping
# the breakdown (rather than only the scalar sum) lets reward weights be changed
# offline without re-running collection. Any new component must be appended here
# so existing replays keep a stable column layout.
REWARD_COMPONENT_NAMES: Tuple[str, ...] = (
    "progress",
    "time_to_collision",
    "speed_limit",
    "comfort",
    "making_progress",
    "collision",
    "drivable_area",
    "driving_direction",
)


@dataclass(frozen=True)
class CandidateBatch:
    """Planner output for one closed-loop decision.

    `scene_inputs` are the *inputs* to the Flow-Planner scene encoder (the clean,
    no-CFG normalized tensors: neighbors / static / lanes / lanes_speed_limit /
    lanes_has_speed_limit / routes), NOT the frozen 192-d tokens. The critic
    re-encodes them with its own encoder, and training re-inferences fresh
    candidates from them. Each tensor has its per-scene shape (batch dim dropped).
    """

    trajectories: Tuple[Any, ...]
    candidates: torch.Tensor  # [N, H, state_dim]
    scene_inputs: Mapping[str, torch.Tensor]
    seeds: Tuple[int, ...] = tuple()

    def __post_init__(self) -> None:
        if self.candidates.ndim != 3:
            raise ValueError("candidates must have shape [N, H, D]")
        if len(self.trajectories) != self.candidates.shape[0]:
            raise ValueError("trajectory and candidate counts differ")


@dataclass(frozen=True)
class CriticObservation:
    batch: CandidateBatch
    episode_id: int
    step_index: int


@dataclass(frozen=True)
class Transition:
    observation: CriticObservation
    action: int
    reward: float
    next_observation: Optional[CriticObservation]
    done: bool
    info: Mapping[str, Any]
