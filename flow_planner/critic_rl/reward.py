from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from flow_planner.critic_rl.dense_reward import DenseRewardConfig, NuPlanStepRewardComputer


WEIGHTED_METRICS = {
    "ego_progress_along_expert_route": 5.0,
    "time_to_collision_within_bound": 5.0,
    "speed_limit_compliance": 4.0,
    "ego_is_comfortable": 2.0,
}
MULTIPLIER_METRICS = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
)


def _metric_score(statistics: Sequence[Any]) -> Optional[float]:
    for result in statistics:
        score = getattr(result, "metric_score", None)
        if score is not None:
            return float(score)
    return None


def flatten_metric_scores(metric_results: Mapping[str, Sequence[Any]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for metric_name, statistics in metric_results.items():
        score = _metric_score(statistics)
        if score is not None:
            scores[metric_name] = score
    return scores


def aggregate_official_score(
    scores: Mapping[str, float],
    include_making_progress: bool = True,
) -> float:
    """In-memory equivalent of nuPlan's closed-loop weighted aggregator."""
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, weight in WEIGHTED_METRICS.items():
        if name in scores:
            weighted_sum += weight * float(scores[name])
            weight_sum += weight
    average = weighted_sum / weight_sum if weight_sum else 0.0
    multiplier = 1.0
    for name in MULTIPLIER_METRICS:
        if name == "ego_is_making_progress" and not include_making_progress:
            continue
        if name in scores:
            multiplier *= float(scores[name])
    return average * multiplier


@dataclass(frozen=True)
class RewardStep:
    reward: float
    raw_reward: float
    components: Mapping[str, float]
    proxy_score: float
    official_score: Optional[float]
    correction: float
    metric_scores: Mapping[str, float]


class OfficialStepReward:
    """Dense per-step reward with exact official terminal correction."""

    def __init__(
        self,
        metrics_engine: Any,
        normalization_steps: Optional[int] = None,
        dense_config: Optional[DenseRewardConfig] = None,
        step_computer: Optional[Any] = None,
    ) -> None:
        self.metrics_engine = metrics_engine
        self.step_computer = step_computer or NuPlanStepRewardComputer(
            metrics_engine,
            normalization_steps=normalization_steps,
            config=dense_config,
        )
        self.reset()

    def reset(self) -> None:
        self._reward_sum = 0.0
        self._raw_reward_sum = 0.0
        self._official_score: Optional[float] = None
        self.step_computer.reset()

    @property
    def reward_sum(self) -> float:
        return self._reward_sum

    @property
    def official_score(self) -> Optional[float]:
        return self._official_score

    def _compute_scores(self, history: Any, scenario: Any) -> Dict[str, float]:
        results = self.metrics_engine.compute_metric_results(history=history, scenario=scenario)
        return flatten_metric_scores(results)

    def step(self, history: Any, scenario: Any, done: bool) -> RewardStep:
        components = {
            name: float(value)
            for name, value in self.step_computer.compute(history, scenario).items()
        }
        raw_reward = float(sum(components.values()))
        self._raw_reward_sum += raw_reward
        if done:
            scores = self._compute_scores(history, scenario)
            official = aggregate_official_score(scores, include_making_progress=True)
            correction = official - (self._reward_sum + raw_reward)
            reward = raw_reward + correction
            self._reward_sum += reward
            self._official_score = official
            return RewardStep(
                reward=reward,
                raw_reward=raw_reward,
                components=components,
                proxy_score=self._raw_reward_sum,
                official_score=official,
                correction=correction,
                metric_scores=scores,
            )

        self._reward_sum += raw_reward
        return RewardStep(
            reward=raw_reward,
            raw_reward=raw_reward,
            components=components,
            proxy_score=self._raw_reward_sum,
            official_score=None,
            correction=0.0,
            metric_scores={},
        )
