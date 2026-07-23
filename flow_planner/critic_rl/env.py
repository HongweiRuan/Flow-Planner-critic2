from typing import Any, Dict, Optional

from flow_planner.critic_rl.reward import OfficialStepReward
from flow_planner.critic_rl.types import CriticObservation, Transition


class NuPlanReplanningEnv:
    """One nuPlan propagation per candidate-index action.

    ``max_steps`` counts macro decisions (``step()`` calls), not simulator
    propagations: with ``execution_horizon = k`` an episode capped at
    ``max_steps`` runs up to ``max_steps * k`` simulator steps.
    """

    def __init__(
        self,
        simulation: Any,
        planner: Any,
        reward: OfficialStepReward,
        num_candidates: int,
        base_seed: int = 0,
        max_steps: Optional[int] = None,
        execution_horizon: int = 1,
    ) -> None:
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if execution_horizon < 1:
            raise ValueError("execution_horizon must be positive")
        self.simulation = simulation
        self.planner = planner
        self.reward = reward
        self.num_candidates = int(num_candidates)
        self.base_seed = int(base_seed)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.execution_horizon = int(execution_horizon)
        self._episode_id = -1
        self._step_index = 0
        self._observation: Optional[CriticObservation] = None

    def _candidate_seeds(self):
        modulus = 2**63 - 1
        start = self.base_seed + self._episode_id * 1_000_003 + self._step_index * self.num_candidates
        return tuple((start + index) % modulus for index in range(self.num_candidates))

    def _plan(self) -> CriticObservation:
        planner_input = self.simulation.get_planner_input()
        batch = self.planner.compute_candidate_trajectories(
            planner_input,
            num_candidates=self.num_candidates,
            seeds=self._candidate_seeds(),
        )
        return CriticObservation(batch=batch, episode_id=self._episode_id, step_index=self._step_index)

    def reset(self, episode_id: Optional[int] = None) -> CriticObservation:
        self._episode_id = self._episode_id + 1 if episode_id is None else int(episode_id)
        self._step_index = 0
        self.reward.reset()
        initialization = self.simulation.initialize()
        self.planner.initialize(initialization)
        self._observation = self._plan()
        return self._observation

    def step(self, action: int) -> Transition:
        if self._observation is None:
            raise RuntimeError("reset must be called before step")
        if not 0 <= int(action) < self.num_candidates:
            raise ValueError(f"action must be in [0, {self.num_candidates})")

        observation = self._observation
        selected_full_trajectory = observation.batch.trajectories[int(action)]

        # Commit `execution_horizon` simulation steps of the selected chunk before
        # replanning. execution_horizon == 1 is the closed-loop k=1 setting; h gives
        # the h-step receding-horizon macro-action whose accumulated reward is the
        # C_h target. gamma is fixed to 1, so the macro-reward is an undiscounted sum.
        macro_reward = 0.0
        macro_raw_reward = 0.0
        macro_components: Dict[str, float] = {}
        reward_step = None
        done = False
        for sub_step in range(self.execution_horizon):
            self.simulation.propagate(selected_full_trajectory)
            running = self.simulation.is_simulation_running()
            is_last_sub_step = sub_step + 1 == self.execution_horizon
            capped = self.max_steps is not None and self._step_index + 1 >= self.max_steps
            done = (not running) or (is_last_sub_step and capped)
            reward_step = self.reward.step(self.simulation.history, self.simulation.scenario, done=done)
            macro_reward += float(reward_step.reward)
            macro_raw_reward += float(getattr(reward_step, "raw_reward", reward_step.reward))
            # The stored breakdown must sum over the whole macro transition, matching
            # macro_reward, not just the final sub-step.
            for name, value in dict(getattr(reward_step, "components", {})).items():
                macro_components[name] = macro_components.get(name, 0.0) + float(value)
            if done:
                break

        self._step_index += 1
        next_observation = None if done else self._plan()
        info: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "step_index": observation.step_index,
            "candidate_seeds": observation.batch.seeds,
            "execution_horizon": self.execution_horizon,
            "proxy_score": reward_step.proxy_score,
            "raw_step_reward": macro_raw_reward,
            "reward_components": macro_components,
            "terminal_correction": reward_step.correction,
        }
        metric_scores = getattr(reward_step, "metric_scores", {})
        if metric_scores:
            info["official_metric_scores"] = dict(metric_scores)
        if reward_step.official_score is not None:
            info["official_score"] = reward_step.official_score
            info["reward_identity_error"] = abs(self.reward.reward_sum - reward_step.official_score)
        transition = Transition(
            observation=observation,
            action=int(action),
            reward=float(macro_reward),
            next_observation=next_observation,
            done=done,
            info=info,
        )
        self._observation = next_observation
        return transition
