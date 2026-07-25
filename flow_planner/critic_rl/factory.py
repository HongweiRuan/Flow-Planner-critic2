import os
from pathlib import Path
from typing import Optional, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flow_planner.critic_rl.env import NuPlanReplanningEnv
from flow_planner.critic_rl.reward import OfficialStepReward


class NuPlanHydraFactory:
    """Build reusable one-step environments with official nuPlan builders."""

    def __init__(
        self,
        overrides: Sequence[str],
        num_candidates: int,
        base_seed: int = 0,
        max_steps: Optional[int] = None,
        execution_horizon: int = 1,
        output_dir: str = "/tmp/flow_planner_critic_rl",
        data_root: str = "/avl-west/nuplan/nuplan-v1.1",
        maps_root: str = "/avl-west/nuplan/maps",
        exp_root: str = "/tmp/nuplan_exp",
        config_dir: str = "/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/ruh1syv/nuplan-devkit/nuplan/planning/script/config/simulation",
    ) -> None:
        os.environ["NUPLAN_DATA_ROOT"] = data_root
        os.environ["NUPLAN_MAPS_ROOT"] = maps_root
        os.environ["NUPLAN_EXP_ROOT"] = exp_root
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        merged_overrides = list(overrides)
        merged_overrides.extend(
            [
                f"output_dir={output_dir}",
                "worker=sequential",
                "run_metric=false",
                "enable_simulation_progress_bar=false",
                "hydra.searchpath=[pkg://flow_planner.nuplan_simulation.scenario_filter,pkg://flow_planner.nuplan_simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
            ]
        )
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="default_simulation", overrides=merged_overrides)
        OmegaConf.resolve(cfg)
        self.cfg = cfg
        self.num_candidates = int(num_candidates)
        self.base_seed = int(base_seed)
        self.max_steps = max_steps
        self.execution_horizon = int(execution_horizon)
        self._build()

    def _build(self) -> None:
        from nuplan.planning.script.builders.metric_builder import build_metrics_engines
        from nuplan.planning.script.builders.planner_builder import build_planners
        from nuplan.planning.script.builders.simulation_builder import build_simulations
        from nuplan.planning.script.builders.worker_pool_builder import build_worker

        worker = build_worker(self.cfg)
        planner = build_planners(self.cfg.planner, scenario=None)[0]
        runners = build_simulations(
            cfg=self.cfg,
            worker=worker,
            callbacks=[],
            callbacks_worker=None,
            pre_built_planners=[planner],
        )
        if not runners:
            raise RuntimeError("nuPlan factory did not produce any scenarios")
        scenarios = [runner.scenario for runner in runners]
        metric_engines = build_metrics_engines(self.cfg, scenarios)
        self._entries = [
            (
                runner.simulation,
                runner.planner,
                metric_engines[runner.scenario.scenario_type],
            )
            for runner in runners
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def scene_encoder(self):
        """The Flow-Planner scene encoder module (structure for the critic to
        embed / load trained weights into)."""
        planner = self._entries[0][1]
        return planner._planner.model_encoder

    def __call__(self, episode_id: int) -> NuPlanReplanningEnv:
        simulation, planner, metrics_engine = self._entries[int(episode_id) % len(self._entries)]
        reward = OfficialStepReward(metrics_engine, normalization_steps=self.max_steps)
        return NuPlanReplanningEnv(
            simulation=simulation,
            planner=planner,
            reward=reward,
            num_candidates=self.num_candidates,
            base_seed=self.base_seed,
            max_steps=self.max_steps,
            execution_horizon=self.execution_horizon,
        )
