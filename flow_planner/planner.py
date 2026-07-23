
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Optional, Sequence, Type
import hydra
from hydra.utils import instantiate
import omegaconf

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from flow_planner.data.data_process.data_processor import DataProcessor
from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.critic_rl.types import CandidateBatch

def identity(ego_state, predictions):
    return predictions


class FlowPlanner(AbstractPlanner):
    def __init__(
            self,
            config_path,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
            use_cfg: bool = True,
            cfg_weight: float = 1.0,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        config = omegaconf.OmegaConf.load(config_path)
        # The released model_config omits the training dataset section while
        # retaining two normalizer interpolations into it.
        if omegaconf.OmegaConf.select(config, "data.dataset.train.future_downsampling_method") is None:
            omegaconf.OmegaConf.update(
                config, "data.dataset.train.future_downsampling_method", "uniform", force_add=True
            )
        if omegaconf.OmegaConf.select(config, "data.dataset.train.predicted_neighbor_num") is None:
            omegaconf.OmegaConf.update(
                config, "data.dataset.train.predicted_neighbor_num", int(config.model.neighbor_pred_num), force_add=True
            )
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = instantiate(config.model)

        self.core = instantiate(config.core)

        self.data_processor = DataProcessor(None)

        self.use_cfg = use_cfg

        self.cfg_weight = cfg_weight
        self._model_loaded = False
        
    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None and not self._model_loaded:
            state_dict = torch.load(self._ckpt_path, weights_only=True, map_location=self._device)
            
            if self._ema_enabled and 'ema_state_dict' in state_dict:
                state_dict = state_dict['ema_state_dict']
            elif "model" in state_dict:
                state_dict = state_dict['model']
            # Support both DDP-prefixed and plain release checkpoints.
            model_state_dict = {
                (key[len("module."):] if key.startswith("module.") else key): value
                for key, value in state_dict.items()
            }
            self._planner.load_state_dict(model_state_dict)
            self._model_loaded = True
        elif self._ckpt_path is None and not self._model_loaded:
            print("load random model")
            self._model_loaded = True
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        data = NuPlanDataSample(
            batched=(model_inputs['ego_current_state'].dim() > 1),
            ego_past=model_inputs['ego_agent_past'],
            ego_current=model_inputs['ego_current_state'],
            neighbor_past=model_inputs['neighbor_agents_past'],
            lanes=model_inputs['lanes'],
            lanes_speedlimit=model_inputs['lanes_speed_limit'],
            lanes_has_speedlimit=model_inputs['lanes_has_speed_limit'],
            routes=model_inputs['route_lanes'],
            routes_speedlimit=model_inputs['route_lanes_speed_limit'],
            routes_has_speedlimit=model_inputs['route_lanes_has_speed_limit'],
            map_objects=model_inputs['static_objects']
        )

        return data

    def outputs_to_trajectory(self, outputs: torch.Tensor, ego_state_history: Deque[EgoState], candidate_index: int = 0) -> List[InterpolatableState]:
        predictions = outputs[candidate_index, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states

    def compute_candidate_trajectories(
        self,
        current_input: PlannerInput,
        num_candidates: int,
        seeds: Optional[Sequence[int]] = None,
    ) -> CandidateBatch:
        """Return N complete trajectories for one frozen planner input.

        Candidate 0 is the same one-candidate Flow-Planner sample used by
        compute_planner_trajectory() under the same RNG/seed convention.
        Additional rows are extra sampled candidates for critic scoring.
        """
        inputs = self.planner_input_to_model_inputs(current_input)
        outputs, scene_tokens, scene_mask = self.core.inference_candidates(
            self._planner,
            inputs,
            num_candidates=num_candidates,
            seeds=seeds,
            use_cfg=self.use_cfg,
            cfg_weight=self.cfg_weight,
        )
        trajectories = tuple(
            InterpolatedTrajectory(
                trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states, index)
            )
            for index in range(num_candidates)
        )
        resolved_seeds = tuple(int(seed) for seed in seeds) if seeds is not None else tuple()
        return CandidateBatch(
            trajectories=trajectories,
            candidates=outputs[:, 0].detach().cpu(),
            scene_tokens=scene_tokens[0].detach().cpu(),
            scene_mask=scene_mask[0].detach().cpu().to(torch.bool),
            seeds=resolved_seeds,
        )
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        return self.compute_candidate_trajectories(
            current_input,
            num_candidates=1,
            seeds=None,
        ).trajectories[0]
    