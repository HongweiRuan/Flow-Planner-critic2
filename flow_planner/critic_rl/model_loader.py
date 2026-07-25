"""Load the Flow-Planner model standalone (no nuPlan scenario needed).

Used by the critic trainer to (a) warm-start / embed the scene encoder and
(b) re-inference candidate trajectories at bootstrap time. Replicates the model
instantiation + checkpoint loading that FlowPlanner (the planner) does in
__init__/initialize, but without needing a PlannerInitialization.
"""
import omegaconf
import torch
from hydra.utils import instantiate


def load_flow_planner_model(config_path: str, ckpt_path: str, device: str = "cuda", enable_ema: bool = True):
    config = omegaconf.OmegaConf.load(config_path)
    # The released model_config omits the training dataset section while retaining
    # two normalizer interpolations into it (same patch as planner.py).
    if omegaconf.OmegaConf.select(config, "data.dataset.train.future_downsampling_method") is None:
        omegaconf.OmegaConf.update(config, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    if omegaconf.OmegaConf.select(config, "data.dataset.train.predicted_neighbor_num") is None:
        omegaconf.OmegaConf.update(
            config, "data.dataset.train.predicted_neighbor_num", int(config.model.neighbor_pred_num), force_add=True
        )

    model = instantiate(config.model)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, weights_only=True, map_location=device)
        if enable_ema and isinstance(state, dict) and "ema_state_dict" in state:
            state = state["ema_state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 20:
            raise RuntimeError(f"too many missing keys ({len(missing)}) -- config/ckpt mismatch: {missing[:5]}")
    model.eval()
    model.to(device)
    model.device = device  # the model uses self.device internally for noise / cfg tensors
    return model
