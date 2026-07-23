"""Critic model architecture -- STUB.

The critic scores planner candidates: given the scene and N candidate
trajectories, it returns one scalar per candidate, and evaluation picks the
argmax. `evaluate.py --scorer critic` calls `HorizonCritic(**critic_kwargs)`
then `.score(...)`; `collect.py` does not need this class at all.

Fill in the architecture here. Keep the constructor kwargs and the `score()`
signature below so `evaluate.py` keeps working unchanged.
"""

from torch import nn
import torch


class HorizonCritic(nn.Module):
    """Scores candidates over a visible horizon. NOT IMPLEMENTED YET.

    Constructor kwargs come straight from the `critic:` block of the config
    (horizon, state_dim, context_dim, d_model, ...). They are stored so a
    checkpoint can be described, but no layers are built yet.
    """

    def __init__(
        self,
        horizon: int,
        state_dim: int = 4,
        context_dim: int = 192,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)

    def forward(
        self,
        scene_tokens: torch.Tensor,
        scene_mask: torch.Tensor,
        candidates: torch.Tensor,
        visible_horizon: int,
    ):
        raise NotImplementedError("HorizonCritic architecture is a stub -- implement critic.py")

    @torch.inference_mode()
    def score(
        self,
        scene_tokens: torch.Tensor,
        scene_mask: torch.Tensor,
        candidates: torch.Tensor,
        visible_horizon: int,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Return [B, N] candidate scores. NOT IMPLEMENTED YET.

        Expected shapes (what evaluate.py passes in):
          scene_tokens [B, S, context_dim], scene_mask [B, S] (True = valid),
          candidates   [B, N, horizon, state_dim].
        `visible_horizon` = how many steps of each candidate the critic may see;
        `reduction` = how to combine twin heads if you use them ("mean"/"min").
        """
        raise NotImplementedError("HorizonCritic.score is a stub -- implement critic.py")
