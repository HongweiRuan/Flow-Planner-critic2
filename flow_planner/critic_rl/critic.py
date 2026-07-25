"""Critic model: HorizonCritic.

Scores planner candidate trajectories. Given a scene and N candidate
trajectories, returns one scalar per candidate; evaluation / the Bellman target
pick the argmax / max.

Design (see the design discussion): the critic embeds its OWN copy of the
Flow-Planner scene encoder (warm-started from the planner's weights), so the
scene representation can be adapted to value estimation instead of being locked
to the planner's frozen tokens. `freeze_encoder` turns that adaptation on/off --
freeze=True reproduces "use the planner's tokens", freeze=False fine-tunes. This
is exactly the A/B knob for "is the planning encoder the bottleneck?".

Pipeline:
    raw scene (6 encoder-input tensors)
        -> embedded FlowPlannerEncoder (freezable) -> 107 scene tokens [B,S,192]
    candidate trajectories [B,N,H,4]
        -> truncate to visible_horizon L -> per-step tokens (+Delta, +time emb)
        -> cross-attend the scene tokens -> attention-pool over the L steps
        -> twin-Q heads -> [B,N] scores

`visible_horizon` is realized purely by truncation, so one set of weights serves
any look-ahead L and the score depends only on the first L steps.
"""

import copy
from typing import Mapping, Optional, Tuple

import torch
from torch import nn


# Keys of the scene dict the critic consumes -- identical to the planner's
# FlowPlanner.extract_encoder_inputs output, so the embedded encoder is fed
# exactly what it was trained on.
SCENE_KEYS = ("neighbors", "static", "lanes", "lanes_speed_limit", "lanes_has_speed_limit", "routes")


class HorizonCritic(nn.Module):
    def __init__(
        self,
        scene_encoder: nn.Module,
        horizon: int,
        state_dim: int = 4,
        encoder_hidden: int = 192,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        use_delta: bool = True,
        freeze_encoder: bool = True,
    ) -> None:
        """
        scene_encoder: a Flow-Planner `FlowPlannerEncoder` instance (deep-copied
            here and warm-started from planner weights by the caller). Its
            forward(**scene) must return encoder_outputs with
            'encodings' = (tokens_a, tokens_b) and 'masks' = (mask_a, mask_b)
            where True = valid; we concatenate them into [B,S,encoder_hidden].
        horizon: max candidate length H (size of the time-embedding table).
        """
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.use_delta = bool(use_delta)

        # --- scene: embed our own copy of the planner encoder ---
        self.scene_encoder = copy.deepcopy(scene_encoder)
        self.set_encoder_frozen(freeze_encoder)
        self.scene_proj = (
            nn.Identity() if encoder_hidden == d_model else nn.Linear(encoder_hidden, d_model)
        )

        # --- candidate trajectory tokenizer ---
        traj_in = state_dim * 2 if use_delta else state_dim  # optionally append step deltas (velocity)
        self.traj_projection = nn.Linear(traj_in, d_model)
        self.time_embedding = nn.Parameter(torch.empty(self.horizon, d_model))
        nn.init.normal_(self.time_embedding, std=0.02)

        # --- candidate steps cross-attend the scene ---
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

        # --- attention pool over the L steps (learned query) ---
        self.pool_query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # --- twin-Q heads ---
        self.q1 = self._head(d_model)
        self.q2 = self._head(d_model)

    @staticmethod
    def _head(d_model: int) -> nn.Module:
        return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def set_encoder_frozen(self, frozen: bool) -> None:
        """A/B knob: freeze=True => planner tokens; freeze=False => fine-tune the encoder."""
        self.encoder_frozen = bool(frozen)
        for p in self.scene_encoder.parameters():
            p.requires_grad_(not self.encoder_frozen)

    def train(self, mode: bool = True):  # keep a frozen encoder in eval mode (no dropout drift)
        super().train(mode)
        if self.encoder_frozen:
            self.scene_encoder.eval()
        return self

    def _encode_scene(self, scene: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the embedded encoder -> scene tokens [B,S,H] and mask [B,S] (True=valid)."""
        inputs = {k: scene[k] for k in SCENE_KEYS}
        # The lane encoder uses has_speed_limit as a boolean index; replay stores
        # it as float, so cast (harmless if already bool, e.g. at eval time).
        inputs["lanes_has_speed_limit"] = inputs["lanes_has_speed_limit"].bool()
        if self.encoder_frozen:
            with torch.no_grad():
                out = self.scene_encoder(**inputs)
        else:
            out = self.scene_encoder(**inputs)
        scene_tokens = torch.cat(out["encodings"], dim=1)
        scene_mask = torch.cat(out["masks"], dim=1)
        return scene_tokens, scene_mask.to(torch.bool)

    def _tokenize_candidates(self, candidates: torch.Tensor, visible_horizon: int) -> torch.Tensor:
        """[B,N,H,D] -> [B,N,L,d] with optional step-deltas and time embedding."""
        B, N, full_H, D = candidates.shape
        if full_H != self.horizon:
            raise ValueError(f"expected candidate horizon {self.horizon}, got {full_H}")
        if not 1 <= visible_horizon <= full_H:
            raise ValueError("visible_horizon must be in [1, H]")
        L = int(visible_horizon)
        cand = candidates[:, :, :L, :]  # truncate: score depends only on first L steps
        if self.use_delta:
            delta = torch.zeros_like(cand)
            delta[:, :, 1:] = cand[:, :, 1:] - cand[:, :, :-1]  # step-to-step motion
            cand = torch.cat([cand, delta], dim=-1)
        tok = self.traj_projection(cand)
        tok = tok + self.time_embedding[:L][None, None]
        return tok

    def forward(
        self,
        scene: Mapping[str, torch.Tensor],
        candidates: torch.Tensor,
        visible_horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return twin-Q scores (q1, q2), each [B, N]."""
        if candidates.ndim != 4:
            raise ValueError("candidates must have shape [B, N, H, D]")
        B, N, _, _ = candidates.shape
        L = int(visible_horizon)

        scene_tokens, scene_mask = self._encode_scene(scene)  # [B,S,Hc], [B,S] valid
        memory = self.scene_proj(scene_tokens)  # [B,S,d]
        mem_pad = ~scene_mask  # TransformerDecoder wants True = pad
        S = memory.shape[1]

        tok = self._tokenize_candidates(candidates, L).reshape(B * N, L, -1)  # [B*N,L,d]
        mem = memory[:, None].expand(-1, N, -1, -1).reshape(B * N, S, -1)
        mem_pad_e = mem_pad[:, None].expand(-1, N, -1).reshape(B * N, S)

        decoded = self.decoder(tgt=tok, memory=mem, memory_key_padding_mask=mem_pad_e)  # [B*N,L,d]

        query = self.pool_query.expand(B * N, -1, -1)
        pooled = self.pool_attn(query, decoded, decoded, need_weights=False)[0][:, 0]  # [B*N,d]

        q1 = self.q1(pooled).reshape(B, N)
        q2 = self.q2(pooled).reshape(B, N)
        return q1, q2

    @torch.inference_mode()
    def score(
        self,
        scene: Mapping[str, torch.Tensor],
        candidates: torch.Tensor,
        visible_horizon: int,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Scorer over the twin heads: 'mean' (default) or conservative 'min'. -> [B,N]."""
        q1, q2 = self(scene, candidates, visible_horizon)
        if reduction == "mean":
            return 0.5 * (q1 + q2)
        if reduction == "min":
            return torch.minimum(q1, q2)
        raise ValueError(f"unknown q reduction {reduction!r}")
