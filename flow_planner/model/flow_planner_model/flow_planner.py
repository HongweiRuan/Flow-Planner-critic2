import re
import os
import sys
from typing import Literal, Callable, Any, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from flow_planner.model.model_base import DiffusionADPlanner
from flow_planner.model.model_utils.input_preprocess import ModelInputProcessor
from flow_planner.model.model_utils.traj_tool import traj_chunking, assemble_actions
from flow_planner.data.dataset.nuplan import NuPlanDataSample

class FlowPlanner(DiffusionADPlanner):

    def __init__(
        self,
        model_encoder,
        model_decoder,

        flow_ode,
        
        model_type: Literal['x_start', 'noise', 'velocity'] = 'x_start',
        kinematic: Literal["waypoints", "velocity", "acceleration"] = 'waypoints',
    
        assemble_method='linear',
        
        data_processor: ModelInputProcessor = None,
        
        device='cuda',
        **planner_params
    ):
        
        super(FlowPlanner, self).__init__()
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self._model_type = model_type
        self.device = device
        
        self.flow_ode = flow_ode # including flow matching path and ode solver
        self.cfg_prob = planner_params['cfg_prob']
        self.cfg_weight = planner_params['cfg_weight']
        self.cfg_type = planner_params['cfg_type']

        self.kinematic = kinematic
        
        self.assemble_method = assemble_method
        
        self.data_processor = data_processor

        self.planner_params = planner_params # including the action_len, future_len etc.
        self.action_num = (self.planner_params['future_len'] - self.planner_params['action_overlap']) // (self.planner_params['action_len'] - self.planner_params['action_overlap'])
        
        self.basic_loss = nn.MSELoss(reduction='none')
        
    def prepare_model_input(self, cfg_flags, data: NuPlanDataSample, use_cfg, is_training):
        B = data.ego_current.shape[0]

        if is_training:
            # modify the data sample according to cfg_flags
            cfg_type = self.cfg_type
            if cfg_type == 'neighbors':
                neighbor_num = self.planner_params['neighbor_num']
                cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                mask_flags = cfg_flags.view(B, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                mask_flags[:, cfg_neighbor_num:, :] = 1
                data.neighbor_past *= mask_flags
            elif cfg_type == 'lanes':
                data.lanes = data.lanes * cfg_flags.view(B, *([1] * (data.lanes.dim()-1)))

        else:
            if use_cfg:
                data = data.repeat(2)
                cfg_type = self.cfg_type
                if cfg_type == 'neighbors':
                    neighbor_num = self.planner_params['neighbor_num']
                    cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                    mask_flags = cfg_flags.view(B * 2, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                    mask_flags[:, cfg_neighbor_num:, :] = 1
                    data.neighbor_past *= mask_flags
                elif cfg_type == 'lanes':
                    data.lanes = data.lanes * cfg_flags.view(B * 2, *([1] * (data.lanes.dim()-1)))
           
        model_inputs, gt = self.data_processor.sample_to_model_input(
            data, device=self.device, kinematic=self.kinematic, is_training=is_training
        )
            
        model_inputs.update({'cfg_flags': cfg_flags})
        
        return model_inputs, gt
        
    def extract_encoder_inputs(self, inputs):
        
        encoder_inputs = {
            'neighbors': inputs['neighbor_past'],
            'lanes': inputs['lanes'],
            'lanes_speed_limit': inputs['lanes_speedlimit'],
            'lanes_has_speed_limit': inputs['lanes_has_speedlimit'],
            'static': inputs['map_objects'],
            'routes': inputs['routes']
        }
        return encoder_inputs
    
    def extract_decoder_inputs(self, encoder_outputs, inputs):
        model_extra = dict(cfg_flags=inputs['cfg_flags'] if 'cfg_flags' in inputs.keys() else None,)
        model_extra.update(encoder_outputs)
        return model_extra
    
    def encoder(self, **encoder_inputs):
        return self.model_encoder(**encoder_inputs)
    
    def decoder(self, x, t, **model_extra):
        return self.model_decoder(x, t, **model_extra)
        
    def forward(self, data: NuPlanDataSample, mode='train', **params):
        if mode == 'train':
            return self.forward_train(data)
        elif mode == 'inference':
            return self.forward_inference(data, params['use_cfg'], params['cfg_weight'])
    
    def forward_train(self, data: NuPlanDataSample):
        '''
        Forward a training step and compute the training loss.
        1. generate cfg_flags
        2. preprocess (masking) according to the cfg_flags
        3. model forward
        4. compute basic mse loss
        
        Return:
            prediction: the raw prediction of the model, specified by model.prediction_type;
            loss_dict: a dict of loss containing unreduced mse loss, consistency loss and neighbor prediction loss (if one exists).
        '''
        B = data.ego_current.shape[0]
        roll_dice = torch.rand((B, 1))
        cfg_flags = (roll_dice > self.cfg_prob).to(torch.int32).to(self.device) # NOTE: 1 for conditioned (unmasked), 0 for unconditioned (masked)
        model_inputs, gt = self.prepare_model_input(cfg_flags, data, use_cfg=False, is_training=True) # note that the cfg_flags are packed into the model_inputs
        
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        B, P, T_, D = gt.shape
        
        noised_traj, target, t = self.flow_ode.sample(gt[:, :, 1:, :], self._model_type)
        noised_traj_tokens = traj_chunking(noised_traj, self.planner_params['action_len'], self.planner_params['action_overlap'])
        noised_traj_tokens = torch.cat(noised_traj_tokens, dim=1)
        target_tokens = traj_chunking(target, self.planner_params['action_len'], self.planner_params['action_overlap'])
        target_tokens = torch.cat(target_tokens, dim=1)
        
        prediction = self.decoder(noised_traj_tokens, t, **decoder_model_extra)
        
        loss_dict = {}
        batch_loss = self.basic_loss(prediction, target_tokens)
        loss_dict['batch_loss'] = batch_loss
        
        loss = torch.sum(batch_loss, dim=-1) # (B, action_num, action_length, dim)
        loss_dict['ego_planning_loss'] = loss.mean()

        if self.planner_params['action_overlap'] > 0:
            consistency_loss = [torch.mean(torch.sum(self.basic_loss(prediction[:, i:i+1, -self.planner_params['action_overlap']:, :], prediction[:, i+1:i+2, :self.planner_params['action_overlap'], :]), dim=-1)) for i in range(0, prediction.shape[1]-2)]
            loss_dict['consistency_loss'] = sum(consistency_loss) / len(consistency_loss)
        else:
            loss_dict['consistency_loss'] = torch.tensor(0.0, device=loss.device)
        
        assert not torch.isnan(loss).sum(), f"loss is NaN"
        
        return prediction, loss_dict
    
    def forward_inference(self, data: NuPlanDataSample, use_cfg=True, cfg_weight=None):
        B = data.ego_current.shape[0]
        model_inputs, encoder_outputs = self.encode_scene(data, use_cfg=use_cfg)
        x_init = torch.randn((B, self.action_num, self.planner_params['action_len'], self.planner_params['state_dim']), device=self.device)
        return self.sample_from_encoded(
            model_inputs=model_inputs,
            encoder_outputs=encoder_outputs,
            x_init=x_init,
            use_cfg=use_cfg,
            cfg_weight=cfg_weight,
        )

    def encode_scene(self, data: NuPlanDataSample, use_cfg=True):
        """Encode a scene once. Candidate sampling must not re-encode the scene."""
        B = data.ego_current.shape[0]
        if use_cfg:
            cfg_flags = torch.cat([torch.ones((B,), device=self.device), torch.zeros((B,), device=self.device)], dim=0).to(torch.int32)
        else:
            cfg_flags = torch.ones((B,), device=self.device).to(torch.int32)
        
        model_inputs, _ = self.prepare_model_input(cfg_flags, data, use_cfg, is_training=False)
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)
        return model_inputs, encoder_outputs

    @staticmethod
    def _repeat_batch(value, repeats):
        if torch.is_tensor(value):
            return value.repeat_interleave(repeats, dim=0)
        if isinstance(value, tuple):
            return tuple(FlowPlanner._repeat_batch(item, repeats) for item in value)
        if isinstance(value, list):
            return [FlowPlanner._repeat_batch(item, repeats) for item in value]
        if isinstance(value, dict):
            return {key: FlowPlanner._repeat_batch(item, repeats) for key, item in value.items()}
        return value

    def sample_from_encoded(self, model_inputs, encoder_outputs, x_init, use_cfg=True, cfg_weight=None):
        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        sample = self.flow_ode.generate(x_init, self.decoder, self._model_type, use_cfg=use_cfg, cfg_weight=cfg_weight, **decoder_model_extra)
        sample = assemble_actions(sample, self.planner_params['future_len'], self.planner_params['action_len'], self.planner_params['action_overlap'], self.planner_params['state_dim'], self.assemble_method)
        sample = self.data_processor.state_postprocess(sample)
        return sample

    def forward_inference_candidates(self, data, num_candidates, seeds=None, use_cfg=True, cfg_weight=None):
        """Sample N trajectories while keeping scene encodings frozen."""
        if data.ego_current.shape[0] != 1:
            raise ValueError("candidate inference expects one scene at a time")
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if seeds is not None and len(seeds) != num_candidates:
            raise ValueError("seeds must contain one seed per candidate")

        model_inputs, encoder_outputs = self.encode_scene(data, use_cfg=use_cfg)
        repeated_inputs = self._repeat_batch(model_inputs, num_candidates)
        repeated_outputs = self._repeat_batch(encoder_outputs, num_candidates)
        shape = (1, self.action_num, self.planner_params['action_len'], self.planner_params['state_dim'])
        if seeds is None:
            x_init = torch.randn((num_candidates, *shape[1:]), device=self.device)
        else:
            samples = []
            for seed in seeds:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(seed))
                samples.append(torch.randn(shape, device=self.device, generator=generator))
            x_init = torch.cat(samples, dim=0)

        sample = self.sample_from_encoded(
            model_inputs=repeated_inputs,
            encoder_outputs=repeated_outputs,
            x_init=x_init,
            use_cfg=use_cfg,
            cfg_weight=cfg_weight,
        )
        scene_tokens = torch.cat(encoder_outputs['encodings'], dim=1)[:1]
        scene_mask = torch.cat(encoder_outputs['masks'], dim=1)[:1]
        return sample, scene_tokens, scene_mask

    # ------------------------------------------------------------------ #
    # Critic pipeline hooks.
    # The critic stores the *inputs* to the scene encoder (not the frozen
    # 192-d tokens), so its own encoder can re-encode them and the planner
    # can re-inference fresh candidates from them at critic-training time.
    # ------------------------------------------------------------------ #
    def scene_encoder_inputs(self, data: NuPlanDataSample):
        """The clean (no-CFG) inputs that enter the scene encoder for `data`.

        Returns the dict `extract_encoder_inputs` produces -- the 6 normalized
        tensors (neighbors / static / lanes / lanes_speed_limit /
        lanes_has_speed_limit / routes) -- WITHOUT running the encoder. This is
        exactly what `self.encoder(**...)` consumes, so it is what the critic
        pipeline persists per transition (V1).
        """
        B = data.ego_current.shape[0]
        cfg_flags = torch.ones((B,), device=self.device, dtype=torch.int32)
        model_inputs, _ = self.prepare_model_input(cfg_flags, data, use_cfg=False, is_training=False)
        return self.extract_encoder_inputs(model_inputs)

    def sample_candidates_from_encoder_inputs(self, encoder_inputs, num_candidates):
        """Re-inference K candidate trajectories per scene from stored encoder inputs.

        Batched: `encoder_inputs` holds B scenes (each tensor has a leading batch
        dim B). Mirrors `forward_inference_candidates` but starts from the
        persisted (no-CFG) encoder inputs, and handles B > 1 by interleaving K
        candidate samples per scene. Returns [B, K, future_len, state_dim] (the
        trajectory, matching `compute_candidate_trajectories`' outputs[:, 0]).
        """
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        # Replay stores has_speed_limit as float; the lane encoder needs it as a
        # boolean index. Cast (harmless if already bool).
        encoder_inputs = dict(encoder_inputs)
        encoder_inputs["lanes_has_speed_limit"] = encoder_inputs["lanes_has_speed_limit"].bool()
        encoder_outputs = self.encoder(**encoder_inputs)  # batch B, no CFG
        B = next(iter(encoder_inputs.values())).shape[0]
        model_inputs = {"cfg_flags": torch.ones((B,), device=self.device, dtype=torch.int32)}
        # repeat_interleave along batch -> rows ordered [scene0 x K, scene1 x K, ...]
        repeated_inputs = self._repeat_batch(model_inputs, num_candidates)
        repeated_outputs = self._repeat_batch(encoder_outputs, num_candidates)
        x_init = torch.randn(
            (B * num_candidates, self.action_num, self.planner_params['action_len'],
             self.planner_params['state_dim']),
            device=self.device,
        )
        sample = self.sample_from_encoded(
            model_inputs=repeated_inputs,
            encoder_outputs=repeated_outputs,
            x_init=x_init,
            use_cfg=False,
            cfg_weight=None,
        )
        traj = sample[:, 0]  # [B*K, future_len, state_dim]
        return traj.reshape(B, num_candidates, *traj.shape[1:])

    @property
    def model_type(self,):
        return self._model_type
    
    def get_optimizer_params(self):
        return [
            {'params': self.model_encoder.parameters()},
            {'params': self.model_decoder.parameters()}
        ]