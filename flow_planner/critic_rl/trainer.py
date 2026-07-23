"""Critic training algorithm -- STUB.

Training reads a replay buffer produced by `collect.py` (a zarr store; see
replay.py: `ZarrReplayReader(path).sample(batch_size, n_step, gamma)` yields
batches of candidates / scene tokens / n-step returns / bootstrap masks) and
fits a `HorizonCritic` (critic.py). It then writes a checkpoint that
`evaluate.py --scorer critic --checkpoint ...` can load.

Nothing here is implemented yet. A minimal implementation would:

    1. reader = ZarrReplayReader(replay_path)
    2. critic = HorizonCritic(**critic_kwargs).to(device)
    3. for each update: batch = reader.sample(...); compute a Q-learning loss
       (target = n-step return + bootstrap * V(next); prediction = critic(...));
       step the optimizer.
    4. torch.save({"critic": critic.state_dict()}, checkpoint_path)

The checkpoint must store the critic weights under the "critic" key, because
that is what `EvaluationWorker` in workers.py loads.
"""


def train(replay_path: str, critic_kwargs: dict, checkpoint_path: str, **kwargs):
    """Train a critic from a replay buffer and save a checkpoint. NOT IMPLEMENTED."""
    raise NotImplementedError("critic training is a stub -- implement trainer.py")
