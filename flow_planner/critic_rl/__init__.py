"""Critic pipeline for Flow-Planner: data collection and closed-loop evaluation.

Two things live here, and only two entry points matter:

    collect.py     -- roll out the planner and save (state, candidates, reward)
                      transitions to a replay buffer (the critic's training data).
    evaluate.py    -- run closed-loop val14 and score candidates with a chosen
                      scorer (candidate0 / random / a trained critic).

The critic model architecture (critic.py) and its training algorithm
(trainer.py) are deliberately left as stubs -- fill them in later.
"""

from flow_planner.critic_rl.types import CandidateBatch, CriticObservation, Transition

__all__ = ["CandidateBatch", "CriticObservation", "Transition"]
