"""On-disk replay for the critic pipeline (zarr).

Stores, per transition:
  - scene   : the scene ENCODER INPUTS (dict of the 6 normalized tensors that
              enter the Flow-Planner encoder; see CandidateBatch.scene_inputs).
              NOT the frozen 192-d tokens -- so the critic can re-encode them and
              training can re-inference fresh candidates from them.
  - action  : the executed candidate trajectory [H, state_dim] (the a in Q(s,a)).
  - reward, reward_components, done, episode_id, step_index.

The next scene s' is NOT duplicated; it is the row with (episode_id, step_index+1)
and is resolved on read. This file just persists raw transitions -- n-step
returns / bootstrapping live in the (in-memory) online trainer buffer.

Dataset shapes are inferred from the first record, so the scene tensor shapes
don't need to be declared up front.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import zarr
from numcodecs import Blosc

from flow_planner.critic_rl.types import REWARD_COMPONENT_NAMES, Transition

SCHEMA_VERSION = 3  # v3: store scene encoder-inputs + action (was: frozen tokens + N candidates)


@dataclass(frozen=True)
class ReplaySpec:
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")


def _to_np(value: Any, dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def transition_to_record(transition: Transition) -> Dict[str, Any]:
    """Flatten a Transition into the stored record (crosses the Ray boundary)."""
    obs = transition.observation
    action_index = int(transition.action)
    action_traj = obs.batch.candidates[action_index]  # [H, state_dim] -- the executed action
    scene = {key: _to_np(val, np.float32) for key, val in obs.batch.scene_inputs.items()}
    components = transition.info.get("reward_components", {}) if transition.info else {}
    reward_components = np.asarray(
        [float(components.get(name, 0.0)) for name in REWARD_COMPONENT_NAMES], dtype=np.float32
    )
    return {
        "scene": scene,
        "action": _to_np(action_traj, np.float32),
        "reward": float(transition.reward),
        "reward_components": reward_components,
        "done": bool(transition.done),
        "episode_id": int(obs.episode_id),
        "step_index": int(obs.step_index),
    }


class ZarrReplayWriter:
    """Single-writer circular replay. Use one instance / one Ray actor."""

    def __init__(self, path: str, spec: ReplaySpec, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.spec = spec
        self.root = zarr.open_group(str(self.path), mode="w" if overwrite else "a")
        self._scene_keys: Optional[List[str]] = None
        if "schema_version" in self.root.attrs:
            self._validate_schema()
            self._scene_keys = list(self.root.attrs["scene_keys"])

    def _validate_schema(self) -> None:
        stored = int(self.root.attrs.get("schema_version", 0))
        if stored != SCHEMA_VERSION:
            raise ValueError(f"replay schema v{stored} != v{SCHEMA_VERSION}; re-collect this replay")

    def _initialize(self, record: Mapping[str, Any]) -> None:
        cap = self.spec.capacity
        comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        self._scene_keys = sorted(record["scene"].keys())

        def make(name: str, shape, dtype):
            chunk = (min(8, cap), *shape[1:])
            self.root.create_dataset(name, shape=shape, chunks=chunk, dtype=dtype, compressor=comp)

        for key in self._scene_keys:
            arr = np.asarray(record["scene"][key])
            make(f"scene/{key}", (cap, *arr.shape), "f4")
        action = np.asarray(record["action"])
        make("action", (cap, *action.shape), "f4")
        make("reward", (cap,), "f8")
        make("reward_components", (cap, len(REWARD_COMPONENT_NAMES)), "f4")
        make("done", (cap,), "b1")
        make("episode_id", (cap,), "i8")
        make("step_index", (cap,), "i8")
        make("sequence_id", (cap,), "i8")
        self.root.attrs.update(
            schema_version=SCHEMA_VERSION, scene_keys=self._scene_keys, head=0, size=0, next_sequence_id=0
        )

    def append_records(self, records: Sequence[Mapping[str, Any]]) -> int:
        if not records:
            return 0
        if self._scene_keys is None:
            self._initialize(records[0])
        cap = self.spec.capacity
        head = int(self.root.attrs["head"])
        size = int(self.root.attrs["size"])
        seq = int(self.root.attrs["next_sequence_id"])

        # column-ify the batch
        columns: Dict[str, np.ndarray] = {}
        for key in self._scene_keys:
            columns[f"scene/{key}"] = np.stack([np.asarray(r["scene"][key], np.float32) for r in records])
        columns["action"] = np.stack([np.asarray(r["action"], np.float32) for r in records])
        columns["reward"] = np.asarray([float(r["reward"]) for r in records], np.float64)
        columns["reward_components"] = np.stack([np.asarray(r["reward_components"], np.float32) for r in records])
        columns["done"] = np.asarray([bool(r["done"]) for r in records], np.bool_)
        columns["episode_id"] = np.asarray([int(r["episode_id"]) for r in records], np.int64)
        columns["step_index"] = np.asarray([int(r["step_index"]) for r in records], np.int64)
        columns["sequence_id"] = np.arange(seq, seq + len(records), dtype=np.int64)

        count = len(records)
        offset = 0
        while offset < count:
            span = min(count - offset, cap - head)
            for name, batch in columns.items():
                self.root[name][head : head + span] = batch[offset : offset + span]
            head = (head + span) % cap
            offset += span
        self.root.attrs.update(head=head, size=min(size + count, cap), next_sequence_id=seq + count)
        return count

    def append(self, transition: Transition) -> int:
        return self.append_records([transition_to_record(transition)])

    def stats(self) -> Dict[str, int]:
        return {
            "size": int(self.root.attrs.get("size", 0)),
            "capacity": self.spec.capacity,
            "head": int(self.root.attrs.get("head", 0)),
            "next_sequence_id": int(self.root.attrs.get("next_sequence_id", 0)),
        }


class ZarrReplayReader:
    """Reads raw transitions; resolves each transition's next scene by
    (episode_id, step_index + 1). Feed `iter_transitions()` into the trainer buffer."""

    def __init__(self, path: str) -> None:
        self.root = zarr.open_group(path, mode="r")
        stored = int(self.root.attrs.get("schema_version", 0))
        if stored != SCHEMA_VERSION:
            raise ValueError(f"replay schema v{stored} != v{SCHEMA_VERSION}; re-collect this replay")
        self.scene_keys = list(self.root.attrs["scene_keys"])
        self._size = int(self.root.attrs["size"])
        self._episode = np.asarray(self.root["episode_id"][: self._size])
        self._step = np.asarray(self.root["step_index"][: self._size])
        self._seq = np.asarray(self.root["sequence_id"][: self._size])
        self._done = np.asarray(self.root["done"][: self._size])
        # (episode, step) -> newest physical row (ring buffer may hold duplicates)
        self._index: Dict[tuple, int] = {}
        for i in range(self._size):
            key = (int(self._episode[i]), int(self._step[i]))
            j = self._index.get(key)
            if j is None or int(self._seq[i]) > int(self._seq[j]):
                self._index[key] = i

    def __len__(self) -> int:
        return self._size

    def _scene_at(self, i: int) -> Dict[str, np.ndarray]:
        return {k: np.asarray(self.root[f"scene/{k}"][i]) for k in self.scene_keys}

    def iter_transitions(self):
        """Yield dicts: scene, action, reward, done, next_scene (None if terminal
        or if the next row was overwritten in the ring buffer)."""
        for i in range(self._size):
            done = bool(self._done[i])
            next_scene = None
            if not done:
                nxt = self._index.get((int(self._episode[i]), int(self._step[i]) + 1))
                if nxt is None:
                    continue  # next got overwritten in the ring buffer -> can't bootstrap, drop
                next_scene = self._scene_at(nxt)
            yield {
                "scene": self._scene_at(i),
                "action": np.asarray(self.root["action"][i]),
                "reward": float(self.root["reward"][i]),
                "done": done,
                "next_scene": next_scene,
            }


def create_replay_writer_actor(path: str, spec: ReplaySpec, overwrite: bool = False):
    """The only process allowed to mutate a replay."""
    import ray

    writer_cls = ray.remote(num_cpus=1)(ZarrReplayWriter)
    return writer_cls.remote(path, spec, overwrite)
