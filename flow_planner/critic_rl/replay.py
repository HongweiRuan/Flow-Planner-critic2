from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from numcodecs import Blosc

from flow_planner.critic_rl.types import REWARD_COMPONENT_NAMES, Transition

# Bump when the on-disk layout changes so stale replays fail loudly instead of
# being silently misread. v2 dropped the next_candidates column; the best-of-N
# target reads the next state's candidates from the next logical row instead.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ReplaySpec:
    capacity: int
    num_candidates: int
    horizon: int
    state_dim: int
    scene_tokens: int
    context_dim: int

    def __post_init__(self) -> None:
        if min(
            self.capacity,
            self.num_candidates,
            self.horizon,
            self.state_dim,
            self.scene_tokens,
            self.context_dim,
        ) < 1:
            raise ValueError("all replay dimensions must be positive")



def transition_to_record(transition: Transition) -> Dict[str, Any]:
    """Strip controller trajectories before crossing the Ray actor boundary."""
    obs = transition.observation
    nxt = transition.next_observation

    def array(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(dtype, copy=False)

    candidates = array(obs.batch.candidates, np.float32)
    scene_tokens = array(obs.batch.scene_tokens, np.float32)
    scene_mask = array(obs.batch.scene_mask, np.bool_)
    # Only the next scene is stored: the critic bootstraps V^H(next_scene), never
    # the next candidates. A terminal transition has no next and does not bootstrap.
    if nxt is None:
        next_scene_tokens = np.zeros_like(scene_tokens)
        next_scene_mask = np.zeros_like(scene_mask)
    else:
        next_scene_tokens = array(nxt.batch.scene_tokens, np.float32)
        next_scene_mask = array(nxt.batch.scene_mask, np.bool_)
    # Persist the per-component reward breakdown so weights can be re-tuned
    # offline; missing components (e.g. legacy/fake transitions) default to zero.
    components = transition.info.get("reward_components", {}) if transition.info else {}
    reward_components = np.asarray(
        [float(components.get(name, 0.0)) for name in REWARD_COMPONENT_NAMES],
        dtype=np.float32,
    )
    return {
        "candidates": candidates,
        "scene_tokens": scene_tokens,
        "scene_mask": scene_mask,
        "next_scene_tokens": next_scene_tokens,
        "next_scene_mask": next_scene_mask,
        "action": int(transition.action),
        "reward": float(transition.reward),
        "reward_components": reward_components,
        "done": bool(transition.done),
        "episode_id": int(obs.episode_id),
        "step_index": int(obs.step_index),
    }

class ZarrReplayWriter:
    """Single-writer circular replay. Use one instance or one Ray actor."""

    def __init__(self, path: str, spec: ReplaySpec, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.spec = spec
        mode = "w" if overwrite else "a"
        self.root = zarr.open_group(str(self.path), mode=mode)
        if "schema_version" not in self.root.attrs:
            self._initialize()
        self._validate_schema()

    def _initialize(self) -> None:
        s = self.spec
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        arrays = {
            "candidates": ((s.capacity, s.num_candidates, s.horizon, s.state_dim), "f4"),
            "scene_tokens": ((s.capacity, s.scene_tokens, s.context_dim), "f4"),
            "scene_mask": ((s.capacity, s.scene_tokens), "b1"),
            "next_scene_tokens": ((s.capacity, s.scene_tokens, s.context_dim), "f4"),
            "next_scene_mask": ((s.capacity, s.scene_tokens), "b1"),
            "action": ((s.capacity,), "i4"),
            "reward": ((s.capacity,), "f8"),
            "reward_components": ((s.capacity, len(REWARD_COMPONENT_NAMES)), "f4"),
            "done": ((s.capacity,), "b1"),
            "episode_id": ((s.capacity,), "i8"),
            "step_index": ((s.capacity,), "i8"),
            "sequence_id": ((s.capacity,), "i8"),
        }
        for name, (shape, dtype) in arrays.items():
            # Small axis-0 chunks: appending one row rewrites only its chunk,
            # avoiding ~64x write amplification when the replay lives on CephFS.
            chunk = (min(8, s.capacity), *shape[1:])
            self.root.create_dataset(name, shape=shape, chunks=chunk, dtype=dtype, compressor=compressor)
        self.root.attrs.update(
            schema_version=SCHEMA_VERSION,
            spec=s.__dict__,
            head=0,
            size=0,
            next_sequence_id=0,
        )

    def _validate_schema(self) -> None:
        stored_version = int(self.root.attrs.get("schema_version", 0))
        if stored_version != SCHEMA_VERSION:
            raise ValueError(
                f"replay schema v{stored_version} != v{SCHEMA_VERSION}; re-collect this replay"
            )
        stored = dict(self.root.attrs["spec"])
        if stored != self.spec.__dict__:
            raise ValueError(f"replay spec mismatch: stored={stored}, requested={self.spec.__dict__}")

    def append(self, transition: Transition) -> int:
        return self.append_record(transition_to_record(transition))

    def append_record(self, record: Mapping[str, Any]) -> int:
        index = int(self.root.attrs["head"])
        self.append_records((record,))
        return index

    def _coerce_record(self, record: Mapping[str, Any], sequence_id: int) -> Dict[str, Any]:
        values = dict(record)
        candidates = np.asarray(values["candidates"], dtype=np.float32)
        scene_tokens = np.asarray(values["scene_tokens"], dtype=np.float32)
        scene_mask = np.asarray(values["scene_mask"], dtype=np.bool_)
        next_scene_tokens = np.asarray(values["next_scene_tokens"], dtype=np.float32)
        next_scene_mask = np.asarray(values["next_scene_mask"], dtype=np.bool_)
        reward_components = np.asarray(values["reward_components"], dtype=np.float32)
        if reward_components.shape != (len(REWARD_COMPONENT_NAMES),):
            raise ValueError(
                f"reward_components shape {reward_components.shape} does not match "
                f"{(len(REWARD_COMPONENT_NAMES),)}"
            )

        current_expected = (
            (self.spec.num_candidates, self.spec.horizon, self.spec.state_dim),
            (self.spec.scene_tokens, self.spec.context_dim),
            (self.spec.scene_tokens,),
        )
        next_expected = ((self.spec.scene_tokens, self.spec.context_dim), (self.spec.scene_tokens,))
        current_shapes = (candidates.shape, scene_tokens.shape, scene_mask.shape)
        next_shapes = (next_scene_tokens.shape, next_scene_mask.shape)
        if current_shapes != current_expected or next_shapes != next_expected:
            raise ValueError(
                f"transition shapes current={current_shapes}, next={next_shapes} do not match replay"
            )

        values.update(
            candidates=candidates,
            scene_tokens=scene_tokens,
            scene_mask=scene_mask,
            next_scene_tokens=next_scene_tokens,
            next_scene_mask=next_scene_mask,
            reward_components=reward_components,
            action=int(values["action"]),
            reward=float(values["reward"]),
            done=bool(values["done"]),
            episode_id=int(values["episode_id"]),
            step_index=int(values["step_index"]),
            sequence_id=int(sequence_id),
        )
        return values

    def append_records(self, records: Sequence[Mapping[str, Any]]) -> int:
        """Append a batch using axis-0 slices aligned with replay chunks."""
        if not records:
            return 0
        head = int(self.root.attrs["head"])
        size = int(self.root.attrs["size"])
        next_sequence_id = int(self.root.attrs["next_sequence_id"])
        coerced = [
            self._coerce_record(record, next_sequence_id + offset)
            for offset, record in enumerate(records)
        ]
        values = {
            name: np.asarray([record[name] for record in coerced])
            for name in coerced[0]
        }
        count = len(coerced)
        offset = 0
        while offset < count:
            span = min(count - offset, self.spec.capacity - head)
            for name, batch in values.items():
                self.root[name][head : head + span] = batch[offset : offset + span]
            head = (head + span) % self.spec.capacity
            offset += span
        self.root.attrs.update(
            head=head,
            size=min(size + count, self.spec.capacity),
            next_sequence_id=next_sequence_id + count,
        )
        return count

    def append_many(self, transitions: Iterable[Transition]) -> int:
        records = [transition_to_record(transition) for transition in transitions]
        return self.append_records(records)

    def stats(self) -> Dict[str, int]:
        return {
            "size": int(self.root.attrs["size"]),
            "capacity": self.spec.capacity,
            "head": int(self.root.attrs["head"]),
            "next_sequence_id": int(self.root.attrs["next_sequence_id"]),
        }


class ZarrReplayReader:
    def __init__(self, path: str) -> None:
        self.root = zarr.open_group(path, mode="r")
        stored_version = int(self.root.attrs.get("schema_version", 0))
        if stored_version != SCHEMA_VERSION:
            raise ValueError(
                f"replay schema v{stored_version} != v{SCHEMA_VERSION}; re-collect this replay"
            )
        self.spec = ReplaySpec(**dict(self.root.attrs["spec"]))
        self._active_size = min(int(self.root.attrs["size"]), self.spec.capacity)
        self._metadata: Dict[str, np.ndarray] = {}
        self._index_by_episode_step: Optional[Dict[Tuple[int, int], int]] = None
        self._cache: Optional[Dict[str, np.ndarray]] = None
        self._cache_default_only = False
        self._cache_deduplicate_next = False

    def __len__(self) -> int:
        return self._active_size

    def _active_indices(self) -> np.ndarray:
        size = len(self)
        if size < self.spec.capacity:
            return np.arange(size, dtype=np.int64)
        return np.arange(self.spec.capacity, dtype=np.int64)

    def _read_active(self, name: str) -> np.ndarray:
        return np.asarray(self.root[name][: self._active_size])

    def _read_block(self, name: str, start: int, stop: int, default_only: bool, scene_float16: bool) -> Tuple[str, int, np.ndarray]:
        values = np.asarray(self.root[name][start:stop])
        if default_only and name == "candidates":
            values = values[:, :1]
        if scene_float16 and name in ("scene_tokens", "next_scene_tokens"):
            values = values.astype(np.float16)
        return name, start, values

    def _ensure_basic_metadata(self) -> None:
        for name in ("reward", "done"):
            if name not in self._metadata:
                self._metadata[name] = self._read_active(name)

    def _ensure_sequence_metadata(self) -> None:
        self._ensure_basic_metadata()
        for name in ("episode_id", "step_index", "sequence_id"):
            if name not in self._metadata:
                self._metadata[name] = self._read_active(name)
        if self._index_by_episode_step is None:
            self._index_by_episode_step = self._build_index_by_episode_step()

    def _build_index_by_episode_step(self) -> Dict[Tuple[int, int], int]:
        """Map logical rollout coordinates to physical zarr rows.

        Multi-worker collection interleaves physical writes, so n-step targets
        must follow (episode_id, step_index) instead of adjacent row numbers.
        If a circular replay contains duplicate logical keys, keep the newest
        row according to sequence_id.
        """
        mapping: Dict[Tuple[int, int], int] = {}
        for raw_index in self._active_indices():
            index = int(raw_index)
            key = (
                int(self._metadata["episode_id"][index]),
                int(self._metadata["step_index"][index]),
            )
            previous = mapping.get(key)
            if previous is None or int(self._metadata["sequence_id"][index]) > int(self._metadata["sequence_id"][previous]):
                mapping[key] = index
        return mapping

    def cache_in_memory(self, max_bytes: int, default_only: bool = False, deduplicate_next: bool = False, scene_float16: bool = False) -> bool:
        """Cache active replay arrays concurrently when they fit in host RAM."""
        active = self._active_indices()
        names = [
            "candidates",
            "scene_tokens",
            "scene_mask",
        ]
        if not deduplicate_next:
            names.extend((
                "next_scene_tokens",
                "next_scene_mask",
            ))
        names.extend((
            "action",
            "reward",
            "done",
            "episode_id",
            "step_index",
            "sequence_id",
        ))
        shapes = {name: (len(active), *self.root[name].shape[1:]) for name in names}
        dtypes = {name: self.root[name].dtype for name in names}
        if scene_float16:
            for name in ("scene_tokens", "next_scene_tokens"):
                if name in dtypes:
                    dtypes[name] = np.dtype(np.float16)
        if default_only:
            if "candidates" in shapes:
                shapes["candidates"] = (len(active), 1, *self.root["candidates"].shape[2:])
        required = sum(int(np.prod(shapes[name])) * dtypes[name].itemsize for name in names)
        if required > int(max_bytes):
            return False
        self._cache = {
            name: np.empty(shapes[name], dtype=dtypes[name])
            for name in names
        }
        block_rows = 256
        with ThreadPoolExecutor(max_workers=24) as pool:
            futures = [
                pool.submit(self._read_block, name, start, min(start + block_rows, len(active)), default_only, scene_float16)
                for name in names
                for start in range(0, len(active), block_rows)
            ]
            total = len(futures)
            report_every = max(1, total // 10)
            for completed, future in enumerate(as_completed(futures), start=1):
                name, start, values = future.result()
                stop = start + len(values)
                self._cache[name][start:stop] = values
                if completed % report_every == 0 or completed == total:
                    print(f"replay_cache {completed}/{total} blocks ({100.0 * completed / total:.0f}%)", flush=True)
        for name in ("episode_id", "step_index", "sequence_id", "reward", "done"):
            self._metadata[name] = self._cache[name]
        if default_only and np.any(self._cache["action"] != 0):
            raise ValueError("default-only replay cache requires every behavior action to be candidate0")
        self._cache_default_only = bool(default_only)
        self._cache_deduplicate_next = bool(deduplicate_next)
        if deduplicate_next:
            self._ensure_sequence_metadata()
        return True

    def _rollout(self, start: int, n_step: int, gamma: float) -> Optional[Tuple[float, float, int]]:
        self._ensure_basic_metadata()
        current = int(start)
        if n_step == 1:
            value = float(self._metadata["reward"][current])
            bootstrap = 0.0 if bool(self._metadata["done"][current]) else float(gamma)
            return value, bootstrap, current
        self._ensure_sequence_metadata()
        assert self._index_by_episode_step is not None
        total = 0.0
        discount = 1.0
        episode = int(self._metadata["episode_id"][current])
        start_step = int(self._metadata["step_index"][current])
        for offset in range(n_step):
            expected_step = start_step + offset
            if int(self._metadata["episode_id"][current]) != episode:
                return None
            if int(self._metadata["step_index"][current]) != expected_step:
                return None
            total += discount * float(self._metadata["reward"][current])
            if bool(self._metadata["done"][current]):
                return total, 0.0, current
            discount *= gamma
            if offset + 1 < n_step:
                next_key = (episode, expected_step + 1)
                next_index = self._index_by_episode_step.get(next_key)
                if next_index is None:
                    return None
                current = int(next_index)
        return total, discount, current

    def _take(self, name: str, indices: np.ndarray) -> np.ndarray:
        if self._cache is not None:
            return self._cache[name][indices]
        return np.asarray(self.root[name].oindex[indices])

    def sample(
        self,
        batch_size: int,
        n_step: int = 10,
        gamma: float = 1.0,
        seed: Optional[int] = None,
        include_next_candidates: bool = False,
    ) -> Dict[str, np.ndarray]:
        if batch_size < 1 or n_step < 1:
            raise ValueError("batch_size and n_step must be positive")
        if len(self) < 1:
            raise ValueError("cannot sample an empty replay")
        rng = np.random.default_rng(seed)
        active = self._active_indices()
        if n_step == 1:
            # A one-step target never rejects (every row is valid and bootstraps
            # from itself), so the whole batch is drawn and scored with vectorized
            # numpy instead of a per-sample Python loop.
            self._ensure_basic_metadata()
            starts_array = rng.choice(active, size=batch_size)
            finals_array = starts_array
            returns = self._metadata["reward"][starts_array].astype(np.float32)
            alive = ~np.asarray(self._metadata["done"][starts_array], dtype=bool)
            bootstraps_array = np.where(alive, np.float32(gamma), np.float32(0.0)).astype(np.float32)
        else:
            starts = []
            returns_list = []
            bootstraps = []
            finals = []
            max_attempts = max(100, batch_size * 50)
            for _ in range(max_attempts):
                start = int(rng.choice(active))
                rollout = self._rollout(start, n_step=n_step, gamma=gamma)
                if rollout is None:
                    continue
                value, bootstrap, final = rollout
                starts.append(start)
                returns_list.append(value)
                bootstraps.append(bootstrap)
                finals.append(final)
                if len(starts) == batch_size:
                    break
            if len(starts) != batch_size:
                raise RuntimeError("not enough contiguous transitions for n-step sampling")
            starts_array = np.asarray(starts)
            finals_array = np.asarray(finals)
            returns = np.asarray(returns_list, dtype=np.float32)
            bootstraps_array = np.asarray(bootstraps, dtype=np.float32)
        # Resolve the logical next row (episode, step+1) when it is needed: for the
        # EMAQ best-of-N target (the next candidates, which are not stored and are
        # read from the next row) or for the deduplicated next scene. Terminal rows
        # keep a dummy next (themselves) and are zeroed by the bootstrap mask.
        need_next_index = include_next_candidates or self._cache_deduplicate_next
        if need_next_index:
            self._ensure_sequence_metadata()
            assert self._index_by_episode_step is not None
            next_indices = finals_array.copy()
            for position in np.flatnonzero(bootstraps_array > 0):
                final = int(finals_array[position])
                key = (int(self._metadata["episode_id"][final]), int(self._metadata["step_index"][final]) + 1)
                next_index = self._index_by_episode_step.get(key)
                if next_index is None:
                    raise RuntimeError(f"missing logical next transition for {key}")
                next_indices[position] = next_index

        if self._cache_deduplicate_next:
            next_scene_tokens = self._take("scene_tokens", next_indices)
            next_scene_mask = self._take("scene_mask", next_indices)
        else:
            next_scene_tokens = self._take("next_scene_tokens", finals_array)
            next_scene_mask = self._take("next_scene_mask", finals_array)

        batch = {
            "candidates": self._take("candidates", starts_array),
            "scene_tokens": self._take("scene_tokens", starts_array),
            "scene_mask": self._take("scene_mask", starts_array),
            "actions": self._take("action", starts_array),
            "returns": np.asarray(returns, dtype=np.float32),
            "bootstrap": bootstraps_array,
            "next_scene_tokens": next_scene_tokens,
            "next_scene_mask": next_scene_mask,
        }
        if include_next_candidates:
            batch["next_candidates"] = self._take("candidates", next_indices)
        return batch


def create_replay_writer_actor(path: str, spec: ReplaySpec, overwrite: bool = False):
    """Create the only process allowed to mutate a replay."""
    import ray

    writer_cls = ray.remote(num_cpus=1)(ZarrReplayWriter)
    return writer_cls.remote(path, spec, overwrite)
