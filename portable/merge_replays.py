"""Merge several shard replays (schema v3) into one, by streaming concatenation.

Usage:
    python merge_replays.py OUT.zarr SHARD0.zarr SHARD1.zarr ...

Each shard was collected with a disjoint contiguous episode-id range (collect.py
--shard-index/--shard-count), so (episode_id, step_index) keys never collide across
shards and the reader's next-scene resolution stays correct. sequence_id is reassigned
globally so "newest wins" duplicate handling is well-defined. Prints a self-check on
duplicate (episode,step) keys -- MUST be 0.
"""
import sys
import numpy as np
import zarr
from numcodecs import Blosc


def main() -> None:
    out_path = sys.argv[1]
    shard_paths = sys.argv[2:]
    assert shard_paths, "need at least one shard"
    shards = [zarr.open_group(p, mode="r") for p in shard_paths]
    sizes = [int(s.attrs["size"]) for s in shards]
    total = int(sum(sizes))
    scene_keys = list(shards[0].attrs["scene_keys"])
    schema = int(shards[0].attrs["schema_version"])
    top = ["action", "reward", "reward_components", "done", "episode_id", "step_index", "sequence_id"]
    names = [f"scene/{k}" for k in scene_keys] + top

    out = zarr.open_group(out_path, mode="w")
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    for name in names:
        d0 = shards[0][name]
        shape = (total,) + tuple(d0.shape[1:])
        chunk = (min(8, total),) + tuple(d0.shape[1:])
        out.create_dataset(name, shape=shape, chunks=chunk, dtype=d0.dtype, compressor=comp)

    off = 0
    for s, n in zip(shards, sizes):
        for name in names:
            out[name][off:off + n] = s[name][:n]
        off += n
    # globally unique, monotonic sequence ids
    out["sequence_id"][:] = np.arange(total, dtype=np.int64)
    out.attrs.update(schema_version=schema, scene_keys=scene_keys, head=total, size=total, next_sequence_id=total)

    # self-check: (episode, step) keys must be unique across the merged set
    ep = np.asarray(out["episode_id"][:]).astype(np.int64)
    st = np.asarray(out["step_index"][:]).astype(np.int64)
    keys = ep * 1_000_000 + st
    dup = total - len(np.unique(keys))
    print(f"MERGE_DONE total={total} sizes={sizes} dup_(ep,step)_keys={dup} -> {out_path}", flush=True)
    if dup != 0:
        raise SystemExit(f"FATAL: {dup} colliding (episode,step) keys -- shards were NOT disjoint")


if __name__ == "__main__":
    main()
