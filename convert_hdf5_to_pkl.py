"""Convert HDF5 expert/supplement files to the pkl trajectory format expected by ExpertDataset.

Usage:
    python convert_hdf5_to_pkl.py

Converts all HDF5 files found in experts/ and supplement/ to matching .pkl files.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def load_hdf5_dataset(hdf5_path: str) -> dict:
    """Load flat arrays from an HDF5 file, skipping group entries."""
    data = {}
    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            item = f[key]
            if hasattr(item, "shape") and item.shape != ():
                data[key] = item[:]
    return data


def split_into_trajectories(dataset: dict, timeout_as_done: bool = False) -> dict:
    """Split flat transition arrays into per-trajectory lists."""
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"].squeeze()
    terminals = dataset["terminals"].astype(bool).squeeze()

    timeouts = dataset.get("timeouts")
    if timeouts is None:
        timeouts = np.zeros_like(terminals, dtype=bool)
    else:
        timeouts = timeouts.astype(bool).squeeze()

    next_observations = dataset.get("next_observations")
    if next_observations is None:
        next_observations = np.concatenate(
            [observations[1:], observations[-1:]], axis=0
        )

    trajs = defaultdict(list)
    current = {
        "states": [],
        "next_states": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    for idx in range(len(observations)):
        done = bool(terminals[idx] or (timeout_as_done and timeouts[idx]))
        end_of_trajectory = bool(terminals[idx] or timeouts[idx])

        current["states"].append(observations[idx])
        current["next_states"].append(next_observations[idx])
        current["actions"].append(actions[idx])
        current["rewards"].append(rewards[idx])
        current["dones"].append(float(done))

        if end_of_trajectory:
            trajs["states"].append(np.asarray(current["states"], dtype=np.float32))
            trajs["next_states"].append(np.asarray(current["next_states"], dtype=np.float32))
            trajs["actions"].append(np.asarray(current["actions"], dtype=np.float32))
            trajs["rewards"].append(np.asarray(current["rewards"], dtype=np.float32))
            trajs["dones"].append(np.asarray(current["dones"], dtype=np.float32))
            trajs["lengths"].append(len(current["states"]))
            for key in current:
                current[key].clear()

    # Handle trailing transitions without a terminal/timeout
    if current["states"]:
        trajs["states"].append(np.asarray(current["states"], dtype=np.float32))
        trajs["next_states"].append(np.asarray(current["next_states"], dtype=np.float32))
        trajs["actions"].append(np.asarray(current["actions"], dtype=np.float32))
        trajs["rewards"].append(np.asarray(current["rewards"], dtype=np.float32))
        trajs["dones"].append(np.asarray(current["dones"], dtype=np.float32))
        trajs["lengths"].append(len(current["states"]))

    return trajs


def convert_one(hdf5_path: Path, output_path: Path) -> None:
    """Convert a single HDF5 file to pkl."""
    print(f"Loading {hdf5_path} ...")
    dataset = load_hdf5_dataset(str(hdf5_path))
    trajs = split_into_trajectories(dataset)

    num_episodes = len(trajs["lengths"])
    total_steps = sum(trajs["lengths"])
    avg_len = np.mean(trajs["lengths"]) if trajs["lengths"] else 0.0
    avg_reward = np.mean([r.sum() for r in trajs["rewards"]]) if trajs["rewards"] else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dict(trajs), f)

    print(f"  -> {output_path}")
    print(f"     episodes={num_episodes}, total_steps={total_steps}, "
          f"avg_len={avg_len:.1f}, avg_episode_reward={avg_reward:.1f}")


def main():
    repo_root = Path(__file__).resolve().parent
    hdf5_files = list(repo_root.glob("experts/*.hdf5")) + list(repo_root.glob("supplement/*.hdf5"))

    if not hdf5_files:
        print("No HDF5 files found in experts/ or supplement/.")
        return

    print(f"Found {len(hdf5_files)} HDF5 file(s) to convert.\n")

    for hdf5_path in sorted(hdf5_files):
        # Output pkl has the same name but with .pkl extension
        output_path = hdf5_path.with_suffix(".pkl")
        convert_one(hdf5_path, output_path)
        print()

    print("Done. All HDF5 files converted to pkl.")


if __name__ == "__main__":
    main()
