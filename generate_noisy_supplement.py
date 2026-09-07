"""
Generate noisy supplement dataset from expert demonstrations.

Takes 20 expert trajectories, replaces actions in the first 10 with
uniform random actions, keeps the remaining 10 unchanged, and saves
the combined dataset to the supplement/ directory in the same pickle
format used by ExpertDataset.

Usage:
    python scripts/generate_noisy_supplement.py --env_demo hopper/hopper_expert_v2.pkl --seed 123
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.expert_dataset import load_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_demo", type=str, required=True,
                        help="Relative path under experts/, e.g. hopper/hopper_expert_v2.pkl")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_trajs", type=int, default=20,
                        help="Total trajectories to load (half will get random actions)")
    args = parser.parse_args()

    expert_path = os.path.join("experts", args.env_demo)
    if not os.path.isfile(expert_path):
        raise FileNotFoundError(f"Expert file not found: {expert_path}")

    trajs = load_trajectories(expert_path, num_trajectories=args.num_trajs, seed=args.seed)

    num_available = len(trajs["lengths"])
    split = num_available // 2
    print(f"Loaded {num_available} trajectories, splitting {split} noisy + {num_available - split} expert")

    rng = np.random.RandomState(args.seed)

    # Replace actions in the first half with uniform random in [-1, 1]
    for i in range(split):
        original_actions = np.asarray(trajs["actions"][i])
        random_actions = rng.uniform(-1.0, 1.0, size=original_actions.shape).astype(original_actions.dtype)
        trajs["actions"][i] = random_actions

    # Save to supplement/ directory
    os.makedirs("supplement", exist_ok=True)

    env_name = os.path.splitext(os.path.basename(args.env_demo))[0]
    output_path = os.path.join("supplement", f"{env_name}_noisy_expert.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(trajs, f)

    total_transitions = sum(int(l) for l in trajs["lengths"])
    print(f"Saved noisy supplement to: {output_path}")
    print(f"  - Random action trajs: {split} (first half)")
    print(f"  - Expert action trajs: {num_available - split} (second half)")
    print(f"  - Total transitions: {total_transitions}")


if __name__ == "__main__":
    main()
